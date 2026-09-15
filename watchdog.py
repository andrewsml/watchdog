"""libtam-based video surveillance service with a Telegram user interface.

Use --setup to authenticate the Telegram account and add initial video/audio
sources. Normal mode creates or reuses a private forum group, exposes one topic
per device, and handles calls, snapshots, RTMP streams, motion events, and
scheduled recordings.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Optional, Union
from urllib.parse import urlsplit, urlunsplit

from motion_detector import (
    MotionDetectorConfig,
    MotionFrameStats,
    MotionVectorDetector,
    PixelMotionConfig,
    codec_supports_motion_vectors,
    export_motion_vector_options,
    frame_bgr_sample,
    frame_motion_sample,
    iter_motion_events,
    verify_gray_motion,
)
from object_detector import (
    YOLO_MODEL_FILENAME,
    YOLO_PRESENCE_SAMPLE_COUNT,
    YOLO_PRESENCE_SAMPLE_INTERVAL_SECONDS,
    YoloObjectDetector,
    YoloPresenceTracker,
    detect_presence_samples,
)
from tam import Tam


av = None
DEFAULT_LANGUAGE = "en"
DEFAULT_CAMERA_FPS = "30"
DEFAULT_RECORD_FPS = 30
DEFAULT_LIVE_VIDEO_FPS = 24
DEFAULT_RTMP_VIDEO_FPS = 15
DEFAULT_RTMP_VIDEO_SIZE = "480p"
DEFAULT_RTMP_GOP = 20
AUDIO_RATE = 48000
AUDIO_CHANNELS = 2
AUDIO_CHUNK_SAMPLES = AUDIO_RATE // 100
AUDIO_CHUNK_BYTES = AUDIO_CHUNK_SAMPLES * AUDIO_CHANNELS * 2
CALL_DRAIN_GRACE_SECONDS = 0.5
RECORDING_QUEUE_MAX_ITEMS = 600
RECORDING_START_GAP_TRIM_US = 250_000
SCHEDULE_RECORDING_MAX_BYTES = 1_000_000_000
MOTION_GOP_MAX_BYTES = 64 * 1024 * 1024
MOTION_PENDING_MAX_BYTES = 96 * 1024 * 1024
YOLO_PRESENCE_INTERVAL_SECONDS = 5.0
YOLO_PRESENCE_MISSES_TO_STOP = 2
YOLO_SAMPLE_MAX_WIDTH = 640
DEFAULT_VIDEO_SIZE = "480p"
VIDEO_SIZE_MODES = {
    "360p": (360, 640),
    "480p": (480, 854),
    "720p": (720, 1280),
}
WATCHDOG_CHAT_STATE_KEY = "watchdog_chat"
WATCHDOG_LOGO_VERSION = 1
GENERAL_TOPIC_ID = 1
TECHNICAL_MESSAGE_PREFIX = "🔹 "
DEFAULT_FORUM_TOPIC_ICON_COLOR = 0x6FB9F0
FORUM_ENABLE_RETRY_DELAY_SECONDS = 1.0
FORUM_ENABLE_MAX_ATTEMPTS = 5
_WINDOWS_SCREEN_DEVICE_CACHE: Optional[list["CaptureDevice"]] = None
DEFAULT_TRANSLATION = {
    "group_title": "My Watchdog",
    "help_title": "My Watchdog commands",
    "help_general_heading": "General:",
    "help_device_topics_heading": "Device topics:",
    "help_text": (
        "My Watchdog commands\n"
        "\n"
        "General:\n"
        "Help - show this message\n"
        "Devices - list configured devices\n"
        "Add webcam - add a webcam\n"
        "Add screen - add a screen capture\n"
        "Add rtsp rtsp://login:password@ip_address/... - add an RTSP camera\n"
        "Cancel - cancel current add-device wizard\n"
        "\n"
        "Device topics:\n"
        "Call or C - call you and stream this device\n"
        "Screen or S - send a snapshot\n"
        "Stream On or S1 - start RTMP stream for this device\n"
        "Stream Off or S0 - stop RTMP stream\n"
        "Rec or r help - get commands to schedule video recording on the device; "
        "send the command to the device topic\n"
        "\n"
        "To extend the free license for another 2 months or get a permanent license, "
        "use @libtam_lic_bot."
    ),
    "initial_trial_activated": (
        "A free 2-month trial license has been activated. You can extend it for another 2 months for free "
        "or purchase a permanent license through @libtam_lic_bot."
    ),
    "record_schedule_hint": "To schedule video recording for this device, send: rec help",
    "record_schedule_title": "Recording schedule",
    "record_schedule_syntax_heading": "Syntax:",
    "record_schedule_days_heading": "Days:",
    "record_schedule_ranges_heading": "Ranges:",
    "record_schedule_multiple_ranges_heading": "Multiple ranges:",
    "record_schedule_examples_heading": "Examples:",
    "record_schedule_priority_heading": "Priority:",
    "record_schedule_commands_heading": "Commands:",
    "record_schedule_help_text": (
        "Recording schedule\n"
        "\n"
        "Syntax:\n"
        "rec or r <days>:<ranges>; <days>:<ranges>\n"
        "rec motion <days>:<ranges>; <days>:<ranges>\n"
        "rm <days>:<ranges>; <days>:<ranges>\n"
        "\n"
        "Days:\n"
        "* all days\n"
        "mon tue wed thu fri sat sun\n"
        "mon-fri range\n"
        "weekdays = mon-fri\n"
        "weekend = sat-sun\n"
        "\n"
        "Ranges:\n"
        "08-18 from 08:00 to 18:00\n"
        "22-07 overnight\n"
        "00-24 all day\n"
        "off disabled\n"
        "\n"
        "Multiple ranges:\n"
        "rec *:08-12,14-18\n"
        "\n"
        "Examples:\n"
        "rec *:22-07 - every night\n"
        "rec weekdays:09-18; weekend:off\n"
        "rec *:22-07; sat:00-24; sun:off\n"
        "rec motion *:22-07 - record only while motion is active\n"
        "\n"
        "Priority:\n"
        "Specific days override *.\n"
        "\n"
        "Commands:\n"
        "rec show - current schedule\n"
        "rec clear - remove schedule\n"
        "rec *:off - disable recording\n"
        "rec *:00-24 - record 24/7\n"
        "Motion On or M1 - enable motion schedule mode, or motion events if no schedule exists\n"
        "Motion Off or M0 - disable motion schedule mode, or motion events if no schedule exists\n"
        "rec help - show this help"
    ),
    "record_schedule_not_set": "Recording schedule is not set.",
    "record_schedule_invalid": "Recording schedule is invalid: {error}\nSpec: {spec}",
    "record_schedule_header": "Recording schedule:\n{command} {spec}",
    "record_schedule_mode": "Mode: {mode}",
    "record_schedule_mode_motion": "motion",
    "record_schedule_mode_continuous": "continuous",
    "record_schedule_active": "Active now: {start} - {end}",
    "record_schedule_next": "Next recording: {time}",
    "record_schedule_disabled": "Recording is disabled.",
    "record_schedule_removed": "Recording schedule removed.",
    "record_schedule_bad": "Bad recording schedule: {error}\nSend rec help for syntax.",
    "scheduled_recording_no_video": "Scheduled recording skipped: device has no video source.",
    "motion_recording_no_video": "Motion recording skipped: device has no video source.",
    "motion_schedule_enabled": "Motion recording mode enabled for current schedule.",
    "motion_schedule_disabled": "Motion recording mode disabled for current schedule.",
    "motion_detection_enabled": "Motion detection enabled.",
    "motion_detection_disabled": "Motion detection disabled.",
    "motion_start": "Motion detected.",
    "motion_stop": "Motion stopped.",
    "motion_detection_stopped": "Motion detection stopped: {error}",
    "no_devices": "No devices configured.",
    "devices_title": "Configured devices:",
    "device_list_item": "- {id}: {name} (topic_id={topic_id})",
    "no_active_wizard": "No active wizard.",
    "enumerate_devices_failed": "Failed to enumerate {kind} devices: {error}",
    "no_available_devices": "No available {kind} devices.",
    "select_device": "Select {kind} device number:",
    "device_kind_webcam": "webcam",
    "device_kind_screen": "screen",
    "cancel_hint": "Cancel - cancel",
    "rtsp_usage": "Usage: Add rtsp rtsp://login:password@ip_address/...",
    "rtsp_device_label": "RTSP camera {url}",
    "wizard_cancelled": "Wizard cancelled.",
    "invalid_device_number": "Send a valid number from the list, or Cancel.",
    "add_microphone": "Add microphone to this device? yes/no",
    "answer_yes_no": "Send yes or no.",
    "no_microphones": "No microphones found; adding device without audio.",
    "select_microphone": "Select microphone number:",
    "no_microphone_option": "0. No microphone",
    "invalid_microphone_number": "Send a valid microphone number, or 0.",
    "no_video_selected": "No video source selected.",
    "device_added": "Added device: {name}\n{hint}",
    "device_no_video": "Device has no video source.",
    "calling": "Calling with {name}...",
    "call_failed": "Failed to start call: {error}",
    "snapshot_caption": "Snapshot from {name}\nTime: {time}",
    "snapshot_failed": "Snapshot failed: {error}",
    "recording_progress": "Recording {seconds} second(s)...",
    "recording_no_media": "Recording failed: no media was captured.",
    "recording_caption": "Recording from {name}\nTime: {time}",
    "recording_failed": "Recording failed: {error}",
    "rtmp_already_active": "RTMP stream is already active or starting.",
    "rtmp_starting": "Starting RTMP stream...",
    "rtmp_create_failed": "Failed to create RTMP stream: {error}",
    "rtmp_url_failed": "Failed to get RTMP URL: {error}",
    "rtmp_started": "RTMP stream started.",
    "rtmp_stopped": "RTMP stream stopped.",
}
CALL_PROTOCOL = {
    "@type": "callProtocol",
    "udp_p2p": True,
    "udp_reflector": True,
    "min_layer": 65,
    "max_layer": 92,
    "library_versions": [
        "14.0.0",
        "13.0.0",
        "12.0.0",
        "11.0.0",
        "10.0.0",
        "9.0.0",
        "8.0.0",
        "7.0.0",
        "5.0.0",
        "2.7.7",
    ],
}


@dataclass(frozen=True)
class CaptureDevice:
    label: str
    input_name: str
    format_name: str
    options: dict[str, str] = field(default_factory=dict)


@dataclass
class CallMediaState:
    stop: threading.Event
    video_done: threading.Event
    audio_done: threading.Event
    device_id: str
    started_monotonic_us: int
    call_id: int = 0
    user_id: int = 0
    drain_started_at: Optional[float] = None
    stop_requested: bool = False


@dataclass(frozen=True)
class CaptureVideoFrame:
    y: bytes
    stride_y: int
    u: bytes
    stride_u: int
    v: bytes
    stride_v: int
    width: int
    height: int
    timestamp_us: int


@dataclass(frozen=True)
class CaptureAudioFrame:
    samples: bytes
    sample_rate_hz: int
    channels: int
    timestamp_us: int


@dataclass(frozen=True)
class RecordingResult:
    path: Path
    duration_seconds: int
    width: int
    height: int
    has_video: bool
    has_audio: bool


@dataclass
class AddDeviceWizard:
    user_id: int
    step: str
    kind: str
    candidates: list[CaptureDevice] = field(default_factory=list)
    video: Optional[CaptureDevice] = None
    audio_candidates: list[CaptureDevice] = field(default_factory=list)


@dataclass
class RtmpRuntime:
    device_id: str
    group_call_id: int
    topic_id: Optional[int]
    stop: threading.Event
    thread: threading.Thread


@dataclass
class ScheduleRuntime:
    device_id: str
    spec: str
    motion: bool
    stop: threading.Event
    thread: threading.Thread


@dataclass
class MotionRuntime:
    device_id: str
    topic_id: Optional[int]
    stop: threading.Event
    thread: threading.Thread


@dataclass(frozen=True)
class ScheduledRecordingResult:
    recording: RecordingResult
    started_at: datetime
    ended_at: datetime
    remuxed: bool = False


@dataclass(frozen=True)
class BufferedMediaPacket:
    sequence: int
    stream_index: int
    media_type: str
    data: bytes
    pts: Optional[int]
    dts: Optional[int]
    duration: Optional[int]
    time_base: Optional[Fraction]
    is_keyframe: bool = False


def configure_console_utf8() -> None:
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def require_pyav() -> None:
    global av
    if av is not None:
        return
    try:
        import av as pyav
    except ImportError as error:
        raise SystemExit(
            "PyAV import failed. The package or one of its native DLLs may be missing: "
            f"{type(error).__name__}: {error}"
        ) from error
    av = pyav


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or configure a libtam-based video surveillance system.")
    parser.add_argument("--setup", action="store_true", help="run interactive setup and save a JSON config")
    parser.add_argument("--login", default=None, help="Telegram login phone number in international format")
    parser.add_argument("--lang", default=DEFAULT_LANGUAGE, help="interface language, for example: en or ru")
    parser.add_argument("--config", default=None, help="path to the JSON config file")
    parser.add_argument("--video-size", choices=sorted(VIDEO_SIZE_MODES), default=DEFAULT_VIDEO_SIZE, help="stream/record video size")
    parser.add_argument("--debug-log", action="store_true", help="write RTSP call and RTMP stream diagnostics to watchdog_debug.log")
    parser.add_argument("--portable-self-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def normalize_language(value: Optional[str]) -> str:
    language = (value or DEFAULT_LANGUAGE).strip().lower()
    if not language:
        return DEFAULT_LANGUAGE
    if not all(char.isalnum() or char in {"-", "_"} for char in language):
        raise SystemExit(f"Invalid language code: {value}")
    return language


def load_translation(stuff_dir: Path, language: str) -> dict[str, str]:
    values = {key: str(value) for key, value in DEFAULT_TRANSLATION.items()}
    path = stuff_dir / f"watchdog_{language}.json"
    if not path.exists():
        if language == DEFAULT_LANGUAGE:
            return values
        raise SystemExit(f"Unsupported language '{language}': translation file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise SystemExit(f"Failed to load translation file {path}: {error}") from error
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid translation file {path}: expected JSON object")
    for key, value in data.items():
        if isinstance(value, str):
            values[key] = value
    return values


def translation_text(ui: Optional[dict[str, str]], key: str, **kwargs: object) -> str:
    values = DEFAULT_TRANSLATION if ui is None else ui
    text = values.get(key, DEFAULT_TRANSLATION.get(key, key))
    return text.format(**kwargs) if kwargs else text


def resolve_login(args: argparse.Namespace) -> str:
    login = (args.login or "").strip()
    while not login:
        login = input("Telegram login phone number: ").strip()
    return login


def prompt_yes_no(question: str, *, default: bool) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        value = input(question + suffix).strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Enter y or n.")


def prompt_menu(title: str, items: list[tuple[str, str]]) -> str:
    if not items:
        raise RuntimeError("empty menu")
    print(f"\n{title}")
    for index, (_, label) in enumerate(items, start=1):
        print(f"  {index}. {label}")
    while True:
        value = input("Select number: ").strip()
        try:
            index = int(value)
        except ValueError:
            print("Enter a number from the list.")
            continue
        if 1 <= index <= len(items):
            return items[index - 1][0]
        print("Enter a number from the list.")


def compact_options(options: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in options.items() if value}


def debug_wall_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def debug_field_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def debug_line(event: str, **fields: object) -> str:
    values = [debug_wall_time(), f"event={event}"]
    values.extend(f"{key}={debug_field_value(value)}" for key, value in fields.items())
    return " ".join(values)


def dshow_low_latency_options(include_audio: bool = False, include_video: bool = True) -> dict[str, str]:
    options = {
        "fflags": "nobuffer",
        "flags": "low_delay",
        "probesize": "32",
        "analyzeduration": "0",
        "rtbufsize": "1000000",
        "thread_queue_size": "4",
    }
    if include_audio:
        options["audio_buffer_size"] = "10"
    if include_video:
        options["framerate"] = DEFAULT_CAMERA_FPS
    return compact_options(options)


def linux_video_options() -> dict[str, str]:
    return compact_options(
        {
            "fflags": "nobuffer",
            "flags": "low_delay",
            "probesize": "32",
            "analyzeduration": "0",
            "thread_queue_size": "4",
            "framerate": DEFAULT_CAMERA_FPS,
        }
    )


def linux_audio_options() -> dict[str, str]:
    return compact_options(
        {
            "sample_rate": "48000",
            "channels": "2",
            "fragment_size": "960",
            "thread_queue_size": "4",
        }
    )


def rtsp_options() -> dict[str, str]:
    return compact_options(
        {
            "rtsp_transport": "tcp",
            "fflags": "nobuffer",
            "flags": "low_delay",
            "stimeout": "5000000",
            "thread_queue_size": "8",
        }
    )


def screen_options(width: int, height: int, *, offset_x: Optional[int] = None, offset_y: Optional[int] = None) -> dict[str, str]:
    options = {
        "framerate": DEFAULT_CAMERA_FPS,
        "video_size": f"{width}x{height}",
        "draw_mouse": "1",
        "thread_queue_size": "8",
    }
    if offset_x is not None:
        options["offset_x"] = str(offset_x)
    if offset_y is not None:
        options["offset_y"] = str(offset_y)
    return compact_options(options)


def dedupe_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        clean = name.strip()
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(clean)
    return result


def run_text_command(args: list[str], timeout: float = 8.0) -> str:
    try:
        completed = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout or ""


def windows_dshow_names_from_pyav() -> tuple[list[str], list[str]]:
    require_pyav()
    old_level = av.logging.get_level()
    old_skip_repeated = av.logging.get_skip_repeated()
    try:
        av.logging.set_level(av.logging.INFO)
        av.logging.set_libav_level(av.logging.INFO)
        av.logging.set_skip_repeated(False)
        with av.logging.Capture() as logs:
            try:
                av.open("dummy", format="dshow", options={"list_devices": "true"})
            except Exception:
                pass
    finally:
        av.logging.set_level(old_level)
        av.logging.set_skip_repeated(old_skip_repeated)

    text = "".join(str(item[2]) for item in logs)
    video_names: list[str] = []
    audio_names: list[str] = []
    for name, kind in re.findall(r'"([^"]+)"\s+\((video|audio)\)', text, flags=re.IGNORECASE):
        if kind.lower() == "video":
            video_names.append(name)
        else:
            audio_names.append(name)
    return dedupe_names(video_names), dedupe_names(audio_names)


def windows_dshow_names_from_ffmpeg() -> tuple[list[str], list[str]]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return [], []
    output = run_text_command([ffmpeg, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"])
    section: Optional[str] = None
    video_names: list[str] = []
    audio_names: list[str] = []
    for line in output.splitlines():
        if "DirectShow video devices" in line:
            section = "video"
            continue
        if "DirectShow audio devices" in line:
            section = "audio"
            continue
        if "Alternative name" in line:
            continue
        match = re.search(r'"([^"]+)"', line)
        if not match or section is None:
            continue
        if section == "video":
            video_names.append(match.group(1))
        else:
            audio_names.append(match.group(1))
    return dedupe_names(video_names), dedupe_names(audio_names)


def powershell_names(command: str) -> list[str]:
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if not powershell:
        return []
    output = run_text_command([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command])
    return dedupe_names([line for line in output.splitlines() if line.strip()])


def windows_video_devices() -> list[CaptureDevice]:
    video_names, _ = windows_dshow_names_from_pyav()
    if not video_names:
        video_names, _ = windows_dshow_names_from_ffmpeg()
    if not video_names:
        video_names = powershell_names(
            "Get-CimInstance Win32_PnPEntity | "
            "Where-Object { ($_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image') -and $_.Name } | "
            "Select-Object -ExpandProperty Name"
        )
    return [CaptureDevice(name, f"video={name}", "dshow", dshow_low_latency_options()) for name in video_names]


def windows_audio_devices() -> list[CaptureDevice]:
    _, audio_names = windows_dshow_names_from_pyav()
    if not audio_names:
        _, audio_names = windows_dshow_names_from_ffmpeg()
    if not audio_names:
        endpoint_names = powershell_names(
            "Get-CimInstance Win32_PnPEntity | "
            "Where-Object { $_.PNPClass -eq 'AudioEndpoint' -and $_.Name } | "
            "Select-Object -ExpandProperty Name"
        )
        input_pattern = re.compile(r"microphone|mic|input|line in|headset|capture", re.IGNORECASE)
        audio_names = [name for name in endpoint_names if input_pattern.search(name)] or endpoint_names
    return [
        CaptureDevice(
            name,
            f"audio={name}",
            "dshow",
            dshow_low_latency_options(include_audio=True, include_video=False),
        )
        for name in audio_names
    ]


def linux_video_devices() -> list[CaptureDevice]:
    devices: list[CaptureDevice] = []
    for path in sorted(Path("/dev").glob("video[0-9]*"), key=lambda item: item.name):
        sys_name = Path("/sys/class/video4linux") / path.name / "name"
        label = path.as_posix()
        if sys_name.exists():
            try:
                label = f"{path.as_posix()} - {sys_name.read_text(encoding='utf-8', errors='replace').strip()}"
            except OSError:
                pass
        devices.append(CaptureDevice(label, path.as_posix(), "v4l2", linux_video_options()))
    return devices


def linux_audio_devices() -> list[CaptureDevice]:
    devices: list[CaptureDevice] = []
    pactl = shutil.which("pactl")
    if pactl:
        output = run_text_command([pactl, "list", "short", "sources"])
        for line in output.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[1]
            if name.endswith(".monitor"):
                continue
            devices.append(CaptureDevice(f"PulseAudio source {name}", name, "pulse", linux_audio_options()))
    if devices:
        return devices

    devices.append(CaptureDevice("ALSA default input", "default", "alsa"))
    arecord = shutil.which("arecord")
    if arecord:
        output = run_text_command([arecord, "-l"])
        for line in output.splitlines():
            match = re.search(r"card\s+(\d+):\s*([^,]+),\s*device\s+(\d+):\s*(.+)", line)
            if not match:
                continue
            card, card_name, device, description = match.groups()
            devices.append(
                CaptureDevice(
                    f"ALSA hw:{card},{device} - {card_name.strip()} {description.strip()}",
                    f"hw:{card},{device}",
                    "alsa",
                )
            )
    return devices


def windows_gdigrab_screen_devices() -> list[CaptureDevice]:
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if not powershell:
        return []
    command = (
        "Add-Type 'using System; using System.Runtime.InteropServices; "
        'public static class DpiAwareness { [DllImport("user32.dll")] '
        "public static extern bool SetProcessDPIAware(); }'; "
        "[void][DpiAwareness]::SetProcessDPIAware(); "
        "Add-Type -AssemblyName System.Windows.Forms; "
        "[System.Windows.Forms.Screen]::AllScreens | ForEach-Object { "
        "\"$($_.DeviceName)|$($_.Bounds.X)|$($_.Bounds.Y)|$($_.Bounds.Width)|$($_.Bounds.Height)|$($_.Primary)\" }"
    )
    output = run_text_command([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command])
    devices: list[CaptureDevice] = []
    for index, line in enumerate(output.splitlines(), start=1):
        parts = line.strip().split("|")
        if len(parts) != 6:
            continue
        name, x_text, y_text, width_text, height_text, primary_text = parts
        try:
            x, y, width, height = int(x_text), int(y_text), int(width_text), int(height_text)
        except ValueError:
            continue
        if width <= 0 or height <= 0:
            continue
        primary = primary_text.strip().lower() == "true"
        label = f"Monitor {index}: {name} {width}x{height}+{x}+{y}"
        if primary:
            label += " primary"
        devices.append(CaptureDevice(label, "desktop", "gdigrab", screen_options(width, height, offset_x=x, offset_y=y)))
    return devices


def windows_capture_options(monitor_index: int, fallback: CaptureDevice) -> dict[str, str]:
    options = {
        "monitor_index": str(monitor_index),
        "framerate": fallback.options.get("framerate", DEFAULT_CAMERA_FPS),
        "cursor_capture": fallback.options.get("draw_mouse", "1"),
        "draw_border": "0",
        "fallback_input_name": fallback.input_name,
        "fallback_format_name": fallback.format_name,
    }
    for key, value in fallback.options.items():
        options[f"fallback_{key}"] = value
    return compact_options(options)


def windows_screen_devices() -> list[CaptureDevice]:
    devices: list[CaptureDevice] = []
    for index, fallback in enumerate(windows_gdigrab_screen_devices(), start=1):
        devices.append(
            CaptureDevice(
                fallback.label,
                f"windows_capture_monitor_{index}",
                "windows_capture",
                windows_capture_options(index, fallback),
            )
        )
    return devices


def linux_screen_devices() -> list[CaptureDevice]:
    display = os.environ.get("DISPLAY", "").strip()
    if not display:
        return []
    xrandr = shutil.which("xrandr")
    if not xrandr:
        return []
    output = run_text_command([xrandr, "--listmonitors"])
    devices: list[CaptureDevice] = []
    for line in output.splitlines():
        match = re.match(r"\s*(\d+):\s+[+*]*\S+\s+(\d+)/\d+x(\d+)/\d+\+(-?\d+)\+(-?\d+)\s+(.+)", line)
        if not match:
            continue
        index_text, width_text, height_text, x_text, y_text, name = match.groups()
        try:
            width, height, x, y = int(width_text), int(height_text), int(x_text), int(y_text)
        except ValueError:
            continue
        if width <= 0 or height <= 0:
            continue
        label = f"Monitor {int(index_text) + 1}: {name.strip()} {width}x{height}+{x}+{y}"
        devices.append(
            CaptureDevice(label, f"{display}+{x},{y}", "x11grab", screen_options(width, height))
        )
    return devices


def enumerate_video_devices() -> list[CaptureDevice]:
    if sys.platform == "win32":
        return windows_video_devices()
    if sys.platform.startswith("linux"):
        return linux_video_devices()
    return []


def enumerate_audio_devices() -> list[CaptureDevice]:
    if sys.platform == "win32":
        return windows_audio_devices()
    if sys.platform.startswith("linux"):
        return linux_audio_devices()
    return []


def enumerate_screen_devices() -> list[CaptureDevice]:
    if sys.platform == "win32":
        return windows_screen_devices()
    if sys.platform.startswith("linux"):
        return linux_screen_devices()
    return []


def choose_capture_device(title: str, devices: list[CaptureDevice]) -> CaptureDevice:
    if not devices:
        raise RuntimeError(f"No {title} devices are available.")
    items = [(str(index), device.label) for index, device in enumerate(devices)]
    selected = prompt_menu(f"Available {title} devices:", items)
    return devices[int(selected)]


def redact_url(value: str) -> str:
    parts = urlsplit(value)
    if not parts.username and not parts.password:
        return value
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    username = parts.username or "user"
    netloc = f"{username}:***@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def prompt_rtsp_device() -> CaptureDevice:
    while True:
        url = input("RTSP URL (rtsp://login:password@ip_address/...): ").strip()
        parts = urlsplit(url)
        if parts.scheme.lower() == "rtsp" and parts.hostname:
            return CaptureDevice(f"RTSP camera {redact_url(url)}", url, "rtsp", rtsp_options())
        print("Enter a valid rtsp:// URL.")


def capture_to_json(device: CaptureDevice) -> dict[str, object]:
    return {
        "display_name": device.label,
        "machine_id": device.input_name,
        "label": device.label,
        "input_name": device.input_name,
        "format_name": device.format_name,
        "options": dict(device.options),
    }


def maybe_choose_microphone(audio_devices: list[CaptureDevice]) -> Optional[dict[str, object]]:
    if not prompt_yes_no("Add a microphone to this source?", default=False):
        return None
    if not audio_devices:
        print("No microphone devices were found.")
        return None
    return capture_to_json(choose_capture_device("audio", audio_devices))


def capture_from_json(value: object) -> Optional[CaptureDevice]:
    if not isinstance(value, dict):
        return None
    label = str(value.get("label") or value.get("display_name") or value.get("input_name") or "").strip()
    input_name = str(value.get("input_name") or value.get("machine_id") or "").strip()
    format_name = str(value.get("format_name") or "").strip()
    options_value = value.get("options")
    options = {str(key): str(item) for key, item in options_value.items()} if isinstance(options_value, dict) else {}
    if not input_name or not format_name:
        return None
    return CaptureDevice(label or input_name, input_name, format_name, options)


def is_rtsp_capture(device: Optional[CaptureDevice]) -> bool:
    return device is not None and device.format_name.lower() == "rtsp"


def windows_display_token(text: str) -> Optional[str]:
    match = re.search(r"DISPLAY\d+", str(text or ""), re.IGNORECASE)
    return match.group(0).upper() if match else None


def windows_monitor_number(text: str) -> Optional[int]:
    capture_match = re.search(r"windows_capture_monitor_(\d+)", str(text or ""), re.IGNORECASE)
    if capture_match:
        return int(capture_match.group(1))
    monitor_match = re.search(r"\bMonitor\s+(\d+)\b", str(text or ""), re.IGNORECASE)
    if monitor_match:
        return int(monitor_match.group(1))
    display_match = re.search(r"DISPLAY(\d+)", str(text or ""), re.IGNORECASE)
    return int(display_match.group(1)) if display_match else None


def windows_screen_devices_cached() -> list[CaptureDevice]:
    global _WINDOWS_SCREEN_DEVICE_CACHE
    if _WINDOWS_SCREEN_DEVICE_CACHE is None:
        _WINDOWS_SCREEN_DEVICE_CACHE = windows_gdigrab_screen_devices()
    return _WINDOWS_SCREEN_DEVICE_CACHE


def normalize_windows_screen_capture(device: CaptureDevice) -> CaptureDevice:
    if sys.platform != "win32" or device.format_name.lower() != "gdigrab":
        return device
    screens = windows_screen_devices_cached()
    if not screens:
        return device

    token = windows_display_token(device.label) or windows_display_token(device.input_name)
    matched = None
    if token is not None:
        matched = next((screen for screen in screens if windows_display_token(screen.label) == token), None)
    if matched is None and len(screens) == 1:
        matched = screens[0]
    if matched is None:
        offset_x = device.options.get("offset_x")
        offset_y = device.options.get("offset_y")
        matched = next(
            (
                screen
                for screen in screens
                if screen.options.get("offset_x") == offset_x and screen.options.get("offset_y") == offset_y
            ),
            None,
        )
    if matched is None:
        return device

    options = dict(matched.options)
    for key in ("framerate", "draw_mouse", "thread_queue_size"):
        if key in device.options:
            options[key] = device.options[key]
    return CaptureDevice(matched.label, matched.input_name, matched.format_name, options)


def windows_gdigrab_fallback_device(device: CaptureDevice) -> Optional[CaptureDevice]:
    if sys.platform != "win32":
        return None
    format_name = device.format_name.lower()
    if format_name == "gdigrab":
        return normalize_windows_screen_capture(device)
    if format_name != "windows_capture":
        return None

    fallback_input_name = device.options.get("fallback_input_name", "desktop")
    fallback_format_name = device.options.get("fallback_format_name", "gdigrab")
    fallback_options = {
        key[len("fallback_") :]: value
        for key, value in device.options.items()
        if key.startswith("fallback_") and key not in {"fallback_input_name", "fallback_format_name"}
    }
    if fallback_options:
        return CaptureDevice(device.label, fallback_input_name, fallback_format_name, fallback_options)

    monitor_index = windows_capture_monitor_index(device)
    screens = windows_screen_devices_cached()
    if monitor_index is not None and 1 <= monitor_index <= len(screens):
        return screens[monitor_index - 1]
    if len(screens) == 1:
        return screens[0]
    return None


def windows_capture_monitor_index(device: CaptureDevice) -> Optional[int]:
    try:
        index = int(str(device.options.get("monitor_index") or "").strip())
        if index > 0:
            return index
    except ValueError:
        pass
    return windows_monitor_number(device.label) or windows_monitor_number(device.input_name)


def windows_capture_device_for_screen(device: CaptureDevice, fallback: Optional[CaptureDevice]) -> Optional[CaptureDevice]:
    if sys.platform != "win32":
        return None
    format_name = device.format_name.lower()
    if format_name == "windows_capture":
        return device
    if format_name != "gdigrab" or device.input_name.lower() != "desktop" or fallback is None:
        return None
    monitor_index = windows_capture_monitor_index(device)
    if monitor_index is None:
        monitor_index = windows_capture_monitor_index(fallback)
    if monitor_index is None:
        monitor_index = 1
    return CaptureDevice(
        fallback.label,
        f"windows_capture_monitor_{monitor_index}",
        "windows_capture",
        windows_capture_options(monitor_index, fallback),
    )


class _SingleVideoStreams:
    def __init__(self) -> None:
        self.video = [object()]


def _option_bool(options: dict[str, str], key: str, default: bool) -> bool:
    value = str(options.get(key, "")).strip().casefold()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _option_int(options: dict[str, str], key: str, default: int, *, minimum: int = 1, maximum: int = 240) -> int:
    try:
        return max(minimum, min(maximum, int(str(options.get(key, default)).strip())))
    except ValueError:
        return default


def is_capture_border_unsupported_error(error: Exception) -> bool:
    text = str(error).casefold()
    return "capture border" in text and "not supported" in text


def windows_build_number() -> int:
    if sys.platform != "win32":
        return 0
    try:
        return int(sys.getwindowsversion().build)
    except Exception:
        return 0


def windows_capture_border_settings_supported() -> bool:
    # GraphicsCaptureSession.IsBorderRequired is available on Windows 11+.
    return windows_build_number() >= 22000


class WindowsGraphicsCaptureContainer:
    def __init__(self, device: CaptureDevice, fallback: Optional[CaptureDevice]) -> None:
        self.device = device
        self.fallback = fallback
        self.streams = _SingleVideoStreams()
        self._frames: "queue.Queue[Optional[object]]" = queue.Queue(maxsize=2)
        self._closed = threading.Event()
        self._last_frame = None
        self._capture = None
        self._capture_control = None
        self._fallback_container = None
        self._fps = _option_int(device.options, "framerate", int(DEFAULT_CAMERA_FPS), minimum=1, maximum=120)

    def __enter__(self):
        try:
            self._start_windows_capture()
        except Exception as error:
            return self._start_fallback(error)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._closed.set()
        if self._capture_control is not None:
            try:
                self._capture_control.stop()
            except Exception:
                pass
        if self._fallback_container is not None:
            try:
                self._fallback_container.close()
            except Exception:
                pass

    def _start_fallback(self, error: Exception):
        if self._capture_control is not None:
            try:
                self._capture_control.stop()
            except Exception:
                pass
        if self.fallback is None:
            raise error
        print(f"[watchdog] Windows.Graphics.Capture unavailable; falling back to gdigrab: {error}", flush=True)
        self._fallback_container = av.open(self.fallback.input_name, format=self.fallback.format_name, options=dict(self.fallback.options))
        self.streams = self._fallback_container.streams
        return self

    def _push_frame(self, frame_array: Optional[object]) -> None:
        if self._closed.is_set():
            return
        try:
            self._frames.put_nowait(frame_array)
            return
        except queue.Full:
            pass
        try:
            self._frames.get_nowait()
        except queue.Empty:
            pass
        try:
            self._frames.put_nowait(frame_array)
        except queue.Full:
            pass

    def _start_windows_capture(self) -> None:
        from windows_capture import WindowsCapture

        monitor_index = windows_capture_monitor_index(self.device)

        def start_capture(*, with_draw_border: bool) -> None:
            self._closed.clear()
            self._frames = queue.Queue(maxsize=2)
            self._capture_control = None
            self._last_frame = None

            kwargs = {
                "cursor_capture": _option_bool(self.device.options, "cursor_capture", True),
                "monitor_index": monitor_index,
                "minimum_update_interval": max(1, int(1000 / max(1, self._fps))),
            }
            if with_draw_border:
                kwargs["draw_border"] = _option_bool(self.device.options, "draw_border", False)
            self._capture = WindowsCapture(**kwargs)

            @self._capture.event
            def on_frame_arrived(frame, capture_control) -> None:
                try:
                    self._push_frame(frame.frame_buffer.copy())
                except Exception:
                    self._closed.set()
                    try:
                        capture_control.stop()
                    except Exception:
                        pass

            @self._capture.event
            def on_closed() -> None:
                self._push_frame(None)
                self._closed.set()

            self._capture_control = self._capture.start_free_threaded()
            try:
                first_frame = self._frames.get(timeout=3.0)
            except queue.Empty as error:
                raise RuntimeError("no Windows.Graphics.Capture frame received") from error
            if first_frame is None:
                raise RuntimeError("capture closed before the first frame")
            self._last_frame = first_frame

        if not windows_capture_border_settings_supported():
            print("[watchdog] Windows.Graphics.Capture draw_border is not supported on this Windows version; starting without draw_border", flush=True)
            start_capture(with_draw_border=False)
            return
        try:
            start_capture(with_draw_border=True)
        except Exception as error:
            if not is_capture_border_unsupported_error(error):
                raise
            if self._capture_control is not None:
                try:
                    self._capture_control.stop()
                except Exception:
                    pass
            print("[watchdog] Windows.Graphics.Capture draw_border was rejected; retrying without draw_border", flush=True)
            start_capture(with_draw_border=False)

    def decode(self, stream):
        if self._fallback_container is not None:
            yield from self._fallback_container.decode(stream)
            return
        frame_interval = 1.0 / max(1, self._fps)
        next_frame_at = time.monotonic()
        if self._last_frame is not None:
            yield av.VideoFrame.from_ndarray(self._last_frame, format="bgra")
            next_frame_at = time.monotonic() + frame_interval
        while not self._closed.is_set():
            timeout = max(0.01, min(frame_interval, max(0.0, next_frame_at - time.monotonic())))
            try:
                frame_array = self._frames.get(timeout=timeout)
                if frame_array is None:
                    break
                self._last_frame = frame_array
            except queue.Empty:
                pass
            if self._last_frame is None or time.monotonic() < next_frame_at:
                continue
            yield av.VideoFrame.from_ndarray(self._last_frame, format="bgra")
            next_frame_at = max(next_frame_at + frame_interval, time.monotonic())


def open_capture(device: CaptureDevice):
    require_pyav()
    if sys.platform == "win32":
        fallback = windows_gdigrab_fallback_device(device)
        windows_capture_device = windows_capture_device_for_screen(device, fallback)
        if windows_capture_device is not None:
            return WindowsGraphicsCaptureContainer(windows_capture_device, fallback)
    device = normalize_windows_screen_capture(device)
    return av.open(device.input_name, format=device.format_name, options=dict(device.options))


def plane_bytes(plane) -> bytes:
    return bytes(plane)


def packed_plane_bytes(data: bytes, stride: int, width: int, height: int) -> bytes:
    if stride == width:
        return data[: width * height]
    rows = bytearray(width * height)
    for row in range(height):
        src_offset = row * stride
        dst_offset = row * width
        rows[dst_offset : dst_offset + width] = data[src_offset : src_offset + width]
    return bytes(rows)


def even_size(value: int) -> int:
    return max(2, int(value) - (int(value) % 2))


def even_offset(value: int) -> int:
    return int(value) - (int(value) % 2)


def target_video_dimensions(frame, video_size: str) -> tuple[int, int]:
    width, height = VIDEO_SIZE_MODES.get(video_size, VIDEO_SIZE_MODES[DEFAULT_VIDEO_SIZE])
    return (height, width) if frame.width > frame.height else (width, height)


def copy_plane_region(
    dst: bytearray,
    dst_stride: int,
    dst_x: int,
    dst_y: int,
    src: bytes,
    src_stride: int,
    width: int,
    height: int,
) -> None:
    for row in range(height):
        dst_offset = (dst_y + row) * dst_stride + dst_x
        src_offset = row * src_stride
        dst[dst_offset : dst_offset + width] = src[src_offset : src_offset + width]


def strided_plane_bytes(data: bytes, source_stride: int, destination_stride: int, row_width: int, rows: int) -> bytes:
    if source_stride == destination_stride:
        return data[: destination_stride * rows]
    output = bytearray(destination_stride * rows)
    for row in range(rows):
        source_offset = row * source_stride
        destination_offset = row * destination_stride
        output[destination_offset : destination_offset + row_width] = data[source_offset : source_offset + row_width]
    return bytes(output)


def fit_video_frame(frame, video_size: str):
    target_width, target_height = target_video_dimensions(frame, video_size)
    frame_format = getattr(getattr(frame, "format", None), "name", "")
    if frame.width == target_width and frame.height == target_height and frame_format == "yuv420p":
        return frame

    scale = min(target_width / frame.width, target_height / frame.height)
    scaled_width = min(target_width, even_size(round(frame.width * scale)))
    scaled_height = min(target_height, even_size(round(frame.height * scale)))
    if scaled_width == target_width and scaled_height == target_height:
        return frame.reformat(width=target_width, height=target_height, format="yuv420p")

    scaled = frame.reformat(width=scaled_width, height=scaled_height, format="yuv420p")
    output = av.VideoFrame(target_width, target_height, "yuv420p")
    y_plane = bytearray([16]) * (output.planes[0].line_size * target_height)
    u_plane = bytearray([128]) * (output.planes[1].line_size * ((target_height + 1) // 2))
    v_plane = bytearray([128]) * (output.planes[2].line_size * ((target_height + 1) // 2))

    offset_x = even_offset((target_width - scaled_width) // 2)
    offset_y = even_offset((target_height - scaled_height) // 2)
    copy_plane_region(
        y_plane,
        output.planes[0].line_size,
        offset_x,
        offset_y,
        bytes(scaled.planes[0]),
        scaled.planes[0].line_size,
        scaled_width,
        scaled_height,
    )
    copy_plane_region(
        u_plane,
        output.planes[1].line_size,
        offset_x // 2,
        offset_y // 2,
        bytes(scaled.planes[1]),
        scaled.planes[1].line_size,
        (scaled_width + 1) // 2,
        (scaled_height + 1) // 2,
    )
    copy_plane_region(
        v_plane,
        output.planes[2].line_size,
        offset_x // 2,
        offset_y // 2,
        bytes(scaled.planes[2]),
        scaled.planes[2].line_size,
        (scaled_width + 1) // 2,
        (scaled_height + 1) // 2,
    )
    output.planes[0].update(bytes(y_plane))
    output.planes[1].update(bytes(u_plane))
    output.planes[2].update(bytes(v_plane))
    return output


def capture_video_frame_from_av(frame, timestamp_us: int, video_size: str) -> CaptureVideoFrame:
    fitted = fit_video_frame(frame, video_size)
    return CaptureVideoFrame(
        bytes(fitted.planes[0]),
        fitted.planes[0].line_size,
        bytes(fitted.planes[1]),
        fitted.planes[1].line_size,
        bytes(fitted.planes[2]),
        fitted.planes[2].line_size,
        fitted.width,
        fitted.height,
        timestamp_us,
    )


def av_frame_from_capture_video(item: CaptureVideoFrame):
    frame = av.VideoFrame(item.width, item.height, "yuv420p")
    frame.planes[0].update(
        strided_plane_bytes(item.y, item.stride_y, frame.planes[0].line_size, item.width, item.height)
    )
    chroma_width = (item.width + 1) // 2
    chroma_height = (item.height + 1) // 2
    frame.planes[1].update(
        strided_plane_bytes(item.u, item.stride_u, frame.planes[1].line_size, chroma_width, chroma_height)
    )
    frame.planes[2].update(
        strided_plane_bytes(item.v, item.stride_v, frame.planes[2].line_size, chroma_width, chroma_height)
    )
    return frame


def send_video_until_ok(
    client: Tam,
    call_id: int,
    frame,
    timestamp_us: int,
    stop_event: threading.Event,
    video_size: str,
    *,
    debug_log: Optional[Callable[[str], None]] = None,
    frame_index: Optional[int] = None,
    source_pts_us: Optional[int] = None,
    decoded_wall: Optional[str] = None,
    decoded_monotonic_us: Optional[int] = None,
    decode_gap_ms: Optional[float] = None,
    skipped_before: int = 0,
) -> bool:
    total_started = time.perf_counter()
    fit_started = time.perf_counter()
    i420 = fit_video_frame(frame, video_size)
    fit_ms = (time.perf_counter() - fit_started) * 1000.0
    attempts = 0
    while not stop_event.is_set():
        try:
            attempts += 1
            send_started = time.perf_counter()
            client.send_video_i420(
                call_id,
                plane_bytes(i420.planes[0]),
                plane_bytes(i420.planes[1]),
                plane_bytes(i420.planes[2]),
                i420.width,
                i420.height,
                stride_y=i420.planes[0].line_size,
                stride_u=i420.planes[1].line_size,
                stride_v=i420.planes[2].line_size,
                timestamp_us=timestamp_us,
            )
            send_ms = (time.perf_counter() - send_started) * 1000.0
            total_ms = (time.perf_counter() - total_started) * 1000.0
            pending_ms: Optional[int] = None
            try:
                pending_ms = client.outgoing_media_pending_ms(call_id)
            except RuntimeError:
                pass
            if debug_log is not None:
                debug_log(
                    debug_line(
                        "rtsp_video_sent",
                        call_id=call_id,
                        frame=frame_index,
                        decoded_wall=decoded_wall,
                        decoded_monotonic_us=decoded_monotonic_us,
                        source_pts_us=source_pts_us,
                        source_width=getattr(frame, "width", None),
                        source_height=getattr(frame, "height", None),
                        output_width=i420.width,
                        output_height=i420.height,
                        timestamp_us=timestamp_us,
                        decode_gap_ms=None if decode_gap_ms is None else round(decode_gap_ms, 3),
                        skipped_before=skipped_before,
                        fit_ms=round(fit_ms, 3),
                        send_ms=round(send_ms, 3),
                        total_ms=round(total_ms, 3),
                        attempts=attempts,
                        pending_ms=pending_ms,
                    )
                )
            return True
        except RuntimeError:
            if debug_log is not None:
                debug_log(debug_line("rtsp_video_send_retry", call_id=call_id, frame=frame_index, attempts=attempts))
            stop_event.wait(0.02)
    if debug_log is not None:
        debug_log(debug_line("rtsp_video_send_stopped", call_id=call_id, frame=frame_index, attempts=attempts))
    return False


def send_audio_until_ok(
    client: Tam,
    call_id: int,
    chunk: bytes,
    timestamp_us: Optional[int],
    stop_event: threading.Event,
) -> bool:
    while not stop_event.is_set():
        try:
            if timestamp_us is None:
                client.send_audio_pcm16(call_id, chunk, sample_rate_hz=AUDIO_RATE, channels=AUDIO_CHANNELS)
            else:
                client.send_audio_pcm16(
                    call_id,
                    chunk,
                    sample_rate_hz=AUDIO_RATE,
                    channels=AUDIO_CHANNELS,
                    timestamp_us=timestamp_us,
                )
            return True
        except RuntimeError:
            stop_event.wait(0.02)
    return False


def append_resampled_audio(pending: bytearray, frames) -> None:
    if frames is None:
        return
    for audio_frame in frames:
        byte_count = audio_frame.samples * AUDIO_CHANNELS * 2
        pending.extend(bytes(audio_frame.planes[0])[:byte_count])


def is_video_frame(frame) -> bool:
    return hasattr(frame, "width") and hasattr(frame, "height") and hasattr(frame, "reformat")


def is_audio_frame(frame) -> bool:
    return hasattr(frame, "samples") and hasattr(frame, "sample_rate")


def fraction_text(value: object) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def frame_pts_us(frame) -> Optional[int]:
    pts = getattr(frame, "pts", None)
    time_base = getattr(frame, "time_base", None)
    if pts is None or time_base is None:
        return None
    try:
        return int(pts * time_base * 1_000_000)
    except Exception:
        return None


def stream_debug_fields(stream) -> dict[str, object]:
    codec_context = getattr(stream, "codec_context", None)
    codec_format = getattr(codec_context, "format", None)
    fields: dict[str, object] = {
        "stream_index": getattr(stream, "index", None),
        "stream_type": getattr(stream, "type", None),
        "codec": getattr(codec_context, "name", None),
        "profile": getattr(codec_context, "profile", None),
        "level": getattr(codec_context, "level", None),
        "time_base": fraction_text(getattr(stream, "time_base", None)),
        "average_rate": fraction_text(getattr(stream, "average_rate", None)),
        "base_rate": fraction_text(getattr(stream, "base_rate", None)),
        "duration": getattr(stream, "duration", None),
        "frames": getattr(stream, "frames", None),
    }
    if getattr(stream, "type", None) == "video":
        fields.update(
            {
                "width": getattr(stream, "width", None),
                "height": getattr(stream, "height", None),
                "pix_fmt": getattr(codec_format, "name", None),
            }
        )
    elif getattr(stream, "type", None) == "audio":
        fields.update(
            {
                "sample_rate": getattr(stream, "sample_rate", None),
                "channels": getattr(stream, "channels", None),
                "layout": getattr(getattr(stream, "layout", None), "name", None),
                "sample_fmt": getattr(codec_format, "name", None),
            }
        )
    return fields


def stream_video_device(client: Tam, call_id: int, device: CaptureDevice, state: CallMediaState, video_size: str) -> None:
    try:
        with open_capture(device) as container:
            stream = container.streams.video[0]
            frame_interval = 1.0 / DEFAULT_LIVE_VIDEO_FPS
            next_send_at = time.monotonic()
            for frame in container.decode(stream):
                if state.stop.is_set():
                    break
                now = time.monotonic()
                if now < next_send_at:
                    continue
                if not send_video_until_ok(client, call_id, frame, -1, state.stop, video_size):
                    break
                next_send_at = max(next_send_at + frame_interval, now + frame_interval)
    except Exception as error:
        print(f"[watchdog] video stream stopped: call_id={call_id}, device={device.label}, error={error}", flush=True)
    finally:
        state.video_done.set()


def stream_audio_device(client: Tam, call_id: int, device: CaptureDevice, state: CallMediaState) -> None:
    try:
        with open_capture(device) as container:
            stream = container.streams.audio[0]
            resampler = av.audio.resampler.AudioResampler(format="s16", layout="stereo", rate=AUDIO_RATE)
            pending = bytearray()
            for frame in container.decode(stream):
                if state.stop.is_set():
                    break
                append_resampled_audio(pending, resampler.resample(frame))
                while len(pending) >= AUDIO_CHUNK_BYTES:
                    chunk = bytes(pending[:AUDIO_CHUNK_BYTES])
                    del pending[:AUDIO_CHUNK_BYTES]
                    if not send_audio_until_ok(client, call_id, chunk, None, state.stop):
                        return
    except Exception as error:
        print(f"[watchdog] audio stream stopped: call_id={call_id}, device={device.label}, error={error}", flush=True)
    finally:
        state.audio_done.set()


def stream_rtsp_call_device(
    client: Tam,
    call_id: int,
    device: CaptureDevice,
    state: CallMediaState,
    video_size: str,
    debug_log: Optional[Callable[[str], None]] = None,
) -> None:
    def log_debug(event: str, **fields: object) -> None:
        if debug_log is not None:
            debug_log(debug_line(event, call_id=call_id, **fields))

    log_debug(
        "rtsp_call_opening",
        device_label=device.label,
        rtsp_url=redact_url(device.input_name),
        options=json.dumps(device.options, ensure_ascii=False, sort_keys=True),
        video_size=video_size,
        live_fps=DEFAULT_LIVE_VIDEO_FPS,
    )
    opened_started = time.perf_counter()
    try:
        with open_capture(device) as container:
            log_debug("rtsp_call_opened", open_ms=round((time.perf_counter() - opened_started) * 1000.0, 3))
            for stream in container.streams:
                log_debug("rtsp_stream", **stream_debug_fields(stream))
            video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
            audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
            if video_stream is None:
                raise RuntimeError("RTSP input has no video stream")

            log_debug("rtsp_video_stream_selected", **stream_debug_fields(video_stream))
            streams = [video_stream]
            resampler = None
            pending_audio = bytearray()
            audio_chunks = 0
            if audio_stream is not None:
                log_debug("rtsp_audio_stream_selected", **stream_debug_fields(audio_stream))
                streams.append(audio_stream)
                resampler = av.audio.resampler.AudioResampler(format="s16", layout="stereo", rate=AUDIO_RATE)
            else:
                log_debug("rtsp_audio_stream_missing")

            frame_interval = 1.0 / DEFAULT_LIVE_VIDEO_FPS
            next_send_at = time.monotonic()
            video_decoded = 0
            video_sent = 0
            video_skipped = 0
            skipped_since_send = 0
            last_video_decode_monotonic: Optional[float] = None
            for frame in container.decode(*streams):
                if state.stop.is_set():
                    break
                if is_video_frame(frame):
                    video_decoded += 1
                    now = time.monotonic()
                    now_us = time.monotonic_ns() // 1_000
                    wall = debug_wall_time()
                    decode_gap_ms = None
                    if last_video_decode_monotonic is not None:
                        decode_gap_ms = (now - last_video_decode_monotonic) * 1000.0
                        if debug_log is not None and decode_gap_ms > 500.0:
                            log_debug(
                                "rtsp_video_decode_gap",
                                decoded=video_decoded,
                                gap_ms=round(decode_gap_ms, 3),
                                source_pts_us=frame_pts_us(frame),
                            )
                    last_video_decode_monotonic = now
                    if now < next_send_at:
                        video_skipped += 1
                        skipped_since_send += 1
                        if debug_log is not None and (skipped_since_send == 1 or skipped_since_send % 30 == 0):
                            log_debug(
                                "rtsp_video_skipped_fps_gate",
                                decoded=video_decoded,
                                skipped_total=video_skipped,
                                skipped_since_send=skipped_since_send,
                                wait_ms=round((next_send_at - now) * 1000.0, 3),
                                source_pts_us=frame_pts_us(frame),
                                source_width=getattr(frame, "width", None),
                                source_height=getattr(frame, "height", None),
                            )
                        continue
                    video_sent += 1
                    if not send_video_until_ok(
                        client,
                        call_id,
                        frame,
                        -1,
                        state.stop,
                        video_size,
                        debug_log=debug_log,
                        frame_index=video_sent,
                        source_pts_us=frame_pts_us(frame),
                        decoded_wall=wall,
                        decoded_monotonic_us=now_us,
                        decode_gap_ms=decode_gap_ms,
                        skipped_before=skipped_since_send,
                    ):
                        break
                    skipped_since_send = 0
                    next_send_at = max(next_send_at + frame_interval, now + frame_interval)
                    continue
                if resampler is not None and is_audio_frame(frame):
                    audio_pts_us = frame_pts_us(frame)
                    append_resampled_audio(pending_audio, resampler.resample(frame))
                    while len(pending_audio) >= AUDIO_CHUNK_BYTES:
                        chunk = bytes(pending_audio[:AUDIO_CHUNK_BYTES])
                        del pending_audio[:AUDIO_CHUNK_BYTES]
                        audio_chunks += 1
                        audio_send_started = time.perf_counter()
                        if not send_audio_until_ok(client, call_id, chunk, None, state.stop):
                            return
                        if debug_log is not None and (audio_chunks <= 5 or audio_chunks % 100 == 0):
                            log_debug(
                                "rtsp_audio_sent",
                                chunk=audio_chunks,
                                source_pts_us=audio_pts_us,
                                chunk_bytes=len(chunk),
                                pending_audio_bytes=len(pending_audio),
                                send_ms=round((time.perf_counter() - audio_send_started) * 1000.0, 3),
                            )
    except Exception as error:
        log_debug("rtsp_call_error", error=error)
        print(f"[watchdog] RTSP call stream stopped: call_id={call_id}, device={device.label}, error={error}", flush=True)
    finally:
        log_debug("rtsp_call_finished")
        state.video_done.set()
        state.audio_done.set()


def audio_layout_name(channels: int) -> Optional[str]:
    if channels == 1:
        return "mono"
    if channels == 2:
        return "stereo"
    return None


def save_snapshot_frame(frame, output_path: Path) -> None:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for snapshots: python -m pip install opencv-python") from error
    image = frame.to_ndarray(format="bgr24")
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"failed to write snapshot: {output_path}")


def capture_snapshot(device: CaptureDevice, output_path: Path, video_size: str) -> tuple[int, int]:
    require_pyav()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open_capture(device) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            fitted = fit_video_frame(frame, video_size)
            save_snapshot_frame(fitted, output_path)
            return fitted.width, fitted.height
    raise RuntimeError("no video frame received")


class DeviceRecorder:
    def __init__(self, path: Path, video_size: str) -> None:
        self.path = path
        self.video_size = video_size
        self.items: "queue.Queue[Optional[Union[CaptureVideoFrame, CaptureAudioFrame]]]" = queue.Queue(
            maxsize=RECORDING_QUEUE_MAX_ITEMS
        )
        self.stop = threading.Event()
        self.result: Optional[RecordingResult] = None
        self.error: Optional[Exception] = None
        self.dropped = 0
        self.thread = threading.Thread(target=self.run, name=f"watchdog-recorder-{path.stem}", daemon=True)
        self.thread.start()

    def submit(self, item: Union[CaptureVideoFrame, CaptureAudioFrame]) -> bool:
        while not self.stop.is_set():
            try:
                self.items.put(item, timeout=0.25)
                return True
            except queue.Full:
                continue
        return False

    def close(self) -> Optional[RecordingResult]:
        self.stop.set()
        while self.thread.is_alive():
            try:
                self.items.put(None, timeout=0.2)
                break
            except queue.Full:
                continue
        self.thread.join()
        if self.result is None and self.path.exists() and self.path.stat().st_size > 0:
            self.result = probe_media_file(self.path)
        if self.dropped:
            print(f"[watchdog] recorder dropped {self.dropped} frame(s): {self.path}", flush=True)
        if self.error is not None:
            print(f"[watchdog] recorder warning: {self.error}", flush=True)
        return self.result

    def run(self) -> None:
        container = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            container = av.open(str(self.path), "w")
            video_stream = None
            audio_stream = None
            first_media_us: Optional[int] = None
            first_video_us: Optional[int] = None
            last_video_us = 0
            last_audio_us = 0
            video_count = 0
            audio_count = 0
            audio_rate = 0
            audio_channels = 0
            started = False
            pending: list[Union[CaptureVideoFrame, CaptureAudioFrame]] = []
            width = 0
            height = 0

            def add_video_stream(item: CaptureVideoFrame) -> None:
                nonlocal video_stream, width, height
                if video_stream is not None:
                    return
                width, height = item.width, item.height
                codec_candidates = ("libx264",)
                for codec_name in codec_candidates:
                    try:
                        video_stream = container.add_stream(codec_name, rate=DEFAULT_RECORD_FPS)
                        if codec_name == "libx264":
                            options = {"preset": "veryfast", "tune": "zerolatency", "crf": "23"}
                            for options_target in (video_stream, getattr(video_stream, "codec_context", None)):
                                if options_target is None:
                                    continue
                                try:
                                    options_target.options = options
                                except Exception:
                                    pass
                        break
                    except Exception:
                        video_stream = None
                if video_stream is None:
                    raise RuntimeError("no supported MP4 video encoder found")
                video_stream.width = width
                video_stream.height = height
                video_stream.pix_fmt = "yuv420p"
                video_stream.time_base = Fraction(1, 1_000_000)

            def add_audio_stream(rate: int, channels: int) -> None:
                nonlocal audio_stream, audio_rate, audio_channels
                if audio_stream is not None:
                    return
                layout = audio_layout_name(channels)
                if layout is None:
                    raise RuntimeError(f"unsupported audio channel count: {channels}")
                audio_stream = container.add_stream("aac", rate=rate)
                audio_stream.layout = layout
                audio_rate = rate
                audio_channels = channels

            def add_default_audio_stream() -> None:
                first_audio = next((pending_item for pending_item in pending if isinstance(pending_item, CaptureAudioFrame)), None)
                if first_audio is None:
                    add_audio_stream(AUDIO_RATE, AUDIO_CHANNELS)
                else:
                    add_audio_stream(first_audio.sample_rate_hz, first_audio.channels)

            def item_timestamp_us(item: Union[CaptureVideoFrame, CaptureAudioFrame]) -> int:
                return max(0, int(item.timestamp_us))

            def write_video(item: CaptureVideoFrame) -> None:
                nonlocal first_media_us, first_video_us, last_video_us, video_count
                if first_video_us is None:
                    first_video_us = item_timestamp_us(item)
                if first_media_us is None:
                    first_media_us = first_video_us
                add_video_stream(item)
                frame = av_frame_from_capture_video(item)
                rel_us = item_timestamp_us(item) - first_video_us
                if video_count and rel_us <= last_video_us:
                    rel_us = last_video_us + int(1_000_000 / DEFAULT_RECORD_FPS)
                frame.pts = max(0, rel_us)
                frame.time_base = Fraction(1, 1_000_000)
                for packet in video_stream.encode(frame):
                    container.mux(packet)
                last_video_us = frame.pts
                video_count += 1

            def write_audio(item: CaptureAudioFrame) -> None:
                nonlocal first_media_us, last_audio_us, audio_count
                if not item.samples:
                    return
                if first_media_us is None:
                    if first_video_us is None:
                        return
                    first_media_us = first_video_us
                timestamp_us = item_timestamp_us(item)
                if timestamp_us < first_media_us:
                    return
                add_audio_stream(item.sample_rate_hz, item.channels)
                if item.sample_rate_hz != audio_rate or item.channels != audio_channels:
                    return
                sample_count = len(item.samples) // (2 * item.channels)
                if sample_count <= 0:
                    return
                rel_us = timestamp_us - first_media_us
                if audio_count and rel_us < last_audio_us:
                    rel_us = last_audio_us
                frame = av.AudioFrame(format="s16", layout=audio_layout_name(item.channels), samples=sample_count)
                frame.sample_rate = item.sample_rate_hz
                frame.pts = int(rel_us * item.sample_rate_hz / 1_000_000)
                frame.time_base = Fraction(1, item.sample_rate_hz)
                frame.planes[0].update(item.samples[: sample_count * item.channels * 2])
                for packet in audio_stream.encode(frame):
                    container.mux(packet)
                last_audio_us = rel_us + int(sample_count * 1_000_000 / item.sample_rate_hz)
                audio_count += 1

            def write_item(item: Union[CaptureVideoFrame, CaptureAudioFrame]) -> None:
                if isinstance(item, CaptureVideoFrame):
                    write_video(item)
                else:
                    write_audio(item)

            def start_pending_recording() -> None:
                nonlocal pending, started, first_media_us, first_video_us
                video_items = [pending_item for pending_item in pending if isinstance(pending_item, CaptureVideoFrame)]
                if not video_items:
                    started = True
                    for pending_item in pending:
                        write_item(pending_item)
                    pending.clear()
                    return

                start_video = video_items[0]
                if len(video_items) >= 2:
                    first_timestamp_us = item_timestamp_us(video_items[0])
                    second_timestamp_us = item_timestamp_us(video_items[1])
                    if second_timestamp_us - first_timestamp_us > RECORDING_START_GAP_TRIM_US:
                        start_video = video_items[1]

                start_timestamp_us = item_timestamp_us(start_video)
                first_video_us = start_timestamp_us
                first_media_us = start_timestamp_us
                add_video_stream(start_video)
                add_default_audio_stream()
                started = True
                for pending_item in pending:
                    if item_timestamp_us(pending_item) < start_timestamp_us:
                        continue
                    write_item(pending_item)
                pending.clear()

            while True:
                item = self.items.get()
                if item is None:
                    break
                if not started:
                    pending.append(item)
                    if isinstance(item, CaptureVideoFrame):
                        pending_video_count = sum(
                            1 for pending_item in pending if isinstance(pending_item, CaptureVideoFrame)
                        )
                        if pending_video_count >= 2:
                            start_pending_recording()
                    elif len(pending) > RECORDING_QUEUE_MAX_ITEMS // 2:
                        pending = pending[-(RECORDING_QUEUE_MAX_ITEMS // 2) :]
                    continue
                write_item(item)

            if not started and pending:
                start_pending_recording()

            if video_stream is not None:
                for packet in video_stream.encode(None):
                    container.mux(packet)
            if audio_stream is not None:
                for packet in audio_stream.encode(None):
                    container.mux(packet)
            duration_us = max(last_video_us, last_audio_us)
            duration_seconds = max(1, int((duration_us + 999_999) // 1_000_000))
            self.result = RecordingResult(self.path, duration_seconds, width, height, video_count > 0, audio_count > 0)
            container.close()
            container = None
            if not self.result.has_video and not self.result.has_audio:
                self.path.unlink(missing_ok=True)
                self.result = None
        except Exception as error:
            self.error = error
            if self.result is None and self.path.exists() and self.path.stat().st_size > 0:
                self.result = probe_media_file(self.path)
        finally:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass


def probe_media_file(path: Path) -> Optional[RecordingResult]:
    try:
        with av.open(str(path)) as container:
            width = 0
            height = 0
            duration_us = 0
            has_video = False
            has_audio = False
            for stream in container.streams:
                if stream.type == "video":
                    has_video = True
                    width = int(stream.width or width)
                    height = int(stream.height or height)
                elif stream.type == "audio":
                    has_audio = True
                if stream.duration is not None and stream.time_base is not None:
                    duration_us = max(duration_us, int(stream.duration * stream.time_base * 1_000_000))
            if not has_video and not has_audio:
                return None
            return RecordingResult(path, max(1, int((duration_us + 999_999) // 1_000_000)), width, height, has_video, has_audio)
    except Exception:
        return None


def record_device(
    video: CaptureDevice,
    audio: Optional[CaptureDevice],
    path: Path,
    duration_seconds: int,
    video_size: str,
    *,
    max_file_bytes: Optional[int] = None,
    external_stop: Optional[threading.Event] = None,
) -> Optional[RecordingResult]:
    recorder = DeviceRecorder(path, video_size)
    stop = threading.Event()
    record_duration_seconds = max(1, duration_seconds)
    recording_end_at: list[Optional[float]] = [None]
    recording_end_lock = threading.Lock()

    def recording_deadline() -> Optional[float]:
        with recording_end_lock:
            return recording_end_at[0]

    def mark_recording_started(monotonic_time: float) -> None:
        with recording_end_lock:
            if recording_end_at[0] is None:
                recording_end_at[0] = monotonic_time + record_duration_seconds

    def should_stop_now(monotonic_time: Optional[float] = None) -> bool:
        if stop.is_set():
            return True
        if external_stop is not None and external_stop.is_set():
            return True
        deadline = recording_deadline()
        if deadline is None:
            return False
        if monotonic_time is None:
            monotonic_time = time.monotonic()
        return monotonic_time >= deadline

    def video_worker() -> None:
        try:
            with open_capture(video) as container:
                stream = container.streams.video[0]
                frame_interval = 1.0 / DEFAULT_RECORD_FPS
                next_submit_at = time.monotonic()
                for frame in container.decode(stream):
                    now = time.monotonic()
                    if should_stop_now(now):
                        break
                    if now < next_submit_at:
                        continue
                    mark_recording_started(now)
                    timestamp_us = time.monotonic_ns() // 1_000
                    capture_frame = capture_video_frame_from_av(frame, timestamp_us, video_size)
                    if not recorder.submit(capture_frame):
                        break
                    next_submit_at = max(next_submit_at + frame_interval, now + frame_interval)
        except Exception as error:
            print(f"[watchdog] record video stopped: {error}\n{traceback.format_exc()}", flush=True)
        finally:
            stop.set()

    def audio_worker() -> None:
        if audio is None:
            return
        try:
            with open_capture(audio) as container:
                stream = container.streams.audio[0]
                resampler = av.audio.resampler.AudioResampler(format="s16", layout="stereo", rate=AUDIO_RATE)
                pending = bytearray()
                sample_cursor = 0
                base_us = time.monotonic_ns() // 1_000
                for frame in container.decode(stream):
                    if should_stop_now():
                        break
                    append_resampled_audio(pending, resampler.resample(frame))
                    while len(pending) >= AUDIO_CHUNK_BYTES:
                        chunk = bytes(pending[:AUDIO_CHUNK_BYTES])
                        del pending[:AUDIO_CHUNK_BYTES]
                        timestamp_us = base_us + int(sample_cursor * 1_000_000 / AUDIO_RATE)
                        if not recorder.submit(CaptureAudioFrame(chunk, AUDIO_RATE, AUDIO_CHANNELS, timestamp_us)):
                            return
                        sample_cursor += AUDIO_CHUNK_SAMPLES
        except Exception as error:
            print(f"[watchdog] record audio stopped: {error}", flush=True)

    def rtsp_worker() -> None:
        next_submit_at = time.monotonic()
        try:
            with open_capture(video) as container:
                video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
                audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
                if video_stream is None:
                    raise RuntimeError("RTSP input has no video stream")

                streams = [video_stream]
                resampler = None
                pending_audio = bytearray()
                sample_cursor = 0
                first_chunk_timestamp_us: Optional[int] = None
                if audio_stream is not None:
                    streams.append(audio_stream)
                    resampler = av.audio.resampler.AudioResampler(format="s16", layout="stereo", rate=AUDIO_RATE)

                frame_interval = 1.0 / DEFAULT_RECORD_FPS
                for frame in container.decode(*streams):
                    now = time.monotonic()
                    if should_stop_now(now):
                        break
                    if is_video_frame(frame):
                        if now < next_submit_at:
                            continue
                        mark_recording_started(now)
                        capture_frame = capture_video_frame_from_av(frame, time.monotonic_ns() // 1_000, video_size)
                        if not recorder.submit(capture_frame):
                            break
                        next_submit_at = max(next_submit_at + frame_interval, now + frame_interval)
                        continue
                    if resampler is not None and is_audio_frame(frame):
                        append_resampled_audio(pending_audio, resampler.resample(frame))
                        while len(pending_audio) >= AUDIO_CHUNK_BYTES:
                            chunk = bytes(pending_audio[:AUDIO_CHUNK_BYTES])
                            del pending_audio[:AUDIO_CHUNK_BYTES]
                            if first_chunk_timestamp_us is None:
                                first_chunk_timestamp_us = time.monotonic_ns() // 1_000
                            timestamp_us = first_chunk_timestamp_us + int(sample_cursor * 1_000_000 / AUDIO_RATE)
                            if not recorder.submit(CaptureAudioFrame(chunk, AUDIO_RATE, AUDIO_CHANNELS, timestamp_us)):
                                return
                            sample_cursor += AUDIO_CHUNK_SAMPLES
        except Exception as error:
            print(f"[watchdog] record RTSP stopped: {error}\n{traceback.format_exc()}", flush=True)
        finally:
            stop.set()

    threads = (
        [threading.Thread(target=rtsp_worker, name="watchdog-record-rtsp", daemon=True)]
        if is_rtsp_capture(video) and audio is None
        else [
            threading.Thread(target=video_worker, name="watchdog-record-video", daemon=True),
            threading.Thread(target=audio_worker, name="watchdog-record-audio", daemon=True),
        ]
    )
    for thread in threads:
        thread.start()
    while not stop.wait(0.1):
        if external_stop is not None and external_stop.is_set():
            break
        if max_file_bytes is not None and path.exists():
            try:
                if path.stat().st_size >= max_file_bytes:
                    break
            except OSError:
                pass
        deadline = recording_deadline()
        if deadline is not None and time.monotonic() >= deadline:
            break
    stop.set()
    for thread in threads:
        thread.join(timeout=2.0)
    return recorder.close()


def join_rtmp_url(url: str, stream_key: str) -> str:
    url = str(url or "").strip()
    stream_key = str(stream_key or "").strip()
    if not stream_key:
        return url
    separator = "" if url.endswith("/") else "/"
    return f"{url}{separator}{stream_key}"


def redact_rtmp_url(value: str) -> str:
    parts = urlsplit(str(value or ""))
    if not parts.scheme:
        return "<empty>"
    path = parts.path or ""
    if path:
        prefix, separator, _ = path.rpartition("/")
        path = f"{prefix}{separator}***" if separator else "***"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def combined_dshow_device(video_device: CaptureDevice, audio_device: CaptureDevice) -> CaptureDevice:
    return CaptureDevice(
        f"{video_device.label} + {audio_device.label}",
        f"{video_device.input_name}:{audio_device.input_name}",
        "dshow",
        dshow_low_latency_options(include_audio=True, include_video=True),
    )


def can_use_combined_dshow(video_device: CaptureDevice, audio_device: Optional[CaptureDevice]) -> bool:
    return (
        audio_device is not None
        and sys.platform == "win32"
        and video_device.format_name.lower() == "dshow"
        and audio_device.format_name.lower() == "dshow"
    )


def add_rtmp_video_stream(container, width: int, height: int, fps: int):
    fps = max(1, int(fps))
    gop = max(1, int(DEFAULT_RTMP_GOP))
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.time_base = Fraction(1, 1_000_000)
    options = {
        "preset": "veryfast",
        "tune": "zerolatency",
        "profile": "baseline",
        "level": "3.1",
        "x264-params": f"ref=1:cabac=0:keyint={gop}:min-keyint={gop}:scenecut=0",
    }
    for target in (stream, getattr(stream, "codec_context", None)):
        if target is None:
            continue
        try:
            target.options = options
        except Exception:
            pass
    codec_context = getattr(stream, "codec_context", None)
    if codec_context is not None:
        for name, value in (("bit_rate", 2_000_000), ("gop_size", gop), ("max_b_frames", 0)):
            try:
                setattr(codec_context, name, value)
            except Exception:
                pass
    return stream


def add_rtmp_audio_stream(container):
    stream = container.add_stream("aac", rate=AUDIO_RATE)
    stream.layout = audio_layout_name(AUDIO_CHANNELS) or "stereo"
    stream.time_base = Fraction(1, AUDIO_RATE)
    codec_context = getattr(stream, "codec_context", None)
    if codec_context is not None:
        try:
            codec_context.bit_rate = 128_000
        except Exception:
            pass
    return stream


def make_silent_audio_frame(sample_cursor: int, samples: int = 1024):
    frame = av.AudioFrame(format="s16", layout=audio_layout_name(AUDIO_CHANNELS), samples=samples)
    frame.sample_rate = AUDIO_RATE
    frame.pts = sample_cursor
    frame.time_base = Fraction(1, AUDIO_RATE)
    frame.planes[0].update(b"\x00" * samples * AUDIO_CHANNELS * 2)
    return frame


def stream_device_to_rtmp_transcode(
    video: CaptureDevice,
    audio: Optional[CaptureDevice],
    output_url: str,
    video_size: str,
    stop: threading.Event,
    *,
    fps: int = DEFAULT_RTMP_VIDEO_FPS,
    debug_log: Optional[Callable[[str], None]] = None,
    debug_context: Optional[dict[str, object]] = None,
) -> None:
    require_pyav()
    fps = max(1, int(fps))
    use_rtsp_embedded_audio = is_rtsp_capture(video) and audio is None
    debug_context = dict(debug_context or {})
    media_queue: "queue.Queue[Optional[Union[CaptureVideoFrame, CaptureAudioFrame]]]" = queue.Queue(
        maxsize=RECORDING_QUEUE_MAX_ITEMS
    )
    started_monotonic_us = time.monotonic_ns() // 1_000
    frame_interval = 1.0 / max(1, fps)
    queue_full_count = 0

    def log_debug(event: str, **fields: object) -> None:
        if debug_log is None:
            return
        values = dict(debug_context)
        values.update(fields)
        debug_log(debug_line(event, **values))

    log_debug(
        "rtmp_stream_opening",
        video_label=video.label,
        video_format=video.format_name,
        video_input=redact_url(video.input_name) if is_rtsp_capture(video) else video.input_name,
        audio_label=None if audio is None else audio.label,
        audio_format=None if audio is None else audio.format_name,
        output_url=redact_rtmp_url(output_url),
        video_size=video_size,
        fps=fps,
        queue_max=RECORDING_QUEUE_MAX_ITEMS,
        rtsp_embedded_audio=use_rtsp_embedded_audio,
    )

    def timestamp_now_us() -> int:
        return max(0, time.monotonic_ns() // 1_000 - started_monotonic_us)

    def submit(item: Union[CaptureVideoFrame, CaptureAudioFrame]) -> bool:
        nonlocal queue_full_count
        while not stop.is_set():
            try:
                media_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                queue_full_count += 1
                if queue_full_count <= 5 or queue_full_count % 10 == 0:
                    log_debug(
                        "rtmp_queue_full",
                        count=queue_full_count,
                        item_type=type(item).__name__,
                        queue_size=media_queue.qsize(),
                    )
                continue
        return False

    def submit_done() -> None:
        while True:
            try:
                media_queue.put(None, timeout=0.1)
                return
            except queue.Full:
                if stop.is_set():
                    continue

    def video_worker() -> None:
        next_frame_at = time.monotonic()
        decoded = 0
        submitted = 0
        skipped = 0
        skipped_since_submit = 0
        last_decode_monotonic: Optional[float] = None
        try:
            with open_capture(video) as container:
                stream = container.streams.video[0]
                log_debug("rtmp_input_video_stream", worker="video", **stream_debug_fields(stream))
                for frame in container.decode(stream):
                    if stop.is_set():
                        break
                    decoded += 1
                    now = time.monotonic()
                    decode_gap_ms = None
                    if last_decode_monotonic is not None:
                        decode_gap_ms = (now - last_decode_monotonic) * 1000.0
                        if decode_gap_ms > 500.0:
                            log_debug(
                                "rtmp_video_decode_gap",
                                worker="video",
                                decoded=decoded,
                                gap_ms=round(decode_gap_ms, 3),
                                source_pts_us=frame_pts_us(frame),
                            )
                    last_decode_monotonic = now
                    if now < next_frame_at:
                        skipped += 1
                        skipped_since_submit += 1
                        if skipped_since_submit == 1 or skipped_since_submit % 30 == 0:
                            log_debug(
                                "rtmp_video_skipped_fps_gate",
                                worker="video",
                                decoded=decoded,
                                skipped_total=skipped,
                                skipped_since_submit=skipped_since_submit,
                                wait_ms=round((next_frame_at - now) * 1000.0, 3),
                                source_pts_us=frame_pts_us(frame),
                            )
                        continue
                    capture_started = time.perf_counter()
                    capture_frame = capture_video_frame_from_av(frame, timestamp_now_us(), video_size)
                    capture_ms = (time.perf_counter() - capture_started) * 1000.0
                    if not submit(capture_frame):
                        break
                    submitted += 1
                    log_debug(
                        "rtmp_video_captured",
                        worker="video",
                        frame=submitted,
                        decoded=decoded,
                        source_pts_us=frame_pts_us(frame),
                        source_width=getattr(frame, "width", None),
                        source_height=getattr(frame, "height", None),
                        output_width=capture_frame.width,
                        output_height=capture_frame.height,
                        timestamp_us=capture_frame.timestamp_us,
                        decode_gap_ms=None if decode_gap_ms is None else round(decode_gap_ms, 3),
                        skipped_before=skipped_since_submit,
                        capture_ms=round(capture_ms, 3),
                        queue_size=media_queue.qsize(),
                    )
                    skipped_since_submit = 0
                    next_frame_at = max(next_frame_at + frame_interval, now + frame_interval)
        except Exception as error:
            log_debug("rtmp_video_capture_error", worker="video", error=error)
            print(f"[watchdog] RTMP video capture stopped: device={video.label}, error={error}", flush=True)
        finally:
            log_debug("rtmp_video_capture_finished", worker="video", decoded=decoded, submitted=submitted, skipped=skipped)
            submit_done()

    def audio_worker() -> None:
        if audio is None:
            return
        audio_chunks = 0
        try:
            with open_capture(audio) as container:
                stream = container.streams.audio[0]
                log_debug("rtmp_input_audio_stream", worker="audio", **stream_debug_fields(stream))
                resampler = av.audio.resampler.AudioResampler(format="s16", layout="stereo", rate=AUDIO_RATE)
                pending = bytearray()
                sample_cursor = 0
                first_chunk_timestamp_us: Optional[int] = None
                for frame in container.decode(stream):
                    if stop.is_set():
                        break
                    append_resampled_audio(pending, resampler.resample(frame))
                    while len(pending) >= AUDIO_CHUNK_BYTES:
                        chunk = bytes(pending[:AUDIO_CHUNK_BYTES])
                        del pending[:AUDIO_CHUNK_BYTES]
                        if first_chunk_timestamp_us is None:
                            first_chunk_timestamp_us = timestamp_now_us()
                        timestamp_us = first_chunk_timestamp_us + int(sample_cursor * 1_000_000 / AUDIO_RATE)
                        sample_cursor += AUDIO_CHUNK_SAMPLES
                        if not submit(CaptureAudioFrame(chunk, AUDIO_RATE, AUDIO_CHANNELS, timestamp_us)):
                            return
                        audio_chunks += 1
                        if audio_chunks <= 5 or audio_chunks % 100 == 0:
                            log_debug(
                                "rtmp_audio_captured",
                                worker="audio",
                                chunk=audio_chunks,
                                timestamp_us=timestamp_us,
                                chunk_bytes=len(chunk),
                                pending_audio_bytes=len(pending),
                                queue_size=media_queue.qsize(),
                            )
        except Exception as error:
            log_debug("rtmp_audio_capture_error", worker="audio", error=error)
            print(f"[watchdog] RTMP audio capture stopped: device={audio.label}, error={error}", flush=True)
        finally:
            log_debug("rtmp_audio_capture_finished", worker="audio", chunks=audio_chunks)
            submit_done()

    def combined_worker() -> None:
        assert audio is not None
        device = combined_dshow_device(video, audio)
        next_frame_at = time.monotonic()
        video_decoded = 0
        video_submitted = 0
        video_skipped = 0
        audio_chunks = 0
        skipped_since_submit = 0
        last_video_decode_monotonic: Optional[float] = None
        try:
            with open_capture(device) as container:
                video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
                audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
                if video_stream is None:
                    raise RuntimeError("combined DirectShow input has no video stream")
                if audio_stream is None:
                    raise RuntimeError("combined DirectShow input has no audio stream")
                log_debug("rtmp_input_video_stream", worker="combined", **stream_debug_fields(video_stream))
                log_debug("rtmp_input_audio_stream", worker="combined", **stream_debug_fields(audio_stream))
                resampler = av.audio.resampler.AudioResampler(format="s16", layout="stereo", rate=AUDIO_RATE)
                pending_audio = bytearray()
                sample_cursor = 0
                first_chunk_timestamp_us: Optional[int] = None
                for frame in container.decode(video_stream, audio_stream):
                    if stop.is_set():
                        break
                    if is_video_frame(frame):
                        video_decoded += 1
                        now = time.monotonic()
                        decode_gap_ms = None
                        if last_video_decode_monotonic is not None:
                            decode_gap_ms = (now - last_video_decode_monotonic) * 1000.0
                            if decode_gap_ms > 500.0:
                                log_debug(
                                    "rtmp_video_decode_gap",
                                    worker="combined",
                                    decoded=video_decoded,
                                    gap_ms=round(decode_gap_ms, 3),
                                    source_pts_us=frame_pts_us(frame),
                                )
                        last_video_decode_monotonic = now
                        if now >= next_frame_at:
                            capture_started = time.perf_counter()
                            capture_frame = capture_video_frame_from_av(frame, timestamp_now_us(), video_size)
                            capture_ms = (time.perf_counter() - capture_started) * 1000.0
                            if not submit(capture_frame):
                                break
                            video_submitted += 1
                            log_debug(
                                "rtmp_video_captured",
                                worker="combined",
                                frame=video_submitted,
                                decoded=video_decoded,
                                source_pts_us=frame_pts_us(frame),
                                source_width=getattr(frame, "width", None),
                                source_height=getattr(frame, "height", None),
                                output_width=capture_frame.width,
                                output_height=capture_frame.height,
                                timestamp_us=capture_frame.timestamp_us,
                                decode_gap_ms=None if decode_gap_ms is None else round(decode_gap_ms, 3),
                                skipped_before=skipped_since_submit,
                                capture_ms=round(capture_ms, 3),
                                queue_size=media_queue.qsize(),
                            )
                            skipped_since_submit = 0
                            next_frame_at = max(next_frame_at + frame_interval, now + frame_interval)
                        else:
                            video_skipped += 1
                            skipped_since_submit += 1
                            if skipped_since_submit == 1 or skipped_since_submit % 30 == 0:
                                log_debug(
                                    "rtmp_video_skipped_fps_gate",
                                    worker="combined",
                                    decoded=video_decoded,
                                    skipped_total=video_skipped,
                                    skipped_since_submit=skipped_since_submit,
                                    wait_ms=round((next_frame_at - now) * 1000.0, 3),
                                    source_pts_us=frame_pts_us(frame),
                                )
                        continue
                    if is_audio_frame(frame):
                        append_resampled_audio(pending_audio, resampler.resample(frame))
                        while len(pending_audio) >= AUDIO_CHUNK_BYTES:
                            chunk = bytes(pending_audio[:AUDIO_CHUNK_BYTES])
                            del pending_audio[:AUDIO_CHUNK_BYTES]
                            if first_chunk_timestamp_us is None:
                                first_chunk_timestamp_us = timestamp_now_us()
                            timestamp_us = first_chunk_timestamp_us + int(sample_cursor * 1_000_000 / AUDIO_RATE)
                            sample_cursor += AUDIO_CHUNK_SAMPLES
                            if not submit(CaptureAudioFrame(chunk, AUDIO_RATE, AUDIO_CHANNELS, timestamp_us)):
                                return
                            audio_chunks += 1
                            if audio_chunks <= 5 or audio_chunks % 100 == 0:
                                log_debug(
                                    "rtmp_audio_captured",
                                    worker="combined",
                                    chunk=audio_chunks,
                                    source_pts_us=frame_pts_us(frame),
                                    timestamp_us=timestamp_us,
                                    chunk_bytes=len(chunk),
                                    pending_audio_bytes=len(pending_audio),
                                    queue_size=media_queue.qsize(),
                                )
        except Exception as error:
            log_debug("rtmp_combined_capture_error", worker="combined", error=error)
            print(f"[watchdog] RTMP combined capture stopped: device={device.label}, error={error}", flush=True)
        finally:
            log_debug(
                "rtmp_combined_capture_finished",
                worker="combined",
                decoded=video_decoded,
                submitted=video_submitted,
                skipped=video_skipped,
                audio_chunks=audio_chunks,
            )
            submit_done()

    def rtsp_worker() -> None:
        nonlocal silent_audio_required
        next_frame_at = time.monotonic()
        video_decoded = 0
        video_submitted = 0
        video_skipped = 0
        audio_chunks = 0
        skipped_since_submit = 0
        last_video_decode_monotonic: Optional[float] = None
        log_debug(
            "rtmp_rtsp_opening",
            rtsp_url=redact_url(video.input_name),
            options=json.dumps(video.options, ensure_ascii=False, sort_keys=True),
        )
        opened_started = time.perf_counter()
        try:
            with open_capture(video) as container:
                log_debug("rtmp_rtsp_opened", open_ms=round((time.perf_counter() - opened_started) * 1000.0, 3))
                for stream in container.streams:
                    log_debug("rtmp_rtsp_stream", **stream_debug_fields(stream))
                video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
                audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
                if video_stream is None:
                    raise RuntimeError("RTSP input has no video stream")

                log_debug("rtmp_input_video_stream", worker="rtsp", **stream_debug_fields(video_stream))
                streams = [video_stream]
                resampler = None
                pending_audio = bytearray()
                sample_cursor = 0
                first_chunk_timestamp_us: Optional[int] = None
                if audio_stream is not None:
                    log_debug("rtmp_input_audio_stream", worker="rtsp", **stream_debug_fields(audio_stream))
                    streams.append(audio_stream)
                    resampler = av.audio.resampler.AudioResampler(format="s16", layout="stereo", rate=AUDIO_RATE)
                else:
                    silent_audio_required = True
                    log_debug("rtmp_input_audio_missing", worker="rtsp")

                for frame in container.decode(*streams):
                    if stop.is_set():
                        break
                    if is_video_frame(frame):
                        video_decoded += 1
                        now = time.monotonic()
                        decode_gap_ms = None
                        if last_video_decode_monotonic is not None:
                            decode_gap_ms = (now - last_video_decode_monotonic) * 1000.0
                            if decode_gap_ms > 500.0:
                                log_debug(
                                    "rtmp_video_decode_gap",
                                    worker="rtsp",
                                    decoded=video_decoded,
                                    gap_ms=round(decode_gap_ms, 3),
                                    source_pts_us=frame_pts_us(frame),
                                )
                        last_video_decode_monotonic = now
                        if now >= next_frame_at:
                            capture_started = time.perf_counter()
                            capture_frame = capture_video_frame_from_av(frame, timestamp_now_us(), video_size)
                            capture_ms = (time.perf_counter() - capture_started) * 1000.0
                            if not submit(capture_frame):
                                break
                            video_submitted += 1
                            log_debug(
                                "rtmp_video_captured",
                                worker="rtsp",
                                frame=video_submitted,
                                decoded=video_decoded,
                                source_pts_us=frame_pts_us(frame),
                                source_width=getattr(frame, "width", None),
                                source_height=getattr(frame, "height", None),
                                output_width=capture_frame.width,
                                output_height=capture_frame.height,
                                timestamp_us=capture_frame.timestamp_us,
                                decode_gap_ms=None if decode_gap_ms is None else round(decode_gap_ms, 3),
                                skipped_before=skipped_since_submit,
                                capture_ms=round(capture_ms, 3),
                                queue_size=media_queue.qsize(),
                            )
                            skipped_since_submit = 0
                            next_frame_at = max(next_frame_at + frame_interval, now + frame_interval)
                        else:
                            video_skipped += 1
                            skipped_since_submit += 1
                            if skipped_since_submit == 1 or skipped_since_submit % 30 == 0:
                                log_debug(
                                    "rtmp_video_skipped_fps_gate",
                                    worker="rtsp",
                                    decoded=video_decoded,
                                    skipped_total=video_skipped,
                                    skipped_since_submit=skipped_since_submit,
                                    wait_ms=round((next_frame_at - now) * 1000.0, 3),
                                    source_pts_us=frame_pts_us(frame),
                                )
                        continue
                    if resampler is not None and is_audio_frame(frame):
                        audio_pts_us = frame_pts_us(frame)
                        append_resampled_audio(pending_audio, resampler.resample(frame))
                        while len(pending_audio) >= AUDIO_CHUNK_BYTES:
                            chunk = bytes(pending_audio[:AUDIO_CHUNK_BYTES])
                            del pending_audio[:AUDIO_CHUNK_BYTES]
                            if first_chunk_timestamp_us is None:
                                first_chunk_timestamp_us = timestamp_now_us()
                            timestamp_us = first_chunk_timestamp_us + int(sample_cursor * 1_000_000 / AUDIO_RATE)
                            sample_cursor += AUDIO_CHUNK_SAMPLES
                            if not submit(CaptureAudioFrame(chunk, AUDIO_RATE, AUDIO_CHANNELS, timestamp_us)):
                                return
                            audio_chunks += 1
                            if audio_chunks <= 5 or audio_chunks % 100 == 0:
                                log_debug(
                                    "rtmp_audio_captured",
                                    worker="rtsp",
                                    chunk=audio_chunks,
                                    source_pts_us=audio_pts_us,
                                    timestamp_us=timestamp_us,
                                    chunk_bytes=len(chunk),
                                    pending_audio_bytes=len(pending_audio),
                                    queue_size=media_queue.qsize(),
                                )
        except Exception as error:
            log_debug("rtmp_rtsp_capture_error", worker="rtsp", error=error)
            print(f"[watchdog] RTMP RTSP capture stopped: device={video.label}, error={error}", flush=True)
        finally:
            log_debug(
                "rtmp_rtsp_capture_finished",
                worker="rtsp",
                decoded=video_decoded,
                submitted=video_submitted,
                skipped=video_skipped,
                audio_chunks=audio_chunks,
            )
            submit_done()

    container = None
    video_stream = None
    audio_stream = None
    first_video_us: Optional[int] = None
    first_media_us: Optional[int] = None
    last_video_us = 0
    last_audio_us = 0
    video_count = 0
    audio_count = 0
    video_packet_count = 0
    audio_packet_count = 0
    started = False
    pending: list[Union[CaptureVideoFrame, CaptureAudioFrame]] = []
    silent_audio_required = False
    silent_audio_sample_cursor = 0

    def open_output(first_video: CaptureVideoFrame) -> None:
        nonlocal container, video_stream, audio_stream
        if container is not None:
            return
        output_started = time.perf_counter()
        log_debug(
            "rtmp_output_opening",
            output_url=redact_rtmp_url(output_url),
            first_video_width=first_video.width,
            first_video_height=first_video.height,
            fps=fps,
            audio_enabled=audio is not None or use_rtsp_embedded_audio,
        )
        container = av.open(output_url, "w", format="flv")
        video_stream = add_rtmp_video_stream(container, first_video.width, first_video.height, fps)
        if audio is not None or use_rtsp_embedded_audio:
            audio_stream = add_rtmp_audio_stream(container)
        log_debug(
            "rtmp_output_opened",
            open_ms=round((time.perf_counter() - output_started) * 1000.0, 3),
            video_width=first_video.width,
            video_height=first_video.height,
            fps=fps,
            audio_enabled=audio_stream is not None,
        )

    def write_video(item: CaptureVideoFrame) -> None:
        nonlocal first_video_us, first_media_us, last_video_us, video_count, video_packet_count
        if first_video_us is None:
            first_video_us = max(0, int(item.timestamp_us))
        if first_media_us is None:
            first_media_us = first_video_us
        open_output(item)
        write_started = time.perf_counter()
        convert_started = time.perf_counter()
        frame = av_frame_from_capture_video(item)
        convert_ms = (time.perf_counter() - convert_started) * 1000.0
        rel_us = max(0, int(item.timestamp_us) - first_video_us)
        if video_count and rel_us <= last_video_us:
            rel_us = last_video_us + int(1_000_000 / fps)
        frame.pts = rel_us
        frame.time_base = Fraction(1, 1_000_000)
        encode_started = time.perf_counter()
        packets = 0
        for packet in video_stream.encode(frame):
            container.mux(packet)
            packets += 1
        encode_mux_ms = (time.perf_counter() - encode_started) * 1000.0
        video_packet_count += packets
        write_silent_audio_until(rel_us)
        last_video_us = rel_us
        video_count += 1
        log_debug(
            "rtmp_video_written",
            frame=video_count,
            timestamp_us=item.timestamp_us,
            rel_us=rel_us,
            width=item.width,
            height=item.height,
            convert_ms=round(convert_ms, 3),
            encode_mux_ms=round(encode_mux_ms, 3),
            total_ms=round((time.perf_counter() - write_started) * 1000.0, 3),
            packets=packets,
            total_packets=video_packet_count,
            queue_size=media_queue.qsize(),
        )

    def write_audio(item: CaptureAudioFrame) -> None:
        nonlocal first_media_us, last_audio_us, audio_count, audio_packet_count
        if audio_stream is None or first_media_us is None or not item.samples:
            return
        timestamp_us = max(0, int(item.timestamp_us))
        if timestamp_us < first_media_us:
            log_debug(
                "rtmp_audio_dropped_before_start",
                timestamp_us=timestamp_us,
                first_media_us=first_media_us,
                bytes=len(item.samples),
            )
            return
        sample_count = len(item.samples) // (2 * item.channels)
        if sample_count <= 0:
            return
        rel_us = timestamp_us - first_media_us
        if audio_count and rel_us < last_audio_us:
            rel_us = last_audio_us
        write_started = time.perf_counter()
        frame = av.AudioFrame(format="s16", layout=audio_layout_name(item.channels), samples=sample_count)
        frame.sample_rate = item.sample_rate_hz
        frame.pts = int(rel_us * item.sample_rate_hz / 1_000_000)
        frame.time_base = Fraction(1, item.sample_rate_hz)
        frame.planes[0].update(item.samples[: sample_count * item.channels * 2])
        packets = 0
        for packet in audio_stream.encode(frame):
            container.mux(packet)
            packets += 1
        audio_packet_count += packets
        last_audio_us = rel_us + int(sample_count * 1_000_000 / item.sample_rate_hz)
        audio_count += 1
        if audio_count <= 5 or audio_count % 100 == 0:
            log_debug(
                "rtmp_audio_written",
                chunk=audio_count,
                timestamp_us=timestamp_us,
                rel_us=rel_us,
                sample_rate=item.sample_rate_hz,
                channels=item.channels,
                samples=sample_count,
                encode_mux_ms=round((time.perf_counter() - write_started) * 1000.0, 3),
                packets=packets,
                total_packets=audio_packet_count,
                queue_size=media_queue.qsize(),
            )

    def write_silent_audio_until(target_rel_us: int) -> None:
        nonlocal silent_audio_sample_cursor, last_audio_us, audio_count, audio_packet_count
        if not silent_audio_required or audio_stream is None or first_media_us is None:
            return
        target_samples = int(max(0, target_rel_us) * AUDIO_RATE / 1_000_000)
        while silent_audio_sample_cursor <= target_samples:
            write_started = time.perf_counter()
            frame = make_silent_audio_frame(silent_audio_sample_cursor)
            packets = 0
            for packet in audio_stream.encode(frame):
                container.mux(packet)
                packets += 1
            timestamp_us = int(silent_audio_sample_cursor * 1_000_000 / AUDIO_RATE)
            silent_audio_sample_cursor += frame.samples
            audio_packet_count += packets
            last_audio_us = timestamp_us
            audio_count += 1
            if audio_count <= 5 or audio_count % 100 == 0:
                log_debug(
                    "rtmp_silent_audio_written",
                    chunk=audio_count,
                    rel_us=timestamp_us,
                    samples=frame.samples,
                    packets=packets,
                    total_packets=audio_packet_count,
                    encode_mux_ms=round((time.perf_counter() - write_started) * 1000.0, 3),
                    queue_size=media_queue.qsize(),
                )

    def write_item(item: Union[CaptureVideoFrame, CaptureAudioFrame]) -> None:
        if isinstance(item, CaptureVideoFrame):
            write_video(item)
        else:
            write_audio(item)

    def start_pending_stream() -> None:
        nonlocal pending, started, first_video_us, first_media_us
        video_items = [pending_item for pending_item in pending if isinstance(pending_item, CaptureVideoFrame)]
        if not video_items:
            return
        log_debug(
            "rtmp_pending_start",
            pending_items=len(pending),
            pending_video=len(video_items),
            pending_audio=sum(1 for pending_item in pending if isinstance(pending_item, CaptureAudioFrame)),
        )
        start_video = video_items[0]
        if len(video_items) >= 2:
            first_timestamp_us = max(0, int(video_items[0].timestamp_us))
            second_timestamp_us = max(0, int(video_items[1].timestamp_us))
            if second_timestamp_us - first_timestamp_us > RECORDING_START_GAP_TRIM_US:
                start_video = video_items[1]
        start_timestamp_us = max(0, int(start_video.timestamp_us))
        first_video_us = start_timestamp_us
        first_media_us = start_timestamp_us
        open_output(start_video)
        started = True
        log_debug(
            "rtmp_stream_started",
            start_timestamp_us=start_timestamp_us,
            first_video_us=first_video_us,
            first_media_us=first_media_us,
        )
        for pending_item in pending:
            if max(0, int(pending_item.timestamp_us)) < start_timestamp_us:
                log_debug("rtmp_pending_dropped_before_start", item_type=type(pending_item).__name__, timestamp_us=pending_item.timestamp_us)
                continue
            write_item(pending_item)
        pending.clear()

    use_combined = can_use_combined_dshow(video, audio)
    if use_combined:
        log_debug("rtmp_worker_mode", mode="combined_dshow")
        workers = [threading.Thread(target=combined_worker, name="watchdog-rtmp-combined", daemon=True)]
    elif use_rtsp_embedded_audio:
        log_debug("rtmp_worker_mode", mode="rtsp_embedded_audio")
        workers = [threading.Thread(target=rtsp_worker, name="watchdog-rtmp-rtsp", daemon=True)]
    else:
        log_debug("rtmp_worker_mode", mode="separate_video_audio", has_audio=audio is not None)
        workers = [threading.Thread(target=video_worker, name="watchdog-rtmp-video", daemon=True)]
    if not use_combined and not use_rtsp_embedded_audio and audio is not None:
        workers.append(threading.Thread(target=audio_worker, name="watchdog-rtmp-audio", daemon=True))

    done_workers = 0
    try:
        for worker in workers:
            worker.start()
            log_debug("rtmp_worker_started", worker_name=worker.name)
        while done_workers < len(workers):
            try:
                item = media_queue.get(timeout=0.1)
            except queue.Empty:
                if stop.is_set() and all(not worker.is_alive() for worker in workers):
                    break
                continue
            if item is None:
                done_workers += 1
                log_debug("rtmp_worker_done_marker", done_workers=done_workers, total_workers=len(workers), queue_size=media_queue.qsize())
                continue
            if not started:
                pending.append(item)
                if isinstance(item, CaptureVideoFrame):
                    pending_video_count = sum(
                        1 for pending_item in pending if isinstance(pending_item, CaptureVideoFrame)
                    )
                    if pending_video_count >= 2:
                        start_pending_stream()
                elif len(pending) > RECORDING_QUEUE_MAX_ITEMS // 2:
                    pending = pending[-(RECORDING_QUEUE_MAX_ITEMS // 2) :]
                continue
            write_item(item)

        if not started and pending:
            start_pending_stream()
        if video_stream is not None:
            flush_started = time.perf_counter()
            packets = 0
            for packet in video_stream.encode(None):
                container.mux(packet)
                packets += 1
            log_debug("rtmp_video_encoder_flushed", packets=packets, flush_ms=round((time.perf_counter() - flush_started) * 1000.0, 3))
        if audio_stream is not None:
            flush_started = time.perf_counter()
            packets = 0
            for packet in audio_stream.encode(None):
                container.mux(packet)
                packets += 1
            log_debug("rtmp_audio_encoder_flushed", packets=packets, flush_ms=round((time.perf_counter() - flush_started) * 1000.0, 3))
    except Exception as error:
        log_debug("rtmp_stream_error", error=error, traceback=traceback.format_exc())
        print(f"[watchdog] RTMP stream stopped: {error}", flush=True)
    finally:
        stop.set()
        for worker in workers:
            worker.join(timeout=2.0)
            log_debug("rtmp_worker_joined", worker_name=worker.name, alive=worker.is_alive())
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
        log_debug(
            "rtmp_stream_finished",
            started=started,
            video_frames=video_count,
            audio_chunks=audio_count,
            video_packets=video_packet_count,
            audio_packets=audio_packet_count,
            last_video_us=last_video_us,
            last_audio_us=last_audio_us,
            queue_full_count=queue_full_count,
        )


def stream_codec_name(stream) -> str:
    codec_context = getattr(stream, "codec_context", None)
    return str(getattr(codec_context, "name", "") or "").casefold()


def can_remux_video_to_rtmp(stream) -> bool:
    return stream_codec_name(stream) in {"h264", "hevc", "h265"}


def can_direct_remux_video_to_telegram_rtmp(stream) -> tuple[bool, str]:
    codec = stream_codec_name(stream)
    if codec not in {"h264", "hevc", "h265"}:
        return False, "unsupported_video_codec"
    width = int(getattr(stream, "width", 0) or 0)
    height = int(getattr(stream, "height", 0) or 0)
    if width <= 0 or height <= 0:
        return False, "unknown_video_size"
    # Telegram RTMP accepts an input stream only after its own transcoder accepts it.
    # Larger camera streams can be valid H.264 but still fail to appear in Telegram.
    if max(width, height) > 1280 or min(width, height) > 720:
        return False, "video_larger_than_720p"
    return True, "ok"


def can_remux_audio_to_rtmp(stream) -> bool:
    return stream_codec_name(stream) in {"aac", "mp3"}


def add_stream_from_template(container, template_stream, *, opaque: bool = True):
    add_from_template = getattr(container, "add_stream_from_template", None)
    if callable(add_from_template):
        try:
            return add_from_template(template_stream, opaque=opaque)
        except TypeError:
            return add_from_template(template_stream)

    add_stream = getattr(container, "add_stream", None)
    if not callable(add_stream):
        raise RuntimeError("PyAV output container does not support adding streams")
    try:
        return add_stream(template=template_stream)
    except TypeError:
        try:
            return add_stream(template_stream)
        except TypeError as fallback_error:
            raise RuntimeError("PyAV version does not support stream-template remuxing") from fallback_error


def packet_pts_us(packet) -> Optional[int]:
    timestamp = packet.pts if packet.pts is not None else packet.dts
    time_base = getattr(packet, "time_base", None)
    if timestamp is None or time_base is None:
        return None
    try:
        return int(timestamp * time_base * 1_000_000)
    except Exception:
        return None


def annexb_start_code_length(data: bytes, offset: int) -> int:
    if data.startswith(b"\x00\x00\x01", offset):
        return 3
    if data.startswith(b"\x00\x00\x00\x01", offset):
        return 4
    return 0


def iter_annexb_nal_units(data: bytes):
    offset = 0
    size = len(data)
    while offset < size:
        start = -1
        start_code_len = 0
        scan = offset
        while scan < size - 3:
            start_code_len = annexb_start_code_length(data, scan)
            if start_code_len:
                start = scan + start_code_len
                break
            scan += 1
        if start < 0:
            return
        next_scan = start
        next_start = size
        while next_scan < size - 3:
            if annexb_start_code_length(data, next_scan):
                next_start = next_scan
                break
            next_scan += 1
        if next_start > start:
            yield data[start:next_start]
        offset = next_start


def iter_length_prefixed_nal_units(data: bytes, length_size: int = 4):
    offset = 0
    size = len(data)
    while offset + length_size <= size:
        nal_size = int.from_bytes(data[offset : offset + length_size], "big")
        offset += length_size
        if nal_size <= 0 or offset + nal_size > size:
            return
        yield data[offset : offset + nal_size]
        offset += nal_size


def iter_packet_nal_units(data: bytes):
    if b"\x00\x00\x01" in data or b"\x00\x00\x00\x01" in data:
        yield from iter_annexb_nal_units(data)
        return
    for length_size in (4, 2, 1):
        units = list(iter_length_prefixed_nal_units(data, length_size))
        if units and sum(len(unit) + length_size for unit in units) == len(data):
            yield from units
            return


def compressed_packet_is_keyframe(packet, codec_name: str) -> bool:
    data = bytes(packet)
    codec = codec_name.casefold()
    try:
        for nal in iter_packet_nal_units(data):
            if not nal:
                continue
            if codec == "h264":
                if nal[0] & 0x1F == 5:
                    return True
            elif codec in {"hevc", "h265"}:
                nal_type = (nal[0] >> 1) & 0x3F
                if 16 <= nal_type <= 21:
                    return True
    except Exception:
        return False
    return False


def rebase_packet_timestamps(packet, bases: dict[int, int]) -> bool:
    stream = packet.stream
    if stream is None:
        return False
    stream_index = int(stream.index)
    reference = packet.dts if packet.dts is not None else packet.pts
    if reference is None:
        return False
    if stream_index not in bases:
        bases[stream_index] = int(reference)
    base = bases[stream_index]
    if packet.pts is not None:
        packet.pts = max(0, int(packet.pts) - base)
    if packet.dts is not None:
        packet.dts = max(0, int(packet.dts) - base)
    return True


def stream_rtsp_to_rtmp_remux(
    video: CaptureDevice,
    output_url: str,
    stop: threading.Event,
    *,
    debug_log: Optional[Callable[[str], None]] = None,
    debug_context: Optional[dict[str, object]] = None,
) -> bool:
    require_pyav()
    debug_context = dict(debug_context or {})

    def log_debug(event: str, **fields: object) -> None:
        if debug_log is None:
            return
        values = dict(debug_context)
        values.update(fields)
        debug_log(debug_line(event, **values))

    log_debug(
        "rtmp_remux_opening",
        rtsp_url=redact_url(video.input_name),
        output_url=redact_rtmp_url(output_url),
        options=json.dumps(video.options, ensure_ascii=False, sort_keys=True),
    )
    opened_started = time.perf_counter()
    input_container = None
    output_container = None
    video_packets = 0
    audio_packets = 0
    dropped_packets = 0
    mux_spikes = 0
    mux_max_ms = 0.0
    first_video_pts_us: Optional[int] = None
    last_video_pts_us: Optional[int] = None
    first_audio_pts_us: Optional[int] = None
    last_audio_pts_us: Optional[int] = None
    skipped_until_keyframe = 0
    detected_keyframes = 0
    try:
        input_container = open_capture(video)
        input_container = input_container.__enter__()
        log_debug("rtmp_remux_rtsp_opened", open_ms=round((time.perf_counter() - opened_started) * 1000.0, 3))
        for stream in input_container.streams:
            log_debug("rtmp_remux_input_stream", **stream_debug_fields(stream))

        input_video = next((stream for stream in input_container.streams if stream.type == "video"), None)
        input_audio = next((stream for stream in input_container.streams if stream.type == "audio"), None)
        if input_video is None:
            raise RuntimeError("RTSP input has no video stream")
        if not can_remux_video_to_rtmp(input_video):
            log_debug("rtmp_remux_unsupported_video", **stream_debug_fields(input_video))
            return False
        compatible, incompatible_reason = can_direct_remux_video_to_telegram_rtmp(input_video)
        if not compatible:
            log_debug("rtmp_remux_incompatible_for_telegram", reason=incompatible_reason, **stream_debug_fields(input_video))
            return False

        output_started = time.perf_counter()
        output_container = av.open(output_url, "w", format="flv")
        output_video = add_stream_from_template(output_container, input_video, opaque=True)
        output_streams_by_index = {int(input_video.index): output_video}
        demux_streams = [input_video]
        log_debug("rtmp_remux_video_enabled", input_codec=stream_codec_name(input_video), **stream_debug_fields(input_video))

        silent_audio_stream = None
        silent_audio_sample_cursor = 0
        first_video_source_pts_us: Optional[int] = None
        if input_audio is not None and can_remux_audio_to_rtmp(input_audio):
            output_audio = add_stream_from_template(output_container, input_audio, opaque=True)
            output_streams_by_index[int(input_audio.index)] = output_audio
            demux_streams.append(input_audio)
            log_debug("rtmp_remux_audio_enabled", input_codec=stream_codec_name(input_audio), **stream_debug_fields(input_audio))
        elif input_audio is not None:
            log_debug("rtmp_remux_audio_skipped", reason="unsupported_audio_codec", **stream_debug_fields(input_audio))
            silent_audio_stream = add_rtmp_audio_stream(output_container)
            log_debug("rtmp_remux_silent_audio_enabled", reason="unsupported_audio_codec", sample_rate=AUDIO_RATE, channels=AUDIO_CHANNELS)
        else:
            log_debug("rtmp_remux_audio_missing")
            silent_audio_stream = add_rtmp_audio_stream(output_container)
            log_debug("rtmp_remux_silent_audio_enabled", reason="missing_audio_stream", sample_rate=AUDIO_RATE, channels=AUDIO_CHANNELS)

        log_debug(
            "rtmp_remux_output_opened",
            open_ms=round((time.perf_counter() - output_started) * 1000.0, 3),
            output_url=redact_rtmp_url(output_url),
        )

        bases: dict[int, int] = {}
        video_started = False
        video_codec = stream_codec_name(input_video)

        def mux_silent_audio_until(target_rel_us: int) -> None:
            nonlocal audio_packets, first_audio_pts_us, last_audio_pts_us, silent_audio_sample_cursor
            if silent_audio_stream is None or target_rel_us < 0:
                return
            target_samples = int(target_rel_us * AUDIO_RATE / 1_000_000)
            while silent_audio_sample_cursor <= target_samples:
                frame = make_silent_audio_frame(silent_audio_sample_cursor)
                for audio_packet in silent_audio_stream.encode(frame):
                    output_container.mux(audio_packet)
                    audio_packets += 1
                timestamp_us = int(silent_audio_sample_cursor * 1_000_000 / AUDIO_RATE)
                first_audio_pts_us = timestamp_us if first_audio_pts_us is None else first_audio_pts_us
                last_audio_pts_us = timestamp_us
                silent_audio_sample_cursor += frame.samples

        for packet in input_container.demux(*demux_streams):
            if stop.is_set():
                break
            if packet.stream is None or int(packet.stream.index) not in output_streams_by_index:
                continue
            if packet.dts is None and packet.pts is None:
                dropped_packets += 1
                continue

            input_stream_index = int(packet.stream.index)
            source_pts_us = packet_pts_us(packet)
            is_video_packet = input_stream_index == int(input_video.index)
            if is_video_packet:
                detected_keyframe = bool(packet.is_keyframe) or compressed_packet_is_keyframe(packet, video_codec)
                if detected_keyframe:
                    detected_keyframes += 1
                    try:
                        packet.is_keyframe = True
                    except Exception:
                        pass
                if not video_started:
                    if not detected_keyframe:
                        skipped_until_keyframe += 1
                        if skipped_until_keyframe <= 5 or skipped_until_keyframe % 100 == 0:
                            log_debug(
                                "rtmp_remux_skip_until_keyframe",
                                skipped=skipped_until_keyframe,
                                source_pts_us=source_pts_us,
                                packet_size=packet.size,
                                pyav_keyframe=packet.is_keyframe,
                            )
                        continue
                    video_started = True
                    log_debug(
                        "rtmp_remux_first_keyframe",
                        skipped_until_keyframe=skipped_until_keyframe,
                        source_pts_us=source_pts_us,
                        packet_size=packet.size,
                        pyav_keyframe=packet.is_keyframe,
                    )
                    first_video_source_pts_us = source_pts_us
                    mux_silent_audio_until(0)
            elif not video_started:
                dropped_packets += 1
                continue
            if is_video_packet and source_pts_us is not None:
                if first_video_source_pts_us is None:
                    first_video_source_pts_us = source_pts_us
                mux_silent_audio_until(max(0, source_pts_us - first_video_source_pts_us))
            if not rebase_packet_timestamps(packet, bases):
                dropped_packets += 1
                continue
            packet.stream = output_streams_by_index[input_stream_index]

            mux_started = time.perf_counter()
            output_container.mux(packet)
            mux_ms = (time.perf_counter() - mux_started) * 1000.0
            mux_max_ms = max(mux_max_ms, mux_ms)
            if mux_ms > 20.0:
                mux_spikes += 1
                log_debug(
                    "rtmp_remux_mux_spike",
                    stream_index=input_stream_index,
                    stream_type=getattr(packet.stream, "type", None),
                    mux_ms=round(mux_ms, 3),
                    packet_size=getattr(packet, "size", None),
                    source_pts_us=source_pts_us,
                    pts=packet.pts,
                    dts=packet.dts,
                    keyframe=packet.is_keyframe,
                    video_packets=video_packets,
                    audio_packets=audio_packets,
                )

            if input_stream_index == int(input_video.index):
                video_packets += 1
                if source_pts_us is not None:
                    first_video_pts_us = source_pts_us if first_video_pts_us is None else first_video_pts_us
                    last_video_pts_us = source_pts_us
                if video_packets <= 5 or video_packets % 100 == 0:
                    log_debug(
                        "rtmp_remux_video_packet",
                        packet=video_packets,
                        source_pts_us=source_pts_us,
                        pts=packet.pts,
                        dts=packet.dts,
                        duration=packet.duration,
                        size=packet.size,
                        keyframe=packet.is_keyframe,
                        detected_keyframes=detected_keyframes,
                        mux_ms=round(mux_ms, 3),
                    )
            else:
                audio_packets += 1
                if source_pts_us is not None:
                    first_audio_pts_us = source_pts_us if first_audio_pts_us is None else first_audio_pts_us
                    last_audio_pts_us = source_pts_us
                if audio_packets <= 5 or audio_packets % 100 == 0:
                    log_debug(
                        "rtmp_remux_audio_packet",
                        packet=audio_packets,
                        source_pts_us=source_pts_us,
                        pts=packet.pts,
                        dts=packet.dts,
                        duration=packet.duration,
                        size=packet.size,
                        mux_ms=round(mux_ms, 3),
                    )
        if silent_audio_stream is not None:
            packets = 0
            for audio_packet in silent_audio_stream.encode(None):
                output_container.mux(audio_packet)
                packets += 1
                audio_packets += 1
            log_debug("rtmp_remux_silent_audio_flushed", packets=packets)
        return True
    finally:
        if output_container is not None:
            try:
                output_container.close()
            except Exception:
                pass
        if input_container is not None:
            try:
                input_container.__exit__(None, None, None)
            except Exception:
                try:
                    input_container.close()
                except Exception:
                    pass
        video_duration_us = (
            None if first_video_pts_us is None or last_video_pts_us is None else max(0, last_video_pts_us - first_video_pts_us)
        )
        audio_duration_us = (
            None if first_audio_pts_us is None or last_audio_pts_us is None else max(0, last_audio_pts_us - first_audio_pts_us)
        )
        log_debug(
            "rtmp_remux_finished",
            video_packets=video_packets,
            audio_packets=audio_packets,
            dropped_packets=dropped_packets,
            skipped_until_keyframe=skipped_until_keyframe,
            detected_keyframes=detected_keyframes,
            mux_spikes=mux_spikes,
            mux_max_ms=round(mux_max_ms, 3),
            video_duration_us=video_duration_us,
            audio_duration_us=audio_duration_us,
        )


def stream_device_to_rtmp(
    video: CaptureDevice,
    audio: Optional[CaptureDevice],
    output_url: str,
    video_size: str,
    stop: threading.Event,
    *,
    fps: int = DEFAULT_RTMP_VIDEO_FPS,
    debug_log: Optional[Callable[[str], None]] = None,
    debug_context: Optional[dict[str, object]] = None,
) -> None:
    if debug_log is not None:
        values = dict(debug_context or {})
        values.update({"video_size": video_size, "fps": fps, "video_codec": "libx264", "audio_codec": "aac"})
        debug_log(debug_line("rtmp_transcode_selected", **values))
    stream_device_to_rtmp_transcode(
        video,
        audio,
        output_url,
        video_size,
        stop,
        fps=fps,
        debug_log=debug_log,
        debug_context=debug_context,
    )


def can_remux_video_to_mp4(stream) -> bool:
    return stream_codec_name(stream) in {"h264", "hevc", "h265", "mpeg4"}


def can_remux_audio_to_mp4(stream) -> bool:
    return stream_codec_name(stream) in {"aac", "mp3"}


def should_transcode_audio_to_aac(stream) -> bool:
    return stream_codec_name(stream).startswith("pcm_")


def add_mp4_aac_audio_stream(container):
    stream = container.add_stream("aac", rate=AUDIO_RATE)
    stream.layout = audio_layout_name(AUDIO_CHANNELS) or "stereo"
    stream.time_base = Fraction(1, AUDIO_RATE)
    codec_context = getattr(stream, "codec_context", None)
    if codec_context is not None:
        try:
            codec_context.bit_rate = 128_000
        except Exception:
            pass
    return stream


def record_device_remux(
    video: CaptureDevice,
    path: Path,
    end_at: datetime,
    max_file_bytes: int,
    stop: threading.Event,
) -> Optional[RecordingResult]:
    require_pyav()
    if not is_rtsp_capture(video):
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    input_container = None
    output_container = None
    video_packets = 0
    audio_packets = 0
    first_video_pts_us: Optional[int] = None
    last_video_pts_us: Optional[int] = None
    try:
        input_container = open_capture(video)
        input_container = input_container.__enter__()
        input_video = next((stream for stream in input_container.streams if stream.type == "video"), None)
        input_audio = next((stream for stream in input_container.streams if stream.type == "audio"), None)
        if input_video is None or not can_remux_video_to_mp4(input_video):
            return None

        audio_mode = "none"
        if input_audio is not None:
            if can_remux_audio_to_mp4(input_audio):
                audio_mode = "copy"
            elif should_transcode_audio_to_aac(input_audio):
                audio_mode = "aac"
            else:
                print(
                    f"[watchdog] remux recording skipped: audio codec {stream_codec_name(input_audio)} requires full transcoding",
                    flush=True,
                )
                return None

        output_container = av.open(str(path), "w", format="mp4")
        output_video = add_stream_from_template(output_container, input_video, opaque=True)
        output_streams_by_index = {int(input_video.index): output_video}
        demux_streams = [input_video]
        output_audio = None
        audio_resampler = None
        audio_sample_cursor = 0
        if input_audio is not None and audio_mode == "copy":
            output_audio = add_stream_from_template(output_container, input_audio, opaque=True)
            output_streams_by_index[int(input_audio.index)] = output_audio
            demux_streams.append(input_audio)
        elif input_audio is not None and audio_mode == "aac":
            output_audio = add_mp4_aac_audio_stream(output_container)
            audio_resampler = av.audio.resampler.AudioResampler(format="s16", layout="stereo", rate=AUDIO_RATE)
            demux_streams.append(input_audio)

        bases: dict[int, int] = {}
        video_started = False
        video_codec = stream_codec_name(input_video)

        def encode_resampled_audio_frame(audio_frame) -> None:
            nonlocal audio_packets, audio_sample_cursor
            if output_audio is None or getattr(audio_frame, "samples", 0) <= 0:
                return
            audio_frame.pts = audio_sample_cursor
            audio_frame.time_base = Fraction(1, AUDIO_RATE)
            audio_frame.sample_rate = AUDIO_RATE
            audio_sample_cursor += int(audio_frame.samples)
            for audio_packet in output_audio.encode(audio_frame):
                output_container.mux(audio_packet)
                audio_packets += 1

        def transcode_audio_packet(packet) -> None:
            if audio_resampler is None:
                return
            for decoded_frame in packet.decode():
                for audio_frame in audio_resampler.resample(decoded_frame):
                    encode_resampled_audio_frame(audio_frame)

        for packet in input_container.demux(*demux_streams):
            if stop.is_set() or datetime.now().astimezone() >= end_at:
                break
            if max_file_bytes > 0 and path.exists():
                try:
                    if path.stat().st_size >= max_file_bytes:
                        break
                except OSError:
                    pass
            if packet.stream is None:
                continue
            input_stream_index = int(packet.stream.index)
            is_transcoded_audio_packet = (
                audio_mode == "aac"
                and input_audio is not None
                and input_stream_index == int(input_audio.index)
            )
            if input_stream_index not in output_streams_by_index and not is_transcoded_audio_packet:
                continue
            if packet.dts is None and packet.pts is None:
                continue

            is_video_packet = input_stream_index == int(input_video.index)
            source_pts_us = packet_pts_us(packet)
            if is_video_packet:
                detected_keyframe = bool(packet.is_keyframe) or compressed_packet_is_keyframe(packet, video_codec)
                if detected_keyframe:
                    try:
                        packet.is_keyframe = True
                    except Exception:
                        pass
                if not video_started:
                    if not detected_keyframe:
                        continue
                    video_started = True
            elif not video_started:
                continue
            elif is_transcoded_audio_packet:
                transcode_audio_packet(packet)
                continue

            if not rebase_packet_timestamps(packet, bases):
                continue
            packet.stream = output_streams_by_index[input_stream_index]
            output_container.mux(packet)
            if is_video_packet:
                video_packets += 1
                if source_pts_us is not None:
                    first_video_pts_us = source_pts_us if first_video_pts_us is None else first_video_pts_us
                    last_video_pts_us = source_pts_us
            else:
                audio_packets += 1

        if output_audio is not None and audio_mode == "aac":
            if audio_resampler is not None:
                try:
                    for audio_frame in audio_resampler.resample(None):
                        encode_resampled_audio_frame(audio_frame)
                except Exception:
                    pass
            for audio_packet in output_audio.encode(None):
                output_container.mux(audio_packet)
                audio_packets += 1

        if output_container is not None:
            output_container.close()
            output_container = None
        if not video_packets and not audio_packets:
            path.unlink(missing_ok=True)
            return None
        result = probe_media_file(path)
        if result is not None:
            return result
        duration_us = 0 if first_video_pts_us is None or last_video_pts_us is None else max(0, last_video_pts_us - first_video_pts_us)
        return RecordingResult(
            path,
            max(1, int((duration_us + 999_999) // 1_000_000)),
            int(getattr(input_video, "width", 0) or 0),
            int(getattr(input_video, "height", 0) or 0),
            video_packets > 0,
            audio_packets > 0,
        )
    except Exception as error:
        print(f"[watchdog] remux recording failed: {error}", flush=True)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    finally:
        if output_container is not None:
            try:
                output_container.close()
            except Exception:
                pass
        if input_container is not None:
            try:
                input_container.__exit__(None, None, None)
            except Exception:
                try:
                    input_container.close()
                except Exception:
                    pass


def buffered_media_packet(packet, sequence: int) -> BufferedMediaPacket:
    stream = packet.stream
    if stream is None:
        raise ValueError("packet has no stream")
    time_base = getattr(packet, "time_base", None)
    copied_time_base = None
    if time_base is not None:
        copied_time_base = Fraction(int(time_base.numerator), int(time_base.denominator))
    duration = getattr(packet, "duration", None)
    return BufferedMediaPacket(
        sequence=sequence,
        stream_index=int(stream.index),
        media_type=str(getattr(stream, "type", "") or ""),
        data=bytes(packet),
        pts=None if packet.pts is None else int(packet.pts),
        dts=None if packet.dts is None else int(packet.dts),
        duration=None if duration is None else int(duration),
        time_base=copied_time_base,
        is_keyframe=bool(getattr(packet, "is_keyframe", False)),
    )


def packet_from_buffered(item: BufferedMediaPacket, source_stream):
    packet = av.Packet(item.data)
    packet.pts = item.pts
    packet.dts = item.dts
    if item.duration is not None:
        packet.duration = item.duration
    if item.time_base is not None:
        packet.time_base = item.time_base
    packet.stream = source_stream
    if item.is_keyframe:
        try:
            packet.is_keyframe = True
        except Exception:
            pass
    return packet


def buffered_packet_pts_us(item: BufferedMediaPacket) -> Optional[int]:
    timestamp = item.pts if item.pts is not None else item.dts
    if timestamp is None or item.time_base is None:
        return None
    return int(timestamp * item.time_base * 1_000_000)


class MotionMp4Writer:
    """Incrementally remux buffered and live packets from one RTSP session."""

    def __init__(
        self,
        input_video,
        input_audio,
        path: Path,
        started_at: datetime,
    ) -> None:
        require_pyav()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.input_video = input_video
        self.input_audio = input_audio
        self.path = path
        self.started_at = started_at
        self.container = av.open(str(path), "w", format="mp4")
        self.output_video = add_stream_from_template(self.container, input_video, opaque=True)
        self.output_streams_by_index = {int(input_video.index): self.output_video}
        self.input_streams_by_index = {int(input_video.index): input_video}
        self.output_audio = None
        self.audio_resampler = None
        self.audio_mode = "none"
        if input_audio is not None:
            self.input_streams_by_index[int(input_audio.index)] = input_audio
            if can_remux_audio_to_mp4(input_audio):
                self.audio_mode = "copy"
                self.output_audio = add_stream_from_template(self.container, input_audio, opaque=True)
                self.output_streams_by_index[int(input_audio.index)] = self.output_audio
            elif should_transcode_audio_to_aac(input_audio):
                self.audio_mode = "aac"
                self.output_audio = add_mp4_aac_audio_stream(self.container)
                self.audio_resampler = av.audio.resampler.AudioResampler(
                    format="s16",
                    layout="stereo",
                    rate=AUDIO_RATE,
                )
        self.bases: dict[int, int] = {}
        self.video_packets = 0
        self.audio_packets = 0
        self.audio_sample_cursor = 0
        self.first_video_pts_us: Optional[int] = None
        self.last_video_pts_us: Optional[int] = None
        self.closed = False

    def _encode_audio_frame(self, audio_frame) -> None:
        if self.output_audio is None or getattr(audio_frame, "samples", 0) <= 0:
            return
        audio_frame.pts = self.audio_sample_cursor
        audio_frame.time_base = Fraction(1, AUDIO_RATE)
        audio_frame.sample_rate = AUDIO_RATE
        self.audio_sample_cursor += int(audio_frame.samples)
        for audio_packet in self.output_audio.encode(audio_frame):
            self.container.mux(audio_packet)
            self.audio_packets += 1

    def write(self, item: BufferedMediaPacket) -> None:
        if self.closed or (item.pts is None and item.dts is None):
            return
        source_stream = self.input_streams_by_index.get(item.stream_index)
        if source_stream is None:
            return
        is_video = item.stream_index == int(self.input_video.index)
        if not is_video and self.audio_mode == "none":
            return
        packet = packet_from_buffered(item, source_stream)
        if not is_video and self.audio_mode == "aac":
            if self.audio_resampler is None:
                return
            for decoded_frame in packet.decode():
                for audio_frame in self.audio_resampler.resample(decoded_frame):
                    self._encode_audio_frame(audio_frame)
            return

        source_pts_us = packet_pts_us(packet)
        if not rebase_packet_timestamps(packet, self.bases):
            return
        output_stream = self.output_streams_by_index.get(item.stream_index)
        if output_stream is None:
            return
        packet.stream = output_stream
        self.container.mux(packet)
        if is_video:
            self.video_packets += 1
            if source_pts_us is not None:
                if self.first_video_pts_us is None:
                    self.first_video_pts_us = source_pts_us
                self.last_video_pts_us = source_pts_us
        else:
            self.audio_packets += 1

    def size_reached(self, max_file_bytes: int) -> bool:
        if max_file_bytes <= 0:
            return False
        try:
            return self.path.exists() and self.path.stat().st_size >= max_file_bytes
        except OSError:
            return False

    def close(self) -> Optional[ScheduledRecordingResult]:
        if self.closed:
            return None
        self.closed = True
        if self.output_audio is not None and self.audio_mode == "aac":
            if self.audio_resampler is not None:
                try:
                    for audio_frame in self.audio_resampler.resample(None):
                        self._encode_audio_frame(audio_frame)
                except Exception:
                    pass
            for audio_packet in self.output_audio.encode(None):
                self.container.mux(audio_packet)
                self.audio_packets += 1
        self.container.close()
        ended_at = datetime.now().astimezone()
        if self.video_packets <= 0:
            self.path.unlink(missing_ok=True)
            return None
        recording = probe_media_file(self.path)
        if recording is None:
            duration_us = (
                0
                if self.first_video_pts_us is None or self.last_video_pts_us is None
                else max(0, self.last_video_pts_us - self.first_video_pts_us)
            )
            recording = RecordingResult(
                self.path,
                max(1, int((duration_us + 999_999) // 1_000_000)),
                int(getattr(self.input_video, "width", 0) or 0),
                int(getattr(self.input_video, "height", 0) or 0),
                True,
                self.audio_packets > 0,
            )
        return ScheduledRecordingResult(recording, self.started_at, ended_at, True)


def record_verified_motion_stream(
    video: CaptureDevice,
    media_dir: Path,
    device_id: str,
    interval_end: datetime,
    max_file_bytes: int,
    stop: threading.Event,
    on_segment: Callable[[ScheduledRecordingResult], None],
    object_detector: YoloObjectDetector,
    *,
    log: Optional[Callable[[str], None]] = None,
    detector_config: Optional[MotionDetectorConfig] = None,
    pixel_config: Optional[PixelMotionConfig] = None,
) -> None:
    """Detect and record motion from one uninterrupted RTSP connection."""

    require_pyav()
    if not is_rtsp_capture(video):
        raise RuntimeError("GOP pre-roll motion recording requires an RTSP source")
    pixel_settings = pixel_config or PixelMotionConfig()
    detector = MotionVectorDetector(detector_config or MotionDetectorConfig())
    motion_video = CaptureDevice(
        video.label,
        video.input_name,
        video.format_name,
        export_motion_vector_options(dict(video.options)),
    )
    input_container = None
    writer: Optional[MotionMp4Writer] = None
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="watchdog-motion-pixel")
    verification_future: Optional[Future] = None
    yolo_future: Optional[Future] = None
    yolo_mode: Optional[str] = None
    verification_stats: Optional[MotionFrameStats] = None
    verified_pixel_result = None
    candidate_packets: Optional[list[BufferedMediaPacket]] = None
    candidate_bytes = 0
    waiting_for_samples = False
    verification_after_sample = 0
    motion_active = False
    presence_tracker: Optional[YoloPresenceTracker] = None
    next_presence_check = 0.0
    next_presence_sample = 0.0
    presence_samples = deque(maxlen=YOLO_PRESENCE_SAMPLE_COUNT)
    latest_video_frame = None
    cooldown_until = 0.0
    segment_index = 0

    def log_message(message: str) -> None:
        if log is not None:
            log(message)

    def segment_started_at(items: list[BufferedMediaPacket]) -> datetime:
        video_times = [
            timestamp
            for item in items
            if item.media_type == "video"
            for timestamp in [buffered_packet_pts_us(item)]
            if timestamp is not None
        ]
        pre_roll_us = 0 if len(video_times) < 2 else max(0, video_times[-1] - video_times[0])
        return datetime.now().astimezone() - timedelta(microseconds=pre_roll_us)

    def create_writer(input_video, input_audio, initial: list[BufferedMediaPacket]) -> MotionMp4Writer:
        nonlocal segment_index
        segment_index += 1
        started_at = segment_started_at(initial)
        timestamp = started_at.strftime("%Y%m%d_%H%M%S_%f")
        path = media_dir / f"motion_{device_id}_{timestamp}_{segment_index}.mp4"
        new_writer = MotionMp4Writer(input_video, input_audio, path, started_at)
        for buffered in initial:
            new_writer.write(buffered)
        log_message(
            f"motion_recording_started path={path} pre_roll_packets={len(initial)} "
            f"pre_roll_bytes={sum(len(item.data) for item in initial)}"
        )
        return new_writer

    def finish_writer() -> None:
        nonlocal writer
        if writer is None:
            return
        active_writer = writer
        writer = None
        result = active_writer.close()
        if result is not None:
            on_segment(result)

    def reset_start_candidate(*, rejected: bool) -> None:
        nonlocal verification_future, yolo_future, yolo_mode, verification_stats
        nonlocal verified_pixel_result, candidate_packets, candidate_bytes
        nonlocal waiting_for_samples, verification_after_sample, motion_active, cooldown_until
        if verification_future is not None:
            verification_future.cancel()
        if yolo_future is not None and yolo_mode == "initial":
            yolo_future.cancel()
            yolo_future = None
            yolo_mode = None
        verification_future = None
        verification_stats = None
        verified_pixel_result = None
        candidate_packets = None
        candidate_bytes = 0
        waiting_for_samples = False
        verification_after_sample = 0
        if rejected:
            detector.reject_motion()
            motion_active = False
            cooldown_until = time.monotonic() + pixel_settings.reject_cooldown_seconds

    def detections_text(detections) -> str:
        return ",".join(
            f"{item.label}:{item.confidence:.2f}"
            for item in detections
        ) or "none"

    try:
        input_container = open_capture(motion_video)
        input_container = input_container.__enter__()
        input_video = next((stream for stream in input_container.streams if stream.type == "video"), None)
        input_audio = next((stream for stream in input_container.streams if stream.type == "audio"), None)
        if input_video is None:
            raise RuntimeError("motion detector input has no video stream")
        video_codec = stream_codec_name(input_video)
        if not codec_supports_motion_vectors(video_codec):
            raise RuntimeError(
                f"motion vectors are supported only for H.264/H.265, got {video_codec or 'unknown'}"
            )
        input_video.codec_context.options = export_motion_vector_options(
            getattr(input_video.codec_context, "options", None)
        )
        demux_streams = [input_video]
        if input_audio is not None:
            demux_streams.append(input_audio)
        log_message(
            "motion_detector_opened "
            f"codec={video_codec} width={getattr(input_video, 'width', None)} "
            f"height={getattr(input_video, 'height', None)}"
        )

        sequence = 0
        gop_packets: list[BufferedMediaPacket] = []
        gop_bytes = 0
        gop_started = False
        gray_samples = []
        frame_in_gop = 0
        sample_serial = 0
        last_analyzed = 0.0
        min_analyze_interval = (
            0.0 if detector.config.max_analyze_fps <= 0 else 1.0 / detector.config.max_analyze_fps
        )
        no_vector_frames = 0
        warned_no_vectors = False

        for packet in input_container.demux(*demux_streams):
            if stop.is_set() or datetime.now().astimezone() >= interval_end:
                break
            if packet.stream is None or (packet.pts is None and packet.dts is None):
                continue
            stream_index = int(packet.stream.index)
            is_video_packet = stream_index == int(input_video.index)
            is_audio_packet = input_audio is not None and stream_index == int(input_audio.index)
            if not is_video_packet and not is_audio_packet:
                continue

            is_keyframe = False
            split_recording = False
            if is_video_packet:
                is_keyframe = bool(packet.is_keyframe) or compressed_packet_is_keyframe(packet, video_codec)
                if is_keyframe:
                    try:
                        packet.is_keyframe = True
                    except Exception:
                        pass
                    if writer is not None and writer.size_reached(max_file_bytes):
                        finish_writer()
                        split_recording = True
                    gop_packets = []
                    gop_bytes = 0
                    gop_started = True
                    keep_samples = max(0, pixel_settings.previous_gop_samples)
                    gray_samples = gray_samples[-keep_samples:] if keep_samples else []
                    frame_in_gop = 0

            sequence += 1
            buffered = buffered_media_packet(packet, sequence)
            if is_keyframe and not buffered.is_keyframe:
                buffered = BufferedMediaPacket(
                    buffered.sequence,
                    buffered.stream_index,
                    buffered.media_type,
                    buffered.data,
                    buffered.pts,
                    buffered.dts,
                    buffered.duration,
                    buffered.time_base,
                    True,
                )

            if gop_started:
                gop_packets.append(buffered)
                gop_bytes += len(buffered.data)
                if gop_bytes > MOTION_GOP_MAX_BYTES:
                    log_message(f"motion_gop_buffer_dropped bytes={gop_bytes} reason=size_limit")
                    gop_packets = []
                    gop_bytes = 0
                    gop_started = False
                    gray_samples = []

            if candidate_packets is not None:
                candidate_packets.append(buffered)
                candidate_bytes += len(buffered.data)
                if candidate_bytes > MOTION_PENDING_MAX_BYTES:
                    log_message(f"motion_candidate_rejected bytes={candidate_bytes} reason=pending_size_limit")
                    reset_start_candidate(rejected=True)

            if writer is not None:
                writer.write(buffered)
            elif split_recording:
                # A file-size split occurs only on a keyframe, so the new file is decodable.
                writer = create_writer(input_video, input_audio, [buffered])

            if is_video_packet:
                for frame in packet.decode():
                    latest_video_frame = frame
                    frame_in_gop += 1
                    sampled_current = False
                    if frame_in_gop == 1 or frame_in_gop % max(1, pixel_settings.sample_every_frames) == 0:
                        gray_samples.append(frame_motion_sample(frame, pixel_settings.max_sample_width))
                        sample_serial += 1
                        sampled_current = True
                        if len(gray_samples) > pixel_settings.max_samples_per_gop:
                            del gray_samples[: len(gray_samples) - pixel_settings.max_samples_per_gop]

                    now = time.monotonic()
                    event = None
                    if min_analyze_interval <= 0.0 or now - last_analyzed >= min_analyze_interval:
                        last_analyzed = now
                        event = detector.process_frame(frame)
                        vector_count = 0 if detector.last_stats is None else detector.last_stats.vector_count
                        if vector_count <= 0:
                            no_vector_frames += 1
                            if not warned_no_vectors and no_vector_frames >= detector.config.no_vector_warning_frames:
                                warned_no_vectors = True
                                log_message("motion_detector_no_motion_vectors")
                        else:
                            no_vector_frames = 0

                    if event is not None and event.kind == "motion_start":
                        motion_active = True
                        if writer is None and verification_future is None and yolo_future is None and candidate_packets is None:
                            if now < cooldown_until:
                                detector.reject_motion()
                                motion_active = False
                            elif not gop_started or not gop_packets:
                                detector.reject_motion()
                                motion_active = False
                                log_message("motion_candidate_rejected reason=no_keyframe_buffer")
                            else:
                                if not sampled_current:
                                    gray_samples.append(frame_motion_sample(frame, pixel_settings.max_sample_width))
                                    sample_serial += 1
                                    if len(gray_samples) > pixel_settings.max_samples_per_gop:
                                        del gray_samples[: len(gray_samples) - pixel_settings.max_samples_per_gop]
                                verification_stats = event.stats
                                candidate_packets = list(gop_packets)
                                candidate_bytes = gop_bytes
                                waiting_for_samples = True
                                verification_after_sample = sample_serial + max(
                                    0,
                                    pixel_settings.post_candidate_samples,
                                )
                    elif event is not None and event.kind == "motion_stop":
                        motion_active = False

                    if (
                        waiting_for_samples
                        and verification_future is None
                        and len(gray_samples) >= pixel_settings.min_samples
                        and sample_serial >= verification_after_sample
                    ):
                        waiting_for_samples = False
                        verification_future = executor.submit(
                            verify_gray_motion,
                            tuple(gray_samples),
                            pixel_settings,
                        )

            if verification_future is not None and verification_future.done():
                result = verification_future.result()
                verification_future = None
                log_message(
                    "motion_pixel_verification "
                    f"result={result.motion} reason={result.reason} "
                    f"samples={result.analyzed_samples} motion_samples={result.motion_samples} "
                    f"scene_changes={result.scene_change_samples} "
                    f"max_foreground_ratio={result.max_foreground_ratio:.4f} "
                    f"max_global_change_ratio={result.max_global_change_ratio:.4f} "
                    f"regions={len(result.motion_regions)}"
                )
                if (
                    result.motion
                    and candidate_packets
                    and verification_stats is not None
                    and latest_video_frame is not None
                ):
                    verified_pixel_result = result
                    yolo_mode = "initial"
                    yolo_image = frame_bgr_sample(latest_video_frame, YOLO_SAMPLE_MAX_WIDTH)
                    yolo_future = executor.submit(object_detector.detect, yolo_image)
                else:
                    reset_start_candidate(rejected=True)

            if yolo_future is not None and yolo_future.done():
                detections = yolo_future.result()
                completed_mode = yolo_mode
                yolo_future = None
                yolo_mode = None
                if completed_mode == "initial":
                    regions = () if verified_pixel_result is None else verified_pixel_result.motion_regions
                    tracker = YoloPresenceTracker.from_initial(
                        detections,
                        regions,
                        misses_to_stop=YOLO_PRESENCE_MISSES_TO_STOP,
                    )
                    log_message(
                        "motion_yolo_initial "
                        f"detections={detections_text(detections)} "
                        f"matched={0 if tracker is None else len(tracker.active)}"
                    )
                    if tracker is not None and candidate_packets:
                        writer = create_writer(input_video, input_audio, candidate_packets)
                        presence_tracker = tracker
                        now = time.monotonic()
                        presence_samples.clear()
                        presence_samples.append(
                            frame_bgr_sample(latest_video_frame, YOLO_SAMPLE_MAX_WIDTH)
                        )
                        next_presence_sample = now + YOLO_PRESENCE_SAMPLE_INTERVAL_SECONDS
                        next_presence_check = now + YOLO_PRESENCE_INTERVAL_SECONDS
                        reset_start_candidate(rejected=False)
                    else:
                        log_message("motion_candidate_rejected reason=no_relevant_moving_object")
                        reset_start_candidate(rejected=True)
                elif completed_mode == "presence" and presence_tracker is not None:
                    present = presence_tracker.update(detections, motion_active=motion_active)
                    log_message(
                        "motion_yolo_presence "
                        f"detections={detections_text(detections)} present={present} "
                        f"misses={presence_tracker.consecutive_misses}/"
                        f"{presence_tracker.effective_misses_to_stop} motion_active={motion_active}"
                    )
                    if presence_tracker.should_stop:
                        log_message("motion_recording_stop reason=tracked_objects_absent")
                        finish_writer()
                        presence_tracker = None
                        presence_samples.clear()
                        detector.reject_motion()
                        motion_active = False
                        cooldown_until = time.monotonic() + pixel_settings.reject_cooldown_seconds

            now = time.monotonic()
            if (
                writer is not None
                and presence_tracker is not None
                and latest_video_frame is not None
                and now >= next_presence_sample
            ):
                presence_samples.append(
                    frame_bgr_sample(latest_video_frame, YOLO_SAMPLE_MAX_WIDTH)
                )
                next_presence_sample = now + YOLO_PRESENCE_SAMPLE_INTERVAL_SECONDS
            if (
                writer is not None
                and presence_tracker is not None
                and yolo_future is None
                and latest_video_frame is not None
                and now >= next_presence_check
            ):
                yolo_mode = "presence"
                samples = tuple(presence_samples) or (
                    frame_bgr_sample(latest_video_frame, YOLO_SAMPLE_MAX_WIDTH),
                )
                yolo_future = executor.submit(
                    detect_presence_samples,
                    object_detector,
                    samples,
                    tuple(presence_tracker.active),
                )
                next_presence_check = now + YOLO_PRESENCE_INTERVAL_SECONDS

    finally:
        try:
            finish_writer()
        finally:
            if verification_future is not None:
                verification_future.cancel()
            if yolo_future is not None:
                yolo_future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            if input_container is not None:
                try:
                    input_container.__exit__(None, None, None)
                except Exception:
                    try:
                        input_container.close()
                    except Exception:
                        pass


def record_device_scheduled(
    video: CaptureDevice,
    audio: Optional[CaptureDevice],
    path: Path,
    end_at: datetime,
    video_size: str,
    max_file_bytes: int,
    stop: threading.Event,
) -> Optional[ScheduledRecordingResult]:
    started_at = datetime.now().astimezone()
    result: Optional[RecordingResult] = None
    remuxed = False
    if is_rtsp_capture(video) and audio is None:
        result = record_device_remux(video, path, end_at, max_file_bytes, stop)
        remuxed = result is not None
    if result is None and not stop.is_set():
        remaining_seconds = max(1, int((end_at - datetime.now().astimezone()).total_seconds()))
        result = record_device(
            video,
            audio,
            path,
            remaining_seconds,
            video_size,
            max_file_bytes=max_file_bytes,
            external_stop=stop,
        )
    ended_at = datetime.now().astimezone()
    if result is None:
        return None
    return ScheduledRecordingResult(result, started_at, ended_at, remuxed)


def tdlib_directory_has_session(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    try:
        return any(path.iterdir())
    except OSError:
        return False


def wait_for_telegram_authorization(login: str, script_dir: Path, *, reset_auth: bool) -> None:
    database_dir = script_dir / "tdlib"
    files_dir = database_dir / "files"
    if reset_auth and database_dir.exists():
        print(f"Removing existing TDLib session: {database_dir}")
        shutil.rmtree(database_dir)

    ready = threading.Event()
    failed = threading.Event()

    def on_auth_code() -> str:
        return input("Telegram code: ").strip()

    def on_td_json(json_text: str) -> None:
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            return
        object_type = str(data.get("@type") or "")
        if object_type == "updateAuthorizationState":
            state = data.get("authorization_state") or {}
            object_type = str(state.get("@type") or "")
        if object_type == "authorizationStateReady":
            ready.set()
        elif object_type in {"authorizationStateClosing", "authorizationStateClosed", "authorizationStateLoggingOut"}:
            failed.set()

    client = Tam(
        phone_number=login,
        database_directory=database_dir,
        files_directory=files_dir,
        on_auth_code=on_auth_code,
        on_td_json=on_td_json,
    )
    try:
        client.start()
        try:
            client.td_send_json({"@type": "getAuthorizationState", "@extra": "watchdog_setup_auth_state"})
        except RuntimeError:
            pass
        while not ready.wait(0.25):
            if failed.is_set():
                raise RuntimeError("Telegram authorization failed or TDLib was closed")
    finally:
        client.close()


def run_authorization_setup(login: str, script_dir: Path) -> None:
    database_dir = script_dir / "tdlib"
    reset_auth = False
    if tdlib_directory_has_session(database_dir):
        reset_auth = prompt_yes_no(
            "Existing TDLib authorization was found. Re-authorize and replace it?",
            default=False,
        )
    wait_for_telegram_authorization(login, script_dir, reset_auth=reset_auth)
    print("Telegram authorization is ready.")


def existing_source_machine_ids(devices: list[dict[str, object]], kind: str) -> set[str]:
    result: set[str] = set()
    for device in devices:
        if device.get("kind") != kind:
            continue
        video = device.get("video")
        if not isinstance(video, dict):
            continue
        value = str(video.get("machine_id") or video.get("input_name") or "").strip()
        if value:
            result.add(value)
        if kind == "screen":
            monitor_index = windows_monitor_number(str(video.get("label") or video.get("display_name") or "")) or windows_monitor_number(value)
            if monitor_index is not None:
                result.add(f"windows_capture_monitor_{monitor_index}")
    return result


def next_camera_id(devices: list[dict[str, object]]) -> str:
    used: set[str] = set()
    for device in devices:
        value = str(device.get("id") or "").strip()
        if value:
            used.add(value)
    index = 1
    while f"camera_{index}" in used:
        index += 1
    return f"camera_{index}"


def choose_video_sources(existing_devices: Optional[list[dict[str, object]]] = None) -> list[dict[str, object]]:
    webcams = enumerate_video_devices()
    microphones = enumerate_audio_devices()
    screens = enumerate_screen_devices()
    existing_devices = [] if existing_devices is None else existing_devices
    selected_webcams = existing_source_machine_ids(existing_devices, "webcam")
    selected_screens = existing_source_machine_ids(existing_devices, "screen")
    devices: list[dict[str, object]] = []

    if not screens:
        print("Screen capture is not available in this environment.")

    while True:
        available_webcams = [device for device in webcams if device.input_name not in selected_webcams]
        available_screens = [device for device in screens if device.input_name not in selected_screens]
        choices: list[tuple[str, str]] = []
        if available_webcams:
            choices.append(("webcam", "Add webcam"))
        choices.append(("rtsp", "Add RTSP camera"))
        if available_screens:
            choices.append(("screen", "Add screen capture"))
        if devices:
            choices.append(("finish", "Finish setup"))

        choice = prompt_menu("Add video source:", choices)
        if choice == "finish":
            break
        if choice == "webcam":
            video = choose_capture_device("video", available_webcams)
            selected_webcams.add(video.input_name)
            kind = "webcam"
        elif choice == "screen":
            video = choose_capture_device("screen", available_screens)
            selected_screens.add(video.input_name)
            kind = "screen"
        else:
            video = prompt_rtsp_device()
            kind = "rtsp"

        audio = None if kind == "rtsp" else maybe_choose_microphone(microphones)
        devices.append(
            {
                "id": next_camera_id(existing_devices + devices),
                "kind": kind,
                "display_name": video.label,
                "label": video.label,
                "video": capture_to_json(video),
                "audio": audio,
            }
        )
        print(f"Added source: {video.label}")

        more_finite_sources = any(device.input_name not in selected_webcams for device in webcams) or any(
            device.input_name not in selected_screens for device in screens
        )
        if not prompt_yes_no("Add another camera or screen?", default=more_finite_sources):
            break

    if not devices:
        raise SystemExit("No video sources were configured.")
    return devices


def load_existing_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Failed to read existing config {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"Existing config must contain a JSON object: {path}")
    return value


def existing_config_devices(config: dict[str, object]) -> list[dict[str, object]]:
    devices = config.get("devices")
    if not isinstance(devices, list):
        return []
    result: list[dict[str, object]] = []
    for item in devices:
        if isinstance(item, dict):
            result.append(dict(item))
    return result


def save_config(
    path: Path,
    login: str,
    script_dir: Path,
    devices: list[dict[str, object]],
    *,
    existing_config: Optional[dict[str, object]] = None,
) -> None:
    config = dict(existing_config or {})
    config.update({
        "version": 1,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "login": login,
        "tdlib": {
            "database_directory": str(script_dir / "tdlib"),
            "files_directory": str(script_dir / "tdlib" / "files"),
        },
        "devices": devices,
    })
    if "created_at" not in config:
        config["created_at"] = config["updated_at"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def td_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def supergroup_id_from_chat_id(chat_id: int) -> Optional[int]:
    if chat_id <= -1_000_000_000_000:
        return -1_000_000_000_000 - chat_id
    return None


def basic_group_id_from_chat_id(chat_id: int) -> Optional[int]:
    if chat_id < 0 and chat_id > -1_000_000_000_000:
        return -chat_id
    return None


def utf16_length(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def text_entity(text: str, start: int, end: int, entity_type: str) -> dict[str, object]:
    return {
        "@type": "textEntity",
        "offset": utf16_length(text[:start]),
        "length": utf16_length(text[start:end]),
        "type": {"@type": entity_type},
    }


def formatted_text(
    text: str,
    entities: Optional[list[dict[str, object]]] = None,
) -> dict[str, object]:
    return {"@type": "formattedText", "text": str(text), "entities": entities or []}


def text_lines(text: str):
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        yield offset, line
        offset += len(raw_line)
    if text and not text.endswith(("\r", "\n")) and offset < len(text):
        yield offset, text[offset:]


def line_separator_index(line: str) -> int:
    indexes = [index for marker in (" - ", " — ") if (index := line.find(marker)) >= 0]
    return min(indexes) if indexes else -1


def sorted_entities(entities: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        entities,
        key=lambda entity: (
            int(entity.get("offset") or 0),
            -int(entity.get("length") or 0),
        ),
    )


def help_formatted_text(ui: dict[str, str]) -> dict[str, object]:
    text = technical_message(ui["help_text"])
    entities: list[dict[str, object]] = []
    headings = {
        ui["help_title"],
        ui["help_general_heading"],
        ui["help_device_topics_heading"],
    }

    for heading in headings:
        start = text.find(heading)
        if start >= 0:
            entities.append(text_entity(text, start, start + len(heading), "textEntityTypeBold"))

    for line_start, line in text_lines(text):
        stripped = line.strip()
        if not stripped or stripped in headings:
            continue
        content_start = line_start + len(line) - len(line.lstrip())
        separator = line_separator_index(line)
        if separator >= 0:
            entities.append(text_entity(text, content_start, line_start + separator, "textEntityTypeCode"))

    command = "rec help"
    command_start = text.rfind(command)
    if command_start >= 0:
        entities.append(
            text_entity(text, command_start, command_start + len(command), "textEntityTypeCode")
        )
    return formatted_text(text, sorted_entities(entities))


def record_schedule_help_formatted_text(ui: dict[str, str]) -> dict[str, object]:
    text = technical_message(ui["record_schedule_help_text"])
    entities: list[dict[str, object]] = []
    heading_by_text = {
        ui["record_schedule_title"]: "title",
        ui["record_schedule_syntax_heading"]: "syntax",
        ui["record_schedule_days_heading"]: "days",
        ui["record_schedule_ranges_heading"]: "ranges",
        ui["record_schedule_multiple_ranges_heading"]: "multiple_ranges",
        ui["record_schedule_examples_heading"]: "examples",
        ui["record_schedule_priority_heading"]: "priority",
        ui["record_schedule_commands_heading"]: "commands",
    }

    for heading in heading_by_text:
        start = text.find(heading)
        if start >= 0:
            entities.append(text_entity(text, start, start + len(heading), "textEntityTypeBold"))

    section = ""
    for line_start, line in text_lines(text):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in heading_by_text:
            section = heading_by_text[stripped]
            continue

        content_start = line_start + len(line) - len(line.lstrip())
        code_length = 0
        if stripped.startswith(("rec ", "rm ", "m1 ", "Motion ")):
            separator = line_separator_index(stripped)
            code_length = separator if separator >= 0 else len(stripped)
        elif section == "days":
            if stripped.startswith("mon tue ") or " = " in stripped:
                code_length = len(stripped)
            else:
                code_length = len(stripped.split(maxsplit=1)[0])
        elif section == "ranges":
            code_length = len(stripped.split(maxsplit=1)[0])

        if code_length > 0:
            entities.append(
                text_entity(
                    text,
                    content_start,
                    content_start + code_length,
                    "textEntityTypeCode",
                )
            )
    return formatted_text(text, sorted_entities(entities))


def input_file_local(path: Path) -> dict[str, object]:
    return {"@type": "inputFileLocal", "path": str(path)}


def input_message_text(text: Union[str, dict[str, object]]) -> dict[str, object]:
    message_text = text if isinstance(text, dict) else formatted_text(text)
    return {
        "@type": "inputMessageText",
        "text": message_text,
        "link_preview_options": None,
        "clear_draft": True,
    }


def input_message_photo(path: Path, caption: str, width: int, height: int) -> dict[str, object]:
    return {
        "@type": "inputMessagePhoto",
        "photo": input_file_local(path),
        "thumbnail": None,
        "video": None,
        "added_sticker_file_ids": [],
        "width": width,
        "height": height,
        "caption": formatted_text(caption),
        "show_caption_above_media": False,
        "self_destruct_type": None,
        "has_spoiler": False,
    }


def input_message_video(path: Path, caption: str, duration: int, width: int, height: int) -> dict[str, object]:
    return {
        "@type": "inputMessageVideo",
        "video": input_file_local(path),
        "thumbnail": None,
        "cover": None,
        "start_timestamp": 0,
        "added_sticker_file_ids": [],
        "duration": duration,
        "width": width,
        "height": height,
        "supports_streaming": True,
        "caption": formatted_text(caption),
        "show_caption_above_media": False,
        "self_destruct_type": None,
        "has_spoiler": False,
    }


def message_topic_forum(forum_topic_id: Optional[int]) -> Optional[dict[str, object]]:
    if forum_topic_id is None:
        return None
    return {"@type": "messageTopicForum", "forum_topic_id": forum_topic_id}


def forum_topic_icon(color: int) -> dict[str, object]:
    return {"@type": "forumTopicIcon", "color": color, "custom_emoji_id": 0}


def forum_topic_id_from_message(message: dict[str, Any]) -> Optional[int]:
    topic = message.get("topic_id")
    if isinstance(topic, dict) and topic.get("@type") == "messageTopicForum":
        return td_int(topic.get("forum_topic_id"))
    return None


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, dict):
        return ""
    if content.get("@type") == "messageText":
        text = content.get("text")
        if isinstance(text, dict):
            return str(text.get("text") or "")
    return ""


def message_sender_user_id(message: dict[str, Any]) -> Optional[int]:
    sender = message.get("sender_id")
    if isinstance(sender, dict) and sender.get("@type") == "messageSenderUser":
        return td_int(sender.get("user_id"))
    return None


def normalize_command_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


DAY_NAMES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
DAY_INDEX = {name: index for index, name in enumerate(DAY_NAMES)}
DAY_GROUPS = {
    "weekdays": [0, 1, 2, 3, 4],
    "weekend": [5, 6],
}


def strip_record_command(text: str) -> Optional[str]:
    parsed = parse_record_command(text)
    return None if parsed is None else parsed[0]


def parse_record_command(text: str) -> Optional[tuple[str, bool]]:
    normalized = normalize_command_text(text)
    if normalized == "rec":
        return "", False
    if normalized.startswith("rec "):
        args = normalized[4:].strip()
        if args == "motion":
            return "", True
        if args.startswith("motion "):
            return args[7:].strip(), True
        return args, False
    if normalized == "r":
        return "", False
    if normalized.startswith("r "):
        return normalized[2:].strip(), False
    if normalized == "rm":
        return "", True
    if normalized.startswith("rm "):
        return normalized[3:].strip(), True
    return None


def parse_schedule_days(text: str) -> tuple[bool, set[int]]:
    value = text.strip().casefold()
    if not value:
        raise ValueError("missing days")
    if value == "*":
        return True, set(range(7))
    days: set[int] = set()
    for token in re.split(r"[\s,]+", value):
        if not token:
            continue
        if token in DAY_GROUPS:
            days.update(DAY_GROUPS[token])
            continue
        if "-" in token:
            start_name, end_name = (part.strip() for part in token.split("-", 1))
            if start_name not in DAY_INDEX or end_name not in DAY_INDEX:
                raise ValueError(f"unknown day range: {token}")
            start_index = DAY_INDEX[start_name]
            end_index = DAY_INDEX[end_name]
            index = start_index
            while True:
                days.add(index)
                if index == end_index:
                    break
                index = (index + 1) % 7
            continue
        if token not in DAY_INDEX:
            raise ValueError(f"unknown day: {token}")
        days.add(DAY_INDEX[token])
    if not days:
        raise ValueError("missing days")
    return False, days


def parse_schedule_ranges(text: str) -> list[tuple[int, int]]:
    value = text.strip().casefold()
    if not value:
        raise ValueError("missing ranges")
    if value == "off":
        return []
    ranges: list[tuple[int, int]] = []
    for token in re.split(r"[\s,]+", value):
        if not token:
            continue
        match = re.fullmatch(r"(\d{1,2})-(\d{1,2})", token)
        if not match:
            raise ValueError(f"bad range: {token}")
        start_hour = int(match.group(1))
        end_hour = int(match.group(2))
        if not (0 <= start_hour <= 23 and 0 <= end_hour <= 24):
            raise ValueError(f"hour out of range: {token}")
        start_minute = start_hour * 60
        end_minute = end_hour * 60
        if start_minute == end_minute:
            raise ValueError(f"empty range: {token}")
        if end_minute <= start_minute:
            end_minute += 24 * 60
        ranges.append((start_minute, end_minute))
    if not ranges:
        raise ValueError("missing ranges")
    return ranges


def normalize_schedule_spec(spec: str) -> str:
    return re.sub(r"\s+", " ", spec.strip().casefold())


def parse_record_schedule_spec(spec: str) -> list[dict[str, object]]:
    value = normalize_schedule_spec(spec)
    if not value:
        raise ValueError("missing schedule")
    rules: list[dict[str, object]] = []
    for part in value.split(";"):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"missing ':' in rule: {item}")
        days_text, ranges_text = item.split(":", 1)
        wildcard, days = parse_schedule_days(days_text)
        ranges = parse_schedule_ranges(ranges_text)
        rules.append({"wildcard": wildcard, "days": days, "ranges": ranges})
    if not rules:
        raise ValueError("missing schedule rules")
    return rules


def schedule_spec_from_device(device: dict[str, object]) -> str:
    value = device.get("record_schedule")
    if isinstance(value, dict):
        return str(value.get("spec") or "").strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def schedule_motion_from_device(device: dict[str, object]) -> bool:
    value = device.get("record_schedule")
    return bool(value.get("motion")) if isinstance(value, dict) else False


def schedule_ranges_for_date(rules: list[dict[str, object]], date_value: datetime) -> list[tuple[int, int]]:
    weekday = date_value.weekday()
    specific = [
        rule
        for rule in rules
        if not bool(rule.get("wildcard")) and weekday in (rule.get("days") or set())
    ]
    selected = specific if specific else [rule for rule in rules if bool(rule.get("wildcard"))]
    ranges: list[tuple[int, int]] = []
    for rule in selected:
        ranges.extend(rule.get("ranges") or [])
    return ranges


def schedule_has_specific_for_date(rules: list[dict[str, object]], date_value: datetime) -> bool:
    weekday = date_value.weekday()
    return any(not bool(rule.get("wildcard")) and weekday in (rule.get("days") or set()) for rule in rules)


def schedule_intervals_around(rules: list[dict[str, object]], now: datetime, days_forward: int) -> list[tuple[datetime, datetime]]:
    base_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    intervals: list[tuple[datetime, datetime]] = []
    for day_offset in range(-1, days_forward + 1):
        day_start = base_midnight + timedelta(days=day_offset)
        for start_minute, end_minute in schedule_ranges_for_date(rules, day_start):
            next_day = day_start + timedelta(days=1)
            if end_minute > 24 * 60 and schedule_has_specific_for_date(rules, next_day):
                end_minute = 24 * 60
                if end_minute <= start_minute:
                    continue
            start_at = day_start + timedelta(minutes=start_minute)
            end_at = day_start + timedelta(minutes=end_minute)
            intervals.append((start_at, end_at))
    intervals.sort(key=lambda item: item[0])
    return intervals


def current_schedule_interval(spec: str, now: Optional[datetime] = None) -> Optional[tuple[datetime, datetime]]:
    if not spec:
        return None
    now = datetime.now().astimezone() if now is None else now
    rules = parse_record_schedule_spec(spec)
    for start_at, end_at in schedule_intervals_around(rules, now, 1):
        if start_at <= now < end_at:
            return start_at, end_at
    return None


def next_schedule_start(spec: str, now: Optional[datetime] = None) -> Optional[datetime]:
    if not spec:
        return None
    now = datetime.now().astimezone() if now is None else now
    rules = parse_record_schedule_spec(spec)
    for start_at, _ in schedule_intervals_around(rules, now, 8):
        if start_at > now:
            return start_at
    return None


def format_record_schedule(spec: str, motion: bool = False, ui: Optional[dict[str, str]] = None) -> str:
    if not spec:
        return translation_text(ui, "record_schedule_not_set")
    try:
        current = current_schedule_interval(spec)
        next_start = next_schedule_start(spec)
    except ValueError as error:
        return translation_text(ui, "record_schedule_invalid", error=error, spec=spec)
    command = "rec motion" if motion else "rec"
    mode = translation_text(ui, "record_schedule_mode_motion" if motion else "record_schedule_mode_continuous")
    lines = [
        translation_text(ui, "record_schedule_header", command=command, spec=spec),
        translation_text(ui, "record_schedule_mode", mode=mode),
    ]
    if current is not None:
        lines.append(
            translation_text(
                ui,
                "record_schedule_active",
                start=current[0].strftime("%Y-%m-%d %H:%M"),
                end=current[1].strftime("%Y-%m-%d %H:%M"),
            )
        )
    elif next_start is not None:
        lines.append(translation_text(ui, "record_schedule_next", time=next_start.strftime("%Y-%m-%d %H:%M")))
    else:
        lines.append(translation_text(ui, "record_schedule_disabled"))
    return "\n".join(lines)


def technical_message(text: str) -> str:
    value = str(text).strip()
    if value.startswith(TECHNICAL_MESSAGE_PREFIX) or value.startswith(TECHNICAL_MESSAGE_PREFIX.strip()):
        return value
    return f"{TECHNICAL_MESSAGE_PREFIX}{value}"


def is_technical_message_text(text: str) -> bool:
    value = str(text).strip()
    return value.startswith(TECHNICAL_MESSAGE_PREFIX) or value.startswith(TECHNICAL_MESSAGE_PREFIX.strip())


def parse_record_seconds(text: str) -> int:
    match = re.search(r"\d+", text)
    if not match:
        return 30
    return max(1, min(3600, int(match.group(0))))


def recording_caption_source(device: dict[str, object]) -> str:
    topic = device.get("topic")
    if isinstance(topic, dict):
        topic_name = str(topic.get("name") or "").strip()
        if topic_name:
            return topic_name
    display_name = str(device.get("display_name") or device.get("label") or "").strip()
    if display_name:
        return display_name
    video = capture_from_json(device.get("video"))
    if is_rtsp_capture(video):
        return redact_url(video.input_name)
    return str(device.get("id") or "device")


def recording_time_range_text(started_at: datetime, ended_at: datetime) -> str:
    start_text = started_at.strftime("%Y-%m-%d %H:%M:%S")
    if started_at.date() == ended_at.date():
        end_text = ended_at.strftime("%H:%M:%S")
    else:
        end_text = ended_at.strftime("%Y-%m-%d %H:%M:%S")
    return f"{start_text} → {end_text}"


def scheduled_recording_caption(device: dict[str, object], result: ScheduledRecordingResult) -> str:
    return f"📹 {recording_caption_source(device)}\n🕒 {recording_time_range_text(result.started_at, result.ended_at)}"


class WatchdogApp:
    def __init__(
        self,
        login: str,
        script_dir: Path,
        config_path: Path,
        video_size: str,
        language: str,
        debug_log_enabled: bool = False,
    ) -> None:
        self.login = login
        self.script_dir = script_dir
        self.config_path = config_path
        self.video_size = video_size
        self.media_dir = script_dir / "media" / "files"
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.stuff_dir = script_dir / "media" / "stuff"
        self.stuff_dir.mkdir(parents=True, exist_ok=True)
        self.language = normalize_language(language)
        self.ui = load_translation(self.stuff_dir, self.language)
        self.lock = threading.RLock()
        self.print_lock = threading.Lock()
        self.debug_log_enabled = debug_log_enabled
        self.debug_log_path = script_dir / "watchdog_debug.log"
        self.debug_log_lock = threading.Lock()
        self.yolo_detector = YoloObjectDetector(
            self.stuff_dir / YOLO_MODEL_FILENAME,
            log=lambda text: self.safe_print(f"[watchdog] {text}"),
        )
        self.yolo_detector.prepare()
        if self.debug_log_enabled:
            self.debug_log_path.write_text("", encoding="utf-8")
            self.write_debug_log(
                debug_line(
                    "watchdog_debug_started",
                    login=login,
                    platform=sys.platform,
                    python=sys.version.replace("\n", " "),
                    language=self.language,
                    video_size=video_size,
                    live_fps=DEFAULT_LIVE_VIDEO_FPS,
                    rtmp_video_size=DEFAULT_RTMP_VIDEO_SIZE,
                    rtmp_fps=DEFAULT_RTMP_VIDEO_FPS,
                    record_fps=DEFAULT_RECORD_FPS,
                    config_path=config_path,
                )
            )
        self.stop_event = threading.Event()
        self.setup_done = threading.Event()
        self.config = load_existing_config(config_path)
        self.devices = existing_config_devices(self.config)
        self.chats: dict[int, dict[str, Any]] = {}
        self.topic_infos: dict[int, dict[str, Any]] = {}
        self.search_started = False
        self.authorization_ready = False
        self.search_pending_sources: set[str] = set()
        self.search_chat_ids: set[int] = set()
        self.pending_chat_inspections: set[int] = set()
        self.pending_topic_creations: dict[str, str] = {}
        self.topic_lookup_started = False
        self.topic_lookup_offsets: set[tuple[int, int, int]] = set()
        self.my_user_id: Optional[int] = None
        self.create_chat_after_get_me = False
        self.saved_chat_id: Optional[int] = self.load_saved_chat_id()
        self.chat_id: Optional[int] = self.saved_chat_id
        self.general_topic_id: Optional[int] = None
        self.pending_call_devices: dict[int, str] = {}
        self.active_calls: dict[int, CallMediaState] = {}
        self.wizards: dict[int, AddDeviceWizard] = {}
        self.pending_rtmp_device_id: Optional[str] = None
        self.pending_rtmp_group_call_id: Optional[int] = None
        self.active_rtmp: Optional[RtmpRuntime] = None
        self.schedule_runtimes: dict[str, ScheduleRuntime] = {}
        self.motion_runtimes: dict[str, MotionRuntime] = {}
        self.pending_initial_trial_license_id: Optional[str] = None
        self.client = Tam(
            phone_number=login,
            database_directory=str(script_dir / "tdlib"),
            files_directory=str(script_dir / "tdlib" / "files"),
            on_auth_code=self.on_auth_code,
            on_incoming_call=self.on_incoming_call,
            on_call_ended=self.on_call_ended,
            on_license_status=self.on_license_status,
            on_td_json=self.on_td_json,
        )

    def tr(self, key: str, **kwargs: object) -> str:
        return translation_text(self.ui, key, **kwargs)

    @property
    def group_title(self) -> str:
        return self.tr("group_title")

    def safe_print(self, text: str) -> None:
        with self.print_lock:
            print(text, flush=True)

    def write_debug_log(self, text: str) -> None:
        if not self.debug_log_enabled:
            return
        with self.debug_log_lock:
            with self.debug_log_path.open("a", encoding="utf-8") as file:
                file.write(str(text).rstrip("\r\n") + "\n")

    def load_saved_chat_id(self) -> Optional[int]:
        state = self.config.get(WATCHDOG_CHAT_STATE_KEY)
        return td_int(state.get("chat_id")) if isinstance(state, dict) else None

    def save_runtime_config(self) -> None:
        with self.lock:
            config = dict(self.config)
            state = dict(config.get(WATCHDOG_CHAT_STATE_KEY) or {})
            if self.chat_id is not None:
                state["chat_id"] = self.chat_id
            if self.general_topic_id is not None:
                state["general_topic_id"] = self.general_topic_id
            config[WATCHDOG_CHAT_STATE_KEY] = state
            self.config = config
            devices = [dict(device) for device in self.devices]
        save_config(self.config_path, self.login, self.script_dir, devices, existing_config=config)

    def td_send(self, request: dict[str, Any]) -> None:
        try:
            self.client.td_send_json(request)
        except RuntimeError as error:
            self.safe_print(f"[tdlib] send failed for {request.get('@type')}: {error}")

    def on_auth_code(self) -> str:
        return input("Telegram code: ").strip()

    def on_license_status(self, status: dict[str, Any]) -> None:
        if status.get("status") != "initial_trial_active":
            return
        license_id = str(status.get("license_id") or "").strip()
        if not license_id:
            return
        state = self.config.get(WATCHDOG_CHAT_STATE_KEY)
        if isinstance(state, dict) and state.get("initial_trial_notice_license_id") == license_id:
            return
        with self.lock:
            self.pending_initial_trial_license_id = license_id
        self.flush_initial_trial_notice()

    def flush_initial_trial_notice(self) -> None:
        with self.lock:
            license_id = self.pending_initial_trial_license_id
            chat_id = self.chat_id
            topic_id = self.general_topic_id
            if not license_id or chat_id is None or topic_id is None:
                return
            self.pending_initial_trial_license_id = None
        self.send_text(self.tr("initial_trial_activated"), topic_id=topic_id, op="send_initial_trial_notice")
        with self.lock:
            state = dict(self.config.get(WATCHDOG_CHAT_STATE_KEY) or {})
            state["initial_trial_notice_license_id"] = license_id
            self.config[WATCHDOG_CHAT_STATE_KEY] = state
        self.save_runtime_config()

    def run(self) -> None:
        require_pyav()
        self.client.start()
        self.safe_print("watchdog is running; press Ctrl+C to stop")
        if self.debug_log_enabled:
            self.safe_print(f"[watchdog] debug log: {self.debug_log_path}")
        try:
            while not self.stop_event.is_set():
                try:
                    self.poll_calls()
                except Exception as error:
                    self.safe_print(f"[watchdog] poll loop error: {error}\n{traceback.format_exc()}")
                self.stop_event.wait(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            self.safe_print("[watchdog] stopping")
            self.stop_event.set()
            self.stop_active_rtmp(notify=False)
            self.stop_schedule_workers()
            self.stop_motion_workers()
            with self.lock:
                call_states = list(self.active_calls.values())
            for state in call_states:
                state.stop.set()
            self.client.close()

    def on_td_json(self, text: str) -> None:
        try:
            self.handle_td_json(text)
        except Exception as error:
            self.safe_print(f"[watchdog] TDLib callback error: {error}\n{traceback.format_exc()}")

    def handle_td_json(self, text: str) -> None:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return
        object_type = str(data.get("@type") or "")
        if object_type == "updateAuthorizationState":
            state = data.get("authorization_state") or {}
            if isinstance(state, dict) and state.get("@type") == "authorizationStateReady":
                with self.lock:
                    self.authorization_ready = True
                self.request_watchdog_chat()
            return
        if object_type == "updateNewMessage":
            message = data.get("message")
            if isinstance(message, dict):
                self.handle_new_message(message)
            return
        if object_type == "updateForumTopicInfo":
            info = data.get("info")
            if isinstance(info, dict) and self.remember_topic_info(info):
                self.save_runtime_config()
            return
        if object_type == "updateDeleteMessages":
            self.handle_deleted_messages(data)
            return
        if object_type == "updateGroupCall":
            group_call = data.get("group_call")
            if isinstance(group_call, dict):
                self.handle_group_call_update(group_call)
            return
        if object_type == "updateChatTitle":
            chat_id = td_int(data.get("chat_id"))
            if chat_id is not None:
                with self.lock:
                    self.chats.setdefault(chat_id, {})["title"] = str(data.get("title") or "")
            return

        extra = data.get("@extra")
        if not isinstance(extra, dict):
            return
        op = str(extra.get("op") or "")
        handlers = {
            "inspect_saved_watchdog_chat": self.handle_saved_chat_inspection,
            "search_watchdog_chat": self.handle_search_result,
            "inspect_watchdog_chat": self.handle_chat_inspection,
            "inspect_watchdog_basic_group": self.handle_inspected_basic_group,
            "inspect_watchdog_supergroup": self.handle_inspected_supergroup,
            "open_watchdog_upgraded_supergroup": self.handle_opened_upgraded_supergroup,
            "create_watchdog_get_me": self.handle_create_get_me,
            "create_watchdog_chat": self.handle_created_basic_group,
            "upgrade_watchdog_chat": self.handle_upgraded_chat,
            "ensure_watchdog_chat": self.handle_ensure_watchdog_chat,
            "ensure_watchdog_forum": self.handle_ensure_watchdog_forum,
            "toggle_watchdog_forum": self.handle_toggle_watchdog_forum,
            "verify_watchdog_forum": self.handle_verify_watchdog_forum,
            "get_watchdog_topics": self.handle_forum_topics_response,
            "create_watchdog_topic": self.handle_created_forum_topic,
            "create_watchdog_call": self.handle_create_call_response,
            "create_watchdog_rtmp_chat": self.handle_create_rtmp_response,
            "get_watchdog_rtmp_url": self.handle_rtmp_url_response,
            "send_watchdog_media": self.handle_send_media_response,
            "set_watchdog_chat_photo": self.handle_set_chat_photo_response,
        }
        handler = handlers.get(op)
        if handler is not None:
            handler(data, extra)

    def handle_deleted_messages(self, data: dict[str, Any]) -> None:
        chat_id = td_int(data.get("chat_id"))
        with self.lock:
            watchdog_chat_id = self.chat_id
        if chat_id is None or chat_id != watchdog_chat_id:
            return
        if bool(data.get("from_cache")) or not bool(data.get("is_permanent")):
            return
        deleted_message_ids = {
            message_id
            for message_id in (td_int(value) for value in data.get("message_ids") or [])
            if message_id is not None
        }
        if deleted_message_ids:
            self.remove_devices_for_topic_ids(deleted_message_ids, source="Telegram update")

    def remove_devices_for_topic_ids(self, topic_ids: set[int], *, source: str) -> bool:
        if not topic_ids:
            return False
        removed: list[tuple[str, str, int]] = []
        call_states: list[CallMediaState] = []
        stop_active_rtmp = False
        pending_group_call_id: Optional[int] = None
        with self.lock:
            remaining_devices: list[dict[str, object]] = []
            for device in self.devices:
                topic = device.get("topic")
                topic_id = td_int(topic.get("id")) if isinstance(topic, dict) else None
                if topic_id not in topic_ids:
                    remaining_devices.append(device)
                    continue
                device_id = str(device.get("id") or "")
                display_name = str(device.get("display_name") or device.get("label") or device_id)
                removed.append((device_id, display_name, topic_id))
            if not removed:
                return False

            self.devices = remaining_devices
            removed_device_ids = {device_id for device_id, _, _ in removed}
            for device_id in removed_device_ids:
                self.pending_topic_creations.pop(device_id, None)
            for topic_id in topic_ids:
                self.topic_infos.pop(topic_id, None)
            for user_id, device_id in list(self.pending_call_devices.items()):
                if device_id in removed_device_ids:
                    self.pending_call_devices.pop(user_id, None)
            call_states = [
                state for state in self.active_calls.values() if state.device_id in removed_device_ids
            ]
            stop_active_rtmp = (
                self.active_rtmp is not None and self.active_rtmp.device_id in removed_device_ids
            )
            if self.pending_rtmp_device_id in removed_device_ids:
                pending_group_call_id = self.pending_rtmp_group_call_id
                self.pending_rtmp_device_id = None
                self.pending_rtmp_group_call_id = None

        for state in call_states:
            state.stop.set()
        self.save_runtime_config()
        if self.setup_done.is_set():
            self.sync_schedule_workers()
            self.sync_motion_workers()
        if stop_active_rtmp:
            threading.Thread(
                target=self.stop_active_rtmp,
                kwargs={"notify": False},
                name="watchdog-rtmp-topic-deleted",
                daemon=True,
            ).start()
        elif pending_group_call_id:
            self.td_send({"@type": "endGroupCall", "group_call_id": pending_group_call_id})
        for device_id, display_name, topic_id in removed:
            self.safe_print(
                f"[watchdog] removed device after topic deletion: "
                f"device_id={device_id}, name={display_name!r}, topic_id={topic_id}, source={source}"
            )
        return True

    def handle_group_call_update(self, group_call: dict[str, Any]) -> None:
        group_call_id = td_int(group_call.get("id"))
        if group_call_id is None or group_call.get("is_active") is not False:
            return
        with self.lock:
            runtime = self.active_rtmp
            pending_group_call_id = self.pending_rtmp_group_call_id
        if runtime is not None and runtime.group_call_id == group_call_id:
            self.safe_print(f"[watchdog] RTMP stream ended externally: group_call_id={group_call_id}")
            threading.Thread(
                target=self.stop_active_rtmp,
                kwargs={"topic_id": runtime.topic_id, "notify": True, "end_group_call": False},
                name="watchdog-rtmp-external-stop",
                daemon=True,
            ).start()
            return
        if pending_group_call_id == group_call_id:
            with self.lock:
                self.pending_rtmp_device_id = None
                self.pending_rtmp_group_call_id = None

    def cache_chat(self, chat: dict[str, Any]) -> None:
        chat_id = td_int(chat.get("id"))
        if chat_id is not None:
            with self.lock:
                self.chats[chat_id] = chat

    def request_watchdog_chat(self) -> None:
        with self.lock:
            if not self.authorization_ready or self.search_started:
                return
            self.search_started = True
            saved_chat_id = self.chat_id
        if saved_chat_id is not None:
            self.safe_print(f"[watchdog] checking saved group: chat_id={saved_chat_id}")
            self.td_send({"@type": "getChat", "@extra": {"op": "inspect_saved_watchdog_chat", "chat_id": saved_chat_id}, "chat_id": saved_chat_id})
            return
        self.start_chat_search()

    def start_chat_search(self) -> None:
        with self.lock:
            self.search_pending_sources = {"local", "server"}
            self.search_chat_ids.clear()
        self.safe_print("[watchdog] searching group")
        for method, source in (("searchChats", "local"), ("searchChatsOnServer", "server")):
            self.td_send(
                {
                    "@type": method,
                    "@extra": {"op": "search_watchdog_chat", "source": source},
                    "query": self.group_title,
                    "limit": 50,
                }
            )

    def handle_saved_chat_inspection(self, data: dict[str, Any], extra: dict[str, Any]) -> None:
        if data.get("@type") == "chat" and self.is_watchdog_chat(data):
            self.cache_chat(data)
            chat_type = data.get("type") or {}
            if isinstance(chat_type, dict) and chat_type.get("@type") == "chatTypeBasicGroup":
                self.upgrade_basic_group(int(data["id"]))
                return
            self.on_chat_ready(int(data["id"]))
            return
        with self.lock:
            self.chat_id = None
            self.search_started = True
        self.start_chat_search()

    def handle_search_result(self, data: dict[str, Any], extra: dict[str, Any]) -> None:
        source = str(extra.get("source") or "unknown")
        if data.get("@type") != "error":
            ids = [chat_id for chat_id in (td_int(item) for item in data.get("chat_ids") or []) if chat_id is not None]
            with self.lock:
                self.search_chat_ids.update(ids)
        with self.lock:
            self.search_pending_sources.discard(source)
            if self.search_pending_sources:
                return
            inspect_ids = sorted(self.search_chat_ids)
            self.pending_chat_inspections = set(inspect_ids)
        if not inspect_ids:
            self.create_watchdog_chat()
            return
        for chat_id in inspect_ids:
            self.td_send({"@type": "getChat", "@extra": {"op": "inspect_watchdog_chat", "chat_id": chat_id}, "chat_id": chat_id})

    def is_watchdog_chat(self, chat: dict[str, Any]) -> bool:
        if str(chat.get("title") or "") != self.group_title:
            return False
        chat_type = chat.get("type")
        if not isinstance(chat_type, dict):
            return False
        if chat_type.get("@type") == "chatTypeSupergroup":
            return not bool(chat_type.get("is_channel"))
        return chat_type.get("@type") == "chatTypeBasicGroup"

    def handle_chat_inspection(self, data: dict[str, Any], extra: dict[str, Any]) -> None:
        chat_id = int(extra.get("chat_id") or 0)
        if data.get("@type") == "chat":
            self.cache_chat(data)
            if self.is_watchdog_chat(data):
                chat_type = data.get("type") or {}
                type_name = chat_type.get("@type")
                if type_name == "chatTypeBasicGroup":
                    basic_group_id = td_int(chat_type.get("basic_group_id")) or basic_group_id_from_chat_id(chat_id)
                    if basic_group_id is not None:
                        self.td_send(
                            {
                                "@type": "getBasicGroup",
                                "@extra": {
                                    "op": "inspect_watchdog_basic_group",
                                    "chat_id": chat_id,
                                    "basic_group_id": basic_group_id,
                                },
                                "basic_group_id": basic_group_id,
                            }
                        )
                        return
                    self.safe_print(f"[watchdog] found '{self.group_title}', but can't determine basic_group_id")
                    self.finish_chat_inspection(chat_id)
                    return
                if type_name == "chatTypeSupergroup" and not bool(chat_type.get("is_channel")):
                    supergroup_id = td_int(chat_type.get("supergroup_id")) or supergroup_id_from_chat_id(chat_id)
                    if supergroup_id is not None:
                        self.td_send(
                            {
                                "@type": "getSupergroup",
                                "@extra": {"op": "inspect_watchdog_supergroup", "chat_id": chat_id},
                                "supergroup_id": supergroup_id,
                            }
                        )
                        return
        self.finish_chat_inspection(chat_id)

    def handle_inspected_supergroup(self, data: dict[str, Any], extra: dict[str, Any]) -> None:
        chat_id = int(extra.get("chat_id") or 0)
        if data.get("@type") == "supergroup" and not bool(data.get("is_channel")):
            self.finish_chat_inspection(chat_id, accepted_chat_id=chat_id)
            return
        self.finish_chat_inspection(chat_id)

    def handle_inspected_basic_group(self, data: dict[str, Any], extra: dict[str, Any]) -> None:
        chat_id = int(extra.get("chat_id") or 0)
        if data.get("@type") != "basicGroup":
            self.safe_print(f"[watchdog] failed to inspect basic group: {data.get('message') or data.get('@type')}")
            self.finish_chat_inspection(chat_id)
            return

        upgraded_to_supergroup_id = td_int(data.get("upgraded_to_supergroup_id"))
        if upgraded_to_supergroup_id:
            self.safe_print(f"[watchdog] opening upgraded supergroup: supergroup_id={upgraded_to_supergroup_id}")
            self.td_send(
                {
                    "@type": "createSupergroupChat",
                    "@extra": {"op": "open_watchdog_upgraded_supergroup", "source_chat_id": chat_id},
                    "supergroup_id": upgraded_to_supergroup_id,
                    "force": True,
                }
            )
            return

        member_count = int(data.get("member_count") or 0)
        if member_count <= 0:
            self.safe_print(f"[watchdog] ignoring empty '{self.group_title}' basic group")
            self.finish_chat_inspection(chat_id)
            return

        if bool(data.get("is_active")):
            self.upgrade_basic_group(chat_id)
            return

        self.safe_print(f"[watchdog] ignoring inactive '{self.group_title}' basic group")
        self.finish_chat_inspection(chat_id)

    def handle_opened_upgraded_supergroup(self, data: dict[str, Any], extra: dict[str, Any]) -> None:
        source_chat_id = int(extra.get("source_chat_id") or 0)
        if data.get("@type") == "chat" and self.is_watchdog_chat(data):
            self.cache_chat(data)
            self.finish_chat_inspection(source_chat_id, accepted_chat_id=int(data["id"]))
            return
        self.safe_print(f"[watchdog] failed to open upgraded supergroup: {data.get('message') or data.get('@type')}")
        self.finish_chat_inspection(source_chat_id)

    def finish_chat_inspection(self, chat_id: int, accepted_chat_id: Optional[int] = None) -> None:
        create_needed = False
        ready_chat_id: Optional[int] = None
        with self.lock:
            self.pending_chat_inspections.discard(chat_id)
            if accepted_chat_id is not None:
                self.chat_id = accepted_chat_id
                self.pending_chat_inspections.clear()
                ready_chat_id = accepted_chat_id
            elif not self.pending_chat_inspections and self.chat_id is None:
                create_needed = True
        if ready_chat_id is not None:
            self.on_chat_ready(ready_chat_id)
        elif create_needed:
            self.create_watchdog_chat()

    def create_watchdog_chat(self) -> None:
        with self.lock:
            my_user_id = self.my_user_id
            if my_user_id is None:
                self.create_chat_after_get_me = True
        if my_user_id is None:
            self.safe_print("[watchdog] loading current user before creating group")
            self.td_send({"@type": "getMe", "@extra": {"op": "create_watchdog_get_me"}})
            return
        self.safe_print(f"[watchdog] creating private basic group: {self.group_title}")
        self.td_send(
            {
                "@type": "createNewBasicGroupChat",
                "@extra": {"op": "create_watchdog_chat"},
                "user_ids": [my_user_id],
                "title": self.group_title,
                "message_auto_delete_time": 0,
            }
        )

    def handle_create_get_me(self, data: dict[str, Any], extra: dict[str, Any]) -> None:
        user_id = td_int(data.get("id"))
        if data.get("@type") != "user" or user_id is None:
            self.safe_print("[watchdog] failed to load current user")
            return
        with self.lock:
            self.my_user_id = user_id
            should_create = self.create_chat_after_get_me
            self.create_chat_after_get_me = False
        if should_create:
            self.create_watchdog_chat()

    def handle_created_basic_group(self, data: dict[str, Any], extra: dict[str, Any]) -> None:
        if data.get("@type") == "createdBasicGroupChat" and td_int(data.get("chat_id")) is not None:
            self.upgrade_basic_group(int(data["chat_id"]), created=True)
            return
        if data.get("@type") == "chat":
            self.cache_chat(data)
            self.upgrade_basic_group(int(data["id"]), created=True)
            return
        self.safe_print(f"[watchdog] failed to create group: {data.get('message') or data.get('@type')}")
        return

    def upgrade_basic_group(self, chat_id: int, *, created: bool = False) -> None:
        if not chat_id:
            return
        self.safe_print(f"[watchdog] upgrading group to supergroup: chat_id={chat_id}")
        self.td_send(
            {
                "@type": "upgradeBasicGroupChatToSupergroupChat",
                "@extra": {"op": "upgrade_watchdog_chat", "source_chat_id": chat_id, "created": created},
                "chat_id": chat_id,
            }
        )

    def handle_upgraded_chat(self, data: dict[str, Any], extra: dict[str, Any]) -> None:
        if data.get("@type") != "chat":
            self.safe_print(f"[watchdog] failed to upgrade group: {data.get('message') or data.get('@type')}")
            return
        self.cache_chat(data)
        source_chat_id = int(extra.get("source_chat_id") or 0)
        with self.lock:
            if source_chat_id:
                self.pending_chat_inspections.discard(source_chat_id)
            self.pending_chat_inspections.clear()
        self.on_chat_ready(int(data["id"]), created=bool(extra.get("created")))

    def on_chat_ready(self, chat_id: int, *, created: bool = False) -> None:
        with self.lock:
            self.chat_id = chat_id
        self.safe_print(f"[watchdog] group chat_id={chat_id}")
        self.save_runtime_config()
        state = self.config.get(WATCHDOG_CHAT_STATE_KEY)
        logo_version = td_int(state.get("logo_version")) if isinstance(state, dict) else None
        if created or logo_version != WATCHDOG_LOGO_VERSION:
            self.set_watchdog_chat_photo(chat_id)
        self.td_send({"@type": "openChat", "@extra": {"op": "open_watchdog_chat"}, "chat_id": chat_id})
        self.ensure_forum(chat_id)

    def set_watchdog_chat_photo(self, chat_id: int) -> None:
        logo_path = self.stuff_dir / "logo_512x512.png"
        if not logo_path.exists():
            self.safe_print(f"[watchdog] group logo not found: {logo_path}")
            return
        self.td_send(
            {
                "@type": "setChatPhoto",
                "@extra": {"op": "set_watchdog_chat_photo", "chat_id": chat_id},
                "chat_id": chat_id,
                "photo": {
                    "@type": "inputChatPhotoStatic",
                    "photo": input_file_local(logo_path.resolve()),
                },
            }
        )

    def handle_set_chat_photo_response(self, data: dict[str, Any], extra: dict[str, Any]) -> None:
        if data.get("@type") == "ok":
            with self.lock:
                state = dict(self.config.get(WATCHDOG_CHAT_STATE_KEY) or {})
                state["logo_version"] = WATCHDOG_LOGO_VERSION
                self.config[WATCHDOG_CHAT_STATE_KEY] = state
            self.save_runtime_config()
            self.safe_print("[watchdog] group logo updated")
        elif data.get("@type") == "error":
            self.safe_print(f"[watchdog] failed to set group logo: {data.get('message')}")

    def supergroup_id_for_chat(self, chat_id: int) -> Optional[int]:
        with self.lock:
            chat = self.chats.get(chat_id)
        chat_type = (chat or {}).get("type") or {}
        if isinstance(chat_type, dict) and chat_type.get("@type") == "chatTypeSupergroup" and not bool(chat_type.get("is_channel")):
            return td_int(chat_type.get("supergroup_id"))
        return supergroup_id_from_chat_id(chat_id)

    def ensure_forum(self, chat_id: int) -> None:
        supergroup_id = self.supergroup_id_for_chat(chat_id)
        if supergroup_id is None:
            self.td_send({"@type": "getChat", "@extra": {"op": "ensure_watchdog_chat", "chat_id": chat_id}, "chat_id": chat_id})
            return
        self.td_send(
            {
                "@type": "getSupergroup",
                "@extra": {"op": "ensure_watchdog_forum", "chat_id": chat_id, "supergroup_id": supergroup_id},
                "supergroup_id": supergroup_id,
            }
        )

    def handle_ensure_watchdog_chat(self, data: dict[str, Any], extra: dict[str, Any]) -> None:
        if data.get("@type") != "chat":
            self.safe_print(f"[watchdog] failed to load group: {data.get('message')}")
            return
        self.cache_chat(data)
        self.ensure_forum(int(data["id"]))

    def handle_ensure_watchdog_forum(self, data: dict[str, Any], extra: dict[str, Any]) -> None:
        chat_id = int(extra.get("chat_id") or 0)
        supergroup_id = int(extra.get("supergroup_id") or 0)
        if data.get("@type") != "supergroup" or bool(data.get("is_channel")):
            self.safe_print(f"[watchdog] failed to inspect supergroup: {data.get('message') or data.get('@type')}")
            return
        if bool(data.get("is_forum")):
            self.request_forum_topics(chat_id)
            return
        self.enable_forum(chat_id, supergroup_id, attempt=0)

    def enable_forum(self, chat_id: int, supergroup_id: int, *, attempt: int) -> None:
        self.td_send(
            {
                "@type": "toggleSupergroupIsForum",
                "@extra": {"op": "toggle_watchdog_forum", "chat_id": chat_id, "supergroup_id": supergroup_id, "attempt": attempt},
                "supergroup_id": supergroup_id,
                "is_forum": True,
                "has_forum_tabs": False,
            }
        )

    def handle_toggle_watchdog_forum(self, data: dict[str, Any], extra: dict[str, Any]) -> None:
        if data.get("@type") != "ok" and "not modified" not in str(data.get("message") or "").lower():
            self.safe_print(f"[watchdog] failed to enable forum topics: {data.get('message')}")
            return
        chat_id = int(extra.get("chat_id") or 0)
        supergroup_id = int(extra.get("supergroup_id") or 0)
        attempt = int(extra.get("attempt") or 0)
        timer = threading.Timer(
            FORUM_ENABLE_RETRY_DELAY_SECONDS,
            lambda: self.td_send(
                {
                    "@type": "getSupergroup",
                    "@extra": {"op": "verify_watchdog_forum", "chat_id": chat_id, "supergroup_id": supergroup_id, "attempt": attempt},
                    "supergroup_id": supergroup_id,
                }
            ),
        )
        timer.daemon = True
        timer.start()

    def handle_verify_watchdog_forum(self, data: dict[str, Any], extra: dict[str, Any]) -> None:
        chat_id = int(extra.get("chat_id") or 0)
        supergroup_id = int(extra.get("supergroup_id") or 0)
        attempt = int(extra.get("attempt") or 0)
        if data.get("@type") == "supergroup" and bool(data.get("is_forum")):
            self.request_forum_topics(chat_id)
            return
        if attempt + 1 < FORUM_ENABLE_MAX_ATTEMPTS:
            self.enable_forum(chat_id, supergroup_id, attempt=attempt + 1)
            return
        self.safe_print("[watchdog] forum topics weren't enabled")

    def request_forum_topics(self, chat_id: int) -> None:
        with self.lock:
            if self.topic_lookup_started:
                return
            self.topic_lookup_started = True
            self.topic_infos.clear()
            self.topic_lookup_offsets = {(0, 0, 0)}
        self.request_forum_topics_page(chat_id, 0, 0, 0)

    def request_forum_topics_page(
        self,
        chat_id: int,
        offset_date: int,
        offset_message_id: int,
        offset_forum_topic_id: int,
    ) -> None:
        self.td_send(
            {
                "@type": "getForumTopics",
                "@extra": {
                    "op": "get_watchdog_topics",
                    "chat_id": chat_id,
                    "offset_date": offset_date,
                    "offset_message_id": offset_message_id,
                    "offset_forum_topic_id": offset_forum_topic_id,
                },
                "chat_id": chat_id,
                "query": "",
                "offset_date": offset_date,
                "offset_message_id": offset_message_id,
                "offset_forum_topic_id": offset_forum_topic_id,
                "limit": 100,
            }
        )

    def handle_forum_topics_response(self, data: dict[str, Any], extra: dict[str, Any]) -> None:
        if data.get("@type") == "error":
            chat_id = int(extra.get("chat_id") or 0)
            message = str(data.get("message") or "")
            if chat_id and "not a forum" in message.lower():
                self.safe_print("[watchdog] chat is not a forum yet; retrying forum enable")
                supergroup_id = self.supergroup_id_for_chat(chat_id)
                if supergroup_id is not None:
                    with self.lock:
                        self.topic_lookup_started = False
                    self.enable_forum(chat_id, supergroup_id, attempt=0)
                    return
            self.safe_print(f"[watchdog] topics load failed: {data.get('message')}")
            return
        topics = [topic for topic in data.get("topics") or [] if isinstance(topic, dict)]
        for topic in topics:
            if isinstance(topic, dict) and isinstance(topic.get("info"), dict):
                self.remember_topic_info(topic["info"])
        current_offset = (
            td_int(extra.get("offset_date")) or 0,
            td_int(extra.get("offset_message_id")) or 0,
            td_int(extra.get("offset_forum_topic_id")) or 0,
        )
        next_offset = (
            td_int(data.get("next_offset_date")) or 0,
            td_int(data.get("next_offset_message_id")) or 0,
            td_int(data.get("next_offset_forum_topic_id")) or 0,
        )
        if topics and next_offset != current_offset:
            with self.lock:
                request_next_page = next_offset not in self.topic_lookup_offsets
                if request_next_page:
                    self.topic_lookup_offsets.add(next_offset)
            if request_next_page:
                self.request_forum_topics_page(int(extra.get("chat_id") or 0), *next_offset)
                return
            self.safe_print("[watchdog] topics load stopped because TDLib repeated a pagination offset")
            return
        chat_id = td_int(extra.get("chat_id"))
        self.sync_device_topics(remove_deleted_topics=chat_id is not None and chat_id == self.saved_chat_id)

    def remember_topic_info(self, info: dict[str, Any]) -> bool:
        topic_id = td_int(info.get("forum_topic_id"))
        if topic_id is None:
            return False
        with self.lock:
            info_chat_id = td_int(info.get("chat_id"))
            if info_chat_id is not None and self.chat_id is not None and info_chat_id != self.chat_id:
                return False
            self.topic_infos[topic_id] = dict(info)
            if bool(info.get("is_general")):
                self.general_topic_id = topic_id
            for device in self.devices:
                topic = device.get("topic")
                if isinstance(topic, dict) and td_int(topic.get("id")) == topic_id and not bool(info.get("is_general")):
                    name = str(info.get("name") or "").strip()
                    if name:
                        topic["name"] = name
                        device["display_name"] = name
        return True

    def sync_device_topics(self, *, remove_deleted_topics: bool = False) -> None:
        if remove_deleted_topics:
            with self.lock:
                deleted_topic_ids = {
                    topic_id
                    for device in self.devices
                    for topic in [device.get("topic")]
                    for topic_id in [td_int(topic.get("id")) if isinstance(topic, dict) else None]
                    if topic_id is not None and topic_id not in self.topic_infos
                }
            self.remove_devices_for_topic_ids(deleted_topic_ids, source="startup topic reconciliation")
        missing: dict[str, str] = {}
        changed = False
        with self.lock:
            if self.general_topic_id is None:
                self.general_topic_id = GENERAL_TOPIC_ID
            topic_by_name = {
                str(info.get("name") or "").casefold(): topic_id
                for topic_id, info in self.topic_infos.items()
                if not bool(info.get("is_general"))
            }
            for device in self.devices:
                device_id = str(device.get("id") or "")
                if not device_id:
                    continue
                display_name = str(device.get("display_name") or device.get("label") or device_id)
                topic = device.get("topic")
                topic_id = td_int(topic.get("id")) if isinstance(topic, dict) else None
                if topic_id is not None and topic_id in self.topic_infos:
                    info = self.topic_infos[topic_id]
                    name = str(info.get("name") or display_name).strip()
                    if name and (device.get("display_name") != name or not isinstance(topic, dict) or topic.get("name") != name):
                        device["display_name"] = name
                        device["topic"] = {"id": topic_id, "name": name}
                        changed = True
                    continue
                matched_topic_id = topic_by_name.get(display_name.casefold())
                if matched_topic_id is not None:
                    device["topic"] = {"id": matched_topic_id, "name": display_name}
                    changed = True
                    continue
                missing[device_id] = display_name
            self.pending_topic_creations.update(missing)
        if changed:
            self.save_runtime_config()
        if missing:
            for device_id, name in missing.items():
                self.create_forum_topic(device_id, name)
            return
        self.on_topics_ready()

    def create_forum_topic(self, device_id: str, name: str) -> None:
        with self.lock:
            chat_id = self.chat_id
        if chat_id is None:
            return
        self.safe_print(f"[watchdog] creating topic for {device_id}: {name}")
        self.td_send(
            {
                "@type": "createForumTopic",
                "@extra": {"op": "create_watchdog_topic", "device_id": device_id, "name": name},
                "chat_id": chat_id,
                "name": name,
                "is_name_implicit": False,
                "icon": forum_topic_icon(DEFAULT_FORUM_TOPIC_ICON_COLOR),
            }
        )

    def handle_created_forum_topic(self, data: dict[str, Any], extra: dict[str, Any]) -> None:
        device_id = str(extra.get("device_id") or "")
        name = str(extra.get("name") or "")
        if data.get("@type") == "forumTopicInfo":
            self.remember_topic_info(data)
            topic_id = td_int(data.get("forum_topic_id"))
            with self.lock:
                for device in self.devices:
                    if str(device.get("id") or "") == device_id and topic_id is not None:
                        device["topic"] = {"id": topic_id, "name": name}
                self.pending_topic_creations.pop(device_id, None)
            self.save_runtime_config()
            if topic_id is not None:
                self.send_text(self.tr("record_schedule_hint"), topic_id=topic_id)
        else:
            self.safe_print(f"[watchdog] topic creation failed for {name}: {data.get('message')}")
            with self.lock:
                self.pending_topic_creations.pop(device_id, None)
        with self.lock:
            pending = bool(self.pending_topic_creations)
        if not pending:
            self.on_topics_ready()

    def on_topics_ready(self) -> None:
        self.setup_done.set()
        self.save_runtime_config()
        self.safe_print("[watchdog] topics ready")
        state = self.config.get(WATCHDOG_CHAT_STATE_KEY)
        help_version = f"7:{self.language}"
        if not isinstance(state, dict) or state.get("help_version") != help_version:
            self.send_help()
            with self.lock:
                state = dict(self.config.get(WATCHDOG_CHAT_STATE_KEY) or {})
                state["help_version"] = help_version
                self.config[WATCHDOG_CHAT_STATE_KEY] = state
            self.save_runtime_config()
        self.flush_initial_trial_notice()
        self.sync_schedule_workers()
        self.sync_motion_workers()

    def send_message_content(self, content: dict[str, object], extra: dict[str, object], *, topic_id: Optional[int] = None) -> None:
        with self.lock:
            chat_id = self.chat_id
        if chat_id is None:
            return
        self.td_send(
            {
                "@type": "sendMessage",
                "@extra": extra,
                "chat_id": chat_id,
                "topic_id": message_topic_forum(topic_id),
                "reply_to": None,
                "options": None,
                "reply_markup": None,
                "input_message_content": content,
            }
        )

    def send_text(
        self,
        text: Union[str, dict[str, object]],
        *,
        topic_id: Optional[int] = None,
        op: str = "send_watchdog_text",
    ) -> None:
        message_text = text if isinstance(text, dict) else technical_message(text)
        self.send_message_content(input_message_text(message_text), {"op": op}, topic_id=topic_id)

    def send_help(self) -> None:
        with self.lock:
            topic_id = self.general_topic_id
        self.send_text(help_formatted_text(self.ui), topic_id=topic_id)

    def topic_id_for_device(self, device: dict[str, object]) -> Optional[int]:
        topic = device.get("topic")
        return td_int(topic.get("id")) if isinstance(topic, dict) else None

    def handle_record_schedule_command(self, text: str, device: dict[str, object], topic_id: Optional[int]) -> bool:
        parsed = parse_record_command(text)
        if parsed is None:
            return False
        args, motion_mode = parsed
        device_id = str(device.get("id") or "")
        if args in {"", "help", "?"}:
            self.send_text(record_schedule_help_formatted_text(self.ui), topic_id=topic_id)
            return True
        if args == "show":
            self.send_text(
                format_record_schedule(schedule_spec_from_device(device), schedule_motion_from_device(device), self.ui),
                topic_id=topic_id,
            )
            return True
        if args == "clear":
            with self.lock:
                for item in self.devices:
                    if str(item.get("id") or "") == device_id:
                        item.pop("record_schedule", None)
                        break
            self.save_runtime_config()
            self.sync_schedule_workers()
            self.sync_motion_workers()
            self.send_text(self.tr("record_schedule_removed"), topic_id=topic_id)
            return True
        try:
            parse_record_schedule_spec(args)
        except ValueError as error:
            self.send_text(self.tr("record_schedule_bad", error=error), topic_id=topic_id)
            return True
        spec = normalize_schedule_spec(args)
        with self.lock:
            for item in self.devices:
                if str(item.get("id") or "") == device_id:
                    item["record_schedule"] = {"spec": spec, "motion": motion_mode}
                    break
        self.save_runtime_config()
        self.sync_schedule_workers()
        self.sync_motion_workers()
        self.send_text(format_record_schedule(spec, motion_mode, self.ui), topic_id=topic_id)
        return True

    def sync_schedule_workers(self) -> None:
        to_stop: list[ScheduleRuntime] = []
        to_start: list[threading.Thread] = []
        with self.lock:
            desired: dict[str, tuple[str, bool]] = {}
            for device in self.devices:
                device_id = str(device.get("id") or "")
                spec = normalize_schedule_spec(schedule_spec_from_device(device))
                motion_mode = schedule_motion_from_device(device)
                if not device_id or not spec:
                    continue
                try:
                    rules = parse_record_schedule_spec(spec)
                    if not any(rule.get("ranges") for rule in rules):
                        continue
                except ValueError as error:
                    self.safe_print(f"[watchdog] invalid recording schedule for {device_id}: {error}")
                    continue
                desired[device_id] = (spec, motion_mode)

            for device_id, runtime in list(self.schedule_runtimes.items()):
                if desired.get(device_id) != (runtime.spec, runtime.motion):
                    runtime.stop.set()
                    self.schedule_runtimes.pop(device_id, None)
                    to_stop.append(runtime)

            for device_id, (spec, motion_mode) in desired.items():
                if device_id in self.schedule_runtimes:
                    continue
                stop = threading.Event()
                thread = threading.Thread(
                    target=self.schedule_worker,
                    args=(device_id, spec, motion_mode, stop),
                    name=f"watchdog-schedule-{device_id}",
                    daemon=True,
                )
                self.schedule_runtimes[device_id] = ScheduleRuntime(device_id, spec, motion_mode, stop, thread)
                to_start.append(thread)

        for runtime in to_stop:
            runtime.thread.join(timeout=2.0)
        for thread in to_start:
            thread.start()

    def stop_schedule_workers(self) -> None:
        with self.lock:
            runtimes = list(self.schedule_runtimes.values())
            self.schedule_runtimes.clear()
        for runtime in runtimes:
            runtime.stop.set()
        for runtime in runtimes:
            runtime.thread.join(timeout=2.0)

    def schedule_worker(self, device_id: str, spec: str, motion_mode: bool, stop: threading.Event) -> None:
        while not stop.is_set() and not self.stop_event.is_set():
            now = datetime.now().astimezone()
            try:
                interval = current_schedule_interval(spec, now)
                next_start = next_schedule_start(spec, now)
            except ValueError as error:
                self.safe_print(f"[watchdog] schedule worker stopped for {device_id}: {error}")
                return
            if interval is not None and interval[1] > now:
                if motion_mode:
                    self.run_motion_scheduled_recording(device_id, interval[1], stop)
                else:
                    self.run_scheduled_recording(device_id, interval[1], stop)
                continue
            if next_start is None:
                stop.wait(60.0)
                continue
            wait_seconds = max(1.0, min(60.0, (next_start - now).total_seconds()))
            stop.wait(wait_seconds)

    def run_scheduled_recording(self, device_id: str, interval_end: datetime, stop: threading.Event) -> None:
        while not stop.is_set() and not self.stop_event.is_set() and datetime.now().astimezone() < interval_end:
            device = self.device_by_id(device_id)
            if device is None:
                return
            video, audio = self.device_capture(device)
            topic_id = self.topic_id_for_device(device)
            if video is None:
                self.send_text(self.tr("scheduled_recording_no_video"), topic_id=topic_id)
                return
            started_at = datetime.now().astimezone()
            timestamp = started_at.strftime("%Y%m%d_%H%M%S")
            path = self.media_dir / f"schedule_{device_id}_{timestamp}.mp4"
            self.safe_print(f"[watchdog] scheduled recording started: {path}")
            result = record_device_scheduled(
                video,
                audio,
                path,
                interval_end,
                self.video_size,
                SCHEDULE_RECORDING_MAX_BYTES,
                stop,
            )
            if result is None:
                if not stop.is_set() and not self.stop_event.is_set():
                    self.safe_print(f"[watchdog] scheduled recording failed for {device_id}: no media")
                return
            self.safe_print(
                f"[watchdog] scheduled recording finished: {result.recording.path}, "
                f"duration={result.recording.duration_seconds}s, remuxed={result.remuxed}"
            )
            self.upload_scheduled_recording(device, result, topic_id)
            if result.recording.path.exists() and result.recording.path.stat().st_size < SCHEDULE_RECORDING_MAX_BYTES:
                return

    def collect_motion_events(
        self,
        device_id: str,
        video: CaptureDevice,
        stop: threading.Event,
        output: "queue.Queue[tuple[str, object]]",
    ) -> None:
        try:
            def log_motion(text: str) -> None:
                self.safe_print(f"[watchdog] {text}")
                if self.debug_log_enabled:
                    self.write_debug_log(debug_line("motion_detector", device_id=device_id, message=text))

            for event in iter_motion_events(
                video.input_name,
                format_name=video.format_name,
                options=dict(video.options),
                config=MotionDetectorConfig(),
                object_detector=self.yolo_detector,
                presence_interval_seconds=YOLO_PRESENCE_INTERVAL_SECONDS,
                presence_misses_to_stop=YOLO_PRESENCE_MISSES_TO_STOP,
                stop=stop,
                log=log_motion,
            ):
                if stop.is_set() or self.stop_event.is_set():
                    break
                output.put((event.kind, event))
            output.put(("detector_end", None))
        except Exception as error:
            output.put(("detector_error", error))

    def run_motion_scheduled_recording(self, device_id: str, interval_end: datetime, stop: threading.Event) -> None:
        device = self.device_by_id(device_id)
        if device is None:
            return
        video, audio = self.device_capture(device)
        topic_id = self.topic_id_for_device(device)
        if video is None:
            self.send_text(self.tr("motion_recording_no_video"), topic_id=topic_id)
            return

        if is_rtsp_capture(video):
            def log_motion(text: str) -> None:
                self.safe_print(f"[watchdog] {text}")
                if self.debug_log_enabled:
                    self.write_debug_log(debug_line("motion_detector", device_id=device_id, message=text))

            def upload_segment(result: ScheduledRecordingResult) -> None:
                current_device = self.device_by_id(device_id) or device
                current_topic_id = self.topic_id_for_device(current_device)
                self.safe_print(
                    f"[watchdog] motion recording finished: {result.recording.path}, "
                    f"duration={result.recording.duration_seconds}s, remuxed={result.remuxed}"
                )
                self.upload_scheduled_recording(current_device, result, current_topic_id)

            try:
                record_verified_motion_stream(
                    video,
                    self.media_dir,
                    device_id,
                    interval_end,
                    SCHEDULE_RECORDING_MAX_BYTES,
                    stop,
                    upload_segment,
                    self.yolo_detector,
                    log=log_motion,
                )
            except Exception as error:
                self.safe_print(
                    f"[watchdog] verified motion recording failed for {device_id}: "
                    f"{error}\n{traceback.format_exc()}"
                )
            if (
                not stop.is_set()
                and not self.stop_event.is_set()
                and datetime.now().astimezone() < interval_end
            ):
                stop.wait(5.0)
            return

        detector_stop = threading.Event()
        event_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        detector_thread = threading.Thread(
            target=self.collect_motion_events,
            args=(device_id, video, detector_stop, event_queue),
            name=f"watchdog-motion-schedule-detector-{device_id}",
            daemon=True,
        )
        detector_thread.start()

        recording_stop: Optional[threading.Event] = None
        recording_thread: Optional[threading.Thread] = None
        recording_holder: dict[str, object] = {}
        motion_active = False
        segment_index = 0

        def interval_active() -> bool:
            return not stop.is_set() and not self.stop_event.is_set() and datetime.now().astimezone() < interval_end

        def start_segment() -> None:
            nonlocal recording_stop, recording_thread, recording_holder, segment_index
            if recording_thread is not None:
                return
            current_device = self.device_by_id(device_id)
            if current_device is None:
                return
            current_video, current_audio = self.device_capture(current_device)
            if current_video is None:
                return
            segment_index += 1
            started_at = datetime.now().astimezone()
            timestamp = started_at.strftime("%Y%m%d_%H%M%S")
            path = self.media_dir / f"motion_{device_id}_{timestamp}_{segment_index}.mp4"
            recording_stop = threading.Event()
            recording_holder = {"path": path}

            def record_worker() -> None:
                try:
                    recording_holder["result"] = record_device_scheduled(
                        current_video,
                        current_audio,
                        path,
                        interval_end,
                        self.video_size,
                        SCHEDULE_RECORDING_MAX_BYTES,
                        recording_stop,
                    )
                except Exception as error:
                    recording_holder["error"] = error

            self.safe_print(f"[watchdog] motion recording started: {path}")
            recording_thread = threading.Thread(
                target=record_worker,
                name=f"watchdog-motion-record-{device_id}",
                daemon=True,
            )
            recording_thread.start()

        def finish_segment(*, request_stop: bool) -> Optional[ScheduledRecordingResult]:
            nonlocal recording_stop, recording_thread, recording_holder
            if recording_thread is None:
                return None
            if request_stop and recording_stop is not None:
                recording_stop.set()
            recording_thread.join()
            result = recording_holder.get("result")
            error = recording_holder.get("error")
            path = recording_holder.get("path")
            recording_thread = None
            recording_stop = None
            recording_holder = {}
            if error is not None:
                self.safe_print(f"[watchdog] motion recording failed for {device_id}: {error}")
                return None
            if not isinstance(result, ScheduledRecordingResult):
                self.safe_print(f"[watchdog] motion recording produced no media for {device_id}: {path}")
                return None
            current_device = self.device_by_id(device_id) or device
            current_topic_id = self.topic_id_for_device(current_device)
            self.safe_print(
                f"[watchdog] motion recording finished: {result.recording.path}, "
                f"duration={result.recording.duration_seconds}s, remuxed={result.remuxed}"
            )
            self.upload_scheduled_recording(current_device, result, current_topic_id)
            return result

        try:
            while interval_active():
                timeout = 1.0
                try:
                    event_kind, event_payload = event_queue.get(timeout=timeout)
                except queue.Empty:
                    event_kind, event_payload = "", None

                if event_kind == "motion_start":
                    motion_active = True
                    if recording_thread is None:
                        start_segment()
                elif event_kind == "motion_stop":
                    motion_active = False
                    if recording_thread is not None:
                        finish_segment(request_stop=True)
                elif event_kind == "detector_error":
                    self.safe_print(f"[watchdog] motion schedule detector failed for {device_id}: {event_payload}")
                    if recording_thread is not None:
                        finish_segment(request_stop=True)
                    return
                elif event_kind == "detector_end":
                    if recording_thread is not None:
                        finish_segment(request_stop=True)
                    if interval_active():
                        self.safe_print(f"[watchdog] motion schedule detector ended for {device_id}; restarting")
                        detector_stop.set()
                        detector_thread.join(timeout=2.0)
                        detector_stop = threading.Event()
                        detector_thread = threading.Thread(
                            target=self.collect_motion_events,
                            args=(device_id, video, detector_stop, event_queue),
                            name=f"watchdog-motion-schedule-detector-{device_id}",
                            daemon=True,
                        )
                        detector_thread.start()
                    continue

                if recording_thread is not None and not recording_thread.is_alive():
                    result = finish_segment(request_stop=False)
                    size_limited = (
                        result is not None
                        and result.recording.path.exists()
                        and result.recording.path.stat().st_size >= SCHEDULE_RECORDING_MAX_BYTES
                    )
                    if size_limited and motion_active and interval_active():
                        start_segment()
                    elif result is None:
                        return

            if recording_thread is not None:
                finish_segment(request_stop=True)
        finally:
            detector_stop.set()
            detector_thread.join(timeout=2.0)
            if recording_thread is not None:
                finish_segment(request_stop=True)

    def upload_scheduled_recording(
        self,
        device: dict[str, object],
        result: ScheduledRecordingResult,
        topic_id: Optional[int],
    ) -> None:
        recording = result.recording
        caption = scheduled_recording_caption(device, result)
        self.send_message_content(
            input_message_video(recording.path.resolve(), caption, recording.duration_seconds, recording.width, recording.height),
            {"op": "send_watchdog_scheduled_recording", "path": str(recording.path)},
            topic_id=topic_id,
        )

    def set_device_motion_enabled(self, device_id: str, enabled: bool) -> bool:
        changed = False
        with self.lock:
            for item in self.devices:
                if str(item.get("id") or "") == device_id:
                    if bool(item.get("motion_enabled")) != enabled:
                        item["motion_enabled"] = enabled
                        changed = True
                    break
        if changed:
            self.save_runtime_config()
        return changed

    def set_device_schedule_motion_enabled(self, device_id: str, enabled: bool) -> bool:
        changed = False
        with self.lock:
            for item in self.devices:
                if str(item.get("id") or "") != device_id:
                    continue
                schedule = item.get("record_schedule")
                if not isinstance(schedule, dict) or not str(schedule.get("spec") or "").strip():
                    break
                updated = dict(schedule)
                if bool(updated.get("motion")) != enabled:
                    updated["motion"] = enabled
                    item["record_schedule"] = updated
                    changed = True
                break
        if changed:
            self.save_runtime_config()
        return changed

    def handle_motion_command(self, normalized: str, device: dict[str, object], topic_id: Optional[int]) -> bool:
        if normalized not in {"motion on", "motion 1", "m1", "motion off", "motion 0", "m0"}:
            return False
        enabled = normalized in {"motion on", "motion 1", "m1"}
        device_id = str(device.get("id") or "")
        if schedule_spec_from_device(device):
            self.set_device_schedule_motion_enabled(device_id, enabled)
            self.sync_schedule_workers()
            self.send_text(
                self.tr("motion_schedule_enabled" if enabled else "motion_schedule_disabled"),
                topic_id=topic_id,
            )
            return True
        self.set_device_motion_enabled(device_id, enabled)
        self.sync_motion_workers()
        self.send_text(self.tr("motion_detection_enabled" if enabled else "motion_detection_disabled"), topic_id=topic_id)
        return True

    def sync_motion_workers(self) -> None:
        to_stop: list[MotionRuntime] = []
        to_start: list[threading.Thread] = []
        with self.lock:
            desired: dict[str, Optional[int]] = {}
            for device in self.devices:
                if schedule_spec_from_device(device):
                    continue
                if not bool(device.get("motion_enabled")):
                    continue
                device_id = str(device.get("id") or "")
                if not device_id:
                    continue
                if capture_from_json(device.get("video")) is None:
                    continue
                desired[device_id] = self.topic_id_for_device(device)

            for device_id, runtime in list(self.motion_runtimes.items()):
                if device_id not in desired or desired.get(device_id) != runtime.topic_id:
                    runtime.stop.set()
                    self.motion_runtimes.pop(device_id, None)
                    to_stop.append(runtime)

            for device_id, topic_id in desired.items():
                if device_id in self.motion_runtimes:
                    continue
                stop = threading.Event()
                thread = threading.Thread(
                    target=self.motion_worker,
                    args=(device_id, topic_id, stop),
                    name=f"watchdog-motion-{device_id}",
                    daemon=True,
                )
                self.motion_runtimes[device_id] = MotionRuntime(device_id, topic_id, stop, thread)
                to_start.append(thread)

        for runtime in to_stop:
            runtime.thread.join(timeout=2.0)
        for thread in to_start:
            thread.start()

    def stop_motion_workers(self) -> None:
        with self.lock:
            runtimes = list(self.motion_runtimes.values())
            self.motion_runtimes.clear()
        for runtime in runtimes:
            runtime.stop.set()
        for runtime in runtimes:
            runtime.thread.join(timeout=2.0)

    def motion_worker(self, device_id: str, topic_id: Optional[int], stop: threading.Event) -> None:
        notified_error = False
        while not stop.is_set() and not self.stop_event.is_set():
            device = self.device_by_id(device_id)
            if device is None:
                return
            video, _ = self.device_capture(device)
            if video is None:
                self.safe_print(f"[watchdog] motion detector stopped for {device_id}: no video source")
                return
            try:
                self.safe_print(f"[watchdog] motion detector started for {device_id}: {video.label}")

                def log_motion(text: str) -> None:
                    self.safe_print(f"[watchdog] {text}")
                    if self.debug_log_enabled:
                        self.write_debug_log(debug_line("motion_detector", device_id=device_id, message=text))

                for event in iter_motion_events(
                    video.input_name,
                    format_name=video.format_name,
                    options=dict(video.options),
                    config=MotionDetectorConfig(),
                    object_detector=self.yolo_detector,
                    presence_interval_seconds=YOLO_PRESENCE_INTERVAL_SECONDS,
                    presence_misses_to_stop=YOLO_PRESENCE_MISSES_TO_STOP,
                    stop=stop,
                    log=log_motion,
                ):
                    if stop.is_set() or self.stop_event.is_set():
                        break
                    current_device = self.device_by_id(device_id)
                    current_topic_id = topic_id if current_device is None else self.topic_id_for_device(current_device)
                    event_key = "motion_start" if event.kind == "motion_start" else "motion_stop"
                    self.send_text(self.tr(event_key), topic_id=current_topic_id)
                if stop.is_set() or self.stop_event.is_set():
                    return
                self.safe_print(f"[watchdog] motion detector stream ended for {device_id}; reconnecting")
                stop.wait(5.0)
            except Exception as error:
                if stop.is_set() or self.stop_event.is_set():
                    return
                message = f"Motion detector error for {device_id}: {error}"
                self.safe_print(f"[watchdog] {message}\n{traceback.format_exc()}")
                if not notified_error:
                    self.send_text(self.tr("motion_detection_stopped", error=error), topic_id=topic_id)
                    notified_error = True
                stop.wait(5.0)

    def handle_send_media_response(self, data: dict[str, Any], extra: dict[str, Any]) -> None:
        if data.get("@type") == "error":
            self.safe_print(f"[watchdog] failed to send media: {data.get('message')}")

    def device_by_id(self, device_id: str) -> Optional[dict[str, object]]:
        with self.lock:
            for device in self.devices:
                if str(device.get("id") or "") == device_id:
                    return device
        return None

    def device_by_topic_id(self, topic_id: Optional[int]) -> Optional[dict[str, object]]:
        if topic_id is None:
            return None
        with self.lock:
            for device in self.devices:
                topic = device.get("topic")
                if isinstance(topic, dict) and td_int(topic.get("id")) == topic_id:
                    return device
        return None

    def device_capture(self, device: dict[str, object]) -> tuple[Optional[CaptureDevice], Optional[CaptureDevice]]:
        video = capture_from_json(device.get("video"))
        if is_rtsp_capture(video):
            return video, None
        return video, capture_from_json(device.get("audio"))

    def handle_new_message(self, message: dict[str, Any]) -> None:
        chat_id = td_int(message.get("chat_id"))
        with self.lock:
            watchdog_chat_id = self.chat_id
        if watchdog_chat_id is None or chat_id != watchdog_chat_id:
            return
        text = message_text(message).strip()
        if not text or is_technical_message_text(text):
            return
        self.mark_message_read(message)
        topic_id = forum_topic_id_from_message(message)
        with self.lock:
            general_topic_id = self.general_topic_id or GENERAL_TOPIC_ID
        if topic_id == general_topic_id:
            self.handle_general_message(message, text)
            return
        device = self.device_by_topic_id(topic_id)
        if device is not None:
            self.handle_device_message(message, text, device)

    def mark_message_read(self, message: dict[str, Any]) -> None:
        chat_id = td_int(message.get("chat_id"))
        message_id = td_int(message.get("id"))
        if chat_id is None or message_id is None:
            return
        self.td_send(
            {
                "@type": "viewMessages",
                "@extra": {"op": "view_watchdog_message"},
                "chat_id": chat_id,
                "message_ids": [message_id],
                "source": None,
                "force_read": True,
            }
        )

    def handle_general_message(self, message: dict[str, Any], text: str) -> None:
        user_id = message_sender_user_id(message)
        if user_id is None:
            return
        if user_id in self.wizards:
            self.handle_wizard_message(user_id, text)
            return
        normalized = normalize_command_text(text)
        if normalized in {"help", "h"}:
            self.send_help()
            return
        if normalized in {"devices", "list", "ls"}:
            self.send_text(self.devices_text(), topic_id=self.general_topic_id)
            return
        if normalized in {"cancel", "stop"}:
            self.wizards.pop(user_id, None)
            self.send_text(self.tr("no_active_wizard"), topic_id=self.general_topic_id)
            return
        if normalized in {"add webcam", "webcam", "add camera", "camera"}:
            self.start_add_source_wizard(user_id, "webcam")
            return
        if normalized in {"add screen", "screen", "display", "add display"}:
            self.start_add_source_wizard(user_id, "screen")
            return
        if normalized.startswith("add rtsp") or normalized.startswith("rtsp "):
            url = text.split(maxsplit=2)[-1].strip()
            self.start_rtsp_wizard(user_id, url)

    def devices_text(self) -> str:
        with self.lock:
            devices = list(self.devices)
        if not devices:
            return self.tr("no_devices")
        lines = [self.tr("devices_title")]
        for device in devices:
            topic = device.get("topic") if isinstance(device.get("topic"), dict) else {}
            lines.append(
                self.tr(
                    "device_list_item",
                    id=device.get("id"),
                    name=device.get("display_name"),
                    topic_id=topic.get("id"),
                )
            )
        return "\n".join(lines)

    def start_add_source_wizard(self, user_id: int, kind: str) -> None:
        kind_name = self.tr(f"device_kind_{kind}")
        try:
            candidates = enumerate_video_devices() if kind == "webcam" else enumerate_screen_devices()
        except Exception as error:
            self.send_text(
                self.tr("enumerate_devices_failed", kind=kind_name, error=error),
                topic_id=self.general_topic_id,
            )
            return
        selected = existing_source_machine_ids(self.devices, kind)
        candidates = [item for item in candidates if item.input_name not in selected]
        if not candidates:
            self.send_text(self.tr("no_available_devices", kind=kind_name), topic_id=self.general_topic_id)
            return
        self.wizards[user_id] = AddDeviceWizard(user_id, "select_video", kind, candidates=candidates)
        lines = [self.tr("select_device", kind=kind_name)]
        lines.extend(f"{index}. {device.label}" for index, device in enumerate(candidates, start=1))
        lines.append(self.tr("cancel_hint"))
        self.send_text("\n".join(lines), topic_id=self.general_topic_id)

    def start_rtsp_wizard(self, user_id: int, url: str) -> None:
        parts = urlsplit(url)
        if parts.scheme.lower() != "rtsp" or not parts.hostname:
            self.send_text(self.tr("rtsp_usage"), topic_id=self.general_topic_id)
            return
        video = CaptureDevice(self.tr("rtsp_device_label", url=redact_url(url)), url, "rtsp", rtsp_options())
        wizard = AddDeviceWizard(user_id, "done", "rtsp", video=video)
        self.commit_wizard_device(user_id, wizard, None)

    def handle_wizard_message(self, user_id: int, text: str) -> None:
        wizard = self.wizards.get(user_id)
        if wizard is None:
            return
        normalized = normalize_command_text(text)
        if normalized == "cancel":
            self.wizards.pop(user_id, None)
            self.send_text(self.tr("wizard_cancelled"), topic_id=self.general_topic_id)
            return
        if wizard.step == "select_video":
            try:
                index = int(text.strip())
                video = wizard.candidates[index - 1]
            except Exception:
                self.send_text(self.tr("invalid_device_number"), topic_id=self.general_topic_id)
                return
            wizard.video = video
            if wizard.kind == "rtsp":
                self.commit_wizard_device(user_id, wizard, None)
                return
            wizard.step = "ask_audio"
            self.send_text(self.tr("add_microphone"), topic_id=self.general_topic_id)
            return
        if wizard.step == "ask_audio":
            if normalized in {"n", "no", "нет"}:
                self.commit_wizard_device(user_id, wizard, None)
                return
            if normalized not in {"y", "yes", "да"}:
                self.send_text(self.tr("answer_yes_no"), topic_id=self.general_topic_id)
                return
            microphones = enumerate_audio_devices()
            if not microphones:
                self.send_text(self.tr("no_microphones"), topic_id=self.general_topic_id)
                self.commit_wizard_device(user_id, wizard, None)
                return
            wizard.audio_candidates = microphones
            wizard.step = "select_audio"
            lines = [self.tr("select_microphone")]
            lines.extend(f"{index}. {device.label}" for index, device in enumerate(microphones, start=1))
            lines.append(self.tr("no_microphone_option"))
            self.send_text("\n".join(lines), topic_id=self.general_topic_id)
            return
        if wizard.step == "select_audio":
            try:
                index = int(text.strip())
            except ValueError:
                self.send_text(self.tr("invalid_microphone_number"), topic_id=self.general_topic_id)
                return
            if index == 0:
                self.commit_wizard_device(user_id, wizard, None)
                return
            if not (1 <= index <= len(wizard.audio_candidates)):
                self.send_text(self.tr("invalid_microphone_number"), topic_id=self.general_topic_id)
                return
            self.commit_wizard_device(user_id, wizard, wizard.audio_candidates[index - 1])

    def commit_wizard_device(self, user_id: int, wizard: AddDeviceWizard, audio: Optional[CaptureDevice]) -> None:
        if wizard.video is None:
            self.send_text(self.tr("no_video_selected"), topic_id=self.general_topic_id)
            self.wizards.pop(user_id, None)
            return
        with self.lock:
            device = {
                "id": next_camera_id(self.devices),
                "kind": wizard.kind,
                "display_name": wizard.video.label,
                "label": wizard.video.label,
                "video": capture_to_json(wizard.video),
                "audio": capture_to_json(audio) if audio is not None else None,
            }
            self.devices.append(device)
            self.wizards.pop(user_id, None)
        self.save_runtime_config()
        self.send_text(
            self.tr("device_added", name=device["display_name"], hint=self.tr("record_schedule_hint")),
            topic_id=self.general_topic_id,
        )
        self.sync_device_topics()

    def handle_device_message(self, message: dict[str, Any], text: str, device: dict[str, object]) -> None:
        user_id = message_sender_user_id(message)
        topic_id = forum_topic_id_from_message(message)
        if user_id is None:
            return
        normalized = normalize_command_text(text)
        if normalized in {"call", "c"}:
            self.start_private_call(user_id, device, topic_id)
            return
        if normalized in {"screen", "s"}:
            self.send_snapshot(device, topic_id)
            return
        if self.handle_record_schedule_command(text, device, topic_id):
            return
        if self.handle_motion_command(normalized, device, topic_id):
            return
        if normalized in {"stream on", "stream 1", "s1"}:
            self.start_rtmp(device, topic_id)
            return
        if normalized in {"stream off", "stream 0", "s0"}:
            self.stop_active_rtmp(topic_id=topic_id)

    def start_private_call(self, user_id: int, device: dict[str, object], topic_id: Optional[int]) -> None:
        device_id = str(device.get("id") or "")
        with self.lock:
            self.pending_call_devices[user_id] = device_id
        self.td_send(
            {
                "@type": "createCall",
                "@extra": {"op": "create_watchdog_call", "user_id": user_id, "device_id": device_id, "topic_id": topic_id},
                "user_id": user_id,
                "protocol": CALL_PROTOCOL,
                "is_video": True,
            }
        )
        self.send_text(self.tr("calling", name=device.get("display_name")), topic_id=topic_id)

    def handle_create_call_response(self, data: dict[str, Any], extra: dict[str, Any]) -> None:
        if data.get("@type") == "error":
            user_id = int(extra.get("user_id") or 0)
            with self.lock:
                self.pending_call_devices.pop(user_id, None)
            self.send_text(
                self.tr("call_failed", error=data.get("message")),
                topic_id=td_int(extra.get("topic_id")),
            )

    def on_incoming_call(self, call_id: int, user_id: int, username: str, is_video: bool) -> None:
        with self.lock:
            device_id = self.pending_call_devices.pop(user_id, None)
        if device_id is None:
            self.safe_print(f"[watchdog] stopping unrequested call: call_id={call_id}, user_id={user_id}")
            try:
                self.client.stop_call(call_id)
            except RuntimeError:
                pass
            return
        device = self.device_by_id(device_id)
        if device is None:
            return
        video, audio = self.device_capture(device)
        if video is None:
            return
        state = CallMediaState(
            threading.Event(),
            threading.Event(),
            threading.Event(),
            device_id,
            time.monotonic_ns() // 1_000,
            call_id=call_id,
            user_id=user_id,
        )
        with self.lock:
            self.active_calls[call_id] = state
        if is_rtsp_capture(video):
            self.write_debug_log(
                debug_line(
                    "rtsp_call_requested",
                    call_id=call_id,
                    user_id=user_id,
                    username=username,
                    device_id=device_id,
                    device_name=device.get("display_name"),
                    rtsp_url=redact_url(video.input_name),
                    video_size=self.video_size,
                )
            )
            threading.Thread(
                target=stream_rtsp_call_device,
                args=(
                    self.client,
                    call_id,
                    video,
                    state,
                    self.video_size,
                    self.write_debug_log if self.debug_log_enabled else None,
                ),
                daemon=True,
            ).start()
        else:
            threading.Thread(target=stream_video_device, args=(self.client, call_id, video, state, self.video_size), daemon=True).start()
        if audio is not None and not is_rtsp_capture(video):
            threading.Thread(target=stream_audio_device, args=(self.client, call_id, audio, state), daemon=True).start()
        elif not is_rtsp_capture(video):
            state.audio_done.set()

    def on_call_ended(self, call_id: int, reason: str) -> None:
        self.safe_print(f"[watchdog] call ended: call_id={call_id}, reason={reason}")
        self.write_debug_log(debug_line("call_ended", call_id=call_id, reason=reason))
        with self.lock:
            state = self.active_calls.pop(call_id, None)
        if state is not None:
            state.stop.set()

    def poll_calls(self) -> None:
        with self.lock:
            states = list(self.active_calls.items())
        for call_id, state in states:
            if state.stop_requested or not (state.video_done.is_set() and state.audio_done.is_set()):
                continue
            try:
                pending_ms = self.client.outgoing_media_pending_ms(call_id)
            except RuntimeError:
                pending_ms = 0
            if pending_ms > 0:
                state.drain_started_at = None
                continue
            now = time.monotonic()
            if state.drain_started_at is None:
                state.drain_started_at = now
                continue
            if now - state.drain_started_at < CALL_DRAIN_GRACE_SECONDS:
                continue
            state.stop_requested = True
            state.stop.set()
            try:
                self.client.stop_call(call_id)
            except RuntimeError:
                pass

    def send_snapshot(self, device: dict[str, object], topic_id: Optional[int]) -> None:
        video, _ = self.device_capture(device)
        if video is None:
            self.send_text(self.tr("device_no_video"), topic_id=topic_id)
            return

        def worker() -> None:
            path = self.media_dir / f"snapshot_{device.get('id')}_{int(time.time())}.jpg"
            try:
                width, height = capture_snapshot(video, path, self.video_size)
                caption = self.tr(
                    "snapshot_caption",
                    name=device.get("display_name"),
                    time=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                )
                self.send_message_content(input_message_photo(path.resolve(), caption, width, height), {"op": "send_watchdog_media"}, topic_id=topic_id)
            except Exception as error:
                self.send_text(self.tr("snapshot_failed", error=error), topic_id=topic_id)

        threading.Thread(target=worker, name="watchdog-snapshot", daemon=True).start()

    def send_recording(self, device: dict[str, object], topic_id: Optional[int], duration_seconds: int) -> None:
        video, audio = self.device_capture(device)
        if video is None:
            self.send_text(self.tr("device_no_video"), topic_id=topic_id)
            return

        def worker() -> None:
            try:
                timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
                path = self.media_dir / f"record_{device.get('id')}_{timestamp}.mp4"
                self.safe_print(f"[watchdog] recording started: {path}")
                self.send_text(self.tr("recording_progress", seconds=duration_seconds), topic_id=topic_id)
                result = record_device(video, audio, path, duration_seconds, self.video_size)
                if result is None:
                    self.safe_print("[watchdog] recording failed: no media was captured")
                    self.send_text(self.tr("recording_no_media"), topic_id=topic_id)
                    return
                self.safe_print(
                    f"[watchdog] recording finished: {result.path}, duration={result.duration_seconds}s, "
                    f"video={result.has_video}, audio={result.has_audio}"
                )
                caption = self.tr(
                    "recording_caption",
                    name=device.get("display_name"),
                    time=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                )
                self.send_message_content(
                    input_message_video(result.path.resolve(), caption, result.duration_seconds, result.width, result.height),
                    {"op": "send_watchdog_media"},
                    topic_id=topic_id,
                )
            except Exception as error:
                self.safe_print(f"[watchdog] recording worker failed: {error}\n{traceback.format_exc()}")
                self.send_text(self.tr("recording_failed", error=error), topic_id=topic_id)

        threading.Thread(target=worker, name="watchdog-record", daemon=True).start()

    def start_rtmp(self, device: dict[str, object], topic_id: Optional[int]) -> None:
        video, _ = self.device_capture(device)
        if video is None:
            self.send_text(self.tr("device_no_video"), topic_id=topic_id)
            return
        with self.lock:
            if self.active_rtmp is not None or self.pending_rtmp_device_id is not None:
                self.send_text(self.tr("rtmp_already_active"), topic_id=topic_id)
                return
            chat_id = self.chat_id
            self.pending_rtmp_device_id = str(device.get("id") or "")
        if chat_id is None:
            return
        self.td_send(
            {
                "@type": "createVideoChat",
                "@extra": {"op": "create_watchdog_rtmp_chat", "device_id": device.get("id"), "topic_id": topic_id},
                "chat_id": chat_id,
                "title": str(device.get("display_name") or "Watchdog stream")[:64],
                "start_date": 0,
                "is_rtmp_stream": True,
            }
        )
        self.send_text(self.tr("rtmp_starting"), topic_id=topic_id)

    def handle_create_rtmp_response(self, data: dict[str, Any], extra: dict[str, Any]) -> None:
        if data.get("@type") == "error":
            with self.lock:
                self.pending_rtmp_device_id = None
            self.send_text(
                self.tr("rtmp_create_failed", error=data.get("message")),
                topic_id=td_int(extra.get("topic_id")),
            )
            return
        group_call_id = td_int(data.get("id"))
        with self.lock:
            self.pending_rtmp_group_call_id = group_call_id
            chat_id = self.chat_id
        if chat_id is None:
            return
        self.td_send(
            {
                "@type": "getVideoChatRtmpUrl",
                "@extra": {
                    "op": "get_watchdog_rtmp_url",
                    "device_id": extra.get("device_id"),
                    "topic_id": extra.get("topic_id"),
                    "group_call_id": group_call_id,
                },
                "chat_id": chat_id,
            }
        )

    def handle_rtmp_url_response(self, data: dict[str, Any], extra: dict[str, Any]) -> None:
        topic_id = td_int(extra.get("topic_id"))
        device_id = str(extra.get("device_id") or "")
        group_call_id = int(extra.get("group_call_id") or 0)
        if data.get("@type") == "error":
            with self.lock:
                self.pending_rtmp_device_id = None
                self.pending_rtmp_group_call_id = None
            self.send_text(self.tr("rtmp_url_failed", error=data.get("message")), topic_id=topic_id)
            return
        device = self.device_by_id(device_id)
        video, audio = self.device_capture(device or {})
        if device is None or video is None:
            return
        output_url = join_rtmp_url(str(data.get("url") or ""), str(data.get("stream_key") or ""))
        self.write_debug_log(
            debug_line(
                "rtmp_url_received",
                device_id=device_id,
                device_name=device.get("display_name"),
                group_call_id=group_call_id,
                topic_id=topic_id,
                video_label=video.label,
                video_format=video.format_name,
                video_input=redact_url(video.input_name) if is_rtsp_capture(video) else video.input_name,
                audio_label=None if audio is None else audio.label,
                audio_format=None if audio is None else audio.format_name,
                output_url=redact_rtmp_url(output_url),
                video_size=DEFAULT_RTMP_VIDEO_SIZE,
                rtmp_fps=DEFAULT_RTMP_VIDEO_FPS,
            )
        )
        stop = threading.Event()

        def worker() -> None:
            try:
                stream_device_to_rtmp(
                    video,
                    audio,
                    output_url,
                    DEFAULT_RTMP_VIDEO_SIZE,
                    stop,
                    fps=DEFAULT_RTMP_VIDEO_FPS,
                    debug_log=self.write_debug_log if self.debug_log_enabled else None,
                    debug_context={
                        "device_id": device_id,
                        "device_name": device.get("display_name"),
                        "group_call_id": group_call_id,
                        "topic_id": topic_id,
                    },
                )
            finally:
                self.write_debug_log(
                    debug_line(
                        "rtmp_worker_finished",
                        device_id=device_id,
                        device_name=device.get("display_name"),
                        group_call_id=group_call_id,
                        topic_id=topic_id,
                    )
                )
                with self.lock:
                    if self.active_rtmp and self.active_rtmp.device_id == device_id:
                        self.active_rtmp = None
                    self.pending_rtmp_device_id = None
                    self.pending_rtmp_group_call_id = None

        thread = threading.Thread(target=worker, name=f"watchdog-rtmp-{device_id}", daemon=True)
        runtime = RtmpRuntime(device_id, group_call_id, topic_id, stop, thread)
        with self.lock:
            self.active_rtmp = runtime
            self.pending_rtmp_device_id = None
        thread.start()
        self.send_text(self.tr("rtmp_started"), topic_id=topic_id)

    def stop_active_rtmp(
        self,
        *,
        topic_id: Optional[int] = None,
        notify: bool = True,
        end_group_call: bool = True,
    ) -> None:
        with self.lock:
            runtime = self.active_rtmp
            self.active_rtmp = None
            self.pending_rtmp_device_id = None
            self.pending_rtmp_group_call_id = None
        if runtime is not None:
            self.write_debug_log(
                debug_line(
                    "rtmp_stop_requested",
                    device_id=runtime.device_id,
                    group_call_id=runtime.group_call_id,
                    topic_id=topic_id if topic_id is not None else runtime.topic_id,
                    end_group_call=end_group_call,
                )
            )
            runtime.stop.set()
            runtime.thread.join(timeout=5.0)
            self.write_debug_log(
                debug_line(
                    "rtmp_stop_joined",
                    device_id=runtime.device_id,
                    group_call_id=runtime.group_call_id,
                    alive=runtime.thread.is_alive(),
                )
            )
            if end_group_call and runtime.group_call_id:
                self.td_send({"@type": "endGroupCall", "group_call_id": runtime.group_call_id})
        if notify:
            self.send_text(self.tr("rtmp_stopped"), topic_id=topic_id)


def run_setup(args: argparse.Namespace) -> None:
    require_pyav()
    script_dir = Path(__file__).resolve().parent
    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else script_dir / "media" / "stuff" / "watchdog_config.json"
    )
    login = resolve_login(args)
    existing_config = load_existing_config(config_path)
    existing_devices = existing_config_devices(existing_config)
    if existing_devices:
        print(f"Existing config has {len(existing_devices)} device(s); new devices will be appended.")

    run_authorization_setup(login, script_dir)
    new_devices = choose_video_sources(existing_devices)
    devices = existing_devices + new_devices
    save_config(config_path, login, script_dir, devices, existing_config=existing_config)
    print(f"Added {len(new_devices)} device(s). Total devices: {len(devices)}")
    print(f"Configuration saved: {config_path}")


def run_watchdog(args: argparse.Namespace) -> None:
    require_pyav()
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else script_dir / "media" / "stuff" / "watchdog_config.json"
    )
    config = load_existing_config(config_path)
    login = (args.login or str(config.get("login") or "")).strip()
    if not login:
        login = resolve_login(args)
    app = WatchdogApp(login, script_dir, config_path, args.video_size, args.lang, args.debug_log)
    app.run()


def run_portable_self_test() -> None:
    """Exercise dependencies and native libraries without creating user state."""

    script_dir = Path(__file__).resolve().parent
    print("[portable-self-test] runtime", flush=True)
    print(f"  executable: {sys.executable}", flush=True)
    print(f"  script_dir: {script_dir}", flush=True)
    print(f"  cwd: {Path.cwd()}", flush=True)
    print(f"  platform: {sys.platform}", flush=True)
    print(f"  python: {sys.version}", flush=True)

    def check(name: str, action: Callable[[], object]) -> object:
        print(f"[portable-self-test] checking {name}...", flush=True)
        try:
            result = action()
        except BaseException as error:
            print(
                f"[portable-self-test] FAILED {name}: {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()
            raise
        print(f"[portable-self-test] OK {name}", flush=True)
        return result

    check("PyAV import and native libraries", require_pyav)
    model_path = script_dir / "media" / "stuff" / YOLO_MODEL_FILENAME
    detector = YoloObjectDetector(model_path)

    from object_detector import require_numpy

    numpy = check("NumPy import and native libraries", require_numpy)
    check(
        f"OpenCV DNN and YOLO model ({model_path})",
        lambda: detector.detect(numpy.zeros((64, 64, 3), dtype="uint8")),
    )

    if sys.platform == "win32":
        def import_windows_capture() -> object:
            from windows_capture import WindowsCapture

            return WindowsCapture

        check("Windows Capture import and native libraries", import_windows_capture)

    with tempfile.TemporaryDirectory(prefix="watchdog-self-test-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        def check_libtam() -> None:
            tam = Tam(
                files_directory=temporary_path / "files",
                database_directory=temporary_path / "database",
            )
            tam.close()

        check("LibTam and TDLib native libraries", check_libtam)

    print("Watchdog portable self-test: OK", flush=True)


def main() -> None:
    configure_console_utf8()
    args = parse_args()
    if args.portable_self_test:
        run_portable_self_test()
        return
    if args.setup:
        run_setup(args)
        return
    run_watchdog(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[watchdog] fatal error: {error}\n{traceback.format_exc()}", flush=True)
        raise
