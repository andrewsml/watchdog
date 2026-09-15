"""Cascaded motion detector for H.264/H.265 CCTV streams.

The inexpensive first stage uses FFmpeg/PyAV exported motion vectors.  A
motion-vector start candidate is checked by OpenCV MOG2 over small color
samples retained from the already decoded GOP, then optionally confirmed and
kept active by a low-rate YOLO presence detector. Frames are never decoded a
second time for the pixel-domain checks.
"""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

from object_detector import (
    YOLO_PRESENCE_SAMPLE_COUNT,
    YOLO_PRESENCE_SAMPLE_INTERVAL_SECONDS,
    YoloPresenceTracker,
    detect_presence_samples,
)


av = None
cv2 = None
np = None


@dataclass(frozen=True)
class MotionZone:
    """Normalized rectangle. Values are in [0, 1]."""

    x: float
    y: float
    width: float
    height: float
    enabled: bool = True


@dataclass(frozen=True)
class MotionDetectorConfig:
    grid_cols: int = 16
    grid_rows: int = 12
    zones: tuple[MotionZone, ...] = ()
    min_vector_magnitude_px: float = 1.5
    min_vectors_per_cell: int = 2
    active_cell_min_magnitude: float = 0.0025
    start_active_cells: float = 6.0
    stop_active_cells: float = 2.0
    start_frames: int = 4
    stop_frames: int = 16
    smoothing_alpha: float = 0.35
    global_active_ratio: float = 0.65
    global_direction_coherence: float = 0.82
    global_motion_weight: float = 0.25
    max_analyze_fps: float = 12.0
    no_vector_warning_frames: int = 120


@dataclass(frozen=True)
class PixelMotionConfig:
    """Low-resolution confirmation settings for motion-vector candidates."""

    sample_every_frames: int = 5
    max_sample_width: int = 320
    min_samples: int = 4
    max_samples_per_gop: int = 120
    previous_gop_samples: int = 2
    post_candidate_samples: int = 2
    warmup_samples: int = 1
    mog2_history: int = 40
    mog2_var_threshold: float = 24.0
    mog2_detect_shadows: bool = True
    mog2_shadow_threshold: float = 0.55
    blur_kernel: int = 5
    open_kernel: int = 3
    close_kernel: int = 5
    min_contour_area_ratio: float = 0.002
    min_foreground_ratio: float = 0.003
    max_foreground_ratio: float = 0.45
    confirm_samples: int = 2
    global_luma_delta: float = 18.0
    global_change_pixel_delta: int = 25
    global_change_ratio: float = 0.55
    reject_cooldown_seconds: float = 3.0
    max_motion_regions: int = 32


@dataclass(frozen=True)
class PixelMotionResult:
    motion: bool
    reason: str
    analyzed_samples: int
    motion_samples: int
    scene_change_samples: int
    max_foreground_ratio: float
    max_global_change_ratio: float
    motion_regions: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class MotionFrameStats:
    frame_index: int
    width: int
    height: int
    vector_count: int
    filtered_vector_count: int
    active_cells: float
    smoothed_active_cells: float
    active_ratio: float
    direction_coherence: float
    global_motion: bool
    score: float
    timestamp: float


@dataclass(frozen=True)
class MotionEvent:
    kind: str
    stats: MotionFrameStats
    verification: Optional[PixelMotionResult] = None


class MotionVectorDetector:
    def __init__(self, config: Optional[MotionDetectorConfig] = None) -> None:
        self.config = config or MotionDetectorConfig()
        self.motion_active = False
        self.start_count = 0
        self.stop_count = 0
        self.smoothed_active_cells = 0.0
        self.frame_index = 0
        self.last_stats: Optional[MotionFrameStats] = None

    def process_frame(self, frame) -> Optional[MotionEvent]:
        self.frame_index += 1
        vectors = motion_vectors_from_frame(frame)
        stats = self._frame_stats(frame, vectors)
        self.last_stats = stats
        if not self.motion_active:
            if stats.smoothed_active_cells >= self.config.start_active_cells:
                self.start_count += 1
            else:
                self.start_count = 0
            if self.start_count >= self.config.start_frames:
                self.motion_active = True
                self.stop_count = 0
                return MotionEvent("motion_start", stats)
            return None

        if stats.smoothed_active_cells <= self.config.stop_active_cells:
            self.stop_count += 1
        else:
            self.stop_count = 0
        if self.stop_count >= self.config.stop_frames:
            self.motion_active = False
            self.start_count = 0
            return MotionEvent("motion_stop", stats)
        return None

    def reject_motion(self) -> None:
        """Return the first-stage detector to idle after failed confirmation."""

        self.motion_active = False
        self.start_count = 0
        self.stop_count = 0
        self.smoothed_active_cells = 0.0

    def _frame_stats(self, frame, vectors) -> MotionFrameStats:
        numpy = require_numpy()
        width = max(1, int(getattr(frame, "width", 0) or 0))
        height = max(1, int(getattr(frame, "height", 0) or 0))
        timestamp = time.time()

        if vectors is None or len(vectors) == 0:
            self.smoothed_active_cells *= 1.0 - self.config.smoothing_alpha
            return MotionFrameStats(
                self.frame_index,
                width,
                height,
                0,
                0,
                0.0,
                self.smoothed_active_cells,
                0.0,
                0.0,
                False,
                0.0,
                timestamp,
            )

        dx, dy, dst_x, dst_y = normalized_vectors(vectors, width, height)
        magnitude_px = numpy.sqrt(dx * dx + dy * dy)
        keep = magnitude_px >= self.config.min_vector_magnitude_px
        if self.config.zones:
            keep &= zone_mask(dst_x, dst_y, width, height, self.config.zones)

        filtered = int(numpy.count_nonzero(keep))
        if filtered == 0:
            self.smoothed_active_cells *= 1.0 - self.config.smoothing_alpha
            return MotionFrameStats(
                self.frame_index,
                width,
                height,
                int(len(vectors)),
                0,
                0.0,
                self.smoothed_active_cells,
                0.0,
                0.0,
                False,
                0.0,
                timestamp,
            )

        dx = dx[keep]
        dy = dy[keep]
        dst_x = dst_x[keep]
        dst_y = dst_y[keep]
        magnitude_px = magnitude_px[keep]
        magnitude_norm = magnitude_px / max(1.0, float((width * width + height * height) ** 0.5))

        grid = aggregate_grid(
            dst_x,
            dst_y,
            magnitude_norm,
            width,
            height,
            self.config.grid_cols,
            self.config.grid_rows,
        )
        active_grid = (grid["count"] >= self.config.min_vectors_per_cell) & (
            grid["mean"] >= self.config.active_cell_min_magnitude
        )
        active_cells = float(numpy.count_nonzero(active_grid))
        zone_cells = zone_cell_count(width, height, self.config)
        active_ratio = active_cells / max(1.0, zone_cells)

        mean_dx = float(numpy.mean(dx))
        mean_dy = float(numpy.mean(dy))
        mean_mag = float(numpy.mean(magnitude_px))
        direction_coherence = 0.0 if mean_mag <= 0.0 else min(1.0, ((mean_dx * mean_dx + mean_dy * mean_dy) ** 0.5) / mean_mag)
        global_motion = (
            active_ratio >= self.config.global_active_ratio
            and direction_coherence >= self.config.global_direction_coherence
        )
        weighted_active_cells = active_cells * (self.config.global_motion_weight if global_motion else 1.0)
        self.smoothed_active_cells = (
            self.config.smoothing_alpha * weighted_active_cells
            + (1.0 - self.config.smoothing_alpha) * self.smoothed_active_cells
        )
        score = float(numpy.sum(grid["mean"][active_grid]))
        if global_motion:
            score *= self.config.global_motion_weight

        return MotionFrameStats(
            self.frame_index,
            width,
            height,
            int(len(vectors)),
            filtered,
            weighted_active_cells,
            self.smoothed_active_cells,
            active_ratio,
            direction_coherence,
            global_motion,
            score,
            timestamp,
        )


def require_pyav():
    global av
    if av is not None:
        return av
    try:
        import av as pyav
    except ImportError as error:
        raise RuntimeError("PyAV is required for motion detection: python -m pip install av") from error
    av = pyav
    return av


def require_numpy():
    global np
    if np is not None:
        return np
    try:
        import numpy as numpy_module
    except ImportError as error:
        raise RuntimeError("numpy is required for motion detection: python -m pip install numpy") from error
    np = numpy_module
    return np


def require_opencv():
    global cv2
    if cv2 is not None:
        return cv2
    try:
        import cv2 as opencv_module
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required for motion confirmation: "
            "python -m pip install opencv-python-headless"
        ) from error
    cv2 = opencv_module
    return cv2


def codec_supports_motion_vectors(codec_name: str) -> bool:
    return str(codec_name or "").casefold() in {"h264", "hevc", "h265"}


def motion_vectors_from_frame(frame):
    for side_data in getattr(frame, "side_data", []) or []:
        if "MOTION" not in str(getattr(side_data, "type", "")).upper():
            continue
        to_ndarray = getattr(side_data, "to_ndarray", None)
        if callable(to_ndarray):
            vectors = to_ndarray()
            if vectors is not None and len(vectors):
                return vectors
    return None


def normalized_vectors(vectors, width: int, height: int):
    numpy = require_numpy()
    scale = vectors["motion_scale"].astype("float32")
    scale = numpy.where(scale == 0.0, 1.0, scale)
    dx = vectors["motion_x"].astype("float32") / scale
    dy = vectors["motion_y"].astype("float32") / scale
    dst_x = numpy.clip(vectors["dst_x"].astype("float32"), 0.0, max(0.0, width - 1.0))
    dst_y = numpy.clip(vectors["dst_y"].astype("float32"), 0.0, max(0.0, height - 1.0))
    return dx, dy, dst_x, dst_y


def zone_mask(dst_x, dst_y, width: int, height: int, zones: tuple[MotionZone, ...]):
    numpy = require_numpy()
    mask = numpy.zeros(dst_x.shape, dtype=bool)
    for zone in zones:
        if not zone.enabled:
            continue
        x0 = max(0.0, min(1.0, zone.x)) * width
        y0 = max(0.0, min(1.0, zone.y)) * height
        x1 = max(0.0, min(1.0, zone.x + zone.width)) * width
        y1 = max(0.0, min(1.0, zone.y + zone.height)) * height
        mask |= (dst_x >= x0) & (dst_x < x1) & (dst_y >= y0) & (dst_y < y1)
    return mask


def aggregate_grid(dst_x, dst_y, magnitude, width: int, height: int, cols: int, rows: int):
    numpy = require_numpy()
    cols = max(1, int(cols))
    rows = max(1, int(rows))
    gx = numpy.clip((dst_x * cols / max(1, width)).astype("int32"), 0, cols - 1)
    gy = numpy.clip((dst_y * rows / max(1, height)).astype("int32"), 0, rows - 1)
    flat = gy * cols + gx
    cell_count = numpy.zeros(rows * cols, dtype="int32")
    cell_sum = numpy.zeros(rows * cols, dtype="float32")
    numpy.add.at(cell_count, flat, 1)
    numpy.add.at(cell_sum, flat, magnitude.astype("float32"))
    cell_mean = numpy.divide(cell_sum, cell_count, out=numpy.zeros_like(cell_sum), where=cell_count > 0)
    return {
        "count": cell_count.reshape(rows, cols),
        "sum": cell_sum.reshape(rows, cols),
        "mean": cell_mean.reshape(rows, cols),
    }


def zone_cell_count(width: int, height: int, config: MotionDetectorConfig) -> float:
    if not config.zones:
        return float(max(1, config.grid_cols * config.grid_rows))
    cells = 0.0
    for zone in config.zones:
        if not zone.enabled:
            continue
        cells += max(1.0, zone.width * config.grid_cols) * max(1.0, zone.height * config.grid_rows)
    return max(1.0, min(float(config.grid_cols * config.grid_rows), cells))


def export_motion_vector_options(options: Optional[dict[str, str]] = None) -> dict[str, str]:
    result = {str(key): str(value) for key, value in (options or {}).items()}
    flags2 = result.get("flags2", "")
    if "export_mvs" not in flags2:
        result["flags2"] = f"{flags2}+export_mvs" if flags2 else "+export_mvs"
    return result


def frame_bgr_sample(frame, max_width: int = 640):
    """Create a small BGR copy from an already decoded PyAV frame."""

    numpy = require_numpy()
    source_width = max(1, int(getattr(frame, "width", 0) or 0))
    source_height = max(1, int(getattr(frame, "height", 0) or 0))
    target_width = min(source_width, max(32, int(max_width)))
    target_height = max(2, int(round(source_height * target_width / source_width)))
    target_width -= target_width % 2
    target_height -= target_height % 2
    sample_frame = frame.reformat(width=max(2, target_width), height=max(2, target_height), format="bgr24")
    return numpy.ascontiguousarray(sample_frame.to_ndarray()).copy()


def frame_motion_sample(frame, max_width: int = 320):
    """Retain color so MOG2 can distinguish chromatic objects from shadows."""

    return frame_bgr_sample(frame, max_width)


def frame_gray_sample(frame, max_width: int = 320):
    """Backward-compatible grayscale sampler used by external callers."""

    numpy = require_numpy()
    color = frame_bgr_sample(frame, max_width)
    return numpy.ascontiguousarray(
        require_opencv().cvtColor(color, require_opencv().COLOR_BGR2GRAY)
    )


def _odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def verify_gray_motion(
    samples,
    config: Optional[PixelMotionConfig] = None,
) -> PixelMotionResult:
    """Run MOG2 and contour-area checks over retained grayscale samples."""

    settings = config or PixelMotionConfig()
    opencv = require_opencv()
    numpy = require_numpy()
    frames = [numpy.asarray(sample, dtype="uint8") for sample in samples if sample is not None]
    if len(frames) < settings.min_samples:
        return PixelMotionResult(False, "insufficient_samples", len(frames), 0, 0, 0.0, 0.0, ())

    subtractor = opencv.createBackgroundSubtractorMOG2(
        history=max(settings.mog2_history, len(frames)),
        varThreshold=float(settings.mog2_var_threshold),
        detectShadows=bool(settings.mog2_detect_shadows),
    )
    set_shadow_threshold = getattr(subtractor, "setShadowThreshold", None)
    if callable(set_shadow_threshold):
        set_shadow_threshold(float(settings.mog2_shadow_threshold))
    open_kernel = opencv.getStructuringElement(
        opencv.MORPH_ELLIPSE,
        (_odd_kernel(settings.open_kernel), _odd_kernel(settings.open_kernel)),
    )
    close_kernel = opencv.getStructuringElement(
        opencv.MORPH_ELLIPSE,
        (_odd_kernel(settings.close_kernel), _odd_kernel(settings.close_kernel)),
    )

    previous_luma = None
    motion_samples = 0
    scene_change_samples = 0
    max_foreground_ratio = 0.0
    max_global_change_ratio = 0.0
    scored_samples = 0
    motion_regions = []
    for index, frame in enumerate(frames):
        blurred = opencv.GaussianBlur(
            frame,
            (_odd_kernel(settings.blur_kernel), _odd_kernel(settings.blur_kernel)),
            0,
        )
        luma = (
            blurred
            if blurred.ndim == 2
            else opencv.cvtColor(blurred, opencv.COLOR_BGR2GRAY)
        )
        global_luma_delta = 0.0
        global_change_ratio = 0.0
        if previous_luma is not None:
            difference = opencv.absdiff(luma, previous_luma)
            global_luma_delta = abs(float(numpy.mean(luma)) - float(numpy.mean(previous_luma)))
            global_change_ratio = float(
                numpy.count_nonzero(difference >= settings.global_change_pixel_delta)
            ) / float(max(1, difference.size))
            max_global_change_ratio = max(max_global_change_ratio, global_change_ratio)
        previous_luma = luma

        learning_rate = 0.5 if index < settings.warmup_samples else 0.02
        foreground = subtractor.apply(blurred, learningRate=learning_rate)
        if index < settings.warmup_samples:
            continue
        scored_samples += 1
        _, foreground = opencv.threshold(foreground, 200, 255, opencv.THRESH_BINARY)
        foreground = opencv.morphologyEx(foreground, opencv.MORPH_OPEN, open_kernel)
        foreground = opencv.morphologyEx(foreground, opencv.MORPH_CLOSE, close_kernel)

        is_scene_change = (
            global_luma_delta >= settings.global_luma_delta
            and global_change_ratio >= settings.global_change_ratio
        )
        if is_scene_change:
            scene_change_samples += 1
            continue

        contours_result = opencv.findContours(foreground, opencv.RETR_EXTERNAL, opencv.CHAIN_APPROX_SIMPLE)
        contours = contours_result[-2]
        frame_area = float(max(1, foreground.size))
        min_contour_area = frame_area * settings.min_contour_area_ratio
        valid_contours = []
        contour_area = 0.0
        for contour in contours:
            area = float(opencv.contourArea(contour))
            if area < min_contour_area:
                continue
            valid_contours.append((area, contour))
            contour_area += area
        foreground_ratio = contour_area / frame_area
        max_foreground_ratio = max(max_foreground_ratio, foreground_ratio)
        if settings.min_foreground_ratio <= foreground_ratio <= settings.max_foreground_ratio:
            motion_samples += 1
            height, width = foreground.shape[:2]
            for _, contour in valid_contours:
                x, y, box_width, box_height = opencv.boundingRect(contour)
                motion_regions.append(
                    (
                        x / max(1, width),
                        y / max(1, height),
                        box_width / max(1, width),
                        box_height / max(1, height),
                    )
                )

    if scene_change_samples:
        reason = "global_scene_change"
        confirmed = False
    else:
        confirmed = motion_samples >= settings.confirm_samples
        reason = "motion_confirmed" if confirmed else "foreground_not_persistent"
    if confirmed:
        motion_regions = sorted(
            motion_regions,
            key=lambda box: box[2] * box[3],
            reverse=True,
        )[: max(1, settings.max_motion_regions)]
    else:
        motion_regions = []
    return PixelMotionResult(
        confirmed,
        reason,
        scored_samples,
        motion_samples,
        scene_change_samples,
        max_foreground_ratio,
        max_global_change_ratio,
        tuple(motion_regions),
    )


def iter_motion_events(
    input_name: str,
    *,
    format_name: Optional[str] = None,
    options: Optional[dict[str, str]] = None,
    config: Optional[MotionDetectorConfig] = None,
    pixel_config: Optional[PixelMotionConfig] = None,
    object_detector=None,
    presence_interval_seconds: float = 5.0,
    presence_misses_to_stop: int = 2,
    stop=None,
    log: Optional[Callable[[str], None]] = None,
) -> Iterator[MotionEvent]:
    pyav = require_pyav()
    detector = MotionVectorDetector(config)
    pixel_settings = pixel_config or PixelMotionConfig()
    open_options = export_motion_vector_options(options)
    with pyav.open(input_name, format=format_name or None, options=open_options) as container:
        video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
        if video_stream is None:
            raise RuntimeError("motion detector input has no video stream")
        codec_name = str(getattr(getattr(video_stream, "codec_context", None), "name", "") or "")
        if not codec_supports_motion_vectors(codec_name):
            raise RuntimeError(f"motion vectors are supported only for H.264/H.265, got {codec_name or 'unknown'}")
        video_stream.codec_context.options = export_motion_vector_options(getattr(video_stream.codec_context, "options", None))
        if log is not None:
            log(
                "motion_detector_opened "
                f"codec={codec_name} width={getattr(video_stream, 'width', None)} "
                f"height={getattr(video_stream, 'height', None)}"
            )

        last_analyzed = 0.0
        warned_no_vectors = False
        no_vector_frames = 0
        min_interval = 0.0
        if detector.config.max_analyze_fps > 0:
            min_interval = 1.0 / detector.config.max_analyze_fps

        gray_samples = []
        frame_in_gop = 0
        sample_serial = 0
        verification_after_sample = 0
        waiting_stats: Optional[MotionFrameStats] = None
        pending_stats: Optional[MotionFrameStats] = None
        pending_future: Optional[Future] = None
        yolo_future: Optional[Future] = None
        yolo_mode: Optional[str] = None
        yolo_stats: Optional[MotionFrameStats] = None
        verified_pixel_result: Optional[PixelMotionResult] = None
        presence_tracker: Optional[YoloPresenceTracker] = None
        confirmed_stats: Optional[MotionFrameStats] = None
        next_presence_check = 0.0
        next_presence_sample = 0.0
        presence_samples = deque(maxlen=YOLO_PRESENCE_SAMPLE_COUNT)
        latest_frame = None
        candidate_stopped = False
        confirmed_active = False
        cooldown_until = 0.0

        def finish_verification():
            nonlocal pending_future, pending_stats, candidate_stopped, confirmed_active, cooldown_until
            nonlocal yolo_future, yolo_mode, yolo_stats, verified_pixel_result, confirmed_stats
            if pending_future is None or not pending_future.done():
                return None
            future = pending_future
            stats = pending_stats
            stopped_before_confirmation = candidate_stopped
            pending_future = None
            pending_stats = None
            candidate_stopped = False
            result = future.result()
            if log is not None:
                log(
                    "motion_pixel_verification "
                    f"result={result.motion} reason={result.reason} "
                    f"samples={result.analyzed_samples} motion_samples={result.motion_samples} "
                    f"scene_changes={result.scene_change_samples} "
                    f"max_foreground_ratio={result.max_foreground_ratio:.4f} "
                    f"max_global_change_ratio={result.max_global_change_ratio:.4f} "
                    f"regions={len(result.motion_regions)}"
                )
            if not result.motion or stats is None:
                detector.reject_motion()
                cooldown_until = time.monotonic() + pixel_settings.reject_cooldown_seconds
                return None
            if object_detector is not None:
                if latest_frame is None:
                    detector.reject_motion()
                    cooldown_until = time.monotonic() + pixel_settings.reject_cooldown_seconds
                    return None
                verified_pixel_result = result
                yolo_stats = stats
                yolo_mode = "initial"
                yolo_future = executor.submit(object_detector.detect, frame_bgr_sample(latest_frame, 640))
                return None
            confirmed_active = True
            confirmed_stats = stats
            return MotionEvent("motion_start", stats, result), stopped_before_confirmation

        def finish_yolo():
            nonlocal yolo_future, yolo_mode, yolo_stats, verified_pixel_result
            nonlocal presence_tracker, next_presence_check, next_presence_sample
            nonlocal confirmed_active, confirmed_stats, cooldown_until
            if yolo_future is None or not yolo_future.done():
                return None
            detections = yolo_future.result()
            completed_mode = yolo_mode
            yolo_future = None
            yolo_mode = None
            if completed_mode == "initial":
                regions = () if verified_pixel_result is None else verified_pixel_result.motion_regions
                tracker = YoloPresenceTracker.from_initial(
                    detections,
                    regions,
                    misses_to_stop=presence_misses_to_stop,
                )
                if log is not None:
                    summary = ",".join(f"{item.label}:{item.confidence:.2f}" for item in detections) or "none"
                    log(
                        "motion_yolo_initial "
                        f"detections={summary} matched={0 if tracker is None else len(tracker.active)}"
                    )
                if tracker is None or yolo_stats is None:
                    detector.reject_motion()
                    cooldown_until = time.monotonic() + pixel_settings.reject_cooldown_seconds
                    yolo_stats = None
                    verified_pixel_result = None
                    return None
                presence_tracker = tracker
                confirmed_active = True
                confirmed_stats = yolo_stats
                now = time.monotonic()
                presence_samples.clear()
                presence_samples.append(frame_bgr_sample(latest_frame, 640))
                next_presence_sample = now + YOLO_PRESENCE_SAMPLE_INTERVAL_SECONDS
                next_presence_check = now + max(0.1, presence_interval_seconds)
                event = MotionEvent("motion_start", yolo_stats, verified_pixel_result)
                yolo_stats = None
                verified_pixel_result = None
                return event
            if completed_mode == "presence" and presence_tracker is not None:
                present = presence_tracker.update(detections, motion_active=detector.motion_active)
                if log is not None:
                    summary = ",".join(f"{item.label}:{item.confidence:.2f}" for item in detections) or "none"
                    log(
                        "motion_yolo_presence "
                        f"detections={summary} present={present} "
                        f"misses={presence_tracker.consecutive_misses}/"
                        f"{presence_tracker.effective_misses_to_stop} "
                        f"motion_active={detector.motion_active}"
                    )
                if presence_tracker.should_stop and confirmed_stats is not None:
                    stop_event = MotionEvent("motion_stop", confirmed_stats, verified_pixel_result)
                    presence_tracker = None
                    confirmed_active = False
                    presence_samples.clear()
                    detector.reject_motion()
                    cooldown_until = time.monotonic() + pixel_settings.reject_cooldown_seconds
                    return stop_event
            return None

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="watchdog-motion-pixel") as executor:
            for packet in container.demux(video_stream):
                if stop is not None and stop.is_set():
                    break
                if packet.stream is None or int(packet.stream.index) != int(video_stream.index):
                    continue
                if bool(getattr(packet, "is_keyframe", False)):
                    keep_samples = max(0, pixel_settings.previous_gop_samples)
                    gray_samples = gray_samples[-keep_samples:] if keep_samples else []
                    frame_in_gop = 0

                for frame in packet.decode():
                    if stop is not None and stop.is_set():
                        break
                    latest_frame = frame
                    if bool(getattr(frame, "key_frame", False)) and frame_in_gop:
                        keep_samples = max(0, pixel_settings.previous_gop_samples)
                        gray_samples = gray_samples[-keep_samples:] if keep_samples else []
                        frame_in_gop = 0
                    frame_in_gop += 1
                    sampled_current = False
                    if frame_in_gop == 1 or frame_in_gop % max(1, pixel_settings.sample_every_frames) == 0:
                        gray_samples.append(frame_motion_sample(frame, pixel_settings.max_sample_width))
                        sample_serial += 1
                        sampled_current = True
                        if len(gray_samples) > pixel_settings.max_samples_per_gop:
                            del gray_samples[: len(gray_samples) - pixel_settings.max_samples_per_gop]

                    now = time.monotonic()
                    if (
                        object_detector is not None
                        and confirmed_active
                        and presence_tracker is not None
                        and now >= next_presence_sample
                    ):
                        presence_samples.append(frame_bgr_sample(frame, 640))
                        next_presence_sample = now + YOLO_PRESENCE_SAMPLE_INTERVAL_SECONDS
                    if min_interval <= 0.0 or now - last_analyzed >= min_interval:
                        last_analyzed = now
                        event = detector.process_frame(frame)
                        vector_count = 0 if detector.last_stats is None else detector.last_stats.vector_count
                        if vector_count <= 0:
                            no_vector_frames += 1
                            if (
                                log is not None
                                and not warned_no_vectors
                                and no_vector_frames >= detector.config.no_vector_warning_frames
                            ):
                                warned_no_vectors = True
                                log("motion_detector_no_motion_vectors")
                        else:
                            no_vector_frames = 0

                        if event is not None and event.kind == "motion_start":
                            if confirmed_active:
                                pass
                            elif now < cooldown_until:
                                detector.reject_motion()
                            else:
                                if not sampled_current:
                                    gray_samples.append(frame_motion_sample(frame, pixel_settings.max_sample_width))
                                    sample_serial += 1
                                    if len(gray_samples) > pixel_settings.max_samples_per_gop:
                                        del gray_samples[: len(gray_samples) - pixel_settings.max_samples_per_gop]
                                waiting_stats = event.stats
                                verification_after_sample = sample_serial + max(
                                    0,
                                    pixel_settings.post_candidate_samples,
                                )
                        elif event is not None and event.kind == "motion_stop":
                            if confirmed_active:
                                if object_detector is None:
                                    confirmed_active = False
                                    yield event
                            elif pending_future is not None or waiting_stats is not None or yolo_mode == "initial":
                                candidate_stopped = True

                    if (
                        waiting_stats is not None
                        and pending_future is None
                        and len(gray_samples) >= pixel_settings.min_samples
                        and sample_serial >= verification_after_sample
                    ):
                        pending_stats = waiting_stats
                        waiting_stats = None
                        pending_future = executor.submit(verify_gray_motion, tuple(gray_samples), pixel_settings)

                    verified = finish_verification()
                    if verified is not None:
                        start_event, stopped_before_confirmation = verified
                        yield start_event
                        if stopped_before_confirmation:
                            confirmed_active = False
                            yield MotionEvent("motion_stop", start_event.stats, start_event.verification)

                    yolo_event = finish_yolo()
                    if yolo_event is not None:
                        yield yolo_event
                    now = time.monotonic()
                    if (
                        object_detector is not None
                        and confirmed_active
                        and presence_tracker is not None
                        and yolo_future is None
                        and latest_frame is not None
                        and now >= next_presence_check
                    ):
                        yolo_mode = "presence"
                        samples = tuple(presence_samples) or (frame_bgr_sample(latest_frame, 640),)
                        yolo_future = executor.submit(
                            detect_presence_samples,
                            object_detector,
                            samples,
                            tuple(presence_tracker.active),
                        )
                        next_presence_check = now + max(0.1, presence_interval_seconds)

                verified = finish_verification()
                if verified is not None:
                    start_event, stopped_before_confirmation = verified
                    yield start_event
                    if stopped_before_confirmation:
                        confirmed_active = False
                        yield MotionEvent("motion_stop", start_event.stats, start_event.verification)

                yolo_event = finish_yolo()
                if yolo_event is not None:
                    yield yolo_event
                now = time.monotonic()
                if (
                    object_detector is not None
                    and confirmed_active
                    and presence_tracker is not None
                    and yolo_future is None
                    and latest_frame is not None
                    and now >= next_presence_check
                ):
                    yolo_mode = "presence"
                    samples = tuple(presence_samples) or (frame_bgr_sample(latest_frame, 640),)
                    yolo_future = executor.submit(
                        detect_presence_samples,
                        object_detector,
                        samples,
                        tuple(presence_tracker.active),
                    )
                    next_presence_check = now + max(0.1, presence_interval_seconds)

            if pending_future is not None and not (stop is not None and stop.is_set()):
                pending_future.result()
                verified = finish_verification()
                if verified is not None:
                    start_event, stopped_before_confirmation = verified
                    yield start_event
                    if stopped_before_confirmation:
                        yield MotionEvent("motion_stop", start_event.stats, start_event.verification)
            if yolo_future is not None and not (stop is not None and stop.is_set()):
                yolo_future.result()
                yolo_event = finish_yolo()
                if yolo_event is not None:
                    yield yolo_event
