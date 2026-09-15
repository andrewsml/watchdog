"""Build clean portable Watchdog archives with Nuitka.

On Windows the default invocation builds Windows locally and Linux inside WSL.
On Linux it builds the Linux archive locally.  Runtime/user data is never copied
from the source tree; release resources are selected by an explicit allowlist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "dist" / "portable"
DEFAULT_ICON_CANDIDATES = (
    PROJECT_DIR / "logo_512x512.ico",
    PROJECT_DIR / "media" / "stuff" / "logo_512x512.ico",
)
LOCAL_LIBTAM_WHEELS = {
    "windows": PROJECT_DIR.parent / "tam" / "dist" / "wheels" / "libtam-1.0.0-py3-none-win_amd64.whl",
    "linux": PROJECT_DIR.parent / "tam" / "dist" / "wheels" / "libtam-1.0.0-py3-none-linux_x86_64.whl",
}
LIBTAM_WHEEL_URLS = {
    "windows": "https://github.com/andrewsml/libtam/releases/download/v1.0.0/libtam-1.0.0-py3-none-win_amd64.whl",
    "linux": "https://github.com/andrewsml/libtam/releases/download/v1.0.0/libtam-1.0.0-py3-none-linux_x86_64.whl",
}
RELEASE_FILES = (
    Path("README.md"),
    Path("media/stuff/watchdog_en.json"),
    Path("media/stuff/watchdog_ru.json"),
    Path("media/stuff/logo_512x512.png"),
    Path("media/stuff/logo_512x512.ico"),
    Path("media/stuff/yolov10n.onnx"),
)
FORBIDDEN_PARTS = frozenset(
    {
        "tdlib",
        "__pycache__",
        ".watchdog-build-venv",
        "watchdog_config.json",
        "watchdog_debug.log",
        "yolov10n.onnx.part",
    }
)
FORBIDDEN_PREFIXES = (
    ("media", "files"),
)
WSL_PACKAGES = ("python3", "python3-venv", "python3-dev", "build-essential", "patchelf", "ccache")
COMMAND_HISTORY: list[str] = []


class BuildError(RuntimeError):
    pass


def display_command(command: Sequence[object]) -> str:
    values = [str(value) for value in command]
    return subprocess.list2cmdline(values) if os.name == "nt" else shlex.join(values)


def run_command(
    command: Sequence[object],
    *,
    cwd: Optional[Path] = None,
    capture_output: bool = False,
    env_overrides: Optional[dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    values = [str(value) for value in command]
    rendered = display_command(values)
    working_directory = str(cwd.resolve()) if cwd is not None else str(Path.cwd())
    print(f"+ {rendered}", flush=True)
    COMMAND_HISTORY.append(f"RUN cwd={working_directory}\n  {rendered}")
    if env_overrides:
        COMMAND_HISTORY.append(
            "ENV " + " ".join(f"{name}={value}" for name, value in sorted(env_overrides.items()))
        )
    process_environment = None
    if env_overrides:
        process_environment = os.environ.copy()
        process_environment.update(env_overrides)
    try:
        result = subprocess.run(
            values,
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=process_environment,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
        )
        COMMAND_HISTORY.append("EXIT 0")
        return result
    except FileNotFoundError as error:
        COMMAND_HISTORY.append(f"FAILED FileNotFoundError: {error}")
        raise BuildError(f"Command not found: {values[0]}") from error
    except subprocess.CalledProcessError as error:
        COMMAND_HISTORY.append(f"EXIT {error.returncode}")
        details = ""
        if capture_output:
            details = f"\n{(error.stdout or '').strip()}\n{(error.stderr or '').strip()}".rstrip()
        raise BuildError(f"Command failed ({error.returncode}): {display_command(values)}{details}") from error


def build_message(target: str, step: int, total: int, message: str) -> None:
    print(f"\n[build:{target}] step {step}/{total}: {message}", flush=True)


def diagnostic_inventory(root: Path, limit: int = 5000) -> list[str]:
    if not root.exists():
        return ["<build directory does not exist>"]
    lines: list[str] = []
    for index, path in enumerate(sorted(root.rglob("*"))):
        if index >= limit:
            lines.append(f"... inventory truncated after {limit} entries")
            break
        try:
            relative = path.relative_to(root)
            if path.is_file():
                lines.append(f"FILE {relative} ({path.stat().st_size} bytes)")
            else:
                lines.append(f"DIR  {relative}")
        except OSError as error:
            lines.append(f"ERR  {path}: {error}")
    return lines


def write_failure_diagnostics(
    *,
    build_root: Path,
    target: str,
    base_python: Path,
    build_venv: Path,
    icon_path: Path,
    libtam_wheel: str,
    error: BaseException,
    command_history_start: int,
) -> Path:
    report = build_root / "BUILD_FAILURE.txt"
    sections = [
        "Watchdog portable build failure",
        "================================",
        f"time: {datetime.now().astimezone().isoformat()}",
        f"target: {target}",
        f"host_platform: {platform.platform()}",
        f"host_machine: {platform.machine()}",
        f"builder_python: {sys.version}",
        f"base_python: {base_python}",
        f"persistent_build_venv: {build_venv}",
        f"project_dir: {PROJECT_DIR}",
        f"build_root: {build_root}",
        f"icon: {icon_path}",
        f"libtam_wheel: {libtam_wheel}",
        "",
        "Failure",
        "-------",
        f"{type(error).__name__}: {error}",
        "",
        "Traceback",
        "---------",
        traceback.format_exc(),
        "",
        "Commands",
        "--------",
        *COMMAND_HISTORY[command_history_start:],
        "",
        "Build directory inventory",
        "-------------------------",
        *diagnostic_inventory(build_root),
        "",
    ]
    try:
        report.write_text("\n".join(sections), encoding="utf-8")
    except OSError as report_error:
        print(f"[build:{target}] could not write diagnostic report: {report_error}", file=sys.stderr, flush=True)
    return report


def default_icon() -> Path:
    for candidate in DEFAULT_ICON_CANDIDATES:
        if candidate.is_file():
            return candidate
    return DEFAULT_ICON_CANDIDATES[0]


def default_libtam_wheel(target: str) -> str:
    local_wheel = LOCAL_LIBTAM_WHEELS[target]
    return str(local_wheel if local_wheel.is_file() else LIBTAM_WHEEL_URLS[target])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build clean Windows and Linux Watchdog ZIP archives with Nuitka."
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=("windows", "linux"),
        dest="targets",
        help="target to build; repeat for both (default: both on Windows, Linux on Linux)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="archive output directory")
    parser.add_argument(
        "--build-venv-dir",
        type=Path,
        default=Path(".watchdog-build-venv"),
        help="persistent build environments, relative to the launch directory",
    )
    parser.add_argument(
        "--refresh-build-venv",
        action="store_true",
        help="update dependencies in the persistent build environments",
    )
    parser.add_argument("--icon", type=Path, default=default_icon(), help="ICO file for watchdog.exe")
    parser.add_argument("--windows-python", type=Path, help="64-bit CPython 3.9-3.12 for the Windows build")
    parser.add_argument(
        "--windows-compiler",
        choices=("mingw64", "msvc"),
        default="mingw64",
        help="Windows C compiler (default: Nuitka-managed MinGW64)",
    )
    parser.add_argument("--wsl-distro", default="Ubuntu-24.04", help="WSL distribution used for Linux")
    parser.add_argument(
        "--no-wsl-bootstrap",
        action="store_true",
        help="do not install missing apt packages in WSL",
    )
    parser.add_argument(
        "--libtam-windows-wheel",
        default=default_libtam_wheel("windows"),
        help="path or URL of the LibTam Windows wheel",
    )
    parser.add_argument(
        "--libtam-linux-wheel",
        default=default_libtam_wheel("linux"),
        help="path or URL of the LibTam Linux wheel",
    )
    parser.add_argument("--nuitka-version", help="optional exact Nuitka version")
    parser.add_argument("--skip-self-test", action="store_true", help="skip execution of the built program")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and print the planned targets only")
    parser.add_argument("--inner-linux", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def unique_targets(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def host_default_targets() -> list[str]:
    if os.name == "nt":
        return ["windows", "linux"]
    if sys.platform.startswith("linux"):
        return ["linux"]
    raise BuildError(f"Unsupported build host: {sys.platform}")


def validate_project_inputs(targets: Sequence[str], icon_path: Path) -> None:
    required = [PROJECT_DIR / "watchdog.py", PROJECT_DIR / "requirements.txt"]
    required.extend(PROJECT_DIR / relative for relative in RELEASE_FILES)
    if "windows" in targets:
        required.append(icon_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise BuildError("Required build files are missing:\n  " + "\n  ".join(missing))


def python_facts(python: Path) -> tuple[int, int, int, str]:
    code = (
        "import platform,struct,sys;"
        "print(sys.version_info[0],sys.version_info[1],struct.calcsize('P')*8,platform.machine())"
    )
    result = run_command([python, "-c", code], capture_output=True)
    fields = result.stdout.strip().split()
    if len(fields) != 4:
        raise BuildError(f"Cannot determine Python version and architecture: {result.stdout!r}")
    return int(fields[0]), int(fields[1]), int(fields[2]), fields[3]


def discover_windows_python(explicit: Optional[Path], compiler: str) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser().resolve())
    else:
        candidates.append(Path(sys.executable).resolve())
        launcher = shutil.which("py")
        if launcher:
            for version in ("3.12", "3.11", "3.10", "3.9"):
                try:
                    result = run_command(
                        [launcher, f"-{version}", "-c", "import sys;print(sys.executable)"],
                        capture_output=True,
                    )
                except BuildError:
                    continue
                candidate = Path(result.stdout.strip())
                if candidate.is_file():
                    candidates.append(candidate.resolve())

    errors: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).casefold()
        if key in seen:
            continue
        seen.add(key)
        if not candidate.is_file():
            errors.append(f"{candidate}: file not found")
            continue
        try:
            major, minor, bits, machine = python_facts(candidate)
        except BuildError as error:
            errors.append(str(error))
            continue
        if bits != 64 or machine.casefold() not in {"amd64", "x86_64"}:
            errors.append(f"{candidate}: expected x86_64, got {bits}-bit {machine}")
            continue
        if major != 3 or minor < 9:
            errors.append(f"{candidate}: Python 3.9 or newer is required")
            continue
        if compiler == "mingw64" and minor > 12:
            errors.append(f"{candidate}: MinGW64 requires Python 3.12 or older")
            continue
        return candidate

    details = "\n  ".join(errors) if errors else "no candidates found"
    raise BuildError(
        "No suitable 64-bit Windows Python was found. Pass --windows-python.\n  " + details
    )


def venv_python(venv_dir: Path, target: str) -> Path:
    if target == "windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_environment_fingerprint(
    base_python: Path,
    target: str,
    libtam_wheel: str,
    nuitka_requirement: str,
) -> dict[str, object]:
    base_python_stat = base_python.stat()
    wheel_identity: dict[str, object]
    if "://" in libtam_wheel:
        wheel_identity = {"url": libtam_wheel}
    else:
        wheel_path = Path(libtam_wheel).expanduser().resolve()
        wheel_stat = wheel_path.stat()
        wheel_identity = {
            "path": str(wheel_path),
            "size": wheel_stat.st_size,
            "mtime_ns": wheel_stat.st_mtime_ns,
        }
    return {
        "schema": 1,
        "target": target,
        "base_python": str(base_python.resolve()),
        "base_python_size": base_python_stat.st_size,
        "base_python_mtime_ns": base_python_stat.st_mtime_ns,
        "requirements_sha256": file_sha256(PROJECT_DIR / "requirements.txt"),
        "libtam": wheel_identity,
        "nuitka": nuitka_requirement,
    }


def read_environment_marker(path: Path) -> Optional[dict[str, object]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def verify_build_environment(python: Path, target: str) -> str:
    imports = "import av,cv2,numpy,nuitka,tam"
    if target == "windows":
        imports += ",windows_capture"
    code = (
        imports
        + "; import sys; print('python=' + sys.version.split()[0]);"
        + " print('nuitka=' + getattr(nuitka, '__version__', 'installed'));"
        + " print('av=' + getattr(av, '__version__', 'unknown'));"
        + " print('numpy=' + getattr(numpy, '__version__', 'unknown'));"
        + " print('opencv=' + getattr(cv2, '__version__', 'unknown'));"
        + " print('tam=' + str(tam.__file__))"
    )
    result = run_command([python, "-c", code], capture_output=True)
    return result.stdout.strip()


def prepare_build_environment(
    base_python: Path,
    environment_dir: Path,
    target: str,
    libtam_wheel: str,
    nuitka_version: Optional[str],
    refresh: bool,
) -> Path:
    python = venv_python(environment_dir, target)
    nuitka_requirement = f"Nuitka=={nuitka_version}" if nuitka_version else "Nuitka"
    marker_path = environment_dir / ".watchdog-build-env.json"
    expected_marker = build_environment_fingerprint(base_python, target, libtam_wheel, nuitka_requirement)
    environment_exists = python.is_file()

    if environment_exists and not refresh and read_environment_marker(marker_path) == expected_marker:
        try:
            versions = verify_build_environment(python, target)
        except BuildError as error:
            print(f"[build:{target}] cached environment verification failed; repairing it: {error}", flush=True)
        else:
            print(f"[build:{target}] reusing persistent venv: {environment_dir}", flush=True)
            if versions:
                print(versions, flush=True)
            return python

    if not environment_exists:
        print(f"[build:{target}] creating persistent venv: {environment_dir}", flush=True)
        environment_dir.parent.mkdir(parents=True, exist_ok=True)
        command: list[object] = [base_python, "-m", "venv"]
        if target == "linux":
            command.append("--copies")
        command.append(environment_dir)
        run_command(command)
        python = venv_python(environment_dir, target)
        run_command([python, "-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "pip"])
    else:
        reason = "refresh requested" if refresh else "build inputs changed"
        print(f"[build:{target}] updating persistent venv ({reason}): {environment_dir}", flush=True)
        if refresh:
            run_command([python, "-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "pip"])

    install_command: list[object] = [
        python,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
    ]
    if refresh:
        install_command.append("--upgrade")
    install_command.extend(
        [
            nuitka_requirement,
            "ordered-set",
            "zstandard",
            "-r",
            PROJECT_DIR / "requirements.txt",
            libtam_wheel,
        ]
    )
    run_command(install_command)
    versions = verify_build_environment(python, target)
    marker_path.write_text(json.dumps(expected_marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[build:{target}] persistent venv is ready: {environment_dir}", flush=True)
    if versions:
        print(versions, flush=True)
    return python


def locate_pyav_installation(python: Path) -> tuple[Path, Optional[Path], Optional[Path], tuple[str, ...]]:
    code = (
        "import av,importlib.machinery,importlib.metadata,json,pathlib;"
        "package=pathlib.Path(av.__file__).resolve().parent;"
        "libs=package.parent/'av.libs';"
        "dist=pathlib.Path(importlib.metadata.distribution('av')._path).resolve();"
        "print(json.dumps({'package':str(package),'libs':str(libs) if libs.is_dir() else None,"
        "'dist_info':str(dist) if dist.is_dir() else None,"
        "'suffixes':importlib.machinery.EXTENSION_SUFFIXES}))"
    )
    result = run_command([python, "-c", code], capture_output=True)
    try:
        data = json.loads(result.stdout.strip())
        package = Path(data["package"])
        libs = Path(data["libs"]) if data.get("libs") else None
        dist_info = Path(data["dist_info"]) if data.get("dist_info") else None
        suffixes = tuple(str(value) for value in data["suffixes"])
    except (KeyError, TypeError, ValueError) as error:
        raise BuildError(f"Cannot parse PyAV installation details: {result.stdout!r}") from error
    if not package.is_dir():
        raise BuildError(f"PyAV package directory not found: {package}")
    return package, libs, dist_info, suffixes


def remove_pyav_shadow_sources(package: Path, extension_suffixes: Sequence[str]) -> list[Path]:
    removed: list[Path] = []
    for source in sorted(package.rglob("*.py")):
        if source.name == "__init__.py":
            continue
        if any(source.with_name(source.stem + suffix).is_file() for suffix in extension_suffixes):
            source.unlink()
            removed.append(source.relative_to(package))
    return removed


def prepare_pyav_build_tree(python: Path, build_root: Path) -> Path:
    package, libs, dist_info, extension_suffixes = locate_pyav_installation(python)
    sanitized_root = build_root / "pyav-runtime"
    sanitized_package = sanitized_root / "av"
    shutil.copytree(package, sanitized_package)
    if libs is not None:
        shutil.copytree(libs, sanitized_root / libs.name)
    if dist_info is not None:
        shutil.copytree(dist_info, sanitized_root / dist_info.name)
    removed = remove_pyav_shadow_sources(sanitized_package, extension_suffixes)
    if not removed:
        raise BuildError(
            "PyAV sanitation found no Python shadow sources next to extension modules; "
            "the wheel layout may have changed"
        )
    print("[build] PyAV shadow sources excluded in favor of wheel extensions:", flush=True)
    for relative in removed:
        print(f"  av/{relative}", flush=True)

    verification = run_command(
        [python, "-c", "import av;print(av.__version__);print(av.audio.frame.__file__)"],
        capture_output=True,
        env_overrides={"PYTHONPATH": str(sanitized_root)},
    )
    print(f"[build] sanitized PyAV import OK:\n{verification.stdout.strip()}", flush=True)
    return sanitized_root


def run_nuitka(
    python: Path,
    build_root: Path,
    target: str,
    icon_path: Path,
    windows_compiler: str,
    pyav_build_root: Path,
) -> tuple[Path, Path]:
    output_dir = build_root / "nuitka"
    output_dir.mkdir(parents=True, exist_ok=True)
    executable_name = "watchdog.exe" if target == "windows" else "watchdog"
    command: list[object] = [
        python,
        "-m",
        "nuitka",
        "--mode=standalone",
        "--assume-yes-for-downloads",
        "--remove-output",
        "--no-prefer-source-code",
        f"--output-dir={output_dir}",
        f"--output-filename={executable_name}",
        f"--report={build_root / 'nuitka-compilation-report.xml'}",
        "--include-module=tam",
        "--include-package=av",
        "--include-package-data=av",
        "--include-module=numpy",
        "--include-module=cv2",
        "--include-package-data=cv2",
    ]
    if target == "windows":
        command.extend(
            [
                "--include-package=windows_capture",
                "--include-package-data=windows_capture",
                f"--windows-icon-from-ico={icon_path}",
                "--windows-console-mode=force",
            ]
        )
        command.append("--mingw64" if windows_compiler == "mingw64" else "--msvc=latest")
    else:
        command.append("--nofollow-import-to=windows_capture")
    command.append(PROJECT_DIR / "watchdog.py")
    run_command(
        command,
        cwd=PROJECT_DIR,
        env_overrides={"PYTHONPATH": str(pyav_build_root)},
    )

    distributions = sorted(output_dir.glob("*.dist"))
    if len(distributions) != 1:
        raise BuildError(f"Expected one Nuitka .dist directory, found: {distributions}")
    distribution = distributions[0]
    executable = distribution / executable_name
    if not executable.is_file() and target == "linux":
        default_binary = distribution / "watchdog.bin"
        if default_binary.is_file():
            default_binary.replace(executable)
    if not executable.is_file():
        raise BuildError(f"Nuitka executable not found: {executable}")
    if target == "linux":
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return distribution, executable


def installed_tam_directory(python: Path) -> Path:
    result = run_command(
        [python, "-c", "import pathlib,tam;print(pathlib.Path(tam.__file__).resolve().parent)"],
        capture_output=True,
    )
    path = Path(result.stdout.strip())
    if not path.is_dir():
        raise BuildError(f"Cannot locate the installed LibTam package directory: {path}")
    return path


def copy_libtam_runtime(python: Path, target: str, destination: Path) -> None:
    source = installed_tam_directory(python) / "platform" / target
    if not source.is_dir():
        raise BuildError(f"LibTam {target} native runtime not found: {source}")
    target_dir = destination / "platform" / target
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source, target_dir)


def copy_release_files(destination: Path) -> None:
    for relative in RELEASE_FILES:
        source = PROJECT_DIR / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    quick_start = destination / "PORTABLE_README.txt"
    quick_start.write_text(
        "Watchdog portable\n"
        "=================\n\n"
        "First run / initial configuration:\n"
        "  Windows: ..\\start_watchdog.bat --setup\n"
        "  Linux:   ../start_watchdog.sh --setup\n\n"
        "Normal run:\n"
        "  Windows: ..\\start_watchdog.bat\n"
        "  Linux:   ../start_watchdog.sh\n\n"
        "Keep the executable inside this extracted directory. Watchdog creates\n"
        "media/stuff/watchdog_config.json, tdlib, media/files and optional logs\n"
        "on first use.\n",
        encoding="utf-8",
    )


def create_start_launcher(release_root: Path, target: str) -> Path:
    if target == "windows":
        launcher = release_root / "start_watchdog.bat"
        launcher.write_text(
            "@echo off\n"
            "setlocal\n"
            "cd /d \"%~dp0watchdog\" || exit /b 1\n"
            "watchdog.exe %*\n"
            "exit /b %errorlevel%\n",
            encoding="utf-8",
        )
    else:
        launcher = release_root / "start_watchdog.sh"
        launcher.write_text(
            "#!/bin/sh\n"
            "SCRIPT_DIR=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\n"
            "cd \"$SCRIPT_DIR/watchdog\" || exit 1\n"
            "exec ./watchdog \"$@\"\n",
            encoding="utf-8",
        )
        launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return launcher


def path_is_forbidden(relative: Path) -> bool:
    parts = tuple(part.casefold() for part in relative.parts)
    if any(part in FORBIDDEN_PARTS for part in parts):
        return True
    return any(parts[: len(prefix)] == prefix for prefix in FORBIDDEN_PREFIXES)


def validate_payload(payload: Path, target: str) -> None:
    forbidden = [path.relative_to(payload) for path in payload.rglob("*") if path_is_forbidden(path.relative_to(payload))]
    if forbidden:
        raise BuildError("Forbidden runtime/private files entered the release: " + ", ".join(map(str, forbidden)))

    executable = payload / ("watchdog.exe" if target == "windows" else "watchdog")
    required = [executable, payload / "PORTABLE_README.txt"]
    required.extend(payload / relative for relative in RELEASE_FILES)
    missing = [str(path.relative_to(payload)) for path in required if not path.is_file()]
    if missing:
        raise BuildError("Portable payload is incomplete: " + ", ".join(missing))
    native_library = (
        payload / "platform" / "windows" / "bin" / "libtam.dll"
        if target == "windows"
        else payload / "platform" / "linux" / "bin" / "libtam.so"
    )
    if not native_library.is_file():
        raise BuildError(f"LibTam native library is missing: {native_library}")


def validate_release_tree(release_root: Path, payload: Path, target: str) -> None:
    validate_payload(payload, target)
    launcher_name = "start_watchdog.bat" if target == "windows" else "start_watchdog.sh"
    expected_entries = {"watchdog", launcher_name}
    actual_entries = {path.name for path in release_root.iterdir()}
    if actual_entries != expected_entries:
        raise BuildError(
            "Portable release root must contain only the application directory and launcher; "
            f"expected {sorted(expected_entries)}, got {sorted(actual_entries)}"
        )
    launcher = release_root / launcher_name
    if not launcher.is_file():
        raise BuildError(f"Portable launcher is missing: {launcher}")
    if target == "linux" and not launcher.stat().st_mode & stat.S_IXUSR:
        raise BuildError(f"Portable launcher is not executable: {launcher}")


def write_zip(release_root: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = archive_path.with_suffix(archive_path.suffix + ".tmp")
    temporary_archive.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for source in sorted(release_root.rglob("*")):
                if not source.is_file():
                    continue
                archive_name = Path(release_root.name) / source.relative_to(release_root)
                info = zipfile.ZipInfo.from_file(source, archive_name.as_posix())
                info.compress_type = zipfile.ZIP_DEFLATED
                mode = source.stat().st_mode & 0xFFFF
                if source in {
                    release_root / "watchdog" / "watchdog",
                    release_root / "start_watchdog.sh",
                }:
                    mode |= stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
                info.create_system = 3
                info.external_attr = mode << 16
                with source.open("rb") as input_file, archive.open(info, "w") as output_file:
                    shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
        os.replace(temporary_archive, archive_path)
    finally:
        temporary_archive.unlink(missing_ok=True)


def validate_zip(archive_path: Path, target: str) -> None:
    root_name = f"watchdog-{target}-x86_64"
    executable = "watchdog.exe" if target == "windows" else "watchdog"
    launcher = "start_watchdog.bat" if target == "windows" else "start_watchdog.sh"
    app_root = f"{root_name}/watchdog"
    required = {
        f"{root_name}/{launcher}",
        f"{app_root}/{executable}",
        f"{app_root}/PORTABLE_README.txt",
        f"{app_root}/media/stuff/yolov10n.onnx",
        f"{app_root}/media/stuff/watchdog_en.json",
        f"{app_root}/media/stuff/watchdog_ru.json",
    }
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = set(archive.namelist())
        missing = sorted(required - names)
        if missing:
            raise BuildError(f"Archive {archive_path.name} is incomplete: {', '.join(missing)}")
        forbidden = []
        for name in names:
            relative_parts = Path(name).parts[1:]
            if relative_parts[:1] == ("watchdog",):
                relative_parts = relative_parts[1:]
            if relative_parts and path_is_forbidden(Path(*relative_parts)):
                forbidden.append(name)
        if forbidden:
            raise BuildError(f"Archive {archive_path.name} contains private/runtime files: {', '.join(forbidden)}")
        if target == "linux":
            for executable_path in (f"{app_root}/{executable}", f"{root_name}/{launcher}"):
                executable_info = archive.getinfo(executable_path)
                executable_mode = (executable_info.external_attr >> 16) & 0xFFFF
                if not executable_mode & stat.S_IXUSR:
                    raise BuildError(
                        f"Archive {archive_path.name} lost the executable permission for {executable_path}"
                    )


def run_built_self_test(payload: Path, target: str) -> None:
    executable = payload / ("watchdog.exe" if target == "windows" else "watchdog")
    result = run_command([executable, "--portable-self-test"], cwd=payload, capture_output=True)
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr, flush=True)


def ensure_yolo_model() -> None:
    from object_detector import ensure_yolo_model as ensure_model

    ensure_model(PROJECT_DIR / "media" / "stuff" / "yolov10n.onnx", log=print)


def build_native(
    *,
    target: str,
    base_python: Path,
    build_venv_dir: Path,
    output_dir: Path,
    icon_path: Path,
    libtam_wheel: str,
    nuitka_version: Optional[str],
    windows_compiler: str,
    skip_self_test: bool,
    refresh_build_venv: bool,
) -> Path:
    if (target == "windows") != (os.name == "nt"):
        raise BuildError(f"The {target} target must be compiled on a native {target} Python")
    machine = platform.machine().casefold()
    if machine not in {"amd64", "x86_64"}:
        raise BuildError(f"The portable archives currently require an x86_64 host, got {machine}")
    print(f"\nBuilding Watchdog for {target} x86_64", flush=True)
    work_parent = output_dir / ".build-work"
    work_parent.mkdir(parents=True, exist_ok=True)
    build_root = Path(tempfile.mkdtemp(prefix=f"watchdog-{target}-", dir=work_parent))
    command_history_start = len(COMMAND_HISTORY)
    try:
        build_message(target, 1, 9, "validate and checksum the YOLO model")
        ensure_yolo_model()
        persistent_venv = build_venv_dir / target
        build_message(target, 2, 9, f"prepare persistent build environment {persistent_venv}")
        python = prepare_build_environment(
            base_python,
            persistent_venv,
            target,
            libtam_wheel,
            nuitka_version,
            refresh_build_venv,
        )
        build_message(target, 3, 9, "prepare PyAV wheel extensions and compile with Nuitka")
        pyav_build_root = prepare_pyav_build_tree(python, build_root)
        distribution, _ = run_nuitka(
            python,
            build_root,
            target,
            icon_path,
            windows_compiler,
            pyav_build_root,
        )
        build_message(target, 4, 9, f"stage Nuitka distribution from {distribution}")
        release_root = build_root / f"watchdog-{target}-x86_64"
        payload = release_root / "watchdog"
        shutil.copytree(distribution, payload)
        build_message(target, 5, 9, "copy LibTam and TDLib native runtime")
        copy_libtam_runtime(python, target, payload)
        build_message(target, 6, 9, "copy release resources from the explicit allowlist")
        copy_release_files(payload)
        create_start_launcher(release_root, target)
        build_message(target, 7, 9, "validate payload and reject private runtime files")
        validate_release_tree(release_root, payload, target)
        if not skip_self_test:
            build_message(target, 8, 9, "run dependency and native-library self-test")
            run_built_self_test(payload, target)
        else:
            build_message(target, 8, 9, "self-test skipped by --skip-self-test")
        build_message(target, 9, 9, "create and validate ZIP archive")
        archive_path = output_dir / f"watchdog-{target}-x86_64.zip"
        write_zip(release_root, archive_path)
        validate_zip(archive_path, target)
    except BaseException as error:
        report = write_failure_diagnostics(
            build_root=build_root,
            target=target,
            base_python=base_python,
            build_venv=build_venv_dir / target,
            icon_path=icon_path,
            libtam_wheel=libtam_wheel,
            error=error,
            command_history_start=command_history_start,
        )
        print(f"\n[build:{target}] FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        print(f"[build:{target}] failed build retained at: {build_root}", file=sys.stderr, flush=True)
        print(f"[build:{target}] diagnostic report: {report}", file=sys.stderr, flush=True)
        executable = build_root / f"watchdog-{target}-x86_64" / "watchdog" / (
            "watchdog.exe" if target == "windows" else "watchdog"
        )
        if executable.is_file():
            print(
                f"[build:{target}] reproduce self-test: {display_command([executable, '--portable-self-test'])}",
                file=sys.stderr,
                flush=True,
            )
        raise
    else:
        try:
            shutil.rmtree(build_root)
            if not any(work_parent.iterdir()):
                work_parent.rmdir()
        except OSError as error:
            print(f"[build:{target}] warning: could not remove build directory: {error}", file=sys.stderr)
    print(f"Created: {archive_path}", flush=True)
    return archive_path


def wsl_command(distro: str, command: Sequence[object], *, user: Optional[str] = None) -> list[object]:
    result: list[object] = ["wsl.exe", "--distribution", distro]
    if user:
        result.extend(["--user", user])
    result.append("--")
    result.extend(command)
    return result


def ensure_wsl(distro: str, bootstrap: bool) -> None:
    run_command(wsl_command(distro, ["true"]))
    package_check = [
        "dpkg-query",
        "-W",
        "-f=${db:Status-Abbrev}\\n",
        *WSL_PACKAGES,
    ]
    try:
        run_command(wsl_command(distro, package_check), capture_output=True)
        return
    except BuildError:
        if not bootstrap:
            raise BuildError(
                "WSL build packages are missing. Install "
                + " ".join(WSL_PACKAGES)
                + " or omit --no-wsl-bootstrap."
            )
    print("Installing the WSL packages required by Nuitka...", flush=True)
    run_command(wsl_command(distro, ["apt-get", "update"], user="root"))
    run_command(
        wsl_command(
            distro,
            ["env", "DEBIAN_FRONTEND=noninteractive", "apt-get", "install", "-y", *WSL_PACKAGES],
            user="root",
        )
    )


def to_wsl_path(path: Path, distro: str) -> str:
    # An unquoted ``D:\dir\file`` argument is processed by WSL's Linux-side
    # argument parser, where backslashes act as escapes.  subprocess.list2cmdline
    # does not quote paths without spaces, so normalize them to the equally valid
    # ``D:/dir/file`` form before handing them to wslpath.
    windows_path = str(path.resolve()).replace("\\", "/")
    result = run_command(
        wsl_command(distro, ["wslpath", "-a", "-u", windows_path]),
        capture_output=True,
    )
    value = result.stdout.strip()
    if not value:
        raise BuildError(f"WSL could not translate path: {path}")
    return value


def wheel_for_wsl(value: str, distro: str) -> str:
    if "://" in value:
        return value
    path = Path(value).expanduser()
    if not path.is_file():
        raise BuildError(f"LibTam Linux wheel not found: {path}")
    return to_wsl_path(path, distro)


def build_linux_in_wsl(args: argparse.Namespace, output_dir: Path, build_venv_dir: Path) -> None:
    print(f"\n[build:linux-wsl] checking WSL distribution '{args.wsl_distro}'", flush=True)
    ensure_wsl(args.wsl_distro, not args.no_wsl_bootstrap)
    script = to_wsl_path(Path(__file__), args.wsl_distro)
    wsl_output = to_wsl_path(output_dir, args.wsl_distro)
    wsl_build_venv = to_wsl_path(build_venv_dir, args.wsl_distro)
    wheel = wheel_for_wsl(args.libtam_linux_wheel, args.wsl_distro)
    command: list[object] = [
        "python3",
        script,
        "--inner-linux",
        "--output-dir",
        wsl_output,
        "--build-venv-dir",
        wsl_build_venv,
        "--libtam-linux-wheel",
        wheel,
    ]
    if args.nuitka_version:
        command.extend(["--nuitka-version", args.nuitka_version])
    if args.skip_self_test:
        command.append("--skip-self-test")
    if args.refresh_build_venv:
        command.append("--refresh-build-venv")
    print("[build:linux-wsl] starting native Linux builder; its diagnostics use Linux paths", flush=True)
    run_command(wsl_command(args.wsl_distro, command))


def dry_run(
    args: argparse.Namespace,
    targets: Sequence[str],
    output_dir: Path,
    build_venv_dir: Path,
    icon_path: Path,
) -> None:
    validate_project_inputs(targets, icon_path)
    print("Portable build plan:")
    for target in targets:
        location = "local Windows" if target == "windows" else ("WSL" if os.name == "nt" else "local Linux")
        wheel = args.libtam_windows_wheel if target == "windows" else args.libtam_linux_wheel
        print(f"  {target:7} -> {location}; LibTam: {wheel}")
    print(f"  output  -> {output_dir}")
    print(f"  venvs   -> {build_venv_dir / 'windows'}; {build_venv_dir / 'linux'}")
    print("  excluded -> tdlib, watchdog_config.json, media/files, watchdog_debug.log")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.inner_linux and not sys.platform.startswith("linux"):
        raise BuildError("--inner-linux may only run inside Linux/WSL")

    targets = ["linux"] if args.inner_linux else unique_targets(args.targets or host_default_targets())
    if os.name != "nt" and "windows" in targets:
        raise BuildError("Run the Windows target from Windows; Nuitka does not cross-compile it from Linux")
    output_dir = args.output_dir.expanduser().resolve()
    build_venv_dir = args.build_venv_dir.expanduser().resolve()
    icon_path = args.icon.expanduser().resolve()
    validate_project_inputs(targets, icon_path)

    if args.dry_run:
        dry_run(args, targets, output_dir, build_venv_dir, icon_path)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    print("Watchdog portable builder", flush=True)
    print(f"  project: {PROJECT_DIR}", flush=True)
    print(f"  output: {output_dir}", flush=True)
    print(f"  persistent build venvs: {build_venv_dir}", flush=True)
    print(f"  targets: {', '.join(targets)}", flush=True)
    print(f"  host: {platform.platform()} ({platform.machine()})", flush=True)
    print(f"  builder Python: {sys.executable} [{sys.version.split()[0]}]", flush=True)
    print(f"  Windows icon: {icon_path}", flush=True)
    if "windows" in targets:
        windows_python = discover_windows_python(args.windows_python, args.windows_compiler)
        print(f"  selected Windows Python: {windows_python}", flush=True)
        print(f"  selected Windows compiler: {args.windows_compiler}", flush=True)
        build_native(
            target="windows",
            base_python=windows_python,
            build_venv_dir=build_venv_dir,
            output_dir=output_dir,
            icon_path=icon_path,
            libtam_wheel=args.libtam_windows_wheel,
            nuitka_version=args.nuitka_version,
            windows_compiler=args.windows_compiler,
            skip_self_test=args.skip_self_test,
            refresh_build_venv=args.refresh_build_venv,
        )
    if "linux" in targets:
        if os.name == "nt" and not args.inner_linux:
            build_linux_in_wsl(args, output_dir, build_venv_dir)
        else:
            build_native(
                target="linux",
                base_python=Path(sys.executable).resolve(),
                build_venv_dir=build_venv_dir,
                output_dir=output_dir,
                icon_path=icon_path,
                libtam_wheel=args.libtam_linux_wheel,
                nuitka_version=args.nuitka_version,
                windows_compiler=args.windows_compiler,
                skip_self_test=args.skip_self_test,
                refresh_build_venv=args.refresh_build_venv,
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as error:
        print(f"build_portable: {error}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as error:
        print(
            f"build_portable: unexpected {type(error).__name__}: {error}\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        raise SystemExit(1)
