from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import sounddevice as sd

from stt.config import SttConfig, load_config
from stt.devices import default_input_device_index, print_input_devices_report
from stt.models import validate_model
from stt.sessions import ensure_sessions_dir_writable
from stt.util import eprint


def _ok(msg: str) -> None:
    eprint(f"[ok] {msg}")


def _warn(msg: str) -> None:
    eprint(f"[warn] {msg}")


def _fail(msg: str) -> None:
    eprint(f"[fail] {msg}")


def run_doctor(
    cfg: SttConfig | None = None,
    *,
    config_path: Path | None = None,
) -> int:
    exit_code = 0
    missing_dep = False

    v = sys.version_info
    if v.major == 3 and v.minor == 12:
        _ok(f"python {v.major}.{v.minor}.{v.micro}")
    else:
        _warn(f"python {v.major}.{v.minor}.{v.micro} (expected 3.12.x)")

    if shutil.which("ffmpeg"):
        _ok("ffmpeg found")
    else:
        _fail("ffmpeg not found (install with: brew install ffmpeg)")
        missing_dep = True

    whisper_spec = importlib.util.find_spec("whisper")
    if whisper_spec is not None:
        try:
            import whisper  # noqa: F401

            _ok("whisper import successful")
        except Exception as e:
            _fail(f"whisper import failed: {e}")
            missing_dep = True
    else:
        _fail("whisper package not found")
        missing_dep = True

    try:
        load_config(config_path, strict_model=True)
        _ok("config parses successfully")
    except Exception as e:
        _fail(f"config error: {e}")
        exit_code = 1

    try:
        if cfg is None:
            cfg = load_config(config_path, strict_model=False)
        try:
            validate_model(cfg.model)
        except ValueError as e:
            _fail(str(e))
            exit_code = 1
        ensure_sessions_dir_writable(cfg.sessions_path())
        _ok("sessions dir writable")
    except Exception as e:
        _fail(str(e))
        exit_code = 1

    try:
        default = sd.query_devices(kind="input")
        name = default.get("name", "?")
        try:
            idx = default_input_device_index()
            _ok(f"default input device available (index {idx}: {name})")
        except Exception:
            _ok(f"default input device available ({name})")
    except Exception as e:
        _warn(f"microphone / input device: {e}")

    try:
        if cfg is not None:
            eprint("")
            print_input_devices_report(cfg.default_device)
    except Exception as e:
        _warn(f"could not print device list: {e}")

    if missing_dep:
        return 3
    return exit_code
