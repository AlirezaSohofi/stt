from __future__ import annotations

import time
from typing import Any

import numpy as np
import sounddevice as sd

from stt.util import eprint


def default_input_device_index() -> int:
    """Resolve PortAudio default input index (handles sounddevice _InputOutputPair)."""
    d = sd.default.device
    if hasattr(d, "__getitem__"):
        try:
            return int(d["input"])
        except Exception:
            try:
                return int(d[0])
            except Exception:
                pass
    if isinstance(d, (list, tuple)):
        return int(d[0])
    return int(d)


DevicePref = int | str | None


def refresh_devices() -> None:
    """Re-scan audio hardware. PortAudio caches the device list at init, so without this
    a long-running process never sees mics plugged in/out or System Settings changes."""
    try:
        sd._terminate()
        sd._initialize()
    except Exception:
        pass


def input_devices() -> list[tuple[int, dict[str, Any]]]:
    return [
        (i, dev)
        for i, dev in enumerate(sd.query_devices())
        if int(dev.get("max_input_channels", 0)) > 0
    ]


def resolve_input_device(pref: DevicePref) -> tuple[int | None, str | None]:
    """Map a device preference to a PortAudio index for the current hardware.

    None → system default input (follows System Settings → Sound → Input).
    str  → first input device whose name contains it (case-insensitive).
    int  → that index, if it is still an input device.
    Returns (index or None for system default, warning message or None).
    """
    if pref is None:
        return None, None
    devs = input_devices()
    if isinstance(pref, str):
        needle = pref.strip().lower()
        for i, dev in devs:
            if needle in str(dev.get("name", "")).lower():
                return i, None
        return None, f"no input device matching {pref!r}; using system default input"
    if any(i == pref for i, _ in devs):
        return pref, None
    return None, f"input device index {pref} not available; using system default input"


def device_name(index: int | None) -> str:
    try:
        if index is None:
            return str(sd.query_devices(kind="input").get("name", "?"))
        return str(sd.query_devices(index).get("name", "?"))
    except Exception:
        return "?"


def parse_device_pref(value: str) -> DevicePref:
    """Parse a user-supplied device: '' / 'default' / 'system' → None, digits → index, else name."""
    v = value.strip()
    if v.lower() in ("", "default", "system", "auto"):
        return None
    if v.isdigit():
        return int(v)
    return v


def describe_device_line(index: int, dev: dict[str, Any], *, mark_default: bool) -> str:
    name = str(dev.get("name", "?"))
    ch_in = int(dev.get("max_input_channels", 0))
    hostapi = int(dev.get("hostapi", 0))
    try:
        api_name = sd.query_hostapis(hostapi)["name"]
    except Exception:
        api_name = "?"
    star = " (default input)" if mark_default else ""
    return f"  [{index}] {name}{star}  —  {ch_in} ch in, {api_name}"


def print_input_devices_report(cfg_device: DevicePref) -> None:
    """Print default input and every device that can record (to stderr)."""
    refresh_devices()
    try:
        default_in = default_input_device_index()
    except Exception as e:
        eprint(f"error: could not read default input device: {e}")
        return

    eprint("--- Input devices ---")
    try:
        devs = input_devices()
    except Exception as e:
        eprint(f"error: could not list devices: {e}")
        return

    for i, dev in devs:
        eprint(describe_device_line(i, dev, mark_default=(i == default_in)))

    eprint("--- Config ---")
    if cfg_device is None:
        eprint("  stt default_device: (not set — following the macOS system input)")
    else:
        eprint(f"  stt default_device: {cfg_device!r}")
    idx, warning = resolve_input_device(cfg_device)
    if warning:
        eprint(f"  warning: {warning}")
    eprint(f"  recording would use: {device_name(idx)}")
    eprint(
        "  tip: leave default_device unset to follow System Settings → Sound → Input, "
        "or set it to part of a mic's name (e.g. \"AirPods\"); indices change when devices connect."
    )


def run_input_meter(
    *,
    sample_rate: int,
    channels: int,
    device: DevicePref,
    seconds: float,
) -> int:
    """Stream from the mic and print peak levels on stderr (debugging)."""
    eprint(f"meter: listening for {seconds:.1f}s (Ctrl+C to stop early)...")
    refresh_devices()
    device, warning = resolve_input_device(device)
    if warning:
        eprint(f"warning: {warning}")
    if device is None:
        try:
            di = default_input_device_index()
            info = sd.query_devices(di)
            eprint(f"meter: using default input index {di} — {info.get('name', '?')}")
        except Exception as e:
            eprint(f"meter: could not describe default device: {e}")
    else:
        try:
            info = sd.query_devices(device)
            eprint(f"meter: using device index {device} — {info.get('name', '?')}")
        except Exception as e:
            eprint(f"error: invalid device index: {e}")
            return 1

    state: dict[str, float] = {"last": 0.0, "peak": 0.0}

    def callback(indata: np.ndarray, frames: int, t: Any, status: Any) -> None:
        del frames, t
        peak = float(np.max(np.abs(indata)))
        state["peak"] = max(state["peak"], peak)
        now = time.monotonic()
        if now - state["last"] >= 0.15:
            state["last"] = now
            bar = _meter_bar(peak)
            stat = f" {status}" if status else ""
            eprint(f"  peak={peak:.5f}  {bar}{stat}")

    try:
        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            device=device,
            blocksize=1024,
            latency="high",
            callback=callback,
        )
        with stream:
            t_end = time.monotonic() + seconds
            while time.monotonic() < t_end:
                time.sleep(0.05)
    except KeyboardInterrupt:
        eprint("")
        eprint("meter: interrupted")
    except Exception as e:
        eprint(f"error: meter failed (permission denied, wrong device, or driver issue): {e}")
        return 1

    eprint(f"meter: session max peak={state['peak']:.5f}")
    if state["peak"] < 1e-5:
        eprint(
            "warning: no signal detected. On macOS check: "
            "System Settings → Privacy & Security → Microphone → enable for this app "
            "(Terminal / iTerm / Cursor); "
            "System Settings → Sound → Input → select the correct mic and raise input volume; "
            "run `stt devices` to see which mic is in use."
        )
        return 1
    return 0


def _meter_bar(peak: float, width: int = 28) -> str:
    x = min(1.0, peak / 0.25)
    filled = int(round(x * width))
    return "|" + ("#" * filled).ljust(width) + "|"
