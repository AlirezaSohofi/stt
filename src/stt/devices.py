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


def print_input_devices_report(cfg_device: int | None) -> None:
    """Print default input and every device that can record (to stderr)."""
    try:
        default_in = default_input_device_index()
    except Exception as e:
        eprint(f"error: could not read default input device: {e}")
        return

    eprint("--- Input devices (index → set as default_device in ~/.config/stt/config.toml) ---")
    try:
        all_devs = sd.query_devices()
    except Exception as e:
        eprint(f"error: could not list devices: {e}")
        return

    for i, dev in enumerate(all_devs):
        if int(dev.get("max_input_channels", 0)) <= 0:
            continue
        eprint(describe_device_line(i, dev, mark_default=(i == default_in)))

    eprint("--- Config ---")
    if cfg_device is None:
        eprint(f"  stt default_device: (not set — using PortAudio default index {default_in})")
    else:
        eprint(f"  stt default_device: {cfg_device}")
        if cfg_device != default_in:
            eprint(
                "  note: config overrides the system default; "
                "ensure this index matches the mic you expect."
            )


def run_input_meter(
    *,
    sample_rate: int,
    channels: int,
    device: int | None,
    seconds: float,
) -> int:
    """Stream from the mic and print peak levels on stderr (debugging)."""
    eprint(f"meter: listening for {seconds:.1f}s (Ctrl+C to stop early)...")
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
            "run `stt devices` and set default_device to your MacBook mic index."
        )
        return 1
    return 0


def _meter_bar(peak: float, width: int = 28) -> str:
    x = min(1.0, peak / 0.25)
    filled = int(round(x * width))
    return "|" + ("#" * filled).ljust(width) + "|"
