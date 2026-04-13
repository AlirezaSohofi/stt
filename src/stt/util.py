from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def read_line(prompt: str = "") -> str:
    """Read one line from stdin. Prompt goes to stderr so stdout stays clean when piped."""
    if prompt:
        print(prompt, end="", file=sys.stderr, flush=True)
    line = sys.stdin.readline()
    if line == "":
        raise EOFError
    return line.rstrip("\r\n")


def expand_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def audio_metrics(audio: np.ndarray, sample_rate: int) -> tuple[float, float, float]:
    """Return (duration_seconds, peak_abs, rms)."""
    if audio.size == 0:
        return 0.0, 0.0, 0.0
    a = np.asarray(audio, dtype=np.float64)
    peak = float(np.max(np.abs(a)))
    rms = float(np.sqrt(np.mean(a * a)))
    return len(a) / float(sample_rate), peak, rms


def rms_to_dbfs(rms: float) -> float:
    return float(20.0 * np.log10(max(rms, 1e-12)))


def format_audio_summary(audio: np.ndarray, sample_rate: int) -> str:
    dur, peak, rms = audio_metrics(audio, sample_rate)
    if dur <= 0:
        return "no samples captured"
    return (
        f"recorded {dur:.2f}s, peak={peak:.5f}, "
        f"rms≈{rms_to_dbfs(rms):.1f} dBFS"
    )


def audio_sounds_empty(audio: np.ndarray, sample_rate: int) -> tuple[bool, str | None]:
    """Heuristic: True if capture is likely silent or too quiet to transcribe reliably."""
    dur, peak, rms = audio_metrics(audio, sample_rate)
    if dur <= 0:
        return True, "no audio samples were captured; check the input device"
    if dur < 0.05:
        return True, "recording is extremely short"
    if peak < 1e-5:
        return True, "signal is effectively silent; wrong input device or muted mic"
    if peak < 0.003 or rms < 1e-5:
        return (
            True,
            "signal is very quiet — playing audio from speakers may not reach the mic; "
            "try a headset, speak close to the mic, or use system loopback (e.g. BlackHole)",
        )
    return False, None
