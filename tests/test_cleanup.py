from __future__ import annotations

import numpy as np

from stt.cleanup import clean_transcript, collapse_repeats, trim_silence

SR = 16000


def _speechlike(seconds: float) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * SR), dtype=np.float32)
    return (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def test_trim_removes_silence_but_keeps_padding() -> None:
    audio = np.concatenate([np.zeros(2 * SR, np.float32), _speechlike(1.0), np.zeros(3 * SR, np.float32)])
    out = trim_silence(audio, SR)
    assert 1.0 * SR < len(out) < 2.0 * SR
    assert len(out) >= 1.0 * SR + 0.4 * SR  # tail padding preserved


def test_trim_all_silence_is_empty() -> None:
    assert trim_silence(np.zeros(SR, np.float32), SR).size == 0
    assert trim_silence((np.random.default_rng(0).standard_normal(SR) * 1e-4).astype(np.float32), SR).size == 0


def test_collapse_repeats_breaks_loops() -> None:
    text = "Hello there. " + "This is a loop. " * 25 + "The end."
    out = collapse_repeats(text)
    assert out.count("This is a loop.") == 2
    assert out.startswith("Hello there.") and out.endswith("The end.")


def test_phantom_dropped_only_when_whole_transcript() -> None:
    assert clean_transcript("Thank you.") == ""
    assert clean_transcript("Thank you for the report, see you Monday.") != ""
