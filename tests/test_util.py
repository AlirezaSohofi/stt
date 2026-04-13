from __future__ import annotations

from io import StringIO

import numpy as np

import stt.util as util
from stt.util import audio_sounds_empty, format_audio_summary


def test_read_line_prompt_on_stderr(monkeypatch, capsys) -> None:
    monkeypatch.setattr(util.sys, "stdin", StringIO("hello\n"))
    assert util.read_line("PROMPT ") == "hello"
    assert capsys.readouterr().err == "PROMPT "


def test_format_audio_summary_empty() -> None:
    assert "no samples" in format_audio_summary(np.array([]), 16000)


def test_audio_sounds_empty_silence() -> None:
    quiet = np.zeros(16000, dtype=np.float32)
    bad, hint = audio_sounds_empty(quiet, 16000)
    assert bad
    assert hint is not None


def test_audio_sounds_empty_speech_like() -> None:
    # ~ -20 dBFS RMS sine-ish
    x = (0.2 * np.sin(np.linspace(0, 6.28, 8000))).astype(np.float32)
    bad, hint = audio_sounds_empty(x, 16000)
    assert not bad
    assert hint is None
