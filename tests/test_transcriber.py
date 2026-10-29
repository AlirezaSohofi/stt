from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from stt.config import SttConfig
from stt.transcriber import WhisperTranscriber


def _tone(seconds: float, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * sr), dtype=np.float32)
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


@patch("whisper.load_model")
def test_transcribe_file_uses_model(mock_load: MagicMock, short_wav_path: Path) -> None:
    model = MagicMock()
    model.transcribe.return_value = {"text": "  hi  "}
    mock_load.return_value = model

    cfg = SttConfig(model="tiny", backend="openai")
    t = WhisperTranscriber(cfg)
    text = t.transcribe_file(short_wav_path)
    assert text == "hi"
    mock_load.assert_called_once_with("tiny")
    model.transcribe.assert_called_once()


@patch("whisper.load_model")
def test_transcribe_array_writes_temp_wav(mock_load: MagicMock) -> None:
    model = MagicMock()
    model.transcribe.return_value = {"text": "done"}
    mock_load.return_value = model

    cfg = SttConfig(model="base", backend="openai")
    t = WhisperTranscriber(cfg)
    audio = _tone(0.5)
    assert t.transcribe_array(audio) == "done"
    path_arg = model.transcribe.call_args[0][0]
    assert str(path_arg).endswith(".wav")


@patch("whisper.load_model")
def test_set_model_reloads(mock_load: MagicMock) -> None:
    m1 = MagicMock()
    m1.transcribe.return_value = {"text": "a"}
    m2 = MagicMock()
    m2.transcribe.return_value = {"text": "b"}
    mock_load.side_effect = [m1, m2]

    cfg = SttConfig(model="tiny", backend="openai")
    t = WhisperTranscriber(cfg)
    t.transcribe_array(_tone(0.5))
    t.set_model("base")
    t.transcribe_array(_tone(0.5))
    assert mock_load.call_count == 2


@patch("whisper.load_model")
def test_silence_never_reaches_model(mock_load: MagicMock) -> None:
    cfg = SttConfig(model="tiny", backend="openai")
    t = WhisperTranscriber(cfg)
    assert t.transcribe_array(np.zeros(16000, dtype=np.float32)) == ""
    mock_load.assert_not_called()


@patch("whisper.load_model")
def test_openai_keeps_temperature_fallback(mock_load: MagicMock) -> None:
    model = MagicMock()
    model.transcribe.return_value = {"text": "ok"}
    mock_load.return_value = model
    t = WhisperTranscriber(SttConfig(model="tiny", backend="openai"))
    t.transcribe_array(_tone(0.5))
    kwargs = model.transcribe.call_args.kwargs
    assert "temperature" not in kwargs  # scalar temperature would disable fallback
    assert kwargs["condition_on_previous_text"] is False


def test_backend_auto_prefers_mlx(monkeypatch) -> None:
    monkeypatch.setattr("stt.transcriber.mlx_available", lambda: True)
    assert WhisperTranscriber(SttConfig(model="turbo")).backend == "mlx"
    assert WhisperTranscriber(SttConfig(model="turbo", backend="openai")).backend == "openai"
    monkeypatch.setattr("stt.transcriber.mlx_available", lambda: False)
    assert WhisperTranscriber(SttConfig(model="turbo")).backend == "openai"
