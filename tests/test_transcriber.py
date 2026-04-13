from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from stt.config import SttConfig
from stt.transcriber import WhisperTranscriber


@patch("stt.transcriber.whisper.load_model")
def test_transcribe_file_uses_model(mock_load: MagicMock, short_wav_path: Path) -> None:
    model = MagicMock()
    model.transcribe.return_value = {"text": "  hi  "}
    mock_load.return_value = model

    cfg = SttConfig(model="tiny")
    t = WhisperTranscriber(cfg)
    text = t.transcribe_file(short_wav_path)
    assert text == "hi"
    mock_load.assert_called_once_with("tiny")
    model.transcribe.assert_called_once()


@patch("stt.transcriber.whisper.load_model")
def test_transcribe_array_writes_temp_wav(mock_load: MagicMock) -> None:
    model = MagicMock()
    model.transcribe.return_value = {"text": "done"}
    mock_load.return_value = model

    cfg = SttConfig(model="base")
    t = WhisperTranscriber(cfg)
    audio = np.zeros(320, dtype=np.float32)
    assert t.transcribe_array(audio) == "done"
    path_arg = model.transcribe.call_args[0][0]
    assert str(path_arg).endswith(".wav")


@patch("stt.transcriber.whisper.load_model")
def test_set_model_reloads(mock_load: MagicMock) -> None:
    m1 = MagicMock()
    m1.transcribe.return_value = {"text": "a"}
    m2 = MagicMock()
    m2.transcribe.return_value = {"text": "b"}
    mock_load.side_effect = [m1, m2]

    cfg = SttConfig(model="tiny")
    t = WhisperTranscriber(cfg)
    t.transcribe_array(np.zeros(100, dtype=np.float32))
    t.set_model("base")
    t.transcribe_array(np.zeros(100, dtype=np.float32))
    assert mock_load.call_count == 2
