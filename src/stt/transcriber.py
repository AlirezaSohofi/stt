from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import whisper

from stt.config import SttConfig
from stt.models import validate_model
from stt.util import write_wav


class WhisperTranscriber:
    def __init__(self, cfg: SttConfig) -> None:
        self._cfg = cfg
        self.model_name = cfg.model
        self._model = None

    @property
    def language(self) -> str | None:
        lang = self._cfg.language
        return None if lang == "auto" else lang

    def set_model(self, model_name: str) -> None:
        validate_model(model_name)
        if model_name != self.model_name:
            self.model_name = model_name
            self._model = None

    def _ensure_model(self) -> None:
        if self._model is None:
            self._model = whisper.load_model(self.model_name)

    def _whisper_verbose(self) -> bool | None:
        # Whisper treats verbose=False as "show tqdm + language line"; None is fully quiet.
        return True if self._cfg.transcription.verbose else None

    def transcribe_array(self, audio: np.ndarray) -> str:
        self._ensure_model()
        assert self._model is not None
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "input.wav"
            write_wav(wav_path, audio, sample_rate=self._cfg.sample_rate)
            result = self._model.transcribe(
                str(wav_path),
                language=self.language,
                fp16=self._cfg.transcription.fp16,
                verbose=self._whisper_verbose(),
                temperature=self._cfg.transcription.temperature,
            )
        return str(result["text"]).strip()

    def transcribe_file(self, path: str | Path) -> str:
        self._ensure_model()
        assert self._model is not None
        result = self._model.transcribe(
            str(path),
            language=self.language,
            fp16=self._cfg.transcription.fp16,
            verbose=self._whisper_verbose(),
            temperature=self._cfg.transcription.temperature,
        )
        return str(result["text"]).strip()
