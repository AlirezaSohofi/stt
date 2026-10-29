from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np

from stt.cleanup import clean_transcript, trim_silence
from stt.config import SttConfig
from stt.models import MLX_REPOS, validate_model
from stt.util import write_wav


def mlx_available() -> bool:
    return importlib.util.find_spec("mlx_whisper") is not None


class WhisperTranscriber:
    def __init__(self, cfg: SttConfig) -> None:
        self._cfg = cfg
        self.model_name = cfg.model
        self._model = None

    @property
    def language(self) -> str | None:
        lang = self._cfg.language
        return None if lang == "auto" else lang

    @property
    def backend(self) -> str:
        """The engine actually used for the current model: 'mlx' or 'openai'."""
        want = self._cfg.backend
        can_mlx = mlx_available() and self.model_name in MLX_REPOS
        if want == "mlx":
            if not mlx_available():
                raise RuntimeError("backend 'mlx' requires: uv sync --extra mlx")
            if self.model_name not in MLX_REPOS:
                raise RuntimeError(f"no MLX build of model {self.model_name!r}")
            return "mlx"
        if want == "openai":
            return "openai"
        return "mlx" if can_mlx else "openai"

    def set_model(self, model_name: str) -> None:
        validate_model(model_name)
        if model_name != self.model_name:
            self.model_name = model_name
            self._model = None

    def preload(self) -> None:
        """Load weights now so the first real transcription isn't slow."""
        if self.backend == "mlx":
            # mlx_whisper caches the model internally; a tiny transcription warms it.
            self._mlx_transcribe(np.zeros(16000, dtype=np.float32))
        else:
            self._ensure_model()

    def _ensure_model(self) -> None:
        if self._model is None:
            import whisper

            self._model = whisper.load_model(self.model_name)

    def _whisper_verbose(self) -> bool | None:
        # Whisper treats verbose=False as "show tqdm + language line"; None is fully quiet.
        return True if self._cfg.transcription.verbose else None

    def _decode_options(self) -> dict:
        tr = self._cfg.transcription
        opts: dict = {
            "language": self.language,
            "verbose": self._whisper_verbose(),
            # Without this, one looped segment poisons every following segment.
            "condition_on_previous_text": tr.condition_on_previous_text,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
        }
        if tr.initial_prompt:
            opts["initial_prompt"] = tr.initial_prompt
        # A scalar temperature disables Whisper's fallback re-decoding, which is what
        # rescues repetition loops; only pass it when the user explicitly set one.
        if tr.temperature is not None:
            opts["temperature"] = tr.temperature
        return opts

    def _mlx_transcribe(self, audio: np.ndarray | str) -> str:
        import mlx_whisper

        opts = self._decode_options()
        result = mlx_whisper.transcribe(
            audio, path_or_hf_repo=MLX_REPOS[self.model_name], **opts
        )
        return str(result["text"]).strip()

    def _openai_transcribe(self, path: str) -> str:
        self._ensure_model()
        assert self._model is not None
        tr = self._cfg.transcription
        result = self._model.transcribe(
            path, fp16=tr.fp16, beam_size=tr.beam_size, **self._decode_options()
        )
        return str(result["text"]).strip()

    def _finish(self, text: str) -> str:
        if self._cfg.transcription.filter_hallucinations:
            text = clean_transcript(text)
        return text.strip()

    def transcribe_array(self, audio: np.ndarray) -> str:
        if self._cfg.transcription.trim_silence:
            audio = trim_silence(audio, self._cfg.sample_rate)
            if audio.size == 0:
                return ""
        if self.backend == "mlx":
            return self._finish(self._mlx_transcribe(np.asarray(audio, dtype=np.float32)))
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "input.wav"
            write_wav(wav_path, audio, sample_rate=self._cfg.sample_rate)
            return self._finish(self._openai_transcribe(str(wav_path)))

    def transcribe_file(self, path: str | Path) -> str:
        if self.backend == "mlx":
            return self._finish(self._mlx_transcribe(str(path)))
        return self._finish(self._openai_transcribe(str(path)))
