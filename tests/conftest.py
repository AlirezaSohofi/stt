from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from stt.util import write_wav


@pytest.fixture()
def short_wav_path(tmp_path: Path) -> Path:
    sr = 16000
    t = np.linspace(0, 0.05, int(0.05 * sr), dtype=np.float32)
    audio = (0.01 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "short.wav"
    write_wav(path, audio, sample_rate=sr)
    return path


@pytest.fixture()
def temp_config_dir(tmp_path: Path) -> Path:
    d = tmp_path / "config"
    d.mkdir(parents=True)
    return d
