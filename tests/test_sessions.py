from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stt.sessions import new_session_id, save_session


def test_new_session_id_format() -> None:
    sid, created = new_session_id()
    assert "T" in sid
    assert "-" in sid
    assert len(created) >= 10


def test_save_session_layout(tmp_path: Path) -> None:
    audio = np.zeros(800, dtype=np.float32)
    sid = "2026-04-13T10-15-30"
    created = "2026-04-13T10:15:30+00:00"
    meta = save_session(
        tmp_path,
        sid,
        created,
        audio,
        "hello",
        model="turbo",
        language="auto",
        sample_rate=16000,
        channels=1,
    )
    d = tmp_path / sid
    assert (d / "audio.wav").is_file()
    assert (d / "transcript.txt").read_text(encoding="utf-8").strip() == "hello"
    raw = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    assert raw["id"] == sid
    assert raw["model"] == "turbo"
    assert meta.duration_seconds >= 0
