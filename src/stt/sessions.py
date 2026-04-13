from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from stt.types import SessionMetadata
from stt.util import write_wav


def new_session_id() -> tuple[str, str]:
    """Return (folder_id, iso_created_at) using local timezone for display."""
    now = datetime.now().astimezone()
    folder_id = now.strftime("%Y-%m-%dT%H-%M-%S")
    created = now.isoformat(timespec="seconds")
    return folder_id, created


def save_session(
    sessions_root: Path,
    session_id: str,
    created_at: str,
    audio: np.ndarray,
    transcript: str,
    *,
    model: str,
    language: str,
    sample_rate: int,
    channels: int,
) -> SessionMetadata:
    root = sessions_root.expanduser().resolve()
    session_dir = root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    audio_path = session_dir / "audio.wav"
    transcript_path = session_dir / "transcript.txt"
    meta_path = session_dir / "metadata.json"

    write_wav(audio_path, audio, sample_rate=sample_rate)
    transcript_path.write_text(transcript + "\n", encoding="utf-8")

    duration_seconds = float(len(audio) / sample_rate) if len(audio) else 0.0

    meta = SessionMetadata(
        id=session_id,
        created_at=created_at,
        model=model,
        language=language,
        sample_rate=sample_rate,
        channels=channels,
        audio_path=str(audio_path),
        transcript_path=str(transcript_path),
        duration_seconds=duration_seconds,
    )
    meta_path.write_text(
        json.dumps(meta.to_json_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return meta


def ensure_sessions_dir_writable(sessions_root: Path) -> None:
    root = sessions_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    test = root / ".write_test"
    try:
        test.write_text("ok", encoding="utf-8")
        test.unlink()
    except OSError as e:
        raise RuntimeError(f"sessions dir not writable: {root}") from e
