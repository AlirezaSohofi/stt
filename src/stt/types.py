from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SessionMetadata:
    id: str
    created_at: str
    model: str
    language: str
    sample_rate: int
    channels: int
    audio_path: str
    transcript_path: str
    duration_seconds: float

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "model": self.model,
            "language": self.language,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "audio_path": self.audio_path,
            "transcript_path": self.transcript_path,
            "duration_seconds": self.duration_seconds,
        }
