from __future__ import annotations

WHISPER_MODEL_NAMES: tuple[str, ...] = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large",
    "turbo",
)


def validate_model(name: str) -> None:
    if name not in WHISPER_MODEL_NAMES:
        allowed = ", ".join(WHISPER_MODEL_NAMES)
        raise ValueError(f"unknown model {name!r}; allowed: {allowed}")


def list_models() -> list[str]:
    return list(WHISPER_MODEL_NAMES)
