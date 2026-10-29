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


# openai-whisper name -> MLX hub repo. Models without an MLX build fall back to openai.
MLX_REPOS: dict[str, str] = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}
