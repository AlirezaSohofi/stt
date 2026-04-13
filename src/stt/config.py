from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib

from stt.models import validate_model


DEFAULT_CONFIG_PATH = Path.home() / ".config" / "stt" / "config.toml"


@dataclass
class UiConfig:
    show_status: bool = True
    show_timestamps: bool = False


@dataclass
class TranscriptionConfig:
    temperature: float = 0.0
    fp16: bool = False
    verbose: bool = False


@dataclass
class SttConfig:
    model: str = "turbo"
    language: str = "auto"
    save_recordings: bool = False
    sessions_dir: str = "~/.local/share/stt/sessions"
    default_device: int | None = None
    sample_rate: int = 16000
    channels: int = 1
    ui: UiConfig = field(default_factory=UiConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)

    def sessions_path(self) -> Path:
        return Path(self.sessions_dir).expanduser().resolve()


def _parse_toml_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _merge_dict_into_config(cfg: SttConfig, data: dict[str, Any]) -> None:
    if "model" in data:
        cfg.model = str(data["model"])
    if "language" in data:
        cfg.language = str(data["language"])
    if "save_recordings" in data:
        cfg.save_recordings = bool(data["save_recordings"])
    if "sessions_dir" in data:
        cfg.sessions_dir = str(data["sessions_dir"])
    if "default_device" in data and data["default_device"] is not None:
        cfg.default_device = int(data["default_device"])
    elif "default_device" in data:
        cfg.default_device = None
    if "sample_rate" in data:
        cfg.sample_rate = int(data["sample_rate"])
    if "channels" in data:
        cfg.channels = int(data["channels"])
    ui = data.get("ui")
    if isinstance(ui, dict):
        if "show_status" in ui:
            cfg.ui.show_status = bool(ui["show_status"])
        if "show_timestamps" in ui:
            cfg.ui.show_timestamps = bool(ui["show_timestamps"])
    tr = data.get("transcription")
    if isinstance(tr, dict):
        if "temperature" in tr:
            cfg.transcription.temperature = float(tr["temperature"])
        if "fp16" in tr:
            cfg.transcription.fp16 = bool(tr["fp16"])
        if "verbose" in tr:
            cfg.transcription.verbose = bool(tr["verbose"])


def _apply_env(cfg: SttConfig) -> None:
    if v := os.environ.get("STT_MODEL"):
        cfg.model = v.strip()
    if v := os.environ.get("STT_LANGUAGE"):
        cfg.language = v.strip()
    if v := os.environ.get("STT_SAVE_RECORDINGS"):
        cfg.save_recordings = v.strip().lower() in ("1", "true", "yes", "on")
    if v := os.environ.get("STT_SESSIONS_DIR"):
        cfg.sessions_dir = v.strip()
    if (v := os.environ.get("STT_DEFAULT_DEVICE")) is not None and v.strip() != "":
        cfg.default_device = int(v.strip())
    if v := os.environ.get("STT_SAMPLE_RATE"):
        cfg.sample_rate = int(v.strip())
    if v := os.environ.get("STT_CHANNELS"):
        cfg.channels = int(v.strip())
    if v := os.environ.get("STT_CONFIG"):
        pass  # handled by load path


def load_config(
    config_path: Path | None = None,
    *,
    cli_overrides: dict[str, Any] | None = None,
    strict_model: bool = True,
) -> SttConfig:
    path = config_path
    if path is None:
        env_path = os.environ.get("STT_CONFIG")
        path = Path(env_path).expanduser() if env_path else DEFAULT_CONFIG_PATH

    cfg = SttConfig()
    file_data = _parse_toml_file(path)
    _merge_dict_into_config(cfg, file_data)
    _apply_env(cfg)

    if cli_overrides:
        _merge_dict_into_config(cfg, cli_overrides)

    if strict_model:
        validate_model(cfg.model)
    if cfg.sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if cfg.channels not in (1, 2):
        raise ValueError("channels must be 1 or 2")
    return cfg
