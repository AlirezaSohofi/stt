from __future__ import annotations

from pathlib import Path

import pytest

from stt.config import SttConfig, load_config


def test_load_defaults() -> None:
    cfg = load_config(Path("/nonexistent/stt/config.toml"))
    assert cfg.model == "turbo"
    assert cfg.language == "auto"
    assert cfg.sample_rate == 16000


def test_load_from_toml(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    p.write_text(
        'model = "base"\nlanguage = "en"\nsave_recordings = true\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.model == "base"
    assert cfg.language == "en"
    assert cfg.save_recordings is True


def test_cli_override_model(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    p.write_text('model = "base"\n', encoding="utf-8")
    cfg = load_config(p, cli_overrides={"model": "tiny"})
    assert cfg.model == "tiny"


def test_invalid_model_strict(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    p.write_text('model = "not-a-model"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown model"):
        load_config(p, strict_model=True)


def test_invalid_model_non_strict(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    p.write_text('model = "not-a-model"\n', encoding="utf-8")
    cfg = load_config(p, strict_model=False)
    assert cfg.model == "not-a-model"


def test_default_device_name_and_index(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    p.write_text('default_device = "AirPods"\n', encoding="utf-8")
    assert load_config(p).default_device == "AirPods"
    p.write_text("default_device = 2\n", encoding="utf-8")
    assert load_config(p).default_device == 2


def test_copy_to_clipboard_default_and_override(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    assert load_config(p).copy_to_clipboard is True
    p.write_text("copy_to_clipboard = false\n", encoding="utf-8")
    assert load_config(p).copy_to_clipboard is False
