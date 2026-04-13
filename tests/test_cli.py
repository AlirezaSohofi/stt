from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from stt.cli import app


runner = CliRunner()


def test_cli_models() -> None:
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "turbo" in result.stdout


@patch("stt.cli.run_doctor", return_value=0)
def test_cli_doctor(mock_doc) -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    mock_doc.assert_called_once()


@patch("stt.cli.WhisperTranscriber")
def test_cli_transcribe(mock_tr_class, short_wav_path) -> None:
    inst = mock_tr_class.return_value
    inst.transcribe_file.return_value = "fixture text"
    result = runner.invoke(app, ["transcribe", str(short_wav_path)])
    assert result.exit_code == 0
    assert result.stdout.strip() == "fixture text"


@patch("stt.cli.WhisperTranscriber")
def test_cli_transcribe_model_flag(mock_tr_class, short_wav_path) -> None:
    inst = mock_tr_class.return_value
    inst.transcribe_file.return_value = "ok"
    result = runner.invoke(
        app, ["transcribe", str(short_wav_path), "--model", "tiny"]
    )
    assert result.exit_code == 0
    cfg = mock_tr_class.call_args[0][0]
    assert cfg.model == "tiny"


def test_cli_transcribe_invalid_model(short_wav_path) -> None:
    result = runner.invoke(
        app, ["transcribe", str(short_wav_path), "-m", "not-a-model"]
    )
    assert result.exit_code == 2
    assert "unknown model" in result.stderr
