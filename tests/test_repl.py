from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from stt.config import SttConfig
from stt.repl import run_repl, scripted_input


def test_repl_quit_immediate() -> None:
    cfg = SttConfig()
    code = run_repl(cfg, input_fn=scripted_input(["quit"]))
    assert code == 0


def test_repl_quit_aliases() -> None:
    cfg = SttConfig()
    for cmd in ("q", "esc", "escape", "Q"):
        code = run_repl(cfg, input_fn=scripted_input([cmd]))
        assert code == 0, cmd


def test_repl_stop_while_idle(capsys) -> None:
    cfg = SttConfig()
    code = run_repl(cfg, input_fn=scripted_input(["stop", "quit"]))
    assert code == 0
    err = capsys.readouterr().err
    assert "no recording in progress" in err


def test_repl_model_and_status() -> None:
    cfg = SttConfig()
    fn = scripted_input(["model tiny", "status", "quit"])
    run_repl(cfg, input_fn=fn)
    assert cfg.model == "tiny"


@patch("builtins.print")
@patch("stt.repl.Recorder")
@patch("stt.repl.WhisperTranscriber")
def test_repl_record_stop_transcribe(
    mock_tr: MagicMock,
    mock_rec_class: MagicMock,
    mock_print: MagicMock,
    no_real_clipboard: list[str],
) -> None:
    cfg = SttConfig()
    mock_rec = MagicMock()
    mock_rec.is_recording = False
    mock_rec.take_stream_reports.return_value = []
    mock_rec_class.return_value = mock_rec

    def start_side() -> None:
        mock_rec.is_recording = True

    def stop_side() -> np.ndarray:
        mock_rec.is_recording = False
        return np.zeros(1600, dtype=np.float32)

    mock_rec.start.side_effect = start_side
    mock_rec.stop.side_effect = stop_side

    mock_tr_instance = MagicMock()
    mock_tr_instance.model_name = "turbo"
    mock_tr_instance.transcribe_array.return_value = "hello world"
    mock_tr.return_value = mock_tr_instance

    run_repl(cfg, input_fn=scripted_input(["", "", "quit"]))

    texts = [c.args[0] for c in mock_print.call_args_list if c.args]
    assert "hello world" in texts
    mock_tr_instance.transcribe_array.assert_called_once()
    assert no_real_clipboard == ["hello world"]


@patch("builtins.print")
@patch("stt.repl.Recorder")
@patch("stt.repl.WhisperTranscriber")
def test_repl_stop_keyword_still_works(
    mock_tr: MagicMock,
    mock_rec_class: MagicMock,
    mock_print: MagicMock,
) -> None:
    cfg = SttConfig()
    mock_rec = MagicMock()
    mock_rec.is_recording = False
    mock_rec.take_stream_reports.return_value = []
    mock_rec_class.return_value = mock_rec

    def start_side() -> None:
        mock_rec.is_recording = True

    def stop_side() -> np.ndarray:
        mock_rec.is_recording = False
        return np.zeros(800, dtype=np.float32)

    mock_rec.start.side_effect = start_side
    mock_rec.stop.side_effect = stop_side
    mock_tr_instance = MagicMock()
    mock_tr_instance.model_name = "turbo"
    mock_tr_instance.transcribe_array.return_value = "via stop"
    mock_tr.return_value = mock_tr_instance

    run_repl(cfg, input_fn=scripted_input(["", "stop", "quit"]))
    texts = [c.args[0] for c in mock_print.call_args_list if c.args]
    assert "via stop" in texts


@patch("stt.repl.print_input_devices_report")
def test_repl_mic_command_sets_device(mock_report: MagicMock) -> None:
    cfg = SttConfig(default_device=3)
    run_repl(cfg, input_fn=scripted_input(["mic airpods", "mic default", "quit"]))
    assert cfg.default_device is None
    assert mock_report.call_args_list[0].args == ("airpods",)


def test_repl_clip_toggle() -> None:
    cfg = SttConfig()
    run_repl(cfg, input_fn=scripted_input(["clip off", "quit"]))
    assert cfg.copy_to_clipboard is False
