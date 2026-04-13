from __future__ import annotations

import pytest

from stt.state import AppState, StateMachine


def test_happy_path() -> None:
    f = StateMachine()
    assert f.state == AppState.IDLE
    f.start_recording()
    assert f.state == AppState.RECORDING
    f.stop_recording()
    assert f.state == AppState.TRANSCRIBING
    f.finish_transcription()
    assert f.state == AppState.IDLE


def test_double_start() -> None:
    f = StateMachine()
    f.start_recording()
    with pytest.raises(RuntimeError, match="already recording"):
        f.start_recording()


def test_stop_while_idle() -> None:
    f = StateMachine()
    with pytest.raises(RuntimeError, match="no recording"):
        f.stop_recording()


def test_abort_to_idle() -> None:
    f = StateMachine()
    f.start_recording()
    f.stop_recording()
    f.abort_to_idle()
    assert f.state == AppState.IDLE
