from __future__ import annotations

from enum import Enum, auto


class AppState(Enum):
    IDLE = auto()
    RECORDING = auto()
    TRANSCRIBING = auto()


class StateMachine:
    def __init__(self) -> None:
        self.state = AppState.IDLE

    def start_recording(self) -> None:
        if self.state != AppState.IDLE:
            raise RuntimeError("already recording")
        self.state = AppState.RECORDING

    def stop_recording(self) -> None:
        if self.state != AppState.RECORDING:
            raise RuntimeError("no recording in progress")
        self.state = AppState.TRANSCRIBING

    def finish_transcription(self) -> None:
        if self.state != AppState.TRANSCRIBING:
            raise RuntimeError("not transcribing")
        self.state = AppState.IDLE

    def abort_to_idle(self) -> None:
        self.state = AppState.IDLE
