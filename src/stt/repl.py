from __future__ import annotations

import time
from collections import deque
from typing import Callable

import numpy as np

from stt.config import SttConfig
from stt.models import validate_model
from stt.recorder import Recorder
from stt.sessions import new_session_id, save_session
from stt.state import AppState, StateMachine
from stt.transcriber import WhisperTranscriber
from stt.util import audio_sounds_empty, eprint, format_audio_summary, read_line

_QUIT_ALIASES = frozenset({"quit", "exit", "q", "esc", "escape"})


def _is_quit_command(stripped: str) -> bool:
    return stripped.lower() in _QUIT_ALIASES


def _report_capture(recorder: Recorder, audio: np.ndarray, cfg: SttConfig) -> None:
    for msg in recorder.take_stream_reports():
        eprint(f"warning: audio stream: {msg}")
    suspect, hint = audio_sounds_empty(audio, cfg.sample_rate)
    if cfg.ui.show_status or suspect:
        eprint(format_audio_summary(audio, cfg.sample_rate))
    if hint:
        eprint(f"warning: {hint}")


def run_repl(
    cfg: SttConfig,
    *,
    input_fn: Callable[[str], str] | None = None,
    verbose: bool = False,
    debug: bool = False,
) -> int:
    read: Callable[[str], str] = input_fn or read_line
    recorder = Recorder(
        sample_rate=cfg.sample_rate,
        channels=cfg.channels,
        device=cfg.default_device,
    )
    transcriber = WhisperTranscriber(cfg)
    fsm = StateMachine()

    eprint("ready")
    eprint("[Enter] start recording, [Enter] again to stop and transcribe")

    while True:
        try:
            prompt = "> " if fsm.state != AppState.RECORDING else "recording> "
            line = read(prompt).rstrip("\n")
        except EOFError:
            eprint("")
            if fsm.state == AppState.RECORDING:
                eprint("error: stop recording first")
                return 1
            return 0
        except KeyboardInterrupt:
            eprint("")
            if fsm.state == AppState.RECORDING:
                try:
                    recorder.stop()
                except RuntimeError:
                    pass
                fsm.abort_to_idle()
                eprint("recording cancelled")
            continue

        stripped = line.strip()

        if fsm.state == AppState.RECORDING:
            if stripped == "stop" or stripped == "":
                try:
                    audio = recorder.stop()
                except RuntimeError as e:
                    eprint(f"error: {e}")
                    fsm.abort_to_idle()
                    continue
                try:
                    fsm.stop_recording()
                except RuntimeError:
                    fsm.abort_to_idle()
                    continue
                if cfg.ui.show_status or verbose:
                    eprint("recording stopped")
                _report_capture(recorder, audio, cfg)
                if debug:
                    eprint(f"debug: samples={len(audio)} duration_s={len(audio)/cfg.sample_rate:.2f}")
                if cfg.ui.show_status or verbose:
                    eprint(f"transcribing with model={transcriber.model_name}")
                t0 = time.perf_counter()
                try:
                    text = transcriber.transcribe_array(audio)
                except Exception as e:
                    fsm.abort_to_idle()
                    eprint(f"error: failed to transcribe audio: {e}")
                    continue
                if debug:
                    eprint(f"debug: transcribe_s={time.perf_counter() - t0:.2f}")
                print(text, flush=True)
                if cfg.save_recordings:
                    sid, created = new_session_id()
                    try:
                        save_session(
                            cfg.sessions_path(),
                            sid,
                            created,
                            audio,
                            text,
                            model=transcriber.model_name,
                            language=cfg.language,
                            sample_rate=cfg.sample_rate,
                            channels=cfg.channels,
                        )
                    except OSError as e:
                        eprint(f"error: could not save session: {e}")
                try:
                    fsm.finish_transcription()
                except RuntimeError:
                    fsm.abort_to_idle()
            elif _is_quit_command(stripped):
                eprint("error: stop recording first")
            elif stripped == "help":
                eprint(
                    "while recording: Enter or 'stop' to finish; "
                    "q / quit / esc only work when not recording"
                )
            else:
                eprint("press Enter or type 'stop' to finish recording")
            continue

        # IDLE
        if stripped == "stop":
            eprint("error: no recording in progress")
            continue

        if stripped == "":
            try:
                recorder.start()
            except RuntimeError as e:
                eprint(f"error: {e}")
                continue
            try:
                fsm.start_recording()
            except RuntimeError as e:
                try:
                    recorder.stop()
                except RuntimeError:
                    pass
                eprint(f"error: {e}")
                continue
            if cfg.ui.show_status or verbose:
                eprint("recording started")
            eprint("recording... press Enter (or type 'stop') to finish")
            continue

        if _is_quit_command(stripped):
            return 0

        if stripped == "help":
            eprint(
                "Enter=start, Enter again (or stop)=finish+transcribe, "
                "model <name>, save on|off, status, q/quit/esc"
            )
            continue

        if stripped == "status":
            eprint(
                f"model={transcriber.model_name} language={cfg.language} "
                f"save_recordings={cfg.save_recordings} sample_rate={cfg.sample_rate} "
                f"channels={cfg.channels} device={cfg.default_device}"
            )
            if debug:
                eprint(f"debug: config={cfg!r}")
            continue

        if stripped.startswith("model "):
            name = stripped.split(" ", 1)[1].strip()
            if not name:
                eprint("error: model name required")
                continue
            try:
                validate_model(name)
            except ValueError as e:
                eprint(f"error: {e}")
                continue
            cfg.model = name
            transcriber.set_model(name)
            eprint(f"model set to {name}")
            continue

        if stripped.startswith("save "):
            arg = stripped.split(" ", 1)[1].strip().lower()
            if arg == "on":
                cfg.save_recordings = True
                eprint("save recordings on")
            elif arg == "off":
                cfg.save_recordings = False
                eprint("save recordings off")
            else:
                eprint("error: use 'save on' or 'save off'")
            continue

        eprint("unknown command; try 'help'")


def run_once(cfg: SttConfig, *, verbose: bool = False, debug: bool = False) -> int:
    recorder = Recorder(
        sample_rate=cfg.sample_rate,
        channels=cfg.channels,
        device=cfg.default_device,
    )
    transcriber = WhisperTranscriber(cfg)
    eprint("recording... press Enter (empty line) or type 'stop' to finish")
    try:
        recorder.start()
    except RuntimeError as e:
        eprint(f"error: {e}")
        return 1
    try:
        while True:
            try:
                line = read_line("recording> ").strip()
            except (EOFError, KeyboardInterrupt):
                eprint("")
                return 1
            low = line.lower()
            if low in _QUIT_ALIASES:
                eprint("aborted")
                return 1
            if line.lower() == "stop" or line == "":
                break
            eprint("press Enter, type 'stop', or q / esc to cancel")
        try:
            audio = recorder.stop()
        except RuntimeError as e:
            eprint(f"error: {e}")
            return 1
        _report_capture(recorder, audio, cfg)
        if cfg.ui.show_status or verbose:
            eprint(f"transcribing with model={transcriber.model_name}")
        if debug:
            eprint(f"debug: samples={len(audio)}")
        try:
            text = transcriber.transcribe_array(audio)
        except Exception as e:
            eprint(f"error: failed to transcribe audio: {e}")
            return 1
        print(text, flush=True)
        if cfg.save_recordings:
            sid, created = new_session_id()
            try:
                save_session(
                    cfg.sessions_path(),
                    sid,
                    created,
                    audio,
                    text,
                    model=transcriber.model_name,
                    language=cfg.language,
                    sample_rate=cfg.sample_rate,
                    channels=cfg.channels,
                )
            except OSError as e:
                eprint(f"error: could not save session: {e}")
        return 0
    finally:
        if recorder.is_recording:
            try:
                recorder.stop()
            except RuntimeError:
                pass


def scripted_input(lines: list[str]) -> Callable[[str], str]:
    q: deque[str] = deque(lines)

    def fn(_prompt: str) -> str:
        if not q:
            raise EOFError
        return q.popleft()

    return fn
