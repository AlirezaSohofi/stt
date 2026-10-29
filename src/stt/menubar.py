"""Menu-bar app: hold Option anywhere to dictate; release to transcribe and paste."""

from __future__ import annotations

import threading
import time

import rumps
from AppKit import NSSound
from PyObjCTools import AppHelper

from stt.config import SttConfig
from stt.hotkey import HoldOptionHotkey, HotkeyPermissionError
from stt.hud import Hud
from stt.models import WHISPER_MODEL_NAMES
from stt.paste import paste_text
from stt.recorder import Recorder
from stt.sessions import save_if_enabled
from stt.state import AppState, StateMachine
from stt.transcriber import WhisperTranscriber
from stt.util import audio_sounds_empty, copy_to_clipboard, eprint

_ICON_IDLE, _ICON_REC, _ICON_BUSY = "🎙", "🔴", "⏳"
_MODELS = ("tiny", "base", "small", "medium", "turbo", "large")


def _sound(name: str) -> None:
    def play() -> None:
        s = NSSound.soundNamed_(name)
        if s is not None:
            s.play()

    AppHelper.callAfter(play)


class SttMenuBarApp(rumps.App):
    def __init__(self, cfg: SttConfig) -> None:
        super().__init__("stt", title=_ICON_BUSY, quit_button=None)
        self.cfg = cfg
        self.recorder = Recorder(
            sample_rate=cfg.sample_rate, channels=cfg.channels, device=cfg.default_device
        )
        self.transcriber = WhisperTranscriber(cfg)
        self.hud = Hud(level_source=lambda: self.recorder.level)
        self.fsm = StateMachine()
        self._lock = threading.Lock()
        self._rec_started = 0.0
        self._ready = False

        self._status = rumps.MenuItem("Loading model…")
        self._paste_item = rumps.MenuItem("Auto-paste", callback=self._toggle_paste)
        self._paste_item.state = cfg.app.auto_paste
        self._sound_item = rumps.MenuItem("Sounds", callback=self._toggle_sounds)
        self._sound_item.state = cfg.app.sounds
        self._model_items = {}
        model_menu = rumps.MenuItem("Model")
        for name in _MODELS:
            item = rumps.MenuItem(name, callback=self._pick_model)
            item.state = name == cfg.model
            self._model_items[name] = item
            model_menu.add(item)
        self.menu = [
            self._status,
            None,
            self._paste_item,
            self._sound_item,
            model_menu,
            None,
            rumps.MenuItem("Quit stt", callback=lambda _: rumps.quit_application()),
        ]
        self.hotkey = HoldOptionHotkey(
            self._start, self._stop, self._cancel, hold_ms=cfg.app.hold_ms
        )

    # -- UI helpers (always hop to the main thread) --------------------------------
    def _set_title(self, icon: str) -> None:
        AppHelper.callAfter(lambda: setattr(self, "title", icon))

    def _set_status(self, text: str) -> None:
        AppHelper.callAfter(lambda: setattr(self._status, "title", text))

    def _hud(self, mode: str, message: str = "", hide_after: float | None = None) -> None:
        if self.cfg.app.show_hud:
            AppHelper.callAfter(self.hud.show, mode, message, hide_after)

    def _hud_hide(self) -> None:
        AppHelper.callAfter(self.hud.hide)

    def _tick(self, name: str) -> None:
        if self.cfg.app.sounds:
            _sound(name)

    # -- lifecycle -----------------------------------------------------------------
    def run(self, **kwargs) -> None:  # type: ignore[override]
        threading.Thread(target=self._warm_up, daemon=True).start()
        AppHelper.callAfter(self._install_hotkey)
        super().run(**kwargs)

    def _install_hotkey(self) -> None:
        try:
            self.hotkey.install()
        except HotkeyPermissionError as e:
            eprint(f"error: {e}")
            rumps.alert("stt needs permission", str(e))
            rumps.quit_application()

    def _warm_up(self) -> None:
        eprint(f"loading model={self.transcriber.model_name} backend={self.transcriber.backend}…")
        t0 = time.perf_counter()
        try:
            self.transcriber.preload()
        except Exception as e:
            eprint(f"error: could not load model: {e}")
            self._set_status(f"Model error: {e}")
            self._set_title("⚠️")
            return
        eprint(f"ready in {time.perf_counter() - t0:.1f}s — hold Option to dictate")
        self._ready = True
        self._set_status(f"Ready · {self.transcriber.model_name} ({self.transcriber.backend})")
        self._set_title(_ICON_IDLE)

    # -- menu callbacks ------------------------------------------------------------
    def _toggle_paste(self, item) -> None:
        item.state = not item.state
        self.cfg.app.auto_paste = bool(item.state)

    def _toggle_sounds(self, item) -> None:
        item.state = not item.state
        self.cfg.app.sounds = bool(item.state)

    def _pick_model(self, item) -> None:
        name = item.title
        if name not in WHISPER_MODEL_NAMES or name == self.transcriber.model_name:
            return
        for n, it in self._model_items.items():
            it.state = n == name
        self.cfg.model = name
        self.transcriber.set_model(name)
        self._ready = False
        self._set_title(_ICON_BUSY)
        self._set_status(f"Loading {name}…")
        threading.Thread(target=self._warm_up, daemon=True).start()

    # -- hotkey handlers (run on the hotkey's ordered worker thread) ---------------
    def _start(self) -> None:
        with self._lock:
            if not self._ready:
                self._hud("notice", "Model still loading…", hide_after=1.5)
                return
            if self.fsm.state != AppState.IDLE:
                return
            try:
                self.recorder.start()
            except RuntimeError as e:
                self._hud("notice", "Mic error", hide_after=2.0)
                eprint(f"error: {e}")
                return
            self.fsm.start_recording()
            self._rec_started = time.monotonic()
        if self.recorder.device_warning:
            eprint(f"warning: {self.recorder.device_warning}")
        self._set_title(_ICON_REC)
        self._hud("recording")
        self._tick("Tink")

    def _cancel(self) -> None:
        with self._lock:
            if self.fsm.state != AppState.RECORDING:
                return
            try:
                self.recorder.stop()
            except RuntimeError:
                pass
            self.fsm.abort_to_idle()
        self._set_title(_ICON_IDLE)
        self._hud_hide()

    def _stop(self) -> None:
        with self._lock:
            if self.fsm.state != AppState.RECORDING:
                return
            try:
                audio = self.recorder.stop()
            except RuntimeError as e:
                eprint(f"error: {e}")
                self.fsm.abort_to_idle()
                self._set_title(_ICON_IDLE)
                self._hud_hide()
                return
            self.fsm.stop_recording()
            held = time.monotonic() - self._rec_started
        self._tick("Pop")
        if held * 1000 < self.cfg.app.min_record_ms:
            self._finish_idle()
            self._hud_hide()
            return
        suspect, hint = audio_sounds_empty(audio, self.cfg.sample_rate)
        if suspect:
            eprint(f"warning: {hint}")
            self._finish_idle()
            self._hud("notice", "No audio picked up", hide_after=2.0)
            return
        self._set_title(_ICON_BUSY)
        self._hud("transcribing")
        threading.Thread(target=self._transcribe, args=(audio,), daemon=True).start()

    def _finish_idle(self) -> None:
        with self._lock:
            self.fsm.abort_to_idle()
        self._set_title(_ICON_IDLE)

    def _transcribe(self, audio) -> None:
        try:
            t0 = time.perf_counter()
            text = self.transcriber.transcribe_array(audio)
            eprint(f"transcribed in {time.perf_counter() - t0:.1f}s: {text!r}")
        except Exception as e:
            eprint(f"error: failed to transcribe audio: {e}")
            self._finish_idle()
            self._hud("notice", "Transcription failed", hide_after=2.5)
            return
        if not text:
            self._finish_idle()
            self._hud("notice", "Nothing heard", hide_after=1.5)
            return
        outcome = "Copied"
        if self.cfg.app.auto_paste:
            err = paste_text(text, restore_clipboard=self.cfg.app.restore_clipboard)
            if err:
                eprint(f"warning: could not paste: {err}")
                copy_to_clipboard(text)
                outcome = "Copied (paste failed)"
            else:
                outcome = "Pasted"
        else:
            copy_to_clipboard(text)
        save_if_enabled(self.cfg, audio, text, self.transcriber.model_name)
        self._finish_idle()
        self._hud("done", outcome, hide_after=1.0)


def run_menubar(cfg: SttConfig) -> int:
    SttMenuBarApp(cfg).run()
    return 0
