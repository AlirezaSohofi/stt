"""Global hold-to-talk hotkey (hold the Option key) via a macOS Quartz event tap."""

from __future__ import annotations

import queue
import threading
from typing import Callable

# kVK_Option / kVK_RightOption
_OPTION_KEYCODES = (58, 61)


class HotkeyPermissionError(RuntimeError):
    pass


class HoldOptionHotkey:
    """Calls on_start after Option is held alone for `hold_ms`, and on_stop on release.

    Option used in a combo (Option+letter for special characters, Option+click, Option with
    another modifier) never triggers: the pending start is cancelled, and a recording that
    already began is aborted through on_cancel.
    """

    def __init__(
        self,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        on_cancel: Callable[[], None],
        hold_ms: int = 250,
    ) -> None:
        self._on_start = on_start
        self._on_stop = on_stop
        self._on_cancel = on_cancel
        self._hold_s = hold_ms / 1000.0
        self._lock = threading.Lock()
        self._option_down = False
        self._tainted = False
        self._active = False
        self._timer: threading.Timer | None = None
        self._tap = None
        # One ordered worker so start/stop/cancel never overtake each other.
        self._events: queue.Queue[Callable[[], None]] = queue.Queue()
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        while True:
            self._events.get()()

    # -- event tap -----------------------------------------------------------------
    def install(self) -> None:
        """Attach the tap to the current (main) run loop. Call from the main thread."""
        import Quartz

        preflight = getattr(Quartz, "CGPreflightListenEventAccess", None)
        if preflight is not None and not preflight():
            getattr(Quartz, "CGRequestListenEventAccess", lambda: None)()

        mask = (
            Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged)
            | Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)
            | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseDown)
            | Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseDown)
        )

        def callback(proxy, etype, event, refcon):
            if etype in (
                Quartz.kCGEventTapDisabledByTimeout,
                Quartz.kCGEventTapDisabledByUserInput,
            ):
                Quartz.CGEventTapEnable(self._tap, True)
                return event
            if etype == Quartz.kCGEventFlagsChanged:
                flags = Quartz.CGEventGetFlags(event)
                keycode = Quartz.CGEventGetIntegerValueField(
                    event, Quartz.kCGKeyboardEventKeycode
                )
                self._flags_changed(flags, keycode, Quartz)
            else:  # key or mouse press while Option may be held
                self._other_input()
            return event

        self._tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            mask,
            callback,
            None,
        )
        if self._tap is None:
            raise HotkeyPermissionError(
                "could not create the key listener. Grant Input Monitoring "
                "to the app running stt in System Settings → Privacy & "
                "Security, then fully restart it."
            )
        source = Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
        Quartz.CFRunLoopAddSource(
            Quartz.CFRunLoopGetCurrent(), source, Quartz.kCFRunLoopCommonModes
        )
        Quartz.CGEventTapEnable(self._tap, True)

    # -- state machine -------------------------------------------------------------
    def _flags_changed(self, flags: int, keycode: int, Quartz) -> None:
        option = bool(flags & Quartz.kCGEventFlagMaskAlternate)
        others = bool(
            flags
            & (
                Quartz.kCGEventFlagMaskCommand
                | Quartz.kCGEventFlagMaskControl
                | Quartz.kCGEventFlagMaskShift
                | Quartz.kCGEventFlagMaskSecondaryFn
            )
        )
        if keycode not in _OPTION_KEYCODES and not self._option_down:
            return
        with self._lock:
            if option and not self._option_down:
                self._option_down = True
                self._tainted = others
                if not others:
                    self._arm_timer()
            elif option and others:
                self._taint_locked()
            elif not option and self._option_down:
                self._option_down = False
                self._disarm_timer()
                was_active, self._active = self._active, False
                tainted = self._tainted
                self._tainted = False
                if was_active and not tainted:
                    self._fire(self._on_stop)

    def _other_input(self) -> None:
        with self._lock:
            if self._option_down:
                self._taint_locked()

    def _taint_locked(self) -> None:
        self._tainted = True
        self._disarm_timer()
        if self._active:
            self._active = False
            self._fire(self._on_cancel)

    def _arm_timer(self) -> None:
        self._disarm_timer()
        self._timer = threading.Timer(self._hold_s, self._timer_fired)
        self._timer.daemon = True
        self._timer.start()

    def _disarm_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _timer_fired(self) -> None:
        with self._lock:
            if not self._option_down or self._tainted or self._active:
                return
            self._active = True
        self._fire(self._on_start)

    def _fire(self, fn: Callable[[], None]) -> None:
        # Never run app logic inside the event-tap callback: a slow handler makes macOS
        # disable the tap.
        self._events.put(fn)
