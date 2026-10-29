"""Paste text into whatever app has focus (macOS)."""

from __future__ import annotations

import subprocess
import time

from stt.util import copy_to_clipboard

_KEYCODE_V = 9


def _read_clipboard() -> str | None:
    try:
        out = subprocess.run(["pbpaste"], capture_output=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.decode("utf-8", errors="replace")


def _press_cmd_v() -> None:
    import Quartz

    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    for down in (True, False):
        ev = Quartz.CGEventCreateKeyboardEvent(src, _KEYCODE_V, down)
        Quartz.CGEventSetFlags(ev, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)


def paste_text(text: str, *, restore_clipboard: bool = True) -> str | None:
    """Type `text` into the focused app via Cmd+V. Returns an error message or None.

    Needs Accessibility permission for the host app. Only plain-text clipboard contents
    are restored afterwards (images/files on the clipboard are not).
    """
    previous = _read_clipboard() if restore_clipboard else None
    err = copy_to_clipboard(text)
    if err:
        return err
    try:
        _press_cmd_v()
    except Exception as e:  # pyobjc raises assorted types
        return str(e)
    if previous is not None:
        time.sleep(0.25)  # let the target app read the pasteboard before we swap it back
        copy_to_clipboard(previous)
    return None
