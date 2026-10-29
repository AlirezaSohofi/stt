"""Floating pill HUD (recording level meter / transcribing / result) for the menu-bar app."""

from __future__ import annotations

import time
from collections import deque

import objc
from AppKit import (
    NSAttributedString,
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSFont,
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSPanel,
    NSScreen,
    NSStatusWindowLevel,
    NSView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)
from Foundation import NSMakeRect, NSTimer

WIDTH, HEIGHT = 220.0, 44.0
_BARS = 22


def _color(r: float, g: float, b: float, a: float = 1.0):
    return NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, a)


_RED = _color(1.0, 0.27, 0.23)
_AMBER = _color(1.0, 0.72, 0.2)
_GREEN = _color(0.3, 0.85, 0.45)
_WHITE = _color(1, 1, 1, 0.92)
_DIM = _color(1, 1, 1, 0.55)


def _draw_text(s, x, y, color, size=13.0, weight=0.4):
    attrs = {
        NSFontAttributeName: NSFont.systemFontOfSize_weight_(size, weight),
        NSForegroundColorAttributeName: color,
    }
    NSAttributedString.alloc().initWithString_attributes_(s, attrs).drawAtPoint_((x, y))


class _PillView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(_PillView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.mode = "idle"
        self.message = ""
        self.levels = deque([0.0] * _BARS, maxlen=_BARS)
        self.level_source = None
        self.t0 = time.monotonic()
        return self

    def isOpaque(self):
        return False

    def drawRect_(self, rect):
        bounds = self.bounds()
        _color(0.09, 0.09, 0.1, 0.92).setFill()
        NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            bounds, HEIGHT / 2, HEIGHT / 2
        ).fill()
        elapsed = time.monotonic() - self.t0
        mid = HEIGHT / 2

        if self.mode == "recording":
            pulse = 0.55 + 0.45 * abs(((elapsed * 1.4) % 2.0) - 1.0)
            _color(1.0, 0.27, 0.23, pulse).setFill()
            NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(18, mid - 5, 10, 10)).fill()
            if self.level_source is not None:
                # Speech RMS is small; sqrt spreads quiet-to-loud across the bar height.
                self.levels.append(min(1.0, (self.level_source() * 8.0) ** 0.6))
            _RED.setFill()
            for i, lv in enumerate(self.levels):
                h = 3.0 + lv * 24.0
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    NSMakeRect(38 + i * 4.6, mid - h / 2, 2.8, h), 1.4, 1.4
                ).fill()
            secs = int(elapsed)
            _draw_text(f"{secs // 60}:{secs % 60:02d}", WIDTH - 44, mid - 8, _DIM, 12.0)
        elif self.mode == "transcribing":
            _AMBER.setFill()
            for i in range(3):
                phase = (elapsed * 2.0 - i * 0.25) % 1.0
                r = 3.0 + 2.5 * (1.0 - abs(phase * 2 - 1.0))
                NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(20 + i * 13 - r, mid - r, r * 2, r * 2)
                ).fill()
            _draw_text(self.message or "Transcribing…", 64, mid - 8, _WHITE)
        elif self.mode == "done":
            _GREEN.setFill()
            NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(18, mid - 5, 10, 10)).fill()
            _draw_text(self.message, 38, mid - 8, _WHITE)
        else:  # error / notice
            _AMBER.setFill()
            NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(18, mid - 5, 10, 10)).fill()
            _draw_text(self.message, 38, mid - 8, _WHITE)


class Hud:
    """All methods must be called on the main thread (use AppHelper.callAfter)."""

    def __init__(self, level_source=None) -> None:
        self._view = None
        self._panel = None
        self._timer = None
        self._hide_timer = None
        self._level_source = level_source

    def _ensure(self) -> None:
        if self._panel is not None:
            return
        rect = NSMakeRect(0, 0, WIDTH, HEIGHT)
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect,
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered,
            False,
        )
        panel.setLevel_(NSStatusWindowLevel)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(True)
        panel.setIgnoresMouseEvents_(True)
        panel.setHidesOnDeactivate_(False)
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        view = _PillView.alloc().initWithFrame_(rect)
        view.level_source = self._level_source
        panel.setContentView_(view)
        self._panel, self._view = panel, view

    def _place(self) -> None:
        screen = NSScreen.mainScreen() or NSScreen.screens()[0]
        vf = screen.visibleFrame()
        x = vf.origin.x + (vf.size.width - WIDTH) / 2
        y = vf.origin.y + vf.size.height - HEIGHT - 10
        self._panel.setFrameOrigin_((x, y))

    def show(self, mode: str, message: str = "", hide_after: float | None = None) -> None:
        self._ensure()
        view = self._view
        if mode != view.mode:
            view.t0 = time.monotonic()
        if mode == "recording":
            view.levels.extend([0.0] * _BARS)
        view.mode, view.message = mode, message
        self._cancel_hide()
        self._place()
        self._panel.orderFrontRegardless()
        view.setNeedsDisplay_(True)
        if self._timer is None:
            self._timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
                1 / 30, True, lambda _t: self._view.setNeedsDisplay_(True)
            )
        if hide_after:
            self._hide_timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
                hide_after, False, lambda _t: self.hide()
            )

    def _cancel_hide(self) -> None:
        if self._hide_timer is not None:
            self._hide_timer.invalidate()
            self._hide_timer = None

    def hide(self) -> None:
        self._cancel_hide()
        if self._timer is not None:
            self._timer.invalidate()
            self._timer = None
        if self._panel is not None:
            self._panel.orderOut_(None)
