"""Coalesce animation repaint requests to the operator's presentation rate."""

from __future__ import annotations

import time
from typing import Callable

from kitogl._qt import QObject, Qt, QTimer


class FrameUpdates:
    """Coalesce repaint requests and preserve the phase of a capped frame clock."""

    def __init__(self, parent: QObject, update: Callable[[], None]) -> None:
        self._update = update
        self._timer = QTimer(parent)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.request)
        self._interval = 0.0
        self._deadline = None

    def set_rate(self, fps: float) -> None:
        interval = 1.0 / fps if fps > 0 else 0.0
        if interval == self._interval:
            return
        pending = self._timer.isActive()
        self._timer.stop()
        self._interval = interval
        self._deadline = None
        if pending:
            self.request()

    def reset(self) -> None:
        self._timer.stop()
        self._deadline = None

    def request(self) -> None:
        if not self._interval:
            self._update()
            return
        if self._timer.isActive():
            return
        now = time.perf_counter()
        deadline = self._deadline
        if deadline is not None:
            delay_ms = round((deadline - now) * 1000)
            if delay_ms > 0:
                self._timer.start(delay_ms)
                return
        # Keep the phase despite millisecond rounding. After a stall, skip old deadlines.
        if deadline is None or now - deadline >= self._interval:
            deadline = now
        self._deadline = deadline + self._interval
        self._update()
