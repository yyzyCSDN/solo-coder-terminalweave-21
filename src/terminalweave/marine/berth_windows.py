from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta
from terminalweave.core.intervals import TimeInterval, merge_intervals
from terminalweave.core.errors import ValidationError

@dataclass(frozen=True)
class BerthClosure:
    berth_id: str
    interval: TimeInterval
    reason: str
    emergency: bool = False

class BerthWindowCalendar:

    def __init__(self):
        self.closures: dict[str, list[BerthClosure]] = {}

    def add_closure(self, closure: BerthClosure) -> None:
        self.closures.setdefault(closure.berth_id, []).append(closure)

    def available(self, berth_id: str, window: TimeInterval) -> list[TimeInterval]:
        blocked = merge_intervals((c.interval for c in self.closures.get(berth_id, []) if c.interval.overlaps(window)))
        cursor = window.start
        out = []
        for interval in blocked:
            start = max(window.start, interval.start)
            end = min(window.end, interval.end)
            if cursor < start:
                out.append(TimeInterval(cursor, start))
            cursor = max(cursor, end)
        if cursor < window.end:
            out.append(TimeInterval(cursor, window.end))
        return out

    def first_fit(self, berth_id: str, window: TimeInterval, duration: timedelta) -> TimeInterval | None:
        if duration.total_seconds() <= 0:
            raise ValidationError('duration must be positive')
        for free in self.available(berth_id, window):
            if free.duration >= duration:
                return TimeInterval(free.start, free.start + duration)
        return None
