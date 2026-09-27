"""Continuous safety guarding for vessel operations while alongside.

Conditions change while a vessel is working a berth: the tide falls, the
weather worsens, mooring lines overload, or the berth is closed. This module
keeps watching operations that are already in progress and reapplies the
latest information on every evaluation.

Each guarded source returns one of three verdicts:

* ``nominal``  - work may continue;
* ``caution``  - pause and hold at the last confirmed checkpoint;
* ``critical`` - evacuate the berth.

The worst verdict across all guarded sources is applied to the operation.
Safety rules observed here:

* completed operations are never rolled back or reopened;
* missing, stale or out-of-range inputs fail closed as ``caution``;
* after conditions recover, work resumes from the last confirmed checkpoint;
* an evacuated vessel must be reported back alongside before it can resume.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from terminalweave.core.clock import ensure_utc
from terminalweave.core.errors import (
    ConflictError,
    NotFoundError,
    SafetyConstraintError,
    StateTransitionError,
    ValidationError,
)
from terminalweave.core.intervals import TimeInterval
from terminalweave.marine.berth_windows import BerthClosure, BerthWindowCalendar
from terminalweave.marine.mooring import (
    MooringChecker,
    MooringLine,
    MooringRequirement,
)
from terminalweave.marine.tide import TideCurve, UnderKeelRequirement
from terminalweave.marine.weather import (
    MarineWeatherLimits,
    WeatherSample,
    WeatherWindowChecker,
)


class HazardLevel(str, Enum):
    NOMINAL = 'nominal'
    CAUTION = 'caution'
    CRITICAL = 'critical'


class GuardAction(str, Enum):
    CONTINUE = 'continue'
    PAUSE = 'pause'
    EVACUATE = 'evacuate'


class Source(str, Enum):
    TIDE = 'tide'
    WEATHER = 'weather'
    MOORING = 'mooring'
    BERTH_CLOSURE = 'berth_closure'


class OperationPhase(str, Enum):
    ACTIVE = 'active'
    PAUSED = 'paused'
    EVACUATED = 'evacuated'
    COMPLETED = 'completed'


_LEVEL_ACTION = {
    HazardLevel.NOMINAL: GuardAction.CONTINUE,
    HazardLevel.CAUTION: GuardAction.PAUSE,
    HazardLevel.CRITICAL: GuardAction.EVACUATE,
}
_LEVEL_RANK = {
    HazardLevel.NOMINAL: 0,
    HazardLevel.CAUTION: 1,
    HazardLevel.CRITICAL: 2,
}


@dataclass(frozen=True)
class SafetyFinding:
    source: Source
    level: HazardLevel
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class TideGuardPolicy:
    requirement: UnderKeelRequirement
    caution_buffer_m: float = 0.3

    def __post_init__(self) -> None:
        if self.caution_buffer_m < 0:
            raise ValidationError('tide caution buffer cannot be negative')


@dataclass(frozen=True)
class WeatherGuardPolicy:
    working_limits: MarineWeatherLimits
    evacuation_limits: MarineWeatherLimits | None = None
    max_staleness: timedelta | None = timedelta(minutes=30)

    def __post_init__(self) -> None:
        if self.max_staleness is not None and self.max_staleness.total_seconds() <= 0:
            raise ValidationError('weather staleness window must be positive')
        evac = self.evacuation_limits
        work = self.working_limits
        if evac is not None and (
            evac.max_wind_kn < work.max_wind_kn
            or evac.max_gust_kn < work.max_gust_kn
            or evac.min_visibility_nm > work.min_visibility_nm
            or evac.max_wave_m < work.max_wave_m
        ):
            raise ValidationError('evacuation limits must not be tighter than working limits')


@dataclass(frozen=True)
class MooringGuardPolicy:
    requirement: MooringRequirement
    max_staleness: timedelta | None = timedelta(minutes=30)

    def __post_init__(self) -> None:
        if self.max_staleness is not None and self.max_staleness.total_seconds() <= 0:
            raise ValidationError('mooring staleness window must be positive')


@dataclass(frozen=True)
class ClosureGuardPolicy:
    # A closure starting within this lead time pauses work so the berth can be cleared.
    imminent_lead: timedelta = timedelta(minutes=30)

    def __post_init__(self) -> None:
        if self.imminent_lead.total_seconds() < 0:
            raise ValidationError('closure lead time cannot be negative')


@dataclass(frozen=True)
class GuardEvent:
    at: datetime
    kind: str
    reasons: tuple[str, ...] = ()
    resumed_from: datetime | None = None


@dataclass
class WatchedOperation:
    call_id: str
    berth_id: str
    started_at: datetime
    phase: OperationPhase = OperationPhase.ACTIVE
    last_confirmed_at: datetime | None = None
    completed_units: float = 0.0
    last_reasons: tuple[str, ...] | None = None
    events: list[GuardEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.started_at = ensure_utc(self.started_at)
        if self.last_confirmed_at is not None:
            self.last_confirmed_at = ensure_utc(self.last_confirmed_at)

    @property
    def checkpoint(self) -> tuple[datetime | None, float]:
        return self.last_confirmed_at, self.completed_units


@dataclass(frozen=True)
class GuardDecision:
    call_id: str
    at: datetime
    action: GuardAction
    phase: OperationPhase
    previous_phase: OperationPhase
    findings: tuple[SafetyFinding, ...]
    resumed_from: datetime | None = None
    notes: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return self.phase != self.previous_phase


class BerthSafetyWatch:
    """Applies live tide/weather/mooring/closure data to watched operations."""

    def __init__(
        self,
        tide_policy: TideGuardPolicy | None = None,
        weather_policy: WeatherGuardPolicy | None = None,
        mooring_policy: MooringGuardPolicy | None = None,
        closure_policy: ClosureGuardPolicy | None = None,
        closures: BerthWindowCalendar | None = None,
    ) -> None:
        if not any((tide_policy, weather_policy, mooring_policy)):
            raise ValidationError('safety watch needs at least one guarded live source')
        self.tide_policy = tide_policy
        self.weather_policy = weather_policy
        self.mooring_policy = mooring_policy
        self.closure_policy = closure_policy or ClosureGuardPolicy()
        self._closures = closures or BerthWindowCalendar()
        self._operations: dict[str, WatchedOperation] = {}
        self._tide: TideCurve | None = None
        self._weather: WeatherSample | None = None
        self._mooring: dict[str, tuple[tuple[MooringLine, ...], datetime]] = {}
        self._mooring_checker = MooringChecker()
        self._weather_checker = (
            WeatherWindowChecker(weather_policy.working_limits) if weather_policy else None
        )
        self._evac_weather_checker = (
            WeatherWindowChecker(weather_policy.evacuation_limits)
            if weather_policy and weather_policy.evacuation_limits
            else None
        )

    @property
    def operations(self) -> tuple[WatchedOperation, ...]:
        return tuple(sorted(self._operations.values(), key=lambda op: op.started_at))

    # ------------------------------------------------------------------ setup

    def watch(self, call_id: str, berth_id: str, started_at: datetime) -> WatchedOperation:
        if call_id in self._operations:
            raise ConflictError(f'operation already watched: {call_id}')
        started_at = ensure_utc(started_at)
        op = WatchedOperation(
            call_id=call_id,
            berth_id=berth_id,
            started_at=started_at,
            last_confirmed_at=started_at,
            events=[GuardEvent(started_at, 'watch_started')],
        )
        self._operations[call_id] = op
        return op

    def get(self, call_id: str) -> WatchedOperation:
        try:
            return self._operations[call_id]
        except KeyError as exc:
            raise NotFoundError(f'operation is not watched: {call_id}') from exc

    # ------------------------------------------------------------ data feeds

    def update_tide(self, curve: TideCurve) -> None:
        self._tide = curve

    def report_weather(self, sample: WeatherSample) -> None:
        self._weather = sample

    def report_mooring(
        self, call_id: str, lines: list[MooringLine] | tuple[MooringLine, ...], at: datetime
    ) -> None:
        self.get(call_id)
        at = ensure_utc(at)
        if not lines:
            raise ValidationError('mooring report needs at least one line')
        self._mooring[call_id] = (tuple(lines), at)

    def add_closure(self, closure: BerthClosure) -> None:
        self._closures.add_closure(closure)

    # ----------------------------------------------------------- progress

    def confirm_progress(
        self, call_id: str, at: datetime, units: float = 0.0
    ) -> WatchedOperation:
        op = self.get(call_id)
        at = ensure_utc(at)
        if op.phase == OperationPhase.COMPLETED:
            raise StateTransitionError('completed operation cannot take more progress')
        if op.phase != OperationPhase.ACTIVE:
            raise StateTransitionError('progress can only be confirmed while active')
        if units < 0:
            raise ValidationError('confirmed units cannot be negative')
        if op.last_confirmed_at is not None and at < op.last_confirmed_at:
            raise ValidationError('confirmation time cannot move backwards')
        self._check_chronology(op, at)
        op.completed_units += units
        op.last_confirmed_at = at
        op.events.append(GuardEvent(at, 'progress_confirmed'))
        return op

    def mark_returned(self, call_id: str, at: datetime) -> WatchedOperation:
        """Record that an evacuated vessel is back all-fast alongside."""
        op = self.get(call_id)
        at = ensure_utc(at)
        if op.phase != OperationPhase.EVACUATED:
            raise StateTransitionError('only an evacuated operation can be marked returned')
        self._check_chronology(op, at)
        op.phase = OperationPhase.PAUSED
        op.events.append(GuardEvent(at, 'returned'))
        return op

    def complete(self, call_id: str, at: datetime) -> WatchedOperation:
        op = self.get(call_id)
        at = ensure_utc(at)
        if op.phase != OperationPhase.ACTIVE:
            raise StateTransitionError('only an active operation can be completed')
        self._check_chronology(op, at)
        op.phase = OperationPhase.COMPLETED
        op.last_confirmed_at = at
        op.events.append(GuardEvent(at, 'completed'))
        return op

    # ------------------------------------------------------------ evaluation

    def evaluate_all(self, at: datetime) -> list[GuardDecision]:
        at = ensure_utc(at)
        return [
            self.evaluate(op.call_id, at)
            for op in self._operations.values()
            if op.phase != OperationPhase.COMPLETED
        ]

    def evaluate(self, call_id: str, at: datetime) -> GuardDecision:
        op = self.get(call_id)
        at = ensure_utc(at)
        self._check_chronology(op, at)
        previous = op.phase

        # Completed work is terminal: later hazards never roll it back.
        if op.phase == OperationPhase.COMPLETED:
            return GuardDecision(
                call_id, at, GuardAction.CONTINUE, op.phase, op.phase, ()
            )

        findings = tuple(
            finding
            for finding in (
                self._tide_finding(at) if self.tide_policy else None,
                self._weather_finding(at) if self.weather_policy else None,
                self._mooring_finding(op, at) if self.mooring_policy else None,
                self._closure_finding(op, at),
            )
            if finding is not None
        )
        worst = max(findings, key=lambda f: _LEVEL_RANK[f.level]).level
        action = _LEVEL_ACTION[worst]
        return self._apply(op, at, action, findings, previous)

    def _apply(
        self,
        op: WatchedOperation,
        at: datetime,
        action: GuardAction,
        findings: tuple[SafetyFinding, ...],
        previous: OperationPhase,
    ) -> GuardDecision:
        resumed_from = None
        notes: tuple[str, ...] = ()
        kind: str | None = None

        if op.phase == OperationPhase.ACTIVE:
            if action == GuardAction.PAUSE:
                op.phase = OperationPhase.PAUSED
                kind = 'paused'
            elif action == GuardAction.EVACUATE:
                op.phase = OperationPhase.EVACUATED
                kind = 'evacuated'
            else:
                kind = 'continued'
        elif op.phase == OperationPhase.PAUSED:
            if action == GuardAction.CONTINUE:
                op.phase = OperationPhase.ACTIVE
                resumed_from = op.last_confirmed_at
                kind = 'resumed'
            elif action == GuardAction.EVACUATE:
                op.phase = OperationPhase.EVACUATED
                kind = 'evacuated'
            else:
                kind = 'held'
        else:  # EVACUATED: conditions cannot move the operation back automatically
            kind = 'awaiting_return'
            if action == GuardAction.CONTINUE:
                notes = ('conditions nominal; vessel must be marked returned before resuming',)

        reasons = _finding_reasons(findings)
        if op.phase != previous or reasons != op.last_reasons:
            op.events.append(GuardEvent(at, kind, reasons, resumed_from))
            op.last_reasons = reasons
        return GuardDecision(
            op.call_id,
            at,
            action,
            op.phase,
            previous,
            findings,
            resumed_from=resumed_from,
            notes=notes,
        )

    # --------------------------------------------------------- source checks

    def _tide_finding(self, at: datetime) -> SafetyFinding:
        policy = self.tide_policy
        if self._tide is None:
            return SafetyFinding(Source.TIDE, HazardLevel.CAUTION, ('tide data unavailable',))
        try:
            clearance = self._tide.clearance_m(at, policy.requirement)
        except SafetyConstraintError:
            return SafetyFinding(
                Source.TIDE, HazardLevel.CAUTION, ('tide curve does not cover evaluation time',)
            )
        minimum = policy.requirement.minimum_clearance_m
        if clearance < minimum:
            return SafetyFinding(
                Source.TIDE,
                HazardLevel.CRITICAL,
                (f'under-keel clearance {clearance:.2f}m below {minimum:.2f}m',),
            )
        if clearance < minimum + policy.caution_buffer_m:
            return SafetyFinding(
                Source.TIDE,
                HazardLevel.CAUTION,
                (f'under-keel clearance {clearance:.2f}m inside caution buffer',),
            )
        return SafetyFinding(Source.TIDE, HazardLevel.NOMINAL)

    def _weather_finding(self, at: datetime) -> SafetyFinding:
        policy = self.weather_policy
        sample = self._weather
        if sample is None:
            return SafetyFinding(Source.WEATHER, HazardLevel.CAUTION, ('weather data unavailable',))
        age = at - sample.at
        if age < timedelta(0):
            return SafetyFinding(
                Source.WEATHER, HazardLevel.CAUTION, ('weather sample timestamp in the future',)
            )
        if policy.max_staleness is not None and age > policy.max_staleness:
            return SafetyFinding(
                Source.WEATHER,
                HazardLevel.CAUTION,
                (f'weather data stale ({int(age.total_seconds() // 60)} min old)',),
            )
        if self._evac_weather_checker is not None and not self._evac_weather_checker.acceptable(sample):
            return SafetyFinding(
                Source.WEATHER,
                HazardLevel.CRITICAL,
                tuple('evacuation: ' + r for r in self._evac_weather_checker.reasons(sample)),
            )
        reasons = tuple(self._weather_checker.reasons(sample))
        if reasons:
            return SafetyFinding(Source.WEATHER, HazardLevel.CAUTION, reasons)
        return SafetyFinding(Source.WEATHER, HazardLevel.NOMINAL)

    def _mooring_finding(self, op: WatchedOperation, at: datetime) -> SafetyFinding:
        policy = self.mooring_policy
        report = self._mooring.get(op.call_id)
        if report is None:
            return SafetyFinding(Source.MOORING, HazardLevel.CAUTION, ('mooring data unavailable',))
        lines, reported_at = report
        age = at - reported_at
        if age < timedelta(0):
            return SafetyFinding(
                Source.MOORING, HazardLevel.CAUTION, ('mooring report timestamp in the future',)
            )
        if policy.max_staleness is not None and age > policy.max_staleness:
            return SafetyFinding(
                Source.MOORING,
                HazardLevel.CAUTION,
                (f'mooring data stale ({int(age.total_seconds() // 60)} min old)',),
            )
        try:
            self._mooring_checker.evaluate(list(lines), policy.requirement)
        except SafetyConstraintError as exc:
            message = str(exc)
            # An overloaded line can be eased with work paused; broken or
            # insufficient mooring strength leaves the ship unsafe alongside.
            level = HazardLevel.CAUTION if 'utilization' in message else HazardLevel.CRITICAL
            return SafetyFinding(Source.MOORING, level, (message,))
        return SafetyFinding(Source.MOORING, HazardLevel.NOMINAL)

    def _closure_finding(self, op: WatchedOperation, at: datetime) -> SafetyFinding:
        policy = self.closure_policy
        for closure in self._closures.closures.get(op.berth_id, ()):
            if closure.interval.contains(at):
                return SafetyFinding(
                    Source.BERTH_CLOSURE,
                    HazardLevel.CRITICAL,
                    (f'berth closed: {closure.reason}',),
                )
        if policy.imminent_lead.total_seconds() > 0:
            window = TimeInterval(at, at + policy.imminent_lead)
            for closure in self._closures.closures.get(op.berth_id, ()):
                if closure.interval.intersection(window) is not None:
                    return SafetyFinding(
                        Source.BERTH_CLOSURE,
                        HazardLevel.CAUTION,
                        (f'berth closure imminent: {closure.reason}',),
                    )
        return SafetyFinding(Source.BERTH_CLOSURE, HazardLevel.NOMINAL)

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _check_chronology(op: WatchedOperation, at: datetime) -> None:
        if at < op.started_at:
            raise ValidationError('evaluation precedes operation start')
        last_event = op.events[-1] if op.events else None
        if last_event is not None and at < last_event.at:
            raise ValidationError('evaluation time cannot move backwards')


def _finding_reasons(findings: tuple[SafetyFinding, ...]) -> tuple[str, ...]:
    return tuple(
        sorted(reason for finding in findings for reason in finding.reasons)
    )
