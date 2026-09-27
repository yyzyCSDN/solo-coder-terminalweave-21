from __future__ import annotations
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum
from terminalweave.core.clock import ensure_utc
from terminalweave.core.errors import ValidationError, StateTransitionError, SafetyConstraintError
from terminalweave.marine.tide import TideCurve, UnderKeelRequirement
from terminalweave.marine.weather import WeatherSample, MarineWeatherLimits, WeatherWindowChecker
from terminalweave.marine.mooring import MooringLine, MooringRequirement
from terminalweave.marine.berth_windows import BerthClosure

TIDE_SOURCE = 'tide'
WEATHER_SOURCE = 'weather'
MOORING_SOURCE = 'mooring'
BERTH_CLOSURE_SOURCE = 'berth_closure'
SOURCES = (TIDE_SOURCE, WEATHER_SOURCE, MOORING_SOURCE, BERTH_CLOSURE_SOURCE)


class WatchVerdict(str, Enum):
    CONTINUE = 'continue'
    SUSPEND = 'suspend'
    EVACUATE = 'evacuate'


_SEVERITY = {WatchVerdict.CONTINUE: 0, WatchVerdict.SUSPEND: 1, WatchVerdict.EVACUATE: 2}


class WorkStepState(str, Enum):
    PENDING = 'pending'
    ACTIVE = 'active'
    PAUSED = 'paused'
    COMPLETED = 'completed'


class PlanState(str, Enum):
    IN_PROGRESS = 'in_progress'
    SUSPENDED = 'suspended'
    EVACUATED = 'evacuated'
    DONE = 'done'


@dataclass(frozen=True)
class HazardAssessment:
    source: str
    verdict: WatchVerdict
    reasons: tuple[str, ...] = ()
    gap: bool = False


@dataclass(frozen=True)
class WatchDecision:
    at: datetime
    verdict: WatchVerdict
    assessments: tuple[HazardAssessment, ...]
    applied: bool = False
    resumed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, 'at', ensure_utc(self.at))

    @property
    def gaps(self) -> tuple[str, ...]:
        return tuple(a.source for a in self.assessments if a.gap)

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(f'{a.source}: {reason}' for a in self.assessments for reason in a.reasons)

    def for_source(self, source: str) -> HazardAssessment | None:
        return next((a for a in self.assessments if a.source == source), None)


@dataclass(frozen=True)
class MooringSurvey:
    at: datetime
    lines: list[MooringLine]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'at', ensure_utc(self.at))


@dataclass
class WorkStep:
    step_id: str
    state: WorkStepState = WorkStepState.PENDING
    confirmed_at: datetime | None = None


@dataclass
class PlanEvent:
    at: datetime
    action: str
    detail: str = ''


@dataclass
class BerthWorkPlan:
    """A vessel operation already executing alongside a berth.

    Confirmed (completed) steps are append-only: pauses and evacuations only
    park the active step, they never roll completed work back.
    """
    call_id: str
    berth_id: str
    steps: list[WorkStep] = field(default_factory=list)
    state: PlanState = PlanState.IN_PROGRESS
    events: list[PlanEvent] = field(default_factory=list)
    last_event_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValidationError('work plan needs at least one step')
        if len({s.step_id for s in self.steps}) != len(self.steps):
            raise ValidationError('duplicate work step id')

    @property
    def completed_steps(self) -> tuple[WorkStep, ...]:
        return tuple(s for s in self.steps if s.state == WorkStepState.COMPLETED)

    @property
    def last_confirmed_step(self) -> WorkStep | None:
        completed = self.completed_steps
        return completed[-1] if completed else None

    @property
    def current_step(self) -> WorkStep | None:
        return next((s for s in self.steps if s.state != WorkStepState.COMPLETED), None)

    def _check_time(self, at: datetime) -> None:
        if self.last_event_at is not None and at < self.last_event_at:
            raise ValidationError('work plan event time cannot move backwards')

    def _record(self, at: datetime, action: str, detail: str = '') -> None:
        self._check_time(at)
        self.events.append(PlanEvent(at, action, detail))
        self.last_event_at = at

    def start_step(self, step_id: str, at: datetime) -> None:
        at = ensure_utc(at)
        self._check_time(at)
        if self.state != PlanState.IN_PROGRESS:
            raise StateTransitionError('cannot start work while plan is not in progress')
        target = next((s for s in self.steps if s.step_id == step_id), None)
        if target is None:
            raise ValidationError(f'unknown work step {step_id}')
        current = self.current_step
        if target is not current:
            raise StateTransitionError('steps must be executed in order from the last confirmed step')
        if target.state not in {WorkStepState.PENDING, WorkStepState.PAUSED}:
            raise StateTransitionError('step is not ready to start')
        target.state = WorkStepState.ACTIVE
        self._record(at, 'started', step_id)

    def confirm_step(self, step_id: str, at: datetime) -> None:
        at = ensure_utc(at)
        self._check_time(at)
        if self.state != PlanState.IN_PROGRESS:
            raise StateTransitionError('cannot confirm work while plan is not in progress')
        target = next((s for s in self.steps if s.step_id == step_id), None)
        if target is None or target.state != WorkStepState.ACTIVE:
            raise StateTransitionError('only the active step can be confirmed')
        target.state = WorkStepState.COMPLETED
        target.confirmed_at = at
        self._record(at, 'confirmed', step_id)
        if all(s.state == WorkStepState.COMPLETED for s in self.steps):
            self.state = PlanState.DONE
            self._record(at, 'completed')

    def pause(self, at: datetime, detail: str = '') -> None:
        at = ensure_utc(at)
        self._check_time(at)
        if self.state == PlanState.DONE:
            raise StateTransitionError('completed work cannot be paused')
        current = self.current_step
        if current is not None and current.state == WorkStepState.ACTIVE:
            current.state = WorkStepState.PAUSED
        self.state = PlanState.SUSPENDED
        self._record(at, 'suspended', detail)

    def evacuate(self, at: datetime, detail: str = '') -> None:
        at = ensure_utc(at)
        self._check_time(at)
        if self.state == PlanState.DONE:
            raise StateTransitionError('completed work cannot be evacuated')
        current = self.current_step
        if current is not None and current.state == WorkStepState.ACTIVE:
            current.state = WorkStepState.PAUSED
        self.state = PlanState.EVACUATED
        self._record(at, 'evacuated', detail)

    def resume(self, at: datetime) -> WorkStep:
        at = ensure_utc(at)
        self._check_time(at)
        if self.state not in {PlanState.SUSPENDED, PlanState.EVACUATED}:
            raise StateTransitionError('only a suspended or evacuated plan can resume')
        current = self.current_step
        if current is None:
            raise StateTransitionError('plan has no remaining work')
        # Resume exactly at the step following the last confirmed step; that
        # step is parked (or never started) and is activated again as-is.
        current.state = WorkStepState.ACTIVE
        self.state = PlanState.IN_PROGRESS
        self._record(at, 'resumed', current.step_id)
        return current


class SafetyWatch:
    """Continuous safety watch for a vessel while it is alongside.

    Every evaluation judges all four hazard sources. Missing, stale or
    out-of-range input is a data gap and fails closed: the watch never treats
    an unobserved condition as safe.
    """

    def __init__(
        self,
        weather_limits: MarineWeatherLimits,
        tide_requirement: UnderKeelRequirement,
        mooring_requirement: MooringRequirement,
        *,
        storm_wind_kn: float | None = None,
        storm_gust_kn: float | None = None,
        max_observation_age: timedelta | None = None,
        data_gap_verdict: WatchVerdict = WatchVerdict.SUSPEND,
    ):
        if storm_wind_kn is not None and storm_wind_kn < 0:
            raise ValidationError('storm wind threshold cannot be negative')
        if storm_gust_kn is not None and storm_gust_kn < 0:
            raise ValidationError('storm gust threshold cannot be negative')
        if max_observation_age is not None and max_observation_age.total_seconds() <= 0:
            raise ValidationError('observation age limit must be positive')
        if data_gap_verdict not in {WatchVerdict.SUSPEND, WatchVerdict.EVACUATE}:
            raise ValidationError('data gaps must resolve to suspend or evacuate')
        self.weather_limits = weather_limits
        self.tide_requirement = tide_requirement
        self.mooring_requirement = mooring_requirement
        self.storm_wind_kn = storm_wind_kn
        self.storm_gust_kn = storm_gust_kn
        self.max_observation_age = max_observation_age
        self.data_gap_verdict = data_gap_verdict
        self._weather_checker = WeatherWindowChecker(weather_limits)

    def _gap(self, source: str, reason: str) -> HazardAssessment:
        return HazardAssessment(source, self.data_gap_verdict, (reason,), gap=True)

    def _is_stale(self, observed_at: datetime, at: datetime) -> bool:
        if self.max_observation_age is None:
            return False
        return at - ensure_utc(observed_at) > self.max_observation_age

    def assess_tide(self, at: datetime, tide_curve: TideCurve | None) -> HazardAssessment:
        if tide_curve is None:
            return self._gap(TIDE_SOURCE, 'no tide curve available')
        try:
            clearance = tide_curve.clearance_m(at, self.tide_requirement)
        except SafetyConstraintError:
            return self._gap(TIDE_SOURCE, 'tide curve does not cover evaluation time')
        if clearance >= self.tide_requirement.minimum_clearance_m:
            return HazardAssessment(TIDE_SOURCE, WatchVerdict.CONTINUE)
        if clearance >= 0.0:
            return HazardAssessment(
                TIDE_SOURCE, WatchVerdict.SUSPEND,
                (f'under-keel clearance {clearance:.2f}m below minimum {self.tide_requirement.minimum_clearance_m:.2f}m',),
            )
        return HazardAssessment(
            TIDE_SOURCE, WatchVerdict.EVACUATE,
            (f'negative under-keel clearance {clearance:.2f}m',),
        )

    def assess_weather(self, at: datetime, sample: WeatherSample | None) -> HazardAssessment:
        if sample is None:
            return self._gap(WEATHER_SOURCE, 'no weather observation available')
        if self._is_stale(sample.at, at):
            return self._gap(WEATHER_SOURCE, 'weather observation is stale')
        reasons: list[str] = []
        storm = False
        if self.storm_wind_kn is not None and sample.wind_kn > self.storm_wind_kn:
            storm = True
            reasons.append(f'sustained wind {sample.wind_kn:.0f}kn over storm threshold')
        if self.storm_gust_kn is not None and sample.gust_kn > self.storm_gust_kn:
            storm = True
            reasons.append(f'gust {sample.gust_kn:.0f}kn over storm threshold')
        if storm:
            return HazardAssessment(WEATHER_SOURCE, WatchVerdict.EVACUATE, tuple(reasons))
        reasons.extend(self._weather_checker.reasons(sample))
        if reasons:
            return HazardAssessment(WEATHER_SOURCE, WatchVerdict.SUSPEND, tuple(reasons))
        return HazardAssessment(WEATHER_SOURCE, WatchVerdict.CONTINUE)

    def assess_mooring(self, at: datetime, survey: MooringSurvey | None) -> HazardAssessment:
        if survey is None:
            return self._gap(MOORING_SOURCE, 'no mooring survey available')
        if self._is_stale(survey.at, at):
            return self._gap(MOORING_SOURCE, 'mooring survey is stale')
        lines = survey.lines
        req = self.mooring_requirement
        if len(lines) < req.minimum_lines:
            return HazardAssessment(
                MOORING_SOURCE, WatchVerdict.EVACUATE,
                (f'only {len(lines)} of {req.minimum_lines} mooring lines rigged',),
            )
        total = sum(line.effective_breaking_load_t for line in lines)
        if total < req.minimum_total_effective_load_t:
            return HazardAssessment(
                MOORING_SOURCE, WatchVerdict.EVACUATE,
                (f'effective mooring strength {total:.1f}t below {req.minimum_total_effective_load_t:.1f}t',),
            )
        overloaded = sorted(
            line.line_id for line in lines
            if line.pretension_t / line.effective_breaking_load_t > req.maximum_utilization
        )
        if overloaded:
            return HazardAssessment(
                MOORING_SOURCE, WatchVerdict.SUSPEND,
                ('mooring line utilization exceeded: ' + ','.join(overloaded),),
            )
        return HazardAssessment(MOORING_SOURCE, WatchVerdict.CONTINUE)

    def assess_berth_closure(self, at: datetime, closures: list[BerthClosure] | None) -> HazardAssessment:
        if closures is None:
            return self._gap(BERTH_CLOSURE_SOURCE, 'berth closure status unavailable')
        active = [c for c in closures if c.interval.contains(at)]
        if not active:
            return HazardAssessment(BERTH_CLOSURE_SOURCE, WatchVerdict.CONTINUE)
        emergency = any(c.emergency for c in active)
        reasons = tuple(f'{c.berth_id}: {c.reason}' for c in active)
        if emergency:
            return HazardAssessment(BERTH_CLOSURE_SOURCE, WatchVerdict.EVACUATE, reasons)
        return HazardAssessment(BERTH_CLOSURE_SOURCE, WatchVerdict.SUSPEND, reasons)

    def assess(
        self,
        at: datetime,
        *,
        weather: WeatherSample | None = None,
        tide_curve: TideCurve | None = None,
        mooring: MooringSurvey | None = None,
        closures: list[BerthClosure] | None = None,
        berth_id: str | None = None,
    ) -> WatchDecision:
        at = ensure_utc(at)
        if closures is not None and berth_id is not None:
            closures = [c for c in closures if c.berth_id == berth_id]
        assessments = (
            self.assess_tide(at, tide_curve),
            self.assess_weather(at, weather),
            self.assess_mooring(at, mooring),
            self.assess_berth_closure(at, closures),
        )
        verdict = max((a.verdict for a in assessments), key=lambda v: _SEVERITY[v])
        return WatchDecision(at, verdict, assessments)

    def apply(self, plan: BerthWorkPlan, decision: WatchDecision) -> WatchDecision:
        """Apply a watch decision to an executing vessel operation.

        Completed work is never touched. A suspended plan automatically
        resumes once conditions return to CONTINUE, restarting from the step
        after the last confirmed one. An evacuated plan requires an explicit
        re-berth confirmation before it can resume.
        """
        if plan.state == PlanState.DONE:
            return replace(decision, applied=False, resumed=False)
        resumed = False
        if decision.verdict == WatchVerdict.EVACUATE:
            if plan.state != PlanState.EVACUATED:
                plan.evacuate(decision.at, '; '.join(decision.reasons))
        elif decision.verdict == WatchVerdict.SUSPEND:
            if plan.state == PlanState.IN_PROGRESS:
                plan.pause(decision.at, '; '.join(decision.reasons))
            # Suspended stays suspended; evacuated stays evacuated (no silent
            # de-escalation from an evacuation).
        elif plan.state == PlanState.SUSPENDED:
            plan.resume(decision.at)
            resumed = True
        return replace(decision, applied=True, resumed=resumed)

    def resume(
        self,
        plan: BerthWorkPlan,
        decision: WatchDecision,
        *,
        reberthed: bool = False,
    ) -> WorkStep:
        """Explicitly resume after an evacuation (or suspension).

        Only a CONTINUE decision is accepted, so missing or stale feeds keep
        the operation parked until the data gap is closed.
        """
        if decision.verdict != WatchVerdict.CONTINUE:
            raise SafetyConstraintError('cannot resume while safety conditions are not clear')
        if plan.state == PlanState.EVACUATED and not reberthed:
            raise SafetyConstraintError('evacuated vessel requires re-berth confirmation before resume')
        if plan.state not in {PlanState.SUSPENDED, PlanState.EVACUATED}:
            raise StateTransitionError('plan is not paused')
        return plan.resume(decision.at)
