from datetime import datetime, timedelta, timezone
import pytest

from terminalweave.marine.tide import TideSample, TideCurve, UnderKeelRequirement
from terminalweave.marine.weather import WeatherSample, MarineWeatherLimits
from terminalweave.marine.mooring import MooringLine, MooringRequirement
from terminalweave.marine.berth_windows import BerthClosure
from terminalweave.marine.safety_watch import (
    SafetyWatch, MooringSurvey, WatchVerdict, PlanState, WorkStepState,
    BerthWorkPlan, WorkStep,
    TIDE_SOURCE, MOORING_SOURCE, BERTH_CLOSURE_SOURCE, WEATHER_SOURCE,
)
from terminalweave.core.errors import SafetyConstraintError, StateTransitionError
from terminalweave.core.intervals import TimeInterval

UTC = timezone.utc
D = datetime(2026, 1, 1, tzinfo=UTC)

LIMITS = MarineWeatherLimits(25, 35, 1.0, 2.0)
TIDE_REQ = UnderKeelRequirement(12.0, 10.0, 1.0)
MOOR_REQ = MooringRequirement(4, 0.5, 200.0)


def make_watch(**kw):
    defaults = dict(storm_wind_kn=40.0, max_observation_age=timedelta(minutes=30))
    defaults.update(kw)
    return SafetyWatch(LIMITS, TIDE_REQ, MOOR_REQ, **defaults)


def make_curve():
    # heights at +4h give UKC 0.5m (suspend), +5h give -0.5m (evacuate)
    return TideCurve([
        TideSample(D, 0.5),
        TideSample(D + timedelta(hours=4), -1.5),
        TideSample(D + timedelta(hours=5), -2.5),
        TideSample(D + timedelta(hours=6), 0.0),
    ])


def good_weather(at=D):
    return WeatherSample(at, 15, 20, 5, 1.0, False)


def good_lines(pretension=20.0):
    return [MooringLine(f'L{i}', 100.0, pretension, 0.0) for i in range(4)]


def good_mooring(at=D):
    return MooringSurvey(at, good_lines())


def make_plan():
    plan = BerthWorkPlan('V1', 'B1', [WorkStep('s1'), WorkStep('s2'), WorkStep('s3')])
    plan.start_step('s1', D)
    plan.confirm_step('s1', D + timedelta(minutes=10))
    plan.start_step('s2', D + timedelta(minutes=20))
    return plan


# ---------- source-level assessments ----------

def test_all_clear_decision_is_continue():
    watch = make_watch()
    decision = watch.assess(D, weather=good_weather(), tide_curve=make_curve(),
                            mooring=good_mooring(), closures=[])
    assert decision.verdict == WatchVerdict.CONTINUE
    assert decision.gaps == ()


def test_tide_below_margin_suspends_negative_ukc_evacuates():
    watch = make_watch()
    d_suspend = watch.assess(D + timedelta(hours=4), weather=good_weather(),
                             tide_curve=make_curve(), mooring=good_mooring(), closures=[])
    assert d_suspend.verdict == WatchVerdict.SUSPEND
    assert d_suspend.for_source(TIDE_SOURCE).reasons[0].startswith('under-keel clearance')
    d_evac = watch.assess(D + timedelta(hours=5), weather=good_weather(),
                          tide_curve=make_curve(), mooring=good_mooring(), closures=[])
    assert d_evac.verdict == WatchVerdict.EVACUATE


def test_weather_wind_suspends_storm_evacuates():
    watch = make_watch()
    windy = watch.assess(D, weather=WeatherSample(D, 30, 30, 5, 1),
                         tide_curve=make_curve(), mooring=good_mooring(), closures=[])
    assert windy.verdict == WatchVerdict.SUSPEND
    storm = watch.assess(D, weather=WeatherSample(D, 45, 45, 5, 1),
                         tide_curve=make_curve(), mooring=good_mooring(), closures=[])
    assert storm.verdict == WatchVerdict.EVACUATE
    lightning = watch.assess(D, weather=WeatherSample(D, 15, 20, 5, 1, True),
                             tide_curve=make_curve(), mooring=good_mooring(), closures=[])
    assert lightning.verdict == WatchVerdict.SUSPEND


def test_mooring_failure_modes():
    watch = make_watch()
    few = watch.assess(D, weather=good_weather(), tide_curve=make_curve(),
                       mooring=MooringSurvey(D, good_lines()[:2]), closures=[])
    assert few.verdict == WatchVerdict.EVACUATE
    weak = [MooringLine(f'L{i}', 40.0, 5.0, 0.0) for i in range(4)]
    d_weak = watch.assess(D, weather=good_weather(), tide_curve=make_curve(),
                          mooring=MooringSurvey(D, weak), closures=[])
    assert d_weak.for_source(MOORING_SOURCE).verdict == WatchVerdict.EVACUATE
    overloaded = good_lines()
    overloaded[0] = MooringLine('L0', 100.0, 60.0, 0.0)
    d_over = watch.assess(D, weather=good_weather(), tide_curve=make_curve(),
                          mooring=MooringSurvey(D, overloaded), closures=[])
    assert d_over.for_source(MOORING_SOURCE).verdict == WatchVerdict.SUSPEND


def test_berth_closure_routine_suspends_emergency_evacuates():
    watch = make_watch()
    interval = TimeInterval(D - timedelta(minutes=5), D + timedelta(hours=1))
    routine = watch.assess(D, weather=good_weather(), tide_curve=make_curve(),
                           mooring=good_mooring(),
                           closures=[BerthClosure('B1', interval, 'port drill')])
    assert routine.verdict == WatchVerdict.SUSPEND
    emergency = watch.assess(D, weather=good_weather(), tide_curve=make_curve(),
                             mooring=good_mooring(),
                             closures=[BerthClosure('B1', interval, 'fire', emergency=True)])
    assert emergency.verdict == WatchVerdict.EVACUATE


# ---------- data gaps fail closed ----------

def test_missing_feed_is_a_data_gap_and_suspends():
    watch = make_watch()
    decision = watch.assess(D, weather=good_weather(), tide_curve=None,
                            mooring=good_mooring(), closures=[])
    assert decision.verdict == WatchVerdict.SUSPEND
    assert decision.gaps == (TIDE_SOURCE,)
    assert decision.for_source(TIDE_SOURCE).gap


def test_stale_observation_is_a_data_gap():
    watch = make_watch()
    decision = watch.assess(D + timedelta(hours=1), weather=good_weather(D),
                            tide_curve=make_curve(), mooring=good_mooring(), closures=[])
    assert decision.verdict == WatchVerdict.SUSPEND
    assert WEATHER_SOURCE in decision.gaps
    assert MOORING_SOURCE in decision.gaps
    # unannounced closure status is also a gap
    no_closure = watch.assess(D, weather=good_weather(), tide_curve=make_curve(),
                              mooring=good_mooring(), closures=None)
    assert BERTH_CLOSURE_SOURCE in no_closure.gaps


def test_data_gap_can_be_configured_to_evacuate():
    watch = make_watch(data_gap_verdict=WatchVerdict.EVACUATE)
    decision = watch.assess(D, weather=good_weather(), tide_curve=make_curve(),
                            mooring=None, closures=[])
    assert decision.verdict == WatchVerdict.EVACUATE
    assert decision.for_source(MOORING_SOURCE).gap


def test_data_gap_verdict_cannot_be_continue():
    with pytest.raises(Exception):
        make_watch(data_gap_verdict=WatchVerdict.CONTINUE)


def test_tide_curve_outside_coverage_is_a_gap():
    watch = make_watch()
    decision = watch.assess(D + timedelta(hours=12), weather=good_weather(D + timedelta(hours=12)),
                            tide_curve=make_curve(),
                            mooring=good_mooring(D + timedelta(hours=12)), closures=[])
    assert TIDE_SOURCE in decision.gaps
    assert decision.verdict == WatchVerdict.SUSPEND


def test_berth_closure_filtered_by_watched_berth():
    watch = make_watch()
    interval = TimeInterval(D - timedelta(minutes=5), D + timedelta(hours=1))
    decision = watch.assess(
        D, weather=good_weather(), tide_curve=make_curve(), mooring=good_mooring(),
        closures=[BerthClosure('OTHER', interval, 'fire', emergency=True)], berth_id='B1',
    )
    assert decision.verdict == WatchVerdict.CONTINUE
    filtered = watch.assess(
        D, weather=good_weather(), tide_curve=make_curve(), mooring=good_mooring(),
        closures=[BerthClosure('B1', interval, 'fire', emergency=True)], berth_id='B1',
    )
    assert filtered.verdict == WatchVerdict.EVACUATE


def test_highest_severity_wins():
    watch = make_watch()
    # storm wind (evacuate) together with routine closure (suspend)
    decision = watch.assess(
        D, weather=WeatherSample(D, 45, 45, 5, 1), tide_curve=make_curve(),
        mooring=good_mooring(),
        closures=[BerthClosure('B1', TimeInterval(D - timedelta(minutes=5), D + timedelta(hours=1)), 'drill')],
    )
    assert decision.verdict == WatchVerdict.EVACUATE


# ---------- applying decisions to an executing plan ----------

def test_continue_does_not_disturb_in_progress_plan():
    watch = make_watch()
    plan = make_plan()
    at = D + timedelta(minutes=25)
    decision = watch.assess(at, weather=good_weather(at), tide_curve=make_curve(),
                            mooring=good_mooring(at), closures=[])
    applied = watch.apply(plan, decision)
    assert plan.state == PlanState.IN_PROGRESS
    assert applied.applied and not applied.resumed


def test_suspend_parks_active_step_without_rolling_back_completed():
    watch = make_watch()
    plan = make_plan()
    at = D + timedelta(minutes=25)
    decision = watch.assess(at, weather=good_weather(at),
                            tide_curve=make_curve(), mooring=good_mooring(at),
                            closures=[BerthClosure('B1', TimeInterval(at - timedelta(minutes=5), at + timedelta(hours=1)), 'drill')])
    watch.apply(plan, decision)
    assert plan.state == PlanState.SUSPENDED
    assert plan.steps[0].state == WorkStepState.COMPLETED
    assert plan.steps[1].state == WorkStepState.PAUSED
    assert plan.steps[2].state == WorkStepState.PENDING
    assert plan.events[-1].action == 'suspended'
    # repeated suspend is idempotent
    watch.apply(plan, decision)
    assert plan.state == PlanState.SUSPENDED


def test_resume_after_recovery_continues_from_last_confirmed_step():
    watch = make_watch()
    plan = make_plan()
    bad_at = D + timedelta(hours=4, minutes=5)
    bad = watch.assess(bad_at, weather=good_weather(bad_at),
                       tide_curve=make_curve(), mooring=good_mooring(bad_at), closures=[])
    watch.apply(plan, bad)
    assert plan.state == PlanState.SUSPENDED
    # feed in a gap a few minutes later: plan stays parked
    gap_at = bad_at + timedelta(minutes=5)
    gap = watch.assess(gap_at, weather=None,
                       tide_curve=make_curve(), mooring=good_mooring(gap_at), closures=[])
    watch.apply(plan, gap)
    assert plan.state == PlanState.SUSPENDED
    # gap closed and tide back above minimum at +6h: auto resume continues s2
    clear_at = D + timedelta(hours=6)
    clear = watch.assess(clear_at, weather=good_weather(clear_at), tide_curve=make_curve(),
                         mooring=good_mooring(clear_at), closures=[])
    assert clear.verdict == WatchVerdict.CONTINUE
    applied = watch.apply(plan, clear)
    assert plan.state == PlanState.IN_PROGRESS
    assert applied.resumed
    assert plan.steps[0].state == WorkStepState.COMPLETED
    assert plan.steps[1].state == WorkStepState.ACTIVE
    assert plan.steps[2].state == WorkStepState.PENDING
    assert plan.current_step.step_id == 's2'
    assert plan.events[-1].action == 'resumed'
    assert plan.events[-1].detail == 's2'


def test_evacuation_keeps_progress_and_requires_reberth_to_resume():
    watch = make_watch()
    plan = make_plan()
    storm_at = D + timedelta(minutes=25)
    evac = watch.assess(storm_at, weather=WeatherSample(storm_at, 45, 45, 5, 1),
                        tide_curve=make_curve(), mooring=good_mooring(storm_at), closures=[])
    watch.apply(plan, evac)
    assert plan.state == PlanState.EVACUATED
    assert plan.steps[0].state == WorkStepState.COMPLETED
    assert plan.steps[1].state == WorkStepState.PAUSED
    # clear conditions alone do NOT silently un-evacuate through apply()
    check_at = storm_at + timedelta(minutes=5)
    clear = watch.assess(check_at, weather=good_weather(check_at),
                         tide_curve=make_curve(), mooring=good_mooring(check_at), closures=[])
    watch.apply(plan, clear)
    assert plan.state == PlanState.EVACUATED
    # explicit resume without re-berth confirmation is refused
    with pytest.raises(SafetyConstraintError):
        watch.resume(plan, clear)
    step = watch.resume(plan, clear, reberthed=True)
    assert step.step_id == 's2'
    assert plan.state == PlanState.IN_PROGRESS
    assert step.state == WorkStepState.ACTIVE
    assert plan.steps[0].state == WorkStepState.COMPLETED


def test_resume_refused_while_conditions_remain_unclear():
    watch = make_watch()
    plan = make_plan()
    plan.pause(D + timedelta(minutes=30), 'drill')
    gap = watch.assess(D + timedelta(minutes=35), weather=None,
                       tide_curve=make_curve(), mooring=good_mooring(), closures=[])
    with pytest.raises(SafetyConstraintError):
        watch.resume(plan, gap)


def test_completed_plan_is_not_rolled_back_or_reopened():
    watch = make_watch()
    plan = make_plan()
    plan.confirm_step('s2', D + timedelta(minutes=30))
    plan.start_step('s3', D + timedelta(minutes=40))
    plan.confirm_step('s3', D + timedelta(minutes=50))
    assert plan.state == PlanState.DONE
    evac_at = D + timedelta(hours=2)
    evac = watch.assess(evac_at, weather=WeatherSample(evac_at, 45, 45, 5, 1),
                        tide_curve=make_curve(), mooring=good_mooring(evac_at), closures=[])
    applied = watch.apply(plan, evac)
    assert plan.state == PlanState.DONE
    assert not applied.applied
    assert all(s.state == WorkStepState.COMPLETED for s in plan.steps)
    with pytest.raises(StateTransitionError):
        plan.pause(evac_at + timedelta(minutes=5))


def test_steps_must_follow_last_confirmed_order():
    plan = make_plan()
    with pytest.raises(StateTransitionError):
        plan.start_step('s3', D + timedelta(minutes=25))


def test_plan_events_cannot_move_backwards():
    plan = make_plan()
    with pytest.raises(Exception):
        plan.confirm_step('s2', D)
