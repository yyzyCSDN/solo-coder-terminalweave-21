from datetime import datetime, timedelta, timezone

import pytest

from terminalweave.core.errors import NotFoundError, StateTransitionError, ValidationError
from terminalweave.core.intervals import TimeInterval
from terminalweave.marine.berth_windows import BerthClosure
from terminalweave.marine.mooring import MooringLine, MooringRequirement
from terminalweave.marine.safety_watch import (
    BerthSafetyWatch,
    ClosureGuardPolicy,
    GuardAction,
    HazardLevel,
    MooringGuardPolicy,
    OperationPhase,
    Source,
    TideGuardPolicy,
    WeatherGuardPolicy,
)
from terminalweave.marine.tide import TideSample, TideCurve, UnderKeelRequirement
from terminalweave.marine.weather import MarineWeatherLimits, WeatherSample

UTC = timezone.utc
D = datetime(2026, 6, 1, tzinfo=UTC)
H = timedelta(hours=1)
M = timedelta(minutes=1)


def tide_curve():
    # clearance = chart 12 + height - draft 9.5 = 2.5 + height; minimum 1.0
    return TideCurve([
        TideSample(D, 2.0),
        TideSample(D + 2 * H, 2.0),
        TideSample(D + 4 * H, -2.0),
        TideSample(D + 8 * H, -2.0),
        TideSample(D + 10 * H, 2.0),
        TideSample(D + 12 * H, 2.0),
    ])


TIDE = TideGuardPolicy(UnderKeelRequirement(12.0, 9.5, 1.0), caution_buffer_m=0.3)
WORK = MarineWeatherLimits(25.0, 35.0, 1.0, 2.0)
EVAC = MarineWeatherLimits(40.0, 55.0, 0.25, 4.0)
WEATHER = WeatherGuardPolicy(WORK, EVAC, max_staleness=30 * M)
MOORING = MooringGuardPolicy(MooringRequirement(4, 0.5, 400.0), max_staleness=30 * M)
CLOSURES = ClosureGuardPolicy(imminent_lead=30 * M)


def watch(policies=(TIDE, WEATHER, MOORING, CLOSURES), tide=True, closures=None):
    w = BerthSafetyWatch(
        tide_policy=TIDE if TIDE in policies else None,
        weather_policy=WEATHER if WEATHER in policies else None,
        mooring_policy=MOORING if MOORING in policies else None,
        closure_policy=CLOSURES if CLOSURES in policies else None,
    )
    if tide:
        w.update_tide(tide_curve())
    for c in closures or []:
        w.add_closure(c)
    return w


def good_lines():
    return [MooringLine(f'L{i}', 120.0, 50.0, 90.0) for i in range(4)]


def full_watch(call='V1', weather=True, mooring=True, at=D):
    w = watch()
    w.watch(call, 'B1', at)
    if weather:
        w.report_weather(WeatherSample(at, 10.0, 12.0, 5.0, 0.5))
    if mooring:
        w.report_mooring(call, good_lines(), at)
    return w


# --------------------------------------------------------------- nominal

def test_nominal_conditions_continue():
    w = full_watch()
    decision = w.evaluate('V1', D + 1 * M)
    assert decision.action is GuardAction.CONTINUE
    assert decision.phase is OperationPhase.ACTIVE
    assert not decision.changed


# --------------------------------------------------------------- tide

def test_tide_below_clearance_evacuates():
    w = full_watch()
    decision = w.evaluate('V1', D + 4 * H)
    assert decision.action is GuardAction.EVACUATE
    assert decision.phase is OperationPhase.EVACUATED
    tide = next(f for f in decision.findings if f.source is Source.TIDE)
    assert tide.level is HazardLevel.CRITICAL and tide.reasons


def test_missing_tide_data_fails_closed():
    w = watch(tide=False)
    w.watch('V1', 'B1', D)
    decision = w.evaluate('V1', D + 1 * H)
    assert decision.action is GuardAction.PAUSE
    assert decision.findings[0].level is HazardLevel.CAUTION
    assert 'unavailable' in decision.findings[0].reasons[0]


def test_tide_curve_gap_fails_closed():
    w = BerthSafetyWatch(tide_policy=TIDE)
    w.update_tide(TideCurve([TideSample(D, 2.0), TideSample(D + 2 * H, 2.0)]))
    w.watch('V1', 'B1', D)
    assert w.evaluate('V1', D + 3 * H).action is GuardAction.PAUSE


# --------------------------------------------------------------- weather

def test_weather_over_working_limit_pauses():
    w = full_watch()
    w.report_weather(WeatherSample(D + 1 * H, 30.0, 30.0, 5.0, 0.5))
    decision = w.evaluate('V1', D + 1 * H)
    assert decision.action is GuardAction.PAUSE
    assert decision.phase is OperationPhase.PAUSED
    weather = next(f for f in decision.findings if f.source is Source.WEATHER)
    assert weather.level is HazardLevel.CAUTION


def test_weather_over_evacuation_limit_evacuates():
    w = full_watch()
    w.report_weather(WeatherSample(D + 1 * H, 45.0, 45.0, 5.0, 0.5))
    assert w.evaluate('V1', D + 1 * H).action is GuardAction.EVACUATE


def test_stale_weather_pauses_then_fresh_resumes():
    w = full_watch()
    assert w.evaluate('V1', D + 45 * M).action is GuardAction.PAUSE
    w.report_weather(WeatherSample(D + 45 * M, 10.0, 12.0, 5.0, 0.5))
    w.report_mooring('V1', good_lines(), D + 45 * M)
    decision = w.evaluate('V1', D + 46 * M)
    assert decision.action is GuardAction.CONTINUE
    assert decision.phase is OperationPhase.ACTIVE


def test_weather_gap_pauses_without_report():
    w = watch(policies=(TIDE, WEATHER, MOORING, CLOSURES))
    w.watch('V1', 'B1', D)
    w.report_mooring('V1', good_lines(), D)
    assert w.evaluate('V1', D + 1 * H).action is GuardAction.PAUSE


# --------------------------------------------------------------- mooring

def test_overloaded_mooring_pauses():
    w = full_watch()
    lines = [MooringLine(f'L{i}', 120.0, 70.0, 90.0) for i in range(4)]
    w.report_mooring('V1', lines, D + 1 * H)
    decision = w.evaluate('V1', D + 1 * H)
    assert decision.action is GuardAction.PAUSE
    assert 'utilization' in decision.findings[2].reasons[0]


def test_insufficient_mooring_evacuates():
    w = full_watch()
    w.report_mooring('V1', [MooringLine(f'L{i}', 120.0, 50.0, 90.0) for i in range(2)], D + 1 * H)
    assert w.evaluate('V1', D + 1 * H).action is GuardAction.EVACUATE


# --------------------------------------------------------------- closures

def test_active_closure_evacuates():
    closure = BerthClosure('B1', TimeInterval(D + 11 * H, D + 12 * H), 'port drill')
    w = full_watch()
    w.add_closure(closure)
    decision = w.evaluate('V1', D + 11 * H)
    assert decision.action is GuardAction.EVACUATE
    assert 'berth closed' in decision.findings[-1].reasons[0]


def test_imminent_closure_pauses():
    closure = BerthClosure('B1', TimeInterval(D + 1 * H + 20 * M, D + 2 * H), 'port drill')
    w = full_watch()
    w.add_closure(closure)
    decision = w.evaluate('V1', D + 1 * H)
    assert decision.action is GuardAction.PAUSE
    assert 'imminent' in decision.findings[-1].reasons[0]


# --------------------------------------------------------------- checkpoint

def test_resume_continues_from_last_confirmed_checkpoint():
    w = full_watch()
    w.confirm_progress('V1', D + 1 * H, units=120.0)
    w.report_weather(WeatherSample(D + 2 * H, 30.0, 30.0, 5.0, 0.5))
    paused = w.evaluate('V1', D + 2 * H)
    assert paused.phase is OperationPhase.PAUSED
    with pytest.raises(StateTransitionError):
        w.confirm_progress('V1', D + 2 * H + 5 * M, units=5.0)

    w.report_weather(WeatherSample(D + 3 * H, 10.0, 12.0, 5.0, 0.5))
    w.report_mooring('V1', good_lines(), D + 3 * H)
    decision = w.evaluate('V1', D + 3 * H)
    op = w.get('V1')
    assert decision.action is GuardAction.CONTINUE
    assert decision.resumed_from == D + 1 * H
    assert op.completed_units == 120.0
    assert op.last_confirmed_at == D + 1 * H
    kinds = [e.kind for e in op.events]
    assert kinds == ['watch_started', 'progress_confirmed', 'paused', 'resumed']


def test_worst_finding_drives_decision():
    w = full_watch()
    w.report_weather(WeatherSample(D + 1 * H, 45.0, 45.0, 5.0, 0.5))
    w.report_mooring('V1', good_lines(), D + 1 * H)
    decision = w.evaluate('V1', D + 1 * H)
    assert decision.action is GuardAction.EVACUATE
    assert {f.level for f in decision.findings if f.level is not HazardLevel.NOMINAL} == {
        HazardLevel.CRITICAL
    }


# --------------------------------------------------------------- evacuation flow

def test_evacuated_vessel_must_return_before_resume():
    w = full_watch()
    w.evaluate('V1', D + 4 * H)  # tide critical -> evacuated
    w.report_weather(WeatherSample(D + 10 * H, 10.0, 12.0, 5.0, 0.5))
    w.report_mooring('V1', good_lines(), D + 10 * H)
    decision = w.evaluate('V1', D + 10 * H + 1 * M)  # conditions nominal again
    assert decision.action is GuardAction.CONTINUE
    assert decision.phase is OperationPhase.EVACUATED
    assert decision.notes

    with pytest.raises(ValidationError):
        w.mark_returned('V1', D + 2 * H)  # cannot move time backwards
    w.mark_returned('V1', D + 10 * H + 2 * M)
    resumed = w.evaluate('V1', D + 10 * H + 3 * M)
    assert resumed.phase is OperationPhase.ACTIVE
    assert resumed.resumed_from == w.get('V1').last_confirmed_at


# --------------------------------------------------------------- completion

def test_completed_operation_is_not_rolled_back():
    w = full_watch()
    w.confirm_progress('V1', D + 1 * H, units=200.0)
    w.complete('V1', D + 1 * H + 10 * M)
    w.add_closure(BerthClosure('B1', TimeInterval(D + 3 * H, D + 5 * H), 'drill'))
    decision = w.evaluate('V1', D + 4 * H)
    op = w.get('V1')
    assert decision.action is GuardAction.CONTINUE
    assert decision.findings == ()
    assert op.phase is OperationPhase.COMPLETED
    assert op.completed_units == 200.0
    with pytest.raises(StateTransitionError):
        w.confirm_progress('V1', D + 4 * H + 1 * M, units=1.0)


def test_evaluate_all_skips_completed_operations():
    w = full_watch(call='V1')
    w.watch('V2', 'B1', D)
    w.report_weather(WeatherSample(D, 10.0, 12.0, 5.0, 0.5))
    w.report_mooring('V2', good_lines(), D)
    w.complete('V1', D + 30 * M)
    decisions = w.evaluate_all(D + 1 * H)
    assert [d.call_id for d in decisions] == ['V2']


# --------------------------------------------------------------- validation

def test_watch_requires_at_least_one_live_source():
    with pytest.raises(ValidationError):
        BerthSafetyWatch(closure_policy=CLOSURES)


def test_evacuation_limits_must_be_looser_than_working_limits():
    with pytest.raises(ValidationError):
        WeatherGuardPolicy(WORK, MarineWeatherLimits(10.0, 35.0, 1.0, 2.0))


def test_duplicate_watch_and_unknown_call_are_rejected():
    w = full_watch()
    with pytest.raises(Exception):
        w.watch('V1', 'B1', D)
    with pytest.raises(NotFoundError):
        w.evaluate('NOPE', D + 1 * H)
