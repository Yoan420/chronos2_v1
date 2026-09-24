"""Causal one-sided interval calibration; tiny deterministic synthetic histories."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from nyx_stress_guard import intervals as module


def make_history(start="2026-01-01", days=45, zones=("DE",), active=True, lower=90., upper=110., actual=120., hours=(0, 1, 2, 3)):
    rows = []
    for day in pd.date_range(start, periods=days, freq="D"):
        for zone in zones:
            for hour in hours:
                timestamp = (day+pd.Timedelta(hours=hour)).tz_localize("Europe/Paris").tz_convert("UTC")
                origin = (day-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
                rows.append({"zone": zone, "timestamp_utc": timestamp, "forecast_origin_utc": origin,
                    "label_available_at_utc": origin+pd.Timedelta(hours=5), "actual": actual,
                    "candidate_forecast": 100., "candidate_q10": lower, "candidate_q90": upper,
                    "intervention_active": active})
    return pd.DataFrame(rows)


def cutoff(day):
    return (pd.Timestamp(day)-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")


def future(day="2026-03-01", zone="DE", active=True):
    return make_history(day, days=1, zones=(zone,), active=active).drop(columns=["actual", "label_available_at_utc"])


def test_corrected_order_statistic_and_unavailable_rank():
    assert module.finite_sample_quantile(np.arange(10)) == (9., 10)
    assert module.finite_sample_quantile(np.arange(9)) == (8., 9)
    assert module.finite_sample_quantile(np.arange(8)) == (None, 9)
    assert module.finite_sample_quantile(np.ones(40)) == (1., 37)


def test_local_active_calibration_is_asymmetric_expansion_only_and_p50_exact():
    history = make_history(days=10)
    state = module.fit_interval_state(history, cutoff("2026-03-01"))
    group = state["groups"]["DE"]["true"]
    assert group["source"] == "country_same_intervention_state"
    assert group["rows"] == 40 and group["days"] == 10
    assert group["lower_expansion_eur_mwh"] == 0
    assert group["upper_expansion_eur_mwh"] == 10
    current = future()
    result = module.apply_interval_state(state, current)
    assert result.candidate_q10.eq(90).all() and result.candidate_q90.eq(120).all()
    pd.testing.assert_series_equal(result.candidate_forecast, current.candidate_forecast)
    json.dumps(state, allow_nan=False)


def test_local_inactive_uses_higher_sample_threshold():
    state = module.fit_interval_state(make_history(days=10, active=False), cutoff("2026-03-01"))
    group = state["groups"]["DE"]["false"]
    assert group["source"] == "none" and group["status"] == "insufficient_history_same_state"
    result = module.apply_interval_state(state, future(active=False))
    assert result.candidate_q90.eq(110).all()


def test_pooled_fallback_is_same_state_and_transparently_labelled():
    history = make_history(days=30, zones=("DE", "BE"), hours=(0, 1), active=False)
    state = module.fit_interval_state(history, cutoff("2026-03-01"))
    for zone in ("DE", "BE"):
        group = state["groups"][zone]["false"]
        assert group["source"] == "pooled_same_intervention_state"
        assert group["local_rows"] == 60 and group["rows"] == 120
        assert group["source_zones"] == ["BE", "DE"]
        assert state["groups"][zone]["true"]["status"] == "insufficient_history_same_state"


def test_no_hidden_active_inactive_pool_when_one_group_is_large():
    history = pd.concat([make_history(days=40, zones=("DE",), active=False),
        make_history(days=2, zones=("BE",), active=True)], ignore_index=True)
    state = module.fit_interval_state(history, cutoff("2026-03-01"))
    group = state["groups"]["BE"]["true"]
    assert group["source"] == "none" and group["rows"] == 8
    assert group["upper_expansion_eur_mwh"] == 0
    assert not state["active_inactive_rows_ever_pooled_together"]


def test_distinct_days_cannot_be_replaced_by_many_hours_on_one_day():
    history = make_history(days=5, hours=tuple(range(24)))
    group = module.fit_interval_state(history, cutoff("2026-03-01"))["groups"]["DE"]["true"]
    assert group["local_rows"] == 120 and group["local_days"] == 5
    assert group["source"] == "none"


def test_late_label_excluded_even_for_old_delivery_day():
    history = make_history(days=11)
    history.loc[:4, "label_available_at_utc"] = pd.Timestamp("2026-03-02", tz="UTC")
    state = module.fit_interval_state(history, cutoff("2026-03-01"))
    group = state["groups"]["DE"]["true"]
    assert group["rows"] == 39 and group["source"] == "none"
    assert pd.Timestamp(state["max_eligible_label_available_at_utc"]) <= cutoff("2026-03-01")


def test_label_available_exactly_at_cutoff_is_allowed():
    history = make_history(days=10)
    history["label_available_at_utc"] = cutoff("2026-03-01")
    state = module.fit_interval_state(history, cutoff("2026-03-01"))
    assert state["eligible_history_rows"] == 40


def test_window_is_365_delivery_days_and_future_rows_are_excluded():
    history = pd.concat([make_history("2025-02-28", days=1), make_history("2025-03-01", days=10),
        make_history("2026-03-01", days=1)], ignore_index=True)
    state = module.fit_interval_state(history, cutoff("2026-03-01"))
    assert state["window_start_day"] == "2025-03-01"
    assert state["eligible_history_rows"] == 40
    assert state["groups"]["DE"]["true"]["first_delivery_day"] == "2025-03-01"


def test_already_calibrated_history_uses_original_bounds_and_apply_is_idempotent():
    history = make_history(days=10)
    state = module.fit_interval_state(history, cutoff("2026-03-01"))
    current = future()
    first = module.apply_interval_state(state, current)
    second = module.apply_interval_state(state, first)
    pd.testing.assert_frame_equal(first, second)
    historical = history.copy()
    historical["precalibration_q10"], historical["precalibration_q90"] = 90., 110.
    historical["candidate_q10"], historical["candidate_q90"] = -1000., 1000.
    historical["interval_calibration_status"] = "calibrated"
    fitted = module.fit_interval_state(historical, cutoff("2026-03-01"))
    assert fitted["groups"]["DE"]["true"]["upper_expansion_eur_mwh"] == 10


def test_ambiguous_or_half_preserved_source_bounds_are_rejected():
    history = make_history(days=10)
    history["interval_calibration_status"] = "calibrated"
    with pytest.raises(module.IntervalCalibrationError, match="original"):
        module.fit_interval_state(history, cutoff("2026-03-01"))
    history["precalibration_q10"] = 90.
    with pytest.raises(module.IntervalCalibrationError, match="together"):
        module.fit_interval_state(history, cutoff("2026-03-01"))


def test_chronological_replay_is_future_label_invariant_and_preserves_index():
    history = make_history(days=15).sample(frac=1, random_state=3)
    result, audit = module.calibrate_intervals(history)
    assert result.index.equals(history.index)
    pd.testing.assert_series_equal(result.candidate_forecast, history.candidate_forecast)
    assert (result.candidate_q10 <= history.candidate_q10).all()
    assert (result.candidate_q90 >= history.candidate_q90).all()
    changed = history.copy()
    boundary = cutoff("2026-01-13")
    later = changed.forecast_origin_utc.ge(boundary)
    changed.loc[later, "actual"] = 10000.
    replay, _ = module.calibrate_intervals(changed)
    pd.testing.assert_frame_equal(result.loc[~later, ["candidate_q10", "candidate_q90"]],
        replay.loc[~later, ["candidate_q10", "candidate_q90"]])
    assert audit["daily_fits"] == 15 and not audit["p50_modified"]


def test_fit_and_apply_do_not_read_current_labels():
    state = module.fit_interval_state(make_history(days=10), cutoff("2026-03-01"))
    current = future()
    clean = module.apply_interval_state(state, current)
    current["actual"], current["label_available_at_utc"] = np.inf, "not even a timestamp"
    polluted = module.apply_interval_state(state, current)
    pd.testing.assert_frame_equal(clean[["candidate_q10", "candidate_q90"]], polluted[["candidate_q10", "candidate_q90"]])


def test_flags_exclude_ineligible_history_rows():
    history = make_history(days=11)
    history["label_eligible"] = True
    history.loc[:4, "label_eligible"] = False
    group = module.fit_interval_state(history, cutoff("2026-03-01"))["groups"]["DE"]["true"]
    assert group["rows"] == 39 and group["source"] == "none"


def test_drift_diagnostics_are_descriptive_and_do_not_change_expansion_recipe():
    history = make_history(days=60, actual=100.)
    recent = history.timestamp_utc.ge(pd.Timestamp("2026-02-01", tz="UTC"))
    history.loc[recent, "actual"] = 140.
    state = module.fit_interval_state(history, cutoff("2026-03-02"))
    record = state["groups"]["DE"]["true"]
    assert record["drift_status"] == "descriptive_recent_vs_older"
    assert record["upper_miss_rate_shift_flag"]
    assert not record["lower_miss_rate_shift_flag"]
    assert record["upper_expansion_eur_mwh"] == 30


@pytest.mark.parametrize("day", ["2026-03-29", "2026-03-30", "2026-10-25", "2026-10-26"])
def test_dst_delivery_days_have_exact_civil_cutoff_and_physical_hours(day):
    start = pd.Timestamp(day, tz="Europe/Paris")
    end = start+pd.DateOffset(days=1)
    times = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
    one = make_history(day, days=1, hours=(0,)).drop(columns=["actual", "label_available_at_utc"])
    current = pd.concat([one]*len(times), ignore_index=True)
    current["timestamp_utc"] = times
    empty_history = make_history(days=1).iloc[:0]
    state = module.fit_interval_state(empty_history, cutoff(day), zones=["DE"])
    output = module.apply_interval_state(state, current)
    assert len(output) == (23 if day.endswith("03-29") else 25 if day.endswith("10-25") else 24)
    assert output.candidate_q90.eq(110).all()


@pytest.mark.parametrize("mutate", ["naive_origin", "early_label", "duplicate", "non_bool_active", "crossed", "unknown_country", "future_state", "stale_state"])
def test_invalid_contracts_rejected(mutate):
    history = make_history(days=10)
    current = future()
    if mutate == "naive_origin":
        history["forecast_origin_utc"] = history.forecast_origin_utc.dt.tz_localize(None)
    elif mutate == "early_label":
        history["label_available_at_utc"] = history.forecast_origin_utc
    elif mutate == "duplicate":
        history = pd.concat([history, history.iloc[[0]]], ignore_index=True)
    elif mutate == "non_bool_active":
        history["intervention_active"] = 1
    elif mutate == "crossed":
        history["candidate_q10"] = 101.
    if mutate in ("unknown_country", "future_state", "stale_state"):
        state = module.fit_interval_state(history, cutoff("2026-03-01"))
        if mutate == "unknown_country":
            current["zone"] = "UNKNOWN"
        else:
            state["fit_cutoff_utc"] = cutoff("2026-03-02" if mutate == "future_state" else "2026-02-28").isoformat()
        with pytest.raises(module.IntervalCalibrationError):
            module.apply_interval_state(state, current)
    else:
        with pytest.raises(module.IntervalCalibrationError):
            module.fit_interval_state(history, cutoff("2026-03-01"))
