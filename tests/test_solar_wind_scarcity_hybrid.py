"""Synthetic-only safeguards for the NYX/Test2 hourly hybrid selector.

No production source or previous experiment is read, edited, or refitted.
"""

from itertools import product
import json

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import solar_wind_scarcity_hybrid as subject


TIMEZONE = "Europe/Berlin"
ORIGIN = pd.Timestamp("2026-04-15").date()
QUANTILES = ("q10", "q50", "q90")


def with_verified_peak_gap(frame):
    result = frame.copy(deep=True)
    days = result.index.tz_convert(TIMEZONE).date
    peak = result["nyx__q50"].groupby([result.zone.to_numpy(), days]).transform("max")
    result["nyx_daily_peak_gap"] = peak.to_numpy() - result["nyx__q50"].to_numpy()
    return result


def panel(start, days, *, historical=False, shift=10.0):
    first = pd.Timestamp(start)
    physical = pd.date_range(first.tz_localize(TIMEZONE),
                             (first + pd.Timedelta(days=days)).tz_localize(TIMEZONE),
                             freq="h", inclusive="left").tz_convert("UTC")
    index = physical.repeat(2)
    frame = pd.DataFrame({"zone": np.tile(["DE", "NL"], len(physical)),
                          "nyx__q10": 180.0, "nyx__q50": 220.0, "nyx__q90": 280.0,
                          "test2__q10": 220.0 + shift - 17.0,
                          "test2__q50": 220.0 + shift,
                          "test2__q90": 220.0 + shift + 31.0,
                          "own_joint_deficit": 0.6, "own_residual_stress": 0.2}, index=index)
    frame = with_verified_peak_gap(frame)
    if historical:
        frame["actual"] = frame["test2__q50"]
        frame["fit_origin"] = frame.index.tz_convert(TIMEZONE).date
    return frame


def past(*, origin=ORIGIN, days=90, shift=10.0):
    return panel(pd.Timestamp(origin) - pd.Timedelta(days=days), days, historical=True, shift=shift)


def selected_policy(*, origin=ORIGIN):
    return subject.select_rule(past(origin=origin), origin)


def assert_exact_routing(result, original):
    selected = result.selected_test2.to_numpy(dtype=bool)
    for quantile in QUANTILES:
        expected = np.where(selected, original["test2__" + quantile], original["nyx__" + quantile])
        np.testing.assert_array_equal(result["hybrid__" + quantile].to_numpy(), expected)
    assert np.isfinite(result[["hybrid__" + q for q in QUANTILES]].to_numpy(float)).all()
    assert (np.diff(result[["hybrid__" + q for q in QUANTILES]].to_numpy(float), axis=1) >= 0).all()
    assert result.selected_model.eq(np.where(selected, "Test2", "NYX")).all()


def test_twelve_preregistered_rules_and_deterministic_safe_tie_break():
    policy = selected_policy()
    expected = [dict(nyx_p50_min=p50, peak_gap_max=gap, upside_min=upside)
                for p50, gap, upside in product((150, 200, 250), (25, 50), (0, 50))]
    assert [score["rule"] for score in policy["candidate_scores"]] == expected
    assert policy["mode"] == "hybrid" and policy["rule"] == expected[0]
    assert policy["window_days"] == 90
    assert policy["rows"] == 90 * 24 * 2 - 2  # Both countries retain physical spring DST hours.
    assert policy["support"]["days"] == 90
    assert policy["pooled_mae_gain"] == 10.0
    json.dumps(policy, allow_nan=False)


def test_selector_chooses_lowest_pooled_mae_not_first_eligible_rule():
    frame = past()
    peak = frame.index.tz_convert(TIMEZONE).hour == 19
    frame.loc[peak, ["nyx__" + q for q in QUANTILES]] += 40.0
    candidate = frame.nyx__q50.to_numpy() + np.where(peak, 10.0, 1.0)
    frame["actual"] = candidate
    for q, offset in (("q10", -17.0), ("q50", 0.0), ("q90", 31.0)):
        frame["test2__" + q] = candidate + offset
    frame = with_verified_peak_gap(frame)
    policy = subject.select_rule(frame, ORIGIN)
    assert policy["candidate_scores"][0]["eligible"]
    assert policy["rule"] == {"nyx_p50_min": 150, "peak_gap_max": 50, "upside_min": 0}
    assert policy["selected_metrics"]["pooled"]["mae"] == 0.0
    assert policy["candidate_scores"][0]["metrics"]["pooled"]["mae"] > 0.0


def test_selection_uses_only_trailing_ninety_days_and_does_not_mutate_inputs():
    frame = past(days=100)
    before = frame.copy(deep=True)
    original = subject.select_rule(frame, ORIGIN)
    old = frame.index.tz_convert(TIMEZONE).date < (pd.Timestamp(ORIGIN) - pd.Timedelta(days=90)).date()
    changed = frame.copy(deep=True)
    changed.loc[old, "actual"] = -9999.0
    altered = subject.select_rule(changed, ORIGIN)
    trailing = subject.select_rule(frame.loc[~old].copy(), ORIGIN)
    for key in ("mode", "rule", "support", "selected_metrics", "baseline_metrics", "pooled_mae_gain"):
        assert original[key] == altered[key] == trailing[key]
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("offset", [0, 1])
def test_origin_or_future_calibration_labels_are_rejected(offset):
    frame = past()
    forbidden = panel(pd.Timestamp(ORIGIN) + pd.Timedelta(days=offset), 1, historical=True)
    with pytest.raises(ValueError):
        subject.select_rule(pd.concat([frame, forbidden]), ORIGIN)


def test_future_fit_origin_for_a_historical_target_is_rejected():
    frame = past()
    frame.iloc[-1, frame.columns.get_loc("fit_origin")] = ORIGIN
    with pytest.raises(ValueError):
        subject.select_rule(frame, ORIGIN)


@pytest.mark.parametrize("origin,hours", [("2026-03-29", 23), ("2025-10-26", 25)])
def test_apply_preserves_complete_physical_dst_profiles_in_both_countries(origin, hours):
    policy = selected_policy(origin=pd.Timestamp(origin).date())
    current = panel(origin, 1)
    result = subject.apply_rule(current, policy)
    pd.testing.assert_index_equal(result.index, current.index)
    assert len(result) == hours * 2
    assert all(int(result.zone.eq(zone).sum()) == hours for zone in ("DE", "NL"))
    assert_exact_routing(result, current)


def test_apply_never_crosses_to_tomorrows_forecast_for_daily_peak_detection():
    policy = selected_policy()
    current = panel(ORIGIN, 2)
    original = subject.apply_rule(current, policy)
    tomorrow = current.index.tz_convert(TIMEZONE).date > ORIGIN
    changed = current.copy(deep=True)
    changed.loc[tomorrow, ["nyx__q10", "nyx__q50", "nyx__q90"]] += 1000.0
    changed = with_verified_peak_gap(changed)
    result = subject.apply_rule(changed, policy)
    pd.testing.assert_frame_equal(original.loc[~tomorrow], result.loc[~tomorrow])


def test_gate_uses_country_specific_nyx_daily_max_and_or_physical_condition_only():
    current = panel(ORIGIN, 1)
    hour = current.index.tz_convert(TIMEZONE).hour
    de = current.zone.eq("DE").to_numpy()
    # A DE peak must not suppress corresponding NL hours.
    current.loc[de & (hour == 19), ["nyx__q10", "nyx__q50", "nyx__q90"]] += 100.0
    current = with_verified_peak_gap(current)
    current["own_joint_deficit"] = 0.0
    current["own_residual_stress"] = 0.0
    current.loc[hour == 18, "own_joint_deficit"] = 0.5
    current.loc[hour == 19, "own_residual_stress"] = 1.0
    result = subject.apply_rule(current, selected_policy())
    expected = ((~de & np.isin(hour, [18, 19])) | (de & (hour == 19)))
    np.testing.assert_array_equal(result.selected_test2, expected)
    assert_exact_routing(result, current)


@pytest.mark.parametrize("boundary", ["level", "upside", "peak_gap"])
def test_current_rule_honors_exact_preregistered_threshold_boundaries(boundary):
    current = panel(ORIGIN, 1)
    policy = selected_policy()
    if boundary == "level":
        policy["rule"] = {"nyx_p50_min": 200, "peak_gap_max": 25, "upside_min": 0}
        current.iloc[0, current.columns.get_loc("nyx__q50")] = 200.0
        current.iloc[1, current.columns.get_loc("nyx__q50")] = 199.999
    elif boundary == "upside":
        policy["rule"] = {"nyx_p50_min": 150, "peak_gap_max": 25, "upside_min": 50}
        current.iloc[0, current.columns.get_loc("nyx__q90")] = 270.0
        current.iloc[1, current.columns.get_loc("nyx__q90")] = 269.999
    else:
        current.iloc[0, current.columns.get_loc("nyx__q50")] = 195.0
        current.iloc[1, current.columns.get_loc("nyx__q50")] = 194.999
    current = with_verified_peak_gap(current)
    result = subject.apply_rule(current, policy)
    assert bool(result.selected_test2.iloc[0]) is True
    assert bool(result.selected_test2.iloc[1]) is False
    assert_exact_routing(result, current)


def test_current_gate_is_invariant_to_test2_predictions_and_preserves_all_three_quantiles():
    current = panel(ORIGIN, 1)
    current.loc[current.index.tz_convert(TIMEZONE).hour < 12, "own_joint_deficit"] = 0.0
    before = current.copy(deep=True)
    policy = selected_policy()
    original = subject.apply_rule(current, policy)
    changed = current.copy(deep=True)
    changed[["test2__" + q for q in QUANTILES]] -= 300.0
    result = subject.apply_rule(changed, policy)
    np.testing.assert_array_equal(original.selected_test2, result.selected_test2)
    assert_exact_routing(result, changed)
    assert (result.loc[result.selected_test2, "hybrid__q50"] < changed.loc[result.selected_test2, "nyx__q50"]).all()
    pd.testing.assert_frame_equal(current, before)


@pytest.mark.parametrize("shift", [-120.0, 120.0])
def test_hybrid_has_no_additional_correction_cap_or_forced_upward_shift(shift):
    policy = subject.select_rule(past(shift=shift), ORIGIN)
    current = panel(ORIGIN, 1, shift=shift)
    result = subject.apply_rule(current, policy)
    assert result.selected_test2.all()
    np.testing.assert_array_equal(result.hybrid__q50 - current.nyx__q50, np.full(len(current), shift))
    assert_exact_routing(result, current)


def test_actual_column_is_forbidden_even_when_entirely_nan_in_current_inputs():
    current = panel(ORIGIN, 1)
    current["actual"] = np.nan
    with pytest.raises(ValueError):
        subject.apply_rule(current, selected_policy())


@pytest.mark.parametrize("fault", ["new_threshold", "test2_selector", "nyx_with_active_rule"])
def test_unregistered_or_inconsistent_current_policies_are_rejected(fault):
    policy = selected_policy()
    if fault == "new_threshold":
        policy["rule"]["nyx_p50_min"] = 175.0
    elif fault == "test2_selector":
        policy["rule"]["test2_p50_min"] = 200.0
    else:
        policy["mode"] = "nyx"
    with pytest.raises(ValueError):
        subject.apply_rule(panel(ORIGIN, 1), policy)


@pytest.mark.parametrize("start,days", [(pd.Timestamp(ORIGIN) - pd.Timedelta(days=1), 1), (ORIGIN, 8)])
def test_apply_refuses_targets_before_origin_or_beyond_seven_day_block(start, days):
    with pytest.raises(ValueError):
        subject.apply_rule(panel(start, days), selected_policy())


@pytest.mark.parametrize("fault", ["missing_hour", "missing_country", "duplicate_key", "invalid_country",
                                   "naive", "partial_day", "missing_quantile", "nan_quantile",
                                   "crossed_nyx", "crossed_test2", "wrong_daily_peak"])
@pytest.mark.parametrize("stage", ["select", "apply"])
def test_invalid_country_hour_grids_quantiles_or_daily_peaks_fail_closed(stage, fault):
    frame = past() if stage == "select" else panel(ORIGIN, 1)
    if fault == "missing_hour":
        frame = frame.iloc[:-1].copy()
    elif fault == "missing_country":
        frame = frame.loc[frame.zone == "DE"].copy()
    elif fault == "duplicate_key":
        frame = pd.concat([frame, frame.iloc[[-1]]])
    elif fault == "invalid_country":
        frame.iloc[-1, frame.columns.get_loc("zone")] = "FR"
    elif fault == "naive":
        frame.index = frame.index.tz_localize(None)
    elif fault == "partial_day":
        frame = frame.iloc[4:].copy()
    elif fault == "missing_quantile":
        frame = frame.drop(columns="test2__q90")
    elif fault == "nan_quantile":
        frame.iloc[0, frame.columns.get_loc("nyx__q90")] = np.nan
    elif fault == "crossed_nyx":
        frame.iloc[0, frame.columns.get_loc("nyx__q10")] = 9999.0
    elif fault == "crossed_test2":
        frame.iloc[0, frame.columns.get_loc("test2__q90")] = -9999.0
    else:
        frame.iloc[0, frame.columns.get_loc("nyx_daily_peak_gap")] = 1.0
    with pytest.raises(ValueError):
        if stage == "select":
            subject.select_rule(frame, ORIGIN)
        else:
            subject.apply_rule(frame, selected_policy())


@pytest.mark.parametrize("gain,mode", [(0.019, "nyx"), (0.021, "hybrid")])
def test_minimum_absolute_pooled_mae_gain_is_respected(gain, mode):
    frame = past(shift=gain)
    policy = subject.select_rule(frame, ORIGIN)
    assert policy["mode"] == mode
    if mode == "nyx":
        assert policy["rule"] is None
        result = subject.apply_rule(panel(ORIGIN, 1), policy)
        assert not result.selected_test2.any()
        assert_exact_routing(result, panel(ORIGIN, 1))


@pytest.mark.parametrize("case", ["pooled_hours", "pooled_days", "country_hours", "country_days"])
def test_minimum_pooled_and_per_country_trigger_support_cannot_be_bypassed(case):
    frame = past()
    local = frame.index.tz_convert(TIMEZONE)
    unique_days = sorted(set(local.date))
    day_number = np.asarray([unique_days.index(day) for day in local.date])
    nl = frame.zone.eq("NL").to_numpy()
    if case == "pooled_hours":
        trigger = (day_number >= 82) & (local.hour == 19)  # 16 hours, 8 days.
    elif case == "pooled_days":
        trigger = day_number >= 86  # 192 hours, but only 4 days.
    elif case == "country_hours":
        trigger = ~nl | (nl & (day_number >= 86) & (local.hour == 19))  # NL: 4 hours/4 days.
    else:
        trigger = ~nl | (nl & (day_number == 89))  # NL: 24 hours, only 1 day.
    frame["own_joint_deficit"] = np.where(trigger, 0.6, 0.0)
    frame["own_residual_stress"] = 0.0
    policy = subject.select_rule(frame, ORIGIN)
    assert policy["mode"] == "nyx" and policy["rule"] is None
    assert not any(score["eligible"] for score in policy["candidate_scores"])


def test_country_mae_guard_blocks_pooled_gain_even_when_country_rmse_improves():
    frame = past()
    nl = frame.zone.eq("NL").to_numpy()
    country_row = np.cumsum(nl) - 1
    rare = nl & (country_row % 100 == 0)
    frame.loc[nl, "actual"] = frame.loc[nl, "nyx__q50"].to_numpy() + np.where(rare[nl], 100.0, 0.0)
    candidate = frame.loc[nl, "actual"].to_numpy() + np.where(rare[nl], 0.0, 2.0)
    for q, offset in (("q10", -17.0), ("q50", 0.0), ("q90", 31.0)):
        frame.loc[nl, "test2__" + q] = candidate + offset
    policy = subject.select_rule(frame, ORIGIN)
    first = policy["candidate_scores"][0]
    assert first["pooled_mae_gain"] > 0.02
    assert first["metrics"]["NL"]["rmse"] < policy["baseline_metrics"]["NL"]["rmse"]
    assert first["metrics"]["NL"]["mae"] > policy["baseline_metrics"]["NL"]["mae"] + 0.05
    assert policy["mode"] == "nyx"


def test_country_rmse_guard_blocks_rare_large_errors_despite_better_mae():
    frame = past()
    nl = frame.zone.eq("NL").to_numpy()
    country_row = np.cumsum(nl) - 1
    rare = nl & (country_row % 100 == 0)
    candidate = frame.loc[nl, "actual"].to_numpy() + np.where(rare[nl], 200.0, 0.0)
    for q, offset in (("q10", -17.0), ("q50", 0.0), ("q90", 31.0)):
        frame.loc[nl, "test2__" + q] = candidate + offset
    policy = subject.select_rule(frame, ORIGIN)
    first = policy["candidate_scores"][0]
    assert first["metrics"]["NL"]["mae"] < policy["baseline_metrics"]["NL"]["mae"]
    assert first["metrics"]["NL"]["rmse"] > policy["baseline_metrics"]["NL"]["rmse"]
    assert policy["mode"] == "nyx"
