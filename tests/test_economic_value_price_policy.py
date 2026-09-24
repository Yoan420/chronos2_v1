from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from economic_value.price_policy import PricePolicyError, _candidate_column, _govern, _net_pnl, _parameters, _prepare, run_price_policy


def panel(days=125, *, start="2026-01-01", zones=("FR",)):
    first = pd.Timestamp(start).tz_localize("Europe/Paris")
    last = (pd.Timestamp(start) + pd.Timedelta(days=days)).tz_localize("Europe/Paris")
    times = pd.date_range(first, last, freq="h", inclusive="left").tz_convert("UTC")
    date = times.tz_convert("Europe/Paris").tz_localize(None).normalize()
    origins = (date - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    available = (date - pd.Timedelta(days=1) + pd.Timedelta(hours=18)).tz_localize("Europe/Paris").tz_convert("UTC")
    return pd.concat([pd.DataFrame({
        "timestamp_utc": times, "zone": zone, "forecast": 95., "reference_price": 100., "actual": 135.,
        "q10": 70., "q90": 120., "forecast_origin_utc": origins, "label_available_at_utc": available,
        "reference_available_at_utc": origins - pd.Timedelta(days=1), "duration_hours": 1.,
        "forecast_eligible": True, "reference_eligible": True, "label_eligible": True,
        "feature_eligible": True, "feature_pit_certified": False, "feature_load_gw": 50.,
        "feature_wind_gw": 5., "benchmark_forecast": 160.,
    }) for zone in zones], ignore_index=True)


def config(**kwargs):
    return {"verbose": False, "max_iter": 2, "min_samples_leaf": 20, "feature_columns": ["feature_load_gw", "feature_wind_gw"], **kwargs}


PRICE_COLUMNS = ["timestamp_utc", "zone", "baseline_forecast", "candidate_forecast", "raw_residual_prediction",
                 "applied_correction", "selected_weight", "expert_ready", "reason", "expert_available_at_utc", "expert_fit_id"]


def test_warmup_is_explicit_and_partial_history_never_claims_365_days():
    data = panel(days=100)
    result = run_price_policy(data, config())
    warmup = result.decisions.loc[result.decisions.timestamp_utc < pd.Timestamp("2026-04-02", tz="Europe/Paris")]
    assert not warmup.expert_ready.any()
    assert warmup.candidate_forecast.eq(warmup.baseline_forecast).all()
    trained = result.folds.loc[result.folds.status.eq("trained")]
    assert trained.training_days.tolist() == [91, 98]
    assert not trained.full_365_day_training.any()
    assert result.audit["actual_training_days_min"] == 91
    assert result.audit["actual_training_days_max"] == 98
    assert not result.audit["all_trained_folds_have_365_days"]
    for row in trained.itertuples():
        assert (pd.Timestamp(row.training_end_day) - pd.Timestamp(row.training_start_day)).days + 1 == row.training_days
        assert row.max_training_label_available_at_utc <= row.fit_cutoff_utc


def test_strict_365_minimum_falls_back_on_first_365_day_panel():
    result = run_price_policy(panel(days=365), config(minimum_training_days=365))
    assert result.audit["trained_folds"] == 0
    assert not result.decisions.expert_ready.any()
    assert not result.decisions.full_365_day_training.any()
    assert result.decisions.candidate_forecast.eq(result.decisions.baseline_forecast).all()
    assert result.folds.training_days.max() == 364


def test_governor_requires_28_complete_oof_days_after_training_warmup():
    result = run_price_policy(panel(days=125), config())
    # First weekly fit with >=90 past days occurs at day index91. Governance can
    # first act at index119 after 28 genuine OOF delivery days, not at day28.
    first_action = pd.Timestamp("2026-01-01", tz="Europe/Paris") + pd.DateOffset(days=119)
    before = result.decisions.loc[result.decisions.timestamp_utc < first_action]
    after = result.decisions.loc[result.decisions.timestamp_utc >= first_action]
    assert before.selected_weight.eq(0).all()
    assert after.selected_weight.eq(.5).all()
    assert after.candidate_forecast.eq(115).all()
    assert after.applied_correction.eq(20).all()
    selected = result.governance.loc[result.governance.get("selected", pd.Series(False, index=result.governance.index)).eq(True)]
    assert selected.oof_complete_days.ge(28).all()
    assert selected.changed_forecast_days.ge(7).all()
    assert selected.mean_mae_gain_eur_mwh.ge(0).all()
    assert selected.net_gain_lower_bound_eur_mwh.ge(.05).all()
    assert (selected.oof_last_day < selected.delivery_day).all()
    assert (selected.max_oof_label_available_at_utc <= selected.decision_cutoff_utc).all()


def test_future_label_poisoning_leaves_forecasts_and_governance_prefix_unchanged():
    original = panel(days=130)
    poisoned = original.copy()
    poison_day = pd.Timestamp("2026-05-04", tz="Europe/Paris")
    poisoned.loc[poisoned.timestamp_utc >= poison_day, "actual"] = -10000.
    a = run_price_policy(original, config(use_past_residual_features=True))
    b = run_price_policy(poisoned, config(use_past_residual_features=True))
    end = poison_day + pd.DateOffset(days=1)
    pd.testing.assert_frame_equal(a.decisions.loc[a.decisions.timestamp_utc < end, PRICE_COLUMNS].reset_index(drop=True),
                                  b.decisions.loc[b.decisions.timestamp_utc < end, PRICE_COLUMNS].reset_index(drop=True))
    prefix = poison_day.strftime("%Y-%m-%d")
    pd.testing.assert_frame_equal(a.governance.loc[a.governance.delivery_day <= prefix].reset_index(drop=True),
                                  b.governance.loc[b.governance.delivery_day <= prefix].reset_index(drop=True))


def test_changing_storm_never_changes_price_fit_or_governance():
    original = panel(days=123)
    changed = original.copy()
    changed["benchmark_forecast"] = -999999.
    changed["storm_pnl"] = 1e30
    a, b = run_price_policy(original, config()), run_price_policy(changed, config())
    pd.testing.assert_frame_equal(a.decisions[PRICE_COLUMNS], b.decisions[PRICE_COLUMNS])
    pd.testing.assert_frame_equal(a.governance, b.governance)
    assert not a.audit["benchmark_used_as_feature"]
    assert not a.audit["benchmark_used_for_governance"]


def test_costs_and_fixed_hurdle_are_symmetric_for_buy_sell_and_fractionless_prices():
    p = _parameters(config())
    actual = np.array([110., 90., 110., 110.])
    forecasts = np.array([110., 90., 106., 106.01])
    profit = _net_pnl(forecasts, np.full(4, 100.), actual, np.ones(4), p)
    assert np.allclose(profit, [9., 9., 0., 9.])


def test_price_candidates_are_not_accepted_for_mae_improvement_alone():
    data = panel(days=125)
    data["forecast"] = 115.
    # Both old and corrected forecasts stay BUY, so net-PnL gain is zero.
    result = run_price_policy(data, config())
    assert result.decisions.selected_weight.eq(0).all()
    evaluated = result.governance.loc[result.governance.weight.gt(0)]
    assert evaluated.mean_mae_gain_eur_mwh.gt(0).all()
    assert evaluated.mean_net_gain_eur_mwh.eq(0).all()
    assert evaluated.reason.eq("insufficient_net_economic_gain").all()


def test_mae_guard_can_reject_an_economically_better_candidate():
    data = panel(days=28)
    data["actual"] = 105.
    parameters = _parameters(config())
    past, _ = _prepare(data, parameters)
    past["expert_ready"] = True
    past["expert_available_at_utc"] = past["forecast_origin_utc"]
    for weight in parameters["candidate_weights"]:
        past[_candidate_column(weight)] = past["baseline_forecast"] + 100. * weight
    # Baseline95 is FLAT against reference100; candidate120 is BUY and earns
    # 4 EUR/MW net, but its MAE15 is worse than baselineMAE10. Default guard must
    # reject despite genuinely positive economic evidence.
    selected, _, records = _govern([past], "2026-01-29", pd.Timestamp("2026-01-28T08:00:00+01:00").tz_convert("UTC"), "FR", parameters)
    assert selected == 0
    assert all(record["mean_net_gain_eur_mwh"] > 0 for record in records)
    assert all(record["mean_mae_gain_eur_mwh"] < 0 for record in records)
    assert all(record["reason"] == "mae_non_regression_failed" for record in records)


def test_full_training_window_is_365_only_when_that_history_really_exists():
    result = run_price_policy(panel(days=373), config(minimum_training_days=365))
    trained = result.folds.loc[result.folds.status.eq("trained")]
    assert len(trained) == 1
    assert trained.iloc[0].training_days == 365
    assert trained.iloc[0].full_365_day_training
    assert (pd.Timestamp(trained.iloc[0].training_end_day) - pd.Timestamp(trained.iloc[0].training_start_day)).days == 364
    assert result.audit["all_trained_folds_have_365_days"]


def test_trigger_and_correction_cap_are_applied_before_fixed_weights():
    data = panel(days=100)
    data["actual"] = 395.
    result = run_price_policy(data, config())
    ready = result.decisions.loc[result.decisions.expert_ready]
    assert ready.raw_residual_prediction.eq(300).all()
    assert ready[_candidate_column(.25)].eq(120).all()
    assert ready[_candidate_column(.5)].eq(145).all()
    below = panel(days=100)
    below["actual"] = 104.
    result = run_price_policy(below, config())
    ready = result.decisions.loc[result.decisions.expert_ready]
    assert ready.raw_residual_prediction.eq(9).all()
    assert ready[_candidate_column(.5)].eq(95).all()
    assert ready.reason.eq("predicted_residual_below_fixed_trigger").all()


def test_missing_calendar_day_is_not_compressed_into_shorter_history():
    data = panel(days=100)
    day = data.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")
    data = data.loc[day.ne("2026-02-01")]
    result = run_price_policy(data, config())
    assert result.audit["trained_folds"] == 0
    assert "nonconsecutive_training_calendar" in set(result.folds.reason)


def test_feature_or_label_unavailable_whole_day_blocks_training_calendar():
    data = panel(days=100)
    day = data.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")
    data.loc[day.eq("2026-02-01"), "label_eligible"] = False
    result = run_price_policy(data, config())
    assert result.audit["trained_folds"] == 0
    assert "training_calendar_has_unavailable_days" in set(result.folds.reason)


def test_late_labels_do_not_enter_training_and_no_vintages_are_invented():
    data = panel(days=100)
    data.loc[:23, "label_available_at_utc"] = pd.Timestamp("2027-01-01", tz="UTC")
    result = run_price_policy(data, config())
    assert result.audit["trained_folds"] == 0
    result = run_price_policy(data.drop(columns="label_available_at_utc"), config())
    assert result.audit["trained_folds"] == 0
    assert not result.audit["label_availability_column_supplied"]
    assert not result.audit["label_availability_assumption_created_by_engine"]


def test_preknown_labels_cannot_be_used_as_oof_residual_training():
    data = panel(days=100)
    data["label_available_at_utc"] = data["forecast_origin_utc"]
    result = run_price_policy(data, config())
    assert result.audit["trained_folds"] == 0


def test_one_incomplete_oof_day_delays_governance_instead_of_counting_zero():
    data = panel(days=121)
    missing_day = pd.Timestamp("2026-01-01", tz="Europe/Paris") + pd.DateOffset(days=105)
    idx = data.index[data.timestamp_utc.eq(missing_day)][0]
    data.loc[idx, "label_eligible"] = False
    result = run_price_policy(data, config())
    expected = pd.Timestamp("2026-01-01", tz="Europe/Paris") + pd.DateOffset(days=120)
    selected = result.decisions.loc[result.decisions.selected_weight.gt(0)]
    assert selected.timestamp_utc.min() == expected


@pytest.mark.parametrize(("start", "dst_day", "hours"), [("2026-01-01", "2026-03-29", 23), ("2026-07-01", "2026-10-25", 25)])
def test_dst_physical_hours_and_calendars_are_retained(start, dst_day, hours):
    result = run_price_policy(panel(days=125, start=start), config())
    selected = result.decisions.loc[result.decisions.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d").eq(dst_day)]
    assert len(selected) == hours
    assert selected.timestamp_utc.nunique() == hours
    trained = result.folds.loc[result.folds.status.eq("trained")]
    assert trained.training_coverage.eq(1).all()


def test_past_residual_features_use_each_rows_origin_not_a_future_fit_cutoff():
    data = panel(days=5)
    before, features = _prepare(data, _parameters(config(use_past_residual_features=True)))
    poisoned = data.copy()
    poison_time = pd.Timestamp("2026-01-03", tz="Europe/Paris")
    poisoned.loc[poisoned.timestamp_utc >= poison_time, "actual"] = 99999.
    after, _ = _prepare(poisoned, _parameters(config(use_past_residual_features=True)))
    through = pd.Timestamp("2026-01-04", tz="Europe/Paris")
    columns = [name for name in features if "known_residual" in name]
    pd.testing.assert_frame_equal(before.loc[before.timestamp_utc < through, columns], after.loc[after.timestamp_utc < through, columns])
    assert before.loc[before._day.eq("2026-01-01"), columns].isna().all().all()
    assert before.loc[before._day.eq("2026-01-02"), columns].eq(40).all().all()


def test_optional_nan_features_and_invalid_intervals_do_not_invent_quantiles():
    data = panel(days=100)
    data["feature_optional_temperature"] = np.nan
    data["q10"], data["q90"] = 150., 50.
    result = run_price_policy(data, config(feature_columns=["feature_load_gw", "feature_wind_gw", "feature_optional_temperature"]))
    assert result.audit["trained_folds"] > 0
    assert result.decisions._context_interval_width.isna().all()
    assert result.decisions.q10.eq(150).all()
    assert not result.audit["quantiles_modified_by_engine"]


def test_missing_features_or_baseline_keep_every_row_and_no_correction():
    data = panel(days=125)
    data.loc[data.index[-2], "feature_eligible"] = False
    data.loc[data.index[-1], "forecast"] = np.nan
    result = run_price_policy(data, config())
    assert len(result.decisions) == len(data)
    assert not result.decisions.iloc[-2:].expert_ready.any()
    assert result.decisions.iloc[-2:].applied_correction.eq(0).all()
    assert np.isnan(result.decisions.iloc[-1].candidate_forecast)


def test_core_availability_and_late_feature_timestamps_are_respected():
    data = panel(days=100)
    data["feature_available_at_utc"] = data["forecast_origin_utc"] + pd.Timedelta(hours=1)
    result = run_price_policy(data, config())
    assert result.audit["trained_folds"] == 0


@pytest.mark.parametrize("feature", ["feature_storm", "feature_actual", "feature_pnl", "feature_baseline_forecast", "feature_eligible"])
def test_forbidden_external_features_are_rejected(feature):
    data = panel(days=2)
    if feature not in data:
        data[feature] = 1.
    with pytest.raises(PricePolicyError):
        run_price_policy(data, config(feature_columns=[feature]))


def test_strict_time_and_bad_configuration_contracts_fail_closed():
    data = panel(days=2)
    invalid = data.copy()
    invalid.loc[0, "forecast_origin_utc"] += pd.Timedelta(hours=1)
    with pytest.raises(PricePolicyError, match="08:00"):
        run_price_policy(invalid, config())
    with pytest.raises(PricePolicyError):
        run_price_policy(data, config(training_window_days=366))
    with pytest.raises(PricePolicyError):
        run_price_policy(data, config(training_window_days=180))
    with pytest.raises(PricePolicyError):
        run_price_policy(data, config(minimum_training_days=89))
    with pytest.raises(PricePolicyError, match="allowlist"):
        run_price_policy(data, config(feature_columns=None))
    with pytest.raises(PricePolicyError):
        run_price_policy(data, config(candidate_weights=[0., 1.1]))
    with pytest.raises(PricePolicyError):
        run_price_policy(data, config(timezone="UTC"))
    with pytest.raises(PricePolicyError):
        run_price_policy(pd.concat([data, data.iloc[:1]]), config())


def test_training_and_governance_are_deterministic_and_input_is_unchanged():
    data = panel(days=100)
    original = data.copy(deep=True)
    a, b = run_price_policy(data, config()), run_price_policy(data, config())
    pd.testing.assert_frame_equal(a.decisions[PRICE_COLUMNS], b.decisions[PRICE_COLUMNS])
    pd.testing.assert_frame_equal(a.folds, b.folds)
    pd.testing.assert_frame_equal(data, original)


def test_changed_forecast_day_guard_and_mae_threshold_are_explicit():
    result = run_price_policy(panel(days=125), config(governance_minimum_changed_days=60))
    assert result.decisions.selected_weight.eq(0).all()
    assert "insufficient_changed_forecast_days" in set(result.governance.reason)
