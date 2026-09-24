from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from economic_value.extreme_policy import ExtremePolicyError, _net_per_mw, run_extreme_policy


def make_data(*, evaluation_start="2026-01-01", evaluation_days=35, history_days=365, zones=("FR",)):
    start = pd.Timestamp(evaluation_start)
    first = (start - pd.Timedelta(days=history_days)).tz_localize("Europe/Paris")
    end = (start + pd.Timedelta(days=evaluation_days)).tz_localize("Europe/Paris")
    times = pd.date_range(first, end, freq="h", inclusive="left").tz_convert("UTC")
    localday = times.tz_convert("Europe/Paris").tz_localize(None).normalize()
    origin = (localday - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    publication = (localday - pd.Timedelta(days=1) + pd.Timedelta(hours=18)).tz_localize("Europe/Paris").tz_convert("UTC")
    parts = []
    for zone in zones:
        parts.append(pd.DataFrame({
            "timestamp_utc": times, "zone": zone, "reference_price": 100., "actual": 180.,
            "forecast_origin_utc": origin, "label_available_at_utc": publication,
            "reference_available_at_utc": origin - pd.Timedelta(days=1),
            "reference_eligible": True, "duration_hours": 1.,
            "feature_load_gw": 50., "feature_wind_gw": 5.,
            "feature_eligible": True, "feature_pit_certified": False,
        }))
    history = pd.concat(parts, ignore_index=True)
    evaluation = history.loc[history.timestamp_utc >= start.tz_localize("Europe/Paris").tz_convert("UTC")].copy().reset_index(drop=True)
    evaluation["forecast"] = 60.
    evaluation["q10"], evaluation["q90"] = 40., 90.
    evaluation["forecast_eligible"] = True
    evaluation["benchmark_forecast"] = 250.
    return history, evaluation


def config(**kwargs):
    return {"verbose": False, "max_iter": 2, "min_samples_leaf": 20, "refit_every_days": 7, **kwargs}


POLICY_COLUMNS = [
    "timestamp_utc", "zone", "baseline_position_fraction", "policy_position_fraction",
    "extreme_probability_up", "extreme_probability_down", "extreme_expected_edge",
    "selected_policy", "policy_available_at_utc", "governance_weight", "policy_reason",
]


def test_full_365_day_fit_and_weekly_cadence_have_explicit_bounds():
    history, evaluation = make_data(evaluation_days=9)
    result = run_extreme_policy(history, evaluation, config())
    assert len(result.folds) == 2
    assert result.folds.status.eq("trained").all()
    assert result.folds.training_calendar_days_present.eq(365).all()
    assert result.folds.training_calendar_days_required.eq(365).all()
    assert result.folds.training_coverage.eq(1).all()
    for fold in result.folds.itertuples():
        assert (pd.Timestamp(fold.training_end_day) - pd.Timestamp(fold.training_start_day)).days == 364
        assert pd.Timestamp(fold.training_end_day) < pd.Timestamp(fold.fit_delivery_day)
        assert fold.max_training_label_available_at_utc <= fold.fit_cutoff_utc
    assert result.decisions.fit_age_civil_days.max() == 6
    assert not result.audit["refit_is_daily"]


def test_governance_warmup_retains_baseline_then_selects_oof_gain():
    history, evaluation = make_data(evaluation_days=35)
    result = run_extreme_policy(history, evaluation, config())
    rows = result.decisions
    early = rows.loc[rows.timestamp_utc < pd.Timestamp("2026-01-29", tz="Europe/Paris")]
    late = rows.loc[rows.timestamp_utc >= pd.Timestamp("2026-01-29", tz="Europe/Paris")]
    assert early.selected_policy.eq("baseline").all()
    assert early.policy_position_fraction.eq(-1).all()
    assert late.selected_policy.eq("blend_tail").all()
    assert late.policy_position_fraction.eq(0).all()
    assert result.audit["changed_position_rows"] == 7 * 24
    selected = result.governance.loc[result.governance.get("selected", pd.Series(False, index=result.governance.index)).fillna(False)]
    assert selected.oof_complete_days.ge(28).all()
    assert (selected.max_oof_label_available_at_utc <= selected.decision_cutoff_utc).all()
    assert (selected.oof_last_day < selected.delivery_day).all()


def test_future_realised_values_cannot_change_earlier_fits_or_decisions():
    history, evaluation = make_data(evaluation_days=37)
    original = run_extreme_policy(history, evaluation, config())
    changed_history, changed_evaluation = history.copy(), evaluation.copy()
    mutation_start = pd.Timestamp("2026-02-03", tz="Europe/Paris")
    changed_history.loc[changed_history.timestamp_utc >= mutation_start, "actual"] = -500.
    changed_evaluation.loc[changed_evaluation.timestamp_utc >= mutation_start, "actual"] = -500.
    changed = run_extreme_policy(changed_history, changed_evaluation, config())
    # The predicted day itself is not known at its own D-1 08:00 cutoff either.
    through_mutated_day = pd.Timestamp("2026-02-04", tz="Europe/Paris")
    a = original.decisions.loc[original.decisions.timestamp_utc < through_mutated_day, POLICY_COLUMNS].reset_index(drop=True)
    b = changed.decisions.loc[changed.decisions.timestamp_utc < through_mutated_day, POLICY_COLUMNS].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_changing_storm_and_baseline_history_never_changes_expert_or_governance():
    history, evaluation = make_data(evaluation_days=30)
    first = run_extreme_policy(history, evaluation, config())
    changed_h = history.copy()
    changed_h["forecast"] = -100000.
    changed_h["benchmark_forecast"] = 100000.
    changed_e = evaluation.copy()
    changed_e["benchmark_forecast"] = -999999.
    second = run_extreme_policy(changed_h, changed_e, config())
    pd.testing.assert_frame_equal(first.decisions[POLICY_COLUMNS], second.decisions[POLICY_COLUMNS])
    assert not first.audit["benchmark_used_for_governance"]
    assert not first.audit["baseline_forecasts_used_for_training"]


@pytest.mark.parametrize("name", ["feature_storm_price", "feature_benchmark", "feature_actual_price", "feature_target", "feature_pnl", "feature_baseline_forecast"])
def test_forbidden_target_or_benchmark_features_are_rejected(name):
    history, evaluation = make_data(evaluation_days=1)
    history[name] = 1.
    with pytest.raises(ExtremePolicyError):
        run_extreme_policy(history, evaluation, config())


def test_metadata_flags_are_not_predictors_and_explicit_misuse_is_rejected():
    history, evaluation = make_data(evaluation_days=1)
    result = run_extreme_policy(history, evaluation, config())
    assert result.audit["feature_columns"] == ["feature_load_gw", "feature_wind_gw"]
    with pytest.raises(ExtremePolicyError):
        run_extreme_policy(history, evaluation, config(feature_columns=["feature_eligible"]))


def test_short_history_does_not_silently_shorten_training_window():
    history, evaluation = make_data(history_days=364, evaluation_days=1)
    result = run_extreme_policy(history, evaluation, config())
    assert result.folds.iloc[0].reason == "insufficient_365_day_calendar"
    assert result.folds.iloc[0].training_calendar_days_present == 364
    assert result.decisions.policy_position_fraction.eq(result.decisions.baseline_position_fraction).all()
    assert not result.decisions.expert_eligible.any()
    assert result.audit["trained_folds"] == 0


def test_missing_whole_calendar_day_is_rejected_even_with_lenient_hourly_coverage():
    history, evaluation = make_data(evaluation_days=1)
    missing = history.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d").eq("2025-03-01")
    result = run_extreme_policy(history.loc[~missing], evaluation, config(minimum_training_coverage=.5))
    assert result.folds.iloc[0].reason == "insufficient_365_day_calendar"


def test_feature_coverage_threshold_is_explicit_and_missing_eval_features_fall_back():
    history, evaluation = make_data(evaluation_days=1)
    bad = history.copy()
    bad.loc[bad.index[:600], "feature_eligible"] = False
    result = run_extreme_policy(bad, evaluation, config())
    assert result.folds.iloc[0].reason == "insufficient_training_feature_or_label_coverage"
    assert result.folds.iloc[0].training_coverage < .95
    evaluation.loc[0, "feature_eligible"] = False
    good = run_extreme_policy(history, evaluation, config())
    assert good.decisions.iloc[0].policy_reason == "fundamental_features_unavailable_baseline_fallback"
    assert not good.decisions.iloc[0].expert_eligible
    assert len(good.decisions) == 24


def test_optional_nans_are_supported_without_zero_filling():
    history, evaluation = make_data(evaluation_days=1)
    history["feature_optional_weather"] = np.nan
    evaluation["feature_optional_weather"] = np.nan
    result = run_extreme_policy(history, evaluation, config())
    assert result.folds.status.eq("trained").all()
    assert result.decisions.expert_eligible.all()
    assert result.decisions.feature_optional_weather.isna().all()


def test_missing_label_availability_blocks_training_instead_of_inventing_publication():
    history, evaluation = make_data(evaluation_days=1)
    result = run_extreme_policy(history.drop(columns="label_available_at_utc"), evaluation, config())
    assert result.audit["trained_folds"] == 0
    assert not result.audit["label_availability_assumption_created_by_engine"]
    assert not result.audit["training_label_availability_is_explicit"]


def test_labels_not_yet_available_at_fit_cutoff_are_excluded():
    history, evaluation = make_data(evaluation_days=1)
    history.loc[:999, "label_available_at_utc"] = pd.Timestamp("2027-01-01", tz="UTC")
    result = run_extreme_policy(history, evaluation, config())
    assert result.folds.iloc[0].labels_unavailable_at_cutoff == 1000
    assert result.folds.iloc[0].status == "unavailable"


def test_explicit_label_rejection_is_respected_for_training_and_governance():
    history, evaluation = make_data(evaluation_days=30)
    history["label_eligible"] = False
    result = run_extreme_policy(history, evaluation, config())
    assert result.folds.training_valid_hours.eq(0).all()
    history["label_eligible"] = True
    evaluation["label_eligible"] = False
    result = run_extreme_policy(history, evaluation, config())
    assert result.decisions.selected_policy.eq("baseline").all()
    assert result.governance.oof_complete_days.eq(0).all()


def test_training_cannot_use_a_later_than_08_information_set():
    history, evaluation = make_data(evaluation_days=1)
    history.loc[0, "forecast_origin_utc"] += pd.Timedelta(hours=1)
    with pytest.raises(ExtremePolicyError, match="08:00"):
        run_extreme_policy(history, evaluation, config())


def test_late_features_are_excluded_before_training_and_inference():
    history, evaluation = make_data(evaluation_days=1)
    history["feature_available_at_utc"] = history["forecast_origin_utc"] + pd.Timedelta(hours=1)
    result = run_extreme_policy(history, evaluation, config())
    assert result.folds.training_valid_hours.eq(0).all()
    history["feature_available_at_utc"] = history["forecast_origin_utc"]
    evaluation["feature_available_at_utc"] = evaluation["forecast_origin_utc"] + pd.Timedelta(hours=1)
    result = run_extreme_policy(history, evaluation, config())
    assert not result.decisions.expert_eligible.any()


def test_preknown_evaluation_labels_cannot_be_mislabelled_as_oof_governance():
    history, evaluation = make_data(evaluation_days=35)
    evaluation["label_available_at_utc"] = evaluation["forecast_origin_utc"]
    result = run_extreme_policy(history, evaluation, config())
    assert result.decisions.selected_policy.eq("baseline").all()
    assert result.governance.oof_complete_days.eq(0).all()


@pytest.mark.parametrize(("start", "hours"), [("2026-03-29", 23), ("2026-10-25", 25)])
def test_dst_day_is_not_forced_to_24_observations(start, hours):
    history, evaluation = make_data(evaluation_start=start, evaluation_days=1)
    result = run_extreme_policy(history, evaluation, config())
    assert len(result.decisions) == hours
    assert result.decisions.timestamp_utc.nunique() == hours
    assert result.folds.iloc[0].training_calendar_days_present == 365
    assert result.folds.iloc[0].training_valid_hours == result.folds.iloc[0].training_expected_hours


def test_fractional_candidates_are_bounded_and_cost_uses_absolute_energy():
    history, evaluation = make_data(evaluation_days=1, zones=("FR", "DE"))
    result = run_extreme_policy(history, evaluation, config())
    assert result.decisions.candidate_reduce_opposite_fraction.eq(-.5).all()
    assert result.decisions.candidate_blend_tail_fraction.eq(0).all()
    assert result.decisions.policy_position_fraction.abs().le(1).all()
    assert np.allclose(_net_per_mw(np.array([-1., -.5, .5, 0.]), np.array([10., 10., 10., 10.]), np.ones(4), 1.), [-11., -5.5, 4.5, 0.])
    # Fractions are per allocated MW; the downstream engine alone allocates MW.
    assert result.decisions.loc[result.decisions.zone.eq("FR"), "policy_position_fraction"].abs().max() * 25 == 25


def test_cost_hurdle_applies_to_baseline_and_tail_candidates():
    history, evaluation = make_data(evaluation_days=1)
    evaluation["forecast"] = 106.
    result = run_extreme_policy(history, evaluation, config())
    assert result.decisions.baseline_position_fraction.eq(0).all()
    assert result.decisions.candidate_blend_tail_fraction.eq(.5).all()
    assert result.decisions.policy_position_fraction.eq(0).all()


def test_history_and_evaluation_are_not_modified():
    history, evaluation = make_data(evaluation_days=1)
    h, e = history.copy(deep=True), evaluation.copy(deep=True)
    run_extreme_policy(history, evaluation, config())
    pd.testing.assert_frame_equal(history, h)
    pd.testing.assert_frame_equal(evaluation, e)


def test_training_is_deterministic_with_mixed_extreme_classes():
    history, evaluation = make_data(evaluation_days=1)
    # A predictable fundamental regime, not a benchmark forecast or current label.
    regime = np.where(history.timestamp_utc.dt.hour < 8, -1, np.where(history.timestamp_utc.dt.hour < 16, 0, 1))
    history["feature_load_gw"] = 50. + 10. * regime
    history["actual"] = 100. + 80. * regime
    evaluation["feature_load_gw"] = 60.
    a = run_extreme_policy(history, evaluation, config(max_iter=8))
    b = run_extreme_policy(history, evaluation, config(max_iter=8))
    pd.testing.assert_frame_equal(a.decisions[POLICY_COLUMNS], b.decisions[POLICY_COLUMNS])
    assert a.folds.iloc[0].classifier_method == "hgb"
    assert a.decisions.extreme_probability_up.between(0, 1).all()
    assert (a.decisions.extreme_probability_up + a.decisions.extreme_probability_down).le(1 + 1e-12).all()


def test_strict_08_cutoff_and_invalid_settings_rejected():
    history, evaluation = make_data(evaluation_days=1)
    evaluation["forecast_origin_utc"] += pd.Timedelta(hours=1)
    with pytest.raises(ExtremePolicyError, match="08:00"):
        run_extreme_policy(history, evaluation, config())
    with pytest.raises(ExtremePolicyError, match="365"):
        run_extreme_policy(history, evaluation, config(training_days=364))
    with pytest.raises(ExtremePolicyError):
        run_extreme_policy(history, evaluation, config(expert_blend_weight=1.1))


def test_changed_day_minimum_and_uncertainty_margin_can_keep_baseline():
    history, evaluation = make_data(evaluation_days=35)
    result = run_extreme_policy(history, evaluation, config(governance_minimum_changed_days=60))
    assert result.decisions.selected_policy.eq("baseline").all()
    assert "insufficient_changed_days" in set(result.governance.reason)
    result = run_extreme_policy(history, evaluation, config(governance_minimum_gain_eur_mwh=1000))
    assert result.decisions.selected_policy.eq("baseline").all()


def test_missing_reference_or_baseline_retains_every_hour_without_new_exposure():
    history, evaluation = make_data(evaluation_days=1)
    evaluation.loc[0, "reference_price"] = np.nan
    evaluation.loc[1, "forecast"] = np.nan
    result = run_extreme_policy(history, evaluation, config())
    assert len(result.decisions) == 24
    assert result.decisions.iloc[:2].policy_position_fraction.eq(0).all()
    assert result.decisions.iloc[:2].policy_reason.eq("baseline_or_reference_unavailable").all()
