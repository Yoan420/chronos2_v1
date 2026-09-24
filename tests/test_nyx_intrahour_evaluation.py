"""Synthetic fixtures only: no provider, NYX model, archive or scientific runner."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nyx_intrahour.evaluation import ZONES, evaluate_variant


OPTIONS = dict(initial_train_days=14, validation_days=7, test_days=14,
               window_days=10, refit_every_days=7, min_train_rows=120,
               bootstrap_repetitions=80, seed=123)
SHAPE = "feature_intrahour_synthetic_load__ramp_gw_per_hour"
MEAN = "feature_intrahour_synthetic_load__mean_gw"


def synthetic_panel(start="2026-03-08", days=36):
    """A forecast-shape signal independent of the hourly control; explicitly fake."""
    first = pd.Timestamp(start, tz="Europe/Paris")
    last = (pd.Timestamp(start) + pd.Timedelta(days=days)).tz_localize("Europe/Paris")
    hours = pd.date_range(first, last, inclusive="left", freq="h").tz_convert("UTC")
    frame = pd.MultiIndex.from_product([hours, ZONES], names=["timestamp_utc", "zone"]).to_frame(index=False)
    rng = np.random.default_rng(410)
    frame[MEAN] = 10 + rng.normal(size=len(frame))
    frame[SHAPE] = rng.normal(size=len(frame))
    frame["feature_hourly_synthetic_forecast"] = rng.normal(size=len(frame))
    frame["nyx_q50"] = 40 + 5 * np.sin(2 * np.pi * frame.timestamp_utc.dt.hour / 24)
    # Repeated negative-price hours make at least one critical regime estimable.
    frame.loc[frame.timestamp_utc.dt.hour.between(1, 3), "nyx_q50"] -= 60
    frame["actual"] = frame.nyx_q50 + 7 * frame[SHAPE] + .05 * rng.normal(size=len(frame))
    frame["training_actual"] = frame.actual
    return frame


def local_days(frame):
    return frame.timestamp_utc.dt.tz_convert("Europe/Paris").dt.tz_localize(None).dt.normalize()


def test_shape_signal_beats_both_controls_without_authorizing_production():
    panel = synthetic_panel()
    original = panel.copy(deep=True)
    result = evaluate_variant(panel, **OPTIONS)
    pd.testing.assert_frame_equal(panel, original)
    assert result["status"] == "complete"
    assert result["selection"]["primary_family"] == "intrahour"
    assert result["decision"]["encouraging"]
    assert result["decision"]["promotion_allowed"] is False
    assert set(result["decision"]["comparisons"]) == {"nyx", "hourly_control"}
    assert all(all(checks.values()) for checks in result["decision"]["comparisons"].values())
    assert result["protocol"]["quantiles_produced"] is False
    assert result["protocol"]["vintages_verified_here"] is False
    assert result["selection"]["test_used_for_selection"] is False
    assert pd.Timestamp(result["selection"]["validation_end"]) == pd.Timestamp(result["selection"]["test_start"]) - pd.Timedelta(days=2)
    train_audit = result["fit_audit"]
    assert (pd.to_datetime(train_audit.fit_last_day) <= pd.to_datetime(train_audit.max_permitted_label_day)).all()
    assert (train_audit.complete_train_rows >= OPTIONS["min_train_rows"]).all()
    assert (train_audit.fit_unique_days <= OPTIONS["window_days"]).all()
    control = result["selection"]["feature_columns"]["hourly_control"]
    assert MEAN in control and SHAPE not in control
    assert SHAPE in result["selection"]["feature_columns"]["intrahour"]
    assert set(result["metrics"].group) >= {"country", "country_hour", "negative", "train_q95"}
    paired = result["paired_deltas"]
    contrast = paired.loc[paired.family.eq("intrahour") & paired.baseline.eq("nyx") & paired.zone.eq("all")].iloc[0]
    p = result["predictions"]
    rows = p.loc[p.stage.eq("test") & p.family.eq("intrahour")]
    assert contrast.n == len(rows)
    assert contrast.bootstrap_repetitions == OPTIONS["bootstrap_repetitions"]
    assert contrast.bootstrap_block_days == 7
    assert contrast.mae_delta_ci_high < 0


def test_unavailable_future_labels_and_features_cannot_change_earlier_fit_or_forecast():
    panel = synthetic_panel()
    baseline = evaluate_variant(panel, **OPTIONS)
    start = pd.Timestamp(baseline["selection"]["test_start"])
    changed = panel.copy()
    dates = local_days(changed)
    # D-1 is the embargo day: neither the first fit nor selection may read its label.
    changed.loc[dates.ge(start - pd.Timedelta(days=1)), "training_actual"] += 10000
    # Features at later refits must not contaminate the first imputer/scaler.
    changed.loc[dates.ge(start + pd.Timedelta(days=7)), [SHAPE, MEAN, "feature_hourly_synthetic_forecast"]] += 1e8
    modified = evaluate_variant(changed, **OPTIONS)
    assert baseline["selection"] == modified["selection"]
    for result in (baseline, modified):
        p = result["predictions"]
        result["first_block"] = p.loc[p.stage.eq("test") & p.delivery_day.lt(start + pd.Timedelta(days=7)),
                                      ["timestamp_utc", "zone", "family", "prediction"]].reset_index(drop=True)
        a = result["fit_audit"]
        result["first_fit"] = a.loc[a.stage.eq("test") & a.refit_day.eq(start.date().isoformat())].reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline["first_block"], modified["first_block"])
    pd.testing.assert_frame_equal(baseline["first_fit"], modified["first_fit"])
    # The chosen imputer is demonstrably fitted only on the declared training window.
    a = baseline["first_fit"].loc[lambda x: x.family.eq("intrahour") & x.zone.eq("FR")].iloc[0]
    past = panel.loc[dates.between(pd.Timestamp(a.window_first_day), pd.Timestamp(a.max_permitted_label_day)) & panel.zone.eq("FR")]
    j = a.feature_columns.index(MEAN)
    assert a.imputation_values[j] == pytest.approx(past[MEAN].median())
    assert a.input_mean[j] == pytest.approx(past[MEAN].mean())


def test_final_test_labels_never_select_alpha_or_recipe():
    panel = synthetic_panel()
    baseline = evaluate_variant(panel, **OPTIONS)
    changed = panel.copy()
    testing = local_days(changed).ge(pd.Timestamp(baseline["selection"]["test_start"]))
    changed.loc[testing, "actual"] = changed.loc[testing, "nyx_q50"]
    modified = evaluate_variant(changed, **OPTIONS)
    assert modified["selection"] == baseline["selection"]
    assert not modified["decision"]["encouraging"]
    pd.testing.assert_series_equal(baseline["predictions"].prediction, modified["predictions"].prediction)
    assert modified["protocol"]["q95_training_thresholds"] == baseline["protocol"]["q95_training_thresholds"]


def test_improving_nyx_without_improving_hourly_control_is_not_encouraging():
    panel = synthetic_panel()
    # Signal is wholly in the existing hourly mean. Quarter-hour shape adds noise.
    panel["actual"] = panel.nyx_q50 + 7 * (panel[MEAN] - 10)
    panel["training_actual"] = panel.actual
    result = evaluate_variant(panel, **OPTIONS)
    paired = result["paired_deltas"]
    control = paired.loc[paired.family.eq("hourly_control") & paired.baseline.eq("nyx") & paired.zone.eq("all")].iloc[0]
    assert control.mae_relative_improvement > .5
    assert not result["decision"]["encouraging"]
    assert not all(result["decision"]["comparisons"]["hourly_control"].values())


def test_global_improvement_cannot_hide_degraded_peak_regimes():
    panel = synthetic_panel()
    days = local_days(panel)
    peaks = days.ge(pd.Timestamp("2026-03-30")) & panel.timestamp_utc.dt.hour.between(18, 21)
    # The learned shape effect remains good elsewhere but overcorrects these peaks.
    panel.loc[peaks, "nyx_q50"] = 100.0
    panel.loc[peaks, SHAPE] = 2.0
    panel.loc[peaks, ["actual", "training_actual"]] = 102.0
    result = evaluate_variant(panel, **OPTIONS)
    assert all(result["decision"]["comparisons"]["nyx"].values())
    assert not result["decision"]["encouraging"]
    assert not result["decision"]["critical_regimes"]["all_sufficient_regimes_within_5pct"]
    checks = result["regime_checks"]
    peak_checks = checks.loc[checks.regime.eq("above_train_q95") & checks.baseline.eq("nyx")]
    assert set(peak_checks.zone) == set(ZONES)
    assert peak_checks.evidence.eq("sufficient").all()
    assert (peak_checks.n >= 30).all() and (peak_checks.distinct_days >= 5).all()
    assert not peak_checks.within_5pct.any()
    assert (peak_checks.mae_relative_degradation > .05).all()
    assert any(":above_train_q95:mae_degradation_over_5pct" in reason for reason in result["decision"]["reasons"])


def test_many_peak_hours_in_only_three_days_are_insufficient_evidence():
    panel = synthetic_panel()
    panel["nyx_q50"] = 45.0
    panel[["actual", "training_actual"]] = 50.0
    days = local_days(panel)
    peaks = days.between(pd.Timestamp("2026-03-30"), pd.Timestamp("2026-04-01")) & panel.timestamp_utc.dt.hour.lt(10)
    panel.loc[peaks, "actual"] = 100.0
    result = evaluate_variant(panel, **OPTIONS)
    checks = result["regime_checks"]
    peak_checks = checks.loc[checks.regime.eq("above_train_q95")]
    assert peak_checks.n.eq(30).all()
    assert peak_checks.distinct_days.eq(3).all()
    assert checks.evidence.eq("insufficient").all()
    assert checks.mae_relative_degradation.isna().all()
    assert checks.within_5pct.isna().all()
    assert checks.candidate_mae.isna().all()
    assert not result["decision"]["encouraging"]
    assert result["decision"]["status"] == "insufficient_regime_evidence"
    assert not result["decision"]["critical_regimes"]["evidence_available"]
    assert len(result["decision"]["critical_regimes"]["insufficient_regimes"]) == 8


def test_missing_current_shape_falls_back_to_nyx_and_blocks_claim():
    panel = synthetic_panel()
    first_test = pd.Timestamp("2026-03-30")  # 14 train + 7 validation + one embargo day
    missing = local_days(panel).between(first_test, first_test + pd.Timedelta(days=1))
    panel.loc[missing, SHAPE] = np.nan
    result = evaluate_variant(panel, **OPTIONS)
    assert result["status"] == "insufficient_data"
    assert not result["decision"]["encouraging"]
    assert result["decision"]["test_fallback_rate"] > .05
    p = result["predictions"]
    bad = p.loc[p.family.eq("intrahour") & p.delivery_day.between(first_test, first_test + pd.Timedelta(days=1))]
    np.testing.assert_array_equal(bad.prediction, bad.nyx_q50)
    assert bad.fallback.all()
    assert bad.fallback_reason.eq("missing_current_features").all()
    control = p.loc[p.family.eq("hourly_control") & p.stage.eq("test")]
    assert not control.fallback.any()
    # All target hours still appear in all three methods, even on missing-feature days.
    counts = p.loc[p.stage.eq("test")].groupby("family").size()
    assert counts.nunique() == 1


@pytest.mark.parametrize("shape_value,reason", [(np.nan, "missing_features"), (0.0, "insufficient_usable_intrahour_features")])
def test_no_usable_shape_never_creates_a_successful_experiment(shape_value, reason):
    panel = synthetic_panel()
    panel[SHAPE] = shape_value
    result = evaluate_variant(panel, **OPTIONS)
    assert result["status"] == "insufficient_data"
    assert result["reason"] == reason
    assert not result["decision"]["encouraging"]
    if np.isnan(shape_value):
        assert result["metrics"].empty and result["paired_deltas"].empty
    else:
        assert not result["decision"]["data_checks"]["shape_varies_in_training"]


def test_imputation_does_not_hide_insufficient_complete_training_rows():
    panel = synthetic_panel()
    # Labels exist, but the first train block has no complete intrahour profiles.
    panel.loc[local_days(panel).lt(pd.Timestamp("2026-03-22")), SHAPE] = np.nan
    result = evaluate_variant(panel, **OPTIONS)
    audit = result["fit_audit"]
    first = audit.loc[audit.family.eq("intrahour") & audit.refit_day.eq("2026-03-22")]
    assert not first.empty
    assert first.complete_train_rows.eq(0).all()
    assert first.status.eq("insufficient_complete_training_rows").all()
    assert not result["decision"]["data_checks"]["complete_training_rows_sufficient"]
    assert result["decision"]["promotion_allowed"] is False


@pytest.mark.parametrize("start,dst_day,physical_hours", [
    ("2026-03-07", "2026-03-29", 23), ("2026-10-03", "2026-10-25", 25),
])
def test_dst_target_hours_are_preserved_and_paired(start, dst_day, physical_hours):
    result = evaluate_variant(synthetic_panel(start), **OPTIONS)
    predictions = result["predictions"]
    day = predictions.loc[predictions.delivery_day.eq(pd.Timestamp(dst_day))]
    assert not day.empty
    assert (day.groupby(["family", "zone"]).size() == physical_hours).all()
    assert not predictions.duplicated(["timestamp_utc", "zone", "family"]).any()
    for family in ("nyx", "hourly_control", "intrahour"):
        p = predictions.loc[predictions.family.eq(family) & predictions.stage.eq("test")]
        row = result["metrics"].loc[lambda f: f.family.eq(family) & f.stage.eq("test") & f.group.eq("overall")].iloc[0]
        error = p.prediction - p.actual
        assert row.n == len(p)
        assert row.mae == pytest.approx(error.abs().mean())
        assert row.rmse == pytest.approx(np.sqrt(np.mean(error ** 2)))


def test_missing_days_or_targets_are_not_compressed_into_a_valid_history():
    assert evaluate_variant(pd.DataFrame(), **OPTIONS)["reason"] == "empty_panel"
    short = evaluate_variant(synthetic_panel(days=12), **OPTIONS)
    assert short["status"] == "insufficient_data" and short["metrics"].empty
    assert short["details"]["available_days"] == 12
    panel = synthetic_panel()
    gap = panel.loc[~local_days(panel).eq(pd.Timestamp("2026-03-25"))]
    assert evaluate_variant(gap, **OPTIONS)["reason"] == "missing_delivery_days"
    assert evaluate_variant(panel.iloc[1:], **OPTIONS)["reason"] == "incomplete_target_grid"
    duplicated = pd.concat([panel, panel.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate zone/timestamp"):
        evaluate_variant(duplicated, **OPTIONS)
    panel["timestamp_utc"] = panel.timestamp_utc.dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone aware"):
        evaluate_variant(panel, **OPTIONS)
