"""Chronological safety and identity-mode tests on small deterministic panels."""
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest

from nyx_solar_ramp.features import build_features
from nyx_solar_ramp.policy import _alert_threshold, _blank, feature_sets, govern, prepare_panel, run_policy


def config():
    # Deliberately small TEST-only history/trees, not the runner's deployment
    # contract. Enough physical rows remain in both chronological holdouts.
    return {
        "enabled": True, "timezone": "Europe/Paris", "training_window_days": 365,
        "minimum_training_days": 15, "calibration_days": 6, "refit_days": 7,
        "final_days": 10, "selection_days": 10, "business_spike": 300.,
        "statistical_quantile": .99, "price_ramp_threshold": 100.,
        "false_alert_budget": .01, "label_delay_days": 2, "max_iter": 2,
        "max_leaf_nodes": 7, "min_samples_leaf": 10, "l2_regularization": 10.,
        "learning_rate": .06, "correction_clip": 300., "governance_days": 90,
        "governance_min_days": 7, "governance_min_changed_days": 2,
        "governance_min_alert_rows": 2, "candidate_weights": [0., .25, .5, 1.],
        "mae_tolerance": 0., "block_days": 7, "seed": 1729, "threads": 1,
    }


def prepared(days=36):
    fixture = runpy.run_path(str(Path(__file__).with_name("test_nyx_solar_ramp_features.py")))
    source = fixture["panel"](days=days)
    hour = source.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour
    source["actual"] = np.where(hour.isin([18, 19]), 500., 100.)
    source["forecast"], source["q10"], source["q90"] = 100., 50., 550.
    source["sample"] = "evaluation"
    return build_features(source, include_baseline=True)[:2]


def test_future_labels_cannot_change_earlier_predictions_or_governance():
    data, groups = prepared()
    cfg = config()
    original = data.copy(deep=True)
    before, folds, _, _ = run_policy(data, groups, cfg)
    boundary = pd.Timestamp("2026-01-31", tz="Europe/Paris").tz_convert("UTC")
    changed = data.copy(deep=True)
    changed.loc[changed.timestamp_utc.ge(boundary), "actual"] = 10000.
    after, _, _, _ = run_policy(changed, groups, cfg)
    earlier = before.timestamp_utc.lt(boundary)
    columns = ["variant", "zone", "timestamp_utc", "risk_probability", "alert_threshold", "alert",
               "candidate_forecast", "candidate_q10", "candidate_q90", "selected_weight", "applied_correction"]
    pd.testing.assert_frame_equal(before.loc[earlier, columns], after.loc[earlier, columns])
    pd.testing.assert_frame_equal(data, original, check_exact=True)
    assert folds
    for fold in folds:
        assert pd.Timestamp(fold["maximum_label_time"]) <= pd.Timestamp(fold["fit_origin_utc"])
        assert fold["train_end"] < fold["calibration_start"]
        assert fold["calibration_end"] < fold["threshold_start"]
        assert fold["threshold_end"] < fold["predict_first_day"]


def test_allowlists_separate_solar_ramp_value_from_non_solar_ramp_controls():
    _, groups = prepared(1)
    sets = feature_sets(groups)
    assert all("solar_drop" in name for name in set(sets["ramps"])-set(sets["regional"]))
    assert len(set(sets["ramps"])-set(sets["regional"])) == 4
    assert "solarx_local_residual_ramp_1h_gwph" in sets["control"]
    assert "solarx_peer_wind_ramp_3h_gwph" in sets["control"]
    assert not any(any(token in name for token in ("actual", "label", "storm")) for names in sets.values() for name in names)


@pytest.mark.parametrize("probabilities", [np.full(1000, .2), np.repeat([.1, .2, .8, .9], 250), np.linspace(0, 1, 1000)])
def test_alert_budget_with_ties_requires_strict_greater_than(probabilities):
    threshold = _alert_threshold(probabilities, np.zeros(len(probabilities)), .01)
    assert 0 <= threshold <= 1
    assert np.mean(probabilities > threshold) <= .01


def test_insufficient_negative_calibration_never_emits_alerts_or_invalid_threshold():
    threshold = _alert_threshold(np.array([.1, .9]), np.array([0., 1.]), .01)
    assert threshold == 1.
    assert not (np.array([0., .5, 1.]) > threshold).any()


def test_disabled_mode_preserves_baseline_exactly_for_every_variant():
    data, groups = prepared(4)
    cfg = config()
    cfg["enabled"] = False
    out, folds, governance, models = run_policy(data, groups, cfg)
    assert not folds and not models and governance
    for _, variant in out.groupby("variant"):
        np.testing.assert_array_equal(variant.candidate_forecast, variant.forecast)
        np.testing.assert_array_equal(variant.candidate_q10, variant.q10)
        np.testing.assert_array_equal(variant.candidate_q90, variant.q90)
        assert variant.applied_correction.eq(0).all()
        assert variant.selected_weight.eq(0).all()
        assert not variant.alert.any() and not variant.expert_ready.any()


def test_governor_excludes_late_labels_even_with_apparently_profitable_prior_proposals():
    data, _ = prepared(12)
    cfg = config()
    hour = data.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour
    data["actual"] = np.where(hour.isin([18, 19]), 400., 140.)
    proposal = _blank(prepare_panel(data, cfg), "interactions")
    proposal["expert_ready"], proposal["alert"] = True, True
    proposal["bounded_correction"] = 20.
    proposal["risk_probability"], proposal["alert_threshold"] = .9, .8
    good, _ = govern(proposal, cfg)
    final = good.local_day.eq(good.local_day.max())
    assert good.loc[final, "applied_correction"].gt(0).all()
    proposal["training_label_available_at_utc"] = proposal.forecast_origin_utc.max()+pd.Timedelta(days=10)
    late, records = govern(proposal, cfg)
    np.testing.assert_array_equal(late.candidate_forecast, late.forecast)
    np.testing.assert_array_equal(late.candidate_q10, late.q10)
    np.testing.assert_array_equal(late.candidate_q90, late.q90)
    assert late.selected_weight.eq(0).all()
    assert all(record["past_oos_rows"] == 0 for record in records)


def test_label_delay_is_after_civil_delivery_end_and_honours_later_upstream_dates():
    data, _ = prepared(1)
    cfg = config()
    out = prepare_panel(data, cfg)
    expected = pd.Timestamp("2026-01-04", tz="Europe/Paris").tz_convert("UTC")
    assert out.training_label_available_at_utc.eq(expected).all()
    data["label_available_at_utc"] = expected+pd.Timedelta(days=1)
    later = prepare_panel(data, cfg)
    assert later.training_label_available_at_utc.eq(expected+pd.Timedelta(days=1)).all()
