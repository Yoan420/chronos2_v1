"""Synthetic causal checks plus an optional local historical parity smoke test."""

from __future__ import annotations

from datetime import date, timedelta
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_negative_probability_cpu as model


ROOT = Path(__file__).resolve().parents[1]


def sample(zone: str = "FR", origin: date = date(2026, 9, 1), days: int = 7):
    first = origin - timedelta(days=365)
    stop = origin + timedelta(days=days)
    grid = model.physical_grid(first, stop, zone)
    rng = np.random.default_rng(527)
    signal = rng.normal(size=len(grid))
    second = rng.normal(size=len(grid))
    features = pd.DataFrame(
        np.zeros((len(grid), model.FEATURE_COUNT)),
        index=grid,
        columns=[f"x{n}" for n in range(model.FEATURE_COUNT)],
    )
    features["x0"], features["x1"] = signal, second
    actual = pd.Series(np.where(signal + 0.3 * second > 1.1, -20.0, 30.0), index=grid, name="actual")
    history_grid = model.physical_grid(first, origin, zone)
    future_grid = model.physical_grid(origin, stop, zone)
    return features, actual, history_grid, future_grid


def test_real_cpu_recipe_and_past_only_split() -> None:
    pytest.importorskip("catboost")
    pytest.importorskip("sklearn")
    features, actual, history, future = sample()
    result = model.fit_predict_block(
        features.loc[history], actual.loc[history], features.loc[future],
        zone="FR", origin_day="2026-09-01", stop_day="2026-09-08", threads=1,
    )
    audit = result.audit
    assert audit["models_fitted"] == 1 and audit["tree_count"] == 120
    assert audit["calibration_status"] == "platt"
    assert audit["parameters"] == model.catboost_parameters(1)
    assert audit["fit_hours"] + audit["calibration_hours"] == len(history)
    assert audit["calibration_first_day"] == "2026-08-04"
    assert audit["fit_negative_hours"] > 0 and audit["calibration_negative_hours"] >= 10
    assert audit["forecast_labels_used"] is False
    assert result.probabilities.index.equals(future)
    assert result.probabilities.p_negative.between(0.0, 1.0).all()
    assert (result.probabilities.is_negative_predicted == (result.probabilities.p_negative >= 0.5)).all()
    with pytest.raises(ValueError, match="historical prices"):
        model.fit_predict_block(
            features.loc[history], actual, features.loc[future],
            zone="FR", origin_day="2026-09-01", stop_day="2026-09-08",
        )


@pytest.mark.parametrize("zone", model.ZONES)
def test_dst_and_zero_price_is_not_a_negative_event(zone: str) -> None:
    origin = date(2025, 10, 25)
    features, actual, history, future = sample(zone, origin, days=3)
    actual.loc[history] = 0.0
    result = model.fit_predict_block(
        features.loc[history], actual.loc[history], features.loc[future],
        zone=zone, origin_day=origin, stop_day=origin + timedelta(days=3),
    )
    assert len(future) == 73
    assert result.audit["models_fitted"] == 0
    assert result.audit["fallback_reason"] == "one_class"
    assert (result.probabilities.p_negative == 0.0).all()
    assert not result.probabilities.is_negative_predicted.any()
    repeated = result.probabilities.loc["2025-10-26T00Z":"2025-10-26T01Z", "timestamp_local"].tolist()
    assert repeated[0].endswith("+02:00") and repeated[1].endswith("+01:00")


def test_weekly_evaluation_uses_each_origins_past_only() -> None:
    origin = date(2026, 9, 1)
    features, actual, _, _ = sample(origin=origin, days=14)
    features.loc[:, :] = 1.0  # frequency fallback avoids a fit while testing causality
    first = model.evaluate_weekly(features, actual, zone="FR", first_origin_day=origin, stop_day=origin + timedelta(days=14))
    changed = actual.copy()
    first_week = model.physical_grid(origin, origin + timedelta(days=7), "FR")
    changed.loc[first_week] = -30.0
    second = model.evaluate_weekly(features, changed, zone="FR", first_origin_day=origin, stop_day=origin + timedelta(days=14))
    assert first.metrics["origins"] == second.metrics["origins"] == 2
    assert first.metrics["hours"] == second.metrics["hours"] == 336
    assert first.metrics["models_fitted"] == 0
    np.testing.assert_array_equal(first.probabilities.loc[first_week, "p_negative"], second.probabilities.loc[first_week, "p_negative"])
    second_week = model.physical_grid(origin + timedelta(days=7), origin + timedelta(days=14), "FR")
    assert not np.array_equal(first.probabilities.loc[second_week, "p_negative"], second.probabilities.loc[second_week, "p_negative"])
    assert first.metrics["brier"] != second.metrics["brier"]


def test_feature_schema_and_future_grid_are_strict() -> None:
    features, actual, history, future = sample()
    invalid = features.loc[future].rename(columns={"x0": "storm_price"})
    with pytest.raises(ValueError, match="non-label"):
        model.fit_predict_block(
            features.loc[history], actual.loc[history], invalid,
            zone="FR", origin_day="2026-09-01", stop_day="2026-09-08",
        )
    shifted = features.loc[future].copy()
    shifted.index = shifted.index + pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="UTC grid"):
        model.fit_predict_block(
            features.loc[history], actual.loc[history], shifted,
            zone="FR", origin_day="2026-09-01", stop_day="2026-09-08",
        )


def test_optional_historical_fr_checkpoint_parity() -> None:
    pytest.importorskip("catboost")
    pytest.importorskip("sklearn")
    pytest.importorskip("pyarrow")
    archive = ROOT / "runs/experiments/nyx_improvement_to20260923"
    plan_path = archive / "negative_prices_v1/plan.json"
    feature_path = archive / "feature_sets/pooled_jao_refresh_v1/compact/features_FR.parquet"
    baseline_path = ROOT / "runs/experiments/nyx_local_365_to20260923/baseline/FR.parquet"
    checkpoint_path = archive / "negative_prices_v1/checkpoints/FR/2026-09-23/receipt.json"
    if not all(path.is_file() for path in (plan_path, feature_path, baseline_path, checkpoint_path)):
        pytest.skip("Historical research archive is absent from this clone")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    runtime = plan["runtime"]
    for package, distribution in (("catboost", "catboost"), ("numpy", "numpy"), ("pandas", "pandas"), ("scikit-learn", "scikit-learn")):
        if version(distribution) != runtime[package]:
            pytest.skip(f"{package} version differs from archived fit")
    receipt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    historical_path = checkpoint_path.parent / receipt["attempt_directory"] / "predictions.parquet"
    if not historical_path.is_file():
        pytest.skip("Historical FR prediction checkpoint is absent")
    features = pd.read_parquet(feature_path)
    actual = pd.read_parquet(baseline_path)["actual"]
    history = model.physical_grid("2025-09-23", "2026-09-23", "FR")
    future = model.physical_grid("2026-09-23", "2026-09-24", "FR")
    result = model.fit_predict_block(
        features.loc[history], actual.loc[history], features.loc[future],
        zone="FR", origin_day="2026-09-23", stop_day="2026-09-24", threads=2,
    )
    archived = pd.read_parquet(historical_path)
    assert result.probabilities.index.equals(archived.index)
    np.testing.assert_array_equal(result.probabilities.p_negative_raw, archived.p_raw)
    np.testing.assert_array_equal(result.probabilities.p_negative, archived.p_negative)
    annual_path = archive / "negative_prices_v1/annual/FR.parquet"
    metrics_path = archive / "negative_prices_v1/metrics.json"
    if annual_path.is_file() and metrics_path.is_file():
        annual = pd.read_parquet(annual_path)
        measured = model.probability_metrics(actual.loc[annual.index], annual.p_negative)
        historical = json.loads(metrics_path.read_text(encoding="utf-8"))["countries"]["FR"]["models"]["model"]
        assert measured["hours"] == historical["n"] == 8760
        assert measured["negative_hours"] == historical["negative_hours"] == 580
        for key in ("brier", "log_loss", "average_precision"):
            assert measured[key] == pytest.approx(historical[key], abs=1e-15)
