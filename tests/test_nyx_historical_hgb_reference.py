from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_historical_hgb_reference as hgb


class FixedEstimator:
    def __init__(self, **parameters):
        self.parameters = parameters

    def fit(self, matrix, target):
        self.columns = tuple(matrix.columns)
        self.target = target.copy()
        self.training_index = matrix.index.copy()
        return self

    def predict(self, matrix):
        assert tuple(matrix.columns) == self.columns
        return np.repeat(100., len(matrix))


def sample_inputs():
    index = hgb.civil_grid("2024-01-01", "2024-05-01", "Europe/Paris")
    features = pd.DataFrame({"x": np.sin(np.arange(len(index)) / 24.),
                             "x__available": True}, index=index)
    actual = pd.Series(np.arange(len(index), dtype=float) / 10., index=index, name="actual")
    nyx = pd.Series(5., index=index, name="nyx__q50")
    return features, actual, nyx


def test_original_parameters_window_residual_clip_and_future_labels(monkeypatch):
    monkeypatch.setattr(hgb, "HistGradientBoostingRegressor", FixedEstimator)
    features, actual, nyx = sample_inputs()
    origin, stop = "2024-04-15", "2024-04-17"
    fitted, point, audit = hgb.fit_hgb_block(
        features, actual, nyx, zone="FR", variant="hist_residual_400",
        origin_day=origin, stop_day=stop, expected_columns=tuple(features.columns),
        initial_training_day="2024-01-01")
    assert fitted.model.parameters == hgb.PARAMETERS
    assert audit["training_days"] == 105
    assert len(fitted.model.training_index) == 105 * 24 - 1  # spring DST
    assert fitted.model.training_index.max() < point.index.min()
    assert (point.point == 65.).all()  # NYX 5 + capped residual 60
    changed = actual.copy()
    changed.loc[point.index] = -1e9
    _, second, second_audit = hgb.fit_hgb_block(
        features, changed, nyx, zone="FR", variant="hist_residual_400",
        origin_day=origin, stop_day=stop, expected_columns=tuple(features.columns),
        initial_training_day="2024-01-01")
    pd.testing.assert_frame_equal(point, second, check_exact=True)
    assert audit["training_target_sha256"] == second_audit["training_target_sha256"]


def test_absolute_variant_schema_and_saved_prediction(monkeypatch):
    monkeypatch.setattr(hgb, "HistGradientBoostingRegressor", FixedEstimator)
    features, actual, nyx = sample_inputs()
    fitted, point, _ = hgb.fit_hgb_block(
        features, actual, nyx, zone="FR", variant="hist_absolute_400",
        origin_day="2024-04-15", stop_day="2024-04-17",
        expected_columns=tuple(features.columns), initial_training_day="2024-01-01")
    assert (point.point == 100.).all()
    replay = hgb.predict_saved_block(fitted, features.loc[point.index], nyx.loc[point.index])
    pd.testing.assert_frame_equal(point, replay, check_exact=True)
    with pytest.raises(ValueError, match="order/schema"):
        hgb.predict_saved_block(fitted, features.loc[point.index, ::-1], nyx.loc[point.index])
    with pytest.raises(ValueError, match="order/schema"):
        hgb.fit_hgb_block(features.iloc[:, ::-1], actual, nyx, zone="FR",
                          variant="hist_absolute_400", origin_day="2024-04-15",
                          stop_day="2024-04-17", expected_columns=tuple(features.columns),
                          initial_training_day="2024-01-01")


def test_archived_schemas_and_reference_arithmetic():
    for zone in hgb.TIMEZONES:
        base = hgb.archived_columns(zone, "hist_residual_400")
        assert base == hgb.archived_columns(zone, "hist_absolute_400")
        augmented = hgb.archived_columns(zone, "augmented_hist_residual_400")
        assert len(base) == 292 and len(augmented) == 334
        assert augmented[:292] == base
    index = hgb.civil_grid("2024-04-15", "2024-04-16", "Europe/Paris")
    s = lambda x: pd.Series(float(x), index=index)
    result = hgb.compose_warmup_reference(s(10), s(20), s(30), s(50), s(100))
    assert (result.three_hgb_mean == 20.).all()
    assert (result["three_hgb_mean__clip20__w0p75"] == 32.5).all()


def test_one_archived_checkpoint_when_local_archive_exists():
    """Optional byte-level replay of one complete historical weekly fit."""
    root = Path(__file__).resolve().parents[1] / "runs" / "experiments"
    baseline = root / "nyx_local_365_to20260923" / "baseline" / "FR.parquet"
    model_root = root / "nyx_improvement_to20260923" / "price_models"
    feature_path = model_root / "features_FR.parquet"
    checkpoint = model_root / "hist_residual_400" / "FR_2026-09-23.parquet"
    if not all(p.exists() for p in (baseline, feature_path, checkpoint)):
        pytest.skip("Archived local HGB matrices are not shipped in Git")
    features = pd.read_parquet(feature_path)
    observed = pd.read_parquet(baseline)
    _, predicted, audit = hgb.fit_hgb_block(
        features, observed.actual, observed.nyx__q50,
        zone="FR", variant="hist_residual_400",
        origin_day="2026-09-23", stop_day="2026-09-24")
    expected = pd.read_parquet(checkpoint)
    pd.testing.assert_frame_equal(predicted, expected, check_exact=True, check_freq=False)
    assert audit["training_days"] == 365
