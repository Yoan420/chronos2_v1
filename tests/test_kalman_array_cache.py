"""Array lookup preserves the previous pandas path's exact rolling results."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import kalman_residual as kalman
from chronos2_hourly.kalman_covariates import BASE_RESIDUAL_LOAD_COVARIATES


def _pandas_market_values(self, timestamp):
    # Reference expression used before the optimization; all update arithmetic
    # remains shared so this isolates ordering/dtype/index lookup differences.
    return self.market.loc[timestamp, list(self.market_feature_columns)].to_numpy(dtype=float)


@pytest.mark.parametrize("kind", list(kalman.EXOGENOUS_FILTER_GROUPS))
def test_cached_market_rows_preserve_exact_column_order_and_dst_hours(kind):
    index = pd.date_range("2025-10-26T00:00Z", periods=4, freq="h")
    market = pd.DataFrame({"a": np.array([0.1, 1.2, 2.3, 3.4], dtype=np.float32),
                           "b": [-7.125, 8.0, 9.33, -0.0]}, index=index)
    market = market.iloc[[2, 0, 3, 1]]  # Timestamp lookup must not assume row positions.
    candidate = kalman._Candidate(kind=kind, config=kalman.KalmanResidualConfig(),
                                   observation_variance=2., market=market,
                                   market_feature_columns=("b", "a"))
    for timestamp in index:
        expected = _pandas_market_values(candidate, timestamp)
        actual = candidate._rolling_market_values(timestamp)
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("kind", list(kalman.EXOGENOUS_FILTER_GROUPS))
def test_hourly_states_and_clipping_are_bitwise_identical(kind):
    index = pd.date_range("2025-10-26T00:00Z", periods=5, freq="h")
    market = pd.DataFrame({"a": [0., 1., -1., 2., -3.], "b": [5., -8., 1., -2., 0.]}, index=index)
    options = dict(kind=kind, config=kalman.KalmanResidualConfig(innovation_clip_sigma=1.),
                   observation_variance=2., market=market, market_feature_columns=("b", "a"))
    optimized, reference = kalman._Candidate(**options), kalman._Candidate(**options)
    reference._rolling_market_values = lambda stamp: _pandas_market_values(reference, stamp)
    rows = pd.DataFrame({"_hour_sin": np.sin(np.arange(5)), "_hour_cos": np.cos(np.arange(5)),
                         "_upstream_error": [0., 10000., -10000., .1, -2.]}, index=index)
    optimized.transition_day(); reference.transition_day()
    for _, row in rows.iterrows():
        assert optimized.update_hour_rolling(row) == reference.update_hour_rolling(row)
        assert optimized.mean.tobytes() == reference.mean.tobytes()
        assert optimized.covariance.tobytes() == reference.covariance.tobytes()
        assert optimized.innovation_clips == reference.innovation_clips
        assert optimized.covariance_repairs == reference.covariance_repairs


def test_full_rolling_replay_and_governance_match_previous_pandas_path(monkeypatch):
    index = pd.date_range("2025-10-20", "2025-10-28", freq="h", tz="Europe/Paris", inclusive="left").tz_convert("UTC")
    t = np.arange(len(index), dtype=float)
    base = 60 + 5 * np.sin(t / 24)
    history = pd.DataFrame({"delivery_start_utc": index, "actual": base + 4 * np.cos(t / 48),
                            "residual_corrected__q10": base - 20, "residual_corrected__q50": base,
                            "residual_corrected__q90": base + 20, "chronos2__q50": base - 3,
                            "residual_correction": 3.})
    covariates = pd.DataFrame({"timestamp": index, **{
        alias: 10 + i + np.sin(t / (i + 15)) for i, alias in enumerate(BASE_RESIDUAL_LOAD_COVARIATES)}})
    config = kalman.KalmanResidualConfig(governance_minimum_days=2, governance_lookback_days=3,
        candidate_kinds=("linear_bias", "linear_harmonic", "linear_market", "linear_scale", "ukf_scale"))
    options = dict(timezone="Europe/Paris", evaluation_start_day="2025-10-24", training_lookback_days=4,
                   config=config, covariates=covariates, rolling_refit_workers=1)
    optimized = kalman.replay_kalman_overlay(history, **options)
    monkeypatch.setattr(kalman._Candidate, "_rolling_market_values", _pandas_market_values)
    reference = kalman.replay_kalman_overlay(history, **options)
    for name in ("predictions", "candidate_predictions", "daily_audit", "state_audit"):
        pd.testing.assert_frame_equal(getattr(optimized, name), getattr(reference, name), check_exact=True)
    assert optimized.audit == reference.audit
