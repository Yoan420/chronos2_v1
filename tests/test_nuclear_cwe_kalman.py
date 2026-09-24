from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import kalman_residual as kalman
from chronos2_hourly import nuclear_cwe_kalman as bridge
from chronos2_hourly.nuclear_cwe_forecast import (
    CWE_NUCLEAR_ALIASES,
    _CWEKalmanBuilder,
    nuclear_cwe_kalman_covariate_config,
)


@pytest.mark.parametrize("column", ["timestamp", "delivery_start_utc"])
def test_bridge_forwards_covariates_and_options_without_mutating_inputs(monkeypatch, column):
    original = pd.DataFrame({
        column: pd.date_range("2026-09-10T22:00:00Z", periods=24, freq="h"),
        "be_nuclear_available_gw": np.linspace(2.0, 4.0, 24),
    })
    before = original.copy(deep=True)
    statistics, forecast, config = object(), object(), object()
    expected = object()
    captured = {}

    def delegate(**kwargs):
        captured.update(kwargs)
        kwargs["covariates"].loc[0, "be_nuclear_available_gw"] = -999.0
        return expected

    monkeypatch.setattr(kalman, "build_operational_kalman_view", delegate)
    result = bridge.build_nuclear_cwe_kalman_view(
        covariates=original, statistics=statistics, source_forecast=forecast,
        config=config, timezone="Europe/Paris", delivery_day="2026-09-11",
        rolling_refit_workers=4, training_lookback_days=365,
    )

    assert result is expected
    assert captured["statistics"] is statistics
    assert captured["source_forecast"] is forecast
    assert captured["config"] is config
    assert captured["timezone"] == "Europe/Paris"
    assert captured["delivery_day"] == "2026-09-11"
    assert captured["rolling_refit_workers"] == 4
    assert captured["training_lookback_days"] == 365
    assert tuple(captured["covariates"].columns) == ("timestamp", "be_nuclear_available_gw")
    pd.testing.assert_series_equal(captured["covariates"].timestamp, before[column], check_names=False)
    pd.testing.assert_frame_equal(original, before)


@pytest.mark.parametrize("fault", ["missing", "both", "different_both", "duplicate_timestamp", "not_frame"])
def test_bridge_refuses_ambiguous_or_missing_timestamp_before_delegate(monkeypatch, fault):
    index = pd.date_range("2026-09-10T22:00:00Z", periods=24, freq="h")
    covariates = pd.DataFrame({"delivery_start_utc": index, "power": 4.0})
    if fault == "missing":
        covariates = covariates.drop(columns="delivery_start_utc")
    elif fault in {"both", "different_both"}:
        covariates["timestamp"] = index + pd.Timedelta(hours=int(fault == "different_both"))
    elif fault == "duplicate_timestamp":
        covariates = pd.concat([covariates, covariates[["delivery_start_utc"]]], axis=1)
    elif fault == "not_frame":
        covariates = {"delivery_start_utc": index}

    def forbidden(**kwargs):
        pytest.fail("An invalid timestamp contract reached the operational Kalman delegate")

    monkeypatch.setattr(kalman, "build_operational_kalman_view", forbidden)
    with pytest.raises((ValueError, TypeError)):
        bridge.build_nuclear_cwe_kalman_view(covariates=covariates)


@pytest.mark.parametrize("cache_argument", [None, "relative", "absolute", "extended"])
def test_bridge_preserves_cache_location_and_uses_windows_extended_paths(monkeypatch, tmp_path, cache_argument):
    if cache_argument == "extended" and os.name != "nt":
        pytest.skip("Windows extended-length path spelling")
    target = tmp_path / "isolated_cache" / ("a" * 70)
    if cache_argument is None:
        supplied = None
    elif cache_argument == "relative":
        monkeypatch.chdir(tmp_path)
        supplied = target.relative_to(tmp_path)
    elif cache_argument == "extended":
        supplied = Path("\\\\?\\" + str(target))
    else:
        supplied = target
    captured = {}
    monkeypatch.setattr(kalman, "build_operational_kalman_view", lambda **kwargs: captured.update(kwargs))
    bridge.build_nuclear_cwe_kalman_view(
        covariates=pd.DataFrame({"timestamp": pd.date_range("2026-09-10T22:00:00Z", periods=24, freq="h")}),
        rolling_refit_cache_dir=supplied,
    )
    result = captured["rolling_refit_cache_dir"]
    if supplied is None:
        assert result is None
    else:
        spelling = str(result)
        if os.name == "nt":
            assert spelling.startswith("\\\\?\\")
            assert not spelling.startswith("\\\\?\\\\\\?\\")
            spelling = spelling[4:]
        assert Path(spelling).resolve() == target.resolve()
    assert not target.exists()


def _operational_inputs(delivery_day):
    day = pd.Timestamp(delivery_day, tz="Europe/Paris")
    index = pd.date_range(day - pd.DateOffset(days=730), day + pd.DateOffset(days=1),
                          freq="h", inclusive="left").tz_convert("UTC")
    base = 60.0 + 5.0 * np.sin(np.arange(len(index)) * np.pi / 12.0)
    frame = pd.DataFrame({
        "delivery_start_utc": index, "actual": base + 1.0,
        "chronos2__q50": base - 2.0, "residual_correction": 2.0,
        "residual_corrected__q10": base - 8.0,
        "residual_corrected__q50": base,
        "residual_corrected__q90": base + 9.0,
    })
    future = index >= day.tz_convert("UTC")
    statistics = frame.loc[~future].copy()
    forecast = frame.loc[future].drop(columns="actual").copy()
    covariates = pd.DataFrame({"delivery_start_utc": index})
    columns = (*kalman.REQUIRED_MARKET_COVARIATES, *CWE_NUCLEAR_ALIASES)
    for position, column in enumerate(columns):
        covariates[column] = (4.0 if column.startswith("be_nuclear") else
                              0.48 if column.startswith("nl_nuclear") else 20.0 + position)
    return statistics, forecast, covariates


@pytest.mark.parametrize(("delivery_day", "expected_hours"), [
    ("2026-09-11", 24), ("2026-03-29", 23), ("2026-10-25", 25),
])
def test_bridge_reaches_real_operational_validation_and_preserves_physical_day(monkeypatch, delivery_day, expected_hours):
    statistics, forecast, covariates = _operational_inputs(delivery_day)
    before = covariates.copy(deep=True)
    captured = {}

    def fake_replay(history, *, future_upstream, covariates, future_covariates, output_model,
                    upstream_model, **kwargs):
        captured.update(history=history.copy(), covariates=covariates.copy(), **kwargs)
        assert future_covariates is covariates

        def predictions(source):
            index = pd.DatetimeIndex(source.delivery_start_utc)
            return pd.DataFrame({f"{output_model}__{q}": source[f"{upstream_model}__{q}"].to_numpy()
                                 for q in ("q10", "q50", "q90")}, index=index)

        historical, future = predictions(history), predictions(future_upstream)
        return kalman.KalmanReplayResult(
            predictions=historical, candidate_predictions=pd.DataFrame(index=historical.index),
            daily_audit=pd.DataFrame(), state_audit=pd.DataFrame(), audit={},
            future_predictions=future, future_candidate_predictions=pd.DataFrame(index=future.index),
            future_audit={},
        )

    monkeypatch.setattr(kalman, "replay_kalman_overlay", fake_replay)
    kwargs = dict(statistics=statistics, source_forecast=forecast, timezone="Europe/Paris",
                  delivery_day=delivery_day, covariate_config=nuclear_cwe_kalman_covariate_config(),
                  training_lookback_days=365, rolling_refit_workers=4)
    with pytest.raises(kalman.KalmanResidualError, match="timestamp est absent"):
        kalman.build_operational_kalman_view(covariates=covariates, **kwargs)
    context = covariates.set_index("delivery_start_utc").rename(columns={
        name: f"known_{name}_oracle" for name in CWE_NUCLEAR_ALIASES
    })
    hook = _CWEKalmanBuilder(context, bridge.build_nuclear_cwe_kalman_view, "test-schema-bridge")
    view = hook(covariates=covariates.rename(columns={"delivery_start_utc": "timestamp"}), **kwargs)

    assert len(view.forecast) == expected_hours
    assert view.forecast.delivery_start_utc.equals(forecast.delivery_start_utc.reset_index(drop=True))
    assert "timestamp" in captured["covariates"]
    assert "delivery_start_utc" not in captured["covariates"]
    assert "delivery_start_utc" in hook.used_covariates
    assert set(CWE_NUCLEAR_ALIASES).issubset(captured["covariates"])
    assert captured["training_lookback_days"] == 365
    assert captured["rolling_refit_workers"] == 4
    day_start = pd.Timestamp(delivery_day, tz="Europe/Paris").tz_convert("UTC")
    assert captured["history"].delivery_start_utc.max() < day_start
    pd.testing.assert_frame_equal(covariates, before)
    for q in ("q10", "q50", "q90"):
        np.testing.assert_array_equal(view.forecast[f"residual_kalman__{q}"],
                                      forecast[f"residual_corrected__{q}"])
