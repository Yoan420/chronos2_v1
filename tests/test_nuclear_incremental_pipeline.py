"""Exercise real cache keys across deliveries without running the neural model."""
from copy import deepcopy
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import chronos2_hourly.nuclear_forecast as engine
from chronos2_modular import forecasting
from test_nuclear_forecast import _prepared, FakeResidual


class CountingResidual(FakeResidual):
    fits = 0

    def fit(self, *args):
        type(self).fits += 1
        return super().fit(*args)


def _forecaster():
    counts = {"history_days": 0, "future_days": 0}

    def historical(**kwargs):
        data = kwargs["data"]
        pieces = []
        for origin in kwargs["origins"]:
            counts["history_days"] += 1
            idx = data.target.index[origin:origin + kwargs["horizon"]]
            value = data.model_context_covariates.loc[idx, engine.NUCLEAR_KNOWN_COLUMN].astype("float32") + np.float32(.137)
            pieces.append(pd.DataFrame({"timestamp": idx, "q10": value.values,
                "q50": value.values + 5, "q90": value.values + 10,
                "actual": data.target.loc[idx].to_numpy(dtype="float32")}))
        return pd.concat(pieces, ignore_index=True)

    def live(**kwargs):
        counts["future_days"] += 1
        data = kwargs["data"]
        idx = pd.date_range(data.target.index[-1] + pd.Timedelta(hours=1),
                            periods=kwargs["horizon"], freq="h")
        value = data.model_context_covariates.loc[idx, engine.NUCLEAR_KNOWN_COLUMN].astype("float32") + np.float32(.137)
        return pd.DataFrame({"timestamp": idx, "q10": value.values,
                             "q50": value.values + 5, "q90": value.values + 10})

    return SimpleNamespace(__file__=__file__, run_backtest_variant=historical,
        run_live_forecast_variant=live, build_origin_frames=forecasting.build_origin_frames,
        build_live_frames=forecasting.build_live_frames,
        prepare_chronos_frame=forecasting.prepare_chronos_frame), counts


def _inputs(tmp_path, delivery):
    config, data, _ = _prepared(tmp_path, delivery)
    # Use absolute timestamps so advancing the delivery does not revise history.
    data.target[:] = 55 + np.sin(data.target.index.asi8 / 3.6e12 / 20)
    data.frequency = "h"
    config["model"]["revision"] = "a" * 40
    return config, data


def test_next_delivery_reuses_days_and_matches_full_current_forecast(tmp_path, monkeypatch):
    first_day = date(2026, 9, 8)
    module, counts = _forecaster()
    captures = []

    def kalman(**kwargs):
        captures.append(kwargs)
        # Inspect the exact data supplied to the existing independent rolling
        # Kalman engine; its daily cache/fit behaviour has its own tests.
        return SimpleNamespace(forecast=kwargs["source_forecast"])

    def run(delivery, mode, name):
        config, data = _inputs(tmp_path, delivery)
        config["nuclear_experiment"] = {"mode": mode,
            "incremental_cache_dir": str(tmp_path / "shared")}
        return engine.run_nuclear_forecast(config=config, data=data, zone="FR",
            delivery_day=delivery, workdir=tmp_path / name, device="cpu", threads=1,
            runtime_factory=lambda *args: object(), forecasting_module=module,
            residual_factory=CountingResidual, kalman_builder=kalman)

    CountingResidual.fits = 0
    # Interrupt after neural checkpoints, before the residual result exists.
    # Restart must recover exact noninteger float32 values, not rounded CSVs.
    with monkeypatch.context() as interrupted:
        def stop_before_residual(**kwargs):
            raise RuntimeError("simulated interruption")
        interrupted.setattr(engine, "causal_residual_replay", stop_before_residual)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            run(first_day, "incremental", "first")
    assert CountingResidual.fits == 0
    assert counts == {"history_days": 730, "future_days": 1}
    first = run(first_day, "incremental", "first")
    assert first.raw_history.q50.dtype == np.dtype("float32")
    initial_fits = CountingResidual.fits
    assert counts == {"history_days": 730, "future_days": 1}
    second = run(first_day + timedelta(days=1), "incremental", "second")
    assert counts == {"history_days": 730, "future_days": 2}
    assert CountingResidual.fits - initial_fits == 1
    assert second.audit["daily_residual_cache"]["misses"] == 1
    assert second.audit["daily_residual_cache"]["hits"] >= 728
    assert captures[0]["rolling_refit_cache_dir"] == captures[1]["rolling_refit_cache_dir"]
    assert captures[0]["rolling_refit_cache_dir"].is_relative_to(tmp_path / "shared")
    assert len(second.residual_daily_audit) == 731
    assert second.residual_daily_audit.phase.eq("evaluation").sum() == 365
    assert second.raw_history.index.equals(pd.DatetimeIndex(second.residual_statistics.delivery_start_utc))
    first_stats = first.residual_statistics.set_index("delivery_start_utc")
    second_stats = second.residual_statistics.set_index("delivery_start_utc")
    shared = first_stats.index.intersection(second_stats.index)
    columns = [f"residual_corrected__{q}" for q in engine.QUANTILES]
    pd.testing.assert_frame_equal(first_stats.loc[shared, columns], second_stats.loc[shared, columns])

    full = run(first_day + timedelta(days=1), "full", "reference")
    pd.testing.assert_frame_equal(second.source_forecast, full.source_forecast)
    # Current-day Kalman sees identical last365 corrected observations plus
    # identical future and covariates, even though bootstrap is now anchored.
    trailing = pd.to_datetime(second.residual_statistics.delivery_start_utc, utc=True).dt.tz_convert("Europe/Paris").dt.date >= first_day + timedelta(days=1-365)
    pd.testing.assert_frame_equal(second.residual_statistics.loc[trailing].reset_index(drop=True),
                                  full.residual_statistics.loc[trailing].reset_index(drop=True))
    pd.testing.assert_frame_equal(second.covariates, full.covariates)


def test_residual_cache_resume_and_revision_recompute_only_dependents(tmp_path):
    from chronos2_hourly.nuclear_residual_cache import ResidualDayCache
    from test_nuclear_forecast import _small_inputs
    raw, future, features, day = _small_inputs(days=5)

    def run(history, X):
        cache = ResidualDayCache(tmp_path / "residual", {"fixture": 1},
                                features=X, raw=history, timezone="Europe/Paris")
        result = engine.causal_residual_replay(raw_history=history, raw_future=future,
            features=X, timezone="Europe/Paris", delivery_day=day,
            residual_factory=CountingResidual, daily_cache=cache)
        return result, cache

    first, _ = run(raw, features)
    again, reused = run(raw, features)
    assert reused.hits == 4 and reused.misses == 0
    pd.testing.assert_frame_equal(first[1], again[1])
    revised = raw.copy()
    revised.loc[revised.index[-1], "actual"] += 100
    changed, cache = run(revised, features)
    assert cache.hits == 3 and cache.misses == 1
    assert not changed[1].equals(first[1])
