"""CleanFuel parallel cache filling preserves the original causal replay."""
from __future__ import annotations

from datetime import date, timedelta
from threading import Barrier, Lock, current_thread

import numpy as np
import pandas as pd

from chronos2_hourly import nuclear_forecast as engine
from chronos2_hourly.chronos_adapter import generate_delivery_plans
from chronos2_hourly.nuclear_residual_cache import ResidualDayCache
from nyx_clean_fuel.parallel_residual import prefill_residual_days


class _Residual:
    min_training_rows = 48
    feature_builder_options = {"include_calendar": False, "include_daily_profiles": False}
    visits = []
    gate = None
    lock = Lock()

    def fit(self, X, actual, base, experts):
        with self.lock:
            self.visits.append(current_thread().name)
            should_wait = self.gate is not None and len(self.visits) <= 2
        if should_wait:
            self.gate.wait(timeout=10)
        self.feature_columns_ = (engine.NUCLEAR_KNOWN_COLUMN,)
        self.end = X.index[-1]
        self.shift = float((actual - base.q50).mean())
        return self

    def predict(self, X, base, experts):
        assert self.end < X.index[0]
        return base.add(self.shift + X[engine.NUCLEAR_KNOWN_COLUMN] * 0.01, axis=0)


def _inputs():
    delivery = date(2024, 4, 2)
    plans = generate_delivery_plans(delivery - timedelta(days=5), delivery,
                                    timezone="Europe/Paris", forecast_origin_local_time="08:00")
    frames = [pd.DataFrame({"q10": 30.0, "q50": 40.0, "q90": 50.0,
                            "actual": 45.0, "forecast_origin_utc": plan.forecast_origin_utc},
                           index=plan.delivery_index_utc) for plan in plans]
    history, future = pd.concat(frames[:-1]), frames[-1].drop(columns="actual")
    features = pd.DataFrame({engine.NUCLEAR_KNOWN_COLUMN: 35.0},
                            index=history.index.append(future.index))
    return dict(raw_history=history, raw_future=future, features=features,
                timezone="Europe/Paris", delivery_day=delivery, residual_factory=_Residual)


def _cache(tmp_path, options):
    return ResidualDayCache(tmp_path, {"recipe": "identical"},
                            features=options["features"], raw=options["raw_history"],
                            timezone="Europe/Paris")


def test_parallel_fits_match_original_across_spring_dst_and_resume_without_refits(tmp_path):
    options = _inputs()
    original = engine.causal_residual_replay
    _Residual.visits, _Residual.gate = [], None
    expected = original(**options)
    assert len(_Residual.visits) == 4

    _Residual.visits, _Residual.gate = [], Barrier(2)
    cache = _cache(tmp_path, options)
    actual = prefill_residual_days(original, workers=2, daily_cache=cache, **options)
    _Residual.gate = None
    for first, second in zip(expected[:2], actual[:2]):
        pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(expected[2], actual[2].drop(columns="daily_cache_hit"))
    assert len(_Residual.visits) == 4
    assert all(name != "MainThread" for name in _Residual.visits)
    assert len(set(_Residual.visits)) == 2
    assert len(list(tmp_path.glob("*.json"))) == 4

    _Residual.visits = []
    replayed = prefill_residual_days(original, workers=2,
                                    daily_cache=_cache(tmp_path, options), **options)
    assert not _Residual.visits
    pd.testing.assert_frame_equal(actual[1], replayed[1])


def test_rejects_future_observations_before_starting_parallel_fit(tmp_path):
    options = _inputs()
    options["raw_future"] = options["raw_future"].assign(actual=999.0)
    _Residual.visits, _Residual.gate = [], None
    import pytest

    with pytest.raises(engine.NuclearForecastError, match="Future observations"):
        prefill_residual_days(engine.causal_residual_replay, workers=2,
                              daily_cache=_cache(tmp_path, options), **options)
    assert not _Residual.visits


def test_real_catboost_parallel_matches_original(tmp_path):
    from run_chronos2_hourly import _residual_corrector_factory

    options = _inputs()
    options["raw_history"].loc[:, "actual"] += np.sin(np.arange(len(options["raw_history"])) / 5.0)
    options["features"].loc[:, engine.NUCLEAR_KNOWN_COLUMN] += np.cos(np.arange(len(options["features"])) / 7.0)
    factory, _ = _residual_corrector_factory({"hourly": {"residual_correction": {
        "enabled": True, "base_model": "chronos2", "backend": "catboost",
        "min_training_rows": 48, "iterations": 2, "depth": 2,
        "min_samples_leaf": 2, "thread_count": 4, "verbose": False,
        "feature_builder": {"timezone": "Europe/Paris", "include_calendar": True,
                            "include_daily_profiles": True,
                            "exclude_historical_prices": True},
    }}}, timezone="Europe/Paris")
    options["residual_factory"] = factory
    original = engine.causal_residual_replay
    expected = original(**options)
    actual = prefill_residual_days(original, workers=2,
                                   daily_cache=_cache(tmp_path, options), **options)
    for first, second in zip(expected[:2], actual[:2]):
        pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(expected[2], actual[2].drop(columns="daily_cache_hit"))


def test_cache_write_denial_falls_back_to_original_replay(tmp_path, monkeypatch):
    options = _inputs()
    _Residual.visits, _Residual.gate = [], None
    original = engine.causal_residual_replay
    expected = original(**options)
    cache = _cache(tmp_path, options)
    monkeypatch.setattr(cache, "store", lambda *_args: False)
    actual = prefill_residual_days(original, workers=2, daily_cache=cache, **options)
    for first, second in zip(expected[:2], actual[:2]):
        pd.testing.assert_frame_equal(first, second)
    assert not list(tmp_path.iterdir())
