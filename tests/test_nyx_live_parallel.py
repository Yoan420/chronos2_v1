"""The isolated NYX scheduler is only an execution-order change."""
from __future__ import annotations

from datetime import date, timedelta
import os

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nuclear_forecast as engine
from chronos2_hourly.chronos_adapter import generate_delivery_plans
from chronos2_hourly.nuclear_residual_cache import ResidualDayCache
from chronos2_hourly.nyx_live_parallel import accelerated_residual_replay, parallel_residual_replay


class _Residual:
    min_training_rows = 48
    feature_builder_options = {"include_calendar": False, "include_daily_profiles": False}

    def fit(self, X, actual, base, experts):
        assert X.index.equals(actual.index)
        assert X.index.equals(base.index)
        assert X.index.equals(experts.index)
        self.end = X.index[-1]
        self.first = X.index[0]
        self.feature_columns_ = (engine.NUCLEAR_KNOWN_COLUMN,)
        self.shift = float((actual - base.q50).mean())
        return self

    def predict(self, X, base, experts):
        assert self.end < X.index[0]
        assert self.first.tz_convert("Europe/Paris").date() >= X.index[0].tz_convert("Europe/Paris").date() - timedelta(days=365)
        return base.add(self.shift + X[engine.NUCLEAR_KNOWN_COLUMN] * .01, axis=0)


def _inputs(delivery=date(2024, 4, 2), days=5):
    plans = generate_delivery_plans(delivery - timedelta(days=days), delivery,
                                    timezone="Europe/Paris", forecast_origin_local_time="08:00")
    frames = [pd.DataFrame({"q10": 30., "q50": 40., "q90": 50., "actual": 45. + i,
                           "forecast_origin_utc": plan.forecast_origin_utc},
                          index=plan.delivery_index_utc) for i, plan in enumerate(plans)]
    history, future = pd.concat(frames[:-1]), frames[-1].drop(columns="actual")
    features = pd.DataFrame({engine.NUCLEAR_KNOWN_COLUMN: 35.}, index=history.index.append(future.index))
    return dict(raw_history=history, raw_future=future, features=features,
                timezone="Europe/Paris", delivery_day=delivery, residual_factory=_Residual)


def _cache(path, options):
    return ResidualDayCache(path, {"unchanged_recipe": "test"}, features=options["features"],
                            raw=options["raw_history"], timezone="Europe/Paris")


def _equal(expected, actual, cache=False):
    for first, second in zip(expected[:2], actual[:2]):
        pd.testing.assert_frame_equal(first, second, check_exact=True)
    audit = actual[2].drop(columns="daily_cache_hit") if cache else actual[2]
    pd.testing.assert_frame_equal(expected[2], audit, check_exact=True)


@pytest.mark.parametrize("backend", ["threading", "loky"])
@pytest.mark.parametrize("delivery", [date(2024, 4, 2), date(2024, 10, 29)])
def test_exact_oracle_equivalence_across_dst_and_cold_start(backend, delivery):
    options = _inputs(delivery)
    expected = engine.causal_residual_replay(**options)
    events = []
    actual = parallel_residual_replay(workers=2, backend=backend, progress=events.append, **options)
    _equal(expected, actual)
    assert events[-1]["phase"] == "residual_complete"
    assert events[-1]["completed"] == events[-1]["total"] == 4
    if backend == "loky":
        assert all(event["worker_pid"] != os.getpid() for event in events if "worker_pid" in event)


def test_persistent_cache_exact_keys_and_resume(tmp_path):
    options = _inputs()
    sequential_cache = _cache(tmp_path / "serial", options)
    parallel_cache = _cache(tmp_path / "parallel", options)
    expected = engine.causal_residual_replay(**options)
    engine.causal_residual_replay(daily_cache=sequential_cache, **options)
    actual = parallel_residual_replay(workers=2, daily_cache=parallel_cache, **options)
    _equal(expected, actual, cache=True)
    assert {p.name for p in (tmp_path / "serial").glob("*.json")} == {p.name for p in (tmp_path / "parallel").glob("*.json")}
    for metadata in (tmp_path / "serial").glob("*.json"):
        assert metadata.read_bytes() == (tmp_path / "parallel" / metadata.name).read_bytes()
    events = []
    resumed = parallel_residual_replay(workers=2, daily_cache=_cache(tmp_path / "parallel", options),
                                       progress=events.append, **options)
    _equal(expected, resumed, cache=True)
    assert events[-1]["total"] == 0
    assert events[-1]["cached_days"] == 4


def test_cache_write_denial_keeps_verified_predictions_without_refit(tmp_path, monkeypatch):
    options = _inputs()
    expected = engine.causal_residual_replay(**options)
    cache = _cache(tmp_path, options)
    monkeypatch.setattr(cache, "store", lambda *_args: False)
    events = []
    actual = parallel_residual_replay(workers=2, daily_cache=cache, progress=events.append, **options)
    _equal(expected, actual, cache=True)
    assert events[-1]["completed"] == 4
    assert not list(tmp_path.iterdir())


def test_full_365_day_window_preserved_without_older_labels():
    options = _inputs(delivery=date(2025, 4, 2), days=370)
    options["output_start_day"] = options["delivery_day"] - timedelta(days=2)
    expected = engine.causal_residual_replay(**options)
    actual = parallel_residual_replay(workers=2, **options)
    _equal(expected, actual)
    assert actual[2].training_lookback_days.eq(365).all()
    assert actual[2].training_rows.ge(8759).all()


def test_target_actual_never_enters_its_fit_but_can_affect_later_days():
    options = _inputs()
    baseline = parallel_residual_replay(workers=2, **options)
    changed = dict(options, raw_history=options["raw_history"].copy(deep=True))
    target_day = options["delivery_day"] - timedelta(days=2)
    mask = changed["raw_history"].index.tz_convert("Europe/Paris").date == target_day
    changed["raw_history"].loc[mask, "actual"] += 1000.
    altered = parallel_residual_replay(workers=2, **changed)
    _equal(engine.causal_residual_replay(**changed), altered)
    base_day = pd.to_datetime(baseline[0].delivery_start_utc, utc=True).dt.tz_convert("Europe/Paris").dt.date
    assert np.array_equal(baseline[0].loc[base_day == target_day, "residual_corrected__q50"],
                          altered[0].loc[base_day == target_day, "residual_corrected__q50"])
    assert not baseline[1].q50.equals(altered[1].q50)


@pytest.mark.parametrize("fault,match", [("actual", "Future observations"), ("origin", "origins"),
                                        ("hole", "complete physical delivery day")])
def test_original_guards_fail_before_dispatch(fault, match):
    options = _inputs()
    if fault == "actual":
        options["raw_future"] = options["raw_future"].assign(actual=999.)
    elif fault == "origin":
        options["raw_future"] = options["raw_future"].assign(forecast_origin_utc=pd.Timestamp("2020-01-01", tz="UTC"))
    else:
        options["raw_future"] = options["raw_future"].iloc[:-1]
    events = []
    with pytest.raises(engine.NuclearForecastError, match=match):
        parallel_residual_replay(workers=2, progress=events.append, **options)
    assert not events


def test_fit_failure_propagates_and_scoped_patch_restores_original():
    class Broken(_Residual):
        def fit(self, *args):
            raise RuntimeError("deliberate fit failure")
    options = dict(_inputs(), residual_factory=Broken)
    original = engine.causal_residual_replay
    events = []
    with pytest.raises(RuntimeError, match="deliberate fit failure"):
        with accelerated_residual_replay(workers=2, progress=events.append, memory_reserve_gib=None):
            engine.causal_residual_replay(**options)
    assert engine.causal_residual_replay is original
    assert all(event["phase"] != "residual_complete" for event in events)


def test_scoped_patch_restores_and_rejects_nesting():
    original = engine.causal_residual_replay
    with accelerated_residual_replay(workers=2, backend="threading", memory_reserve_gib=None):
        with pytest.raises(RuntimeError, match="already patched"):
            with accelerated_residual_replay(workers=2):
                pass
        actual = engine.causal_residual_replay(**_inputs())
    assert engine.causal_residual_replay is original
    _equal(original(**_inputs()), actual)


@pytest.mark.parametrize("workers", [0, 9, True, 2.5])
def test_worker_ceiling(workers):
    with pytest.raises(ValueError, match="workers"):
        parallel_residual_replay(workers=workers, **_inputs())


@pytest.mark.parametrize("thread_count", [1, 2])
def test_real_catboost_local_factory_exact_process_equivalence(thread_count):
    from run_chronos2_hourly import _residual_corrector_factory
    options = _inputs()
    options["raw_history"].loc[:, "actual"] += np.sin(np.arange(len(options["raw_history"])) / 5.)
    options["features"].loc[:, engine.NUCLEAR_KNOWN_COLUMN] += np.cos(np.arange(len(options["features"])) / 7.)
    factory, _ = _residual_corrector_factory({"hourly": {"residual_correction": {
        "enabled": True, "base_model": "chronos2", "backend": "catboost",
        "min_training_rows": 48, "iterations": 2, "depth": 2, "thread_count": thread_count,
        "min_samples_leaf": 2, "verbose": False,
        "feature_builder": {"timezone": "Europe/Paris", "include_calendar": True,
                            "include_daily_profiles": True, "exclude_historical_prices": True},
    }}}, timezone="Europe/Paris")
    options["residual_factory"] = factory
    expected = engine.causal_residual_replay(**options)
    actual = parallel_residual_replay(workers=2, **options)
    _equal(expected, actual)


def test_real_interaction_factory_pickles_and_is_exact():
    from chronos2_hourly.nyx_live_baseline import make_residual_factory
    options = _inputs()
    score = pd.DataFrame({"nyx_test_interaction": .5}, index=options["features"].index)
    factory = make_residual_factory({"hourly": {"residual_correction": {
        "enabled": True, "base_model": "chronos2", "backend": "catboost",
        "min_training_rows": 48, "iterations": 2, "depth": 2, "thread_count": 2,
        "min_samples_leaf": 2, "verbose": False, "max_abs_correction": 40., "correction_scale": 1.,
        "feature_builder": {"timezone": "Europe/Paris", "include_calendar": True,
                            "include_daily_profiles": True, "exclude_historical_prices": True},
    }}}, score)
    options["residual_factory"] = factory
    expected = engine.causal_residual_replay(**options)
    actual = parallel_residual_replay(workers=2, **options)
    _equal(expected, actual)
    fitted = actual[2].residual_feature_columns.map(bool)
    assert all("nyx_test_interaction" in columns for columns in actual[2].loc[fitted, "residual_feature_columns"])


def test_memory_capacity_waits_and_recaps_without_starting_fits(monkeypatch):
    from chronos2_hourly import nyx_live_parallel as scheduler
    memory = iter([3., 3.5, 4.49, 5.8, 4.9])
    waits, events = [], []
    monkeypatch.setattr(scheduler, "_available_gib", lambda: next(memory))
    monkeypatch.setattr(scheduler.time, "sleep", waits.append)
    report = lambda phase, **extra: events.append(dict(phase=phase, **extra))
    assert scheduler._memory_capacity(4, 3.5, 1., report, 20.) == 2
    assert scheduler._memory_capacity(4, 3.5, 1., report, 20.) == 1
    assert waits == [20., 20., 20.]
    assert all(event["phase"] == "waiting_memory" for event in events)


def test_runtime_memory_guard_bounds_waves(monkeypatch):
    from chronos2_hourly import nyx_live_parallel as scheduler
    monkeypatch.setattr(scheduler, "_available_gib", lambda: 5.6)
    options, events = _inputs(), []
    actual = parallel_residual_replay(workers=4, backend="threading", memory_reserve_gib=3.5,
                                     progress=events.append, **options)
    _equal(engine.causal_residual_replay(**options), actual)
    assert all(event["active_wave_workers"] <= 2 for event in events if "active_wave_workers" in event)


def test_owned_loky_workers_are_released_before_return():
    import psutil
    events = []
    parallel_residual_replay(workers=2, progress=events.append, **_inputs())
    worker_pids = {event["worker_pid"] for event in events if "worker_pid" in event}
    assert worker_pids
    assert os.getpid() not in worker_pids
    assert all(not psutil.pid_exists(pid) for pid in worker_pids)


def test_memory_capacity_can_grow_between_waves(monkeypatch):
    from chronos2_hourly import nyx_live_parallel as scheduler
    memory = iter([5.6, 7.6])
    monkeypatch.setattr(scheduler, "_available_gib", lambda: next(memory))
    options, events = _inputs(days=7), []
    actual = parallel_residual_replay(workers=4, backend="threading", memory_reserve_gib=3.5,
                                     progress=events.append, **options)
    _equal(engine.causal_residual_replay(**options), actual)
    waves = [event["active_wave_workers"] for event in events
             if "active_wave_workers" in event and "last_day" not in event]
    assert waves == [2, 4]
