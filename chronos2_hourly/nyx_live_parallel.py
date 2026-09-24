"""Opt-in process-local scheduling adapter for isolated NYX live experiments.

Only independent daily residual fits are scheduled concurrently. The original
causal replay remains the numerical oracle, input validator, cache consumer,
and chronological output assembler. Importing this module patches nothing.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, timedelta
import os
import time
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from . import nuclear_forecast as engine


_SERIAL_REPLAY = engine.causal_residual_replay
MAX_WORKERS = 8


def _available_gib() -> float:
    import psutil
    return psutil.virtual_memory().available / 1024 ** 3


def _memory_capacity(requested, reserve_gib, worker_gib, report, poll_seconds):
    if reserve_gib is None:
        return requested
    while True:
        available = _available_gib()
        if available >= reserve_gib + worker_gib:
            return min(requested, int((available - reserve_gib) / worker_gib))
        report("waiting_memory", available_memory_gib=available,
               memory_reserve_gib=reserve_gib, minimum_available_gib=reserve_gib + worker_gib,
               poll_seconds=poll_seconds)
        time.sleep(poll_seconds)


class _PreflightComplete(Exception):
    """Stop after all original input checks, before any original fit."""


class _PreflightCache:
    def load(self, *_args: Any) -> None:
        raise _PreflightComplete


class _CaptureCache:
    def __init__(self) -> None:
        self.value = None

    def load(self, *_args: Any) -> None:
        return None

    def store(self, day, training_index, predicted_index, prediction_base, predicted, columns):
        if self.value is not None:
            raise RuntimeError("A residual job must produce exactly one day")
        self.value = (predicted.copy(deep=True), tuple(columns))
        return True


class _ReadyCache:
    """Verified per-invocation values; no changed persistent cache signatures."""
    def __init__(self, values):
        self.values = values

    def load(self, day, _training_index, predicted_index, _prediction_base):
        predicted, columns = self.values[day]
        if not predicted.index.equals(predicted_index):
            raise engine.NuclearForecastError("Prefetched residual day index changed")
        return predicted.copy(deep=True), columns

    def store(self, *_args):
        raise RuntimeError("Every eligible residual day must have been prefetched")


def _fit_day(day, history, future, features, timezone, factory):
    """Run the unchanged oracle for one target, never passing its actual label."""
    started = time.monotonic()
    capture = _CaptureCache()
    _SERIAL_REPLAY(raw_history=history, raw_future=future, features=features,
                   timezone=timezone, delivery_day=day, residual_factory=factory,
                   output_start_day=day, daily_cache=capture)
    if capture.value is None:
        raise RuntimeError("Eligible residual day produced no prediction")
    return day, capture.value, time.monotonic() - started, os.getpid()


def parallel_residual_replay(
    *, workers: int, raw_history: pd.DataFrame, raw_future: pd.DataFrame,
    features: pd.DataFrame, timezone: str, delivery_day: date,
    residual_factory: Callable[[], Any], output_start_day: date | None = None,
    daily_cache: Any | None = None, progress: Callable[[dict], None] | None = None,
    backend: str = "loky",
    memory_reserve_gib: float | None = None, worker_memory_gib: float = 1.,
    memory_poll_seconds: float = 20.,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prefetch independent fits, retain exact recipe, then assemble serially.

The caller selects workers using its RAM/production reserve. ``loky`` is the
isolated production backend; ``threading`` is supported for equivalence tests.
No recipe parameter (including CatBoost thread_count) is changed here.
"""
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be an integer between 1 and {MAX_WORKERS}")
    if backend not in ("loky", "threading"):
        raise ValueError("Only loky and threading backends are supported")
    if (memory_reserve_gib is not None and memory_reserve_gib < 0
            or worker_memory_gib <= 0 or not 0 < memory_poll_seconds <= 60):
        raise ValueError("Invalid memory reserve, worker estimate or polling interval")
    options = dict(raw_history=raw_history, raw_future=raw_future, features=features,
                   timezone=timezone, delivery_day=delivery_day,
                   residual_factory=residual_factory, output_start_day=output_start_day,
                   daily_cache=daily_cache)
    try:
        _SERIAL_REPLAY(**{**options, "daily_cache": _PreflightCache()})
    except _PreflightComplete:
        pass
    else:
        # There are no fitted days; retain the original cold-start audit.
        return _SERIAL_REPLAY(**options)

    history = engine._utc_frame(raw_history, name="raw_history")
    future = engine._utc_frame(raw_future, name="raw_future")
    X = engine._utc_frame(features, name="features")
    base = engine._quantiles(history, name="raw_history")
    future_base = engine._quantiles(future, name="raw_future")
    days = np.asarray(history.index.tz_convert(timezone).date, dtype=object)
    output_days = tuple(day for day in pd.Index(days).unique()
                        if output_start_day is None or day >= output_start_day)
    minimum_rows = engine._residual_recipe_preflight(residual_factory)
    values, pending, inputs = {}, [], {}
    for day in (*output_days, delivery_day):
        training_index = history.index[(days >= day - timedelta(days=engine.RESIDUAL_LOOKBACK_DAYS))
                                       & (days < day)]
        if len(training_index) < minimum_rows:
            continue
        is_future = day == delivery_day
        predicted_index = future.index if is_future else history.index[days == day]
        prediction_base = future_base if is_future else base.loc[predicted_index]
        inputs[day] = training_index, predicted_index, prediction_base
        cached = daily_cache.load(day, training_index, predicted_index, prediction_base) if daily_cache else None
        if cached is not None:
            values[day] = cached
        else:
            pending.append(day)

    started = time.monotonic()
    completed = 0
    cached_count = len(values)
    actual_workers = min(workers, len(pending))

    def report(phase, **extra):
        if progress is not None:
            progress(dict(phase=phase, completed=completed, total=len(pending),
                          cached_days=cached_count, workers=actual_workers,
                          elapsed_seconds=time.monotonic() - started, **extra))

    def jobs(wave):
        for day in wave:
            training_index, predicted_index, _ = inputs[day]
            target = future if day == delivery_day else history.loc[predicted_index]
            # Bounded payload: exact D-365..D-1 labels and target-day features.
            # D's actual is deliberately absent even for historical jobs.
            yield delayed(_fit_day)(day, history.loc[training_index].copy(deep=True),
                                    target.drop(columns="actual", errors="ignore").copy(deep=True),
                                    X.loc[training_index.append(predicted_index)].copy(deep=True),
                                    timezone, residual_factory)

    report("residual_prefill")
    owned_executors = {}
    try:
        if pending:
            # Reassess after Chronos has allocated its memory, and between bounded
            # waves. At most one day per active worker is submitted at any time.
            dispatched = 0
            while dispatched < len(pending):
                wave_workers = _memory_capacity(actual_workers, memory_reserve_gib, worker_memory_gib,
                                                report, memory_poll_seconds)
                wave = pending[dispatched:dispatched + wave_workers]
                config = dict(backend=backend, n_jobs=wave_workers)
                if backend == "loky":
                    config.update(inner_max_num_threads=1, idle_worker_timeout=30)
                report("residual_prefill", active_wave_workers=len(wave))
                with parallel_config(**config):
                    with Parallel(return_as="generator_unordered", pre_dispatch=wave_workers, batch_size=1) as pool:
                        # Keep the exact executor handle, rather than discovering
                        # unrelated processes or creating a new reusable executor
                        # merely to shut it down. Parallel clears this reference
                        # on exit but normally retains the resident loky pool.
                        executor = getattr(pool._backend, "_workers", None) if backend == "loky" else None
                        if executor is not None:
                            owned_executors[id(executor)] = executor
                        for day, value, fit_seconds, worker_pid in pool(jobs(wave)):
                            predicted, columns = value
                            values[day] = value
                            if daily_cache is not None:
                                training_index, predicted_index, prediction_base = inputs[day]
                                # Cache is disposable. A denied write does not discard
                                # this already-validated in-memory prediction.
                                daily_cache.store(day, training_index, predicted_index,
                                                  prediction_base, predicted, columns)
                            completed += 1
                            report("residual_prefill", last_day=day.isoformat(),
                                   last_fit_seconds=fit_seconds, worker_pid=worker_pid,
                                   active_wave_workers=len(wave))
                dispatched += len(wave)
    finally:
        # Dedicated launcher process only. Release this adapter's idle workers
        # before Kalman, including after errors; no system-wide process kill.
        # During memory waits they also expire naturally after 30 idle seconds.
        for executor in owned_executors.values():
            executor.shutdown(wait=True)
    report("residual_assembly")
    # Preserve all chronological audit phases, feature guards, quantile/index
    # checks and output selection through the unchanged original implementation.
    result = _SERIAL_REPLAY(**{**options, "daily_cache": _ReadyCache(values)})
    if daily_cache is None:
        result = (*result[:2], result[2].drop(columns="daily_cache_hit"))
    report("residual_complete")
    return result


@contextmanager
def accelerated_residual_replay(*, workers: int, progress=None, backend="loky",
                               memory_reserve_gib=3.5, worker_memory_gib=1.,
                               memory_poll_seconds=20.) -> Iterator[None]:
    """Explicit single-run patch; restore the original even after a failure.

Do not use in a process serving unrelated forecasts concurrently. The live
accelerated launcher owns its dedicated process and immutable run identity.
"""
    if engine.causal_residual_replay is not _SERIAL_REPLAY:
        raise RuntimeError("Residual replay is already patched; refusing nested scheduling")

    def scheduled(**kwargs):
        return parallel_residual_replay(workers=workers, progress=progress, backend=backend,
                                       memory_reserve_gib=memory_reserve_gib,
                                       worker_memory_gib=worker_memory_gib,
                                       memory_poll_seconds=memory_poll_seconds, **kwargs)

    engine.causal_residual_replay = scheduled
    try:
        yield
    finally:
        engine.causal_residual_replay = _SERIAL_REPLAY
