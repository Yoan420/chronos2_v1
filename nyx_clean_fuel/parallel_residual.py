"""Fill CleanFuel's existing daily residual cache concurrently before replay.

The numerical replay and its source hash stay unchanged. Each day's fit uses
exactly the same past 365 days and factory as the sequential implementation;
the original replay then reads the checksum-verified daily cache in order.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd

from chronos2_hourly import nuclear_forecast as engine


class _PreflightComplete(Exception):
    """Stop at the first cache read, after the original replay's input checks."""


class _PreflightCache:
    def load(self, *_args: Any) -> None:
        raise _PreflightComplete


def prefill_residual_days(
    original: Callable[..., tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    *, workers: int, raw_history: pd.DataFrame, raw_future: pd.DataFrame,
    features: pd.DataFrame, timezone: str, delivery_day: date,
    residual_factory: Callable[[], Any], output_start_day: date | None = None,
    daily_cache: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resume verified days, fit misses in parallel, then run the original replay."""
    options = dict(raw_history=raw_history, raw_future=raw_future, features=features,
                   timezone=timezone, delivery_day=delivery_day,
                   residual_factory=residual_factory, output_start_day=output_start_day,
                   daily_cache=daily_cache)
    if workers < 2 or daily_cache is None:
        return original(**options)
    try:
        original(**{**options, "daily_cache": _PreflightCache()})
    except _PreflightComplete:
        pass
    else:
        return original(**options)

    history = engine._utc_frame(raw_history, name="raw_history")
    future = engine._utc_frame(raw_future, name="raw_future")
    X = engine._utc_frame(features, name="features")
    base = engine._quantiles(history, name="raw_history")
    future_base = engine._quantiles(future, name="raw_future")
    experts = base.rename(columns={q: f"chronos2__{q}" for q in engine.QUANTILES})
    future_experts = future_base.rename(columns={q: f"chronos2__{q}" for q in engine.QUANTILES})
    actual = pd.to_numeric(history.actual, errors="coerce")
    days = np.asarray(history.index.tz_convert(timezone).date, dtype=object)
    unique_days = tuple(pd.Index(days).unique())
    output_days = tuple(day for day in unique_days if output_start_day is None or day >= output_start_day)
    minimum_rows = engine._residual_recipe_preflight(residual_factory)
    pending = []
    counters = (daily_cache.hits, daily_cache.misses)
    try:
        for day in (*output_days, delivery_day):
            training_index = history.index[(days >= day - timedelta(days=engine.RESIDUAL_LOOKBACK_DAYS)) & (days < day)]
            if len(training_index) < minimum_rows:
                continue
            is_future = day == delivery_day
            predicted_index = future.index if is_future else history.index[days == day]
            prediction_base = future_base if is_future else base.loc[predicted_index]
            if daily_cache.load(day, training_index, predicted_index, prediction_base) is None:
                pending.append((day, training_index, predicted_index, prediction_base,
                                future_experts if is_future else experts.loc[predicted_index]))
    finally:
        # Discovery is read-only. The original replay owns the public hit/miss
        # counters and performs the final causal validation in delivery order.
        daily_cache.hits, daily_cache.misses = counters

    if pending:
        print(f"[CleanFuel] Correcteur : {len(pending)} jour(s) absents du cache; "
              f"{workers} calculs simultanes, puis verification par le replay original.", flush=True)

    def fit_one(item):
        day, training_index, predicted_index, prediction_base, prediction_experts = item
        fitted = residual_factory()
        fitted.fit(X.loc[training_index].copy(deep=True), actual.loc[training_index].copy(deep=True),
                   base.loc[training_index].copy(deep=True), experts.loc[training_index].copy(deep=True))
        fitted_features = tuple(map(str, fitted.feature_columns_))
        if not any(engine.NUCLEAR_ALIAS in column for column in fitted_features):
            raise engine.NuclearForecastError("Fitted residual recipe dropped the nuclear input")
        predicted = fitted.predict(X.loc[predicted_index].copy(deep=True),
                                   prediction_base.copy(deep=True), prediction_experts.copy(deep=True))
        if not predicted.index.equals(predicted_index):
            raise engine.NuclearForecastError("Residual output index differs from requested day")
        return day, training_index, predicted_index, prediction_base, engine._quantiles(predicted, name=f"residual {day}"), fitted_features

    if pending:
        with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as executor:
            futures = [executor.submit(fit_one, item) for item in pending]
            storage_blocked = False
            try:
                for completed in as_completed(futures):
                    day, training_index, predicted_index, prediction_base, predicted, columns = completed.result()
                    if not daily_cache.store(day, training_index, predicted_index, prediction_base, predicted, columns):
                        storage_blocked = True
                        for task in futures:
                            task.cancel()
                        break
            except BaseException:
                for task in futures:
                    task.cancel()
                raise
        if storage_blocked:
            # The original numerical engine treats a cache write failure as a
            # disposable miss. Preserve that behavior without dropping a fit.
            print("[CleanFuel] Cache du correcteur indisponible; reprise "
                  "sequentielle des seules journees non sauvegardees.", flush=True)
    return original(**options)
