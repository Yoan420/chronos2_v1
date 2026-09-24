"""Audited, prediction-only variable attribution for one live delivery day.

The production forecaster is nonlinear and its Chronos-2 component does not
expose native feature weights.  This module therefore explains the *published*
P50 with grouped Shapley counterfactuals. Every configurable physical series,
and the historical target-price context, is kept at its available value or
replaced by a causal historical reference curve. Target-derived lag/rolling
features follow the same substituted past target; future target observations
are never supplied. Chronos-2, the frozen residual corrector and (when enabled)
the fixed MKOnline blend are evaluated on the exact same coalition.

The resulting values are predictive attributions, not causal economic effects.
They are computed after the official forecast has been frozen and are never fed
back into the prediction path.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from chronos2_modular.common import DEFAULT_QUANTILES
from chronos2_modular.forecasting import (
    build_live_frames,
    normalize_prediction_columns,
    prepare_chronos_frame,
)


VARIABLE_ATTRIBUTION_HOURLY = "variable_attribution_hourly.csv.gz"
VARIABLE_ATTRIBUTION_AUDIT = "variable_attribution_audit.json"
ATTRIBUTION_SCHEMA_VERSION = 1
BASELINE_REFERENCE = "causal_56d_hour_of_week_median"
EXACT_SHAPLEY_MAX_GROUPS = 6
DEFAULT_APPROXIMATE_PERMUTATIONS = 32
MAX_APPROXIMATE_PERMUTATIONS = 128
MAX_ATTRIBUTION_SCENARIOS = 256
MAX_ATTRIBUTION_GROUPS = 32
PAST_PRICE_GROUP_KEY = "historical_target_price"
_PRICE_LAG_PATTERN = re.compile(r"^price_lag_([1-9][0-9]*)h$")
_PRICE_ROLLING_PATTERN = re.compile(r"^price_rolling_(mean|std|min|max|median)_([1-9][0-9]*)h$")
QUANTILES = ("q10", "q50", "q90")


class VariableAttributionError(ValueError):
    """Raised when an explanation would not reproduce the frozen forecast."""


@dataclass(frozen=True)
class VariableGroup:
    """One user-facing physical input and all of its derived model columns."""

    key: str
    label: str
    aliases: tuple[str, ...]
    context_columns: tuple[str, ...]
    future_columns: tuple[str, ...]
    baseline_reference: str = BASELINE_REFERENCE
    target_context: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_index(index: pd.Index, *, name: str) -> pd.DatetimeIndex:
    values = pd.DatetimeIndex(pd.to_datetime(index, utc=True, errors="raise"))
    if values.has_duplicates or not values.is_monotonic_increasing:
        raise VariableAttributionError(f"{name} must be unique and increasing")
    return values.rename("delivery_start_utc")


def _human_label(alias: str) -> str:
    normalized = str(alias).strip().lower()
    country = normalized.split("_", 1)[0].upper()
    if normalized.endswith("_residual_load_fcst") and len(country) == 2:
        return f"Charge résiduelle {country}"
    if normalized.endswith("_nuclear_generation_fcst_gw") and len(country) == 2:
        return f"Production nucléaire prévue {country} (GW)"
    return normalized.replace("_fcst", "").replace("_", " ").strip().capitalize()


def build_variable_groups(
    *,
    required_covariates: Sequence[str],
    context_columns: Sequence[str],
    future_columns: Sequence[str],
    include_past_prices: bool = True,
) -> tuple[VariableGroup, ...]:
    """Map every configured series to its raw and known-future columns.

    Calendar remains fixed background information. Past target prices form a
    separate group, including their supported lag/rolling features. The actual
    target lives in ZoneData.target, not in model_context_covariates. Passing
    include_past_prices=False retains the former physical-only explanation.
    """

    context = tuple(str(value) for value in context_columns)
    future = tuple(str(value) for value in future_columns)
    groups: list[VariableGroup] = []
    seen: set[str] = set()
    for raw_alias in required_covariates:
        alias = str(raw_alias).strip()
        key = alias.casefold()
        if not alias or key in seen:
            raise VariableAttributionError(
                f"invalid or duplicated attribution alias: {raw_alias!r}"
            )
        if "storm" in key or "mkonline" in key:
            raise VariableAttributionError(
                f"evaluation-only/external comparator cannot be a variable: {alias}"
            )
        if key == "calendar" or key.startswith(("calendar_", "known_cal_")):
            raise VariableAttributionError("calendar stays fixed outside attribution groups")
        if key in {"target", PAST_PRICE_GROUP_KEY} or _price_feature_columns((key,)):
            raise VariableAttributionError(f"target-derived input is reserved for past prices: {alias}")
        seen.add(key)
        prefix = f"known_{key}_"
        context_matches = tuple(
            column
            for column in context
            if column.casefold() == key or column.casefold().startswith(prefix)
        )
        future_matches = tuple(
            column
            for column in future
            if column.casefold() == key or column.casefold().startswith(prefix)
        )
        if not context_matches and not future_matches:
            raise VariableAttributionError(
                f"configured variable {alias!r} has no model column"
            )
        groups.append(
            VariableGroup(
                key=key,
                label=_human_label(alias),
                aliases=(alias,),
                context_columns=context_matches,
                future_columns=future_matches,
            )
        )
    if include_past_prices:
        if "target" in context or "target" in future:
            raise VariableAttributionError("target is context-only in ZoneData.target, not a future covariate")
        groups.append(VariableGroup(
            key=PAST_PRICE_GROUP_KEY,
            label="Prix passés (contexte Chronos)",
            aliases=("target",),
            context_columns=("target", *_price_feature_columns(context)),
            future_columns=_price_feature_columns(future),
            target_context=True,
        ))
    if not groups:
        raise VariableAttributionError("at least one physical variable is required")
    return tuple(groups)


def _price_feature_columns(columns: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for raw in columns:
        column = str(raw)
        if _PRICE_LAG_PATTERN.fullmatch(column) or _PRICE_ROLLING_PATTERN.fullmatch(column):
            result.append(column)
        elif column.startswith(("price_lag_", "price_rolling_", "target_price_lag_", "target_price_rolling_")):
            raise VariableAttributionError(f"unsupported target-derived feature: {column}")
    return tuple(result)


def _target_price_derivatives(
    target: pd.Series, *, output_index: pd.DatetimeIndex,
    columns: Sequence[str], timezone: str,
) -> pd.DataFrame:
    """Rebuild the existing price feature contract without future labels."""
    from .features import _price_history_features, validate_utc_hourly_index

    requested = _price_feature_columns(columns)
    if not requested:
        return pd.DataFrame(index=output_index)
    source = pd.Series(
        pd.to_numeric(target, errors="coerce").to_numpy(dtype=float),
        index=_utc_index(target.index, name="past_target.index"), name="target",
    )
    # Only NaNs are added after the actual historical endpoint. This preserves
    # production lag masking on the 25th DST hour and day-start-frozen rolls.
    combined_index = source.index.union(output_index).sort_values()
    validate_utc_hourly_index(combined_index, name="price_baseline.index")
    combined = source.reindex(combined_index)
    lags = tuple(sorted({int(match.group(1)) for col in requested if (match := _PRICE_LAG_PATTERN.fullmatch(col))}))
    windows = tuple(sorted({int(match.group(2)) for col in requested if (match := _PRICE_ROLLING_PATTERN.fullmatch(col))}))
    statistics = tuple(sorted({match.group(1) for col in requested if (match := _PRICE_ROLLING_PATTERN.fullmatch(col))}))
    derived = _price_history_features(
        combined, price_lags=lags, rolling_windows=windows,
        rolling_statistics=statistics, timezone=timezone,
    )
    return derived.loc[output_index, list(requested)]


def _hour_of_week(index: pd.DatetimeIndex, timezone: str) -> np.ndarray:
    local = index.tz_convert(timezone)
    return local.dayofweek.to_numpy(dtype=int) * 24 + local.hour.to_numpy(dtype=int)


def _causal_profile(
    values: pd.Series,
    *,
    historical_end: pd.Timestamp,
    output_index: pd.DatetimeIndex,
    timezone: str,
    lookback_days: int = 56,
) -> pd.Series:
    """Return a causal hour-of-week median curve on ``output_index``."""

    source = pd.to_numeric(values, errors="coerce").astype(float)
    source_index = _utc_index(source.index, name=f"{values.name or 'variable'}.index")
    source = pd.Series(source.to_numpy(dtype=float), index=source_index)
    end = pd.Timestamp(historical_end)
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    start = end - pd.Timedelta(days=int(lookback_days)) + pd.Timedelta(hours=1)
    history = source.loc[(source.index >= start) & (source.index <= end)]
    history = history.loc[np.isfinite(history.to_numpy(dtype=float))]
    if history.empty:
        history = source.loc[
            (source.index <= end) & np.isfinite(source.to_numpy(dtype=float))
        ]
    if history.empty:
        raise VariableAttributionError(
            f"no causal finite baseline values for {values.name or 'variable'}"
        )
    keys = _hour_of_week(history.index, timezone)
    profile = pd.Series(history.to_numpy(dtype=float), index=keys).groupby(level=0).median()
    fallback = float(np.median(history.to_numpy(dtype=float)))
    output_keys = _hour_of_week(output_index, timezone)
    output = np.asarray([profile.get(int(key), fallback) for key in output_keys], dtype=float)
    if not np.isfinite(output).all():
        raise VariableAttributionError("causal reference curve contains non-finite values")
    return pd.Series(output, index=output_index, name=values.name)


def _baseline_frames(
    *,
    data: Any,
    groups: Sequence[VariableGroup],
    historical_end: pd.Timestamp,
    timezone: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None]:
    model_context = data.model_context_covariates.copy()
    model_index = _utc_index(model_context.index, name="model_context.index")
    model_context.index = model_index
    covariates = data.covariates.copy()
    covariate_index = _utc_index(covariates.index, name="covariates.index")
    covariates.index = covariate_index
    baseline_context = pd.DataFrame(index=model_index)
    baseline_covariates = pd.DataFrame(index=covariate_index)
    baseline_target: pd.Series | None = None
    for group in groups:
        if group.target_context:
            target_index = _utc_index(data.target.index, name="target.index")
            baseline_target = _causal_profile(
                data.target, historical_end=historical_end,
                output_index=target_index, timezone=timezone,
            )
            derivative_columns = tuple(dict.fromkeys(
                column for column in (*group.context_columns, *group.future_columns) if column != "target"
            ))
            derived = _target_price_derivatives(
                baseline_target, output_index=model_index,
                columns=derivative_columns, timezone=timezone,
            )
            for column in derivative_columns:
                baseline_context[column] = derived[column]
            continue
        for column in group.context_columns:
            if column not in model_context:
                raise VariableAttributionError(
                    f"context attribution column is missing: {column}"
                )
            baseline_context[column] = _causal_profile(
                model_context[column],
                historical_end=historical_end,
                output_index=model_index,
                timezone=timezone,
            )
        for alias in group.aliases:
            if alias not in covariates:
                continue
            baseline_covariates[alias] = _causal_profile(
                covariates[alias],
                historical_end=historical_end,
                output_index=covariate_index,
                timezone=timezone,
            )
    return baseline_context, baseline_covariates, baseline_target


def _coalition_masks(
    n_groups: int,
    *,
    seed: int,
    approximate_permutations: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], str]:
    """Return masks to predict and permutation paths used for Shapley values."""

    if (isinstance(n_groups, bool) or int(n_groups) != n_groups
            or not 1 <= int(n_groups) <= MAX_ATTRIBUTION_GROUPS):
        raise VariableAttributionError(f"attribution requires 1..{MAX_ATTRIBUTION_GROUPS} groups")
    if n_groups <= EXACT_SHAPLEY_MAX_GROUPS:
        masks = tuple(range(1 << n_groups))
        return masks, (), "exact_grouped_shapley_end_to_end"
    requested = int(approximate_permutations)
    if not 1 <= requested <= MAX_APPROXIMATE_PERMUTATIONS:
        raise VariableAttributionError(f"approximate_permutations requires 1..{MAX_APPROXIMATE_PERMUTATIONS}")
    # A path has at most n_groups-1 non-endpoint masks. Keep inference bounded
    # even when a configuration adds many physical series alongside prices.
    repetitions = min(requested, (MAX_ATTRIBUTION_SCENARIOS - 2) // (n_groups - 1))
    rng = np.random.default_rng(int(seed))
    orders = tuple(tuple(int(v) for v in rng.permutation(n_groups)) for _ in range(repetitions))
    mask_set = {0, (1 << n_groups) - 1}
    for order in orders:
        mask = 0
        for group_index in order:
            mask |= 1 << group_index
            mask_set.add(mask)
    return (
        tuple(sorted(mask_set)),
        orders,
        "permutation_grouped_shapley_end_to_end",
    )


def _scenario_frames(
    *,
    data: Any,
    groups: Sequence[VariableGroup],
    masks: Sequence[int],
    baseline_context: pd.DataFrame,
    baseline_covariates: pd.DataFrame,
    baseline_target: pd.Series | None = None,
    context_length: int,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[int, str]]:
    contexts: list[pd.DataFrame] = []
    futures: list[pd.DataFrame] = []
    item_by_mask: dict[int, str] = {}
    for mask in masks:
        scenario = copy.copy(data)
        scenario.target = data.target.copy()
        model_context = data.model_context_covariates.copy()
        model_context.index = _utc_index(model_context.index, name="scenario.model_context.index")
        covariates = data.covariates.copy()
        covariates.index = _utc_index(covariates.index, name="scenario.covariates.index")
        for group_index, group in enumerate(groups):
            if mask & (1 << group_index):
                continue
            if group.target_context:
                if baseline_target is None:
                    raise VariableAttributionError("past-price coalition lacks a target baseline")
                target_index = _utc_index(data.target.index, name="scenario.target.index")
                scenario.target = baseline_target.reindex(target_index).copy()
            for column in group.context_columns:
                if group.target_context and column == "target":
                    continue
                model_context[column] = baseline_context[column].reindex(model_context.index)
            for alias in group.aliases:
                if alias in covariates and alias in baseline_covariates:
                    covariates[alias] = baseline_covariates[alias].reindex(covariates.index)
        scenario.model_context_covariates = model_context
        scenario.covariates = covariates
        item_id = f"{data.zone}_variable_attribution_{mask:0{max(1, len(groups))}x}"
        context, future = build_live_frames(
            scenario,
            int(context_length),
            int(horizon),
            item_id,
            True,
        )
        contexts.append(context)
        if future is not None:
            futures.append(future)
        item_by_mask[int(mask)] = item_id
    context_frame = prepare_chronos_frame(
        pd.concat(contexts, ignore_index=True),
        "variable_attribution_context",
    )
    future_frame = (
        prepare_chronos_frame(
            pd.concat(futures, ignore_index=True),
            "variable_attribution_future",
        )
        if futures
        else None
    )
    return context_frame, future_frame, item_by_mask


def _predict_chronos_scenarios(
    *,
    data: Any,
    runtime: Any,
    context_frame: pd.DataFrame,
    future_frame: pd.DataFrame | None,
    item_by_mask: Mapping[int, str],
    expected_index: pd.DatetimeIndex,
    context_length: int,
    model_batch_size: int,
) -> dict[int, pd.DataFrame]:
    n_variates = len(
        [column for column in context_frame if column not in {"item_id", "timestamp"}]
    )
    raw = runtime.pipeline.predict_df(
        context_frame,
        future_df=future_frame,
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
        prediction_length=len(expected_index),
        quantile_levels=list(DEFAULT_QUANTILES),
        batch_size=max(int(model_batch_size), n_variates),
        context_length=int(context_length),
        cross_learning=False,
        validate_inputs=False,
        freq=data.frequency,
    )
    normalized = normalize_prediction_columns(raw)
    result: dict[int, pd.DataFrame] = {}
    for mask, item_id in item_by_mask.items():
        selected = normalized.loc[normalized["item_id"].eq(item_id)].copy()
        delivery = _utc_index(selected["timestamp"], name=f"scenario[{mask}].timestamp")
        selected.index = delivery
        selected = selected.sort_index()
        if not selected.index.equals(expected_index):
            raise VariableAttributionError(
                f"Chronos scenario {mask} does not cover the exact live horizon"
            )
        frame = selected.loc[:, list(QUANTILES)].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise VariableAttributionError(f"Chronos scenario {mask} is non-finite")
        result[int(mask)] = frame.astype(float)
    return result


def _fresh_scenario(
    *,
    fresh_future: pd.DataFrame,
    groups: Sequence[VariableGroup],
    mask: int,
    baseline_context: pd.DataFrame,
) -> pd.DataFrame:
    result = fresh_future.copy()
    result.index = _utc_index(result.index, name="fresh_future.index")
    for group_index, group in enumerate(groups):
        if mask & (1 << group_index):
            continue
        for column in group.future_columns:
            if column not in result:
                raise VariableAttributionError(
                    f"future attribution column is missing: {column}"
                )
            if column not in baseline_context:
                raise VariableAttributionError(
                    f"future column has no causal context baseline: {column}"
                )
            result[column] = baseline_context[column].reindex(result.index).to_numpy(float)
    return result


def _prediction_q50(frame: pd.DataFrame, *, expected_index: pd.DatetimeIndex) -> pd.Series:
    if "q50" not in frame:
        raise VariableAttributionError("prediction is missing q50")
    index = _utc_index(frame.index, name="prediction.index")
    values = pd.to_numeric(frame["q50"], errors="coerce").to_numpy(dtype=float)
    result = pd.Series(values, index=index, name="q50").sort_index()
    if not result.index.equals(expected_index) or not np.isfinite(values).all():
        raise VariableAttributionError("prediction q50 violates the live contract")
    return result


def _scenario_final_values(
    *,
    chronos: Mapping[int, pd.DataFrame],
    corrector: Any,
    fresh_future: pd.DataFrame,
    groups: Sequence[VariableGroup],
    baseline_context: pd.DataFrame,
    expected_index: pd.DatetimeIndex,
    primary: pd.Series | None,
    autonomous_weight: float,
    mkonline_weight: float,
) -> dict[str, dict[int, np.ndarray]]:
    autonomous: dict[int, np.ndarray] = {}
    blend: dict[int, np.ndarray] = {}
    primary_values: np.ndarray | None = None
    if primary is not None:
        primary_index = _utc_index(primary.index, name="mkonline_primary.index")
        primary_series = pd.Series(
            pd.to_numeric(primary, errors="coerce").to_numpy(dtype=float),
            index=primary_index,
        ).sort_index()
        if not primary_series.index.equals(expected_index) or not np.isfinite(
            primary_series.to_numpy(dtype=float)
        ).all():
            raise VariableAttributionError("MKOnline primary violates the live horizon")
        primary_values = primary_series.to_numpy(dtype=float)
    for mask, chronos_frame in chronos.items():
        fresh = _fresh_scenario(
            fresh_future=fresh_future,
            groups=groups,
            mask=mask,
            baseline_context=baseline_context,
        )
        base = chronos_frame.loc[:, list(QUANTILES)].copy()
        experts = base.rename(columns={value: f"chronos2__{value}" for value in QUANTILES})
        corrected = corrector.predict(fresh, base, experts)
        auto_q50 = _prediction_q50(corrected, expected_index=expected_index).to_numpy(float)
        autonomous[int(mask)] = auto_q50
        if primary_values is not None:
            blend[int(mask)] = (
                float(autonomous_weight) * auto_q50
                + float(mkonline_weight) * primary_values
            )
    result = {"autonomous": autonomous}
    if blend:
        result["mkonline_blend"] = blend
    return result


def _shapley_values(
    values: Mapping[int, np.ndarray],
    *,
    n_groups: int,
    orders: Sequence[Sequence[int]],
) -> np.ndarray:
    horizon = len(next(iter(values.values())))
    contributions = np.zeros((n_groups, horizon), dtype=float)
    if not orders:
        factorial = math.factorial
        denominator = float(factorial(n_groups))
        for group_index in range(n_groups):
            bit = 1 << group_index
            for mask in range(1 << n_groups):
                if mask & bit:
                    continue
                size = int(mask.bit_count())
                weight = (
                    factorial(size) * factorial(n_groups - size - 1) / denominator
                )
                contributions[group_index] += weight * (
                    values[mask | bit] - values[mask]
                )
        return contributions
    for order in orders:
        mask = 0
        for group_index in order:
            next_mask = mask | (1 << int(group_index))
            contributions[int(group_index)] += values[next_mask] - values[mask]
            mask = next_mask
    return contributions / float(len(orders))


def _official_q50(
    value: pd.DataFrame | pd.Series,
    *,
    expected_index: pd.DatetimeIndex,
) -> np.ndarray:
    if isinstance(value, pd.DataFrame):
        return _prediction_q50(value, expected_index=expected_index).to_numpy(float)
    index = _utc_index(value.index, name="official_q50.index")
    series = pd.Series(
        pd.to_numeric(value, errors="coerce").to_numpy(dtype=float), index=index
    ).sort_index()
    if not series.index.equals(expected_index) or not np.isfinite(
        series.to_numpy(dtype=float)
    ).all():
        raise VariableAttributionError("official q50 violates the live contract")
    return series.to_numpy(dtype=float)


def write_variable_attribution(
    *,
    output_dir: str | Path,
    forecast_path: str | Path,
    data: Any,
    runtime: Any,
    fresh_future: pd.DataFrame,
    corrector: Any,
    official_autonomous: pd.DataFrame | pd.Series,
    required_covariates: Sequence[str],
    context_length: int,
    model_batch_size: int,
    zone: str,
    timezone: str,
    delivery_day: str,
    official_blend: pd.DataFrame | pd.Series | None = None,
    primary: pd.Series | None = None,
    autonomous_weight: float = 1.0,
    mkonline_weight: float = 0.0,
    seed: int = 42,
    approximate_permutations: int = DEFAULT_APPROXIMATE_PERMUTATIONS,
    reproduction_tolerance_eur_mwh: float = 1e-3,
    include_past_prices: bool = True,
) -> dict[str, Any]:
    """Compute and seal a local grouped Shapley explanation of final P50.

    The caller must invoke this after writing the official forecast.  A hash is
    checked before and after attribution so the explanation cannot rewrite it.
    """

    started = time.perf_counter()
    destination = Path(output_dir).expanduser().resolve()
    forecast = Path(forecast_path).expanduser().resolve()
    if not forecast.is_file():
        raise FileNotFoundError(forecast)
    forecast_sha256 = _sha256(forecast)
    expected_index = _utc_index(fresh_future.index, name="fresh_future.index")
    historical_end = pd.Timestamp(data.target.index[-1])
    historical_end = (
        historical_end.tz_localize("UTC")
        if historical_end.tzinfo is None
        else historical_end.tz_convert("UTC")
    )
    if historical_end != expected_index[0] - pd.Timedelta(hours=1):
        raise VariableAttributionError(
            "attribution target must end exactly one hour before delivery"
        )
    groups = build_variable_groups(
        required_covariates=required_covariates,
        context_columns=data.model_context_covariates.columns,
        future_columns=fresh_future.columns,
        include_past_prices=bool(include_past_prices),
    )
    baseline_context, baseline_covariates, baseline_target = _baseline_frames(
        data=data,
        groups=groups,
        historical_end=historical_end,
        timezone=timezone,
    )
    masks, orders, method = _coalition_masks(
        len(groups),
        seed=int(seed),
        approximate_permutations=int(approximate_permutations),
    )
    context_frame, future_frame, item_by_mask = _scenario_frames(
        data=data,
        groups=groups,
        masks=masks,
        baseline_context=baseline_context,
        baseline_covariates=baseline_covariates,
        baseline_target=baseline_target,
        context_length=int(context_length),
        horizon=len(expected_index),
    )
    chronos = _predict_chronos_scenarios(
        data=data,
        runtime=runtime,
        context_frame=context_frame,
        future_frame=future_frame,
        item_by_mask=item_by_mask,
        expected_index=expected_index,
        context_length=int(context_length),
        model_batch_size=int(model_batch_size),
    )
    final_values = _scenario_final_values(
        chronos=chronos,
        corrector=corrector,
        fresh_future=fresh_future,
        groups=groups,
        baseline_context=baseline_context,
        expected_index=expected_index,
        primary=primary,
        autonomous_weight=float(autonomous_weight),
        mkonline_weight=float(mkonline_weight),
    )
    official: dict[str, np.ndarray] = {
        "autonomous": _official_q50(
            official_autonomous, expected_index=expected_index
        )
    }
    if official_blend is not None:
        if primary is None or "mkonline_blend" not in final_values:
            raise VariableAttributionError(
                "official blend requires MKOnline primary and blend scenarios"
            )
        official["mkonline_blend"] = _official_q50(
            official_blend, expected_index=expected_index
        )
    full_mask = (1 << len(groups)) - 1
    rows: list[dict[str, Any]] = []
    max_base_error = 0.0
    max_reconstruction_error = 0.0
    local_index = expected_index.tz_convert(timezone)
    for variant, scenario_values in final_values.items():
        if variant not in official:
            continue
        contributions = _shapley_values(
            scenario_values,
            n_groups=len(groups),
            orders=orders,
        )
        baseline = scenario_values[0]
        full = scenario_values[full_mask]
        base_error = float(np.max(np.abs(full - official[variant])))
        reconstruction = baseline + contributions.sum(axis=0)
        reconstruction_error = float(
            np.max(np.abs(reconstruction - official[variant]))
        )
        max_base_error = max(max_base_error, base_error)
        max_reconstruction_error = max(
            max_reconstruction_error, reconstruction_error
        )
        absolute_means = np.mean(np.abs(contributions), axis=1)
        total = float(absolute_means.sum())
        weights = (
            np.zeros(len(groups), dtype=float)
            if total <= 0.0
            else 100.0 * absolute_means / total
        )
        for group_index, group in enumerate(groups):
            for hour_index, timestamp in enumerate(expected_index):
                contribution = float(contributions[group_index, hour_index])
                rows.append(
                    {
                        "delivery_start_utc": str(timestamp),
                        "delivery_start_local": str(local_index[hour_index]),
                        "variant": variant,
                        "variable_key": group.key,
                        "variable_label": group.label,
                        "baseline_reference": group.baseline_reference,
                        "forecast_q50": float(official[variant][hour_index]),
                        "counterfactual_q50": float(baseline[hour_index]),
                        "contribution_eur_mwh": contribution,
                        "absolute_contribution_eur_mwh": abs(contribution),
                        "weight_pct": float(weights[group_index]),
                    }
                )
    if max_base_error > float(reproduction_tolerance_eur_mwh):
        raise VariableAttributionError(
            "attribution rerun does not reproduce frozen forecast: "
            f"{max_base_error:.6f} EUR/MWh"
        )
    if max_reconstruction_error > float(reproduction_tolerance_eur_mwh):
        raise VariableAttributionError(
            "Shapley reconstruction does not reproduce frozen forecast: "
            f"{max_reconstruction_error:.6f} EUR/MWh"
        )
    if _sha256(forecast) != forecast_sha256:
        raise VariableAttributionError("attribution modified the frozen forecast")
    frame = pd.DataFrame(rows)
    expected_columns = [
        "delivery_start_utc",
        "delivery_start_local",
        "variant",
        "variable_key",
        "variable_label",
        "baseline_reference",
        "forecast_q50",
        "counterfactual_q50",
        "contribution_eur_mwh",
        "absolute_contribution_eur_mwh",
        "weight_pct",
    ]
    frame = frame.loc[:, expected_columns]
    destination.mkdir(parents=True, exist_ok=True)
    hourly_path = destination / VARIABLE_ATTRIBUTION_HOURLY
    audit_path = destination / VARIABLE_ATTRIBUTION_AUDIT
    frame.to_csv(hourly_path, index=False, compression="gzip")
    architecture_weights = {
        "autonomous": {"autonomous": 1.0, "mkonline_primary": 0.0}
    }
    if "mkonline_blend" in official:
        architecture_weights["mkonline_blend"] = {
            "autonomous": float(autonomous_weight),
            "mkonline_primary": float(mkonline_weight),
        }
    audit: dict[str, Any] = {
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "status": "complete",
        "zone": str(zone).upper(),
        "timezone": str(timezone),
        "delivery_day": str(delivery_day),
        "quantile": "q50",
        "method": method,
        "scope": "local_delivery_day",
        "expected_hours": int(len(expected_index)),
        "variants": list(official),
        "architecture_weights": architecture_weights,
        "groups": [
            {
                "key": group.key,
                "label": group.label,
                "aliases": list(group.aliases),
                "context_columns": list(group.context_columns),
                "future_columns": list(group.future_columns),
                "baseline_reference": group.baseline_reference,
                "target_context": group.target_context,
            }
            for group in groups
        ],
        "baseline": {
            "policy": BASELINE_REFERENCE,
            "lookback_days": 56,
            "historical_end_utc": str(historical_end),
            "uses_post_cutoff_data": False,
            "fixed_background_inputs": ["calendar"] if include_past_prices else ["historical_target_price", "calendar"],
            "past_price_scope": "historical_Chronos_target_and_consistent_target_derived_features" if include_past_prices else "fixed_background",
            "past_price_baseline_uses_future_targets": False,
            "past_price_baseline_causality": "known_at_local_attribution_origin_not_per_historical_context_timestamp",
        },
        "past_prices_included": bool(include_past_prices),
        "calendar_attributed": False,
        "parameters_frozen": {
            "chronos_weights": True, "residual_corrector": True, "blend_weights": True,
            "prediction_only": True, "refitting_performed": False,
        },
        "kalman_scope": {"included": False, "reason": "Kalman states, refits and governor are not replayed by this attribution module"},
        "scenario_count": int(len(masks)),
        "exact_shapley_max_groups": EXACT_SHAPLEY_MAX_GROUPS,
        "scenario_budget": MAX_ATTRIBUTION_SCENARIOS,
        "requested_approximate_permutations": int(approximate_permutations),
        "approximate_permutations": int(len(orders)),
        "seed": int(seed),
        "forecast_sha256": forecast_sha256,
        "attribution_hourly_sha256": _sha256(hourly_path),
        "used_for_prediction": False,
        "storm_used": False,
        "forecast_modified": False,
        "max_base_prediction_error_eur_mwh": max_base_error,
        "max_reconstruction_error_eur_mwh": max_reconstruction_error,
        "reproduction_tolerance_eur_mwh": float(
            reproduction_tolerance_eur_mwh
        ),
        "elapsed_seconds": float(time.perf_counter() - started),
        "interpretation": (
            "Grouped predictive Shapley attribution relative to a causal "
            "historical reference curve; not an economic causal effect."
        ),
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if _sha256(forecast) != forecast_sha256:
        hourly_path.unlink(missing_ok=True)
        audit_path.unlink(missing_ok=True)
        raise VariableAttributionError("attribution modified the frozen forecast")
    return audit


def remove_variable_attribution_artifacts(directory: str | Path) -> None:
    """Remove only optional attribution outputs after a shadow failure."""

    root = Path(directory).expanduser().resolve()
    for name in (VARIABLE_ATTRIBUTION_HOURLY, VARIABLE_ATTRIBUTION_AUDIT):
        (root / name).unlink(missing_ok=True)


__all__ = [
    "ATTRIBUTION_SCHEMA_VERSION",
    "BASELINE_REFERENCE",
    "PAST_PRICE_GROUP_KEY",
    "MAX_ATTRIBUTION_SCENARIOS",
    "VARIABLE_ATTRIBUTION_AUDIT",
    "VARIABLE_ATTRIBUTION_HOURLY",
    "VariableAttributionError",
    "VariableGroup",
    "build_variable_groups",
    "remove_variable_attribution_artifacts",
    "write_variable_attribution",
]
