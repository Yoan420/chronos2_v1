"""Opt-in nuclear input replay, with no dependency on live/LoRA archives.

The caller supplies already materialized, cutoff-audited PIT ZoneData. This
module never fetches or refreshes data. It recomputes Chronos with the nuclear
forecast, refits the existing residual recipe daily on earlier delivery days,
then gives precisely that upstream to the existing rolling-365 Kalman engine.
The first 365 replay days are a diagnostic warmup, not a nested validation set.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from datetime import date, timedelta
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from .chronos_adapter import (
    execute_grouped_chronos_backtest,
    generate_delivery_plans,
    make_existing_forecasting_executor,
    run_existing_live_forecast,
)
from .historical_price_replay import generate_checkpointed_price_replay, replay_identity
from .kalman_covariates import BASE_RESIDUAL_LOAD_COVARIATES, KalmanCovariateConfig
from .kalman_residual import (
    KalmanOperationalView,
    KalmanResidualConfig,
    build_operational_kalman_view,
)
from chronos2_modular.common import deep_get, set_reproducibility


NUCLEAR_ALIAS = "fr_nuclear_generation_fcst_gw"
NUCLEAR_SERIES = "power.fr.generation.nuclear.gw.fcst"
NUCLEAR_KNOWN_COLUMN = f"known_{NUCLEAR_ALIAS}_oracle"
HISTORY_DAYS = 730
EVALUATION_DAYS = 365
RESIDUAL_LOOKBACK_DAYS = 365
QUANTILES = ("q10", "q50", "q90")
ZONE_TIMEZONES = {
    "FR": "Europe/Paris", "DE": "Europe/Berlin", "BE": "Europe/Brussels",
    "NL": "Europe/Amsterdam", "ES": "Europe/Madrid",
}


class NuclearForecastError(ValueError):
    """The opt-in causal input or output contract is invalid."""


@dataclass(frozen=True)
class NuclearForecastResult:
    raw_history: pd.DataFrame
    residual_statistics: pd.DataFrame
    source_forecast: pd.DataFrame
    covariates: pd.DataFrame
    residual_daily_audit: pd.DataFrame
    kalman_view: KalmanOperationalView
    audit: Mapping[str, Any]


def _digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _digest_frame(value: pd.DataFrame | pd.Series) -> str:
    digest = hashlib.sha256()
    frame = value.to_frame() if isinstance(value, pd.Series) else value
    digest.update(_digest_json([(str(c), str(t)) for c, t in frame.dtypes.items()]).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _factory_cache_identity(factory: Callable[..., Any] | None) -> Any:
    if factory is None:
        return "configured"
    # Test/research hooks must not reuse another implementation's predictions.
    classes = inspect.getmro(factory) if inspect.isclass(factory) else (factory,)
    sources = {}
    for item in classes:
        if item is object:
            continue
        source = inspect.getsourcefile(item)
        if source is None:
            raise NuclearForecastError("Persistent residual caching requires a file-backed factory")
        sources[str(Path(source).resolve())] = _file_sha256(Path(source))
    return {"module": factory.__module__, "name": factory.__qualname__, "sources": sources,
            "closure": [repr(cell.cell_contents) for cell in (getattr(factory, "__closure__", None) or ())]}


def _load_result_cache(directory: Path, identity: Mapping[str, Any]) -> dict[str, pd.DataFrame] | None:
    if not directory.exists():
        return None
    try:
        manifest = json.loads((directory / "cache_manifest.json").read_text(encoding="utf-8"))
        if manifest["identity"] != identity or manifest["schema_version"] != 1:
            raise NuclearForecastError("Cached nuclear result identity mismatch")
        files = manifest["files"]
        if not isinstance(files, dict) or not files:
            raise NuclearForecastError("Cached nuclear result has no files")
        result = {}
        for filename, digest in files.items():
            if Path(filename).name != filename or not filename.endswith(".parquet"):
                raise NuclearForecastError("Unsafe nuclear result cache filename")
            source = directory / filename
            if not source.is_file() or _file_sha256(source) != digest:
                raise NuclearForecastError(f"Cached nuclear result checksum mismatch: {filename}")
            result[filename.removesuffix(".parquet")] = pd.read_parquet(source)
        return result
    except (KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        raise NuclearForecastError(f"Incomplete nuclear result cache: {directory}") from exc


def _save_result_cache(
    directory: Path, identity: Mapping[str, Any], frames: Mapping[str, pd.DataFrame],
) -> None:
    """Publish one immutable cache directory; never rewrite an existing seal."""
    if directory.exists():
        raise NuclearForecastError(f"Refusing to replace a sealed result cache: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
    try:
        files = {}
        for name, frame in frames.items():
            filename = f"{name}.parquet"
            if Path(filename).name != filename:
                raise NuclearForecastError("Unsafe nuclear result cache name")
            path = staging / filename
            frame.to_parquet(path, index=True)
            files[filename] = _file_sha256(path)
        (staging / "cache_manifest.json").write_text(
            json.dumps({"schema_version": 1, "identity": dict(identity), "files": files},
                       ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
        )
        os.rename(staging, directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _utc_frame(value: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame) or value.empty:
        raise NuclearForecastError(f"{name}: nonempty DataFrame required")
    index = value.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise NuclearForecastError(f"{name}: timezone-aware index required")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise NuclearForecastError(f"{name}: unique increasing index required")
    result = value.copy()
    result.index = index.tz_convert("UTC")
    return result


def _quantiles(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not set(QUANTILES).issubset(frame.columns):
        raise NuclearForecastError(f"{name}: q10/q50/q90 required")
    result = frame.loc[:, list(QUANTILES)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise NuclearForecastError(f"{name}: finite predictions required")
    if ((result.q10 > result.q50) | (result.q50 > result.q90)).any():
        raise NuclearForecastError(f"{name}: crossing quantiles")
    return result


def _validate_origins(frame: pd.DataFrame, timezone: str) -> None:
    if "forecast_origin_utc" not in frame:
        raise NuclearForecastError("Raw forecasts require forecast_origin_utc")
    parsed = [pd.Timestamp(value) for value in frame.forecast_origin_utc]
    if any(value.tzinfo is None or pd.isna(value) for value in parsed):
        raise NuclearForecastError("Raw forecast origins require timezone-aware timestamps")
    days = frame.index.tz_convert(timezone).date
    expected = {
        day: (pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=8))
        .tz_localize(timezone).tz_convert("UTC")
        for day in pd.Index(days).unique()
    }
    if any(value.tz_convert("UTC") != expected[day] for value, day in zip(parsed, days)):
        raise NuclearForecastError("Raw forecast origins must equal civil D-1 08:00")


def nuclear_kalman_covariate_config() -> KalmanCovariateConfig:
    """Keep standard candidates/aggregates; add nuclear to a consumed group."""
    base = KalmanCovariateConfig()
    result = replace(
        base,
        input_columns=(*base.input_columns, NUCLEAR_ALIAS),
        groups={name: (*columns, NUCLEAR_ALIAS) for name, columns in base.groups.items()},
        history_missing_policy="complete_trailing",
        minimum_history_coverage=1.0,
        require_future_complete=True,
    )
    result.validate()
    return result


def _kalman_filter_configuration(resolved: Mapping[str, Any]) -> tuple[KalmanResidualConfig, dict[str, Any]]:
    parameters = deep_get(resolved, "nuclear_experiment.filter_parameters", {})
    if not isinstance(parameters, Mapping):
        raise NuclearForecastError("nuclear_experiment.filter_parameters must be a mapping")
    parameters = dict(parameters)
    if "candidate_kinds" in parameters:
        parameters["candidate_kinds"] = tuple(parameters["candidate_kinds"])
    try:
        result = KalmanResidualConfig(**parameters)
        result.validate()
    except (TypeError, ValueError) as exc:
        raise NuclearForecastError(f"Invalid nuclear Kalman filter parameters: {exc}") from exc
    if "linear_market" not in result.candidate_kinds:
        raise NuclearForecastError("Nuclear Kalman parameters must retain the linear_market candidate")
    return result, parameters


def _residual_recipe_preflight(factory: Callable[[], Any]) -> int:
    from .models.residual_corrector import ResidualMetaFeatureBuilder
    prototype = factory()
    minimum_rows = int(prototype.min_training_rows)
    if minimum_rows < 2:
        raise NuclearForecastError("Residual min_training_rows must be >=2")
    options = getattr(prototype, "feature_builder_options", {})
    builder = getattr(prototype, "feature_builder", None) or ResidualMetaFeatureBuilder(**options)
    if not builder.exclude_historical_prices or builder._is_excluded(NUCLEAR_KNOWN_COLUMN):
        raise NuclearForecastError("Residual builder must exclude historical prices and retain nuclear")
    return minimum_rows


def _covariates(data: Any, expected: pd.DatetimeIndex) -> pd.DataFrame:
    context = _utc_frame(data.model_context_covariates, name="model_context_covariates")
    if NUCLEAR_ALIAS not in context or NUCLEAR_KNOWN_COLUMN not in context:
        raise NuclearForecastError("Nuclear must enter both base context and known future")
    if NUCLEAR_KNOWN_COLUMN not in data.known_future_columns:
        raise NuclearForecastError("Nuclear is not a declared known-future model input")
    selected: dict[str, pd.Series] = {}
    for alias in (*BASE_RESIDUAL_LOAD_COVARIATES, NUCLEAR_ALIAS):
        known = f"known_{alias}_oracle"
        if known not in context:
            raise NuclearForecastError(f"Missing PIT known-future input: {known}")
        values = pd.to_numeric(context[known].reindex(expected), errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise NuclearForecastError(f"Incomplete PIT history/future: {alias}")
        selected[alias] = values
    nuclear_base = pd.to_numeric(context[NUCLEAR_ALIAS].reindex(expected), errors="coerce")
    if not np.isfinite(nuclear_base.to_numpy(dtype=float)).all():
        raise NuclearForecastError("Incomplete nuclear base context")
    if not np.allclose(nuclear_base, selected[NUCLEAR_ALIAS], rtol=0, atol=1e-9):
        raise NuclearForecastError("Nuclear context and known-future vintages diverge")
    if (selected[NUCLEAR_ALIAS] < 0).any() or (selected[NUCLEAR_ALIAS] > 100).any():
        raise NuclearForecastError("Nuclear generation outside plausible GW range [0, 100]")
    result = pd.DataFrame(selected, index=expected)
    result.index.name = "timestamp"
    return result.reset_index()


def causal_residual_replay(
    *,
    raw_history: pd.DataFrame,
    raw_future: pd.DataFrame,
    features: pd.DataFrame,
    timezone: str,
    delivery_day: date,
    residual_factory: Callable[[], Any],
    output_start_day: date | None = None,
    daily_cache: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit every D only on D-365..D-1; no reuse of incumbent corrected rows.

    This smaller pure boundary is independently testable without Chronos or
    Kalman. Identity output before minimum_training_rows is always audited.
    """
    history = _utc_frame(raw_history, name="raw_history")
    future = _utc_frame(raw_future, name="raw_future")
    X = _utc_frame(features, name="features")
    _validate_origins(history, timezone)
    _validate_origins(future, timezone)
    base = _quantiles(history, name="raw_history")
    future_base = _quantiles(future, name="raw_future")
    if "actual" not in history:
        raise NuclearForecastError("raw_history.actual required")
    actual = pd.to_numeric(history.actual, errors="coerce")
    if not np.isfinite(actual.to_numpy(dtype=float)).all():
        raise NuclearForecastError("Historical labels must be finite")
    if "actual" in future and future.actual.notna().any():
        raise NuclearForecastError("Future observations must never enter the replay")
    if not (history.index.tz_convert(timezone).date < delivery_day).all():
        raise NuclearForecastError("History includes a future delivery label")
    expected_future = generate_delivery_plans(
        delivery_day, delivery_day, timezone=timezone, forecast_origin_local_time="08:00",
    )[0].delivery_index_utc
    if not future.index.equals(expected_future):
        raise NuclearForecastError("Future must be one complete physical delivery day")
    replay_index = history.index.append(future.index)
    if len(replay_index.difference(X.index)):
        raise NuclearForecastError("Residual features do not cover all replay hours")
    if NUCLEAR_KNOWN_COLUMN not in X:
        raise NuclearForecastError("Nuclear missing from residual features")
    nuclear = pd.to_numeric(X.loc[replay_index, NUCLEAR_KNOWN_COLUMN], errors="coerce")
    if not np.isfinite(nuclear.to_numpy(dtype=float)).all():
        raise NuclearForecastError("Residual nuclear features are incomplete")

    days = np.asarray(history.index.tz_convert(timezone).date, dtype=object)
    unique_days = tuple(pd.Index(days).unique())
    if not unique_days:
        raise NuclearForecastError("Residual history is empty")
    expected_history = pd.date_range(
        pd.Timestamp(unique_days[0], tz=timezone), pd.Timestamp(delivery_day, tz=timezone),
        freq="h", inclusive="left",
    ).tz_convert("UTC")
    if not history.index.equals(expected_history):
        raise NuclearForecastError("Residual history must contain complete contiguous days")
    # The existing factory can retain model-specific feature exclusions. Check
    # those before expensive fits, and check the learned schema after fitting.
    minimum_rows = _residual_recipe_preflight(residual_factory)
    experts = base.rename(columns={q: f"chronos2__{q}" for q in QUANTILES})
    future_experts = future_base.rename(columns={q: f"chronos2__{q}" for q in QUANTILES})
    corrected = base.copy()
    corrected_future = future_base.copy()
    records: list[dict[str, Any]] = []
    evaluation_start = delivery_day - timedelta(days=EVALUATION_DAYS)
    output_days = tuple(day for day in unique_days if output_start_day is None or day >= output_start_day)
    for day in (*output_days, delivery_day):
        mask = (days >= day - timedelta(days=RESIDUAL_LOOKBACK_DAYS)) & (days < day)
        training_index = history.index[mask]
        is_future = day == delivery_day
        predicted_index = future.index if is_future else history.index[days == day]
        prediction_base = future_base if is_future else base.loc[predicted_index]
        prediction_experts = future_experts if is_future else experts.loc[predicted_index]
        source = "identity_chronos_cold_start"
        fitted_features: tuple[str, ...] = ()
        cache_hit = False
        if len(training_index) >= minimum_rows:
            cached = (daily_cache.load(day, training_index, predicted_index, prediction_base)
                      if daily_cache is not None else None)
            if cached is None:
                fitted = residual_factory()
                fitted.fit(
                    X.loc[training_index], actual.loc[training_index],
                    base.loc[training_index], experts.loc[training_index],
                )
                fitted_features = tuple(map(str, fitted.feature_columns_))
                predicted = fitted.predict(X.loc[predicted_index], prediction_base, prediction_experts)
            else:
                predicted, fitted_features = cached
                cache_hit = True
            if not any(NUCLEAR_ALIAS in column for column in fitted_features):
                raise NuclearForecastError("Fitted residual recipe dropped the nuclear input")
            if not predicted.index.equals(predicted_index):
                raise NuclearForecastError("Residual output index differs from requested day")
            predicted = _quantiles(predicted, name=f"residual {day}")
            if daily_cache is not None and not cache_hit:
                daily_cache.store(day, training_index, predicted_index, prediction_base,
                                  predicted, fitted_features)
            if is_future:
                corrected_future.loc[:, list(QUANTILES)] = predicted.to_numpy()
            else:
                corrected.loc[predicted_index, list(QUANTILES)] = predicted.to_numpy()
            source = "daily_prequential_refit"
        records.append({
            "delivery_day": day.isoformat(),
            "phase": "future" if is_future else "evaluation" if day >= evaluation_start else "diagnostic_warmup",
            "generation_source": source,
            "training_rows": len(training_index), "minimum_training_rows": minimum_rows,
            "training_lookback_days": RESIDUAL_LOOKBACK_DAYS,
            "fit_start_day": str(training_index[0].tz_convert(timezone).date()) if len(training_index) else None,
            "fit_end_day": str(training_index[-1].tz_convert(timezone).date()) if len(training_index) else None,
            "forecast_hours": len(predicted_index), "causality_violations": 0,
            "nuclear_feature_used": bool(fitted_features),
            "residual_feature_columns": list(fitted_features),
            **({"daily_cache_hit": cache_hit} if daily_cache is not None else {}),
        })
    statistics = pd.DataFrame({"actual": actual}, index=history.index)
    forecast = pd.DataFrame(index=future.index)
    for q in QUANTILES:
        statistics[f"chronos2__{q}"] = base[q]
        statistics[f"residual_corrected__{q}"] = corrected[q]
        forecast[f"chronos2__{q}"] = future_base[q]
        forecast[f"residual_corrected__{q}"] = corrected_future[q]
        forecast[q] = corrected_future[q]
    for output, original in ((statistics, history), (forecast, future)):
        output["residual_correction"] = output.residual_corrected__q50 - output.chronos2__q50
        if "forecast_origin_utc" not in original:
            raise NuclearForecastError("Raw forecasts require forecast_origin_utc")
        output["forecast_origin_utc"] = original.forecast_origin_utc
        output.index.name = "delivery_start_utc"
    forecast["price_eur_mwh"] = forecast.residual_corrected__q50
    if output_start_day is not None:
        statistics = statistics.loc[statistics.index.tz_convert(timezone).date >= output_start_day]
    return statistics.reset_index(), forecast.reset_index(), pd.DataFrame(records)


def run_nuclear_forecast(
    *,
    config: Mapping[str, Any], data: Any, zone: str, delivery_day: str | date,
    workdir: str | Path, device: str = "auto", threads: int = 4, workers: int = 1,
    runtime_factory: Callable[..., Any] | None = None,
    forecasting_module: Any | None = None,
    residual_factory: Callable[[], Any] | None = None,
    kalman_builder: Callable[..., Any] | None = None,
) -> NuclearForecastResult:
    """Execute the explicit nuclear challenger; write only disposable workdir.

    Runtime/forecasting/residual/Kalman hooks exist for lightweight unit tests.
    No production checkpoint, model activation or checksum seal is rewritten.
    """
    from run_chronos2_hourly import _feature_inputs, _residual_corrector_factory
    from chronos2_modular.forecasting import load_model

    code = str(zone).upper()
    if code not in ZONE_TIMEZONES or str(data.zone).upper() != code:
        raise NuclearForecastError("Zone does not match prepared data")
    timezone = ZONE_TIMEZONES[code]
    if str(data.timezone) != timezone:
        raise NuclearForecastError("Prepared data timezone does not match zone")
    day = pd.Timestamp(delivery_day).date()
    if device not in {"auto", "cpu", "cuda"} or threads < 1 or workers < 1:
        raise NuclearForecastError("Invalid device/threads/workers")
    resolved = copy.deepcopy(dict(config))
    if deep_get(resolved, "model.model_id", "amazon/chronos-2") != "amazon/chronos-2":
        raise NuclearForecastError("Only the unchanged amazon/chronos-2 base model is supported")
    if any(token in str(column).casefold()
           for column in data.model_context_covariates.columns for token in ("storm", "mkonline")):
        raise NuclearForecastError("Competing forecasts are forbidden as nuclear experiment inputs")
    nuclear_spec = deep_get(resolved, f"zones.{code}.covariates.{NUCLEAR_ALIAS}", {})
    if (not isinstance(nuclear_spec, Mapping) or nuclear_spec.get("enabled") is not True
            or nuclear_spec.get("source") != "pit_parquet"
            or nuclear_spec.get("series") != NUCLEAR_SERIES):
        raise NuclearForecastError("Explicit generation-GW PIT nuclear configuration required")
    target, _, future_covariates, features = _feature_inputs(data, resolved)
    if NUCLEAR_KNOWN_COLUMN not in features:
        raise NuclearForecastError("Nuclear excluded by residual feature column selection")
    origin_time = str(deep_get(resolved, "data.forecast_origin_local_time", "08:00"))
    if origin_time != "08:00":
        raise NuclearForecastError("Nuclear experiment requires civil D-1 08:00 cutoff")
    plans = generate_delivery_plans(
        day - timedelta(days=HISTORY_DAYS), day - timedelta(days=1),
        forecast_origin_local_time=origin_time, timezone=timezone,
    )
    future_plan = generate_delivery_plans(
        day, day, forecast_origin_local_time=origin_time, timezone=timezone,
    )[0]
    if target.index[-1] != future_plan.delivery_index_utc[0] - pd.Timedelta(hours=1):
        raise NuclearForecastError("Target must end before requested future day")
    if not future_covariates.index.equals(future_plan.delivery_index_utc):
        raise NuclearForecastError("Prepared future must be the requested 23/24/25h day")
    history_index = plans[0].delivery_index_utc
    for plan in plans[1:]:
        history_index = history_index.append(plan.delivery_index_utc)
    if len(history_index.difference(target.index)):
        raise NuclearForecastError("730 complete historical delivery days required")
    context_length = int(deep_get(resolved, "model.context_length", 2048))
    if context_length < 1 or int(target.index.get_indexer([history_index[0]])[0]) < context_length:
        raise NuclearForecastError("Insufficient Chronos context before first replay day")
    covariates = _covariates(data, history_index.append(future_plan.delivery_index_utc))
    if NUCLEAR_KNOWN_COLUMN not in features or not np.isfinite(
        features.loc[history_index.append(future_plan.delivery_index_utc), NUCLEAR_KNOWN_COLUMN]
    ).all():
        raise NuclearForecastError("Nuclear residual history/future must be complete")
    residual_options = resolved.setdefault("hourly", {}).setdefault("residual_correction", {})
    residual_options["thread_count"] = int(threads)
    configured_factory, base_model = _residual_corrector_factory(resolved, timezone=timezone)
    if configured_factory is None or base_model != "chronos2":
        raise NuclearForecastError("Enabled Chronos-based residual recipe required")
    selected_residual_factory = residual_factory or configured_factory
    minimum_rows = _residual_recipe_preflight(selected_residual_factory)
    if minimum_rows > sum(plan.horizon for plan in plans[-RESIDUAL_LOOKBACK_DAYS:]):
        raise NuclearForecastError("Residual minimum rows cannot be met within rolling365 history")
    kalman_filter_config, filter_parameters = _kalman_filter_configuration(resolved)
    workspace = Path(workdir).expanduser().resolve()
    project = Path(str(deep_get(resolved, "data.project_root", "."))).expanduser().resolve()
    if workspace == project or any(
        workspace == protected or workspace.is_relative_to(protected)
        for protected in (project / "runs" / "live", project / "runs" / "exports", project / "runs" / "cache")
    ) or (workspace / "artifact_checksums.json").exists():
        raise NuclearForecastError("Workdir must be an isolated unsealed experimental directory")
    # The initial warmup belongs to a persistent epoch. Moving it daily would
    # change otherwise identical old residual predictions and invalidate Kalman.
    # Keep up to 1095 raw days to fit the earliest of the 730 reported days.
    from .nuclear_incremental import prepare_incremental_settings
    epoch_config = copy.deepcopy(dict(config))
    incremental_dir, history_anchor, raw_start = prepare_incremental_settings(epoch_config, day)
    if incremental_dir is not None:
        resolved.setdefault("nuclear_experiment", {}).update(epoch_config["nuclear_experiment"])
        plans = generate_delivery_plans(raw_start, day - timedelta(days=1),
                                       forecast_origin_local_time=origin_time, timezone=timezone)
        if int(target.index.get_indexer([plans[0].delivery_index_utc[0]])[0]) < context_length:
            raise NuclearForecastError("Incremental history/context missing; prepare the anchored history first")
        raw_index = plans[0].delivery_index_utc
        for plan in plans[1:]:
            raw_index = raw_index.append(plan.delivery_index_utc)
        if len(raw_index.difference(target.index)) or len(raw_index.difference(features.index)):
            raise NuclearForecastError("Incomplete anchored raw history for incremental residual fits")
    prepared_hashes = {
        "target": _digest_frame(target),
        "model_context_covariates": _digest_frame(data.model_context_covariates),
        "covariates": _digest_frame(data.covariates),
        "residual_features": _digest_frame(features),
        "resolved_config": _digest_json(resolved),
        "known_future_columns": _digest_json(list(data.known_future_columns)),
        "engine_file_sha256": _file_sha256(Path(__file__).resolve()),
    }
    identity = replay_identity(
        plans, zone=code, timezone=timezone,
        model_id=str(deep_get(resolved, "model.model_id", "amazon/chronos-2")),
        model_revision=str(deep_get(resolved, "model.revision", "configured_local_checkpoint")),
        residual_load_source="saturn_with_nuclear_generation_gw",
        source_hashes=prepared_hashes,
        feature_schema_sha256=_digest_json(list(data.model_context_covariates.columns)),
        execution_signature={"engine": "nuclear_forecast_v1", "context_length": context_length,
                             "nuclear_alias": NUCLEAR_ALIAS, "units": "GW"},
    )
    result_cache_identity = {"replay_identity": identity, "delivery_day": day.isoformat()}
    result_cache_root = workspace / "cache" / "nuclear_result" / _digest_json(result_cache_identity)
    raw_future_cache_path = result_cache_root / "raw_future"
    residual_cache_path = result_cache_root / "residual"
    cached_future = _load_result_cache(raw_future_cache_path, result_cache_identity)
    cached_residual = _load_result_cache(residual_cache_path, result_cache_identity)
    set_reproducibility(int(deep_get(resolved, "model.seed", 42)))
    import torch
    torch.set_num_threads(int(threads))
    runtime = None
    def get_runtime():
        nonlocal runtime
        if runtime is None:
            runtime = (runtime_factory or load_model)(
                resolved, device, bool(deep_get(resolved, "model.local_files_only", True)),
            )
        return runtime

    def execute_chunk(chunk):
        executor = make_existing_forecasting_executor(
            data=data, runtime=get_runtime(), context_length=context_length,
            origin_batch_size=int(deep_get(resolved, "model.origin_batch_size", 12)),
            model_batch_size=int(deep_get(resolved, "model.model_batch_size", 128)),
            with_covariates=True, variant="nuclear_input_experiment_oof",
            forecasting_module=forecasting_module,
        )
        return execute_grouped_chronos_backtest(chunk, executor)

    daily_chronos = None
    if incremental_dir is not None:
        from .nuclear_daily_cache import NuclearDailyChronosCache
        daily_chronos = NuclearDailyChronosCache(
            incremental_dir / "chronos", data=data, config=resolved,
            context_length=context_length, device=device,
            model_batch_size=int(deep_get(resolved, "model.model_batch_size", 128)),
            origin_batch_size=int(deep_get(resolved, "model.origin_batch_size", 12)),
            forecasting_module=forecasting_module, execution_signature={"threads": int(threads)},
        )

    raw_checkpoint = workspace / "checkpoints" / "nuclear_chronos_oof.csv.gz"
    checkpoint_resumed = raw_checkpoint.exists()
    raw_history = generate_checkpointed_price_replay(
        plans, target=target, execute_chunk=(
            (lambda chunk: daily_chronos.resolve_history(chunk, execute_chunk))
            if daily_chronos is not None else execute_chunk),
        output_path=raw_checkpoint,
        identity=identity, chunk_days=31, resume=True,
    )
    if daily_chronos is not None and checkpoint_resumed:
        # CSV checkpoints are progress markers; their float32 values are read
        # back as rounded float64. Recover the exact daily payloads before any
        # residual fit, so stopping/restarting cannot alter labels or forecasts.
        raw_history = daily_chronos.resolve_history(plans, execute_chunk)
    def execute_future():
        return run_existing_live_forecast(
            future_plan, data=data, runtime=get_runtime(), context_length=context_length,
            model_batch_size=int(deep_get(resolved, "model.model_batch_size", 128)),
            with_covariates=True, variant="nuclear_input_experiment_live",
            forecasting_module=forecasting_module,
        )
    if cached_future is None:
        raw_future = (daily_chronos.resolve_future(future_plan, execute_future)
                      if daily_chronos is not None else execute_future())
        _save_result_cache(raw_future_cache_path, result_cache_identity, {"raw_future": raw_future})
    else:
        if set(cached_future) != {"raw_future"}:
            raise NuclearForecastError("Unexpected raw future cache members")
        raw_future = cached_future["raw_future"]
    daily_residual = None
    if incremental_dir is not None:
        from .nuclear_residual_cache import ResidualDayCache
        from .models.residual_corrector import ResidualMetaFeatureBuilder
        prototype = selected_residual_factory()
        builder = (getattr(prototype, "feature_builder", None) or
                   ResidualMetaFeatureBuilder(**getattr(prototype, "feature_builder_options", {})))
        # Price rolling statistics are excluded by this recipe. Pandas may
        # change their last floating-point bits when its context prefix moves;
        # they must not invalidate a fit which never consumes those values.
        cache_features = features.loc[:, [column for column in features if not builder._is_excluded(column)]]
        daily_residual = ResidualDayCache(
            incremental_dir / "residual", {
                "recipe": residual_options, "timezone": timezone,
                "factory": _factory_cache_identity(residual_factory),
                "engine_sha256": prepared_hashes["engine_file_sha256"],
                "input_schema": [(str(column), str(dtype)) for column, dtype in features.dtypes.items()],
            }, features=cache_features, raw=raw_history, timezone=timezone,
        )
    if cached_residual is None:
        statistics, forecast, daily_audit = causal_residual_replay(
            raw_history=raw_history, raw_future=raw_future, features=features,
            timezone=timezone, delivery_day=day, residual_factory=selected_residual_factory,
            output_start_day=day - timedelta(days=HISTORY_DAYS), daily_cache=daily_residual,
        )
        _save_result_cache(residual_cache_path, result_cache_identity, {
            "statistics": statistics, "forecast": forecast, "daily_audit": daily_audit,
        })
    else:
        if set(cached_residual) != {"statistics", "forecast", "daily_audit"}:
            raise NuclearForecastError("Unexpected residual cache members")
        statistics = cached_residual["statistics"]
        forecast = cached_residual["forecast"]
        daily_audit = cached_residual["daily_audit"]
    kalman_covariates = nuclear_kalman_covariate_config()
    kalman = (kalman_builder or build_operational_kalman_view)(
        statistics=statistics.copy(deep=True), source_forecast=forecast.copy(deep=True),
        covariates=covariates.copy(deep=True), timezone=timezone, delivery_day=day,
        config=kalman_filter_config, covariate_config=kalman_covariates,
        upstream_model="residual_corrected", output_model="residual_kalman",
        training_lookback_days=365, rolling_refit_workers=int(workers),
        rolling_refit_cache_dir=((incremental_dir / "kalman_rolling") if incremental_dir is not None
                                else workspace / "cache" / "kalman_rolling" / code.lower()),
    )
    audit = {
        "schema_version": 1, "engine": "nuclear_forecast_v1", "zone": code,
        "delivery_day": day.isoformat(), "nuclear_alias": NUCLEAR_ALIAS,
        "nuclear_series": NUCLEAR_SERIES, "nuclear_units": "GW",
        "nuclear_chronos_base_context": True, "nuclear_chronos_known_future": True,
        "nuclear_residual_feature": NUCLEAR_KNOWN_COLUMN,
        "nuclear_kalman_market_feature": NUCLEAR_ALIAS,
        "history_days": HISTORY_DAYS, "evaluation_days": EVALUATION_DAYS,
        "computation_mode": "incremental" if incremental_dir is not None else "full",
        "history_anchor_day": str(history_anchor), "raw_history_start_day": str(raw_start),
        "incremental_cache_directory": str(incremental_dir) if incremental_dir is not None else None,
        "daily_chronos_cache": daily_chronos.audit if daily_chronos is not None else {},
        "daily_residual_cache": ({"hits": daily_residual.hits, "misses": daily_residual.misses,
                                  "writes": daily_residual.writes} if daily_residual is not None else {}),
        "evaluation_start_day": str(day - timedelta(days=EVALUATION_DAYS)),
        "evaluation_end_day": str(day - timedelta(days=1)),
        "warmup_scope": ("fixed_epoch_diagnostic_warmup_not_full_nested_validation" if incremental_dir is not None
                         else "first_365_days_diagnostic_cold_start_not_full_nested_validation"),
        "residual_fit_rule": "D-365 <= local_delivery_day < D",
        "residual_refit_cadence_days": 1,
        "residual_recipe": copy.deepcopy(residual_options),
        "residual_expert_models": ["chronos2"],
        "residual_training_cold_start_days": int(daily_audit.generation_source.eq("identity_chronos_cold_start").sum()),
        "kalman_training_lookback_days": 365,
        "kalman_covariate_config": kalman_covariates.to_dict(),
        "kalman_filter_parameters": filter_parameters,
        "raw_future_cache_hit": cached_future is not None,
        "residual_result_cache_hit": cached_residual is not None,
        "result_cache_directory": str(result_cache_root),
        "source_hashes": prepared_hashes,
        "resolved_config_sha256": _digest_json(resolved),
        "shared_upstream_statistics_sha256": _digest_frame(statistics),
        "shared_upstream_forecast_sha256": _digest_frame(forecast),
        "incumbent_upstream_prefix_used": False, "lora_used": False,
        "production_changed": False, "sealed_live_contract_modified": False,
        "storm_used_as_input": False, "mkonline_used_as_input": False,
        "pit_validation_responsibility": "caller_materialized_and_audited_per_delivery_cutoff",
    }
    return NuclearForecastResult(raw_history.loc[history_index], statistics, forecast, covariates, daily_audit, kalman, audit)


__all__ = [
    "NuclearForecastError", "NuclearForecastResult", "NUCLEAR_ALIAS", "NUCLEAR_SERIES",
    "NUCLEAR_KNOWN_COLUMN", "run_nuclear_forecast", "causal_residual_replay",
    "nuclear_kalman_covariate_config",
]
