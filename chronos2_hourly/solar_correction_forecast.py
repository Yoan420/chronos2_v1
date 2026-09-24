"""Isolated solar correction ablation on immutable NuclearFR Chronos outputs.

There is deliberately no neural execution path in this module. The existing
causal residual replay is run once, and its exact outputs feed two unchanged
Kalman implementations: standard NuclearFR inputs, then those inputs + solar.
The caller owns archive verification and per-delivery PIT source auditing.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from . import nuclear_forecast as base
from .nuclear_cwe_forecast import _frame
from .nuclear_residual_cache import ResidualDayCache
from .kalman_residual import build_operational_kalman_view
from .solar_cwe_forecast import (
    INPUT_ALIASES, SOLAR_ALIASES, SOLAR_KNOWN_COLUMNS, SOLAR_SERIES,
    _prepare_inputs, solar_cwe_kalman_covariate_config,
)


ENGINE = "solar_correction_v1"
VARIANTS = {"residual": "solar_residual", "residual_kalman": "solar_residual_kalman"}


class SolarCorrectionForecastError(base.NuclearForecastError):
    """The immutable Chronos or isolated correction-only contract was violated."""


def solar_correction_code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def solar_correction_input_protocol() -> str:
    return "civil_pit_solar_correction_v1_" + base._digest_json({
        "engine": ENGINE, "adapter_sha256": solar_correction_code_sha256(),
        "sources": SOLAR_SERIES, "units": "GW", "daily_broadcast": False,
        "cutoff": "civil_D_minus_1_08:00", "chronos": "frozen_nuclear_archive",
    })


def frozen_chronos_sha256(frame: pd.DataFrame, prefix: str = "") -> str:
    """Canonical exact-value/dtype fingerprint shared by engine and reporting."""
    normalized = _frame(frame, "frozen Chronos fingerprint")
    selected = normalized.loc[:, [prefix + q for q in base.QUANTILES]].copy()
    selected.columns = list(base.QUANTILES)
    selected.index.name = "delivery_start_utc"
    return base._digest_frame(selected)


def _isolated_paths(config: Mapping[str, Any], workdir: str | Path) -> tuple[Path, Path]:
    project = Path(str(config.get("data", {}).get("project_root", "."))).resolve()
    allowed = project / "runs" / "experiments" / ENGINE

    def check(path: str | Path) -> Path:
        value = Path(path).absolute()
        if (allowed.resolve() != allowed or value.resolve() != value
                or not value.is_relative_to(allowed) or value == allowed):
            raise SolarCorrectionForecastError(
                "Solar correction writes require an unredirected child of runs/experiments/solar_correction_v1")
        for parent in (value, *value.parents):
            if (parent / "artifact_checksums.json").exists():
                raise SolarCorrectionForecastError("Solar correction cannot write inside a sealed archive")
            if parent == allowed:
                break
        return value

    workspace = check(workdir)
    options = config.get("nuclear_experiment", {})
    mode = options.get("mode", "full")
    if mode not in {"full", "incremental"}:
        raise SolarCorrectionForecastError("Solar correction mode must be full or incremental")
    if mode == "incremental":
        root = options.get("incremental_cache_dir")
        if not root:
            raise SolarCorrectionForecastError("Incremental solar correction requires its isolated cache directory")
        check(root)
        cache = check(options.get("incremental_namespace", root))
        if not cache.is_relative_to(Path(root).absolute()):
            raise SolarCorrectionForecastError("Incremental namespace escapes its configured cache root")
    else:
        cache = check(workspace / "cache")
    # Check exact write targets as well: existing child symlinks must not escape.
    for relative in ("residual", "kalman_residual", "kalman_residual_kalman"):
        check(cache / relative)
    return workspace, cache


def _long_path(path: Path) -> Path:
    if os.name == "nt":
        value = str(path.resolve())
        if not value.startswith("\\\\?\\"):
            value = "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
        return Path(value)
    return path


def _member(value: Any, name: str) -> Any:
    return value[name] if isinstance(value, Mapping) else getattr(value, name)


def _frozen_inputs(incumbent: Any, data: Any, expected: pd.DatetimeIndex,
                   day: date, zone: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        history = _frame(_member(incumbent, "raw_history"), "incumbent raw history")
        source = _frame(_member(incumbent, "source_forecast"), "incumbent source forecast")
        audit = _member(incumbent, "audit")
    except (KeyError, AttributeError, TypeError) as exc:
        raise SolarCorrectionForecastError("A verified incumbent raw_history/source_forecast/audit bundle is required") from exc
    if not isinstance(audit, Mapping):
        raise SolarCorrectionForecastError("Incumbent audit must be a mapping")
    if audit.get("engine") != "nuclear_forecast_v1" or audit.get("zone") != zone or audit.get("delivery_day") != str(day):
        raise SolarCorrectionForecastError("Frozen incumbent must be the NuclearFR archive for exactly this zone and delivery")
    if audit.get("candidate_engine") not in (None, "nuclear_forecast_v1"):
        raise SolarCorrectionForecastError("A challenger archive cannot substitute for frozen NuclearFR predictions")
    future_index = expected[expected.tz_convert(data.timezone).date == day]
    historical_index = expected[expected.tz_convert(data.timezone).date < day]
    if not history.index.equals(historical_index) or not source.index.equals(future_index):
        raise SolarCorrectionForecastError("Incumbent must contain every pinned anchored history hour and the exact future day")
    base._validate_origins(history, data.timezone)
    base._validate_origins(source, data.timezone)
    base._quantiles(history, name="incumbent raw history")
    required = [f"chronos2__{q}" for q in base.QUANTILES]
    if not set(required).issubset(source):
        raise SolarCorrectionForecastError("Frozen future requires chronos2__q10/q50/q90, never corrected q columns")
    if "actual" in source and source.actual.notna().any():
        raise SolarCorrectionForecastError("Future actual observations are forbidden")
    future = source[required + ["forecast_origin_utc"]].rename(
        columns={f"chronos2__{q}": q for q in base.QUANTILES}).copy(deep=True)
    base._quantiles(future, name="incumbent raw future")
    if "actual" not in history or not np.isfinite(pd.to_numeric(history.actual, errors="coerce")).all():
        raise SolarCorrectionForecastError("Frozen historical observations must be finite")
    target = data.target.copy(deep=True)
    if not isinstance(target.index, pd.DatetimeIndex) or target.index.tz is None:
        raise SolarCorrectionForecastError("Prepared target requires timezone-aware timestamps")
    target.index = target.index.tz_convert("UTC")
    if (not target.index.is_unique or not target.index.is_monotonic_increasing
            or target.index[-1] != future.index[0] - pd.Timedelta(hours=1)):
        raise SolarCorrectionForecastError("Prepared target must end immediately before the future day")
    # Nuclear archive actuals can be float32 by its existing forecast adapter.
    # Compare at that archived precision; never replace the archived labels.
    observed = pd.to_numeric(target.reindex(history.index), errors="coerce")
    if (not np.isfinite(observed.to_numpy(float)).all()
            or not np.array_equal(observed.to_numpy(dtype=history.actual.dtype), history.actual.to_numpy())):
        raise SolarCorrectionForecastError("Prepared historical actuals differ from the pinned incumbent")
    return history, future


def _require_exact_chronos(raw: pd.DataFrame, future: pd.DataFrame,
                           statistics: pd.DataFrame, forecast: pd.DataFrame) -> None:
    stats = _frame(statistics, "solar residual statistics")
    issued = _frame(forecast, "solar residual forecast")
    if (frozen_chronos_sha256(raw.loc[stats.index]) != frozen_chronos_sha256(stats, "chronos2__")
            or frozen_chronos_sha256(future) != frozen_chronos_sha256(issued, "chronos2__")):
        raise SolarCorrectionForecastError("Solar correction altered frozen Chronos quantiles or their dtypes")
    for archived, replayed in ((raw.loc[stats.index], stats), (future, issued)):
        if not archived.forecast_origin_utc.equals(replayed.forecast_origin_utc):
            raise SolarCorrectionForecastError("Solar correction altered frozen forecast origins")


def run_solar_correction_forecast(*, config: Mapping[str, Any], data: Any, incumbent: Any,
    zone: str, delivery_day: str | date, workdir: str | Path, device: str = "auto",
    threads: int = 4, workers: int = 1, residual_factory: Callable[[], Any] | None = None,
    kalman_builder: Callable[..., Any] | None = None, runtime_factory: Callable[..., Any] | None = None,
    forecasting_module: Any | None = None,
) -> dict[str, base.NuclearForecastResult]:
    """Return A: solar residual/standard Kalman; B: same residual/solar Kalman.

    ``runtime_factory`` and ``forecasting_module`` are deliberately never used;
    callers/tests can pass rejecting sentinels to verify this guarantee. The
    autonomous residual forecast is each result's identical ``source_forecast``.
    """
    from run_chronos2_hourly import _feature_inputs, _residual_corrector_factory
    from .models.residual_corrector import ResidualMetaFeatureBuilder

    resolved = deepcopy(dict(config))
    _, cache_root = _isolated_paths(resolved, workdir)
    for section in (resolved, resolved.get("nuclear_experiment", {}), resolved.get("solar_correction_experiment", {})):
        if any(section.get(key) for key in ("promote", "promote_to_production", "promotion_eligible", "activate")):
            raise SolarCorrectionForecastError("Solar correction is experimental and cannot promote or activate production")
    options = resolved.get("nuclear_experiment", {})
    if options.get("input_protocol") != solar_correction_input_protocol():
        raise SolarCorrectionForecastError("Solar correction input_protocol must pin this frozen-Chronos adapter")
    parsed = pd.Timestamp(delivery_day)
    if pd.isna(parsed) or parsed.tzinfo is not None or parsed != parsed.normalize():
        raise SolarCorrectionForecastError("Solar correction delivery_day must be an explicit civil date")
    day, code = parsed.date(), str(zone).upper()
    for relative in ("residual", "kalman_residual", "kalman_residual_kalman"):
        _isolated_paths(resolved, cache_root / relative / code.lower())
    if device not in {"auto", "cpu", "cuda"} or threads < 1 or workers < 1:
        raise SolarCorrectionForecastError("Invalid device/threads/workers")
    if resolved.get("data", {}).get("forecast_origin_local_time", "08:00") != "08:00":
        raise SolarCorrectionForecastError("Solar correction requires exact civil D-1 08:00 cutoff")
    if resolved.get("model", {}).get("model_id", "amazon/chronos-2") != "amazon/chronos-2":
        raise SolarCorrectionForecastError("Frozen incumbent must retain the unchanged Chronos-2 model")
    context, coverage, expected = _prepare_inputs(resolved, data, code, day)
    raw, future = _frozen_inputs(incumbent, data, expected, day, code)
    frozen_full_hash, frozen_future_hash = frozen_chronos_sha256(raw), frozen_chronos_sha256(future)
    target, _, future_covariates, features = _feature_inputs(data, resolved)
    if not future_covariates.index.equals(future.index):
        raise SolarCorrectionForecastError("Prepared future features differ from the frozen physical delivery day")
    standard_covariates = base._covariates(data, expected)
    residual_options = resolved.setdefault("hourly", {}).setdefault("residual_correction", {})
    residual_options["thread_count"] = int(threads)
    configured_factory, upstream = _residual_corrector_factory(resolved, timezone=data.timezone)
    if configured_factory is None or upstream != "chronos2":
        raise SolarCorrectionForecastError("The existing enabled Chronos-based residual recipe is required")
    selected_factory = residual_factory or configured_factory
    required_features = {f"known_{alias}_oracle" for alias in INPUT_ALIASES}
    for factory in (configured_factory, selected_factory):
        prototype = factory()
        builder = getattr(prototype, "feature_builder", None) or ResidualMetaFeatureBuilder(
            **getattr(prototype, "feature_builder_options", {}))
        if not builder.exclude_historical_prices or any(builder._is_excluded(column) for column in required_features):
            raise SolarCorrectionForecastError("Residual recipe must retain RL5/nuclear/solar and exclude historical prices")
    minimum_rows = base._residual_recipe_preflight(selected_factory)
    year_index = raw.index[raw.index.tz_convert(data.timezone).date >= day - timedelta(days=365)]
    if minimum_rows > len(year_index):
        raise SolarCorrectionForecastError("Residual minimum training rows cannot fit into rolling365")
    kalman_config, filter_parameters = base._kalman_filter_configuration(resolved)
    standard_config = base.nuclear_kalman_covariate_config()
    solar_config = solar_cwe_kalman_covariate_config(standard_config)
    # Persist only correction predictions. There is no Chronos cache, model
    # loader, neural frame construction, or data-fetch fallback here.
    cache_features = features.loc[:, [column for column in features if not builder._is_excluded(column)]]
    daily = ResidualDayCache(_long_path(cache_root / "residual" / code.lower()), {
        "engine": ENGINE, "adapter_sha256": solar_correction_code_sha256(),
        "causal_replay_sha256": base._file_sha256(Path(base.__file__)),
        "input_protocol": solar_correction_input_protocol(), "recipe": residual_options,
        "factory": base._factory_cache_identity(residual_factory),
        "input_schema": [(str(column), str(dtype)) for column, dtype in features.dtypes.items()],
    }, features=cache_features, raw=raw, timezone=data.timezone)
    output_start = day - timedelta(days=base.HISTORY_DAYS)
    statistics, forecast, daily_audit = base.causal_residual_replay(
        raw_history=raw.copy(deep=True), raw_future=future.copy(deep=True), features=features,
        timezone=data.timezone, delivery_day=day, residual_factory=selected_factory,
        output_start_day=output_start, daily_cache=daily)
    _require_exact_chronos(raw, future, statistics, forecast)
    fitted = daily_audit.loc[daily_audit.generation_source.eq("daily_prequential_refit"), "residual_feature_columns"]
    if fitted.empty or any(not required_features.issubset(columns) for columns in fitted):
        raise SolarCorrectionForecastError("A fitted residual corrector dropped a required baseline or solar input")
    report_index = expected[expected.tz_convert(data.timezone).date >= output_start]
    standard_covariates = _frame(standard_covariates, "standard Kalman covariates").loc[report_index]
    solar_covariates = standard_covariates.copy(deep=True)
    for alias in SOLAR_ALIASES:
        solar_covariates[alias] = context[f"known_{alias}_oracle"].reindex(report_index)
    raw_report = raw.loc[raw.index.tz_convert(data.timezone).date >= output_start].copy(deep=True)
    shared_audit = {
        "schema_version": 1, "engine": "nuclear_forecast_v1", "candidate_engine": ENGINE,
        "zone": code, "delivery_day": str(day), "history_days": 730, "evaluation_days": 365,
        "raw_history_start_day": str(expected[0].tz_convert(data.timezone).date()),
        "history_anchor_day": str(options.get("history_anchor_day", expected[0].tz_convert(data.timezone).date())),
        "computation_mode": options.get("mode", "full"), "incremental_cache_directory": str(cache_root),
        "evaluation_start_day": str(day - timedelta(days=365)), "evaluation_end_day": str(day - timedelta(days=1)),
        "warmup_scope": "fixed_epoch_diagnostic_warmup_not_full_nested_validation",
        "residual_fit_rule": "D-365 <= local_delivery_day < D", "residual_refit_cadence_days": 1,
        "residual_recipe": deepcopy(residual_options), "residual_expert_models": ["chronos2"],
        "residual_training_cold_start_days": int(daily_audit.generation_source.eq("identity_chronos_cold_start").sum()),
        "residual_replay_count": 1, "kalman_training_lookback_days": 365,
        "kalman_filter_parameters": filter_parameters,
        "daily_residual_cache": {"hits": daily.hits, "misses": daily.misses, "writes": daily.writes},
        "daily_chronos_cache": {}, "chronos_recomputed": False, "chronos_runtime_calls": 0,
        "frozen_chronos_input_aliases": list(INPUT_ALIASES[:6]),
        "frozen_chronos_exact_match": True, "frozen_chronos_history_sha256": frozen_chronos_sha256(raw_report),
        "frozen_chronos_full_history_sha256": frozen_full_hash, "frozen_chronos_future_sha256": frozen_future_hash,
        "shared_upstream_statistics_sha256": base._digest_frame(statistics),
        "shared_upstream_forecast_sha256": base._digest_frame(forecast),
        "solar_input_protocol": solar_correction_input_protocol(), "solar_adapter_sha256": solar_correction_code_sha256(),
        "solar_input_coverage": coverage,
        "solar_sources": {alias: {"series": series, "unit": "GW", "semantic": "forecast_generation", "daily_broadcast": False}
                          for alias, series in SOLAR_SERIES.items()},
        "solar_chronos_context_columns": [], "solar_chronos_known_future_columns": [],
        "solar_residual_features": list(SOLAR_KNOWN_COLUMNS), "baseline_inputs_preserved": True,
        "additional_raw_input_count": 4, "new_ramps_or_aggregates": False, "spike_classifier_added": False,
        "inherited_residual_feature_builder_unchanged": True, "production_pit_evidence": False,
        "source_audit_responsibility": "caller_verified_incumbent_and_pinned_hourly_solar_PIT_civil_D_minus_1_08_cutoff_audits",
        "incumbent_upstream_prefix_used": False, "lora_used": False, "production_changed": False,
        "promotion_eligible": False, "sealed_live_contract_modified": False,
        "storm_used_as_input": False, "mkonline_used_as_input": False,
        "source_hashes": {"engine_file_sha256": base._file_sha256(Path(base.__file__)),
                          "solar_correction_engine_file_sha256": solar_correction_code_sha256(),
                          "target": base._digest_frame(target), "residual_features": base._digest_frame(features)},
        "resolved_config_sha256": base._digest_json(resolved),
    }
    results = {}
    delegate = kalman_builder or build_operational_kalman_view
    signature = base._digest_json(base._factory_cache_identity(kalman_builder)) if kalman_builder is not None else None
    for variant, covariates, covariate_config in (
        ("residual", standard_covariates, standard_config),
        ("residual_kalman", solar_covariates, solar_config),
    ):
        covariates = covariates.rename_axis("timestamp").reset_index()
        kalman_cache = cache_root / f"kalman_{variant}" / code.lower()
        if signature:
            kalman_cache /= signature[:24]
        # This check includes pre-existing zone/delegate subdirectories.
        _isolated_paths(resolved, kalman_cache)
        view = delegate(statistics=statistics.copy(deep=True), source_forecast=forecast.copy(deep=True),
            covariates=covariates.copy(deep=True), timezone=data.timezone, delivery_day=day,
            config=deepcopy(kalman_config), covariate_config=covariate_config,
            upstream_model="residual_corrected", output_model="residual_kalman", training_lookback_days=365,
            rolling_refit_workers=int(workers), rolling_refit_cache_dir=_long_path(kalman_cache))
        _require_exact_chronos(raw, future, statistics, forecast)
        if frozen_chronos_sha256(raw) != frozen_full_hash or frozen_chronos_sha256(future) != frozen_future_hash:
            raise SolarCorrectionForecastError("A correction stage mutated frozen Chronos inputs")
        audit = deepcopy(shared_audit)
        audit.update(candidate_variant=VARIANTS[variant], solar_kalman_market_features=(list(SOLAR_ALIASES) if variant == "residual_kalman" else []),
                     kalman_covariate_config=covariate_config.to_dict(), solar_kalman_delegate_signature=signature)
        results[variant] = base.NuclearForecastResult(raw_report.copy(deep=True), statistics.copy(deep=True),
            forecast.copy(deep=True), covariates.copy(deep=True), daily_audit.copy(deep=True), view, audit)
    return results


__all__ = ["ENGINE", "VARIANTS", "SOLAR_SERIES", "SOLAR_ALIASES", "SOLAR_KNOWN_COLUMNS", "INPUT_ALIASES",
           "SolarCorrectionForecastError", "solar_correction_code_sha256", "solar_correction_input_protocol",
           "frozen_chronos_sha256", "run_solar_correction_forecast"]
