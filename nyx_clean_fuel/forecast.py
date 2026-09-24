"""Full-chain clean-fuel challenger: Chronos context/future, residual, Kalman.

This adapter never downloads data, changes incumbent recipes or applies a
fuel-cost floor. Native CGC/CCC already include CO2 and thermal efficiency.
The caller prepares and pins audited daily close vintages at civil D-1 08h.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from chronos2_hourly import nuclear_forecast as base
from chronos2_hourly.kalman_covariates import KalmanCovariateConfig
from chronos2_hourly.kalman_residual import build_operational_kalman_view
from .sources import SERIES, FORMULAS


ENGINE = "nyx_clean_fuel_full_v1"
FUEL_ALIASES = tuple(SERIES)
FUEL_KNOWN_COLUMNS = tuple(f"known_{alias}_oracle" for alias in FUEL_ALIASES)
SCHEMA = {alias: {"series": SERIES[alias], "formula": FORMULAS[alias], "unit": "EUR/MWh_e",
                  "semantic": "native_clean_fuel_cost", "daily_broadcast": True,
                  "carbon_included": True} for alias in FUEL_ALIASES}


class CleanFuelForecastError(base.NuclearForecastError):
    """Unsafe source/schema, dropped features or an operational write refused."""


def clean_fuel_code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def clean_fuel_input_protocol() -> str:
    """Invalidate every incremental stage when this adapter/schema changes."""
    contract = {"schema_version": 1, "engine": ENGINE, "schema": SCHEMA,
                "adapter_sha256": clean_fuel_code_sha256(),
                "sources_sha256": hashlib.sha256(Path(__file__).with_name("sources.py").read_bytes()).hexdigest(),
                "cutoff": "civil_D_minus_1_08:00", "same_day_closes_excluded": True}
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    return f"civil_pit_clean_fuel_full_v1_{digest}"


protocol = clean_fuel_input_protocol


def clean_fuel_kalman_covariate_config(original: KalmanCovariateConfig | None = None) -> KalmanCovariateConfig:
    original = original or base.nuclear_kalman_covariate_config()
    result = replace(original,
        input_columns=tuple(dict.fromkeys((*original.input_columns, *FUEL_ALIASES))),
        groups={name: tuple(dict.fromkeys((*columns, *FUEL_ALIASES))) for name, columns in original.groups.items()},
        history_missing_policy="complete_trailing", minimum_history_coverage=1.0, require_future_complete=True)
    # Do not combine EUR/MWh costs with GW in residual_load_mean/spread.
    result.validate()
    return result


def _frame(value: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame) or value.empty:
        raise CleanFuelForecastError(f"{name}: nonempty DataFrame required")
    result = value.copy(deep=True)
    if not isinstance(result.index, pd.DatetimeIndex):
        keys = [key for key in ("delivery_start_utc", "timestamp", "timestamp_utc") if key in result]
        if len(keys) != 1:
            raise CleanFuelForecastError(f"{name}: one explicit timestamp column required")
        values = result.pop(keys[0])
        if any(pd.isna(v) or pd.Timestamp(v).tzinfo is None for v in values):
            raise CleanFuelForecastError(f"{name}: timezone-aware timestamps required")
        result.index = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    if result.index.tz is None or result.index.hasnans or result.index.has_duplicates:
        raise CleanFuelForecastError(f"{name}: unique timezone-aware timestamps required")
    result.index = result.index.tz_convert("UTC")
    if not result.index.is_monotonic_increasing or not result.index.equals(result.index.floor("h")) or result.columns.has_duplicates:
        raise CleanFuelForecastError(f"{name}: increasing physical hourly index and unique columns required")
    return result


def _isolated_paths(config: Mapping[str, Any], workdir: str | Path) -> None:
    project = Path(str(config.get("data", {}).get("project_root", "."))).resolve()
    root = project / "runs/experiments" / ENGINE
    supplied = Path(workdir).absolute()
    work = supplied.resolve()
    if supplied != work or not work.is_relative_to(root) or work == root:
        raise CleanFuelForecastError(f"Full clean-fuel workdir must be a real child of {root}")
    if (work / "artifact_checksums.json").exists():
        raise CleanFuelForecastError("Full clean-fuel replay cannot modify a sealed archive")
    experiment = config.get("nuclear_experiment", {})
    if experiment.get("mode", "full") == "incremental":
        original = Path(str(experiment.get("incremental_cache_dir", ""))).absolute()
        cache = original.resolve()
        if original != cache or not cache.is_relative_to(root) or cache == root:
            raise CleanFuelForecastError("Full clean-fuel incremental cache must use its isolated namespace")


def _source_specs(config: Mapping[str, Any], zone: str) -> None:
    specs = config.get("zones", {}).get(zone, {}).get("covariates", {})
    for alias, series in {base.NUCLEAR_ALIAS: base.NUCLEAR_SERIES, **SERIES}.items():
        spec = specs.get(alias)
        if (not isinstance(spec, Mapping) or spec.get("enabled") is not True
                or spec.get("source") != "pit_parquet" or spec.get("series") != series):
            raise CleanFuelForecastError(f"{alias}: exact enabled native PIT series required")
        future = spec.get("future", {})
        if future.get("known_future") is not True or future.get("strategies") != ["oracle"]:
            raise CleanFuelForecastError(f"{alias}: declared known-future oracle column required")
        if alias in FUEL_ALIASES:
            expected = {key: value for key, value in SCHEMA[alias].items() if key not in {"series", "formula"}}
            expected["fill_method"] = "none"
            if any(spec.get(key) != value for key, value in expected.items()):
                raise CleanFuelForecastError(f"{alias}: explicit native EUR/MWh_e, carbon, daily-broadcast and no-fill semantics required")


def _prepare_inputs(config: Mapping[str, Any], data: Any, zone: str, day: date):
    from run_chronos2_hourly import _feature_inputs
    if zone not in {"FR", "DE", "BE", "NL"} or str(data.zone).upper() != zone or str(data.timezone) != base.ZONE_TIMEZONES[zone]:
        raise CleanFuelForecastError("Clean-fuel zone/timezone must match prepared FR/DE/BE/NL data")
    timezone = str(data.timezone)
    _source_specs(config, zone)
    context = _frame(data.model_context_covariates, "model_context_covariates")
    if any(token in str(col).lower() for col in context for token in ("storm", "mkonline", "actual", "observed")):
        raise CleanFuelForecastError("Competing forecasts and observed/actual labels cannot be exogenous inputs")
    start = pd.Timestamp(config.get("nuclear_experiment", {}).get("raw_history_start_day", day - timedelta(days=730)))
    if (pd.isna(start) or start.tzinfo is not None or start != start.normalize()
            or not day - timedelta(days=1095) <= start.date() <= day - timedelta(days=730)):
        raise CleanFuelForecastError("Clean-fuel raw history must cover 730 to 1095 civil days")
    expected = pd.date_range(start.tz_localize(timezone), pd.Timestamp(day + timedelta(days=1), tz=timezone),
                             freq="h", inclusive="left").tz_convert("UTC")
    if len(expected.difference(context.index)):
        raise CleanFuelForecastError("Clean-fuel context misses replay/future physical hours")
    target, _, future, features = _feature_inputs(data, config)
    historical = _frame(data.covariates, "covariates")
    historical_required = expected.intersection(target.index)
    audit = {}
    for alias in FUEL_ALIASES:
        known = f"known_{alias}_oracle"
        if alias not in context or alias not in historical or known not in context or known not in data.known_future_columns:
            raise CleanFuelForecastError(f"{alias}: Chronos history/context and declared known future required")
        if known not in features or known not in future:
            raise CleanFuelForecastError(f"{alias}: residual/future feature selection dropped the fuel input")
        values = context[[alias, known]].apply(pd.to_numeric, errors="coerce")
        mandatory = values.reindex(expected).to_numpy(float)
        if not np.isfinite(mandatory).all() or np.isinf(values.to_numpy(float)).any():
            raise CleanFuelForecastError(f"{alias}: nonfinite required native clean-fuel inputs")
        history_values = pd.to_numeric(historical[alias].reindex(historical_required), errors="coerce")
        if not np.isfinite(history_values).all() or not np.allclose(history_values, values[alias].reindex(historical_required), rtol=0, atol=1e-9):
            raise CleanFuelForecastError(f"{alias}: historical covariates differ from model context")
        finite = np.isfinite(values.to_numpy(float))
        paired = finite.all(axis=1)
        if not np.array_equal(finite[:, 0], finite[:, 1]) or not np.allclose(values.loc[paired, alias], values.loc[paired, known], rtol=0, atol=1e-9):
            raise CleanFuelForecastError(f"{alias}: context and known-future vintages diverge")
        selected = features[known].reindex(expected).to_numpy(float)
        if not np.isfinite(selected).all() or not np.allclose(selected, mandatory[:, 1], rtol=0, atol=1e-9):
            raise CleanFuelForecastError(f"{alias}: residual values differ from the native PIT source")
        daily = values.loc[expected, known].groupby(expected.tz_convert(timezone).date)
        if ((daily.max() - daily.min()) > 1e-9).any():
            raise CleanFuelForecastError(f"{alias}: native daily clean costs must be broadcast unchanged")
        optional = values.loc[values.index < expected[0]]
        audit[alias] = {"required_hours": len(expected), "missing_required_hours": 0,
            "optional_pre_anchor_context_hours": len(optional),
            "missing_optional_pre_anchor_context_hours": int(optional[known].isna().sum()),
            "pre_anchor_policy": "preserve_incumbent_support_no_imputation"}
    return context, audit, expected


def _validate_residual_factory(factory: Callable[[], Any]) -> None:
    from chronos2_hourly.models.residual_corrector import ResidualMetaFeatureBuilder
    prototype = factory()
    builder = getattr(prototype, "feature_builder", None) or ResidualMetaFeatureBuilder(**getattr(prototype, "feature_builder_options", {}))
    if not builder.exclude_historical_prices:
        raise CleanFuelForecastError("Clean-fuel residual keeps the incumbent historical-price exclusion")
    if any(builder._is_excluded(known) for known in (*FUEL_KNOWN_COLUMNS, base.NUCLEAR_KNOWN_COLUMN)):
        raise CleanFuelForecastError("Residual builder excludes a required clean-fuel/nuclear input")


class _FuelKalmanBuilder:
    def __init__(self, context: pd.DataFrame, delegate: Callable[..., Any], signature: str, custom: bool):
        self.context, self.delegate, self.signature, self.custom = context, delegate, signature, custom
        self.used_config = self.used_covariates = None

    def __call__(self, **kwargs):
        incoming = _frame(kwargs["covariates"], "Kalman covariates")
        for alias, known in zip(FUEL_ALIASES, FUEL_KNOWN_COLUMNS):
            incoming[alias] = self.context[known].reindex(incoming.index)
        if not np.isfinite(incoming[list(FUEL_ALIASES)].to_numpy(float)).all():
            raise CleanFuelForecastError("Clean-fuel Kalman inputs must cover every replay/future hour")
        selected = clean_fuel_kalman_covariate_config(kwargs["covariate_config"])
        # The native operational Kalman parser requires a `timestamp` column.
        # A delivery_start_utc column is correct for predictions, not inputs.
        self.used_config, self.used_covariates = selected, incoming.rename_axis("timestamp").reset_index()
        kwargs = dict(kwargs, covariates=self.used_covariates.copy(deep=True), covariate_config=selected)
        if kwargs.get("rolling_refit_cache_dir") is not None:
            path = Path(kwargs["rolling_refit_cache_dir"])
            if self.custom:
                path = path / self.signature[:24]
            if os.name == "nt":
                absolute = str(path.resolve())
                if not absolute.startswith("\\\\?\\"):
                    absolute = "\\\\?\\UNC\\" + absolute[2:] if absolute.startswith("\\\\") else "\\\\?\\" + absolute
                path = Path(absolute)
            kwargs["rolling_refit_cache_dir"] = path
        return self.delegate(**kwargs)


def run_clean_fuel_forecast(*, config: Mapping[str, Any], data: Any, zone: str, delivery_day: str | date,
    workdir: str | Path, device: str = "auto", threads: int = 4, workers: int = 1,
    runtime_factory: Callable[..., Any] | None = None, forecasting_module: Any | None = None,
    residual_factory: Callable[[], Any] | None = None, kalman_builder: Callable[..., Any] | None = None,
) -> base.NuclearForecastResult:
    """Recompute the three unchanged stages, changing only audited fuel inputs."""
    from run_chronos2_hourly import _residual_corrector_factory
    resolved = deepcopy(dict(config))
    _isolated_paths(resolved, workdir)
    if resolved.get("nuclear_experiment", {}).get("input_protocol") != clean_fuel_input_protocol():
        raise CleanFuelForecastError("Clean-fuel input_protocol must pin current adapter/source/schema before preparation")
    parsed = pd.Timestamp(delivery_day)
    if pd.isna(parsed) or parsed.tzinfo is not None or parsed != parsed.normalize():
        raise CleanFuelForecastError("Clean-fuel delivery_day must be an explicit civil date")
    day, code = parsed.date(), str(zone).upper()
    context, coverage, expected = _prepare_inputs(resolved, data, code, day)
    factory, upstream = _residual_corrector_factory(resolved, timezone=str(data.timezone))
    if factory is None or upstream != "chronos2":
        raise CleanFuelForecastError("Clean-fuel requires the unchanged Chronos-based residual recipe")
    _validate_residual_factory(residual_factory or factory)
    identity = base._factory_cache_identity(kalman_builder) if kalman_builder is not None else "unchanged_build_operational_kalman_view"
    signature = hashlib.sha256(json.dumps({"protocol": clean_fuel_input_protocol(), "delegate": identity}, sort_keys=True, default=str).encode()).hexdigest()
    hook = _FuelKalmanBuilder(context, kalman_builder or build_operational_kalman_view, signature, kalman_builder is not None)
    result = base.run_nuclear_forecast(config=resolved, data=data, zone=code, delivery_day=day, workdir=workdir,
        device=device, threads=threads, workers=workers, runtime_factory=runtime_factory,
        forecasting_module=forecasting_module, residual_factory=residual_factory, kalman_builder=hook)
    if hook.used_config is None or hook.used_covariates is None:
        raise CleanFuelForecastError("Clean-fuel Kalman adapter was not executed")
    daily = result.residual_daily_audit
    if not {"generation_source", "residual_feature_columns"}.issubset(daily):
        raise CleanFuelForecastError("Residual fitted-feature audit is required")
    fitted = daily.loc[daily.generation_source.eq("daily_prequential_refit"), "residual_feature_columns"]
    if fitted.empty or any(not set((*FUEL_KNOWN_COLUMNS, base.NUCLEAR_KNOWN_COLUMN)).issubset(columns) for columns in fitted):
        raise CleanFuelForecastError("A fitted residual corrector dropped a required clean-fuel/nuclear input")
    if str(result.audit.get("raw_history_start_day")) != expected[0].tz_convert(data.timezone).date().isoformat():
        raise CleanFuelForecastError("Clean-fuel engine changed the pinned replay anchor")
    audit = deepcopy(dict(result.audit))
    audit.update(candidate_engine=ENGINE, candidate_variant="clean_fuel_kalman", clean_fuel_schema=SCHEMA,
        clean_fuel_input_coverage=coverage, clean_fuel_input_protocol=clean_fuel_input_protocol(),
        clean_fuel_adapter_sha256=clean_fuel_code_sha256(), clean_fuel_kalman_hook_signature=signature,
        clean_fuel_chronos_context_columns=list(FUEL_ALIASES), clean_fuel_chronos_known_future_columns=list(FUEL_KNOWN_COLUMNS),
        clean_fuel_residual_features=list(FUEL_KNOWN_COLUMNS), clean_fuel_kalman_market_features=list(FUEL_ALIASES),
        kalman_covariate_config=hook.used_config.to_dict(), native_costs_rescaled=False, carbon_added_again=False,
        enforced_price_floor=False, source_audit_responsibility="caller_pinned_native_PIT_sources_and_civil_D_minus_1_08_cutoff_audits",
        production_pit_evidence=False, promotion_eligible=False, production_changed=False, sealed_live_contract_modified=False)
    return replace(result, covariates=hook.used_covariates.copy(deep=True), audit=audit)


__all__ = ["ENGINE", "FUEL_ALIASES", "FUEL_KNOWN_COLUMNS", "SCHEMA", "CleanFuelForecastError", "protocol",
           "clean_fuel_code_sha256", "clean_fuel_input_protocol", "clean_fuel_kalman_covariate_config", "run_clean_fuel_forecast"]
