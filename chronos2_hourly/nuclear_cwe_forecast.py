"""Isolated CWE nuclear-availability challenger using the existing replay.

The FR input remains forecast generation. BE/NL inputs are forecast available
capacity (Pmax), not generation; their daily values are broadcast to physical
hours by the separately audited source materializer. Nothing is fetched here.
The unchanged Chronos, residual and Kalman implementations are reused. Only
their explicitly configured input schema differs from the incumbent.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from . import nuclear_forecast as base
from .kalman_covariates import KalmanCovariateConfig
from .kalman_residual import build_operational_kalman_view


CWE_ENGINE = "nuclear_cwe_forecast_v1"
CWE_AVAILABILITY_SERIES = {
    "be_nuclear_available_gw": "power.nrjscan.be.3mv.availability.pmax.type.nuclear.gw",
    "nl_nuclear_available_gw": "power.nrjscan.nl.3mv.availability.pmax.type.nuclear.gw",
}
CWE_NUCLEAR_ALIASES = (base.NUCLEAR_ALIAS, *CWE_AVAILABILITY_SERIES)


class NuclearCWEForecastError(base.NuclearForecastError):
    """A challenger-only schema, source or isolation contract was refused."""


def nuclear_cwe_code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def nuclear_cwe_input_protocol() -> str:
    """Pin this adapter in the existing semantic daily-cache identity."""
    return f"civil_pit_cwe_v1_{nuclear_cwe_code_sha256()}"


def nuclear_cwe_kalman_covariate_config(
    original: KalmanCovariateConfig | None = None,
) -> KalmanCovariateConfig:
    original = original or base.nuclear_kalman_covariate_config()
    result = replace(
        original,
        input_columns=tuple(dict.fromkeys((*original.input_columns, *CWE_AVAILABILITY_SERIES))),
        groups={name: tuple(dict.fromkeys((*columns, *CWE_AVAILABILITY_SERIES)))
                for name, columns in original.groups.items()},
        history_missing_policy="complete_trailing", minimum_history_coverage=1.0,
        require_future_complete=True,
    )
    result.validate()
    return result


def _frame(value: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise NuclearCWEForecastError(f"{name}: DataFrame required")
    result = value.copy(deep=True)
    if not isinstance(result.index, pd.DatetimeIndex):
        columns = [key for key in ("delivery_start_utc", "timestamp", "timestamp_utc") if key in result]
        if len(columns) != 1:
            raise NuclearCWEForecastError(f"{name}: one explicit timestamp column required")
        raw = result.pop(columns[0])
        if any(pd.isna(item) or pd.Timestamp(item).tzinfo is None for item in raw):
            raise NuclearCWEForecastError(f"{name}: timezone-aware timestamps required")
        result.index = pd.DatetimeIndex(pd.to_datetime(raw, utc=True))
    if result.index.tz is None or result.index.hasnans or result.index.has_duplicates:
        raise NuclearCWEForecastError(f"{name}: unique timezone-aware timestamps required")
    # Alignment is physical, not civil: floor() on an autumn local 02:00
    # would otherwise ask pandas to resolve an already explicit DST fold.
    result.index = result.index.tz_convert("UTC")
    if not result.index.is_monotonic_increasing or not result.index.equals(result.index.floor("h")):
        raise NuclearCWEForecastError(f"{name}: increasing physical hourly index required")
    if result.columns.has_duplicates:
        raise NuclearCWEForecastError(f"{name}: duplicate columns refused")
    return result


def _isolated_paths(config: Mapping[str, Any], workdir: str | Path) -> None:
    project = Path(str(config.get("data", {}).get("project_root", "."))).resolve()
    root = project / "runs" / "experiments"
    work = Path(workdir).resolve()
    if not work.is_relative_to(root) or work == root:
        raise NuclearCWEForecastError("CWE workdir must be inside its own runs/experiments directory")
    owner = work.relative_to(root).parts[0]
    if owner == "nuclear_forecast_v1":
        raise NuclearCWEForecastError("CWE must not write the incumbent nuclear_forecast_v1 directory")
    if (work / "artifact_checksums.json").exists():
        raise NuclearCWEForecastError("CWE cannot modify a sealed operational archive")
    experiment = config.get("nuclear_experiment", {})
    if experiment.get("mode", "full") == "incremental":
        cache = Path(str(experiment.get("incremental_cache_dir", ""))).resolve()
        if not cache.is_relative_to(root / owner) or cache == root / owner:
            raise NuclearCWEForecastError("CWE incremental cache must belong to this isolated experiment")


def _source_specs(config: Mapping[str, Any], zone: str) -> None:
    specs = config.get("zones", {}).get(zone, {}).get("covariates", {})
    for alias, series in {base.NUCLEAR_ALIAS: base.NUCLEAR_SERIES, **CWE_AVAILABILITY_SERIES}.items():
        spec = specs.get(alias)
        if (not isinstance(spec, Mapping) or spec.get("enabled") is not True
                or spec.get("source") != "pit_parquet" or spec.get("series") != series):
            raise NuclearCWEForecastError(f"{alias}: explicit enabled PIT source with exact series required")
        future = spec.get("future", {})
        if (future.get("known_future") is not True or future.get("strategies") != ["oracle"]):
            raise NuclearCWEForecastError(f"{alias}: known_future with the audited oracle column required")
        if alias in CWE_AVAILABILITY_SERIES:
            # Metadata, when supplied, must not misrepresent Pmax as generation.
            if spec.get("unit", spec.get("units", "GW")) != "GW":
                raise NuclearCWEForecastError(f"{alias}: GW units required")
            if spec.get("semantic", "forecast_available_capacity") != "forecast_available_capacity":
                raise NuclearCWEForecastError(f"{alias}: Pmax is forecast_available_capacity, not generation")
            if spec.get("daily_broadcast", True) is not True:
                raise NuclearCWEForecastError(f"{alias}: daily broadcast semantics required")


def _prepare_inputs(config: Mapping[str, Any], data: Any, zone: str, day: date):
    from run_chronos2_hourly import _feature_inputs

    timezone = base.ZONE_TIMEZONES.get(zone)
    if timezone is None or str(data.zone).upper() != zone or str(data.timezone) != timezone:
        raise NuclearCWEForecastError("CWE zone and prepared timezone must match")
    _source_specs(config, zone)
    context = _frame(data.model_context_covariates, "model_context_covariates")
    start = pd.Timestamp(config.get("nuclear_experiment", {}).get(
        "raw_history_start_day", day - timedelta(days=730)))
    if (pd.isna(start) or start.tzinfo is not None or start != start.normalize()
            or not day - timedelta(days=1095) <= start.date() <= day - timedelta(days=730)):
        raise NuclearCWEForecastError("CWE raw history must cover 730 to 1095 civil days")
    expected = pd.date_range(start.tz_localize(timezone),
                             pd.Timestamp(day + timedelta(days=1), tz=timezone),
                             freq="h", inclusive="left").tz_convert("UTC")
    if len(expected.difference(context.index)):
        raise NuclearCWEForecastError("CWE context does not cover every replay/future physical hour")
    target, _, future, features = _feature_inputs(data, config)
    historical_covariates = _frame(data.covariates, "covariates")
    historical_required = expected.intersection(target.index)
    diagnostics = {}
    for alias in CWE_NUCLEAR_ALIASES:
        known = f"known_{alias}_oracle"
        if (alias not in context or alias not in historical_covariates or known not in context
                or known not in data.known_future_columns):
            raise NuclearCWEForecastError(f"{alias}: base context and declared known-future column required")
        if known not in features or known not in future:
            raise NuclearCWEForecastError(f"{alias}: residual/future feature selection excluded the input")
        values = context[[alias, known]].apply(pd.to_numeric, errors="coerce")
        mandatory = values.reindex(expected).to_numpy(float)
        if not np.isfinite(mandatory).all():
            raise NuclearCWEForecastError(f"{alias}: incomplete PIT replay/future")
        history_values = pd.to_numeric(historical_covariates[alias].reindex(historical_required), errors="coerce")
        if not np.isfinite(history_values).all() or not np.allclose(
                history_values, values[alias].reindex(historical_required), rtol=0, atol=1e-9):
            raise NuclearCWEForecastError(f"{alias}: historical covariates and model context diverge")
        # Before the fixed replay anchor, preserve the incumbent's optional
        # exogenous-context support. Never fill missing values with hindsight.
        optional = values.loc[values.index < expected[0]]
        present = np.isfinite(values.to_numpy(float))
        finite_pairs = present.all(axis=1)
        if not np.array_equal(present[:, 0], present[:, 1]) or not np.allclose(
                values.loc[finite_pairs, alias], values.loc[finite_pairs, known], rtol=0, atol=1e-9):
            raise NuclearCWEForecastError(f"{alias}: context and known-future vintages diverge")
        finite_values = values.to_numpy(float)[present]
        maximum = 100.0 if alias == base.NUCLEAR_ALIAS else 20.0
        if ((finite_values < 0) | (finite_values > maximum)).any():
            raise NuclearCWEForecastError(f"{alias}: values outside plausible GW range [0, {maximum:g}]")
        if not np.isfinite(features[known].reindex(expected).to_numpy(float)).all():
            raise NuclearCWEForecastError(f"{alias}: nonfinite residual replay/future feature")
        if not np.allclose(features[known].reindex(expected), mandatory[:, 1], rtol=0, atol=1e-9):
            raise NuclearCWEForecastError(f"{alias}: residual features differ from PIT known future")
        if alias in CWE_AVAILABILITY_SERIES:
            days = values.loc[expected, known].groupby(expected.tz_convert(timezone).date)
            if (days.max() - days.min() > 1e-9).any():
                raise NuclearCWEForecastError(f"{alias}: forecast daily Pmax must be broadcast unchanged")
        diagnostics[alias] = {
            "required_hours": len(expected), "missing_required_hours": 0,
            "optional_pre_anchor_context_hours": len(optional),
            "missing_optional_pre_anchor_context_hours": int(optional[known].isna().sum()),
            "pre_anchor_policy": "preserve_incumbent_support_no_imputation",
        }
    return context, diagnostics, expected


def _validate_residual_factory(factory: Callable[[], Any]) -> None:
    from .models.residual_corrector import ResidualMetaFeatureBuilder

    prototype = factory()
    builder = getattr(prototype, "feature_builder", None) or ResidualMetaFeatureBuilder(
        **getattr(prototype, "feature_builder_options", {}))
    if not builder.exclude_historical_prices:
        raise NuclearCWEForecastError("CWE residual recipe must retain the incumbent historical-price exclusion")
    for alias in CWE_NUCLEAR_ALIASES:
        if builder._is_excluded(f"known_{alias}_oracle"):
            raise NuclearCWEForecastError(f"{alias}: residual builder excludes this nuclear input")


class _CWEKalmanBuilder:
    """File-backed adapter; the downstream filter and its tuning stay unchanged."""

    def __init__(self, context: pd.DataFrame, delegate: Callable[..., Any], signature: str,
                 isolate_delegate: bool = False):
        self.context, self.delegate, self.signature = context, delegate, signature
        self.isolate_delegate = isolate_delegate
        self.used_config = None
        self.used_covariates = None

    def __call__(self, **kwargs):
        incoming = _frame(kwargs["covariates"], "Kalman covariates")
        for alias in CWE_AVAILABILITY_SERIES:
            incoming[alias] = self.context[f"known_{alias}_oracle"].reindex(incoming.index)
        if not np.isfinite(incoming[list(CWE_NUCLEAR_ALIASES)].to_numpy(float)).all():
            raise NuclearCWEForecastError("CWE Kalman inputs are incomplete")
        selected = nuclear_cwe_kalman_covariate_config(kwargs["covariate_config"])
        self.used_config = selected
        self.used_covariates = incoming.rename_axis("delivery_start_utc").reset_index()
        kwargs = dict(kwargs, covariates=self.used_covariates.copy(deep=True), covariate_config=selected)
        # The standard adapter is already pinned by input_protocol in the epoch.
        # Only a custom research delegate needs another namespace. Avoid adding
        # unnecessary path length to Windows' already deep rolling cache.
        if self.isolate_delegate and kwargs.get("rolling_refit_cache_dir") is not None:
            kwargs["rolling_refit_cache_dir"] = Path(kwargs["rolling_refit_cache_dir"]) / self.signature[:24]
        return self.delegate(**kwargs)


def run_nuclear_cwe_forecast(
    *, config: Mapping[str, Any], data: Any, zone: str, delivery_day: str | date,
    workdir: str | Path, device: str = "auto", threads: int = 4, workers: int = 1,
    runtime_factory: Callable[..., Any] | None = None,
    forecasting_module: Any | None = None,
    residual_factory: Callable[[], Any] | None = None,
    kalman_builder: Callable[..., Any] | None = None,
) -> base.NuclearForecastResult:
    """Run the unchanged three-stage replay with audited FR/BE/NL inputs.

    The caller creates an isolated pinned configuration and initializes its
    epoch to the incumbent's anchor. No live configuration, source or report
    is changed by this adapter. Hooks preserve the base engine's test API.
    """
    from run_chronos2_hourly import _residual_corrector_factory

    resolved = deepcopy(dict(config))
    _isolated_paths(resolved, workdir)
    if resolved.get("nuclear_experiment", {}).get("input_protocol") != nuclear_cwe_input_protocol():
        raise NuclearCWEForecastError("CWE input_protocol must pin the current adapter SHA before preparation")
    code = str(zone).upper()
    parsed = pd.Timestamp(delivery_day)
    if pd.isna(parsed) or parsed.tzinfo is not None or parsed != parsed.normalize():
        raise NuclearCWEForecastError("CWE delivery_day must be an explicit civil date")
    day = parsed.date()
    context, coverage, expected = _prepare_inputs(resolved, data, code, day)
    factory, upstream = _residual_corrector_factory(resolved, timezone=str(data.timezone))
    if factory is None or upstream != "chronos2":
        raise NuclearCWEForecastError("CWE requires the incumbent Chronos-based residual recipe")
    _validate_residual_factory(residual_factory or factory)
    delegate_identity = (base._factory_cache_identity(kalman_builder)
                         if kalman_builder is not None else "unchanged_build_operational_kalman_view")
    signature = hashlib.sha256(json.dumps(
        {"adapter_sha256": nuclear_cwe_code_sha256(), "delegate": delegate_identity},
        sort_keys=True, default=str).encode()).hexdigest()
    hook = _CWEKalmanBuilder(context, kalman_builder or build_operational_kalman_view, signature,
                             isolate_delegate=kalman_builder is not None)
    result = base.run_nuclear_forecast(
        config=resolved, data=data, zone=code, delivery_day=day, workdir=workdir,
        device=device, threads=threads, workers=workers, runtime_factory=runtime_factory,
        forecasting_module=forecasting_module, residual_factory=residual_factory, kalman_builder=hook,
    )
    if hook.used_config is None or hook.used_covariates is None:
        raise NuclearCWEForecastError("CWE Kalman adapter was not executed")
    daily = result.residual_daily_audit
    if not {"generation_source", "residual_feature_columns"}.issubset(daily):
        raise NuclearCWEForecastError("CWE residual fitted-feature audit is required")
    fitted = daily.loc[daily.generation_source.eq("daily_prequential_refit"), "residual_feature_columns"]
    if fitted.empty or any(not all(f"known_{alias}_oracle" in columns for alias in CWE_NUCLEAR_ALIASES)
                           for columns in fitted):
        raise NuclearCWEForecastError("A fitted residual corrector dropped a CWE nuclear input")
    if str(result.audit.get("raw_history_start_day")) != expected[0].tz_convert(data.timezone).date().isoformat():
        raise NuclearCWEForecastError("CWE engine changed the pinned replay anchor")
    audit = deepcopy(dict(result.audit))
    audit.update(
        candidate_engine=CWE_ENGINE, candidate_variant="nuclear_cwe_kalman",
        nuclear_covariate_aliases=list(CWE_NUCLEAR_ALIASES),
        nuclear_covariate_semantics={
            base.NUCLEAR_ALIAS: {"series": base.NUCLEAR_SERIES, "unit": "GW", "semantic": "forecast_generation"},
            **{alias: {"series": series, "unit": "GW", "semantic": "forecast_available_capacity",
                       "daily_broadcast": True} for alias, series in CWE_AVAILABILITY_SERIES.items()},
        },
        cwe_input_coverage=coverage,
        cwe_adapter_sha256=nuclear_cwe_code_sha256(), cwe_kalman_hook_signature=signature,
        nuclear_chronos_context_columns=list(CWE_NUCLEAR_ALIASES),
        nuclear_chronos_known_future_columns=[f"known_{alias}_oracle" for alias in CWE_NUCLEAR_ALIASES],
        nuclear_residual_features=[f"known_{alias}_oracle" for alias in CWE_NUCLEAR_ALIASES],
        nuclear_kalman_market_features=list(CWE_NUCLEAR_ALIASES),
        kalman_covariate_config=hook.used_config.to_dict(),
        source_audit_responsibility="caller_pinned_PIT_sources_and_civil_D_minus_1_08_cutoff_audits",
        production_changed=False, sealed_live_contract_modified=False,
    )
    return replace(result, covariates=hook.used_covariates.copy(deep=True), audit=audit)


__all__ = ["CWE_ENGINE", "CWE_AVAILABILITY_SERIES", "CWE_NUCLEAR_ALIASES",
           "NuclearCWEForecastError", "nuclear_cwe_code_sha256", "nuclear_cwe_input_protocol",
           "nuclear_cwe_kalman_covariate_config", "run_nuclear_cwe_forecast"]
