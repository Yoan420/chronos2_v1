"""Four hourly solar forecasts added to the unchanged NuclearFR model chain.

No ramp, regional aggregate, classifier or price correction is introduced here.
The existing Chronos, residual recipe and governed Kalman are replayed with
four additional audited GW inputs. Source materialization belongs to the caller.
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

from . import nuclear_forecast as base
from .nuclear_cwe_forecast import _frame
from .kalman_covariates import BASE_RESIDUAL_LOAD_COVARIATES, KalmanCovariateConfig
from .kalman_residual import build_operational_kalman_view
from chronos2_modular.common import CALENDAR_COLUMNS


ENGINE = "solar_cwe_v1"
SOLAR_SERIES = {f"{zone}_solar_generation_fcst": f"power.{zone}.generation.solar.hourly.gw.fcst"
                for zone in ("fr", "de", "be", "nl")}
SOLAR_ALIASES = tuple(SOLAR_SERIES)
SOLAR_KNOWN_COLUMNS = tuple(f"known_{alias}_oracle" for alias in SOLAR_ALIASES)
INPUT_ALIASES = (*BASE_RESIDUAL_LOAD_COVARIATES, base.NUCLEAR_ALIAS, *SOLAR_ALIASES)


class SolarCWEForecastError(base.NuclearForecastError):
    """An incomplete input, unintended schema or non-isolated path was refused."""


def solar_cwe_code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def solar_cwe_input_protocol() -> str:
    contract = {"engine": ENGINE, "adapter_sha256": solar_cwe_code_sha256(), "sources": SOLAR_SERIES,
                "units": "GW", "daily_broadcast": False, "cutoff": "civil_D_minus_1_08:00"}
    return "civil_pit_solar_cwe_v1_" + hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def solar_cwe_kalman_covariate_config(original: KalmanCovariateConfig | None = None) -> KalmanCovariateConfig:
    original = original or base.nuclear_kalman_covariate_config()
    result = replace(original,
        input_columns=tuple(dict.fromkeys((*original.input_columns, *SOLAR_ALIASES))),
        groups={name: tuple(dict.fromkeys((*columns, *SOLAR_ALIASES))) for name, columns in original.groups.items()},
        history_missing_policy="complete_trailing", minimum_history_coverage=1.0, require_future_complete=True)
    # Keep residual-load mean/spread unchanged: solar is neither a new aggregate
    # nor an extra subtraction from already-net residual load.
    result.validate()
    return result


def _isolated_paths(config: Mapping[str, Any], workdir: str | Path) -> None:
    project = Path(str(config.get("data", {}).get("project_root", "."))).resolve()
    allowed = project / "runs/experiments" / ENGINE
    work = Path(workdir).absolute()
    if (allowed.resolve() != allowed or work.resolve() != work or not work.is_relative_to(allowed)
            or work == allowed or (work / "artifact_checksums.json").exists()):
        raise SolarCWEForecastError("SolarCWE workdir must be an unsealed, unredirected child of runs/experiments/solar_cwe_v1.")
    experiment = config.get("nuclear_experiment", {})
    if experiment.get("mode", "full") == "incremental":
        cache = Path(str(experiment.get("incremental_cache_dir", ""))).absolute()
        if cache.resolve() != cache or not cache.is_relative_to(allowed) or cache == allowed:
            raise SolarCWEForecastError("SolarCWE daily caches must belong to the isolated SolarCWE namespace.")


def _prepare_inputs(config: Mapping[str, Any], data: Any, zone: str, day: date):
    from run_chronos2_hourly import _feature_inputs

    if zone not in {"FR", "DE", "BE", "NL"} or data.zone != zone or data.timezone != base.ZONE_TIMEZONES[zone]:
        raise SolarCWEForecastError("Prepared SolarCWE zone/timezone must match FR/DE/BE/NL.")
    specs = config.get("zones", {}).get(zone, {}).get("covariates", {})
    active = {alias for alias, spec in specs.items() if spec.get("enabled", True)}
    if active != set(INPUT_ALIASES):
        raise SolarCWEForecastError("SolarCWE accepts exactly baseline RL5 + FR nuclear + four solar inputs, no additional raw features.")
    for alias, series in SOLAR_SERIES.items():
        spec = specs[alias]
        if (spec.get("enabled") is not True or spec.get("source") != "pit_parquet" or spec.get("series") != series
                or spec.get("include_base_context", True) is not True or spec.get("fill_method") != "none"
                or spec.get("fill_limit", 0) != 0 or spec.get("unit") != "GW"
                or spec.get("semantic") != "forecast_generation" or spec.get("daily_broadcast") is not False
                or spec.get("future", {}).get("known_future") is not True
                or spec.get("future", {}).get("strategies") != ["oracle"]):
            raise SolarCWEForecastError(f"{alias}: exact hourly GW generation PIT, unfilled base context and known future required.")
    context = _frame(data.model_context_covariates, "SolarCWE model context")
    required_known = {f"known_{a}_oracle" for a in INPUT_ALIASES}
    allowed_future = required_known | set(CALENDAR_COLUMNS)
    allowed_columns = set(INPUT_ALIASES) | allowed_future
    if (set(context) - allowed_columns or set(data.known_future_columns) - allowed_future
            or not required_known.issubset(data.known_future_columns) or not set(INPUT_ALIASES).issubset(context)):
        raise SolarCWEForecastError("Unexpected model input: no realised solar, ramp, classifier, fuel or competing forecast is allowed.")
    if config.get("hourly", {}).get("supply_stack", {}).get("enabled", False):
        raise SolarCWEForecastError("SolarCWE does not enable additional supply-stack features.")
    start = pd.Timestamp(config.get("nuclear_experiment", {}).get("raw_history_start_day", day - timedelta(days=730)))
    if (pd.isna(start) or start.tzinfo is not None or start != start.normalize()
            or not day - timedelta(days=1095) <= start.date() <= day - timedelta(days=730)):
        raise SolarCWEForecastError("SolarCWE raw history requires the pinned 730-to-1095-day replay anchor.")
    expected = pd.date_range(start.tz_localize(data.timezone), pd.Timestamp(day + timedelta(days=1), tz=data.timezone),
                             freq="h", inclusive="left").tz_convert("UTC")
    if len(expected.difference(context.index)):
        raise SolarCWEForecastError("SolarCWE context misses replay or future physical hours.")
    target, _, future, features = _feature_inputs(data, config)
    if not required_known.issubset(features):
        raise SolarCWEForecastError("Baseline or solar known-future input dropped from residual feature selection.")
    history = _frame(data.covariates, "SolarCWE historical covariates")
    historical_required = expected.intersection(target.index)
    coverage = {}
    for alias in SOLAR_ALIASES:
        known = f"known_{alias}_oracle"
        if (alias not in context or alias not in history or known not in context or known not in data.known_future_columns
                or known not in features or known not in future):
            raise SolarCWEForecastError(f"{alias}: input dropped from Chronos context/future or residual features.")
        values = context[[alias, known]].apply(pd.to_numeric, errors="coerce")
        mandatory = values.reindex(expected)
        if not np.isfinite(mandatory.to_numpy(float)).all() or np.isinf(values.to_numpy(float)).any():
            raise SolarCWEForecastError(f"{alias}: every replay/future physical hour must be finite.")
        finite = np.isfinite(values.to_numpy(float))
        paired = finite.all(axis=1)
        if (not np.array_equal(finite[:, 0], finite[:, 1]) or not np.allclose(values.loc[paired, alias], values.loc[paired, known], rtol=0, atol=1e-9)
                or (values.to_numpy(float)[finite] < 0).any()):
            raise SolarCWEForecastError(f"{alias}: negative generation or divergent base/future vintages.")
        historical_values = pd.to_numeric(history[alias].reindex(historical_required), errors="coerce")
        selected = pd.to_numeric(features[known].reindex(expected), errors="coerce")
        if (not np.isfinite(historical_values).all() or not np.isfinite(selected).all()
                or not np.allclose(historical_values, mandatory[alias].reindex(historical_required), rtol=0, atol=1e-9)
                or not np.allclose(selected, mandatory[known], rtol=0, atol=1e-9)):
            raise SolarCWEForecastError(f"{alias}: historical or residual values diverge from the PIT input.")
        coverage[alias] = {"required_hours": len(expected), "missing_required_hours": 0,
            "optional_pre_anchor_context_hours": int((context.index < expected[0]).sum()),
            "missing_optional_pre_anchor_context_hours": int(values.loc[values.index < expected[0], known].isna().sum()),
            "pre_anchor_policy": "preserve_incumbent_support_no_imputation"}
    return context, coverage, expected


class _SolarKalmanBuilder:
    def __init__(self, context: pd.DataFrame, delegate: Callable[..., Any], signature: str | None):
        self.context, self.delegate, self.signature = context, delegate, signature
        self.used_config = self.used_covariates = None

    def __call__(self, **kwargs):
        incoming = _frame(kwargs["covariates"], "SolarCWE Kalman covariates")
        for alias in SOLAR_ALIASES:
            incoming[alias] = self.context[f"known_{alias}_oracle"].reindex(incoming.index)
        if set(incoming) != set(INPUT_ALIASES) or not np.isfinite(incoming.to_numpy(float)).all():
            raise SolarCWEForecastError("SolarCWE Kalman must retain complete RL5/nuclear/four-solar inputs only.")
        self.used_config = solar_cwe_kalman_covariate_config(kwargs["covariate_config"])
        self.used_covariates = incoming.rename_axis("timestamp").reset_index()
        kwargs = dict(kwargs, covariates=self.used_covariates.copy(deep=True), covariate_config=self.used_config)
        if kwargs.get("rolling_refit_cache_dir") is not None:
            path = Path(kwargs["rolling_refit_cache_dir"])
            if self.signature:
                path = path / self.signature[:24]
            if os.name == "nt":
                absolute = str(path.resolve())
                if not absolute.startswith("\\\\?\\"):
                    absolute = "\\\\?\\UNC\\" + absolute[2:] if absolute.startswith("\\\\") else "\\\\?\\" + absolute
                path = Path(absolute)
            kwargs["rolling_refit_cache_dir"] = path
        return self.delegate(**kwargs)


def run_solar_cwe_forecast(*, config: Mapping[str, Any], data: Any, zone: str, delivery_day: str | date,
    workdir: str | Path, device: str = "auto", threads: int = 4, workers: int = 1,
    runtime_factory: Callable[..., Any] | None = None, forecasting_module: Any | None = None,
    residual_factory: Callable[[], Any] | None = None, kalman_builder: Callable[..., Any] | None = None,
) -> base.NuclearForecastResult:
    """Recompute the unchanged NuclearFR chain with four added solar channels."""
    from run_chronos2_hourly import _residual_corrector_factory
    from .models.residual_corrector import ResidualMetaFeatureBuilder

    resolved = deepcopy(dict(config))
    _isolated_paths(resolved, workdir)
    if resolved.get("nuclear_experiment", {}).get("input_protocol") != solar_cwe_input_protocol():
        raise SolarCWEForecastError("SolarCWE input_protocol must pin the current adapter and four-source schema.")
    parsed = pd.Timestamp(delivery_day)
    if pd.isna(parsed) or parsed.tzinfo is not None or parsed != parsed.normalize():
        raise SolarCWEForecastError("SolarCWE delivery_day must be an explicit civil date.")
    day, code = parsed.date(), str(zone).upper()
    context, coverage, expected = _prepare_inputs(resolved, data, code, day)
    factory, upstream = _residual_corrector_factory(resolved, timezone=data.timezone)
    if factory is None or upstream != "chronos2":
        raise SolarCWEForecastError("SolarCWE requires the unchanged enabled Chronos-based residual recipe.")
    prototype = (residual_factory or factory)()
    builder = getattr(prototype, "feature_builder", None) or ResidualMetaFeatureBuilder(**getattr(prototype, "feature_builder_options", {}))
    required = (*SOLAR_KNOWN_COLUMNS, base.NUCLEAR_KNOWN_COLUMN)
    if not builder.exclude_historical_prices or any(builder._is_excluded(column) for column in required):
        raise SolarCWEForecastError("Residual recipe must retain solar/nuclear inputs and its historical-price exclusion.")
    solar_cwe_kalman_covariate_config()  # Dimension/group validation before any costly model call.
    signature = (hashlib.sha256(json.dumps(base._factory_cache_identity(kalman_builder), sort_keys=True,
                                           default=str).encode()).hexdigest() if kalman_builder is not None else None)
    hook = _SolarKalmanBuilder(context, kalman_builder or build_operational_kalman_view, signature)
    result = base.run_nuclear_forecast(config=resolved, data=data, zone=code, delivery_day=day, workdir=workdir,
        device=device, threads=threads, workers=workers, runtime_factory=runtime_factory,
        forecasting_module=forecasting_module, residual_factory=residual_factory, kalman_builder=hook)
    if hook.used_config is None or hook.used_covariates is None:
        raise SolarCWEForecastError("SolarCWE Kalman input adapter was not executed.")
    daily = result.residual_daily_audit
    if not {"generation_source", "residual_feature_columns"}.issubset(daily):
        raise SolarCWEForecastError("Fitted residual-feature audit is required.")
    fitted = daily.loc[daily.generation_source.eq("daily_prequential_refit"), "residual_feature_columns"]
    if fitted.empty or any(not set(required).issubset(columns) for columns in fitted):
        raise SolarCWEForecastError("A fitted residual corrector dropped a required solar/nuclear input.")
    if str(result.audit.get("raw_history_start_day")) != expected[0].tz_convert(data.timezone).date().isoformat():
        raise SolarCWEForecastError("SolarCWE engine changed the pinned replay anchor.")
    audit = deepcopy(dict(result.audit))
    audit.update(candidate_engine=ENGINE, candidate_variant="solar_cwe_kalman", solar_input_protocol=solar_cwe_input_protocol(),
        solar_adapter_sha256=solar_cwe_code_sha256(), solar_input_coverage=coverage, solar_kalman_delegate_signature=signature,
        solar_sources={alias: {"series": series, "unit": "GW", "semantic": "forecast_generation", "daily_broadcast": False}
                       for alias, series in SOLAR_SERIES.items()},
        solar_chronos_context_columns=list(SOLAR_ALIASES), solar_chronos_known_future_columns=list(SOLAR_KNOWN_COLUMNS),
        solar_residual_features=list(SOLAR_KNOWN_COLUMNS), solar_kalman_market_features=list(SOLAR_ALIASES),
        kalman_covariate_config=hook.used_config.to_dict(), baseline_inputs_preserved=True,
        additional_raw_input_count=4, new_ramps_or_aggregates=False, spike_classifier_added=False,
        inherited_residual_feature_builder_unchanged=True, production_pit_evidence=False,
        source_audit_responsibility="caller_pinned_hourly_solar_PIT_sources_and_civil_D_minus_1_08_cutoff_audits",
        promotion_eligible=False, production_changed=False, sealed_live_contract_modified=False)
    return replace(result, covariates=hook.used_covariates.copy(deep=True), audit=audit)


__all__ = ["ENGINE", "SOLAR_SERIES", "SOLAR_ALIASES", "SOLAR_KNOWN_COLUMNS", "SolarCWEForecastError",
           "solar_cwe_code_sha256", "solar_cwe_input_protocol", "solar_cwe_kalman_covariate_config", "run_solar_cwe_forecast"]
