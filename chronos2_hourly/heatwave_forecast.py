"""Isolated temperature/forecast-heat extension of the frozen nuclear recipe.

The existing Chronos, residual and Kalman algorithms remain unchanged. New
audited PIT channels enter all three stages. Existing results are never used as
replacement historical predictions for this new input schema.
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
from .heatwave_features import (HeatwaveFeatureConfig, heatwave_feature_aliases,
    temperature_aliases, feature_metadata, heatwave_features_sha256,
    validate_heatwave_aggregates, HEAT_FRACTION_ALIAS, COOLING_MEAN_ALIAS)
from .kalman_covariates import KalmanCovariateConfig, BASE_RESIDUAL_LOAD_COVARIATES
from .kalman_residual import build_operational_kalman_view

HEATWAVE_ENGINE = "heatwave_forecast_v1"


class HeatwaveForecastError(base.NuclearForecastError):
    """A heatwave-only schema, source or isolation check failed."""


def _feature_config(config: Mapping[str, Any]) -> HeatwaveFeatureConfig:
    return HeatwaveFeatureConfig.from_mapping(config.get("nuclear_experiment", {}).get("heatwave_features"))


def heatwave_code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def heatwave_input_protocol(feature_config: HeatwaveFeatureConfig | Mapping[str, Any] | None = None) -> str:
    selected = feature_config if isinstance(feature_config, HeatwaveFeatureConfig) else HeatwaveFeatureConfig.from_mapping(feature_config)
    payload = {"adapter_sha256": heatwave_code_sha256(), "features_sha256": heatwave_features_sha256(),
               "feature_config": selected.to_dict()}
    return "civil_pit_heatwave_v1_" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _frame(value: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame) or value.columns.has_duplicates:
        raise HeatwaveForecastError(f"{name}: DataFrame with unique columns required.")
    result = value.copy(deep=True)
    if not isinstance(result.index, pd.DatetimeIndex):
        columns = [c for c in ("timestamp", "delivery_start_utc", "timestamp_utc") if c in result]
        if len(columns) != 1:
            raise HeatwaveForecastError(f"{name}: one explicit time column required.")
        values = result.pop(columns[0])
        if any(pd.isna(v) or pd.Timestamp(v).tzinfo is None for v in values):
            raise HeatwaveForecastError(f"{name}: aware timestamps required.")
        result.index = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    if result.index.tz is None or result.index.hasnans or result.index.has_duplicates:
        raise HeatwaveForecastError(f"{name}: aware unique timestamp index required.")
    result.index = result.index.tz_convert("UTC")
    if not result.index.is_monotonic_increasing or not result.index.equals(result.index.floor("h")):
        raise HeatwaveForecastError(f"{name}: increasing physical HH:00 index required.")
    return result


def _isolated_paths(config: Mapping[str, Any], workdir: str | Path) -> None:
    project = Path(str(config.get("data", {}).get("project_root", "."))).resolve()
    allowed = project / "runs/experiments/heatwave_v1"
    work = Path(workdir).resolve()
    if not work.is_relative_to(allowed) or work == allowed or (work / "artifact_checksums.json").exists():
        raise HeatwaveForecastError("Heatwave workdir must be an unsealed child of its own runs/experiments/heatwave_v1.")
    experiment = config.get("nuclear_experiment", {})
    if experiment.get("mode", "full") == "incremental":
        cache = Path(str(experiment.get("incremental_cache_dir", ""))).resolve()
        if not cache.is_relative_to(allowed) or cache == allowed:
            raise HeatwaveForecastError("Heatwave cache must belong to this isolated experiment.")


def _extra_aliases(config: Mapping[str, Any], zone: str, selected: HeatwaveFeatureConfig) -> tuple[str, ...]:
    aliases = heatwave_feature_aliases(selected)
    known = {*BASE_RESIDUAL_LOAD_COVARIATES, base.NUCLEAR_ALIAS, *aliases}
    specs = config.get("zones", {}).get(zone, {}).get("covariates", {})
    # Optional inherited Pmax channels remain effective when a CWE baseline is
    # explicitly chosen. Never invent them for the default FR-nuclear baseline.
    extras = tuple(alias for alias, spec in specs.items() if spec.get("enabled", True) and alias not in known)
    return tuple(dict.fromkeys((*aliases, *extras)))


def heatwave_kalman_covariate_config(
    original: KalmanCovariateConfig | None = None, *,
    feature_config: HeatwaveFeatureConfig | Mapping[str, Any] | None = None,
    extra_aliases: tuple[str, ...] = (),
) -> KalmanCovariateConfig:
    original = original or base.nuclear_kalman_covariate_config()
    additions = (*heatwave_feature_aliases(feature_config), *extra_aliases)
    result = replace(original,
        input_columns=tuple(dict.fromkeys((*original.input_columns, *additions))),
        groups={name: tuple(dict.fromkeys((*columns, *additions))) for name, columns in original.groups.items()},
        history_missing_policy="complete_trailing", minimum_history_coverage=1.0, require_future_complete=True)
    result.validate()
    return result


def _prepare_inputs(config: Mapping[str, Any], data: Any, zone: str, day: date):
    from run_chronos2_hourly import _feature_inputs

    selected = _feature_config(config)
    timezone = base.ZONE_TIMEZONES.get(zone)
    if timezone is None or str(data.zone).upper() != zone or str(data.timezone) != timezone:
        raise HeatwaveForecastError("Heatwave zone and prepared timezone must match.")
    aliases = _extra_aliases(config, zone, selected)
    specs = config.get("zones", {}).get(zone, {}).get("covariates", {})
    for alias in aliases:
        spec = specs.get(alias, {})
        if (spec.get("enabled") is not True or spec.get("source") != "pit_parquet"
                or spec.get("fill_method", "none") != "none"
                or spec.get("future", {}).get("known_future") is not True
                or spec.get("future", {}).get("strategies") != ["oracle"]):
            raise HeatwaveForecastError(f"{alias}: explicit enabled unfilled PIT known-future source required.")
        if any(token in str(spec.get("series", "")).lower() for token in ("observed", "actual", "reanalysis")):
            raise HeatwaveForecastError(f"{alias}: realised or reanalysis source cannot be a forecast feature.")
    context = _frame(data.model_context_covariates, "model_context_covariates")
    start = pd.Timestamp(config.get("nuclear_experiment", {}).get("raw_history_start_day", day - timedelta(days=730)))
    if (pd.isna(start) or start.tzinfo is not None or start != start.normalize()
            or not day - timedelta(days=1095) <= start.date() <= day - timedelta(days=730)):
        raise HeatwaveForecastError("Heatwave anchored raw history must cover 730 to 1095 civil days.")
    expected = pd.date_range(start.tz_localize(timezone), pd.Timestamp(day + timedelta(days=1), tz=timezone),
                             freq="h", inclusive="left").tz_convert("UTC")
    if len(expected.difference(context.index)):
        raise HeatwaveForecastError("Heatwave context misses replay/future physical hours.")
    target, _, future, features = _feature_inputs(data, config)
    history = _frame(data.covariates, "historical covariates")
    required_history = expected.intersection(target.index)
    diagnostics = {}
    for alias in (base.NUCLEAR_ALIAS, *aliases):
        known = f"known_{alias}_oracle"
        if (alias not in context or alias not in history or known not in context
                or known not in data.known_future_columns or known not in features or known not in future):
            raise HeatwaveForecastError(f"{alias}: required input dropped from Chronos/history/residual/future.")
        values = context[[alias, known]].apply(pd.to_numeric, errors="coerce")
        mandatory = values.reindex(expected)
        if not np.isfinite(mandatory.to_numpy(float)).all() or not np.allclose(mandatory[alias], mandatory[known], rtol=0, atol=1e-9):
            raise HeatwaveForecastError(f"{alias}: incomplete or divergent PIT context and known-future vintages.")
        old = pd.to_numeric(history[alias].reindex(required_history), errors="coerce")
        if not np.isfinite(old).all() or not np.allclose(old, mandatory[alias].reindex(required_history), rtol=0, atol=1e-9):
            raise HeatwaveForecastError(f"{alias}: historical/context covariates differ.")
        if not np.isfinite(features[known].reindex(expected)).all() or not np.allclose(
                features[known].reindex(expected), mandatory[known], rtol=0, atol=1e-9):
            raise HeatwaveForecastError(f"{alias}: residual features differ from the PIT input.")
        finite = np.isfinite(values.to_numpy(float))
        pairs = finite.all(axis=1)
        if not np.array_equal(finite[:, 0], finite[:, 1]) or not np.allclose(
                values.loc[pairs, alias], values.loc[pairs, known], rtol=0, atol=1e-9):
            raise HeatwaveForecastError(f"{alias}: optional historical PIT vintages differ.")
        if alias in heatwave_feature_aliases(selected):
            blocks = mandatory[known].groupby(expected.tz_convert(timezone).date)
            if (blocks.max() - blocks.min() > 1e-9).any():
                raise HeatwaveForecastError(f"{alias}: national daily heat indices must be broadcast unchanged.")
            vector = mandatory[known].to_numpy(float)
            if alias in temperature_aliases(selected):
                valid = (vector >= -60) & (vector <= 60)
            elif alias.endswith("_heat_streak_fcst_days"):
                valid = (vector >= 0) & (vector <= selected.streak_clip_days) & (vector == np.floor(vector))
            elif alias == HEAT_FRACTION_ALIAS:
                valid = (vector >= 0) & (vector <= 1)
            else:
                valid = (vector >= 0) & (vector <= 120)
            if not valid.all():
                raise HeatwaveForecastError(f"{alias}: feature bounds/units invalid.")
        diagnostics[alias] = {"required_hours": len(expected), "missing_required_hours": 0,
            "optional_pre_anchor_context_hours": int((context.index < expected[0]).sum()),
            "missing_optional_pre_anchor_context_hours": int(values.loc[values.index < expected[0], known].isna().sum()),
            "pre_anchor_policy": "preserve_incumbent_support_no_imputation"}
    feature_aliases = heatwave_feature_aliases(selected)
    aggregate_frame = context.loc[expected, [f"known_{a}_oracle" for a in feature_aliases]].rename(
        columns={f"known_{a}_oracle": a for a in feature_aliases})
    diagnostics[HEAT_FRACTION_ALIAS]["aggregate_precision"] = validate_heatwave_aggregates(aggregate_frame, selected)
    return context, aliases, diagnostics, expected


class _HeatwaveKalmanBuilder:
    def __init__(self, context: pd.DataFrame, aliases: tuple[str, ...], selected: HeatwaveFeatureConfig,
                 delegate: Callable[..., Any], delegate_signature: str | None):
        self.context, self.aliases, self.selected = context, aliases, selected
        self.delegate, self.delegate_signature = delegate, delegate_signature
        self.used_config = self.used_covariates = None

    def __call__(self, **kwargs):
        incoming = _frame(kwargs["covariates"], "Kalman covariates")
        for alias in self.aliases:
            incoming[alias] = self.context[f"known_{alias}_oracle"].reindex(incoming.index)
        if not np.isfinite(incoming.to_numpy(float)).all():
            raise HeatwaveForecastError("Heatwave Kalman inputs are incomplete.")
        selected = heatwave_kalman_covariate_config(kwargs["covariate_config"],
            feature_config=self.selected, extra_aliases=self.aliases)
        self.used_config = selected
        self.used_covariates = incoming.rename_axis("timestamp").reset_index()
        kwargs = dict(kwargs, covariates=self.used_covariates.copy(deep=True), covariate_config=selected)
        if kwargs.get("rolling_refit_cache_dir") is not None:
            path = Path(kwargs["rolling_refit_cache_dir"])
            if self.delegate_signature:
                path = path / self.delegate_signature[:24]
            if os.name == "nt":
                absolute = str(path.resolve())
                if not absolute.startswith("\\\\?\\"):
                    absolute = "\\\\?\\UNC\\" + absolute[2:] if absolute.startswith("\\\\") else "\\\\?\\" + absolute
                path = Path(absolute)
            kwargs["rolling_refit_cache_dir"] = path
        return self.delegate(**kwargs)


def run_heatwave_forecast(
    *, config: Mapping[str, Any], data: Any, zone: str, delivery_day: str | date,
    workdir: str | Path, device: str = "auto", threads: int = 4, workers: int = 1,
    runtime_factory: Callable[..., Any] | None = None, forecasting_module: Any | None = None,
    residual_factory: Callable[[], Any] | None = None, kalman_builder: Callable[..., Any] | None = None,
) -> base.NuclearForecastResult:
    """Replay the nuclear recipe with the frozen temperature/heat input schema."""
    from run_chronos2_hourly import _residual_corrector_factory
    from .models.residual_corrector import ResidualMetaFeatureBuilder

    resolved = deepcopy(dict(config))
    selected = _feature_config(resolved)
    _isolated_paths(resolved, workdir)
    if resolved.get("nuclear_experiment", {}).get("input_protocol") != heatwave_input_protocol(selected):
        raise HeatwaveForecastError("Heatwave input_protocol must pin current adapter, feature code and settings before preparation.")
    code = str(zone).upper()
    parsed = pd.Timestamp(delivery_day)
    if pd.isna(parsed) or parsed.tzinfo is not None or parsed != parsed.normalize():
        raise HeatwaveForecastError("delivery_day must be an explicit civil date.")
    day = parsed.date()
    context, aliases, coverage, expected = _prepare_inputs(resolved, data, code, day)
    factory, upstream = _residual_corrector_factory(resolved, timezone=str(data.timezone))
    if factory is None or upstream != "chronos2":
        raise HeatwaveForecastError("Heatwave requires the incumbent Chronos residual recipe.")
    prototype = (residual_factory or factory)()
    builder = getattr(prototype, "feature_builder", None) or ResidualMetaFeatureBuilder(
        **getattr(prototype, "feature_builder_options", {}))
    if not builder.exclude_historical_prices or any(builder._is_excluded(f"known_{a}_oracle")
            for a in (base.NUCLEAR_ALIAS, *aliases)):
        raise HeatwaveForecastError("Residual recipe must exclude historical prices and retain every heat/nuclear feature.")
    # Validate dimensionality before any costly Chronos inference.
    heatwave_kalman_covariate_config(feature_config=selected, extra_aliases=aliases)
    custom_signature = (hashlib.sha256(json.dumps(base._factory_cache_identity(kalman_builder),
        sort_keys=True, default=str).encode()).hexdigest() if kalman_builder is not None else None)
    hook = _HeatwaveKalmanBuilder(context, aliases, selected, kalman_builder or build_operational_kalman_view,
                                  custom_signature)
    result = base.run_nuclear_forecast(config=resolved, data=data, zone=code, delivery_day=day,
        workdir=workdir, device=device, threads=threads, workers=workers, runtime_factory=runtime_factory,
        forecasting_module=forecasting_module, residual_factory=residual_factory, kalman_builder=hook)
    if hook.used_config is None or hook.used_covariates is None:
        raise HeatwaveForecastError("Heatwave Kalman input adapter was not executed.")
    daily = result.residual_daily_audit
    if not {"generation_source", "residual_feature_columns"}.issubset(daily):
        raise HeatwaveForecastError("Heatwave fitted residual-feature audit is required.")
    fitted = daily.loc[daily.generation_source.eq("daily_prequential_refit"), "residual_feature_columns"]
    if fitted.empty or any(not all(f"known_{a}_oracle" in columns for a in (base.NUCLEAR_ALIAS, *aliases)) for columns in fitted):
        raise HeatwaveForecastError("A fitted residual corrector dropped a heat/nuclear input.")
    if str(result.audit.get("raw_history_start_day")) != expected[0].tz_convert(data.timezone).date().isoformat():
        raise HeatwaveForecastError("Heatwave engine changed the pinned replay anchor.")
    audit = deepcopy(dict(result.audit))
    audit.update(candidate_engine=HEATWAVE_ENGINE, candidate_variant="heatwave_kalman",
        heatwave_base_variant="nuclear_cwe" if "be_nuclear_available_gw" in aliases or "nl_nuclear_available_gw" in aliases else "nuclear_fr",
        heatwave_feature_config=selected.to_dict(), heatwave_feature_aliases=list(heatwave_feature_aliases(selected)),
        heatwave_feature_metadata=feature_metadata(selected), heatwave_input_coverage=coverage,
        heatwave_aggregate_precision=coverage[HEAT_FRACTION_ALIAS]["aggregate_precision"],
        heatwave_adapter_sha256=heatwave_code_sha256(), heatwave_features_sha256=heatwave_features_sha256(),
        heatwave_input_protocol=heatwave_input_protocol(selected), heatwave_kalman_delegate_signature=custom_signature,
        additional_covariate_aliases=[a for a in aliases if a not in heatwave_feature_aliases(selected)],
        heatwave_chronos_context_columns=list(aliases),
        heatwave_chronos_known_future_columns=[f"known_{a}_oracle" for a in aliases],
        heatwave_residual_features=[f"known_{a}_oracle" for a in aliases],
        heatwave_kalman_features=list(aliases), kalman_covariate_config=hook.used_config.to_dict(),
        source_audit_responsibility="caller_pinned_raw_PIT_temperature_and_causal_heat_features_at_civil_D_minus_1_08",
        production_changed=False, sealed_live_contract_modified=False)
    return replace(result, covariates=hook.used_covariates.copy(deep=True), audit=audit)
