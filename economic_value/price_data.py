"""Read-only, sealed inputs for a diagnostic correction of nuclear Kalman prices.

The source contains 365 baseline days, not 730: its older fundamental history
does not contain the same baseline forecast.  Never substitute the autonomous
warm-up prefix for missing Kalman predictions.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import extreme_runner, runner as base


class PriceDataError(ValueError):
    """The sealed price-correction inputs cannot support a fair comparison."""


_CORE = (
    "feature_fr_residual_load_gw", "feature_de_residual_load_gw",
    "feature_be_residual_load_gw", "feature_nl_residual_load_gw",
    "feature_fr_nuclear_generation_gw", "feature_residual_region_mean_gw",
    "feature_residual_region_range_gw", "feature_hour_sin", "feature_hour_cos",
    "feature_weekday_sin", "feature_weekday_cos", "feature_month_sin",
    "feature_month_cos", "feature_local_residual_load_gw",
    "feature_local_residual_vs_region_gw", "feature_previous_da_price_eur_mwh",
)
ALLOWED_FEATURES = frozenset((*_CORE, *(name + "_missing" for name in _CORE)))
REQUIRED_CORE = frozenset((*_CORE[:5], "feature_previous_da_price_eur_mwh"))
_FEATURE_METADATA = {"feature_eligible", "feature_pit_certified"}
_BINDING_KEYS = ("snapshot_files", "source_code_sha256", "evaluation_start",
                 "evaluation_end", "evaluation_days", "zones", "models")


def _validate_settings(config: dict) -> None:
    if not isinstance(config, dict):
        raise PriceDataError("Price configuration must be a mapping.")
    if config.get("baseline_model") != "nuclear_kalman" or config.get("candidate_model") != "nuclear_kalman_extreme":
        raise PriceDataError("Expected nuclear_kalman and nuclear_kalman_extreme.")
    if type(config.get("training_window_days")) is not int or config["training_window_days"] != 365:
        raise PriceDataError("The maximum training window must remain 365 calendar days.")
    minimum = config.get("minimum_training_days")
    if type(minimum) is not int or not 90 <= minimum <= 365:
        raise PriceDataError("minimum_training_days must be an integer between 90 and 365.")
    zones = config.get("zones")
    if not isinstance(zones, list) or not zones or any(not isinstance(z, str) for z in zones) or len(set(zones)) != len(zones):
        raise PriceDataError("Select the exact, unique countries of the frozen baseline.")
    if not isinstance(config.get("source_expert_snapshot"), (str, Path)) or not str(config["source_expert_snapshot"]).strip():
        raise PriceDataError("An explicit source_expert_snapshot is required; no latest pointer is followed.")


def _validate_features(panel: pd.DataFrame, audit: dict) -> tuple[list[str], list[str]]:
    names, required = audit.get("feature_columns"), audit.get("required_core_feature_columns")
    if not isinstance(names, list) or any(not isinstance(n, str) for n in names) or len(set(names)) != len(names):
        raise PriceDataError("A unique explicit source feature_columns allowlist is required.")
    if set(names) != ALLOWED_FEATURES:
        raise PriceDataError("Source feature allowlist differs from the 32 audited fundamentals/calendar/previous-price features; labels, Storm and eligibility flags are forbidden predictors.")
    if not isinstance(required, list) or len(set(required)) != len(required) or set(required) != REQUIRED_CORE:
        raise PriceDataError("The six required core features must match the frozen feature audit.")
    declared = {name for name in panel if name.startswith("feature_")} - _FEATURE_METADATA
    if declared != set(names) or not _FEATURE_METADATA.issubset(panel):
        raise PriceDataError("Panel features do not match the explicit source allowlist.")
    for name in names:
        if not pd.api.types.is_numeric_dtype(panel[name]) or pd.api.types.is_bool_dtype(panel[name]):
            raise PriceDataError(f"Numeric feature required: {name}.")
        values = panel[name].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise PriceDataError(f"Infinite feature values are forbidden: {name}.")
        if name.endswith("_missing"):
            expected = panel[name.removesuffix("_missing")].isna().to_numpy(dtype=float)
            if not np.array_equal(values, expected):
                raise PriceDataError(f"Feature missingness flag disagrees with its source: {name}.")
    expected_eligible = np.isfinite(panel[required].to_numpy(dtype=float)).all(axis=1)
    if not pd.api.types.is_bool_dtype(panel.feature_eligible) or not np.array_equal(panel.feature_eligible.to_numpy(), expected_eligible):
        raise PriceDataError("feature_eligible must describe core availability, never be a predictor.")
    if audit.get("storm_used_as_feature") is not False or audit.get("selected_cutoff_violations") != 0:
        raise PriceDataError("Source features must exclude Storm and report zero selected cutoff violations.")
    return names, required


def _bind_baseline_results(snapshot: Path, panel: pd.DataFrame, audit: dict, zones: list[str]) -> pd.DataFrame:
    """A valid hash alone does not bind a transplanted result to its panel."""
    rows = pd.read_parquet(snapshot / "rows.parquet")
    if not {"model", "strategy", *panel.columns}.issubset(rows):
        raise PriceDataError("Source result is missing baseline input fields.")
    rows = rows.loc[rows.model.eq("nuclear_kalman") & rows.strategy.eq("model")]
    keys = ["zone", "timestamp_utc", "model"]
    if len(rows) != len(panel) or rows.duplicated(keys).any():
        raise PriceDataError("Source results do not retain every baseline hour exactly once.")
    try:
        pd.testing.assert_frame_equal(
            panel.sort_values(keys).reset_index(drop=True),
            rows[panel.columns].sort_values(keys).reset_index(drop=True),
            check_dtype=False, check_exact=True,
        )
    except AssertionError as exc:
        raise PriceDataError("Source result baseline fields differ from the sealed panel.") from exc
    metrics = pd.read_parquet(snapshot / "metrics.parquet")
    if not {"model", "zone", "strategy"}.issubset(metrics):
        raise PriceDataError("Missing source baseline metrics.")
    metrics = metrics.loc[metrics.model.eq("nuclear_kalman")].copy()
    expected = {(zone, strategy) for zone in [*zones, "PORTFOLIO"] for strategy in ("no_forecast", "benchmark", "model")}
    if metrics.duplicated(["zone", "strategy"]).any() or set(zip(metrics.zone, metrics.strategy)) != expected:
        raise PriceDataError("Missing or duplicated country/portfolio baseline metrics.")
    original = pd.DataFrame(audit.get("source_baseline_metrics", []))
    if not {"zone", "strategy", "model"}.issubset(original) or not original.model.eq("nuclear_kalman").all():
        raise PriceDataError("Original EVA baseline metric evidence is required.")
    if original.duplicated(["zone", "strategy"]).any() or set(zip(original.zone, original.strategy)) != expected:
        raise PriceDataError("Original EVA baseline metrics have incomplete support.")
    original = original.set_index(["zone", "strategy"])
    frozen = metrics.set_index(["zone", "strategy"]).reindex(original.index)
    for name in ("pnl_net_eur", "pnl_gross_eur", "trading_cost_eur", "absolute_energy_mwh", "eligible_hours", "active_hours", "max_drawdown_eur"):
        if name not in frozen or name not in original or not np.allclose(original[name].to_numpy(float), frozen[name].to_numpy(float), rtol=1e-10, atol=1e-6, equal_nan=True):
            raise PriceDataError(f"Source baseline metric differs from original EVA: {name}.")
    return metrics


def load_price_inputs(root: Path | str, config: dict) -> tuple[pd.DataFrame, dict, dict]:
    """Load exactly the frozen baseline year; never read or infer older forecasts.

    Returns the unchanged baseline panel (including optional NaNs), a provenance
    and warm-up audit, and the original economic configuration without reallocating
    capacity.  The caller must use ``audit['feature_columns']``, not a prefix scan.
    No source file, production configuration or output is modified here.
    """
    _validate_settings(config)
    root = Path(root).resolve()
    snapshot = base._output(root, config["source_expert_snapshot"])
    files = sorted(extreme_runner.INPUTS | extreme_runner.RESULTS | {"manifest.json", "results_manifest.json"})
    before = {name: base.digest(snapshot / name) for name in files}
    snapshot, source_config, baseline_config, manifest, panel = extreme_runner.read_snapshot(snapshot, root=root)
    results = json.loads((snapshot / "results_manifest.json").read_text(encoding="utf-8"))
    for name in _BINDING_KEYS:
        if name not in manifest or results.get(name) != manifest[name]:
            raise PriceDataError(f"Source expert result/input manifest mismatch: {name}.")
    if results.get("status") != "completed_hypothetical_diagnostic" or results.get("forecast_values_unchanged") is not True:
        raise PriceDataError("A completed source expert with unchanged forecasts is required.")
    extreme_runner._verify_files(snapshot, results.get("result_files", {}), extreme_runner.RESULTS)
    base.validate_config(baseline_config)
    zones = config["zones"]
    if set(zones) != set(source_config["zones"]) or set(zones) != set(baseline_config["zones"]):
        raise PriceDataError("Keep every frozen source country; a subset would change the original MW allocation.")
    if baseline_config["models"] != ["nuclear_kalman"] or not panel.model.eq("nuclear_kalman").all():
        raise PriceDataError("The source must contain the nuclear_kalman baseline, not a substitute or identity prefix.")
    if not panel["sample"].eq("evaluation").all():
        raise PriceDataError("This price experiment retains only the exact 365-day evaluation source, without extra live rows.")
    coverage = base._validate_panel(panel, baseline_config)
    if not np.isfinite(panel.forecast.to_numpy(dtype=float)).all():
        raise PriceDataError("Missing baseline forecasts cannot be invented or replaced by autonomous forecasts.")
    feature_audit = json.loads((snapshot / "feature_audit.json").read_text(encoding="utf-8"))
    names, required = _validate_features(panel, feature_audit)
    needed = {"label_available_at_utc", "label_publication_time_assumed"}
    if not needed.issubset(panel) or not panel.label_publication_time_assumed.eq(True).all():
        raise PriceDataError("Frozen label availability and its explicit publication-time assumption are required.")
    days = pd.DatetimeIndex(panel.timestamp_utc).tz_convert(baseline_config["timezone"]).tz_localize(None).normalize()
    expected_label_time = (days - pd.Timedelta(days=1) + pd.Timedelta(hours=18)).tz_localize(baseline_config["timezone"]).tz_convert("UTC")
    if not pd.DatetimeIndex(panel.label_available_at_utc).equals(expected_label_time):
        raise PriceDataError("Label availability must retain the source's D-1 18:00 civil assumption.")
    metrics = _bind_baseline_results(snapshot, panel, feature_audit, baseline_config["zones"])
    if any(base.digest(snapshot / name) != checksum for name, checksum in before.items()):
        raise PriceDataError("Source snapshot changed during loading; retry with an immutable completed source.")
    minimum = config["minimum_training_days"]
    first_possible = pd.Timestamp(coverage["evaluation_start"]) + pd.Timedelta(days=minimum)
    audit = {
        "schema_version": 1, "source_expert_snapshot": str(snapshot), **coverage,
        "baseline_model": "nuclear_kalman", "candidate_model": "nuclear_kalman_extreme",
        "zones": deepcopy(baseline_config["zones"]), "feature_columns": list(names),
        "required_core_feature_columns": list(required), "source_feature_audit": feature_audit,
        "source_baseline_metrics": json.loads(metrics.to_json(orient="records", double_precision=15)),
        "source_hashes": [{"path": str(snapshot / name), "sha256": before[name]} for name in files],
        "source_manifest_sha256": before["manifest.json"],
        "source_result_manifest_sha256": before["results_manifest.json"],
        "available_baseline_days": 365, "pre_evaluation_baseline_days": 0,
        "training_window_days": 365, "minimum_training_days": minimum,
        "training_window_policy": "past-only progressive history capped at 365 calendar days; no invented pre-evaluation baseline",
        "warmup_baseline_days": minimum, "warmup_last_day": (first_possible-pd.Timedelta(days=1)).date().isoformat(),
        "first_possible_fit_day": first_possible.date().isoformat(),
        "potential_post_warmup_days": max(0, 365-minimum),
        "strict_365_training_available_in_evaluation": False,
        "evaluation_prices_forecasts_references_unchanged": True,
        "source_result_baseline_fields_verified": True,
        "storm_used_as_feature": False, "eligibility_flags_used_as_features": False,
        "selected_cutoff_violations": 0, "baseline_neural_oof_certified": False,
        "forecast_and_label_vintages_certified": False,
        "label_publication_time_assumed": True, "diagnostic_only": True,
        "production_modified": False, "activation_performed": False,
        "limitations": [
            "Only 365 nuclear_kalman baseline days are frozen; no preceding baseline calibration year exists in this source.",
            "The first minimum_training_days retain the baseline. Later fits use growing past history capped at 365 days, not a full 365-day training window at every origin.",
            "The year has already been examined: this is retrospective development, not an untouched final test.",
            "Forecast/label vintage and neural OOF evidence remain uncertified; publication availability uses the frozen civil-time assumption.",
            "Previous-day day-ahead prices are a non-executable proxy; economic results do not demonstrate realizable trading profit.",
        ],
    }
    return panel, audit, deepcopy(baseline_config)
