"""Frozen price-residual experiment; never writes to operational forecast paths."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
import json
import logging
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import yaml

from chronos2_hourly.process_lock import exclusive_process_lock
from . import runner as base
from .engine import simulate
from .extreme_runner import _protected, _verify_files

LOGGER = logging.getLogger(__name__)
INPUTS = {"config.json", "baseline_config.json", "panel.parquet", "source_audit.json"}
RESULTS = {"decisions.parquet", "folds.parquet", "governance.parquet", "rows.parquet",
           "metrics.parquet", "daily.parquet", "breakdowns.parquet", "forecast_metrics.parquet",
           "forecast_daily.parquet", "model_audit.json"}


def validate_config(config: dict) -> None:
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Expected price-expert schema_version 1.")
    if config.get("baseline_model") != "nuclear_kalman" or config.get("candidate_model") != "nuclear_kalman_extreme":
        raise ValueError("This isolated price expert compares nuclear_kalman and nuclear_kalman_extreme.")
    if config.get("training_window_days") != 365:
        raise ValueError("The trailing training window must remain capped at 365 calendar days.")
    minimum = config.get("minimum_training_days")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or not 90 <= minimum <= 365:
        raise ValueError("minimum_training_days must be an explicit integer from 90 to 365.")
    zones = config.get("zones")
    if not isinstance(zones, list) or not zones or len(set(zones)) != len(zones) or set(zones)-base.ZONES:
        raise ValueError("Choose unique FR/DE/BE/NL zones.")
    for key in ("source_expert_snapshot", "output_root"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"Missing {key}.")
    if config.get("diagnostic_only") is not True or config.get("production_modified") is not False or config.get("order_execution_enabled") is not False:
        raise ValueError("Diagnostic only: no operational changes, activation or orders.")
    if not isinstance(config.get("expert"), dict) or not isinstance(config.get("governance"), dict):
        raise ValueError("Explicit expert and governance settings are required.")


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    validate_config(config)
    return config


def model_config(config: dict, baseline_config: dict, audit: dict) -> dict:
    if set(config["expert"]) & set(config["governance"]):
        raise ValueError("Expert/governance parameter collisions are forbidden.")
    if any(not key.startswith("governance_") for key in config["governance"]) or any(key.startswith("governance_") for key in config["expert"]):
        raise ValueError("Keep expert settings and governance_ settings in their dedicated sections.")
    locked = {"training_window_days", "minimum_training_days", "feature_columns", "required_feature_columns",
              "signal_threshold_eur_mwh", "transaction_cost_eur_mwh", "slippage_eur_mwh", "timezone"}
    if locked & (set(config["expert"]) | set(config["governance"])):
        raise ValueError("Training limits, input allowlist and frozen economic costs cannot be overridden inside expert/governance.")
    return {**config["expert"], **config["governance"], **baseline_config["strategy"],
            "training_window_days": config["training_window_days"], "minimum_training_days": config["minimum_training_days"],
            "timezone": "Europe/Paris", "feature_columns": audit["feature_columns"],
            "required_feature_columns": audit["required_core_feature_columns"], "verbose": True}


def _seals() -> dict:
    root = Path(__file__).resolve().parents[1]
    names = ["economic_value/" + name for name in ("price_runner.py", "price_data.py", "price_policy.py", "price_reporting.py",
             "runner.py", "data.py", "engine.py", "extreme_runner.py", "reporting.py")]
    names += ["chronos2_hourly/process_lock.py", "run_price_expert.py", "PriceExpert.ps1"]
    return {name: base.digest(root / name) for name in names}


def audit_inputs(config: dict, *, root: Path):
    from .price_data import load_price_inputs
    validate_config(config)
    panel, audit, baseline_config = load_price_inputs(root, config)
    return panel, {**audit, **base._validate_panel(panel, baseline_config)}, baseline_config


def prepare(config: dict, *, root: Path) -> Path:
    validate_config(config)
    output = base._output(root, config["output_root"])
    with exclusive_process_lock(output / "prepare.lock"):
        before = _protected(root)
        panel, audit, baseline_config = audit_inputs(config, root=root)
        token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        snapshot = output / "snapshots" / token
        snapshot.mkdir(parents=True, exist_ok=False)
        panel.to_parquet(snapshot / "panel.parquet", index=False)
        for name, value in (("config", config), ("baseline_config", baseline_config), ("source_audit", audit)):
            base._json(snapshot / (name + ".json"), value)
        if _protected(root) != before:
            raise ValueError("Operational files changed concurrently during Prepare; retry with a stable workspace.")
        manifest = {"schema_version": 1, "kind": "nuclear_kalman_price_residual_expert",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "snapshot_files": {name: base.digest(snapshot / name) for name in sorted(INPUTS)},
                    "source_code_sha256": _seals(), "protected_files_sha256": before,
                    **{key: audit[key] for key in ("evaluation_start", "evaluation_end", "evaluation_days")},
                    "zones": config["zones"], "models": [config["baseline_model"], config["candidate_model"]],
                    "diagnostic_only": True, "production_modified": False, "activation_performed": False,
                    "orders_placed": False, "baseline_neural_oof_certified": False,
                    "forecast_and_reference_pit_certified": False}
        base._json(snapshot / "manifest.json", manifest)
        base._json(snapshot / "status.json", {"status": "prepared", "snapshot": str(snapshot)})
        base._json(output / "latest_prepared.json", {"snapshot": str(snapshot)})
        LOGGER.info("Price-expert inputs frozen: %s", snapshot)
        return snapshot


def read_snapshot(snapshot: Path, *, root: Path):
    snapshot = base._output(root, snapshot)
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "nuclear_kalman_price_residual_expert":
        raise ValueError("Not a price-expert snapshot.")
    _verify_files(snapshot, manifest["snapshot_files"], INPUTS)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    validate_config(config)
    baseline_config = json.loads((snapshot / "baseline_config.json").read_text(encoding="utf-8"))
    panel = pd.read_parquet(snapshot / "panel.parquet")
    coverage = base._validate_panel(panel, baseline_config)
    if any(manifest.get(k) != coverage[k] for k in ("evaluation_start", "evaluation_end", "evaluation_days")):
        raise ValueError("Price-expert evaluation calendar mismatch.")
    if manifest["zones"] != config["zones"] or manifest["models"] != [config["baseline_model"], config["candidate_model"]]:
        raise ValueError("Frozen model/country selection mismatch.")
    return snapshot, config, baseline_config, manifest, panel


def candidate_panel(panel: pd.DataFrame, decisions: pd.DataFrame, config: dict) -> pd.DataFrame:
    keys = ["timestamp_utc", "zone"]
    required = {*keys, "baseline_forecast", "candidate_forecast", "applied_correction", "raw_residual_prediction",
                "selected_weight", "expert_ready", "reason", "forecast_origin_utc", "expert_available_at_utc"}
    if required-set(decisions) or decisions.duplicated(keys).any() or len(decisions) != len(panel):
        raise ValueError("Price decisions must retain every baseline interval exactly once with explicit provenance.")
    if "policy_position_fraction" in panel or "policy_position_fraction" in decisions:
        raise ValueError("Position-only policy outputs cannot be used as price forecasts.")
    added = [c for c in decisions if c not in panel or c in keys]
    result = panel.merge(decisions[added], on=keys, how="left", validate="one_to_one", indicator=True)
    if not result.pop("_merge").eq("both").all():
        raise ValueError("Price-expert timestamps differ from the frozen baseline.")
    indexed = decisions.set_index(keys).reindex(pd.MultiIndex.from_frame(panel[keys]))
    if not np.array_equal(indexed.baseline_forecast, panel.forecast, equal_nan=True):
        raise ValueError("The expert changed its frozen baseline input.")
    if not pd.DatetimeIndex(indexed.forecast_origin_utc).equals(pd.DatetimeIndex(panel.forecast_origin_utc)):
        raise ValueError("Price-expert forecast origin differs from the baseline.")
    for column in ("candidate_forecast", "applied_correction", "selected_weight"):
        if not np.isfinite(result[column]).all():
            raise ValueError(f"Nonfinite price-expert {column}; use the baseline on unavailable hours.")
    if not np.allclose(result.candidate_forecast, result.forecast+result.applied_correction, rtol=0, atol=1e-9):
        raise ValueError("Price forecast must equal baseline plus the applied residual correction.")
    weights = np.asarray(config["expert"]["candidate_weights"], dtype=float)
    if not np.isclose(result.selected_weight.to_numpy()[:, None], weights[None, :], rtol=0, atol=1e-12).any(axis=1).all():
        raise ValueError("Selected correction weights are outside the frozen candidate bank.")
    if not result.expert_ready.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("Expert readiness must contain explicit booleans.")
    raw = result.raw_residual_prediction.to_numpy(dtype=float)
    if (result.expert_ready & ~np.isfinite(raw)).any() or (~result.expert_ready & result.selected_weight.ne(0)).any():
        raise ValueError("Unavailable expert must have zero weight; ready experts require a finite residual.")
    expected = result.selected_weight.to_numpy()*np.where(
        result.expert_ready & (np.abs(raw) >= float(config["expert"]["minimum_residual_eur_mwh"])),
        np.clip(raw, -float(config["expert"]["correction_clip_eur_mwh"]), float(config["expert"]["correction_clip_eur_mwh"])), 0.)
    if not np.allclose(result.applied_correction, expected, rtol=0, atol=1e-9):
        raise ValueError("Applied residual correction must match its fixed trigger, clip and selected weight.")
    if result.applied_correction.abs().gt(float(config["expert"]["correction_clip_eur_mwh"])*weights.max()+1e-9).any():
        raise ValueError("Applied price correction exceeds the frozen risk cap.")
    changed = result.applied_correction.abs().gt(1e-10)
    if any(pd.Timestamp(value).tzinfo is None for value in result.expert_available_at_utc.dropna()):
        raise ValueError("Expert availability requires explicit UTC-aware timestamps.")
    available = pd.to_datetime(result.expert_available_at_utc, utc=True, errors="raise")
    if (changed & (~result.expert_ready | available.isna() | (available > result.forecast_origin_utc))).any():
        raise ValueError("Changed price forecast has no expert available at the 08:00 origin.")
    result["forecast"] = result.candidate_forecast
    result["model"] = config["candidate_model"]
    # A changed point estimate does not confer calibrated interval forecasts.
    result.loc[changed, ["q10", "q90"]] = np.nan
    result["quantile_policy"] = np.where(changed, "unavailable_after_point_correction", "unchanged_baseline")
    return result


def _scores(error: pd.Series) -> dict:
    return {"hours": int(len(error)), "mae_eur_mwh": float(error.abs().mean()) if len(error) else None,
            "rmse_eur_mwh": float(np.sqrt(np.square(error).mean())) if len(error) else None,
            "bias_eur_mwh": float(error.mean()) if len(error) else None}


def forecast_statistics(panel: pd.DataFrame, candidate: pd.DataFrame, config: dict, settings: dict):
    frame = panel[["timestamp_utc", "zone", "actual", "benchmark_forecast", "forecast_eligible", "sample"]].copy()
    frame[config["baseline_model"]] = panel.forecast.to_numpy()
    frame[config["candidate_model"]] = candidate.forecast.to_numpy()
    frame["storm"] = frame.benchmark_forecast
    frame["expert_ready"] = candidate.expert_ready.to_numpy()
    frame["delivery_day"] = frame.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")
    models = [config["baseline_model"], config["candidate_model"], "storm"]
    mask = np.isfinite(frame[["actual", *models]]).all(axis=1) & frame.forecast_eligible & frame["sample"].eq("evaluation")
    if "benchmark_eligible" in panel:
        mask &= panel.benchmark_eligible
    # No yesterday-reference requirement for price accuracy. EVA has its own
    # stricter pairing mask; do not erase these hours from price statistics.
    metrics, daily = [], []
    for zone in [*config["zones"], "PORTFOLIO"]:
        block = frame.loc[mask & (frame.zone.eq(zone) if zone != "PORTFOLIO" else True)]
        subsets = {"all": np.ones(len(block), dtype=bool), "high": block.actual.ge(settings["spike_high_eur_mwh"]),
                   "negative": block.actual.le(settings["spike_low_eur_mwh"]),
                   "normal": block.actual.between(settings["spike_low_eur_mwh"], settings["spike_high_eur_mwh"], inclusive="neither"),
                   "expert_ready": block.expert_ready}
        for model in models:
            for subset, selected in subsets.items():
                value = block.loc[selected]
                metrics.append({"zone": zone, "model": model, "subset": subset, **_scores(value[model]-value.actual)})
            for day, value in block.groupby("delivery_day", sort=True):
                daily.append({"zone": zone, "model": model, "delivery_day": day, **_scores(value[model]-value.actual)})
    return pd.DataFrame(metrics), pd.DataFrame(daily)


def _check_economic_baseline(result, audit: dict, config: dict) -> None:
    source = audit.get("source_baseline_metrics")
    if source:
        original = pd.DataFrame(source).set_index(["zone", "strategy"])
        unchanged = result.metrics.loc[result.metrics.model.eq(config["baseline_model"])].set_index(["zone", "strategy"])
        original = original.reindex(unchanged.index)
        for name in ("pnl_net_eur", "pnl_gross_eur", "absolute_energy_mwh", "eligible_hours", "active_hours"):
            if not np.allclose(original[name], unchanged[name], rtol=1e-10, atol=1e-6, equal_nan=True):
                raise ValueError(f"Frozen baseline economic metric changed: {name}.")
    for flag in ("paired_eligible", "portfolio_eligible"):
        masks = result.rows.loc[result.rows.strategy.eq("model")].pivot(index=["timestamp_utc", "zone"], columns="model", values=flag)
        if not masks[config["baseline_model"]].equals(masks[config["candidate_model"]]):
            raise ValueError("Price expert and baseline must use identical economic pairing masks.")


def _summary(result, price_metrics, candidate, config):
    output = []
    for zone in [*config["zones"], "PORTFOLIO"]:
        errors = price_metrics.loc[price_metrics.zone.eq(zone) & price_metrics.subset.eq("all")].set_index("model")
        economic = result.metrics.loc[result.metrics.zone.eq(zone) & result.metrics.strategy.eq("model")].set_index("model")
        rows = candidate.loc[candidate.zone.eq(zone)] if zone != "PORTFOLIO" else candidate
        mae_gain = float(errors.loc[config["baseline_model"], "mae_eur_mwh"]-errors.loc[config["candidate_model"], "mae_eur_mwh"])
        pnl_gain = float(economic.loc[config["candidate_model"], "pnl_net_eur"]-economic.loc[config["baseline_model"], "pnl_net_eur"])
        output.append({"zone": zone, "changed_hours": int(rows.applied_correction.abs().gt(1e-10).sum()),
                       "mean_weight": float(rows.selected_weight.mean()), "expert_ready_hours": int(rows.expert_ready.sum()),
                       "annual_mae_gain_eur_mwh": mae_gain, "net_gain_vs_baseline_eur": pnl_gain,
                       "annual_non_regression": mae_gain >= -1e-10, "annual_eva_gain_pass": pnl_gain > 0})
    return output


def evaluate(snapshot: Path, *, root: Path) -> Path:
    from .price_policy import run_price_policy
    snapshot, config, baseline_config, manifest, panel = read_snapshot(snapshot, root=root)
    with exclusive_process_lock(snapshot / "evaluation.lock"):
        if (snapshot / "results_manifest.json").exists():
            path = report(snapshot, root=root)
            _publish(snapshot, config, path, root)
            return path
        if manifest["source_code_sha256"] != _seals():
            raise ValueError("Code changed since Prepare; create a new snapshot.")
        before = _protected(root)
        base._json(snapshot / "status.json", {"status": "running", "stage": "price_residual_expert", "snapshot": str(snapshot)})
        try:
            source_audit = json.loads((snapshot / "source_audit.json").read_text(encoding="utf-8"))
            trained = run_price_policy(panel, model_config(config, baseline_config, source_audit))
            candidate = candidate_panel(panel, trained.decisions, config)
            combined = pd.concat([panel, candidate], ignore_index=True)
            settings = base.engine_config(baseline_config)
            settings.update(evaluation_start_day=manifest["evaluation_start"], evaluation_end_day=manifest["evaluation_end"])
            result = simulate(combined, settings)  # Recompute fixed decisions from NEW prices, no supplied positions.
            _check_economic_baseline(result, source_audit, config)
            price_metrics, price_daily = forecast_statistics(panel, candidate, config, settings)
            flat = pd.concat([v.assign(group=v["group"].astype(str), breakdown=k) for k, v in result.breakdowns.items()], ignore_index=True)
            for name, frame in (("decisions", trained.decisions), ("folds", trained.folds), ("governance", trained.governance),
                                ("rows", result.rows), ("metrics", result.metrics), ("daily", result.daily), ("breakdowns", flat),
                                ("forecast_metrics", price_metrics), ("forecast_daily", price_daily)):
                temporary = snapshot / f".tmp_{uuid.uuid4().hex}.parquet"
                frame.to_parquet(temporary, index=False)
                temporary.replace(snapshot / f"{name}.parquet")
            base._json(snapshot / "model_audit.json", trained.audit)
            if _protected(root) != before:
                raise ValueError("Operational files changed concurrently; review before publishing this experiment.")
            details = {"baseline_model": config["baseline_model"], "candidate_model": config["candidate_model"],
                       "baseline_comparison_paired": True, "training_window_days": config["training_window_days"],
                       "minimum_training_days": config["minimum_training_days"], "calibration_policy": "progressive_history_capped_at_365_days",
                       "refit_every_days": config["expert"]["refit_every_days"], "governance_minimum_days": config["governance"]["governance_minimum_days"],
                       "summary": _summary(result, price_metrics, candidate, config), "model_audit": trained.audit,
                       "price_metric_mask": "same actual, baseline, corrected and Storm hours; reference is not required",
                       "portfolio_price_metric": "pooled hourly-country errors, not error of an averaged zonal price",
                       "quantiles": "baseline intervals unchanged only during fallback; missing on corrected hours, not recalibrated",
                       "limitations": ["Reference veille non executable : aucune preuve de profit negociable.",
                                       "Seulement 365 jours de baseline : calibration progressive, pas 365 jours complets avant chaque prediction.",
                                       "Annee deja examinee avant conception : diagnostic repete, pas un test final intact.",
                                       "PIT historique et OOF neuronal de la baseline non certifies ; correction interne chronologique seulement.",
                                       "Gouvernance sans Storm : prix corrige puis meme regle economique, meme cout, meme capacite.",
                                       "Non-regression annuelle mesuree apres coup, jamais garantie ni utilisee pour reecrire le backtest."]}
            audit = {**manifest, "status": "completed_hypothetical_diagnostic", "engine": result.audit, "price_expert": details,
                     "portfolio_capacity_mw": settings["portfolio_capacity_mw"], "zone_capacity_mw": settings["zone_capacity_mw"],
                     "rule": baseline_config["strategy"], "reference_kind": "lagged_day_ahead_proxy", "data": source_audit,
                     "forecast_values_unchanged": False, "forecast_changes_restricted_to_candidate": True,
                     "fixed_economic_decision_rule": True, "forecast_expert_parameters_fitted_on_past_evaluation_prefix": True,
                     "operational_files_unchanged_during_evaluation": True,
                     "runtime_versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "threadpoolctl", "pyarrow")},
                     "result_files": {name: base.digest(snapshot / name) for name in sorted(RESULTS)}}
            base._json(snapshot / "results_manifest.json", audit)
            path = report(snapshot, root=root)
            _publish(snapshot, config, path, root)
            return path
        except BaseException as exc:
            base._json(snapshot / "status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                       "snapshot": str(snapshot), "error": f"{type(exc).__name__}: {exc}"})
            raise


def report(snapshot: Path, *, root: Path) -> Path:
    from .price_reporting import render_price_report
    snapshot, _, _, manifest, _ = read_snapshot(snapshot, root=root)
    audit = json.loads((snapshot / "results_manifest.json").read_text(encoding="utf-8"))
    for key in ("snapshot_files", "source_code_sha256", "evaluation_start", "evaluation_end", "zones", "models"):
        if manifest[key] != audit.get(key):
            raise ValueError("Price-expert result/input manifest mismatch.")
    _verify_files(snapshot, audit["result_files"], RESULTS)
    with exclusive_process_lock(snapshot / "report.lock"):
        flat = pd.read_parquet(snapshot / "breakdowns.parquet")
        path = snapshot / "nuclear_kalman_extreme_report.html"
        temporary = snapshot / f".tmp_report_{uuid.uuid4().hex}.html"
        render_price_report(pd.read_parquet(snapshot / "rows.parquet"), pd.read_parquet(snapshot / "metrics.parquet"),
                            pd.read_parquet(snapshot / "daily.parquet"), {k: v.drop(columns="breakdown") for k, v in flat.groupby("breakdown")},
                            audit, temporary, **{name: pd.read_parquet(snapshot / f"{name}.parquet")
                                               for name in ("forecast_metrics", "forecast_daily", "decisions", "governance")})
        temporary.replace(path)
        return path


def _publish(snapshot: Path, config: dict, path: Path, root: Path):
    base._json(snapshot / "status.json", {"status": "completed", "snapshot": str(snapshot), "report": str(path)})
    base._json(base._output(root, config["output_root"]) / "latest.json", {"snapshot": str(snapshot), "report": str(path)})


def resolve_snapshot(root: Path, config: dict, path: Path | None, *, prepared=False):
    return base.resolve_snapshot(root, config, path, prepared=prepared)
