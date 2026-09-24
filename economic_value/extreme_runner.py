"""Isolated chronological economic-policy experiment over a frozen EVA baseline."""
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
from .reporting import render_report

LOGGER = logging.getLogger(__name__)
INPUTS = {"config.json", "baseline_config.json", "history.parquet", "panel.parquet", "feature_audit.json"}
RESULTS = {"decisions.parquet", "folds.parquet", "governance.parquet", "rows.parquet", "metrics.parquet", "daily.parquet", "breakdowns.parquet", "policy_audit.json"}
PROTECTED = ["Forecast.ps1", "run_complete_forecast.py", "run_multicountry_forecast.py", "run_mkonline_live_hourly.py",
             "chronos2_hourly_live_zones.yaml", "config/nuclear_forecast.yaml", "config/economic_value.yaml"]


def validate_config(config: dict) -> None:
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Expected extreme-policy schema_version 1.")
    if config.get("training_days") != 365:
        raise ValueError("The expert requires an explicit rolling 365-day training window.")
    if config.get("baseline_model") != "nuclear_kalman" or config.get("candidate_model") != "nuclear_kalman_extreme_governed":
        raise ValueError("This experiment compares nuclear_kalman and nuclear_kalman_extreme_governed.")
    zones = config.get("zones")
    if not isinstance(zones, list) or not zones or len(set(zones)) != len(zones) or set(zones)-base.ZONES:
        raise ValueError("Choose unique FR/DE/BE/NL zones.")
    if config.get("diagnostic_only") is not True or config.get("production_modified") is not False or config.get("order_execution_enabled") is not False:
        raise ValueError("Diagnostic only: operational changes and order execution are forbidden.")
    for key in ("source_snapshot", "output_root"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"Missing {key}.")
    if not isinstance(config.get("expert"), dict) or not isinstance(config.get("governance"), dict):
        raise ValueError("Explicit expert/governance settings are required.")


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    validate_config(config)
    return config


def _seals() -> dict:
    names = ["data.py", "engine.py", "runner.py", "extreme_data.py", "extreme_policy.py", "extreme_runner.py"]
    seals = {name: base.digest(Path(__file__).with_name(name)) for name in names}
    root = Path(__file__).resolve().parents[1]
    for name in ("marginal_cost_expert/evaluation.py", "chronos2_hourly/process_lock.py"):
        seals[name] = base.digest(root / name)
    return seals


def _protected(root: Path) -> dict:
    return {name: base.digest(root / name) for name in PROTECTED if (root / name).is_file()}


def _verify_files(snapshot: Path, hashes: dict, expected: set) -> None:
    if set(hashes) != expected:
        raise ValueError("Incomplete extreme-policy snapshot manifest.")
    for name, digest in hashes.items():
        if Path(name).name != name or base.digest(snapshot / name) != digest:
            raise ValueError(f"Extreme-policy checksum mismatch: {name}.")


def audit_inputs(config: dict, *, root: Path):
    from .extreme_data import load_extreme_inputs
    validate_config(config)
    source, source_config, source_manifest, _ = base.read_snapshot(base._path(root, config["source_snapshot"]), root=root)
    source_results = json.loads((source / "results_manifest.json").read_text(encoding="utf-8"))
    if set(config["zones"]) != set(source_config["zones"]):
        raise ValueError("Keep the source snapshot countries: changing them would change the frozen MW allocation. Prepare a separate baseline EVA first.")
    base._verify(source, source_results, results=True)
    for key in ("snapshot_files", "source_code_sha256", "evaluation_start", "evaluation_end", "models", "zones"):
        if source_manifest[key] != source_results.get(key):
            raise ValueError("Source EVA results do not match the frozen baseline inputs.")
    history, panel, audit = load_extreme_inputs(root, config)
    baseline_config = deepcopy(source_config)
    baseline_config.update(zones=config["zones"], models=[config["baseline_model"]])
    coverage = base._validate_panel(panel, baseline_config)
    source_metrics = pd.read_parquet(source / "metrics.parquet")
    source_metrics = source_metrics.loc[source_metrics.model.eq(config["baseline_model"])]
    return history, panel, {**audit, **coverage, "source_snapshot": str(source),
                           "source_baseline_metrics": json.loads(source_metrics.to_json(orient="records")),
                           "source_manifest_sha256": base.digest(source / "manifest.json"),
                           "source_result_manifest_sha256": base.digest(source / "results_manifest.json")}, baseline_config


def prepare(config: dict, *, root: Path) -> Path:
    output = base._output(root, config["output_root"])
    with exclusive_process_lock(output / "prepare.lock"):
        before = _protected(root)
        history, panel, audit, baseline_config = audit_inputs(config, root=root)
        token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        snapshot = output / "snapshots" / token
        snapshot.mkdir(parents=True, exist_ok=False)
        history.to_parquet(snapshot / "history.parquet", index=False)
        panel.to_parquet(snapshot / "panel.parquet", index=False)
        for name, value in (("config", config), ("baseline_config", baseline_config), ("feature_audit", audit)):
            base._json(snapshot / (name + ".json"), value)
        if _protected(root) != before:
            raise ValueError("Operational files changed concurrently during Prepare; retry with a stable workspace.")
        manifest = {"schema_version": 1, "kind": "extreme_economic_policy", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "snapshot_files": {name: base.digest(snapshot / name) for name in sorted(INPUTS)},
                    "source_code_sha256": _seals(), "protected_files_sha256": before,
                    **{name: audit[name] for name in ("evaluation_start", "evaluation_end", "evaluation_days")},
                    "zones": config["zones"], "models": [config["baseline_model"], config["candidate_model"]],
                    "diagnostic_only": True, "forecast_and_reference_pit_certified": False,
                    "baseline_neural_oof_certified": False, "production_modified": False,
                    "activation_performed": False, "orders_placed": False}
        base._json(snapshot / "manifest.json", manifest)
        base._json(snapshot / "status.json", {"status": "prepared", "snapshot": str(snapshot)})
        base._json(output / "latest_prepared.json", {"snapshot": str(snapshot)})
        LOGGER.info("Expert inputs frozen: %s", snapshot)
        return snapshot


def read_snapshot(snapshot: Path, *, root: Path):
    snapshot = base._output(root, snapshot)
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "extreme_economic_policy":
        raise ValueError("Not an extreme-policy snapshot.")
    _verify_files(snapshot, manifest["snapshot_files"], INPUTS)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    validate_config(config)
    baseline_config = json.loads((snapshot / "baseline_config.json").read_text(encoding="utf-8"))
    panel = pd.read_parquet(snapshot / "panel.parquet")
    coverage = base._validate_panel(panel, baseline_config)
    if any(coverage[k] != manifest[k] for k in ("evaluation_start", "evaluation_end", "evaluation_days")):
        raise ValueError("Frozen evaluation calendar mismatch.")
    if manifest["zones"] != config["zones"] or manifest["models"] != [config["baseline_model"], config["candidate_model"]]:
        raise ValueError("Frozen selection mismatch.")
    return snapshot, config, baseline_config, manifest, panel


def policy_config(config: dict, baseline_config: dict) -> dict:
    protected = {"signal_threshold_eur_mwh", "transaction_cost_eur_mwh", "slippage_eur_mwh", "timezone", "training_days"}
    if protected & (set(config["expert"]) | set(config["governance"])):
        raise ValueError("Training window and comparison costs cannot be overridden in expert/governance.")
    return {**config["expert"], **config["governance"],
            **baseline_config["strategy"], "timezone": "Europe/Paris", "training_days": 365, "verbose": True}


def _summary(result, decisions: pd.DataFrame, config: dict) -> list[dict]:
    summaries = []
    for zone in [*config["zones"], "PORTFOLIO"]:
        rows = result.rows.loc[result.rows.strategy.eq("model") & result.rows.model.eq(config["candidate_model"]) & result.rows.paired_eligible]
        rows = rows.loc[rows.portfolio_eligible] if zone == "PORTFOLIO" else rows.loc[rows.zone.eq(zone)]
        metrics = result.metrics.loc[result.metrics.zone.eq(zone) & result.metrics.strategy.eq("model")].set_index("model")
        spike = result.breakdowns["spike"]
        spike = spike.loc[spike.zone.eq(zone) & spike.strategy.eq("model") & spike.group.eq("high")].set_index("model")
        summaries.append({"zone": zone, "intervention_hours": int((rows.policy_position_fraction-rows.baseline_position_fraction).abs().gt(1e-10).sum()),
                          "mean_governance_weight": float(rows.governance_weight.mean()),
                          "net_gain_vs_baseline_eur": float(metrics.loc[config["candidate_model"], "pnl_net_eur"]-metrics.loc[config["baseline_model"], "pnl_net_eur"]),
                          "spike_net_gain_vs_baseline_eur": float(spike.loc[config["candidate_model"], "pnl_net_eur"]-spike.loc[config["baseline_model"], "pnl_net_eur"]) if len(spike) == 2 else None})
    return summaries


def evaluate(snapshot: Path, *, root: Path) -> Path:
    from .extreme_policy import run_extreme_policy
    snapshot, config, baseline_config, manifest, panel = read_snapshot(snapshot, root=root)
    with exclusive_process_lock(snapshot / "evaluation.lock"):
        if (snapshot / "results_manifest.json").exists():
            path = report(snapshot, root=root)
            _publish(snapshot, config, path, root)
            return path
        if manifest["source_code_sha256"] != _seals():
            raise ValueError("Code changed since Prepare; create a new snapshot.")
        before = _protected(root)
        base._json(snapshot / "status.json", {"status": "running", "stage": "rolling_expert", "snapshot": str(snapshot)})
        try:
            features = json.loads((snapshot / "feature_audit.json").read_text(encoding="utf-8"))
            training_config = policy_config(config, baseline_config)
            training_config.update(feature_columns=features["feature_columns"], required_feature_columns=features["required_core_feature_columns"])
            trained = run_extreme_policy(pd.read_parquet(snapshot / "history.parquet"), panel, training_config)
            decisions = trained.decisions
            keys = ["timestamp_utc", "zone"]
            if decisions.duplicated(keys).any() or len(decisions) != len(panel):
                raise ValueError("Policy must retain every baseline interval exactly once.")
            cols = [name for name in decisions if name not in panel or name in keys]
            candidate = panel.merge(decisions[cols], on=keys, how="left", validate="one_to_one", indicator=True)
            if not candidate["_merge"].eq("both").all():
                raise ValueError("Policy decisions and baseline timestamps disagree.")
            candidate = candidate.drop(columns="_merge").assign(model=config["candidate_model"])
            combined = pd.concat([panel, candidate], ignore_index=True)
            settings = base.engine_config(baseline_config)
            settings.update(governed_models=[config["candidate_model"]], evaluation_start_day=manifest["evaluation_start"], evaluation_end_day=manifest["evaluation_end"])
            result = simulate(combined, settings)
            if features.get("source_baseline_metrics"):
                original = pd.DataFrame(features["source_baseline_metrics"]).set_index(["zone", "strategy"])
                unchanged = result.metrics.loc[result.metrics.model.eq(config["baseline_model"])].set_index(["zone", "strategy"])
                original = original.reindex(unchanged.index)
                for name in ("pnl_net_eur", "pnl_gross_eur", "absolute_energy_mwh", "eligible_hours", "active_hours"):
                    if not np.allclose(original[name].to_numpy(float), unchanged[name].to_numpy(float), rtol=1e-10, atol=1e-6, equal_nan=True):
                        raise ValueError(f"Frozen baseline metric changed: {name}. No comparison will be published.")
            for name in ("paired_eligible", "portfolio_eligible"):
                masks = result.rows.loc[result.rows.strategy.eq("model")].pivot(index=keys, columns="model", values=name)
                if not masks[config["baseline_model"]].equals(masks[config["candidate_model"]]):
                    raise ValueError("Baseline and expert must be scored on identical masks.")
            if not candidate.forecast.equals(panel.forecast) or not candidate.q10.equals(panel.q10) or not candidate.q90.equals(panel.q90):
                raise ValueError("The economic policy may not change the operational forecast or quantiles.")
            flat = pd.concat([value.assign(group=value["group"].astype(str), breakdown=name) for name, value in result.breakdowns.items()], ignore_index=True)
            for name, frame in (("decisions", decisions), ("folds", trained.folds), ("governance", trained.governance), ("rows", result.rows), ("metrics", result.metrics), ("daily", result.daily), ("breakdowns", flat)):
                temporary = snapshot / f".tmp_{uuid.uuid4().hex}.parquet"
                frame.to_parquet(temporary, index=False)
                temporary.replace(snapshot / f"{name}.parquet")
            base._json(snapshot / "policy_audit.json", trained.audit)
            if _protected(root) != before:
                raise ValueError("Operational files changed concurrently; review before publishing this experiment.")
            details = {"enabled": True, "baseline_model": config["baseline_model"], "candidate_model": config["candidate_model"],
                       "training_days": 365, "refit_every_days": config["expert"]["refit_every_days"],
                       "governance_minimum_days": config["governance"]["governance_minimum_days"],
                       "governance_lookback_days": config["governance"]["governance_lookback_days"], "baseline_comparison_paired": True,
                       "summary": _summary(result, decisions, config), "model_audit": trained.audit,
                       "limitations": ["Reference veille non executable; aucun profit de trading demontre.",
                                       "Annee examinee avant conception: diagnostic retrospectif, pas un test final intact.",
                                       "PIT historique et OOF neuronal de la baseline non certifies.",
                                       "Gouvernance sans Storm en entree; historique OOF interne uniquement, retour baseline au demarrage.",
                                       "Couts et capacite identiques; regle de decision differente: gain de strategie, pas nouvelle precision du forecast."]}
            audit = {**manifest, "status": "completed_hypothetical_diagnostic", "engine": result.audit, "extreme_policy": details,
                     "portfolio_capacity_mw": settings["portfolio_capacity_mw"], "zone_capacity_mw": settings["zone_capacity_mw"],
                     "rule": baseline_config["strategy"], "reference_kind": "lagged_day_ahead_proxy",
                     "data": json.loads((snapshot / "feature_audit.json").read_text(encoding="utf-8")),
                     "runtime_versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "threadpoolctl", "pyarrow")},
                     "forecast_values_unchanged": True, "operational_files_unchanged_during_evaluation": True,
                     "result_files": {name: base.digest(snapshot / name) for name in sorted(RESULTS)}}
            base._json(snapshot / "results_manifest.json", audit)
            path = report(snapshot, root=root)
            _publish(snapshot, config, path, root)
            return path
        except Exception as exc:
            base._json(snapshot / "status.json", {"status": "failed", "snapshot": str(snapshot), "error": f"{type(exc).__name__}: {exc}"})
            raise


def report(snapshot: Path, *, root: Path) -> Path:
    snapshot, _, _, manifest, _ = read_snapshot(snapshot, root=root)
    audit = json.loads((snapshot / "results_manifest.json").read_text(encoding="utf-8"))
    for key in ("snapshot_files", "source_code_sha256", "evaluation_start", "evaluation_end", "zones", "models"):
        if manifest[key] != audit.get(key):
            raise ValueError("Expert result/input manifest mismatch.")
    _verify_files(snapshot, audit["result_files"], RESULTS)
    with exclusive_process_lock(snapshot / "report.lock"):
        flat = pd.read_parquet(snapshot / "breakdowns.parquet")
        temporary = snapshot / f".tmp_report_{uuid.uuid4().hex}.html"
        path = snapshot / "economic_extreme_policy_report.html"
        render_report(pd.read_parquet(snapshot / "rows.parquet"), pd.read_parquet(snapshot / "metrics.parquet"),
                      pd.read_parquet(snapshot / "daily.parquet"), {name: value.drop(columns="breakdown") for name, value in flat.groupby("breakdown")}, audit, temporary)
        temporary.replace(path)
        return path


def _publish(snapshot: Path, config: dict, path: Path, root: Path):
    base._json(snapshot / "status.json", {"status": "completed", "snapshot": str(snapshot), "report": str(path)})
    base._json(base._output(root, config["output_root"]) / "latest.json", {"snapshot": str(snapshot), "report": str(path)})


def resolve_snapshot(root: Path, config: dict, path: Path | None, *, prepared=False):
    return base.resolve_snapshot(root, config, path, prepared=prepared)
