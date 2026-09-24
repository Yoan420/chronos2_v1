"""Sealed research runner, restricted to its namespace; Report never fits models."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
import json
import logging
from pathlib import Path
import uuid

import joblib
import numpy as np
import pandas as pd
import yaml

from chronos2_hourly.process_lock import exclusive_process_lock
from economic_value.data import load_report_panel
from nyx_scarcity.runner import digest, _json, _parquet, protected_state
from .features import build_features
from .policy import run_policy

LOGGER = logging.getLogger(__name__)
NAMESPACE = Path("runs/experiments/nyx_solar_ramp_v1")
INPUT_FILES = {"config.json", "feature_groups.json", "source_audit.json", "panel.parquet", "literature.json"}
TRAINING_FILES = {"predictions.parquet", "folds.json", "governance.json", "models.joblib"}
RESULT_FILES = TRAINING_FILES | {"predictions.csv", "metrics.json", "training_manifest.json"}


def safe(root, path):
    root = root.resolve()
    raw = (root / path).absolute()
    resolved = raw.resolve()
    namespace = root/NAMESPACE
    if namespace.resolve() != namespace or raw != resolved or not resolved.is_relative_to(namespace):
        raise ValueError("Solar outputs must stay in runs/experiments/nyx_solar_ramp_v1 without aliases.")
    return resolved


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    allowed = {"schema_version", "source_snapshot", "baseline_delivery_day", "output_root", "enabled", "mode",
        "primary_variant", "zones", "timezone", "cutoff_time", "training_window_days", "minimum_training_days",
        "calibration_days", "refit_days", "final_days", "selection_days", "business_spike", "statistical_quantile",
        "price_ramp_threshold", "false_alert_budget", "label_delay_days", "max_iter", "max_leaf_nodes", "min_samples_leaf",
        "l2_regularization", "learning_rate", "correction_clip", "governance_days", "governance_min_days",
        "governance_min_changed_days", "governance_min_alert_rows", "candidate_weights", "mae_tolerance",
        "bootstrap_samples", "block_days", "seed", "threads"}
    if not isinstance(config, dict) or set(config) != allowed:
        raise ValueError("Missing or unknown solar configuration fields.")
    if (config.get("schema_version") != 1 or config.get("mode") != "diagnostic_shadow"
            or config.get("timezone") != "Europe/Paris" or config.get("cutoff_time") != "08:00"
            or config.get("primary_variant") != "governed" or config.get("mae_tolerance") != 0
            or config.get("training_window_days") != 365 or type(config.get("enabled")) is not bool):
        raise ValueError("Strict isolated shadow / D-1 08h / rolling365 / zero-MAE-tolerance contract required.")
    if config.get("zones") != ["FR", "DE", "BE", "NL"]:
        raise ValueError("Keep the fixed four-country scope for paired ablations.")
    for name in ("minimum_training_days", "calibration_days", "refit_days", "max_iter", "max_leaf_nodes", "min_samples_leaf",
                 "governance_days", "governance_min_days", "governance_min_changed_days", "governance_min_alert_rows",
                 "bootstrap_samples", "block_days", "threads", "final_days", "selection_days", "label_delay_days"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"{name}: positive integer required.")
    if not 90 <= config["minimum_training_days"] <= 365 or config["calibration_days"] >= config["minimum_training_days"]:
        raise ValueError("Chronological training must precede calibration.")
    if config["threads"] > 4 or config["max_iter"] > 500 or config["bootstrap_samples"] > 10000:
        raise ValueError("Bounded experimental compute required.")
    if not 0 < config["false_alert_budget"] < .1 or not .9 <= config["statistical_quantile"] < 1:
        raise ValueError("Invalid predeclared alert/statistical thresholds.")
    if (not isinstance(config["candidate_weights"], list) or not config["candidate_weights"]
            or config["candidate_weights"][0] != 0
            or any(type(w) not in (int, float) or w < 0 or w > 1 for w in config["candidate_weights"])):
        raise ValueError("Governance must include identity and bounded weights.")
    for name in ("business_spike", "price_ramp_threshold", "false_alert_budget", "statistical_quantile",
                 "l2_regularization", "learning_rate", "correction_clip"):
        if type(config[name]) not in (int, float) or not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name}: finite positive value required.")
    if (type(config["seed"]) is not int or config["seed"] < 0
            or len(set(config["candidate_weights"])) != len(config["candidate_weights"])
            or not all(np.isfinite(config["candidate_weights"]))
            or config["final_days"] + config["selection_days"] + config["minimum_training_days"] > 365):
        raise ValueError("Invalid fixed chronological protocol or seed/weight grid.")
    return config


def _seals(root):
    paths = sorted((root/"nyx_solar_ramp").glob("*.py")) + [root/"run_nyx_solar_ramp.py", root/"SolarRamp.ps1",
            root/"config/nyx_solar_ramp_literature.json", root/"economic_value/data.py",
            root/"nyx_scarcity/runner.py", root/"nyx_scarcity_zonal/reporting.py", root/"chronos2_modular/report.py"]
    return {p.relative_to(root).as_posix(): digest(p) for p in paths if p.is_file()}


def prepare(config, *, root):
    output = safe(root, config["output_root"])
    source = (root/config["source_snapshot"]).resolve()
    if not source.is_relative_to(root/"runs/experiments"):
        raise ValueError("Use a trusted local research source snapshot.")
    manifest = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    for name in ("config.json", "data_audit.json", "panel.parquet"):
        if digest(source/name) != manifest["input_files"][name]:
            raise ValueError(f"Frozen source checksum mismatch: {name}")
    state = protected_state(root)
    data_audit = json.loads((source/"data_audit.json").read_text(encoding="utf-8"))
    historical = pd.read_parquet(source/"panel.parquet")
    baseline_day = config["baseline_delivery_day"]
    end_day = (pd.Timestamp(baseline_day)-pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    LOGGER.info("[Solar] Freeze baseline %s and as-of fundamentals; no forecast refresh.", baseline_day)
    baseline, baseline_audit = load_report_panel(root, config["zones"], ["nuclear_kalman"], delivery_day=baseline_day, end_day=end_day)
    feature_names = [name for name in historical if name.startswith("feature_") or name in ("label_available_at_utc", "label_eligible", "forecast_eligible")]
    features = historical[["zone", "timestamp_utc", *feature_names]]
    panel = baseline.merge(features, on=["zone", "timestamp_utc"], how="left", validate="one_to_one")
    panel["feature_baseline_forecast"] = panel.forecast
    panel["feature_baseline_interval_width"] = panel.q90-panel.q10
    panel["feature_baseline_upper_distance"] = panel.q90-panel.forecast
    panel["label_availability_assumed"] = True
    if panel[["forecast", "q10", "q90"]].isna().any().any():
        raise ValueError("Baseline archive has missing forecast/quantile hours; no reconstruction permitted.")
    augmented, groups, feature_audit = build_features(panel, include_baseline=True)
    source_audit = {"source_data_audit": {**data_audit, "baseline": baseline_audit,
                    "delivery_day": baseline_day, "evaluation_start_day": baseline_audit["evaluation_start_day"],
                    "evaluation_end_day": baseline_audit["evaluation_end_day"]},
                    "original_feature_snapshot": str(source), "source_manifest_sha256": digest(source/"manifest.json"),
                    "frozen_feature_panel_sha256": manifest["input_files"]["panel.parquet"],
                    "feature_audit": feature_audit, "feature_last_day": str(historical.timestamp_utc.max().tz_convert("Europe/Paris").date()),
                    "production_pit_evidence": False, "historical_price_publication_verified": False,
                    "labels": "Original J-1 18h assumption not certified; modelling further delays labels until delivery civil end + two days (configurable), still only a diagnostic assumption.",
                    "region": "FR, DE, BE, NL as the four report bidding zones; not asserted to equal all historical geographic definitions of CWE, CORE or STORM backend aggregates.",
                    "native_resolution": "1 physical hour; original EPEX quarter-hours may have been averaged upstream; extremes here are hourly averages, not quarter-hour maxima.",
                    "realised_solar_used": False, "post_coupling_features_used": False,
                    "jao": "Not included: 08h availability/publication qualification incomplete.",
                    "prospective": {"status": "blocked", "reasons": ["No certified contemporaneous input receipt for the next unknown auction.",
                        "Frozen fundamentals end before the next operational origin.", "Fresh EPEX and canonical publication checks not both available; absence is not assumed.",
                        "The report delivery day is historical, not prospective."]}}
    with exclusive_process_lock(output/"prepare.lock"):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]
        directory = safe(root, output/"snapshots"/stamp)
        directory.mkdir(parents=True, exist_ok=False)
        for name, value in (("config.json", config), ("feature_groups.json", groups), ("source_audit.json", source_audit)):
            _json(directory/name, value)
        _json(directory/"literature.json", json.loads((root/"config/nyx_solar_ramp_literature.json").read_text(encoding="utf-8")))
        _parquet(directory/"panel.parquet", augmented)
        inputs = ["config.json", "feature_groups.json", "source_audit.json", "panel.parquet", "literature.json"]
        _json(directory/"manifest.json", {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": config, "input_files": {n: digest(directory/n) for n in inputs}, "code_sha256": _seals(root),
            "versions": {n: version(n) for n in ("numpy", "pandas", "scikit-learn", "scipy")},
            "protected_files": state, "prospective": source_audit["prospective"],
            "primary_variant": "governed", "diagnostic_only": True, "production_modified": False, "promotion_eligible": False,
            "final_holdout_virgin": False, "protocol": "Predeclared chronological final90; all history and initiating case already examined previously, never independent confirmation."})
        if protected_state(root) != state:
            raise ValueError("Operational files changed concurrently; refusing baseline identity claim.")
        _json(output/"latest_prepared.json", {"snapshot": str(directory)})
        _json(directory/"status.json", {"status": "prepared", "rows": len(augmented)})
    return directory


def read_snapshot(directory, *, root):
    directory = safe(root, directory)
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if set(manifest.get("input_files", {})) != INPUT_FILES:
        raise ValueError("Incomplete frozen solar input manifest.")
    for name, checksum in manifest["input_files"].items():
        if Path(name).name != name or digest(directory/name) != checksum:
            raise ValueError("Frozen solar input integrity failure.")
    return directory, manifest, json.loads((directory/"config.json").read_text(encoding="utf-8"))


def verify_results(directory):
    record = json.loads((directory/"results_manifest.json").read_text(encoding="utf-8"))
    if record.get("status") != "completed" or record["manifest_sha256"] != digest(directory/"manifest.json"):
        raise ValueError("Unsealed solar results.")
    if set(record.get("result_files", {})) != RESULT_FILES:
        raise ValueError("Incomplete solar result manifest.")
    for name, checksum in record["result_files"].items():
        if Path(name).name != name or digest(directory/name) != checksum:
            raise ValueError("Solar result checksum mismatch.")


def report(directory, *, root):
    directory, _, _ = read_snapshot(directory, root=root)
    verify_results(directory)
    from .reporting import render_report
    return render_report(directory, root=root)


def backtest(directory, *, root):
    directory, manifest, config = read_snapshot(directory, root=root)
    with exclusive_process_lock(directory/"backtest.lock"):
        if (directory/"results_manifest.json").is_file():
            verify_results(directory)
            LOGGER.info("[Solar] Sealed predictions reused; no retraining.")
            return report(directory, root=root)
        if _seals(root) != manifest["code_sha256"]:
            raise ValueError("Code changed since Prepare. Prepare a new snapshot; do not alter a frozen experiment.")
        before = protected_state(root)
        _json(directory/"status.json", {"status": "running"})
        panel = pd.read_parquet(directory/"panel.parquet")
        groups = json.loads((directory/"feature_groups.json").read_text(encoding="utf-8"))
        try:
            training_record = directory/"training_manifest.json"
            if training_record.is_file():
                record = json.loads(training_record.read_text(encoding="utf-8"))
                if record["manifest_sha256"] != digest(directory/"manifest.json"):
                    raise ValueError("Training checkpoint belongs to another snapshot.")
                if set(record.get("files", {})) != TRAINING_FILES:
                    raise ValueError("Incomplete solar training checkpoint manifest.")
                for name, checksum in record["files"].items():
                    if Path(name).name != name or digest(directory/name) != checksum:
                        raise ValueError("Training checkpoint integrity failure.")
                predictions = pd.read_parquet(directory/"predictions.parquet")
                LOGGER.info("[Solar] Completed training checkpoint reused; evaluation only.")
            else:
                predictions, folds, governance, models = run_policy(panel, groups, config)
                _parquet(directory/"predictions.parquet", predictions)
                _json(directory/"folds.json", folds)
                _json(directory/"governance.json", governance)
                joblib.dump(models, directory/"models.joblib", compress=3)
                training_files = ["predictions.parquet", "folds.json", "governance.json", "models.joblib"]
                _json(training_record, {"manifest_sha256": digest(directory/"manifest.json"),
                    "files": {n: digest(directory/n) for n in training_files}})
            from .evaluation import evaluate
            from .analysis import analyse
            metrics = evaluate(predictions, config)
            metrics["physical_analysis"] = analyse(panel, predictions, config)
            _json(directory/"metrics.json", metrics)
            export = predictions[["variant", "zone", "timestamp_utc", "forecast_origin_utc", "actual", "forecast", "candidate_forecast",
                "candidate_q10", "candidate_q90", "risk_probability", "alert", "gate_reason", "evaluation_phase"]].copy()
            export["delivery_start_local"] = export.timestamp_utc.dt.tz_convert("Europe/Paris").map(lambda t: t.isoformat())
            export.to_csv(directory/"predictions.csv", index=False)
            if before != protected_state(root):
                raise ValueError("Operational sources changed during backtest.")
            files = ["predictions.parquet", "predictions.csv", "folds.json", "governance.json", "metrics.json", "models.joblib", "training_manifest.json"]
            _json(directory/"results_manifest.json", {"status": "completed", "manifest_sha256": digest(directory/"manifest.json"),
                "result_files": {n: digest(directory/n) for n in files}, "production_modified": False, "activation_performed": False})
            _json(directory/"status.json", {"status": "completed", "decision": metrics.get("decision"), "production_modified": False})
        except BaseException as exc:
            _json(directory/"status.json", {"status": "failed", "error": str(exc)})
            raise
    return report(directory, root=root)


def resolve_latest(config, *, root):
    pointer = safe(root, config["output_root"])/"latest_prepared.json"
    return safe(root, json.loads(pointer.read_text(encoding="utf-8"))["snapshot"])
