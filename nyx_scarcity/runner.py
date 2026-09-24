"""Sealed research snapshots, never an operational forecast writer.

All writes are confined to runs/experiments/nyx_scarcity_v1. Completed
backtests are reused without fitting again; a changed recipe needs Prepare.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import yaml

from chronos2_hourly.process_lock import exclusive_process_lock

LOGGER = logging.getLogger(__name__)
NAMESPACE = Path("runs/experiments/nyx_scarcity_v1")
INPUTS = {"config.json", "panel.parquet", "data_audit.json"}
RESULTS = {"predictions.parquet", "folds.parquet", "governance.parquet",
           "model_audit.json", "metrics.json", "daily.parquet", "hourly.parquet"}
PROTECTED = ("Forecast.ps1", "run_complete_forecast.py", "run_multicountry_forecast.py",
             "run_mkonline_live_hourly.py", "chronos2_hourly_live_zones.yaml",
             "config/nuclear_forecast.yaml", "config/kalman_operational.yaml",
             "config/nuclear_kalman_extreme.yaml", "PriceExpert.ps1")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_output(root: Path, value: str | Path) -> Path:
    """Reject escapes, including junctions pointing at another experiment."""
    root = root.resolve()
    namespace = root / NAMESPACE
    raw = Path(value)
    path = (raw if raw.is_absolute() else root / raw).absolute()
    resolved = path.resolve()
    if namespace.resolve() != namespace or not resolved.is_relative_to(namespace):
        raise ValueError(f"Output must remain inside {namespace}; operational paths are forbidden.")
    if path != resolved:
        raise ValueError("Output path aliases, symlinks and parent traversal are forbidden.")
    return resolved


def _json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                        allow_nan=False, default=str) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _parquet(path: Path, value: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        value.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def protected_state(root: Path) -> dict:
    return {name: digest(root / name) for name in PROTECTED if (root / name).is_file()}


def code_seals() -> dict:
    root = Path(__file__).resolve().parents[1]
    names = [str(p.relative_to(root)).replace("\\", "/") for p in sorted((root / "nyx_scarcity").glob("*.py"))]
    names += ["run_nyx_scarcity.py", "Scarcity.ps1", "economic_value/data.py",
              "economic_value/extreme_data.py", "marginal_cost_expert/data.py",
              "marginal_cost_expert/evaluation.py", "chronos2_hourly/process_lock.py",
              "materialize_saturn_kalman_fuel.py", "materialize_saturn_kalman_weather.py"]
    return {name: digest(root / name) for name in names}


def validate_config(config: dict) -> None:
    allowed = {"schema_version", "output_root", "baseline_model", "candidate_model", "zones",
               "delivery_day", "end_day", "timezone", "cutoff_time", "evaluation_days",
               "data", "policy", "refresh", "diagnostic_only", "production_modified", "activation_performed"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Unknown scarcity configuration fields.")
    if config.get("schema_version") != 1 or config.get("baseline_model") != "nuclear_kalman" or config.get("candidate_model") != "nyx_scarcity":
        raise ValueError("Expected schema 1, frozen nuclear_kalman baseline and nyx_scarcity challenger.")
    if config.get("diagnostic_only") is not True or config.get("production_modified") is not False or config.get("activation_performed") is not False:
        raise ValueError("Research only: production modification and activation are forbidden.")
    if config.get("timezone") != "Europe/Paris" or config.get("cutoff_time") != "08:00" or config.get("evaluation_days") != 365:
        raise ValueError("Keep 365 civil evaluation days with D-1 08:00 Europe/Paris cutoff.")
    zones = config.get("zones")
    if not isinstance(zones, list) or not zones or any(not isinstance(z, str) for z in zones) or len(set(zones)) != len(zones) or set(zones) - {"FR", "DE", "BE", "NL"}:
        raise ValueError("Select unique countries among FR, DE, BE and NL.")
    if not isinstance(config.get("output_root"), str) or not config["output_root"].strip():
        raise ValueError("An isolated output_root is required.")
    for key in ("delivery_day", "end_day"):
        value = config.get(key)
        if value is not None:
            if not isinstance(value, str) or pd.Timestamp(value).strftime("%Y-%m-%d") != value:
                raise ValueError(f"{key} must be YYYY-MM-DD or null.")
    for name in ("data", "policy"):
        if not isinstance(config.get(name), dict):
            raise ValueError(f"Explicit {name} settings required.")
    policy = config["policy"]
    if {"feature_columns", "required_feature_columns", "timezone"} & set(policy):
        raise ValueError("Feature allowlists and timezone come only from the audited data contract.")
    if policy.get("training_window_days") != 365 or type(policy.get("minimum_training_days")) is not int or not 90 <= policy["minimum_training_days"] <= 365:
        raise ValueError("Training uses 365 prior civil days at most, minimum_training_days between 90 and 365.")


def load_config(path: Path, **overrides) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    config.update({k: v for k, v in overrides.items() if v is not None})
    validate_config(config)
    return config


def model_settings(config: dict, audit: dict) -> dict:
    names = audit["feature_columns"]
    required = audit["required_feature_columns"]
    if not names or len(set(names)) != len(names) or not set(required).issubset(names):
        raise ValueError("Invalid audited feature allowlist.")
    return {**deepcopy(config["policy"]), "timezone": config["timezone"],
            "feature_columns": list(names), "required_feature_columns": list(required)}


def audit_inputs(config: dict, *, root: Path):
    from .data import load_inputs
    from .policy import _parameters
    validate_config(config)
    safe_output(root, config["output_root"])
    before = protected_state(root)
    panel, audit = load_inputs(config, root=root)
    _parameters(model_settings(config, audit))
    if protected_state(root) != before:
        raise ValueError("Operational files changed concurrently while reading; retry with stable sources.")
    return panel, audit


def prepare(config: dict, *, root: Path) -> Path:
    validate_config(config)
    output = safe_output(root, config["output_root"])
    with exclusive_process_lock(output / "prepare.lock"):
        LOGGER.info("[Scarcity] Lecture des forecasts et des sources PIT, sans calcul Chronos.")
        panel, audit = audit_inputs(config, root=root)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        snapshot = safe_output(root, output / "snapshots" / stamp)
        snapshot.mkdir(parents=True, exist_ok=False)
        _json(snapshot / "config.json", config)
        _json(snapshot / "data_audit.json", audit)
        _parquet(snapshot / "panel.parquet", panel)
        manifest = {"schema_version": 1, "snapshot": str(snapshot), "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "input_files": {name: digest(snapshot / name) for name in sorted(INPUTS)},
                    "code_sha256": code_seals(), "diagnostic_only": True, "activation_performed": False,
                    "production_modified": False, "config": config}
        _json(snapshot / "manifest.json", manifest)
        _json(snapshot / "status.json", {"status": "prepared", "snapshot": str(snapshot), "rows": len(panel)})
        _json(output / "latest_prepared.json", {"snapshot": str(snapshot)})
        LOGGER.info("[Scarcity] Snapshot prepare : %s (%d lignes).", snapshot, len(panel))
        return snapshot


def _verify(snapshot: Path, hashes: dict, expected: set[str]) -> None:
    if not isinstance(hashes, dict) or set(hashes) != expected:
        raise ValueError("Incomplete snapshot checksum manifest.")
    for name, checksum in hashes.items():
        if (snapshot / name).resolve().parent != snapshot or digest(snapshot / name) != checksum:
            raise ValueError(f"Snapshot checksum mismatch: {name}.")


def read_snapshot(snapshot: Path, *, root: Path):
    snapshot = safe_output(root, snapshot)
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    _verify(snapshot, manifest.get("input_files"), INPUTS)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    validate_config(config)
    if config != manifest.get("config") or not snapshot.is_relative_to(safe_output(root, config["output_root"]) / "snapshots"):
        raise ValueError("Snapshot configuration binding mismatch.")
    audit = json.loads((snapshot / "data_audit.json").read_text(encoding="utf-8"))
    return snapshot, config, manifest, pd.read_parquet(snapshot / "panel.parquet"), audit


def validate_predictions(panel: pd.DataFrame, predictions: pd.DataFrame, config: dict) -> None:
    """A challenger never drops difficult hours or changes reference values."""
    if len(predictions) != len(panel) or not set(panel).issubset(predictions):
        raise ValueError("The candidate must preserve every baseline row and field.")
    try:
        pd.testing.assert_frame_equal(panel.reset_index(drop=True), predictions[panel.columns].reset_index(drop=True), check_exact=True)
    except AssertionError as exc:
        raise ValueError("Frozen inputs or row ordering were changed by the candidate.") from exc
    required = {"candidate_forecast", "candidate_q10", "candidate_q90", "applied_correction",
                "raw_correction", "selected_weight", "expert_ready", "spike_probability",
                "gate_reason", "threshold_eur_mwh", "interval_status"}
    if required - set(predictions):
        raise ValueError(f"Missing candidate fields: {sorted(required-set(predictions))}")
    valid = np.isfinite(panel.forecast)
    candidate = predictions.candidate_forecast
    if not np.isfinite(candidate[valid]).all() or np.isfinite(candidate[~valid]).any():
        raise ValueError("Candidate must preserve the baseline's finite forecast support.")
    if not np.allclose(candidate[valid], panel.loc[valid, "forecast"] + predictions.loc[valid, "applied_correction"], atol=1e-9, rtol=0):
        raise ValueError("Candidate is not baseline plus the audited correction.")
    probability = predictions.spike_probability.dropna()
    if not probability.between(0, 1).all():
        raise ValueError("Invalid spike probability.")
    if not pd.api.types.is_bool_dtype(predictions.expert_ready):
        raise ValueError("expert_ready must be explicit booleans.")
    if predictions.loc[~predictions.expert_ready, "applied_correction"].abs().gt(1e-10).any():
        raise ValueError("Missing expert must retain the baseline.")
    weights = np.asarray(config["policy"]["candidate_weights"], dtype=float)
    if not np.isclose(predictions.selected_weight.to_numpy()[:, None], weights[None, :], atol=1e-12, rtol=0).any(axis=1).all():
        raise ValueError("Selected weights are outside the frozen candidate bank.")
    if predictions.applied_correction.abs().gt(config["policy"]["correction_clip_eur_mwh"] + 1e-9).any():
        raise ValueError("Candidate correction exceeds the configured cap.")
    for name in ("candidate_q10", "candidate_q90", "raw_correction", "applied_correction"):
        if np.isinf(predictions[name].to_numpy(float)).any():
            raise ValueError(f"Infinite candidate field: {name}.")
    quantiles = predictions[["candidate_q10", "candidate_forecast", "candidate_q90"]].dropna()
    if (quantiles.candidate_q10.gt(quantiles.candidate_forecast) | quantiles.candidate_forecast.gt(quantiles.candidate_q90)).any():
        raise ValueError("Crossed candidate quantiles.")


def evaluate(snapshot: Path, *, root: Path) -> Path:
    from .policy import run_policy
    from .reporting import evaluate_predictions
    snapshot, config, manifest, panel, data_audit = read_snapshot(snapshot, root=root)
    with exclusive_process_lock(snapshot / "evaluation.lock"):
        if (snapshot / "results_manifest.json").is_file():
            LOGGER.info("[Scarcity] Backtest deja termine, aucun reentrainement.")
            return report(snapshot, root=root)
        if manifest["code_sha256"] != code_seals():
            raise ValueError("Code changed since Prepare. Create a new snapshot; no stale partial model is reused.")
        before = protected_state(root)
        _json(snapshot / "status.json", {"status": "running", "stage": "rolling_tail_expert", "snapshot": str(snapshot)})
        try:
            trained = run_policy(panel, model_settings(config, data_audit))
            validate_predictions(panel, trained.predictions, config)
            metrics, daily, hourly = evaluate_predictions(trained.predictions)
            for name, frame in (("predictions", trained.predictions), ("folds", trained.folds),
                                ("governance", trained.governance), ("daily", daily), ("hourly", hourly)):
                _parquet(snapshot / f"{name}.parquet", frame)
            _json(snapshot / "model_audit.json", trained.audit)
            _json(snapshot / "metrics.json", metrics)
            if before != protected_state(root):
                raise ValueError("Operational files changed concurrently during the experiment; publication stopped.")
            result = {**manifest, "status": "completed_diagnostic", "data": data_audit, "model": trained.audit,
                      "production_modified": False, "activation_performed": False, "diagnostic_only": True,
                      "operational_files_unchanged": True, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                      "runtime_versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "pyarrow", "plotly")},
                      "result_files": {name: digest(snapshot / name) for name in sorted(RESULTS)}}
            _json(snapshot / "results_manifest.json", result)
            return report(snapshot, root=root)
        except BaseException as exc:
            _json(snapshot / "status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                             "snapshot": str(snapshot), "error": f"{type(exc).__name__}: {exc}"})
            raise


def report(snapshot: Path, *, root: Path) -> Path:
    from .reporting import render_report
    snapshot, config, manifest, _, _ = read_snapshot(snapshot, root=root)
    result = json.loads((snapshot / "results_manifest.json").read_text(encoding="utf-8"))
    if result.get("status") != "completed_diagnostic" or any(result.get(k) != manifest[k] for k in ("input_files", "code_sha256", "config")):
        raise ValueError("Result manifest is not bound to the prepared snapshot.")
    _verify(snapshot, result.get("result_files"), RESULTS)
    with exclusive_process_lock(snapshot / "report.lock"):
        path = snapshot / "nyx_scarcity_report.html"
        temporary = snapshot / f".report_{uuid.uuid4().hex}.html"
        try:
            render_report(pd.read_parquet(snapshot / "predictions.parquet"),
                          json.loads((snapshot / "metrics.json").read_text(encoding="utf-8")),
                          pd.read_parquet(snapshot / "daily.parquet"), pd.read_parquet(snapshot / "hourly.parquet"),
                          result, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        _json(snapshot / "status.json", {"status": "completed", "snapshot": str(snapshot), "report": str(path)})
        _json(safe_output(root, config["output_root"]) / "latest.json", {"snapshot": str(snapshot), "report": str(path)})
        LOGGER.info("[Scarcity] Rapport : %s", path)
        return path


def resolve_snapshot(root: Path, config: dict, value: Path | None, *, prepared: bool = False) -> Path:
    if value is not None:
        return safe_output(root, value)
    output = safe_output(root, config["output_root"])
    pointer = output / ("latest_prepared.json" if prepared else "latest.json")
    if not pointer.is_file():
        raise ValueError("No prepared/completed snapshot; first use -Action Prepare or -Action Run.")
    return safe_output(root, json.loads(pointer.read_text(encoding="utf-8"))["snapshot"])
