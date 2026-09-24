"""Resumable, source-sealed variant suite confined to the scarcity laboratory."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
import uuid

import pandas as pd
import yaml

from chronos2_hourly.process_lock import exclusive_process_lock
from . import runner as base
from .variant_runtime import runtime_seals

LOGGER = logging.getLogger(__name__)
NAMESPACE = base.NAMESPACE / "variants"
INPUTS = {"config.json", "base_config.json", "panel.parquet", "control_predictions.parquet", "data_audit.json"}
VARIANT_RESULTS = {"predictions.parquet", "folds.parquet", "governance.parquet", "model_audit.json",
                   "shap_values.parquet", "shap_observations.parquet", "shap_summary.json"}
IDS = {f"xgb_{weight}_{threshold}": (weight == "weighted", threshold)
       for weight in ("unweighted", "weighted") for threshold in ("fixed", "dwt")}


def safe_path(root: Path, value: str | Path) -> Path:
    path = base.safe_output(root, value)
    namespace = root.resolve() / NAMESPACE
    if not path.is_relative_to(namespace) or namespace.resolve() != namespace:
        raise ValueError("Variant writes must stay inside the separate scarcity variants namespace.")
    return path


def validate_config(config: dict) -> None:
    keys = {"schema_version", "source_snapshot", "output_root", "max_parallel", "policy_overrides", "variants",
            "explanations", "diagnostic_only", "production_modified", "activation_performed"}
    if not isinstance(config, dict) or set(config) != keys or config["schema_version"] != 1:
        raise ValueError("Expected the complete variant-suite schema 1, without unknown fields.")
    if config["diagnostic_only"] is not True or config["production_modified"] is not False or config["activation_performed"] is not False:
        raise ValueError("No production mutation or automatic activation is available.")
    if type(config["max_parallel"]) is not int or config["max_parallel"] not in (1, 2):
        raise ValueError("max_parallel must be 1 or 2 CPU workers.")
    for name in ("source_snapshot", "output_root"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"Explicit {name} required.")
    variants = config["variants"]
    if not isinstance(variants, list) or not 1 <= len(variants) <= 4:
        raise ValueError("Select one to four predeclared variants.")
    seen = set()
    for variant in variants:
        if not isinstance(variant, dict) or set(variant) != {"id", "weighted", "threshold_kind"}:
            raise ValueError("Each variant requires id, weighted and threshold_kind only.")
        name = variant["id"]
        if name not in IDS or name in seen or type(variant["weighted"]) is not bool or (variant["weighted"], variant["threshold_kind"]) != IDS[name]:
            raise ValueError("Variant id must match its unique weighting/threshold recipe.")
        seen.add(name)
    overrides = config["policy_overrides"]
    if not isinstance(overrides, dict) or {"feature_columns", "required_feature_columns", "timezone"} & overrides.keys():
        raise ValueError("Policy overrides must not alter audited input identities.")
    explain = config["explanations"]
    limits = {"sample_days_last": (1, 14), "historical_stride_days": (1, 30), "historical_hour": (0, 23)}
    if not isinstance(explain, dict) or set(explain) != set(limits):
        raise ValueError("Explicit deterministic SHAP sampling recipe required.")
    for key, (low, high) in limits.items():
        if type(explain[key]) is not int or not low <= explain[key] <= high:
            raise ValueError(f"explanations.{key}: integer in [{low},{high}] required.")


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    validate_config(config)
    return config


def code_seals() -> dict:
    root = Path(__file__).resolve().parents[1]
    return {**base.code_seals(), **{name: base.digest(root / name) for name in
            ("run_nyx_scarcity_variants.py", "ScarcityVariants.ps1", "config/nyx_scarcity_requirements.txt")}}


def _settings(config: dict, base_config: dict, audit: dict) -> dict:
    from .policy import _parameters
    settings = base.model_settings(base_config, audit)
    settings.update(config["policy_overrides"])
    settings = _parameters(settings)
    if settings["threads"] * config["max_parallel"] > 4:
        raise ValueError("The isolated suite is limited to four CPU threads in total.")
    if settings["threads"] > 2:
        raise ValueError("Each variant is limited to two CPU threads.")
    return settings


def prepare(config: dict, *, root: Path) -> Path:
    validate_config(config)
    output = safe_path(root, config["output_root"])
    original = base.safe_output(root, config["source_snapshot"])
    before = base.protected_state(root)
    with exclusive_process_lock(output / "prepare.lock"):
        source, base_config, source_manifest, panel, audit = base.read_snapshot(original, root=root)
        result_path = source / "results_manifest.json"
        source_result_hash = base.digest(result_path)
        source_result = json.loads(result_path.read_text(encoding="utf-8"))
        if source_result.get("status") != "completed_diagnostic" or any(source_result.get(k) != source_manifest.get(k) for k in ("input_files", "config", "code_sha256")):
            raise ValueError("The HGB control must be a completed result bound to its original input snapshot.")
        base._verify(source, source_result.get("result_files"), base.RESULTS)
        control = pd.read_parquet(source / "predictions.parquet")
        base.validate_predictions(panel, control, base_config)
        settings = _settings(config, base_config, audit)
        runtime = runtime_seals()
        code = code_seals()
        if base.digest(result_path) != source_result_hash:
            raise ValueError("Original HGB result changed during the snapshot read.")
        base._verify(source, source_manifest["input_files"], base.INPUTS)
        base._verify(source, source_result["result_files"], base.RESULTS)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        directory = safe_path(root, output / "snapshots" / stamp)
        directory.mkdir(parents=True, exist_ok=False)
        for name, data in (("config.json", config), ("base_config.json", base_config), ("data_audit.json", audit)):
            base._json(directory / name, data)
        base._parquet(directory / "panel.parquet", panel)
        base._parquet(directory / "control_predictions.parquet", control)
        manifest = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "config": config, "settings": settings, "code_sha256": code, "runtime": runtime,
                    "source_snapshot": str(source), "source_results_manifest_sha256": source_result_hash,
                    "source_input_files": source_manifest["input_files"], "source_result_files": source_result["result_files"],
                    "input_files": {name: base.digest(directory / name) for name in sorted(INPUTS)},
                    "protected_files": before, "diagnostic_only": True, "activation_performed": False}
        if base.protected_state(root) != before:
            raise ValueError("Operational files changed concurrently; no experiment is started.")
        base._json(directory / "manifest.json", manifest)
        base._json(directory / "status.json", {"status": "prepared", "snapshot": str(directory),
                                               "variants": [v["id"] for v in config["variants"]]})
        base._json(output / "latest_prepared.json", {"snapshot": str(directory)})
        LOGGER.info("[Variants] Prepared identical frozen panel: %s (%s rows).", directory, len(panel))
        return directory


def read_suite(directory: Path, *, root: Path):
    directory = safe_path(root, directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    base._verify(directory, manifest.get("input_files"), INPUTS)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    validate_config(config)
    if config != manifest.get("config") or not directory.is_relative_to(safe_path(root, config["output_root"]) / "snapshots"):
        raise ValueError("Variant suite manifest is not bound to its frozen configuration.")
    baseline = json.loads((directory / "base_config.json").read_text(encoding="utf-8"))
    audit = json.loads((directory / "data_audit.json").read_text(encoding="utf-8"))
    if manifest.get("settings") != _settings(config, baseline, audit):
        raise ValueError("Variant suite settings differ from the frozen input recipe.")
    return directory, config, manifest


def _verify_code(manifest: dict) -> None:
    if manifest["code_sha256"] != code_seals() or manifest["runtime"] != runtime_seals():
        raise ValueError("Code/runtime changed after Prepare. Create a fresh suite; completed experiments remain readable.")


def _verify_variant(directory: Path, name: str, manifest: dict) -> dict:
    result = json.loads((directory / name / "results_manifest.json").read_text(encoding="utf-8"))
    if (result.get("status") != "completed" or result.get("variant_id") != name
            or result.get("suite_manifest_sha256") != base.digest(directory / "manifest.json")):
        raise ValueError(f"{name}: completed results are not bound to this suite.")
    if name not in {v["id"] for v in manifest["config"]["variants"]}:
        raise ValueError("Unregistered variant.")
    base._verify(directory / name, result.get("result_files"), VARIANT_RESULTS)
    return result


def _audit_parquet_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Empty dictionaries (valid for DWT) cannot be Arrow struct columns.

    Encode nested audit fields uniformly as JSON text. The actual predictions
    remain numeric and the authoritative model audit remains ordinary JSON.
    """
    out = frame.copy()
    for column in out:
        if out[column].map(lambda value: isinstance(value, (dict, list, tuple))).any():
            out[column] = out[column].map(
                lambda value: json.dumps(value, sort_keys=True, allow_nan=False, default=str)
                if isinstance(value, (dict, list, tuple)) else value)
    return out


def run_worker(directory: Path, name: str, *, root: Path) -> dict:
    from .variant_policy import run_variant_policy
    from .variant_explain import ShapCollector
    manifest_hash = base.digest(safe_path(root, directory) / "manifest.json")
    directory, config, manifest = read_suite(directory, root=root)
    if base.digest(directory / "manifest.json") != manifest_hash:
        raise ValueError("Suite manifest changed while the worker loaded its recipe.")
    candidates = {v["id"]: v for v in config["variants"]}
    if name not in candidates:
        raise ValueError("Worker variant is not registered in the sealed recipe.")
    destination = safe_path(root, directory / name)
    with exclusive_process_lock(destination / "worker.lock"):
        if (destination / "results_manifest.json").is_file():
            _verify_variant(directory, name, manifest)
            return {"status": "reused", "variant": name}
        _verify_code(manifest)
        before = base.protected_state(root)
        panel = pd.read_parquet(directory / "panel.parquet")
        audit = json.loads((directory / "data_audit.json").read_text(encoding="utf-8"))
        baseline = json.loads((directory / "base_config.json").read_text(encoding="utf-8"))
        collector = ShapCollector(name, audit["evaluation_start_day"], audit["evaluation_end_day"],
                                  live_day=audit["delivery_day"], **config["explanations"])
        base._json(destination / "status.json", {"status": "running", "variant": name})
        try:
            trained = run_variant_policy(panel, manifest["settings"], candidates[name], on_predict=collector.observe)
            check_config = deepcopy(baseline)
            check_config["policy"].update(config["policy_overrides"])
            base.validate_predictions(panel, trained.predictions, check_config)
            shap_values, shap_observations = collector.frames()
            for key, frame in (("predictions", trained.predictions), ("folds", trained.folds),
                               ("governance", trained.governance), ("shap_values", shap_values),
                               ("shap_observations", shap_observations)):
                if key in {"folds", "governance"}:
                    frame = _audit_parquet_frame(frame)
                base._parquet(destination / f"{key}.parquet", frame)
            base._json(destination / "model_audit.json", trained.audit)
            base._json(destination / "shap_summary.json", collector.summary())
            if base.protected_state(root) != before:
                raise ValueError("Operational files changed concurrently; result publication stopped.")
            _verify_code(manifest)
            if base.digest(directory / "manifest.json") != manifest_hash:
                raise ValueError("Suite manifest changed during the worker; results are not published.")
            result = {"status": "completed", "variant_id": name, "suite_manifest_sha256": manifest_hash,
                      "completed_at_utc": datetime.now(timezone.utc).isoformat(), "diagnostic_only": True,
                      "production_modified": False, "activation_performed": False,
                      "parquet_nested_audit_encoding": "json_text",
                      "runtime_versions": {key: version(key) for key in ("numpy", "pandas", "scikit-learn", "xgboost")},
                      "result_files": {key: base.digest(destination / key) for key in sorted(VARIANT_RESULTS)}}
            base._json(destination / "results_manifest.json", result)
            base._json(destination / "status.json", {"status": "completed", "variant": name})
            return {"status": "completed", "variant": name}
        except BaseException as exc:
            base._json(destination / "status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                                    "variant": name, "error": f"{type(exc).__name__}: {exc}"})
            raise


def _collect(directory: Path, config: dict, manifest: dict):
    predictions = {"hgb_v1": pd.read_parquet(directory / "control_predictions.parquet")}
    explanations, audits = {}, {}
    for variant in config["variants"]:
        name = variant["id"]
        _verify_variant(directory, name, manifest)
        at = directory / name
        predictions[name] = pd.read_parquet(at / "predictions.parquet")
        audits[name] = json.loads((at / "model_audit.json").read_text(encoding="utf-8"))
        explanations[name] = {"summary": json.loads((at / "shap_summary.json").read_text(encoding="utf-8")),
                              "values": pd.read_parquet(at / "shap_values.parquet"),
                              "observations": pd.read_parquet(at / "shap_observations.parquet")}
    return predictions, explanations, audits


def run_suite(directory: Path, *, root: Path) -> Path:
    directory, config, manifest = read_suite(directory, root=root)
    with exclusive_process_lock(directory / "suite.lock"):
        if (directory / "comparison_manifest.json").is_file():
            LOGGER.info("[Variants] Completed suite reused, no fitting.")
            return report(directory, root=root)
        _verify_code(manifest)
        pending = []
        for v in config["variants"]:
            if (directory / v["id"] / "results_manifest.json").is_file():
                _verify_variant(directory, v["id"], manifest)
                LOGGER.info("[Variants] Reusing completed %s.", v["id"])
            else:
                pending.append(v["id"])
        active, completed = {}, [v["id"] for v in config["variants"] if v["id"] not in pending]
        try:
            while pending or active:
                while pending and len(active) < config["max_parallel"]:
                    name = pending.pop(0)
                    destination = safe_path(root, directory / name)
                    destination.mkdir(parents=True, exist_ok=True)
                    log = (destination / "worker.log").open("a", encoding="utf-8")
                    command = [sys.executable, str(root / "run_nyx_scarcity_variants.py"), "--action", "worker",
                               "--run-directory", str(directory), "--variant", name]
                    try:
                        proc = subprocess.Popen(command, cwd=root, shell=False, stdout=log, stderr=subprocess.STDOUT)
                    except BaseException:
                        log.close()
                        raise
                    active[name] = (proc, log, time.monotonic())
                    LOGGER.info("[Variants] Started %s; log=%s", name, destination / "worker.log")
                base._json(directory / "status.json", {"status": "running", "snapshot": str(directory),
                                                       "completed": completed, "active": list(active), "pending": pending})
                for name, (proc, log, started) in list(active.items()):
                    rc = proc.poll()
                    if rc is None and time.monotonic() - started > 7200:
                        raise TimeoutError(f"{name}: two-hour worker limit exceeded; inspect its private log.")
                    if rc is not None:
                        log.close()
                        del active[name]
                        if rc:
                            raise ValueError(f"{name} returned {rc}; inspect {directory / name / 'worker.log'}. Completed variants are reusable.")
                        _verify_variant(directory, name, manifest)
                        completed.append(name)
                        LOGGER.info("[Variants] Completed %s (%s/%s).", name, len(completed), len(config["variants"]))
                if pending or active:
                    time.sleep(.5)
            from .variant_reporting import build_comparison
            predictions, _, _ = _collect(directory, config, manifest)
            comparison = build_comparison(predictions)
            base._json(directory / "comparison.json", comparison)
            result = {"status": "completed", "suite_manifest_sha256": base.digest(directory / "manifest.json"),
                      "comparison_sha256": base.digest(directory / "comparison.json"),
                      "variant_manifests": {v["id"]: base.digest(directory / v["id"] / "results_manifest.json") for v in config["variants"]},
                      "production_modified": False, "activation_performed": False, "diagnostic_only": True}
            base._json(directory / "comparison_manifest.json", result)
            return report(directory, root=root)
        except BaseException as exc:
            for proc, log, _ in active.values():
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
                log.close()
            base._json(directory / "status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                                   "snapshot": str(directory), "completed": completed,
                                                   "error": f"{type(exc).__name__}: {exc}"})
            raise


def report(directory: Path, *, root: Path) -> Path:
    from .variant_reporting import render_comparison
    directory, config, manifest = read_suite(directory, root=root)
    result = json.loads((directory / "comparison_manifest.json").read_text(encoding="utf-8"))
    if (result.get("status") != "completed" or result.get("suite_manifest_sha256") != base.digest(directory / "manifest.json")
            or result.get("comparison_sha256") != base.digest(directory / "comparison.json")
            or result.get("variant_manifests") != {v["id"]: base.digest(directory / v["id"] / "results_manifest.json") for v in config["variants"]}):
        raise ValueError("Comparison manifest has changed or is not bound to this suite.")
    predictions, explanations, audits = _collect(directory, config, manifest)
    with exclusive_process_lock(directory / "report.lock"):
        destination = directory / "nyx_scarcity_variants.html"
        temporary = directory / f".report_{uuid.uuid4().hex}.html"
        try:
            render_comparison(predictions, json.loads((directory / "comparison.json").read_text(encoding="utf-8")),
                              {**manifest, "models": audits, "data": json.loads((directory / "data_audit.json").read_text(encoding="utf-8")),
                               "diagnostic_only": True, "production_modified": False, "activation_performed": False},
                              temporary, explanations=explanations)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        base._json(directory / "status.json", {"status": "completed", "snapshot": str(directory), "report": str(destination)})
        base._json(safe_path(root, config["output_root"]) / "latest.json", {"snapshot": str(directory), "report": str(destination)})
        LOGGER.info("[Variants] Report: %s", destination)
        return destination


def resolve_latest(config: dict, *, root: Path, completed: bool) -> Path:
    pointer = safe_path(root, config["output_root"]) / ("latest.json" if completed else "latest_prepared.json")
    if not pointer.is_file():
        raise ValueError("No variant suite yet. First use Run or Prepare.")
    return safe_path(root, json.loads(pointer.read_text(encoding="utf-8"))["snapshot"])
