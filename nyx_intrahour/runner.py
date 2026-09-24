"""Prepare, evaluate and report an isolated, reproducible intrahour challenger."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time
import uuid

import numpy as np
import pandas as pd
import yaml

from .data import DataUnavailable, digest, join_features, load_baseline, read_native_sources
from .features import build_hourly_features
from .reporting import render_report

NAMESPACE = Path("runs/experiments/nyx_intrahour_v1")
EVALUATION_KEYS = {"initial_train_days", "validation_days", "test_days", "window_days",
                   "refit_every_days", "min_train_rows", "bootstrap_repetitions", "seed"}


def safe_output(root: Path, value: str | Path) -> Path:
    root = root.resolve()
    path = Path(value)
    path = (root/path).resolve() if not path.is_absolute() else path.resolve()
    namespace = root/NAMESPACE
    if namespace.resolve() != namespace:
        raise ValueError("The intrahour output namespace cannot be redirected by a link or junction.")
    if not path.is_relative_to(namespace):
        raise ValueError("All intrahour outputs must stay inside runs/experiments/nyx_intrahour_v1.")
    return path


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (str, int)) or value is None:
        return value
    return str(value)


def write_json(path: Path, data: dict) -> None:
    temp = path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    expected = {"schema_version", "baseline_delivery_day", "source_manifest", "output_root",
                "evaluation", "diagnostic_only", "activation_performed"}
    if not isinstance(config, dict) or set(config) != expected or config["schema_version"] != 1:
        raise ValueError("Complete intrahour schema 1 required; unknown settings rejected.")
    if config["diagnostic_only"] is not True or config["activation_performed"] is not False:
        raise ValueError("This experiment cannot activate a model.")
    if not isinstance(config["evaluation"], dict) or set(config["evaluation"]) != EVALUATION_KEYS:
        raise ValueError("Complete explicit evaluation settings required.")
    for key, value in config["evaluation"].items():
        if type(value) is not int or value < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid evaluation setting {key}.")
    return config


def _code_identity(root: Path) -> dict:
    files = sorted((root/"nyx_intrahour").glob("*.py"))+[root/"run_nyx_intrahour.py"]
    return {str(p.relative_to(root)): digest(p) for p in files if p.is_file()}


def run(config: dict, *, root: Path, audit_only: bool = False) -> Path:
    began = time.perf_counter()
    output = safe_output(root, config["output_root"])
    directory = safe_output(root, output/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]))
    directory.mkdir(parents=True, exist_ok=False)
    summary = {"schema_version": 1, "status": "preparing", "diagnostic_only": True,
               "activation_performed": False, "production_modified": False,
               "variant": "nyx_hourly_with_intrahour_forecast_profiles",
               "config": config, "code_sha256": _code_identity(root),
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "decision": {"encouraging": False, "promotion_allowed": False}}
    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "pyarrow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    summary["versions"] = versions
    write_json(directory/"status.json", summary)
    tables = {}
    try:
        value = Path(config["source_manifest"])
        source_path = value if value.is_absolute() else root/value
        vintages, sources, native_audit = read_native_sources(source_path)
        summary["native_source_audit"] = native_audit
        baseline, baseline_audit = load_baseline(root, config["baseline_delivery_day"])
        summary["baseline_audit"] = baseline_audit
        civil = baseline.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")
        hourly, audit = build_hourly_features(vintages, sources, civil.min(), civil.max())
        summary["feature_audit"] = audit
        panel = join_features(baseline, hourly)
        panel.to_parquet(directory/"panel.parquet", index=False)
        hourly.to_parquet(directory/"hourly_features.parquet", index=True)
        # Sealed copies allow re-evaluation without refreshing a provider or NYX.
        vintages.to_parquet(directory/"native_vintages.parquet", index=False)
        write_json(directory/"sources.json", {"sources": sources, "provenance": native_audit})
        write_json(directory/"inputs.manifest.json", {"files": {name: digest(directory/name)
                   for name in ("panel.parquet", "hourly_features.parquet", "native_vintages.parquet", "sources.json")},
                   "baseline_audit": baseline_audit, "source_audit": native_audit})
        if audit_only:
            summary["status"] = "prepared"
        elif audit["complete_hours"] == 0:
            raise DataUnavailable("Aucune heure ne possède quatre prévisions natives admissibles pour toutes les sources.")
        else:
            from .evaluation import evaluate_variant
            result = evaluate_variant(panel, **config["evaluation"])
            for name, value in result.items():
                if isinstance(value, pd.DataFrame):
                    if not name.replace("_", "").isalnum():
                        raise ValueError("Unsafe evaluation artifact name.")
                    tables[name] = value
                    value.to_parquet(directory/f"{name}.parquet", index=False)
                    value.to_csv(directory/f"{name}.csv", index=False)
                else:
                    summary[name] = clean(value)
            summary["status"] = result.get("status", "complete")
            summary["activation_performed"] = False
            summary["production_modified"] = False
        if digest(source_path) != native_audit["manifest_sha256"] or digest(Path(native_audit["data_path"])) != native_audit["data_sha256"]:
            raise ValueError("Native source changed during evaluation; results cannot be accepted.")
        for item in baseline_audit["identities"]:
            for filename, expected in {**item["files"], **item["observations"]}.items():
                if digest(Path(filename)) != expected:
                    raise ValueError("A baseline source changed during evaluation.")
    except DataUnavailable as exc:
        summary.update(status="data_unavailable", reason=str(exc))
    except Exception as exc:
        # The detailed exception is retained locally; never turn a failure into a successful result.
        summary.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        summary["decision"] = {"encouraging": False, "promotion_allowed": False}
        summary["results_valid"] = False
        tables = {}
    summary["runtime_seconds"] = time.perf_counter()-began
    write_json(directory/"summary.json", summary)
    write_json(directory/"status.json", {"status": summary["status"], "reason": summary.get("reason"),
                                         "report": str(directory/"report.html"), "activation_performed": False})
    render_report(directory, summary, tables)
    write_json(directory/"outputs.manifest.json", {"files": {p.name: digest(p)
               for p in sorted(directory.iterdir()) if p.is_file() and p.name != "outputs.manifest.json"}})
    write_json(output/"latest.json", {"directory": str(directory), "status": summary["status"]})
    return directory
