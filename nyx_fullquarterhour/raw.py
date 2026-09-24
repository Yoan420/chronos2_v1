"""Generate fresh causal Chronos quantiles for every train/test delivery day."""
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

from nyx_intrahour.data import digest, load_baseline, ZONES
from nyx_quarterhour.data import day_index
from nyx_quarterhour.sources import read_native_prices
from nyx_quarterhour.inference import checkpoint_identity, load_pipeline
from .inference import infer_batch
from .storage import NAMESPACE, identity, safe_path, write_json


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    keys = {"schema_version", "baseline_delivery_day", "source_manifest", "output_root", "model_id",
            "context_hours", "torch_threads", "model_batch_size", "seed", "raw_start_day", "start_day",
            "end_day", "training_lookback_days", "minimum_training_days", "worker_count", "catboost_threads",
            "bootstrap_repetitions", "bootstrap_seed", "diagnostic_only", "activation_performed", "postprocessing"}
    if not isinstance(cfg, dict) or set(cfg) != keys or cfg["schema_version"] != 1:
        raise ValueError("Complete full-chain schema 1 required.")
    if cfg["diagnostic_only"] is not True or cfg["activation_performed"] is not False:
        raise ValueError("This experiment cannot activate or publish a production model.")
    if cfg["postprocessing"] != "daily_catboost_then_daily_refitted_governed_kalman":
        raise ValueError("The complete frozen architecture is required.")
    for key in ("context_hours", "torch_threads", "model_batch_size", "seed", "bootstrap_seed",
                "training_lookback_days", "minimum_training_days", "worker_count", "catboost_threads", "bootstrap_repetitions"):
        if type(cfg[key]) is not int or cfg[key] < (0 if "seed" in key else 1):
            raise ValueError(f"Invalid {key}.")
    if cfg["context_hours"] * 4 > 8192 or cfg["training_lookback_days"] != 365 or cfg["minimum_training_days"] != 30:
        raise ValueError("Physical context and incumbent training policy must be preserved.")
    for key in ("baseline_delivery_day", "raw_start_day", "start_day", "end_day"):
        day_index(cfg[key], "h")
    if not pd.Timestamp(cfg["raw_start_day"]) < pd.Timestamp(cfg["start_day"]) <= pd.Timestamp(cfg["end_day"]):
        raise ValueError("Raw history must precede the evaluation period.")
    if cfg["model_id"] != "amazon/chronos-2" or Path(cfg["output_root"]) != NAMESPACE:
        raise ValueError("Only the local Chronos-2 checkpoint and isolated output namespace are allowed.")
    return cfg


def versions():
    return {p: importlib.metadata.version(p) for p in
            ("chronos-forecasting", "torch", "transformers", "pandas", "numpy", "catboost", "pykalman", "holidays")}


def verify_files(root: Path, files: dict):
    for name, expected in files.items():
        path = Path(name)
        path = path if path.is_absolute() else root / path
        if digest(path) != expected:
            raise ValueError(f"Sealed file changed: {name}")


def validate_raw(frame: pd.DataFrame, civil: str, frequency: str):
    expected = day_index(civil, frequency)
    if set(frame.zone) != set(ZONES) or frame.duplicated(["zone", "timestamp_utc"]).any():
        raise ValueError("Raw quantile identities are incomplete or duplicated.")
    for zone in ZONES:
        block = frame.loc[frame.zone.eq(zone)].sort_values("timestamp_utc")
        if not pd.DatetimeIndex(block.timestamp_utc).equals(expected):
            raise ValueError("Raw quantiles do not cover the complete physical delivery day.")
        q = block[["q10", "q50", "q90"]].to_numpy(float)
        if not np.isfinite(q).all() or (q[:, 0] > q[:, 1]).any() or (q[:, 1] > q[:, 2]).any():
            raise ValueError("Raw quantiles must be finite and ordered.")


def run_raw(config: dict, *, root: Path, resume: Path | None = None) -> Path:
    from .data import FullChainInputs

    began = time.perf_counter()
    root = root.resolve()
    output = safe_path(root, Path(config["output_root"]))
    directory = safe_path(root, resume if resume else output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]))
    if resume is None:
        directory.mkdir(parents=True, exist_ok=False)
    if not directory.is_dir():
        raise ValueError("Resume directory absent.")
    daily = safe_path(root, directory / "raw_daily")
    daily.mkdir(exist_ok=True)
    names = ["nyx_fullquarterhour/raw.py", "nyx_fullquarterhour/data.py", "nyx_fullquarterhour/storage.py", "nyx_fullquarterhour/inference.py",
             "nyx_quarterhour/data.py", "nyx_quarterhour/inference.py", "nyx_quarterhour/sources.py",
             "nyx_intrahour/data.py", "nyx_intrahour/runner.py", "chronos2_hourly/features.py", "chronos2_modular/common.py"]
    code_hashes = {name: digest(root / name) for name in names}
    source = Path(config["source_manifest"])
    source = source if source.is_absolute() else root / source
    source_sha = digest(source)
    prices, source_audit = read_native_prices(source)
    baseline, baseline_audit = load_baseline(root, config["baseline_delivery_day"])
    inputs = FullChainInputs(prices, baseline)
    scoring, scoring_audit = inputs.evaluation_baseline(baseline, config["start_day"], config["end_day"])
    recipes, recipe_files = {}, {}
    for zone in ZONES:
        audit_path = root / "runs/experiments/nuclear_forecast_v1" / config["baseline_delivery_day"] / zone.lower() / "civil_pit_v2/report_only/frozen_result/audits.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))["result"]
        recipes[zone] = {key: audit[key] for key in ("residual_recipe", "kalman_filter_parameters", "kalman_covariate_config")}
        recipe_files[str(audit_path)] = digest(audit_path)
    checkpoint = checkpoint_identity(config["model_id"])
    protocol = {"config": config, "code_sha256": code_hashes, "source_sha256": source_sha,
                "source_audit": source_audit, "baseline_audit": baseline_audit, "recipes": recipes,
                "recipe_files": recipe_files, "checkpoint": checkpoint, "versions": versions(),
                "scoring_audit": scoring_audit,
                "training_scope": "same_available_native_history_capped_at_365_days_in_both_frequencies",
                "architecture": "Chronos2_then_daily_CatBoost_then_daily_refitted_governed_Kalman",
                "production_modified": False, "publication_vintage_verified": False}
    protocol_id = identity(protocol)
    lock = safe_path(root, directory / "raw_protocol.json")
    manifest_path = safe_path(root, directory / "inputs.manifest.json")
    if resume is not None:
        old = json.loads(lock.read_text(encoding="utf-8"))
        if old["identity"] != protocol_id:
            raise ValueError("Raw protocol, code, source or dependencies changed; resume refused.")
        if digest(manifest_path) != old["inputs_manifest_sha256"]:
            raise ValueError("Input manifest changed.")
        input_hashes = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
        verify_files(directory, input_hashes)
    else:
        for name, frame in (("native_prices.parquet", prices), ("baseline.parquet", baseline), ("scoring_baseline.parquet", scoring)):
            frame.to_parquet(safe_path(root, directory / name), index=False)
        input_hashes = {name: digest(directory / name) for name in ("native_prices.parquet", "baseline.parquet", "scoring_baseline.parquet")}
        write_json(manifest_path, {"files": input_hashes}, root=root)
        write_json(lock, {"identity": protocol_id, "protocol": protocol, "inputs_manifest_sha256": digest(manifest_path)}, root=root)
    lock_sha = digest(lock)
    completed_manifest = safe_path(root, directory / "raw_outputs.manifest.json")
    if completed_manifest.is_file():
        completed = json.loads(completed_manifest.read_text(encoding="utf-8"))
        if completed["status"] != "complete" or completed["protocol_identity"] != protocol_id:
            raise ValueError("Completed raw run identity mismatch.")
        verify_files(directory, completed["files"])
        # An already sealed raw run remains byte-identical when resuming the
        # later learned stages; in particular do not rewrite its runtime.
        write_json(directory / "status.json", {"status": "raw_complete", "directory": str(directory)}, root=root)
        return directory
    frames = {"h": [], "15min": []}
    audits = []
    days = pd.date_range(config["raw_start_day"], config["end_day"], freq="D")
    pipeline = None
    completed = 0
    try:
        for day in days:
            civil = day.date().isoformat()
            for frequency in ("h", "15min"):
                path = safe_path(root, daily / f"{civil}_{frequency}.parquet")
                record_path = safe_path(root, path.with_suffix(".json"))
                if path.exists() or record_path.exists():
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                    if record["protocol_identity"] != protocol_id or digest(path) != record["data_sha256"]:
                        raise ValueError("Raw daily checkpoint mismatch.")
                    predicted = pd.read_parquet(path)
                    audit = record["origins"]
                else:
                    contexts, futures, audit = inputs.build_origin(civil, frequency, context_hours=config["context_hours"])
                    if pipeline is None:
                        pipeline = load_pipeline(config["model_id"], torch_threads=config["torch_threads"], seed=config["seed"])
                    raw = infer_batch(contexts, futures, freq=frequency,
                                      context_length=config["context_hours"] * (4 if frequency == "15min" else 1),
                                      prediction_length=len(futures[0]), model_batch_size=config["model_batch_size"], pipeline=pipeline)
                    identifiers = {frame.item_id.iloc[0]: zone for frame, zone in zip(contexts, ZONES)}
                    predicted = pd.DataFrame({"timestamp_utc": pd.to_datetime(raw.timestamp, utc=True), "zone": raw.item_id.map(identifiers)})
                    for quantile in ("q10", "q50", "q90"):
                        predicted[quantile] = raw[quantile].to_numpy(float)
                    validate_raw(predicted, civil, frequency)
                    predicted.to_parquet(path, index=False)
                    write_json(record_path, {"protocol_identity": protocol_id, "data_sha256": digest(path), "origins": audit}, root=root)
                validate_raw(predicted, civil, frequency)
                frames[frequency].append(predicted)
                audits.extend(audit)
                completed += 1
                progress = {"status": "raw_inference", "completed_batches": completed, "total_batches": len(days) * 2,
                            "delivery_day": civil, "frequency": frequency, "elapsed_seconds": time.perf_counter() - began,
                            "directory": str(directory), "activation_performed": False}
                write_json(directory / "status.json", progress, root=root)
                print(json.dumps(progress), flush=True)
        for frequency, parts in frames.items():
            pd.concat(parts, ignore_index=True).to_parquet(safe_path(root, directory / f"raw_{frequency}.parquet"), index=False)
        pd.DataFrame(audits).to_parquet(safe_path(root, directory / "raw_origin_audit.parquet"), index=False)
        verify_files(root, code_hashes)
        verify_files(root, recipe_files)
        verify_files(directory, input_hashes)
        for record in baseline_audit["identities"]:
            verify_files(root, {**record["files"], **record["observations"]})
        if digest(source) != source_sha or checkpoint_identity(config["model_id"]) != checkpoint or digest(lock) != lock_sha:
            raise ValueError("Source, checkpoint or protocol changed during inference.")
        read_native_prices(source)
        sealed = {str(p.relative_to(directory)): digest(p) for p in sorted(daily.iterdir()) if p.is_file()}
        sealed.update({name: digest(directory / name) for name in ("raw_h.parquet", "raw_15min.parquet", "raw_origin_audit.parquet", "raw_protocol.json", "inputs.manifest.json")})
        write_json(directory / "raw_outputs.manifest.json", {"files": sealed, "protocol_identity": protocol_id,
                   "status": "complete", "runtime_seconds": time.perf_counter() - began}, root=root)
        write_json(directory / "status.json", {"status": "raw_complete", "directory": str(directory), "total_batches": completed}, root=root)
    except Exception as exc:
        write_json(directory / "status.json", {"status": "failed", "phase": "raw_inference", "reason": f"{type(exc).__name__}: {exc}",
                   "directory": str(directory), "results_valid": False}, root=root)
        raise
    return directory
