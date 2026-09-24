"""Run and seal the matched-frequency Chronos experiment locally."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import stat
import time
import uuid

import pandas as pd
import yaml

from nyx_intrahour.data import digest, load_baseline, ZONES
from nyx_intrahour.runner import clean, write_json as _write_json
from .data import MatchedInputs, day_index

NAMESPACE = Path("runs/experiments/nyx_quarterhour_v1")


def write_json(path: Path, value: dict, *, root: Path):
    safe_path(root,path)
    safe_path(root,path.with_name(path.name+".tmp"))
    _write_json(path,value)


def safe_path(root: Path, path: Path) -> Path:
    root = root.resolve()
    namespace = root/NAMESPACE
    path = path if path.is_absolute() else root/path
    if namespace.resolve() != namespace or path.resolve() != path.absolute() or not path.resolve().is_relative_to(namespace):
        raise ValueError("Quarter-hour outputs cannot leave their namespace or follow links/junctions.")
    for current in (path,*path.parents):
        if current == root:
            break
        try:
            attributes = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(attributes.st_mode) or getattr(attributes,"st_file_attributes",0)&getattr(stat,"FILE_ATTRIBUTE_REPARSE_POINT",1024):
            raise ValueError("Quarter-hour outputs cannot traverse a reparse point.")
    return path


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    expected = {"schema_version", "baseline_delivery_day", "source_manifest", "output_root", "model_id",
                "context_hours", "torch_threads", "model_batch_size", "seed", "start_day", "end_day",
                "bootstrap_repetitions", "bootstrap_seed", "diagnostic_only", "activation_performed", "postprocessing"}
    if not isinstance(cfg, dict) or set(cfg) != expected or cfg["schema_version"] != 1:
        raise ValueError("Complete quarter-hour schema 1 required.")
    if cfg["diagnostic_only"] is not True or cfg["activation_performed"] is not False or cfg["postprocessing"] != "none":
        raise ValueError("This fixed experiment cannot train a postprocessor or activate a model.")
    for key in ("context_hours", "torch_threads", "model_batch_size", "bootstrap_repetitions", "seed", "bootstrap_seed"):
        if type(cfg[key]) is not int or cfg[key] < (0 if "seed" in key else 1):
            raise ValueError(f"Invalid {key}.")
    if cfg["context_hours"]*4 > 8192:
        raise ValueError("Matched quarter-hour context would exceed the checkpoint limit.")
    if pd.Timestamp(cfg["end_day"]) < pd.Timestamp(cfg["start_day"]):
        raise ValueError("Invalid evaluation dates.")
    return cfg


def _json_identity(value):
    return hashlib.sha256(json.dumps(clean(value),sort_keys=True,allow_nan=False).encode()).hexdigest()


def run(config: dict, *, root: Path, resume: Path | None = None) -> Path:
    from .sources import read_native_prices
    from .inference import load_pipeline, infer_batch, checkpoint_identity
    from .evaluation import aggregate_quarterhour_predictions, evaluate_predictions
    from .reporting import render_report

    began = time.perf_counter()
    root = root.resolve()
    output = safe_path(root, Path(config["output_root"]))
    directory = safe_path(root, resume if resume else output/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]))
    if not resume:
        directory.mkdir(parents=True,exist_ok=False)
    if not directory.is_dir():
        raise ValueError("Resume directory must exist.")
    daily = safe_path(root,directory/"daily")
    daily.mkdir(exist_ok=True)
    source = Path(config["source_manifest"])
    source = source if source.is_absolute() else root/source
    files = sorted((root/"nyx_quarterhour").glob("*.py"))+[root/"run_nyx_quarterhour.py",root/"nyx_intrahour/data.py",root/"nyx_intrahour/runner.py"]
    code_hashes = {str(p.relative_to(root)):digest(p) for p in files if p.is_file()}
    summary = {"schema_version":1,"status":"preparing","config":config,"code_sha256":code_hashes,
               "created_at_utc":datetime.now(timezone.utc).isoformat(),"diagnostic_only":True,
               "activation_performed":False,"production_modified":False,
               "architecture":"frozen_Chronos2_native15_to_hourly_mean_vs_matched_hourly",
               "postprocessing":"none_for_both_Chronos_candidates",
               "decision":{"encouraging":False,"promotion_allowed":False},
               "production_pit_evidence":False}
    versions = {}
    for package in ("chronos-forecasting","torch","transformers","pandas","numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    summary["versions"] = versions
    tables = {}
    try:
        prices,native_audit = read_native_prices(source)
        baseline,baseline_audit = load_baseline(root,config["baseline_delivery_day"])
        inputs = MatchedInputs(prices,baseline)
        scoring,label_audit = inputs.evaluation_baseline(baseline,config["start_day"],config["end_day"])
        source_sha = digest(source)
        checkpoint = checkpoint_identity(config["model_id"])
        summary["checkpoint"] = checkpoint
        identity = _json_identity({"config":config,"code":code_hashes,"source_sha":source_sha,
                                  "baseline":baseline_audit,"checkpoint":checkpoint,"versions":versions})
        lock = safe_path(root,directory/"protocol.lock.json")
        if resume:
            protocol_lock = json.loads(lock.read_text(encoding="utf-8"))
            if protocol_lock["identity"] != identity:
                raise ValueError("Inputs, code or protocol changed; cannot resume this experiment.")
            input_manifest = safe_path(root,directory/"inputs.manifest.json")
            if digest(input_manifest) != protocol_lock["inputs_manifest_sha256"]:
                raise ValueError("Sealed input manifest changed.")
            for name,expected in json.loads(input_manifest.read_text(encoding="utf-8"))["files"].items():
                if digest(safe_path(root,directory/name)) != expected:
                    raise ValueError("Sealed input copy changed.")
        else:
            for name,frame in (("native_prices.parquet",prices),("baseline.parquet",baseline),("scoring_baseline.parquet",scoring)):
                frame.to_parquet(safe_path(root,directory/name),index=False)
            write_json(safe_path(root,directory/"inputs.manifest.json"),{"files":{name:digest(directory/name)
                       for name in ("native_prices.parquet","baseline.parquet","scoring_baseline.parquet")},
                       "source_manifest_sha256":source_sha,"baseline_audit":baseline_audit},root=root)
            write_json(lock,{"identity":identity,"config":config,"code_sha256":code_hashes,
                             "native_source_audit":native_audit,"baseline_audit":baseline_audit,
                             "checkpoint":checkpoint,"versions":versions,"inputs_manifest_sha256":digest(directory/"inputs.manifest.json"),
                             "scope":"No CatBoost/Kalman postprocessing in either matched Chronos candidate; full NYX is a separate reference."},root=root)
        lock_sha = digest(lock)
        input_manifest_sha = digest(directory/"inputs.manifest.json")
        input_hashes = json.loads((directory/"inputs.manifest.json").read_text(encoding="utf-8"))["files"]
        # Replay only the fixed evaluation days; frozen foundation weights require no target fitting.
        days = pd.date_range(config["start_day"],config["end_day"],freq="D")
        summary.update(native_source_audit=native_audit,baseline_audit=baseline_audit,
                       label_comparison=label_audit,protocol_identity=identity,
                       status="inference",total_batches=len(days)*2)
        write_json(safe_path(root,directory/"status.json"),summary,root=root)
        pipeline = None
        frames = {"h":[],"15min":[]}
        audits = []
        completed = 0
        for day in days:
            for frequency in ("h","15min"):
                civil = day.date().isoformat()
                parquet = safe_path(root,daily/f"{civil}_{frequency}.parquet")
                audit_path = safe_path(root,daily/f"{civil}_{frequency}.json")
                if parquet.exists() or audit_path.exists():
                    if not (parquet.is_file() and audit_path.is_file()):
                        raise ValueError("Incomplete daily checkpoint; do not silently reuse it.")
                    record = json.loads(audit_path.read_text(encoding="utf-8"))
                    if record["protocol_identity"] != identity or digest(parquet) != record["data_sha256"]:
                        raise ValueError("Daily checkpoint identity/checksum mismatch.")
                    predicted = pd.read_parquet(parquet)
                    audit = record["origins"]
                else:
                    contexts,futures,audit = inputs.build_origin(civil,frequency,context_hours=config["context_hours"])
                    if pipeline is None:
                        pipeline = load_pipeline(config["model_id"],torch_threads=config["torch_threads"],seed=config["seed"])
                    raw = infer_batch(contexts,futures,freq=frequency,
                                      context_length=config["context_hours"]*(4 if frequency=="15min" else 1),
                                      prediction_length=len(futures[0]),model_batch_size=config["model_batch_size"],pipeline=pipeline)
                    identifiers = {frame.item_id.iloc[0]:zone for frame,zone in zip(contexts,ZONES)}
                    if set(raw.item_id) != set(identifiers):
                        raise ValueError("Unexpected model output identifiers.")
                    predicted = pd.DataFrame({"timestamp_utc":pd.to_datetime(raw.timestamp,utc=True),
                                              "zone":raw.item_id.map(identifiers),"prediction":raw.q50.to_numpy(float)})
                    # Inference validates a complete output grid, including DST.
                    predicted.to_parquet(parquet,index=False)
                    write_json(audit_path,{"protocol_identity":identity,"data_sha256":digest(parquet),"origins":audit},root=root)
                frames[frequency].append(predicted)
                audits.extend(audit)
                completed += 1
                progress={"status":"inference","completed_batches":completed,"total_batches":len(days)*2,
                          "frequency":frequency,"delivery_day":civil,"elapsed_seconds":time.perf_counter()-began,
                          "directory":str(directory),"activation_performed":False}
                write_json(safe_path(root,directory/"status.json"),progress,root=root)
                print(json.dumps(progress),flush=True)
        hourly = pd.concat(frames["h"],ignore_index=True)
        quarters = pd.concat(frames["15min"],ignore_index=True)
        candidate = aggregate_quarterhour_predictions(quarters,start_day=config["start_day"],end_day=config["end_day"])
        quarters.to_parquet(safe_path(root,directory/"native_quarterhour_predictions.parquet"),index=False)
        pd.DataFrame(audits).to_parquet(safe_path(root,directory/"origin_audit.parquet"),index=False)
        before = min(pd.Timestamp("2026-03-14",tz="Europe/Paris").tz_convert("UTC"),
                     day_index(config["start_day"],"h")[0])
        threshold_train = baseline.loc[baseline.timestamp_utc < before]
        q95 = {zone:float(threshold_train.loc[threshold_train.zone.eq(zone),"training_actual"].quantile(.95)) for zone in ZONES}
        result = evaluate_predictions(scoring,{"native_quarterhour":candidate,"hourly_control":hourly},
                                      q95_thresholds=q95,start_day=config["start_day"],end_day=config["end_day"],
                                      bootstrap_repetitions=config["bootstrap_repetitions"],seed=config["bootstrap_seed"],
                                      matched_control_verified=True)
        for name,value in result.items():
            if isinstance(value,pd.DataFrame):
                tables[name]=value
                value.to_parquet(safe_path(root,directory/f"{name}.parquet"),index=False)
                value.to_csv(safe_path(root,directory/f"{name}.csv"),index=False)
            else:
                summary[name]=clean(value)
        if digest(source)!=source_sha:
            raise ValueError("Native source manifest changed during inference.")
        read_native_prices(source)  # Recheck the archive bytes, not only the manifest.
        for entry in baseline_audit["identities"]:
            for name,expected in {**entry["files"],**entry["observations"]}.items():
                if digest(Path(name))!=expected:
                    raise ValueError("NYX baseline changed during inference.")
        for name,expected in code_hashes.items():
            if digest(root/name)!=expected:
                raise ValueError("Experiment code changed during inference.")
        if checkpoint_identity(config["model_id"]) != checkpoint:
            raise ValueError("Model checkpoint changed during inference.")
        if digest(lock) != lock_sha or digest(directory/"inputs.manifest.json") != input_manifest_sha:
            raise ValueError("Sealed protocol or input manifest changed during inference.")
        for name,expected in input_hashes.items():
            if digest(safe_path(root,directory/name)) != expected:
                raise ValueError("Sealed input copy changed during inference.")
        summary["status"] = result.get("status","complete")
    except Exception as exc:
        summary.update(status="failed",reason=f"{type(exc).__name__}: {exc}",results_valid=False,
                       decision={"encouraging":False,"promotion_allowed":False})
        tables={}
    summary["runtime_seconds"]=time.perf_counter()-began
    write_json(safe_path(root,directory/"summary.json"),summary,root=root)
    write_json(safe_path(root,directory/"status.json"),{"status":summary["status"],"reason":summary.get("reason"),"report":str(directory/"report.html")},root=root)
    safe_path(root,directory/"report.html")
    render_report(directory,summary,tables)
    write_json(safe_path(root,directory/"outputs.manifest.json"),{"files":{str(p.relative_to(directory)):digest(p)
               for p in sorted(directory.rglob("*")) if p.is_file() and p.name!="outputs.manifest.json"}},root=root)
    write_json(safe_path(root,output/"latest.json"),{"directory":str(directory),"status":summary["status"]},root=root)
    return directory
