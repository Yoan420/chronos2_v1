"""Bounded, isolated Chronos CPU probe; never writes to source experiment paths."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for option in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[option] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workdir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", nargs="+", type=int, default=[8, 4, 2])
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    import psutil
    import torch
    import yaml
    from chronos2_modular.common import build_zone_configs, deep_get, set_reproducibility
    from chronos2_modular.forecasting import load_model
    from chronos2_hourly.nuclear_preparation import prepare_nuclear_zone_data
    from chronos2_hourly.nuclear_forecast import _digest_frame
    from run_chronos2_hourly import _feature_inputs
    from chronos2_hourly.chronos_adapter import (
        execute_grouped_chronos_backtest, generate_delivery_plans,
        make_existing_forecasting_executor,
    )

    source = Path(args.source_workdir).resolve(strict=True)
    allowed_source = ROOT / "runs/experiments/nyx_test2_live_v1/2026-09-24/de"
    if not source.is_relative_to(allowed_source):
        raise ValueError("Probe source must be the isolated DE September24 workdir")
    output = Path(args.output).absolute()
    allowed_output = ROOT / "runs/monitoring/nyx_test2_acceleration_20260923/probe"
    if output.resolve() != output or not output.is_relative_to(allowed_output):
        raise ValueError("Output must be a fresh isolated monitoring probe directory")
    if output.exists():
        raise ValueError("Refusing to overwrite an earlier probe")
    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or physical
    if not args.threads or len(args.threads) > 3 or any(t not in (2, 4, 8, 16) or t > logical for t in args.threads):
        raise ValueError("At most three candidates, each2/4/8/16 and no more than logical CPUs")
    if len(set(args.threads)) != len(args.threads):
        raise ValueError("Duplicate thread candidates")
    snapshot_path = source / "input_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    config_path = source / "resolved_config.yaml"
    expected = {str(config_path): snapshot["resolved_config_sha256"], str(snapshot_path): sha(snapshot_path)}
    for record in snapshot["files"]:
        path = Path(record["snapshot_path"]).resolve(strict=True)
        if not path.is_relative_to(source):
            raise ValueError("Snapshot file escaped source workdir")
        expected[str(path)] = record["sha256"]
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError(f"Source changed before probe: {path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["nuclear_experiment"].get("raw_history_start_day") is None:
        raise ValueError("Explicit frozen raw history start is required; no epoch creation allowed")
    output.mkdir(parents=True)
    started = time.perf_counter()
    spec = build_zone_configs(config, ["DE"], None, None)[0]
    data = prepare_nuclear_zone_data(spec, config, source, output / "prepared")
    manifest = json.loads((source / "checkpoints/nuclear_chronos_oof.csv.gz.manifest.json").read_text())
    replay_target, _, _, _ = _feature_inputs(data, config)
    observed_hashes = {"target": _digest_frame(replay_target),
                       "model_context_covariates": _digest_frame(data.model_context_covariates),
                       "covariates": _digest_frame(data.covariates)}
    for name, digest in observed_hashes.items():
        if digest != manifest["source_hashes"][name]:
            raise ValueError(f"Prepared probe input differs from original replay: {name}")
    # Two adjacent complete civil days, same batch settings and local model as run.
    plans = generate_delivery_plans("2026-09-21", "2026-09-22", forecast_origin_local_time="08:00", timezone=data.timezone)
    seed = int(deep_get(config, "model.seed", 42))
    set_reproducibility(seed)
    torch.set_num_threads(args.threads[0])
    runtime = load_model(config, "cpu", True)
    executor = make_existing_forecasting_executor(
        data=data, runtime=runtime,
        context_length=int(deep_get(config, "model.context_length", 2048)),
        origin_batch_size=int(deep_get(config, "model.origin_batch_size", 12)),
        model_batch_size=int(deep_get(config, "model.model_batch_size", 128)),
        with_covariates=True, variant="nuclear_input_experiment_oof",
    )
    expected_index = plans[0].delivery_index_utc.append(plans[1].delivery_index_utc)
    result = {"schema_version": 1, "source_workdir": str(source), "source_hashes": observed_hashes,
              "pid": os.getpid(), "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
              "physical_cores": physical, "logical_cores": psutil.cpu_count(),
              "history_origins": [str(p.delivery_date) for p in plans],
              "origin_batch_size": int(deep_get(config, "model.origin_batch_size", 12)),
              "model_batch_size": int(deep_get(config, "model.model_batch_size", 128)),
              "production_modified": False, "candidates": []}
    process = psutil.Process()
    reference = None
    for count in args.threads:
        if time.perf_counter() - started > 170:
            result["budget_exhausted_before_candidate"] = count
            break
        if psutil.virtual_memory().available < 3.5 * 1024**3:
            raise RuntimeError("Free memory reserve below3.5GiB")
        torch.set_num_threads(count)
        set_reproducibility(seed)
        execute_grouped_chronos_backtest(plans[:1], executor)  # One unmeasured warmup per candidate.
        stats = {"peak_rss": process.memory_info().rss, "min_available": psutil.virtual_memory().available}
        stop = threading.Event()
        def sample():
            while not stop.wait(.1):
                stats["peak_rss"] = max(stats["peak_rss"], process.memory_info().rss)
                stats["min_available"] = min(stats["min_available"], psutil.virtual_memory().available)
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        set_reproducibility(seed)
        begin = time.perf_counter()
        try:
            panel = execute_grouped_chronos_backtest(plans, executor)
        finally:
            seconds = time.perf_counter() - begin
            stop.set()
            sampler.join()
        values = panel[["q10", "q50", "q90"]].to_numpy(dtype=float)
        if not panel.index.equals(expected_index) or len(panel) != 48:
            raise ValueError("Unexpected probe delivery-hour grid")
        if not np.isfinite(values).all() or np.any(np.diff(values, axis=1) < 0):
            raise ValueError("Probe quantiles nonfinite/unordered")
        if reference is None:
            reference = values.copy()
        record = {"threads": count, "wall_seconds": seconds, "seconds_per_day": seconds / 2,
                  "peak_rss_gib": stats["peak_rss"] / 1024**3,
                  "min_available_gib": stats["min_available"] / 1024**3,
                  "max_abs_difference_vs_first": float(np.max(np.abs(values-reference))),
                  "hours": len(panel), "finite_ordered": True}
        panel.to_parquet(output / f"probe_threads_{count}.parquet")
        result["candidates"].append(record)
        print(json.dumps(record), flush=True)
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError(f"Source changed during probe: {path}")
    result.update(source_snapshot_unchanged=True, elapsed_seconds=time.perf_counter()-started,
                  completed_utc=pd.Timestamp.now(tz="UTC").isoformat(),
                  recommended_threads=min(result["candidates"], key=lambda r:r["wall_seconds"])["threads"])
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(output / "result.json"), "recommended_threads": result["recommended_threads"]}), flush=True)


if __name__ == "__main__":
    main()
