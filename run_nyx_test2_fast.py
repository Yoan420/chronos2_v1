"""Accelerated isolated NYX/Test2 shadow run; frozen sources and unchanged recipes."""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
import sys
import time
import traceback
import uuid

# This process and its descendants must not compete with the production pool.
for _option in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_option] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import pandas as pd
import psutil

from chronos2_hourly.nyx_fast_sources import OUTPUT, safe, atomic_json, ensure_live_sources
from chronos2_hourly.process_lock import exclusive_process_lock

ROOT = Path(__file__).resolve().parent
ENGINE = "nyx_test2_fast_v1"
CHRONOS_THREADS = 16
KALMAN_WORKERS = 4
RESIDUAL_THREADS = 2
RESIDUAL_WORKERS = 4
ORIGINAL_PLAN_SHA256 = "9ff4b0a1c979fa8fc7b802870c03ab02e9a1d8662e1edfc86586b8d90edb993a"
ORIGINAL_ROOT = ROOT / "runs/experiments/nyx_test2_live_v1/2026-09-24"
REFERENCE = ROOT / "runs/model_references/nyx_test2_hybrid/3a08985a6007d5b8.json"
CODE = ("run_nyx_test2_fast.py", "chronos2_hourly/nyx_fast_sources.py",
        "chronos2_hourly/nyx_fast_baseline.py", "chronos2_hourly/nyx_live_parallel.py",
        "run_nyx_test2_live.py", "chronos2_hourly/nyx_live_sources.py",
        "chronos2_hourly/nyx_live_baseline.py", "chronos2_hourly/nyx_live_hybrid.py",
        "chronos2_hourly/nyx_live_reporting.py", "chronos2_hourly/solar_wind_scarcity_hybrid.py",
        "chronos2_hourly/solar_wind_scarcity_ablation.py", "chronos2_hourly/solar_wind_scarcity_regime.py",
        "chronos2_hourly/process_lock.py", "chronos2_modular/report.py",
        "chronos2_hourly/solar_cwe_sources.py", "materialize_saturn_daily_asof.py",
        "materialize_saturn_kalman_weather.py", "chronos2_hourly/reporting.py",
        "chronos2_hourly/nuclear_daily_cache.py", "chronos2_hourly/nuclear_residual_cache.py",
        "chronos2_hourly/nuclear_run_archive.py", "chronos2_hourly/nuclear_exports.py",
        "chronos2_hourly/nuclear_cwe_forecast.py", "chronos2_modular/common.py",
        "chronos2_modular/metrics.py", "chronos2_hourly/hourly_contract.py")
PAIR_ORDER = (("DE", "NL"), ("BE", "FR"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str).encode()



def frozen_source_manifest():
    """Read-only evidence for the unchanged six curves already captured."""
    plan_path = ORIGINAL_ROOT / "plan.json"
    if sha(plan_path) != ORIGINAL_PLAN_SHA256:
        raise ValueError("Original immutable plan changed")
    receipt_path = ORIGINAL_ROOT / "sources/receipt_denl.json"
    receipt = read(receipt_path)
    if receipt.get("delivery_day") != "2026-09-24":
        raise ValueError("Original source receipt has a different delivery day")
    expected_aliases = {
        "de_solar_generation_fcst", "nl_solar_generation_fcst",
        "be_solar_generation_fcst", "fr_solar_generation_fcst",
        "de_wind_generation_fcst", "nl_wind_generation_fcst",
    }
    if set(receipt["sources"]) != expected_aliases:
        raise ValueError("Original six-source inventory differs")
    files = {}
    directory = ORIGINAL_ROOT / "sources"
    if directory.resolve() != directory:
        raise ValueError("Original source directory is redirected")
    for alias in sorted(expected_aliases):
        record = receipt["sources"][alias]
        for suffix, digest_key in ((".parquet", "sha256"), (".parquet.audit.json", "audit_sha256")):
            path = directory / (alias + suffix)
            if path.resolve() != path or not path.is_file() or sha(path) != record[digest_key]:
                raise ValueError(f"Original immutable source changed: {alias}{suffix}")
            files[path.name] = {"path": str(path), "sha256": record[digest_key]}
    return {"original_plan_sha256": ORIGINAL_PLAN_SHA256, "receipt_path": str(receipt_path),
            "receipt_sha256": sha(receipt_path), "files": files}


def clone_captured_sources(output, expected_manifest):
    """Copy only the previously frozen source bytes, never refresh DE/NL."""
    output = safe(output)
    if frozen_source_manifest() != expected_manifest:
        raise ValueError("Frozen source identity changed before copy")
    output.mkdir(parents=True, exist_ok=True)
    for name, record in expected_manifest["files"].items():
        source, destination = Path(record["path"]), safe(output / name)
        if not destination.exists():
            temporary = safe(destination.with_name(f".{name}.{uuid.uuid4().hex}.copy.tmp"))
            shutil.copyfile(source, temporary)
            if sha(temporary) != record["sha256"]:
                raise ValueError("Incomplete immutable source copy")
            temporary.replace(destination)
        if sha(destination) != record["sha256"] or sha(source) != record["sha256"]:
            raise ValueError(f"Frozen source copy mismatch: {name}")
    if frozen_source_manifest() != expected_manifest:
        raise ValueError("Frozen source identity changed during copy")


def verify_captured_copies(output, expected_manifest):
    """The BE/FR extension may add wind, but cannot refresh six frozen curves."""
    output = safe(output)
    for name, record in expected_manifest["files"].items():
        if sha(safe(output / name)) != record["sha256"]:
            raise ValueError(f"A frozen copied curve changed: {name}")


def verify_reference():
    reference = read(REFERENCE)
    directory = Path(reference["workdir"])
    if reference["reference_identity"] != "3a08985a6007d5b8" or sha(directory / "completion.json") != reference["completion_sha256"]:
        raise ValueError("Chosen reference completion changed")
    completion = read(directory / "completion.json")
    if completion["status"] != "COMPLETE" or len(completion["files"]) != 52:
        raise ValueError("Chosen reference is not fully complete")
    for relative, digest in completion["files"].items():
        path = (directory / relative).absolute()
        if path.resolve() != path or not path.is_relative_to(directory) or sha(path) != digest:
            raise ValueError(f"Chosen reference artifact changed: {relative}")
    for relative, digest in reference["code_sha256"].items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f"Chosen scientific reference code changed: {relative}")
    if sha(ROOT / reference["protocol_path"]) != reference["protocol_sha256"]:
        raise ValueError("Chosen scientific protocol changed")
    return reference


def build_plan(day):
    from run_solar_wind_forecast import scientific_identity
    parsed = pd.Timestamp(day)
    if day != "2026-09-24":
        raise ValueError("Accelerated continuation is fixed to 2026-09-24")
    if pd.isna(parsed) or parsed.tzinfo is not None or parsed != parsed.normalize() or str(parsed.date()) != day:
        raise ValueError("Exact YYYY-MM-DD delivery required")
    verify_reference()
    return {"schema_version": 1, "engine": ENGINE, "delivery_day": day,
        "reference_identity": "3a08985a6007d5b8", "reference_sha256": sha(REFERENCE),
        "source_migration": frozen_source_manifest(),
        "predecessor_engine": "nyx_test2_live_v1", "predecessor_plan_sha256": ORIGINAL_PLAN_SHA256,
        "code": {p: sha(ROOT / p) for p in CODE}, "pairs": [list(p) for p in PAIR_ORDER],
        "scientific_baseline": scientific_identity(),
        "runtime_dependencies": {p: importlib.metadata.version(p) for p in ("pykalman", "scipy", "joblib", "holidays", "scikit-learn")},
        "zone_order": ["DE", "NL", "BE", "FR"], "source_start_day": "2024-09-09",
        "threads": CHRONOS_THREADS, "workers": KALMAN_WORKERS, "device": "cpu", "priority": "Normal",
        "residual_threads": RESIDUAL_THREADS, "residual_workers": RESIDUAL_WORKERS,
        "test2_threads": 1, "execution_change_only": True,
        "source_policy": "six frozen captured curves; previously authorized BE/FR suffix only",
        "min_free_memory_gib": 3.5, "source_suffix_limit_days": 31,
        "test2_oof_days": 184, "gate_past_complete_days": 90, "hybrid_evaluation_days": 93,
        "test2_iterations": 120, "test2_seed": 20260923,
        "production_modified": False, "automatic_promotion": False,
        "historical_pit_evidence": "query_asof_cutoff_only"}


def save_plan(day):
    plan = build_plan(day)
    path = safe(OUTPUT / day / "plan.json")
    if path.exists():
        if read(path) != plan:
            raise ValueError("A different plan exists; refusing overwrite")
    else:
        atomic_json(path, plan)
    return path, sha(path)


def verify_plan(day, expected):
    path = safe(OUTPUT / day / "plan.json")
    if not expected or sha(path) != expected:
        raise ValueError("Expected immutable plan SHA required")
    plan = read(path)
    if plan != build_plan(day):
        raise ValueError("Code, reference or plan changed; no automatic migration")
    return plan


def write_parquet(path, frame):
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.parquet")
    frame.to_parquet(temporary, index=True)
    temporary.replace(path)


def indexed(frame):
    frame = frame.copy()
    keys = [c for c in ("timestamp", "timestamp_utc", "delivery_start_utc") if c in frame]
    if keys:
        index = pd.DatetimeIndex(pd.to_datetime(frame.pop(keys[0]), utc=True))
    else:
        index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("Aware physical timestamps required")
    frame.index = index.tz_convert("UTC")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Ordered unique hourly series required")
    return frame


def verify_completed_pair(work, identity):
    """Read-only final inventory, SHA, physical grids and exact routing audit."""
    import numpy as np
    work = safe(work)
    saved = read(work / "completion.json")
    if saved.get("identity") != identity or saved.get("status") != "COMPLETE":
        raise ValueError("Pair completion identity mismatch")
    observed = {p.relative_to(work).as_posix() for p in work.rglob("*") if p.is_file()}
    if observed != set(saved["files"]) | {"completion.json"}:
        raise ValueError("Completed pair inventory differs from exact sealed inventory")
    for relative, digest in saved["files"].items():
        if sha(safe(work / relative)) != digest:
            raise ValueError("Pair completed artifact changed")
    day = pd.Timestamp(identity["delivery_day"])
    for label, start, stop in (("historical_hybrid", day - pd.Timedelta(days=93), day),
                               ("forecast_hybrid", day, day + pd.Timedelta(days=1))):
        frame = pd.read_parquet(work / f"{label}.parquet")
        if set(frame.zone) != set(identity["pair"]):
            raise ValueError("Completed pair country set differs")
        grid = pd.date_range(start.tz_localize("Europe/Berlin"), stop.tz_localize("Europe/Berlin"),
                             freq="h", inclusive="left").tz_convert("UTC")
        for zone in identity["pair"]:
            sub = frame.loc[frame.zone.eq(zone)]
            if not sub.index.equals(grid) or not pd.api.types.is_bool_dtype(sub.selected_test2.dtype) or sub.selected_test2.isna().any():
                raise ValueError("Completed physical hourly grid or route is invalid")
            for prefix in ("nyx", "test2", "hybrid"):
                array = sub[[f"{prefix}__{q}" for q in ("q10", "q50", "q90")]].to_numpy(float)
                if not np.isfinite(array).all() or (np.diff(array, axis=1) < 0).any():
                    raise ValueError("Completed quantiles nonfinite or crossed")
            for q in ("q10", "q50", "q90"):
                if not np.array_equal(sub[f"hybrid__{q}"], np.where(sub.selected_test2, sub[f"test2__{q}"], sub[f"nyx__{q}"])):
                    raise ValueError("Completed hybrid is not an exact source-triplet switch")
            if label == "historical_hybrid":
                if not np.isfinite(sub.actual.to_numpy(float)).all():
                    raise ValueError("Historical observations missing")
            elif "actual" in sub and sub.actual.notna().any():
                raise ValueError("Future actuals entered the completed forecast")
    return saved


class Run:
    def __init__(self, day, expected):
        self.day, self.expected = day, expected
        self.plan = verify_plan(day, expected)
        self.root = safe(OUTPUT / day)
        self.started = time.time()
        self.state = {"engine": self.plan["engine"], "delivery_day": day,
            "started_utc": pd.Timestamp.now(tz="UTC").isoformat(), "plan_sha256": expected,
            "pid": os.getpid(), "process_create_time": psutil.Process().create_time(),
            "command": [sys.executable, *sys.argv], "threads": CHRONOS_THREADS, "workers": KALMAN_WORKERS,
            "residual_threads": RESIDUAL_THREADS, "residual_workers": RESIDUAL_WORKERS,
            "production_modified": False, "completed_zones": [], "completed_pairs": [],
            "baseline_results": {}, "reports": {}, "eta_seconds": None,
            "eta_basis": "unavailable_until_measured_new_baseline_replay"}

    def status(self, phase, **extra):
        self.state.update({"phase": phase, "status": "RUNNING", "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(), **extra})
        atomic_json(self.root / "status.json", self.state)
        print(f"[NYX fast] {phase}: {json.dumps(extra, default=str, ensure_ascii=False)}", flush=True)

    def resource_guard(self):
        while psutil.virtual_memory().available < self.plan["min_free_memory_gib"] * 1024**3:
            verify_plan(self.day, self.expected)
            self.status("waiting_memory", available_gib=round(psutil.virtual_memory().available / 1024**3, 2))
            time.sleep(20)

    def donor(self, zone):
        from chronos2_hourly.nuclear_run_archive import load_nuclear_result_bundle
        path = ROOT / "runs/experiments/nuclear_forecast_v1" / self.day / zone.lower() / "civil_pit_v2"
        deadline = self.started + 24 * 3600
        while not (path / "report_only/frozen_result/manifest.json").is_file():
            verify_plan(self.day, self.expected)
            if time.time() > deadline:
                raise RuntimeError(f"Missing sealed production donor after24h: {zone}; user review required")
            self.status("waiting_sealed_parent", zone=zone, donor=str(path))
            time.sleep(30)
        # Never wait through a checksum failure: an existing invalid bundle is a hard error.
        load_nuclear_result_bundle(workdir=path)
        return path

    def pair(self, pair, baselines):
        from chronos2_hourly.nyx_live_hybrid import run_pair_pipeline
        from chronos2_hourly.nyx_live_reporting import render_live_report
        name = "_".join(pair)
        histories, forecasts, covariates = {}, {}, {}
        hashes = {}
        for zone in pair:
            entry = baselines[zone]
            for key, target in (("history_path", histories), ("forecast_path", forecasts), ("hybrid_covariates_path", covariates)):
                path = Path(entry[key])
                target[zone] = indexed(pd.read_parquet(path))
                hashes[f"{zone}:{key}"] = {"path": str(path), "sha256": sha(path)}
            forecasts[zone] = forecasts[zone].drop(columns=["actual"], errors="ignore")
        identity = {"pair": list(pair), "plan_sha256": self.expected, "inputs": hashes,
                    "delivery_day": self.day, "reference": self.plan["reference_identity"]}
        key = hashlib.sha256(encoded(identity)).hexdigest()[:16]
        work = safe(self.root / "pairs" / name / key)
        complete = work / "completion.json"
        if complete.exists():
            verify_completed_pair(work, identity)
            return {z: str(work / f"forecast_{z.lower()}_{self.day}_nyx_test2.html") for z in pair}
        atomic_json(work / "identity.json", identity)

        def load(stage, checkpoint_key):
            location = safe(work / "checkpoints" / stage / checkpoint_key)
            receipt = location.with_suffix(".json")
            if not receipt.exists():
                return None
            checkpoint = read(receipt)
            frame_path = location.with_suffix(".parquet")
            if checkpoint["identity"] != key or checkpoint["sha256"] != sha(frame_path):
                raise ValueError("Hybrid checkpoint identity/SHA changed")
            return pd.read_parquet(frame_path), checkpoint["audit"]

        def save(stage, checkpoint_key, panel, audit):
            verify_plan(self.day, self.expected)
            self.resource_guard()
            location = safe(work / "checkpoints" / stage / checkpoint_key)
            frame_path = location.with_suffix(".parquet")
            write_parquet(frame_path, panel)
            atomic_json(location.with_suffix(".json"), {"identity": key, "sha256": sha(frame_path), "audit": audit})
            self.status(f"{stage}_checkpoint", pair=name, origin=checkpoint_key, pair_workdir=str(work))

        self.status("training_test2_and_routes", pair=name, pair_workdir=str(work))
        self.resource_guard()
        result = run_pair_pipeline(histories, forecasts, covariates, pair=pair, delivery_day=self.day,
            threads=1, iterations=120, seed=20260923, load_checkpoint=load, save_checkpoint=save)
        for field in ("historical_test2", "historical_hybrid", "forecast_test2", "forecast_hybrid"):
            write_parquet(work / f"{field}.parquet", result[field])
        for field in ("feature_audit", "fit_audits", "policies", "protocol"):
            atomic_json(work / f"{field}.json", result[field])
        reports = {}
        for zone in pair:
            history = result["historical_hybrid"].loc[lambda f: f.zone.eq(zone)].copy()
            forecast = result["forecast_hybrid"].loc[lambda f: f.zone.eq(zone)].copy()
            path = work / f"forecast_{zone.lower()}_{self.day}_nyx_test2.html"
            if path.exists():
                # An unsealed existing report cannot be trusted or silently overwritten.
                raise ValueError(f"Unsealed existing report requires review: {path}")
            render_live_report(history, forecast, zone=zone, delivery_day=self.day, output_path=path,
                metadata={"reference_identity": self.plan["reference_identity"], "pair": list(pair),
                    "extension_befr": pair == ("BE", "FR"), "plan_sha256": self.expected,
                    "source_cutoff": "D-1 08:00 civil", "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    "oof_test2_days": 184, "gate_warmup_complete_days": 90, "protocol": result["protocol"],
                    "production_modified": False, "historical_pit_evidence": "query_asof_cutoff_only"})
            reports[zone] = str(path)
        verify_plan(self.day, self.expected)
        for item in hashes.values():
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("Pair input changed while training")
        artifacts = {str(p.relative_to(work)).replace("\\", "/"): sha(p) for p in work.rglob("*") if p.is_file()}
        atomic_json(complete, {"status": "COMPLETE", "identity": identity, "files": artifacts,
            "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(), "reports": reports, "production_modified": False})
        verify_completed_pair(work, identity)
        return reports

    def execute(self):
        from chronos2_hourly.nyx_fast_baseline import run_zone
        from chronos2_hourly.nyx_live_hybrid import build_pair_interaction
        if os.name == "nt":
            psutil.Process().nice(psutil.NORMAL_PRIORITY_CLASS)
        self.status("copy_frozen_sources")
        self.resource_guard()
        clone_captured_sources(self.root / "sources", self.plan["source_migration"])
        self.status("validate_captured_sources")
        sources = ensure_live_sources(self.root / "sources", start_day=self.plan["source_start_day"],
                                      delivery_day=self.day, include_befr=False)
        verify_captured_copies(self.root / "sources", self.plan["source_migration"])
        verify_plan(self.day, self.expected)
        for pair in PAIR_ORDER:
            if pair == ("BE", "FR"):
                self.status("capture_befr_extension_sources")
                self.resource_guard()
                sources = ensure_live_sources(self.root / "sources", start_day=self.plan["source_start_day"],
                                              delivery_day=self.day, include_befr=True)
                verify_captured_copies(self.root / "sources", self.plan["source_migration"])
                verify_plan(self.day, self.expected)
            for zone in pair:
                donor = self.donor(zone)
                self.resource_guard()
                verify_plan(self.day, self.expected)
                self.status("baseline_replay", zone=zone)
                result = run_zone(zone, self.day, donor, sources, workroot=OUTPUT,
                    threads=CHRONOS_THREADS, workers=KALMAN_WORKERS, device="cpu", action="run",
                    interaction_builder=build_pair_interaction,
                    progress=lambda event: self.status("baseline_progress", zone=zone, baseline_progress=event))
                self.state["baseline_results"][zone] = result
                self.state["completed_zones"].append(zone)
                self.status("baseline_complete", zone=zone)
            self.state["reports"].update(self.pair(pair, self.state["baseline_results"]))
            self.state["completed_pairs"].append(list(pair))
            self.status("pair_complete", pair="_".join(pair))
            links = "\n".join(f'<li><a href="{html.escape(Path(p).relative_to(self.root).as_posix())}">{z} — {self.day}</a></li>' for z, p in self.state["reports"].items())
            index = safe(self.root / "index.html")
            temporary = index.with_name(".index.html.tmp")
            temporary.write_text(f'<!doctype html><meta charset="utf-8"><title>NYX / Test2 {self.day}</title><h1>NYX / Test2 — {self.day}</h1><p>Candidat isolé accéléré, production inchangée. BE/FR : extension distincte.</p><ul>{links}</ul>', encoding="utf-8")
            temporary.replace(index)
        self.status("complete", status="COMPLETE", eta_seconds=0., eta_basis="complete")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "validate", "verify-results", "run", "status"), required=True)
    parser.add_argument("--delivery-day", required=True)
    parser.add_argument("--expected-plan")
    args = parser.parse_args(argv)
    if args.action == "prepare":
        path, digest = save_plan(args.delivery_day)
        print(json.dumps({"plan": str(path), "sha256": digest}))
        return 0
    if args.action == "status":
        print(json.dumps(read(safe(OUTPUT / args.delivery_day / "status.json")), ensure_ascii=False))
        return 0
    plan = verify_plan(args.delivery_day, args.expected_plan)
    if args.action == "validate":
        print(json.dumps({"status": "VALID", "plan": plan}))
        return 0
    if args.action == "verify-results":
        state = read(safe(OUTPUT / args.delivery_day / "status.json"))
        if state.get("status") != "COMPLETE" or set(state.get("reports", {})) != {"DE", "NL", "BE", "FR"}:
            raise ValueError("Four-country chain incomplete")
        parents = {Path(p).parent for p in state["reports"].values()}
        for work in parents:
            verify_completed_pair(work, read(work / "identity.json"))
        print(json.dumps({"status": "COMPLETE_VERIFIED", "reports": state["reports"]}))
        return 0
    with exclusive_process_lock(safe(OUTPUT / args.delivery_day / "run.lock")):
        run = Run(args.delivery_day, args.expected_plan)
        try:
            run.execute()
        except Exception as exc:
            run.status("error", status="FAILED", error_type=type(exc).__name__, error=str(exc),
                       requires_user_review=True, automatic_restart=False)
            traceback.print_exc()
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


