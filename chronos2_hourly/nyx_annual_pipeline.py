"""One daily entry point for the four annual CPU country models.

Capture public vintages before 08:00; after 08:00 collect Saturn, replay the
baseline, build features/references, and train only a qualified consumer.
Preparation and production have distinct outcomes. Every stage can resume
from its own verified cache; no missing source is silently substituted.
"""
from __future__ import annotations

from datetime import date, timedelta
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import pandas as pd

from .nyx_annual_live_preflight import ROOT, delivery_grid
from .process_lock import exclusive_process_lock

PROTOCOL = "nyx_annual_cpu_daily_pipeline_v1"
COUNTRIES = ("FR", "DE", "NL", "BE")
ARCHIVE_GROUPS = ("jao_initial", "public_hydro", "lagged_exchange")
MODULES = ("numpy", "pandas", "pyarrow", "catboost", "sklearn", "torch", "chronos",
           "huggingface_hub", "httpx", "requests", "tshistory_lite", "holidays", "pykalman")


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        deadline = time.monotonic() + 5
        while True:
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.05)
    finally:
        temporary.unlink(missing_ok=True)


def paths(day, bundle=None, output=None):
    delivery_grid(day)
    return (Path(bundle or ROOT / "runs/live/nyx_annual_cpu" / day).resolve(),
            Path(output or ROOT / "runs/nyx_annual_cpu_live" / day).resolve())


def inspect(day, *, bundle=None, output=None):
    """Read-only installation, source and activation status; never sync or fit."""
    from .nyx_annual_cpu_live import preflight, verify_activation
    bundle, output = paths(day, bundle, output)
    missing = [name for name in MODULES if importlib.util.find_spec(name) is None]
    _, _, cutoff = delivery_grid(day)
    now = pd.Timestamp.now(tz="UTC")
    report = preflight(bundle, day, output)
    activation_error = None
    try:
        verify_activation()
    except (ValueError, OSError, KeyError, TypeError) as error:
        activation_error = str(error)
    source_groups = ("saturn", "auction_prices", "fuel", "thermal_capacity", "jao_initial", "public_hydro", "lagged_exchange")
    sources = {}
    for group in source_groups:
        path = bundle / "source_receipts" / (group + ".json")
        if path.is_file():
            try:
                receipt = json.loads(path.read_text(encoding="utf-8"))
                sources[group] = receipt.get("state", "INVALID")
            except (OSError, ValueError):
                sources[group] = "INVALID"
        else:
            sources[group] = "MISSING"
    return {"protocol": PROTOCOL, "delivery_day": day, "countries": list(COUNTRIES),
        "bundle": str(bundle), "output": str(output), "ready": report["ready"],
        "can_prepare": not missing and now >= cutoff,
        "can_request_forecast": not missing and now >= cutoff and activation_error is None,
        "can_capture": (now.tz_convert("Europe/Paris").date() + timedelta(days=1)).isoformat() == day
                       and cutoff - pd.Timedelta(hours=6, minutes=45) <= now < cutoff,
        "missing_modules": missing, "activation_error": activation_error,
        "cutoff_utc": cutoff.isoformat(), "source_states": sources,
        "blockers": report["blockers"], "consumer": report}


def source_commands(day, bundle, cache_root=None):
    last = date.fromisoformat(day)
    start = (last - timedelta(days=469)).isoformat()
    base = ["--delivery-day", day, "--bundle", str(bundle)]
    # Saturn profiles have their own stable 834-day initial bootstrap anchor.
    # The longer auction/fuel history is used by prior90 and HGB OOF warmup.
    fuel_cache = Path(cache_root) / "fuel" if cache_root else ROOT / "data/pit/nyx_annual_cpu_live_fuel"
    saturn_args = ["--cache", str(Path(cache_root) / "saturn")] if cache_root else []
    thermal_args = ["--cache", str(Path(cache_root) / "thermal")] if cache_root else []
    audit_path = fuel_cache / "market_fuel_features.parquet.audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        existing_start = audit.get("start_day")
        if isinstance(existing_start, str) and date.fromisoformat(existing_start) <= date.fromisoformat(start):
            start = existing_start
    return [
        ("saturn", "run_nyx_annual_saturn_source.py", [*base, *saturn_args]),
        ("auction_prices", "run_nyx_annual_auction_prices_source.py", [*base, "--history-start-day", start]),
        ("fuel", "run_nyx_annual_fuel_source.py", [*base, "--history-start-day", start, "--cache-dir", str(fuel_cache)]),
        ("thermal_capacity", "run_nyx_annual_thermal_source.py", [*base, *thermal_args]),
        ("jao_initial", "run_nyx_annual_jao_source.py", [*base, "--verify-only"]),
        ("public_hydro", "run_nyx_annual_hydro_source.py", [*base, "--action", "assemble"]),
        ("lagged_exchange", "run_nyx_annual_exchange_source.py", [*base, "--action", "assemble"]),
    ]


def validate_prepared_bundle(bundle, day):
    """Revalidate a sealed day without touching newer mutable download caches."""
    from .nyx_annual_live_preflight import inspect_bundle
    from .nyx_annual_source_validation import validate_source_packet
    from .nyx_annual_cpu_baseline import validate_cpu_baseline_evidence
    from .nyx_annual_cpu_reference_builder import validate_cpu_reference_source
    report = inspect_bundle(bundle, day)
    if report.get("input_bundle_valid") is not True:
        failures = [str(item.get("reason", item)) for item in report["checks"] if not item.get("passed")]
        raise ValueError("Sealed input bundle differs: " + "; ".join(failures[:4]))
    sources = validate_source_packet(bundle, day)
    if sources.get("passed") is not True or sources.get("source_snapshot_asof_verified") is not True:
        raise ValueError("Raw source evidence does not verify")
    validate_cpu_baseline_evidence(bundle, day)
    validate_cpu_reference_source(bundle, day)


def run(day, *, action="forecast", bundle=None, output=None, runner=None, cache_root=None):
    if action not in ("capture", "prepare", "forecast"):
        raise ValueError("Expected capture, prepare or forecast")
    bundle, output = paths(day, bundle, output)
    # Keep logs beside, not inside, the consumer output; it must start empty.
    status_path = output.parent / (output.name + ".pipeline.json")
    state = {"protocol": PROTOCOL, "action": action, "delivery_day": day,
             "countries": list(COUNTRIES), "bundle": str(bundle), "output": str(output),
             "state": "RUNNING", "stages": [], "forecast_published": False}
    def publish():
        state["updated_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        _write(status_path, state)
    def execute(script, args):
        command = [sys.executable, "-u", str(ROOT / script), *args]
        print(json.dumps({"stage_command": command}, ensure_ascii=False), flush=True)
        return subprocess.run(command, cwd=ROOT, shell=False, stdin=subprocess.DEVNULL,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).returncode
    runner = runner or execute
    def step(name, callback):
        item = {"name": name, "state": "RUNNING"}
        state["stages"].append(item)
        publish()
        try:
            callback()
            item["state"] = "COMPLETE"
        except Exception as error:
            item.update(state="ERROR", error=f"{type(error).__name__}: {error}")
        publish()
        print(json.dumps(item, ensure_ascii=False), flush=True)
        return item["state"] == "COMPLETE"
    def cli(script, args):
        code = runner(script, args)
        if code != 0:
            raise RuntimeError(f"{script} exited with {code}; see preceding source error")
    def blocked():
        state["state"] = "BLOCKED"
        publish()
        return state
    with exclusive_process_lock(ROOT / "runs/live/nyx_annual_cpu/pipeline.lock"):
        publish()
        if action == "capture":
            ok = step("capture_public_sources", lambda: cli("run_nyx_annual_daily_capture.py", ["--delivery-day", day]))
            state["state"] = "CAPTURED" if ok else "BLOCKED"
            publish()
            return state
        _, _, cutoff = delivery_grid(day)
        if pd.Timestamp.now(tz="UTC") < cutoff:
            state["stages"].append({"name": "cutoff", "state": "ERROR", "error": "Wait until D-1 08:00 Paris; capture public sources before then"})
            return blocked()
        if action == "forecast":
            from .nyx_annual_cpu_live import verify_activation
            if not step("qualification", verify_activation):
                return blocked()
        from .nyx_annual_live_preflight import MATERIALIZATION_PATH
        if (bundle / MATERIALIZATION_PATH).is_file():
            if not step("verified_existing_bundle", lambda: validate_prepared_bundle(bundle, day)):
                return blocked()
        else:
            commands = source_commands(day, bundle, cache_root)
            # These are local, strict archive validators/assemblers. Check all
            # three before the 834-day Saturn bootstrap or any model fitting.
            # Capturing tomorrow's public data remains a separate action.
            phases = ([command for command in commands if command[0] in ARCHIVE_GROUPS],
                      [command for command in commands if command[0] not in ARCHIVE_GROUPS])
            for phase in phases:
                for name, script, args in phase:
                    step(name, lambda script=script, args=args: cli(script, args))
                if any(item["state"] == "ERROR" for item in state["stages"]):
                    return blocked()
            from .nyx_annual_cpu_baseline import build_from_bundle as baseline
            from .nyx_annual_cpu_bundle_builder import materialize_features, seal_bundle
            from .nyx_annual_cpu_reference_builder import build_from_bundle as reference
            for name, callback in (
                ("baseline_cpu", lambda: baseline(bundle, day, progress=lambda p: print(json.dumps(p), flush=True))),
                ("feature_matrices", lambda: materialize_features(bundle, day)),
                ("reference_cpu", lambda: reference(bundle, day)),
                ("seal_bundle", lambda: seal_bundle(bundle, day)),
            ):
                if not step(name, callback):
                    return blocked()
        if action == "prepare":
            state["state"] = "PREPARED"
            publish()
            return state
        from run_nyx_annual_cpu_live import run as forecast
        if not step("qualified_forecast", lambda: forecast(bundle=bundle, delivery_day=day, output=output)):
            return blocked()
        state.update(state="COMPLETE", forecast_published=True)
        publish()
        return state
