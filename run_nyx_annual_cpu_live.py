#!/usr/bin/env python
"""Consume a qualified daily NYX annual CPU bundle; never collect sources."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import traceback

import pandas as pd

from chronos2_hourly.nyx_annual_cpu_live import (
    COUNTRIES, MANIFEST, PROTOCOL, QUALIFICATION_CODE, ROOT, bundle_hashes,
    execute_models, load_bundle, preflight, require, verify_activation,
)
from chronos2_hourly.nyx_annual_live_preflight import inspect_bundle, sha256


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, filename = tempfile.mkstemp(prefix="." + path.name + ".",
                                        suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(filename, path)
    finally:
        Path(filename).unlink(missing_ok=True)


def run(*, bundle: Path, delivery_day: str, output: Path) -> dict:
    """Fit and publish only after both gates pass and every input is pinned."""
    plan = preflight(bundle, delivery_day, output)
    require(plan["ready"], "; ".join(plan["blockers"]))
    bundle, output = bundle.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = pd.Timestamp.now(tz="UTC").isoformat()

    def status(phase: str, state: str = "RUNNING", **details) -> None:
        payload = {"protocol": PROTOCOL, "status": state, "phase": phase,
                   "delivery_day": delivery_day, "started_utc": started,
                   "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                   **details}
        _atomic_json(output / "status.json", payload)
        print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)

    try:
        status("pin_bundle")
        initial_hashes = bundle_hashes(bundle)
        data = load_bundle(bundle, delivery_day)
        require(bundle_hashes(bundle) == initial_hashes,
                "Live bundle changed while inputs were loaded")
        status("fit_models", total_models=6)
        frames, model_audit = execute_models(data, delivery_day, output / "models",
            progress=lambda phase, **details: status(phase, **details))
        require(bundle_hashes(bundle) == initial_hashes,
                "Live bundle changed during CPU retraining")
        require(inspect_bundle(bundle, delivery_day)["input_bundle_valid"],
                "Live bundle failed structural verification after retraining")
        require(verify_activation() == plan["activation"],
                "Annual CPU activation changed during retraining")
        status("publish")
        countries = {}
        for zone in COUNTRIES:
            frame = frames[zone]
            directory = output / "zones" / zone
            directory.mkdir(parents=True, exist_ok=False)
            name = f"forecast_{zone.lower()}_{delivery_day}_nyx_annual_cpu"
            parquet_path, csv_path = (directory / f"{name}.{suffix}"
                                      for suffix in ("parquet", "csv"))
            require(not parquet_path.exists() and not csv_path.exists(),
                    "Forecast output already exists")
            frame.to_parquet(parquet_path, index=True)
            frame.to_csv(csv_path, index=True, float_format="%.17g")
            countries[zone] = {"hours": len(frame), "composition":
                model_audit["compositions"][zone],
                "csv": str(csv_path), "csv_sha256": sha256(csv_path),
                "parquet": str(parquet_path), "parquet_sha256": sha256(parquet_path)}
        receipt = {"protocol": PROTOCOL, "status": "COMPLETE",
            "delivery_day": delivery_day, "countries": countries,
            "bundle": str(bundle), "input_sha256": initial_hashes,
            "manifest_sha256": sha256(MANIFEST),
            "qualification_sha256": plan["activation"]["qualification_sha256"],
            "code_sha256": {name: sha256(ROOT / name) for name in QUALIFICATION_CODE},
            "model_audit": model_audit, "saturn_fetched": False,
            "future_labels_used": False, "Storm_used_as_model_input": False}
        _atomic_json(output / "receipt.json", receipt)
        status("complete", "COMPLETE", receipt=str(output / "receipt.json"))
        return receipt
    except BaseException as error:
        status("failed", "FAILED", error=f"{type(error).__name__}: {error}",
               traceback=traceback.format_exc())
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--delivery-day", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or ROOT / "runs" / "nyx_annual_cpu_live" / args.delivery_day
    try:
        if args.preflight:
            report = preflight(args.bundle, args.delivery_day, output)
            print(json.dumps(report, ensure_ascii=False, default=str))
            return 0 if report["ready"] else 2
        receipt = run(bundle=args.bundle, delivery_day=args.delivery_day,
                      output=output)
        print(json.dumps({"status": "COMPLETE", "receipt": str(output / "receipt.json"),
                          "countries": list(receipt["countries"])}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "BLOCKED", "protocol": PROTOCOL,
                          "reason": f"{type(error).__name__}: {error}"}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
