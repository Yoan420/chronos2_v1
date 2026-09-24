"""Independent full NYX clean-fuel replay and standard production-layout reports."""
from __future__ import annotations
import argparse
import html
import json
import logging
import os
from pathlib import Path

# Only local, pinned weights. No model upload, download or change of activation.
if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main(argv=None):
    from nyx_clean_fuel import full_runner as r
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=r.ROOT / "config/nyx_clean_fuel_full.yaml")
    parser.add_argument("--action", choices=("audit", "sync", "prepare", "run", "report", "status"), default="run")
    parser.add_argument("--delivery-day")
    parser.add_argument("--zones", nargs="+", choices=("FR", "DE", "BE", "NL"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--skip-attribution", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 32:
        parser.error("--threads must be between 1 and 32.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = r.load_settings(args.config)
    for key in ("delivery_day", "zones"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    r.validate_settings(cfg)
    audit = r.input_audit(cfg)
    receipt_path = r.safe(r.OUTPUT / cfg["delivery_day"] / ("latest_" + "_".join(cfg["zones"]) + ".json"))
    bank = r.residual_lab.NAMESPACE / "inputs" / cfg["delivery_day"] / "bank.parquet"
    if args.action == "audit":
        print(json.dumps({**audit, "chain": ["Chronos-2", "residual", "Kalman"],
                          "fuel_bank_ready": bank.is_file(), "operational_pipeline_modified": False}, indent=2))
        return 0
    if args.action == "status":
        if receipt_path.is_file():
            receipt = r.read(receipt_path)
            states = {zone: r.read(Path(path) / "run_status.json") if (Path(path) / "run_status.json").exists()
                      else {"status": "prepared", "workdir": path} for zone, path in receipt["workdirs"].items()}
            print(json.dumps(states, indent=2))
        else:
            days = list((bank.parent / "daily").glob("*.json"))
            print(json.dumps({"phase": "fuel_inputs", "daily_checkpoints": len(days), "bank_ready": bank.is_file(),
                              "collection_lock_present": (bank.parent / ".materialize.lock").exists(),
                              "training_started": False, "production_modified": False}, indent=2))
        return 0
    if args.action == "report":
        if not receipt_path.is_file():
            raise ValueError("No full-chain run for these zones/date; run CleanFuel -Action Run first.")
        workdirs = {zone: r.safe(path) for zone, path in r.read(receipt_path)["workdirs"].items()}
        if set(workdirs) != set(cfg["zones"]):
            raise ValueError("Incomplete zone set; cannot publish a complete comparison.")
    else:
        bank = r.fuel_bank(cfg, workers=args.workers)
        if args.action == "sync":
            print(json.dumps({"bank": str(bank), "production_modified": False}))
            return 0
        # Validate and freeze EVERY zone before starting a costly neural replay.
        workdirs = {zone: r.prepare(cfg, zone, bank) for zone in cfg["zones"]}
        r.write_json(receipt_path, {"workdirs": workdirs, "status": "prepared", "chain": "full", "production_modified": False})
    results = {}
    original_residual_replay = None
    if args.action == "run" and args.workers > 1:
        # Keep the checksum-pinned numerical engine unchanged. This process is
        # dedicated to CleanFuel, so the temporary binding affects no other run.
        from functools import partial
        from chronos2_hourly import nuclear_forecast as engine
        from nyx_clean_fuel.parallel_residual import prefill_residual_days

        original_residual_replay = engine.causal_residual_replay
        engine.causal_residual_replay = partial(
            prefill_residual_days, original_residual_replay, workers=args.workers)
    try:
        for zone, work in workdirs.items():
            results[zone] = r.run_zone(cfg, zone, work, action=args.action, device=args.device,
                                      threads=args.threads, workers=args.workers, skip_attribution=args.skip_attribution)
            r.write_json(receipt_path, {"workdirs": workdirs, "status": "running", "results": results,
                                       "chain": "full", "production_modified": False})
    finally:
        if original_residual_replay is not None:
            engine.causal_residual_replay = original_residual_replay
    r.write_json(receipt_path, {"workdirs": workdirs, "status": "prepared" if args.action == "prepare" else "complete",
                               "results": results, "chain": "full", "production_modified": False})
    if args.action != "prepare":
        index = r.safe(receipt_path.parent / ("clean_fuel_" + "_".join(cfg["zones"]) + "_index.html"))
        items = []
        for zone, result in results.items():
            for name, raw in result["reports"].items():
                path = Path(raw)
                if path.suffix == ".html":
                    link = os.path.relpath(path, index.parent).replace("\\", "/")
                    items.append(f'<li><a href="{html.escape(link, quote=True)}">{html.escape(zone + " — " + name)}</a></li>')
        index.write_text('<!doctype html><meta charset="utf-8"><title>NYX Clean Fuel</title>'
            '<style>body{font:18px system-ui;max-width:900px;margin:4rem auto;background:#111827;color:#e5e7eb}'
            'a{color:#7dd3fc}li{margin:1rem}</style><h1>NYX nucléaire + CGC/CCC</h1>'
            '<p>Variante indépendante : Chronos-2 + correcteur + Kalman. Production inchangée.</p><ul>'
            + ''.join(items) + '</ul>', encoding="utf-8")
        print(json.dumps({"index": str(index), "reports": {z: v["reports"] for z, v in results.items()},
                          "production_modified": False}, default=str, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        logging.exception("[CleanFuel full] %s", exc)
        raise SystemExit(1)
