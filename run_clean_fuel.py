"""Opt-in native Saturn CGC/CCC features in the existing residual correction."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path


def main(argv=None):
    from nyx_clean_fuel import runner as r
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=r.ROOT / "config/nyx_clean_fuel.yaml")
    parser.add_argument("--action", choices=("audit", "sync", "run", "report", "status"), default="run")
    parser.add_argument("--delivery-day")
    parser.add_argument("--zones", nargs="+", choices=("FR", "DE", "BE", "NL"))
    parser.add_argument("--threads", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = r.load_config(args.config)
    for key in ("delivery_day", "zones", "threads", "workers"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    r.validate_options(cfg)
    audit = r.source_audit(cfg)
    input_dir = r.safe(r.NAMESPACE / "inputs" / cfg["delivery_day"])
    bank = input_dir / "bank.parquet"
    receipt = r.safe(r.NAMESPACE / cfg["delivery_day"] / ("latest_" + "_".join(cfg["zones"]) + ".json"))
    if args.action == "audit":
        print(json.dumps({**audit, "fuel_bank_exists": bank.is_file()}, indent=2))
        return 0
    if args.action in ("sync", "run"):
        from nyx_clean_fuel.sources import materialize
        bank = materialize({"sources": cfg["fuel_sources"]}, audit["start_day"], audit["end_day"],
                           input_dir, workers=cfg["workers"])
        if args.action == "sync":
            print(json.dumps({"bank": str(bank), "production_modified": False}))
            return 0
        directories = []
        for zone in cfg["zones"]:
            directory = r.prepare_zone(cfg, zone, bank)
            directories.append(directory)
            r.write_json(receipt, {"directories": list(map(str, directories)), "zones_requested": cfg["zones"], "complete": False})
            r.evaluate(directory)
            r.report([directory], directory / "clean_fuel_report.html")
        r.write_json(receipt, {"directories": list(map(str, directories)), "zones_requested": cfg["zones"], "complete": True})
    else:
        if not receipt.is_file():
            if args.action == "status":
                daily = input_dir / "daily"
                checkpoints = list(daily.glob("*.json")) if daily.is_dir() else []
                print(json.dumps({"phase": "fuel_inputs", "bank_ready": bank.is_file(),
                                  "daily_checkpoints": len(checkpoints), "start_day": audit["start_day"],
                                  "end_day": audit["end_day"], "collection_lock_present": (input_dir / ".materialize.lock").exists(),
                                  "training_started": False, "production_modified": False}, indent=2))
                return 0
            raise ValueError("No run for the requested delivery/zones. Run CleanFuel -Action Run first.")
        saved = json.loads(receipt.read_text())
        directories = [r.safe(p) for p in saved["directories"]]
        if args.action == "status":
            print(json.dumps({str(p): json.loads((p / "status.json").read_text()) for p in directories}, indent=2))
            return 0
        if not saved["complete"]:
            raise ValueError("Not all requested zones have finished. Resume Run before the full comparison.")
    output = directories[0].parent.parent / ("clean_fuel_" + "_".join(cfg["zones"]) + "_report.html")
    r.report(directories, output)
    print(json.dumps({"report": str(output), "production_modified": False}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        logging.error("[CleanFuel] %s", exc)
        raise SystemExit(1)
