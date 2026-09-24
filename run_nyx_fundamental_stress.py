"""Separate fundamental backtests/reports, with no operational writes or promotion."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["run", "prepare", "backtest", "report", "status", "worker"], default="run")
    parser.add_argument("--config", type=Path, default=ROOT/"config/nyx_fundamental_stress.yaml")
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--variant")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        from nyx_fundamental_stress import runner
        if args.variant and args.action != "worker":
            raise ValueError("--variant is private to a sealed worker.")
        if args.run_directory and args.action in ("run", "prepare"):
            raise ValueError("Use Backtest to resume a frozen run; Run prepares a new snapshot.")
        if args.action in ("run", "prepare"):
            directory = runner.prepare(runner.load_config(args.config), root=ROOT)
            result = {"snapshot": str(directory)}
            if args.action == "run":
                result["report"] = str(runner.run_suite(directory, root=ROOT))
        else:
            directory = args.run_directory or runner.resolve_latest(runner.load_config(args.config), root=ROOT,
                completed=args.action == "report")
            if args.action == "worker":
                result = runner.run_worker(directory, args.variant, root=ROOT)
            elif args.action == "backtest":
                result = {"report": str(runner.run_suite(directory, root=ROOT))}
            elif args.action == "report":
                result = {"report": str(runner.report(directory, root=ROOT))}
            else:
                directory, _, _ = runner.read_suite(directory, root=ROOT)
                result = json.loads((directory/"status.json").read_text(encoding="utf-8"))
        print(json.dumps({**result, "diagnostic_only": True, "production_modified": False,
                          "activation_performed": False}, indent=2, default=str))
        return 0
    except Exception:
        logging.exception("Fundamental experiment failed. Existing operational forecasts are unchanged.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
