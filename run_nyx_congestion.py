"""Dedicated congestion laboratory; never invokes production forecasts."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("collect", "prepare", "run", "backtest", "report", "status"), default="run")
    parser.add_argument("--config", type=Path, default=ROOT/"config/nyx_congestion.yaml")
    parser.add_argument("--run-directory")
    parser.add_argument("--start-day")
    parser.add_argument("--end-day")
    args = parser.parse_args(argv)
    if args.action == 'collect' and not (args.start_day and args.end_day):
        parser.error('Collect exige --start-day et --end-day.')
    if args.action != "collect" and any((args.start_day, args.end_day)):
        parser.error("Les dates sont exclusivement reserves a Collect.")
    if args.action in ("prepare", "collect") and args.run_directory:
        parser.error("Prepare/Collect ne reutilisent pas --run-directory.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from nyx_congestion import runner
    try:
        config = runner.load_config(args.config)
        if args.action == "collect":
            result = runner.collect(config, root=ROOT, start_day=args.start_day, end_day=args.end_day)
        elif args.action == "prepare":
            result = {"status": "prepared", "snapshot": str(runner.prepare(config, root=ROOT))}
        else:
            directory = runner.resolve_snapshot(config, root=ROOT, value=args.run_directory, create=args.action in ("run", "backtest"))
            if args.action in ("run", "backtest"):
                runner.evaluate(directory, root=ROOT)
            if args.action in ("run", "backtest", "report"):
                result = {"status": "completed", "report_directory": str(runner.report(directory, root=ROOT)), "production_modified": False}
            else:
                result = runner.status(directory, root=ROOT)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False, default=str), flush=True)
        return 1 if args.action == 'collect' and result.get('status') != 'complete' else 0
    except (ValueError, OSError, KeyError) as exc:
        logging.error("[Congestion] %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

