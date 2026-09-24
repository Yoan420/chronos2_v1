"""Calibration-only diagnostic; frozen congestion signals, no production calls."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "run", "backtest", "report", "status", "dryrun"), default="run")
    parser.add_argument("--config", type=Path, default=ROOT/"config/nyx_congestion_calibration.yaml")
    parser.add_argument("--run-directory")
    args = parser.parse_args(argv)
    if args.action in ("prepare", "dryrun") and args.run_directory:
        parser.error("Prepare/DryRun ne reutilisent pas --run-directory.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from nyx_congestion_calibration import runner
    try:
        config = runner.load_config(args.config)
        if args.action == "dryrun":
            result = runner.dry_run(config, root=ROOT)
        elif args.action == "prepare":
            result = dict(status="prepared", snapshot=str(runner.prepare(config, root=ROOT)), stage1_retrained=False)
        else:
            directory = runner.resolve_snapshot(config, root=ROOT, value=args.run_directory, create=args.action in ("run", "backtest"))
            if args.action in ("run", "backtest"):
                runner.evaluate(directory, root=ROOT)
            if args.action in ("run", "backtest", "report"):
                result = dict(status="completed", report_directory=str(runner.report(directory, root=ROOT)),
                    stage1_retrained=False, production_modified=False)
            else:
                result = runner.status(directory, root=ROOT)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False, default=str), flush=True)
        return 0
    except (ValueError, OSError, KeyError, AssertionError) as exc:
        logging.error("[Congestion Calibration] %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
