"""Dedicated offline NYX RMSE experiment; never invokes Forecast.ps1."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "run", "backtest", "report", "status"), default="run")
    parser.add_argument("--config", type=Path, default=ROOT/"config/nyx_rmse.yaml")
    parser.add_argument("--run-directory")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from nyx_rmse import runner
    try:
        config = runner.load_config(args.config)
        if args.action == "prepare":
            if args.run_directory:
                parser.error("Prepare cree un nouveau snapshot; ne pas fournir --run-directory.")
            result = {"status": "prepared", "snapshot": str(runner.prepare(config, root=ROOT))}
        else:
            directory = runner.resolve_snapshot(config, root=ROOT, value=args.run_directory, create=args.action in ("run", "backtest"))
            if args.action in ("run", "backtest"):
                runner.evaluate(directory, root=ROOT)
            if args.action in ("run", "backtest", "report"):
                result = {"status": "completed", "report_directory": str(runner.report(directory, root=ROOT)), "production_modified": False}
            else:
                result = runner.status(directory, root=ROOT)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False, default=str), flush=True)
        return 0
    except (ValueError, OSError, KeyError) as exc:
        logging.error("[NYX RMSE] %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
