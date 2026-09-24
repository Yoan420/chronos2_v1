"""NYX solar-ramp shadow laboratory; no operational forecast is changed."""
import argparse
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action", choices=["run", "prepare", "backtest", "report", "status", "prospective"], default="run")
    p.add_argument("--config", type=Path, default=ROOT/"config/nyx_solar_ramp.yaml")
    p.add_argument("--run-directory", type=Path)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from nyx_solar_ramp import runner
    try:
        config = runner.load_config(args.config)
        if args.action in ("run", "prepare"):
            if args.run_directory:
                raise ValueError("Use Backtest or Report with a frozen run directory.")
            directory = runner.prepare(config, root=ROOT)
        else:
            directory = args.run_directory or runner.resolve_latest(config, root=ROOT)
        result = {"snapshot": str(directory), "production_modified": False, "activation_performed": False}
        if args.action in ("run", "backtest"):
            result["report"] = str(runner.backtest(directory, root=ROOT))
        elif args.action == "report":
            result["report"] = str(runner.report(directory, root=ROOT))
        elif args.action == "prospective":
            _, manifest, _ = runner.read_snapshot(directory, root=ROOT)
            result["prospective"] = manifest["prospective"]
            result["forecast_issued"] = False
        elif args.action == "status":
            directory, _, _ = runner.read_snapshot(directory, root=ROOT)
            result["status"] = json.loads((directory/"status.json").read_text(encoding="utf-8"))
        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception:
        logging.exception("Solar experiment failed; operational forecasts unchanged.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
