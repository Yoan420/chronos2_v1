"""Dedicated CLI; never invokes Forecast.ps1 or promotes an experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["audit", "prepare", "backtest", "run", "report"], default="run")
    parser.add_argument("--config", default="config/marginal_cost_expert.yaml")
    parser.add_argument("--zones", nargs="+", choices=["FR", "DE", "BE", "NL"])
    parser.add_argument("--run-directory", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    try:
        # A supplied snapshot decides its own schema, not a mutable live config.
        if args.run_directory is not None:
            frozen = args.run_directory / "config.json"
            schema = json.loads(frozen.read_text(encoding="utf-8"))["schema_version"]
        else:
            import yaml
            schema = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))["schema_version"]
        if schema == 2:
            from marginal_cost_expert.runner_v2 import load_config, prepare, backtest, report
        elif schema == 1:
            from marginal_cost_expert.runner import load_config, prepare, backtest, report
        else:
            raise ValueError(f"Unsupported experiment schema: {schema}")
        if args.action == "report":
            if args.run_directory is None:
                config = load_config(config_path)
                pointer = root / config["output_root"] / "latest.json"
                args.run_directory = Path(json.loads(pointer.read_text())["snapshot"])
            result = report(args.run_directory, project_root=root)
        elif args.action == "backtest" and args.run_directory is not None:
            result = backtest(args.run_directory.resolve(), project_root=root)
        else:
            config = load_config(config_path)
            snapshot = prepare(config, project_root=root, zones=args.zones)
            result = backtest(snapshot, project_root=root) if args.action in {"run", "backtest"} else snapshot
        print(json.dumps({"result": str(result), "production_modified": False, "activation_performed": False}, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f"[Marginal] ECHEC : {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
