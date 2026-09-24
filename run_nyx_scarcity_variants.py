"""Isolated, sealed factorial comparison of NYX scarcity classifier variants."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["run", "prepare", "backtest", "report", "status", "install", "worker"], default="run")
    parser.add_argument("--config", type=Path, default=Path("config/nyx_scarcity_variants.yaml"))
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--source-snapshot", type=Path)
    parser.add_argument("--variant", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from nyx_scarcity import variants_runner as lab
    try:
        if args.source_snapshot and args.action not in {"run", "prepare"}:
            raise ValueError("SourceSnapshot is only allowed for Run/Prepare; existing runs are frozen.")
        if args.run_directory and args.action in {"run", "prepare", "install"}:
            raise ValueError("RunDirectory is only allowed for Backtest/Report/Status.")
        if args.variant is not None and args.action != "worker":
            raise ValueError("Variant is an internal worker argument only.")
        if args.action == "install":
            from nyx_scarcity.variant_runtime import install_runtime
            result = {"status": "installed_private_dependency", "path": str(install_runtime(root))}
        elif args.action == "worker":
            if not args.run_directory or not args.variant:
                raise ValueError("Worker needs its sealed RunDirectory and Variant.")
            result = lab.run_worker(args.run_directory, args.variant, root=root)
        else:
            if args.run_directory:
                directory, config, _ = lab.read_suite(args.run_directory, root=root)
            else:
                config = lab.load_config(args.config if args.config.is_absolute() else root / args.config)
                if args.source_snapshot:
                    config["source_snapshot"] = str(args.source_snapshot)
                directory = None
            if args.action in {"run", "prepare"}:
                directory = lab.prepare(config, root=root)
                report = lab.run_suite(directory, root=root) if args.action == "run" else None
                result = {"snapshot": str(directory), "report": str(report) if report else None}
            else:
                directory = directory or lab.resolve_latest(config, root=root, completed=args.action == "report")
                if args.action == "status":
                    result = json.loads((directory / "status.json").read_text(encoding="utf-8"))
                else:
                    report = lab.run_suite(directory, root=root) if args.action == "backtest" else lab.report(directory, root=root)
                    result = {"snapshot": str(directory), "report": str(report)}
        print(json.dumps({**result, "diagnostic_only": True, "production_modified": False,
                          "activation_performed": False}, ensure_ascii=False, indent=2, allow_nan=False, default=str))
        return 0
    except KeyboardInterrupt:
        logging.error("Interrupted. Backtest resumes only unfinished variants from their frozen inputs.")
        return 130
    except Exception:
        logging.exception("[Scarcity variants] Failed; operational forecasts have not been changed.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
