"""Dedicated Economic Value Added launcher; local read-only sources, no trading."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["audit", "prepare", "backtest", "run", "report", "status"], default="run")
    parser.add_argument("--config", type=Path, default=Path("config/economic_value.yaml"))
    parser.add_argument("--zones", nargs="+", choices=["FR", "DE", "BE", "NL"])
    parser.add_argument("--models", nargs="+", choices=["autonomous", "kalman", "nuclear_autonomous", "nuclear_kalman"])
    parser.add_argument("--delivery-day")
    parser.add_argument("--end-day")
    parser.add_argument("--portfolio-mw", type=float)
    parser.add_argument("--run-directory", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    root = Path(__file__).resolve().parent
    from economic_value import runner
    try:
        config_path = args.config if args.config.is_absolute() else root / args.config
        if args.run_directory is not None and args.action in {"backtest", "report", "status"}:
            snapshot_path = runner._output(root, args.run_directory)
            config = json.loads((snapshot_path / "config.json").read_text(encoding="utf-8"))
            runner.validate_config(config)
        else:
            config = runner.load_config(config_path, zones=args.zones, models=args.models, delivery_day=args.delivery_day,
                                        end_day=args.end_day, portfolio_mw=args.portfolio_mw)
        if args.action == "audit":
            _, data, reference = runner.audit_inputs(config, root=root)
            result = {"status": "audit_read_only", "data": data, "reference": reference}
        elif args.action == "status":
            snapshot = runner.resolve_snapshot(root, config, args.run_directory, prepared=True)
            result = json.loads((snapshot / "status.json").read_text(encoding="utf-8"))
        elif args.action in {"report", "backtest"}:
            if any(value is not None for value in (args.zones, args.models, args.delivery_day, args.end_day, args.portfolio_mw)):
                raise ValueError("Report/Backtest use the sealed snapshot. Use Prepare/Run to change its parameters.")
            snapshot = runner.resolve_snapshot(root, config, args.run_directory, prepared=args.action == "backtest")
            path = runner.report(snapshot, root=root) if args.action == "report" else runner.evaluate(snapshot, root=root)
            result = {"snapshot": str(snapshot), "report": str(path), "diagnostic_only": True}
        else:
            if args.run_directory is not None:
                raise ValueError("Prepare/Run create a new snapshot; RunDirectory is for Backtest/Report/Status.")
            snapshot = runner.prepare(config, root=root)
            path = runner.evaluate(snapshot, root=root) if args.action == "run" else None
            result = {"snapshot": str(snapshot), "report": str(path) if path else None, "diagnostic_only": True}
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False, default=str))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        logging.error("[Economic Value] %s", exc)
        return 2
    except KeyboardInterrupt:
        logging.error("[Economic Value] Interrompu. Les snapshots precedents restent disponibles.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
