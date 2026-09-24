"""Isolated nuclear-Kalman price-residual expert: frozen inputs, training, price and EVA evaluation."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["run", "audit", "prepare", "backtest", "report", "status"], default="run")
    parser.add_argument("--config", type=Path, default=Path("config/nuclear_kalman_extreme.yaml"))
    parser.add_argument("--run-directory", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from economic_value import price_runner as runner
    from economic_value.runner import _path, _output
    try:
        if args.run_directory is not None and args.action in {"run", "audit", "prepare"}:
            raise ValueError("RunDirectory is reserved for Backtest, Report and Status; Run creates a new snapshot.")
        if args.run_directory is not None:
            config = json.loads((_output(root, args.run_directory) / "config.json").read_text(encoding="utf-8"))
            runner.validate_config(config)
        else:
            config = runner.load_config(_path(root, args.config))
        if args.action == "audit":
            panel, audit, _ = runner.audit_inputs(config, root=root)
            result = {"status": "audit_read_only", "evaluation_rows": len(panel), "audit": audit}
        elif args.action in {"backtest", "report", "status"}:
            snapshot = runner.resolve_snapshot(root, config, args.run_directory, prepared=args.action != "report")
            if args.action == "status":
                result = json.loads((snapshot / "status.json").read_text(encoding="utf-8"))
            else:
                path = runner.evaluate(snapshot, root=root) if args.action == "backtest" else runner.report(snapshot, root=root)
                result = {"snapshot": str(snapshot), "report": str(path)}
        else:
            snapshot = runner.prepare(config, root=root)
            path = runner.evaluate(snapshot, root=root) if args.action == "run" else None
            result = {"snapshot": str(snapshot), "report": str(path) if path else None}
        print(json.dumps({**result, "diagnostic_only": True, "production_modified": False}, ensure_ascii=False,
                         indent=2, allow_nan=False, default=str))
        return 0
    except KeyboardInterrupt:
        logging.error("Interrupted. Backtest restarts an incomplete snapshot deterministically; completed reports remain intact.")
        return 130
    except (ValueError, OSError, KeyError, TypeError) as exc:
        logging.error("[Price Expert] %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
