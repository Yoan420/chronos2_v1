"""Evaluate the hourly NYX challenger enriched with native 15-minute forecasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"config/nyx_intrahour.yaml")
    parser.add_argument("--action", choices=["run", "audit"], default="run")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--baseline-delivery-day")
    args = parser.parse_args(argv)
    from nyx_intrahour.runner import load_config, run
    config = load_config(args.config)
    if args.source_manifest:
        config["source_manifest"] = str(args.source_manifest.resolve())
    if args.baseline_delivery_day:
        config["baseline_delivery_day"] = args.baseline_delivery_day
    directory = run(config, root=ROOT, audit_only=args.action == "audit")
    summary = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    print(json.dumps({"status": summary["status"], "reason": summary.get("reason"),
                      "report": str(directory/"report.html"), "directory": str(directory),
                      "activation_performed": False}, ensure_ascii=True, indent=2))
    return 0 if summary["status"] in {"complete", "prepared"} else 3 if summary["status"] in {"data_unavailable", "insufficient_data"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
