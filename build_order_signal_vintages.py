#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from chronos2_order_signals.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construit les labels ex post, les scores OOF et les vintages PIT "
            "de structure de marché pour Chronos-2."
        )
    )
    parser.add_argument(
        "--config",
        default="chronos2_inputs_asof_jplus1.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=("backfill", "live", "backfill-live"),
        default="live",
    )
    parser.add_argument("--zone", default="FR")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--start-day", default=None)
    parser.add_argument("--end-day", default=None)
    parser.add_argument("--delivery-day", default=None)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    result = run_pipeline(
        config_path=Path(args.config),
        mode=args.mode,
        zone_name=args.zone,
        refresh_data=args.refresh_data,
        start_day=args.start_day,
        end_day=args.end_day,
        delivery_day=args.delivery_day,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
