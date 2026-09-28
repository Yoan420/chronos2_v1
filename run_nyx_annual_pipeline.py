"""Daily Saturn update, CPU retraining and four-country annual NYX forecasts."""
from __future__ import annotations
import argparse
from datetime import timedelta
import json
from pathlib import Path
import pandas as pd
from chronos2_hourly.nyx_annual_pipeline import inspect, run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-day", default=(pd.Timestamp.now(tz="Europe/Paris").date() + timedelta(days=1)).isoformat())
    parser.add_argument("--action", choices=("inspect", "capture", "prepare", "forecast"), default="forecast")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-cache-root", type=Path,
                        help="Separate Saturn/fuel/thermal caches for chronological evaluation")
    args = parser.parse_args(argv)
    if args.action == "inspect":
        result = inspect(args.delivery_day, bundle=args.bundle, output=args.output)
        print(json.dumps(result, ensure_ascii=False, default=str))
        return 0 if result["ready"] else 2
    result = run(args.delivery_day, action=args.action, bundle=args.bundle, output=args.output,
                 cache_root=args.source_cache_root)
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0 if result["state"] in ("COMPLETE", "PREPARED", "CAPTURED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
