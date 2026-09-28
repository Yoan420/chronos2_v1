"""Download/resume Saturn inputs at each historical and current NYX cutoff."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from chronos2_hourly.nyx_annual_saturn_source import DEFAULT_CACHE, ROOT, sync, publish


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-day", required=True)
    parser.add_argument("--first-day")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--assemble-only", action="store_true")
    args = parser.parse_args(argv)
    bundle = args.bundle or ROOT / "runs/live/nyx_annual_cpu" / args.delivery_day
    if not args.assemble_only:
        sync(args.delivery_day, first_day=args.first_day, cache=args.cache, workers=args.workers)
    receipt = publish(bundle, args.delivery_day, first_day=args.first_day, cache=args.cache)
    print(json.dumps({"state": "COMPLETE", "source_group": "saturn", "receipt": str(receipt)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
