"""Read-only preflight for future annual CWE CPU input bundles."""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

from chronos2_hourly.nyx_annual_live_preflight import ROOT, inspect_bundle


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-day", required=True, help="Europe/Paris civil delivery day, YYYY-MM-DD")
    parser.add_argument("--bundle", type=Path, help="Dated live input bundle directory")
    args = parser.parse_args(argv)
    day = date.fromisoformat(args.delivery_day)
    bundle = args.bundle or ROOT / "runs/live/nyx_annual_cpu" / day.isoformat()
    result = inspect_bundle(bundle, args.delivery_day)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["input_bundle_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
