#!/usr/bin/env python
"""Read original public NOAA GFS forecasts into an experimental hourly panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-day", required=True)
    parser.add_argument("--end-day", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--cache-dir")
    parser.add_argument("--zones", nargs="+", default=["FR", "DE", "BE", "NL"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=45)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--ca-bundle")
    parser.add_argument("--radiation-tolerance", type=float, default=0.5, help="Packing tolerance in W/m2, at most 1; every clipped negative is counted.")
    parser.add_argument("--dependency-directory", help="Isolated directory containing the optional ecCodes installation.")
    args = parser.parse_args()
    if args.dependency_directory:
        dependencies = Path(args.dependency_directory).resolve(strict=True)
        if not dependencies.is_dir():
            parser.error("--dependency-directory must be a directory.")
        sys.path.insert(0, str(dependencies))
    from auxiliary_lab.noaa_gfs import materialize_noaa_gfs_weather
    result = materialize_noaa_gfs_weather(start_day=args.start_day, end_day=args.end_day, output_path=args.output, zones=args.zones, cache_dir=args.cache_dir, manifest_path=args.manifest, workers=args.workers, timeout_seconds=args.timeout_seconds, retries=args.retries, ca_bundle=args.ca_bundle, radiation_tolerance=args.radiation_tolerance)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
