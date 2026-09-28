"""Inspect archived 2026-09-23 annual CWE models without running a forecast."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_console.annual_cwe_inspection import COUNTRIES, inspect_annual_cwe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", action="append", choices=COUNTRIES, help="country to inspect; repeat for multiple")
    parser.add_argument("--preflight", action="store_true", help="return exit 2 when a forecast cannot be launched")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="repository checkout to inspect")
    args = parser.parse_args()
    result = inspect_annual_cwe(args.root, tuple(args.country or COUNTRIES))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["manifest_valid"] or (args.preflight and not result["forecast_ready"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
