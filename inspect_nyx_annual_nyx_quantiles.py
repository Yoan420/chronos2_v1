"""Audit four prospective CPU NYX baselines and their bound producer receipts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chronos2_hourly.nyx_annual_nyx_quantiles_gate import validate_nyx_quantiles_source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--delivery-day", required=True)
    args = parser.parse_args(argv)
    try:
        result = validate_nyx_quantiles_source(args.bundle, args.delivery_day)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"ready": False, "reason": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ready": True, **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
