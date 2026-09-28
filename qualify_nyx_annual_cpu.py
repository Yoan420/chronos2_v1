#!/usr/bin/env python
"""Audit the full price and negative CPU replays, then seal qualification.

This command never starts a replay and never enables the NYX forecast gate.
Both replay directories must contain completed, immutable annual results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from chronos2_hourly.nyx_annual_cpu_qualification import (
    ROOT, prepare_receipt, seal_receipt,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--price-replay", type=Path, required=True)
    parser.add_argument("--negative-replay", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true",
                        help="verify both replays in read-only mode")
    args = parser.parse_args(argv)
    try:
        if args.preflight:
            receipt = prepare_receipt(args.root, args.price_replay,
                                      args.negative_replay)
            print(json.dumps({"ready": True, "price_expert_replay_qualified":
                              receipt["price_expert_replay_qualified"],
                              "negative_replay_verified": receipt["negative_replay_verified"],
                              "full_input_chain_qualified": False,
                              "qualified": False,
                              "price_country_metrics": receipt["price_country_metrics"],
                              "negative_country_metrics": receipt["negative_country_metrics"],
                              "replay_receipts_sha256": receipt["replay_receipts_sha256"]},
                             ensure_ascii=False, allow_nan=False))
            return 0
        result = seal_receipt(args.root, args.price_replay,
                              args.negative_replay)
        print(json.dumps({"price_expert_replay_qualified": True,
                          "full_input_chain_qualified": False,
                          "qualified": False, "path": result["path"],
                          "sha256": result["sha256"]}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        print(json.dumps({"ready": False, "qualified": False,
                          "blockers": [f"{type(error).__name__}: {error}"]},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
