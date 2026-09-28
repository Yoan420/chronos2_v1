#!/usr/bin/env python
"""Verify a complete CPU-chain evaluation or inspect legacy expert replays.

Only --full-chain-evaluation combined with --activate can enable NYX forecasts.
It rechecks daily source, producer, model and official score evidence first.
Legacy expert replays remain conditional and cannot activate the forecast gate.
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
    parser.add_argument("--price-replay", type=Path)
    parser.add_argument("--negative-replay", type=Path)
    parser.add_argument("--full-chain-evaluation", type=Path,
        help="new chronological daily producer-bundle evaluation directory")
    parser.add_argument("--activate", action="store_true",
        help="enable production only after independently rechecking full-chain evidence")
    parser.add_argument("--preflight", action="store_true",
                        help="verify both replays in read-only mode")
    args = parser.parse_args(argv)
    try:
        if args.full_chain_evaluation:
            if args.price_replay or args.negative_replay:
                parser.error("Full-chain and legacy expert replay modes are distinct")
            from chronos2_hourly.nyx_annual_cpu_full_chain import qualify, score_plan
            if args.preflight:
                if args.activate:
                    parser.error("--preflight is read-only and cannot be combined with --activate")
                receipt = score_plan(root=args.root, output=args.full_chain_evaluation)
            else:
                receipt = qualify(root=args.root, output=args.full_chain_evaluation, activate=args.activate)
            print(json.dumps(receipt, allow_nan=False))
            return 0 if receipt["qualified"] else 2
        if args.activate:
            parser.error("--activate requires --full-chain-evaluation")
        if not args.price_replay or not args.negative_replay:
            parser.error("Legacy inspection requires --price-replay and --negative-replay")
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
