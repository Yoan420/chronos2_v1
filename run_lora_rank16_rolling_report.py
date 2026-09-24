"""Resumable, isolated rolling365 research reports for the two LoRA16 chains."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from chronos2_exogenous.prospective_trial import load_trial_config
from chronos2_exogenous.rolling_research import run_rolling_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "config/chronos2_exogenous_rank16_trial.yaml"))
    parser.add_argument("--delivery-day", required=True)
    parser.add_argument("--zones", nargs="+", choices=["FR", "DE", "BE", "NL"], default=["FR"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stage", choices=["plan", "prefix", "run"], default="run")
    parser.add_argument("--max-new-prefix-days", type=int)
    args = parser.parse_args()
    config = load_trial_config(args.config)
    result = run_rolling_report(config, delivery_day=args.delivery_day, zones=args.zones,
        device=args.device, threads=args.threads, workers=args.workers,
        stage=args.stage, max_new_prefix_days=args.max_new_prefix_days)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
