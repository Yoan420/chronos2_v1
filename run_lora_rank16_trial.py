"""Dedicated research commands; never alters Forecast.ps1 production modes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

from chronos2_exogenous.prospective_trial import (
    execute_trial, load_trial_config, prepare_trial, resolve_and_report, trial_status,
    now_utc, _day,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["prepare", "bootstrap", "run", "compare", "fullreport", "resolve", "status"], required=True)
    parser.add_argument("--config", default=str(Path(__file__).parent / "config/chronos2_exogenous_rank16_trial.yaml"))
    parser.add_argument("--zones", nargs="+", choices=["FR", "DE", "BE", "NL"], default=["FR", "DE", "BE", "NL"])
    parser.add_argument("--delivery-day")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    config = load_trial_config(args.config)
    if args.action == "fullreport":
        if not args.delivery_day:
            parser.error("--delivery-day requis pour fullreport")
        from chronos2_exogenous.rolling_research import run_rolling_report
        result = run_rolling_report(config, delivery_day=args.delivery_day, zones=args.zones,
                                    device=args.device, threads=args.threads)
    elif args.action == "prepare":
        prepare_trial(config)
        result = trial_status(config)
    elif args.action == "status":
        result = trial_status(config)
    elif args.action == "resolve":
        result = resolve_and_report(config)
    else:
        if not args.delivery_day:
            parser.error("--delivery-day requis pour bootstrap/run")
        day = _day(args.delivery_day)
        deadline = (pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=11, minutes=45)).tz_localize("Europe/Paris")
        retrospective = args.action == "compare" or (args.action == "run" and now_utc() >= deadline)
        if retrospective:
            from chronos2_exogenous.retrospective_trial import execute_retrospective_comparison
            print("Mode RETROSPECTIF : cette execution ne sera pas comptabilisee comme une emission avant enchere.", flush=True)
            result = execute_retrospective_comparison(config, delivery_day=day, zones=args.zones,
                                                       device=args.device, threads=args.threads)
        else:
            result = execute_trial(config, delivery_day=day, zones=args.zones,
                                   prospective=args.action == "run", device=args.device, threads=args.threads)
            if args.action == "run":
                result["reporting"] = resolve_and_report(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
