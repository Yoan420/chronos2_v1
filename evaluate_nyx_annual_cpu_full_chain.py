"""Evaluate daily four-country forecasts using qualified producer bundles.

Plan freezes 365 daily bundles and official comparisons; predict performs CPU
retraining without reading labels; score reads labels after predictions seal.
Shorter periods are diagnostic and can never activate production.
"""
import argparse
import json
from pathlib import Path

from chronos2_hourly.nyx_annual_cpu_full_chain import prepare_plan, predict_plan, qualify


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "predict", "score"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundles", type=Path)
    parser.add_argument("--comparisons", type=Path)
    parser.add_argument("--first-day")
    parser.add_argument("--stop-day-exclusive")
    parser.add_argument("--max-days", type=int)
    args = parser.parse_args()
    try:
        if args.action == "plan":
            if not all((args.bundles, args.comparisons, args.first_day, args.stop_day_exclusive)):
                parser.error("plan requires --bundles --comparisons --first-day --stop-day-exclusive")
            result = prepare_plan(root=args.root, bundles=args.bundles,
                comparisons=args.comparisons, output=args.output,
                first=args.first_day, stop=args.stop_day_exclusive)
            result = {"state": "PLAN_FROZEN", "days": len(result["delivery_days"])}
        elif args.action == "predict":
            result = predict_plan(root=args.root, output=args.output, max_days=args.max_days)
        else:
            result = qualify(root=args.root, output=args.output)
        print(json.dumps(result, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"state": "BLOCKED", "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
