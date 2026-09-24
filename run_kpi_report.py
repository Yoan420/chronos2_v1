"""Dedicated offline KPI report; does not call Forecast.ps1 or a model."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("report", "list", "status"), default="report")
    parser.add_argument("--end-day")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--zones", nargs="+")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        from kpi_report.runner import produce_report, status
        if args.action == "report":
            result = produce_report(root=ROOT, end_day=args.end_day, models=args.models, zones=args.zones)
            result = {k: result[k] for k in ("status", "report", "end_day", "models", "zones", "production_modified")}
        elif args.action == "status":
            result = status(root=ROOT)
        else:
            from kpi_report.data import load_recent_models
            _, catalog, audit = load_recent_models(ROOT)
            result = {"models": catalog, "last_common_evaluation_day": audit["recommended_end_day"], "production_modified": False}
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception:
        logging.exception("[KPI] Report stopped. No operational forecast was changed.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
