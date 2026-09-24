#!/usr/bin/env python
"""CLI for causal ECMWF IFS weather-run materialisation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from auxiliary_lab.weather import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_CUTOFF_TIME,
    DEFAULT_ENDPOINT,
    DEFAULT_FORECAST_HOURS,
    DEFAULT_MODEL,
    DEFAULT_TIMEZONE,
    ZONE_POINTS,
    materialize_open_meteo_weather,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialise des runs ECMWF IFS emis, causalement alignes sur les "
            "jours de livraison (D-2 18Z, sans fill/interpolation)."
        )
    )
    parser.add_argument("--start-day", required=True, help="Date locale incluse YYYY-MM-DD")
    parser.add_argument("--end-day", required=True, help="Date locale incluse YYYY-MM-DD")
    parser.add_argument("--output", required=True, help="Parquet PIT de sortie")
    parser.add_argument(
        "--zones",
        default=",".join(ZONE_POINTS),
        help="Zones separees par des virgules (FR,DE,BE,NL,ES)",
    )
    parser.add_argument("--manifest", default=None, help="Manifest JSON (optionnel)")
    parser.add_argument("--cache-dir", default=None, help="Cache des reponses JSON brutes")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("OPEN_METEO_SINGLE_RUNS_ENDPOINT", DEFAULT_ENDPOINT),
        help="Endpoint Single Runs; configurable aussi via OPEN_METEO_SINGLE_RUNS_ENDPOINT",
    )
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help="Nom de la variable d'environnement contenant la cle (jamais journalisee)",
    )
    parser.add_argument("--require-api-key", action="store_true")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--cutoff-time", default=DEFAULT_CUTOFF_TIME)
    parser.add_argument("--forecast-hours", type=int, default=DEFAULT_FORECAST_HOURS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore le cache existant et recharge les reponses",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zones = tuple(item.strip().upper() for item in args.zones.split(",") if item.strip())
    result = materialize_open_meteo_weather(
        start_day=args.start_day,
        end_day=args.end_day,
        output_path=Path(args.output),
        zones=zones,
        manifest_path=args.manifest,
        cache_dir=args.cache_dir,
        endpoint=args.endpoint,
        api_key_env=args.api_key_env,
        require_api_key=args.require_api_key,
        timezone=args.timezone,
        cutoff_time=args.cutoff_time,
        forecast_hours=args.forecast_hours,
        model=args.model,
        workers=args.workers,
        retries=args.retries,
        timeout_seconds=args.timeout_seconds,
        force_refresh=args.force_refresh,
    )
    print(
        f"{result.output_path} | rows={result.row_count} | days={result.day_count} | "
        f"responses={result.response_count} | sha256={result.dataset_sha256}",
        flush=True,
    )
    print(f"Manifest: {result.manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
