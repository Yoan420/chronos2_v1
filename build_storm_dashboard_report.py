#!/usr/bin/env python
"""Build a report-only snapshot compared with Storm's dashboard series."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from chronos2_hourly.storm_dashboard import (
    create_report_only_copy,
    fetch_native_dashboard_snapshot,
    load_dashboard_from_basecase_vintages,
    load_materialized_dashboard_comparator,
    normalize_native_dashboard_series,
    statistics_contract,
    storm_dashboard_series,
)
from chronos2_modular.saturn import create_saturn_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crée une copie report-only avec le contrat exact du dashboard "
            "Storm: extraction courante de la série native exacte, figée et "
            "checksumée, sans modifier le run source ni recalculer son "
            "forecast."
        )
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--saturn-native-series",
        help=(
            "Identifiant Saturn natif exact (par exemple "
            "power.price.fr.euromwh.h.fcst.3mv.storm)."
        ),
    )
    source.add_argument(
        "--materialized",
        help=(
            "CSV/Parquet explicite avec delivery_start_utc et "
            "storm_dashboard_official__q50/value/q50."
        ),
    )
    source.add_argument(
        "--pit-vintages",
        help=(
            "Parquet PIT Saturn basecase avec value_time_utc, "
            "snapshot_time_utc, revision_time_utc et value."
        ),
    )
    parser.add_argument("--value-column", default=None)
    parser.add_argument("--zone", default="FR")
    parser.add_argument("--timezone", default="Europe/Paris")
    parser.add_argument(
        "--saturn-url",
        default="https://saturn-energyscan.gem.myengie.com//api",
    )
    parser.add_argument("--saturn-author", default="BQ6757")
    parser.add_argument("--native-model", default=None)
    parser.add_argument("--baseline-model", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--extreme-threshold", type=float, default=150.0)
    parser.add_argument("--history-hours", type=int, default=168)
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Injecte et audite les artefacts sans régénérer le HTML.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_run = Path(args.source_run).expanduser().resolve()
    expected, actual = statistics_contract(source_run)
    if args.saturn_native_series:
        client = create_saturn_client(args.saturn_url, args.saturn_author)
        expected_series = storm_dashboard_series(args.zone)
        if args.saturn_native_series != expected_series:
            raise ValueError(
                "--saturn-native-series must equal the verified zone series: "
                f"{expected_series}"
            )
        raw, source = fetch_native_dashboard_snapshot(
            client,
            zone=args.zone,
            expected_index=expected,
        )
        comparator = normalize_native_dashboard_series(
            raw,
            zone=args.zone,
            expected_index=expected,
            actual=actual,
            source=source,
        )
    elif args.materialized:
        comparator = load_materialized_dashboard_comparator(
            args.materialized,
            expected_index=expected,
            actual=actual,
            value_column=args.value_column,
            timezone=args.timezone,
            zone=args.zone,
        )
    else:
        comparator = load_dashboard_from_basecase_vintages(
            args.pit_vintages,
            expected_index=expected,
            actual=actual,
            timezone=args.timezone,
            zone=args.zone,
        )

    output, report = create_report_only_copy(
        source_run_dir=source_run,
        output_dir=args.output_dir,
        comparator=comparator,
        regenerate_html=not args.no_html,
        report_title=args.title,
        native_model=args.native_model,
        baseline_model=args.baseline_model,
        zone=args.zone,
        timezone=args.timezone,
        extreme_threshold=args.extreme_threshold,
        history_hours=args.history_hours,
    )
    metrics = comparator.audit["metrics"]
    print(f"Snapshot report-only : {output}")
    if report is not None:
        print(f"Rapport HTML : {report}")
    print(
        "Storm dashboard : "
        f"MAE={metrics['mae']:.6f}, "
        f"couverture={100.0 * comparator.audit['coverage']:.4f}% "
        f"({comparator.audit['available_hours']}/"
        f"{comparator.audit['expected_hours']} h)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
