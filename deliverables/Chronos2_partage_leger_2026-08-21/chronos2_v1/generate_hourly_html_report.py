#!/usr/bin/env python
"""Generate the standard standalone Plotly report from an hourly run."""

from __future__ import annotations

import argparse
from pathlib import Path

from chronos2_hourly.reporting import write_hourly_html_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genere un rapport HTML autonome avec le meme gabarit que les "
            "rapports Chronos-2 existants."
        )
    )
    parser.add_argument(
        "run_dir",
        help="Dossier contenant backtest_hourly_oof.csv.gz et les artefacts.",
    )
    parser.add_argument("--output", default=None, help="Chemin du rapport HTML.")
    parser.add_argument("--title", default=None, help="Titre affiche dans le rapport.")
    parser.add_argument(
        "--native-model",
        choices=("mkonline_blend", "residual_corrected", "ensemble", "chronos2", "catboost", "lear"),
        default=None,
        help="Modele principal; detection automatique si omis.",
    )
    parser.add_argument(
        "--baseline-model",
        choices=("mkonline_blend", "residual_corrected", "ensemble", "chronos2", "catboost", "lear"),
        default=None,
        help="Modele de comparaison; detection automatique si omis.",
    )
    parser.add_argument("--zone", default="FR")
    parser.add_argument("--timezone", default="Europe/Paris")
    parser.add_argument("--extreme-threshold", type=float, default=150.0)
    parser.add_argument("--history-hours", type=int, default=168)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = write_hourly_html_report(
        args.run_dir,
        output_path=args.output,
        title=args.title,
        native_model=args.native_model,
        baseline_model=args.baseline_model,
        zone=args.zone,
        timezone=args.timezone,
        extreme_threshold=args.extreme_threshold,
        history_hours=args.history_hours,
    )
    print(f"Rapport HTML : {Path(output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
