"""CLI for the sealed operational LoRA + residual-corrector evaluation."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from chronos2_exogenous.final_pipeline import run_final_pipeline_evaluation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Applique le correcteur residuel OOF gele au holdout LoRA brut, "
            "l'apparie a l'incumbent residual_corrected et publie la preuve "
            "finale rolling 365 jours."
        )
    )
    parser.add_argument("--run-directory", required=True, help="Bundle LoRA verifie.")
    parser.add_argument(
        "--residual-corrector",
        required=True,
        help="residual_corrector.json produit sur 365 jours OOF pre-holdout.",
    )
    parser.add_argument(
        "--oof-audit", required=True, help="Sidecar JSON de l'entrainement OOF."
    )
    parser.add_argument(
        "--incumbent-statistics",
        required=True,
        help=(
            "CSV/CSV.gz de l'incumbent avec actual, forecast_origin_utc et "
            "residual_corrected__q10/q50/q90."
        ),
    )
    parser.add_argument("--zone", required=True, help="Zone per-zone (ex. FR).")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remplace explicitement une evaluation finale deja publiee.",
    )
    parser.add_argument(
        "--report-title",
        default="Chronos-2 + LoRA — pipeline final rolling 365 jours",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_final_pipeline_evaluation(
        run_directory=args.run_directory,
        residual_corrector_path=args.residual_corrector,
        oof_audit_path=args.oof_audit,
        incumbent_statistics_path=args.incumbent_statistics,
        zone=args.zone,
        overwrite=args.overwrite,
        report_title=args.report_title,
    )
    payload = {
        "output_directory": str(result.output_directory),
        "evidence_path": str(result.evidence_path),
        "metrics_path": str(result.metrics_path),
        "report_path": str(result.report_path),
        "audit_path": str(result.audit_path),
        "manifest_path": str(result.manifest_path),
        "experiment_manifest_path": str(result.experiment_manifest_path),
        "physical_days": result.metrics["physical_days"],
        "physical_hours": result.metrics["physical_hours"],
        "baseline_mae_eur_mwh": result.metrics["baseline_mae_eur_mwh"],
        "candidate_mae_eur_mwh": result.metrics["candidate_mae_eur_mwh"],
        "mae_gain_eur_mwh": result.metrics["mae_gain_eur_mwh"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
