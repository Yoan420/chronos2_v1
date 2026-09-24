"""Seal corrected LoRA-versus-incumbent prospective shadow evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from chronos2_exogenous.shadow_final import finalize_shadow_evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transforme le journal shadow LoRA brut en preuve du pipeline final "
            "appariee a l'incumbent residual_corrected."
        )
    )
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--raw-observed-evidence", required=True, type=Path)
    parser.add_argument("--raw-shadow-manifest", required=True, type=Path)
    parser.add_argument("--raw-shadow-journal", type=Path, default=None)
    parser.add_argument("--residual-corrector", required=True, type=Path)
    parser.add_argument("--oof-audit", required=True, type=Path)
    parser.add_argument("--incumbent-statistics", required=True, type=Path)
    parser.add_argument("--zone", required=True)
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument(
        "--allow-no-observed",
        action="store_true",
        help=(
            "Retourne pending sans publier de preuve si le prix observe du jour "
            "n'est pas encore disponible."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = finalize_shadow_evidence(
        run_directory=args.run_directory,
        raw_observed_evidence_path=args.raw_observed_evidence,
        raw_shadow_manifest_path=args.raw_shadow_manifest,
        raw_shadow_journal_path=args.raw_shadow_journal,
        residual_corrector_path=args.residual_corrector,
        oof_audit_path=args.oof_audit,
        incumbent_statistics_path=args.incumbent_statistics,
        zone=args.zone,
        output_directory=args.output_directory,
        allow_no_observed=args.allow_no_observed,
    )
    print(
        json.dumps(
            {
                "status": "pending_observation" if result.pending else "sealed",
                "output_directory": str(result.output_directory),
                "evidence_path": (
                    str(result.evidence_path)
                    if result.evidence_path is not None
                    else None
                ),
                "manifest_path": (
                    str(result.manifest_path)
                    if result.manifest_path is not None
                    else None
                ),
                "rows": result.rows,
                "candidate_output_stage": "exogenous_residual_corrected",
                "baseline_output_stage": "residual_corrected",
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
