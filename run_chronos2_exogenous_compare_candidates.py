"""CLI for the read-only final LoRA rank-8/rank-16 comparison."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from chronos2_exogenous.candidate_comparison import compare_final_candidates


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare les preuves LoRA rang 8 et rang 16 sur la meme fenetre "
            "rolling365. Les preuves finales corrigees sont preferees; deux "
            "preuves raw restent comparables mais non eligibles a la promotion."
        )
    )
    parser.add_argument(
        "--rank8",
        required=True,
        help="Run ou artefact d'evaluation final/raw du candidat rang 8.",
    )
    parser.add_argument(
        "--rank16",
        required=True,
        help="Run ou artefact d'evaluation final/raw du candidat rang 16.",
    )
    parser.add_argument("--policy", required=True, help="Politique de gouvernance YAML/JSON.")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--zone", help="Zone attendue; doit correspondre aux deux preuves.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--report-title",
        default="Chronos-2 + LoRA — comparaison rang 8 / rang 16",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = compare_final_candidates(
        rank8_source=args.rank8,
        rank16_source=args.rank16,
        policy_path=args.policy,
        output_directory=args.output_directory,
        expected_zone=args.zone,
        overwrite=args.overwrite,
        report_title=args.report_title,
    )
    print(
        json.dumps(
            {
                "output_directory": str(result.output_directory),
                "comparison_path": str(result.comparison_path),
                "report_path": str(result.report_path),
                "daily_path": str(result.daily_path),
                "decision": result.decision,
                "winner": result.winner,
                "evidence_stage": result.comparison["evidence_stage"],
                "candidate_selection_eligible": result.comparison[
                    "candidate_selection_eligible"
                ],
                "promotion_eligible": False,
                "promotion_performed": False,
                "activation_performed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
