"""CLI for the isolated, non-promotable validation-30 residual experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from typing import Sequence

import pandas as pd

from chronos2_exogenous.lora_finetune import load_config
from chronos2_exogenous.research_validation_corrector import (
    evaluate_research_validation_corrector,
    fit_research_validation_corrector,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Correcteur residuel exploratoire ajuste sur validation30. "
            "Il n'est ni OOF ni utilisable par la production."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser(
        "fit",
        help="Infere uniquement validation30, ajuste puis scelle le correcteur.",
    )
    evaluate = subparsers.add_parser(
        "evaluate",
        help="Verifie le sceau, puis ouvre et evalue le holdout365 sans refit.",
    )
    for subparser in (fit, evaluate):
        subparser.add_argument("--config", required=True)
        subparser.add_argument("--run-directory")
        subparser.add_argument("--item-id", required=True)
        subparser.add_argument(
            "--research-directory",
            help=(
                "Sortie scellee du fit, ou source scellee de l'evaluation; "
                "valeur deterministe par defaut."
            ),
        )
    fit.add_argument("--device-map")
    fit.add_argument("--batch-size", type=int, default=64)
    fit.add_argument("--inference-chunk-size", type=int, default=30)
    return parser


def _progress(number: int, total: int, origin: pd.Timestamp) -> None:
    if number == 1 or number == total or number % 10 == 0:
        print(
            f"[RESEARCH-VALIDATION] {number}/{total} origine={origin.isoformat()}",
            file=sys.stderr,
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "fit":
        if args.batch_size <= 0 or args.inference_chunk_size <= 0:
            raise SystemExit(
                "--batch-size et --inference-chunk-size doivent etre positifs."
            )
        if args.device_map:
            config = replace(config, device_map=str(args.device_map))
        result = fit_research_validation_corrector(
            config,
            run_directory=args.run_directory,
            output_directory=args.research_directory,
            item_id=args.item_id,
            batch_size=args.batch_size,
            inference_chunk_size=args.inference_chunk_size,
            progress=_progress,
        )
        payload = {
            "mode": "research_validation_corrector_fit",
            "research_only": True,
            "promotion_eligible": False,
            "output_directory": str(result.output_directory),
            "corrector_path": str(result.corrector_path),
            "validation_predictions_path": str(result.predictions_path),
            "manifest_path": str(result.manifest_path),
        }
    else:
        result = evaluate_research_validation_corrector(
            config,
            run_directory=args.run_directory,
            research_directory=args.research_directory,
            item_id=args.item_id,
        )
        payload = {
            "mode": "research_validation_corrector_evaluate",
            "research_only": True,
            "promotion_eligible": False,
            "output_directory": str(result.output_directory),
            "predictions_path": str(result.predictions_path),
            "metrics_path": str(result.metrics_path),
            "daily_path": str(result.daily_path),
            "report_path": str(result.report_path),
            "manifest_path": str(result.manifest_path),
            "raw_lora_mae_eur_mwh": result.metrics["baseline_mae_eur_mwh"],
            "corrected_lora_mae_eur_mwh": result.metrics[
                "candidate_mae_eur_mwh"
            ],
            "exploratory_mae_gain_eur_mwh": result.metrics["mae_gain_eur_mwh"],
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
