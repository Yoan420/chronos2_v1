#!/usr/bin/env python
"""Command-line entry point for the isolated auxiliary-model laboratory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from auxiliary_lab import (
    compare_runs,
    evaluate_run,
    load_lab_config,
    predict_artifact,
    train_experiment,
)
from auxiliary_lab.config import MODEL_NAMES


def _models(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("Au moins un modele est requis.")
    unknown = sorted(set(result).difference(MODEL_NAMES))
    if unknown:
        raise argparse.ArgumentTypeError(f"Modeles inconnus: {unknown}.")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tuning, calibrage et evaluation des modeles auxiliaires "
            "sans modifier le pipeline Chronos-2."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-models", help="Liste les adapters disponibles.")

    validate = sub.add_parser("validate", help="Valide un YAML sans entrainer.")
    validate.add_argument("--config", type=Path, required=True)

    train = sub.add_parser(
        "train",
        aliases=["run", "retrain"],
        help="Tune, entraine, teste et publie.",
    )
    train.add_argument("--config", type=Path, required=True)
    train.add_argument(
        "--models",
        type=_models,
        default=None,
        help="Sous-ensemble separe par virgules; sinon utilise enabled dans le YAML.",
    )
    train.add_argument("--overwrite", action="store_true")

    evaluate = sub.add_parser("evaluate", help="Recalcule les metriques sans charger les modeles.")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, default=None)

    predict = sub.add_parser("predict", aliases=["infer"], help="Charge un artefact et predit un run futur.")
    predict.add_argument("--artifact", type=Path, required=True)
    predict.add_argument("--source-run", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)

    compare = sub.add_parser("compare", help="Compare plusieurs experiences figees.")
    compare.add_argument("--runs", nargs="+", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list-models":
        print("Modeles auxiliaires disponibles:")
        print("  - residual_corrector : fit CatBoost/HGB ou blend de correcteurs")
        print("  - mkonline_blend      : calibration du poids convexe MKOnline")
        print("  - kalman              : recherche KF/EKF/UKF et replay causal")
        return 0
    if args.command == "validate":
        config = load_lab_config(args.config)
        print(f"Configuration valide: {config.experiment_id}")
        print(f"Source: {config.source_run}")
        print(f"Sortie: {config.output_directory}")
        print("Modeles: " + ", ".join(config.enabled_models))
        return 0
    if args.command in {"train", "run", "retrain"}:
        output = train_experiment(
            args.config,
            models=args.models,
            overwrite=bool(args.overwrite),
        )
        print(f"Experience publiee: {output}")
        if (output / "report.html").is_file():
            print(f"Rapport: {output / 'report.html'}")
        return 0
    if args.command == "evaluate":
        print(f"Evaluation publiee: {evaluate_run(args.run_dir, output_directory=args.output_dir)}")
        return 0
    if args.command in {"predict", "infer"}:
        print(
            "Prediction publiee: "
            + str(
                predict_artifact(
                    args.artifact,
                    source_run=args.source_run,
                    output_path=args.output,
                )
            )
        )
        return 0
    if args.command == "compare":
        print(
            "Comparaison publiee: "
            + str(compare_runs(args.runs, output_directory=args.output_dir))
        )
        return 0
    raise RuntimeError(f"Commande non geree: {args.command}.")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
