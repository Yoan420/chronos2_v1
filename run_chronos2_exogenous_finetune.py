#!/usr/bin/env python
"""CLI for the isolated causal Chronos-2 exogenous LoRA laboratory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from chronos2_exogenous.lora_finetune import (
    config_summary,
    load_checkpoint,
    load_config,
    publish_evaluation_evidence,
    read_panel,
    train_lora,
    validate_panel,
    verify_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tuning LoRA exogene Chronos-2 avec origines D-1 PIT et "
            "holdout rolling-365 gele."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser(
        "validate", help="Valide le YAML, le panel PIT et les splits sans charger Chronos-2."
    )
    validate.add_argument("--config", type=Path, required=True)

    train = commands.add_parser("train", help="Entraine et sauvegarde un adaptateur LoRA.")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--overwrite", action="store_true")

    inspect = commands.add_parser(
        "inspect", help="Verifie les checksums d'un bundle sans charger le modele."
    )
    inspect.add_argument("--run-dir", type=Path, required=True)

    load = commands.add_parser(
        "load", help="Verifie puis recharge reellement le checkpoint LoRA."
    )
    load.add_argument("--run-dir", type=Path, required=True)
    load.add_argument("--device-map", default="auto")

    publish = commands.add_parser(
        "publish-evaluation",
        help="Valide et publie le CSV.gz apparie de la reserve rolling-365.",
    )
    publish.add_argument("--run-dir", type=Path, required=True)
    publish.add_argument("--input", type=Path, required=True)
    publish.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        config = load_config(args.config)
        _, split, audit = validate_panel(read_panel(config.panel_path), config)
        result = config_summary(config)
        result["split_counts"] = {
            "train": len(split.train),
            "validation": len(split.validation),
            "evaluation_holdout": len(split.evaluation),
        }
        result["pit_audit"] = audit
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "train":
        output = train_lora(args.config, overwrite=bool(args.overwrite))
        print(f"Bundle LoRA publie: {output}")
        print(f"Manifest: {output / 'experiment_manifest.json'}")
        return 0
    if args.command == "inspect":
        print(json.dumps(verify_bundle(args.run_dir), indent=2, ensure_ascii=False))
        return 0
    if args.command == "load":
        load_checkpoint(args.run_dir, device_map=str(args.device_map))
        print(f"Checkpoint LoRA recharge et verifie: {args.run_dir}")
        return 0
    if args.command == "publish-evaluation":
        frame = read_panel(args.input)
        output = publish_evaluation_evidence(
            args.run_dir,
            frame,
            overwrite=bool(args.overwrite),
        )
        print(f"Evidence rolling publiee: {output}")
        return 0
    raise RuntimeError(f"Commande inconnue: {args.command}.")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
