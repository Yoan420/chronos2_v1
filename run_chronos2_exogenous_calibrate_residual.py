#!/usr/bin/env python
"""CLI for fold-specific LoRA OOF residual calibration."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Sequence

from chronos2_exogenous.lora_finetune import load_config
from chronos2_exogenous.oof_residual import OofFold, calibrate_residual_corrector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Produit 365 jours LoRA OOF blocked/prequentiels avec checkpoints "
            "fold-specific, puis calibre le correcteur residuel sans ouvrir le holdout."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--run-directory",
        type=Path,
        help="Bundle LoRA final; par defaut output.directory du YAML.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Cache/resultats; par defaut <bundle>/residual_calibration/<item>.",
    )
    parser.add_argument(
        "--shared-checkpoint-cache-directory",
        type=Path,
        help=(
            "Cache partage scelle des checkpoints de folds uniquement. Par "
            "defaut, les copies PrepareZones d'un meme artefact utilisent "
            "automatiquement <source-parent>/shared_oof_fold_checkpoints."
        ),
    )
    parser.add_argument(
        "--panel",
        type=Path,
        help="Panel 365 amorcage + 365 OOF + 365 holdout.",
    )
    parser.add_argument(
        "--panel-audit",
        type=Path,
        help="Sidecar SHA-lie au panel de calibration.",
    )
    parser.add_argument("--item-id", required=True, help="Zone/item a calibrer, ex. FR.")
    parser.add_argument("--target-column")
    parser.add_argument("--block-days", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device-map", help="Surcharge du device de fit et inference.")
    return parser


def _progress(number: int, total: int, fold: OofFold, phase: str) -> None:
    first = fold.prediction_origins[0].isoformat()
    last = fold.prediction_origins[-1].isoformat()
    print(
        f"[EXOGENOUS OOF] fold {number}/{total} {phase}: {first} -> {last}",
        file=sys.stderr,
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.block_days <= 0 or args.batch_size <= 0:
        raise SystemExit("--block-days et --batch-size doivent etre positifs.")
    config = load_config(args.config)
    if args.device_map:
        config = replace(config, device_map=str(args.device_map))
    result = calibrate_residual_corrector(
        config,
        run_directory=args.run_directory,
        panel_path=args.panel,
        panel_audit_path=args.panel_audit,
        output_directory=args.output_directory,
        shared_checkpoint_cache_directory=args.shared_checkpoint_cache_directory,
        item_id=str(args.item_id),
        target_column=args.target_column,
        block_days=int(args.block_days),
        batch_size=int(args.batch_size),
        progress=_progress,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "directory": str(result.directory),
                "folds": result.folds,
                "oof_predictions": str(result.predictions_path),
                "oof_audit": str(result.audit_path),
                "residual_corrector": str(result.corrector_path),
                "manifest": str(result.manifest_path),
                "shared_checkpoint_cache_directory": (
                    str(result.shared_checkpoint_cache_directory)
                    if result.shared_checkpoint_cache_directory is not None
                    else None
                ),
                "shared_checkpoint_cache_contract_sha256": (
                    result.shared_checkpoint_cache_contract_sha256
                ),
                "promotion_performed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
