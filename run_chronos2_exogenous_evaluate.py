"""CLI for isolated rolling-365 and prospective shadow LoRA evaluation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd

from chronos2_exogenous.evaluation import run_evaluation, run_shadow
from chronos2_exogenous.lora_finetune import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Chronos-2 base et l'adaptateur LoRA avec les mêmes entrées "
            "exogènes, ou alimente le journal shadow append-only."
        )
    )
    parser.add_argument("--config", required=True, help="Configuration YAML du fine-tuning.")
    parser.add_argument(
        "--mode", choices=("backtest", "shadow"), default="backtest"
    )
    parser.add_argument(
        "--run-directory",
        help="Bundle LoRA; par défaut output.directory de la configuration.",
    )
    parser.add_argument("--item-id", help="Zone/item unique à évaluer (ex. FR).")
    parser.add_argument("--target-column", help="Target du modèle multi-target.")
    parser.add_argument("--device-map", help="Surcharge du device (auto, cpu, xpu...).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Budget de variates par batch Chronos-2 (défaut: 64).",
    )
    parser.add_argument(
        "--inference-chunk-size",
        type=int,
        default=32,
        help="Nombre d'origines de même horizon par cache atomique (backtest).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Réécrit le backtest.")
    parser.add_argument("--report-title", default="POC Chronos-2 exogène — rolling 365 jours")
    parser.add_argument(
        "--panel",
        help=(
            "Panel alternatif: shadow, ou panel holdout resolu tardivement en "
            "backtest (toutes les autres valeurs doivent rester identiques)."
        ),
    )
    parser.add_argument(
        "--panel-audit",
        help="Sidecar JSON obligatoire, lie cryptographiquement au panel shadow.",
    )
    parser.add_argument(
        "--origin",
        action="append",
        help="Origine UTC shadow; répétable. Par défaut, dernière origine du panel.",
    )
    parser.add_argument(
        "--shadow-predictions",
        help="Journal shadow .csv.gz; par défaut <bundle>/shadow_predictions.csv.gz.",
    )
    return parser


def _progress(number: int, total: int, origin: pd.Timestamp) -> None:
    if number == 1 or number == total or number % 10 == 0:
        print(
            f"[EXOGENOUS] {number}/{total} origine={origin.isoformat()}",
            file=sys.stderr,
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.batch_size <= 0 or args.inference_chunk_size <= 0:
        raise SystemExit("--batch-size et --inference-chunk-size doivent être positifs.")
    config = load_config(args.config)
    if args.device_map:
        config = replace(config, device_map=str(args.device_map))

    if args.mode == "backtest":
        if (args.panel is None) != (args.panel_audit is None):
            raise SystemExit("--panel et --panel-audit doivent etre fournis ensemble.")
        if args.origin or args.shadow_predictions:
            raise SystemExit(
                "--origin et --shadow-predictions sont réservés au mode shadow."
            )
        result = run_evaluation(
            config,
            run_directory=args.run_directory,
            panel_path=args.panel,
            panel_audit_path=args.panel_audit,
            item_id=args.item_id,
            target_column=args.target_column,
            batch_size=args.batch_size,
            inference_chunk_size=args.inference_chunk_size,
            overwrite=args.overwrite,
            progress=_progress,
            report_title=args.report_title,
        )
        payload = {
            "mode": "backtest",
            "evidence_path": str(result.evidence_path),
            "daily_path": str(result.daily_path),
            "metrics_path": str(result.metrics_path),
            "report_path": str(result.report_path),
            "manifest_path": str(result.manifest_path),
            "physical_days": result.metrics["physical_days"],
            "baseline_mae_eur_mwh": result.metrics["baseline_mae_eur_mwh"],
            "candidate_mae_eur_mwh": result.metrics["candidate_mae_eur_mwh"],
            "mae_gain_eur_mwh": result.metrics["mae_gain_eur_mwh"],
        }
    else:
        if args.overwrite:
            raise SystemExit(
                "--overwrite est interdit en shadow: le journal est append-only/idempotent."
            )
        result = run_shadow(
            config,
            run_directory=args.run_directory,
            panel_path=args.panel,
            panel_audit_path=args.panel_audit,
            journal_path=args.shadow_predictions,
            origins=args.origin,
            item_id=args.item_id,
            target_column=args.target_column,
            batch_size=args.batch_size,
            progress=_progress,
        )
        payload = {
            "mode": "shadow",
            "journal_path": str(result.journal_path),
            "observed_evidence_path": (
                str(result.observed_evidence_path)
                if result.observed_evidence_path is not None
                else None
            ),
            "shadow_manifest_path": (
                str(result.manifest_path) if result.manifest_path is not None else None
            ),
            "appended_rows": result.appended_rows,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
