#!/usr/bin/env python
"""Plan, freeze and verify a prospective Chronos-2/LoRA shadow epoch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from chronos2_exogenous.shadow_epoch import (
    ShadowEpochError,
    assess_shadow_epoch,
    earliest_new_epoch_plan,
    freeze_shadow_epoch,
    validate_finalisation_against_epoch,
    validate_shadow_delivery_against_epoch,
    verify_shadow_epoch,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-engagement prospectif two-phase: checkpoint, correcteur OOF, "
            "holdout 365 et premier jour shadow sont geles avant l'origine."
        )
    )
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan", help="Audit read-only d'un bundle existant.")
    freeze = sub.add_parser("freeze", help="Publie un pre-engagement immuable.")
    verify = sub.add_parser("verify", help="Revalide un pre-engagement existant.")
    phase_b = sub.add_parser(
        "finalize-check",
        help="Lie un FinalBacktest tardif au pre-engagement phase A.",
    )
    delivery = sub.add_parser(
        "delivery-check",
        help="Refuse un premier jour ou une suite shadow avec un trou.",
    )
    earliest = sub.add_parser(
        "earliest", help="Calcule la prochaine fenetre d'un nouveau bundle."
    )
    for command in (plan, freeze):
        command.add_argument("--run-directory", required=True, type=Path)
        command.add_argument("--residual-corrector", required=True, type=Path)
        command.add_argument("--oof-audit", required=True, type=Path)
        command.add_argument("--zone", required=True)
        command.add_argument("--first-shadow-day")
        command.add_argument("--shadow-days", type=int, default=30)
    freeze.add_argument("--output-directory", required=True, type=Path)
    verify.add_argument("epoch_directory", type=Path)
    verify.add_argument("--run-directory", type=Path)
    phase_b.add_argument("epoch_directory", type=Path)
    phase_b.add_argument("--run-directory", required=True, type=Path)
    delivery.add_argument("epoch_directory", type=Path)
    delivery.add_argument("--run-directory", required=True, type=Path)
    delivery.add_argument("--journal", required=True, type=Path)
    delivery.add_argument("--delivery-day", required=True)
    earliest.add_argument("--timezone", default="Europe/Paris")
    earliest.add_argument("--cutoff", default="08:00")
    earliest.add_argument("--shadow-days", type=int, default=30)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "earliest":
        payload = earliest_new_epoch_plan(
            timezone_name=args.timezone,
            cutoff_local_time=args.cutoff,
            shadow_days_required=args.shadow_days,
        )
    elif args.action == "verify":
        payload = verify_shadow_epoch(
            args.epoch_directory, run_directory=args.run_directory
        )
    elif args.action == "finalize-check":
        payload = validate_finalisation_against_epoch(
            args.epoch_directory, run_directory=args.run_directory
        )
    elif args.action == "delivery-check":
        payload = validate_shadow_delivery_against_epoch(
            args.epoch_directory,
            run_directory=args.run_directory,
            journal_path=args.journal,
            delivery_day=args.delivery_day,
        )
    elif args.action == "plan":
        payload = assess_shadow_epoch(
            run_directory=args.run_directory,
            residual_corrector_path=args.residual_corrector,
            oof_audit_path=args.oof_audit,
            zone=args.zone,
            requested_first_shadow_day=args.first_shadow_day,
            shadow_days_required=args.shadow_days,
        ).to_dict()
    else:
        manifest = freeze_shadow_epoch(
            run_directory=args.run_directory,
            residual_corrector_path=args.residual_corrector,
            oof_audit_path=args.oof_audit,
            zone=args.zone,
            output_directory=args.output_directory,
            requested_first_shadow_day=args.first_shadow_day,
            shadow_days_required=args.shadow_days,
        )
        payload = {
            "status": "frozen",
            "manifest": str(manifest),
            "promotion_eligible": False,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ShadowEpochError as exc:
        raise SystemExit(f"ECHEC epoch shadow: {exc}") from exc
