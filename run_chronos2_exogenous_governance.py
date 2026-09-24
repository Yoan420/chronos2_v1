"""Evaluate and seal a Chronos-2 exogenous promotion candidate.

This CLI is intentionally separate from ``Forecast.ps1``.  It can authorise a
shadow or production promotion decision, but it never rewrites the incumbent
live registry, recipes, checkpoints, or forecasts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from chronos2_exogenous.governance import (
    ExogenousGovernanceError,
    evaluate_promotion,
    load_policy,
    load_prediction_evidence,
    seal_promotion_bundle,
    verify_promotion_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_POLICY = PROJECT_ROOT / "config" / "chronos2_exogenous_promotion_v1.yaml"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "runs" / "experiments" / "chronos2_exogenous" / "promotion"
)


def _artifact(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Format attendu: role=chemin")
    role, path = value.split("=", 1)
    if not role.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Format attendu: role=chemin")
    return role.strip(), Path(path.strip())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gate rolling365 + live shadow du challenger Chronos-2 exogene, "
            "sans mutation de la production."
        )
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    evaluate = subparsers.add_parser(
        "evaluate", help="Recalculer les gates et sceller leurs preuves."
    )
    evaluate.add_argument("--zone", required=True)
    evaluate.add_argument("--rolling-predictions", required=True, type=Path)
    evaluate.add_argument("--rolling-end-day", default=None)
    evaluate.add_argument("--shadow-predictions", type=Path, default=None)
    evaluate.add_argument(
        "--shadow-manifest",
        type=Path,
        default=None,
        help=(
            "Manifeste scelle liant les emissions live au checkpoint; "
            "obligatoire avec --shadow-predictions."
        ),
    )
    evaluate.add_argument("--shadow-end-day", default=None)
    evaluate.add_argument("--experiment-manifest", required=True, type=Path)
    evaluate.add_argument(
        "--shadow-epoch-directory",
        type=Path,
        default=None,
        help=(
            "Epoch phase A a revalider; obligatoire lorsque le manifeste "
            "autorise un dernier jour d'evaluation non resolu."
        ),
    )
    evaluate.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    evaluate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    evaluate.add_argument("--candidate-id", default=None)
    evaluate.add_argument(
        "--artifact",
        action="append",
        type=_artifact,
        required=True,
        metavar="ROLE=PATH",
        help=(
            "Artefact recopie dans le bundle; checkpoint=... et schema=... "
            "sont obligatoires. Option repetable."
        ),
    )
    evaluate.add_argument(
        "--require-promote",
        action="store_true",
        help="Retourner le code 3 si la decision finale n'est pas promote.",
    )

    verify = subparsers.add_parser(
        "verify", help="Verifier tous les fichiers et checksums d'un bundle."
    )
    verify.add_argument("bundle", type=Path)
    return parser.parse_args(argv)


def _evaluate(args: argparse.Namespace) -> int:
    artifacts: dict[str, Path] = {}
    for role, path in args.artifact:
        if role in artifacts:
            raise ExogenousGovernanceError(f"Role d'artefact duplique: {role}")
        artifacts[role] = path
    policy = load_policy(args.policy)
    rolling = load_prediction_evidence(args.rolling_predictions)
    shadow = (
        load_prediction_evidence(args.shadow_predictions)
        if args.shadow_predictions is not None
        else None
    )
    decision = evaluate_promotion(
        rolling_predictions=rolling,
        experiment_manifest=args.experiment_manifest,
        zone=args.zone,
        policy=policy,
        rolling_end_day=args.rolling_end_day,
        shadow_predictions=shadow,
        shadow_manifest=args.shadow_manifest,
        shadow_end_day=args.shadow_end_day,
        shadow_epoch_directory=args.shadow_epoch_directory,
    )
    bundle = seal_promotion_bundle(
        output_root=args.output_root,
        rolling_predictions_path=args.rolling_predictions,
        shadow_predictions_path=args.shadow_predictions,
        shadow_manifest_path=args.shadow_manifest,
        experiment_manifest_path=args.experiment_manifest,
        decision=decision,
        policy=policy,
        candidate_artifacts=artifacts,
        candidate_id=args.candidate_id,
    )
    print(
        json.dumps(
            {
                "decision": decision.decision,
                "candidate_model": decision.candidate_model,
                "zone": decision.zone,
                "bundle": str(bundle),
                "rolling365_mae_gain_eur_mwh": (
                    decision.rolling365.mae_gain_eur_mwh
                ),
                "rolling365_relative_mae_gain": (
                    decision.rolling365.relative_mae_gain
                ),
                "rolling365_gate_passes": decision.rolling365_gate.passes,
                "shadow_gate_passes": (
                    decision.live_shadow_gate.passes
                    if decision.live_shadow_gate is not None
                    else None
                ),
                "production_pit_evidence": decision.production_pit_evidence,
                "production_pit_gate_passes": (
                    decision.production_pit_gate_passes
                ),
                "production_pipeline_evidence": (
                    decision.production_pipeline_evidence
                ),
                "production_pipeline_gate_passes": (
                    decision.production_pipeline_gate_passes
                ),
                "production_activation_performed": False,
                "reasons": list(decision.reasons),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.require_promote and decision.decision != "promote":
        return 3
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.action == "verify":
        manifest = verify_promotion_bundle(args.bundle)
        print(
            json.dumps(
                {
                    "status": "valid",
                    "bundle": str(args.bundle.resolve()),
                    "candidate_id": manifest["candidate_id"],
                    "decision": manifest["decision"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    return _evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
