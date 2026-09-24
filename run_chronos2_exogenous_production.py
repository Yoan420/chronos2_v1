"""Dedicated opt-in entry point for a promoted exogenous challenger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from chronos2_exogenous.production import (
    fit_oof_residual_corrector,
    load_registered_bundle,
    register_promoted_bundle,
    run_registered_candidate,
    validate_promoted_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY = (
    PROJECT_ROOT
    / "runs"
    / "experiments"
    / "chronos2_exogenous"
    / "production_registry.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enregistre, preflight ou execute explicitement un bundle exogene "
            "promu. Cette commande ne modifie jamais Forecast.ps1 ni Mode All."
        )
    )
    sub = parser.add_subparsers(dest="action", required=True)

    register = sub.add_parser("register")
    register.add_argument("--bundle", type=Path, required=True)
    register.add_argument("--alias", required=True)
    register.add_argument("--zone", default=None)
    register.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)

    preflight = sub.add_parser("preflight")
    target = preflight.add_mutually_exclusive_group(required=True)
    target.add_argument("--bundle", type=Path)
    target.add_argument("--alias")
    preflight.add_argument("--zone", default=None)
    preflight.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)

    run = sub.add_parser("run")
    run.add_argument("--alias", required=True)
    run.add_argument("--delivery-day", required=True)
    run.add_argument("--live-panel", type=Path, required=True)
    run.add_argument("--live-panel-audit", type=Path, required=True)
    run.add_argument("--output-directory", type=Path, required=True)
    run.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    run.add_argument("--device-map", default="auto")

    fit = sub.add_parser("fit-corrector")
    fit.add_argument("--oof-predictions", type=Path, required=True)
    fit.add_argument(
        "--oof-audit",
        type=Path,
        required=True,
        help="Sidecar blocked/prequential lie par SHA aux predictions OOF.",
    )
    fit.add_argument("--holdout-start-day", required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--timezone", default="Europe/Paris")
    fit.add_argument("--features", nargs="+", default=["intercept", "local_hour_sin", "local_hour_cos", "local_dow_sin", "local_dow_cos"])
    fit.add_argument("--ridge-alpha", type=float, default=1.0)
    fit.add_argument("--clip", type=float, default=20.0)
    return parser.parse_args(argv)


def _bundle_payload(bundle: object) -> dict[str, object]:
    return {
        "status": "valid",
        "alias": getattr(bundle, "alias"),
        "candidate_id": getattr(bundle, "candidate_id"),
        "candidate_model": getattr(bundle, "candidate_model"),
        "zone": getattr(bundle, "zone"),
        "bundle": str(getattr(bundle, "bundle_path")),
        "enabled_by_default": False,
        "included_in_mode_all": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.action == "register":
        bundle = register_promoted_bundle(
            args.bundle,
            registry_path=args.registry,
            alias=args.alias,
            expected_zone=args.zone,
        )
        payload = _bundle_payload(bundle)
        payload["status"] = "registered_inactive"
    elif args.action == "preflight":
        if args.bundle is not None:
            bundle = validate_promoted_bundle(
                args.bundle,
                alias="preflight",
                expected_zone=args.zone,
            )
        else:
            bundle = load_registered_bundle(
                registry_path=args.registry,
                alias=args.alias,
                expected_zone=args.zone,
            )
        payload = _bundle_payload(bundle)
    elif args.action == "run":
        result = run_registered_candidate(
            registry_path=args.registry,
            alias=args.alias,
            live_panel_path=args.live_panel,
            live_panel_audit_path=args.live_panel_audit,
            delivery_day=args.delivery_day,
            output_directory=args.output_directory,
            device_map=args.device_map,
        )
        payload = {
            "status": "forecast_published",
            "zone": result.zone,
            "delivery_day": result.delivery_day,
            "model": result.model,
            "output_directory": str(result.output_directory),
            "forecast": str(result.forecast_path),
            "backtest": str(result.backtest_path),
            "manifest": str(result.manifest_path),
            "included_in_mode_all": False,
        }
    else:
        output = fit_oof_residual_corrector(
            oof_predictions_path=args.oof_predictions,
            oof_audit_path=args.oof_audit,
            holdout_start_day=args.holdout_start_day,
            output_path=args.output,
            timezone_name=args.timezone,
            feature_columns=args.features,
            ridge_alpha=args.ridge_alpha,
            maximum_absolute_shift_eur_mwh=args.clip,
        )
        payload = {"status": "corrector_fitted", "output": str(output), "included_in_mode_all": False}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
