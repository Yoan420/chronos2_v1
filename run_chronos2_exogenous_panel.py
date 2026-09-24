#!/usr/bin/env python
"""Materialise the causal origin-aware panel for the Chronos-2 exogenous POC.

The command is deliberately read-only with respect to the live registry.  It
only reads already materialised PIT inputs and canonical target caches, then
writes an isolated Parquet plus its audit next to the experiment artefacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from chronos2_modular.common import load_yaml, parse_series_spec
from chronos2_modular.saturn import cache_path_for_series
from chronos2_exogenous.feature_bank import (
    ABLATION_PACKS,
    ExogenousBankError,
    build_default_project_bank,
    build_exogenous_bank,
    delivery_utc_index,
)
from chronos2_exogenous.live_sources import load_live_source_manifest
from chronos2_exogenous.panel import (
    DEFAULT_CWE_ZONES,
    OriginPanelError,
    build_origin_panel,
    load_target_cache,
    write_origin_panel,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "runs"
    / "experiments"
    / "chronos2_exogenous_lora_poc_v1"
    / "inputs"
    / "training_panel.parquet"
)
DEFAULT_TIMEZONE = "Europe/Paris"
DELIVERY_TIMEZONE_BY_ZONE = {
    "FR": "Europe/Paris",
    "DE": "Europe/Berlin",
    "BE": "Europe/Brussels",
    "NL": "Europe/Amsterdam",
    "ES": "Europe/Madrid",
}
LIVE_CONFIG_BY_ZONE = {
    zone: f"chronos2_hourly_{zone.casefold()}_mkonline_live_v1.yaml"
    for zone in ("FR", "DE", "BE", "NL", "ES")
}


def _zone_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Format attendu: ZONE=chemin")
    raw_zone, raw_path = value.split("=", 1)
    zone = raw_zone.strip().upper()
    if zone not in {"FR", "DE", "BE", "NL", "ES"} or not raw_path.strip():
        raise argparse.ArgumentTypeError("Format attendu: ZONE=chemin")
    return zone, Path(raw_path.strip())


def _canonical_target_path(root: Path, zone: str) -> tuple[Path, dict[str, object]]:
    """Resolve the one cache identity declared by the operational contract.

    Selecting the newest ``target__*.csv.gz`` is unsafe: two Saturn series can
    cover the same dates while carrying materially different prices.  The
    cache filename must therefore be derived from the exact base-config
    ``SeriesSpec`` used by the live runner.
    """

    live_path = root / LIVE_CONFIG_BY_ZONE[zone]
    if not live_path.is_file():
        raise OriginPanelError(f"{zone}: configuration live introuvable: {live_path}.")
    live = load_yaml(live_path)
    live_section = live.get("live")
    if not isinstance(live_section, Mapping):
        raise OriginPanelError(f"{zone}: section live invalide dans {live_path}.")
    base_value = live_section.get("base_config")
    if not isinstance(base_value, str) or not base_value.strip():
        raise OriginPanelError(f"{zone}: live.base_config absent dans {live_path}.")
    base_path = Path(base_value).expanduser()
    if not base_path.is_absolute():
        base_path = live_path.parent / base_path
    base_path = base_path.resolve()
    if not base_path.is_file():
        raise OriginPanelError(f"{zone}: base_config introuvable: {base_path}.")
    base = load_yaml(base_path)
    zones = base.get("zones")
    zone_config = zones.get(zone) if isinstance(zones, Mapping) else None
    target_raw = zone_config.get("target") if isinstance(zone_config, Mapping) else None
    if not isinstance(target_raw, Mapping):
        raise OriginPanelError(f"{zone}: zones.{zone}.target absent dans {base_path}.")
    data = base.get("data")
    data = data if isinstance(data, Mapping) else {}
    default_fill = int(data.get("default_fill_limit", 3))
    spec = parse_series_spec("target", target_raw, default_fill)
    configured_live_series = live_section.get("target_series")
    if configured_live_series not in (None, spec.series):
        raise OriginPanelError(
            f"{zone}: cible live {configured_live_series!r} differente de la cible "
            f"du base_config {spec.series!r}."
        )
    cache_root = Path(str(data.get("cache_dir", "data/cache"))).expanduser()
    if not cache_root.is_absolute():
        cache_root = base_path.parent / cache_root
    expected = cache_path_for_series(cache_root.resolve(), zone, spec).resolve()
    contract = {
        "live_config": str(live_path.resolve()),
        "base_config": str(base_path),
        "series": spec.series,
        "naive_timezone": spec.naive_timezone,
        "cache_path": str(expected),
    }
    return expected, contract


def _last_complete_day(target: pd.Series, *, timezone: str) -> pd.Timestamp:
    local_days = sorted(set(target.index.tz_convert(timezone).date), reverse=True)
    for day in local_days:
        expected = delivery_utc_index(str(day), str(day), timezone=timezone)
        values = target.reindex(expected)
        if len(values) == len(expected) and values.notna().all():
            return pd.Timestamp(day)
    raise OriginPanelError("Aucune journee cible civile complete.")


def _resolve_target(
    root: Path,
    zone: str,
    overrides: Mapping[str, Path],
    *,
    timezone: str,
) -> tuple[pd.Series, Path, pd.Timestamp]:
    canonical_path, _ = _canonical_target_path(root, zone)
    if zone in overrides:
        path = overrides[zone]
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if path != canonical_path:
            raise OriginPanelError(
                f"{zone}: override cible non canonique refuse. Attendu: "
                f"{canonical_path}; recu: {path}."
            )
        if not path.is_file():
            raise OriginPanelError(f"{zone}: cache cible introuvable: {path}.")
        target = load_target_cache(path, zone=zone)
        return target, path, _last_complete_day(target, timezone=timezone)

    if not canonical_path.is_file():
        raise OriginPanelError(
            f"{zone}: cache de la cible canonique introuvable: {canonical_path}."
        )
    target = load_target_cache(canonical_path, zone=zone)
    return target, canonical_path, _last_complete_day(target, timezone=timezone)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Construit le panel PIT Chronos-2 exogene: fit/OOF/holdout "
            "chronologiques, avec horizons DST physiques."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("training", "calibration", "shadow"),
        default="training",
        help=(
            "training construit 365+365 jours; calibration construit "
            "365 amorcage + 365 OOF + 365 holdout; shadow construit un seul D+1."
        ),
    )
    parser.add_argument("--zones", nargs="+", default=list(DEFAULT_CWE_ZONES))
    parser.add_argument("--pack", choices=sorted(ABLATION_PACKS), default="full")
    parser.add_argument("--layout", choices=("per_zone", "cwe_wide"), default="per_zone")
    parser.add_argument(
        "--end-day",
        default=None,
        help=(
            "YYYY-MM-DD; dernier jour commun complet en training, "
            "jour de livraison obligatoire en shadow"
        ),
    )
    parser.add_argument("--training-days", type=int, default=365)
    parser.add_argument("--evaluation-days", type=int, default=365)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--weather-root",
        type=Path,
        default=None,
        help=(
            "Repertoire explicite des Parquet meteo Saturn; si omis, utilise "
            "le repertoire meteo historique configure par le projet."
        ),
    )
    parser.add_argument(
        "--target-cache",
        action="append",
        type=_zone_path,
        default=[],
        metavar="ZONE=PATH",
        help="Remplace explicitement le cache cible d'une zone; option repetable.",
    )
    parser.add_argument(
        "--require-production-pit",
        action="store_true",
        help="Refuse toute source sans preuve de capture operationnelle.",
    )
    parser.add_argument(
        "--allow-unresolved-final-evaluation-day",
        action="store_true",
        help=(
            "Mode prospectif explicite: seule la target de l'horizon physique "
            "de la derniere origine holdout peut etre entierement absente."
        ),
    )
    parser.add_argument(
        "--live-source-manifest",
        type=Path,
        default=None,
        help=("JSON explicite de captures prospectives. Autorise uniquement en shadow; "
              "force les controles production et ne reclassifie aucun backfill."),
    )
    parser.add_argument(
        "--require-complete-context",
        action="store_true",
        help="Refuse aussi les NaN historiques de contexte (strict mais incompatible avec le JAO historique actuel).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.project_root.expanduser().resolve()
    zones = tuple(dict.fromkeys(str(value).strip().upper() for value in args.zones))
    unknown = sorted(set(zones).difference({"FR", "DE", "BE", "NL", "ES"}))
    if unknown or not zones:
        raise OriginPanelError(f"Zones invalides: {unknown or list(zones)}")
    if args.mode == "training" and (
        args.training_days != 365 or args.evaluation_days != 365
    ):
        raise OriginPanelError(
            "Le POC gouverne exige exactement --training-days 365 et --evaluation-days 365."
        )
    if args.mode == "calibration" and (
        args.training_days != 730 or args.evaluation_days != 365
    ):
        raise OriginPanelError(
            "Le correcteur OOF exige exactement --training-days 730 "
            "(365 amorcage + 365 OOF) et --evaluation-days 365."
        )
    if args.context_length <= 0:
        raise OriginPanelError("--context-length doit etre positif.")
    if args.live_source_manifest is not None and args.mode != "shadow":
        raise OriginPanelError("--live-source-manifest est reserve au mode shadow.")
    if args.allow_unresolved_final_evaluation_day and args.mode not in {
        "training",
        "calibration",
    }:
        raise OriginPanelError(
            "--allow-unresolved-final-evaluation-day est reserve a training/calibration."
        )
    if args.mode == "shadow" and args.end_day is None:
        raise OriginPanelError("--end-day est obligatoire en mode shadow.")
    overrides: dict[str, Path] = {}
    for zone, path in args.target_cache:
        if zone in overrides:
            raise OriginPanelError(f"Override target duplique: {zone}.")
        overrides[zone] = path

    targets: dict[str, pd.Series] = {}
    target_paths: dict[str, str] = {}
    target_contracts: dict[str, dict[str, object]] = {}
    complete_days: dict[str, pd.Timestamp] = {}
    for zone in zones:
        target, path, complete = _resolve_target(
            root, zone, overrides, timezone=args.timezone
        )
        targets[zone] = target
        target_paths[zone] = str(path)
        _, target_contracts[zone] = _canonical_target_path(root, zone)
        complete_days[zone] = complete
    end_day = (
        pd.Timestamp(args.end_day).normalize()
        if args.end_day is not None
        else min(complete_days.values())
    )
    maximum_target_day = {
        zone: value
        + (
            pd.Timedelta(days=1)
            if args.allow_unresolved_final_evaluation_day
            else pd.Timedelta(0)
        )
        for zone, value in complete_days.items()
    }
    if args.mode in {"training", "calibration"} and any(
        end_day > value for value in maximum_target_day.values()
    ):
        shortages = {
            zone: str(value.date())
            for zone, value in maximum_target_day.items()
            if end_day > value
        }
        raise OriginPanelError(
            f"Fin demandee {end_day.date()} hors de la limite cible autorisee: "
            f"{shortages}. Le mode prospectif n'autorise au plus qu'un seul "
            "jour final non resolu."
        )
    if args.mode in {"training", "calibration"}:
        total_days = args.training_days + args.evaluation_days
        start_day = end_day - pd.Timedelta(days=total_days - 1)
        delivery_days = pd.date_range(start_day, end_day, freq="D")
    else:
        total_days = 1
        start_day = end_day
        delivery_days = pd.DatetimeIndex([end_day])

    # The bank must cover the physical context preceding the first origin.
    context_margin_days = (int(args.context_length) + 23) // 24 + 2
    bank_start = start_day - pd.Timedelta(days=context_margin_days)
    banks = {}
    live_manifest_audits: dict[str, object] = {}
    for zone in zones:
        if args.live_source_manifest is not None:
            sources, source_manifest_audit = load_live_source_manifest(
                args.live_source_manifest, zone=zone
            )
            banks[zone] = build_exogenous_bank(
                sources, start_day=bank_start, end_day=end_day,
                timezone=args.timezone, require_complete=True,
                require_operational_evidence=True,
            )
            live_manifest_audits[zone] = source_manifest_audit
        else:
            banks[zone] = build_default_project_bank(
                root,
                zone=zone,
                start_day=bank_start,
                end_day=end_day,
                pack=args.pack,
                weather_root=args.weather_root,
                timezone=args.timezone,
                require_complete=(
                    bool(args.require_complete_context) or args.mode == "shadow"
                ),
                require_operational_evidence=bool(args.require_production_pit),
            )

    panel = build_origin_panel(
        banks,
        targets,
        delivery_days=delivery_days,
        context_length=int(args.context_length),
        layout=args.layout,
        zones=zones,
        timezone=args.timezone,
        require_horizon_targets=(
            args.mode in {"training", "calibration"}
            and not args.allow_unresolved_final_evaluation_day
        ),
        # Missing historical CNEC/RAM remains NaN and is audited.  The LoRA
        # validator separately requires every known-future horizon value.
        require_complete_covariates=(
            bool(args.require_complete_context) or args.mode == "shadow"
        ),
    )
    panel = type(panel)(
        frame=panel.frame,
        audit={
            **dict(panel.audit),
            "pack": str(args.pack),
            "target_contracts": target_contracts,
            "canonical_target_contracts_verified": True,
            "live_source_manifests": live_manifest_audits,
            "live_sources_explicit": bool(args.live_source_manifest is not None),
            "allow_unresolved_final_evaluation_day": bool(
                args.allow_unresolved_final_evaluation_day
            ),
        },
    )
    if args.mode == "shadow":
        horizon = panel.frame.loc[panel.frame["phase"].eq("horizon")]
        target_columns = [
            column
            for column in panel.frame
            if column == "target" or column.startswith("target_")
        ]
        horizon_has_actual = bool(
            target_columns
            and horizon.loc[:, target_columns].notna().any(axis=None)
        )
        shadow_origins = pd.DatetimeIndex(
            pd.to_datetime(panel.frame["origin_timestamp"], utc=True)
        ).unique()
        shadow_days = tuple(horizon["delivery_day"].drop_duplicates().astype(str))
        if len(shadow_origins) != 1 or len(shadow_days) != 1:
            raise OriginPanelError(
                "Un panel shadow doit contenir exactement une origine et un jour."
            )
        panel = type(panel)(
            frame=panel.frame,
            audit={
                **dict(panel.audit),
                "purpose": "prospective_shadow",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "delivery_day": shadow_days[0],
                "forecast_origin_utc": pd.Timestamp(shadow_origins[0]).isoformat(),
                "forecast_origin_timezone": str(args.timezone),
                "delivery_timezones": {
                    zone: DELIVERY_TIMEZONE_BY_ZONE[zone] for zone in zones
                },
                "pack": str(args.pack),
                "horizon_actuals_present": horizon_has_actual,
                "first_publication_with_actuals_must_be_rejected": True,
            },
        )
    output = args.output if args.output.is_absolute() else root / args.output
    panel_path, audit_path = write_origin_panel(panel, output)
    payload = {
        "status": "ready",
        "mode": args.mode,
        "panel": str(panel_path),
        "audit": str(audit_path),
        "layout": args.layout,
        "pack": args.pack,
        "zones": list(zones),
        "delivery_start": str(start_day.date()),
        "delivery_end": str(end_day.date()),
        "training_days": (
            args.training_days
            if args.mode in {"training", "calibration"}
            else None
        ),
        "evaluation_days": (
            args.evaluation_days
            if args.mode in {"training", "calibration"}
            else None
        ),
        "rows": int(len(panel.frame)),
        "target_caches": target_paths,
        "production_ready": bool(panel.audit.get("production_ready")),
        "production_pit_evidence": panel.audit.get("production_pit_evidence"),
        "allow_unresolved_final_evaluation_day": bool(
            args.allow_unresolved_final_evaluation_day
        ),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ExogenousBankError, OriginPanelError) as exc:
        raise SystemExit(f"ECHEC panel exogene: {exc}") from exc


__all__ = ["build_parser", "main"]
