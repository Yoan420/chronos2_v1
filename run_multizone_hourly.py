#!/usr/bin/env python
"""Run autonomous hourly candidate configurations zone by zone.

The orchestrator does a complete local configuration preflight first.  It does
not provide any fallback: every requested zone owns an explicit configuration
and ``run_chronos2_hourly.py`` receives the matching ``--zone`` value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from chronos2_hourly.zone_live import (
    MARKET_ZONES,
    ZoneBundleError,
    canonical_zone,
    load_zone_registry,
)
from chronos2_modular.common import build_zone_configs, load_yaml


def _resolve(value: Any, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def audit_training_candidate(
    registry: Mapping[str, Any],
    *,
    registry_dir: str | Path,
    zone: str,
) -> dict[str, Any]:
    code = canonical_zone(zone)
    raw = registry.get("zones", {}).get(code)
    if not isinstance(raw, Mapping):
        raise ZoneBundleError(f"{code}: zone absente du registre")
    blockers: list[str] = []
    if not bool(raw.get("training_candidate_ready", False)):
        blockers.append("configuration non marquee training_candidate_ready")
    if raw.get("target_status") != "audited_dst_strict":
        blockers.append("cible non auditee DST-stricte")
    if raw.get("price_unit") != "EUR/MWh":
        blockers.append("unite cible differente de EUR/MWh")
    config_path = _resolve(raw.get("training_config"), Path(registry_dir).resolve())
    if config_path is None or not config_path.is_file():
        blockers.append(f"configuration d'entrainement absente: {config_path}")
        return {
            "zone": code,
            "ready": False,
            "config": str(config_path) if config_path else None,
            "blockers": blockers,
        }
    try:
        config = load_yaml(config_path)
        selected = build_zone_configs(config, [code], None, None)
    except Exception as exc:
        blockers.append(f"configuration invalide: {exc}")
        selected = []
    if len(selected) == 1:
        parsed = selected[0]
        expected_timezone = MARKET_ZONES[code].timezone
        if parsed.zone != code:
            blockers.append(f"zone configuree {parsed.zone} != {code}")
        if parsed.timezone != expected_timezone:
            blockers.append(
                f"timezone configuree {parsed.timezone} != {expected_timezone}"
            )
        if parsed.target.series != raw.get("target_series"):
            blockers.append("serie cible differente du registre audite")
        required = {str(value).lower() for value in raw.get("required_covariates", [])}
        missing = sorted(required - set(parsed.covariates))
        if missing:
            blockers.append(f"covariables requises absentes: {missing}")
    return {
        "zone": code,
        "ready": not blockers,
        "config": str(config_path),
        "blockers": blockers,
    }


def build_training_command(
    audit: Mapping[str, Any],
    *,
    refresh_data: bool = False,
    full_target_refresh: bool = False,
    data_as_of: str | None = None,
    device: str | None = None,
    local_files_only: bool = False,
) -> list[str]:
    if not bool(audit.get("ready", False)):
        raise ZoneBundleError(
            f"{audit.get('zone')}: entrainement refuse: "
            + "; ".join(audit.get("blockers", []))
        )
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("run_chronos2_hourly.py")),
        "--config",
        str(audit["config"]),
        "--zone",
        str(audit["zone"]),
    ]
    if refresh_data:
        command.append("--refresh-data")
    if full_target_refresh:
        command.append("--full-target-refresh")
    if data_as_of:
        command.extend(["--data-as-of", data_as_of])
    if device:
        command.extend(["--device", device])
    if local_files_only:
        command.append("--local-files-only")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrainement horaire multi-zone, sequentiel et fail-closed."
    )
    parser.add_argument("--registry", default="chronos2_hourly_live_zones.yaml")
    parser.add_argument("--zones", nargs="+", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    refresh = parser.add_mutually_exclusive_group()
    refresh.add_argument("--refresh-data", action="store_true")
    refresh.add_argument("--full-target-refresh", action="store_true")
    parser.add_argument("--data-as-of", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry, registry_dir = load_zone_registry(args.registry)
    requested: list[str] = []
    for value in args.zones:
        code = canonical_zone(value)
        if code not in requested:
            requested.append(code)
    audits = [
        audit_training_candidate(
            registry,
            registry_dir=registry_dir,
            zone=zone,
        )
        for zone in requested
    ]
    print(json.dumps(audits, ensure_ascii=False, indent=2))
    failed_preflight = [audit for audit in audits if not audit["ready"]]
    if failed_preflight:
        raise ZoneBundleError(
            "Preflight refuse avant tout run: "
            + ", ".join(str(audit["zone"]) for audit in failed_preflight)
        )
    if args.preflight_only:
        return 0

    failures: list[str] = []
    for audit in audits:
        command = build_training_command(
            audit,
            refresh_data=bool(args.refresh_data),
            full_target_refresh=bool(args.full_target_refresh),
            data_as_of=args.data_as_of,
            device=args.device,
            local_files_only=bool(args.local_files_only),
        )
        completed = subprocess.run(command, cwd=registry_dir, check=False)
        if completed.returncode:
            failures.append(f"{audit['zone']}={completed.returncode}")
    if failures:
        raise RuntimeError("Run(s) en echec: " + ", ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

