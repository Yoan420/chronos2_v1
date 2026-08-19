#!/usr/bin/env python
"""Build one autonomous hourly training config from the zone registry."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from chronos2_hourly.zone_live import (
    build_zone_training_config,
    canonical_zone,
    load_zone_registry,
)
from chronos2_modular.common import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genere une configuration d'entrainement horaire par zone."
    )
    parser.add_argument("--registry", default="chronos2_hourly_live_zones.yaml")
    parser.add_argument("--template", default="chronos2_hourly_fr_residual_v1.yaml")
    parser.add_argument("--zone", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry, _registry_dir = load_zone_registry(args.registry)
    code = canonical_zone(args.zone)
    contract = registry["zones"][code]
    generated = build_zone_training_config(
        load_yaml(Path(args.template).expanduser().resolve()),
        zone=code,
        contract=contract,
    )
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refus d'ecraser une configuration existante: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(generated, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

