#!/usr/bin/env python
"""Build or audit the five-country Chronos-2 residual-load live bundle."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import os
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from chronos2_hourly.chronos_residual_load import (
    build_live_residual_load_bundle,
    planned_live_residual_load_manifest_path,
    validate_live_residual_load_bundle,
)
from chronos2_modular.common import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "chronos2_hourly_fr_residual_v1.yaml"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "experiments"
    / "chronos2_residual_load"
    / "upstream"
)
TIMEZONE = ZoneInfo("Europe/Paris")


def _delivery_day(value: str | None) -> date:
    return (
        date.fromisoformat(value)
        if value
        else pd.Timestamp.now(tz=TIMEZONE).date() + timedelta(days=1)
    )


def _cutoff(day: date) -> pd.Timestamp:
    return (
        pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
    ).tz_localize(TIMEZONE, ambiguous="raise", nonexistent="raise")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Forecast Chronos-2 des charges residuelles FR/DE/BE/NL/ES "
            "a partir des historiques ENTSO-E observes."
        )
    )
    parser.add_argument("--delivery-day", default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--validate", default=None, metavar="MANIFEST")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate:
        manifest = Path(args.validate).expanduser().resolve()
        validate_live_residual_load_bundle(manifest)
        print(manifest)
        return 0

    day = _delivery_day(args.delivery_day)
    cutoff = _cutoff(day)
    output_root = Path(args.output_root).expanduser().resolve()
    planned = planned_live_residual_load_manifest_path(
        delivery_day=day,
        runtime_cutoff=cutoff,
        output_root=output_root,
    )
    if args.plan_only:
        print(planned)
        return 0

    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    data = config.get("data")
    if not isinstance(data, dict):
        raise TypeError(f"{config_path}: data doit etre un mapping.")
    saturn_url = str(data.get("saturn_url") or "").strip()
    saturn_author = str(
        os.getenv("SATURN_AUTHOR") or data.get("saturn_author") or ""
    ).strip()
    manifest = build_live_residual_load_bundle(
        delivery_day=day,
        runtime_cutoff=cutoff,
        output_root=output_root,
        saturn_url=saturn_url,
        saturn_author=saturn_author,
        device=args.device,
        local_files_only=bool(args.local_files_only),
        batch_size=int(args.batch_size),
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
