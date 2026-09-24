#!/usr/bin/env python
"""Build the causal residual-corrected prefix required by weather Kalman.

This is a bootstrap utility, not a daily forecast step.  It reuses the frozen
residual recipe from the auxiliary lab and refits it prequentially: every
forecast block sees labels from strictly earlier local delivery days.  Daily
Forecast.ps1 runs then combine this immutable prefix with the latest issued
Statistics rows; no historical refit is needed again.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Sequence
import uuid

from auxiliary_lab import load_lab_config
from auxiliary_lab.runner import _run_residual


ROOT = Path(__file__).resolve().parent
ZONE_TIMEZONES = {
    "FR": "Europe/Paris",
    "DE": "Europe/Berlin",
    "BE": "Europe/Brussels",
    "NL": "Europe/Amsterdam",
    "ES": "Europe/Madrid",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialise le prefixe prequentiel residual_corrected."
    )
    parser.add_argument("--zones", nargs="+", default=list(ZONE_TIMEZONES))
    parser.add_argument("--delivery-day", required=True)
    parser.add_argument(
        "--base-config",
        default="config/auxiliary_lab_kalman_weather.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="data/pit/kalman_weather",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _zones(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        result.extend(item.strip().upper() for item in str(value).split(","))
    result = [item for item in result if item]
    unknown = sorted(set(result).difference(ZONE_TIMEZONES))
    if unknown or not result or len(result) != len(set(result)):
        raise ValueError(f"Selection de zones invalide: {values!r}.")
    return tuple(result)


def _source_run(zone: str, delivery_day: str) -> Path:
    relative = {
        "FR": f"runs/live/fr_day_ahead_{delivery_day}",
        "DE": f"runs/live/de/de_day_ahead_{delivery_day}",
        "BE": f"runs/live/be/be_day_ahead_{delivery_day}",
        "NL": f"runs/live/nl_mkonline_v1/nl_day_ahead_{delivery_day}",
        "ES": f"runs/live/es/es_day_ahead_{delivery_day}",
    }[zone]
    path = (ROOT / relative).resolve()
    required = (
        path / "backtest_hourly_oof.csv.gz",
        path / "statistics_history_hourly.csv.gz",
        path / "inputs" / "aligned_inputs.csv.gz",
    )
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Archive source incomplete pour {zone}: {', '.join(missing)}"
        )
    return path


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def materialize_zone(
    *,
    zone: str,
    delivery_day: str,
    base_config: Path,
    output_dir: Path,
    overwrite: bool,
) -> tuple[Path, Path]:
    lower = zone.casefold()
    history_output = output_dir / f"{lower}_residual_corrected_prequential.csv.gz"
    audit_output = output_dir / f"{lower}_residual_corrected_prequential.audit.json"
    if history_output.is_file() and audit_output.is_file() and not overwrite:
        audit = json.loads(audit_output.read_text(encoding="utf-8"))
        if (
            isinstance(audit, dict)
            and audit.get("causality_violations") == 0
            and audit.get("selected_recipe_is_frozen") is True
        ):
            print(f"[{zone}] SKIP | prefixe prequentiel deja audite", flush=True)
            return history_output, audit_output
        raise RuntimeError(
            f"[{zone}] artefact existant non auditable; utilisez --overwrite."
        )

    base = load_lab_config(base_config)
    model = base.models["residual_corrector"]
    options = dict(model.options)
    bridge = dict(options["prequential_bridge"])
    bridge["history_prefix_path"] = (
        ROOT
        / "runs"
        / f"chronos2_hourly_{lower}_residual_extended_v1"
        / "inputs"
        / "chronos_oof_extended.csv.gz"
    ).resolve()
    if not bridge["history_prefix_path"].is_file():
        raise FileNotFoundError(bridge["history_prefix_path"])
    options["prequential_bridge"] = bridge
    zone_model = replace(model, options=options)

    temporary_parent = ROOT / "runs" / "tmp" / "kalman_weather_upstream"
    temporary_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{lower}_", dir=temporary_parent)
    ).resolve()
    zone_config = replace(
        base,
        experiment_id=f"kalman_weather_upstream_{lower}_{delivery_day}",
        source_run=_source_run(zone, delivery_day),
        output_directory=staging,
        timezone=ZONE_TIMEZONES[zone],
        models={**base.models, "residual_corrector": zone_model},
    )
    try:
        print(f"[{zone}] Construction prequentielle...", flush=True)
        _run_residual(
            zone_config,
            zone_model,
            staging,
            build_kalman_history=True,
        )
        model_dir = staging / "models" / "residual_corrector"
        history = model_dir / "kalman_upstream_history.csv.gz"
        audit = model_dir / "kalman_upstream_prequential_audit.json"
        if not history.is_file() or not audit.is_file():
            raise RuntimeError(f"[{zone}] artefacts prequentiels non produits.")
        payload = json.loads(audit.read_text(encoding="utf-8"))
        if (
            payload.get("causality_violations") != 0
            or payload.get("selected_recipe_is_frozen") is not True
            or payload.get("fit_label_rule")
            != "local_delivery_day < block_start_day"
        ):
            raise RuntimeError(f"[{zone}] audit prequentiel invalide.")
        _atomic_copy(history, history_output)
        _atomic_copy(audit, audit_output)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"[{zone}] Prefixe: {history_output}", flush=True)
    return history_output, audit_output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    zones = _zones(args.zones)
    base_config = Path(args.base_config).expanduser()
    base_config = (
        base_config if base_config.is_absolute() else ROOT / base_config
    ).resolve()
    output_dir = Path(args.output_dir).expanduser()
    output_dir = (
        output_dir if output_dir.is_absolute() else ROOT / output_dir
    ).resolve()
    for zone in zones:
        materialize_zone(
            zone=zone,
            delivery_day=str(args.delivery_day),
            base_config=base_config,
            output_dir=output_dir,
            overwrite=bool(args.overwrite),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "materialize_zone", "parse_args"]
