#!/usr/bin/env python
"""Build isolated research LoRA panels from original NOAA forecasts and prices.

No proxy is assigned a Saturn alias; no operational cache or gate is changed.
The price caches preserve the canonical series identity but do not prove
historical price publication times. Both production evidence flags stay false.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import uuid

import pandas as pd

from auxiliary_lab.noaa_gfs import _output_reservations, _publish_pair
from chronos2_exogenous.feature_bank import (
    ParquetFeatureSource, build_exogenous_bank, delivery_utc_index,
)
from chronos2_exogenous.panel import OriginPanel, OriginPanelError, build_origin_panel, load_target_cache
from run_chronos2_exogenous_panel import _canonical_target_path

ROOT = Path(__file__).resolve().parent
ZONES = ("FR", "DE", "BE", "NL")
VARIABLES = ("temperature_2m_c", "wind_speed_100m_ms", "shortwave_radiation_wm2")
WEATHER_DEFAULT = "runs/experiments/chronos2_exogenous_public_weather_v1/history/aggregate/noaa_gfs_weather.parquet"
EXPERIMENT = "runs/experiments/chronos2_exogenous_noaa_lora_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def panel_period(*, mode: str, end_day: str, context_length: int = 2048) -> dict:
    if mode not in {"training", "calibration"} or isinstance(context_length, bool) or context_length < 1:
        raise OriginPanelError("Mode ou contexte NOAA invalide.")
    end = pd.Timestamp(end_day)
    if end.tzinfo is not None or end != end.normalize():
        raise OriginPanelError("La fin doit etre une date civile sans fuseau.")
    total = 730 if mode == "training" else 1095
    start = end - pd.Timedelta(days=total - 1)
    margin = (context_length + 23) // 24 + 2
    return {
        "mode": mode, "start_day": str(start.date()), "end_day": str(end.date()),
        "origin_days": total, "weather_start_day": str((start - pd.Timedelta(days=margin)).date()),
        "context_length": context_length, "holdout_days": 365,
        "holdout_start_day": str((end - pd.Timedelta(days=364)).date()),
    }


def load_canonical_targets(root: Path, zones: tuple[str, ...], period: dict):
    start = delivery_utc_index(period["start_day"], period["start_day"])[0]
    end = delivery_utc_index(period["end_day"], period["end_day"])[-1]
    required = pd.date_range(start - pd.Timedelta(hours=period["context_length"]), end, freq="h")
    targets, contracts, coverage = {}, {}, {}
    for zone in zones:
        path, contract = _canonical_target_path(root, zone)
        target = load_target_cache(path, zone=zone)
        missing = int(target.reindex(required).isna().sum())
        if missing:
            raise OriginPanelError(f"{zone}: {missing} heures cibles canoniques manquantes.")
        targets[zone], contracts[zone] = target, contract
        coverage[zone] = {"path": str(path), "required_hours": len(required), "missing_hours": missing,
                          "sha256": target.attrs["source_sha256"]}
    return targets, contracts, coverage


def noaa_source(weather: Path, audit: Path, zone: str) -> ParquetFeatureSource:
    # Common source name keeps the four per-zone schemas (including quality
    # columns) identical, while each mapping reads only its own country's data.
    return ParquetFeatureSource(
        name="noaa_gfs_weather", family="weather", path=weather, audit_path=audit,
        value_columns={f"local_gfs_{name}": f"{zone.lower()}_gfs_{name}" for name in VARIABLES},
        timestamp_column="delivery_start_utc", cutoff_column="cutoff_utc",
        information_time_columns=("run_init_utc", "publication_max_utc"),
        age_column="publication_max_utc",
    )


def build_noaa_panel(*, root: Path, weather: Path, weather_audit: Path, output: Path,
                     mode: str, end_day: str, zones=ZONES, context_length: int = 2048) -> dict:
    zones = tuple(str(zone).upper() for zone in zones)
    if not zones or len(set(zones)) != len(zones) or set(zones) - set(ZONES):
        raise OriginPanelError("Zones NOAA uniques parmi FR, DE, BE, NL attendues.")
    period = panel_period(mode=mode, end_day=end_day, context_length=context_length)
    targets, contracts, coverage = load_canonical_targets(root, zones, period)
    output = output.resolve()
    audit_output = output.with_suffix(output.suffix + ".audit.json")
    with _output_reservations(output, audit_output):
        banks = {zone: build_exogenous_bank(
            (noaa_source(weather, weather_audit, zone),),
            start_day=period["weather_start_day"], end_day=period["end_day"],
            require_complete=True,
        ) for zone in zones}
        panel = build_origin_panel(
            banks, targets, delivery_days=pd.date_range(period["start_day"], period["end_day"], freq="D"),
            context_length=context_length, layout="per_zone", zones=zones,
            require_horizon_targets=True, require_complete_covariates=True,
        )
        if panel.audit["production_ready"]:
            raise OriginPanelError("NOAA historique ne doit pas declarer une capture prospective.")
        panel = OriginPanel(panel.frame, {
            **panel.audit, "pack": "noaa_weather", "canonical_target_contracts_verified": True,
            "target_contracts": contracts, "target_publication_evidence": "not_attested_latest_canonical_cache",
            "purpose": "noaa_lora_research", "period": period,
            "production_pipeline_evidence": False, "promotion_eligible": False,
            "allow_unresolved_final_evaluation_day": False,
            "noaa_materializer_audit_path": str(weather_audit.resolve()),
            "noaa_materializer_audit_sha256": file_sha256(weather_audit),
            "panel_builder_sha256": file_sha256(Path(__file__)),
        })
        token = uuid.uuid4().hex
        stage = output.with_name(output.name + f".{token}.tmp")
        audit_stage = audit_output.with_name(audit_output.name + f".{token}.tmp")
        try:
            panel.frame.to_parquet(stage, index=False)
            reloaded = pd.read_parquet(stage)
            pd.testing.assert_frame_equal(panel.frame, reloaded)
            del reloaded
            metadata = {**panel.audit, "panel_path": str(output), "panel_sha256": file_sha256(stage)}
            with audit_stage.open("x", encoding="utf-8") as handle:
                json.dump(metadata, handle, sort_keys=True, indent=2, allow_nan=False)
                handle.write("\n")
            _publish_pair(stage, audit_stage, output, audit_output)
        finally:
            stage.unlink(missing_ok=True)
            audit_stage.unlink(missing_ok=True)
    return {"status": "complete", "panel": str(output), "audit": str(audit_output),
            "period": period, "zones": list(zones), "rows": len(panel.frame),
            "production_ready": False, "target_coverage": coverage}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("training", "calibration"), default="training")
    parser.add_argument("--end-day", default="2026-09-02")
    parser.add_argument("--zones", nargs="+", choices=ZONES, default=ZONES)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--weather", type=Path, default=Path(WEATHER_DEFAULT))
    parser.add_argument("--weather-audit", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", action="store_true", help="Controle les cibles et les dates sans ecrire ni charger la meteo.")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    resolve = lambda path: path.resolve() if path.is_absolute() else (root / path).resolve()
    period = panel_period(mode=args.mode, end_day=args.end_day, context_length=args.context_length)
    if args.plan:
        _, _, coverage = load_canonical_targets(root, tuple(args.zones), period)
        print(json.dumps({"period": period, "target_coverage": coverage, "mutates_files": False}, indent=2))
        return 0
    name = "training_panel.parquet" if args.mode == "training" else "residual_calibration_panel.parquet"
    weather = resolve(args.weather)
    result = build_noaa_panel(
        root=root, weather=weather, weather_audit=resolve(args.weather_audit) if args.weather_audit else weather.with_suffix(".manifest.json"),
        output=resolve(args.output or Path(EXPERIMENT) / "inputs" / name), mode=args.mode,
        end_day=args.end_day, zones=args.zones, context_length=args.context_length,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
