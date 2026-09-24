"""Single-day retrospective comparison, separate from every prospective ledger.

This reuses the frozen rank-16 trial recipe, not an independently selected test.
Future observations are attached only after the two forecast chains are saved.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from . import prospective_trial as trial
from .prospective_auxiliary import RAW_MODEL, QUANTILES, MARKET_COLUMNS, ResidualRecipe


class RetrospectiveComparisonError(ValueError):
    pass


RETRO_FLAGS = {**trial.FLAGS, "prospective_eligible": False,
               "historical_metrics_are_independent_test": False,
               "kind": "rank16_single_day_retrospective_comparison"}
COMPARISON_CODE = (
    "chronos2_exogenous/retrospective_trial.py",
    "chronos2_exogenous/retrospective_incumbents.py",
    "chronos2_exogenous/retrospective_reporting.py",
)


def _comparison_identity(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    root, output = Path(config["project_root"]), Path(config["output_root"])
    return {"trial_manifest": trial._file(output / "trial_manifest.json"),
            "effective_freeze_utc": manifest.get("_effective_freeze_utc", manifest["created_at_utc"]),
            "checkpoint_sha256": config["checkpoint_sha256"],
            "comparison_code": [trial._file(root / name) for name in COMPARISON_CODE]}


def _verify_zone_cache(path: Path, identity: Mapping[str, Any], *, zone: str, day: str) -> dict[str, Any]:
    record = trial._json(path)
    if (record.get("identity") != dict(identity) or record.get("zone") != zone
            or record.get("delivery_day") != day
            or any(record.get(k) != v for k, v in RETRO_FLAGS.items())):
        raise RetrospectiveComparisonError("Identite du calcul retrospectif existant divergente; aucun ecrasement.")
    for key, name in (("predictions", "predictions.csv.gz"), ("raw_future", "raw.csv.gz"),
                      ("calibration_history", "history.csv.gz"), ("auxiliary_audit", "auxiliary_audit.json")):
        if trial._verify_file(record[key]).resolve() != (path.parent / name).resolve():
            raise RetrospectiveComparisonError("Artefact retrospectif hors de son dossier.")
    source = record.get("raw_provenance", {})
    for key in ("daily_manifest", "source_raw", "input_manifest", "panel"):
        if key in source:
            trial._verify_file(source[key])
    if trial._read_frame(Path(record["predictions"]["path"])).actual.notna().any():
        raise RetrospectiveComparisonError("Le forecast retrospectif scelle ne doit pas contenir les observations du jour.")
    return record


def _sealed_day_raw(manifest: Mapping[str, Any], zone: str, day: str) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    output = Path(manifest["config"]["output_root"])
    directory = output / "days" / day / zone
    path = directory / "manifest.json"
    if not path.exists():
        if directory.exists():
            raise RetrospectiveComparisonError(f"{zone}/{day}: ancienne publication incomplete a inspecter.")
        return None, {}
    daily = trial._verify_daily(path, output=output, expected_zone=zone)
    raw = trial._read_frame(trial._verify_file(daily["raw_predictions"]))
    raw["actual"] = np.nan
    return raw, {"selection": "reuse_immutable_raw_forecast", "daily_manifest": trial._file(path),
                 "source_raw": daily["raw_predictions"], "horizon_observations_removed": True}


def _find_existing_panel(config: Mapping[str, Any], manifest: Mapping[str, Any],
                         day: str, zones: Sequence[str]) -> dict[str, Any] | None:
    output = Path(config["output_root"])
    for path in sorted((output / "captures").glob("*/input_manifest.json"), reverse=True):
        capture = trial._json(path)
        if (day not in capture.get("delivery_days", []) or not set(zones).issubset(capture.get("zones", []))
                or capture.get("known_future_covariates") != manifest["schema"]["known_future_covariates"]):
            continue
        for daily_path in (output / "days").glob("*/*/manifest.json"):
            reference = trial._json(daily_path).get("input_manifest", {})
            if reference.get("path") and Path(reference["path"]).resolve() == path.resolve():
                trial._verify_file(reference)
        for name in ("panel", "panel_audit", "seed_manifest"):
            trial._verify_file({"path": capture[f"{name}_path"], "sha256": capture[f"{name}_sha256"]})
        return {**capture, "manifest_path": str(path)}
    return None


def execute_retrospective_comparison(config: Mapping[str, Any], *, delivery_day: str,
                                     zones: Sequence[str], device: str = "auto", threads: int = 4,
                                     refresh_observations: bool = True) -> dict[str, Any]:
    from .lora_finetune import load_config, load_checkpoint
    from .evaluation import _prepare_shadow_panel
    from .prospective_auxiliary import forecast_trial_chains, fit_residual_corrector
    from .prospective_inputs import prepare_trial_inputs, _capture_target
    from .retrospective_incumbents import load_incumbent_comparison
    from .retrospective_reporting import render_retrospective_report
    from chronos2_hourly.kalman_residual import KalmanResidualConfig

    day = trial._day(delivery_day)
    origin = (pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris")
    if trial.now_utc() < origin:
        raise RetrospectiveComparisonError("Comparaison retrospective refusee avant l'origine de prevision.")
    if type(threads) is not int or not 1 <= threads <= 64:
        raise RetrospectiveComparisonError("threads doit etre entre 1 et 64.")
    manifest = trial.verify_trial(config)
    if not zones or len(zones) != len(set(zones)) or set(zones) - set(manifest["zones"]):
        raise RetrospectiveComparisonError("Selection de pays invalide.")
    output, root = Path(config["output_root"]), Path(config["project_root"])
    destination = output / "retrospective" / day
    identity = _comparison_identity(config, manifest)
    records, histories, raw_frames, provenance, needed = {}, {}, {}, {}, []
    print(f"[LoRA16] RETROSPECTIF {day} : comparaison apres coup, hors statistiques prospectives.", flush=True)
    for zone in zones:
        path = destination / zone / "forecast_manifest.json"
        if path.exists():
            records[zone] = _verify_zone_cache(path, identity, zone=zone, day=day)
            print(f"[LoRA16] {zone}: deux chaines deja calculees, reutilisation verifiee.", flush=True)
            continue
        if path.parent.exists():
            raise RetrospectiveComparisonError(f"{zone}: tentative retrospective incomplete, aucun ecrasement.")
        history_all = trial.load_history(manifest, zone)
        # Crucial: Bootstrap may already contain D (with its actual). Exclude
        # the entire target day and every later day BEFORE any auxiliary fit.
        dates = pd.DatetimeIndex(history_all.delivery_start_utc).tz_convert("Europe/Paris").date
        history = history_all.loc[dates < pd.Timestamp(day).date()].copy()
        fit_residual_corrector(history, target_day=day, recipe=ResidualRecipe(**manifest["recipe"]),
                               require_full_window=True)
        histories[zone] = history
        raw, source = _sealed_day_raw(manifest, zone, day)
        if raw is None:
            needed.append(zone)
        else:
            raw_frames[zone], provenance[zone] = raw, source
    # Inspect comparator hashes before performing any expensive new inference.
    incumbents, comparator_audit = load_incumbent_comparison(root, delivery_day=day, zones=zones)
    if needed:
        capture = _find_existing_panel(config, manifest, day, needed)
        if capture is None:
            capture = prepare_trial_inputs(
                project_root=root, output_directory=output / "retrospective_inputs" / (
                    trial.now_utc().strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]),
                delivery_days=[day], zones=needed, expected_schema=manifest["schema"])
        candidate = load_config(config["candidate_config"])
        original = pd.read_parquet(capture["panel_path"])
        original = original.loc[original.delivery_day.eq(day) & original.item_id.isin(needed)].copy()
        original.loc[original.phase.eq("horizon"), "target"] = np.nan
        panel = _prepare_shadow_panel(original, candidate)
        keys = [candidate.timestamp_column, candidate.origin_column, candidate.item_column]
        tags = original[[*keys, "phase", "delivery_day"]].copy()
        for column in (candidate.timestamp_column, candidate.origin_column):
            tags[column] = pd.to_datetime(tags[column], utc=True)
        panel = panel.merge(tags, on=keys, how="left", validate="one_to_one")
        import torch
        torch.set_num_threads(threads)
        pipeline = load_checkpoint(manifest["zones"][needed[0]]["bundle"], device_map=device)
        for zone in needed:
            print(f"[LoRA16] {zone}: inference brute retrospective (sans prix du jour).", flush=True)
            group = panel.loc[panel.item_id.eq(zone)]
            raw = trial._predict_group(group, candidate, pipeline)
            raw["actual"] = np.nan
            raw_frames[zone] = raw
            provenance[zone] = {"selection": "retrospective_frozen_checkpoint_inference",
                                "input_manifest": trial._file(Path(capture["manifest_path"])),
                                "panel": {"path": capture["panel_path"], "sha256": capture["panel_sha256"]},
                                "horizon_observations_removed": True}
        del pipeline
    for zone in zones:
        if zone in records:
            continue
        print(f"[LoRA16] {zone}: correcteur et Kalman, calibration strictement anterieure au {day}.", flush=True)
        result = forecast_trial_chains(histories[zone], raw_frames[zone],
            recipe=ResidualRecipe(**manifest["recipe"]), kalman_config=KalmanResidualConfig(**manifest["kalman_config"]))
        frame = result.kalman_future.copy()
        frame["zone"], frame["actual"], frame["prospective_eligible"] = zone, np.nan, False
        directory = destination / zone
        directory.mkdir(parents=True, exist_ok=False)
        trial._write_json(directory / "auxiliary_audit.json", result.audit)
        record = {**RETRO_FLAGS, "identity": identity, "zone": zone, "delivery_day": day,
                  "computed_at_utc": trial.now_utc().isoformat(), "raw_provenance": provenance[zone],
                  "training_end_day": (pd.Timestamp(day) - pd.Timedelta(days=1)).date().isoformat(),
                  "training_lookback_days": 365, "target_observations_used": 0,
                  "calibration_history": trial._write_frame(directory / "history.csv.gz", histories[zone]),
                  "raw_future": trial._write_frame(directory / "raw.csv.gz", raw_frames[zone]),
                  "predictions": trial._write_frame(directory / "predictions.csv.gz", frame),
                  "auxiliary_audit": trial._file(directory / "auxiliary_audit.json")}
        trial._write_json(directory / "forecast_manifest.json", record)
        records[zone] = record
    # Price reads are deliberately AFTER the forecasts have been saved.
    # Missing observations affect scores, never the ability to produce forecasts.
    report_dir = destination / "reports" / (trial.now_utc().strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    report_dir.mkdir(parents=True, exist_ok=False)
    frames, observations = [], {}
    for zone in zones:
        frame = trial._read_frame(trial._verify_file(records[zone]["predictions"]))
        target, audit = _capture_target(root=root, zone=zone, start=day, end=day, refresh=refresh_observations)
        snapshot = trial._write_frame(report_dir / f"{zone}_observations.csv.gz",
            target.rename("target").rename_axis("timestamp").reset_index())
        observed_hours = int(np.isfinite(target.to_numpy(float)).sum())
        observations[zone] = {**audit, "snapshot": snapshot, "observed_hours": observed_hours,
                              "used_for_prediction": False, "used_for_calibration": False}
        frames.append(trial._attach_actuals(frame, target))
    comparison = pd.concat(frames, ignore_index=True).merge(
        incumbents, on=["zone", "delivery_start_utc"], how="left", validate="one_to_one")
    evidence = trial._write_frame(report_dir / "comparison.csv.gz", comparison)
    meta = {**RETRO_FLAGS, "delivery_day": day, "training_lookback_days": 365,
            "evaluation_days": 1, "comparator_audit": comparator_audit,
            "message": "Comparaison retrospective sur une journee; aucun nouveau backtest 365 jours."}
    report = render_retrospective_report(comparison, report_dir / "lora16_comparison.html", metadata=meta)
    trial._write_json(report_dir / "report_manifest.json", {**meta, "identity": identity,
        "created_at_utc": trial.now_utc().isoformat(), "observations": observations,
        "forecasts": {zone: trial._file(destination / zone / "forecast_manifest.json") for zone in zones},
        "evidence": evidence, "report": trial._file(report)})
    return {"status": "retrospective_comparison_completed", "delivery_day": day,
            "report": str(report), "forecast_directory": str(destination),
            "observed_hours": {zone: a["observed_hours"] for zone, a in observations.items()}, **RETRO_FLAGS}
