"""Isolated, byte-pinned inputs for the original LoRA prospective *trial*.

Historical Saturn as-of reconstructions remain research evidence.  Capturing
their bytes today does not make them prospectively captured at an old origin.
This module deliberately does not use or issue production live-source proofs.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from chronos2_modular.common import load_yaml
from chronos2_modular.saturn import create_saturn_client, fetch_saturn_series_from_client
from .feature_bank import build_default_project_bank, default_project_sources, delivery_utc_index
from .panel import OriginPanel, build_origin_panel, load_target_cache, write_origin_panel


class ProspectiveInputError(ValueError):
    """Input capture was incomplete, mutable, or outside the isolated trial."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _day(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is not None or timestamp != timestamp.normalize():
        raise ProspectiveInputError(f"Date civile invalide: {value!r}.")
    return timestamp.date().isoformat()


def build_trial_input_plan(
    *, project_root: str | Path, output_directory: str | Path,
    delivery_days: Sequence[str], zones: Sequence[str] = ("FR", "DE", "BE", "NL"),
    python_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Read-only plan; only source-specific materializers will access the network."""
    root = Path(project_root).resolve()
    output = Path(output_directory).expanduser()
    output = (output if output.is_absolute() else root / output).resolve()
    experiments = root / "runs" / "experiments"
    if output == experiments or not output.is_relative_to(experiments):
        raise ProspectiveInputError("Le dossier doit etre un nouveau sous-dossier de runs/experiments.")
    if output.exists():
        raise ProspectiveInputError(f"Dossier deja present, aucun ecrasement autorise: {output}.")
    codes = tuple(dict.fromkeys(str(z).upper() for z in zones))
    if not codes or set(codes).difference({"FR", "DE", "BE", "NL"}):
        raise ProspectiveInputError("Zones du candidat original: FR, DE, BE, NL.")
    days = tuple(sorted(set(_day(d) for d in delivery_days)))
    if not days:
        raise ProspectiveInputError("Au moins une journee est requise.")
    bank_start = (pd.Timestamp(days[0]) - pd.Timedelta(days=88)).date().isoformat()
    end = days[-1]
    entries: dict[str, dict[str, Any]] = {}
    source_audits: dict[str, dict[str, Any]] = {}
    for code in codes:
        for source in default_project_sources(root, zone=code, pack="full"):
            audit = json.loads(source.audit_path.read_text(encoding="utf-8"))
            expected = audit.get("output_sha256", audit.get("sha256"))
            # JAO's aggregate uses the canonical parquet hash key.
            expected = expected or audit.get("parquet_sha256")
            actual = _sha256(source.path)
            if expected != actual:
                raise ProspectiveInputError(f"SHA source/audit divergent: {source.path}.")
            source_audits[source.name] = audit
            for path in (source.path, source.audit_path):
                relative = path.resolve().relative_to(root).as_posix()
                entries[relative] = {"path": str(path), "relative_path": relative, "sha256": _sha256(path)}

    # Copy only daily JAO bundles needed for this context, not the large store.
    jao_root = root / "data" / "pit" / "jao_core_flowbased"
    from materialize_jao_core_flowbased import _existing_partition
    for day in pd.date_range(bank_start, end):
        token = day.date().isoformat()
        if _existing_partition(jao_root, day.date()) is None:
            continue
        for tail in (
            f"raw/initialComputation/{token}.json.gz",
            f"raw/initialComputation/{token}.audit.json",
            f"normalised/{token}.parquet", f"daily_features/{token}.parquet",
        ):
            path = jao_root / tail
            relative = path.relative_to(root).as_posix()
            entries[relative] = {"path": str(path), "relative_path": relative, "sha256": _sha256(path)}
    weather_starts = {str(a["start_day"]) for n, a in source_audits.items() if n.startswith("weather_")}
    if len(weather_starts) != 1:
        raise ProspectiveInputError("Les banques meteo n'ont pas la meme date de debut.")
    python = str(Path(python_executable or sys.executable).resolve())
    isolated = output / "source_root"
    commands = [
        [python, str(root / "materialize_saturn_kalman_weather.py"),
         "--start-day", weather_starts.pop(), "--end-day", end, "--zones", *codes,
         "--output-dir", str(isolated / "data/pit/kalman_weather"),
         "--series-workers", "2", "--day-workers", "4"],
        [python, str(root / "materialize_saturn_kalman_fuel.py"),
         "--start-day", str(source_audits["fuel_market"]["start_day"]),
         "--residual-start-day", str(source_audits["residual_load"]["start_day"]),
         "--end-day", end, "--output-dir", str(isolated / "data/pit/kalman_hybrid"),
         "--series-workers", "2", "--day-workers", "4"],
        [python, str(root / "materialize_jao_core_flowbased.py"),
         "--start-day", bank_start, "--end-day", end, "--training-days", "0",
         "--evaluation-days", "1", "--future-days", "0", "--workers", "2",
         "--output-root", str(isolated / "data/pit/jao_core_flowbased")],
    ]
    return {
        "schema_version": 1, "kind": "original_lora_trial_input_plan",
        "project_root": str(root), "output_directory": str(output),
        "source_root": str(isolated), "zones": list(codes), "delivery_days": list(days),
        "bank_start_day": bank_start, "bank_end_day": end, "context_length": 2048,
        "source_history": list(entries.values()), "commands": commands,
        "diagnostic_only": True, "production_pit_evidence": False,
        "promotion_eligible": False, "network_executed": False,
    }


def _copy_pinned_sources(plan: Mapping[str, Any]) -> None:
    isolated = Path(plan["source_root"])
    for entry in plan["source_history"]:
        source = Path(entry["path"])
        if _sha256(source) != entry["sha256"]:
            raise ProspectiveInputError(f"Source modifiee depuis le plan: {source}.")
        destination = isolated / entry["relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ProspectiveInputError(f"Capture deja presente: {destination}.")
        shutil.copyfile(source, destination)
        if _sha256(destination) != entry["sha256"] or _sha256(source) != entry["sha256"]:
            raise ProspectiveInputError(f"Source modifiee pendant la copie: {source}.")


def _capture_target(
    *, root: Path, zone: str, start: str, end: str, refresh: bool,
) -> tuple[pd.Series, dict[str, Any]]:
    from run_chronos2_exogenous_panel import _canonical_target_path
    cache, contract = _canonical_target_path(root, zone)
    before = _sha256(cache)
    index = delivery_utc_index(start, end, timezone="Europe/Paris")
    started = _now()
    if refresh:
        data = load_yaml(Path(contract["base_config"])).get("data", {})
        client = create_saturn_client(
            str(data.get("saturn_url", "")),
            str(os.getenv("SATURN_AUTHOR") or data.get("saturn_author", "")),
        )
        values = fetch_saturn_series_from_client(
            client, contract["series"], index[0] - pd.Timedelta(hours=2),
            index[-1] + pd.Timedelta(hours=2), "Europe/Paris",
            naive_timezone=contract["naive_timezone"], nocache=True, live=True,
        )
    else:
        values = load_target_cache(cache, zone=zone)
    completed = _now()
    if _sha256(cache) != before:
        raise ProspectiveInputError(f"Cache canonique modifie pendant la capture: {cache}.")
    timestamps = pd.DatetimeIndex(values.index)
    if timestamps.tz is None or timestamps.has_duplicates:
        raise ProspectiveInputError(f"{zone}: cible sans fuseau ou heures dupliquees.")
    target = pd.Series(pd.to_numeric(values, errors="raise").to_numpy(dtype=float),
                       index=timestamps.tz_convert("UTC"), name="target").reindex(index)
    if np.isinf(target.to_numpy()).any():
        raise ProspectiveInputError(f"{zone}: cible infinie.")
    return target, {
        **contract, "canonical_source_cache_sha256": before,
        "capture_started_at_utc": started, "capture_completed_at_utc": completed,
        "fresh_api_read": bool(refresh), "nocache": bool(refresh),
        "source_cache_modified": False, "fallback_used": False,
        "historical_publication_times_verified": False,
    }


def prepare_trial_inputs(
    *, project_root: str | Path, output_directory: str | Path,
    delivery_days: Sequence[str], zones: Sequence[str] = ("FR", "DE", "BE", "NL"),
    python_executable: str | Path | None = None, refresh: bool = True,
    expected_schema: Mapping[str, Any] | None = None,
    command_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Capture once into a new directory; errors preserve evidence for inspection.

    ``refresh=False`` is an offline diagnostic only, never a fresh daily capture.
    Labels are saved separately and removed from every prediction horizon. The
    caller must enforce its own emission deadline and reject resolved deliveries.
    """
    plan = build_trial_input_plan(
        project_root=project_root, output_directory=output_directory,
        delivery_days=delivery_days, zones=zones, python_executable=python_executable,
    )
    root, output, isolated = (Path(plan[k]) for k in ("project_root", "output_directory", "source_root"))
    output.mkdir(parents=True, exist_ok=False)
    started = _now()
    _write_new_json(output / "seed_manifest.json", {**plan, "capture_started_at_utc": started})
    _copy_pinned_sources(plan)
    if refresh:
        runner = command_runner or subprocess.run
        for command in plan["commands"]:
            runner(command, check=True, cwd=root, shell=False)
    banks = {
        zone: build_default_project_bank(
            isolated, zone=zone, pack="full", start_day=plan["bank_start_day"],
            end_day=plan["bank_end_day"], require_complete=True,
            require_operational_evidence=False,
        ) for zone in plan["zones"]
    }
    targets: dict[str, pd.Series] = {}
    target_audits: dict[str, Any] = {}
    (output / "targets").mkdir()
    for zone in plan["zones"]:
        target, audit = _capture_target(root=root, zone=zone, start=plan["bank_start_day"],
                                        end=plan["bank_end_day"], refresh=refresh)
        path = output / "targets" / f"{zone.lower()}_canonical.csv.gz"
        target.rename_axis("timestamp").reset_index().to_csv(path, index=False, compression="gzip")
        target.attrs.update(source_path=str(path), source_sha256=_sha256(path))
        targets[zone] = target
        target_audits[zone] = {**audit, "snapshot_path": str(path), "snapshot_sha256": _sha256(path)}
    built = build_origin_panel(
        banks, targets, delivery_days=plan["delivery_days"], context_length=2048,
        layout="per_zone", zones=plan["zones"], require_horizon_targets=False,
        require_complete_covariates=True,
    )
    frame = built.frame.copy()
    value_schemas = {}
    for zone, bank in banks.items():
        lower = zone.lower()
        renames = {
            f"{lower}_temperature_fcst": "local_temperature_fcst",
            f"{lower}_wind_generation_fcst": "local_wind_generation_fcst",
            f"{lower}_solar_generation_fcst": "local_solar_generation_fcst",
        }
        value_schemas[zone] = tuple(renames.get(c, c) for c in bank.columns_for("chronos", include_quality=False))
    if expected_schema is not None:
        columns = tuple(expected_schema.get("known_future_covariates", ()))
        if len(columns) != len(set(columns)) or any(
            set(values) != set(columns) for values in value_schemas.values()
        ):
            raise ProspectiveInputError("Schema des covariables different de l'artefact LoRA fige.")
        if (expected_schema.get("context_length") != 2048
                or expected_schema.get("target_columns") != ["target"]
                or expected_schema.get("past_only_covariates", []) != []):
            raise ProspectiveInputError("Contrat de contexte/cible incompatible avec le rang 16 original.")
    available_labels: dict[str, dict[str, int]] = {}
    for day in plan["delivery_days"]:
        horizon = frame.loc[frame["delivery_day"].eq(day) & frame["phase"].eq("horizon")]
        available_labels[day] = {z: int(horizon.loc[horizon["item_id"].eq(z), "target"].notna().sum())
                                 for z in plan["zones"]}
    frame.loc[frame["phase"].eq("horizon"), "target"] = np.nan
    completed = _now()
    ledger = {
        "purpose": "original_lora_isolated_research_trial", "diagnostic_only": True,
        "production_pit_evidence": False, "promotion_eligible": False,
        "production_pipeline": False, "production_ready": False,
        "fresh_source_refresh": bool(refresh),
        "historical_context_evidence": "reconstructed_asof_not_prospective_capture",
        "capture_started_at_utc": started, "capture_completed_at_utc": completed,
        "target_snapshots": target_audits, "horizon_labels_available_at_capture": available_labels,
        "known_future_covariates": list(expected_schema["known_future_covariates"] if expected_schema else value_schemas[plan["zones"][0]]),
        "prediction_horizon_labels_removed": True,
        "capture_after_origin_by_day": {
            day: pd.Timestamp(completed) > (pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris")
            for day in plan["delivery_days"]
        },
    }
    panel, audit_path = write_origin_panel(OriginPanel(frame=frame, audit={**built.audit, **ledger}), output / "panel.parquet")
    # A concurrent operational refresh cannot silently change a pinned seed.
    for entry in plan["source_history"]:
        if _sha256(Path(entry["path"])) != entry["sha256"]:
            raise ProspectiveInputError(f"Source historique modifiee pendant le travail: {entry['path']}.")
    manifest = {
        **ledger, "schema_version": 1, "kind": "original_lora_trial_inputs",
        "zones": plan["zones"], "delivery_days": plan["delivery_days"],
        "panel_path": str(panel), "panel_sha256": _sha256(panel),
        "panel_audit_path": str(audit_path), "panel_audit_sha256": _sha256(audit_path),
        "seed_manifest_path": str(output / "seed_manifest.json"),
        "seed_manifest_sha256": _sha256(output / "seed_manifest.json"),
        "source_root": str(isolated), "status": "ready",
    }
    manifest_path = output / "input_manifest.json"
    _write_new_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


__all__ = ["ProspectiveInputError", "build_trial_input_plan", "prepare_trial_inputs"]
