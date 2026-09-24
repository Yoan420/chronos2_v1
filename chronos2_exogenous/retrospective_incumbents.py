"""Read-only, checksum-checked comparators from an existing Forecast batch.

These are published operational predictions, not observations or calibration
inputs. A current export manifest identifies the residual stage but does not
prove the upstream neural checkpoint or the time of first forecast issuance.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import stat
from typing import Any, Sequence

import numpy as np
import pandas as pd


class IncumbentComparisonError(ValueError):
    """An existing comparator is malformed, inconsistent or tampered with."""


_STAGES = {"autonomous": "residual_corrected", "kalman": "residual_kalman"}
_LABELS = {"autonomous": "Forecast existant — autonome (correcteur résiduel)",
           "kalman": "Forecast existant — correcteur résiduel + Kalman"}
_PROVENANCE_WARNING = (
    "Le manifeste d'export identifie l'étape résiduelle, pas le checkpoint neural "
    "amont ni l'heure de première émission : aucun statut LoRA ou prospectif n'est déduit."
)


def _hours(delivery_day: str, timezone: str) -> pd.DatetimeIndex:
    if not isinstance(delivery_day, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", delivery_day):
        raise IncumbentComparisonError("Une date civile YYYY-MM-DD est requise.")
    try:
        day = date.fromisoformat(delivery_day)
        start = pd.Timestamp(day).tz_localize(timezone)
        end = pd.Timestamp(day + timedelta(days=1)).tz_localize(timezone)
    except (ValueError, TypeError, KeyError) as exc:
        raise IncumbentComparisonError(f"Date ou fuseau invalide : {delivery_day}/{timezone}.") from exc
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def _check_plain_path(path: Path, project_root: Path) -> None:
    """Reject symlinks and Windows junctions, including parent directories."""
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise IncumbentComparisonError(f"Comparateur hors du projet : {path}.") from exc
    for component in (project_root, *reversed(path.parents), path):
        if component != project_root and not component.is_relative_to(project_root):
            continue
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or (
            getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise IncumbentComparisonError(f"Lien ou reparse point interdit : {component}.")


def _unique_json(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IncumbentComparisonError(f"Clé JSON dupliquée dans le manifeste : {key}.")
        result[key] = value
    return result


def _load_export(
    entry: dict[str, Any], *, batch_root: Path, project_root: Path,
    delivery_day: str, zone: str, variant: str, hours: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prefix = f"{zone}/{variant}"
    if entry.get("source_model") != _STAGES[variant]:
        raise IncumbentComparisonError(f"{prefix}: source_model inattendu dans le manifeste.")
    expected = f"{zone.lower()}/{variant}/forecast_{zone.lower()}_{delivery_day}_{variant}.csv"
    csv_entry = entry.get("csv")
    if not isinstance(csv_entry, dict) or not isinstance(csv_entry.get("path"), str):
        raise IncumbentComparisonError(f"{prefix}: référence CSV absente ou invalide.")
    if csv_entry["path"].replace("\\", "/") != expected:
        raise IncumbentComparisonError(f"{prefix}: chemin CSV non conforme au pays/date/variante.")
    expected_sha = csv_entry.get("sha256")
    if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise IncumbentComparisonError(f"{prefix}: empreinte CSV invalide.")
    path = batch_root / expected
    _check_plain_path(path, project_root)
    if not path.is_file():
        raise IncumbentComparisonError(f"{prefix}: CSV référencé absent : {path}.")
    # Hash and parse the same bytes, even if a concurrent export refresh occurs.
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha:
        raise IncumbentComparisonError(f"{prefix}: SHA256 du CSV divergent.")
    try:
        frame = pd.read_csv(BytesIO(raw))
    except (ValueError, pd.errors.ParserError, UnicodeError) as exc:
        raise IncumbentComparisonError(f"{prefix}: CSV illisible.") from exc
    required = {"zone", "forecast_variant", "source_model", "delivery_start_utc", "q10", "q50", "q90"}
    if not required.issubset(frame.columns):
        raise IncumbentComparisonError(f"{prefix}: colonnes CSV manquantes : {sorted(required - set(frame.columns))}.")
    for column, expected_value in (("zone", zone), ("forecast_variant", variant), ("source_model", _STAGES[variant])):
        if frame.empty or not frame[column].eq(expected_value).all():
            raise IncumbentComparisonError(f"{prefix}: identité CSV divergente ({column}).")
    try:
        stamps = [pd.Timestamp(value) for value in frame["delivery_start_utc"]]
        if any(pd.isna(value) or value.tzinfo is None for value in stamps):
            raise ValueError("timestamp sans fuseau")
        index = pd.DatetimeIndex(pd.to_datetime(stamps, utc=True))
    except (ValueError, TypeError) as exc:
        raise IncumbentComparisonError(f"{prefix}: timestamps UTC invalides.") from exc
    if index.has_duplicates or not index.sort_values().equals(hours):
        raise IncumbentComparisonError(f"{prefix}: couverture du jour physique invalide ({len(hours)} heures attendues).")
    try:
        quantiles = frame[["q10", "q50", "q90"]].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise IncumbentComparisonError(f"{prefix}: quantiles non numériques.") from exc
    if not np.isfinite(quantiles).all() or (quantiles[:, 0] > quantiles[:, 1]).any() or (quantiles[:, 1] > quantiles[:, 2]).any():
        raise IncumbentComparisonError(f"{prefix}: quantiles non finis ou croisés.")
    result = pd.DataFrame(quantiles, index=index, columns=[f"incumbent_{variant}__q{q}" for q in (10, 50, 90)])
    audit = {"zone": zone, "variant": variant, "status": "available",
             "source_model": _STAGES[variant], "label": _LABELS[variant],
             "csv": {"path": str(path), "sha256": digest}, "hours": len(hours),
             "upstream_checkpoint_verified": False, "prospective_issuance_verified": False,
             "warnings": [_PROVENANCE_WARNING]}
    return result.reindex(hours), audit


def load_incumbent_comparison(
    project_root: Path, *, delivery_day: str, zones: Sequence[str],
    timezone: str = "Europe/Paris",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return physical hourly forecasts and an audit, without writing anything.

    Missing manifests or variants produce NaN comparator columns. A manifest
    which references a missing, modified or invalid export fails closed. No
    actual-price, Storm, HTML fallback or calibration column is ever loaded.
    """
    hours = _hours(delivery_day, timezone)
    selected = list(zones)
    if not selected or len(set(selected)) != len(selected) or any(
        not isinstance(zone, str) or not re.fullmatch(r"[A-Z]{2}", zone) for zone in selected
    ):
        raise IncumbentComparisonError("Une liste non vide de pays uniques en majuscules est requise.")
    project_root = Path(project_root).absolute()
    batch_root = project_root / "runs" / "exports" / delivery_day
    manifest_path = batch_root / "current_batch_manifest.json"
    _check_plain_path(manifest_path, project_root)
    audit: dict[str, Any] = {"delivery_day": delivery_day, "timezone": timezone,
        "batch_manifest": None, "comparators": [], "warnings": [],
        "observations_imported": False, "calibration_input": False}
    entries: dict[tuple[str, str], dict[str, Any]] = {}
    missing_reason = "variant_not_in_batch"
    if not manifest_path.exists():
        missing_reason = "batch_manifest_unavailable"
        audit["warnings"].append("Aucun manifeste d'export pour cette livraison : comparateurs laissés vides.")
    else:
        if not manifest_path.is_file():
            raise IncumbentComparisonError("Le manifeste d'export n'est pas un fichier.")
        raw = manifest_path.read_bytes()
        try:
            manifest = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_json)
        except (ValueError, UnicodeError) as exc:
            raise IncumbentComparisonError("Manifeste d'export illisible ou ambigu.") from exc
        if not isinstance(manifest, dict) or manifest.get("delivery_day") != delivery_day or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
            raise IncumbentComparisonError("Version ou livraison du manifeste divergente.")
        exports = manifest.get("exports")
        if not isinstance(exports, list):
            raise IncumbentComparisonError("Liste d'exports absente du manifeste.")
        for entry in exports:
            if not isinstance(entry, dict) or not isinstance(entry.get("zone"), str) or not isinstance(entry.get("variant"), str):
                raise IncumbentComparisonError("Entrée d'export invalide.")
            key = (entry["zone"], entry["variant"])
            if key in entries:
                raise IncumbentComparisonError(f"Comparateur dupliqué : {key[0]}/{key[1]}.")
            entries[key] = entry
        audit["batch_manifest"] = {"path": str(manifest_path), "sha256": hashlib.sha256(raw).hexdigest()}
        audit["batch_mode"] = manifest.get("mode")
    results = []
    for zone in selected:
        frame = pd.DataFrame(index=hours)
        for variant in _STAGES:
            for q in (10, 50, 90):
                frame[f"incumbent_{variant}__q{q}"] = np.nan
            entry = entries.get((zone, variant))
            if entry is None:
                audit["comparators"].append({"zone": zone, "variant": variant,
                    "status": "unavailable", "reason": missing_reason, "warnings": []})
                continue
            values, comparator_audit = _load_export(entry, batch_root=batch_root, project_root=project_root,
                delivery_day=delivery_day, zone=zone, variant=variant, hours=hours)
            frame[values.columns] = values
            audit["comparators"].append(comparator_audit)
        frame.insert(0, "zone", zone)
        frame.index.name = "delivery_start_utc"
        results.append(frame.reset_index())
    return pd.concat(results, ignore_index=True), audit
