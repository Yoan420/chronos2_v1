"""Publish nuclear variants beside regular exports, without replacing them."""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .hourly_contract import local_delivery_day_index
from .process_lock import exclusive_process_lock


class NuclearExportRecoveryError(OSError):
    """An incomplete rollback whose recovery copies must remain on disk."""


class _PublicationStage:
    def __init__(self, destination: Path):
        self.destination = destination
        self.path: Path | None = None
        self.preserve = False

    def __enter__(self):
        self.path = Path(tempfile.mkdtemp(prefix=".nuclear_stage_", dir=self.destination))
        return self

    def __exit__(self, *_exc):
        if self.path is not None and not self.preserve:
            if self.path.resolve().parent != self.destination.resolve():
                raise OSError("Nuclear staging cleanup path escapes its export directory.")
            try:
                shutil.rmtree(self.path)
            except OSError as error:
                # Only leftover private staging files are affected. Do not
                # mask the publication failure or invalidate a committed batch.
                self.preserve = True
                try:
                    print(f"[Nuclear] AVERTISSEMENT : nettoyage temporaire incomplet "
                          f"dans {self.path}: {error}. Fichiers restants conserves.",
                          file=sys.stderr, flush=True)
                except OSError:
                    pass


def _same_content(source: Path, target: Path) -> bool:
    """Compare bytes without the metadata cache used by filecmp.

    Excel commonly allows reading a CSV while denying its replacement. A
    report refresh does not need to replace an unchanged forecast at all.
    """
    try:
        if source.stat().st_size != target.stat().st_size:
            return False
        with source.open("rb") as left, target.open("rb") as right:
            while True:
                block = left.read(1024 * 1024)
                if block != right.read(1024 * 1024):
                    return False
                if not block:
                    return True
    except (FileNotFoundError, PermissionError):
        return False


def _replace_with_retry(source: Path, target: Path, *, attempts: int = 6,
                        retry_seconds: float = 0.2) -> None:
    """Keep atomic replacement; tolerate bounded transient Windows locks."""
    if attempts < 1 or retry_seconds < 0:
        raise ValueError("Invalid nuclear publication retry settings.")
    for attempt in range(1, attempts + 1):
        try:
            os.replace(source, target)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (None, 5, 32, 33):
                raise
            if attempt == attempts:
                raise PermissionError(
                    f"Publication nucleaire bloquee : {target}. "
                    "Le fichier est verrouille ou Windows refuse son remplacement. "
                    "Fermez ce fichier dans Excel ou l'application qui le retient, "
                    "puis relancez la publication. Les resultats calcules sont conserves."
                ) from error
            time.sleep(retry_seconds * attempt)


def publish_nuclear_exports(
    result: Any, reports: Mapping[str, Path], *, project_root: Path,
    zone: str, delivery_day: str, timezone: str,
    report_variants: tuple[str, ...] = ("autonomous", "kalman"),
) -> dict[str, str]:
    """Stage both CSV/HTML reports and roll back this publication on failure.

    The separate nuclear manifest never overwrites current_batch_manifest.json
    or the autonomous/kalman directories belonging to the incumbent launcher.
    """
    project = Path(project_root).resolve()
    selected = tuple(dict.fromkeys(report_variants))
    if not selected or set(selected) - {"autonomous", "kalman"}:
        raise ValueError("Nuclear export variants must be autonomous and/or kalman.")
    code = str(zone).upper()
    day = pd.Timestamp(delivery_day)
    if code not in {"FR", "DE", "BE", "NL", "ES"} or day.tzinfo or day != day.normalize():
        raise ValueError("Invalid nuclear export zone/delivery day.")
    date_text = day.date().isoformat()
    destination = project / "runs" / "exports" / date_text / code.lower()
    if destination.resolve() != destination or not destination.resolve().is_relative_to(project / "runs" / "exports"):
        raise ValueError("Nuclear exports must remain inside runs/exports.")
    expected = local_delivery_day_index(day.date(), timezone=timezone)
    audit = json.loads(Path(reports["audit"]).read_text(encoding="utf-8"))
    if audit.get("zone") != code or audit.get("delivery_day") != date_text:
        raise ValueError("Nuclear report audit does not match export identity.")
    destination.mkdir(parents=True, exist_ok=True)
    lock = destination / ".nuclear_publish.lock"
    # OS guard survives concurrent callers and releases after process death.
    with exclusive_process_lock(lock):
        with _PublicationStage(destination) as stage:
            staging = stage.path
            files: list[tuple[Path, Path]] = []
            records = []
            outputs = {}
            for kind, raw, model in (
                ("autonomous", result.source_forecast, "residual_corrected"),
                ("kalman", result.kalman_view.forecast, "residual_kalman"),
            ):
                if kind not in selected:
                    continue
                variant = f"nuclear_{kind}"
                frame = raw.copy(deep=True)
                if "delivery_start_utc" not in frame:
                    frame = frame.reset_index()
                index = pd.DatetimeIndex(pd.to_datetime(frame.delivery_start_utc, utc=True))
                if not index.equals(expected):
                    raise ValueError("Nuclear CSV must cover the exact physical delivery day.")
                quantiles = frame[[f"{model}__{q}" for q in ("q10", "q50", "q90")]].to_numpy(float)
                if not np.isfinite(quantiles).all() or (np.diff(quantiles, axis=1) < 0).any():
                    raise ValueError("Invalid nuclear export quantiles.")
                local = index.tz_convert(timezone)
                export = pd.DataFrame({
                    "zone": code, "forecast_variant": variant, "source_model": model,
                    "uses_mkonline": False, "uses_nuclear": True,
                    "delivery_start_utc": index, "delivery_start_local": [t.isoformat() for t in local],
                    "local_date": [str(t.date()) for t in local], "local_hour": local.hour,
                    "utc_offset_minutes": [int(t.utcoffset().total_seconds() / 60) for t in local],
                    "fold": [t.fold for t in local], "delivery_hour_position": np.arange(1, len(index) + 1),
                    "hours_in_local_day": len(index),
                    "q10": quantiles[:, 0], "q50": quantiles[:, 1], "q90": quantiles[:, 2],
                    "price_eur_mwh": quantiles[:, 1],
                })
                name = f"forecast_{code.lower()}_{date_text}_{variant}"
                group = staging / variant
                group.mkdir()
                export.to_csv(group / f"{name}.csv", index=False)
                shutil.copy2(reports[kind], group / f"{name}.html")
                shutil.copy2(reports["audit"], group / "nuclear_report_audit.json")
                record = {"variant": variant, "zone": code, "source_model": model,
                          "nuclear_series": "power.fr.generation.nuclear.gw.fcst", "files": []}
                for source in group.iterdir():
                    if source.stat().st_size == 0:
                        raise ValueError("Empty nuclear export artifact.")
                    target = destination / variant / source.name
                    if target.resolve() != destination.resolve() / variant / source.name:
                        raise ValueError("Nuclear export path escapes its zone directory.")
                    files.append((source, target))
                    record["files"].append({"path": f"{variant}/{source.name}",
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
                records.append(record)
                outputs[kind] = str(destination / variant / f"{name}.html")
            manifest = {"schema_version": 1, "delivery_day": date_text, "zone": code,
                "mode": "both_with_nuclear" if len(selected) == 2 else "nuclear_" + selected[0],
                "generated_at_utc": str(pd.Timestamp.now(tz="UTC")),
                "incumbent_exports_modified": False, "exports": records}
            manifest_file = staging / "current_nuclear_batch_manifest.json"
            manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            index_file = staging / "nuclear_index.html"
            links = "".join(f'<li><a href="nuclear_{kind}/{Path(path).name}">{html.escape(label)}</a></li>'
                            for kind, path, label in (
                                ("autonomous", outputs.get("autonomous", ""), "Chronos-2 + nucléaire FR + correcteur résiduel"),
                                ("kalman", outputs.get("kalman", ""), "Chronos-2 + nucléaire FR + correcteur résiduel + Kalman"))
                            if kind in selected)
            index_file.write_text('<!doctype html><html lang="fr"><meta charset="utf-8">'
                '<meta name="color-scheme" content="light dark"><title>Rapports nucléaires</title>'
                f'<h1>{code} — {date_text}</h1><ul>{links}</ul></html>', encoding="utf-8")
            files.extend([(index_file, destination / index_file.name),
                          (manifest_file, destination / manifest_file.name)])
            # Reject collisions before changing any previous export. A late
            # directory collision must not break rollback of earlier files.
            for _, target in files:
                if target.resolve() != destination / target.relative_to(destination):
                    raise ValueError("Nuclear publication target is a link or redirected path.")
                if target.exists() and not target.is_file():
                    raise ValueError(f"Nuclear publication target is not a file: {target}")
                if target.parent.exists() and not target.parent.is_dir():
                    raise ValueError(f"Nuclear publication parent is not a directory: {target.parent}")
            backups = []
            try:
                for number, (source, target) in enumerate(files):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.is_file() and _same_content(source, target):
                        continue
                    backup = staging / f"backup_{number}"
                    existed = target.is_file()
                    if existed:
                        shutil.copy2(target, backup)
                    # Keep recovery copies even if an interrupt arrives between
                    # atomic replacement and recording its successful return.
                    stage.preserve = True
                    _replace_with_retry(source, target)
                    backups.append((target, backup if existed else None))
            except Exception as publication_error:
                rollback = []
                for target, backup in reversed(backups):
                    item = {"target": str(target), "backup": backup.name if backup else None,
                            "action": "restore" if backup else "remove_new_file"}
                    try:
                        if backup is None:
                            target.unlink(missing_ok=True)
                        else:
                            item["backup_sha256"] = hashlib.sha256(backup.read_bytes()).hexdigest()
                            _replace_with_retry(backup, target)
                        item["status"] = "restored" if backup else "removed"
                    except Exception as rollback_error:
                        item.update(status="failed", error_type=type(rollback_error).__name__,
                                    error=str(rollback_error))
                    rollback.append(item)
                failures = [item for item in rollback if item["status"] == "failed"]
                if failures:
                    # The stage was marked for retention before any replacement.
                    # In particular, a failed audit write must not erase backups.
                    recovery = {"schema_version": 1, "status": "rollback_incomplete",
                        "destination": str(destination), "zone": code, "delivery_day": date_text,
                        "created_at_utc": str(pd.Timestamp.now(tz="UTC")),
                        "publication_error": {"type": type(publication_error).__name__,
                                              "message": str(publication_error)},
                        "rollback": rollback}
                    audit_path = staging / "publication_recovery.json"
                    audit_error = ""
                    try:
                        with audit_path.open("w", encoding="utf-8") as stream:
                            json.dump(recovery, stream, ensure_ascii=False, indent=2, allow_nan=False)
                            stream.write("\n")
                            stream.flush()
                            os.fsync(stream.fileno())
                    except Exception as error:
                        audit_error = f" Audit de reprise non ecrit : {type(error).__name__}: {error}."
                    raise NuclearExportRecoveryError(
                        f"Publication nucleaire echouee : {publication_error}. "
                        f"Restauration incomplete pour {len(failures)} fichier(s). "
                        f"Sauvegardes conservees dans {staging}; audit : {audit_path}."
                        + audit_error
                    ) from publication_error
                stage.preserve = False
                raise
            stage.preserve = False
            outputs.update(index=str(destination / index_file.name), manifest=str(destination / manifest_file.name))
            return outputs
