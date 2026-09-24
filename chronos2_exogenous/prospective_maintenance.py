"""One explicit, append-only repair of an already frozen research trial.

This is not a general code-hash override. Only the original bootstrap runner
may be replaced, and only before any prospective forecast has been issued.
The original trial manifest, calibration and daily predictions remain intact.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Any, Mapping

import pandas as pd


class ProspectiveMaintenanceError(ValueError):
    """The narrowly scoped bootstrap repair cannot be authenticated."""


ORIGINAL_RUNNER_SHA256 = "833d856c1d4d66c99a47af3cf81a23881190c7360a2846e5c6b3e1dcd65927f2"
RUNNER = "chronos2_exogenous/prospective_trial.py"
ADDED_CODE = (
    "chronos2_exogenous/prospective_bootstrap.py",
    "chronos2_exogenous/prospective_maintenance.py",
)
REVISION_DIRECTORY = "maintenance/bootstrap_resume_v1"
REASON = "user_authorized_bootstrap_missing_labels_and_immutable_resume_fix"
FLAGS = dict(diagnostic_only=True, production_pit_evidence=False,
             production_pipeline_evidence=False, promotion_eligible=False,
             activation_performed=False, neural_oof=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProspectiveMaintenanceError(f"Manifeste de maintenance illisible: {path}.") from exc
    if not isinstance(value, dict):
        raise ProspectiveMaintenanceError(f"Objet JSON attendu: {path}.")
    return value


def _safe_path(raw: str | Path, *, root: Path, expected: Path | None = None) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        raise ProspectiveMaintenanceError(f"Chemin absolu requis: {path}.")
    # Check the written path before resolving it: resolution alone hides links.
    for part in (path, *path.parents):
        if part.is_symlink() or (
            part.exists() and getattr(part.stat(), "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            raise ProspectiveMaintenanceError(f"Lien ou point de reanalyse interdit: {part}.")
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ProspectiveMaintenanceError(f"Chemin hors du projet: {path}.")
    if expected is not None and resolved != expected.resolve():
        raise ProspectiveMaintenanceError(f"Chemin hors de la liste de maintenance autorisee: {path}.")
    return resolved


def _verify_entry(entry: Mapping[str, Any], *, root: Path, expected: Path | None = None) -> Path:
    if (not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}
            or not isinstance(entry.get("path"), str)
            or not isinstance(entry.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None):
        raise ProspectiveMaintenanceError("Entree d'empreinte de maintenance invalide.")
    path = _safe_path(entry["path"], root=root, expected=expected)
    if not path.is_file() or _sha256(path) != entry["sha256"]:
        raise ProspectiveMaintenanceError(f"Empreinte divergente ou fichier absent: {path}.")
    return path


def _context(manifest: Mapping[str, Any], output: Path) -> tuple[Path, Path, dict[Path, Mapping[str, Any]]]:
    try:
        root = Path(manifest["config"]["project_root"]).resolve()
        output = _safe_path(output, root=root, expected=Path(manifest["config"]["output_root"]))
        if not output.is_relative_to(root / "runs" / "experiments"):
            raise ProspectiveMaintenanceError("Maintenance hors du laboratoire de recherche.")
        manifest_path = _safe_path(output / "trial_manifest.json", root=root)
        if _json(manifest_path) != dict(manifest):
            raise ProspectiveMaintenanceError("Manifeste fourni different du protocole scelle.")
        entries = manifest["code_files"]
        if not isinstance(entries, list) or not entries:
            raise ProspectiveMaintenanceError("Liste de code scelle absente.")
        indexed = {}
        for entry in entries:
            path = _safe_path(entry["path"], root=root)
            if path in indexed:
                raise ProspectiveMaintenanceError("Entree de code scelle dupliquee.")
            indexed[path] = entry
    except (KeyError, TypeError) as exc:
        raise ProspectiveMaintenanceError("Protocole de maintenance incomplet.") from exc
    return root, output, indexed


def _timestamp(value: Any) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ProspectiveMaintenanceError("Horodatage UTC de maintenance invalide.") from exc
    if pd.isna(result) or result.tzinfo is None or result.utcoffset().total_seconds() != 0:
        raise ProspectiveMaintenanceError("Horodatage UTC explicite requis pour la maintenance.")
    return result.tz_convert("UTC")


def _original_runner(entries: Mapping[Path, Mapping[str, Any]], root: Path) -> Mapping[str, Any]:
    runner = (root / RUNNER).resolve()
    entry = entries.get(runner)
    if entry is None or entry.get("sha256") != ORIGINAL_RUNNER_SHA256:
        raise ProspectiveMaintenanceError("Cette revision ne s'applique qu'au runner bootstrap original.")
    if any((root / name).resolve() in entries for name in ADDED_CODE):
        raise ProspectiveMaintenanceError("Les modules ajoutes etaient deja presents dans le protocole initial.")
    return entry


def _verify_revision(manifest: Mapping[str, Any], root: Path, output: Path,
                     entries: Mapping[Path, Mapping[str, Any]]) -> pd.Timestamp:
    original = _original_runner(entries, root)
    directory = output / REVISION_DIRECTORY
    path = _safe_path(directory / "revision.json", root=root)
    revision = _json(path)
    required = {"schema_version", "kind", "revision_id", "reason", "created_at_utc",
                "trial_manifest_sha256", "original_runner", "before_backup", "after_runner",
                "added_code_files", *FLAGS}
    if (set(revision) != required or type(revision.get("schema_version")) is not int
            or revision["schema_version"] != 1
            or revision.get("kind") != "rank16_trial_bootstrap_maintenance"
            or revision.get("revision_id") != "bootstrap_resume_v1"
            or revision.get("reason") != REASON
            or any(type(revision.get(k)) is not bool or revision[k] is not v for k, v in FLAGS.items())
            or revision.get("trial_manifest_sha256") != _sha256(output / "trial_manifest.json")
            or revision.get("original_runner") != original):
        raise ProspectiveMaintenanceError("Identite ou contrat de la revision de maintenance divergent.")
    backup = _verify_entry(revision["before_backup"], root=root,
                           expected=directory / "prospective_trial.before.py")
    if _sha256(backup) != ORIGINAL_RUNNER_SHA256:
        raise ProspectiveMaintenanceError("La sauvegarde ne correspond pas au runner initial scelle.")
    _verify_entry(revision["after_runner"], root=root, expected=root / RUNNER)
    if revision["after_runner"]["sha256"] == ORIGINAL_RUNNER_SHA256:
        raise ProspectiveMaintenanceError("La revision ne contient aucun correctif du runner.")
    additions = revision["added_code_files"]
    if not isinstance(additions, list) or len(additions) != len(ADDED_CODE):
        raise ProspectiveMaintenanceError("Liste des modules de maintenance invalide.")
    for entry, name in zip(additions, ADDED_CODE):
        _verify_entry(entry, root=root, expected=root / name)
    for path, entry in entries.items():
        if path != (root / RUNNER).resolve():
            _verify_entry(entry, root=root)
    created = _timestamp(revision["created_at_utc"])
    if not _timestamp(manifest["created_at_utc"]) < created <= pd.Timestamp(datetime.now(timezone.utc)):
        raise ProspectiveMaintenanceError("La maintenance doit etre datee apres la preparation, sans antidatage futur.")
    return created


def verify_trial_code(manifest: Mapping[str, Any], output: Path) -> pd.Timestamp | None:
    """Verify original code or the single registered repair, never auto-register.

    The caller must use the returned timestamp as the earliest effective code
    freeze when enforcing the origin of any prospective forecast.
    """
    root, output, entries = _context(manifest, output)
    revision = output / REVISION_DIRECTORY / "revision.json"
    if revision.exists() or revision.is_symlink():
        return _verify_revision(manifest, root, output, entries)
    for entry in entries.values():
        _verify_entry(entry, root=root)
    return None


def _refuse_existing_prospective(output: Path, root: Path) -> None:
    for directory in sorted((output / "days").glob("*/*")):
        _safe_path(directory, root=root)
        path = directory / "manifest.json"
        if (directory / "predictions.csv.gz").exists():
            raise ProspectiveMaintenanceError("Maintenance refusee apres une emission prospective ou sa publication partielle.")
        if path.exists():
            daily = _json(path)
            if (daily.get("prospective_eligible") is not False
                    or daily.get("role") != "retrospective_calibration_only"):
                raise ProspectiveMaintenanceError("Maintenance refusee: une journee prospective ou ambigue existe deja.")


def register_bootstrap_resume_revision(config: Mapping[str, Any]) -> dict[str, Any]:
    """Explicit one-time registration; repeated calls only verify/read its bytes."""
    output = Path(config["output_root"])
    manifest = _json(output / "trial_manifest.json")
    if manifest.get("config") != dict(config):
        raise ProspectiveMaintenanceError("Configuration actuelle differente du protocole scelle.")
    root, output, entries = _context(manifest, output)
    directory = output / REVISION_DIRECTORY
    revision_path = directory / "revision.json"
    if revision_path.exists() or revision_path.is_symlink():
        _verify_revision(manifest, root, output, entries)
        return {"status": "existing", "revision_path": str(revision_path), "revision": _json(revision_path)}
    original = _original_runner(entries, root)
    _refuse_existing_prospective(output, root)
    for path, entry in entries.items():
        if path != (root / RUNNER).resolve():
            _verify_entry(entry, root=root)
    backup = _verify_entry({"path": str(directory / "prospective_trial.before.py"),
                            "sha256": ORIGINAL_RUNNER_SHA256}, root=root)
    after = _safe_path(root / RUNNER, root=root)
    if not after.is_file() or _sha256(after) == ORIGINAL_RUNNER_SHA256:
        raise ProspectiveMaintenanceError("Le runner corrige est absent ou identique a l'original.")
    additions = []
    for name in ADDED_CODE:
        path = _safe_path(root / name, root=root)
        if not path.is_file():
            raise ProspectiveMaintenanceError(f"Module de maintenance absent: {path}.")
        additions.append({"path": str(path), "sha256": _sha256(path)})
    created = pd.Timestamp(datetime.now(timezone.utc))
    if created <= _timestamp(manifest["created_at_utc"]):
        raise ProspectiveMaintenanceError("Horloge de maintenance anterieure au protocole initial.")
    revision = {**FLAGS, "schema_version": 1, "kind": "rank16_trial_bootstrap_maintenance",
                "revision_id": "bootstrap_resume_v1", "reason": REASON,
                "created_at_utc": created.isoformat(),
                "trial_manifest_sha256": _sha256(output / "trial_manifest.json"),
                "original_runner": dict(original),
                "before_backup": {"path": str(backup), "sha256": ORIGINAL_RUNNER_SHA256},
                "after_runner": {"path": str(after), "sha256": _sha256(after)},
                "added_code_files": additions}
    # The parent directory already exists because its original backup is required.
    # Exclusive creation also prevents a second registration racing this one.
    try:
        with revision_path.open("x", encoding="utf-8") as stream:
            json.dump(revision, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except FileExistsError:
        _verify_revision(manifest, root, output, entries)
        return {"status": "existing", "revision_path": str(revision_path), "revision": _json(revision_path)}
    _verify_revision(manifest, root, output, entries)
    return {"status": "registered", "revision_path": str(revision_path), "revision": revision}


__all__ = ["ProspectiveMaintenanceError", "verify_trial_code", "register_bootstrap_resume_revision"]
