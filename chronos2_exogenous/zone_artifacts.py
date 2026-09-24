"""Prepare immutable per-zone working copies of one trained LoRA artefact.

The LoRA checkpoint may be trained from a multi-zone panel, but evaluation,
residual calibration, final evidence, shadow evidence and promotion are all
deliberately per-zone.  Those later stages write canonical file names and
eventually bind ``experiment_manifest.json`` to one scalar ``zone``.  This
module therefore forks the pristine training artefact before any evaluation.

Copies are physical, complete, staged and verified.  The source is never
modified and an existing destination is reusable only while its complete
sealed tree is still byte-for-byte identical to the prepared copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from .governance import (
    GOVERNABLE_EVALUATION_ROLE,
    REQUIRED_EXPERIMENT_FLAGS,
    validate_experiment_manifest,
)
from .lora_finetune import verify_bundle


ZONE_COPY_SCHEMA_VERSION = 1
ZONE_COPY_MANIFEST_NAME = "zone_artifact_manifest.json"
ZONE_PROVENANCE_KEY = "zone_artifact_provenance"
_ZONE_PATTERN = re.compile(r"[A-Z]{2,8}")
_TEMPORARY_MARKERS = (".tmp-", ".staging-", ".prepare-")
_DERIVED_MANIFEST_KEYS = (
    "evaluation_evidence",
    "raw_evaluation_evidence",
    "final_pipeline_evaluation",
)
_DERIVED_ROOT_NAMES = {
    "evaluation_predictions.csv.gz",
    "evaluation_daily.csv.gz",
    "evaluation_metrics.json",
    "evaluation_report.html",
    "evaluation_manifest.json",
    "evaluation_cache",
    "final_pipeline",
    "residual_calibration",
    "shadow_predictions.csv.gz",
    "shadow_observed_evidence.csv.gz",
    "shadow_manifest.json",
    "shadow_final",
    "shadow_inputs",
}


class ZoneArtifactPreparationError(RuntimeError):
    """Raised when a safe, pristine per-zone copy cannot be established."""


@dataclass(frozen=True)
class PreparedZoneArtifact:
    zone: str
    path: Path
    manifest_path: Path
    created: bool


@dataclass(frozen=True)
class PreparedZoneArtifacts:
    source_directory: Path
    output_root: Path
    source_tree_sha256: str
    artifacts: tuple[PreparedZoneArtifact, ...]


@dataclass(frozen=True)
class _TreeSnapshot:
    sha256: str
    entries: int
    files: tuple[str, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(records: Sequence[Mapping[str, object]]) -> str:
    encoded = json.dumps(
        list(records),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink() or os.path.islink(path):
        return True
    try:
        attributes = path.stat(follow_symlinks=False).st_file_attributes
    except (AttributeError, OSError):
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _resolve_without_links(value: str | Path, *, label: str) -> Path:
    """Resolve only after rejecting symlink/junction components."""

    absolute = Path(value).expanduser().absolute()
    for component in (absolute, *absolute.parents):
        if _is_link_or_reparse(component):
            raise ZoneArtifactPreparationError(
                f"{label}: lien symbolique/junction/reparse interdit: {component}."
            )
    return absolute.resolve()


def _tree_snapshot(
    root: Path, *, exclude: Iterable[str] = ()
) -> _TreeSnapshot:
    if not root.is_dir() or _is_link_or_reparse(root):
        raise ZoneArtifactPreparationError(
            f"Repertoire absent, lien ou reparse interdit: {root}."
        )
    excluded = {Path(value).as_posix() for value in exclude}
    records: list[dict[str, object]] = []
    files: list[str] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        parent = Path(directory)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            child = parent / name
            relative = child.relative_to(root).as_posix()
            if _is_link_or_reparse(child):
                raise ZoneArtifactPreparationError(
                    f"Lien/reparse interdit dans l'artefact: {child}."
                )
            if relative not in excluded:
                records.append({"kind": "directory", "path": relative})
        for name in file_names:
            child = parent / name
            relative = child.relative_to(root).as_posix()
            if _is_link_or_reparse(child) or not child.is_file():
                raise ZoneArtifactPreparationError(
                    f"Fichier non regulier interdit dans l'artefact: {child}."
                )
            if relative in excluded:
                continue
            files.append(relative)
            records.append(
                {
                    "kind": "file",
                    "path": relative,
                    "size_bytes": child.stat().st_size,
                    "sha256": _sha256_file(child),
                }
            )
    records.sort(key=lambda record: (str(record["path"]), str(record["kind"])))
    return _TreeSnapshot(
        sha256=_canonical_digest(records),
        entries=len(records),
        files=tuple(sorted(files)),
    )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or _is_link_or_reparse(path):
        raise ZoneArtifactPreparationError(f"{label} absent ou non regulier: {path}.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ZoneArtifactPreparationError(f"{label} illisible: {path}.") from exc
    if not isinstance(payload, dict):
        raise ZoneArtifactPreparationError(f"{label} doit etre un objet JSON.")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_zones(values: Sequence[str]) -> tuple[str, ...]:
    if not values:
        raise ZoneArtifactPreparationError("Au moins une zone explicite est requise.")
    zones = tuple(str(value).strip().upper() for value in values)
    invalid = [value for value in zones if _ZONE_PATTERN.fullmatch(value) is None]
    if invalid:
        raise ZoneArtifactPreparationError(
            "Zones invalides: " + ", ".join(repr(value) for value in invalid) + "."
        )
    if len(set(zones)) != len(zones):
        raise ZoneArtifactPreparationError("Les zones demandees doivent etre uniques.")
    return zones


def _manifest_zone_inventory(
    manifest: Mapping[str, Any], *, requested_zones: Sequence[str]
) -> tuple[str, ...]:
    summary = manifest.get("panel_audit_summary")
    pit = manifest.get("pit_audit")
    if not isinstance(summary, Mapping) or not isinstance(pit, Mapping):
        raise ZoneArtifactPreparationError(
            "Le manifeste ne contient pas les preuves du panel multi-zone."
        )
    declared = summary.get("zones")
    items = pit.get("items")
    if (
        not isinstance(declared, list)
        or not declared
        or any(not isinstance(value, str) for value in declared)
        or not isinstance(items, list)
        or any(not isinstance(value, str) for value in items)
    ):
        raise ZoneArtifactPreparationError(
            "Les zones/items du panel doivent etre des listes explicites."
        )
    summary_zones = tuple(value.strip().upper() for value in declared)
    item_zones = tuple(value.strip().upper() for value in items)
    if (
        len(set(summary_zones)) != len(summary_zones)
        or len(set(item_zones)) != len(item_zones)
        or set(summary_zones) != set(item_zones)
    ):
        raise ZoneArtifactPreparationError(
            "Les zones de panel_audit_summary et pit_audit.items divergent."
        )

    audit_value = summary.get("audit_path")
    if not isinstance(audit_value, str) or not audit_value.strip():
        raise ZoneArtifactPreparationError("panel_audit_summary.audit_path absent.")
    audit_path = Path(audit_value).expanduser().resolve()
    expected_audit_sha = manifest.get("panel_audit_sha256")
    if (
        not audit_path.is_file()
        or _is_link_or_reparse(audit_path)
        or not isinstance(expected_audit_sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_audit_sha)
        or _sha256_file(audit_path) != expected_audit_sha
    ):
        raise ZoneArtifactPreparationError(
            "Le sidecar du panel est absent ou son SHA-256 diverge du training."
        )
    audit = _read_json(audit_path, label="sidecar du panel")
    audit_zones_raw = audit.get("zones")
    if (
        audit.get("layout") != "per_zone"
        or not isinstance(audit_zones_raw, list)
        or any(not isinstance(value, str) for value in audit_zones_raw)
    ):
        raise ZoneArtifactPreparationError(
            "Le sidecar du panel doit declarer layout=per_zone et ses zones."
        )
    audit_zones = tuple(value.strip().upper() for value in audit_zones_raw)
    if len(set(audit_zones)) != len(audit_zones) or set(audit_zones) != set(
        summary_zones
    ):
        raise ZoneArtifactPreparationError(
            "Les zones du sidecar du panel divergent du manifeste d'entrainement."
        )
    if audit.get("panel_sha256") != manifest.get("panel_sha256"):
        raise ZoneArtifactPreparationError(
            "Le SHA du panel diverge entre sidecar et manifeste d'entrainement."
        )
    missing = [zone for zone in requested_zones if zone not in set(audit_zones)]
    if missing:
        raise ZoneArtifactPreparationError(
            "Zones absentes du panel entraine: "
            + ", ".join(missing)
            + "; disponibles="
            + ", ".join(audit_zones)
            + "."
        )
    return audit_zones


def _validate_copy_manifest(manifest: Mapping[str, Any], *, zone: str) -> None:
    """Validate a physical copy, without making diagnostic runs governable.

    Primary candidates retain the original governance validator. Diagnostics
    preserve their original role and every causal/freeze check, but copying
    their bytes is not a scoring, promotion or activation decision.
    """
    role = manifest.get("evaluation_role")
    if role == GOVERNABLE_EVALUATION_ROLE:
        validate_experiment_manifest(manifest, zone=zone)
        return
    if role != "diagnostic_only":
        raise ZoneArtifactPreparationError(
            "PrepareZones: evaluation_role doit etre primary_predeclared ou diagnostic_only."
        )
    model_id = manifest.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ZoneArtifactPreparationError("PrepareZones: model_id absent.")
    declared_zone = manifest.get("zone")
    if declared_zone is not None and str(declared_zone).strip().upper() != zone.upper():
        raise ZoneArtifactPreparationError(
            f"Zone du manifeste {declared_zone!r} incompatible avec {zone}."
        )
    failures = []
    for key, expected in REQUIRED_EXPERIMENT_FLAGS.items():
        actual = manifest.get(key)
        if type(actual) is not type(expected) or actual != expected:
            failures.append(f"{key}={actual!r}, attendu={expected!r}")
    if failures:
        raise ZoneArtifactPreparationError(
            "Preuves causales/freeze incompletes pour la copie diagnostic: "
            + "; ".join(failures)
        )
    for key in ("production_pit_evidence", "production_pipeline_evidence"):
        if type(manifest.get(key)) is not bool:
            raise ZoneArtifactPreparationError(
                f"PrepareZones: {key} doit etre un booleen explicite."
            )
    for key, expected in (
        ("diagnostic_only", True),
        ("promotion_eligible", False),
        ("activation_performed", False),
    ):
        if key in manifest and manifest[key] is not expected:
            raise ZoneArtifactPreparationError(
                f"PrepareZones diagnostic: {key} doit valoir {expected!r}."
            )
    for key in ("checkpoint_sha256", "schema_sha256"):
        value = manifest.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ZoneArtifactPreparationError(
                f"PrepareZones: {key} doit etre un SHA-256 hexadecimal."
            )


def _validate_pristine_source(
    source: Path, *, requested_zones: Sequence[str]
) -> tuple[dict[str, Any], _TreeSnapshot]:
    if any(marker in source.name.casefold() for marker in _TEMPORARY_MARKERS):
        raise ZoneArtifactPreparationError(
            f"Artefact temporaire/en cours refuse comme source: {source}."
        )
    manifest = verify_bundle(source)
    if not isinstance(manifest, dict):
        manifest = dict(manifest)
    declared_zone = manifest.get("zone")
    if declared_zone not in (None, ""):
        raise ZoneArtifactPreparationError(
            f"La source est deja liee a la zone {declared_zone!r}."
        )
    if manifest.get("production_pipeline_evidence") is not False:
        raise ZoneArtifactPreparationError(
            "La source doit etre l'artefact d'entrainement avant preuve finale."
        )
    if manifest.get("candidate_output_stage") not in (None, ""):
        raise ZoneArtifactPreparationError(
            "La source possede deja un candidate_output_stage derive."
        )
    derived_keys = [key for key in _DERIVED_MANIFEST_KEYS if key in manifest]
    derived_paths = sorted(name for name in _DERIVED_ROOT_NAMES if (source / name).exists())
    if derived_keys or derived_paths:
        detail = ", ".join([*derived_keys, *derived_paths])
        raise ZoneArtifactPreparationError(
            "La source n'est plus un artefact d'entrainement vierge: " + detail + "."
        )
    if (source / ZONE_COPY_MANIFEST_NAME).exists() or ZONE_PROVENANCE_KEY in manifest:
        raise ZoneArtifactPreparationError("Une copie par zone ne peut pas servir de source.")
    _validate_copy_manifest(manifest, zone=requested_zones[0])
    _manifest_zone_inventory(manifest, requested_zones=requested_zones)
    snapshot = _tree_snapshot(source)
    suspicious = [
        path
        for path in snapshot.files
        if any(marker in Path(path).name.casefold() for marker in _TEMPORARY_MARKERS)
        or Path(path).name.casefold().endswith(".lock")
    ]
    if suspicious:
        raise ZoneArtifactPreparationError(
            "Fichiers temporaires/de verrouillage dans la source: "
            + ", ".join(suspicious)
            + "."
        )
    return manifest, snapshot


def _assert_manifest_is_source_plus_zone(
    clone: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    zone: str,
    provenance: Mapping[str, Any],
) -> None:
    if clone.get("zone") != zone or clone.get(ZONE_PROVENANCE_KEY) != provenance:
        raise ZoneArtifactPreparationError(
            f"La provenance de la copie {zone} est absente ou divergente."
        )
    restored = dict(clone)
    restored.pop("zone", None)
    restored.pop(ZONE_PROVENANCE_KEY, None)
    if restored != dict(source):
        raise ZoneArtifactPreparationError(
            f"La copie {zone} a modifie le contrat d'entrainement."
        )


def _assert_physical_copy(source: Path, destination: Path, files: Sequence[str]) -> None:
    for relative in files:
        source_file = source / relative
        destination_file = destination / relative
        if not destination_file.is_file():
            raise ZoneArtifactPreparationError(
                f"Fichier absent de la copie physique: {destination_file}."
            )
        try:
            shared = os.path.samefile(source_file, destination_file)
        except OSError as exc:
            raise ZoneArtifactPreparationError(
                f"Impossible de verifier l'identite physique de {destination_file}."
            ) from exc
        if shared:
            raise ZoneArtifactPreparationError(
                f"Hardlink partage avec la source interdit: {destination_file}."
            )


def _assert_zone_copies_are_independent(
    destinations: Mapping[str, Path], files: Sequence[str]
) -> None:
    """Reject hardlinks shared by any two supposedly isolated zone copies."""

    zones = sorted(destinations)
    for index, left_zone in enumerate(zones):
        for right_zone in zones[index + 1 :]:
            left = destinations[left_zone]
            right = destinations[right_zone]
            for relative in files:
                try:
                    shared = os.path.samefile(left / relative, right / relative)
                except OSError as exc:
                    raise ZoneArtifactPreparationError(
                        "Impossible de verifier l'independance physique entre "
                        f"{left_zone} et {right_zone}: {relative}."
                    ) from exc
                if shared:
                    raise ZoneArtifactPreparationError(
                        "Hardlink partage entre copies de zones interdit: "
                        f"{left_zone}/{right_zone}/{relative}."
                    )


def _assert_source_bytes_preserved(
    source: Path, destination: Path, files: Sequence[str]
) -> None:
    for relative in files:
        if relative == "experiment_manifest.json":
            continue
        source_file = source / relative
        destination_file = destination / relative
        if (
            not destination_file.is_file()
            or source_file.stat().st_size != destination_file.stat().st_size
            or _sha256_file(source_file) != _sha256_file(destination_file)
        ):
            raise ZoneArtifactPreparationError(
                f"La copie physique differe de la source: {destination_file}."
            )


def _validate_zone_copy(
    *,
    source: Path,
    destination: Path,
    zone: str,
    source_manifest: Mapping[str, Any],
    source_snapshot: _TreeSnapshot,
) -> None:
    seal_path = destination / ZONE_COPY_MANIFEST_NAME
    seal = _read_json(seal_path, label=f"sceau de copie {zone}")
    expected_seal = {
        "schema_version": ZONE_COPY_SCHEMA_VERSION,
        "kind": "chronos2_exogenous_zone_artifact_copy",
        "zone": zone,
        "source_artifact_path": str(source),
        "source_experiment_manifest_sha256": _sha256_file(
            source / "experiment_manifest.json"
        ),
        "source_tree_sha256": source_snapshot.sha256,
        "source_tree_entries": source_snapshot.entries,
        "copy_mode": "physical_complete",
        "shared_hardlinks_with_source": False,
    }
    mismatches = [key for key, expected in expected_seal.items() if seal.get(key) != expected]
    if mismatches:
        raise ZoneArtifactPreparationError(
            f"Sceau de copie {zone} divergent: " + ", ".join(mismatches) + "."
        )
    created_at = seal.get("created_at_utc")
    clone_sha = seal.get("clone_tree_sha256")
    clone_entries = seal.get("clone_tree_entries")
    if (
        not isinstance(created_at, str)
        or not created_at.strip()
        or not isinstance(clone_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", clone_sha) is None
        or isinstance(clone_entries, bool)
        or not isinstance(clone_entries, int)
        or clone_entries < 1
    ):
        raise ZoneArtifactPreparationError(f"Sceau de copie {zone} incomplet.")
    snapshot = _tree_snapshot(destination, exclude=(ZONE_COPY_MANIFEST_NAME,))
    if snapshot.sha256 != clone_sha or snapshot.entries != clone_entries:
        raise ZoneArtifactPreparationError(
            f"La copie scellee {zone} a ete modifiee ou completee."
        )
    if (
        snapshot.entries != source_snapshot.entries
        or snapshot.files != source_snapshot.files
    ):
        raise ZoneArtifactPreparationError(
            f"La structure de la copie {zone} differe de la source."
        )
    provenance = {
        "schema_version": ZONE_COPY_SCHEMA_VERSION,
        "kind": "chronos2_exogenous_zone_artifact",
        "zone": zone,
        "source_artifact_path": str(source),
        "source_experiment_manifest_sha256": expected_seal[
            "source_experiment_manifest_sha256"
        ],
        "source_tree_sha256": source_snapshot.sha256,
        "copy_mode": "physical_complete",
        "created_at_utc": created_at,
    }
    clone_manifest = _read_json(
        destination / "experiment_manifest.json",
        label=f"manifeste d'experience {zone}",
    )
    _assert_manifest_is_source_plus_zone(
        clone_manifest,
        source_manifest,
        zone=zone,
        provenance=provenance,
    )
    verified = verify_bundle(destination)
    _validate_copy_manifest(verified, zone=zone)
    _assert_source_bytes_preserved(source, destination, source_snapshot.files)
    _assert_physical_copy(source, destination, source_snapshot.files)


def _stage_zone_copy(
    *,
    source: Path,
    destination: Path,
    zone: str,
    source_manifest: Mapping[str, Any],
    source_snapshot: _TreeSnapshot,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.prepare-{uuid4().hex}"
    created_at = datetime.now(timezone.utc).isoformat()
    try:
        shutil.copytree(source, staging, copy_function=shutil.copy2, symlinks=False)
        copied_manifest_path = staging / "experiment_manifest.json"
        copied_manifest = _read_json(
            copied_manifest_path, label=f"manifeste copie {zone}"
        )
        copied_snapshot = _tree_snapshot(staging)
        if (
            copied_manifest != dict(source_manifest)
            or copied_snapshot != source_snapshot
        ):
            raise ZoneArtifactPreparationError(
                f"La source a change ou a ete mal copiee pendant la copie {zone}."
            )
        provenance = {
            "schema_version": ZONE_COPY_SCHEMA_VERSION,
            "kind": "chronos2_exogenous_zone_artifact",
            "zone": zone,
            "source_artifact_path": str(source),
            "source_experiment_manifest_sha256": _sha256_file(
                source / "experiment_manifest.json"
            ),
            "source_tree_sha256": source_snapshot.sha256,
            "copy_mode": "physical_complete",
            "created_at_utc": created_at,
        }
        zone_manifest = dict(copied_manifest)
        zone_manifest["zone"] = zone
        zone_manifest[ZONE_PROVENANCE_KEY] = provenance
        _write_json_atomic(copied_manifest_path, zone_manifest)
        verify_bundle(staging)
        _validate_copy_manifest(zone_manifest, zone=zone)
        _assert_physical_copy(source, staging, source_snapshot.files)
        clone_snapshot = _tree_snapshot(
            staging, exclude=(ZONE_COPY_MANIFEST_NAME,)
        )
        seal = {
            "schema_version": ZONE_COPY_SCHEMA_VERSION,
            "kind": "chronos2_exogenous_zone_artifact_copy",
            "zone": zone,
            "source_artifact_path": str(source),
            "source_experiment_manifest_sha256": provenance[
                "source_experiment_manifest_sha256"
            ],
            "source_tree_sha256": source_snapshot.sha256,
            "source_tree_entries": source_snapshot.entries,
            "clone_tree_sha256": clone_snapshot.sha256,
            "clone_tree_entries": clone_snapshot.entries,
            "copy_mode": "physical_complete",
            "shared_hardlinks_with_source": False,
            "created_at_utc": created_at,
        }
        _write_json_atomic(staging / ZONE_COPY_MANIFEST_NAME, seal)
        _validate_zone_copy(
            source=source,
            destination=staging,
            zone=zone,
            source_manifest=source_manifest,
            source_snapshot=source_snapshot,
        )
        return staging
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _commit_staging(staging: Path, destination: Path) -> None:
    """Commit one verified directory; split out to make rollback testable."""

    os.replace(staging, destination)


def _remove_directory(path: Path) -> None:
    shutil.rmtree(path)


def prepare_zone_artifacts(
    source_run_directory: str | Path,
    *,
    zones: Sequence[str],
    output_root: str | Path | None = None,
) -> PreparedZoneArtifacts:
    """Physically fork one pristine trained bundle into isolated zone runs.

    The default layout is ``<source-parent>/zones/<ZONE>/artifact``.  When
    ``output_root`` is provided it is the directory *above* ``zones``.
    Existing copies are accepted only when their seal, full tree, source
    fingerprint and physical independence all still verify.
    """

    canonical_zones = _canonical_zones(zones)
    source = _resolve_without_links(
        source_run_directory, label="Artefact source PrepareZones"
    )
    source_manifest, source_snapshot = _validate_pristine_source(
        source, requested_zones=canonical_zones
    )
    root = (
        _resolve_without_links(output_root, label="Racine PrepareZones")
        if output_root is not None
        else source.parent
    )
    zones_root = root / "zones"
    destinations = {
        zone: _resolve_without_links(
            zones_root / zone / "artifact",
            label=f"Destination PrepareZones {zone}",
        )
        for zone in canonical_zones
    }
    for zone, destination in destinations.items():
        if destination == source or source in destination.parents:
            raise ZoneArtifactPreparationError(
                f"La destination {zone} ne peut pas etre dans la source {source}."
            )

    existing: dict[str, PreparedZoneArtifact] = {}
    missing: list[str] = []
    for zone, destination in destinations.items():
        if destination.exists():
            _validate_zone_copy(
                source=source,
                destination=destination,
                zone=zone,
                source_manifest=source_manifest,
                source_snapshot=source_snapshot,
            )
            existing[zone] = PreparedZoneArtifact(
                zone=zone,
                path=destination,
                manifest_path=destination / ZONE_COPY_MANIFEST_NAME,
                created=False,
            )
        else:
            missing.append(zone)

    staged: dict[str, Path] = {}
    committed: list[Path] = []
    try:
        for zone in missing:
            staged[zone] = _stage_zone_copy(
                source=source,
                destination=destinations[zone],
                zone=zone,
                source_manifest=source_manifest,
                source_snapshot=source_snapshot,
            )
        if _tree_snapshot(source) != source_snapshot:
            raise ZoneArtifactPreparationError(
                "La source a change pendant la preparation; aucune copie n'est publiee."
            )
        for zone in missing:
            destination = destinations[zone]
            if destination.exists():
                raise ZoneArtifactPreparationError(
                    f"La destination est apparue pendant la preparation: {destination}."
                )
            _commit_staging(staged[zone], destination)
            committed.append(destination)
        for zone in missing:
            _validate_zone_copy(
                source=source,
                destination=destinations[zone],
                zone=zone,
                source_manifest=source_manifest,
                source_snapshot=source_snapshot,
            )
        _assert_zone_copies_are_independent(destinations, source_snapshot.files)
    except Exception as exc:
        rollback_errors: list[str] = []
        for destination in reversed(committed):
            try:
                _remove_directory(destination)
            except Exception as rollback_exc:  # pragma: no cover - filesystem failure.
                rollback_errors.append(f"{destination}: {rollback_exc}")
        for staging in staged.values():
            if staging.exists():
                try:
                    _remove_directory(staging)
                except Exception as cleanup_exc:  # pragma: no cover - filesystem failure.
                    rollback_errors.append(f"{staging}: {cleanup_exc}")
        if rollback_errors:
            raise ZoneArtifactPreparationError(
                "Preparation echouee et rollback incomplet: "
                + "; ".join(rollback_errors)
            ) from exc
        raise

    artifacts = tuple(
        existing.get(zone)
        or PreparedZoneArtifact(
            zone=zone,
            path=destinations[zone],
            manifest_path=destinations[zone] / ZONE_COPY_MANIFEST_NAME,
            created=True,
        )
        for zone in canonical_zones
    )
    return PreparedZoneArtifacts(
        source_directory=source,
        output_root=root,
        source_tree_sha256=source_snapshot.sha256,
        artifacts=artifacts,
    )


__all__ = [
    "PreparedZoneArtifact",
    "PreparedZoneArtifacts",
    "ZONE_COPY_MANIFEST_NAME",
    "ZONE_COPY_SCHEMA_VERSION",
    "ZONE_PROVENANCE_KEY",
    "ZoneArtifactPreparationError",
    "prepare_zone_artifacts",
]
