"""Fail-closed evidence contract for reconstructed historical PIT inputs.

The contract deliberately separates two claims which are often conflated:

``historical_backtest_pit_evidence``
    An offline reconstruction can be used for causal training/evaluation.  It
    is backed by an explicitly reviewed provider capability, an immutable
    local request/response ledger and the exact bytes used by the model.

``prospective_capture``
    The bytes were captured before the operational forecast cutoff.  This is
    the stronger claim required by live/shadow execution and is validated by
    :mod:`chronos2_exogenous.feature_bank` through a separate contract.

Passing this verifier never creates prospective evidence and never upgrades a
legacy ``revision_date=...`` reconstruction.  The complete evidence manifest
must be pinned by SHA-256 by the caller; a self-declared JSON boolean is not an
attestation.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

import pandas as pd


HISTORICAL_EVIDENCE_SCHEMA_VERSION = 1
HISTORICAL_EVIDENCE_KIND = "chronos2_attested_historical_asof_archive"
HISTORICAL_EVIDENCE_SCOPE = "offline_training_backtest_only"
SUPPORTED_HISTORICAL_MECHANISMS = frozenset(
    {
        "provider_asof_state",
        "current_snapshot_last_modified_watermark",
    }
)

# Trust anchors live in reviewed application code, never in a data bundle or
# in the source declaration which points at that bundle.  A capability must be
# added here (with the exact canonical digest and identity recorded by the
# independent review) before any locally supplied manifest can become
# historical backtest evidence.  The registry is deliberately empty until a
# real provider/materializer review has been completed.  Unit tests replace
# the immutable mapping in their isolated process; production code exposes no
# registration API.
_APPROVED_HISTORICAL_CAPABILITIES: Mapping[str, Mapping[str, str]] = (
    MappingProxyType({})
)
# A reviewed capability is necessary but not sufficient: otherwise anybody
# able to write local data files could invent a ledger under that capability.
# The exact, completed evidence manifest must also be independently approved.
# This immutable registry is intentionally empty until that review happens.
_APPROVED_HISTORICAL_EVIDENCE_MANIFESTS: Mapping[
    str, Mapping[str, str]
] = MappingProxyType({})

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class HistoricalPITEvidenceError(ValueError):
    """Raised when historical evidence is incomplete or self-contradictory."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: object, *, label: str) -> str:
    candidate = str(value).strip().lower() if isinstance(value, str) else ""
    if _SHA256.fullmatch(candidate) is None:
        raise HistoricalPITEvidenceError(f"{label} doit etre un SHA-256.")
    return candidate


def _utc(value: object, *, label: str) -> pd.Timestamp:
    if not isinstance(value, str) or not value.strip():
        raise HistoricalPITEvidenceError(f"{label} doit etre un timestamp UTC.")
    try:
        parsed = pd.Timestamp(value.strip())
    except (TypeError, ValueError) as exc:
        raise HistoricalPITEvidenceError(
            f"{label} doit etre un timestamp UTC."
        ) from exc
    if pd.isna(parsed) or parsed.tzinfo is None or parsed.utcoffset() is None:
        raise HistoricalPITEvidenceError(f"{label} doit etre un timestamp UTC.")
    if parsed.utcoffset() != pd.Timedelta(0):
        raise HistoricalPITEvidenceError(
            f"{label} doit declarer explicitement le fuseau UTC."
        )
    return parsed.tz_convert("UTC")


def _strict_positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HistoricalPITEvidenceError(f"{label} doit etre un entier positif.")
    return int(value)


def _load_mapping(path: Path, *, label: str) -> tuple[Mapping[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HistoricalPITEvidenceError(f"{label} illisible: {path}.") from exc
    if not isinstance(payload, Mapping):
        raise HistoricalPITEvidenceError(f"{label} doit etre un objet JSON.")
    return payload, hashlib.sha256(raw).hexdigest()


def _bundle_file(
    raw_value: object,
    *,
    bundle_directory: Path,
    label: str,
) -> Path:
    """Resolve one immutable dependency without permitting bundle escape."""

    if not isinstance(raw_value, str) or not raw_value.strip():
        raise HistoricalPITEvidenceError(f"{label} doit etre un chemin relatif.")
    candidate = Path(raw_value.strip()).expanduser()
    if candidate.is_absolute():
        raise HistoricalPITEvidenceError(
            f"{label} doit rester relatif au bundle de preuve."
        )
    root = bundle_directory.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise HistoricalPITEvidenceError(
            f"{label} sort du bundle de preuve."
        ) from exc
    if not resolved.is_file():
        raise HistoricalPITEvidenceError(f"{label} absent: {resolved}.")
    return resolved


def _validate_capability(
    value: object, *, manifest_path: Path, source_name: str
) -> tuple[dict[str, Any], str]:
    if not isinstance(value, Mapping):
        raise HistoricalPITEvidenceError("source_capability doit etre un objet.")
    required_strings = (
        "provider",
        "endpoint",
        "source_identity",
        "api_client",
        "api_client_version",
        "api_client_code_path",
        "api_client_code_sha256",
        "materializer_code_path",
        "materializer_code_sha256",
        "provider_contract_path",
        "provider_contract_sha256",
        "mechanism",
        "trust_boundary",
        "independent_review_reference",
    )
    capability: dict[str, Any] = {}
    for field in required_strings:
        raw = value.get(field)
        if not isinstance(raw, str) or not raw.strip():
            raise HistoricalPITEvidenceError(
                f"source_capability.{field} doit etre une chaine non vide."
            )
        capability[field] = raw.strip()
    dependency_paths: set[Path] = set()
    for prefix in ("api_client_code", "materializer_code", "provider_contract"):
        hash_field = f"{prefix}_sha256"
        path_field = f"{prefix}_path"
        capability[hash_field] = _digest(
            capability[hash_field], label=f"source_capability.{hash_field}"
        )
        bound_path = _bundle_file(
            capability[path_field],
            bundle_directory=manifest_path.parent,
            label=f"source_capability.{path_field}",
        )
        if bound_path == manifest_path or bound_path in dependency_paths:
            raise HistoricalPITEvidenceError(
                f"source_capability.{path_field} doit etre une dependance "
                "distincte du manifeste et des autres roles."
            )
        dependency_paths.add(bound_path)
        if sha256_file(bound_path) != capability[hash_field]:
            raise HistoricalPITEvidenceError(
                f"source_capability.{hash_field} divergent de {bound_path}."
            )
    mechanism = capability["mechanism"]
    if mechanism not in SUPPORTED_HISTORICAL_MECHANISMS:
        raise HistoricalPITEvidenceError(
            f"source_capability.mechanism inconnu: {mechanism!r}."
        )
    exact = {
        "approved_scope": HISTORICAL_EVIDENCE_SCOPE,
        "history_mutability": "mutable_trusted_provider_archive",
        "trust_boundary": "provider_archive_access_controls_and_pinned_local_capture",
        "raw_response_archiving": "required",
    }
    for field, expected in exact.items():
        if value.get(field) != expected:
            raise HistoricalPITEvidenceError(
                f"source_capability.{field} doit valoir {expected!r}."
            )
        capability[field] = expected
    for field in (
        "historical_mutability_risk_acknowledged",
        "tls_verification_required",
    ):
        if value.get(field) is not True:
            raise HistoricalPITEvidenceError(
                f"source_capability.{field} doit etre true."
            )
        capability[field] = True
    forbidden_scopes = value.get("forbidden_scopes")
    if not isinstance(forbidden_scopes, list) or {
        "prospective_shadow",
        "live_inference",
    }.difference(map(str, forbidden_scopes)):
        raise HistoricalPITEvidenceError(
            "source_capability.forbidden_scopes doit interdire shadow et live."
        )
    capability["forbidden_scopes"] = sorted(map(str, forbidden_scopes))
    revision_available = value.get("provider_revision_timestamp_available")
    if type(revision_available) is not bool:
        raise HistoricalPITEvidenceError(
            "source_capability.provider_revision_timestamp_available doit etre booleen."
        )
    capability["provider_revision_timestamp_available"] = revision_available
    request_static_fields = value.get("request_static_fields")
    if not isinstance(request_static_fields, Mapping) or not request_static_fields:
        raise HistoricalPITEvidenceError(
            "source_capability.request_static_fields doit etre un objet non vide."
        )
    if not all(isinstance(key, str) and key for key in request_static_fields):
        raise HistoricalPITEvidenceError(
            "source_capability.request_static_fields contient une cle invalide."
        )
    try:
        canonical_static = json.loads(
            json.dumps(
                dict(request_static_fields),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise HistoricalPITEvidenceError(
            "source_capability.request_static_fields n'est pas du JSON canonique."
        ) from exc
    if canonical_static.get("endpoint") != capability["endpoint"] or (
        canonical_static.get("source_identity") != capability["source_identity"]
    ):
        raise HistoricalPITEvidenceError(
            "request_static_fields doit lier endpoint et source_identity."
        )
    capability["request_static_fields"] = canonical_static
    if mechanism == "provider_asof_state":
        if value.get("asof_parameter") != "insertion_date" or value.get(
            "asof_semantics"
        ) != "state_at_or_before_requested_instant":
            raise HistoricalPITEvidenceError(
                "La capacite provider_asof_state doit attester insertion_date et "
                "state_at_or_before_requested_instant."
            )
        capability["asof_parameter"] = "insertion_date"
        capability["asof_semantics"] = "state_at_or_before_requested_instant"
    else:
        if revision_available is not True or value.get(
            "watermark_semantics"
        ) != "last_modification_of_returned_snapshot":
            raise HistoricalPITEvidenceError(
                "Le mecanisme lastModified exige un watermark fournisseur et sa "
                "semantique sur le snapshot retourne."
            )
        if value.get("watermark_monotonicity") != "provider_contract":
            raise HistoricalPITEvidenceError(
                "La monotonie du watermark doit etre couverte par le contrat fournisseur."
            )
        capability["watermark_semantics"] = (
            "last_modification_of_returned_snapshot"
        )
        capability["watermark_monotonicity"] = "provider_contract"
    capability_sha = _sha256_json(capability)
    attestation = _APPROVED_HISTORICAL_CAPABILITIES.get(capability_sha)
    expected_attestation = {
        "source_name": source_name,
        "provider": capability["provider"],
        "endpoint": capability["endpoint"],
        "source_identity": capability["source_identity"],
        "mechanism": capability["mechanism"],
        "independent_review_reference": capability[
            "independent_review_reference"
        ],
    }
    if attestation is None or dict(attestation) != expected_attestation:
        raise HistoricalPITEvidenceError(
            "source_capability non approuvee par le registre de confiance "
            "independant du bundle."
        )
    return capability, capability_sha


def verify_historical_asof_evidence(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
    source_name: str,
    parquet_sha256: str,
    source_contract_sha256: str,
    required_days_and_cutoffs: Mapping[str, pd.Timestamp],
) -> dict[str, Any]:
    """Verify a pinned historical as-of evidence bundle.

    Every required delivery day must have a stable, TLS-verified response whose
    exact archived bytes are present locally.  The verifier trusts the reviewed
    provider capability only within its explicit offline-backtest boundary.
    """

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise HistoricalPITEvidenceError(
            f"Manifeste de preuve historique absent: {path}."
        )
    expected_digest = _digest(
        expected_manifest_sha256, label="expected_manifest_sha256"
    )
    payload, actual_digest = _load_mapping(
        path, label="Manifeste de preuve historique"
    )
    if actual_digest != expected_digest:
        raise HistoricalPITEvidenceError(
            "Le manifeste de preuve historique ne correspond pas au SHA epingle."
        )
    manifest_attestation = _APPROVED_HISTORICAL_EVIDENCE_MANIFESTS.get(
        actual_digest
    )
    if manifest_attestation is None:
        raise HistoricalPITEvidenceError(
            "manifeste historique non approuve par le registre de confiance "
            "independant du bundle."
        )
    generated_at = _utc(
        payload.get("evidence_generated_at_utc"),
        label="evidence_generated_at_utc",
    )
    if generated_at > pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=5):
        raise HistoricalPITEvidenceError(
            "evidence_generated_at_utc ne peut pas etre dans le futur."
        )
    expected_header = {
        "schema_version": HISTORICAL_EVIDENCE_SCHEMA_VERSION,
        "evidence_kind": HISTORICAL_EVIDENCE_KIND,
        "scope": HISTORICAL_EVIDENCE_SCOPE,
        "historical_backtest_pit_evidence": True,
        "prospective_capture_evidence": False,
        "source_name": source_name,
        "parquet_sha256": _digest(parquet_sha256, label="parquet_sha256"),
        "source_contract_sha256": _digest(
            source_contract_sha256, label="source_contract_sha256"
        ),
    }
    mismatches = [
        field
        for field, expected in expected_header.items()
        if type(payload.get(field)) is not type(expected)
        or payload.get(field) != expected
    ]
    if mismatches:
        raise HistoricalPITEvidenceError(
            "En-tete de preuve historique invalide: " + ", ".join(mismatches) + "."
        )
    capability, capability_sha = _validate_capability(
        payload.get("source_capability"),
        manifest_path=path,
        source_name=source_name,
    )
    if _digest(
        payload.get("source_capability_sha256"),
        label="source_capability_sha256",
    ) != capability_sha:
        raise HistoricalPITEvidenceError(
            "source_capability_sha256 divergent de la capacite canonique."
        )
    expected_manifest_attestation = {
        "source_name": source_name,
        "parquet_sha256": str(expected_header["parquet_sha256"]),
        "source_capability_sha256": capability_sha,
        "source_contract_sha256": str(
            expected_header["source_contract_sha256"]
        ),
        "independent_review_reference": capability[
            "independent_review_reference"
        ],
    }
    if dict(manifest_attestation) != expected_manifest_attestation:
        raise HistoricalPITEvidenceError(
            "attestation du manifeste historique incoherente avec son contenu."
        )
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise HistoricalPITEvidenceError("entries doit etre une liste non vide.")
    if _digest(
        payload.get("ledger_root_sha256"), label="ledger_root_sha256"
    ) != _sha256_json(entries):
        raise HistoricalPITEvidenceError(
            "ledger_root_sha256 divergent des entrees canoniques."
        )

    required = {
        str(day): pd.Timestamp(cutoff).tz_convert("UTC")
        for day, cutoff in required_days_and_cutoffs.items()
    }
    if not required:
        raise HistoricalPITEvidenceError("Aucun jour requis a verifier.")
    seen: dict[str, Mapping[str, Any]] = {}
    archive_hashes: dict[str, str] = {}
    mechanism = capability["mechanism"]
    for position, raw in enumerate(entries):
        label = f"entries[{position}]"
        if not isinstance(raw, Mapping):
            raise HistoricalPITEvidenceError(f"{label} doit etre un objet.")
        day = raw.get("delivery_day")
        if not isinstance(day, str) or _ISO_DAY.fullmatch(day) is None:
            raise HistoricalPITEvidenceError(f"{label}.delivery_day invalide.")
        try:
            if pd.Timestamp(day).date().isoformat() != day:
                raise ValueError
        except ValueError as exc:
            raise HistoricalPITEvidenceError(
                f"{label}.delivery_day invalide."
            ) from exc
        if day in seen:
            raise HistoricalPITEvidenceError(f"Jour duplique dans le ledger: {day}.")
        seen[day] = raw
        if day not in required:
            continue
        cutoff = _utc(raw.get("cutoff_utc"), label=f"{label}.cutoff_utc")
        if cutoff != required[day]:
            raise HistoricalPITEvidenceError(
                f"{label}.cutoff_utc ne correspond pas au cutoff civil attendu."
            )
        retrieved_at = _utc(
            raw.get("retrieved_at_utc"), label=f"{label}.retrieved_at_utc"
        )
        if retrieved_at < cutoff or generated_at < cutoff:
            raise HistoricalPITEvidenceError(
                f"{label}: une reconstruction historique doit etre acquise et "
                "attestee apres son cutoff."
            )
        if raw.get("tls_verified") is not True:
            raise HistoricalPITEvidenceError(f"{label}.tls_verified doit etre true.")
        if type(raw.get("causality_violations")) is not int or raw.get(
            "causality_violations"
        ) != 0:
            raise HistoricalPITEvidenceError(
                f"{label}.causality_violations doit etre l'entier zero."
            )
        scans = _strict_positive_integer(
            raw.get("response_verification_scans"),
            label=f"{label}.response_verification_scans",
        )
        response_scans = raw.get("response_sha256_scans")
        scan_times = raw.get("response_scan_retrieved_at_utc")
        if (
            not isinstance(response_scans, list)
            or not isinstance(scan_times, list)
            or len(response_scans) != scans
            or len(scan_times) != scans
            or scans < 2
        ):
            raise HistoricalPITEvidenceError(
                f"{label}: au moins deux empreintes et dates de relecture sont requises."
            )
        parsed_scan_times = [
            _utc(value, label=f"{label}.response_scan_retrieved_at_utc")
            for value in scan_times
        ]
        if len(set(parsed_scan_times)) != scans or parsed_scan_times != sorted(
            parsed_scan_times
        ):
            raise HistoricalPITEvidenceError(
                f"{label}: les dates de relecture doivent etre uniques et croissantes."
            )
        if retrieved_at != parsed_scan_times[0]:
            raise HistoricalPITEvidenceError(
                f"{label}: retrieved_at_utc doit etre la premiere relecture."
            )
        if parsed_scan_times[-1] > generated_at:
            raise HistoricalPITEvidenceError(
                f"{label}: une relecture est posterieure a la generation de la preuve."
            )
        if parsed_scan_times[-1] > pd.Timestamp.now(tz="UTC") + pd.Timedelta(
            minutes=5
        ):
            raise HistoricalPITEvidenceError(
                f"{label}: une date de relecture est dans le futur."
            )
        scan_ids = raw.get("response_scan_request_ids")
        scan_tls = raw.get("response_scan_tls_verified")
        if (
            not isinstance(scan_ids, list)
            or len(scan_ids) != scans
            or any(not isinstance(item, str) or not item.strip() for item in scan_ids)
            or len(set(scan_ids)) != scans
            or not isinstance(scan_tls, list)
            or len(scan_tls) != scans
            or any(item is not True for item in scan_tls)
        ):
            raise HistoricalPITEvidenceError(
                f"{label}: les relectures doivent avoir des identifiants uniques "
                "et une preuve TLS individuelle."
            )
        response_hashes = [
            _digest(value, label=f"{label}.response_sha256_scans")
            for value in response_scans
        ]
        if len(set(response_hashes)) != 1:
            raise HistoricalPITEvidenceError(
                f"{label}: les relectures historiques ne sont pas stables."
            )
        request = raw.get("request")
        if not isinstance(request, Mapping) or not request:
            raise HistoricalPITEvidenceError(f"{label}.request doit etre un objet.")
        if _digest(
            raw.get("request_sha256"), label=f"{label}.request_sha256"
        ) != _sha256_json(request):
            raise HistoricalPITEvidenceError(
                f"{label}.request_sha256 divergent de la requete canonique."
            )
        expected_request = dict(capability["request_static_fields"])
        archive_values = raw.get("response_scan_raw_archive_paths")
        if not isinstance(archive_values, list) or len(archive_values) != scans:
            raise HistoricalPITEvidenceError(
                f"{label}: une archive brute distincte est requise par relecture."
            )
        scan_archive_paths: list[Path] = []
        for scan_number, archive_value in enumerate(archive_values):
            archive_path = _bundle_file(
                archive_value,
                bundle_directory=path.parent,
                label=(
                    f"{label}.response_scan_raw_archive_paths[{scan_number}]"
                ),
            )
            if archive_path == path or any(
                os.path.samefile(archive_path, previous)
                for previous in scan_archive_paths
            ):
                raise HistoricalPITEvidenceError(
                    f"{label}: chaque relecture doit avoir une archive physique "
                    "distincte du manifeste et des autres relectures."
                )
            scan_archive_paths.append(archive_path)
            archive_key = str(archive_path)
            archive_digest = sha256_file(archive_path)
            archive_hashes[archive_key] = archive_digest
            if archive_digest != response_hashes[scan_number]:
                raise HistoricalPITEvidenceError(
                    f"{label}: archive brute non liee aux reponses stables."
                )
        legacy_archive = raw.get("raw_archive_path")
        legacy_archive_sha = _digest(
            raw.get("raw_archive_sha256"), label=f"{label}.raw_archive_sha256"
        )
        if (
            not isinstance(legacy_archive, str)
            or not legacy_archive.strip()
            or Path(legacy_archive) != Path(str(archive_values[0]))
            or legacy_archive_sha != response_hashes[0]
        ):
            raise HistoricalPITEvidenceError(
                f"{label}: l'archive primaire doit identifier la premiere relecture."
            )
        if mechanism == "provider_asof_state":
            requested_asof = _utc(
                raw.get("requested_asof_utc"),
                label=f"{label}.requested_asof_utc",
            )
            if requested_asof != cutoff:
                raise HistoricalPITEvidenceError(
                    f"{label}: la requete as-of n'est pas le cutoff attendu."
                )
            expected_request[str(capability["asof_parameter"])] = cutoff.isoformat()
            if _utc(
                request.get(str(capability["asof_parameter"])),
                label=f"{label}.request.{capability['asof_parameter']}",
            ) != cutoff:
                raise HistoricalPITEvidenceError(
                    f"{label}: le parametre as-of de la requete est divergent."
                )
            provider_revision = raw.get("provider_revision_max_utc")
            if (
                capability["provider_revision_timestamp_available"] is True
                and provider_revision is None
            ):
                raise HistoricalPITEvidenceError(
                    f"{label}: timestamp de revision fournisseur obligatoire."
                )
            if provider_revision is not None:
                revision_timestamp = _utc(
                    provider_revision,
                    label=f"{label}.provider_revision_max_utc",
                )
                if revision_timestamp > cutoff:
                    raise HistoricalPITEvidenceError(
                        f"{label}: revision fournisseur posterieure au cutoff."
                    )
        else:
            modified = _utc(
                raw.get("provider_last_modified_utc"),
                label=f"{label}.provider_last_modified_utc",
            )
            if modified > cutoff:
                raise HistoricalPITEvidenceError(
                    f"{label}: lastModified fournisseur posterieur au cutoff."
                )
        if dict(request) != expected_request:
            raise HistoricalPITEvidenceError(
                f"{label}: la requete n'est pas exactement la requete canonique "
                "approuvee."
            )

    missing = sorted(set(required).difference(seen))
    if missing:
        raise HistoricalPITEvidenceError(
            "Ledger historique incomplet; jours absents: " + ", ".join(missing[:10]) + "."
        )
    unexpected = sorted(set(seen).difference(required))
    if unexpected:
        raise HistoricalPITEvidenceError(
            "Le ledger doit couvrir exactement la fenetre demandee; jours en "
            "trop: " + ", ".join(unexpected[:10]) + "."
        )
    return {
        "verified": True,
        "classification": "attested_historical_asof_archive",
        "scope": HISTORICAL_EVIDENCE_SCOPE,
        "prospective_capture_evidence": False,
        "manifest_path": str(path),
        "manifest_sha256": actual_digest,
        "source_capability_sha256": capability_sha,
        "source_contract_sha256": str(
            expected_header["source_contract_sha256"]
        ),
        "ledger_root_sha256": str(payload["ledger_root_sha256"]),
        "mechanism": mechanism,
        "required_days": len(required),
        "ledger_days": len(entries),
        "raw_archives_verified": len(archive_hashes),
        "trust_boundary": capability["trust_boundary"],
        "independent_trust_anchor_verified": True,
        "independent_review_reference": capability[
            "independent_review_reference"
        ],
        "evidence_generated_at_utc": generated_at.isoformat(),
    }


__all__ = [
    "HISTORICAL_EVIDENCE_KIND",
    "HISTORICAL_EVIDENCE_SCHEMA_VERSION",
    "HISTORICAL_EVIDENCE_SCOPE",
    "HistoricalPITEvidenceError",
    "SUPPORTED_HISTORICAL_MECHANISMS",
    "sha256_file",
    "verify_historical_asof_evidence",
]
