"""Strict declaration loader for prospectively captured live features.

This module validates declarations only.  It never creates, upgrades, or
reclassifies PIT evidence; every Parquet must already have an independently
written audit sidecar accepted by :mod:`chronos2_exogenous.feature_bank`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .feature_bank import ConsumerRoute, ExogenousBankError, ParquetFeatureSource


LIVE_SOURCE_MANIFEST_VERSION = 1
_SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path(value: object, *, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ExogenousBankError(f"{label}: chemin absent.")
    result = Path(value).expanduser()
    if not result.is_absolute():
        result = base / result
    result = result.resolve()
    if not result.is_file():
        raise ExogenousBankError(f"{label}: fichier absent: {result}.")
    return result


def load_live_source_manifest(
    manifest_path: str | Path,
    *,
    zone: str,
) -> tuple[tuple[ParquetFeatureSource, ...], dict[str, Any]]:
    """Load sources that explicitly claim prospective capture, fail closed."""

    path = Path(manifest_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExogenousBankError(f"Manifeste live illisible: {path}.") from exc
    if not isinstance(payload, Mapping):
        raise ExogenousBankError("Le manifeste live doit etre un objet JSON.")
    expected = {
        "schema_version": LIVE_SOURCE_MANIFEST_VERSION,
        "manifest_kind": "chronos2_exogenous_live_sources",
        "production_evidence_kind": "prospective_capture",
    }
    failures = [
        key for key, value in expected.items()
        if type(payload.get(key)) is not type(value) or payload.get(key) != value
    ]
    if failures:
        raise ExogenousBankError("Contrat du manifeste live invalide: " + ", ".join(failures))
    code = str(zone).strip().upper()
    zones = payload.get("zones")
    raw_sources = zones.get(code) if isinstance(zones, Mapping) else None
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ExogenousBankError(f"{code}: aucune source prospective declaree.")
    sources: list[ParquetFeatureSource] = []
    for index, raw in enumerate(raw_sources):
        label = f"zones.{code}[{index}]"
        if not isinstance(raw, Mapping):
            raise ExogenousBankError(f"{label}: declaration invalide.")
        if raw.get("production_evidence_kind") != "prospective_capture":
            raise ExogenousBankError(f"{label}: seule prospective_capture est acceptee.")
        name = str(raw.get("name", ""))
        family = str(raw.get("family", ""))
        if not _SAFE.fullmatch(name) or not _SAFE.fullmatch(family):
            raise ExogenousBankError(f"{label}: name/family invalides.")
        values = raw.get("value_columns")
        if not isinstance(values, Mapping):
            raise ExogenousBankError(f"{label}: value_columns doit etre un objet.")
        info = raw.get("information_time_columns")
        if not isinstance(info, list) or not info or not all(isinstance(v, str) and v for v in info):
            raise ExogenousBankError(f"{label}: information_time_columns invalide.")
        consumers = raw.get("consumers", ["chronos"])
        if not isinstance(consumers, list) or not consumers:
            raise ExogenousBankError(f"{label}: consumers invalide.")
        allowed = raw.get("allowed_stages", [])
        if not isinstance(allowed, list):
            raise ExogenousBankError(f"{label}: allowed_stages invalide.")
        source = ParquetFeatureSource(
            name=name,
            family=family,
            path=_path(raw.get("path"), base=path.parent, label=f"{label}.path"),
            audit_path=_path(raw.get("audit_path"), base=path.parent, label=f"{label}.audit_path"),
            value_columns={str(key): str(value) for key, value in values.items()},
            route=ConsumerRoute(tuple(str(value).casefold() for value in consumers)),
            timestamp_column=str(raw.get("timestamp_column", "value_time_utc")),
            cutoff_column=str(raw["cutoff_column"]) if raw.get("cutoff_column") else None,
            cutoff_timezone=str(raw["cutoff_timezone"]) if raw.get("cutoff_timezone") else None,
            information_time_columns=tuple(info),
            age_column=str(raw["age_column"]) if raw.get("age_column") else None,
            eligibility_column=str(raw["eligibility_column"]) if raw.get("eligibility_column") else None,
            operational_eligibility_column=(str(raw["operational_eligibility_column"]) if raw.get("operational_eligibility_column") else None),
            stage_column=str(raw["stage_column"]) if raw.get("stage_column") else None,
            allowed_stages=tuple(str(value) for value in allowed),
            known_future=raw.get("known_future", True) is True,
            transform=str(raw.get("transform", "identity")),
            production_evidence_kind="prospective_capture",
        )
        source.validate()
        sources.append(source)
    names = [source.name for source in sources]
    if len(names) != len(set(names)):
        raise ExogenousBankError(f"{code}: noms de sources dupliques.")
    return tuple(sources), {
        "path": str(path),
        "sha256": _sha256(path),
        "schema_version": LIVE_SOURCE_MANIFEST_VERSION,
        "zone": code,
        "source_names": names,
        "evidence_created_by_loader": False,
    }


__all__ = ["LIVE_SOURCE_MANIFEST_VERSION", "load_live_source_manifest"]
