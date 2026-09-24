from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from chronos2_exogenous.feature_bank import ExogenousBankError, build_exogenous_bank
from chronos2_exogenous.live_sources import load_live_source_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_manifest(tmp_path: Path, *, kind: str = "prospective_capture") -> Path:
    parquet = tmp_path / "captured.parquet"
    frame = pd.DataFrame({
        "value_time_utc": ["2026-09-04T00:00:00Z"],
        "snapshot_time_utc": ["2026-09-03T05:00:00Z"],
        "revision_time_utc": ["2026-09-03T05:00:00Z"],
        "operational_eligible": [True],
        "value": [10.0],
    })
    frame.to_parquet(parquet, index=False)
    digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
    audit = tmp_path / "captured.audit.json"
    audit.write_text(json.dumps({
        "parquet_sha256": digest,
        "causality_violations": 0,
        "operational_capture_violations": 0,
        "production_evidence_kind": "prospective_capture",
        "production_pit_evidence": True,
    }), encoding="utf-8")
    manifest = tmp_path / "live_sources.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "manifest_kind": "chronos2_exogenous_live_sources",
        "production_evidence_kind": "prospective_capture",
        "zones": {"FR": [{
            "name": "weather_fr_temperature_live",
            "family": "weather",
            "path": parquet.name,
            "audit_path": audit.name,
            "value_columns": {"fr_temperature_fcst": "value"},
            "cutoff_column": "snapshot_time_utc",
            "information_time_columns": ["snapshot_time_utc", "revision_time_utc"],
            "operational_eligibility_column": "operational_eligible",
            "production_evidence_kind": kind,
        }]},
    }), encoding="utf-8")
    return manifest


def test_manifest_loads_only_explicit_prospective_sources(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path)
    sources, audit = load_live_source_manifest(path, zone="FR")
    assert len(sources) == 1
    assert sources[0].production_evidence_kind == "prospective_capture"
    assert audit["evidence_created_by_loader"] is False
    assert len(audit["sha256"]) == 64


def test_manifest_never_reclassifies_versioned_history(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, kind="versioned_revision_history")
    with pytest.raises(ExogenousBankError, match="prospective_capture"):
        load_live_source_manifest(path, zone="FR")


def test_sidecar_claim_is_still_verified_by_feature_bank(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path)
    sources, _ = load_live_source_manifest(path, zone="FR")
    audit_path = sources[0].audit_path
    assert audit_path is not None
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["parquet_sha256"] = "0" * 64
    audit_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExogenousBankError, match="non lie au Parquet"):
        build_exogenous_bank(
            sources,
            start_day="2026-09-04",
            end_day="2026-09-04",
            require_operational_evidence=True,
        )


def test_full_fr_example_uses_the_training_source_identities() -> None:
    example = json.loads(
        (
            PROJECT_ROOT
            / "config"
            / "chronos2_exogenous_live_sources.example.json"
        ).read_text(encoding="utf-8")
    )
    sources = example["zones"]["FR"]
    assert {source["name"] for source in sources} == {
        "residual_load",
        "weather_fr_temperature",
        "weather_fr_wind_generation",
        "weather_fr_solar_generation",
        "fuel_market",
        "flowbased_core",
    }
    assert all(
        source["production_evidence_kind"] == "prospective_capture"
        for source in sources
    )
