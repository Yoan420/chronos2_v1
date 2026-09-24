from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from chronos2_exogenous.feature_bank import (
    ConsumerRoute,
    ExogenousBankError,
    ParquetFeatureSource,
    build_exogenous_bank,
    cutoff_by_delivery_hour,
    delivery_utc_index,
)
from chronos2_exogenous.pit_evidence import HISTORICAL_EVIDENCE_KIND
from chronos2_exogenous.panel import build_origin_panel
import chronos2_exogenous.pit_evidence as pit_evidence_module


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _source_contract_sha256() -> str:
    return _sha256_json(
        {
            "schema_version": 1,
            "source_name": "wind_fr",
            "family": "weather",
            "value_columns": {"fr_wind_fcst": "value"},
            "consumer_route": ["chronos"],
            "timestamp_column": "value_time_utc",
            "cutoff_column": "snapshot_time_utc",
            "forecast_origin_timezone": "Europe/Paris",
            "source_cutoff_timezone": "Europe/Paris",
            "cutoff_time": "08:00",
            "information_time_columns": [
                "snapshot_time_utc",
                "revision_time_utc",
            ],
            "age_column": "revision_time_utc",
            "eligibility_column": None,
            "operational_eligibility_column": None,
            "stage_column": None,
            "allowed_stages": [],
            "known_future": True,
            "transform": "identity",
            "production_evidence_kind": "attested_historical_asof_archive",
        }
    )


def _source(path: Path) -> ParquetFeatureSource:
    return ParquetFeatureSource(
        name="wind_fr",
        family="weather",
        path=path,
        value_columns={"fr_wind_fcst": "value"},
        route=ConsumerRoute(("chronos",)),
        cutoff_column="snapshot_time_utc",
        information_time_columns=("snapshot_time_utc", "revision_time_utc"),
        age_column="revision_time_utc",
    )


def _write_attested_bundle(tmp_path: Path, *, day: str = "2026-01-10") -> ParquetFeatureSource:
    parquet = tmp_path / "wind.parquet"
    index = delivery_utc_index(day, day)
    cutoff = cutoff_by_delivery_hour(index)
    pd.DataFrame(
        {
            "value_time_utc": index,
            "snapshot_time_utc": cutoff,
            "revision_time_utc": cutoff - pd.Timedelta(hours=2),
            "value": np.arange(len(index), dtype=float),
        }
    ).to_parquet(parquet, index=False)
    raw_paths = [
        tmp_path / "raw-response-scan-1.bin",
        tmp_path / "raw-response-scan-2.bin",
    ]
    for raw_path in raw_paths:
        raw_path.write_bytes(b"exact historical response bytes")
    raw_sha = _sha256_file(raw_paths[0])
    client_code = tmp_path / "client.py"
    client_code.write_text("# pinned API client\n", encoding="utf-8")
    materializer_code = tmp_path / "materializer.py"
    materializer_code.write_text("# pinned materializer\n", encoding="utf-8")
    provider_contract = tmp_path / "provider-contract.txt"
    provider_contract.write_text(
        "revision_date returns the state known at the requested insertion date\n",
        encoding="utf-8",
    )
    capability = {
        "provider": "Saturn",
        "endpoint": "/series/state",
        "source_identity": "power.fr.wind.fcst",
        "api_client": "tshistory_lite",
        "api_client_version": "0.5",
        "api_client_code_path": client_code.name,
        "api_client_code_sha256": _sha256_file(client_code),
        "materializer_code_path": materializer_code.name,
        "materializer_code_sha256": _sha256_file(materializer_code),
        "provider_contract_path": provider_contract.name,
        "provider_contract_sha256": _sha256_file(provider_contract),
        "mechanism": "provider_asof_state",
        "trust_boundary": "provider_archive_access_controls_and_pinned_local_capture",
        "independent_review_reference": "DATA-GOV-42",
        "approved_scope": "offline_training_backtest_only",
        "history_mutability": "mutable_trusted_provider_archive",
        "raw_response_archiving": "required",
        "historical_mutability_risk_acknowledged": True,
        "tls_verification_required": True,
        "forbidden_scopes": ["live_inference", "prospective_shadow"],
        "provider_revision_timestamp_available": False,
        "asof_parameter": "insertion_date",
        "asof_semantics": "state_at_or_before_requested_instant",
        "request_static_fields": {
            "endpoint": "/series/state",
            "source_identity": "power.fr.wind.fcst",
        },
    }
    cutoff_one = pd.Timestamp(cutoff[0]).isoformat()
    request = {
        "endpoint": "/series/state",
        "source_identity": "power.fr.wind.fcst",
        "insertion_date": cutoff_one,
    }
    entry = {
        "delivery_day": day,
        "cutoff_utc": cutoff_one,
        "retrieved_at_utc": "2026-02-01T00:00:00+00:00",
        "tls_verified": True,
        "causality_violations": 0,
        "response_verification_scans": 2,
        "response_sha256_scans": [raw_sha, raw_sha],
        "response_scan_retrieved_at_utc": [
            "2026-02-01T00:00:00+00:00",
            "2026-02-01T00:00:01+00:00",
        ],
        "response_scan_request_ids": ["scan-1", "scan-2"],
        "response_scan_tls_verified": [True, True],
        "request": request,
        "request_sha256": _sha256_json(request),
        "response_scan_raw_archive_paths": [path.name for path in raw_paths],
        "raw_archive_path": raw_paths[0].name,
        "raw_archive_sha256": raw_sha,
        "requested_asof_utc": cutoff_one,
        "provider_revision_max_utc": None,
    }
    entries = [entry]
    manifest = {
        "schema_version": 1,
        "evidence_kind": HISTORICAL_EVIDENCE_KIND,
        "scope": "offline_training_backtest_only",
        "historical_backtest_pit_evidence": True,
        "prospective_capture_evidence": False,
        "source_name": "wind_fr",
        "parquet_sha256": _sha256_file(parquet),
        "source_contract_sha256": _source_contract_sha256(),
        "evidence_generated_at_utc": "2026-02-01T00:00:02+00:00",
        "source_capability": capability,
        "source_capability_sha256": _sha256_json(capability),
        "entries": entries,
        "ledger_root_sha256": _sha256_json(entries),
    }
    evidence_path = tmp_path / "historical-evidence.json"
    evidence_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evidence_sha = _sha256_file(evidence_path)
    sidecar = {
        "output_sha256": _sha256_file(parquet),
        "causality_violations": 0,
        "cutoff_time": "08:00",
        "cutoff_timezone": "Europe/Paris",
        "historical_backtest_pit_evidence": True,
        "historical_evidence_kind": HISTORICAL_EVIDENCE_KIND,
        "historical_evidence_manifest_sha256": evidence_sha,
    }
    audit_path = tmp_path / "wind.audit.json"
    audit_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return replace(
        _source(parquet),
        audit_path=audit_path,
        production_evidence_kind="attested_historical_asof_archive",
        historical_evidence_manifest_path=evidence_path,
        historical_evidence_manifest_sha256=evidence_sha,
    )


def _approve_bundle(source: ParquetFeatureSource, monkeypatch) -> None:
    payload = json.loads(
        Path(source.historical_evidence_manifest_path).read_text(encoding="utf-8")
    )
    capability = payload["source_capability"]
    capability_sha = _sha256_json(capability)
    attestation = {
        "source_name": "wind_fr",
        "provider": capability["provider"],
        "endpoint": capability["endpoint"],
        "source_identity": capability["source_identity"],
        "mechanism": capability["mechanism"],
        "independent_review_reference": capability["independent_review_reference"],
    }
    monkeypatch.setattr(
        pit_evidence_module,
        "_APPROVED_HISTORICAL_CAPABILITIES",
        MappingProxyType({capability_sha: MappingProxyType(attestation)}),
    )
    evidence_sha = _sha256_file(Path(source.historical_evidence_manifest_path))
    manifest_attestation = {
        "source_name": "wind_fr",
        "parquet_sha256": payload["parquet_sha256"],
        "source_capability_sha256": capability_sha,
        "source_contract_sha256": payload["source_contract_sha256"],
        "independent_review_reference": capability["independent_review_reference"],
    }
    monkeypatch.setattr(
        pit_evidence_module,
        "_APPROVED_HISTORICAL_EVIDENCE_MANIFESTS",
        MappingProxyType(
            {evidence_sha: MappingProxyType(manifest_attestation)}
        ),
    )


def _rewrite_evidence(
    source: ParquetFeatureSource,
    mutate,
) -> ParquetFeatureSource:
    evidence = Path(source.historical_evidence_manifest_path)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    mutate(payload)
    payload["source_capability_sha256"] = _sha256_json(
        payload["source_capability"]
    )
    payload["ledger_root_sha256"] = _sha256_json(payload["entries"])
    evidence.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    new_sha = _sha256_file(evidence)
    sidecar_path = Path(source.audit_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["historical_evidence_manifest_sha256"] = new_sha
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return replace(source, historical_evidence_manifest_sha256=new_sha)


def _rewrite_parquet(source: ParquetFeatureSource, mutate) -> ParquetFeatureSource:
    parquet = Path(source.path)
    frame = pd.read_parquet(parquet)
    frame = mutate(frame.copy())
    frame.to_parquet(parquet, index=False)
    parquet_sha = _sha256_file(parquet)

    evidence = Path(source.historical_evidence_manifest_path)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["parquet_sha256"] = parquet_sha
    evidence.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evidence_sha = _sha256_file(evidence)

    sidecar_path = Path(source.audit_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["output_sha256"] = parquet_sha
    sidecar["historical_evidence_manifest_sha256"] = evidence_sha
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return replace(source, historical_evidence_manifest_sha256=evidence_sha)


def _entry(payload: dict[str, object]) -> dict[str, object]:
    entries = payload["entries"]
    assert isinstance(entries, list) and entries
    result = entries[0]
    assert isinstance(result, dict)
    return result


def test_locally_fabricated_bundle_is_rejected_without_independent_trust_anchor(
    tmp_path: Path,
) -> None:
    source = _write_attested_bundle(tmp_path)

    with pytest.raises(ExogenousBankError, match="registre de confiance"):
        build_exogenous_bank(
            [source], start_day="2026-01-10", end_day="2026-01-10"
        )


def test_attested_history_is_backtest_ready_but_never_prospective(
    tmp_path: Path, monkeypatch
) -> None:
    source = _write_attested_bundle(tmp_path)
    _approve_bundle(source, monkeypatch)

    bank = build_exogenous_bank(
        [source],
        start_day="2026-01-10",
        end_day="2026-01-10",
        require_historical_backtest_evidence=True,
    )

    audit = bank.audit["sources"]["wind_fr"]
    assert bank.audit["historical_backtest_ready"] is True
    assert bank.audit["production_ready"] is False
    assert audit["historical_backtest_pit_evidence"] is True
    assert audit["production_pit_evidence"] is False
    assert audit["source_audit"]["historical_backtest_verified"] is True
    assert audit["source_audit"]["prospective_capture_verified"] is False
    assert bank.audit["historical_evidence_manifest_hashes"] == {
        "wind_fr": source.historical_evidence_manifest_sha256
    }
    assert bank.audit["historical_source_contract_hashes"] == {
        "wind_fr": _source_contract_sha256()
    }
    assert Path(
        bank.audit["historical_evidence_manifest_paths"]["wind_fr"]
    ).resolve() == Path(source.historical_evidence_manifest_path).resolve()
    horizon = delivery_utc_index("2026-01-10", "2026-01-10")
    target_index = pd.DatetimeIndex(
        [horizon[0] - pd.Timedelta(hours=1), *horizon]
    )
    panel = build_origin_panel(
        bank,
        {"FR": pd.Series(np.arange(len(target_index)), index=target_index)},
        delivery_days=["2026-01-10"],
        context_length=1,
        zones=["FR"],
        require_complete_covariates=False,
    )
    assert panel.audit["historical_backtest_ready"] is True
    assert panel.audit["historical_backtest_pit_evidence"] == {"FR": True}
    assert panel.audit["exogenous_banks"]["FR"][
        "historical_source_contract_hashes"
    ] == {"wind_fr": _source_contract_sha256()}
    assert panel.audit["production_ready"] is False
    assert panel.audit["production_pit_evidence"] == {"FR": False}
    with pytest.raises(ExogenousBankError, match="operationnelle"):
        build_exogenous_bank(
            [source],
            start_day="2026-01-10",
            end_day="2026-01-10",
            require_operational_evidence=True,
        )


def test_attested_history_fails_closed_when_raw_response_changes(
    tmp_path: Path, monkeypatch
) -> None:
    source = _write_attested_bundle(tmp_path)
    _approve_bundle(source, monkeypatch)
    (tmp_path / "raw-response-scan-1.bin").write_bytes(b"tampered")

    with pytest.raises(ExogenousBankError, match="archive brute non liee"):
        build_exogenous_bank(
            [source], start_day="2026-01-10", end_day="2026-01-10"
        )


def test_attested_history_fails_when_capability_dependency_changes(
    tmp_path: Path, monkeypatch
) -> None:
    source = _write_attested_bundle(tmp_path)
    _approve_bundle(source, monkeypatch)
    (tmp_path / "client.py").write_text("# modified client\n", encoding="utf-8")

    with pytest.raises(ExogenousBankError, match="api_client_code_sha256.*divergent"):
        build_exogenous_bank(
            [source], start_day="2026-01-10", end_day="2026-01-10"
        )


def test_attested_history_cannot_claim_live_scope(tmp_path: Path, monkeypatch) -> None:
    source = _write_attested_bundle(tmp_path)
    source = _rewrite_evidence(
        source,
        lambda payload: payload.__setitem__("prospective_capture_evidence", True),
    )
    _approve_bundle(source, monkeypatch)

    with pytest.raises(ExogenousBankError, match="En-tete de preuve historique"):
        build_exogenous_bank(
            [source], start_day="2026-01-10", end_day="2026-01-10"
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("production_pit_evidence", True),
        ("production_evidence_kind", "prospective_capture"),
        ("prospective_capture_evidence", True),
    ],
)
def test_historical_sidecar_cannot_claim_production_scope(
    tmp_path: Path, monkeypatch, field: str, value: object
) -> None:
    source = _write_attested_bundle(tmp_path)
    sidecar_path = Path(source.audit_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar[field] = value
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    _approve_bundle(source, monkeypatch)

    with pytest.raises(ExogenousBankError, match="capture prospective.*production"):
        build_exogenous_bank(
            [source], start_day="2026-01-10", end_day="2026-01-10"
        )


def test_attested_history_requires_every_requested_delivery_day(
    tmp_path: Path, monkeypatch
) -> None:
    source = _write_attested_bundle(tmp_path)
    _approve_bundle(source, monkeypatch)

    with pytest.raises(ExogenousBankError, match="jours absents"):
        build_exogenous_bank(
            [source], start_day="2026-01-10", end_day="2026-01-11"
        )


def test_historical_requirement_rejects_partial_hourly_coverage(
    tmp_path: Path, monkeypatch
) -> None:
    source = _rewrite_parquet(
        _write_attested_bundle(tmp_path), lambda frame: frame.iloc[:-1]
    )
    _approve_bundle(source, monkeypatch)

    with pytest.raises(ExogenousBankError, match="heure.*incomplete"):
        build_exogenous_bank(
            [source],
            start_day="2026-01-10",
            end_day="2026-01-10",
            require_historical_backtest_evidence=True,
        )


@pytest.mark.parametrize(
    ("day", "physical_hours"),
    [("2025-03-30", 23), ("2025-10-26", 25)],
)
def test_attested_history_preserves_exact_dst_physical_hours(
    tmp_path: Path, monkeypatch, day: str, physical_hours: int
) -> None:
    source = _write_attested_bundle(tmp_path, day=day)
    _approve_bundle(source, monkeypatch)

    bank = build_exogenous_bank(
        [source],
        start_day=day,
        end_day=day,
        require_historical_backtest_evidence=True,
    )

    assert len(bank.frame) == physical_hours
    assert bank.audit["day_length_counts"] == {str(physical_hours): 1}
    assert bank.audit["historical_backtest_ready"] is True


def test_capability_dependencies_cannot_escape_evidence_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside-client.py"
    outside.write_text("# outside\n", encoding="utf-8")

    def mutate(payload: dict[str, object]) -> None:
        capability = payload["source_capability"]
        assert isinstance(capability, dict)
        capability["api_client_code_path"] = f"../{outside.name}"
        capability["api_client_code_sha256"] = _sha256_file(outside)

    source = _rewrite_evidence(_write_attested_bundle(tmp_path), mutate)
    _approve_bundle(source, monkeypatch)

    with pytest.raises(ExogenousBankError, match="sort du bundle"):
        build_exogenous_bank(
            [source], start_day="2026-01-10", end_day="2026-01-10"
        )


def test_raw_archive_cannot_use_an_absolute_external_path(
    tmp_path: Path, monkeypatch
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside-response.bin"
    outside.write_bytes(b"exact historical response bytes")

    def mutate(payload: dict[str, object]) -> None:
        entry = _entry(payload)
        entry["response_scan_raw_archive_paths"] = [
            str(outside.resolve()),
            "raw-response-scan-2.bin",
        ]
        entry["raw_archive_path"] = str(outside.resolve())

    source = _rewrite_evidence(_write_attested_bundle(tmp_path), mutate)
    _approve_bundle(source, monkeypatch)

    with pytest.raises(ExogenousBankError, match="relatif au bundle"):
        build_exogenous_bank(
            [source], start_day="2026-01-10", end_day="2026-01-10"
        )


def test_attested_history_rejects_noncanonical_query_even_when_rehashed(
    tmp_path: Path, monkeypatch
) -> None:
    def mutate(payload: dict[str, object]) -> None:
        entry = _entry(payload)
        request = entry["request"]
        assert isinstance(request, dict)
        request["aggregation"] = "observed_final"
        entry["request_sha256"] = _sha256_json(request)

    source = _rewrite_evidence(_write_attested_bundle(tmp_path), mutate)
    _approve_bundle(source, monkeypatch)

    with pytest.raises(ExogenousBankError, match="requete canonique"):
        build_exogenous_bank(
            [source], start_day="2026-01-10", end_day="2026-01-10"
        )


def test_approved_parquet_cannot_be_reinterpreted_with_another_feature_contract(
    tmp_path: Path, monkeypatch
) -> None:
    source = _write_attested_bundle(tmp_path)
    _approve_bundle(source, monkeypatch)
    reinterpreted = replace(
        source,
        value_columns={"fr_temperature_fcst": "value"},
        route=ConsumerRoute(("residual",)),
    )

    with pytest.raises(ExogenousBankError, match="source_contract_sha256"):
        build_exogenous_bank(
            [reinterpreted], start_day="2026-01-10", end_day="2026-01-10"
        )


def test_attested_history_rejects_naive_and_future_audit_timestamps(
    tmp_path: Path, monkeypatch
) -> None:
    def naive_cutoff(payload: dict[str, object]) -> None:
        _entry(payload)["cutoff_utc"] = "2026-01-09T07:00:00"

    source = _rewrite_evidence(_write_attested_bundle(tmp_path), naive_cutoff)
    _approve_bundle(source, monkeypatch)
    with pytest.raises(ExogenousBankError, match="timestamp UTC"):
        build_exogenous_bank(
            [source], start_day="2026-01-10", end_day="2026-01-10"
        )

    def future_generation(payload: dict[str, object]) -> None:
        payload["evidence_generated_at_utc"] = "2099-01-01T00:00:00+00:00"

    source = _rewrite_evidence(_write_attested_bundle(tmp_path), future_generation)
    _approve_bundle(source, monkeypatch)
    with pytest.raises(ExogenousBankError, match="dans le futur"):
        build_exogenous_bank(
            [source], start_day="2026-01-10", end_day="2026-01-10"
        )


@pytest.mark.parametrize("failure", ["duplicate_id", "missing_tls"])
def test_attested_history_requires_individually_identified_tls_rereads(
    tmp_path: Path, monkeypatch, failure: str
) -> None:
    def mutate(payload: dict[str, object]) -> None:
        entry = _entry(payload)
        if failure == "duplicate_id":
            entry["response_scan_request_ids"] = ["same", "same"]
        else:
            entry["response_scan_tls_verified"] = [True, False]

    source = _rewrite_evidence(_write_attested_bundle(tmp_path), mutate)
    _approve_bundle(source, monkeypatch)

    with pytest.raises(ExogenousBankError, match="identifiants uniques.*TLS"):
        build_exogenous_bank(
            [source], start_day="2026-01-10", end_day="2026-01-10"
        )


def _as_last_modified_contract(payload: dict[str, object], *, late: bool) -> None:
    capability = payload["source_capability"]
    assert isinstance(capability, dict)
    capability.update(
        {
            "provider": "JAO Core",
            "endpoint": "/initialComputation",
            "source_identity": "initialComputation/Presolved=true",
            "api_client": "chronos2_jao_client",
            "api_client_version": "1",
            "mechanism": "current_snapshot_last_modified_watermark",
            "provider_revision_timestamp_available": True,
            "watermark_semantics": "last_modification_of_returned_snapshot",
            "watermark_monotonicity": "provider_contract",
        }
    )
    capability.pop("asof_parameter")
    capability.pop("asof_semantics")
    capability["request_static_fields"] = {
        "endpoint": "/initialComputation",
        "source_identity": "initialComputation/Presolved=true",
        "presolved": True,
    }
    entries = payload["entries"]
    assert isinstance(entries, list)
    entry = entries[0]
    assert isinstance(entry, dict)
    cutoff = pd.Timestamp(entry["cutoff_utc"])
    entry.pop("requested_asof_utc")
    entry.pop("provider_revision_max_utc")
    entry["provider_last_modified_utc"] = (
        cutoff + pd.Timedelta(minutes=1)
        if late
        else cutoff - pd.Timedelta(hours=1)
    ).isoformat()
    request = entry["request"]
    assert isinstance(request, dict)
    request.clear()
    request.update(
        {
            "endpoint": "/initialComputation",
            "source_identity": "initialComputation/Presolved=true",
            "presolved": True,
        }
    )
    entry["request_sha256"] = _sha256_json(request)


def test_attested_last_modified_snapshot_is_offline_only(
    tmp_path: Path, monkeypatch
) -> None:
    source = _rewrite_evidence(
        _write_attested_bundle(tmp_path),
        lambda payload: _as_last_modified_contract(payload, late=False),
    )
    _approve_bundle(source, monkeypatch)

    bank = build_exogenous_bank(
        [source],
        start_day="2026-01-10",
        end_day="2026-01-10",
        require_historical_backtest_evidence=True,
    )

    evidence = bank.audit["sources"]["wind_fr"]["source_audit"][
        "historical_evidence"
    ]
    assert evidence["mechanism"] == "current_snapshot_last_modified_watermark"
    assert evidence["prospective_capture_evidence"] is False


def test_attested_last_modified_rejects_post_cutoff_watermark(
    tmp_path: Path, monkeypatch
) -> None:
    source = _rewrite_evidence(
        _write_attested_bundle(tmp_path),
        lambda payload: _as_last_modified_contract(payload, late=True),
    )
    _approve_bundle(source, monkeypatch)

    with pytest.raises(ExogenousBankError, match="lastModified.*posterieur"):
        build_exogenous_bank(
            [source], start_day="2026-01-10", end_day="2026-01-10"
        )
