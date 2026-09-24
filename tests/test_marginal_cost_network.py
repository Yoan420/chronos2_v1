"""Offline tests for the original, non-imputed physical network contract."""
from copy import deepcopy
import gzip
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.jao_flowbased import (
    CORE_PTDF_ZONES, JAO_CORE_DATA_URL, expected_cutoff_utc, local_day_utc_bounds,
)
from marginal_cost_expert.network import (
    AHC_HUBS, NetworkContractError, archive_initial_probe, load_network_day,
    normalise_network_payload,
)


def payload(day="2026-06-24", *, captured=False):
    start, end = local_day_utc_bounds(day)
    cutoff = expected_cutoff_utc(day)
    rows = []
    for number, stamp in enumerate(pd.date_range(start, end, freq="h", inclusive="left")):
        row = {"id": number, "dateTimeUtc": stamp.isoformat(), "cnec": True,
               "presolved": True, "cneName": "Line", "ram": 100.,
               "fmax": 140., "frm": 10., "frefInit": 30., "fcore": 12.}
        row.update({f"ptdf_{hub}": (number + 1) / 100 for hub in CORE_PTDF_ZONES})
        row.update({f"ptdf_{hub}": .01 if pd.Timestamp(day).date() >= pd.Timestamp("2026-06-11").date()
                    else None for hub in AHC_HUBS})
        rows.append(row)
    return {"fetch": {"api_base_url": JAO_CORE_DATA_URL, "endpoint": "initialComputation",
                      "filters": {"Presolved": True}, "start_utc": start.isoformat(),
                      "end_utc": end.isoformat(), "total_rows": len(rows),
                      "last_modified_utc": (cutoff - pd.Timedelta(hours=5)).isoformat(),
                      "retrieved_at_utc": (cutoff - pd.Timedelta(hours=1) if captured else
                                           end + pd.Timedelta(days=20)).isoformat()}, "records": rows}


def write_pair(root, day, data, extra_audit=None):
    directory = root / "raw" / "initialComputation"
    directory.mkdir(parents=True)
    blob = gzip.compress(json.dumps(data).encode())
    path = directory / f"{day}.json.gz"
    path.write_bytes(blob)
    (directory / f"{day}.audit.json").write_text(json.dumps({
        "raw_gzip_sha256": hashlib.sha256(blob).hexdigest(), **(extra_audit or {})}), encoding="utf-8")
    return path


@pytest.mark.parametrize("day,hours", [("2025-03-30", 23), ("2025-10-26", 25), ("2026-06-24", 24)])
def test_exact_dst_complete_original_domain(day, hours):
    result = normalise_network_payload(payload(day), day)
    assert len(result.qualification) == hours
    assert result.qualification.delivery_start_utc.is_unique
    assert result.audit["inputs_qualified"]
    assert result.audit["available_hours"] == hours
    assert not result.audit["operational_pit_eligible"]
    assert not result.audit["publication_timestamp_certified"]
    assert not result.audit["domain_reference_qualified"]
    assert not result.audit["usable_zero_based_domain"]
    assert not result.audit["boundary_qualified"]
    assert not result.audit["imputation_performed"]


def test_full_23_ptdf_and_ram_reference_preserved():
    result = normalise_network_payload(payload(), "2026-06-24")
    assert len(result.active_hubs) == 23
    assert len([col for col in result.constraints if col.startswith("ptdf_")]) == 23
    assert result.inactive_hubs == ()
    first = result.constraints.iloc[0]
    assert first.ram_mw == 100.
    assert first.fref_init_mw == 30.
    assert first.fcore_mw == 12.
    # No quiet translation to RAM=118 or zero-reference certification.
    assert "refprog_balanced" in result.audit["domain_reference"]


def test_pre_ahc_null_hubs_inactive_not_zeros():
    result = normalise_network_payload(payload("2026-06-10"), "2026-06-10")
    assert set(result.active_hubs) == set(CORE_PTDF_ZONES)
    assert set(result.inactive_hubs) == set(AHC_HUBS)
    assert result.constraints[[f"ptdf_{hub}" for hub in AHC_HUBS]].isna().all().all()
    assert len([col for col in result.active_constraints() if col.startswith("ptdf_")]) == 14


def test_post_ahc_missing_hub_abstains_whole_day():
    data = payload()
    for row in data["records"]:
        row.pop("ptdf_DE_DK1_VH")
    result = normalise_network_payload(data, "2026-06-24")
    assert not result.qualification.inputs_qualified.any()
    assert result.constraints.ptdf_DE_DK1_VH.isna().all()
    assert "DE_DK1_VH" in result.audit["missing_ptdf_columns"]


def test_unrecognized_hub_never_dropped_or_filled():
    data = payload()
    for row in data["records"]:
        row["ptdf_NEW_PROVIDER_HUB"] = None
    result = normalise_network_payload(data, "2026-06-24")
    assert "NEW_PROVIDER_HUB" in result.active_hubs
    assert "ptdf_NEW_PROVIDER_HUB" in result.active_constraints()
    assert not result.qualification.inputs_qualified.any()


def test_missing_hours_never_imputed_even_old_feature_store_was_imputed():
    data = payload("2025-10-26")
    data["records"] = data["records"][:1]
    data["fetch"]["total_rows"] = 1
    result = normalise_network_payload(data, "2025-10-26", source_audit={"missing_source_mtus": 24})
    assert len(result.constraints) == 1
    assert len(result.qualification) == 25
    assert result.audit["missing_hours"] == 24
    assert result.qualification.inputs_qualified.sum() == 1
    assert not result.audit["inputs_qualified"]
    assert not result.audit["imputation_performed"]


def test_external_equality_negative_ram_retained():
    data = payload()
    for i, kind in enumerate(("External constraint BE_AL_export", "Equality constraint")):
        row = deepcopy(data["records"][0])
        row.update(id=1000+i, cnec=False, cneName=kind, ram=-12.)
        data["records"].append(row)
    data["fetch"]["total_rows"] = len(data["records"])
    result = normalise_network_payload(data, "2026-06-24")
    assert len(result.constraints) == 26
    assert set(result.constraints.constraint_kind) == {"cnec", "external_constraint", "equality_constraint"}
    assert (result.constraints.ram_mw == -12.).sum() == 2
    assert result.audit["inputs_qualified"]


def test_hour_with_only_external_constraint_not_qualified():
    data = payload()
    data["records"][0].update(cnec=False, cneName="External constraint")
    result = normalise_network_payload(data, "2026-06-24")
    assert result.qualification.inputs_qualified.sum() == 23
    assert result.qualification.reason.iloc[0] == "no_cnec_for_hour"


def test_legacy_nullable_cnec_flag_requires_identified_physical_element():
    data = payload("2024-09-10")
    for row in data["records"]:
        row.update(cnec=None, cneEic="14T-LINE", elementType="Line")
    result = normalise_network_payload(data, "2024-09-10")
    assert result.audit["inputs_qualified"]
    assert result.audit["legacy_classified_cnec_rows"] == 24
    assert result.constraints.api_cnec.isna().all()
    data["records"][0]["elementType"] = None
    assert normalise_network_payload(data, "2024-09-10").qualification.inputs_qualified.sum() == 23


@pytest.mark.parametrize("field,value", [("ram", np.nan), ("ptdf_BE", np.inf), ("presolved", False)])
def test_invalid_row_makes_its_hour_unavailable(field, value):
    data = payload()
    data["records"][0][field] = value
    result = normalise_network_payload(data, "2026-06-24")
    assert result.qualification.inputs_qualified.sum() == 23


@pytest.mark.parametrize("endpoint", ["finalComputation", "preFinalComputation", "netPos", "scheduledExchanges"])
def test_later_or_post_coupling_sources_rejected(endpoint):
    data = payload()
    data["fetch"]["endpoint"] = endpoint
    with pytest.raises(NetworkContractError, match="Only initialComputation"):
        normalise_network_payload(data, "2026-06-24")


def test_parallel_run_source_rejected():
    data = payload()
    data["fetch"]["api_base_url"] = "https://parallelrun-publicationtool.jao.eu/core/api/data"
    with pytest.raises(NetworkContractError, match="source identity"):
        normalise_network_payload(data, "2026-06-24")


def test_presolved_filter_mandatory():
    data = payload()
    data["fetch"]["filters"] = {}
    with pytest.raises(NetworkContractError, match="Presolved=true"):
        normalise_network_payload(data, "2026-06-24")


def test_modified_after_cutoff_and_causal_fallback_fail_closed():
    data = payload()
    data["fetch"]["last_modified_utc"] = (expected_cutoff_utc("2026-06-24") + pd.Timedelta(minutes=1)).isoformat()
    assert not normalise_network_payload(data, "2026-06-24").qualification.inputs_qualified.any()
    result = normalise_network_payload(payload(), "2026-06-24",
                                       source_audit={"pit_status": "causal_previous_initial_fallback"})
    assert not result.qualification.inputs_qualified.any()


def test_operational_capture_strict_is_distinct_from_historical_watermark():
    strict = normalise_network_payload(payload(), "2026-06-24", require_operational_capture=True)
    assert not strict.audit["inputs_qualified"]
    captured = normalise_network_payload(payload(captured=True), "2026-06-24", require_operational_capture=True)
    assert captured.audit["operational_pit_eligible"]
    assert not captured.audit["publication_timestamp_certified"]
    assert not captured.audit["domain_reference_qualified"]


def test_naive_timestamps_count_mismatch_duplicate_and_wrong_bounds_raise():
    data = payload()
    data["records"][0]["dateTimeUtc"] = "2026-06-23T22:00:00"
    with pytest.raises(NetworkContractError, match="explicit timezone"):
        normalise_network_payload(data, "2026-06-24")
    data = payload()
    data["fetch"]["total_rows"] += 1
    with pytest.raises(NetworkContractError, match="record count"):
        normalise_network_payload(data, "2026-06-24")
    data["records"].append(deepcopy(data["records"][0]))
    with pytest.raises(NetworkContractError, match="Duplicate"):
        normalise_network_payload(data, "2026-06-24")
    with pytest.raises(NetworkContractError, match="bounds"):
        normalise_network_payload(payload(), "2026-06-25")


def test_quarter_hour_domain_not_averaged_to_fake_hourly_ram():
    data = payload()
    data["records"][0]["dateTimeUtc"] = (pd.Timestamp(data["records"][0]["dateTimeUtc"]) + pd.Timedelta(minutes=15)).isoformat()
    result = normalise_network_payload(data, "2026-06-24")
    assert not result.qualification.inputs_qualified.any()
    assert result.constraints.delivery_start_utc.dt.minute.eq(15).any()
    assert "non_hourly_domain_not_aggregated" in result.audit["blockers"]


def test_missing_archive_and_checksum_failure(tmp_path):
    result = load_network_day(tmp_path, "2025-10-26")
    assert len(result.qualification) == 25
    assert result.constraints.empty
    path = write_pair(tmp_path, "2026-06-24", payload())
    loaded = load_network_day(tmp_path, "2026-06-24")
    assert loaded.audit["inputs_qualified"]
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(NetworkContractError, match="checksum mismatch"):
        load_network_day(tmp_path, "2026-06-24")


def test_probe_is_isolated_and_never_overwrites(tmp_path):
    data = payload()
    class Client:
        calls = 0
        def fetch_initial_day(self, day):
            self.calls += 1
            return SimpleNamespace(audit_dict=lambda: data["fetch"], rows=tuple(data["records"]))
    client = Client()
    result = archive_initial_probe(client, tmp_path, "2026-06-24")
    assert result.audit["inputs_qualified"]
    with pytest.raises(NetworkContractError, match="already exists"):
        archive_initial_probe(client, tmp_path, "2026-06-24")
    assert client.calls == 1
    with pytest.raises(NetworkContractError, match="read-only"):
        archive_initial_probe(client, "data/pit/jao_core_flowbased", "2026-06-24")


def test_window_keeps_missing_calendar_days(tmp_path):
    from marginal_cost_expert.network import audit_network_window
    write_pair(tmp_path, "2025-10-26", payload("2025-10-26"))
    result = audit_network_window(tmp_path, "2025-10-25", "2025-10-27")
    assert result["calendar_days"] == 3
    assert result["expected_hours"] == 73
    assert result["complete_research_days"] == 1
    assert result["qualified_original_hours"] == 25
    assert result["operational_capture_days"] == 0


def test_audit_overlay_precedence_and_source_seals(tmp_path):
    from marginal_cost_expert.network import audit_network_window
    original, overlay = tmp_path / "original", tmp_path / "overlay"
    write_pair(original, "2026-06-24", payload())
    modified = payload()
    modified["records"][0]["ram"] = 42.
    overlay_raw = write_pair(overlay, "2026-06-24", modified)
    output = tmp_path / "window_audit.json"
    audit = audit_network_window(original, "2026-06-24", "2026-06-25",
                                 overlay_root=overlay, output_path=output)
    row = audit["daily"][0]
    assert row["raw_path"] == str(overlay_raw.resolve())
    assert row["raw_gzip_sha256"] == hashlib.sha256(overlay_raw.read_bytes()).hexdigest()
    assert row["source_audit_sha256"] == hashlib.sha256(
        (overlay_raw.parent / "2026-06-24.audit.json").read_bytes()).hexdigest()
    assert len(audit["network_code_sha256"]) == 64
    assert audit["timezone"] == "Europe/Paris"
    assert audit["cutoff_time"] == "08:00"
    assert json.loads(output.read_text()) == audit
    # A corrupt explicitly selected overlay must not fall back to the old one.
    overlay_raw.write_bytes(b"invalid")
    audit = audit_network_window(original, "2026-06-24", "2026-06-24", overlay_root=overlay)
    assert audit["complete_research_days"] == 0
    assert "checksum mismatch" in audit["daily"][0]["error"]
    assert audit["daily"][0]["raw_path"] == str(overlay_raw.resolve())
    assert audit["daily"][0]["raw_gzip_sha256"] == hashlib.sha256(b"invalid").hexdigest()
    assert audit["daily"][0]["source_audit_sha256"]
