"""Offline contracts for ex-post labels attached to pre-08 physical features."""
from copy import deepcopy
import gzip
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.jao_flowbased import local_day_utc_bounds
from chronos2_hourly.jao_flowbased import CORE_PTDF_ZONES, JAO_CORE_DATA_URL, expected_cutoff_utc
from marginal_cost_expert.network import AHC_HUBS
from nyx_congestion import data


def bundles(day="2026-09-14"):
    start, end = local_day_utc_bounds(day)
    final, shadow = [], []
    for i, stamp in enumerate(pd.date_range(start, end, freq="h", inclusive="left")):
        for j in (0, 1):
            final.append({"id": i * 2 + j, "dateTimeUtc": stamp.isoformat(), "cneEic": "LINE"+str(j),
                "cneName": "Display line " + str(j), "direction": "DIRECT", "contName": "Provider final name",
                "contingencies": [{"branchEic": "OUTAGE"+str(j)}], "presolved": True,
                "ptdf_FR": .2, "ptdf_DE": -.1, "ptdf_BE": 0., "ptdf_NL": .1, "ram": 100.})
        shadow.append({"id": i, "dateTimeUtc": stamp.isoformat(), "cnecEic": "LINE0", "cnecName": "Other display",
                       "direction": "DIRECT", "contName": "Different label text", "branchEic": "OUTAGE0",
                       "shadowPrice": 40., "hub_FR": .2, "hub_DE": -.1, "hub_BE": 0., "hub_NL": .1})
    def wrap(endpoint, rows):
        return {"contract": data._contract(day, endpoint), "records": rows,
                "fetch": {"total_rows": len(rows), "complete_response": True, "tls_verified": True,
                          "last_modified_utc": (start-pd.Timedelta(hours=8)).isoformat(),
                          "retrieved_at_utc": (end+pd.Timedelta(days=1)).isoformat()}}
    return wrap("finalComputation", final), wrap("shadowPrices", shadow)


def test_exact_eic_contingency_not_fuzzy_text_and_quarter_hour_average():
    final, shadow = bundles()
    constraints, zones, audit = data.label_day("2026-09-14", final, shadow)
    assert len(constraints) == 48
    assert constraints.label_eligible.all()
    active = constraints.loc[constraints.cne_eic.eq("LINE0")]
    inactive = constraints.loc[constraints.cne_eic.eq("LINE1")]
    assert active.label_active.eq(True).all()
    assert active.label_shadow_price.eq(10.).all()  # 40 in one quarter, three proved sparse zeros
    assert inactive.label_active.eq(False).all()
    assert inactive.label_shadow_price.eq(0.).all()
    np.testing.assert_allclose(zones.loc[zones.zone.eq("DE"), "label_directional_contribution_eur_mwh"], 3., atol=1e-12)
    np.testing.assert_allclose(zones.loc[zones.zone.eq("DE"), "label_absolute_contribution_eur_mwh"], 3., atol=1e-12)
    np.testing.assert_allclose(active.label_contribution_DE_fr_eur_mwh, 3., atol=1e-12)
    assert inactive.label_absolute_contribution_DE_fr_eur_mwh.eq(0.).all()
    assert audit["mtu_minutes"] == 15
    assert (active.label_available_at_utc < active.label_capture_available_at_utc).all()
    assert not audit["production_pit_evidence"]


@pytest.mark.parametrize("day,n,price", [("2025-09-15", 24, 40.), ("2025-10-26", 25, 10.), ("2026-03-29", 23, 10.)])
def test_hourly_then_quarterly_dst_contract(day, n, price):
    final, shadow = bundles(day)
    result, _, audit = data.label_day(day, final, shadow)
    assert result.timestamp_utc.nunique() == n
    assert result.loc[result.cne_eic.eq("LINE0"), "label_shadow_price"].eq(price).all()
    assert result.label_eligible.all()


def test_missing_final_hour_cannot_become_zero_or_a_forward_fill():
    final, shadow = bundles()
    first = final["records"][0]["dateTimeUtc"]
    final["records"] = [r for r in final["records"] if r["dateTimeUtc"] != first]
    final["fetch"]["total_rows"] = len(final["records"])
    result, zones, audit = data.label_day("2026-09-14", final, shadow)
    assert not result.timestamp_utc.eq(pd.Timestamp(first)).any()
    assert not zones.loc[zones.timestamp_utc.eq(pd.Timestamp(first)), "label_eligible"].any()
    assert zones.loc[zones.timestamp_utc.eq(pd.Timestamp(first)), "label_directional_contribution_eur_mwh"].isna().all()
    assert audit["hours"][0]["complete_label_mtu_count"] == 0


def test_unmatched_physical_shadow_invalidates_zeros_not_fuzzy_matched():
    final, shadow = bundles()
    shadow["records"][0]["branchEic"] = "UNRELATED_OUTAGE"
    result, _, audit = data.label_day("2026-09-14", final, shadow)
    first = result.timestamp_utc.min()
    assert not result.loc[result.timestamp_utc.eq(first), "label_eligible"].any()
    assert result.loc[result.timestamp_utc.eq(first), "label_shadow_price"].isna().all()
    assert audit["hours"][0]["unmatched_physical_shadow_count"] == 1


def test_same_cne_reciprocal_outage_is_not_same_constraint():
    one = {"cneEic": "LINE_A", "direction": "DIRECT", "contName": "B", "contingencies": [{"branchEic": "LINE_B"}]}
    two = {"cneEic": "LINE_B", "direction": "DIRECT", "contName": "A", "contingencies": [{"branchEic": "LINE_A"}]}
    assert data.constraint_identity(one) != data.constraint_identity(two)


def test_contingency_duplicate_eics_are_identity_set_not_duplicate_events():
    row = {"cneEic": "LINE", "direction": "DIRECT", "contName": "Provider composite",
           "contingencies": [{"branchEic": "C2"}, {"branchEic": "C1"}, {"branchEic": "C2"}]}
    assert data.constraint_identity(row) == ("LINE", "DIRECT", ("C1", "C2"))
    row["contingencies"].append({"branchName": "Unknown EIC"})
    assert data.constraint_identity(row) is None


def test_unknown_watermark_is_unknown_labels_not_zero():
    final, shadow = bundles()
    shadow["fetch"]["last_modified_utc"] = None
    result, zones, _ = data.label_day("2026-09-14", final, shadow)
    assert not result.label_eligible.any()
    assert result.label_shadow_price.isna().all()
    assert not zones.label_eligible.any()


def test_globally_empty_shadow_response_is_not_proof_of_zero_market_congestion():
    final, shadow = bundles()
    shadow["records"] = []
    shadow["fetch"]["total_rows"] = 0
    result, zones, _ = data.label_day("2026-09-14", final, shadow)
    assert not result.label_eligible.any()
    assert result.label_shadow_price.isna().all()
    assert not zones.label_eligible.any()


@pytest.mark.parametrize("field,value", [("complete_response", False), ("tls_verified", False), ("total_rows", 1), ("total_rows", True)])
def test_response_integrity_is_required_before_label_construction(field, value):
    final, shadow = bundles()
    shadow["fetch"][field] = value
    with pytest.raises(data.CongestionDataError):
        data.label_day("2026-09-14", final, shadow)


def test_endpoint_bounds_and_forecast_role_cannot_be_changed():
    final, shadow = bundles()
    shadow["contract"]["endpoint"] = "initialComputation"
    with pytest.raises(data.CongestionDataError, match="identity"):
        data.label_day("2026-09-14", final, shadow)


def test_late_label_revision_delays_historical_training_availability():
    final, shadow = bundles()
    shadow["fetch"]["last_modified_utc"] = "2026-09-15T12:00:00Z"
    result, _, _ = data.label_day("2026-09-14", final, shadow)
    assert result.label_available_at_utc.eq(pd.Timestamp("2026-09-15T12:00:00Z")).all()
    assert (result.label_available_at_utc > pd.Timestamp("2026-09-15T06:00:00Z")).all()


def test_shadow_ptdf_mismatch_cannot_create_mislabeled_zero():
    final, shadow = bundles()
    shadow["records"][0]["hub_DE"] += .01
    result, _, _ = data.label_day("2026-09-14", final, shadow)
    first = result.timestamp_utc.min()
    assert not result.loc[result.timestamp_utc.eq(first), "label_eligible"].any()


def test_matched_contribution_uses_shadow_coefficients_not_rounded_final_ptdf():
    final, shadow = bundles()
    for row in final["records"]:
        row["ptdf_FR"] += data.PTDF_TOLERANCE / 2
    constraints, zones, _ = data.label_day("2026-09-14", final, shadow)
    assert constraints.label_eligible.all()
    active = constraints.loc[constraints.cne_eic.eq("LINE0")]
    np.testing.assert_allclose(active.label_contribution_DE_fr_eur_mwh, 3., rtol=0, atol=1e-12)
    np.testing.assert_allclose(zones.loc[zones.zone.eq("DE"), "label_absolute_contribution_eur_mwh"], 3., rtol=0, atol=1e-12)


def test_only_known_inactive_pre_ahc_null_hubs_may_be_ignored():
    final, shadow = bundles("2025-12-01")
    for r in final["records"]:
        r.update({"ptdf_" + h: None for h in AHC_HUBS})
    for r in shadow["records"]:
        r.update({"hub_" + h: None for h in AHC_HUBS})
    result, _, _ = data.label_day("2025-12-01", final, shadow)
    assert result.label_eligible.all()
    for r in shadow["records"]:
        r["hub_UNKNOWN_HUB"] = None
    result, _, _ = data.label_day("2025-12-01", final, shadow)
    assert not result.label_eligible.any()


def test_post_ahc_null_hub_is_unknown():
    final, shadow = bundles()
    for r in final["records"]:
        r["ptdf_DE_DK1_VH"] = None
    for r in shadow["records"]:
        r["hub_DE_DK1_VH"] = None
    result, _, _ = data.label_day("2026-09-14", final, shadow)
    assert not result.label_eligible.any()


def test_positive_below_active_epsilon_has_exactly_zero_inactive_intensity():
    final, shadow = bundles()
    for r in shadow["records"]:
        r["shadowPrice"] = data.ACTIVE_EPSILON / 2
    result, _, _ = data.label_day("2026-09-14", final, shadow)
    assert result.label_active.eq(False).all()
    assert result.label_shadow_price.eq(0.).all()


def test_duplicate_shadow_identity_invalidates_hour_even_distinct_source_ids():
    final, shadow = bundles()
    duplicate = deepcopy(shadow["records"][0]); duplicate["id"] = 500000
    shadow["records"].append(duplicate)
    shadow["fetch"]["total_rows"] += 1
    result, _, _ = data.label_day("2026-09-14", final, shadow)
    assert not result.loc[result.timestamp_utc.eq(result.timestamp_utc.min()), "label_eligible"].any()


def test_resume_reuses_immutable_bundles_and_detects_corruption(tmp_path, monkeypatch):
    namespace = tmp_path / "nyx_congestion_v1"
    monkeypatch.setattr(data, "NAMESPACE", namespace)
    target = namespace / "raw" / "labels"
    calls, closed = [], []
    class Fake:
        def fetch(self, day, endpoint):
            calls.append((day, endpoint))
            return bundles(day)[0 if endpoint == "finalComputation" else 1]
        def close(self): closed.append(True)
    result = data.collect_labels("2026-09-14", "2026-09-14", output_root=target, workers=1, client_factory=Fake)
    assert result["status"] == "complete" and len(calls) == 2 and closed
    result = data.collect_labels("2026-09-14", "2026-09-14", output_root=target, workers=1, client_factory=Fake)
    assert result["days"][0]["reused_endpoints"] == 2 and len(calls) == 2
    raw = target / "2026-09-14" / "shadowPrices" / "payload.json.gz"
    raw.write_bytes(raw.read_bytes() + b"modified")
    result = data.collect_labels("2026-09-14", "2026-09-14", output_root=target, workers=1, client_factory=Fake)
    assert result["status"] == "incomplete" and "SHA" in result["failures"][0]["error"]
    assert len(calls) == 2


@pytest.mark.parametrize("first,last,workers", [("2025-01-01", "2026-09-14", 2), ("2026-09-14", "2026-09-13", 2), ("2026-09-14", "2026-09-14", 3)])
def test_collection_is_bounded_before_any_download(first, last, workers):
    with pytest.raises(data.CongestionDataError):
        data.collect_labels(first, last, workers=workers)


def test_output_outside_new_namespace_refused(tmp_path):
    with pytest.raises(data.CongestionDataError, match="nyx_congestion_v1"):
        data.collect_labels("2026-09-14", "2026-09-14", output_root=tmp_path)


def test_atomic_commit_retries_transient_directory_lock(tmp_path, monkeypatch):
    from pathlib import Path
    namespace = tmp_path / "nyx_congestion_v1"
    monkeypatch.setattr(data, "NAMESPACE", namespace)
    calls = []
    original = Path.rename
    def flaky(self, target):
        calls.append(str(target))
        if len(calls) == 1:
            raise PermissionError("Transient antivirus handle")
        return original(self, target)
    monkeypatch.setattr(Path, "rename", flaky)
    monkeypatch.setattr(data.time, "sleep", lambda delay: None)
    directory = namespace / "raw" / "2026-09-14"
    data._commit(directory, "2026-09-14", "finalComputation", bundles()[0])
    assert len(calls) == 2
    assert data._load_endpoint(directory, "2026-09-14", "finalComputation") is not None


@pytest.mark.parametrize("duplicate", [False, True])
def test_initial_unmatched_final_is_unknown_and_only_initial_quantities_are_features(tmp_path, monkeypatch, duplicate):
    namespace = tmp_path / "nyx_congestion_v1"
    monkeypatch.setattr(data, "NAMESPACE", namespace)
    target = namespace / "raw" / "labels"
    day = "2026-09-14"
    final, shadow = bundles(day)
    for endpoint, payload in zip(data.ENDPOINTS, (final, shadow)):
        data._commit(target/day, day, endpoint, payload)
    initial_rows = []
    for row in final["records"]:
        row = deepcopy(row)
        row.update(cnec=True, fmax=200., frm=10., ram=-50.)
        for hub in CORE_PTDF_ZONES:
            row.setdefault("ptdf_"+hub, 0.)
        for hub in AHC_HUBS:
            row["ptdf_"+hub] = 0.
        initial_rows.append(row)
    unmatched = deepcopy(initial_rows[0])
    unmatched.update(cneEic="UNMATCHED_INITIAL", id=100000)
    initial_rows.append(unmatched)
    if duplicate:
        ambiguous = deepcopy(initial_rows[0]); ambiguous["id"] = 200000
        initial_rows.append(ambiguous)
    first, end = local_day_utc_bounds(day)
    cutoff = expected_cutoff_utc(day)
    payload = {"fetch": {"api_base_url": JAO_CORE_DATA_URL, "endpoint": "initialComputation", "filters": {"Presolved": True},
                          "start_utc": first.isoformat(), "end_utc": end.isoformat(), "total_rows": len(initial_rows),
                          "last_modified_utc": (cutoff-pd.Timedelta(hours=4)).isoformat(),
                          "retrieved_at_utc": (end+pd.Timedelta(days=1)).isoformat()}, "records": initial_rows}
    source = tmp_path / "old_initial"
    directory = source / "raw" / "initialComputation"
    directory.mkdir(parents=True)
    raw = directory / (day+".json.gz")
    blob = gzip.compress(json.dumps(payload).encode())
    raw.write_bytes(blob)
    sha = hashlib.sha256(blob).hexdigest()
    sidecar = directory / (day+".audit.json")
    sidecar.write_text(json.dumps({"raw_gzip_sha256": sha}))
    hours = pd.date_range(first, end, freq="h", inclusive="left")
    panel = pd.DataFrame({"zone": "DE", "timestamp_utc": hours, "forecast_origin_utc": cutoff,
                          "actual": "never read realised price", "forecast": float("inf")})
    network = pd.DataFrame({"network_eligible": True}, index=panel.index)
    audit = {"source_files": {str(raw): sha, str(sidecar): hashlib.sha256(sidecar.read_bytes()).hexdigest()},
             "days": [{"delivery_day": day, "selected_root": str(source)}]}
    constraints, zones, evidence = data.build_constraint_panel(panel, network, audit, label_root=target)
    assert len(constraints) == 49 + int(duplicate)
    assert constraints.label_eligible.sum() == 48 - int(duplicate)
    assert constraints.constraint_identity_ambiguous.sum() == 2 * int(duplicate)
    if duplicate:
        amb = constraints.loc[constraints.constraint_identity_ambiguous]
        assert not amb.constraint_identified.any() and not amb.label_eligible.any()
        assert amb.label_active.isna().all() and amb.constraint_key.nunique() == 2
    un = constraints.loc[constraints.cne_eic.eq("UNMATCHED_INITIAL")]
    assert un.label_shadow_price.isna().all() and un.label_active.isna().all()
    assert constraints.feature_cnec_initial_ram_mw.eq(-50.).all()  # final RAM is +100, never copied into X
    assert constraints.feature_cnec_ptdf_FR.eq(.2).all()
    assert not any("label" in n or "final" in n or "shadow" in n for n in evidence["feature_columns"])
    assert (constraints.loc[constraints.label_eligible, "label_available_at_utc"] >
            constraints.loc[constraints.label_eligible, "forecast_origin_utc"]).all()
    assert zones.initial_hour_available.all()
    de = zones.loc[zones.zone.eq("DE")]
    assert de.iloc[int(duplicate):].initial_absolute_contribution_coverage.eq(1.).all()
    assert zones.loc[zones.zone.eq("FR"), "initial_absolute_contribution_coverage"].isna().all()
    if duplicate:
        assert de.iloc[0].initial_absolute_contribution_coverage == 0.
        assert de.iloc[0].outside_initial_absolute_contribution_eur_mwh == pytest.approx(3.)
    else:
        assert zones.outside_initial_absolute_contribution_eur_mwh.eq(0.).all()
