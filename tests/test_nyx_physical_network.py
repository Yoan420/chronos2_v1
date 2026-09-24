"""Offline physical-network contracts: direction is not deliverable capacity."""
from copy import deepcopy
import gzip
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.jao_flowbased import CORE_PTDF_ZONES, JAO_CORE_DATA_URL, expected_cutoff_utc, local_day_utc_bounds
from marginal_cost_expert.network import AHC_HUBS, NetworkContractError
from nyx_physical_p50 import network


def _payload(day="2026-09-14", captured=False):
    first, end = local_day_utc_bounds(day)
    cutoff = expected_cutoff_utc(day)
    records = []
    for i, stamp in enumerate(pd.date_range(first, end, freq="h", inclusive="left")):
        for j, (ram, fr, be, nl) in enumerate(((100., .4, .1, .2), (900., -.4, -.1, -.2), (-20., .2, .1, .1))):
            row = {"id": i * 3 + j, "dateTimeUtc": stamp.isoformat(), "cnec": True,
                   "presolved": True, "cneName": "Generic line", "ram": ram,
                   "fmax": 140., "frm": 10., "frefInit": 30., "fcore": 12.}
            row.update({"ptdf_" + z: 0. for z in CORE_PTDF_ZONES})
            row.update(ptdf_FR=fr, ptdf_DE=0., ptdf_BE=be, ptdf_NL=nl)
            row.update({"ptdf_" + hub: 0. if pd.Timestamp(day) >= pd.Timestamp("2026-06-11") else None for hub in AHC_HUBS})
            records.append(row)
    return {"fetch": {"api_base_url": JAO_CORE_DATA_URL, "endpoint": "initialComputation",
                      "filters": {"Presolved": True}, "start_utc": first.isoformat(), "end_utc": end.isoformat(),
                      "total_rows": len(records), "last_modified_utc": (cutoff - pd.Timedelta(hours=5)).isoformat(),
                      "retrieved_at_utc": (cutoff - pd.Timedelta(hours=1) if captured else end + pd.Timedelta(days=2)).isoformat()},
            "records": records}


def _write(root, day="2026-09-14", payload=None, extra_audit=None):
    directory = root / "raw" / "initialComputation"
    directory.mkdir(parents=True, exist_ok=True)
    blob = gzip.compress(json.dumps(payload or _payload(day)).encode(), mtime=0)
    raw = directory / (day + ".json.gz")
    raw.write_bytes(blob)
    audit = directory / (day + ".audit.json")
    audit.write_text(json.dumps({"raw_gzip_sha256": hashlib.sha256(blob).hexdigest(), **(extra_audit or {})}), encoding="utf-8")
    return raw, audit


def _panel(day="2026-09-14", zones=("FR", "DE", "BE", "NL")):
    first, end = local_day_utc_bounds(day)
    hours = pd.date_range(first, end, inclusive="left", freq="h")
    panel = pd.concat([pd.DataFrame({"zone": z, "timestamp_utc": hours, "forecast_origin_utc": expected_cutoff_utc(day)})
                       for z in zones], ignore_index=True)
    panel.index = pd.Index(np.arange(len(panel)) * 7 + 13, name="original_identity")
    return panel


def _features(panel, roots):
    return network.prepare_network_features(panel, raw_roots=roots)


@pytest.mark.parametrize("day,hours", [("2025-03-30", 23), ("2025-10-26", 25), ("2026-09-14", 24)])
def test_exact_dst_hourly_index_and_historical_not_operational(tmp_path, day, hours):
    _write(tmp_path, day)
    panel = _panel(day)
    out, audit = _features(panel, [tmp_path])
    assert out.index.equals(panel.index)
    assert len(out) == hours * 4
    assert out.network_eligible.all()
    assert not out.network_operational_capture_eligible.any()
    assert audit["eligible_rows"] == len(out)
    assert not audit["production_pit_evidence"]
    assert not audit["domain_reference_qualified"]
    assert not audit["feasible_imports_computed"]
    assert not audit["imputation_performed"]
    assert len(audit["source_files"]) == 2
    assert set(audit["feature_columns"]).issubset(out)


def test_signed_import_direction_and_negative_ram_remain_distinct(tmp_path):
    _write(tmp_path)
    panel = _panel()
    out, _ = _features(panel, [tmp_path])
    de = out.loc[panel.zone.eq("DE")].iloc[0]
    fr = out.loc[panel.zone.eq("FR")].iloc[0]
    assert de.feature_network_from_fr_tightening_fraction == pytest.approx(2/3)
    assert de.feature_network_from_fr_relieving_fraction == pytest.approx(1/3)
    assert de.feature_network_from_fr_tightening_ram_p10_mw == pytest.approx(-8.)
    assert de.feature_network_from_fr_tightening_negative_ram_share == .5
    assert de.feature_network_from_fr_tightening_low_ram_share == 1.
    assert fr.feature_network_from_de_tightening_fraction == pytest.approx(1/3)
    assert fr.feature_network_from_de_tightening_ram_p10_mw == 900.
    assert fr.feature_network_from_de_tightening_low_ram_share == 0.
    assert de.feature_network_raw_negative_ram_share == pytest.approx(1/3)


def test_missing_hour_stays_entirely_missing_including_flags(tmp_path):
    data = _payload()
    data["records"] = data["records"][3:]
    data["fetch"]["total_rows"] = len(data["records"])
    _write(tmp_path, payload=data)
    panel = _panel()
    out, audit = _features(panel, [tmp_path])
    missing = panel.timestamp_utc.eq(panel.timestamp_utc.min())
    assert not out.loc[missing, "network_source_hour_present"].any()
    assert not out.loc[missing, "network_eligible"].any()
    assert out.loc[missing, audit["feature_columns"]].isna().all().all()
    assert out.loc[~missing, "network_eligible"].all()


def test_absent_day_does_not_reuse_previous_day(tmp_path):
    _write(tmp_path, "2026-09-13")
    out, audit = _features(_panel(), [tmp_path])
    assert not out.network_eligible.any()
    assert out[audit["feature_columns"]].isna().all().all()
    assert not audit["source_files"]


def test_modified_after_cutoff_abstains_even_if_data_numerically_complete(tmp_path):
    data = _payload()
    data["fetch"]["last_modified_utc"] = (expected_cutoff_utc("2026-09-14") + pd.Timedelta(seconds=1)).isoformat()
    _write(tmp_path, payload=data)
    out, audit = _features(_panel(), [tmp_path])
    assert out.network_source_hour_present.all()
    assert not out.network_eligible.any()
    assert out[audit["feature_columns"]].isna().all().all()
    assert "modified_after_cutoff" in audit["days"][0]["source_audit"]["blockers"]


def test_capture_requirement_is_stricter_than_historical_watermark(tmp_path):
    _write(tmp_path)
    out, _ = network.prepare_network_features(_panel(), [tmp_path], require_operational_capture=True)
    assert not out.network_eligible.any()
    captured = tmp_path / "captured"
    _write(captured, payload=_payload(captured=True))
    out, audit = network.prepare_network_features(_panel(), [captured], require_operational_capture=True)
    assert out.network_operational_capture_eligible.all()
    assert audit["operational_capture_rows"] == len(out)
    assert not audit["publication_timestamp_certified"]


@pytest.mark.parametrize("endpoint", ["finalComputation", "preFinalComputation", "shadowPrices", "netPos", "scheduledExchanges"])
def test_later_or_postcoupling_sources_are_refused(tmp_path, endpoint):
    data = _payload()
    data["fetch"]["endpoint"] = endpoint
    _write(tmp_path, payload=data)
    with pytest.raises(NetworkContractError, match="Only initialComputation"):
        _features(_panel(), [tmp_path])


def test_checksum_corruption_is_not_missingness(tmp_path):
    raw, _ = _write(tmp_path)
    raw.write_bytes(raw.read_bytes() + b"tampered")
    with pytest.raises(NetworkContractError, match="checksum"):
        _features(_panel(), [tmp_path])


def test_partial_newer_pair_cannot_fall_back_to_older_complete_archive(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    _write(old)
    _, sidecar = _write(new)
    sidecar.unlink()
    with pytest.raises(NetworkContractError, match="Incomplete raw/audit"):
        _features(_panel(), [old, new])


def test_newer_late_source_is_not_hidden_by_older_eligible_source(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    _write(old)
    data = _payload()
    data["fetch"]["last_modified_utc"] = (expected_cutoff_utc("2026-09-14") + pd.Timedelta(minutes=1)).isoformat()
    _write(new, payload=data)
    out, _ = _features(_panel(), [old, new / "raw"])
    assert not out.network_eligible.any()


@pytest.mark.parametrize("bad", ["missing_hub", "null_hub", "quarter_hour", "non_presolved"])
def test_required_schema_defects_abstain_without_fill(tmp_path, bad):
    data = _payload()
    if bad == "missing_hub":
        for row in data["records"]:
            row.pop("ptdf_DE_DK1_VH")
    elif bad == "null_hub":
        for row in data["records"]:
            row["ptdf_DE_DK1_VH"] = None
    elif bad == "quarter_hour":
        data["records"][0]["dateTimeUtc"] = (pd.Timestamp(data["records"][0]["dateTimeUtc"]) + pd.Timedelta(minutes=15)).isoformat()
    else:
        for row in data["records"]:
            row["presolved"] = False
    _write(tmp_path, payload=data)
    out, audit = _features(_panel(), [tmp_path])
    assert not out.network_eligible.any()
    assert out[audit["feature_columns"]].isna().all().all()


def test_future_prices_labels_forecasts_and_cnec_names_do_not_drive_features(tmp_path):
    _write(tmp_path)
    panel = _panel()
    first, audit = _features(panel, [tmp_path])
    changed = panel.assign(actual="unreadable price", forecast=np.inf, storm=-1e99,
                           label_available_at_utc="not a timestamp", q10=None, q90=1e200)
    second, _ = _features(changed, [tmp_path])
    pd.testing.assert_frame_equal(first, second)
    assert set(audit["original_panel_columns_read"]) == {"zone", "timestamp_utc", "forecast_origin_utc"}
    data = _payload()
    for row in data["records"]:
        row["cneName"] = "Another physical line never selected by name"
    renamed = tmp_path / "renamed"
    _write(renamed, payload=data)
    third, _ = _features(panel, [renamed])
    pd.testing.assert_frame_equal(first, third)


@pytest.mark.parametrize("bad", ["origin", "naive", "half_hour", "duplicate", "country"])
def test_forecast_identity_must_be_exact_before_sources_read(tmp_path, bad):
    panel = _panel()
    if bad == "origin":
        panel.forecast_origin_utc += pd.Timedelta(hours=1)
    elif bad == "naive":
        panel.timestamp_utc = panel.timestamp_utc.dt.tz_localize(None)
    elif bad == "half_hour":
        panel.timestamp_utc += pd.Timedelta(minutes=30)
    elif bad == "duplicate":
        panel = pd.concat([panel, panel.iloc[:1]])
    else:
        panel["zone"] = "de"
    with pytest.raises(NetworkContractError):
        _features(panel, [tmp_path])


def test_feature_preparation_never_calls_external_client(tmp_path, monkeypatch):
    def forbidden():
        raise AssertionError("Feature preparation cannot fetch")
    monkeypatch.setattr(network, "_client_factory", forbidden)
    out, _ = _features(_panel(), [tmp_path])
    assert not out.network_eligible.any()


def test_no_exposure_is_not_zero_import_headroom(tmp_path):
    data = _payload()
    for row in data["records"]:
        for zone in ("FR", "DE", "BE", "NL"):
            row["ptdf_" + zone] = 0.
    _write(tmp_path, payload=data)
    out, _ = _features(_panel(zones=("DE",)), [tmp_path])
    assert out.network_eligible.all()
    assert out.feature_network_from_fr_tightening_fraction.eq(0).all()
    assert out.feature_network_from_fr_tightening_ram_p10_mw.isna().all()


def test_duplicate_raw_id_rejected(tmp_path):
    data = _payload()
    data["records"].append(deepcopy(data["records"][0]))
    data["fetch"]["total_rows"] = len(data["records"])
    _write(tmp_path, payload=data)
    with pytest.raises(NetworkContractError, match="Duplicate"):
        _features(_panel(), [tmp_path])


def test_old_causal_fill_archive_not_admitted(tmp_path):
    _write(tmp_path, extra_audit={"pit_status": "causal_previous_initial_fallback"})
    out, audit = _features(_panel(), [tmp_path])
    assert not out.network_eligible.any()
    assert out[audit["feature_columns"]].isna().all().all()


def _empty_payload():
    data = _payload()
    data["records"] = []
    data["fetch"].update(total_rows=0, last_modified_utc=None)
    return data


def test_valid_empty_initial_payload_needs_no_invented_modification_timestamp(tmp_path):
    _write(tmp_path, payload=_empty_payload())
    out, audit = _features(_panel(), [tmp_path])
    assert not out.network_eligible.any()
    assert not out.network_source_hour_present.any()
    assert out[audit["feature_columns"]].isna().all().all()
    assert len(audit["source_files"]) == 2
    day_audit = audit["days"][0]["source_audit"]
    assert day_audit.get("last_modified_utc") is None
    assert not day_audit["publication_timestamp_certified"]


@pytest.mark.parametrize("bad", ["checksum", "endpoint", "bounds", "filter", "count", "retrieval"])
def test_empty_payload_is_not_a_bypass_for_the_source_contract(tmp_path, bad):
    data = _empty_payload()
    if bad == "endpoint":
        data["fetch"]["endpoint"] = "finalComputation"
    elif bad == "bounds":
        data["fetch"]["start_utc"] = (pd.Timestamp(data["fetch"]["start_utc"]) + pd.Timedelta(hours=1)).isoformat()
    elif bad == "filter":
        data["fetch"]["filters"] = {}
    elif bad == "count":
        data["fetch"]["total_rows"] = 1
    elif bad == "retrieval":
        data["fetch"]["retrieved_at_utc"] = "2026-09-15 12:00:00"
    raw, _ = _write(tmp_path, payload=data)
    if bad == "checksum":
        raw.write_bytes(raw.read_bytes() + b"tampered")
    with pytest.raises((NetworkContractError, ValueError)):
        _features(_panel(), [tmp_path])
