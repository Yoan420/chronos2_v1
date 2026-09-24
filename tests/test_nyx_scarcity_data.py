from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nyx_scarcity import data


def _store(tmp_path, key="de_gas_available", *, stamps=None, snapshots=None, revisions=None,
           values=None, audit_changes=None, frame_changes=None):
    spec = data.source_registry()[key]
    path = tmp_path / spec["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    index = pd.DatetimeIndex(stamps if stamps is not None else ["2026-09-14T17:00:00Z"])
    origins = data._origin(index, "Europe/Paris")
    frame = pd.DataFrame({"value_time_utc": index, "snapshot_time_utc": snapshots if snapshots is not None else origins,
                          "revision_time_utc": revisions if revisions is not None else origins,
                          spec["column"]: values if values is not None else np.full(len(index), 10.)})
    if spec.get("source_time_column"):
        frame[spec["source_time_column"]] = origins - pd.Timedelta(hours=12)
    for k, v in (frame_changes or {}).items():
        frame[k] = v
    frame.to_parquet(path, index=False)
    identity = {spec["column"]: spec["series"]} if key in {"ttf", "eua"} or key.endswith("residual_load") else spec["series"]
    audit = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "series": identity,
             "cutoff_time": "08:00", "cutoff_timezone": "Europe/Paris", "unit": spec["unit"]}
    if "required_materialized_scale" in spec:
        audit["value_scale"] = spec["required_materialized_scale"]
    audit.update(audit_changes or {})
    Path(str(path) + ".audit.json").write_text(json.dumps(audit), encoding="utf-8")
    return key, spec, index


def test_latest_eligible_same_hour_vintage_and_range(tmp_path):
    t = pd.Timestamp("2026-09-14T17:00Z")
    key, spec, _ = _store(tmp_path, stamps=[t, t, t],
        snapshots=pd.to_datetime(["2026-09-13T03:00Z", "2026-09-13T05:00Z", "2026-09-13T07:00Z"]),
        revisions=pd.to_datetime(["2026-09-13T03:00Z", "2026-09-13T05:00Z", "2026-09-13T07:00Z"]),
        values=[10., 14., 99.])
    f, a = data._read_source(tmp_path, key, spec, pd.DatetimeIndex([t]))
    assert f.iloc[0].tolist() == [14., 4., 4.]
    assert a["late_rows_excluded"] == 1
    assert a["eligible_multivintage_hours"] == 1
    assert a["production_pit_evidence"] is False


def test_one_vintage_is_not_an_ensemble_or_revision(tmp_path):
    key, spec, index = _store(tmp_path)
    f, _ = data._read_source(tmp_path, key, spec, index)
    assert f.iloc[0, 0] == 10.
    assert f.iloc[0, 1:].isna().all()


def test_late_only_is_missing_not_usable(tmp_path):
    key, spec, index = _store(tmp_path, snapshots=pd.to_datetime(["2026-09-13T07:00Z"]))
    f, a = data._read_source(tmp_path, key, spec, index)
    assert f.isna().all().all()
    assert a["late_rows_excluded"] == 1


@pytest.mark.parametrize("change,match", [
    ({"series": "power.price.da.de.bzn.hourly.entsoe.eurmwh"}, "allowlist"),
    ({"series": "power.nrjscan.de.3mv.availability.pmax.type.ccgt.gw"}, "allowlist"),
    ({"sha256": "bad"}, "checksum"),
    ({"cutoff_time": "10:30"}, "08:00"),
    ({"causality_violations": 1}, "causality"),
    ({"unit": "MW"}, "unit"),
])
def test_audited_identity_and_contract_are_required(tmp_path, change, match):
    key, spec, index = _store(tmp_path, audit_changes=change)
    with pytest.raises(data.ScarcityDataError, match=match):
        data._read_source(tmp_path, key, spec, index)


def test_conflicting_duplicate_vintage_rejected(tmp_path):
    key, spec, index = _store(tmp_path, stamps=["2026-09-14T17:00Z"] * 2, values=[10, 11])
    with pytest.raises(data.ScarcityDataError, match="conflicting"):
        data._read_source(tmp_path, key, spec, index.unique())


def test_fuel_observation_after_cutoff_is_excluded(tmp_path):
    key, spec, index = _store(tmp_path, "ttf", frame_changes={"ttf_source_value_time_utc": pd.to_datetime(["2026-09-13T18:00Z"])})
    f, _ = data._read_source(tmp_path, key, spec, index)
    assert f.isna().all().all()


def test_nl_wind_materialized_mw_to_gw_scale_is_not_reapplied(tmp_path):
    key, spec, index = _store(tmp_path, "nl_wind_generation", values=[4.])
    f, _ = data._read_source(tmp_path, key, spec, index)
    assert f.iloc[0, 0] == 4.


def test_nl_wind_wrong_units_fail_closed(tmp_path):
    key, spec, index = _store(tmp_path, "nl_wind_generation", audit_changes={"value_scale": 1.})
    with pytest.raises(data.ScarcityDataError, match="GW"):
        data._read_source(tmp_path, key, spec, index)


def test_missing_source_stays_nan(tmp_path):
    spec = data.source_registry()["de_gas_available"]
    index = pd.date_range("2026-09-14", periods=24, freq="h", tz="UTC")
    f, a = data._read_source(tmp_path, "de_gas_available", spec, index)
    assert f.isna().all().all()
    assert a["status"] == "missing_source"


def test_unknown_source_rejected_before_report_loading(tmp_path):
    with pytest.raises(data.ScarcityDataError, match="Unknown source"):
        data.load_inputs({"data": {"source_overrides": {"storm": "data/pit/malicious.parquet"}}}, root=tmp_path)


def test_source_override_cannot_escape_pit(tmp_path):
    with pytest.raises(data.ScarcityDataError, match="data/pit"):
        data._path(tmp_path, "../other.parquet")


@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2025-10-26", 25)])
def test_cutoffs_and_ramps_respect_dst_physical_hours(tmp_path, day, hours):
    start = pd.Timestamp(day).tz_localize("Europe/Paris")
    stop = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize("Europe/Paris")
    index = pd.date_range(start, stop, freq="h", inclusive="left").tz_convert("UTC")
    assert len(index) == hours
    key, spec, _ = _store(tmp_path, stamps=index, values=np.arange(hours))
    f, _ = data._read_source(tmp_path, key, spec, index)
    assert len(f) == hours
    assert data._origin(index, "Europe/Paris").tz_convert("Europe/Paris").hour.tolist() == [8] * hours
    ramp = data._daily_ramp(f[spec["feature"]], index, 1)
    assert pd.isna(ramp.iloc[0])
    assert ramp.iloc[1:].eq(1).all()


def test_ramp_does_not_mix_different_delivery_day_vintages():
    index = pd.date_range("2026-09-13T21:00Z", periods=3, freq="h")
    ramp = data._daily_ramp(pd.Series([10., 50., 51.], index=index), index, 1)
    assert ramp.isna().tolist() == [True, True, False]
    assert ramp.iloc[-1] == 1.


def _mock_report(root, zones, models, **kwargs):
    index = pd.date_range("2026-09-13T22:00Z", periods=25, freq="h")
    blocks = []
    for zone in zones:
        f = pd.DataFrame({"timestamp_utc": index, "zone": zone, "model": "nuclear_kalman", "forecast": 100.,
                          "q10": 80., "q90": 120., "actual": 105., "benchmark_forecast": 110.,
                          "forecast_origin_utc": data._origin(index, "Europe/Paris"), "sample": "evaluation"})
        f.loc[24, ["actual", "sample"]] = [np.nan, "live"]
        blocks.append(f)
    return pd.concat(blocks), {"delivery_day": "2026-09-15", "evaluation_start_day": "2025-09-15", "evaluation_end_day": "2026-09-14"}


def _mock_source(root, key, spec, expected):
    value = 20. if key.endswith("residual_load") else 10.
    if key == "ttf":
        value = 40.
    elif key == "eua":
        value = 80.
    f = pd.DataFrame({spec["feature"]: value, spec["feature"] + "_revision_delta": np.nan,
                      spec["feature"] + "_vintage_range_proxy": np.nan}, index=expected)
    return f, {"key": key, "status": "mock"}


def test_panel_features_do_not_use_actual_or_storm_and_no_double_netting(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load_report_panel", _mock_report)
    monkeypatch.setattr(data, "_read_source", _mock_source)
    panel, audit = data.load_inputs({"zones": ["DE"]}, root=tmp_path)
    f = panel.iloc[0]
    assert f.feature_local_gas_minus_residual_proxy_gw == -10.
    assert f.feature_local_selected_supply_minus_residual_proxy_gw == 10.
    assert f.feature_local_residual_to_gas_ratio_proxy == 2.
    assert f.feature_clean_gas_cost_ccgt_proxy_eur_mwh == pytest.approx((40 + 80 * .202) / .58)
    assert f.feature_cgc_minus_baseline_proxy_eur_mwh == pytest.approx((40 + 80 * .202) / .58 - 100.)
    assert f.label_available_at_utc == pd.Timestamp("2026-09-13T16:00Z")
    assert panel.iloc[-1]["sample"] == "live"
    assert not panel.iloc[-1].label_eligible
    assert panel.feature_eligible.all()
    assert not set(["actual", "benchmark_forecast", "zone", "sample"]) & set(audit["feature_columns"])
    assert audit["clean_gas_cost"]["native_saturn_cgc_used"] is False
    assert audit["jao"]["status"] == "excluded_fail_closed"
    assert len(panel) == 25  # No invented previous training year.
    original = _mock_report
    def mutated(*args, **kwargs):
        p, a = original(*args, **kwargs)
        p["actual"], p["benchmark_forecast"] = 100000., -100000.
        return p, a
    monkeypatch.setattr(data, "load_report_panel", mutated)
    changed, _ = data.load_inputs({"zones": ["DE"]}, root=tmp_path)
    pd.testing.assert_frame_equal(panel[audit["feature_columns"]], changed[audit["feature_columns"]])


def test_missing_required_capacity_forces_baseline_eligibility_false(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load_report_panel", _mock_report)
    def missing(root, key, spec, expected):
        f, a = _mock_source(root, key, spec, expected)
        if key == "de_gas_available":
            f.loc[:, spec["feature"]] = np.nan
        return f, a
    monkeypatch.setattr(data, "_read_source", missing)
    panel, audit = data.load_inputs({"zones": ["DE"]}, root=tmp_path)
    assert not panel.feature_eligible.any()
    assert panel.feature_local_gas_available_gw.isna().all()
    assert panel.feature_local_gas_available_gw__missing.eq(1).all()
    assert audit["eligibility"]["DE"]["fallback_rows"] == len(panel)


def test_missing_optional_fuel_remains_nan_without_blocking_physical_case(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load_report_panel", _mock_report)
    def missing(root, key, spec, expected):
        f, a = _mock_source(root, key, spec, expected)
        if key == "ttf":
            f.loc[:, spec["feature"]] = np.nan
        return f, a
    monkeypatch.setattr(data, "_read_source", missing)
    panel, _ = data.load_inputs({"zones": ["DE"]}, root=tmp_path)
    assert panel.feature_eligible.all()
    assert panel.feature_clean_gas_cost_ccgt_proxy_eur_mwh.isna().all()


def test_zero_gas_is_valid_scarcity_not_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "load_report_panel", _mock_report)
    def zero(root, key, spec, expected):
        f, a = _mock_source(root, key, spec, expected)
        if key == "de_gas_available":
            f.loc[:, spec["feature"]] = 0.
        return f, a
    monkeypatch.setattr(data, "_read_source", zero)
    panel, audit = data.load_inputs({"zones": ["DE"]}, root=tmp_path)
    assert panel.feature_eligible.all()
    assert panel.feature_local_gas_available_gw.eq(0).all()
    assert panel.feature_local_gas_zero_available.eq(1).all()
    assert panel.feature_local_residual_to_gas_ratio_proxy.eq(200).all()
    assert "feature_available_at_utc" not in audit["feature_columns"]
    assert set(audit["required_feature_columns"]).issubset(audit["feature_columns"])


@pytest.mark.parametrize("setting", [{"ccgt_efficiency": 0}, {"ocgt_efficiency": 2}, {"ccgt_efficiency": True}, {"features": ["actual"]}])
def test_invalid_data_options_fail_before_loading(tmp_path, setting):
    with pytest.raises(data.ScarcityDataError):
        data.load_inputs({"data": setting}, root=tmp_path)
