import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from marginal_cost_expert.data import MarginalCostDataError, expected_hours, load_zonal_inputs
from marginal_cost_expert.sources import _daily_values, allowed_output, capacity_catalog, civil_cutoff


def fixture_config(tmp_path, day="2026-06-24"):
    hours = expected_hours(day, day)
    cutoff = civil_cutoff(pd.Timestamp(day))
    values = {"load": 10., "nuclear": 2., "ccgt": 4., "gt": 1., "ttf": 40., "eua": 80.}
    sources = {}
    for name, value in values.items():
        frame = pd.DataFrame({"value_time_utc": hours, "snapshot_time_utc": cutoff,
                              "revision_time_utc": cutoff, "value": value})
        path = tmp_path / (name + ".parquet")
        frame.to_parquet(path, index=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        audit_path = path.with_suffix(".parquet.audit.json")
        audit_path.write_text(json.dumps({"sha256": digest, "cutoff_time": "08:00",
                                         "cutoff_timezone": "Europe/Paris"}))
        sources[name] = {"path": str(path), "unit": "GW" if name not in {"ttf", "eua"} else
                         ("EUR/MWh_th" if name == "ttf" else "EUR/tCO2"),
                         "information_type": "day_ahead_forecast", "component": name}
    fields = dict(zip(("demand_mw", "must_run_mw", "ccgt_available_mw", "ocgt_available_mw",
                       "ttf_eur_mwh_th", "eua_eur_tco2"), values))
    config = {"sources": sources, "zones": {"FR": {"inputs": {
        field: {"terms": [{"source": source}]} for field, source in fields.items()}}}}
    return config


def run(config, root, day="2026-06-24"):
    return load_zonal_inputs(config, project_root=root, start_day=day, end_day=day)


def replace_source(config, source, change):
    path = config["sources"][source]["path"]
    frame = pd.read_parquet(path)
    change(frame)
    frame.to_parquet(path, index=False)
    audit_path = path + ".audit.json"
    with open(audit_path) as stream:
        audit = json.load(stream)
    with open(path, "rb") as stream:
        audit["sha256"] = hashlib.sha256(stream.read()).hexdigest()
    with open(audit_path, "w") as stream:
        json.dump(audit, stream)


def test_explicit_units_no_labels_and_complete_grid(tmp_path):
    config = fixture_config(tmp_path)
    frame, audit = run(config, tmp_path)
    assert len(frame) == 24
    assert frame.demand_mw.eq(10000).all()
    assert frame.ttf_eur_mwh_th.eq(40).all()
    assert not audit["labels_loaded"] and not audit["electricity_price_inputs"]
    assert not audit["promotion_eligible"]
    assert set(frame).isdisjoint({"actual", "q50", "target", "price_lag_24h"})


@pytest.mark.parametrize("day,count", [("2025-03-30", 23), ("2025-10-26", 25)])
def test_dst_keeps_physical_hours(tmp_path, day, count):
    frame, _ = run(fixture_config(tmp_path, day), tmp_path, day)
    assert len(frame) == count
    assert not frame.delivery_start_utc.duplicated().any()


def test_cutoff_is_civil_after_dst():
    assert civil_cutoff(pd.Timestamp("2026-03-30")) == pd.Timestamp("2026-03-29T06:00Z")
    assert civil_cutoff(pd.Timestamp("2025-10-27")) == pd.Timestamp("2025-10-26T07:00Z")


def test_tampered_sha_rejected(tmp_path):
    config = fixture_config(tmp_path)
    path = config["sources"]["load"]["path"]
    with open(path, "ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(MarginalCostDataError, match="checksum"):
        run(config, tmp_path)


@pytest.mark.parametrize("column", ["snapshot_time_utc", "revision_time_utc"])
def test_future_vintage_rejected(tmp_path, column):
    config = fixture_config(tmp_path)
    replace_source(config, "load", lambda f: f.__setitem__(column, f[column] + pd.Timedelta(hours=1)))
    with pytest.raises(MarginalCostDataError, match="missing/nonfinite PIT"):
        run(config, tmp_path)


def test_latest_missing_state_not_replaced_by_older_finite(tmp_path):
    config = fixture_config(tmp_path)
    path = config["sources"]["load"]["path"]
    old = pd.read_parquet(path)
    old["revision_time_utc"] -= pd.Timedelta(hours=2)
    new = old.copy()
    new["revision_time_utc"] += pd.Timedelta(hours=1)
    new["value"] = np.nan
    pd.concat([old, new]).to_parquet(path, index=False)
    replace_source(config, "load", lambda f: None)
    with pytest.raises(MarginalCostDataError, match="missing/nonfinite"):
        run(config, tmp_path)


def test_naive_timestamp_rejected(tmp_path):
    config = fixture_config(tmp_path)
    replace_source(config, "load", lambda f: f.__setitem__("value_time_utc", f.value_time_utc.dt.tz_localize(None)))
    with pytest.raises(MarginalCostDataError, match="timezone-aware"):
        run(config, tmp_path)


def test_hidden_price_series_rejected(tmp_path):
    config = fixture_config(tmp_path)
    config["sources"]["load"]["series"] = "power.fr.price.day_ahead.eurmwh"
    with pytest.raises(MarginalCostDataError, match="forbidden"):
        run(config, tmp_path)


def test_unknown_output_feature_rejected(tmp_path):
    config = fixture_config(tmp_path)
    config["zones"]["FR"]["inputs"]["price_lag_24h"] = {"constant": 2, "unit": "MW", "assumption": "forbidden"}
    with pytest.raises(MarginalCostDataError, match="allowlist"):
        run(config, tmp_path)


def test_signed_residual_and_formula_evidence(tmp_path):
    config = fixture_config(tmp_path)
    replace_source(config, "load", lambda f: f.__setitem__("value", -2.))
    with pytest.raises(MarginalCostDataError, match="negative"):
        run(config, tmp_path)
    zone = config["zones"]["FR"]
    zone["demand_basis"] = "residual"
    with pytest.raises(MarginalCostDataError, match="formula evidence"):
        run(config, tmp_path)
    zone["residual_definition"] = {"evidence": "Pinned provider formula: load-wind-solar-hydro",
                                    "already_subtracted_components": ["wind", "solar", "hydro"]}
    frame, _ = run(config, tmp_path)
    assert frame.demand_mw.eq(-2000).all()
    config["sources"]["nuclear"]["component"] = "hydro"
    with pytest.raises(MarginalCostDataError, match="double subtraction"):
        run(config, tmp_path)


def test_capacity_cannot_silently_become_generation(tmp_path):
    config = fixture_config(tmp_path)
    config["sources"]["nuclear"]["information_type"] = "capacity_forecast"
    with pytest.raises(MarginalCostDataError, match="not must-run"):
        run(config, tmp_path)
    config["zones"]["FR"]["inputs"]["must_run_mw"]["terms"][0]["assumption"] = "Available nuclear used as curtailable low-cost-block proxy"
    _, audit = run(config, tmp_path)
    assert len(audit["assumptions"]) == 1


def test_constant_requires_explicit_assumption(tmp_path):
    config = fixture_config(tmp_path)
    config["zones"]["FR"]["inputs"]["must_run_mw"] = {"constant": 0, "unit": "MW"}
    with pytest.raises(MarginalCostDataError, match="constant needs"):
        run(config, tmp_path)


def test_source_daily_contract_and_safe_output(tmp_path):
    with pytest.raises(ValueError, match="isolated"):
        allowed_output(tmp_path)
    with pytest.raises(ValueError, match="local midnight"):
        _daily_values(pd.Series([1.], index=pd.to_datetime(["2026-06-24T01:00"])))
    catalog = capacity_catalog(["FR", "DE", "BE", "NL"])
    assert len(catalog) == 14
    assert "de_lignite_available_gw" in catalog
    assert all(not x["series"].endswith("cet.hourly") for x in catalog.values())


def test_optional_missing_keeps_calendar_and_records_absence(tmp_path):
    config = fixture_config(tmp_path)
    replace_source(config, "ccgt", lambda f: f.__setitem__("value", np.nan))
    with pytest.raises(MarginalCostDataError, match="missing/nonfinite"):
        run(config, tmp_path)
    config["allow_missing_hours"] = True
    frame, audit = run(config, tmp_path)
    assert len(frame) == 24 and frame.ccgt_available_mw.isna().all()
    assert audit["incomplete_physical_rows"] == 24
    assert audit["source_audits"]["ccgt"]["missing_hours"] == 24
    assert audit["incomplete_days_by_zone"] == {"FR": ["2026-06-24"]}
    assert frame.demand_mw.notna().all()


@pytest.mark.parametrize("bad", [np.inf, "invalid"])
def test_optional_missing_does_not_hide_corrupt_values(tmp_path, bad):
    config = fixture_config(tmp_path)
    config["allow_missing_hours"] = True
    replace_source(config, "ccgt", lambda f: f.__setitem__("value", bad))
    with pytest.raises(MarginalCostDataError, match="malformed or infinite"):
        run(config, tmp_path)


def test_optional_missing_does_not_hide_causality_error(tmp_path):
    config = fixture_config(tmp_path)
    config["allow_missing_hours"] = True
    replace_source(config, "ccgt", lambda f: f.__setitem__("revision_time_utc", f.revision_time_utc + pd.Timedelta(hours=1)))
    with pytest.raises(MarginalCostDataError, match="causality"):
        run(config, tmp_path)


def test_optional_missing_does_not_hide_checksum_error(tmp_path):
    config = fixture_config(tmp_path)
    config["allow_missing_hours"] = True
    with open(config["sources"]["ccgt"]["path"], "ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(MarginalCostDataError, match="checksum"):
        run(config, tmp_path)
