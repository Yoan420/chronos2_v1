from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from marginal_cost_expert.data import MarginalCostDataError, expected_hours
from marginal_cost_expert.sources import civil_cutoff, sha256
from marginal_cost_expert.supply_sources import (
    api2_thermal_cost, isolated_path, load_supply_inputs, select_previous_close,
    validate_physical_identity, _validate_market_record, MARKET_SOURCES,
)


def test_coal_conversion_is_thermal_not_electric_and_not_fx_inverse():
    assert api2_thermal_cost(120, 1.2, net_calorific_mwh_t=6.978) == pytest.approx(100 / 6.978)
    with pytest.raises(ValueError):
        api2_thermal_cost(100, 0, net_calorific_mwh_t=6.978)


def test_previous_close_excludes_same_day_labels_even_when_api_returns_them():
    cutoff = pd.Timestamp("2026-06-23 06:00Z")
    values = pd.Series([1.1, 1.2, 9.9], index=pd.to_datetime(["2026-06-19", "2026-06-22", "2026-06-23"]))
    value, stamp = select_previous_close(values, cutoff)
    assert value == 1.2
    assert stamp == pd.Timestamp("2026-06-21 22:00Z")


def test_stale_close_refused_and_weekend_close_preserved():
    raw = pd.Series([100.0], index=pd.to_datetime(["2026-06-19"]))
    assert select_previous_close(raw, pd.Timestamp("2026-06-22 06:00Z"))[0] == 100
    with pytest.raises(ValueError, match="maximum"):
        select_previous_close(raw, pd.Timestamp("2026-06-29 06:00Z"))


@pytest.mark.parametrize("name", ["power.de.avail.aggr.coal.mw.h.full.utc.3mv.storm",
                                   "power.be.avail.aggr.nuclear.mw.h.obs.cet.3mv",
                                   "power.fr.price.da.storm", "chronos2_forecast"])
def test_observed_full_and_electricity_price_inputs_rejected(name):
    with pytest.raises(ValueError):
        validate_physical_identity(name)


def test_physical_storm_forecast_tag_is_distinguished_from_storm_price():
    validate_physical_identity("power.de.avail.aggr.coal.mw.h.fct.utc.3mv.storm")


def test_collection_cannot_modify_v1_or_operational_cache():
    with pytest.raises(ValueError):
        isolated_path("data/pit/marginal_cost_expert/capacities")
    with pytest.raises(ValueError):
        isolated_path("data/pit/kalman_hybrid")


def test_resumed_checkpoint_cannot_admit_same_day_close_or_nan():
    record = {"day": "2026-06-24", "cutoff_utc": "2026-06-23T06:00:00+00:00", "series": MARKET_SOURCES,
              "api2_usd_t": 114.6, "eurusd_usd_per_eur": 1.14282,
              "api2_usd_t_source_time_utc": "2026-06-21T22:00:00+00:00",
              "eurusd_usd_per_eur_source_time_utc": "2026-06-21T22:00:00+00:00"}
    _validate_market_record(record, "2026-06-24")
    record["eurusd_usd_per_eur_source_time_utc"] = "2026-06-22T22:00:00+00:00"
    with pytest.raises(ValueError, match="causality"):
        _validate_market_record(record, "2026-06-24")
    record["api2_usd_t"] = float("nan")
    with pytest.raises(ValueError, match="value"):
        _validate_market_record(record, "2026-06-24")


def _fixture(tmp_path):
    day = "2026-06-24"
    index = expected_hours(day, day)
    frame = pd.DataFrame({"value_time_utc": index, "snapshot_time_utc": civil_cutoff(pd.Timestamp(day)),
                          "revision_time_utc": civil_cutoff(pd.Timestamp(day)), "value": 1.5})
    path = tmp_path / "capacity.parquet"
    frame.to_parquet(path, index=False)
    audit = {"sha256": sha256(path), "series": "power.nrjscan.be.3mv.availability.pmax.type.ccgt.gw"}
    path.with_suffix(".parquet.audit.json").write_text(json.dumps(audit), encoding="utf-8")
    config = {"allow_missing_hours": True, "sources": {"physical_capacity": {
        "path": str(path), "unit": "GW", "information_type": "capacity_forecast", "series": audit["series"]}},
        "zones": {"BE": {"demand_basis": "residual", "residual_netting": ["wind", "solar"],
            "qualification": {"national_fleet_attested": False, "historical_pit_attested": False},
            "fields": {"ccgt_available_mw": {"source": "physical_capacity", "scale": 1000, "unit": "MW"}}}}}
    return config, day


def test_complete_is_not_national_qualification(tmp_path):
    config, day = _fixture(tmp_path)
    frame, audit = load_supply_inputs(config, project_root=tmp_path, start_day=day, end_day=day)
    assert frame.ccgt_available_mw.eq(1500).all()
    assert frame.inputs_complete.all()
    assert not frame.inputs_qualified.any()
    assert audit["zones"]["BE"]["qualified_hours"] == 0


def test_missing_capacity_stays_nan_not_zero(tmp_path):
    config, day = _fixture(tmp_path)
    frame, _ = load_supply_inputs(config, project_root=tmp_path, start_day=day, end_day="2026-06-25")
    assert frame.ccgt_available_mw.isna().sum() == 24
    assert not frame.iloc[-24:].inputs_complete.any()


def test_unit_mismatch_and_hidden_constant_refused(tmp_path):
    config, day = _fixture(tmp_path)
    config["zones"]["BE"]["fields"]["ccgt_available_mw"]["scale"] = 1
    with pytest.raises(MarginalCostDataError, match="unit conversion"):
        load_supply_inputs(config, project_root=tmp_path, start_day=day, end_day=day)
    config["zones"]["BE"]["fields"] = {"coal_eur_mwh_th": {"constant": 2.3, "unit": "EUR/MWh_th"}}
    with pytest.raises(MarginalCostDataError, match="constant"):
        load_supply_inputs(config, project_root=tmp_path, start_day=day, end_day=day)


def test_fuel_column_cannot_disguise_physical_capacity_units(tmp_path):
    config, day = _fixture(tmp_path)
    config["zones"]["BE"]["fields"] = {
        "ttf_eur_mwh_th": {"source": "physical_capacity", "unit": "MW", "scale": 1000}}
    with pytest.raises(MarginalCostDataError, match="output unit"):
        load_supply_inputs(config, project_root=tmp_path, start_day=day, end_day=day)


def test_source_derivation_must_match_declared_contract(tmp_path):
    config, day = _fixture(tmp_path)
    config["sources"]["physical_capacity"]["expected_audit"] = {"net_calorific_mwh_t": 6.978}
    with pytest.raises(MarginalCostDataError, match="derivation contract"):
        load_supply_inputs(config, project_root=tmp_path, start_day=day, end_day=day)
