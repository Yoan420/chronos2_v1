"""Temporal and joint-scenario contracts, independent of market data access."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from nyx_demand_response import inputs


def record(start="2026-09-14T17:00:00Z", *, duration=60, sid="one", weight=1., price=50.):
    stamp = pd.Timestamp(start)
    civil = pd.Timestamp(stamp.tz_convert("Europe/Paris").date())
    origin = (civil-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    return dict(delivery_start_utc=stamp.isoformat(), forecast_origin_utc=origin.isoformat(),
        inputs_available_at_utc=(origin-pd.Timedelta(minutes=1)).isoformat(),
        training_end_utc=(origin-pd.Timedelta(days=1)).isoformat(),
        duration_minutes=duration, scenario_id=sid, scenario_weight=weight,
        curve_evidence="forecast_from_past_curves",
        supply=[dict(zone=z, segment="base", capacity_mw=100., bid_eur_mwh=price) for z in inputs.ZONES],
        demand=[dict(zone=z, demand_mw=40.) for z in inputs.ZONES],
        flexibility=[dict(zone=z, segment="none", capacity_mw=0., reservation_eur_mwh=500.) for z in inputs.ZONES],
        network=[dict(constraint_id="hypothetical_copperplate", ram_mw=0., **{"ptdf_"+z: 0. for z in inputs.ZONES})],
        qualification=dict(physical_inputs_qualified=True, demand_basis="before_price_response",
            flexibility_basis="additional_voluntary_reduction", evidence_kind="audited_inputs",
            evidence_description="Offline test fixture, not an assertion about any real market.",
            duration_hours=duration/60, evidence_reference="test-fixture-no-real-data",
            **{flag: True for flag in inputs.FLAGS},
            network=dict(qualified=True, basis="reference_net_positions", balanced_zones=list(inputs.ZONES),
                reference_net_positions_mw={z: 0. for z in inputs.ZONES}, reference_evidence="Explicit fixture reference.")))


def bundle(records=None):
    return dict(schema_version=1, kind="ex_ante_demand_response_scenarios", sources=["test-fixture"],
                periods=[record()] if records is None else records)


@pytest.mark.parametrize("start,expected", [
    ("2026-03-28T23:00:00Z", "2026-03-28T07:00:00Z"),
    ("2026-10-24T22:00:00Z", "2026-10-24T06:00:00Z"),
])
def test_dminus1_origin_uses_its_own_civil_utc_offset_before_dst_transition(start, expected):
    value = bundle([record(start)])
    normalized = inputs.validate_bundle(value)
    assert normalized[0]["forecast_origin_utc"] == pd.Timestamp(expected)
    value["periods"][0]["forecast_origin_utc"] = (pd.Timestamp(expected)+pd.Timedelta(hours=1)).isoformat()
    with pytest.raises(inputs.InputContractError, match="08:00 Paris"):
        inputs.validate_bundle(value)


@pytest.mark.parametrize("field", ["inputs_available_at_utc", "training_end_utc"])
def test_future_inputs_or_any_last_training_label_after_origin_are_refused(field):
    value = bundle()
    value["periods"][0][field] = "2026-09-13T06:00:00.000001Z"
    with pytest.raises(inputs.InputContractError):
        inputs.validate_bundle(value)


def test_forecast_curve_needs_training_label_availability_and_naive_dates_are_refused():
    value = bundle()
    value["periods"][0]["training_end_utc"] = None
    with pytest.raises(inputs.InputContractError, match="last training-label"):
        inputs.validate_bundle(value)
    value["periods"][0]["curve_evidence"] = "published_flexibility_offers"
    assert len(inputs.validate_bundle(value)) == 1
    value["periods"][0]["inputs_available_at_utc"] = "2026-09-13T05:00:00"
    with pytest.raises(inputs.InputContractError, match="explicit timezone"):
        inputs.validate_bundle(value)


def test_current_auction_curve_is_forbidden_even_with_a_claimed_pre08_timestamp():
    value = bundle()
    value["periods"][0]["curve_evidence"] = "realised_current_auction_curve"
    with pytest.raises(inputs.InputContractError, match="Realised/current auction"):
        inputs.validate_bundle(value)


@pytest.mark.parametrize("damage", ["synthetic", "missing_network", "uncertified_asof", "partial_zones", "duration"])
def test_unqualified_or_synthetic_inputs_cannot_become_a_real_forecast(damage):
    value = bundle()
    r = value["periods"][0]
    if damage == "synthetic":
        r["qualification"]["evidence_kind"] = "synthetic_assumption"
    elif damage == "missing_network":
        r["network"] = None
    elif damage == "uncertified_asof":
        r["qualification"]["asof_certified"] = False
    elif damage == "partial_zones":
        r["demand"] = r["demand"][:-1]
    else:
        r["qualification"]["duration_hours"] = .25
    with pytest.raises(inputs.InputContractError):
        inputs.validate_bundle(value)


def quarterly_records():
    return [record(pd.Timestamp("2026-09-14T17:00Z")+pd.Timedelta(minutes=15*i), duration=15)
            for i in range(4)]


@pytest.mark.parametrize("damage", ["missing", "duplicate", "scenario_identity", "changing_weight", "mixed_resolution"])
def test_quarter_hours_must_form_complete_joint_scenarios(damage):
    records = quarterly_records()
    if damage == "missing":
        records.pop()
    elif damage == "duplicate":
        records.append(deepcopy(records[0]))
    elif damage == "scenario_identity":
        records[1]["scenario_id"] = "different"
    elif damage == "changing_weight":
        records[1]["scenario_weight"] = .5
    else:
        records[0]["duration_minutes"] = 60
        records[0]["qualification"]["duration_hours"] = 1.
    with pytest.raises(inputs.InputContractError):
        inputs.validate_bundle(bundle(records))


def fake_solver(supply, demand, flexibility, *, qualification, network):
    return dict(prices_eur_mwh={z: float(supply.loc[supply.zone.eq(z), "bid_eur_mwh"].iloc[0]) for z in inputs.ZONES},
                served_demand_mw={z: float(demand.loc[demand.zone.eq(z), "demand_mw"].iloc[0]) for z in inputs.ZONES},
                price_brackets={z: {"ambiguous": False} for z in inputs.ZONES})


def test_hourly_quantile_is_of_joint_path_average_not_average_of_quarterly_quantiles(monkeypatch):
    records = []
    for sid, prices in (("A", [0., 0., 100., 100.]), ("B", [100., 100., 0., 0.])):
        for i, price in enumerate(prices):
            records.append(record(pd.Timestamp("2026-09-14T17:00Z")+pd.Timedelta(minutes=15*i),
                duration=15, sid=sid, weight=.5, price=price))
    monkeypatch.setattr(inputs, "solve_period", fake_solver)
    forecasts, audits = inputs.scenario_forecasts(bundle(records))
    assert len(forecasts) == 4 and len(audits) == 8
    assert forecasts.expert_price.eq(50.).all()  # mean of marginal medians would incorrectly be zero.
    assert forecasts.expert_q10.eq(50.).all() and forecasts.expert_q90.eq(50.).all()


def test_scenario_quantiles_use_weights_and_do_not_silently_renormalize(monkeypatch):
    records = [record(sid=sid, weight=w, price=p) for sid, w, p in (("low", .05, 10.), ("mid", .5, 50.), ("high", .45, 100.))]
    monkeypatch.setattr(inputs, "solve_period", fake_solver)
    forecasts, _ = inputs.scenario_forecasts(bundle(records))
    np.testing.assert_array_equal(forecasts[["expert_q10", "expert_price", "expert_q90"]], np.tile([50., 50., 100.], (4, 1)))
    records[0]["scenario_weight"] = .04
    with pytest.raises(inputs.InputContractError, match="sum to one"):
        inputs.validate_bundle(bundle(records))


def test_ambiguous_dual_is_not_silently_published_as_forecast(monkeypatch):
    def ambiguous(*args, **kwargs):
        result = fake_solver(*args, **kwargs)
        result["price_brackets"]["DE"]["ambiguous"] = True
        return result
    monkeypatch.setattr(inputs, "solve_period", ambiguous)
    with pytest.raises(inputs.InputContractError, match="Non-unique marginal price"):
        inputs.scenario_forecasts(bundle())


def test_valid_hourly_bundle_runs_actual_dispatch_without_mutating_input():
    value = bundle()
    original = deepcopy(value)
    forecasts, audits = inputs.scenario_forecasts(value)
    assert value == original
    assert len(forecasts) == 4 and forecasts.expert_price.eq(50.).all()
    assert audits[0]["result"]["involuntary_shortage_mw"] == {z: 0. for z in inputs.ZONES}


@pytest.mark.parametrize("damage", ["negative_capacity", "nonfinite_capacity", "negative_demand",
                                  "duplicate_segment", "network_reference", "extra_supply_field"])
def test_validate_bundle_checks_lp_physics_without_running_an_optimizer(monkeypatch, damage):
    value = bundle()
    period = value["periods"][0]
    if damage == "negative_capacity":
        period["supply"][0]["capacity_mw"] = -1.
    elif damage == "nonfinite_capacity":
        period["supply"][0]["capacity_mw"] = np.nan
    elif damage == "negative_demand":
        period["demand"][0]["demand_mw"] = -1.
    elif damage == "duplicate_segment":
        period["supply"].append(deepcopy(period["supply"][0]))
    elif damage == "network_reference":
        del period["qualification"]["network"]["reference_net_positions_mw"]["FR"]
    else:
        for row in period["supply"]:
            row["unexpected_final_price"] = 1000.
    monkeypatch.setattr(inputs, "solve_period", lambda *a, **k: pytest.fail("Validation ran an optimizer"))
    with pytest.raises(ValueError):
        inputs.validate_bundle(value)


def test_empty_flexibility_list_is_a_valid_no_flexibility_offer_not_a_missing_schema():
    value = bundle()
    value["periods"][0]["flexibility"] = []
    assert len(inputs.validate_bundle(value)) == 1
    forecasts, solutions = inputs.scenario_forecasts(value)
    assert forecasts.expert_price.eq(50.).all()
    assert solutions[0]["result"]["voluntary_demand_reduction_mw"] == {z: 0. for z in inputs.ZONES}


def test_fully_curtailed_demand_has_no_identified_price_forecast_even_with_equal_derivatives():
    value = bundle()
    period = value["periods"][0]
    for row in period["supply"]:
        row["capacity_mw"], row["bid_eur_mwh"] = 100., 1000.
    for row in period["flexibility"]:
        row["capacity_mw"], row["reservation_eur_mwh"] = 100., 500.
    # Gross demand remains 40 MW in each zone. The whole demand is voluntarily
    # removed: an objective derivative of 500 is not an identified DA price.
    with pytest.raises(inputs.InputContractError, match="Fully curtailed"):
        inputs.scenario_forecasts(value)
