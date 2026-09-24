"""Independent LP identities, reference-frame and no-invented-shortage tests."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from nyx_demand_response.dispatch import (
    DispatchInfeasibleError, DispatchQualificationError, solve_period,
)


def supply(rows):
    return pd.DataFrame(rows, columns=["zone", "segment", "capacity_mw", "bid_eur_mwh"])


def demand(rows):
    return pd.DataFrame(rows, columns=["zone", "demand_mw"])


def flexibility(rows=()):
    return pd.DataFrame(rows, columns=["zone", "segment", "capacity_mw", "reservation_eur_mwh"])


def qualification(**overrides):
    return dict(physical_inputs_qualified=True, demand_basis="before_price_response",
        flexibility_basis="additional_voluntary_reduction", evidence_kind="synthetic_assumption",
        evidence_description="Explicit hypothetical supply and demand bids for unit testing only.", **overrides)


def network_example(*, reference=0.):
    net = pd.DataFrame({"constraint_id": ["FR_to_DE", "DE_to_FR"], "ram_mw": [40., 40.],
                        "ptdf_FR": [.5, -.5], "ptdf_DE": [-.5, .5]})
    q = qualification(network=dict(qualified=True, basis="reference_net_positions",
        reference_net_positions_mw={"FR": reference, "DE": -reference}, balanced_zones=["FR", "DE"],
        reference_evidence="Synthetic balanced reference and residual margins explicitly defined together."))
    s = supply([["FR", "cheap", 250., 50.], ["DE", "local", 250., 100.]])
    d = demand([["FR", 100.], ["DE", 100.]])
    return s, d, flexibility(), q, net


def test_500mw_gap_is_voluntary_reduction_at_650_not_involuntary_shortage():
    out = solve_period(supply([["DE", "thermal", 1000., 80.]]), demand([["DE", 1500.]]),
        flexibility([["DE", "voluntary", 1000., 650.]]), qualification=qualification())
    assert out["prices_eur_mwh"]["DE"] == pytest.approx(650.)
    assert out["generation_mw"]["DE"] == pytest.approx(1000.)
    assert out["voluntary_demand_reduction_mw"]["DE"] == pytest.approx(500.)
    assert out["served_demand_mw"]["DE"] == pytest.approx(1000.)
    assert out["involuntary_shortage_mw"]["DE"] == 0.
    assert out["objective_eur"] == pytest.approx(1000*80+500*650)
    bracket = out["price_brackets"]["DE"]
    assert not bracket["ambiguous"]
    for side in ("left", "right"):
        assert bracket[side]["price_eur_mwh"] == pytest.approx(650., abs=1e-4)
    assert out["synthetic_assumption"] and not out["diagnostics"]["demand_curve_identified_from_prices"]


def test_insufficient_supply_and_offered_flexibility_refuse_instead_of_4000_penalty():
    with pytest.raises(DispatchInfeasibleError, match="no artificial shortage"):
        solve_period(supply([["DE", "thermal", 1000., 80.]]), demand([["DE", 1500.]]),
                     flexibility([["DE", "voluntary", 400., 650.]]), qualification=qualification())


def test_three_demand_shocks_are_monotone_and_use_flexibility_in_merit_order():
    s = supply([["DE", "base", 100., 50.], ["DE", "peak", 100., 200.]])
    f = flexibility([["DE", "low", 10., 100.], ["DE", "high", 30., 500.]])
    outputs = [solve_period(s, demand([["DE", value]]), f, qualification=qualification())
               for value in (90., 105., 220.)]
    np.testing.assert_allclose([o["prices_eur_mwh"]["DE"] for o in outputs], [50., 100., 500.])
    np.testing.assert_allclose([o["voluntary_demand_reduction_mw"]["DE"] for o in outputs], [0., 5., 20.])
    assert outputs[2]["flexibility"][0]["reduction_mw"] == pytest.approx(10.)
    assert outputs[2]["flexibility"][1]["reduction_mw"] == pytest.approx(10.)


def test_stack_breakpoint_is_not_claimed_as_unique_price():
    out = solve_period(supply([["DE", "base", 100., 50.], ["DE", "peak", 100., 200.]]),
        demand([["DE", 100.]]), flexibility(), qualification=qualification())
    bracket = out["price_brackets"]["DE"]
    assert bracket["ambiguous"]
    assert bracket["left"]["price_eur_mwh"] == pytest.approx(50., abs=1e-4)
    assert bracket["right"]["price_eur_mwh"] == pytest.approx(200., abs=1e-4)
    assert 50.-1e-6 <= out["prices_eur_mwh"]["DE"] <= 200.+1e-6


def test_capacity_boundary_reports_missing_right_derivative_without_inventing_price():
    out = solve_period(supply([["DE", "base", 100., 50.]]), demand([["DE", 100.]]),
                       flexibility(), qualification=qualification())
    assert out["price_brackets"]["DE"]["right"] == {"price_eur_mwh": None, "status": "infeasible"}
    assert out["price_brackets"]["DE"]["ambiguous"]
    assert out["involuntary_shortage_mw"] == {"DE": 0.}


def test_negative_supply_bid_is_valid_without_negative_demand_or_extra_curtailment():
    out = solve_period(supply([["DE", "renewable", 100., -20.]]), demand([["DE", 50.]]),
        flexibility([["DE", "free", 100., 0.]]), qualification=qualification())
    assert out["prices_eur_mwh"]["DE"] == pytest.approx(-20.)
    assert out["voluntary_demand_reduction_mw"]["DE"] == 0.
    assert out["generation_mw"]["DE"] == pytest.approx(50.)


def test_curtailment_cannot_exceed_gross_demand_and_full_reduction_has_distinct_derivative():
    out = solve_period(supply([]), demand([["DE", 40.]]),
        flexibility([["DE", "first", 100., 100.], ["DE", "second", 100., 200.]]),
        qualification=qualification())
    assert out["voluntary_demand_reduction_mw"]["DE"] == pytest.approx(40.)
    assert out["served_demand_mw"]["DE"] == pytest.approx(0.)
    assert out["demand_marginal_cost_eur_mwh"]["DE"] == pytest.approx(100.)
    for side in ("left", "right"):
        assert out["price_brackets"]["DE"][side]["price_eur_mwh"] == pytest.approx(100., abs=1e-4)


def test_zero_demand_has_boundary_not_a_fabricated_left_price():
    out = solve_period(supply([["DE", "base", 100., 50.]]), demand([["DE", 0.]]),
        flexibility([["DE", "first", 100., 100.]]), qualification=qualification())
    assert out["price_brackets"]["DE"]["left"]["price_eur_mwh"] is None
    assert out["served_demand_mw"]["DE"] == 0.


def test_duration_changes_euros_but_not_marginal_prices_or_mw():
    s, d, f = supply([["DE", "base", 100., 50.]]), demand([["DE", 40.]]), flexibility()
    hourly = solve_period(s, d, f, qualification=qualification(duration_hours=1.))
    quarter = solve_period(s, d, f, qualification=qualification(duration_hours=.25))
    assert quarter["objective_eur"] == pytest.approx(hourly["objective_eur"]*.25)
    assert quarter["prices_eur_mwh"] == hourly["prices_eur_mwh"]
    assert quarter["generation_mw"] == hourly["generation_mw"]


def test_qualified_network_produces_zonal_prices_congestion_shadow_and_balanced_exports():
    s, d, f, q, net = network_example()
    out = solve_period(s, d, f, qualification=q, network=net)
    assert out["prices_eur_mwh"] == pytest.approx({"FR": 50., "DE": 100.})
    assert out["net_exports_mw"] == pytest.approx({"FR": 40., "DE": -40.})
    assert out["generation_mw"] == pytest.approx({"FR": 140., "DE": 60.})
    assert out["network"][0]["margin_mw"] == pytest.approx(0.)
    assert out["network"][0]["shadow_price_eur_mwh"] == pytest.approx(50.)
    for zone in ("FR", "DE"):
        for side in ("left", "right"):
            assert out["price_brackets"][zone][side]["price_eur_mwh"] == pytest.approx(out["prices_eur_mwh"][zone], abs=1e-4)


def test_nonzero_reference_is_applied_not_silently_treated_as_zero_balanced_ram():
    s, d, f, q, net = network_example(reference=20.)
    out = solve_period(s, d, f, qualification=q, network=net)
    assert out["net_exports_mw"] == pytest.approx({"FR": 60., "DE": -60.})
    assert out["network"][0]["flow_change_from_reference_mw"] == pytest.approx(40.)


def test_negative_residual_ram_is_allowed_with_explicit_reference_and_feasible_domain():
    s, d, f, q, net = network_example(reference=60.)
    net.loc[0, "ram_mw"] = -20.
    net.loc[1, "ram_mw"] = 100.
    out = solve_period(s, d, f, qualification=q, network=net)
    assert out["net_exports_mw"]["FR"] == pytest.approx(40.)


def test_no_network_means_isolated_zones_not_implicit_copperplate_imports():
    s, d, f, q, net = network_example()
    s.loc[s.zone.eq("DE"), "capacity_mw"] = 70.
    with pytest.raises(DispatchInfeasibleError):
        solve_period(s, d, f, qualification=q)
    assert solve_period(s, d, f, qualification=q, network=net)["status"] == "optimal"


@pytest.mark.parametrize("damage", ["missing", "pre_netted", "double_counted_flex", "uncertified"])
def test_missing_or_inconsistent_physical_qualification_is_refused(damage):
    q = qualification()
    if damage == "missing":
        q = {}
    elif damage == "pre_netted":
        q["demand_basis"] = "already_after_flexibility"
    elif damage == "double_counted_flex":
        q["flexibility_basis"] = "included_in_demand"
    else:
        q["physical_inputs_qualified"] = False
    with pytest.raises(DispatchQualificationError):
        solve_period(supply([["DE", "base", 100., 50.]]), demand([["DE", 50.]]),
                     flexibility(), qualification=q)


@pytest.mark.parametrize("damage", ["missing_reference", "unbalanced", "unknown_zone", "missing_ptdf", "extra_ptdf"])
def test_network_reference_requires_complete_qualified_balanced_domain(damage):
    s, d, f, q, net = network_example()
    if damage == "missing_reference":
        del q["network"]["reference_net_positions_mw"]
    elif damage == "unbalanced":
        q["network"]["reference_net_positions_mw"]["FR"] = 1.
    elif damage == "unknown_zone":
        q["network"]["balanced_zones"] += ["BE"]
    elif damage == "missing_ptdf":
        net = net.drop(columns="ptdf_DE")
    else:
        net["ptdf_BE"] = 0.
    with pytest.raises(DispatchQualificationError):
        solve_period(s, d, f, qualification=q, network=net)


@pytest.mark.parametrize("damage", ["duplicate_supply", "duplicate_demand", "nan", "infinite", "negative_capacity", "unknown_zone"])
def test_invalid_normalized_inputs_are_rejected(damage):
    s, d, f, q, _ = network_example()
    if damage == "duplicate_supply":
        s = pd.concat([s, s.iloc[:1]])
    elif damage == "duplicate_demand":
        d = pd.concat([d, d.iloc[:1]])
    elif damage == "nan":
        s.loc[0, "bid_eur_mwh"] = np.nan
    elif damage == "infinite":
        f = flexibility([["FR", "invalid", np.inf, 500.]])
    elif damage == "negative_capacity":
        s.loc[0, "capacity_mw"] = -1.
    else:
        f = flexibility([["BE", "unmodeled", 10., 500.]])
    with pytest.raises(DispatchQualificationError):
        solve_period(s, d, f, qualification=q)


def test_network_duplicate_and_nonfinite_coefficients_are_rejected():
    s, d, f, q, net = network_example()
    for bad in (pd.concat([net, net.iloc[:1]]), net.assign(ptdf_FR=[np.nan, 0.])):
        with pytest.raises(DispatchQualificationError):
            solve_period(s, d, f, qualification=q, network=bad)


def test_input_tables_and_nested_qualification_are_not_modified():
    s, d, f, q, net = network_example()
    before = (s.copy(deep=True), d.copy(deep=True), f.copy(deep=True), deepcopy(q), net.copy(deep=True))
    out = solve_period(s, d, f, qualification=q, network=net)
    for first, second in ((s, before[0]), (d, before[1]), (f, before[2]), (net, before[4])):
        pd.testing.assert_frame_equal(first, second)
    assert q == before[3]
    out["qualification"]["network"]["balanced_zones"].append("BAD")
    assert q == before[3]
