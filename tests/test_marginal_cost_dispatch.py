"""Synthetic V2 physics and fail-closed contracts; no production/network runs."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from marginal_cost_expert.dispatch import DispatchError, build_offers, dispatch_periods


T = pd.Timestamp("2026-06-24T00:00:00Z")
K = ["delivery_start_utc", "zone"]


def physical_config(zones=("FR",), technologies=("ccgt",)):
    efficiencies = {"ccgt": .5, "ocgt": .3, "coal": .4, "lignite": .35}
    emissions = {"ccgt": .2, "ocgt": .2, "coal": .34, "lignite": .4}
    units = {"demand_mw": "MW", "eua": "EUR/tCO2"}
    segments = []
    for tech in technologies:
        units[f"{tech}_capacity"] = "MW"
        if tech in efficiencies:
            units[f"{tech}_fuel"] = "EUR/MWh_th"
            bid = {"kind": "fuel", "fuel_column": f"{tech}_fuel", "eua_column": "eua",
                   "efficiency": efficiencies[tech], "emissions_tco2_mwh_th": emissions[tech],
                   "vom_eur_mwh": 3}
        else:
            bid = {"kind": "assumed", "value_eur_mwh": 20,
                   "hypothesis": "Explicit synthetic opportunity-cost assumption"}
        segments.append({"id": tech, "technology": tech, "capacity_column": f"{tech}_capacity", "bid": bid})
    return {"demand_basis": "gross", "units": units,
            "required_technologies": {z: list(technologies) for z in zones},
            "stack_scope_qualified_by_zone": {z: True for z in zones}, "segments": segments}


def features(zones=("FR",), technologies=("ccgt",)):
    frame = pd.DataFrame({"delivery_start_utc": [T] * len(zones), "zone": zones,
                          "demand_mw": 150., "inputs_qualified": True, "eua": 80.})
    for tech in technologies:
        frame[f"{tech}_capacity"] = 200.
        frame[f"{tech}_fuel"] = 30.
    return frame


def demand(loads=None, timestamp=T):
    return pd.DataFrame([{"delivery_start_utc": timestamp, "zone": zone, "demand_mw": load}
                         for zone, load in (loads or {"FR": 150.}).items()])


def offers(specs=None, timestamp=T):
    specs = specs or [("FR", "ccgt", 200., 40.)]
    return pd.DataFrame([{"delivery_start_utc": timestamp, "zone": z, "segment": s,
                          "capacity_mw": cap, "bid_eur_mwh": bid, "technology": s}
                         for z, s, cap, bid in specs])


def qualified(d):
    return d[K].assign(inputs_qualified=True, stack_complete=True,
                        physical_scope="synthetic_complete_stack")


def run(d, o, **kwargs):
    kwargs.setdefault("qualification", qualified(d))
    return dispatch_periods(d, o, scarcity_price_eur_mwh=4000., **kwargs)


def network(ram=100., timestamp=T, extra=False):
    rows = pd.DataFrame({"delivery_start_utc": [timestamp] * 2, "cnec_id": ["east", "west"],
                         "ram_mw": [ram, ram], "ptdf_FR": [1., -1.], "ptdf_DE": [0., 0.]})
    if extra:
        rows["ptdf_BE"] = [0.5, -0.5]
    return rows


def network_config(**overrides):
    return {"zones": ["FR", "DE"], "domain_reference_qualified": True,
            "inputs_qualified": True, **overrides}


def coupled(ram=100., load_de=150., **kwargs):
    d = demand({"FR": 50., "DE": load_de})
    o = offers([("FR", "ccgt", 500., 40.), ("DE", "coal", 500., 100.)])
    kwargs.setdefault("network", network(ram))
    kwargs.setdefault("network_config", network_config())
    return run(d, o, **kwargs)


def test_distinct_thermal_efficiency_and_carbon_costs():
    technologies = ("ccgt", "ocgt", "coal", "lignite")
    book = build_offers(features(technologies=technologies), physical_config(technologies=technologies))
    cost = book.offers.set_index("technology").bid_eur_mwh
    assert cost.ccgt == pytest.approx((30 + 80 * .2) / .5 + 3)
    assert cost.ocgt == pytest.approx((30 + 80 * .2) / .3 + 3)
    assert cost.coal == pytest.approx((30 + 80 * .34) / .4 + 3)
    assert cost.lignite == pytest.approx((30 + 80 * .4) / .35 + 3)
    assert book.qualification.stack_complete.all()


@pytest.mark.parametrize("tech", ["nuclear", "hydro_reservoir", "biomass", "chp"])
def test_opportunity_cost_requires_explicit_hypothesis(tech):
    cfg = physical_config(technologies=(tech,))
    cfg["segments"][0]["bid"].pop("hypothesis")
    with pytest.raises(DispatchError, match="hypothesis"):
        build_offers(features(technologies=(tech,)), cfg)


def test_missing_coal_is_not_filled_with_zero_and_country_requirements_differ():
    cfg = physical_config(("FR", "DE"), ("ccgt", "lignite"))
    cfg["required_technologies"]["FR"] = ["ccgt"]
    cfg["segments"][1]["zones"] = ["DE"]
    f = features(("FR", "DE"), ("ccgt", "lignite"))
    f["lignite_capacity"] = np.nan
    book = build_offers(f, cfg)
    q = book.qualification.set_index("zone")
    assert q.loc["FR", "stack_complete"]
    assert not q.loc["DE", "stack_complete"]
    assert q.loc["DE", "missing_technologies"] == "lignite"
    assert "lignite" not in book.offers.technology.tolist()
    assert book.audit["missing_capacity_imputed"] is False


def test_known_zero_capacity_does_not_need_unused_fuel():
    f = features(technologies=("ccgt", "coal"))
    f["coal_capacity"], f["coal_fuel"] = 0., np.nan
    book = build_offers(f, physical_config(technologies=("ccgt", "coal")))
    assert book.qualification.stack_complete.all()
    assert set(book.offers.technology) == {"ccgt"}


def test_positive_capacity_without_required_fuel_has_no_diagnostic_price():
    f = features()
    f["ccgt_fuel"] = np.nan
    book = build_offers(f, physical_config())
    assert book.offers.empty
    assert not book.qualification.stack_complete.any()
    assert not book.qualification.consumed_inputs_complete.any()
    p = run(book.demand, book.offers, qualification=book.qualification).prices.iloc[0]
    assert np.isnan(p.raw_price_eur_mwh)
    assert p.status == "missing_period_inputs"
    assert not p.eligible


@pytest.mark.parametrize("mutation", ["negative_capacity", "naive_time", "duplicate_identity", "string_boolean"])
def test_invalid_physical_contracts_fail_explicitly(mutation):
    f = features()
    if mutation == "negative_capacity":
        f["ccgt_capacity"] = -1
    elif mutation == "naive_time":
        f["delivery_start_utc"] = T.tz_localize(None)
    elif mutation == "duplicate_identity":
        f = pd.concat([f, f], ignore_index=True)
    else:
        f["inputs_qualified"] = "false"
    with pytest.raises(DispatchError):
        build_offers(f, physical_config())


@pytest.mark.parametrize("column,unit", [("ccgt_fuel", "EUR/MWh"), ("eua", "EUR/kgCO2"),
                                         ("ccgt_capacity", "GW"), ("ccgt_fuel", "USD/t")])
def test_declared_units_are_strict(column, unit):
    cfg = physical_config()
    cfg["units"][column] = unit
    with pytest.raises(DispatchError, match="explicit unit"):
        build_offers(features(), cfg)


def test_shared_gas_chp_capacity_fractions_cannot_double_count():
    cfg = physical_config()
    second = deepcopy(cfg["segments"][0])
    second.update(id="chp", technology="chp")
    cfg["segments"].append(second)
    with pytest.raises(DispatchError, match="fractions exceed"):
        build_offers(features(), cfg)
    cfg["segments"][0]["capacity_fraction"] = .7
    cfg["segments"][1]["capacity_fraction"] = .3
    book = build_offers(features(), cfg)
    assert book.offers.capacity_mw.sum() == pytest.approx(200)


def test_residual_negative_load_accounting_and_no_double_hydro():
    cfg = physical_config(technologies=("ccgt", "nuclear"))
    cfg.update(demand_basis="residual", residual_netting={"FR": ["wind", "solar", "hydro_ror"]},
               residual_surplus_bid_eur_mwh=-35.)
    f = features(technologies=("ccgt", "nuclear"))
    f["demand_mw"] = -50.
    book = build_offers(f, cfg)
    assert book.demand.demand_mw.item() == 0
    assert book.demand.residual_surplus_mw.item() == 50
    assert book.demand.input_demand_mw.item() == -50
    result = run(book.demand, book.offers, qualification=book.qualification)
    assert result.prices.price_eur_mwh.item() == -35
    assert result.prices.curtailment_mw.item() == 250
    assert result.prices.balance_error_mw.item() == 0
    assert not book.audit["full_renewable_curtailment_modeled"]
    hydro = {"id": "hydro", "technology": "hydro_ror", "capacity_column": "hydro_capacity",
             "bid": {"kind": "assumed", "value_eur_mwh": 10., "hypothesis": "test"}}
    cfg["units"]["hydro_capacity"] = "MW"
    cfg["segments"].append(hydro)
    with pytest.raises(DispatchError, match="double counting"):
        build_offers(f, cfg)


def test_unqualified_sources_stay_unqualified_and_input_objects_unchanged():
    f, cfg = features(), physical_config()
    f = f.drop(columns="inputs_qualified")
    before_f, before_cfg = f.copy(deep=True), deepcopy(cfg)
    book = build_offers(f, cfg)
    assert not book.qualification.inputs_qualified.any()
    assert not run(book.demand, book.offers, qualification=book.qualification).prices.eligible.any()
    pd.testing.assert_frame_equal(f, before_f)
    assert cfg == before_cfg


def test_numeric_technology_coverage_is_not_national_fleet_attestation():
    cfg = physical_config()
    cfg.pop("stack_scope_qualified_by_zone")
    book = build_offers(features(), cfg)
    assert book.qualification.missing_technologies.item() == ""
    assert book.qualification.inputs_qualified.item()
    assert not book.qualification.stack_scope_qualified.item()
    assert not book.qualification.stack_complete.item()
    p = run(book.demand, book.offers, qualification=book.qualification).prices.iloc[0]
    assert p.raw_price_eur_mwh == 95
    assert p.status == "stack_incomplete"
    assert np.isnan(p.price_eur_mwh)


@pytest.mark.parametrize("bad", ["storm_price", "actual_price", "price_lag_24", "chronos_q50"])
def test_electric_price_inputs_forbidden(bad):
    f = features()
    f[bad] = 100.
    with pytest.raises(DispatchError, match="Electricity"):
        build_offers(f, physical_config())


def test_zonal_merit_order_and_extra_100mw_conserve_energy():
    o = offers([("FR", "lignite", 100., 20.), ("FR", "coal", 100., 25.), ("FR", "ccgt", 200., 40.)])
    first = run(demand({"FR": 150}), o)
    second = run(demand({"FR": 250}), o)
    assert first.prices.price_eur_mwh.item() == 25
    assert second.prices.price_eur_mwh.item() == 40
    assert second.dispatch.generation_mw.sum() - first.dispatch.generation_mw.sum() == 100
    assert second.prices.balance_error_mw.item() == 0
    assert second.prices.observed_marginal_unit_identified.item() is False
    assert not {"q10", "q50", "q90"}.intersection(second.prices.columns)


def test_omitted_stack_shortage_is_diagnostic_not_active_scarcity():
    d = demand({"FR": 300})
    q = qualified(d).assign(stack_complete=False)
    p = run(d, offers(), qualification=q).prices.iloc[0]
    assert p.raw_price_eur_mwh == 4000
    assert p.shortage_mw == 100
    assert np.isnan(p.price_eur_mwh)
    assert not p.eligible
    assert p.status == "shortage_source_incomplete"
    assert p.regime == "unavailable"


def test_known_complete_zero_stack_has_only_explicit_scarcity_proxy():
    o = offers().iloc[:0]
    p = run(demand({"FR": 300}), o).prices.iloc[0]
    assert p.price_eur_mwh == 4000
    assert p.regime == "scarcity_proxy"
    assert p.generation_mw == 0
    assert p.shortage_mw == 300


def test_stack_incomplete_without_shortage_also_abstains():
    d = demand()
    p = run(d, offers(), qualification=qualified(d).assign(stack_complete=False)).prices.iloc[0]
    assert p.raw_price_eur_mwh == 40
    assert np.isnan(p.price_eur_mwh)
    assert p.status == "stack_incomplete"


def test_missing_one_hour_is_not_forward_filled_or_series_failure():
    d = pd.concat([demand(), demand({"FR": np.nan}, T + pd.Timedelta(hours=1)),
                   demand(timestamp=T + pd.Timedelta(hours=2))], ignore_index=True)
    o = pd.concat([offers(timestamp=t) for t in d.delivery_start_utc], ignore_index=True)
    p = run(d, o).prices
    assert p.eligible.tolist() == [True, False, True]
    assert p.status.iloc[1] == "missing_period_inputs"
    assert np.isnan(p.raw_price_eur_mwh.iloc[1])


def test_missing_source_flag_blocks_only_affected_zonal_country():
    f = features(("FR", "DE"))
    f.loc[f.zone.eq("DE"), "ccgt_capacity"] = np.nan
    book = build_offers(f, physical_config(("FR", "DE")))
    result = run(book.demand, book.offers, qualification=book.qualification)
    p = result.prices.set_index("zone")
    assert p.loc["FR", "eligible"]
    assert p.loc["FR", "raw_price_eur_mwh"] == 95
    assert np.isnan(p.loc["DE", "raw_price_eur_mwh"])
    assert p.loc["DE", "status"] == "missing_period_inputs"


def test_missing_consumed_source_blocks_whole_coupled_period_before_lp(monkeypatch):
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, "linprog", lambda *a, **kw: pytest.fail("Missing inputs reached LP"))
    d = demand({"FR": 50., "DE": 150.})
    q = qualified(d).assign(consumed_inputs_complete=[True, False])
    p = coupled(qualification=q).prices
    assert set(p.status) == {"missing_period_inputs"}
    assert p.raw_price_eur_mwh.isna().all()


def test_zonal_empty_demand_empty_offers_has_indeterminate_price():
    p = run(demand({"FR": 0}), offers().iloc[:0]).prices.iloc[0]
    assert p.status == "price_indeterminate"
    assert not p.eligible


def test_coupled_congestion_duals_balances_and_extra_100mw():
    initial, extra = coupled(), coupled(load_de=250.)
    p = initial.prices.set_index("zone")
    assert p.price_eur_mwh.to_dict() == {"DE": 100., "FR": 40.}
    assert p.net_position_mw.to_dict() == {"DE": -100., "FR": 100.}
    assert p.balance_error_mw.abs().max() < 1e-8
    assert p.net_position_mw.sum() == 0
    assert initial.constraints.remaining_ram_mw.min() == 0
    assert initial.constraints.ram_relaxation_value_eur_mwh.max() == 60
    a = initial.dispatch.groupby("zone").generation_mw.sum()
    b = extra.dispatch.groupby("zone").generation_mw.sum()
    assert b.DE - a.DE == 100
    assert b.FR == a.FR


def test_uncongested_coupling_equalizes_price():
    p = coupled(ram=1000).prices
    assert p.price_eur_mwh.tolist() == [40., 40.]
    assert p.eligible.all()


def test_coupled_matrices_are_sparse_and_method_highs(monkeypatch):
    import scipy.optimize
    from scipy.sparse import issparse
    original = scipy.optimize.linprog
    calls = []
    def checking(*args, **kwargs):
        assert issparse(kwargs["A_eq"]) and issparse(kwargs["A_ub"])
        assert kwargs["method"] == "highs"
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(scipy.optimize, "linprog", checking)
    coupled()
    assert calls == [True]


@pytest.mark.parametrize("override,status", [({"domain_reference_qualified": False}, "unqualified_domain_reference"),
    ({"inputs_qualified": False}, "network_inputs_unqualified"),
    ({"boundary_net_positions_mw": {"BE": 0}}, "boundary_unqualified")])
def test_unqualified_network_never_even_computes_diagnostic_lp(monkeypatch, override, status):
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, "linprog", lambda *a, **kw: pytest.fail("Unqualified domain reached LP"))
    p = coupled(network_config=network_config(**override)).prices
    assert set(p.status) == {status}
    assert p.raw_price_eur_mwh.isna().all()
    assert p.price_eur_mwh.isna().all()
    assert not p.eligible.any()


@pytest.mark.parametrize("change", ["missing_coordinate", "extra_coordinate", "undeclared_boundary"])
def test_ptdf_universe_is_complete_or_contract_rejected(change):
    net, cfg = network(), network_config()
    if change == "missing_coordinate":
        net = net.drop(columns="ptdf_DE")
    elif change == "extra_coordinate":
        net["ptdf_BE"] = 0.
    else:
        cfg["zones"].append("BE")
    with pytest.raises(DispatchError, match="PTDF"):
        coupled(network=net, network_config=cfg)


def test_explicit_boundary_net_position_and_ram_reference_are_preserved():
    cfg = network_config(zones=["FR", "DE", "BE"], boundary_net_positions_mw={"BE": 50.},
                         boundary_qualified=True, boundary_hypothesis="Synthetic fixed pre-cutoff export 50 MW")
    result = coupled(network=network(extra=True), network_config=cfg)
    p = result.prices.set_index("zone")
    assert p.net_position_mw.sum() == -50
    assert p.loc["FR", "net_position_mw"] == 75
    assert p.loc["DE", "net_position_mw"] == -125
    assert p.generation_mw.sum() + p.shortage_mw.sum() == p.demand_mw.sum() - 50
    assert result.constraints.ptdf_expression_mw.max() == 100


def test_one_unqualified_coupled_stack_taints_all_prices_but_keeps_diagnostics():
    d = demand({"FR": 50., "DE": 150.})
    q = qualified(d)
    q.loc[q.zone.eq("DE"), "stack_complete"] = False
    p = coupled(qualification=q).prices
    assert p.price_eur_mwh.isna().all()
    assert p.raw_price_eur_mwh.notna().all()
    assert not p.eligible.any()
    assert set(p.status) == {"stack_incomplete", "coupled_stack_incomplete"}


def test_missing_network_period_or_active_coordinate_never_becomes_zero():
    p = coupled(network=network(timestamp=T + pd.Timedelta(hours=1))).prices
    assert set(p.status) == {"network_period_missing"}
    net = network()
    net["ptdf_DE"] = np.nan
    p = coupled(network=net).prices
    assert set(p.status) == {"network_period_incomplete"}
    assert p.raw_price_eur_mwh.isna().all()


def test_malformed_ptdf_is_contract_error_not_missing_data():
    net = network()
    net["ptdf_DE"] = "not_a_number"
    with pytest.raises(DispatchError, match="malformed"):
        coupled(network=net)


def test_infeasible_network_abstains_without_hiding_fallback():
    p = coupled(ram=-1).prices
    assert set(p.status) == {"dispatch_infeasible"}
    assert p.price_eur_mwh.isna().all()


@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2026-10-25", 25)])
def test_dst_delivery_periods_preserved_without_resampling(day, hours):
    start = pd.Timestamp(day, tz="Europe/Paris")
    end = start + pd.DateOffset(days=1)
    stamps = pd.date_range(start, end, inclusive="left", freq="h").tz_convert("UTC")
    d = pd.concat([demand(timestamp=t) for t in stamps], ignore_index=True)
    o = pd.concat([offers(timestamp=t) for t in stamps], ignore_index=True)
    p = run(d, o).prices
    assert len(p) == hours
    assert p.delivery_start_utc.tolist() == list(stamps)
    assert p.eligible.all()


def test_vectorized_zonal_matches_scalar_oracle_for_prices_ledger_and_gaps():
    from marginal_cost_expert.dispatch import _zonal, _output_rows, _price_record
    rng = np.random.default_rng(38021)
    timestamps = pd.date_range(T, periods=36, freq="h")
    rows, entries = [], []
    for time in timestamps:
        for zone in ["FR", "DE", "NL"]:
            rows.append({"delivery_start_utc": time, "zone": zone, "demand_mw": float(rng.integers(0, 900))})
            for segment in ["nuclear", "coal", "lignite", "ccgt", "ocgt"]:
                entries.append({"delivery_start_utc": time, "zone": zone, "segment": segment,
                                "technology": segment, "capacity_mw": float(rng.integers(0, 200)),
                                "bid_eur_mwh": float(rng.choice([-20, 10, 40, 70, 150])), "is_fatal": segment == "nuclear"})
    d, o = pd.DataFrame(rows), pd.DataFrame(entries)
    q = qualified(d).assign(consumed_inputs_complete=True)
    q.loc[2::11, "inputs_qualified"] = False
    q.loc[3::13, "stack_complete"] = False
    q.loc[5::19, "consumed_inputs_complete"] = False
    d.loc[7, "demand_mw"] = np.nan
    o.loc[90, "capacity_mw"] = np.nan
    result = run(d, o, qualification=q)
    expected_prices, expected_ledger = [], []
    indexed = q.set_index(K)
    for _, row in d.iterrows():
        ledger = o.loc[o.delivery_start_utc.eq(row.delivery_start_utc) & o.zone.eq(row.zone)].reset_index(drop=True)
        flags = indexed.loc[(row.delivery_start_utc, row.zone)]
        if not flags.consumed_inputs_complete or not np.isfinite(row.demand_mw) or not np.isfinite(ledger[["capacity_mw", "bid_eur_mwh"]]).all().all():
            record = _price_record(row, flags, candidate_id="central", mode="zonal_degraded", reason="missing_period_inputs")
        else:
            allocation, price, shortage = _zonal(row.demand_mw, ledger, 4000.)
            record, detail = _output_rows(row, flags, ledger, allocation, price, shortage, 0.,
                                         candidate_id="central", mode="zonal_degraded")
            expected_ledger.append(detail)
        expected_prices.append(record)
    expected = pd.DataFrame(expected_prices).sort_values(K).reset_index(drop=True)
    pd.testing.assert_frame_equal(result.prices, expected, check_like=True, check_exact=False, atol=1e-9, rtol=1e-12)
    expected = pd.concat(expected_ledger).sort_values([*K, "segment"]).reset_index(drop=True)
    actual = result.dispatch.sort_values([*K, "segment"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_like=True, check_exact=False, atol=1e-9, rtol=1e-12)


def test_dst_two_02h_periods_keep_distinct_merit_orders():
    # Both are 02:00 local on the fall-back day, with distinct physical offers.
    first, second = pd.Timestamp("2026-10-25T00:00:00Z"), pd.Timestamp("2026-10-25T01:00:00Z")
    assert first.tz_convert("Europe/Paris").hour == second.tz_convert("Europe/Paris").hour == 2
    d = pd.concat([demand(timestamp=first), demand(timestamp=second)], ignore_index=True)
    o = pd.concat([offers(timestamp=first), offers([("FR", "ccgt", 200., 95.)], timestamp=second)], ignore_index=True)
    result = run(d, o)
    assert result.prices.price_eur_mwh.tolist() == [40., 95.]
    assert result.dispatch.groupby("delivery_start_utc").generation_mw.sum().tolist() == [150., 150.]
