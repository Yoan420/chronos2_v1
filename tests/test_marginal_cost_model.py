"""Synthetic conservation, pricing and isolation tests; no operational runs."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from marginal_cost_expert.model import (
    MarginalCostExpert, MarginalCostModelError, candidate_scores, select_candidate,
)


def _config(**overrides):
    config = {
        "parameters": {
            "ccgt_efficiency": 0.5, "ocgt_efficiency": 0.25, "coal_efficiency": 0.4,
            "ccgt_vom_eur_mwh": 0.0, "ocgt_vom_eur_mwh": 0.0, "coal_vom_eur_mwh": 0.0,
            "gas_emission_tco2_mwh_th": 0.2,
            "must_run_bid_eur_mwh": 0.0, "scarcity_price_eur_mwh": 4000.0,
        },
        "scenarios": [{"id": "central", "hypothesis": "Declared central physical availability."}],
    }
    config.update(overrides)
    return config


def _features(zones=("FR",), **overrides):
    rows = []
    for zone in zones:
        row = {"delivery_start_utc": pd.Timestamp("2026-06-24T00:00:00Z"), "zone": zone,
               "demand_mw": 60.0, "must_run_mw": 20.0, "ccgt_available_mw": 100.0,
               "ocgt_available_mw": 100.0, "ttf_eur_mwh_th": 20.0, "eua_eur_tco2": 0.0}
        row.update(overrides)
        rows.append(row)
    return pd.DataFrame(rows)


def _network(timestamp, limit=100.0):
    return pd.DataFrame([
        {"delivery_start_utc": timestamp, "cnec_id": "forward", "ram_mw": limit, "ptdf_FR": 1.0, "ptdf_DE": 0.0},
        {"delivery_start_utc": timestamp, "cnec_id": "reverse", "ram_mw": limit, "ptdf_FR": -1.0, "ptdf_DE": 0.0},
    ])


def _coupled_inputs():
    features = _features(("FR", "DE"), must_run_mw=0.0, ccgt_available_mw=300.0, ocgt_available_mw=0.0)
    features.loc[features.zone == "FR", "demand_mw"] = 50.0
    features.loc[features.zone == "DE", ["demand_mw", "ttf_eur_mwh_th"]] = [150.0, 50.0]
    return features, _network(features.delivery_start_utc.iloc[0])


def test_zonal_merit_order_uses_direct_fuel_cost_and_no_exchange():
    model = MarginalCostExpert(_config())
    result = model.predict_candidates(_features()).iloc[0]
    assert result.price_eur_mwh == pytest.approx(40.0)
    assert result.ccgt_generation_mw == pytest.approx(40.0)
    assert result.must_run_generation_mw == pytest.approx(20.0)
    assert result.network_mode == "zonal_degraded"
    assert result.net_position_mw == 0.0
    assert result.balance_error_mw == pytest.approx(0.0)
    assert not result.observed_marginal_unit_identified
    assert np.isnan(result.congestion_component_eur_mwh)


def test_additional_100_mw_changes_segment_and_preserves_energy():
    model = MarginalCostExpert(_config())
    original = model.predict_candidates(_features()).iloc[0]
    increased = model.predict_candidates(_features(demand_mw=160.0)).iloc[0]
    assert increased.generation_mw - original.generation_mw == pytest.approx(100.0)
    assert increased.ccgt_generation_mw == pytest.approx(100.0)
    assert increased.ocgt_generation_mw == pytest.approx(40.0)
    assert increased.price_eur_mwh == pytest.approx(80.0)
    assert increased.balance_error_mw == pytest.approx(0.0)


def test_emissions_use_thermal_units_before_efficiency_conversion():
    result = MarginalCostExpert(_config()).predict_candidates(
        _features(ttf_eur_mwh_th=30.0, eua_eur_tco2=50.0)
    ).iloc[0]
    assert result.price_eur_mwh == pytest.approx((30.0 + 50.0 * 0.2) / 0.5)


def test_scarcity_is_explicit_and_has_shortage_diagnostic():
    result = MarginalCostExpert(_config()).predict_candidates(_features(demand_mw=300.0)).iloc[0]
    assert result.price_eur_mwh == pytest.approx(4000.0)
    assert result.shortage_mw == pytest.approx(80.0)
    assert result.capacity_margin_mw == pytest.approx(-80.0)
    assert result.generation_mw + result.shortage_mw == pytest.approx(result.demand_mw)
    assert result.marginal_regime_proxy == "scarcity"


def test_negative_price_requires_configured_low_bid_and_available_surplus():
    config = _config()
    config["parameters"]["must_run_bid_eur_mwh"] = -35.0
    result = MarginalCostExpert(config).predict_candidates(_features(demand_mw=30.0, must_run_mw=80.0)).iloc[0]
    assert result.price_eur_mwh == pytest.approx(-35.0)
    assert result.must_run_generation_mw == pytest.approx(30.0)
    assert result.curtailment_mw == pytest.approx(50.0)
    assert result.marginal_regime_proxy == "surplus"
    assert result.ccgt_generation_mw == 0.0


@pytest.mark.parametrize("parameter", ["must_run_bid_eur_mwh", "scarcity_price_eur_mwh"])
def test_negative_and_scarcity_bids_are_never_invented(parameter):
    config = _config()
    del config["parameters"][parameter]
    with pytest.raises(MarginalCostModelError, match="explicitly configured"):
        MarginalCostExpert(config)


def test_optional_coal_is_in_thermal_energy_units_and_respects_capacity():
    result = MarginalCostExpert(_config()).predict_candidates(
        _features(demand_mw=60.0, must_run_mw=0.0, coal_available_mw=80.0, coal_eur_mwh_th=10.0)
    ).iloc[0]
    assert result.price_eur_mwh == pytest.approx(25.0)
    assert result.coal_generation_mw == pytest.approx(60.0)
    assert result.ccgt_generation_mw == pytest.approx(0.0)


def test_coal_cost_cannot_be_silently_imputed():
    with pytest.raises(MarginalCostModelError, match="together"):
        MarginalCostExpert(_config()).predict_candidates(_features(coal_available_mw=10.0))


def test_full_zone_ptdf_congestion_changes_prices_and_respects_net_balance():
    features, network = _coupled_inputs()
    model = MarginalCostExpert(_config(network={"zones": ["FR", "DE"]}))
    result = model.predict_candidates(features, network).set_index("zone")
    assert result.loc["FR", "price_eur_mwh"] == pytest.approx(40.0)
    assert result.loc["DE", "price_eur_mwh"] == pytest.approx(100.0)
    assert result.loc["FR", "net_position_mw"] == pytest.approx(100.0)
    assert result.loc["DE", "net_position_mw"] == pytest.approx(-100.0)
    assert result.net_position_mw.sum() == pytest.approx(0.0)
    assert result.generation_mw.sum() == pytest.approx(features.demand_mw.sum())
    assert (result.network_mode == "ptdf_coupled").all()
    assert (result.binding_constraints == 1).all()
    assert result.balance_error_mw.abs().max() < 1e-8
    assert (result.price_eur_mwh - result.reference_price_eur_mwh).to_numpy() == pytest.approx(
        result.congestion_component_eur_mwh.to_numpy()
    )


def test_relaxed_network_equalizes_prices_without_inventing_ntc():
    features, network = _coupled_inputs()
    network["ram_mw"] = 1000.0
    result = MarginalCostExpert(_config(network={"zones": ["FR", "DE"]})).predict_candidates(features, network)
    assert result.price_eur_mwh.to_numpy() == pytest.approx([40.0, 40.0])
    assert result.net_position_mw.sum() == pytest.approx(0.0)
    assert (result.binding_constraints == 0).all()


def test_additional_100_mw_in_congested_zone_balances_locally():
    features, network = _coupled_inputs()
    model = MarginalCostExpert(_config(network={"zones": ["FR", "DE"]}))
    original = model.predict_candidates(features, network).set_index("zone")
    features.loc[features.zone == "DE", "demand_mw"] += 100.0
    increased = model.predict_candidates(features, network).set_index("zone")
    assert increased.loc["DE", "generation_mw"] - original.loc["DE", "generation_mw"] == pytest.approx(100.0)
    assert increased.loc["FR", "generation_mw"] == pytest.approx(original.loc["FR", "generation_mw"])
    assert increased.loc["DE", "price_eur_mwh"] == pytest.approx(100.0)
    assert increased.net_position_mw.sum() == pytest.approx(0.0)


def test_solver_uses_sparse_highs_matrices(monkeypatch):
    import scipy.optimize
    import scipy.sparse

    original = scipy.optimize.linprog
    calls = []

    def checked(*args, **kwargs):
        assert scipy.sparse.issparse(kwargs["A_eq"])
        assert scipy.sparse.issparse(kwargs["A_ub"])
        assert kwargs["method"] == "highs"
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(scipy.optimize, "linprog", checked)
    features, network = _coupled_inputs()
    MarginalCostExpert(_config(network={"zones": ["FR", "DE"]})).predict_candidates(features, network)
    assert calls == [True]


def test_zonal_mode_does_not_load_solver(monkeypatch):
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, "linprog", lambda *args, **kwargs: pytest.fail("No LP needed for isolated stacks"))
    result = MarginalCostExpert(_config()).predict_candidates(_features(("FR", "DE")))
    assert len(result) == 2
    assert (result.net_position_mw == 0.0).all()


@pytest.mark.parametrize("violation", ["missing_zone", "missing_ptdf", "extra_ptdf", "empty", "wrong_time", "nan_ram"])
def test_invalid_network_is_never_silently_downgraded_or_reduced(violation):
    features, network = _coupled_inputs()
    if violation == "missing_zone":
        features = features.loc[features.zone == "FR"]
    elif violation == "missing_ptdf":
        network = network.drop(columns="ptdf_DE")
    elif violation == "extra_ptdf":
        network["ptdf_BE"] = 0.0
    elif violation == "empty":
        network = network.iloc[:0]
    elif violation == "wrong_time":
        network["delivery_start_utc"] += pd.Timedelta(hours=1)
    else:
        network.loc[0, "ram_mw"] = np.nan
    with pytest.raises(MarginalCostModelError):
        MarginalCostExpert(_config(network={"zones": ["FR", "DE"]})).predict_candidates(features, network)


def test_explicit_external_zone_position_is_conserved_and_applied_to_ram():
    features = _features(must_run_mw=0.0, demand_mw=100.0)
    config = _config(network={"zones": ["FR", "DE"], "boundary_net_positions_mw": {"DE": 50.0},
                              "boundary_hypothesis": "Fixed 50 MW modeled import from omitted DE region."})
    network = _network(features.delivery_start_utc.iloc[0])
    network["ptdf_DE"] = [0.2, -0.2]
    result = MarginalCostExpert(config).predict_candidates(features, network).iloc[0]
    assert result.net_position_mw == pytest.approx(-50.0)
    assert result.generation_mw == pytest.approx(50.0)
    assert result.net_position_mw + 50.0 == pytest.approx(0.0)
    # forward flow = -50 + .2 * 50 = -40; reverse = 40; min slack = 60 MW.
    assert result.min_ram_slack_mw == pytest.approx(60.0)


def test_boundary_positions_need_explicit_hypothesis_and_real_network():
    config = _config(network={"zones": ["FR", "DE"], "boundary_net_positions_mw": {"DE": 50.0}})
    features = _features()
    with pytest.raises(MarginalCostModelError, match="hypothesis"):
        MarginalCostExpert(config).predict_candidates(features, _network(features.delivery_start_utc.iloc[0]))
    with pytest.raises(MarginalCostModelError, match="full PTDF"):
        MarginalCostExpert(config).predict_candidates(features)


def test_infeasible_network_fails_instead_of_publishing_arbitrary_price():
    features, network = _coupled_inputs()
    network["ram_mw"] = -1000.0
    with pytest.raises(MarginalCostModelError, match="infeasible"):
        MarginalCostExpert(_config(network={"zones": ["FR", "DE"]})).predict_candidates(features, network)


@pytest.mark.parametrize("forbidden", ["actual", "target", "price_lag_24h", "chronos2__q50", "storm"])
def test_electric_prices_and_forecasts_are_not_model_features(forbidden):
    with pytest.raises(MarginalCostModelError, match="Non-physical"):
        MarginalCostExpert(_config()).predict_candidates(_features(**{forbidden: 100.0}))


@pytest.mark.parametrize("kind", ["naive", "duplicate", "negative", "nan"])
def test_invalid_physical_inputs_fail_closed(kind):
    features = _features()
    if kind == "naive":
        features["delivery_start_utc"] = pd.Timestamp("2026-06-24")
    elif kind == "duplicate":
        features = pd.concat([features, features])
    elif kind == "negative":
        features["demand_mw"] = -1.0
    else:
        features["ttf_eur_mwh_th"] = np.nan
    with pytest.raises(MarginalCostModelError):
        MarginalCostExpert(_config()).predict_candidates(features)


@pytest.mark.parametrize("parameter,value", [("ccgt_efficiency", 1.0), ("thermal_availability_scale", 5.0),
                                               ("thermal_bid_premium_eur_mwh", 250.0)])
def test_scenario_parameters_are_bounded_physical_hypotheses(parameter, value):
    config = _config()
    config["scenarios"][0][parameter] = value
    with pytest.raises(MarginalCostModelError):
        MarginalCostExpert(config)


def test_daily_scenario_selection_reuses_precomputed_predictions_without_solver(monkeypatch):
    config = _config(scenarios=[
        {"id": "central", "hypothesis": "Declared central physical availability."},
        {"id": "premium", "hypothesis": "Explicit modest bid premium, not fitted price offset.",
         "thermal_bid_premium_eur_mwh": 10.0},
    ])
    features = _features()
    model = MarginalCostExpert(config)
    predictions = model.predict_candidates(features)
    labels = features.loc[:, ["delivery_start_utc", "zone"]].assign(actual=49.0)
    assert select_candidate(predictions, labels) == "premium"
    scores = candidate_scores(predictions, labels)
    assert scores.mae.to_list() == pytest.approx([1.0, 9.0])
    monkeypatch.setattr(model, "predict_candidates", lambda *args, **kwargs: pytest.fail("No recomputation when bank is cached"))
    assert model.fit(features, labels, candidate_predictions=predictions) is model
    assert model.selected_candidate_id_ == "premium"
    assert model.selection_audit_["labels_used_as_features"] is False
    assert model.selection_audit_["continuous_parameter_fit"] is False


def test_selection_uses_explicit_label_keys_not_other_periods():
    first = _features()
    second = _features(delivery_start_utc=pd.Timestamp("2026-06-25T00:00:00Z"))
    features = pd.concat([first, second])
    model = MarginalCostExpert(_config(scenarios=[
        {"id": "central", "hypothesis": "Central."},
        {"id": "premium", "hypothesis": "Small fixed premium.", "thermal_bid_premium_eur_mwh": 10.0},
    ]))
    predictions = model.predict_candidates(features)
    predictions.loc[predictions.delivery_start_utc == second.delivery_start_utc.iloc[0], "price_eur_mwh"] = 9999.0
    labels = first.loc[:, ["delivery_start_utc", "zone"]].assign(actual=40.0)
    assert select_candidate(predictions, labels) == "central"
    assert candidate_scores(predictions, labels).n_rows.eq(1).all()


def test_misaligned_scenario_support_cannot_win_by_dropping_hard_labels():
    features = pd.concat([_features(), _features(delivery_start_utc=pd.Timestamp("2026-06-25T00:00:00Z"))])
    model = MarginalCostExpert(_config())
    predictions = model.predict_candidates(features).iloc[:1]
    labels = features.loc[:, ["delivery_start_utc", "zone"]].assign(actual=40.0)
    with pytest.raises(MarginalCostModelError, match="cover"):
        select_candidate(predictions, labels)


def test_dst_physical_hours_remain_distinct_and_not_fixed_to_24_rows():
    start = pd.Timestamp("2025-10-26", tz="Europe/Paris").tz_convert("UTC")
    end = pd.Timestamp("2025-10-27", tz="Europe/Paris").tz_convert("UTC")
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    features = pd.concat([_features(delivery_start_utc=hour) for hour in hours], ignore_index=True)
    result = MarginalCostExpert(_config()).predict_candidates(features)
    assert len(result) == 25
    assert pd.DatetimeIndex(result.delivery_start_utc).equals(hours)
    assert result.delivery_start_utc.dt.tz_convert("Europe/Paris").dt.hour.eq(2).sum() == 2


def test_config_and_inputs_unchanged_and_output_contains_no_invented_quantiles():
    config, features = _config(), _features()
    original_config, original_features = deepcopy(config), features.copy(deep=True)
    model = MarginalCostExpert(config)
    result = model.predict_candidates(features)
    assert config == original_config
    pd.testing.assert_frame_equal(features, original_features)
    assert not {"q10", "q50", "q90"}.intersection(result.columns)
    with pytest.raises(MarginalCostModelError, match="fit"):
        model.predict(features)
    labels = features.loc[:, ["delivery_start_utc", "zone"]].assign(actual=40.0)
    model.fit(features, labels, candidate_predictions=result)
    assert model.predict(features).price_eur_mwh.iloc[0] == pytest.approx(40.0)


def test_signed_residual_demand_preserves_net_requirement_without_double_subtraction():
    config = _config(demand_basis="residual")
    config["parameters"]["must_run_bid_eur_mwh"] = -35.0
    features = _features(demand_mw=-20.0, must_run_mw=50.0)
    original = features.copy(deep=True)
    result = MarginalCostExpert(config).predict_candidates(features).iloc[0]
    assert result.price_eur_mwh == pytest.approx(-35.0)
    assert result.input_demand_mw == pytest.approx(-20.0)
    assert result.input_must_run_mw == pytest.approx(50.0)
    assert result.residual_surplus_mw == pytest.approx(20.0)
    assert result.demand_mw == pytest.approx(0.0)
    assert result.curtailment_mw == pytest.approx(70.0)
    assert result.shortage_mw == pytest.approx(0.0)
    assert result.physical_scope == "residual_stack_proxy"
    assert result.marginal_regime_proxy == "surplus"
    assert not result.full_renewable_curtailment_modeled
    pd.testing.assert_frame_equal(features, original)


def test_positive_residual_demand_subtracts_nuclear_once():
    result = MarginalCostExpert(_config(demand_basis="residual")).predict_candidates(
        _features(demand_mw=100.0, must_run_mw=60.0)
    ).iloc[0]
    assert result.ccgt_generation_mw == pytest.approx(40.0)
    assert result.must_run_generation_mw == pytest.approx(60.0)
    assert result.residual_surplus_mw == 0.0
    assert result.price_eur_mwh == pytest.approx(40.0)


def test_negative_residual_zone_can_export_surplus_through_full_ptdf():
    features, network = _coupled_inputs()
    features.loc[features.zone == "FR", "demand_mw"] = -20.0
    features.loc[features.zone == "DE", "demand_mw"] = 80.0
    config = _config(demand_basis="residual", network={"zones": ["FR", "DE"]})
    result = MarginalCostExpert(config).predict_candidates(features, network).set_index("zone")
    assert result.net_position_mw.sum() == pytest.approx(0.0)
    assert result.loc["FR", "residual_surplus_mw"] == pytest.approx(20.0)
    assert result.loc["FR", "must_run_generation_mw"] == pytest.approx(20.0)
    assert result.loc["FR", "ccgt_generation_mw"] == pytest.approx(60.0)
    assert result.loc["FR", "net_position_mw"] == pytest.approx(80.0)
    assert result.loc["DE", "net_position_mw"] == pytest.approx(-80.0)
    assert result.balance_error_mw.abs().max() < 1e-8


def test_residual_basis_must_be_valid_and_cached_fit_labels_must_match_features():
    with pytest.raises(MarginalCostModelError, match="demand_basis"):
        MarginalCostExpert(_config(demand_basis="automatic"))
    first = _features()
    second = _features(delivery_start_utc=pd.Timestamp("2026-06-25T00:00:00Z"))
    model = MarginalCostExpert(_config())
    predictions = model.predict_candidates(second)
    labels = second.loc[:, ["delivery_start_utc", "zone"]].assign(actual=40.0)
    with pytest.raises(MarginalCostModelError, match="outside"):
        model.fit(first, labels, candidate_predictions=predictions)
