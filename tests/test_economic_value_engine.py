from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from economic_value.engine import EconomicValueError, simulate


def panel(days=1, *, start="2026-01-01", zones=("FR",), models=("candidate",)):
    first = pd.Timestamp(start, tz="Europe/Paris")
    last = (pd.Timestamp(start) + pd.Timedelta(days=days)).tz_localize("Europe/Paris")
    times = pd.date_range(first, last, freq="h", inclusive="left").tz_convert("UTC")
    records = []
    for model in models:
        for zone in zones:
            for timestamp in times:
                localday = timestamp.tz_convert("Europe/Paris").date()
                origin = (pd.Timestamp(localday) - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
                records.append({
                    "timestamp_utc": timestamp, "zone": zone, "model": model,
                    "forecast": 120., "q10": 115., "q90": 130., "actual": 110.,
                    "benchmark_forecast": 90., "reference_price": 100.,
                    "reference_available_at_utc": origin - pd.Timedelta(hours=1),
                    "forecast_origin_utc": origin, "reference_eligible": True,
                    "forecast_eligible": True, "duration_hours": 1., "sample": "evaluation",
                })
    return pd.DataFrame(records)


def config(**overrides):
    return {"zone_capacity_mw": {"FR": 100.}, **overrides}


def metric(result, strategy="model", zone="FR", model="candidate"):
    return result.metrics.loc[
        result.metrics.strategy.eq(strategy) & result.metrics.zone.eq(zone) & result.metrics.model.eq(model)
    ].iloc[0]


def test_fixed_signed_positions_and_incremental_eva():
    result = simulate(panel(), config())
    assert metric(result).pnl_net_eur == 24000
    assert metric(result, "benchmark").pnl_net_eur == -24000
    assert metric(result, "no_forecast").pnl_net_eur == 0
    assert metric(result).economic_value_added_eur == 48000
    assert metric(result).pnl_per_mwh == 10
    assert metric(result).hit_ratio_directional == 1
    assert metric(result, "benchmark").hit_ratio_directional == 0
    assert math.isnan(metric(result).pnl_per_mw_year)


def test_governed_position_scores_actual_direction_and_fraction_without_changing_forecast():
    data = panel()
    data["policy_position_fraction"] = -.5
    data["policy_available_at_utc"] = data.forecast_origin_utc
    result = simulate(data, config(governed_models=["candidate"], transaction_cost_eur_mwh=1))
    chosen = result.rows.loc[result.rows.strategy.eq("model")]
    assert chosen.forecast.eq(120).all()
    assert chosen.position_mw.eq(-50).all()
    assert metric(result).pnl_net_eur == -13200
    assert metric(result).hit_ratio_directional == 0
    assert metric(result).absolute_energy_mwh == 1200
    assert metric(result, "benchmark").pnl_net_eur == -26400
    assert metric(result, "no_forecast").pnl_net_eur == 0
    assert result.audit["fixed_policy"] is False


def test_external_positions_are_opt_in_and_bounded_and_available_before_cutoff():
    data = panel()
    data["policy_position_fraction"] = .5
    data["policy_available_at_utc"] = data.forecast_origin_utc
    with pytest.raises(EconomicValueError, match="Positions externes"):
        simulate(data, config())
    data.loc[0, "policy_position_fraction"] = 1.01
    with pytest.raises(EconomicValueError, match="Fraction"):
        simulate(data, config(governed_models=["candidate"]))
    data.loc[0, "policy_position_fraction"] = .5
    data.loc[0, "policy_available_at_utc"] += pd.Timedelta(seconds=1)
    with pytest.raises(EconomicValueError, match="cutoff"):
        simulate(data, config(governed_models=["candidate"]))


def test_costs_charged_on_absolute_energy_and_applied_to_both_sides():
    result = simulate(panel(), config(transaction_cost_eur_mwh=1, slippage_eur_mwh=.5))
    assert metric(result).trading_cost_eur == 3600
    assert metric(result).pnl_net_eur == 20400
    assert metric(result, "benchmark").pnl_net_eur == -27600
    assert metric(result).economic_value_added_eur == 48000


def test_signal_must_exceed_threshold_including_costs():
    data = panel()
    data["forecast"] = 103.
    data["benchmark_forecast"] = 96.9
    result = simulate(data, config(signal_threshold_eur_mwh=2, transaction_cost_eur_mwh=1))
    assert metric(result).active_hours == 0
    assert metric(result, "benchmark").active_hours == 24
    assert math.isnan(metric(result).pnl_per_mwh)


def test_negative_prices_are_valid_and_sell_makes_money():
    data = panel()
    data[["reference_price", "forecast", "benchmark_forecast", "actual"]] = [-10, -30, 0, -20]
    result = simulate(data, config())
    assert metric(result).pnl_net_eur == 24000
    assert set(result.rows.loc[result.rows.strategy.eq("model"), "signal"]) == {"SELL"}
    assert set(result.breakdowns["spike"].group) == {"low"}


@pytest.mark.parametrize(("start", "hours"), [("2026-03-29", 23), ("2026-10-25", 25)])
def test_dst_elapsed_hours_and_energy_conservation(start, hours):
    result = simulate(panel(start=start), config())
    value = metric(result)
    assert value.eligible_hours == hours
    assert value.expected_hours == hours
    assert value.absolute_energy_mwh == hours * 100
    assert value.pnl_net_eur == hours * 1000
    assert result.daily.complete_day.all()
    assert len(result.rows.loc[result.rows.strategy.eq("model")]) == hours


def test_explicit_half_hour_duration_scales_settlement():
    data = panel().iloc[:1].copy()
    data["duration_hours"] = .5
    result = simulate(data, config())
    assert metric(result).pnl_net_eur == 500
    assert metric(result).absolute_energy_mwh == 50
    assert metric(result).coverage_ratio == pytest.approx(.5 / 24)


def test_first_loss_is_included_in_drawdown_from_initial_zero():
    data = panel()
    data["actual"] = 90.
    result = simulate(data, config())
    assert metric(result).max_drawdown_eur == 24000
    assert math.isnan(metric(result).max_drawdown_percent)
    result_capital = simulate(data, config(initial_capital_eur=100000))
    assert metric(result_capital).max_drawdown_percent == 24


def test_flat_and_constant_pnl_have_undefined_sharpe():
    result = simulate(panel(days=2), config())
    assert math.isnan(metric(result, "no_forecast").pnl_per_mwh)
    assert math.isnan(metric(result, "no_forecast").hit_ratio_directional)
    assert math.isnan(metric(result, "no_forecast").sharpe_daily_pnl)
    assert math.isnan(metric(result).sharpe_daily_pnl)


def test_daily_sharpe_uses_complete_days_and_sample_std():
    data = panel(days=3)
    localday = data.timestamp_utc.dt.tz_convert("Europe/Paris").dt.day
    data["actual"] = localday.map({1: 101., 2: 102., 3: 99.})
    result = simulate(data, config())
    pnl = np.array([2400., 4800., -2400.])
    assert metric(result).sharpe_daily_pnl == pytest.approx(np.sqrt(365) * pnl.mean() / pnl.std(ddof=1))


@pytest.mark.parametrize("column", ["actual", "forecast", "benchmark_forecast", "reference_price"])
def test_missing_input_excluded_symmetrically_and_not_counted_as_zero(column):
    data = panel()
    data.loc[0, column] = np.nan
    result = simulate(data, config())
    for strategy in ("model", "benchmark", "no_forecast"):
        assert metric(result, strategy).eligible_hours == 23
        row = result.rows.loc[result.rows.strategy.eq(strategy) & result.rows.timestamp_utc.eq(data.iloc[0].timestamp_utc)].iloc[0]
        assert math.isnan(row.pnl_net_eur)
    assert metric(result).sharpe_complete_days == 0


def test_pending_observation_can_have_exante_signal_but_never_pnl():
    data = panel()
    data["actual"] = np.nan
    result = simulate(data, config())
    model = result.rows.loc[result.rows.strategy.eq("model")]
    assert set(model.signal) == {"BUY"}
    assert set(model.evaluation_status) == {"pending_observation"}
    assert model.pnl_net_eur.isna().all()
    assert math.isnan(metric(result).pnl_net_eur)
    assert math.isnan(metric(result, "no_forecast").pnl_net_eur)


def test_late_reference_and_invalid_flags_fail_closed():
    data = panel()
    data.loc[0, "reference_available_at_utc"] = data.loc[0, "forecast_origin_utc"] + pd.Timedelta(seconds=1)
    data.loc[1, "reference_eligible"] = False
    data.loc[2, "forecast_eligible"] = False
    result = simulate(data, config())
    assert metric(result).eligible_hours == 21
    model = result.rows.loc[result.rows.strategy.eq("model")].head(3)
    assert model.position_mw.eq(0).all()
    assert model.pnl_net_eur.isna().all()


def test_reference_available_exactly_at_origin_is_eligible():
    data = panel()
    data["reference_available_at_utc"] = data["forecast_origin_utc"]
    assert metric(simulate(data, config())).eligible_hours == 24


def test_forecast_origin_must_precede_delivery():
    data = panel()
    data.loc[0, "forecast_origin_utc"] = data.loc[0, "timestamp_utc"]
    result = simulate(data, config())
    assert metric(result).eligible_hours == 23
    assert result.audit["ineligible_reasons"]["forecast_not_before_delivery"] == 1


def test_live_rows_never_enter_annual_or_daily_pnl_even_if_actual_known():
    data = panel(days=2)
    data.loc[24:, "sample"] = "live"
    result = simulate(data, config())
    assert result.audit["evaluation_calendar_days"] == 1
    assert result.rows.loc[result.rows["sample"].eq("live"), "pnl_net_eur"].isna().all()
    assert result.daily.delivery_day.nunique() == 1
    assert metric(result).pnl_net_eur == 24000
    assert set(result.breakdowns["monthly"].group) == {"2026-01"}


def test_fixed_portfolio_capacity_is_not_multiplied_by_countries_or_candidates():
    data = panel(zones=("FR", "DE"), models=("first", "second"))
    result = simulate(data, config(zone_capacity_mw={"FR": 40, "DE": 60}))
    for model in ("first", "second"):
        assert metric(result, zone="PORTFOLIO", model=model).pnl_net_eur == 24000
        assert metric(result, zone="PORTFOLIO", model=model).allocated_capacity_mw == 100
        assert metric(result, zone="FR", model=model).pnl_net_eur == 9600
    assert result.audit["model_alternatives_are_not_cumulative"]


def test_portfolio_requires_all_zones_at_each_hour_without_redistributing_capacity():
    data = panel(zones=("FR", "DE"))
    data = data.drop(data.loc[data.zone.eq("DE")].index[0])
    result = simulate(data, config(zone_capacity_mw={"FR": 40, "DE": 60}))
    assert metric(result, zone="FR").pnl_net_eur == 9600
    assert metric(result, zone="DE").pnl_net_eur == 13800
    assert metric(result, zone="PORTFOLIO").pnl_net_eur == 23000
    assert metric(result, zone="PORTFOLIO").coverage_ratio == 23 / 24
    assert metric(result, zone="PORTFOLIO").sharpe_complete_days == 0


def test_absent_allocated_country_does_not_create_an_apparently_complete_portfolio():
    result = simulate(panel(), config(zone_capacity_mw={"FR": 50, "DE": 50}))
    assert math.isnan(metric(result, zone="PORTFOLIO").pnl_net_eur)
    assert metric(result, zone="DE").eligible_hours == 0
    assert metric(result, zone="PORTFOLIO").allocated_capacity_mw == 100


def test_exact_full_365_day_window_enables_per_mw_year_without_annualisation():
    result = simulate(panel(days=365, start="2025-09-10"), config())
    value = metric(result, zone="PORTFOLIO")
    assert value.annual_window
    assert value.annual_fully_observed
    assert value.pnl_per_mw_year == 87600
    assert value.economic_value_added_per_mw_year == 175200
    assert value.pnl_net_eur == 8760000


def test_missing_hour_in_365_days_blocks_complete_annual_claim():
    data = panel(days=365, start="2025-09-10")
    data.loc[0, "actual"] = np.nan
    result = simulate(data, config())
    assert metric(result).annual_window
    assert not metric(result).annual_fully_observed
    assert math.isnan(metric(result).economic_value_added_per_mw_year)
    assert metric(result).eligible_hours == 8759


def test_ties_are_separate_directional_non_hits():
    data = panel()
    data.loc[0, "actual"] = 100
    result = simulate(data, config())
    assert metric(result).direction_ties == 1
    assert metric(result).hit_ratio_directional == pytest.approx(23 / 24)
    assert metric(result).winning_trade_ratio == pytest.approx(23 / 24)


def test_spikes_seasons_and_hourly_slices_are_analysis_only():
    data = pd.concat([panel(start="2026-01-01"), panel(start="2026-06-01")], ignore_index=True)
    data.loc[:23, "actual"] = 250
    result = simulate(data, config())
    assert set(result.breakdowns["season"].group) == {"winter", "summer"}
    assert set(result.breakdowns["spike"].group) == {"high", "normal"}
    model = result.rows.loc[result.rows.strategy.eq("model")]
    assert model.position_mw.eq(100).all()
    hourly = result.breakdowns["hour"]
    assert hourly.loc[hourly.strategy.eq("model") & hourly.zone.eq("FR"), "pnl_net_eur"].sum() == metric(result).pnl_net_eur


def test_confidence_is_heuristic_and_comparison_uses_same_candidate_buckets():
    data = panel()
    data.loc[0, ["q10", "q90"]] = [50, 200]
    result = simulate(data, config())
    model_rows = result.rows.loc[result.rows.strategy.eq("model")]
    assert set(model_rows.confidence) == {"low", "high"}
    assert set(result.rows.loc[result.rows.strategy.eq("benchmark"), "confidence"]) == {"unknown"}
    slices = result.breakdowns["confidence"]
    assert set(slices.group) == {"low", "high"}
    assert "economic_value_added_eur" in slices
    for confidence in ("low", "high"):
        selected = slices.loc[slices.zone.eq("FR") & slices.group.eq(confidence)]
        assert selected.eligible_hours.nunique() == 1
    assert not result.audit["confidence_is_probability"]
    assert metric(result).quantile_scored_intervals == 24
    assert metric(result).quantile_coverage == 1 / 24


def test_confidence_filter_cannot_privilege_model_when_benchmark_intervals_unknown():
    result = simulate(panel(), config(confidence_filter="high"))
    assert metric(result).eligible_hours == 0
    assert metric(result, "benchmark").eligible_hours == 0
    assert result.audit["ineligible_reasons"] == {"symmetric_uncertainty_missing": 24}


def test_confidence_filter_symmetric_when_both_intervals_provided():
    data = panel()
    data["benchmark_q10"], data["benchmark_q90"] = 85., 95.
    result = simulate(data, config(confidence_filter="high"))
    assert metric(result).active_hours == 24
    assert metric(result, "benchmark").active_hours == 24


@pytest.mark.parametrize("mutation", ["naive", "duplicate", "duration_zero", "overlap", "string_bool", "unallocated"])
def test_invalid_contracts_raise(mutation):
    data = panel()
    if mutation == "naive":
        data["timestamp_utc"] = data.timestamp_utc.dt.tz_localize(None)
    elif mutation == "duplicate":
        data = pd.concat([data, data.iloc[:1]], ignore_index=True)
    elif mutation == "duration_zero":
        data.loc[0, "duration_hours"] = 0
    elif mutation == "overlap":
        data.loc[0, "duration_hours"] = 2
    elif mutation == "string_bool":
        data["reference_eligible"] = "false"
    elif mutation == "unallocated":
        data["zone"] = "BE"
    with pytest.raises(EconomicValueError):
        simulate(data, config())


def test_overallocation_negative_cost_and_invalid_confidence_raise():
    with pytest.raises(EconomicValueError):
        simulate(panel(), config(zone_capacity_mw={"FR": 101}))
    with pytest.raises(EconomicValueError):
        simulate(panel(), config(transaction_cost_eur_mwh=-1))
    with pytest.raises(EconomicValueError):
        simulate(panel(), config(confidence_filter="optimize"))


def test_declared_calendar_retains_whole_missing_days_and_does_not_compress_time():
    data = panel(days=3)
    data = data.drop(data.iloc[24:48].index)
    result = simulate(data, config())
    series = result.daily.loc[result.daily.zone.eq("FR") & result.daily.strategy.eq("model")]
    assert len(series) == 3
    missing = series.iloc[1]
    assert missing.eligible_hours == 0
    assert math.isnan(missing.pnl_net_eur)
    assert not missing.complete_day
    assert metric(result).coverage_ratio == 2 / 3


def test_input_frame_is_not_mutated_and_alias_is_supported():
    data = panel().rename(columns={"timestamp_utc": "delivery_start_utc"})
    original = data.copy(deep=True)
    result = simulate(data, config())
    pd.testing.assert_frame_equal(data, original)
    assert "timestamp_utc" in result.rows


def test_live_only_panel_returns_signals_without_evaluation_metrics():
    data = panel()
    data["sample"] = "live"
    result = simulate(data, config())
    assert result.daily.empty
    assert result.audit["evaluation_calendar_days"] == 0
    assert math.isnan(metric(result).pnl_net_eur)
    assert result.rows.pnl_net_eur.isna().all()
