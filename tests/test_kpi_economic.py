from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from economic_value.runner import load_config
from kpi_report.economic import DEFAULT_CONFIG, compute_economic_kpis
from kpi_report.metrics import KPIError


def panel(start="2026-09-13", days=2, models=("a", "b"), zones=("FR",)):
    begin = pd.Timestamp(start, tz="Europe/Paris")
    dates = pd.date_range(begin, begin+pd.DateOffset(days=days), freq="h", inclusive="left").tz_convert("UTC")
    rows = []
    for zone in zones:
        for model in models:
            for stamp in dates:
                is_first = stamp.tz_convert("Europe/Paris").date() == begin.date()
                rows.append({"zone": zone, "model_id": model, "timestamp_utc": stamp,
                             "forecast": 30., "actual": 10. if is_first else 20., "storm": 0.})
    return pd.DataFrame(rows)


def row(result, model="a", zone="FR"):
    return next(record for record in result["rows"] if record["zone"] == zone and record["model_id"] == model)


def coverage(result, zone="FR"):
    return next(record for record in result["coverage"] if record["zone"] == zone)


def test_existing_policy_net_gain_and_common_potential_denominator():
    result = compute_economic_kpis(panel(), end_day="2026-09-14", days=1)
    current = row(result)
    assert current["pnl_net_eur"] == 24*25*(10-1)
    assert current["storm_pnl_net_eur"] == 24*25*(-10-1)
    assert current["gain_vs_storm_eur"] == 12000.
    assert current["potential_energy_mwh"] == 600.
    assert current["gain_vs_storm_per_potential_mwh"] == 20.
    assert current["buy_intervals"] == 24
    assert current["sell_intervals"] == 0
    assert current["no_forecast_pnl_eur"] == 0.
    assert row(result, "__storm__")["gain_vs_storm_eur"] == 0.
    assert result["audit"]["signal_hurdle_eur_mwh"] == 6.
    assert result["audit"]["net_cost_eur_mwh"] == 1.
    assert result["audit"]["source_config_sha256"]
    json.dumps(result, allow_nan=False)


def test_fr_only_selection_never_receives_100_mw():
    result = compute_economic_kpis(panel(zones=("FR", "DE", "BE", "NL")), end_day="2026-09-14", days=1, zones=["FR"])
    assert row(result)["potential_energy_mwh"] == 24*25
    assert row(result, zone="ALL")["potential_energy_mwh"] == 24*25
    assert result["audit"]["portfolio_capacity_mw"] == 100.
    assert result["audit"]["selected_allocated_capacity_mw"] == 25.


def test_four_zones_share_total_capacity_100_mw():
    result = compute_economic_kpis(panel(zones=("FR", "DE", "BE", "NL")), end_day="2026-09-14", days=1)
    aggregate = row(result, zone="ALL")
    assert aggregate["potential_energy_mwh"] == 24*100.
    assert aggregate["n_hours"] == 24
    assert aggregate["n_country_hours"] == 96
    assert aggregate["pnl_net_eur"] == 24*100*9


def test_flat_at_exact_hurdle_and_fixed_denominator_when_no_trades():
    frame = panel(models=("a",))
    frame["forecast"] = 16.
    result = compute_economic_kpis(frame, end_day="2026-09-14", days=1)
    current = row(result)
    assert current["pnl_net_eur"] == 0.
    assert current["active_intervals"] == 0
    assert current["absolute_energy_mwh"] == 0.
    assert current["potential_energy_mwh"] == 600.
    assert current["gain_vs_storm_per_potential_mwh"] == 11.


def test_negative_positions_and_costs_on_absolute_energy():
    frame = panel(models=("a",))
    current = frame.timestamp_utc.ge(pd.Timestamp("2026-09-14", tz="Europe/Paris"))
    frame.loc[current, "actual"] = -10.
    frame["forecast"] = -20.
    frame["storm"] = 30.
    result = compute_economic_kpis(frame, end_day="2026-09-14", days=1)
    assert row(result)["sell_intervals"] == 24
    assert row(result)["pnl_net_eur"] == 24*25*19
    assert row(result)["storm_pnl_net_eur"] == 24*25*(-21)
    assert row(result)["gain_vs_storm_per_potential_mwh"] == 40.


def test_configuration_costs_used_without_parameter_fitting():
    config = deepcopy(load_config(DEFAULT_CONFIG))
    config["strategy"]["transaction_cost_eur_mwh"] = 2.
    config["strategy"]["slippage_eur_mwh"] = 1.
    result = compute_economic_kpis(panel(), end_day="2026-09-14", days=1, config=config)
    assert row(result)["pnl_net_eur"] == 24*25*7
    assert result["audit"]["signal_hurdle_eur_mwh"] == 8.
    assert result["audit"]["parameters_fitted_on_evaluation"] is False


def test_initial_day_has_no_reference_instead_of_same_day_leakage():
    result = compute_economic_kpis(panel(), end_day="2026-09-14", days=2)
    assert coverage(result)["price_common_hours"] == 48
    assert coverage(result)["missing_previous_civil_hour"] == 24
    assert row(result)["n_hours"] == 24


@pytest.mark.parametrize("start,end,hours,missing,ambiguous", [
    ("2026-03-28", "2026-03-29", 23, 0, 0),
    ("2026-03-29", "2026-03-30", 23, 1, 0),
    ("2025-10-25", "2025-10-26", 25, 0, 0),
    ("2025-10-26", "2025-10-27", 23, 0, 1),
])
def test_dst_reference_matches_previous_civil_hour_only(start, end, hours, missing, ambiguous):
    result = compute_economic_kpis(panel(start), end_day=end, days=1)
    assert row(result)["n_hours"] == hours
    assert coverage(result)["missing_previous_civil_hour"] == missing
    assert coverage(result)["ambiguous_previous_civil_hour"] == ambiguous


def test_common_model_support_is_symmetric_for_storm_and_all_models():
    frame = panel()
    current = frame.timestamp_utc.eq(pd.Timestamp("2026-09-14", tz="Europe/Paris")) & frame.model_id.eq("b")
    frame.loc[current, "forecast"] = np.nan
    result = compute_economic_kpis(frame, end_day="2026-09-14", days=1)
    assert row(result, "a")["n_hours"] == row(result, "b")["n_hours"] == row(result, "__storm__")["n_hours"] == 23


def test_reference_conflict_in_history_is_rejected():
    frame = panel()
    old = frame.timestamp_utc.lt(pd.Timestamp("2026-09-14", tz="Europe/Paris")) & frame.model_id.eq("b")
    frame.loc[old, "actual"] += 1.
    with pytest.raises(KPIError, match="Conflicting shared actual"):
        compute_economic_kpis(frame, end_day="2026-09-14", days=1)


def test_missing_reference_is_removed_identically_for_all_models():
    frame = panel()
    frame.loc[frame.timestamp_utc.eq(frame.timestamp_utc.min()), "actual"] = np.nan
    result = compute_economic_kpis(frame, end_day="2026-09-14", days=1)
    assert coverage(result)["price_common_hours"] == 24
    assert coverage(result)["reference_eligible_hours"] == 23
    assert all(row(result, model)["n_hours"] == 23 for model in ("a", "b", "__storm__"))


def test_late_forecast_origin_excludes_hour_for_every_alternative():
    frame = panel()
    frame["forecast_origin_utc"] = pd.Timestamp("2026-09-13T06:00Z")
    late = frame.model_id.eq("b") & frame.timestamp_utc.eq(pd.Timestamp("2026-09-14", tz="Europe/Paris"))
    frame.loc[late, "forecast_origin_utc"] = pd.Timestamp("2026-09-13T06:01Z")
    result = compute_economic_kpis(frame, end_day="2026-09-14", days=1)
    assert row(result)["n_hours"] == 23
    assert coverage(result)["execution_ineligible_hours"] == 1
    assert result["audit"]["forecast_origin_assumed_when_absent"] is False


def test_reference_not_yet_available_at_origin_abstains():
    frame = panel()
    frame["forecast_origin_utc"] = pd.Timestamp("2026-09-12T12:00Z")  # Earlier than D-2 18 civil.
    result = compute_economic_kpis(frame, end_day="2026-09-14", days=1)
    assert row(result)["n_hours"] == 0
    assert row(result)["pnl_net_eur"] is None
    assert coverage(result)["execution_ineligible_hours"] == 24


def test_explicit_forecast_or_storm_ineligibility_is_never_ignored():
    frame = panel()
    frame["forecast_eligible"] = True
    frame["benchmark_eligible"] = True
    frame.loc[frame.model_id.eq("b"), "benchmark_eligible"] = False
    result = compute_economic_kpis(frame, end_day="2026-09-14", days=1)
    assert row(result)["n_hours"] == 0
    assert row(result, "__storm__")["pnl_net_eur"] is None


def test_portfolio_requires_all_selected_countries_without_redistribution():
    frame = panel(zones=("FR", "DE"))
    absent = frame.zone.eq("DE") & frame.timestamp_utc.eq(pd.Timestamp("2026-09-14", tz="Europe/Paris"))
    frame.loc[absent, "forecast"] = np.nan
    result = compute_economic_kpis(frame, end_day="2026-09-14", days=1)
    assert row(result, zone="FR")["n_hours"] == 24
    assert row(result, zone="DE")["n_hours"] == 23
    assert row(result, zone="ALL")["n_hours"] == 23
    assert row(result, zone="ALL")["potential_energy_mwh"] == 23*50.


def test_empty_comparison_is_json_safe_without_zero_fictitious_pnl():
    result = compute_economic_kpis(panel().iloc[:0], end_day="2026-09-14", days=1, models=["a"], zones=["FR"])
    assert row(result)["pnl_net_eur"] is None
    assert row(result)["gain_vs_storm_eur"] is None
    json.dumps(result, allow_nan=False)


def test_missing_model_is_explicit_no_economic_support():
    result = compute_economic_kpis(panel(), end_day="2026-09-14", days=1, models=["a", "absent"])
    assert row(result)["n_hours"] == 0


def test_input_frame_and_existing_configuration_unchanged():
    frame = panel()
    before = frame.copy(deep=True)
    config_bytes = DEFAULT_CONFIG.read_bytes()
    compute_economic_kpis(frame, end_day="2026-09-14", days=1)
    pd.testing.assert_frame_equal(frame, before, check_exact=True)
    assert DEFAULT_CONFIG.read_bytes() == config_bytes


def test_model_not_known_until_after_delivery_never_receives_scored_signal():
    frame = panel()
    frame["forecast_origin_utc"] = frame.timestamp_utc+pd.Timedelta(hours=1)
    result = compute_economic_kpis(frame, end_day="2026-09-14", days=1)
    assert row(result)["pnl_net_eur"] is None
    assert row(result, "__storm__")["pnl_net_eur"] is None


def test_string_flags_are_rejected_instead_of_truthy():
    frame = panel()
    frame["forecast_eligible"] = "false"
    with pytest.raises(ValueError, match="booleens explicites"):
        compute_economic_kpis(frame, end_day="2026-09-14", days=1)


def test_configuration_mapping_with_wrong_declared_source_is_refused():
    config = deepcopy(load_config(DEFAULT_CONFIG))
    config["portfolio"]["capacity_mw"] = 200.
    with pytest.raises(KPIError, match="differ from their source"):
        compute_economic_kpis(panel(), end_day="2026-09-14", days=1, config=config, config_path=DEFAULT_CONFIG)


def test_reserved_selection_is_explicitly_refused():
    with pytest.raises(KPIError, match="Reserved"):
        compute_economic_kpis(panel(), end_day="2026-09-14", days=1, models=["a", "__storm__"])


def test_observed_live_rows_are_not_relabelled_as_evaluation():
    frame = panel()
    frame["sample"] = "evaluation"
    frame.loc[frame.model_id.eq("b"), "sample"] = "live"
    result = compute_economic_kpis(frame, end_day="2026-09-14", days=1)
    assert row(result)["n_hours"] == 0
    assert row(result, "__storm__")["pnl_net_eur"] is None
    assert result["audit"]["sample_assumed_evaluation_when_absent"] is False
