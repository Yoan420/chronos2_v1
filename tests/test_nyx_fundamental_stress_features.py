from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from nyx_fundamental_stress.features import CALENDAR_NAMES, PREFIX, SUPPLY_COMPONENTS, ZONES, make_fundamental_features


def panel(*, start="2026-01-01", days=2):
    first = pd.Timestamp(start).tz_localize("Europe/Paris")
    end = (pd.Timestamp(start)+pd.Timedelta(days=days)).tz_localize("Europe/Paris")
    timestamps = pd.date_range(first, end, freq="h", inclusive="left").tz_convert("UTC")
    rows = []
    for timestamp in timestamps:
        local = timestamp.tz_convert("Europe/Paris")
        civil = local.tz_localize(None).normalize()
        origin = (civil-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
        source = {
            "feature_fr_residual_load_gw": 30.+local.hour,
            "feature_de_residual_load_gw": 25.+local.hour*2,
            "feature_be_residual_load_gw": 7.+local.hour*.2,
            "feature_nl_residual_load_gw": 8.+local.hour*.3,
            "feature_fr_gas_available_gw": 10., "feature_de_gas_available_gw": 20.,
            "feature_be_gas_available_gw": 5., "feature_nl_gas_available_gw": 10.,
            "feature_fr_nuclear_generation_gw": 40., "feature_de_coal_available_gw": 15.,
            "feature_de_lignite_available_gw": 5., "feature_be_nuclear_available_gw": 2.,
            "feature_nl_coal_available_gw": 3., "feature_nl_nuclear_available_gw": 1.,
            "feature_clean_gas_cost_ccgt_proxy_eur_mwh": 80., "feature_clean_gas_cost_ocgt_proxy_eur_mwh": 120.,
            "feature_gas_merit_slope_proxy_eur_mwh": 40.,
        }
        for zone in ZONES:
            source[f"feature_{zone.lower()}_wind_generation_gw"] = 2.+local.hour*.1
            source[f"feature_{zone.lower()}_solar_generation_gw"] = max(0., 12.-abs(local.hour-12))
            source[f"feature_{zone.lower()}_temperature_c"] = 20.
        for zone in ZONES:
            rows.append({"zone": zone, "timestamp_utc": timestamp, "forecast_origin_utc": origin,
                         "forecast": 100., "actual": 200., "q10": 80., "q90": 120.,
                         "benchmark_forecast": 150., "feature_eligible": True, **source})
    return pd.DataFrame(rows)


def test_original_columns_order_types_and_duplicate_index_are_untouched():
    data = panel().iloc[::-1].copy()
    data.index = pd.Index([i//2 for i in range(len(data))], name="duplicated_user_index")
    data["actual"] = "not a model input"
    original = data.copy(deep=True)
    augmented, names, required, audit = make_fundamental_features(data)
    pd.testing.assert_frame_equal(augmented[original.columns], original, check_exact=True)
    pd.testing.assert_frame_equal(data, original, check_exact=True)
    assert all(name.startswith(PREFIX) for name in names)
    assert set(required) <= set(names)
    assert audit["electricity_price_forecast_quantile_actual_storm_or_lag_used"] is False
    assert audit["model_fit_performed"] is False and audit["source_reads_performed"] is False


@pytest.mark.parametrize("variant", ["fundamental", "calendar"])
def test_all_electricity_prices_and_oracle_columns_can_change_without_changing_features(variant):
    data = panel()
    before, names, required, audit = make_fundamental_features(data, variant)
    changed = deepcopy(data)
    for column in ("forecast", "actual", "q10", "q90", "benchmark_forecast",
                   "feature_baseline_forecast", "feature_baseline_interval_width", "feature_baseline_upper_distance",
                   "feature_cgc_minus_baseline_proxy_eur_mwh", "feature_past_price", "feature_actual_oracle", "feature_storm"):
        changed[column] = "poisoned price field, must not be parsed"
    changed["label_available_at_utc"] = "unknown future label"
    after, new_names, new_required, new_audit = make_fundamental_features(changed, variant)
    assert names == new_names and required == new_required
    pd.testing.assert_frame_equal(before[audit["all_constructed_feature_columns"]], after[new_audit["all_constructed_feature_columns"]])
    forbidden = {"forecast", "actual", "q10", "q90", "benchmark_forecast", "feature_cgc_minus_baseline_proxy_eur_mwh"}
    assert forbidden.isdisjoint(new_audit["input_source_columns_read"])
    assert not any(any(word in name.lower() for word in ("baseline", "nyx", "storm", "q10", "q90", "price")) for name in names)


def test_no_electricity_price_columns_are_required_to_build_inputs():
    data = panel().drop(columns=["forecast", "actual", "q10", "q90", "benchmark_forecast"])
    augmented, names, _, _ = make_fundamental_features(data)
    assert len(augmented) == len(data) and names


def test_calendar_selects_only_calendar_but_preserves_same_physical_augmented_columns():
    data = panel()
    fundamental, _, _, audit = make_fundamental_features(data)
    calendar, names, required, control = make_fundamental_features(data, "calendar")
    pd.testing.assert_frame_equal(fundamental, calendar)
    assert names == CALENDAR_NAMES and required == CALENDAR_NAMES[:3]
    assert len(required) == 3 and audit["all_constructed_feature_columns"] == control["all_constructed_feature_columns"]


def test_calendar_eligibility_is_not_blocked_by_missing_all_physical_sources():
    data = panel().drop(columns=[name for name in panel() if name.startswith("feature_")])
    data["feature_eligible"] = False
    calendar, names, required, audit = make_fundamental_features(data, "calendar")
    assert calendar[required].notna().all().all()
    assert calendar[PREFIX+"local_pressure"].isna().all()
    assert audit["eligible_rows"] == len(data)
    assert calendar.feature_eligible.eq(False).all()  # Consumer selects its own validated contract.
    _, _, _, physical_audit = make_fundamental_features(data, "fundamental")
    assert physical_audit["eligible_rows"] == 0


def test_pmax_components_and_fr_nuclear_generation_are_mapped_without_double_counting():
    augmented, _, _, audit = make_fundamental_features(panel())
    rows = augmented.groupby("zone").first()
    selected = {"FR": 50., "DE": 40., "BE": 7., "NL": 14.}
    for zone in ZONES:
        assert rows.loc[zone, PREFIX+"local_selected_supply_proxy_gw"] == selected[zone]
        assert rows.loc[zone, PREFIX+"peer_selected_supply_proxy_gw"] == sum(selected.values())-selected[zone]
        assert audit["selected_supply_components"][zone] == list(SUPPLY_COMPONENTS[zone])
    assert "generation forecast" in audit["supply_semantics"] and "not full reserve margin" in audit["supply_semantics"]
    assert not any("ccgt" in name or "type.gt" in name for name in audit["input_source_columns_read"] if "available_gw" in name)


def test_local_peer_pressure_and_neighbor_stress_count_have_explicit_physical_meaning():
    augmented, _, _, _ = make_fundamental_features(panel())
    hour = augmented.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour
    current = augmented.loc[hour.eq(19)].groupby("zone").first()
    residual = {"FR": 49., "DE": 63., "BE": 10.8, "NL": 13.7}
    supply = {"FR": 50., "DE": 40., "BE": 7., "NL": 14.}
    for zone in ZONES:
        assert current.loc[zone, PREFIX+"local_pressure"] == pytest.approx(residual[zone]/supply[zone])
        assert current.loc[zone, PREFIX+"peer_pressure"] == pytest.approx((sum(residual.values())-residual[zone])/(sum(supply.values())-supply[zone]))
        assert current.loc[zone, PREFIX+"peer_stressed_count"] == sum(residual[other] > supply[other] for other in ZONES if other != zone)


def test_missing_nuclear_propagates_nan_even_when_old_aggregate_is_present():
    data = panel()
    data["feature_fr_nuclear_generation_gw"] = np.nan
    data["feature_local_selected_supply_proxy_gw"] = data.zone.map({"FR": 50., "DE": 40., "BE": 7., "NL": 14.})
    augmented, _, _, audit = make_fundamental_features(data)
    assert augmented.loc[augmented.zone.eq("FR"), PREFIX+"local_pressure"].isna().all()
    assert augmented.loc[~augmented.zone.eq("FR"), PREFIX+"peer_pressure"].isna().all()
    assert augmented.loc[~augmented.zone.eq("FR"), PREFIX+"peer_stressed_count"].isna().all()
    assert audit["eligible_rows"] == 0


def test_local_and_peer_ramps_use_exact_physical_lags_not_row_order():
    data = panel().sample(frac=1., random_state=17)
    augmented, _, _, _ = make_fundamental_features(data)
    hour = augmented.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour
    fr = augmented.loc[augmented.zone.eq("FR") & hour.ge(3)]
    np.testing.assert_allclose(fr[PREFIX+"local_residual_ramp_3h_gw_per_hour"], 1.)
    np.testing.assert_allclose(fr[PREFIX+"peer_residual_ramp_3h_gw_per_hour"], 2.5)
    np.testing.assert_allclose(fr[PREFIX+"local_wind_ramp_1h_gw_per_hour"], .1)
    np.testing.assert_allclose(fr[PREFIX+"peer_wind_ramp_1h_gw_per_hour"], .3)


def test_day_boundary_and_absent_lag_are_not_interpolated_or_taken_from_another_origin():
    data = panel()
    timestamp = pd.Timestamp("2026-01-01 16:00", tz="Europe/Paris").tz_convert("UTC")
    data = data.loc[~(data.zone.eq("FR") & data.timestamp_utc.eq(timestamp))]
    # An old cached ramp with misleading first-day values must not be reused.
    data["feature_local_residual_ramp_3h_gw_per_hour"] = 999.
    augmented, _, _, _ = make_fundamental_features(data)
    local = augmented.timestamp_utc.dt.tz_convert("Europe/Paris")
    early = local.dt.hour.lt(3)
    assert augmented.loc[early, PREFIX+"local_residual_ramp_3h_gw_per_hour"].isna().all()
    target = augmented.zone.eq("FR") & augmented.timestamp_utc.eq(timestamp+pd.Timedelta(hours=3))
    assert augmented.loc[target, PREFIX+"local_residual_ramp_3h_gw_per_hour"].isna().all()
    assert augmented.loc[target, PREFIX+"peer_residual_ramp_3h_gw_per_hour"].isna().all()


def test_future_physical_profiles_do_not_change_past_features():
    data = panel()
    original, names, _, _ = make_fundamental_features(data)
    later = data.timestamp_utc.ge(pd.Timestamp("2026-01-02", tz="Europe/Paris"))
    changed = deepcopy(data)
    changed.loc[later, "feature_de_residual_load_gw"] = 2000.
    changed.loc[later, "feature_fr_nuclear_generation_gw"] = 0.
    changed.loc[later, "actual"] = -1e10
    updated, _, _, _ = make_fundamental_features(changed)
    pd.testing.assert_frame_equal(original.loc[~later, names], updated.loc[~later, names])


def test_dst_repeated_hour_preserved_with_physical_lags_and_civil_calendar():
    data = panel(start="2025-10-26", days=1)
    augmented, _, _, _ = make_fundamental_features(data)
    fr = augmented.loc[augmented.zone.eq("FR")]
    assert len(fr) == 25 and fr.timestamp_utc.nunique() == 25
    hour = fr.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour
    repeated = fr.loc[hour.eq(2)]
    assert len(repeated) == 2
    np.testing.assert_allclose(repeated[PREFIX+"hour_sin"], .5)
    assert repeated[PREFIX+"local_residual_ramp_1h_gw_per_hour"].tolist() == [1., 0.]


def test_optional_temperature_and_revision_gaps_are_preserved():
    data = panel()
    data[[name for name in data if name.endswith("temperature_c")]] = np.nan
    data["feature_fr_temperature_c_revision_delta"] = np.nan
    augmented, names, required, audit = make_fundamental_features(data)
    assert augmented[PREFIX+"local_temperature_c"].isna().all()
    assert augmented[PREFIX+"local_temperature_c__missing"].eq(1.).all()
    assert augmented.feature_fr_temperature_c_revision_delta.isna().all()
    assert not any("revision" in name for name in names)
    assert augmented[required].notna().all().all()
    assert audit["forecast_revision_features_used"] is False


def test_clean_gas_costs_are_allowed_but_never_subtracted_from_nyx():
    augmented, names, _, audit = make_fundamental_features(panel())
    assert augmented[PREFIX+"clean_gas_cost_ccgt_proxy_eur_mwh"].eq(80.).all()
    assert augmented[PREFIX+"gas_merit_slope_proxy_eur_mwh"].eq(40.).all()
    assert not any("minus_baseline" in name for name in names)
    assert "feature_clean_gas_cost_ccgt_proxy_eur_mwh" in audit["input_source_columns_read"]


@pytest.mark.parametrize("kind", ["origin", "naive", "duplicate", "zone", "reserved", "variant", "negative_capacity", "infinite_capacity"])
def test_invalid_contract_rejected(kind):
    data = panel()
    variant = "fundamental"
    if kind == "origin":
        data["forecast_origin_utc"] += pd.Timedelta(hours=1)
    elif kind == "naive":
        data["timestamp_utc"] = data.timestamp_utc.dt.tz_localize(None)
    elif kind == "duplicate":
        data = pd.concat([data, data.iloc[:1]])
    elif kind == "zone":
        data["zone"] = "ES"
    elif kind == "reserved":
        data[PREFIX+"local_pressure"] = 1.
    elif kind == "variant":
        variant = "price"
    else:
        data["feature_fr_nuclear_generation_gw"] = -1. if kind == "negative_capacity" else np.inf
    with pytest.raises(ValueError):
        make_fundamental_features(data, variant)


def test_nonpositive_residuals_and_zero_valid_supply_are_not_replaced_with_fake_capacity():
    data = panel()
    data["feature_fr_residual_load_gw"] = -10.
    data[["feature_fr_gas_available_gw", "feature_fr_nuclear_generation_gw"]] = 0.
    augmented, _, _, audit = make_fundamental_features(data)
    fr = augmented.zone.eq("FR")
    assert augmented.loc[fr, PREFIX+"local_selected_supply_proxy_gw"].eq(0.).all()
    assert augmented.loc[fr, PREFIX+"local_residual_load_gw"].eq(-10.).all()
    assert augmented.loc[fr, PREFIX+"local_pressure"].eq(-10.).all()
    assert audit["ratio_denominator_floor_gw"] == 1.
