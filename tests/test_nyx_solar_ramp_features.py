import numpy as np
import pandas as pd
import pytest

from nyx_solar_ramp.features import PREFIX, SUPPLY_COMPONENTS, ZONES, build_features


def panel(start="2026-01-01", days=2):
    first = pd.Timestamp(start).tz_localize("Europe/Paris")
    end = (pd.Timestamp(start)+pd.Timedelta(days=days)).tz_localize("Europe/Paris")
    rows = []
    for timestamp in pd.date_range(first, end, freq="h", inclusive="left").tz_convert("UTC"):
        local = timestamp.tz_convert("Europe/Paris")
        origin = (local.tz_localize(None).normalize()-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
        source = {}
        for n, zone in enumerate(ZONES, start=1):
            for suffix, value in {
                "solar_generation_gw": n*max(0, 12-abs(local.hour-12)),
                "wind_generation_gw": n+local.hour*.1,
                "residual_load_gw": n*10+local.hour,
                "temperature_c": 15+n, "gas_available_gw": n*5,
            }.items():
                source[f"feature_{zone.lower()}_{suffix}"] = float(value)
            for component in SUPPLY_COMPONENTS[zone]:
                source.setdefault(component, 5.)
        source.update(feature_clean_gas_cost_ccgt_proxy_eur_mwh=80.,
                      feature_clean_gas_cost_ocgt_proxy_eur_mwh=120.,
                      feature_baseline_forecast=100., feature_baseline_interval_width=30.)
        for zone in ZONES:
            rows.append({"zone": zone, "timestamp_utc": timestamp, "forecast_origin_utc": origin,
                         "actual": "forbidden label", "forecast": "forbidden unqualified price", **source})
    return pd.DataFrame(rows)


def feature_names(groups):
    return [name for names in groups.values() for name in names]


def test_contract_disjoint_groups_gate_and_input_unchanged():
    data = panel().sample(frac=1, random_state=9)
    data.index = pd.Index([i//2 for i in range(len(data))], name="duplicate_user_index")
    original = data.copy(deep=True)
    result, groups, audit = build_features(data)
    pd.testing.assert_frame_equal(data, original, check_exact=True)
    pd.testing.assert_frame_equal(result[original.columns], original, check_exact=True)
    assert list(groups) == ["calendar", "controls", "baseline", "local_solar", "regional_solar", "ramps", "interactions"]
    names = feature_names(groups)
    assert len(names) == len(set(names)) == 44
    assert len(groups["controls"]) == 24 and len(groups["ramps"]) == 4
    assert all("solar_drop" in name for name in groups["ramps"])
    assert sum("_ramp_" in name for name in groups["controls"]) == 8
    assert groups["baseline"] == [] and "solarx_eligible" not in names
    assert result.solarx_eligible.all() and audit["eligible_rows"] == len(data)
    assert not audit["labels_actual_storm_or_realised_power_used"]


def test_labels_prices_and_future_oracles_never_change_physical_inputs():
    data = panel()
    before, groups, audit = build_features(data)
    for column in ("actual", "forecast", "q10", "q90", "benchmark_forecast", "feature_storm",
                   "feature_actual_oracle", "label_available_at_utc", "feature_past_price",
                   "feature_baseline_forecast", "feature_baseline_interval_width"):
        data[column] = "poison, must not parse"
    after, next_groups, next_audit = build_features(data)
    assert next_groups == groups
    pd.testing.assert_frame_equal(before[feature_names(groups)], after[feature_names(groups)])
    assert audit["input_source_columns_read"] == next_audit["input_source_columns_read"]
    assert "feature_baseline_forecast" not in next_audit["input_source_columns_read"]


def test_baseline_requires_opt_in_and_is_kept_separate():
    data = panel()
    result, groups, audit = build_features(data, include_baseline=True)
    assert groups["baseline"] == ["solarx_baseline_forecast", "solarx_baseline_interval_width"]
    assert result.solarx_baseline_forecast.eq(100.).all()
    assert not set(groups["baseline"]) & set(groups["controls"])
    assert audit["baseline_opt_in"] and len(feature_names(groups)) == 46
    data["actual"] = np.inf
    same, _, _ = build_features(data, include_baseline=True)
    pd.testing.assert_frame_equal(result[feature_names(groups)], same[feature_names(groups)])


def test_solar_local_peer_and_residual_not_double_subtracted():
    result, _, audit = build_features(panel())
    at = result.loc[result.zone.eq("DE") & result.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour.eq(17)].iloc[0]
    assert at.solarx_local_solar_gw == 14.
    assert at.solarx_peer_solar_gw == 56.
    assert at.solarx_local_residual_gw == 37.
    assert at.solarx_peer_residual_gw == 131.
    assert at.solarx_local_solar_drop_1h_gwph == 2.
    assert at.solarx_peer_solar_drop_1h_gwph == 8.
    assert at.solarx_local_pressure_proxy == pytest.approx(37/20)
    assert at.solarx_local_drop_x_residual_gw2ph == 74.
    assert "never subtracted again" in audit["renewable_accounting"]


def test_missing_peer_solar_and_residual_disable_gate_without_partial_sums():
    data = panel()
    data["feature_fr_solar_generation_gw"] = np.nan
    result, _, audit = build_features(data)
    assert not result.solarx_eligible.any()
    assert result.loc[result.zone.eq("FR"), "solarx_local_solar_gw"].isna().all()
    assert result.loc[~result.zone.eq("FR"), "solarx_peer_solar_gw"].isna().all()
    assert audit["missing_required_rows"] == len(result)
    data = panel().drop(columns="feature_nl_residual_load_gw")
    assert not build_features(data)[0].solarx_eligible.any()


def test_optional_supply_weather_gaps_propagate_but_do_not_disable_solar_gate():
    data = panel().drop(columns=["feature_fr_nuclear_generation_gw", "feature_be_temperature_c"])
    result, _, _ = build_features(data)
    assert result.solarx_eligible.all()
    assert result.loc[result.zone.eq("FR"), "solarx_local_pressure_proxy"].isna().all()
    assert result.loc[~result.zone.eq("FR"), "solarx_peer_pressure_proxy"].isna().all()
    assert result.loc[~result.zone.eq("BE"), "solarx_peer_temperature_mean_c"].isna().all()


def test_exact_ramps_are_order_invariant_and_do_not_cross_origins_or_missing_hours():
    data = panel()
    full, groups, _ = build_features(data)
    shuffled, _, _ = build_features(data.sample(frac=1, random_state=23))
    pd.testing.assert_frame_equal(full[feature_names(groups)], shuffled.sort_index()[feature_names(groups)])
    early = full.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour.eq(0)
    assert full.loc[early, "solarx_local_solar_drop_1h_gwph"].isna().all()
    timestamp = pd.Timestamp("2026-01-01T15:00Z")
    data = data.loc[~(data.zone.eq("DE") & data.timestamp_utc.eq(timestamp))]
    result, _, _ = build_features(data)
    target = result.zone.eq("DE") & result.timestamp_utc.eq(timestamp+pd.Timedelta(hours=1))
    assert result.loc[target, "solarx_local_solar_drop_1h_gwph"].isna().all()
    assert result.loc[target, "solarx_peer_solar_drop_1h_gwph"].isna().all()


@pytest.mark.parametrize("day,hours", [("2025-10-26", 25), ("2026-03-29", 23)])
def test_dst_physical_hours_and_civil_origin(day, hours):
    result, _, _ = build_features(panel(day, 1))
    de = result.loc[result.zone.eq("DE")]
    assert len(de) == hours and de.timestamp_utc.nunique() == hours
    assert de.forecast_origin_utc.nunique() == 1
    assert de.forecast_origin_utc.dt.tz_convert("Europe/Paris").dt.hour.eq(8).all()
    two = de.loc[de.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour.eq(2)]
    if hours == 25:
        assert len(two) == 2
        assert two.solarx_local_residual_ramp_1h_gwph.tolist() == [1., 0.]
    else:
        assert two.empty


def test_later_delivery_profiles_do_not_change_earlier_rows():
    data = panel()
    before, groups, _ = build_features(data)
    later = data.timestamp_utc.ge(pd.Timestamp("2026-01-01T23:00Z"))
    data.loc[later, "feature_de_solar_generation_gw"] = 500.
    data.loc[later, "feature_de_residual_load_gw"] = 800.
    after, _, _ = build_features(data)
    pd.testing.assert_frame_equal(before.loc[~later, feature_names(groups)], after.loc[~later, feature_names(groups)])


def test_saved_feature_availability_is_checked_against_each_origin():
    data = panel()
    data["feature_available_at_utc"] = data.forecast_origin_utc
    data.loc[0, "feature_available_at_utc"] += pd.Timedelta(seconds=1)
    data.loc[1, "feature_available_at_utc"] = pd.NaT
    result, _, audit = build_features(data)
    assert not result.loc[:1, "solarx_eligible"].any()
    assert result.loc[2:, "solarx_eligible"].all() and audit["unavailable_rows"] == 2


@pytest.mark.parametrize("failure", ["origin", "naive", "duplicate", "half_hour", "zone", "reserved", "infinite", "negative"])
def test_invalid_contract_fails_closed(failure):
    data = panel()
    if failure == "origin":
        data.loc[0, "forecast_origin_utc"] += pd.Timedelta(hours=1)
    elif failure == "naive":
        data["timestamp_utc"] = data.timestamp_utc.dt.tz_localize(None)
    elif failure == "duplicate":
        data = pd.concat([data, data.iloc[[0]]])
    elif failure == "half_hour":
        data.loc[0, "timestamp_utc"] += pd.Timedelta(minutes=30)
    elif failure == "zone":
        data.loc[0, "zone"] = "ES"
    elif failure == "reserved":
        data[PREFIX+"existing"] = 1.
    else:
        data.loc[0, "feature_de_solar_generation_gw"] = np.inf if failure == "infinite" else -1.
    with pytest.raises(ValueError):
        build_features(data)
