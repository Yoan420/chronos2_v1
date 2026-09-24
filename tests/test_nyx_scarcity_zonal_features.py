from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from nyx_scarcity_zonal.features import PREFIX, ZONES, make_zonal_features


def panel():
    rows = []
    raw = {
        "fr_residual_load_gw": 30., "de_residual_load_gw": 24., "be_residual_load_gw": 8., "nl_residual_load_gw": 7.,
        "fr_gas_available_gw": 10., "de_gas_available_gw": 20., "be_gas_available_gw": 5., "nl_gas_available_gw": 10.,
        "fr_nuclear_generation_gw": 40., "de_coal_available_gw": 15., "de_lignite_available_gw": 5.,
        "be_nuclear_available_gw": 2., "nl_coal_available_gw": 3., "nl_nuclear_available_gw": 1.,
        "clean_gas_cost_ccgt_proxy_eur_mwh": 80., "clean_gas_cost_ocgt_proxy_eur_mwh": 120.,
    }
    raw.update({f"{zone.lower()}_{kind}_generation_gw": 2. if kind == "wind" else 1. for zone in ZONES for kind in ("wind", "solar")})
    raw.update({f"{zone.lower()}_temperature_c": 20. for zone in ZONES})
    for hour in (16, 17):
        for zone, price in zip(ZONES, (100., 200., 300., 400.)):
            rows.append({"zone": zone, "timestamp_utc": pd.Timestamp(f"2026-09-13 {hour}:00", tz="UTC"),
                         "forecast_origin_utc": pd.Timestamp("2026-09-12 06:00", tz="UTC"),
                         "forecast": price, "q10": price-10, "q90": price+20,
                         "actual": 999., "benchmark_forecast": -100.,
                         **{"feature_"+key: value for key, value in raw.items()}})
    return pd.DataFrame(rows, index=pd.Index([10, 10, 12, 13, 14, 15, 16, 17], name="user_index"))


def test_exact_original_values_order_duplicate_index_and_compact_allowlist_preserved():
    data = panel().iloc[::-1]
    data["forecast"] = data.forecast.map(str)
    original = data.copy(deep=True)
    augmented, features, required, audit = make_zonal_features(data)
    pd.testing.assert_frame_equal(data, original)
    pd.testing.assert_frame_equal(augmented[original.columns], original)
    assert all(name.startswith(PREFIX) for name in features)
    assert len(features) < 100 and set(required) <= set(features)
    assert not set(features).intersection(original.columns)
    assert not any(name in features for name in ("actual", "benchmark_forecast", "feature_fr_residual_load_gw"))
    assert audit["observed_price_or_storm_used"] is False
    assert audit["fit_performed"] is False and audit["sources_read"] is False


def test_selected_supply_maps_generation_versus_pmax_and_peer_exclusion_correctly():
    augmented, _, _, audit = make_zonal_features(panel())
    first = augmented.groupby("zone").first()
    supply = {"FR": 50., "DE": 40., "BE": 7., "NL": 14.}
    residual = {"FR": 30., "DE": 24., "BE": 8., "NL": 7.}
    for zone in ZONES:
        assert first.loc[zone, PREFIX+"local_selected_supply_proxy_gw"] == supply[zone]
        assert first.loc[zone, PREFIX+"peer_selected_supply_proxy_gw"] == sum(supply.values())-supply[zone]
        assert first.loc[zone, PREFIX+"peer_residual_load_gw"] == sum(residual.values())-residual[zone]
        assert first.loc[zone, PREFIX+"local_residual_to_selected_supply_proxy"] == residual[zone]/supply[zone]
    assert first.loc["FR", PREFIX+"local_residual_to_selected_supply_proxy"] == .6
    assert "generation" in audit["supply_semantics"] and "neither full reserve" in audit["supply_semantics"]


def test_peer_price_uses_exactly_three_other_country_forecasts_at_same_physical_hour():
    augmented, _, _, _ = make_zonal_features(panel())
    fr = augmented.loc[augmented.zone.eq("FR")]
    assert fr[PREFIX+"peer_baseline_mean_forecast"].eq(300.).all()
    assert fr[PREFIX+"local_minus_peer_baseline_forecast"].eq(-200.).all()
    be = augmented.loc[augmented.zone.eq("BE")]
    np.testing.assert_allclose(be[PREFIX+"peer_baseline_mean_forecast"], 700/3)


def test_absent_country_leaves_peer_prices_unknown_not_two_country_average():
    data = panel()
    data = data.loc[~(data.zone.eq("NL") & data.timestamp_utc.eq(data.timestamp_utc.iloc[0]))]
    augmented, _, _, _ = make_zonal_features(data)
    first = augmented.timestamp_utc.eq(augmented.timestamp_utc.min())
    assert augmented.loc[first, PREFIX+"peer_baseline_mean_forecast"].isna().all()
    assert augmented.loc[first, PREFIX+"peer_baseline_mean_forecast__missing"].eq(1).all()
    assert augmented.loc[~first, PREFIX+"peer_baseline_mean_forecast"].notna().all()


def test_missing_nuclear_component_never_zero_imputed_or_hidden_by_cached_total():
    data = panel()
    data["feature_fr_nuclear_generation_gw"] = np.nan
    data["feature_local_selected_supply_proxy_gw"] = data.zone.map({"FR": 50., "DE": 40., "BE": 7., "NL": 14.})
    augmented, _, required, audit = make_zonal_features(data)
    assert augmented.loc[augmented.zone.eq("FR"), PREFIX+"local_selected_supply_proxy_gw"].isna().all()
    assert augmented.loc[~augmented.zone.eq("FR"), PREFIX+"peer_selected_supply_proxy_gw"].isna().all()
    assert not augmented[required].notna().all(axis=1).any()
    assert audit["eligible_rows"] == 0


def test_optional_missing_temperature_ramps_and_fuels_have_flags_without_fabrication():
    data = panel().drop(columns=[c for c in panel() if c.endswith("temperature_c") or "clean_gas_cost" in c])
    augmented, _, required, audit = make_zonal_features(data)
    for name in ("local_temperature_c", "local_residual_ramp_3h_to_selected_supply_proxy_per_hour", "clean_gas_cost_ccgt_proxy_eur_mwh"):
        assert augmented[PREFIX+name].isna().all()
        assert augmented[PREFIX+name+"__missing"].eq(1.).all()
    assert augmented[required].notna().all(axis=1).all()
    assert audit["eligible_rows"] == len(data)


def test_saved_ramps_scaled_only_by_selected_supply_and_no_cross_day_reconstruction():
    data = panel()
    data["feature_local_residual_ramp_1h_gw_per_hour"] = 5.
    data["feature_local_solar_ramp_3h_gw_per_hour"] = -3.
    augmented, _, _, _ = make_zonal_features(data)
    fr = augmented.zone.eq("FR")
    np.testing.assert_allclose(augmented.loc[fr, PREFIX+"local_residual_ramp_1h_to_selected_supply_proxy_per_hour"], .1)
    np.testing.assert_allclose(augmented.loc[fr, PREFIX+"local_solar_ramp_3h_to_selected_supply_proxy_per_hour"], -.06)


@pytest.mark.parametrize("column", ["feature_local_residual_load_gw", "feature_local_selected_supply_proxy_gw",
                                    "feature_local_selected_supply_minus_residual_proxy_gw", "feature_baseline_forecast",
                                    "feature_baseline_interval_width", "feature_gas_merit_slope_proxy_eur_mwh"])
def test_inconsistent_cached_arithmetic_is_rejected(column):
    data = panel()
    data[column] = -1234.
    with pytest.raises(ValueError, match="disagrees"):
        make_zonal_features(data)


def test_realized_prices_storm_and_future_labels_have_no_effect_on_features():
    original = panel()
    before, features, _, _ = make_zonal_features(original)
    changed = deepcopy(original)
    changed["actual"] = "not even a numeric input"
    changed["benchmark_forecast"] = np.inf
    changed["label_available_at_utc"] = "future metadata is not a feature"
    changed["feature_actual_oracle"] = 1e9
    after, second, _, _ = make_zonal_features(changed)
    assert features == second
    pd.testing.assert_frame_equal(before[features], after[features])


def test_dst_repeated_hour_uses_distinct_peer_prices_and_correct_origins():
    data = panel()
    stamp = [pd.Timestamp("2025-10-26 00:00", tz="UTC")]*4+[pd.Timestamp("2025-10-26 01:00", tz="UTC")]*4
    data["timestamp_utc"] = stamp
    data["forecast_origin_utc"] = pd.Timestamp("2025-10-25 06:00", tz="UTC")
    data.iloc[4:, data.columns.get_loc("forecast")] += 40
    augmented, _, _, _ = make_zonal_features(data)
    fr = augmented.loc[augmented.zone.eq("FR")]
    assert fr[PREFIX+"peer_baseline_mean_forecast"].tolist() == [300., 340.]
    np.testing.assert_allclose(fr[PREFIX+"hour_sin"], .5)


@pytest.mark.parametrize("kind", ["origin", "naive", "duplicate", "nonhourly", "unknown_zone", "reserved"])
def test_invalid_identity_or_overwrite_attempt_fails(kind):
    data = panel()
    if kind == "origin":
        data["forecast_origin_utc"] += pd.Timedelta(hours=1)
    elif kind == "naive":
        data["timestamp_utc"] = data.timestamp_utc.dt.tz_localize(None)
    elif kind == "duplicate":
        data = pd.concat([data, data.iloc[:1]])
    elif kind == "nonhourly":
        data["timestamp_utc"] += pd.Timedelta(minutes=1)
    elif kind == "unknown_zone":
        data["zone"] = "ES"
    else:
        data[PREFIX+"existing"] = 1.
    with pytest.raises(ValueError):
        make_zonal_features(data)


def test_zero_selected_supply_stays_finite_and_is_explicit_not_imputed():
    data = panel()
    data[["feature_fr_gas_available_gw", "feature_fr_nuclear_generation_gw"]] = 0.
    augmented, _, _, audit = make_zonal_features(data)
    fr = augmented.zone.eq("FR")
    assert augmented.loc[fr, PREFIX+"local_selected_supply_proxy_gw"].eq(0.).all()
    assert augmented.loc[fr, PREFIX+"local_residual_to_selected_supply_proxy"].eq(30.).all()
    assert audit["ratio_denominator_floor_gw"] == 1.


@pytest.mark.parametrize("value", [-1., np.inf])
def test_invalid_capacity_is_rejected(value):
    data = panel()
    data["feature_fr_nuclear_generation_gw"] = value
    with pytest.raises(ValueError):
        make_zonal_features(data)
