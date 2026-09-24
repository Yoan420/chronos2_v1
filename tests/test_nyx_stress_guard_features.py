from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from nyx_stress_guard.features import PREFIX, REFERENCE_SOURCES, make_stress_features, fit_reference, apply_reference


def panel(start="2026-01-01", days=1):
    first = pd.Timestamp(start)
    times = pd.date_range(first.tz_localize("Europe/Paris"),
                          (first+pd.Timedelta(days=days)).tz_localize("Europe/Paris"),
                          inclusive="left", freq="h").tz_convert("UTC")
    blocks = []
    for zone in ("FR", "DE", "BE", "NL"):
        b = pd.DataFrame({"zone": zone, "timestamp_utc": times})
        local = b.timestamp_utc.dt.tz_convert("Europe/Paris")
        civil = local.dt.tz_localize(None).dt.normalize()
        b["forecast_origin_utc"] = (civil-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).dt.tz_localize("Europe/Paris").dt.tz_convert("UTC")
        for z, scale in (("fr", 1.), ("de", 2.), ("be", .2), ("nl", .3)):
            b[f"feature_{z}_residual_load_gw"] = 10.+scale*local.dt.hour + (civil-first).dt.days*.01
            b[f"feature_{z}_wind_generation_gw"] = 5.-local.dt.hour*.1
            b[f"feature_{z}_solar_generation_gw"] = np.maximum(0., 12.-np.abs(local.dt.hour-12))
            b[f"feature_{z}_gas_available_gw"] = 10.
            b[f"feature_{z}_temperature_c"] = 20.
        b["feature_fr_nuclear_generation_gw"] = 40.
        b["feature_de_coal_available_gw"] = 15.
        b["feature_de_lignite_available_gw"] = 5.
        b["feature_nl_coal_available_gw"] = 3.
        b["feature_be_nuclear_available_gw"] = 2.
        b["feature_nl_nuclear_available_gw"] = 1.
        b["feature_clean_gas_cost_ccgt_proxy_eur_mwh"] = 80.
        b["feature_clean_gas_cost_ocgt_proxy_eur_mwh"] = 120.
        b["feature_gas_merit_slope_proxy_eur_mwh"] = 40.
        b["actual"], b["forecast"], b["benchmark_forecast"] = 200., 100., 150.
        b["q10"], b["q90"], b["feature_eligible"] = 80., 120., True
        blocks.append(b)
    return pd.concat(blocks, ignore_index=True)


def rank_frame(start="2026-01-01", days=35):
    """No expensive feature preparation is needed to test rank-only helpers."""
    f = panel(start, days)
    for i, name in enumerate(REFERENCE_SOURCES.values()):
        f[name] = f.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour.astype(float)+i
    for name in REFERENCE_SOURCES:
        f[name] = np.nan
        f[name+"__missing"] = 1.
    return f


def test_schema_reuses_old_bank_without_duplicates_and_reserves_reference_ranks():
    f, names, required, a = make_stress_features(panel())
    assert len(names) == len(set(names))
    assert set(required) <= set(names)
    assert set(REFERENCE_SOURCES).issubset(names)
    assert not set(REFERENCE_SOURCES).intersection(required)
    assert f[list(REFERENCE_SOURCES)].isna().all().all()
    assert f[[c+"__missing" for c in REFERENCE_SOURCES]].eq(1.).all().all()
    assert f[required].notna().all().all()
    assert not a["electricity_price_forecast_quantile_actual_storm_or_lag_used"]
    assert a["physical_gate_applied"] is False and a["network_inputs_used"] is False
    assert all(c.startswith("feature_fundamental_") for c in names)


def test_original_columns_and_indices_preserved_exactly():
    f = panel(days=2).sample(frac=1., random_state=7)
    f.index = pd.Index([i//2 for i in range(len(f))], name="duplicate")
    saved = f.copy(deep=True)
    result, _, _, _ = make_stress_features(f)
    pd.testing.assert_frame_equal(result[saved.columns], saved, check_exact=True)
    pd.testing.assert_frame_equal(f, saved, check_exact=True)


def test_poisoned_electricity_prices_labels_or_oracles_are_never_read():
    f = panel()
    before, names, _, _ = make_stress_features(f)
    for c in ("actual", "forecast", "q10", "q90", "benchmark_forecast", "label_available_at_utc",
              "feature_baseline_forecast", "feature_cgc_minus_baseline_proxy_eur_mwh", "_error", "feature_storm"):
        f[c] = "not numeric and forbidden"
    after, _, _, a = make_stress_features(f)
    pd.testing.assert_frame_equal(before[names], after[names], check_exact=True)
    assert {"actual", "forecast", "benchmark_forecast", "_error"}.isdisjoint(a["input_source_columns_read"])


def test_no_electricity_price_fields_are_required():
    f = panel().drop(columns=["actual", "forecast", "benchmark_forecast", "q10", "q90"])
    result, _, required, _ = make_stress_features(f)
    assert result[required].notna().all().all()


@pytest.mark.parametrize("hours", [1, 3])
def test_synchronous_ramps_and_renewable_direction_count_have_exact_formulas(hours):
    f, _, _, _ = make_stress_features(panel())
    row = f.loc[f.zone.eq("DE") & f.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour.eq(19)].iloc[0]
    assert row[PREFIX+f"regional_residual_ramp_{hours}h_gw_per_hour"] == pytest.approx(3.5)
    assert row[PREFIX+f"regional_wind_ramp_{hours}h_gw_per_hour"] == pytest.approx(-.4)
    assert row[PREFIX+f"regional_solar_ramp_{hours}h_gw_per_hour"] == pytest.approx(-4.)
    for kind in ("residual_rising_count", "wind_falling_count", "solar_falling_count", "residual_rise_and_renewable_fall_count"):
        assert row[PREFIX+f"{kind}_{hours}h"] == 4.
    assert row[PREFIX+f"regional_residual_ramp_{hours}h_normalized_per_hour"] == pytest.approx(3.5/106.)
    assert row[PREFIX+f"local_renewable_fall_ramp_{hours}h_normalized_per_hour"] == pytest.approx(1.1/30.)
    assert row[PREFIX+f"local_solar_fall_with_residual_rise_{hours}h"] == 1.
    # RL is untouched: it does not become RL - solar - wind.
    assert row["feature_fundamental_local_residual_load_gw"] == 48.


def test_daily_mid_rank_level_shape_uses_only_its_known_profile():
    f, _, _, _ = make_stress_features(panel())
    row = f.loc[f.zone.eq("FR") & f.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour.eq(19)].iloc[0]
    assert row[PREFIX+"local_residual_day_rank"] == pytest.approx(19.5/24)
    assert row[PREFIX+"local_residual_minus_day_median_gw"] == 7.5
    assert row[PREFIX+"local_residual_below_day_peak_gw"] == 4.


def test_constant_profile_has_mid_rank_half_and_zero_shape_distance():
    f = panel()
    for c in [c for c in f if c.endswith("solar_generation_gw")]:
        f[c] = 0.
    result, _, _, _ = make_stress_features(f)
    assert result[PREFIX+"local_solar_day_rank"].eq(.5).all()
    assert result[PREFIX+"peer_solar_below_day_peak_gw"].eq(0.).all()


@pytest.mark.parametrize("date,n", [("2026-03-29", 23), ("2025-10-26", 25), ("2026-02-01", 24)])
def test_dst_complete_profile_is_physical_not_fixed_twenty_four_rows(date, n):
    f, _, required, a = make_stress_features(panel(date))
    assert len(f) == n*4 and a["complete_profile_rows"] == n*4
    assert f[required].notna().all().all()
    assert f[PREFIX+"local_residual_day_rank"].between(0., 1.).all()


def test_missing_physical_hour_invalidates_daily_shapes_without_cross_day_repair():
    f = panel(days=2)
    target = pd.Timestamp("2026-01-01 16:00", tz="Europe/Paris").tz_convert("UTC")
    f = f.loc[~(f.zone.eq("FR") & f.timestamp_utc.eq(target))]
    result, _, required, _ = make_stress_features(f)
    local = result.timestamp_utc.dt.tz_convert("Europe/Paris")
    bad = result.zone.eq("FR") & local.dt.day.eq(1)
    assert result.loc[bad, PREFIX+"local_residual_day_rank"].isna().all()
    assert result.loc[bad, required].isna().any(axis=1).all()
    later = local.dt.day.eq(2)
    assert result.loc[later, required].notna().all().all()
    early = local.dt.hour.lt(3)
    assert result.loc[early, PREFIX+"regional_residual_ramp_3h_gw_per_hour"].isna().all()


def test_missing_raw_source_cannot_be_hidden_by_saved_cached_aggregate():
    f = panel()
    f["feature_local_wind_generation_gw"] = 99.
    f["feature_fr_wind_generation_gw"] = np.nan
    # Do not provide conflicting cached values for the other country views.
    for z in ("DE", "BE", "NL"):
        where = f.zone.eq(z)
        f.loc[where, "feature_local_wind_generation_gw"] = f.loc[where, f"feature_{z.lower()}_wind_generation_gw"]
    result, _, required, _ = make_stress_features(f)
    assert result[PREFIX+"complete_physical_profile"].isna().all()
    assert result[required].isna().any(axis=1).all()
    assert result[PREFIX+"wind_falling_count_1h"].isna().all()


def test_missing_optional_temperature_does_not_block_physical_profile():
    f = panel()
    f[[c for c in f if c.endswith("temperature_c")]] = np.nan
    result, _, required, _ = make_stress_features(f)
    assert result[required].notna().all().all()
    assert result["feature_fundamental_local_temperature_c"].isna().all()


def test_future_days_cannot_modify_earlier_profiles():
    f = panel(days=2)
    before, names, _, _ = make_stress_features(f)
    future = f.timestamp_utc.ge(pd.Timestamp("2026-01-02", tz="Europe/Paris"))
    f.loc[future, "feature_de_residual_load_gw"] = 999.
    after, _, _, _ = make_stress_features(f)
    pd.testing.assert_frame_equal(before.loc[~future, names], after.loc[~future, names], check_exact=True)


@pytest.mark.parametrize("bad", ["origin", "naive", "duplicate", "country", "infinite", "shared", "reserved"])
def test_invalid_feature_identity_or_physical_contract_fails(bad):
    f = panel()
    if bad == "origin":
        f.forecast_origin_utc += pd.Timedelta(hours=1)
    elif bad == "naive":
        f.timestamp_utc = f.timestamp_utc.dt.tz_localize(None)
    elif bad == "duplicate":
        f = pd.concat([f, f.iloc[:1]])
    elif bad == "country":
        f.loc[0, "zone"] = "ES"
    elif bad == "infinite":
        f["feature_fr_wind_generation_gw"] = np.inf
    elif bad == "shared":
        f.loc[0, "feature_de_wind_generation_gw"] += 1.
    else:
        f[PREFIX+"already_exists"] = 1.
    with pytest.raises(ValueError):
        make_stress_features(f)


def test_reference_is_country_hour_mid_ecdf_not_global_panel_rank():
    f = rank_frame()
    state = fit_reference(f, ["FR", "DE", "BE", "NL"], cutoff=pd.Timestamp("2026-02-06 08:00", tz="Europe/Paris"))
    out = apply_reference(f, state)
    assert out[list(REFERENCE_SOURCES)].eq(.5).all().all()
    assert out[[c+"__missing" for c in REFERENCE_SOURCES]].eq(0.).all().all()
    assert state["audit"]["support"][next(iter(REFERENCE_SOURCES))]["FR"]["available_hour_groups"] == list(range(24))
    assert state["audit"]["labels_or_electricity_prices_read"] is False


def test_reference_preserves_all_other_columns_and_never_uses_labels():
    f = rank_frame()
    for name in ("actual", "forecast", "benchmark_forecast", "_error", "label_available_at_utc"):
        f[name] = "not read"
    f.index = pd.Index([i//2 for i in range(len(f))], name="nonunique")
    saved = f.copy(deep=True)
    state = fit_reference(f, ["FR", "DE", "BE", "NL"], cutoff=pd.Timestamp("2026-02-06 08:00", tz="Europe/Paris"))
    out = apply_reference(f, state)
    reserved = list(REFERENCE_SOURCES)+[c+"__missing" for c in REFERENCE_SOURCES]
    pd.testing.assert_frame_equal(out.drop(columns=reserved), f.drop(columns=reserved), check_exact=True)
    pd.testing.assert_frame_equal(f, saved, check_exact=True)


def test_sparse_hour_reference_uses_country_pool_and_short_core_stays_nan():
    f = rank_frame(days=20)
    cutoff = pd.Timestamp("2026-01-22 08:00", tz="Europe/Paris")
    state = fit_reference(f, ["FR", "DE", "BE", "NL"], cutoff=cutoff)
    name = next(iter(REFERENCE_SOURCES))
    assert state["audit"]["support"][name]["FR"]["available_hour_groups"] == []
    out = apply_reference(f, state)
    expected = (f.timestamp_utc.dt.tz_convert("Europe/Paris").dt.hour+.5)/24.
    np.testing.assert_allclose(out[name], expected)
    short = rank_frame(days=19)
    shortstate = fit_reference(short, ["FR", "DE", "BE", "NL"], cutoff=cutoff)
    assert apply_reference(short, shortstate)[list(REFERENCE_SOURCES)].isna().all().all()


def test_reference_is_frozen_for_later_data_and_missing_remains_missing():
    f = rank_frame()
    state = fit_reference(f, ["FR", "DE", "BE", "NL"], cutoff=pd.Timestamp("2026-02-06 08:00", tz="Europe/Paris"))
    current = rank_frame("2026-02-10", 1)
    source = next(iter(REFERENCE_SOURCES.values()))
    current[source] = 100.
    current.loc[0, source] = np.nan
    result = apply_reference(current, state)
    name = next(iter(REFERENCE_SOURCES))
    assert result.loc[0, name+"__missing"] == 1.
    assert result.loc[1:, name].eq(1.).all()
    # Extreme CURRENT values have not altered stored empirical support.
    assert state["references"][name]["FR"]["pooled"]["values"].max() == 23.


def test_reference_separates_countries():
    f = rank_frame()
    source = next(iter(REFERENCE_SOURCES.values()))
    f.loc[f.zone.eq("FR"), source] += 100.
    state = fit_reference(f, ["FR", "DE", "BE", "NL"], cutoff=pd.Timestamp("2026-02-06 08:00", tz="Europe/Paris"))
    current = rank_frame("2026-02-10", 1)
    out = apply_reference(current, state)
    name = next(iter(REFERENCE_SOURCES))
    assert out.loc[out.zone.eq("FR"), name].eq(0.).all()
    assert out.loc[out.zone.ne("FR"), name].eq(.5).all()


@pytest.mark.parametrize("bad", ["naive_cutoff", "non08_cutoff", "late_core", "missing_source", "infinite", "unknown_zone", "duplicate_zone"])
def test_invalid_reference_training_contract_fails(bad):
    f = rank_frame()
    cutoff = pd.Timestamp("2026-02-06 08:00", tz="Europe/Paris")
    zones = ["FR", "DE", "BE", "NL"]
    if bad == "naive_cutoff":
        cutoff = cutoff.tz_localize(None)
    elif bad == "non08_cutoff":
        cutoff += pd.Timedelta(minutes=1)
    elif bad == "late_core":
        cutoff = pd.Timestamp("2026-01-05 08:00", tz="Europe/Paris")
    elif bad == "missing_source":
        f = f.drop(columns=[next(iter(REFERENCE_SOURCES.values()))])
    elif bad == "infinite":
        f[next(iter(REFERENCE_SOURCES.values()))] = np.inf
    elif bad == "unknown_zone":
        zones = ["FR", "DE", "BE"]
    else:
        zones += ["FR"]
    with pytest.raises(ValueError):
        fit_reference(f, zones, cutoff=cutoff)


def test_reference_missing_country_is_explicit_unavailable_and_unknown_apply_rejected():
    f = rank_frame()
    state = fit_reference(f.loc[f.zone.ne("NL")], ["FR", "DE", "BE", "NL"], cutoff=pd.Timestamp("2026-02-06 08:00", tz="Europe/Paris"))
    out = apply_reference(f, state)
    assert out.loc[out.zone.eq("NL"), list(REFERENCE_SOURCES)].isna().all().all()
    state["zones"] = ["FR", "DE", "BE"]
    with pytest.raises(ValueError):
        apply_reference(f, state)


@pytest.mark.parametrize("bad", ["schema", "sort", "nonfinite", "short_support", "missing_placeholders"])
def test_corrupt_reference_fails_closed(bad):
    f = rank_frame()
    state = fit_reference(f, ["FR", "DE", "BE", "NL"], cutoff=pd.Timestamp("2026-02-06 08:00", tz="Europe/Paris"))
    name = next(iter(REFERENCE_SOURCES))
    sample = state["references"][name]["FR"]["hourly"]["0"]
    if bad == "schema":
        state["sources"] = {"oracle": "actual"}
    elif bad == "sort":
        sample["values"] = np.array([1., 0.]+[0.]*33)
    elif bad == "nonfinite":
        sample["values"][0] = np.inf
    elif bad == "short_support":
        sample["values"] = sample["values"][:2]
    else:
        f = f.drop(columns=[name])
    with pytest.raises(ValueError):
        apply_reference(f, state)
