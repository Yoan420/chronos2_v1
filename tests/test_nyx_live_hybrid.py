"""Synthetic-only contracts for live pair integration; no real forecasts fitted."""
import copy
import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_live_hybrid as subject
from chronos2_hourly import solar_wind_interaction_features as interaction
from chronos2_hourly import solar_wind_scarcity_regime as regime
from chronos2_hourly import solar_wind_scarcity_hybrid as hybrid


DELIVERY = pd.Timestamp("2026-09-24").date()


def grid(first, stop):
    return pd.date_range(pd.Timestamp(first).tz_localize("Europe/Berlin"),
                         pd.Timestamp(stop).tz_localize("Europe/Berlin"),
                         freq="h", inclusive="left").tz_convert("UTC")


def covariates(pair, *, first=None, stop=None):
    first = first or DELIVERY - timedelta(days=390)
    stop = stop or DELIVERY + timedelta(days=1)
    index = grid(first, stop)
    hour = index.tz_convert("Europe/Berlin").hour.to_numpy()
    days = np.asarray([(x - first).days for x in index.tz_convert("Europe/Berlin").date])
    result = {}
    for position, zone in enumerate(pair):
        factor = 1.0 + position * .4
        result[zone] = pd.DataFrame({
            zone.lower() + "_wind_generation_fcst": factor * (2 + hour / 8 + days % 3),
            zone.lower() + "_solar_generation_fcst": factor * np.maximum(0, 8 - np.abs(hour - 12)),
            zone.lower() + "_residual_load_fcst": factor * (15 + hour + days % 5),
        }, index=index)
    return result


def baselines(pair):
    history, forecast = {}, {}
    for zone in pair:
        for store, first, stop, labels in (
            (history, DELIVERY - timedelta(days=365), DELIVERY, True),
            (forecast, DELIVERY, DELIVERY + timedelta(days=1), False),
        ):
            index = grid(first, stop)
            p50 = 200. + index.tz_convert("Europe/Berlin").hour.to_numpy()
            frame = pd.DataFrame({"residual_kalman__q10": p50 - 30,
                                  "residual_kalman__q50": p50,
                                  "residual_kalman__q90": p50 + 60}, index=index)
            if labels: frame["actual"] = p50 + 10
            store[zone] = frame
    return history, forecast


def routing_panel(pair, first, stop, labels=True):
    output = []
    for zone in pair:
        index = grid(first, stop)
        frame = pd.DataFrame({"zone": zone, "nyx__q10": 170., "nyx__q50": 200., "nyx__q90": 270.,
                              "test2__q10": 180., "test2__q50": 210., "test2__q90": 280.,
                              "own_joint_deficit": .75, "own_residual_stress": 1.2,
                              "nyx_daily_peak_gap": 0., "fit_origin": str(first)}, index=index)
        if labels: frame["actual"] = 210.
        output.append(frame)
    return pd.concat(output)


def test_de_nl_feature_frames_and_audit_are_bit_exact_delegations():
    source = covariates(("DE", "NL"), first=DELIVERY - timedelta(days=30))
    expected, expected_audit = regime.build_features(source)
    actual, audit = subject.build_pair_features(source, ("DE", "NL"))
    assert audit == expected_audit
    for zone in source:
        pd.testing.assert_frame_equal(actual[zone], expected[zone], check_exact=True)


@pytest.mark.parametrize("zone,timezone", [("DE", "Europe/Berlin"), ("NL", "Europe/Amsterdam")])
def test_de_nl_interaction_is_bit_exact(zone, timezone):
    frame = covariates(("DE", "NL"), first=DELIVERY - timedelta(days=30))[zone]
    expected, old_audit = interaction.build_interaction(frame, zone, timezone)
    actual, audit = subject.build_pair_interaction(frame, zone, timezone)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert audit == old_audit


def test_be_fr_has_explicit_country_indicator_and_neighbour_with_no_aliases():
    source = covariates(("BE", "FR"), first=DELIVERY - timedelta(days=30))
    before = {z: f.copy(deep=True) for z, f in source.items()}
    features, audit = subject.build_pair_features(source, ("BE", "FR"))
    assert audit["neighbor_mapping"] == {"BE": "FR", "FR": "BE"}
    assert not audit["country_aliasing_performed"]
    for zone, neighbor, flag in (("BE", "FR", 0.), ("FR", "BE", 1.)):
        frame = features[zone]
        assert "country_is_fr" in frame and "country_is_nl" not in frame
        assert frame.iloc[:14*24].isna().all().all()
        assert (frame.iloc[14*24:].country_is_fr == flag).all()
        np.testing.assert_array_equal(frame.iloc[14*24:].other_wind_gw,
                                      source[neighbor].iloc[14*24:][neighbor.lower() + "_wind_generation_fcst"])
        pd.testing.assert_frame_equal(source[zone], before[zone], check_exact=True)


@pytest.mark.parametrize("zone,timezone", [("BE", "Europe/Brussels"), ("FR", "Europe/Paris")])
def test_be_fr_interaction_uses_own_forecasts_and_documented_zero_warmup(zone, timezone):
    source = covariates(("BE", "FR"), first=DELIVERY - timedelta(days=30))
    features, _ = subject.build_pair_features(source, ("BE", "FR"))
    score, audit = subject.build_pair_interaction(source[zone], zone, timezone)
    assert list(score) == [zone.lower() + "_low_wind_solar_stress"]
    assert (score.iloc[:14*24, 0] == 0).all()
    expected = features[zone].own_joint_deficit * features[zone].own_residual_stress.clip(0, 1)
    np.testing.assert_allclose(score.iloc[14*24:, 0], expected.iloc[14*24:], rtol=0, atol=0)
    assert audit["zone"] == zone and not audit["country_aliasing_performed"]


@pytest.mark.parametrize("first,stop,day,hours", [
    ("2026-03-10", "2026-04-01", "2026-03-29", 23),
    ("2025-10-07", "2025-10-28", "2025-10-26", 25),
])
def test_be_fr_preserves_dst_and_prefix_causality(first, stop, day, hours):
    source = covariates(("BE", "FR"), first=pd.Timestamp(first).date(), stop=pd.Timestamp(stop).date())
    features, _ = subject.build_pair_features(source, ("BE", "FR"))
    selected = source["BE"].index.tz_convert("Europe/Berlin").date == pd.Timestamp(day).date()
    assert selected.sum() == hours
    changed = {z: f.copy() for z, f in source.items()}
    cutoff = pd.Timestamp(day).date()
    for frame in changed.values():
        later = frame.index.tz_convert("Europe/Berlin").date > cutoff
        frame.loc[later, :] *= 100
    altered, _ = subject.build_pair_features(changed, ("BE", "FR"))
    for zone in source:
        prefix = source[zone].index.tz_convert("Europe/Berlin").date <= cutoff
        pd.testing.assert_frame_equal(features[zone].loc[prefix], altered[zone].loc[prefix], check_exact=True)
        assert np.isfinite(features[zone].loc[selected].to_numpy()).all()


@pytest.mark.parametrize("pair", [("FR", "BE"), ("DE", "FR"), ("BE", "FR", "DE", "NL")])
def test_unsupported_or_reversed_pairs_are_not_silently_recoded(pair):
    with pytest.raises(ValueError): subject.build_pair_features({}, pair)


def test_de_nl_routing_and_complete_quantiles_are_bit_exact():
    pair = ("DE", "NL")
    past = routing_panel(pair, DELIVERY-timedelta(days=90), DELIVERY)
    old = hybrid.select_rule(past, DELIVERY)
    policy = subject.select_pair_rule(past, pair, DELIVERY)
    assert policy == old
    current = routing_panel(pair, DELIVERY, DELIVERY+timedelta(days=1), labels=False)
    pd.testing.assert_frame_equal(subject.apply_pair_rule(current, policy, pair),
                                  hybrid.apply_rule(current, old), check_exact=True)


def test_be_fr_routing_is_learned_separately_and_copies_full_triplets():
    pair = ("BE", "FR")
    past = routing_panel(pair, DELIVERY-timedelta(days=90), DELIVERY)
    policy = subject.select_pair_rule(past, pair, DELIVERY)
    assert policy["mode"] == "hybrid" and policy["pair"] == list(pair)
    assert set(policy["support"]["by_country"]) == set(pair)
    assert len(policy["candidate_scores"]) == 12
    current = routing_panel(pair, DELIVERY, DELIVERY+timedelta(days=1), labels=False)
    result = subject.apply_pair_rule(current, policy, pair)
    assert result.selected_test2.all()
    for q in ("q10", "q50", "q90"):
        np.testing.assert_array_equal(result["hybrid__"+q], current["test2__"+q])
    json.dumps(policy, allow_nan=False)
    invalid = dict(policy, pair=["DE", "NL"])
    with pytest.raises(ValueError): subject.apply_pair_rule(current, invalid, pair)


@pytest.mark.parametrize("fault", ["future_label", "missing_hour", "duplicate", "wrong_peak", "crossed"])
def test_be_fr_routing_missing_or_invalid_parents_fail_closed(fault):
    pair = ("BE", "FR")
    frame = routing_panel(pair, DELIVERY-timedelta(days=90), DELIVERY)
    if fault == "future_label":
        frame = pd.concat([frame, routing_panel(pair, DELIVERY, DELIVERY+timedelta(days=1))])
    elif fault == "missing_hour": frame = frame.iloc[1:]
    elif fault == "duplicate": frame = pd.concat([frame.iloc[:1], frame])
    elif fault == "wrong_peak": frame.iloc[0, frame.columns.get_loc("nyx_daily_peak_gap")] = 100
    else: frame.iloc[0, frame.columns.get_loc("nyx__q10")] = 1000
    with pytest.raises(ValueError): subject.select_pair_rule(frame, pair, DELIVERY)


@pytest.mark.parametrize("pair", [("DE", "NL"), ("BE", "FR")])
def test_current_labels_are_forbidden(pair):
    past = routing_panel(pair, DELIVERY-timedelta(days=90), DELIVERY)
    policy = subject.select_pair_rule(past, pair, DELIVERY)
    current = routing_panel(pair, DELIVERY, DELIVERY+timedelta(days=1), labels=True)
    with pytest.raises(ValueError): subject.apply_pair_rule(current, policy, pair)


@pytest.mark.parametrize("pair", [("DE", "NL"), ("BE", "FR")])
def test_pipeline_causal_origins_label_free_checkpoints_and_resume_without_fits(pair, monkeypatch):
    history, forecast = baselines(pair)
    sources = covariates(pair)
    calls, cache = [], {}

    def synthetic_fit(train_x, train_y, test_x, train_days, **kwargs):
        origin = pd.Timestamp(kwargs["origin_day"]).date()
        assert max(train_days) < origin
        assert min(train_days) >= origin-timedelta(days=365)
        assert not set(subject.DAILY_PEAK_COLUMNS).intersection(train_x.columns)
        indicator = "country_is_nl" if pair == ("DE", "NL") else "country_is_fr"
        assert indicator in train_x and indicator in test_x
        assert "baseline_p50" in test_x
        assert kwargs["threads"] == 1
        calls.append(origin)
        return np.tile([-20., 10., 30.], (len(test_x), 1)), np.full(len(test_x), .2), {"synthetic": True}

    def save(stage, key, panel, audit):
        assert "actual" not in panel
        assert panel.fit_origin.eq(key).all()
        cache[(stage, key)] = (panel.copy(deep=True), copy.deepcopy(audit))

    monkeypatch.setattr(regime, "fit_predict", synthetic_fit)
    result = subject.run_pair_pipeline(history, forecast, sources, pair=pair, delivery_day=DELIVERY,
                                       iterations=3, save_checkpoint=save)
    assert len(calls) == 28
    assert result["protocol"]["test2_historical_days"] == 184
    assert result["protocol"]["hybrid_historical_days"] == 93
    for zone in pair:
        sub = result["historical_hybrid"].loc[lambda x: x.zone == zone]
        assert sub.index.equals(grid(DELIVERY-timedelta(days=93), DELIVERY))
        np.testing.assert_array_equal(sub.actual, history[zone].loc[sub.index, "actual"])
    assert "actual" not in result["forecast_hybrid"]
    assert result["forecast_hybrid"].groupby("zone").size().to_dict() == {z: 24 for z in pair}
    json.dumps(result["protocol"], allow_nan=False)

    def forbidden_fit(*args, **kwargs): raise AssertionError("Resume must not refit cached models")
    monkeypatch.setattr(regime, "fit_predict", forbidden_fit)
    resumed = subject.run_pair_pipeline(history, forecast, sources, pair=pair, delivery_day=DELIVERY,
                                        iterations=3, load_checkpoint=lambda stage, key: cache.get((stage, key)))
    for name in ("historical_test2", "historical_hybrid", "forecast_test2", "forecast_hybrid"):
        pd.testing.assert_frame_equal(result[name], resumed[name], check_exact=True)
    assert len(cache) == 43  #28 Test2 model blocks +15 routing blocks.


def test_baseline_alias_disagreement_and_missing_annual_parent_fail_closed():
    pair = ("BE", "FR")
    history, forecast = baselines(pair)
    source = covariates(pair)
    invalid = {z: f.copy() for z, f in history.items()}
    invalid["BE"] = invalid["BE"].iloc[1:]
    with pytest.raises(ValueError):
        subject.run_pair_pipeline(invalid, forecast, source, pair=pair, delivery_day=DELIVERY)
    bad = forecast["BE"].copy()
    for q in ("q10", "q50", "q90"): bad["nyx__"+q] = bad["residual_kalman__"+q] + 1
    with pytest.raises(ValueError): subject._baseline(bad, allow_actual=False)


def test_de_nl_test2_passes_identical_reference_matrices_and_parameters(monkeypatch):
    pair = ("DE", "NL")
    history, forecast = baselines(pair)
    features, _ = subject.build_pair_features(covariates(pair), pair)
    capture = {}

    def spy(train_x, residual, test_x, days, **kwargs):
        capture.update(train_x=train_x.copy(), residual=residual.copy(), test_x=test_x.copy(),
                       days=list(days), kwargs=kwargs)
        return np.zeros((len(test_x), 3)), np.zeros(len(test_x)), {"spy": True}

    monkeypatch.setattr(regime, "fit_predict", spy)
    subject.fit_test2_block(history, forecast, features, pair, origin_day=DELIVERY)
    reference_train, reference_test, reference_y, reference_days = [], [], [], []
    for zone in pair:
        train = features[zone].loc[history[zone].index].copy()
        test = features[zone].loc[forecast[zone].index].copy()
        train["baseline_p50"] = history[zone].residual_kalman__q50.to_numpy()
        test["baseline_p50"] = forecast[zone].residual_kalman__q50.to_numpy()
        reference_train.append(train.drop(columns=list(subject.DAILY_PEAK_COLUMNS)))
        reference_test.append(test.drop(columns=list(subject.DAILY_PEAK_COLUMNS)))
        reference_y.append((history[zone].actual-history[zone].residual_kalman__q50).to_numpy())
        reference_days.extend(history[zone].index.tz_convert(subject.TZ).date)
    pd.testing.assert_frame_equal(capture["train_x"], pd.concat(reference_train), check_exact=True)
    pd.testing.assert_frame_equal(capture["test_x"], pd.concat(reference_test), check_exact=True)
    np.testing.assert_array_equal(capture["residual"], np.concatenate(reference_y))
    assert capture["days"] == reference_days
    assert capture["kwargs"] == {"origin_day": DELIVERY, "threads": 1, "iterations": 120, "seed": 20260923}


def test_prediction_checkpoint_rejects_country_reordering_before_label_attachment(monkeypatch):
    pair = ("BE", "FR")
    history, forecast = baselines(pair)
    features, _ = subject.build_pair_features(covariates(pair), pair)
    def spy(train_x, residual, test_x, days, **kwargs):
        return np.zeros((len(test_x), 3)), np.zeros(len(test_x)), {"spy": True}
    monkeypatch.setattr(regime, "fit_predict", spy)
    panel, audit = subject.fit_test2_block(history, forecast, features, pair, origin_day=DELIVERY)
    reversed_countries = pd.concat([panel.loc[panel.zone == z] for z in reversed(pair)])
    with pytest.raises(ValueError, match="country ordering"):
        subject._validate_checkpoint(reversed_countries, audit, forecast, features, pair,
                                     DELIVERY, DELIVERY+timedelta(days=1))
