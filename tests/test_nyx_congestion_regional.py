"""Independent causal contracts for the whole-domain congestion pressure head."""
import numpy as np
import pandas as pd
import pytest

from nyx_congestion import regional


def fixture(days=84):
    rows, labels = [], []
    levels = {"FR": -20., "DE": 0., "BE": 40., "NL": 100.}
    for d in pd.date_range("2026-03-01", periods=days, freq="D"):
        origin = (d-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
        for hour in (0, 4, 8, 12, 16, 20):
            stamp = (d+pd.Timedelta(hours=hour)).tz_localize("Europe/Paris").tz_convert("UTC")
            for zone in regional.ZONES:
                rows.append(dict(zone=zone, timestamp_utc=stamp, forecast_origin_utc=origin,
                    forecast=100., actual=101., q10=80., q90=120.,
                    label_available_at_utc=origin+pd.Timedelta(hours=6),
                    feature_eligible=True, forecast_eligible=True,
                    feature_row_id=float(len(rows)), **{regional.FUEL:100.}))
                labels.append(dict(zone=zone, timestamp_utc=stamp,
                    label_directional_contribution_eur_mwh=levels[zone],
                    label_eligible=True, label_available_at_utc=origin+pd.Timedelta(hours=6)))
    panel = pd.DataFrame(rows)
    network = pd.DataFrame({"network_eligible":True, "feature_network_test":1.}, index=panel.index)
    return panel, network, pd.DataFrame(labels)


def test_targets_are_gauge_invariant_and_preserve_subthreshold_pressure():
    panel, _, labels = fixture(2)
    one = regional.regional_targets(panel, labels)
    shifted = labels.copy()
    shifted["label_directional_contribution_eur_mwh"] += shifted.timestamp_utc.factorize()[0]*37.-500.
    two = regional.regional_targets(panel, shifted)
    pd.testing.assert_frame_equal(one, two)
    assert one.loc[one.zone.eq("FR"), "realised_fb_premium"].eq(0.).all()
    assert one.loc[one.zone.eq("DE"), "realised_fb_premium"].eq(20.).all()
    assert one.loc[one.zone.eq("DE"), "label_active"].eq(0.).all()
    assert one.loc[one.zone.eq("DE"), "label_shadow_price"].eq(0.).all()
    assert one.loc[one.zone.eq("BE"), "realised_fb_premium"].eq(60.).all()
    assert one.loc[one.zone.eq("BE"), "label_shadow_price"].eq(60.).all()


@pytest.mark.parametrize("mutation", ["absent", "false_flag", "missing_value", "missing_watermark"])
def test_one_unqualified_zone_invalidates_the_whole_hour(mutation):
    panel, _, labels = fixture(2)
    first = labels.timestamp_utc.iloc[0]
    selected = labels.timestamp_utc.eq(first) & labels.zone.eq("FR")
    if mutation == "absent": labels = labels.loc[~selected]
    elif mutation == "false_flag": labels.loc[selected, "label_eligible"] = False
    elif mutation == "missing_value": labels.loc[selected, "label_directional_contribution_eur_mwh"] = np.nan
    else: labels.loc[selected, "label_available_at_utc"] = pd.NaT
    result = regional.regional_targets(panel, labels)
    unknown = result.loc[result.timestamp_utc.eq(first)]
    assert not unknown.label_eligible.any()
    assert unknown[["realised_fb_premium", "label_active", "label_shadow_price", "label_available_at_utc"]].isna().all().all()
    assert result.loc[~result.timestamp_utc.eq(first), "label_eligible"].all()


def test_latest_of_all_four_publication_times_controls_label_availability():
    panel, _, labels = fixture(1)
    last = pd.Timestamp("2026-03-04T10:00:00Z")
    labels.loc[labels.zone.eq("FR"), "label_available_at_utc"] = last
    result = regional.regional_targets(panel, labels)
    assert result.label_available_at_utc.eq(last).all()


@pytest.mark.parametrize("mutation", ["duplicate", "unknown_zone", "infinite", "naive_timestamp", "own_origin", "numeric_flag", "naive_watermark"])
def test_invalid_label_contract_is_rejected(mutation):
    panel, _, labels = fixture(1)
    if mutation == "duplicate": labels = pd.concat([labels, labels.iloc[:1]])
    elif mutation == "unknown_zone": labels.loc[0, "zone"] = "AT"
    elif mutation == "infinite": labels.loc[0, "label_directional_contribution_eur_mwh"] = np.inf
    elif mutation == "naive_timestamp": labels["timestamp_utc"] = labels.timestamp_utc.dt.tz_localize(None)
    elif mutation == "own_origin": labels["label_available_at_utc"] = panel.forecast_origin_utc
    elif mutation == "numeric_flag": labels["label_eligible"] = 1
    else: labels["label_available_at_utc"] = labels.label_available_at_utc.dt.tz_localize(None)
    with pytest.raises(ValueError):
        regional.regional_targets(panel, labels)


def install_fast_models(monkeypatch):
    calls = []
    def fundamentals(panel, **kwargs):
        return panel.copy(), ["feature_row_id", regional.FUEL], ["feature_row_id", regional.FUEL], {}
    def fit(x, active, shadow, fuel, validation, calibration_active, **kwargs):
        calls.append((x.copy(), np.asarray(active).copy(), np.asarray(shadow).copy(), validation.copy()))
        assert set(x[:,0]).isdisjoint(validation[:,0])
        return dict(mean=float(np.mean(shadow)), probability=float(np.mean(active)))
    def predict(state, x, fuel):
        return dict(activation_probability=np.full(len(x), state["probability"]),
            intensity_if_active=np.full(len(x), state["mean"]),
            expected_shadow_price=np.full(len(x), state["probability"]*state["mean"]),
            climatology_probability=np.full(len(x), state["probability"]))
    monkeypatch.setattr(regional, "make_fundamental_features", fundamentals)
    monkeypatch.setattr(regional, "fit_activation_intensity", fit)
    monkeypatch.setattr(regional, "predict_activation_intensity", predict)
    return calls


def test_weekly_prequential_replay_is_invariant_to_future_and_unavailable_labels(monkeypatch):
    calls = install_fast_models(monkeypatch)
    panel, network, labels = fixture()
    unavailable = labels.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d").eq("2026-03-10")
    labels.loc[unavailable, "label_available_at_utc"] = pd.Timestamp("2027-01-01T00:00:00Z")
    one, folds, signals, audit = regional.run_regional(panel, network, labels, {}, threads=1)
    assert one.expert_ready.any() and calls
    excluded_ids = set(panel.loc[unavailable, "feature_row_id"])
    for x, _, _, validation in calls:
        assert not excluded_ids.intersection(x[:,0])
        assert not excluded_ids.intersection(validation[:,0])
    trained = folds.loc[folds.status.eq("trained")]
    assert trained.max_label_available_at_utc.le(trained.fit_cutoff_utc).all()
    assert trained.model_training_end_day.lt(trained.calibration_start_day).all()
    assert pd.to_datetime(folds.fit_day).diff().dropna().dt.days.eq(7).all()
    assert not any("label" in n or "shadow" in n or "actual" in n for n in audit["features"])
    assert signals.index.equals(panel.index)
    changed = labels.copy()
    boundary = pd.Timestamp("2026-05-12T00:00:00", tz="Europe/Paris").tz_convert("UTC")
    edit = (changed.timestamp_utc.ge(boundary) | unavailable) & changed.zone.eq("BE")
    changed.loc[edit, "label_directional_contribution_eur_mwh"] += 9999.
    two, _, second_signals, _ = regional.run_regional(panel, network, changed, {}, threads=1)
    before = panel.timestamp_utc.lt(boundary)
    cols = ["activation_probability", "intensity_if_active", "expected_shadow_price", "expert_ready", "fit_day"]
    pd.testing.assert_frame_equal(one.loc[before, cols], two.loc[before, cols])
    pd.testing.assert_frame_equal(signals.loc[before], second_signals.loc[before])


def test_current_missing_labels_do_not_prevent_qualified_forecasts(monkeypatch):
    install_fast_models(monkeypatch)
    panel, network, labels = fixture(70)
    last_day = labels.timestamp_utc.dt.tz_convert("Europe/Paris").dt.date.max()
    current = labels.timestamp_utc.dt.tz_convert("Europe/Paris").dt.date.eq(last_day)
    labels.loc[current, "label_eligible"] = False
    labels.loc[current, "label_directional_contribution_eur_mwh"] = np.nan
    labels.loc[current, "label_available_at_utc"] = pd.NaT
    result, _, signals, _ = regional.run_regional(panel, network, labels, {}, threads=1)
    assert result.loc[current, "expert_ready"].all()
    assert result.loc[current, "realised_fb_premium"].isna().all()
    assert signals.loc[current, regional.REGIONAL_SIGNALS].notna().all().all()


def test_missing_network_or_late_fundamental_features_abstain(monkeypatch):
    install_fast_models(monkeypatch)
    panel, network, labels = fixture(70)
    selected = panel.index[-8:]
    network.loc[selected[:4], "network_eligible"] = False
    network.loc[selected[:4], "feature_network_test"] = np.nan
    panel["feature_available_at_utc"] = panel.forecast_origin_utc
    panel.loc[selected[4:], "feature_available_at_utc"] += pd.Timedelta(seconds=1)
    result, _, signals, _ = regional.run_regional(panel, network, labels, {}, threads=1)
    assert not result.loc[selected, "expert_ready"].any()
    assert signals.loc[selected, regional.REGIONAL_SIGNALS].isna().all().all()


def test_nondefault_original_index_does_not_create_rows_or_misalign_predictions(monkeypatch):
    install_fast_models(monkeypatch)
    panel, network, labels = fixture(70)
    panel.index = pd.Index(np.arange(len(panel))*3+10000, name="source_row")
    network.index = panel.index
    result, _, signals, _ = regional.run_regional(panel, network, labels, {}, threads=1)
    assert len(result) == len(panel)
    assert signals.index.equals(panel.index)
    assert result.timestamp_utc.notna().all()
    assert signals.regional_ready.any()
