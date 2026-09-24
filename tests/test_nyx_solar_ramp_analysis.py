import json

import numpy as np
import pandas as pd
import pytest

from nyx_solar_ramp.analysis import analyse


def fixture():
    rows = []
    for day in pd.date_range("2026-09-10", periods=6):
        for zone in ("DE", "BE"):
            for hour in (17, 18, 19):
                stamp = (day+pd.Timedelta(hours=hour)).tz_localize("Europe/Paris").tz_convert("UTC")
                origin = (day-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
                drop = 1. if day.day == 10 else 2. if day.day == 11 else 0. if day.day == 13 else 3.
                actual = 400. if day.day == 13 else 100.
                rows.append({"zone": zone, "timestamp_utc": stamp, "forecast_origin_utc": origin,
                             "sample": "live" if day.day == 15 else "evaluation", "actual": actual, "forecast": 110.,
                             "solarx_local_solar_gw": 10., "solarx_peer_solar_gw": 15.,
                             "solarx_local_solar_drop_1h_gwph": drop,
                             "solarx_peer_solar_drop_1h_gwph": 2*drop,
                             "solarx_local_residual_gw": 50., "solarx_local_wind_gw": 5.,
                             "solarx_local_pressure_proxy": 1.2})
    panel = pd.DataFrame(rows)
    pred = panel[["zone", "timestamp_utc", "forecast_origin_utc", "actual", "forecast"]].copy()
    pred["variant"], pred["candidate_forecast"], pred["risk_probability"], pred["alert"] = "governed", 120., .1, False
    return panel, pred, {"minimum_training_days": 2, "business_spike": 300., "primary_variant": "governed"}


def test_descriptive_counts_counterexamples_common_strata_and_json_safe():
    panel, pred, cfg = fixture()
    original, original_pred = panel.copy(deep=True), pred.copy(deep=True)
    result = analyse(panel, pred, cfg)
    assert result["coverage"]["evaluation_rows"] == 30
    assert result["coverage"]["post_warmup_rows"] == 18
    assert result["coverage"]["large_drop_without_spike_rows"] == 12
    assert result["coverage"]["spike_without_drop_rows"] == 6
    for summary in result["matched_strata"]["summary"]:
        assert summary["n_matched_strata"] == 3
        assert summary["weighted_spike_rate_large_drop"] == 0.
        assert summary["weighted_spike_rate_other"] == 1.
    assert all(row["solar_drop_gwph"] == 3. for row in result["counterexamples"]["large_drop_without_spike"])
    assert all(row["solar_drop_gwph"] == 0. for row in result["counterexamples"]["spike_without_drop"])
    assert not result["protocol"]["causal_identification"]
    json.dumps(result, allow_nan=False)
    pd.testing.assert_frame_equal(panel, original, check_exact=True)
    pd.testing.assert_frame_equal(pred, original_pred, check_exact=True)


def test_warmup_thresholds_ignore_prices_and_all_post_warmup_covariates():
    panel, pred, cfg = fixture()
    before = analyse(panel, pred, cfg)
    later = panel.timestamp_utc.dt.tz_convert("Europe/Paris").dt.day.gt(11)
    panel.loc[later, "solarx_local_solar_drop_1h_gwph"] = 10000.
    panel.loc[later, "solarx_local_residual_gw"] = 1e6
    panel["actual"] = np.arange(len(panel))*1e4
    pred = pred.drop(columns="actual")
    after = analyse(panel, pred, cfg)
    assert before["warmup_thresholds"] == after["warmup_thresholds"]
    assert all(row["solar_drop_q90_gwph"] == 2. for row in before["warmup_thresholds"])


def test_live_rows_never_influence_thresholds_rates_matching_or_examples():
    panel, pred, cfg = fixture()
    before = analyse(panel, pred, cfg)
    live = panel["sample"].eq("live")
    panel.loc[live, "actual"] = 1e8
    panel.loc[live, "solarx_local_solar_drop_1h_gwph"] = 1e9
    pred.loc[live, "actual"] = -1e10
    assert before == analyse(panel, pred, cfg)


def test_event_intervals_are_start_labelled_local_and_utc_with_candidate():
    panel, pred, cfg = fixture()
    event = analyse(panel, pred, cfg)["event_rows"]
    assert len(event) == 6
    first = next(row for row in event if row["zone"] == "DE" and row["hour"] == 17)
    assert first["timestamp_utc"] == "2026-09-14T15:00:00+00:00"
    assert first["delivery_start_local"] == "2026-09-14T17:00:00+02:00"
    assert first["delivery_end_local"] == "2026-09-14T18:00:00+02:00"
    assert first["actual"] == 100. and first["baseline"] == 110. and first["candidate"] == 120.


def test_missing_physics_remain_unknown_not_imputed_and_missing_candidate_is_explicit():
    panel, _, cfg = fixture()
    panel["solarx_local_solar_drop_1h_gwph"] = np.nan
    panel["solarx_local_pressure_proxy"] = np.nan
    result = analyse(panel, pd.DataFrame(), cfg)
    assert not result["matched_strata"]["rows"]
    assert all(row["solar_drop_q90_gwph"] is None for row in result["warmup_thresholds"])
    assert all(row["candidate"] is None for row in result["event_rows"])
    assert all(row["drop_regime"] == "unknown" for row in result["regime_rates"])
    json.dumps(result, allow_nan=False)


def test_spikes_without_drop_do_not_require_a_positive_warmup_drop_threshold():
    panel, pred, cfg = fixture()
    warm = panel.timestamp_utc.dt.tz_convert("Europe/Paris").dt.day.le(11)
    panel.loc[warm, "solarx_local_solar_drop_1h_gwph"] = 0.
    result = analyse(panel, pred, cfg)
    assert all(row["solar_drop_q90_gwph"] is None for row in result["warmup_thresholds"])
    assert result["coverage"]["spike_without_drop_rows"] == 6
    assert result["coverage"]["large_drop_without_spike_rows"] == 0


def test_predicted_actual_mismatch_and_duplicate_identity_fail_closed():
    panel, pred, cfg = fixture()
    pred.loc[0, "actual"] += 1.
    with pytest.raises(ValueError, match="disagree"):
        analyse(panel, pred, cfg)
    panel, pred, cfg = fixture()
    with pytest.raises(ValueError, match="Duplicate"):
        analyse(panel, pd.concat([pred, pred.iloc[[0]]]), cfg)


def test_order_invariance():
    panel, pred, cfg = fixture()
    before = analyse(panel, pred, cfg)
    after = analyse(panel.sample(frac=1, random_state=7), pred.sample(frac=1, random_state=11), cfg)
    assert before == after
