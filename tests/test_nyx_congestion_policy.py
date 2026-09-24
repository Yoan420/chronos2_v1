"""Fast independent checks of causal fitting and coherent price interventions."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from nyx_congestion import policy


@pytest.fixture
def history():
    days = pd.date_range("2026-04-01", periods=150, freq="D")
    rows = []
    for d, day in enumerate(days):
        for z, zone in enumerate(("FR", "DE", "BE", "NL")):
            for hour in range(6):
                rows.append({"_day": str(day.date()), "zone": zone, "_error": 100. if (d+hour)%2 else 0.,
                    "_features_valid": True, "_label_valid": True,
                    "label_available_at_utc": (day-pd.Timedelta(days=1)+pd.Timedelta(hours=14)).tz_localize("Europe/Paris").tz_convert("UTC"),
                    policy.FUEL: 100., "feature_test": len(rows),
                    **{name: (z+1)*10+hour for name in policy.SIGNALS}})
    return pd.DataFrame(rows)


def fit(history, monkeypatch):
    calls = []
    def classifier(X, y, V, cy, **kwargs):
        calls.append(("classifier", X.copy(), y.copy(), V.copy(), cy.copy()))
        return {"fake": "classifier"}
    def cdf(X, errors, thresholds, fuel, zones, **kwargs):
        calls.append(("cdf", X.copy(), errors.copy(), thresholds.copy(), fuel.copy(), zones.copy()))
        return {"fake": "cdf"}
    monkeypatch.setattr(policy, "fit_probability", classifier)
    monkeypatch.setattr(policy, "fit_physical_cdf", cdf)
    day = "2026-08-15"
    cutoff = pd.Timestamp("2026-08-14 08:00", tz="Europe/Paris").tz_convert("UTC")
    state, record = policy._fit(history, day, cutoff,
        {"minimum_threshold_eur_mwh": 50., "threshold_quantile": .95, "threads": 1},
        ("FR", "DE", "BE", "NL"), ["feature_test"], None, None, None)
    return state, record, calls


def test_control_and_congestion_share_exact_core_calibration_rows(history, monkeypatch):
    state, record, calls = fit(history, monkeypatch)
    assert record["status"] == "trained" and record["control_and_congestion_same_training_rows"]
    control, congestion = calls[0], calls[2]
    # First matrix column is the immutable row identifier; remaining feature
    # columns differ only by the seven prequential congestion signals.
    np.testing.assert_array_equal(control[1][:, 0], congestion[1][:, 0])
    np.testing.assert_array_equal(control[2], congestion[2])
    np.testing.assert_array_equal(control[3][:, 0], congestion[3][:, 0])
    np.testing.assert_array_equal(control[4], congestion[4])
    assert set(control[1][:, 0]).isdisjoint(control[3][:, 0])
    assert congestion[1].shape[1]-control[1].shape[1] == len(policy.SIGNALS)
    assert state["models"]["control"]["features"] == ["feature_test"]
    assert state["models"]["congestion"]["features"] == ["feature_test", *policy.SIGNALS]


def test_current_future_and_late_labels_cannot_change_fit(history, monkeypatch):
    cutoff = pd.Timestamp("2026-08-14 08:00", tz="Europe/Paris").tz_convert("UTC")
    delayed = history.index[history._day.eq("2026-07-01")]
    history.loc[delayed, "label_available_at_utc"] = cutoff+pd.Timedelta(days=1)
    first_state, first_record, before = fit(history, monkeypatch)
    changed = history.copy(deep=True)
    excluded = changed._day.ge("2026-08-15") | changed.index.isin(delayed)
    changed.loc[excluded, "_error"] = 999999.
    changed.loc[excluded, "feature_test"] = -888888.
    second_state, second_record, after = fit(changed, monkeypatch)
    assert first_state["thresholds"] == second_state["thresholds"]
    assert first_record == second_record
    for a, b in zip(before, after):
        assert a[0] == b[0]
        for x, y in zip(a[1:], b[1:]):
            np.testing.assert_array_equal(x, y)
    used = set(before[0][1][:, 0]) | set(before[0][3][:, 0])
    assert not used.intersection(delayed)


def test_insufficient_distinct_eligible_days_abstains(history, monkeypatch):
    limited = history.loc[history._day.ge("2026-06-01")].copy()
    _, record, calls = fit(limited, monkeypatch)
    assert record["status"] == "fallback" and not calls
    assert min(record["eligible_training_days_by_zone"].values()) < 90


def test_checkpoint_cannot_change_chronological_split(history, monkeypatch):
    state, _, _ = fit(history, monkeypatch)
    state["core_rows"] += 1
    with pytest.raises(ValueError, match="chronological split"):
        policy._fit(history, "2026-08-15", pd.Timestamp("2026-08-14 06:00Z"),
            {"minimum_threshold_eur_mwh": 50., "threshold_quantile": .95, "threads": 1},
            ("FR", "DE", "BE", "NL"), ["feature_test"], lambda day: state, None, None)


@pytest.fixture
def inference():
    frame = pd.DataFrame({"zone": ["DE", "BE", "FR"],
        "timestamp_utc": pd.to_datetime(["2026-09-14 17:00Z"]*3),
        "forecast_origin_utc": pd.to_datetime(["2026-09-13 06:00Z"]*3),
        "congestion_ready": True, "feature_test": [1., 2., 3.],
        "physical_gate_passed": False, "forecast": 100., "q10": 80., "q90": 120., policy.FUEL: 100.})
    state = {"fit_cutoff": pd.Timestamp("2026-09-13 06:00Z"), "zones": ("DE", "BE", "FR"),
        "thresholds": {z: 50. for z in ("DE", "BE", "FR")},
        "priors": {z: .2 for z in ("DE", "BE", "FR")},
        "models": {"congestion": {"features": ["feature_test"], "classifier": {}, "cdf": {}}}}
    return frame, state


def test_prediction_uses_new_probability_and_cdf_median_not_mean_or_old_gate(inference, monkeypatch):
    current, state = inference
    calls = []
    monkeypatch.setattr(policy, "predict_probability", lambda state, X: np.array([.3, .1, .2]))
    def quantiles(cdf, X, zones, p, u, fuel):
        calls.append((X.copy(), p.copy(), u.copy()))
        return np.array([[-10., 40., 80.]]), {}
    monkeypatch.setattr(policy, "predict_physical_cdf", quantiles)
    p, correction, thresholds, detail = policy._predict(state, current, "congestion")
    np.testing.assert_array_equal(correction, [40., 0., 0.])
    np.testing.assert_array_equal(p, [.3, .1, .2])
    np.testing.assert_array_equal(thresholds, [50., 50., 50.])
    assert len(calls) == 1 and calls[0][0].shape[0] == 1
    assert detail.cdf_evaluated.tolist() == [True, False, False]
    assert detail.mixture_error_q50.tolist() == [40., 0., 0.]
    # 0.3*40 != 40: do not turn a conditional mean into a median.
    assert correction[0] != p[0]*40
    assert not current.physical_gate_passed.any()  # Old zonal gate is not retained.


def test_negative_conditional_median_does_not_lower_price(inference, monkeypatch):
    current, state = inference
    monkeypatch.setattr(policy, "predict_probability", lambda state, X: np.full(len(X), .9))
    monkeypatch.setattr(policy, "predict_physical_cdf", lambda *args: (np.tile([-30., -10., 50.], (3, 1)), {}))
    _, correction, _, detail = policy._predict(state, current, "congestion")
    assert not correction.any()
    assert detail.proposal_reason.eq("conditional_median_not_positive").all()


@pytest.mark.parametrize("change", ["future_fit", "not_ready"])
def test_inference_refuses_future_fit_or_missing_oof_signal(inference, change):
    current, state = inference
    if change == "future_fit":
        state["fit_cutoff"] += pd.Timedelta(hours=1)
    else:
        current.loc[0, "congestion_ready"] = False
    with pytest.raises(ValueError, match="causal fitted state"):
        policy._predict(state, current, "congestion")


def replay_inputs(monkeypatch):
    panel = pd.DataFrame({"zone": ["FR", "DE"], "feature_test": [1., 2.], "forecast": 100.,
        "q10": 80., "q90": 120., "timestamp_utc": pd.to_datetime(["2026-09-14 17:00Z"]*2),
        "forecast_origin_utc": pd.to_datetime(["2026-09-13 06:00Z"]*2)})
    network = pd.DataFrame({"feature_network": [1., 2.], "network_eligible": True})
    signals = pd.DataFrame({name: [1., 2.] for name in policy.SIGNALS}).assign(congestion_ready=True)
    monkeypatch.setattr(policy, "make_fundamental_features", lambda panel, variant: (panel.copy(), ["feature_test"], ["feature_test"], {}))
    monkeypatch.setattr(policy, "_check_features", lambda features: None)
    monkeypatch.setattr(policy, "_network_contract", lambda panel, network: ["feature_network"])
    return panel, network, signals


@pytest.mark.parametrize("damage", ["index", "missing", "extra", "duplicates", "nonboolean", "infinite", "filled_not_ready", "missing_ready", "unqualified_network"])
def test_strict_oof_signal_contract_before_any_fit(monkeypatch, damage):
    panel, network, signals = replay_inputs(monkeypatch)
    monkeypatch.setattr(policy.base, "run_policy", lambda *a, **k: pytest.fail("Invalid signals reached training."))
    if damage == "index":
        signals.index = [1, 0]
    elif damage == "missing":
        signals = signals.drop(columns=policy.SIGNALS[0])
    elif damage == "extra":
        signals["actual_shadow"] = 1.
    elif damage == "duplicates":
        signals = pd.concat([signals, signals[[policy.SIGNALS[0]]]], axis=1)
    elif damage == "nonboolean":
        signals["congestion_ready"] = "true"
    elif damage == "infinite":
        signals.loc[0, policy.SIGNALS[0]] = np.inf
    elif damage == "filled_not_ready":
        signals.loc[0, "congestion_ready"] = False
    elif damage == "missing_ready":
        signals.loc[0, policy.SIGNALS[0]] = np.nan
    else:
        network.loc[0, "network_eligible"] = False
    with pytest.raises(ValueError):
        policy.run_replay(panel, network, signals, {})


@dataclass
class FakeResult:
    predictions: pd.DataFrame
    folds: pd.DataFrame
    governance: pd.DataFrame
    audit: dict


@pytest.mark.parametrize("ordered", [True, False])
def test_final_quantiles_order_and_original_panel_preservation(monkeypatch, ordered):
    panel, network, signals = replay_inputs(monkeypatch)
    monkeypatch.setattr(policy.base, "_parameters", lambda value: value)
    def fake_run(augmented, settings, **kwargs):
        return FakeResult(augmented.assign(candidate_forecast=110.), pd.DataFrame({"fit": [1]}), pd.DataFrame({"day": [1]}), {})
    monkeypatch.setattr(policy.base, "run_policy", fake_run)
    monkeypatch.setattr(policy, "_finalize", lambda result, direct: result)
    monkeypatch.setattr(policy, "_bands", lambda frame: (pd.DataFrame({
        "candidate_q10": [80. if ordered else 130.]*2,
        "candidate_q90": [150.]*2, "interval_calibration_status": ["test"]*2}), {}))
    if not ordered:
        with pytest.raises(ValueError, match="ordered quantiles"):
            policy.run_replay(panel, network, signals, {})
    else:
        result = policy.run_replay(panel, network, signals, {})
        pd.testing.assert_frame_equal(result["predictions"][panel.columns], panel, check_exact=True)
        for model in policy.MODELS:
            assert result["predictions"][model].eq(110).all()
        assert result["audit"]["control_has_exact_same_training_rows"] is True
        assert result["audit"]["post_coupling_features_used"] is False
