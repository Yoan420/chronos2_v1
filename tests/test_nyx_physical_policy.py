from copy import deepcopy
import numpy as np
import pandas as pd
import pytest

from nyx_physical_p50 import policy
from test_nyx_coherent_p50_policy import fixture_data
from test_nyx_fundamental_stress_policy import NAMES, FEATURE


@pytest.fixture
def sample(monkeypatch):
    original, source, folds, settings = fixture_data(days=99)
    original[policy.FUEL] = 100.
    source[policy.FUEL] = 100.
    monkeypatch.setattr(policy, "make_fundamental_features", lambda f, variant:
        (f.copy(), list(NAMES), [FEATURE], {"synthetic": True}))
    def fit(X, errors, thresholds, fuel, zones, **kwargs):
        return {"known_zones": tuple(sorted(set(zones))), "core_rows": len(X),
                "maximum_error": float(errors.max())}
    def predict(state, X, zones, probabilities, thresholds, fuel):
        median = np.where(probabilities > .5, thresholds+fuel*.5, -20.)
        return np.column_stack([median-50, median, median+100]), {
            "fuel_outside_core_support": fuel > 100,
            "fuel_ratio_to_core_max": fuel/100,
            "normal_effective_sample_size": np.full(len(X), 100.),
            "spike_effective_sample_size": np.full(len(X), 40.)}
    monkeypatch.setattr(policy, "fit_physical_cdf", fit)
    monkeypatch.setattr(policy, "predict_physical_cdf", predict)
    # Calendar/cutoff/score mechanics are separately covered by the actual
    # interval calibrator suite; keep repeated causal policy tests fast here.
    def bands(frame):
        f = frame[[*policy.KEYS, "candidate_forecast", "candidate_q10", "candidate_q90"]].copy()
        f["precalibration_q10"] = f.candidate_q10
        f["precalibration_q90"] = f.candidate_q90
        f["interval_calibration_status"] = "test_only"
        return f, {"test_only": True}
    monkeypatch.setattr(policy, "_bands", bands)
    network = pd.DataFrame({"network_eligible": True, "feature_network_proxy": .25}, index=original.index)
    return original, source, folds, network, settings


def test_replay_keeps_inputs_probabilities_and_monotone_quantiles(sample):
    original, source, folds, network, settings = sample
    result = policy.run_replay(original, source, folds, network, settings)
    out = result["predictions"]
    pd.testing.assert_frame_equal(out[original.columns], original, check_exact=True)
    pd.testing.assert_series_equal(out.spike_probability, source.spike_probability)
    pd.testing.assert_series_equal(out.expert_ready, source.expert_ready)
    for name in policy.MODELS:
        q = out[[name+"_q10", name, name+"_q90"]].to_numpy()
        assert (np.diff(q, axis=1) >= 0).all()
        np.testing.assert_array_equal(out.loc[~source.expert_ready, name], original.loc[~source.expert_ready, "forecast"])
    assert out.network_fuel_direct.ne(out.forecast).any()


def test_missing_network_never_imputed_and_fallback_exact(sample):
    original, source, folds, network, settings = sample
    network.loc[:, "network_eligible"] = False
    network.loc[:, "feature_network_proxy"] = np.nan
    out = policy.run_replay(original, source, folds, network, settings)["predictions"]
    for name in ("network_fuel_direct", "nyx_physical_p50"):
        np.testing.assert_array_equal(out[name], original.forecast)
    assert out.fuel_transport_direct.ne(out.forecast).any()


def test_late_current_labels_cannot_change_the_same_days_p50(sample):
    original, source, folds, network, settings = sample
    first = policy.run_replay(original, source, folds, network, settings)["predictions"]
    day = original.timestamp_utc.dt.tz_convert("Europe/Paris").dt.date
    mask = day.eq(day.max())
    edited, revised = original.copy(), source.copy()
    edited.loc[mask, "actual"] += 10000
    revised.loc[mask, "actual"] += 10000
    second = policy.run_replay(edited, revised, folds, network, settings)["predictions"]
    pd.testing.assert_frame_equal(first.loc[mask, list(policy.MODELS)], second.loc[mask, list(policy.MODELS)], check_exact=True)


def test_both_ablation_fits_are_checkpointed_once_per_day(sample):
    store, loads = {}, []
    def save(day, state):
        assert day not in store
        store[day] = deepcopy(state)
    def load(day):
        loads.append(day)
        return deepcopy(store.get(day))
    original, source, folds, network, settings = sample
    first = policy.run_replay(original, source, folds, network, settings, save_fit=save, load_fit=load)
    second = policy.run_replay(original, source, folds, network, settings, save_fit=save, load_fit=load)
    assert len(store) == folds.status.eq("trained").sum()
    pd.testing.assert_frame_equal(first["predictions"], second["predictions"], check_exact=True)


def test_revised_core_contract_rejected(sample):
    original, source, folds, network, settings = sample
    at = folds.index[folds.status.eq("trained")][0]
    folds.loc[at, "signed_model_training_rows"] += 1
    with pytest.raises(ValueError, match="training core"):
        policy.run_replay(original, source, folds, network, settings)


@pytest.mark.parametrize("change", ["populated_invalid", "shadow", "string_flag", "wrong_index"])
def test_bad_network_contract_rejected(sample, change):
    original, _, _, network, _ = sample
    if change == "populated_invalid":
        network.loc[0, "network_eligible"] = False
    elif change == "shadow":
        network = network.rename(columns={"feature_network_proxy": "feature_network_shadow_price"})
    elif change == "string_flag":
        network["network_eligible"] = "true"
    else:
        network.index = network.index+1
    with pytest.raises(ValueError):
        policy._network_contract(original, network)
