"""Independent numerical tests of the fixed, monotone rare-event calibrator."""
from copy import deepcopy
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from nyx_congestion_calibration import models


def sample(n=120):
    raw = np.linspace(.01, .7, n)
    zones = np.resize(np.array(["FR", "DE"]), n)
    days = np.array([f"2026-06-{1 + i % 28:02d}" for i in range(n)])
    return raw, zones, days


@pytest.mark.parametrize("positive", [False, True])
def test_one_class_calibration_is_finite_and_moves_toward_observations(positive):
    raw, zones, days = sample()
    state = models.fit_calibrator(raw, np.full(len(raw), int(positive)), zones, days, ("FR", "DE"))
    predicted = models.apply_calibrator(state, raw, zones)
    assert np.isfinite(predicted).all() and ((predicted > 0) & (predicted < 1)).all()
    assert predicted.mean() > raw.mean() if positive else predicted.mean() < raw.mean()
    assert state["slope"] >= 0 and state["converged"]
    assert all(v["status"] == "regularized_sparse_support" for v in state["support"].values())
    assert state["loss"] == "sum_unweighted_log_loss"


def test_anti_correlated_sample_cannot_invert_probability_within_a_country():
    raw, zones, days = sample()
    state = models.fit_calibrator(raw, (raw < .25).astype(int), zones, days, ("FR", "DE"))
    grid = np.linspace(0, 1, 101)
    for zone in ("FR", "DE"):
        predicted = models.apply_calibrator(state, grid, np.full(len(grid), zone))
        assert np.all(np.diff(predicted) >= -1e-14)
    assert state["slope"] >= 0


def test_tiny_zone_effects_are_shrunk_instead_of_separating_to_infinity():
    zones = np.array(["FR", "FR", "DE", "DE"])
    state = models.fit_calibrator(np.full(4, .5), np.array([0, 0, 1, 1]), zones,
                                  np.array(["2026-06-01"] * 4), ("FR", "DE"))
    assert -.2 < state["offsets"]["FR"] < 0 < state["offsets"]["DE"] < .2
    assert state["ridge"] == {"slope": 10., "intercept": 2., "zone": 20.}
    assert state["support"]["FR"]["positives"] == 0
    assert state["support"]["DE"]["positive_days"] == 1


def test_supported_status_requires_distinct_positive_dates_not_only_hours():
    raw = np.full(20, .2)
    y = np.r_[np.ones(10), np.zeros(10)]
    state = models.fit_calibrator(raw, y, ["DE"] * 20, ["2026-06-01"] * 20, ("DE",))
    assert state["support"]["DE"]["status"] == "regularized_sparse_support"
    state = models.fit_calibrator(raw, y, ["DE"] * 20,
                                  [f"2026-06-{1+i % 5:02d}" for i in range(20)], ("DE",))
    assert state["support"]["DE"]["status"] == "regularized_supported"


@pytest.mark.parametrize("damage", ["empty", "unknown_zone", "nonbinary", "nonfinite"])
def test_invalid_calibration_sample_fails_closed(damage):
    raw, zones, days = sample(20)
    y = (raw > .3).astype(float)
    if damage == "empty":
        raw, zones, days, y = raw[:0], zones[:0], days[:0], y[:0]
    elif damage == "unknown_zone":
        zones[0] = "BE"
    elif damage == "nonbinary":
        y[0] = .5
    else:
        raw[0] = np.nan
    with pytest.raises(ValueError):
        models.fit_calibrator(raw, y, zones, days, ("FR", "DE"))


def test_unsuccessful_optimizer_is_not_accepted(monkeypatch):
    monkeypatch.setattr(models, "minimize", lambda *a, **k: SimpleNamespace(
        success=False, fun=0., x=np.array([1., 0., 0., 0.]), message="test iteration limit", nit=1))
    raw, zones, days = sample()
    with pytest.raises(models.CalibrationError, match="did not converge"):
        models.fit_calibrator(raw, (raw > .3).astype(int), zones, days, ("FR", "DE"))


def test_serialized_calibration_predictions_repeat_exactly_and_unknown_zone_is_rejected():
    raw, zones, days = sample()
    state = models.fit_calibrator(raw, (raw > .4).astype(int), zones, days, ("FR", "DE"))
    loaded = pickle.loads(pickle.dumps(state))
    np.testing.assert_array_equal(models.apply_calibrator(state, raw, zones),
                                  models.apply_calibrator(loaded, raw, zones))
    with pytest.raises(ValueError, match="known country"):
        models.apply_calibrator(state, [.1], ["NL"])
    assert models.apply_calibrator(state, np.empty(0), []).shape == (0,)
    invalid = deepcopy(state)
    invalid["coefficients"][0] = -1.
    with pytest.raises(models.CalibrationError, match="Invalid monotone"):
        models.apply_calibrator(invalid, raw, zones)


def test_actual_classifier_accepts_zero_positive_calibration_and_survives_serialization(monkeypatch):
    monkeypatch.setattr(models, "PARAMETERS", {**models.PARAMETERS, "max_iter": 5, "min_samples_leaf": 5})
    x = np.linspace(-2, 2, 100).reshape(-1, 1)
    y = (x[:, 0] > 0).astype(int)
    v = np.linspace(-1, 1, 40).reshape(-1, 1)
    countries = np.resize(np.array(["FR", "DE"]), len(v))
    state = models.fit_probability(x, y, v, np.zeros(len(v)), cal_zones=countries,
        cal_days=[f"2026-06-{1+i % 20:02d}" for i in range(len(v))], zones=("FR", "DE"), threads=1)
    predicted, raw = models.predict_probability(state, v, countries)
    assert np.isfinite(predicted).all() and predicted.mean() < raw.mean()
    repeat = models.predict_probability(pickle.loads(pickle.dumps(state)), v, countries)
    np.testing.assert_array_equal(predicted, repeat[0])
    np.testing.assert_array_equal(raw, repeat[1])
    assert models.predict_probability(state, np.empty((0, 1)), [])[0].shape == (0,)


def test_classifier_still_requires_both_core_classes():
    with pytest.raises(ValueError, match="CORE classes"):
        models.fit_probability(np.ones((60, 1)), np.zeros(60), np.ones((20, 1)), np.zeros(20),
            cal_zones=["FR"] * 20, cal_days=["2026-06-01"] * 20, zones=("FR",), threads=1)
