import numpy as np
import pytest
from nyx_congestion import models


@pytest.fixture
def state(monkeypatch):
    monkeypatch.setitem(models.PARAMETERS, 'max_iter', 5)
    random = np.random.default_rng(1729)
    x = random.normal(size=(240, 3))
    active = (x[:, 0] > 0).astype(float)
    shadow = active*(20+np.abs(x[:, 1])*10)
    return models.fit_activation_intensity(x[:180], active[:180], shadow[:180], np.full(180, 10.),
                                           x[180:], active[180:], threads=1)


def test_hurdle_semantics_and_positive_fuel_scaling(state):
    x = np.array([[1., 0., 0.], [-1., 0., 0.]])
    a = models.predict_activation_intensity(state, x, [10., 10.])
    b = models.predict_activation_intensity(state, x, [20., 20.])
    np.testing.assert_allclose(a['expected_shadow_price'], a['activation_probability']*a['intensity_if_active'])
    np.testing.assert_allclose(b['intensity_if_active'], 2*a['intensity_if_active'])
    assert 'p50' not in a and 'forecast' not in a
    assert np.isfinite(a['intensity_if_active']).all()


@pytest.mark.parametrize('fuel', [[0.], [-1.], [np.nan], [np.inf]])
def test_invalid_fuel_rejected(state, fuel):
    with pytest.raises(ValueError):
        models.predict_activation_intensity(state, [[1., 0., 0.]], fuel)


def test_unknown_dimension_and_infinity_rejected(state):
    for x in ([[1., 0.]], [[np.inf, 0., 0.]]):
        with pytest.raises(ValueError):
            models.predict_activation_intensity(state, x, [10.])


def test_classes_and_inconsistent_activity_rejected():
    x = np.ones((50, 2))
    with pytest.raises(ValueError):
        models.fit_probability(x, np.ones(50), x, np.ones(50))
    with pytest.raises(ValueError):
        models.fit_activation_intensity(x, np.zeros(50), np.ones(50), np.ones(50), x, np.zeros(50))
