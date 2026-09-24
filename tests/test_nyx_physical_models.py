import numpy as np
import pytest

from nyx_physical_p50 import models
from nyx_coherent_p50.distribution import predict_quantiles, mixture_quantiles, weighted_quantiles


@pytest.fixture
def cdf():
    rng = np.random.default_rng(112)
    X = rng.normal(size=(180, 3))
    error = np.r_[rng.uniform(-100, 49, 120), rng.uniform(50, 300, 60)]
    u, g = np.full(180, 50.), np.full(180, 100.)
    state = models.fit_physical_cdf(X, error, u, g, np.full(180, "DE"), threads=1)
    return X, state


def test_identical_fuel_and_threshold_recovers_exact_historical_cdf(cdf):
    X, state = cdf
    probability = np.linspace(0, 1, 12)
    expected, _ = predict_quantiles(state, X[:12], np.full(12, "DE"), probability, np.full(12, 50.))
    actual, diagnostics = models.predict_physical_cdf(state, X[:12], np.full(12, "DE"),
        probability, np.full(12, 50.), np.full(12, 100.))
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    assert not diagnostics["fuel_outside_core_support"].any()


def test_only_tail_excess_scales_not_the_event_threshold_or_normal_distribution(cdf):
    X, state = cdf
    probability = np.array([0., .3, .5, .51, .8, 1.])
    u = np.full(6, 50.)
    normal, _ = models.predict_physical_cdf(state, X[:6], np.full(6, "DE"), probability, u, np.full(6, 100.))
    doubled, audit = models.predict_physical_cdf(state, X[:6], np.full(6, "DE"), probability, u, np.full(6, 200.))
    in_tail = np.array([.1, .5, .9])[None, :] > 1-probability[:, None]
    expected = np.where(in_tail, 50+2*(normal-50), normal)
    np.testing.assert_allclose(doubled, expected, rtol=0, atol=1e-10)
    assert audit["fuel_outside_core_support"].all()
    assert (doubled[probability <= .5, 1] < 50.).all()
    assert (doubled[probability > .5, 1] >= 50.).all()


def test_transport_can_reorder_support_and_inverse_cdf_still_exact():
    # Heterogeneous historic fuel prices invert the order of residual excesses.
    normal = np.array([-1., 0.])
    support = 1+np.array([10/20, 20/200])*150/50
    answer = mixture_quantiles(normal, [.5, .5], support, [.9, .1], .8, [.12, .5, .9])
    expected = weighted_quantiles(np.r_[normal, support], np.array([.1, .1, .72, .08]), [.12, .5, .9])
    np.testing.assert_array_equal(answer, expected)


@pytest.mark.parametrize("fuel", [[0.], [-1.], [np.nan], [np.inf], []])
def test_invalid_current_fuel_is_rejected_not_imputed(cdf, fuel):
    X, state = cdf
    with pytest.raises(ValueError):
        models.predict_physical_cdf(state, X[:1], ["DE"], [.7], [50.], fuel)


@pytest.mark.parametrize("options", [{"threads": True}, {"threads": 3}, {"boost": 2}, {"threads": 1.5}])
def test_options_are_bounded(options):
    with pytest.raises(ValueError):
        models.validate_options(options)


def test_training_fuel_required_for_every_used_core_row():
    with pytest.raises(ValueError):
        models.fit_physical_cdf([[0], [1]], [0, 100], [50, 50], [100, np.nan], ["DE", "DE"], threads=1)


def test_inference_never_reads_price_label_or_date(cdf):
    X, state = cdf
    answer, _ = models.predict_physical_cdf(state, X[:1], ["DE"], [.1], [50.], [100.])
    assert answer.shape == (1, 3) and np.isfinite(answer).all()
    assert "date" not in state and "actual" not in state
