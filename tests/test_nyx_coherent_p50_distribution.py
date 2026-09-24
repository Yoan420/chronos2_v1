"""Finite-support, causal-input-neutral tests; no full backtest or source writes."""
import numpy as np
import pytest

from nyx_coherent_p50 import distribution as module


def test_weighted_quantiles_are_left_inverse_not_interpolation():
    result = module.weighted_quantiles([10, -20, 0, 999], [.25, .25, .5, 0], [0, .25, .5, .75, .8, 1])
    np.testing.assert_array_equal(result, [-20, -20, 0, 0, 10, 10])


def test_atoms_and_zero_mass_support_are_preserved():
    result = module.weighted_quantiles([-100, 2, 2, 10, 100], [0, .2, .3, .5, 0], [.1, .5, .5001, 1])
    np.testing.assert_array_equal(result, [2, 2, 10, 10])


@pytest.mark.parametrize("p,median", [(0, 0), (.2, .4), (.5, .9), (.6, 1.), (1, 2.)])
def test_exact_two_regime_median(p, median):
    result = module.mixture_quantiles([-.4, 0, .4, .9], [1, 1, 1, 1], [1, 2, 3], [1, 1, 1], p)
    assert result[1] == median
    assert np.all(np.diff(result) >= 0)
    assert result[1] < 1 if p <= .5 else result[1] >= 1


def test_median_switches_only_across_probability_half():
    p_low, p_high = np.nextafter(.5, 0), np.nextafter(.5, 1)
    args = ([0, .9], [1, 1], [1, 5], [1, 1])
    assert module.mixture_quantiles(*args, p_low)[1] == .9
    assert module.mixture_quantiles(*args, .5)[1] == .9
    assert module.mixture_quantiles(*args, p_high)[1] == 1


def test_all_quantiles_are_monotone_in_spike_probability_at_fixed_context():
    args = ([-2, -.1, .9], [1, 2, 1], [1, 2, 10], [1, 2, 1])
    quantiles = np.array([module.mixture_quantiles(*args, p) for p in np.linspace(0, 1, 101)])
    assert np.all(np.diff(quantiles, axis=0) >= 0)


def test_zero_probability_does_not_hide_an_invalid_spike_distribution():
    with pytest.raises(ValueError):
        module.mixture_quantiles([0], [1], [2], [0], 0)


def test_leaf_weights_use_every_tree_and_leaf_observation():
    fake = {"support": np.array([-1, 0, .5]),
        "leaf_members": [{7: np.array([0, 1])}, {3: np.array([1, 2])}]}
    weights = module.leaf_weights(fake, [7, 3])
    np.testing.assert_allclose(weights, [.25, .5, .25])
    assert weights.sum() == 1


@pytest.fixture(scope="module")
def core():
    # Optional all-NaN column must be retained, with no inference-time fit.
    errors = np.r_[np.linspace(-2, .95, 120), np.linspace(1, 8, 60)]
    zones = np.array(["DE"]*90+["BE"]*90)
    X = np.column_stack([np.arange(180)%24, np.linspace(0, 1, 180), np.full(180, np.nan)])
    X[::11, 1] = np.nan
    return X, errors, zones


@pytest.fixture(scope="module")
def forest(core):
    X, errors, zones = core
    return module.fit_distributions(X, errors, zones, threads=1)


def test_forest_recipe_and_all_signed_labels_are_retained(core, forest):
    assert forest["kind"] == "forest"
    assert forest["forest_parameters"] == module.FOREST_PARAMETERS
    assert forest["regimes"]["normal"]["core_rows"] == 120
    assert forest["regimes"]["spike"]["core_rows"] == 60
    assert (forest["regimes"]["normal"]["support"] < 0).any()
    assert (forest["regimes"]["normal"]["support"] > 0).any()
    assert np.all(forest["regimes"]["normal"]["support"] < 1)
    assert np.all(forest["regimes"]["spike"]["support"] >= 1)
    assert forest["imputer"].statistics_.shape == (3,)


def test_real_forest_quantiles_finite_supported_ordered_and_scaled(core, forest):
    X = np.array([[12., np.nan, np.nan]]*5)
    probabilities = np.array([0, .2, .5, .6, 1])
    result, audit = module.predict_quantiles(forest, X, ["DE"]*5, probabilities, [50]*5)
    assert np.isfinite(result).all()
    assert np.all(np.diff(result, axis=1) >= 0)
    assert (result[:3, 1] < 50).all()
    assert (result[3:, 1] >= 50).all()
    assert (audit["normal_effective_sample_size"] >= 1).all()
    assert set(audit["normal_cdf_source"]) == {"forest_leaf_conditional"}
    support = np.r_[forest["regimes"]["normal"]["support"], forest["regimes"]["spike"]["support"]]*50
    assert np.isin(result, support).all()
    assert np.array_equal(audit["mixture_spike_probability"], probabilities)


def test_prediction_does_not_refit_imputer_or_modify_inputs(core, forest):
    X = np.array([[100., np.nan, np.nan]])
    before = X.copy()
    statistics = forest["imputer"].statistics_.copy()
    module.predict_quantiles(forest, X, ["DE"], [.7], [100])
    np.testing.assert_array_equal(X, before)
    np.testing.assert_array_equal(forest["imputer"].statistics_, statistics)


def test_empirical_country_minimum_and_same_regime_pool():
    errors = np.r_[np.linspace(-1, .8, 12), np.full(5, .9), np.arange(1, 13), np.full(5, 99)]
    zones = np.array(["DE"]*12+["BE"]*5+["DE"]*12+["BE"]*5)
    X = np.zeros((len(errors), 2))
    state = module.fit_distributions(X, errors, zones, kind="empirical", threads=1)
    result, audit = module.predict_quantiles(state, X[:2], ["DE", "BE"], [.6, .6], [50, 50])
    assert audit["normal_cdf_source"].tolist() == ["empirical_country", "empirical_pooled_normalized"]
    assert audit["spike_cdf_source"].tolist() == ["empirical_country", "empirical_pooled_normalized"]
    assert audit["normal_support_count"].tolist() == [12, 17]
    assert (result[:, 1] >= 50).all()


@pytest.mark.parametrize("values", [[0, .5], [1, 2], [0, np.nan], [0, np.inf]])
def test_fit_rejects_missing_regime_or_bad_target(values):
    with pytest.raises(ValueError):
        module.fit_distributions([[0], [1]], values, ["DE", "DE"], threads=1)


@pytest.mark.parametrize("p,u", [(-.1, 50), (1.1, 50), (np.nan, 50), (.6, 0), (.6, np.inf)])
def test_bad_probabilities_or_thresholds_fail(core, forest, p, u):
    with pytest.raises(ValueError):
        module.predict_quantiles(forest, [[0, 0, 0]], ["DE"], [p], [u])


def test_empty_prediction_and_bad_shape(core, forest):
    result, audit = module.predict_quantiles(forest, np.empty((0, 3)), [], [], [])
    assert result.shape == (0, 3)
    assert audit["normal_support_count"].shape == (0,)
    with pytest.raises(ValueError):
        module.predict_quantiles(forest, [[0, 0]], ["DE"], [.2], [50])
    with pytest.raises(ValueError):
        module.predict_quantiles(forest, [[0, 0, 0]], ["UNKNOWN"], [.2], [50])


@pytest.mark.parametrize("normal,spike", [([1], [2]), ([0], [.9]), ([np.nan], [2])])
def test_invalid_mixture_support_fails(normal, spike):
    with pytest.raises(ValueError):
        module.mixture_quantiles(normal, [1], spike, [1], .6)


def test_negative_weights_and_unknown_leaf_fail():
    with pytest.raises(ValueError):
        module.weighted_quantiles([1, 2], [-1, 2], [.5])
    with pytest.raises(ValueError):
        module.leaf_weights({"support": [1], "leaf_members": [{1: np.array([0])}]}, [2])
