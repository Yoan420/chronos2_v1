"""Isolated mean functional, chronology, signed governance and cache tests."""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest

from nyx_rmse import models, policy
from nyx_coherent_p50.distribution import fit_distributions, leaf_weights
from test_nyx_coherent_p50_policy import fixture_data
from test_nyx_fundamental_stress_policy import NAMES, FEATURE, panel, parameters


def fake_fit(X, errors, thresholds, options):
    normalized = errors/thresholds
    return {"core_rows": len(errors), "normal_rows": int((normalized < 1).sum()),
        "spike_rows": int((normalized >= 1).sum()), "negative_rows": int((errors < 0).sum()),
        "options": deepcopy(options), "core_mean": float(errors.mean())}


def fake_predict(state, X, probabilities, thresholds):
    return {"residual_mse": X[:, 0]*400-100+state["core_mean"],
            "mixture_mean": X[:, 0]*450-120+state["core_mean"]}


@pytest.fixture
def synthetic(monkeypatch):
    monkeypatch.setattr(policy, "make_fundamental_features", lambda frame, variant:
        (frame.copy(), list(NAMES), [FEATURE], {"synthetic_test": True}))
    monkeypatch.setattr(policy, "fit_means", fake_fit)
    monkeypatch.setattr(policy, "predict_means", fake_predict)
    return fixture_data(days=105)


@pytest.mark.parametrize("options", [
    {"threads": True}, {"threads": 3}, {"learner": {"loss": "absolute_error"}},
    {"learner": {"max_iter": 1.5}}, {"learner": {"learning_rate": float("nan")}},
    {"governance": {"weights": [0, .5]}}, {"governance": {"weights": [0, 1, 1]}},
    {"governance": {"minimum_days": 91}}, {"governance": {"maximum_mae_degradation": -1}},
    {"correction_clip_eur_mwh": 0}, {"require_full_training_history": "false"}, {"unknown": 0},
])
def test_invalid_options_rejected(options):
    with pytest.raises(ValueError):
        models.validate_options(options)


def test_defaults_are_not_mutated():
    first = models.validate_options()
    first["governance"]["weights"].append(.1)
    assert .1 not in models.validate_options()["governance"]["weights"]


def test_fast_forest_mean_equals_weighted_leaf_cdf_expectation():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(180, 5))
    X[::11, 1] = np.nan
    X[:, -1] = np.nan
    normalized = np.r_[rng.normal(-.1, .2, 130), rng.uniform(1., 4., 50)]
    thresholds = np.linspace(50., 150., len(X))
    settings = {"threads": 1, "learner": {"max_iter": 4}}
    means = models.fit_means(X, normalized*thresholds, thresholds, settings)
    cdfs = fit_distributions(X, normalized, np.full(len(X), "FR"), threads=1)
    query = X[[2, 7, 23, 76, 140]]
    probability = np.array([0., .1, .4, .8, 1.])
    u = np.array([50., 75., 100., 130., 150.])
    fast = models.predict_means(means, query, probability, u)["mixture_mean"]
    expected = np.zeros(len(query))
    transformed = cdfs["imputer"].transform(query)
    for regime, mass in (("normal", 1-probability), ("spike", probability)):
        distribution = cdfs["regimes"][regime]
        leaves = distribution["forest"].apply(transformed)
        expected += mass*np.array([np.dot(distribution["support"], leaf_weights(distribution, row)) for row in leaves])
        np.testing.assert_allclose(means["forests"][regime].predict(transformed),
                                   distribution["forest"].predict(transformed), atol=1e-12)
    np.testing.assert_allclose(fast, u*expected, atol=1e-10, rtol=1e-12)
    assert "leaf_members" not in means and "support" not in means
    assert means["direct"].loss == "squared_error"
    assert means["direct"].early_stopping is False


def test_direct_mse_fits_raw_eur_errors_not_normalized(monkeypatch):
    fitted = []
    class RecordingRegressor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def fit(self, X, y):
            fitted.append((y.copy(), self.kwargs))
            return self
    monkeypatch.setattr(models, "HistGradientBoostingRegressor", RecordingRegressor)
    X = np.arange(60.).reshape(30, 2)
    errors = np.r_[np.full(15, -20.), np.full(15, 180.)]
    models.fit_means(X, errors, np.linspace(50., 130., 30), {"threads": 1})
    np.testing.assert_array_equal(fitted[0][0], errors)
    assert fitted[0][1]["loss"] == "squared_error"


@pytest.mark.parametrize("probabilities,thresholds", [([-.1], [50]), ([1.1], [50]), ([np.nan], [50]), ([.5], [0])])
def test_invalid_mean_inference_rejected(probabilities, thresholds):
    with pytest.raises(ValueError):
        models.predict_means({"feature_count": 2}, [[1, 2]], probabilities, thresholds)


def test_replay_preserves_panel_and_readiness_signed_points_not_p50(synthetic):
    original, source, folds, settings = synthetic
    result = policy.run_replay(original, source, folds, settings)
    out = result["predictions"]
    pd.testing.assert_frame_equal(out[original.columns], original, check_exact=True)
    pd.testing.assert_series_equal(out.expert_ready, source.expert_ready)
    pd.testing.assert_series_equal(out.spike_probability, source.spike_probability)
    ready = out.expert_ready
    assert (out.loc[ready, "residual_mse_direct"] < out.loc[ready, "forecast"]).any()
    assert (out.loc[ready, "residual_mse_direct"] > out.loc[ready, "q90"]).any()
    assert (out.loc[ready, "residual_mse_direct"] < out.loc[ready, "q10"]).any()
    for candidates in policy.STRATEGIES.values():
        for candidate in candidates:
            np.testing.assert_array_equal(out.loc[~ready, candidate], original.loc[~ready, "forecast"])
    assert not result["audit"]["candidate_forecasts_are_p50"]
    assert not result["audit"]["candidate_intervals_produced"]
    assert not result["audit"]["risk_or_positive_only_gate_used"]
    assert not result["audit"]["all_trained_folds_have_365_days"]


def test_row_order_and_custom_index_preserved(synthetic):
    original, source, folds, settings = synthetic
    order = np.random.default_rng(21).permutation(len(original))
    original, source = original.iloc[order].copy(), source.iloc[order].copy()
    original.index = source.index = pd.Index(np.arange(len(original))+1200, name="custom")
    out = policy.run_replay(original, source, folds, settings)["predictions"]
    pd.testing.assert_frame_equal(out[original.columns], original, check_exact=True)
    pd.testing.assert_series_equal(out.expert_ready, source.expert_ready)


def test_signed_symmetric_cap(synthetic, monkeypatch):
    original, source, folds, settings = synthetic
    monkeypatch.setattr(policy, "predict_means", lambda state, X, p, u:
        {"residual_mse": np.full(len(X), -1000.), "mixture_mean": np.full(len(X), 1000.)})
    out = policy.run_replay(original, source, folds, settings, {"correction_clip_eur_mwh": 400})["predictions"]
    ready = out.expert_ready
    assert out.loc[ready, "residual_mse_bounded_correction"].eq(-400).all()
    assert out.loc[ready, "mixture_mean_bounded_correction"].eq(400).all()
    assert out.loc[ready, "residual_mse_raw_correction"].eq(-1000).all()
    assert out.loc[ready, "mixture_mean_raw_correction"].eq(1000).all()


def test_fitting_reuses_valid_cache_without_training(synthetic, monkeypatch):
    original, source, folds, settings = synthetic
    cache = {}
    first = policy.run_replay(original, source, folds, settings,
                             save_fit=lambda day, state: cache.update({day: deepcopy(state)}))
    assert len(cache) == first["audit"]["trained_folds"]
    def forbidden(*args, **kwargs):
        raise AssertionError("A valid cached fold must not be fitted again.")
    monkeypatch.setattr(policy, "fit_means", forbidden)
    second = policy.run_replay(original, source, folds, settings, load_fit=lambda day: deepcopy(cache.get(day)))
    pd.testing.assert_frame_equal(first["predictions"], second["predictions"], check_exact=True)
    assert second["audit"]["cached_folds"] == len(cache)
    cache[next(iter(cache))]["fit_cutoff"] += pd.Timedelta(days=1)
    with pytest.raises(policy.RMSEPolicyError, match="Cached fit metadata"):
        policy.run_replay(original, source, folds, settings, load_fit=lambda day: deepcopy(cache.get(day)))


def test_strict_365_mode_falls_back_without_training(synthetic, monkeypatch):
    original, source, folds, settings = synthetic
    def forbidden(*args, **kwargs):
        raise AssertionError("Insufficient strict training prefix must not fit.")
    monkeypatch.setattr(policy, "fit_means", forbidden)
    result = policy.run_replay(original, source, folds, settings, {"require_full_training_history": True})
    assert not result["predictions"].expert_ready.any()
    assert result["predictions"].source_expert_ready.any()
    assert result["audit"]["trained_folds"] == 0
    assert result["predictions"].nyx_rmse.equals(original.forecast.rename("nyx_rmse"))


@pytest.mark.parametrize("missing_eligible_hour", [False, True])
def test_strict_365_requires_all_physical_hours_not_only_calendar_span(synthetic, missing_eligible_hour):
    settings = parameters()
    data = policy.base._prepare(panel(days=366, zones=("FR", "DE")), settings)
    if missing_eligible_hour:
        data.loc[0, "_features_valid"] = False
    day = "2027-01-01"
    cutoff = pd.Timestamp("2026-12-31 08:00", tz="Europe/Paris").tz_convert("UTC")
    split = "2026-12-04"
    train = data.loc[data._day.lt(day) & data._features_valid & data._label_valid & data.label_available_at_utc.le(cutoff)]
    core = train.loc[train._day.lt(split)]
    thresholds = {zone: max(50., float(core.loc[core.zone.eq(zone), "_error"].quantile(.95))) for zone in ("FR", "DE")}
    frozen = {"fit_day": day, "fit_cutoff_utc": cutoff, "training_rows": len(train),
        "training_start_day": "2026-01-01", "training_days": 365, "full_365_day_training": True,
        "calibration_start_day": split, "signed_model_training_rows": len(core),
        "status": "trained", "reason": "", "thresholds_eur_mwh": thresholds}
    options = models.validate_options({"require_full_training_history": True})
    state, record = policy._fit(data, policy.base._matrix(data, NAMES, ("DE", "FR")),
        day, cutoff, settings, ("DE", "FR"), frozen, options, None, None)
    assert record["full_365_calendar_span"]
    assert record["complete_365_eligible_history"] == (not missing_eligible_hour)
    assert (state is None) == missing_eligible_hour
    assert record["status"] == ("fallback" if missing_eligible_hour else "trained")


def test_current_and_future_labels_cannot_change_prior_predictions_or_governance(synthetic):
    original, source, folds, settings = synthetic
    options = {"governance": {"minimum_days": 2, "minimum_changed_days": 1, "uncertainty_z": 0}}
    first = policy.run_replay(original, source, folds, settings, options)
    day = "2026-04-11"
    civil_days = original.timestamp_utc.dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")
    changed_original, changed_source = original.copy(), source.copy()
    changed_original.loc[civil_days.ge(day), "actual"] = -7000.
    changed_source.loc[civil_days.ge(day), "actual"] = -7000.
    second = policy.run_replay(changed_original, changed_source, folds, settings, options)
    columns = ["nyx_rmse", "residual_mse_direct", "mixture_mean_direct", "mixture_mean_governed",
               "residual_mse_selected_weight", "mixture_mean_selected_weight"]
    pd.testing.assert_frame_equal(first["predictions"].loc[civil_days.le(day), columns],
                                  second["predictions"].loc[civil_days.le(day), columns], check_exact=True)
    a, b = first["governance"], second["governance"]
    pd.testing.assert_frame_equal(a.loc[a.delivery_day.le(day)], b.loc[b.delivery_day.le(day)], check_exact=True)


@pytest.mark.parametrize("column", ["training_rows", "signed_model_training_rows", "calibration_start_day", "thresholds_eur_mwh"])
def test_changed_training_core_is_rejected(synthetic, column):
    original, source, folds, settings = synthetic
    folds = folds.copy(deep=True)
    at = folds.loc[folds.status.eq("trained")].index[0]
    if column == "thresholds_eur_mwh":
        folds.at[at, column] = {"FR": 50., "DE": 50.}
    elif column == "calibration_start_day":
        folds.at[at, column] = "2026-01-01"
    else:
        folds.at[at, column] += 1
    with pytest.raises(policy.RMSEPolicyError):
        policy.run_replay(original, source, folds, settings)


def test_delayed_training_label_rejects_changed_frozen_membership(synthetic):
    original, source, folds, settings = synthetic
    original, source = original.copy(), source.copy()
    original.loc[0, "label_available_at_utc"] = pd.Timestamp("2027-01-01", tz="UTC")
    source.loc[0, "label_available_at_utc"] = pd.Timestamp("2027-01-01", tz="UTC")
    with pytest.raises(policy.RMSEPolicyError, match="core differs"):
        policy.run_replay(original, source, folds, settings)


def governance_days(first="2026-03-28", days=3, error=20., correction=10.):
    start = pd.Timestamp(first, tz="Europe/Paris")
    stamps = pd.date_range(start, start+pd.DateOffset(days=days), freq="h", inclusive="left")
    civil = stamps.tz_localize(None).normalize()
    end = (civil+pd.Timedelta(days=1)).tz_localize("Europe/Paris")
    begin = civil.tz_localize("Europe/Paris")
    return pd.DataFrame({"zone": "FR", "_day": civil.strftime("%Y-%m-%d"),
        "_physical_day_hours": (end-begin).total_seconds()/3600.,
        "_label_valid": True, "expert_ready": True,
        "label_available_at_utc": (civil-pd.Timedelta(days=1)+pd.Timedelta(hours=18)).tz_localize("Europe/Paris").tz_convert("UTC"),
        "_error": error, "threshold_eur_mwh": 100.,
        "residual_mse_bounded_correction": correction,
        "mixture_mean_bounded_correction": correction})


@pytest.mark.parametrize("first,expected", [("2026-03-28", 71), ("2026-10-24", 73)])
def test_governor_uses_complete_dst_days_and_rejects_late_labels(first, expected):
    past = governance_days(first)
    day = (pd.Timestamp(first)+pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    cutoff = (pd.Timestamp(day)-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    complete = policy._eligible([past], "FR", day, cutoff, 90)
    assert len(complete) == expected
    missing = policy._eligible([past.iloc[1:]], "FR", day, cutoff, 90)
    assert len(missing) == expected-24
    delayed = past.copy()
    delayed.loc[0, "label_available_at_utc"] = cutoff+pd.Timedelta(seconds=1)
    assert len(policy._eligible([delayed], "FR", day, cutoff, 90)) == expected-24


def test_governor_selects_signed_mse_gain_and_ties_fall_back():
    past = governance_days(error=-20., correction=-10.)
    governance = models.validate_options({"governance": {"minimum_days": 2, "minimum_changed_days": 1}})["governance"]
    cutoff = pd.Timestamp("2026-04-01", tz="UTC")
    weight, _, records = policy._govern(past, "residual_mse", "FR", "2026-04-01", cutoff, governance)
    assert weight == 1.
    best = next(row for row in records if row["selected"])
    assert best["mse_gain"] == 300.
    assert best["mae_gain"] == 10.
    past["residual_mse_bounded_correction"] = 0.
    weight, _, records = policy._govern(past, "residual_mse", "FR", "2026-04-01", cutoff, governance)
    assert weight == 0
    assert sum(row["selected"] for row in records) == 1


def test_ordinary_mae_guard_can_block_large_tail_mse_gain():
    past = governance_days(error=0., correction=10.)
    # A few giant misses make a blanket positive shift improve MSE and even
    # pooled MAE, while ordinary hours are worsened and must still veto it.
    for _, indices in past.groupby("_day").groups.items():
        chosen = list(indices)[:10]
        past.loc[chosen, "_error"] = 1000.
        past.loc[chosen, "residual_mse_bounded_correction"] = 500.
    governance = models.validate_options({"governance": {"minimum_days": 2, "minimum_changed_days": 1}})["governance"]
    weight, _, records = policy._govern(past, "residual_mse", "FR", "2026-04-01", pd.Timestamp("2026-04-01", tz="UTC"), governance)
    assert weight == 0
    assert all("ordinary_mae_guard" in row["reason"] for row in records if row["weight"])
    assert all(row["mse_gain"] > 0 and row["mae_gain"] > 0 for row in records if row["weight"])


def test_cluster_standard_error_matches_unequal_day_weight_formula():
    past = governance_days()
    past.loc[past._day.eq("2026-03-29"), "_error"] = 50.
    governance = models.validate_options({"governance": {"minimum_days": 2, "minimum_changed_days": 1}})["governance"]
    _, _, records = policy._govern(past, "residual_mse", "FR", "2026-04-01", pd.Timestamp("2026-04-01", tz="UTC"), governance)
    one = next(row for row in records if row["weight"] == 1)
    gain = np.array([300., 900., 300.])
    counts = np.array([24., 23., 24.])
    average = np.dot(gain, counts)/counts.sum()
    expected = np.sqrt(3/2*np.square(counts*(gain-average)).sum())/counts.sum()
    assert one["mse_gain_standard_error"] == pytest.approx(expected)


@pytest.mark.parametrize("mutation", ["probability", "fit_day", "readiness", "duplicate"])
def test_bad_detector_inputs_rejected(synthetic, mutation):
    original, source, folds, settings = synthetic
    source = source.copy()
    at = source.index[source.expert_ready][0]
    if mutation == "probability":
        # Keep real validation when the cheap fake inference is used.
        source.at[at, "threshold_eur_mwh"] = 0.
    elif mutation == "fit_day":
        source.at[at, "expert_fit_day"] = "2027-01-01"
    elif mutation == "readiness":
        source.at[at, "expert_ready"] = False
    else:
        folds = pd.concat([folds, folds.iloc[[0]]], ignore_index=True)
    with pytest.raises(policy.RMSEPolicyError):
        policy.run_replay(original, source, folds, settings)
