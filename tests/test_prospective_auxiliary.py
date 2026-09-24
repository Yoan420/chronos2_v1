from dataclasses import replace
from datetime import timedelta
import json

import numpy as np
import pandas as pd
import pytest

import chronos2_exogenous.prospective_auxiliary as trial


TZ = "Europe/Paris"


def history(start="2025-09-03", days=365, *, bias=3.0):
    first = pd.Timestamp(start, tz=TZ)
    index = pd.date_range(first, first + pd.DateOffset(days=days), freq="h", inclusive="left").tz_convert("UTC")
    local = index.tz_convert(TZ)
    median = 50 + 4 * np.sin(2 * np.pi * local.hour / 24)
    frame = pd.DataFrame({"delivery_start_utc": index, "actual": median + bias})
    for q, offset in zip(trial.QUANTILES, (-7, 0, 9)):
        frame[f"{trial.RAW_MODEL}__{q}"] = median + offset
    for i, name in enumerate(trial.MARKET_COLUMNS):
        frame[name] = 10 + i + np.cos(2 * np.pi * local.hour / 24)
    return frame


def test_exact_ridge_audit_and_application():
    raw = history(days=30)
    original = raw.copy(deep=True)
    fitted = trial.fit_residual_corrector(raw, target_day="2025-10-03")
    assert fitted.training_days == 30
    assert fitted.training_hours == 720
    assert not fitted.identity_cold_start
    assert fitted.coefficients[0] == pytest.approx(3)
    assert fitted.coefficients[1:] == pytest.approx([0, 0, 0, 0], abs=1e-12)
    assert fitted.feature_means[0] == 0
    assert fitted.feature_scales[0] == 1
    audit = fitted.to_audit()
    assert audit["training_end_day"] == "2025-10-02"
    assert audit["target_observations_used"] == 0
    assert not audit["neural_oof"]
    json.dumps(audit, allow_nan=False)
    predicted = trial.apply_residual_corrector(history("2025-10-03", 1), fitted)
    np.testing.assert_allclose(predicted["residual_correction"], 3)
    assert "actual" not in predicted
    for q in trial.QUANTILES:
        np.testing.assert_allclose(predicted[f"{trial.RESIDUAL_MODEL}__{q}"], predicted[f"{trial.RAW_MODEL}__{q}"] + 3)
    pd.testing.assert_frame_equal(raw, original)


@pytest.mark.parametrize("recipe", [trial.ResidualRecipe(ridge_alpha=2), trial.ResidualRecipe(maximum_shift_eur_mwh=30),
                                      trial.ResidualRecipe(minimum_training_days=10), trial.ResidualRecipe(lookback_days=30),
                                      trial.ResidualRecipe(ridge_alpha=True), trial.ResidualRecipe(lookback_days=365.0)])
def test_recipe_cannot_be_tuned_silently(recipe):
    with pytest.raises(trial.ProspectiveAuxiliaryError):
        trial.fit_residual_corrector(history(days=30), target_day="2025-10-03", recipe=recipe)


def test_cold_start_identity_then_prequential_fit():
    raw = history(days=32)
    corrected, audits = trial.build_prequential_residual_history(raw)
    assert len(audits) == 32
    assert all(a["identity_cold_start"] for a in audits[:30])
    np.testing.assert_array_equal(corrected["residual_correction"].iloc[:720], 0)
    np.testing.assert_allclose(corrected["residual_correction"].iloc[720:], 3)
    assert audits[0]["training_days"] == 0
    assert audits[30]["training_days"] == 30
    assert audits[31]["training_end_day"] == "2025-10-03"
    assert "actual" in corrected


def test_prequential_future_label_perturbation_does_not_change_earlier_predictions():
    raw = history(days=34)
    changed = raw.copy()
    changed.loc[changed.index >= 31 * 24, "actual"] += 200
    before, audits_before = trial.build_prequential_residual_history(raw)
    after, audits_after = trial.build_prequential_residual_history(changed)
    np.testing.assert_array_equal(before["residual_correction"].iloc[:32 * 24], after["residual_correction"].iloc[:32 * 24])
    assert audits_before[:32] == audits_after[:32]
    assert not np.array_equal(before["residual_correction"].iloc[32 * 24:], after["residual_correction"].iloc[32 * 24:])


@pytest.mark.parametrize("bad", ["gap_hour", "gap_day", "duplicate", "infinite", "quantile_cross", "naive", "wrong_origin"])
def test_bad_history_is_rejected(bad):
    frame = history(days=32)
    if bad == "gap_hour":
        frame = frame.drop(index=20)
    elif bad == "gap_day":
        frame = frame.drop(index=range(24, 48))
    elif bad == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif bad == "infinite":
        frame.loc[10, "actual"] = np.inf
    elif bad == "quantile_cross":
        frame.loc[10, f"{trial.RAW_MODEL}__q10"] = 100000
    elif bad == "naive":
        frame["delivery_start_utc"] = frame["delivery_start_utc"].dt.tz_localize(None)
    else:
        frame["forecast_origin_utc"] = frame["delivery_start_utc"]
    with pytest.raises(trial.ProspectiveAuxiliaryError):
        trial.build_prequential_residual_history(frame)


def test_same_day_actual_and_stale_history_are_rejected():
    raw = history(days=365)
    with pytest.raises(trial.ProspectiveAuxiliaryError, match="exclure"):
        trial.fit_residual_corrector(raw, target_day="2026-09-02")
    with pytest.raises(trial.ProspectiveAuxiliaryError, match="D-1"):
        trial.fit_residual_corrector(raw, target_day="2026-09-04")


def test_strict_365_future_boundary_and_capped_lookback():
    with pytest.raises(trial.ProspectiveAuxiliaryError, match="365"):
        trial.fit_residual_corrector(history(days=364), target_day="2026-09-02", require_full_window=True)
    exact = trial.fit_residual_corrector(history(days=365), target_day="2026-09-03", require_full_window=True)
    assert exact.training_days == 365
    assert exact.training_hours == 8760
    extra = history("2025-09-02", 366)
    extra.loc[extra.index < 24, "actual"] += 1e6
    capped = trial.fit_residual_corrector(extra, target_day="2026-09-03", require_full_window=True)
    assert capped == exact


@pytest.mark.parametrize(("start", "target", "hours"), [("2026-02-27", "2026-03-29", 23),
                                                         ("2025-09-26", "2025-10-26", 25)])
def test_dst_physical_horizon(start, target, hours):
    fit = trial.fit_residual_corrector(history(start, 30), target_day=target)
    forecast = history(target, 1)
    result = trial.apply_residual_corrector(forecast, fit)
    assert len(result) == hours
    assert not result["delivery_start_utc"].duplicated().any()
    assert np.isfinite(result["residual_correction"]).all()


def test_shift_clipped_and_fit_day_is_bound():
    fit = trial.fit_residual_corrector(history(days=30, bias=300), target_day="2025-10-03")
    np.testing.assert_allclose(trial.apply_residual_corrector(history("2025-10-03", 1), fit)["residual_correction"], 20)
    with pytest.raises(trial.ProspectiveAuxiliaryError, match="unique jour"):
        trial.apply_residual_corrector(history("2025-10-04", 1), fit)
    with pytest.raises(trial.ProspectiveAuxiliaryError, match="Parametres"):
        trial.apply_residual_corrector(history("2025-10-03", 1), replace(fit, feature_scales=(0,) * 5))


def test_future_actual_is_not_even_converted():
    class Poison:
        def __float__(self):
            raise AssertionError("Future actual must never be read")
    fit = trial.fit_residual_corrector(history(days=30), target_day="2025-10-03")
    future = history("2025-10-03", 1)
    without = trial.apply_residual_corrector(future.drop(columns="actual"), fit)
    future["actual"] = [Poison()] * len(future)
    with_poison = trial.apply_residual_corrector(future, fit)
    pd.testing.assert_frame_equal(without, with_poison)


def test_timezone_and_timestamp_units_preserve_identical_physical_hours():
    raw = history(days=30)
    expected = trial.fit_residual_corrector(raw, target_day="2025-10-03")
    raw["delivery_start_utc"] = raw["delivery_start_utc"].dt.tz_convert(TZ).dt.as_unit("us")
    assert trial.fit_residual_corrector(raw, target_day="2025-10-03") == expected


def test_duplicate_columns_are_rejected():
    raw = history(days=30)
    raw = pd.concat([raw, raw[["actual"]]], axis=1)
    with pytest.raises(trial.ProspectiveAuxiliaryError, match="Colonnes dupliquees"):
        trial.fit_residual_corrector(raw, target_day="2025-10-03")


def fake_kalman_fit(**kwargs):
    training = kwargs["training_frame"]
    target = kwargs["target_block"]
    assert len(set(training["_local_day"])) == 365
    assert "actual" not in target
    assert target.index.min() > training.index.max()
    assert set(kwargs["candidate_feature_columns"]) == set(trial.STANDARD_CANDIDATES)
    assert "residual_load_mean" in kwargs["covariate_columns"]
    prediction = pd.DataFrame(index=target.index)
    prediction.index.name = "delivery_start_utc"
    for q in trial.QUANTILES:
        prediction[f"{trial.KALMAN_MODEL}__{q}"] = target[f"{trial.RESIDUAL_MODEL}__{q}"] + 0.5
    return {"predictions": prediction, "daily_audit": {"target_observations_assimilated": 0},
            "window_audit": {"training_window_days": 365}, "state_audit": [], "market_scalers": {}}


def test_chains_use_prequential_history_and_same_corrected_upstream(monkeypatch):
    from chronos2_hourly import kalman_residual as engine
    observed = {}
    def fit(**kwargs):
        observed.update(kwargs)
        return fake_kalman_fit(**kwargs)
    monkeypatch.setattr(engine, "_fit_rolling_target_day", fit)
    raw = history()
    future = history("2026-09-03", 1)
    result = trial.forecast_trial_chains(raw, future)
    assert result.audit["residual_fit"]["training_days"] == 365
    assert result.audit["kalman_window_audit"]["training_window_days"] == 365
    assert result.audit["production_pit_evidence"] is False
    assert result.audit["promotion_eligible"] is False
    assert result.audit["neural_oof"] is False
    assert result.audit["historical_metrics_are_independent_test"] is False
    json.dumps(result.audit, allow_nan=False)
    np.testing.assert_array_equal(observed["training_frame"]["residual_correction"].iloc[:720], 0)
    np.testing.assert_allclose(result.corrected_future["residual_correction"], 3)
    for q in trial.QUANTILES:
        np.testing.assert_allclose(result.kalman_future[f"{trial.KALMAN_MODEL}__{q}"], result.corrected_future[f"{trial.RESIDUAL_MODEL}__{q}"] + 0.5)
    assert "actual" not in result.corrected_future
    assert "actual" not in result.kalman_future
    assert len(result.prequential_history) == len(raw)


def test_chain_ignores_future_actual_mutation(monkeypatch):
    from chronos2_hourly import kalman_residual as engine
    monkeypatch.setattr(engine, "_fit_rolling_target_day", fake_kalman_fit)
    raw, future = history(), history("2026-09-03", 1)
    first = trial.forecast_trial_chains(raw, future)
    future["actual"] = np.arange(len(future)) * 100000
    second = trial.forecast_trial_chains(raw, future)
    pd.testing.assert_frame_equal(first.kalman_future, second.kalman_future)
    pd.testing.assert_frame_equal(first.prequential_history, second.prequential_history)
    assert first.audit == second.audit


def test_chain_rejects_missing_market_input_and_nonstandard_bank(monkeypatch):
    from chronos2_hourly.kalman_residual import KalmanResidualConfig
    with pytest.raises(trial.ProspectiveAuxiliaryError, match="cinq"):
        trial.forecast_trial_chains(history(), history("2026-09-03", 1),
                                    kalman_config=KalmanResidualConfig(candidate_kinds=("linear_bias",)))
    with pytest.raises(trial.ProspectiveAuxiliaryError, match="Covariables"):
        trial.forecast_trial_chains(history(), history("2026-09-03", 1).drop(columns=trial.MARKET_COLUMNS[0]))


@pytest.mark.parametrize("candidate_kinds", [("linear_bias",), trial.STANDARD_CANDIDATES])
def test_real_kalman_primitive_two_days_is_finite_and_excludes_actual(candidate_kinds):
    pytest.importorskip("pykalman")
    from chronos2_hourly.kalman_residual import KalmanResidualConfig
    # Small numerical primitive test, not an exception to the public 365-day contract.
    prior, _ = trial.build_prequential_residual_history(history(days=2))
    fitted = trial.fit_residual_corrector(history(days=2), target_day="2025-09-05")
    future = trial.apply_residual_corrector(history("2025-09-05", 1), fitted)
    result = trial._fit_kalman_future(prior, future, timezone=TZ, config=KalmanResidualConfig(
        candidate_kinds=candidate_kinds, governance_lookback_days=2, governance_minimum_days=1))
    assert result["window_audit"]["training_window_days"] == 2
    assert result["window_audit"]["target_observations_assimilated"] == 0
    assert np.isfinite(result["predictions"][[f"{trial.KALMAN_MODEL}__{q}" for q in trial.QUANTILES]]).all().all()
    audit = trial._audit_values({name: result[name] for name in ("state_audit", "daily_audit", "market_scalers")})
    json.dumps(audit, allow_nan=False)
    assert audit["state_audit"][0]["mean_innovation"] is None
