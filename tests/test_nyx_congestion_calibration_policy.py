"""Causal, partition-level tests; estimator fitting is mocked for fast checks."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from nyx_congestion_calibration import policy


DAY = "2026-09-14"
CUTOFF = pd.Timestamp("2026-09-13 06:00Z")
ZONES = ("FR", "DE", "BE", "NL")
PARAMETERS = {"minimum_threshold_eur_mwh": 50., "threshold_quantile": .95, "threads": 1}


@pytest.fixture
def history():
    rows = []
    for d, day in enumerate(pd.date_range("2026-03-01", periods=205, freq="D")):
        for z, zone in enumerate(ZONES):
            for hour in range(6):
                rows.append({"_day": str(day.date()), "zone": zone, "_error": 100. if (d+hour) % 2 else 0.,
                    "timestamp_utc": (day.tz_localize("Europe/Paris")+pd.Timedelta(hours=hour)).tz_convert("UTC"),
                    "_features_valid": True, "_label_valid": True,
                    "label_available_at_utc": (day-pd.Timedelta(days=1)+pd.Timedelta(hours=14)).tz_localize("Europe/Paris").tz_convert("UTC"),
                    policy.FUEL: 100., "feature_test": len(rows),
                    **{name: (z+1)*10+hour for name in policy.SIGNALS}})
    return pd.DataFrame(rows)


def install_estimators(monkeypatch):
    calls = []
    def classifier(X, y, V, cy, **kwargs):
        calls.append(dict(kind="classifier", X=X.copy(), y=y.copy(), V=V.copy(), cy=cy.copy(), **kwargs))
        return {"calibration": {"coefficients": np.array([1., 0.]), "support": {
            z: {"status": "regularized_sparse_support"} for z in ZONES}}}
    def cdf(X, errors, thresholds, fuel, zones, **kwargs):
        calls.append(dict(kind="cdf", X=X.copy(), errors=errors.copy(), thresholds=thresholds.copy(),
                          fuel=fuel.copy(), zones=zones.copy()))
        return {"fake": "cdf"}
    monkeypatch.setattr(policy, "fit_probability", classifier)
    monkeypatch.setattr(policy, "fit_physical_cdf", cdf)
    return calls


def fit(history, *, load=None, save=None):
    return policy._fit(history, DAY, CUTOFF, PARAMETERS, ZONES, ["feature_test"], load, save, None)


def test_fixed90_classifier_and_fixed28_severity_partitions_are_identical_between_variants(history, monkeypatch):
    calls = install_estimators(monkeypatch)
    state, record = fit(history)
    assert record["status"] == "trained"
    assert record["calibration_window_days"] == 90 and record["severity_exclusion_days"] == 28
    control, severity, enriched, enriched_severity = calls
    for key in ("y", "cy"):
        np.testing.assert_array_equal(control[key], enriched[key])
    for key in ("X", "V"):
        np.testing.assert_array_equal(control[key][:, 0], enriched[key][:, 0])
    np.testing.assert_array_equal(severity["X"][:, 0], enriched_severity["X"][:, 0])
    core_ids, cal_ids, severity_ids = (set(control["X"][:, 0]), set(control["V"][:, 0]), set(severity["X"][:, 0]))
    assert core_ids.isdisjoint(cal_ids)
    assert core_ids.issubset(severity_ids) and cal_ids.intersection(severity_ids)
    core_days = history.loc[history.feature_test.isin(core_ids), "_day"]
    cal_days = history.loc[history.feature_test.isin(cal_ids), "_day"]
    severity_days = history.loc[history.feature_test.isin(severity_ids), "_day"]
    assert core_days.max() < "2026-06-16" <= cal_days.min()
    assert severity_days.max() < "2026-08-17"
    assert enriched["X"].shape[1]-control["X"].shape[1] == len(policy.SIGNALS)
    assert state["thresholds"] == {z: 100. for z in ZONES}
    for c in (severity, enriched_severity):
        np.testing.assert_array_equal(c["thresholds"], np.full(len(c["X"]), 100.))


def test_thresholds_are_core90_only_even_when_recent_severity_errors_are_huge(history, monkeypatch):
    calls = install_estimators(monkeypatch)
    history.loc[history._day.ge("2026-06-16"), "_error"] = 100000.
    state, record = fit(history)
    assert record["status"] == "trained" and state["thresholds"] == {z: 100. for z in ZONES}
    assert calls[1]["errors"].max() == 100000.
    assert np.all(calls[1]["thresholds"] == 100.)


def test_sparse_calibration_with_zero_positive_events_no_longer_abstains(history, monkeypatch):
    calls = install_estimators(monkeypatch)
    history.loc[history._day.ge("2026-06-16"), "_error"] = 0.
    state, record = fit(history)
    assert state is not None and record["status"] == "trained"
    assert record["calibration_tail_rows"] == 0 and record["calibration_positive_days"] == 0
    assert not calls[0]["cy"].any() and calls[0]["y"].any()


def test_future_delayed_and_outside365_labels_cannot_change_any_fitted_component(history, monkeypatch):
    delayed = history.index[history._day.isin(["2026-04-01", "2026-07-01", "2026-09-01"])]
    history.loc[delayed, "label_available_at_utc"] = CUTOFF+pd.Timedelta(hours=1)
    old = history.iloc[:24].copy()
    old["_day"] = "2025-08-01"
    old["feature_test"] = np.arange(-24, 0)
    old["_error"] = 1000000.
    history = pd.concat([old, history], ignore_index=True)
    before_calls = install_estimators(monkeypatch)
    before_state, before_record = fit(history)
    changed = history.copy(deep=True)
    excluded = changed._day.ge(DAY) | changed.label_available_at_utc.gt(CUTOFF) | changed._day.lt("2025-09-14")
    changed.loc[excluded, "_error"] = 999999.
    changed.loc[excluded, "feature_test"] = -888888.
    after_calls = install_estimators(monkeypatch)
    after_state, after_record = fit(changed)
    assert before_record == after_record
    assert before_state["partitions_sha256"] == after_state["partitions_sha256"]
    for before, after in zip(before_calls, after_calls):
        for key, value in before.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, after[key])
            else:
                assert value == after[key]
    used = set(before_calls[0]["X"][:, 0]) | set(before_calls[0]["V"][:, 0]) | set(before_calls[1]["X"][:, 0])
    assert -888888. not in used and not used.intersection(range(-24, 0))


def test_insufficient_classifier_core_still_abstains(history, monkeypatch):
    calls = install_estimators(monkeypatch)
    history.loc[history._day.lt("2026-06-16"), "_error"] = 0.
    state, record = fit(history)
    assert state is None and not calls
    assert record["reason"] == "insufficient_classifier_or_severity_events"


def test_minimum_calibration_days_apply_to_every_country(history, monkeypatch):
    calls = install_estimators(monkeypatch)
    insufficient = history.zone.eq("FR") & history._day.ge("2026-06-16") & ~history._day.eq("2026-09-01")
    history.loc[insufficient, "_features_valid"] = False
    state, record = fit(history)
    assert record["calibration_days_by_zone"]["FR"] == 1
    assert state is None and not calls


def test_missing_core_feature_or_label_or_nonpositive_fuel_never_enters_fit(history, monkeypatch):
    calls = install_estimators(monkeypatch)
    masked = history.index[:4]
    history.loc[masked[0], "_features_valid"] = False
    history.loc[masked[1], "_label_valid"] = False
    history.loc[masked[2], policy.FUEL] = 0.
    history.loc[masked[3], policy.FUEL] = np.nan
    state, record = fit(history)
    assert record["status"] == "trained"
    assert not set(calls[0]["X"][:, 0]).intersection(masked)
    assert not set(calls[1]["X"][:, 0]).intersection(masked)


def test_cache_reuses_only_identical_partition_and_rejects_changed_label(history, monkeypatch):
    install_estimators(monkeypatch)
    saved = []
    state, _ = fit(history, save=lambda day, value: saved.append((day, value)))
    assert saved[0][0] == DAY and saved[0][1] is state
    monkeypatch.setattr(policy, "fit_probability", lambda *a, **k: pytest.fail("Identical fit recomputed"))
    restored, record = fit(history, load=lambda day: state)
    assert restored is state and record["cache_reused"]
    changed = history.copy(deep=True)
    changed.loc[changed._day.eq("2026-07-01"), "_error"] += .001
    with pytest.raises(ValueError, match="chronological split"):
        fit(changed, load=lambda day: state)


def test_optimizer_failure_does_not_save_a_partial_model(history, monkeypatch):
    install_estimators(monkeypatch)
    def failed(*args, **kwargs):
        raise policy.CalibrationError("numerical test")
    monkeypatch.setattr(policy, "fit_probability", failed)
    saved = []
    state, record = fit(history, save=lambda *a: saved.append(a))
    assert state is None and not saved
    assert record["reason"] == "calibration_optimization_failed"


def test_prediction_keeps_probability_and_distribution_median_distinct(monkeypatch):
    current = pd.DataFrame({"zone": ["DE", "FR"], "feature_test": [1., 2.],
        "timestamp_utc": pd.to_datetime(["2026-09-14 17:00Z"] * 2),
        "forecast_origin_utc": [CUTOFF] * 2, "congestion_ready": True,
        "forecast": 100., "q10": 80., "q90": 120., policy.FUEL: 100.})
    state = {"fit_day": DAY, "fit_cutoff": CUTOFF, "zones": ZONES,
        "thresholds": {z: 50. for z in ZONES}, "priors": {z: .2 for z in ZONES},
        "models": {"congestion": {"features": ["feature_test"], "cdf": {}, "classifier": {
            "calibration": {"support": {z: {"status": "regularized_sparse_support"} for z in ZONES}}}}}}
    monkeypatch.setattr(policy, "predict_probability", lambda *a: (np.array([.3, .1]), np.array([.6, .2])))
    monkeypatch.setattr(policy, "predict_physical_cdf", lambda *a: (np.array([[-10., 40., 80.]]), {}))
    p, correction, thresholds, detail = policy._predict(state, current, "congestion")
    np.testing.assert_array_equal(correction, [40., 0.])
    np.testing.assert_array_equal(p, [.3, .1])
    assert correction[0] != p[0] * 40.
    assert detail.raw_probability.tolist() == [.6, .2]
    assert detail.calibration_status.eq("regularized_sparse_support").all()
    assert detail.cdf_evaluated.tolist() == [True, False]
    invalid = deepcopy(state)
    invalid["fit_cutoff"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="causal fitted state"):
        policy._predict(invalid, current, "congestion")
