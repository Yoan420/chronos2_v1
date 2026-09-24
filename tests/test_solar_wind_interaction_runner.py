"""Fast ablation-runner contract tests; no forecasting model is executed."""
from __future__ import annotations

from contextlib import nullcontext
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import chronos2_hourly.solar_wind_interaction as subject


def _frame():
    index = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC", name="delivery_start_utc")
    median = np.array([100.0, 250.0, 180.0, 350.0])
    return pd.DataFrame({
        "actual": [100.0, 240.0, 220.0, 320.0],
        "chronos2__q50": median - 2,
        "residual_corrected__q10": median - 12,
        "residual_corrected__q50": median - 2,
        "residual_corrected__q90": median + 8,
        "residual_correction": np.zeros(4),
        "residual_kalman__q10": median - 10,
        "residual_kalman__q50": median,
        "residual_kalman__q90": median + 10,
        "kalman_raw_correction": np.full(4, 2.0),
        "kalman_correction": np.full(4, 2.0),
        "kalman_weight": np.ones(4),
        "kalman_selected_filter": ["linear_market"] * 4,
    }, index=index)


def _identity():
    return {"run_id": "test-identity", "feature": "de_low_wind_solar_stress"}


def _fake_prepared():
    return (None, None, None, None, None, None, _identity())


def test_indexed_returns_detached_physical_grid():
    source = _frame().reset_index()
    retained = source.copy(deep=True)
    result = subject.indexed(source)
    result.iloc[0, 0] = -1
    pd.testing.assert_frame_equal(source, retained)
    assert str(result.index.tz) == "UTC"
    assert result.index.name == "delivery_start_utc"


def test_restore_audit_covariate_config_preserves_exact_order_and_values():
    original = subject.KalmanCovariateConfig()
    serialized = json.loads(json.dumps(original.to_dict()))
    retained = json.loads(json.dumps(serialized))
    restored = subject.restore_covariate_config(serialized)
    assert restored == original
    assert restored.input_columns == original.input_columns
    assert restored.feature_columns == original.feature_columns
    assert restored.derived == original.derived
    assert serialized == retained
    assert subject.json_bytes(restored.to_dict()) == subject.json_bytes(original.to_dict())


def test_safe_path_rejects_output_outside_experiment(monkeypatch, tmp_path):
    monkeypatch.setattr(subject, "OUTPUT", tmp_path / "experiment")
    with pytest.raises(ValueError, match="Unsafe or redirected"):
        subject.safe_path(tmp_path / "production" / "report.html")


def test_safe_path_rejects_windows_reparse_descendant(monkeypatch, tmp_path):
    child = tmp_path / "candidate_cache"
    child.mkdir()
    monkeypatch.setattr(subject, "OUTPUT", tmp_path)
    original_lstat = subject.Path.lstat
    def reparse_lstat(path):
        stat = original_lstat(path)
        if path == child:
            return SimpleNamespace(st_file_attributes=0x400, st_mode=stat.st_mode)
        return stat
    monkeypatch.setattr(subject.Path, "lstat", reparse_lstat)
    with pytest.raises(ValueError, match="Redirected output descendant"):
        subject.safe_path(tmp_path)


@pytest.mark.parametrize("order", [[0, 0, 2, 3], [1, 0, 2, 3]])
def test_indexed_rejects_duplicate_or_unordered_hours(order):
    with pytest.raises(ValueError, match="Unique increasing"):
        subject.indexed(_frame().iloc[order])


@pytest.mark.parametrize("column", ["actual", "chronos2__q50", "residual_corrected__q50", "residual_correction"])
def test_upstream_must_be_bitwise_unchanged(column):
    baseline = _frame()
    candidate = baseline.copy(deep=True)
    candidate.loc[candidate.index[0], column] += 1e-10
    with pytest.raises(ValueError, match="Upstream modified"):
        subject.check_same_upstream(baseline, candidate)


def test_upstream_alignment_must_not_change():
    with pytest.raises(ValueError, match="alignment"):
        subject.check_same_upstream(_frame(), _frame().iloc[1:])


@pytest.mark.parametrize("column", ["forecast_origin_utc", "residual_corrected_forecast_origin_utc"])
def test_upstream_forecast_origins_must_not_change(column):
    baseline = _frame()
    baseline[column] = baseline.index - pd.Timedelta(days=1)
    candidate = baseline.copy(deep=True)
    subject.check_same_upstream(baseline, candidate)
    candidate.loc[candidate.index[0], column] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="Forecast origin modified"):
        subject.check_same_upstream(baseline, candidate)


def test_future_forecast_accepts_no_actual_or_all_missing_actual():
    frame = _frame().drop(columns="actual")
    subject.validate_forecast(frame)
    frame["actual"] = np.nan
    subject.validate_forecast(frame)


@pytest.mark.parametrize("kind", ["actual", "nonfinite", "crossing"])
def test_future_forecast_rejects_actuals_and_invalid_quantiles(kind):
    frame = _frame().drop(columns="actual")
    if kind == "actual":
        frame["actual"] = [np.nan, np.nan, 0.0, np.nan]
    else:
        frame.loc[frame.index[0], subject.QUANTILES[0]] = np.nan if kind == "nonfinite" else 500
    with pytest.raises(ValueError, match="Future"):
        subject.validate_forecast(frame)


def test_baseline_control_compares_quantiles_correction_and_governance():
    baseline = _frame()
    subject.compare_replay(baseline.copy(deep=True), baseline)
    for column in subject.QUANTILES + ["kalman_correction", "kalman_raw_correction", "kalman_weight"]:
        candidate = baseline.copy(deep=True)
        candidate.loc[candidate.index[0], column] += 1e-6
        with pytest.raises(ValueError, match="Unmodified Kalman replay differs"):
            subject.compare_replay(candidate, baseline)
    candidate = baseline.copy(deep=True)
    candidate.loc[candidate.index[0], "kalman_selected_filter"] = "linear_bias"
    with pytest.raises(ValueError, match="governance selection"):
        subject.compare_replay(candidate, baseline)


def test_evaluation_keeps_paired_actuals_and_counts_false_spikes():
    baseline = _frame()
    candidate = baseline.copy(deep=True)
    candidate[subject.QUANTILES] += np.array([110.0, 0.0, 40.0, 0.0])[:, None]
    retained_baseline, retained_candidate = baseline.copy(deep=True), candidate.copy(deep=True)
    result = subject.evaluate(baseline, candidate, "Europe/Berlin")
    assert result["changed_hours"] == 2
    assert result["slices"]["all"]["baseline"]["mae"] == 20
    assert result["slices"]["actual_ge_200"]["baseline"]["hours"] == 3
    assert result["spikes"]["200"]["baseline"] == {"tp": 2, "fp": 0, "fn": 1, "precision": 1.0, "recall": 2/3}
    assert result["spikes"]["200"]["interaction"] == {"tp": 3, "fp": 1, "fn": 0, "precision": .75, "recall": 1.0}
    pd.testing.assert_frame_equal(baseline, retained_baseline)
    pd.testing.assert_frame_equal(candidate, retained_candidate)


@pytest.mark.parametrize("kind", ["nonfinite", "crossing"])
def test_evaluation_rejects_invalid_quantiles(kind):
    baseline, candidate = _frame(), _frame()
    candidate.loc[candidate.index[0], subject.QUANTILES[0]] = np.nan if kind == "nonfinite" else 500
    with pytest.raises(ValueError, match="Nonfinite values or quantile crossings"):
        subject.evaluate(baseline, candidate, "Europe/Berlin")


def test_validate_never_creates_output_or_calls_replay(monkeypatch, tmp_path):
    output = tmp_path / "must-remain-absent"
    monkeypatch.setattr(subject, "OUTPUT", output)
    monkeypatch.setattr(subject, "prepare", lambda zone: _fake_prepared())
    def forbidden(*args, **kwargs):
        pytest.fail("Validation must not fit or acquire a write lock")
    monkeypatch.setattr(subject, "build_operational_kalman_view", forbidden)
    monkeypatch.setattr(subject, "baseline_controls", forbidden)
    monkeypatch.setattr(subject, "exclusive_process_lock", forbidden)
    subject.run(action="validate", zones=["DE", "NL"])
    assert not output.exists()


@pytest.mark.parametrize("zones,workers,threads", [(["DE", "DE"], 2, 2), (["FR"], 2, 2), (["DE"], 3, 2), (["DE"], 2, 3)])
def test_invalid_scope_rejected_before_preparation(monkeypatch, zones, workers, threads):
    monkeypatch.setattr(subject, "prepare", lambda zone: pytest.fail("Invalid scope must fail before loading sources"))
    with pytest.raises(ValueError):
        subject.run(action="validate", zones=zones, workers=workers, threads=threads)


def test_completed_receipt_cannot_omit_all_outputs(monkeypatch, tmp_path):
    directory = tmp_path / "de" / _identity()["run_id"]
    directory.mkdir(parents=True)
    (directory / "experiment.json").write_text(json.dumps(_identity()), encoding="utf-8")
    (directory / "status.json").write_text(json.dumps({"status": "COMPLETE", "annual_complete": True,
        "identity": _identity()["run_id"]}), encoding="utf-8")
    (directory / "completion.json").write_text(json.dumps({"identity": _identity()["run_id"], "files": {}}), encoding="utf-8")
    monkeypatch.setattr(subject, "OUTPUT", tmp_path)
    monkeypatch.setattr(subject, "prepare", lambda zone: _fake_prepared())
    monkeypatch.setattr(subject, "exclusive_process_lock", lambda path: nullcontext())
    monkeypatch.setattr(subject, "threadpool_limits", lambda **kwargs: nullcontext())
    with pytest.raises(ValueError):
        subject.run(action="run", zones=["DE"])


def test_complete_run_reuses_only_verified_inventory_without_refit(monkeypatch, tmp_path):
    directory = tmp_path / "de" / _identity()["run_id"]
    directory.mkdir(parents=True)
    for name in subject.RESULT_FILES:
        (directory / name).write_text(json.dumps(_identity()) if name == "experiment.json" else "sealed", encoding="utf-8")
    (directory / "status.json").write_text(json.dumps({"status": "COMPLETE", "annual_complete": True,
        "identity": _identity()["run_id"]}), encoding="utf-8")
    inventory = {name: subject.sha(directory / name) for name in subject.RESULT_FILES}
    (directory / "completion.json").write_text(json.dumps({"identity": _identity()["run_id"], "files": inventory}), encoding="utf-8")
    monkeypatch.setattr(subject, "OUTPUT", tmp_path)
    monkeypatch.setattr(subject, "prepare", lambda zone: _fake_prepared())
    monkeypatch.setattr(subject, "exclusive_process_lock", lambda path: nullcontext())
    monkeypatch.setattr(subject, "threadpool_limits", lambda **kwargs: nullcontext())
    monkeypatch.setattr(subject, "baseline_controls", lambda *args, **kwargs: pytest.fail("A verified complete run must not refit"))
    monkeypatch.setattr(subject, "build_operational_kalman_view", lambda *args, **kwargs: pytest.fail("A verified complete run must not refit"))
    subject.run(action="run", zones=["DE"])
    assert {name: subject.sha(directory / name) for name in subject.RESULT_FILES} == inventory
    pointer = json.loads((tmp_path / "latest_DE_NL.json").read_text(encoding="utf-8"))
    assert pointer["status"] == "PARTIAL"
    assert set(pointer["results"]) == {"DE"}
    (directory / "report.html").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="Completed output changed"):
        subject.run(action="run", zones=["DE"])


def test_baseline_controls_covers_first_middle_last_and_future(monkeypatch, tmp_path):
    history = _frame()
    # Three short synthetic delivery days suffice because the replay is stubbed.
    indexes = pd.DatetimeIndex(["2025-09-22T01:00Z", "2026-03-23T01:00Z", "2026-09-21T01:00Z"])
    history = history.iloc[:3].copy()
    history.index = indexes.rename("delivery_start_utc")
    future = _frame().iloc[:1].copy()
    future.index = pd.DatetimeIndex(["2026-09-22T01:00Z"], name="delivery_start_utc")
    bundle = SimpleNamespace(residual_statistics=history, covariates=pd.DataFrame(), source_forecast=future,
        kalman_view=SimpleNamespace(backtest=history, forecast=future))
    calls = []
    def replay_stub(prefix, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(predictions=subject.indexed(prefix), future_predictions=future)
    monkeypatch.setattr(subject, "replay_kalman_overlay", replay_stub)
    result = subject.baseline_controls(bundle, object(), object(), "DE", tmp_path, 2)
    assert [item["day"] for item in result] == ["2025-09-22", "2026-03-23", "2026-09-21"]
    assert all(call["training_lookback_days"] == 365 for call in calls)
    assert calls[0]["future_upstream"] is None and calls[1]["future_upstream"] is None
    assert calls[-1]["future_upstream"] is not None
    assert result[-1]["forecast_checked"]
