"""Fast parallel coordinator/migration contracts; scientific models are stubbed."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import chronos2_hourly.solar_wind_corrector_parallel as subject
from chronos2_hourly import solar_wind_corrector_interaction as original


def _bounded(tasks, worker, callback, **options):
    maximum = options.pop("max_workers", 4)
    with ThreadPoolExecutor(max_workers=maximum) as executor:
        return subject.bounded_tasks(tasks, worker, executor, callback,
            options.pop("should_stop", lambda: False),
            options.pop("available_gb", lambda: 20.),
            min_free_memory_gb=3., max_workers=maximum, **options)


def test_parallel_math_matches_serial_bitwise_and_writes_only_on_parent():
    index = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")
    base = pd.DataFrame({"q10": 80., "q50": 100., "q90": 130.}, index=index)
    tasks = list(range(9))
    parent_id = threading.get_ident()
    worker_threads, callbacks = [], []
    def compute(day):
        worker_threads.append(threading.get_ident())
        raw = np.linspace(-100., 100., 24) + day
        time.sleep(.002 * (day % 3))
        return {upper: original.apply_variant(base, raw, upper=upper).to_numpy() for upper in (40, 80)}
    expected = {day: compute(day) for day in tasks}
    worker_threads.clear()
    results = {}
    def record(day, result):
        callbacks.append(threading.get_ident())
        assert day not in results
        results[day] = result
    outcome = _bounded(tasks, compute, record)
    assert outcome["stopped"] is False and outcome["completed"] == len(tasks)
    assert set(results) == set(tasks)
    assert all(tid != parent_id for tid in worker_threads)
    assert callbacks == [parent_id] * len(tasks)
    for day in sorted(results):
        for upper in (40, 80):
            np.testing.assert_array_equal(results[day][upper], expected[day][upper])


@pytest.mark.parametrize("max_workers", [2, 3, 4])
def test_worker_concurrency_never_exceeds_requested_bound(max_workers):
    lock = threading.Lock()
    active, maximum_seen = 0, 0
    def compute(task):
        nonlocal active, maximum_seen
        with lock:
            active += 1
            maximum_seen = max(maximum_seen, active)
        time.sleep(.004)
        with lock:
            active -= 1
        return task
    observed = []
    outcome = _bounded(list(range(20)), compute, lambda task, result: observed.append(result),
                       max_workers=max_workers)
    assert 1 <= maximum_seen <= max_workers
    assert sorted(observed) == list(range(20))
    assert outcome["completed"] == 20


def test_stop_drains_started_work_without_submitting_new_days():
    lock = threading.Lock()
    stop = threading.Event()
    started, committed = [], []
    def compute(task):
        with lock:
            started.append(task)
        time.sleep(.004)
        return task + 100
    def record(task, value):
        committed.append(task)
        assert value == task + 100
        stop.set()
    outcome = _bounded(list(range(50)), compute, record, should_stop=stop.is_set)
    assert outcome["stopped"] is True
    assert 1 <= len(started) <= 4
    assert sorted(committed) == sorted(started)
    assert outcome["completed"] == len(committed)


def test_stop_before_first_submission_preserves_all_tasks():
    outcome = _bounded([1, 2, 3], lambda task: pytest.fail("No work after stop"),
                       lambda *args: pytest.fail("Nothing should commit"), should_stop=lambda: True)
    assert outcome["stopped"] is True and outcome["completed"] == 0


def test_failure_preserves_successful_inflight_checkpoint_callbacks():
    gate = threading.Barrier(2)
    started, committed = [], []
    def compute(task):
        started.append(task)
        gate.wait(timeout=2.)
        if task == 0:
            raise ValueError("synthetic deterministic failure")
        time.sleep(.005)
        return task
    with pytest.raises(ValueError, match="synthetic deterministic failure"):
        _bounded(list(range(12)), compute, lambda task, result: committed.append(result), max_workers=2)
    assert sorted(started) == [0, 1]
    assert committed == [1]


@pytest.mark.parametrize("threads,workers", [(1, 2), (3, 2), (2, 1), (2, 5), (2, 0)])
def test_unapproved_thread_or_worker_count_rejected_before_source_read(monkeypatch, threads, workers):
    monkeypatch.setattr(original, "prepare", lambda *args, **kwargs: pytest.fail("Invalid scope must fail before loading sources"))
    with pytest.raises(ValueError):
        subject.run(action="validate", threads=threads, workers=workers)


def _migration_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(original, "OUTPUT", tmp_path / "old")
    monkeypatch.setattr(subject, "OUTPUT", tmp_path / "new")
    index = pd.date_range("2026-01-01", periods=24, freq="h", tz="Europe/Berlin").tz_convert("UTC")
    index = index.rename("delivery_start_utc")
    raw = pd.DataFrame({"q10": 80., "q50": 100., "q90": 130., "actual": 105.}, index=index)
    corrected = raw.rename(columns={q: "residual_corrected__" + q for q in original.QUANTILES})
    identity = {"run_id": "old-experiment", "zone": "DE", "scientific_files": {"source.py": "sealed"}}
    prepared = SimpleNamespace(identity=identity, raw_history=raw, raw_future=raw.drop(columns="actual"),
        bundle=SimpleNamespace(residual_statistics=corrected, source_forecast=corrected.drop(columns="actual")))
    new_identity = subject.make_identity(prepared, workers=4, min_free_memory_gb=3.)
    source_dir = original.OUTPUT / "de" / identity["run_id"]
    source_path = source_dir / "corrector_daily/2026-01-01.json"
    source_path.parent.mkdir(parents=True)
    (source_dir / "experiment.json").write_bytes(subject.json_bytes(identity))
    payload = {"identity": identity["run_id"], "day": "2026-01-01", "index_ns": index.asi8.tolist(),
        "raw": {"original": [0.] * 24, "interaction": [55.] * 24},
        "audit": {"original": {"causality_violations": 0}, "interaction": {"causality_violations": 0}}}
    _seal(source_path, payload)
    return prepared, new_identity, source_path, payload, index


def _seal(path, payload):
    path.write_bytes(subject.json_bytes({"payload": payload, "sha256": subject._digest(payload)}))


def test_import_is_readonly_and_preserves_exact_predictions_with_provenance(monkeypatch, tmp_path):
    prepared, identity, source, payload, index = _migration_fixture(monkeypatch, tmp_path)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in original.OUTPUT.rglob("*") if path.is_file()}
    imported, provenance = subject.import_original_checkpoint(prepared, identity, source, index)
    assert imported["identity"] == identity["run_id"] != payload["identity"]
    assert imported["raw"] == payload["raw"] and imported["audit"] == payload["audit"]
    assert provenance["original_file_sha256"] == subject.sha(source)
    assert provenance["original_payload_sha256"] == subject._digest(payload)
    assert provenance["original_run_id"] == payload["identity"]
    assert provenance["new_run_id"] == identity["run_id"]
    assert provenance["raw_predictions_unchanged"] is True and provenance["thread_count_unchanged"] == 2
    assert not subject.OUTPUT.exists()
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}


@pytest.mark.parametrize("kind", ["checksum", "identity", "day", "grid", "duplicate_hour", "nonfinite", "wrong_baseline", "missing_recipe", "manifest"])
def test_migration_rejects_changed_identity_grid_values_and_manifest(monkeypatch, tmp_path, kind):
    prepared, identity, source, payload, index = _migration_fixture(monkeypatch, tmp_path)
    if kind == "identity":
        payload["identity"] = "different-experiment"
    elif kind == "day":
        payload["day"] = "2026-01-02"
    elif kind == "grid":
        payload["index_ns"][0] += 1
    elif kind == "duplicate_hour":
        payload["index_ns"][1] = payload["index_ns"][0]
    elif kind == "nonfinite":
        payload["raw"]["interaction"][0] = "nan"
    elif kind == "wrong_baseline":
        payload["raw"]["original"][0] = 3.
    elif kind == "missing_recipe":
        del payload["audit"]["interaction"]
    elif kind == "manifest":
        (source.parent.parent / "experiment.json").write_text('{"run_id":"other"}', encoding="utf-8")
    _seal(source, payload)
    if kind == "checksum":
        record = json.loads(source.read_text(encoding="utf-8"))
        record["payload"]["raw"]["interaction"][0] = 56.
        source.write_text(json.dumps(record), encoding="utf-8")
    before = source.read_bytes()
    with pytest.raises(ValueError):
        subject.import_original_checkpoint(prepared, identity, source, index)
    assert source.read_bytes() == before
    assert not subject.OUTPUT.exists()


def test_migration_rejects_identical_file_from_unallowed_directory(monkeypatch, tmp_path):
    prepared, identity, source, _, index = _migration_fixture(monkeypatch, tmp_path)
    outside = tmp_path / "unrelated/2026-01-01.json"
    outside.parent.mkdir()
    outside.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="checkpoint path"):
        subject.import_original_checkpoint(prepared, identity, outside, index)


def test_new_identity_pins_original_and_execution_signature(monkeypatch, tmp_path):
    prepared, identity, _, _, _ = _migration_fixture(monkeypatch, tmp_path)
    assert identity["original_identity"] == prepared.identity
    assert identity["run_id"] != prepared.identity["run_id"]
    assert set(identity["coordinator_files"]) == set(subject.IMPLEMENTATION)
    assert identity["execution"]["threads_per_fit"] == 2
    assert identity["execution"]["checkpoint_writer"] == "parent_only"
    assert subject.make_identity(prepared, 4, 3.) == identity
    assert subject.make_identity(prepared, 2, 3.)["run_id"] != identity["run_id"]
    assert subject.make_identity(prepared, 4, 4.)["run_id"] != identity["run_id"]


def test_validate_is_readonly_without_pool_lock_or_fit(monkeypatch, tmp_path):
    prepared, identity, _, _, _ = _migration_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(original, "prepare", lambda *args, **kwargs: prepared)
    def forbidden(*args, **kwargs):
        pytest.fail("validate must not fit, lock, write or start a pool")
    monkeypatch.setattr(subject, "_pool", forbidden)
    monkeypatch.setattr(subject, "exclusive_process_lock", forbidden)
    monkeypatch.setattr(original, "corrector_controls", forbidden)
    monkeypatch.setattr(subject, "write_json", forbidden)
    assert subject.run(action="validate", zones=("DE",)) == {"DE": identity}
    assert not subject.OUTPUT.exists()


def test_low_memory_stops_new_work_without_killing_or_fitting():
    result = _bounded([1, 2], lambda task: pytest.fail("No task below memory reserve"),
        lambda *args: pytest.fail("No publication expected"), available_gb=lambda: 2.)
    assert result == {"stopped": True, "reason": "memory_reserve", "completed": 0}


def test_shared_old_batch_lock_refuses_parallel_writer(monkeypatch, tmp_path):
    prepared, _, _, _, _ = _migration_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(original, "prepare", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(original, "corrector_controls", lambda *args, **kwargs: pytest.fail("An old batch still owns the lock"))
    lock_path = original.OUTPUT / "batch.lock"
    with original.exclusive_process_lock(lock_path):
        lock_bytes = lock_path.read_bytes()
        with pytest.raises(ValueError, match="Verrou"):
            subject.run(action="run", zones=("DE",))
        assert lock_path.read_bytes() == lock_bytes


def _completion_fixture(monkeypatch, tmp_path):
    prepared, identity, _, _, _ = _migration_fixture(monkeypatch, tmp_path)
    directory = subject.OUTPUT / "de" / identity["run_id"]
    directory.mkdir(parents=True)
    for relative in subject.RESULT_FILES:
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subject.json_bytes(identity) if relative == "experiment.json" else b"sealed")
    receipt = {"identity": identity["run_id"], "annual_complete": True,
        "files": {relative: subject.sha(directory / relative) for relative in subject.RESULT_FILES}}
    state = {"identity": identity["run_id"], "zone": "DE", "status": "COMPLETE", "annual_complete": True,
        "evaluation_days": 365, "evaluation_hours": 8760, "future_forecast_hours": 24}
    (directory / "completion.json").write_bytes(subject.json_bytes(receipt))
    (directory / "status.json").write_bytes(subject.json_bytes(state))
    return prepared, identity, directory, receipt, state


def test_exact_23_artifact_inventory_can_resume_without_fits(monkeypatch, tmp_path):
    prepared, identity, directory, receipt, _ = _completion_fixture(monkeypatch, tmp_path)
    assert len(subject.RESULT_FILES) == len(set(subject.RESULT_FILES)) == 23
    assert "migration_audit.json" in subject.RESULT_FILES
    monkeypatch.setattr(original, "prepare", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(original, "corrector_controls", lambda *args, **kwargs: pytest.fail("No refit of complete run"))
    assert subject._verify_completion(directory, identity)
    assert subject.run(action="run", zones=("DE",)) == {"DE": str(directory / "report.html")}
    assert receipt["files"] == {relative: subject.sha(directory / relative) for relative in subject.RESULT_FILES}


@pytest.mark.parametrize("kind", ["omit_migration", "empty_inventory", "tamper_variant", "one_day", "no_forecast", "wrong_zone", "wrong_identity"])
def test_completion_rejects_partial_or_changed_inventory(monkeypatch, tmp_path, kind):
    _, identity, directory, receipt, state = _completion_fixture(monkeypatch, tmp_path)
    if kind == "omit_migration":
        del receipt["files"]["migration_audit.json"]
    elif kind == "empty_inventory":
        receipt["files"] = {}
    elif kind == "tamper_variant":
        (directory / "interaction_80/backtest.parquet").write_bytes(b"altered")
    elif kind == "one_day":
        state["evaluation_days"], state["evaluation_hours"] = 1, 24
    elif kind == "no_forecast":
        state["future_forecast_hours"] = 0
    elif kind == "wrong_zone":
        state["zone"] = "NL"
    else:
        receipt["identity"] = "other"
    (directory / "completion.json").write_bytes(subject.json_bytes(receipt))
    (directory / "status.json").write_bytes(subject.json_bytes(state))
    with pytest.raises(ValueError):
        subject._verify_completion(directory, identity)


def _annual_prepared():
    timezone = "Europe/Berlin"
    start, stop = pd.Timestamp("2025-09-22", tz=timezone), pd.Timestamp("2026-09-22", tz=timezone)
    history_index = pd.date_range(start, stop, freq="h", inclusive="left").tz_convert("UTC").rename("delivery_start_utc")
    future_index = pd.date_range(stop, periods=24, freq="h").tz_convert("UTC").rename("delivery_start_utc")
    def raw(index, actual):
        result = pd.DataFrame({"q10": 80., "q50": 100., "q90": 130.}, index=index)
        if actual:
            result["actual"] = 103.
        return result
    history, future = raw(history_index, True), raw(future_index, False)
    def upstream(frame):
        result = frame.copy(deep=True)
        for q in original.QUANTILES:
            result["chronos2__" + q] = frame[q]
            result["residual_corrected__" + q] = frame[q]
        result["residual_correction"] = 0.
        result["forecast_origin_utc"] = result.index - pd.Timedelta(days=1)
        return result
    statistics, forecast = upstream(history), upstream(future)
    def overlay(frame):
        result = pd.DataFrame(index=frame.index)
        for q in original.QUANTILES:
            result["residual_kalman__" + q] = frame["residual_corrected__" + q] + 2.
        result["kalman_correction"] = 2.
        result["kalman_raw_correction"] = 2.
        result["kalman_weight"] = 1.
        result["kalman_selected_filter"] = "linear_market"
        return result
    baseline, baseline_future = original.merge_overlay(statistics, overlay(statistics)), original.merge_overlay(forecast, overlay(forecast))
    full_index = history_index.append(future_index)
    feature = pd.DataFrame({"de_low_wind_solar_stress": .5}, index=full_index)
    covariates = pd.DataFrame({"de_residual_load_fcst": 10.}, index=full_index)
    config, covconfig = {"candidate_kinds": ("linear_bias", "linear_market")}, {"input_columns": ("de_residual_load_fcst",)}
    identity = {"run_id": "serial-annual", "zone": "DE", "feature": feature.columns[0],
        "kalman_config": json.loads(json.dumps(config)), "kalman_covariate_config": json.loads(json.dumps(covconfig))}
    bundle = SimpleNamespace(residual_statistics=statistics, source_forecast=forecast, covariates=covariates,
        kalman_view=SimpleNamespace(backtest=baseline, forecast=baseline_future))
    return SimpleNamespace(identity=identity, config=config, covconfig=covconfig, bundle=bundle,
        raw_history=history, raw_future=future, interaction=feature, feature_audit={}), overlay


def test_mock_annual_run_assembles_reverse_completed_days_and_seals_23_artifacts(monkeypatch, tmp_path):
    """Whole coordinator, 365 days and three variants; no model or subprocess."""
    prepared, overlay = _annual_prepared()
    monkeypatch.setattr(original, "OUTPUT", tmp_path / "serial")
    monkeypatch.setattr(subject, "OUTPUT", tmp_path / "parallel")
    # Path redirection is covered by the migration guards. This single large
    # mock avoids quadratic rescans of its own growing temporary directory.
    def confined_test_path(path):
        path = Path(path).absolute()
        assert path.is_relative_to(subject.OUTPUT.absolute())
        return path
    monkeypatch.setattr(subject, "safe_path", confined_test_path)
    monkeypatch.setattr(original, "prepare", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(original, "corrector_controls", lambda *args, **kwargs: ([], {}))
    monkeypatch.setattr(original, "baseline_controls", lambda *args, **kwargs: [])
    submitted = []
    def fake_fit(item, day, *, interaction):
        train = original.train_window(item.raw_history.index, day, "Europe/Berlin")
        assert all(train.tz_convert("Europe/Berlin").date < pd.Timestamp(day).date())
        base = subject._base_for_day(item, day)
        return np.full(len(base), 55. if interaction else 0.), {
            "causality_violations": 0, "training_rows": len(train), "loss": "MAE"}
    monkeypatch.setattr(original, "_fit_raw", fake_fit)
    def fake_replay(prefix, **kwargs):
        history = original.indexed(prefix)
        day = pd.Timestamp(kwargs["evaluation_start_day"]).date()
        selected = history.loc[history.index.tz_convert("Europe/Berlin").date == day]
        future = None if kwargs["future_upstream"] is None else original.indexed(kwargs["future_upstream"])
        pd.testing.assert_frame_equal(kwargs["covariates"], prepared.bundle.covariates)
        assert kwargs["training_lookback_days"] == 365 and kwargs["rolling_refit_workers"] == 1
        assert not any("stress" in name for name in kwargs["covariates"])
        audit = {"causality_violations": 0, "quantile_crossings": 0, "future_observations_assimilated": 0,
            "evaluation_days": 1, "evaluation_hours": len(selected), "future_forecast_hours": 0 if future is None else 24,
            "training_lookback_days": 365, "config": prepared.config, "covariate_config": prepared.covconfig}
        return SimpleNamespace(predictions=overlay(selected), future_predictions=None if future is None else overlay(future), audit=audit)
    monkeypatch.setattr(original, "replay_kalman_overlay", fake_replay)
    def reverse_pool(item, identity, tasks, worker, on_result, progress, *, upstreams=None, control_raw=None):
        monkeypatch.setattr(subject, "_PREPARED", item)
        monkeypatch.setattr(subject, "_UPSTREAMS", upstreams)
        monkeypatch.setattr(subject, "_CONTROL_RAW", control_raw or {})
        for task in reversed(tasks):
            submitted.append((worker.__name__, task["day"]))
            on_result(task, worker(task))
    monkeypatch.setattr(subject, "_run_pool_tasks", reverse_pool)
    # Checkpoint serialization itself is covered separately; keep this annual
    # orchestration test light while preserving real atomic file checksums.
    saved_frames = {}
    def lightweight_frame(path, frame):
        path = subject.safe_path(path)
        saved_frames[path] = frame.copy(deep=True)
        fingerprint = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
        path.write_bytes(fingerprint)
    monkeypatch.setattr(subject, "write_frame", lightweight_frame)
    result = subject.run(action="run", zones=("DE",), workers=4)
    identity = subject.make_identity(prepared, 4, 3.)
    directory = subject.OUTPUT / "de" / identity["run_id"]
    assert result == {"DE": str(directory / "report.html")}
    assert len(submitted) == 366 + 365 * 3
    assert submitted[0][1] == "2026-09-22"  # deliberately not chronological finish
    assert subject._verify_completion(directory, identity)
    receipt = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
    assert len(receipt["files"]) == 23
    for variant in subject.VARIANTS:
        annual = saved_frames[directory / variant / "backtest.parquet"]
        future = saved_frames[directory / variant / "forecast.parquet"]
        pd.testing.assert_index_equal(annual.index, prepared.raw_history.index)
        assert len(annual) == 8760 and len(future) == 24
        pd.testing.assert_series_equal(annual.actual, prepared.raw_history.actual)
        assert "actual" not in future
        pd.testing.assert_series_equal(annual["chronos2__q50"], prepared.raw_history.q50, check_names=False)
        expected = 142. if variant == "interaction_40" else (157. if variant == "interaction_80" else 102.)
        np.testing.assert_array_equal(annual["residual_kalman__q50"], np.full(8760, expected))
    migration = json.loads((directory / "migration_audit.json").read_text(encoding="utf-8"))
    assert migration["original_files_modified"] is False and migration["imported_count"] == 0
    assert not (original.OUTPUT / "de" / prepared.identity["run_id"]).exists()
    monkeypatch.setattr(subject, "_run_pool_tasks", lambda *args, **kwargs: pytest.fail("Complete experiment must not restart workers"))
    assert subject.run(action="run", zones=("DE",), workers=4) == result
