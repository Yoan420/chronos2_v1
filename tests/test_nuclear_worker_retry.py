"""A denied Windows Kalman worker pipe can resume without changing model caches."""
from __future__ import annotations

import multiprocessing.connection as mp_connection
from pathlib import Path

import pandas as pd
import pytest
from joblib import _parallel_backends

import run_nuclear_forecast as runner
import chronos2_hourly.kalman_residual as kalman
import chronos2_hourly.nuclear_forecast as engine
from test_kalman_rolling_window import TIMEZONE, _config, _history


def _denied(*, winerror=5, filename=None):
    error = PermissionError(13, "Windows worker pipe denied", filename)
    error.winerror = winerror
    return error


@pytest.fixture
def denied_worker_pipe(monkeypatch):
    """Exercise the real startup frames without opening handles or processes."""
    if not hasattr(mp_connection, "_winapi"):
        pytest.skip("The denied named-pipe startup is specific to Windows")
    errors = []

    def deny_create_file(*args):
        error = _denied()
        errors.append(error)
        raise error

    def initialize_executor(*args, **kwargs):
        # Leave Parallel._initialize_backend, LokyBackend.configure and Pipe
        # untouched: those real code objects establish the error's provenance.
        return mp_connection.Pipe(duplex=False)

    monkeypatch.setattr(_parallel_backends, "get_memmapping_executor", initialize_executor)
    monkeypatch.setattr(mp_connection._winapi, "CreateNamedPipe", lambda *args: 0)
    monkeypatch.setattr(mp_connection._winapi, "CreateFile", deny_create_file)
    return errors


def _replay(*, workers, cache=None):
    return kalman.replay_kalman_overlay(
        _history(start="2026-01-01", days=9),
        timezone=TIMEZONE,
        evaluation_start_day="2026-01-05",
        training_lookback_days=4,
        rolling_refit_workers=workers,
        rolling_refit_cache_dir=cache,
        config=_config(),
    )


def _call(stage, tmp_path, *, workers=4, mode="incremental", **kwargs):
    return runner.run_forecast_with_storage_retry(
        stage, workdir=tmp_path, zone="FR", workers=workers,
        config={"nuclear_experiment": {"mode": mode}}, **kwargs,
    )


def test_named_pipe_denial_resumes_identical_kalman_and_reuses_cache(
    tmp_path, monkeypatch, denied_worker_pipe,
):
    expected = _replay(workers=1)
    calls = []
    data = object()
    config = {"nuclear_experiment": {"mode": "incremental"}}
    arguments = dict(workdir=tmp_path, zone="FR", workers=4, threads=4,
                     device="auto", data=data, config=config)

    def forecast(**kwargs):
        calls.append(dict(kwargs))
        return _replay(workers=kwargs["workers"], cache=tmp_path / "kalman")

    actual = runner.run_forecast_with_storage_retry(forecast, **arguments)
    assert [call["workers"] for call in calls] == [4, 1]
    assert len(denied_worker_pipe) == 1
    assert arguments["workers"] == 4
    assert all(call["config"] is config and call["data"] is data for call in calls)
    assert all({key: value for key, value in call.items() if key != "workers"}
               == {key: value for key, value in arguments.items() if key != "workers"}
               for call in calls)
    for name in ("predictions", "candidate_predictions", "daily_audit", "state_audit"):
        pd.testing.assert_frame_equal(getattr(actual, name), getattr(expected, name))
    assert actual.audit["rolling_refit_workers"] == 1
    assert actual.audit["rolling_refit_cache"]["writes"] == 5

    monkeypatch.setattr(kalman, "_fit_rolling_target_day",
                        lambda **kwargs: pytest.fail("completed fits must be reused"))
    cached = runner.run_forecast_with_storage_retry(forecast, **arguments)
    assert len(denied_worker_pipe) == 1  # No worker startup when every fit is cached.
    assert cached.audit["rolling_refit_cache"]["history_hits"] == 5
    pd.testing.assert_frame_equal(cached.predictions, actual.predictions)


@pytest.mark.parametrize("case", ["full", "already_serial"])
def test_recognized_startup_error_is_not_retried_for_full_or_serial_runs(
    tmp_path, denied_worker_pipe, case,
):
    calls = []

    def forecast(**kwargs):
        calls.append(kwargs["workers"])
        # Force the same genuine startup failure, including when the caller
        # already requested one worker. Retrying cannot improve that request.
        return _replay(workers=4)

    with pytest.raises(PermissionError) as captured:
        _call(forecast, tmp_path, workers=1 if case == "already_serial" else 4,
              mode="full" if case == "full" else "incremental")
    assert len(calls) == len(denied_worker_pipe) == 1
    assert captured.value is denied_worker_pipe[0]


def test_repeated_startup_denial_cannot_loop_after_serial_fallback(
    tmp_path, denied_worker_pipe,
):
    calls = []

    def forecast(**kwargs):
        calls.append(kwargs["workers"])
        return _replay(workers=4)

    with pytest.raises(PermissionError):
        _call(forecast, tmp_path)
    assert calls == [4, 1]
    assert len(denied_worker_pipe) == 2


@pytest.mark.parametrize("winerror", [5, 32, 33, 87, None])
def test_unrelated_permission_error_is_never_retried(tmp_path, winerror):
    error = _denied(winerror=winerror)
    calls = []

    def forecast(**kwargs):
        calls.append(kwargs["workers"])
        raise error

    with pytest.raises(PermissionError) as captured:
        _call(forecast, tmp_path)
    assert calls == [4]
    assert captured.value is error


@pytest.mark.parametrize("winerror", [32, 33, 87, None])
def test_non_access_denial_from_worker_pipe_is_not_retried(
    tmp_path, monkeypatch, denied_worker_pipe, winerror,
):
    error = _denied(winerror=winerror)
    calls = []

    def deny(*args):
        raise error

    def forecast(**kwargs):
        calls.append(kwargs["workers"])
        return _replay(workers=kwargs["workers"])

    monkeypatch.setattr(mp_connection._winapi, "CreateFile", deny)
    with pytest.raises(PermissionError) as captured:
        _call(forecast, tmp_path)
    assert calls == [4]
    assert captured.value is error


def test_backend_error_without_pipe_provenance_is_not_retried(
    tmp_path, monkeypatch, denied_worker_pipe,
):
    error = _denied()
    calls = []

    def deny(*args, **kwargs):
        raise error

    def forecast(**kwargs):
        calls.append(kwargs["workers"])
        return _replay(workers=kwargs["workers"])

    monkeypatch.setattr(_parallel_backends, "get_memmapping_executor", deny)
    with pytest.raises(PermissionError) as captured:
        _call(forecast, tmp_path)
    assert calls == [4]
    assert captured.value is error


def test_two_cache_recoveries_do_not_consume_worker_fallback_budget(
    tmp_path, monkeypatch, denied_worker_pipe,
):
    parent = tmp_path / "cache/nuclear_result" / engine._digest_json({})
    original_rename = engine.os.rename
    publications, calls = [], []

    def locked_first_publication(source, target):
        if "-recovery-" not in Path(source).name:
            publications.append(Path(target).name)
            error = _denied(filename=str(source))
            error.filename2 = str(target)
            raise error
        return original_rename(source, target)

    def forecast(**kwargs):
        calls.append(kwargs["workers"])
        for kind in ("raw_future", "residual"):
            if engine._load_result_cache(parent / kind, {}) is None:
                names = ("raw_future",) if kind == "raw_future" else ("statistics", "forecast", "daily_audit")
                frames = {name: pd.DataFrame({"x": [1.0]}) for name in names}
                engine._save_result_cache(parent / kind, {}, frames)
        return _replay(workers=kwargs["workers"])

    monkeypatch.setattr(engine.os, "rename", locked_first_publication)
    result = _call(forecast, tmp_path)
    assert calls == [4, 4, 4, 1]
    assert publications == ["raw_future", "residual"]
    assert len(denied_worker_pipe) == 1
    assert result.audit["rolling_refit_workers"] == 1
    assert engine._load_result_cache(parent / "raw_future", {}) is not None
    assert engine._load_result_cache(parent / "residual", {}) is not None
