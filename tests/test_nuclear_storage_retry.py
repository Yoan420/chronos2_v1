"""Retry the failed Windows publication without invalidating daily model caches."""
from pathlib import Path

import pandas as pd
import pytest

import run_nuclear_forecast as runner
import chronos2_hourly.nuclear_forecast as engine
from chronos2_hourly.nuclear_residual_cache import ResidualDayCache
from chronos2_hourly import atomic_directory as atomic
from test_nuclear_forecast import FakeResidual, _small_inputs


def _locked(source, target, winerror=5):
    error = PermissionError(13, "Windows publication locked", str(source))
    error.filename2 = str(target)
    error.winerror = winerror
    return error


def _call(stage, workdir, mode="incremental"):
    return runner.run_forecast_with_storage_retry(stage, workdir=workdir, zone="FR",
        config={"nuclear_experiment": {"mode": mode}})


def _frames(kind="residual"):
    names = ["raw_future"] if kind == "raw_future" else ["statistics", "forecast", "daily_audit"]
    return {name: pd.DataFrame({"x": [1.]}) for name in names}


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_failed_residual_publication_resumes_exact_daily_cache_without_refit(tmp_path, monkeypatch, capsys, winerror):
    raw, future, features, day = _small_inputs()
    destination = tmp_path / "cache/nuclear_result" / engine._digest_json({"test": 1}) / "residual"
    original_rename = engine.os.rename
    attempts, fits, snapshots, caches, sleeps = [], [], [], [], []

    class CountingResidual(FakeResidual):
        def fit(self, *args):
            fits.append(True)
            return super().fit(*args)

    def rename(source, target, *args, **kwargs):
        if Path(target) == destination:
            attempts.append(Path(source))
            # Every newly created directory is briefly locked, not just the
            # first overall attempt. Recovery must retry the SAME directory.
            if attempts.count(Path(source)) == 1:
                raise _locked(source, target, winerror)
        return original_rename(source, target, *args, **kwargs)

    def stage(**kwargs):
        retained = engine._load_result_cache(destination, {"test": 1})
        if retained is not None:
            return retained
        cache = ResidualDayCache(tmp_path / "daily", {"fixture": 1},
                                 features=features, raw=raw, timezone="Europe/Paris")
        caches.append(cache)
        statistics, forecast, audit = engine.causal_residual_replay(
            raw_history=raw, raw_future=future, features=features, timezone="Europe/Paris",
            delivery_day=day, residual_factory=CountingResidual, daily_cache=cache)
        frames = {"statistics": statistics, "forecast": forecast, "daily_audit": audit}
        snapshots.append(frames)
        engine._save_result_cache(destination, {"test": 1}, frames)
        return frames

    monkeypatch.setattr(engine.os, "rename", rename)
    monkeypatch.setattr(atomic, "_rename_no_replace", rename)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    output = _call(stage, tmp_path)
    assert len(attempts) == 3 and sleeps == [0.25]
    assert attempts[1] == attempts[2] and attempts[0] != attempts[1]
    assert len(fits) == 4
    assert caches[0].misses == 4 and caches[0].writes == 4
    assert len(caches) == len(snapshots) == 1  # no replay or fit on resume
    retained = engine._load_result_cache(destination, {"test": 1})
    for name in ("statistics", "forecast"):
        pd.testing.assert_frame_equal(snapshots[0][name], output[name])
        pd.testing.assert_frame_equal(retained[name], output[name])
    assert "Reprise depuis les caches quotidiens verifies" in capsys.readouterr().out


@pytest.mark.parametrize("kind", ["raw_future", "residual"])
def test_persistent_publication_failure_preserves_payload_after_bounded_attempts(tmp_path, monkeypatch, kind):
    destination = tmp_path / "cache/nuclear_result" / engine._digest_json({}) / kind
    errors, sleeps = [], []

    def deny(source, target):
        error = _locked(source, target)
        errors.append(error)
        raise error

    monkeypatch.setattr(engine.os, "rename", deny)
    monkeypatch.setattr(atomic, "_rename_no_replace", deny)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    with pytest.raises(atomic.AtomicDirectoryPublishError) as captured:
        _call(lambda **_: engine._save_result_cache(destination, {}, _frames(kind)), tmp_path)
    assert len(errors) == 8 and captured.value.__cause__ is errors[-1]
    assert sleeps == [0.25, 0.5, 0.75, 1., 1.25, 1.5]
    assert not destination.exists()
    stages = list(destination.parent.glob("." + kind + "-recovery-*"))
    assert len(stages) == 1
    assert set(engine._load_result_cache(stages[0], {})) == set(_frames(kind))


@pytest.mark.parametrize("case", ["full", "other_winerror", "missing_winerror", "foreign_directory", "wrong_error_path", "wrong_identity", "wrong_members"])
def test_unrelated_or_full_replay_errors_are_not_retried(tmp_path, monkeypatch, case):
    parent = "elsewhere" if case == "foreign_directory" else "cache/nuclear_result"
    digest = "a" * 64 if case == "wrong_identity" else engine._digest_json({})
    destination = tmp_path / parent / digest / "residual"
    attempts = []

    def deny(source, target):
        attempts.append(True)
        error = _locked(source, target, 5 if case != "other_winerror" else 87)
        if case == "missing_winerror":
            del error.winerror
        if case == "wrong_error_path":
            error.filename2 = str(tmp_path / "unrelated")
        raise error

    monkeypatch.setattr(engine.os, "rename", deny)
    monkeypatch.setattr(runner.time, "sleep", lambda _: pytest.fail("must not retry this failure"))
    with pytest.raises(PermissionError):
        _call(lambda **_: engine._save_result_cache(destination, {},
              {"wrong": pd.DataFrame({"x": [1.]})} if case == "wrong_members" else _frames()),
              tmp_path, mode="full" if case == "full" else "incremental")
    assert len(attempts) == 1


def test_matching_filenames_without_cache_writer_traceback_are_not_retried(tmp_path, monkeypatch):
    destination = tmp_path / "cache/nuclear_result" / engine._digest_json({}) / "residual"
    original = _locked(destination.with_name(".residual-stage"), destination)

    def training(**kwargs):
        raise original

    monkeypatch.setattr(runner.time, "sleep", lambda _: pytest.fail("training failure must not be retried"))
    with pytest.raises(PermissionError) as captured:
        _call(training, tmp_path)
    assert captured.value is original


def test_destination_collision_is_never_overwritten_or_retried(tmp_path, monkeypatch):
    destination = tmp_path / "cache/nuclear_result" / engine._digest_json({}) / "residual"

    def collision(source, target):
        destination.mkdir()
        (destination / "owner.txt").write_text("another publication")
        raise _locked(source, target)

    monkeypatch.setattr(engine.os, "rename", collision)
    monkeypatch.setattr(runner.time, "sleep", lambda _: pytest.fail("collision must not be retried"))
    with pytest.raises(PermissionError):
        _call(lambda **_: engine._save_result_cache(destination, {}, _frames()), tmp_path)
    assert (destination / "owner.txt").read_text() == "another publication"


def test_recovery_rejects_changed_serialization_before_publishing(tmp_path, monkeypatch):
    destination = tmp_path / "cache/nuclear_result" / engine._digest_json({}) / "residual"
    frames = _frames()

    def changed_after_write(source, target):
        frames["forecast"].loc[0, "x"] = 99.
        raise _locked(source, target)

    monkeypatch.setattr(engine.os, "rename", changed_after_write)
    monkeypatch.setattr(atomic, "_rename_no_replace", lambda *_: pytest.fail("changed bytes must never be published"))
    with pytest.raises(ValueError, match="divergent"):
        _call(lambda **_: engine._save_result_cache(destination, {}, frames), tmp_path)
    assert not destination.exists()


def test_two_successive_cache_publications_resume_with_three_engine_calls(tmp_path, monkeypatch):
    parent = tmp_path / "cache/nuclear_result" / engine._digest_json({})
    original = engine.os.rename
    calls, generated, failures = [], [], []

    def first_writer_locked(source, target):
        if "-recovery-" not in Path(source).name:
            failures.append(Path(target).name)
            raise _locked(source, target)
        return original(source, target)

    def stage(**kwargs):
        calls.append(True)
        for kind in ["raw_future", "residual"]:
            if engine._load_result_cache(parent / kind, {}) is None:
                generated.append(kind)
                engine._save_result_cache(parent / kind, {}, _frames(kind))
        return "complete"

    monkeypatch.setattr(engine.os, "rename", first_writer_locked)
    monkeypatch.setattr(atomic, "_rename_no_replace", first_writer_locked)
    assert _call(stage, tmp_path) == "complete"
    assert len(calls) == 3
    assert generated == failures == ["raw_future", "residual"]
