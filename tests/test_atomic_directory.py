import errno
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from chronos2_hourly import atomic_directory as atomic


def windows_error(code):
    error = PermissionError(errno.EACCES, "Access is denied")
    error.winerror = code
    return error


def make_stage(parent):
    stage = parent / "staged"
    stage.mkdir()
    pd.DataFrame({"timestamp": pd.date_range("2026-09-10", periods=24, freq="h", tz="UTC"),
                  "actual": 50.0}).to_parquet(stage / "values.parquet")
    return stage


@pytest.mark.parametrize("code", [5, 32, 33])
def test_transient_windows_locks_retry_same_complete_directory(tmp_path, monkeypatch, code):
    stage, final = make_stage(tmp_path), tmp_path / "final"
    digest = hashlib.sha256((stage / "values.parquet").read_bytes()).hexdigest()
    rename = atomic._rename_no_replace
    calls, sleeps = [], []

    def fail_twice(source, target):
        calls.append((source, target))
        assert hashlib.sha256((source / "values.parquet").read_bytes()).hexdigest() == digest
        if len(calls) < 3:
            raise windows_error(code)
        rename(source, target)

    monkeypatch.setattr(atomic, "_rename_no_replace", fail_twice)
    monkeypatch.setattr(atomic.time, "sleep", sleeps.append)
    assert atomic.publish_directory_no_replace(stage, final) == final
    assert calls == [(stage, final)] * 3 and sleeps == [0.25, 0.5]
    assert len(pd.read_parquet(final / "values.parquet")) == 24
    assert not stage.exists()


def test_permanent_windows_lock_preserves_stage_and_original_error(tmp_path, monkeypatch):
    failure = windows_error(5)
    attempts, sleeps = [], []

    def locked(source, target):
        attempts.append((source, target))
        raise failure

    monkeypatch.setattr(atomic, "_rename_no_replace", locked)
    monkeypatch.setattr(atomic.time, "sleep", sleeps.append)
    with pytest.raises(atomic.AtomicDirectoryPublishError) as captured:
        with atomic.AtomicDirectoryStaging(tmp_path, prefix=".stage-") as publication:
            pd.DataFrame({"actual": [1.0]}).to_parquet(publication.path / "result.parquet")
            stage = publication.path
            publication.publish(tmp_path / "final")
    assert len(attempts) == 7 and sum(sleeps) <= 10
    assert captured.value.__cause__ is failure
    assert str(stage) in str(captured.value)
    assert pd.read_parquet(stage / "result.parquet").actual.tolist() == [1.0]
    assert not (tmp_path / "final").exists()


@pytest.mark.parametrize("kind", ["empty_directory", "nonempty_directory", "file"])
def test_existing_destination_is_never_replaced(tmp_path, monkeypatch, kind):
    stage, final = make_stage(tmp_path), tmp_path / "final"
    if kind == "file":
        final.write_text("existing")
    else:
        final.mkdir()
        if kind == "nonempty_directory":
            (final / "keep.txt").write_text("existing")
    monkeypatch.setattr(atomic, "_rename_no_replace", lambda *_: pytest.fail("collision must not be renamed"))
    with pytest.raises(FileExistsError):
        atomic.publish_directory_no_replace(stage, final)
    assert (stage / "values.parquet").exists()
    if kind == "file":
        assert final.read_text() == "existing"
    elif kind == "nonempty_directory":
        assert (final / "keep.txt").read_text() == "existing"
    else:
        assert list(final.iterdir()) == []


def test_destination_appearing_during_attempt_stops_retry_and_keeps_both(tmp_path, monkeypatch):
    stage, final = make_stage(tmp_path), tmp_path / "final"

    def competitor(source, target):
        target.mkdir()
        (target / "other.txt").write_text("another publisher")
        raise windows_error(5)

    monkeypatch.setattr(atomic, "_rename_no_replace", competitor)
    monkeypatch.setattr(atomic.time, "sleep", lambda _: pytest.fail("no retry after collision"))
    with pytest.raises(FileExistsError):
        atomic.publish_directory_no_replace(stage, final)
    assert (stage / "values.parquet").exists()
    assert (final / "other.txt").read_text() == "another publisher"


@pytest.mark.parametrize("failure", [OSError(errno.EIO, "storage failed"), PermissionError("non-Windows denial")])
def test_unrelated_io_error_is_not_retried(tmp_path, monkeypatch, failure):
    stage, final = make_stage(tmp_path), tmp_path / "final"
    calls = []

    def fail(source, target):
        calls.append((source, target))
        raise failure

    monkeypatch.setattr(atomic, "_rename_no_replace", fail)
    monkeypatch.setattr(atomic.time, "sleep", lambda _: pytest.fail("not a Windows sharing lock"))
    with pytest.raises(OSError) as captured:
        atomic.publish_directory_no_replace(stage, final)
    assert captured.value is failure and len(calls) == 1
    assert stage.is_dir() and not final.exists()


def test_native_rename_rejects_existing_empty_directory(tmp_path):
    stage, final = make_stage(tmp_path), tmp_path / "final"
    final.mkdir()
    with pytest.raises(OSError):
        atomic._rename_no_replace(stage, final)
    assert (stage / "values.parquet").is_file() and list(final.iterdir()) == []
