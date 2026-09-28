from pathlib import Path
import errno
import hashlib
import os

import pytest

from chronos2_hourly import nyx_local_io as io


def test_immutable_artifacts_share_storage_and_remain_independently_readable(tmp_path):
    source, target = tmp_path / "cache/closed.parquet", tmp_path / "bundle/closed.parquet"
    source.parent.mkdir()
    source.write_bytes(b"frozen numerical artifact")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert io.publish_verified_immutable_copy(source, target, digest) == "linked"
    assert os.path.samefile(source, target)
    assert io.publish_verified_immutable_copy(source, target, digest) == "existing"
    source.unlink()
    assert target.read_bytes() == b"frozen numerical artifact"


def test_immutable_cross_volume_copy_has_identical_bytes(tmp_path, monkeypatch):
    source, target = tmp_path / "closed.bin", tmp_path / "bundle/closed.bin"
    source.write_bytes(b"frozen numerical artifact")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    def cross_device(*args):
        raise OSError(errno.EXDEV, "different volumes")
    monkeypatch.setattr(io.os, "link", cross_device)
    assert io.publish_verified_immutable_copy(source, target, digest) == "copied"
    assert not os.path.samefile(source, target)
    assert target.read_bytes() == source.read_bytes()
    assert not list(target.parent.glob("*.tmp"))


def test_immutable_copy_refuses_source_or_destination_tampering(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"frozen")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    target.write_bytes(b"different")
    with pytest.raises(ValueError, match="copied artifact differs"):
        io.publish_verified_immutable_copy(source, target, digest)
    assert target.read_bytes() == b"different"
    target.unlink()
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source differs"):
        io.publish_verified_immutable_copy(source, target, digest)
    assert not target.exists()


def test_hardlink_failure_does_not_hide_disk_error(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"closed")
    def disk_error(*args):
        raise OSError(errno.EIO, "disk failure")
    monkeypatch.setattr(io.os, "link", disk_error)
    with pytest.raises(OSError, match="disk failure"):
        io.publish_verified_immutable_copy(source, tmp_path / "target", hashlib.sha256(b"closed").hexdigest())


def test_transient_reader_lock_retries_without_removing_old_file(tmp_path, monkeypatch):
    target = tmp_path / "status.json"
    target.write_bytes(b"old")
    original = Path.replace
    attempts = []
    def replace(source, destination):
        attempts.append(source)
        if len(attempts) <= 2:
            assert target.read_bytes() == b"old"
            raise PermissionError("Windows sharing violation")
        return original(source, destination)
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(io.time, "sleep", lambda seconds: None)
    io.publish_bytes(target, b"new")
    assert target.read_bytes() == b"new"
    assert len(attempts) == 3
    assert list(tmp_path.iterdir()) == [target]


def test_permanent_denial_preserves_previous_status_and_raises(tmp_path, monkeypatch):
    source, target = tmp_path / "new.tmp", tmp_path / "status.json"
    source.write_bytes(b"new")
    target.write_bytes(b"old")
    def denied(*args):
        raise PermissionError("permanent")
    monkeypatch.setattr(Path, "replace", denied)
    with pytest.raises(PermissionError, match="permanent"):
        io.replace_retry(source, target, timeout=0)
    assert source.read_bytes() == b"new"
    assert target.read_bytes() == b"old"


def test_other_io_failure_is_not_treated_as_transient(tmp_path, monkeypatch):
    def denied(*args):
        raise OSError("disk failure")
    monkeypatch.setattr(Path, "replace", denied)
    with pytest.raises(OSError, match="disk failure"):
        io.publish_bytes(tmp_path / "status.json", b"new")
    assert not list(tmp_path.iterdir())


def test_directory_promotion_retries_transient_locks_and_keeps_sealed_files(tmp_path, monkeypatch):
    source, target = tmp_path / "attempt", tmp_path / "published"
    source.mkdir()
    (source / "model.cbm").write_bytes(b"sealed model")
    original, calls = Path.rename, []
    def rename(path, destination):
        calls.append(path)
        if len(calls) < 3:
            assert (source / "model.cbm").read_bytes() == b"sealed model"
            assert not target.exists()
            raise PermissionError("transient folder sharing violation")
        return original(path, destination)
    monkeypatch.setattr(Path, "rename", rename)
    monkeypatch.setattr(io.time, "sleep", lambda *_: None)
    io.promote_directory_retry(source, target)
    assert len(calls) == 3 and not source.exists()
    assert (target / "model.cbm").read_bytes() == b"sealed model"


def test_directory_promotion_never_overwrites_and_preserves_permanent_failure(tmp_path, monkeypatch):
    source, target = tmp_path / "attempt", tmp_path / "published"
    source.mkdir()
    (source / "model.cbm").write_bytes(b"sealed model")
    target.mkdir()
    (target / "receipt.json").write_bytes(b"completed receipt")
    with pytest.raises(FileExistsError, match="Publication already exists"):
        io.promote_directory_retry(source, target)
    assert (target / "receipt.json").read_bytes() == b"completed receipt"
    def denied(*_):
        raise PermissionError("permanent")
    monkeypatch.setattr(Path, "rename", denied)
    with pytest.raises(PermissionError, match="permanent"):
        io.promote_directory_retry(source, tmp_path/"unused", timeout=0)
    assert (source / "model.cbm").read_bytes() == b"sealed model"
