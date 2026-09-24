"""Regressions for republishing nuclear exports opened by Windows readers."""

import errno
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from chronos2_hourly import nuclear_exports as exports
from chronos2_hourly.hourly_contract import local_delivery_day_index


def _permission_error(target, winerror=5):
    error = PermissionError(errno.EACCES, "Access is denied", str(target))
    error.winerror = winerror
    return error


def _publication(tmp_path):
    day = "2026-09-10"
    frame = pd.DataFrame({
        "delivery_start_utc": local_delivery_day_index(day, timezone="Europe/Brussels"),
    })
    for model in ("residual_corrected", "residual_kalman"):
        for quantile, value in (("q10", 40.0), ("q50", 50.0), ("q90", 60.0)):
            frame[f"{model}__{quantile}"] = value
    result = SimpleNamespace(
        source_forecast=frame,
        kalman_view=SimpleNamespace(forecast=frame.copy(deep=True)),
    )
    source = tmp_path / "reports"
    source.mkdir()
    reports = {}
    for kind in ("autonomous", "kalman"):
        reports[kind] = source / f"{kind}.html"
        reports[kind].write_text("<html>Statistics Storm</html>", encoding="utf-8")
    reports["audit"] = source / "audit.json"
    reports["audit"].write_text(
        json.dumps({"zone": "BE", "delivery_day": day}), encoding="utf-8",
    )
    kwargs = dict(
        project_root=tmp_path, zone="BE", delivery_day=day, timezone="Europe/Brussels",
    )
    return result, reports, kwargs


def _files_under(directory):
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*") if path.is_file()
    }


def test_identical_locked_csvs_allow_changed_html_and_valid_manifest(tmp_path, monkeypatch):
    result, reports, kwargs = _publication(tmp_path)
    outputs = exports.publish_nuclear_exports(result, reports, **kwargs)
    locked = {Path(outputs[kind]).with_suffix(".csv") for kind in ("autonomous", "kalman")}
    before_csvs = {path: path.read_bytes() for path in locked}
    # Same-size content changes must still publish, rather than comparing size only.
    for kind in ("autonomous", "kalman"):
        reports[kind].write_text("<html>Statistics Model</html>", encoding="utf-8")
    replace = exports.os.replace
    attempted = []

    def deny_locked_csvs(source, target):
        target = Path(target)
        attempted.append(target)
        if target in locked:
            raise _permission_error(target)
        return replace(source, target)

    monkeypatch.setattr(exports.os, "replace", deny_locked_csvs)
    monkeypatch.setattr(exports.time, "sleep", lambda _: None)
    updated = exports.publish_nuclear_exports(result, reports, **kwargs)

    assert not locked.intersection(attempted)
    assert {path: path.read_bytes() for path in locked} == before_csvs
    for kind in ("autonomous", "kalman"):
        assert Path(updated[kind]).read_text(encoding="utf-8") == "<html>Statistics Model</html>"
    manifest_path = Path(updated["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert {record["variant"] for record in manifest["exports"]} == {
        "nuclear_autonomous", "nuclear_kalman",
    }
    for record in manifest["exports"]:
        assert len(record["files"]) == 3
        for artifact in record["files"]:
            actual = manifest_path.parent / artifact["path"]
            assert hashlib.sha256(actual.read_bytes()).hexdigest() == artifact["sha256"]
    assert not (manifest_path.parent / ".nuclear_publish.lock").exists()


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_transient_windows_lock_retries_atomic_replace(tmp_path, monkeypatch, winerror):
    source, target = tmp_path / "staged.csv", tmp_path / "forecast.csv"
    source.write_text("new content", encoding="utf-8")
    target.write_text("old content", encoding="utf-8")
    replace = exports.os.replace
    attempts = []
    sleeps = []

    def transient_lock(staged, destination):
        attempts.append((Path(staged), Path(destination)))
        if len(attempts) <= 2:
            assert target.read_text(encoding="utf-8") == "old content"
            raise _permission_error(destination, winerror)
        return replace(staged, destination)

    monkeypatch.setattr(exports.os, "replace", transient_lock)
    monkeypatch.setattr(exports.time, "sleep", sleeps.append)
    exports._replace_with_retry(source, target)

    assert attempts == [(source, target)] * 3
    assert len(sleeps) == 2
    assert target.read_text(encoding="utf-8") == "new content"
    assert not source.exists()


def test_persistent_changed_csv_lock_restores_previous_publication(tmp_path, monkeypatch):
    result, reports, kwargs = _publication(tmp_path)
    outputs = exports.publish_nuclear_exports(result, reports, **kwargs)
    destination = Path(outputs["manifest"]).parent
    before = _files_under(destination)
    locked = Path(outputs["kalman"]).with_suffix(".csv")
    # Force a real update before reaching the changed, locked Kalman CSV.
    reports["autonomous"].write_text("<html>Updated autonomous report</html>", encoding="utf-8")
    result.kalman_view.forecast["residual_kalman__q50"] = 51.0
    replace = exports.os.replace
    locked_attempts = []
    replaced = []

    def persistent_lock(source, target):
        target = Path(target)
        if target == locked:
            locked_attempts.append(target)
            raise _permission_error(target)
        replaced.append(target)
        return replace(source, target)

    monkeypatch.setattr(exports.os, "replace", persistent_lock)
    monkeypatch.setattr(exports.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError) as error:
        exports.publish_nuclear_exports(result, reports, **kwargs)

    assert locked.name in str(error.value)
    assert 1 < len(locked_attempts) < 20
    assert Path(outputs["autonomous"]) in replaced
    assert _files_under(destination) == before
    assert not (destination / ".nuclear_publish.lock").exists()
    assert not list(destination.glob(".nuclear_stage_*"))


def test_unrelated_io_error_is_not_retried(tmp_path, monkeypatch):
    source, target = tmp_path / "staged.csv", tmp_path / "forecast.csv"
    source.write_text("new", encoding="utf-8")
    target.write_text("old", encoding="utf-8")
    attempts = []
    sleeps = []

    def io_error(staged, destination):
        attempts.append((staged, destination))
        raise OSError(errno.EIO, "Storage failure")

    monkeypatch.setattr(exports.os, "replace", io_error)
    monkeypatch.setattr(exports.time, "sleep", sleeps.append)
    with pytest.raises(OSError, match="Storage failure"):
        exports._replace_with_retry(source, target)
    assert len(attempts) == 1
    assert not sleeps
    assert source.read_text(encoding="utf-8") == "new"
    assert target.read_text(encoding="utf-8") == "old"


@pytest.mark.parametrize("audit_write_fails", [False, True])
def test_double_failure_restores_other_files_and_preserves_blocked_backup(tmp_path, monkeypatch, audit_write_fails):
    result, reports, kwargs = _publication(tmp_path)
    outputs = exports.publish_nuclear_exports(result, reports, **kwargs)
    autonomous, kalman = Path(outputs["autonomous"]), Path(outputs["kalman"])
    destination = Path(outputs["manifest"]).parent
    before = {path: path.read_bytes() for path in (autonomous, kalman, Path(outputs["manifest"]))}
    for kind in ("autonomous", "kalman"):
        reports[kind].write_text(f"<html>New {kind} report</html>", encoding="utf-8")
    replace = exports.os.replace

    def publication_then_rollback_fail(source, target):
        source, target = Path(source), Path(target)
        if source.name == "current_nuclear_batch_manifest.json":
            raise OSError("original publication failure")
        if source.name.startswith("backup_") and target == kalman:
            raise _permission_error(target)
        return replace(source, target)

    monkeypatch.setattr(exports.os, "replace", publication_then_rollback_fail)
    monkeypatch.setattr(exports.time, "sleep", lambda _: None)
    if audit_write_fails:
        original_open = Path.open

        def block_audit(path, *args, **options):
            if path.name == "publication_recovery.json":
                raise PermissionError("recovery audit unavailable")
            return original_open(path, *args, **options)

        monkeypatch.setattr(Path, "open", block_audit)
    with pytest.raises(OSError) as captured:
        exports.publish_nuclear_exports(result, reports, **kwargs)

    stages = list(destination.glob(".nuclear_stage_*"))
    preserved = [path for stage in stages for path in stage.glob("backup_*")]
    assert autonomous.read_bytes() == before[autonomous], (
        "Restoration stopped after the locked Kalman report; "
        f"remaining backup files: {preserved}"
    )
    assert kalman.read_bytes() != before[kalman]
    manifest_path = Path(outputs["manifest"])
    assert manifest_path.read_bytes() == before[manifest_path]
    assert len(stages) == 1
    backup = next(path for path in preserved if path.read_bytes() == before[kalman])
    assert backup.read_bytes() == before[kalman]
    if audit_write_fails:
        assert "recovery audit unavailable" in str(captured.value)
    else:
        audit_path = stages[0] / "publication_recovery.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        assert audit["status"] == "rollback_incomplete"
        assert audit["publication_error"]["message"] == "original publication failure"
        blocked = next(item for item in audit["rollback"] if item["target"] == str(kalman))
        assert blocked["status"] == "failed"
        assert stages[0] / blocked["backup"] == backup
        assert hashlib.sha256(backup.read_bytes()).hexdigest() == blocked["backup_sha256"]
        restored = next(item for item in audit["rollback"] if item["target"] == str(autonomous))
        assert restored["status"] == "restored"
    assert str(stages[0]) in str(captured.value)
    assert str(captured.value.__cause__) == "original publication failure"
    assert not (destination / ".nuclear_publish.lock").exists()
    # A normal retry repairs the published generation without losing the
    # retained evidence of the previous partial rollback.
    monkeypatch.setattr(exports.os, "replace", replace)
    recovered = exports.publish_nuclear_exports(result, reports, **kwargs)
    manifest = json.loads(Path(recovered["manifest"]).read_text())
    for record in manifest["exports"]:
        for artifact in record["files"]:
            assert hashlib.sha256((destination / artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
    assert backup.read_bytes() == before[kalman]
    assert list(destination.glob(".nuclear_stage_*")) == stages


def test_failed_removal_of_new_file_does_not_stop_other_rollback_actions(tmp_path, monkeypatch):
    result, reports, kwargs = _publication(tmp_path)
    destination = tmp_path / "runs/exports/2026-09-10/be"
    locked = destination / "nuclear_kalman/forecast_be_2026-09-10_nuclear_kalman.html"
    replace, unlink = exports.os.replace, Path.unlink

    def fail_manifest(source, target):
        if Path(source).name == "current_nuclear_batch_manifest.json":
            raise OSError("manifest publication failed")
        return replace(source, target)

    def block_removal(path, *args, **options):
        if path == locked:
            raise _permission_error(path)
        return unlink(path, *args, **options)

    monkeypatch.setattr(exports.os, "replace", fail_manifest)
    monkeypatch.setattr(Path, "unlink", block_removal)
    with pytest.raises(exports.NuclearExportRecoveryError):
        exports.publish_nuclear_exports(result, reports, **kwargs)
    assert locked.is_file()
    assert not list((destination / "nuclear_autonomous").glob("*"))
    assert not locked.with_suffix(".csv").exists()
    assert not (destination / "current_nuclear_batch_manifest.json").exists()
    stage = next(destination.glob(".nuclear_stage_*"))
    audit = json.loads((stage / "publication_recovery.json").read_text())
    blocked = next(item for item in audit["rollback"] if item["target"] == str(locked))
    assert blocked["status"] == "failed"
    assert blocked["action"] == "remove_new_file" and blocked["backup"] is None
    assert any(item["status"] == "removed" for item in audit["rollback"])


@pytest.mark.parametrize("publication_fails", [False, True])
def test_temporary_cleanup_lock_never_changes_publication_outcome(tmp_path, monkeypatch, capsys, publication_fails):
    result, reports, kwargs = _publication(tmp_path)
    outputs = exports.publish_nuclear_exports(result, reports, **kwargs)
    destination = Path(outputs["manifest"]).parent
    original_files = {path: path.read_bytes() for path in destination.rglob("*") if path.is_file()}
    reports["kalman"].write_text("<html>Updated Kalman report</html>", encoding="utf-8")
    original_error = OSError("original publication failure")
    replace = exports.os.replace

    def maybe_fail(source, target):
        if publication_fails and Path(source).name == "current_nuclear_batch_manifest.json":
            raise original_error
        return replace(source, target)

    def locked_stage(path, *args, **options):
        assert Path(path).parent == destination
        assert Path(path).name.startswith(".nuclear_stage_")
        raise _permission_error(path)

    monkeypatch.setattr(exports.os, "replace", maybe_fail)
    monkeypatch.setattr(exports.shutil, "rmtree", locked_stage)
    if publication_fails:
        with pytest.raises(OSError) as captured:
            exports.publish_nuclear_exports(result, reports, **kwargs)
        assert captured.value is original_error
        assert all(path.read_bytes() == content for path, content in original_files.items())
    else:
        updated = exports.publish_nuclear_exports(result, reports, **kwargs)
        assert Path(updated["kalman"]).read_text() == "<html>Updated Kalman report</html>"
        manifest = json.loads(Path(updated["manifest"]).read_text())
        for record in manifest["exports"]:
            for item in record["files"]:
                assert hashlib.sha256((destination / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    assert "nettoyage temporaire incomplet" in capsys.readouterr().err
    assert len(list(destination.glob(".nuclear_stage_*"))) == 1
    assert not (destination / ".nuclear_publish.lock").exists()
