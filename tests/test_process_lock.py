from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import psutil
import pytest

import chronos2_hourly.process_lock as locks


def test_lock_publishes_owner_and_cleans_up_on_success_or_error(tmp_path):
    path = tmp_path / "nested" / "run.lock"
    for fail in [False, True]:
        try:
            with locks.exclusive_process_lock(path):
                owner = json.loads(path.read_text())
                assert owner["pid"] == os.getpid()
                assert owner["process_create_time"] == psutil.Process().create_time()
                assert owner["hostname"] == socket.gethostname()
                assert len(owner["owner_token"]) == 32
                if fail:
                    raise RuntimeError("work failed")
        except RuntimeError as exc:
            assert str(exc) == "work failed"
        assert not path.exists()


def test_same_process_second_acquisition_cannot_reenter_or_remove_owner(tmp_path):
    path = tmp_path / "run.lock"
    with locks.exclusive_process_lock(path):
        before = path.read_bytes()
        with pytest.raises(ValueError, match="Verrou"):
            with locks.exclusive_process_lock(path):
                pytest.fail("second owner entered")
        assert path.read_bytes() == before


@pytest.mark.parametrize("raw", [
    b"other worker", b"", b"[]", b'{"pid": true}', b'{"pid": -1}',
    b'{"pid": "12"}', b'{"pid": 1, "created_at": "yesterday"}',
    b'{"pid": 1, "created_at": "2026-09-10T08:00:00"}',
    b'{"pid": 1, "process_create_time": "invalid"}',
    b'{"pid": 1, "process_create_time": NaN}',
])
def test_unreadable_or_invalid_legacy_sentinel_is_preserved(tmp_path, raw):
    path = tmp_path / "run.lock"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="Verrou"):
        with locks.exclusive_process_lock(path):
            pytest.fail("invalid lock accepted")
    assert path.read_bytes() == raw


def test_live_legacy_pid_only_sentinel_is_respected(tmp_path):
    path = tmp_path / "run.lock"
    raw = json.dumps({"pid": os.getpid()}).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="encore actif"):
        with locks.exclusive_process_lock(path):
            pytest.fail("legacy owner ignored")
    assert path.read_bytes() == raw


def test_live_legacy_pid_and_created_at_sentinel_is_respected(tmp_path):
    path = tmp_path / "run.lock"
    raw = json.dumps({"pid": os.getpid(), "created_at": datetime.now(timezone.utc).isoformat()}).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="encore actif"):
        with locks.exclusive_process_lock(path):
            pytest.fail("legacy owner ignored")
    assert path.read_bytes() == raw


@pytest.mark.parametrize("legacy", [False, True])
def test_demonstrably_reused_pid_is_recovered(tmp_path, legacy):
    path = tmp_path / "run.lock"
    previous_start = psutil.Process().create_time() - 20.0
    record = {"pid": os.getpid()}
    if legacy:
        record["created_at"] = datetime.fromtimestamp(previous_start, timezone.utc).isoformat()
    else:
        record["process_create_time"] = previous_start
    path.write_text(json.dumps(record))
    with locks.exclusive_process_lock(path):
        assert json.loads(path.read_text())["process_create_time"] > previous_start
    assert not path.exists()


def test_different_hostname_never_uses_local_pid_as_proof(tmp_path):
    path = tmp_path / "run.lock"
    raw = json.dumps({"pid": 99999999, "hostname": "another-computer"}).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="Verrou"):
        with locks.exclusive_process_lock(path):
            pytest.fail("foreign host owner ignored")
    assert path.read_bytes() == raw


def test_access_denied_owner_is_not_guessed_dead(tmp_path, monkeypatch):
    path = tmp_path / "run.lock"
    raw = json.dumps({"pid": 1234}).encode()
    path.write_bytes(raw)

    def denied(_pid):
        raise psutil.AccessDenied(pid=1234)

    monkeypatch.setattr(locks.psutil, "Process", denied)
    with pytest.raises(ValueError, match="Impossible de verifier"):
        with locks.exclusive_process_lock(path):
            pytest.fail("unverifiable owner ignored")
    assert path.read_bytes() == raw


def test_release_only_removes_the_acquirers_token(tmp_path):
    path = tmp_path / "run.lock"
    replacement = {"pid": os.getpid(), "owner_token": "another owner"}
    with locks.exclusive_process_lock(path):
        path.write_text(json.dumps(replacement))
    assert json.loads(path.read_text()) == replacement


def test_legacy_race_during_atomic_publication_keeps_legacy_owner(tmp_path, monkeypatch):
    path = tmp_path / "run.lock"
    raw = json.dumps({"pid": os.getpid()}).encode()
    real_link = locks.os.link

    def competing_creation(source, target):
        path.write_bytes(raw)
        return real_link(source, target)

    monkeypatch.setattr(locks.os, "link", competing_creation)
    with pytest.raises(ValueError, match="autre processus"):
        with locks.exclusive_process_lock(path):
            pytest.fail("legacy race replaced the owner")
    assert path.read_bytes() == raw
    assert not list(tmp_path.glob(".run.lock.*.tmp"))


_CHILD = """
import json, os, sys
from pathlib import Path
from chronos2_hourly.process_lock import exclusive_process_lock
path, ready, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
if mode == 'legacy':
    with path.open('x') as stream:
        json.dump({'pid': os.getpid()}, stream)
    ready.write_text('ready')
    sys.stdin.read()
    path.unlink()
else:
    with exclusive_process_lock(path):
        ready.write_text('ready')
        sys.stdin.read()
"""


@contextmanager
def _holder(path, *, legacy=False):
    ready = path.with_suffix(".ready")
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(path), str(ready), "legacy" if legacy else "current"],
        cwd=Path(__file__).resolve().parents[1], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, **options,
    )
    owner_pid = None
    try:
        deadline = time.monotonic() + 15
        while not ready.exists():
            if child.poll() is not None:
                _, error = child.communicate(timeout=5)
                pytest.fail(f"lock holder exited: {error.decode(errors='replace')}")
            if time.monotonic() >= deadline:
                pytest.fail("lock holder readiness timed out")
            time.sleep(0.025)
        # Windows venv python.exe can be a redirector whose worker owns the
        # lock.  Kill that recorded owner, not merely its launching parent.
        owner_pid = json.loads(path.read_text())["pid"]
        child.lock_owner_pid = owner_pid
        yield child
    finally:
        if owner_pid is not None and owner_pid != os.getpid():
            try:
                owner = psutil.Process(owner_pid)
                owner.kill()
                owner.wait(timeout=10)
            except psutil.NoSuchProcess:
                pass
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)


@pytest.mark.parametrize("legacy", [False, True])
def test_real_other_process_is_excluded_then_recovered_after_abrupt_death(tmp_path, legacy):
    path = tmp_path / "run.lock"
    with _holder(path, legacy=legacy) as child:
        before = path.read_bytes()
        with pytest.raises(ValueError, match="Verrou"):
            with locks.exclusive_process_lock(path):
                pytest.fail("simultaneous owners entered")
        assert path.read_bytes() == before
        owner = psutil.Process(child.lock_owner_pid)
        owner.kill()
        owner.wait(timeout=10)
        child.wait(timeout=10)
        assert path.exists(), "an abrupt stop should leave the sentinel"
        with locks.exclusive_process_lock(path):
            assert json.loads(path.read_text())["pid"] == os.getpid()
        assert not path.exists()


def test_os_guard_protects_against_a_forged_dead_json_owner(tmp_path):
    path = tmp_path / "run.lock"
    with _holder(path):
        raw = b'{"pid": 99999999}'
        path.write_bytes(raw)
        with pytest.raises(ValueError, match="verrou systeme"):
            with locks.exclusive_process_lock(path):
                pytest.fail("OS guard ignored")
        assert path.read_bytes() == raw
