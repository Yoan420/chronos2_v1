"""Real process-manager integration tests; all commands below are tiny test fixtures.

No forecasting, model training, external data, or scientific run is executed.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time
import uuid

import psutil
import pytest

from experiment_console.manager import Manager
import experiment_console.manager as manager_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted", "unknown"}

# This script is written only inside pytest's temporary directory.  The child
# announces its own PID because Windows virtualenv launchers may be redirectors.
FIXTURE_SCRIPT = r'''
"""Explicit test fixture: bounded local process, no scientific computation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

mode, output, release = sys.argv[1:]
output = Path(output)
output.mkdir(parents=True, exist_ok=True)
with (output / "launches.txt").open("a", encoding="utf-8") as stream:
    stream.write("launched\n")
(output / "fixture.pid").write_text(str(os.getpid()), encoding="utf-8")
print("INFO explicit test fixture started", flush=True)
if mode == "secrets":
    print("api_key=synthetic-secret-value", flush=True)
    print("-----BEGIN PRIVATE KEY-----", flush=True)
    print("SYNTHETIC_PRIVATE_KEY_BODY_NO_REAL_SECRET", flush=True)
    print("-----END PRIVATE KEY-----", flush=True)
if mode == "child":
    code = """
import os, sys, time
from pathlib import Path
output = Path(sys.argv[1])
(output / 'child.pid').write_text(str(os.getpid()), encoding='utf-8')
while True:
    (output / 'heartbeat.txt').write_text(str(time.time()), encoding='utf-8')
    time.sleep(0.08)
"""
    subprocess.Popen([sys.executable, "-u", "-c", code, str(output)])
if mode in {"slow", "child"}:
    deadline = time.monotonic() + 45
    while not Path(release).exists():
        if time.monotonic() >= deadline:
            print("ERROR fixture release timeout", flush=True)
            sys.exit(19)
        time.sleep(0.04)
if mode == "fail":
    print("ERROR deliberate fixture failure", flush=True)
    sys.exit(7)
(output / "result.json").write_text(json.dumps({"fixture": True, "value": 42}), encoding="utf-8")
print("INFO explicit test fixture completed", flush=True)
'''


class FixtureRegistry:
    """Allow only identified test scripts; never invoke repository pipelines."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.script = directory / "explicit test fixture ; $literal.py"
        self.script.write_text(FIXTURE_SCRIPT, encoding="utf-8")

    def catalog(self):
        return {"types": [{"id": "fixture", "label": "Explicit test fixture"}]}

    def prepare(self, request, run_directory, write=False):
        mode = request.get("mode", "success")
        if mode not in {"success", "fail", "slow", "child", "secrets"}:
            raise ValueError("Unknown test fixture mode")
        run_directory = Path(run_directory)
        output_dir = run_directory / "results"
        config = {"fixture": True, "mode": mode, "resource": request.get("resource")}
        if write:
            run_directory.mkdir(parents=True, exist_ok=True)
            (run_directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return {
            "command": [sys.executable, "-u", str(self.script), mode, str(output_dir),
                        str(request.get("release_file", self.directory / "never-released"))],
            "cwd": str(PROJECT_ROOT),
            "output_dir": str(output_dir),
            "config": config,
            "type": "fixture",
            "model": "explicit-test-fixture",
            "resource_keys": [request["resource"]] if request.get("resource") else [],
            "warnings": ["Explicit test fixture; contains no real scientific results."],
        }


def wait_until(predicate, *, timeout=20, description="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.05)
    pytest.fail(f"Timed out waiting for {description}")


def wait_status(manager, run_id, statuses=TERMINAL, timeout=20):
    def check():
        run = manager.get(run_id)
        return run if run["status"] in statuses else None
    return wait_until(check, timeout=timeout, description=f"run {run_id} status in {statuses}")


def pid_is_active(pid):
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.fixture
def manager_factory(tmp_path):
    registry = FixtureRegistry(tmp_path)
    managers = []

    def create(*, state_root=None, max_concurrency=1, start_scheduler=True):
        manager = Manager(
            project_root=PROJECT_ROOT,
            state_root=state_root or tmp_path / "console-state",
            python_executable=sys.executable,
            max_concurrency=max_concurrency,
            registry=registry,
            start_scheduler=start_scheduler,
        )
        managers.append(manager)
        return manager

    yield create
    # Tests must leave neither fixture parents nor grandchildren alive, even if
    # an assertion fails. Recorded fixture PIDs are local to this temp directory.
    for manager in reversed(managers):
        for run in manager.list_runs():
            if run["status"] not in TERMINAL:
                try:
                    manager.cancel(run["id"])
                except (ValueError, RuntimeError):
                    pass
        manager.close()
    for pid_file in tmp_path.rglob("*.pid"):
        try:
            process = psutil.Process(int(pid_file.read_text(encoding="utf-8")))
            if process.pid != os.getpid():
                process.kill()
                process.wait(timeout=5)
        except (ValueError, psutil.NoSuchProcess, psutil.TimeoutExpired):
            pass


def launch(manager, **request):
    plan = manager.preview({"type": "fixture", "name": "Explicit process test", **request})
    return manager.launch(plan["id"], idempotency_key=uuid.uuid4().hex)


def test_success_persists_configuration_logs_and_result_after_restart(manager_factory):
    manager = manager_factory()
    run = launch(manager)
    finished = wait_status(manager, run["id"])
    assert finished["status"] == "succeeded"
    assert finished["return_code"] == 0
    assert finished["started_at"] and finished["finished_at"]
    output = Path(finished["output_dir"])
    assert json.loads((output / "result.json").read_text())["fixture"] is True
    assert json.loads((Path(finished["run_dir"]) / "config.json").read_text())["mode"] == "success"
    log_files = list(Path(finished["run_dir"]).glob("*.log"))
    assert log_files, "the complete log must persist outside browser memory"
    assert "explicit test fixture completed" in "\n".join(p.read_text(encoding="utf-8") for p in log_files)
    manager.close()
    restarted = manager_factory()
    assert restarted.get(run["id"])["status"] == "succeeded"
    assert [row["id"] for row in restarted.list_runs()] == [run["id"]]


def test_failure_records_exit_code_and_does_not_stop_next_run(manager_factory):
    manager = manager_factory()
    failed = wait_status(manager, launch(manager, mode="fail")["id"])
    assert failed["status"] == "failed"
    assert failed["return_code"] == 7
    next_run = wait_status(manager, launch(manager)["id"])
    assert next_run["status"] == "succeeded"


def test_persisted_logs_redact_synthetic_credentials_including_multiline_keys(manager_factory):
    manager = manager_factory()
    finished = wait_status(manager, launch(manager, mode="secrets")["id"])
    assert finished["status"] == "succeeded"
    logs = "\n".join(path.read_text(encoding="utf-8") for path in Path(finished["run_dir"]).glob("*.log"))
    assert "explicit test fixture completed" in logs
    assert "synthetic-secret-value" not in logs
    assert "SYNTHETIC_PRIVATE_KEY_BODY_NO_REAL_SECRET" not in logs


def test_concurrent_duplicate_launches_execute_only_once(manager_factory):
    manager = manager_factory()
    plan = manager.preview({"type": "fixture", "name": "Double-click test"})
    with ThreadPoolExecutor(max_workers=6) as pool:
        launches = list(pool.map(lambda _: manager.launch(plan["id"], "same-browser-request"), range(6)))
    assert len({run["id"] for run in launches}) == 1
    finished = wait_status(manager, launches[0]["id"])
    assert finished["status"] == "succeeded"
    assert len(manager.list_runs()) == 1
    assert (Path(finished["output_dir"]) / "launches.txt").read_text().splitlines() == ["launched"]


def test_plan_itself_cannot_launch_twice_with_different_idempotency_keys(manager_factory):
    manager = manager_factory()
    plan = manager.preview({"type": "fixture", "name": "Same reviewed plan"})
    first = manager.launch(plan["id"], "first-key")
    second = manager.launch(plan["id"], "different-key")
    assert first["id"] == second["id"]
    assert wait_status(manager, first["id"])["status"] == "succeeded"
    assert len(manager.list_runs()) == 1


def test_default_concurrency_queues_then_starts_after_first_finishes(manager_factory, tmp_path):
    manager = manager_factory()
    release = tmp_path / "release-first"
    first = launch(manager, mode="slow", release_file=str(release))
    wait_status(manager, first["id"], {"running"})
    second = launch(manager)
    time.sleep(0.5)
    assert manager.get(second["id"])["status"] == "queued"
    assert not (Path(second["output_dir"]) / "fixture.pid").exists()
    release.touch()
    assert wait_status(manager, first["id"])["status"] == "succeeded"
    assert wait_status(manager, second["id"])["status"] == "succeeded"


def test_resource_conflict_blocks_only_conflicting_run(manager_factory, tmp_path):
    manager = manager_factory(max_concurrency=2)
    release = tmp_path / "release-resource"
    first = launch(manager, mode="slow", release_file=str(release), resource="shared-cache")
    wait_status(manager, first["id"], {"running"})
    conflicting = launch(manager, resource="shared-cache")
    independent = launch(manager, resource="different-cache")
    assert wait_status(manager, independent["id"])["status"] == "succeeded"
    assert manager.get(conflicting["id"])["status"] == "queued"
    release.touch()
    assert wait_status(manager, first["id"])["status"] == "succeeded"
    assert wait_status(manager, conflicting["id"])["status"] == "succeeded"


def test_cancel_queued_run_never_creates_child_process(manager_factory):
    manager = manager_factory(start_scheduler=False)
    run = launch(manager)
    assert run["status"] == "queued"
    manager.cancel(run["id"])
    assert manager.get(run["id"])["status"] == "cancelled"
    assert not (Path(run["output_dir"]) / "fixture.pid").exists()


def test_cancel_running_run_stops_its_grandchild(manager_factory):
    manager = manager_factory()
    run = launch(manager, mode="child")
    output = Path(run["output_dir"])
    wait_until(lambda: (output / "child.pid").exists(), description="fixture grandchild readiness")
    parent_pid = int((output / "fixture.pid").read_text())
    child_pid = int((output / "child.pid").read_text())
    assert pid_is_active(parent_pid) and pid_is_active(child_pid)
    manager.cancel(run["id"])
    assert wait_status(manager, run["id"])["status"] == "cancelled"
    wait_until(lambda: not pid_is_active(parent_pid) and not pid_is_active(child_pid),
               description="entire fixture process tree terminated")


def test_backend_restart_keeps_worker_and_child_alive_then_recovers_completion(manager_factory, tmp_path):
    manager = manager_factory()
    release = tmp_path / "release-after-restart"
    run = launch(manager, mode="child", release_file=str(release))
    output = Path(run["output_dir"])
    wait_until(lambda: (output / "child.pid").exists(), description="child ready before backend restart")
    parent_pid = int((output / "fixture.pid").read_text())
    child_pid = int((output / "child.pid").read_text())
    manager.close()
    restarted = manager_factory()
    recovered = restarted.get(run["id"])
    assert recovered["status"] == "running"
    assert pid_is_active(parent_pid) and pid_is_active(child_pid)
    assert (output / "launches.txt").read_text().splitlines() == ["launched"]
    release.touch()
    assert wait_status(restarted, run["id"])["status"] == "succeeded"
    wait_until(lambda: not pid_is_active(child_pid), description="child cleanup after normal supervisor exit")


def test_reconcile_disappeared_worker_marks_interrupted(manager_factory):
    manager = manager_factory(start_scheduler=False)
    run = launch(manager)
    manager.store.update(run["id"], status="running", worker_pid=99999999,
                         worker_created=1.0, started_at="2026-09-11T00:00:00+00:00")
    manager.close()
    restarted = manager_factory(start_scheduler=False)
    restarted.reconcile()
    recovered = restarted.get(run["id"])
    assert recovered["status"] == "interrupted"
    assert recovered["finished_at"] is None, "actual finish time is unavailable after worker disappearance"
    assert recovered["recovery_at"]


def test_reconcile_cannot_overwrite_completion_published_during_pid_check(manager_factory, monkeypatch):
    manager = manager_factory(start_scheduler=False)
    run = launch(manager)
    manager.store.update(run["id"], status="running", worker_pid=99999999, worker_created=1.0)

    def completes_during_check(pid, created):
        manager.store.update(run["id"], status="succeeded", return_code=0,
                             finished_at="2026-09-11T12:00:00+00:00")
        return False

    monkeypatch.setattr(manager_module, "alive", completes_during_check)
    manager.reconcile()
    recovered = manager.get(run["id"])
    assert recovered["status"] == "succeeded"
    assert recovered["return_code"] == 0


def test_unverifiable_worker_identity_retains_resource_reservation(manager_factory, monkeypatch):
    manager = manager_factory(start_scheduler=False)
    run = launch(manager, resource="shared-cache")
    manager.store.update(run["id"], status="running", worker_pid=99999999, worker_created=1.0)
    monkeypatch.setattr(manager_module, "alive", lambda pid, created: None)
    manager.reconcile()
    recovered = manager.get(run["id"])
    assert recovered["status"] == "running"
    assert recovered["resource_keys"] == ["shared-cache"]
    assert recovered["recovery_warning"]


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object supervisor-crash guarantee")
def test_killing_supervisor_interrupts_run_and_cleans_children(manager_factory):
    manager = manager_factory()
    run = launch(manager, mode="child")
    output = Path(run["output_dir"])
    wait_until(lambda: (output / "child.pid").exists(), description="child ready before worker crash")
    parent_pid = int((output / "fixture.pid").read_text())
    child_pid = int((output / "child.pid").read_text())
    actual = manager.store.get(run["id"])
    assert actual["worker_pid"] not in {parent_pid, child_pid, os.getpid()}
    worker = psutil.Process(actual["worker_pid"])
    worker.kill()
    worker.wait(timeout=10)
    manager.reconcile()
    assert wait_status(manager, run["id"])["status"] == "interrupted"
    wait_until(lambda: not pid_is_active(parent_pid) and not pid_is_active(child_pid),
               description="Job Object cleans process tree after supervisor crash")
