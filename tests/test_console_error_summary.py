"""Failure summaries from synthetic local logs, without network or scientific work."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from experiment_console.error_summary import FailureSummary, read_failure_summary
from experiment_console.adapters import AdapterRegistry
from experiment_console.manager import Manager
import experiment_console.manager as manager_module


PROXY_CAUSE = "[Nuclear] ECHEC : Saturn ProxyError: HTTPSConnectionPool(host='saturn.fixture.invalid', port=443): ProxyError('Unable to connect to proxy', NewConnectionError(HTTPConnection(host='127.0.0.1', port=9): connection refused))"
GENERIC_FAILURE = "[Nuclear Kalman] ECHEC sources : returncode=2"
TRAILER = "Journal et statut du batch : runs/logs/nuclear_kalman/explicit_fixture"


def summarize(*lines):
    summary = FailureSummary()
    for line in lines:
        summary.feed(line)
    return summary.message


def assert_saturn_proxy(message):
    assert isinstance(message, str)
    assert "saturn" in message.lower() and "proxy" in message.lower()
    assert "127.0.0.1:9" in message
    assert "Journal et statut" not in message and "returncode=2" not in message


@pytest.mark.parametrize("lines", [
    (PROXY_CAUSE, GENERIC_FAILURE, TRAILER),
    (GENERIC_FAILURE, PROXY_CAUSE, TRAILER),
])
def test_substantive_saturn_proxy_cause_outlives_generic_failure_and_trailing_log_location(lines):
    assert_saturn_proxy(summarize(*lines))


def test_proxy_without_known_source_is_never_attributed_to_saturn():
    message = summarize("ProxyError: HTTPConnectionPool(host='127.0.0.1', port=9): Unable to connect to proxy", TRAILER)
    assert message and "proxy" in message.lower()
    assert "saturn" not in message.lower()


@pytest.mark.parametrize("cause,retained", [
    ("[Nuclear] ECHEC : FileNotFoundError: missing observations fixture.csv", "fixture.csv"),
    ("[Nuclear] ECHEC : HTTPError: 401 Unauthorized for source fixture", "401"),
])
def test_other_real_error_types_keep_their_observed_cause(cause, retained):
    message = summarize(cause, GENERIC_FAILURE, TRAILER)
    assert message and retained in message
    assert "proxy" not in message.lower()


def test_direct_python_exception_is_more_informative_than_outer_return_code():
    message = summarize("FileNotFoundError: missing explicit_observations_fixture.csv", GENERIC_FAILURE, TRAILER)
    assert message and "explicit_observations_fixture.csv" in message
    assert "returncode=2" not in message


def test_outer_https_target_is_never_presented_as_the_proxy_address():
    message = summarize("[Nuclear] ECHEC : Saturn ProxyError: HTTPSConnectionPool(host='saturn.fixture.invalid', port=443): Unable to connect to proxy")
    assert message and "proxy" in message.lower()
    assert "saturn.fixture.invalid:443" not in message


def test_ordinary_info_and_report_locations_do_not_create_an_error():
    assert summarize("INFO lecture des données", "INFO 4 pays traités", TRAILER) is None


def test_secrets_are_redacted_before_summary_is_shortened():
    credential = "SYNTHETIC_SECRET_VALUE_" * 150
    message = summarize(f'[Nuclear] ECHEC : authentication rejected api_key="{credential}" password=OTHER_SYNTHETIC_PASSWORD', TRAILER)
    assert message
    assert "SYNTHETIC_SECRET_VALUE" not in message
    assert "OTHER_SYNTHETIC_PASSWORD" not in message
    assert len(message) < 4096


@pytest.mark.parametrize("terminated", [False, True])
def test_private_key_blocks_never_supply_an_error_summary_even_when_unterminated(terminated):
    summary = FailureSummary()
    summary.feed("-----BEGIN PRIVATE KEY-----")
    summary.feed("[Nuclear] ECHEC : SYNTHETIC_PRIVATE_KEY_BODY")
    if terminated:
        summary.feed("-----END PRIVATE KEY-----")
        summary.feed("[Nuclear] ECHEC : missing visible fixture.csv")
        assert summary.message and "fixture.csv" in summary.message
    else:
        summary.feed(TRAILER)
        assert summary.message is None
    assert "SYNTHETIC_PRIVATE_KEY_BODY" not in (summary.message or "")


def test_legacy_reader_uses_only_own_console_log_and_preserves_its_bytes(tmp_path):
    (tmp_path / "external.log").write_text(PROXY_CAUSE, encoding="utf-8")
    nested = tmp_path / "unrelated_run"
    nested.mkdir()
    (nested / "console.log").write_text(PROXY_CAUSE, encoding="utf-8")
    assert read_failure_summary(tmp_path) is None
    log = tmp_path / "console.log"
    log.write_text("\n".join([PROXY_CAUSE, GENERIC_FAILURE, TRAILER]), encoding="utf-8")
    before = hashlib.sha256(log.read_bytes()).hexdigest()
    assert_saturn_proxy(read_failure_summary(tmp_path))
    assert hashlib.sha256(log.read_bytes()).hexdigest() == before


def test_bounded_legacy_reader_does_not_expose_error_like_text_inside_unterminated_pem(tmp_path):
    log = tmp_path / "console.log"
    log.write_text("-----BEGIN PRIVATE KEY-----\n" + "A" * 100_000 + "\n[Nuclear] ECHEC : SYNTHETIC_LONG_PRIVATE_BODY\n", encoding="utf-8")
    before = log.read_bytes()
    message = read_failure_summary(tmp_path, max_bytes=512)
    assert message is None
    assert log.read_bytes() == before


def test_legacy_reader_refuses_windows_junction_or_directory_symlink(tmp_path):
    outside = tmp_path / "outside fixture"
    outside.mkdir()
    (outside / "console.log").write_text(PROXY_CAUSE, encoding="utf-8")
    link = tmp_path / "run metadata link"
    if os.name == "nt":
        result = subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(link), str(outside)],
                                capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        assert result.returncode == 0, result.stderr
    else:
        link.symlink_to(outside, target_is_directory=True)
    try:
        assert read_failure_summary(link) is None
    finally:
        link.rmdir() if os.name == "nt" else link.unlink()
    assert (outside / "console.log").read_text(encoding="utf-8") == PROXY_CAUSE


class ErrorFixtureRegistry:
    """The only launchable program in this registry is a tiny local test script."""

    def __init__(self, project, mode="failure"):
        self.project = project
        self.mode = mode
        self.script = project / "explicit_error_summary_fixture.py"
        self.script.write_text(
            "import sys,time\nfrom pathlib import Path\n"
            f"print({PROXY_CAUSE!r}, flush=True)\nprint({GENERIC_FAILURE!r}, flush=True)\nprint({TRAILER!r}, flush=True)\n"
            "output=Path(sys.argv[1]); output.mkdir(parents=True,exist_ok=True)\n"
            "(output/'fixture-ready').write_text('Explicit test fixture only')\n"
            "if sys.argv[2]=='cancel': time.sleep(20)\n"
            "raise SystemExit(0 if sys.argv[2]=='success' else 2)\n", encoding="utf-8")

    def primary_defaults(self):
        return {"delivery_day": "2026-09-12"}

    def prepare_primary_run(self, run_directory, write=False):
        run_directory = Path(run_directory)
        if write:
            (run_directory / "config.json").write_text('{"fixture":true}', encoding="utf-8")
        return {"command": [sys.executable, "-u", str(self.script), str(run_directory / "output"), self.mode],
                "cwd": str(self.project), "output_dir": str(run_directory / "output"),
                "config": {"fixture": True}, "delivery_day": "2026-09-12",
                "type": "fixture", "model": "explicit-error-summary-fixture", "resource_keys": [], "warnings": []}

    def validate_primary_sources(self, run):
        pass


@pytest.fixture
def error_manager(tmp_path):
    managers = []

    def create(*, mode="failure", start_scheduler=False):
        project = tmp_path / f"error fixture project {len(managers)}"
        project.mkdir()
        registry = ErrorFixtureRegistry(project, mode)
        manager = Manager(project, project / "state", sys.executable, registry=registry, start_scheduler=start_scheduler)
        managers.append(manager)
        return manager

    yield create
    for manager in managers:
        for run in manager.list_runs():
            if run["status"] in {"queued", "starting", "running", "cancelling"}:
                manager.cancel(run["id"])
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(run["status"] in {"starting", "running", "cancelling"} for run in manager.list_runs()):
            time.sleep(.05)
        manager.close()


def test_legacy_failed_primary_status_uses_read_only_log_fallback_without_updating_store(error_manager):
    manager = error_manager()
    launched = manager.launch_primary_run()
    run_id = launched["run"]["id"]
    run_dir = Path(manager.get(run_id)["run_dir"])
    log = run_dir / "console.log"
    log.write_text("\n".join([PROXY_CAUSE, GENERIC_FAILURE, TRAILER]), encoding="utf-8")
    manager.store.update(run_id, status="failed", return_code=2, activity=TRAILER)
    stored_before = manager.store.get(run_id)
    bytes_before = log.read_bytes()
    status = manager.primary_run_status()
    assert_saturn_proxy(status["latest"]["error"])
    assert manager.store.get(run_id) == stored_before
    assert "failure_summary" not in manager.store.get(run_id)
    assert log.read_bytes() == bytes_before


@pytest.mark.parametrize("status", ["succeeded", "cancelled"])
def test_terminal_nonfailure_never_exposes_stale_error_lines_or_summary(error_manager, status):
    manager = error_manager()
    run_id = manager.launch_primary_run()["run"]["id"]
    log = Path(manager.get(run_id)["run_dir"]) / "console.log"
    log.write_text(PROXY_CAUSE, encoding="utf-8")
    manager.store.update(run_id, status=status, activity=PROXY_CAUSE, failure_summary=PROXY_CAUSE)
    latest = manager.primary_run_status()["latest"]
    assert latest["status"] == status
    assert not latest.get("error")


@pytest.mark.parametrize("mode,status", [("failure", "failed"), ("success", "succeeded"), ("cancel", "cancelled")])
def test_actual_fixture_worker_persists_root_cause_only_when_primary_run_fails(error_manager, mode, status):
    manager = error_manager(mode=mode, start_scheduler=True)
    run_id = manager.launch_primary_run()["run"]["id"]
    ready = Path(manager.get(run_id)["output_dir"]) / "fixture-ready"
    deadline = time.monotonic() + 20
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(.05)
    assert ready.is_file(), "the tiny fixture must actually execute"
    if mode == "cancel":
        manager.cancel(run_id)
    while time.monotonic() < deadline:
        run = manager.get(run_id)
        if run["status"] in {"failed", "succeeded", "cancelled", "interrupted"}:
            break
        time.sleep(.05)
    assert run["status"] == status
    if mode == "failure":
        assert run["return_code"] == 2
        assert_saturn_proxy(run.get("failure_summary"))
        assert_saturn_proxy(manager.primary_run_status()["latest"]["error"])
    else:
        assert not run.get("failure_summary")
        assert not manager.primary_run_status()["latest"].get("error")


def test_real_registry_environment_guard_blocks_new_and_queued_work_without_changing_environment(tmp_path, monkeypatch):
    project = tmp_path / "environment guard fixture"
    project.mkdir()
    registry = AdapterRegistry(project, sys.executable)
    monkeypatch.setattr(registry, "primary_defaults", lambda: {"delivery_day": "2026-09-12"})
    monkeypatch.setattr(registry, "prepare_primary_run", lambda *args, **kwargs: pytest.fail("Guard must run before source preparation"))
    monkeypatch.setattr(registry, "validate_primary_sources", lambda *args, **kwargs: pytest.fail("Guard must run before source validation"))
    monkeypatch.setattr(manager_module.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Guard must not spawn any subprocess"))
    monkeypatch.setenv("CODEX_SANDBOX_NETWORK_DISABLED", "1")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    # Compare a digest so even a failing assertion cannot print real environment secrets.
    before = hashlib.sha256(json.dumps(dict(os.environ), sort_keys=True).encode()).hexdigest()
    manager = Manager(project, project / "state", sys.executable, registry=registry, start_scheduler=False)
    try:
        status = manager.primary_run_status()
        assert status["available"] is False and "bloqué" in status["warning"]
        with pytest.raises(ValueError, match="réseau.*bloqué"):
            manager.launch_primary_run()
        assert manager.store.list() == []
        assert not (manager.state_root / "executions").exists()
        run_id = "queued-environment-guard-fixture"
        run_dir = manager.state_root / "executions" / run_id
        manager.store.insert({"id": run_id, "source": "managed", "source_key": str(run_dir),
                              "adapter_id": "primary_nuclear_kalman", "status": "queued", "run_dir": str(run_dir),
                              "resource_keys": ["scientific-cache"], "created_at": "2026-09-12T00:00:00+00:00"})
        manager._tick()
        rejected = manager.store.get(run_id)
        assert rejected["status"] == "failed" and "bloqué" in rejected["activity"]
        assert not run_dir.exists()
        after = hashlib.sha256(json.dumps(dict(os.environ), sort_keys=True).encode()).hexdigest()
        assert after == before
    finally:
        manager.close()
