"""Primary-run dates: PowerShell DryRun and tiny Python fixtures only."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from experiment_console.adapters import AdapterRegistry
from experiment_console.manager import Manager
from experiment_console.server import ConsoleHTTPServer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PrimaryFixtureRegistry:
    """Injected test registry; commands cannot invoke any scientific script."""

    def __init__(self, project):
        self.project = project
        self.mode = "success"
        self.script = project / "explicit_primary_process_fixture.py"
        self.script.write_text(
            "import json, sys\nfrom pathlib import Path\n"
            "print('INFO explicit primary process fixture', flush=True)\n"
            "output = Path(sys.argv[1]); output.mkdir(parents=True, exist_ok=True)\n"
            "(output / 'fixture-result.json').write_text(json.dumps({'fixture': True}))\n"
            "raise SystemExit(7 if sys.argv[2] == 'failure' else 0)\n", encoding="utf-8")

    def catalog(self):
        return []

    def primary_defaults(self):
        return {"delivery_day": "2026-09-12", "countries": ["BE", "DE", "FR", "NL"],
                "device": "auto", "threads": 4, "workers": 4,
                "sync_enabled": True, "attribution_enabled": False}

    def prepare_primary_run(self, run_directory, write=False):
        directory = Path(run_directory)
        output = directory / "fixture-results"
        defaults = self.primary_defaults()
        if write:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "config.json").write_text(json.dumps(defaults), encoding="utf-8")
        return {"command": [sys.executable, "-u", str(self.script), str(output), self.mode],
                "cwd": str(self.project), "output_dir": str(output), "config": defaults,
                "config_path": str(directory / "config.json"), "primary_defaults": defaults,
                "delivery_day": defaults["delivery_day"],
                "type": "fixture", "model": "explicit-primary-fixture",
                "resource_keys": ["scientific-cache", "nyx-primary-pipeline", "canonical-publications"],
                "warnings": ["Explicit primary process fixture; no scientific results."]}

    def validate_primary_sources(self, run):
        pass


class DatedPrimaryFixtureRegistry(PrimaryFixtureRegistry):
    """Date-aware fixture alongside the original zero-argument registry contract."""

    def __init__(self, project):
        super().__init__(project)
        self.tomorrow = "2026-09-12"
        self.prepare_calls = []
        self.script.write_text(
            "import json, sys\nfrom pathlib import Path\n"
            "print('INFO explicit dated primary process fixture', flush=True)\n"
            "output = Path(sys.argv[1]); output.mkdir(parents=True, exist_ok=True)\n"
            "(output / 'fixture-result.json').write_text(json.dumps("
            "{'fixture': True, 'delivery_day': sys.argv[3]}))\n"
            "raise SystemExit(7 if sys.argv[2] == 'failure' else 0)\n", encoding="utf-8")

    def primary_defaults(self, delivery_day=None):
        return {**super().primary_defaults(), "delivery_day": delivery_day or self.tomorrow}

    def prepare_primary_run(self, run_directory, write=False, *, delivery_day=None):
        self.prepare_calls.append((delivery_day, write))
        prepared = super().prepare_primary_run(run_directory, write=False)
        defaults = self.primary_defaults(delivery_day)
        prepared.update(config=defaults, primary_defaults=defaults, delivery_day=defaults["delivery_day"])
        prepared["command"].append(defaults["delivery_day"])
        if write:
            Path(run_directory).mkdir(parents=True, exist_ok=True)
            Path(prepared["config_path"]).write_text(json.dumps(defaults), encoding="utf-8")
        return prepared


@pytest.fixture
def primary_run_api(tmp_path):
    instances = []

    def create(*, start_scheduler=False, registry_type=PrimaryFixtureRegistry):
        project = tmp_path / f"primary fixture project {len(instances)}"
        project.mkdir()
        registry = registry_type(project)
        state = project / "isolated-console-state"
        manager = Manager(project, state, sys.executable, registry=registry, start_scheduler=start_scheduler)
        server = ConsoleHTTPServer(("127.0.0.1", 0), manager)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        client = httpx.Client(base_url=base, trust_env=False, timeout=15)
        api = SimpleNamespace(project=project, state=state, registry=registry, manager=manager,
                              server=server, thread=thread, client=client,
                              headers={"X-Console-Token": server.token, "Origin": base})
        instances.append(api)
        return api

    yield create
    for api in instances:
        for run in api.manager.list_runs():
            if run["status"] in {"queued", "starting", "running", "cancelling"}:
                api.manager.cancel(run["id"])
        api.client.close()
        api.server.shutdown()
        api.server.server_close()
        api.thread.join(timeout=3)
        api.manager.close()
        assert not api.thread.is_alive()


def post_primary(api, *, key=None, body=None):
    headers = {**api.headers, **({"Idempotency-Key": key} if key is not None else {})}
    return api.client.post("/api/primary-run", json={} if body is None else body, headers=headers)


def restart_backend(api):
    api.manager.close()
    api.manager = Manager(api.project, api.state, sys.executable, registry=api.registry, start_scheduler=False)
    api.server.manager = api.manager


def test_primary_defaults_match_actual_powershell_dryrun_without_launching_science(tmp_path):
    powershell = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    assert powershell.is_file(), "This acceptance test requires the real Windows PowerShell launcher"
    script = PROJECT_ROOT / "NuclearKalman.ps1"
    before = script.read_bytes()
    result = subprocess.run([str(powershell), "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(script), "-DryRun"],
                            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30, check=True,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    prefix = "Commande (argv, shell=False): "
    command = json.loads(next(line[len(prefix):] for line in result.stdout.splitlines() if line.startswith(prefix)))
    registry = AdapterRegistry(PROJECT_ROOT, sys.executable)
    defaults = registry.primary_defaults()
    assert defaults["countries"] == command[command.index("--zones") + 1:command.index("--delivery-day")]
    for key, flag in [("delivery_day", "--delivery-day"), ("device", "--device")]:
        assert defaults[key] == command[command.index(flag) + 1]
    for key in ("threads", "workers"):
        assert defaults[key] == int(command[command.index("--" + key) + 1])
    assert defaults["sync_enabled"] == ("--skip-observed-sync" not in command)
    assert defaults["attribution_enabled"] == ("--with-attribution" in command)
    preview_root = PROJECT_ROOT / "tmp" / f"primary-dryrun-only-{time.time_ns()}"
    prepared = registry.prepare_primary_run(preview_root, write=False)
    actual = prepared["command"]
    assert Path(actual[0]).resolve() == powershell.resolve()
    assert actual[actual.index("-File") + 1:] == [str(script), "-NoOpen"]
    assert not preview_root.exists()
    assert script.read_bytes() == before


def test_primary_status_exposes_defaults_and_post_with_no_parameters_queues_run(primary_run_api):
    api = primary_run_api()
    status = api.client.get("/api/primary-run")
    assert status.status_code == 200, status.text
    assert status.json()["available"] is True
    assert status.json()["defaults"]["countries"] == ["BE", "DE", "FR", "NL"]
    assert status.json()["active"] is None
    response = post_primary(api)
    assert response.status_code in {200, 202}, response.text
    run = response.json()["run"]
    assert run["status"] == "queued"
    assert len(api.manager.list_runs()) == 1
    status = api.client.get("/api/primary-run").json()
    assert status["active"]["id"] == run["id"]


def test_primary_launch_refuses_a_configured_python_different_from_script_default(tmp_path):
    different_python = tmp_path / "python.exe"
    different_python.write_bytes(b"Non-executable test fixture; this path must never be launched")
    registry = AdapterRegistry(PROJECT_ROOT, different_python)
    with pytest.raises(ValueError, match="Python"):
        registry.primary_defaults()  # Uses only the existing script's -DryRun.


@pytest.mark.parametrize("body", [{"countries": ["FR"]}, {"command": ["arbitrary.exe"]}, {"parameters": {"threads": 99}}])
def test_primary_endpoint_does_not_accept_scientific_or_command_overrides(primary_run_api, body):
    api = primary_run_api()
    response = post_primary(api, body=body)
    assert response.status_code == 400, response.text
    assert api.manager.list_runs() == []


def test_concurrent_zero_parameter_clicks_and_backend_restart_reuse_one_durable_run(primary_run_api):
    api = primary_run_api()
    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(pool.map(lambda _: post_primary(api), range(6)))
    assert all(response.status_code in {200, 202} for response in responses), [response.text for response in responses]
    identifiers = {response.json()["run"]["id"] for response in responses}
    assert len(identifiers) == 1
    run_id = identifiers.pop()
    assert len(api.manager.list_runs()) == 1
    restart_backend(api)
    retried = post_primary(api)
    assert retried.status_code in {200, 202}, retried.text
    assert retried.json()["run"]["id"] == run_id
    assert retried.json()["already_active"] is True
    assert len(api.manager.list_runs()) == 1


def test_explicit_idempotency_key_survives_completion_and_restart(primary_run_api):
    api = primary_run_api()
    response = post_primary(api, key="synthetic-durable-primary-click")
    assert response.status_code in {200, 202}, response.text
    run_id = response.json()["run"]["id"]
    api.manager.store.update(run_id, status="succeeded", return_code=0, finished_at="2026-09-12T08:00:00+00:00")
    restart_backend(api)
    duplicate = post_primary(api, key="synthetic-durable-primary-click")
    assert duplicate.status_code in {200, 202}, duplicate.text
    assert duplicate.json()["run"]["id"] == run_id
    assert len(api.manager.list_runs()) == 1


@pytest.mark.parametrize("mode,expected,code", [("success", "succeeded", 0), ("failure", "failed", 7)])
def test_primary_queue_runs_only_tiny_fixture_and_publishes_real_completion(primary_run_api, mode, expected, code):
    api = primary_run_api(start_scheduler=True)
    api.registry.mode = mode
    response = post_primary(api)
    assert response.status_code in {200, 202}, response.text
    run_id = response.json()["run"]["id"]
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        run = api.manager.get(run_id)
        if run["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
            break
        time.sleep(.05)
    assert run["status"] == expected, run
    assert run["return_code"] == code
    assert json.loads((Path(run["output_dir"]) / "fixture-result.json").read_text())["fixture"] is True
    current = api.client.get("/api/primary-run").json()
    assert current["latest"]["id"] == run_id
    assert current["latest"]["status"] == expected


@pytest.mark.parametrize("delivery_day", ["2024-02-29", "2026-09-01"])
def test_selected_date_matches_actual_powershell_dryrun_and_canonical_output(delivery_day):
    registry = AdapterRegistry(PROJECT_ROOT, sys.executable)
    script = PROJECT_ROOT / "NuclearKalman.ps1"
    before = script.read_bytes()
    command = registry._primary_command(delivery_day)
    assert command[command.index("-File") + 1:] == [str(script), "-NoOpen", "-DeliveryDay", delivery_day]
    # This is the only execution of a real pipeline entry point: -DryRun is mandatory.
    result = subprocess.run(command + ["-DryRun"], cwd=PROJECT_ROOT, capture_output=True,
                            text=True, timeout=30, check=True, shell=False,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    prefix = "Commande (argv, shell=False): "
    resolved = json.loads(next(line[len(prefix):] for line in result.stdout.splitlines() if line.startswith(prefix)))
    assert resolved[resolved.index("--delivery-day") + 1] == delivery_day
    expected_output = PROJECT_ROOT / f"runs/reports/model_storm/CWE_Model_Storm_{delivery_day}.html"
    assert Path(resolved[resolved.index("--output") + 1]).resolve() == expected_output
    defaults = registry.primary_defaults(delivery_day)
    assert defaults["delivery_day"] == delivery_day
    assert Path(defaults["output"]).resolve() == expected_output
    preview = PROJECT_ROOT / "tmp" / f"selected-date-dryrun-only-{time.time_ns()}"
    prepared = registry.prepare_primary_run(preview, write=False, delivery_day=delivery_day)
    assert prepared["command"] == command
    assert prepared["delivery_day"] == delivery_day
    assert prepared["config"]["defaults_at_request"]["delivery_day"] == delivery_day
    assert prepared["request"]["delivery_day"] == delivery_day
    assert not preview.exists()
    assert script.read_bytes() == before


INVALID_DELIVERY_DAYS = [None, 20260912, True, [], {}, "", " 2026-09-12", "2026-09-12 ",
                         "2026-9-12", "2026-09-1", "2026-02-30", "2025-02-29",
                         "2026-13-01", "2026-09-12T00:00:00", "2026-09-12; echo unexpected"]


@pytest.mark.parametrize("delivery_day", INVALID_DELIVERY_DAYS)
def test_invalid_selected_dates_are_rejected_before_preparation_or_state_writes(primary_run_api, monkeypatch, delivery_day):
    api = primary_run_api(registry_type=DatedPrimaryFixtureRegistry)
    monkeypatch.setattr(api.registry, "prepare_primary_run",
                        lambda *args, **kwargs: pytest.fail("Invalid dates must not prepare a run"))
    def state_files():
        # Windows deliberately denies reads of a live FileLock. All persisted
        # database and execution files remain covered by the byte comparison.
        return {path.relative_to(api.state): path.read_bytes() for path in api.state.rglob("*")
                if path.is_file() and path.name != "backend.lock"}
    before = state_files()
    response = post_primary(api, body={"delivery_day": delivery_day})
    assert response.status_code == 400, response.text
    # None is the manager's deliberate default mode, while explicit HTTP null is invalid.
    if delivery_day is not None:
        with pytest.raises(ValueError):
            api.manager.launch_primary_run(delivery_day=delivery_day)
    assert api.manager.list_runs() == []
    assert not (api.state / "executions").exists()
    assert state_files() == before


def test_explicit_date_is_durable_and_not_reset_at_dispatch_across_simulated_midnight(primary_run_api):
    api = primary_run_api(registry_type=DatedPrimaryFixtureRegistry)
    selected = "2024-02-29"
    response = post_primary(api, body={"delivery_day": selected})
    assert response.status_code == 202, response.text
    run_id = response.json()["run"]["id"]
    queued = api.manager.get(run_id)
    assert queued["delivery_day"] == queued["request"]["delivery_day"] == selected
    assert json.loads(Path(queued["config_path"]).read_text())["delivery_day"] == selected
    metadata = json.loads((Path(queued["run_dir"]) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["request"]["delivery_day"] == metadata["delivery_day"] == selected
    assert api.registry.prepare_calls == [(selected, False), (selected, True)]
    api.registry.tomorrow = "2026-09-13"  # Simulate midnight while the accepted run is queued.
    restart_backend(api)
    assert api.client.get("/api/primary-run").json()["defaults"]["delivery_day"] == "2026-09-13"
    api.manager.start()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        completed = api.manager.get(run_id)
        if completed["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
            break
        time.sleep(.05)
    assert completed["status"] == "succeeded", completed
    assert completed["delivery_day"] == completed["request"]["delivery_day"] == selected
    assert json.loads((Path(completed["output_dir"]) / "fixture-result.json").read_text()) == {
        "fixture": True, "delivery_day": selected}
    current = api.client.get("/api/primary-run").json()
    assert current["latest"]["delivery_day"] == selected
    assert current["defaults"]["delivery_day"] == "2026-09-13"


def test_concurrent_same_selected_date_clicks_reuse_one_durable_run(primary_run_api):
    api = primary_run_api(registry_type=DatedPrimaryFixtureRegistry)
    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(pool.map(lambda _: post_primary(api, body={"delivery_day": "2024-02-29"}), range(6)))
    assert all(response.status_code == 202 for response in responses), [response.text for response in responses]
    identifiers = {response.json()["run"]["id"] for response in responses}
    assert len(identifiers) == 1
    assert len(api.manager.list_runs()) == 1
    restart_backend(api)
    repeated = post_primary(api, body={"delivery_day": "2024-02-29"})
    assert repeated.status_code == 202, repeated.text
    assert repeated.json()["run"]["id"] in identifiers
    assert repeated.json()["already_active"] is True


@pytest.mark.parametrize("first,conflicting", [
    ({"delivery_day": "2026-09-12"}, {"delivery_day": "2026-09-13"}),
    ({}, {"delivery_day": "2026-09-12"}),
    ({"delivery_day": "2026-09-12"}, {}),
])
def test_active_run_rejects_different_dates_or_default_modes_without_creating_run(primary_run_api, first, conflicting):
    api = primary_run_api(registry_type=DatedPrimaryFixtureRegistry)
    initial = post_primary(api, body=first)
    assert initial.status_code == 202, initial.text
    calls = list(api.registry.prepare_calls)
    conflict = post_primary(api, body=conflicting)
    assert conflict.status_code == 409, conflict.text
    assert api.registry.prepare_calls == calls
    assert [run["id"] for run in api.manager.list_runs()] == [initial.json()["run"]["id"]]
    # Repeating the actual accepted request remains allowed after the conflict.
    repeated = post_primary(api, body=first)
    assert repeated.status_code == 202, repeated.text
    assert repeated.json()["run"]["id"] == initial.json()["run"]["id"]


@pytest.mark.parametrize("first,conflicting", [
    ({"delivery_day": "2024-02-29"}, {"delivery_day": "2024-03-01"}),
    ({}, {"delivery_day": "2026-09-12"}),
    ({"delivery_day": "2026-09-12"}, {}),
])
def test_idempotency_key_keeps_original_date_and_mode_after_completion_and_restart(primary_run_api, first, conflicting):
    api = primary_run_api(registry_type=DatedPrimaryFixtureRegistry)
    key = "explicit-date-durable-fixture-request"
    initial = post_primary(api, key=key, body=first)
    assert initial.status_code == 202, initial.text
    run_id = initial.json()["run"]["id"]
    api.manager.store.update(run_id, status="succeeded", return_code=0, finished_at="2026-09-12T08:00:00+00:00")
    restart_backend(api)
    conflict = post_primary(api, key=key, body=conflicting)
    assert conflict.status_code == 409, conflict.text
    repeated = post_primary(api, key=key, body=first)
    assert repeated.status_code == 202, repeated.text
    assert repeated.json()["run"]["id"] == run_id
    assert repeated.json()["run"]["status"] == "succeeded"
    assert len(api.manager.list_runs()) == 1
