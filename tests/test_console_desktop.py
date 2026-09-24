"""Desktop startup tests: ephemeral loopback services, no browser or model runs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

import experiment_console.desktop as desktop
from experiment_console.manager import Manager
from experiment_console.server import ConsoleHTTPServer


@pytest.fixture
def configured(tmp_path, monkeypatch):
    root = tmp_path / "isolated project ; $desktop"
    config = root / "config"
    config.mkdir(parents=True)
    environment = root / "Python environment ; $fixture"
    environment.mkdir()
    # These explicit placeholders validate paths only; Popen is always mocked
    # before a launcher function could execute one of them.
    (environment / "python.exe").write_bytes(b"Desktop test fixture, never executed")
    (environment / "pythonw.exe").write_bytes(b"Desktop test fixture, never executed")
    source = config / "desktop settings.json"
    values = {"python_executable": str(environment / "python.exe"),
              "state_root": "runs/desktop metadata ; $fixture", "port": 8765}
    source.write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setattr(desktop, "__file__", str(root / "experiment_console" / "desktop.py"))
    return SimpleNamespace(settings=desktop.DesktopSettings(source), values=values, path=source)


def bootstrap(settings):
    return {"project_root": str(settings.root), "state_root": str(settings.state),
            "python_executable": str(settings.python), "catalog": []}


def use_free_port(settings):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        settings.port = reservation.getsockname()[1]
    settings.url = f"http://127.0.0.1:{settings.port}"


@pytest.fixture
def loopback_services():
    services = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.server.requests.append(self.path)
            if self.server.delay:
                time.sleep(self.server.delay)
            body = self.server.body
            if body is None:
                body = json.dumps(self.server.payload).encode("utf-8")
            self.send_response(self.server.response_status)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "application/json")
            if self.server.redirect:
                self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/redirected")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # Expected when the client correctly rejects a delayed response.

    def start(settings, *, port=0):
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        server.daemon_threads = True
        server.payload = bootstrap(settings)
        server.requests = []
        server.body = None
        server.delay = 0
        server.response_status = 200
        server.redirect = False
        settings.port = server.server_port
        settings.url = f"http://127.0.0.1:{settings.port}"
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        thread.start()
        services.append((server, thread))
        return server

    yield start
    for server, thread in services:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()


def test_real_console_bootstrap_matches_project_state_and_configured_python(configured):
    settings = configured.settings
    settings.python = Path(sys.executable).resolve()
    manager = Manager(settings.root, settings.state, settings.python,
                      registry=SimpleNamespace(catalog=lambda: []), start_scheduler=False)
    server = ConsoleHTTPServer(("127.0.0.1", 0), manager)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    settings.port = server.server_port
    settings.url = f"http://127.0.0.1:{settings.port}"
    thread.start()
    try:
        assert desktop.backend_ready(settings) is True
        assert manager.list_runs() == []
    finally:
        server.shutdown()
        server.server_close()
        manager.close()
        thread.join(timeout=3)
    assert not thread.is_alive()


@pytest.mark.parametrize("field", ["project_root", "state_root", "python_executable", "catalog"])
def test_other_app_or_checkout_on_the_port_is_not_reused(configured, loopback_services, field):
    settings = configured.settings
    server = loopback_services(settings)
    server.payload[field] = {} if field == "catalog" else str(settings.root / "another-instance")
    with pytest.raises(desktop.DesktopError):
        desktop.backend_ready(settings)


@pytest.mark.parametrize("body", [b"<html>Another local app</html>", b"[]", b"null", b"x" * 1_000_001],
                         ids=["html", "array", "null", "oversized"])
def test_unexpected_or_oversized_bootstrap_is_rejected(configured, loopback_services, body):
    settings = configured.settings
    server = loopback_services(settings)
    server.body = body
    with pytest.raises(desktop.DesktopError):
        desktop.backend_ready(settings)


def test_local_redirect_is_rejected_without_following_it(configured, loopback_services):
    settings = configured.settings
    server = loopback_services(settings)
    server.response_status = 302
    server.redirect = True
    with pytest.raises(desktop.DesktopError):
        desktop.backend_ready(settings)
    assert server.requests == ["/api/bootstrap"]


def test_http_error_and_slow_service_are_distinct_from_absent_backend(configured, loopback_services):
    settings = configured.settings
    server = loopback_services(settings)
    server.response_status = 404
    with pytest.raises(desktop.DesktopError, match="occupé"):
        desktop.backend_ready(settings)
    server.response_status = 200
    server.delay = .2
    with pytest.raises(desktop.DesktopError, match="temps|répond"):
        desktop.backend_ready(settings, timeout=.03)


def test_connection_refused_means_backend_has_not_started(configured):
    settings = configured.settings
    use_free_port(settings)
    assert desktop.backend_ready(settings, timeout=.2) is False


def test_local_readiness_does_not_inherit_proxy_environment(configured, loopback_services, monkeypatch):
    settings = configured.settings
    loopback_services(settings)
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:1")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "")
    assert desktop.backend_ready(settings, timeout=1) is True


def test_existing_correct_backend_is_reused_without_child_spawn(configured, loopback_services, monkeypatch):
    settings = configured.settings
    loopback_services(settings)
    monkeypatch.setattr(desktop.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Existing backend must be reused"))
    assert desktop.ensure_backend(settings) is False


def test_preexisting_busy_backend_becomes_ready_without_spawning_competitor(configured, loopback_services, monkeypatch):
    settings = configured.settings
    server = loopback_services(settings)
    server.delay = .25
    real_probe = desktop.backend_ready
    transitions = []

    def probe_with_short_http_deadline(settings):
        try:
            result = real_probe(settings, timeout=.05)
        except desktop.BackendBusy:
            transitions.append("busy")
            server.delay = 0
            raise
        transitions.append(result)
        return result

    monkeypatch.setattr(desktop, "backend_ready", probe_with_short_http_deadline)
    monkeypatch.setattr(desktop.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Do not compete with a bound server"))
    assert desktop.ensure_backend(settings, startup_timeout=2) is False
    assert transitions[0] == "busy" and transitions[-1] is True
    assert len(server.requests) >= 2


def test_spawned_backend_survives_busy_socket_race_and_is_launched_only_once(configured, loopback_services, monkeypatch):
    settings = configured.settings
    use_free_port(settings)
    real_probe = desktop.backend_ready
    transitions = []
    servers = []
    spawns = []

    def fake_child_spawn(command, **options):
        spawns.append(command)
        server = loopback_services(settings, port=settings.port)
        server.delay = .25
        servers.append(server)
        return SimpleNamespace(poll=lambda: None)

    def probe_with_short_http_deadline(settings):
        try:
            result = real_probe(settings, timeout=.05)
        except desktop.BackendBusy:
            transitions.append("busy")
            assert servers, "this busy socket belongs to the just-started service"
            servers[0].delay = 0
            raise
        transitions.append(result)
        return result

    monkeypatch.setattr(desktop, "backend_ready", probe_with_short_http_deadline)
    monkeypatch.setattr(desktop.subprocess, "Popen", fake_child_spawn)
    assert desktop.ensure_backend(settings, startup_timeout=2) is True
    assert transitions[0] is False and "busy" in transitions and transitions[-1] is True
    assert len(spawns) == 1


def test_wrong_backend_identity_fails_immediately_without_busy_retry(configured, loopback_services, monkeypatch):
    settings = configured.settings
    server = loopback_services(settings)
    server.payload["state_root"] = str(settings.root / "different metadata")
    monkeypatch.setattr(desktop.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Wrong app must not trigger another server"))
    with pytest.raises(desktop.DesktopError, match="autre instance") as error:
        desktop.ensure_backend(settings, startup_timeout=1)
    assert not isinstance(error.value, desktop.BackendBusy)
    assert server.requests == ["/api/bootstrap"]


def test_concurrent_desktop_launches_start_exactly_one_server_with_structured_arguments(configured, loopback_services, monkeypatch):
    settings = configured.settings
    use_free_port(settings)
    spawns = []
    start_gate = threading.Barrier(6)

    def fake_child_spawn(command, **options):
        spawns.append((command, options))
        loopback_services(settings, port=settings.port)
        return SimpleNamespace(poll=lambda: None)

    def click(_):
        start_gate.wait(timeout=5)
        return desktop.ensure_backend(settings, startup_timeout=3)

    monkeypatch.setattr(desktop.subprocess, "Popen", fake_child_spawn)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(click, range(6)))
    assert results.count(True) == 1 and results.count(False) == 5
    assert len(spawns) == 1
    command, options = spawns[0]
    assert command == [str(settings.pythonw), "-m", "experiment_console.server", "--settings",
                       str(settings.path), "--port", str(settings.port)]
    assert isinstance(command, list) and options.get("shell", False) is False
    assert options["cwd"] == settings.root
    assert options["stdin"] == subprocess.DEVNULL
    assert options["env"]["PYTHONUTF8"] == "1"
    assert options["close_fds"] is True
    assert (settings.state / "desktop_startup.log").is_file()
    assert desktop.backend_ready(settings)


def test_server_startup_failure_reports_diagnostic_without_opening_browser(configured, monkeypatch):
    settings = configured.settings
    use_free_port(settings)
    calls = []

    def failed_child(command, **options):
        calls.append(command)
        return SimpleNamespace(poll=lambda: 7)

    monkeypatch.setattr(desktop.subprocess, "Popen", failed_child)
    with pytest.raises(desktop.DesktopError, match="desktop_startup.log"):
        desktop.ensure_backend(settings, startup_timeout=.5)
    assert len(calls) == 1
    assert calls[0][0] == str(settings.pythonw)


def test_startup_timeout_does_not_kill_an_uncertain_service(configured, monkeypatch):
    settings = configured.settings
    use_free_port(settings)
    process = SimpleNamespace(poll=lambda: None,
                              terminate=lambda: pytest.fail("An uncertain service must not be terminated"),
                              kill=lambda: pytest.fail("An uncertain service must not be killed"))
    monkeypatch.setattr(desktop.subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(desktop.DesktopError, match="plus de temps"):
        desktop.ensure_backend(settings, startup_timeout=.05)


def test_dedicated_window_arguments_preserve_spaces_and_metacharacters_without_shell(configured, monkeypatch):
    settings = configured.settings
    browser = settings.root / "Browser application ; $fixture" / "msedge.exe"
    calls = []
    fake_process = object()

    def capture(command, **options):
        calls.append((command, options))
        return fake_process

    monkeypatch.setattr(desktop.subprocess, "Popen", capture)
    assert desktop.open_window(settings, browser) is fake_process
    command, options = calls[0]
    assert command[0] == str(browser)
    assert f"--app={settings.url}/" in command
    assert f'--user-data-dir={settings.state / "desktop_browser"}' in command
    assert "--no-first-run" in command and "--no-default-browser-check" in command
    assert isinstance(command, list) and options.get("shell", False) is False
    assert options["cwd"] == settings.root
    assert options["stdin"] == options["stdout"] == options["stderr"] == subprocess.DEVNULL


@pytest.mark.parametrize("overrides", [
    {"port": True}, {"port": "8765"}, {"port": 80}, {"port": 65536},
    {"python_executable": "python.exe"}, {"state_root": "../outside-the-project"},
])
def test_invalid_desktop_configuration_is_rejected(configured, overrides):
    configured.path.write_text(json.dumps({**configured.values, **overrides}), encoding="utf-8")
    with pytest.raises(desktop.DesktopError):
        desktop.DesktopSettings(configured.path)


def test_missing_python_and_missing_pythonw_are_rejected(configured):
    settings = configured.settings
    settings.pythonw.unlink()
    with pytest.raises(desktop.DesktopError, match="Python"):
        desktop.DesktopSettings(configured.path)
    settings.python.unlink()
    with pytest.raises(desktop.DesktopError, match="Python"):
        desktop.DesktopSettings(configured.path)


def test_invalid_json_cannot_start_a_service(configured, monkeypatch):
    configured.path.write_text("{broken config", encoding="utf-8")
    monkeypatch.setattr(desktop.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Invalid config must not spawn"))
    with pytest.raises((desktop.DesktopError, json.JSONDecodeError)):
        desktop.main(configured.path)
