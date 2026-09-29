"""Launch/restart tests with tiny isolated servers, never scientific commands."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

from filelock import FileLock
import psutil
import pytest

from experiment_console import desktop, launcher


@pytest.fixture
def settings(tmp_path):
    root = tmp_path / 'project'
    state = root / 'state'
    state.mkdir(parents=True)
    path = root / 'settings.json'
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    python = Path(sys.executable).resolve()
    path.write_text(json.dumps({'python_executable': str(python), 'state_root': str(state),
                                'port': port, 'max_concurrency': 1}), encoding='utf-8')
    return SimpleNamespace(root=root.resolve(), state=state.resolve(), path=path.resolve(),
                           python=python, pythonw=python.with_name('pythonw.exe'), port=port,
                           url=f'http://127.0.0.1:{port}')


TINY_SERVER = r'''
import argparse, json, sqlite3, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from filelock import FileLock
parser = argparse.ArgumentParser()
parser.add_argument('--settings', required=True)
parser.add_argument('--port', type=int, required=True)
args = parser.parse_args()
settings = json.loads(Path(args.settings).read_text(encoding='utf-8'))
root = Path.cwd().resolve()
state = Path(settings['state_root']).resolve()
state.mkdir(parents=True, exist_ok=True)
lock = FileLock(str(state / 'backend.lock'))
lock.acquire(timeout=0)
database = state / 'console.sqlite3'
with sqlite3.connect(database) as db:
    db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, source_key TEXT UNIQUE, data TEXT NOT NULL)')
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def do_GET(self):
        if self.path == '/api/bootstrap':
            value = {'project_root': str(root), 'state_root': str(state),
                     'python_executable': settings['python_executable'], 'catalog': [],
                     'token': 'fixture-token'}
        elif self.path == '/api/runs':
            with sqlite3.connect(database) as db:
                rows = [json.loads(row[0]) for row in db.execute('SELECT data FROM runs')]
            value = {'runs': rows, 'import': {'running': False}}
        elif self.path == '/api/import-status':
            value = {'running': False}
        else:
            self.send_error(404)
            return
        body = json.dumps(value).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
try:
    server.serve_forever(poll_interval=.02)
finally:
    server.server_close()
    lock.release()
'''


@pytest.fixture
def tiny_server():
    owned = []
    def start(settings, *, actual_launcher=False, restart=False):
        if not actual_launcher:
            package = settings.root / 'experiment_console'
            package.mkdir(exist_ok=True)
            (package / '__init__.py').write_text('', encoding='utf-8')
            (package / 'server.py').write_text(TINY_SERVER, encoding='utf-8')
        module = 'experiment_console.launcher' if actual_launcher else 'experiment_console.server'
        command = [str(settings.python), '-m', module,
                   '--settings', str(settings.path), '--port', str(settings.port)]
        if restart:
            command.append('--restart')
        log_path = settings.state / ('restart.log' if restart else 'startup.log')
        with log_path.open('wb') as output:
            process = subprocess.Popen(command,
                                       cwd=settings.root, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=output,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        identities = []
        owned.append((process, identities))
        deadline = time.monotonic() + (40 if actual_launcher else 10)
        while time.monotonic() < deadline:
            assert process.poll() is None, log_path.read_text(errors='replace')
            try:
                ready = desktop.backend_ready(settings, timeout=2 if actual_launcher else .2)
            except desktop.BackendBusy:
                ready = False
            if ready:
                parent = psutil.Process(process.pid)
                current = [(child.pid, child.create_time()) for child in [parent, *parent.children(recursive=True)]]
                try:
                    listener, _ = launcher.verify_backend_process(settings)
                except desktop.DesktopError:
                    if restart:
                        time.sleep(.03)
                        continue
                    raise
                if listener.pid in {pid for pid, _created in current}:
                    identities.extend(current)
                    return SimpleNamespace(process=process, identities=identities)
            time.sleep(.03)
        pytest.fail('Fixture server did not become ready: ' + log_path.read_text(errors='replace'))
    yield start
    for process, identities in owned:
        if not identities and process.poll() is None:
            try:
                parent = psutil.Process(process.pid)
                identities.extend((child.pid, child.create_time()) for child in [parent, *parent.children(recursive=True)])
            except psutil.NoSuchProcess:
                pass
        for pid, created in reversed(identities):
            try:
                child = psutil.Process(pid)
                if abs(child.create_time() - created) < .05:
                    child.terminate()
                    child.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()  # Only the explicitly spawned tiny fixture wrapper.
            process.wait(timeout=5)


def insert_run(settings, status, source='managed'):
    record = {'id': 'fixture-run', 'source': source, 'status': status,
              'adapter_id': 'primary_nuclear_kalman'}
    with sqlite3.connect(settings.state / 'console.sqlite3') as db:
        db.execute('INSERT OR REPLACE INTO runs VALUES (?,?,?)',
                   ('fixture-run', 'fixture-run', json.dumps(record)))


def assert_server_alive(settings):
    assert desktop.backend_ready(settings, timeout=.5) is True


def test_existing_backend_is_reused_repeatedly_without_model_or_child(settings, tiny_server, monkeypatch):
    tiny_server(settings)
    monkeypatch.setattr(launcher, 'restart_backend', lambda *a, **k: pytest.fail('Normal opening cannot restart'))
    monkeypatch.setattr(desktop, 'start_backend', lambda *a, **k: pytest.fail('Existing service cannot spawn'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        reused = list(pool.map(lambda _: launcher.prepare_launch(settings), range(4)))
    assert reused == [True] * 4
    with sqlite3.connect(settings.state / 'console.sqlite3') as db:
        assert db.execute('SELECT count(*) FROM runs').fetchone()[0] == 0
    assert_server_alive(settings)


def test_actual_foreground_server_reuses_then_restarts_with_empty_queue(settings, tiny_server, monkeypatch):
    settings.root = Path(launcher.__file__).resolve().parents[1]
    first = tiny_server(settings, actual_launcher=True)
    original_token = launcher.read_json(settings, '/api/bootstrap')['token']
    monkeypatch.setattr(launcher, 'DesktopSettings', lambda *a, **k: settings)
    arguments = ['--settings', str(settings.path), '--port', str(settings.port)]
    assert launcher.main(arguments) == 0
    assert first.process.poll() is None
    second = tiny_server(settings, actual_launcher=True, restart=True)
    first.process.wait(timeout=5)
    assert second.process.poll() is None
    assert launcher.read_json(settings, '/api/bootstrap')['token'] != original_token
    assert launcher.read_json(settings, '/api/runs')['runs'] == []


def test_backend_lock_during_cold_start_waits_for_same_service(settings, monkeypatch):
    calls = []
    monkeypatch.setattr(launcher, 'backend_ready', lambda current: bool(calls))
    def locked(current):
        calls.append(True)
        raise launcher.BackendLocked('Explicit cold-start fixture')
    monkeypatch.setattr(launcher, '_assert_backend_lock_free', locked)
    assert launcher.prepare_launch(settings, timeout=1) is True


def test_simultaneous_foreground_launch_recovers_backend_lock_race(settings, monkeypatch):
    import threading
    from experiment_console import server
    from experiment_console.manager import BackendAlreadyRunning
    rendezvous = threading.Barrier(2)
    ready = threading.Event()
    calls = []
    mutex = threading.Lock()
    def serve(argv=None):
        with mutex:
            index = len(calls)
            calls.append(argv)
        rendezvous.wait(timeout=5)
        ready.set()
        if index:
            raise BackendAlreadyRunning('Explicit backend.lock race fixture')
    monkeypatch.setattr(launcher, 'DesktopSettings', lambda *a, **k: settings)
    monkeypatch.setattr(launcher, 'backend_ready', lambda current: ready.is_set())
    monkeypatch.setattr(server, 'main', serve)
    arguments = ['--settings', str(settings.path), '--port', str(settings.port)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(launcher.main, arguments) for _ in range(2)]
        assert [future.result(timeout=10) for future in futures] == [0, 0]
    assert len(calls) == 2


def test_legacy_state_locked_on_other_port_never_starts_competitor(settings, monkeypatch):
    lock = FileLock(str(settings.state / 'backend.lock'))
    monkeypatch.setattr(desktop, 'start_backend', lambda *a, **k: pytest.fail('Do not launch with occupied state'))
    with lock.acquire(timeout=0):
        with pytest.raises(desktop.DesktopError):
            launcher.prepare_launch(settings, timeout=.08)
        assert lock.is_locked


def test_stale_lock_file_without_owner_does_not_block_new_server(settings):
    (settings.state / 'backend.lock').write_text('old fixture marker', encoding='utf-8')
    assert launcher.prepare_launch(settings, timeout=.2) is False
    with FileLock(str(settings.state / 'backend.lock')).acquire(timeout=0):
        pass


def test_busy_socket_is_retried_without_terminating_or_spawning(settings, monkeypatch):
    probes = iter([desktop.BackendBusy('warming up'), True])
    def probe(_settings):
        result = next(probes)
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(launcher, 'backend_ready', probe)
    monkeypatch.setattr(desktop, 'start_backend', lambda *a, **k: pytest.fail('Busy service must not compete'))
    monkeypatch.setattr(launcher, 'restart_backend', lambda *a, **k: pytest.fail('Busy is not permission to restart'))
    assert launcher.prepare_launch(settings, timeout=1) is True


@pytest.mark.parametrize('status', ['queued', 'starting', 'running', 'cancelling', 'unknown'])
def test_restart_refuses_non_idle_managed_run(settings, tiny_server, status):
    tiny_server(settings)
    insert_run(settings, status)
    with pytest.raises(desktop.DesktopError):
        launcher.restart_backend(settings)
    assert_server_alive(settings)
    with sqlite3.connect(settings.state / 'console.sqlite3') as db:
        assert json.loads(db.execute('SELECT data FROM runs').fetchone()[0])['status'] == status


def test_restart_refuses_locked_database_without_terminating(settings, tiny_server):
    tiny_server(settings)
    holder = sqlite3.connect(settings.state / 'console.sqlite3', timeout=0)
    holder.execute('BEGIN IMMEDIATE')
    try:
        with pytest.raises(desktop.DesktopError):
            launcher.restart_backend(settings)
        assert_server_alive(settings)
    finally:
        holder.rollback()
        holder.close()


def test_restart_rechecks_database_after_new_run_races_http_snapshot(settings, tiny_server, monkeypatch):
    tiny_server(settings)
    original = launcher.verify_backend_process
    inserted = []
    def identity_then_queue(current):
        result = original(current)
        if not inserted:
            insert_run(settings, 'queued')
            inserted.append(True)
        return result
    monkeypatch.setattr(launcher, 'verify_backend_process', identity_then_queue)
    with pytest.raises(desktop.DesktopError):
        launcher.restart_backend(settings)
    assert inserted
    assert_server_alive(settings)


def test_idle_restart_of_real_legacy_venv_backend_holds_writer_lock_until_exit(settings, tiny_server, monkeypatch):
    fixture = tiny_server(settings)
    insert_run(settings, 'succeeded')
    identity, _ = launcher.verify_backend_process(settings)
    assert identity.pid in {pid for pid, _created in fixture.identities}
    original = launcher.verify_backend_process
    checks = []
    def assert_write_blocked(phase):
        challenger = sqlite3.connect(settings.state / 'console.sqlite3', timeout=0)
        try:
            with pytest.raises(sqlite3.OperationalError, match='locked'):
                challenger.execute('BEGIN IMMEDIATE')
            checks.append(phase)
        finally:
            challenger.close()
    class ProcessProxy:
        def __init__(self, process):
            self.process = process
        def __getattr__(self, name):
            return getattr(self.process, name)
        def terminate(self):
            assert_write_blocked('terminate')
            self.process.terminate()
        def wait(self, timeout=None):
            assert_write_blocked('wait')
            return self.process.wait(timeout=timeout)
    def verified(current):
        process, created = original(current)
        return ProcessProxy(process), created
    monkeypatch.setattr(launcher, 'verify_backend_process', verified)
    launcher.restart_backend(settings)
    assert 'terminate' in checks and 'wait' in checks
    assert desktop.backend_ready(settings, timeout=.5) is False
    with sqlite3.connect(settings.state / 'console.sqlite3', timeout=0) as db:
        db.execute('BEGIN IMMEDIATE')
        assert json.loads(db.execute('SELECT data FROM runs').fetchone()[0])['status'] == 'succeeded'
    with FileLock(str(settings.state / 'backend.lock')).acquire(timeout=0):
        pass


@pytest.mark.parametrize('field', ['root', 'path', 'python', 'port'])
def test_process_identity_mismatch_refuses_legacy_restart(settings, tiny_server, field):
    tiny_server(settings)
    mismatched = SimpleNamespace(**vars(settings))
    if field == 'port':
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            mismatched.port = reservation.getsockname()[1]
            mismatched.url = f'http://127.0.0.1:{mismatched.port}'
            with pytest.raises(desktop.DesktopError):
                launcher.verify_backend_process(mismatched)
    else:
        if field == 'root':
            mismatched.root = settings.root / 'different-checkout'
        elif field == 'path':
            mismatched.path = settings.root / 'different-settings.json'
            mismatched.path.write_bytes(settings.path.read_bytes())
        else:
            mismatched.python = settings.root / 'unrelated-python.exe'
            mismatched.python.write_bytes(b'Non-executable identity test fixture')
            mismatched.pythonw = settings.root / 'unrelated-pythonw.exe'
        with pytest.raises(desktop.DesktopError):
            launcher.verify_backend_process(mismatched)
    assert_server_alive(settings)


@pytest.mark.parametrize('open_browser', [False, True])
def test_main_reuses_exact_service_and_only_opens_browser_when_requested(settings, tiny_server, monkeypatch, open_browser):
    from experiment_console import server
    tiny_server(settings)
    opened = []
    monkeypatch.setattr(launcher, 'DesktopSettings', lambda *args, **kwargs: settings)
    monkeypatch.setattr(launcher.webbrowser, 'open', lambda url: opened.append(url))
    monkeypatch.setattr(server, 'main', lambda *args, **kwargs: pytest.fail('A reused server cannot be started again'))
    arguments = ['--settings', str(settings.path), '--port', str(settings.port)]
    if open_browser:
        arguments.append('--open')
    launcher.main(arguments)
    assert len(opened) == int(open_browser)
    if opened:
        assert opened[0].rstrip('/') == settings.url
    with sqlite3.connect(settings.state / 'console.sqlite3') as db:
        assert db.execute('SELECT count(*) FROM runs').fetchone()[0] == 0
    assert_server_alive(settings)
