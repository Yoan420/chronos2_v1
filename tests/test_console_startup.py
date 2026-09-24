"""Desktop-safe startup tests; no scientific command is executed."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import sys

from filelock import FileLock
import pytest

from experiment_console.manager import Manager
import experiment_console.manager as manager_module
import experiment_console.server as server_module


def settings_file(tmp_path):
    state = tmp_path / 'isolated desktop state'
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'python_executable': str(Path(sys.executable).resolve()),
                                'state_root': str(state), 'max_concurrency': 1, 'port': 0}), encoding='utf-8')
    return path, state


def assert_lock_available(state):
    lock = FileLock(str(state / 'backend.lock'))
    with lock.acquire(timeout=0):
        assert lock.is_locked


def test_manager_start_is_idempotent_across_concurrent_callers(tmp_path):
    manager = Manager(tmp_path, tmp_path / 'state', Path(sys.executable).resolve(),
                      registry=object(), start_scheduler=False)
    try:
        assert manager._thread is None
        with ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(lambda _: manager.start(), range(10)))
        first = manager._thread
        assert first is not None and first.is_alive()
        manager.start()
        assert manager._thread is first
    finally:
        manager.close()
    assert not first.is_alive()
    with pytest.raises(ValueError, match='fermée'):
        manager.start()
    assert_lock_available(manager.state_root)


def test_manager_initialization_error_releases_backend_lock(tmp_path, monkeypatch):
    def failed_store(_path):
        raise OSError('Explicit store-initialization test failure')
    monkeypatch.setattr(manager_module, 'Store', failed_store)
    state = tmp_path / 'state'
    with pytest.raises(OSError, match='initialization test failure'):
        Manager(tmp_path, state, Path(sys.executable).resolve(), registry=object())
    assert_lock_available(state)


def test_port_conflict_never_starts_scheduler_and_releases_lock(tmp_path, monkeypatch):
    path, state = settings_file(tmp_path)
    starts = []
    monkeypatch.setattr(Manager, 'start', lambda manager: starts.append(manager))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            occupied.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        occupied.bind(('127.0.0.1', 0))
        occupied.listen(1)
        monkeypatch.setattr(sys, 'argv', ['console', '--settings', str(path), '--port', str(occupied.getsockname()[1])])
        with pytest.raises(OSError):
            server_module.main()
    assert not starts
    assert_lock_available(state)


@pytest.mark.parametrize('raise_in_service', [False, True])
def test_bound_service_starts_without_stdout_and_cleans_up_on_exit(tmp_path, monkeypatch, raise_in_service):
    path, state = settings_file(tmp_path)
    instances = []
    actual_server = server_module.ConsoleHTTPServer

    class BoundedFixtureServer(actual_server):
        def __init__(self, address, manager):
            assert manager._thread is None
            super().__init__(address, manager)
            instances.append(self)

        def serve_forever(self, poll_interval):
            assert self.socket.fileno() != -1
            assert self.manager._thread is not None and self.manager._thread.is_alive()
            if raise_in_service:
                raise RuntimeError('Explicit service fixture failure')

    monkeypatch.setattr(server_module, 'ConsoleHTTPServer', BoundedFixtureServer)
    monkeypatch.setattr(sys, 'argv', ['console', '--settings', str(path)])
    with monkeypatch.context() as stdio:
        stdio.setattr(sys, 'stdout', None)
        if raise_in_service:
            with pytest.raises(RuntimeError, match='service fixture failure'):
                server_module.main()
        else:
            server_module.main()
    assert len(instances) == 1
    server = instances[0]
    assert server.socket.fileno() == -1
    assert not server.manager._thread.is_alive()
    assert_lock_available(state)
