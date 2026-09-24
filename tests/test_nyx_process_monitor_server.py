import json
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest

from nyx_process_monitor.server import make_server, SnapshotCache, BUILD


class FakeCollector:
    def __init__(self, artifact=None):
        self.path = artifact

    def snapshot(self):
        return {'app': 'nyx-process-monitor', 'jobs': [], 'warnings': []}

    def artifact(self, identity):
        return self.path if identity == 'allowed' else None


@pytest.fixture
def local_server(tmp_path):
    report = tmp_path / 'report.html'
    report.write_text('<h1>test</h1>', encoding='utf-8')
    server = make_server(tmp_path, 0, FakeCollector(report))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}'
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_health_and_state(local_server):
    with urlopen(local_server + '/api/health') as response:
        health = json.load(response)
        assert health['read_only'] is True
        assert health['build'] == BUILD
    with urlopen(local_server + '/api/state') as response:
        assert json.load(response)['jobs'] == []
        assert response.headers['Cache-Control'] == 'no-store'
        assert response.headers['X-Content-Type-Options'] == 'nosniff'


@pytest.mark.parametrize('headers', [{'Host': 'evil.test'}, {'Origin': 'https://evil.test'}, {'Sec-Fetch-Site': 'cross-site'}])
def test_reject_nonlocal_browser_context(local_server, headers):
    with pytest.raises(HTTPError) as error:
        urlopen(Request(local_server + '/api/state', headers=headers))
    assert error.value.code == 403


@pytest.mark.parametrize('method', ['POST', 'PUT', 'DELETE', 'PATCH'])
def test_no_mutation_endpoints(local_server, method):
    with pytest.raises(HTTPError) as error:
        urlopen(Request(local_server + '/api/state', data=b'{}', method=method))
    assert error.value.code == 405


def test_opaque_artifact_is_sandboxed(local_server):
    with urlopen(local_server + '/artifact?id=allowed') as response:
        assert response.read() == b'<h1>test</h1>'
        assert 'sandbox allow-scripts;' in response.headers['Content-Security-Policy']
        assert 'allow-same-origin' not in response.headers['Content-Security-Policy']
    with pytest.raises(HTTPError) as error:
        urlopen(local_server + '/artifact?id=../../config/secret')
    assert error.value.code == 404


def test_no_arbitrary_static_path(local_server):
    with pytest.raises(HTTPError) as error:
        urlopen(local_server + '/../config/experiment_console.json')
    assert error.value.code == 404


def test_sampler_keeps_last_snapshot_readable_during_next_collection():
    collecting = threading.Event()
    release = threading.Event()

    class SlowCollector:
        def __init__(self):
            self.calls = 0

        def snapshot(self):
            self.calls += 1
            if self.calls > 1:
                collecting.set()
                release.wait(timeout=5)
            return {'jobs': [], 'updated_at': 'fixed', 'sample': self.calls}

    collector = SlowCollector()
    cache = SnapshotCache(collector, interval=.02)
    cache.start()
    try:
        first = cache.snapshot()
        assert first['sample'] == 1
        assert collecting.wait(timeout=3)
        # A running collection does not lock HTTP readers or rewrite timestamps.
        assert cache.snapshot() is first
        assert cache.snapshot()['updated_at'] == 'fixed'
    finally:
        release.set()
        cache.close()
    assert not cache.thread.is_alive()


def test_sampler_failure_retains_old_timestamp_and_recovers():
    failed = threading.Event()
    recover = threading.Event()

    class RecoveringCollector:
        def __init__(self):
            self.calls = 0

        def snapshot(self):
            self.calls += 1
            if self.calls > 1 and not recover.is_set():
                failed.set()
                raise OSError('Synthetic transient collection failure')
            return {'jobs': [], 'updated_at': str(self.calls)}

    cache = SnapshotCache(RecoveringCollector(), interval=.03)
    cache.start()
    try:
        first = cache.snapshot()
        assert failed.wait(timeout=3)
        assert cache.snapshot() is first
        recover.set()
        # Use an event-bounded condition, not a multi-second blind delay.
        for _ in range(100):
            if cache.snapshot()['updated_at'] != first['updated_at']:
                break
            threading.Event().wait(.02)
        assert cache.snapshot()['updated_at'] != first['updated_at']
    finally:
        cache.close()
