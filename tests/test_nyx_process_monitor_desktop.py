import json
from pathlib import Path

import pytest

from nyx_process_monitor import desktop


class Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit):
        return json.dumps(self.body).encode()[:limit]


def test_reuse_only_own_application_and_project(monkeypatch):
    class Opener:
        def open(self, url, timeout):
            assert url.endswith('/api/health')
            return Response({'app': 'nyx-process-monitor', 'project_root': str(desktop.ROOT)})
    monkeypatch.setattr(desktop, 'build_opener', lambda *_args: Opener())
    assert desktop.ready()


@pytest.mark.parametrize('payload', [
    {'app': 'unrelated', 'project_root': 'irrelevant'},
    {'app': 'nyx-process-monitor', 'project_root': 'different-project'},
])
def test_refuse_other_local_application(monkeypatch, payload):
    class Opener:
        def open(self, *_args, **_kwargs):
            return Response(payload)
    monkeypatch.setattr(desktop, 'build_opener', lambda *_args: Opener())
    with pytest.raises(RuntimeError, match='autre programme'):
        desktop.ready()


def test_opening_existing_monitor_never_starts_second_service(monkeypatch, tmp_path):
    monkeypatch.setattr(desktop, 'STATE', tmp_path)
    monkeypatch.setattr(desktop, 'ready', lambda: True)
    monkeypatch.setattr(desktop.subprocess, 'Popen', lambda *_args, **_kwargs: pytest.fail('No second service'))
    desktop.start()


def test_no_redirects():
    with pytest.raises(RuntimeError, match='redirection refusée'):
        desktop.NoRedirect().redirect_request(None)
