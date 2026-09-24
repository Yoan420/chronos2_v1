"""Less observation work without caching path trust or experiment completion."""
import json
from pathlib import Path
from types import SimpleNamespace

from nyx_process_monitor.collector import Collector


def test_launch_discovery_once_per_sample(tmp_path, monkeypatch):
    collector = Collector(tmp_path)
    calls = []
    monkeypatch.setattr(collector, '_launches', lambda: calls.append(True) or [])
    original = collector._processes
    monkeypatch.setattr('nyx_process_monitor.collector.psutil.process_iter', lambda *a, **kw: [])
    monkeypatch.setattr('nyx_process_monitor.collector.psutil._ppid_map', lambda: {})
    collector.snapshot()
    assert len(calls) == 1
    assert collector._sample_launches is None
    original()
    assert len(calls) == 2


def test_tail_reuses_only_unchanged_content_and_rechecks_paths(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    runs.mkdir()
    log = runs / 'sample.log'
    log.write_text('password=first-secret\n', encoding='utf-8')
    collector = Collector(tmp_path)
    original_open = Path.open
    opens = []
    def counted(path, *args, **kwargs):
        if path == log and args and args[0] == 'rb':
            opens.append(True)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', counted)
    first = collector._tail(log)
    assert collector._tail(log) == first
    assert len(opens) == 1
    assert 'first-secret' not in first[0]
    log.write_text('new line\npassword=second-secret\n', encoding='utf-8')
    second = collector._tail(log)
    assert len(opens) == 2 and 'new line' in second[0]
    assert 'second-secret' not in second[0]
    log.unlink()
    assert collector._tail(log) == ('', None)


def test_cached_log_rejects_later_redirect_without_host_privilege(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    runs.mkdir()
    log = runs / 'sample.log'
    log.write_text('okay', encoding='utf-8')
    collector = Collector(tmp_path)
    collector._tail(log)
    original = Path.lstat
    def redirected(path):
        if path == runs:
            return SimpleNamespace(st_mode=0o40755, st_file_attributes=0x400)
        return original(path)
    monkeypatch.setattr(Path, 'lstat', redirected)
    assert collector._safe(log) is None
    assert collector._tail(log) == ('', None)


def test_safe_checks_components_each_time_and_resolves_once(tmp_path, monkeypatch):
    path = tmp_path / 'runs/a/b/c/file.json'
    path.parent.mkdir(parents=True)
    path.write_text('{}', encoding='utf-8')
    collector = Collector(tmp_path)
    original = Path.resolve
    calls = []
    def counted(value, *args, **kwargs):
        calls.append(value)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(Path, 'resolve', counted)
    assert collector._safe(path) == path
    assert calls == [path]
    assert collector._safe(path) == path
    assert calls == [path, path]


def test_json_state_never_stale_from_content_cache(tmp_path):
    path = tmp_path / 'runs/status.json'
    path.parent.mkdir()
    collector = Collector(tmp_path)
    path.write_text(json.dumps({'status': 'RUNNING'}), encoding='utf-8')
    assert collector._json(path)['status'] == 'RUNNING'
    path.write_text(json.dumps({'status': 'COMPLETE'}), encoding='utf-8')
    assert collector._json(path)['status'] == 'COMPLETE'
    path.unlink()
    assert collector._json(path) is None


def test_absent_launch_does_not_read_its_historical_logs(tmp_path, monkeypatch):
    collector = Collector(tmp_path)
    meta = {'python_pid': 1, 'python_created_utc': '2026-09-22T12:00:00Z',
            'stdout': str(tmp_path / 'runs/old.stdout.log')}
    monkeypatch.setattr(collector, '_tail', lambda _: (_ for _ in ()).throw(AssertionError('inactive log read')))
    path = tmp_path / 'runs/experiments/old/launcher_logs/old.launch.json'
    job, processes = collector._registered(path, meta, {}, {}, 1)
    assert not processes
    assert job['stdout_tail'] == job['stderr_tail'] == ''
