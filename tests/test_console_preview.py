"""Preview launcher keeps the normal NYX desktop settings untouched."""
from __future__ import annotations

import json

from experiment_console import desktop
from experiment_console.preview import PREVIEW_PORT, PREVIEW_STATE_ROOT, preview_settings_path


def test_preview_settings_reuse_python_and_isolate_state(tmp_path, monkeypatch):
    root = tmp_path / 'separate branch checkout'
    config = root / 'config'
    config.mkdir(parents=True)
    environment = tmp_path / 'configured Python'
    environment.mkdir()
    python = environment / 'python.exe'
    python.write_bytes(b'fixture')
    python.with_name('pythonw.exe').write_bytes(b'fixture')
    original = {'python_executable': str(python), 'state_root': 'runs/.experiment_console',
                'max_concurrency': 1, 'port': 8765}
    source = config / 'experiment_console.json'
    source.write_text(json.dumps(original), encoding='utf-8')
    original_bytes = source.read_bytes()

    path = preview_settings_path(root)
    generated = json.loads(path.read_text(encoding='utf-8'))
    assert path == root / PREVIEW_STATE_ROOT / 'desktop_settings.json'
    assert generated == {**original, 'state_root': PREVIEW_STATE_ROOT.as_posix(),
                         'port': PREVIEW_PORT}
    assert source.read_bytes() == original_bytes

    monkeypatch.setattr(desktop, '__file__', str(root / 'experiment_console' / 'desktop.py'))
    settings = desktop.DesktopSettings(path)
    assert settings.python == python.resolve()
    assert settings.state == (root / PREVIEW_STATE_ROOT).resolve()
    assert settings.port == 8766
    assert settings.state != (root / original['state_root']).resolve()


def test_preview_refreshes_configured_python_without_changing_normal_settings(tmp_path):
    root = tmp_path / 'checkout'
    config = root / 'config'
    config.mkdir(parents=True)
    source = config / 'experiment_console.json'
    source.write_text(json.dumps({'python_executable': 'C:/Python/first/python.exe',
                                  'state_root': 'runs/.experiment_console', 'port': 8765}), encoding='utf-8')
    path = preview_settings_path(root)
    source.write_text(json.dumps({'python_executable': 'C:/Python/second/python.exe',
                                  'state_root': 'runs/.experiment_console', 'port': 8765}), encoding='utf-8')
    assert preview_settings_path(root) == path
    generated = json.loads(path.read_text(encoding='utf-8'))
    assert generated['python_executable'] == 'C:/Python/second/python.exe'
    assert generated['port'] == 8766
    assert json.loads(source.read_text(encoding='utf-8'))['port'] == 8765
