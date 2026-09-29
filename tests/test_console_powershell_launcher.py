"""Run the real PowerShell entry point against an explicit argv-only Python fixture."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def launcher_project(tmp_path):
    root = tmp_path / 'console fixture with spaces'
    root.mkdir()
    script = root / 'Start-ExperimentConsole.ps1'
    script.write_bytes((Path(__file__).resolve().parents[1] / script.name).read_bytes())
    package = root / 'experiment_console'
    package.mkdir()
    (package / '__init__.py').write_text('', encoding='utf-8')
    (package / 'launcher.py').write_text(
        'import json, sys\nfrom pathlib import Path\n'
        "Path('captured.json').write_text(json.dumps(sys.argv[1:]), encoding='utf-8')\n",
        encoding='utf-8')
    config = root / 'config'
    config.mkdir()
    settings = config / 'experiment_console.json'
    settings.write_text(json.dumps({'python_executable': sys.executable, 'port': 8767}), encoding='utf-8')
    return root, script, settings, package


def invoke(script, *options):
    powershell = Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    if not powershell.is_file():
        pytest.skip('Windows PowerShell integration test')
    return subprocess.run([str(powershell), '-NoProfile', '-NonInteractive', '-File', str(script), *options],
                          cwd=script.parent, capture_output=True, text=True, timeout=20,
                          creationflags=subprocess.CREATE_NO_WINDOW)


@pytest.mark.parametrize('options,expected', [
    ([], ['--port', '8767', '--open']),
    (['-NoOpen'], ['--port', '8767']),
    (['-Restart'], ['--port', '8767', '--open', '--restart']),
    (['-Restart', '-NoOpen', '-Port', '8769'], ['--port', '8769', '--restart']),
])
def test_real_powershell_passes_restart_open_and_port_to_launcher(launcher_project, options, expected):
    root, script, settings, _ = launcher_project
    result = invoke(script, *options)
    assert result.returncode == 0, result.stdout + result.stderr
    captured = json.loads((root / 'captured.json').read_text(encoding='utf-8'))
    assert captured == ['--settings', str(settings), *expected]


def test_real_powershell_preserves_settings_path_and_reports_failed_restart(launcher_project):
    root, script, settings, package = launcher_project
    custom = settings.with_name('alternate settings.json')
    settings.rename(custom)
    with (package / 'launcher.py').open('a', encoding='utf-8') as stream:
        stream.write('raise SystemExit(7)\n')
    result = invoke(script, '-Settings', str(custom), '-Restart', '-NoOpen')
    assert result.returncode != 0
    assert 'code 7' in result.stdout + result.stderr
    assert json.loads((root / 'captured.json').read_text(encoding='utf-8')) == [
        '--settings', str(custom), '--port', '8767', '--restart']
