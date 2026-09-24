import json
from pathlib import Path
import shutil
import subprocess

import pandas as pd
import pytest

import run_nyx_scarcity_adjustments as app


@pytest.mark.parametrize('target', ['runs/exports/a', 'runs/live/a',
    'runs/experiments/nyx_scarcity_v1/variants/snapshots/a',
    'runs/experiments/nyx_scarcity_v1/variants/adjustments/../../snapshots/a'])
def test_outputs_cannot_overwrite_sources_or_production(tmp_path, target):
    with pytest.raises(ValueError):
        app.output_path(target, root=tmp_path)


def test_cli_rejects_source_override_on_frozen_report():
    assert app.main(['--action', 'report', '--source-suite', 'anything']) == 2
    assert app.main(['--action', 'run', '--run-directory', 'anything']) == 2


def test_launcher_from_another_directory(tmp_path):
    shell = shutil.which('powershell.exe') or shutil.which('pwsh')
    if not shell:
        pytest.skip('PowerShell unavailable')
    result = subprocess.run([shell, '-NoProfile', '-File', str(app.ROOT/'ScarcityAdjustments.ps1'),
        '-Action', 'Run', '-DryRun'], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'run_nyx_scarcity_adjustments.py' in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.fixture
def fake_analysis(tmp_path, monkeypatch):
    from nyx_scarcity import adjustment_analysis, adjustment_reporting
    protected = tmp_path/'Forecast.ps1'
    protected.write_text('untouched')
    rows = pd.DataFrame({'variant': ['xgb_unweighted_fixed'], 'proposal100': [200.]})
    audit = {'source_settings': {'correction_clip_eur_mwh': 400.}, 'source_suite': 'frozen'}
    monkeypatch.setattr(app, 'read_source', lambda *a, **kw: ({'frozen': rows.copy()}, audit.copy()))
    monkeypatch.setattr(app, '_code_seals', lambda: {'test': 'test'})
    monkeypatch.setattr(adjustment_analysis, 'build_adjustments', lambda *a, **kw: (rows.copy(), {'test': True}))
    def render(rows, summary, audit, path):
        path.write_text('<html>research only</html>')
        return path
    monkeypatch.setattr(adjustment_reporting, 'render_adjustments', render)
    return tmp_path, protected


def test_run_report_reuse_and_tamper_guard(fake_analysis, monkeypatch):
    from nyx_scarcity import adjustment_analysis
    root, protected = fake_analysis
    result = app.run(Path('unused-mocked-source'), root=root)
    assert result.is_file() and protected.read_text() == 'untouched'
    directory = result.parent
    checksum = app.base.digest(directory/'adjustments.parquet')
    monkeypatch.setattr(adjustment_analysis, 'build_adjustments', lambda *a, **kw: pytest.fail('Report recomputed proposals'))
    assert app.report(directory, root=root) == result
    assert app.base.digest(directory/'adjustments.parquet') == checksum
    (directory/'metrics.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        app.report(directory, root=root)


def test_changed_source_during_analysis_refuses_publication(fake_analysis, monkeypatch):
    root, _ = fake_analysis
    calls = []
    def source(*a, **kw):
        calls.append(1)
        return {}, {'source_settings': {'correction_clip_eur_mwh': 400.}, 'source_hash': len(calls)}
    monkeypatch.setattr(app, 'read_source', source)
    with pytest.raises(ValueError, match='Sources or protected'):
        app.run(Path('mocked-source'), root=root)
    assert not list((root/app.NAMESPACE).glob('snapshots/*'))


def test_incomplete_source_comparison_refused(tmp_path, monkeypatch):
    source = tmp_path/'source'
    source.mkdir()
    for name, content in [('manifest.json', {}), ('comparison.json', {}),
                          ('comparison_manifest.json', {'status': 'failed'})]:
        (source/name).write_text(json.dumps(content))
    monkeypatch.setattr(app.variants, 'read_suite', lambda *a, **kw: (source, {}, {}))
    with pytest.raises(ValueError, match='completed and checksum'):
        app.read_source(source, root=tmp_path)
