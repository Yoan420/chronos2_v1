from pathlib import Path
from types import SimpleNamespace
import json

import pandas as pd
import pytest
import yaml

import run_nuclear_cwe_forecast as runner


@pytest.mark.parametrize('value', ['../2026-09-11', '2026-09-11T08:00', '2026-02-30', '2026-9-11', None])
def test_delivery_strict(value):
    with pytest.raises((ValueError, TypeError)):
        runner._day(value)


def test_day_valid():
    assert runner._day('2026-09-11') == '2026-09-11'


def test_paths_must_remain_in_isolated_tree(tmp_path):
    assert runner._inside(tmp_path / 'new/file', tmp_path) == (tmp_path / 'new/file').resolve()
    with pytest.raises(ValueError):
        runner._inside(tmp_path, tmp_path)
    with pytest.raises(ValueError):
        runner._inside(tmp_path / '../escape', tmp_path)


def test_copy_is_immutable_and_checks_source(tmp_path):
    source, destination = tmp_path / 'source', tmp_path / 'new/file'
    source.write_bytes(b'original')
    record = runner._copy_checked(source, destination)
    assert source.read_bytes() == destination.read_bytes() == b'original'
    assert runner._copy_checked(source, destination) == record
    source.write_bytes(b'revised')
    with pytest.raises(ValueError, match='SHA source'):
        runner._copy_checked(source, destination, expected=record['sha256'])
    with pytest.raises(ValueError, match='deja presente'):
        runner._copy_checked(source, destination)
    assert destination.read_bytes() == b'original'


def test_config_rejects_operational_output(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    config = tmp_path / 'config/cwe.yaml'
    config.parent.mkdir()
    config.write_text(yaml.safe_dump({'schema_version': 1, 'project_root': '..', 'delivery_day': '2026-09-11',
                                     'output_root': 'runs/exports'}))
    with pytest.raises(ValueError, match='perimetre'):
        runner.load_settings(config)


def test_freeze_reporting_relocates_only_paths_and_can_resume(tmp_path):
    baseline, workspace = tmp_path / 'baseline', tmp_path / 'candidate'
    original = baseline / 'reporting/snapshot'
    (original / 'inputs').mkdir(parents=True)
    observed = original / 'inputs/observed_latest.parquet'
    observed.write_bytes(b'frozen observations')
    storm = original / 'inputs/storm_dashboard_official_statistics.parquet'
    storm.write_bytes(b'frozen Storm')
    audit = {'status': 'complete', 'snapshot_directory': str(original),
             'extracted_at_utc': '2026-09-10T12:06:03Z', 'storm_dashboard': {'status': 'complete'},
             'observed': {'artifact_path': str(observed), 'relative_artifact_path': 'inputs/observed_latest.parquet',
                          'artifact_sha256': runner.sha256(observed)}}
    original_audit = original / 'statistics_history_audit.json'
    original_audit.write_text(json.dumps(audit))
    before = runner.sha256(original_audit)
    a, records = runner._freeze_reporting(baseline, workspace, {'reporting_sources': audit})
    b, again = runner._freeze_reporting(baseline, workspace, {'reporting_sources': audit})
    assert a == b and records == again
    assert runner.sha256(original_audit) == before
    assert Path(a['observed']['artifact_path']).read_bytes() == b'frozen observations'
    assert a['extracted_at_utc'] == audit['extracted_at_utc']
    assert a['observed']['artifact_sha256'] == audit['observed']['artifact_sha256']
    assert all(Path(x['snapshot_path']).is_relative_to(workspace) for x in records)


def test_status_does_not_create_workdir(tmp_path, monkeypatch, capsys):
    settings = {'delivery_day': '2026-09-11', 'zones': ['FR'], 'output_root': tmp_path / 'candidate'}
    monkeypatch.setattr(runner, 'load_settings', lambda _: settings)
    assert runner.main(['--action', 'status']) == 0
    assert not settings['output_root'].exists()
    result = json.loads(capsys.readouterr().out)
    assert result['progress'] is None and not result['reports_available']


def test_workdir_rejects_unknown_zones(tmp_path):
    with pytest.raises(ValueError):
        runner.workdir_for({'output_root': tmp_path}, '2026-09-11', '../live')


def test_launcher_has_no_operational_invocation():
    path = Path(runner.__file__).with_name('NuclearCWE.ps1')
    text = path.read_text(encoding='utf-8')
    assert 'run_nuclear_cwe_forecast.py' in text
    assert 'Invoke-Expression' not in text
    assert '& $PythonExecutable @CweArguments' in text
    assert 'run_multicountry_forecast.py' not in text


def test_invalid_worker_count_fails_before_actions():
    with pytest.raises(SystemExit):
        runner.main(['--workers', '0'])


def test_prepared_snapshot_is_rejected_when_pinned_data_changes(tmp_path, monkeypatch):
    from chronos2_hourly import nuclear_cwe_forecast as engine
    monkeypatch.setattr(engine, 'nuclear_cwe_input_protocol', lambda: 'test')
    config = tmp_path / 'resolved_config.yaml'
    config.write_text('model: {}')
    pinned = tmp_path / 'snapshot/source'
    pinned.parent.mkdir()
    pinned.write_bytes(b'frozen')
    manifest = {'identity': {'zone': 'FR', 'delivery_day': '2026-09-11', 'config_sha256': 'config', 'input_protocol': 'test'},
                'files': [{'snapshot_path': str(pinned), 'sha256': runner.sha256(pinned)}], 'reference_files': [],
                'resolved_config_sha256': runner.sha256(config)}
    runner.write_json(tmp_path / 'input_snapshot.json', manifest)
    assert runner._verify_snapshot(tmp_path, {'config_sha256': 'config'}, '2026-09-11', 'FR') == {'model': {}}
    pinned.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Snapshot CWE modifie'):
        runner._verify_snapshot(tmp_path, {'config_sha256': 'config'}, '2026-09-11', 'FR')


def test_run_zone_uses_cwe_kalman_boundary_for_resumed_forecast(tmp_path, monkeypatch):
    from chronos2_hourly import nuclear_cwe_forecast, nuclear_preparation
    from chronos2_hourly.nuclear_cwe_kalman import build_nuclear_cwe_kalman_view
    from chronos2_modular import common

    class ForecastReached(Exception):
        pass

    config = {'nuclear_experiment': {'history_anchor_day': '2024-09-09'}}
    data = object()
    monkeypatch.setattr(runner, '_verify_snapshot', lambda *args: config)
    monkeypatch.setattr(runner, '_verify_reporting', lambda *args, **kwargs: (pd.Series(dtype=float), {}))
    monkeypatch.setattr(common, 'build_zone_configs',
                        lambda *args: [SimpleNamespace(timezone='Europe/Paris')])
    monkeypatch.setattr(nuclear_preparation, 'prepare_nuclear_zone_data', lambda *args: data)
    captured = {}

    def capture_forecast(function, **kwargs):
        captured.update(function=function, **kwargs)
        raise ForecastReached

    monkeypatch.setattr(runner, 'run_forecast_with_storage_retry', capture_forecast)
    settings = {'output_root': tmp_path / 'candidate'}
    args = SimpleNamespace(action='run', device='auto', threads=4, workers=4)
    with pytest.raises(ForecastReached):
        runner.run_zone(settings, '2026-09-11', 'FR', args)
    assert captured['function'] is nuclear_cwe_forecast.run_nuclear_cwe_forecast
    assert captured['kalman_builder'] is build_nuclear_cwe_kalman_view
    assert captured['config'] is config and captured['data'] is data
    assert captured['delivery_day'] == '2026-09-11'
    assert captured['threads'] == captured['workers'] == 4
