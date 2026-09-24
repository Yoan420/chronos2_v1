"""Launcher/snapshot contracts. No network or operational writes."""
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

import run_heatwave_forecast as runner


@pytest.mark.parametrize('value', ['../2026-09-11', '2026-09-11T08:00', '2026-02-30', '2026-9-11', None])
def test_strict_date(value):
    with pytest.raises((ValueError, TypeError)):
        runner._day(value)


@pytest.mark.parametrize('key,value', [
    ('output_root','runs/exports'), ('output_root','runs/experiments/nuclear_cwe_v1'),
    ('source_root','data/pit/kalman_weather'), ('output_root','runs/experiments/heatwave_v1/../../live'),
    ('baseline_root','runs/live'), ('baseline_kind','unknown')])
def test_settings_refuse_out_of_scope_paths(tmp_path, monkeypatch, key, value):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    config = tmp_path / 'config/heat.yaml'
    config.parent.mkdir()
    config.write_text(yaml.safe_dump({'schema_version': 1, 'project_root': '..',
        'delivery_day': '2026-09-11', key: value}))
    with pytest.raises(ValueError):
        runner.load_settings(config)


def test_settings_accept_custom_isolated_experiment_and_features(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    config = tmp_path / 'config/heat.yaml'
    config.parent.mkdir()
    config.write_text(yaml.safe_dump({'schema_version': 1, 'project_root': '..',
        'delivery_day': '2026-09-11', 'output_root': 'runs/experiments/heatwave_v1/second',
        'features': {'persistent_days': 4}}))
    settings = runner.load_settings(config)
    assert settings['features']['persistent_days'] == 4
    assert settings['features']['countries'] == ['FR','DE','BE','NL','ES']
    assert str(runner.baseline_for(settings, '2026-09-11', 'FR')).endswith('civil_pit_v2')
    settings['baseline_kind'] = 'nuclear_cwe'
    assert runner.baseline_for(settings, '2026-09-11', 'FR').name == 'fr'


def test_status_is_read_only(tmp_path, monkeypatch, capsys):
    settings = {'delivery_day': '2026-09-11', 'zones': ['FR'], 'output_root': tmp_path / 'candidate'}
    monkeypatch.setattr(runner, 'load_settings', lambda _: settings)
    assert runner.main(['--action','status']) == 0
    assert not settings['output_root'].exists()
    assert json.loads(capsys.readouterr().out)['progress'] is None


def test_snapshot_pins_code_recipe_and_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, '_protocol', lambda _: 'test-protocol')
    config = tmp_path / 'resolved_config.yaml'
    config.write_text('model: {}')
    pinned = tmp_path / 'snapshot/source'
    pinned.parent.mkdir()
    pinned.write_bytes(b'frozen')
    manifest = {'identity': {'zone':'FR','delivery_day':'2026-09-11','config_sha256':'recipe',
        'input_protocol':'test-protocol','orchestrator_sha256':runner.sha256(Path(runner.__file__))},
        'files':[{'snapshot_path':str(pinned),'sha256':runner.sha256(pinned)}],
        'reference_files':[], 'resolved_config_sha256':runner.sha256(config)}
    runner.write_json(tmp_path / 'input_snapshot.json', manifest)
    assert runner._verify_snapshot(tmp_path, {'config_sha256':'recipe'}, '2026-09-11','FR') == {'model':{}}
    with pytest.raises(ValueError, match='incompatible'):
        runner._verify_snapshot(tmp_path, {'config_sha256':'other'}, '2026-09-11','FR')
    pinned.write_bytes(b'changed')
    with pytest.raises(ValueError, match='modifie'):
        runner._verify_snapshot(tmp_path, {'config_sha256':'recipe'}, '2026-09-11','FR')


def test_launcher_passes_argv_and_has_no_operational_invocation():
    text = Path(runner.__file__).with_name('Heatwave.ps1').read_text(encoding='utf-8')
    assert 'run_heatwave_forecast.py' in text
    assert '& $PythonExecutable @HeatwaveArguments' in text
    assert 'Invoke-Expression' not in text and 'run_multicountry_forecast.py' not in text


def test_invalid_workers_stop_before_io():
    with pytest.raises(SystemExit):
        runner.main(['--workers','0'])


def test_report_missing_result_never_starts_training(tmp_path, monkeypatch):
    settings = {'delivery_day':'2026-09-11', 'zones':['FR'], 'output_root':tmp_path/'candidate'}
    monkeypatch.setattr(runner, 'load_settings', lambda _:settings)
    with pytest.raises(ValueError, match='Prepare / Run'):
        runner.main(['--action','report'])


def test_sync_does_not_require_baseline(tmp_path, monkeypatch):
    from chronos2_hourly import heatwave_sources
    settings = {'delivery_day':'2026-09-11', 'zones':['FR'], 'output_root':tmp_path/'candidate',
                'source_root':tmp_path/'sources','source_start_day':'2024-06-30'}
    monkeypatch.setattr(runner, 'load_settings', lambda _:settings)
    calls = []
    monkeypatch.setattr(heatwave_sources, 'materialize_temperature_sources',
                        lambda *a, **k: calls.append((a,k)) or {})
    assert runner.main(['--action','sync']) == 0
    assert calls[0][0][1:] == ('2024-06-30','2026-09-11')


@pytest.mark.parametrize('kind', ['nuclear_fr','nuclear_cwe'])
def test_real_baseline_clone_and_native_dst_preflight(tmp_path, kind):
    """Optional local frozen fixture; all outputs stay in pytest temp, no fitting."""
    from chronos2_hourly.heatwave_sources import audit_temperature_store
    from chronos2_hourly.heatwave_forecast import _prepare_inputs
    from chronos2_hourly.nuclear_preparation import prepare_nuclear_zone_data
    from chronos2_hourly.nuclear_run_archive import _source_contract
    from chronos2_modular.common import build_zone_configs
    baseline = runner.ROOT / ('runs/experiments/nuclear_forecast_v1/2026-09-11/fr/civil_pit_v2'
                             if kind == 'nuclear_fr' else 'runs/experiments/nuclear_cwe_v1/2026-09-11/fr')
    if not (baseline / 'run_result.json').is_file():
        pytest.skip('Optional frozen baseline unavailable')
    # Use only completed, separately audited temperature bundles.
    sources = {}
    for country in ['FR','DE','BE','NL','ES']:
        alias = country.lower() + '_temperature_fcst'
        source = runner.ROOT / f'data/pit/heatwave/saturn_daily_v1/{alias}/bundles/2024-06-30_2026-09-11/{alias}.parquet'
        if not source.is_file():
            pytest.skip('Optional completed temperature bundle unavailable')
        audit = audit_temperature_store(source,country,'2024-06-30','2026-09-11')
        sources[alias] = {'path':str(source), 'audit':audit,
            'specification':{'series':f'meteo.nrjscan.{country.lower()}.t_2m.index.fcst.d'}}
    files = [baseline/name for name in ['run_result.json','resolved_config.yaml','input_snapshot.json']]
    original = yaml.safe_load(files[1].read_text(encoding='utf-8'))
    configured = {'target':original['zones']['FR']['target']['file'], **original['data']['pit_files']}
    files += [Path(p) for p in configured.values()]
    before = {p:runner.sha256(p) for p in files}
    settings = {'output_root':tmp_path/'candidate','baseline_root':baseline.parents[2 if kind == 'nuclear_fr' else 1],
        'baseline_kind':kind,'config_sha256':'test','source_start_day':'2024-06-30'}
    config = runner.prepare_snapshot(settings,'2026-09-11','FR',sources)
    workdir = runner.workdir_for(settings,'2026-09-11','FR')
    assert _source_contract(workdir)['snapshot_identity']['zone'] == 'FR'
    assert config['model'] == original['model']
    assert config['backtest'] == original['backtest']
    assert config['hourly']['residual_correction'] == original['hourly']['residual_correction']
    assert config['nuclear_experiment']['filter_parameters'] == original['nuclear_experiment']['filter_parameters']
    assert config['nuclear_experiment']['history_anchor_day'] == original['nuclear_experiment']['history_anchor_day']
    candidate_paths = {'target':config['zones']['FR']['target']['file'],**config['data']['pit_files']}
    assert len(set(candidate_paths)-set(configured)) == 17
    for alias, path in configured.items():
        assert runner.sha256(Path(candidate_paths[alias])) == runner.sha256(Path(path))
    spec = build_zone_configs(config,['FR'],None,None)[0]
    data = prepare_nuclear_zone_data(spec,config,workdir,workdir/'prepared')
    assert str(data.model_context_covariates.index.tz) == 'Europe/Paris'
    context, aliases, coverage, hours = _prepare_inputs(config,data,'FR',pd.Timestamp('2026-09-11').date())
    assert len(aliases) == (17 if kind == 'nuclear_fr' else 19)
    assert len(context.loc[hours]) == 17592
    assert all(v['missing_required_hours'] == 0 for v in coverage.values())
    assert runner.prepare_snapshot(settings,'2026-09-11','FR',{}) == config
    assert {p:runner.sha256(p) for p in files} == before
