from contextlib import nullcontext
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

import run_solarwind_forecast as cli
from chronos2_hourly import solarwind_live as m


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    output = tmp_path / 'sw'
    model = output
    monkeypatch.setattr(m, 'OUTPUT', output)
    monkeypatch.setattr(m, 'MODEL_ROOT', model)
    monkeypatch.setattr(m, 'MODEL_OUTPUT', model / 'runs/experiments/solar_wind_v1')
    return output


@pytest.fixture
def recipe():
    return {'model': {'model_id': 'amazon/chronos-2'}, 'hourly': {'residual_correction': {
        'enabled': True, 'base_model': 'chronos2', 'backend': 'catboost', 'iterations': 700,
        'depth': 6, 'learning_rate': .03, 'l2_leaf_reg': 15., 'min_samples_leaf': 30,
        'min_training_rows': 720, 'random_state': 42, 'correction_scale': 1.,
        'max_abs_correction': 40., 'feature_builder': {'exclude_historical_prices': True}}}}


def test_cli_dispatch_is_explicit_and_no_default_production_call(monkeypatch):
    seen = []
    monkeypatch.setattr(m, 'run', lambda *args, **kwargs: seen.append((args, kwargs)))
    assert cli.main(['--delivery-day', '2026-09-24']) == 0
    assert len(seen) == 1
    args, kwargs = seen[0]
    assert args == ('2026-09-24',)
    assert tuple(kwargs.pop('zones')) == ('DE', 'NL')
    assert kwargs == {'threads': 2, 'workers': 2, 'device': 'auto', 'sync': True}


@pytest.mark.parametrize('args', [[], ['--delivery-day', '2026-09-24', '--zones', 'FR'],
    ['--delivery-day', '2026-09-24', '--zones', 'DE'],
    ['--delivery-day', '2026-09-24', '--zones', 'DE', 'DE'],
    ['--delivery-day', '2026-09-24', '--threads', '0']])
def test_cli_rejects_unsupported_request_before_model_import(args):
    with pytest.raises(SystemExit):
        cli.main(args)


@pytest.mark.parametrize('field,value', [('max_abs_correction', 80.), ('correction_scale', .5),
                                       ('backend', 'hist_gradient_boosting'), ('iterations', 701)])
def test_model_recipe_drift_is_rejected(recipe, field, value):
    recipe['hourly']['residual_correction'][field] = value
    with pytest.raises(ValueError, match='recipe'):
        m.validate_recipe(recipe)


def test_output_escape_refused(isolated):
    with pytest.raises(ValueError):
        m.safe(isolated.parent / 'production.json')
    with pytest.raises(ValueError):
        m.safe(isolated)


def final_result(day='2026-09-24', zone='DE'):
    timezone = 'Europe/Berlin' if zone == 'DE' else 'Europe/Amsterdam'
    start = pd.Timestamp(day)
    index = pd.date_range(start.tz_localize(timezone), (start + pd.Timedelta(days=1)).tz_localize(timezone),
                          freq='h', inclusive='left').tz_convert('UTC')
    frame = pd.DataFrame({'timestamp': index, 'residual_kalman__q10': 10.,
                          'residual_kalman__q50': 20., 'residual_kalman__q90': 30.})
    return SimpleNamespace(kalman_view=SimpleNamespace(forecast=frame))


@pytest.mark.parametrize('day,hours', [('2026-09-24', 24), ('2026-10-25', 25), ('2026-03-29', 23)])
def test_published_quantiles_match_exact_delivery_including_dst(isolated, day, hours):
    directory = isolated / day
    directory.mkdir(parents=True)
    files = m.publish(directory, {'DE': final_result(day)})
    assert set(files) == {'index.html', 'forecast_de.csv'}
    assert all(m.sha(directory / name) == digest for name, digest in files.items())
    frame = pd.read_csv(directory / 'forecast_de.csv')
    assert len(frame) == hours
    assert frame.p50_eur_mwh.eq(20.).all()
    assert not (directory / 'manifest.json').exists()


@pytest.mark.parametrize('kind', ['missing_hour', 'crossed', 'nonfinite', 'future_actual'])
def test_publication_rejects_invalid_final_output(isolated, kind):
    directory = isolated / '2026-09-24'
    directory.mkdir(parents=True)
    result = final_result()
    if kind == 'missing_hour': result.kalman_view.forecast = result.kalman_view.forecast.iloc[:-1]
    if kind == 'crossed': result.kalman_view.forecast['residual_kalman__q10'] = 100.
    if kind == 'nonfinite': result.kalman_view.forecast.loc[0, 'residual_kalman__q50'] = np.nan
    if kind == 'future_actual': result.kalman_view.forecast['actual'] = 1.
    with pytest.raises(ValueError):
        m.publish(directory, {'DE': result})
    assert not (directory / 'manifest.json').exists()


def test_completed_delivery_is_verified_and_never_resynchronizes(monkeypatch, isolated):
    import run_nuclear_forecast as nuclear
    day = '2026-09-24'
    directory = isolated / day
    directory.mkdir(parents=True)
    files = m.publish(directory, {zone: final_result(day, zone) for zone in m.ZONES})
    receipt = {'model': m.ENGINE, 'status': 'COMPLETE', 'delivery_day': day, 'files': files}
    m.write_json(directory / 'manifest.json', receipt)
    monkeypatch.setattr(nuclear, 'delivery_date', lambda value: pd.Timestamp(value))
    monkeypatch.setattr(m, 'isolated_settings', lambda: pytest.fail('completed result must stay immutable'))
    assert m.run(day) == receipt
    (directory / 'forecast_de.csv').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='checksum'):
        m.run(day)
    assert json.loads((directory / 'manifest.json').read_text()) == receipt


def test_missing_model_fails_before_sources_and_preserves_production(monkeypatch, isolated, recipe):
    import run_nuclear_forecast as nuclear
    protected = isolated.parent / 'current-production.txt'
    protected.parent.mkdir(parents=True, exist_ok=True)
    protected.write_text('unchanged', encoding='utf-8')
    monkeypatch.setattr(nuclear, 'delivery_date', lambda value: pd.Timestamp(value))
    monkeypatch.setattr(nuclear, 'check_lora_inactive', lambda *args: None)
    monkeypatch.setattr(nuclear, 'zone_inputs', lambda *args: (recipe, None, {}))
    monkeypatch.setattr(m, 'isolated_settings', lambda: {})
    monkeypatch.setattr(m, 'capture_sources', lambda *args, **kwargs: pytest.fail('model preflight first'))
    def unavailable(config):
        raise OSError('weights absent')
    monkeypatch.setattr(nuclear, 'resolve_local_model_revision', unavailable)
    with pytest.raises(ValueError, match='local weights'):
        m.run('2026-09-24')
    assert protected.read_text() == 'unchanged'
    status = json.loads((isolated / '2026-09-24/run_status.json').read_text())
    assert status['status'] == 'FAILED'
    assert not (isolated / '2026-09-24/manifest.json').exists()


def test_forecast_wires_exact_solarwind_and_corrector_only_interaction(monkeypatch, isolated):
    from dataclasses import make_dataclass
    import chronos2_modular.common as common
    import chronos2_hourly.nuclear_preparation as preparation
    import chronos2_hourly.nuclear_run_archive as archive
    import chronos2_hourly.nyx_live_baseline as interaction
    import chronos2_hourly.solar_wind_forecast as solarwind
    import chronos2_hourly.solar_wind_interaction_features as features
    import run_nuclear_forecast as nuclear
    work = m.MODEL_OUTPUT / '2026-09-24/de'
    work.mkdir(parents=True)
    identity = {'zone': 'DE', 'delivery_day': '2026-09-24'}
    config = {'selected': 'recipe'}
    score = pd.DataFrame({'stress': [.2]}, index=pd.date_range('2026-09-24', periods=1, tz='UTC'))
    expected_factory, data = object(), object()
    seen, stored = {}, {}
    ScientificResult = make_dataclass('ScientificResult', ['residual_daily_audit', 'covariates', 'audit'])
    result = ScientificResult(pd.DataFrame({'generation_source': ['daily_prequential_refit'],
        'residual_feature_columns': [['stress', 'known_de_wind_generation_fcst_oracle']]}),
        pd.DataFrame({'de_wind_generation_fcst': [1.]}), {})
    monkeypatch.setattr(common, 'build_zone_configs', lambda *args: [object()])
    monkeypatch.setattr(preparation, 'prepare_nuclear_zone_data', lambda *args: data)
    def interaction_feature(actual_data, actual_config, actual_identity, actual_work, builder):
        assert actual_data is data and actual_config is config and actual_identity is identity
        assert builder is features.build_interaction
        return score, {'feature_location': 'corrector_only'}, None
    monkeypatch.setattr(interaction, 'interaction_feature', interaction_feature)
    def factory(actual_config, actual_score):
        assert actual_config is config and actual_score is score
        return expected_factory
    monkeypatch.setattr(interaction, 'make_residual_factory', factory)
    def forecast(**kwargs):
        seen.update(kwargs)
        return result
    monkeypatch.setattr(solarwind, 'run_solar_wind_forecast', forecast)
    monkeypatch.setattr(nuclear, 'run_forecast_with_storage_retry', lambda fn, **kwargs: fn(**kwargs))
    monkeypatch.setattr(nuclear, 'run_progress', lambda *args: nullcontext(lambda *a, **k: None))
    monkeypatch.setattr(m, 'verify_snapshot', lambda *args: config)
    monkeypatch.setattr(archive, 'save_nuclear_result_bundle', lambda item, **kwargs: stored.update(result=item))
    monkeypatch.setattr(archive, 'load_nuclear_result_bundle', lambda **kwargs: stored['result'])
    actual = m.run_zone(work, config, identity, threads=2, workers=1, device='cpu')
    assert seen['residual_factory'] is expected_factory
    assert seen['delivery_day'] == '2026-09-24' and seen['zone'] == 'DE'
    assert seen['config'] is config and seen['data'] is data
    assert actual.audit['solarwind_app_identity'] is identity
    assert actual.audit['candidate_variant'] == 'solarwind_interaction40'


def test_current_source_paths_are_never_used_as_outputs():
    settings = m.isolated_settings()
    assert settings['output_root'] == m.MODEL_OUTPUT
    assert settings['nuclear_store'].parent == m.NUCLEAR_ROOT
    assert settings['residual_bank'].parent == m.NUCLEAR_ROOT
    assert settings['residual_bank_seed'] != settings['residual_bank']
    assert settings['output_root'].as_posix().endswith('runs/experiments/solar_wind_v1')


def test_run_rejects_partial_country_publication_before_writes(isolated):
    with pytest.raises(ValueError):
        m.run('2026-09-24', zones=['DE'])
    assert not isolated.exists()


def test_prepare_relocates_and_seals_every_input_with_real_epoch(monkeypatch, isolated, recipe):
    from copy import deepcopy
    from pathlib import Path
    import run_nuclear_forecast as nuclear
    from chronos2_hourly.nuclear_incremental import prepare_incremental_settings
    from chronos2_hourly.solar_wind_forecast import GENERATION_SERIES, solar_wind_input_protocol
    day = pd.Timestamp('2026-09-24')
    baseline = [f'{zone}_residual_load_fcst' for zone in ('fr', 'de', 'be', 'nl', 'es')]
    baseline += ['fr_nuclear_generation_fcst_gw']
    before, old_paths = {}, {}

    def snapshot(settings, zone, requested_day, work, **kwargs):
        assert settings['output_root'] == m.MODEL_OUTPUT
        assert requested_day == day and zone == 'DE'
        work.mkdir(parents=True)
        snapshot_dir = work / 'snapshot'
        snapshot_dir.mkdir()
        files = []
        for alias in ['target', *baseline]:
            path = snapshot_dir / (alias + '.parquet')
            path.write_bytes(('synthetic audited bytes: ' + alias).encode())
            old_paths[alias] = str(path)
            before[path] = m.sha(path)
            files.append({'snapshot': str(path), 'sha256': m.sha(path)})
        sidecar = snapshot_dir / 'fr_residual_load_fcst.parquet.audit.json'
        sidecar.write_text('{"kind":"synthetic residual audit"}', encoding='utf-8')
        before[sidecar] = m.sha(sidecar)
        files.append({'snapshot': str(sidecar), 'sha256': m.sha(sidecar)})
        config = deepcopy(recipe)
        config['hourly']['feature_engineering'] = {
            'covariate_columns': [f'known_{alias}_oracle' for alias in baseline]}
        config.update(
            zones={'DE': {'timezone': 'Europe/Berlin', 'target': {'file': old_paths['target']},
                'covariates': {alias: {'enabled': True, 'pit_file': old_paths[alias]}
                               for alias in baseline}}},
            data={'pit_files': {alias: old_paths[alias] for alias in baseline}},
            output={'directory': str(work)},
            nuclear_experiment={'mode': 'incremental', 'input_protocol': 'civil_pit_v2',
                'incremental_cache_dir': str(m.MODEL_OUTPUT / '_daily_cache/de/civil_pit_v2'),
                'residual_bank_audit': str(sidecar)})
        prepare_incremental_settings(config, requested_day.date())
        old_paths['base_epoch'] = str(Path(config['nuclear_experiment']['incremental_namespace']) / 'epoch.json')
        before[Path(old_paths['base_epoch'])] = m.sha(old_paths['base_epoch'])
        m.write_json(work / 'input_snapshot.json', {'files': files})
        return config

    monkeypatch.setattr(nuclear, 'snapshot_config', snapshot)
    monkeypatch.setattr(nuclear, 'refreshed_target_snapshot',
                        lambda *args, **kwargs: pytest.fail('offline prepare must not refresh targets'))
    donor = isolated / 'synthetic-generation-donor'
    donor.mkdir(parents=True)
    sources = {}
    for alias, series in GENERATION_SERIES.items():
        path = donor / (alias + '.parquet')
        path.write_bytes(('synthetic generation: ' + alias).encode())
        audit = Path(str(path) + '.audit.json')
        audit.write_text(json.dumps({'alias': alias, 'series': series}), encoding='utf-8')
        before[path], before[audit] = m.sha(path), m.sha(audit)
        sources[alias] = {'path': str(path), 'sha256': m.sha(path), 'audit_sha256': m.sha(audit)}
    settings = {'output_root': m.MODEL_OUTPUT}
    work, config, identity = m.prepare(settings, 'DE', day, sources, threads=2, sync=False)
    manifest = m.read(work / 'input_snapshot.json')
    sealed = {Path(item['snapshot']) for item in manifest['files']}
    expected_inputs = {Path(config['zones']['DE']['target']['file']),
                       *(Path(value) for value in config['data']['pit_files'].values()),
                       Path(config['nuclear_experiment']['residual_bank_audit'])}
    assert expected_inputs <= sealed
    assert all(path.is_relative_to(work / 'snapshot') for path in sealed)
    assert all(Path(value).is_relative_to(work / 'snapshot')
               for value in config['data']['pit_files'].values())
    assert set(config['data']['pit_files']) == set(baseline) | set(GENERATION_SERIES)
    for alias in GENERATION_SERIES:
        assert config['zones']['DE']['covariates'][alias]['pit_file'] == config['data']['pit_files'][alias]
        assert f'known_{alias}_oracle' in config['hourly']['feature_engineering']['covariate_columns']
        assert Path(str(config['data']['pit_files'][alias]) + '.audit.json') in sealed
    assert config['hourly']['residual_correction']['max_abs_correction'] == 40.
    assert config['hourly']['residual_correction']['thread_count'] == 2
    assert config['data']['project_root'] == str(m.MODEL_ROOT)
    experiment = config['nuclear_experiment']
    assert experiment['input_protocol'] == solar_wind_input_protocol()
    assert experiment['history_anchor_day'] == (day - pd.Timedelta(days=730)).date().isoformat()
    namespace = Path(experiment['incremental_namespace'])
    assert namespace.is_relative_to(m.MODEL_OUTPUT / '_daily_cache/de/i40')
    assert namespace / 'epoch.json' != Path(old_paths['base_epoch'])
    assert (namespace / 'epoch.json').is_file()
    assert all(m.sha(path) == digest for path, digest in before.items())
    assert m.verify_snapshot(work, identity) == config
    assert m.prepare(settings, 'DE', day, sources, threads=2, sync=False) == (work, config, identity)
    changed = Path(config['data']['pit_files']['de_wind_generation_fcst'])
    changed.write_bytes(b'altered snapshot')
    with pytest.raises(ValueError, match='snapshot'):
        m.verify_snapshot(work, identity)


@pytest.mark.parametrize('redirected_kind', ['nuclear_file', 'nuclear_audit', 'nuclear_sync_lock', 'residual_file', 'residual_lock'])
def test_capture_rejects_existing_redirected_output_before_sync(monkeypatch, isolated, redirected_kind):
    from pathlib import Path
    import run_nuclear_forecast as nuclear
    cache = isolated / 'source-cache'
    cache.mkdir(parents=True)
    monkeypatch.setattr(m, 'NUCLEAR_ROOT', cache)
    nuclear_file = cache / 'fr_nuclear_generation_fcst_gw.parquet'
    residual_file = cache / 'residual_load_market_features.parquet'
    targets = {'nuclear_file': nuclear_file, 'nuclear_audit': Path(str(nuclear_file) + '.audit.json'),
        'nuclear_sync_lock': nuclear_file.with_suffix('.sync.lock'), 'residual_file': residual_file,
        'residual_lock': Path(str(residual_file) + '.lock')}
    for target in targets.values():
        target.write_text('existing optional cache', encoding='utf-8')
    Path(str(residual_file) + '.audit.json').write_text('{}', encoding='utf-8')
    protected = isolated.parent / 'production.parquet'
    protected.write_text('production remains untouched', encoding='utf-8')
    redirected = targets[redirected_kind]
    original_resolve = Path.resolve
    def resolve(path, *args, **kwargs):
        if path.absolute() == redirected.absolute():
            return protected.absolute()
        return original_resolve(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'resolve', resolve)
    monkeypatch.setattr(nuclear, 'sync_source', lambda *a, **k: pytest.fail('sync must not run'))
    monkeypatch.setattr(nuclear, 'ensure_residual_inputs', lambda *a, **k: pytest.fail('residual sync must not run'))
    monkeypatch.setattr(nuclear, 'required_source_bounds', lambda *a, **k: pytest.fail('redirected paths rejected first'))
    settings = {'nuclear_store': nuclear_file, 'residual_bank': residual_file}
    with pytest.raises(ValueError, match='unredirected'):
        m.capture_sources(settings, pd.Timestamp('2026-09-24'), 1, sync=True)
    assert protected.read_text() == 'production remains untouched'
