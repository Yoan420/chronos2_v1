"""Bounded synthetic tests; never run a scientific fit or production writer."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_live_baseline as m


@pytest.fixture
def frame():
    index = pd.date_range('2026-09-20', periods=8, freq='h', tz='UTC')
    return pd.DataFrame({'known_fr_nuclear_generation_fcst_gw_oracle': np.arange(8.)}, index=index)


@pytest.fixture
def recipe():
    return {'hourly': {'residual_correction': {'enabled': True, 'base_model': 'chronos2',
        'backend': 'catboost', 'iterations': 700, 'depth': 6, 'thread_count': 1,
        'min_training_rows': 720, 'max_abs_correction': 40., 'correction_scale': 1.,
        'feature_builder': {'timezone': 'Europe/Berlin', 'exclude_historical_prices': True}}}}


def test_add_feature_copy_and_zero_prefix(frame):
    before = frame.copy(deep=True)
    score = pd.DataFrame({'stress': [.1, .2, .3, .4]}, index=frame.index[4:])
    actual = m.add_feature(frame, score)
    pd.testing.assert_frame_equal(frame, before)
    assert actual.stress.tolist() == [0., 0., 0., 0., .1, .2, .3, .4]


@pytest.mark.parametrize('kind', ['internal_gap', 'future_gap', 'duplicate', 'negative', 'nan', 'large', 'naive', 'empty'])
def test_score_bad_contract_rejected(frame, kind):
    score = pd.DataFrame({'stress': np.full(8, .25)}, index=frame.index)
    if kind == 'internal_gap': score = score.drop(score.index[4])
    if kind == 'future_gap': score = score.iloc[:-1]
    if kind == 'duplicate': score = pd.concat([score, score.iloc[-1:]])
    if kind == 'negative': score.iloc[0, 0] = -.1
    if kind == 'nan': score.iloc[0, 0] = np.nan
    if kind == 'large': score.iloc[0, 0] = 1.01
    if kind == 'naive': score.index = score.index.tz_localize(None)
    if kind == 'empty': score = score.iloc[:0]
    with pytest.raises(ValueError): m.add_feature(frame, score)


def test_double_injection_refused(frame):
    with pytest.raises(ValueError): m.add_feature(frame, frame.rename(columns={frame.columns[0]: frame.columns[0]}))


def test_factory_persistent_identity_stable_and_score_sensitive(recipe, frame):
    from chronos2_hourly.nuclear_forecast import _factory_cache_identity
    score = pd.DataFrame({'stress': np.full(8, .25)}, index=frame.index)
    before = deepcopy(recipe)
    first = m.make_residual_factory(recipe, score)
    second = m.make_residual_factory(recipe, score.copy())
    changed = m.make_residual_factory(recipe, score * 2)
    assert _factory_cache_identity(first) == _factory_cache_identity(second)
    assert _factory_cache_identity(first) != _factory_cache_identity(changed)
    assert first().correction_bounds_ == (-40., 40.)
    assert recipe == before


def test_factory_injects_only_at_fit_predict_boundary(monkeypatch, recipe, frame):
    seen = []
    def fake_fit(self, X, y, base, experts=None):
        seen.append(X.copy()); self.feature_columns_ = tuple(X); return self
    def fake_predict(self, X, base, experts=None):
        seen.append(X.copy()); return base.copy()
    monkeypatch.setattr(m.ResidualCorrector, 'fit', fake_fit)
    monkeypatch.setattr(m.ResidualCorrector, 'predict', fake_predict)
    score = pd.DataFrame({'stress': np.full(8, .25)}, index=frame.index)
    base = pd.DataFrame({'q10': 1., 'q50': 2., 'q90': 3.}, index=frame.index)
    model = m.make_residual_factory(recipe, score)()
    model.fit(frame, pd.Series(2., index=frame.index), base)
    result = model.predict(frame, base)
    assert all('stress' in x for x in seen)
    assert 'stress' not in frame and 'stress' not in base
    pd.testing.assert_frame_equal(result, base)


@pytest.mark.parametrize('field,value', [('max_abs_correction', 80.), ('correction_scale', .5), ('enabled', False)])
def test_recipe_change_rejected(recipe, frame, field, value):
    recipe['hourly']['residual_correction'][field] = value
    with pytest.raises(ValueError):
        m.make_residual_factory(recipe, pd.DataFrame({'stress': .25}, index=frame.index))


def test_safe_paths_confined_and_json_atomic(monkeypatch, tmp_path):
    root = tmp_path / 'isolated'
    monkeypatch.setattr(m, 'OUTPUT', root)
    with pytest.raises(ValueError): m.safe_path(tmp_path / 'production/file.json')
    with pytest.raises(ValueError): m.safe_path(root)
    path = root / 'state.json'
    m.write_json(path, {'status': 'ok'})
    assert json.loads(path.read_text()) == {'status': 'ok'}
    assert not path.with_suffix('.json.tmp').exists()


def test_windows_retry_is_bounded(monkeypatch, tmp_path):
    calls = []
    def denied(self, dst):
        calls.append(dst)
        error = PermissionError('locked'); error.winerror = 5; raise error
    monkeypatch.setattr(Path, 'replace', denied)
    monkeypatch.setattr(m.time, 'sleep', lambda duration: None)
    with pytest.raises(PermissionError): m._replace(tmp_path / 'a', tmp_path / 'b')
    assert len(calls) == 8


def test_readonly_validate_never_prepares_or_locks(monkeypatch, tmp_path):
    work = tmp_path / 'uncreated'
    identity = {'run_id': 'test'}
    monkeypatch.setattr(m, '_inputs', lambda *a: (identity, work, work, work, None, {}, {}))
    monkeypatch.setattr(m, '_prepare', lambda *a: pytest.fail('validate must not prepare'))
    monkeypatch.setattr(m, 'exclusive_process_lock', lambda *a: pytest.fail('validate must not lock'))
    result = m.run_zone('DE', '2026-09-24', work, {}, action='validate')
    assert result['status'] == 'VALIDATED'
    assert not work.exists()


def test_befr_requires_explicit_extension_before_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(m, '_inputs', lambda *a: pytest.fail('must reject before donor loading'))
    with pytest.raises(ValueError, match='separately approved'):
        m.run_zone('BE', '2026-09-24', tmp_path, {}, action='validate')


def test_source_capture_receipts_and_extra_inputs(monkeypatch, tmp_path):
    root = tmp_path / 'isolated'
    root.mkdir()
    monkeypatch.setattr(m, 'OUTPUT', root)
    records = {}
    for alias, series in {**m.GENERATION_SERIES, **m.EXTRA_SERIES}.items():
        path = root / (alias + '.parquet'); path.write_bytes(b'sealed fixture')
        audit = {'alias': alias, 'series': series, 'sha256': m.sha(path),
            'start_day': '2024-09-09', 'end_day': '2026-09-24', 'daily_broadcast': False}
        sidecar = Path(str(path) + '.audit.json'); sidecar.write_text(json.dumps(audit))
        records[alias] = {'path': str(path), 'sha256': m.sha(path), 'audit_sha256': m.sha(sidecar)}
    result = m.source_records(records, zone='DE', day='2026-09-24', raw_start='2024-09-10', root=root)
    assert len(result) == 8
    bad = deepcopy(records); bad['de_wind_generation_fcst']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='SHA'):
        m.source_records(bad, zone='DE', day='2026-09-24', raw_start='2024-09-10', root=root)
    del records['be_wind_generation_fcst']
    with pytest.raises(ValueError):
        m.source_records(records, zone='BE', day='2026-09-24', raw_start='2024-09-10', root=root)


def test_prepare_isolated_sources_extra_wind_stays_outside_core(monkeypatch, tmp_path, recipe):
    from chronos2_hourly import nuclear_incremental
    root = tmp_path / 'isolated'; donor = tmp_path / 'donor'
    (donor / 'snapshot').mkdir(parents=True)
    root.mkdir()
    monkeypatch.setattr(m, 'OUTPUT', root)
    baseline = [f'{z}_residual_load_fcst' for z in ('fr', 'de', 'be', 'nl', 'es')]
    baseline += ['fr_nuclear_generation_fcst_gw']
    mapping, prior = {}, {'files': []}
    for alias in ['target', *baseline]:
        p = donor / 'snapshot' / (alias + '.parquet'); p.write_bytes(alias.encode())
        mapping[alias] = str(p)
        prior['files'].append({'snapshot': str(p), 'sha256': m.sha(p)})
    sources = {}
    for alias, series in {**m.GENERATION_SERIES, **m.EXTRA_SERIES}.items():
        p = root / (alias + '.parquet'); p.write_bytes(alias.encode())
        ap = Path(str(p) + '.audit.json'); ap.write_text('{}')
        sources[alias] = {'path': str(p), 'sha256': m.sha(p), 'audit_path': str(ap),
                          'audit_sha256': m.sha(ap), 'series': series}
    config = deepcopy(recipe)
    config['hourly']['feature_engineering'] = {'covariate_columns': ['known_' + a + '_oracle' for a in baseline]}
    config.update(zones={'DE': {'target': {'file': mapping['target']},
        'covariates': {a: {'enabled': True, 'pit_file': mapping[a]} for a in baseline}}},
        data={'pit_files': {a: mapping[a] for a in baseline}}, output={'directory': str(donor)},
        nuclear_experiment={'mode': 'incremental'})
    identity = {'zone': 'DE', 'delivery_day': '2026-09-24', 'run_id': 'synthetic',
        'source_records': sources, 'threads': 1, 'interaction_builder': {}, 'history_anchor_day': '2024-09-10'}
    mini = root / 'synthetic'; work = mini / 'runs/experiments/solar_wind_v1/result'
    def epoch(cfg, day):
        cfg['nuclear_experiment'].update(history_anchor_day='2024-09-10', raw_history_start_day='2024-09-10')
        return None, pd.Timestamp('2024-09-10').date(), pd.Timestamp('2024-09-10').date()
    monkeypatch.setattr(nuclear_incremental, 'prepare_incremental_settings', epoch)
    cloned = m._prepare(identity, work, mini, donor, None, prior, config)
    assert set(cloned['data']['pit_files']) == set(baseline) | set(m.GENERATION_SERIES)
    assert not set(m.EXTRA_SERIES).intersection(cloned['zones']['DE']['covariates'])
    assert (work / 'snapshot/be_wind_generation_fcst.parquet').exists()
    assert cloned['data']['project_root'] == str(mini)
    assert config['data']['pit_files'] == {a: mapping[a] for a in baseline}
    assert m._prepare(identity, work, mini, donor, None, prior, config) == cloned
    (work / 'snapshot/be_wind_generation_fcst.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Snapshot file changed'):
        m.verify_snapshot(work, identity)


def test_reference_recipe_exact_except_threads(monkeypatch, tmp_path, recipe):
    from chronos2_hourly import nuclear_forecast
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    folder = tmp_path / 'runs/experiments/solar_wind_v1/2026-09-22/de' / m.REFERENCE_RUNS['DE'] / 'report_only/frozen_result'
    folder.mkdir(parents=True)
    saved = deepcopy(recipe['hourly']['residual_correction']); saved['thread_count'] = 2
    audit = folder / 'audits.json'
    audit.write_text(json.dumps({'result': {'residual_recipe': saved, 'kalman_filter_parameters': {'shift': 20}}}))
    (folder / 'manifest.json').write_text(json.dumps({'files': {'audits.json': m.sha(audit)}}))
    monkeypatch.setattr(nuclear_forecast, '_kalman_filter_configuration', lambda cfg: (None, {'shift': 20}))
    assert m.verify_reference_recipe(recipe, 'DE')['status'] == 'same_except_explicit_thread_count'
    recipe['hourly']['residual_correction']['depth'] = 5
    with pytest.raises(ValueError, match='corrector recipe differs'):
        m.verify_reference_recipe(recipe, 'DE')


@pytest.mark.parametrize('zone', ['DE', 'NL'])
def test_interaction_local_context_normalized_to_utc_without_numeric_change(zone, tmp_path):
    tz = m.TIMEZONES[zone]
    end = pd.Timestamp('2026-09-24')
    local_index = pd.date_range((end - pd.Timedelta(days=730)).tz_localize(tz),
        (end + pd.Timedelta(days=1)).tz_localize(tz), freq='h', inclusive='left')
    values = np.full(len(local_index), np.nextafter(.1, 1.), dtype=np.float64)
    aliases = [f'{z}_residual_load_fcst' for z in ('fr', 'de', 'be', 'nl', 'es')]
    aliases += ['fr_nuclear_generation_fcst_gw', *m.GENERATION_SERIES]
    context = pd.DataFrame({f'known_{a}_oracle': values for a in aliases}, index=local_index)
    context['fr_nuclear_generation_fcst_gw'] = values
    # Deliberately different raw alias demonstrates exact known-column usage.
    context['de_wind_generation_fcst'] = values + 1e-10
    before = context.copy(deep=True)
    captured = []
    def builder(frame, *, zone, timezone):
        captured.append(frame.copy(deep=True))
        assert str(frame.index.tz) == 'UTC'
        return pd.DataFrame({'stress': .5}, index=frame.index), {'status': 'synthetic'}
    feature, audit, hybrid = m.interaction_feature(SimpleNamespace(model_context_covariates=context,
        known_future_columns=[f'known_{a}_oracle' for a in aliases]),
        {}, {'zone': zone, 'delivery_day': '2026-09-24', 'source_records': {}}, tmp_path, builder)
    assert str(feature.index.tz) == str(hybrid.index.tz) == 'UTC'
    assert hybrid.index.equals(local_index.tz_convert('UTC'))
    assert hybrid.de_wind_generation_fcst.dtype == np.dtype('float64')
    np.testing.assert_array_equal(captured[0].de_wind_generation_fcst.to_numpy(), values)
    pd.testing.assert_frame_equal(context, before)
