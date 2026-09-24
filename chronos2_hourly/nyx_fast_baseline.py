"""Isolated accelerated Test2 execution, with unchanged numerical model recipe.

No source synchronizer or production writer is used. The residual adapter is
scoped to this isolated process, and restores its callable on exit. A nested project root satisfies the unchanged SolarWind path
contract while keeping every newly written artifact under this namespace.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import yaml

from .process_lock import exclusive_process_lock
from .models.residual_corrector import ResidualCorrector
from .solar_wind_forecast import GENERATION_SERIES, solar_wind_input_protocol
from .solar_wind_interaction_features import build_interaction

ROOT = Path(__file__).resolve().parents[1]
ENGINE = 'nyx_test2_fast_v1'
OUTPUT = ROOT / 'runs/experiments/n2'
RESIDUAL_THREADS = 2
RESIDUAL_WORKERS = 4
MIN_FREE_MEMORY_GIB = 3.5


def memory_bound_kalman(**kwargs):
    """Keep the unchanged Kalman recipe, bound only independent CPU workers."""
    import psutil
    from joblib import parallel_config
    from threadpoolctl import threadpool_limits
    from .kalman_residual import build_operational_kalman_view
    while psutil.virtual_memory().available / 1024**3 < MIN_FREE_MEMORY_GIB + 1.0:
        print('[NYX fast] waiting_memory_before_kalman', flush=True)
        time.sleep(20)
    available = psutil.virtual_memory().available / 1024**3
    selected = max(1, min(int(kwargs['rolling_refit_workers']),
                          int((available - MIN_FREE_MEMORY_GIB) / 1.0)))
    kwargs = dict(kwargs, rolling_refit_workers=selected)
    print(f'[NYX fast] kalman_workers={selected}, free_gib={available:.2f}', flush=True)
    with threadpool_limits(limits=1), parallel_config(backend='loky', inner_max_num_threads=1):
        return build_operational_kalman_view(**kwargs)
TIMEZONES = {'DE': 'Europe/Berlin', 'NL': 'Europe/Amsterdam',
             'BE': 'Europe/Brussels', 'FR': 'Europe/Paris'}
EXTRA_SERIES = {f'{z}_wind_generation_fcst': f'power.{z}.generation.wind.hourly.gw.fcst'
                for z in ('be', 'fr')}
REFERENCE_RUNS = {'DE': '45ec8314c37a2fe5', 'NL': 'b1da388bb4df6c63'}
IMPLEMENTATION = ('chronos2_hourly/nyx_fast_baseline.py', 'chronos2_hourly/nyx_live_parallel.py',
    'chronos2_hourly/solar_wind_interaction_features.py',
    'chronos2_hourly/nuclear_daily_cache.py', 'chronos2_hourly/nuclear_residual_cache.py',
    'chronos2_hourly/nuclear_run_archive.py', 'chronos2_hourly/nuclear_exports.py',
    'chronos2_hourly/nuclear_cwe_forecast.py', 'chronos2_modular/common.py',
    'chronos2_hourly/hourly_contract.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      default=str).encode('utf-8')


def safe_path(path, *, root=None):
    root, path = Path(OUTPUT if root is None else root).absolute(), Path(path).absolute()
    if (root.resolve() != root or not root.is_relative_to(OUTPUT.absolute())
            or path.resolve() != path or not path.is_relative_to(root) or path == root):
        raise ValueError('All writes must be real children of the isolated NYX namespace')
    if path.exists() and path.is_dir():
        for child in path.rglob('*'):
            if child.absolute() != child.resolve():
                raise ValueError('Redirected artifact in isolated NYX tree')
    return path


def _replace(source, destination):
    for attempt in range(8):
        try:
            Path(source).replace(destination)
            return
        except PermissionError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == 7:
                raise
            time.sleep(min(.05 * 2 ** attempt, .5))


def write_json(path, value):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = safe_path(path.with_suffix(path.suffix + '.tmp'))
    temp.write_bytes(canonical(value) + b'\n')
    _replace(temp, path)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _day(value):
    day = pd.Timestamp(value)
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError('An explicit civil delivery day is required')
    return day.date().isoformat()


def _copy(source, destination, digest):
    source, destination = Path(source).absolute(), safe_path(destination)
    if source.resolve() != source or sha(source) != digest:
        raise ValueError(f'Source changed or redirected: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = safe_path(destination.with_suffix(destination.suffix + '.copy.tmp'))
        with source.open('rb') as src, temporary.open('wb') as dst:
            shutil.copyfileobj(src, dst)
        if sha(temporary) != digest:
            raise ValueError('Incomplete immutable source copy')
        _replace(temporary, destination)
    if sha(source) != digest or sha(destination) != digest:
        raise ValueError(f'Immutable copy mismatch: {destination}')
    return {'source': str(source), 'source_path': str(source),
            'snapshot': str(destination), 'snapshot_path': str(destination), 'sha256': digest}


def _builder_identity(builder):
    path = inspect.getsourcefile(builder)
    if path is None or '<locals>' in builder.__qualname__:
        raise ValueError('Interaction builder must be an explicit file-backed function')
    return {'module': builder.__module__, 'name': builder.__qualname__,
            'path': str(Path(path).resolve()), 'sha256': sha(path)}


def verify_reference_recipe(config, zone):
    """DE/NL retain the selected Test2 recipe, apart from explicit CPU threads."""
    if zone not in REFERENCE_RUNS:
        return {'status': 'separately_approved_country_extension'}
    from .nuclear_forecast import _kalman_filter_configuration
    directory = ROOT / 'runs/experiments/solar_wind_v1/2026-09-22' / zone.lower() / REFERENCE_RUNS[zone]
    manifest_path = directory / 'report_only/frozen_result/manifest.json'
    audits_path = directory / 'report_only/frozen_result/audits.json'
    manifest = _read(manifest_path)
    # The source archive's checksum schema is checked before using the recipe.
    expected = manifest['files']['audits.json']
    if sha(audits_path) != expected:
        raise ValueError('Selected Test2 reference audit changed')
    audit = _read(audits_path)['result']
    wanted, actual = deepcopy(audit['residual_recipe']), deepcopy(config['hourly']['residual_correction'])
    wanted.pop('thread_count', None); actual.pop('thread_count', None)
    if canonical(wanted) != canonical(actual):
        raise ValueError('Donor corrector recipe differs from selected DE/NL Test2 reference')
    _, parameters = _kalman_filter_configuration(config)
    if canonical(parameters) != canonical(audit['kalman_filter_parameters']):
        raise ValueError('Donor Kalman parameters differ from selected DE/NL Test2 reference')
    return {'status': 'same_except_explicit_thread_count', 'workdir': str(directory),
            'manifest_sha256': sha(manifest_path), 'audits_sha256': sha(audits_path)}


def source_records(sources, *, zone, day, raw_start, root=None):
    """Verify already captured sources. Never collect or rewrite a source."""
    expected = dict(GENERATION_SERIES)
    if zone in ('BE', 'FR'):
        expected[f'{zone.lower()}_wind_generation_fcst'] = EXTRA_SERIES[f'{zone.lower()}_wind_generation_fcst']
    if not set(expected).issubset(sources) or set(sources) - set(GENERATION_SERIES) - set(EXTRA_SERIES):
        raise ValueError('Exactly the approved generation sources are accepted')
    records = {}
    selected = {**GENERATION_SERIES, **{a: s for a, s in EXTRA_SERIES.items() if a in sources}}
    for alias, series in selected.items():
        rec = sources[alias]
        path = safe_path(rec['path'], root=root)
        audit_path = safe_path(str(path) + '.audit.json', root=root)
        if sha(path) != rec['sha256'] or sha(audit_path) != rec['audit_sha256']:
            raise ValueError(f'Captured source SHA differs: {alias}')
        audit = _read(audit_path)
        if (audit.get('alias') != alias or audit.get('series') != series
                or audit.get('sha256', '').lower() != rec['sha256'].lower()
                or _day(audit['start_day']) > raw_start or _day(audit['end_day']) < day
                or audit.get('daily_broadcast') is not False):
            raise ValueError(f'Wrong or incomplete captured source contract: {alias}')
        records[alias] = {'path': str(path), 'sha256': rec['sha256'].lower(),
            'audit_path': str(audit_path), 'audit_sha256': rec['audit_sha256'].lower(), 'series': series}
    return records


def _inputs(zone, day, nuclear_workdir, sources, workroot, threads, workers, device, builder):
    from run_solar_wind_forecast import scientific_identity
    from .nuclear_run_archive import load_nuclear_result_bundle
    if zone not in TIMEZONES or threads not in (1, 2, 4, 8, 16) or workers not in (1, 2, 3, 4) or device not in ('cpu', 'auto', 'cuda'):
        raise ValueError('CWE, supported CPU profile and one-to-four workers required')
    root = Path(workroot).absolute()
    if root.resolve() != root or not root.is_relative_to(OUTPUT):
        raise ValueError('Invalid isolated workroot')
    donor = Path(nuclear_workdir).absolute()
    if donor.resolve() != donor or not donor.is_relative_to(ROOT / 'runs/experiments'):
        raise ValueError('Nuclear donor must be a real experimental directory')
    bundle = load_nuclear_result_bundle(workdir=donor)
    if bundle.audit['zone'] != zone or bundle.audit['delivery_day'] != day:
        raise ValueError('Completed donor zone/day mismatch')
    prior = _read(donor / 'input_snapshot.json')
    config = yaml.safe_load((donor / 'resolved_config.yaml').read_text(encoding='utf-8'))
    if sha(donor / 'resolved_config.yaml') != prior['resolved_config_sha256']:
        raise ValueError('Donor resolved configuration changed')
    raw_start = _day(bundle.audit['raw_history_start_day'])
    records = source_records(sources, zone=zone, day=day, raw_start=raw_start, root=root)
    recipe = config['hourly']['residual_correction']
    if (recipe.get('backend') != 'catboost' or recipe.get('max_abs_correction') != 40.0
            or recipe.get('correction_scale') != 1.0 or recipe.get('min_training_rows') != 720
            or recipe.get('iterations') != 700 or recipe.get('depth') != 6):
        raise ValueError('Test2 requires the unchanged MAE CatBoost ±40 recipe')
    scientific = scientific_identity()
    reference_recipe = verify_reference_recipe(config, zone)
    identity = {'engine': ENGINE, 'zone': zone, 'delivery_day': day, 'variant': 'interaction_40',
        'source_records': records, 'nuclear_workdir': str(donor),
        'nuclear_snapshot_sha256': sha(donor / 'input_snapshot.json'),
        'nuclear_manifest_sha256': sha(donor / 'report_only/frozen_result/manifest.json'),
        'nuclear_resolved_sha256': prior['resolved_config_sha256'],
        'scientific_identity': scientific, 'implementation': {p: sha(ROOT / p) for p in IMPLEMENTATION},
        'interaction_builder': _builder_identity(builder),
        'dependencies': {p: importlib.metadata.version(p) for p in ('pykalman', 'scipy', 'joblib', 'holidays', 'scikit-learn')},
        'input_protocol': solar_wind_input_protocol(), 'history_anchor_day': bundle.audit['history_anchor_day'],
        'reference_recipe': reference_recipe,
        'raw_history_start_day': raw_start, 'threads': threads, 'workers': workers, 'device': device,
        'residual_threads': RESIDUAL_THREADS, 'residual_workers_max': RESIDUAL_WORKERS,
        'memory_reserve_gib': MIN_FREE_MEMORY_GIB, 'execution_only_acceleration': True,
        'workroot': str(root), 'correction_clip': [-40., 40.], 'interaction_location': 'corrector_only',
        'chronos_generation_aliases': list(GENERATION_SERIES), 'kalman_generation_aliases': list(GENERATION_SERIES),
        'production_modified': False, 'prospective_validation': False, 'production_pit_evidence': False}
    identity['run_id'] = hashlib.sha256(canonical(identity)).hexdigest()[:16]
    mini = root / 'm' / identity['run_id']
    work = safe_path(mini / 'runs/experiments/solar_wind_v1/r')
    longest_daily = mini / 'runs/experiments/solar_wind_v1/c/epochs' / ('a' * 32) / 'chronos/2026-09-24' / ('a' * 64 + '.json')
    if len(str(longest_daily)) >= 240:
        raise ValueError('Daily cache path too long for the portable Windows contract')
    return identity, work, mini, donor, bundle, prior, config


def verify_snapshot(work, identity):
    work = safe_path(work)
    manifest = _read(work / 'input_snapshot.json')
    if manifest['identity'] != identity:
        raise ValueError('Snapshot identity changed; never resume another experiment')
    seen = set()
    for item in manifest['files']:
        path = safe_path(item['snapshot'])
        if path in seen or not path.is_relative_to(work / 'snapshot') or sha(path) != item['sha256']:
            raise ValueError('Snapshot file changed, duplicated or escaped')
        seen.add(path)
    config_path = work / 'resolved_config.yaml'
    if sha(config_path) != manifest['resolved_config_sha256']:
        raise ValueError('Resolved snapshot recipe changed')
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    zone = identity['zone']
    needed = {Path(config['zones'][zone]['target']['file']),
              *(Path(v) for v in config['data']['pit_files'].values())}
    if config['nuclear_experiment'].get('residual_bank_audit'):
        needed.add(Path(config['nuclear_experiment']['residual_bank_audit']))
    if not needed.issubset(seen):
        raise ValueError('A configured model input is outside the immutable snapshot')
    mini = Path(config['data']['project_root'])
    safe_path(mini)
    if not work.is_relative_to(mini / 'runs/experiments/solar_wind_v1'):
        raise ValueError('Resolved project/output isolation mismatch')
    return config


def _prepare(identity, work, mini, donor, bundle, prior, config):
    from .nuclear_incremental import prepare_incremental_settings
    if (work / 'input_snapshot.json').exists():
        return verify_snapshot(work, identity)
    config = deepcopy(config)
    zone, day = identity['zone'], identity['delivery_day']
    mapping = {'target': config['zones'][zone]['target']['file'], **config['data']['pit_files']}
    lookup = {str(Path(p).resolve()): alias for alias, p in mapping.items()}
    files, paths, relocations = [], {}, {}
    for number, item in enumerate(prior['files']):
        source = Path(item['snapshot']).absolute()
        if source.resolve() != source or not source.is_relative_to(donor / 'snapshot'):
            raise ValueError('Donor input escaped sealed snapshot')
        record = _copy(source, work / 'snapshot' / f'nuclear_{number}_{source.name}', item['sha256'])
        files.append(record); relocations[str(source)] = record['snapshot']
        if str(source) in lookup:
            paths[lookup[str(source)]] = record['snapshot']
    if set(paths) != set(mapping):
        raise ValueError('Donor snapshot does not contain all selected model inputs')
    for alias, rec in identity['source_records'].items():
        for key, suffix in (('path', '.parquet'), ('audit_path', '.parquet.audit.json')):
            digest = rec['sha256' if key == 'path' else 'audit_sha256']
            record = _copy(rec[key], work / 'snapshot' / (alias + suffix), digest)
            files.append(record)
            if key == 'path':
                paths[alias] = record['snapshot']
    spec = config['zones'][zone]
    config['zones'] = {zone: spec}
    spec['target']['file'] = paths['target']
    for alias, series in GENERATION_SERIES.items():
        spec['covariates'][alias] = {'enabled': True, 'source': 'pit_parquet', 'series': series,
            'include_base_context': True, 'fill_method': 'none', 'fill_limit': 0, 'minimum_coverage': .01,
            'future': {'known_future': True, 'strategies': ['oracle']}, 'unit': 'GW',
            'semantic': 'forecast_generation', 'daily_broadcast': False}
    for alias, value in spec['covariates'].items():
        if value.get('enabled', True):
            value['pit_file'] = paths[alias]
            if 'file' in value:
                value['file'] = paths[alias]
    config['data'].update(project_root=str(mini),
        pit_files={a: paths[a] for a, v in spec['covariates'].items() if v.get('enabled', True)},
        pit_vintage_dir=str(work / 'snapshot'), cache_dir=str(work / 'snapshot/cache'))
    config['output']['directory'] = str(work)
    columns = config['hourly']['feature_engineering'].get('covariate_columns')
    if columns is not None:
        columns.extend(f'known_{a}_oracle' for a in GENERATION_SERIES if f'known_{a}_oracle' not in columns)
    config['hourly']['residual_correction']['thread_count'] = identity['residual_threads']
    experiment = config['nuclear_experiment']
    for key in ('history_anchor_day', 'raw_history_start_day', 'incremental_namespace'):
        experiment.pop(key, None)
    experiment.update(input_protocol=solar_wind_input_protocol(), mode='incremental',
        incremental_cache_dir=str(mini / 'runs/experiments/solar_wind_v1/c'),
        candidate_model='solar_wind_interaction_40', production_modified=False,
        interaction_protocol=identity['interaction_builder'], interaction_source_sha256={
            a: r['sha256'] for a, r in identity['source_records'].items()})
    if experiment.get('residual_bank_audit'):
        experiment['residual_bank_audit'] = relocations[str(Path(experiment['residual_bank_audit']).resolve())]
    anchor = pd.Timestamp(identity['history_anchor_day']).date()
    prepare_incremental_settings(config, anchor + timedelta(days=730))
    _, selected, _ = prepare_incremental_settings(config, day)
    if str(selected) != identity['history_anchor_day']:
        raise ValueError('Isolated epoch changed the reference cold-start anchor')
    resolved = safe_path(work / 'resolved_config.yaml')
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temp = safe_path(resolved.with_suffix('.yaml.tmp'))
    temp.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
    _replace(temp, resolved)
    write_json(work / 'input_snapshot.json', {'schema_version': 1, 'identity': identity,
        'files': files, 'reference_files': [], 'resolved_config_sha256': sha(resolved)})
    return verify_snapshot(work, identity)


def interaction_feature(data, config, identity, work, builder):
    """The score sees only PIT forecasts; additional BE/FR wind stays outside models."""
    from .nuclear_preparation import _strict_selection, _require_complete
    from .nuclear_cwe_forecast import _frame
    from .nuclear_forecast import _covariates
    from chronos2_modular.common import parse_series_spec
    zone, day = identity['zone'], pd.Timestamp(identity['delivery_day']).date()
    tz = TIMEZONES[zone]
    index = pd.date_range(pd.Timestamp(day - timedelta(days=730), tz=tz),
        pd.Timestamp(day + timedelta(days=1), tz=tz), freq='h', inclusive='left').tz_convert('UTC')
    # Label-based selection against a UTC index does not convert the source
    # DatetimeIndex timezone. Use the same non-numeric UTC normalization as
    # the existing SolarWind input adapter; values are not cast or rounded.
    model_context = _frame(data.model_context_covariates, 'Test2 model context')
    # Reconstruct exactly the covariate frame on which the selected Test2
    # interaction was calibrated: nuclear/RL known forecasts, then the six
    # known generation columns as in _SolarWindKalmanBuilder. Raw aliases are
    # not a substitute even when an upstream guard accepts 1e-9 differences.
    context = _frame(_covariates(data, index), 'Test2 nuclear/RL context')
    for alias in GENERATION_SERIES:
        known = f'known_{alias}_oracle'
        if known not in model_context:
            raise ValueError(f'Missing exact known-future interaction input: {known}')
        context[alias] = model_context[known].reindex(index)
    additional_audit = {}
    for alias in EXTRA_SERIES:
        if alias not in identity['source_records']:
            continue
        record = identity['source_records'][alias]
        path = safe_path(work / 'snapshot' / f'{alias}.parquet')
        spec = parse_series_spec(alias, {'source': 'pit_parquet', 'series': record['series'],
            'fill_method': 'none', 'future': {'known_future': True, 'strategies': ['oracle']}}, 0)
        selected, alias_audit = _strict_selection(pd.read_parquet(path), spec,
            timezone=tz, runtime_as_of=config['data']['runtime_as_of'])
        _require_complete(selected, index, alias=alias, timezone=tz)
        context[alias] = selected.loc[index]
        additional_audit[alias] = alias_audit
    feature, audit = builder(context, zone=zone, timezone=tz)
    if (not feature.index.equals(index) or feature.shape[1] != 1 or feature.columns.has_duplicates
            or not np.isfinite(feature.to_numpy(float)).all()
            or ((feature.to_numpy(float) < 0) | (feature.to_numpy(float) > 1)).any()):
        raise ValueError('Interaction builder violated the one-score [0,1] contract')
    return feature, dict(audit, feature_location='corrector_only', kalman_covariates_unchanged=True,
        additional_wind_selection=additional_audit, pre_calibration_prefix_policy='zero_no_available_calibration'), context


def add_feature(features, score):
    if (score.empty or features.empty or features.index.has_duplicates or not features.index.is_monotonic_increasing
            or features.columns.has_duplicates or len(score.columns) != 1
            or score.index.has_duplicates or not score.index.is_monotonic_increasing
            or score.index.tz is None or features.index.tz is None):
        raise ValueError('Unique ordered timezone-aware feature grids required')
    name = str(score.columns[0])
    if name in features:
        raise ValueError('Interaction injected twice')
    missing = features.index.difference(score.index)
    if len(missing) and (missing >= score.index[0]).any():
        raise ValueError('Missing interaction hour inside calibration/forecast support')
    values = score.reindex(features.index).copy()
    values.loc[values.index < score.index[0], name] = 0.
    array = values.to_numpy(float)
    if not np.isfinite(array).all() or ((array < 0) | (array > 1)).any():
        raise ValueError('Finite bounded interaction required')
    result = features.copy()
    result[name] = values[name]
    return result


def make_residual_factory(config, score):
    """File-backed factory whose persistent cache identity pins exact score bytes."""
    from .nuclear_forecast import _digest_frame
    recipe = deepcopy(config['hourly']['residual_correction'])
    if not recipe.pop('enabled', False) or recipe.pop('base_model', None) != 'chronos2':
        raise ValueError('Enabled Chronos residual corrector required')
    builder_options = recipe.pop('feature_builder', {})
    if recipe.get('max_abs_correction') != 40. or recipe.get('correction_scale') != 1.:
        raise ValueError('Test2 correction must remain ±40')
    score = score.copy(deep=True)
    digest = hashlib.sha256(canonical({'recipe': recipe, 'builder': builder_options,
                                     'score': _digest_frame(score)})).hexdigest()

    class InteractionCorrector(ResidualCorrector):
        def __init__(self):
            super().__init__(feature_builder_options=deepcopy(builder_options), **deepcopy(recipe))

        def fit(self, X, y, base_predictions, expert_predictions=None):
            result = super().fit(add_feature(X, score), y, base_predictions, expert_predictions)
            if str(score.columns[0]) not in self.feature_columns_:
                raise ValueError('Corrector dropped the interaction')
            return result

        def predict(self, X, base_predictions, expert_predictions=None):
            return super().predict(add_feature(X, score), base_predictions, expert_predictions)

        def predict_correction(self, X, base_predictions, expert_predictions=None):
            return super().predict_correction(add_feature(X, score), base_predictions, expert_predictions)

    # This local class, not any imported global, owns its source-backed identity.
    InteractionCorrector.__name__ = 'InteractionCorrector_' + digest
    InteractionCorrector.__qualname__ = InteractionCorrector.__name__
    return InteractionCorrector


def _outputs(work, identity):
    frozen = work / 'report_only/frozen_result'
    return {'identity': identity, 'workdir': str(work), 'result_dir': str(work),
        'history_path': str(frozen / 'kalman_backtest.parquet'),
        'forecast_path': str(frozen / 'kalman_forecast.parquet'),
        'covariates_path': str(frozen / 'covariates.parquet'),
        'hybrid_covariates_path': str(work / 'hybrid_covariates.parquet'),
        'source_forecast_path': str(frozen / 'source_forecast.parquet'),
        'residual_history_path': str(frozen / 'residual_statistics.parquet')}


def verify_hybrid_covariates(work, identity):
    receipt = _read(safe_path(work / 'hybrid_covariates_receipt.json'))
    if (receipt.get('identity') != identity['run_id']
            or receipt.get('sha256') != sha(work / 'hybrid_covariates.parquet')
            or receipt.get('interaction_audit_sha256') != sha(work / 'interaction_audit.json')):
        raise ValueError('Hybrid-only covariates or interaction audit changed')


def run_zone(zone, delivery_day, nuclear_workdir, sources, *, workroot=OUTPUT,
             threads=16, workers=4, device='cpu', action='run', interaction_builder=None, progress=None):
    """Validate read-only, prepare isolated inputs, or run/resume one approved zone.

    ``sources`` maps aliases to path/sha256/audit_sha256 records captured below
    ``workroot``; the audit sidecar is ``<path>.audit.json``. No sync is performed.
    Reports are deliberately delegated to the caller from sealed returned paths.
    """
    from chronos2_modular.common import build_zone_configs
    from .nuclear_preparation import prepare_nuclear_zone_data
    from .nuclear_run_archive import load_nuclear_result_bundle, save_nuclear_result_bundle
    from .solar_wind_forecast import run_solar_wind_forecast
    if action not in ('validate', 'prepare', 'run'):
        raise ValueError('Choose validate, prepare or run')
    zone, day = str(zone).upper(), _day(delivery_day)
    builder = interaction_builder or build_interaction
    if zone in ('BE', 'FR') and builder is build_interaction:
        raise ValueError('BE/FR require the separately approved own-country interaction builder')
    prepared = _inputs(zone, day, nuclear_workdir, sources, workroot, threads, workers, device, builder)
    identity, work, mini, donor, incumbent, prior, config = prepared
    output = _outputs(work, identity)
    if action == 'validate':
        return dict(output, status='VALIDATED', production_modified=False)
    with exclusive_process_lock(safe_path(Path(workroot) / '_batch_locks' / f'{day}_{zone}.lock')):
        status_path = work / 'status.json'
        def status(state, phase, **extra):
            write_json(status_path, {'status': state, 'phase': phase, 'zone': zone,
                'identity': identity['run_id'], 'delivery_day': day,
                'updated_utc': pd.Timestamp.now(tz='UTC').isoformat(),
                'production_modified': False, **extra})
        try:
            config = _prepare(*prepared)
            if (work / 'report_only/frozen_result/manifest.json').exists():
                result = load_nuclear_result_bundle(workdir=work)
                if result.audit.get('nyx_test2_identity') != identity:
                    raise ValueError('Frozen NYX result belongs to another identity')
                verify_hybrid_covariates(work, identity)
                status('COMPLETE', 'complete', annual_complete=True, evaluation_days=365,
                       evaluation_hours=len(result.kalman_view.backtest), future_forecast_hours=len(result.source_forecast))
                return dict(output, status='COMPLETE', result_reused=True)
            status('RUNNING', 'prepare_inputs')
            spec = build_zone_configs(config, [zone], None, None)[0]
            data = prepare_nuclear_zone_data(spec, config, work, safe_path(work / 'prepared'))
            score, score_audit, hybrid_covariates = interaction_feature(data, config, identity, work, builder)
            factory = make_residual_factory(config, score)
            write_json(work / 'interaction_audit.json', score_audit)
            hybrid_path = safe_path(work / 'hybrid_covariates.parquet')
            temporary = safe_path(hybrid_path.with_suffix('.parquet.tmp'))
            hybrid_covariates.to_parquet(temporary)
            _replace(temporary, hybrid_path)
            write_json(work / 'hybrid_covariates_receipt.json', {
                'identity': identity['run_id'], 'sha256': sha(hybrid_path),
                'interaction_audit_sha256': sha(work / 'interaction_audit.json'),
                'columns': list(hybrid_covariates), 'hours': len(hybrid_covariates),
                'role': 'report_and_hybrid_only_not_added_to_Chronos_or_Kalman'})
            if action == 'prepare':
                status('PREPARED', 'prepared')
                return dict(output, status='PREPARED')
            # Re-read all original identities before starting costly work.
            if _inputs(zone, day, nuclear_workdir, sources, workroot, threads, workers, device, builder)[0] != identity:
                raise ValueError('Source/code identity changed during preparation')
            status('RUNNING', 'chronos_corrector_kalman')
            from .nyx_live_parallel import accelerated_residual_replay
            parallel_workers = RESIDUAL_WORKERS
            def update_progress(event):
                status('RUNNING', 'parallel_residual', progress=event)
                if progress is not None:
                    progress(dict(event, zone=zone))
            with accelerated_residual_replay(workers=parallel_workers, progress=update_progress):
                result = run_solar_wind_forecast(config=config, data=data, zone=zone,
                    delivery_day=day, workdir=work, device=device, threads=threads, workers=workers,
                    residual_factory=factory, kalman_builder=memory_bound_kalman)
            if str(score.columns[0]) in result.covariates:
                raise ValueError('Interaction unexpectedly entered Kalman')
            fitted = result.residual_daily_audit.loc[
                result.residual_daily_audit.generation_source.eq('daily_prequential_refit'), 'residual_feature_columns']
            if fitted.empty or any(str(score.columns[0]) not in row for row in fitted):
                raise ValueError('A daily fitted corrector omitted the interaction')
            if _inputs(zone, day, nuclear_workdir, sources, workroot, threads, workers, device, builder)[0] != identity:
                raise ValueError('Source/code identity changed during execution')
            verify_snapshot(work, identity)
            engine_recipe = deepcopy(result.audit['residual_recipe'])
            actual_recipe = dict(engine_recipe, thread_count=RESIDUAL_THREADS)
            result = replace(result, audit=dict(result.audit, residual_recipe=actual_recipe,
                engine_nominal_residual_recipe=engine_recipe,
                execution_profile={'chronos_threads': threads, 'residual_threads': RESIDUAL_THREADS,
                    'residual_workers_max': RESIDUAL_WORKERS, 'kalman_workers_max': workers,
                    'min_free_memory_gib': MIN_FREE_MEMORY_GIB}, nyx_test2_identity=identity,
                nyx_test2_feature_audit=score_audit, nyx_test2_variant='interaction_40',
                nyx_test2_feature_location='corrector_only', production_changed=False))
            save_nuclear_result_bundle(result, workdir=work)
            verified = load_nuclear_result_bundle(workdir=work)
            status('COMPLETE', 'complete', annual_complete=True, evaluation_days=365,
                evaluation_hours=len(verified.kalman_view.backtest), future_forecast_hours=len(verified.source_forecast))
            return dict(output, status='COMPLETE', result_reused=False)
        except Exception as exc:
            status('FAILED', 'failed', error_type=type(exc).__name__, error=str(exc))
            raise
