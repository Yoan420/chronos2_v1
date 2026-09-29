"""Dated, isolated SolarWind interaction ±40 app forecast for DE and NL.

Build new inputs from versioned configuration, without an old result bundle or
Test2 hybrid. Reuse scientific adapters, PIT audits and corrector interaction.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import html
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml

from .process_lock import exclusive_process_lock
from .solar_wind_forecast import GENERATION_SERIES, solar_wind_input_protocol

ROOT = Path(__file__).resolve().parents[1]
ENGINE = 'solarwind_interaction40'
OUTPUT = ROOT / 'runs' / ENGINE
# Keep internal cache paths short enough for Windows; all stay below OUTPUT.
MODEL_ROOT = OUTPUT
MODEL_OUTPUT = MODEL_ROOT / 'runs/experiments/solar_wind_v1'
GENERATION_ROOT = ROOT / 'data/pit/solar_wind_v1' / ENGINE
NUCLEAR_ROOT = ROOT / 'data/pit/nuclear_forecast' / ENGINE
ZONES = ('DE', 'NL')
IMPLEMENTATIONS = ('run_solarwind_forecast.py', 'chronos2_hourly/solarwind_live.py',
    'chronos2_hourly/nyx_live_baseline.py', 'chronos2_hourly/solar_wind_interaction_features.py',
    'chronos2_hourly/nuclear_incremental.py', 'chronos2_hourly/nuclear_run_archive.py',
    'chronos2_hourly/nuclear_exports.py', 'run_nuclear_forecast.py')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe(path, *, root=None):
    root = Path(OUTPUT if root is None else root).absolute()
    path = Path(path).absolute()
    if root.resolve() != root or path.resolve() != path or path == root or not path.is_relative_to(root):
        raise ValueError('SolarWind output must remain in its dedicated unredirected directory')
    return path


def write_json(path, value):
    from .nuclear_exports import _replace_with_retry
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = safe(path.with_suffix(path.suffix + '.tmp'))
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                                   allow_nan=False, default=str) + '\n', encoding='utf-8')
    _replace_with_retry(temporary, path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def isolated_settings():
    """Copy settings; production configuration and module globals stay untouched."""
    from run_nuclear_forecast import load_settings
    settings = deepcopy(load_settings(ROOT / 'config/nuclear_forecast.yaml'))
    settings.update(output_root=MODEL_OUTPUT, computation_mode='incremental',
        nuclear_store=NUCLEAR_ROOT / 'fr_nuclear_generation_fcst_gw.parquet',
        residual_bank=NUCLEAR_ROOT / 'residual_load_market_features.parquet',
        residual_bank_seed=ROOT / 'data/pit/nuclear_forecast/residual_load_market_features.parquet')
    return settings


def validate_recipe(config):
    """Reject another recipe; do not reinterpret it as the selected interaction."""
    expected = {'enabled': True, 'base_model': 'chronos2', 'backend': 'catboost',
        'iterations': 700, 'depth': 6, 'learning_rate': .03, 'l2_leaf_reg': 15.,
        'min_samples_leaf': 30, 'min_training_rows': 720, 'random_state': 42,
        'correction_scale': 1., 'max_abs_correction': 40.}
    recipe = config['hourly']['residual_correction']
    if any(recipe.get(key) != value for key, value in expected.items()):
        raise ValueError('SolarWind interaction requires the versioned CatBoost ±40 recipe')
    if recipe.get('feature_builder', {}).get('exclude_historical_prices') is not True:
        raise ValueError('SolarWind corrector must retain historical-price exclusion')
    if config['model'].get('model_id') != 'amazon/chronos-2':
        raise ValueError('SolarWind requires amazon/chronos-2')


def _seed_pair(source, destination, record, *, root):
    """Copy a validated donor only into an empty dedicated source cache."""
    source, destination = Path(source).absolute(), safe(destination, root=root)
    audit_source, audit_destination = Path(str(source) + '.audit.json'), safe(str(destination) + '.audit.json', root=root)
    if destination.exists() or audit_destination.exists():
        return
    if source.resolve() != source or audit_source.resolve() != audit_source:
        raise ValueError('Redirected SolarWind source donor')
    expected = record['sha256']
    expected_audit = record.get('audit_sha256') or sha(audit_source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for src, dst, digest in ((source, destination, expected), (audit_source, audit_destination, expected_audit)):
        if sha(src) != digest:
            raise ValueError('Source donor changed before isolated copy')
        shutil.copy2(src, dst)
        if sha(src) != digest or sha(dst) != digest:
            raise ValueError('Source donor changed during isolated copy')


def capture_sources(settings, day, workers, *, sync):
    from run_nuclear_forecast import required_source_bounds, sync_source, ensure_residual_inputs
    from .nuclear_sources import audit_nuclear_store
    from .nuclear_residual_inputs import audit_residual_bank
    from . import solar_wind_sources as wind
    # Validate existing caches too: legacy synchronizers resolve paths before writing.
    for key in ('nuclear_store', 'residual_bank'):
        path = safe(settings[key], root=NUCLEAR_ROOT)
        safe(str(path) + '.audit.json', root=NUCLEAR_ROOT)
        safe(path.with_suffix('.sync.lock'), root=NUCLEAR_ROOT)
        safe(str(path) + '.lock', root=NUCLEAR_ROOT)
    first, last = required_source_bounds(settings, day)
    if sync:
        source = ROOT / 'data/pit/nuclear_forecast/fr_nuclear_generation_fcst_gw.parquet'
        if not settings['nuclear_store'].exists():
            if not source.is_file():
                raise ValueError('Missing audited nuclear source; prepare the usual input history first')
            audit = read(str(source) + '.audit.json')
            result = audit_nuclear_store(source, audit['start_day'], audit['end_day'])
            if not result['complete']:
                raise ValueError('Nuclear source donor failed its audit: ' + '; '.join(result['blockers']))
            _seed_pair(source, settings['nuclear_store'], {'sha256': sha(source)}, root=NUCLEAR_ROOT)
        sync_source(settings, day, workers)
        ensure_residual_inputs(settings, day, workers)
        for plan in wind._plans(GENERATION_ROOT):
            if plan.output.exists() or plan.audit_path.exists():
                continue
            donor = replace(plan, output=ROOT / 'data/pit/solar_wind_v1' / plan.output.name)
            if not donor.output.is_file():
                raise ValueError(f'Missing audited SolarWind history: {donor.output.name}')
            record = wind._inspect(donor, pd.Timestamp(first), pd.Timestamp(last),
                wind_dst_policy='duplicate', wind_gap_policy=wind.NL_SPRING_GAP_POLICY)
            _seed_pair(donor.output, plan.output, record, root=GENERATION_ROOT)
    nuclear = audit_nuclear_store(settings['nuclear_store'], first, last)
    residual = audit_residual_bank(settings['residual_bank'], first, last)
    if not nuclear['complete'] or not residual['complete']:
        raise ValueError('Incomplete isolated nuclear/residual history: ' + '; '.join(
            nuclear['blockers'] + residual['blockers']))
    return wind.ensure_solar_wind_sources(output_root=GENERATION_ROOT, start_day=first, end_day=last,
        workers=workers, sync=sync, wind_dst_policy='duplicate', wind_gap_policy=wind.NL_SPRING_GAP_POLICY)


def scientific_identity():
    from run_solar_wind_forecast import scientific_identity as source_identity
    return {'chain': source_identity(), 'app': {name: sha(ROOT / name) for name in IMPLEMENTATIONS}}


def verify_snapshot(work, identity):
    work = safe(work)
    manifest = read(work / 'input_snapshot.json')
    if manifest['identity'] != identity or identity['implementation'] != scientific_identity():
        raise ValueError('SolarWind code or sealed delivery identity changed')
    pinned = set()
    for item in manifest['files']:
        path = safe(item['snapshot'])
        if (path in pinned or not path.is_relative_to(work / 'snapshot')
                or sha(path) != item['sha256']):
            raise ValueError('SolarWind snapshot was changed or escaped its directory')
        pinned.add(path)
    resolved = work / 'resolved_config.yaml'
    if sha(resolved) != manifest['resolved_config_sha256']:
        raise ValueError('SolarWind resolved recipe was changed')
    config = yaml.safe_load(resolved.read_text(encoding='utf-8'))
    required = {Path(config['zones'][identity['zone']]['target']['file']),
                *(Path(p) for p in config['data']['pit_files'].values())}
    if config['nuclear_experiment'].get('residual_bank_audit'):
        required.add(Path(config['nuclear_experiment']['residual_bank_audit']))
    if not required.issubset(pinned):
        raise ValueError('SolarWind input is missing from the sealed snapshot')
    validate_recipe(config)
    from .solar_wind_forecast import _isolated_paths
    _isolated_paths(config, work)
    return config


def prepare(settings, zone, day, sources, *, threads, sync):
    """Prepare from current recipes and raw inputs, never from an old model run."""
    from run_nuclear_forecast import snapshot_config, refreshed_target_snapshot
    from .nuclear_incremental import prepare_incremental_settings
    if zone not in ZONES or set(sources) != set(GENERATION_SERIES):
        raise ValueError('SolarWind requires DE/NL and exactly four solar/two wind sources')
    work = safe(MODEL_OUTPUT / str(day.date()) / zone.lower())
    if (work / 'input_snapshot.json').exists():
        identity = read(work / 'input_snapshot.json')['identity']
        if identity['zone'] != zone or identity['delivery_day'] != str(day.date()):
            raise ValueError('SolarWind snapshot date/country mismatch')
        return work, verify_snapshot(work, identity), identity
    base_work = safe(OUTPUT / str(day.date()) / '_base_inputs' / zone.lower())
    target = None
    if sync and not (base_work / 'input_snapshot.json').exists():
        target = refreshed_target_snapshot(settings, zone, day, base_work)
    config = snapshot_config(settings, zone, day, base_work, target_override=target)
    validate_recipe(config)
    config = deepcopy(config)
    prior = read(base_work / 'input_snapshot.json')
    files, relocated = [], {}
    for item in prior['files']:
        source = Path(item['snapshot'])
        if not source.is_relative_to(base_work / 'snapshot') or sha(source) != item['sha256']:
            raise ValueError('Base input snapshot failed verification')
        # Preserve audit sidecar adjacency used by the source validators.
        destination = safe(work / 'snapshot' / source.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha(source) != item['sha256'] or sha(destination) != item['sha256']:
            raise ValueError('Base input changed during copy')
        relocated[str(source)] = str(destination)
        files.append({'snapshot': str(destination), 'sha256': item['sha256']})
    spec = config['zones'][zone]
    spec['target']['file'] = relocated[spec['target']['file']]
    paths = {alias: relocated[value] for alias, value in config['data']['pit_files'].items()}
    for alias, series in GENERATION_SERIES.items():
        record = sources[alias]
        for suffix, digest in (('', record['sha256']), ('.audit.json', record['audit_sha256'])):
            source = Path(str(record['path']) + suffix)
            destination = safe(work / 'snapshot' / (alias + '.parquet' + suffix))
            if source.resolve() != source.absolute() or sha(source) != digest:
                raise ValueError(f'Generation source changed before snapshot: {alias}')
            shutil.copy2(source, destination)
            if sha(source) != digest or sha(destination) != digest:
                raise ValueError(f'Generation source changed during snapshot: {alias}')
            files.append({'snapshot': str(destination), 'sha256': digest})
            if not suffix:
                paths[alias] = str(destination)
        spec['covariates'][alias] = {'enabled': True, 'source': 'pit_parquet', 'series': series,
            'include_base_context': True, 'fill_method': 'none', 'fill_limit': 0, 'minimum_coverage': .01,
            'future': {'known_future': True, 'strategies': ['oracle']}, 'unit': 'GW',
            'semantic': 'forecast_generation', 'daily_broadcast': False}
    for alias, value in spec['covariates'].items():
        value['pit_file'] = paths[alias]
        if 'file' in value:
            value['file'] = paths[alias]
    config['data'].update(project_root=str(MODEL_ROOT), pit_files=paths,
        pit_vintage_dir=str(work / 'snapshot'), cache_dir=str(work / 'snapshot/cache'))
    config['output']['directory'] = str(work)
    config['hourly']['residual_correction']['thread_count'] = threads
    columns = config['hourly']['feature_engineering'].get('covariate_columns')
    if columns is not None:
        columns.extend(f'known_{a}_oracle' for a in GENERATION_SERIES if f'known_{a}_oracle' not in columns)
    experiment = config['nuclear_experiment']
    for field in ('history_anchor_day', 'raw_history_start_day', 'incremental_namespace'):
        experiment.pop(field, None)
    experiment.update(input_protocol=solar_wind_input_protocol(), mode='incremental',
        incremental_cache_dir=str(MODEL_OUTPUT / '_daily_cache' / zone.lower() / 'i40'),
        candidate_model=ENGINE, production_modified=False,
        interaction_protocol={'builder_sha256': sha(ROOT / 'chronos2_hourly/solar_wind_interaction_features.py'),
                              'corrector_sha256': sha(ROOT / 'chronos2_hourly/nyx_live_baseline.py')})
    if experiment.get('residual_bank_audit'):
        experiment['residual_bank_audit'] = relocated[experiment['residual_bank_audit']]
    prepare_incremental_settings(config, day.date())
    identity = {'engine': ENGINE, 'zone': zone, 'delivery_day': str(day.date()),
        'implementation': scientific_identity(), 'source_records': sources,
        'base_snapshot_sha256': sha(base_work / 'input_snapshot.json'),
        'recipe_origin': 'versioned_main_configuration', 'interaction_location': 'corrector_only',
        'correction_clip': [-40., 40.], 'production_modified': False, 'production_pit_evidence': False}
    identity = json.loads(json.dumps(identity, sort_keys=True, default=str, allow_nan=False))
    path = safe(work / 'resolved_config.yaml')
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
    write_json(work / 'input_snapshot.json', {'identity': identity, 'files': files,
        'resolved_config_sha256': sha(path)})
    return work, verify_snapshot(work, identity), identity


def run_zone(work, config, identity, *, threads, workers, device):
    from chronos2_modular.common import build_zone_configs
    from .nuclear_preparation import prepare_nuclear_zone_data
    from .nuclear_run_archive import load_nuclear_result_bundle, save_nuclear_result_bundle
    from .nyx_live_baseline import interaction_feature, make_residual_factory
    from .solar_wind_interaction_features import build_interaction
    from .solar_wind_forecast import run_solar_wind_forecast
    from run_nuclear_forecast import run_forecast_with_storage_retry, run_progress
    zone, day = identity['zone'], identity['delivery_day']
    with exclusive_process_lock(work / 'run.lock'), run_progress(work, zone, pd.Timestamp(day), 'run') as progress:
        if (work / 'report_only/frozen_result/manifest.json').exists():
            result = load_nuclear_result_bundle(workdir=work)
            if result.audit.get('solarwind_app_identity') != identity:
                raise ValueError('Frozen SolarWind result belongs to another delivery/recipe')
            return result
        progress('prepare_inputs')
        spec = build_zone_configs(config, [zone], None, None)[0]
        data = prepare_nuclear_zone_data(spec, config, work, safe(work / 'prepared'))
        score, feature_audit, _ = interaction_feature(data, config, identity, work, build_interaction)
        factory = make_residual_factory(config, score)
        write_json(work / 'interaction_audit.json', feature_audit)
        progress('forecast')
        result = run_forecast_with_storage_retry(run_solar_wind_forecast, config=config, data=data,
            zone=zone, delivery_day=day, workdir=work, threads=threads, workers=workers,
            device=device, residual_factory=factory)
        fitted = result.residual_daily_audit.loc[
            result.residual_daily_audit.generation_source.eq('daily_prequential_refit'), 'residual_feature_columns']
        if str(score.columns[0]) in result.covariates or fitted.empty or any(str(score.columns[0]) not in row for row in fitted):
            raise ValueError('Interaction must enter every fitted corrector and remain outside Kalman')
        verify_snapshot(work, identity)
        result = replace(result, audit=dict(result.audit, solarwind_app_identity=identity,
            solarwind_interaction_audit=feature_audit, candidate_variant=ENGINE))
        save_nuclear_result_bundle(result, workdir=work)
        return load_nuclear_result_bundle(workdir=work)


def publish(directory, results):
    """Dedicated artifacts, atomically replaced; the caller seals manifest last."""
    from .nuclear_exports import _replace_with_retry
    files, sections = {}, []
    for zone, result in results.items():
        frame = result.kalman_view.forecast.copy()
        columns = ['residual_kalman__q10', 'residual_kalman__q50', 'residual_kalman__q90']
        if any(column not in frame for column in columns):
            raise ValueError('SolarWind final Kalman quantiles are missing')
        values = frame[columns].to_numpy(float)
        if not len(frame) or not np.isfinite(values).all() or (np.diff(values, axis=1) < 0).any():
            raise ValueError('SolarWind final forecast is incomplete or has crossing quantiles')
        if 'actual' in frame and frame.actual.notna().any():
            raise ValueError('Future observations must not enter forecast exports')
        timestamp = pd.DatetimeIndex(pd.to_datetime(frame['timestamp'] if 'timestamp' in frame else frame.index, utc=True))
        timezone = 'Europe/Berlin' if zone == 'DE' else 'Europe/Amsterdam'
        day = pd.Timestamp(directory.name)
        expected = pd.date_range(day.tz_localize(timezone), (day + pd.Timedelta(days=1)).tz_localize(timezone),
                                 freq='h', inclusive='left').tz_convert('UTC')
        if not timestamp.equals(expected):
            raise ValueError('SolarWind forecast must contain exactly the physical hours of delivery day')
        exported = pd.DataFrame({'timestamp': timestamp, 'p10_eur_mwh': values[:, 0],
            'p50_eur_mwh': values[:, 1], 'p90_eur_mwh': values[:, 2]})
        path = safe(directory / f'forecast_{zone.lower()}.csv')
        temporary = safe(path.with_suffix('.csv.tmp'))
        exported.to_csv(temporary, index=False)
        _replace_with_retry(temporary, path)
        files[path.name] = sha(path)
        sections.append('<h2>' + zone + '</h2>'
                        + exported.to_html(index=False, escape=True, float_format=lambda x: f'{x:.2f}'))
    report = safe(directory / 'index.html')
    temporary = safe(report.with_suffix('.html.tmp'))
    temporary.write_text('<!doctype html><html lang="fr"><meta charset="utf-8">'
        '<title>NYX · SolarWind interaction ±40</title><style>body{font:16px system-ui;max-width:1100px;'
        'margin:40px auto;padding:20px;background:#111827;color:#e5e7eb}a{color:#7dd3fc}'
        'table{border-collapse:collapse;width:100%}th,td{padding:8px;border-bottom:1px solid #374151}</style>'
        '<h1>SolarWind interaction ±40</h1><p>Livraison : ' + html.escape(directory.name)
        + '. Chronos-2, CatBoost avec interaction plafonnée à ±40 €/MWh, puis Kalman.</p>'
        '<p>Solaire FR/DE/BE/NL et éolien DE/NL explicites. Prévision optionnelle isolée ; '
        'les résultats du lancement habituel restent inchangés. Sources interrogées as-of J−1 08 h ; '
        'publication originale PIT non certifiée. Les substitutions NL historiques auditées sont conservées.</p>'
        + ''.join(sections) + '</html>', encoding='utf-8')
    _replace_with_retry(temporary, report)
    files[report.name] = sha(report)
    return files


def completed_delivery(directory, zones):
    path = directory / 'manifest.json'
    if not path.exists():
        return None
    receipt = read(safe(path))
    expected = {'index.html', *(f'forecast_{zone.lower()}.csv' for zone in zones)}
    if (receipt.get('status') != 'COMPLETE' or receipt.get('model') != ENGINE
            or receipt.get('delivery_day') != directory.name or set(receipt.get('files', {})) != expected
            or any(sha(safe(directory / name)) != digest for name, digest in receipt['files'].items())):
        raise ValueError('Existing SolarWind publication differs or failed checksum verification')
    return receipt


def run(delivery_day, *, zones=ZONES, threads=2, workers=2, device='auto', sync=True):
    from run_nuclear_forecast import delivery_date, check_lora_inactive, zone_inputs, resolve_local_model_revision
    if (len(zones) != 2 or set(zones) != set(ZONES)
            or not 1 <= threads <= 32 or workers not in (1, 2) or device not in ('auto', 'cpu', 'cuda')):
        raise ValueError('SolarWind app requires both DE and NL, 1-32 threads and 1-2 workers')
    zones = ZONES
    day = delivery_date(delivery_day)
    directory = safe(OUTPUT / str(day.date()))
    with exclusive_process_lock(safe(OUTPUT / 'batch.lock')):
        completed = completed_delivery(directory, zones)
        if completed is not None:
            return completed
        def status(state, stage, **extra):
            write_json(directory / 'run_status.json', {'status': state, 'stage': stage,
                'delivery_day': str(day.date()), 'production_modified': False, **extra})
            print(f'[SolarWind interaction ±40] {state}: {stage}', flush=True)
        try:
            status('RUNNING', 'preflight')
            settings = isolated_settings()
            check_lora_inactive(settings, list(zones))
            for zone in zones:
                config, _, _ = zone_inputs(settings, zone)
                validate_recipe(config)
                try:
                    resolve_local_model_revision(config)
                except OSError as exc:
                    raise ValueError('Chronos-2 local weights are unavailable; install the model before using SolarWind') from exc
            status('RUNNING', 'sources')
            sources = capture_sources(settings, day, workers, sync=sync)
            results, workdirs = {}, {}
            for zone in zones:
                status('RUNNING', 'prepare', active_zone=zone)
                work, config, identity = prepare(settings, zone, day, sources, threads=threads, sync=sync)
                workdirs[zone] = str(work)
                status('RUNNING', 'forecast', active_zone=zone, workdirs=workdirs)
                results[zone] = run_zone(work, config, identity, threads=threads, workers=workers, device=device)
            status('RUNNING', 'publish')
            files = publish(directory, results)
            receipt = {'model': ENGINE, 'status': 'COMPLETE', 'delivery_day': str(day.date()),
                'zones': list(zones), 'files': files, 'workdirs': workdirs,
                'report': 'index.html', 'production_modified': False}
            write_json(directory / 'manifest.json', receipt)
            status('COMPLETE', 'complete', workdirs=workdirs)
            return receipt
        except Exception as exc:
            status('FAILED', 'failed', error=f'{type(exc).__name__}: {exc}')
            raise
