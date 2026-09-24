"""Resumable, offline CatBoost MAE-vs-RMSE experiment on frozen nuclear Chronos.

Only the residual objective changes. This launcher never calls the neural model,
Kalman, Saturn, Forecast.ps1, or a production export writer. Prepared inputs and
completed daily predictions are immutable; live status/reports are replaceable.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from uuid import uuid4

import numpy as np
import pandas as pd
import psutil
import yaml

from chronos2_hourly.atomic_directory import AtomicDirectoryStaging
from chronos2_hourly.process_lock import exclusive_process_lock

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'runs/experiments/catboost_rmse_v1'
ZONES = ('FR', 'DE', 'BE', 'NL')
SOURCE_CODE = (
    'nyx_catboost_rmse/core.py', 'run_catboost_rmse.py',
    'chronos2_hourly/models/residual_corrector.py', 'run_chronos2_hourly.py',
    'chronos2_hourly/nuclear_preparation.py', 'chronos2_hourly/nuclear_forecast.py',
    'chronos2_hourly/features.py',
)
INPUT_FILES = ('features.parquet', 'raw_history.parquet', 'future.parquet',
               'reference.parquet', 'baseline_daily_audit.parquet', 'recipe.json', 'provenance.json')


def safe(path):
    value = Path(path).absolute()
    if (OUTPUT.resolve() != OUTPUT or value.resolve() != value
            or not value.is_relative_to(OUTPUT) or value == OUTPUT):
        raise ValueError('All RMSE writes must remain inside its unredirected experiment directory.')
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode('utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = safe(path.with_name('.' + uuid4().hex + '.tmp'))
    try:
        with temp.open('xb') as stream:
            stream.write(encoded(value)); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def settings(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
    expected = {'schema_version', 'delivery_day', 'evaluation_days', 'zones', 'source_state',
                'output_root', 'loss_function', 'max_abs_correction', 'diagnostic_only', 'production_modified'}
    if (not isinstance(cfg, dict) or set(cfg) != expected or cfg['schema_version'] != 1
            or cfg['evaluation_days'] != 365 or cfg['zones'] != list(ZONES)
            or cfg['delivery_day'] != '2026-09-19' or cfg['loss_function'] != 'RMSE'
            or cfg['max_abs_correction'] != 40.0 or cfg['diagnostic_only'] is not True
            or cfg['production_modified'] is not False or (ROOT / cfg['output_root']).resolve() != OUTPUT):
        raise ValueError('This ablation fixes the four-country 365-day scope and the current +/-40 ceiling.')
    source = (ROOT / cfg['source_state']).absolute()
    allowed_source = ROOT / 'runs/experiments/solar_correction_v1'
    if source.resolve() != source or not source.is_relative_to(allowed_source):
        raise ValueError('Expected the verified frozen nuclear reference inside SolarCorrection.')
    return cfg


def period(cfg, timezone):
    end = pd.Timestamp(cfg['delivery_day'], tz=timezone) + pd.DateOffset(days=1)
    start = end - pd.DateOffset(days=365)
    return pd.date_range(start, end, freq='h', inclusive='left').tz_convert('UTC').rename('delivery_start_utc')


def indexed(frame):
    from chronos2_hourly.nuclear_reporting import _timestamped
    return _timestamped(frame, name='RMSE frozen input')


def code_identity():
    import catboost
    import sklearn
    return {'files': {name: sha(ROOT / name) for name in SOURCE_CODE},
            'runtime': {'python': platform.python_version(), 'catboost': catboost.__version__,
                        'pandas': pd.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__}}


def experiment_root(cfg):
    return safe(OUTPUT / cfg['delivery_day'])


def verify_prepared(work):
    work = safe(work)
    record = read(work / 'inputs/manifest.json')
    if set(record.get('files', {})) != set(INPUT_FILES):
        raise ValueError('Incomplete immutable RMSE input inventory.')
    for name, value in record['files'].items():
        if sha(safe(work / 'inputs' / name)) != value:
            raise ValueError(f'RMSE input checksum mismatch: {name}')
    return record


def prepare_zone(cfg, zone, root):
    """Freeze only processed nuclear inputs; solar forecasts never enter this test."""
    import run_solar_correction as solar
    from run_nuclear_cwe_forecast import _verify_reporting
    from chronos2_modular.common import build_zone_configs
    from chronos2_hourly.nuclear_preparation import prepare_nuclear_zone_data
    from chronos2_hourly.nuclear_report_benchmark import _load_verified_snapshot
    from chronos2_hourly.nuclear_forecast import _digest_frame
    from run_chronos2_hourly import _feature_inputs

    work = safe(root / zone.lower())
    with exclusive_process_lock(safe(work / 'prepare.lock')):
        if (work / 'inputs').exists():
            record = verify_prepared(work)
            if record['zone'] != zone or record['delivery_day'] != cfg['delivery_day']:
                raise ValueError('Prepared zone/day changed.')
            return work
        state = read(ROOT / cfg['source_state'])
        source = solar.safe(state['workdirs'][zone])
        solar.verify_snapshot(source, zone=zone, day=cfg['delivery_day'])
        source_manifest = read(source / 'input_snapshot.json')
        source_manifest_sha = sha(source / 'input_snapshot.json')
        original_dir = solar.baseline(cfg['delivery_day'], zone)
        expected_recipe = source_manifest['identity']['baseline']['resolved_config.yaml']
        if sha(original_dir / 'resolved_config.yaml') != expected_recipe:
            raise ValueError('Original frozen nuclear recipe differs from its pinned fingerprint.')
        resolved = yaml.safe_load((original_dir / 'resolved_config.yaml').read_text(encoding='utf-8'))
        original_inputs = read(original_dir / 'input_snapshot.json')
        if original_inputs['resolved_config_sha256'] != expected_recipe:
            raise ValueError('Nuclear recipe/source manifest mismatch.')
        sealed = {str(Path(item['snapshot_path']).absolute()): item['sha256'] for item in source_manifest['files']}
        relocated = {}
        for entry in original_inputs['files']:
            original_path = Path(entry['snapshot']).absolute()
            candidate = source / 'snapshot' / original_path.name
            if sealed.get(str(candidate)) != entry['sha256']:
                raise ValueError('Nuclear source is not identically pinned in the reference snapshot.')
            relocated[str(original_path)] = str(candidate)
        def move_path(value):
            key = str(Path(value).absolute())
            if key not in relocated:
                raise ValueError(f'Unverified original nuclear input: {value}')
            return relocated[key]
        spec = resolved['zones'][zone]
        spec['target']['file'] = move_path(spec['target']['file'])
        for alias, cov in spec['covariates'].items():
            if cov.get('enabled', True):
                cov['pit_file'] = move_path(resolved['data']['pit_files'][alias])
                if 'file' in cov:
                    cov['file'] = cov['pit_file']
        resolved['data']['pit_files'] = {a: s['pit_file'] for a, s in spec['covariates'].items() if s.get('enabled', True)}
        if any('solar' in a for a in resolved['data']['pit_files']):
            raise ValueError('An explicit solar channel would confound this nuclear-only loss ablation.')
        resolved['data']['cache_dir'] = str(safe(work / 'prepared/cache'))
        resolved['output']['directory'] = str(work)
        for key in ('incremental_cache_dir', 'incremental_namespace'):
            resolved['nuclear_experiment'][key] = str(safe(work / 'unused_model_cache'))
        if resolved['nuclear_experiment'].get('residual_bank_audit'):
            resolved['nuclear_experiment']['residual_bank_audit'] = move_path(resolved['nuclear_experiment']['residual_bank_audit'])
        incumbent = solar.load_incumbent(source, full_history=True)
        zone_spec = build_zone_configs(resolved, [zone], None, None)[0]
        data = prepare_nuclear_zone_data(zone_spec, resolved, work, safe(work / 'prepared'))
        _, _, _, features = _feature_inputs(data, resolved)
        raw = indexed(incumbent.raw_history)
        future_original = indexed(incumbent.source_forecast)
        future = future_original[[f'chronos2__{q}' for q in ('q10', 'q50', 'q90')]].rename(
            columns={f'chronos2__{q}': q for q in ('q10', 'q50', 'q90')})
        future['forecast_origin_utc'] = future_original.forecast_origin_utc
        required = raw.index.append(future.index)
        if len(required.difference(features.index)):
            raise ValueError('Frozen residual feature population incomplete.')
        expected_hash = incumbent.audit.get('source_hashes', {}).get('residual_features')
        features_hash = _digest_frame(features)
        # This hash covers all generated features, including price columns that
        # the unchanged residual feature builder subsequently excludes.
        if expected_hash is not None and features_hash != expected_hash:
            raise ValueError('Reconstructed nuclear features differ from the original archived feature matrix.')
        observed, observed_audit = _verify_reporting(source, zone=zone, day=cfg['delivery_day'], timezone=data.timezone)
        observed.index = pd.DatetimeIndex(observed.index).tz_convert('UTC')
        evaluated = period(cfg, data.timezone)
        baseline = pd.concat([indexed(incumbent.residual_statistics), future_original])
        if not baseline.index.is_unique:
            raise ValueError('Duplicate frozen baseline predictions.')
        reference = pd.DataFrame({'actual': observed.reindex(evaluated),
            'chronos_q50': baseline.chronos2__q50.reindex(evaluated),
            'mae_q50': baseline.residual_corrected__q50.reindex(evaluated)}, index=evaluated)
        if not np.isfinite(reference.to_numpy(float)).all():
            raise ValueError('Paired 365-day baseline/observations are incomplete; no imputation is permitted.')
        storm = _load_verified_snapshot(source / 'reference/reporting', zone=zone, timezone=data.timezone)
        reference['storm_q50'] = np.nan if storm is None else storm[0].reindex(evaluated)
        recipe = {'hourly': {'residual_correction': deepcopy(resolved['hourly']['residual_correction'])}}
        archived_recipe = deepcopy(incumbent.audit['residual_recipe'])
        fresh_recipe = deepcopy(recipe['hourly']['residual_correction'])
        archived_recipe.pop('thread_count', None); fresh_recipe.pop('thread_count', None)
        if archived_recipe != fresh_recipe:
            raise ValueError('The CatBoost recipe differs from the archived nuclear corrector.')
        provenance = {'source_workdir': str(source), 'source_manifest_sha256': source_manifest_sha,
            'nuclear_config_sha256': expected_recipe, 'reconstructed_features_sha256': features_hash,
            'archived_features_sha256': expected_hash, 'feature_hash_exact_match': expected_hash == features_hash,
            'pit_selection': data.diagnostics['nuclear_pit_selection'], 'reporting_observations': observed_audit,
            'storm': None if storm is None else storm[1:],
            'baseline_recipe': incumbent.audit['residual_recipe'], 'source_inventory': source_manifest,
            'production_pit_evidence': False, 'independent_validation': False,
            'statement': 'Same historically reconstructed/frozen inputs as incumbent, not proof of contemporaneous publication.'}
        solar.verify_snapshot(source, zone=zone, day=cfg['delivery_day'])
        if sha(source / 'input_snapshot.json') != source_manifest_sha:
            raise ValueError('Frozen source changed during preparation.')
        with AtomicDirectoryStaging(work, prefix='.inputs-') as staging:
            for name, frame in {'features': features, 'raw_history': raw, 'future': future,
                                'reference': reference, 'baseline_daily_audit': incumbent.residual_daily_audit}.items():
                frame = frame.copy(deep=False); frame.attrs = {}
                frame.to_parquet(safe(staging.path / (name + '.parquet')))
            write(staging.path / 'recipe.json', recipe)
            write(staging.path / 'provenance.json', provenance)
            record = {'schema_version': 1, 'zone': zone, 'timezone': data.timezone, 'delivery_day': cfg['delivery_day'],
                'evaluation_start': str(evaluated[0].tz_convert(data.timezone).date()),
                'evaluation_end': cfg['delivery_day'], 'evaluation_days': 365, 'evaluation_hours': len(evaluated),
                'files': {name: sha(staging.path / name) for name in INPUT_FILES},
                'chronos_recomputed': False, 'production_modified': False, 'prepared_at_utc': str(pd.Timestamp.now(tz='UTC'))}
            write(staging.path / 'manifest.json', record)
            staging.publish(safe(work / 'inputs'))
        verify_prepared(work)
        print(f'[CatBoostRMSE/{zone}] entrees figees : {len(evaluated)} heures, Chronos intact, aucun solaire ajoute.', flush=True)
    return work


def load_inputs(work):
    record = verify_prepared(work)
    source = safe(work / 'inputs')
    return record, {name: pd.read_parquet(source / (name + '.parquet'))
                    for name in ('features', 'raw_history', 'future', 'reference', 'baseline_daily_audit')}, read(source / 'recipe.json')


def fit_contract(work, threads):
    return {'input_manifest_sha256': sha(work / 'inputs/manifest.json'), 'code': code_identity(),
            'threads': threads, 'loss_function': 'RMSE', 'eval_metric': 'RMSE',
            'training_days': 365, 'daily_refit': True, 'cap_eur_mwh': 40., 'kalman_recomputed': False}


def prediction_day(inputs, day, timezone):
    raw = inputs['raw_history']
    day = pd.Timestamp(day).date()
    mask = raw.index.tz_convert(timezone).date == day
    selected = raw.loc[mask] if mask.any() else inputs['future']
    return selected[['q10', 'q50', 'q90', 'forecast_origin_utc']].copy()


def expected_features(inputs, day):
    daily = inputs['baseline_daily_audit']
    selected = daily.loc[daily.delivery_day.astype(str).eq(str(day))]
    if len(selected) != 1 or selected.iloc[0]['generation_source'] != 'daily_prequential_refit':
        raise ValueError('Scored day has no fitted MAE reference schema.')
    return list(selected.iloc[0]['residual_feature_columns'])


def cached_day(folder, *, contract_sha, day, expected_index):
    folder = safe(folder)
    if not folder.exists():
        return None
    record = read(folder / 'manifest.json')
    if (record.get('contract_sha256') != contract_sha or record.get('day') != str(day)
            or set(record.get('files', {})) != {'predictions.parquet', 'audit.json'}):
        raise ValueError('Incompatible RMSE daily checkpoint; it is not silently reused or overwritten.')
    for name, value in record['files'].items():
        if sha(safe(folder / name)) != value:
            raise ValueError('Corrupt RMSE daily checkpoint; retained for inspection.')
    predicted = pd.read_parquet(folder / 'predictions.parquet')
    audit = read(folder / 'audit.json')
    if (audit.get('delivery_day') != str(day) or audit.get('loss') != 'RMSE'
            or audit.get('training_days') != 365 or audit.get('current_day_labels_used') is not False):
        raise ValueError('Invalid RMSE daily checkpoint training audit.')
    if (not predicted.index.equals(expected_index)
            or not np.isfinite(predicted[['q10', 'q50', 'q90', 'raw_correction', 'applied_correction']].to_numpy(float)).all()
            or (predicted.q10 > predicted.q50).any() or (predicted.q50 > predicted.q90).any()
            or not np.allclose(predicted.applied_correction, predicted.raw_correction.clip(-40, 40), atol=1e-8, rtol=0)):
        raise ValueError('Invalid RMSE daily checkpoint population or corrections.')
    return predicted, audit


def save_day(folder, predicted, audit, contract_sha, day):
    folder = safe(folder)
    if folder.exists():
        raise ValueError('Refusing to overwrite a completed daily fit.')
    with AtomicDirectoryStaging(folder.parent, prefix='.day-') as staging:
        saved = predicted.copy(deep=False); saved.attrs = {}
        saved.to_parquet(safe(staging.path / 'predictions.parquet'))
        write(staging.path / 'audit.json', audit)
        write(staging.path / 'manifest.json', {'contract_sha256': contract_sha, 'day': str(day),
            'files': {name: sha(staging.path / name) for name in ('predictions.parquet', 'audit.json')}})
        staging.publish(folder)


def parity(work, record, inputs, config, threads, cache, signature):
    from nyx_catboost_rmse.core import fit_day
    path = safe(cache / 'mae_parity.json')
    if path.exists():
        audit = read(path)
        if audit.get('contract_sha256') != signature or audit.get('passed') is not True:
            raise ValueError('Invalid MAE parity receipt.')
        return audit
    day = record['evaluation_start']
    print(f'[CatBoostRMSE/{record["zone"]}] controle de reproduction MAE {day}...', flush=True)
    predicted, audit = fit_day(history=inputs['raw_history'], future=prediction_day(inputs, day, record['timezone']),
        features=inputs['features'], timezone=record['timezone'], day=day, config=config, threads=threads,
        loss='MAE', expected_features=expected_features(inputs, day))
    ref = inputs['reference'].mae_q50.reindex(predicted.index)
    gap = float((predicted.q50 - ref).abs().max())
    if not np.isfinite(gap) or gap > 0.0001:
        raise ValueError(f'MAE parity failed (max difference {gap:.8f} EUR/MWh); RMSE annual run refused.')
    receipt = {'passed': True, 'contract_sha256': signature, 'day': day,
               'maximum_absolute_difference_eur_mwh': gap, 'tolerance_eur_mwh': .0001, 'audit': audit}
    write(path, receipt)
    print(f'[CatBoostRMSE/{record["zone"]}] MAE reproduite, ecart max={gap:.9f} EUR/MWh.', flush=True)
    return receipt


def paired(inputs, predicted):
    ref = inputs['reference'].reindex(predicted.index).copy()
    for q in ('q10', 'q50', 'q90'):
        ref['rmse_' + q] = predicted[q]
    for name in ('raw_correction', 'applied_correction'):
        ref[name] = predicted[name]
    if not np.isfinite(ref[['actual', 'chronos_q50', 'mae_q50', 'rmse_q50']].to_numpy(float)).all():
        raise ValueError('Labels enter evaluation only after fitting; missing labels are never filled.')
    if not np.allclose(ref.rmse_q50 - ref.chronos_q50, ref.applied_correction, atol=.0001, rtol=0):
        raise ValueError('Candidate altered frozen Chronos.')
    return ref


def report(cfg, root, threads):
    from nyx_catboost_rmse.report import write_report
    frames, timezones, states = {}, {}, {}
    for zone in ZONES:
        work = safe(root / zone.lower())
        if not (work / 'inputs/manifest.json').exists():
            continue
        record, inputs, _ = load_inputs(work)
        timezones[zone] = record['timezone']
        pointer = work / 'latest_fit.json'
        if not pointer.exists():
            states[zone] = {'completed_days': 0, 'planned_days': 365}
            continue
        committed = read(pointer)
        signature = committed['contract_sha256']
        if digest(committed['contract']) != signature or committed['contract']['input_manifest_sha256'] != sha(work / 'inputs/manifest.json'):
            raise ValueError('Invalid committed fit identity for reporting.')
        cache = safe(work / 'checkpoints' / signature[:16])
        parts = []
        for day in pd.date_range(record['evaluation_start'], record['evaluation_end'], freq='D').date:
            base = prediction_day(inputs, day, record['timezone'])
            cached = cached_day(cache / str(day), contract_sha=signature, day=day, expected_index=base.index)
            if cached is not None:
                parts.append(paired(inputs, cached[0]))
        if parts:
            frames[zone] = pd.concat(parts).sort_index()
        states[zone] = {'completed_days': len(parts), 'planned_days': 365,
                        'input_manifest_sha256': sha(work / 'inputs/manifest.json'), 'contract_sha256': signature}
    metadata = {'evaluation_start': str((pd.Timestamp(cfg['delivery_day']) - pd.Timedelta(days=364)).date()),
        'evaluation_end': cfg['delivery_day'], 'planned_zones': list(ZONES), 'timezones': timezones,
        'states': states, 'production_modified': False, 'kalman_recomputed': False,
        'chronos_recomputed': False, 'independent_validation': False,
        'created_at_utc': str(pd.Timestamp.now(tz='UTC'))}
    return write_report(frames, metadata, safe(root / 'reports'))


def run_zone(cfg, root, zone, threads, limit, smoke=False):
    from nyx_catboost_rmse.core import fit_day
    work = safe(root / zone.lower())
    with exclusive_process_lock(safe(work / 'run.lock')):
        record, inputs, config = load_inputs(work)
        contract = fit_contract(work, threads)
        signature = digest(contract)
        cache = safe(work / 'checkpoints' / signature[:16])
        cache.mkdir(parents=True, exist_ok=True)
        pointer = safe(work / 'latest_fit.json')
        if pointer.exists() and read(pointer).get('contract_sha256') != signature:
            raise ValueError('Recipe/code/runtime/threads changed: refusing to mix fits or silently restart. Use the original settings.')
        write(pointer, {'contract_sha256': signature, 'contract': contract})
        write(cache / 'contract.json', contract)
        parity(work, record, inputs, config, threads, cache, signature)
        days = pd.date_range(record['evaluation_start'], record['evaluation_end'], freq='D').date
        if smoke:
            days = days[:1]
        elif limit:
            days = days[:limit]
        start = time.monotonic()
        for ordinal, day in enumerate(days, 1):
            if code_identity() != contract['code']:
                raise ValueError('Training code changed during replay. Existing checkpoints are preserved.')
            base = prediction_day(inputs, day, record['timezone'])
            folder = safe(cache / str(day))
            cached = cached_day(folder, contract_sha=signature, day=day, expected_index=base.index)
            if cached is None:
                write(work / 'status.json', {'status': 'running', 'zone': zone, 'day': str(day),
                    'completed_in_scope': ordinal - 1, 'requested_days': len(days), 'planned_days': 365,
                    'pid': os.getpid(), 'updated_at_utc': str(pd.Timestamp.now(tz='UTC'))})
                predicted, audit = fit_day(history=inputs['raw_history'], future=base, features=inputs['features'],
                    timezone=record['timezone'], day=day, config=config, threads=threads,
                    loss='RMSE', expected_features=expected_features(inputs, day))
                paired(inputs, predicted)
                save_day(folder, predicted, audit, signature, day)
                source = 'fit'
            else:
                predicted, audit = cached; source = 'checkpoint'
            print(f'[CatBoostRMSE/{zone}] {ordinal}/{len(days)} {day} {source}, '
                  f'heures={len(predicted)}, duree_cumulee={time.monotonic()-start:.0f}s', flush=True)
            if ordinal == 1 or ordinal % 30 == 0:
                report(cfg, root, threads)
        verify_prepared(work)
        complete = len(days) == 365
        write(work / 'status.json', {'status': 'complete' if complete else 'partial', 'zone': zone,
            'completed_in_scope': len(days), 'planned_days': 365, 'pid': os.getpid(),
            'updated_at_utc': str(pd.Timestamp.now(tz='UTC')), 'production_modified': False,
            'contract_sha256': signature})
        report(cfg, root, threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config/catboost_rmse.yaml'))
    parser.add_argument('--action', choices=('prepare', 'smoke', 'run', 'status', 'report'), default='run')
    parser.add_argument('--zones', nargs='+', choices=ZONES, default=list(ZONES))
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--max-days', type=int, default=0, help='First N chronological days only; output is explicitly partial.')
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 16 or not 0 <= args.max_days <= 365 or len(set(args.zones)) != len(args.zones):
        parser.error('Invalid threads/day count or duplicate zones.')
    cfg = settings(args.config); root = experiment_root(cfg)
    if args.action == 'status':
        states = {}
        for zone in ZONES:
            path = root / zone.lower() / 'status.json'
            states[zone] = read(path) if path.exists() else {'status': 'not_started'}
            pid = states[zone].get('pid')
            if pid and states[zone]['status'] == 'running':
                states[zone]['pid_exists'] = psutil.pid_exists(pid)
        print(json.dumps({'root': str(root), 'zones': states, 'production_modified': False}, indent=2)); return 0
    if args.action == 'report':
        print(json.dumps({key: str(value) for key, value in report(cfg, root, args.threads).items()}, indent=2)); return 0
    # The new experiment is less urgent than the user's existing operational runs.
    if os.name == 'nt':
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    with exclusive_process_lock(safe(root / 'batch.lock')):
        for zone in args.zones:
            prepare_zone(cfg, zone, root)
        if args.action == 'prepare':
            print(json.dumps({'status': 'prepared', 'root': str(root), 'zones': args.zones})); return 0
        try:
            for zone in args.zones:
                run_zone(cfg, root, zone, args.threads, args.max_days, smoke=args.action == 'smoke')
        except Exception as exc:
            write(root / zone.lower() / 'status.json', {'status': 'failed', 'zone': zone, 'error': str(exc),
                'pid': os.getpid(), 'updated_at_utc': str(pd.Timestamp.now(tz='UTC'))})
            raise
    paths = report(cfg, root, args.threads)
    print(json.dumps({'status': 'complete' if args.action == 'run' and not args.max_days else 'partial',
                      'reports': {key: str(value) for key, value in paths.items()}}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
