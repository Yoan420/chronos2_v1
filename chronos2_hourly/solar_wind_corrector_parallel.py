"""Versioned parallel coordinator for the unchanged, approved corrector test.

Workers only compute; this parent alone publishes atomic checkpoints. The
original serial experiment is never rewritten, relabelled, or inferred complete.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import html
import json
import math
import multiprocessing
import os
import time

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from . import solar_wind_corrector_interaction as original
from .process_lock import exclusive_process_lock

ROOT, DAY = original.ROOT, original.DAY
ENGINE = 'solar_wind_corrector_parallel_v1'
OUTPUT = ROOT / 'runs/experiments' / ENGINE / DAY
VARIANTS = original.VARIANTS
RESULT_FILES = (*original.RESULT_FILES, 'migration_audit.json')
IMPLEMENTATION = ('run_solar_wind_corrector_parallel.py', 'chronos2_hourly/solar_wind_corrector_parallel.py')
json_bytes, sha, indexed = original.json_bytes, original.sha, original.indexed
_PREPARED = _UPSTREAMS = _LIMITER = _CONTROL_RAW = None


def _digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def safe_path(path):
    path = Path(path).absolute()
    if path.resolve() != path or not path.is_relative_to(OUTPUT.absolute()):
        raise ValueError(f'Unsafe parallel experiment output: {path}')
    if path.is_dir():
        for child in path.rglob('*'):
            if child.is_symlink() or (getattr(child.lstat(), 'st_file_attributes', 0) & 0x400):
                raise ValueError(f'Redirected parallel output descendant: {child}')
    return path


def write_json(path, value):
    path = safe_path(path)
    temporary = safe_path(path.with_suffix(path.suffix + '.tmp'))
    temporary.write_bytes(json_bytes(value)); temporary.replace(path)


def write_frame(path, frame):
    path = safe_path(path)
    temporary = safe_path(path.with_suffix(path.suffix + '.tmp'))
    frame.to_parquet(temporary); temporary.replace(path)


def make_identity(prepared, workers=4, min_free_memory_gb=3.0):
    if workers not in (2, 3, 4) or not math.isfinite(min_free_memory_gb) or not 2 <= min_free_memory_gb <= 8:
        raise ValueError('Workers 2..4 and memory reserve 2..8 GiB required')
    source = original.OUTPUT / prepared.identity['zone'].lower() / prepared.identity['run_id']
    result = {'engine': ENGINE, 'zone': prepared.identity['zone'], 'delivery_day': DAY,
        'original_identity': prepared.identity, 'original_run_directory': str(source),
        'original_experiment_sha256': _digest(prepared.identity),
        'coordinator_files': {f: sha(ROOT / f) for f in IMPLEMENTATION},
        'execution': {'executor': 'ProcessPoolExecutor_spawn', 'max_workers': workers,
            'initial_workers': 2, 'threads_per_fit': 2, 'min_free_memory_gb': float(min_free_memory_gb),
            'growth_reserve_gb': 0.8, 'checkpoint_writer': 'parent_only', 'zones': 'sequential',
            'scientific_functions': 'unchanged_original_module', 'assembly_order': 'chronological',
            'migration': 'validated_original_daily_raw_predictions_with_explicit_provenance'},
        'production_modified': False, 'promotion_eligible': False}
    result['run_id'] = _digest(result)[:16]
    return result


def _source_directory(prepared):
    return original.OUTPUT / prepared.identity['zone'].lower() / prepared.identity['run_id']


def verify_original_manifest(prepared):
    source = _source_directory(prepared)
    if not source.exists():
        return None
    original.safe_path(source)
    path = source / 'experiment.json'
    if not path.is_file() or json.loads(path.read_text(encoding='utf-8')) != prepared.identity or sha(path) != _digest(prepared.identity):
        raise ValueError('Original experiment identity/manifest differs; migration refused')
    return sha(path)


def _base_for_day(prepared, day):
    day = pd.Timestamp(day).date()
    if str(day) == DAY:
        selected = prepared.raw_future
    else:
        selected = prepared.raw_history.loc[prepared.raw_history.index.tz_convert(original.TIMEZONES[prepared.identity['zone']]).date == day]
    if selected.empty:
        raise ValueError('Requested day absent from immutable raw forecasts')
    return selected.loc[:, list(original.QUANTILES)].astype(float)


def _validate_raw_payload(payload, identity, day, index):
    if (payload.get('identity') != identity or payload.get('day') != str(day)
            or payload.get('index_ns') != index.asi8.tolist()
            or set(payload.get('raw', {})) != {'original', 'interaction'}
            or set(payload.get('audit', {})) != {'original', 'interaction'}):
        raise ValueError('Parallel daily checkpoint identity/grid/schema mismatch')
    for values in payload['raw'].values():
        array = np.asarray(values, dtype=float)
        if array.shape != (len(index),) or not np.isfinite(array).all():
            raise ValueError('Invalid raw corrections in parallel checkpoint')


def _check_baseline(prepared, day, payload):
    base = _base_for_day(prepared, day)
    expected = indexed(prepared.bundle.source_forecast if str(day) == DAY else prepared.bundle.residual_statistics).loc[base.index]
    original._compare_corrected(original.apply_variant(base, payload['raw']['original']), expected)


def verify_checkpoint(path, identity, day, prediction_index):
    path = safe_path(path)
    if not path.exists():
        return None
    sealed = json.loads(path.read_text(encoding='utf-8')); payload = sealed['payload']
    if sealed.get('sha256') != _digest(payload):
        raise ValueError('Parallel daily checkpoint checksum mismatch')
    _validate_raw_payload(payload, identity, day, prediction_index)
    return payload


def import_original_checkpoint(prepared, newidentity, source_path, prediction_index):
    """Read, validate and describe an import; never mutate either experiment."""
    source_path = Path(source_path).absolute()
    day = source_path.stem
    expected = _source_directory(prepared) / 'corrector_daily' / (day + '.json')
    if source_path != expected or source_path.resolve() != source_path:
        raise ValueError('Unexpected original checkpoint path')
    original.safe_path(source_path)
    if not source_path.exists():
        return None, None
    manifest_sha = verify_original_manifest(prepared)
    if manifest_sha is None:
        raise ValueError('Original checkpoint has no validated experiment manifest')
    before = sha(source_path)
    source_payload = original.verify_day_checkpoint(source_path, prepared.identity['run_id'], day, prediction_index)
    if source_payload is None or sha(source_path) != before:
        raise ValueError('Original checkpoint changed while importing')
    _validate_raw_payload(source_payload, prepared.identity['run_id'], day, prediction_index)
    if not _base_for_day(prepared, day).index.equals(prediction_index):
        raise ValueError('Imported day grid differs from immutable Chronos')
    _check_baseline(prepared, day, source_payload)
    provenance = {'original_run_id': prepared.identity['run_id'], 'original_path': str(source_path),
        'original_file_sha256': before, 'original_payload_sha256': _digest(source_payload),
        'original_experiment_sha256': manifest_sha, 'new_run_id': newidentity['run_id'],
        'raw_predictions_unchanged': True, 'thread_count_unchanged': 2}
    payload = dict(source_payload, identity=newidentity['run_id'], origin=provenance)
    return payload, provenance


def bounded_tasks(tasks, worker, executor, on_result, should_stop, available_gb,
                  min_free_memory_gb=3.0, max_workers=4, progress=None):
    """Bound submissions, drain on stop/error, and publish only in the caller."""
    if max_workers not in (2, 3, 4) or not math.isfinite(min_free_memory_gb) or not 2 <= min_free_memory_gb <= 8:
        raise ValueError('Invalid concurrency/memory limits')
    items = iter(tasks); pending = {}; exhausted = False
    capacity = 2; completed = 0; reason = None; errors = []
    while pending or not exhausted:
        if should_stop() and reason is None:
            reason = 'user_stop'
        free = float(available_gb())
        if (not math.isfinite(free) or free < min_free_memory_gb) and reason is None:
            reason = 'memory_reserve'
        if reason is None and completed >= capacity and capacity < max_workers and free >= min_free_memory_gb + 0.8:
            capacity += 1
        while reason is None and not exhausted and len(pending) < capacity:
            if should_stop():
                reason = 'user_stop'; break
            if float(available_gb()) < min_free_memory_gb:
                reason = 'memory_reserve'; break
            try:
                task = next(items)
            except StopIteration:
                exhausted = True; break
            pending[executor.submit(worker, task)] = task
        if progress:
            progress({'worker_count': capacity, 'pending_tasks': len(pending), 'computed_tasks': completed,
                      'available_memory_gb': free, 'draining': reason is not None, 'stop_reason': reason})
        if not pending:
            break
        ready, _ = wait(tuple(pending), timeout=0.5, return_when=FIRST_COMPLETED)
        for future in ready:
            task = pending.pop(future)
            try:
                value = future.result()
                on_result(task, value)
                completed += 1
            except Exception as exc:
                errors.append(exc); reason = 'worker_or_publication_error'
        # Never cancel: other finished valid work is verified and committed.
        if reason is not None and not pending:
            break
    if errors:
        raise errors[0]
    return {'stopped': reason is not None, 'reason': reason, 'completed': completed}


def _worker_init(zone, identity, upstreams=None, control_raw=None):
    global _PREPARED, _UPSTREAMS, _LIMITER, _CONTROL_RAW
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    _LIMITER = threadpool_limits(limits=2)
    _PREPARED = original.prepare(zone, threads=2)
    if _PREPARED.identity != identity['original_identity']:
        raise ValueError('Worker sources/scientific identity changed')
    if {name: sha(ROOT / name) for name in IMPLEMENTATION} != identity['coordinator_files']:
        raise ValueError('Coordinator changed before worker initialization')
    _UPSTREAMS, _CONTROL_RAW = upstreams, control_raw or {}


def _compute_corrector(task):
    day = pd.Timestamp(task['day']).date(); started = time.monotonic()
    raw_original, audit_original = _CONTROL_RAW.get(str(day), (None, None))
    if raw_original is None:
        raw_original, audit_original = original._fit_raw(_PREPARED, day, interaction=False)
    raw_interaction, audit_interaction = original._fit_raw(_PREPARED, day, interaction=True)
    base = _base_for_day(_PREPARED, day)
    payload = {'day': str(day), 'index_ns': base.index.asi8.tolist(),
        'raw': {'original': np.asarray(raw_original).tolist(), 'interaction': raw_interaction.tolist()},
        'audit': {'original': audit_original, 'interaction': audit_interaction}}
    _check_baseline(_PREPARED, day, payload)
    return {'payload': payload, 'elapsed_seconds': time.monotonic() - started}


def _validate_kalman(prepared, name, day, history, forecast, output, future, audit):
    tz = original.TIMEZONES[prepared.identity['zone']]
    selected = history.loc[history.index.tz_convert(tz).date == pd.Timestamp(day).date()]
    last = str(day) == '2026-09-21'
    original.validate_replayed_output(output, selected, future=False)
    if last:
        original.validate_replayed_output(future, forecast, future=True)
    elif future is not None:
        raise ValueError('Unexpected future on nonfinal Kalman day')
    if (audit['causality_violations'] or audit['quantile_crossings'] or audit['future_observations_assimilated']
            or audit['evaluation_days'] != 1 or audit['evaluation_hours'] != len(selected)
            or audit['future_forecast_hours'] != (len(forecast) if last else 0)
            or audit['training_lookback_days'] != 365
            or json_bytes(audit['config']) != json_bytes(prepared.identity['kalman_config'])
            or json_bytes(audit['covariate_config']) != json_bytes(prepared.identity['kalman_covariate_config'])):
        raise ValueError('Parallel Kalman audit violates frozen scientific protocol')


def _compute_kalman(task):
    prepared = _PREPARED; name, day = task['variant'], pd.Timestamp(task['day']).date()
    started = time.monotonic(); history, forecast = _UPSTREAMS[name]
    tz = original.TIMEZONES[prepared.identity['zone']]
    selected = history.loc[history.index.tz_convert(tz).date == day]
    prefix = history.loc[history.index.tz_convert(tz).date <= day]
    last = str(day) == '2026-09-21'
    replay = original.replay_kalman_overlay(prefix.reset_index(), timezone=tz, evaluation_start_day=str(day),
        covariates=prepared.bundle.covariates.copy(deep=True), config=prepared.config,
        covariate_config=prepared.covconfig, training_lookback_days=365, rolling_refit_workers=1,
        future_upstream=forecast.reset_index() if last else None,
        future_covariates=prepared.bundle.covariates.copy(deep=True) if last else None)
    output = original.merge_overlay(selected, replay.predictions.loc[selected.index])
    future = original.merge_overlay(forecast, replay.future_predictions) if last else None
    _validate_kalman(prepared, name, day, history, forecast, output, future, replay.audit)
    return {'backtest': output, 'forecast': future, 'audit': replay.audit,
            'elapsed_seconds': time.monotonic() - started}


def _pool(zone, identity, upstreams=None, control_raw=None):
    return ProcessPoolExecutor(max_workers=identity['execution']['max_workers'],
        mp_context=multiprocessing.get_context('spawn'), initializer=_worker_init,
        initargs=(zone, identity, upstreams, control_raw))


def worker_smoke(zone='DE', days=('2025-09-22', '2026-03-23'), workers=2):
    """Bounded, read-only real parallel/serial numerical equivalence diagnostic."""
    if zone not in original.BASELINES or workers != 2 or not 1 <= len(days) <= 2:
        raise ValueError('Smoke permits DE/NL, 1..2 days and exactly two workers')
    prepared = original.prepare(zone, threads=2)
    identity = make_identity(prepared, workers, 3.0)
    tasks = [{'day': str(pd.Timestamp(day).date())} for day in days]
    parallel = {}; began = time.monotonic()
    with _pool(zone, identity) as executor:
        state = bounded_tasks(tasks, _compute_corrector, executor,
            lambda task, value: parallel.__setitem__(task['day'], value), lambda: False,
            lambda: psutil.virtual_memory().available / 1024**3, max_workers=2)
    if state['stopped']:
        raise RuntimeError('Smoke stopped: ' + str(state['reason']))
    elapsed_parallel = time.monotonic() - began
    checks = []
    with threadpool_limits(limits=2):
        for task in tasks:
            day = task['day']; start = time.monotonic()
            for use, label in ((False, 'original'), (True, 'interaction')):
                serial, _ = original._fit_raw(prepared, pd.Timestamp(day).date(), interaction=use)
                if not np.allclose(serial, parallel[day]['payload']['raw'][label], rtol=0, atol=1e-9):
                    raise ValueError('Parallel and serial raw corrections differ')
            checks.append({'day': day, 'matched': True, 'tolerance': 1e-9,
                'parallel_task_seconds': parallel[day]['elapsed_seconds'], 'serial_task_seconds': time.monotonic() - start})
    if original.prepare(zone, threads=2).identity != prepared.identity:
        raise ValueError('Scientific inputs changed during smoke')
    return {'zone': zone, 'checks': checks, 'parallel_total_including_startup_seconds': elapsed_parallel,
        'original_run_id': prepared.identity['run_id'], 'coordinator_run_id': identity['run_id'], 'readonly': True}


class Paused(RuntimeError):
    pass


def _stop_requested(identity):
    path = safe_path(OUTPUT / 'request_stop.json')
    if not path.exists():
        return False
    request = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(request, dict):
        raise ValueError('Pause request must be an object')
    target = request.get('identity')
    return target is None or target == identity['run_id']


def _run_pool_tasks(prepared, identity, tasks, worker, on_result, progress, *, upstreams=None, control_raw=None):
    if not tasks:
        return
    def report(details):
        progress(execution=details)
    with _pool(prepared.identity['zone'], identity, upstreams, control_raw) as executor:
        state = bounded_tasks(tasks, worker, executor, on_result,
            lambda: _stop_requested(identity), lambda: psutil.virtual_memory().available / 1024**3,
            min_free_memory_gb=identity['execution']['min_free_memory_gb'],
            max_workers=identity['execution']['max_workers'], progress=report)
    if state['stopped']:
        raise Paused(state['reason'])


def _corrector_stage(prepared, identity, directory, progress, control_raw):
    zone = prepared.identity['zone']; tz = original.TIMEZONES[zone]
    historical = indexed(prepared.bundle.residual_statistics); future = indexed(prepared.bundle.source_forecast)
    days = list(pd.Index(historical.index.tz_convert(tz).date).unique()) + [pd.Timestamp(DAY).date()]
    cache_dir = safe_path(directory / 'corrector_daily'); cache_dir.mkdir(exist_ok=True)
    payloads = {}; imported = {}; tasks = []; count_cached = 0; count_computed = 0
    for day in days:
        base = _base_for_day(prepared, day); target = cache_dir / (str(day) + '.json')
        payload = verify_checkpoint(target, identity['run_id'], day, base.index)
        if payload is None:
            source = _source_directory(prepared) / 'corrector_daily' / (str(day) + '.json')
            payload, provenance = import_original_checkpoint(prepared, identity, source, base.index)
            if payload is not None:
                write_json(target, {'payload': payload, 'sha256': _digest(payload)})
        if payload is not None:
            _check_baseline(prepared, day, payload)
            payloads[str(day)] = payload; count_cached += 1
            if payload.get('origin') is not None:
                imported[str(day)] = payload['origin']
        else:
            tasks.append({'day': str(day)})
    migration = {'new_identity': identity['run_id'], 'original_identity': prepared.identity['run_id'],
        'original_experiment_sha256': verify_original_manifest(prepared), 'imported_days': imported,
        'imported_count': len(imported), 'original_files_modified': False, 'thread_count': 2}
    write_json(directory / 'migration_audit.json', migration)
    def update(**extra):
        progress('corrector_refits', len(payloads), len(days), completed_cached=count_cached,
            completed_computed=count_computed, **extra)
    def publish(task, result):
        nonlocal count_computed
        day = task['day']; payload = dict(result['payload'], identity=identity['run_id'], origin=None)
        _validate_raw_payload(payload, identity['run_id'], day, _base_for_day(prepared, day).index)
        _check_baseline(prepared, day, payload)
        payload['performance'] = {'compute_seconds': result['elapsed_seconds']}
        write_json(cache_dir / (day + '.json'), {'payload': payload, 'sha256': _digest(payload)})
        payloads[day] = payload; count_computed += 1; update(day=day)
    update()
    _run_pool_tasks(prepared, identity, tasks, _compute_corrector, publish, update, control_raw=control_raw)
    if set(payloads) != {str(day) for day in days}:
        raise ValueError('Incomplete corrector day set')
    frames = {name: (historical.copy(deep=True), future.copy(deep=True)) for name in VARIANTS}; records = []
    for day in days:
        payload = payloads[str(day)]; base = _base_for_day(prepared, day); is_future = str(day) == DAY
        for name, (use, upper) in VARIANTS.items():
            corrected = original.apply_variant(base, payload['raw']['interaction' if use else 'original'], upper=upper)
            destination = frames[name][1 if is_future else 0]
            for q in original.QUANTILES:
                destination.loc[base.index, 'residual_corrected__' + q] = corrected[q]
                if is_future:
                    destination.loc[base.index, q] = corrected[q]
            destination.loc[base.index, 'residual_correction'] = corrected.q50 - base.q50
            if is_future:
                destination.loc[base.index, 'price_eur_mwh'] = corrected.q50
        clipping = {recipe: {'hours': len(values), 'raw_gt_40': int((np.asarray(values) > 40).sum()),
            'raw_gt_80': int((np.asarray(values) > 80).sum()), 'raw_lt_minus40': int((np.asarray(values) < -40).sum())}
            for recipe, values in payload['raw'].items()}
        records.append({'day': str(day), **payload['audit'], 'clipping': clipping,
            'baseline_reproduced': True, 'origin': payload.get('origin'), 'performance': payload.get('performance')})
    return frames, records, migration


def _kalman_stage(prepared, identity, directory, frames, progress):
    tz = original.TIMEZONES[prepared.identity['zone']]
    days = list(pd.Index(indexed(prepared.bundle.kalman_view.backtest).index.tz_convert(tz).date).unique())
    payloads = {}; tasks = []; cached = 0; computed = 0
    for name, (history, forecast) in frames.items():
        subdir = safe_path(directory / name); subdir.mkdir(exist_ok=True)
        day_dir = safe_path(subdir / 'kalman_daily'); day_dir.mkdir(exist_ok=True)
        write_frame(subdir / 'upstream_history.parquet', history); write_frame(subdir / 'upstream_forecast.parquet', forecast)
        for day in days:
            receipt_path = day_dir / (str(day) + '.json')
            if not receipt_path.exists():
                tasks.append({'variant': name, 'day': str(day)}); continue
            sealed = json.loads(receipt_path.read_text(encoding='utf-8')); receipt = sealed['payload']
            if (sealed.get('sha256') != _digest(receipt) or receipt.get('identity') != identity['run_id']
                    or receipt.get('variant') != name or receipt.get('day') != str(day)):
                raise ValueError('Parallel Kalman checkpoint receipt identity/checksum mismatch')
            path = day_dir / (str(day) + '.parquet'); fpath = day_dir / 'forecast.parquet'
            last = day == days[-1]
            if sha(path) != receipt['backtest_sha256'] or (last and sha(fpath) != receipt['forecast_sha256']):
                raise ValueError('Parallel Kalman checkpoint file checksum mismatch')
            output = indexed(pd.read_parquet(path)); future = indexed(pd.read_parquet(fpath)) if last else None
            _validate_kalman(prepared, name, day, history, forecast, output, future, receipt['audit'])
            payloads[(name, str(day))] = {'backtest': output, 'forecast': future, 'audit': receipt['audit']}; cached += 1
    total = len(frames) * (len(days) + 1)
    def complete_count():
        return len(payloads) + sum(value['forecast'] is not None for value in payloads.values())
    def update(**extra):
        progress('kalman_parallel', complete_count(), total, completed_cached=cached, completed_computed=computed, **extra)
    def publish(task, result):
        nonlocal computed
        name, day = task['variant'], task['day']; history, forecast = frames[name]
        _validate_kalman(prepared, name, day, history, forecast, result['backtest'], result['forecast'], result['audit'])
        day_dir = safe_path(directory / name / 'kalman_daily'); path = day_dir / (day + '.parquet'); fpath = day_dir / 'forecast.parquet'
        write_frame(path, result['backtest'])
        if result['forecast'] is not None:
            write_frame(fpath, result['forecast'])
        receipt = {'identity': identity['run_id'], 'variant': name, 'day': day, 'audit': result['audit'],
            'backtest_sha256': sha(path), 'forecast_sha256': sha(fpath) if result['forecast'] is not None else None,
            'performance': {'compute_seconds': result['elapsed_seconds']}}
        write_json(day_dir / (day + '.json'), {'payload': receipt, 'sha256': _digest(receipt)})
        payloads[(name, day)] = result; computed += 1; update(day=day, variant=name)
    update()
    _run_pool_tasks(prepared, identity, tasks, _compute_kalman, publish, update, upstreams=frames)
    backtests = {'baseline': indexed(prepared.bundle.kalman_view.backtest)}
    futures = {'baseline': indexed(prepared.bundle.kalman_view.forecast)}
    for name in frames:
        outputs = [payloads[(name, str(day))]['backtest'] for day in days]
        annual = pd.concat(outputs); future = payloads[(name, str(days[-1]))]['forecast']
        if not annual.index.equals(backtests['baseline'].index) or future is None or not future.index.equals(futures['baseline'].index):
            raise ValueError('Incomplete deterministic annual/future assembly')
        audits = [payloads[(name, str(day))]['audit'] for day in days]
        write_frame(directory / name / 'backtest.parquet', annual); write_frame(directory / name / 'forecast.parquet', future)
        write_json(directory / name / 'replay_audit.json', audits)
        backtests[name], futures[name] = annual, future
    return backtests, futures


def _report(directory, identity, metrics, futures):
    esc = html.escape
    def num(v):
        return '—' if v is None else f'{v:.3f}'
    rows = ''.join('<tr><td>'+esc(period)+'</td><td>'+esc(name)+'</td><td>'+str(v['hours'])+'</td>'+''.join('<td>'+num(v[k])+'</td>' for k in ('mae','rmse','bias'))+'</tr>' for period, variants in metrics['slices'].items() for name,v in variants.items())
    predictions = ''.join('<tr><td>'+esc(str(t))+'</td>'+''.join('<td>'+num(frame.loc[t,original.FINAL_COLUMNS[1]])+'</td>' for frame in futures.values())+'</tr>' for t in futures['baseline'].index)
    document = '<!doctype html><html lang="fr"><meta charset="utf-8"><title>NYX correcteur '+identity['zone']+'</title><style>body{background:#090a0e;color:#e6e6ed;font:15px system-ui;max-width:1200px;margin:40px auto;padding:20px}h1,h2{color:#efb35b}table{width:100%;border-collapse:collapse;margin:20px 0}td,th{padding:8px;border-bottom:1px solid #292b33;text-align:right}td:first-child,td:nth-child(2){text-align:left}pre{white-space:pre-wrap;font-size:12px}</style><h1>NYX · SolarWind · correcteur '+identity['zone']+'</h1><p>Exécution parallèle du protocole inchangé. Interaction dans CatBoost seulement ; bornes [−40,+40] ou [−40,+80] €/MWh. Chronos figé, Kalman original rejoué sur chaque nouvel amont. Deux threads par fit ; checkpoints antérieurs conservés et importés avec provenance.</p><p>365 jours du 22/09/2025 au 21/09/2026 ; prévision distincte du 22/09/2026 sans observation future. Expérience rétrospective post-hoc, publication d’origine PIT non certifiée, substitutions NL héritées, aucune promotion ni modification de production.</p><h2>Erreurs (€/MWh)</h2><table><tr><th>Période</th><th>Variante</th><th>Heures</th><th>MAE</th><th>RMSE</th><th>Biais</th></tr>'+rows+'</table><h2>Pics et faux pics</h2><pre>'+esc(json_bytes(metrics['spikes']).decode())+'</pre><h2>Corrections brutes</h2><pre>'+esc(json_bytes(metrics['raw_correction_threshold_counts']).decode())+'</pre><h2>Prévision Q50 séparée (UTC)</h2><table><tr><th>Heure</th>'+''.join('<th>'+esc(n)+'</th>' for n in futures)+'</tr>'+predictions+'</table><details><summary>Provenance et paramètres</summary><pre>'+esc(json_bytes(identity).decode())+'</pre></details></html>'
    safe_path(directory / 'report.html').write_text(document, encoding='utf-8')


def _validate_scope(action, zones, threads, workers, min_free_memory_gb):
    if action not in ('validate','run') or not zones or len(zones) != len(set(zones)) or not set(zones) <= set(original.BASELINES):
        raise ValueError('Expected validate/run and unique DE/NL zones')
    if threads != 2 or workers not in (2,3,4) or not math.isfinite(min_free_memory_gb) or not 2 <= min_free_memory_gb <= 8:
        raise ValueError('Frozen two-thread recipe, workers 2..4, reserve 2..8 GiB required')


def _verify_completion(directory, identity):
    path = directory / 'completion.json'
    if not path.exists():
        return False
    receipt = json.loads(path.read_text(encoding='utf-8')); state = json.loads((directory/'status.json').read_text(encoding='utf-8'))
    if (receipt.get('identity') != identity['run_id'] or receipt.get('annual_complete') is not True
            or set(receipt.get('files',{})) != set(RESULT_FILES) or state.get('status') != 'COMPLETE'
            or state.get('identity') != identity['run_id'] or state.get('annual_complete') is not True
            or state.get('evaluation_days') != 365 or state.get('evaluation_hours') != 8760
            or state.get('future_forecast_hours') != 24 or state.get('zone') != identity['zone']
            or any(sha(safe_path(directory/f)) != digest for f,digest in receipt['files'].items())):
        raise ValueError('Parallel completion is incomplete, changed or belongs to another identity')
    return True


def run(*, action='validate', zones=('DE','NL'), threads=2, workers=4, min_free_memory_gb=3.0):
    _validate_scope(action,zones,threads,workers,min_free_memory_gb)
    prepared = {z: original.prepare(z, threads=2) for z in zones}
    identities = {z: make_identity(p, workers, min_free_memory_gb) for z,p in prepared.items()}
    for zone,p in prepared.items():
        verify_original_manifest(p)
        print(json.dumps({'zone':zone,'validation':'PASS','identity':identities[zone]['run_id'],
            'original_identity':p.identity['run_id']}),flush=True)
    if action == 'validate':
        return identities
    safe_path(OUTPUT); OUTPUT.mkdir(parents=True,exist_ok=True)
    workdirs = {z:OUTPUT/z.lower()/identity['run_id'] for z,identity in identities.items()}
    pointer = OUTPUT/('latest_'+'_'.join(zones)+'.json'); results={}
    # The serial batch lock excludes an old writer even in a different namespace.
    with exclusive_process_lock(original.OUTPUT/'batch.lock'), exclusive_process_lock(OUTPUT/'batch.lock'), threadpool_limits(limits=2):
        for zone,p in prepared.items():
            identity=identities[zone]; directory=safe_path(workdirs[zone]); directory.mkdir(parents=True,exist_ok=True)
            if (directory/'experiment.json').exists() and json.loads((directory/'experiment.json').read_text(encoding='utf-8'))!=identity:
                raise ValueError('Parallel experiment identity changed')
            if _verify_completion(directory,identity):
                results[zone]=str(directory/'report.html');continue
            write_json(directory/'experiment.json',identity)
            current_phase=None; phase_start=time.monotonic(); phase_utc=None; last_state={}; last_emit=0.0
            def progress(phase,completed=0,total=1,**extra):
                nonlocal current_phase,phase_start,phase_utc,last_state,last_emit
                now=datetime.now(timezone.utc).isoformat();tick=time.monotonic()
                if phase!=current_phase:
                    current_phase=phase;phase_start=tick;phase_utc=now
                state={'status':'RUNNING','phase':phase,'zone':zone,'identity':identity['run_id'],'pid':os.getpid(),
                    'updated_utc':now,'phase_started_utc':phase_utc,'phase_elapsed_seconds':tick-phase_start,
                    'progress':{'completed':completed,'total':total,'unit':'jours / tâches','label':phase},**extra}
                if last_state.get('progress')==state['progress'] and tick-last_emit<1 and not extra.get('day'):
                    return
                write_json(directory/'status.json',state);write_json(directory/'progress.json',state)
                write_json(pointer,{'engine':ENGINE,'status':'RUNNING','active_zone':zone,'pid':os.getpid(),
                    'workdirs':workdirs,'results':results,'production_modified':False})
                if last_state.get('progress')!=state['progress']:
                    print(f'{zone} {phase}: {completed}/{total}',flush=True)
                last_state=state;last_emit=tick
            try:
                if _stop_requested(identity):
                    raise Paused('user_stop')
                progress('baseline_corrector_controls',0,5)
                controls,raw_cache=original.corrector_controls(p,lambda phase,n,total,**kw:progress(phase,n,total,**kw))
                if _stop_requested(identity):
                    raise Paused('user_stop')
                progress('baseline_kalman_controls',0,4)
                kalman=original.baseline_controls(p.bundle,p.config,p.covconfig,zone,None,1)
                write_json(directory/'baseline_controls.json',{'identity':identity['run_id'],'corrector':controls,'kalman':kalman})
                write_frame(directory/'interaction.parquet',p.interaction);write_json(directory/'feature_audit.json',p.feature_audit)
                frames,records,migration=_corrector_stage(p,identity,directory,progress,raw_cache)
                write_json(directory/'corrector_audit.json',records)
                backtests,futures=_kalman_stage(p,identity,directory,frames,progress)
                progress('verify_and_report')
                checked=original.prepare(zone,threads=2)
                if checked.identity!=p.identity or make_identity(checked,workers,min_free_memory_gb)!=identity:
                    raise ValueError('Scientific sources or coordinator changed during run')
                verify_original_manifest(checked)
                for origin in migration['imported_days'].values():
                    if sha(Path(origin['original_path']))!=origin['original_file_sha256']:
                        raise ValueError('Original imported checkpoint changed')
                metrics=original.evaluate(backtests,original.TIMEZONES[zone])
                metrics['raw_correction_threshold_counts']={phase:{recipe:{key:sum(record['clipping'][recipe][key] for record in records
                    if (('2025-09-22'<=record['day']<DAY) if phase=='evaluation' else record['day']==DAY))
                    for key in ('hours','raw_gt_40','raw_gt_80','raw_lt_minus40')} for recipe in ('original','interaction')}
                    for phase in ('evaluation','forecast')}
                write_json(directory/'metrics.json',metrics);_report(directory,identity,metrics,futures)
                write_json(directory/'completion.json',{'identity':identity['run_id'],'annual_complete':True,
                    'files':{f:sha(directory/f) for f in RESULT_FILES}})
                state={'status':'COMPLETE','phase':'complete','zone':zone,'identity':identity['run_id'],'annual_complete':True,
                    'evaluation_days':365,'evaluation_hours':8760,'future_forecast_hours':24,'variants':list(VARIANTS),
                    'updated_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid()}
                write_json(directory/'status.json',state);write_json(directory/'progress.json',state)
                results[zone]=str(directory/'report.html')
            except Paused as exc:
                state={**last_state,'status':'PAUSED','zone':zone,'identity':identity['run_id'],'pid':os.getpid(),
                    'reason':str(exc),'automatic_resume':False,
                    'updated_utc':datetime.now(timezone.utc).isoformat()}
                write_json(directory/'status.json',state);write_json(directory/'progress.json',state)
                write_json(pointer,{'engine':ENGINE,'status':'PAUSED','active_zone':zone,'workdirs':workdirs,'results':results,
                    'reason':str(exc),'automatic_resume':False,'production_modified':False})
                return {'status':'PAUSED','reason':str(exc),'results':results}
            except Exception as exc:
                state={**last_state,'status':'FAILED','zone':zone,'identity':identity['run_id'],'pid':os.getpid(),
                    'error':f'{type(exc).__name__}: {exc}',
                    'updated_utc':datetime.now(timezone.utc).isoformat()}
                write_json(directory/'status.json',state);write_json(directory/'progress.json',state)
                write_json(pointer,{'engine':ENGINE,'status':'FAILED','failed_zone':zone,'workdirs':workdirs,'results':results,
                    'error':str(exc),'production_modified':False})
                raise
        write_json(pointer,{'engine':ENGINE,'status':'COMPLETE','annual_complete':True,'workdirs':workdirs,'results':results})
        links=''.join('<li><a href="'+html.escape(Path(path).relative_to(OUTPUT).as_posix())+'">'+zone+'</a></li>' for zone,path in results.items())
        safe_path(OUTPUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>NYX · correcteur parallèle</title><h1>SolarWind · résultats DE/NL</h1><p>Exécution accélérée, protocole scientifique inchangé, hors production.</p><ul>'+links+'</ul>',encoding='utf-8')
    return results
