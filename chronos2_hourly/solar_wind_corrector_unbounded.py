"""Unbounded corrector ablation, strictly after both sealed DE/NL predecessors.

No CatBoost fit is performed here. Already sealed daily raw corrections are
reused with their full provenance; only clipping is removed before an unchanged
rolling-365 Kalman replay. All writes remain in this experiment namespace.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import html
from html.parser import HTMLParser
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
from . import solar_wind_corrector_parallel as prior_parallel
from . import solar_wind_corrector_reuse as predecessor
from .models.residual_corrector import apply_residual_correction
from .process_lock import exclusive_process_lock

ROOT, DAY = original.ROOT, original.DAY
ENGINE = 'solar_wind_corrector_unbounded_v1'
OUTPUT = ROOT / 'runs/experiments' / ENGINE / DAY
SOURCE_RUNS = {'DE': '48d7c14fd8bc2e4f', 'NL': '8f246d6bbd54b511'}
VARIANTS = ('baseline_unbounded', 'interaction_unbounded')
RESULT_FILES = ('experiment.json', 'source_audit.json', 'unbounded_corrections_audit.json',
    'metrics.json', 'report.html', 'baseline_backtest.parquet', 'baseline_forecast.parquet',
    *(f'{variant}/{name}' for variant in VARIANTS for name in (
        'upstream_history.parquet', 'upstream_forecast.parquet', 'backtest.parquet',
        'forecast.parquet', 'replay_audit.json')))
IMPLEMENTATION = ('run_solar_wind_corrector_unbounded.py',
    'chronos2_hourly/solar_wind_corrector_unbounded.py',
    'run_solar_wind_corrector_reuse.py', 'chronos2_hourly/solar_wind_corrector_reuse.py',
    'run_solar_wind_corrector_parallel.py', 'chronos2_hourly/solar_wind_corrector_parallel.py')
json_bytes, sha, indexed = original.json_bytes, original.sha, original.indexed
_PREPARED = _UPSTREAMS = _LIMITER = None


def _digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


class PendingPredecessor(RuntimeError):
    code = 'PENDING_PREDECESSOR'

    def __init__(self, message, states=None):
        super().__init__(message)
        self.states = states or {}


class _ReportLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = set()

    def handle_starttag(self, tag, attrs):
        if tag.lower() == 'a':
            self.links.update(value for name, value in attrs if name.lower() == 'href' and value)


def safe_path(path):
    path = Path(path).absolute()
    if path.resolve() != path or not path.is_relative_to(OUTPUT.absolute()):
        raise ValueError(f'Unsafe parallel experiment output: {path}')
    if path.is_dir():
        for child in path.rglob('*'):
            if child.is_symlink() or (getattr(child.lstat(), 'st_file_attributes', 0) & 0x400):
                raise ValueError(f'Redirected parallel output descendant: {child}')
    return path


def _replace_retry(temporary, destination):
    """Retry only transient Windows sharing/access-denied atomic replacements."""
    delays = (0.05, 0.1, 0.2, 0.4, 0.5, 0.5, 0.5)
    for attempt in range(len(delays) + 1):
        try:
            temporary.replace(destination)
            return
        except PermissionError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def write_json(path, value):
    path = safe_path(path)
    temporary = safe_path(path.with_suffix(path.suffix + '.tmp'))
    temporary.write_bytes(json_bytes(value)); _replace_retry(temporary, path)


def write_frame(path, frame):
    path = safe_path(path)
    temporary = safe_path(path.with_suffix(path.suffix + '.tmp'))
    frame.to_parquet(temporary); _replace_retry(temporary, path)


def write_text(path, value):
    path = safe_path(path)
    temporary = safe_path(path.with_suffix(path.suffix + '.tmp'))
    temporary.write_text(value, encoding='utf-8'); _replace_retry(temporary, path)



def predecessor_gate():
    """Read-only gate: BOTH countries, all 24 sealed files, no scientific work."""
    states = {}; directories = {}
    for zone, run_id in SOURCE_RUNS.items():
        directory = predecessor.OUTPUT / zone.lower() / run_id
        directories[zone] = directory
        status = directory / 'status.json'
        if not status.exists():
            states[zone] = {'status': 'ABSENT'}
        else:
            state = json.loads(status.read_text(encoding='utf-8'))
            states[zone] = state
            if state.get('identity') != run_id or state.get('zone') != zone:
                raise ValueError('Predecessor status has a different pinned identity/zone')
    pending = [z for z, state in states.items() if state.get('status') != 'COMPLETE']
    if pending:
        raise PendingPredecessor('Both DE and NL predecessors must finish first: ' + ', '.join(pending), states)
    pointer = predecessor.OUTPUT / 'latest_DE_NL.json'
    index = predecessor.OUTPUT / 'index.html'
    if not pointer.exists():
        raise PendingPredecessor('Predecessor batch publication has not finished', states)
    batch = json.loads(pointer.read_text(encoding='utf-8'))
    if batch.get('status') != 'COMPLETE':
        raise PendingPredecessor('Predecessor batch is still publishing its combined results', states)
    expected_workdirs = {zone:str(directory) for zone,directory in directories.items()}
    expected_results = {zone:str(directory/'report.html') for zone,directory in directories.items()}
    if (batch.get('engine') != predecessor.ENGINE or batch.get('annual_complete') is not True
            or batch.get('workdirs') != expected_workdirs or batch.get('results') != expected_results):
        raise ValueError('Predecessor batch completion pointer differs from the pinned DE/NL results')
    if not index.is_file():
        raise PendingPredecessor('Predecessor combined index is not yet published', states)
    index_text = index.read_text(encoding='utf-8'); links = _ReportLinks(); links.feed(index_text)
    expected_links = {(directory/'report.html').relative_to(predecessor.OUTPUT).as_posix() for directory in directories.values()}
    if not index_text.strip() or not expected_links <= links.links:
        raise ValueError('Predecessor combined index does not link both pinned reports')
    result = {}
    for zone, directory in directories.items():
        predecessor.safe_path(directory)
        identity = json.loads((directory / 'experiment.json').read_text(encoding='utf-8'))
        if (identity.get('run_id') != SOURCE_RUNS[zone] or identity.get('zone') != zone
                or identity.get('engine') != predecessor.ENGINE or identity.get('delivery_day') != DAY):
            raise ValueError('Predecessor manifest identity differs from required experiment')
        if not predecessor._verify_completion(directory, identity):
            raise ValueError('Complete predecessor lacks a valid completion receipt')
        state = states[zone]
        if set(state.get('variants', [])) != set(predecessor.VARIANTS):
            raise ValueError('Predecessor did not complete every expected variant')
        for name, digest in identity['coordinator_files'].items():
            if sha(ROOT / name) != digest:
                raise ValueError('Predecessor coordinator implementation changed')
        receipt = json.loads((directory / 'completion.json').read_text(encoding='utf-8'))
        if len(receipt['files']) != 24:
            raise ValueError('Expected all 24 predecessor artifacts')
        report = (directory / 'report.html').read_text(encoding='utf-8')
        if not report.strip() or '<html' not in report.lower() or '</html>' not in report.lower():
            raise ValueError('Predecessor report is absent/incomplete')
        result[zone] = {'identity': identity, 'directory': str(directory),
            'completion_sha256': sha(directory / 'completion.json'),
            'status_sha256': sha(directory / 'status.json'), 'files': receipt['files'],
            'batch_pointer_sha256':sha(pointer), 'batch_index_sha256':sha(index)}
    return result


def apply_unbounded(base, raw_correction):
    """Apply a finite raw shift with BOTH corrector clipping mechanisms disabled."""
    if (not isinstance(base, pd.DataFrame) or tuple(base.columns) != tuple(original.QUANTILES)
            or not isinstance(base.index, pd.DatetimeIndex) or base.index.tz is None
            or not base.index.is_unique or not base.index.is_monotonic_increasing or base.index.hasnans):
        raise ValueError('Unbounded correction requires unique ordered physical hours and three quantiles')
    values = base.to_numpy(dtype=float); raw = np.asarray(raw_correction, dtype=float)
    if (raw.shape != (len(base),) or not len(base) or not np.isfinite(raw).all()
            or not np.isfinite(values).all()
            or (values[:, 0] > values[:, 1]).any() or (values[:, 1] > values[:, 2]).any()):
        raise ValueError('Unbounded correction inputs must be finite and ordered')
    if isinstance(raw_correction, pd.Series) and not raw_correction.index.equals(base.index):
        raise ValueError('Raw correction and quantile indices differ')
    with np.errstate(over='ignore', invalid='ignore'):
        result = apply_residual_correction(base, raw, correction_scale=1.0,
            correction_clip=None, max_abs_correction=None)
    output = result.to_numpy(dtype=float)
    if (not result.index.equals(base.index) or not np.isfinite(output).all()
            or (output[:, 0] > output[:, 1]).any() or (output[:, 1] > output[:, 2]).any()):
        raise ValueError('Unbounded correction produced nonfinite/crossed quantiles')
    if not np.array_equal(output, values + raw[:, None]):
        raise ValueError('Unexpected scaling/clipping in unbounded corrector')
    return result


def _source_base(prepared, day):
    return predecessor._base_for_day(prepared, day)


def _frozen_base_preserved(frame, frozen, *, future):
    if not frame.index.equals(frozen.index):
        raise ValueError('Predecessor source grid differs from immutable baseline')
    for column in frozen:
        if column == 'actual' or column.startswith('chronos2__') or 'forecast_origin' in column:
            pd.testing.assert_series_equal(frame[column], frozen[column], check_names=False, check_dtype=False)
    if future and 'actual' in frame and frame.actual.notna().any():
        raise ValueError('A future source contains observed target labels')


def _source_artifacts(prepared, source):
    directory = Path(source['directory'])
    historical = indexed(prepared.bundle.residual_statistics)
    future_base = indexed(prepared.bundle.source_forecast)
    annual_grid = indexed(prepared.bundle.kalman_view.backtest).index
    expected_annual = pd.date_range(pd.Timestamp('2025-09-22', tz=original.TIMEZONES[prepared.identity['zone']]),
        pd.Timestamp(DAY, tz=original.TIMEZONES[prepared.identity['zone']]), freq='h', inclusive='left').tz_convert('UTC')
    if not annual_grid.equals(expected_annual) or len(annual_grid) != 8760 or len(future_base) != 24:
        raise ValueError('Expected complete 365-day baseline and 24-hour future')
    frames = {}
    for variant in predecessor.VARIANTS:
        history = indexed(pd.read_parquet(directory / variant / 'upstream_history.parquet'))
        forecast = indexed(pd.read_parquet(directory / variant / 'upstream_forecast.parquet'))
        annual = indexed(pd.read_parquet(directory / variant / 'backtest.parquet'))
        final = indexed(pd.read_parquet(directory / variant / 'forecast.parquet'))
        _frozen_base_preserved(history, historical, future=False)
        _frozen_base_preserved(forecast, future_base, future=True)
        if not annual.index.equals(annual_grid):
            raise ValueError('Predecessor annual result is partial')
        original.validate_replayed_output(annual, history.loc[annual_grid], future=False)
        original.validate_replayed_output(final, forecast, future=True)
        audits = json.loads((directory / variant / 'replay_audit.json').read_text(encoding='utf-8'))
        if len(audits) != 365:
            raise ValueError('Predecessor Kalman replay is not annual')
        if any(a.get('causality_violations') or a.get('quantile_crossings') or a.get('future_observations_assimilated')
               or a.get('evaluation_days') != 1 or a.get('training_lookback_days') != 365
               or json_bytes(a.get('config')) != json_bytes(prepared.identity['kalman_config'])
               or json_bytes(a.get('covariate_config')) != json_bytes(prepared.identity['kalman_covariate_config']) for a in audits):
            raise ValueError('Predecessor Kalman audit does not match frozen protocol')
        if sum(a['evaluation_hours'] for a in audits) != 8760 or sum(a['future_forecast_hours'] for a in audits) != 24:
            raise ValueError('Predecessor Kalman audit horizon coverage is incomplete')
        frames[variant] = history, forecast
    metrics = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
    if not metrics.get('slices') or not metrics.get('spikes'):
        raise ValueError('Predecessor metrics are incomplete')
    return frames


def load_source_checkpoints(prepared, source):
    """Validate every sealed daily raw result and all inherited provenance, no fit."""
    identity = source['identity']
    if (identity.get('original_identity') != prepared.identity
            or identity.get('run_id') != SOURCE_RUNS[prepared.identity['zone']]):
        raise ValueError('Completed predecessor scientific identity changed')
    if predecessor.make_identity(prepared, identity['execution']['max_workers'],
            identity['execution']['min_free_memory_gb']) != identity:
        raise ValueError('Completed predecessor identity cannot be reproduced')
    directory = Path(source['directory'])
    frames = _source_artifacts(prepared, source)
    tz = original.TIMEZONES[prepared.identity['zone']]
    days = list(pd.Index(indexed(prepared.bundle.residual_statistics).index.tz_convert(tz).date).unique())
    days.append(pd.Timestamp(DAY).date())
    if len(days) != 731:
        raise ValueError('Expected 730 history days and one forecast day')
    audit_path = directory / 'corrector_audit.json'
    if sha(audit_path) != source['files'].get('corrector_audit.json'):
        raise ValueError('Sealed predecessor corrector audit checksum changed')
    daily_audits = json.loads(audit_path.read_text(encoding='utf-8'))
    if (not isinstance(daily_audits, list) or len(daily_audits) != len(days)
            or any(not isinstance(record, dict) for record in daily_audits)):
        raise ValueError('Sealed predecessor corrector audit is incomplete')
    audited_days = {record.get('day'): record for record in daily_audits}
    if len(audited_days) != len(days) or set(audited_days) != set(map(str, days)):
        raise ValueError('Sealed predecessor corrector audit has duplicate/missing days')
    payloads = {}; receipts = {}
    for day in days:
        base = _source_base(prepared, day); path = directory / 'corrector_daily' / (str(day)+'.json')
        before = sha(path)
        payload = predecessor.verify_checkpoint(path, identity['run_id'], day, base.index)
        if payload is None or sha(path) != before:
            raise ValueError('A required completed-predecessor checkpoint is missing/changed')
        predecessor.verify_origin(prepared, identity, day, base.index, payload)
        predecessor._check_baseline(prepared, day, payload)
        record = audited_days[str(day)]
        clipping = {recipe: {'hours': len(values), 'raw_gt_40': int((np.asarray(values)>40).sum()),
            'raw_gt_80': int((np.asarray(values)>80).sum()), 'raw_lt_minus40': int((np.asarray(values)<-40).sum())}
            for recipe, values in payload['raw'].items()}
        if (record.get('baseline_reproduced') is not True
                or any(json_bytes(record.get(recipe)) != json_bytes(payload['audit'][recipe])
                    for recipe in ('original', 'interaction'))
                or json_bytes(record.get('clipping')) != json_bytes(clipping)
                or json_bytes(record.get('origin')) != json_bytes(payload.get('origin'))
                or json_bytes(record.get('baseline_reconstruction')) != json_bytes(payload.get('baseline_reconstruction'))):
            raise ValueError('Daily raw checkpoint audit/provenance/counts differ from sealed completed audit')
        for name, (interaction, upper) in predecessor.VARIANTS.items():
            expected = frames[name][1 if str(day) == DAY else 0].loc[base.index]
            clipped = original.apply_variant(base, payload['raw']['interaction' if interaction else 'original'], upper=upper)
            original._compare_corrected(clipped, expected)
        payloads[str(day)] = payload
        receipts[str(day)] = {'source_path': str(path), 'file_sha256': before,
            'payload_sha256': _digest(payload), 'origin': payload.get('origin'),
            'baseline_reconstruction': payload.get('baseline_reconstruction')}
    for name, digest in source['files'].items():
        if sha(directory / name) != digest:
            raise ValueError('Completed predecessor artifact changed during raw validation')
    if sha(directory / 'completion.json') != source['completion_sha256'] or sha(directory / 'status.json') != source['status_sha256']:
        raise ValueError('Predecessor completion changed during validation')
    return payloads, {'source_identity': identity['run_id'], 'source_directory': str(directory),
        'source_completion_sha256': source['completion_sha256'], 'source_files': source['files'],
        'corrector_audit_sha256': source['files']['corrector_audit.json'],
        'raw_tail_evidence': 'sealed_daily_raw_receipts_and_consistent_completed_audit_not_inversion_of_clipped_values',
        'daily_checkpoints': receipts, 'days': len(payloads), 'new_catboost_fits': 0,
        'existing_reconstructed_baseline_days': sum(p.get('baseline_reconstruction') is not None for p in payloads.values()),
        'sources_modified': False}


def make_identity(prepared, source=None, source_audit=None, workers=4, min_free_memory_gb=3.0, all_sources=None):
    _validate_scope('validate', (prepared.identity['zone'],), 2, workers, min_free_memory_gb)
    all_sources = predecessor_gate() if all_sources is None else all_sources
    source = all_sources[prepared.identity['zone']] if source is None else source
    if source_audit is None:
        _, source_audit = load_source_checkpoints(prepared, source)
    result = {'engine': ENGINE, 'zone': prepared.identity['zone'], 'delivery_day': DAY,
        'original_identity': prepared.identity, 'predecessor_identity': source['identity'],
        'predecessor_gate': {z: {'identity': item['identity']['run_id'],
            'completion_sha256': item['completion_sha256'], 'status_sha256': item['status_sha256'],
            'batch_pointer_sha256':item['batch_pointer_sha256'], 'batch_index_sha256':item['batch_index_sha256'],
            'files': item['files']} for z, item in all_sources.items()},
        'source_checkpoints': {day: {'file_sha256': record['file_sha256'], 'payload_sha256': record['payload_sha256']}
            for day, record in source_audit['daily_checkpoints'].items()},
        'coordinator_files': {name: sha(ROOT/name) for name in IMPLEMENTATION},
        'variants': {name: {'raw_recipe': 'interaction' if name == 'interaction_unbounded' else 'original',
            'correction_clip': None, 'max_abs_correction': None, 'correction_scale': 1.0} for name in VARIANTS},
        'scientific_change': 'remove_both_residual_corrector_bounds_only',
        'kalman_config': prepared.identity['kalman_config'],
        'kalman_covariate_config': prepared.identity['kalman_covariate_config'],
        'new_catboost_fits': 0, 'execution': {'executor': 'ProcessPoolExecutor_spawn',
            'max_workers': workers, 'initial_workers': 2, 'threads_per_fit': 2,
            'min_free_memory_gb': float(min_free_memory_gb), 'growth_reserve_gb': .8,
            'checkpoint_writer': 'parent_only', 'zones': 'sequential', 'assembly_order': 'chronological'},
        'production_modified': False, 'promotion_eligible': False,
        'evaluation_start_day': '2025-09-22', 'evaluation_end_day': '2026-09-21',
        'evaluation_days': 365, 'evaluation_hours': 8760, 'future_forecast_hours': 24}
    result['run_id'] = _digest(result)[:16]
    return result


def assemble_unbounded(prepared, payloads):
    historical = indexed(prepared.bundle.residual_statistics); future = indexed(prepared.bundle.source_forecast)
    tz = original.TIMEZONES[prepared.identity['zone']]
    days = list(pd.Index(historical.index.tz_convert(tz).date).unique()) + [pd.Timestamp(DAY).date()]
    if set(payloads) != set(map(str, days)):
        raise ValueError('Unbounded upstream requires every sealed day')
    frames = {name: (historical.copy(deep=True), future.copy(deep=True)) for name in VARIANTS}
    audits = []
    for day in days:
        base = _source_base(prepared, day); payload = payloads[str(day)]; is_future = str(day) == DAY
        for name in VARIANTS:
            if name == 'baseline_unbounded' and payload.get('baseline_reconstruction') is not None:
                # On a verified strict-interior day unbounded == source, bitwise.
                continue
            recipe = 'interaction' if name == 'interaction_unbounded' else 'original'
            corrected = apply_unbounded(base, payload['raw'][recipe]); target = frames[name][int(is_future)]
            for q in original.QUANTILES:
                target.loc[base.index, 'residual_corrected__'+q] = corrected[q]
                if is_future:
                    target.loc[base.index, q] = corrected[q]
            target.loc[base.index, 'residual_correction'] = corrected.q50 - base.q50
            if is_future:
                target.loc[base.index, 'price_eur_mwh'] = corrected.q50
        audits.append({'day': str(day), 'source_payload_sha256': _digest(payload),
            'original_source_audit': payload['audit'], 'origin': payload.get('origin'),
            'baseline_reconstruction': payload.get('baseline_reconstruction'),
            'raw_counts': {recipe: {'hours': len(values), 'raw_gt_40': int((np.asarray(values)>40).sum()),
                'raw_gt_80': int((np.asarray(values)>80).sum()), 'raw_lt_minus40': int((np.asarray(values)<-40).sum()),
                'minimum': float(np.min(values)), 'maximum': float(np.max(values))}
                for recipe,values in payload['raw'].items()},
            'corrector_clipping_applied': False, 'new_catboost_fit_performed': False})
    for history, forecast in frames.values():
        _frozen_base_preserved(history, historical, future=False)
        _frozen_base_preserved(forecast, future, future=True)
        for frame in (history, forecast):
            q = frame.loc[:, ['residual_corrected__'+name for name in original.QUANTILES]].to_numpy(float)
            if not np.isfinite(q).all() or (q[:,0]>q[:,1]).any() or (q[:,1]>q[:,2]).any():
                raise ValueError('Invalid assembled unbounded upstream')
    return frames, audits

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
    global _PREPARED, _UPSTREAMS, _LIMITER
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    _LIMITER = threadpool_limits(limits=2)
    _PREPARED = original.prepare(zone, threads=2)
    if _PREPARED.identity != identity['original_identity']:
        raise ValueError('Worker scientific sources/identity changed')
    if {name:sha(ROOT/name) for name in IMPLEMENTATION} != identity['coordinator_files']:
        raise ValueError('Unbounded coordinator changed before worker initialization')
    _UPSTREAMS = upstreams


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
        initargs=(zone, identity, upstreams, None))


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



def _report(directory, identity, metrics, futures, correction_audit):
    esc = html.escape
    def num(value):
        return '—' if value is None else f'{value:.3f}'
    labels = {'baseline':'Baseline SolarWind plafonnée ±40',
        'baseline_unbounded':'Correcteur original sans bornes',
        'interaction_unbounded':'Correcteur avec interaction sans bornes'}
    rows = ''.join('<tr><td>'+esc(period)+'</td><td>'+esc(labels[name])+'</td><td>'+str(value['hours'])+'</td>'+
        ''.join('<td>'+num(value[key])+'</td>' for key in ('mae','rmse','bias'))+'</tr>'
        for period, variants in metrics['slices'].items() for name,value in variants.items())
    future_rows = ''.join('<tr><td>'+esc(str(hour))+'</td>'+''.join('<td>'+num(frame.loc[hour,original.FINAL_COLUMNS[1]])+'</td>'
        for frame in futures.values())+'</tr>' for hour in futures['baseline'].index)
    document = '<!doctype html><html lang="fr"><meta charset="utf-8"><title>NYX · correcteur sans bornes '+identity['zone']+'</title>'
    document += '<style>body{background:#090a0e;color:#e6e6ed;font:15px system-ui;max-width:1200px;margin:40px auto;padding:20px}h1,h2{color:#efb35b}table{width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #292b33;text-align:right}td:first-child,td:nth-child(2){text-align:left}pre{white-space:pre-wrap;font-size:12px}</style>'
    document += '<h1>NYX · SolarWind · correcteurs sans bornes · '+identity['zone']+'</h1>'
    document += '<p>Les bornes basse ET haute du correcteur résiduel sont supprimées. Corrections brutes journalières scellées réutilisées, sans aucun nouvel entraînement CatBoost. Chronos figé, MAE et fenêtre passée inchangés. Deux variantes : recette originale et recette avec interaction. Kalman est rejoué sur chaque amont, avec ses paramètres et ses propres limites strictement inchangés.</p>'
    document += '<p>Comparateur : SolarWind original plafonné à ±40 €/MWh avec Kalman. Évaluation appariée de 365 jours, 22/09/2025–21/09/2026 (8760 heures). Prévision du 22/09/2026 distincte, 24 heures sans observation future. Expérience rétrospective post-hoc, PIT publication non certifié, aucune modification ni promotion de production.</p>'
    document += '<h2>Erreurs (€/MWh)</h2><table><tr><th>Période</th><th>Modèle</th><th>Heures</th><th>MAE</th><th>RMSE</th><th>Biais</th></tr>'+rows+'</table>'
    document += '<h2>Pics et faux pics</h2><pre>'+esc(json_bytes(metrics['spikes']).decode())+'</pre>'
    document += '<h2>Prévision Q50 séparée (UTC)</h2><table><tr><th>Heure</th>'+''.join('<th>'+esc(labels[name])+'</th>' for name in futures)+'</tr>'+future_rows+'</table>'
    document += '<h2>Preuves de non-plafonnement</h2><p>731 journées sources par pays, aucun nouveau fit CatBoost. Les journées de baseline reconstruites strictement à l’intérieur des anciennes bornes conservent leur amont scellé bitwise. Les autres journées utilisent les corrections brutes originales, y compris au-delà de +80 et en dessous de −40.</p>'
    document += '<details><summary>Identité, sources et protocole</summary><pre>'+esc(json_bytes(identity).decode())+'</pre></details></html>'
    write_text(directory/'report.html', document)


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
            or set(state.get('variants', [])) != set(VARIANTS)
            or any(sha(safe_path(directory/f)) != digest for f,digest in receipt['files'].items())):
        raise ValueError('Parallel completion is incomplete, changed or belongs to another identity')
    return True



def _verify_source_seals(sources, source_audits):
    fresh = predecessor_gate()
    if json_bytes(fresh) != json_bytes(sources):
        raise ValueError('Either completed predecessor changed after validation')
    for zone, audit in source_audits.items():
        for record in audit['daily_checkpoints'].values():
            if sha(Path(record['source_path'])) != record['file_sha256']:
                raise ValueError('A pinned predecessor raw checkpoint changed')


def run(*, action='validate', zones=('DE','NL'), threads=2, workers=4, min_free_memory_gb=3.0):
    _validate_scope(action,zones,threads,workers,min_free_memory_gb)
    # This MUST precede prepare, pool construction, directory creation and locks.
    sources = predecessor_gate()
    prepared = {}; payloads = {}; source_audits = {}
    for zone in SOURCE_RUNS:
        prepared[zone] = original.prepare(zone, threads=2)
        payloads[zone], source_audits[zone] = load_source_checkpoints(prepared[zone], sources[zone])
    identities = {zone:make_identity(prepared[zone],sources[zone],source_audits[zone],
        workers,min_free_memory_gb,sources) for zone in zones}
    _verify_source_seals(sources,source_audits)
    for zone in zones:
        print(json.dumps({'zone':zone,'validation':'PASS','identity':identities[zone]['run_id'],
            'predecessor':sources[zone]['identity']['run_id'],'new_catboost_fits':0}),flush=True)
    if action == 'validate':
        return identities
    safe_path(OUTPUT); OUTPUT.mkdir(parents=True,exist_ok=True)
    workdirs = {zone:OUTPUT/zone.lower()/identity['run_id'] for zone,identity in identities.items()}
    pointer = OUTPUT/('latest_'+'_'.join(zones)+'.json'); results={}
    with exclusive_process_lock(original.OUTPUT/'batch.lock'), exclusive_process_lock(prior_parallel.OUTPUT/'batch.lock'), exclusive_process_lock(predecessor.OUTPUT/'batch.lock'), exclusive_process_lock(OUTPUT/'batch.lock'), threadpool_limits(limits=2):
        _verify_source_seals(sources,source_audits)
        for zone in zones:
            item=prepared[zone]; identity=identities[zone]
            directory=safe_path(workdirs[zone]);directory.mkdir(parents=True,exist_ok=True)
            if (directory/'experiment.json').exists() and json.loads((directory/'experiment.json').read_text(encoding='utf-8'))!=identity:
                raise ValueError('Unbounded experiment identity changed')
            if _verify_completion(directory,identity):
                results[zone]=str(directory/'report.html');continue
            write_json(directory/'experiment.json',identity)
            current_phase=None;phase_start=time.monotonic();phase_utc=None;last_state={};last_emit=0.0
            def progress(phase,completed=0,total=1,**extra):
                nonlocal current_phase,phase_start,phase_utc,last_state,last_emit
                now=datetime.now(timezone.utc).isoformat();tick=time.monotonic()
                if phase!=current_phase:
                    current_phase=phase;phase_start=tick;phase_utc=now
                state={'status':'RUNNING','phase':phase,'zone':zone,'identity':identity['run_id'],'pid':os.getpid(),
                    'updated_utc':now,'phase_started_utc':phase_utc,'phase_elapsed_seconds':tick-phase_start,
                    'new_catboost_fits':0,
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
                progress('assemble_unbounded_sources',0,731)
                frames,correction_audit=assemble_unbounded(item,payloads[zone])
                write_json(directory/'source_audit.json',source_audits[zone])
                write_json(directory/'unbounded_corrections_audit.json',{'identity':identity['run_id'],
                    'days':len(correction_audit),'new_catboost_fits':0,'corrector_clipping_applied':False,
                    'kalman_constraints_unchanged':True,'daily':correction_audit})
                write_frame(directory/'baseline_backtest.parquet',indexed(item.bundle.kalman_view.backtest))
                write_frame(directory/'baseline_forecast.parquet',indexed(item.bundle.kalman_view.forecast))
                progress('assemble_unbounded_sources',731,731)
                backtests,futures=_kalman_stage(item,identity,directory,frames,progress)
                progress('verify_and_report')
                checked=original.prepare(zone,threads=2)
                if checked.identity!=item.identity or make_identity(checked,sources[zone],source_audits[zone],
                        workers,min_free_memory_gb,sources)!=identity:
                    raise ValueError('Scientific sources or unbounded implementation changed')
                _verify_source_seals(sources,source_audits)
                for day,payload in payloads[zone].items():
                    predecessor.verify_origin(checked,sources[zone]['identity'],day,_source_base(checked,day).index,payload)
                metrics=original.evaluate(backtests,original.TIMEZONES[zone])
                write_json(directory/'metrics.json',metrics);_report(directory,identity,metrics,futures,correction_audit)
                write_json(directory/'completion.json',{'identity':identity['run_id'],'annual_complete':True,
                    'files':{name:sha(directory/name) for name in RESULT_FILES}})
                state={'status':'COMPLETE','phase':'complete','zone':zone,'identity':identity['run_id'],
                    'annual_complete':True,'evaluation_days':365,'evaluation_hours':8760,
                    'future_forecast_hours':24,'variants':list(VARIANTS),'new_catboost_fits':0,
                    'updated_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid()}
                write_json(directory/'status.json',state);write_json(directory/'progress.json',state)
                results[zone]=str(directory/'report.html')
            except Paused as exc:
                state={**last_state,'status':'PAUSED','zone':zone,'identity':identity['run_id'],'pid':os.getpid(),
                    'reason':str(exc),'automatic_resume':False,'updated_utc':datetime.now(timezone.utc).isoformat()}
                write_json(directory/'status.json',state);write_json(directory/'progress.json',state)
                write_json(pointer,{'engine':ENGINE,'status':'PAUSED','active_zone':zone,'workdirs':workdirs,
                    'results':results,'reason':str(exc),'automatic_resume':False,'production_modified':False})
                return {'status':'PAUSED','reason':str(exc),'results':results}
            except Exception as exc:
                state={**last_state,'status':'FAILED','zone':zone,'identity':identity['run_id'],'pid':os.getpid(),
                    'error':f'{type(exc).__name__}: {exc}','updated_utc':datetime.now(timezone.utc).isoformat()}
                write_json(directory/'status.json',state);write_json(directory/'progress.json',state)
                write_json(pointer,{'engine':ENGINE,'status':'FAILED','failed_zone':zone,'workdirs':workdirs,
                    'results':results,'error':str(exc),'production_modified':False})
                raise
        write_json(pointer,{'engine':ENGINE,'status':'COMPLETE','annual_complete':True,
            'workdirs':workdirs,'results':results,'new_catboost_fits':0})
        links=''.join('<li><a href="'+html.escape(Path(path).relative_to(OUTPUT).as_posix())+'">'+zone+'</a></li>' for zone,path in results.items())
        write_text(OUTPUT/'index.html','<!doctype html><meta charset="utf-8"><title>NYX · correcteurs sans bornes</title><h1>DE/NL · résultats sans plafonds du correcteur</h1><p>Kalman inchangé, hors production.</p><ul>'+links+'</ul>')
    return results

