"""Isolated, paired CatBoost interaction/positive-cap ablation on frozen Chronos.

No production/configuration/source writes and no Chronos inference are used.
Baseline raw forecasts and prepared inputs must reproduce their sealed hashes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import hashlib
import html
import importlib.metadata
import json
import os
import time

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from .solar_wind_interaction import (BASELINES, DAY, ROOT, TIMEZONES, indexed,
    json_bytes, sha, restore_covariate_config, scores, baseline_controls)
from .solar_wind_interaction_features import build_interaction
from .kalman_residual import KalmanResidualConfig, replay_kalman_overlay
from .nuclear_run_archive import load_nuclear_result_bundle
from .models.residual_corrector import apply_residual_correction
from .process_lock import exclusive_process_lock

ENGINE = 'solar_wind_corrector_interaction_v1'
OUTPUT = ROOT / 'runs' / 'experiments' / ENGINE / DAY
VARIANTS = {'interaction_40': (True, 40), 'cap_80': (False, 80), 'interaction_80': (True, 80)}
RESULT_FILES = ('experiment.json', 'baseline_controls.json', 'interaction.parquet', 'feature_audit.json',
    'corrector_audit.json', 'metrics.json', 'report.html',
    *(name + '/' + f for name in VARIANTS for f in ('upstream_history.parquet', 'upstream_forecast.parquet',
        'backtest.parquet', 'forecast.parquet', 'replay_audit.json')))
QUANTILES = ('q10', 'q50', 'q90')
FINAL_COLUMNS = ['residual_kalman__' + q for q in QUANTILES]
IMPLEMENTATION = ('run_solar_wind_corrector_interaction.py',
    'chronos2_hourly/solar_wind_corrector_interaction.py',
    'chronos2_hourly/solar_wind_interaction_features.py',
    'chronos2_hourly/solar_wind_interaction.py',
    'chronos2_hourly/nuclear_daily_cache.py', 'chronos2_hourly/nuclear_run_archive.py',
    'chronos2_modular/common.py', 'chronos2_modular/exogenous_extensions.py')


def safe_path(path):
    path = Path(path).absolute()
    if path.resolve() != path or not path.is_relative_to(OUTPUT.absolute()):
        raise ValueError(f'Unsafe or redirected experiment output: {path}')
    if path.is_dir():
        for child in path.rglob('*'):
            if child.is_symlink() or (getattr(child.lstat(), 'st_file_attributes', 0) & 0x400):
                raise ValueError(f'Redirected experiment descendant: {child}')
    return path


def write_json(path, value):
    path = safe_path(path)
    temporary = safe_path(path.with_suffix(path.suffix + '.tmp'))
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def write_frame(path, frame):
    path = safe_path(path)
    temporary = safe_path(path.with_suffix(path.suffix + '.tmp'))
    frame.to_parquet(temporary)
    temporary.replace(path)


def apply_variant(base, raw_correction, *, upper=40, scale=1.0):
    if upper not in (40, 80):
        raise ValueError('Only the predeclared +40/+80 upper caps are allowed')
    return apply_residual_correction(base, raw_correction, correction_scale=scale,
                                     correction_clip=(-40.0, float(upper)), max_abs_correction=None)


def add_corrector_feature(features, feature):
    if (not features.index.equals(feature.index) or len(feature.columns) != 1
            or features.columns.has_duplicates or feature.columns.has_duplicates
            or set(features).intersection(feature)):
        raise ValueError('One unique, exactly aligned new corrector feature is required')
    values = feature.to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Interaction must be finite and in [0, 1]')
    return pd.concat([features.copy(deep=True), feature.copy(deep=True)], axis=1)


def align_interaction_prefix(feature, index):
    """Zero only earlier uncalibrated raw days, never an internal/future gap."""
    if feature.empty or len(feature.columns) != 1 or feature.index.has_duplicates:
        raise ValueError('Nonempty unique single interaction feature required')
    missing = index.difference(feature.index)
    if len(missing) and (missing >= feature.index[0]).any():
        raise ValueError('Interaction has a missing in-support or future hour')
    return feature.reindex(index, fill_value=0.0)


def train_window(index, day, timezone):
    day = pd.Timestamp(day).date()
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError('Unique ordered timezone-aware training index required')
    days = index.tz_convert(timezone).date
    return index[(days >= day - timedelta(days=365)) & (days < day)]


def verify_day_checkpoint(path, identity, day, prediction_index):
    path = safe_path(path)
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding='utf-8'))
    payload = record['payload']
    if record.get('sha256') != hashlib.sha256(json_bytes(payload)).hexdigest():
        raise ValueError('Daily corrector checkpoint checksum mismatch')
    if (payload.get('identity') != identity or payload.get('day') != str(day)
            or payload.get('index_ns') != prediction_index.asi8.tolist()
            or set(payload.get('raw', {})) != {'original', 'interaction'}):
        raise ValueError('Daily corrector checkpoint identity/index mismatch')
    for values in payload['raw'].values():
        if len(values) != len(prediction_index) or not np.isfinite(np.asarray(values, dtype=float)).all():
            raise ValueError('Invalid daily raw corrections')
    return payload


@dataclass
class Prepared:
    bundle: object
    config: object
    covconfig: object
    features: pd.DataFrame
    interaction: pd.DataFrame
    feature_audit: dict
    raw_history: pd.DataFrame
    raw_future: pd.DataFrame
    identity: dict
    factory: object


def _read_prepared(work, bundle, resolved, zone):
    """CSV is accepted only after exact original in-memory digest reproduction."""
    from chronos2_modular.common import CALENDAR_COLUMNS
    from .nuclear_forecast import _digest_frame, _digest_json
    from run_chronos2_hourly import _feature_inputs

    def read_csv(path):
        value = pd.read_csv(path, float_precision='round_trip')
        value.index = pd.DatetimeIndex(pd.to_datetime(value.pop('timestamp'), utc=True))
        return value

    context = read_csv(work / 'prepared/model_covariates_with_future.csv.gz').astype(np.float32)
    target = read_csv(work / 'prepared/selected_pit/target_context.csv')['value'].astype(np.float32)
    target.name = 'target'
    # ZoneData is civil-time indexed: future_proxy_frame builds its calendar
    # from that index before the adapter converts prediction timestamps to UTC.
    context.index = context.index.tz_convert(TIMEZONES[zone])
    target.index = target.index.tz_convert(TIMEZONES[zone])
    known = [c for c in context if c in CALENDAR_COLUMNS or c.startswith('known_')]
    aliases = [c for c in context if c not in known]
    data = SimpleNamespace(target=target, model_context_covariates=context,
        known_future_columns=known, timezone=TIMEZONES[zone], zone=zone, frequency='h',
        covariates=context.loc[target.index, aliases].copy())
    converted_target, _, _, features = _feature_inputs(data, resolved)
    observed = {'target': _digest_frame(converted_target),
        'model_context_covariates': _digest_frame(context), 'residual_features': _digest_frame(features),
        'known_future_columns': _digest_json(known)}
    for name, digest in observed.items():
        if digest != bundle.audit['source_hashes'][name]:
            raise ValueError(f'Original prepared inputs do not reproduce sealed digest: {zone}/{name}')
    return data, features, observed


def _raw_inputs(work, bundle, resolved, data, zone):
    """Read the exact pre-730-day Chronos prefix, never the rounded CSV checkpoint."""
    from .nuclear_daily_cache import NuclearDailyChronosCache
    from .chronos_adapter import generate_delivery_plans
    from .nuclear_forecast import _load_result_cache

    frozen = indexed(bundle.raw_history)
    start = pd.Timestamp(bundle.audit['raw_history_start_day']).date()
    end = frozen.index[0].tz_convert(TIMEZONES[zone]).date() - timedelta(days=1)
    cache_dir = Path(bundle.audit['incremental_cache_directory']) / 'chronos'
    allowed = ROOT / 'runs/experiments/solar_wind_v1/_daily_cache'
    if cache_dir.resolve() != cache_dir or not cache_dir.is_relative_to(allowed):
        raise ValueError('Original Chronos cache must remain in its isolated namespace')
    loader = NuclearDailyChronosCache(cache_dir, data=data, config=resolved,
        context_length=int(resolved['model']['context_length']), device='cpu',
        model_batch_size=int(resolved['model'].get('model_batch_size', 128)),
        origin_batch_size=int(resolved['model'].get('origin_batch_size', 12)),
        execution_signature={'threads': int(bundle.audit['residual_recipe']['thread_count'])})
    prefix = []
    receipts = {}
    plans = generate_delivery_plans(start, end, timezone=TIMEZONES[zone], forecast_origin_local_time='08:00') if start <= end else []
    for plan in plans:
        frame = loader.load(plan, historical=True)
        if frame is None:
            raise ValueError(f'Exact sealed Chronos prefix unavailable: {zone}/{plan.delivery_date}')
        prefix.append(indexed(frame))
        path = loader._path(loader.identity(plan))
        receipts[str(path)] = sha(path)
    # Also prove the prefix reader reproduces the first visible frozen day.
    first_day = frozen.index[0].tz_convert(TIMEZONES[zone]).date()
    plan = generate_delivery_plans(first_day, first_day, timezone=TIMEZONES[zone], forecast_origin_local_time='08:00')[0]
    overlap = loader.load(plan, historical=True)
    if overlap is None:
        raise ValueError('Cannot verify Chronos prefix reader against sealed overlap')
    overlap = indexed(overlap)
    for column in (*QUANTILES, 'actual', 'forecast_origin_utc'):
        if not np.array_equal(overlap[column].to_numpy(), frozen.loc[overlap.index, column].to_numpy()):
            raise ValueError(f'Chronos prefix reader differs from frozen overlap: {column}')
    receipts[str(loader._path(loader.identity(plan)))] = sha(loader._path(loader.identity(plan)))
    raw = pd.concat([*prefix, frozen]) if prefix else frozen.copy(deep=True)
    raw = indexed(raw)
    result_root = Path(bundle.audit['result_cache_directory'])
    if result_root.resolve() != result_root or not result_root.is_relative_to(work / 'cache/nuclear_result'):
        raise ValueError('Unexpected original result cache path')
    manifest = json.loads((result_root / 'raw_future/cache_manifest.json').read_text(encoding='utf-8'))
    expected_identity = {'delivery_day': DAY, 'replay_identity': manifest['identity']['replay_identity']}
    if expected_identity['replay_identity']['source_hashes'] != bundle.audit['source_hashes']:
        raise ValueError('Raw future cache source identity differs from frozen result')
    cache = _load_result_cache(result_root / 'raw_future', expected_identity)
    if cache is None or set(cache) != {'raw_future'}:
        raise ValueError('Sealed raw future cache absent')
    future = indexed(cache['raw_future'])
    old_future = indexed(bundle.source_forecast)
    if not future.index.equals(old_future.index):
        raise ValueError('Raw future hours differ from frozen forecast')
    if not np.array_equal(future.forecast_origin_utc.to_numpy(), old_future.forecast_origin_utc.to_numpy()):
        raise ValueError('Raw future forecast origins differ from frozen forecast')
    for q in QUANTILES:
        if not np.array_equal(future[q].to_numpy(), old_future['chronos2__' + q].to_numpy()):
            raise ValueError('Raw future differs from frozen Chronos')
    if 'actual' in future and future.actual.notna().any():
        raise ValueError('Future labels are forbidden')
    for name in ('cache_manifest.json', 'raw_future.parquet'):
        path = result_root / 'raw_future' / name
        receipts[str(path)] = sha(path)
    return raw, future, receipts


def prepare(zone, threads=2):
    """Validate and load immutable inputs; no directory/file creation or fitting."""
    from run_chronos2_hourly import _residual_corrector_factory
    from .nuclear_forecast import _validate_origins, _digest_frame

    if zone not in BASELINES or threads not in (1, 2):
        raise ValueError('DE/NL and at most two threads are required')
    work = ROOT / 'runs/experiments/solar_wind_v1' / DAY / zone.lower() / BASELINES[zone]
    manifest_path = work / 'report_only/frozen_result/manifest.json'
    manifest_digest = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    pinned = manifest['identity']['sources']['snapshot_identity']['scientific_identity']
    for name, digest in pinned['files'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'Baseline scientific implementation changed: {name}')
    for name, version in pinned['dependencies'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Baseline dependency changed: {name}')
    bundle = load_nuclear_result_bundle(workdir=work)
    resolved = yaml.safe_load((work / 'resolved_config.yaml').read_text(encoding='utf-8'))
    data, features, prepared_hashes = _read_prepared(work, bundle, resolved, zone)
    raw, future, prefix_receipts = _raw_inputs(work, bundle, resolved, data, zone)
    _validate_origins(raw, TIMEZONES[zone]); _validate_origins(future, TIMEZONES[zone])
    replay = bundle.kalman_view.replay.audit
    if importlib.metadata.version('pykalman') != replay['pykalman_version']:
        raise ValueError('Kalman dependency differs from frozen baseline')
    config = KalmanResidualConfig(**dict(replay['config'], candidate_kinds=tuple(replay['config']['candidate_kinds'])))
    covconfig = restore_covariate_config(replay['covariate_config'])
    frozen_covariates = indexed(bundle.covariates)
    feature, feature_audit = build_interaction(frozen_covariates, zone=zone, timezone=TIMEZONES[zone])
    # Keep precisely the previous experiment's normalization and feature values.
    # Extra raw Chronos prefix dates precede the available calibration window.
    needed = raw.index.append(future.index)
    feature = align_interaction_prefix(feature, needed)
    feature_audit = dict(feature_audit, pre_calibration_prefix_hours=int((needed < frozen_covariates.index[0]).sum()),
        pre_calibration_prefix_policy='zero_no_available_calibration', feature_location='corrector_only',
        kalman_covariates_unchanged=True)
    features = features.loc[needed].copy()
    add_corrector_feature(features, feature)
    if set(feature).intersection(covconfig.input_columns):
        raise ValueError('The interaction must not enter Kalman')
    recipe = dict(bundle.audit['residual_recipe'])
    if recipe.get('backend') != 'catboost' or recipe.get('max_abs_correction') != 40.0 or recipe.get('correction_scale') != 1.0:
        raise ValueError('Expected unchanged CatBoost baseline ±40 recipe')
    recipe['thread_count'] = threads
    factory, upstream = _residual_corrector_factory({'hourly': {'residual_correction': recipe}}, timezone=TIMEZONES[zone])
    if factory is None or upstream != 'chronos2' or factory().correction_bounds_ != (-40.0, 40.0):
        raise ValueError('Unexpected corrector recipe')
    files = {name: sha(ROOT / name) for name in (*pinned['files'], *IMPLEMENTATION)}
    identity = {'engine': ENGINE, 'zone': zone, 'delivery_day': DAY,
        'baseline_workdir': str(work), 'baseline_manifest_sha256': manifest_digest,
        'scientific_files': files, 'prepared_source_hashes': prepared_hashes,
        'prefix_receipts': prefix_receipts, 'raw_history_sha256': _digest_frame(raw),
        'raw_future_sha256': _digest_frame(future), 'interaction_sha256': _digest_frame(feature),
        'dependencies': {n: importlib.metadata.version(n) for n in ('catboost', 'numpy', 'pandas', 'pykalman', 'scipy', 'joblib', 'scikit-learn', 'holidays')},
        'variants': {n: {'interaction': use, 'correction_clip': [-40, upper]} for n, (use, upper) in VARIANTS.items()},
        'residual_recipe': recipe, 'kalman_config': replay['config'], 'kalman_covariate_config': replay['covariate_config'],
        'feature': str(feature.columns[0]), 'normalization': 'previous_Kalman_interaction_values_bitwise_same_with_zero_precalibration_prefix',
        'fit_rule': 'D-365 <= civil_day < D; unchanged MAE; two fits then four clips',
        'evaluation_start_day': '2025-09-22', 'evaluation_end_day': '2026-09-21',
        'evaluation_days': 365, 'production_modified': False, 'promotion_eligible': False,
        'pit_publication_evidence_verified': False, 'prospective_validation': False}
    identity['run_id'] = hashlib.sha256(json_bytes(identity)).hexdigest()[:16]
    if sha(manifest_path) != manifest_digest:
        raise ValueError('Frozen manifest changed during preparation')
    return Prepared(bundle, config, covconfig, features, feature, feature_audit, raw, future, identity, factory)


def _fit_raw(prepared, day, *, interaction):
    zone = prepared.identity['zone']; tz = TIMEZONES[zone]
    history, future = prepared.raw_history, prepared.raw_future
    train = train_window(history.index, day, tz)
    is_future = str(day) == DAY
    selected = future if is_future else history.loc[history.index.tz_convert(tz).date == pd.Timestamp(day).date()]
    base = selected.loc[:, list(QUANTILES)].astype(float)
    model = prepared.factory()
    if len(train) < model.min_training_rows:
        return np.zeros(len(base)), {'training_rows': len(train), 'generation_source': 'identity_chronos_cold_start', 'features': []}
    X = add_corrector_feature(prepared.features, prepared.interaction) if interaction else prepared.features
    train_base = history.loc[train, list(QUANTILES)].astype(float)
    model.fit(X.loc[train], history.loc[train, 'actual'].astype(float), train_base,
        train_base.rename(columns={q: 'chronos2__' + q for q in QUANTILES}))
    predicted = model.predict_correction(X.loc[base.index], base,
        base.rename(columns={q: 'chronos2__' + q for q in QUANTILES}))
    if interaction and prepared.identity['feature'] not in model.feature_columns_:
        raise ValueError('The fitted corrector dropped the interaction')
    raw = predicted.attrs['raw_correction'].to_numpy(float)
    return raw, {'training_rows': len(train), 'training_first_utc': str(train[0]), 'training_last_utc': str(train[-1]),
        'generation_source': 'daily_prequential_refit', 'features': list(model.feature_columns_),
        'causality_violations': 0, 'loss': 'MAE'}


def _compare_corrected(actual, expected):
    if not actual.index.equals(expected.index):
        raise ValueError('Baseline corrector index differs')
    for q in QUANTILES:
        if not np.allclose(actual[q], expected['residual_corrected__' + q], rtol=0, atol=1e-9):
            delta = float(np.max(np.abs(actual[q] - expected['residual_corrected__' + q])))
            raise ValueError(f'Baseline corrector reproduction differs: {q}, max error {delta}')


def corrector_controls(prepared, progress=None):
    zone = prepared.identity['zone']; tz = TIMEZONES[zone]
    frozen = indexed(prepared.bundle.kalman_view.backtest)
    days = list(pd.Index(frozen.index.tz_convert(tz).date).unique())
    results, raw_cache = [], {}
    daily = prepared.bundle.residual_daily_audit
    first_fitted = pd.Timestamp(daily.loc[daily.generation_source.eq('daily_prequential_refit'), 'delivery_day'].iloc[0]).date()
    controls = tuple(dict.fromkeys((first_fitted, days[0], days[len(days)//2], days[-1], pd.Timestamp(DAY).date())))
    for position, day in enumerate(controls):
        if progress:
            progress('baseline_corrector_controls', position, len(controls), day=str(day))
        started = time.monotonic()
        raw, audit = _fit_raw(prepared, day, interaction=False)
        future = str(day) == DAY
        base = prepared.raw_future if future else prepared.raw_history.loc[prepared.raw_history.index.tz_convert(tz).date == day]
        expected = indexed(prepared.bundle.source_forecast if future else prepared.bundle.residual_statistics).loc[base.index]
        _compare_corrected(apply_variant(base.loc[:, list(QUANTILES)], raw), expected)
        raw_cache[str(day)] = (raw, audit)
        results.append({'day': str(day), 'matched': True, 'tolerance': 1e-9, 'future': future,
            'elapsed_seconds': time.monotonic() - started})
    return results, raw_cache


def replay_correctors(prepared, directory, raw_cache, progress):
    tz = TIMEZONES[prepared.identity['zone']]
    historical = indexed(prepared.bundle.residual_statistics)
    future = indexed(prepared.bundle.source_forecast)
    frames = {name: (historical.copy(deep=True), future.copy(deep=True)) for name in VARIANTS}
    days = list(pd.Index(historical.index.tz_convert(tz).date).unique()) + [pd.Timestamp(DAY).date()]
    cache_dir = safe_path(directory / 'corrector_daily'); cache_dir.mkdir(exist_ok=True)
    records = []
    for number, day in enumerate(days):
        progress('corrector_refits', number, len(days), day=str(day), unit='jours / deux recettes')
        is_future = str(day) == DAY
        selected = prepared.raw_future if is_future else prepared.raw_history.loc[prepared.raw_history.index.tz_convert(tz).date == day]
        base = selected.loc[:, list(QUANTILES)].astype(float)
        path = cache_dir / (str(day) + '.json')
        checkpoint = verify_day_checkpoint(path, prepared.identity['run_id'], day, base.index)
        if checkpoint is None:
            original, original_audit = raw_cache.get(str(day), (None, None))
            if original is None:
                original, original_audit = _fit_raw(prepared, day, interaction=False)
            changed, interaction_audit = _fit_raw(prepared, day, interaction=True)
            checkpoint = {'identity': prepared.identity['run_id'], 'day': str(day), 'index_ns': base.index.asi8.tolist(),
                'raw': {'original': original.tolist(), 'interaction': changed.tolist()},
                'audit': {'original': original_audit, 'interaction': interaction_audit}}
            write_json(path, {'payload': checkpoint, 'sha256': hashlib.sha256(json_bytes(checkpoint)).hexdigest()})
        expected = future if is_future else historical.loc[base.index]
        _compare_corrected(apply_variant(base, checkpoint['raw']['original']), expected)
        for name, (use, upper) in VARIANTS.items():
            corrected = apply_variant(base, checkpoint['raw']['interaction' if use else 'original'], upper=upper)
            destination = frames[name][1 if is_future else 0]
            for q in QUANTILES:
                destination.loc[base.index, 'residual_corrected__' + q] = corrected[q]
                if is_future:
                    destination.loc[base.index, q] = corrected[q]
            destination.loc[base.index, 'residual_correction'] = corrected['q50'] - base['q50']
            if is_future:
                destination.loc[base.index, 'price_eur_mwh'] = corrected['q50']
        clipping = {recipe: {'hours': len(values), 'raw_gt_40': int((np.asarray(values) > 40).sum()),
            'raw_gt_80': int((np.asarray(values) > 80).sum()), 'raw_lt_minus40': int((np.asarray(values) < -40).sum())}
            for recipe, values in checkpoint['raw'].items()}
        records.append({'day': str(day), **checkpoint['audit'], 'baseline_reproduced': True, 'clipping': clipping})
    progress('corrector_refits', len(days), len(days), unit='jours / deux recettes')
    return frames, records


def replay_variant(prepared, name, frames, directory, progress):
    """One independent rolling365 Kalman day at a time, with sealed checkpoints."""
    zone = prepared.identity['zone']; tz = TIMEZONES[zone]
    history, forecast = frames
    days = list(pd.Index(indexed(prepared.bundle.kalman_view.backtest).index.tz_convert(tz).date).unique())
    day_dir = safe_path(directory / name / 'kalman_daily'); day_dir.mkdir(parents=True, exist_ok=True)
    outputs, audits, final = [], [], None
    for number, day in enumerate(days):
        progress('kalman_' + name, number, 366, day=str(day), unit='jours + prévision')
        result_path = day_dir / (str(day) + '.parquet')
        receipt_path = day_dir / (str(day) + '.json')
        future_path = day_dir / 'forecast.parquet'
        last = day == days[-1]
        if receipt_path.exists():
            sealed = json.loads(receipt_path.read_text(encoding='utf-8'))
            receipt = sealed['payload']
            if sealed.get('sha256') != hashlib.sha256(json_bytes(receipt)).hexdigest():
                raise ValueError('Kalman daily checkpoint receipt checksum mismatch')
            if receipt.get('identity') != prepared.identity['run_id'] or receipt.get('variant') != name or receipt.get('day') != str(day):
                raise ValueError('Kalman daily checkpoint identity mismatch')
            if sha(result_path) != receipt['backtest_sha256'] or (last and sha(future_path) != receipt['forecast_sha256']):
                raise ValueError('Kalman daily checkpoint checksum mismatch')
            output = indexed(pd.read_parquet(result_path))
            audit = receipt['audit']
            if last:
                final = indexed(pd.read_parquet(future_path))
        else:
            prefix = history.loc[history.index.tz_convert(tz).date <= day]
            replay = replay_kalman_overlay(prefix.reset_index(), timezone=tz, evaluation_start_day=str(day),
                covariates=prepared.bundle.covariates.copy(deep=True), config=prepared.config,
                covariate_config=prepared.covconfig, training_lookback_days=365, rolling_refit_workers=1,
                future_upstream=forecast.reset_index() if last else None,
                future_covariates=prepared.bundle.covariates.copy(deep=True) if last else None)
            expected_index = history.index[history.index.tz_convert(tz).date == day]
            output = merge_overlay(history.loc[expected_index], replay.predictions.loc[expected_index]); audit = replay.audit
            if last:
                final = merge_overlay(forecast, replay.future_predictions)
                write_frame(future_path, final)
            write_frame(result_path, output)
            receipt = {'identity': prepared.identity['run_id'], 'variant': name, 'day': str(day), 'audit': audit,
                'backtest_sha256': sha(result_path), 'forecast_sha256': sha(future_path) if last else None}
            write_json(receipt_path, {'payload': receipt, 'sha256': hashlib.sha256(json_bytes(receipt)).hexdigest()})
        expected_index = history.index[history.index.tz_convert(tz).date == day]
        if not output.index.equals(expected_index):
            raise ValueError('Kalman daily checkpoint physical hours mismatch')
        validate_replayed_output(output, history.loc[expected_index], future=False)
        if last:
            validate_replayed_output(final, forecast, future=True)
        if audit['causality_violations'] or audit['quantile_crossings'] or audit['future_observations_assimilated']:
            raise ValueError('Kalman replay contract violation')
        if (audit['evaluation_days'] != 1 or audit['evaluation_hours'] != len(expected_index)
                or audit['future_forecast_hours'] != (len(forecast) if last else 0)
                or audit['training_lookback_days'] != 365
                or json_bytes(audit['config']) != json_bytes(prepared.identity['kalman_config'])
                or json_bytes(audit['covariate_config']) != json_bytes(prepared.identity['kalman_covariate_config'])):
            raise ValueError('Kalman daily checkpoint audit differs from protocol')
        outputs.append(output); audits.append(audit)
    combined = pd.concat(outputs)
    if len(combined) != 8760 or final is None or len(final) != 24:
        raise ValueError('Incomplete annual/future Kalman replay')
    progress('kalman_' + name, 366, 366, unit='jours + prévision')
    return combined, final, audits


def merge_overlay(upstream, overlay):
    upstream, overlay = indexed(upstream), indexed(overlay)
    if not upstream.index.equals(overlay.index):
        raise ValueError('Kalman overlay and upstream timestamps differ')
    result = upstream.copy(deep=True)
    for column in overlay:
        if column in upstream and not np.array_equal(upstream[column].to_numpy(), overlay[column].to_numpy()):
            raise ValueError(f'Kalman overlay attempts to change upstream: {column}')
        result[column] = overlay[column]
    return result


def validate_replayed_output(frame, upstream, *, future):
    if frame is None or not frame.index.equals(upstream.index):
        raise ValueError('Replayed output must exactly match upstream hours')
    values = frame[FINAL_COLUMNS].to_numpy(float)
    if not np.isfinite(values).all() or (np.diff(values, axis=1) < 0).any():
        raise ValueError('Nonfinite or crossed replayed quantiles')
    if future and 'actual' in frame and frame.actual.notna().any():
        raise ValueError('Future observations are forbidden')
    for column in upstream:
        if (column == 'actual' and not future) or column == 'residual_correction' or column.startswith(('chronos2__', 'residual_corrected__')) or 'forecast_origin' in column:
            if column not in frame or not np.array_equal(frame[column].to_numpy(), upstream[column].to_numpy()):
                raise ValueError(f'Replayed upstream changed: {column}')


def evaluate(frames, timezone):
    baseline = frames['baseline']; actual = baseline.actual.to_numpy(float)
    predictions = {}
    for name, frame in frames.items():
        if not baseline.index.equals(frame.index) or not np.array_equal(actual, frame.actual.to_numpy(float)):
            raise ValueError('Paired observations/index changed')
        values = frame[FINAL_COLUMNS].to_numpy(float)
        if not np.isfinite(values).all() or (np.diff(values, axis=1) < 0).any():
            raise ValueError('Nonfinite or crossed final quantiles')
        predictions[name] = values[:, 1]
    months = baseline.index.tz_convert(timezone).strftime('%Y-%m')
    masks = {'all': np.ones(len(actual), dtype=bool), 'actual_ge_200': actual >= 200, 'actual_ge_300': actual >= 300,
        **{m: months == m for m in sorted(set(months))}}
    result = {'slices': {label: {name: scores(actual[mask], pred[mask]) for name, pred in predictions.items()}
                        for label, mask in masks.items()}, 'spikes': {}}
    for threshold in (200, 300):
        result['spikes'][str(threshold)] = {}
        observed = actual >= threshold
        for name, pred in predictions.items():
            predicted = pred >= threshold
            tp, fp, fn = int((observed & predicted).sum()), int((~observed & predicted).sum()), int((observed & ~predicted).sum())
            result['spikes'][str(threshold)][name] = {'tp': tp, 'fp': fp, 'fn': fn,
                'precision': tp/(tp+fp) if tp+fp else None, 'recall': tp/(tp+fn) if tp+fn else None}
    return result


def render_report(directory, zone, metrics, identity, futures):
    esc = html.escape
    def number(value):
        return '—' if value is None else f'{value:.3f}'
    rows = ''.join('<tr><td>'+esc(label)+'</td><td>'+esc(name)+'</td><td>'+str(v['hours'])+'</td>'+''.join('<td>'+number(v[k])+'</td>' for k in ('mae','rmse','bias'))+'</tr>'
        for label, variants in metrics['slices'].items() for name, v in variants.items())
    future_rows = ''.join('<tr><td>'+esc(str(hour))+'</td>'+''.join('<td>'+number(frame.loc[hour, FINAL_COLUMNS[1]])+'</td>' for frame in futures.values())+'</tr>' for hour in futures['baseline'].index)
    text = '<!doctype html><html lang="fr"><meta charset="utf-8"><title>NYX correcteur '+zone+'</title><style>body{background:#090a0e;color:#e6e6ed;font:15px system-ui;max-width:1200px;margin:40px auto;padding:20px}h1,h2{color:#efb35b}table{width:100%;border-collapse:collapse;margin:20px 0}th,td{padding:8px;border-bottom:1px solid #292b33;text-align:right}td:first-child,td:nth-child(2){text-align:left}pre{white-space:pre-wrap;font-size:12px}a{color:#73cbd0}</style><h1>NYX · SolarWind · correcteur '+zone+'</h1><p>Expérience isolée : interaction uniquement dans CatBoost ; plafonds [−40,+40] ou [−40,+80] €/MWh. Chronos figé. Recette MAE, gouvernance et entrées Kalman inchangées ; Kalman réentraîné après chaque nouvel amont.</p><p>365 jours du 22/09/2025 au 21/09/2026, prévision du 22/09/2026 séparée. Hypothèse rétrospective post-hoc, non validation prospective. Publication d’origine PIT non certifiée ; substitutions NL héritées. Aucune promotion ni modification de production.</p><h2>Erreurs appariées (€/MWh)</h2><table><tr><th>Période</th><th>Variante</th><th>Heures</th><th>MAE</th><th>RMSE</th><th>Biais</th></tr>'+rows+'</table><h2>Pics et faux pics</h2><pre>'+esc(json_bytes(metrics['spikes']).decode())+'</pre><h2>Prévision distincte (UTC)</h2><table><tr><th>Heure</th>'+''.join('<th>'+esc(n)+' Q50</th>' for n in futures)+'</tr>'+future_rows+'</table><details><summary>Identité et protocole</summary><pre>'+esc(json_bytes(identity).decode())+'</pre></details></html>'
    counts = '<h2>Corrections brutes et plafonds</h2><p>Comptages séparés pour les 365 jours et la prévision ; le plafond ne modifie pas la cible MAE.</p><pre>' + esc(json_bytes(metrics.get('raw_correction_threshold_counts', {})).decode()) + '</pre>'
    text = text.replace('<h2>Prévision distincte', counts + '<h2>Prévision distincte')
    safe_path(directory / 'report.html').write_text(text, encoding='utf-8')


def run(*, action='validate', zones=('DE', 'NL'), threads=2, workers=2):
    if action not in ('validate', 'run') or not zones or len(zones) != len(set(zones)) or not set(zones) <= set(BASELINES):
        raise ValueError('Expected validate/run and unique nonempty DE/NL zones')
    if threads not in (1, 2) or workers not in (1, 2):
        raise ValueError('CPU limits exceeded')
    prepared = {zone: prepare(zone, threads=threads) for zone in zones}
    for zone, item in prepared.items():
        print(json.dumps({'zone': zone, 'validation': 'PASS', 'identity': item.identity['run_id']}), flush=True)
    if action == 'validate':
        return {zone: item.identity for zone, item in prepared.items()}
    safe_path(OUTPUT); OUTPUT.mkdir(parents=True, exist_ok=True)
    workdirs = {zone: OUTPUT / zone.lower() / item.identity['run_id'] for zone, item in prepared.items()}
    pointer = OUTPUT / ('latest_' + '_'.join(zones) + '.json')
    with exclusive_process_lock(OUTPUT / 'batch.lock'), threadpool_limits(limits=threads):
        results = {}
        for zone, item in prepared.items():
            directory = safe_path(workdirs[zone]); directory.mkdir(parents=True, exist_ok=True)
            identity_path = directory / 'experiment.json'
            if identity_path.exists() and json.loads(identity_path.read_text(encoding='utf-8')) != item.identity:
                raise ValueError('Existing experiment identity differs')
            receipt_path = directory / 'completion.json'
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
                state = json.loads((directory / 'status.json').read_text(encoding='utf-8'))
                if (receipt.get('identity') != item.identity['run_id'] or receipt.get('annual_complete') is not True
                        or set(receipt.get('files', {})) != set(RESULT_FILES)
                        or state.get('status') != 'COMPLETE' or state.get('identity') != item.identity['run_id']
                        or state.get('zone') != zone or state.get('annual_complete') is not True or state.get('evaluation_days') != 365
                        or any(sha(safe_path(directory / f)) != digest for f,digest in receipt['files'].items())):
                    raise ValueError('Completed result checksum/identity mismatch')
                results[zone] = str(directory / 'report.html'); continue
            write_json(identity_path, item.identity)
            current_phase = None; phase_started = None
            def progress(phase, completed=0, total=1, **extra):
                nonlocal current_phase, phase_started
                now = datetime.now(timezone.utc).isoformat()
                if current_phase != phase:
                    current_phase, phase_started = phase, now
                state = {'status': 'RUNNING', 'zone': zone, 'identity': item.identity['run_id'], 'pid': os.getpid(),
                    'phase': phase, 'updated_utc': now, 'phase_started_utc': phase_started,
                    'progress': {'completed': completed, 'total': total, 'unit': extra.pop('unit', 'étapes'), 'label': phase}, **extra}
                write_json(directory / 'status.json', state); write_json(directory / 'progress.json', state)
                write_json(pointer, {'engine': ENGINE, 'status': 'RUNNING', 'active_zone': zone, 'pid': os.getpid(),
                    'workdirs': workdirs, 'results': results, 'production_modified': False})
                print(f'{zone} {phase}: {completed}/{total} '+str(extra.get('day','')), flush=True)
            try:
                progress('baseline_corrector_controls', 0, 5)
                controls, raw_cache = corrector_controls(item, progress)
                progress('baseline_kalman_controls', 0, 4)
                kalman_controls = baseline_controls(item.bundle, item.config, item.covconfig, zone, directory / 'control_cache', workers)
                write_json(directory / 'baseline_controls.json', {'corrector': controls, 'kalman': kalman_controls})
                write_frame(directory / 'interaction.parquet', item.interaction)
                write_json(directory / 'feature_audit.json', item.feature_audit)
                frames, records = replay_correctors(item, directory, raw_cache, progress)
                write_json(directory / 'corrector_audit.json', records)
                backtests = {'baseline': indexed(item.bundle.kalman_view.backtest)}
                futures = {'baseline': indexed(item.bundle.kalman_view.forecast)}
                report_files = ['experiment.json', 'baseline_controls.json', 'interaction.parquet', 'feature_audit.json', 'corrector_audit.json', 'metrics.json', 'report.html']
                for name, upstream in frames.items():
                    subdir = safe_path(directory / name); subdir.mkdir(exist_ok=True)
                    write_frame(subdir / 'upstream_history.parquet', upstream[0]); write_frame(subdir / 'upstream_forecast.parquet', upstream[1])
                    backtest, future, audits = replay_variant(item, name, upstream, directory, progress)
                    if 'actual' in future and future.actual.notna().any():
                        raise ValueError('Future target leaked into output')
                    write_frame(subdir / 'backtest.parquet', backtest); write_frame(subdir / 'forecast.parquet', future)
                    write_json(subdir / 'replay_audit.json', audits)
                    backtests[name], futures[name] = backtest, future
                    report_files.extend(name + '/' + f for f in ('upstream_history.parquet', 'upstream_forecast.parquet', 'backtest.parquet', 'forecast.parquet', 'replay_audit.json'))
                progress('verify_and_report')
                if prepare(zone, threads=threads).identity != item.identity:
                    raise ValueError('Code, sources or dependencies changed during run')
                metrics = evaluate(backtests, TIMEZONES[zone])
                metrics['raw_correction_threshold_counts'] = {
                    phase: {recipe: {key: sum(record['clipping'][recipe][key] for record in records
                        if (('2025-09-22' <= record['day'] < DAY) if phase == 'evaluation' else record['day'] == DAY))
                        for key in ('hours', 'raw_gt_40', 'raw_gt_80', 'raw_lt_minus40')}
                        for recipe in ('original', 'interaction')}
                    for phase in ('evaluation', 'forecast')}
                write_json(directory / 'metrics.json', metrics)
                render_report(directory, zone, metrics, item.identity, futures)
                write_json(receipt_path, {'identity': item.identity['run_id'], 'annual_complete': True,
                    'files': {f: sha(directory / f) for f in report_files}})
                completed_state = {'status': 'COMPLETE', 'phase': 'complete', 'zone': zone,
                    'identity': item.identity['run_id'], 'annual_complete': True, 'evaluation_days': 365,
                    'evaluation_hours': 8760, 'future_forecast_hours': 24, 'variants': list(VARIANTS),
                    'updated_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid()}
                write_json(directory / 'status.json', completed_state)
                write_json(directory / 'progress.json', completed_state)
                results[zone] = str(directory / 'report.html')
            except Exception as exc:
                write_json(directory / 'status.json', {'status': 'FAILED', 'zone': zone, 'phase': current_phase,
                    'identity': item.identity['run_id'], 'error': f'{type(exc).__name__}: {exc}', 'pid': os.getpid(),
                    'updated_utc': datetime.now(timezone.utc).isoformat()})
                write_json(pointer, {'engine': ENGINE, 'status': 'FAILED', 'failed_zone': zone, 'workdirs': workdirs,
                    'results': results, 'error': str(exc), 'production_modified': False})
                raise
        write_json(pointer, {'engine': ENGINE, 'status': 'COMPLETE', 'workdirs': workdirs, 'results': results, 'annual_complete': True})
        links = ''.join('<li><a href="'+html.escape(Path(p).relative_to(OUTPUT).as_posix())+'">'+z+'</a></li>' for z,p in results.items())
        safe_path(OUTPUT / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>NYX · interaction correcteur</title><h1>SolarWind · interaction correcteur et plafond positif</h1><p>Expérience rétrospective isolée, hors production.</p><ul>'+links+'</ul>', encoding='utf-8')
    return results
