"""One predeclared low-wind/low-solar/high-residual-load Kalman ablation.

No source sync, Chronos inference, CatBoost fit, or production writer is used.
The archive is consumed as immutable predictions, never as a training resume.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import html
import importlib.metadata
import json
import os

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .kalman_covariates import KalmanCovariateConfig, DerivedCovariateSpec
from .kalman_residual import KalmanResidualConfig, build_operational_kalman_view, replay_kalman_overlay
from .nuclear_run_archive import load_nuclear_result_bundle
from .process_lock import exclusive_process_lock
from .solar_wind_interaction_features import build_interaction

ROOT = Path(__file__).resolve().parents[1]
ENGINE = 'solar_wind_interaction_v1'
DAY = '2026-09-22'
BASELINES = {'DE': '45ec8314c37a2fe5', 'NL': 'b1da388bb4df6c63'}
TIMEZONES = {'DE': 'Europe/Berlin', 'NL': 'Europe/Amsterdam'}
OUTPUT = ROOT / 'runs' / 'experiments' / ENGINE / DAY
QUANTILES = ['residual_kalman__' + q for q in ('q10', 'q50', 'q90')]
IMPLEMENTATION = ('run_solar_wind_interaction.py', 'chronos2_hourly/solar_wind_interaction.py',
    'chronos2_hourly/solar_wind_interaction_features.py', 'chronos2_hourly/kalman_residual.py',
    'chronos2_hourly/kalman_covariates.py', 'chronos2_hourly/nuclear_run_archive.py')
RESULT_FILES = ('experiment.json', 'baseline_controls.json', 'interaction_covariates.parquet', 'feature_audit.json',
    'backtest.parquet', 'forecast.parquet', 'replay_audit.json', 'metrics.json', 'report.html')


def safe_path(path):
    path = Path(path).absolute()
    if path.resolve() != path or not path.is_relative_to(OUTPUT.absolute()):
        raise ValueError(f'Unsafe or redirected output: {path}')
    # Junctions/symlinks in an existing cache tree must not be followed by a writer.
    if path.is_dir():
        for child in path.rglob('*'):
            if child.is_symlink() or (getattr(child.lstat(), 'st_file_attributes', 0) & 0x400):
                raise ValueError(f'Redirected output descendant: {child}')
    return path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x)).encode('utf-8')


def write_json(path, value):
    path = safe_path(path)
    temporary = safe_path(path.with_suffix(path.suffix + '.tmp'))
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def indexed(frame):
    result = frame.copy(deep=True)
    column = next((c for c in ('delivery_start_utc', 'timestamp') if c in result), None)
    index = pd.DatetimeIndex(pd.to_datetime(result.pop(column), utc=True)) if column else result.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None or index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError('Unique increasing timezone-aware physical hours required')
    result.index = index.tz_convert('UTC')
    result.index.name = 'delivery_start_utc'
    return result


def prepare(zone):
    work = ROOT / 'runs/experiments/solar_wind_v1' / DAY / zone.lower() / BASELINES[zone]
    manifest_path = work / 'report_only/frozen_result/manifest.json'
    manifest_digest = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    pinned = manifest['identity']['sources']['snapshot_identity']['scientific_identity']['files']
    for name in ('chronos2_hourly/kalman_residual.py', 'chronos2_hourly/kalman_covariates.py'):
        if sha(ROOT / name) != pinned[name]:
            raise ValueError(f'Baseline scientific implementation changed: {name}')
    baseline = load_nuclear_result_bundle(workdir=work)
    if sha(manifest_path) != manifest_digest:
        raise ValueError('Baseline manifest changed during validation')
    replay = baseline.kalman_view.replay.audit
    if importlib.metadata.version('pykalman') != replay['pykalman_version']:
        raise ValueError('Kalman dependency differs from frozen baseline')
    params = dict(replay['config'])
    params['candidate_kinds'] = tuple(params['candidate_kinds'])
    config = KalmanResidualConfig(**params)
    original = restore_covariate_config(replay['covariate_config'])
    covariates = indexed(baseline.covariates)
    feature, feature_audit = build_interaction(covariates, zone=zone, timezone=TIMEZONES[zone])
    if len(feature.columns) != 1 or not feature.index.equals(covariates.index):
        raise ValueError('Exactly one aligned derived feature required')
    name = feature.columns[0]
    groups = dict(original.groups)
    groups['market'] = (*groups['market'], name)
    candidate_config = replace(original, input_columns=(*original.input_columns, name), groups=groups)
    candidate_config.validate()
    covariates[name] = feature[name]
    identity = {'engine': ENGINE, 'zone': zone, 'delivery_day': DAY,
        'baseline_workdir': str(work), 'baseline_manifest_sha256': sha(manifest_path),
        'scientific_files': {name: sha(ROOT / name) for name in IMPLEMENTATION},
        'dependencies': {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'pykalman', 'scipy', 'joblib')},
        'feature': name, 'variant': 'kalman_market_one_interaction_only',
        'normalization': 'per_delivery_day_strict_past_365_days',
        'kalman_config': replay['config'], 'baseline_covariate_config': replay['covariate_config'],
        'evaluation': 'sealed_365_day_backtest_excludes_delivery_day',
        'promotion_eligible': False, 'production_modified': False}
    identity['run_id'] = hashlib.sha256(json_bytes(identity)).hexdigest()[:16]
    return baseline, config, original, candidate_config, covariates, feature_audit, identity


def restore_covariate_config(audit):
    """The serialized audit uses a list, unlike the YAML loader's mapping."""
    options = dict(audit)
    options['input_columns'] = tuple(options['input_columns'])
    options['groups'] = {key: tuple(value) for key, value in options['groups'].items()}
    options['derived'] = tuple(DerivedCovariateSpec(**dict(item, sources=tuple(item['sources']))) for item in options['derived'])
    result = KalmanCovariateConfig(**options)
    result.validate()
    return result


def check_same_upstream(baseline, candidate):
    if not baseline.index.equals(candidate.index):
        raise ValueError('Physical hour alignment changed')
    for c in baseline:
        if 'forecast_origin' in c:
            if c not in candidate or not baseline[c].equals(candidate[c]):
                raise ValueError(f'Forecast origin modified: {c}')
        if c == 'actual' or c == 'residual_correction' or c.startswith(('chronos2__', 'residual_corrected__')):
            if c not in candidate or not np.array_equal(baseline[c].to_numpy(), candidate[c].to_numpy(), equal_nan=True):
                raise ValueError(f'Upstream modified: {c}')


def validate_forecast(frame):
    if 'actual' in frame and frame.actual.notna().any():
        raise ValueError('Future observations are forbidden')
    values = frame[QUANTILES].to_numpy(float)
    if not np.isfinite(values).all() or (np.diff(values, axis=1) < 0).any():
        raise ValueError('Future nonfinite values or quantile crossings')


def baseline_controls(bundle, config, covconfig, zone, cache, workers):
    """Reproduce first/middle/last backtest days and forecast, without full refit."""
    history = indexed(bundle.residual_statistics)
    frozen = indexed(bundle.kalman_view.backtest)
    days = pd.Index(frozen.index.tz_convert(TIMEZONES[zone]).date).unique()
    checks = []
    for day in (days[0], days[len(days)//2], days[-1]):
        print(f'{zone} baseline control {day}', flush=True)
        local_days = history.index.tz_convert(TIMEZONES[zone]).date
        prefix = history.loc[local_days <= day].reset_index()
        last = day == days[-1]
        replay = replay_kalman_overlay(prefix, timezone=TIMEZONES[zone], evaluation_start_day=str(day),
            covariates=bundle.covariates.copy(deep=True), config=config, covariate_config=covconfig,
            training_lookback_days=365, rolling_refit_workers=workers, rolling_refit_cache_dir=cache,
            future_upstream=bundle.source_forecast.copy(deep=True) if last else None,
            future_covariates=bundle.covariates.copy(deep=True) if last else None)
        expected = frozen.loc[frozen.index.tz_convert(TIMEZONES[zone]).date == day]
        compare_replay(replay.predictions.loc[expected.index], expected)
        if last:
            compare_replay(replay.future_predictions, indexed(bundle.kalman_view.forecast))
        checks.append({'day': str(day), 'hours': len(expected), 'max_tolerance': 1e-9,
                       'forecast_checked': last, 'matched': True})
    return checks


def compare_replay(actual, expected):
    if not actual.index.equals(expected.index):
        raise ValueError('Baseline replay timestamps differ')
    for c in QUANTILES + ['kalman_correction', 'kalman_raw_correction', 'kalman_weight']:
        if not np.allclose(actual[c], expected[c], rtol=0, atol=1e-9):
            raise ValueError(f'Unmodified Kalman replay differs: {c}')
    if not np.array_equal(actual.kalman_selected_filter, expected.kalman_selected_filter):
        raise ValueError('Baseline governance selection differs')


def scores(actual, pred):
    n = len(actual)
    error = pred - actual
    return {'hours': n, 'mae': float(np.abs(error).mean()) if n else None,
        'rmse': float(np.sqrt(np.square(error).mean())) if n else None,
        'bias': float(error.mean()) if n else None}


def evaluate(baseline, candidate, timezone):
    check_same_upstream(baseline, candidate)
    for frame in (baseline, candidate):
        values = frame[QUANTILES + ['actual']].to_numpy(float)
        if not np.isfinite(values).all() or (np.diff(values[:, :3], axis=1) < 0).any():
            raise ValueError('Nonfinite values or quantile crossings')
    actual = baseline.actual.to_numpy(float)
    predictions = {'baseline': baseline[QUANTILES[1]].to_numpy(float), 'interaction': candidate[QUANTILES[1]].to_numpy(float)}
    masks = {'all': np.ones(len(actual), dtype=bool), 'actual_ge_200': actual >= 200, 'actual_ge_300': actual >= 300}
    months = baseline.index.tz_convert(timezone).strftime('%Y-%m')
    masks.update({m: months == m for m in sorted(set(months))})
    result = {'slices': {key: {name: scores(actual[mask], pred[mask]) for name, pred in predictions.items()} for key, mask in masks.items()},
        'changed_hours': int(np.count_nonzero(np.abs(predictions['interaction'] - predictions['baseline']) > 1e-9)), 'spikes': {}}
    for threshold in (200, 300):
        observed = actual >= threshold
        result['spikes'][str(threshold)] = {}
        for name, pred in predictions.items():
            forecast = pred >= threshold
            tp = int((observed & forecast).sum()); fp = int((~observed & forecast).sum()); fn = int((observed & ~forecast).sum())
            result['spikes'][str(threshold)][name] = {'tp': tp, 'fp': fp, 'fn': fn,
                'precision': tp/(tp+fp) if tp+fp else None, 'recall': tp/(tp+fn) if tp+fn else None}
    return result


def render_report(directory, zone, metrics, provenance, forecast, baseline_forecast):
    esc = html.escape
    rows = []
    def number(v):
        return '—' if v is None else f'{v:.3f}'
    for label, variants in metrics['slices'].items():
        for variant, values in variants.items():
            rows.append('<tr><td>'+esc(label)+'</td><td>'+esc(variant)+'</td><td>'+str(values['hours'])+'</td>'+''.join('<td>'+number(values[k])+'</td>' for k in ('mae', 'rmse', 'bias'))+'</tr>')
    future_rows = ''.join('<tr><td>'+esc(str(t))+'</td><td>'+number(baseline_forecast.loc[t, QUANTILES[1]])+'</td><td>'+number(row[QUANTILES[1]])+'</td></tr>' for t, row in forecast.iterrows())
    document = '<!doctype html><html lang="fr"><meta charset="utf-8"><title>SolarWind interaction '+zone+'</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#192b3c}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,td:nth-child(2){text-align:left}pre{white-space:pre-wrap;font-size:12px}h1,h2{color:#144b65}</style><h1>SolarWind — interaction Kalman '+zone+'</h1><p>Test rétrospectif isolé, hors production. Chronos et CatBoost figés ; une seule covariable ajoutée au groupe market du Kalman. Plafond ±20 €/MWh inchangé avant pondération.</p><p>Score = faible vent × faible solaire × charge résiduelle élevée. Chaque normalisation utilise uniquement les jours antérieurs. Ce score n’est pas une probabilité calibrée. Aucune hausse de prix forcée.</p><p>365 jours scellés du 22/09/2025 au 21/09/2026, mêmes heures et mêmes observations. Le 22/09/2026 est présenté séparément sans prix réalisé. Ces dates diffèrent de la fenêtre glissante affichée dans les anciens rapports.</p><p>Limites héritées : sources rétrospectives as-of J−1 08h, publication d’origine non certifiée, substitutions NL documentées dans la référence ; cas choisi après observation. Ce test ne constitue ni une validation prospective ni une autorisation de promotion.</p><h2>Erreurs appariées (€/MWh)</h2><table><tr><th>Période</th><th>Variante</th><th>Heures</th><th>MAE</th><th>RMSE</th><th>Biais prédit−observé</th></tr>'+''.join(rows)+'</table><h2>Détection des pics : seuils fixés avant calcul</h2><pre>'+esc(json_bytes(metrics['spikes']).decode())+'</pre><h2>Prévisions du 22/09/2026 (UTC)</h2><table><tr><th>Heure</th><th>Référence Q50</th><th>Interaction Q50</th></tr>'+future_rows+'</table><details><summary>Provenance et contrôles</summary><pre>'+esc(json_bytes(provenance).decode())+'</pre></details></html>'
    safe_path(directory / 'report.html').write_text(document, encoding='utf-8')


def run(*, action, zones, workers=2, threads=2):
    if len(zones) != len(set(zones)) or not set(zones) <= set(BASELINES):
        raise ValueError('Unique DE/NL zones required')
    if workers not in (1, 2) or threads not in (1, 2):
        raise ValueError('CPU limits exceeded')
    prepared = {zone: prepare(zone) for zone in zones}
    for zone, data in prepared.items():
        print(json.dumps({'zone': zone, 'validation': 'PASS', 'identity': data[-1]['run_id'], 'feature': data[-1]['feature']}), flush=True)
    if action == 'validate':
        return
    if action != 'run':
        raise ValueError('Unknown action')
    safe_path(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with exclusive_process_lock(OUTPUT / 'batch.lock'), threadpool_limits(limits=threads):
        results = {}
        for zone, (bundle, config, original, candidate_config, covariates, feature_audit, identity) in prepared.items():
            directory = OUTPUT / zone.lower() / identity['run_id']
            safe_path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            status_path = directory / 'status.json'
            manifest_path = directory / 'experiment.json'
            if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != identity:
                raise ValueError('Experiment identity changed')
            if status_path.exists() and json.loads(status_path.read_text(encoding='utf-8'))['status'] == 'COMPLETE':
                state = json.loads(status_path.read_text(encoding='utf-8'))
                receipt = json.loads((directory / 'completion.json').read_text(encoding='utf-8'))
                if (receipt.get('identity') != identity['run_id'] or set(receipt.get('files', {})) != set(RESULT_FILES)
                        or state.get('annual_complete') is not True or state.get('identity') != identity['run_id']
                        or any(sha(safe_path(directory / name)) != digest for name, digest in receipt['files'].items())):
                    raise ValueError('Completed output changed')
                results[zone] = str(directory / 'report.html')
                continue
            write_json(manifest_path, identity)
            def status(state, phase, **extra):
                write_json(status_path, {'status': state, 'phase': phase, 'pid': os.getpid(),
                    'updated_utc': datetime.now(timezone.utc).isoformat(), 'zone': zone, 'identity': identity['run_id'], **extra})
                print(f'{zone}: {state} {phase}', flush=True)
            try:
                status('RUNNING', 'baseline_controls')
                controls = baseline_controls(bundle, config, original, zone, directory / 'control_cache', workers)
                write_json(directory / 'baseline_controls.json', controls)
                covariates.to_parquet(safe_path(directory / 'interaction_covariates.parquet'))
                write_json(directory / 'feature_audit.json', feature_audit)
                status('RUNNING', 'kalman_replay')
                view = build_operational_kalman_view(statistics=bundle.residual_statistics.copy(deep=True),
                    source_forecast=bundle.source_forecast.copy(deep=True), covariates=covariates.rename_axis('timestamp').reset_index(),
                    timezone=TIMEZONES[zone], delivery_day=DAY, config=config, covariate_config=candidate_config,
                    training_lookback_days=365, rolling_refit_workers=workers, rolling_refit_cache_dir=directory / 'candidate_cache')
                baseline, candidate = indexed(bundle.kalman_view.backtest), indexed(view.backtest)
                future, old_future = indexed(view.forecast), indexed(bundle.kalman_view.forecast)
                check_same_upstream(old_future, future)
                validate_forecast(future)
                metrics = evaluate(baseline, candidate, TIMEZONES[zone])
                audit = view.replay.audit
                if audit['causality_violations'] or audit['quantile_crossings'] or audit['future_observations_assimilated']:
                    raise ValueError('Replay causality/quantile contract violated')
                if audit['evaluation_days'] != 365 or audit['evaluation_hours'] != 8760 or audit['future_forecast_hours'] != 24:
                    raise ValueError('Incomplete replay')
                # Full source checksum/semantic verification again before publishing completion.
                _, _, _, _, _, _, checked_identity = prepare(zone)
                if identity != checked_identity:
                    raise ValueError('Sources or code changed during calculation')
                status('RUNNING', 'report')
                candidate.to_parquet(safe_path(directory / 'backtest.parquet'))
                future.to_parquet(safe_path(directory / 'forecast.parquet'))
                write_json(directory / 'replay_audit.json', audit)
                write_json(directory / 'metrics.json', metrics)
                render_report(directory, zone, metrics, {'identity': identity, 'baseline_controls': controls,
                    'baseline_clipped_raw_shifts': bundle.kalman_view.replay.audit['clipped_raw_shifts'],
                    'candidate_clipped_raw_shifts': audit['clipped_raw_shifts']}, future, old_future)
                write_json(directory / 'completion.json', {'files': {n: sha(directory / n) for n in RESULT_FILES}, 'identity': identity['run_id']})
                status('COMPLETE', 'complete', annual_complete=True)
                results[zone] = str(directory / 'report.html')
            except Exception as exc:
                status('FAILED', 'blocked', error=f'{type(exc).__name__}: {exc}')
                raise
        write_json(OUTPUT / 'latest_DE_NL.json', {'engine': ENGINE, 'status': 'COMPLETE' if set(results) == set(BASELINES) else 'PARTIAL', 'results': results})
        links = ''.join('<li><a href="'+html.escape(Path(p).relative_to(OUTPUT).as_posix())+'">'+z+'</a></li>' for z, p in results.items())
        safe_path(OUTPUT / 'index.html').write_text('<!doctype html><html lang="fr"><meta charset="utf-8"><title>SolarWind interaction</title><h1>SolarWind — interaction Kalman seule</h1><p>Expérience rétrospective hors production.</p><ul>'+links+'</ul></html>', encoding='utf-8')
