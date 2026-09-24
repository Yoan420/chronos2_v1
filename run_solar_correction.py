"""Frozen nuclear Chronos + solar residual, followed by two independent Kalman arms."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import html
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from chronos2_hourly.process_lock import exclusive_process_lock
from chronos2_hourly.atomic_directory import AtomicDirectoryStaging
from run_nuclear_forecast import sha256, write_json, run_progress
from run_nuclear_cwe_forecast import _copy_checked, _freeze_reporting, _verify_reporting, _day

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/'runs/experiments/solar_correction_v1'
SOURCE = ROOT/'data/pit/solar_cwe'
BASELINE = ROOT/'runs/experiments/nuclear_forecast_v1'
ZONES = ('FR', 'DE', 'BE', 'NL')
VARIANTS = ('residual', 'residual_kalman')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe(path):
    path = Path(path).absolute()
    if OUTPUT.resolve() != OUTPUT or path.resolve() != path or not path.is_relative_to(OUTPUT) or path == OUTPUT:
        raise ValueError('SolarCorrection writes must remain in its unredirected experiment namespace.')
    return path


def settings(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
    fields = {'schema_version', 'delivery_day', 'zones', 'output_root', 'source_root', 'historical_end', 'diagnostic_only', 'production_modified'}
    if (not isinstance(cfg, dict) or set(cfg) != fields or type(cfg['schema_version']) is not int
            or cfg['schema_version'] != 1 or cfg['zones'] != list(ZONES)
            or cfg['historical_end'] != '2026-09-19' or cfg['diagnostic_only'] is not True
            or cfg['production_modified'] is not False or (ROOT/cfg['output_root']).resolve() != OUTPUT
            or (ROOT/cfg['source_root']).resolve() != SOURCE):
        raise ValueError('Fixed four-country isolated SolarCorrection configuration required.')
    _day(cfg['delivery_day'])
    return cfg


def baseline(day, zone):
    if zone not in ZONES:
        raise ValueError('Unknown zone.')
    return BASELINE/_day(day)/zone.lower()/'civil_pit_v2'


def baseline_signature(day, zone):
    directory = baseline(day, zone)
    required = ('input_snapshot.json', 'resolved_config.yaml', 'run_result.json',
                'report_only/frozen_result/manifest.json',
                'checkpoints/nuclear_chronos_oof.csv.gz', 'checkpoints/nuclear_chronos_oof.csv.gz.manifest.json')
    if any(not (directory/name).is_file() for name in required):
        raise ValueError(f'{zone}/{day}: completed frozen nuclear forecast absent. Produce it with the unchanged normal process first.')
    return {name: sha256(directory/name) for name in required}


def preregister(cfg):
    from chronos2_hourly.solar_correction_protocol import create_or_load_protocol
    from chronos2_hourly.solar_cwe_forecast import SOLAR_SERIES
    recipes = {}
    for zone in ZONES:
        config = yaml.safe_load((baseline(cfg['delivery_day'], zone)/'resolved_config.yaml').read_text(encoding='utf-8'))
        recipes[zone] = {k: config[k] for k in ('model', 'hourly')}
        recipes[zone]['filter_parameters'] = config['nuclear_experiment']['filter_parameters']
    code = ('chronos2_hourly/solar_correction_forecast.py',
            'chronos2_hourly/solar_correction_protocol.py', 'chronos2_hourly/nuclear_forecast.py',
            'chronos2_hourly/models/residual_corrector.py', 'chronos2_hourly/kalman_residual.py',
            'chronos2_hourly/kalman_covariates.py', 'chronos2_hourly/nuclear_preparation.py',
            'chronos2_hourly/solar_cwe_forecast.py', 'chronos2_hourly/solar_cwe_sources.py',
            'run_chronos2_hourly.py')
    return create_or_load_protocol(OUTPUT, recipe_contract={'recipes': recipes, 'solar_series': SOLAR_SERIES,
        'code_sha256': {name: sha256(ROOT/name) for name in code}, 'chronos_recomputed': False,
        'country_specific_posthoc_selection': False}, historical_end=cfg['historical_end'])


def verify_snapshot(work, *, zone, day):
    work = safe(work)
    manifest = read(work/'input_snapshot.json')
    if work.parent != OUTPUT/day/zone.lower() or manifest['identity']['zone'] != zone or manifest['identity']['delivery_day'] != day:
        raise ValueError('SolarCorrection snapshot zone/day mismatch.')
    inputs, seen = set(), set()
    for group in ('files', 'reference_files'):
        if not isinstance(manifest.get(group), list) or not manifest[group]:
            raise ValueError('Incomplete SolarCorrection input inventory.')
        for item in manifest[group]:
            path = safe(item['snapshot_path'])
            if path in seen or not path.is_relative_to(work) or sha256(path) != item['sha256']:
                raise ValueError(f'Frozen SolarCorrection input changed: {path}')
            seen.add(path)
            if group == 'files':
                if not path.is_relative_to(work/'snapshot'):
                    raise ValueError('Model input outside its frozen snapshot.')
                inputs.add(path)
    path = work/'resolved_config.yaml'
    if sha256(path) != manifest['resolved_config_sha256']:
        raise ValueError('Frozen SolarCorrection recipe changed.')
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    required = [config['zones'][zone]['target']['file'], *config['data']['pit_files'].values()]
    if any(Path(p).resolve() not in inputs for p in required):
        raise ValueError('Unsealed model input.')
    return config


def load_incumbent(work, *, full_history=False):
    from chronos2_hourly.nuclear_run_archive import _FRAMES, _validate
    from chronos2_hourly.nuclear_forecast import ZONE_TIMEZONES, _validate_origins
    directory = work/'reference/baseline'
    frames = {name: pd.read_parquet(directory/(name+'.parquet')) for name in _FRAMES}
    audits = read(directory/'audits.json')
    _validate(frames, audits, {'snapshot_identity': read(work/'input_snapshot.json')['identity']})
    if full_history:
        checkpoint = directory/'nuclear_chronos_oof.csv.gz'
        receipt = read(Path(str(checkpoint)+'.manifest.json'))
        audit = audits['result']
        if (receipt.get('status') != 'complete' or receipt.get('zone') != audit['zone']
                or receipt.get('output_sha256') != sha256(checkpoint)
                or receipt.get('first_delivery_day_local') != audit['raw_history_start_day']
                or receipt.get('last_delivery_day_local') != str((pd.Timestamp(audit['delivery_day'])-pd.Timedelta(days=1)).date())):
            raise ValueError('Unverified anchored nuclear Chronos checkpoint.')
        full = pd.read_csv(checkpoint, float_precision='round_trip')
        full.index = pd.DatetimeIndex(pd.to_datetime(full.pop('delivery_start_utc'), utc=True))
        full.forecast_origin_utc = pd.to_datetime(full.forecast_origin_utc, utc=True)
        timezone = ZONE_TIMEZONES[audit['zone']]
        expected = pd.date_range(pd.Timestamp(audit['raw_history_start_day'],tz=timezone),
            pd.Timestamp(audit['delivery_day'],tz=timezone),freq='h',inclusive='left').tz_convert('UTC')
        if (not full.index.equals(expected) or receipt.get('n_delivery_hours') != len(expected)
                or receipt.get('completed_hours') != len(expected)
                or receipt.get('completed_days') != receipt.get('n_delivery_days')
                or receipt.get('n_delivery_days') != len(set(expected.tz_convert(timezone).date))):
            raise ValueError('Frozen Chronos checkpoint misses anchored physical hours or days.')
        _validate_origins(full, timezone)
        saved = frames['raw_history']
        for column in ('q10', 'q50', 'q90', 'actual'):
            full[column] = full[column].astype(saved[column].dtype)
        full = full.loc[:, saved.columns]
        if not saved.equals(full.reindex(saved.index)):
            raise ValueError('Recovered Chronos quantiles differ from the exact archived forecast.')
        frames['raw_history'] = full
    view = SimpleNamespace(backtest=frames.pop('kalman_backtest'), forecast=frames.pop('kalman_forecast'),
                           replay=SimpleNamespace(audit=audits['kalman_replay']))
    return SimpleNamespace(**frames, kalman_view=view, audit=audits['result'])


def prepare(cfg, zone, sources, protocol):
    from chronos2_hourly.nuclear_run_archive import load_nuclear_result_bundle
    from chronos2_hourly.solar_cwe_forecast import SOLAR_SERIES
    from chronos2_hourly.solar_correction_forecast import solar_correction_input_protocol
    prior_dir = baseline(cfg['delivery_day'], zone)
    prior_signature = baseline_signature(cfg['delivery_day'], zone)
    signature = {'zone': zone, 'delivery_day': cfg['delivery_day'], 'baseline': prior_signature,
                 'protocol_sha256': protocol['protocol_sha256'],
                 'runner_sha256': sha256(Path(__file__)),
                 'solar': {a: (r['sha256'], r['audit_sha256']) for a, r in sources.items()}}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    work = safe(OUTPUT/cfg['delivery_day']/zone.lower()/key)
    with exclusive_process_lock(work/'prepare.lock'):
        if (work/'input_snapshot.json').exists():
            verify_snapshot(work, zone=zone, day=cfg['delivery_day'])
            return work
        incumbent = load_nuclear_result_bundle(workdir=prior_dir)
        config = yaml.safe_load((prior_dir/'resolved_config.yaml').read_text(encoding='utf-8'))
        prior = read(prior_dir/'input_snapshot.json')
        if sha256(prior_dir/'resolved_config.yaml') != prior['resolved_config_sha256']:
            raise ValueError('Nuclear reference recipe changed.')
        files, references, relocated = [], [], {}
        for item in prior['files']:
            source = Path(item['snapshot']).resolve()
            if not source.is_relative_to(prior_dir/'snapshot'):
                raise ValueError('Unsafe nuclear source path.')
            record = _copy_checked(source, safe(work/'snapshot'/source.name), expected=item['sha256'])
            files.append(record); relocated[str(source)] = record['snapshot_path']
        for alias, record in sources.items():
            for field, digest in (('path', 'sha256'), ('audit_path', 'audit_sha256')):
                source = Path(record[field])
                files.append(_copy_checked(source, safe(work/'snapshot'/source.name), expected=record[digest]))
        spec = config['zones'][zone]
        spec['target']['file'] = relocated[str(Path(spec['target']['file']).resolve())]
        for alias, covar in spec['covariates'].items():
            if covar.get('enabled', True):
                covar['pit_file'] = relocated[str(Path(config['data']['pit_files'][alias]).resolve())]
                if 'file' in covar:
                    covar['file'] = covar['pit_file']
        if set(sources) != set(SOLAR_SERIES) or set(spec['covariates']).intersection(SOLAR_SERIES):
            raise ValueError('Four new solar channels required; pre-existing explicit solar refused.')
        for alias, series in SOLAR_SERIES.items():
            spec['covariates'][alias] = {'enabled': True, 'source': 'pit_parquet', 'series': series,
                'pit_file': str(work/'snapshot'/Path(sources[alias]['path']).name), 'include_base_context': True,
                'fill_method': 'none', 'fill_limit': 0, 'minimum_coverage': .01,
                'future': {'known_future': True, 'strategies': ['oracle']}, 'unit': 'GW',
                'semantic': 'forecast_generation', 'daily_broadcast': False,
                'description': f'{alias[:2].upper()} solar forecast; residual/Kalman only, Chronos frozen'}
        config['data'].update(pit_files={a: s['pit_file'] for a, s in spec['covariates'].items() if s.get('enabled', True)},
                              pit_vintage_dir=str(work/'snapshot'), cache_dir=str(work/'prepared/cache'))
        columns = config['hourly']['feature_engineering'].get('covariate_columns')
        if columns is not None:
            columns.extend(f'known_{a}_oracle' for a in SOLAR_SERIES if f'known_{a}_oracle' not in columns)
        config['output']['directory'] = str(work)
        experiment = config['nuclear_experiment']
        experiment.update(input_protocol=solar_correction_input_protocol(), mode='incremental', candidate_model='solar_correction',
                          incremental_cache_dir=str(OUTPUT/'_daily_cache'/zone.lower()),
                          incremental_namespace=str(OUTPUT/'_daily_cache'/zone.lower()), production_modified=False)
        if experiment.get('residual_bank_audit'):
            experiment['residual_bank_audit'] = relocated[str(Path(experiment['residual_bank_audit']).resolve())]
        _, refs = _freeze_reporting(prior_dir, work, read(prior_dir/'run_result.json'))
        references.extend(refs)
        frozen_dir = prior_dir/'report_only/frozen_result'
        for source in sorted(frozen_dir.iterdir()):
            if not source.is_file():
                raise ValueError('Unexpected frozen reference directory.')
            references.append(_copy_checked(source, safe(work/'reference/baseline'/source.name)))
        for name in ('nuclear_chronos_oof.csv.gz', 'nuclear_chronos_oof.csv.gz.manifest.json'):
            references.append(_copy_checked(prior_dir/'checkpoints'/name, safe(work/'reference/baseline'/name)))
        audit = {'solar': sources, 'chronos_frozen': True, 'chronos_recomputed': False,
                 'baseline': str(prior_dir), 'same_history_anchor': incumbent.audit['history_anchor_day'],
                 'production_modified': False, 'production_pit_evidence': False,
                 'historical_diagnostic_only': cfg['delivery_day'] <= cfg['historical_end'],
                 'publication_evidence': 'historical query-asof is not proof of original publication'}
        write_json(work/'source_audit.json', audit)
        references.append({'snapshot_path': str(work/'source_audit.json'), 'sha256': sha256(work/'source_audit.json')})
        resolved = work/'resolved_config.yaml'
        resolved.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
        if baseline_signature(cfg['delivery_day'], zone) != prior_signature:
            raise ValueError('Reference changed during snapshotting.')
        write_json(work/'input_snapshot.json', {'schema_version': 1, 'identity': signature, 'files': files,
                   'reference_files': references, 'resolved_config_sha256': sha256(resolved)})
        verify_snapshot(work, zone=zone, day=cfg['delivery_day'])
        load_incumbent(work, full_history=True)
    return work


def archive_results(work, results=None):
    """Atomic two-arm archive, signed against the shared immutable input manifest."""
    from chronos2_hourly.nuclear_run_archive import _collect, _validate, _FRAMES
    directory = safe(work/'results')
    if results is not None and directory.exists():
        raise ValueError('Refusing to overwrite an already sealed two-arm result; load it without new results.')
    identity = {'input_sha256': sha256(work/'input_snapshot.json'), 'config_sha256': sha256(work/'resolved_config.yaml')}
    sources = {'snapshot_identity': read(work/'input_snapshot.json')['identity']}
    if results is not None and not directory.exists():
        if set(results) != set(VARIANTS):
            raise ValueError('Both predefined variants must finish before publication.')
        with AtomicDirectoryStaging(work, prefix='.results-') as staging:
            inventories = {}
            for variant, result in results.items():
                dest = staging.path/variant; dest.mkdir()
                frames, audits = _collect(result); _validate(frames, audits, sources)
                for name, frame in frames.items():
                    frame.to_parquet(dest/(name+'.parquet'))
                write_json(dest/'audits.json', audits)
                inventories[variant] = {p.name: sha256(p) for p in dest.iterdir()}
            write_json(staging.path/'manifest.json', {'identity': identity, 'files': inventories,
                       'sealed_at_utc': str(pd.Timestamp.now(tz='UTC'))})
            staging.publish(directory)
    record = read(directory/'manifest.json')
    if record['identity'] != identity or set(record['files']) != set(VARIANTS):
        raise ValueError('Frozen two-arm results do not match their inputs.')
    loaded = {}
    for variant in VARIANTS:
        path = safe(directory/variant)
        inventory = record['files'][variant]
        if set(inventory) != {*(name+'.parquet' for name in _FRAMES), 'audits.json'}:
            raise ValueError('Incomplete two-arm result inventory.')
        for name, digest in inventory.items():
            if sha256(safe(path/name)) != digest:
                raise ValueError('Frozen two-arm result checksum changed.')
        frames = {name: pd.read_parquet(path/(name+'.parquet')) for name in _FRAMES}
        audits = read(path/'audits.json'); _validate(frames, audits, sources)
        view = SimpleNamespace(backtest=frames.pop('kalman_backtest'), forecast=frames.pop('kalman_forecast'), replay=SimpleNamespace(audit=audits['kalman_replay']))
        loaded[variant] = SimpleNamespace(**frames, kalman_view=view, audit=audits['result'])
    return loaded, record


def run_zone(cfg, zone, work, args, protocol):
    from chronos2_modular.common import build_zone_configs
    from chronos2_hourly.nuclear_preparation import prepare_nuclear_zone_data
    from chronos2_hourly.solar_correction_forecast import run_solar_correction_forecast
    from chronos2_hourly.solar_correction_reporting import render_solar_correction_reports
    config = verify_snapshot(work, zone=zone, day=cfg['delivery_day'])
    if read(work/'input_snapshot.json')['identity']['protocol_sha256'] != protocol['protocol_sha256']:
        raise ValueError('Snapshot belongs to a different preregistered recipe.')
    with exclusive_process_lock(work/'run.lock'), run_progress(work, zone, pd.Timestamp(cfg['delivery_day']), args.action) as progress:
        spec = build_zone_configs(config, [zone], None, None)[0]
        progress('prepare_inputs')
        data = prepare_nuclear_zone_data(spec, config, work, work/'prepared')
        actual, reporting = _verify_reporting(work, zone=zone, day=cfg['delivery_day'], timezone=spec.timezone)
        if args.action == 'prepare':
            return {'status': 'prepared', 'workdir': str(work)}
        if args.action != 'report' and not (work/'results').exists():
            progress('residual_then_two_kalman', chronos_recomputed=False)
            print(f'[SolarCorrection/{zone}] Chronos archive intact; un correcteur solaire, deux Kalman.', flush=True)
            results = run_solar_correction_forecast(config=config, data=data, incumbent=load_incumbent(work, full_history=True),
                zone=zone, delivery_day=cfg['delivery_day'], workdir=work, threads=args.threads, workers=args.workers)
            results, record = archive_results(work, results)
        else:
            results, record = archive_results(work)
        progress('render_reports')
        reports = render_solar_correction_reports(results, data=replace(data, target=actual), zone=zone,
            delivery_day=cfg['delivery_day'], output_directory=work/'reports', incumbent=load_incumbent(work),
            storm_archive=work/'reference/reporting', observed_source_audit=reporting,
            source_audit=read(work/'source_audit.json'), validation_protocol=protocol)
        receipt = {'status': 'complete', 'reports': reports, 'sealed_at_utc': record['sealed_at_utc'],
                   'chronos_recomputed': False, 'production_modified': False, 'activation_performed': False}
        write_json(work/'run_result.json', receipt)
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/solar_correction.yaml')
    parser.add_argument('--action', choices=('run','prepare','audit','status','report','evaluate'), default='run')
    parser.add_argument('--delivery-day')
    parser.add_argument('--zones', nargs='+', choices=ZONES)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--workers', type=int, choices=(1,2), default=2)
    parser.add_argument('--after-solar-pid', type=int, default=0,
                        help='Run only: wait for this existing SolarCWE process before using the CPU.')
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 32:
        parser.error('threads must be between 1 and 32')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    cfg = settings(args.config)
    if args.delivery_day:
        cfg['delivery_day'] = _day(args.delivery_day)
    zones = args.zones or list(ZONES)
    if len(set(zones)) != len(zones):
        parser.error('duplicate zones')
    pointer = safe(OUTPUT/cfg['delivery_day']/('latest_'+'_'.join(zones)+'.json'))
    if args.action == 'status':
        state = read(pointer) if pointer.exists() else {'status': 'not_prepared'}
        state['states'] = {}
        for zone, raw in state.get('workdirs', {}).items():
            work = safe(raw)
            if zone not in zones or work.parent != OUTPUT/cfg['delivery_day']/zone.lower():
                raise ValueError('Status pointer outside requested scope.')
            state['states'][zone] = read(work/'run_status.json') if (work/'run_status.json').exists() else {'status':'prepared'}
        print(json.dumps(state, default=str, indent=2)); return 0
    if args.action == 'evaluate':
        return evaluate(cfg)
    if args.after_solar_pid:
        if args.action != 'run':
            raise ValueError('after-solar-pid is allowed only for Run.')
        import psutil
        import time
        try:
            prior = psutil.Process(args.after_solar_pid)
            if str(ROOT/'run_solar_cwe_forecast.py').casefold() not in [a.casefold() for a in prior.cmdline()]:
                raise ValueError('Only an existing SolarCWE run in this project can be awaited.')
            queued = read(pointer) if pointer.exists() else {}
            queued.update(status='waiting_existing_solar_run',waiting_pid=args.after_solar_pid,
                          queue_pid=os.getpid(),queued_at_utc=str(pd.Timestamp.now(tz='UTC')),production_modified=False)
            write_json(pointer,queued)
            print(f'[SolarCorrection] En attente de SolarCWE PID {args.after_solar_pid}; aucun processus interrompu.',flush=True)
            while prior.is_running() and prior.status() != psutil.STATUS_ZOMBIE:
                time.sleep(15)
        except psutil.NoSuchProcess:
            pass
    if args.action == 'report':
        protocol = read(safe(OUTPUT/'solar_correction_protocol.json'))
        from chronos2_hourly.solar_correction_protocol import prospective_gate
        prospective_gate(protocol, [])
        workdirs = {z: safe(w) for z,w in read(pointer)['workdirs'].items()}
        if set(workdirs) != set(zones):
            raise ValueError('Report countries differ from selected run.')
    else:
        audit = {z: baseline_signature(cfg['delivery_day'], z) for z in ZONES}
        if args.action == 'audit':
            print(json.dumps({'delivery_day':cfg['delivery_day'], 'baselines':audit, 'chronos_recomputed':False}, indent=2)); return 0
        protocol = preregister(cfg)
        from chronos2_hourly.solar_cwe_sources import ensure_solar_sources
        start = min(read(baseline(cfg['delivery_day'],z)/'report_only/frozen_result/audits.json')['result']['raw_history_start_day'] for z in zones)
        sources = ensure_solar_sources(output_root=SOURCE,start_day=start,end_day=cfg['delivery_day'],workers=args.workers,sync=True)
        workdirs = {z:prepare(cfg,z,sources,protocol) for z in zones}
    results = {}
    for zone, work in workdirs.items():
        write_json(pointer, {'status':'running','active_zone':zone,'workdirs':workdirs,'results':results,'production_modified':False})
        try:
            results[zone] = run_zone(cfg,zone,work,args,protocol)
        except Exception as error:
            write_json(pointer, {'status':'failed','failed_zone':zone,'error':str(error),'workdirs':workdirs,'results':results,'production_modified':False})
            raise
    status = 'prepared' if args.action == 'prepare' else 'complete'
    write_json(pointer, {'status':status,'workdirs':workdirs,'results':results,'production_modified':False})
    if status == 'complete':
        index = safe(pointer.parent/('solar_correction_'+'_'.join(zones)+'_index.html'))
        links = [f'<li><a href="{html.escape(os.path.relpath(path,index.parent).replace(chr(92),chr(47)),quote=True)}">{zone} — {html.escape(model)}</a></li>'
                 for zone,res in results.items() for model,path in res['reports'].items() if str(path).endswith('.html')]
        index.write_text('<!doctype html><meta charset="utf-8"><title>SolarCorrection</title><style>body{font:18px system-ui;margin:3rem;background:#111827;color:#e5e7eb}a{color:#7dd3fc}li{margin:1rem}</style>'
            '<h1>Chronos nucléaire figé — solaire dans la correction</h1><p>Deux variantes prédéfinies, aucune promotion. Année historique diagnostique ; nouvelles journées évaluées séparément.</p><ul>'+''.join(links)+'</ul>',encoding='utf-8')
    print(json.dumps({'status':status,'workdirs':workdirs,'results':results},default=str,indent=2))
    return 0


def evaluate(cfg):
    from chronos2_hourly.solar_correction_protocol import prospective_gate, complete_delivery
    path = safe(OUTPUT/'solar_correction_protocol.json')
    if not path.exists():
        raise ValueError('Prepare or Run first to freeze the prospective protocol.')
    protocol = read(path)
    prospective_gate(protocol, [])
    records = []
    # Score only published, sealed delivery predictions; never rerun a model here.
    for run_receipt in sorted(OUTPUT.glob('????-??-??/*/*/run_result.json')):
        work = safe(run_receipt.parent); day, zone = work.parent.parent.name, work.parent.name.upper()
        if day <= cfg['historical_end'] or read(run_receipt).get('status') != 'complete':
            continue
        config = verify_snapshot(work,zone=zone,day=day)
        manifest = read(work/'input_snapshot.json')
        if manifest['identity']['protocol_sha256'] != protocol['protocol_sha256']:
            raise ValueError('Validation forecast belongs to another preregistered recipe.')
        results, seal = archive_results(work)
        from chronos2_hourly.nuclear_reporting_refresh import refresh_nuclear_reporting_sources
        from chronos2_hourly.hourly_contract import local_delivery_day_index
        stamp = pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
        target, reporting_directory, observed_audit = refresh_nuclear_reporting_sources(config,zone,config['zones'][zone]['timezone'],day,
            safe(OUTPUT/'validation_sources'/stamp/day/zone.lower()))
        index = local_delivery_day_index(pd.Timestamp(day).date(),timezone=config['zones'][zone]['timezone'])
        actual = target.reindex(index)
        from chronos2_hourly.nuclear_report_benchmark import _load_verified_snapshot
        storm = _load_verified_snapshot(Path(reporting_directory),zone=zone,timezone=config['zones'][zone]['timezone'])[0].reindex(index)
        incumbent = load_incumbent(work).kalman_view.forecast
        baseline_values = incumbent.set_index(pd.to_datetime(incumbent.delivery_start_utc,utc=True)).residual_kalman__q50.reindex(index)
        if not np.isfinite(baseline_values).all():
            raise ValueError('Incomplete frozen incumbent validation forecast.')
        for variant,candidate in (('residual','solar_residual_standard_kalman'),('residual_kalman','solar_residual_solar_kalman')):
            frame = results[variant].kalman_view.forecast
            forecast = frame.set_index(pd.to_datetime(frame.delivery_start_utc,utc=True)).residual_kalman__q50.reindex(index)
            complete = complete_delivery(day, index, actual, forecast)
            error = forecast-actual
            baseline_error = baseline_values-actual
            storm_complete = complete_delivery(day, index, actual, storm)
            spike_slices = {}
            for threshold in protocol['spike_thresholds_eur_mwh']:
                mask = actual >= threshold
                spike_slices[str(threshold)] = {'hours':int(mask.sum()) if complete else 0,
                    'candidate_abs_error_sum':float(error[mask].abs().sum()) if complete else None,
                    'baseline_abs_error_sum':float(baseline_error[mask].abs().sum()) if complete else None}
            records.append({'delivery_date':day,'zone':zone,'candidate':candidate,'complete':complete,
                'sealed_at':seal['sealed_at_utc'],'protocol_sha256':protocol['protocol_sha256'],
                'forecast_sha256':seal['files'][variant]['kalman_forecast.parquet'],
                'mae_eur_mwh':float(error.abs().mean()) if complete else None,
                'rmse_eur_mwh':float(np.sqrt((error**2).mean())) if complete else None,
                'baseline_mae_eur_mwh':float(baseline_error.abs().mean()) if complete else None,
                'baseline_rmse_eur_mwh':float(np.sqrt((baseline_error**2).mean())) if complete else None,
                'day_mae_win_vs_baseline':bool(error.abs().mean()<baseline_error.abs().mean()) if complete else None,
                'day_mae_win_vs_storm':bool(error.abs().mean()<(storm-actual).abs().mean()) if complete and storm_complete else None,
                'spike_slices_ex_post':spike_slices,'hours':len(index),'observed_source':observed_audit})
    gate = prospective_gate(protocol, records)
    summary = {}
    for phase, days in gate['common_days'].items():
        summary[phase] = {}
        for zone in ZONES:
            for candidate in protocol['candidates']:
                selected = [r for r in records if r['delivery_date'] in days and r['zone']==zone and r['candidate']==candidate]
                if not selected:
                    continue
                hours = sum(r['hours'] for r in selected)
                storm_wins = [r['day_mae_win_vs_storm'] for r in selected if r['day_mae_win_vs_storm'] is not None]
                summary[phase][zone+'/'+candidate] = {
                    'days':len(selected), 'hours':hours,
                    'mae_eur_mwh':sum(r['mae_eur_mwh']*r['hours'] for r in selected)/hours,
                    'rmse_eur_mwh':float(np.sqrt(sum(r['rmse_eur_mwh']**2*r['hours'] for r in selected)/hours)),
                    'baseline_mae_eur_mwh':sum(r['baseline_mae_eur_mwh']*r['hours'] for r in selected)/hours,
                    'baseline_rmse_eur_mwh':float(np.sqrt(sum(r['baseline_rmse_eur_mwh']**2*r['hours'] for r in selected)/hours)),
                    'day_mae_win_rate_vs_baseline':float(np.mean([r['day_mae_win_vs_baseline'] for r in selected])),
                    'storm_complete_days':len(storm_wins),
                    'day_mae_win_rate_vs_storm':float(np.mean(storm_wins)) if storm_wins else None}
    result = {'validation':gate,'summary':summary,'records':records,'production_modified':False,'activation_performed':False}
    stamp = pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    write_json(safe(OUTPUT/'validation'/stamp/'evaluation.json'),result)
    print(json.dumps(result,default=str,indent=2)); return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError,OSError,KeyError,RuntimeError) as exc:
        logging.exception('[SolarCorrection] %s; production unchanged',exc)
        raise SystemExit(1)
