"""Sealed demand-response laboratory, never an operational activation path."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from importlib.metadata import version
from pathlib import Path
import shutil
import uuid
import numpy as np
import pandas as pd
import yaml
from chronos2_hourly.process_lock import exclusive_process_lock
from nyx_congestion_calibration import runner as source_runner
from nyx_congestion_calibration.report import assemble_panel
from nyx_coherent_p50.runner import protected_state
from nyx_scarcity.runner import digest, _json, _parquet
from nyx_rmse.runner import runtime_identity as base_runtime_identity, _verify_files
from nyx_physical_p50.report import _safe
from kpi_report.metrics import compute_kpis
from .inputs import validate_bundle, scenario_forecasts, KEYS
from .dispatch import solve_period

NAMESPACE = Path('runs/experiments/nyx_demand_response_v1')
SOURCE_AUDIT = 'config/nyx_demand_response_sources.json'
EVIDENCE_CONFIGS = ('config/marginal_cost_expert_v2.yaml','config/marginal_cost_sources_v2.yaml',SOURCE_AUDIT)
RESULTS = {'predictions.parquet','evaluation.json','scenario_solutions.json'}


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]


def safe_path(root, value):
    root = Path(root).resolve()
    path = Path(value)
    path = (path if path.is_absolute() else root/path).absolute()
    if path != path.resolve() or not path.is_relative_to(root/NAMESPACE) or (root/NAMESPACE).resolve() != root/NAMESPACE:
        raise ValueError('Outputs exclusively in the isolated demand-response namespace, without aliases.')
    return path


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
    fields = {'schema_version','source_suite','output_root','bundle_path','diagnostic_only','production_modified'}
    if (not isinstance(config, dict) or set(config) != fields or type(config['schema_version']) is not int or config['schema_version'] != 1
            or config['diagnostic_only'] is not True or config['production_modified'] is not False
            or config['output_root'] != NAMESPACE.as_posix() or not isinstance(config['source_suite'], str)
            or not config['source_suite'].strip() or (config['bundle_path'] is not None and
                (not isinstance(config['bundle_path'], str) or not config['bundle_path'].strip()))):
        raise ValueError('Strict schema1 diagnostic-only config required.')
    return config


def code_identity(root):
    paths = list((root/'nyx_demand_response').glob('*.py'))
    paths += [root/'DemandResponse.ps1',root/'run_nyx_demand_response.py']
    paths += [root/name for name in ('kpi_report/metrics.py', 'kpi_report/economic.py',
        'nyx_congestion_calibration/report.py', 'nyx_congestion/report.py', 'nyx_physical_p50/report.py')]
    return {**source_runner.training_code(root), **{p.relative_to(root).as_posix():digest(p) for p in paths}}


def runtime_identity():
    identity = base_runtime_identity()
    return {**identity, 'versions':{**identity['versions'], 'scipy':version('scipy')}}


def prepare(config, root):
    output = safe_path(root, config['output_root'])
    with exclusive_process_lock(output/'prepare.lock'):
        protected = protected_state(root)
        source, _, manifest = source_runner.read_snapshot(root/config['source_suite'], root=root)
        source_runner.verify_result(source, manifest)
        sources = {n:digest(source/n) for n in ('manifest.json','results_manifest.json',*source_runner.RESULTS)}
        directory = safe_path(root, output/'snapshots'/stamp())
        directory.mkdir(parents=True, exist_ok=False)
        _json(directory/'config.json', config)
        shutil.copyfile(source/'predictions.parquet', directory/'source_predictions.parquet')
        evidence = {name:digest(root/name) for name in EVIDENCE_CONFIGS}
        _json(directory/'source_evidence.json', dict(discovery=json.loads((root/SOURCE_AUDIT).read_text(encoding='utf8')),
            physical_config=yaml.safe_load((root/EVIDENCE_CONFIGS[0]).read_text(encoding='utf8')),
            sources_config=yaml.safe_load((root/EVIDENCE_CONFIGS[1]).read_text(encoding='utf8'))))
        if config['bundle_path'] is not None:
            bundle_path = (root/config['bundle_path']).resolve()
            bundle = json.loads(bundle_path.read_text(encoding='utf8'))
            validate_bundle(bundle)
            shutil.copyfile(bundle_path, directory/'bundle.json')
            if digest(bundle_path) != digest(directory/'bundle.json'):
                raise ValueError('Scenario bundle changed during capture.')
        inputs = {n:digest(directory/n) for n in ('config.json','source_predictions.parquet','source_evidence.json')}
        if (directory/'bundle.json').exists():
            inputs['bundle.json'] = digest(directory/'bundle.json')
        _verify_files(source, sources, set(sources))
        if evidence != {n:digest(root/n) for n in evidence} or protected != protected_state(root):
            raise ValueError('Concurrent source or production change; publication refused.')
        _json(directory/'manifest.json', dict(schema_version=1,created_at_utc=datetime.now(timezone.utc).isoformat(),
            source_dir=str(source),source_files=sources,evidence_files=evidence,input_files=inputs,
            code=code_identity(root),runtime=runtime_identity(),protected_files=protected,config=config))
        _json(directory/'status.json', dict(status='prepared',snapshot=str(directory)))
        _json(output/'latest_prepared.json',dict(snapshot=str(directory),manifest_sha256=digest(directory/'manifest.json')))
        return directory


def read_snapshot(directory, root):
    directory = safe_path(root, directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf8'))
    inputs = {'config.json','source_predictions.parquet','source_evidence.json'}
    if manifest['config']['bundle_path'] is not None:
        inputs.add('bundle.json')
    if (directory/'bundle.json').exists() != ('bundle.json' in inputs):
        raise ValueError('Unexpected or missing scenario bundle; all consumed inputs must be sealed at Prepare.')
    _verify_files(directory, manifest['input_files'], inputs)
    config = json.loads((directory/'config.json').read_text(encoding='utf8'))
    if config != manifest['config'] or directory.parent != root/NAMESPACE/'snapshots':
        raise ValueError('Snapshot configuration or directory differs.')
    if manifest['code'] != code_identity(root) or manifest['runtime'] != runtime_identity():
        raise ValueError('Code/runtime changed; prepare a NEW snapshot, do not reseal the old one.')
    source, _, original = source_runner.read_snapshot(Path(manifest['source_dir']),root=root)
    if source != (root/config['source_suite']).resolve():
        raise ValueError('Source identity differs.')
    source_runner.verify_result(source, original)
    _verify_files(source, manifest['source_files'], {'manifest.json','results_manifest.json',*source_runner.RESULTS})
    if manifest['input_files']['source_predictions.parquet'] != manifest['source_files']['predictions.parquet']:
        raise ValueError('Frozen predictions differ from source.')
    return directory, manifest


def demos():
    q = dict(physical_inputs_qualified=True, demand_basis='before_price_response',
        flexibility_basis='additional_voluntary_reduction',evidence_kind='synthetic_assumption',
        evidence_description='Fictitious single-zone mechanism demonstration, not FR/DE/BE/NL.',duration_hours=1.)
    supply = pd.DataFrame([dict(zone='X',segment='generation',capacity_mw=49000.,bid_eur_mwh=150.)])
    flex = pd.DataFrame([dict(zone='X',segment='first',capacity_mw=200.,reservation_eur_mwh=350.),
                         dict(zone='X',segment='second',capacity_mw=400.,reservation_eur_mwh=650.)])
    rows = []
    for demand in (48500.,49050.,49150.,49250.,49350.,49500.,49550.):
        result = solve_period(supply,pd.DataFrame([dict(zone='X',demand_mw=demand)]),flex,qualification=q)
        rows.append(dict(label='Exemple fictif, demande '+str(int(demand))+' MW',demand_mw=demand,supply_mw=49000.,
            price_eur_mwh=result['prices_eur_mwh']['X'],voluntary_reduction_mw=result['voluntary_demand_reduction_mw']['X'],
            status='synthetic_assumption_not_forecast'))
    return rows


def evaluate(directory, root):
    directory, manifest = read_snapshot(directory,root)
    with exclusive_process_lock(directory/'run.lock'):
        if (directory/'results_manifest.json').exists():
            verify_results(directory)
            return directory
        before = protected_state(root)
        source = Path(manifest['source_dir'])
        previous_manifest = json.loads((source/'manifest.json').read_text(encoding='utf8'))
        long, wide, audit, seals = assemble_panel(pd.read_parquet(directory/'source_predictions.parquet'), Path(previous_manifest['source_dir']))
        config = audit['source_config']
        end = pd.Timestamp(config['end_day'] if config.get('end_day') else pd.Timestamp(config['delivery_day'])-pd.Timedelta(days=1))
        start = end-pd.Timedelta(days=364)
        evaluation = wide.loc[wide['sample'].eq('evaluation') & wide.timestamp_utc.dt.tz_convert('Europe/Paris').dt.strftime('%Y-%m-%d').between(start.strftime('%Y-%m-%d'),end.strftime('%Y-%m-%d'))].copy()
        predictions = wide[KEYS+['forecast','q10','q90','actual','storm','sample']].copy()
        predictions['expert_price'] = np.nan
        predictions['expert_q10'] = np.nan
        predictions['expert_q90'] = np.nan
        predictions['expert_status'] = 'unavailable_no_evidenced_demand_curve'
        solutions = []
        if 'bundle.json' in manifest['input_files']:
            forecast, solutions = scenario_forecasts(json.loads((directory/'bundle.json').read_text(encoding='utf8')))
            index = pd.MultiIndex.from_frame(predictions[KEYS])
            supplied = forecast.set_index(KEYS)
            if not supplied.index.isin(index).all():
                raise ValueError('Scenario origin/delivery not present in the frozen comparison panel.')
            for name in ('expert_price','expert_q10','expert_q90','expert_status'):
                values = supplied[name].reindex(index).to_numpy()
                mask = pd.notna(values)
                predictions.loc[mask,name] = values[mask]
        # No operational policy is created or silently activated by this lab.
        predictions['nyx_unchanged'] = predictions.forecast
        scored = long.loc[long['sample'].eq('evaluation')]
        metrics = compute_kpis(scored,end_day=end.strftime('%Y-%m-%d'),days=365,zones=sorted(evaluation.zone.unique()))
        in_period = predictions.timestamp_utc.dt.tz_convert('Europe/Paris').dt.strftime('%Y-%m-%d').between(start.strftime('%Y-%m-%d'),end.strftime('%Y-%m-%d'))
        measured = predictions.loc[predictions['sample'].eq('evaluation') & predictions.actual.notna() & in_period]
        count = int(measured.expert_price.notna().sum())
        evidence = json.loads((directory/'source_evidence.json').read_text(encoding='utf8'))
        blockers = []
        if count == 0:
            blockers.append(dict(id='missing_ex_ante_curves',detail='Aucune prévision de courbe de demande documentée et disponible avant 08 h dans ce snapshot.'))
        physical = evidence['physical_config']
        if not all(physical['expert']['stack_scope_qualified_by_zone'].get(z) is True for z in ('FR','DE','BE','NL')):
            blockers.append(dict(id='partial_supply',detail='Le parc de l’expert physique existant est partiel ; son déficit ne mesure pas un besoin réel d’effacement.'))
        if not physical['network'].get('domain_reference_qualified') or not physical['network'].get('boundary_qualified'):
            blockers.append(dict(id='network_not_dispatch_ready',detail='Référence du domaine et frontières non qualifiées : RAM brut ≠ imports simultanément réalisables.'))
        blockers.append(dict(id='no_independent_validation',detail='Aucun entraînement validé des courbes futures ni test indépendant de l’intervention ; pas de promotion automatique.'))
        kpi = metrics['rows']
        if count:
            part = predictions[KEYS+['actual','storm','sample']].copy()
            part['model_id'],part['forecast'] = 'nyx_demand_response',predictions.expert_price
            # Explicit separate support; never compare an eligible-only score to a full-year reference.
            paired = compute_kpis(pd.concat([scored,part.loc[part['sample'].eq('evaluation')]],ignore_index=True),end_day=end.strftime('%Y-%m-%d'),days=365,zones=sorted(evaluation.zone.unique()))
        else:
            paired = None
        case = predictions.loc[predictions.timestamp_utc.dt.tz_convert('Europe/Paris').dt.strftime('%Y-%m-%d').eq('2026-09-14') & predictions.timestamp_utc.dt.tz_convert('Europe/Paris').dt.hour.eq(19)]
        demonstration = demos()
        payload = _safe(dict(schema_version=1,snapshot=str(directory),period=dict(start_day=start.strftime('%Y-%m-%d'),end_day=end.strftime('%Y-%m-%d'),days=365),
            source_report=str(source),decision=dict(integration_ready=False,
                reason='Mécanisme économique implémenté ; aucune preuve suffisante pour intégrer ce candidat à NYX.',
                qualified_country_hours=count,total_country_hours=len(measured),interventions=0,empirical_gain_demonstrated=False),
            blockers=blockers,sources=evidence['discovery']['sources'],kpi_rows=kpi,paired_expert_evaluation=paired,
            cases=[dict(zone=r.zone,timestamp_utc=r.timestamp_utc,actual=r.actual,storm=r.storm,nyx=r.forecast,
                        expert=r.expert_price,status=r.expert_status) for r in case.itertuples()],
            demos=[demonstration[i] for i in (0,1,5)],sensitivity=demonstration,
            audit=dict(source_snapshot=str(source),source_sha256=seals,coverage=metrics['coverage'],
                production_modified=False,activation_performed=False,automatic_training_performed=False,
                demand_destruction_identified=False,scenario_quantiles_statistically_calibrated=False,
                unavailable_expert_is_not_a_forecast=True,fit_to_observed_spike_performed=False,
                evidence=evidence,source_fields_available=[c for c in wide if c.startswith('feature_')],
                scenario_contract='Whole physical hours, coupled paths, all inputs available at D-1 08h; no final auction curve.')))
        pd.testing.assert_series_equal(predictions.nyx_unchanged,predictions.forecast,check_names=False,check_exact=True)
        _parquet(directory/'predictions.parquet',predictions)
        _json(directory/'evaluation.json',payload)
        _json(directory/'scenario_solutions.json',_safe(solutions))
        read_snapshot(directory,root)
        if before != protected_state(root) or any(digest(Path(path)) != value for path,value in seals.items()):
            raise ValueError('Concurrent mutation; evaluation publication refused.')
        _json(directory/'results_manifest.json',dict(status='completed_diagnostic',manifest_sha256=digest(directory/'manifest.json'),
            files={n:digest(directory/n) for n in RESULTS}))
        _json(directory/'status.json',dict(status='evaluated_not_promotable',qualified_country_hours=count))
        return directory


def verify_results(directory):
    result = json.loads((directory/'results_manifest.json').read_text(encoding='utf8'))
    if result['status'] != 'completed_diagnostic' or result['manifest_sha256'] != digest(directory/'manifest.json'):
        raise ValueError('Unsealed diagnostic results.')
    _verify_files(directory,result['files'],RESULTS)


def report(directory,root):
    from .report import render_report
    directory,_ = read_snapshot(directory,root)
    verify_results(directory)
    with exclusive_process_lock(directory/'report.lock'):
        before = protected_state(root)
        destination = safe_path(root,directory/'reports'/stamp())
        destination.mkdir(parents=True,exist_ok=False)
        payload = json.loads((directory/'evaluation.json').read_text(encoding='utf8'))
        target = render_report(payload,destination/'nyx_demand_response_report.html')
        read_snapshot(directory,root)
        verify_results(directory)
        if before != protected_state(root):
            raise ValueError('Concurrent production mutation; report publication refused.')
        pointer = dict(status='completed_not_promotable',snapshot=str(directory),report=str(target),
            report_sha256=digest(target),results_manifest_sha256=digest(directory/'results_manifest.json'))
        _json(destination/'report_manifest.json',pointer)
        _json(directory/'latest_report.json',pointer)
        _json(root/NAMESPACE/'latest.json',pointer)
        _json(directory/'status.json',pointer)
        return pointer


def status(directory,root):
    directory,_ = read_snapshot(directory,root)
    result = json.loads((directory/'status.json').read_text(encoding='utf8'))
    if (directory/'results_manifest.json').exists():
        verify_results(directory)
        result['results_verified'] = True
    if (directory/'latest_report.json').exists():
        report = json.loads((directory/'latest_report.json').read_text(encoding='utf8'))
        path = safe_path(root,report['report'])
        if not path.is_relative_to(directory/'reports') or digest(path) != report['report_sha256'] or report['results_manifest_sha256'] != digest(directory/'results_manifest.json'):
            raise ValueError('Report checksum/source mismatch.')
        result['report_verified'] = True
    return result


def resolve(config,root,value=None,create=False):
    if value:
        directory = safe_path(root,value)
    else:
        pointer = root/NAMESPACE/'latest_prepared.json'
        if not pointer.exists():
            if create:
                return prepare(config,root)
            raise ValueError('No prepared experiment; run Prepare first.')
        p = json.loads(pointer.read_text(encoding='utf8'))
        directory = safe_path(root,p['snapshot'])
        if digest(directory/'manifest.json') != p['manifest_sha256']:
            raise ValueError('Prepared snapshot identity changed.')
    _,manifest = read_snapshot(directory,root)
    if manifest['config'] != config:
        raise ValueError('Configuration differs; prepare a new snapshot.')
    return directory
