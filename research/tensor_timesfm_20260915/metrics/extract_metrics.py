"""Read-only NYX extraction and measured diagnostics; outputs only beside this script."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import psutil

from chronos2_hourly.nuclear_run_archive import load_nuclear_result_bundle
from chronos2_hourly.nuclear_report_benchmark import _load_verified_snapshot
from chronos2_hourly.nuclear_reporting_refresh import verify_refreshed_observations
from chronos2_hourly.model_storm_data import _series, _latest_source

ZONES = ['BE', 'DE', 'FR', 'NL']
TZ = dict(zip(ZONES, ['Europe/Brussels', 'Europe/Berlin', 'Europe/Paris', 'Europe/Amsterdam']))
RUN_DAY = '2026-09-16'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    valid = np.isfinite(y) & np.isfinite(p)
    e = p[valid] - y[valid]
    return {'n': len(e), 'mae': float(np.abs(e).mean()) if len(e) else None,
            'rmse': float(np.sqrt(np.mean(e*e))) if len(e) else None,
            'bias': float(e.mean()) if len(e) else None}

def interval(y, lo, med, hi):
    y, lo, med, hi = [np.asarray(a, float) for a in (y, lo, med, hi)]
    valid = np.isfinite(y) & np.isfinite(lo) & np.isfinite(med) & np.isfinite(hi)
    y, lo, med, hi = [a[valid] for a in (y, lo, med, hi)]
    if not len(y): return {}
    score = hi-lo + 10*np.maximum(lo-y, 0) + 10*np.maximum(y-hi, 0)
    return {'n': len(y), 'coverage80': float(np.mean((y>=lo)&(y<=hi))),
            'below_p10': float(np.mean(y<lo)), 'above_p90': float(np.mean(y>hi)),
            'width80': float(np.mean(hi-lo)), 'interval_score80': float(score.mean()),
            'wis80_single_interval': float(np.mean((.5*np.abs(y-med)+.1*score)/1.5))}

def spectrum(matrix):
    matrix = np.asarray(matrix, float)
    matrix = matrix[np.isfinite(matrix).all(axis=1)]
    centered = matrix-matrix.mean(axis=0)
    eig = np.linalg.eigvalsh(centered.T@centered/(len(centered)-1))[::-1]
    shares = eig/eig.sum()
    return {'n': len(matrix), 'variance_shares': shares.tolist(),
            'participation_rank': float(1/(shares@shares)),
            'entropy_rank': float(np.exp(-np.sum(shares*np.log(np.maximum(shares,1e-30))))),
            'correlation': np.corrcoef(matrix, rowvar=False).tolist()}

def main():
    began=time.perf_counter(); cpu=time.process_time()
    rows=[]; provenance=[]
    for zone in ZONES:
        work=ROOT/'runs/experiments/nuclear_forecast_v1'/RUN_DAY/zone.lower()/'civil_pit_v2'
        result=load_nuclear_result_bundle(workdir=work)
        assert result.audit['zone']==zone and result.audit['delivery_day']==RUN_DAY
        assert result.kalman_view.replay.audit['causality_violations']==0
        assert result.kalman_view.replay.audit['target_actuals_assimilated_before_forecast']==0
        _, ap, audit=_latest_source(work)
        source=ap.parent
        storm, _, storm_provenance=_load_verified_snapshot(source,zone=zone,timezone=TZ[zone])
        observed=_series(pd.read_parquet(source/'inputs/observed_latest.parquet'),'actual')
        verify_refreshed_observations(observed,audit,zone=zone,timezone=TZ[zone],delivery_day=RUN_DAY)
        backtest=result.kalman_view.backtest
        idx=pd.DatetimeIndex(pd.to_datetime(backtest.delivery_start_utc,utc=True))
        frame=pd.DataFrame({'timestamp':idx,'zone':zone,'observed':observed.reindex(idx).to_numpy(),
                            'storm':storm.reindex(idx).to_numpy(),'frozen_actual':backtest.actual.to_numpy()})
        for column, original in [('nyx','residual_kalman__q50'),('nyx_p10','residual_kalman__q10'),
                                 ('nyx_p90','residual_kalman__q90'),('chronos','chronos2__q50'),
                                 ('residual','residual_corrected__q50'),('residual_correction','residual_correction'),
                                 ('kalman_correction','kalman_correction'),('kalman_raw_correction','kalman_raw_correction')]:
            frame[column]=backtest[original].to_numpy()
        frame['day']=idx.tz_convert(TZ[zone]).strftime('%Y-%m-%d')
        frame['hour']=idx.tz_convert(TZ[zone]).hour
        frame['month']=idx.tz_convert(TZ[zone]).month
        frame['season']=frame.month.map({12:'DJF',1:'DJF',2:'DJF',3:'MAM',4:'MAM',5:'MAM',6:'JJA',7:'JJA',8:'JJA',9:'SON',10:'SON',11:'SON'})
        assert frame.observed.notna().all() and frame.nyx.notna().all()
        assert ((frame.nyx_p10<=frame.nyx)&(frame.nyx<=frame.nyx_p90)).all()
        rows.append(frame)
        provenance.append({'zone':zone,'workspace':str(work),'bundle_manifest_sha256':sha(work/'report_only/frozen_result/manifest.json'),
                           'backtest_sha256':sha(work/'report_only/frozen_result/kalman_backtest.parquet'),
                           'source_audit_path':str(ap),'source_audit_sha256':sha(ap),
                           'observed_sha256':audit['observed']['artifact_sha256'],
                           'storm_sha256':storm_provenance['artifact_sha256'],
                           'extracted_at_utc':audit['extracted_at_utc'],'series':audit['observed']['series'],
                           'evaluation_start_day':result.audit['evaluation_start_day'],'evaluation_end_day':result.audit['evaluation_end_day'],
                           'model_future_day':RUN_DAY,'model_future_included':False,
                           'frozen_causality_violations':result.kalman_view.replay.audit['causality_violations'],
                           'width_change_vs_chronos_max':float(np.max(np.abs((backtest['residual_kalman__q90']-backtest['residual_kalman__q10'])-(backtest['chronos2__q90']-backtest['chronos2__q10']))))})
        print('Verified '+zone,flush=True)
    frame=pd.concat(rows,ignore_index=True)
    days=sorted(frame.day.unique()); assert len(days)==365
    # Lock 180 initial days, 95 validation days, and 90 final test days.
    split={'train_days':days[:180],'validation_days':days[180:-90],'test_days':days[-90:]}
    sets={day:name.removesuffix('_days') for name,items in split.items() for day in items}
    frame['split']=frame.day.map(sets)
    assert set(frame['split'])=={'train','validation','test'}
    thresholds={z:{'q95':float(frame.loc[(frame.zone==z)&(frame.split=='train'),'frozen_actual'].quantile(.95)),
                   'q99':float(frame.loc[(frame.zone==z)&(frame.split=='train'),'frozen_actual'].quantile(.99))} for z in ZONES}
    descriptive=[]; intervals=[]; saturation=[]; sensitivity=[]
    for zone in ZONES:
        base=frame.loc[frame.zone==zone]
        revisions=np.abs(base.observed-base.frozen_actual)
        nonexact = revisions.to_numpy() > 1e-9
        representation = (base.observed.to_numpy().astype(np.float32) == base.frozen_actual.to_numpy().astype(np.float32)) & (revisions.to_numpy() <= 5e-5)
        material = nonexact & ~representation
        sensitivity.append({'zone':zone,'hours':len(base),'nonexact_hours_gt_1e_minus9':int(nonexact.sum()),
                            'representation_only_hours':int((nonexact & representation).sum()),
                            'material_revision_hours':int(material.sum()),'material_revision_fraction':float(material.mean()),
                            'material_revision_pairs':base.loc[material,['timestamp','frozen_actual','observed']].assign(absolute_difference=revisions.loc[material]).astype({'timestamp':str}).to_dict('records'),
                            'revision_max':float(revisions.max()),
                            'revision_mae':float(revisions.mean()),'nyx_latest':metrics(base.observed,base.nyx),
                            'nyx_frozen':metrics(base.frozen_actual,base.nyx)})
        groups=[('all','all',base)]
        groups += [('split',str(k),g) for k,g in base.groupby('split')]
        groups += [('hour',str(k),g) for k,g in base.groupby('hour')]
        groups += [('season',str(k),g) for k,g in base.groupby('season')]
        groups += [('regime','negative',base.loc[base.observed<0]),
                   ('regime','upper95_train',base.loc[base.observed>thresholds[zone]['q95']]),
                   ('regime','upper99_train',base.loc[base.observed>thresholds[zone]['q99']])]
        for grouping,label,g in groups:
            for model in ['nyx','storm','chronos','residual']:
                # Match comparator hours so same-group NYX/Storm numbers are paired.
                paired=g.loc[g.storm.notna()] if model in ['nyx','storm'] else g
                descriptive.append({'zone':zone,'grouping':grouping,'group':label,'model':model,
                                    **metrics(paired.observed,paired[model])})
            intervals.append({'zone':zone,'grouping':grouping,'group':label,**interval(g.observed,g.nyx_p10,g.nyx,g.nyx_p90)})
            saturation.append({'zone':zone,'grouping':grouping,'group':label,'n':len(g),
                               'residual_cap40_fraction':float((g.residual_correction.abs()>=40-1e-7).mean()) if len(g) else None,
                               'kalman_cap20_fraction':float((g.kalman_correction.abs()>=20-1e-7).mean()) if len(g) else None,
                               'kalman_raw_above20_fraction':float((g.kalman_raw_correction.abs()>20+1e-7).mean()) if len(g) else None})
    frame['residual_nyx']=frame.nyx-frame.observed
    frame['residual_storm']=frame.storm-frame.observed
    dependency={}
    for split_name in ['all','train','validation','test']:
        part=frame if split_name=='all' else frame.loc[frame.split==split_name]
        matrix=part.pivot(index='timestamp',columns='zone',values='residual_nyx').reindex(columns=ZONES)
        dependency[split_name]={'raw_eur_mwh':spectrum(matrix),
                                'standardized':spectrum((matrix-matrix.mean())/matrix.std()),
                                'nyx_storm_error_correlation':{z:float(g[['residual_nyx','residual_storm']].corr().iloc[0,1]) for z,g in part.groupby('zone')}}
        daily=part.groupby(['day','hour','zone']).residual_nyx.mean().unstack(['hour','zone']).dropna()
        if len(daily)>1:
            centered=daily.to_numpy()-daily.to_numpy().mean(axis=0)
            eig=np.linalg.svd(centered,compute_uv=False)**2
            shares=eig/eig.sum()
            dependency[split_name]['daily_96_axis']={'civil_hour_complete_days':len(daily),'dimensions':daily.shape[1],
                                                   'dst_policy':'Repeated autumn civil hour averaged for this descriptive lag-axis matrix only; spring missing hour drops that day. Physical hourly scores preserve both autumn hours.',
                                                   'first10_variance_shares':shares[:10].tolist(),
                                                   'participation_rank':float(1/(shares@shares))}
    frame.to_csv(OUT/'verified_hourly_pairs.csv.gz',index=False,compression='gzip')
    pd.DataFrame(descriptive).to_csv(OUT/'descriptive_metrics.csv',index=False)
    pd.DataFrame(intervals).to_csv(OUT/'nyx_interval_calibration.csv',index=False)
    pd.DataFrame(saturation).to_csv(OUT/'correction_saturation.csv',index=False)
    pd.DataFrame(sensitivity).to_json(OUT/'label_revision_sensitivity.json',orient='records',indent=2)
    info={'run_day':RUN_DAY,'units':'EUR/MWh','bias_sign':'forecast minus observed',
          'forecast_frequency':'physical hourly UTC; civil-hour groups Europe/Paris',
          'days':len(days),'hourly_rows':len(frame),'timestamp_first':str(frame.timestamp.min()),'timestamp_last':str(frame.timestamp.max()),
          'physical_day_lengths':frame.loc[frame.zone=='BE'].groupby('day').size().value_counts().to_dict(),
          'split':split,'thresholds_initial_train_frozen_actual':thresholds,'provenance':provenance,
          'dependency':dependency,'runtime_seconds':time.perf_counter()-began,'cpu_seconds':time.process_time()-cpu,
          'rss_mb':psutil.Process().memory_info().rss/1024**2,
          'limitations':['Retrospective prequential backtests, not a year of forecasts actually issued live.',
                         'Latest source observations are revised; historical availability at each origin is not recoverable from these extracts.',
                         'Residual experiments must remain exploratory until a prospective or vintage-correct replay is available.',
                         'Storm is the audited official day-ahead comparator, not proven strict-08 input availability.',
                         'Negative regime is actual<0. Peak thresholds are frozen from the first180 training days only.']}
    (OUT/'extraction_manifest.json').write_text(json.dumps(info,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    print(json.dumps({'output':str(OUT),'runtime_seconds':info['runtime_seconds'],'split':{k:[v[0],v[-1],len(v)] for k,v in split.items()},'overall':[r for r in descriptive if r['grouping']=='all' and r['model'] in ['nyx','storm']]}),flush=True)

if __name__=='__main__': main()
