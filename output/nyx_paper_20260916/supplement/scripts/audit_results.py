"""Independent read-only audit of already saved predictions; no model fitting."""
from pathlib import Path
import argparse
import os
parser = argparse.ArgumentParser(description="Rescore saved NYX outputs and audit provenance; no model fitting or network calls.")
parser.add_argument("--workspace", type=Path, default=Path(os.environ.get("NYX_WORKSPACE", Path.cwd())), help="Authorized local NYX repository root; defaults to NYX_WORKSPACE or current directory.")
args = parser.parse_args()
ROOT = args.workspace.expanduser().resolve()
OUT = Path(__file__).resolve().parent.parent / "recomputed"
OUT.mkdir(parents=True, exist_ok=True)
def archived_path(value):
    """Resolve source-manifest paths relative to the chosen workspace."""
    normalized = str(value).replace(chr(92), "/")
    marker = "/runs/"
    if marker in normalized:
        relative = "runs/" + normalized.split(marker, 1)[1]
    elif normalized.startswith("runs/"):
        relative = normalized
    else:
        raise ValueError("Expected a repository runs/ path in the provenance manifest")
    resolved = (ROOT / relative).resolve()
    if not resolved.is_relative_to(ROOT / "runs"):
        raise ValueError("Source path escapes the repository runs directory")
    return resolved

import hashlib, json, math
import numpy as np
import pandas as pd

SRC = ROOT / 'research/tensor_timesfm_20260915/metrics'
TMP = ROOT / 'tmp/tensor_timesfm_audit/metrics'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def score(f, model, label='observed'):
    g = f.dropna(subset=[model,label]); e = g[model].to_numpy()-g[label].to_numpy()
    return dict(n=len(e),mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.mean(e*e))),bias=float(e.mean()))

f = pd.read_csv(SRC/'verified_hourly_pairs.csv.gz',parse_dates=['timestamp'])
t = pd.read_csv(SRC/'locked_test_predictions.csv.gz',parse_dates=['timestamp'])
manifest=load(SRC/'extraction_manifest.json')
summary=load(SRC/'experiment_summary.json')
selection=load(SRC/'validation_selection_locked.json')
audit={'input_matrix_sha256':sha(SRC/'verified_hourly_pairs.csv.gz'),
       'matrix_hash_matches_summary':sha(SRC/'verified_hourly_pairs.csv.gz')==summary['input_matrix_sha256'],
       'selection_hash_matches_summary':sha(SRC/'validation_selection_locked.json')==summary['validation_selection_sha256'],
       'tmp_research_metrics_byte_equal':{p.name:sha(p)==sha(TMP/p.name) for p in SRC.iterdir() if p.is_file() and (TMP/p.name).exists()},
       'rows':len(f),'duplicate_zone_timestamp':int(f.duplicated(['zone','timestamp']).sum()),
       'zones':sorted(f.zone.unique()),'n_timestamps':int(f.timestamp.nunique()),
       'utc_first':str(f.timestamp.min()),'utc_last':str(f.timestamp.max()),
       'storm_missing':f.loc[f.storm.isna(),['zone','timestamp','day','hour']].astype({'timestamp':str}).to_dict('records'),
       'splits':{s:dict(first=g.day.min(),last=g.day.max(),days=int(g.day.nunique()),n=len(g)) for s,g in f.groupby('split')},
       'quantile_crossings':int(((f.nyx_p10>f.nyx)|(f.nyx>f.nyx_p90)).sum()),
       'run_checks':[]}

for p in manifest['provenance']:
    z=p['zone']; w=archived_path(p['workspace']); b=w/'report_only/frozen_result'
    a=load(b/'audits.json'); replay=a['kalman_replay']; windows=replay['rolling_training_windows']
    bk=pd.read_parquet(b/'kalman_backtest.parquet'); fg=f.loc[f.zone==z].sort_values('timestamp')
    bk=bk.assign(timestamp=pd.to_datetime(bk.delivery_start_utc,utc=True)).sort_values('timestamp')
    columns={'nyx':'residual_kalman__q50','chronos':'chronos2__q50','residual':'residual_corrected__q50','frozen_actual':'actual',
             'nyx_p10':'residual_kalman__q10','nyx_p90':'residual_kalman__q90'}
    src=archived_path(p['source_audit_path']); source_audit=load(src)
    record={'zone':z,'bundle_manifest_hash_ok':sha(b/'manifest.json')==p['bundle_manifest_sha256'],
            'backtest_hash_ok':sha(b/'kalman_backtest.parquet')==p['backtest_sha256'],
            'source_audit_hash_ok':sha(src)==p['source_audit_sha256'],
            'snapshot_observed_hash_ok':sha(src.parent/'inputs/observed_latest.parquet')==p['observed_sha256'],
            'snapshot_storm_hash_ok':sha(src.parent/'inputs/storm_dashboard_official_statistics.parquet')==p['storm_sha256'],
            'timestamp_matches':list(fg.timestamp)==list(bk.timestamp),
            'matrix_vs_bundle_max_abs':{k:float(np.max(np.abs(fg[k].to_numpy()-bk[v].to_numpy()))) for k,v in columns.items()},
            'replay_causality_violations':replay['causality_violations'],
            'target_actuals_assimilated_before_forecast':replay['target_actuals_assimilated_before_forecast'],
            'training_windows_count':len(windows),
            'training_windows_labels_strictly_before_target':all(v['training_window_end']<v['target_day'] for v in windows),
            'window_target_assimilations':sum(v['target_observations_assimilated'] for v in windows),
            'history_caches':a['result'].get('daily_chronos_cache'),
            'warmup_scope':a['result'].get('warmup_scope'),
            'width_change_vs_chronos_max':p['width_change_vs_chronos_max'],
            'selected_filter_counts':replay.get('selected_filter_counts')}
    audit['run_checks'].append(record)

# Annual matched support for ALL four stages, including Storm.
rows=[]
for period,g in [('annual',f),('train',f[f.split=='train']),('validation',f[f.split=='validation']),('test90',f[f.split=='test'])]:
    for support in ['all_nyx_hours','four_model_common']:
        base=g if support=='all_nyx_hours' else g.dropna(subset=['storm','nyx','chronos','residual'])
        for z in ['pooled']+sorted(f.zone.unique()):
            q=base if z=='pooled' else base[base.zone==z]
            for m in ['chronos','residual','nyx','storm']:
                if support=='all_nyx_hours' and m=='storm': continue
                rows.append(dict(period=period,support=support,zone=z,model=m,**score(q,m)))
metrics=pd.DataFrame(rows); metrics.to_csv(OUT/'verified_stage_metrics.csv',index=False)

# Independent moving-block bootstrap: resample days jointly across every zone.
def bootstrap(g, comp, base='nyx', label='observed', block=7, seed=20260915, reps=2000):
    q=g.dropna(subset=[comp,base,label]).copy()
    eb=q[base]-q[label]; ec=q[comp]-q[label]
    q=q.assign(ac=abs(ec),ab=abs(eb),sc=ec*ec,sb=eb*eb,n=1)
    a=q.groupby('day')[['ac','ab','sc','sb','n']].sum().to_numpy(); d=len(a)
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,d-block+1,size=(reps,math.ceil(d/block)))
    ix=(starts[:,:,None]+np.arange(block)[None,None,:]).reshape(reps,-1)[:,:d]
    sampled=a[ix].sum(axis=1); total=a.sum(axis=0)
    dm=(sampled[:,0]-sampled[:,1])/sampled[:,4]
    dr=np.sqrt(sampled[:,2]/sampled[:,4])-np.sqrt(sampled[:,3]/sampled[:,4])
    return dict(n=int(total[4]),days=d,block_days=block,reps=reps,seed=seed,
                mae_delta=float((total[0]-total[1])/total[4]),
                mae_delta_ci_low=float(np.quantile(dm,.025)),mae_delta_ci_high=float(np.quantile(dm,.975)),
                rmse_delta=float(np.sqrt(total[2]/total[4])-np.sqrt(total[3]/total[4])),
                rmse_delta_ci_low=float(np.quantile(dr,.025)),rmse_delta_ci_high=float(np.quantile(dr,.975)))

br=[]
for period,g in [('annual',f),('test90',f[f.split=='test'])]:
    g=g.dropna(subset=['storm','nyx','chronos','residual'])
    for z in ['pooled']+sorted(f.zone.unique()):
        q=g if z=='pooled' else g[g.zone==z]
        for comp,base in [('nyx','chronos'),('residual','chronos'),('nyx','residual'),('nyx','storm')]:
            for block in ([7,14,28] if z=='pooled' else [7]):
                br.append(dict(period=period,zone=z,comparison=f'{comp} minus {base}',**bootstrap(q,comp,base,block=block)))
pd.DataFrame(br).to_csv(OUT/'stage_paired_block_bootstrap.csv',index=False)

# Exact saved residual-experiment score and bootstrap replication, without fitting.
saved=pd.read_csv(SRC/'locked_test_metrics.csv'); savedb=pd.read_csv(SRC/'locked_test_paired_bootstrap.csv')
score_max=0.; boot_max=0.; recomputed=[]
for row in saved.to_dict('records'):
    g=t if row['zone']=='pooled' else t[t.zone==row['zone']]
    label='observed' if row['label']=='latest' else 'frozen_actual'
    measured=score(g,row['model'],label)
    score_max=max(score_max,*[abs(measured[k]-row[k]) for k in measured])
    boot=bootstrap(g,row['model'],label=label)
    prior=savedb[(savedb.model==row['model'])&(savedb.zone==row['zone'])&(savedb.label==row['label'])].iloc[0]
    boot_max=max(boot_max,*[abs(boot[k]-prior[k]) for k in ['n','mae_delta','mae_delta_ci_low','mae_delta_ci_high','rmse_delta','rmse_delta_ci_low','rmse_delta_ci_high']])
    recomputed.append({**row,**boot})
audit['residual_metrics_max_abs_difference']=score_max
audit['residual_bootstrap_max_abs_difference']=boot_max
pd.DataFrame(recomputed).to_csv(OUT/'verified_residual_test_metrics.csv',index=False)
fits=load(SRC/'rolling_fit_audit.json')
audit['residual_refit_audit']={'count':len(fits),'all_fit_last_day_le_permitted':all(v['fit_last_day']<=v['max_permitted_label_day'] for v in fits),
                             'stages':pd.Series([v['stage'] for v in fits]).value_counts().to_dict()}
audit['material_label_revisions']=[]
for z,g in f.groupby('zone'):
    diff=abs(g.observed-g.frozen_actual)
    representation=(g.observed.to_numpy().astype('float32')==g.frozen_actual.to_numpy().astype('float32'))&(diff<=5e-5)
    mat=(diff>1e-9)&~representation
    audit['material_label_revisions'].append({'zone':z,'count':int(mat.sum()),'max_abs':float(diff.max()),'mean_abs':float(diff.mean()),
      
      'annual_mae_latest_minus_frozen':score(g,'nyx','observed')['mae']-score(g,'nyx','frozen_actual')['mae'],
      'test_mae_latest_minus_frozen':score(g[g.split=='test'],'nyx','observed')['mae']-score(g[g.split=='test'],'nyx','frozen_actual')['mae']})
(OUT/'results_audit_checks.json').write_text(json.dumps(audit,indent=2,default=str),encoding='utf-8')
# Requested seasonal-naive comparator: source labels D-7, identical civil hour.
# Repeated lag-hour values averaged; missing lag excluded, never target-imputed.
lags=f.groupby(['zone','day','hour'],as_index=False).frozen_actual.mean()
lags['day']=(pd.to_datetime(lags.day)+pd.Timedelta(days=7)).dt.strftime('%Y-%m-%d')
lags=lags.rename(columns={'frozen_actual':'weekly_naive'})
nf=f.merge(lags,on=['zone','day','hour'],how='left',validate='many_to_one')
ns=[]; nb=[]
for support in ['nyx_naive_common','all_models_common']:
    ng=nf.dropna(subset=['weekly_naive','nyx']+(['chronos','residual','storm'] if support=='all_models_common' else []))
    for z in ['pooled']+sorted(f.zone.unique()):
        g=ng if z=='pooled' else ng[ng.zone==z]
        for label in ['observed','frozen_actual']:
            for model in ['weekly_naive','nyx','chronos','residual']+(['storm'] if support=='all_models_common' else []):
                ns.append(dict(support=support,zone=z,label=label,model=model,days=int(g.day.nunique()),first=g.day.min(),last=g.day.max(),**score(g,model,label)))
        for model in ['nyx','chronos','residual']+(['storm'] if support=='all_models_common' else []):
            nb.append(dict(support=support,zone=z,comparison=f'{model} minus weekly_naive',**bootstrap(g,model,'weekly_naive')))
pd.DataFrame(ns).to_csv(OUT/'weekly_naive_matched_metrics.csv',index=False)
pd.DataFrame(nb).to_csv(OUT/'weekly_naive_paired_bootstrap.csv',index=False)
audit['weekly_naive']={'definition':'frozen_actual from D-7 at same civil hour, repeated source hour averaged, missing source excluded',
 'missing_total':int(nf.weekly_naive.isna().sum()),
 'missing_after_warmup':nf[(nf.day>='2025-09-23')&nf.weekly_naive.isna()][['zone','timestamp','day','hour']].astype({'timestamp':str}).to_dict('records')}
(OUT/'results_audit_checks.json').write_text(json.dumps(audit,indent=2,default=str),encoding='utf-8')
print(json.dumps({'checks':{k:v for k,v in audit.items() if k not in ['run_checks','tmp_research_metrics_byte_equal','material_label_revisions']},
 'run_checks':audit['run_checks'],
 'annual':metrics[(metrics.period=='annual')&(metrics.support=='four_model_common')].to_dict('records'),
 'annual_bootstrap':[r for r in br if r['period']=='annual' and r['zone']=='pooled'],
 'residual_test':[r for r in recomputed if r['label']=='latest' and r['zone']=='pooled']},indent=2))
