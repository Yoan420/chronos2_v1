"""Rescore saved NYX forecasts through 2026-08-31; no training or reselection."""
from pathlib import Path
import argparse, hashlib, json, math, os
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workspace',type=Path,default=Path(os.environ.get('NYX_WORKSPACE',Path.cwd())))
args=parser.parse_args()
ROOT=args.workspace.resolve(); OUT=Path(__file__).resolve().parent
SRC=ROOT/'research/tensor_timesfm_20260915/metrics'
CUTOFF='2026-08-31'; ZONES=['BE','DE','FR','NL']; SEED=20260915; B=2000
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
def csv(name,rows):
    obj=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    obj.to_csv(OUT/name,index=False); return obj
def archived_path(s): return ROOT/('runs/'+str(s).replace(chr(92),'/').split('/runs/',1)[1])
def met(g,m,label='observed'):
    q=g.dropna(subset=[m,label]); e=q[m].to_numpy()-q[label].to_numpy()
    return dict(n=len(e),mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.mean(e*e))),bias=float(e.mean()))
def interval(g):
    y,l,m,u=[g[k].to_numpy() for k in ['observed','nyx_p10','nyx','nyx_p90']]
    sc=u-l+10*np.maximum(l-y,0)+10*np.maximum(y-u,0)
    return dict(n=len(g),coverage80=float(((y>=l)&(y<=u)).mean()),below_p10=float((y<l).mean()),above_p90=float((y>u).mean()),
                width80=float((u-l).mean()),interval_score80=float(sc.mean()),wis80_single_interval=float(((.5*abs(y-m)+.1*sc)/1.5).mean()))
def bootstrap(g,comp,base='nyx',label='observed',block=7):
    q=g.dropna(subset=[comp,base,label]).copy(); ec=q[comp]-q[label]; eb=q[base]-q[label]
    q=q.assign(ac=abs(ec),ab=abs(eb),sc=ec*ec,sb=eb*eb,n=1)
    a=q.groupby('day')[['ac','ab','sc','sb','n']].sum().to_numpy(); d=len(a)
    starts=np.random.default_rng(SEED).integers(0,d-block+1,size=(B,math.ceil(d/block)))
    ix=(starts[:,:,None]+np.arange(block)[None,None,:]).reshape(B,-1)[:,:d]
    samples=a[ix].sum(axis=1); total=a.sum(axis=0)
    dm=(samples[:,0]-samples[:,1])/samples[:,4]; dr=np.sqrt(samples[:,2]/samples[:,4])-np.sqrt(samples[:,3]/samples[:,4])
    return dict(n=int(total[4]),days=d,block_days=block,reps=B,seed=SEED,
      mae_delta=float((total[0]-total[1])/total[4]),mae_delta_ci_low=float(np.quantile(dm,.025)),mae_delta_ci_high=float(np.quantile(dm,.975)),
      rmse_delta=float(np.sqrt(total[2]/total[4])-np.sqrt(total[3]/total[4])),rmse_delta_ci_low=float(np.quantile(dr,.025)),rmse_delta_ci_high=float(np.quantile(dr,.975)))

full=pd.read_csv(SRC/'verified_hourly_pairs.csv.gz',parse_dates=['timestamp'])
f=full.loc[full.day<=CUTOFF].copy()
test_full=pd.read_csv(SRC/'locked_test_predictions.csv.gz',parse_dates=['timestamp'])
t=test_full.loc[test_full.day<=CUTOFF].copy()
manifest=load(SRC/'extraction_manifest.json'); selection=load(SRC/'validation_selection_locked.json')
assert len(f)==33600 and f.day.nunique()==350 and len(t)==7200 and t.day.nunique()==75
assert set(f.zone)==set(ZONES) and not f.duplicated(['zone','timestamp']).any()
assert f.day.min()=='2025-09-16' and t.day.min()=='2026-06-18'
assert f.day.max()==t.day.max()==CUTOFF
assert selection['primary_family_selected_before_test']=='nyx'
thresholds=manifest['thresholds_initial_train_frozen_actual']
dump('validation_selection_locked.json',selection)
f.to_csv(OUT/'verified_hourly_pairs.csv.gz',index=False,compression={'method':'gzip','mtime':0})
t.to_csv(OUT/'locked_test_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})

# Descriptive and staged scores, always state the finite support.
periods=[('full350',f),('train',f[f.split=='train']),('validation',f[f.split=='validation']),('test75',f[f.split=='test'])]
stages=[]; blocks=[]; description=[]; descriptives=[]; intervals=[]; saturation=[]; extremes=[]; monthly=[]
for period,g in periods:
    for support in ['all_nyx_hours','four_model_common']:
        q=g if support=='all_nyx_hours' else g.dropna(subset=['storm','nyx','chronos','residual'])
        for z in ['pooled']+ZONES:
            h=q if z=='pooled' else q[q.zone==z]
            for m in ['chronos','residual','nyx','storm']:
                if support=='all_nyx_hours' and m=='storm':continue
                stages.append(dict(period=period,support=support,zone=z,model=m,**met(h,m)))
    if period in ['full350','test75']:
        q=g.dropna(subset=['storm','nyx','chronos','residual'])
        for z in ['pooled']+ZONES:
            h=q if z=='pooled' else q[q.zone==z]
            for comp,base in [('nyx','chronos'),('residual','chronos'),('nyx','residual'),('nyx','storm')]:
                blocks.append(dict(period=period,zone=z,comparison=f'{comp} minus {base}',**bootstrap(h,comp,base)))
for z in ZONES:
    g=f[f.zone==z]; y=g.observed
    description.append(dict(zone=z,n=len(g),mean=y.mean(),sd=y.std(),min=y.min(),median=y.median(),max=y.max(),negative_hours=int((y<0).sum())))
    groups=[('all','all',g)]
    groups += [('split',str(k),h) for k,h in g.groupby('split')]
    groups += [('hour',str(k),h) for k,h in g.groupby('hour')]
    groups += [('season',str(k),h) for k,h in g.groupby('season')]
    groups += [('regime','negative',g[g.observed<0]),('regime','upper95_train',g[g.observed>thresholds[z]['q95']]),('regime','upper99_train',g[g.observed>thresholds[z]['q99']])]
    for grouping,group,h in groups:
        # Unlike legacy descriptive output, every four-stage row shares support.
        q=h.dropna(subset=['storm','nyx','chronos','residual'])
        for m in ['nyx','storm','chronos','residual']:descriptives.append(dict(zone=z,grouping=grouping,group=group,model=m,**met(q,m)))
        intervals.append(dict(zone=z,grouping=grouping,group=group,**interval(h)))
        saturation.append(dict(zone=z,grouping=grouping,group=group,n=len(h),residual_cap40_fraction=float((h.residual_correction.abs()>=40-1e-7).mean()),
          kalman_cap20_fraction=float((h.kalman_correction.abs()>=20-1e-7).mean()),kalman_raw_above20_fraction=float((h.kalman_raw_correction.abs()>20+1e-7).mean())))
    for name,h in [('negative',g[g.observed<0]),('high',g[g.observed>thresholds[z]['q99']])]:
        extremes.append(dict(zone=z,regime=name,q99_initial=thresholds[z]['q99'],**met(h,'nyx'),coverage80_pct=100*interval(h)['coverage80']))
    q=g.dropna(subset=['nyx','chronos','storm']).copy();q['month_label']=q.day.str[:7]
    for month,h in q.groupby('month_label'):
        for m in ['chronos','nyx','storm']:monthly.append(dict(zone=z,model=m,month=month,n=len(h),mae=met(h,m)['mae']))
stage=csv('verified_stage_metrics.csv',stages)
csv('annual_paired_metrics.csv',stage[(stage.period=='full350')&(stage.support=='four_model_common')&(stage.zone!='pooled')].drop(columns=['period','support']))
csv('stage_paired_block_bootstrap.csv',blocks); data=csv('data_description.csv',description)
csv('descriptive_metrics.csv',descriptives); cal=csv('nyx_interval_calibration.csv',intervals)
csv('correction_saturation.csv',saturation); ext=csv('extreme_regimes.csv',extremes); csv('monthly_mae.csv',monthly)
vint=[]
for z in ['pooled']+ZONES:
    h=f if z=='pooled' else f[f.zone==z]; q=interval(h)
    vint.append(dict(zone=z,n=q['n'],coverage80=q['coverage80'],below_p10=q['below_p10'],above_p90=q['above_p90'],mean_width=q['width80'],interval_score80=q['interval_score80'],wis_one80=q['wis80_single_interval']))
csv('verified_interval_metrics.csv',vint)

# Frozen residual configurations and predictions: just truncate, never fit/select.
residual=[]; rb=[]; rg=[]
models=['nyx','ewma_country','ewma_country_hour','ridge_univariate','ridge_multivariate','pca_ridge','storm']
for m in models:
    for lab,col in [('latest','observed'),('frozen','frozen_actual')]:
        for z in ['pooled']+ZONES:
            g=t if z=='pooled' else t[t.zone==z]
            residual.append(dict(model=m,label=lab,zone=z,**met(g,m,col)))
            rb.append(dict(model=m,label=lab,zone=z,**bootstrap(g,m,label=col)))
    for z in ZONES:
        g=t[t.zone==z]
        for grouping,group,h in [('regime','negative',g[g.observed<0]),('regime','upper95_train',g[g.observed>thresholds[z]['q95']]),('regime','upper99_train',g[g.observed>thresholds[z]['q99']])]+[('hour',str(k),h) for k,h in g.groupby('hour')]:
            rg.append(dict(model=m,zone=z,grouping=grouping,group=group,**met(h,m)))
rt=csv('locked_test_metrics.csv',residual); rboot=csv('locked_test_paired_bootstrap.csv',rb);csv('locked_test_group_metrics.csv',rg)
csv('verified_residual_test_metrics.csv',rt.merge(rboot.drop(columns=['n']),on=['model','label','zone'],validate='one_to_one'))

# Weekly naive from D-7 frozen labels, physical target hours retained.
lags=f.groupby(['zone','day','hour'],as_index=False).frozen_actual.mean()
lags['day']=(pd.to_datetime(lags.day)+pd.Timedelta(days=7)).dt.strftime('%Y-%m-%d')
nf=f.merge(lags.rename(columns={'frozen_actual':'weekly_naive'}),on=['zone','day','hour'],how='left',validate='many_to_one')
ns=[]; nb=[]
for support in ['nyx_naive_common','all_models_common']:
    g=nf.dropna(subset=['weekly_naive','nyx']+(['chronos','residual','storm'] if support=='all_models_common' else []))
    for z in ['pooled']+ZONES:
        q=g if z=='pooled' else g[g.zone==z]
        for lab in ['observed','frozen_actual']:
            for m in ['weekly_naive','nyx','chronos','residual']+(['storm'] if support=='all_models_common' else []):
                ns.append(dict(support=support,zone=z,label=lab,model=m,days=q.day.nunique(),first=q.day.min(),last=q.day.max(),**met(q,m,lab)))
        for m in ['nyx','chronos','residual']+(['storm'] if support=='all_models_common' else []):nb.append(dict(support=support,zone=z,comparison=f'{m} minus weekly_naive',**bootstrap(q,m,'weekly_naive')))
csv('weekly_naive_matched_metrics.csv',ns);csv('weekly_naive_paired_bootstrap.csv',nb)

# Source identity and cutoff-aware fallback masks. Do not redate snapshots.
provenance=[]; hashes=[]; governance=[]; ukf_days=[]; f['storm_fallback']=False;f['label_fallback']=False
for p in manifest['provenance']:
    z=p['zone']; path=archived_path(p['source_audit_path']);a=load(path);s=a['storm_dashboard']['source'];o=a['observed']['source'].get('post_auction_fallback',{})
    assert sha(path)==p['source_audit_sha256']
    w=archived_path(p['workspace']); b=w/'report_only/frozen_result'; ar=load(b/'audits.json'); windows=ar['kalman_replay']['rolling_training_windows']
    kept=[v for v in windows if v['target_day']<=CUTOFF]; assert len(kept)==350
    assert all(v['training_window_end']<v['target_day'] and v['target_observations_assimilated']==0 for v in kept)
    bk=pd.read_parquet(b/'kalman_backtest.parquet')
    bk['day']=pd.to_datetime(bk.delivery_start_utc,utc=True).dt.tz_convert(a['timezone']).dt.strftime('%Y-%m-%d')
    bk=bk[bk.day<=CUTOFF]
    assert (bk.groupby('day').kalman_selected_filter.nunique()==1).all()
    for period,q in [('full350',bk),('test75',bk[bk.day>='2026-06-18'])]:
        for kind,h in q.groupby('kalman_selected_filter'):
            governance.append(dict(zone=z,period=period,selected_filter=kind,days=h.day.nunique(),physical_hours=len(h)))
    for day,h in bk[bk.kalman_selected_filter=='ukf_scale'].groupby('day'):
        ukf_days.append(dict(zone=z,day=day,physical_hours=len(h),weight_min=h.kalman_weight.min(),weight_max=h.kalman_weight.max(),shift_min=h.kalman_correction.min(),shift_max=h.kalman_correction.max(),sum_absolute_shift=h.kalman_correction.abs().sum(),in_test75=day>='2026-06-18'))
    for fp,key in [(b/'manifest.json','bundle_manifest_sha256'),(b/'kalman_backtest.parquet','backtest_sha256'),(path,'source_audit_sha256'),(path.parent/'inputs/observed_latest.parquet','observed_sha256'),(path.parent/'inputs/storm_dashboard_official_statistics.parquet','storm_sha256')]:
        digest=sha(fp);assert digest==p[key];hashes.append(dict(path=fp.relative_to(ROOT).as_posix(),sha256=digest))
    expected=pd.date_range(a['storm_dashboard']['period']['start_utc'],a['storm_dashboard']['period']['end_utc'],freq='h')
    affected=expected.tz_convert(a['timezone']).strftime('%Y-%m-%d').isin(s['fallback_used_local_days'])
    assert int(affected.sum())==s['fallback_used_hours']
    f.loc[f.zone==z,'storm_fallback']=f.loc[f.zone==z,'day'].isin(s['fallback_used_local_days'])
    f.loc[f.zone==z,'label_fallback']=f.loc[f.zone==z,'timestamp'].isin(pd.to_datetime(o.get('applied_value_times_utc',[]),utc=True))
    g=f[f.zone==z];q=g[g.split=='test']
    provenance.append(dict(zone=z,snapshot_extracted_at_utc=a['extracted_at_utc'],snapshot_hours=len(expected),storm_snapshot_fallback_hours=s['fallback_used_hours'],
      full350_storm_fallback_hours=int(g.storm_fallback.sum()),test75_storm_fallback_hours=int(q.storm_fallback.sum()),
      full350_storm_fallback_fraction_of_paired=float(g.storm_fallback.sum()/g.storm.notna().sum()),test75_storm_fallback_fraction=float(q.storm_fallback.mean()),
      full350_label_fallback_hours=int(g.label_fallback.sum()),test75_label_fallback_hours=int(q.label_fallback.sum()),
      fallback_count_equals_full_day_capacity=True,storm_fallback_live_recomputation=s['fallback_live_recomputation'],retained_kalman_windows=len(kept),retained_window_checks_pass=True))
assert f.label_fallback.sum()==0
csv('source_fallback_counts.csv',provenance)
csv('governance_selected_filter_counts.csv',governance);csv('ukf_selected_days.csv',ukf_days)
sens=[]
for period,g in [('full350',f),('test75',f[f.split=='test'])]:
    g=g.dropna(subset=['nyx','storm'])
    for subset,mask in [('all_common',np.ones(len(g),bool)),('cache_only',~g.storm_fallback),('native_fallback_only',g.storm_fallback),('cache_only_no_label_fallback',~g.storm_fallback&~g.label_fallback)]:
        h=g.loc[mask]
        for z in ['pooled']+ZONES:
            q=h if z=='pooled' else h[h.zone==z]
            for m in ['nyx','storm']:sens.append(dict(period=period,subset=subset,zone=z,model=m,days=q.day.nunique(),**met(q,m)))
sn=csv('storm_provenance_sensitivity.csv',sens)

# Fit records contributing to the truncated predictions (no new fitting).
fits=load(SRC/'rolling_fit_audit.json'); fitrows=[]
for v in fits:
    if v['refit_day']>CUTOFF:continue
    d=pd.Timestamp(v['refit_day']); end=min(d+pd.Timedelta(days=6),pd.Timestamp(CUTOFF),pd.Timestamp('2026-06-17') if v['stage']=='validation' else pd.Timestamp(CUTOFF))
    kept_days=pd.date_range(d,end,freq='D').strftime('%Y-%m-%d')
    count=int(f[(f.zone=='BE')&f.day.isin(kept_days)].shape[0])
    assert v['fit_last_day']<=v['max_permitted_label_day'] and v['fit_last_day']<v['refit_day']
    fitrows.append(dict(stage=v['stage'],family=v['spec']['family'],spec=json.dumps(v['spec'],sort_keys=True),refit_day=v['refit_day'],fit_first_day=v['fit_first_day'],fit_last_day=v['fit_last_day'],max_permitted_label_day=v['max_permitted_label_day'],fit_physical_hours=v['fit_physical_hours'],original_predicted_physical_hours=v['predicted_physical_hours'],retained_predicted_physical_hours=count))
fitsdf=csv('retained_fit_audit.csv',fitrows);assert len(fitsdf)==145 and (fitsdf.stage=='test').sum()==33
revision=[]
for z,g in f.groupby('zone'):
    diff=abs(g.observed-g.frozen_actual); representation=(g.observed.to_numpy().astype('float32')==g.frozen_actual.to_numpy().astype('float32'))&(diff<=5e-5)
    material=(diff>1e-9)&~representation
    revision.append(dict(zone=z,n=len(g),material_revision_hours=int(material.sum()),material_dates=sorted(g.loc[material,'day'].unique()),max_abs_difference=float(diff.max()),mean_abs_difference=float(diff.mean()),
      nyx_mae_latest_minus_frozen=met(g,'nyx')['mae']-met(g,'nyx','frozen_actual')['mae'],test_mae_latest_minus_frozen=met(g[g.split=='test'],'nyx')['mae']-met(g[g.split=='test'],'nyx','frozen_actual')['mae']))
dump('label_revision_sensitivity.json',revision)
train=f[f.split=='train'].pivot(index='timestamp',columns='zone',values='residual_nyx')[ZONES]
cov=np.cov(train.to_numpy(),rowvar=False); eig=np.linalg.eigvalsh(cov)[::-1];share=eig/eig.sum()
zcov=np.cov(((train-train.mean())/train.std()).to_numpy(),rowvar=False);zeig=np.linalg.eigvalsh(zcov)[::-1];zs=zeig/zeig.sum()
dep=dict(initial_train_days=180,physical_hours_per_country=len(train),zones=ZONES,correlation=train.corr().to_numpy().tolist(),variance_shares=share.tolist(),participation_rank=float(1/(share@share)),standardized_variance_shares=zs.tolist())
dump('initial_train_dependence.json',dep);csv('initial_train_residual_correlation.csv',train.corr().reset_index(names='zone'))
extracted=dict(run_day=manifest['run_day'],analysis_cutoff_local=CUTOFF,source_matrix_sha256=sha(SRC/'verified_hourly_pairs.csv.gz'),
  source_manifest_sha256=sha(SRC/'extraction_manifest.json'),units='EUR/MWh',bias_sign='forecast minus observed',
  days=350,hourly_rows=len(f),timestamp_first=str(f.timestamp.min()),timestamp_last=str(f.timestamp.max()),
  physical_day_lengths=f[f.zone=='BE'].groupby('day').size().value_counts().to_dict(),
  split={key:[d for d in vals if d<=CUTOFF] for key,vals in manifest['split'].items()},
  thresholds_initial_train_frozen_actual=thresholds,provenance=manifest['provenance'],
  dependency_scope='Initial-train covariance recomputed in initial_train_dependence.json; no stale full-year dependency values copied.',
  limitations=['Evaluation timestamps truncated to 2026-08-31; source snapshots were materialized in September 2026.',
               'Original provenance metadata retain original archive evaluation dates; these are not the truncated scoring window.',
               'No model training, parameter reselection or claim of archive availability on the evaluation cutoff.'])
dump('extraction_manifest.json',extracted)

# Ten manuscript tables: retain purely methodological tables, replace ALL result cells.
tables=load(ROOT/'output/nyx_paper_20260916/tables/manuscript_tables.json')
fmt=lambda x,n=3:f'{x:.{n}f}'
tables['data_description']['rows']=[[r.zone,fmt(r['mean'],2),fmt(r.sd,2),fmt(r['min'],2),fmt(r['median'],2),fmt(r['max'],2),str(int(r.negative_hours))] for _,r in data.iterrows()]
tables['data_description']['note']='Prices in EUR/MWh; sample standard deviation. Each country has 8,400 observations over 350 delivery days, 16 September 2025–31 August 2026. Negative hours have observed price below zero.'
a=stage[(stage.period=='full350')&(stage.support=='four_model_common')]
def row(z,m):return a[(a.zone==z)&(a.model==m)].iloc[0]
tables['annual_stages']['caption']='Table 4. MAE across the retained NYX stages through 31 August 2026'
tables['annual_stages']['rows']=[[z.upper() if z=='pooled' else z,*[fmt(row(z,m).mae) for m in ['chronos','residual','nyx']],fmt(100*(1-row(z,'nyx').mae/row(z,'chronos').mae),2)] for z in ZONES+['pooled']]
tables['annual_stages']['note']='MAE in EUR/MWh. Same 8,399 hours per country as Table 5; pooled n = 33,596. Reduction is relative to Chronos-2. This is a 350-day retrospective evaluation, not a full annual cycle.'
tables['annual_benchmark']['caption']='Table 5. NYX and Storm reporting-composite performance over 350 days'
tables['annual_benchmark']['rows']=[[z.upper() if z=='pooled' else z,*[fmt(row(z,m)[metric]) for metric in ['mae','rmse','bias'] for m in ['nyx','storm']]] for z in ZONES+['pooled']]
tables['annual_benchmark']['note']='All errors in EUR/MWh on 33,596 common country–hour pairs through 31 August 2026. Bias = forecast minus observation. Storm includes recomputed fallback values; Table 9 excludes them in sensitivity analysis.'
tables['extremes']['rows']=[[r.zone,'Negative' if r.regime=='negative' else 'Above initial q99',str(int(r.n)),fmt(r.mae,2),fmt(r.bias,2),fmt(r.coverage80_pct,2)] for _,r in ext.iterrows()]
ca=cal[(cal.grouping=='all')]
tables['intervals']['caption']='Table 7. NYX interval evaluation over 350 days'
tables['intervals']['rows']=[[r.zone,fmt(100*r.coverage80,2),fmt(r.width80,2),fmt(r.interval_score80,2),fmt(r.wis80_single_interval)] for _,r in ca.iterrows()]
tables['intervals']['note']='Coverage target: 80%; n = 8,400 per country. Width and scores in EUR/MWh. WIS80 uses one interval plus the median, not a complete-distribution score.'
tables['residual_test']['caption']='Table 8. Previously validation-selected residual configurations on the retained 75 test days'
names={'nyx':'NYX (identity)','ewma_country':'EWMA country (28 d)','ewma_country_hour':'EWMA country × hour (28 d)','ridge_univariate':'Univariate ridge (α = 100)','ridge_multivariate':'Multivariate ridge (α = 100)','pca_ridge':'PCA ridge (rank 1, α = 100)','storm':'Storm reporting composite'}
rr=[]
for m in models:
    r=rt[(rt.model==m)&(rt.label=='latest')&(rt.zone=='pooled')].iloc[0]; b=rboot[(rboot.model==m)&(rboot.label=='latest')&(rboot.zone=='pooled')].iloc[0]
    delta='0 (reference)' if m=='nyx' else f'{b.mae_delta:+.3f}\n[{b.mae_delta_ci_low:+.3f}, {b.mae_delta_ci_high:+.3f}]'
    rr.append([names[m],fmt(selection['selected_per_family'][m]['mae']) if m!='storm' else '—',fmt(r.mae),fmt(r.rmse),fmt(r.bias),delta])
tables['residual_test']['rows']=rr
tables['residual_test']['note']='n = 7,200 country–hours, 18 June–31 August 2026. Previously saved configurations and predictions are truncated without reselection or retraining. Validation uses frozen labels; test uses refreshed labels. Positive delta is worse than NYX; 2,000 paired seven-day moving-block replicates. Storm is excluded from model selection.'
ss=[]
for period,label in [('full350','Full 350 d'),('test75','Test 75 d')]:
    for subset,name in [('all_common','All common hours'),('cache_only','Exclude Storm recomputation')]:
        q=sn[(sn.period==period)&(sn.subset==subset)&(sn.zone=='pooled')].set_index('model')
        ss.append([label,name,f'{int(q.loc["nyx","n"]):,}',fmt(q.loc['nyx','mae']),fmt(q.loc['storm','mae'])])
tables['provenance_sensitivity']['rows']=ss
tables['provenance_sensitivity']['note']='Paired within each row; country–hour weighting. No EPEX label fallback occurs on or before 31 August, so an additional label-fallback exclusion gives identical scores. Cache-only values still lack independently verified historical strict-08 availability.'
dump('manuscript_tables.json',tables)

inputs=['verified_hourly_pairs.csv.gz','locked_test_predictions.csv.gz','extraction_manifest.json','validation_selection_locked.json','experiment_summary.json','rolling_fit_audit.json']
for name in inputs:hashes.append(dict(path=(SRC/name).relative_to(ROOT).as_posix(),sha256=sha(SRC/name)))
info=dict(evaluation_cutoff_local=CUTOFF,evaluation_start_local=f.day.min(),civil_days=350,physical_hours_per_country=8400,country_hour_pairs=33600,common_country_hour_pairs=33596,
  first_utc=str(f.timestamp.min()),last_utc=str(f.timestamp.max()),day_lengths=f[f.zone=='BE'].groupby('day').size().value_counts().to_dict(),
  split={s:dict(first=g.day.min(),last=g.day.max(),days=g.day.nunique(),country_hour_pairs=len(g)) for s,g in f.groupby('split')},
  bootstrap=dict(repetitions=B,block_days=7,seed=SEED,shared_days_across_countries=True),
  residual_configuration_selection='original saved validation selection; no reselection or refitting',
  original_source_test_days=90,retained_test_days=75,original_fit_count=len(fits),retained_fit_count=len(fitsdf),retained_validation_fits=112,retained_test_fits=33,
  final_test_refit_day='2026-08-27',final_test_refit_last_training_day='2026-08-25',final_test_refit_retained_forecast_days=5,
  original_run_delivery='2026-09-16',snapshot_materialization='2026-09-15',snapshot_available_on_cutoff_proven=False,
  label_versions='retrospectively frozen_actual and refreshed observed snapshots; no assertion of August-31 availability',
  no_provider_label_fallback_in_evaluation=bool(f.label_fallback.sum()==0),source_hashes=hashes,
  retained_ukf_selected_country_days=sum(v['days'] for v in governance if v['period']=='full350' and v['selected_filter']=='ukf_scale'),
  retained_ukf_selected_hours=sum(v['physical_hours'] for v in ukf_days),ukf_selected_days=ukf_days,
  thresholds_initial_train_frozen_actual=thresholds,initial_train_dependence=dep,
  source_selection_json_sha256=sha(SRC/'validation_selection_locked.json'),copied_selection_json_semantically_identical=selection==load(OUT/'validation_selection_locked.json'))
dump('cutoff_manifest.json',info)
outputs={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in OUT.iterdir() if p.is_file() and p.name not in ['output_sha256_manifest.json','cutoff_results.md']}
dump('output_sha256_manifest.json',outputs)
print(json.dumps({'manifest':{k:v for k,v in info.items() if k not in ['source_hashes','initial_train_dependence']},
 'main':a[a.zone=='pooled'].to_dict('records'),'test':rt[(rt.zone=='pooled')&(rt.label=='latest')].to_dict('records'),
 'test_intervals':rboot[(rboot.zone=='pooled')&(rboot.label=='latest')].to_dict('records'),
 'full_bootstrap':[r for r in blocks if r['period']=='full350' and r['zone']=='pooled'],
 'intervals':vint,'provenance':provenance},indent=2,default=str))
