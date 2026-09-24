"""Small, retrospective, time-ordered residual experiments on verified frozen NYX.

Never imports model training or market clients. Writes only into this directory.
Labels for day D are usable no later than D-2; this is not an observation-vintage proof.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import psutil
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ZONES = ['BE', 'DE', 'FR', 'NL']
SEED = 20260915
BOOTSTRAPS = 2000
BLOCK_DAYS = 7


def scores(y, p):
    good = np.isfinite(y) & np.isfinite(p)
    err = (p-y)[good]
    return {'n': int(good.sum()), 'mae': float(np.abs(err).mean()),
            'rmse': float(np.sqrt(np.mean(err**2))), 'bias': float(err.mean())}


def dump(name, data):
    (OUT/name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


class Experiment:
    def __init__(self):
        frame = pd.read_csv(OUT/'verified_hourly_pairs.csv.gz', parse_dates=['timestamp'])
        self.frame = frame
        self.manifest = json.loads((OUT/'extraction_manifest.json').read_text(encoding='utf-8'))
        self.days = sorted(frame.day.unique())
        self.timestamps = pd.DatetimeIndex(sorted(frame.timestamp.unique()))
        common = frame.loc[frame.zone=='BE'].set_index('timestamp').reindex(self.timestamps)
        self.di = pd.Index(self.days).get_indexer(common.day)
        self.hour = common.hour.to_numpy(int)
        self.actual = self.wide('observed')
        self.frozen = self.wide('frozen_actual')
        self.nyx = self.wide('nyx')
        self.storm = self.wide('storm')
        self.error = self.nyx-self.frozen
        assert len(self.timestamps)==8760 and len(self.days)==365
        assert np.isfinite(self.error).all() and np.isfinite(self.actual).all()
        self.daily = np.empty((365, 4))
        self.byhour = np.full((365, 24, 4), np.nan)
        for day in range(365):
            mask = self.di==day
            self.daily[day] = self.error[mask].mean(axis=0)
            for h in np.unique(self.hour[mask]):
                self.byhour[day,h] = self.error[mask & (self.hour==h)].mean(axis=0)
        self.ewmas = {hl:self.ewma(hl) for hl in [7,14,28]}
        lag = np.full_like(self.byhour, np.nan)
        lag[2:] = self.byhour[:-2]
        country, hour = self.ewmas[14]
        self.X = np.column_stack([lag[self.di,self.hour], country[self.di], hour[self.di,self.hour]])
        self.fit_audit = []

    def wide(self, column):
        return self.frame.pivot(index='timestamp', columns='zone', values=column).reindex(index=self.timestamps,columns=ZONES).to_numpy(float)

    def ewma(self, halflife):
        alpha = 1-np.exp(-np.log(2)/halflife)
        c = np.full((365, 4), np.nan)
        h = np.full((365, 24, 4), np.nan)
        cstate = np.full(4, np.nan)
        hstate = np.full((24, 4), np.nan)
        for d in range(365):
            # Explicitly assimilate D-2, never D-1 or D; all target DST hours retained.
            if d>=2:
                latest = self.daily[d-2]
                cstate = np.where(np.isfinite(cstate),(1-alpha)*cstate+alpha*latest,latest)
                latesth = self.byhour[d-2]
                hstate = np.where(np.isfinite(latesth),
                                  np.where(np.isfinite(hstate),(1-alpha)*hstate+alpha*latesth,latesth),hstate)
            c[d],h[d] = cstate,hstate
        return c,h

    def model_prediction(self, spec, start, stop, stage):
        family = spec['family']
        wanted = (self.di>=start)&(self.di<stop)
        result = np.full_like(self.error, np.nan)
        if family=='nyx':
            result[wanted] = 0
            return result
        if family.startswith('ewma'):
            c,h = self.ewmas[spec['halflife']]
            pred = c[self.di] if family=='ewma_country' else h[self.di,self.hour]
            result[wanted] = pred[wanted]
            assert np.isfinite(result[wanted]).all()
            return result
        for refit in range(start, stop, 7):
            # Frozen protocol: up to 180 eligible days after warm-up, through D-2.
            train = (self.di>=max(14,refit-181)) & (self.di<=refit-2)
            predmask = (self.di>=refit)&(self.di<min(refit+7,stop))
            xfit, yfit = self.X[train],self.error[train]
            assert self.di[train].max()<=refit-2
            impute = np.nanmean(xfit,axis=0)
            assert np.isfinite(impute).all()
            xfit = np.where(np.isfinite(xfit),xfit,impute)
            xp = np.where(np.isfinite(self.X[predmask]),self.X[predmask],impute)
            alpha = spec['alpha']
            components = None
            target_mean,target_scale = yfit.mean(axis=0),yfit.std(axis=0)
            target_scale = np.maximum(target_scale,1e-8)
            if family=='pca_ridge':
                yz = (yfit-target_mean)/target_scale
                _, _, vt = np.linalg.svd(yz,full_matrices=False)
                components = vt[:spec['rank']].T
                xfit = np.column_stack([((xfit[:,k:k+4]-target_mean)/target_scale)@components for k in [0,4,8]])
                xp = np.column_stack([((xp[:,k:k+4]-target_mean)/target_scale)@components for k in [0,4,8]])
                ylatent = yz@components
            mean,scale = xfit.mean(axis=0),np.maximum(xfit.std(axis=0),1e-8)
            zfit,zp = (xfit-mean)/scale,(xp-mean)/scale
            if family=='ridge_univariate':
                pred = np.empty((len(xp),4))
                for z in range(4):
                    keep = [z,z+4,z+8]
                    model = Ridge(alpha=alpha,solver='svd').fit(zfit[:,keep],yfit[:,z])
                    pred[:,z] = model.predict(zp[:,keep])
            elif family=='ridge_multivariate':
                model = Ridge(alpha=alpha,solver='svd').fit(zfit,yfit)
                pred = model.predict(zp)
            elif family=='pca_ridge':
                model = Ridge(alpha=alpha,solver='svd').fit(zfit,ylatent)
                latent_prediction = np.asarray(model.predict(zp)).reshape(len(zp),spec['rank'])
                pred = (latent_prediction@components.T)*target_scale+target_mean
            else:
                raise ValueError(family)
            result[predmask] = pred
            self.fit_audit.append({'stage':stage,'spec':spec,'refit_day':self.days[refit],
                                   'fit_first_day':self.days[int(self.di[train].min())],
                                   'fit_last_day':self.days[int(self.di[train].max())],
                                   'max_permitted_label_day':self.days[refit-2],
                                   'fit_physical_hours':int(train.sum()),'predicted_physical_hours':int(predmask.sum()),
                                   'imputation_training_only':True,'normalization_training_only':True,
                                   'pca_training_only':family=='pca_ridge',
                                   'pca_components':components.tolist() if components is not None else None,
                                   'input_normalization_mean':mean.tolist(),'input_normalization_scale':scale.tolist(),
                                   'imputation_values':impute.tolist(),
                                   'target_mean':target_mean.tolist(),'target_scale':target_scale.tolist()})
        assert np.isfinite(result[wanted]).all()
        return result

    def causal_checks(self):
        # Meaningful influence check: changing D-1 through the end cannot affect D features.
        probe=210
        original = self.error.copy()
        daily, byhour = self.daily.copy(),self.byhour.copy()
        self.error[self.di>=probe-1] += 10000
        self.daily[probe-1:] += 10000
        self.byhour[probe-1:] += 10000
        changed = self.ewma(14)
        assert np.array_equal(changed[0][probe],self.ewmas[14][0][probe])
        assert np.array_equal(changed[1][probe],self.ewmas[14][1][probe])
        self.error,self.daily,self.byhour = original,daily,byhour
        assert len(set(self.manifest['split']['train_days']) & set(self.manifest['split']['test_days']))==0
        assert len(self.manifest['split']['test_days'])==90


def paired_bootstrap(days, y, base, competitor, rng_indices):
    """Moving contiguous 7-day blocks, common paired support, nonlinear RMSE recomputed."""
    good = np.isfinite(y)&np.isfinite(base)&np.isfinite(competitor)
    sums = np.zeros((90,5))
    for i,d in enumerate(range(275,365)):
        mask = (days==d)
        valid = good[mask]
        eb = (base[mask]-y[mask])[valid]
        ec = (competitor[mask]-y[mask])[valid]
        sums[i] = [np.abs(ec).sum(),np.abs(eb).sum(),(ec**2).sum(),(eb**2).sum(),valid.sum()]
    sample = sums[rng_indices].sum(axis=1)
    mae_delta = (sample[:,0]-sample[:,1])/sample[:,4]
    rmse_delta = np.sqrt(sample[:,2]/sample[:,4])-np.sqrt(sample[:,3]/sample[:,4])
    original = sums.sum(axis=0)
    return {'n':int(original[4]),'mae_delta':float((original[0]-original[1])/original[4]),
            'mae_delta_ci_low':float(np.quantile(mae_delta,.025)),
            'mae_delta_ci_high':float(np.quantile(mae_delta,.975)),
            'rmse_delta':float(np.sqrt(original[2]/original[4])-np.sqrt(original[3]/original[4])),
            'rmse_delta_ci_low':float(np.quantile(rmse_delta,.025)),
            'rmse_delta_ci_high':float(np.quantile(rmse_delta,.975))}


def main():
    began,cpu = time.perf_counter(),time.process_time()
    experiment=Experiment()
    experiment.causal_checks()
    validation = (experiment.di>=180)&(experiment.di<275)
    test = experiment.di>=275
    specs = [{'family':'nyx'}]
    specs += [{'family':family,'halflife':hl} for family in ['ewma_country','ewma_country_hour'] for hl in [7,28]]
    specs += [{'family':family,'alpha':alpha} for family in ['ridge_univariate','ridge_multivariate'] for alpha in [10,100]]
    specs += [{'family':'pca_ridge','rank':rank,'alpha':alpha} for rank in [1,2] for alpha in [10,100]]
    validation_rows=[]
    for spec in specs:
        correction = experiment.model_prediction(spec,180,275,'validation')
        pred = experiment.nyx-correction
        metric = scores(experiment.frozen[validation],pred[validation])
        validation_rows.append({'spec':spec,**metric})
    selected={}
    for row in validation_rows:
        family=row['spec']['family']
        if family not in selected or row['mae']<selected[family]['mae']:
            selected[family]=row
    global_choice=min(selected,key=lambda f:selected[f]['mae'])
    selection={'selection_label':'frozen_actual','selection_objective':'pooled paired MAE over all four zones, validation only',
               'validation_days':[experiment.days[180],experiment.days[274]],
               'locked_final_test_days':[experiment.days[275],experiment.days[364]],
               'all_candidates':validation_rows,'selected_per_family':selected,
               'primary_family_selected_before_test':global_choice}
    # Persist the exact choice before evaluating any final-test predictions.
    dump('validation_selection_locked.json',selection)
    selection_sha=hashlib.sha256((OUT/'validation_selection_locked.json').read_bytes()).hexdigest()
    print('Locked validation choice: '+global_choice,flush=True)
    predictions={}
    for family,row in selected.items():
        correction=experiment.model_prediction(row['spec'],275,365,'test')
        predictions[family]=experiment.nyx-correction
    predictions['storm']=experiment.storm
    metric_rows=[]
    grouped_rows=[]
    sensitivity=[]
    rng=np.random.default_rng(SEED)
    starts=rng.integers(0,90-BLOCK_DAYS+1,size=(BOOTSTRAPS,int(np.ceil(90/BLOCK_DAYS))))
    samples=(starts[:,:,None]+np.arange(BLOCK_DAYS)[None,None,:]).reshape(BOOTSTRAPS,-1)[:,:90]
    bootstrap=[]
    for family,pred in predictions.items():
        for label,y in [('latest',experiment.actual),('frozen',experiment.frozen)]:
            for zone_index in [None,0,1,2,3]:
                zone='pooled' if zone_index is None else ZONES[zone_index]
                yp=y if zone_index is None else y[:,zone_index]
                pp=pred if zone_index is None else pred[:,zone_index]
                baseline=experiment.nyx if zone_index is None else experiment.nyx[:,zone_index]
                metric_rows.append({'model':family,'label':label,'zone':zone,**scores(yp[test],pp[test])})
                bootstrap.append({'model':family,'label':label,'zone':zone,
                                  **paired_bootstrap(experiment.di[test],yp[test],baseline[test],pp[test],samples)})
        for z,zone in enumerate(ZONES):
            y=experiment.actual[:,z]
            thresholds=experiment.manifest['thresholds_initial_train_frozen_actual'][zone]
            for grouping,masks in [('regime',{'negative':y<0,'upper95_train':y>thresholds['q95'],'upper99_train':y>thresholds['q99']}),
                                   ('hour',{str(h):experiment.hour==h for h in range(24)})]:
                for label,mask in masks.items():
                    group=test&mask
                    if not group.any(): continue
                    grouped_rows.append({'model':family,'zone':zone,'grouping':grouping,'group':label,
                                         **scores(y[group],pred[group,z])})
    final_rows=[]
    for z,zone in enumerate(ZONES):
        result=pd.DataFrame({'timestamp':experiment.timestamps[test],'zone':zone,
                             'day':np.asarray(experiment.days)[experiment.di[test]],
                             'hour':experiment.hour[test],'observed':experiment.actual[test,z],
                             'frozen_actual':experiment.frozen[test,z]})
        for name,pred in predictions.items(): result[name]=pred[test,z]
        final_rows.append(result)
    pd.concat(final_rows,ignore_index=True).to_csv(OUT/'locked_test_predictions.csv.gz',index=False,compression='gzip')
    pd.DataFrame(metric_rows).to_csv(OUT/'locked_test_metrics.csv',index=False)
    pd.DataFrame(grouped_rows).to_csv(OUT/'locked_test_group_metrics.csv',index=False)
    pd.DataFrame(bootstrap).to_csv(OUT/'locked_test_paired_bootstrap.csv',index=False)
    dump('rolling_fit_audit.json',experiment.fit_audit)
    summary={'protocol':{'initial_train_days':180,'validation_days':95,'locked_test_days':90,
                         'last_label_lag_days':2,'rolling_fit_max_days':180,'first_validation_fit_days':165,'refit_every_days':7,
                         'issue_frequency':'fresh day-ahead features daily, coefficients refit weekly; not a seven-day forecast issued at refit',
                         'feature_warmup_days':14,'ridge_ewma_halflife_days':14,
                         'training_and_validation_labels':'frozen_actual, retrospective source vintage',
                         'test_labels':'both audited latest and frozen_actual',
                         'normalization_imputation_pca_fit':'eligible training rows only at every refit',
                         'late_test_adaptation':'earlier test labels become training data under the fixed D-2 protocol; no hyperparameter reselection',
                         'features':'D-2 same civil-hour four-country residuals, causal country and country-hour EWMA residuals',
                         'dst':'both physical target hours retained; repeated civil-hour lag residuals averaged; missing lag feature imputed using training data only',
                         'corrections':'point median only, no corrected quantiles or uncertainty claims',
                         'bootstrap_repetitions':BOOTSTRAPS,'bootstrap_block_days':BLOCK_DAYS,'bootstrap_seed':SEED,
                         'delta_sign':'competitor loss minus NYX loss; negative favors competitor',
                         'bootstrap_limits':'conditional on these 90 summer-dominated days and selected hyperparameters; neither vintage uncertainty nor selection uncertainty included'},
             'primary_family_selected_before_test':global_choice,
             'validation_selection_sha256':selection_sha,
             'input_matrix_sha256':hashlib.sha256((OUT/'verified_hourly_pairs.csv.gz').read_bytes()).hexdigest(),
             'fit_count':len(experiment.fit_audit),
             'fit_cutoffs_all_valid':all(x['fit_last_day']<=x['max_permitted_label_day'] for x in experiment.fit_audit),
             'future_label_influence_check_passed':True,
             'runtime_seconds':time.perf_counter()-began,'cpu_seconds':time.process_time()-cpu,
             'rss_mb':psutil.Process().memory_info().rss/1024**2,
             'limitations':['No prospective gain claimed: retrospective frozen prequential forecasts are not actually issued vintages.',
                            'Actual and frozen labels can be revised after historical origins; D-2 alone cannot prove availability.',
                            'Final 90-day holdout covers June-September rather than the entire seasonal cycle.',
                            'Secondary family/country/regime contrasts are descriptive; no multiplicity adjustment.',
                            'Storm official benchmark availability at strict 08:00 cutoff has not been established.',
                            'No Tensor-TimesFM forecasts or training were run; no complementarity claim against that model.']}
    dump('experiment_summary.json',summary)
    print(json.dumps({'primary_family':global_choice,'runtime_seconds':summary['runtime_seconds'],
                      'pooled_latest_test':[r for r in metric_rows if r['label']=='latest' and r['zone']=='pooled'],
                      'pooled_latest_bootstrap':[r for r in bootstrap if r['label']=='latest' and r['zone']=='pooled']}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
