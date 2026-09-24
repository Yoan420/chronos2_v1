"""Read-only diagnostic of a sealed KPI snapshot; writes only own tmp result."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
snap = Path(json.loads((root/'runs/reports/kpi/latest.json').read_text())['snapshot'])
data = json.loads((snap/'kpi_metrics.json').read_text(encoding='utf-8'))
catalog = {m['id']: m for m in data['catalog']}
out = {'snapshot': str(snap), 'periods': {}}
cols = ['model_id','mae_eur_mwh','rmse_eur_mwh','win_rate_hour_pct','win_rate_day_mae_pct','win_rate_day_mean_price_pct','mae_day_mean_price_eur_mwh','mean_price_eur_mwh','observed_mean_price_eur_mwh','n_hours','n_days']
for period, value in data['periods'].items():
    rows = pd.DataFrame(value['rows'])
    period_out = {'coverage': value['coverage'], 'zones':{}}
    for zone, frame in rows.groupby('zone'):
        b = frame.set_index('model_id').loc['nuclear_kalman']
        s = frame.set_index('model_id').loc['__storm__']
        f = frame[cols].copy()
        f['gain_mae_vs_nyx'] = b.mae_eur_mwh - f.mae_eur_mwh
        f['gain_rmse_vs_nyx'] = b.rmse_eur_mwh - f.rmse_eur_mwh
        f['gain_mae_vs_storm'] = s.mae_eur_mwh - f.mae_eur_mwh
        f['bias_eur_mwh'] = f.mean_price_eur_mwh - f.observed_mean_price_eur_mwh
        period_out['zones'][zone] = json.loads(f.sort_values('mae_eur_mwh').to_json(orient='records',double_precision=15))
    out['periods'][period] = period_out

daily = pd.DataFrame(data['periods']['365']['daily_rows'])
# Calendar blocks synchronized across countries, retain missing days as zero weights.
calendar = pd.date_range(data['periods']['365']['period']['start_day'], data['periods']['365']['period']['end_day'], freq='D').strftime('%Y-%m-%d')
baseline = daily[daily.model_id == 'nuclear_kalman'].set_index(['delivery_day','zone'])
out['block_bootstrap'] = {'replications':2000, 'block_length_calendar_days':7, 'seed':20260915, 'results':{}}
rng=np.random.default_rng(20260915)
n=len(calendar)
starts=rng.integers(0,n,size=(2000,int(np.ceil(n/7))))
indices=((starts[:,:,None]+np.arange(7))%n).reshape(2000,-1)[:,:n]
for model_id in [*catalog, '__storm__']:
    m=daily[daily.model_id==model_id].set_index(['delivery_day','zone'])
    joined=baseline[['mae_eur_mwh','n_hours']].join(m[['mae_eur_mwh']],rsuffix='_model')
    joined['gain_abs_sum']=(joined.mae_eur_mwh-joined.mae_eur_mwh_model)*joined.n_hours
    date=joined.groupby('delivery_day')[['gain_abs_sum','n_hours']].sum().reindex(calendar,fill_value=0)
    gains=date.gain_abs_sum.to_numpy()
    weights=date.n_hours.to_numpy()
    sims=gains[indices].sum(axis=1)/weights[indices].sum(axis=1)
    total_gain=float(gains.sum())
    changed=joined[np.abs(joined.gain_abs_sum)>1e-8]
    best=date.sort_values('gain_abs_sum',ascending=False).head(10)
    worst=date.sort_values('gain_abs_sum').head(10)
    out['block_bootstrap']['results'][model_id]={
        'gain_mae_on_complete_days':float(total_gain/weights.sum()),
        'ci95':np.quantile(sims,[.025,.975]).tolist(),
        'bootstrap_positive_fraction':float((sims>0).mean()),
        'changed_country_days':len(changed),
        'improved_country_days':int((joined.gain_abs_sum>1e-8).sum()),
        'worsened_country_days':int((joined.gain_abs_sum<-1e-8).sum()),
        'changed_calendar_days':int((date.gain_abs_sum.abs()>1e-8).sum()),
        'absolute_error_gain_sum':total_gain,
        'top10_positive':best.reset_index(names='delivery_day').to_dict('records'),
        'top10_negative':worst.reset_index(names='delivery_day').to_dict('records'),
        'share_gain_top1':None if total_gain==0 else float(best.gain_abs_sum.iloc[0]/total_gain),
        'share_gain_top5':None if total_gain==0 else float(best.gain_abs_sum.iloc[:5].sum()/total_gain),
    }
out['monthly']=[]
daily['month']=daily.delivery_day.str[:7]
for (month,model_id),f in daily.groupby(['month','model_id']):
    out['monthly'].append({'month':month,'model_id':model_id,'mae':float(np.average(f.mae_eur_mwh,weights=f.n_hours)), 'n_hours':int(f.n_hours.sum())})
target=root/'tmp/kpi_analysis_stats.json'
target.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
print(target)
for period in out['periods']:
    print('\nPERIOD',period)
    for zone in ['ALL','FR','DE','BE','NL']:
        f=pd.DataFrame(out['periods'][period]['zones'][zone])
        print(zone, f[['model_id','mae_eur_mwh','rmse_eur_mwh','gain_mae_vs_nyx','win_rate_hour_pct','win_rate_day_mae_pct','win_rate_day_mean_price_pct','mae_day_mean_price_eur_mwh','bias_eur_mwh']].to_string(index=False,float_format=lambda x:f'{x:.6f}'))
print('\nBOOTSTRAP')
for k,v in out['block_bootstrap']['results'].items():
    print(k,{x:y for x,y in v.items() if not x.startswith('top10')})
