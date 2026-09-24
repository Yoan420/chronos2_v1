"""Read-only post-hoc diagnostic; no fitting, policy changes or source writes."""
from pathlib import Path
import sys
import json
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kpi_report.data import load_recent_models, verify_sources
from kpi_report.economic import _lagged_references, _policy
from kpi_report.metrics import _references

pointer = json.loads((ROOT/'runs/reports/kpi/latest.json').read_text())
report = json.loads((Path(pointer['snapshot'])/'kpi_metrics.json').read_text(encoding='utf-8'))
frame, catalog, audit = load_recent_models(ROOT)
models = [m['id'] for m in catalog]
refs = _references(frame)
wide = frame.pivot(index=['zone','timestamp_utc'], columns='model_id', values='forecast').join(refs)
price = wide.dropna(subset=models+['actual','storm']).copy()
lag = _lagged_references(refs, price.reset_index()[['zone','timestamp_utc']], 'Europe/Paris')
economic = price.join(lag.set_index(['zone','timestamp_utc'])[['reference_price']]).dropna(subset=['reference_price'])
hours = economic.reset_index().groupby('timestamp_utc').zone.nunique()
economic = economic.loc[economic.index.get_level_values('timestamp_utc').isin(hours.index[hours.eq(4)])].copy()
settings, _ = _policy(None, ROOT/'config/economic_value.yaml')
cost = settings['strategy']['transaction_cost_eur_mwh']+settings['strategy']['slippage_eur_mwh']
hurdle = settings['strategy']['signal_threshold_eur_mwh']+cost
capacity = settings['portfolio']['capacity_mw']/len(settings['zones'])
local = economic.index.get_level_values('timestamp_utc').tz_convert('Europe/Paris')
economic['day'] = local.strftime('%Y-%m-%d')
economic['hour'] = local.hour
economic['month'] = local.strftime('%Y-%m')
move = economic.actual-economic.reference_price
predictions = models+['storm']
pnl, positions = {}, {}
for model in predictions:
    edge = economic[model]-economic.reference_price
    position = np.where(edge.abs().gt(hurdle), np.sign(edge)*capacity, 0.)
    positions[model] = pd.Series(position,index=economic.index)
    pnl[model] = positions[model]*move-positions[model].abs()*cost
    source_id = '__storm__' if model=='storm' else model
    saved = next(r for r in report['periods']['365']['economic']['rows'] if r['zone']=='ALL' and r['model_id']==source_id)
    assert len(economic)==saved['n_country_hours']
    assert abs(float(pnl[model].sum())-saved['pnl_net_eur'])<1e-6, (model,pnl[model].sum(),saved)

date_index=pd.date_range(report['periods']['365']['period']['start_day'],report['periods']['365']['period']['end_day']).strftime('%Y-%m-%d')
rng=np.random.default_rng(20260915)
boots=((rng.integers(0,len(date_index),size=(4000,math.ceil(len(date_index)/7),1))+np.arange(7))%len(date_index)).reshape(4000,-1)[:,:len(date_index)]

def summary(model, mask):
    pos=positions[model].loc[mask]; realised=move.loc[mask]
    own=pnl[model].loc[mask]; benchmark=pnl['storm'].loc[mask]
    return {'n':int(mask.sum()),'pnl_net':float(own.sum()),'gain_vs_storm':float((own-benchmark).sum()),
            'active':int(pos.ne(0).sum()), 'disagreements_vs_storm':int(pos.ne(positions['storm'].loc[mask]).sum()),
            'direction_hit_active':float((np.sign(pos[pos.ne(0)])==np.sign(realised[pos.ne(0)])).mean()) if pos.ne(0).any() else None}

out={'source':pointer['snapshot'],'economic_common_country_hours':len(economic),'rows':[], 'diagnostic_only':True}
for model in predictions:
    delta=pnl[model]-pnl['storm']; improvement=pnl[model]-pnl['nuclear_kalman']
    daily=delta.groupby(economic.day).sum().reindex(date_index,fill_value=0.)
    daily_improvement=improvement.groupby(economic.day).sum().reindex(date_index,fill_value=0.)
    ci=np.quantile(daily.to_numpy()[boots].sum(axis=1),[.025,.975])
    ci_up=np.quantile(daily_improvement.to_numpy()[boots].sum(axis=1),[.025,.975])
    masks={'all':np.ones(len(economic),dtype=bool),'observed_ge200':economic.actual.ge(200),
           'observed_le0':economic.actual.le(0),'normal_0_200':economic.actual.gt(0)&economic.actual.lt(200),
           'reference_move_abs_ge100':move.abs().ge(100),'positive_move_ge100':move.ge(100),'negative_move_le_minus100':move.le(-100)}
    by_zone={z:summary(model,economic.index.get_level_values('zone')==z) for z in audit['zones']}
    dpos=positions[model]-positions['nuclear_kalman']
    changed_forecast=~np.isclose(economic[model],economic['nuclear_kalman'],rtol=0,atol=1e-9)
    out['rows'].append({'model':model,'total':summary(model,masks['all']),
      'subgroups':{k:summary(model,v) for k,v in masks.items() if k!='all'},'by_zone':by_zone,
      'changed_forecast_vs_nyx':int(changed_forecast.sum()),'changed_positions_vs_nyx':int(dpos.ne(0).sum()),
      'improvement_vs_nyx_eur':float(improvement.sum()),'exploratory_block7_ci_vs_storm':ci.tolist(),
      'exploratory_block7_ci_vs_nyx':ci_up.tolist(),
      'worst_days_vs_storm':daily.sort_values().head(8).to_dict(),'best_days_vs_storm':daily.sort_values().tail(8).to_dict(),
      'gain_vs_storm_by_hour':delta.groupby(economic.hour).sum().to_dict(),
      'gain_vs_storm_by_month':delta.groupby(economic.month).sum().to_dict(),
      'increment_vs_nyx_by_day':daily_improvement[daily_improvement.abs().gt(1e-9)].to_dict()})
verify_sources(audit)
output=ROOT/'tmp/kpi_economic_analysis_20260915.json'
with output.open('x',encoding='utf-8') as stream: json.dump(out,stream,indent=2,allow_nan=False)
print(json.dumps({'output':str(output),'n':len(economic),'models':[{'id':r['model'],'gain_vs_storm':r['total']['gain_vs_storm'],'gain_vs_nyx':r['improvement_vs_nyx_eur'],'changed_positions':r['changed_positions_vs_nyx'],'ci_vs_nyx':r['exploratory_block7_ci_vs_nyx']} for r in out['rows']]},indent=2))
