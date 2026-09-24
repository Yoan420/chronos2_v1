"""Read-only event forensic calculations; stdout only, no fitting or writes."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from nyx_stress_guard.features import make_stress_features

root = Path(__file__).resolve().parents[1]
source = root / 'runs/experiments/nyx_scarcity_v1/coherent_p50/snapshots/20260914T160104Z_d7366f80'
panel = pd.read_parquet(source/'panel.parquet')
d, _, _, _ = make_stress_features(panel)
d['local'] = d.timestamp_utc.dt.tz_convert('Europe/Paris')
d['day'] = d.local.dt.strftime('%Y-%m-%d')
d['hour'] = d.local.dt.hour
features = {
    'rl_gw': 'feature_fundamental_local_residual_load_gw',
    'supply_gw': 'feature_fundamental_local_selected_supply_proxy_gw',
    'margin_proxy_gw': 'feature_fundamental_local_selected_supply_minus_residual_proxy_gw',
    'pressure_proxy': 'feature_fundamental_local_pressure',
    'peer_pressure': 'feature_fundamental_peer_pressure',
    'gas_gw': 'feature_fundamental_local_gas_available_gw',
    'solar_gw': 'feature_fundamental_local_solar_generation_gw',
    'wind_gw': 'feature_fundamental_local_wind_generation_gw',
    'rl_ramp1_gw_h': 'feature_fundamental_local_residual_ramp_1h_gw_per_hour',
    'rl_ramp3_gw_h': 'feature_fundamental_local_residual_ramp_3h_gw_per_hour',
    'solar_ramp3_gw_h': 'feature_fundamental_local_solar_ramp_3h_gw_per_hour',
    'wind_ramp3_gw_h': 'feature_fundamental_local_wind_ramp_3h_gw_per_hour',
    'regional_rl_ramp3_gw_h': 'feature_fundamental_stress_regional_residual_ramp_3h_gw_per_hour',
    'temperature_c': 'feature_fundamental_local_temperature_c',
    'cgc_eur_mwh': 'feature_fundamental_clean_gas_cost_ccgt_proxy_eur_mwh',
    'ocgt_eur_mwh': 'feature_fundamental_clean_gas_cost_ocgt_proxy_eur_mwh',
    'residual_day_rank': 'feature_fundamental_stress_local_residual_day_rank',
    'rising_zones3': 'feature_fundamental_stress_residual_rising_count_3h',
}

def dump(name, obj):
    if len(sys.argv)>1 and sys.argv[1]=='ranks' and name != 'PAST_SAMEHOUR_RANKS':
        return
    print(name)
    for record in obj:
        print(json.dumps(record, default=str, allow_nan=True))

selected = d[d.day.isin(['2026-09-14','2026-06-24','2026-06-25','2026-06-26']) & d.hour.eq(19)]
cols = ['zone','day','hour','actual','forecast','benchmark_forecast',*features.values()]
dump('EVENT_19H', selected[cols].rename(columns={v:k for k,v in features.items()}).round(6).to_dict('records'))
profile=d[d.day.eq('2026-09-14') & d.hour.between(15,22)]
dump('SEPT14_PROFILE',profile[['zone','hour','actual','forecast','benchmark_forecast',*list(features.values())[:14]]].rename(columns={v:k for k,v in features.items()}).round(4).to_dict('records'))

ranks=[]
for _, row in selected.iterrows():
    historic = d[(d.zone==row.zone) & (d.timestamp_utc < row.forecast_origin_utc) & (d.forecast_origin_utc < row.forecast_origin_utc) & (d.hour==row.hour)]
    records={}
    for label,col in features.items():
        values=historic[col].dropna().to_numpy()
        val=row[col]
        if len(values) and pd.notna(val):
            records[label]={'value':round(float(val),6),'n':len(values),'mid_rank_pct':round(float(100*((values<val).sum()+.5*(values==val).sum())/len(values)),3),'past_median':round(float(np.median(values)),6),'past_p95':round(float(np.quantile(values,.95)),6)}
    ranks.append({'zone':row.zone,'day':row.day,'ranks':records})
dump('PAST_SAMEHOUR_RANKS',ranks)

preds=[]
paths={'fundamental':source/'source_predictions.parquet','forest':source/'forest/predictions.parquet','empirical':source/'empirical/predictions.parquet','stress':root/'runs/experiments/nyx_scarcity_v1/stress_guard/snapshots/20260914T171142Z_86e6f8f9/physics_direct.parquet'}
for label,path in paths.items():
    if not path.is_file():
        print('MISSING',path)
        continue
    p=pd.read_parquet(path)
    local=p.timestamp_utc.dt.tz_convert('Europe/Paris')
    p=p[local.dt.strftime('%Y-%m-%d').isin(['2026-09-14','2026-06-24','2026-06-25','2026-06-26']) & local.dt.hour.eq(19)]
    wanted=['zone','timestamp_utc','actual','forecast','candidate_forecast','spike_probability','probability_gate','threshold_eur_mwh','expert_ready','expert_fit_day','raw_correction','applied_correction','physical_gate_passed','risk_probability_gate','predicted_signed_residual_median','gate_reason','proposal_reason']
    preds.append({'name':label,'records':p[[c for c in wanted if c in p]].to_dict('records')})
dump('PREDICTIONS',preds)
