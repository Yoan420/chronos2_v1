from pathlib import Path
import json
import pandas as pd

root = Path(__file__).resolve().parents[1]
latest = json.loads((root/'runs/experiments/nyx_scarcity_v1/variants/adjustments/latest.json').read_text())
directory = Path(latest['snapshot'])
summary = json.loads((directory/'metrics.json').read_text())
rows = pd.read_parquet(directory/'adjustments.parquet')
result = {'snapshot':str(directory), 'annual':{}, 'by_zone':{}, 'sept14_19h':[]}
for variant, scores in summary['overall']['annual']['variants'].items():
    result['annual'][variant] = {key:value['mae_eur_mwh'] for key,value in scores.items()}
for zone, group in summary['by_zone'].items():
    result['by_zone'][zone] = {
        'hours':group['paired_hours'],
        'nyx_mae':group['annual']['nyx']['mae_eur_mwh'],
        'storm_mae':group['annual']['storm']['mae_eur_mwh'],
        'xgb_unweighted_fixed_mae':{key:value['mae_eur_mwh'] for key,value in group['annual']['variants']['xgb_unweighted_fixed'].items()},
        'tail1_mae': {'nyx':group['tails']['top_1_percent']['scores']['nyx']['mae_eur_mwh'],
                      'storm':group['tails']['top_1_percent']['scores']['storm']['mae_eur_mwh'],
                      **{key:value['mae_eur_mwh'] for key,value in group['tails']['top_1_percent']['scores']['variants']['xgb_unweighted_fixed'].items()}},
        'interventions':group['interventions']['xgb_unweighted_fixed'],
    }
case = rows[rows.variant_id.eq('xgb_unweighted_fixed') & rows.timestamp_utc.eq(pd.Timestamp('2026-09-14T17:00:00Z'))]
result['sept14_19h'] = case[['zone','forecast','spike_probability','bounded_correction','proposal25','proposal50','proposal100','candidate_forecast','actual','benchmark_forecast']].to_dict('records')
print(json.dumps(result,indent=2))
