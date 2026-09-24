"""Audit target identity only: no model, correction, training or forecast score."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from nyx_intrahour.data import load_baseline
from nyx_quarterhour.sources import digest, hourly_means, read_native_prices, safe_output


def main():
    directory=ROOT/'data/pit/nyx_quarterhour/primary_prices_20251001_20260915'
    native, source=read_native_prices(directory/'manifest.json')
    baseline, baseline_audit=load_baseline(ROOT,'2026-09-16')
    hourly=hourly_means(native)
    pairs=hourly.merge(baseline[['timestamp_utc','zone','actual','training_actual']],on=['timestamp_utc','zone'],how='left',validate='one_to_one')
    if len(pairs)!=len(hourly) or pairs[['actual','training_actual']].isna().any().any():
        raise ValueError('All native hourly means must match a verified NYX baseline label.')
    pairs=pairs.rename(columns={'actual':'nyx_report_only_actual','training_actual':'nyx_frozen_actual'})
    results=[]
    for zone,g in pairs.groupby('zone',sort=True):
        item={'zone':zone,'paired_hours':len(g)}
        for field in ['nyx_report_only_actual','nyx_frozen_actual']:
            delta=g.actual_hourly_from_15m-g[field]
            item[field]={'mae_eur_mwh':float(np.mean(np.abs(delta))),
                         'rmse_eur_mwh':float(np.sqrt(np.mean(delta**2))),
                         'max_abs_difference_eur_mwh':float(np.max(np.abs(delta))),
                         'hours_over_1e_9':int((np.abs(delta)>1e-9).sum()),
                         'hours_over_0_01':int((np.abs(delta)>0.01).sum())}
        results.append(item)
    out=safe_output(directory/'hourly_label_pairs.parquet')
    if out.exists():
        raise ValueError('This comparison is already sealed.')
    pairs.to_parquet(out,index=False)
    audit={'purpose':'Label identity comparison only, never a forecast score or gain.',
           'native_manifest_sha256':source['manifest_sha256'], 'native_data_sha256':source['data_sha256'],
           'baseline_delivery_day':'2026-09-16','baseline_audit':baseline_audit,
           'hourly_rule':'Arithmetic mean of exactly four finite, unique, physical UTC quarters.',
           'rows':len(pairs),'results':results,'pairs_file':out.name,'pairs_sha256':digest(out),
           'decision':'Score both future candidate and NYX predictions against these same native hourly means; keep report-only label differences visible.',
           'price_vintage':'latest_observations','production_pit_evidence':False}
    safe_output(directory/'hourly_label_comparison.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print(json.dumps({'rows':len(pairs),'results':results},indent=2))

if __name__=='__main__':
    main()
