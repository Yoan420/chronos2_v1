"""No fitting: inspect fallback provenance and rescore saved comparator outputs."""
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

import json
import pandas as pd
import numpy as np
SRC=ROOT/'research/tensor_timesfm_20260915/metrics'
f=pd.read_csv(SRC/'verified_hourly_pairs.csv.gz',parse_dates=['timestamp'])
manifest=json.loads((SRC/'extraction_manifest.json').read_text(encoding='utf-8'))
records=[]
f['storm_fallback']=False; f['label_fallback']=False
for p in manifest['provenance']:
    z=p['zone']; a=json.loads(archived_path(p['source_audit_path']).read_text(encoding='utf-8'))
    s=a['storm_dashboard']['source']; o=a['observed']['source'].get('post_auction_fallback',{})
    days=s['fallback_used_local_days']
    expected=pd.date_range(a['storm_dashboard']['period']['start_utc'],a['storm_dashboard']['period']['end_utc'],freq='h')
    affected=expected.tz_convert(a['timezone']).strftime('%Y-%m-%d').isin(days)
    # Source code records every distinct fallback day, without truncation.
    # If aggregate fallback count saturates their full physical-hour capacity,
    # every physical hour on those dates must be a fallback hour.
    assert int(affected.sum())==s['fallback_used_hours']
    f.loc[f.zone==z,'storm_fallback']=f.loc[f.zone==z,'day'].isin(days)
    label_times=pd.to_datetime(o.get('applied_value_times_utc',[]),utc=True)
    f.loc[f.zone==z,'label_fallback']=f.loc[f.zone==z,'timestamp'].isin(label_times)
    g=f[f.zone==z]; test=g[g.split=='test']
    records.append({'zone':z,'snapshot_hours':len(expected),'storm_snapshot_fallback_hours':s['fallback_used_hours'],
       'storm_fallback_days':len(days),'fallback_count_equals_full_day_capacity':True,
       'annual_storm_fallback_hours':int(g.storm_fallback.sum()),'test90_storm_fallback_hours':int(test.storm_fallback.sum()),
       'annual_storm_fallback_fraction_of_paired':int(g.storm_fallback.sum())/int(g.storm.notna().sum()),
       'test90_storm_fallback_fraction':int(test.storm_fallback.sum())/len(test),
       'annual_label_fallback_hours':int(g.label_fallback.sum()),'test90_label_fallback_hours':int(test.label_fallback.sum()),
       'label_fallback_days':sorted(g.loc[g.label_fallback,'day'].unique()),
       'storm_fallback_live_recomputation':s['fallback_live_recomputation'],
       'label_overlap_validation_n':o.get('validation_paired_hours'),'label_overlap_max_abs':o.get('validation_maximum_absolute_difference_eur_mwh'),
       'storm_fallback_dates':days})
pd.DataFrame(records).drop(columns=['storm_fallback_dates','label_fallback_days']).to_csv(OUT/'source_fallback_counts.csv',index=False)
(OUT/'source_fallback_detail.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
rows=[]
for period,g in [('annual',f),('test90',f[f.split=='test'])]:
    g=g.dropna(subset=['storm','nyx'])
    for subset,mask in [('all_common',np.ones(len(g),dtype=bool)),('cache_only',~g.storm_fallback),('native_fallback_only',g.storm_fallback),('cache_only_no_label_fallback',~g.storm_fallback&~g.label_fallback)]:
        h=g.loc[mask]
        for z in ['pooled','BE','DE','FR','NL']:
            q=h if z=='pooled' else h[h.zone==z]
            for model in ['nyx','storm']:
                e=q[model]-q.observed
                rows.append({'period':period,'subset':subset,'zone':z,'model':model,'n':len(q),'days':q.day.nunique(),
                    'mae':abs(e).mean(),'rmse':np.sqrt((e*e).mean()),'bias':e.mean()})
pd.DataFrame(rows).to_csv(OUT/'storm_provenance_sensitivity.csv',index=False)
intervals=[]
for z in ['pooled','BE','DE','FR','NL']:
    q=f if z=='pooled' else f[f.zone==z]
    lo,hi,y,m=q.nyx_p10,q.nyx_p90,q.observed,q.nyx
    si=hi-lo+10*np.maximum(lo-y,0)+10*np.maximum(y-hi,0)
    intervals.append({'zone':z,'n':len(q),'coverage80':((y>=lo)&(y<=hi)).mean(),'below_p10':(y<lo).mean(),
      'above_p90':(y>hi).mean(),'mean_width':(hi-lo).mean(),'interval_score80':si.mean(),'wis_one80':((.5*abs(y-m)+.1*si)/1.5).mean()})
pd.DataFrame(intervals).to_csv(OUT/'verified_interval_metrics.csv',index=False)
print('PROVENANCE',json.dumps(records,indent=2))
print('SENSITIVITY',pd.DataFrame(rows).query("zone=='pooled'").to_string(index=False))
print('INTERVAL',pd.DataFrame(intervals).to_string(index=False))
