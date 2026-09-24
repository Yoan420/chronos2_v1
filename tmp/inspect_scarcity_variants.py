"""Read-only final experiment checks, not a model entry point."""
from pathlib import Path
import json
import sys
import pandas as pd
from nyx_scarcity.variant_reporting import _native_classification

root = Path(__file__).resolve().parents[1]
d = root / 'runs/experiments/nyx_scarcity_v1/variants/snapshots/20260914T133114Z_8f239442'
if len(sys.argv) > 1:
    d = Path(sys.argv[1])
fields = ['hours', 'positives', 'threshold_min', 'threshold_max', 'true_positive',
          'false_positive', 'precision', 'recall', 'brier', 'average_precision']
out = {}
for name in ['hgb_v1', 'xgb_unweighted_fixed', 'xgb_weighted_fixed', 'xgb_unweighted_dwt', 'xgb_weighted_dwt']:
    path = d / ('control_predictions.parquet' if name == 'hgb_v1' else name + '/predictions.parquet')
    if name != 'hgb_v1' and not (path.parent / 'results_manifest.json').exists():
        continue
    p = pd.read_parquet(path)
    e = p[p['sample'].eq('evaluation') & p[['actual', 'forecast', 'benchmark_forecast']].notna().all(axis=1)]
    s = _native_classification(e)
    item = {k: s[k] for k in fields}
    item.update(mae=(e.candidate_forecast-e.actual).abs().mean(),
                baseline_mae=(e.forecast-e.actual).abs().mean(),
                changed=int(e.applied_correction.ne(0).sum()))
    item['proposal_mae'] = {str(a): float((e.forecast+a*e.bounded_correction-e.actual).abs().mean())
                            for a in [.25, .5, 1.]}
    item['sept14_19h'] = p.loc[p.timestamp_utc.eq(pd.Timestamp('2026-09-14T17:00:00Z')),
        ['zone', 'actual', 'forecast', 'candidate_forecast', 'spike_probability', 'selected_weight']].to_dict('records')
    if name != 'hgb_v1':
        shap = json.loads((path.parent/'shap_summary.json').read_text())
        item['shap_status'] = shap['status']
        item['shap_reconstruction_verified'] = shap['audit']['reconstruction_verified']
    out[name] = item
print(json.dumps(out, indent=2))
