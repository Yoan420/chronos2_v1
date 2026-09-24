"""Read-only one-fold integration smoke, without publishing a forecast."""
from pathlib import Path
import json
import sys
import time
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nyx_fundamental_stress import runner as source_runner
from nyx_coherent_p50 import policy
from nyx_scarcity import policy as base
from nyx_fundamental_stress.features import make_fundamental_features

directory = ROOT/'runs/experiments/nyx_scarcity_v1/fundamental/snapshots/20260914T152452Z_41529943'
_, _, manifest = source_runner.read_suite(directory, root=ROOT)
source_runner.verify_result(directory, 'fundamental', manifest)
panel = pd.read_parquet(directory/'panel.parquet')
source = pd.read_parquet(directory/'fundamental/predictions.parquet')
folds = pd.read_parquet(directory/'fundamental/folds.parquet').set_index('fit_day', drop=False)
augmented, features, required, audit = make_fundamental_features(panel)
p = base._parameters({**manifest['settings'], 'feature_columns': features, 'required_feature_columns': required})
prepared = base._prepare(augmented, p)
day = '2026-09-14'
cutoff = pd.Timestamp('2026-09-13T08:00:00+02:00').tz_convert('UTC')
started = time.monotonic()
state, record = policy._fit(prepared, day, cutoff, p, ('BE','DE','FR','NL'),
    source_folds=folds, kind=sys.argv[1] if len(sys.argv)>1 else 'empirical')
print(json.dumps({'fit_seconds':time.monotonic()-started,'core_rows':record['cdf_core_rows'],
                  'normal_rows':record['cdf_normal_rows'],'spike_rows':record['cdf_spike_rows']},indent=2))
current = prepared.loc[prepared._day.eq(day)]
lookup = source.set_index(policy.KEYS, drop=False)
probability, raw, threshold, detail = policy.predict_state(state, current, p, policy._source_rows(lookup,current))
detail['probability'], detail['raw_correction'] = probability, raw
take = detail.timestamp_utc.dt.tz_convert('Europe/Paris').dt.hour.eq(19)
print(detail.loc[take,['zone','probability','mixture_error_q10','mixture_error_q50','mixture_error_q90','raw_correction']].to_string(index=False))
print(json.dumps({'total_seconds':time.monotonic()-started},indent=2))
