"""Real archived data, one causal fit; NOT the annual performance evaluation."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import yaml
from nyx_clean_fuel import runner as r
from nyx_clean_fuel.features import build_features
from nyx_clean_fuel.sources import _validate_frame, _contract, _json_bytes, _sha, _columns
from chronos2_modular.common import build_zone_configs
from chronos2_hourly.nuclear_preparation import prepare_nuclear_zone_data
from chronos2_hourly.nuclear_run_archive import load_nuclear_result_bundle
from chronos2_hourly.nuclear_forecast import _digest_frame
from run_chronos2_hourly import _feature_inputs, _residual_corrector_factory

cfg = r.load_config(r.ROOT/'config/nyx_clean_fuel.yaml')
work = r.source_dir(cfg, 'FR')
source = load_nuclear_result_bundle(workdir=work)
config = yaml.safe_load((work/'resolved_config.yaml').read_text(encoding='utf-8-sig'))
spec = build_zone_configs(config, ['FR'], None, None)[0]
data = prepare_nuclear_zone_data(spec, config, work, r.ROOT/'tmp/clean_fuel_smoke_prepared')
X = _feature_inputs(data, config)[-1]
assert _digest_frame(X) == source.audit['source_hashes']['residual_features']
raw = r.load_raw_history(work, source)
paths = sorted((r.NAMESPACE/'inputs'/cfg['delivery_day']/'daily').glob('*.json'))
records = [json.loads(p.read_text()) for p in paths]
for rec in records:
    assert rec['row_sha256'] == _sha(_json_bytes(rec['row']))
    assert rec['contract_sha256'] == _sha(_json_bytes(_contract(176.0)))
bank = pd.DataFrame([v['row'] for v in records], columns=_columns()).sort_values('delivery_day')
days = pd.date_range(bank.delivery_day.iloc[0], periods=41).strftime('%Y-%m-%d').tolist()
bank = _validate_frame(bank[bank.delivery_day.isin(days)], maximum_age_hours=176, expected_days=days)
base = raw.loc[raw.index.tz_convert(spec.timezone).strftime('%Y-%m-%d').isin(days)]
extras = build_features(bank, base.index, zone='FR', timezone=spec.timezone, base=base)
X = X.loc[base.index].join(extras)
recipe = source.audit['residual_recipe'].copy()
recipe['thread_count'] = 4
factory, _ = _residual_corrector_factory({'hourly': {'residual_correction': recipe}}, timezone=spec.timezone)
train = base.index.tz_convert(spec.timezone).strftime('%Y-%m-%d') < days[-1]
model = factory()
experts = base.loc[:,r.Q].rename(columns={q:'chronos2__'+q for q in r.Q})
start=time.monotonic()
model.fit(X.loc[train], base.actual.loc[train], base.loc[train,r.Q], experts.loc[train])
prediction=model.predict(X.loc[~train], base.loc[~train,r.Q], experts.loc[~train])
assert set(extras).issubset(model.feature_columns_)
assert np.isfinite(prediction).all().all() and (np.diff(prediction.loc[:,r.Q].to_numpy(),axis=1)>=0).all()
print(json.dumps({'status':'smoke_passed', 'not_annual_backtest':True, 'training_days':40,
                  'prediction_day':days[-1], 'hours':len(prediction), 'added_features':len(extras.columns),
                  'fit_predict_seconds':round(time.monotonic()-start,2), 'production_modified':False}))
