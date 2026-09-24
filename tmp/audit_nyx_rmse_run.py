"""Read-only numerical audit of a completed local experiment."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from nyx_rmse import runner

directory = Path(sys.argv[1])
directory, config, manifest = runner.read_snapshot(directory, root=ROOT)
runner.verify_result(directory, manifest)
panel = pd.read_parquet(directory/'panel.parquet')
pred = pd.read_parquet(directory/'predictions.parquet')
runner.validate_predictions(panel, pred)
folds = pd.read_parquet(directory/'folds.parquet')
trained = folds.loc[folds.status.eq('trained')]
assert (pd.to_datetime(trained.mean_max_label_available_at_utc, utc=True) <= pd.to_datetime(trained.fit_cutoff_utc, utc=True)).all()
gov = pd.read_parquet(directory/'governance.parquet')
known = gov.max_label_available_at_utc.notna()
assert (pd.to_datetime(gov.loc[known, 'max_label_available_at_utc'], utc=True) <= pd.to_datetime(gov.loc[known, 'forecast_origin_utc'], utc=True)).all()
assert gov.groupby(['zone','delivery_day','strategy']).selected.sum().eq(1).all()
cache = runner.FitCache(directory, runner.digest(directory/'manifest.json'), root=ROOT)
new = cache.load(str(trained.fit_day.iloc[-1]))
source = Path(manifest['source_dir'])
# Hash checked by read_snapshot before deserialization of trusted local artifact.
old = joblib.load(source/'forest/latest_model.joblib')['state']
assert new['fit_day'] == old['fit_day']
comparisons = {}
for regime in ('normal', 'spike'):
    fresh = new['models']['forests'][regime]
    historical = old['distributions']['regimes'][regime]['forest']
    assert len(fresh.estimators_) == len(historical.estimators_)
    max_delta = 0.
    for left, right in zip(fresh.estimators_, historical.estimators_):
        assert np.array_equal(left.tree_.feature, right.tree_.feature)
        assert np.array_equal(left.tree_.children_left, right.tree_.children_left)
        np.testing.assert_allclose(left.tree_.threshold, right.tree_.threshold, rtol=0, atol=1e-12)
        np.testing.assert_allclose(left.tree_.value, right.tree_.value, rtol=0, atol=1e-12)
        max_delta = max(max_delta, float(np.max(np.abs(left.tree_.value-right.tree_.value))))
    comparisons[regime] = {'trees':len(fresh.estimators_),'maximum_leaf_mean_difference':max_delta}
print(json.dumps({'status':'passed','rows':len(pred),'trained_folds':len(trained),
    'governance_decisions':int(gov.selected.sum()),'original_inputs_preserved':True,
    'future_labels_excluded':True,'last_forest_identical_to_original_cdf':comparisons},indent=2))
