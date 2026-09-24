import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from nyx_congestion import runner

ROOT = Path(__file__).resolve().parents[1]


def test_config_is_research_only():
    config = runner.load_config(ROOT/'config/nyx_congestion.yaml')
    for field, value in [('production_modified', True), ('activation_performed', True),
                         ('diagnostic_only', False), ('output_root', 'runs/exports'),
                         ('unknown', 1), ('options', {'threads':4}), ('options', {'threads':True})]:
        changed = copy.deepcopy(config); changed[field] = value
        with pytest.raises(ValueError): runner.validate_config(changed)


@pytest.mark.parametrize('path', ['runs/exports/a', '../outside', 'runs/experiments/nyx_congestion_v1/../../outside'])
def test_output_escape_refused(tmp_path, path):
    with pytest.raises(ValueError): runner.safe_path(tmp_path, path)


def test_checkpoint_sealed_before_load(tmp_path):
    directory = tmp_path/runner.NAMESPACE/'snapshots/test/activation'
    cache = runner.FitCache(directory, 'one', root=tmp_path)
    state = {'fit_day':'2026-09-14', 'model':'only synthetic test data'}
    cache.save('2026-09-14', state)
    assert cache.load('2026-09-14') == state
    with pytest.raises(ValueError): runner.FitCache(directory, 'two', root=tmp_path).load('2026-09-14')
    with pytest.raises(ValueError): cache.save('2026-09-14', state)
    with pytest.raises(ValueError): cache.load('../escape')
    path = directory/'fits/2026-09-14.joblib'
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError): cache.load('2026-09-14')


def test_forecast_support_and_quantiles():
    panel = pd.DataFrame({'forecast':[10., 20.], 'actual':[12., 25.]})
    pred = panel.copy()
    for model in runner.MODELS:
        pred[model], pred[model+'_q10'], pred[model+'_q90'] = panel.forecast, panel.forecast-5, panel.forecast+5
    runner.validate_predictions(panel, pred)
    for name,value in [('actual',0), (runner.MODELS[0],np.nan), (runner.MODELS[0]+'_q10',100.)]:
        broken = pred.copy(); broken.loc[0,name] = value
        with pytest.raises((ValueError, AssertionError)): runner.validate_predictions(panel, broken)


def test_activation_result_identity_and_sha(tmp_path):
    directory=tmp_path/runner.NAMESPACE/'snapshots/test'; directory.mkdir(parents=True)
    (directory/'manifest.json').write_text('{}')
    for name in runner.ACTIVATION_FILES: (directory/name).write_text('fixture')
    content={'suite_manifest_sha256':runner.digest(directory/'manifest.json'),
             'files':{n:runner.digest(directory/n) for n in runner.ACTIVATION_FILES}}
    (directory/'activation_manifest.json').write_text(json.dumps(content))
    runner.verify_activation(directory)
    (directory/'signals.parquet').write_text('tampered')
    with pytest.raises(ValueError): runner.verify_activation(directory)


def test_first_collection_creates_audit_directory(tmp_path, monkeypatch):
    from nyx_congestion import data
    config = runner.load_config(ROOT/'config/nyx_congestion.yaml')
    monkeypatch.setattr(data, 'collect_labels', lambda *a, **k: {'status':'incomplete', 'days':[], 'failures':[{'day':'2026-09-14'}]})
    monkeypatch.setattr(runner.source_runner, 'protected_state', lambda root: {})
    result=runner.collect(config, root=tmp_path, start_day='2026-09-14', end_day='2026-09-14')
    assert Path(result['audit_path']).is_file()
    assert result['failed_days'] == 1


def test_partial_collection_exits_nonzero(monkeypatch):
    import run_nyx_congestion as cli
    monkeypatch.setattr(runner, 'collect', lambda *a, **k: {'status':'incomplete', 'failed_days':1})
    assert cli.main(['--action','collect','--start-day','2026-09-14','--end-day','2026-09-14']) == 1
