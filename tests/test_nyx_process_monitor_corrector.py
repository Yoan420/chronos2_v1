"""Instrumented ablation counters do not imply equal-duration scientific steps."""
import json
from datetime import datetime, timezone

import pytest

from nyx_process_monitor.collector import Collector, PARALLEL_CORRECTOR_RESULTS, REUSE_CORRECTOR_RESULTS


@pytest.mark.parametrize('counter,expected', [
    ({'completed': 12, 'total': 731, 'unit': 'jours'}, '12/731 jours'),
    ({'completed': True, 'total': 731}, None),
    ({'completed': 732, 'total': 731}, None),
    ({'completed': 1, 'total': 0}, None),
    ({'completed': 1, 'total': 1000001}, None),
])
@pytest.mark.parametrize('engine_name,title', [
    ('solar_wind_corrector_interaction_v1', 'SolarWind — correcteur × plafond +80'),
    ('solar_wind_corrector_parallel_v1', 'SolarWind — correcteur × plafond +80 · parallèle'),
    ('solar_wind_corrector_reuse_v1', 'SolarWind — correcteur × plafond +80 · réemploi vérifié'),
])
def test_corrector_phase_counter_keeps_total_percent_indeterminate(tmp_path, monkeypatch, counter, expected, engine_name, title):
    root = tmp_path / 'project'
    engine = root / 'runs/experiments' / engine_name
    stamp = datetime.now(timezone.utc).isoformat()
    identity = 'b' * 16
    logs = engine / 'launcher_logs'
    logs.mkdir(parents=True)
    (logs / 'sample.launch.json').write_text(json.dumps({
        'started_utc': stamp, 'python_pid': 42, 'python_created_utc': stamp,
        'delivery_day': '2026-09-22', 'run_identities': {'DE': identity},
    }), encoding='utf-8')
    directory = engine / '2026-09-22/de' / identity
    directory.mkdir(parents=True)
    (directory / 'status.json').write_text(json.dumps({
        'identity': identity, 'zone': 'DE', 'status': 'RUNNING',
        'phase': 'catboost_replay', 'updated_utc': stamp, 'progress': counter,
    }), encoding='utf-8')
    collector = Collector(root)
    monkeypatch.setattr(collector, '_processes', lambda: ({42: {
        'pid': 42, 'ppid': 1, 'created': datetime.fromisoformat(stamp).timestamp(),
        'argv': ['python.exe', 'run_solar_wind_corrector_interaction.py'],
        'name': 'python.exe', 'cpu': 1, 'memory': 1024,
    }}, []))
    job = collector.snapshot()['jobs'][0]
    assert job['title'] == title
    assert job['active'] is True
    assert job['progress']['percent'] is None
    assert job['eta']['remaining_seconds'] > 0
    if expected:
        assert expected in job['progress']['label']
    else:
        assert job['zones'][0]['phase_progress'] is None


@pytest.mark.parametrize('defect', [None, 'missing_migration', 'partial_year', 'missing_forecast', 'no_annual_receipt'])
@pytest.mark.parametrize('reuse', [False, True])
def test_parallel_completion_requires_exact_inventory_and_annual_scope(tmp_path, defect, reuse):
    root = tmp_path / 'project'
    engine = 'solar_wind_corrector_reuse_v1' if reuse else 'solar_wind_corrector_parallel_v1'
    identity = 'c' * 16
    directory = root / 'runs/experiments' / engine / '2026-09-22/de' / identity
    directory.mkdir(parents=True)
    inventory = {name: 'd' * 64 for name in (REUSE_CORRECTOR_RESULTS if reuse else PARALLEL_CORRECTOR_RESULTS)}
    assert len(inventory) == (24 if reuse else 23)
    for name in inventory:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}', encoding='utf-8')
    state = dict(status='COMPLETE', identity=identity, zone='DE', annual_complete=True,
                 evaluation_days=365, evaluation_hours=8760, future_forecast_hours=24)
    receipt = dict(identity=identity, annual_complete=True, files=inventory)
    if defect == 'missing_migration':
        inventory.pop('migration_audit.json')
    elif defect == 'partial_year':
        state['evaluation_days'] = 1
    elif defect == 'missing_forecast':
        state['future_forecast_hours'] = 0
    elif defect == 'no_annual_receipt':
        receipt['annual_complete'] = False
    (directory / 'status.json').write_text(json.dumps(state), encoding='utf-8')
    (directory / 'completion.json').write_text(json.dumps(receipt), encoding='utf-8')
    zones, warnings = Collector(root)._zones(engine, {'run_identities': {'DE': identity}})
    assert zones[0]['status'] == ('complete' if defect is None else 'unknown')
    assert bool(warnings) is (defect is not None)
