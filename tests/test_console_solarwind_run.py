"""Opt-in isolation, queue serialization and tiny-process tests; no science runs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time

import pytest

from experiment_console.adapters import AdapterRegistry
from experiment_console.manager import Manager
from experiment_console.solarwind_runs import ADAPTER, SOURCES
from test_console_primary_run import (DatedPrimaryFixtureRegistry, primary_run_api,
                                      post_primary, restart_backend)


class SolarWindFixtureRegistry(DatedPrimaryFixtureRegistry):
    def __init__(self, project):
        super().__init__(project)
        self.solarwind_mode = 'success'
        self.changed = False

    def solarwind_defaults(self, delivery_day=None):
        return {'delivery_day': delivery_day or self.tomorrow, 'countries': ['DE', 'NL'],
                'model': 'SolarWind interaction ±40'}

    def prepare_solarwind_run(self, run_directory, write=False, *, delivery_day):
        prepared = super().prepare_primary_run(run_directory, write=write, delivery_day=delivery_day)
        prepared['command'][-2] = self.solarwind_mode
        prepared.update(adapter_id=ADAPTER, model='SolarWind interaction ±40')
        return prepared

    def validate_solarwind_sources(self, run):
        if self.changed:
            raise ValueError('Une source SolarWind a changé pendant l’attente.')


def post_solarwind(api, day='2026-09-12', key=None):
    headers = {**api.headers, **({'Idempotency-Key': key} if key is not None else {})}
    return api.client.post('/api/solarwind-run', json={'delivery_day': day}, headers=headers)


def test_optional_unavailable_does_not_change_legacy_launch(primary_run_api):
    api = primary_run_api(registry_type=DatedPrimaryFixtureRegistry)
    optional = api.client.get('/api/solarwind-run')
    assert optional.status_code == 200
    assert optional.json()['available'] is False
    assert api.client.get('/api/primary-run').json()['available'] is True
    response = post_primary(api)
    assert response.status_code == 202, response.text
    saved = api.manager.get(response.json()['run']['id'])
    assert saved['adapter_id'] == 'primary_nuclear_kalman'
    assert 'delivery_day' not in saved['request']
    assert api.registry.prepare_calls == [(None, False), (None, True)]


@pytest.mark.parametrize('body', [{}, {'delivery_day': None}, {'delivery_day': '2026-02-30'},
    {'delivery_day': '2026-09-12', 'model': 'NYX'}, {'delivery_day': '2026-09-12', 'command': []}])
def test_solarwind_endpoint_requires_date_and_rejects_overrides(primary_run_api, body):
    api = primary_run_api(registry_type=SolarWindFixtureRegistry)
    result = api.client.post('/api/solarwind-run', json=body, headers=api.headers)
    assert result.status_code == 400, result.text
    assert api.manager.list_runs() == []


def test_opt_in_has_separate_identity_durable_key_and_date(primary_run_api):
    api = primary_run_api(registry_type=SolarWindFixtureRegistry)
    primary = post_primary(api, key='same-key', body={'delivery_day': '2026-09-12'})
    assert primary.status_code == 202
    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(pool.map(lambda _: post_solarwind(api, key='same-key'), range(6)))
    assert all(response.status_code == 202 for response in responses)
    ids = {response.json()['run']['id'] for response in responses}
    assert len(ids) == 1
    run_id = ids.pop()
    assert run_id != primary.json()['run']['id']
    assert len(api.manager.list_runs()) == 2
    saved = api.manager.get(run_id)
    assert saved['request'] == {'adapter_id': ADAPTER, 'delivery_day': '2026-09-12'}
    assert 'scientific-cache' in saved['resource_keys']
    assert api.client.get('/api/primary-run').json()['active']['id'] == primary.json()['run']['id']
    assert api.client.get('/api/solarwind-run').json()['active']['id'] == run_id
    api.manager.store.update(run_id, status='failed', return_code=7)
    restart_backend(api)
    repeated = post_solarwind(api, key='same-key')
    assert repeated.status_code == 202
    assert repeated.json()['run']['id'] == run_id
    assert repeated.json()['run']['status'] == 'failed'
    assert post_solarwind(api, day='2026-09-13', key='same-key').status_code == 409
    assert len(api.manager.list_runs()) == 2


def test_active_solarwind_conflicts_on_another_date(primary_run_api):
    api = primary_run_api(registry_type=SolarWindFixtureRegistry)
    first = post_solarwind(api)
    assert first.status_code == 202
    assert post_solarwind(api, day='2026-09-13').status_code == 409
    assert len(api.manager.list_runs()) == 1


def test_missing_optional_sources_do_not_prepare_or_launch_primary(tmp_path, monkeypatch):
    project = tmp_path / 'repo'
    project.mkdir()
    manager = Manager(project, project / 'state', sys.executable, start_scheduler=False)
    monkeypatch.setattr(manager, '_primary_environment_error', lambda: None)
    monkeypatch.setattr(manager.registry, 'prepare_primary_run',
                        lambda *a, **k: pytest.fail('SolarWind must never fall back to primary'))
    try:
        assert manager.solarwind_run_status()['available'] is False
        with pytest.raises(ValueError, match='Source introuvable'):
            manager.launch_solarwind_run(delivery_day='2026-09-12')
        assert manager.list_runs() == []
    finally:
        manager.close()


def test_real_adapter_snapshots_sources_and_refuses_changes(tmp_path):
    project = tmp_path / 'repo'
    project.mkdir()
    for relative in SOURCES:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n' if path.suffix == '.yaml' else '# Non-executable source fixture.\n', encoding='utf-8')
    production = project / 'runs/reports/model_storm/existing.html'
    production.parent.mkdir(parents=True)
    production.write_text('Existing production report', encoding='utf-8')
    registry = AdapterRegistry(project, Path(sys.executable).resolve())
    directory = project / 'state/executions/new'
    prepared = registry.prepare_solarwind_run(directory, write=False, delivery_day='2026-09-12')
    assert not directory.exists()
    assert prepared['command'] == [str(Path(sys.executable).resolve()), '-u',
                                  str(project / 'run_solarwind_forecast.py'), '--delivery-day', '2026-09-12']
    assert Path(prepared['output_dir']) == project / 'runs/solarwind_interaction40/2026-09-12'
    assert 'canonical-publications' not in prepared['resource_keys']
    written = registry.prepare_solarwind_run(directory, write=True, delivery_day='2026-09-12')
    assert written == prepared
    assert set(written['config']['source_hashes']) == set(SOURCES)
    assert all((directory / 'source_snapshots' / relative).is_file() for relative in SOURCES)
    registry.validate_solarwind_sources(written)
    (project / SOURCES[-1]).write_text('modified: true\n', encoding='utf-8')
    with pytest.raises(ValueError, match='changé'):
        registry.validate_solarwind_sources(written)
    assert production.read_text(encoding='utf-8') == 'Existing production report'
    assert not Path(prepared['output_dir']).exists()


def test_shared_resources_serialize_primary_and_solarwind(primary_run_api, monkeypatch):
    from types import SimpleNamespace
    import experiment_console.manager as manager_module
    api = primary_run_api(registry_type=SolarWindFixtureRegistry)
    api.manager.max_concurrency = 2
    primary = post_primary(api).json()['run']['id']
    optional = post_solarwind(api).json()['run']['id']
    launched = []
    def popen(command, **kwargs):
        launched.append(command[-1])
        return SimpleNamespace(pid=1000 + len(launched), poll=lambda: None)
    monkeypatch.setattr(manager_module.subprocess, 'Popen', popen)
    monkeypatch.setattr(api.manager, 'reconcile', lambda: None)
    api.manager._tick()
    assert launched == [primary]
    assert api.manager.get(optional)['status'] == 'queued'
    api.manager.store.update(primary, status='failed', return_code=7)
    api.manager._tick()
    assert launched == [primary, optional]
    assert api.manager.get(optional)['status'] == 'starting'


def test_optional_validation_failure_allows_primary_to_dispatch(primary_run_api, monkeypatch):
    from types import SimpleNamespace
    import experiment_console.manager as manager_module
    api = primary_run_api(registry_type=SolarWindFixtureRegistry)
    optional = post_solarwind(api).json()['run']['id']
    primary = post_primary(api).json()['run']['id']
    api.registry.changed = True
    launched = []
    def popen(command, **kwargs):
        launched.append(command[-1])
        return SimpleNamespace(pid=2000, poll=lambda: None)
    monkeypatch.setattr(manager_module.subprocess, 'Popen', popen)
    api.manager._tick()
    assert api.manager.get(optional)['status'] == 'failed'
    assert launched == [primary]
    assert api.manager.get(primary)['status'] == 'starting'
    assert api.client.get('/api/solarwind-run').json()['latest']['error']


def test_optional_process_failure_is_independent_of_primary(primary_run_api):
    api = primary_run_api(start_scheduler=True, registry_type=SolarWindFixtureRegistry)
    api.registry.solarwind_mode = 'failure'
    optional = post_solarwind(api).json()['run']['id']
    primary = post_primary(api).json()['run']['id']
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        if all(api.manager.get(run_id)['status'] in {'succeeded', 'failed', 'cancelled', 'interrupted'}
               for run_id in (optional, primary)):
            break
        time.sleep(.05)
    assert api.manager.get(optional)['status'] == 'failed'
    assert api.manager.get(optional)['return_code'] == 7
    assert api.manager.get(primary)['status'] == 'succeeded'
    assert api.client.get('/api/solarwind-run').json()['latest']['status'] == 'failed'
    assert api.client.get('/api/primary-run').json()['latest']['status'] == 'succeeded'
