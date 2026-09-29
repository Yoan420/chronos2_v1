"""Optional SolarWind launch contract; imported only by the opt-in path."""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import uuid
from zoneinfo import ZoneInfo

import yaml

from .security import has_secrets, redact
from .store import now

ADAPTER = 'solarwind_interaction40'
MODEL = 'SolarWind interaction ±40'
SOURCES = (
    'run_solarwind_forecast.py',
    'chronos2_hourly/solarwind_live.py',
    'config/nuclear_forecast.yaml',
    'config/kalman_operational.yaml',
    'config/chronos2_exogenous_activation_v1.yaml',
    'chronos2_hourly_de_residual_candidate_v1.yaml',
    'chronos2_hourly_nl_residual_candidate_v1.yaml',
)
RESOURCES = ['scientific-cache', 'nyx-primary-pipeline', 'solarwind-interaction40']


def defaults(registry, delivery_day=None):
    from .adapters import validate_primary_delivery_day
    selected = validate_primary_delivery_day(delivery_day) if delivery_day is not None else (
        datetime.now(ZoneInfo('Europe/Paris')).date() + timedelta(days=1)).isoformat()
    for relative in SOURCES:
        registry._input_path(relative)
    output = registry.project_root / 'runs/solarwind_interaction40' / selected
    if output.resolve() != output or not output.is_relative_to(registry.project_root):
        raise ValueError('Les résultats SolarWind doivent rester dans ce dépôt.')
    return {'delivery_day': selected, 'countries': ['DE', 'NL'], 'model': MODEL,
            'output': str(output / 'index.html'), 'output_dir': str(output),
            'command': [str(registry.python_executable), '-u',
                        str(registry.project_root / 'run_solarwind_forecast.py'),
                        '--delivery-day', selected],
            'delivery_day_resolved_at_execution': False}


def prepare(registry, run_directory, *, write=False, delivery_day):
    from .adapters import _no_secrets, _read_yaml
    root = Path(run_directory).resolve()
    if not root.is_relative_to(registry.project_root) or root == registry.project_root:
        raise ValueError('Les métadonnées SolarWind doivent rester sous ce dépôt.')
    snapshot = root / 'config.yaml'
    if snapshot.exists():
        raise FileExistsError('Ce lancement SolarWind possède déjà un snapshot.')
    values = defaults(registry, delivery_day)
    sources = {}
    for relative in SOURCES:
        source = registry._input_path(relative)
        if source.stat().st_size > 2_000_000:
            raise ValueError('Source SolarWind trop volumineuse pour son snapshot.')
        content = source.read_bytes()
        _no_secrets(_read_yaml(source) if source.suffix in {'.yaml', '.yml'} else content.decode('utf-8-sig'))
        sources[relative] = content
    config = {'adapter_id': ADAPTER, 'model': MODEL, 'defaults_at_request': values,
              'selected_delivery_day': delivery_day, 'snapshots_for_audit_only': True,
              'source_hashes': {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()}}
    prepared = {'adapter_id': ADAPTER, 'type': 'solarwind_forecast', 'model': MODEL,
                'command': values['command'], 'cwd': str(registry.project_root),
                'config': config, 'config_path': str(snapshot), 'delivery_day': delivery_day,
                'output_dir': values['output_dir'], 'resource_keys': list(RESOURCES),
                'warnings': ['Variante SolarWind interaction ±40 pour Allemagne et Pays-Bas.',
                             'Résultats séparés de la production ; les sources doivent être disponibles.'],
                'request': {'adapter_id': ADAPTER, 'delivery_day': delivery_day}}
    _no_secrets(prepared)
    if write:
        root.mkdir(parents=True, exist_ok=True)
        with snapshot.open('x', encoding='utf-8') as stream:
            yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
        for relative, content in sources.items():
            target = root / 'source_snapshots' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(content)
    return prepared


def validate_sources(registry, run):
    from .adapters import validate_primary_delivery_day
    selected = validate_primary_delivery_day(run.get('request', {}).get('delivery_day'))
    if selected != run.get('delivery_day') or selected != run.get('config', {}).get('selected_delivery_day'):
        raise ValueError('La date SolarWind ne correspond plus à la sélection enregistrée.')
    expected = defaults(registry, selected)
    if run.get('command') != expected['command'] or run.get('output_dir') != expected['output_dir']:
        raise ValueError('Le lanceur SolarWind a changé. Préparez un nouveau lancement.')
    hashes = run.get('config', {}).get('source_hashes', {})
    if set(hashes) != set(SOURCES):
        raise ValueError('Le snapshot des sources SolarWind est incomplet.')
    for relative, expected_hash in hashes.items():
        if hashlib.sha256(registry._input_path(relative).read_bytes()).hexdigest() != expected_hash:
            raise ValueError('Une source SolarWind a changé pendant l’attente. Préparez un nouveau lancement.')


def summary(run):
    if run is None:
        return None
    phases = {'queued': 'En attente des ressources scientifiques', 'starting': 'Démarrage de SolarWind',
              'running': 'Calcul SolarWind interaction ±40 · Allemagne et Pays-Bas',
              'cancelling': 'Arrêt demandé', 'cancelled': 'Exécution annulée',
              'succeeded': 'Calcul SolarWind terminé', 'failed': 'Le calcul SolarWind a échoué',
              'interrupted': 'Superviseur interrompu', 'unknown': 'État non confirmé'}
    status = run.get('status')
    value = {key: run.get(key) for key in ('id', 'status', 'created_at', 'started_at', 'finished_at',
                                        'delivery_day', 'duration_seconds')}
    value.update(model=MODEL, exit_code=run.get('return_code'), progress={
        'phase': phases.get(status, 'État non confirmé'), 'completed': None, 'total': None,
        'percent': None, 'steps': [], 'source': 'process',
        'indeterminate': status in {'starting', 'running', 'cancelling'}})
    if status in {'failed', 'interrupted', 'unknown'}:
        value['error'] = run.get('failure_summary') or run.get('activity') or phases[status]
    return redact(value)


def status(manager):
    from .manager import ACTIVE
    runs = [run for run in manager.store.list() if manager._is_solarwind(run)]
    active = next((run for run in runs if run['status'] in ACTIVE | {'queued'}), None)
    result = {'available': True, 'defaults': {}, 'active': summary(active),
              'latest': summary(runs[0] if runs else None)}
    try:
        result['defaults'] = manager.registry.solarwind_defaults()
    except (AttributeError, OSError, ValueError, KeyError) as exc:
        result.update(available=False, warning=str(redact(str(exc))))
    environment_error = manager._primary_environment_error()
    if environment_error:
        result.update(available=False, warning=environment_error)
    return redact(result)


def launch(manager, idempotency_key=None, *, delivery_day):
    from .adapters import AdapterRegistry, validate_primary_delivery_day
    from .manager import ACTIVE, SolarWindRunConflict
    delivery_day = validate_primary_delivery_day(delivery_day)
    if idempotency_key is not None and (not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 150):
        raise ValueError('Clé de lancement SolarWind invalide.')
    request_key = 'solarwind:' + (idempotency_key or uuid.uuid4().hex)
    with manager._lock:
        if manager._stop.is_set():
            raise ValueError('Le backend est fermé.')
        previous = manager.store.request_run(request_key)
        if previous:
            if not manager._is_solarwind(previous) or previous.get('request', {}).get('delivery_day') != delivery_day:
                raise SolarWindRunConflict('Cette clé SolarWind appartient à une autre demande de livraison.')
            return {'run': summary(previous), 'already_active': previous['status'] in ACTIVE | {'queued'}}
        active = next((run for run in manager.store.list()
                       if manager._is_solarwind(run) and run['status'] in ACTIVE | {'queued'}), None)
        if active:
            if active.get('delivery_day') != delivery_day:
                raise SolarWindRunConflict('Un calcul SolarWind est déjà actif ou en attente pour une autre livraison.')
            manager.store.bind_request(request_key, active['id'])
            return {'run': summary(active), 'already_active': True}
        environment_error = manager._primary_environment_error()
        if environment_error:
            raise ValueError(environment_error)
        run_id = uuid.uuid4().hex
        run_dir = manager.state_root / 'executions' / run_id
        prepared = manager.registry.prepare_solarwind_run(run_dir, write=False, delivery_day=delivery_day)
        if prepared.get('delivery_day') != delivery_day:
            raise ValueError('Le lancement SolarWind ne conserve pas la livraison choisie.')
        if has_secrets(prepared):
            raise ValueError('Le lancement SolarWind contient une valeur sensible non persistable.')
        if not prepared.get('command') or Path(prepared['command'][0]).resolve() != manager.python_executable:
            raise ValueError('SolarWind doit utiliser le Python configuré dans NYX.')
        if isinstance(manager.registry, AdapterRegistry):
            values = defaults(manager.registry, delivery_day)
            if prepared['command'] != values['command'] or prepared['output_dir'] != values['output_dir']:
                raise ValueError('Commande ou sortie SolarWind non autorisée.')
        run_dir.mkdir(parents=True, exist_ok=False)
        written = manager.registry.prepare_solarwind_run(run_dir, write=True, delivery_day=delivery_day)
        if any(written.get(key) != prepared.get(key)
               for key in ('command', 'config', 'output_dir', 'resource_keys', 'delivery_day')):
            raise ValueError('Les sources SolarWind ont changé pendant leur copie. Réessayez.')
        run = {**prepared, 'id': run_id, 'adapter_id': ADAPTER, 'source': 'managed',
               'source_key': str(run_dir), 'run_dir': str(run_dir), 'created_at': now(),
               'request': {'adapter_id': ADAPTER, 'delivery_day': delivery_day},
               'name': 'NYX · ' + MODEL, 'description': 'SolarWind interaction ±40 · DE/NL · résultats isolés.',
               'python_executable': str(manager.python_executable), 'status': 'queued',
               'started_at': None, 'finished_at': None, 'duration_seconds': None, 'return_code': None,
               'cancel_requested': False, 'tags': [], 'note': '',
               'activity': 'En attente des ressources scientifiques', 'git': manager._git_version(),
               'log_paths': [str(run_dir / 'console.log')],
               'resource_keys': sorted(set(prepared.get('resource_keys', [])) | set(RESOURCES))}
        (run_dir / 'metadata.json').write_text(json.dumps(redact(run), indent=2, ensure_ascii=False), encoding='utf-8')
        (run_dir / 'console.log').touch(exist_ok=False)
        with manager.store.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT INTO runs VALUES (?,?,?)', (run['id'], run['source_key'], json.dumps(run, ensure_ascii=False)))
            db.execute('INSERT INTO requests VALUES (?,?)', (request_key, run['id']))
        return {'run': summary(run), 'already_active': False}
