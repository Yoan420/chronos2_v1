"""Durable local queue; browser requests never own calculation processes."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import uuid

from filelock import FileLock, Timeout
import psutil

from .processes import alive
from .security import has_secrets, redact
from .store import Store, now

ACTIVE = {'starting', 'running', 'cancelling'}
TERMINAL = {'succeeded', 'failed', 'cancelled', 'interrupted', 'unknown'}
PRIMARY_ADAPTER = 'primary_nuclear_kalman'


class PrimaryRunConflict(ValueError):
    """A retry or active run belongs to a different delivery-date choice."""


class Manager:
    def __init__(self, project_root, state_root, python_executable, max_concurrency=1, registry=None, start_scheduler=True):
        self.project_root = Path(project_root).resolve()
        self.state_root = Path(state_root).resolve()
        self.python_executable = Path(python_executable)
        if not self.python_executable.is_absolute() or not self.python_executable.is_file():
            raise ValueError('Configurez un chemin absolu vers un interpréteur Python existant.')
        self.python_executable = self.python_executable.resolve()
        if not 1 <= int(max_concurrency) <= 8:
            raise ValueError('La concurrence doit être comprise entre 1 et 8.')
        self.max_concurrency = int(max_concurrency)
        self.state_root.mkdir(parents=True, exist_ok=True)
        self._instance_lock = FileLock(str(self.state_root / 'backend.lock'), thread_local=False)
        try:
            self._instance_lock.acquire(timeout=0)
        except Timeout as exc:
            raise ValueError('Une console utilise déjà ce dossier de métadonnées.') from exc
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._children = {}
        self.scheduler_error = None
        self._thread = None
        try:
            self.store = Store(self.state_root / 'console.sqlite3')
            if registry is None:
                from .adapters import AdapterRegistry
                registry = AdapterRegistry(self.project_root, self.python_executable)
            self.registry = registry
            self.reconcile()
            if start_scheduler:
                self.start()
        except BaseException:
            self.close()
            raise

    def start(self):
        """Start dispatch once, after the caller has bound its local service."""
        with self._lock:
            if self._stop.is_set():
                raise ValueError('Cette instance de la console a déjà été fermée.')
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._loop, name='experiment-queue', daemon=True)
            try:
                self._thread.start()
            except BaseException:
                self._thread = None
                raise

    def preview(self, request):
        if not isinstance(request, dict):
            raise ValueError('Requête de lancement invalide.')
        if self.registry.__class__.__module__ == 'experiment_console.adapters':
            allowed = {'adapter_id', 'config_id', 'model', 'parameters', 'name', 'description', 'duplicate_of'}
            if set(request) - allowed:
                raise ValueError('Champs de lancement non autorisés : ' + ', '.join(sorted(set(request) - allowed)))
        if has_secrets(request):
            raise ValueError('La demande contient un secret. Utilisez les variables d’environnement du processus backend.')
        run_id = uuid.uuid4().hex
        run_dir = self.state_root / 'executions' / run_id
        prepared = self._prepare(request, run_dir, write=False)
        if has_secrets(prepared):
            raise ValueError('La configuration ou la commande contient un secret : lancement refusé pour ne pas le persister.')
        command = prepared.get('command', [])
        if not command or Path(command[0]).resolve() != self.python_executable:
            raise ValueError('L’adaptateur doit utiliser l’environnement Python configuré.')
        output = Path(prepared['output_dir']).resolve()
        if not output.is_relative_to(run_dir.resolve()):
            raise ValueError('La sortie doit être isolée dans le dossier du nouveau run.')
        normalized = {**request, **prepared.get('request', {})}
        plan = {**prepared, 'id': run_id, 'run_dir': str(run_dir), 'created_at': now(), 'request': normalized, 'name': str(request.get('name') or prepared.get('model') or 'Run')[:150], 'description': str(request.get('description', ''))[:4000], 'python_executable': str(self.python_executable)}
        self.store.save_plan(plan)
        return plan

    def _prepare(self, request, run_dir, write=False):
        if any(k.startswith('_') or k == 'snapshot_config' for k in request):
            raise ValueError('Configuration interne non autorisée dans la requête.')
        source_id = request.get('duplicate_of')
        if source_id and request.get('adapter_id') == 'hourly_forecast':
            previous = self.get(source_id)
            if previous.get('source') != 'managed' or previous.get('request', {}).get('adapter_id') != 'hourly_forecast':
                raise ValueError('Snapshot de duplication indisponible.')
            if request.get('config_id') != previous['request'].get('config_id'):
                raise ValueError('Pour choisir une autre configuration, créez un run sans duplication.')
            return self.registry.prepare(request, run_dir, write=write, snapshot_config=previous['config'])
        return self.registry.prepare(request, run_dir, write=write)

    def launch(self, plan_id, idempotency_key):
        if not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 200:
            raise ValueError('Clé de lancement manquante ou invalide.')
        with self._lock:
            existing = self.store.request_run(idempotency_key)
            if existing:
                if existing['id'] != plan_id:
                    raise ValueError('Cette clé de lancement appartient à un autre run.')
                return existing
            try:
                existing = self.store.get(plan_id)
                self.store.bind_request(idempotency_key, plan_id)
                return existing
            except KeyError:
                pass
            plan = self.store.plan(plan_id)
            current = self._prepare(plan['request'], Path(plan['run_dir']), write=False)
            for key in ('command', 'config', 'output_dir', 'resource_keys'):
                if current.get(key) != plan.get(key):
                    raise ValueError('La configuration source a changé depuis le récapitulatif. Préparez de nouveau le run.')
            run_dir = Path(plan['run_dir'])
            run_dir.mkdir(parents=True, exist_ok=False)
            prepared = self._prepare(plan['request'], run_dir, write=True)
            if any(prepared.get(key) != plan.get(key) for key in ('command', 'config', 'output_dir', 'resource_keys')):
                raise ValueError('La configuration a changé pendant sa copie. Préparez de nouveau le run.')
            git = self._git_version()
            run = {**plan, 'source': 'managed', 'source_key': str(run_dir), 'status': 'queued', 'started_at': None, 'finished_at': None, 'duration_seconds': None, 'return_code': None, 'cancel_requested': False, 'tags': [], 'note': '', 'activity': 'En attente d’une place ou d’une ressource', 'git': git, 'log_paths': [str(run_dir / 'console.log')]}
            (run_dir / 'metadata.json').write_text(json.dumps(redact(run), indent=2, ensure_ascii=False), encoding='utf-8')
            (run_dir / 'console.log').touch(exist_ok=False)
            self.store.insert(run)
            self.store.bind_request(idempotency_key, run['id'])
            return run

    def _git_version(self):
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        try:
            commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=self.project_root, capture_output=True, text=True, timeout=5, creationflags=flags, check=True).stdout.strip()
            dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=self.project_root, capture_output=True, text=True, timeout=5, creationflags=flags, check=True).stdout)
            return {'commit': commit, 'dirty_tracked_files': dirty}
        except (OSError, subprocess.SubprocessError):
            return {'commit': None, 'dirty_tracked_files': None}

    def get(self, run_id):
        return self.store.get(run_id)

    def list_runs(self):
        return self.store.list()

    @staticmethod
    def _is_primary(run):
        return run.get('source') == 'managed' and run.get('adapter_id', run.get('request', {}).get('adapter_id')) == PRIMARY_ADAPTER

    def _primary_summary(self, run):
        if run is None:
            return None
        phases = {'queued': 'En attente des ressources scientifiques', 'starting': 'Démarrage de NuclearKalman.ps1',
                  'running': 'Synchronisation, calcul et publication', 'cancelling': 'Arrêt demandé',
                  'cancelled': 'Exécution annulée', 'succeeded': 'Calcul et publication terminés',
                  'failed': 'L’exécution a échoué', 'interrupted': 'Superviseur interrompu', 'unknown': 'État non confirmé'}
        value = {key: run.get(key) for key in ('id', 'status', 'created_at', 'started_at', 'finished_at', 'delivery_day', 'duration_seconds')}
        from .primary_progress import read_primary_progress
        value.update(exit_code=run.get('return_code'), progress=read_primary_progress(self.project_root, run))
        if run.get('status') in {'failed', 'interrupted', 'unknown'}:
            error = run.get('failure_summary')
            if not error and run.get('status') == 'failed':
                from .error_summary import read_failure_summary
                expected = self.state_root / 'executions' / run['id']
                if re.fullmatch(r'[A-Za-z0-9_-]{1,100}', str(run['id'])) and Path(run.get('run_dir', '')).absolute() == expected:
                    error = read_failure_summary(expected)
            fallback = run.get('activity')
            if fallback and fallback.startswith('Journal et statut du batch'):
                fallback = None
            value['error'] = error or fallback or phases[run['status']]
        return redact(value)

    def _primary_environment_error(self):
        from .adapters import AdapterRegistry
        if isinstance(self.registry, AdapterRegistry):
            from .runtime_environment import primary_environment_error
            return primary_environment_error()
        return None

    def primary_run_status(self):
        runs = [run for run in self.store.list() if self._is_primary(run)]
        active = next((run for run in runs if run['status'] in ACTIVE | {'queued'}), None)
        response = {'available': True, 'defaults': {}, 'active': self._primary_summary(active),
                    'latest': self._primary_summary(runs[0] if runs else None)}
        try:
            response['defaults'] = self.registry.primary_defaults()
        except (AttributeError, OSError, ValueError, KeyError) as exc:
            response.update(available=False, warning=str(redact(str(exc))))
        environment_error = self._primary_environment_error()
        if environment_error:
            response.update(available=False, warning=environment_error)
        return redact(response)

    def launch_primary_run(self, idempotency_key=None, delivery_day=None):
        from .adapters import validate_primary_delivery_day
        if delivery_day is not None:
            delivery_day = validate_primary_delivery_day(delivery_day)
        if idempotency_key is not None and (not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 150):
            raise ValueError('Clé de lancement NYX invalide.')
        request_key = 'primary:' + (idempotency_key or uuid.uuid4().hex)
        with self._lock:
            if self._stop.is_set():
                raise ValueError('Le backend est fermé.')
            environment_error = self._primary_environment_error()
            if environment_error:
                raise ValueError(environment_error)
            previous = self.store.request_run(request_key)
            if previous:
                if not self._is_primary(previous):
                    raise ValueError('Cette clé appartient à une autre demande.')
                if previous.get('request', {}).get('delivery_day') != delivery_day:
                    raise PrimaryRunConflict('Cette clé de lancement a déjà été utilisée pour un autre choix de date. Préparez une nouvelle demande.')
                return {'run': self._primary_summary(previous), 'already_active': previous['status'] in ACTIVE | {'queued'}}
            active = next((run for run in self.store.list() if self._is_primary(run) and run['status'] in ACTIVE | {'queued'}), None)
            if active:
                target_day = delivery_day if delivery_day is not None else self.registry.primary_defaults()['delivery_day']
                if active.get('delivery_day') != target_day or active.get('request', {}).get('delivery_day') != delivery_day:
                    raise PrimaryRunConflict(f"Un lancement NYX est déjà actif ou en attente pour le {active.get('delivery_day', 'jour enregistré')} avec un autre choix de date. Attendez sa fin avant d’en lancer un autre.")
                self.store.bind_request(request_key, active['id'])
                return {'run': self._primary_summary(active), 'already_active': True}
            run_id = uuid.uuid4().hex
            run_dir = self.state_root / 'executions' / run_id
            selected_args = {'delivery_day': delivery_day} if delivery_day is not None else {}
            prepared = self.registry.prepare_primary_run(run_dir, write=False, **selected_args)
            if delivery_day is not None and prepared.get('delivery_day') != delivery_day:
                raise ValueError('Le lancement préparé ne conserve pas la date de livraison sélectionnée.')
            if has_secrets(prepared):
                raise ValueError('Le lancement NYX contient une valeur sensible non persistable.')
            from .adapters import AdapterRegistry
            if isinstance(self.registry, AdapterRegistry):
                if prepared.get('command') != self.registry._primary_command(delivery_day):
                    raise ValueError('Commande NYX non autorisée.')
                if Path(prepared['output_dir']).resolve() != self.project_root / 'runs':
                    raise ValueError('Les sorties NYX doivent utiliser le dossier runs canonique.')
            elif not prepared.get('command') or Path(prepared['command'][0]).resolve() != self.python_executable:
                # Dependency-injected test adapters retain the existing Python-only
                # execution contract. HTTP cannot inject a registry or command.
                raise ValueError('Le lanceur de test doit utiliser le Python configuré.')
            run_dir.mkdir(parents=True, exist_ok=False)
            written = self.registry.prepare_primary_run(run_dir, write=True, **selected_args)
            if any(written.get(key) != prepared.get(key) for key in ('command', 'config', 'output_dir', 'resource_keys', 'delivery_day')):
                raise ValueError('Les sources du lancement ont changé pendant leur copie. Réessayez.')
            run = {**prepared, 'id': run_id, 'adapter_id': PRIMARY_ADAPTER, 'source': 'managed',
                   'source_key': str(run_dir), 'run_dir': str(run_dir), 'created_at': now(),
                   'request': {'adapter_id': PRIMARY_ADAPTER, **selected_args}, 'name': 'NYX · Nuclear Kalman',
                   'description': ('NuclearKalman.ps1 avec ses paramètres scientifiques par défaut et -NoOpen.' +
                                   (f' Livraison choisie : {delivery_day}.' if delivery_day is not None else '')),
                   'python_executable': str(self.python_executable), 'status': 'queued',
                   'started_at': None, 'finished_at': None, 'duration_seconds': None, 'return_code': None,
                   'cancel_requested': False, 'tags': [], 'note': '', 'activity': 'En attente des ressources scientifiques',
                   'git': self._git_version(), 'log_paths': [str(run_dir / 'console.log')],
                   'resource_keys': sorted(set(prepared.get('resource_keys', [])) | {'scientific-cache', 'nyx-primary-pipeline', 'canonical-publications'})}
            (run_dir / 'metadata.json').write_text(json.dumps(redact(run), indent=2, ensure_ascii=False), encoding='utf-8')
            (run_dir / 'console.log').touch(exist_ok=False)
            # Persist the run and its retry key together, even if the backend
            # exits just after accepting the request or the fixture finishes fast.
            with self.store.connection() as db:
                db.execute('BEGIN IMMEDIATE')
                db.execute('INSERT INTO runs VALUES (?,?,?)', (run['id'], run['source_key'], json.dumps(run, ensure_ascii=False)))
                db.execute('INSERT INTO requests VALUES (?,?)', (request_key, run['id']))
            return {'run': self._primary_summary(run), 'already_active': False}

    def cancel(self, run_id):
        with self._lock:
            run = self.get(run_id)
            if run.get('source') != 'managed':
                raise ValueError('Les exécutions externes sont suivies en lecture seule.')
            if run['status'] in TERMINAL:
                return run
            if run['status'] == 'queued':
                return self.store.update(run_id, status='cancelled', cancel_requested=True, finished_at=now(), activity='Annulé dans la file d’attente')
            return self.store.update(run_id, cancel_requested=True, status='cancelling', activity='Arrêt demandé au superviseur')

    def reconcile(self):
        for run in self.store.list():
            if run.get('source') != 'managed' or run['status'] not in ACTIVE:
                continue
            identity = alive(run.get('worker_pid'), run.get('worker_created', run.get('worker_create_time')))
            if identity is None:
                self.store.update(run['id'], recovery_warning='Identité du superviseur inaccessible ; ses ressources restent réservées.')
                continue
            if identity:
                continue
            if not run.get('worker_pid'):
                stamp = run.get('dispatch_at', run.get('created_at'))
                if stamp and (datetime.now(timezone.utc) - datetime.fromisoformat(stamp)).total_seconds() < 10:
                    continue
            self.store.update_if_status(run['id'], ACTIVE, status='interrupted', activity='Superviseur disparu. Le code de retour et l’heure de fin du calcul ne sont pas disponibles.', recovery_at=now(), finished_at=None, return_code=None)

    def _tick(self):
        with self._lock:
            self.reconcile()
            for pid, process in list(self._children.items()):
                if process.poll() is not None:
                    del self._children[pid]
            runs = self.store.list()
            active = [r for r in runs if r.get('source') == 'managed' and r['status'] in ACTIVE]
            held = {key for r in active for key in r.get('resource_keys', [])}
            for run in reversed(runs):
                if len(active) >= self.max_concurrency:
                    break
                if run.get('source') != 'managed' or run['status'] != 'queued':
                    continue
                if held.intersection(run.get('resource_keys', [])):
                    continue
                if self._is_primary(run):
                    try:
                        environment_error = self._primary_environment_error()
                        if environment_error:
                            raise ValueError(environment_error)
                        self.registry.validate_primary_sources(run)
                        selected_day = run.get('request', {}).get('delivery_day')
                        if selected_day is None:
                            defaults = self.registry.primary_defaults()
                            self.store.update(run['id'], delivery_day=defaults['delivery_day'])
                        else:
                            from .adapters import validate_primary_delivery_day
                            validate_primary_delivery_day(selected_day)
                            if run.get('delivery_day') != selected_day:
                                raise ValueError('La date du lancement diffère de la date sélectionnée. Préparez une nouvelle demande.')
                    except Exception as exc:
                        self.store.update(run['id'], status='failed', finished_at=now(), activity=str(redact(str(exc))))
                        continue
                self.store.update(run['id'], status='starting', dispatch_at=now(), activity='Démarrage du superviseur')
                try:
                    flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == 'nt' else 0
                    cmd = [str(self.python_executable), '-m', 'experiment_console.worker', '--database', str(self.store.path), '--run-id', run['id']]
                    process = subprocess.Popen(cmd, cwd=Path(__file__).resolve().parents[1], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False, creationflags=flags, start_new_session=os.name != 'nt', close_fds=True)
                    self._children[process.pid] = process
                    # The worker writes its own identity: venv launchers can have a different PID.
                    active.append(run)
                    held.update(run.get('resource_keys', []))
                except Exception as exc:
                    self.store.update(run['id'], status='failed', finished_at=now(), activity=str(redact(str(exc))))

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._tick()
                self.scheduler_error = None
            except Exception as exc:
                self.scheduler_error = str(redact(str(exc)))
            self._stop.wait(0.3)

    def close(self):
        with self._lock:
            self._stop.set()
            thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=5)
        if self._instance_lock.is_locked:
            self._instance_lock.release()

    def annotate(self, run_id, note, tags):
        if not isinstance(tags, list) or len(tags) > 30 or any(not isinstance(t, str) or len(t) > 60 for t in tags):
            raise ValueError('30 tags maximum, 60 caractères par tag.')
        return self.store.update(run_id, note=redact(str(note)[:20000]), tags=redact(list(dict.fromkeys(t.strip() for t in tags if t.strip()))))

    def import_history(self, root):
        from .artifacts import discover_runs
        path = Path(root).resolve()
        allowed = [self.project_root / 'runs', self.project_root / 'output']
        if not any(path.is_relative_to(p.resolve()) for p in allowed) or path.is_relative_to(self.state_root):
            raise ValueError('Sélectionnez un dossier sous runs/ ou output/, hors métadonnées de la console.')
        result = discover_runs(path)
        added = updated = 0
        with self._lock:
            known = {r.get('source_key'): r for r in self.store.list()}
            for candidate in result['candidates']:
                output = Path(candidate['output_dir']).resolve()
                if output.is_relative_to(self.state_root):
                    continue
                source_key = candidate.get('source_key', str(output))
                old = known.get(source_key)
                if old and old.get('source') == 'managed':
                    continue
                record = redact({**candidate, 'source_key': source_key})
                if old:
                    record.pop('id', None)
                    record.pop('note', None)
                    record.pop('tags', None)
                    self.store.update(old['id'], **record)
                    updated += 1
                else:
                    record.update(id='h-' + hashlib.sha256(str(source_key).casefold().encode()).hexdigest()[:24], imported_at=now(), note='', tags=[])
                    record.setdefault('source', 'historical')
                    self.store.insert(record)
                    known[source_key] = record
                    added += 1
        return {'added': added, 'updated': updated, 'warnings': result['warnings']}
