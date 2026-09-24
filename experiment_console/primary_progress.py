"""Read the exact primary batch announced by a managed run's own console."""
from __future__ import annotations

from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import re

from .primary_results import _read_file, _safe_path


MAX_CONSOLE_BYTES = 1024 * 1024
MAX_STATUS_BYTES = 256 * 1024
_BATCH_PATH = re.compile(r'runs/logs/nuclear_kalman/(\d{4}-\d{2}-\d{2})/(\d{8}T\d{6}_\d{6}Z_[0-9a-f]{8})/status\.json\Z')
_ANNOUNCEMENT = re.compile(r'^(?:Statut du batch|Journal et statut du batch) : (.+)$', re.MULTILINE)
_PLAN = [('sources', None), *[('nuclear_kalman', zone) for zone in ('BE', 'DE', 'FR', 'NL')], ('CWE_Model_Storm', None)]
_COUNTRIES = {'BE': 'Belgique', 'DE': 'Allemagne', 'FR': 'France', 'NL': 'Pays-Bas'}
_TERMINAL_STEPS = {'complete', 'failed', 'skipped', 'interrupted'}
_STEP_STATUSES = _TERMINAL_STEPS | {'pending', 'running'}
_ACTIVE = {'starting', 'running', 'cancelling'}
_PROCESS_PHASES = {
    'queued': 'En attente des ressources scientifiques',
    'starting': 'Démarrage de NuclearKalman.ps1',
    'running': 'Initialisation du lot', 'cancelling': 'Arrêt demandé',
    'succeeded': 'Calcul et publication terminés', 'failed': 'L’exécution a échoué',
    'cancelled': 'Exécution annulée', 'interrupted': 'Superviseur interrompu',
    'unknown': 'État non confirmé',
}


def _fallback(run, warning=None):
    result = {'phase': _PROCESS_PHASES.get(run.get('status'), 'État non confirmé'),
              'completed': None, 'total': None, 'percent': None,
              'indeterminate': run.get('status') in _ACTIVE, 'steps': [], 'source': 'process'}
    if warning:
        result['warning'] = warning
    return result


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError('timestamp')
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('timezone')
    return result.astimezone(timezone.utc)


def _console_text(root, run):
    directory = Path(run['run_dir'])
    if not directory.is_absolute() or directory.name != run['id']:
        raise ValueError('console identity')
    relative = (directory / 'console.log').relative_to(root).as_posix()
    path, before = _safe_path(root, relative)
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError('console replaced')
        half = MAX_CONSOLE_BYTES // 2
        head = stream.read(half)
        tail = b''
        if opened.st_size > half:
            offset = max(half, opened.st_size - half)
            stream.seek(offset)
            tail = stream.read(half)
            if offset > half:
                tail = tail.partition(b'\n')[2]
        _, after = _safe_path(root, relative)
        if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError('console replaced')
    # Keep separated excerpts from manufacturing an announcement across a gap.
    if opened.st_size > MAX_CONSOLE_BYTES:
        head = head.rpartition(b'\n')[0]
        head += b'\n'
    return (head + tail).decode('utf-8-sig', errors='replace')


def _batch_document(root, run):
    text = _console_text(root, run)
    announcements = {value.strip() for value in _ANNOUNCEMENT.findall(text)}
    if not announcements:
        return None
    if len(announcements) != 1:
        raise ValueError('multiple batches')
    path = Path(announcements.pop())
    if not path.is_absolute():
        raise ValueError('relative batch')
    relative = path.relative_to(root).as_posix()
    match = _BATCH_PATH.fullmatch(relative)
    if not match or date.fromisoformat(match[1]).isoformat() != match[1]:
        raise ValueError('batch path')
    _, content = _read_file(root, relative, MAX_STATUS_BYTES)
    batch = json.loads(content.decode('utf-8-sig'))
    if not isinstance(batch, dict) or batch.get('run_id') != match[2] or batch.get('delivery_day') != match[1]:
        raise ValueError('batch identity')
    if batch['delivery_day'] != run.get('delivery_day') or Path(batch.get('status_file', '')) != path:
        raise ValueError('run identity')
    started = _timestamp(batch.get('started_at_utc'))
    lower = _timestamp(run.get('started_at') or run.get('dispatch_at'))
    upper = _timestamp(run['finished_at']) if run.get('finished_at') else datetime.now(timezone.utc)
    if not lower <= started <= upper:
        raise ValueError('batch belongs to another execution')
    if batch.get('finished_at_utc'):
        if not started <= _timestamp(batch['finished_at_utc']) <= upper:
            raise ValueError('batch end outside execution')
    if batch.get('status') not in {'running', 'complete', 'failed', 'interrupted'}:
        raise ValueError('batch status')
    steps = batch.get('steps')
    if (not isinstance(steps, list) or len(steps) != len(_PLAN)
            or any(not isinstance(step, dict) for step in steps)
            or [(step.get('name'), step.get('zone')) for step in steps] != _PLAN
            or any(step.get('status') not in _STEP_STATUSES for step in steps)
            or sum(step['status'] == 'running' for step in steps) > 1):
        raise ValueError('batch plan')
    if batch['status'] == 'complete' and any(step['status'] != 'complete' for step in steps):
        raise ValueError('inconsistent completed batch')
    return batch


def read_primary_progress(project_root, run):
    """Return processed-step progress; never estimate time or consult latest_status."""
    if (run.get('source') != 'managed' or run.get('adapter_id') != 'primary_nuclear_kalman'
            or run.get('status') == 'queued'):
        return _fallback(run)
    try:
        batch = _batch_document(Path(project_root).resolve(), run)
    except FileNotFoundError:
        return _fallback(run)
    except (OSError, ValueError, TypeError, KeyError, UnicodeError):
        return _fallback(run, 'La progression détaillée de ce lancement ne peut pas être vérifiée.')
    if batch is None:
        return _fallback(run)
    steps = []
    for item in batch['steps']:
        name, zone = item['name'], item.get('zone')
        label = ('Synchronisation des sources' if name == 'sources' else
                 'Publication du rapport CWE' if name == 'CWE_Model_Storm' else
                 f'{_COUNTRIES[zone]} · Prévision et publication')
        steps.append({'name': name, 'zone': zone, 'status': item['status'], 'label': label})
    completed = sum(step['status'] in _TERMINAL_STEPS for step in steps)
    phase = next((step['label'] for step in steps if step['status'] == 'running'), 'Finalisation du lot')
    if batch['status'] != 'running':
        phase = {'complete': 'Calcul et publication terminés', 'failed': 'Lot terminé avec des échecs',
                 'interrupted': 'Lot interrompu'}[batch['status']]
    stale = batch['status'] == 'running' and run.get('status') not in _ACTIVE
    if stale or run.get('status') in {'cancelling', 'cancelled'}:
        phase = _PROCESS_PHASES.get(run.get('status'), 'État non confirmé')
    result = {'phase': phase, 'completed': completed, 'total': len(steps),
              'percent': round(completed * 100 / len(steps)), 'indeterminate': False,
              'steps': steps, 'source': 'batch_status', 'batch_status': batch['status']}
    if stale:
        result['warning'] = 'Le dernier statut du lot précède l’arrêt du processus.'
    return result
