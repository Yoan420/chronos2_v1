"""Opt-in SolarWind publications. Never scans or writes nuclear publications."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .primary_results import (
    MAX_EXPORT_BYTES, MAX_MANIFEST_BYTES, _directory_entries, _metadata,
    _read_file, _relative_parts, _valid_day,
)

PREFIX = 'solarwind_interaction40'
MODEL = 'solarwind_interaction40'
FILES = {'index.html', 'forecast_de.csv', 'forecast_nl.csv'}
ARTIFACT = re.compile(r'solarwind_interaction40/(\d{4}-\d{2}-\d{2})/(index\.html|forecast_(?:de|nl)\.csv)\Z')


def _publication(root, day):
    folder = f'runs/{PREFIX}/{day}'
    _, raw = _read_file(root, f'{folder}/manifest.json', MAX_MANIFEST_BYTES)
    manifest = json.loads(raw)
    if (not isinstance(manifest, dict) or manifest.get('model') != MODEL or manifest.get('delivery_day') != day
            or manifest.get('status') != 'COMPLETE' or not isinstance(manifest.get('files'), dict)
            or set(manifest['files']) != FILES):
        raise ValueError('Publication SolarWind incomplète ou identité invalide.')
    payloads = {}
    for name, expected in manifest['files'].items():
        target, content = _read_file(root, f'{folder}/{name}', MAX_EXPORT_BYTES)
        if not content or hashlib.sha256(content).hexdigest() != expected:
            raise ValueError('Publication SolarWind modifiée depuis sa validation.')
        payloads[name] = (target, content)
    return payloads


def build_solarwind_results(project_root):
    root = Path(project_root).resolve()
    warnings = []
    days = []
    entries = _directory_entries(root, f'runs/{PREFIX}', warnings)
    for day in sorted((name for name in entries if _valid_day(name)), reverse=True):
        try:
            _publication(root, day)
            relative = f'{PREFIX}/{day}'
            days.append({'date': day, 'model': MODEL,
                         'report': _metadata(root, relative + '/index.html'),
                         'zones': [{'zone': zone, 'csv': _metadata(root, f'{relative}/forecast_{zone.lower()}.csv')}
                                   for zone in ('DE', 'NL')]})
        except FileNotFoundError:
            continue
        except (OSError, ValueError, TypeError, AttributeError, RecursionError):
            warnings.append(f'{day} : publication SolarWind non vérifiée, rapport masqué.')
    return {'days': days, 'warnings': warnings}


def read_solarwind_artifact(project_root, relative, max_bytes=MAX_EXPORT_BYTES):
    _relative_parts(relative)
    match = ARTIFACT.fullmatch(relative)
    if not match or not _valid_day(match.group(1)):
        raise ValueError('Ce fichier ne fait pas partie des résultats SolarWind.')
    target, content = _publication(Path(project_root).resolve(), match.group(1))[match.group(2)]
    if len(content) > max_bytes:
        raise ValueError('Publication SolarWind trop volumineuse.')
    return target, content
