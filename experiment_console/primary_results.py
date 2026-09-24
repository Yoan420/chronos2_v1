"""Read-only index of canonical NYX publications, independent of experiment state.

Verification attests the manifest identity and file integrity, not scientific
quality or completion of the latest computation. Legacy reports remain readable.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat


MAX_EXPORT_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_DIRECTORY_ENTRIES = 10_000
ZONES = {'BE': 'Belgique', 'DE': 'Allemagne', 'FR': 'France', 'NL': 'Pays-Bas'}
_DAY = r'\d{4}-\d{2}-\d{2}'
_CWE = re.compile(rf'reports/model_storm/CWE_Model_Storm_({_DAY})\.html\Z')
_ZONE = re.compile(rf'exports/({_DAY})/(be|de|fr|nl)/nuclear_kalman/forecast_\2_\1_nuclear_kalman\.(html|csv)\Z')


def _valid_day(value):
    try:
        return bool(re.fullmatch(_DAY, value)) and date.fromisoformat(value).isoformat() == value
    except (TypeError, ValueError):
        return False


def _relative_parts(relative):
    if not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative or '\x00' in relative:
        raise ValueError('Chemin de publication non autorisé.')
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or '..' in parsed.parts or relative != parsed.as_posix():
        raise ValueError('Chemin de publication non autorisé.')
    return parsed.parts


def _safe_path(root, relative, *, directory=False):
    """Check every lexical parent before resolution, including Windows junctions."""
    parts = _relative_parts(relative)
    current = Path(root).resolve()
    for index, part in enumerate(parts):
        current = current / part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
            raise ValueError('Les liens et jonctions ne sont pas des publications consultables.')
        expects_directory = index < len(parts) - 1 or directory
        if expects_directory and not stat.S_ISDIR(info.st_mode):
            raise ValueError('Dossier de publication absent.')
        if not expects_directory and not stat.S_ISREG(info.st_mode):
            raise ValueError('Fichier de publication absent.')
    if not current.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Chemin de publication non autorisé.')
    return current, info


def _signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _read_file(root, relative, max_bytes):
    target, before = _safe_path(root, relative)
    if before.st_size > max_bytes:
        raise ValueError('Publication trop volumineuse pour cet aperçu.')
    with target.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        _, current = _safe_path(root, relative)
        if _signature(opened) != _signature(before) or _signature(current) != _signature(before):
            raise ValueError('Publication modifiée pendant la lecture. Actualisez la page.')
        content = stream.read(max_bytes + 1)
        _, after = _safe_path(root, relative)
        if _signature(os.fstat(stream.fileno())) != _signature(before) or _signature(after) != _signature(before):
            raise ValueError('Publication modifiée pendant la lecture. Actualisez la page.')
    if len(content) > max_bytes:
        raise ValueError('Publication trop volumineuse pour cet aperçu.')
    return target, content


def read_primary_artifact(project_root, relative, max_bytes=MAX_EXPORT_BYTES):
    """Read only a currently present canonical HTML/CSV, never a cached path."""
    _relative_parts(relative)
    match = _CWE.fullmatch(relative) or _ZONE.fullmatch(relative)
    if match is None or not _valid_day(match.group(1)):
        raise ValueError('Ce fichier ne fait pas partie des résultats principaux.')
    return _read_file(Path(project_root).resolve(), 'runs/' + relative, max_bytes)


def _metadata(project_root, relative):
    target, info = _safe_path(project_root, 'runs/' + relative)
    return {'path': relative, 'name': target.name, 'size': info.st_size,
            'updated_at': datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat()}


def _optional_metadata(root, relative, warnings, label):
    try:
        result = _metadata(root, relative)
        if result['size'] == 0:
            warnings.append(f'{label} vide.')
        return result
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        warnings.append(f'{label} inaccessible ou chemin non autorisé.')
        return None


def _directory_entries(root, relative, warnings):
    try:
        folder, _ = _safe_path(root, relative, directory=True)
        result = []
        with os.scandir(folder) as entries:
            for index, entry in enumerate(entries):
                if index >= MAX_DIRECTORY_ENTRIES:
                    warnings.append('Liste des publications tronquée : trop d’entrées dans un dossier.')
                    break
                result.append(entry.name)
        return result
    except FileNotFoundError:
        return []
    except (OSError, ValueError):
        warnings.append('Un dossier de publications est inaccessible ou utilise un lien non autorisé.')
        return []


def _verify_manifest(root, day, zone, report, csv_file, warnings):
    folder = f'exports/{day}/{zone.lower()}'
    manifest_relative = f'runs/{folder}/current_nuclear_batch_manifest.json'
    complete_files = bool(report and csv_file and report['size'] and csv_file['size'])
    if not report:
        warnings.append('Rapport HTML principal absent.')
    if not csv_file:
        warnings.append('Prévision CSV principale absente.')
    try:
        _, content = _read_file(root, manifest_relative, MAX_MANIFEST_BYTES)
    except FileNotFoundError:
        warnings.append('Manifeste de publication absent : intégrité non vérifiée.')
        return 'unverified' if complete_files else 'incomplete'
    except (OSError, ValueError):
        warnings.append('Manifeste inaccessible ou incomplet : publication non vérifiée.')
        return 'incomplete'
    try:
        document = json.loads(content.decode('utf-8-sig'))
        if not isinstance(document, dict) or document.get('zone') != zone or document.get('delivery_day') != day:
            raise ValueError('identity')
        records = document.get('exports')
        if not isinstance(records, list):
            raise ValueError('records')
        matches = [record for record in records if isinstance(record, dict) and record.get('variant') == 'nuclear_kalman']
        if len(matches) != 1:
            raise ValueError('variant')
        record = matches[0]
        if record.get('zone', zone) != zone:
            raise ValueError('record zone')
        files = record.get('files')
        if not isinstance(files, list) or not 1 <= len(files) <= 3:
            raise ValueError('files')
        prefix = f'nuclear_kalman/forecast_{zone.lower()}_{day}_nuclear_kalman'
        required = {prefix + '.html', prefix + '.csv'}
        permitted = required | {'nuclear_kalman/nuclear_report_audit.json'}
        verified = set()
        for item in files:
            if not isinstance(item, dict):
                raise ValueError('file')
            relative, digest = item.get('path'), item.get('sha256')
            if relative not in permitted or relative in verified or not isinstance(digest, str) or not re.fullmatch('[a-fA-F0-9]{64}', digest):
                raise ValueError('path or digest')
            _, data = _read_file(root, f'runs/{folder}/{relative}', MAX_EXPORT_BYTES)
            if not data or hashlib.sha256(data).hexdigest() != digest.lower():
                raise ValueError('integrity')
            verified.add(relative)
        if not required.issubset(verified) or not complete_files:
            raise ValueError('incomplete')
    except (OSError, ValueError, TypeError, RecursionError):
        warnings.append('Publication non vérifiée : manifeste incohérent, fichier modifié ou incomplet. Actualisez après la publication.')
        return 'incomplete'
    return 'verified'


def build_primary_results(project_root):
    """Discover immediate canonical publications without reading or writing SQLite."""
    root = Path(project_root).resolve()
    warnings, days = [], {}
    for filename in _directory_entries(root, 'runs/reports/model_storm', warnings):
        relative = 'reports/model_storm/' + filename
        match = _CWE.fullmatch(relative)
        if match is None or not _valid_day(match.group(1)):
            continue
        item = _optional_metadata(root, relative, warnings, 'Rapport CWE')
        if item:
            day = match.group(1)
            days.setdefault(day, {'date': day, 'cwe': None, 'zones': []})['cwe'] = item
    for day in _directory_entries(root, 'runs/exports', warnings):
        if not _valid_day(day):
            continue
        try:
            _safe_path(root, f'runs/exports/{day}', directory=True)
        except (OSError, ValueError):
            warnings.append(f'{day} : dossier de publication inaccessible ou lié.')
            continue
        for zone, label in ZONES.items():
            folder = f'exports/{day}/{zone.lower()}/nuclear_kalman'
            try:
                _safe_path(root, 'runs/' + folder, directory=True)
            except FileNotFoundError:
                continue
            except (OSError, ValueError):
                warnings.append(f'{day} {zone} : dossier de publication inaccessible ou lié.')
                continue
            zone_warnings = []
            base = f'{folder}/forecast_{zone.lower()}_{day}_nuclear_kalman'
            report = _optional_metadata(root, base + '.html', zone_warnings, 'Rapport HTML')
            csv_file = _optional_metadata(root, base + '.csv', zone_warnings, 'Prévision CSV')
            if not report and not csv_file:
                try:
                    _safe_path(root, f'runs/exports/{day}/{zone.lower()}/current_nuclear_batch_manifest.json')
                except (OSError, ValueError):
                    # A folder containing only incidental experiments is not a
                    # primary publication. A real manifest can attest an attempt.
                    continue
            publication_status = _verify_manifest(root, day, zone, report, csv_file, zone_warnings)
            item = {'zone': zone, 'label': label, 'report': report, 'csv': csv_file,
                    'publication_status': publication_status, 'warnings': zone_warnings}
            days.setdefault(day, {'date': day, 'cwe': None, 'zones': []})['zones'].append(item)
    return {'root': 'runs', 'model_name': 'NYX', 'days': [days[key] for key in sorted(days, reverse=True)],
            'warnings': list(dict.fromkeys(warnings))}
