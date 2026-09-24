"""Single loopback HTTP service serving a buildless UI and durable queue API."""
from __future__ import annotations

import argparse
import csv
import gzip
from functools import lru_cache
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
from pathlib import Path
import re
import secrets
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit
import webbrowser

import yaml

from .artifacts import inspect_run, list_artifacts, compare_scopes, read_forecast
from .manager import Manager, PrimaryRunConflict
from .primary_results import build_primary_results, read_primary_artifact
from .security import MASK, is_secret_key, redact, redact_text

STATIC = Path(__file__).parent / 'static'
MAX_EXPORT_BYTES = 64 * 1024 * 1024


def sanitized_text(value):
    # An external log may end halfway through a private-key block while its
    # process is still writing. Never expose the unfinished continuation.
    value = re.sub(r'(?is)-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----.*?(?:-----END (?:[A-Z0-9]+ )?PRIVATE KEY-----|\Z)', MASK, value)
    return redact_text(value)


@lru_cache(maxsize=1)
def trusted_plotly_script():
    """Trust only the exact installed vendor bytes, never a name or HTML tag."""
    from plotly.offline import get_plotlyjs
    return get_plotlyjs()


@lru_cache(maxsize=1)
def trusted_plotly_windows_script():
    # Windows text-mode writes perform precisely this newline conversion. No
    # whitespace trimming, JavaScript rewriting or arbitrary variants are trusted.
    return trusted_plotly_script().replace('\n', '\r\n')


class _ScriptSpans(HTMLParser):
    """Locate script content without reserializing the surrounding document."""
    def __init__(self, source):
        super().__init__(convert_charrefs=False)
        self.source = source
        self.line_offsets = [0] + [match.end() for match in re.finditer('\n', source)]
        self.current = None
        self.spans = []

    def position(self):
        line, column = self.getpos()
        return self.line_offsets[line - 1] + column

    def handle_starttag(self, tag, attrs):
        if tag == 'script' and self.current is None:
            start = self.position()
            self.current = (start, start + len(self.get_starttag_text()), dict(attrs))

    def handle_startendtag(self, tag, attrs):
        if tag == 'script':
            # In text/html a self-closing flag does not terminate a script.
            self.handle_starttag(tag, attrs)
            self.set_cdata_mode(tag)

    def handle_endtag(self, tag):
        if tag == 'script' and self.current is not None:
            content_end = self.position()
            end = self.source.find('>', content_end) + 1
            start, content_start, attrs = self.current
            self.spans.append((start, content_start, content_end, end, attrs))
            self.current = None

    def finish(self):
        self.feed(self.source)
        self.close()
        if self.current is not None:
            start, content_start, attrs = self.current
            self.spans.append((start, content_start, len(self.source), len(self.source), attrs))
        return self.spans


def sanitized_html(value):
    pieces, cursor, removed = [], 0, 0
    for start, content_start, content_end, end, attrs in _ScriptSpans(value).finish():
        pieces.append(sanitized_text(value[cursor:start]))
        content = value[content_start:content_end]
        script_type = (attrs.get('type') or '').split(';')[0].strip().lower()
        safe = True
        if content in (trusted_plotly_script(), trusted_plotly_windows_script()):
            # Redacting minified vendor identifiers would corrupt executable JS.
            # Exact identity proves this block contains only installed library code.
            rendered = content
        elif script_type in {'application/json', 'application/ld+json'}:
            try:
                rendered = json.dumps(redact(json.loads(content)), ensure_ascii=False)
                # JSON remains valid without allowing a string to close its tag.
                rendered = (rendered.replace('<', '\\u003c').replace('>', '\\u003e')
                            .replace('&', '\\u0026').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029'))
            except (ValueError, RecursionError):
                safe = False
        else:
            rendered = sanitized_text(content)
            safe = rendered == content
        opening = value[start:content_start]
        safe = safe and sanitized_text(opening) == opening
        if safe:
            pieces.extend((opening, rendered, value[content_end:end]))
        else:
            # Removing the full block is explicit and avoids broken executable
            # code or partially exposed credentials in an otherwise opaque script.
            removed += 1
        cursor = end
    pieces.append(sanitized_text(value[cursor:]))
    result = ''.join(pieces)
    if removed:
        warning = ('<div role="note" style="padding:12px;background:#493744;color:#ffdfaa">'
                   f'La console a désactivé {removed} script(s) du rapport : contenu sensible ou données JSON illisibles. '
                   'Les graphiques concernés peuvent être indisponibles. Le fichier original reste inchangé.</div>')
        closing = re.search(r'(?i)</body\s*>', result)
        result = result[:closing.start()] + warning + result[closing.start():] if closing else result + warning
    return result


def sanitized_artifact_text(value, suffix):
    if suffix in {'.html', '.htm'}:
        return sanitized_html(value)
    if suffix in {'.json', '.yaml', '.yml'}:
        try:
            parsed = json.loads(value) if suffix == '.json' else yaml.safe_load(value)
            parsed = redact(parsed)
            value = (json.dumps(parsed, indent=2, ensure_ascii=False) if suffix == '.json'
                     else yaml.safe_dump(parsed, allow_unicode=True, sort_keys=False))
        except (ValueError, yaml.YAMLError, RecursionError) as exc:
            # Textual fallback cannot safely identify multiline YAML secrets.
            raise ValueError('Configuration illisible ou incomplète ; export désactivé.') from exc
    elif suffix == '.csv':
        try:
            reader = csv.reader(io.StringIO(value, newline=''), strict=True)
            header = next(reader, [])
            secret_columns = {index for index, name in enumerate(header) if is_secret_key(name)}
            if secret_columns:
                output = io.StringIO(newline='')
                writer = csv.writer(output)
                writer.writerow(header)
                for row in reader:
                    writer.writerow([MASK if index in secret_columns else sanitized_text(cell)
                                     for index, cell in enumerate(row)])
                value = output.getvalue()
        except (csv.Error, ValueError) as exc:
            raise ValueError('CSV illisible ou incomplet ; export désactivé.') from exc
    return sanitized_text(value)


def themed_cwe_html(value):
    """Apply NYX's display palette without touching a publication on disk."""
    # Only the documented chart-data figure is restyled. Traces, values, axes
    # ranges and the original interactive JavaScript are preserved.
    for _, first, last, _, attrs in reversed(_ScriptSpans(value).finish()):
        if attrs.get('id') != 'chart-data' or attrs.get('type') != 'application/json':
            continue
        try:
            figure = json.loads(value[first:last])
        except ValueError:
            continue
        if not isinstance(figure, dict) or not isinstance(figure.get('layout'), dict):
            continue
        layout = figure['layout']
        layout.update(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        if not isinstance(layout.get('font'), dict):
            layout['font'] = {}
        layout['font']['color'] = '#e7e4dc'
        layout['font']['family'] = 'Inter, Segoe UI, sans-serif'
        for key, axis in layout.items():
            if re.fullmatch(r'[xy]axis\d*', key) and isinstance(axis, dict):
                axis.update(gridcolor='#33343d', zerolinecolor='#66574f', linecolor='#66574f', color='#aaa5ac')
                if not isinstance(axis.get('tickfont'), dict):
                    axis['tickfont'] = {}
                axis['tickfont']['color'] = '#aaa5ac'
                if isinstance(axis.get('title'), dict):
                    if not isinstance(axis['title'].get('font'), dict):
                        axis['title']['font'] = {}
                    axis['title']['font']['color'] = '#aaa5ac'
        annotations = layout.get('annotations', [])
        for annotation in annotations if isinstance(annotations, list) else []:
            if isinstance(annotation, dict):
                if not isinstance(annotation.get('font'), dict):
                    annotation['font'] = {}
                annotation['font']['color'] = '#e7e4dc'
        if isinstance(layout.get('legend'), dict):
            layout['legend'].update(bgcolor='rgba(0,0,0,0)', bordercolor='#33343d')
            if not isinstance(layout['legend'].get('font'), dict):
                layout['legend']['font'] = {}
            layout['legend']['font']['color'] = '#e7e4dc'
        rendered = json.dumps(figure, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
        value = value[:first] + rendered + value[last:]
    style = '<style id="nyx-cwe-theme">' + (STATIC / 'cwe-theme.css').read_text(encoding='utf-8') + '</style>'
    style += '<script id="nyx-cwe-navigation">' + (STATIC / 'cwe-navigation.js').read_text(encoding='utf-8') + '</script>'
    closing = re.search(r'(?i)</head\s*>', value)
    return value[:closing.start()] + style + value[closing.start():] if closing else style + value


def detail(manager, run_id):
    stored = manager.get(run_id)
    current = inspect_run(stored['output_dir'])
    if stored.get('source') == 'managed':
        result = {**current, **stored}
        result['metrics'] = current['metrics']
        result['scope'] = current['scope']
        result['artifacts'] = current['artifacts']
        current_warnings = [w for w in current.get('warnings', []) if not w.startswith(('Fin d\'exécution non attestée', 'Format de run non reconnu'))]
        result['warnings'] = list(dict.fromkeys(stored.get('warnings', []) + current_warnings))
    else:
        result = {**stored, **current, 'id': stored['id'], 'note': stored.get('note', ''), 'tags': stored.get('tags', [])}
    return redact(result)


def artifact_path(run, relative):
    root = Path(run['output_dir']).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise ValueError('Artefact absent ou chemin non autorisé.')
    if path not in [(root / item['path']).resolve() for item in list_artifacts(root)]:
        raise ValueError('Ce fichier ne fait pas partie des artefacts consultables.')
    return path


class ConsoleHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, manager):
        self.manager = manager
        self.token = secrets.token_urlsafe(32)
        self.import_state = {'running': False, 'result': None, 'error': None}
        self.import_lock = threading.Lock()
        self.external_refreshed = 0.0
        self.external_lock = threading.Lock()
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = 'ChronosConsole/1'

    def log_message(self, *_):
        pass

    def send_bytes(self, body, content_type, *, status=200, attachment=None, report=False):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        self.send_header('Content-Security-Policy', "sandbox allow-scripts; default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; font-src data:" if report else "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'self'")
        if attachment:
            safe = ''.join(c for c in attachment if c.isascii() and (c.isalnum() or c in '._-')) or 'export.txt'
            self.send_header('Content-Disposition', f'attachment; filename="{safe}"')
        self.end_headers()
        self.wfile.write(body)

    def json(self, value, status=200, *, sanitize=True):
        self.send_bytes(json.dumps(redact(value) if sanitize else value, ensure_ascii=False, allow_nan=False, default=str).encode('utf-8'), 'application/json; charset=utf-8', status=status)

    def boundary(self, mutation=False):
        port = self.server.server_port
        hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
        if self.headers.get('Host') not in hosts:
            raise PermissionError('Hôte non autorisé : ouvrez l’adresse locale affichée au démarrage.')
        origin = self.headers.get('Origin')
        if origin and origin not in {f'http://{host}' for host in hosts}:
            raise PermissionError('Origine non autorisée.')
        if mutation:
            if not secrets.compare_digest(self.headers.get('X-Console-Token', ''), self.server.token):
                raise PermissionError('Session expirée. Actualisez la page avant de réessayer.')
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                raise ValueError('Un corps JSON est requis.')

    def do_GET(self):
        try:
            self.boundary()
            self.get_route()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except PermissionError as exc:
            self.json({'error': str(exc)}, 403)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            self.json({'error': str(exc)}, 404)
        except Exception as exc:
            self.json({'error': f'{type(exc).__name__}: {exc}'}, 500)

    def get_route(self):
        parsed = urlsplit(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        manager = self.server.manager
        if path == '/api/bootstrap':
            bootstrap = redact({'catalog': manager.registry.catalog(), 'python_executable': str(manager.python_executable), 'project_root': str(manager.project_root), 'state_root': str(manager.state_root), 'max_concurrency': manager.max_concurrency, 'import_roots': [str(manager.project_root / p) for p in ('runs', 'output')]})
            # This generated CSRF nonce is intentionally returned to this local
            # browser; pipeline credentials above still pass through redaction.
            bootstrap['token'] = self.server.token
            self.json(bootstrap, sanitize=False)
        elif path == '/api/primary-results':
            self.json(build_primary_results(manager.project_root))
        elif path == '/api/primary-run':
            self.json(manager.primary_run_status())
        elif path == '/api/primary-artifact':
            if len(query.get('path', [])) != 1:
                raise ValueError('Un chemin de résultat principal est requis.')
            target, body = read_primary_artifact(manager.project_root, query['path'][0], MAX_EXPORT_BYTES)
            suffix = target.suffix.lower()
            rendered = sanitized_artifact_text(body.decode('utf-8-sig', errors='replace'), suffix)
            if suffix == '.html' and query.get('download') != ['1'] and query['path'][0].startswith('reports/model_storm/'):
                rendered = themed_cwe_html(rendered)
            body = rendered.encode('utf-8')
            if len(body) > MAX_EXPORT_BYTES:
                raise ValueError('Le résultat assaini dépasse la limite de 64 Mo.')
            content_type = 'text/html' if suffix == '.html' else 'text/csv'
            self.send_bytes(body, content_type + '; charset=utf-8',
                            attachment=target.name if query.get('download') == ['1'] else None,
                            report=suffix == '.html')
        elif path == '/api/runs':
            if time.monotonic() - self.server.external_refreshed > 5 and self.server.external_lock.acquire(blocking=False):
                try:
                    for external in manager.list_runs():
                        if external.get('source') == 'external':
                            current = inspect_run(external['output_dir'], include_artifacts=False)
                            manager.store.update(external['id'], **{k: current.get(k) for k in ('status', 'reported_status', 'activity', 'started_at', 'finished_at', 'duration_seconds', 'return_code', 'warnings')})
                    self.server.external_refreshed = time.monotonic()
                finally:
                    self.server.external_lock.release()
            fields = {'id', 'name', 'description', 'status', 'reported_status', 'source', 'type', 'model', 'created_at', 'started_at', 'finished_at', 'duration_seconds', 'activity', 'tags', 'note', 'metrics', 'output_dir', 'warnings', 'recovery_warning', 'imported_at'}
            rows = []
            for run in manager.list_runs():
                if run.get('source') == 'managed' and run['status'] == 'succeeded' and 'metrics' not in run:
                    metrics = inspect_run(run['output_dir'])['metrics']
                    run = manager.store.update(run['id'], metrics=metrics)
                rows.append({k: v for k, v in run.items() if k in fields})
            self.json({'runs': rows, 'scheduler_error': manager.scheduler_error, 'import': self.server.import_state})
        elif path == '/api/import-status':
            self.json(self.server.import_state)
        elif path == '/api/architecture':
            from .architecture import build_architecture
            self.json(build_architecture(manager.project_root))
        elif path == '/api/compare':
            ids = query.get('id', [])
            if not 2 <= len(ids) <= 8:
                raise ValueError('Sélectionnez entre 2 et 8 runs.')
            runs = [detail(manager, run_id) for run_id in ids]
            self.json({'runs': runs, **compare_scopes(runs)})
        elif path.startswith('/api/runs/'):
            parts = path.strip('/').split('/')
            run = manager.get(parts[2])
            action = parts[3] if len(parts) > 3 else ''
            if not action:
                self.json(detail(manager, run['id']))
            elif action == 'logs':
                logs = (run.get('log_paths', []) if run.get('source') == 'managed'
                        else inspect_run(run['output_dir']).get('log_paths', []))
                contents = []
                max_bytes = 512_000
                truncated = False
                for log in logs[:10]:
                    log_path = Path(log).resolve()
                    roots = [Path(run['output_dir']).resolve()]
                    if run.get('run_dir'):
                        roots.append(Path(run['run_dir']).resolve())
                    if not any(log_path.is_relative_to(root) for root in roots) or not log_path.is_file():
                        continue
                    clean = self.sanitized_log(log_path)
                    if query.get('download') != ['1']:
                        encoded = clean.encode('utf-8')
                        if len(encoded) > max_bytes:
                            truncated = True
                            clean = encoded[-max_bytes:].decode('utf-8', errors='replace')
                            clean = clean.split('\n', 1)[-1]
                    contents.append(clean)
                text = '\n'.join(contents)
                if query.get('download') == ['1']:
                    self.send_bytes(text.encode(), 'text/plain; charset=utf-8', attachment='console.log')
                else:
                    self.json({'text': text, 'truncated': truncated, 'available': bool(contents)})
            elif action == 'artifact':
                target = artifact_path(run, query.get('path', [''])[0])
                suffix = target.suffix.lower()
                if target.stat().st_size > MAX_EXPORT_BYTES:
                    raise ValueError('Fichier supérieur à 64 Mo : consultez-le dans le dossier de sortie affiché.')
                with target.open('rb') as stream:
                    body = stream.read(MAX_EXPORT_BYTES + 1)
                if len(body) > MAX_EXPORT_BYTES:
                    raise ValueError('Fichier supérieur à 64 Mo : consultez-le dans le dossier de sortie affiché.')
                content_type = mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
                if suffix in {'.json', '.yaml', '.yml', '.txt', '.log', '.csv', '.html', '.htm', '.svg'}:
                    decoded = body.decode('utf-8-sig', errors='replace')
                    body = sanitized_artifact_text(decoded, suffix).encode('utf-8')
                    content_type += '; charset=utf-8'
                elif suffix == '.gz':
                    # Only textual CSV gzip is in the artifact allowlist.
                    with gzip.open(target, 'rb') as stream:
                        decoded = stream.read(MAX_EXPORT_BYTES + 1)
                    if len(decoded) > MAX_EXPORT_BYTES:
                        raise ValueError('CSV décompressé supérieur à 64 Mo. Consultez le fichier sur disque.')
                    body = gzip.compress(sanitized_artifact_text(decoded.decode('utf-8-sig', errors='replace'), '.csv').encode('utf-8'))
                self.send_bytes(body, content_type, attachment=target.name if query.get('download') == ['1'] else None, report=suffix in {'.html', '.htm', '.svg'})
            elif action == 'forecast':
                target = artifact_path(run, query.get('path', [''])[0])
                self.json(read_forecast(target, scope=detail(manager, run['id']).get('scope')))
            else:
                raise ValueError('Route inconnue.')
        else:
            assets = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css', '/favicon.svg': 'favicon.svg', '/architecture.js': 'architecture.js', '/architecture.css': 'architecture.css'}
            if path == '/plotly.js':
                from plotly.offline import get_plotlyjs
                self.send_bytes(get_plotlyjs().encode('utf-8'), 'text/javascript; charset=utf-8')
                return
            if path not in assets:
                raise ValueError('Page inconnue.')
            target = STATIC / assets[path]
            self.send_bytes(target.read_bytes(), (mimetypes.guess_type(target.name)[0] or 'text/plain') + '; charset=utf-8')

    @staticmethod
    def sanitized_log(path):
        if path.stat().st_size > MAX_EXPORT_BYTES:
            raise ValueError('Log supérieur à 64 Mo : consultez le fichier sur disque.')
        with path.open('rb') as stream:
            content = stream.read(MAX_EXPORT_BYTES + 1)
        if len(content) > MAX_EXPORT_BYTES:
            raise ValueError('Log supérieur à 64 Mo : consultez le fichier sur disque.')
        return sanitized_text(content.decode('utf-8', errors='replace'))

    def do_POST(self):
        try:
            self.boundary(mutation=True)
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 128_000:
                raise ValueError('Corps JSON absent ou trop volumineux (128 ko maximum).')
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError('Un objet JSON est requis.')
            manager = self.server.manager
            if self.path == '/api/primary-run':
                if set(body) - {'delivery_day'}:
                    raise ValueError('Seule la date de livraison peut être choisie pour un lancement NYX.')
                delivery_day = None
                if 'delivery_day' in body:
                    from .adapters import validate_primary_delivery_day
                    delivery_day = validate_primary_delivery_day(body['delivery_day'])
                self.json(manager.launch_primary_run(self.headers.get('Idempotency-Key'), delivery_day=delivery_day), 202)
            elif self.path == '/api/preview':
                self.json(manager.preview(body))
            elif self.path == '/api/launch':
                self.json(manager.launch(body['plan_id'], body['idempotency_key']))
            elif self.path == '/api/import':
                with self.server.import_lock:
                    if self.server.import_state['running']:
                        raise ValueError('Un import est déjà en cours.')
                    self.server.import_state = {'running': True, 'result': None, 'error': None}
                    def perform():
                        try:
                            self.server.import_state['result'] = manager.import_history(body['root'])
                        except Exception as exc:
                            self.server.import_state['error'] = str(redact(str(exc)))
                        finally:
                            self.server.import_state['running'] = False
                    threading.Thread(target=perform, daemon=True).start()
                self.json(self.server.import_state, 202)
            elif self.path.startswith('/api/runs/'):
                parts = self.path.strip('/').split('/')
                run_id, action = parts[2:4]
                if action == 'cancel':
                    if body.get('confirmed') is not True:
                        raise ValueError('Confirmez l’arrêt de ce run et de ses processus enfants.')
                    self.json(manager.cancel(run_id))
                elif action == 'annotate':
                    self.json(manager.annotate(run_id, body.get('note', ''), body.get('tags', [])))
                elif action == 'duplicate':
                    run = manager.get(run_id)
                    if run.get('request'):
                        request = dict(run['request'])
                        request['name'] = f"{run.get('name', 'Run')} · copie"
                        request['duplicate_of'] = run['id']
                        self.json({'request': request, 'warnings': ['La configuration conservée sera copiée dans un nouveau dossier.']})
                    else:
                        self.json({'request': {'adapter_id': 'hourly_report', 'config_id': 'local-results', 'model': 'auto', 'name': f"{run.get('name', 'Archive')} · rapport", 'parameters': {'source_run': run['output_dir']}}, 'warnings': ['La configuration historique effective est inconnue : un nouveau rapport est préparé depuis les résultats existants.']})
                else:
                    raise ValueError('Action inconnue.')
            else:
                raise ValueError('Route inconnue.')
        except PermissionError as exc:
            self.json({'error': str(exc)}, 403)
        except PrimaryRunConflict as exc:
            self.json({'error': str(exc)}, 409)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            self.json({'error': str(exc)}, 400)
        except Exception as exc:
            self.json({'error': f'{type(exc).__name__}: {exc}'}, 500)


def announce(message):
    """Console output is optional under pythonw and detached desktop startup."""
    if sys.stdout is not None:
        try:
            print(message, file=sys.stdout, flush=True)
        except (OSError, ValueError):
            pass


def main():
    parser = argparse.ArgumentParser(description='Console locale des expériences Chronos')
    parser.add_argument('--settings', default=str(Path(__file__).resolve().parents[1] / 'config' / 'experiment_console.json'))
    parser.add_argument('--port', type=int)
    parser.add_argument('--open', action='store_true', help='Ouvrir le navigateur après la création du serveur local.')
    args = parser.parse_args()
    settings_path = Path(args.settings).resolve()
    settings = json.loads(settings_path.read_text(encoding='utf-8-sig'))
    root = Path(__file__).resolve().parents[1]
    state = Path(settings.get('state_root', 'runs/.experiment_console'))
    if not state.is_absolute():
        state = root / state
    manager = server = None
    port = args.port if args.port is not None else settings.get('port', 8765)
    try:
        manager = Manager(root, state, settings['python_executable'], settings.get('max_concurrency', 1), start_scheduler=False)
        server = ConsoleHTTPServer(('127.0.0.1', port), manager)
        # A port conflict must not resume queued computations in an invisible,
        # unusable backend. Socket binding succeeds before dispatch is enabled.
        manager.start()
        announce(f'Console Chronos : http://127.0.0.1:{server.server_port}')
        announce('Ctrl+C ferme la console. Les calculs démarrés continuent ; la file reprendra au prochain démarrage.')
        if args.open:
            webbrowser.open(f'http://127.0.0.1:{server.server_port}')
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if server is not None:
                server.server_close()
        finally:
            if manager is not None:
                manager.close()


if __name__ == '__main__':
    main()
