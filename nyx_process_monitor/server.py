"""Loopback-only HTTP viewer. No endpoint starts, stops or modifies a job."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import threading
import time
from urllib.parse import parse_qs, urlsplit

from .collector import Collector

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).parent / 'static'
APP = 'nyx-process-monitor'
BUILD = 'nyx-live-v2'


class SnapshotCache:
    """One sampler for every viewer, never one expensive process scan per request."""

    def __init__(self, collector, interval=1.0):
        self.collector = collector
        self.lock = threading.RLock()
        self.data = None
        self.interval = interval
        self.first = threading.Event()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, name='nyx-read-only-sampler', daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2)

    def _sample(self):
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                data = self.collector.snapshot()
                data['build'] = BUILD
                with self.lock:
                    self.data = data
                self.first.set()
            except Exception:
                # Keep the last timestamp unchanged: the client explicitly marks
                # old observations stale. Never fabricate current job activity.
                pass
            self.stop_event.wait(max(.05, self.interval - (time.monotonic() - started)))

    def snapshot(self):
        if not self.first.wait(timeout=8):
            raise ValueError('First observation not yet available')
        with self.lock:
            return self.data


def make_server(root=ROOT, port=8766, collector=None):
    if not 0 <= port <= 65535:
        raise ValueError('Invalid local port')
    root = Path(root).resolve()
    cache = SnapshotCache(collector or Collector(root))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, code, content, content_type='application/json; charset=utf-8', *, report=False):
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('X-Frame-Options', 'DENY')
            policy = ("sandbox allow-scripts; default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; font-src data:; connect-src 'none'; base-uri 'none'; form-action 'none'" if report else
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
            self.send_header('Content-Security-Policy', policy)
            self.end_headers()
            self.wfile.write(content)

        def error(self, code, message):
            self.send(code, json.dumps({'error': message}, ensure_ascii=False).encode('utf-8'))

        def do_GET(self):
            port = self.server.server_port
            hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
            origins = {f'http://{host}' for host in hosts}
            if (self.headers.get('Host') not in hosts
                    or self.headers.get('Origin') not in origins | {None}
                    or self.headers.get('Sec-Fetch-Site') == 'cross-site'):
                return self.error(403, 'Accès local uniquement')
            parsed = urlsplit(self.path)
            try:
                if parsed.path == '/api/health':
                    return self.send(200, json.dumps({'app': APP, 'build': BUILD, 'project_root': str(root), 'read_only': True}).encode())
                if parsed.path == '/api/state':
                    return self.send(200, json.dumps(cache.snapshot(), ensure_ascii=False, allow_nan=False).encode('utf-8'))
                if parsed.path == '/artifact':
                    query = parse_qs(parsed.query)
                    if set(query) != {'id'} or len(query['id']) != 1:
                        return self.error(400, 'Identifiant de rapport requis')
                    cache.snapshot()
                    target = cache.collector.artifact(query['id'][0])
                    if target is None:
                        return self.error(404, 'Rapport absent ou non autorisé')
                    target = Path(target)
                    if target.suffix.lower() != '.html' or target.stat().st_size > 32 * 1024 * 1024:
                        return self.error(413, 'Rapport non pris en charge')
                    return self.send(200, target.read_bytes(), 'text/html; charset=utf-8', report=True)
                static = {'/': ('index.html', 'text/html; charset=utf-8'),
                    '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                    '/style.css': ('style.css', 'text/css; charset=utf-8'),
                    '/static/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                    '/static/style.css': ('style.css', 'text/css; charset=utf-8')}
                if parsed.path in static:
                    name, mime = static[parsed.path]
                    return self.send(200, (STATIC / name).read_bytes(), mime)
                return self.error(404, 'Page inconnue')
            except (OSError, ValueError, KeyError, TypeError):
                # Do not leak paths, traces or raw unreadable files to the browser.
                return self.error(503, 'Lecture momentanément indisponible ; nouvel essai automatique')

        def do_POST(self):
            self.error(405, 'Moniteur en lecture seule')

        do_PUT = do_POST
        do_DELETE = do_POST
        do_PATCH = do_POST

    class LocalServer(ThreadingHTTPServer):
        allow_reuse_address = False

        def server_bind(self):
            if os.name == 'nt':
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            super().server_bind()

        def server_close(self):
            cache.close()
            super().server_close()

    server = LocalServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    cache.start()
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Use an unprivileged local port')
    server = make_server(port=args.port)
    print(f'NYX Process Monitor ready on http://127.0.0.1:{server.server_port}', flush=True)
    server.serve_forever(poll_interval=.5)


if __name__ == '__main__':
    main()
