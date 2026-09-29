"""Idempotent foreground launcher and verified idle restart for local NYX."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener
import webbrowser

from filelock import FileLock, Timeout
import psutil

from .desktop import BackendBusy, DesktopError, DesktopSettings, _NoRedirect, backend_ready

IDLE_STATUSES = {'succeeded', 'failed', 'cancelled', 'interrupted'}
MAX_JSON_BYTES = 8 * 1024 * 1024


class BackendLocked(DesktopError):
    """The backend owns its state, possibly before binding its socket."""


def read_json(settings, path, timeout=2):
    """Read bounded local JSON without proxies or redirects."""
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(settings.url + path, timeout=timeout) as response:
            payload = response.read(MAX_JSON_BYTES + 1)
        if len(payload) > MAX_JSON_BYTES:
            raise DesktopError('La réponse de NYX est trop volumineuse pour vérifier son arrêt.')
        return json.loads(payload)
    except (URLError, OSError, ValueError, UnicodeError) as exc:
        raise DesktopError('Impossible de vérifier le service NYX ; aucun processus n’a été arrêté.') from exc


def _same_path(value, expected):
    return isinstance(value, str) and Path(value).resolve() == Path(expected).resolve()


def _process_identity(settings, process, created):
    """Confirm listener identity and the exact configured Windows venv parent."""
    if (not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
            or process.create_time() != created):
        raise DesktopError('L’identité du service NYX a changé ; arrêt refusé.')
    command = process.cmdline()
    configured = {Path(settings.python).resolve(), Path(settings.pythonw).resolve()}
    if (len(command) < 3 or command[1] != '-m'
            or command[2] not in {'experiment_console.server', 'experiment_console.launcher'}
            or Path(process.cwd()).resolve() != Path(settings.root).resolve()):
        raise DesktopError('Le processus présent ne correspond pas au backend NYX configuré ; arrêt refusé.')
    executable = Path(process.exe()).resolve()
    command_executable = Path(command[0]).resolve()
    direct = executable in configured and command_executable in configured
    if not direct:
        # A Windows venv redirector may spawn its base Python. Its live parent
        # must prove the selected environment; a base executable alone cannot.
        base = Path(getattr(sys, '_base_executable', sys.executable)).resolve()
        base_executables = {base, base.with_name('pythonw.exe')}
        parent = process.parent()
        if (Path(sys.executable).resolve() not in configured
                or executable not in base_executables or command_executable not in base_executables
                or parent is None or not parent.is_running()
                or parent.create_time() > created
                or Path(parent.exe()).resolve() not in configured
                or Path(parent.cwd()).resolve() != Path(settings.root).resolve()):
            raise DesktopError('L’environnement Python du backend NYX ne peut pas être confirmé ; arrêt refusé.')
        parent_command = parent.cmdline()
        if (not parent_command or Path(parent_command[0]).resolve() not in configured
                or parent_command[1:] != command[1:]):
            raise DesktopError('Le parent du backend NYX utilise un autre environnement ; arrêt refusé.')
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    parser.add_argument('--settings', default=str(Path(settings.root) / 'config/experiment_console.json'))
    parser.add_argument('--port', type=int)
    parser.add_argument('--open', action='store_true')
    parser.add_argument('--restart', action='store_true')
    try:
        options, unknown = parser.parse_known_args(command[3:])
    except (argparse.ArgumentError, ValueError) as exc:
        raise DesktopError('Arguments du backend NYX non vérifiables ; arrêt refusé.') from exc
    process_settings = Path(options.settings)
    if not process_settings.is_absolute():
        process_settings = Path(settings.root) / process_settings
    if (unknown or not _same_path(str(process_settings), settings.path)
            or (options.port is not None and options.port != settings.port)):
        raise DesktopError('Le processus NYX utilise une autre configuration ; arrêt refusé.')
    listeners = [connection for connection in process.net_connections(kind='tcp')
                 if connection.status == psutil.CONN_LISTEN and connection.laddr
                 and connection.laddr.ip == '127.0.0.1' and connection.laddr.port == settings.port]
    if not listeners:
        raise DesktopError('Le processus NYX ne possède plus le port attendu ; arrêt refusé.')


def verify_backend_process(settings):
    """Return the exact listener and its process creation time."""
    try:
        pids = {connection.pid for connection in psutil.net_connections(kind='tcp')
                if connection.status == psutil.CONN_LISTEN and connection.laddr
                and connection.laddr.ip == '127.0.0.1' and connection.laddr.port == settings.port}
        if len(pids) != 1 or None in pids:
            raise DesktopError('Le processus du backend NYX ne peut pas être identifié sans ambiguïté.')
        process = psutil.Process(pids.pop())
        created = process.create_time()
        _process_identity(settings, process, created)
        return process, created
    except (psutil.Error, OSError, ValueError) as exc:
        raise DesktopError('L’identité du processus NYX est inaccessible ; aucun arrêt effectué.') from exc


def _assert_idle_rows(rows):
    if not isinstance(rows, list):
        raise DesktopError('État des calculs NYX illisible ; arrêt refusé.')
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('source'), str):
            raise DesktopError('État d’une exécution NYX ambigu ; arrêt refusé.')
        if row['source'] == 'managed' and row.get('status') not in IDLE_STATUSES:
            raise DesktopError('Un calcul NYX est actif, en attente ou dans un état non confirmé. Attendez sa fin avant -Restart.')


def _assert_no_import(value):
    if not isinstance(value, dict) or type(value.get('running')) is not bool:
        raise DesktopError('État de l’import NYX non vérifiable ; arrêt refusé.')
    if value['running']:
        raise DesktopError('Un import NYX est en cours. Attendez sa fin avant -Restart.')


def restart_backend(settings):
    """Stop only the verified idle backend, retaining a database writer lock.

    Dispatch commits `starting` before spawning a worker. BEGIN IMMEDIATE
    prevents every new queue insertion and dispatch update until backend exit.
    Scientific workers and process trees are never terminated here.
    """
    if not backend_ready(settings):
        return
    current = read_json(settings, '/api/runs')
    if not isinstance(current, dict):
        raise DesktopError('État du backend NYX illisible ; arrêt refusé.')
    _assert_idle_rows(current.get('runs'))
    _assert_no_import(current.get('import'))
    process, created = verify_backend_process(settings)
    database = Path(settings.state) / 'console.sqlite3'
    if not database.is_file() or database.resolve() != database.absolute():
        raise DesktopError('La base NYX est absente ou redirigée ; arrêt refusé.')
    connection = None
    try:
        # Existing database only; no records are changed.
        connection = sqlite3.connect(database.as_uri() + '?mode=rw', uri=True, timeout=2)
        connection.execute('BEGIN IMMEDIATE')
        records = [json.loads(row[0]) for row in connection.execute('SELECT data FROM runs').fetchall()]
        _assert_idle_rows(records)
        _assert_no_import(read_json(settings, '/api/import-status'))
        if not backend_ready(settings):
            raise DesktopError('Le service NYX a changé pendant la vérification ; arrêt refusé.')
        _process_identity(settings, process, created)
        # Only the verified listener PID; never children(), kill() or stop_tree().
        process.terminate()
        process.wait(timeout=5)
    except (sqlite3.Error, json.JSONDecodeError, TypeError, psutil.Error, OSError, ValueError) as exc:
        raise DesktopError('Le redémarrage sûr de NYX n’a pas pu être confirmé. Aucun calcul n’a été arrêté ; réessayez après vérification du service.') from exc
    finally:
        if connection is not None:
            connection.rollback()
            connection.close()


def _assert_backend_lock_free(settings):
    try:
        with FileLock(str(Path(settings.state) / 'backend.lock'), timeout=0):
            pass
    except Timeout as exc:
        raise BackendLocked('Une instance NYX utilise déjà ce dossier, mais ne répond pas sur le port demandé. Vérifiez son port ou attendez sa fermeture ; aucun second serveur n’a été lancé.') from exc


def prepare_launch(settings, restart=False, timeout=6):
    """Return True when reused; release desktop lock before foreground serve."""
    Path(settings.state).mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(Path(settings.state) / 'desktop-launch.lock'), timeout=timeout + 5):
            deadline = time.monotonic() + timeout
            while True:
                try:
                    ready = backend_ready(settings)
                except BackendBusy:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.2)
                    continue
                if ready and not restart:
                    return True
                if ready:
                    restart_backend(settings)
                try:
                    _assert_backend_lock_free(settings)
                except BackendLocked:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.2)
                    continue
                return False
    except Timeout as exc:
        raise DesktopError('Un lancement de NYX est déjà en cours. Patientez puis réessayez.') from exc


def _reuse(settings, open_browser):
    print(f'NYX est déjà démarré : {settings.url}')
    print('Après une mise à jour, utilisez Start-ExperimentConsole.ps1 -Restart pour charger le nouveau code lorsque les calculs sont terminés.')
    if open_browser:
        webbrowser.open(settings.url)


def _wait_for_reuse(settings, timeout=6):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if backend_ready(settings):
                return
        except BackendBusy:
            pass
        time.sleep(.2)
    raise DesktopError('Une autre instance NYX possède le dossier mais ne répond pas sur ce port. Vérifiez son port ; aucun processus n’a été arrêté.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings')
    parser.add_argument('--port', type=int)
    parser.add_argument('--open', action='store_true')
    parser.add_argument('--restart', action='store_true')
    args = parser.parse_args(argv)
    settings = DesktopSettings(args.settings, require_pythonw=False)
    if args.port is not None:
        if not 1024 <= args.port <= 65535:
            parser.error('Le port doit être compris entre 1024 et 65535.')
        settings.port = args.port
        settings.url = f'http://127.0.0.1:{settings.port}'
    if prepare_launch(settings, restart=args.restart):
        _reuse(settings, args.open)
        return 0
    from .manager import BackendAlreadyRunning
    from .server import main as serve
    arguments = ['--settings', str(settings.path), '--port', str(settings.port)]
    if args.open:
        arguments.append('--open')
    try:
        serve(arguments)
    except BackendAlreadyRunning:
        # Another launcher can win after the desktop lock is released. Reuse
        # only the owner's verified HTTP identity, never terminate a competitor.
        _wait_for_reuse(settings)
        _reuse(settings, args.open)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except DesktopError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
