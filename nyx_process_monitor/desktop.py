"""Open the read-only viewer in a dedicated app window; closing it stops no job."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import build_opener, ProxyHandler, HTTPRedirectHandler
from urllib.error import URLError

from filelock import FileLock
from experiment_console.desktop import find_browser

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'runs/.process_monitor'
PORT = 8766
URL = f'http://127.0.0.1:{PORT}'


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError('Service local inattendu : redirection refusée')


def ready():
    try:
        opener = build_opener(ProxyHandler({}), NoRedirect())
        with opener.open(URL + '/api/health', timeout=1) as response:
            value = json.loads(response.read(10000))
        if value.get('app') != 'nyx-process-monitor' or Path(value.get('project_root', '')).resolve() != ROOT:
            raise RuntimeError(f'Le port {PORT} est utilisé par un autre programme')
        return True
    except (URLError, TimeoutError, ConnectionError):
        return False


def start():
    if STATE.resolve() != STATE.absolute():
        raise RuntimeError('Dossier du moniteur redirigé')
    STATE.mkdir(parents=True, exist_ok=True)
    with FileLock(str(STATE / 'startup.lock'), timeout=25):
        if ready():
            return
        # Never replace a bound but unresponsive service.
        with socket.socket() as probe:
            if os.name == 'nt':
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                probe.bind(('127.0.0.1', PORT))
            except OSError as exc:
                raise RuntimeError(f'Le port {PORT} est occupé. Aucun service arrêté.') from exc
        executable = Path(sys.executable)
        pythonw = executable.with_name('pythonw.exe')
        if pythonw.is_file():
            executable = pythonw
        flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS if os.name == 'nt' else 0
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + f'_{os.getpid()}'
        with (STATE / f'{stamp}.stdout.log').open('xb') as out, (STATE / f'{stamp}.stderr.log').open('xb') as err:
            child = subprocess.Popen([str(executable), '-B', '-u', '-m', 'nyx_process_monitor.server', '--port', str(PORT)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err, creationflags=flags, close_fds=True)
        for _ in range(60):
            if ready():
                return
            if child.poll() is not None:
                raise RuntimeError('Le moniteur n’a pas démarré. Consulter runs/.process_monitor ; aucun calcul modifié.')
            time.sleep(.25)
        raise RuntimeError('Démarrage lent. Réessayer dans quelques instants ; aucun calcul modifié.')


def main():
    try:
        browser = find_browser()
        start()
        # This window is explicitly requested. The dedicated profile does not
        # reuse or change the user's personal browser tabs or the main NYX UI.
        subprocess.Popen([str(browser), f'--app={URL}/', f'--user-data-dir={STATE / "browser"}',
            '--no-first-run', '--no-default-browser-check', '--window-size=1150,850'],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    except Exception as exc:
        if os.name == 'nt':
            ctypes.windll.user32.MessageBoxW(None, str(exc), 'Moniteur NYX', 0x10)
        else:
            raise


if __name__ == '__main__':
    main()
