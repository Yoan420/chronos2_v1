"""Silent Windows entry point for the local NYX application.

The browser supplies the dedicated app window; the existing Python service owns
the durable queue. Closing the window therefore never interrupts an experiment.
"""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

from filelock import FileLock, Timeout


class DesktopError(RuntimeError):
    """A startup error safe to show in a Windows message box."""


class BackendBusy(DesktopError):
    """A bound service may still be warming up; never start a competing one."""


class DesktopSettings:
    def __init__(self, settings_path=None):
        self.root = Path(__file__).resolve().parents[1]
        self.path = Path(settings_path or self.root / 'config' / 'experiment_console.json').resolve()
        settings = json.loads(self.path.read_text(encoding='utf-8-sig'))
        self.python = Path(settings['python_executable'])
        if not self.python.is_absolute() or not self.python.is_file():
            raise DesktopError(f'L’environnement Python configuré est introuvable.\nConfiguration : {self.path}')
        self.python = self.python.resolve()
        self.pythonw = self.python.with_name('pythonw.exe')
        if not self.pythonw.is_file():
            raise DesktopError(f'Le lanceur Windows Python est introuvable :\n{self.pythonw}')
        self.state = Path(settings.get('state_root', 'runs/.experiment_console'))
        if not self.state.is_absolute():
            self.state = self.root / self.state
        self.state = self.state.resolve()
        if not self.state.is_relative_to(self.root):
            raise DesktopError('Le dossier de la console doit rester dans le projet NYX.')
        self.port = settings.get('port', 8765)
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1024 <= self.port <= 65535:
            raise DesktopError('Le port de la console doit être un entier entre 1024 et 65535.')
        self.url = f'http://127.0.0.1:{self.port}'


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise DesktopError('Une autre application répond à l’adresse locale de NYX.')


def _port_available(settings):
    # Some Windows networking policies time out even on an unused loopback port.
    # An exclusive bind distinguishes this from a live but unresponsive service.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if os.name == 'nt':
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind(('127.0.0.1', settings.port))
        except OSError:
            return False
    return True


def backend_ready(settings, timeout=2.0):
    """Reuse only the service for this exact checkout, state and interpreter."""
    # Local requests must not inherit a corporate proxy or follow a redirect.
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(settings.url + '/api/bootstrap', timeout=timeout) as response:
            payload = response.read(1_000_001)
            if len(payload) > 1_000_000:
                raise DesktopError('La réponse à l’adresse locale de NYX est inattendue.')
            data = json.loads(payload)
    except HTTPError as exc:
        raise DesktopError(f'Le port {settings.port} est occupé par un service qui ne répond pas comme NYX.') from exc
    except URLError as exc:
        reason = exc.reason
        if (isinstance(reason, (ConnectionRefusedError, TimeoutError)) or getattr(reason, 'errno', None) in {errno.ECONNREFUSED, 10061}) and _port_available(settings):
            return False
        raise BackendBusy('Le service local ne répond pas. Patientez puis ouvrez NYX à nouveau.') from exc
    except (TimeoutError, socket.timeout) as exc:
        if _port_available(settings):
            return False
        raise BackendBusy('Le service local met trop de temps à répondre. Patientez puis réessayez.') from exc
    except (ValueError, UnicodeError) as exc:
        raise DesktopError('Une autre application utilise le port local de NYX.') from exc
    if not isinstance(data, dict):
        raise DesktopError('La réponse à l’adresse locale de NYX est inattendue.')
    for key, expected in [('project_root', settings.root), ('state_root', settings.state), ('python_executable', settings.python)]:
        value = data.get(key)
        if not isinstance(value, str) or Path(value).resolve() != expected.resolve():
            raise DesktopError(f'Le port {settings.port} est déjà utilisé par une autre instance.\nFermez cette instance ou choisissez un autre port dans la configuration.')
    if not isinstance(data.get('catalog'), list):
        raise DesktopError('La réponse à l’adresse locale de NYX est inattendue.')
    return True


def find_browser():
    """Find an installed app-window host without depending on PATH."""
    candidates = []
    if os.name == 'nt':
        import winreg
        for executable in ('msedge.exe', 'chrome.exe'):
            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                try:
                    with winreg.OpenKey(hive, rf'SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{executable}') as key:
                        candidates.append(Path(winreg.QueryValue(key, None)))
                except OSError:
                    pass
    for parent in ('ProgramFiles(x86)', 'ProgramFiles', 'LOCALAPPDATA'):
        base = os.environ.get(parent)
        if base:
            candidates.extend(Path(base) / relative for relative in (
                'Microsoft/Edge/Application/msedge.exe', 'Google/Chrome/Application/chrome.exe'))
    for candidate in candidates:
        if candidate.is_absolute() and candidate.is_file():
            return candidate.resolve()
    raise DesktopError('Microsoft Edge ou Google Chrome est nécessaire pour afficher la fenêtre NYX. Aucun des deux n’a été trouvé.')


def start_backend(settings):
    log_path = settings.state / 'desktop_startup.log'
    if log_path.exists() and log_path.stat().st_size > 256_000:
        log_path.replace(log_path.with_suffix('.previous.log'))
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    with log_path.open('ab') as output:
        return subprocess.Popen(
            [str(settings.pythonw), '-m', 'experiment_console.server', '--settings', str(settings.path), '--port', str(settings.port)],
            cwd=settings.root, stdin=subprocess.DEVNULL, stdout=output, stderr=output,
            env={**os.environ, 'PYTHONUTF8': '1'}, creationflags=flags,
            close_fds=True,
        )


def ensure_backend(settings, startup_timeout=40.0):
    settings.state.mkdir(parents=True, exist_ok=True)
    # Serialize separate desktop launcher processes, including rapid double clicks.
    try:
        with FileLock(str(settings.state / 'desktop-launch.lock'), timeout=startup_timeout + 5):
            process = None
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                try:
                    if backend_ready(settings):
                        return process is not None
                except BackendBusy:
                    # The socket can become bound between connect() and our
                    # exclusive-bind probe. Wait for the same server to answer.
                    pass
                else:
                    if process is None:
                        process = start_backend(settings)
                if process is not None and process.poll() is not None:
                    raise DesktopError(f'NYX n’a pas pu démarrer.\nLe diagnostic est enregistré dans :\n{settings.state / "desktop_startup.log"}')
                time.sleep(0.25)
            # Do not kill an uncertain service: it may already own active runs.
            raise DesktopError('Le démarrage de NYX prend plus de temps que prévu.\nPatientez quelques instants, puis cliquez à nouveau sur l’icône.')
    except Timeout as exc:
        raise DesktopError('NYX est déjà en cours de démarrage. Patientez puis réessayez.') from exc


def open_window(settings, browser):
    # A private application profile keeps NYX separate from personal tabs.
    return subprocess.Popen(
        [str(browser), f'--app={settings.url}/', f'--user-data-dir={settings.state / "desktop_browser"}', '--no-first-run', '--no-default-browser-check'],
        cwd=settings.root, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        close_fds=True,
    )


def main(settings_path=None):
    settings = DesktopSettings(settings_path)
    browser = find_browser()
    ensure_backend(settings)
    open_window(settings, browser)
