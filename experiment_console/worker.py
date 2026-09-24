"""Detached run supervisor: persists status and redacted logs without a browser/backend."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import threading
import time
import psutil

from .processes import WindowsJob, stop_tree
from .error_summary import FailureSummary
from .security import redact_text
from .store import Store, now


def execute(database, run_id):
    store = Store(database)
    run = store.get(run_id)
    process = None
    job = None
    reader = None
    started = time.monotonic()
    stage = {'line': 'Démarrage du processus'}
    primary = run.get('adapter_id', run.get('request', {}).get('adapter_id')) == 'primary_nuclear_kalman'
    failure = FailureSummary()
    log_path = Path(run['run_dir']) / 'console.log'
    try:
        identity = psutil.Process()
        store.update(run_id, worker_pid=os.getpid(), worker_created=identity.create_time(), heartbeat_at=now())
        if store.get(run_id).get('cancel_requested'):
            store.update(run_id, status='cancelled', finished_at=now(), duration_seconds=0, activity='Annulé avant démarrage')
            return
        job = WindowsJob()
        flags = (subprocess.CREATE_NO_WINDOW | 0x00000004) if os.name == 'nt' else 0  # CREATE_SUSPENDED: assign before any child can escape
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
        process = subprocess.Popen(run['command'], cwd=run['cwd'], env=env, shell=False, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=flags, start_new_session=os.name != 'nt')
        job.assign(process)
        child = psutil.Process(process.pid)
        store.update(run_id, status='running', started_at=now(), child_pid=process.pid, child_created=child.create_time(), activity=stage['line'])
        if os.name == 'nt':
            child.resume()

        def consume():
            with log_path.open('a', encoding='utf-8', buffering=1) as log:
                private_block = False
                while True:
                    raw = process.stdout.readline(65537)
                    if not raw:
                        break
                    if len(raw) > 65536:
                        while raw and not raw.endswith(b'\n'):
                            raw = process.stdout.readline(65537)
                        line = '[Console] Ligne trop longue masquée.\n'
                    else:
                        decoded = raw.decode('utf-8', errors='replace')
                        if '-----BEGIN ' in decoded and 'PRIVATE KEY-----' in decoded:
                            private_block = True
                            log.write('[Console] Clé privée masquée.\n')
                        if private_block:
                            if '-----END ' in decoded and 'PRIVATE KEY-----' in decoded:
                                private_block = False
                            continue
                        line = redact_text(decoded)
                    log.write(line)
                    if primary:
                        failure.feed(line)
                    if line.strip():
                        stage['line'] = line.strip()[-500:]

        reader = threading.Thread(target=consume, daemon=True)
        reader.start()
        cancelled = False
        while process.poll() is None:
            current = store.get(run_id)
            if current.get('cancel_requested'):
                cancelled = True
                store.update(run_id, status='cancelling', activity='Arrêt du processus et de ses enfants')
                stop_tree(process, job)
                break
            store.update(run_id, heartbeat_at=now(), activity=stage['line'])
            time.sleep(0.25)
        code = process.wait(timeout=10)
        job.close()  # Also terminate children left behind after a successful parent exit.
        reader.join(timeout=3)
        summary = {'failure_summary': failure.message if not cancelled and code != 0 else None} if primary else {}
        store.update(run_id, status='cancelled' if cancelled else ('succeeded' if code == 0 else 'failed'), return_code=code, finished_at=now(), duration_seconds=round(time.monotonic() - started, 3), activity=stage['line'], heartbeat_at=now(), **summary)
    except BaseException as exc:
        if process is not None:
            try:
                if job is not None:
                    stop_tree(process, job)
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
            except Exception:
                pass
        message = redact_text(f'{type(exc).__name__}: {exc}')
        with log_path.open('a', encoding='utf-8') as log:
            log.write(f'\n[Console ERROR] {message}\n')
        store.update(run_id, status='failed', finished_at=now(), duration_seconds=round(time.monotonic() - started, 3), activity=message,
                     **({'failure_summary': message[:600]} if primary else {}))
    finally:
        if job is not None:
            job.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    execute(args.database, args.run_id)
