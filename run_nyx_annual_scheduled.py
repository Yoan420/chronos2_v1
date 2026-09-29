"""Run annual NYX with durable, shareable diagnostics in console or scheduler."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from uuid import uuid4
from zoneinfo import ZoneInfo

import psutil

from experiment_console.processes import WindowsJob, stop_tree
from experiment_console.security import redact, redact_text

ROOT = Path(__file__).resolve().parent
ACTIONS = ("inspect", "capture", "bootstrap", "prepare", "forecast")
MAX_LINE_BYTES = 1024 * 1024


def _console(message):
    if sys.stdout is not None:
        try:
            sys.stdout.write(message)
            sys.stdout.flush()
        except UnicodeEncodeError:
            sys.stdout.write(message.encode(sys.stdout.encoding or "ascii", errors="replace").decode(
                sys.stdout.encoding or "ascii"))
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            pass


def _errors(value, prefix=""):
    """Preserve source causes, including nested source reports from this attempt."""
    found = []
    if not isinstance(value, dict):
        return found
    sources = value.get("sources")
    if isinstance(sources, dict):
        for name, item in sources.items():
            found.extend(_errors(item, f"{prefix}{name}: "))
    for item in value.get("stages", ()):
        if isinstance(item, dict) and item.get("state") in ("ERROR", "BLOCKED"):
            found.extend(_errors(item, f"{item.get('name', 'etape')}: "))
    for key in ("source_result", "details", "result", "diagnostic"):
        child = value.get(key)
        if isinstance(child, dict):
            found.extend(_errors(child, prefix))
    if value.get("error"):
        found.append(prefix + str(value["error"]))
    if value.get("activation_error"):
        found.append("Qualification: " + str(value["activation_error"]))
    if value.get("missing_modules"):
        found.append("Modules absents: " + ", ".join(map(str, value["missing_modules"])))
    return list(dict.fromkeys(found))


def launch(action, day=None, *, console=False, result_file=None):
    if action not in ACTIONS:
        raise ValueError("Unknown annual NYX action")
    day = day or (datetime.now(ZoneInfo("Europe/Paris")).date() + timedelta(days=1)).isoformat()
    from datetime import date
    if date.fromisoformat(day).isoformat() != day:
        raise ValueError("Expected delivery day YYYY-MM-DD")
    folder = ROOT / "runs/logs/nyx_annual_cpu" / day
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("UTC")).strftime("%Y%m%dT%H%M%SZ")
    attempt = f"{action}_{stamp}_{uuid4().hex[:8]}"
    log = folder / f"{attempt}.log"
    diagnostic_path = folder / f"{attempt}.diagnostic.json"
    python = Path(sys.executable)
    if python.name.lower() == "pythonw.exe":
        python = python.with_name("python.exe")
    command = [str(python), "-u", str(ROOT / "run_nyx_annual_pipeline.py"),
               "--action", action, "--delivery-day", day]
    report = {"protocol": "nyx_annual_launcher_v1", "attempt_id": attempt,
              "action": action, "delivery_day": day, "started_at_utc": stamp,
              "log_path": str(log), "diagnostic_path": str(diagnostic_path)}
    current_result, source_errors, exceptions = {}, [], []
    process = job = reader = None
    code, interrupted = 1, False
    if console:
        _console(f"Journal du calcul : {log}\n")
    with log.open("w", encoding="utf-8", buffering=1) as stream:
        stream.write(json.dumps(report, ensure_ascii=False) + "\n")

        def consume():
            private = False
            while True:
                raw = process.stdout.readline(MAX_LINE_BYTES + 1)
                if not raw:
                    break
                decoded = raw.decode("utf-8", errors="replace")
                if "-----BEGIN " in decoded and "PRIVATE KEY-----" in decoded:
                    private = True
                    stream.write("[NYX] Cle privee masquee.\n")
                if len(raw) > MAX_LINE_BYTES:
                    while raw and not raw.endswith(b"\n"):
                        raw = process.stdout.readline(MAX_LINE_BYTES + 1)
                    line = "[NYX] Ligne trop longue masquee.\n"
                else:
                    if private:
                        if "-----END " in decoded and "PRIVATE KEY-----" in decoded:
                            private = False
                        continue
                    match = re.match(r"^([\w.]*(?:Error|Exception):\s*)(.*)$", decoded, re.S)
                    line = match[1] + redact_text(match[2]) if match else redact_text(decoded)
                    try:
                        item = json.loads(decoded)
                    except (ValueError, TypeError):
                        item = None
                    if isinstance(item, dict):
                        cleaned = redact(item)
                        line = json.dumps(cleaned, ensure_ascii=False) + "\n"
                        if item.get("protocol") == "nyx_annual_cpu_daily_pipeline_v1":
                            current_result.clear()
                            current_result.update(cleaned)
                        elif item.get("state") in ("ERROR", "BLOCKED"):
                            source_errors.extend(_errors(cleaned))
                    elif "Error:" in line or "Exception:" in line:
                        exceptions.append(line.strip())
                stream.write(line)
                if console:
                    _console(line)

        try:
            job = WindowsJob()
            env = os.environ.copy()
            env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
            flags = (subprocess.CREATE_NO_WINDOW | 0x00000004) if os.name == "nt" else 0
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False,
                creationflags=flags, start_new_session=os.name != "nt")
            job.assign(process)
            if os.name == "nt":
                psutil.Process(process.pid).resume()
            reader = threading.Thread(target=consume, daemon=True)
            reader.start()
            while True:
                try:
                    code = process.wait(timeout=.25)
                    break
                except subprocess.TimeoutExpired:
                    pass
        except KeyboardInterrupt:
            interrupted, code = True, 130
            exceptions.append("Calcul interrompu par l'utilisateur.")
        except Exception as error:
            exceptions.append(f"{type(error).__name__}: {redact_text(str(error))}")
        finally:
            if process is not None and process.poll() is None:
                try:
                    stop_tree(process, job)
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=10)
            if job is not None:
                job.close()
            if reader is not None:
                reader.join(timeout=10)
            if process is not None and (reader is None or not reader.is_alive()):
                process.stdout.close()
        failures = list(dict.fromkeys([*_errors(current_result), *source_errors, *exceptions])) if code else []
        if interrupted:
            state = "INTERRUPTED"
        elif action == "inspect" and code == 2 and current_result.get("ready") is False:
            state = "NOT_READY"
        else:
            state = current_result.get("state", "COMPLETE" if code == 0 else "ERROR")
        if code and not failures:
            failures = [f"Le processus s'est termine avec le code {code}. Consulter le journal."]
        report.update(return_code=code, state=state, errors=failures,
                      failure_summary="; ".join(failures)[:4000], pipeline_result=current_result,
                      finished_at_utc=datetime.now(ZoneInfo("UTC")).isoformat())
        report = redact(report)
        stream.write(json.dumps(report, ensure_ascii=False) + "\n")
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    diagnostic_path.write_text(serialized, encoding="utf-8")
    if result_file:
        target = Path(result_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(serialized, encoding="utf-8")
    if console:
        _console(f"\nEtat : {state} (code {code})\n")
        for failure in failures:
            _console(f"- {failure}\n")
        _console(f"Journal : {log}\nDiagnostic a transmettre : {diagnostic_path}\n")
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=ACTIONS, required=True)
    parser.add_argument("--delivery-day")
    parser.add_argument("--console", action="store_true", help="Afficher et enregistrer le journal")
    parser.add_argument("--result-file", type=Path, help="Copie du diagnostic de cette tentative pour le lanceur")
    args = parser.parse_args(argv)
    return launch(args.action, args.delivery_day, console=args.console, result_file=args.result_file)


if __name__ == "__main__":
    raise SystemExit(main())
