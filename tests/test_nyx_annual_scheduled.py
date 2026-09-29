import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil
import pytest

import run_nyx_annual_scheduled as m


@pytest.fixture
def project(monkeypatch, tmp_path):
    root = tmp_path / "NYX avec espaces et accents"
    root.mkdir()
    monkeypatch.setattr(m, "ROOT", root)
    return root


def pipeline(root, code=2, action="prepare", *, secret=False):
    result = {"protocol": "nyx_annual_cpu_daily_pipeline_v1", "action": action,
              "state": "BLOCKED" if code else "PREPARED", "ready": not bool(code),
              "stages": [{"name": "saturn", "state": "ERROR", "error": "Accès Saturn indisponible"}] if code else []}
    text = "import json, os, sys\n"
    text += "print('Historique reçu', flush=True)\n"
    text += "print('avertissement natif sur stderr', file=sys.stderr, flush=True)\n"
    text += "assert os.environ['PYTHONIOENCODING'] == 'utf-8'\n"
    if secret:
        text += "print(json.dumps({'state': 'ERROR', 'error': 'password=example-secret'}), flush=True)\n"
        text += "print('-----BEGIN PRIVATE KEY-----\\nprivate-body\\n-----END PRIVATE KEY-----', flush=True)\n"
    text += f"print(json.dumps({result!r}, ensure_ascii=False), flush=True)\n"
    text += f"raise SystemExit({code})\n"
    (root / "run_nyx_annual_pipeline.py").write_text(text, encoding="utf-8")


@pytest.mark.parametrize("code", [0, 2])
def test_console_preserves_stderr_unicode_exit_and_current_attempt(project, capsys, code):
    pipeline(project, code)
    stale = project / "runs/nyx_annual_cpu_live/2026-09-30.pipeline.json"
    stale.parent.mkdir(parents=True)
    stale.write_text('{"error": "obsolete failure"}')
    handoff = project / "result with spaces.json"
    before = os.environ.get("PYTHONIOENCODING")
    assert m.launch("prepare", "2026-09-30", console=True, result_file=handoff) == code
    assert os.environ.get("PYTHONIOENCODING") == before
    report = json.loads(handoff.read_text(encoding="utf-8"))
    content = Path(report["log_path"]).read_text(encoding="utf-8")
    assert "Historique reçu" in content and "avertissement natif sur stderr" in content
    assert report["return_code"] == code
    assert report["state"] == ("BLOCKED" if code else "PREPARED")
    assert "obsolete failure" not in content
    assert json.loads(Path(report["diagnostic_path"]).read_text(encoding="utf-8")) == report
    output = capsys.readouterr().out
    assert "Diagnostic a transmettre" in output
    assert "Accès Saturn indisponible" in report["failure_summary"] if code else not report["errors"]


def test_inspect_not_ready_is_distinct_from_source_failure(project):
    pipeline(project, action="inspect")
    handoff = project / "result.json"
    assert m.launch("inspect", "2026-09-30", result_file=handoff) == 2
    assert json.loads(handoff.read_text(encoding="utf-8"))["state"] == "NOT_READY"


def test_shared_log_and_json_redact_nested_error_and_private_key(project, capsys):
    pipeline(project, secret=True)
    handoff = project / "result.json"
    assert m.launch("prepare", "2026-09-30", console=True, result_file=handoff) == 2
    report = json.loads(handoff.read_text(encoding="utf-8"))
    combined = (handoff.read_text(encoding="utf-8")
                + Path(report["log_path"]).read_text(encoding="utf-8") + capsys.readouterr().out)
    assert "example-secret" not in combined and "private-body" not in combined
    assert "MASQUÉ" in combined


def test_child_start_failure_produces_diagnostic_without_old_pipeline(project, monkeypatch):
    monkeypatch.setattr(m.sys, "executable", str(project / "missing python.exe"))
    handoff = project / "result.json"
    assert m.launch("bootstrap", "2026-09-30", result_file=handoff) == 1
    report = json.loads(handoff.read_text(encoding="utf-8"))
    assert report["state"] == "ERROR" and report["pipeline_result"] == {}
    assert "FileNotFoundError" in report["failure_summary"]


def test_logs_are_unique_for_repeated_attempts(project):
    pipeline(project, code=0)
    for _ in range(2):
        assert m.launch("prepare", "2026-09-30") == 0
    assert len(list((project / "runs/logs/nyx_annual_cpu/2026-09-30").glob("*.log"))) == 2


def test_interrupt_stops_process_tree_and_retains_diagnostic(project, monkeypatch):
    child_pid = project / "descendant.pid"
    source = ("import subprocess, sys, time\n"
              "from pathlib import Path\n"
              "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
              f"Path({str(child_pid)!r}).write_text(str(child.pid))\n"
              "print('started', flush=True)\ntime.sleep(60)\n")
    (project / "run_nyx_annual_pipeline.py").write_text(source, encoding="utf-8")
    real_popen = subprocess.Popen
    processes = []

    def invoke(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        original_wait = process.wait
        state = {"first": True}

        def interrupt_once(*args, **kwargs):
            if state["first"]:
                state["first"] = False
                deadline = time.monotonic() + 5
                while not child_pid.exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                raise KeyboardInterrupt()
            return original_wait(*args, **kwargs)

        process.wait = interrupt_once
        processes.append(process)
        return process

    monkeypatch.setattr(m.subprocess, "Popen", invoke)
    handoff = project / "result.json"
    assert m.launch("prepare", "2026-09-30", result_file=handoff) == 130
    assert processes[0].poll() is not None
    assert child_pid.exists()
    pid = int(child_pid.read_text())
    deadline = time.monotonic() + 3
    while psutil.pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(.02)
    assert not psutil.pid_exists(pid)
    assert json.loads(handoff.read_text(encoding="utf-8"))["state"] == "INTERRUPTED"


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell 5.1 wrapper")
@pytest.mark.parametrize("action,code", [("prepare", 0), ("prepare", 2), ("inspect", 2)])
def test_powershell_wrapper_keeps_native_warnings_and_reports_cause(project, action, code):
    repository = Path(m.__file__).resolve().parent
    pipeline(project, code=code, action=action)
    shutil.copy2(repository / "NYXAnnualCPU.ps1", project)
    shutil.copy2(repository / "run_nyx_annual_scheduled.py", project)
    # The launcher's helpers are imported from the actual checkout; the stand-in
    # pipeline remains local and never contacts a network or trains a model.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repository)
    env["PYTHONIOENCODING"] = "utf-8"
    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    command = ("[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding; "
               "$ErrorActionPreference = 'Stop'; try { & " + quote(project / "NYXAnnualCPU.ps1")
               + " -Action " + action + " -DeliveryDay 2026-09-30 -PythonExecutable " + quote(sys.executable)
               + "; Write-Output ('WRAPPER_EXIT=' + $LASTEXITCODE) } catch { Write-Output ('WRAPPER_ERROR=' + $_.Exception.Message); exit 1 }")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    output = result.stdout.decode("utf-8", errors="replace")
    assert "avertissement natif sur stderr" in output
    assert "NativeCommandError" not in output
    if code and action != "inspect":
        assert result.returncode == 1
        assert "Accès Saturn indisponible" in output
        assert "Diagnostic a transmettre" in output
    else:
        assert result.returncode == 0
        assert f"WRAPPER_EXIT={code}" in output
