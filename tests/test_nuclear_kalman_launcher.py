from __future__ import annotations

import json
from contextlib import contextmanager
import io
from pathlib import Path
import shutil
import subprocess
import sys
from textwrap import dedent

import pytest

import run_nuclear_kalman as runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell.exe") or shutil.which("pwsh") or shutil.which("powershell")


def plan(root: Path, **kwargs):
    args = dict(project_root=root, python_executable=sys.executable, zones=("BE", "DE", "FR", "NL"),
        delivery_day="2026-09-11", nuclear_config=Path("config/nuclear.yaml"),
        nuclear_root=Path("runs/experiments/saved results"), report_output=Path("reports/.new-batch.html"))
    args.update(kwargs)
    return runner.build_nuclear_kalman_plan(**args)


def test_plan_only_nuclear_kalman_one_sync_and_one_grouped_report(tmp_path):
    steps = plan(tmp_path, zones=("DE", "BE", "DE"), threads=7, workers=3, device="cpu")
    assert [step.name for step in steps] == ["sources", "nuclear_kalman", "nuclear_kalman", "CWE_Model_Storm"]
    assert [step.zone for step in steps] == [None, "DE", "BE", None]
    assert steps[0].command[-5:] == ("--stage", "Sync", "--zones", "DE", "BE")
    for step in steps:
        assert step.command[step.command.index("--delivery-day") + 1] == "2026-09-11"
        assert "run_multicountry_forecast.py" not in " ".join(step.command)
        assert "run_complete_forecast.py" not in " ".join(step.command)
    for step in steps[1:-1]:
        assert step.command[step.command.index("--stage") + 1] == "Run"
        assert step.command[step.command.index("--report-variants") + 1] == "kalman"
        assert "--skip-source-sync" in step.command
        assert "--skip-attribution" in step.command
        assert step.command[step.command.index("--threads") + 1] == "7"
        assert step.command[step.command.index("--device") + 1] == "cpu"
    assert steps[-1].command[-1] == str(tmp_path / "reports/.new-batch.html")
    assert steps[-1].command[-3] == str(tmp_path / "runs/experiments/saved results")
    assert "--skip-vps-sync" not in steps[-1].command


def test_attribution_and_explicit_local_mode_are_opt_in(tmp_path):
    steps = plan(tmp_path, with_attribution=True, skip_observed_sync=True)
    for step in steps[1:-1]:
        assert "--skip-attribution" not in step.command
        assert "--skip-observed-sync" in step.command
    assert "--skip-vps-sync" in steps[-1].command
    assert steps[-1].command[-1] == str(tmp_path / "reports/.new-batch.html")


@pytest.mark.parametrize("overrides", [dict(zones=[]), dict(zones=["ES"]), dict(threads=0),
    dict(workers=129), dict(device="other"), dict(delivery_day="2026-02-30")])
def test_plan_rejects_invalid_arguments(tmp_path, overrides):
    with pytest.raises(ValueError):
        plan(tmp_path, **overrides)


@pytest.fixture
def subprocess_project(tmp_path):
    root = tmp_path / "project with spaces"
    root.mkdir()
    (root / "run_nuclear_forecast.py").write_text(dedent('''
        import argparse, hashlib, json, os
        from pathlib import Path
        parser=argparse.ArgumentParser()
        parser.add_argument('--stage')
        parser.add_argument('--zones', nargs='+')
        parser.add_argument('--delivery-day')
        args, other=parser.parse_known_args()
        root=Path(__file__).parent
        with (root/'executed.jsonl').open('a') as stream:
            stream.write(json.dumps({'stage': args.stage, 'zones': args.zones})+'\\n')
        print('Persistent child output', flush=True)
        if args.stage == 'Sync':
            raise SystemExit(int(os.environ.get('STUB_SYNC_CODE', '0')))
        zone=args.zones[0]
        code=int(os.environ.get('STUB_CODE_'+zone, '0'))
        if code:
            print('Simulated country failure', flush=True)
            raise SystemExit(code)
        if os.environ.get('STUB_NO_RECEIPT') == zone:
            raise SystemExit(0)
        destination=root/'runs/exports'/args.delivery_day/zone.lower()
        folder=destination/'nuclear_kalman'
        folder.mkdir(parents=True, exist_ok=True)
        stem=f'forecast_{zone.lower()}_{args.delivery_day}_nuclear_kalman'
        html=folder/(stem+'.html')
        csv=folder/(stem+'.csv')
        html.write_text('<html>Fresh '+zone+'</html>')
        csv.write_text('q50\\n80.0\\n')
        records=[{'path': 'nuclear_kalman/'+p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in (html, csv)]
        manifest=destination/'current_nuclear_batch_manifest.json'
        manifest.write_text(json.dumps({'zone': zone, 'delivery_day': args.delivery_day,
            'exports':[{'variant':'nuclear_kalman', 'files':records}]}))
        if os.environ.get('STUB_TAMPER') == zone:
            csv.write_text('bad changed bytes')
        print(json.dumps({'zone': zone, 'status': 'complete', 'exports': {
            'kalman': str(html), 'manifest': str(manifest)}}), flush=True)
    '''), encoding="utf-8")
    (root / "run_model_storm_report.py").write_text(dedent('''
        import argparse, json, os
        from pathlib import Path
        parser=argparse.ArgumentParser()
        parser.add_argument('--output')
        args, other=parser.parse_known_args()
        root=Path(__file__).parent
        with (root/'executed.jsonl').open('a') as stream:
            stream.write(json.dumps({'stage': 'Report'})+'\\n')
        code=int(os.environ.get('STUB_REPORT_CODE', '0'))
        if code:
            raise SystemExit(code)
        if os.environ.get('STUB_REPORT_NO_OUTPUT') != '1':
            output=Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('<html>Fresh CWE Model / Storm</html>')
        print('Grouped report stage finished', flush=True)
    '''), encoding="utf-8")
    return root


def execute(root):
    output = root / "reports/CWE_Model_Storm_2026-09-11.html"
    output.parent.mkdir(exist_ok=True)
    output.write_text("<html>Old report</html>")
    log_directory = root / "runs/logs/nuclear_kalman/2026-09-11/batch-test"
    code = runner.execute_nuclear_kalman_plan(plan(root), project_root=root, delivery_day="2026-09-11",
        output=output, log_directory=log_directory, run_id="batch-test")
    status = json.loads((log_directory / "status.json").read_text(encoding="utf-8"))
    latest = json.loads((log_directory.parent / "latest_status.json").read_text(encoding="utf-8"))
    assert status == latest
    assert not (root / "reports/.new-batch.html").exists()
    assert status["finished_at_utc"]
    calls = [json.loads(line) for line in (root / "executed.jsonl").read_text().splitlines()]
    return code, status, calls, output


def test_real_subprocess_batch_saves_logs_validates_exports_and_publishes_once(subprocess_project):
    code, status, calls, output = execute(subprocess_project)
    assert code == 0
    assert status["status"] == "complete"
    assert [call["stage"] for call in calls] == ["Sync", "Run", "Run", "Run", "Run", "Report"]
    assert [item["zone"] for item in status["steps"] if item["zone"]] == ["BE", "DE", "FR", "NL"]
    assert "Fresh CWE" in output.read_text()
    for item in status["steps"]:
        assert item["status"] == "complete"
        assert Path(item["log"]).is_file()
        assert "Commande" in Path(item["log"]).read_text()
    assert "Persistent child output" in Path(status["steps"][1]["log"]).read_text()


def test_country_failure_preserved_while_other_countries_and_report_complete(subprocess_project, monkeypatch):
    monkeypatch.setenv("STUB_CODE_DE", "17")
    code, status, calls, output = execute(subprocess_project)
    assert code == 1
    assert status["status"] == "failed"
    assert [item["status"] for item in status["steps"]] == ["complete", "complete", "failed", "complete", "complete", "complete"]
    assert status["steps"][2]["returncode"] == 17
    assert len(calls) == 6
    assert "Fresh CWE" in output.read_text()


def test_sync_failure_skips_models_but_assembles_available_results(subprocess_project, monkeypatch):
    monkeypatch.setenv("STUB_SYNC_CODE", "19")
    code, status, calls, output = execute(subprocess_project)
    assert code == 1
    assert [call["stage"] for call in calls] == ["Sync", "Report"]
    assert all(item["status"] == "skipped" for item in status["steps"][1:-1])
    assert status["steps"][-1]["status"] == "complete"
    assert "Fresh CWE" in output.read_text()


@pytest.mark.parametrize("setting", ["STUB_NO_RECEIPT", "STUB_TAMPER"])
def test_zero_exit_without_valid_current_publication_is_failure(subprocess_project, monkeypatch, setting):
    # A prior successful publication must not mask this launch's broken receipt or checksum.
    assert execute(subprocess_project)[0] == 0
    monkeypatch.setenv(setting, "DE")
    code, status, _, _ = execute(subprocess_project)
    assert code == 1
    assert status["steps"][2]["status"] == "failed"
    assert status["steps"][2]["error"]


@pytest.mark.parametrize("setting,value", [("STUB_REPORT_NO_OUTPUT", "1"), ("STUB_REPORT_CODE", "21")])
def test_old_grouped_html_cannot_mask_failed_or_missing_new_report(subprocess_project, monkeypatch, setting, value):
    monkeypatch.setenv(setting, value)
    code, status, _, output = execute(subprocess_project)
    assert code == 1
    assert status["steps"][-1]["status"] == "failed"
    assert output.read_text() == "<html>Old report</html>"


def test_interrupt_stops_next_countries_and_preserves_old_report(subprocess_project, monkeypatch):
    monkeypatch.setenv("STUB_CODE_DE", "130")
    code, status, calls, output = execute(subprocess_project)
    assert code == 130
    assert status["status"] == "interrupted"
    assert [call.get("zones") for call in calls] == [["BE", "DE", "FR", "NL"], ["BE"], ["DE"]]
    assert [item["status"] for item in status["steps"]] == ["complete", "complete", "interrupted", "skipped", "skipped", "skipped"]
    assert output.read_text() == "<html>Old report</html>"


def test_launch_error_is_persisted_with_other_zones_continuing(subprocess_project, monkeypatch):
    original = runner.subprocess.Popen

    def fail_de(command, **kwargs):
        if "Run" in command and command[command.index("--zones") + 1] == "DE":
            raise OSError("No child process available")
        return original(command, **kwargs)

    monkeypatch.setattr(runner.subprocess, "Popen", fail_de)
    code, status, calls, _ = execute(subprocess_project)
    assert code == 1
    assert "No child process" in status["steps"][2]["error"]
    assert calls[-1]["stage"] == "Report"
    assert status["steps"][3]["status"] == "complete"


def _quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def _powershell(command, *, cwd):
    if POWERSHELL is None:
        pytest.skip("PowerShell unavailable")
    return subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
        command + "; exit $LASTEXITCODE"], cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False)


@pytest.fixture
def ps_launcher(tmp_path):
    root = tmp_path / "launcher with spaces"
    root.mkdir()
    shutil.copyfile(PROJECT_ROOT / "NuclearKalman.ps1", root / "NuclearKalman.ps1")
    (root / "run_nuclear_kalman.py").write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "Path(__file__).with_name('invocation.json').write_text(json.dumps(sys.argv[1:]))\n")
    return root / "NuclearKalman.ps1"


def test_powershell_dry_run_never_starts_python_and_pins_paths(ps_launcher, tmp_path):
    before = sorted(str(path) for path in tmp_path.rglob("*"))
    completed = _powershell(f"& {_quote(ps_launcher)} -PythonExecutable {_quote(sys.executable)} "
        "-Countries DE,BE -DeliveryDay 2026-09-11 -NuclearConfig 'config with spaces/nuclear.yaml' "
        "-OutputPath 'reports with spaces/report.html' -Threads 6 -Workers 2 -Device cpu -DryRun", cwd=tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    line = next(line for line in completed.stdout.splitlines() if line.startswith("Commande (argv, shell=False): "))
    args = json.loads(line.split(": ", 1)[1])
    assert args[0] == str(Path(sys.executable).resolve())
    assert args[2:5] == ["--zones", "DE", "BE"]
    assert args[args.index("--nuclear-config") + 1] == str(ps_launcher.parent / "config with spaces/nuclear.yaml")
    assert args[args.index("--output") + 1] == str(ps_launcher.parent / "reports with spaces/report.html")
    assert sorted(str(path) for path in tmp_path.rglob("*")) == before


def test_powershell_forwards_options_and_deduplicates_zones(ps_launcher, tmp_path):
    completed = _powershell(f"& {_quote(ps_launcher)} -PythonExecutable {_quote(sys.executable)} "
        "-Countries DE,BE,DE -DeliveryDay 2026-09-11 -WithAttribution -SkipObservedSync -NoOpen", cwd=tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    args = json.loads((ps_launcher.parent / "invocation.json").read_text())
    assert args[:3] == ["--zones", "DE", "BE"]
    assert args.count("DE") == 1
    assert {"--with-attribution", "--skip-observed-sync", "--no-open"}.issubset(args)


def test_powershell_normalizes_case_accepted_by_validate_set(ps_launcher, tmp_path):
    completed = _powershell(f"& {_quote(ps_launcher)} -PythonExecutable {_quote(sys.executable)} "
        "-Countries de,FR,De -Device CPU -DeliveryDay 2026-09-11 -NoOpen", cwd=tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    args = json.loads((ps_launcher.parent / "invocation.json").read_text())
    assert args[:3] == ["--zones", "DE", "FR"]
    assert args.count("DE") == 1
    assert args[args.index("--device") + 1] == "cpu"


def test_powershell_default_date_is_pinned_and_defaults_all_four_zones(ps_launcher, tmp_path):
    before = runner.delivery_date(None)
    completed = _powershell(f"& {_quote(ps_launcher)} -PythonExecutable {_quote(sys.executable)} -NoOpen", cwd=tmp_path)
    after = runner.delivery_date(None)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    args = json.loads((ps_launcher.parent / "invocation.json").read_text())
    assert args[:5] == ["--zones", "BE", "DE", "FR", "NL"]
    assert args[args.index("--delivery-day") + 1] in {before, after}


def test_powershell_preserves_child_failure_code(ps_launcher, tmp_path):
    (ps_launcher.parent / "run_nuclear_kalman.py").write_text("raise SystemExit(17)\n")
    completed = _powershell(f"& {_quote(ps_launcher)} -PythonExecutable {_quote(sys.executable)} "
                           "-DeliveryDay 2026-09-11 -NoOpen", cwd=tmp_path)
    assert completed.returncode == 17


def test_powershell_invalid_calendar_date_does_not_start_python(ps_launcher, tmp_path):
    completed = _powershell(f"& {_quote(ps_launcher)} -PythonExecutable {_quote(sys.executable)} "
                           "-DeliveryDay 2026-02-30 -NoOpen", cwd=tmp_path)
    assert completed.returncode != 0
    assert not (ps_launcher.parent / "invocation.json").exists()


def test_cli_dry_run_does_not_create_logs_or_locks(monkeypatch, tmp_path):
    import run_nuclear_forecast as nuclear

    for name in ("run_nuclear_forecast.py", "run_model_storm_report.py", "zone.yaml"):
        (tmp_path / name).write_text("stub")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(nuclear, "load_settings", lambda path: {"project_root": tmp_path,
        "output_root": tmp_path / "runs/experiments/nuclear", "zone_configs": {zone: "zone.yaml" for zone in runner.ZONES}})
    monkeypatch.setattr(nuclear, "check_lora_inactive", lambda *args: None)
    monkeypatch.setattr(runner, "_run_logged", lambda *args, **kwargs: pytest.fail("Dry run launched a child"))
    before = sorted(str(path) for path in tmp_path.rglob("*"))
    assert runner.main(["--delivery-day", "2026-09-11", "--dry-run"]) == 0
    assert sorted(str(path) for path in tmp_path.rglob("*")) == before


def test_interrupt_terminates_only_owned_tree_and_kills_only_survivors(monkeypatch):
    events = []

    class Owned:
        def __init__(self, name):
            self.name = name

        def children(self, recursive):
            assert recursive is True
            return [interpreter, worker]

        def terminate(self):
            events.append(("terminate", self.name))

        def kill(self):
            events.append(("kill", self.name))

    parent, interpreter, worker = Owned("venv-shim"), Owned("interpreter"), Owned("worker")

    def wait_procs(processes, timeout):
        assert timeout == 5
        if len(processes) == 3:
            assert set(processes) == {parent, interpreter, worker}
            return [parent, interpreter], [worker]
        assert processes == [worker]
        return [worker], []

    class Child:
        def wait(self, timeout):
            events.append(("wait", timeout))

    monkeypatch.setattr(runner.psutil, "wait_procs", wait_procs)
    runner._stop_child_tree(Child(), parent)
    assert events == [("terminate", "venv-shim"), ("terminate", "worker"),
                      ("terminate", "interpreter"), ("kill", "worker"), ("wait", 5)]


@pytest.mark.parametrize("exit_code,open_count", [(0, 1), (1, 0), (130, 0)])
def test_cli_uses_one_delivery_lock_and_opens_only_after_complete_success(monkeypatch, tmp_path, exit_code, open_count):
    import run_nuclear_forecast as nuclear

    for name in ("run_nuclear_forecast.py", "run_model_storm_report.py", "zone.yaml"):
        (tmp_path / name).write_text("stub")
    output_root = tmp_path / "runs/experiments/nuclear"
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(nuclear, "load_settings", lambda path: {"project_root": tmp_path,
        "output_root": output_root, "zone_configs": {zone: "zone.yaml" for zone in runner.ZONES}})
    monkeypatch.setattr(nuclear, "check_lora_inactive", lambda *args: None)
    locks = []
    opened = []

    @contextmanager
    def lock(path):
        locks.append(path)
        yield

    monkeypatch.setattr(nuclear, "exclusive_lock", lock)
    monkeypatch.setattr(runner, "execute_nuclear_kalman_plan", lambda *args, **kwargs: exit_code)
    monkeypatch.setattr(runner.webbrowser, "open", opened.append)
    assert runner.main(["--delivery-day", "2026-09-11", "--zones", "DE", "FR"]) == exit_code
    assert locks == [output_root / "_batch_locks/nuclear_kalman_2026-09-11.lock"]
    assert len(opened) == open_count


def test_cli_no_open_does_not_open_browser_after_success(monkeypatch, tmp_path):
    import run_nuclear_forecast as nuclear

    for name in ("run_nuclear_forecast.py", "run_model_storm_report.py", "zone.yaml"):
        (tmp_path / name).write_text("stub")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(nuclear, "load_settings", lambda path: {"project_root": tmp_path,
        "output_root": tmp_path / "runs/experiments/nuclear", "zone_configs": {zone: "zone.yaml" for zone in runner.ZONES}})
    monkeypatch.setattr(nuclear, "check_lora_inactive", lambda *args: None)
    monkeypatch.setattr(runner, "execute_nuclear_kalman_plan", lambda *args, **kwargs: 0)
    monkeypatch.setattr(runner.webbrowser, "open", lambda *args: pytest.fail("Opening must be disabled"))
    assert runner.main(["--delivery-day", "2026-09-11", "--no-open"]) == 0


def test_cp1252_console_preserves_unicode_tqdm_output_in_utf8_log(monkeypatch, tmp_path):
    original = "Loading checkpoint shards: 100%|██████████| 2/2 — terminé 🔋"
    child = tmp_path / "unicode_child.py"
    child.write_text("print(" + repr(original) + ", flush=True)\n", encoding="utf-8")
    step = runner.RunStep("unicode", None, (sys.executable, "-u", str(child)))
    console_bytes = io.BytesIO()
    console = io.TextIOWrapper(console_bytes, encoding="cp1252", errors="strict")
    log = tmp_path / "child.log"
    with monkeypatch.context() as context:
        context.setattr(runner.sys, "stdout", console)
        outcome = runner._run_logged(step, project_root=tmp_path, log_path=log)
    console.flush()
    visible = console_bytes.getvalue().decode("cp1252")
    assert outcome.returncode == 0
    assert outcome.child_status == "exited"
    assert original in log.read_text(encoding="utf-8")
    assert "\\u2588" in visible
    assert "\\U0001f50b" in visible
    assert "terminé" in visible


def test_unexpected_forwarding_failure_stops_real_child_tree_and_records_pid(monkeypatch, tmp_path):
    child = tmp_path / "waiting_child.py"
    child.write_text("import os, time\nprint('Interpreter pid=' + str(os.getpid()), flush=True)\n"
                     "time.sleep(60)\n", encoding="utf-8")
    step = runner.RunStep("failure", None, (sys.executable, "-u", str(child)))
    log = tmp_path / "child.log"

    class BrokenConsole:
        encoding = "cp1252"

        def __init__(self):
            self.calls = 0

        def write(self, value):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("Unexpected console forwarding failure")

        def flush(self):
            pass

    with monkeypatch.context() as context:
        context.setattr(runner.sys, "stdout", BrokenConsole())
        outcome = runner._run_logged(step, project_root=tmp_path, log_path=log)
    contents = log.read_text(encoding="utf-8")
    interpreter_pid = int(next(line.split("=", 1)[1] for line in contents.splitlines()
                               if line.startswith("Interpreter pid=")))
    assert outcome.returncode == 1
    assert outcome.child_status == "stopped_after_supervision_failure"
    assert "RuntimeError" in outcome.error
    assert "Arbre enfant arrete" in contents
    assert not runner.psutil.pid_exists(interpreter_pid)
    assert not runner.psutil.pid_exists(outcome.child_pid)


def test_unconfirmed_cleanup_stops_batch_instead_of_starting_next_country(subprocess_project, monkeypatch):
    original = runner._run_logged

    def fail_de(step, **kwargs):
        if step.zone == "DE":
            raise runner.ChildCleanupError("Owned child could not be stopped")
        return original(step, **kwargs)

    monkeypatch.setattr(runner, "_run_logged", fail_de)
    code, status, calls, output = execute(subprocess_project)
    assert code == 1
    assert [call["stage"] for call in calls] == ["Sync", "Run"]
    assert status["steps"][2]["status"] == "failed"
    assert status["steps"][2]["child_status"] == "cleanup_failed"
    assert all(item["status"] == "skipped" for item in status["steps"][3:])
    assert output.read_text() == "<html>Old report</html>"
