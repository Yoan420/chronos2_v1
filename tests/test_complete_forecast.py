"""Combined launcher contracts; model subprocesses are always replaced by fakes."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import run_complete_forecast as runner


def _value(command, option: str) -> str:
    return command[command.index(option) + 1]


def _values(command, option: str) -> list[str]:
    values = []
    for value in command[command.index(option) + 1:]:
        if value.startswith("--"):
            break
        values.append(value)
    return values


def _plan(tmp_path: Path, **kwargs):
    return runner.build_complete_plan(
        project_root=tmp_path / "project with spaces",
        python_executable=str(tmp_path / "runtime with spaces" / "python.exe"),
        zones=kwargs.pop("zones", ["FR", "DE"]),
        delivery_day=kwargs.pop("delivery_day", "2026-09-09"),
        **kwargs,
    )


def test_plan_runs_regular_both_then_each_nuclear_country_without_side_effects(tmp_path):
    plan = _plan(tmp_path, zones=["NL", "FR", "DE"])
    assert len(plan) == 4
    assert plan[0].zone is None
    assert [step.zone for step in plan[1:]] == ["NL", "FR", "DE"]
    assert _values(plan[0].command, "--zones") == ["NL", "FR", "DE"]
    assert _value(plan[0].command, "--mode").lower() == "both"
    assert Path(plan[0].command[1]).name == "run_multicountry_forecast.py"
    for step in plan[1:]:
        assert Path(step.command[1]).name == "run_nuclear_forecast.py"
        assert _values(step.command, "--zones") == [step.zone]
        assert _value(step.command, "--stage").lower() == "run"
        assert "--mode" not in step.command
    assert not (tmp_path / "project with spaces").exists()


def test_plan_uses_one_delivery_date_and_preserves_runtime_arguments(tmp_path):
    custom = tmp_path / "config with spaces" / "nuclear.yaml"
    plan = _plan(tmp_path, delivery_day="2025-10-27", nuclear_config=custom,
                 device="cpu", threads=3, workers=2, stop_on_error=True)
    for step in plan:
        assert _value(step.command, "--delivery-day") == "2025-10-27"
        assert _value(step.command, "--device") == "cpu"
        assert _value(step.command, "--threads") == "3"
        assert _value(step.command, "--workers") == "2"
        assert step.command[0] == str(tmp_path / "runtime with spaces" / "python.exe")
        assert "--allow-model-download" not in step.command
        assert "--skip-observed-sync" not in step.command
    assert "--stop-on-error" in plan[0].command
    for step in plan[1:]:
        assert _value(step.command, "--config") == str(custom)
        assert "--stop-on-error" not in step.command
        assert "--kalman-config" not in step.command
        assert "--lora-activation-config" not in step.command


def test_plan_deduplicates_selected_countries_without_reordering(tmp_path):
    plan = _plan(tmp_path, zones=["NL", "FR", "NL", "FR"])
    assert _values(plan[0].command, "--zones") == ["NL", "FR"]
    assert [step.zone for step in plan[1:]] == ["NL", "FR"]


@pytest.mark.parametrize("overrides", [
    {"zones": []}, {"zones": ["UK"]}, {"threads": 0}, {"threads": 129},
    {"workers": 0}, {"workers": 129}, {"device": "invalid"},
])
def test_plan_rejects_invalid_arguments_before_any_execution(tmp_path, overrides):
    with pytest.raises(ValueError):
        _plan(tmp_path, **overrides)


def _execution_steps():
    return (
        runner.RunStep("regular", None, ("python.exe", "normal.py", "--zones", "FR", "DE")),
        runner.RunStep("nuclear", "FR", ("python.exe", "nuclear.py", "--zones", "FR")),
        runner.RunStep("nuclear", "DE", ("python.exe", "nuclear.py", "--zones", "DE")),
    )


@pytest.mark.parametrize("codes,expected", [([0, 0, 0], 0), ([2, 0, 0], 1), ([0, 2, 0], 1), ([0, 0, 3], 1)])
def test_execute_is_sequential_shell_free_and_never_masks_prior_failure(tmp_path, monkeypatch, codes, expected):
    steps = _execution_steps()
    calls = []

    def execute(command, **kwargs):
        code = codes[len(calls)]
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    assert runner.execute_complete_plan(steps, project_root=tmp_path) == expected
    assert [command for command, _ in calls] == [list(step.command) for step in steps]
    assert all(kwargs == {"check": False, "cwd": tmp_path, "shell": False} for _, kwargs in calls)


@pytest.mark.parametrize("codes,executed", [([2], 1), ([0, 2], 2)])
def test_stop_on_error_does_not_start_remaining_stages(tmp_path, monkeypatch, codes, executed):
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=codes[len(calls) - 1])

    monkeypatch.setattr(runner.subprocess, "run", execute)
    assert runner.execute_complete_plan(_execution_steps(), project_root=tmp_path, stop_on_error=True) == 1
    assert len(calls) == executed


@pytest.mark.parametrize("stop_on_error,expected_calls", [(False, 3), (True, 1)])
def test_os_error_is_visible_failure_with_configured_continuation(tmp_path, monkeypatch, stop_on_error, expected_calls):
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise OSError("Runtime inaccessible")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    assert runner.execute_complete_plan(_execution_steps(), project_root=tmp_path,
                                        stop_on_error=stop_on_error) == 1
    assert len(calls) == expected_calls


def test_keyboard_interrupt_stops_entire_command_instead_of_starting_nuclear(tmp_path, monkeypatch):
    calls = []

    def interrupt(command, **kwargs):
        calls.append(command)
        raise KeyboardInterrupt

    monkeypatch.setattr(runner.subprocess, "run", interrupt)
    assert runner.execute_complete_plan(_execution_steps(), project_root=tmp_path) == 130
    assert len(calls) == 1


@pytest.mark.parametrize("code", [-2, 130, 3221225786, -1073741510])
def test_child_interrupt_exit_code_stops_all_remaining_stages(tmp_path, monkeypatch, code):
    calls = []

    def interrupt(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(runner.subprocess, "run", interrupt)
    assert runner.execute_complete_plan(_execution_steps(), project_root=tmp_path) == 130
    assert len(calls) == 1


@pytest.mark.parametrize("requested,selected", [
    (None, "2026-09-09"), ("2025-03-31", "2025-03-31"), ("2025-10-27", "2025-10-27"),
])
def test_main_pins_civil_delivery_date_once_for_entire_batch(monkeypatch, requested, selected):
    import run_nuclear_forecast as nuclear

    date_calls, validations, executions = [], [], []

    def date_once(value):
        date_calls.append(value)
        assert len(date_calls) == 1, "Date must not be recomputed between long-running stages"
        return pd.Timestamp(selected)

    def execute(plan, **kwargs):
        assert len(validations) == 1
        executions.append((plan, kwargs))
        return 0

    monkeypatch.setattr(nuclear, "delivery_date", date_once)
    monkeypatch.setattr(runner, "validate_complete_inputs", lambda **kwargs: validations.append(kwargs))
    monkeypatch.setattr(runner, "execute_complete_plan", execute)
    args = ["--zones", "FR", "DE", "--stop-on-error"]
    if requested is not None:
        args.extend(["--delivery-day", requested])
    assert runner.main(args) == 0
    assert date_calls == [requested]
    assert len(executions) == 1
    assert all(_value(step.command, "--delivery-day") == selected for step in executions[0][0])
    assert executions[0][1]["stop_on_error"] is True


def test_main_dry_run_validates_but_never_starts_pipeline(monkeypatch, capsys):
    import run_nuclear_forecast as nuclear

    checked = []
    monkeypatch.setattr(nuclear, "delivery_date", lambda value: pd.Timestamp("2026-09-09"))
    monkeypatch.setattr(runner, "validate_complete_inputs", lambda **kwargs: checked.append(kwargs))
    monkeypatch.setattr(runner, "execute_complete_plan", lambda *args, **kwargs: pytest.fail("Dry run must not execute"))
    assert runner.main(["--zones", "FR", "DE", "--dry-run"]) == 0
    assert len(checked) == 1
    output = capsys.readouterr().out
    assert "run_multicountry_forecast.py" in output
    assert "run_nuclear_forecast.py" in output
    assert "730" in output


@pytest.mark.parametrize("failure_stage", ["cutoff", "inputs"])
def test_main_preflight_failure_never_starts_half_of_batch(monkeypatch, failure_stage):
    import run_nuclear_forecast as nuclear

    def fail(*args, **kwargs):
        raise ValueError("preflight deliberately refused")

    monkeypatch.setattr(nuclear, "delivery_date", fail if failure_stage == "cutoff"
                        else lambda value: pd.Timestamp("2026-09-09"))
    monkeypatch.setattr(runner, "validate_complete_inputs", fail)
    monkeypatch.setattr(runner, "execute_complete_plan", lambda *args, **kwargs: pytest.fail("No partial pipeline start"))
    assert runner.main(["--zones", "FR"]) == 2


def _preflight_fixture(tmp_path, monkeypatch):
    import run_nuclear_forecast as nuclear

    project = tmp_path / "project"
    project.mkdir()
    for name in ("run_multicountry_forecast.py", "run_nuclear_forecast.py", "base_fr.yaml", "base_de.yaml",
                 "kalman.yaml", "activation.yaml"):
        (project / name).write_text("fixture", encoding="utf-8")
    settings = {"project_root": project.resolve(), "zone_configs": {"FR": "base_fr.yaml", "DE": "base_de.yaml"},
                "kalman_config": project / "kalman.yaml", "lora_activation_config": project / "activation.yaml"}
    monkeypatch.setattr(nuclear, "load_settings", lambda path: settings)
    checks = []
    monkeypatch.setattr(nuclear, "check_lora_inactive", lambda settings, zones: checks.append(zones))
    return project, settings, checks


def test_preflight_checks_lora_schema_without_starting_or_mutating_sources(tmp_path, monkeypatch):
    project, _, checks = _preflight_fixture(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for path in project.iterdir()}
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("Preflight is read only"))
    runner.validate_complete_inputs(project_root=project, zones=["FR", "DE"], nuclear_config=project / "config.yaml")
    assert checks == [["FR", "DE"]]
    assert {path: path.read_bytes() for path in project.iterdir()} == before


@pytest.mark.parametrize("failure", ["root", "zone", "base", "launcher", "kalman_config", "lora_activation_config"])
def test_preflight_rejects_incompatible_layout_before_lora_or_model_work(tmp_path, monkeypatch, failure):
    project, settings, checks = _preflight_fixture(tmp_path, monkeypatch)
    if failure == "root":
        settings["project_root"] = tmp_path / "other"
    elif failure == "zone":
        del settings["zone_configs"]["DE"]
    elif failure == "base":
        settings["zone_configs"]["DE"] = "absent_base.yaml"
    elif failure in {"kalman_config", "lora_activation_config"}:
        settings[failure] = project / "absent_settings.yaml"
    else:
        (project / "run_multicountry_forecast.py").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.validate_complete_inputs(project_root=project, zones=["FR", "DE"], nuclear_config=project / "config.yaml")
    assert not checks


def _powershell(arguments: str):
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    if executable is None:
        pytest.skip("Windows PowerShell unavailable")
    project = Path(__file__).resolve().parents[1]
    script = str(project / "Forecast.ps1").replace("'", "''")
    python = sys.executable.replace("'", "''")
    return subprocess.run(
        [executable, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
         f"& '{script}' {arguments} -PythonExecutable '{python}' -DryRun"],
        cwd=project, check=False, capture_output=True, text=True, encoding="utf-8",
    )


def _powershell_argv(arguments: str):
    completed = _powershell(arguments)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    prefix = "Commande (argv, shell=False): "
    line = next(line for line in completed.stdout.splitlines() if line.startswith(prefix))
    return json.loads(line[len(prefix):])


def test_powershell_complete_builds_one_combined_launcher_argv():
    argv = _powershell_argv("-Action Run -Mode Complete -Countries FR,DE,BE,NL "
                           "-DeliveryDay 2026-09-09 -Device cpu -Threads 3 -Workers 2 -StopOnError")
    assert Path(argv[1]).name == "run_complete_forecast.py"
    assert _values(argv, "--zones") == ["FR", "DE", "BE", "NL"]
    assert _value(argv, "--delivery-day") == "2026-09-09"
    assert _value(argv, "--device") == "cpu"
    assert _value(argv, "--threads") == "3"
    assert _value(argv, "--workers") == "2"
    assert "--stop-on-error" in argv


@pytest.mark.parametrize("arguments", [
    "-Action Backfill -Mode Complete", "-Action App -Mode Complete",
    "-Action Run -Mode Complete -WithNuclear",
    "-Action Run -Mode Complete -NuclearStage Report",
    "-Action Run -Mode Complete -SkipObservedSync",
    "-Action Run -Mode Complete -KalmanConfig config/kalman_operational.yaml",
    "-Action Run -Mode Complete -LoraActivationConfig config/chronos2_exogenous_activation_v1.yaml",
    "-Action Run -Mode Complete -AllowModelDownload",
    "-Action Run -Mode Complete -ResidualLoadSource Chronos2",
])
def test_powershell_complete_rejects_ambiguous_or_incompatible_flags(arguments):
    completed = _powershell(arguments)
    assert completed.returncode != 0


@pytest.mark.parametrize("arguments,script,mode", [
    ("-Action Run -Mode Both -Countries FR", "run_multicountry_forecast.py", "both"),
    ("-Action Run -Mode All -Countries FR", "run_multicountry_forecast.py", "all"),
    ("-Action Run -Mode Both -Countries FR -WithNuclear", "run_nuclear_forecast.py", None),
])
def test_existing_powershell_mode_routing_is_unchanged(arguments, script, mode):
    argv = _powershell_argv(arguments)
    assert Path(argv[1]).name == script
    if mode is not None:
        assert _value(argv, "--mode").lower() == mode
    else:
        assert _value(argv, "--stage").lower() == "run"
