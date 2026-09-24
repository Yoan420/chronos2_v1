from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType

import pytest

import run_model_storm_report as runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "ModelStorm.ps1"
POWERSHELL = shutil.which("powershell.exe") or shutil.which("pwsh") or shutil.which("powershell")


def _quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _powershell(command: str, *, cwd: Path) -> subprocess.CompletedProcess[str]:
    if POWERSHELL is None:
        pytest.skip("PowerShell is unavailable")
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
         command + "; exit $LASTEXITCODE"],
        cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )


@pytest.fixture
def local_launcher(tmp_path: Path) -> Path:
    root = tmp_path / "project with spaces"
    root.mkdir()
    shutil.copyfile(LAUNCHER, root / LAUNCHER.name)
    (root / "run_model_storm_report.py").write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--delivery-day')\n"
        "p.add_argument('--output')\n"
        "p.add_argument('--nuclear-root')\n"
        "p.add_argument('--skip-vps-sync', action='store_true')\n"
        "args=p.parse_args()\n"
        "Path(__file__).with_name('invocation.json').write_text(json.dumps(vars(args)))\n"
        "output=Path(args.output)\n"
        "output.parent.mkdir(parents=True, exist_ok=True)\n"
        "output.write_text('<html>Model / Storm</html>')\n",
        encoding="utf-8",
    )
    return root / LAUNCHER.name


def test_powershell_dry_run_is_read_only_and_resolves_own_root(local_launcher: Path, tmp_path: Path) -> None:
    before = sorted(str(path) for path in tmp_path.rglob("*"))
    completed = _powershell(
        f"& {_quote(local_launcher)} -PythonExecutable {_quote(sys.executable)} "
        "-DeliveryDay 2026-09-10 -OutputPath 'reports with spaces/report.html' "
        "-NuclearRoot 'local results' -DryRun",
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    prefix = "Commande (argv, shell=False): "
    line = next(line for line in completed.stdout.splitlines() if line.startswith(prefix))
    argv = json.loads(line[len(prefix):])
    root = local_launcher.parent
    assert argv == [
        str(Path(sys.executable).resolve()), str(root / "run_model_storm_report.py"),
        "--delivery-day", "2026-09-10", "--output", str(root / "reports with spaces/report.html"),
        "--nuclear-root", str(root / "local results"),
    ]
    assert sorted(str(path) for path in tmp_path.rglob("*")) == before


def test_powershell_no_open_generates_only_requested_html(local_launcher: Path, tmp_path: Path) -> None:
    completed = _powershell(
        "function Invoke-Item { throw 'Opening must be disabled' }; "
        f"& {_quote(local_launcher)} -PythonExecutable {_quote(sys.executable)} "
        "-DeliveryDay 2026-09-10 -NoOpen",
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = local_launcher.parent / "runs/reports/model_storm/CWE_Model_Storm_2026-09-10.html"
    assert output.is_file()
    assert list(local_launcher.parent.rglob("*.html")) == [output]
    invocation = json.loads((local_launcher.parent / "invocation.json").read_text())
    assert invocation["output"] == str(output)


def test_powershell_forwards_skip_vps_sync(local_launcher: Path, tmp_path: Path) -> None:
    completed = _powershell(
        f"& {_quote(local_launcher)} -PythonExecutable {_quote(sys.executable)} "
        "-DeliveryDay 2026-09-10 -SkipVpsSync -NoOpen",
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    invocation = json.loads((local_launcher.parent / "invocation.json").read_text())
    assert invocation["skip_vps_sync"] is True


def test_powershell_skip_vps_dry_run_has_no_side_effect(local_launcher: Path, tmp_path: Path) -> None:
    before = sorted(str(path) for path in tmp_path.rglob("*"))
    completed = _powershell(
        f"& {_quote(local_launcher)} -PythonExecutable {_quote(sys.executable)} "
        "-DeliveryDay 2026-09-10 -SkipVpsSync -DryRun",
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    prefix = "Commande (argv, shell=False): "
    argv = json.loads(next(line[len(prefix):] for line in completed.stdout.splitlines() if line.startswith(prefix)))
    assert "--skip-vps-sync" in argv
    assert sorted(str(path) for path in tmp_path.rglob("*")) == before


def test_powershell_opens_exactly_the_output(local_launcher: Path, tmp_path: Path) -> None:
    marker = tmp_path / "opened.txt"
    completed = _powershell(
        "function Invoke-Item { param([string]$LiteralPath) "
        f"Add-Content -LiteralPath {_quote(marker)} -Value $LiteralPath }}; "
        f"& {_quote(local_launcher)} -PythonExecutable {_quote(sys.executable)} "
        "-DeliveryDay 2026-09-10",
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    expected = local_launcher.parent / "runs/reports/model_storm/CWE_Model_Storm_2026-09-10.html"
    assert marker.read_text().splitlines() == [str(expected)]


def test_powershell_preserves_python_failure_code(local_launcher: Path, tmp_path: Path) -> None:
    (local_launcher.parent / "run_model_storm_report.py").write_text("raise SystemExit(17)\n")
    completed = _powershell(
        f"& {_quote(local_launcher)} -PythonExecutable {_quote(sys.executable)} "
        "-DeliveryDay 2026-09-10 -NoOpen",
        cwd=tmp_path,
    )
    assert completed.returncode == 17, completed.stdout + completed.stderr
    assert not list(local_launcher.parent.rglob("*.html"))


def test_powershell_rejects_missing_html_after_success(local_launcher: Path, tmp_path: Path) -> None:
    (local_launcher.parent / "run_model_storm_report.py").write_text("raise SystemExit(0)\n")
    completed = _powershell(
        f"& {_quote(local_launcher)} -PythonExecutable {_quote(sys.executable)} "
        "-DeliveryDay 2026-09-10 -NoOpen",
        cwd=tmp_path,
    )
    assert completed.returncode == 2
    assert "Rapport attendu introuvable" in completed.stderr


def test_powershell_invalid_calendar_date_does_not_run_python(local_launcher: Path, tmp_path: Path) -> None:
    completed = _powershell(
        f"& {_quote(local_launcher)} -PythonExecutable {_quote(sys.executable)} "
        "-DeliveryDay 2026-02-30 -NoOpen",
        cwd=tmp_path,
    )
    assert completed.returncode != 0
    assert not (local_launcher.parent / "invocation.json").exists()


def test_powershell_default_date_is_pinned_for_paris(local_launcher: Path, tmp_path: Path) -> None:
    before = runner.delivery_date(None)
    completed = _powershell(
        f"& {_quote(local_launcher)} -PythonExecutable {_quote(sys.executable)} -DryRun",
        cwd=tmp_path,
    )
    after = runner.delivery_date(None)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    prefix = "Commande (argv, shell=False): "
    line = next(line for line in completed.stdout.splitlines() if line.startswith(prefix))
    argv = json.loads(line[len(prefix):])
    day = argv[argv.index("--delivery-day") + 1]
    assert day in {before, after}
    assert argv[argv.index("--output") + 1].endswith(f"CWE_Model_Storm_{day}.html")


@pytest.mark.parametrize("value", ["2026-02-30", "20260910", "2026-W37-4", "", "2026-9-10"])
def test_cli_rejects_noncanonical_or_invalid_dates(value: str) -> None:
    with pytest.raises(ValueError):
        runner.delivery_date(value)


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (datetime(2026, 9, 9, 22, 30, tzinfo=timezone.utc), "2026-09-11"),
        (datetime(2026, 1, 9, 23, 30, tzinfo=timezone.utc), "2026-01-11"),
        (datetime(2026, 3, 28, 23, 30, tzinfo=timezone.utc), "2026-03-30"),
        (datetime(2026, 10, 24, 22, 30, tzinfo=timezone.utc), "2026-10-26"),
    ],
)
def test_cli_default_date_uses_paris_across_midnight_and_dst(now: datetime, expected: str) -> None:
    assert runner.delivery_date(None, now=now) == expected


def _stub_report_modules(monkeypatch: pytest.MonkeyPatch, payload: dict) -> dict:
    calls: dict = {}
    loader = ModuleType("chronos2_hourly.model_storm_data")
    renderer = ModuleType("chronos2_hourly.model_storm_report")
    vps = ModuleType("chronos2_hourly.model_storm_vps")

    def load(root: Path, day: str, *, nuclear_root: Path | None = None) -> dict:
        calls.setdefault("order", []).append("load")
        calls["load"] = (root, day, nuclear_root)
        return payload

    def refresh(data: dict, root: Path) -> dict:
        calls.setdefault("order", []).append("refresh")
        calls["refresh"] = (data, root)
        return data

    def render(data: dict, output: Path) -> Path:
        calls.setdefault("order", []).append("render")
        calls["render"] = (data, output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("<html>one report</html>")
        return output

    loader.load_model_storm_payload = load
    renderer.render_model_storm_report = render
    vps.refresh_vps_payload = refresh
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    monkeypatch.setitem(sys.modules, renderer.__name__, renderer)
    monkeypatch.setitem(sys.modules, vps.__name__, vps)
    return calls


def test_cli_assembles_partial_country_results_at_project_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project = tmp_path / "project with spaces"
    project.mkdir()
    monkeypatch.setattr(runner, "PROJECT_ROOT", project)
    monkeypatch.chdir(tmp_path)
    payload = {"delivery_day": "2026-09-10", "has_data": True,
               "zones": [{"zone": "DE", "rows": []}, {"zone": "BE", "rows": [{"model": 123.0}]}]}
    calls = _stub_report_modules(monkeypatch, payload)
    assert runner.main(["--delivery-day", "2026-09-10", "--nuclear-root", "saved results"]) == 0
    output = project / "runs/reports/model_storm/CWE_Model_Storm_2026-09-10.html"
    assert calls["load"] == (project, "2026-09-10", project / "saved results")
    assert calls["render"] == (payload, output)
    assert "refresh" not in calls
    assert calls["order"] == ["load", "render"]
    assert list(tmp_path.rglob("*.html")) == [output]


def test_cli_honors_absolute_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path / "project")
    output = tmp_path / "reports with spaces/report.html"
    calls = _stub_report_modules(monkeypatch, {"has_data": True})
    assert runner.main(["--delivery-day", "2026-09-10", "--output", str(output)]) == 0
    assert calls["render"][1] == output
    assert calls["load"][2] is None


def test_cli_empty_delivery_does_not_create_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    calls = _stub_report_modules(monkeypatch, {"has_data": False, "zones": []})
    assert runner.main(["--delivery-day", "2026-09-10"]) == 2
    assert "render" not in calls
    assert "refresh" not in calls
    assert "Aucun resultat local exploitable" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_cli_loader_error_does_not_create_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    calls = _stub_report_modules(monkeypatch, {"has_data": True})

    def fail(*args, **kwargs):
        raise OSError("Resultats locaux illisibles")

    monkeypatch.setattr(sys.modules["chronos2_hourly.model_storm_data"], "load_model_storm_payload", fail)
    assert runner.main(["--delivery-day", "2026-09-10"]) == 2
    assert "render" not in calls
    assert "refresh" not in calls
    assert "Resultats locaux illisibles" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_cli_rejects_non_html_output_before_loading(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    calls = _stub_report_modules(monkeypatch, {"has_data": True})
    assert runner.main(["--delivery-day", "2026-09-10", "--output", "report.csv"]) == 2
    assert not calls


def test_cli_invalid_date_does_not_collect_or_write(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    calls = _stub_report_modules(monkeypatch, {"has_data": True})
    assert runner.main(["--delivery-day", "2026-02-30"]) == 2
    assert not calls
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("compatibility_args", [[], ["--skip-vps-sync"]])
def test_cli_is_always_offline_and_does_not_reuse_stale_vps_rows(monkeypatch, tmp_path, capsys,
                                                              compatibility_args):
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    payload = {"has_data": True, "vps_snapshot": {"directory": "old cache"},
               "zones": [{"zone": "FR", "rows": [{"model": 123.0, "storm": 124.0}],
                          "rolling_history": {"rows": [{"model": -10.0, "model_p10": -30.0,
                                                           "model_p90": 20.0, "observed": -5.0}]},
                          "vps_history": {"source": {"status": "complete"}, "rows": [{"pnl": 123.0}]}}]}
    calls = _stub_report_modules(monkeypatch, payload)
    # Neither the default nor the deprecated compatibility option can import a collector.
    monkeypatch.setitem(sys.modules, "chronos2_hourly.model_storm_vps", None)
    assert runner.main(["--delivery-day", "2026-09-10", *compatibility_args]) == 0
    assert calls["order"] == ["load", "render"]
    rendered = calls["render"][0]
    assert rendered is not payload
    assert "vps_snapshot" not in rendered
    assert "vps_history" not in rendered["zones"][0]
    assert rendered["zones"][0]["rows"] == payload["zones"][0]["rows"]
    assert rendered["zones"][0]["rolling_history"] == payload["zones"][0]["rolling_history"]
    assert payload["zones"][0]["vps_history"]["rows"] == [{"pnl": 123.0}]
    assert [path.name for path in tmp_path.rglob("*") if path.is_file()] == ["CWE_Model_Storm_2026-09-10.html"]
    assert "VPS" not in capsys.readouterr().out


def test_cli_unavailable_vps_client_has_no_effect_on_local_report(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    payload = {"has_data": True, "zones": [{"zone": "FR"}]}
    calls = _stub_report_modules(monkeypatch, payload)
    def unavailable(*args):
        pytest.fail("The local report must never call Saturn VPS")
    monkeypatch.setattr(sys.modules["chronos2_hourly.model_storm_vps"], "refresh_vps_payload", unavailable)
    assert runner.main(["--delivery-day", "2026-09-10"]) == 0
    assert calls["render"][0] == payload
    assert calls["order"] == ["load", "render"]


def test_cli_render_failure_preserves_existing_report(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "old.html"
    output.write_text("old verified report", encoding="utf-8")
    calls = _stub_report_modules(monkeypatch, {"has_data": True})
    def fail(*args):
        raise OSError("Unable to publish local report")
    monkeypatch.setattr(sys.modules["chronos2_hourly.model_storm_report"], "render_model_storm_report", fail)
    assert runner.main(["--delivery-day", "2026-09-10", "--output", str(output)]) == 2
    assert "render" not in calls
    assert output.read_text(encoding="utf-8") == "old verified report"
    assert "Unable to publish local report" in capsys.readouterr().err
