from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_multicountry_forecast as launcher
from chronos2_hourly.app_service import ForecastProcess, ForecastSkip, ZoneStatus


def _status(tmp_path: Path, zone: str, *, ready: bool = True) -> ZoneStatus:
    return ZoneStatus(
        code=zone,
        timezone={
            "FR": "Europe/Paris",
            "DE": "Europe/Berlin",
            "BE": "Europe/Brussels",
            "NL": "Europe/Amsterdam",
            "ES": "Europe/Madrid",
        }[zone],
        enabled=True,
        production_ready=True,
        ready=ready,
        runner=tmp_path / "runner.py",
        live_config=tmp_path / f"{zone.lower()}_live.yaml",
        checks=("ok",),
        blockers=() if ready else ("bundle incomplet",),
    )


def _runtime_paths(tmp_path: Path) -> tuple[Path, Path]:
    registry = tmp_path / "zones.yaml"
    registry.write_text("schema_version: 1\nzones: {}\n", encoding="utf-8")
    executable = tmp_path / "python.exe"
    executable.write_bytes(b"fake executable")
    return registry, executable


def _archive(tmp_path: Path, zone: str) -> Path:
    archive = tmp_path / f"{zone.lower()}_day_ahead_2026-08-20"
    archive.mkdir()
    (archive / "run_manifest.json").write_text(
        json.dumps({"reporting_status": "complete"}), encoding="utf-8"
    )
    (archive / "live_run_summary.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    (archive / f"chronos2_{zone.lower()}_2026-08-20.html").write_text(
        "<!doctype html><title>Forecast detaille</title>", encoding="utf-8"
    )
    return archive


def _patch_statuses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    ready: dict[str, bool] | None = None,
) -> None:
    readiness = ready or {}

    def fake_inspect(_registry: Path, *, zones: tuple[str, ...]):
        return [
            _status(tmp_path, zone, ready=readiness.get(zone, True))
            for zone in zones
        ]

    monkeypatch.setattr(launcher, "inspect_zone_statuses", fake_inspect)


def test_normalize_zones_preserves_order_and_rejects_ambiguous_input() -> None:
    assert launcher.normalize_zones([" fr ", "DE", "be", "Nl", "ES"]) == (
        "FR",
        "DE",
        "BE",
        "NL",
        "ES",
    )

    with pytest.raises(ValueError, match="duplique"):
        launcher.normalize_zones(["FR", "fr"])
    with pytest.raises(ValueError, match="au moins un"):
        launcher.normalize_zones([])
    with pytest.raises(ValueError, match="disponible"):
        launcher.normalize_zones(["GB"])
    with pytest.raises(ValueError, match="inconnu|supporte"):
        launcher.normalize_zones(["FR; Remove-Item C:\\data"])


def test_find_detailed_html_report_accepts_one_complete_non_empty_report(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path, "FR")

    report = launcher.find_detailed_html_report(archive)

    assert report == (archive / "chronos2_fr_2026-08-20.html").resolve()


@pytest.mark.parametrize("defect", ("missing", "multiple", "empty", "degraded"))
def test_find_detailed_html_report_fails_closed(
    tmp_path: Path,
    defect: str,
) -> None:
    archive = _archive(tmp_path, "DE")
    report = archive / "chronos2_de_2026-08-20.html"
    if defect == "missing":
        report.unlink()
    elif defect == "multiple":
        (archive / "other.html").write_text("<html>other</html>", encoding="utf-8")
    elif defect == "empty":
        report.write_bytes(b"")
    else:
        (archive / "reporting_errors.json").write_text("{}", encoding="utf-8")

    with pytest.raises(launcher.DetailedReportError):
        launcher.find_detailed_html_report(archive)


def test_dry_run_builds_one_argv_per_country_without_launching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    _patch_statuses(monkeypatch, tmp_path)
    launch = SimpleNamespace(calls=0)

    def forbidden_launch(**_kwargs):
        launch.calls += 1
        raise AssertionError("Un dry-run ne doit lancer aucun processus")

    def fake_build(status: ZoneStatus, **kwargs):
        assert kwargs["local_files_only"] is True
        return [
            str(executable),
            str(tmp_path / "run_mkonline_live_zone.py"),
            "--zone",
            status.code,
            "--delivery-day",
            "2026-08-20",
        ]

    monkeypatch.setattr(launcher, "launch_zone_forecast", forbidden_launch)
    monkeypatch.setattr(launcher, "build_dispatch_command", fake_build)

    batch = launcher.run_forecast_batch(
        zones=("BE", "FR"),
        delivery_day="2026-08-20",
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        log_dir=tmp_path / "logs",
        dry_run=True,
    )

    assert batch.ok
    assert batch.zones == ("BE", "FR")
    assert [item.state for item in batch.results] == ["dry_run", "dry_run"]
    assert [item.command[-3] for item in batch.results] == ["BE", "FR"]
    assert all(isinstance(argument, str) for item in batch.results for argument in item.command)
    assert launch.calls == 0


def test_existing_archive_is_skipped_only_after_detailed_report_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    archive = _archive(tmp_path, "NL")
    _patch_statuses(monkeypatch, tmp_path)
    monkeypatch.setattr(
        launcher,
        "launch_zone_forecast",
        lambda **_kwargs: ForecastSkip(
            zone="NL", delivery_day="2026-08-20", archive_path=archive
        ),
    )
    monkeypatch.setattr(
        launcher,
        "validate_existing_forecast_archive",
        lambda *_args, **_kwargs: archive,
    )

    batch = launcher.run_forecast_batch(
        zones=("NL",),
        delivery_day="2026-08-20",
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        log_dir=tmp_path / "logs",
    )

    assert batch.ok
    assert batch.results[0].state == "skipped"
    assert batch.results[0].report_path == next(archive.glob("*.html")).resolve()


def test_successful_process_keeps_exact_argv_and_report_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    archive = _archive(tmp_path, "ES")
    log = tmp_path / "es.log"
    log.write_text("ok", encoding="utf-8")
    _patch_statuses(monkeypatch, tmp_path)
    process = SimpleNamespace(wait=lambda: 0, poll=lambda: 0)
    command = (str(executable), "runner.py", "--zone", "ES")
    handle = ForecastProcess(
        zone="ES",
        command=command,
        log_path=log,
        started_at=datetime.now(timezone.utc),
        process=process,
    )
    monkeypatch.setattr(launcher, "launch_zone_forecast", lambda **_kwargs: handle)
    monkeypatch.setattr(
        launcher,
        "validate_existing_forecast_archive",
        lambda *_args, **_kwargs: archive,
    )

    batch = launcher.run_forecast_batch(
        zones=("ES",),
        delivery_day="2026-08-20",
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        log_dir=tmp_path / "logs",
    )

    result = batch.results[0]
    assert batch.ok
    assert result.state == "success"
    assert result.return_code == 0
    assert result.command == command
    assert result.report_path is not None


def test_failed_process_preserves_runner_exit_code_and_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    log = tmp_path / "fr_failed.log"
    log.write_text("erreur runner explicite", encoding="utf-8")
    _patch_statuses(monkeypatch, tmp_path)
    process = SimpleNamespace(wait=lambda: 7, poll=lambda: 7)
    command = (str(executable), "runner.py", "--zone", "FR")
    handle = ForecastProcess(
        zone="FR",
        command=command,
        log_path=log,
        started_at=datetime.now(timezone.utc),
        process=process,
    )
    monkeypatch.setattr(launcher, "launch_zone_forecast", lambda **_kwargs: handle)

    batch = launcher.run_forecast_batch(
        zones=("FR",),
        delivery_day="2026-08-20",
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        log_dir=tmp_path / "logs",
    )

    result = batch.results[0]
    assert not batch.ok
    assert result.state == "failed"
    assert result.return_code == 7
    assert result.command == command
    assert "erreur runner explicite" in result.message


def test_one_country_failure_does_not_hide_later_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    de_archive = _archive(tmp_path, "DE")
    _patch_statuses(monkeypatch, tmp_path)
    launched: list[str] = []

    def fake_launch(*, zone: str, **_kwargs):
        launched.append(zone)
        if zone == "FR":
            raise RuntimeError("echec FR explicite")
        return ForecastSkip(
            zone="DE", delivery_day="2026-08-20", archive_path=de_archive
        )

    monkeypatch.setattr(launcher, "launch_zone_forecast", fake_launch)
    monkeypatch.setattr(
        launcher,
        "validate_existing_forecast_archive",
        lambda *_args, **_kwargs: de_archive,
    )

    batch = launcher.run_forecast_batch(
        zones=("FR", "DE"),
        delivery_day="2026-08-20",
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        log_dir=tmp_path / "logs",
        stop_on_error=False,
    )

    assert launched == ["FR", "DE"]
    assert [item.state for item in batch.results] == ["failed", "skipped"]
    assert not batch.ok


def test_stop_on_error_stops_before_next_country(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    _patch_statuses(monkeypatch, tmp_path)
    launched: list[str] = []

    def fail(*, zone: str, **_kwargs):
        launched.append(zone)
        raise RuntimeError("runner indisponible")

    monkeypatch.setattr(launcher, "launch_zone_forecast", fail)

    batch = launcher.run_forecast_batch(
        zones=("FR", "BE"),
        delivery_day="2026-08-20",
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        log_dir=tmp_path / "logs",
        stop_on_error=True,
    )

    assert launched == ["FR"]
    assert len(batch.results) == 1
    assert not batch.ok


def test_degraded_report_turns_a_successful_forecast_into_batch_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    archive = _archive(tmp_path, "BE")
    (archive / "reporting_errors.json").write_text("{}", encoding="utf-8")
    _patch_statuses(monkeypatch, tmp_path)
    monkeypatch.setattr(
        launcher,
        "launch_zone_forecast",
        lambda **_kwargs: ForecastSkip(
            zone="BE", delivery_day="2026-08-20", archive_path=archive
        ),
    )
    monkeypatch.setattr(
        launcher,
        "validate_existing_forecast_archive",
        lambda *_args, **_kwargs: archive,
    )

    batch = launcher.run_forecast_batch(
        zones=("BE",),
        delivery_day="2026-08-20",
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        log_dir=tmp_path / "logs",
    )

    assert not batch.ok
    assert batch.results[0].state == "failed"
    assert "degrade" in batch.results[0].message


def test_main_exit_codes_distinguish_success_batch_failure_and_input_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "print_batch_summary", lambda _batch: None)
    ok = launcher.BatchForecastResult(
        delivery_day="2026-08-20",
        zones=("FR",),
        results=(
            launcher.BatchZoneResult(
                zone="FR",
                delivery_day="2026-08-20",
                state="dry_run",
                return_code=0,
                message="ok",
            ),
        ),
    )
    failed = launcher.BatchForecastResult(
        delivery_day="2026-08-20",
        zones=("FR",),
        results=(
            launcher.BatchZoneResult(
                zone="FR",
                delivery_day="2026-08-20",
                state="failed",
                return_code=1,
                message="ko",
            ),
        ),
    )

    monkeypatch.setattr(launcher, "run_forecast_batch", lambda **_kwargs: ok)
    assert launcher.main(["--zones", "FR", "--dry-run"]) == 0
    monkeypatch.setattr(launcher, "run_forecast_batch", lambda **_kwargs: failed)
    assert launcher.main(["--zones", "FR", "--dry-run"]) == 1

    def invalid(**_kwargs):
        raise ValueError("selection invalide")

    monkeypatch.setattr(launcher, "run_forecast_batch", invalid)
    assert launcher.main(["--zones", "FR", "--dry-run"]) == 2


def test_powershell_wrapper_uses_argument_array_and_propagates_exit_code() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "Start-MultiCountryForecast.ps1"
    ).read_text(encoding="utf-8")
    lowered = script.casefold()

    assert "& $PythonExe @Arguments" in script
    assert "exit $LASTEXITCODE" in script
    assert "'--zones'" in script
    assert "invoke-expression" not in lowered
    assert "cmd.exe" not in lowered
    assert "-command" not in lowered
    assert "shell=true" not in lowered
