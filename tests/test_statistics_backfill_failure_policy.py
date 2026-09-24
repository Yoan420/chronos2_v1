from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import run_statistics_backfill as launcher
from chronos2_hourly.app_service import ForecastSkip, ZoneStatus


def _status(tmp_path: Path, zone: str) -> ZoneStatus:
    return ZoneStatus(
        code=zone,
        timezone={"FR": "Europe/Paris", "DE": "Europe/Berlin"}[zone],
        enabled=True,
        production_ready=True,
        ready=True,
        runner=tmp_path / "runner.py",
        live_config=tmp_path / f"{zone.lower()}_live.yaml",
        checks=("ok",),
        blockers=(),
    )


def _contract(tmp_path: Path, zone: str) -> launcher.ZoneHistoryContract:
    return launcher.ZoneHistoryContract(
        zone=zone,
        timezone=_status(tmp_path, zone).timezone,
        forecast_origin_timezone="Europe/Paris",
        forecast_origin_local_time="08:00",
        target_series=f"target.{zone.lower()}",
        prediction_mode="mkonline_blend" if zone == "FR" else "autonomous_only",
        candidate_model="mkonline_blend" if zone == "FR" else "residual_corrected",
        forecast_name=f"forecast_hourly_{zone.lower()}.csv",
        output_root=tmp_path / "runs" / zone.lower(),
        sealed_benchmark_run=tmp_path / "sealed" / zone.lower(),
    )


def _install_failure_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[dict[str, tuple[date, ...]], list[tuple[str, date]]]:
    gaps = {
        "FR": (date(2026, 8, 16), date(2026, 8, 17)),
        "DE": (date(2026, 8, 16),),
    }
    completed: dict[str, set[date]] = {zone: set() for zone in gaps}
    launched: list[tuple[str, date]] = []
    monkeypatch.setattr(
        launcher,
        "_status_map",
        lambda _registry, zones: {zone: _status(tmp_path, zone) for zone in zones},
    )
    monkeypatch.setattr(
        launcher,
        "load_zone_history_contract",
        lambda status, **_kwargs: _contract(tmp_path, status.code),
    )
    monkeypatch.setattr(
        launcher,
        "_latest_issued_day",
        lambda _status, **_kwargs: date(2026, 8, 20),
    )
    monkeypatch.setattr(
        launcher,
        "plan_zone_backfill",
        lambda contract, **_kwargs: tuple(
            day for day in gaps[contract.zone] if day not in completed[contract.zone]
        ),
    )
    monkeypatch.setattr(
        launcher,
        "build_dispatch_command",
        lambda status, **kwargs: [
            str(tmp_path / "python.exe"),
            "dispatcher.py",
            "--zone",
            status.code,
            "--delivery-day",
            kwargs["delivery_day"].isoformat(),
            "--pit-replay",
        ],
    )

    def fake_launch(*, zone: str, delivery_day: date, **_kwargs):
        launched.append((zone, delivery_day))
        if zone == "FR" and delivery_day == date(2026, 8, 16):
            raise RuntimeError("fraicheur PIT depassee")
        return ForecastSkip(
            zone=zone,
            delivery_day=delivery_day.isoformat(),
            archive_path=tmp_path / "replays" / f"{zone}_{delivery_day}",
        )

    def fake_validate(status: ZoneStatus, *, delivery_day: date, **_kwargs):
        completed[status.code].add(delivery_day)
        return tmp_path / "replays" / f"{status.code}_{delivery_day}"

    monkeypatch.setattr(launcher, "launch_zone_forecast", fake_launch)
    monkeypatch.setattr(launcher, "validate_existing_forecast_archive", fake_validate)
    return gaps, launched


def test_failure_stops_dependent_days_for_that_zone_but_continues_next_zone_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "zones.yaml"
    registry.write_text("schema_version: 1\nzones: {}\n", encoding="utf-8")
    executable = tmp_path / "python.exe"
    executable.write_bytes(b"fake")
    gaps, launched = _install_failure_harness(monkeypatch, tmp_path)

    result = launcher.run_statistics_backfill(
        zones=("FR", "DE"),
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        log_dir=tmp_path / "logs",
    )

    # D+1 Statistics depends on the successfully published D archive.  Once
    # FR D16 fails, trying FR D17 in the same batch is therefore invalid.
    assert launched == [
        ("FR", date(2026, 8, 16)),
        ("DE", date(2026, 8, 16)),
    ]
    assert [(item.zone, item.delivery_day, item.state) for item in result.replay_results] == [
        ("FR", date(2026, 8, 16), "failed"),
        ("DE", date(2026, 8, 16), "skipped"),
    ]
    assert result.remaining_days == {"FR": gaps["FR"], "DE": ()}
    assert not result.ok


def test_global_stop_on_error_also_prevents_later_zones(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "zones.yaml"
    registry.write_text("schema_version: 1\nzones: {}\n", encoding="utf-8")
    executable = tmp_path / "python.exe"
    executable.write_bytes(b"fake")
    gaps, launched = _install_failure_harness(monkeypatch, tmp_path)

    result = launcher.run_statistics_backfill(
        zones=("FR", "DE"),
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        log_dir=tmp_path / "logs",
        stop_on_error=True,
    )

    assert launched == [("FR", date(2026, 8, 16))]
    assert [item.state for item in result.replay_results] == ["failed"]
    assert result.remaining_days == gaps
    assert not result.ok


def test_incomplete_summary_is_explicit_and_main_never_starts_live_refresh(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    failed_day = date(2026, 8, 16)
    dependent_day = date(2026, 8, 17)
    result = launcher.BackfillResult(
        source_delivery_days={"FR": date(2026, 8, 20)},
        planned_days={"FR": (failed_day, dependent_day)},
        replay_results=(
            launcher.ReplayResult(
                zone="FR",
                delivery_day=failed_day,
                cutoff_local="2026-08-15T08:00:00+02:00",
                state="failed",
                return_code=1,
                message="fraicheur PIT depassee\ndetail technique",
            ),
        ),
        remaining_days={"FR": (failed_day, dependent_day)},
    )
    live_calls: list[dict] = []
    monkeypatch.setattr(launcher, "run_statistics_backfill", lambda **_kwargs: result)
    monkeypatch.setattr(
        launcher,
        "run_forecast_batch",
        lambda **kwargs: live_calls.append(kwargs)
        or pytest.fail("un backfill incomplet ne doit pas rafraichir les rapports live"),
    )

    exit_code = launcher.main(
        [
            "--zones",
            "FR",
            "--then-run-live",
            "--live-delivery-day",
            "2026-08-21",
        ]
    )
    output = capsys.readouterr().out.splitlines()

    assert exit_code == 1
    assert live_calls == []
    assert "FR: source live 2026-08-20 | replays planifies: 2026-08-16, 2026-08-17" in output
    assert "FR 2026-08-16 | FAILED   | fraicheur PIT depassee" in output
    assert "FR: RESTE A RECONSTRUIRE: 2026-08-16, 2026-08-17" in output

