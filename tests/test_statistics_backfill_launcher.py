from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import run_statistics_backfill as launcher
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
        blockers=() if ready else ("contrat scelle invalide",),
    )


def _contract(tmp_path: Path, zone: str) -> launcher.ZoneHistoryContract:
    return launcher.ZoneHistoryContract(
        zone=zone,
        timezone=_status(tmp_path, zone).timezone,
        forecast_origin_timezone="Europe/Paris",
        forecast_origin_local_time="08:00",
        target_series=f"target.{zone.lower()}",
        prediction_mode=(
            "mkonline_blend" if zone in {"FR", "NL"} else "autonomous_only"
        ),
        candidate_model=(
            "mkonline_blend" if zone in {"FR", "NL"} else "residual_corrected"
        ),
        forecast_name=f"forecast_hourly_{zone.lower()}.csv",
        output_root=tmp_path / "runs" / zone.lower(),
        sealed_benchmark_run=tmp_path / "sealed" / zone.lower(),
    )


def _runtime_paths(tmp_path: Path) -> tuple[Path, Path]:
    registry = tmp_path / "zones.yaml"
    registry.write_text("schema_version: 1\nzones: {}\n", encoding="utf-8")
    executable = tmp_path / "python.exe"
    executable.write_bytes(b"fake executable")
    return registry, executable


def _patch_planning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    gaps: dict[str, tuple[date, ...]],
    *,
    ready: dict[str, bool] | None = None,
) -> dict[str, bool]:
    readiness = ready or {}
    completed = {zone: False for zone in gaps}
    monkeypatch.setattr(
        launcher,
        "_status_map",
        lambda _registry, zones: {
            zone: _status(tmp_path, zone, ready=readiness.get(zone, True))
            for zone in zones
        },
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

    def fake_plan(contract, *, source_delivery_day):
        assert source_delivery_day == date(2026, 8, 20)
        return () if completed[contract.zone] else gaps[contract.zone]

    monkeypatch.setattr(launcher, "plan_zone_backfill", fake_plan)
    return completed


@pytest.mark.parametrize(
    ("delivery_day", "expected"),
    (
        (date(2026, 8, 16), "2026-08-15T08:00:00+02:00"),
        # Delivery DST days still use the physical civil D-1 08:00 cutoff.
        (date(2026, 3, 29), "2026-03-28T08:00:00+01:00"),
        (date(2026, 10, 25), "2026-10-24T08:00:00+02:00"),
    ),
)
def test_replay_cutoff_is_exact_civil_d_minus_one_eight_across_dst(
    delivery_day: date,
    expected: str,
) -> None:
    cutoff = launcher.replay_cutoff_local(
        delivery_day,
        timezone="Europe/Paris",
        local_time="08:00",
    )

    assert cutoff.isoformat() == expected
    assert cutoff.date() == delivery_day - pd.Timedelta(days=1)


def test_plan_zone_backfill_preserves_nl_promoted_root_gaps_and_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract(tmp_path, "NL")
    expected = (
        date(2026, 8, 13),
        date(2026, 8, 14),
        date(2026, 8, 16),
        date(2026, 8, 17),
        date(2026, 8, 18),
        date(2026, 8, 19),
    )
    captured = {}

    def fake_missing(**kwargs):
        captured.update(kwargs)
        return list(reversed(expected))

    monkeypatch.setattr(launcher, "missing_statistics_archive_days", fake_missing)

    planned = launcher.plan_zone_backfill(
        contract,
        source_delivery_day=date(2026, 8, 20),
    )

    assert planned == expected
    assert captured["live_output_root"] == contract.output_root
    assert captured["replay_output_root"] == contract.output_root / "_replays"
    assert captured["candidate_model"] == "mkonline_blend"
    assert captured["prediction_mode"] == "mkonline_blend"
    assert captured["current_delivery_day"] == date(2026, 8, 20)


def test_dry_run_auto_plans_all_zone_gaps_and_builds_causal_argv_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    gaps = {
        "FR": tuple(date(2026, 8, day) for day in range(16, 20)),
        "NL": (
            date(2026, 8, 13),
            date(2026, 8, 14),
            *tuple(date(2026, 8, day) for day in range(16, 20)),
        ),
    }
    _patch_planning(monkeypatch, tmp_path, gaps)
    launches: list[dict] = []
    built: list[dict] = []

    def fake_build(status, **kwargs):
        built.append({"zone": status.code, **kwargs})
        return [
            str(executable),
            str(tmp_path / "run_mkonline_live_zone.py"),
            "--zone",
            status.code,
            "--delivery-day",
            str(kwargs["delivery_day"]),
            "--data-as-of",
            kwargs["data_as_of"],
            "--pit-replay",
        ]

    monkeypatch.setattr(launcher, "build_dispatch_command", fake_build)
    monkeypatch.setattr(
        launcher,
        "launch_zone_forecast",
        lambda **kwargs: launches.append(kwargs),
    )

    result = launcher.run_statistics_backfill(
        zones=("FR", "NL"),
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        log_dir=tmp_path / "logs",
        dry_run=True,
    )

    assert result.planned_days == gaps
    assert result.remaining_days == gaps
    assert len(result.replay_results) == 10
    assert all(item.state == "dry_run" for item in result.replay_results)
    assert launches == []
    assert all(item["pit_replay"] is True for item in built)
    assert all(item["local_files_only"] is True for item in built)
    assert built[0]["data_as_of"] == "2026-08-15T08:00:00+02:00"
    assert built[4]["zone"] == "NL"
    assert built[4]["data_as_of"] == "2026-08-12T08:00:00+02:00"
    assert all(isinstance(arg, str) for item in result.replay_results for arg in item.command)


def test_every_zone_is_preflighted_and_planned_before_first_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    _patch_planning(
        monkeypatch,
        tmp_path,
        {"FR": (date(2026, 8, 16),), "DE": (date(2026, 8, 16),)},
        ready={"DE": False},
    )
    launches: list[str] = []
    monkeypatch.setattr(
        launcher,
        "launch_zone_forecast",
        lambda **kwargs: launches.append(kwargs["zone"]),
    )

    with pytest.raises(RuntimeError, match="DE: preflight refuse"):
        launcher.run_statistics_backfill(
            zones=("FR", "DE"),
            project_root=tmp_path,
            registry_path=registry,
            python_executable=executable,
            log_dir=tmp_path / "logs",
        )

    assert launches == []


def test_replays_are_sequential_and_each_success_is_revalidated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    days = (date(2026, 8, 16), date(2026, 8, 17))
    completed = _patch_planning(monkeypatch, tmp_path, {"FR": days})
    events: list[str] = []
    validations: list[tuple[date, bool]] = []

    monkeypatch.setattr(
        launcher,
        "build_dispatch_command",
        lambda status, **kwargs: [
            str(executable),
            "dispatcher.py",
            "--zone",
            status.code,
            "--delivery-day",
            str(kwargs["delivery_day"]),
            "--pit-replay",
        ],
    )

    def fake_launch(*, zone, delivery_day, data_as_of, pit_replay, **_kwargs):
        assert pit_replay is True
        assert data_as_of.endswith(("+02:00", "+01:00"))
        if delivery_day == days[1]:
            assert events[-1] == f"wait:{days[0]}"
        process = SimpleNamespace(
            wait=lambda day=delivery_day: events.append(f"wait:{day}") or 0,
            poll=lambda: 0,
        )
        return ForecastProcess(
            zone=zone,
            command=(str(executable), "dispatcher.py", "--zone", zone),
            log_path=tmp_path / f"{zone}_{delivery_day}.log",
            started_at=datetime.now(timezone.utc),
            process=process,
        )

    def fake_validate(_status, *, delivery_day, pit_replay, **_kwargs):
        validations.append((delivery_day, pit_replay))
        if delivery_day == days[-1]:
            completed["FR"] = True
        return tmp_path / "replays" / f"fr_day_ahead_{delivery_day}"

    monkeypatch.setattr(launcher, "launch_zone_forecast", fake_launch)
    monkeypatch.setattr(launcher, "validate_existing_forecast_archive", fake_validate)

    result = launcher.run_statistics_backfill(
        zones=("FR",),
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        log_dir=tmp_path / "logs",
    )

    assert events == [f"wait:{days[0]}", f"wait:{days[1]}"]
    assert validations == [(days[0], True), (days[1], True)]
    assert [item.state for item in result.replay_results] == ["success", "success"]
    assert result.remaining_days["FR"] == ()
    assert result.ok


def test_existing_replay_is_skipped_only_after_immutable_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    replay_day = date(2026, 8, 16)
    completed = _patch_planning(monkeypatch, tmp_path, {"BE": (replay_day,)})
    archive = tmp_path / "runs" / "be" / "_replays" / "be_day_ahead_2026-08-16"
    validated: list[dict] = []
    monkeypatch.setattr(launcher, "build_dispatch_command", lambda *_a, **_k: ["argv"])
    monkeypatch.setattr(
        launcher,
        "launch_zone_forecast",
        lambda **_kwargs: ForecastSkip(
            zone="BE",
            delivery_day=replay_day.isoformat(),
            archive_path=archive,
        ),
    )

    def fake_validate(_status, **kwargs):
        validated.append(kwargs)
        completed["BE"] = True
        return archive

    monkeypatch.setattr(launcher, "validate_existing_forecast_archive", fake_validate)

    result = launcher.run_statistics_backfill(
        zones=("BE",),
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        log_dir=tmp_path / "logs",
    )

    assert result.ok
    assert result.replay_results[0].state == "skipped"
    assert result.replay_results[0].archive_path == archive
    assert validated[0]["pit_replay"] is True
    assert validated[0]["delivery_day"] == replay_day


def test_one_replay_failure_does_not_hide_later_zone_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    replay_day = date(2026, 8, 16)
    completed = _patch_planning(
        monkeypatch,
        tmp_path,
        {"FR": (replay_day,), "DE": (replay_day,)},
    )
    launched: list[str] = []
    de_archive = tmp_path / "de_replay"
    monkeypatch.setattr(launcher, "build_dispatch_command", lambda *_a, **_k: ["argv"])

    def fake_launch(*, zone, **_kwargs):
        launched.append(zone)
        if zone == "FR":
            raise RuntimeError("echec FR explicite")
        return ForecastSkip(
            zone="DE",
            delivery_day=replay_day.isoformat(),
            archive_path=de_archive,
        )

    def fake_validate(status, **_kwargs):
        completed[status.code] = True
        return de_archive

    monkeypatch.setattr(launcher, "launch_zone_forecast", fake_launch)
    monkeypatch.setattr(launcher, "validate_existing_forecast_archive", fake_validate)

    result = launcher.run_statistics_backfill(
        zones=("FR", "DE"),
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        log_dir=tmp_path / "logs",
        stop_on_error=False,
    )

    assert launched == ["FR", "DE"]
    assert [item.state for item in result.replay_results] == ["failed", "skipped"]
    assert result.remaining_days["FR"] == (replay_day,)
    assert result.remaining_days["DE"] == ()
    assert not result.ok


def _backfill_result(*, ok: bool) -> launcher.BackfillResult:
    remaining = {} if ok else {"FR": (date(2026, 8, 16),)}
    return launcher.BackfillResult(
        source_delivery_days={"FR": date(2026, 8, 20)},
        planned_days={"FR": ()},
        replay_results=(),
        remaining_days=remaining,
    )


def test_main_runs_live_only_after_success_and_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_calls: list[dict] = []
    monkeypatch.setattr(launcher, "print_backfill_summary", lambda _result: None)
    monkeypatch.setattr(launcher, "print_batch_summary", lambda _result: None)
    monkeypatch.setattr(
        launcher,
        "run_forecast_batch",
        lambda **kwargs: live_calls.append(kwargs)
        or SimpleNamespace(ok=True),
    )

    monkeypatch.setattr(
        launcher,
        "run_statistics_backfill",
        lambda **_kwargs: _backfill_result(ok=True),
    )
    assert launcher.main(["--zones", "FR"]) == 0
    assert live_calls == []
    assert launcher.main(
        [
            "--zones",
            "FR",
            "--then-run-live",
            "--live-delivery-day",
            "2026-08-21",
        ]
    ) == 0
    assert len(live_calls) == 1
    assert live_calls[0]["delivery_day"] == "2026-08-21"

    live_calls.clear()
    monkeypatch.setattr(
        launcher,
        "run_statistics_backfill",
        lambda **_kwargs: _backfill_result(ok=False),
    )
    assert launcher.main(["--zones", "FR", "--then-run-live"]) == 1
    assert live_calls == []


def test_main_dry_run_never_starts_the_live_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "print_backfill_summary", lambda _result: None)
    monkeypatch.setattr(
        launcher,
        "run_statistics_backfill",
        lambda **_kwargs: _backfill_result(ok=True),
    )
    monkeypatch.setattr(
        launcher,
        "run_forecast_batch",
        lambda **_kwargs: pytest.fail("dry-run must never launch a live forecast"),
    )

    assert launcher.main(
        ["--zones", "FR", "--dry-run", "--then-run-live"]
    ) == 0


def test_powershell_wrapper_uses_safe_argv_and_gates_live_delivery_day() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "Start-StatisticsBackfill.ps1"
    ).read_text(encoding="utf-8")
    lowered = script.casefold()

    assert "& $PythonExe @Arguments" in script
    assert "exit $LASTEXITCODE" in script
    assert "'--zones'" in script
    assert "'--then-run-live'" in script
    assert "'--live-delivery-day'" in script
    assert "-not $ThenRunForecast" in script
    assert "[ValidateSet('FR', 'DE', 'BE', 'NL', 'ES')]" in script
    assert "invoke-expression" not in lowered
    assert "cmd.exe" not in lowered
    assert "-command" not in lowered
    assert "shell=true" not in lowered
