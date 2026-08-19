from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import live_history
from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.multizone_contract import (
    ZoneArtifactHashes,
    ZoneBlendWeights,
    ZoneModelContract,
    ZoneModelPaths,
)
from chronos2_hourly.multizone_live import (
    CandidateArtifacts,
    LiveSchedule,
    MAX_AUTOMATIC_REPLAY_DAYS,
    PredictionPolicy,
    ZoneLiveExecutionError,
    ZoneLiveHooks,
    run_zone_live,
)


ZONE = {
    "BE": (
        "Europe/Brussels",
        "power.price.da.be.bzn.hourly.entsoe.utc.cdh.eurmwh",
    ),
    "DE": (
        "Europe/Berlin",
        "power.price.da.de_lu.bzn.hourly.entsoe.utc.cdh.eurmwh",
    ),
    "NL": (
        "Europe/Amsterdam",
        "power.price.da.nl.bzn.hourly.entsoe.utc.cdh.eurmwh",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _bundle(
    tmp_path: Path,
    *,
    zone: str,
    benchmark_end: date = date(2026, 8, 13),
) -> tuple[ZoneModelContract, dict[str, Any]]:
    timezone, target = ZONE[zone]
    root = tmp_path / zone.lower()
    frozen = root / "frozen_autonomous"
    benchmark = root / "sealed_benchmark"
    frozen.mkdir(parents=True)
    benchmark.mkdir()
    live_config = root / "live.yaml"
    registry = root / "registry.yaml"
    base_config = root / "base.yaml"
    recipe = root / "recipe.json"
    live_config.write_text("live", encoding="utf-8")
    registry.write_text("registry", encoding="utf-8")
    base_config.write_text("base", encoding="utf-8")
    _write_json(
        recipe,
        {
            "prediction_mode": "autonomous_only",
            "mkonline_enabled": False,
            "autonomous_only_reason": "strict gate rejected MKOnline",
        },
    )
    for directory in (frozen, benchmark):
        _write_json(
            directory / "artifact_checksums.json",
            {"algorithm": "sha256", "artifacts": []},
        )
        _write_json(
            directory / "run_manifest.json",
            {
                "zone": zone,
                "timezone": timezone,
                "target_series": target,
                "candidate_model": "residual_corrected",
                "prediction_mode": "autonomous_only",
                "prediction_inputs": ["autonomous_extended_residual"],
                "storm_used_as_feature": False,
            },
        )

    index = local_delivery_day_index(benchmark_end, timezone=timezone)
    values = np.arange(len(index), dtype=float) + 50.0
    pd.DataFrame(
        {
            "delivery_start_utc": index.astype(str),
            "actual": values + 0.5,
            "residual_corrected__q10": values - 5.0,
            "residual_corrected__q50": values,
            "residual_corrected__q90": values + 5.0,
        }
    ).to_csv(benchmark / live_history.BACKTEST_NAME, index=False)
    _write_json(
        benchmark / live_history.METRICS_NAME,
        {
            "training_diagnostics": {
                "evaluation_start_local_date": benchmark_end.isoformat(),
                "evaluation_end_local_date": benchmark_end.isoformat(),
            }
        },
    )

    paths = ZoneModelPaths(
        live_config=live_config,
        registry=registry,
        base_config=base_config,
        frozen_autonomous_run=frozen,
        sealed_benchmark_run=benchmark,
        recipe_manifest=recipe,
        dependency_manifest=None,
        output_root=root / "live_runs",
    )
    contract = ZoneModelContract(
        schema_version=1,
        zone=zone,
        delivery_timezone=timezone,
        forecast_origin_timezone="Europe/Paris",
        forecast_origin_local_time="08:00",
        target_series=target,
        primary_series=None,
        storm_dashboard_series=f"power.price.{zone.lower()}.euromwh.h.fcst.3mv.storm",
        storm_dashboard_primary_series=None,
        storm_dashboard_naive_timezone=timezone,
        storm_strict_08_series=None,
        forecast_filename=f"forecast_hourly_{zone.lower()}.csv",
        required_covariates=(f"{zone.lower()}_residual_load_fcst",),
        paths=paths,
        checksum_hashes=ZoneArtifactHashes(
            base_config_sha256="a" * 64,
            frozen_autonomous_checksum_manifest_sha256=_sha256(
                frozen / "artifact_checksums.json"
            ),
            sealed_benchmark_checksum_manifest_sha256=_sha256(
                benchmark / "artifact_checksums.json"
            ),
            recipe_manifest_sha256=_sha256(recipe),
            dependency_manifest_sha256=None,
        ),
        weights=ZoneBlendWeights(autonomous=1.0, mkonline_primary=0.0),
        prediction_mode="autonomous_only",
        mkonline_enabled=False,
    )
    live = {
        "prediction_mode": "autonomous_only",
        "mkonline_enabled": False,
        "threads": 1,
        "workers": 1,
        "report": {
            "filename": f"report_{zone.lower()}_{{delivery_day}}.html",
        },
    }
    return contract, live


def _forecast(schedule: LiveSchedule, candidate_model: str) -> pd.DataFrame:
    values = np.arange(len(schedule.delivery_index), dtype=float) + 50.0
    return pd.DataFrame(
        {
            "delivery_start_utc": schedule.delivery_index,
            "forecast_origin_utc": schedule.cutoff_origin_local.tz_convert("UTC"),
            "q10": values - 5.0,
            "q50": values,
            "q90": values + 5.0,
            f"{candidate_model}__q10": values - 5.0,
            f"{candidate_model}__q50": values,
            f"{candidate_model}__q90": values + 5.0,
        }
    )


def _candidate(
    schedule: LiveSchedule,
    policy: PredictionPolicy,
) -> CandidateArtifacts:
    return CandidateArtifacts(
        forecast=_forecast(schedule, policy.candidate_model),
        canonical_target=pd.Series(
            np.arange(len(schedule.delivery_index), dtype=float),
            index=schedule.delivery_index,
        ),
        fit_audit={"target_availability": {"status": "mocked"}},
        input_diagnostics={},
        pit_freshness={},
        source_paths={},
        data_config={},
    )


def _assert_forecast_checksum(archive: Path, forecast_name: str) -> None:
    forecast = archive / forecast_name
    checksums = json.loads(
        (archive / "artifact_checksums.json").read_text(encoding="utf-8")
    )
    declaration = next(
        item
        for item in checksums["artifacts"]
        if item["path"] == forecast_name and item["role"] == "run_artifact"
    )
    assert declaration["sha256"] == _sha256(forecast)


def _planner_kwargs(contract: ZoneModelContract) -> dict[str, Any]:
    return {
        "sealed_benchmark_run": contract.paths.sealed_benchmark_run,
        "live_output_root": contract.paths.output_root,
        "replay_output_root": contract.paths.output_root / "_replays",
        "current_delivery_day": date(2026, 8, 15),
        "timezone": contract.delivery_timezone,
        "forecast_name": contract.forecast_filename,
        "candidate_model": "residual_corrected",
        "zone": contract.zone,
        "target_series": contract.target_series,
        "prediction_mode": "autonomous_only",
    }


def test_live_d15_bootstraps_missing_d14_before_storm_and_atomic_publish(
    tmp_path: Path,
) -> None:
    contract, live = _bundle(tmp_path, zone="DE")
    events: list[tuple[str, str]] = []

    def build(
        _contract: ZoneModelContract,
        schedule: LiveSchedule,
        policy: PredictionPolicy,
        _live: Any,
        _options: Any,
        _staging: Path,
    ) -> CandidateArtifacts:
        day = schedule.delivery_day.isoformat()
        events.append(("candidate", day))
        if day == "2026-08-14":
            assert schedule.as_of_origin_local == pd.Timestamp(
                "2026-08-13T08:00:00+02:00"
            )
            assert schedule.cutoff_origin_local == schedule.as_of_origin_local
        return _candidate(schedule, policy)

    def dashboard(
        _contract: ZoneModelContract,
        schedule: LiveSchedule,
        _data: Any,
    ) -> tuple[pd.Series, dict[str, Any]]:
        events.append(("storm", schedule.delivery_day.isoformat()))
        replay = (
            contract.paths.output_root
            / "_replays"
            / "de_day_ahead_2026-08-14"
        )
        assert (replay / contract.forecast_filename).is_file()
        return pd.Series(1.0, index=schedule.delivery_index), {
            "series": contract.storm_dashboard_series,
            "used_for_prediction": False,
        }

    def statistics(**kwargs: Any) -> dict[str, Any]:
        events.append(("statistics", kwargs["current_delivery_day"].isoformat()))
        assert kwargs["zone"] == "DE"
        assert kwargs["storm_dashboard_native"] is not None
        return {"status": "complete", "missing_realized_days": []}

    def report(
        _run_dir: Path,
        *,
        output_path: Path,
        **_kwargs: Any,
    ) -> Path:
        events.append(("report", "2026-08-15"))
        output_path.write_text("<html>ok</html>", encoding="utf-8")
        return output_path

    output = run_zone_live(
        contract,
        live_settings=live,
        data_as_of="2026-08-14T08:00:00+02:00",
        delivery_day="2026-08-15",
        hooks=ZoneLiveHooks(
            build,
            dashboard,
            statistics,
            report,
            live_history.missing_statistics_archive_days,
        ),
        wall_clock=pd.Timestamp("2026-08-14T09:00:00+02:00"),
    )

    assert events == [
        ("candidate", "2026-08-14"),
        ("candidate", "2026-08-15"),
        ("storm", "2026-08-15"),
        ("statistics", "2026-08-15"),
        ("report", "2026-08-15"),
    ]
    replay = (
        contract.paths.output_root
        / "_replays"
        / "de_day_ahead_2026-08-14"
    )
    assert output == contract.paths.output_root / "de_day_ahead_2026-08-15"
    assert replay.is_dir() and output.is_dir()
    replay_manifest = json.loads(
        (replay / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert replay_manifest["forecast_cutoff_local"].startswith(
        "2026-08-13 08:00:00+02:00"
    )
    assert replay_manifest["automatic_statistics_bootstrap"] is True
    assert replay_manifest["statistics_reporting_deferred"] is True
    assert replay_manifest["storm_loaded_for_prediction"] is False
    assert replay_manifest["storm_used_as_feature"] is False
    assert all(
        "storm" not in str(item).casefold()
        for item in replay_manifest["prediction_inputs"]
    )
    assert not (replay / "report_de_2026-08-14.html").exists()
    assert not (replay / live_history.STATISTICS_HISTORY_NAME).exists()
    _assert_forecast_checksum(replay, contract.forecast_filename)
    _assert_forecast_checksum(output, contract.forecast_filename)
    current = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert current["zone"] == "DE"
    assert current["storm_loaded_for_prediction"] is False
    assert current["storm_used_as_feature"] is False
    assert current["statistics_history"]["status"] == "complete"
    assert live_history.missing_statistics_archive_days(
        **_planner_kwargs(contract)
    ) == []
    assert not list(contract.paths.output_root.glob(".*.tmp-*"))
    assert not list((contract.paths.output_root / "_replays").glob(".*.tmp-*"))


def test_failed_d14_bootstrap_preserves_d15_with_actionable_statistics_blocker(
    tmp_path: Path,
) -> None:
    contract, live = _bundle(tmp_path, zone="BE")
    events: list[tuple[str, str]] = []

    def build(
        _contract: ZoneModelContract,
        schedule: LiveSchedule,
        policy: PredictionPolicy,
        _live: Any,
        _options: Any,
        _staging: Path,
    ) -> CandidateArtifacts:
        day = schedule.delivery_day.isoformat()
        events.append(("candidate", day))
        if day == "2026-08-14":
            raise RuntimeError("historical PIT vintage unavailable")
        return _candidate(schedule, policy)

    def forbidden_dashboard(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("Storm must be skipped when Statistics is blocked")

    def forbidden_statistics(**_kwargs: Any) -> Any:
        raise AssertionError("Statistics must be skipped after failed bootstrap")

    def report(
        run_dir: Path,
        *,
        output_path: Path,
        **_kwargs: Any,
    ) -> Path:
        events.append(("report", "2026-08-15"))
        assert (run_dir / "statistics_update_blocked.json").is_file()
        output_path.write_text("<html>benchmark only</html>", encoding="utf-8")
        return output_path

    output = run_zone_live(
        contract,
        live_settings=live,
        data_as_of="2026-08-14T08:00:00+02:00",
        delivery_day="2026-08-15",
        hooks=ZoneLiveHooks(
            build,
            forbidden_dashboard,
            forbidden_statistics,
            report,
            live_history.missing_statistics_archive_days,
        ),
        wall_clock=pd.Timestamp("2026-08-14T09:00:00+02:00"),
    )

    assert events == [
        ("candidate", "2026-08-14"),
        ("candidate", "2026-08-15"),
        ("report", "2026-08-15"),
    ]
    assert output.is_dir()
    _assert_forecast_checksum(output, contract.forecast_filename)
    replay = (
        contract.paths.output_root
        / "_replays"
        / "be_day_ahead_2026-08-14"
    )
    assert not replay.exists()
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    blocker = manifest["statistics_history"]
    assert blocker["status"] == "blocked_missing_causal_archives"
    assert blocker["missing_realized_days"] == ["2026-08-14"]
    assert blocker["failed_replay_day"] == "2026-08-14"
    assert blocker["failed_replay_cutoff_local"].startswith(
        "2026-08-13 08:00:00+02:00"
    )
    assert blocker["error_type"] == "RuntimeError"
    assert "historical PIT vintage unavailable" in blocker["error"]
    assert blocker["candidate_forecast_publication"] == "continued"
    assert blocker["storm_used_for_prediction"] is False
    assert manifest["storm_loaded_for_prediction"] is False
    assert manifest["storm_used_as_feature"] is False
    assert manifest[
        "storm_dashboard_loaded_after_candidate_frozen_for_statistics"
    ] is False
    summary = json.loads(
        (output / "live_run_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "forecast_complete_statistics_blocked"
    assert (output / "report_be_2026-08-15.html").is_file()
    assert (output / "statistics_update_blocked.json").is_file()
    assert not list(contract.paths.output_root.glob(".*.tmp-*"))


@pytest.mark.parametrize(
    ("planned", "limit", "message"),
    [
        (
            [date(2026, 8, 15)],
            MAX_AUTOMATIC_REPLAY_DAYS,
            "current/future day",
        ),
        (
            [],
            MAX_AUTOMATIC_REPLAY_DAYS + 1,
            f"hard safety cap {MAX_AUTOMATIC_REPLAY_DAYS}",
        ),
    ],
)
def test_unsafe_gap_plan_is_rejected_before_any_candidate(
    tmp_path: Path,
    planned: list[date],
    limit: int,
    message: str,
) -> None:
    contract, live = _bundle(tmp_path, zone="NL")
    live["max_automatic_replay_days"] = limit
    candidate_called = False

    def forbidden_build(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal candidate_called
        candidate_called = True
        raise AssertionError("candidate must not run for an unsafe plan")

    hooks = ZoneLiveHooks(
        forbidden_build,
        lambda *_args, **_kwargs: (None, {}),
        lambda **_kwargs: {},
        lambda *_args, **_kwargs: Path("unused"),
        lambda **_kwargs: planned,
    )
    with pytest.raises(ZoneLiveExecutionError, match=message):
        run_zone_live(
            contract,
            live_settings=live,
            data_as_of="2026-08-14T08:00:00+02:00",
            delivery_day="2026-08-15",
            hooks=hooks,
            wall_clock=pd.Timestamp("2026-08-14T09:00:00+02:00"),
        )
    assert candidate_called is False
    assert not contract.paths.output_root.exists()


def test_cross_zone_archive_fails_closed_before_candidate(
    tmp_path: Path,
) -> None:
    contract, live = _bundle(tmp_path, zone="DE")
    replay_root = contract.paths.output_root / "_replays"
    archive = replay_root / "be_day_ahead_2026-08-14"
    archive.mkdir(parents=True)
    index = local_delivery_day_index(
        date(2026, 8, 14),
        timezone=contract.delivery_timezone,
    )
    cutoff = pd.Timestamp("2026-08-13T08:00:00+02:00")
    values = np.arange(len(index), dtype=float) + 50.0
    forecast = archive / contract.forecast_filename
    pd.DataFrame(
        {
            "delivery_start_utc": index.astype(str),
            "forecast_origin_utc": str(cutoff.tz_convert("UTC")),
            "residual_corrected__q10": values - 5.0,
            "residual_corrected__q50": values,
            "residual_corrected__q90": values + 5.0,
        }
    ).to_csv(forecast, index=False)
    _write_json(
        archive / "run_manifest.json",
        {
            "run_type": "pit_replay",
            "zone": "BE",
            "timezone": "Europe/Brussels",
            "target_series": ZONE["BE"][1],
            "candidate_model": "residual_corrected",
            "prediction_mode": "autonomous_only",
            "prediction_inputs": ["autonomous_extended_residual"],
            "storm_used_as_feature": False,
            "delivery_day_local": "2026-08-14",
            "forecast_cutoff_local": str(cutoff),
        },
    )
    _write_json(
        archive / "artifact_checksums.json",
        {
            "algorithm": "sha256",
            "artifacts": [
                {
                    "path": contract.forecast_filename,
                    "role": "run_artifact",
                    "sha256": _sha256(forecast),
                }
            ],
        },
    )
    candidate_called = False

    def forbidden_build(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal candidate_called
        candidate_called = True
        raise AssertionError("cross-zone archive must fail before candidate")

    hooks = ZoneLiveHooks(
        forbidden_build,
        lambda *_args, **_kwargs: (None, {}),
        lambda **_kwargs: {},
        lambda *_args, **_kwargs: Path("unused"),
        live_history.missing_statistics_archive_days,
    )
    with pytest.raises(ValueError, match=r"zone='BE'.*'DE'"):
        run_zone_live(
            contract,
            live_settings=live,
            data_as_of="2026-08-14T08:00:00+02:00",
            delivery_day="2026-08-15",
            hooks=hooks,
            wall_clock=pd.Timestamp("2026-08-14T09:00:00+02:00"),
        )
    assert candidate_called is False
    assert not (
        contract.paths.output_root / "de_day_ahead_2026-08-15"
    ).exists()


def test_all_post_freeze_reporting_failures_preserve_checksummed_forecast(
    tmp_path: Path,
) -> None:
    contract, live = _bundle(tmp_path, zone="DE")

    def build(
        _contract: ZoneModelContract,
        schedule: LiveSchedule,
        policy: PredictionPolicy,
        _live: Any,
        _options: Any,
        _staging: Path,
    ) -> CandidateArtifacts:
        return _candidate(schedule, policy)

    def storm_failure(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Storm unavailable")

    def statistics_failure(**_kwargs: Any) -> Any:
        raise RuntimeError("Statistics unavailable")

    def report_failure(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("HTML unavailable")

    output = run_zone_live(
        contract,
        live_settings=live,
        data_as_of="2026-08-14T08:00:00+02:00",
        delivery_day="2026-08-15",
        hooks=ZoneLiveHooks(
            build,
            storm_failure,
            statistics_failure,
            report_failure,
            lambda **_kwargs: [],
        ),
        wall_clock=pd.Timestamp("2026-08-14T09:00:00+02:00"),
    )

    assert output.is_dir()
    _assert_forecast_checksum(output, contract.forecast_filename)
    manifest = json.loads(
        (output / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["reporting_status"] == "degraded"
    assert [item["stage"] for item in manifest["reporting_errors"]] == [
        "storm_dashboard",
        "statistics",
        "html_report",
    ]
    assert manifest["statistics_history"]["status"] == (
        "blocked_reporting_error"
    )
    fallback = output / "report_de_2026-08-15.html"
    assert fallback.is_file()
    assert "forecast DE" in fallback.read_text(encoding="utf-8")
    diagnostics = json.loads(
        (output / "reporting_errors.json").read_text(encoding="utf-8")
    )
    assert diagnostics["forecast_modified"] is False
    assert diagnostics["candidate_forecast_sha256"] == _sha256(
        output / contract.forecast_filename
    )
    assert not list(contract.paths.output_root.glob(".*.tmp-*"))


def test_partial_statistics_uses_only_prefix_before_first_gap(
    tmp_path: Path,
) -> None:
    contract, _live = _bundle(
        tmp_path,
        zone="DE",
        benchmark_end=date(2026, 8, 13),
    )
    replay = (
        contract.paths.output_root
        / "_replays"
        / "de_day_ahead_2026-08-14"
    )
    replay.mkdir(parents=True)
    day = date(2026, 8, 14)
    index = local_delivery_day_index(day, timezone=contract.delivery_timezone)
    cutoff = pd.Timestamp("2026-08-13T08:00:00+02:00")
    values = np.arange(len(index), dtype=float) + 60.0
    forecast = replay / contract.forecast_filename
    pd.DataFrame(
        {
            "delivery_start_utc": index.astype(str),
            "forecast_origin_utc": str(cutoff.tz_convert("UTC")),
            "residual_corrected__q10": values - 5.0,
            "residual_corrected__q50": values,
            "residual_corrected__q90": values + 5.0,
        }
    ).to_csv(forecast, index=False)
    _write_json(
        replay / "run_manifest.json",
        {
            "run_type": "pit_replay",
            "zone": contract.zone,
            "timezone": contract.delivery_timezone,
            "target_series": contract.target_series,
            "candidate_model": "residual_corrected",
            "prediction_mode": "autonomous_only",
            "prediction_inputs": ["autonomous_extended_residual"],
            "storm_used_as_feature": False,
            "delivery_day_local": day.isoformat(),
            "forecast_cutoff_local": str(cutoff),
        },
    )
    _write_json(
        replay / "artifact_checksums.json",
        {
            "algorithm": "sha256",
            "artifacts": [
                {
                    "path": contract.forecast_filename,
                    "role": "run_artifact",
                    "sha256": _sha256(forecast),
                }
            ],
        },
    )
    staging = tmp_path / "statistics_staging"
    actual = pd.Series(values + 0.5, index=index)
    blocker = {
        "status": "blocked_missing_causal_archives",
        "missing_realized_days": ["2026-08-15"],
    }

    audit = live_history.update_live_statistics_history(
        staging_run_dir=staging,
        sealed_benchmark_run=contract.paths.sealed_benchmark_run,
        live_output_root=contract.paths.output_root,
        replay_output_root=contract.paths.output_root / "_replays",
        current_delivery_day=date(2026, 8, 16),
        canonical_target=actual,
        storm_pit_path=tmp_path / "missing_storm.parquet",
        zone=contract.zone,
        timezone=contract.delivery_timezone,
        forecast_name=contract.forecast_filename,
        candidate_model="residual_corrected",
        target_series=contract.target_series,
        prediction_mode="autonomous_only",
        allow_partial_prefix=True,
        statistics_blocker=blocker,
    )

    assert audit["status"] == "partial_contiguous_prefix"
    assert audit["evaluated_realized_days"] == ["2026-08-14"]
    assert audit["missing_realized_days"] == ["2026-08-15"]
    assert audit["statistics_prefix_end_local"] == "2026-08-14"
    assert audit["statistics_complete"] is False
    assert audit["statistics_blocker"] == blocker
    history = pd.read_csv(staging / live_history.STATISTICS_HISTORY_NAME)
    history_days = pd.to_datetime(
        history["delivery_start_utc"], utc=True
    ).dt.tz_convert(contract.delivery_timezone).dt.date
    assert history_days.min() == date(2026, 8, 13)
    assert history_days.max() == date(2026, 8, 14)
    assert date(2026, 8, 15) not in set(history_days)
