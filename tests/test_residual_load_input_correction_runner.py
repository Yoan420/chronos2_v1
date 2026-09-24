from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import run_residual_load_input_correction as runner
from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.residual_load_input_corrector import (
    RESIDUAL_LOAD_ALIASES,
    RESIDUAL_LOAD_OBS_SERIES_BY_ALIAS,
    conservative_label_end_utc,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path) -> runner.RunnerConfig:
    project = tmp_path / "project"
    source = project / "config" / "experiment.yaml"
    source.parent.mkdir(parents=True)
    source.write_text("schema_version: 1\n", encoding="utf-8")
    implementation = project / "chronos2_hourly" / "residual_load_input_corrector.py"
    implementation.parent.mkdir(parents=True)
    implementation.write_text("# test implementation seal\n", encoding="utf-8")
    forecast_root = project / "data" / "pit" / "vintages"
    observed_root = project / "observed"
    aliases = tuple(
        runner.AliasSpec(
            alias=alias,
            country=alias[:2].upper(),
            forecast_path=forecast_root / f"{alias}.parquet",
            observed_path=observed_root / f"{alias}.parquet",
            observed_series=RESIDUAL_LOAD_OBS_SERIES_BY_ALIAS[alias],
        )
        for alias in RESIDUAL_LOAD_ALIASES
    )
    return runner.RunnerConfig(
        project_root=project,
        source_path=source,
        raw={"schema_version": 1},
        experiment_id="residual_load_input_correction_v1",
        output_root=(
            project / "runs" / "experiments" / "residual_load_input_correction_v1"
        ),
        corrected_directory="corrected",
        audit_directory="audit",
        report_directory="report",
        timezone="Europe/Paris",
        origin_clock="08:00",
        history_start_day=date(2026, 1, 1),
        evaluation_start_day=date(2026, 1, 4),
        evaluation_end_day=date(2026, 1, 4),
        final_start_day=date(2026, 1, 4),
        final_end_day=date(2026, 1, 4),
        label_delay_days=2,
        forecast_fill_limit_hours=0,
        forecast_minimum_coverage=0.5,
        refit_every_days=1,
        cold_start_policy="raw_passthrough",
        minimum_training_rows=48,
        max_abs_correction_gw=3.0,
        forecast_change_lags_hours=(1, 24),
        error_lags_hours=(48,),
        error_rolling_windows_hours=(24,),
        minimum_scoring_coverage=0.95,
        minimum_scoring_ramps=1,
        aliases=aliases,
    )


def _touch_forecasts(config: runner.RunnerConfig) -> None:
    for item in config.aliases:
        item.forecast_path.parent.mkdir(parents=True, exist_ok=True)
        item.forecast_path.write_bytes(b"present but deliberately unopened")


def _write_small_vintages(config: runner.RunnerConfig) -> None:
    grid = runner._full_index(config)
    position = np.arange(len(grid), dtype=float) / 100.0
    for offset, item in enumerate(config.aliases):
        raw = 20.0 + offset + position
        raw[-(offset + 1)] = np.nan
        forecast = pd.DataFrame(
            {
                "value_time_utc": grid,
                "snapshot_time_utc": grid - pd.Timedelta(days=2),
                "revision_time_utc": grid - pd.Timedelta(days=2),
                "value": raw,
            }
        )
        observed = pd.DataFrame(
            {
                "value_time_utc": grid,
                "snapshot_time_utc": grid + pd.Timedelta(hours=1),
                "revision_time_utc": grid + pd.Timedelta(hours=1),
                "value": raw + 1.0,
            }
        )
        item.forecast_path.parent.mkdir(parents=True, exist_ok=True)
        item.observed_path.parent.mkdir(parents=True, exist_ok=True)
        forecast.to_parquet(item.forecast_path, index=False)
        observed.to_parquet(item.observed_path, index=False)


def test_plan_succeeds_without_observations_and_never_opens_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _touch_forecasts(config)
    monkeypatch.setattr(
        runner.pd,
        "read_parquet",
        lambda *_args, **_kwargs: pytest.fail("plan() must not open Parquet"),
    )

    result = runner.plan(config)

    assert result["missing_forecasts"] == []
    assert result["missing_observed_vintages"] == list(RESIDUAL_LOAD_ALIASES)
    assert result["observed_sync_required"] is True
    assert "SyncObserved" in result["observed_sync_command"]
    assert result["production_changed"] is False
    assert result["writes_data_pit"] is False
    assert result["writes_runs_live"] is False
    assert not config.output_root.exists()


def test_build_missing_observations_has_actionable_error_before_any_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _touch_forecasts(config)
    monkeypatch.setattr(
        runner.pd,
        "read_parquet",
        lambda *_args, **_kwargs: pytest.fail("missing-input guard must run first"),
    )

    with pytest.raises(FileNotFoundError, match="SyncObserved|Forecast.ps1"):
        runner.build(config, correction_generator=lambda *_args, **_kwargs: None)

    assert not config.output_root.exists()


def test_forecast_and_observation_selection_apply_both_pit_cutoffs(
    tmp_path: Path,
) -> None:
    alias = RESIDUAL_LOAD_ALIASES[0]
    grid = local_delivery_day_index(date(2026, 1, 4), timezone="Europe/Paris")
    origins = runner._origin_by_value_time(grid, timezone="Europe/Paris")
    causal = pd.DataFrame(
        {
            "value_time_utc": grid,
            "snapshot_time_utc": origins.to_numpy() - pd.Timedelta(hours=2),
            "revision_time_utc": origins.to_numpy() - pd.Timedelta(hours=1),
            "value": 10.0,
        }
    )
    late_snapshot = causal.assign(
        snapshot_time_utc=origins.to_numpy() + pd.Timedelta(minutes=1),
        value=999.0,
    )
    late_revision = causal.assign(
        revision_time_utc=origins.to_numpy() + pd.Timedelta(minutes=1),
        value=888.0,
    )
    normalized_forecast = runner._normalize_vintages(
        pd.concat([causal, late_snapshot, late_revision], ignore_index=True),
        path=tmp_path / "forecast.parquet",
    )

    selected, _, metadata = runner._select_forecast_asof(
        normalized_forecast,
        alias=alias,
        grid=grid,
        timezone="Europe/Paris",
        fill_limit=0,
    )

    np.testing.assert_allclose(selected.to_numpy(), 10.0)
    assert metadata["cutoff_violations"] == 0

    origin = pd.Timestamp("2026-01-03T07:00:00Z")
    label_end = grid[-1]
    observed_causal = causal.assign(
        snapshot_time_utc=origin - pd.Timedelta(hours=2),
        revision_time_utc=origin - pd.Timedelta(hours=1),
        value=20.0,
    )
    observed_late_snapshot = observed_causal.assign(
        snapshot_time_utc=origin + pd.Timedelta(minutes=1),
        value=777.0,
    )
    observed_late_revision = observed_causal.assign(
        revision_time_utc=origin + pd.Timedelta(minutes=1),
        value=666.0,
    )
    normalized_observed = runner._normalize_vintages(
        pd.concat(
            [observed_causal, observed_late_snapshot, observed_late_revision],
            ignore_index=True,
        ),
        path=tmp_path / "observed.parquet",
    )
    observations, audits = runner._select_observations_asof(
        {name: normalized_observed for name in RESIDUAL_LOAD_ALIASES},
        origin_utc=origin,
        label_end_utc=label_end,
        grid=grid,
    )

    np.testing.assert_allclose(observations.to_numpy(), 20.0)
    assert all(item["cutoff_violations"] == 0 for item in audits)
    assert all(item["maximum_selected_snapshot_time_utc"] <= origin for item in audits)
    assert all(item["maximum_selected_revision_time_utc"] <= origin for item in audits)


def test_load_config_rejects_output_outside_runs_experiments(tmp_path: Path) -> None:
    payload = yaml.safe_load(
        (PROJECT_ROOT / "config" / "residual_load_input_correction.yaml").read_text(
            encoding="utf-8"
        )
    )
    payload["outputs"]["experiment_root"] = "runs/live/forbidden"
    source = tmp_path / "bad_output.yaml"
    source.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="runs/experiments"):
        runner.load_config(source, project_root=tmp_path)


def test_load_config_rejects_another_experiment_or_colliding_directories(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(
        (PROJECT_ROOT / "config" / "residual_load_input_correction.yaml").read_text(
            encoding="utf-8"
        )
    )
    payload["outputs"]["experiment_root"] = "runs/experiments/another_experiment"
    source = tmp_path / "wrong_experiment.yaml"
    source.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="exactement"):
        runner.load_config(source, project_root=tmp_path)

    payload["outputs"]["experiment_root"] = (
        "runs/experiments/residual_load_input_correction_v1"
    )
    payload["outputs"]["report_directory"] = payload["outputs"][
        "corrected_directory"
    ]
    source.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="disjoints"):
        runner.load_config(source, project_root=tmp_path)


def test_atomic_publish_restores_previous_output_on_swap_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiments = tmp_path / "runs" / "experiments"
    final = experiments / "candidate"
    staging = experiments / ".candidate.staging"
    final.mkdir(parents=True)
    staging.mkdir()
    (final / "state.txt").write_text("old", encoding="utf-8")
    (staging / "state.txt").write_text("new", encoding="utf-8")
    real_replace = os.replace

    def fail_publication(source, destination):
        if Path(source) == staging and Path(destination) == final:
            raise OSError("simulated publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", fail_publication)

    with pytest.raises(OSError, match="simulated"):
        runner._publish_staged_directory(staging, final, overwrite=True)

    assert (final / "state.txt").read_text(encoding="utf-8") == "old"
    assert (staging / "state.txt").read_text(encoding="utf-8") == "new"
    assert not list(experiments.glob("*.backup"))


def test_small_build_is_confined_atomic_and_report_metrics_are_exact(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_small_vintages(config)
    calls: list[dict[str, object]] = []

    def fake_generator(forecasts, observations, **kwargs):
        calls.append({"observations": observations.copy(), **kwargs})
        block_index = pd.date_range(
            local_delivery_day_index(
                kwargs["start_day"], timezone=kwargs["timezone"]
            )[0],
            local_delivery_day_index(
                kwargs["end_day"], timezone=kwargs["timezone"]
            )[-1],
            freq="h",
            tz="UTC",
        )
        raw = forecasts.loc[block_index, list(RESIDUAL_LOAD_ALIASES)].copy()
        shift = pd.DataFrame(
            1.0,
            index=block_index,
            columns=RESIDUAL_LOAD_ALIASES,
        ).where(raw.notna())
        return SimpleNamespace(raw=raw, correction=shift, corrected=raw + shift)

    built = runner.build(
        config,
        threads=2,
        correction_generator=fake_generator,
    )

    assert built.output_root == config.output_root
    assert built.output_root.resolve().is_relative_to(
        (config.project_root / "runs" / "experiments").resolve()
    )
    assert len(calls) == 1
    assert calls[0]["thread_count"] == 2
    assert calls[0]["max_abs_correction_gw"] == pytest.approx(3.0)
    assert calls[0]["cold_start_policy"] == "raw_passthrough"
    assert calls[0]["minimum_training_rows"] == 48
    observations = calls[0]["observations"]
    assert isinstance(observations, pd.DataFrame)
    assert observations.index.max() == conservative_label_end_utc(
        config.evaluation_start_day
    )
    assert built.manifest_path.is_file()
    assert built.checksum_manifest_path.is_file()
    assert built.audit_path.is_file()
    assert set(built.corrected_paths) == set(RESIDUAL_LOAD_ALIASES)
    assert all(path.is_file() for path in built.corrected_paths.values())
    assert not list((config.project_root / "runs" / "experiments").glob("*.tmp"))
    manifest = json.loads(built.manifest_path.read_text(encoding="utf-8"))
    assert manifest["production_changed"] is False
    assert manifest["writes_data_pit"] is False
    assert manifest["writes_runs_live"] is False

    report_json, metrics_csv = runner.report(config)

    metrics = pd.read_csv(metrics_csv)
    assert set(metrics["alias"]) == set(RESIDUAL_LOAD_ALIASES)
    assert set(metrics["regime"]) == {"all", "observed_ramp_top10pct"}
    np.testing.assert_allclose(metrics["raw_mae_gw"], 1.0)
    np.testing.assert_allclose(metrics["corrected_mae_gw"], 0.0)
    np.testing.assert_allclose(metrics["mae_gain_gw"], 1.0)
    report_payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert report_payload["status"] == "complete"
    assert len(report_payload["metrics"]) == 2 * len(RESIDUAL_LOAD_ALIASES)
    runner._verify_checksums(config.output_root)

    config.source_path.write_text("schema_version: 1\nchanged: true\n", encoding="utf-8")
    with pytest.raises(runner.ResidualLoadInputRunnerError, match="configuration a change"):
        runner.report(config, overwrite=True)
