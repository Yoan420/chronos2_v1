from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.hourly_contract import local_delivery_day_index
import run_mkonline_live_hourly as live_runner
from run_mkonline_live_hourly import (
    LiveSchedule,
    _build_dynamic_data,
    _forecast_frame,
    _resolve_schedule,
    _update_live_metrics,
    _validate_live_forecast,
    _verify_benchmark,
    _verify_frozen_training_source,
)


def _schedule(
    *,
    as_of: str = "2026-08-13T12:00:00+02:00",
    delivery_day: str = "2026-08-14",
) -> LiveSchedule:
    return _resolve_schedule(as_of, delivery_day)


def _valid_forecast(schedule: LiveSchedule) -> pd.DataFrame:
    size = len(schedule.delivery_index)
    origin = schedule.cutoff_local.tz_convert("UTC")
    return pd.DataFrame(
        {
            "delivery_start_utc": schedule.delivery_index,
            "forecast_origin_utc": origin,
            "q10": np.linspace(30.0, 40.0, size),
            "q50": np.linspace(40.0, 50.0, size),
            "q90": np.linspace(50.0, 60.0, size),
        }
    )


def test_resolve_schedule_uses_tomorrow_and_civil_eight_cutoff() -> None:
    schedule = _resolve_schedule(
        "2026-08-13T10:00:00Z",
        None,
    )

    assert schedule.as_of_local == pd.Timestamp(
        "2026-08-13T12:00:00+02:00"
    )
    assert schedule.delivery_day == date(2026, 8, 14)
    assert schedule.cutoff_local == pd.Timestamp(
        "2026-08-13 08:00", tz="Europe/Paris"
    )
    assert schedule.delivery_index.equals(
        local_delivery_day_index("2026-08-14", timezone="Europe/Paris")
    )


@pytest.mark.parametrize(
    ("as_of", "delivery_day", "expected_hours"),
    [
        ("2027-03-27T12:00:00+01:00", "2027-03-28", 23),
        ("2026-10-24T12:00:00+02:00", "2026-10-25", 25),
    ],
)
def test_resolve_schedule_preserves_physical_dst_delivery_day(
    as_of: str,
    delivery_day: str,
    expected_hours: int,
) -> None:
    schedule = _resolve_schedule(as_of, delivery_day)

    assert len(schedule.delivery_index) == expected_hours
    assert schedule.delivery_index.equals(
        local_delivery_day_index(delivery_day, timezone="Europe/Paris")
    )
    assert schedule.cutoff_local.hour == 8


def test_resolve_schedule_rejects_ambiguous_or_non_operational_inputs() -> None:
    with pytest.raises(ValueError):
        _resolve_schedule("2026-08-13T12:00:00", "2026-08-14")
    with pytest.raises(ValueError):
        _resolve_schedule("2026-08-13T07:59:59+02:00", "2026-08-14")
    with pytest.raises(ValueError):
        _resolve_schedule("2026-08-13T12:00:00+02:00", "2026-08-15")


def test_resolve_schedule_accepts_the_exact_civil_cutoff() -> None:
    schedule = _resolve_schedule(
        "2026-08-13T08:00:00+02:00",
        "2026-08-14",
    )

    assert schedule.as_of_local == schedule.cutoff_local


def test_validate_live_forecast_accepts_exact_finite_ordered_day() -> None:
    schedule = _schedule()
    forecast = _valid_forecast(schedule)

    validated = _validate_live_forecast(forecast, schedule)

    assert len(validated) == len(schedule.delivery_index)
    assert pd.DatetimeIndex(
        pd.to_datetime(validated["delivery_start_utc"], utc=True)
    ).equals(schedule.delivery_index)


@pytest.mark.parametrize("mutation", ["missing_hour", "unordered", "extra_day"])
def test_validate_live_forecast_rejects_non_exact_timeline(mutation: str) -> None:
    schedule = _schedule()
    forecast = _valid_forecast(schedule)
    if mutation == "missing_hour":
        forecast = forecast.iloc[:-1].copy()
    elif mutation == "unordered":
        forecast = forecast.iloc[::-1].copy()
    else:
        extra = forecast.iloc[[-1]].copy()
        extra["delivery_start_utc"] = (
            pd.Timestamp(extra["delivery_start_utc"].iloc[0])
            + pd.Timedelta(hours=1)
        )
        forecast = pd.concat([forecast, extra], ignore_index=True)

    with pytest.raises(ValueError):
        _validate_live_forecast(forecast, schedule)


@pytest.mark.parametrize("mutation", ["nan", "crossed", "noncausal_origin"])
def test_validate_live_forecast_rejects_invalid_predictions_or_origin(
    mutation: str,
) -> None:
    schedule = _schedule()
    forecast = _valid_forecast(schedule)
    if mutation == "nan":
        forecast.loc[0, "q50"] = np.nan
    elif mutation == "crossed":
        forecast.loc[0, "q10"] = forecast.loc[0, "q90"] + 1.0
    else:
        forecast.loc[0, "forecast_origin_utc"] = forecast.loc[
            0, "delivery_start_utc"
        ]

    with pytest.raises(ValueError):
        _validate_live_forecast(forecast, schedule)


def test_validate_live_forecast_rejects_an_origin_after_the_da_cutoff() -> None:
    schedule = _schedule()
    forecast = _valid_forecast(schedule)
    forecast["forecast_origin_utc"] = (
        schedule.cutoff_local + pd.Timedelta(hours=1)
    ).tz_convert("UTC")

    with pytest.raises(ValueError):
        _validate_live_forecast(forecast, schedule)


def test_final_forecast_origin_represents_every_input_of_the_blend() -> None:
    schedule = _schedule()
    index = schedule.delivery_index
    chronos_origin = (
        schedule.cutoff_local - pd.Timedelta(hours=1)
    ).tz_convert("UTC")
    chronos = pd.DataFrame(
        {
            "q10": 30.0,
            "q50": 40.0,
            "q90": 50.0,
            "forecast_origin_utc": chronos_origin,
        },
        index=index,
    )
    extended = chronos.loc[:, ["q10", "q50", "q90"]].copy()
    mkonline = pd.Series(45.0, index=index)
    cutoff = pd.Series(schedule.cutoff_local.tz_convert("UTC"), index=index)

    result = _forecast_frame(
        schedule=schedule,
        chronos_live=chronos,
        extended_live=extended,
        mkonline=mkonline,
        cutoff=cutoff,
    )

    expected = pd.DatetimeIndex(
        [schedule.cutoff_local.tz_convert("UTC")] * len(index)
    )
    observed = pd.DatetimeIndex(
        pd.to_datetime(result["forecast_origin_utc"], utc=True)
    )
    assert observed.equals(expected)


def test_every_frozen_training_input_consumed_by_live_fit_is_verified() -> None:
    project_root = Path(__file__).resolve().parents[1]
    verified = _verify_frozen_training_source(
        project_root / "runs" / "chronos2_hourly_fr_residual_extended_v1"
    )

    assert "inputs/model_covariates_with_future.csv.gz" in verified


def test_benchmark_manifest_consumed_as_output_template_is_verified() -> None:
    project_root = Path(__file__).resolve().parents[1]
    verified = _verify_benchmark(
        project_root / "runs" / "chronos2_hourly_fr_mkonline_blend_v1"
    )

    assert "run_manifest.json" in verified


def test_dynamic_inputs_are_bounded_by_the_auction_cutoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = _schedule()
    expected_target_end = schedule.delivery_index[0] - pd.Timedelta(hours=1)
    target = pd.Series([50.0], index=pd.DatetimeIndex([expected_target_end]))
    future = pd.DataFrame({"known": 1.0}, index=schedule.delivery_index)
    all_features = future.copy()
    zone = SimpleNamespace(zone="FR")
    sync_calls: list[tuple[pd.Timestamp, dict]] = []

    monkeypatch.setattr(
        live_runner,
        "load_yaml",
        lambda _path: {
            "model": {"seed": 42, "context_length": 2048},
            "data": {},
            "zones": {
                "FR": {
                    "covariates": {
                        "nl_residual_load_fcst": {"enabled": True}
                    }
                }
            },
        },
    )
    monkeypatch.setattr(live_runner, "set_reproducibility", lambda _seed: None)
    monkeypatch.setattr(
        live_runner,
        "build_zone_configs",
        lambda *_args, **_kwargs: [zone],
    )

    def fake_sync(_zones, sync_config, _config_dir, **kwargs):
        sync_calls.append((kwargs["as_of"], sync_config))
        return pd.DataFrame({"status": ["ok"]})

    monkeypatch.setattr(live_runner, "sync_saturn_data", fake_sync)
    monkeypatch.setattr(
        live_runner,
        "prepare_zone_data",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        live_runner,
        "_feature_inputs",
        lambda *_args, **_kwargs: (target, pd.DataFrame(), future, all_features),
    )

    dynamic, _data, fresh = _build_dynamic_data(
        base_config_path=tmp_path / "base.yaml",
        schedule=schedule,
        inputs_dir=tmp_path / "inputs",
        sync_manifest_path=tmp_path / "sync.csv",
        naive_timezone_overrides={"nl_residual_load_fcst": "UTC"},
    )

    assert dynamic["data"]["runtime_as_of"] == schedule.cutoff_local.isoformat()
    assert len(sync_calls) == 1
    sync_as_of, sync_config = sync_calls[0]
    assert sync_as_of == schedule.cutoff_local
    expected_sync_start = schedule.delivery_index[0] - pd.Timedelta(hours=48)
    assert pd.Timestamp(sync_config["data"]["start"]) == expected_sync_start
    assert "start" not in dynamic["data"]
    assert (
        dynamic["zones"]["FR"]["covariates"]["nl_residual_load_fcst"][
            "naive_timezone"
        ]
        == "UTC"
    )
    assert fresh.index.equals(schedule.delivery_index)


def test_live_metrics_replace_only_live_diagnostics(tmp_path: Path) -> None:
    schedule = _schedule()
    path = tmp_path / "metrics_hourly.json"
    sealed_metrics = [{"model": "mkonline_blend", "mae": 11.23}]
    payload = {
        "metrics": sealed_metrics,
        "training_diagnostics": {
            "evaluation_start_local_date": "2025-08-12",
            "evaluation_end_local_date": "2026-08-11",
            "mkonline_blend": {"target_availability": {"stale": True}},
        },
        "forecast_diagnostics": {"delivery_day_local": "2026-08-12"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    availability = {
        "forecast_delivery_day_local": schedule.delivery_day.isoformat(),
    }

    _update_live_metrics(
        path,
        schedule=schedule,
        fit_audit={"target_availability": availability},
    )

    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated["metrics"] == sealed_metrics
    assert (
        updated["training_diagnostics"]["evaluation_start_local_date"]
        == "2025-08-12"
    )
    assert (
        updated["training_diagnostics"]["mkonline_blend"][
            "target_availability"
        ]
        == availability
    )
    assert updated["forecast_diagnostics"]["delivery_day_local"] == "2026-08-14"
    assert updated["forecast_diagnostics"]["n_forecast_hours"] == 24


@pytest.mark.parametrize("pit_replay", [False, True])
def test_main_publishes_one_coherent_live_run_with_mocked_engines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pit_replay: bool,
) -> None:
    as_of = (
        "2026-08-12T12:00:00+02:00"
        if pit_replay
        else "2026-08-13T12:00:00+02:00"
    )
    delivery_day = "2026-08-13" if pit_replay else "2026-08-14"
    schedule = _resolve_schedule(as_of, delivery_day)
    frozen = tmp_path / "frozen"
    benchmark = tmp_path / "benchmark"
    output_root = tmp_path / "runs"
    output = (
        output_root / "_replays" / f"fr_day_ahead_{delivery_day}"
        if pit_replay
        else output_root / f"fr_day_ahead_{delivery_day}"
    )
    frozen.mkdir()
    benchmark.mkdir()
    (frozen / "artifact_checksums.json").write_text("{}", encoding="utf-8")
    (benchmark / "artifact_checksums.json").write_text("{}", encoding="utf-8")
    (benchmark / "run_manifest.json").write_text(
        json.dumps({"target_availability": {"stale": True}}),
        encoding="utf-8",
    )
    base_config = tmp_path / "base.yaml"
    recipe = tmp_path / "recipe.json"
    dependency = tmp_path / "dependency.json"
    base_config.write_text("model: {}", encoding="utf-8")
    recipe.write_text("{}", encoding="utf-8")
    dependency.write_text("{}", encoding="utf-8")
    config = tmp_path / "live.yaml"
    config.write_text(
        "\n".join(
            [
                "live:",
                f"  base_config: '{base_config.as_posix()}'",
                f"  frozen_autonomous_run: '{frozen.as_posix()}'",
                f"  sealed_benchmark_run: '{benchmark.as_posix()}'",
                f"  recipe_manifest: '{recipe.as_posix()}'",
                f"  dependency_manifest: '{dependency.as_posix()}'",
                f"  output_root: '{output_root.as_posix()}'",
                "report:",
                "  filename: 'live_{delivery_day}.html'",
                "  title: 'Live {delivery_day}'",
            ]
        ),
        encoding="utf-8",
    )
    index = schedule.delivery_index
    fresh = pd.DataFrame({"feature": 1.0}, index=index)
    chronos = pd.DataFrame(
        {
            "forecast_origin_utc": schedule.cutoff_local.tz_convert("UTC"),
            "q10": 30.0,
            "q50": 40.0,
            "q90": 50.0,
        },
        index=index,
    )
    chronos.index.name = "delivery_start_utc"
    extended = chronos.loc[:, ["q10", "q50", "q90"]].copy()
    availability = {
        "forecast_delivery_day_local": schedule.delivery_day.isoformat(),
    }
    events: list[str] = []

    real_timestamp = pd.Timestamp

    class FrozenTimestamp(real_timestamp):
        @classmethod
        def now(cls, tz=None):
            value = real_timestamp("2026-08-13T12:00:00+02:00")
            return value.tz_convert(tz) if tz is not None else value.tz_localize(None)

    monkeypatch.setattr(live_runner.pd, "Timestamp", FrozenTimestamp)

    monkeypatch.setattr(
        live_runner,
        "_load_recipe",
        lambda *_args, **_kwargs: {
            "external_expert": {"commercial_entitlement_status": "test"}
        },
    )
    monkeypatch.setattr(live_runner, "_verify_source_run", lambda *_args: None)
    monkeypatch.setattr(
        live_runner,
        "_verify_frozen_training_source",
        lambda *_args: {},
    )
    monkeypatch.setattr(live_runner, "_verify_benchmark", lambda *_args: {})
    monkeypatch.setattr(
        live_runner,
        "_build_dynamic_data",
        lambda **_kwargs: (
            {"model": {}},
            SimpleNamespace(target=pd.Series([50.0]), diagnostics={}),
            fresh,
        ),
    )
    monkeypatch.setattr(
        live_runner,
        "_validate_live_input_audit",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        live_runner,
        "_audit_future_pit_freshness",
        lambda **_kwargs: {"status": "fresh"},
    )
    monkeypatch.setattr(live_runner, "load_model", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        live_runner,
        "run_existing_live_forecast",
        lambda *_args, **_kwargs: chronos,
    )
    monkeypatch.setattr(
        live_runner,
        "_train_and_predict_extended",
        lambda **_kwargs: (
            extended,
            {"target_availability": availability},
        ),
    )

    def fake_materialize(**kwargs):
        kwargs["output"].write_bytes(b"parquet placeholder")
        return ["materialize", schedule.delivery_day.isoformat()]

    monkeypatch.setattr(live_runner, "_materialize", fake_materialize)
    monkeypatch.setattr(
        live_runner,
        "_load_primary",
        lambda *_args, **_kwargs: (
            pd.Series(45.0, index=index),
            pd.Series(schedule.cutoff_local.tz_convert("UTC"), index=index),
            {"coverage": 1.0},
        ),
    )

    def fake_copy(_benchmark: Path, staging: Path) -> None:
        (staging / "metrics_hourly.json").write_text(
            json.dumps(
                {
                    "metrics": [{"model": "mkonline_blend", "mae": 11.23}],
                    "training_diagnostics": {"mkonline_blend": {}},
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(live_runner, "_copy_benchmark_for_report", fake_copy)

    def fake_history(**kwargs):
        events.append("history")
        staging = Path(kwargs["staging_run_dir"])
        pd.DataFrame(
            {
                "delivery_start_utc": schedule.delivery_index,
                "actual": 40.0,
                "mkonline_blend__q10": 35.0,
                "mkonline_blend__q50": 40.0,
                "mkonline_blend__q90": 45.0,
                "storm_evaluation_only__q50": 42.0,
            }
        ).to_csv(
            staging / "statistics_history_hourly.csv.gz",
            index=False,
            compression="gzip",
        )
        (staging / "statistics_history_audit.json").write_text(
            json.dumps({"report_scope_note": "test scope"}),
            encoding="utf-8",
        )
        return {"status": "complete", "run_type": (
            "pit_replay" if pit_replay else "live_day_ahead"
        )}

    monkeypatch.setattr(
        live_runner,
        "update_live_statistics_history",
        fake_history,
    )
    native_index = pd.date_range("2025-08-12", periods=24, freq="h")
    monkeypatch.setattr(
        live_runner,
        "_load_storm_dashboard_statistics_snapshot",
        lambda **_kwargs: (
            pd.Series(42.0, index=native_index),
            {
                "requested_series": "power.price.fr.euromwh.h.fcst.3mv.storm",
                "used_for_prediction": False,
            },
        ),
    )

    def fake_report(run_dir, *, output_path, **_kwargs):
        staging = Path(run_dir)
        assert events == ["history"]
        assert (staging / "statistics_history_hourly.csv.gz").is_file()
        assert (staging / "statistics_history_audit.json").is_file()
        events.append("html")
        return Path(output_path).write_text("<html>live</html>", encoding="utf-8")

    monkeypatch.setattr(
        live_runner,
        "write_hourly_html_report",
        fake_report,
    )
    monkeypatch.setattr(
        live_runner,
        "_write_checksums",
        lambda staging, **_kwargs: (staging / "artifact_checksums.json").write_text(
            "{}", encoding="utf-8"
        ),
    )
    if not pit_replay:
        import chronos2_hourly.rolling_capture as capture_module

        def fail_capture_after_publish(**kwargs):
            events.append("capture")
            archive = Path(kwargs["issued_live_archive"])
            assert archive.is_dir()
            assert (archive / "forecast_hourly_fr.csv").is_file()
            raise RuntimeError("capture failure after publication")

        monkeypatch.setattr(
            capture_module,
            "capture_supported_issued_live_block_isolated",
            fail_capture_after_publish,
        )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mkonline_live_hourly.py",
            "--config",
            str(config),
            "--data-as-of",
            as_of,
            "--delivery-day",
            delivery_day,
            *(
                ["--rolling365-capture-root", str(tmp_path / "rolling")]
                if not pit_replay
                else []
            ),
            *(["--pit-replay"] if pit_replay else []),
        ],
    )

    assert live_runner.main() == 0
    forecast = pd.read_csv(output / "forecast_hourly_fr.csv")
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((output / "metrics_hourly.json").read_text(encoding="utf-8"))
    assert len(forecast) == 24
    assert manifest["config"] == str(config)
    expected_run_type = "pit_replay" if pit_replay else "live_day_ahead"
    assert manifest["delivery_day_local"] == delivery_day
    assert manifest["run_type"] == expected_run_type
    assert manifest["statistics_history"]["run_type"] == expected_run_type
    assert manifest["target_availability"] == availability
    assert metrics["forecast_diagnostics"]["delivery_day_local"] == delivery_day
    assert (output / f"live_{delivery_day}.html").is_file()
    assert events == (
        ["history", "html", "capture"]
        if not pit_replay
        else ["history", "html"]
    )
    assert ("_replays" in output.parts) is pit_replay
