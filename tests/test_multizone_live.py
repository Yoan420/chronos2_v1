from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.multizone_contract import (
    ZoneArtifactHashes,
    ZoneBlendWeights,
    ZoneModelContract,
    ZoneModelPaths,
)
from chronos2_hourly.multizone_live import (
    CandidateArtifacts,
    LiveRuntimeOptions,
    LiveSchedule,
    PredictionPolicy,
    ZoneLiveExecutionError,
    ZoneLiveHooks,
    blend_quantiles,
    load_prediction_policy,
    load_primary_materialization,
    materialize_primary,
    resolve_live_schedule,
    run_zone_live,
    validate_live_forecast,
)


ZONE = {
    "BE": ("Europe/Brussels", "41555_native", "power.price.da.be"),
    "DE": ("Europe/Berlin", "41550_native", "power.price.da.de"),
    "NL": ("Europe/Amsterdam", "41554_native", "power.price.da.nl"),
    "ES": ("Europe/Madrid", "58307_native", "power.price.da.es"),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _contract(
    tmp_path: Path,
    *,
    zone: str = "BE",
    mode: str = "mkonline_blend",
) -> tuple[ZoneModelContract, dict[str, Any]]:
    timezone, primary, target = ZONE[zone]
    root = tmp_path / zone.lower()
    root.mkdir(parents=True)
    frozen = root / f"autonomous_{zone.lower()}"
    benchmark = root / f"benchmark_{zone.lower()}"
    frozen.mkdir()
    benchmark.mkdir()
    files = {
        "live": root / f"live_{zone.lower()}.yaml",
        "registry": root / "registry.yaml",
        "base": root / f"base_{zone.lower()}.yaml",
        "recipe": root / f"recipe_{zone.lower()}.json",
        "dependency": root / f"dependency_{zone.lower()}.json",
    }
    for key in ("live", "registry", "base"):
        files[key].write_text(key, encoding="utf-8")
    enabled = mode == "mkonline_blend"
    if enabled:
        files["dependency"].write_text("dependency", encoding="utf-8")
    weights = (0.6, 0.4) if enabled else (1.0, 0.0)
    recipe = {
        "prediction_mode": mode,
        "mkonline_enabled": enabled,
    }
    if not enabled:
        recipe["autonomous_only_reason"] = "B1/B2 gate rejected MKOnline"
    _write_json(files["recipe"], recipe)
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
                "storm_used_as_feature": False,
                "prediction_inputs": ["autonomous"],
            },
        )
    candidate_model = "mkonline_blend" if enabled else "residual_corrected"
    pd.DataFrame(
        {
            "delivery_start_utc": ["2026-08-12T00:00:00Z"],
            "actual": [50.0],
            f"{candidate_model}__q10": [45.0],
            f"{candidate_model}__q50": [50.0],
            f"{candidate_model}__q90": [55.0],
        }
    ).to_csv(benchmark / "backtest_hourly_oof.csv.gz", index=False)
    _write_json(
        benchmark / "metrics_hourly.json",
        {
            "training_diagnostics": {
                "evaluation_start_local_date": "2025-08-12",
                "evaluation_end_local_date": "2026-08-11",
            }
        },
    )
    paths = ZoneModelPaths(
        live_config=files["live"],
        registry=files["registry"],
        base_config=files["base"],
        frozen_autonomous_run=frozen,
        sealed_benchmark_run=benchmark,
        recipe_manifest=files["recipe"],
        dependency_manifest=files["dependency"] if enabled else None,
        output_root=root / f"live_{zone.lower()}_runs",
    )
    hashes = ZoneArtifactHashes(
        base_config_sha256="a" * 64,
        frozen_autonomous_checksum_manifest_sha256=_sha(
            frozen / "artifact_checksums.json"
        ),
        sealed_benchmark_checksum_manifest_sha256=_sha(
            benchmark / "artifact_checksums.json"
        ),
        recipe_manifest_sha256=_sha(files["recipe"]),
        dependency_manifest_sha256=(
            _sha(files["dependency"]) if enabled else None
        ),
    )
    dashboard = None if zone == "ES" else f"power.price.{zone.lower()}.storm"
    contract = ZoneModelContract(
        schema_version=1,
        zone=zone,
        delivery_timezone=timezone,
        forecast_origin_timezone="Europe/Paris",
        forecast_origin_local_time="08:00",
        target_series=target,
        primary_series=primary if enabled else None,
        storm_dashboard_series=dashboard,
        storm_dashboard_primary_series=None,
        storm_dashboard_naive_timezone=timezone if dashboard else None,
        storm_strict_08_series=f"power.price.{zone.lower()}.storm.da.basecase",
        forecast_filename=f"forecast_hourly_{zone.lower()}.csv",
        required_covariates=(f"{zone.lower()}_load_fcst",),
        paths=paths,
        checksum_hashes=hashes,
        weights=ZoneBlendWeights(
            autonomous=weights[0],
            mkonline_primary=weights[1],
        ),
        prediction_mode=mode,
        mkonline_enabled=enabled,
    )
    live = {
        "prediction_mode": mode,
        "mkonline_enabled": enabled,
        "threads": 1,
        "workers": 1,
        "report": {
            "filename": f"report_{zone.lower()}_{{delivery_day}}.html",
        },
    }
    return contract, live


@pytest.mark.parametrize(
    ("zone", "as_of", "delivery_day", "hours"),
    [
        ("BE", "2026-03-28T08:00:00+01:00", "2026-03-29", 23),
        ("DE", "2026-10-24T08:00:00+02:00", "2026-10-25", 25),
        ("NL", "2026-08-13T08:00:00+02:00", "2026-08-14", 24),
        ("ES", "2026-08-13T08:00:00+02:00", "2026-08-14", 24),
    ],
)
def test_schedule_uses_zone_delivery_dst_and_paris_origin(
    tmp_path: Path,
    zone: str,
    as_of: str,
    delivery_day: str,
    hours: int,
) -> None:
    contract, _live = _contract(tmp_path, zone=zone)
    schedule = resolve_live_schedule(
        contract,
        as_of=as_of,
        delivery_day=delivery_day,
    )
    assert len(schedule.delivery_index) == hours
    assert str(schedule.delivery_index.tz) == "UTC"
    assert str(schedule.cutoff_origin_local.tz) == "Europe/Paris"


def test_schedule_rejects_before_cutoff_and_non_jplus1(tmp_path: Path) -> None:
    contract, _live = _contract(tmp_path)
    with pytest.raises(ZoneLiveExecutionError, match="before civil cutoff"):
        resolve_live_schedule(
            contract,
            as_of="2026-08-13T07:59:59+02:00",
            delivery_day="2026-08-14",
        )
    with pytest.raises(ZoneLiveExecutionError, match=r"J\+1 only"):
        resolve_live_schedule(
            contract,
            as_of="2026-08-13T08:00:00+02:00",
            delivery_day="2026-08-15",
        )


def test_prediction_policy_is_explicit_and_mirrored_in_recipe(tmp_path: Path) -> None:
    contract, live = _contract(tmp_path)
    policy = load_prediction_policy(contract, live)
    assert policy == PredictionPolicy("mkonline_blend", True, "mkonline_blend")
    with pytest.raises(ZoneLiveExecutionError, match="must be declared"):
        load_prediction_policy(contract, {})
    live["mkonline_enabled"] = False
    with pytest.raises(ZoneLiveExecutionError, match="mkonline_enabled"):
        load_prediction_policy(contract, live)


def test_autonomous_policy_requires_explicit_reason_and_exact_weights(
    tmp_path: Path,
) -> None:
    contract, live = _contract(tmp_path, zone="ES", mode="autonomous_only")
    policy = load_prediction_policy(contract, live)
    assert policy.candidate_model == "residual_corrected"
    recipe = json.loads(contract.paths.recipe_manifest.read_text(encoding="utf-8"))
    recipe.pop("autonomous_only_reason")
    _write_json(contract.paths.recipe_manifest, recipe)
    with pytest.raises(ZoneLiveExecutionError, match="explicit recipe reason"):
        load_prediction_policy(contract, live)


def test_blend_preserves_quantile_width_and_uses_declared_weights() -> None:
    index = pd.date_range("2026-08-13", periods=3, freq="h", tz="UTC")
    autonomous = pd.DataFrame(
        {"q10": [10, 20, 30], "q50": [20, 30, 40], "q90": [35, 45, 55]},
        index=index,
    )
    primary = pd.Series([40, 50, 60], index=index)
    result = blend_quantiles(
        autonomous,
        primary,
        autonomous_weight=0.6,
        primary_weight=0.4,
    )
    assert np.allclose(result["q50"], [28, 38, 48])
    assert np.allclose(result["q90"] - result["q10"], [25, 25, 25])


def _forecast(schedule: LiveSchedule, candidate_model: str) -> pd.DataFrame:
    n = len(schedule.delivery_index)
    values = np.arange(n, dtype=float) + 50.0
    frame = pd.DataFrame(
        {
            "delivery_start_utc": schedule.delivery_index,
            "forecast_origin_utc": schedule.cutoff_origin_local.tz_convert("UTC"),
            "q10": values - 5,
            "q50": values,
            "q90": values + 5,
            f"{candidate_model}__q10": values - 5,
            f"{candidate_model}__q50": values,
            f"{candidate_model}__q90": values + 5,
        }
    )
    return frame


def test_live_forecast_validation_rejects_timeline_and_noncausal_origin(
    tmp_path: Path,
) -> None:
    contract, _live = _contract(tmp_path)
    schedule = resolve_live_schedule(
        contract,
        as_of="2026-08-13T08:00:00+02:00",
        delivery_day="2026-08-14",
    )
    frame = _forecast(schedule, "mkonline_blend")
    assert len(
        validate_live_forecast(
            frame,
            schedule=schedule,
            candidate_model="mkonline_blend",
        )
    ) == 24
    with pytest.raises(ZoneLiveExecutionError, match="23/24/25"):
        validate_live_forecast(
            frame.iloc[:-1],
            schedule=schedule,
            candidate_model="mkonline_blend",
        )
    bad = frame.copy()
    bad["forecast_origin_utc"] = bad["delivery_start_utc"]
    with pytest.raises(ZoneLiveExecutionError, match="non-causal"):
        validate_live_forecast(
            bad,
            schedule=schedule,
            candidate_model="mkonline_blend",
        )
    inconsistent = frame.copy()
    inconsistent["q50"] += 1.0
    with pytest.raises(ZoneLiveExecutionError, match="declared candidate model"):
        validate_live_forecast(
            inconsistent,
            schedule=schedule,
            candidate_model="mkonline_blend",
        )


def test_primary_loader_enforces_exact_paris_cutoff(
    tmp_path: Path,
) -> None:
    contract, _live = _contract(tmp_path, zone="ES")
    schedule = resolve_live_schedule(
        contract,
        as_of="2026-08-13T08:00:00+02:00",
        delivery_day="2026-08-14",
    )
    cutoff = schedule.cutoff_origin_local.tz_convert("UTC")
    path = tmp_path / "primary.parquet"
    pd.DataFrame(
        {
            "value_time_utc": schedule.delivery_index,
            "snapshot_time_utc": cutoff,
            "revision_time_utc": cutoff,
            "value": np.arange(len(schedule.delivery_index), dtype=float),
        }
    ).to_parquet(path, index=False)
    values, selected_cutoff, audit = load_primary_materialization(
        path,
        contract=contract,
        schedule=schedule,
    )
    assert len(values) == 24
    assert selected_cutoff.iloc[0] == cutoff
    assert audit["series"] == "58307_native"
    frame = pd.read_parquet(path)
    frame["revision_time_utc"] = cutoff + pd.Timedelta(minutes=1)
    frame.to_parquet(path, index=False)
    with pytest.raises(ZoneLiveExecutionError, match="PIT marker"):
        load_primary_materialization(path, contract=contract, schedule=schedule)


def test_materializer_command_is_zone_parameterised_and_no_fr_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, _live = _contract(tmp_path, zone="DE")
    schedule = resolve_live_schedule(
        contract,
        as_of="2026-08-13T08:00:00+02:00",
        delivery_day="2026-08-14",
    )
    captured: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> None:
        captured.append(command)

    monkeypatch.setattr("chronos2_hourly.multizone_live.subprocess.run", fake_run)
    command = materialize_primary(
        project_root=tmp_path,
        contract=contract,
        schedule=schedule,
        output=tmp_path / "de.parquet",
        workers=3,
    )
    assert captured == [command]
    assert command[command.index("--series") + 1] == "41550_native"
    assert command[command.index("--alias") + 1] == "mkonline_de_primary"
    assert command[command.index("--timezone") + 1] == "Europe/Berlin"
    assert command[command.index("--cutoff-timezone") + 1] == "Europe/Paris"
    assert "41551_native" not in command


@pytest.mark.parametrize(
    ("zone", "mode", "dashboard_expected"),
    [
        ("BE", "mkonline_blend", True),
        ("ES", "autonomous_only", False),
    ],
)
def test_mocked_end_to_end_freezes_candidate_before_report_only_storm(
    tmp_path: Path,
    zone: str,
    mode: str,
    dashboard_expected: bool,
) -> None:
    contract, live = _contract(tmp_path, zone=zone, mode=mode)
    events: list[str] = []
    delivery = "2026-08-14"
    as_of = "2026-08-13T08:00:00+02:00"
    wall_clock = pd.Timestamp("2026-08-13T09:00:00+02:00")

    def build(
        received: ZoneModelContract,
        schedule: LiveSchedule,
        policy: PredictionPolicy,
        _live: Any,
        _options: Any,
        staging: Path,
    ) -> CandidateArtifacts:
        events.append("candidate")
        assert received.zone == zone
        assert not (staging / received.forecast_filename).exists()
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

    def dashboard(
        received: ZoneModelContract,
        schedule: LiveSchedule,
        _data: Any,
    ) -> tuple[pd.Series | None, dict[str, Any]]:
        events.append("storm")
        candidate_path = next(received.paths.output_root.glob(".*.tmp-*/")) / received.forecast_filename
        assert candidate_path.is_file()
        if not dashboard_expected:
            return None, {"status": "native_dashboard_unavailable"}
        raw_index = schedule.delivery_index.tz_convert(
            received.delivery_timezone
        ).tz_localize(None)
        return pd.Series(np.arange(len(raw_index)), index=raw_index), {
            "series": received.storm_dashboard_series,
            "used_for_prediction": False,
        }

    def statistics(**kwargs: Any) -> dict[str, Any]:
        events.append("statistics")
        assert kwargs["forecast_name"] == contract.forecast_filename
        expected_model = "mkonline_blend" if mode == "mkonline_blend" else "residual_corrected"
        assert kwargs["candidate_model"] == expected_model
        assert (kwargs["storm_dashboard_native"] is not None) is dashboard_expected
        return {"status": "mocked"}

    def report(
        _run_dir: Path,
        *,
        output_path: Path,
        **kwargs: Any,
    ) -> Path:
        events.append("report")
        assert kwargs["zone"] == zone
        output_path.write_text("<html>ok</html>", encoding="utf-8")
        return output_path

    hooks = ZoneLiveHooks(build, dashboard, statistics, report)
    output = run_zone_live(
        contract,
        live_settings=live,
        data_as_of=as_of,
        delivery_day=delivery,
        options=LiveRuntimeOptions(threads=1, workers=1),
        hooks=hooks,
        wall_clock=wall_clock,
    )
    assert events == ["candidate", "storm", "statistics", "report"]
    assert output.name == f"{zone.lower()}_day_ahead_{delivery}"
    assert (output / contract.forecast_filename).is_file()
    assert (output / "artifact_checksums.json").is_file()
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["zone"] == zone
    assert manifest["prediction_mode"] == mode
    assert manifest["storm_loaded_for_prediction"] is False
    assert (
        manifest["storm_dashboard_loaded_after_candidate_frozen_for_statistics"]
        is dashboard_expected
    )
    prediction_inputs = manifest["prediction_inputs"]
    assert (contract.primary_series in prediction_inputs) == (mode == "mkonline_blend")


def test_run_rejects_fr_and_output_escape(tmp_path: Path) -> None:
    contract, live = _contract(tmp_path)
    fr = ZoneModelContract(**{**contract.__dict__, "zone": "FR"})
    with pytest.raises(ZoneLiveExecutionError, match="France runner"):
        run_zone_live(fr, live_settings=live)
    schedule = resolve_live_schedule(
        contract,
        as_of="2026-08-13T08:00:00+02:00",
        delivery_day="2026-08-14",
    )
    from chronos2_hourly.multizone_live import _validated_output

    with pytest.raises(ZoneLiveExecutionError, match="zone root"):
        _validated_output(
            contract,
            schedule,
            output_dir=tmp_path / "escaped",
            pit_replay=False,
        )


def test_optional_rolling_capture_failure_after_publish_is_non_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, live = _contract(tmp_path, zone="ES", mode="autonomous_only")
    delivery = "2026-08-14"
    events: list[str] = []

    def build(
        _contract_value: ZoneModelContract,
        schedule: LiveSchedule,
        policy: PredictionPolicy,
        _live: Any,
        _options: Any,
        _staging: Path,
    ) -> CandidateArtifacts:
        events.append("candidate")
        raw = pd.DataFrame(
            {
                "forecast_origin_utc": schedule.cutoff_origin_local.tz_convert("UTC"),
                "q10": 40.0,
                "q50": 50.0,
                "q90": 60.0,
            },
            index=schedule.delivery_index,
        )
        return CandidateArtifacts(
            forecast=_forecast(schedule, policy.candidate_model),
            canonical_target=pd.Series(50.0, index=schedule.delivery_index),
            fit_audit={},
            input_diagnostics={"target": {}},
            pit_freshness={},
            source_paths={},
            data_config={},
            rolling_capture_features=pd.DataFrame(
                {"known_fr_residual_load_fcst_oracle": 1.0},
                index=schedule.delivery_index,
            ),
            rolling_capture_chronos=raw,
        )

    def report(_run_dir: Path, *, output_path: Path, **_kwargs: Any) -> Path:
        events.append("report")
        output_path.write_text("<html>ok</html>", encoding="utf-8")
        return output_path

    hooks = ZoneLiveHooks(
        build,
        lambda *_args, **_kwargs: (None, {"used_for_prediction": False}),
        lambda **_kwargs: {"status": "mocked"},
        report,
    )

    import chronos2_hourly.rolling_capture as capture_module

    def fail_after_publish(**kwargs: Any):
        events.append("capture")
        archive = Path(kwargs["issued_live_archive"])
        assert archive.is_dir()
        assert (archive / contract.forecast_filename).is_file()
        raise RuntimeError("shadow capture failure")

    monkeypatch.setattr(
        capture_module,
        "capture_supported_issued_live_block_isolated",
        fail_after_publish,
    )
    output = run_zone_live(
        contract,
        live_settings=live,
        data_as_of="2026-08-13T08:00:00+02:00",
        delivery_day=delivery,
        options=LiveRuntimeOptions(
            threads=1,
            workers=1,
            rolling365_capture_root=tmp_path / "rolling",
        ),
        hooks=hooks,
        wall_clock=pd.Timestamp("2026-08-13T09:00:00+02:00"),
    )
    assert output.is_dir()
    assert (output / contract.forecast_filename).is_file()
    assert events[-2:] == ["report", "capture"]
