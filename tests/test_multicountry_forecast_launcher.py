from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import run_multicountry_forecast as launcher
from chronos2_hourly.app_service import ForecastProcess, ForecastSkip, ZoneStatus
from chronos2_hourly.hourly_contract import local_delivery_day_index


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


def _exportable_archive(tmp_path: Path, zone: str) -> Path:
    archive = _archive(tmp_path, zone)
    delivery = pd.date_range("2026-08-19T22:00:00Z", periods=2, freq="h")
    forecast = pd.DataFrame(
        {
            "delivery_start_utc": delivery.astype(str),
            "delivery_start_local": delivery.astype(str),
            "chronos2__q10": [30.0, 31.0],
            "chronos2__q50": [40.0, 41.0],
            "chronos2__q90": [50.0, 51.0],
            "residual_corrected__q10": [32.0, 33.0],
            "residual_corrected__q50": [42.0, 43.0],
            "residual_corrected__q90": [52.0, 53.0],
            "mkonline_blend__q10": [34.0, 35.0],
            "mkonline_blend__q50": [44.0, 45.0],
            "mkonline_blend__q90": [54.0, 55.0],
        }
    )
    forecast.to_csv(archive / f"forecast_hourly_{zone.lower()}.csv", index=False)
    backtest = forecast.copy()
    backtest["actual"] = [41.0, 44.0]
    backtest.to_csv(
        archive / "backtest_hourly_oof.csv.gz",
        index=False,
        compression="gzip",
    )
    inputs = archive / "inputs"
    inputs.mkdir()
    for filename in launcher._REPORT_INPUT_FILES:
        (inputs / filename).write_text("placeholder\n", encoding="utf-8")
    return archive


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


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
    # These launcher fixtures have no production, checksum-pinned prefixes.
    # Loader argument/validation coverage below uses dedicated explicit mocks.
    monkeypatch.setattr(
        "chronos2_hourly.kalman_configuration.load_kalman_operational_configuration",
        lambda *_args, **_kwargs: SimpleNamespace(training_lookback_days=365),
    )


def _patch_promoted_lora_all(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    zones: tuple[str, ...],
) -> list[str]:
    """Provide a tiny final-pipeline runtime for unrelated All-mode tests."""

    prepared_calls: list[str] = []
    run_root = tmp_path / "lora-runs"
    run_root.mkdir(exist_ok=True)
    checksum = run_root / "artifact_checksums.json"
    checksum.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "live-sources.json"
    manifest.write_text("{}\n", encoding="utf-8")
    bundles = {
        zone: SimpleNamespace(
            alias=f"lora-{zone.lower()}-v1",
            candidate_id=f"candidate-{zone.lower()}",
            candidate_model="chronos2-exogenous-final",
            bundle_manifest_sha256=(zone.lower()[0] * 64),
            artifact_checksums_sha256="b" * 64,
            schema_path=tmp_path / "schema.json",
            experiment_manifest_path=tmp_path / "experiment.json",
        )
        for zone in zones
    }

    class FakeContract:
        project_root = tmp_path.resolve()
        registry_path = tmp_path / "lora-registry.json"

        def resolve(self, *, zone: str, mode: str):
            return launcher.ActivationResolution(
                zone=zone,
                mode=mode,
                lora_enabled=True,
                selected_model=(
                    "exogenous_residual_corrected"
                    if mode == "autonomous"
                    else "residual_kalman"
                ),
                fallback_model=(
                    "residual_corrected"
                    if mode == "autonomous"
                    else "residual_kalman"
                ),
                kalman_upstream_model=(
                    "exogenous_residual_corrected" if mode == "kalman" else None
                ),
                fallback_upstream_model=(
                    "residual_corrected" if mode == "kalman" else None
                ),
                bundle=bundles[zone],
                live_source_manifest_path=manifest,
                live_source_manifest_sha256="c" * 64,
                runtime_output_root=run_root,
                reason="synthetic_promoted_bundle",
            )

    contract = FakeContract()
    monkeypatch.setattr(launcher, "load_activation_contract", lambda _path: contract)

    def prepare(routes, **_kwargs):
        result = {}
        for zone in zones:
            prepared_calls.append(zone)
            resolution = routes[(zone, "autonomous")]
            result[zone] = launcher._LoRACandidateRuntime(
                resolution=resolution,
                run=SimpleNamespace(
                    zone=zone,
                    output_directory=run_root,
                ),
                history=pd.DataFrame(),
                forecast=pd.DataFrame(),
                history_start_utc="2025-08-20T00:00:00+00:00",
                history_end_utc="2026-08-19T23:00:00+00:00",
                issued_days_reused=(),
            )
        return result

    monkeypatch.setattr(launcher, "_prepare_lora_candidates", prepare)

    def overlay(spec, destination):
        if spec.lora_candidate is None:
            return
        view = Path(destination)
        for path in (
            launcher._forecast_path(view, spec.zone),
            view / "backtest_hourly_oof.csv.gz",
        ):
            frame = pd.read_csv(path)
            for quantile in ("q10", "q50", "q90"):
                frame[f"exogenous_residual_corrected__{quantile}"] = (
                    frame[f"residual_corrected__{quantile}"] + 0.5
                )
            frame.to_csv(
                path,
                index=False,
                compression="gzip" if path.name.endswith(".gz") else None,
            )
        (view / "exogenous_lora_overlay_audit.json").write_text(
            json.dumps({"status": "complete", "pipeline": "final"}),
            encoding="utf-8",
        )

    monkeypatch.setattr(launcher, "_materialize_lora_reporting_view", overlay)
    # The two-hour synthetic archive is intentionally not a 730-day history.
    monkeypatch.setattr(
        launcher, "_preflight_kalman_reporting_view", lambda *_args: None,
    )
    return prepared_calls


def test_promoted_lora_overlay_uses_final_corrected_pipeline_and_fallback_warmup(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "view"
    destination.mkdir()
    old = pd.Timestamp("2026-08-17T22:00:00Z")
    historical = pd.Timestamp("2026-08-18T22:00:00Z")
    future = pd.Timestamp("2026-08-19T22:00:00Z")

    def incumbent_frame(index: list[pd.Timestamp]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "delivery_start_utc": [str(value) for value in index],
                "actual": [55.0] * len(index),
                "residual_corrected__q10": [40.0] * len(index),
                "residual_corrected__q50": [50.0] * len(index),
                "residual_corrected__q90": [60.0] * len(index),
            }
        )

    incumbent_frame([old, historical]).to_csv(
        destination / "backtest_hourly_oof.csv.gz",
        index=False,
        compression="gzip",
    )
    incumbent_frame([old, historical, future]).to_csv(
        destination / "statistics_history_hourly.csv.gz",
        index=False,
        compression="gzip",
    )
    incumbent_frame([future]).drop(columns="actual").to_csv(
        destination / "forecast_hourly_fr.csv", index=False
    )
    (destination / "statistics_history_audit.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (destination / "run_manifest.json").write_text("{}\n", encoding="utf-8")

    bundle = SimpleNamespace(
        alias="fr-lora-final-v1",
        candidate_id="fr-final",
        candidate_model="chronos2-exogenous-final",
        bundle_manifest_sha256="a" * 64,
        artifact_checksums_sha256="b" * 64,
    )
    resolution = launcher.ActivationResolution(
        zone="FR",
        mode="autonomous",
        lora_enabled=True,
        selected_model="exogenous_residual_corrected",
        fallback_model="residual_corrected",
        kalman_upstream_model=None,
        fallback_upstream_model=None,
        bundle=bundle,
        live_source_manifest_path=tmp_path / "live.json",
        live_source_manifest_sha256="c" * 64,
        runtime_output_root=tmp_path / "runs",
        reason="promoted_bundle_verified",
    )
    run_dir = tmp_path / "candidate"
    run_dir.mkdir()
    (run_dir / "artifact_checksums.json").write_text("{}\n", encoding="utf-8")
    history = pd.DataFrame(
        {
            "exogenous_residual_corrected__q10": [70.0],
            "exogenous_residual_corrected__q50": [80.0],
            "exogenous_residual_corrected__q90": [90.0],
        },
        index=pd.DatetimeIndex([historical], name="delivery_start_utc"),
    )
    forecast = pd.DataFrame(
        {
            "exogenous_residual_corrected__q10": [71.0],
            "exogenous_residual_corrected__q50": [81.0],
            "exogenous_residual_corrected__q90": [91.0],
            "chronos2_exogenous__q10": [69.0],
            "chronos2_exogenous__q50": [79.0],
            "chronos2_exogenous__q90": [89.0],
            "forecast_origin_utc": ["2026-08-19T06:00:00+00:00"],
        },
        index=pd.DatetimeIndex([future], name="delivery_start_utc"),
    )
    runtime = launcher._LoRACandidateRuntime(
        resolution=resolution,
        run=SimpleNamespace(zone="FR", output_directory=run_dir),
        history=history,
        forecast=forecast,
        history_start_utc=str(historical),
        history_end_utc=str(historical),
        issued_days_reused=(),
    )
    spec = launcher._ExportSpec(
        zone="FR",
        timezone="Europe/Paris",
        delivery_day="2026-08-20",
        variant="autonomous",
        source_model="exogenous_residual_corrected",
        baseline_model="residual_corrected",
        archive=tmp_path / "immutable-live",
        history_archive=tmp_path / "immutable-live",
        csv_path=tmp_path / "forecast.csv",
        report_path=tmp_path / "forecast.html",
        history_contract=None,
        lora_resolution=resolution,
        lora_candidate=runtime,
    )

    launcher._materialize_lora_reporting_view(spec, destination)

    statistics = pd.read_csv(destination / "statistics_history_hourly.csv.gz")
    assert statistics["exogenous_residual_corrected__q50"].tolist() == [
        50.0,
        80.0,
        81.0,
    ]
    assert statistics["residual_correction"].tolist() == [0.0, 30.0, 31.0]
    live = pd.read_csv(destination / "forecast_hourly_fr.csv")
    assert live["chronos2_exogenous__q50"].tolist() == [79.0]
    assert live["exogenous_residual_corrected__q50"].tolist() == [81.0]
    audit = json.loads(
        (destination / "exogenous_lora_overlay_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["raw_lora_checkpoint_used_as_report_model"] is False
    assert audit["final_pipeline_model"] == "exogenous_residual_corrected"
    assert audit["warmup_fallback_is_outside_scored_final365"] is True


def test_kalman_validator_accepts_promoted_lora_as_dynamic_upstream(
    tmp_path: Path,
) -> None:
    requested = date(2026, 8, 20)
    evaluation_start = requested - pd.Timedelta(days=365)
    evaluation_index = pd.date_range(
        pd.Timestamp(evaluation_start, tz="Europe/Paris"),
        pd.Timestamp(requested, tz="Europe/Paris"),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    future_index = pd.date_range(
        pd.Timestamp(requested, tz="Europe/Paris"),
        pd.Timestamp(requested + pd.Timedelta(days=1), tz="Europe/Paris"),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")

    def output_frame(length: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "residual_kalman__q10": np.full(length, 10.0),
                "residual_kalman__q50": np.full(length, 20.0),
                "residual_kalman__q90": np.full(length, 30.0),
            }
        )

    replay = SimpleNamespace(
        audit={
            "status": "complete",
            "model_key": "residual_kalman",
            "upstream_model": "exogenous_residual_corrected",
            "filter_only": True,
            "smoother_used": False,
            "em_used": False,
            "storm_used_as_input": False,
            "causality_violations": 0,
            "quantile_crossings": 0,
            "evaluation_days": 365,
            "evaluation_hours": len(evaluation_index),
            "training_policy": "fixed_length_rolling_local_days",
            "training_lookback_days": 365,
            "warmup_days": 365,
            "future_observations_assimilated": 0,
        }
    )
    view = SimpleNamespace(
        replay=replay,
        evaluation_start_day=pd.Timestamp(evaluation_start).date(),
        evaluation_end_day=requested - pd.Timedelta(days=1),
        evaluation_index=evaluation_index,
        future_index=future_index,
        statistics=output_frame(len(evaluation_index)),
        backtest=output_frame(len(evaluation_index)),
        forecast=output_frame(len(future_index)),
    )
    spec = launcher._ExportSpec(
        zone="FR",
        timezone="Europe/Paris",
        delivery_day=requested.isoformat(),
        variant="kalman",
        source_model="residual_kalman",
        baseline_model="exogenous_residual_corrected",
        archive=tmp_path,
        history_archive=tmp_path,
        csv_path=tmp_path / "forecast.csv",
        report_path=tmp_path / "forecast.html",
        history_contract=None,
    )

    launcher._validate_materialized_kalman_view(spec, view)


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


@pytest.mark.parametrize("mode", ("All", "Both"))
@pytest.mark.parametrize("zone", ("FR", "DE", "BE", "NL", "ES"))
def test_all_and_both_export_standard_kalman_for_every_country(mode, zone) -> None:
    assert launcher.normalise_forecast_mode(mode) == mode.lower()
    assert launcher.requested_export_variants(mode, zone) == (
        "autonomous",
        "kalman",
    )
    assert launcher._model_for_variant("kalman") == (
        "residual_kalman",
        "residual_corrected",
    )


@pytest.mark.parametrize("zone", ("FR", "DE", "BE", "NL", "ES"))
def test_explicit_blend_keeps_its_promoted_country_scope(zone) -> None:
    if zone in {"FR", "NL"}:
        assert launcher.requested_export_variants("Blend", zone) == ("blend",)
    else:
        with pytest.raises(ValueError, match="FR et NL"):
            launcher.requested_export_variants("Blend", zone)


def test_experimental_kalman_variants_keep_distinct_models() -> None:
    assert launcher._model_for_variant("kalman_weather") == (
        "residual_kalman_weather",
        "residual_corrected",
    )
    assert launcher._model_for_variant("kalman_hybrid") == (
        "residual_kalman_hybrid",
        "residual_corrected",
    )


@pytest.mark.parametrize("mode", ("All", "Both"))
def test_two_model_modes_do_not_require_experimental_kalman_sidecars(
    tmp_path: Path,
    mode: str,
) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    result = launcher.BatchZoneResult(
        zone="FR",
        delivery_day="2026-08-20",
        state="success",
        return_code=0,
        message="ok",
        archive_path=archive,
    )

    specs = launcher._build_export_specs(
        zone_results=(result,),
        statuses={"FR": _status(tmp_path, "FR")},
        mode=mode,
        delivery_day="2026-08-20",
        export_root=tmp_path / "exports",
        project_root=tmp_path,
    )

    assert [spec.variant for spec in specs] == [
        "autonomous",
        "kalman",
    ]
    assert all(
        spec.variant not in {"kalman_weather", "kalman_hybrid"}
        for spec in specs
    )


def test_all_refuses_inactive_lora_before_any_live_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    calls = SimpleNamespace(inspect=0)

    def forbidden_inspect(*_args, **_kwargs):
        calls.inspect += 1
        raise AssertionError("All sans LoRA doit echouer avant le preflight live")

    monkeypatch.setattr(launcher, "inspect_zone_statuses", forbidden_inspect)
    with pytest.raises(ValueError, match="aucun fallback incumbent|Bundle final promu"):
        launcher.run_forecast_batch(
            zones=("FR", "DE"),
            delivery_day="2026-08-20",
            project_root=tmp_path,
            registry_path=registry,
            python_executable=executable,
            mode="All",
        )
    assert calls.inspect == 0


def test_inactive_lora_contract_keeps_both_routes_on_incumbent() -> None:
    contract = launcher.load_activation_contract(
        launcher.DEFAULT_LORA_ACTIVATION_CONFIG
    )
    routes = launcher._resolve_lora_routes(
        contract,
        zones=("FR", "DE", "BE", "NL", "ES"),
        mode="Both",
    )

    assert set(routes) == {
        (zone, variant)
        for zone in ("FR", "DE", "BE", "NL", "ES")
        for variant in ("autonomous", "kalman")
    }
    for zone in ("FR", "DE", "BE", "NL", "ES"):
        autonomous = routes[(zone, "autonomous")]
        kalman = routes[(zone, "kalman")]
        assert autonomous.lora_enabled is False
        assert autonomous.selected_model == "residual_corrected"
        assert kalman.lora_enabled is False
        assert kalman.selected_model == "residual_kalman"
        assert kalman.kalman_upstream_model == "residual_corrected"


def test_both_validates_standard_kalman_before_any_country_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    events = []
    config = tmp_path / "isolated_standard_kalman.yaml"

    def validate_kalman(path, *, project_root, zone, upstream_model):
        assert Path(path) == config.resolve()
        assert Path(project_root) == tmp_path.resolve()
        assert upstream_model == "residual_corrected"
        events.append(f"kalman_preflight:{zone}")

    def inspect(_registry, *, zones):
        events.append("country_preflight")
        return [_status(tmp_path, zone) for zone in zones]

    monkeypatch.setattr(
        "chronos2_hourly.kalman_configuration.load_kalman_operational_configuration",
        validate_kalman,
    )
    monkeypatch.setattr(launcher, "inspect_zone_statuses", inspect)
    monkeypatch.setattr(
        launcher, "build_dispatch_command",
        lambda status, **_kwargs: [str(executable), status.code],
    )
    monkeypatch.setattr(
        launcher, "launch_zone_forecast",
        lambda **_kwargs: pytest.fail("A dry run must not launch a forecast"),
    )
    batch = launcher.run_forecast_batch(
        zones=("FR", "DE", "BE", "NL", "ES"),
        delivery_day="2026-08-20", project_root=tmp_path,
        registry_path=registry, python_executable=executable,
        mode="Both", kalman_config=config, dry_run=True,
    )
    assert batch.ok
    assert events == [
        *(f"kalman_preflight:{zone}" for zone in ("FR", "DE", "BE", "NL", "ES")),
        "country_preflight",
    ]
    assert [result.state for result in batch.results] == ["dry_run"] * 5


def test_both_invalid_kalman_sidecar_refuses_before_forecast_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    export_root = tmp_path / "exports"
    calls = []

    def refuse_sidecar(*_args, **_kwargs):
        calls.append("kalman_preflight")
        raise ValueError("Invalid standard Kalman sidecar")

    monkeypatch.setattr(
        "chronos2_hourly.kalman_configuration.load_kalman_operational_configuration",
        refuse_sidecar,
    )
    for function in ("inspect_zone_statuses", "launch_zone_forecast", "_publish_exports"):
        monkeypatch.setattr(
            launcher, function,
            lambda *_args, **_kwargs: pytest.fail("Kalman preflight must fail first"),
        )
    with pytest.raises(ValueError, match="Invalid standard Kalman sidecar"):
        launcher.run_forecast_batch(
            zones=("FR", "DE", "BE", "NL", "ES"),
            delivery_day="2026-08-20", project_root=tmp_path,
            registry_path=registry, python_executable=executable,
            mode="Both", export_root=export_root,
        )
    assert calls == ["kalman_preflight"]
    assert not export_root.exists()


def _kalman_reporting_spec(
    tmp_path: Path, *, zone: str = "FR", upstream_model: str = "residual_corrected",
    variant: str = "kalman",
) -> launcher._ExportSpec:
    output = tmp_path / "exports" / "2026-08-20" / zone.lower() / variant
    return launcher._ExportSpec(
        zone=zone, timezone=_status(tmp_path, zone).timezone,
        delivery_day="2026-08-20", variant=variant,
        source_model="residual_kalman" if variant == "kalman" else upstream_model,
        baseline_model=upstream_model,
        archive=tmp_path / f"archive-{zone}",
        history_archive=tmp_path / f"archive-{zone}",
        csv_path=output / "forecast.csv", report_path=output / "forecast.html",
        history_contract=None, batch_mode="both",
        kalman_config=tmp_path / "kalman.yaml", project_root=tmp_path,
    )


@pytest.mark.parametrize("zone", ["FR", "DE", "BE", "NL", "ES"])
@pytest.mark.parametrize("upstream_model", ["residual_corrected", "exogenous_residual_corrected"])
def test_kalman_materialization_loads_its_country_and_exact_upstream_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, zone: str, upstream_model: str,
) -> None:
    spec = _kalman_reporting_spec(tmp_path, zone=zone, upstream_model=upstream_model)
    view = tmp_path / "view"
    (view / "inputs").mkdir(parents=True)
    frame = pd.DataFrame({"delivery_start_utc": ["2026-08-19T22:00:00Z"]})
    frame.to_csv(view / "statistics_history_hourly.csv.gz", index=False)
    frame.to_csv(view / f"forecast_hourly_{zone.lower()}.csv", index=False)
    frame.to_csv(view / "inputs" / "model_covariates_with_future.csv.gz", index=False)
    before = _tree_bytes(view)
    configuration = SimpleNamespace(
        filter_config=object(), covariate_config=object(),
        training_lookback_days=365, rolling_refit_workers=4,
    )
    loaded = []

    def load(path, **kwargs):
        loaded.append((path, kwargs))
        return configuration

    def attach(data, config, **_kwargs):
        assert config is configuration
        return data, {}

    class ReachedBuildWithoutTraining(Exception):
        pass

    def stop_before_training(**kwargs):
        assert kwargs["upstream_model"] == upstream_model
        assert kwargs["output_model"] == "residual_kalman"
        assert kwargs["training_lookback_days"] == 365
        assert kwargs["rolling_refit_cache_dir"] == (
            tmp_path / "runs" / "cache" / "kalman_rolling" / zone.lower() / "kalman"
        )
        raise ReachedBuildWithoutTraining

    for name, replacement in (
        ("load_kalman_operational_configuration", load),
        ("attach_additional_kalman_sources", attach),
        ("attach_kalman_upstream_history", attach),
    ):
        monkeypatch.setattr(f"chronos2_hourly.kalman_configuration.{name}", replacement)
    monkeypatch.setattr(
        "chronos2_hourly.kalman_residual.build_operational_kalman_view",
        stop_before_training,
    )
    with pytest.raises(ReachedBuildWithoutTraining):
        launcher._materialize_kalman_reporting_view(spec, view)
    assert loaded == [(spec.kalman_config, {
        "project_root": tmp_path, "zone": zone, "upstream_model": upstream_model,
    })]
    assert _tree_bytes(view) == before


@pytest.mark.parametrize("history_days", [365, 730])
@pytest.mark.parametrize("upstream_model", ["residual_corrected", "exogenous_residual_corrected"])
def test_kalman_reporting_preflight_uses_real_730_day_validator_without_fitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, history_days: int, upstream_model: str,
) -> None:
    spec = _kalman_reporting_spec(tmp_path, upstream_model=upstream_model)
    view = tmp_path / "view"
    view.mkdir()
    requested = date.fromisoformat(spec.delivery_day)
    index = pd.date_range(
        pd.Timestamp(requested - timedelta(days=history_days), tz=spec.timezone),
        pd.Timestamp(requested, tz=spec.timezone), freq="h", inclusive="left",
    ).tz_convert("UTC")
    frame = pd.DataFrame({
        "delivery_start_utc": index.astype(str), "actual": 50.0,
        f"{upstream_model}__q10": 40.0, f"{upstream_model}__q50": 50.0,
        f"{upstream_model}__q90": 60.0,
    })
    frame.to_csv(view / "statistics_history_hourly.csv.gz", index=False)
    before = _tree_bytes(view)
    loaded = []

    def load(path, **kwargs):
        loaded.append((path, kwargs))
        return SimpleNamespace(training_lookback_days=365)

    monkeypatch.setattr(
        "chronos2_hourly.kalman_configuration.load_kalman_operational_configuration", load,
    )
    monkeypatch.setattr(
        "chronos2_hourly.kalman_configuration.attach_kalman_upstream_history",
        lambda data, _config, **_kwargs: (data, {}),
    )
    monkeypatch.setattr(
        "chronos2_hourly.kalman_residual.build_operational_kalman_view",
        lambda **_kwargs: pytest.fail("History preflight must never fit Kalman"),
    )
    if history_days == 365:
        with pytest.raises(ValueError, match=r"FR/kalman: preflight historique Kalman refuse:.*730 jours"):
            launcher._preflight_kalman_reporting_view(spec, view)
    else:
        launcher._preflight_kalman_reporting_view(spec, view)
    assert loaded == [(spec.kalman_config, {
        "project_root": tmp_path, "zone": "FR", "upstream_model": upstream_model,
    })]
    assert _tree_bytes(view) == before


@pytest.mark.parametrize("reject_last_country", [False, True])
def test_publish_preflights_all_prepared_kalman_views_before_any_fit_or_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reject_last_country: bool,
) -> None:
    specs = tuple(
        _kalman_reporting_spec(tmp_path, zone=zone, variant=variant)
        for zone in ("FR", "DE") for variant in ("autonomous", "kalman")
    )
    batch = tmp_path / "exports" / "2026-08-20"
    batch.mkdir(parents=True)
    (batch / "current_batch_manifest.json").write_text('{"previous": true}', encoding="utf-8")
    before = _tree_bytes(tmp_path / "exports")
    events = []
    identities = [(spec.zone, spec.variant) for spec in specs]

    def copy(spec, destination, **_kwargs):
        destination.mkdir()
        events.append(("copy", spec.zone, spec.variant))

    def overlay(spec, destination):
        assert destination.is_dir()
        events.append(("lora", spec.zone, spec.variant))

    def preflight(spec, _destination):
        assert [(zone, variant) for phase, zone, variant in events if phase == "copy"] == identities
        assert [(zone, variant) for phase, zone, variant in events if phase == "lora"] == identities
        assert not any(phase == "fit" for phase, *_ in events)
        events.append(("preflight", spec.zone, spec.variant))
        if reject_last_country and (spec.zone, spec.variant) == ("DE", "kalman"):
            raise ValueError("DE/kalman: historique insuffisant, 730 jours requis")

    def stop_at_first_fit(spec, _destination):
        assert not reject_last_country, "An insufficient country must prevent every fit"
        assert [(zone, variant) for phase, zone, variant in events if phase == "preflight"] == identities
        events.append(("fit", spec.zone, spec.variant))
        raise RuntimeError("All preflights passed; stopped before fitting")

    monkeypatch.setattr(launcher, "_validate_export_spec", lambda _spec: None)
    monkeypatch.setattr(launcher, "_copy_reporting_view", copy)
    monkeypatch.setattr(launcher, "_materialize_lora_reporting_view", overlay)
    monkeypatch.setattr(launcher, "_preflight_kalman_reporting_view", preflight)
    monkeypatch.setattr(launcher, "_materialize_kalman_reporting_view", stop_at_first_fit)
    for name in ("_write_forecast_csv", "write_hourly_html_report", "_publish_current_batch_manifest"):
        monkeypatch.setattr(
            launcher, name,
            lambda *_args, **_kwargs: pytest.fail("No partial report or manifest may be published"),
        )
    error = ValueError if reject_last_country else RuntimeError
    match = "730 jours requis" if reject_last_country else "All preflights passed"
    with pytest.raises(error, match=match):
        launcher._publish_exports(specs)
    assert events[:8] == [
        (phase, zone, variant)
        for zone, variant in identities for phase in ("copy", "lora")
    ]
    assert events[8:12] == [("preflight", zone, variant) for zone, variant in identities]
    assert _tree_bytes(tmp_path / "exports") == before


def test_lora_live_source_manifest_is_rechecked_immediately_before_use(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "live-sources.json"
    manifest.write_text('{"revision": 2}\n', encoding="utf-8")
    resolution = launcher.ActivationResolution(
        zone="FR",
        mode="autonomous",
        lora_enabled=True,
        selected_model="exogenous_residual_corrected",
        fallback_model="residual_corrected",
        kalman_upstream_model=None,
        fallback_upstream_model=None,
        bundle=SimpleNamespace(alias="promoted-final"),
        live_source_manifest_path=manifest,
        live_source_manifest_sha256="0" * 64,
        runtime_output_root=tmp_path / "runtime",
        reason="promoted_bundle_verified",
    )

    with pytest.raises(ValueError, match="manifest de sources live LoRA a change"):
        launcher._prepare_one_lora_candidate(
            resolution,
            delivery_day="2026-08-20",
            project_root=tmp_path,
            registry_path=tmp_path / "registry.json",
            python_executable=tmp_path / "python.exe",
            device="cpu",
        )


def test_lora_runtime_first_activation_requires_pending_issued_shadow_day(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delivery_day = "2026-09-03"
    timezone_name = "Europe/Paris"
    requested = date.fromisoformat(delivery_day)
    history_index = pd.date_range(
        pd.Timestamp(requested - pd.Timedelta(days=365), tz=timezone_name),
        pd.Timestamp(requested, tz=timezone_name),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    observed_shadow_day = requested - pd.Timedelta(days=2)
    pending_issued_day = requested - pd.Timedelta(days=1)
    observed_shadow_index = pd.date_range(
        pd.Timestamp(observed_shadow_day, tz=timezone_name),
        pd.Timestamp(pending_issued_day, tz=timezone_name),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    issued_shadow_index = pd.date_range(
        pd.Timestamp(pending_issued_day, tz=timezone_name),
        pd.Timestamp(requested, tz=timezone_name),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    shadow_index = observed_shadow_index.append(issued_shadow_index)
    base_index = history_index.difference(shadow_index)
    future_index = pd.date_range(
        pd.Timestamp(requested, tz=timezone_name),
        pd.Timestamp(requested + pd.Timedelta(days=1), tz=timezone_name),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")

    def predictions(index: pd.DatetimeIndex, value: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "delivery_start_utc": index,
                "exogenous_residual_corrected__q10": value - 1.0,
                "exogenous_residual_corrected__q50": value,
                "exogenous_residual_corrected__q90": value + 1.0,
            }
        )

    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps({"timezone": timezone_name}), encoding="utf-8"
    )
    backtest_path = tmp_path / "backtest.csv.gz"
    predictions(base_index, 50.0).to_csv(backtest_path, index=False)
    forecast_path = tmp_path / "forecast.csv"
    predictions(future_index, 52.0).to_csv(forecast_path, index=False)
    bundle = SimpleNamespace(
        alias="fr-lora-v1",
        schema_path=schema_path,
        bundle_manifest_sha256="d" * 64,
    )
    resolution = launcher.ActivationResolution(
        zone="FR",
        mode="autonomous",
        lora_enabled=True,
        selected_model="exogenous_residual_corrected",
        fallback_model="residual_corrected",
        kalman_upstream_model=None,
        fallback_upstream_model=None,
        bundle=bundle,
        live_source_manifest_path=tmp_path / "live.json",
        live_source_manifest_sha256="a" * 64,
        runtime_output_root=tmp_path / "runtime",
        reason="promoted_bundle_verified",
    )
    run = SimpleNamespace(
        backtest_path=backtest_path,
        forecast_path=forecast_path,
    )
    observed_shadow = predictions(observed_shadow_index, 51.0)
    issued_shadow = predictions(issued_shadow_index, 51.5)
    include_issued = SimpleNamespace(value=False)
    monkeypatch.setattr(
        launcher,
        "load_promoted_shadow_history",
        lambda _bundle: SimpleNamespace(
            predictions=(
                pd.concat([observed_shadow, issued_shadow], ignore_index=True)
                if include_issued.value
                else observed_shadow
            ),
            predictions_sha256="b" * 64,
            manifest_sha256="c" * 64,
            issued_history_sha256="e" * 64,
        ),
    )

    with pytest.raises(ValueError, match=pending_issued_day.isoformat()):
        launcher._assemble_lora_runtime(
            resolution,
            run=run,
            delivery_day=delivery_day,
            registry_path=tmp_path / "registry.json",
        )

    include_issued.value = True
    runtime = launcher._assemble_lora_runtime(
        resolution,
        run=run,
        delivery_day=delivery_day,
        registry_path=tmp_path / "registry.json",
    )

    assert runtime.history.index.equals(history_index)
    assert runtime.history.reindex(observed_shadow_index)[
        "exogenous_residual_corrected__q50"
    ].eq(51.0).all()
    assert runtime.history.reindex(issued_shadow_index)[
        "exogenous_residual_corrected__q50"
    ].eq(51.5).all()
    assert runtime.shadow_days_reused == (
        observed_shadow_day.isoformat(),
        pending_issued_day.isoformat(),
    )
    assert runtime.issued_days_reused == ()
    assert runtime.shadow_predictions_sha256 == "b" * 64
    assert runtime.shadow_manifest_sha256 == "c" * 64
    assert runtime.shadow_issued_history_sha256 == "e" * 64


def test_lora_runtime_refuses_gap_between_holdout_and_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = date(2026, 9, 4)
    timezone_name = "Europe/Paris"
    expected = pd.date_range(
        pd.Timestamp(requested - timedelta(days=365), tz=timezone_name),
        pd.Timestamp(requested, tz=timezone_name),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    holdout_end = date(2026, 8, 11)
    shadow_start = date(2026, 9, 2)
    local_days = expected.tz_convert(timezone_name).date
    base_index = expected[local_days <= holdout_end]
    shadow_index = expected[local_days >= shadow_start]

    def frame(index: pd.DatetimeIndex, value: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "delivery_start_utc": index,
                "exogenous_residual_corrected__q10": value - 1.0,
                "exogenous_residual_corrected__q50": value,
                "exogenous_residual_corrected__q90": value + 1.0,
            }
        )

    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps({"timezone": timezone_name}), encoding="utf-8"
    )
    backtest_path = tmp_path / "backtest.csv.gz"
    frame(base_index, 50.0).to_csv(backtest_path, index=False)
    bundle = SimpleNamespace(
        alias="fr-lora-v1",
        schema_path=schema_path,
        bundle_manifest_sha256="d" * 64,
    )
    resolution = launcher.ActivationResolution(
        zone="FR",
        mode="autonomous",
        lora_enabled=True,
        selected_model="exogenous_residual_corrected",
        fallback_model="residual_corrected",
        kalman_upstream_model=None,
        fallback_upstream_model=None,
        bundle=bundle,
        live_source_manifest_path=tmp_path / "live.json",
        live_source_manifest_sha256="a" * 64,
        runtime_output_root=tmp_path / "runtime",
        reason="promoted_bundle_verified",
    )
    monkeypatch.setattr(
        launcher,
        "load_promoted_shadow_history",
        lambda _bundle: SimpleNamespace(
            predictions=frame(shadow_index, 51.0),
            predictions_sha256="b" * 64,
            manifest_sha256="c" * 64,
            issued_history_sha256="e" * 64,
        ),
    )

    with pytest.raises(ValueError, match="2026-08-12"):
        launcher._assemble_lora_runtime(
            resolution,
            run=SimpleNamespace(
                backtest_path=backtest_path,
                forecast_path=tmp_path / "unused.csv",
            ),
            delivery_day=requested.isoformat(),
            registry_path=tmp_path / "registry.json",
        )


def test_hybrid_runtime_reuses_weather_and_materializes_fuel_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = tmp_path / "materialize_saturn_kalman_fuel.py"
    materializer.write_text("# test materializer\n", encoding="utf-8")
    executable = tmp_path / "python.exe"
    executable.write_bytes(b"python")
    template = tmp_path / "kalman_hybrid_operational.yaml"
    template.write_text("version: 1\n", encoding="utf-8")
    weather_root = tmp_path / "data" / "pit" / "kalman_weather"
    weather_root.mkdir(parents=True)
    for zone in ("FR", "DE"):
        for kind in ("temperature", "wind_generation", "solar_generation"):
            weather_path = (
                weather_root / f"{zone.casefold()}_{kind}_fcst.parquet"
            )
            weather_path.write_bytes(b"weather")
            weather_path.with_name(
                weather_path.name + ".audit.json"
            ).write_text("{}", encoding="utf-8")
        prefix = (
            weather_root
            / f"{zone.casefold()}_residual_corrected_prequential.csv.gz"
        )
        prefix.write_bytes(f"prefix-{zone}".encode("ascii"))
        (
            weather_root
            / f"{zone.casefold()}_residual_corrected_prequential.audit.json"
        ).write_text("{}", encoding="utf-8")

    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        output_root = Path(command[command.index("--output-dir") + 1])
        output_root.mkdir(parents=True)
        fuel_path = output_root / "market_fuel_features.parquet"
        fuel_path.write_bytes(b"fuel")
        fuel_path.with_name(fuel_path.name + ".audit.json").write_text(
            json.dumps(
                {
                    "sha256": hashlib.sha256(fuel_path.read_bytes()).hexdigest(),
                    "information_type": (
                        "market_observation_known_before_cutoff"
                    ),
                    "causality_violations": 0,
                    "end_day": "2026-08-20",
                }
            ),
            encoding="utf-8",
        )
        residual_path = output_root / "residual_load_market_features.parquet"
        residual_path.write_bytes(b"residual")
        residual_path.with_name(
            residual_path.name + ".audit.json"
        ).write_text(
            json.dumps(
                {
                    "sha256": hashlib.sha256(
                        residual_path.read_bytes()
                    ).hexdigest(),
                    "causality_violations": 0,
                    "end_day": "2026-08-20",
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    rendered: list[tuple[str, Path, Path]] = []

    def fake_render(
        _template,
        *,
        zone,
        output_path,
        runtime_source_root,
        **_kwargs,
    ):
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("version: 1\n", encoding="utf-8")
        rendered.append((zone, path, Path(runtime_source_root)))
        return SimpleNamespace(path=path)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "chronos2_hourly.kalman_configuration."
        "render_kalman_weather_operational_configuration",
        fake_render,
    )

    configs = launcher._prepare_kalman_hybrid_runtime_configs(
        zones=("FR", "DE"),
        delivery_day="2026-08-20",
        project_root=tmp_path,
        python_executable=executable,
        template_path=template,
        threads=3,
        workers=2,
    )

    assert len(commands) == 1
    assert commands[0][0:2] == [str(executable), str(materializer)]
    assert "--zones" not in commands[0]
    assert commands[0][commands[0].index("--start-day") + 1] == "2024-06-30"
    assert (
        commands[0][commands[0].index("--residual-start-day") + 1]
        == "2024-08-20"
    )
    assert commands[0][commands[0].index("--series-workers") + 1] == "5"
    assert commands[0][commands[0].index("--day-workers") + 1] == "1"
    assert list(configs) == ["FR", "DE"]
    assert [zone for zone, _path, _root in rendered] == ["FR", "DE"]
    assert len({root for _zone, _path, root in rendered}) == 1
    source_root = rendered[0][2]
    assert (source_root / "source_bundle_manifest.json").is_file()
    assert (source_root / "market_fuel_features.parquet.audit.json").is_file()
    assert (source_root / "residual_load_market_features.parquet").is_file()
    for zone in ("fr", "de"):
        assert (
            source_root / f"{zone}_residual_corrected_prequential.csv.gz"
        ).is_file()
        assert (
            source_root
            / f"{zone}_residual_corrected_prequential.audit.json"
        ).is_file()
    assert all(path.is_file() for path in configs.values())


def test_hybrid_runtime_requires_prequential_prefix_before_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = tmp_path / "materialize_saturn_kalman_fuel.py"
    materializer.write_text("# test materializer\n", encoding="utf-8")
    executable = tmp_path / "python.exe"
    executable.write_bytes(b"python")
    template = tmp_path / "kalman_hybrid_operational.yaml"
    template.write_text("version: 1\n", encoding="utf-8")
    weather_root = tmp_path / "data" / "pit" / "kalman_weather"
    weather_root.mkdir(parents=True)
    for kind in ("temperature", "wind_generation", "solar_generation"):
        weather_path = weather_root / f"fr_{kind}_fcst.parquet"
        weather_path.write_bytes(b"weather")
        weather_path.with_name(
            weather_path.name + ".audit.json"
        ).write_text("{}", encoding="utf-8")
    # The prefix data exists but its audit is deliberately absent: a partial
    # immutable input pair must fail before the shared materializer is called.
    (
        weather_root / "fr_residual_corrected_prequential.csv.gz"
    ).write_bytes(b"prefix")

    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "Le materializer ne doit pas demarrer sans prefixe audite."
        ),
    )

    with pytest.raises(FileNotFoundError, match="prefixes prequentiels"):
        launcher._prepare_kalman_hybrid_runtime_configs(
            zones=("FR",),
            delivery_day="2026-08-20",
            project_root=tmp_path,
            python_executable=executable,
            template_path=template,
            threads=2,
            workers=2,
        )


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


def test_degraded_saturn_report_can_be_rebuilt_only_as_a_derived_export(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path, "DE")
    (archive / "reporting_errors.json").write_text("{}", encoding="utf-8")

    assert launcher._report_or_deferred_export(
        archive,
        mode="both",
        residual_load_source="saturn",
    ) is None
    with pytest.raises(launcher.DetailedReportError):
        launcher._report_or_deferred_export(
            archive,
            mode="production",
            residual_load_source="saturn",
        )
    with pytest.raises(launcher.DetailedReportError):
        launcher._report_or_deferred_export(
            archive,
            mode="both",
            residual_load_source="chronos2",
        )


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


@pytest.mark.parametrize("mode", ["Production", "Blend"])
def test_modes_without_autonomous_branch_do_not_load_lora_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    _patch_statuses(monkeypatch, tmp_path)

    def forbidden_activation(_path: Path):
        raise AssertionError("Ce mode ne consomme aucune route LoRA")

    monkeypatch.setattr(launcher, "load_activation_contract", forbidden_activation)
    monkeypatch.setattr(
        launcher,
        "build_dispatch_command",
        lambda status, **_kwargs: [str(executable), status.code],
    )

    batch = launcher.run_forecast_batch(
        zones=("FR",),
        delivery_day="2026-08-20",
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        mode=mode,
        dry_run=True,
    )

    assert batch.ok


def test_chronos2_bundle_is_built_once_and_shared_by_every_zone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    _patch_statuses(monkeypatch, tmp_path)
    bundle = tmp_path / "runs" / "experiments" / "upstream" / "manifest.json"
    bundle.parent.mkdir(parents=True)
    bundle.write_text("{}\n", encoding="utf-8")
    build_calls: list[dict] = []
    launch_calls: list[tuple[str, str, Path]] = []
    archives = {zone: _archive(tmp_path, zone) for zone in ("FR", "DE")}

    def fake_build(**kwargs):
        build_calls.append(kwargs)
        return bundle

    def fake_launch(*, zone, residual_load_source, residual_load_bundle_manifest, **_kwargs):
        launch_calls.append(
            (zone, residual_load_source, Path(residual_load_bundle_manifest))
        )
        return ForecastSkip(
            zone=zone,
            delivery_day="2026-08-20",
            archive_path=archives[zone],
        )

    monkeypatch.setattr(launcher, "_build_residual_load_bundle_once", fake_build)
    monkeypatch.setattr(launcher, "launch_zone_forecast", fake_launch)
    monkeypatch.setattr(
        launcher,
        "validate_existing_forecast_archive",
        lambda status, **_kwargs: archives[status.code],
    )

    batch = launcher.run_forecast_batch(
        zones=("FR", "DE"),
        delivery_day="2026-08-20",
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        residual_load_source="chronos2",
    )

    assert batch.ok
    assert len(build_calls) == 1
    assert launch_calls == [
        ("FR", "chronos2", bundle),
        ("DE", "chronos2", bundle),
    ]
    assert batch.residual_load_source == "chronos2"
    assert batch.residual_load_bundle_manifest == bundle


def test_blend_rejects_unsupported_country_before_any_preflight_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = SimpleNamespace(inspect=0)

    def forbidden_inspect(*_args, **_kwargs):
        calls.inspect += 1
        raise AssertionError("La selection Blend invalide doit echouer avant l'audit")

    monkeypatch.setattr(launcher, "inspect_zone_statuses", forbidden_inspect)
    export_root = tmp_path / "exports"

    with pytest.raises(ValueError, match="FR et NL"):
        launcher.run_forecast_batch(
            zones=("FR", "DE"),
            delivery_day="2026-08-20",
            project_root=tmp_path,
            registry_path=tmp_path / "missing.yaml",
            python_executable=tmp_path / "missing.exe",
            mode="Blend",
            export_root=export_root,
        )

    assert calls.inspect == 0
    assert not export_root.exists()


def test_batch_preflight_checks_every_country_before_starting_a_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    _patch_statuses(monkeypatch, tmp_path, ready={"DE": False})
    launches: list[str] = []
    monkeypatch.setattr(
        launcher,
        "launch_zone_forecast",
        lambda *, zone, **_kwargs: launches.append(zone),
    )

    batch = launcher.run_forecast_batch(
        zones=("FR", "DE"),
        delivery_day="2026-08-20",
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
    )

    assert launches == []
    assert not batch.ok
    assert [result.state for result in batch.results] == ["failed", "failed"]
    assert "Batch annule" in batch.results[0].message
    assert "bundle incomplet" in batch.results[1].message


def test_existing_archive_without_blend_columns_fails_before_launcher_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    archive = _exportable_archive(tmp_path, "NL")
    forecast_path = archive / "forecast_hourly_nl.csv"
    forecast = pd.read_csv(forecast_path)
    forecast = forecast.drop(
        columns=[column for column in forecast if column.startswith("mkonline_blend__")]
    )
    forecast.to_csv(forecast_path, index=False)
    _patch_statuses(monkeypatch, tmp_path)
    launches: list[str] = []
    monkeypatch.setattr(
        launcher,
        "validate_existing_forecast_archive",
        lambda *_args, **_kwargs: archive,
    )
    monkeypatch.setattr(
        launcher,
        "launch_zone_forecast",
        lambda *, zone, **_kwargs: launches.append(zone),
    )

    batch = launcher.run_forecast_batch(
        zones=("NL",),
        delivery_day="2026-08-20",
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        mode="Blend",
        export_root=tmp_path / "exports",
    )

    assert launches == []
    assert not batch.ok
    assert "Preflight export refuse" in batch.results[0].message
    assert "mkonline_blend__q50" in batch.results[0].message
    assert not (tmp_path / "exports").exists()


def test_autonomous_export_uses_residual_columns_without_mutating_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    archive = _exportable_archive(tmp_path, "FR")
    archive_before = _tree_bytes(archive)
    _patch_statuses(monkeypatch, tmp_path)
    monkeypatch.setattr(
        launcher,
        "launch_zone_forecast",
        lambda **_kwargs: ForecastSkip(
            zone="FR", delivery_day="2026-08-20", archive_path=archive
        ),
    )
    monkeypatch.setattr(
        launcher,
        "validate_existing_forecast_archive",
        lambda *_args, **_kwargs: archive,
    )

    def fake_report(run_dir, *, output_path, native_model, baseline_model, **_kwargs):
        assert Path(run_dir).resolve() != archive.resolve()
        assert native_model == "residual_corrected"
        assert baseline_model == "chronos2"
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("<!doctype html><title>Autonomous</title>", encoding="utf-8")
        return output

    monkeypatch.setattr(launcher, "write_hourly_html_report", fake_report)
    monkeypatch.setattr(
        launcher,
        "_validate_statistics_price_report",
        lambda *_args, **_kwargs: None,
    )
    export_root = tmp_path / "exports"

    batch = launcher.run_forecast_batch(
        zones=("FR",),
        delivery_day="2026-08-20",
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        mode="Autonomous",
        export_root=export_root,
    )

    assert batch.ok
    assert batch.mode == "autonomous"
    assert len(batch.results[0].exports) == 1
    forecast_export = batch.results[0].exports[0]
    assert forecast_export.variant == "autonomous"
    exported = pd.read_csv(forecast_export.csv_path)
    assert "residual_load_source" not in exported.columns
    assert exported["uses_mkonline"].eq(False).all()
    assert exported["q50"].tolist() == [42.0, 43.0]
    assert forecast_export.report_path.is_file()
    assert _tree_bytes(archive) == archive_before


def test_chronos2_export_uses_canonical_report_with_paired_saturn_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    challenger_root = tmp_path / "challenger"
    control_root = tmp_path / "control"
    challenger_root.mkdir()
    control_root.mkdir()
    archive = _exportable_archive(challenger_root, "FR")
    control = _exportable_archive(control_root, "FR")
    (archive / "backtest_hourly_oof.csv.gz").unlink()
    control_backtest = pd.read_csv(control / "backtest_hourly_oof.csv.gz")
    control_backtest.to_csv(
        control / "statistics_history_hourly.csv.gz",
        index=False,
        compression="gzip",
    )
    (control / "statistics_history_audit.json").write_text(
        json.dumps(
            {
                "source": "paired_saturn_control",
                "report_scope_note": "Historique de reference.",
            }
        ),
        encoding="utf-8",
    )
    archive_before = _tree_bytes(archive)
    control_before = _tree_bytes(control)
    bundle = tmp_path / "chronos2_residual_load_manifest.json"
    bundle.write_text("{}\n", encoding="utf-8")
    _patch_statuses(monkeypatch, tmp_path)
    monkeypatch.setattr(
        launcher,
        "_build_residual_load_bundle_once",
        lambda **_kwargs: bundle,
    )
    monkeypatch.setattr(
        launcher,
        "launch_zone_forecast",
        lambda **_kwargs: ForecastSkip(
            zone="FR", delivery_day="2026-08-20", archive_path=archive
        ),
    )
    monkeypatch.setattr(
        launcher,
        "validate_existing_forecast_archive",
        lambda *_args, **_kwargs: archive,
    )
    monkeypatch.setattr(
        launcher,
        "_paired_saturn_history_archive",
        lambda **_kwargs: control,
    )

    def fake_report(run_dir, *, output_path, title, **_kwargs):
        view = Path(run_dir)
        assert pd.read_csv(view / "backtest_hourly_oof.csv.gz").equals(
            control_backtest
        )
        assert pd.read_csv(view / "forecast_hourly_fr.csv").equals(
            pd.read_csv(archive / "forecast_hourly_fr.csv")
        )
        audit = json.loads(
            (view / "statistics_history_audit.json").read_text(
                encoding="utf-8"
            )
        )
        context = audit["residual_load_challenger_reporting_context"]
        assert context["historical_metrics_source"] == (
            "paired_sealed_saturn_control"
        )
        assert context["historical_metrics_are_challenger_performance"] is False
        assert "ne constituent pas une performance historique" in audit[
            "report_scope_note"
        ]
        assert "challenger residual_load Chronos-2" in title
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            '<!doctype html><html lang="fr" data-theme="light">'
            '<button id="theme-toggle"></button>'
            '<section id="statistics"></section></html>',
            encoding="utf-8",
        )
        return output

    monkeypatch.setattr(launcher, "write_hourly_html_report", fake_report)
    monkeypatch.setattr(
        launcher,
        "_validate_statistics_price_report",
        lambda *_args, **_kwargs: None,
    )

    batch = launcher.run_forecast_batch(
        zones=("FR",),
        delivery_day="2026-08-20",
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        mode="Autonomous",
        export_root=tmp_path / "exports",
        residual_load_source="chronos2",
    )

    assert batch.ok
    forecast_export = batch.results[0].exports[0]
    exported = pd.read_csv(forecast_export.csv_path)
    assert exported["residual_load_source"].eq("chronos2").all()
    assert "actual" not in exported.columns
    assert exported["q50"].tolist() == [42.0, 43.0]
    report = forecast_export.report_path.read_text(encoding="utf-8")
    assert 'data-theme="light"' in report
    assert 'id="theme-toggle"' in report
    assert 'id="statistics"' in report
    assert "FORECAST UNIQUEMENT" not in report
    assert _tree_bytes(archive) == archive_before
    assert _tree_bytes(control) == control_before


def test_both_refreshes_and_exports_autonomous_and_standard_kalman_for_every_country(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    zones = ("FR", "DE", "BE", "NL", "ES")
    archives = {
        zone: _exportable_archive(tmp_path, zone) for zone in zones
    }
    (archives["FR"] / "live_run_summary.json").write_text(
        json.dumps({"status": "forecast_complete_statistics_partial"}),
        encoding="utf-8",
    )
    # Both must no longer require or consume MKOnline forecast columns.
    for zone, archive in archives.items():
        for path in (archive / f"forecast_hourly_{zone.lower()}.csv", archive / "backtest_hourly_oof.csv.gz"):
            frame = pd.read_csv(path)
            frame.drop(
                columns=[column for column in frame if column.startswith("mkonline_blend__")]
            ).to_csv(path, index=False)
    archive_before = {zone: _tree_bytes(path) for zone, path in archives.items()}
    _patch_statuses(monkeypatch, tmp_path)
    monkeypatch.setattr(
        launcher,
        "launch_zone_forecast",
        lambda *, zone, **_kwargs: ForecastSkip(
            zone=zone, delivery_day="2026-08-20", archive_path=archives[zone]
        ),
    )
    monkeypatch.setattr(
        launcher,
        "validate_existing_forecast_archive",
        lambda status, **_kwargs: archives[status.code],
    )
    events = []
    monkeypatch.setattr(
        launcher, "_complete_statistics_archives",
        lambda **_kwargs: events.append("statistics_catchup"),
    )
    monkeypatch.setattr(
        launcher, "_prepare_lora_candidates",
        lambda *_args, **_kwargs: pytest.fail("Inactive LoRA must not prepare a candidate"),
    )
    real_refresh = launcher._refresh_statistics_view
    refreshed = []

    def track_refresh(spec, destination, **kwargs):
        refreshed.append((spec.zone, spec.variant))
        return real_refresh(spec, destination, **kwargs)

    monkeypatch.setattr(launcher, "_refresh_statistics_view", track_refresh)

    def standard_kalman(spec, destination):
        if spec.variant != "kalman":
            return
        assert spec.lora_candidate is None
        assert spec.lora_resolution is None
        assert spec.source_model == "residual_kalman"
        assert spec.baseline_model == "residual_corrected"
        assert spec.kalman_config == launcher.DEFAULT_KALMAN_CONFIG.resolve()
        events.append(f"kalman:{spec.zone}")
        view = Path(destination)
        assert view.resolve() != spec.archive.resolve()
        forecast_path = launcher._forecast_path(view, spec.zone)
        forecast = pd.read_csv(forecast_path)
        for quantile in ("q10", "q50", "q90"):
            forecast[f"residual_kalman__{quantile}"] = (
                forecast[f"residual_corrected__{quantile}"] + 1.0
            )
        forecast.to_csv(forecast_path, index=False)
        for filename in ("kalman_filter_audit.json", "kalman_operational_sidecar_audit.json"):
            (view / filename).write_text(
                json.dumps({"status": "complete", "config_sha256": "a" * 64}),
                encoding="utf-8",
            )

    monkeypatch.setattr(launcher, "_materialize_kalman_reporting_view", standard_kalman)
    monkeypatch.setattr(
        launcher, "_preflight_kalman_reporting_view", lambda *_args: None,
    )

    def fake_report(_run_dir, *, output_path, **_kwargs):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("<!doctype html><title>Detailed</title>", encoding="utf-8")
        return output

    monkeypatch.setattr(launcher, "write_hourly_html_report", fake_report)
    monkeypatch.setattr(
        launcher,
        "_validate_statistics_price_report",
        lambda *_args, **_kwargs: None,
    )

    batch = launcher.run_forecast_batch(
        zones=zones,
        delivery_day="2026-08-20",
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        mode="Both",
        export_root=tmp_path / "exports",
    )

    assert batch.ok, [result.message for result in batch.results]
    assert events == ["statistics_catchup", *(f"kalman:{zone}" for zone in zones)]
    assert refreshed == [(zone, variant) for zone in zones for variant in ("autonomous", "kalman")]
    for result in batch.results:
        assert [item.variant for item in result.exports] == ["autonomous", "kalman"]
        for item in result.exports:
            exported = pd.read_csv(item.csv_path)
            assert exported["uses_mkonline"].eq(False).all()
            expected = [42.0, 43.0] if item.variant == "autonomous" else [43.0, 44.0]
            assert exported["q50"].tolist() == expected
            assert item.report_path.is_file()
            assert item.statistics_end_local == "2026-08-19"
    manifest = json.loads(
        (tmp_path / "exports" / "2026-08-20" / "current_batch_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["mode"] == "both"
    assert len(manifest["exports"]) == 10
    assert {(entry["zone"], entry["variant"]) for entry in manifest["exports"]} == set(refreshed)
    assert all(_tree_bytes(archives[zone]) == archive_before[zone] for zone in zones)
    assert "statistics_partial" not in batch.results[0].message
    assert "completes jusqu'au 2026-08-19" in batch.results[0].message


def test_all_publishes_kalman_from_the_disposable_view_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    archives = {
        zone: _exportable_archive(tmp_path, zone) for zone in ("FR", "DE")
    }
    archive_bytes = {zone: _tree_bytes(path) for zone, path in archives.items()}
    _patch_statuses(monkeypatch, tmp_path)
    prepared_lora = _patch_promoted_lora_all(
        monkeypatch, tmp_path, zones=("FR", "DE")
    )
    monkeypatch.setattr(
        launcher,
        "launch_zone_forecast",
        lambda *, zone, **_kwargs: ForecastSkip(
            zone=zone, delivery_day="2026-08-20", archive_path=archives[zone]
        ),
    )
    monkeypatch.setattr(
        launcher,
        "validate_existing_forecast_archive",
        lambda status, **_kwargs: archives[status.code],
    )
    events: list[str] = []
    monkeypatch.setattr(
        launcher,
        "_complete_statistics_archives",
        lambda **_kwargs: events.append("backfill"),
    )

    def fake_materialize(spec, destination):
        if spec.variant not in launcher.KALMAN_VARIANTS:
            return
        events.append(f"{spec.variant}:{spec.zone}")
        view = Path(destination)
        assert view.resolve() != spec.archive.resolve()
        assert not (view / "variable_attribution_hourly.csv.gz").exists()
        forecast_path = launcher._forecast_path(view, spec.zone)
        forecast = pd.read_csv(forecast_path)
        for quantile in ("q10", "q50", "q90"):
            forecast[f"{spec.source_model}__{quantile}"] = (
                forecast[f"{spec.baseline_model}__{quantile}"] + 1.0
            )
        forecast.to_csv(forecast_path, index=False)
        (view / "kalman_filter_audit.json").write_text(
            json.dumps({"status": "complete"}), encoding="utf-8"
        )
        (view / "kalman_operational_sidecar_audit.json").write_text(
            json.dumps({"status": "complete", "config_sha256": "a" * 64}),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        launcher,
        "_materialize_kalman_reporting_view",
        fake_materialize,
    )
    report_calls: list[tuple[str, str, str]] = []

    def fake_report(
        _run_dir,
        *,
        output_path,
        title,
        native_model,
        baseline_model,
        **_kwargs,
    ):
        report_calls.append((native_model, baseline_model, title))
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("<!doctype html><title>Detailed</title>", encoding="utf-8")
        return output

    monkeypatch.setattr(launcher, "write_hourly_html_report", fake_report)
    monkeypatch.setattr(
        launcher,
        "_validate_statistics_price_report",
        lambda *_args, **_kwargs: None,
    )
    export_root = tmp_path / "exports"
    stale = export_root / "2026-08-20" / "fr" / "kalman_weather"
    stale.mkdir(parents=True)
    (stale / "old_report.html").write_text("stale", encoding="utf-8")
    manifest_publications: list[dict[str, object]] = []
    real_publish_manifest = launcher._publish_current_batch_manifest

    def checked_publish_manifest(payload, *, batch_directory):
        # The pointer may only change once every file it describes is present
        # with the exact content hashed while staging.
        for record in payload["exports"]:
            for key in ("csv", "html"):
                file_record = record[key]
                path = Path(batch_directory) / file_record["path"]
                assert path.is_file()
                assert hashlib.sha256(path.read_bytes()).hexdigest() == (
                    file_record["sha256"]
                )
            for file_record in record["audits"]:
                path = Path(batch_directory) / file_record["path"]
                assert path.is_file()
                assert hashlib.sha256(path.read_bytes()).hexdigest() == (
                    file_record["sha256"]
                )
        manifest_publications.append(payload)
        return real_publish_manifest(payload, batch_directory=batch_directory)

    monkeypatch.setattr(
        launcher,
        "_publish_current_batch_manifest",
        checked_publish_manifest,
    )

    batch = launcher.run_forecast_batch(
        zones=("FR", "DE"),
        delivery_day="2026-08-20",
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        mode="All",
        export_root=export_root,
    )

    assert batch.ok
    assert events == [
        "backfill",
        "kalman:FR",
        "kalman:DE",
    ]
    assert [item.variant for item in batch.results[0].exports] == [
        "autonomous",
        "kalman",
    ]
    assert [item.variant for item in batch.results[1].exports] == [
        "autonomous",
        "kalman",
    ]
    for result in batch.results:
        kalman_export = next(
            item for item in result.exports if item.variant == "kalman"
        )
        exported = pd.read_csv(kalman_export.csv_path)
        assert exported["source_model"].eq("residual_kalman").all()
        assert exported["uses_mkonline"].eq(False).all()
        assert exported["q50"].tolist() == [43.5, 44.5]
        assert kalman_export.kalman_audit_path is not None
        assert kalman_export.kalman_audit_path.is_file()
        assert (
            kalman_export.report_path.parent / "kalman_filter_audit.json"
        ).is_file()
        assert (
            kalman_export.csv_path
            == (
                tmp_path
                / "exports"
                / "2026-08-20"
                / result.zone.lower()
                / "kalman"
                / f"forecast_{result.zone.lower()}_2026-08-20_kalman.csv"
            ).resolve()
        )
    assert any(
        native == "residual_kalman"
        and baseline == "exogenous_residual_corrected"
        and "Kalman gouverne" in title
        for native, baseline, title in report_calls
    )
    assert prepared_lora == ["FR", "DE"]
    assert all(_tree_bytes(archives[zone]) == archive_bytes[zone] for zone in archives)
    assert len(manifest_publications) == 1
    manifest_path = export_root / "2026-08-20" / "current_batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest == manifest_publications[0]
    assert manifest["schema_version"] == 1
    assert manifest["delivery_day"] == "2026-08-20"
    assert manifest["mode"] == "all"
    assert manifest["zones"] == ["FR", "DE"]
    assert {
        (entry["zone"], entry["variant"], entry["source_model"])
        for entry in manifest["exports"]
    } == {
        ("FR", "autonomous", "exogenous_residual_corrected"),
        ("FR", "kalman", "residual_kalman"),
        ("DE", "autonomous", "exogenous_residual_corrected"),
        ("DE", "kalman", "residual_kalman"),
    }
    assert all(
        entry["statistics_end_local"] == "2026-08-19"
        for entry in manifest["exports"]
    )
    assert all(
        entry["statistics_start_local"] is None
        for entry in manifest["exports"]
    )
    assert not any(
        "kalman_weather" in entry["csv"]["path"]
        or "kalman_hybrid" in entry["csv"]["path"]
        or "/blend/" in entry["csv"]["path"]
        for entry in manifest["exports"]
    )
    for entry in manifest["exports"]:
        for key in ("csv", "html"):
            record = entry[key]
            assert hashlib.sha256(
                (manifest_path.parent / record["path"]).read_bytes()
            ).hexdigest() == record["sha256"]
        for record in entry["audits"]:
            assert hashlib.sha256(
                (manifest_path.parent / record["path"]).read_bytes()
            ).hexdigest() == record["sha256"]


def test_all_kalman_failure_publishes_no_partial_derived_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    archive = _exportable_archive(tmp_path, "FR")
    archive_before = _tree_bytes(archive)
    _patch_statuses(monkeypatch, tmp_path)
    _patch_promoted_lora_all(monkeypatch, tmp_path, zones=("FR",))
    monkeypatch.setattr(
        launcher,
        "launch_zone_forecast",
        lambda **_kwargs: ForecastSkip(
            zone="FR", delivery_day="2026-08-20", archive_path=archive
        ),
    )
    monkeypatch.setattr(
        launcher,
        "validate_existing_forecast_archive",
        lambda *_args, **_kwargs: archive,
    )
    monkeypatch.setattr(
        launcher,
        "_complete_statistics_archives",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        launcher,
        "write_hourly_html_report",
        lambda _run_dir, *, output_path, **_kwargs: Path(output_path).write_text(
            "<!doctype html>", encoding="utf-8"
        ),
    )
    monkeypatch.setattr(
        launcher,
        "_validate_statistics_price_report",
        lambda *_args, **_kwargs: None,
    )

    def fail_on_kalman(spec, _destination):
        if spec.variant == "kalman":
            raise RuntimeError("Kalman indisponible")

    monkeypatch.setattr(
        launcher,
        "_materialize_kalman_reporting_view",
        fail_on_kalman,
    )
    export_root = tmp_path / "exports"
    manifest_path = export_root / "2026-08-20" / "current_batch_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    previous_manifest = b'{"batch": "previous-success"}\n'
    manifest_path.write_bytes(previous_manifest)

    batch = launcher.run_forecast_batch(
        zones=("FR",),
        delivery_day="2026-08-20",
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        mode="All",
        export_root=export_root,
    )

    assert not batch.ok
    assert "Kalman indisponible" in batch.results[0].message
    assert not list(export_root.rglob("*.csv"))
    assert not list(export_root.rglob("*.html"))
    assert manifest_path.read_bytes() == previous_manifest
    assert _tree_bytes(archive) == archive_before


def test_all_statistics_bound_mismatch_publishes_no_partial_derived_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    archive = _exportable_archive(tmp_path, "FR")
    archive_before = _tree_bytes(archive)
    _patch_statuses(monkeypatch, tmp_path)
    _patch_promoted_lora_all(monkeypatch, tmp_path, zones=("FR",))
    monkeypatch.setattr(
        launcher,
        "launch_zone_forecast",
        lambda **_kwargs: ForecastSkip(
            zone="FR", delivery_day="2026-08-20", archive_path=archive
        ),
    )
    monkeypatch.setattr(
        launcher,
        "validate_existing_forecast_archive",
        lambda *_args, **_kwargs: archive,
    )
    monkeypatch.setattr(
        launcher,
        "_complete_statistics_archives",
        lambda **_kwargs: None,
    )

    def materialize_mismatched_kalman(spec, destination):
        if spec.variant not in launcher.KALMAN_VARIANTS:
            return
        view = Path(destination)
        forecast_path = launcher._forecast_path(view, spec.zone)
        forecast = pd.read_csv(forecast_path)
        for quantile in ("q10", "q50", "q90"):
            forecast[f"{spec.source_model}__{quantile}"] = forecast[
                f"{spec.baseline_model}__{quantile}"
            ]
        forecast.to_csv(forecast_path, index=False)
        (view / "statistics_history_audit.json").write_text(
            json.dumps(
                {
                    "statistics_through_day_local": (
                        "2026-08-18"
                        if spec.variant == "kalman"
                        else "2026-08-19"
                    )
                }
            ),
            encoding="utf-8",
        )
        (view / "kalman_filter_audit.json").write_text(
            json.dumps({"status": "complete"}), encoding="utf-8"
        )
        (view / "kalman_operational_sidecar_audit.json").write_text(
            json.dumps({"status": "complete", "config_sha256": "a" * 64}),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        launcher,
        "_materialize_kalman_reporting_view",
        materialize_mismatched_kalman,
    )
    monkeypatch.setattr(
        launcher,
        "write_hourly_html_report",
        lambda _run_dir, *, output_path, **_kwargs: Path(output_path).write_text(
            "<!doctype html>", encoding="utf-8"
        ),
    )
    monkeypatch.setattr(
        launcher,
        "_validate_statistics_price_report",
        lambda *_args, **_kwargs: None,
    )
    export_root = tmp_path / "exports"

    batch = launcher.run_forecast_batch(
        zones=("FR",),
        delivery_day="2026-08-20",
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        mode="All",
        export_root=export_root,
    )

    assert not batch.ok
    assert "meme borne Statistics" in batch.results[0].message
    assert not list(export_root.rglob("*.csv"))
    assert not list(export_root.rglob("*.html"))
    assert _tree_bytes(archive) == archive_before


@pytest.mark.parametrize("mode", ("All", "Both"))
def test_kalman_modes_reject_chronos2_residual_load_history_before_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    for function in ("inspect_zone_statuses", "launch_zone_forecast", "_build_residual_load_bundle_once"):
        monkeypatch.setattr(
            launcher, function,
            lambda *_args, **_kwargs: pytest.fail("The residual-load contract must fail first"),
        )
    export_root = tmp_path / "exports"
    with pytest.raises(ValueError, match="exige ResidualLoadSource Saturn"):
        launcher.run_forecast_batch(
            zones=("FR", "DE", "BE", "NL", "ES"),
            delivery_day="2026-08-20",
            project_root=tmp_path,
            mode=mode,
            residual_load_source="chronos2",
            export_root=export_root,
        )
    assert not export_root.exists()


def test_both_completes_statistics_before_publishing_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    archive = _exportable_archive(tmp_path, "FR")
    _patch_statuses(monkeypatch, tmp_path)
    monkeypatch.setattr(
        launcher,
        "launch_zone_forecast",
        lambda **_kwargs: ForecastSkip(
            zone="FR", delivery_day="2026-08-20", archive_path=archive
        ),
    )
    monkeypatch.setattr(
        launcher,
        "validate_existing_forecast_archive",
        lambda *_args, **_kwargs: archive,
    )
    events: list[str] = []
    monkeypatch.setattr(
        launcher,
        "_complete_statistics_archives",
        lambda **_kwargs: events.append("backfill"),
    )

    def fake_publish(specs):
        assert [(spec.zone, spec.variant) for spec in specs] == [
            ("FR", "autonomous"), ("FR", "kalman"),
        ]
        assert [spec.source_model for spec in specs] == ["residual_corrected", "residual_kalman"]
        assert specs[1].baseline_model == "residual_corrected"
        events.append("publish")
        return {}

    monkeypatch.setattr(launcher, "_publish_exports", fake_publish)

    batch = launcher.run_forecast_batch(
        zones=("FR",),
        delivery_day="2026-08-20",
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        mode="Both",
        export_root=tmp_path / "exports",
    )

    assert batch.ok
    assert events == ["backfill", "publish"]


def test_both_fails_closed_when_statistics_catchup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    archive = _exportable_archive(tmp_path, "FR")
    _patch_statuses(monkeypatch, tmp_path)
    monkeypatch.setattr(
        launcher,
        "launch_zone_forecast",
        lambda **_kwargs: ForecastSkip(
            zone="FR", delivery_day="2026-08-20", archive_path=archive
        ),
    )
    monkeypatch.setattr(
        launcher,
        "validate_existing_forecast_archive",
        lambda *_args, **_kwargs: archive,
    )

    def fail_catchup(**_kwargs):
        raise RuntimeError("rattrapage incomplet")

    monkeypatch.setattr(launcher, "_complete_statistics_archives", fail_catchup)
    monkeypatch.setattr(
        launcher,
        "_publish_exports",
        lambda _specs: pytest.fail("aucun rapport ne doit etre publie"),
    )

    batch = launcher.run_forecast_batch(
        zones=("FR",),
        delivery_day="2026-08-20",
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        mode="Both",
        export_root=tmp_path / "exports",
    )

    assert not batch.ok
    assert batch.results[0].state == "failed"
    assert "rattrapage incomplet" in batch.results[0].message


def test_refreshed_exports_attach_storm_once_and_keep_unsupported_zone_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive"
    inputs = archive / "inputs"
    inputs.mkdir(parents=True)
    delivery = pd.date_range("2026-08-18T22:00:00Z", periods=24, freq="h")
    pd.DataFrame(
        {
            "timestamp": delivery.astype(str),
            "target": range(24),
        }
    ).to_csv(
        inputs / "aligned_inputs.csv.gz", index=False, compression="gzip"
    )
    forecast_delivery = pd.date_range(
        "2026-08-19T22:00:00Z", periods=24, freq="h"
    )
    pd.DataFrame(
        {
            "delivery_start_utc": forecast_delivery.astype(str),
            "q50": range(24),
        }
    ).to_csv(archive / "forecast_hourly_fr.csv", index=False)
    pd.DataFrame(
        {
            "delivery_start_utc": forecast_delivery.astype(str),
            "q50": range(24),
        }
    ).to_csv(archive / "forecast_hourly_es.csv", index=False)

    def contract(*, dashboard: bool) -> launcher._ExportHistoryContract:
        return launcher._ExportHistoryContract(
            sealed_benchmark_run=tmp_path / "benchmark",
            live_output_root=tmp_path / "live",
            forecast_name="forecast_hourly_fr.csv",
            candidate_model="mkonline_blend",
            target_series="target",
            prediction_mode="mkonline_blend",
            storm_pit_path=None,
            storm_dashboard_enabled=dashboard,
            saturn_url="https://saturn.test/api",
            saturn_author="tester",
        )

    def spec(
        variant: str,
        *,
        dashboard: bool,
    ) -> launcher._ExportSpec:
        return launcher._ExportSpec(
            zone="FR" if dashboard else "ES",
            timezone="Europe/Paris" if dashboard else "Europe/Madrid",
            delivery_day="2026-08-20",
            variant=variant,
            source_model="mkonline_blend",
            baseline_model="residual_corrected",
            archive=archive,
            history_archive=archive,
            csv_path=tmp_path / f"{variant}.csv",
            report_path=tmp_path / f"{variant}.html",
            history_contract=contract(dashboard=dashboard),
        )

    update_calls: list[bool] = []

    def fake_update(**kwargs):
        has_official = kwargs.get("storm_dashboard_native") is not None
        update_calls.append(has_official)
        destination = Path(kwargs["staging_run_dir"])
        actual = pd.Series(range(24), dtype=float)
        history = pd.DataFrame(
            {
                "delivery_start_utc": delivery.astype(str),
                "actual": actual,
                "storm_evaluation_only__q50": actual + 3.0,
            }
        )
        audit = {
            "status": "complete",
            "statistics_complete": True,
            "statistics_prefix_end_local": str(
                kwargs.get("statistics_through_day")
                or pd.Timestamp(kwargs["current_delivery_day"])
                - pd.Timedelta(days=1)
            )[:10],
        }
        source = kwargs.get("canonical_target_source")
        if source is not None:
            audit["canonical_actuals"] = {"source": dict(source)}
        if has_official:
            history["storm_dashboard_official__q50"] = actual + 2.0
            audit.update(
                {
                    "storm_primary_report_benchmark": (
                        "storm_dashboard_official__q50"
                    ),
                    "storm_dashboard": {
                        "expected_hours": 24,
                        "available_hours": 24,
                        "missing_hours": 0,
                    },
                }
            )
        history.to_csv(
            destination / "statistics_history_hourly.csv.gz",
            index=False,
            compression="gzip",
        )
        (destination / "statistics_history_audit.json").write_text(
            json.dumps(audit), encoding="utf-8"
        )
        return audit

    fetches = SimpleNamespace(count=0)
    actual_fetches = SimpleNamespace(count=0)

    def fake_actual_fetch(
        _client,
        _series,
        _start,
        _end,
        _timezone,
        **kwargs,
    ):
        actual_fetches.count += 1
        assert kwargs["naive_timezone"] == "UTC"
        assert kwargs["nocache"] is True
        assert kwargs["live"] is True
        combined = delivery.union(forecast_delivery)
        return pd.Series(range(len(combined)), index=combined, dtype=float)

    def fake_fetch(_client, *, zone, expected_index):
        fetches.count += 1
        assert zone == "FR"
        local_naive = expected_index.tz_convert("Europe/Paris").tz_localize(None)
        return pd.Series(range(len(expected_index)), index=local_naive, dtype=float), {
            "series": "power.price.fr.euromwh.h.fcst.3mv.storm.da.cache"
        }

    monkeypatch.setattr(launcher, "update_live_statistics_history", fake_update)
    monkeypatch.setattr(launcher, "create_saturn_client", lambda *_args: object())
    monkeypatch.setattr(
        launcher,
        "fetch_saturn_series_from_client",
        fake_actual_fetch,
    )
    monkeypatch.setattr(launcher, "fetch_native_dashboard_snapshot", fake_fetch)
    cache = {}
    actual_cache = {}
    for variant in ("autonomous", "blend"):
        destination = tmp_path / f"view_{variant}"
        (destination / "inputs").mkdir(parents=True)
        launcher._refresh_statistics_view(
            spec(variant, dashboard=True),
            destination,
            actual_snapshot_cache=actual_cache,
            storm_snapshot_cache=cache,
        )
        history = pd.read_csv(
            destination / "statistics_history_hourly.csv.gz"
        )
        assert "storm_dashboard_official__q50" in history
        live_storm = pd.read_parquet(
            destination
            / launcher.STORM_DASHBOARD_LIVE_FORECAST_ARTIFACT
        )
        assert len(live_storm) == 24
        assert live_storm[
            launcher.STORM_DASHBOARD_COLUMN
        ].notna().all()

    assert fetches.count == 1
    assert actual_fetches.count == 1
    assert update_calls == [False, True, False, True]

    unsupported = tmp_path / "view_es"
    (unsupported / "inputs").mkdir(parents=True)
    launcher._refresh_statistics_view(
        spec("autonomous_es", dashboard=False),
        unsupported,
    )
    unsupported_history = pd.read_csv(
        unsupported / "statistics_history_hourly.csv.gz"
    )
    assert not any("storm" in column for column in unsupported_history)
    unsupported_audit = json.loads(
        (unsupported / "statistics_history_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert unsupported_audit["storm_refresh_status"] == "unsupported_for_zone"


def test_latest_statistics_day_requires_the_complete_forecast_curve() -> None:
    spec = SimpleNamespace(delivery_day="2026-08-20")
    delivery = pd.date_range("2026-08-19T22:00:00Z", periods=24, freq="h")
    complete = pd.Series(np.arange(24, dtype=float), index=delivery)
    incomplete = complete.iloc[:-1]

    assert launcher._latest_statistics_day(
        spec,
        observed=complete,
        forecast_delivery=delivery,
    ) == date(2026, 8, 20)
    assert launcher._latest_statistics_day(
        spec,
        observed=incomplete,
        forecast_delivery=delivery,
    ) == date(2026, 8, 19)


def test_latest_observed_snapshot_fills_only_missing_suffix_from_validated_auction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_delivery = pd.date_range(
        "2026-08-19T22:00:00Z", periods=24, freq="h"
    )
    expected = pd.date_range(
        current_delivery[0] - pd.Timedelta(days=8),
        current_delivery[-1],
        freq="h",
    )
    canonical = pd.Series(np.arange(len(expected), dtype=float), index=expected)
    # The production Saturn response omits unpublished hours rather than
    # necessarily returning explicit NaNs.
    previous_delivery = current_delivery - pd.Timedelta(days=1)
    canonical = canonical.drop(index=current_delivery.union(previous_delivery))
    auction = pd.Series(np.arange(len(expected), dtype=float), index=expected)
    spec = SimpleNamespace(
        zone="DE",
        timezone="Europe/Berlin",
        delivery_day="2026-08-20",
        history_contract=SimpleNamespace(target_series="canonical_target"),
    )

    calls: list[str] = []

    def fake_fetch(_client, series, *_args, **_kwargs):
        calls.append(series)
        return canonical if series == "canonical_target" else auction

    monkeypatch.setattr(launcher, "fetch_saturn_series_from_client", fake_fetch)
    observed, source = launcher._fetch_latest_observed_snapshot(
        object(), spec=spec, expected_index=expected
    )

    assert calls == [
        "canonical_target",
        launcher.POST_AUCTION_OBSERVED_SERIES_BY_ZONE["DE"],
    ]
    pd.testing.assert_series_equal(
        observed.reindex(current_delivery),
        auction.reindex(current_delivery).rename("actual"),
    )
    fallback = source["post_auction_fallback"]
    assert fallback["status"] == "complete_missing_suffix"
    assert fallback["applied_current_hours"] == 24
    assert fallback["applied_suffix_hours"] == 48
    assert fallback["missing_canonical_suffix_hours"] == 48
    assert fallback["used_for_prediction"] is False
    assert fallback["validation_paired_hours"] == 7 * 24
    assert fallback["validation_minimum_paired_hours"] == 7 * 24


def test_latest_observed_snapshot_rejects_divergent_auction_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_delivery = pd.date_range(
        "2026-08-19T22:00:00Z", periods=24, freq="h"
    )
    expected = pd.date_range(
        current_delivery[0] - pd.Timedelta(days=8),
        current_delivery[-1],
        freq="h",
    )
    canonical = pd.Series(np.arange(len(expected), dtype=float), index=expected)
    canonical.loc[current_delivery] = np.nan
    auction = pd.Series(np.arange(len(expected), dtype=float) + 1.0, index=expected)
    spec = SimpleNamespace(
        zone="BE",
        timezone="Europe/Brussels",
        delivery_day="2026-08-20",
        history_contract=SimpleNamespace(target_series="canonical_target"),
    )

    def fake_fetch(_client, series, *_args, **_kwargs):
        return canonical if series == "canonical_target" else auction

    monkeypatch.setattr(launcher, "fetch_saturn_series_from_client", fake_fetch)
    with pytest.raises(ValueError, match="source post-enchere diverge"):
        launcher._fetch_latest_observed_snapshot(
            object(), spec=spec, expected_index=expected
        )


def test_kalman_rebases_storm_dst_audit_after_removing_delivery_day() -> None:
    timezone_name = "Europe/Paris"
    full_delivery = pd.date_range(
        pd.Timestamp("2025-08-12", tz=timezone_name),
        pd.Timestamp("2026-08-29", tz=timezone_name),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    local_days = pd.Index(full_delivery.tz_convert(timezone_name).date)
    retained = full_delivery[local_days < date(2026, 8, 28)]
    allowed_missing = pd.Timestamp("2025-10-26T00:00:00Z")
    storm = np.full(len(retained), 80.0)
    storm[retained == allowed_missing] = np.nan
    statistics = pd.DataFrame(
        {
            "delivery_start_utc": retained,
            "actual": np.full(len(retained), 79.0),
            "storm_dashboard_official__q50": storm,
        }
    )
    audit: dict[str, object] = {
        "storm_primary_report_benchmark": "storm_dashboard_official__q50",
        "storm_dashboard": {
            "source": {
                "series": "power.price.fr.euromwh.h.fcst.3mv.storm.da.cache"
            },
            "expected_hours": len(full_delivery),
            "available_hours": len(full_delivery) - 1,
            "missing_hours": 1,
            "dst": {
                "interpolation": False,
                "strict_08_fallback": False,
                "native_allowed_missing_hours": 1,
                "native_allowed_missing_utc": [str(allowed_missing)],
                "native_actual_missing_matches_allowed": True,
            },
        },
    }

    launcher._rebase_storm_dashboard_audit_for_kalman(
        audit,
        statistics,
        timezone=timezone_name,
    )

    storm_audit = audit["storm_dashboard"]
    assert isinstance(storm_audit, dict)
    assert storm_audit["expected_hours"] == len(retained) == 9144
    assert storm_audit["available_hours"] == len(retained) - 1
    assert storm_audit["missing_hours"] == 1
    dst = storm_audit["dst"]
    assert dst["expected_local_day_hour_histogram"] == {
        "23": 1,
        "24": 379,
        "25": 1,
    }
    assert dst["available_local_day_hour_histogram"] == {
        "23": 1,
        "24": 380,
    }
    assert dst["native_allowed_missing_hours"] == 1
    assert dst["native_actual_missing_matches_allowed"] is True


def test_kalman_rebases_storm_with_empty_current_actual_placeholder() -> None:
    timezone_name = "Europe/Paris"
    previous = local_delivery_day_index(
        date(2026, 8, 28), timezone=timezone_name
    )
    current = local_delivery_day_index(
        date(2026, 8, 29), timezone=timezone_name
    )
    delivery = previous.append(current)
    statistics = pd.DataFrame(
        {
            "delivery_start_utc": delivery,
            "actual": np.r_[
                np.full(len(previous), 79.0),
                np.full(len(current), np.nan),
            ],
            "storm_dashboard_official__q50": np.full(len(delivery), 80.0),
        }
    )
    audit: dict[str, object] = {
        "storm_primary_report_benchmark": "storm_dashboard_official__q50",
        "canonical_actuals": {
            "current_delivery_placeholder": True,
            "current_delivery_day_local": "2026-08-29",
        },
        "storm_dashboard": {
            "source": {
                "series": "power.price.fr.euromwh.h.fcst.3mv.storm.da.cache"
            },
            "dst": {
                "interpolation": False,
                "strict_08_fallback": False,
                "native_allowed_missing_hours": 0,
                "native_allowed_missing_utc": [],
                "native_actual_missing_matches_allowed": True,
            },
        },
    }

    launcher._rebase_storm_dashboard_audit_for_kalman(
        audit,
        statistics,
        timezone=timezone_name,
    )

    storm_audit = audit["storm_dashboard"]
    assert isinstance(storm_audit, dict)
    assert storm_audit["actual_available_hours"] == len(previous)
    assert storm_audit["actual_missing_hours"] == len(current)
    assert storm_audit["allowed_missing_actual_hours"] == len(current)
    assert storm_audit["metrics"]["n"] == len(previous)


def test_statistics_completion_message_uses_the_exported_dynamic_end() -> None:
    export = launcher.ForecastExport(
        variant="blend",
        source_model="mkonline_blend",
        csv_path=Path("forecast.csv"),
        report_path=Path("forecast.html"),
        source_archive=Path("archive"),
        statistics_end_local="2026-08-27",
    )

    assert launcher._statistics_completion_message((export,)) == (
        " Statistics des rapports completes jusqu'au 2026-08-27."
    )


def test_statistics_price_report_is_verified_against_refreshed_snapshot(
    tmp_path: Path,
) -> None:
    reporting_view = tmp_path / "view"
    reporting_view.mkdir()
    delivery = pd.date_range("2026-08-25T22:00:00Z", periods=48, freq="h")
    history = pd.DataFrame(
        {
            "delivery_start_utc": delivery,
            "actual": np.r_[np.full(24, 100.0), np.full(24, 120.0)],
            "mkonline_blend__q50": np.r_[
                np.full(24, 101.0),
                np.full(24, 119.0),
            ],
            "storm_dashboard_official__q50": np.r_[
                np.full(24, 103.0),
                np.full(24, 118.0),
            ],
        }
    )
    history.to_csv(
        reporting_view / "statistics_history_hourly.csv.gz",
        index=False,
        compression="gzip",
    )
    (reporting_view / "statistics_history_audit.json").write_text(
        json.dumps(
            {
                "canonical_actuals": {
                    "source": {"extracted_at_utc": "2026-08-27T12:00:00Z"}
                },
                "storm_primary_report_benchmark": (
                    "storm_dashboard_official__q50"
                ),
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "metrics": [{"key": "mean_price"}],
        "records": [
            {
                "zone": "FR",
                "sample": "daily",
                "period_start": "2026-08-26",
                "observed_mean_price": 100.0,
                "mean_price": 101.0,
                "benchmark_mean_price": 103.0,
            },
            {
                "zone": "FR",
                "sample": "daily",
                "period_start": "2026-08-27",
                "observed_mean_price": 120.0,
                "mean_price": 119.0,
                "benchmark_mean_price": 118.0,
            },
        ],
    }
    report = tmp_path / "report.html"
    report.write_text(
        "<div data-report-subsection=\"mean-price-refresh\"></div>"
        f"<script>const payload = {json.dumps(payload)};</script>",
        encoding="utf-8",
    )
    spec = SimpleNamespace(
        zone="FR",
        source_model="mkonline_blend",
        timezone="Europe/Paris",
    )

    launcher._validate_statistics_price_report(
        spec,
        reporting_view=reporting_view,
        report_path=report,
    )

    payload["records"][-1]["observed_mean_price"] = 121.0
    report.write_text(
        "<div data-report-subsection=\"mean-price-refresh\"></div>"
        f"<script>const payload = {json.dumps(payload)};</script>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="observed_mean_price HTML divergent"):
        launcher._validate_statistics_price_report(
            spec,
            reporting_view=reporting_view,
            report_path=report,
        )


def test_statistics_price_report_accepts_current_blank_observed_placeholder(
    tmp_path: Path,
) -> None:
    reporting_view = tmp_path / "view"
    reporting_view.mkdir()
    delivery = pd.date_range("2026-08-25T22:00:00Z", periods=72, freq="h")
    history = pd.DataFrame(
        {
            "delivery_start_utc": delivery,
            "actual": np.r_[
                np.full(24, 100.0),
                np.full(24, 120.0),
                np.full(24, np.nan),
            ],
            "mkonline_blend__q50": np.r_[
                np.full(24, 101.0),
                np.full(24, 119.0),
                np.full(24, 130.0),
            ],
            "storm_dashboard_official__q50": np.r_[
                np.full(24, 103.0),
                np.full(24, 118.0),
                np.full(24, 128.0),
            ],
        }
    )
    history.to_csv(
        reporting_view / "statistics_history_hourly.csv.gz",
        index=False,
        compression="gzip",
    )
    (reporting_view / "statistics_history_audit.json").write_text(
        json.dumps(
            {
                "canonical_actuals": {
                    "source": {"extracted_at_utc": "2026-08-28T08:00:00Z"}
                },
                "storm_primary_report_benchmark": (
                    "storm_dashboard_official__q50"
                ),
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "metrics": [{"key": "mean_price"}],
        "records": [
            {
                "zone": "FR",
                "sample": "daily",
                "period_start": "2026-08-26",
                "observed_mean_price": 100.0,
                "mean_price": 101.0,
                "benchmark_mean_price": 103.0,
            },
            {
                "zone": "FR",
                "sample": "daily",
                "period_start": "2026-08-27",
                "observed_mean_price": 120.0,
                "mean_price": 119.0,
                "benchmark_mean_price": 118.0,
            },
            {
                "zone": "FR",
                "sample": "daily",
                "period_start": "2026-08-28",
                "observed_mean_price": None,
                "mean_price": 130.0,
                "benchmark_mean_price": 128.0,
            },
        ],
    }
    report = tmp_path / "report.html"
    report.write_text(
        '<div data-report-subsection="mean-price-refresh"></div>'
        f"<script>const payload = {json.dumps(payload)};</script>",
        encoding="utf-8",
    )
    spec = SimpleNamespace(
        zone="FR",
        source_model="mkonline_blend",
        timezone="Europe/Paris",
        delivery_day="2026-08-28",
    )

    launcher._validate_statistics_price_report(
        spec,
        reporting_view=reporting_view,
        report_path=report,
    )

    payload["records"][-1]["observed_mean_price"] = 0.0
    report.write_text(
        '<div data-report-subsection="mean-price-refresh"></div>'
        f"<script>const payload = {json.dumps(payload)};</script>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="observed_mean_price HTML divergent"):
        launcher._validate_statistics_price_report(
            spec,
            reporting_view=reporting_view,
            report_path=report,
        )


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


def test_failed_process_reuses_concurrently_published_valid_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, executable = _runtime_paths(tmp_path)
    archive = _archive(tmp_path, "DE")
    log = tmp_path / "de_publish_race.log"
    log.write_text("PermissionError: [WinError 5] Access is denied", encoding="utf-8")
    _patch_statuses(monkeypatch, tmp_path)
    process = SimpleNamespace(wait=lambda: 1, poll=lambda: 1)
    command = (str(executable), "runner.py", "--zone", "DE")
    handle = ForecastProcess(
        zone="DE",
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
        zones=("DE",),
        delivery_day="2026-08-20",
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        log_dir=tmp_path / "logs",
    )

    result = batch.results[0]
    assert batch.ok
    assert result.state == "skipped"
    assert result.return_code == 0
    assert result.archive_path == archive
    assert result.command == command
    assert "execution concurrente" in result.message
    assert "archive integralement validee et reutilisee" in result.message


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
