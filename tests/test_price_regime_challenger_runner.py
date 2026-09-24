from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import run_price_regime_challenger as challenger

from chronos2_hourly.hourly_contract import local_delivery_day_index
from run_price_regime_challenger import (
    ControlArchive,
    RegimeChallengerRunnerError,
    _atomic_publish,
    _existing_control_paths,
    _live_hourly,
    build_plan,
    load_control_archive,
    parse_args,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "config" / "price_regime_challenger.yaml"
REGISTRY = PROJECT_ROOT / "chronos2_hourly_live_zones.yaml"


def test_plan_preserves_both_semantics_and_shadow_contract() -> None:
    plan = build_plan(
        zones=["FR", "DE", "BE", "NL", "ES"],
        mode="Both",
        delivery_day="2026-08-25",
        config_path=CONFIG,
        reuse_forecasts=True,
    )
    assert plan["zones"] == ["FR", "DE", "BE", "NL", "ES"]
    assert plan["variants"] == {
        "FR": ["autonomous", "blend"],
        "DE": ["autonomous"],
        "BE": ["autonomous"],
        "NL": ["autonomous", "blend"],
        "ES": ["autonomous"],
    }
    assert plan["production_eligible"] is False
    assert plan["official_run_first"] is False


def test_plan_reuses_official_controls_by_default() -> None:
    plan = build_plan(
        zones=["FR"],
        mode="Autonomous",
        delivery_day="2026-08-25",
        config_path=CONFIG,
    )
    assert plan["reuse_forecasts"] is True
    assert plan["official_run_first"] is False


def test_cli_requires_explicit_opt_in_to_run_official_forecasts() -> None:
    assert parse_args([]).reuse_forecasts is True
    assert parse_args(["--reuse-forecasts"]).reuse_forecasts is True
    assert parse_args(["--run-forecasts-first"]).reuse_forecasts is False


@pytest.mark.parametrize("mode", ["Production", "All"])
def test_plan_rejects_unsupported_modes(mode: str) -> None:
    with pytest.raises(ValueError, match="Autonomous, Blend ou Both"):
        build_plan(
            zones=["FR"],
            mode=mode,
            delivery_day="2026-08-25",
            config_path=CONFIG,
        )


def test_run_rejects_all_before_loading_configuration_or_launching_controls(monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("Unsupported challenger mode launched an official forecast")
    monkeypatch.setattr(challenger, "run_forecast_batch", forbidden)
    with pytest.raises(ValueError, match="Autonomous, Blend ou Both"):
        challenger.run_challenger(
            zones=["FR"], mode="All", delivery_day="2026-08-25",
            config_path="missing-config.yaml", reuse_forecasts=False,
        )


@pytest.mark.parametrize("zone", ["DE", "BE", "ES"])
def test_challenger_blend_is_restricted_to_original_promoted_zones(zone):
    with pytest.raises(ValueError, match="FR et NL"):
        build_plan(zones=[zone], mode="Blend", delivery_day="2026-08-25", config_path=CONFIG)


def test_both_plan_records_production_control_launch_without_kalman_exports():
    plan = build_plan(zones=["FR", "DE", "BE", "NL", "ES"], mode="Both",
                      delivery_day="2026-08-25", config_path=CONFIG, reuse_forecasts=False)
    assert plan["official_control_mode"] == "production"
    assert plan["official_run_first"] is True
    assert plan["variants"]["FR"] == plan["variants"]["NL"] == ["autonomous", "blend"]
    assert all("kalman" not in values for values in plan["variants"].values())


def test_reuse_paths_follow_each_configured_zone_output_root() -> None:
    paths = _existing_control_paths(
        registry_path=REGISTRY,
        zones=("FR", "DE", "BE", "NL", "ES"),
        delivery_day="2026-08-25",
    )
    assert paths["FR"] == (
        PROJECT_ROOT / "runs" / "live" / "fr_day_ahead_2026-08-25"
    ).resolve()
    assert paths["DE"] == (
        PROJECT_ROOT / "runs" / "live" / "de" / "de_day_ahead_2026-08-25"
    ).resolve()
    assert paths["NL"] == (
        PROJECT_ROOT
        / "runs"
        / "live"
        / "nl_mkonline_v1"
        / "nl_day_ahead_2026-08-25"
    ).resolve()


def test_atomic_publish_retries_transient_windows_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".challenger.tmp"
    final = tmp_path / "2026-08-28"
    staging.mkdir()
    (staging / "report.html").write_text("sealed", encoding="utf-8")
    real_replace = Path.replace
    attempts = 0

    def flaky_replace(path: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("transient Windows lock")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr(
        "chronos2_hourly.multizone_live.time.sleep", lambda _delay: None
    )

    _atomic_publish(staging, final, overwrite=False)

    assert attempts == 3
    assert (final / "report.html").read_text(encoding="utf-8") == "sealed"
    assert not staging.exists()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal_control(archive: Path) -> None:
    artifacts = []
    for path in sorted(
        (
            archive / "run_manifest.json",
            archive / "forecast_hourly_fr.csv",
            archive / "backtest_hourly_oof.csv.gz",
        )
    ):
        artifacts.append(
            {
                "path": path.name,
                "role": "run_artifact",
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    (archive / "artifact_checksums.json").write_text(
        json.dumps({"algorithm": "sha256", "artifacts": artifacts}),
        encoding="utf-8",
    )


def _write_control(
    tmp_path: Path,
    *,
    malformed_grid: bool = False,
    live_origin: str = "2026-08-24T06:00:00Z",
    manifest_cutoff: str = "2026-08-24T06:00:00Z",
    include_autonomous_history: bool = True,
) -> Path:
    archive = tmp_path / "fr_day_ahead_2026-08-25"
    archive.mkdir()
    grid = local_delivery_day_index("2026-08-25", timezone="Europe/Paris")
    if malformed_grid:
        grid = grid[:-1]
    forecast = pd.DataFrame({"delivery_start_utc": grid})
    for model in ("residual_corrected", "mkonline_blend"):
        forecast[f"{model}__q10"] = 40.0
        forecast[f"{model}__q50"] = 50.0
        forecast[f"{model}__q90"] = 60.0
    forecast["forecast_origin_utc"] = live_origin
    forecast["mkonline_blend_forecast_origin_utc"] = live_origin
    forecast.to_csv(archive / "forecast_hourly_fr.csv", index=False)

    history_grid = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
    history = pd.DataFrame(
        {
            "delivery_start_utc": history_grid,
            "actual": 52.0,
            "forecast_origin_utc": history_grid - pd.Timedelta(hours=18),
            "mkonline_blend_forecast_origin_utc": history_grid
            - pd.Timedelta(hours=18),
        }
    )
    for model in ("residual_corrected", "mkonline_blend"):
        history[f"{model}__q10"] = 40.0
        history[f"{model}__q50"] = 50.0
        history[f"{model}__q90"] = 60.0
    if not include_autonomous_history:
        history = history.drop(
            columns=[
                "residual_corrected__q10",
                "residual_corrected__q50",
                "residual_corrected__q90",
            ]
        )
    history.to_csv(archive / "backtest_hourly_oof.csv.gz", index=False)
    (archive / "run_manifest.json").write_text(
        json.dumps(
            {
                "zone": "FR",
                "delivery_day_local": "2026-08-25",
                "run_type": "live_day_ahead",
                "forecast_status": "issued_live",
                "timezone": "Europe/Paris",
                "forecast_cutoff_utc": manifest_cutoff,
            }
        ),
        encoding="utf-8",
    )
    _seal_control(archive)
    return archive


def test_control_archive_validates_exact_both_grid(tmp_path: Path) -> None:
    archive = _write_control(tmp_path)
    control = load_control_archive(
        archive,
        zone="FR",
        delivery_day="2026-08-25",
        mode="Both",
        project_root=tmp_path,
    )
    assert len(control.forecast) == 24
    assert control.current_actual.isna().all()
    assert len(control.hashes["forecast_sha256"]) == 64
    assert len(control.hashes["checksums_manifest_sha256"]) == 64


@pytest.mark.parametrize("mode,batch_mode,variants", [
    ("Both", "production", ["autonomous", "blend"]),
    ("Autonomous", "autonomous", ["autonomous"]),
    ("Blend", "blend", ["blend"]),
])
def test_run_forecasts_first_consumes_sealed_archive_with_original_variants(
    tmp_path, monkeypatch, mode, batch_mode, variants,
):
    archive = _write_control(tmp_path)
    calls = []
    used_variants = []

    def batch(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(results=[SimpleNamespace(ok=True, zone="FR", archive_path=archive)])

    original_history = challenger._variant_history
    def history(control, *, variant, label_end_day):
        assert control.path == archive
        used_variants.append(variant)
        return original_history(control, variant=variant, label_end_day=label_end_day)

    class ControlsConsumed(Exception):
        pass

    def before_feature_loading(*args, **kwargs):
        raise ControlsConsumed

    monkeypatch.setattr(challenger, "run_forecast_batch", batch)
    monkeypatch.setattr(challenger, "print_batch_summary", lambda value: None)
    monkeypatch.setattr(challenger, "_variant_history", history)
    monkeypatch.setattr(challenger, "_load_residual_features", before_feature_loading)
    with pytest.raises(ControlsConsumed):
        challenger.run_challenger(
            zones=["FR"], mode=mode, delivery_day="2026-08-25",
            config_path=CONFIG, reuse_forecasts=False,
        )
    assert len(calls) == 1 and calls[0]["mode"] == batch_mode
    assert used_variants == variants


def test_control_archive_rejects_partial_local_day(tmp_path: Path) -> None:
    archive = _write_control(tmp_path, malformed_grid=True)
    with pytest.raises(RegimeChallengerRunnerError, match="grille live"):
        load_control_archive(
            archive,
            zone="FR",
            delivery_day="2026-08-25",
            mode="Both",
            project_root=tmp_path,
        )


def test_control_archive_rejects_late_live_origin(tmp_path: Path) -> None:
    archive = _write_control(tmp_path, live_origin="2026-08-24T07:00:00Z")
    with pytest.raises(RegimeChallengerRunnerError, match="origine live attendue"):
        load_control_archive(
            archive,
            zone="FR",
            delivery_day="2026-08-25",
            mode="Both",
            project_root=tmp_path,
        )


def test_control_archive_rejects_wrong_manifest_cutoff(tmp_path: Path) -> None:
    archive = _write_control(
        tmp_path, manifest_cutoff="2026-08-24T07:00:00Z"
    )
    with pytest.raises(RegimeChallengerRunnerError, match="forecast_cutoff_utc"):
        load_control_archive(
            archive,
            zone="FR",
            delivery_day="2026-08-25",
            mode="Both",
            project_root=tmp_path,
        )


def test_control_archive_rejects_file_changed_after_seal(tmp_path: Path) -> None:
    archive = _write_control(tmp_path)
    forecast_path = archive / "forecast_hourly_fr.csv"
    forecast_path.write_bytes(forecast_path.read_bytes() + b"\n")
    with pytest.raises(RegimeChallengerRunnerError, match="checksum"):
        load_control_archive(
            archive,
            zone="FR",
            delivery_day="2026-08-25",
            mode="Both",
            project_root=tmp_path,
        )


def test_blend_control_requires_autonomous_history_context(tmp_path: Path) -> None:
    archive = _write_control(tmp_path, include_autonomous_history=False)
    with pytest.raises(RegimeChallengerRunnerError, match="autonomous.*historique"):
        load_control_archive(
            archive,
            zone="FR",
            delivery_day="2026-08-25",
            mode="Blend",
            project_root=tmp_path,
        )


def test_live_rows_use_the_target_from_the_matching_control() -> None:
    index = local_delivery_day_index("2026-08-25", timezone="Europe/Paris")
    predicted = pd.DataFrame(
        {
            "baseline_q10": 40.0,
            "baseline_q50": 50.0,
            "baseline_q90": 60.0,
            "challenger_q10": 45.0,
            "challenger_q50": 60.0,
            "challenger_q90": 75.0,
            "shock_probability": 0.8,
            "shock_magnitude_if_active": 20.0,
            "shock_premium": 10.0,
            "regime_predicted": 1,
        },
        index=index,
    )
    control = ControlArchive(
        zone="FR",
        timezone="Europe/Paris",
        delivery_day="2026-08-25",
        path=Path("control"),
        manifest={},
        forecast=pd.DataFrame(index=index),
        history=pd.DataFrame(),
        current_actual=pd.Series(np.arange(24, dtype=float), index=index),
        hashes={},
    )
    result = _live_hourly(
        predicted,
        control=control,
        variant="autonomous",
        threshold=25.0,
        solar_hours=(9, 10, 11, 12, 13, 14, 15, 16),
    )
    assert result["actual"].tolist() == list(np.arange(24, dtype=float))
    assert set(result["variant"]) == {"autonomous"}
