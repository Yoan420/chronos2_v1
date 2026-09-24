from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import app_multizone
import chronos2_hourly.app_service as app_service
from chronos2_hourly.app_service import (
    ExistingForecastArchiveError,
    ForecastProcess,
    ForecastSkip,
    PerformanceArtifact,
    StatisticsDataset,
    ZoneStatus,
    build_dispatch_command,
    build_statistics_view,
    list_run_artifacts,
    load_best_statistics_history,
    load_forecast_curve,
    load_statistics_history,
    validate_existing_forecast_archive,
    start_dispatch_process,
)
from chronos2_hourly.zone_live import ZoneBundleError
from app_multizone import _forecast_chart, _refresh_results_after_queue, _start_next


def _status(*, ready: bool = True) -> ZoneStatus:
    return ZoneStatus(
        code="DE",
        timezone="Europe/Berlin",
        enabled=ready,
        production_ready=ready,
        ready=ready,
        runner=Path("runner.py") if ready else None,
        live_config=Path("live.yaml") if ready else None,
        checks=("ok",) if ready else (),
        blockers=() if ready else ("benchmark scellé absent",),
    )


def _write_existing_live_archive(
    tmp_path: Path,
    *,
    day: str = "2026-08-15",
    zone: str = "DE",
    timezone_name: str = "Europe/Berlin",
) -> tuple[ZoneStatus, Path]:
    code = zone.upper()
    zone_lower = code.lower()
    config = tmp_path / f"{zone_lower}_live.yaml"
    config.write_text(
        "live:\n"
        f"  output_root: runs/live/{zone_lower}\n"
        f"  forecast_filename: forecast_hourly_{zone_lower}.csv\n",
        encoding="utf-8",
    )
    archive = (
        tmp_path
        / "runs"
        / "live"
        / zone_lower
        / f"{zone_lower}_day_ahead_{day}"
    )
    archive.mkdir(parents=True)
    delivery = pd.date_range(
        pd.Timestamp(day, tz=timezone_name),
        periods=24,
        freq="h",
    ).tz_convert("UTC")
    pd.DataFrame(
        {
            "delivery_start_utc": delivery,
            "q10": np.arange(24, dtype=float),
            "q50": np.arange(24, dtype=float) + 1.0,
            "q90": np.arange(24, dtype=float) + 2.0,
        }
    ).to_csv(archive / f"forecast_hourly_{zone_lower}.csv", index=False)
    (archive / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_type": "live_day_ahead",
                "forecast_status": "issued_live",
                "zone": code,
                "timezone": timezone_name,
                "delivery_day_local": day,
                "sha256_manifest": "artifact_checksums.json",
            }
        ),
        encoding="utf-8",
    )
    (archive / "live_run_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "run_type": "live_day_ahead",
                "zone": code,
                "delivery_day_local": day,
                "hours": 24,
                "forecast_path": f"forecast_hourly_{zone_lower}.csv",
            }
        ),
        encoding="utf-8",
    )
    (archive / "report.html").write_text("<html>ok</html>", encoding="utf-8")
    artifacts = []
    for path in sorted(item for item in archive.rglob("*") if item.is_file()):
        artifacts.append(
            {
                "path": path.relative_to(archive).as_posix(),
                "role": "run_artifact",
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (archive / "artifact_checksums.json").write_text(
        json.dumps(
            {
                "algorithm": "sha256",
                "output_directory": str(archive.resolve()),
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    status = ZoneStatus(
        code=code,
        timezone=timezone_name,
        enabled=True,
        production_ready=True,
        ready=True,
        runner=tmp_path / "runner.py",
        live_config=config,
        checks=("ok",),
        blockers=(),
    )
    return status, archive


def _reseal_archive(archive: Path) -> None:
    artifacts = []
    for path in sorted(
        item
        for item in archive.rglob("*")
        if item.is_file() and item.name != "artifact_checksums.json"
    ):
        artifacts.append(
            {
                "path": path.relative_to(archive).as_posix(),
                "role": "run_artifact",
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (archive / "artifact_checksums.json").write_text(
        json.dumps(
            {
                "algorithm": "sha256",
                "output_directory": str(archive.resolve()),
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )


def _convert_archive_to_replay(archive: Path) -> Path:
    replay = archive.parent / "_replays" / archive.name
    replay.parent.mkdir(parents=True, exist_ok=True)
    archive.rename(replay)
    manifest_path = replay / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_type"] = "pit_replay"
    manifest["forecast_status"] = "pit_reconstruction"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    summary_path = replay / "live_run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["run_type"] = "pit_replay"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    _reseal_archive(replay)
    return replay


def _add_statistics_history(
    archive: Path,
    *,
    timezone_name: str,
    history_end_local: str,
    include_prefix: bool = True,
    include_storm: bool = True,
    internal_sha: str | None = None,
) -> None:
    timeline = pd.date_range(
        pd.Timestamp(history_end_local, tz=timezone_name),
        periods=24,
        freq="h",
    ).tz_convert("UTC")
    statistics = pd.DataFrame(
        {
            "delivery_start_utc": timeline,
            "actual": np.linspace(50.0, 73.0, len(timeline)),
            "residual_corrected__q50": np.linspace(51.0, 74.0, len(timeline)),
        }
    )
    if include_storm:
        statistics["storm_dashboard_official__q50"] = np.linspace(
            52.0,
            75.0,
            len(timeline),
        )
    statistics_path = archive / "statistics_history_hourly.csv.gz"
    statistics.to_csv(statistics_path, index=False, compression="gzip")
    delivery_day = archive.name.rsplit("_", 1)[-1]
    audit = {
        "status": "complete",
        "statistics_history_path": statistics_path.name,
        "statistics_history_sha256": internal_sha
        or hashlib.sha256(statistics_path.read_bytes()).hexdigest(),
        "statistics_audit_path": "statistics_history_audit.json",
        "current_delivery_day_local": delivery_day,
        "n_total_statistics_hours": len(statistics),
        "statistics_complete": True,
        "missing_realized_days": [],
    }
    if include_prefix:
        audit["statistics_prefix_end_local"] = history_end_local
    if include_storm:
        audit["storm_primary_report_benchmark"] = (
            "storm_dashboard_official__q50"
        )
    (archive / "statistics_history_audit.json").write_text(
        json.dumps(audit),
        encoding="utf-8",
    )
    _reseal_archive(archive)


def test_dispatch_command_is_an_argv_list_and_refuses_unready_zone(
    tmp_path: Path,
) -> None:
    (tmp_path / "run_mkonline_live_zone.py").write_text("pass\n", encoding="utf-8")
    registry = tmp_path / "zones.yaml"
    registry.write_text("schema_version: 1\nzones: {}\n", encoding="utf-8")
    executable = tmp_path / "python.exe"
    executable.write_bytes(b"")

    command = build_dispatch_command(
        _status(),
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        delivery_day="2026-08-14",
        device="cpu",
        threads=3,
        workers=2,
        local_files_only=True,
    )
    assert isinstance(command, list)
    assert command[:2] == [
        str(executable.resolve()),
        str((tmp_path / "run_mkonline_live_zone.py").resolve()),
    ]
    assert command[command.index("--zone") + 1] == "DE"
    assert command[-1] == "--local-files-only"
    with pytest.raises(ZoneBundleError, match="lancement refusé"):
        build_dispatch_command(
            _status(ready=False),
            project_root=tmp_path,
            registry_path=registry,
            python_executable=executable,
        )


def test_dispatch_command_adds_only_the_opt_in_chronos2_bundle_flags(
    tmp_path: Path,
) -> None:
    (tmp_path / "run_mkonline_live_zone.py").write_text("pass\n", encoding="utf-8")
    registry = tmp_path / "zones.yaml"
    registry.write_text("schema_version: 1\nzones: {}\n", encoding="utf-8")
    executable = tmp_path / "python.exe"
    executable.write_bytes(b"")
    bundle = tmp_path / "runs" / "experiments" / "bundle" / "manifest.json"

    production = build_dispatch_command(
        _status(),
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        local_files_only=True,
    )
    challenger = build_dispatch_command(
        _status(),
        project_root=tmp_path,
        registry_path=registry,
        python_executable=executable,
        local_files_only=True,
        residual_load_source="chronos2",
        residual_load_bundle_manifest=bundle,
    )

    assert "--residual-load-source" not in production
    assert challenger[-4:] == [
        "--residual-load-source",
        "chronos2",
        "--residual-load-bundle-manifest",
        str(bundle.resolve()),
    ]


def test_start_process_forces_shell_false_and_redirects_to_project_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    class FakeProcess:
        pid = 123

        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr("chronos2_hourly.app_service.subprocess.Popen", fake_popen)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.corp.example:8080")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.corp.example:8080")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.corp.example:8080")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(tmp_path / "corporate-ca.pem"))
    handle = start_dispatch_process(
        ["python.exe", "runner.py", "--zone", "FR"],
        zone="FR",
        project_root=tmp_path,
        log_dir=tmp_path / "runs" / "logs",
    )
    assert isinstance(handle, ForecastProcess)
    assert captured["shell"] is False
    assert captured["command"] == ["python.exe", "runner.py", "--zone", "FR"]
    assert handle.log_path.is_file()
    with pytest.raises(ValueError, match="doit rester dans le projet"):
        start_dispatch_process(
            ["python.exe", "runner.py"],
            zone="FR",
            project_root=tmp_path,
            log_dir=tmp_path.parent / "outside",
        )


def test_dispatch_detects_only_known_loopback_blackhole() -> None:
    blocked = app_service._local_blackhole_proxy_names(
        {
            "HTTPS_PROXY": "http://localhost:9",
            "ALL_PROXY": "[::1]:9",
            "HTTP_PROXY": "http://proxy.corp.example:3128",
            "REQUESTS_CA_BUNDLE": r"C:\certs\company.pem",
            "UNCHANGED": "yes",
        }
    )

    assert set(blocked) == {"HTTPS_PROXY", "ALL_PROXY"}


def test_start_process_refuses_sandbox_proxy_without_spawning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    popen = Mock(side_effect=AssertionError("Popen ne doit pas etre appele"))
    monkeypatch.setattr("chronos2_hourly.app_service.subprocess.Popen", popen)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("ALL_PROXY", raising=False)
    monkeypatch.setenv("CODEX_SANDBOX_NETWORK_DISABLED", "1")

    with pytest.raises(RuntimeError, match="Quittez completement Codex"):
        start_dispatch_process(
            ["python.exe", "runner.py", "--zone", "FR"],
            zone="FR",
            project_root=tmp_path,
            log_dir=tmp_path / "runs" / "logs",
        )
    popen.assert_not_called()


def test_start_process_cleans_stale_proxy_only_outside_codex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    class FakeProcess:
        pid = 456

        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr("chronos2_hourly.app_service.subprocess.Popen", fake_popen)
    monkeypatch.delenv("CODEX_SANDBOX_NETWORK_DISABLED", raising=False)
    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.corp.example:8080")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(tmp_path / "corporate-ca.pem"))

    start_dispatch_process(
        ["python.exe", "runner.py", "--zone", "NL"],
        zone="NL",
        project_root=tmp_path,
        log_dir=tmp_path / "runs" / "logs",
    )

    assert "HTTP_PROXY" not in captured["env"]
    assert "ALL_PROXY" not in captured["env"]
    assert captured["env"]["HTTPS_PROXY"] == "http://proxy.corp.example:8080"
    assert captured["env"]["REQUESTS_CA_BUNDLE"] == str(tmp_path / "corporate-ca.pem")


def test_existing_complete_archive_is_verified_and_skipped_without_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status, archive = _write_existing_live_archive(tmp_path)
    dispatcher = tmp_path / "run_mkonline_live_zone.py"
    dispatcher.write_text("pass\n", encoding="utf-8")
    registry = tmp_path / "zones.yaml"
    registry.write_text("schema_version: 1\nzones: {}\n", encoding="utf-8")
    executable = tmp_path / "python.exe"
    executable.write_bytes(b"")
    starter = Mock(side_effect=AssertionError("subprocess interdit pour un skip"))
    monkeypatch.setattr(app_service, "inspect_zone_statuses", lambda *_a, **_k: [status])
    monkeypatch.setattr(app_service, "start_dispatch_process", starter)

    assert validate_existing_forecast_archive(
        status,
        project_root=tmp_path,
        delivery_day="2026-08-15",
    ) == archive.resolve()
    result = app_service.launch_zone_forecast(
        zone="DE",
        project_root=tmp_path,
        registry_path=registry,
        log_dir=tmp_path / "logs",
        python_executable=executable,
        delivery_day="2026-08-15",
    )

    assert isinstance(result, ForecastSkip)
    assert result.status == "Déjà publié — ignoré"
    assert result.return_code == 0
    assert result.archive_path == archive.resolve()
    starter.assert_not_called()


def test_chronos2_launch_rejects_missing_upstream_before_process_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = _status()
    registry = tmp_path / "registry.yaml"
    registry.write_text("schema_version: 1\nzones: {}\n", encoding="utf-8")
    (tmp_path / "live.yaml").write_text(
        "live:\n"
        "  output_root: runs/live/de\n"
        "  forecast_filename: forecast_hourly_de.csv\n",
        encoding="utf-8",
    )
    starter = Mock(side_effect=AssertionError("subprocess interdit"))
    monkeypatch.setattr(
        app_service,
        "inspect_zone_statuses",
        lambda *_args, **_kwargs: [status],
    )
    monkeypatch.setattr(app_service, "start_dispatch_process", starter)

    missing = tmp_path / "runs" / "experiments" / "missing_manifest.json"
    with pytest.raises(FileNotFoundError, match="doit exister avant le lancement"):
        app_service.launch_zone_forecast(
            zone="DE",
            project_root=tmp_path,
            registry_path=registry,
            log_dir=tmp_path / "logs",
            delivery_day="2026-08-15",
            residual_load_source="chronos2",
            residual_load_bundle_manifest=missing,
        )

    starter.assert_not_called()


def test_shadow_archive_is_autonomous_after_upstream_manifest_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronos2_hourly import chronos_residual_load as residual_provider

    status, production_archive = _write_existing_live_archive(tmp_path)
    shadow = (
        production_archive.parent
        / "_challengers"
        / "residual_load_chronos2"
        / (production_archive.name + "_residual_load_chronos2")
    )
    shadow.parent.mkdir(parents=True)
    production_archive.rename(shadow)
    bundle = tmp_path / "runs" / "experiments" / "upstream" / "manifest.json"
    bundle.parent.mkdir(parents=True)
    inputs = shadow / "inputs"
    overlay_root = inputs / "residual_load_pit_overlay"
    overlay_root.mkdir(parents=True)
    archived_bundle_root = inputs / "residual_load_bundle"
    archived_bundle_root.mkdir()
    composite_manifest = overlay_root / "composite_manifest.json"
    composite_manifest.write_text("{}\n", encoding="utf-8")
    composite_sha = hashlib.sha256(composite_manifest.read_bytes()).hexdigest()
    validated_files: dict[str, dict[str, object]] = {}
    provenance_files: dict[str, dict[str, object]] = {}
    upstream_declarations: list[dict[str, object]] = []
    for alias in residual_provider.EXPECTED_ALIASES:
        artifact = overlay_root / f"{alias}.parquet"
        artifact.write_bytes(alias.encode("utf-8"))
        artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        validated_files[alias] = {
            "path": artifact.resolve(),
            "sha256": artifact_sha,
            "historical_rows": 100,
            "delivery_day_rows": 24,
        }
        provenance_files[alias] = {
            "path": artifact.relative_to(inputs).as_posix(),
            "sha256": artifact_sha,
            "historical_rows": 100,
            "delivery_day_rows": 24,
        }
        upstream_artifact = archived_bundle_root / f"{alias}.parquet"
        upstream_artifact.write_bytes(f"upstream-{alias}".encode("utf-8"))
        upstream_sha = hashlib.sha256(upstream_artifact.read_bytes()).hexdigest()
        upstream_declarations.append(
            {
                "alias": alias,
                "path": upstream_artifact.name,
                "sha256": upstream_sha,
            }
        )
        provenance_files[alias].update(
            {
                "upstream_forecast_path": str(
                    (bundle.parent / upstream_artifact.name).resolve()
                ),
                "upstream_forecast_sha256": upstream_sha,
                "archived_upstream_forecast_path": (
                    f"residual_load_bundle/{upstream_artifact.name}"
                ),
            }
        )
    bundle_payload = {
        "provider": "chronos2",
        "runtime_cutoff_utc": None,
        "model_id": None,
        "model_revision": None,
        "artifacts": upstream_declarations,
    }
    bundle.write_text(json.dumps(bundle_payload) + "\n", encoding="utf-8")
    archived_bundle = archived_bundle_root / "manifest.json"
    archived_bundle.write_bytes(bundle.read_bytes())
    bundle_sha = hashlib.sha256(bundle.read_bytes()).hexdigest()

    def fake_validate_overlay(
        archive_inputs_dir,
        *,
        upstream_manifest_path,
        upstream_origin_manifest_path,
        expected_delivery_day,
        expected_runtime_cutoff,
    ):
        assert Path(archive_inputs_dir).resolve() == inputs.resolve()
        assert Path(upstream_manifest_path).resolve() == archived_bundle.resolve()
        assert Path(upstream_origin_manifest_path).resolve() == bundle.resolve()
        assert expected_delivery_day == "2026-08-15"
        assert pd.Timestamp(expected_runtime_cutoff) == pd.Timestamp(
            "2026-08-14T06:00:00Z"
        )
        return {
            "manifest_path": composite_manifest.resolve(),
            "manifest_sha256": composite_sha,
            "files": validated_files,
        }

    monkeypatch.setattr(
        residual_provider,
        "validate_archived_residual_load_overlay",
        fake_validate_overlay,
    )

    manifest_path = shadow / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "run_type": "shadow_live_day_ahead",
            "forecast_status": "shadow_challenger",
            "residual_load_source": "chronos2",
            "production_eligible": False,
            "residual_load_bundle_manifest_path": (
                "inputs/residual_load_bundle/manifest.json"
            ),
            "residual_load_bundle_origin_manifest_path": str(bundle.resolve()),
            "residual_load_bundle_manifest_sha256": bundle_sha,
            "forecast_cutoff_utc": "2026-08-14T06:00:00Z",
            "input_diagnostics": {
                "residual_load_provider": {
                    "provider": "chronos2",
                    "target_source_kind": "observed_entsoe",
                    "saturn_forecast_series_used": False,
                    "historical_context_source": "production_pit_unchanged",
                    "delivery_day_values_source": "chronos2",
                    "manifest_path": str(bundle.resolve()),
                    "manifest_sha256": bundle_sha,
                    "archived_manifest_path": (
                        "residual_load_bundle/manifest.json"
                    ),
                    "archived_manifest_sha256": bundle_sha,
                    "composite_path_base": "runtime_inputs_directory",
                    "composite_manifest_path": composite_manifest.relative_to(
                        inputs
                    ).as_posix(),
                    "composite_manifest_sha256": composite_sha,
                    "delivery_day_local": "2026-08-15",
                    "runtime_cutoff_utc": None,
                    "model_id": None,
                    "model_revision": None,
                    "files": provenance_files,
                }
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    summary_path = shadow / "live_run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "run_type": "shadow_live_day_ahead",
            "forecast_status": "shadow_challenger",
            "residual_load_source": "chronos2",
            "production_eligible": False,
        }
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    _reseal_archive(shadow)

    assert shadow.parent == (
        tmp_path
        / "runs"
        / "live"
        / "de"
        / "_challengers"
        / "residual_load_chronos2"
    ).resolve()
    assert production_archive.parent == (
        tmp_path / "runs" / "live" / "de"
    ).resolve()

    assert (
        validate_existing_forecast_archive(
            status,
            project_root=tmp_path,
            delivery_day="2026-08-15",
        )
        is None
    )
    assert validate_existing_forecast_archive(
        status,
        project_root=tmp_path,
        delivery_day="2026-08-15",
        residual_load_source="chronos2",
        residual_load_bundle_manifest=bundle,
    ) == shadow.resolve()

    bundle.unlink()
    assert validate_existing_forecast_archive(
        status,
        project_root=tmp_path,
        delivery_day="2026-08-15",
        residual_load_source="chronos2",
    ) == shadow.resolve()

    starter = Mock(side_effect=AssertionError("subprocess interdit pour un skip"))
    monkeypatch.setattr(
        app_service,
        "inspect_zone_statuses",
        lambda *_args, **_kwargs: [status],
    )
    monkeypatch.setattr(app_service, "start_dispatch_process", starter)
    reused = app_service.launch_zone_forecast(
        zone="DE",
        project_root=tmp_path,
        registry_path=tmp_path / "unused-registry.yaml",
        log_dir=tmp_path / "logs",
        delivery_day="2026-08-15",
        residual_load_source="chronos2",
    )
    assert isinstance(reused, ForecastSkip)
    assert reused.archive_path == shadow.resolve()
    starter.assert_not_called()

    archived_bundle.write_text('{"provider":"tampered"}\n', encoding="utf-8")
    with pytest.raises(ExistingForecastArchiveError, match="archive divergent"):
        validate_existing_forecast_archive(
            status,
            project_root=tmp_path,
            delivery_day="2026-08-15",
            residual_load_source="chronos2",
        )


def test_shadow_validator_rejects_a_challenger_junction_escape(
    tmp_path: Path,
) -> None:
    status, production_archive = _write_existing_live_archive(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    junction = production_archive.parent / "_challengers"
    try:
        junction.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Liens de dossier indisponibles: {exc}")

    with pytest.raises(
        ExistingForecastArchiveError,
        match="racine challenger hors output_root",
    ):
        validate_existing_forecast_archive(
            status,
            project_root=tmp_path,
            delivery_day="2026-08-15",
            residual_load_source="chronos2",
        )


@pytest.mark.parametrize("corruption", ("incomplete", "tampered"))
def test_existing_invalid_archive_is_an_explicit_blocker(
    tmp_path: Path,
    corruption: str,
) -> None:
    status, archive = _write_existing_live_archive(tmp_path)
    if corruption == "incomplete":
        (archive / "run_manifest.json").unlink()
    else:
        with (archive / "forecast_hourly_de.csv").open("a", encoding="utf-8") as stream:
            stream.write("\n")

    with pytest.raises(ExistingForecastArchiveError, match="Archive existante invalide|divergente"):
        validate_existing_forecast_archive(
            status,
            project_root=tmp_path,
            delivery_day="2026-08-15",
        )


def test_app_queue_records_skip_success_and_invalid_archive_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = SimpleNamespace(
        active_forecast=None,
        forecast_queue=["DE"],
        forecast_jobs=[],
        launch_options={"delivery_day": date(2026, 8, 15)},
    )
    st = SimpleNamespace(session_state=state)
    skip = ForecastSkip(
        zone="DE",
        delivery_day="2026-08-15",
        archive_path=tmp_path / "de_day_ahead_2026-08-15",
    )
    monkeypatch.setattr(app_multizone, "launch_zone_forecast", Mock(return_value=skip))

    _start_next(st)

    assert state.active_forecast is None
    assert state.forecast_queue == []
    assert state.forecast_jobs == [skip]

    state.forecast_queue = ["DE"]
    monkeypatch.setattr(
        app_multizone,
        "launch_zone_forecast",
        Mock(side_effect=ExistingForecastArchiveError("checksum divergent")),
    )
    _start_next(st)

    blocked = state.forecast_jobs[-1]
    assert blocked["status"] == "Bloqué — archive existante invalide"
    assert "checksum divergent" in blocked["error"]
    assert state.active_forecast is None
    assert state.forecast_queue == []


def test_completed_queue_refreshes_result_caches_and_full_app_once() -> None:
    cached_artifacts = Mock()
    cached_statistics = Mock()
    st = SimpleNamespace(
        session_state=SimpleNamespace(
            active_forecast=None,
            forecast_queue=[],
            results_refresh_pending=True,
            results_refresh_completed=False,
        ),
        rerun=Mock(),
    )

    assert _refresh_results_after_queue(
        st,
        cached_artifacts=cached_artifacts,
        cached_statistics=cached_statistics,
    )
    cached_artifacts.clear.assert_called_once_with()
    cached_statistics.clear.assert_called_once_with()
    st.rerun.assert_called_once_with(scope="app")
    assert st.session_state.results_refresh_pending is False
    assert st.session_state.results_refresh_completed is True

    assert not _refresh_results_after_queue(
        st,
        cached_artifacts=cached_artifacts,
        cached_statistics=cached_statistics,
    )
    cached_artifacts.clear.assert_called_once_with()
    cached_statistics.clear.assert_called_once_with()
    st.rerun.assert_called_once_with(scope="app")


def test_result_refresh_waits_until_queue_is_idle() -> None:
    cached_artifacts = Mock()
    cached_statistics = Mock()
    st = SimpleNamespace(
        session_state=SimpleNamespace(
            active_forecast=object(),
            forecast_queue=["NL"],
            results_refresh_pending=True,
            results_refresh_completed=False,
        ),
        rerun=Mock(),
    )

    assert not _refresh_results_after_queue(
        st,
        cached_artifacts=cached_artifacts,
        cached_statistics=cached_statistics,
    )
    cached_artifacts.clear.assert_not_called()
    cached_statistics.clear.assert_not_called()
    st.rerun.assert_not_called()
    assert st.session_state.results_refresh_pending is True
    assert st.session_state.results_refresh_completed is False


def test_artifact_listing_finds_reports_and_statistics(tmp_path: Path) -> None:
    run = tmp_path / "_reports" / "de_day_ahead_2026-08-14"
    run.mkdir(parents=True)
    (run / "statistics_history_hourly.csv.gz").write_bytes(b"stats")
    (run / "de_report.html").write_text("<html></html>", encoding="utf-8")
    (run / "live_run_summary.json").write_text(
        json.dumps({"zone": "DE"}), encoding="utf-8"
    )
    artifacts = list_run_artifacts(tmp_path, zones=("DE",))
    assert len(artifacts) == 1
    assert artifacts[0].zone == "DE"
    assert artifacts[0].kind == "rapport"
    assert artifacts[0].report_path == run / "de_report.html"
    assert artifacts[0].statistics_path == run / "statistics_history_hourly.csv.gz"


def test_artifact_listing_labels_residual_load_shadow_as_prospective(
    tmp_path: Path,
) -> None:
    run = (
        tmp_path
        / "de"
        / "_challengers"
        / "residual_load_chronos2"
        / "de_day_ahead_2026-08-25_residual_load_chronos2"
    )
    run.mkdir(parents=True)
    (run / "live_run_summary.json").write_text(
        json.dumps(
            {
                "zone": "DE",
                "forecast_status": "shadow_challenger",
                "production_eligible": False,
            }
        ),
        encoding="utf-8",
    )
    (run / "forecast_hourly_de.csv").write_text(
        "delivery_start_utc,q50\n2026-08-24T22:00:00Z,50\n",
        encoding="utf-8",
    )

    artifacts = list_run_artifacts(tmp_path, zones=("DE",))

    assert len(artifacts) == 1
    assert artifacts[0].kind == "challenger prospectif"
    assert artifacts[0].directory == run


def test_artifact_listing_exposes_explicit_sealed_benchmark(tmp_path: Path) -> None:
    live_root = tmp_path / "live"
    live_root.mkdir()
    benchmark = tmp_path / "chronos2_hourly_de_sealed_benchmark_v1"
    benchmark.mkdir()
    (benchmark / "run_manifest.json").write_text(
        json.dumps({"zone": "DE"}), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "delivery_start_utc": ["2026-08-01T00:00:00Z"],
            "actual": [1.0],
            "residual_corrected__q50": [1.5],
        }
    ).to_csv(
        benchmark / "statistics_history_hourly.csv.gz",
        index=False,
        compression="gzip",
    )
    (benchmark / "de.html").write_text("<html></html>", encoding="utf-8")

    artifacts = list_run_artifacts(
        live_root,
        zones=("DE",),
        sealed_benchmark_root=tmp_path,
    )

    assert len(artifacts) == 1
    assert artifacts[0].kind == "benchmark scellé"
    assert artifacts[0].statistics_path == (
        benchmark / "statistics_history_hourly.csv.gz"
    )


def test_statistics_load_requires_official_storm_audit_and_computes_all_win_rates(
    tmp_path: Path,
) -> None:
    hours = pd.date_range("2026-08-01", periods=48, freq="h", tz="UTC")
    actual = np.r_[np.full(24, 10.0), np.full(24, 20.0)]
    frame = pd.DataFrame(
        {
            "delivery_start_utc": hours,
            "actual": actual,
            "mkonline_blend__q50": actual + np.r_[np.ones(24), np.full(24, 3.0)],
            "storm_evaluation_only__q50": actual + 4.0,
            "storm_dashboard_official__q50": actual + 2.0,
        }
    )
    path = tmp_path / "statistics_history_hourly.csv.gz"
    frame.to_csv(path, index=False, compression="gzip")

    unaudited = load_statistics_history(path)
    assert unaudited.benchmark_column == "storm_evaluation_only__q50"
    (tmp_path / "statistics_history_audit.json").write_text(
        json.dumps(
            {
                "storm_primary_report_benchmark": "storm_dashboard_official__q50",
                "report_scope_note": "Historique apparié.",
            }
        ),
        encoding="utf-8",
    )
    audited = load_statistics_history(path)
    assert audited.benchmark_column == "storm_dashboard_official__q50"
    assert audited.benchmark_label == "Storm officiel dashboard"
    assert audited.scope_note == "Historique apparié."

    view = build_statistics_view(
        audited,
        timezone_name="Europe/Paris",
        sample="daily",
    )
    assert set(view.summary["metric"]) == {
        "mae",
        "rmse",
        "mape",
        "explained_variance",
        "r2",
        "std_error",
        "correlation",
    }
    mae = view.summary.set_index("metric").loc["mae"]
    assert mae["comparable_periods"] == 3  # local-time split creates 3 days
    assert 0.0 <= mae["win_rate"] <= 1.0
    assert len(view.time_series) == 48


def test_best_statistics_history_prefers_freshest_scored_replay(
    tmp_path: Path,
) -> None:
    status, live_archive = _write_existing_live_archive(
        tmp_path,
        day="2026-08-20",
    )
    _add_statistics_history(
        live_archive,
        timezone_name=status.timezone,
        history_end_local="2026-08-15",
    )
    _same_status, replay_source = _write_existing_live_archive(
        tmp_path,
        day="2026-08-19",
    )
    replay_archive = _convert_archive_to_replay(replay_source)
    _add_statistics_history(
        replay_archive,
        timezone_name=status.timezone,
        history_end_local="2026-08-18",
    )

    artifact, dataset = load_best_statistics_history(
        status,
        project_root=tmp_path,
    )

    assert isinstance(artifact, PerformanceArtifact)
    assert artifact.archive_kind == "pit_replay"
    assert artifact.delivery_day == "2026-08-19"
    assert artifact.archive_path == replay_archive.resolve()
    assert artifact.statistics_prefix_end_local == "2026-08-18"
    assert artifact.n_total_statistics_hours == 24
    assert dataset.benchmark_column == "storm_dashboard_official__q50"


def test_best_statistics_history_uses_timestamp_fallback_and_allows_es_without_storm(
    tmp_path: Path,
) -> None:
    status, archive = _write_existing_live_archive(
        tmp_path,
        day="2026-08-19",
        zone="ES",
        timezone_name="Europe/Madrid",
    )
    _add_statistics_history(
        archive,
        timezone_name=status.timezone,
        history_end_local="2026-08-18",
        include_prefix=False,
        include_storm=False,
    )

    artifact, dataset = load_best_statistics_history(
        status,
        project_root=tmp_path,
        variant="autonomous",
    )

    assert artifact.zone == "ES"
    assert artifact.statistics_prefix_end_local == "2026-08-18"
    assert dataset.candidate_column == "residual_corrected__q50"
    assert dataset.benchmark_column is None
    assert dataset.benchmark_label is None
    assert dataset.frame["benchmark"].isna().all()


def test_best_statistics_history_rejects_fresher_bad_internal_checksum(
    tmp_path: Path,
) -> None:
    status, live_archive = _write_existing_live_archive(
        tmp_path,
        day="2026-08-20",
    )
    _add_statistics_history(
        live_archive,
        timezone_name=status.timezone,
        history_end_local="2026-08-15",
    )
    _same_status, replay_source = _write_existing_live_archive(
        tmp_path,
        day="2026-08-19",
    )
    replay_archive = _convert_archive_to_replay(replay_source)
    _add_statistics_history(
        replay_archive,
        timezone_name=status.timezone,
        history_end_local="2026-08-18",
        internal_sha="0" * 64,
    )

    with pytest.raises(
        ExistingForecastArchiveError,
        match="statistics_history_sha256 divergent",
    ):
        load_best_statistics_history(status, project_root=tmp_path)


def test_mape_uses_absolute_actual_and_excludes_only_near_zero_values() -> None:
    timestamps = pd.date_range("2026-08-01", periods=3, freq="h", tz="UTC")
    dataset = StatisticsDataset(
        path=Path("stats.csv.gz"),
        frame=pd.DataFrame(
            {
                "timestamp": timestamps,
                "actual": [-10.0, 0.0, 20.0],
                "candidate": [-8.0, 999.0, 18.0],
                "benchmark": [-5.0, 500.0, 16.0],
            }
        ),
        candidate_column="candidate",
        candidate_label="Candidate",
        benchmark_column="benchmark",
        benchmark_label="Storm",
        scope_note=None,
    )

    view = build_statistics_view(dataset, timezone_name="UTC", sample="daily")
    mape = view.summary.set_index("metric").loc["mape"]

    assert mape["candidate"] == pytest.approx(15.0)
    assert mape["benchmark"] == pytest.approx(35.0)
    assert mape["wins"] == 1
    assert mape["win_rate"] == 1.0


def test_forecast_curve_is_causal_ordered_and_renders_p10_p90_band(
    tmp_path: Path,
) -> None:
    delivery = pd.date_range("2026-08-14", periods=24, freq="h", tz="UTC")
    path = tmp_path / "forecast_hourly_de.csv"
    pd.DataFrame(
        {
            "delivery_start_utc": delivery,
            "forecast_origin_utc": delivery - pd.Timedelta(days=1),
            "q10": np.arange(24, dtype=float),
            "q50": np.arange(24, dtype=float) + 1.0,
            "q90": np.arange(24, dtype=float) + 2.0,
        }
    ).to_csv(path, index=False)

    dataset = load_forecast_curve(path, timezone_name="Europe/Berlin")
    chart = _forecast_chart(dataset.frame).to_dict()

    assert list(dataset.frame.columns) == ["timestamp", "P10", "P50", "P90"]
    assert len(dataset.frame) == 24
    assert str(dataset.frame["timestamp"].dt.tz) == "Europe/Berlin"
    assert len(chart["layer"]) == 2
    assert chart["layer"][0]["mark"]["type"] == "area"
    assert chart["layer"][1]["mark"]["type"] == "line"

    crossed = pd.read_csv(path)
    crossed.loc[0, "q10"] = crossed.loc[0, "q90"] + 1.0
    crossed.to_csv(path, index=False)
    with pytest.raises(ValueError, match="quantiles croisés"):
        load_forecast_curve(path, timezone_name="Europe/Berlin")


def test_forecast_curve_switches_between_autonomous_and_mkonline_without_mutation(
    tmp_path: Path,
) -> None:
    delivery = pd.date_range("2026-08-20", periods=24, freq="h", tz="UTC")
    path = tmp_path / "forecast_hourly_fr.csv"
    frame = pd.DataFrame(
        {
            "delivery_start_utc": delivery,
            "forecast_origin_utc": delivery - pd.Timedelta(days=1),
            "q10": np.full(24, 90.0),
            "q50": np.full(24, 100.0),
            "q90": np.full(24, 110.0),
            "residual_corrected__q10": np.full(24, 40.0),
            "residual_corrected__q50": np.full(24, 50.0),
            "residual_corrected__q90": np.full(24, 60.0),
            "mkonline_blend__q10": np.full(24, 90.0),
            "mkonline_blend__q50": np.full(24, 100.0),
            "mkonline_blend__q90": np.full(24, 110.0),
        }
    )
    frame.to_csv(path, index=False)
    before = path.read_bytes()

    autonomous = load_forecast_curve(
        path,
        timezone_name="Europe/Paris",
        variant="autonomous",
    )
    blend = load_forecast_curve(
        path,
        timezone_name="Europe/Paris",
        variant="mkonline_blend",
    )

    assert autonomous.frame["P50"].tolist() == [50.0] * 24
    assert blend.frame["P50"].tolist() == [100.0] * 24
    assert path.read_bytes() == before

    with pytest.raises(ValueError, match="variant"):
        load_forecast_curve(path, timezone_name="Europe/Paris", variant="Storm")


def test_statistics_view_rejects_unknown_sample() -> None:
    dataset = StatisticsDataset(
        path=Path("stats.csv.gz"),
        frame=pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-08-01T00:00:00Z"]),
                "actual": [1.0],
                "candidate": [1.0],
                "benchmark": [2.0],
            }
        ),
        candidate_column="candidate",
        candidate_label="Candidate",
        benchmark_column="benchmark",
        benchmark_label="Storm",
        scope_note=None,
    )
    with pytest.raises(ValueError, match="Échantillonnage inconnu"):
        build_statistics_view(dataset, timezone_name="UTC", sample="hourly")
