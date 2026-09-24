from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_residual_load_source_comparison as comparison
from chronos2_hourly.app_service import ZoneStatus
from chronos2_hourly.hourly_contract import local_delivery_day_index


def _spec(tmp_path: Path, *, zone: str = "FR") -> comparison.ZoneComparisonSpec:
    timezone = {
        "FR": "Europe/Paris",
        "DE": "Europe/Berlin",
        "BE": "Europe/Brussels",
        "NL": "Europe/Amsterdam",
        "ES": "Europe/Madrid",
    }[zone]
    live_config = tmp_path / f"{zone.lower()}_live.yaml"
    live_config.write_text("live: {}\n", encoding="utf-8")
    output_root = tmp_path / "runs" / "live" / zone.lower()
    output_root.mkdir(parents=True)
    status = ZoneStatus(
        code=zone,
        timezone=timezone,
        enabled=True,
        production_ready=True,
        ready=True,
        runner=None,
        live_config=live_config,
        checks=(),
        blockers=(),
    )
    return comparison.ZoneComparisonSpec(
        code=zone,
        timezone=timezone,
        project_root=tmp_path,
        live_config=live_config,
        output_root=output_root,
        forecast_filename=f"forecast_hourly_{zone.lower()}.csv",
        status=status,
    )


def _manifest(
    spec: comparison.ZoneComparisonSpec,
    delivery_day: str,
    *,
    source: str,
) -> dict[str, object]:
    common: dict[str, object] = {
        "script_version": "test-frozen-v1",
        "config": str(spec.live_config),
        "model_id": f"chronos2_{spec.code.lower()}_frozen",
        "zone": spec.code,
        "timezone": spec.timezone,
        "delivery_day_local": delivery_day,
        "target_contract": "hourly_utc_no_interpolation",
        "delivery_horizon": "dynamic_23_24_25",
        "active_features": ["same_feature_a", "same_feature_b"],
        "candidate_model": "residual_corrected",
        "prediction_mode": "autonomous_only",
    }
    if source == "saturn":
        common.update(
            {
                "run_type": "live_day_ahead",
                "forecast_status": "issued_live",
            }
        )
    else:
        common.update(
            {
                "residual_load_source": "chronos2",
                "run_type": "shadow_live_day_ahead",
                "forecast_status": "shadow_challenger",
                "production_eligible": False,
            }
        )
    return common


def _write_pair(
    spec: comparison.ZoneComparisonSpec,
    delivery_day: str,
    *,
    production_offset: float = 2.0,
    challenger_offset: float = 1.0,
    challenger_timeline: pd.DatetimeIndex | None = None,
) -> tuple[pd.Series, Path, Path]:
    expected = local_delivery_day_index(delivery_day, timezone=spec.timezone)
    actual = pd.Series(
        np.linspace(20.0, 80.0, len(expected)),
        index=expected,
        name="actual",
    )
    production = (
        spec.output_root
        / f"{spec.code.lower()}_day_ahead_{delivery_day}"
    )
    challenger = (
        spec.output_root
        / "_challengers"
        / "residual_load_chronos2"
        / (
            f"{spec.code.lower()}_day_ahead_{delivery_day}"
            "_residual_load_chronos2"
        )
    )
    production.mkdir()
    challenger.mkdir(parents=True)
    (production / "run_manifest.json").write_text(
        json.dumps(_manifest(spec, delivery_day, source="saturn")),
        encoding="utf-8",
    )
    (challenger / "run_manifest.json").write_text(
        json.dumps(_manifest(spec, delivery_day, source="chronos2")),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "delivery_start_utc": expected.astype(str),
            "q50": actual.to_numpy() + production_offset,
        }
    ).to_csv(production / spec.forecast_filename, index=False)
    selected_challenger_timeline = (
        challenger_timeline if challenger_timeline is not None else expected
    )
    pd.DataFrame(
        {
            "delivery_start_utc": selected_challenger_timeline.astype(str),
            "candidate_model__q50": (
                actual.iloc[: len(selected_challenger_timeline)].to_numpy()
                + challenger_offset
            ),
        }
    ).to_csv(challenger / spec.forecast_filename, index=False)
    historical_index = pd.DatetimeIndex(
        [expected[0] - pd.Timedelta(hours=1)]
    )
    production_inputs = pd.DataFrame(
        {
            "timestamp": historical_index.astype(str),
            "target": [10.0],
        }
    )
    challenger_inputs = production_inputs.copy()
    for position, feature in enumerate(
        comparison.RESIDUAL_LOAD_TREATMENT_FEATURES,
        start=1,
    ):
        production_inputs[feature] = [float(position)]
        challenger_inputs[feature] = [float(position)]
    covariate_index = historical_index.append(expected)
    production_covariates = pd.DataFrame(
        {
            "timestamp": covariate_index.astype(str),
            "known_hour_sin": np.sin(np.arange(len(covariate_index))),
        }
    )
    challenger_covariates = production_covariates.copy()
    for position, feature in enumerate(
        comparison.RESIDUAL_LOAD_TREATMENT_FEATURES,
        start=1,
    ):
        production_values = [
            float(position),
            *([100.0 + position] * len(expected)),
        ]
        challenger_values = [
            float(position),
            *([200.0 + position] * len(expected)),
        ]
        production_covariates[feature] = production_values
        challenger_covariates[feature] = challenger_values
        production_covariates[f"known_{feature}_oracle"] = production_values
        challenger_covariates[f"known_{feature}_oracle"] = challenger_values
    for archive, inputs in (
        (production, production_inputs),
        (challenger, challenger_inputs),
    ):
        inputs_dir = archive / "inputs"
        inputs_dir.mkdir()
        inputs.to_csv(
            inputs_dir / "aligned_inputs.csv.gz",
            index=False,
            compression="gzip",
        )
        covariates = (
            production_covariates
            if archive == production
            else challenger_covariates
        )
        covariates.to_csv(
            inputs_dir / "model_covariates_with_future.csv.gz",
            index=False,
            compression="gzip",
        )
    common_sources = [
        ("live_config", spec.live_config.resolve().as_posix(), "1" * 64),
        (
            "base_config",
            (spec.project_root / "base.yaml").resolve().as_posix(),
            "2" * 64,
        ),
        (
            "frozen_recipe",
            (spec.project_root / "recipe.json").resolve().as_posix(),
            "3" * 64,
        ),
        (
            "frozen_autonomous_checksum_manifest",
            (spec.project_root / "frozen" / "artifact_checksums.json")
            .resolve()
            .as_posix(),
            "4" * 64,
        ),
        (
            "sealed_benchmark_checksum_manifest",
            (spec.project_root / "benchmark" / "artifact_checksums.json")
            .resolve()
            .as_posix(),
            "5" * 64,
        ),
        (
            "source_code",
            "run_mkonline_live_hourly.py",
            "6" * 64,
        ),
    ]
    source_entries = [
        {
            "role": role,
            "path": path,
            "size_bytes": position + 1,
            "sha256": sha256,
        }
        for position, (role, path, sha256) in enumerate(common_sources)
    ]
    (production / "artifact_checksums.json").write_text(
        json.dumps(
            {
                "algorithm": "sha256",
                "artifacts": source_entries,
            }
        ),
        encoding="utf-8",
    )
    (challenger / "artifact_checksums.json").write_text(
        json.dumps(
            {
                "algorithm": "sha256",
                "artifacts": [
                    *source_entries,
                    {
                        "role": "residual_load_bundle_manifest",
                        "path": "upstream/manifest.json",
                        "size_bytes": 99,
                        "sha256": "f" * 64,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return actual, production, challenger


def _validator(
    spec: comparison.ZoneComparisonSpec,
    delivery_day: date,
    source: str,
) -> Path:
    archive_name = f"{spec.code.lower()}_day_ahead_{delivery_day.isoformat()}"
    if source == "chronos2":
        return (
            spec.output_root
            / "_challengers"
            / "residual_load_chronos2"
            / f"{archive_name}_residual_load_chronos2"
        ).resolve()
    return (spec.output_root / archive_name).resolve()


def test_scan_uses_dedicated_challenger_root_and_ignores_legacy_sibling(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    _, production_path, challenger_path = _write_pair(spec, "2026-08-20")
    legacy_sibling = (
        spec.output_root
        / "fr_day_ahead_2026-08-21_residual_load_chronos2"
    )
    legacy_sibling.mkdir()

    production, challenger = comparison._scan_archive_days(
        spec,
        start=date(2026, 8, 20),
        end=date(2026, 8, 21),
    )

    assert production == {date(2026, 8, 20): production_path.resolve()}
    assert challenger == {date(2026, 8, 20): challenger_path.resolve()}
    assert challenger_path.parent == (
        spec.output_root / "_challengers" / "residual_load_chronos2"
    ).resolve()


@pytest.mark.parametrize(
    ("delivery_day", "expected_hours"),
    [("2026-03-29", 23), ("2026-10-25", 25)],
)
def test_prospective_pair_scores_complete_dst_day(
    tmp_path: Path,
    delivery_day: str,
    expected_hours: int,
) -> None:
    spec = _spec(tmp_path)
    actual, _, _ = _write_pair(spec, delivery_day)

    result = comparison.build_comparison(
        [spec],
        start=delivery_day,
        end=delivery_day,
        actual_loader=lambda _spec: actual,
        archive_validator=_validator,
    )

    assert len(result.paired_hourly) == expected_hours
    assert result.paired_hourly["hours_in_local_day"].eq(expected_hours).all()
    assert result.paired_hourly["production_q50_column"].eq("q50").all()
    assert result.paired_hourly["challenger_q50_column"].eq(
        "candidate_model__q50"
    ).all()
    zone_metrics = result.metrics.query("zone == 'FR'").iloc[0]
    assert zone_metrics["production_mae"] == pytest.approx(2.0)
    assert zone_metrics["challenger_mae"] == pytest.approx(1.0)
    assert zone_metrics["challenger_win_rate"] == pytest.approx(1.0)
    assert result.manifest["comparison_type"] == "prospective_paired_ab"
    assert result.manifest["downstream_policy"] == (
        "frozen_current_production_downstream"
    )
    assert "distribution shift" in result.manifest["interpretation_warning"]


@pytest.mark.parametrize(
    ("archive_kind", "field", "value", "message"),
    [
        ("production", "residual_load_source", "chronos2", "production"),
        ("challenger", "run_type", "live_day_ahead", "run_type"),
        ("challenger", "production_eligible", True, "production_eligible=false"),
        ("challenger", "model_id", "different-downstream", "Downstream non gelé"),
    ],
)
def test_comparison_refuses_invalid_identity_or_downstream(
    tmp_path: Path,
    archive_kind: str,
    field: str,
    value: object,
    message: str,
) -> None:
    spec = _spec(tmp_path)
    actual, production, challenger = _write_pair(spec, "2026-08-20")
    archive = production if archive_kind == "production" else challenger
    manifest_path = archive / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(comparison.ComparisonError, match=message):
        comparison.build_comparison(
            [spec],
            start="2026-08-20",
            end="2026-08-20",
            actual_loader=lambda _spec: actual,
            archive_validator=_validator,
        )


def test_comparison_refuses_different_or_incomplete_dst_timeline(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    expected = local_delivery_day_index("2026-10-25", timezone=spec.timezone)
    actual, _, _ = _write_pair(
        spec,
        "2026-10-25",
        challenger_timeline=expected[:-1],
    )

    with pytest.raises(comparison.ComparisonError, match="timeline"):
        comparison.build_comparison(
            [spec],
            start="2026-10-25",
            end="2026-10-25",
            actual_loader=lambda _spec: actual,
            archive_validator=_validator,
        )


def test_comparison_refuses_a_downstream_source_sha_divergence(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    actual, _, challenger = _write_pair(spec, "2026-08-20")
    checksum_path = challenger / "artifact_checksums.json"
    payload = json.loads(checksum_path.read_text(encoding="utf-8"))
    base_config = next(
        item for item in payload["artifacts"] if item["role"] == "base_config"
    )
    base_config["sha256"] = "a" * 64
    checksum_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(comparison.ComparisonError, match="Sources downstream non gelées"):
        comparison.build_comparison(
            [spec],
            start="2026-08-20",
            end="2026-08-20",
            actual_loader=lambda _spec: actual,
            archive_validator=_validator,
        )


def test_comparison_refuses_a_non_treatment_input_divergence(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    actual, _, challenger = _write_pair(spec, "2026-08-20")
    inputs_path = challenger / "inputs" / "aligned_inputs.csv.gz"
    inputs = pd.read_csv(inputs_path)
    inputs.loc[0, "target"] = 999.0
    inputs.to_csv(inputs_path, index=False, compression="gzip")

    with pytest.raises(comparison.ComparisonError, match="hors traitement"):
        comparison.build_comparison(
            [spec],
            start="2026-08-20",
            end="2026-08-20",
            actual_loader=lambda _spec: actual,
            archive_validator=_validator,
        )


def test_incomplete_actual_day_is_audited_but_not_scored(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    actual, _, _ = _write_pair(spec, "2026-08-20")
    actual.iloc[-1] = np.nan

    result = comparison.build_comparison(
        [spec],
        start="2026-08-20",
        end="2026-08-20",
        actual_loader=lambda _spec: actual,
        archive_validator=_validator,
    )

    assert result.paired_hourly.empty
    assert result.metrics.query("zone == 'FR'")["n_days"].item() == 0
    assert result.manifest["n_discovered_pairs"] == 1
    assert result.manifest["n_scored_pairs"] == 0
    assert result.manifest["pairs"][0]["realized_complete"] is False


def test_publication_is_immutable_and_manifest_seals_csv_files(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    actual, _, _ = _write_pair(spec, "2026-08-20")
    build = comparison.build_comparison(
        [spec],
        start="2026-08-20",
        end="2026-08-20",
        actual_loader=lambda _spec: comparison.ActualSeries(
            actual,
            source="test_audited_actuals",
        ),
        archive_validator=_validator,
    )
    output = tmp_path / "comparison"

    published = comparison.publish_comparison(build, output_dir=output)

    assert published == output.resolve()
    paired_path = output / "paired_hourly.csv.gz"
    metrics_path = output / "metrics.csv"
    report_path = output / "comparison_report.html"
    manifest = json.loads(
        (output / "comparison_manifest.json").read_text(encoding="utf-8")
    )
    artifacts = {item["path"]: item for item in manifest["artifacts"]}
    for path in (paired_path, metrics_path, report_path):
        assert artifacts[path.name]["sha256"] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    assert len(pd.read_csv(paired_path)) == 24
    assert set(pd.read_csv(metrics_path)["zone"]) == {"FR", "ALL"}
    rendered = report_path.read_text(encoding="utf-8")
    assert "Comparaison prospective · charge résiduelle" in rendered
    assert "production avec prévisions de charge résiduelle Saturn" in rendered
    assert "challenger avec prévisions de charge résiduelle Chronos-2" in rendered
    with pytest.raises(comparison.ComparisonError, match="existe déjà"):
        comparison.publish_comparison(build, output_dir=output)


def test_cli_accepts_all_requested_boundaries() -> None:
    args = comparison.parse_args(
        [
            "--zones",
            "FR,DE",
            "BE",
            "--start",
            "2026-08-01",
            "--end",
            "2026-08-31",
            "--registry",
            "zones.yaml",
            "--output-dir",
            "comparison",
        ]
    )

    assert comparison.normalize_zones(args.zones) == ("FR", "DE", "BE")
    assert args.start == "2026-08-01"
    assert args.end == "2026-08-31"
    assert args.registry == "zones.yaml"
    assert args.output_dir == "comparison"


def test_run_comparison_scans_the_registry_output_root(tmp_path: Path) -> None:
    live_config = tmp_path / "fr_live.yaml"
    live_config.write_text(
        "live:\n"
        "  output_root: runs/live/fr\n"
        "  forecast_filename: forecast_hourly_fr.csv\n",
        encoding="utf-8",
    )
    registry = tmp_path / "zones.yaml"
    registry.write_text(
        "schema_version: 1\n"
        "zones:\n"
        "  FR:\n"
        "    enabled: true\n"
        "    production_ready: true\n"
        "    delivery_timezone: Europe/Paris\n"
        "    live_config: fr_live.yaml\n",
        encoding="utf-8",
    )
    spec = comparison.load_zone_specs(registry, zones=["FR"])[0]
    spec.output_root.mkdir(parents=True)
    actual, _, _ = _write_pair(spec, "2026-08-20")
    output = tmp_path / "published"

    result = comparison.run_comparison(
        registry_path=registry,
        zones=["FR"],
        start="2026-08-20",
        end="2026-08-20",
        output_dir=output,
        actual_loader=lambda _spec: actual,
        archive_validator=_validator,
    )

    assert result == output.resolve()
    manifest = json.loads(
        (output / "comparison_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["registry_path"] == str(registry.resolve())
    assert manifest["registry_sha256"] == hashlib.sha256(
        registry.read_bytes()
    ).hexdigest()
    assert manifest["n_scored_pairs"] == 1
