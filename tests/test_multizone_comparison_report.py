from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app_multizone import _comparison_chart
from chronos2_hourly.app_service import (
    ExistingForecastArchiveError,
    MixedForecastDeliveryDaysError,
    ZoneStatus,
    load_latest_forecast_comparison,
)
from chronos2_hourly.consolidated_report import (
    consolidated_report_filename,
    render_consolidated_forecast_report,
)
from chronos2_hourly.hourly_contract import local_delivery_day_index


def _seal_archive(archive: Path) -> None:
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


def _write_live_archive(
    tmp_path: Path,
    *,
    zone: str,
    timezone_name: str,
    day: str,
    level: float,
    manifest_zone: str | None = None,
    storm_used_as_feature: bool = False,
) -> tuple[ZoneStatus, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    zone_lower = zone.lower()
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
    delivery = local_delivery_day_index(
        pd.Timestamp(day).date(), timezone=timezone_name
    )
    hours = np.arange(len(delivery), dtype=float)
    pd.DataFrame(
        {
            "delivery_start_utc": delivery,
            "forecast_origin_utc": delivery[0] - pd.Timedelta(hours=16),
            "q10": level + hours - 2.0,
            "q50": level + hours,
            "q90": level + hours + 2.0,
        }
    ).to_csv(archive / f"forecast_hourly_{zone_lower}.csv", index=False)
    (archive / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_type": "live_day_ahead",
                "forecast_status": "issued_live",
                "zone": manifest_zone or zone,
                "timezone": timezone_name,
                "delivery_day_local": day,
                "forecast_path": f"forecast_hourly_{zone_lower}.csv",
                "sha256_manifest": "artifact_checksums.json",
                "prediction_inputs": ["autonomous_extended_residual"],
                "storm_used_as_feature": storm_used_as_feature,
                "storm_loaded_for_prediction": False,
            }
        ),
        encoding="utf-8",
    )
    (archive / "live_run_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "run_type": "live_day_ahead",
                "zone": zone,
                "delivery_day_local": day,
                "hours": len(delivery),
                "forecast_path": f"forecast_hourly_{zone_lower}.csv",
            }
        ),
        encoding="utf-8",
    )
    (archive / "report.html").write_text("<html>source</html>", encoding="utf-8")
    _seal_archive(archive)
    return (
        ZoneStatus(
            code=zone,
            timezone=timezone_name,
            enabled=True,
            production_ready=True,
            ready=True,
            runner=tmp_path / "runner.py",
            live_config=config,
            checks=("ok",),
            blockers=(),
        ),
        archive,
    )


def test_same_day_comparison_and_standalone_report_are_audited(
    tmp_path: Path,
) -> None:
    de, _de_archive = _write_live_archive(
        tmp_path,
        zone="DE",
        timezone_name="Europe/Berlin",
        day="2026-08-15",
        level=50.0,
    )
    be, _be_archive = _write_live_archive(
        tmp_path,
        zone="BE",
        timezone_name="Europe/Brussels",
        day="2026-08-15",
        level=60.0,
    )

    comparison = load_latest_forecast_comparison(
        [de, be],
        project_root=tmp_path,
        zones=("DE", "BE"),
    )

    assert comparison.zones == ("DE", "BE")
    assert comparison.delivery_days == {"DE": "2026-08-15", "BE": "2026-08-15"}
    assert comparison.timeline_aligned
    assert not comparison.mixed_delivery_days
    assert len(comparison.frame) == 48
    assert "storm" not in " ".join(comparison.frame.columns).lower()
    assert comparison.frame.groupby("zone")["timestamp_utc"].nunique().to_dict() == {
        "BE": 24,
        "DE": 24,
    }

    line_spec = _comparison_chart(
        comparison.frame,
        include_intervals=False,
        mixed_delivery_days=False,
    ).to_dict()
    band_spec = _comparison_chart(
        comparison.frame,
        include_intervals=True,
        mixed_delivery_days=False,
    ).to_dict()
    assert line_spec["mark"]["type"] == "line"
    assert len(band_spec["layer"]) == 2
    assert band_spec["layer"][0]["mark"]["type"] == "area"

    plain_html = render_consolidated_forecast_report(
        comparison, include_intervals=False
    ).decode("utf-8")
    band_html = render_consolidated_forecast_report(
        comparison, include_intervals=True
    ).decode("utf-8")
    assert "DE=2026-08-15" in plain_html
    assert "BE=2026-08-15" in plain_html
    assert comparison.archives[0].forecast_sha256 in plain_html
    assert "plotly.js" in plain_html.lower()
    assert "<script src=" not in plain_html.lower()
    assert "connect-src 'none'" in plain_html.lower()
    assert 'id="forecast-comparison-chart"' in plain_html
    assert 'id="statistics-summary-table"' in plain_html
    assert '"initial_include_intervals":false' in plain_html
    assert '"initial_include_intervals":true' in band_html
    assert consolidated_report_filename(comparison) == (
        "chronos2_forecasts_de-be_2026-08-15.html"
    )


def test_mixed_latest_days_require_explicit_opt_in_and_remain_visible(
    tmp_path: Path,
) -> None:
    de, _ = _write_live_archive(
        tmp_path,
        zone="DE",
        timezone_name="Europe/Berlin",
        day="2026-08-15",
        level=50.0,
    )
    be, _ = _write_live_archive(
        tmp_path,
        zone="BE",
        timezone_name="Europe/Brussels",
        day="2026-08-14",
        level=60.0,
    )

    with pytest.raises(MixedForecastDeliveryDaysError) as captured:
        load_latest_forecast_comparison(
            [de, be], project_root=tmp_path, zones=("DE", "BE")
        )
    assert captured.value.delivery_days == {
        "DE": "2026-08-15",
        "BE": "2026-08-14",
    }

    comparison = load_latest_forecast_comparison(
        [de, be],
        project_root=tmp_path,
        zones=("DE", "BE"),
        allow_mixed_delivery_days=True,
    )
    assert comparison.mixed_delivery_days
    assert not comparison.timeline_aligned
    html = render_consolidated_forecast_report(comparison).decode("utf-8")
    assert "Dates de livraison différentes" in html
    assert "Comparaison explicite" in html


@pytest.mark.parametrize(
    ("day", "expected_hours"),
    (("2026-03-29", 23), ("2026-10-25", 25)),
)
def test_comparison_preserves_dst_delivery_timelines(
    tmp_path: Path,
    day: str,
    expected_hours: int,
) -> None:
    de, _ = _write_live_archive(
        tmp_path,
        zone="DE",
        timezone_name="Europe/Berlin",
        day=day,
        level=50.0,
    )
    be, _ = _write_live_archive(
        tmp_path,
        zone="BE",
        timezone_name="Europe/Brussels",
        day=day,
        level=60.0,
    )

    comparison = load_latest_forecast_comparison(
        [de, be], project_root=tmp_path, zones=("DE", "BE")
    )

    assert comparison.timeline_aligned
    assert comparison.frame.groupby("zone").size().to_dict() == {
        "BE": expected_hours,
        "DE": expected_hours,
    }


def test_newest_invalid_archive_is_not_silently_replaced_by_older_one(
    tmp_path: Path,
) -> None:
    status, _ = _write_live_archive(
        tmp_path,
        zone="DE",
        timezone_name="Europe/Berlin",
        day="2026-08-14",
        level=40.0,
    )
    status, newest = _write_live_archive(
        tmp_path,
        zone="DE",
        timezone_name="Europe/Berlin",
        day="2026-08-15",
        level=50.0,
    )
    with (newest / "forecast_hourly_de.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(ExistingForecastArchiveError, match="divergente"):
        load_latest_forecast_comparison(
            [status], project_root=tmp_path, zones=("DE",)
        )


def test_comparison_refuses_cross_zone_storm_and_cross_timezone_timeline(
    tmp_path: Path,
) -> None:
    wrong_zone, _ = _write_live_archive(
        tmp_path,
        zone="BE",
        timezone_name="Europe/Brussels",
        day="2026-08-15",
        level=60.0,
        manifest_zone="DE",
    )
    with pytest.raises(ExistingForecastArchiveError, match="run_manifest.zone"):
        load_latest_forecast_comparison(
            [wrong_zone], project_root=tmp_path, zones=("BE",)
        )

    storm, _ = _write_live_archive(
        tmp_path / "storm_case",
        zone="DE",
        timezone_name="Europe/Berlin",
        day="2026-08-15",
        level=50.0,
        storm_used_as_feature=True,
    )
    with pytest.raises(ExistingForecastArchiveError, match="storm_used_as_feature"):
        load_latest_forecast_comparison(
            [storm], project_root=tmp_path / "storm_case", zones=("DE",)
        )

    timeline_root = tmp_path / "timeline_case"
    de, _ = _write_live_archive(
        timeline_root,
        zone="DE",
        timezone_name="Europe/Berlin",
        day="2026-08-15",
        level=50.0,
    )
    gb, _ = _write_live_archive(
        timeline_root,
        zone="GB",
        timezone_name="Europe/London",
        day="2026-08-15",
        level=70.0,
    )
    with pytest.raises(ExistingForecastArchiveError, match="timelines UTC"):
        load_latest_forecast_comparison(
            [de, gb], project_root=timeline_root, zones=("DE", "GB")
        )


def test_report_rehashes_archives_immediately_before_export(tmp_path: Path) -> None:
    status, archive = _write_live_archive(
        tmp_path,
        zone="DE",
        timezone_name="Europe/Berlin",
        day="2026-08-15",
        level=50.0,
    )
    comparison = load_latest_forecast_comparison(
        [status], project_root=tmp_path, zones=("DE",)
    )
    (archive / "report.html").write_text("<html>tampered</html>", encoding="utf-8")

    with pytest.raises(ValueError, match="artefact modifie"):
        render_consolidated_forecast_report(comparison)
