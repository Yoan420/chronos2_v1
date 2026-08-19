from __future__ import annotations

from collections import defaultdict
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.app_service import (
    APP_ZONES,
    ZoneStatus,
    load_latest_forecast_comparison,
)
from chronos2_hourly.consolidated_report import (
    render_consolidated_forecast_report,
)
from chronos2_hourly.hourly_contract import local_delivery_day_index


ZONE_TIMEZONES = {
    "FR": "Europe/Paris",
    "DE": "Europe/Berlin",
    "BE": "Europe/Brussels",
    "NL": "Europe/Amsterdam",
    "ES": "Europe/Madrid",
}
REQUIRED_METRICS = {
    "mae",
    "rmse",
    "mape",
    "explained_variance",
    "r2",
    "std_error",
    "correlation",
}
OPTIONAL_METRICS = {"bias"}
SAMPLES = {"daily", "weekly", "monthly"}


class _ReportDocument(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.elements: list[tuple[str, dict[str, str | None]]] = []
        self.by_id: dict[str, tuple[str, dict[str, str | None]]] = {}
        self.script_text: dict[str, list[str]] = defaultdict(list)
        self._script_id: str | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        self.elements.append((tag, attributes))
        element_id = attributes.get("id")
        if element_id:
            self.by_id[element_id] = (tag, attributes)
        if tag == "script":
            self._script_id = element_id

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._script_id = None

    def handle_data(self, data: str) -> None:
        if self._script_id is not None:
            self.script_text[self._script_id].append(data)


def _parse_document(rendered: bytes) -> tuple[str, _ReportDocument, dict[str, Any]]:
    html = rendered.decode("utf-8")
    document = _ReportDocument()
    document.feed(html)
    data_tag, data_attributes = document.by_id["consolidated-report-data"]
    assert data_tag == "script"
    assert data_attributes.get("type") == "application/json"
    payload = json.loads(
        "".join(document.script_text["consolidated-report-data"])
    )
    return html, document, payload


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


def _statistics_frame(
    *,
    timezone_name: str,
    benchmark_mode: str,
    duplicate_timestamp: bool,
) -> pd.DataFrame:
    delivery = pd.DatetimeIndex([], tz="UTC")
    for value in pd.date_range("2026-06-01", periods=40, freq="D"):
        delivery = delivery.append(
            local_delivery_day_index(value.date(), timezone=timezone_name)
        )
    position = np.arange(len(delivery), dtype=float)
    day_number = np.floor_divide(position.astype(int), 24)
    actual = 55.0 + 8.0 * np.sin(position / 19.0) + 0.015 * position
    candidate_error = np.where(day_number % 3 == 0, 1.0, 2.8)
    candidate_error = candidate_error + 0.35 * np.sin(position / 7.0)
    benchmark_error = np.where(day_number % 2 == 0, 2.0, 2.4)
    benchmark_error = benchmark_error - 0.25 * np.cos(position / 9.0)
    frame = pd.DataFrame(
        {
            "delivery_start_utc": delivery,
            "actual": actual,
            "mkonline_blend__q50": actual + candidate_error,
        }
    )
    if benchmark_mode in {"official", "unaudited_official"}:
        frame["storm_dashboard_official__q50"] = actual + benchmark_error
    if duplicate_timestamp:
        frame.loc[1, "delivery_start_utc"] = frame.loc[0, "delivery_start_utc"]
    return frame


def _write_live_archive(
    tmp_path: Path,
    *,
    zone: str,
    day: str = "2026-08-15",
    benchmark_mode: str = "official",
    with_statistics: bool = True,
    duplicate_statistics_timestamp: bool = False,
    scope_note: str | None = None,
) -> ZoneStatus:
    timezone_name = ZONE_TIMEZONES[zone]
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
    level = 35.0 + 10.0 * APP_ZONES.index(zone)
    pd.DataFrame(
        {
            "delivery_start_utc": delivery,
            "forecast_origin_utc": delivery[0] - pd.Timedelta(hours=16),
            "q10": level + hours - 3.0,
            "q50": level + hours,
            "q90": level + hours + 3.0,
        }
    ).to_csv(archive / f"forecast_hourly_{zone_lower}.csv", index=False)
    (archive / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_type": "live_day_ahead",
                "forecast_status": "issued_live",
                "zone": zone,
                "timezone": timezone_name,
                "delivery_day_local": day,
                "forecast_path": f"forecast_hourly_{zone_lower}.csv",
                "sha256_manifest": "artifact_checksums.json",
                "prediction_inputs": ["autonomous_extended_residual"],
                "storm_used_as_feature": False,
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
    if with_statistics:
        _statistics_frame(
            timezone_name=timezone_name,
            benchmark_mode=benchmark_mode,
            duplicate_timestamp=duplicate_statistics_timestamp,
        ).to_csv(
            archive / "statistics_history_hourly.csv.gz",
            index=False,
            compression="gzip",
        )
        audit: dict[str, Any] = {}
        if benchmark_mode == "official":
            audit["storm_primary_report_benchmark"] = (
                "storm_dashboard_official__q50"
            )
        if scope_note is not None:
            audit["report_scope_note"] = scope_note
        (archive / "statistics_history_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False),
            encoding="utf-8",
        )
    (archive / "report.html").write_text("<html>source</html>", encoding="utf-8")
    _seal_archive(archive)
    return ZoneStatus(
        code=zone,
        timezone=timezone_name,
        enabled=True,
        production_ready=True,
        ready=True,
        runner=tmp_path / "runner.py",
        live_config=config,
        checks=("ok",),
        blockers=(),
    )


def _comparison(
    tmp_path: Path,
    *,
    zones: tuple[str, ...] = APP_ZONES,
    day: str = "2026-08-15",
    es_benchmark_mode: str = "missing",
    scope_note: str | None = None,
) -> Any:
    statuses = [
        _write_live_archive(
            tmp_path,
            zone=zone,
            day=day,
            benchmark_mode=(es_benchmark_mode if zone == "ES" else "official"),
            scope_note=scope_note,
        )
        for zone in zones
    ]
    return load_latest_forecast_comparison(
        statuses,
        project_root=tmp_path,
        zones=zones,
    )


def _assert_metric_rows(rows: list[dict[str, Any]], *, benchmark: bool) -> None:
    by_metric = {row["metric"]: row for row in rows}
    assert REQUIRED_METRICS.issubset(by_metric)
    assert set(by_metric).issubset(REQUIRED_METRICS | OPTIONAL_METRICS)
    for row in by_metric.values():
        assert {
            "candidate",
            "benchmark",
            "wins",
            "ties",
            "losses",
            "comparable_periods",
            "win_rate",
        }.issubset(row)
        assert row["candidate"] is not None
        if benchmark:
            assert row["benchmark"] is not None
            assert row["comparable_periods"] > 0
            assert row["win_rate"] is not None
        else:
            assert row["benchmark"] is None
            assert row["comparable_periods"] == 0
            assert row["win_rate"] is None


def test_report_is_offline_and_exposes_all_interactive_controls(
    tmp_path: Path,
) -> None:
    comparison = _comparison(tmp_path)

    html, document, payload = _parse_document(
        render_consolidated_forecast_report(comparison, include_intervals=True)
    )

    expected_controls = {
        "forecast-comparison-chart": "div",
        "forecast-zone-select": "div",
        "forecast-start-date": "input",
        "forecast-end-date": "input",
        "forecast-interval-toggle": "input",
        "forecast-reset": "button",
        "statistics-timeseries-chart": "div",
        "statistics-zone-select": "select",
        "history-start": "input",
        "history-end": "input",
        "history-mode": "select",
        "stats-sample": "select",
        "statistics-metric-select": "select",
        "winrate-plot": "div",
        "trend-zone": "select",
        "trend-metric": "select",
        "statistics-period-chart": "div",
        "statistics-summary-table": "table",
        "statistics-period-table": "table",
    }
    for element_id, expected_tag in expected_controls.items():
        assert document.by_id[element_id][0] == expected_tag
    assert document.by_id["history-start"][1].get("type") == "date"
    assert document.by_id["history-end"][1].get("type") == "date"
    assert document.by_id["forecast-start-date"][1].get("type") == "date"
    assert document.by_id["forecast-end-date"][1].get("type") == "date"
    assert document.by_id["forecast-interval-toggle"][1].get("type") == "checkbox"
    assert {
        attrs["data-sample"]
        for tag, attrs in document.elements
        if tag == "option" and attrs.get("data-sample")
    } == SAMPLES
    assert SAMPLES.issubset(
        {
            attrs["value"]
            for tag, attrs in document.elements
            if tag == "option" and attrs.get("value")
        }
    )

    external_resources = []
    for tag, attrs in document.elements:
        for attribute in ("src", "href"):
            target = (attrs.get(attribute) or "").strip().lower()
            if target.startswith(("http://", "https://", "//")):
                external_resources.append((tag, attribute, target))
    assert external_resources == []
    assert not any(
        tag == "script" and attrs.get("src")
        for tag, attrs in document.elements
    )
    assert "@import url(" not in html.lower()
    assert html.lower().count("plotly.js v") == 1
    assert "Plotly.react" in html
    assert "hovertemplate" in html
    assert "scrollZoom" in html
    assert "dragmode:'zoom'" in html
    assert "rangeslider" in html
    assert "groupclick:'togglegroup'" in html
    assert "#forecast-zone-select input:checked" in html
    assert "i.type='checkbox'" in html

    referenced_ids = set(re.findall(r"\$\(['\"]([^'\"]+)['\"]\)", html))
    plot_targets = set(
        re.findall(r"Plotly\.(?:newPlot|react)\(['\"]([^'\"]+)['\"]", html)
    )
    assert referenced_ids.union(plot_targets).issubset(document.by_id)

    policies = [
        attrs.get("content", "")
        for tag, attrs in document.elements
        if tag == "meta"
        and (attrs.get("http-equiv") or "").lower() == "content-security-policy"
    ]
    assert len(policies) == 1
    policy = policies[0].lower()
    assert "connect-src 'none'" in policy
    assert "object-src 'none'" in policy
    assert "base-uri 'none'" in policy

    assert payload["initial_include_intervals"] is True
    assert payload["zones"] == list(APP_ZONES)
    assert {row["zone"] for row in payload["forecast"]} == set(APP_ZONES)
    assert len(payload["forecast"]) == 24 * len(APP_ZONES)


def test_report_contains_complete_statistics_for_five_countries(
    tmp_path: Path,
) -> None:
    _html, _document, payload = _parse_document(
        render_consolidated_forecast_report(_comparison(tmp_path))
    )

    assert set(payload["statistics"]) == set(APP_ZONES)
    metric_keys = {metric["key"] for metric in payload["metrics"]}
    assert REQUIRED_METRICS.issubset(metric_keys)
    assert metric_keys.issubset(REQUIRED_METRICS | OPTIONAL_METRICS)
    assert sum(
        len(payload["statistics"][zone]["views"]["daily"]["summary"])
        for zone in APP_ZONES
    ) == 5 * len(metric_keys)

    for zone in APP_ZONES:
        statistics = payload["statistics"][zone]
        assert statistics["available"] is True
        assert statistics["candidate_label"]
        assert statistics["history_hours"] == len(statistics["time_series"])
        assert statistics["history_hours"] > 0
        assert set(statistics["views"]) == SAMPLES
        assert {
            "timestamp",
            "actual",
            "candidate",
            "benchmark",
        }.issubset(statistics["time_series"][0])
        has_benchmark = zone != "ES"
        _assert_metric_rows(
            statistics["views"]["daily"]["summary"],
            benchmark=has_benchmark,
        )
        for sample in SAMPLES:
            view = statistics["views"][sample]
            assert view["periods"]
            assert len(view["summary"]) == len(metric_keys)
            _assert_metric_rows(view["summary"], benchmark=has_benchmark)
            first_period = view["periods"][0]
            for metric in metric_keys:
                assert f"candidate_{metric}" in first_period
                assert f"benchmark_{metric}" in first_period
                assert f"outcome_{metric}" in first_period

    es = payload["statistics"]["ES"]
    assert es["benchmark_label"] is None
    assert es["benchmark_hours"] == 0
    assert all(row["benchmark"] is None for row in es["time_series"])


def test_unaudited_official_comparator_is_not_presented_as_official(
    tmp_path: Path,
) -> None:
    comparison = _comparison(
        tmp_path,
        zones=("ES",),
        es_benchmark_mode="unaudited_official",
    )

    _html, _document, payload = _parse_document(
        render_consolidated_forecast_report(comparison)
    )

    statistics = payload["statistics"]["ES"]
    assert statistics["available"] is True
    assert statistics["benchmark_label"] is None
    assert statistics["benchmark_hours"] == 0
    _assert_metric_rows(
        statistics["views"]["daily"]["summary"],
        benchmark=False,
    )


def test_statistics_with_duplicate_timestamps_fail_closed(tmp_path: Path) -> None:
    status = _write_live_archive(
        tmp_path,
        zone="FR",
        duplicate_statistics_timestamp=True,
    )
    comparison = load_latest_forecast_comparison(
        [status],
        project_root=tmp_path,
        zones=("FR",),
    )

    with pytest.raises(ValueError, match="doublons"):
        render_consolidated_forecast_report(comparison)


@pytest.mark.parametrize(
    ("day", "expected_hours"),
    (("2026-03-29", 23), ("2026-10-25", 25)),
)
def test_interactive_payload_preserves_dst_civil_days(
    tmp_path: Path,
    day: str,
    expected_hours: int,
) -> None:
    statuses = [
        _write_live_archive(
            tmp_path,
            zone=zone,
            day=day,
            with_statistics=False,
        )
        for zone in APP_ZONES
    ]
    comparison = load_latest_forecast_comparison(
        statuses,
        project_root=tmp_path,
        zones=APP_ZONES,
    )

    _html, _document, payload = _parse_document(
        render_consolidated_forecast_report(comparison)
    )

    counts = pd.DataFrame(payload["forecast"]).groupby("zone").size().to_dict()
    assert counts == {zone: expected_hours for zone in APP_ZONES}
    assert all(
        archive["hours"] == expected_hours for archive in payload["archives"]
    )


def test_embedded_json_and_visible_html_resist_script_breakout(
    tmp_path: Path,
) -> None:
    attack = '</script><script id="xss-script">alert(1)</script><img id="xss-img" src=x onerror=alert(2)>'
    comparison = _comparison(
        tmp_path,
        zones=("FR",),
        scope_note=attack,
    )
    comparison.frame.loc[0, "local_delivery"] = attack

    html, document, payload = _parse_document(
        render_consolidated_forecast_report(comparison)
    )

    assert "xss-script" not in document.by_id
    assert "xss-img" not in document.by_id
    assert "</script><script id=\"xss-script\"" not in html
    assert payload["statistics"]["FR"]["scope_note"] == attack
    assert payload["forecast"][0]["local_delivery"] == attack
    assert "NaN" not in "".join(
        document.script_text["consolidated-report-data"]
    )
    assert "Infinity" not in "".join(
        document.script_text["consolidated-report-data"]
    )
