"""Synthetic snapshots only: actual reference labels and shared score inputs."""
from copy import deepcopy
from html import unescape
import json

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import model_storm_data, model_storm_report, nuclear_reporting
from chronos2_hourly import nuclear_reporting_refresh as refresh
from test_model_storm_data import workspace, zone
from test_model_storm_report import _payload, _figure
from test_nuclear_reporting import _fixture, _capture_renderer
from test_nuclear_reporting_refresh import CONFIG, TZ, TARGET, sources


DAY = "2026-09-10"


@pytest.fixture
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("ref")


def _snapshot(project, monkeypatch):
    sources(monkeypatch, day=DAY, actual_count=24, storm_count=24)
    return refresh.refresh_nuclear_reporting_sources(
        CONFIG, "FR", TZ, DAY, workspace(project) / "report_only/sources", client=object())


def _legacy(directory, audit, *, applied=None):
    """Recreate an old audit identity around fixture prices; never live data."""
    audit = deepcopy(audit)
    audit["schema_version"] = 1
    audit.pop("observation_policy", None)
    old = audit["observed"]["source"]
    source = {key: old[key] for key in (
        "zone", "nocache", "live_recomputation", "used_for_prediction", "extracted_at_utc",
        "partial_daily_average_forbidden", "current_delivery_day_local",
        "current_delivery_actual_status", "current_delivery_expected_hours")}
    source.update(kind="saturn_target_latest_extraction", series=TARGET)
    if applied is not None:
        source["post_auction_fallback"] = {
            "applied_hours": len(applied), "applied_value_times_utc": [str(stamp) for stamp in applied]}
    audit["observed"].update(series=TARGET, source=source)
    audit["canonical_actuals"]["source"] = deepcopy(source)
    (directory / "statistics_history_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    return audit


@pytest.mark.parametrize("day", [DAY, "2026-09-09"])
def test_verified_epex_snapshot_labels_current_and_historical_display(tmp_path, monkeypatch, day):
    _, directory, _ = _snapshot(tmp_path, monkeypatch)
    observed_file = directory / "inputs/observed_latest.parquet"
    before = observed_file.read_bytes()
    payload = model_storm_data.load_model_storm_payload(
        tmp_path, day, history_from_delivery=DAY if day != DAY else None)
    current = zone(payload)
    assert current["sources"]["observed"]["actual_reference_label"] == "EPEX"
    assert current["coverage"]["observed"] == 24
    assert {row["observed"] for row in current["rows"]} == {120 if day == DAY else 100}
    assert "Realized DA · EPEX" in model_storm_report._card(current, day)
    assert observed_file.read_bytes() == before


@pytest.mark.parametrize("applied_day,expected_label", [
    ("2026-09-10", "ENTSO-E"), ("2026-09-09", "ENTSO-E + EPEX")])
def test_legacy_reference_follows_applied_hours_not_source_delivery_day(
        tmp_path, monkeypatch, applied_day, expected_label):
    _, directory, audit = _snapshot(tmp_path, monkeypatch)
    applied = [pd.Timestamp(applied_day, tz=TZ).tz_convert("UTC")]
    _legacy(directory, audit, applied=applied)
    payload = model_storm_data.load_model_storm_payload(tmp_path, "2026-09-09", history_from_delivery=DAY)
    current = zone(payload)
    assert current["sources"]["observed"]["actual_reference_label"] == expected_label
    assert current["sources"]["observed"]["displayed_epex_hours"] == int(applied_day == "2026-09-09")
    assert current["coverage"]["observed"] == 24


@pytest.mark.parametrize("reference", ["EPEX", "ENTSO-E", None])
def test_cwe_visible_reference_legend_hover_and_table_are_consistent(tmp_path, monkeypatch, reference):
    payload = _payload(DAY)
    for current in payload["zones"]:
        if reference:
            current["sources"]["observed"]["actual_reference_label"] = reference
    before = deepcopy(payload)
    monkeypatch.setattr(model_storm_report, "get_plotlyjs", lambda: "/* fixture */")
    path = model_storm_report.render_model_storm_report(payload, tmp_path / "fixture.html")
    document = unescape(path.read_text(encoding="utf-8"))
    figure = _figure(document)
    label = f"Observed ({reference})" if reference else "Observed"
    assert f'<th scope="col">{label}</th>' in document
    assert "Model and Storm errors use the same observed hourly prices" in document
    observations = [trace for trace in figure["data"] if trace["legendgroup"] == "observed"]
    assert all(trace["name"] == label and label in trace["hovertemplate"] for trace in observations)
    if reference:
        assert f"Realized DA · {reference}" in document
        assert f"BE: {reference}" in document
        assert all(f"vs {reference}" in trace["hovertemplate"]
                   for trace in figure["data"] if trace["xaxis"] in ("x5", "x6", "x7", "x8"))
    else:
        assert "EPEX" not in document
        assert "reference not identified" in document
    assert payload == before


def test_mixed_legacy_and_epex_zones_are_not_given_one_false_global_label(tmp_path, monkeypatch):
    payload = _payload(DAY)
    for current in payload["zones"]:
        current["sources"]["observed"]["actual_reference_label"] = "EPEX" if current["zone"] == "FR" else "ENTSO-E"
    figure = model_storm_report._chart(payload)
    observed = [trace for trace in figure["data"] if trace["legendgroup"] == "observed"]
    assert all(trace["name"] == "Observed (reference per zone)" for trace in observed)
    assert "Observed (EPEX)" in observed[2]["hovertemplate"]
    assert "Observed (ENTSO-E)" in observed[0]["hovertemplate"]


@pytest.mark.parametrize("legacy", [False, True])
def test_nuclear_metrics_use_exact_same_reference_for_nyx_and_storm_without_mutation(
        tmp_path, monkeypatch, legacy):
    actuals, directory, source_audit = _snapshot(tmp_path, monkeypatch)
    if legacy:
        source_audit = _legacy(directory, source_audit)
    reference = "ENTSO-E" if legacy else "EPEX"
    result, data = _fixture(DAY)
    frozen_history = result.residual_statistics.copy(deep=True)
    frozen_forecast = result.kalman_view.forecast.copy(deep=True)
    data.target = actuals.tz_convert(TZ)
    captured = _capture_renderer(monkeypatch)
    paths = nuclear_reporting.render_nuclear_reports(
        result, data=data, zone="FR", delivery_day=DAY, output_directory=tmp_path / "stub_reports",
        observed_source_audit=source_audit, storm_archive=directory, operational_layout=True)
    for item in captured:
        assert reference in item.statistics_scope_note
        assert item.backtest_native.actual.eq(100).all()
        comparison = item.hourly_comparison_source
        expected = actuals.reindex(pd.DatetimeIndex(comparison["_timestamp_utc"]))
        np.testing.assert_array_equal(comparison.actual.to_numpy(), expected.to_numpy())
        # Both residual errors subtract this one shared price column.
        assert np.isfinite(comparison.actual).all()
        assert item.statistics_candidate.tail(24).actual.eq(120).all()
    for variant in ("autonomous", "kalman"):
        document = paths[variant].read_text(encoding="utf-8")
        assert f"Prix réalisés et référence des scores : {reference}" in document
        assert document.index('data-report-section="actual-price-reference"') < document.index('data-report-section="nuclear-methodology"')
        if legacy:
            assert "EPEX" not in document
    audit = json.loads(paths["audit"].read_text())
    assert audit["actual_price_reference"]["actual_reference_label"] == reference
    pd.testing.assert_frame_equal(result.residual_statistics, frozen_history)
    pd.testing.assert_frame_equal(result.kalman_view.forecast, frozen_forecast)
