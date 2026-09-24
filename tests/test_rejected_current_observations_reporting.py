"""Synthetic verified snapshots; no network, model execution or heavy HTML render."""
from copy import deepcopy
from html import unescape
import json

import pandas as pd
import pytest

from chronos2_hourly import model_storm_data, model_storm_report, nuclear_reporting
from chronos2_hourly import nuclear_reporting_refresh as refresh
from test_model_storm_data import frozen, workspace, zone
from test_nuclear_reporting import _fixture, _capture_renderer
from test_nuclear_reporting_refresh import CONFIG, TZ, sources
from test_epex_reporting_reference import _legacy


DAY = "2026-09-10"
FR_NOTE = (
    "Prix réalisés du jour non validés : les sources sont en désaccord. "
    "Prévisions disponibles ; scores du jour non calculés."
)


@pytest.fixture
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("orj")


def rejection_snapshot(project, monkeypatch, *, rejected=True):
    sources(monkeypatch, day=DAY, actual_count=0, storm_count=24)
    values, directory, audit = refresh.refresh_nuclear_reporting_sources(
        CONFIG, "FR", TZ, DAY, workspace(project) / "report_only/sources", client=object())
    if rejected:
        # Synthetic old snapshot: new EPEX-only refreshes never create this
        # legacy disagreement state, but archived reports must still explain it.
        audit = _legacy(directory, audit)
        source = audit["observed"]["source"]
        source["current_delivery_actual_reason"] = "post_auction_source_rejected"
        source["post_auction_fallback"] = {
            "status": "rejected_divergent_optional_current_day",
            "validation_status": "rejected_divergence", "applied_hours": 0,
            "rejection_diagnostic": {"zone": "FR", "maximum_difference_eur_mwh": 7.89},
        }
        audit["canonical_actuals"]["source"] = deepcopy(source)
        (directory / "statistics_history_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    return values, directory, audit


def test_verified_rejection_reaches_cwe_without_mutating_predictions_or_prices(tmp_path, monkeypatch):
    frozen(tmp_path)
    _, directory, _ = rejection_snapshot(tmp_path, monkeypatch)
    paths = [directory / "inputs/observed_latest.parquet",
             workspace(tmp_path) / "report_only/frozen_result/kalman_forecast.parquet"]
    before = [path.read_bytes() for path in paths]
    payload = model_storm_data.load_model_storm_payload(tmp_path, DAY)
    current = zone(payload)
    assert current["coverage"] == {"storm": 24, "observed": 0, "model": 24}
    assert all(row["observed"] is None and row["model"] == 51.5 for row in current["rows"])
    assert current["sources"]["observed"]["validation_status"] == "rejected_divergence"
    assert "sources disagree" in current["message"]
    assert "Observed prices unavailable" not in current["message"]
    assert before == [path.read_bytes() for path in paths]
    json.dumps(payload, allow_nan=False)


def test_rejection_is_explicit_in_cwe_card_and_error_annotation_without_scores(tmp_path, monkeypatch):
    frozen(tmp_path)
    rejection_snapshot(tmp_path, monkeypatch)
    payload = model_storm_data.load_model_storm_payload(tmp_path, DAY)
    current = zone(payload)
    before = deepcopy(payload)
    card = unescape(model_storm_report._card(current, DAY))
    assert model_storm_report.OBSERVED_REJECTION_NOTE in card
    extracted = pd.Timestamp(current["sources"]["observed"]["extracted_at_utc"]).tz_convert(TZ)
    assert "Last source extraction: " + extracted.strftime("%Y-%m-%d %H:%M %Z") in card
    assert "Observed unavailable" not in card
    assert 'class="delta ' not in card
    assert model_storm_report.daily_metrics(current)["observed"] is None
    figure = model_storm_report._chart(payload)
    assert any("Sources disagree; no daily scores" in a["text"] for a in figure["layout"]["annotations"])
    fr_errors = [t for t in figure["data"] if t.get("xaxis") == "x7"]
    assert fr_errors and all(all(value is None for value in t["y"]) for t in fr_errors)
    assert payload == before


def test_unpublished_day_keeps_distinct_unavailable_message(tmp_path, monkeypatch):
    rejection_snapshot(tmp_path, monkeypatch, rejected=False)
    payload = model_storm_data.load_model_storm_payload(tmp_path, DAY)
    current = zone(payload)
    assert current["sources"]["observed"].get("validation_status") != "rejected_divergence"
    assert "Observed prices unavailable" in current["message"]
    card = unescape(model_storm_report._card(current, DAY))
    assert "sources disagree" not in card
    assert "Observed unavailable" in card


def test_later_current_day_rejection_does_not_mark_verified_historical_prices(tmp_path, monkeypatch):
    rejection_snapshot(tmp_path, monkeypatch)
    payload = model_storm_data.load_model_storm_payload(
        tmp_path, "2026-09-09", history_from_delivery=DAY)
    previous = zone(payload)
    assert previous["coverage"]["observed"] == 24
    assert all(row["observed"] == 100 for row in previous["rows"])
    assert previous["sources"]["observed"].get("current_delivery_actual_reason") is None
    assert "sources disagree" not in previous["message"]
    assert "sources disagree" not in unescape(model_storm_report._card(previous, "2026-09-09"))


@pytest.mark.parametrize("operational", [False, True])
def test_nuclear_warning_visible_above_collapsed_method_and_day_labels_stay_empty(
        tmp_path, monkeypatch, operational):
    values, _, audit = rejection_snapshot(tmp_path, monkeypatch)
    result, data = _fixture(DAY)
    data.target = values.tz_convert(TZ)
    source_before = result.source_forecast.copy(deep=True)
    kalman_before = result.kalman_view.forecast.copy(deep=True)
    target_before = data.target.copy(deep=True)
    captured = _capture_renderer(monkeypatch)
    paths = nuclear_reporting.render_nuclear_reports(
        result, data=data, zone="FR", delivery_day=DAY,
        output_directory=tmp_path / "render_stub", observed_source_audit=audit,
        operational_layout=operational)
    report_audit = json.loads(paths["audit"].read_text())
    assert report_audit["delivery_day_observed_hours"] == 0
    assert report_audit["delivery_day_actual_reason"] == "post_auction_source_rejected"
    for key in ("autonomous", "kalman"):
        html = paths[key].read_text(encoding="utf-8")
        assert FR_NOTE in html
        extracted = pd.Timestamp(audit["observed"]["source"]["extracted_at_utc"]).tz_convert(TZ)
        assert "Dernière extraction des sources : " + extracted.strftime("%d/%m/%Y %H:%M %Z") in html
        assert 'data-validation-status="rejected_divergence"' in html
        if operational:
            assert html.index(FR_NOTE) < html.index('<details data-report-section="nuclear-methodology">')
    for item in captured:
        # All scored backtest observations precede D; the delivery prediction is
        # retained separately and its report actuals are NaN.
        frame = item.backtest_native
        days = pd.to_datetime(frame.timestamp, utc=True).dt.tz_convert(TZ).dt.date
        assert frame.loc[days.eq(pd.Timestamp(DAY).date()), "actual"].isna().all()
    pd.testing.assert_frame_equal(result.source_forecast, source_before)
    pd.testing.assert_frame_equal(result.kalman_view.forecast, kalman_before)
    pd.testing.assert_series_equal(data.target, target_before)


def test_generic_pending_nuclear_report_does_not_claim_disagreement(tmp_path, monkeypatch):
    values, _, audit = rejection_snapshot(tmp_path, monkeypatch, rejected=False)
    result, data = _fixture(DAY)
    data.target = values.tz_convert(TZ)
    _capture_renderer(monkeypatch)
    paths = nuclear_reporting.render_nuclear_reports(
        result, data=data, zone="FR", delivery_day=DAY, output_directory=tmp_path / "pending_stub",
        observed_source_audit=audit, operational_layout=True, report_variants=("kalman",))
    assert FR_NOTE not in paths["kalman"].read_text(encoding="utf-8")
    assert json.loads(paths["audit"].read_text())["delivery_day_actual_reason"] is None
