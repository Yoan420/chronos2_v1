from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

import chronos2_exogenous.rolling_research_reporting as report
from chronos2_modular.report import build_statistics_records


TZ = "Europe/Paris"


def _data(delivery_day="2026-09-08"):
    day = pd.Timestamp(delivery_day).date()
    index = pd.date_range(pd.Timestamp(day - timedelta(days=364), tz=TZ),
        pd.Timestamp(day + timedelta(days=1), tz=TZ), freq="h", inclusive="left").tz_convert("UTC")
    local_days = index.tz_convert(TZ).date
    origins = pd.DatetimeIndex([pd.Timestamp(f"{d - timedelta(days=1)} 08:00", tz=TZ)
        for d in local_days]).tz_convert("UTC")
    actual = 45 + 12 * np.sin(np.arange(len(index)) / 19)
    predictions = pd.DataFrame({"delivery_start_utc": index, "forecast_origin_utc": origins, "actual": actual})
    for model, shift in ((report.RAW_MODEL, 4), (report.RESIDUAL_MODEL, 2), (report.KALMAN_MODEL, 1)):
        for quantile, delta in zip(report.QUANTILES, (-10, 0, 10)):
            predictions[f"{model}__{quantile}"] = actual + shift + delta
    raw_inputs = pd.DataFrame({"timestamp": index})
    for column_index, name in enumerate(report.KNOWN_INPUT_COLUMNS):
        raw_inputs[name] = column_index + np.sin(np.arange(len(index)) / (21 + column_index))
    storm = pd.DataFrame({"timestamp": index, "actual": actual, "q50": actual + 3})
    contract = {"id": "provided_official_snapshot", "report_label": "Storm officiel dashboard",
        "official_dashboard_metric": True, "used_for_prediction": False,
        "report_note": "Snapshot fourni explicitement pour le test."}
    return predictions, raw_inputs, storm, contract


@pytest.fixture
def captured(monkeypatch):
    results = []
    def fake_render(values, config, path):
        results.extend(values)
        Path(path).write_text("<html><body><main>Prévision opérationnelle du x. Prévision Day-Ahead réelle.</main></body></html>", encoding="utf-8")
    monkeypatch.setattr(report, "write_html_report", fake_render)
    return results


def _render(tmp_path, predictions, inputs, storm=None, contract=None, **kwargs):
    return report.render_rolling_research_reports(predictions, inputs,
        output_directory=tmp_path / "reports", zone="FR", delivery_day="2026-09-08",
        storm=storm, storm_contract=contract, **kwargs)


def _audit(path):
    source = path.read_text(encoding="utf-8")
    return json.loads(re.search(r'id="rolling-research-audit">(.*?)</script>', source, re.S).group(1))


def test_two_chains_use_real_shared_renderer_inputs_and_metrics(tmp_path, captured):
    predictions, inputs, storm, contract = _data()
    original_predictions, original_inputs = predictions.copy(deep=True), inputs.copy(deep=True)
    paths = _render(tmp_path, predictions, inputs, storm, contract)
    assert set(paths) == {report.RESIDUAL_MODEL, report.KALMAN_MODEL}
    assert len(captured) == 2
    assert captured[0].metrics_native["mae_q50"] == pytest.approx(2)
    assert captured[1].metrics_native["mae_q50"] == pytest.approx(1)
    assert captured[0].metrics_baseline["mae_q50"] == pytest.approx(4)
    assert captured[1].metrics_baseline["mae_q50"] == pytest.approx(2)
    for result in captured:
        assert len(result.backtest_native) == len(predictions)
        assert len(result.zone_data.covariates.columns) == 35
        assert result.zone_data.target.index.max().date() < pd.Timestamp("2026-09-08").date()
        assert not hasattr(result, "variable_attribution")
        assert not hasattr(result, "kalman_diagnostics")
        records = [r for r in build_statistics_records([result]) if r["sample"] == "daily"]
        assert len(records) == 365
        assert records[0]["period_key"] == "2025-09-09"
        assert records[-1]["period_key"] == "2026-09-08"
        assert records[-1]["benchmark_mae"] == pytest.approx(3)
    for path in paths.values():
        audit = _audit(path)
        assert audit["calendar_days"] == audit["scored_days"] == 365
        assert not audit["incumbent_predictions_used"]
        assert not audit["neural_oof"]
    pd.testing.assert_frame_equal(predictions, original_predictions)
    pd.testing.assert_frame_equal(inputs, original_inputs)


def test_missing_actual_whole_day_placeholder_no_backshift_or_zero(tmp_path, captured):
    predictions, inputs, storm, contract = _data()
    predictions.loc[len(predictions) - 4, "actual"] = np.nan
    paths = _render(tmp_path, predictions, inputs, storm, contract)
    result = captured[0]
    assert result.metrics_native["n"] == len(predictions) - 24
    assert result.forecast_native["actual"].isna().all()
    records = [r for r in build_statistics_records([result]) if r["sample"] == "daily"]
    assert len(records) == 365
    assert records[-1]["mae"] is None
    assert records[-1]["observed_mean_price"] is None
    assert records[-1]["mean_price"] is not None
    audit = _audit(paths[report.RESIDUAL_MODEL])
    assert audit["scored_days"] == 364
    assert audit["pending_observation_days"] == ["2026-09-08"]
    assert audit["evaluation_start_day"] == "2025-09-09"


def test_real_zero_observation_is_not_missing(tmp_path, captured):
    predictions, inputs, _, _ = _data()
    predictions.loc[len(predictions) - 24:, "actual"] = 0.0
    _render(tmp_path, predictions, inputs)
    records = [r for r in build_statistics_records([captured[0]]) if r["sample"] == "daily"]
    assert records[-1]["observed_mean_price"] == 0
    assert records[-1]["mae"] is not None


def test_dst_physical_days_and_partial_storm_compare_same_hours(tmp_path, captured):
    predictions, inputs, storm, contract = _data()
    local = pd.DatetimeIndex(predictions.delivery_start_utc).tz_convert(TZ)
    autumn = local.date == pd.Timestamp("2025-10-26").date()
    spring = local.date == pd.Timestamp("2026-03-29").date()
    assert autumn.sum() == 25 and spring.sum() == 23
    missing_row = np.flatnonzero(autumn)[3]
    storm.loc[missing_row, "q50"] = np.nan
    paths = _render(tmp_path, predictions, inputs, storm, contract)
    result = captured[0]
    assert result.metrics_native["n"] == len(predictions)
    records = [r for r in build_statistics_records([result]) if r["sample"] == "daily"]
    assert len(records) == 365
    autumn_record = next(r for r in records if r["period_key"] == "2025-10-26")
    assert autumn_record["n"] == autumn_record["benchmark_n"] == 24
    assert autumn_record["mae"] == pytest.approx(2)
    assert autumn_record["benchmark_mae"] == pytest.approx(3)
    assert _audit(paths[report.RESIDUAL_MODEL])["storm_missing_hours"] == 1


def test_whole_storm_day_unavailable_does_not_shift_window(tmp_path, captured):
    predictions, inputs, storm, contract = _data()
    storm.loc[len(storm) - 24:, "q50"] = np.nan
    _render(tmp_path, predictions, inputs, storm, contract)
    result = captured[0]
    assert result.metrics_native["n"] == len(predictions)
    assert not hasattr(result, "forecast_benchmark")
    records = [r for r in build_statistics_records([result]) if r["sample"] == "daily"]
    assert len(records) == 365
    assert records[-1]["benchmark_mean_price"] is None
    assert "journées entières" in result.statistics_scope_note


@pytest.mark.parametrize("mutation,match", [
    (lambda p, i, s: (p.iloc[1:].copy(), i, s), "365 jours physiques"),
    (lambda p, i, s: (p.assign(forecast_origin_utc=p.delivery_start_utc), i, s), "origines"),
    (lambda p, i, s: (p.assign(lora16_residual__q10=999999), i, s), "croisés"),
    (lambda p, i, s: (p.assign(actual=np.nan), i, s), "Aucune journée"),
    (lambda p, i, s: (p, i.drop(columns="local_temperature_fcst"), s), "vrais inputs"),
    (lambda p, i, s: (p, i.iloc[1:].copy(), s), "chaque heure"),
    (lambda p, i, s: (p, i, s.assign(actual=s.actual + 1)), "divergent"),
])
def test_reject_unsupported_claim_before_writing(tmp_path, captured, mutation, match):
    predictions, inputs, storm, contract = _data()
    predictions, inputs, storm = mutation(predictions, inputs, storm)
    with pytest.raises(report.RollingResearchReportingError, match=match):
        _render(tmp_path, predictions, inputs, storm, contract)
    assert not captured


def test_no_contract_no_false_official_storm(tmp_path, captured):
    predictions, inputs, storm, _ = _data()
    with pytest.raises(report.RollingResearchReportingError, match="contrat explicite"):
        _render(tmp_path, predictions, inputs, storm)


def test_reports_immutable_and_metadata_cannot_promote(tmp_path, captured):
    predictions, inputs, _, _ = _data()
    paths = _render(tmp_path, predictions, inputs,
        metadata={"production_pit_evidence": True, "promotion_eligible": True,
            "neural_oof": True, "user_note": "</script><script>evil</script>"})
    first = paths[report.RESIDUAL_MODEL]
    original = first.read_bytes()
    with pytest.raises(FileExistsError):
        _render(tmp_path, predictions, inputs)
    assert first.read_bytes() == original
    audit = _audit(first)
    assert audit["production_pit_evidence"] is audit["promotion_eligible"] is audit["neural_oof"] is False
    assert "<script>evil</script>" not in first.read_text(encoding="utf-8")


def test_real_renderer_reproduces_statistics_calendar_theme_and_research_labels(tmp_path):
    predictions, inputs, storm, contract = _data()
    paths = _render(tmp_path, predictions, inputs, storm, contract)
    for model, path in paths.items():
        source = path.read_text(encoding="utf-8")
        assert report.MODEL_LABELS[model] in source
        assert "research-methodology" in source
        assert "Statistics" in source
        assert "statistics-price-calendar" in source
        assert "theme-toggle" in source
        assert "Backtest et probabilités" in source
        assert "interpolés linéairement" in source
        assert "Attribution des variables : non recalculée" in source
        assert "Prévision opérationnelle du" not in source
        assert "Prévision Day-Ahead réelle" not in source
        assert "prévision Day-Ahead opérationnelle" not in source
        assert "2025-09-09 → 2026-09-08" in source
        assert _audit(path)["physical_hours"] == len(predictions)


def test_visible_methodology_and_snapshot_provenance_are_explicit_and_escaped(tmp_path):
    path = tmp_path / "report.html"
    path.write_text("<html><body><main></main></body></html>", encoding="utf-8")
    audit = {"comparator_audit": {
        "snapshot_generated_text": "2026-09-07 15:18 CEST <img src=x>",
        "observation_extraction_timestamps": ["2026-09-07 12:27:00+00:00", "2026-09-07 12:28:00+00:00"],
        "network_refreshed": False}}
    report._research_html(path, label=report.MODEL_LABELS[report.RESIDUAL_MODEL],
        baseline_label=report.MODEL_LABELS[report.RAW_MODEL], scope="365 jours", audit=audit)
    source = path.read_text(encoding="utf-8")
    visible = re.search(r'<section id="research-methodology".*?</section>', source, re.S).group(0)
    assert "recouvre la période d’entraînement/validation LoRA" in visible
    assert "peut appartenir" not in visible
    assert "Il n’est pas réentraîné chaque jour" in visible
    assert "Pour chacun des 365 jours évalués" in visible
    assert "le correcteur résiduel est ajusté sur les 365 jours antérieurs" in visible
    assert "le Kalman est recalibré sur les 365 jours antérieurs" in visible
    assert "30 jours sans correction (identité)" in visible
    assert "fenêtre croissante jusqu’à 365 jours" in visible
    assert "snapshot exact du rapport publié" in visible
    assert "2026-09-07 15:18 CEST &lt;img src=x&gt;" in visible
    assert "2026-09-07 12:27:00+00:00" in visible
    assert "2026-09-07 12:28:00+00:00" in visible
    assert "sans actualisation API" in visible
    assert "jamais à la calibration" in visible
    assert "<img src=x>" not in source
