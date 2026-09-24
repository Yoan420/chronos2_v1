"""Solar/wind reports reuse the operational renderer; synthetic frozen inputs only."""
from copy import deepcopy
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import solar_wind_reporting as report
from chronos2_hourly.solar_wind_forecast import GENERATION_SERIES
from chronos2_modular.report import build_statistics_records


def fixture(day="2026-09-22", observed=False):
    nuclear = runpy.run_path(str(Path(__file__).with_name("test_nuclear_reporting.py")))
    result, data = nuclear["_fixture"](day)
    cov = result.covariates.set_index("timestamp").rename(columns={"fr_nuclear_fcst": report.NUCLEAR_ALIAS})
    cov[report.NUCLEAR_ALIAS] /= 1000
    for number, alias in enumerate(report.GENERATION_ALIASES):
        cov[alias] = 1. + number
    context = cov.copy()
    for alias in cov:
        context[f"known_{alias}_oracle"] = cov[alias]
    result.covariates = cov.copy()
    data.model_context_covariates = context.copy()
    data.covariates = cov.reindex(data.target.index)
    data.known_future_columns = [f"known_{alias}_oracle" for alias in cov]
    known = [f"known_{alias}_oracle" for alias in report.GENERATION_ALIASES]
    result.audit = {"candidate_engine": report.ENGINE, "candidate_variant": "solar_wind_kalman",
        "solar_wind_chronos_context_columns": list(report.GENERATION_ALIASES),
        "solar_wind_chronos_known_future_columns": known, "solar_wind_residual_features": known,
        "solar_wind_kalman_market_features": list(report.GENERATION_ALIASES), "baseline_inputs_preserved": True,
        "solar_wind_sources": {alias: {"series": GENERATION_SERIES[alias], "unit": "GW",
            "semantic": "forecast_generation", "daily_broadcast": False} for alias in report.GENERATION_ALIASES}}
    incumbent = deepcopy(result)
    incumbent.audit = {"engine": "nuclear_forecast_v1"}
    for frame in (incumbent.residual_statistics, incumbent.source_forecast,
                  incumbent.kalman_view.backtest, incumbent.kalman_view.forecast):
        for column in frame:
            if column.startswith(("residual_corrected__", "residual_kalman__")):
                frame[column] += .5
    if observed:
        times = pd.DatetimeIndex(result.source_forecast.delivery_start_utc).tz_convert(data.timezone)
        data.target = pd.concat([data.target, pd.Series(53., index=times, name=data.target.name)])
    return result, incumbent, data


def capture(monkeypatch):
    rendered = []
    def write(results, config, path):
        rendered.extend(results)
        path.write_text('<html><main><h3>Comparaison au modèle prix seul</h3><table><tr><td>obsolete</td></tr></table>'
                        '<p>Prévision opérationnelle du jour</p>Covariables natives</main></html>', encoding="utf-8")
    monkeypatch.setattr(report, "write_html_report", write)
    return rendered


def run(tmp_path, monkeypatch, result, incumbent, data, **kwargs):
    monkeypatch.setattr(report, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "runs/experiments/solar_wind_v1/reports"
    day = str(pd.Timestamp(result.source_forecast.delivery_start_utc.iloc[0]).tz_convert(data.timezone).date())
    options = dict(storm_archive=None, observed_source_audit=None, source_audit={"fixture": True})
    options.update(kwargs)
    return report.render_solar_wind_reports(result, incumbent=incumbent, data=data, zone=data.zone,
        delivery_day=day, output_directory=output, **options)


@pytest.mark.parametrize("observed", [False, True])
def test_same_family_reports_share_actuals_and_preserve_inputs(tmp_path, monkeypatch, observed):
    result, incumbent, data = fixture(observed=observed)
    before = deepcopy((result, incumbent, data))
    rendered = capture(monkeypatch)
    paths = run(tmp_path, monkeypatch, result, incumbent, data)
    assert set(paths) == {"autonomous", "kalman", "audit", "comparison"}
    assert len(rendered) == 2
    comparison = json.loads(paths["comparison"].read_text(encoding="utf-8"))
    for family in ("autonomous", "kalman"):
        assert comparison[family]["candidate_model"] == "solar_wind_" + family
        assert comparison[family]["incumbent_model"] == "nuclear_" + family
        assert comparison[family]["annual"]["gain_incumbent_minus_candidate"]["mae_eur_mwh"] == pytest.approx(.5)
        assert comparison[family]["statistics_days"] == 365
        assert comparison[family]["delivery_day_included"] is observed
        assert "june_24_26" not in comparison[family]
        document = paths[family].read_text(encoding="utf-8")
        assert "SolarWind" in document and "solaire CWE" in document and "éolien DE/NL" in document
        assert "rétrospective as-of 08 h" in document
        assert "choisi après consultation du marché" in document
        assert "Prévision opérationnelle" not in document
        assert "CGC" not in document and "CCC" not in document and "obsolete" not in document
        assert f"_solar_wind_{family}.html" in paths[family].name
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    assert audit["candidate_engine"] == "solar_wind_v1"
    assert audit["candidate_variant"] == "solar_wind_kalman"
    assert audit["publication_mode"] == "experimental_retrospective_asof_08"
    assert audit["prospective_validation"] is False
    assert audit["test_case_selected_after_market_observation"] is True
    assert audit["target_cutoff_local"] == "08:00"
    assert len(audit["generation_inputs"]) == 6
    assert {value["technology"] for value in audit["generation_inputs"].values()} == {"solar", "wind"}
    assert audit["model_fitted_by_report"] is False and audit["production"] is False
    assert audit["evaluation_hours"] == 8760
    for item in rendered:
        assert item.metrics_native["n"] == 8760
        assert item.statistics_candidate.tail(24).actual.notna().all() == observed
        records = [r for r in build_statistics_records([item]) if r["sample"] == "daily"]
        assert len(records) == (365 if observed else 366)
        for alias in report.GENERATION_ALIASES:
            assert alias in item.zone_data.model_context_covariates
            assert f"known_{alias}_oracle" in item.zone_data.known_future_columns
            row = item.zone_data.input_manifest.set_index("alias").loc[alias]
            assert row["unit"] == "GW"
            assert row["source_zone"] == alias[:2].upper()
            assert row["information_type"] == ("solar_generation_forecast" if "_solar_" in alias else "wind_generation_forecast")
    pd.testing.assert_frame_equal(result.residual_statistics, before[0].residual_statistics)
    pd.testing.assert_frame_equal(result.kalman_view.forecast, before[0].kalman_view.forecast)
    pd.testing.assert_frame_equal(incumbent.source_forecast, before[1].source_forecast)
    pd.testing.assert_frame_equal(data.model_context_covariates, before[2].model_context_covariates)


@pytest.mark.parametrize("fault", ["engine", "chronos", "residual", "kalman", "unit", "broadcast", "known", "values", "nuclear", "incumbent_grid", "kalman_upstream"])
def test_invalid_full_chain_or_incumbent_fails_before_any_write(tmp_path, monkeypatch, fault):
    result, incumbent, data = fixture()
    alias = report.GENERATION_ALIASES[-1]
    if fault == "engine":
        result.audit["candidate_engine"] = "residual_only_solar"
    elif fault in ("chronos", "residual", "kalman"):
        key = {"chronos": "solar_wind_chronos_context_columns", "residual": "solar_wind_residual_features", "kalman": "solar_wind_kalman_market_features"}[fault]
        result.audit[key] = []
    elif fault == "unit":
        result.audit["solar_wind_sources"][alias]["unit"] = "MW"
    elif fault == "broadcast":
        result.audit["solar_wind_sources"][alias]["daily_broadcast"] = True
    elif fault == "known":
        data.known_future_columns.remove(f"known_{alias}_oracle")
    elif fault == "values":
        result.covariates[alias] += 1
    elif fault == "nuclear":
        result.covariates = result.covariates.drop(columns=report.NUCLEAR_ALIAS)
    elif fault == "incumbent_grid":
        incumbent.kalman_view.backtest = incumbent.kalman_view.backtest.iloc[1:]
    else:
        result.kalman_view.forecast["residual_corrected__q50"] += 1
    capture(monkeypatch)
    with pytest.raises(ValueError):
        run(tmp_path, monkeypatch, result, incumbent, data)
    assert not (tmp_path / "runs").exists()


def test_refuses_production_output_directory(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    monkeypatch.setattr(report, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="SolarWind reports must stay"):
        report.render_solar_wind_reports(result, incumbent=incumbent, data=data, zone="FR", delivery_day="2026-09-08",
            output_directory=tmp_path / "runs/exports", storm_archive=None, observed_source_audit=None, source_audit={})
    assert not (tmp_path / "runs").exists()


def test_storm_uses_same_observations_and_does_not_filter_family_comparison(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    calls = []
    def attach(prepared, path, **kwargs):
        calls.append(prepared)
        truth = prepared["autonomous"].statistics_candidate.actual
        assert all(p.statistics_candidate.actual.equals(truth) for p in prepared.values())
        for p in prepared.values():
            p.statistics_benchmark = p.statistics_candidate[["timestamp", "actual", "q50"]].copy()
            p.statistics_benchmark.loc[p.statistics_benchmark.index[:3], "q50"] = np.nan
        return {"status": "complete", "fixture_missing_storm_hours": 3}
    monkeypatch.setattr("chronos2_hourly.nuclear_report_benchmark.attach_nuclear_storm", attach)
    rendered = capture(monkeypatch)
    paths = run(tmp_path, monkeypatch, result, incumbent, data, storm_archive=tmp_path / "frozen_storm")
    assert len(calls) == 1
    assert all(item.statistics_benchmark.q50.isna().sum() == 3 for item in rendered)
    comparison = json.loads(paths["comparison"].read_text())
    assert comparison["kalman"]["annual"]["candidate"]["hours"] == 8760


def test_old_attribution_without_generation_inputs_is_refused_before_write(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    capture(monkeypatch)
    def attach(target, **kwargs):
        target.variable_attribution = {"groups": [{"key": report.NUCLEAR_ALIAS}]}
    monkeypatch.setattr("chronos2_hourly.reporting._attach_variable_attribution", attach)
    with pytest.raises(ValueError, match="Attribution must cover all six"):
        run(tmp_path, monkeypatch, result, incumbent, data, attribution_directory=tmp_path / "old_attribution")
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("alias", report.GENERATION_ALIASES)
@pytest.mark.parametrize("stage", ["chronos_context_columns", "chronos_known_future_columns", "residual_features", "kalman_market_features"])
def test_every_generation_channel_is_required_at_each_stage(tmp_path, monkeypatch, alias, stage):
    result, incumbent, data = fixture()
    key = "solar_wind_" + stage
    column = f"known_{alias}_oracle" if stage in ("chronos_known_future_columns", "residual_features") else alias
    result.audit[key].remove(column)
    with pytest.raises(ValueError, match="all six solar/wind inputs"):
        run(tmp_path, monkeypatch, result, incumbent, data)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("fault", ["missing_history", "missing_future", "nan_prediction", "crossed_quantiles", "different_truth", "wrong_source_series", "negative_wind", "baseline_not_preserved"])
def test_incomplete_or_invalid_history_cannot_be_reported_as_annual(tmp_path, monkeypatch, fault):
    result, incumbent, data = fixture()
    alias = report.GENERATION_ALIASES[-1]
    if fault == "missing_history":
        result.residual_statistics = result.residual_statistics.iloc[:-1]
    elif fault == "missing_future":
        result.source_forecast = result.source_forecast.iloc[:-1]
    elif fault == "nan_prediction":
        result.kalman_view.backtest.loc[result.kalman_view.backtest.index[-1], "residual_kalman__q50"] = np.nan
    elif fault == "crossed_quantiles":
        result.kalman_view.forecast["residual_kalman__q10"] = result.kalman_view.forecast["residual_kalman__q90"] + 1
    elif fault == "different_truth":
        incumbent.residual_statistics["actual"] += 2
        incumbent.kalman_view.backtest["actual"] += 2
    elif fault == "wrong_source_series":
        result.audit["solar_wind_sources"][alias]["series"] = "power.nl.generation.solar.hourly.gw.fcst"
    elif fault == "negative_wind":
        result.covariates[alias] = -1
    else:
        result.audit["baseline_inputs_preserved"] = False
    with pytest.raises(ValueError):
        run(tmp_path, monkeypatch, result, incumbent, data)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2026-10-25", 25)])
def test_dst_delivery_keeps_every_physical_hour(tmp_path, monkeypatch, day, hours):
    result, incumbent, data = fixture(day=day)
    rendered = capture(monkeypatch)
    paths = run(tmp_path, monkeypatch, result, incumbent, data)
    for item in rendered:
        assert len(item.forecast_native) == hours
        assert item.statistics_candidate.tail(hours).actual.isna().all()
        assert item.statistics_candidate.timestamp.is_unique
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    comparison = json.loads(paths["comparison"].read_text(encoding="utf-8"))
    assert audit["evaluation_days"] == 365
    assert audit["test_case_selected_after_market_observation"] is False
    assert all(c["annual"]["candidate"]["days"] == 365 for c in comparison.values())
    assert all(c["annual"]["candidate"]["hours"] == audit["evaluation_hours"] for c in comparison.values())
    assert "choisi après consultation du marché" not in paths["kalman"].read_text(encoding="utf-8")


def test_retrospective_source_and_nonzero_wind_dst_caveats_are_visible(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    capture(monkeypatch)
    source = {"wind_dst_policy": "duplicate", "singleton_fold_dates": ["2025-10-26"],
              "nonzero_singleton_fold_values": [9.1], "original_publication_certified": False}
    paths = run(tmp_path, monkeypatch, result, incumbent, data, source_audit=source)
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    assert audit["source_audit"] == source
    assert audit["original_publication_time_certified_by_report"] is False
    for family in ("autonomous", "kalman"):
        document = paths[family].read_text(encoding="utf-8")
        assert "as-of de requête" in document
        assert "pas une certification de la publication originale" in document
        assert "politique duplicate" in document
        assert "y compris lorsqu’elle est non nulle" in document
        assert "peut biaiser les entrées et les scores" in document
        assert "2025-10-26" in document


def test_matching_generation_attribution_is_upstream_only_for_kalman(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    rendered = capture(monkeypatch)
    def attach(target, **kwargs):
        target.variable_attribution = {"groups": [
            {"key": alias, "context_columns": [alias], "future_columns": [f"known_{alias}_oracle"]}
            for alias in report.GENERATION_ALIASES]}
    monkeypatch.setattr("chronos2_hourly.reporting._attach_variable_attribution", attach)
    paths = run(tmp_path, monkeypatch, result, incumbent, data, attribution_directory=tmp_path / "matching_attribution")
    assert rendered[1].variable_attribution["is_upstream_attribution"] is True
    assert rendered[1].variable_attribution["explained_model"] == "residual_corrected"
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    assert audit["variable_attribution_status"] == "verified_upstream_artifact_not_total_Kalman_influence"


def test_standard_renderer_preserves_statistics_calendar_and_components(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    paths = run(tmp_path, monkeypatch, result, incumbent, data,
                source_audit={"wind_dst_policy": "duplicate"})
    for family in ("autonomous", "kalman"):
        document = paths[family].read_text(encoding="utf-8")
        for marker in ('data-report-section="average-prices"', 'data-report-section="statistics"',
                       'data-report-section="forecast-components"', 'data-report-section="attribution-methodology"',
                       'chronos2-theme-change', 'calendar', 'Plotly.newPlot', 'Performance par heure locale'):
            assert marker in document
        assert "Attribution chiffrée indisponible" in document
        assert "solaire CWE + éolien DE/NL" in document
        assert "Prévision opérationnelle du" not in document
        assert "choisi après consultation du marché" in document
        assert "politique duplicate" in document
        assert "Comparaison au modèle prix seul" not in document


def source_substitution(day):
    value_time = pd.Timestamp(day + "T02:00:00Z")
    cutoff = (pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Amsterdam")
    return {"policy": "nl_ecmwf_spring_2025_2026", "alias": "nl_wind_generation_fcst",
        "native_series": GENERATION_SERIES["nl_wind_generation_fcst"],
        "fallback_series": "power.nrjscan.nl.prod.total.wind.mw.ecmwf_avg.pointconnect.6h.cache",
        "delivery_day": day, "value_time_utc": value_time.isoformat(),
        "local_time": value_time.tz_convert("Europe/Amsterdam").isoformat(),
        "query_cutoff_utc": cutoff.tz_convert("UTC").isoformat(), "query_cutoff_local": cutoff.isoformat(),
        "raw_value_mw": 1234., "value_scale": .001, "scaled_value_gw": 1.234,
        "native_missing": True, "fallback_downloaded_at_utc": "2026-09-21T17:00:00+00:00",
        "provenance": "approved test <source>", "provider_revision_timestamp_available": False,
        "production_pit_evidence": False}


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("location", ["root", "record", "nested_audit", "repeated"])
def test_actual_ecmwf_substitutions_are_counted_and_disclosed(tmp_path, monkeypatch, count, location):
    result, incumbent, data = fixture()
    capture(monkeypatch)
    entries = [source_substitution(day) for day in ("2025-03-30", "2026-03-29")[:count]]
    source = {"wind_gap_policy": "nl_ecmwf_spring_2025_2026"}
    record = {"alias": "nl_wind_generation_fcst"}
    if location in ("root", "repeated"):
        source["source_substitutions"] = deepcopy(entries)
    if location in ("record", "repeated"):
        record["source_substitutions"] = deepcopy(entries)
    if location in ("nested_audit", "repeated"):
        record["audit"] = {"source_substitutions": deepcopy(entries)}
    if location != "root":
        source["solar_wind"] = {"nl_wind_generation_fcst": record}
    before = deepcopy(source)
    paths = run(tmp_path, monkeypatch, result, incumbent, data, source_audit=source)
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    assert source == before and audit["source_audit"] == before
    assert audit["source_substitution_count"] == count
    assert audit["source_substitutions"] == entries
    for family in ("autonomous", "kalman"):
        document = paths[family].read_text(encoding="utf-8")
        assert 'data-report-section="solar-wind-source-substitutions"' in document
        assert f"Substitutions effectives dans l’audit source : <strong>{count}</strong>" in document
        assert "La courbe native NL est conservée" in document
        assert "par ECMWF au même cutoff de requête D−1 à 08 h" in document
        assert "sans interpolation" in document
        assert "pas nécessairement dans la fenêtre annuelle évaluée" in document
        assert "MW sont converties en GW" in document
        assert "ne certifie ni la publication originale ni la disponibilité PIT" in document
        assert "choisi après consultation du marché" in document
        assert "approved test &lt;source&gt;" in document and "approved test <source>" not in document
        for entry in entries:
            for key in ("value_time_utc", "query_cutoff_utc", "native_series", "fallback_series"):
                assert entry[key] in document


def test_ecmwf_policy_or_claimed_count_alone_does_not_invent_substitutions(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    capture(monkeypatch)
    source = {"wind_gap_policy": "nl_ecmwf_spring_2025_2026", "source_substitution_count": 2,
              "source_substitutions": [], "solar_wind": {"nl_wind_generation_fcst": {
                  "source_substitution_count": 2, "audit": {"source_substitutions": []}}}}
    paths = run(tmp_path, monkeypatch, result, incumbent, data, source_audit=source)
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    assert audit["source_substitution_count"] == 0
    assert audit["source_substitutions"] == []
    assert audit["source_audit"] == source
    for family in ("autonomous", "kalman"):
        assert 'data-report-section="solar-wind-source-substitutions"' not in paths[family].read_text(encoding="utf-8")


def test_conflicting_substitution_references_fail_before_report_write(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    entry = source_substitution("2025-03-30")
    different = {**entry, "scaled_value_gw": 9.999}
    source = {"source_substitutions": [entry], "solar_wind": {
        "nl_wind_generation_fcst": {"audit": {"source_substitutions": [different]}}}}
    with pytest.raises(ValueError, match="audits disagree"):
        run(tmp_path, monkeypatch, result, incumbent, data, source_audit=source)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("entry", [{"alias": "nl_wind_generation_fcst"},
                                    {"alias": "nl_wind_generation_fcst", "value_time_utc": "2025-03-30T02:00:00"}])
def test_unidentifiable_substitutions_cannot_be_silently_counted(entry):
    with pytest.raises(ValueError, match="physical hour"):
        report._source_substitutions({"source_substitutions": [entry]})
