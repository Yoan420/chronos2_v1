"""Solar reports reuse the operational renderer; synthetic frozen inputs only."""
from copy import deepcopy
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import solar_cwe_reporting as report
from chronos2_hourly.solar_cwe_forecast import SOLAR_SERIES
from chronos2_modular.report import build_statistics_records


def fixture(day="2026-09-08", observed=False):
    nuclear = runpy.run_path(str(Path(__file__).with_name("test_nuclear_reporting.py")))
    result, data = nuclear["_fixture"](day)
    cov = result.covariates.set_index("timestamp").rename(columns={"fr_nuclear_fcst": report.NUCLEAR_ALIAS})
    cov[report.NUCLEAR_ALIAS] /= 1000
    for number, alias in enumerate(report.SOLAR_ALIASES):
        cov[alias] = 1. + number
    context = cov.copy()
    for alias in cov:
        context[f"known_{alias}_oracle"] = cov[alias]
    result.covariates = cov.copy()
    data.model_context_covariates = context.copy()
    data.covariates = cov.reindex(data.target.index)
    data.known_future_columns = [f"known_{alias}_oracle" for alias in cov]
    known = [f"known_{alias}_oracle" for alias in report.SOLAR_ALIASES]
    result.audit = {"candidate_engine": report.ENGINE, "candidate_variant": "solar_cwe_kalman",
        "solar_chronos_context_columns": list(report.SOLAR_ALIASES),
        "solar_chronos_known_future_columns": known, "solar_residual_features": known,
        "solar_kalman_market_features": list(report.SOLAR_ALIASES), "baseline_inputs_preserved": True,
        "solar_sources": {alias: {"series": SOLAR_SERIES[alias], "unit": "GW",
            "semantic": "forecast_generation", "daily_broadcast": False} for alias in report.SOLAR_ALIASES}}
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
    output = tmp_path / "runs/experiments/solar_cwe_v1/reports"
    day = str(pd.Timestamp(result.source_forecast.delivery_start_utc.iloc[0]).tz_convert(data.timezone).date())
    options = dict(storm_archive=None, observed_source_audit=None, source_audit={"fixture": True})
    options.update(kwargs)
    return report.render_solar_cwe_reports(result, incumbent=incumbent, data=data, zone=data.zone,
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
        assert comparison[family]["candidate_model"] == "solar_cwe_" + family
        assert comparison[family]["incumbent_model"] == "nuclear_" + family
        assert comparison[family]["annual"]["gain_incumbent_minus_candidate"]["mae_eur_mwh"] == pytest.approx(.5)
        assert comparison[family]["statistics_days"] == 365
        assert comparison[family]["delivery_day_included"] is observed
        assert "june_24_26" not in comparison[family]
        document = paths[family].read_text(encoding="utf-8")
        assert "SolarCWE" in document and "solaire CWE" in document
        assert "CGC" not in document and "CCC" not in document and "obsolete" not in document
        assert f"_solar_{family}.html" in paths[family].name
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    assert audit["candidate_engine"] == "solar_cwe_v1"
    assert audit["candidate_variant"] == "solar_cwe_kalman"
    assert audit["model_fitted_by_report"] is False and audit["production"] is False
    assert audit["evaluation_hours"] == 8760
    for item in rendered:
        assert item.metrics_native["n"] == 8760
        assert item.statistics_candidate.tail(24).actual.notna().all() == observed
        records = [r for r in build_statistics_records([item]) if r["sample"] == "daily"]
        assert len(records) == (365 if observed else 366)
        for alias in report.SOLAR_ALIASES:
            assert alias in item.zone_data.model_context_covariates
            assert f"known_{alias}_oracle" in item.zone_data.known_future_columns
            assert item.zone_data.input_manifest.set_index("alias").loc[alias, "unit"] == "GW"
    pd.testing.assert_frame_equal(result.residual_statistics, before[0].residual_statistics)
    pd.testing.assert_frame_equal(result.kalman_view.forecast, before[0].kalman_view.forecast)
    pd.testing.assert_frame_equal(incumbent.source_forecast, before[1].source_forecast)
    pd.testing.assert_frame_equal(data.model_context_covariates, before[2].model_context_covariates)


@pytest.mark.parametrize("fault", ["engine", "chronos", "residual", "kalman", "unit", "broadcast", "known", "values", "nuclear", "incumbent_grid", "kalman_upstream"])
def test_invalid_full_chain_or_incumbent_fails_before_any_write(tmp_path, monkeypatch, fault):
    result, incumbent, data = fixture()
    alias = report.SOLAR_ALIASES[-1]
    if fault == "engine":
        result.audit["candidate_engine"] = "residual_only_solar"
    elif fault in ("chronos", "residual", "kalman"):
        key = {"chronos": "solar_chronos_context_columns", "residual": "solar_residual_features", "kalman": "solar_kalman_market_features"}[fault]
        result.audit[key] = []
    elif fault == "unit":
        result.audit["solar_sources"][alias]["unit"] = "MW"
    elif fault == "broadcast":
        result.audit["solar_sources"][alias]["daily_broadcast"] = True
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
    with pytest.raises(ValueError, match="SolarCWE reports must stay"):
        report.render_solar_cwe_reports(result, incumbent=incumbent, data=data, zone="FR", delivery_day="2026-09-08",
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


def test_old_attribution_without_solar_inputs_is_refused_before_write(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    capture(monkeypatch)
    def attach(target, **kwargs):
        target.variable_attribution = {"groups": [{"key": report.NUCLEAR_ALIAS}]}
    monkeypatch.setattr("chronos2_hourly.reporting._attach_variable_attribution", attach)
    with pytest.raises(ValueError, match="Attribution must cover all four"):
        run(tmp_path, monkeypatch, result, incumbent, data, attribution_directory=tmp_path / "old_attribution")
    assert not (tmp_path / "runs").exists()
