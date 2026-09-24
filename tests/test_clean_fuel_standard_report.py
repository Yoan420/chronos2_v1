"""Full-chain clean-fuel reports keep the operational renderer and safeguards."""
from copy import deepcopy
from datetime import timedelta
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_modular.common import ZoneData
from chronos2_modular.report import build_statistics_records
from nyx_clean_fuel import standard_report as report


def fixture(day="2026-09-18", zone="DE", observed=False):
    tz = {"FR": "Europe/Paris", "BE": "Europe/Brussels", "DE": "Europe/Berlin", "NL": "Europe/Amsterdam"}[zone]
    date = pd.Timestamp(day).date()
    full = pd.date_range(pd.Timestamp(date-timedelta(days=730), tz=tz), pd.Timestamp(date, tz=tz),
                         freq="h", inclusive="left").tz_convert("UTC").rename("delivery_start_utc")
    future = local_delivery_day_index(date, timezone=tz).rename("delivery_start_utc")
    actual = 50 + np.sin(np.arange(len(full)) / 24)
    history = pd.DataFrame({"actual": actual, "forecast_origin_utc": full[0]-pd.Timedelta(days=1)}, index=full)
    forecast = pd.DataFrame({"forecast_origin_utc": future[0]-pd.Timedelta(hours=16)}, index=future)
    for q, offset in (("q10", -5), ("q50", 0), ("q90", 5)):
        history[f"chronos2__{q}"] = actual + 2 + offset
        history[f"residual_corrected__{q}"] = actual + 1 + offset
        forecast[f"chronos2__{q}"] = 57 + offset
        forecast[f"residual_corrected__{q}"] = 56 + offset
    cutoff = pd.Timestamp(date-timedelta(days=365), tz=tz).tz_convert("UTC")
    kh, kf = history.loc[history.index >= cutoff].copy(), forecast.copy()
    for q in ("q10", "q50", "q90"):
        kh[f"residual_kalman__{q}"] = kh[f"residual_corrected__{q}"] - .25
        kf[f"residual_kalman__{q}"] = kf[f"residual_corrected__{q}"] - .25
    cov = pd.DataFrame({"fr_residual_load_fcst": 30., "fr_nuclear_generation_fcst_gw": 40.,
                        **{alias: 60.+i for i, alias in enumerate(report.FUEL_ALIASES)}}, index=full.append(future))
    context = cov.copy()
    for alias in cov:
        context[f"known_{alias}_oracle"] = context[alias]
    target = pd.Series(actual, index=full.tz_convert(tz), name="target")
    if observed:
        target = pd.concat([target, pd.Series(53., index=future.tz_convert(tz), name="target")])
    data = ZoneData(zone=zone, timezone=tz, frequency="h", target=target,
                    covariates=cov.reindex(full), model_context_covariates=context,
                    known_future_columns=[f"known_{alias}_oracle" for alias in cov], coverage=pd.DataFrame(),
                    input_manifest=pd.DataFrame({"alias": ["fr_residual_load_fcst"]}), diagnostics={})
    result = SimpleNamespace(raw_history=history.copy(), residual_statistics=history,
                             source_forecast=forecast, covariates=cov,
                             kalman_view=SimpleNamespace(backtest=kh, forecast=kf),
                             audit={"candidate_engine": report.ENGINE})
    incumbent = deepcopy(result)
    incumbent.audit = {"engine": "nuclear_forecast_v1"}
    for frame in (incumbent.residual_statistics, incumbent.source_forecast):
        for q in ("q10", "q50", "q90"):
            frame[f"residual_corrected__{q}"] += .5
    for frame in (incumbent.kalman_view.backtest, incumbent.kalman_view.forecast):
        for q in ("q10", "q50", "q90"):
            frame[f"residual_kalman__{q}"] += .5
    return result, incumbent, data


def capture(monkeypatch):
    collected = []
    def render(results, config, path):
        collected.extend(results)
        path.write_text('<html><main><h3>Comparaison au modèle prix seul</h3><table><tr><td>obsolete</td></tr></table>'
                        'Covariables natives<p>Prévision opérationnelle du jour</p></main></html>', encoding="utf-8")
    monkeypatch.setattr(report, "write_html_report", render)
    return collected


def run(tmp_path, monkeypatch, result, incumbent, data, **kwargs):
    monkeypatch.setattr(report, "PROJECT_ROOT", tmp_path)
    day = str(result.source_forecast.index.tz_convert(data.timezone)[0].date())
    directory = tmp_path / "runs/experiments/nyx_clean_fuel_full_v1/reports"
    return report.render_clean_fuel_reports(result, incumbent=incumbent, data=data, zone=data.zone,
        delivery_day=day, output_directory=directory, **kwargs)


@pytest.mark.parametrize("zone", ["FR", "DE", "BE", "NL"])
def test_same_family_reports_and_inputs_are_preserved(tmp_path, monkeypatch, zone):
    result, incumbent, data = fixture(zone=zone)
    before = deepcopy((result, incumbent, data))
    collected = capture(monkeypatch)
    paths = run(tmp_path, monkeypatch, result, incumbent, data)
    assert set(paths) == {"autonomous", "kalman", "audit", "comparison"}
    assert len(collected) == 2
    comparisons = json.loads(paths["comparison"].read_text(encoding="utf-8"))
    for family in ("autonomous", "kalman"):
        comparison = comparisons[family]
        assert comparison["candidate_model"] == f"clean_fuel_{family}"
        assert comparison["incumbent_model"] == f"nuclear_{family}"
        assert comparison["annual"]["gain_incumbent_minus_candidate"]["mae_eur_mwh"] == pytest.approx(.5)
        assert comparison["annual"]["candidate"]["hours"] == 8760
        text = paths[family].read_text(encoding="utf-8")
        assert 'data-report-section="clean-fuel-methodology"' in text
        assert "CGC/CCC" in text and "obsolete" not in text
        assert "Attribution chiffrée indisponible" in text
    for item in collected:
        assert item.zone == zone
        assert item.statistics_candidate.tail(24).actual.isna().all()
        assert not hasattr(item, "variable_attribution")
        manifest = item.zone_data.input_manifest.set_index("alias")
        assert manifest.loc["ccc", "unit"] == "EUR/MWh_e"
        for alias in report.FUEL_ALIASES:
            assert f"known_{alias}_oracle" in item.zone_data.known_future_columns
    pd.testing.assert_frame_equal(result.source_forecast, before[0].source_forecast)
    pd.testing.assert_frame_equal(result.kalman_view.forecast, before[0].kalman_view.forecast)
    pd.testing.assert_frame_equal(incumbent.residual_statistics, before[1].residual_statistics)
    pd.testing.assert_frame_equal(data.model_context_covariates, before[2].model_context_covariates)
    pd.testing.assert_frame_equal(data.input_manifest, before[2].input_manifest)


def test_observed_delivery_advances_statistics_only(tmp_path, monkeypatch):
    result, incumbent, data = fixture(observed=True)
    collected = capture(monkeypatch)
    paths = run(tmp_path, monkeypatch, result, incumbent, data)
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    assert audit["evaluation_start_day"] == "2025-09-18"
    assert audit["evaluation_end_day"] == "2026-09-17"
    assert audit["statistics_start_day"] == "2025-09-19"
    assert audit["statistics_end_day"] == "2026-09-18"
    assert audit["delivery_day_observed_hours"] == 24
    for item in collected:
        daily = [row for row in build_statistics_records([item]) if row["sample"] == "daily"]
        assert len(daily) == 365
        assert daily[-1]["period_key"] == "2026-09-18"


@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2026-10-25", 25)])
def test_dst_days_preserved(tmp_path, monkeypatch, day, hours):
    result, incumbent, data = fixture(day=day, observed=True)
    collected = capture(monkeypatch)
    paths = run(tmp_path, monkeypatch, result, incumbent, data)
    assert json.loads(paths["audit"].read_text())["delivery_day_observed_hours"] == hours
    assert all(len(item.forecast_native) == hours for item in collected)


@pytest.mark.parametrize("defect", ["old_engine", "fuel_missing", "known_missing", "divergent_input", "hours", "upstream", "partial_actual"])
def test_invalid_inputs_fail_before_publication(tmp_path, monkeypatch, defect):
    result, incumbent, data = fixture()
    collected = capture(monkeypatch)
    if defect == "old_engine": result.audit = {"engine": "nuclear_forecast_v1"}
    elif defect == "fuel_missing": result.covariates = result.covariates.drop(columns="ccc")
    elif defect == "known_missing": data.known_future_columns.remove("known_ccc_oracle")
    elif defect == "divergent_input": data.model_context_covariates["known_ccc_oracle"] += 1
    elif defect == "hours": result.kalman_view.backtest = result.kalman_view.backtest.iloc[1:]
    elif defect == "upstream": result.kalman_view.forecast["residual_corrected__q50"] += 1
    elif defect == "partial_actual": data.target = pd.concat([data.target, pd.Series(53., index=result.source_forecast.index[:1])])
    with pytest.raises(ValueError):
        run(tmp_path, monkeypatch, result, incumbent, data)
    assert not collected
    assert not list(tmp_path.rglob("*.html"))


def test_output_cannot_publish_operational_exports(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    monkeypatch.setattr(report, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="must remain inside"):
        report.render_clean_fuel_reports(result, incumbent=incumbent, data=data, zone=data.zone,
            delivery_day="2026-09-18", output_directory=tmp_path / "runs/exports")
    assert not list(tmp_path.iterdir())


def test_storm_attached_to_same_physical_observations(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    collected = capture(monkeypatch)
    calls = []
    def attach(prepared, archive, *, zone, timezone):
        calls.append(archive)
        assert set(prepared) == {"autonomous", "kalman", "incumbent_autonomous", "incumbent_kalman"}
        frames = [item.statistics_candidate for item in prepared.values()]
        for frame in frames[1:]:
            pd.testing.assert_frame_equal(frame[["timestamp", "actual"]], frames[0][["timestamp", "actual"]])
        return {"status": "complete"}
    monkeypatch.setattr("chronos2_hourly.nuclear_report_benchmark.attach_nuclear_storm", attach)
    run(tmp_path, monkeypatch, result, incumbent, data, storm_archive=tmp_path / "storm")
    assert len(calls) == 1
    assert all("Storm officiel vérifié" in item.statistics_scope_note for item in collected)


def test_old_attribution_cannot_be_relabelled(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    collected = capture(monkeypatch)
    def attach(item, **kwargs):
        item.variable_attribution = {"groups": [{"key": "fr_nuclear_generation_fcst_gw",
            "context_columns": ["fr_nuclear_generation_fcst_gw"], "future_columns": ["known_fr_nuclear_generation_fcst_gw_oracle"]}]}
    monkeypatch.setattr("chronos2_hourly.reporting._attach_variable_attribution", attach)
    with pytest.raises(ValueError, match="recalculated"):
        run(tmp_path, monkeypatch, result, incumbent, data, attribution_directory=tmp_path / "old")
    assert not collected


def test_new_attribution_explicitly_stays_upstream_of_kalman(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    collected = capture(monkeypatch)
    raw = {"groups": [{"key": alias, "label": alias,
                       "context_columns": [alias, f"known_{alias}_oracle"],
                       "future_columns": [f"known_{alias}_oracle"]} for alias in report.FUEL_ALIASES],
           "hourly": pd.DataFrame({"variable_key": report.FUEL_ALIASES,
                                    "variable_label": report.FUEL_ALIASES})}
    before = deepcopy(raw)
    def attach(item, **kwargs):
        item.variable_attribution = raw
    monkeypatch.setattr("chronos2_hourly.reporting._attach_variable_attribution", attach)
    paths = run(tmp_path, monkeypatch, result, incumbent, data, attribution_directory=tmp_path / "new")
    autonomous, kalman = collected
    assert autonomous.variable_attribution["reported_model_label"] == report.CLEAN_FUEL_AUTONOMOUS_LABEL
    assert kalman.variable_attribution["is_upstream_attribution"] is True
    assert kalman.variable_attribution["explained_model_label"] == report.CLEAN_FUEL_AUTONOMOUS_LABEL
    assert kalman.variable_attribution["reported_model_label"] == report.CLEAN_FUEL_KALMAN_LABEL
    text = paths["kalman"].read_text(encoding="utf-8")
    assert "le décalage Kalman est tenu fixe" in text
    assert "ne mesure donc pas l’influence totale" in text
    assert raw["groups"] == before["groups"]
    pd.testing.assert_frame_equal(raw["hourly"], before["hourly"])


def test_source_audit_is_escaped(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    capture(monkeypatch)
    paths = run(tmp_path, monkeypatch, result, incumbent, data,
                source_audit={"note": "</pre><script>unsafe()</script>"})
    text = paths["kalman"].read_text(encoding="utf-8")
    assert "<script>unsafe()" not in text
    assert "&lt;script&gt;unsafe()" in text


def test_real_renderer_keeps_requested_operational_sections(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    paths = run(tmp_path, monkeypatch, result, incumbent, data)
    for family in ("autonomous", "kalman"):
        text = paths[family].read_text(encoding="utf-8")
        for marker in ('data-report-section="average-prices"', 'data-report-section="statistics"',
                       'data-report-section="forecast-components"', 'data-report-section="attribution-methodology"',
                       'chronos2-theme-change', 'calendar', 'Plotly.newPlot', 'Performance par heure locale',
                       'Backtest et probabilités', 'Variables d’entrée'):
            assert marker in text
        assert "CGC/CCC" in text
        assert '<h2>DE</h2>' in text and '<h2>FR</h2>' not in text
