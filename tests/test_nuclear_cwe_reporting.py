"""Offline format/source/label parity for the isolated CWE reporting adapter."""
from copy import deepcopy
from dataclasses import asdict
from datetime import timedelta
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nuclear_cwe_reporting as report
from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.variable_attribution import build_variable_groups
from chronos2_modular.common import ZoneData
from chronos2_modular.report import build_statistics_records


def fixture(day="2026-09-11", zone="FR", observed=False):
    tz = {"FR": "Europe/Paris", "BE": "Europe/Brussels", "DE": "Europe/Berlin", "NL": "Europe/Amsterdam"}[zone]
    date = pd.Timestamp(day).date()
    full = pd.date_range(pd.Timestamp(date-timedelta(days=730), tz=tz), pd.Timestamp(date, tz=tz),
                         freq="h", inclusive="left").tz_convert("UTC").rename("delivery_start_utc")
    future = local_delivery_day_index(date, timezone=tz).rename("delivery_start_utc")
    actual = 50.0 + np.sin(np.arange(len(full)) / 24)
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
    incumbent = SimpleNamespace(backtest=kh.copy(), forecast=kf.copy())
    for frame in (incumbent.backtest, incumbent.forecast):
        for q in ("q10", "q50", "q90"):
            frame[f"residual_kalman__{q}"] += .5
    cov = pd.DataFrame({"fr_residual_load_fcst": 30., "fr_nuclear_generation_fcst_gw": 40.,
                        "be_nuclear_available_gw": 3., "nl_nuclear_available_gw": .5}, index=full.append(future))
    target = pd.Series(actual, index=full.tz_convert(tz), name="target")
    if observed:
        target = pd.concat([target, pd.Series(53., index=future.tz_convert(tz), name="target")])
    data = ZoneData(zone=zone, timezone=tz, frequency="h", target=target,
                    covariates=cov.reindex(full), model_context_covariates=cov,
                    known_future_columns=["fr_residual_load_fcst"], coverage=pd.DataFrame(),
                    input_manifest=pd.DataFrame({"alias": ["fr_residual_load_fcst"]}), diagnostics={})
    result = SimpleNamespace(raw_history=history.copy(), residual_statistics=history,
                             source_forecast=forecast, covariates=cov,
                             kalman_view=SimpleNamespace(backtest=kh, forecast=kf),
                             audit={"candidate_engine": "nuclear_cwe_forecast_v1"})
    return result, incumbent, data


def capture(monkeypatch):
    collected = []
    def render(results, config, path):
        collected.extend(results)
        path.write_text('<html><main><h3>Comparaison au modèle prix seul</h3><table><tr><td>obsolete</td></tr></table>'
                        'Covariables natives<p>Prévision opérationnelle du jour</p></main></html>', encoding="utf-8")
    monkeypatch.setattr(report, "write_html_report", render)
    return collected


def run(tmp_path, result, incumbent, data, **kwargs):
    day = str(result.source_forecast.index.tz_convert(data.timezone)[0].date())
    return report.render_nuclear_cwe_reports(result, incumbent=incumbent, data=data, zone=data.zone,
        delivery_day=day, output_directory=tmp_path, source_audit=kwargs.pop("source_audit", {"test": True}), **kwargs)


@pytest.mark.parametrize("zone", ["FR", "BE", "DE", "NL"])
def test_two_new_reports_are_country_independent_and_preserve_inputs(tmp_path, monkeypatch, zone):
    result, incumbent, data = fixture(zone=zone)
    before = deepcopy((result, incumbent, data))
    collected = capture(monkeypatch)
    paths = run(tmp_path, result, incumbent, data)
    assert set(paths) == {"autonomous", "kalman", "audit", "comparison"}
    assert len(collected) == 2
    assert all(f"forecast_{zone.lower()}_2026-09-11_nuclear_cwe_" in paths[key].name for key in ("autonomous", "kalman"))
    for item in collected:
        assert item.zone == zone
        assert item.backtest_baseline is None and item.metrics_baseline is None
        assert not hasattr(item, "kalman_diagnostics")
        assert item.statistics_candidate.tail(24).actual.isna().all()
        assert str(item.zone_data.target.index.tz) == data.timezone
        assert item.zone_data.known_future_columns == data.known_future_columns
        manifest = item.zone_data.input_manifest.set_index("alias")
        assert manifest.loc["fr_nuclear_generation_fcst_gw", "information_type"] == "generation_forecast"
        assert manifest.loc["be_nuclear_available_gw", "information_type"] == "capacity_forecast"
    comparison = json.loads(paths["comparison"].read_text(encoding="utf-8"))
    assert comparison["statistics_start_day"] == "2025-09-11"
    assert comparison["statistics_end_day"] == "2026-09-10"
    assert comparison["annual"]["candidate"]["hours"] == 8760
    assert comparison["annual"]["candidate"]["mae_eur_mwh"] == pytest.approx(.75)
    assert comparison["annual"]["incumbent"]["mae_eur_mwh"] == pytest.approx(1.25)
    assert comparison["annual"]["gain_incumbent_minus_candidate"]["daily_mean_mae_eur_mwh"] == pytest.approx(.5)
    assert comparison["june_24_26"]["candidate"]["hours"] == 72
    assert comparison["june_24_26"]["days"] == ["2026-06-24", "2026-06-25", "2026-06-26"]
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    assert audit["warmup"]["raw_days_before_final365"] == 365
    assert audit["source_audit"] == {"test": True}
    assert all(not value["model_channel_added"] for value in audit["structural_countries"].values())
    assert audit["variable_attribution_status"].startswith("unavailable")
    for key in ("autonomous", "kalman"):
        text = paths[key].read_text(encoding="utf-8")
        assert 'data-report-section="nuclear-cwe-incumbent-comparison"' in text
        assert "obsolete" not in text and "duplication de l'heure d'automne" not in text
        assert "prévision de puissance maximale disponible (Pmax, GW)" in text
        assert "n’est pas certifiée" in text and "Prévision opérationnelle" not in text
    pd.testing.assert_frame_equal(result.residual_statistics, before[0].residual_statistics)
    pd.testing.assert_frame_equal(result.kalman_view.forecast, before[0].kalman_view.forecast)
    pd.testing.assert_frame_equal(incumbent.backtest, before[1].backtest)
    pd.testing.assert_series_equal(data.target, before[2].target)
    pd.testing.assert_frame_equal(data.input_manifest, before[2].input_manifest)


def test_current_observation_moves_exact_statistics_comparison_window(tmp_path, monkeypatch):
    result, incumbent, data = fixture(observed=True)
    collected = capture(monkeypatch)
    paths = run(tmp_path, result, incumbent, data)
    comparison = json.loads(paths["comparison"].read_text(encoding="utf-8"))
    assert comparison["delivery_day_included"] is True
    assert comparison["statistics_start_day"] == "2025-09-12"
    assert comparison["statistics_end_day"] == "2026-09-11"
    for item in collected:
        daily = [row for row in build_statistics_records([item]) if row["sample"] == "daily"]
        assert len(daily) == 365
        assert daily[0]["period_key"] == comparison["statistics_start_day"]
        assert daily[-1]["period_key"] == comparison["statistics_end_day"]
        assert daily[-1]["n"] == 24
    assert comparison["annual"]["candidate"]["mae_eur_mwh"] == pytest.approx((364*.75+2.75)/365)


@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2026-10-25", 25)])
def test_dst_delivery_grid_not_duplicated_or_dropped(tmp_path, monkeypatch, day, hours):
    result, incumbent, data = fixture(day=day, observed=True)
    collected = capture(monkeypatch)
    paths = run(tmp_path, result, incumbent, data)
    for item in collected:
        assert len(item.forecast_native) == hours
        assert item.forecast_native.timestamp.nunique() == hours
    assert json.loads(paths["audit"].read_text())["delivery_day_observed_hours"] == hours


@pytest.mark.parametrize("defect", ["incumbent_hours", "incumbent_actual", "candidate_actual", "missing_covariate", "upstream", "quantile", "partial_actual"])
def test_invalid_pairing_fails_before_writing(tmp_path, monkeypatch, defect):
    result, incumbent, data = fixture()
    collected = capture(monkeypatch)
    if defect == "incumbent_hours": incumbent.backtest = incumbent.backtest.iloc[1:]
    elif defect == "incumbent_actual": incumbent.backtest["actual"] += 1
    elif defect == "candidate_actual": result.kalman_view.backtest["actual"] += 1
    elif defect == "missing_covariate": result.covariates = result.covariates.drop(columns="be_nuclear_available_gw")
    elif defect == "upstream": result.kalman_view.forecast["residual_corrected__q50"] += 1
    elif defect == "quantile": incumbent.forecast["residual_kalman__q10"] = 100
    elif defect == "partial_actual":
        data.target = pd.concat([data.target, pd.Series(53., index=result.source_forecast.index[:1])])
    with pytest.raises(ValueError): run(tmp_path, result, incumbent, data)
    assert not collected and not list(tmp_path.iterdir())


def test_source_audit_is_retained_and_escaped(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    capture(monkeypatch)
    audit = {"be_nuclear_available_gw": {"specification": {"unit": "GW"}, "audit": {"note": "</pre><script>bad()</script>"}}}
    paths = run(tmp_path, result, incumbent, data, source_audit=audit)
    assert json.loads(paths["audit"].read_text(encoding="utf-8"))["source_audit"] == audit
    document = paths["kalman"].read_text(encoding="utf-8")
    assert "<script>bad()" not in document
    assert "&lt;script&gt;bad()" in document


def test_storm_attached_once_to_all_three_exactly_paired_results(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    collected = capture(monkeypatch)
    calls = []
    def attach(prepared, archive, *, zone, timezone):
        calls.append((archive, zone, timezone))
        assert set(prepared) == {"autonomous", "kalman", "incumbent"}
        frames = [item.statistics_candidate for item in prepared.values()]
        for frame in frames[1:]:
            pd.testing.assert_frame_equal(frame[["timestamp", "actual"]], frames[0][["timestamp", "actual"]])
        return {"status": "complete", "exact_snapshot": "tested"}
    monkeypatch.setattr("chronos2_hourly.nuclear_report_benchmark.attach_nuclear_storm", attach)
    paths = run(tmp_path, result, incumbent, data, storm_archive=tmp_path/"frozen_storm")
    assert len(calls) == 1 and calls[0][1:] == ("FR", "Europe/Paris")
    assert all("Storm officiel vérifié" in item.statistics_scope_note for item in collected)
    assert json.loads(paths["audit"].read_text())["storm_hourly_comparison"]["exact_snapshot"] == "tested"


def test_old_fr_only_attribution_is_never_relabelled_cwe(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    collected = capture(monkeypatch)
    def old_attribution(item, **kwargs):
        item.variable_attribution = {"groups": [{"context_columns": ["fr_nuclear_generation_fcst_gw"],
                                                 "future_columns": ["fr_nuclear_generation_fcst_gw"]}]}
    monkeypatch.setattr("chronos2_hourly.reporting._attach_variable_attribution", old_attribution)
    with pytest.raises(ValueError, match="three new nuclear inputs"):
        run(tmp_path, result, incumbent, data, attribution_directory=tmp_path/"old")
    assert not collected and not list(tmp_path.iterdir())


def real_attribution_groups():
    aliases = tuple(report._SOURCE_LABELS)
    known = tuple(f"known_{alias}_oracle" for alias in aliases)
    return [asdict(group) for group in build_variable_groups(
        required_covariates=aliases, context_columns=(*aliases, *known),
        future_columns=known, include_past_prices=True)]


@pytest.mark.parametrize("zone", ["FR", "BE", "NL"])
def test_real_known_future_attribution_and_semantic_labels_survive_report(tmp_path, monkeypatch, zone):
    result, incumbent, data = fixture(zone=zone)
    collected = capture(monkeypatch)
    groups = real_attribution_groups()
    raw = {"groups": groups, "audit": {"groups": deepcopy(groups), "original": True},
           "hourly": pd.DataFrame({"variable_key": [group["key"] for group in groups],
                                   "variable_label": [group["label"] for group in groups]})}
    original = deepcopy(raw)
    def attach(item, **kwargs):
        item.variable_attribution = raw
    monkeypatch.setattr("chronos2_hourly.reporting._attach_variable_attribution", attach)
    paths = run(tmp_path, result, incumbent, data, attribution_directory=tmp_path/"new_attribution")
    autonomous, kalman = collected
    for item in collected:
        attribution = item.variable_attribution
        assert attribution["audit"] == original["audit"]
        labels = attribution["hourly"].set_index("variable_key").variable_label
        assert labels["be_nuclear_available_gw"] == "Pmax nucléaire disponible prévue BE (GW)"
        assert labels["nl_nuclear_available_gw"] == "Pmax nucléaire disponible prévue NL (GW)"
        assert labels["fr_nuclear_generation_fcst_gw"] == "Génération nucléaire prévue FR (GW)"
        assert "Prix passés" in labels["historical_target_price"]
        assert attribution["groups"][1]["future_columns"] == ("known_be_nuclear_available_gw_oracle",)
    assert kalman.variable_attribution["is_upstream_attribution"] is True
    assert kalman.variable_attribution["explained_model_label"] == report.CWE_AUTONOMOUS_LABEL
    assert kalman.variable_attribution["reported_model_label"] == report.CWE_KALMAN_LABEL
    assert autonomous.variable_attribution["reported_model_label"] == report.CWE_AUTONOMOUS_LABEL
    assert json.loads(paths["audit"].read_text())["variable_attribution_status"] == "verified_new_forecast_artifact"
    assert raw["groups"] == original["groups"] and raw["audit"] == original["audit"]
    pd.testing.assert_frame_equal(raw["hourly"], original["hourly"])


@pytest.mark.parametrize("defect", ["missing_nl", "wrong_group", "wrong_future_channel", "missing_context", "duplicate_group"])
def test_known_channel_acceptance_does_not_accept_unrelated_or_missing_sources(defect):
    groups = real_attribution_groups()
    if defect == "missing_nl": groups = groups[:2]
    elif defect == "wrong_group": groups[1]["key"] = "another_variable"
    elif defect == "wrong_future_channel": groups[1]["future_columns"] = ("known_nl_nuclear_available_gw_oracle",)
    elif defect == "missing_context": groups[1]["context_columns"] = ()
    elif defect == "duplicate_group": groups.append(deepcopy(groups[1]))
    with pytest.raises(ValueError, match="three new nuclear inputs"):
        report._cwe_attribution({"groups": groups})


def test_report_context_preserves_exact_declared_oracle_channels_without_raw_future_invention(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    capture(monkeypatch)
    for alias in report._SOURCE_LABELS:
        data.model_context_covariates[f"known_{alias}_oracle"] = data.model_context_covariates[alias]
    data.known_future_columns = [f"known_{alias}_oracle" for alias in report._SOURCE_LABELS]
    before = data.model_context_covariates.copy(deep=True)
    raw_covariates = result.covariates.copy(deep=True)
    copied = report._report_data(data, raw_covariates)
    assert copied.known_future_columns == data.known_future_columns
    assert not set(report._SOURCE_LABELS).intersection(copied.known_future_columns)
    for column in data.known_future_columns:
        pd.testing.assert_series_equal(copied.model_context_covariates[column],
                                        before[column].reindex(copied.model_context_covariates.index))
    pd.testing.assert_frame_equal(data.model_context_covariates, before)
    pd.testing.assert_frame_equal(result.covariates, raw_covariates)


def test_standard_renderer_retains_night_statistics_calendar_and_explanations(tmp_path):
    result, incumbent, data = fixture(zone="NL")
    paths = run(tmp_path, result, incumbent, data)
    for key in ("autonomous", "kalman"):
        document = paths[key].read_text(encoding="utf-8")
        for marker in ('data-report-section="average-prices"', 'data-report-section="statistics"',
                       'data-report-section="forecast-components"', 'data-report-section="attribution-methodology"',
                       'chronos2-theme-change', 'calendar', 'Plotly.newPlot', 'Performance par heure locale'):
            assert marker in document
        assert "Attribution chiffrée indisponible" in document
        assert "Prix passés" in document or "prix passés" in document
        assert '<h2>NL</h2>' in document and '<h2>FR</h2>' not in document
