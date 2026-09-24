"""Offline report parity and forecast-defined temperature regime pairing."""
from copy import deepcopy
from datetime import timedelta
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import heatwave_reporting as report
from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_modular.common import ZoneData
from chronos2_modular.report import build_statistics_records


def fixture(day="2026-09-11", zone="FR", observed=False):
    tz = {"FR": "Europe/Paris", "BE": "Europe/Brussels", "DE": "Europe/Berlin", "NL": "Europe/Amsterdam"}[zone]
    date = pd.Timestamp(day).date()
    full = pd.date_range(pd.Timestamp(date-timedelta(days=730), tz=tz), pd.Timestamp(date, tz=tz),
                         freq="h", inclusive="left").tz_convert("UTC").rename("delivery_start_utc")
    future = local_delivery_day_index(date, timezone=tz).rename("delivery_start_utc")
    actual = 50. + np.sin(np.arange(len(full)) / 24)
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
    cov = pd.DataFrame({"fr_residual_load_fcst": 30., "fr_nuclear_generation_fcst_gw": 40.}, index=full.append(future))
    # Seven completely forecast-defined summer days. No observed error enters
    # the definition, and three named June days deliberately are not selected.
    heat = np.array([str(value) >= "2026-07-01" and str(value) <= "2026-07-07"
                     for value in cov.index.tz_convert(tz).date])
    for country in report.COUNTRIES:
        cov[f"{country.lower()}_temperature_fcst"] = np.where(heat, 29., 16.)
        cov[f"{country.lower()}_heat_excess_fcst_c"] = np.where(heat, 5., 0.)
        cov[f"{country.lower()}_heat_streak_fcst_days"] = np.where(heat, 3., 0.)
    cov["heat_fraction_fcst"] = heat.astype(float)
    cov["cooling_degree_mean_fcst_c"] = np.where(heat, 7., 0.)
    target = pd.Series(actual, index=full.tz_convert(tz), name="target")
    if observed:
        target = pd.concat([target, pd.Series(53., index=future.tz_convert(tz), name="target")])
    data = ZoneData(zone=zone, timezone=tz, frequency="h", target=target,
                    covariates=cov.reindex(full), model_context_covariates=cov.copy(),
                    known_future_columns=["fr_residual_load_fcst"], coverage=pd.DataFrame(),
                    input_manifest=pd.DataFrame({"alias": ["fr_residual_load_fcst"]}), diagnostics={})
    result = SimpleNamespace(raw_history=history.copy(), residual_statistics=history,
                             source_forecast=forecast, covariates=cov,
                             kalman_view=SimpleNamespace(backtest=kh, forecast=kf),
                             audit={"candidate_engine": "heatwave_forecast_v1", "heatwave_base_variant": "nuclear_fr"})
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
    return report.render_heatwave_reports(result, incumbent=incumbent, data=data, zone=data.zone,
        delivery_day=day, output_directory=tmp_path, source_audit=kwargs.pop("source_audit", {"test": True}), **kwargs)


@pytest.mark.parametrize("zone", ["FR", "DE", "BE", "NL"])
def test_independent_reports_keep_standard_frames_and_inputs(tmp_path, monkeypatch, zone):
    result, incumbent, data = fixture(zone=zone)
    before = deepcopy((result, incumbent, data))
    collected = capture(monkeypatch)
    paths = run(tmp_path, result, incumbent, data)
    assert set(paths) == {"autonomous", "kalman", "audit", "comparison"}
    assert len(collected) == 2
    assert all(f"forecast_{zone.lower()}_2026-09-11_heatwave_" in paths[key].name for key in ("autonomous", "kalman"))
    for item in collected:
        assert item.statistics_candidate.tail(24).actual.isna().all()
        assert item.zone_data.known_future_columns == data.known_future_columns
        manifest = item.zone_data.input_manifest.set_index("alias")
        assert manifest.loc["be_temperature_fcst", "information_type"] == "daily_temperature_forecast_index"
        assert manifest.loc["be_heat_excess_fcst_c", "information_type"] == "causal_forecast_derived_index"
    comparison = json.loads(paths["comparison"].read_text(encoding="utf-8"))
    assert comparison["statistics_start_day"] == "2025-09-11"
    assert comparison["statistics_end_day"] == "2026-09-10"
    annual = comparison["variants"]["kalman"]["annual"]["full_support"]
    assert annual["candidate"]["hours"] == 8760
    assert annual["candidate"]["mae_eur_mwh"] == pytest.approx(.75)
    assert annual["incumbent"]["mae_eur_mwh"] == pytest.approx(1.25)
    assert annual["gain_incumbent_minus_candidate"]["mae_eur_mwh"] == pytest.approx(.5)
    assert comparison["variants"]["autonomous"]["annual"]["full_support"]["candidate"]["mae_eur_mwh"] == pytest.approx(1.)
    assert comparison["variants"]["kalman"]["annual"]["storm_paired"]["storm"]["mae_eur_mwh"] is None
    assert comparison["variants"]["kalman"]["annual"]["unpaired_storm_hours"] == 8760
    assert comparison["heat_partition"]["probable_heatwave_days"] == 7
    assert comparison["heat_partition"]["other_days"] == 358
    for variant in comparison["variants"].values():
        heat, normal = variant["probable_heatwave"]["full_support"], variant["other_days"]["full_support"]
        assert heat["candidate"]["hours"] + normal["candidate"]["hours"] == annual["candidate"]["hours"]
        assert variant["june_24_26_2026"]["full_support"]["candidate"]["hours"] == 72
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    assert audit["warmup"]["raw_days_before_final365"] == 365
    assert audit["source_audit"] == {"test": True}
    for key in ("autonomous", "kalman"):
        text = paths[key].read_text(encoding="utf-8")
        assert 'data-report-section="heatwave-regime-comparison"' in text
        assert "obsolete" not in text and "Prévision opérationnelle" not in text
        assert "indices journaliers de température" in text and "ni des Tmax" in text
    pd.testing.assert_frame_equal(result.residual_statistics, before[0].residual_statistics)
    pd.testing.assert_frame_equal(result.covariates, before[0].covariates)
    pd.testing.assert_frame_equal(incumbent.backtest, before[1].backtest)
    pd.testing.assert_series_equal(data.target, before[2].target)
    pd.testing.assert_frame_equal(data.input_manifest, before[2].input_manifest)


def test_current_observation_moves_the_same_statistics_window(tmp_path, monkeypatch):
    result, incumbent, data = fixture(observed=True)
    collected = capture(monkeypatch)
    paths = run(tmp_path, result, incumbent, data)
    comparison = json.loads(paths["comparison"].read_text())
    assert comparison["delivery_day_included"] is True
    assert comparison["statistics_start_day"] == "2025-09-12"
    assert comparison["statistics_end_day"] == "2026-09-11"
    for item in collected:
        daily = [row for row in build_statistics_records([item]) if row["sample"] == "daily"]
        assert len(daily) == 365
        assert daily[0]["period_key"] == comparison["statistics_start_day"]
        assert daily[-1]["period_key"] == comparison["statistics_end_day"]


@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2026-10-25", 25)])
def test_dst_hours_kept_in_forecast_and_partition(tmp_path, monkeypatch, day, hours):
    result, incumbent, data = fixture(day=day, observed=True)
    collected = capture(monkeypatch)
    paths = run(tmp_path, result, incumbent, data)
    for item in collected:
        assert len(item.forecast_native) == hours
        assert item.forecast_native.timestamp.nunique() == hours
    assert json.loads(paths["audit"].read_text())["delivery_day_observed_hours"] == hours
    comparison = json.loads(paths["comparison"].read_text())
    for variant in comparison["variants"].values():
        total = sum(variant[key]["full_support"]["candidate"]["hours"] for key in ("probable_heatwave", "other_days"))
        assert total == variant["annual"]["full_support"]["candidate"]["hours"]


@pytest.mark.parametrize("defect", ["incumbent_hours", "incumbent_actual", "candidate_actual", "missing_temperature",
                                   "missing_fraction", "fraction_mismatch", "upstream", "quantile", "partial_actual",
                                   "hourly_not_daily", "negative_excess", "invalid_streak"])
def test_invalid_data_fails_before_writing(tmp_path, monkeypatch, defect):
    result, incumbent, data = fixture()
    collected = capture(monkeypatch)
    at = result.kalman_view.backtest.index[0]
    if defect == "incumbent_hours": incumbent.backtest = incumbent.backtest.iloc[1:]
    elif defect == "incumbent_actual": incumbent.backtest["actual"] += 1
    elif defect == "candidate_actual": result.kalman_view.backtest["actual"] += 1
    elif defect == "missing_temperature": result.covariates = result.covariates.drop(columns="nl_temperature_fcst")
    elif defect == "missing_fraction": result.covariates = result.covariates.drop(columns="heat_fraction_fcst")
    elif defect == "fraction_mismatch": result.covariates["heat_fraction_fcst"] = .6
    elif defect == "upstream": result.kalman_view.forecast["residual_corrected__q50"] += 1
    elif defect == "quantile": incumbent.forecast["residual_kalman__q10"] = 100
    elif defect == "partial_actual": data.target = pd.concat([data.target, pd.Series(53., index=result.source_forecast.index[:1])])
    elif defect == "hourly_not_daily": result.covariates.loc[at, "fr_temperature_fcst"] += .5
    elif defect == "negative_excess": result.covariates["fr_heat_excess_fcst_c"] = -1
    elif defect == "invalid_streak": result.covariates["fr_heat_streak_fcst_days"] = 8
    with pytest.raises(ValueError): run(tmp_path, result, incumbent, data)
    assert not collected and not list(tmp_path.iterdir())


def test_storm_uses_identical_mask_and_never_changes_full_support(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    capture(monkeypatch)
    calls = []
    def attach(prepared, archive, *, zone, timezone):
        calls.append(archive)
        for item in prepared.values():
            benchmark = item.statistics_candidate[["timestamp", "actual"]].copy()
            benchmark["q50"] = benchmark.actual + 2.
            heat = pd.DatetimeIndex(benchmark.timestamp).tz_convert(timezone).strftime("%Y-%m-%d") == "2026-07-01"
            benchmark.loc[heat, "q50"] = np.nan
            item.statistics_benchmark = benchmark
        return {"status": "complete", "used_for_prediction": False}
    monkeypatch.setattr("chronos2_hourly.nuclear_report_benchmark.attach_nuclear_storm", attach)
    paths = run(tmp_path, result, incumbent, data, storm_archive=tmp_path/"frozen")
    assert len(calls) == 1
    comp = json.loads(paths["comparison"].read_text())
    for variant in comp["variants"].values():
        annual = variant["annual"]
        assert annual["full_support"]["candidate"]["hours"] == 8760
        assert annual["storm_paired"]["candidate"]["hours"] == 8736
        assert annual["storm_paired"]["incumbent"]["hours"] == 8736
        assert annual["storm_paired"]["storm"]["hours"] == 8736
        assert annual["storm_paired"]["storm"]["mae_eur_mwh"] == pytest.approx(2.)
        heat = variant["probable_heatwave"]
        assert heat["full_support"]["candidate"]["days"] == 7
        assert heat["storm_paired"]["candidate"]["days"] == 6
        assert heat["unpaired_storm_hours"] == 24


def test_no_heat_days_stays_empty_not_zero_mae(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    capture(monkeypatch)
    for country in report.COUNTRIES:
        result.covariates[f"{country.lower()}_heat_streak_fcst_days"] = 0
    result.covariates["heat_fraction_fcst"] = 0
    paths = run(tmp_path, result, incumbent, data)
    comp = json.loads(paths["comparison"].read_text())
    assert comp["heat_partition"]["probable_heatwave_days"] == 0
    assert comp["variants"]["kalman"]["probable_heatwave"]["full_support"]["candidate"]["mae_eur_mwh"] is None


def test_nondefault_persistence_and_optional_cooling_are_honoured(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    collected = capture(monkeypatch)
    result.audit["heatwave_feature_config"] = {"persistent_days": 5, "streak_clip_days": 10,
                                               "include_cooling_mean": False}
    # A three-day streak no longer meets this configured definition.
    result.covariates["heat_fraction_fcst"] = 0.
    result.covariates = result.covariates.drop(columns="cooling_degree_mean_fcst_c")
    paths = run(tmp_path, result, incumbent, data)
    comparison = json.loads(paths["comparison"].read_text())
    assert comparison["heat_partition"]["probable_heatwave_days"] == 0
    assert comparison["heat_partition"]["persistent_days"] == 5
    assert comparison["heat_partition"]["streak_clip_days"] == 10
    assert "atteint 5 journées" in paths["kalman"].read_text(encoding="utf-8")
    manifest = collected[0].zone_data.input_manifest.set_index("alias")
    assert "5 jours" in manifest.loc["heat_fraction_fcst", "source"]


def test_conflicting_feature_and_engine_parameters_are_refused(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    collected = capture(monkeypatch)
    result.audit["heatwave_feature_config"] = {"persistent_days": 3}
    with pytest.raises(ValueError, match="disagree on heatwave parameters"):
        run(tmp_path, result, incumbent, data, feature_audit={"feature_config": {"persistent_days": 4}})
    assert not collected and not list(tmp_path.iterdir())


def test_cooling_values_match_declared_threshold(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    capture(monkeypatch)
    result.audit["heatwave_feature_config"] = {"cooling_threshold_c": 25.}
    with pytest.raises(ValueError):
        run(tmp_path, result, incumbent, data)
    result.covariates["cooling_degree_mean_fcst_c"] = (
        result.covariates[list(report.TEMPERATURE_ALIASES)] - 25.).clip(lower=0).mean(axis=1)
    assert run(tmp_path, result, incumbent, data)["kalman"].is_file()


@pytest.mark.parametrize("persistent_countries", [1, 2, 3, 4])
def test_float32_derived_aggregates_use_bounded_rounding(persistent_countries):
    result, _, data = fixture()
    frame = result.covariates.iloc[:24].copy()
    temperature = [26.1234567, 25.5432167, 24.3456789, 27.2345678, 23.3456789]
    for i, country in enumerate(report.COUNTRIES):
        frame[f"{country.lower()}_temperature_fcst"] = temperature[i]
        frame[f"{country.lower()}_heat_streak_fcst_days"] = 3. if i < persistent_countries else 0.
    frame["heat_fraction_fcst"] = persistent_countries / 5.
    frame["cooling_degree_mean_fcst_c"] = np.mean(np.maximum(np.asarray(temperature) - 22., 0.))
    frame = frame.astype(np.float32)
    assert float(frame["heat_fraction_fcst"].iloc[0]) != persistent_countries / 5.
    checked = report._feature_frame(frame, frame.index, data.timezone)
    assert (checked["heat_fraction_fcst"] == persistent_countries / 5.).all()
    # The report does not mutate the rounded inputs used by the model.
    assert frame["heat_fraction_fcst"].dtype == np.float32
    assert float(frame["heat_fraction_fcst"].iloc[0]) != persistent_countries / 5.
    for column in ("heat_fraction_fcst", "cooling_degree_mean_fcst_c"):
        inconsistent = frame.copy()
        inconsistent[column] += np.float32(.001)
        with pytest.raises(ValueError):
            report._feature_frame(inconsistent, inconsistent.index, data.timezone)


def test_exact_zero_float32_fraction_does_not_become_a_heat_day():
    result, _, data = fixture()
    frame = result.covariates.iloc[:24].astype(np.float32)
    checked = report._feature_frame(frame, frame.index, data.timezone)
    assert checked["heat_fraction_fcst"].eq(0).all()


def test_non_heat_forecast_errors_cannot_change_episode_selection(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    capture(monkeypatch)
    result.residual_statistics.loc[:, "residual_corrected__q50"] += .2
    result.kalman_view.backtest.loc[:, "residual_kalman__q50"] += .2
    paths = run(tmp_path, result, incumbent, data)
    comp = json.loads(paths["comparison"].read_text())
    expected = [f"2026-07-0{day}" for day in range(1, 8)]
    assert comp["variants"]["kalman"]["probable_heatwave"]["selected_days"] == expected


def test_audits_escaped_and_correct_incumbent_base_label(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    result.audit.update(heatwave_base_variant="nuclear_cwe", incumbent_label="Référence CWE <script>bad()</script>")
    capture(monkeypatch)
    paths = run(tmp_path, result, incumbent, data,
                source_audit={"note": "</pre><script>bad()</script>"}, feature_audit={"causal": True})
    document = paths["kalman"].read_text(encoding="utf-8")
    assert "<script>bad()" not in document
    assert "&lt;script&gt;bad()" in document
    assert "nucléaire CWE + températures Europe" in document
    assert json.loads(paths["audit"].read_text())["feature_audit"] == {"causal": True}


def test_oracle_feature_channels_preserve_exact_identity(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    capture(monkeypatch)
    names = list(report._source_labels())
    for alias in names:
        data.model_context_covariates[f"known_{alias}_oracle"] = data.model_context_covariates[alias]
    data.known_future_columns = [f"known_{alias}_oracle" for alias in names]
    result.covariates = result.covariates.rename(columns={alias: f"known_{alias}_oracle" for alias in names})
    paths = run(tmp_path, result, incumbent, data)
    assert paths["kalman"].is_file()
    copied = report._report_data(data, result.covariates)
    assert copied.known_future_columns == data.known_future_columns
    assert not set(names).intersection(copied.known_future_columns)


def test_old_nuclear_attribution_rejected(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    collected = capture(monkeypatch)
    def attach(item, **kwargs):
        item.variable_attribution = {"groups": [{"key": "fr_nuclear_generation_fcst_gw",
                                                "context_columns": ["fr_nuclear_generation_fcst_gw"],
                                                "future_columns": ["known_fr_nuclear_generation_fcst_gw_oracle"]}]}
    monkeypatch.setattr("chronos2_hourly.reporting._attach_variable_attribution", attach)
    with pytest.raises(ValueError, match="new input"):
        run(tmp_path, result, incumbent, data, attribution_directory=tmp_path/"old")
    assert not collected and not list(tmp_path.iterdir())


def test_matching_attribution_all_new_inputs_is_relabelled_only(tmp_path, monkeypatch):
    result, incumbent, data = fixture()
    collected = capture(monkeypatch)
    groups = [{"key": alias, "label": alias, "context_columns": [alias],
               "future_columns": [f"known_{alias}_oracle"]} for alias in report._source_labels()]
    raw = {"groups": groups, "audit": {"original": True}, "hourly": pd.DataFrame({
        "variable_key": list(report._source_labels()), "variable_label": list(report._source_labels())})}
    before = deepcopy(raw)
    def attach(item, **kwargs): item.variable_attribution = raw
    monkeypatch.setattr("chronos2_hourly.reporting._attach_variable_attribution", attach)
    run(tmp_path, result, incumbent, data, attribution_directory=tmp_path/"new")
    assert collected[1].variable_attribution["is_upstream_attribution"] is True
    assert collected[0].variable_attribution["groups"][0]["label"] == "Indice journalier de température prévu FR"
    assert raw["groups"] == before["groups"]
    pd.testing.assert_frame_equal(raw["hourly"], before["hourly"])


def test_standard_html_keeps_all_requested_existing_sections(tmp_path):
    result, incumbent, data = fixture(zone="NL")
    paths = run(tmp_path, result, incumbent, data)
    for variant in ("autonomous", "kalman"):
        document = paths[variant].read_text(encoding="utf-8")
        for marker in ('data-report-section="average-prices"', 'data-report-section="statistics"',
                       'data-report-section="forecast-components"', 'data-report-section="attribution-methodology"',
                       'chronos2-theme-change', 'calendar', 'Plotly.newPlot', 'Performance par heure locale',
                       'data-report-section="heatwave-methodology"', 'data-report-section="heatwave-regime-comparison"'):
            assert marker in document
        assert "Attribution chiffrée indisponible" in document
        assert '<h2>NL</h2>' in document and '<h2>FR</h2>' not in document
