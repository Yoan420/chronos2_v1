"""Frozen synthetic nuclear histories exercise isolated three-way solar reports."""
from copy import deepcopy
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import solar_correction_reporting as report
from chronos2_modular.report import build_statistics_records


def fixture(day="2026-09-08", observed=False):
    helper = runpy.run_path(str(Path(__file__).with_name("test_nuclear_reporting.py")))
    incumbent, data = helper["_fixture"](day)
    baseline = incumbent.covariates.set_index("timestamp").rename(columns={"fr_nuclear_fcst": report.NUCLEAR_ALIAS})
    baseline[report.NUCLEAR_ALIAS] /= 1000
    incumbent.covariates = baseline.copy()
    context = baseline.copy()
    for i, alias in enumerate(report.SOLAR_ALIASES):
        context[alias] = float(i + 1)
    for alias in tuple(context):
        context[f"known_{alias}_oracle"] = context[alias]
    data.model_context_covariates = context.copy()
    data.covariates = context.reindex(data.target.index)
    data.known_future_columns = [c for c in context if c.startswith("known_")]
    a = deepcopy(incumbent)
    for frame in (a.residual_statistics, a.source_forecast, a.kalman_view.backtest, a.kalman_view.forecast):
        for column in frame:
            if column.startswith(("residual_corrected__", "residual_kalman__")):
                frame[column] -= .5
    b = deepcopy(a)
    b.covariates = context[list(baseline) + list(report.SOLAR_ALIASES)].copy()
    for frame in (b.kalman_view.backtest, b.kalman_view.forecast):
        for q in report._Q:
            frame[f"residual_kalman__{q}"] -= .1
    results = {"residual": a, "residual_kalman": b}
    for key, result in results.items():
        result.audit = {"candidate_engine": report.ENGINE, "candidate_variant": "solar_" + key,
            "chronos_recomputed": False, "chronos_runtime_calls": 0, "frozen_chronos_exact_match": True,
            "baseline_inputs_preserved": True, "inherited_residual_feature_builder_unchanged": True,
            "additional_raw_input_count": 4, "new_ramps_or_aggregates": False,
            "spike_classifier_added": False, "promotion_eligible": False, "production_changed": False,
            "sealed_live_contract_modified": False, "solar_chronos_context_columns": [],
            "solar_chronos_known_future_columns": [], "solar_residual_features": list(report.SOLAR_KNOWN_COLUMNS),
            "solar_kalman_market_features": list(report.SOLAR_ALIASES) if key == "residual_kalman" else [],
            "kalman_covariate_config": {"input_columns": list(result.covariates), "groups": {"market": list(result.covariates)}},
            "solar_sources": {alias: {"series": report.SOLAR_SERIES[alias], "unit": "GW",
                "semantic": "forecast_generation", "daily_broadcast": False} for alias in report.SOLAR_ALIASES}}
    reseal(results)
    if observed:
        index = pd.DatetimeIndex(a.source_forecast.delivery_start_utc).tz_convert(data.timezone)
        data.target = pd.concat([data.target, pd.Series(53., index=index, name=data.target.name)])
    return results, incumbent, data


def reseal(results):
    for result in results.values():
        for field, digest in (("residual_statistics", "shared_upstream_statistics_sha256"),
                              ("source_forecast", "shared_upstream_forecast_sha256")):
            result.audit[digest] = report._digest_frame(getattr(result, field))
        result.audit["frozen_chronos_history_sha256"] = report._digest_frame(result.raw_history[list(report._Q)])
        forecast = report._timestamped(result.source_forecast, name="forecast")
        result.audit["frozen_chronos_future_sha256"] = report._digest_frame(forecast[
            [f"chronos2__{q}" for q in report._Q]].rename(columns={f"chronos2__{q}": q for q in report._Q}))


def capture(monkeypatch):
    rendered = []
    def write(results, config, path):
        rendered.extend(results)
        path.write_text('<html><main><h3>Comparaison au modèle prix seul</h3><table><tr><td>obsolete</td></tr></table>'
                        '<p>Prévision opérationnelle du jour</p>Covariables natives</main></html>', encoding="utf-8")
    monkeypatch.setattr(report, "write_html_report", write)
    return rendered


def run(tmp_path, monkeypatch, results, incumbent, data, **kwargs):
    monkeypatch.setattr(report, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "runs/experiments/solar_correction_v1/reports"
    day = str(pd.Timestamp(results["residual"].source_forecast.delivery_start_utc.iloc[0]).tz_convert(data.timezone).date())
    options = dict(storm_archive=None, observed_source_audit=None, source_audit={"fixture": True})
    options.update(kwargs)
    return report.render_solar_correction_reports(results, incumbent=incumbent, data=data, zone=data.zone,
        delivery_day=day, output_directory=output, **options)


@pytest.mark.parametrize("observed", [False, True])
def test_three_reports_use_same_family_same_hours_and_never_mutate(tmp_path, monkeypatch, observed):
    results, incumbent, data = fixture(observed=observed)
    before = deepcopy((results, incumbent, data))
    rendered = capture(monkeypatch)
    paths = run(tmp_path, monkeypatch, results, incumbent, data, validation_protocol={"frozen_at": "2026-09-20"})
    assert set(paths) == {*report.FAMILIES, "audit", "comparison"}
    assert len(rendered) == 3
    comparisons = json.loads(paths["comparison"].read_text(encoding="utf-8"))
    for key in report.FAMILIES:
        compared = comparisons[key]
        assert compared["candidate_model"] == key
        assert compared["incumbent_model"] == ("nuclear_autonomous" if key == "solar_residual" else "nuclear_kalman")
        assert compared["annual"]["candidate"]["hours"] == compared["annual"]["incumbent"]["hours"] == 8760
        assert compared["statistics_days"] == 365 and compared["delivery_day_included"] is observed
        assert "june_24_26" not in compared
        assert compared["annual"]["daily_mae_win_rate"] == pytest.approx(1.)
        assert compared["high_price_slices"]["ex_post_only"] is True
        assert compared["high_price_slices"]["thresholds"]["200"]["hours"] == 0
        assert compared["high_price_slices"]["thresholds"]["300"]["candidate"]["mae_eur_mwh"] is None
        document = paths[key].read_text(encoding="utf-8")
        assert "aucun solaire dans le transformer" in document and "préenregistrés" in document
        assert "obsolete" not in document and "SolarCWE" not in document
    if not observed:
        assert comparisons[report.FAMILIES[0]]["annual"]["gain_incumbent_minus_candidate"]["mae_eur_mwh"] == pytest.approx(.5)
        assert comparisons[report.FAMILIES[1]]["annual"]["gain_incumbent_minus_candidate"]["mae_eur_mwh"] == pytest.approx(.5)
        assert comparisons[report.FAMILIES[2]]["annual"]["gain_incumbent_minus_candidate"]["mae_eur_mwh"] == pytest.approx(.6)
        assert comparisons["solar_kalman_vs_standard_kalman"]["annual"]["gain_incumbent_minus_candidate"]["mae_eur_mwh"] == pytest.approx(.1)
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    assert audit["evaluation_hours"] == 8760 and audit["evaluation_days"] == 365
    assert audit["model_fitted_by_report"] is audit["production"] is audit["chronos_recomputed"] is False
    assert audit["prospective_validation_status"] == "pending_new_days_not_evaluated_by_report"
    for item in rendered:
        assert item.metrics_native["n"] == 8760
        assert item.statistics_candidate.tail(24).actual.notna().all() == observed
        records = [r for r in build_statistics_records([item]) if r["sample"] == "daily"]
        assert len(records) == (365 if observed else 366)
        assert set(report.SOLAR_ALIASES).issubset(item.zone_data.model_context_covariates)
    for key in results:
        pd.testing.assert_frame_equal(results[key].source_forecast, before[0][key].source_forecast)
        pd.testing.assert_frame_equal(results[key].residual_statistics, before[0][key].residual_statistics)
    pd.testing.assert_frame_equal(incumbent.kalman_view.forecast, before[1].kalman_view.forecast)
    pd.testing.assert_frame_equal(data.model_context_covariates, before[2].model_context_covariates)


@pytest.mark.parametrize("fault", ["engine", "chronos", "residual", "kalman", "solar_value", "nuclear",
    "unit", "broadcast", "shared_source", "crossing", "kalman_upstream", "incumbent_grid", "incumbent_raw", "digest", "kalman_config"])
def test_invalid_stages_quantiles_and_frozen_upstream_fail_before_write(tmp_path, monkeypatch, fault):
    results, incumbent, data = fixture()
    a, b = results.values()
    if fault == "engine":
        a.audit["candidate_engine"] = "solar_cwe_v1"
    elif fault == "chronos":
        a.audit["solar_chronos_context_columns"] = list(report.SOLAR_ALIASES)
    elif fault == "residual":
        a.audit["solar_residual_features"] = []
    elif fault == "kalman":
        a.audit["solar_kalman_market_features"] = list(report.SOLAR_ALIASES)
    elif fault == "solar_value":
        b.covariates[report.SOLAR_ALIASES[0]] += 1
    elif fault == "nuclear":
        a.covariates = a.covariates.drop(columns=report.NUCLEAR_ALIAS)
    elif fault in ("unit", "broadcast"):
        a.audit["solar_sources"][report.SOLAR_ALIASES[0]]["unit" if fault == "unit" else "daily_broadcast"] = "MW" if fault == "unit" else True
    elif fault == "shared_source":
        b.source_forecast["residual_corrected__q50"] += .01
        reseal(results)
    elif fault == "crossing":
        b.kalman_view.forecast["residual_kalman__q10"] += 100
    elif fault == "kalman_upstream":
        b.kalman_view.forecast["residual_corrected__q50"] += .01
    elif fault == "incumbent_grid":
        incumbent.kalman_view.backtest = incumbent.kalman_view.backtest.iloc[1:]
    elif fault == "incumbent_raw":
        incumbent.raw_history["q50"] += .01
    elif fault == "kalman_config":
        b.audit["kalman_covariate_config"]["groups"]["market"] = []
    else:
        a.audit["frozen_chronos_history_sha256"] = "not_matching"
    capture(monkeypatch)
    with pytest.raises(ValueError):
        run(tmp_path, monkeypatch, results, incumbent, data)
    assert not (tmp_path / "runs").exists()


def test_storm_gaps_do_not_reduce_paired_annual_population(tmp_path, monkeypatch):
    results, incumbent, data = fixture()
    calls = []
    def attach(prepared, path, **kwargs):
        calls.append(prepared)
        truth = prepared["solar_residual"].statistics_candidate.actual
        for item in prepared.values():
            assert item.statistics_candidate.actual.equals(truth)
            item.statistics_benchmark = item.statistics_candidate[["timestamp", "actual", "q50"]].copy()
            item.statistics_benchmark.loc[:2, "q50"] = np.nan
        return {"status": "complete", "missing_hours": 3}
    monkeypatch.setattr("chronos2_hourly.nuclear_report_benchmark.attach_nuclear_storm", attach)
    rendered = capture(monkeypatch)
    paths = run(tmp_path, monkeypatch, results, incumbent, data, storm_archive=tmp_path / "storm")
    assert len(calls) == 1 and len(calls[0]) == 5
    assert all(item.statistics_benchmark.q50.isna().sum() == 3 for item in rendered)
    assert all(value["annual"]["candidate"]["hours"] == 8760 for value in json.loads(paths["comparison"].read_text()).values())


def test_reports_refuse_output_outside_experiment(tmp_path, monkeypatch):
    results, incumbent, data = fixture()
    monkeypatch.setattr(report, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="must stay"):
        report.render_solar_correction_reports(results, data=data, zone="FR", delivery_day="2026-09-08",
            output_directory=tmp_path / "runs/exports", incumbent=incumbent, storm_archive=None,
            observed_source_audit=None, source_audit={})
    assert not (tmp_path / "runs").exists()


def test_win_rate_excludes_ties_and_spike_slices_use_only_observed_prices():
    from types import SimpleNamespace
    index = pd.date_range("2026-09-01", periods=48, freq="h", tz="Europe/Paris")
    left = pd.DataFrame({"timestamp": index, "actual": np.tile([199., 200., 299., 300.], 12), "q50": 0.})
    right = left.copy()
    right.loc[24:, "q50"] = -1.
    compared = {"statistics_start_day": "2026-09-01", "statistics_end_day": "2026-09-02", "annual": {}}
    report._diagnostic_slices(SimpleNamespace(statistics_candidate=left), SimpleNamespace(statistics_candidate=right), compared, "Europe/Paris")
    assert compared["annual"]["daily_mae_win_rate"] == .5
    slices = compared["high_price_slices"]["thresholds"]
    assert slices["200"]["hours"] == 36 and slices["300"]["hours"] == 12
    assert slices["300"]["candidate"]["mae_eur_mwh"] == 300.


def test_standard_html_renderer_smoke(tmp_path, monkeypatch):
    results, incumbent, data = fixture()
    paths = run(tmp_path, monkeypatch, results, incumbent, data)
    for family in report.FAMILIES:
        document = paths[family].read_text(encoding="utf-8")
        assert '<section data-report-section="solar-correction"' in document
        assert "Statistics" in document and report.LABELS[family] in document
        assert "Prévision expérimentale du" in document
        assert paths[family].stat().st_size > 10_000


@pytest.mark.parametrize("field", ["residual_statistics", "source_forecast"])
def test_corrected_bare_quantiles_are_not_compared_as_frozen_chronos(tmp_path, monkeypatch, field):
    results, incumbent, data = fixture()
    # Real source archives also expose the corrected forecast under bare q*.
    # Those values must differ when testing a different residual corrector.
    for result in (incumbent, *results.values()):
        frame = getattr(result, field)
        for q in report._Q:
            frame[q] = frame[f"residual_corrected__{q}"]
    reseal(results)
    assert not getattr(results["residual"], field).q50.equals(getattr(incumbent, field).q50)
    rendered = capture(monkeypatch)
    paths = run(tmp_path, monkeypatch, results, incumbent, data)
    assert len(rendered) == 3 and all(paths[family].exists() for family in report.FAMILIES)


@pytest.mark.parametrize("field", ["residual_statistics", "source_forecast"])
def test_matching_bare_quantiles_cannot_hide_changed_chronos(tmp_path, monkeypatch, field):
    results, incumbent, data = fixture()
    for result in (incumbent, *results.values()):
        frame = getattr(result, field)
        for q in report._Q:
            frame[q] = getattr(results["residual"], field)[f"residual_corrected__{q}"]
    getattr(incumbent, field)["chronos2__q50"] += .01
    reseal(results)
    capture(monkeypatch)
    with pytest.raises(ValueError, match="Frozen nuclear Chronos quantiles differ"):
        run(tmp_path, monkeypatch, results, incumbent, data)
    assert not (tmp_path / "runs").exists()
