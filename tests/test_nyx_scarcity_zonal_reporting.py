from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.storm_dashboard import STORM_DASHBOARD_COLUMN
from chronos2_modular.report import _statistics_source, build_statistics_records
from nyx_scarcity_zonal import reporting as report


def fixture(tmp_path):
    timezone = "Europe/Paris"
    stamps = pd.date_range("2025-09-15", "2026-09-16", tz=timezone, inclusive="left", freq="h").tz_convert("UTC")
    local = stamps.tz_convert(timezone)
    live = local.date == pd.Timestamp("2026-09-15").date()
    values = np.full(len(stamps), 102.)
    hole = stamps == pd.Timestamp("2025-10-26T00:00:00Z")
    values[hole] = np.nan
    frame = pd.DataFrame({"zone": "FR", "timestamp_utc": stamps,
        "forecast_origin_utc": [pd.Timestamp(day).tz_localize(timezone)-pd.DateOffset(days=1)+pd.Timedelta(hours=8) for day in local.date],
        "sample": np.where(live,"live","evaluation"), "actual": np.where(live,10000.,104.),
        "forecast": 100., "q10": 90., "q90": 120., "benchmark_forecast": values,
        "candidate_forecast": 101., "candidate_q10": 91., "candidate_q90": 118.,
        "spike_probability": .7, "expert_ready": True, "gate_reason": "fixed_alpha25",
        "bounded_correction": 4., "applied_correction": 1., "selected_weight": .25})
    # This deliberately excluded day has source values. The report must not refill it.
    frame.loc[:23, "benchmark_forecast"] = np.nan
    archive = tmp_path / "original_archive"; (archive/"inputs").mkdir(parents=True)
    artifact = archive/"inputs/storm_dashboard_official_statistics.parquet"
    pd.DataFrame({"delivery_start_utc":stamps,STORM_DASHBOARD_COLUMN:values}).to_parquet(artifact,index=False)
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    series = "power.price.fr.euromwh.h.fcst.3mv.storm.da.cache"
    holes = [stamp.isoformat() for stamp in stamps[hole]]
    materialization = {"role":"evaluation_only_dashboard_comparator","series":series,"column":STORM_DASHBOARD_COLUMN,
        "used_for_prediction":False,"used_for_live_forecast":False,"normalized_artifact_path":"inputs/storm_dashboard_official_statistics.parquet",
        "normalized_artifact_sha256":sha(artifact),
        "source":{"kind":"saturn_storm_day_ahead_cache_with_native_gap_fallback","zone":"FR","series":series,"primary_series":series,
            "series_kind":"frozen_day_ahead_cache","used_for_prediction":False,"fallback_policy":"native_only_where_day_ahead_cache_is_missing",
            "cache_precedence":True,"fallback_series":"power.price.fr.euromwh.h.fcst.3mv.storm","extracted_at_utc":"2026-09-14T12:00:00Z"},
        "period":{"start_utc":stamps[0].isoformat(),"end_utc":stamps[-1].isoformat(),"timezone":timezone},
        "expected_hours":len(stamps),"available_hours":len(stamps)-1,"missing_hours":1,
        "dst":{"interpolation":False,"strict_08_fallback":False,"native_allowed_missing_utc":holes,"native_allowed_missing_hours":1,"native_actual_missing_matches_allowed":True}}
    history = archive/"statistics_history_audit.json"
    history.write_text(json.dumps({"status":"complete","storm_primary_report_benchmark":STORM_DASHBOARD_COLUMN,"storm_dashboard":materialization}),encoding="utf-8")
    nuclear = tmp_path/"nuclear_report_audit.json"
    nuclear.write_text(json.dumps({"zone":"FR","storm_hourly_comparison":{"snapshot":{"archive":str(archive),"artifact_path":str(artifact),
        "artifact_sha256":sha(artifact),"audit_path":str(history),"audit_sha256":sha(history)}}}),encoding="utf-8")
    audit = {"source_data_audit":{"baseline":{"evaluation_start_day":"2025-09-15","evaluation_end_day":"2026-09-14",
        "sources":[{"zone":"FR","model":"nuclear_kalman","time_axis_audits":[{"path":str(nuclear),"sha256":sha(nuclear)}]}]}}}
    return frame,audit


def test_production_statistics_exact_window_mask_and_price_means(tmp_path):
    frame,audit = fixture(tmp_path)
    before, prior = frame.copy(deep=True),deepcopy(audit)
    prepared,proof = report._result(report._prepare(frame),source_audit=audit,directory=tmp_path/"output",label="Zonal")
    stats = _statistics_source(prepared)
    assert len(stats)==8760 and stats._timestamp_local.max().date().isoformat()=="2026-09-14"
    assert stats._benchmark_q50.isna().sum()==25
    assert proof["paired_hours"]==8735
    assert proof["storm"]["archive_values_excluded_by_frozen_panel_mask"]==24
    assert prepared.metrics_native["mae_q50"]==3 and prepared.metrics_baseline["mae_q50"]==4
    assert prepared.backtest_native.actual.eq(104).all()
    assert prepared.forecast_native.q50.eq(101).all()
    records = build_statistics_records([prepared])
    daily = [r for r in records if r["sample"]=="daily"]
    assert len(daily)==365 and daily[-1]["period_key"]=="2026-09-14"
    assert daily[-1]["mean_price"]==101 and daily[-1]["observed_mean_price"]==104 and daily[-1]["benchmark_mean_price"]==102
    assert daily[0]["benchmark_mean_price"] is None
    dst = next(r for r in daily if r["period_key"]=="2025-10-26")
    assert dst["n"]==24 and dst["benchmark_n"]==24
    assert len(prepared.backtest_native.loc[prepared.backtest_native.timestamp.dt.date==pd.Timestamp("2025-10-26").date()])==24
    pd.testing.assert_frame_equal(frame,before); assert audit==prior


def test_true_operational_renderer_sections_and_experimental_banner(tmp_path):
    frame,audit = fixture(tmp_path)
    audit["decision_policy"]="fixed_25_percent_experimental"
    audit["strict_governor_enforced"]=False
    paths = report.render_zonal_reports(frame,source_audit=audit,output_directory=tmp_path/"output")
    text = paths["FR"].read_text(encoding="utf-8")
    for marker in ("EXPÉRIMENTAL", "nyx_zonal", "Backtest et probabilités", "Statistics", "STATISTICS",
                   "zonal-experiment", "storm-comparison", "hourly-comparison", "variable-attribution",
                   "Prix zonal retenu", "NYX nucléaire + Kalman figé", "projection du snapshot comparatif figé"):
        assert marker in text
    assert "Prévision opérationnelle du" not in text
    assert "2026-09-15 affichée séparément" in text and "exclue des scores" in text
    assert "correction fixe de 25 %" in text and "p &gt; 0,6" in text
    assert "n’est PAS appliqué" in text and "testée séparément" in text
    assert paths["index"].is_file() and paths["comparison"].is_file()
    assert '../zonal_comparison.html' in paths["index"].read_text(encoding="utf-8")
    assert json.loads(paths["audit"].read_text(encoding="utf-8"))["reports"]["FR"]["live_excluded_from_statistics"] is True
    node=Path(r"C:\Users\BQ6757\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")
    if node.is_file():
        scripts=re.findall(r'<script(?:\s+type="text/javascript")?>(.*?)</script>',text,re.S)
        assert scripts
        checked=subprocess.run([str(node),"--check"],input="\n".join(scripts),text=True,encoding="utf-8",capture_output=True)
        assert checked.returncode==0,checked.stderr


@pytest.mark.parametrize("change",["storm","source_hash","quantile","origin","missing_hour","window","price_arithmetic"])
def test_invalid_frozen_contract_fails_before_writing(tmp_path,change):
    frame,audit = fixture(tmp_path)
    if change=="storm": frame.loc[24,"benchmark_forecast"]+=1
    elif change=="source_hash": audit["source_data_audit"]["baseline"]["sources"][0]["time_axis_audits"][0]["sha256"]="bad"
    elif change=="quantile": frame.loc[0,"candidate_q10"]=np.nan
    elif change=="origin": frame.loc[0,"forecast_origin_utc"]=frame.loc[0,"timestamp_utc"]
    elif change=="missing_hour": frame=frame.iloc[1:]
    elif change=="price_arithmetic": frame.loc[0,"candidate_forecast"]+=1
    else: audit["source_data_audit"]["baseline"]["evaluation_start_day"]="2025-09-16"
    output=tmp_path/"output"
    with pytest.raises(ValueError): report.render_zonal_reports(frame,source_audit=audit,output_directory=output)
    assert not output.exists()


def test_production_output_rejected_before_sources_are_loaded(tmp_path):
    root=Path(report.__file__).resolve().parents[1]
    for folder in (root,root/"runs/exports/2026-09-15/fr/zonal",root/"runs/live/test",root/"config/reports"):
        with pytest.raises(ValueError): report.render_zonal_reports(pd.DataFrame(),source_audit={},output_directory=folder)


def test_unsafe_model_name_rejected():
    with pytest.raises(ValueError): report.render_zonal_reports(pd.DataFrame(),source_audit={},output_directory=Path("tmp/x"),model_name="../../export")


def test_index_comparison_displays_annual_prices_and_worsened_corrections():
    score={"hours":8735,"mae_eur_mwh":10.12,"rmse_eur_mwh":20.5,"bias_eur_mwh":-1.2,
        "daily_mean_mae_eur_mwh":4.5,"mean_forecast_eur_mwh":88.1,"mean_observed_eur_mwh":89.3}
    comparison={"by_zone":{"FR":{"annual":{key:dict(score) for key in ("nuclear_kalman","storm","regional_25","zonal_hiercal")},
        "interventions":{"regional_25":{"active_hours":100,"worsened_absolute_error_hours":27},
                         "zonal_hiercal":{"active_hours":80,"worsened_absolute_error_hours":19}}}}}
    before=deepcopy(comparison)
    rendered=report._index_comparison(comparison)
    for word in ("NYX nucléaire", "Storm figé", "régional précédent", "zonal + calibration", "10.120", "88.100", "89.300",
                 "régional 25 % = 27"):
        assert word in rendered
    assert "zonal calibré 25 % = 19" in rendered and "zonal − régional = -8" in rendered
    assert "faux positif" in rendered and "aucune promotion" in rendered
    assert comparison==before
    assert "aucun classement" in report._index_comparison(None)
