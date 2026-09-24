"""Isolated fixture-only integrity and orchestration tests; no real NYX run/data.

All files are created below pytest's temporary directory. Scientific evaluation
and archive readers are replaced only where necessary to test orchestration.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from nyx_intrahour import data, reporting, runner
from nyx_intrahour.features import build_hourly_features


DAY = "2026-09-16"
SOURCE = {"alias": "fixture_fr_load", "zone": "FR", "driver": "load", "unit": "GW",
          "series": "synthetic.test.fixture.native15min", "native_resolution_minutes": 15,
          "is_forecast": True, "interpolation": "none",
          "native_resolution_evidence": "Synthetic unit-test fixture, not provider data."}


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def config(root):
    return {"schema_version": 1, "baseline_delivery_day": DAY,
            "source_manifest": str(root/"fixture_sources/sources.manifest.json"),
            "output_root": str(runner.NAMESPACE), "diagnostic_only": True,
            "activation_performed": False,
            "evaluation": {"initial_train_days": 2, "validation_days": 1, "test_days": 1,
                           "window_days": 2, "refit_every_days": 1, "min_train_rows": 4,
                           "bootstrap_repetitions": 1000, "seed": 20260916}}


def source_fixture(root, *, day=DAY, source=None):
    source = deepcopy(source or SOURCE)
    directory = root/"fixture_sources"
    directory.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(day).tz_localize("Europe/Paris")
    end = (pd.Timestamp(day)+pd.Timedelta(days=1)).tz_localize("Europe/Paris")
    quarters = pd.date_range(start, end, freq="15min", inclusive="left").tz_convert("UTC")
    cutoff = (pd.Timestamp(day)-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    vintages = pd.DataFrame({"source_alias": source["alias"], "value_time_utc": quarters,
                             "snapshot_time_utc": cutoff, "revision_time_utc": cutoff,
                             "value": np.tile([1.,2.,3.,4.], len(quarters)//4)})
    parquet = directory/"fixture_native.parquet"
    vintages.to_parquet(parquet, index=False)
    manifest = {"schema_version": 1, "artifact_type": "nyx_intrahour_forecast_vintages",
                "data_file": parquet.name, "data_sha256": data.digest(parquet),
                "sources": [source], "temporal_evidence": "retrospective_asof",
                "provider_revision_timestamp_available": False, "fixture_only": True}
    path = directory/"sources.manifest.json"
    write_json(path, manifest)
    return path, parquet, vintages, manifest


def baseline_fixture(root, quarters):
    hours = pd.DatetimeIndex(quarters[::4])
    panel = pd.concat([pd.DataFrame({"timestamp_utc": hours, "zone": zone,
                                     "actual": np.arange(len(hours), dtype=float)+40,
                                     "training_actual": np.arange(len(hours), dtype=float)+39,
                                     "nyx_q50": np.arange(len(hours), dtype=float)+41})
                       for zone in data.ZONES], ignore_index=True)
    for i, alias in enumerate(data.HOURLY_ALIASES):
        panel[f"feature_hourly_{alias}"] = 1.0+i
    artifact = root/"fixture_baseline_identity.txt"
    artifact.write_text("Synthetic baseline identity, not an operational model.", encoding="utf-8")
    audit = {"fixture_only": True, "production_pit_evidence": False,
             "identities": [{"zone": zone, "files": {str(artifact): data.digest(artifact)},
                              "observations": {}} for zone in data.ZONES]}
    return panel, audit


def prohibit_evaluation(monkeypatch):
    module = ModuleType("nyx_intrahour.evaluation")
    def prohibited(*args, **kwargs):
        raise AssertionError("Scientific evaluation must not run in this test.")
    module.evaluate_variant = prohibited
    monkeypatch.setitem(sys.modules, "nyx_intrahour.evaluation", module)


def test_native_manifest_checksum_is_enforced_before_parquet_read(tmp_path, monkeypatch):
    path, parquet, _, _ = source_fixture(tmp_path)
    parquet.write_bytes(parquet.read_bytes()+b"tampered")
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("Corrupt source was read"))
    with pytest.raises(ValueError, match="checksum"):
        data.read_native_sources(path)


@pytest.mark.parametrize("change", [
    {"schema_version":2}, {"artifact_type":"observations"},
    {"data_file":"../outside.parquet"}, {"data_file":"subdir/source.parquet"},
    {"temporal_evidence":"latest_only"}, {"provider_revision_timestamp_available":"false"},
    {"temporal_evidence":"archived_at_issue"}])
def test_native_manifest_rejects_invalid_identity_paths_and_timing(tmp_path, change):
    path, _, _, manifest = source_fixture(tmp_path)
    manifest.update(change)
    write_json(path, manifest)
    with pytest.raises(ValueError):
        data.read_native_sources(path)


def test_manifest_hash_binds_metadata_read_not_a_later_file_version(tmp_path, monkeypatch):
    path, _, _, manifest = source_fixture(tmp_path)
    original_read = pd.read_parquet
    def mutate_during_read(*args, **kwargs):
        modified = deepcopy(manifest)
        modified["sources"][0]["series"] = "different.fixture.identity"
        write_json(path, modified)
        return original_read(*args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", mutate_during_read)
    try:
        _, sources, audit = data.read_native_sources(path)
    except ValueError:
        return  # Failing immediately on the mutation is the stronger behavior.
    assert sources[0]["series"] == SOURCE["series"]
    assert audit["manifest_sha256"] != data.digest(path), "Old metadata must not be sealed with the new manifest's hash"


@pytest.mark.parametrize("change", [
    {"is_forecast":False}, {"native_resolution_minutes":60},
    {"interpolation":"linear"}, {"driver":"price"},
    {"native_resolution_evidence":""}])
def test_source_metadata_is_enforced_before_audit_prepared(tmp_path, monkeypatch, change):
    _, _, vintages, _ = source_fixture(tmp_path, source={**SOURCE, **change})
    baseline, audit = baseline_fixture(tmp_path, vintages.value_time_utc)
    monkeypatch.setattr(runner, "load_baseline", lambda *a, **k: (baseline,audit))
    prohibit_evaluation(monkeypatch)
    directory = runner.run(config(tmp_path), root=tmp_path, audit_only=True)
    summary = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["activation_performed"] is False
    assert not (directory/"metrics.csv").exists()


@pytest.mark.parametrize("output", [
    "runs/experiments/nuclear_forecast_v1", "runs/reports",
    "runs/experiments/nyx_intrahour_v1/../nuclear_forecast_v1", "outside"])
def test_runner_refuses_output_outside_research_namespace_before_writes(tmp_path, output):
    cfg = config(tmp_path)
    cfg["output_root"] = output
    with pytest.raises(ValueError):
        runner.run(cfg, root=tmp_path)
    assert list(tmp_path.rglob("*")) == []


@pytest.mark.parametrize("redirect_namespace", [True,False])
def test_resolved_namespace_cannot_be_redirected_to_another_directory(tmp_path, redirect_namespace):
    outside = tmp_path/"operational_sentinel"
    outside.mkdir()
    sentinel = outside/"keep.txt"
    sentinel.write_text("unchanged",encoding="utf-8")
    namespace = tmp_path/runner.NAMESPACE
    link = namespace if redirect_namespace else namespace/"escape"
    link.parent.mkdir(parents=True,exist_ok=True)
    try:
        link.symlink_to(outside,target_is_directory=True)
    except (OSError,NotImplementedError):
        if sys.platform != "win32":
            pytest.skip("Creating directory links is unavailable on this platform.")
        # Windows junctions need no symlink privilege; both endpoints remain
        # inside this test's temporary directory, never the real workspace.
        import _winapi
        try:
            _winapi.CreateJunction(str(outside),str(link))
        except OSError:
            pytest.skip("Creating either a directory symlink or junction is unavailable.")
    with pytest.raises(ValueError):
        runner.safe_output(tmp_path,link/"new_run")
    assert sentinel.read_text(encoding="utf-8")=="unchanged"
    assert sorted(p.name for p in outside.iterdir())==["keep.txt"]


def test_absent_real_input_writes_honest_unavailable_report_without_scores(tmp_path,monkeypatch):
    prohibit_evaluation(monkeypatch)
    monkeypatch.setattr(runner,"load_baseline",lambda *a,**k: pytest.fail("Missing native data should stop before baseline reading"))
    directory=runner.run(config(tmp_path),root=tmp_path)
    summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    assert summary["status"]=="data_unavailable"
    assert summary["activation_performed"] is False and summary["production_modified"] is False
    assert summary["decision"]["promotion_allowed"] is False
    assert not list(directory.glob("*.parquet"))
    assert not list(directory.glob("*.csv"))
    html=(directory/"report.html").read_text(encoding="utf-8")
    assert "Données à 15 minutes indisponibles" in html
    assert "Qualité des prévisions" not in html
    assert "Écarts appariés et incertitude" not in html
    latest=json.loads((tmp_path/runner.NAMESPACE/"latest.json").read_text(encoding="utf-8"))
    assert latest["status"]=="data_unavailable"
    assert all(p.is_relative_to(tmp_path/runner.NAMESPACE) for p in tmp_path.rglob("*") if p.is_file())


def test_audit_only_seals_fixture_inputs_without_fit_or_activation(tmp_path,monkeypatch):
    path,parquet,vintages,_=source_fixture(tmp_path)
    baseline,baseline_audit=baseline_fixture(tmp_path,vintages.value_time_utc)
    before={str(p):data.digest(p) for p in [path,parquet,tmp_path/"fixture_baseline_identity.txt"]}
    monkeypatch.setattr(runner,"load_baseline",lambda *a,**k:(baseline,baseline_audit))
    prohibit_evaluation(monkeypatch)
    directory=runner.run(config(tmp_path),root=tmp_path,audit_only=True)
    summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    assert summary["status"]=="prepared"
    assert summary["feature_audit"]["complete_hours"]==24
    assert summary["native_source_audit"]["fixture_only"] is True
    assert not summary["activation_performed"] and not summary["production_modified"]
    assert not (directory/"metrics.csv").exists()
    panel=pd.read_parquet(directory/"panel.parquet")
    assert len(panel)==96 and not panel.duplicated(["timestamp_utc","zone"]).any()
    assert panel["nyx_q50"].tolist()==baseline.nyx_q50.tolist()
    inputs=json.loads((directory/"inputs.manifest.json").read_text(encoding="utf-8"))
    for name,sha in inputs["files"].items():
        assert data.digest(directory/name)==sha
    outputs=json.loads((directory/"outputs.manifest.json").read_text(encoding="utf-8"))
    for name,sha in outputs["files"].items():
        assert data.digest(directory/name)==sha
    assert before=={p:data.digest(Path(p)) for p in before}


def test_join_keeps_all_physical_hours_and_incomplete_rows(tmp_path):
    _,_,vintages,_=source_fixture(tmp_path,day="2026-10-25")
    baseline,_=baseline_fixture(tmp_path,vintages.value_time_utc)
    hourly,_=build_hourly_features(vintages.iloc[1:],[SOURCE],"2026-10-25","2026-10-25")
    panel=data.join_features(baseline,hourly)
    assert len(panel)==100
    assert panel.timestamp_utc.nunique()==25
    assert panel.groupby("zone").size().eq(25).all()
    assert panel.filter(like="__mean_gw").isna().sum().iloc[0]==4
    pd.testing.assert_series_equal(panel.actual,baseline.actual)


@pytest.mark.parametrize("side",["baseline","hourly","collision"])
def test_join_rejects_ambiguous_population_or_overwrite(tmp_path,side):
    _,_,vintages,_=source_fixture(tmp_path)
    baseline,_=baseline_fixture(tmp_path,vintages.value_time_utc)
    hourly,_=build_hourly_features(vintages,[SOURCE],DAY,DAY)
    if side=="baseline":
        baseline=pd.concat([baseline,baseline.iloc[[0]]],ignore_index=True)
    elif side=="hourly":
        hourly=pd.concat([hourly,hourly.iloc[[0]]])
    else:
        hourly["nyx_q50"]=999.0
    with pytest.raises(ValueError):
        data.join_features(baseline,hourly)


def install_baseline_reader_fixtures(monkeypatch,root,*,with_observed=True):
    """Stub model/archive objects, keeping actual file reads and provenance checks."""
    index=pd.date_range("2026-09-14T22:00Z",periods=3,freq="1h")
    results,refreshes={},{}
    for zone in data.ZONES:
        work=root/"runs/experiments/nuclear_forecast_v1"/DAY/zone.lower()/"civil_pit_v2"
        bundle=work/"report_only/frozen_result"
        bundle.mkdir(parents=True)
        write_json(bundle/"manifest.json",{"fixture_only":True,"zone":zone})
        backtest=pd.DataFrame({"delivery_start_utc":index,"actual":[10.,11.,12.],
                               "residual_kalman__q50":[11.,12.,13.]})
        cov=pd.DataFrame({"timestamp":index,**{a:np.full(3,i+1.) for i,a in enumerate(data.HOURLY_ALIASES)}})
        results[str(work)]=SimpleNamespace(audit={"zone":zone,"delivery_day":DAY},covariates=cov,
            kalman_view=SimpleNamespace(backtest=backtest,replay=SimpleNamespace(audit={
                "causality_violations":0,"target_actuals_assimilated_before_forecast":0})))
        if with_observed:
            source=work/"report_only/sources/fixture"
            (source/"inputs").mkdir(parents=True)
            observed_path=source/"inputs/observed_latest.parquet"
            pd.DataFrame({"actual":[20.,21.,22.]},index=index).to_parquet(observed_path)
            payload={"fixture_only":True,"status":"complete","zone":zone,
                     "delivery_day_local":DAY,"used_for_prediction":False,
                     "observed":{"artifact_sha256":data.digest(observed_path)}}
            audit_path=source/"audit.json"
            write_json(audit_path,payload)
            refreshes[str(work)]=(None,audit_path,payload)
    archive=ModuleType("chronos2_hourly.nuclear_run_archive")
    archive.load_nuclear_result_bundle=lambda *,workdir:results[str(workdir)]
    storm=ModuleType("chronos2_hourly.model_storm_data")
    storm._latest_source=lambda work:refreshes.get(str(work))
    storm._series=lambda frame,column:frame[column]
    refresh=ModuleType("chronos2_hourly.nuclear_reporting_refresh")
    refresh.verify_refreshed_observations=lambda *a,**k:None
    for name,module in [(archive.__name__,archive),(storm.__name__,storm),(refresh.__name__,refresh)]:
        monkeypatch.setitem(sys.modules,name,module)
    return results,refreshes,archive


def test_baseline_reader_keeps_frozen_training_labels_and_verified_latest_scores(tmp_path,monkeypatch):
    install_baseline_reader_fixtures(monkeypatch,tmp_path)
    panel,audit=data.load_baseline(tmp_path,DAY)
    assert len(panel)==12
    assert set(panel.actual)=={20.,21.,22.}
    assert set(panel.training_actual)=={10.,11.,12.}
    assert set(panel.nyx_q50)=={11.,12.,13.}
    assert audit["production_pit_evidence"] is False
    for item in audit["identities"]:
        assert item["evaluation_label_source"]=="verified_latest_observed"
        assert len(item["observations"])==2
        for path,sha in {**item["files"],**item["observations"]}.items():
            assert data.digest(Path(path))==sha


def test_baseline_reader_rejects_observed_artifact_checksum_mismatch(tmp_path,monkeypatch):
    _,refreshes,_=install_baseline_reader_fixtures(monkeypatch,tmp_path)
    _,audit_path,payload=next(iter(refreshes.values()))
    payload["observed"]["artifact_sha256"]="0"*64
    write_json(audit_path,payload)
    with pytest.raises(ValueError,match="checksum"):
        data.load_baseline(tmp_path,DAY)


def test_baseline_reader_rejects_changed_bundle_and_country_hour_mismatch(tmp_path,monkeypatch):
    results,_,archive=install_baseline_reader_fixtures(monkeypatch,tmp_path,with_observed=False)
    original=archive.load_nuclear_result_bundle
    def mutate(*,workdir):
        (workdir/"report_only/frozen_result/manifest.json").write_text("changed",encoding="utf-8")
        return original(workdir=workdir)
    archive.load_nuclear_result_bundle=mutate
    with pytest.raises(ValueError,match="changed"):
        data.load_baseline(tmp_path,DAY)
    archive.load_nuclear_result_bundle=original
    result=list(results.values())[1]
    result.kalman_view.backtest["delivery_start_utc"]+=pd.Timedelta(hours=1)
    result.covariates["timestamp"]+=pd.Timedelta(hours=1)
    with pytest.raises(ValueError,match="same physical hours"):
        data.load_baseline(tmp_path,DAY)


def test_source_mutation_after_evaluation_invalidates_scores_in_report(tmp_path,monkeypatch):
    _,parquet,vintages,_=source_fixture(tmp_path)
    baseline,audit=baseline_fixture(tmp_path,vintages.value_time_utc)
    monkeypatch.setattr(runner,"load_baseline",lambda *a,**k:(baseline,audit))
    module=ModuleType("nyx_intrahour.evaluation")
    def fixture_evaluation(*args,**kwargs):
        parquet.write_bytes(parquet.read_bytes()+b"changed during fixture evaluation")
        return {"status":"complete","decision":{"encouraging":True,"promotion_allowed":False},
                "metrics":pd.DataFrame({"model":["fixture_not_real"],"mae":[0.123]})}
    module.evaluate_variant=fixture_evaluation
    monkeypatch.setitem(sys.modules,"nyx_intrahour.evaluation",module)
    directory=runner.run(config(tmp_path),root=tmp_path)
    summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    assert summary["status"]=="failed"
    assert summary["decision"]["encouraging"] is False
    assert summary["activation_performed"] is False
    html=(directory/"report.html").read_text(encoding="utf-8")
    assert "Qualité des prévisions" not in html
    assert "Les critères de poursuite sont satisfaits" not in html


def test_report_escapes_untrusted_reason_and_table_values(tmp_path):
    summary={"status":"failed","reason":"<script>bad()</script>","activation_performed":False}
    path=reporting.render_report(tmp_path,summary,{})
    html=path.read_text(encoding="utf-8")
    assert "<script>" not in html and "&lt;script&gt;" in html


@pytest.mark.parametrize("change",[{"activation_performed":True},{"diagnostic_only":False},{"unexpected":1}])
def test_config_cannot_activate_or_silently_accept_unknown_fields(tmp_path,change):
    cfg=config(tmp_path)
    cfg.update(change)
    path=tmp_path/"fixture_config.yaml"
    path.write_text(yaml.safe_dump(cfg),encoding="utf-8")
    with pytest.raises(ValueError):
        runner.load_config(path)


@pytest.mark.parametrize("status,expected",[("prepared",0),("data_unavailable",3),("failed",2)])
def test_cli_audit_maps_status_to_exit_code_without_real_execution(tmp_path,monkeypatch,capsys,status,expected):
    import run_nyx_intrahour as cli
    cfg=config(tmp_path)
    path=tmp_path/"fixture_config.yaml"
    path.write_text(yaml.safe_dump(cfg),encoding="utf-8")
    directory=tmp_path/"fixture_cli_result"
    directory.mkdir()
    write_json(directory/"summary.json",{"status":status,"activation_performed":False})
    calls=[]
    def fake_run(config,*,root,audit_only):
        calls.append((root,audit_only))
        return directory
    monkeypatch.setattr(cli,"ROOT",tmp_path)
    monkeypatch.setattr(runner,"run",fake_run)
    assert cli.main(["--config",str(path),"--action","audit"])==expected
    payload=json.loads(capsys.readouterr().out)
    assert payload["status"]==status and payload["activation_performed"] is False
    assert calls==[(tmp_path,True)]
