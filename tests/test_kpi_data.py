from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kpi_report import data


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def panel():
    rows = []
    # Spring DST: the complete evaluation day has 23 physical hours.
    index = pd.date_range("2026-03-29", "2026-03-31", freq="h", inclusive="left", tz="Europe/Paris").tz_convert("UTC")
    for zone in data.ZONES:
        for timestamp in index:
            local_day = timestamp.tz_convert("Europe/Paris").tz_localize(None).normalize()
            rows.append({"zone": zone, "timestamp_utc": timestamp,
                "forecast_origin_utc": (local_day - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC"),
                "sample": "evaluation" if timestamp.tz_convert("Europe/Paris").day == 29 else "live",
                "forecast": 40., "actual": 42., "benchmark_forecast": 41.})
    return pd.DataFrame(rows)


def reseal(directory, family):
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["input_files"] = {name: digest(directory / name) for name in manifest["input_files"]}
    write_json(directory / "manifest.json", manifest)
    if family == "stress_guard":
        file = directory / "results_manifest.json"
        result = json.loads(file.read_text())
        result["suite_manifest_sha256"] = digest(directory / "manifest.json")
        result["result_files"] = {name: digest(directory / name) for name in result["result_files"]}
        write_json(file, result)
    else:
        variants = manifest["config"]["variants"]
        for name in variants:
            file = directory / name / "results_manifest.json"
            result = json.loads(file.read_text())
            result["suite_manifest_sha256"] = digest(directory / "manifest.json")
            result["result_files"] = {key: digest(directory / name / key) for key in result["result_files"]}
            write_json(file, result)
        comparison = json.loads((directory / "comparison_manifest.json").read_text())
        comparison["suite_manifest_sha256"] = digest(directory / "manifest.json")
        comparison["variant_manifests"] = {name: digest(directory / name / "results_manifest.json") for name in variants}
        comparison["comparison_sha256"] = digest(directory / "comparison.json")
        write_json(directory / "comparison_manifest.json", comparison)


def snapshot(root, family, name="20260914T171142Z_86e6f8f9"):
    directory = root / data.NAMESPACE / family / "snapshots" / name
    directory.mkdir(parents=True)
    variants = ["fundamental", "calendar"] if family == "fundamental" else ["forest", "empirical"]
    config = {"diagnostic_only": True, "production_modified": False, "activation_performed": False}
    if family != "stress_guard":
        config["variants"] = variants
    write_json(directory / "config.json", config)
    write_json(directory / "source_audit.json", {})
    source = panel()
    source.to_parquet(directory / "panel.parquet", index=False)
    manifest = {"config": config, "input_files": {name: digest(directory / name) for name in
        ["config.json", "source_audit.json", "panel.parquet"]}, "created_at_utc": "2026-09-14T17:11:42Z"}
    write_json(directory / "manifest.json", manifest)
    for spec in data.MODEL_SPECS:
        if spec[0] != family:
            continue
        destination = directory / spec[3]
        destination.parent.mkdir(exist_ok=True)
        result = source.copy()
        result["candidate_forecast"] = 41.
        result.to_parquet(destination, index=False)
    if family == "stress_guard":
        write_json(directory / "results_manifest.json", {"status": "completed", "suite_manifest_sha256": digest(directory / "manifest.json"),
            "result_files": {spec[3]: digest(directory / spec[3]) for spec in data.MODEL_SPECS if spec[0] == family}})
    else:
        for name in variants:
            write_json(directory / name / "results_manifest.json", {"status": "completed", "variant": name,
                "suite_manifest_sha256": digest(directory / "manifest.json"),
                "result_files": {Path(spec[3]).name: digest(directory / spec[3]) for spec in data.MODEL_SPECS
                                 if spec[0] == family and Path(spec[3]).parts[0] == name}})
        write_json(directory / "comparison.json", {})
        write_json(directory / "comparison_manifest.json", {"status": "completed", "suite_manifest_sha256": digest(directory / "manifest.json"),
            "comparison_sha256": digest(directory / "comparison.json"),
            "variant_manifests": {name: digest(directory / name / "results_manifest.json") for name in variants}})
    report = {"fundamental": "fundamental_comparison.html", "coherent_p50": "coherent_p50_comparison.html", "stress_guard": "stress_guard_comparison.html"}[family]
    (directory / report).write_text("<html>Test</html>")
    write_json(directory.parents[1] / "latest.json", {"status": "completed", "snapshot": str(directory)})
    return directory


@pytest.fixture
def tree(tmp_path, monkeypatch):
    directories = {name: snapshot(tmp_path, name) for name in data.FAMILIES}
    monkeypatch.setattr(data, "_verify_production", lambda *args: {"delivery_day": "2026-03-30",
        "description": "NYX nucléaire + Kalman · production (historique figé)",
        "sources": [{"zone": zone, "path": str(tmp_path / f"{zone}.html")} for zone in data.ZONES]})
    return tmp_path, directories


def test_load_twelve_models_complete_dst_live_excluded(tree):
    root, _ = tree
    frame, catalog, audit = data.load_recent_models(root)
    assert len(catalog) == 12
    assert len(frame) == 12 * 4 * 23
    assert set(frame["sample"]) == {"evaluation"}
    assert audit["recommended_end_day"] == "2026-03-29"
    assert audit["source_delivery_day"] == "2026-03-30"
    assert set(audit["live_rows_excluded_by_model"].values()) == {96}
    assert catalog[-1]["price_alias_of"] == "coherent_forest_direct"
    assert audit["external_api_used"] is False
    assert audit["models_retrained"] is False
    assert audit["production_modified"] is False
    json.dumps(audit, allow_nan=False)
    data.verify_sources(audit)


@pytest.mark.parametrize("column", ["forecast", "actual", "benchmark_forecast"])
def test_different_reference_vintage_is_not_silently_replaced(tree, column):
    root, directories = tree
    directory = directories["coherent_p50"]
    frame = pd.read_parquet(directory / "panel.parquet")
    frame.loc[0, column] += 1
    frame.to_parquet(directory / "panel.parquet", index=False)
    reseal(directory, "coherent_p50")
    with pytest.raises(data.KPIDataError, match="Different frozen"):
        data.load_recent_models(root)


def test_result_must_keep_baseline_reference(tree):
    root, directories = tree
    directory = directories["stress_guard"]
    path = directory / "physics_direct.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "benchmark_forecast"] += 1
    frame.to_parquet(path, index=False)
    reseal(directory, "stress_guard")
    with pytest.raises(data.KPIDataError, match="Different frozen benchmark_forecast"):
        data.load_recent_models(root)


def test_recalibrated_alias_must_preserve_point_forecast(tree):
    root, directories = tree
    directory = directories["stress_guard"]
    path = directory / "p50_calibrated.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "candidate_forecast"] += 1
    frame.to_parquet(path, index=False)
    reseal(directory, "stress_guard")
    with pytest.raises(data.KPIDataError, match="Interval-only control"):
        data.load_recent_models(root)


def test_tampered_parquet_seal_rejected(tree):
    root, directories = tree
    path = directories["coherent_p50"] / "forest/predictions.parquet"
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(data.KPIDataError, match="SHA256 mismatch"):
        data.load_recent_models(root)


def test_result_manifest_bound_to_snapshot(tree):
    root, directories = tree
    path = directories["stress_guard"] / "results_manifest.json"
    result = json.loads(path.read_text())
    result["suite_manifest_sha256"] = "0" * 64
    write_json(path, result)
    with pytest.raises(data.KPIDataError, match="unbound result"):
        data.load_recent_models(root)


def test_missing_seal_rejected(tree):
    root, directories = tree
    directory = directories["stress_guard"]
    path = directory / "results_manifest.json"
    result = json.loads(path.read_text())
    del result["result_files"]["physics_direct.parquet"]
    write_json(path, result)
    with pytest.raises(data.KPIDataError, match="Incomplete file seals"):
        data.load_recent_models(root)


def test_sources_recheck_detects_mutation(tree):
    root, directories = tree
    _, _, audit = data.load_recent_models(root)
    path = directories["stress_guard"] / "config.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(data.KPIDataError, match="Source changed after KPI capture"):
        data.verify_sources(audit)


def test_fallback_ignores_unfinished_latest_pointer(tree):
    root, directories = tree
    directory = directories["stress_guard"]
    pointer = directory.parents[1] / "latest.json"
    write_json(pointer, {"status": "running", "snapshot": "unfinished"})
    _, _, audit = data.load_recent_models(root)
    assert audit["selected_families"]["stress_guard"]["selection"] == "bounded_latest_completed_directory"
    assert str(pointer) not in audit["source_files"]


def test_corrupt_completed_pointer_never_silently_falls_back(tree):
    root, directories = tree
    directory = directories["stress_guard"]
    pointer = directory.parents[1] / "latest.json"
    write_json(pointer, {"status": "completed", "snapshot": str(directory.parent / "20260915T171142Z_86e6f8f9")})
    with pytest.raises(data.KPIDataError, match="Cannot read source"):
        data.load_recent_models(root)


@pytest.mark.parametrize("mutation", ["duplicate", "naive", "half_hour", "unknown_sample", "missing_forecast", "infinite_actual", "wrong_zone", "late_origin", "naive_origin"])
def test_prediction_schema_refuses_invalid_input(mutation):
    frame = panel()
    if mutation == "duplicate": frame = pd.concat([frame, frame.iloc[[0]]])
    elif mutation == "naive": frame["timestamp_utc"] = frame.timestamp_utc.dt.tz_localize(None)
    elif mutation == "half_hour": frame.loc[0, "timestamp_utc"] += pd.Timedelta(minutes=30)
    elif mutation == "unknown_sample": frame.loc[0, "sample"] = "backfilled_live"
    elif mutation == "missing_forecast": frame.loc[0, "forecast"] = np.nan
    elif mutation == "infinite_actual": frame.loc[0, "actual"] = np.inf
    elif mutation == "wrong_zone": frame.loc[0, "zone"] = "ES"
    elif mutation == "late_origin": frame.loc[0, "forecast_origin_utc"] += pd.Timedelta(hours=1)
    elif mutation == "naive_origin": frame["forecast_origin_utc"] = frame.forecast_origin_utc.dt.tz_localize(None)
    with pytest.raises(data.KPIDataError):
        data._normalise(frame, "test")


def test_spring_dst_missing_hour_is_incomplete():
    frame = data._normalise(panel(), "test")
    frame = frame.loc[frame["sample"].eq("evaluation")].copy()
    frame["model_id"] = "one"
    frame = frame.drop(frame.index[0])
    with pytest.raises(data.KPIDataError, match="No complete observed evaluation day"):
        data._recommended_end(frame)


def test_recommended_end_does_not_require_storm_complete():
    frame = data._normalise(panel(), "test")
    frame = frame.loc[frame["sample"].eq("evaluation")].copy()
    frame["model_id"] = "one"
    frame["benchmark_forecast"] = np.nan
    assert data._recommended_end(frame)[0] == "2026-03-29"


@pytest.mark.parametrize("value", ["../escape", "/outside/project"])
def test_paths_cannot_leave_project(tmp_path, value):
    with pytest.raises(data.KPIDataError):
        data._safe(tmp_path, value)


def test_link_sources_are_rejected(tmp_path):
    source = tmp_path / "source"
    source.write_text("data")
    link = tmp_path / "link"
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("OS account cannot create symlinks")
    with pytest.raises(data.KPIDataError, match="Linked"):
        data._safe(tmp_path, link)


def test_empty_sources_cannot_be_verified(tmp_path):
    with pytest.raises(data.KPIDataError, match="Missing captured"):
        data.verify_sources({"project_root": str(tmp_path), "sources": []})


def production_fixture(tmp_path, monkeypatch, changed_column=None):
    from economic_value import data as old_data

    baseline = data._normalise(panel(), "production fixture")
    sources = []
    for zone in data.ZONES:
        directory = tmp_path / "runs/exports/2026-03-30" / zone.lower() / "nuclear_kalman"
        directory.mkdir(parents=True)
        path = directory / f"forecast_{zone.lower()}_2026-03-30_nuclear_kalman.html"
        path.write_text("<html>synthetic source</html>")
        sources.append({"zone": zone, "model": "nuclear_kalman", "path": str(path), "sha256": digest(path)})

    def read_report(path, *, zone, model):
        assert model == "nuclear_kalman"
        frame = baseline.loc[baseline.zone.eq(zone), ["timestamp_utc", "forecast", "actual", "benchmark_forecast"]].reset_index(drop=True)
        if changed_column:
            frame.loc[0, changed_column] += 0.01
        return frame, {"sha256": digest(path)}

    monkeypatch.setattr(old_data, "read_report", read_report)
    return baseline, {"source_data_audit": {"baseline": {"sources": sources, "delivery_day": "2026-03-30"}}}


def test_production_requires_exact_published_prices(tmp_path, monkeypatch):
    baseline, audit = production_fixture(tmp_path, monkeypatch)
    result = data._verify_production(data._Capture(tmp_path), audit, baseline)
    assert result["all_p50_and_references_identical_to_published_production"]
    assert len(result["sources"]) == 4


@pytest.mark.parametrize("changed_column", ["forecast", "actual", "benchmark_forecast"])
def test_good_production_sha_but_different_parsed_prices_refused(tmp_path, monkeypatch, changed_column):
    baseline, audit = production_fixture(tmp_path, monkeypatch, changed_column)
    with pytest.raises(data.KPIDataError, match="differs from the frozen baseline"):
        data._verify_production(data._Capture(tmp_path), audit, baseline)


def test_production_sha_mismatch_refused_before_parsing(tmp_path, monkeypatch):
    baseline, audit = production_fixture(tmp_path, monkeypatch)
    item = audit["source_data_audit"]["baseline"]["sources"][0]
    Path(item["path"]).write_text("a new production report")
    with pytest.raises(data.KPIDataError, match="SHA256 mismatch"):
        data._verify_production(data._Capture(tmp_path), audit, baseline)


def test_four_production_zones_are_required(tmp_path, monkeypatch):
    baseline, audit = production_fixture(tmp_path, monkeypatch)
    audit["source_data_audit"]["baseline"]["sources"].pop()
    with pytest.raises(data.KPIDataError, match="Four explicit"):
        data._verify_production(data._Capture(tmp_path), audit, baseline)
