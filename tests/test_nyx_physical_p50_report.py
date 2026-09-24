from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from nyx_physical_p50.report import MODELS, assemble_panel, build_report, diagnostics, render_report

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def source(tmp_path):
    folder = tmp_path/"source"
    folder.mkdir()
    timestamps = pd.date_range("2026-09-13", periods=72, freq="h", tz="Europe/Paris").tz_convert("UTC")
    local = timestamps.tz_convert("Europe/Paris").tz_localize(None)
    origins = (local.normalize()-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    base = pd.DataFrame({"zone": "FR", "timestamp_utc": timestamps, "forecast_origin_utc": origins,
        "actual": 100., "forecast": 90., "benchmark_forecast": 80., "q10": 70., "q90": 110.,
        "sample": np.where(local.day < 15, "evaluation", "live")})
    base.to_parquet(folder/"panel.parquet", index=False)
    source_audit = {"source_config": {"delivery_day": "2026-09-15", "end_day": "2026-09-14", "zones": ["FR"],
        "evaluation_days": 365, "timezone": "Europe/Paris", "cutoff_time": "08:00"}}
    (folder/"source_audit.json").write_text(json.dumps(source_audit), encoding="utf8")
    manifest = {"input_files": {n: sha(folder/n) for n in ("panel.parquet", "source_audit.json")}}
    (folder/"manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    forest = folder/"forest"
    forest.mkdir()
    for name, value in (("predictions.parquet", 93.), ("governed_predictions.parquet", 91.)):
        base.assign(candidate_forecast=value).to_parquet(forest/name, index=False)
    result = {"status": "completed", "suite_manifest_sha256": sha(folder/"manifest.json"),
        "result_files": {n: sha(forest/n) for n in ("predictions.parquet", "governed_predictions.parquet")}}
    (forest/"results_manifest.json").write_text(json.dumps(result), encoding="utf8")
    return folder, base.assign(**{name: 95. for name in MODELS})


def test_shared_panel_and_inputs_preserved(source):
    folder, predictions = source
    before = predictions.copy(deep=True)
    long, wide, _, seals = assemble_panel(predictions, folder)
    assert len(long) == 72*7
    assert wide.coherent_forest_direct.eq(93).all()
    assert wide.coherent_forest_governed.eq(91).all()
    assert wide.nuclear_kalman.eq(90).all()
    assert len(seals) == 6
    pd.testing.assert_frame_equal(predictions, before)


@pytest.mark.parametrize("name", ["actual", "forecast", "benchmark_forecast", "q10", "sample"])
def test_changed_comparison_data_rejected(source, name):
    folder, predictions = source
    predictions.loc[0, name] = "live" if name == "sample" else -100.
    with pytest.raises(ValueError, match="Changed shared"):
        assemble_panel(predictions, folder)


def test_duplicate_hour_rejected(source):
    folder, predictions = source
    with pytest.raises(ValueError, match="unique"):
        assemble_panel(pd.concat([predictions, predictions.iloc[[0]]]), folder)


def test_timezone_naive_rejected(source):
    folder, predictions = source
    predictions["forecast_origin_utc"] = predictions.forecast_origin_utc.dt.tz_localize(None)
    with pytest.raises(ValueError, match="aware"):
        assemble_panel(predictions, folder)


def test_infinity_rejected(source):
    folder, predictions = source
    predictions.loc[0, "nyx_physical_p50"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        assemble_panel(predictions, folder)


def test_comparator_checksum_rejected(source):
    folder, predictions = source
    with (folder/"forest/governed_predictions.parquet").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        assemble_panel(predictions, folder)


def test_diagnostics_common_support_and_live_excluded(source):
    folder, predictions = source
    predictions.loc[0, "nyx_physical_p50"] = np.nan
    _, wide, _, _ = assemble_panel(predictions, folder)
    data = diagnostics(wide, end_day="2026-09-14", days=2, zones=["FR"])
    assert {r["n_hours"] for r in data["interventions"]} == {47}
    assert all(r["better"] == 47 for r in data["interventions"] if r["model_id"] in MODELS)


def test_safe_render_no_external_script_or_injection(tmp_path):
    path = tmp_path/"report.html"
    render_report({"unsafe": "</script><script>alert(1)</script>", "nan": np.nan}, path)
    text = path.read_text(encoding="utf8")
    assert "</script><script>alert" not in text
    assert "\\u003c/script\\u003e" in text
    assert "<script src=" not in text and "cdn." not in text
    assert "Mode nuit" in text and "post-couplage" in text
    with pytest.raises(FileExistsError):
        render_report({}, path)


def test_real_kpi_report_excludes_live_and_keeps_economic_policy(source, tmp_path):
    folder, predictions = source
    root = tmp_path/"root"
    (root/"config").mkdir(parents=True)
    shutil.copyfile(ROOT/"config/economic_value.yaml", root/"config/economic_value.yaml")
    dest = root/"runs/experiments/nyx_physical_p50_v1/test-report"
    result = build_report(predictions, folder, dest, root=root, audit={"diagnostic_only": True})
    payload = json.loads(Path(result["metrics_path"]).read_text(encoding="utf8"))
    assert set(payload["periods"]) == {"365", "7"}
    row = next(r for r in payload["periods"]["365"]["rows"] if r["zone"] == "FR" and r["model_id"] == "nyx_physical_p50")
    assert row["n_hours"] == 48
    assert row["mae_eur_mwh"] == 5
    assert len(payload["case"]) == 24
    assert payload["production_modified"] is False and payload["independent_validation"] is False
    assert payload["periods"]["365"]["economic"]["audit"]["zone_capacity_mw"]["FR"] == 25
    assert "P50 précédent · gouverné" in Path(result["report_path"]).read_text(encoding="utf8")


def test_output_outside_namespace_rejected(source, tmp_path):
    folder, predictions = source
    with pytest.raises(ValueError):
        build_report(predictions, folder, tmp_path/"outside", root=tmp_path, audit={})
