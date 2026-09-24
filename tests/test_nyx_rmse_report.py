from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
import pytest

from nyx_rmse.report import CANDIDATES, CATALOG, _extra_metrics, assemble_panel, build_report, render_report


ROOT = Path(__file__).resolve().parents[1]


def seal(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    start = pd.Timestamp("2026-09-13", tz="Europe/Paris")
    timestamps = pd.date_range(start, start+pd.DateOffset(days=3), freq="h", inclusive="left").tz_convert("UTC")
    local = timestamps.tz_convert("Europe/Paris").tz_localize(None)
    origins = (local.normalize()-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    baseline = pd.DataFrame({"zone": "FR", "timestamp_utc": timestamps, "forecast_origin_utc": origins,
        "forecast": 90., "actual": 100., "benchmark_forecast": 80., "q10": 85., "q90": 95.,
        "sample": np.where(local.date < pd.Timestamp("2026-09-15").date(), "evaluation", "live")})
    baseline.to_parquet(directory / "panel.parquet", index=False)
    source_audit = {"source_config": {"delivery_day": "2026-09-15", "end_day": "2026-09-14", "zones": ["FR"],
                                     "evaluation_days": 365, "timezone": "Europe/Paris", "cutoff_time": "08:00"}}
    (directory / "source_audit.json").write_text(json.dumps(source_audit), encoding="utf-8")
    manifest = {"input_files": {name: seal(directory/name) for name in ("panel.parquet", "source_audit.json")}}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for kind in ("forest", "empirical"):
        folder = directory / kind
        folder.mkdir()
        baseline.assign(candidate_forecast=92.).to_parquet(folder / "predictions.parquet", index=False)
        (folder / "results_manifest.json").write_text(json.dumps({"status": "completed", "suite_manifest_sha256": seal(directory/"manifest.json"),
            "result_files": {"predictions.parquet": seal(folder/"predictions.parquet")}}), encoding="utf-8")
    predictions = baseline.assign(**{name: 150. for name in CANDIDATES}, expert_ready=True)
    return directory, predictions


def test_assemble_keeps_baseline_and_all_mean_points_even_outside_interval(source):
    directory, predictions = source
    original = predictions.copy(deep=True)
    long, wide, audit, seals = assemble_panel(predictions, directory)
    assert len(long) == 72 * 7
    assert len(wide) == 72
    assert set(long.model_id) == {item["id"] for item in CATALOG if item["id"] != "__storm__"}
    assert wide.nyx_rmse.gt(wide.q90).all()
    assert long.storm.eq(80).all()
    assert wide.nuclear_kalman.eq(90).all()
    assert wide.coherent_forest_direct.eq(92).all()
    assert len(seals) == 7
    assert audit["source_config"]["delivery_day"] == "2026-09-15"
    pd.testing.assert_frame_equal(predictions, original)


@pytest.mark.parametrize("name", ["actual", "forecast", "benchmark_forecast", "q10", "sample"])
def test_assemble_rejects_reference_and_provenance_changes(source, name):
    directory, predictions = source
    predictions.loc[0, name] = "live" if name == "sample" else -99.
    with pytest.raises(ValueError, match="changed shared"):
        assemble_panel(predictions, directory)


def test_assemble_rejects_duplicate_physical_hour(source):
    directory, predictions = source
    with pytest.raises(ValueError, match="unique"):
        assemble_panel(pd.concat([predictions, predictions.iloc[[0]]]), directory)


def test_assemble_rejects_naive_origin(source):
    directory, predictions = source
    predictions["forecast_origin_utc"] = predictions.forecast_origin_utc.dt.tz_localize(None)
    with pytest.raises(ValueError, match="aware"):
        assemble_panel(predictions, directory)


def test_assemble_rejects_corrupted_comparator(source):
    directory, predictions = source
    with (directory / "forest" / "predictions.parquet").open("ab") as handle:
        handle.write(b"unexpected mutation")
    with pytest.raises(ValueError, match="checksum"):
        assemble_panel(predictions, directory)


def test_diagnostics_use_common_hours_not_individual_model_support(source):
    directory, predictions = source
    predictions.loc[0, "nyx_rmse"] = np.nan
    _, wide, _, _ = assemble_panel(predictions, directory)
    result = _extra_metrics(wide, [], end_day="2026-09-14", days=2, zones=["FR"])
    for row in result["interventions"]:
        assert row["n_hours"] == 47
    rmse = next(row for row in result["interventions"] if row["model_id"] == "nyx_rmse" and row["zone"] == "FR")
    assert rmse["changed_hours"] == rmse["worsened_hours"] == 47
    assert rmse["mae_on_interventions_eur_mwh"] == 50
    assert rmse["nyx_mae_on_interventions_eur_mwh"] == 10
    assert next(row for row in result["expert_ready"] if row["model_id"] == "nyx_rmse")["n_hours"] == 47


def test_diagnostic_ready_is_not_nonzero_intervention(source):
    directory, predictions = source
    predictions.loc[:, "nyx_rmse"] = 90.
    predictions.loc[:10, "expert_ready"] = False
    _, wide, _, _ = assemble_panel(predictions, directory)
    result = _extra_metrics(wide, [], end_day="2026-09-14", days=2, zones=["FR"])
    assert next(r for r in result["expert_ready"] if r["model_id"] == "nyx_rmse")["n_hours"] == 37
    assert next(r for r in result["interventions"] if r["model_id"] == "nyx_rmse")["changed_hours"] == 0


def test_build_report_excludes_known_live_labels_and_uses_original_fixed_eva(source, tmp_path):
    directory, predictions = source
    (tmp_path / "config").mkdir()
    shutil.copyfile(ROOT/"config"/"economic_value.yaml", tmp_path/"config"/"economic_value.yaml")
    destination = tmp_path / "runs" / "experiments" / "nyx_rmse_v1" / "reports"
    result = build_report(predictions, directory, destination, root=tmp_path, audit={"actual_training_days_min": 91})
    payload = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))
    assert payload["diagnostic_only"] is True
    assert payload["production_modified"] is False
    assert payload["mean_is_not_p50"] is True
    rows = payload["periods"]["365"]["rows"]
    assert {row["n_hours"] for row in rows} == {48}
    economic = payload["periods"]["365"]["economic"]
    assert economic["audit"]["portfolio_capacity_mw"] == 100
    assert economic["audit"]["zone_capacity_mw"] == {zone: 25 for zone in ("FR", "DE", "BE", "NL")}
    assert economic["audit"]["signal_hurdle_eur_mwh"] == 6
    assert {row["n_country_hours"] for row in economic["rows"]} == {24}
    assert Path(result["report_path"]).is_file()
    with pytest.raises(FileExistsError):
        build_report(predictions, directory, destination, root=tmp_path, audit={})


def test_output_must_stay_in_isolated_experiment_namespace(source, tmp_path):
    directory, predictions = source
    with pytest.raises(ValueError):
        build_report(predictions, directory, tmp_path/"runs"/"exports", root=tmp_path, audit={})


def test_safe_json_no_external_scripts_no_overwrite(tmp_path):
    payload = {"source_directory": '</p><img src=x onerror="evil()">', "catalog": [], "audit": {"text": '</script><script>evil()</script>', "bad": np.nan}, "days": np.int64(365)}
    original = deepcopy(payload)
    path = render_report(payload, tmp_path/"report.html")
    markup = path.read_text(encoding="utf-8")
    match = re.search(r'<script id="rmse-data" type="application/json">(.*?)</script>', markup, re.DOTALL)
    parsed = json.loads(match[1])
    assert parsed["audit"]["bad"] is None
    assert parsed["audit"]["text"] == payload["audit"]["text"]
    assert '<script>evil()' not in markup
    assert '<img src=x' not in markup
    assert '\\u003c/script\\u003e' in match[1]
    assert "&lt;img" in markup
    assert '<script src=' not in markup and '<link ' not in markup
    assert not re.search(r'\b(fetch|XMLHttpRequest|WebSocket)\s*\(', markup)
    assert payload["source_directory"] == original["source_directory"]
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        render_report(payload, path)
    assert path.read_bytes() == before


def test_template_discloses_mean_gate_and_validation_limits(tmp_path):
    markup = render_report({}, tmp_path/"report.html").read_text(encoding="utf-8")
    for phrase in ("n'est pas un P50", "pas une promotion", "non gouvernés", "pas un signal disponible", "pas une preuve statistique à 95", "ne sont pas certifiés PIT", "n'isole pas uniquement une loss", "365 jours supplémentaires", "ne garantit donc pas 365 jours"):
        assert phrase in markup


def test_data_containing_template_markers_is_not_substituted(tmp_path):
    payload = {"source_directory": "@@DATA@@", "audit": {"text": "@@SOURCE@@"}}
    markup = render_report(payload, tmp_path/"report.html").read_text(encoding="utf-8")
    match = re.search(r'<script id="rmse-data" type="application/json">(.*?)</script>', markup, re.DOTALL)
    assert json.loads(match[1]) == payload
