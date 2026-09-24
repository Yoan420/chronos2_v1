"""Isolated report tests: source integrity, honest denominators and safe rendering."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from nyx_congestion.report import (MODELS, assemble_panel, build_report,
    congestion_metrics, constraint_case, domain_coverage, label_timing, price_diagnostics, regional_metrics, render_report,
    stage2_fit_diagnostics)

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def source(tmp_path):
    source, ancestor = tmp_path/"physical", tmp_path/"ancestor"
    source.mkdir()
    ancestor.mkdir()
    timestamps = pd.date_range("2026-09-13", periods=72, freq="h", tz="Europe/Paris").tz_convert("UTC")
    local = timestamps.tz_convert("Europe/Paris").tz_localize(None)
    origins = (local.normalize()-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    base = pd.DataFrame({"zone": "FR", "timestamp_utc": timestamps, "forecast_origin_utc": origins,
        "actual": 100., "forecast": 90., "benchmark_forecast": 80., "q10": 70., "q90": 110.,
        "feature_test": np.arange(72), "sample": np.where(local.day < 15, "evaluation", "live")})
    base.to_parquet(source/"panel.parquet", index=False)
    audit = {"source_config": {"delivery_day": "2026-09-15", "end_day": "2026-09-14", "zones": ["FR"],
        "evaluation_days": 365, "timezone": "Europe/Paris", "cutoff_time": "08:00"}}
    (ancestor/"source_audit.json").write_text(json.dumps(audit), encoding="utf8")
    manifest = {"input_files": {"panel.parquet": sha(source/"panel.parquet")},
        "source_files": {"source_audit.json": sha(ancestor/"source_audit.json")}, "source_dir": str(ancestor)}
    (source/"manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    base.assign(network_fuel_direct=93., nyx_physical_p50=92.).to_parquet(source/"predictions.parquet", index=False)
    results = {"status": "completed", "suite_manifest_sha256": sha(source/"manifest.json"),
        "result_files": {"predictions.parquet": sha(source/"predictions.parquet")}}
    (source/"results_manifest.json").write_text(json.dumps(results), encoding="utf8")
    predictions = base.assign(**{model: 95. for model in MODELS},
        **{f"{model}_{q}": 70. if q == "q10" else 115. for model in MODELS for q in ("q10", "q90")})
    constraints = pd.DataFrame({"timestamp_utc": timestamps, "forecast_origin_utc": origins,
        "constraint_key": "eic-a|eic-b|DIRECT", "cne_name": "Test CNEC", "label_active": np.arange(72)%2 == 0,
        "label_shadow_price": np.where(np.arange(72)%2 == 0, 20., 0.), "label_eligible": True,
        "label_available_at_utc": origins+pd.Timedelta(hours=6),
        "activation_probability": np.where(np.arange(72)%2 == 0, .8, .2),
        "intensity_if_active": 20., "expected_shadow_price": np.where(np.arange(72)%2 == 0, 16., 4.),
        "climatology_probability": .5, "expert_ready": True, "fit_day": "2026-09-12"})
    return source, predictions, constraints


def metric(frame, **kwargs):
    return congestion_metrics(frame, end_day="2026-09-14", days=365,
                              evaluation_timestamps=frame.timestamp_utc.unique(), **kwargs)


def test_source_preserved_and_exact_comparators(source):
    folder, predictions, _ = source
    before = predictions.copy(deep=True)
    long, wide, _, seals = assemble_panel(predictions, folder)
    assert len(long) == 72*7 and wide.network_fuel_direct.eq(93).all()
    assert wide.nyx_physical_p50.eq(92).all() and wide.nuclear_kalman.eq(90).all()
    assert len(seals) == 5
    pd.testing.assert_frame_equal(predictions, before)


@pytest.mark.parametrize("name", ["actual", "forecast", "benchmark_forecast", "q10", "feature_test"])
def test_changed_original_panel_rejected(source, name):
    folder, predictions, _ = source
    predictions.loc[0, name] = -99.
    with pytest.raises(ValueError, match="Changed"):
        assemble_panel(predictions, folder)


def test_missing_preserved_feature_rejected(source):
    folder, predictions, _ = source
    with pytest.raises(ValueError, match="preserved panel"):
        assemble_panel(predictions.drop(columns="feature_test"), folder)


def test_comparator_checksum_rejected(source):
    folder, predictions, _ = source
    with (folder/"predictions.parquet").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        assemble_panel(predictions, folder)


@pytest.mark.parametrize("operation", ["duplicate", "naive", "infinite", "unordered"])
def test_invalid_price_panel_rejected(source, operation):
    folder, predictions, _ = source
    if operation == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[[0]]])
    elif operation == "naive":
        predictions["forecast_origin_utc"] = predictions.forecast_origin_utc.dt.tz_localize(None)
    elif operation == "infinite":
        predictions.loc[0, "nyx_congestion"] = np.inf
    else:
        predictions.loc[0, "nyx_congestion_q10"] = 120.
    with pytest.raises(ValueError):
        assemble_panel(predictions, folder)


def test_classification_and_intensity_metrics(source):
    _, _, frame = source
    before = frame.copy(deep=True)
    result = metric(frame)
    assert result["coverage"]["n_common_scored"] == 48
    expert, prior = result["scores"]
    assert expert["brier"] == pytest.approx(.04)
    assert prior["brier"] == pytest.approx(.25)
    assert expert["average_precision"] == 1
    assert prior["average_precision"] == .5
    assert expert["n_active"] == 24
    assert result["intensity"]["global_mae_expected_shadow_price"] == 4
    assert result["intensity"]["global_rmse_expected_shadow_price"] == 4
    assert result["intensity"]["active_mae_conditional_intensity"] == 0
    assert result["intensity"]["zero_prediction_global_mae"] == 10
    assert sum(r["n_rows"] for r in result["reliability"] if r["model_id"] == "congestion") == 48
    pd.testing.assert_frame_equal(frame, before)


def test_common_cnec_support_no_missing_labels_as_zero(source):
    _, _, frame = source
    frame["label_active"] = frame.label_active.astype(object)
    frame.loc[0, "label_active"] = None
    frame.loc[1, "label_eligible"] = False
    frame.loc[2, "label_available_at_utc"] = pd.NaT
    frame.loc[3, "climatology_probability"] = np.nan
    frame.loc[4, "expert_ready"] = False
    result = metric(frame)
    assert result["coverage"]["n_initial_constraint_hours"] == 48
    assert result["coverage"]["n_common_scored"] == 43
    assert result["coverage"]["n_unknown_or_unqualified_labels"] == 3
    assert result["coverage"]["n_qualified_but_not_common"] == 2
    assert {r["n_rows"] for r in result["scores"]} == {43}


def test_no_duplicate_country_count_or_future_scoring(source):
    _, _, frame = source
    result = congestion_metrics(frame, end_day="2026-09-14", days=1,
        evaluation_timestamps=np.tile(frame.timestamp_utc.unique(), 4))
    assert result["coverage"]["n_common_scored"] == 24
    assert result["scores"][0]["n_active"] == 12


@pytest.mark.parametrize("active", [True, False])
def test_ap_single_class_is_not_discrimination_claim(source, active):
    _, _, frame = source
    frame["label_active"] = active
    result = metric(frame)
    assert result["scores"][0]["average_precision"] is None
    assert result["scores"][0]["brier"] is not None
    assert not result["pr_curves"]


@pytest.mark.parametrize("field,value", [("activation_probability", 1.01), ("climatology_probability", -.1),
    ("intensity_if_active", -1), ("expected_shadow_price", 999), ("label_active", "false"),
    ("label_shadow_price", np.inf)])
def test_invalid_constraint_signal_rejected(source, field, value):
    _, _, frame = source
    if field == "label_active":
        frame[field] = frame[field].astype(object)
    frame.loc[0, field] = value
    with pytest.raises(ValueError):
        metric(frame)


def test_constraint_duplicates_and_naive_rejected(source):
    _, _, frame = source
    with pytest.raises(ValueError, match="Duplicate"):
        metric(pd.concat([frame, frame.iloc[[0]]]))
    frame["label_available_at_utc"] = frame.label_available_at_utc.dt.tz_localize(None)
    with pytest.raises(ValueError, match="aware"):
        metric(frame)


def test_case_selected_only_for_display_and_qualified(source):
    _, _, frame = source
    before = frame.copy(deep=True)
    result = constraint_case(frame)
    assert len(result["rows"]) == 24
    assert "display only" in result["selection"]
    assert len(result["constraint_keys"]) == 1
    pd.testing.assert_frame_equal(frame, before)


def test_price_diagnostics_common_support(source):
    folder, predictions, _ = source
    predictions.loc[0, "nyx_congestion"] = np.nan
    _, wide, _, _ = assemble_panel(predictions, folder)
    result = price_diagnostics(wide, end_day="2026-09-14", days=365, zones=["FR"])
    assert {r["n_hours"] for r in result["interventions"]} == {47}


def test_safe_html_and_exclusive_output(tmp_path):
    path = tmp_path/"report.html"
    render_report({"unsafe": "</script><script>alert(1)</script>", "nan": np.nan}, path)
    text = path.read_text(encoding="utf8")
    assert "</script><script>alert" not in text and "\\u003c/script\\u003e" in text
    assert "<script src=" not in text and "cdn." not in text
    assert "Mode nuit" in text and "non exécutable" in text and "PR-AUC" in text
    assert "Production inchangée" in text and "ce n'est ni le prix" in text
    with pytest.raises(FileExistsError):
        render_report({}, path)


def setup_root(tmp_path):
    root = tmp_path/"root"
    (root/"config").mkdir(parents=True)
    shutil.copyfile(ROOT/"config/economic_value.yaml", root/"config/economic_value.yaml")
    return root


def test_build_report_frozen_pairing_policy_and_live_exclusion(source, tmp_path):
    folder, predictions, constraints = source
    root = setup_root(tmp_path)
    dest = root/"runs/experiments/nyx_congestion_v1/test-report"
    result = build_report(predictions, folder, dest, root=root, audit={}, constraint_predictions=constraints)
    payload = json.loads(Path(result["metrics_path"]).read_text(encoding="utf8"))
    rows = payload["periods"]["365"]["rows"]
    assert len(rows) == 16
    assert {r["n_hours"] for r in rows} == {48}
    assert next(r for r in rows if r["zone"] == "FR" and r["model_id"] == "nyx_congestion")["mae_eur_mwh"] == 5
    assert payload["periods"]["365"]["congestion"]["coverage"]["n_common_scored"] == 48
    assert payload["periods"]["365"]["economic"]["audit"]["zone_capacity_mw"]["FR"] == 25
    assert len(payload["case"]) == 24
    assert payload["production_modified"] is False and payload["independent_validation"] is False
    assert payload["label_contract"]["activation_epsilon"] == 1e-9
    with pytest.raises(FileExistsError):
        build_report(predictions, folder, dest, root=root, audit={}, constraint_predictions=constraints)


def test_report_rejects_nonisolated_output(source, tmp_path):
    folder, predictions, constraints = source
    with pytest.raises(ValueError):
        build_report(predictions, folder, tmp_path/"prod", root=tmp_path, audit={}, constraint_predictions=constraints)


def test_report_rejects_shifted_constraint_origin(source, tmp_path):
    folder, predictions, constraints = source
    root = setup_root(tmp_path)
    constraints.loc[0, "forecast_origin_utc"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="support/origin"):
        build_report(predictions, folder, root/"runs/experiments/nyx_congestion_v1/fail",
                     root=root, audit={}, constraint_predictions=constraints)


def test_domain_coverage_ratio_of_sums_and_zero_fr_denominator():
    timestamps = pd.to_datetime(["2026-09-14 16:00Z", "2026-09-14 17:00Z"]*2)
    labels = pd.DataFrame({"timestamp_utc": timestamps, "zone": ["DE", "DE", "FR", "FR"],
        "label_absolute_contribution_eur_mwh": [100., 10., 0., 0.],
        "initial_covered_absolute_contribution_eur_mwh": [50., 10., 0., 0.],
        "outside_initial_absolute_contribution_eur_mwh": [50., 0., 0., 0.],
        "label_eligible": True, "label_physical_matching_complete": True, "initial_hour_available": True})
    result = domain_coverage(labels, end_day="2026-09-14", days=365,
                             zones=["DE", "FR"], evaluation_timestamps=timestamps.unique())
    de = next(row for row in result["rows"] if row["zone"] == "DE")
    fr = next(row for row in result["rows"] if row["zone"] == "FR")
    assert de["coverage_ratio"] == pytest.approx(60/110)
    assert de["coverage_ratio"] != .75
    assert fr["coverage_ratio"] is None and fr["n_qualified_hours"] == 2
    assert len(result["case"]) == 2
    assert "NOT the model" in result["reference"]
    labels.loc[0, "initial_hour_available"] = False
    changed = domain_coverage(labels, end_day="2026-09-14", days=365,
                              zones=["DE", "FR"], evaluation_timestamps=timestamps.unique())
    de = next(row for row in changed["rows"] if row["zone"] == "DE")
    assert de["coverage_ratio"] == 1 and de["n_unqualified_hours"] == 1
    labels.loc[1, "outside_initial_absolute_contribution_eur_mwh"] = 99.
    with pytest.raises(ValueError, match="do not sum"):
        domain_coverage(labels, end_day="2026-09-14", days=365,
                        zones=["DE", "FR"], evaluation_timestamps=timestamps.unique())


def test_domain_missing_sidecar_is_unavailable_not_zero():
    result = domain_coverage(None, end_day="2026-09-14", days=365, zones=["FR"], evaluation_timestamps=[])
    assert result["available"] is False and result["rows"] == []


@pytest.fixture
def regional(source):
    _, _, frame = source
    parts = []
    for zone, actual, probability, intensity in (("FR", 0., .1, 55.), ("DE", 100., .8, 90.),
                                                ("BE", 50., .6, 60.), ("NL", 20., .3, 55.)):
        parts.append(frame.assign(zone=zone, realised_fb_premium=actual, label_active=actual >= 50,
            label_shadow_price=actual if actual >= 50 else 0., activation_probability=probability,
            intensity_if_active=intensity, expected_shadow_price=probability*intensity))
    return pd.concat(parts, ignore_index=True).drop(columns="constraint_key")


def regional_scores(frame):
    return regional_metrics(frame, end_day="2026-09-14", days=365,
        zones=["FR", "DE", "BE", "NL"], evaluation_timestamps=frame.timestamp_utc.unique())


def test_regional_pressure_is_distinct_from_cnec_and_exact_threshold(regional):
    before = regional.copy(deep=True)
    result = regional_scores(regional)
    assert result["available"] and result["threshold_eur_mwh"] == 50
    all_row = next(row for row in result["rows"] if row["zone"] == "ALL" and row["model_id"] == "regional")
    assert all_row["n_rows"] == 48*4 and all_row["n_active"] == 48*2
    assert all_row["average_precision"] == 1
    assert all_row["global_rmse_expected_pressure"] is not None
    assert len(result["case"]) == 4
    assert next(row for row in result["case"] if row["zone"] == "BE")["label_active"] is True
    assert next(row for row in result["case"] if row["zone"] == "NL")["realised_fb_premium"] == 20
    assert all("country-hour" in row["unit"] for row in result["coverage"])
    pd.testing.assert_frame_equal(regional, before)


def test_regional_missing_country_must_not_redefine_minimum(regional):
    with pytest.raises(ValueError, match="all four"):
        regional_scores(regional.loc[regional.zone.ne("FR")])
    regional.loc[0, "label_eligible"] = False
    with pytest.raises(ValueError, match="all four"):
        regional_scores(regional)


def test_regional_minimum_and_severe_label_contract(regional):
    shifted = regional.copy()
    shifted["realised_fb_premium"] += 5.
    with pytest.raises(ValueError, match="four-country minimum"):
        regional_scores(shifted)
    regional.loc[regional.zone.eq("BE"), "realised_fb_premium"] = 49.
    with pytest.raises(ValueError, match="G>=50"):
        regional_scores(regional)


def test_regional_all_four_unknown_excluded_not_filled(regional):
    stamp = regional.timestamp_utc.iloc[0]
    regional.loc[regional.timestamp_utc.eq(stamp), "label_eligible"] = False
    result = regional_scores(regional)
    row = next(row for row in result["rows"] if row["zone"] == "ALL" and row["model_id"] == "regional")
    assert row["n_rows"] == 47*4
    absent = regional_metrics(None, end_day="2026-09-14", days=365, zones=["FR"], evaluation_timestamps=[])
    assert absent["available"] is False and absent["rows"] == []


def test_label_timing_counts_distinct_hours_and_does_not_fill_unknown():
    times = pd.to_datetime(["2026-09-13 17:00Z", "2026-09-14 17:00Z", "2026-09-15 17:00Z"]*2)
    frame = pd.DataFrame({"timestamp_utc": times, "zone": ["DE"]*3+["BE"]*3,
        "label_available_at_utc": pd.to_datetime(["2026-09-12 12:00Z", "2026-09-16 06:00Z", "2026-09-16 08:00Z"]*2),
        "label_eligible": True})
    before = frame.copy(deep=True)
    result = label_timing(frame, end_day="2026-09-14", days=365, zones=["DE", "BE"], evaluation_timestamps=times.unique())
    row = result["rows"][0]
    assert row["n_qualified_label_hours"] == 4 and row["n_delayed_label_hours"] == 2
    assert row["n_distinct_delayed_timestamps"] == 1 and row["delayed_fraction"] == .5
    assert row["maximum_delay_hours"] == 48 and row["maximum_delay_days"] == 2
    assert len(result["monthly"]) == 3
    pd.testing.assert_frame_equal(frame, before)
    frame.loc[1, "label_available_at_utc"] = pd.NaT
    result = label_timing(frame, end_day="2026-09-14", days=365, zones=["DE", "BE"], evaluation_timestamps=times.unique())
    assert result["rows"][0]["n_unknown_label_hours"] == 1
    assert result["rows"][0]["n_delayed_label_hours"] == 1


def test_label_timing_uses_civil_next_origin_at_dst_transition():
    frame = pd.DataFrame({"zone": ["DE"], "timestamp_utc": pd.to_datetime(["2026-03-29 17:00Z"]),
        "label_available_at_utc": pd.to_datetime(["2026-03-29 06:30Z"]), "label_eligible": True})
    result = label_timing(frame, end_day="2026-03-29", days=1, zones=["DE"], evaluation_timestamps=frame.timestamp_utc)
    assert result["rows"][0]["n_delayed_label_hours"] == 1
    assert result["rows"][0]["maximum_delay_hours"] == .5
    frame.loc[0, "label_available_at_utc"] = pd.Timestamp("2026-03-29 06:00Z")
    exact = label_timing(frame, end_day="2026-03-29", days=1, zones=["DE"], evaluation_timestamps=frame.timestamp_utc)
    assert exact["rows"][0]["n_delayed_label_hours"] == 0
    assert exact["rows"][0]["maximum_delay_hours"] == 0


@pytest.fixture
def fold_snapshot(tmp_path, source):
    _, wide, _ = source
    directory = tmp_path/"runs/experiments/nyx_congestion_v1/snapshots/test"
    directory.mkdir(parents=True)
    (directory/"manifest.json").write_text('{"snapshot":"test"}', encoding="utf8")
    records = []
    for strategy in ("control", "congestion"):
        records.extend([{"strategy": strategy, "fit_day": day, "status": status, "reason": reason}
            for day, status, reason in (("2026-09-07", "trained", ""),
                ("2026-09-14", "fallback", "insufficient_distinct_calibration_events"),
                ("2026-09-21", "trained", ""))])
    pd.DataFrame(records).to_parquet(directory/"folds.parquet", index=False)
    result = {"status": "completed", "suite_manifest_sha256": sha(directory/"manifest.json"),
              "result_files": {"folds.parquet": sha(directory/"folds.parquet")}}
    (directory/"results_manifest.json").write_text(json.dumps(result), encoding="utf8")
    audit = {"snapshot": str(directory), "suite_manifest_sha256": sha(directory/"manifest.json")}
    return directory, audit, wide.assign(control_expert_ready=False, congestion_expert_ready=False)


def test_stage2_last_attempt_is_not_last_success_and_future_fits_excluded(fold_snapshot, tmp_path):
    _, audit, wide = fold_snapshot
    seals = {}
    result = stage2_fit_diagnostics(wide, audit, root=tmp_path, seals=seals, end_day="2026-09-14", zones=["FR"])
    assert result["available"] and len(seals) == 3
    for row in result["case_fits"]:
        assert row["fit_day"] == "2026-09-14" and row["status"] == "fallback"
        assert row["reason"] == "insufficient_distinct_calibration_events"
        assert row["last_trained_fit_day"] == "2026-09-07"
        assert row["annual_trained_fits"] == 1 and row["annual_fits"] == 2
    for row in result["readiness"]:
        assert row["annual_country_hours"] == 48
        assert row["annual_ready_country_hours"] == 0 and row["case_ready_country_hours"] == 0
        assert row["case_country_hours"] == 1


@pytest.mark.parametrize("tamper", ["folds", "suite", "results"])
def test_stage2_fold_provenance_strict(fold_snapshot, tmp_path, tamper):
    directory, audit, wide = fold_snapshot
    if tamper == "folds":
        with (directory/"folds.parquet").open("ab") as handle:
            handle.write(b"tamper")
    elif tamper == "suite":
        audit["suite_manifest_sha256"] = "wrong"
    else:
        result = json.loads((directory/"results_manifest.json").read_text(encoding="utf8"))
        result["suite_manifest_sha256"] = "other-snapshot"
        (directory/"results_manifest.json").write_text(json.dumps(result), encoding="utf8")
    with pytest.raises(ValueError):
        stage2_fit_diagnostics(wide, audit, root=tmp_path, seals={}, end_day="2026-09-14", zones=["FR"])
