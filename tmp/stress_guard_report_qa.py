"""Validate saved StressGuard reports without rerendering, fitting or API reads.

Only create-only QA JSON files inside the supplied NEW StressGuard snapshot
are written. Node VM tests execute saved UI logic against DOM/Plotly doubles;
they are behavioural checks, not browser screenshots or pixel verification.
"""
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from economic_value.data import _plots
from marginal_cost_expert.evaluation import _array, _trace_index
from nyx_scarcity.reporting import _prepare
from nyx_scarcity_zonal.reporting import _verified_storm
from nyx_stress_guard import runner
from nyx_stress_guard.policy import validate_output

NODE = Path(r"C:\Users\BQ6757\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")
PRODUCTION_HTML_BEFORE = {
    "be": "8177031a943ff1522beef089bedf6b3cbfdbcec9d3230a97f142e12eefb3ab01",
    "de": "ad84143e4f05fd8d5d6d0e4a747a80ea9fbe85ae4a3db5dd87565d39422039ed",
    "fr": "f6e6107ff4e4d8befab3db75137e631313d8e68d077eb9544f87261bae1cd131",
    "nl": "9b55b26bfc99f483946e3bae67b6b58dc09bf0119e8113a701477fa8c2907d28",
}
START, END, LIVE = "2025-09-15", "2026-09-14", "2026-09-15"
VARIANTS = ("physics_governed", "physics_direct")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    np.testing.assert_allclose(a, b, atol=1e-8, rtol=1e-10, equal_nan=True)


def run_node(script, *, payload=None):
    call = subprocess.run([str(NODE), "-e", script], input=json.dumps(payload) if payload is not None else "",
                          encoding="utf8", text=True, capture_output=True, timeout=120)
    assert call.returncode == 0, call.stderr
    return json.loads(call.stdout)


def check_decisions(frame, *, direct):
    validate_output(frame)
    close(frame.candidate_forecast, frame.forecast+frame.applied_correction)
    close(frame.applied_correction, frame.selected_weight*frame.bounded_correction)
    proposed = frame.expert_ready & frame.spike_probability.gt(.5)
    close(frame.loc[proposed, "raw_correction"], frame.loc[proposed, "mixture_error_q50"])
    assert frame.loc[~proposed, "raw_correction"].eq(0.).all()
    close(frame.bounded_correction, frame.raw_correction.clip(lower=0, upper=400.))
    if direct:
        close(frame.selected_weight, proposed.astype(float))
    assert frame.selected_weight.isin([0., .25, .5, 1.]).all()
    assert frame.loc[~proposed, "selected_weight"].eq(0.).all()
    weights, active = frame.selected_weight.to_numpy(), frame.applied_correction.gt(0).to_numpy()
    for q in (10, 90):
        nyx = frame[f"q{q}"].to_numpy(float)
        expert = frame.forecast.to_numpy(float)+np.minimum(frame[f"mixture_error_q{q}"].to_numpy(float), 400.)
        shifted = (1-weights)*nyx+weights*expert
        union = np.minimum(nyx, shifted) if q == 10 else np.maximum(nyx, shifted)
        expected = np.where(active, union, nyx)
        close(frame[f"precalibration_q{q}"], expected)
    lower = frame.interval_calibration_lower_expansion_eur_mwh
    upper = frame.interval_calibration_upper_expansion_eur_mwh
    assert lower.ge(0).all() and upper.ge(0).all()
    close(frame.candidate_q10, frame.precalibration_q10-lower)
    close(frame.candidate_q90, frame.precalibration_q90+upper)
    assert pd.to_datetime(frame.interval_calibration_cutoff_utc, utc=True).eq(frame.forecast_origin_utc).all()
    available = pd.to_datetime(frame.interval_calibration_max_label_available_at_utc, utc=True)
    assert (available.isna() | available.le(frame.forecast_origin_utc)).all()
    assert frame.candidate_q10.le(frame.q10+1e-9).all() and frame.candidate_q90.ge(frame.q90-1e-9).all()
    finite = frame.actual.notna()
    nyx_covered = frame.actual.between(frame.q10, frame.q90)
    guarded = frame.actual.between(frame.candidate_q10, frame.candidate_q90)
    assert not (finite & nyx_covered & ~guarded).any()
    return {"arithmetic_and_strong_probability_gate": True, "direct": direct,
            "precalibration_weighted_quantile_envelope_recomputed": True,
            "saved_one_sided_expansions_recomputed": True, "known_labels_before_calibration_cutoff": True,
            "nyx_interval_contained_pathwise": True, "p50_unchanged_by_interval_expansion": True}


def interval_summary(frame):
    result = {}
    for regime in ("all", "active", "inactive"):
        b = frame if regime == "all" else frame.loc[frame.applied_correction.gt(0).eq(regime == "active")]
        if b.empty:
            result[regime] = {"hours": 0}
            continue
        low, high, actual = b.candidate_q10, b.candidate_q90, b.actual
        result[regime] = {"hours": len(b), "coverage": actual.between(low, high).mean(),
            "nyx_same_hours_coverage": actual.between(b.q10, b.q90).mean(),
            "mean_width": (high-low).mean(), "nyx_mean_width": (b.q90-b.q10).mean(),
            "interval_score_80": (high-low+10*(low-actual).clip(lower=0)+10*(actual-high).clip(lower=0)).mean()}
    return result


def check_statistics(statistics, evaluation):
    payload = json.JSONDecoder().raw_decode(statistics.split("const payload =", 1)[1].lstrip())[0]
    daily = [r for r in payload["records"] if r["sample"] == "daily"]
    assert len(daily) == 365 and {r["period_key"] for r in daily} == set(pd.date_range(START, END).strftime("%Y-%m-%d"))
    for row in daily:
        block = evaluation.loc[evaluation.local_day.eq(row["period_key"])]
        has_storm = block.benchmark_forecast.notna().any()
        selected = block.dropna(subset=["actual", "candidate_forecast", "benchmark_forecast"] if has_storm else ["actual", "candidate_forecast"])
        assert row["n"] == len(selected)
        error = selected.candidate_forecast-selected.actual
        for key, value in (("mean_price", selected.candidate_forecast.mean()), ("observed_mean_price", selected.actual.mean()),
                           ("mae", error.abs().mean()), ("rmse", np.sqrt((error**2).mean())), ("bias", error.mean())):
            close(row[key], value)
        if has_storm:
            assert row["benchmark_n"] == len(selected)
            close(row["benchmark_mean_price"], selected.benchmark_forecast.mean())
            close(row["benchmark_mae"], (selected.benchmark_forecast-selected.actual).abs().mean())
        else:
            assert row["benchmark_n"] == 0 and row["benchmark_mean_price"] is None
    autumn = next(r for r in daily if r["period_key"] == "2025-10-26")
    spring = next(r for r in daily if r["period_key"] == "2026-03-29")
    assert autumn["benchmark_n"] == 24 and spring["n"] == 23
    return {"daily_records": len(daily), "start": START, "end": END,
            "all_daily_means_mae_rmse_bias_and_storm_counts_recomputed": True,
            "autumn25_physical_hours24_storm_pairs": True, "spring23_hours_preserved": True}


def trace_matches(trace, index, values, missing=None):
    if len(trace.get("x", [])) != len(index):
        return False
    try:
        axis = _trace_index(trace["x"], timezone="Europe/Paris", missing=missing)
    except ValueError:
        return False
    return axis.equals(index) and np.allclose(_array(trace.get("y", [])), values, atol=1e-8, rtol=1e-10, equal_nan=True)


def comparison_qa(path):
    original = (ROOT/"tmp/coherent_p50_comparison_qa.js").read_text(encoding="utf8")
    adapted = original.replace("const sourcePath = process.argv[2];", "const sourcePath = "+json.dumps(str(path))+";")
    adapted = adapted.replace("'forest'", "'physics_governed'")
    # Disable the old tool's write. The parent Python process publishes all
    # validation JSON only after every check has succeeded and sources match.
    lines = adapted.splitlines()
    writes = [line for line in lines if line.startswith("fs.writeFileSync(")]
    assert len(writes) == 1
    adapted = "\n".join("console.log(JSON.stringify(result));" if line.startswith("console.log(") else line
                         for line in lines if not line.startswith("fs.writeFileSync("))
    return run_node(adapted)


def main():
    assert len(sys.argv) == 2, "Usage: stress_guard_report_qa.py <completed StressGuard snapshot>"
    directory, config, manifest = runner.read_suite(Path(sys.argv[1]).resolve(), root=ROOT)
    runner.verify_result(directory, manifest)
    status = json.loads((directory/"status.json").read_text(encoding="utf8"))
    assert status["status"] == "completed", status
    output = runner.safe_path(ROOT, directory/"report_validation.json")
    comparison_output = runner.safe_path(ROOT, directory/"comparison_ui_validation.json")
    assert not output.exists() and not comparison_output.exists(), "QA is create-only; previous validation is preserved."
    before = runner.previous.protected_state(ROOT)
    assert before == manifest["protected_files"]
    production = [ROOT/f"runs/exports/{LIVE}/{zone}/nuclear_kalman/forecast_{zone}_{LIVE}_nuclear_kalman.html" for zone in PRODUCTION_HTML_BEFORE]
    assert all(sha(path) == PRODUCTION_HTML_BEFORE[path.parent.parent.name] for path in production), "Production HTML changed from previous QA hashes."
    report_paths = [runner.safe_path(ROOT, directory/name) for name in status["report_files"]]
    assert all(sha(runner.safe_path(ROOT, directory/name)) == digest for name, digest in status["report_files"].items())
    watched = {*report_paths, *production, *(directory/name for name in runner.INPUTS | runner.OUTPUTS), directory/"status.json", directory/"manifest.json", directory/"results_manifest.json"}
    initial = {str(path): sha(path) for path in watched}
    source = json.loads((directory/"source_audit.json").read_text(encoding="utf8"))
    reports = json.loads((directory/"report_audit.json").read_text(encoding="utf8"))
    comparison = json.loads((directory/"comparison.json").read_text(encoding="utf8"))
    interval = json.loads((directory/"interval_comparison.json").read_text(encoding="utf8"))
    frames = {name: _prepare(frame) for name, frame in runner.collect(directory).items()}
    assert frames["p50_calibrated"].candidate_forecast.equals(frames["p50_previous"].candidate_forecast)
    assert frames["p50_calibrated"].applied_correction.equals(frames["p50_previous"].applied_correction)
    baseline = pd.read_parquet(directory/"panel.parquet")
    original_keys = ["zone", "timestamp_utc", "forecast_origin_utc", "sample", "forecast", "actual", "q10", "q90", "benchmark_forecast"]
    for name, frame in frames.items():
        expected = _prepare(baseline)[original_keys]
        pd.testing.assert_frame_equal(frame[original_keys], expected, check_exact=True)
    # Independently intersect the finite evaluation keys of ALL controls, as
    # opposed to allowing each interval table to silently choose its support.
    common_keys = None
    for frame in frames.values():
        finite = np.isfinite(frame[["actual", "forecast", "benchmark_forecast", "candidate_forecast"]].to_numpy(float)).all(axis=1)
        selected = frame.loc[frame.in_evaluation_window & finite, ["zone", "timestamp_utc"]]
        keys = set(selected.itertuples(index=False, name=None))
        common_keys = keys if common_keys is None else common_keys.intersection(keys)
    assert len(common_keys) == 4*8735
    baseline_intervals = frames["physics_governed"].copy(deep=True)
    baseline_intervals["candidate_forecast"] = baseline_intervals.forecast
    baseline_intervals["candidate_q10"] = baseline_intervals.q10
    baseline_intervals["candidate_q90"] = baseline_intervals.q90
    baseline_intervals["applied_correction"] = 0.
    assert set(interval) == set(frames) | {"nuclear_kalman"} and "storm" not in interval
    interval_frames = {**frames, "nuclear_kalman": baseline_intervals}
    for name, frame in interval_frames.items():
        selected = frame.loc[[key in common_keys for key in frame[["zone", "timestamp_utc"]].itertuples(index=False, name=None)]]
        for zone in ("all", "FR", "DE", "BE", "NL"):
            group = selected if zone == "all" else selected.loc[selected.zone.eq(zone)]
            for regime, values in interval_summary(group).items():
                for key, value in values.items():
                    close(interval[name][zone][regime][key], value)
                if name == "nuclear_kalman":
                    assert "calibration_status" not in interval[name][zone][regime]
    assert comparison["primary_variant"] == "physics_governed"
    assert comparison["annual_non_regression_guaranteed"] is False
    checks, bundles = {}, []
    for variant in VARIANTS:
        decision = check_decisions(frames[variant], direct=variant == "physics_direct")
        for zone in ("FR", "DE", "BE", "NL"):
            group = frames[variant].loc[frames[variant].zone.eq(zone)].reset_index(drop=True)
            evaluation, live = group.loc[group["sample"].eq("evaluation")], group.loc[group["sample"].eq("live")]
            common = evaluation.loc[evaluation.paired]
            assert set(common[["zone", "timestamp_utc"]].itertuples(index=False, name=None)) == {key for key in common_keys if key[0] == zone}
            proof = reports[variant+"_"+zone]
            assert len(evaluation) == 8760 and evaluation.local_day.nunique() == 365 and len(live) == 24 and len(common) == 8735
            assert proof["evaluation_start_day"] == START and proof["evaluation_end_day"] == END
            assert proof["live_excluded_from_statistics"] is True and proof["paired_hours"] == 8735
            assert proof["strict_governor_enforced"] == (variant == "physics_governed")
            assert all(name.startswith("feature_fundamental_") for name in proof["feature_allowlist"])
            storm, _ = _verified_storm(group, source, zone)
            assert storm["panel_missing_values_filled"] is False
            expected_intervals = interval_summary(common)
            for regime, values in expected_intervals.items():
                for key, value in values.items():
                    close(interval[variant][zone][regime][key], value)
                    close(proof["interval_summary"][regime][key], value)
            annual = comparison["by_zone"][zone]["annual"][variant]
            close(annual["mae_eur_mwh"], (common.candidate_forecast-common.actual).abs().mean())
            close(annual["mean_forecast_eur_mwh"], common.candidate_forecast.mean())
            close(annual["mean_observed_eur_mwh"], common.actual.mean())
            path = directory/variant/f"forecast_{zone.lower()}_{LIVE}_{variant}.html"
            document = path.read_text(encoding="utf8")
            for marker in ("STRESS GUARD", "STATISTICS — PRIX MOYENS", "storm-comparison", "hourly-comparison",
                           "CALIBRATION CHRONOLOGIQUE", "p &gt; 50 %", "90 jours", "P50 inchangé", "hors Statistics", "aucun import JAO", "Score intervalle80",
                           "Le CRPS du rapport natif est approché", "Événement détecté : sous-prévision de NYX"):
                assert marker in document, (zone, variant, marker)
            assert "p &gt; 0,6" not in document
            scripts = re.findall(r'<script(?:\s+type="text/javascript")?>(.*?)</script>', document, re.S)
            compiled = subprocess.run([str(NODE), "--check"], input="\n".join(scripts), encoding="utf8", text=True, capture_output=True, timeout=60)
            assert compiled.returncode == 0, compiled.stderr
            statistics = next(s for s in scripts if "const payload =" in s and "statistics-metric-select" in s)
            stats_check = check_statistics(statistics, evaluation)
            traces = [trace for plot in _plots(document) for trace in plot]
            gap = pd.DatetimeIndex(evaluation.loc[evaluation.benchmark_forecast.isna(), "timestamp_utc"])
            for field in ("candidate_forecast", "candidate_q10", "candidate_q90"):
                assert any(trace_matches(trace, pd.DatetimeIndex(live.timestamp_utc), live[field]) for trace in traces), (zone, variant, "live", field)
            # Native annual plots deliberately use the same paired support as
            # Storm. The independent daily Statistics above retain all 365
            # days, including the first day where no Storm value exists.
            assert any(trace_matches(trace, pd.DatetimeIndex(common.timestamp_utc), common.candidate_forecast, gap) for trace in traces), (zone, variant, "annual candidate paired axis")
            assert any(trace_matches(trace, pd.DatetimeIndex(common.timestamp_utc), common.benchmark_forecast, gap) for trace in traces), (zone, variant, "Storm official paired axis")
            banner = document.split('data-report-section="stress-guard"', 1)[1].split("</section>", 1)[0]
            tables = re.findall(r"<tbody>(.*?)</tbody>", banner, re.S)
            assert len(tables) == 2
            rows = re.findall(r"<tr>(.*?)</tr>", tables[1], re.S)
            assert len(rows) == len(live)
            for saved, row in zip(live.to_dict("records"), rows):
                cells = [html.unescape(v) for v in re.findall(r"<td>(.*?)</td>", row, re.S)]
                assert len(cells) == 9 and cells[0] == saved["local_label"]
                for i, column in ((1, "spike_probability"), (2, "forecast"), (3, "candidate_forecast"), (4, "applied_correction"), (5, "candidate_q10"), (6, "candidate_q90")):
                    if pd.isna(saved[column]): assert cells[i] == "—"
                    else: close(float(cells[i]), round(saved[column], 3))
                assert cells[7] == saved["gate_reason"] and cells[8] == saved["interval_calibration_status"]
            for raw_name in (c for c in group if c.startswith("feature_") and not c.startswith("feature_fundamental_")):
                assert raw_name not in document
            bundles.append({"zone": zone, "policy": variant, "statistics": statistics,
                "bootstrap": next(s for s in scripts if "document.documentElement.dataset.theme = theme" in s),
                "theme": next(s for s in scripts if "function applyPlotlyTheme" in s)})
            checks[variant+"_"+zone] = {"file": str(path), "statistics": stats_check,
                "paired_hours": len(common), "storm_archive_sha_verified": True,
                "saved_live_p10_p50_p90_and_common_annual_timestamp_values_verified": True,
                "live_decision_table_values_verified": True, "interval_summary_independently_recomputed": True,
                "annual_comparison_numbers_recomputed": True, "decisions": decision}
    javascript = run_node((ROOT/"tmp/fundamental_report_qa.js").read_text(encoding="utf8"), payload=bundles)
    comparison_javascript = comparison_qa(directory/"stress_guard_comparison.html")
    assert len(javascript) == 8 and comparison_javascript["default_variant"] == "physics_governed"
    runner.read_suite(directory, root=ROOT); runner.verify_result(directory, manifest)
    assert initial == {str(path): sha(path) for path in watched}
    assert before == runner.previous.protected_state(ROOT) == manifest["protected_files"]
    result = {"status": "passed", "reports": 8, "checks": checks, "javascript": javascript,
        "previous_p50_point_and_applied_correction_unchanged": True, "source_panel_all_variants_identical": True,
        "all_seven_interval_summaries_recomputed_on_global_common_support": True,
        "nyx_interval_baseline_included_without_spurious_calibration_status": True,
        "storm_not_assigned_unavailable_quantiles": True,
        "production_html_original_hashes": {str(path): initial[str(path)] for path in production},
        "all_watched_artifacts_unchanged_during_qa": True, "protected_code_matches_prerun_manifest": True,
        "new_fit_or_render_or_api_performed": False, "snapshot": str(directory),
        "visual_qa_limit": "Saved JavaScript ran in NodeVM with DOM/Plotly doubles. No browser pixel screenshot was taken."}
    for path, payload in ((comparison_output, comparison_javascript), (output, result)):
        with path.open("x", encoding="utf8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"status": "passed", "reports": 8, "javascript_contexts": len(javascript), "snapshot": str(directory)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
