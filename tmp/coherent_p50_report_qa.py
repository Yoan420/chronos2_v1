"""Read-only saved-report QA; writes only new generated report_validation.json."""
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
from marginal_cost_expert.evaluation import _array
from nyx_scarcity.reporting import _prepare
from nyx_scarcity_zonal.reporting import _verified_storm
from nyx_coherent_p50.runner import read_suite, verify_result, protected_state
from nyx_coherent_p50.reporting import _validate_decisions

NODE = Path(r"C:\Users\BQ6757\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")
PRODUCTION_HTML_BEFORE = {
    "be": "8177031a943ff1522beef089bedf6b3cbfdbcec9d3230a97f142e12eefb3ab01",
    "de": "ad84143e4f05fd8d5d6d0e4a747a80ea9fbe85ae4a3db5dd87565d39422039ed",
    "fr": "f6e6107ff4e4d8befab3db75137e631313d8e68d077eb9544f87261bae1cd131",
    "nl": "9b55b26bfc99f483946e3bae67b6b58dc09bf0119e8113a701477fa8c2907d28",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    np.testing.assert_allclose(a, b, atol=1e-8, rtol=1e-10, equal_nan=True)


def main():
    directory = Path(sys.argv[1]).resolve()
    status = json.loads((directory/"status.json").read_text(encoding="utf8"))
    assert status["status"] == "completed", status
    directory, config, manifest = read_suite(directory, root=ROOT)
    for variant in config["variants"]:
        verify_result(directory, variant, manifest)
    before = protected_state(ROOT)
    assert before == manifest["protected_files"], "Protected production files differ from pre-run sealed hashes."
    prod = [ROOT/f"runs/exports/2026-09-15/{z}/nuclear_kalman/forecast_{z}_2026-09-15_nuclear_kalman.html" for z in ("fr", "de", "be", "nl")]
    assert all(sha(path) == PRODUCTION_HTML_BEFORE[path.parent.parent.name] for path in prod), "Original production HTML differs from pre-run hashes."
    watched = [*directory.glob("reports/*.html"), *directory.glob("reports_governed/*.html"),
        directory/"source_audit.json", directory/"forest/predictions.parquet", directory/"forest/governed_predictions.parquet", *prod]
    initial = {str(path): sha(path) for path in watched}
    source = json.loads((directory/"source_audit.json").read_text(encoding="utf8"))
    combined, bundles = {}, []
    for folder, filename, policy in (("reports", "predictions.parquet", "direct"), ("reports_governed", "governed_predictions.parquet", "governed")):
        frame = _prepare(pd.read_parquet(directory/"forest"/filename))
        audit = json.loads((directory/folder/"p50_report_audit.json").read_text(encoding="utf8"))
        checks = {}
        for zone in ("FR", "DE", "BE", "NL"):
            group = frame.loc[frame.zone.eq(zone)].reset_index(drop=True)
            evaluation, live = group.loc[group["sample"].eq("evaluation")], group.loc[group["sample"].eq("live")]
            proof = audit["reports"][zone]
            assert len(evaluation) == 8760 and evaluation.local_day.nunique() == 365 and len(live) == 24
            assert proof["paired_hours"] == 8735 and proof["live_excluded_from_statistics"] is True
            assert proof["evaluation_start_day"] == "2025-09-15" and proof["evaluation_end_day"] == "2026-09-14"
            assert proof["strict_governor_enforced"] == (policy == "governed")
            assert all(c.startswith("feature_fundamental_") for c in proof["allowed_input_features"])
            assert proof["decision_checks"]["quantile_function_interpolation_checked"]
            assert not proof["decision_checks"]["classifier_probability_is_final_decision_distribution_probability"]
            decision = _validate_decisions(group, policy)
            storm, _ = _verified_storm(group, source, zone)
            assert storm["panel_missing_values_filled"] is False
            close(group.candidate_forecast, group.forecast+group.applied_correction)
            close(group.applied_correction, group.selected_weight*group.bounded_correction)
            path = next((directory/folder).glob(f"forecast_{zone.lower()}_*.html"))
            document = path.read_text(encoding="utf8")
            for marker in ("P50 COHÉRENT", "STATISTICS — PRIX MOYENS", "storm-comparison", "hourly-comparison",
                "Motif de proposition", "Erreur Q50 brute", "P50 brut NYX + erreur", "14 septembre à 19 h",
                "PAS un mélange des CDF", "ni p × sévérité", "masse artificielle à zéro", "si p &gt; 50 %",
                "90 jours" if policy == "governed" else "poids de 100 %"):
                assert marker in document, (zone, marker)
            assert "p &gt; 0,6" not in document
            for feature in (c for c in group if c.startswith("feature_") and not c.startswith("feature_fundamental_")):
                assert feature not in document, (zone, feature)
            scripts = re.findall(r'<script(?:\s+type="text/javascript")?>(.*?)</script>', document, re.S)
            compiled = subprocess.run([str(NODE), "--check"], input="\n".join(scripts), encoding="utf8", text=True, capture_output=True)
            assert compiled.returncode == 0, compiled.stderr
            statistics = next(s for s in scripts if "const payload =" in s and "statistics-metric-select" in s)
            payload = json.JSONDecoder().raw_decode(statistics.split("const payload =", 1)[1].lstrip())[0]
            daily = [r for r in payload["records"] if r["sample"] == "daily"]
            assert len(daily) == 365 and max(r["period_key"] for r in daily) == "2026-09-14"
            assert min(r["period_key"] for r in daily) == "2025-09-15"
            assert next(r for r in daily if r["period_key"] == "2025-09-15")["benchmark_mean_price"] is None
            assert next(r for r in daily if r["period_key"] == "2025-10-26")["benchmark_n"] == 24
            means = {}
            for day in ("2026-09-14", "2026-06-24", "2026-06-25", "2026-06-26"):
                row = next(r for r in daily if r["period_key"] == day)
                block = evaluation.loc[evaluation.local_day.eq(day)]
                common = np.isfinite(block[["actual", "forecast", "candidate_forecast", "benchmark_forecast"]]).all(axis=1)
                block = block.loc[common]
                for key, column in (("mean_price", "candidate_forecast"), ("observed_mean_price", "actual"), ("benchmark_mean_price", "benchmark_forecast")):
                    close(row[key], block[column].mean())
                means[day] = {key: row[key] for key in ("mean_price", "observed_mean_price", "benchmark_mean_price")}
            traces = [trace for plot in _plots(document) for trace in plot]
            live_traces = [t for t in traces if len(t.get("x", [])) == 24 and str(t.get("x", [""])[0]).startswith("2026-09-15")]
            for column in ("candidate_forecast", "candidate_q10", "candidate_q90"):
                assert any(np.allclose(_array(t.get("y", [])), live[column], atol=1e-8, rtol=1e-10, equal_nan=True)
                    for t in live_traces if len(_array(t.get("y", []))) == 24), (zone, column)
            banner = document.split('data-report-section="coherent-p50-experiment"', 1)[1].split("</section>", 1)[0]
            tables = re.findall(r"<tbody>(.*?)</tbody>", banner, re.S)
            assert len(tables) == 2
            case = evaluation.loc[evaluation.local_day.eq("2026-09-14") & evaluation.local_hour.eq(19)]
            assert len(case) == 1
            for table, saved_frame in zip(tables, (live, case)):
                rows = re.findall(r"<tr>(.*?)</tr>", table, re.S)
                assert len(rows) == len(saved_frame)
                for saved, row in zip(saved_frame.to_dict("records"), rows):
                    cells = [html.unescape(value) for value in re.findall(r"<td>(.*?)</td>", row, re.S)]
                    assert len(cells) == 18
                    for cell, column in ((1, "forecast"), (2, "spike_probability"), (3, "threshold_eur_mwh"),
                        (4, "mixture_error_q10"), (5, "mixture_error_q50"), (6, "mixture_error_q90"),
                        (7, "mixture_raw_p50_eur_mwh"), (8, "risk_probability_gate"), (11, "bounded_correction"),
                        (12, "selected_weight"), (13, "applied_correction"), (14, "candidate_forecast")):
                        value = saved[column]
                        if pd.isna(value):
                            assert cells[cell] == "—"
                        else:
                            close(float(cells[cell]), round(value, 3))
                    assert cells[9] == ("Oui" if saved["physical_gate_passed"] else "Non")
                    assert cells[10] == saved["proposal_reason"]
            bundles.append({"zone": zone, "policy": policy, "statistics": statistics,
                "bootstrap": next(s for s in scripts if "document.documentElement.dataset.theme = theme" in s),
                "theme": next(s for s in scripts if "function applyPlotlyTheme" in s)})
            checks[zone] = {"statistics_days": 365, "paired_hours": 8735, "statistics_end": "2026-09-14",
                "live_day": "2026-09-15", "live_p10_p50_p90_plot_traces_verified": True, "price_means": means,
                "decision_and_cdf_checks": decision, "storm_original_archive_verified": True,
                "frozen_missing_values_preserved": True, "live_and_sep14_19h_mechanism_tables_verified": True,
                "file": str(path)}
        index = (directory/folder/"index.html").read_text(encoding="utf8")
        assert "../coherent_p50_comparison.html" in index and "Corrections aggravantes" in index
        combined[folder] = {"status": "passed", "variant": "forest", "decision_policy": policy, "checks": checks,
            "index_comparison_link_and_harm_table": True, "source_result_verified": True, "no_render_or_fit_performed": True}
    javascript = subprocess.run([str(NODE), str(ROOT/"tmp/fundamental_report_qa.js")], input=json.dumps(bundles), encoding="utf8", text=True, capture_output=True)
    assert javascript.returncode == 0, javascript.stderr
    results = json.loads(javascript.stdout)
    assert initial == {str(path): sha(path) for path in watched}
    assert before == protected_state(ROOT) == manifest["protected_files"]
    for folder, result in combined.items():
        result.update(javascript=[r for r in results if r["policy"] == result["decision_policy"]],
            html_predictions_code_and_protected_files_unchanged_during_qa=True,
            protected_code_matches_before_run_manifest=True,
            production_html_sha256_unchanged_during_qa={str(p): initial[str(p)] for p in prod})
        (directory/folder/"report_validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({"status": "passed", "reports": 8, "javascript_contexts": len(results), "snapshot": str(directory)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
