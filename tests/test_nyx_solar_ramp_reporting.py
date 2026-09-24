from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
import pytest

from nyx_solar_ramp import reporting as report


def _decode_payload(value):
    decoded = deepcopy(value)
    for zone in decoded["zones"].values():
        for variant in zone["variants"].values():
            for field, reference in list(variant.items()):
                values = zone["values"][reference["baseline"]] if "baseline" in reference else zone["arrays"][reference["array"]]
                variant[field] = ([decoded["dictionaries"][field][code] if code >= 0 else None for code in values]
                                  if field in decoded["dictionaries"] else values)
        del zone["arrays"]
    return decoded


def _frames(zones=("FR", "DE", "BE", "NL"), *, annual=False):
    start, end = ("2025-09-18", "2026-09-19") if annual else ("2026-09-14", "2026-09-16")
    stamps = pd.date_range(start, end, tz="Europe/Paris", freq="h", inclusive="left").tz_convert("UTC")
    local = stamps.tz_convert("Europe/Paris")
    last = local[-1].date()
    frames = []
    for zone in zones:
        frame = pd.DataFrame({"zone": zone, "timestamp_utc": stamps,
            "forecast_origin_utc": [(pd.Timestamp(day)-pd.Timedelta(days=1)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris") for day in local.date],
            "sample": np.where(local.date == last, "live", "evaluation"), "actual": 110.,
            "forecast": 100., "q10": 80., "q90": 130., "benchmark_forecast": 104.,
            "feature_fr_solar_generation_gw": 10.})
        for column in report.SERIES:
            frame[column] = 4.
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    variants = []
    for variant in ("baseline", "governed", "ramps"):
        frame = panel.copy(deep=True)
        correction = 0. if variant == "baseline" else 1.
        frame["variant"] = variant
        frame["candidate_forecast"] = frame.forecast + correction
        frame["candidate_q10"] = frame.q10
        frame["candidate_q90"] = frame.q90
        frame["risk_probability"] = .7
        frame["alert_threshold"] = .6
        frame["alert"] = True
        frame["expert_ready"] = True
        frame["applied_correction"] = correction
        frame["bounded_correction"] = 4.
        frame["selected_weight"] = correction/4.
        frame["gate_reason"] = "frozen_oos_governor"
        frame["evaluation_phase"] = np.where(frame["sample"].eq("live"), "live_historical", "final_diagnostic")
        frame["fold_id"] = "frozen-fold"
        variants.append(frame)
    return panel, pd.concat(variants, ignore_index=True)


def _run(tmp_path, monkeypatch):
    root = tmp_path / "project"
    directory = root / "runs/experiments/nyx_solar_ramp_v1/frozen"
    directory.mkdir(parents=True)
    panel, predictions = _frames()
    panel.to_parquet(directory / "panel.parquet", index=False)
    predictions.to_parquet(directory / "predictions.parquet", index=False)
    values = {"metrics.json": {"decision": "retain_baseline", "scores": [{"variant": "governed", "zone": "FR", "phase": "final_diagnostic", "n": 24, "mae": 9., "pinball_q10": 3.}],
                              "paired_bootstrap": [{"variant": "governed", "lower": -1., "upper": 2.}],
                              "risk_bins": [{"bin": "0.6–0.7", "n": 24}]},
              "manifest.json": {"prospective": {"status": "blocked", "reasons": ["Prix déjà publiés"]}},
              "source_audit.json": {"source_data_audit": {}, "limitations": ["query_asof_cutoff_only"]},
              "literature.json": {"references": [{"title": "A <paper>", "url": "https://example.org/paper"}]}}
    for name, value in values.items():
        (directory / name).write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(report, "_standard_reports", lambda *args: ({}, {"FR": {"status": "unavailable", "reason": "test: short window"}}))
    return root, directory


def test_report_offline_four_countries_no_refit_and_inputs_unchanged(tmp_path, monkeypatch):
    root, directory = _run(tmp_path, monkeypatch)
    before = {name: (directory/name).read_bytes() for name in report.INPUTS}
    path = report.render_report(directory, root=root)
    assert path == directory / "reports/index.html"
    document = path.read_text(encoding="utf-8")
    for marker in ("NYX Solar Ramp", "14 septembre 2026", "étude de cas connue", "Prospective : BLOQUÉE",
                   "publication du fournisseur", "pas les horodatages", "live_historical", "final_diagnostic", "retain_baseline",
                   "Storm est un comparateur figé ex post", "plotly_relayout", "candidate_q10", "Bootstrap apparié"):
        assert marker in document
    for zone in report.ZONES:
        assert f'id="chart-{zone}"' in document and f'id="hours-{zone}"' in document
    assert not re.search(r'<script[^>]+src\s*=', document, re.I)
    assert "A &lt;paper&gt;" in document
    payload = _decode_payload(json.loads(re.search(r'<script id="solar-ramp-data" type="application/json">(.*?)</script>', document, re.S).group(1)))
    assert set(payload["zones"]) == set(report.ZONES)
    assert payload["zones"]["FR"]["variants"]["governed"]["candidate_q10"][0] == 80.
    assert "NaN" not in json.dumps(payload, allow_nan=False)
    audit = json.loads((directory/"reports/report_audit.json").read_text(encoding="utf-8"))
    assert audit["model_fitted"] is False and audit["external_sources_fetched"] is False
    assert audit["production_modified"] is False and audit["source_live_day_is_historical"] is True
    assert audit["renderer_code_sha256"] == hashlib.sha256(Path(report.__file__).read_bytes()).hexdigest()
    assert audit["payload_schema_version"] == 2
    assert before == {name: (directory/name).read_bytes() for name in report.INPUTS}
    assert not (root/"runs/exports").exists()


def test_script_payload_escapes_markup_and_keeps_missing_values():
    raw = {"label": '</script><script>alert("x")</script>', "missing": np.nan, "other": pd.NA}
    encoded = report._script_json(raw)
    assert "</script>" not in encoded and "\\u003c" in encoded
    assert json.loads(encoded)["label"] == raw["label"]
    assert json.loads(encoded)["missing"] is None and json.loads(encoded)["other"] is None


def test_compressed_payload_lossless_full_history_and_no_unused_variant_fields():
    panel, predictions = _frames(("FR",), annual=True)
    predictions.loc[predictions.index[0], "gate_reason"] = pd.NA
    predictions.loc[predictions.index[1], "risk_probability"] = np.nan
    predictions["fold_id"] = "NOT_READ_BY_JAVASCRIPT"
    compact = report._payload(panel, predictions)
    decoded = _decode_payload(compact)
    assert len(decoded["zones"]["FR"]["timestamp"]) == 8784
    assert len(set(decoded["zones"]["FR"]["day"])) == 366
    assert "NOT_READ_BY_JAVASCRIPT" not in report._script_json(compact)
    plain = {}
    for variant, group in predictions.groupby("variant"):
        ordered = group.sort_values("timestamp_utc")
        expected = {field: report._clean(ordered[field].tolist()) for field in report.PREDICTIONS}
        assert decoded["zones"]["FR"]["variants"][variant] == expected
        plain[variant] = expected
        assert not {"fold_id", "alert", "expert_ready", "bounded_correction"}.intersection(compact["zones"]["FR"]["variants"][variant])
    encoded_variant_bytes = len(report._script_json({"arrays": compact["zones"]["FR"]["arrays"],
        "variants": compact["zones"]["FR"]["variants"], "dictionaries": compact["dictionaries"]}))
    assert encoded_variant_bytes < len(report._script_json(plain)) * .5


def test_payload_preserves_full_float_precision_and_signed_zero():
    panel, predictions = _frames(("FR",))
    panel["q10"] = 0.
    predictions["q10"] = 0.
    predictions["candidate_q10"] = -0.
    predictions["candidate_q90"] = 130.12345678901234
    encoded = report._script_json(report._payload(panel, predictions))
    decoded = _decode_payload(json.loads(encoded))["zones"]["FR"]
    assert np.signbit(decoded["variants"]["governed"]["candidate_q10"][0])
    assert not np.signbit(decoded["values"]["q10"][0])
    assert decoded["variants"]["governed"]["candidate_q90"][0] == 130.12345678901234


def test_bibliography_expands_authors_year_and_application_safely():
    rendered = report._literature({"references": [{"title": "Research", "url": "https://example.org/paper",
        "authors": ["Alice <A>", "Bob B"], "year": 2021, "application": "Compare <ramps> without claiming causality."}]})
    assert "Alice &lt;A&gt;; Bob B (2021)" in rendered
    assert "Application à ce laboratoire" in rendered and "Compare &lt;ramps&gt;" in rendered


def test_different_main_and_storm_comparison_supports_are_visible():
    rendered = report._standard_support({"scores": [{"variant": "governed", "zone": "FR", "phase": "all", "n": 8760}]},
        {"FR": {"status": "complete", "evaluation_hours": 8760, "paired_hours": 8735}})
    assert "8760" in rendered and "8735" in rendered and ">25<" in rendered
    assert "Aucun trou Storm n’est rempli" in rendered and "peuvent donc différer" in rendered


def test_summary_uses_saved_candidate_scores_and_explicit_common_coverage():
    scores = [{"variant": "governed", "zone": "ALL", "phase": "all", "n": 35040,
               "mae": 9.123, "baseline_mae": 999.876, "rmse": 12.12, "spike_mae": 45.6, "pr_auc": .3},
              {"variant": "baseline", "zone": "ALL", "phase": "all", "n": 35040, "mae": 10.123},
              {"variant": "ramps", "zone": "ALL", "phase": "common_oos", "n": 12000,
               "mae": 8.765, "n_probability": 12000},
              {"variant": "local", "zone": "DE", "phase": "all", "n": 8760, "mae": 777.777}]
    summary = report._score_summary({"scores": scores})
    assert "9.123" in summary and "8.765" in summary
    assert "999.876" not in summary and "777.777" not in summary
    assert summary.index("baseline") < summary.index("governed")
    assert "warm-up et repli inclus" in summary and "Support OOS commun" in summary
    assert "candidate_forecast" in summary
    assert report._score_summary({}) == ""


def test_optional_physical_study_preserves_case_counterexamples_and_caveats():
    physical = {"protocol": {"limitation": "source publication not certified"}, "coverage": {"post_warmup_rows": 123},
                "warmup_thresholds": [{"zone": "DE", "n_positive_drop": 120, "solar_drop_q90_gwph": 4.2}],
                "event_rows": [{"zone": "DE", "delivery_start_local": "2026-09-14T18:00:00+02:00",
                                 "actual": 355.1, "baseline": 220.2, "candidate": 230.3}],
                "counterexamples": {"large_drop_without_spike": [{"zone": "BE", "actual": 50.4}],
                                    "spike_without_drop": [{"zone": "DE", "actual": 401.2}]},
                "matched_strata": {"summary": [{"zone": "BE", "difference": -.042}], "rows": [{"hour": 18, "n": 90}]},
                "regime_rates": [{"zone": "DE", "season": "summer", "n": 42, "spike_rate": .125}]}
    result = report._physical_analysis(physical)
    assert "355.1" in result and "50.4" in result and "401.2" in result and "-0.042" in result
    assert "événement déjà connu" in result and "pas une démonstration causale" in result
    assert "source publication not certified" in result
    assert report._physical_analysis(None) == ""


def test_training_window_and_label_assumptions_are_explicit():
    result = report._methodology({"config": {"minimum_training_days": 120, "training_window_days": 365,
                                              "label_delay_days": 2}}, {})
    assert "minimum 120 jours éligibles" in result
    assert "fenêtre maximale 365 jours" in result
    assert "pas d’un apprentissage sur 365 jours complets" in result
    assert "fin de livraison + 2 jours" in result
    assert "pas une publication fournisseur vérifiée" in result


@pytest.mark.parametrize("change", ["baseline", "quantile", "price", "weight", "duplicate", "missing", "probability"])
def test_invalid_saved_values_rejected_not_repaired(change):
    panel, predictions = _frames(("FR",))
    if change == "baseline": predictions.loc[0, "forecast"] += 1
    elif change == "quantile": predictions.loc[0, "candidate_q10"] = np.nan
    elif change == "price": predictions.loc[0, "candidate_forecast"] += 1
    elif change == "weight": predictions.loc[0, "selected_weight"] = .9
    elif change == "duplicate": predictions = pd.concat([predictions, predictions.iloc[:1]])
    elif change == "missing": predictions = predictions.iloc[1:]
    else: predictions.loc[0, "risk_probability"] = 1.1
    with pytest.raises(ValueError):
        report._validated_frames(panel, predictions)


def test_no_mutation_and_preservation_of_unknown_risk():
    panel, predictions = _frames(("FR",))
    predictions.loc[0, "risk_probability"] = np.nan
    original_panel, original_predictions = panel.copy(deep=True), predictions.copy(deep=True)
    _, result = report._validated_frames(panel, predictions)
    assert result.risk_probability.isna().sum() == 1
    pd.testing.assert_frame_equal(panel, original_panel)
    pd.testing.assert_frame_equal(predictions, original_predictions)


def test_production_and_namespace_root_paths_rejected(tmp_path):
    for folder in (tmp_path, tmp_path/"runs/exports/x", tmp_path/"runs/experiments/nyx_solar_ramp_v1"):
        with pytest.raises(ValueError, match="run below"):
            report.render_report(folder, root=tmp_path)


def test_standard_contract_failure_is_explicit(tmp_path):
    _, predictions = _frames(("FR",))
    links, audits = report._standard_reports(predictions, {}, tmp_path/"standard", {}, {})
    assert not links and audits["FR"]["status"] == "unavailable"
    assert "365" in audits["FR"]["reason"]
    assert not (tmp_path/"standard").exists()


def _storm_archive(tmp_path, frame):
    from chronos2_hourly.storm_dashboard import STORM_DASHBOARD_COLUMN
    stamps = pd.DatetimeIndex(frame.timestamp_utc)
    values = frame.benchmark_forecast.to_numpy(float)
    hole = stamps == pd.Timestamp("2025-10-26T00:00:00Z")
    values[hole] = np.nan
    frame.loc[hole, "benchmark_forecast"] = np.nan
    archive = tmp_path / "original_archive"
    (archive/"inputs").mkdir(parents=True)
    artifact = archive/"inputs/storm_dashboard_official_statistics.parquet"
    pd.DataFrame({"delivery_start_utc": stamps, STORM_DASHBOARD_COLUMN: values}).to_parquet(artifact, index=False)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    series = "power.price.fr.euromwh.h.fcst.3mv.storm.da.cache"
    holes = [stamp.isoformat() for stamp in stamps[hole]]
    materialization = {"role": "evaluation_only_dashboard_comparator", "series": series, "column": STORM_DASHBOARD_COLUMN,
        "used_for_prediction": False, "used_for_live_forecast": False,
        "normalized_artifact_path": "inputs/storm_dashboard_official_statistics.parquet", "normalized_artifact_sha256": sha(artifact),
        "source": {"kind": "saturn_storm_day_ahead_cache_with_native_gap_fallback", "zone": "FR", "series": series,
            "primary_series": series, "series_kind": "frozen_day_ahead_cache", "used_for_prediction": False,
            "fallback_policy": "native_only_where_day_ahead_cache_is_missing", "cache_precedence": True,
            "fallback_series": "power.price.fr.euromwh.h.fcst.3mv.storm", "extracted_at_utc": "2026-09-18T12:00:00Z"},
        "period": {"start_utc": stamps[0].isoformat(), "end_utc": stamps[-1].isoformat(), "timezone": "Europe/Paris"},
        "expected_hours": len(stamps), "available_hours": len(stamps)-1, "missing_hours": 1,
        "dst": {"interpolation": False, "strict_08_fallback": False, "native_allowed_missing_utc": holes,
                "native_allowed_missing_hours": 1, "native_actual_missing_matches_allowed": True}}
    history = archive/"statistics_history_audit.json"
    history.write_text(json.dumps({"status": "complete", "storm_primary_report_benchmark": STORM_DASHBOARD_COLUMN,
                                  "storm_dashboard": materialization}), encoding="utf-8")
    nuclear = tmp_path/"nuclear_report_audit.json"
    nuclear.write_text(json.dumps({"zone": "FR", "storm_hourly_comparison": {"snapshot": {"archive": str(archive),
        "artifact_path": str(artifact), "artifact_sha256": sha(artifact), "audit_path": str(history), "audit_sha256": sha(history)}}}), encoding="utf-8")
    return {"source_data_audit": {"baseline": {"evaluation_start_day": "2025-09-18", "evaluation_end_day": "2026-09-17",
            "sources": [{"zone": "FR", "model": "nuclear_kalman", "time_axis_audits": [{"path": str(nuclear), "sha256": sha(nuclear)}]}]}}}


def test_exact_standard_renderer_365_days_saved_quantiles_historical_not_prospective(tmp_path):
    panel, predictions = _frames(("FR",), annual=True)
    audit = _storm_archive(tmp_path, panel)
    missing = panel.loc[panel.benchmark_forecast.isna(), "timestamp_utc"]
    predictions.loc[predictions.timestamp_utc.isin(missing), "benchmark_forecast"] = np.nan
    prior = predictions.copy(deep=True)
    links, proofs = report._standard_reports(predictions, audit, tmp_path/"standard", {}, {"decision": "retain_baseline"})
    assert set(links) == {"FR"}, proofs
    document = (tmp_path/links["FR"]).read_text(encoding="utf-8")
    for marker in ("Backtest et probabilités", "Statistics", "variable-attribution", "solar-ramp-methodology",
                   "2025-09-18", "2026-09-17", "Courbe historique sauvegardée du", "BLOQUÉE", "retain_baseline"):
        assert marker in document
    assert "Prévision opérationnelle du" not in document
    assert proofs["FR"]["evaluation_days"] == 365
    assert proofs["FR"]["evaluation_hours"] == 8760 and proofs["FR"]["paired_hours"] == 8759
    assert proofs["FR"]["live_excluded_from_statistics"] is True
    assert proofs["FR"]["new_prospective_forecast"] is False
    pd.testing.assert_frame_equal(predictions, prior)


def test_interaction_javascript_parses_if_bundled_node_available():
    node = Path(r"C:\Users\BQ6757\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")
    if not node.is_file():
        pytest.skip("Bundled Node unavailable")
    checked = subprocess.run([str(node), "--check"], input=report._JS, text=True, encoding="utf-8", capture_output=True)
    assert checked.returncode == 0, checked.stderr


def test_real_javascript_payload_decoder_matches_saved_fields_if_node_available():
    node = Path(r"C:\Users\BQ6757\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")
    if not node.is_file():
        pytest.skip("Bundled Node unavailable")
    panel, predictions = _frames(("FR",))
    encoded = report._payload(panel, predictions)
    script = report._PAYLOAD_DECODER_JS + "\nprocess.stdout.write(JSON.stringify(decodePayload(" + report._script_json(encoded) + ")));"
    checked = subprocess.run([str(node)], input=script, text=True, encoding="utf-8", capture_output=True)
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout) == _decode_payload(encoded)
