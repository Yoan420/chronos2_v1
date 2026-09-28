"""Annual qualification must be derived from replay evidence, not assertions."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_annual_cpu_qualification as audit


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _price_fixture(root: Path, monkeypatch) -> tuple[Path, dict]:
    folder = root / "runs/price"
    folder.mkdir(parents=True)
    (root / "run_nyx_selected_cpu_historical.py").write_bytes(b"frozen runner")
    model = root / "chronos2_hourly/nyx_pooled_cpu_price_model.py"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"frozen model")
    grid = audit.price_replay.grid(audit.FIRST, audit.STOP)
    assert len(grid) == audit.NEGATIVE_HOURS
    preflight = {"references": {}, "official_comparisons": {}}
    for zone in audit.COUNTRIES:
        comparison = pd.DataFrame({"actual": np.full(len(grid), 50.),
                                   "storm": np.full(len(grid), 60.)}, index=grid)
        comparison.iloc[-1, 1] = np.nan
        relative = f"runs/inputs/comparison_{zone}.parquet"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_parquet(path)
        preflight["official_comparisons"][zone] = {"path": relative,
                                                    "sha256": audit.sha256(path)}
        if zone in ("FR", "BE"):
            relative = f"runs/inputs/reference_{zone}.parquet"
            path = root / relative
            pd.DataFrame({"reference": np.full(len(grid), 100.)},
                         index=grid).to_parquet(path)
            preflight["references"][zone] = {"path": relative,
                                              "sha256": audit.sha256(path)}
    monkeypatch.setattr(audit, "_verify_sources", lambda *_: {"source": "a" * 64})

    manifest = {"protocol": "nyx_selected_cwe_cpu_historical_v1",
                "preflight": preflight,
                "qualification_gate": audit.price_replay.QUALIFICATION_GATE,
                "thread_count": audit.PRICE_THREADS,
                "first_training_day": audit.price_replay.FIRST_TRAINING_DAY,
                "origins": [list(pair) for pair in audit.price_replay.annual_origins()],
                "configs": {role: asdict(config) for role, config in
                            audit.price_replay.ROLE_CONFIGS.items()},
                "role_countries": {role: list(zones) for role, zones in
                                   audit.price_replay.ROLE_COUNTRIES.items()},
                "runner_code_sha256": audit.sha256(root / "run_nyx_selected_cpu_historical.py"),
                "cpu_model_code_sha256": audit.sha256(model),
                "dependency_versions": audit._runtime_versions()}
    _write_json(folder / "run_manifest.json", manifest)
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def fake_checkpoint(path, *, role, origin, stop, manifest_sha256):
        assert path.is_file() and manifest_sha256 == digest
        index = audit.price_replay.grid(origin, stop)
        return ({zone: pd.Series(np.full(len(index), 50.), index=index)
                 for zone in audit.price_replay.ROLE_COUNTRIES[role]},
                {"tree_count": audit.price_replay.ROLE_CONFIGS[role].iterations})

    monkeypatch.setattr(audit.price_replay, "_read_checkpoint", fake_checkpoint)
    fit_audits = {}
    for role in audit.PRICE_EXPERTS:
        fit_audits[role] = []
        for origin, _ in audit.price_replay.annual_origins():
            path = folder / "checkpoints" / role / f"{origin}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{role}/{origin}".encode())
            fit_audits[role].append({"tree_count":
                audit.price_replay.ROLE_CONFIGS[role].iterations,
                "reused_checkpoint": False})
    scores, point_hashes = {}, {}
    for zone in audit.COUNTRIES:
        point = pd.Series(np.full(len(grid), 50.), index=grid)
        comparison = pd.read_parquet(root / preflight["official_comparisons"][zone]["path"])
        scores[zone] = audit.price_replay.score_selected(point, comparison)
        path = folder / f"{zone}.parquet"
        pd.DataFrame({"point": point}).to_parquet(path)
        point_hashes[zone] = audit.sha256(path)
    receipt = {"action": "full_annual_replay", "manifest_sha256": digest,
        "preflight": preflight,
        "qualification_gate": audit.price_replay.QUALIFICATION_GATE,
        "final_verdict_available": True, "total_cpu_fits": 159,
        "GPU_historical_scores_reproduced": False,
        "selected_recipes": {
            "FR": "fr_residual_1000 if abs(expert-reference)>=20 else reference",
            "BE": "mean(cwe_residual_2000,cwe_absolute_2000) if abs(mean-reference)>=20 else reference",
            "NL": "mean(cwe_residual_2000,cwe_absolute_2000) on every hour"},
        "fit_audits": fit_audits,
        "scores_on_official_epex_storm_rows": scores,
        "selected_point_sha256": point_hashes,
        "qualifies_all_three_countries": True}
    _write_json(folder / "receipt.json", receipt)
    return folder, receipt


def test_price_qualification_reconstructs_year_and_detects_altered_points(tmp_path,
                                                                            monkeypatch):
    folder, receipt = _price_fixture(tmp_path, monkeypatch)
    result = audit._price_replay(tmp_path, folder)
    assert all(item["hours"] == 8759 and item["strict_wins"] == 8759
               for item in result["metrics"].values())
    path = folder / "BE.parquet"
    frame = pd.read_parquet(path)
    frame.iloc[0, 0] = 60.
    frame.to_parquet(path)
    receipt["selected_point_sha256"]["BE"] = audit.sha256(path)
    _write_json(folder / "receipt.json", receipt)
    with pytest.raises(ValueError, match="final price points differ"):
        audit._price_replay(tmp_path, folder)


def test_price_qualification_requires_exact_storm_coverage(tmp_path, monkeypatch):
    folder, receipt = _price_fixture(tmp_path, monkeypatch)
    path = tmp_path / receipt["preflight"]["official_comparisons"]["FR"]["path"]
    frame = pd.read_parquet(path)
    frame.iloc[-2, 1] = np.nan
    frame.to_parquet(path)
    with pytest.raises(ValueError, match="Official Storm annual pairing changed"):
        audit._price_replay(tmp_path, folder)


def _negative_fixture(root: Path, monkeypatch) -> Path:
    folder = root / "runs/negative"
    folder.mkdir(parents=True)
    expected = {"catboost": "test", "numpy": "test", "pandas": "test",
                "scikit-learn": "test"}
    monkeypatch.setattr(audit.negative_replay, "preflight",
                        lambda *_: {"ready": True})
    monkeypatch.setattr(audit.negative_replay, "_runtime_versions",
                        lambda: expected)
    config = {"expected_runtime": expected, "origin_step_civil_days": 7,
              "origins_per_country": 53, "sources": {},
              "evidence": {"metrics": {"path": "runs/inputs/metrics.json"}}}
    report = {"countries": {}}
    comparisons = {}
    for zone in audit.COUNTRIES:
        grid = audit.negative_model.physical_grid(audit.FIRST, audit.STOP, zone)
        actual = pd.Series(np.full(len(grid), 50.), index=grid)
        actual.iloc[:10] = -5.
        probability = pd.Series(np.full(len(grid), .01), index=grid)
        probability.iloc[:10] = .8
        output = pd.DataFrame({"p_negative_raw": probability,
                               "p_negative": probability,
                               "is_negative_predicted": probability >= .5}, index=grid)
        output.to_parquet(folder / f"{zone}_probabilities.parquet")
        baseline = root / f"runs/inputs/baseline_{zone}.parquet"
        baseline.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"actual": actual}, index=grid).to_parquet(baseline)
        archived = root / f"runs/inputs/archived_{zone}.parquet"
        output.rename(columns={"p_negative_raw": "p_raw"}).to_parquet(archived)
        config["sources"][zone] = {"baseline": {"path": baseline.relative_to(root).as_posix()},
                                   "archived_probabilities": {"path": archived.relative_to(root).as_posix()}}
        score = audit.negative_model.probability_metrics(actual, probability)
        tp, fp, fn = (score[key] for key in
                      ("true_positive", "false_positive", "false_negative"))
        score["precision"] = tp / (tp + fp) if tp + fp else None
        score["recall"] = tp / (tp + fn) if tp + fn else None
        report["countries"][zone] = {"models": {"model": {
            "n": score["hours"], **{key: score[key] for key in
            ("negative_hours", "brier", "log_loss", "average_precision",
             "precision", "recall")}}}}
    _write_json(root / "config/nyx_negative_annual_replay.json", config)
    _write_json(root / "runs/inputs/metrics.json", report)
    for zone in audit.COUNTRIES:
        spec = config["sources"][zone]
        baseline = pd.read_parquet(root / spec["baseline"]["path"])
        archived = pd.read_parquet(root / spec["archived_probabilities"]["path"])
        output = pd.read_parquet(folder / f"{zone}_probabilities.parquet")
        comparisons[zone] = {**audit.negative_replay._score_comparison(
            baseline["actual"], output, archived, report["countries"][zone]),
            "fits": 53}
    receipt = {"state": "COMPLETE",
        "identity": "nyx_negative_annual_cpu_replay_20260923_v1",
        "countries": list(audit.COUNTRIES), "origins_per_country": 53,
        "completed_fits": 159, "total_fits": 159,
        "same_historical_period": True, "independent_validation": False,
        "source_hashes_unchanged": True, "all_metrics_and_series_match": True,
        "comparisons": comparisons}
    _write_json(folder / "receipt.json", receipt)
    return folder


def test_negative_qualification_recomputes_brier_and_detects_changed_series(tmp_path,
                                                                               monkeypatch):
    folder = _negative_fixture(tmp_path, monkeypatch)
    result = audit._negative_replay(tmp_path, folder)
    assert set(result["metrics"]) == set(audit.COUNTRIES)
    assert all(item["hours"] == 8760 for item in result["metrics"].values())
    path = folder / "NL_probabilities.parquet"
    frame = pd.read_parquet(path)
    frame.iloc[0, frame.columns.get_loc("p_negative")] = .1
    frame.to_parquet(path)
    with pytest.raises(ValueError, match="negative replay series or scores differ"):
        audit._negative_replay(tmp_path, folder)


def test_seal_is_exclusive_and_failure_creates_no_receipt(tmp_path, monkeypatch):
    price, negative = (tmp_path / "runs" / name for name in ("price", "negative"))
    for folder in (price, negative):
        folder.mkdir(parents=True)
        (folder / "receipt.json").write_bytes(b"{}")
        for zone in audit.COUNTRIES:
            suffix = f"{zone}.parquet" if folder == price else f"{zone}_probabilities.parquet"
            (folder / suffix).write_bytes(zone.encode())
    (tmp_path / "config").mkdir()
    code = tmp_path / "code.py"
    code.write_bytes(b"pinned")
    destination = tmp_path / "config" / audit.QUALIFICATION_RECEIPT.name
    monkeypatch.setattr(audit, "prepare_receipt", lambda *_:
                        (_ for _ in ()).throw(ValueError("not qualified")))
    with pytest.raises(ValueError, match="not qualified"):
        audit.seal_receipt(tmp_path, price, negative)
    assert not destination.exists()
    payload = {"qualified": False, "code_sha256": {"code.py": audit.sha256(code)},
        "evidence_sha256": {
            "price_selected_points": {zone: audit.sha256(price / f"{zone}.parquet")
                                      for zone in audit.COUNTRIES},
            "negative_probabilities": {zone: audit.sha256(negative / f"{zone}_probabilities.parquet")
                                       for zone in audit.COUNTRIES}}}
    monkeypatch.setattr(audit, "prepare_receipt", lambda *_: payload)
    result = audit.seal_receipt(tmp_path, price, negative)
    assert result["price_expert_replay_qualified"] is True
    assert result["full_input_chain_qualified"] is False
    assert result["qualified"] is False
    assert result["sha256"] == audit.sha256(destination)
    with pytest.raises(ValueError, match="already exists"):
        audit.seal_receipt(tmp_path, price, negative)
