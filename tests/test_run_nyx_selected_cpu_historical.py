from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import run_nyx_selected_cpu_historical as historical


def test_weekly_schedule_preserves_the_complete_physical_year():
    origins = historical.annual_origins()
    assert len(origins) == 53
    assert origins[0] == ("2025-09-24", "2025-10-01")
    assert origins[-1] == ("2026-09-23", "2026-09-24")
    chunks = [historical.grid(first, stop) for first, stop in origins]
    assert chunks[0].append(chunks[1:]).equals(historical.ANNUAL_GRID)
    assert len(historical.ANNUAL_GRID) == 8760


def test_selected_compositions_use_fixed_threshold_and_mean():
    index = pd.date_range("2026-09-23", periods=3, freq="h", tz="UTC")
    series = lambda values: pd.Series(values, index=index, dtype=float)
    experts = {
        "fr_residual_1000_FR": series([119, 120, 81]),
        "cwe_residual_2000_BE": series([118, 122, 75]),
        "cwe_absolute_2000_BE": series([120, 118, 85]),
        "cwe_residual_2000_NL": series([1, 2, 3]),
        "cwe_absolute_2000_NL": series([3, 4, 5]),
    }
    references = {"FR": series([100, 100, 100]), "BE": series([100, 100, 100])}
    result = historical.compose_selected(experts, references)
    assert result["FR"].tolist() == [100, 120, 100]
    assert result["BE"].tolist() == [100, 120, 80]
    assert result["NL"].tolist() == [2, 3, 4]


def test_official_paired_rows_drive_rmse_and_wins():
    index = pd.date_range("2025-09-24", periods=8760, freq="h", tz="UTC")
    point = pd.Series(np.zeros(8760), index=index)
    observed = np.zeros(8760)
    storm = np.ones(8760)
    storm[-1] = np.nan
    comparison = pd.DataFrame({"actual": observed, "storm": storm}, index=index)
    score = historical.score_selected(point, comparison)
    assert score["paired_hours"] == 8759
    assert score["rmse"] == 0
    assert score["storm_rmse"] == 1
    assert score["wins_vs_storm"] == 8759
    assert score["both_criteria_met"] is True


def test_missing_local_archive_fails_before_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(historical, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="Missing local historical artifact"):
        historical.preflight()


def test_checkpoint_reuses_only_exact_manifest_and_utc_points(tmp_path):
    role, origin, stop = "fr_residual_1000", "2025-09-24", "2025-10-01"
    path = historical._checkpoint_path(tmp_path, role, origin)
    index = historical.grid(origin, stop)
    points = {"FR": pd.Series(np.arange(len(index), dtype=float), index=index)}
    audit = {"origin_day": origin, "stop_day_exclusive": stop,
             "tree_count": 1000, "fit_seconds": 1.25}
    historical._write_checkpoint(path, role=role, origin=origin, stop=stop,
                                 manifest_sha256="source-and-code-digest",
                                 points=points, audit=audit)
    loaded, loaded_audit = historical._read_checkpoint(
        path, role=role, origin=origin, stop=stop,
        manifest_sha256="source-and-code-digest")
    assert loaded["FR"].equals(points["FR"])
    assert loaded_audit == audit
    with pytest.raises(ValueError, match="identity differs"):
        historical._read_checkpoint(path, role=role, origin=origin, stop=stop,
                                    manifest_sha256="different-digest")
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    data["points"]["FR"][0] = -999
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="point digest differs"):
        historical._read_checkpoint(path, role=role, origin=origin, stop=stop,
                                    manifest_sha256="source-and-code-digest")


def test_manifest_refuses_version_or_thread_changes(tmp_path):
    output = tmp_path / "replay"
    manifest = historical._run_manifest({"sources": "fixed"}, thread_count=2)
    digest = historical._prepare_output(output, manifest)
    assert digest == historical._prepare_output(output, manifest)
    assert "pyarrow" in manifest["dependency_versions"]
    changed = historical._run_manifest({"sources": "fixed"}, thread_count=8)
    with pytest.raises(ValueError, match="manifest differs"):
        historical._prepare_output(output, changed)


def test_partial_replay_has_metrics_without_final_verdict(tmp_path, monkeypatch):
    index = historical.grid("2025-09-24", "2025-10-01")
    full = historical.ANNUAL_GRID
    reference = {z: pd.Series(np.zeros(len(full)), index=full)
                 for z in ("FR", "BE")}
    comparisons = {z: pd.DataFrame({"actual": np.zeros(len(full)),
                                    "storm": np.ones(len(full))}, index=full)
                   for z in historical.COUNTRIES}
    monkeypatch.setattr(historical, "preflight", lambda: {"frozen": "hash"})
    monkeypatch.setattr(historical, "_plan", lambda role: {})
    monkeypatch.setattr(historical, "_load_baselines", lambda plan: ({}, {}, {}))
    monkeypatch.setattr(historical, "_load_references", lambda: (reference, {}))
    monkeypatch.setattr(historical, "_load_comparisons", lambda: (comparisons, {}))

    def fake_fit(role, origins, actual, nyx, thread_count, **kwargs):
        return ({zone: pd.Series(np.zeros(len(index)), index=index)
                 for zone in historical.ROLE_COUNTRIES[role]},
                [{"reused_checkpoint": False}])

    monkeypatch.setattr(historical, "_fit_role", fake_fit)
    output = tmp_path / "partial"
    progress = historical.run_replay(output, thread_count=2, max_origins=1)
    assert progress["final_verdict_available"] is False
    assert progress["annual_score_computed"] is False
    assert progress["origins_completed_in_prefix"] == 1
    assert progress["new_cpu_fits_this_invocation"] == 3
    assert set(progress["provisional_metrics_diagnostic_only"]) == set(historical.COUNTRIES)
    assert (output / "progress.json").is_file()
    assert not (output / "receipt.json").exists()
    assert not (output / "rapport.md").exists()
