from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from chronos2_exogenous import prospective_inputs as module
from chronos2_exogenous.feature_bank import (
    _required_columns, cutoff_by_delivery_hour, default_project_sources,
    delivery_utc_index,
)


def _fixture(tmp_path, monkeypatch, *, days=("2026-09-03",), zones=("FR",)):
    root = tmp_path / "project"
    root.mkdir()
    start = (pd.Timestamp(min(days)) - pd.Timedelta(days=88)).date().isoformat()
    index = delivery_utc_index(start, max(days))
    cutoffs = cutoff_by_delivery_hour(index)
    for zone in zones:
        for source in default_project_sources(root, zone=zone):
            if source.path.exists():
                continue
            source.path.parent.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame({c: np.ones(len(index)) for c in _required_columns(source)})
            frame[source.timestamp_column] = index
            for column in source.information_time_columns:
                frame[column] = cutoffs - pd.Timedelta(hours=1)
            if source.cutoff_column:
                frame[source.cutoff_column] = cutoffs
            if source.age_column:
                frame[source.age_column] = cutoffs - pd.Timedelta(hours=1)
            if source.cutoff_column:
                frame[source.cutoff_column] = cutoffs
            if source.stage_column:
                frame[source.stage_column] = source.allowed_stages[0]
            if source.eligibility_column:
                frame[source.eligibility_column] = True
            if source.operational_eligibility_column:
                frame[source.operational_eligibility_column] = False
            frame.to_parquet(source.path, index=False)
            source.audit_path.write_text(json.dumps({
                "output_sha256": module._sha256(source.path), "start_day": start,
                "end_day": max(days), "causality_violations": 0,
                "cutoff_time": "08:00", "cutoff_timezone": source.cutoff_timezone or "Europe/Paris",
            }), encoding="utf-8")
    def capture_target(**kwargs):
        return pd.Series(50.0, index=index, name="target"), {
            "series": f"canonical.{kwargs['zone']}", "fresh_api_read": kwargs["refresh"],
        }
    monkeypatch.setattr(module, "_capture_target", capture_target)
    output = root / "runs/experiments/trial/inputs"
    return root, output, index


def test_plan_is_read_only_and_routes_every_write_to_isolated_root(tmp_path, monkeypatch):
    root, output, _ = _fixture(tmp_path, monkeypatch)
    plan = module.build_trial_input_plan(project_root=root, output_directory=output, delivery_days=["2026-09-03"], zones=["FR"])
    assert not output.exists()
    assert plan["network_executed"] is False
    assert len(plan["commands"]) == 3
    assert len(plan["source_history"]) == 12
    for command in plan["commands"]:
        assert "--insecure" not in command
        assert "--overwrite" not in command
        destination_flag = "--output-root" if "--output-root" in command else "--output-dir"
        assert Path(command[command.index(destination_flag) + 1]).is_relative_to(output)


@pytest.mark.parametrize("relative", ["data/pit", "runs/experiments", "runs/live/a", "../outside"])
def test_plan_rejects_non_experiment_or_broad_destination(tmp_path, relative):
    with pytest.raises(module.ProspectiveInputError, match="sous-dossier"):
        module.build_trial_input_plan(project_root=tmp_path, output_directory=tmp_path / relative, delivery_days=["2026-09-03"])


def test_plan_refuses_existing_directory_before_inspecting_sources(tmp_path):
    output = tmp_path / "runs/experiments/existing"
    output.mkdir(parents=True)
    sentinel = output / "keep.txt"
    sentinel.write_text("user data")
    with pytest.raises(module.ProspectiveInputError, match="aucun ecrasement"):
        module.build_trial_input_plan(project_root=tmp_path, output_directory=output, delivery_days=["2026-09-03"])
    assert sentinel.read_text() == "user data"


def test_plan_rejects_bad_sha(tmp_path, monkeypatch):
    root, output, _ = _fixture(tmp_path, monkeypatch)
    source = default_project_sources(root, zone="FR")[0]
    source.path.write_bytes(b"tampered")
    with pytest.raises(module.ProspectiveInputError, match="SHA source"):
        module.build_trial_input_plan(project_root=root, output_directory=output, delivery_days=["2026-09-03"], zones=["FR"])
    assert not output.exists()


def test_copy_detects_source_mutation_since_read_only_plan(tmp_path, monkeypatch):
    root, output, _ = _fixture(tmp_path, monkeypatch)
    plan = module.build_trial_input_plan(project_root=root, output_directory=output, delivery_days=["2026-09-03"], zones=["FR"])
    Path(plan["source_history"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(module.ProspectiveInputError, match="depuis le plan"):
        module._copy_pinned_sources(plan)


@pytest.mark.parametrize("days,expected_hours", [
    (("2026-09-03",), 24),
    (("2026-03-29",), 23),
    (("2026-10-25",), 25),
    (tuple(pd.date_range("2026-09-03", "2026-09-08").strftime("%Y-%m-%d")), 144),
])
def test_offline_real_bank_panel_keeps_physical_hours_and_masks_labels(tmp_path, monkeypatch, days, expected_hours):
    root, output, _ = _fixture(tmp_path, monkeypatch, days=days)
    original = {s.path: module._sha256(s.path) for s in default_project_sources(root, zone="FR")}
    result = module.prepare_trial_inputs(project_root=root, output_directory=output,
                                         delivery_days=days, zones=["FR"], refresh=False)
    assert result["diagnostic_only"] is True
    assert result["production_pit_evidence"] is False
    assert result["promotion_eligible"] is False
    assert result["fresh_source_refresh"] is False
    assert result["historical_context_evidence"] == "reconstructed_asof_not_prospective_capture"
    assert result["panel_sha256"] == module._sha256(Path(result["panel_path"]))
    assert result["panel_audit_sha256"] == module._sha256(Path(result["panel_audit_path"]))
    panel = pd.read_parquet(result["panel_path"])
    assert panel.loc[panel.phase.eq("horizon"), "target"].isna().all()
    assert panel.phase.eq("horizon").sum() == expected_hours
    assert panel.phase.eq("context").sum() == 2048 * len(days)
    assert sum(v["FR"] for v in result["horizon_labels_available_at_capture"].values()) == expected_hours
    assert all(module._sha256(p) == sha for p, sha in original.items())
    snapshot = pd.read_csv(result["target_snapshots"]["FR"]["snapshot_path"])
    assert snapshot.target.notna().all()


def test_refresh_executes_only_three_checked_shell_false_commands(tmp_path, monkeypatch):
    root, output, _ = _fixture(tmp_path, monkeypatch)
    calls = []
    result = module.prepare_trial_inputs(project_root=root, output_directory=output,
        delivery_days=["2026-09-03"], zones=["FR"], refresh=True,
        command_runner=lambda argv, **kwargs: calls.append((argv, kwargs)))
    assert len(calls) == 3
    assert all(k == {"check": True, "cwd": root, "shell": False} for _, k in calls)
    assert result["fresh_source_refresh"] is True
    assert result["target_snapshots"]["FR"]["fresh_api_read"] is True


def test_refresh_failure_leaves_no_ready_manifest_and_keeps_seed(tmp_path, monkeypatch):
    root, output, _ = _fixture(tmp_path, monkeypatch)
    def fail(*args, **kwargs):
        raise RuntimeError("network error")
    with pytest.raises(RuntimeError, match="network error"):
        module.prepare_trial_inputs(project_root=root, output_directory=output,
            delivery_days=["2026-09-03"], zones=["FR"], command_runner=fail)
    assert (output / "seed_manifest.json").exists()
    assert not (output / "input_manifest.json").exists()


def test_schema_uses_only_values_and_accepts_checkpoint_column_order(tmp_path, monkeypatch):
    root, output, _ = _fixture(tmp_path, monkeypatch)
    schema = {
        "context_length": 2048, "target_columns": ["target"], "past_only_covariates": [],
        "known_future_covariates": [
            "fr_residual_load_fcst", "de_residual_load_fcst", "be_residual_load_fcst",
            "nl_residual_load_fcst", "es_residual_load_fcst", "known_hour_sin",
            "known_hour_cos", "known_dow_sin", "known_dow_cos", "known_doy_sin",
            "known_doy_cos", "known_is_weekend", "local_temperature_fcst",
            "local_wind_generation_fcst", "local_solar_generation_fcst",
            "ttf_m1_eur_mwh_th", "eua_first_dec_eur_tco2", "ttf_change_1d", "ttf_change_5d",
            "eua_change_1d", "eua_change_5d", "fuel_volatility_20d", "ccgt_marginal_cost_eur_mwh",
            "flowbased_availability", "flowbased_hour_imputed", "flowbased_cnec_count",
            "flowbased_external_ram_p10_gw", "flowbased_ram_p10_gw",
            "flowbased_ram_headroom_p10_to_median_gw", "flowbased_low_ram_share",
            "flowbased_ram_to_fmax_p05", "flowbased_fr_neighbor_ptdf_spread_p90",
            "flowbased_fr_neighbor_ram_stress_p95_per_gw", "flowbased_core_ram_stress_p95_per_gw",
            "flowbased_stress_hhi",
        ],
    }
    result = module.prepare_trial_inputs(project_root=root, output_directory=output,
        delivery_days=["2026-09-03"], zones=["FR"], refresh=False, expected_schema=schema)
    assert result["known_future_covariates"] == schema["known_future_covariates"]


def test_schema_drift_is_rejected_without_ready_manifest(tmp_path, monkeypatch):
    root, output, _ = _fixture(tmp_path, monkeypatch)
    with pytest.raises(module.ProspectiveInputError, match="Schema des covariables"):
        module.prepare_trial_inputs(project_root=root, output_directory=output,
            delivery_days=["2026-09-03"], zones=["FR"], refresh=False,
            expected_schema={"known_future_covariates": ["new_variable"]})
    assert not (output / "input_manifest.json").exists()


def test_target_context_hole_is_not_imputed(tmp_path, monkeypatch):
    root, output, index = _fixture(tmp_path, monkeypatch)
    target = pd.Series(50.0, index=index)
    target.iloc[-25] = np.nan
    monkeypatch.setattr(module, "_capture_target", lambda **kwargs: (target, {}))
    with pytest.raises(ValueError, match="context"):
        module.prepare_trial_inputs(project_root=root, output_directory=output,
            delivery_days=["2026-09-03"], zones=["FR"], refresh=False)
    assert not (output / "input_manifest.json").exists()


def test_canonical_target_refresh_uses_exact_series_no_fallback_no_cache_write(tmp_path, monkeypatch):
    import run_chronos2_exogenous_panel as resolver
    cache = tmp_path / "target.csv.gz"
    cache.write_bytes(b"unchanged canonical cache")
    base = tmp_path / "base.yaml"
    base.write_text("data:\n  saturn_url: https://example.invalid\n  saturn_author: unit_test\n")
    contract = {"series": "canonical.exact.series", "naive_timezone": "UTC",
                "base_config": str(base), "cache_path": str(cache)}
    monkeypatch.setattr(resolver, "_canonical_target_path", lambda *args: (cache, contract))
    monkeypatch.setattr(module, "create_saturn_client", lambda *args: object())
    calls = []
    index = delivery_utc_index("2026-09-03", "2026-09-03")
    def fetch(*args, **kwargs):
        calls.append((args, kwargs))
        return pd.Series(70.0, index=index)
    monkeypatch.setattr(module, "fetch_saturn_series_from_client", fetch)
    values, audit = module._capture_target(root=tmp_path, zone="FR", start="2026-09-03", end="2026-09-03", refresh=True)
    assert len(calls) == 1
    assert calls[0][0][1] == "canonical.exact.series"
    assert calls[0][1] == {"naive_timezone": "UTC", "nocache": True, "live": True}
    assert audit["fallback_used"] is False
    assert audit["historical_publication_times_verified"] is False
    assert cache.read_bytes() == b"unchanged canonical cache"
    assert values.index.equals(index)


def test_multi_zone_schema_preserves_country_weather_aliases(tmp_path, monkeypatch):
    zones = ["FR", "DE", "BE", "NL"]
    root, output, _ = _fixture(tmp_path, monkeypatch, zones=zones)
    result = module.prepare_trial_inputs(project_root=root, output_directory=output,
        delivery_days=["2026-09-03"], zones=zones, refresh=False)
    panel = pd.read_parquet(result["panel_path"])
    assert set(panel.item_id) == set(zones)
    assert "local_temperature_fcst" in panel
    assert not any(f"{z.lower()}_temperature_fcst" in panel for z in zones)
    assert all(v == 24 for v in result["horizon_labels_available_at_capture"]["2026-09-03"].values())
    assert len(result["target_snapshots"]) == 4
