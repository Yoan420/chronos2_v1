from __future__ import annotations

import copy
from datetime import timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from chronos2_hourly.topology_contract import (
    EXPECTED_ADJACENCY,
    EXPECTED_CORRECTION_SCALE_GRID,
    EXPECTED_MINIMUM_PIT_COVERAGE,
    EXPECTED_MODEL_VALUES,
    FORMAL_GATE_HALF_DAYS,
    SPLIT_DAY_COUNTS,
    SUPPORTED_ZONES,
    TOPOLOGY_CONTEXT_COLUMNS,
    TopologyContractError,
    audit_blend_source,
    audit_topology_sources,
    load_topology_contract,
)


TIMEZONES = {
    "FR": "Europe/Paris",
    "DE": "Europe/Berlin",
    "BE": "Europe/Brussels",
    "NL": "Europe/Amsterdam",
    "ES": "Europe/Madrid",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_sealed_run(
    root: Path, zone: str, *, full_timeline: bool
) -> tuple[Path, str, Path, str]:
    run = root / "runs" / f"chronos2_hourly_{zone.lower()}_residual_extended_v1"
    inputs = run / "inputs"
    inputs.mkdir(parents=True)
    target_cache = root / "data" / "cache" / zone.lower() / "target__sealed.csv.gz"
    target_cache.parent.mkdir(parents=True)
    if full_timeline:
        index = pd.date_range(
            pd.Timestamp("2025-08-12", tz=TIMEZONES[zone]).tz_convert("UTC"),
            pd.Timestamp("2026-08-12", tz=TIMEZONES[zone]).tz_convert("UTC"),
            freq="h",
            inclusive="left",
        )
        local_days = index.tz_convert(TIMEZONES[zone]).date
        origins = [
            pd.Timestamp(day - timedelta(days=1), tz=TIMEZONES[zone])
            .replace(hour=8)
            .tz_convert("UTC")
            for day in local_days
        ]
        base = 50.0 + np.sin(np.arange(len(index)) / 24.0)
        pd.DataFrame(
            {
                "delivery_start_utc": index,
                "forecast_origin_utc": origins,
                "actual": base,
                "residual_corrected__q10": base - 5.0,
                "residual_corrected__q50": base,
                "residual_corrected__q90": base + 5.0,
            }
        ).to_csv(run / "backtest_hourly_oof.csv.gz", index=False)
        aligned = {"timestamp": index, "target": base}
        for code in SUPPORTED_ZONES:
            aligned[f"{code.lower()}_residual_load_fcst"] = np.arange(len(index), dtype=float)
        pd.DataFrame(aligned).to_csv(inputs / "aligned_inputs.csv.gz", index=False)
        target_index = pd.date_range(
            index[0] - pd.Timedelta(hours=24), index[-1], freq="h"
        )
        target_values = 50.0 + np.sin((np.arange(len(target_index)) - 24) / 24.0)
        pd.DataFrame(
            {"timestamp": target_index, "value": target_values}
        ).to_csv(target_cache, index=False)
    else:
        (run / "backtest_hourly_oof.csv.gz").write_bytes(b"sealed-backtest")
        (inputs / "aligned_inputs.csv.gz").write_bytes(b"sealed-inputs")
        pd.DataFrame(
            {
                "timestamp": ["2025-08-11T00:00:00Z"],
                "value": [50.0],
            }
        ).to_csv(target_cache, index=False)

    _write_json(
        run / "run_manifest.json",
        {
            "zone": zone,
            "timezone": TIMEZONES[zone],
            "target_contract": "hourly_utc_no_interpolation",
            "delivery_horizon": "dynamic_23_24_25",
            "prediction_inputs": [
                "chronos2_native",
                "supervised_hourly_experts",
                "pit_residual_load_covariates",
            ],
            "active_features": ["known_hour_sin"],
            "external_price_forecasts_loaded": [],
            "storm_used_as_feature": False,
        },
    )
    (run / "feature_manifest.csv").write_text(
        "feature\nknown_hour_sin\n", encoding="utf-8"
    )
    _write_json(
        run / "extended_residual_recipe.json",
        {
            "external_price_forecasts_loaded": [],
            "protocol": {"final_targets_used_for_evaluation_fit": False},
        },
    )
    _write_json(
        run / "metrics_hourly.json",
        {"training_diagnostics": {"metric_scope": "sealed_final_365_delivery_days"}},
    )
    members = [
        ("backtest_hourly_oof.csv.gz", "run_artifact"),
        ("inputs/aligned_inputs.csv.gz", "materialized_input"),
        ("run_manifest.json", "run_artifact"),
        ("feature_manifest.csv", "run_artifact"),
        ("extended_residual_recipe.json", "run_artifact"),
        ("metrics_hourly.json", "run_artifact"),
    ]
    _write_json(
        run / "artifact_checksums.json",
        {
            "artifacts": [
                {"path": relative, "role": role, "sha256": _sha(run / relative)}
                for relative, role in members
            ]
        },
    )
    return (
        run,
        _sha(run / "artifact_checksums.json"),
        target_cache,
        _sha(target_cache),
    )


def _write_blend_dependency(
    root: Path, zone: str, *, full_timeline: bool
) -> dict[str, object]:
    series = {"FR": "41551_native", "NL": "41554_native"}[zone]
    weight_mkonline = {"FR": 0.5022390717075804, "NL": 0.4677256033079484}[zone]
    weight_autonomous = 1.0 - weight_mkonline
    manifest = root / f"mkonline_{zone.lower()}_primary_dependency_v1.json"
    _write_json(
        manifest,
        {
            "terminal_series": series,
            "storm_token_found": False,
            "dependency_gate_passed": True,
        },
    )
    recipe = root / f"mkonline_{zone.lower()}_blend_recipe_v1.json"
    _write_json(
        recipe,
        {
            "schema_version": 1,
            "status": "frozen_before_final_opening",
            "zone": zone,
            "source_autonomous_model": "residual_corrected",
            "weights": {
                "autonomous": weight_autonomous,
                "mkonline_primary": weight_mkonline,
            },
            "external_expert": {
                "series": series,
                "dependency_manifest_sha256": _sha(manifest),
                "storm_used_as_feature": False,
                "interpolation_allowed": False,
            },
            "selection_protocol": {
                "final_target_used_for_weight_or_hyperparameters": False,
            },
        },
    )
    forecast = root / "runs" / "blend_inputs" / f"{zone.lower()}_primary.parquet"
    forecast.parent.mkdir(parents=True, exist_ok=True)
    if full_timeline:
        index = pd.date_range(
            pd.Timestamp("2025-08-12", tz=TIMEZONES[zone]).tz_convert("UTC"),
            pd.Timestamp("2026-08-12", tz=TIMEZONES[zone]).tz_convert("UTC"),
            freq="h",
            inclusive="left",
        )
        days = index.tz_convert(TIMEZONES[zone]).date
        cutoff = [
            pd.Timestamp(day - timedelta(days=1), tz="Europe/Paris")
            .replace(hour=8)
            .tz_convert("UTC")
            for day in days
        ]
        pd.DataFrame(
            {
                "value_time_utc": index,
                "snapshot_time_utc": cutoff,
                "revision_time_utc": cutoff,
                "value": np.linspace(20.0, 100.0, len(index)),
            }
        ).to_parquet(forecast, index=False)
    else:
        forecast.write_bytes(b"sealed-primary")
    return {
        "recipe_manifest": recipe.relative_to(root).as_posix(),
        "recipe_manifest_sha256": _sha(recipe),
        "dependency_manifest": manifest.relative_to(root).as_posix(),
        "dependency_manifest_sha256": _sha(manifest),
        "forecast_file": forecast.relative_to(root).as_posix(),
        "forecast_file_sha256": _sha(forecast),
        "primary_series": series,
        "current_weight_autonomous": weight_autonomous,
        "current_weight_mkonline": weight_mkonline,
    }


def _payload(root: Path, *, full_timeline: bool = False) -> dict:
    sources = {}
    for zone in SUPPORTED_ZONES:
        run, checksum, target_cache, target_cache_sha = _write_sealed_run(
            root, zone, full_timeline=full_timeline
        )
        sources[zone] = {
            "timezone": TIMEZONES[zone],
            "autonomous_run": run.relative_to(root).as_posix(),
            "checksum_manifest_sha256": checksum,
            "target_cache": target_cache.relative_to(root).as_posix(),
            "target_cache_sha256": target_cache_sha,
        }
    dependencies = {
        zone: _write_blend_dependency(root, zone, full_timeline=full_timeline)
        for zone in ("FR", "NL")
    }
    return {
        "schema_version": 1,
        "experiment_id": "pricefm_topology_v1",
        "output_directory": "runs/experiments/pricefm_topology_v1",
        "sources": sources,
        "topology": {
            "radii": [0, 1],
            "adjacency": {zone: list(EXPECTED_ADJACENCY[zone]) for zone in SUPPORTED_ZONES},
            "residual_load": {
                "columns": {
                    zone: f"{zone.lower()}_residual_load_fcst"
                    for zone in SUPPORTED_ZONES
                },
                "source": "sealed_pit_materialization",
                "interpolation": "none",
            },
            "price": {
                "column": "actual",
                "lag_hours": 24,
                "lag_basis": "physical_utc_rows",
                "source_local_day_strictly_before_delivery": True,
            },
            "minimum_pit_coverage": 0.90,
            "missing_policy": "preserve_nan",
            "fallback": "identity",
        },
        "model": {
            **dict(EXPECTED_MODEL_VALUES),
            "correction_scale_grid": list(EXPECTED_CORRECTION_SCALE_GRID),
        },
        "protocol": {
            "start_local_day": "2025-08-12",
            "splits": dict(SPLIT_DAY_COUNTS),
            "selection_min_gain_eur_mwh": 0.02,
            "training": {
                "selection_fit_blocks": ["seed"],
                "selection_score_block": "A",
                "selection_includes_identity": True,
                "selection_radius_candidates": [0, 1],
                "selection_correction_scale_grid": list(EXPECTED_CORRECTION_SCALE_GRID),
                "development_score_block": "development",
                "development_used_for_formal_gate": False,
                "formal_gate_periods_unopened_before_freeze": True,
                "recipe_and_config_frozen_before_formal_gates": True,
                "gate_fit_blocks": ["seed", "A", "development"],
                "gate_score_blocks": ["B1", "B2"],
                "refit_between_gate_blocks": True,
                "b2_refit_add_blocks": ["B1"],
                "final_fit_blocks": ["seed", "A", "development", "B1", "B2"],
                "final_score_block": "final",
                "single_final_opening": True,
            },
            "gates": {
                "minimum_mae_gain_eur_mwh": 0.05,
                "require_positive_halves": True,
                "bootstrap_samples": 20_000,
                "bootstrap_seed": 20260821,
                "bootstrap_confidence": 0.95,
                "require_bootstrap_lower_bound_positive": True,
            },
            "storm": {
                "evaluation_only": True,
                "load_after_candidate_freeze": True,
                "used_for_selection_or_tuning": False,
            },
        },
        "variants": {
            "autonomous": {
                "zones": list(SUPPORTED_ZONES),
                "native_model": "topology_autonomous",
                "baseline_model": "residual_corrected",
                "prediction_inputs": [
                    "sealed_autonomous_quantiles",
                    "pit_residual_load_forecasts",
                    "lagged_day_ahead_prices",
                ],
            },
            "mkonline_blend": {
                "zones": ["FR", "NL"],
                "native_model": "topology_mkonline_blend",
                "baseline_model": "mkonline_blend",
                "recalibrate_after_autonomous_gates": True,
                "weight_fit_block": "A",
                "weight_fit_method": "constrained_l1_grid",
                "weight_grid_step": 0.025,
                "veto_blocks": ["B1", "B2"],
                "final_targets_used_for_weight_or_hyperparameters": False,
                "cutoff_timezone": "Europe/Paris",
                "dependencies": dependencies,
            },
        },
    }


def _project(tmp_path: Path, *, full_timeline: bool = False) -> tuple[Path, dict]:
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    payload = _payload(root, full_timeline=full_timeline)
    path = root / "config" / "topology.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path, payload


def _rewrite(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_contract_exposes_frozen_api_and_safe_outputs(tmp_path: Path) -> None:
    path, _ = _project(tmp_path)
    contract = load_topology_contract(path)

    assert tuple(contract.sources) == SUPPORTED_ZONES
    assert contract.radii == (0, 1)
    assert contract.zones_within_radius("FR", 0) == ("FR",)
    assert set(contract.zones_within_radius("FR", 1)) == {"FR", "BE", "DE", "ES"}
    assert len(TOPOLOGY_CONTEXT_COLUMNS) == 22
    assert contract.model.correction_scale_grid == EXPECTED_CORRECTION_SCALE_GRID
    assert contract.output_directory_for("DE", "autonomous").is_relative_to(
        contract.project_root / "runs" / "experiments"
    )
    with pytest.raises(TopologyContractError, match="forbidden"):
        contract.output_directory_for("DE", "mkonline_blend")
    with pytest.raises(TopologyContractError, match="must not be read"):
        contract.blend_dependency_for("ES")


def test_windows_are_exact_and_dst_aware() -> None:
    # Build a contract-shaped instance through the normal loader in the next
    # tests; here the public constants certify the governance split itself.
    assert dict(SPLIT_DAY_COUNTS) == {
        "seed": 120,
        "A": 65,
        "development": 60,
        "B1": 30,
        "B2": 30,
        "final": 60,
    }
    assert sum(SPLIT_DAY_COUNTS.values()) == 365
    assert dict(FORMAL_GATE_HALF_DAYS) == {
        "B1": (15, 15),
        "B2": (15, 15),
        "final": (30, 30),
    }
    assert EXPECTED_MINIMUM_PIT_COVERAGE == 0.90


def test_loaded_windows_match_the_sealed_year(tmp_path: Path) -> None:
    path, _ = _project(tmp_path)
    windows = load_topology_contract(path).split_windows("FR")
    assert [(item.name, str(item.start_local_day), str(item.end_local_day), item.expected_physical_hours) for item in windows] == [
        ("seed", "2025-08-12", "2025-12-09", 2881),
        ("A", "2025-12-10", "2026-02-12", 1560),
        ("development", "2026-02-13", "2026-04-13", 1439),
        ("B1", "2026-04-14", "2026-05-13", 720),
        ("B2", "2026-05-14", "2026-06-12", 720),
        ("final", "2026-06-13", "2026-08-11", 1440),
    ]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda p: p["protocol"]["splits"].update({"development": 59}), "splits"),
        (lambda p: p["protocol"]["training"].update({"development_used_for_formal_gate": True}), "training"),
        (lambda p: p["protocol"]["training"].update({"formal_gate_periods_unopened_before_freeze": False}), "training"),
        (lambda p: p["protocol"]["training"].update({"gate_fit_blocks": ["seed", "A"]}), "training"),
        (lambda p: p["protocol"]["training"].update({"refit_between_gate_blocks": False}), "training"),
        (lambda p: p["protocol"]["gates"].update({"minimum_mae_gain_eur_mwh": 0.10}), "0.05"),
        (lambda p: p["model"].update({"max_iter": 240}), "frozen"),
        (lambda p: p["model"].update({"correction_scale_grid": [0, 1]}), "correction_scale_grid"),
        (lambda p: p["topology"].update({"minimum_pit_coverage": 0.60}), "0.90"),
    ],
)
def test_governance_cannot_be_relaxed(tmp_path: Path, mutator, message: str) -> None:
    path, payload = _project(tmp_path)
    mutator(payload)
    _rewrite(path, payload)
    with pytest.raises(TopologyContractError, match=message):
        load_topology_contract(path)


@pytest.mark.parametrize("token", ["Storm", "mkonline"])
def test_competing_forecasts_are_forbidden_from_topology_inputs(
    tmp_path: Path, token: str
) -> None:
    path, payload = _project(tmp_path)
    payload["variants"]["autonomous"]["prediction_inputs"].append(
        f"hidden_{token}_forecast"
    )
    _rewrite(path, payload)
    with pytest.raises(TopologyContractError, match="forbidden prediction input"):
        load_topology_contract(path)


def test_storm_policy_cannot_load_before_freeze(tmp_path: Path) -> None:
    path, payload = _project(tmp_path)
    payload["protocol"]["storm"]["load_after_candidate_freeze"] = False
    _rewrite(path, payload)
    with pytest.raises(TopologyContractError, match="post-freeze"):
        load_topology_contract(path)


def test_source_checksum_tamper_fails_closed(tmp_path: Path) -> None:
    path, payload = _project(tmp_path)
    run = path.parents[1] / payload["sources"]["FR"]["autonomous_run"]
    (run / "backtest_hourly_oof.csv.gz").write_bytes(b"tampered")
    with pytest.raises(TopologyContractError, match="checksum mismatch"):
        load_topology_contract(path)


def test_target_cache_tamper_fails_before_price_read(tmp_path: Path) -> None:
    path, payload = _project(tmp_path)
    target = path.parents[1] / payload["sources"]["FR"]["target_cache"]
    target.write_bytes(b"tampered")
    with pytest.raises(TopologyContractError, match="target cache hash mismatch"):
        load_topology_contract(path)


def test_loader_performs_no_mkonline_file_io(tmp_path: Path) -> None:
    path, payload = _project(tmp_path)
    root = path.parents[1]
    for dependency in payload["variants"]["mkonline_blend"]["dependencies"].values():
        for key in ("recipe_manifest", "dependency_manifest", "forecast_file"):
            (root / dependency[key]).unlink()

    # Even verify_hashes=True authenticates only autonomous sources here.
    contract = load_topology_contract(path, verify_hashes=True)
    assert contract.blend_dependency_for("FR").primary_series == "41551_native"
    with pytest.raises(TopologyContractError, match="must be promoted"):
        audit_blend_source(contract, "FR", autonomous_promoted=False)
    with pytest.raises(TopologyContractError, match="must not be read"):
        audit_blend_source(contract, "DE", autonomous_promoted=True)


def test_current_blend_weights_are_reconciled_with_sealed_recipe(tmp_path: Path) -> None:
    path, payload = _project(tmp_path, full_timeline=True)
    dependency = payload["variants"]["mkonline_blend"]["dependencies"]["FR"]
    recipe_path = path.parents[1] / dependency["recipe_manifest"]
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe["weights"] = {"autonomous": 0.60, "mkonline_primary": 0.40}
    _write_json(recipe_path, recipe)
    dependency["recipe_manifest_sha256"] = _sha(recipe_path)
    _rewrite(path, payload)

    contract = load_topology_contract(path)
    with pytest.raises(TopologyContractError, match="production weights differ"):
        audit_blend_source(contract, "FR", autonomous_promoted=True)


def test_output_must_remain_below_runs_experiments(tmp_path: Path) -> None:
    path, payload = _project(tmp_path)
    payload["output_directory"] = "runs/live/pricefm_topology_v1"
    _rewrite(path, payload)
    with pytest.raises(TopologyContractError, match="output_directory"):
        load_topology_contract(path)


def test_full_frozen_sources_and_blend_are_audited(tmp_path: Path) -> None:
    path, _ = _project(tmp_path, full_timeline=True)
    contract = load_topology_contract(path)
    audits = audit_topology_sources(contract)
    assert set(audits) == set(SUPPORTED_ZONES)
    assert {item["n_hours"] for item in audits.values()} == {8760}
    assert all(item["forecast_origin_violations"] == 0 for item in audits.values())
    assert all(min(item["pit_coverage"].values()) == 1.0 for item in audits.values())
    assert all(item["price_lag_masked_same_day_hours"] == 1 for item in audits.values())
    assert all(item["price_lag_strict_prior_day_hours"] == 8759 for item in audits.values())

    fr = audit_blend_source(contract, "FR", autonomous_promoted=True)
    nl = audit_blend_source(contract, "NL", autonomous_promoted=True)
    assert fr["n_hours"] == nl["n_hours"] == 8760
    assert fr["cutoff_violations"] == nl["cutoff_violations"] == 0
    with pytest.raises(TopologyContractError, match="must not be read"):
        audit_blend_source(contract, "BE", autonomous_promoted=True)
