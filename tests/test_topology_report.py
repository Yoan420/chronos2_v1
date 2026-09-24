from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.topology_report import (
    ANNUAL_EXPECTED_HOURS,
    PRICEFM_MODEL_URL,
    PRICEFM_PAPER_URL,
    PRICEFM_REPOSITORY_URL,
    TopologyAnnualReportArtifact,
    TopologyReportArtifact,
    TopologyReportError,
    load_topology_annual_evaluation,
    load_topology_evaluation,
    write_topology_annual_html_report,
    write_topology_annual_report_index,
    write_topology_html_report,
    write_topology_report_index,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _variant(
    *,
    candidate: str,
    baseline: str,
    gain: float = 0.2,
    passes: bool = True,
) -> dict[str, object]:
    baseline_mae = 10.0
    candidate_mae = baseline_mae - gain
    return {
        "candidate_model": candidate,
        "baseline_model": baseline,
        "selected_radius": 1,
        "topology_context": {
            "radius": 1,
            "neighbours": ["BE", "DE"],
        },
        "pit_coverage": {
            "fr_residual_load": {"coverage": 0.99},
            "be_residual_load": 0.97,
        },
        "metrics": {
            "b1": {
                "candidate_mae": 10.1,
                "baseline_mae": 10.2,
                "gain_eur_mwh": 0.1,
            },
            "b2": {
                "candidate_mae": 9.9,
                "baseline_mae": 10.1,
                "gain_eur_mwh": 0.2,
            },
            "final": {
                "n_days": 60,
                "n_hours": 1440,
                "candidate_mae": candidate_mae,
                "baseline_mae": baseline_mae,
                "gain_eur_mwh": gain,
                "daily_win_rate": 0.6,
            },
        },
        "gates": {
            "b1": {"passes": True, "reasons": []},
            "b2": {"passes": True, "reasons": []},
            "final": {
                "passes": passes,
                "reasons": [] if passes else ["gain final insuffisant"],
            },
        },
    }


def _payload(*, zone: str = "FR") -> dict[str, object]:
    timezone = {
        "FR": "Europe/Paris",
        "DE": "Europe/Berlin",
        "BE": "Europe/Brussels",
        "NL": "Europe/Amsterdam",
        "ES": "Europe/Madrid",
    }[zone]
    autonomous = _variant(
        candidate="topology_autonomous",
        baseline="residual_corrected",
    )
    blend = _variant(
        candidate="topology_mkonline_blend",
        baseline="mkonline_blend",
        gain=-0.05,
        passes=False,
    )
    blend.update(
        {
            "candidate_weights": {
                "topology_autonomous": 0.35,
                "mkonline_primary": 0.65,
            },
            "previous_production_weights": {
                "autonomous": 0.4,
                "mkonline_primary": 0.6,
            },
            "weight_grid_step": 0.025,
            "weight_fit_method": "constrained_l1_grid",
            "fitted_on": "A",
            "final_used_for_tuning": False,
            "autonomous_gates_passed_before_mkonline_load": True,
            "recipe_frozen_before_final": True,
            "dependency_manifest_sha256": SHA_C,
            "recommended_variant": "autonomous",
            "production_weights_unchanged": True,
        }
    )
    return {
        "schema_version": 1,
        "experiment_id": "pricefm_topology_v1",
        "zone": zone,
        "timezone": timezone,
        "feature_schema_sha256": SHA_A,
        "model_hyperparameters_sha256": SHA_B,
        "storm_loaded_after_candidate_freeze": True,
        "storm_used_as_prediction_input": False,
        "mkonline_used_by_topology": False,
        "variants": {
            "autonomous": autonomous,
            "mkonline_blend": blend,
        },
    }


def _write_evaluation(run_dir: Path, payload: dict[str, object]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "topology_evaluation.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _modern_autonomous_fallback(
    payload: dict[str, object], *, decision_stage: str
) -> dict[str, object]:
    autonomous = payload["variants"]["autonomous"]  # type: ignore[index]
    assert isinstance(autonomous, dict)
    autonomous["selected_arm"] = "radius1_scale0.5"
    autonomous["development"] = {
        "used_for_formal_gate": False,
        "metrics": {
            "candidate_mae": 10.05,
            "baseline_mae": 10.10,
            "gain_eur_mwh": 0.05,
        },
    }
    decision_index = ("b1", "b2", "final").index(decision_stage)
    opened_formal = ("b1", "b2", "final")[: decision_index + 1]
    autonomous["metrics"] = {
        stage: autonomous["metrics"][stage]  # type: ignore[index]
        for stage in opened_formal
    }
    autonomous["gates"] = {
        stage: {
            "passes": stage != decision_stage,
            "reasons": ["gain insuffisant"] if stage == decision_stage else [],
        }
        for stage in opened_formal
    }
    autonomous.update(
        {
            "opened_stages": ["a", "development", *opened_formal],
            "decision_stage": decision_stage,
            "promotion_decision": "fallback_identity",
            "sequential_decision_complete": True,
            "unopened_holdouts_spared": list(
                ("b1", "b2", "final")[decision_index + 1 :]
            ),
            "promoted": False,
        }
    )
    return autonomous


def _fake_report_writer(run_dir: Path, *, output_path: Path, **kwargs: object) -> Path:
    assert Path(run_dir).is_dir()
    assert kwargs["zone"] in {"FR", "DE"}
    path = Path(output_path)
    path.write_text(
        "<!doctype html><html><head><style>body{color:#111}</style></head>"
        "<body><header><h1>Rapport existant</h1></header><main>"
        "<section id='existing'>Graphiques existants</section></main>"
        "</body></html>",
        encoding="utf-8",
    )
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _segment_metrics(*, active_hours: int, active_days: int) -> dict[str, object]:
    return {
        "n_hours": active_hours,
        "n_days": active_days,
        "candidate_mae": 0.5,
        "baseline_mae": 1.0,
        "gain_eur_mwh": 0.5,
        "daily_win_rate": 1.0,
        "first_half_gain_eur_mwh": 0.5,
        "second_half_gain_eur_mwh": 0.5,
        "bootstrap_ci95_lower_eur_mwh": 0.5,
        "bootstrap_ci95_upper_eur_mwh": 0.5,
    }


def _annual_strategy_metrics(
    *,
    active_hours: int,
    active_days: int,
) -> dict[str, object]:
    coverage = active_hours / 8760
    candidate_mae = 1.0 - 0.5 * coverage
    return {
        "full_period_metrics": {
            "n_hours": 8760,
            "n_days": 365,
            "candidate_mae": candidate_mae,
            "baseline_mae": 1.0,
            "gain_eur_mwh": 1.0 - candidate_mae,
            "daily_win_rate": active_days / 365,
            "first_half_gain_eur_mwh": 0.0,
            "second_half_gain_eur_mwh": 2.0 * (1.0 - candidate_mae),
            "bootstrap_ci95_lower_eur_mwh": 0.0,
            "bootstrap_ci95_upper_eur_mwh": 1.0 - candidate_mae,
        },
        "active_only_metrics": (
            _segment_metrics(active_hours=active_hours, active_days=active_days)
            if active_hours
            else None
        ),
        "active_hours": active_hours,
        "active_days": active_days,
        "active_coverage": coverage,
        "daily_outcomes": {
            "all_days": {
                "n_days": 365,
                "wins": active_days,
                "ties": 365 - active_days,
                "losses": 0,
                "win_rate": active_days / 365,
                "tie_rate": (365 - active_days) / 365,
                "loss_rate": 0.0,
            },
            "active_days": {
                "n_days": active_days,
                "wins": active_days,
                "ties": 0,
                "losses": 0,
                "win_rate": 1.0 if active_days else None,
                "tie_rate": 0.0 if active_days else None,
                "loss_rate": 0.0 if active_days else None,
            },
        },
    }


def _refresh_annual_checksums(zone_dir: Path) -> None:
    names = [
        "annual_strategy_hourly.csv.gz",
        "annual_strategy_seal.json",
        "annual_policy_seal.json",
        "annual_metrics.json",
        "run_manifest.json",
    ]
    payload = {
        "algorithm": "sha256",
        "artifacts": [
            {
                "path": name,
                "role": "sealed_annual_artifact",
                "sha256": _sha(zone_dir / name),
                "size_bytes": (zone_dir / name).stat().st_size,
            }
            for name in names
        ],
    }
    (zone_dir / "artifact_checksums.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _annual_bundle(
    root: Path,
    *,
    zone: str = "BE",
    storm_available: bool = True,
    mkonline_available: bool = False,
) -> Path:
    timezone = {
        "FR": "Europe/Paris",
        "DE": "Europe/Berlin",
        "BE": "Europe/Brussels",
        "NL": "Europe/Amsterdam",
        "ES": "Europe/Madrid",
    }[zone]
    zone_dir = root / zone.lower()
    zone_dir.mkdir(parents=True)
    start = pd.Timestamp("2025-08-12", tz=timezone).tz_convert("UTC")
    stop = pd.Timestamp("2026-08-12", tz=timezone).tz_convert("UTC")
    index = pd.date_range(start, stop, freq="h", inclusive="left")
    assert len(index) == ANNUAL_EXPECTED_HOURS
    local = index.tz_convert(timezone)
    day_number = (local.normalize().tz_localize(None) - pd.Timestamp("2025-08-12")).days
    stage = np.select(
        [
            day_number < 120,
            day_number < 185,
            day_number < 245,
            day_number < 275,
            day_number < 305,
        ],
        ["seed", "a", "development", "b1", "b2"],
        default="final",
    )
    candidate_available = stage != "seed"
    candidate_formal = np.isin(stage, ["b1", "b2", "final"])
    governed_active = np.isin(stage, ["b2", "final"])
    shadow_active = candidate_formal
    actual = np.full(len(index), 100.0)
    base_q50 = np.full(len(index), 101.0)
    opened_q50 = np.where(candidate_available, 100.5, np.nan)
    governed_q50 = np.where(governed_active, 100.5, base_q50)
    shadow_q50 = np.where(shadow_active, 100.5, base_q50)
    local_days = local.strftime("%Y-%m-%d")
    origins = [
        pd.Timestamp(
            (pd.Timestamp(day) - pd.Timedelta(days=1)).date(), tz=timezone
        )
        .replace(hour=8)
        .tz_convert("UTC")
        for day in local_days
    ]
    frame = pd.DataFrame(
        {
            "delivery_start_utc": index,
            "forecast_origin_utc": origins,
            "protocol_stage": stage,
            "actual": actual,
            "candidate_available": candidate_available,
            "candidate_formal_oos": candidate_formal,
            "governed_topology_active": governed_active,
            "governed_reason": np.where(governed_active, "gate_passed", "identity"),
            "formal_shadow_topology_active": shadow_active,
            "formal_shadow_reason": np.where(shadow_active, "formal_oos", "identity"),
        }
    )
    for model, q50 in {
        "residual_corrected": base_q50,
        "topology_opened_candidate": opened_q50,
        "sequential_governed_strategy": governed_q50,
        "causal_formal_shadow_strategy": shadow_q50,
    }.items():
        frame[f"{model}__q10"] = q50 - 10.0
        frame[f"{model}__q50"] = q50
        frame[f"{model}__q90"] = q50 + 10.0
    strategy_path = zone_dir / "annual_strategy_hourly.csv.gz"
    frame.to_csv(strategy_path, index=False)
    policy = {
        "schema_version": 1,
        "zone": zone,
        "out_of_sample_scope": "mixed_sequential_governed",
    }
    policy_path = zone_dir / "annual_policy_seal.json"
    policy_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    strategy_seal_path = zone_dir / "annual_strategy_seal.json"
    strategy_seal_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "annual_strategy_sha256": _sha(strategy_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    governed = _annual_strategy_metrics(active_hours=2160, active_days=90)
    shadow = _annual_strategy_metrics(active_hours=2880, active_days=120)
    if storm_available:
        storm: dict[str, object] = {
            "available": True,
            "comparator_only": True,
            "used_for_prediction": False,
            "used_for_gate": False,
            "used_for_selection": False,
            "used_for_promotion": False,
            "audit": {"source_sha256": SHA_A},
            "metrics": {
                "candidate_model": "sequential_governed_strategy",
                "baseline_model": "residual_corrected",
                "benchmark_model": "storm_dashboard_official",
                "n_expected_hours": 8760,
                "n_paired_hours": 8759,
                "n_paired_days": 365,
                "pairing_coverage": 8759 / 8760,
                "candidate_mae": governed["full_period_metrics"]["candidate_mae"],  # type: ignore[index]
                "baseline_mae": 1.0,
                "storm_mae": 0.4,
                "candidate_gain_vs_storm_eur_mwh": -0.4,
                "baseline_gain_vs_storm_eur_mwh": -0.6,
                "candidate_vs_storm_daily_win_rate": 0.4,
            },
        }
    else:
        storm = {"available": False, "reason": f"Storm indisponible pour {zone}"}
    if mkonline_available:
        mkonline: dict[str, object] = {
            "available": True,
            "comparator_only": True,
            "topology_blend_candidate_available": False,
            "weights_recomputed": False,
            "weights": {"autonomous": 0.4, "mkonline_primary": 0.6},
            "metrics": {
                "n_hours": 8760,
                "n_days": 365,
                "mae": 0.8,
                "gain_vs_sequential_governed_eur_mwh": 0.05,
                "gain_vs_residual_corrected_eur_mwh": 0.2,
            },
            "audit": {
                "recipe_manifest_sha256": SHA_A,
                "dependency_manifest_sha256": SHA_B,
                "forecast_file_sha256": SHA_C,
                "cutoff_violations": 0,
            },
        }
    else:
        mkonline = {
            "available": False,
            "topology_blend_candidate_available": False,
            "reason": f"MKOnline production indisponible pour {zone}",
        }
    metrics = {
        "schema_version": 1,
        "report_type": "sealed_annual_365_strategy",
        "zone": zone,
        "timezone": timezone,
        "period": {
            "start_local_day": "2025-08-12",
            "end_local_day": "2026-08-11",
            "start_utc": index[0].isoformat(),
            "end_utc": index[-1].isoformat(),
            "n_local_days": 365,
            "n_hours": 8760,
        },
        "n_days": 365,
        "n_hours": 8760,
        "out_of_sample_scope": "mixed_sequential_governed",
        "formal_candidate_scope": "formal_holdouts_only",
        "selection_A_excluded_from_strategies": True,
        "development_excluded_from_strategies": True,
        "identity_fallback_is_not_candidate_prediction": True,
        "no_fit": True,
        "no_predict": True,
        "no_refit": True,
        "no_new_prediction": True,
        "used_for_gate": False,
        "used_for_selection": False,
        "used_for_promotion": False,
        "baseline": {
            "model": "residual_corrected",
            "n_hours": 8760,
            "n_days": 365,
            "mae": 1.0,
        },
        "strategies": {
            "sequential_governed_strategy": governed,
            "causal_formal_shadow_strategy": shadow,
        },
        "storm": storm,
        "mkonline_production_reference": mkonline,
        "pure_topology_annual": {
            "available": False,
            "reason": "seed et selection A ne permettent pas un candidat pur sans biais",
        },
        "calibration_artifact_manifest_sha256": SHA_A,
        "source_backtest_sha256": SHA_B,
        "source_candidate_prediction_sha256": SHA_C,
        "config_sha256": "d" * 64,
        "annual_policy_sha256": _sha(policy_path),
        "annual_strategy_sha256": _sha(strategy_path),
        "annual_strategy_seal_sha256": _sha(strategy_seal_path),
    }
    (zone_dir / "annual_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (zone_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "report_type": "sealed_annual_365_strategy",
                "zone": zone,
                "period": metrics["period"],
                "rolling365_enabled": False,
                "production_changed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _refresh_annual_checksums(zone_dir)
    return zone_dir


def _fake_annual_report_writer(
    run_dir: Path, *, output_path: Path, **kwargs: object
) -> Path:
    assert Path(run_dir).is_dir()
    assert kwargs["native_model"] == "sequential_governed_strategy"
    assert kwargs["baseline_model"] == "residual_corrected"
    path = Path(output_path)
    path.write_text(
        "<!doctype html><html><head><style>body{color:#111}</style></head>"
        "<body><main><section id='existing'>Moteur HTML existant</section></main>"
        "</body></html>",
        encoding="utf-8",
    )
    return path


def test_load_autonomous_evaluation_normalizes_audited_values(
    tmp_path: Path,
) -> None:
    evaluation = _write_evaluation(tmp_path / "run", _payload())

    record = load_topology_evaluation(evaluation)

    assert record.zone == "FR"
    assert record.variant == "autonomous"
    assert record.selected_radius == 1
    assert record.neighbours == ("BE", "DE")
    assert record.candidate_mae == pytest.approx(9.8)
    assert record.baseline_mae == pytest.approx(10.0)
    assert record.gain_eur_mwh == pytest.approx(0.2)
    assert record.daily_win_rate == pytest.approx(0.6)
    assert record.pit_min_coverage == pytest.approx(0.97)
    assert record.feature_schema_sha256 == SHA_A
    assert record.model_hyperparameters_sha256 == SHA_B
    assert record.complete is True
    assert record.status == "promoted"


def test_identity_selection_is_a_complete_fail_closed_fallback(
    tmp_path: Path,
) -> None:
    payload = _payload()
    autonomous = payload["variants"]["autonomous"]  # type: ignore[index]
    assert isinstance(autonomous, dict)
    autonomous.update(
        {
            "selected_arm": "identity",
            "selected_radius": 0,
            "topology_context": {"radius": 0, "neighbours": []},
            "development": {
                "used_for_formal_gate": False,
                "metrics": {
                    "candidate_mae": 10.0,
                    "baseline_mae": 10.0,
                    "gain_eur_mwh": 0.0,
                },
            },
            "metrics": {},
            "gates": {},
            "opened_stages": ["a", "development"],
            "decision_stage": "A_identity",
            "promotion_decision": "fallback_identity",
            "sequential_decision_complete": True,
            "unopened_holdouts_spared": ["b1", "b2", "final"],
            "promoted": False,
        }
    )
    evaluation = _write_evaluation(tmp_path / "run", payload)

    record = load_topology_evaluation(evaluation)

    assert record.complete is True
    assert record.status == "fallback"
    assert record.gate_passes is False
    assert record.candidate_mae is None
    assert record.baseline_mae is None
    assert record.decision_stage == "A_identity"
    assert record.opened_stages == ("a", "development")
    assert record.promotion_decision == "fallback_identity"
    assert record.promoted is False

    output = tmp_path / "identity_fallback.html"
    write_topology_html_report(
        tmp_path / "run",
        evaluation_path=evaluation,
        output_path=output,
        report_writer=_fake_report_writer,
    )
    rendered = output.read_text(encoding="utf-8")
    assert "Developpement (diagnostic)" in rendered
    assert "A_identity" in rendered
    assert "Motifs de la gate de decision" in rendered


@pytest.mark.parametrize("decision_stage", ["b1", "b2"])
def test_early_formal_rejection_is_complete_without_later_holdouts(
    tmp_path: Path,
    decision_stage: str,
) -> None:
    payload = _payload()
    _modern_autonomous_fallback(payload, decision_stage=decision_stage)
    evaluation = _write_evaluation(tmp_path / "run", payload)

    record = load_topology_evaluation(evaluation)

    assert record.complete is True
    assert record.status == "fallback"
    assert record.gate_passes is False
    assert record.decision_stage == decision_stage
    assert record.opened_stages[-1] == decision_stage
    assert record.promotion_decision == "fallback_identity"
    assert record.candidate_mae is not None
    assert "final" not in record.variant_payload["metrics"]
    assert "final" not in record.variant_payload["gates"]


def test_early_rejection_requires_explicit_fail_and_absent_later_stages(
    tmp_path: Path,
) -> None:
    payload = _payload()
    autonomous = _modern_autonomous_fallback(payload, decision_stage="b1")
    autonomous["gates"]["b1"]["passes"] = True  # type: ignore[index]
    autonomous["metrics"]["b2"] = {  # type: ignore[index]
        "candidate_mae": 9.9,
        "baseline_mae": 10.0,
    }
    evaluation = _write_evaluation(tmp_path / "run", payload)

    record = load_topology_evaluation(evaluation)

    assert record.complete is False
    assert record.status == "incomplete"
    assert any("explicitement fail" in issue for issue in record.completeness_issues)
    assert any("posterieurs" in issue for issue in record.completeness_issues)


def test_promoted_decision_still_requires_a_passing_final_gate(
    tmp_path: Path,
) -> None:
    payload = _payload()
    autonomous = payload["variants"]["autonomous"]  # type: ignore[index]
    assert isinstance(autonomous, dict)
    autonomous.update(
        {
            "selected_arm": "radius1_scale0.5",
            "development": {"used_for_formal_gate": False, "metrics": {}},
            "opened_stages": ["a", "development", "b1", "b2", "final"],
            "decision_stage": "final",
            "promotion_decision": "promoted",
            "sequential_decision_complete": True,
            "unopened_holdouts_spared": [],
            "promoted": True,
        }
    )
    autonomous["gates"]["final"]["passes"] = False  # type: ignore[index]
    evaluation = _write_evaluation(tmp_path / "run", payload)

    record = load_topology_evaluation(evaluation)

    assert record.complete is False
    assert record.status == "incomplete"
    assert any("final explicitement pass" in issue for issue in record.completeness_issues)


def test_load_rejects_a_contradictory_declared_gain(tmp_path: Path) -> None:
    payload = _payload()
    payload["variants"]["autonomous"]["metrics"]["final"][  # type: ignore[index]
        "gain_eur_mwh"
    ] = 9.0
    evaluation = _write_evaluation(tmp_path / "run", payload)

    with pytest.raises(TopologyReportError, match="contredit"):
        load_topology_evaluation(evaluation)


def test_missing_audit_evidence_is_visible_as_incomplete(tmp_path: Path) -> None:
    payload = _payload()
    payload.pop("feature_schema_sha256")
    payload["variants"]["autonomous"]["topology_context"][  # type: ignore[index]
        "neighbours"
    ] = []
    evaluation = _write_evaluation(tmp_path / "run", payload)

    record = load_topology_evaluation(evaluation)

    assert record.complete is False
    assert record.status == "incomplete"
    assert "feature_schema_sha256 absent" in record.completeness_issues
    assert "voisins absents pour delta=1" in record.completeness_issues


def test_mkonline_topology_input_flag_alias_is_accepted(tmp_path: Path) -> None:
    payload = _payload()
    payload.pop("mkonline_used_by_topology")
    payload["mkonline_used_as_topology_input"] = False
    evaluation = _write_evaluation(tmp_path / "run", payload)

    record = load_topology_evaluation(evaluation)

    assert record.complete is True
    assert record.status == "promoted"


def test_blend_has_distinct_gate_and_validates_candidate_and_previous_weights(
    tmp_path: Path,
) -> None:
    evaluation = _write_evaluation(tmp_path / "run", _payload())

    record = load_topology_evaluation(evaluation, variant="mkonline_blend")

    assert record.gate_passes is False
    assert record.status == "fallback"
    assert record.complete is True
    assert record.blend_weights is not None
    assert record.blend_weights.valid is True
    assert record.blend_weights.topology_autonomous == pytest.approx(0.35)
    assert record.blend_weights.previous_autonomous == pytest.approx(0.4)
    assert "poids MKOnline de production restent inchanges" in (
        record.fallback_message
    )


def test_modern_blend_rejection_falls_back_to_current_production_blend(
    tmp_path: Path,
) -> None:
    payload = _payload()
    blend = payload["variants"]["mkonline_blend"]  # type: ignore[index]
    assert isinstance(blend, dict)
    blend.update(
        {
            "metrics": {"b1": blend["metrics"]["b1"]},  # type: ignore[index]
            "gates": {
                "b1": {"passes": False, "reasons": ["gain insuffisant"]}
            },
            "opened_stages": ["a", "b1"],
            "decision_stage": "b1",
            "promotion_decision": "fallback_production_blend",
            "sequential_decision_complete": True,
            "unopened_holdouts_spared": ["b2", "final"],
            "promoted": False,
            "recommended_variant": "production_mkonline_blend",
            "production_weights_unchanged": True,
        }
    )
    evaluation = _write_evaluation(tmp_path / "run", payload)

    record = load_topology_evaluation(evaluation, variant="mkonline_blend")

    assert record.complete is True
    assert record.status == "fallback"
    assert record.decision_stage == "b1"
    assert record.opened_stages == ("a", "b1")
    assert record.promotion_decision == "fallback_production_blend"
    assert record.gate_passes is False
    assert record.candidate_mae == pytest.approx(10.1)
    assert "blend actuel" in record.fallback_message


def test_invalid_blend_weights_never_get_invented(tmp_path: Path) -> None:
    payload = _payload()
    blend = payload["variants"]["mkonline_blend"]  # type: ignore[index]
    blend["candidate_weights"] = {  # type: ignore[index]
        "topology_autonomous": 0.4,
        "mkonline_primary": 0.7,
    }
    evaluation = _write_evaluation(tmp_path / "run", payload)

    record = load_topology_evaluation(evaluation, variant="mkonline_blend")

    assert record.status == "incomplete"
    assert record.blend_weights is not None
    assert record.blend_weights.valid is False
    assert any("somme differe de 1" in issue for issue in record.completeness_issues)


def test_detailed_report_reuses_renderer_adds_banner_and_preserves_sources(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    evaluation = _write_evaluation(run_dir, _payload())
    backtest = run_dir / "backtest_hourly_oof.csv.gz"
    backtest.write_bytes(b"sealed-candidate")
    before = hashlib.sha256(backtest.read_bytes()).hexdigest()
    output = tmp_path / "reports" / "fr_autonomous.html"

    artifact = write_topology_html_report(
        run_dir,
        evaluation_path=evaluation,
        output_path=output,
        report_writer=_fake_report_writer,
    )

    rendered = output.read_text(encoding="utf-8")
    assert artifact.path == output.resolve()
    assert artifact.sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert rendered.count('data-report-section="topology-audit"') == 1
    assert "Graphiques existants" in rendered
    assert "Rayon topologique δ" in rendered
    assert "Fallback identite" not in rendered
    assert "Aucun code, poids ou jeu de données PriceFM" in rendered
    assert PRICEFM_PAPER_URL in rendered
    assert PRICEFM_REPOSITORY_URL in rendered
    assert PRICEFM_MODEL_URL in rendered
    assert 'id="topology-report-data"' in rendered
    assert hashlib.sha256(backtest.read_bytes()).hexdigest() == before
    assert not list(output.parent.glob("*.rendering-*.html"))


def test_banner_escapes_gate_reasons_and_embedded_json(tmp_path: Path) -> None:
    payload = _payload()
    payload["variants"]["autonomous"]["gates"]["final"][  # type: ignore[index]
        "reasons"
    ] = ["<img src=x onerror=alert(1)>"]
    evaluation = _write_evaluation(tmp_path / "run", payload)
    output = tmp_path / "report.html"

    write_topology_html_report(
        tmp_path / "run",
        evaluation_path=evaluation,
        output_path=output,
        report_writer=_fake_report_writer,
    )

    rendered = output.read_text(encoding="utf-8")
    assert "&lt;img src=x onerror=alert(1)&gt;" in rendered
    assert "<img src=x onerror=alert(1)>" not in rendered
    assert "\\u003cimg src=x onerror=alert(1)\\u003e" in rendered


def test_report_refuses_runs_live_before_calling_renderer(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "live" / "fr"
    _write_evaluation(run_dir, _payload())
    calls: list[bool] = []

    def writer(*args: object, **kwargs: object) -> Path:
        calls.append(True)
        raise AssertionError("renderer must not run")

    with pytest.raises(TopologyReportError, match="runs/live"):
        write_topology_html_report(
            run_dir,
            output_path=tmp_path / "report.html",
            report_writer=writer,
        )
    assert calls == []


def test_renderer_mutating_a_critical_artifact_fails_closed(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_evaluation(run_dir, _payload())
    backtest = run_dir / "backtest_hourly_oof.csv.gz"
    backtest.write_bytes(b"sealed")

    def mutating_writer(
        source: Path, *, output_path: Path, **kwargs: object
    ) -> Path:
        backtest.write_bytes(b"mutated")
        return _fake_report_writer(source, output_path=output_path, **kwargs)

    with pytest.raises(TopologyReportError, match="modifie"):
        write_topology_html_report(
            run_dir,
            output_path=tmp_path / "report.html",
            report_writer=mutating_writer,
        )
    assert not (tmp_path / "report.html").exists()


def test_consolidated_index_has_separate_slots_and_missing_country_status(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    evaluation = _write_evaluation(run_dir, _payload())
    autonomous = write_topology_html_report(
        run_dir,
        evaluation_path=evaluation,
        output_path=tmp_path / "reports" / "fr_auto.html",
        report_writer=_fake_report_writer,
    )
    blend = write_topology_html_report(
        run_dir,
        variant="mkonline_blend",
        evaluation_path=evaluation,
        output_path=tmp_path / "reports" / "fr_blend.html",
        report_writer=_fake_report_writer,
    )
    index = write_topology_report_index(
        [autonomous, blend],
        output_path=tmp_path / "reports" / "index.html",
        selected_zones=["FR", "DE"],
    )

    rendered = index.read_text(encoding="utf-8")
    assert rendered.count('data-variant="autonomous"') == 2
    assert rendered.count('data-variant="mkonline_blend"') == 1
    assert rendered.count('data-status="missing"') == 1
    assert "cand. 0.350 / 0.650 · prod. 0.400 / 0.600" in rendered
    assert "fr_auto.html" in rendered
    assert "fr_blend.html" in rendered
    assert "constrained_l1_grid" in (
        blend.path.read_text(encoding="utf-8")
    )
    assert "Aucun code, poids ou dataset PriceFM" in rendered


def test_duplicate_index_slot_is_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    evaluation = _write_evaluation(run_dir, _payload())
    record = load_topology_evaluation(evaluation)
    report = tmp_path / "report.html"
    report.write_text("<html></html>", encoding="utf-8")
    artifact = TopologyReportArtifact(
        path=report,
        evaluation_path=evaluation,
        record=record,
        sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
    )

    with pytest.raises(TopologyReportError, match="duplique"):
        write_topology_report_index(
            [artifact, replace(artifact)],
            output_path=tmp_path / "index.html",
        )


def test_load_annual_evaluation_requires_exact_365_scope_and_two_strategies(
    tmp_path: Path,
) -> None:
    zone_dir = _annual_bundle(tmp_path / "annual", zone="BE")

    record = load_topology_annual_evaluation(zone_dir)

    assert record.zone == "BE"
    assert record.n_days == 365
    assert record.n_hours == 8760
    assert record.out_of_sample_scope == "mixed_sequential_governed"
    assert record.baseline_mae == pytest.approx(1.0)
    assert record.sequential_governed_strategy.active_hours == 2160
    assert record.sequential_governed_strategy.active_days == 90
    assert record.sequential_governed_strategy.ties_all_days == 275
    assert record.sequential_governed_strategy.win_rate_active_days == pytest.approx(1.0)
    assert record.causal_formal_shadow_strategy.active_hours == 2880
    assert record.storm.available is True
    assert record.storm.paired_hours == 8759
    assert record.pure_topology_annual_available is False


def test_annual_report_reuses_engine_and_labels_mixed_scope_without_pure_claim(
    tmp_path: Path,
) -> None:
    zone_dir = _annual_bundle(tmp_path / "annual", zone="FR", mkonline_available=True)
    output = tmp_path / "reports" / "annual_fr.html"
    internal_names: dict[str, str] = {}

    def recording_writer(
        run_dir: Path, *, output_path: Path, **kwargs: object
    ) -> Path:
        internal_names["staging"] = Path(run_dir).name
        internal_names["temporary"] = Path(output_path).name
        return _fake_annual_report_writer(
            run_dir, output_path=output_path, **kwargs
        )

    artifact = write_topology_annual_html_report(
        zone_dir,
        output_path=output,
        report_writer=recording_writer,
    )

    rendered = output.read_text(encoding="utf-8")
    assert artifact.path == output.resolve()
    assert artifact.sha256 == _sha(output)
    assert rendered.count('data-report-section="topology-annual-audit"') == 1
    assert "365 jours scellés — stratégie mixte, sans refit rolling" in rendered
    assert "Stratégie séquentielle gouvernée" in rendered
    assert "Shadow causal formel" in rendered
    assert "Gagnés" in rendered and "Égalités" in rendered and "Perdus" in rendered
    assert "Jours topology actifs" in rendered
    assert "Moteur HTML existant" in rendered
    assert "Storm — référence appariée" in rendered
    assert "MKOnline — blend de production" in rendered
    assert "production_mkonline_blend" in rendered
    assert "topology_blend_candidate_available=false" in rendered
    assert "Topologie pure sur 365 jours" in rendered
    assert "N/A" in rendered
    assert 'id="topology-annual-report-data"' in rendered
    assert internal_names["staging"].startswith(".ae-")
    assert len(internal_names["staging"]) == 12
    assert internal_names["temporary"].startswith(".ar-")
    assert internal_names["temporary"].endswith(".html")
    assert len(internal_names["temporary"]) == 17
    assert not list(output.parent.glob(".ae-*"))
    assert not list(output.parent.glob(".ar-*.html"))


def test_annual_es_report_marks_storm_and_mkonline_unavailable(
    tmp_path: Path,
) -> None:
    zone_dir = _annual_bundle(
        tmp_path / "annual",
        zone="ES",
        storm_available=False,
        mkonline_available=False,
    )
    output = tmp_path / "reports" / "annual_es.html"

    artifact = write_topology_annual_html_report(
        zone_dir,
        output_path=output,
        report_writer=_fake_annual_report_writer,
    )

    rendered = artifact.path.read_text(encoding="utf-8")
    assert "Storm indisponible pour ES" in rendered
    assert "MKOnline production indisponible pour ES" in rendered
    assert rendered.count("N/A") >= 3


def test_annual_index_has_country_links_nas_and_mkonline_reference(
    tmp_path: Path,
) -> None:
    fr_dir = _annual_bundle(
        tmp_path / "annual", zone="FR", mkonline_available=True
    )
    es_dir = _annual_bundle(
        tmp_path / "annual", zone="ES", storm_available=False
    )
    fr = write_topology_annual_html_report(
        fr_dir,
        output_path=tmp_path / "reports" / "fr.html",
        report_writer=_fake_annual_report_writer,
    )
    es = write_topology_annual_html_report(
        es_dir,
        output_path=tmp_path / "reports" / "es.html",
        report_writer=_fake_annual_report_writer,
    )

    index = write_topology_annual_report_index(
        [fr, es],
        output_path=tmp_path / "reports" / "index.html",
        selected_zones=["FR", "ES", "DE"],
    )

    rendered = index.read_text(encoding="utf-8")
    assert "365 jours scellés — stratégie mixte, sans refit rolling" in rendered
    assert 'data-zone="FR"' in rendered
    assert 'data-zone="ES"' in rendered
    assert 'data-zone="DE" data-status="missing"' in rendered
    assert "fr.html" in rendered and "es.html" in rendered
    assert "Storm indisponible pour ES" in rendered
    assert "MKOnline production indisponible pour ES" in rendered
    assert "topology_blend_candidate_available=false" in rendered
    assert "Topologie pure 365j" in rendered


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("n_hours", 8759, "8760"),
        ("out_of_sample_scope", "selected_A_reused", "hors echantillon"),
        ("pure_topology_annual", {"available": True, "reason": "optimiste"}, "false"),
    ],
)
def test_annual_report_rejects_wrong_scope_or_ambiguous_oos(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    zone_dir = _annual_bundle(tmp_path / "annual", zone="BE")
    metrics_path = zone_dir / "annual_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics[field] = value
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _refresh_annual_checksums(zone_dir)

    with pytest.raises(TopologyReportError, match=match):
        load_topology_annual_evaluation(zone_dir)


def test_annual_report_refuses_runs_live_and_any_overwrite(
    tmp_path: Path,
) -> None:
    live_dir = _annual_bundle(tmp_path / "runs" / "live" / "annual", zone="FR")
    with pytest.raises(TopologyReportError, match="runs/live"):
        write_topology_annual_html_report(
            live_dir,
            output_path=tmp_path / "live.html",
            report_writer=_fake_annual_report_writer,
        )

    zone_dir = _annual_bundle(tmp_path / "safe", zone="FR")
    with pytest.raises(TopologyReportError, match="overwrite"):
        write_topology_annual_html_report(
            zone_dir,
            output_path=tmp_path / "safe.html",
            overwrite=True,
            report_writer=_fake_annual_report_writer,
        )
    existing = tmp_path / "existing.html"
    existing.write_text("sealed", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_topology_annual_html_report(
            zone_dir,
            output_path=existing,
            report_writer=_fake_annual_report_writer,
        )


def test_annual_renderer_mutating_a_sealed_input_fails_closed(
    tmp_path: Path,
) -> None:
    zone_dir = _annual_bundle(tmp_path / "annual", zone="BE")
    metrics_path = zone_dir / "annual_metrics.json"

    def mutating_writer(
        source: Path, *, output_path: Path, **kwargs: object
    ) -> Path:
        metrics_path.write_text("{}", encoding="utf-8")
        return _fake_annual_report_writer(
            source, output_path=output_path, **kwargs
        )

    output = tmp_path / "mutated.html"
    with pytest.raises(TopologyReportError, match="modifie"):
        write_topology_annual_html_report(
            zone_dir,
            output_path=output,
            report_writer=mutating_writer,
        )
    assert not output.exists()


def test_annual_manifest_can_be_finalized_with_html_only_after_render(
    tmp_path: Path,
) -> None:
    zone_dir = _annual_bundle(tmp_path / "annual", zone="BE")
    artifact = write_topology_annual_html_report(
        zone_dir,
        report_writer=_fake_annual_report_writer,
    )
    manifest_path = zone_dir / "artifact_checksums.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"].append(
        {
            "path": artifact.path.relative_to(zone_dir).as_posix(),
            "role": "annual_html_report",
            "sha256": artifact.sha256,
            "size_bytes": artifact.path.stat().st_size,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    reloaded = load_topology_annual_evaluation(zone_dir)

    assert reloaded.zone == "BE"
    assert _sha(artifact.path) == artifact.sha256
