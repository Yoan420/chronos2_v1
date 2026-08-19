from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from runs.tmp import prototype_be_crosszone_autonomous as screen


def _hourly_days(start: str, days: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=days * 24, freq="h", tz="UTC")


def test_convex_l1_is_nonnegative_and_sums_to_one() -> None:
    levels = np.array(
        [
            [0.0, 1.0, 4.0, 6.0, 8.0],
            [0.0, 3.0, 4.0, 6.0, 8.0],
            [0.0, 5.0, 4.0, 6.0, 8.0],
        ]
    )
    actual = levels[:, 1]

    parameters = screen._fit_convex_l1(levels, actual)
    prediction = screen._predict_candidate(parameters, levels)
    weights = np.asarray(parameters["weights"])

    assert np.all(weights >= -1e-12)
    assert weights.sum() == pytest.approx(1.0)
    assert prediction == pytest.approx(actual)


def test_ridge_uses_only_neighbor_spreads_relative_to_be() -> None:
    rng = np.random.default_rng(7)
    base = rng.normal(50.0, 5.0, size=200)
    spreads = rng.normal(size=(200, 4))
    levels = np.column_stack([base, base[:, None] + spreads])
    actual = base + 2.0 + 0.8 * spreads[:, 0] - 0.3 * spreads[:, 2]

    parameters = screen._fit_ridge_spreads(levels, actual)
    original = screen._predict_candidate(parameters, levels)
    shifted = screen._predict_candidate(parameters, levels + 17.0)

    # A common level shift leaves spreads/correction unchanged and shifts the
    # price prediction by the same amount.
    assert shifted - original == pytest.approx(np.full(len(levels), 17.0))
    assert parameters["alpha"] == screen.RIDGE_ALPHA
    assert parameters["neighbor_zones"] == ["FR", "DE", "NL", "ES"]


def test_second_level_crossfit_never_fits_its_validation_fold() -> None:
    index = _hourly_days("2024-08-11 22:00:00", screen.A_DAYS)
    hour = np.arange(len(index), dtype=float)
    be = 40.0 + np.sin(hour / 24.0)
    levels = pd.DataFrame(
        {
            "BE": be,
            "FR": be + 1.0,
            "DE": be + 2.0,
            "NL": be + 3.0,
            "ES": be + 4.0,
        },
        index=index,
    ).loc[:, list(screen.ZONE_ORDER)]
    actual = be + 0.5

    scores, selected = screen._crossfit_family(levels, actual)

    assert selected in screen.FAMILY_ORDER
    for candidate in screen.FAMILY_ORDER:
        folds = scores[candidate]["folds"]
        assert [item["fit_prior_A_days"] for item in folds] == [49, 98, 147, 196]
        assert [item["validation_A_days"] for item in folds] == [49] * 4
        assert scores[candidate]["all"]["days"] == 196


def test_a_always_freezes_best_predeclared_recipe_even_when_not_robust() -> None:
    index = _hourly_days("2024-08-11 22:00:00", screen.A_DAYS)
    be = np.linspace(20.0, 80.0, len(index))
    levels = pd.DataFrame(
        {
            "BE": be,
            "FR": be + 10.0,
            "DE": be + 20.0,
            "NL": be + 30.0,
            "ES": be + 40.0,
        },
        index=index,
    ).loc[:, list(screen.ZONE_ORDER)]

    scores, selected = screen._crossfit_family(levels, be.copy())

    assert selected in screen.FAMILY_ORDER
    assert scores[selected]["all"]["gain"] == pytest.approx(0.0, abs=1e-9)
    assert scores[selected]["robust_on_A"] is False


def test_gate_requires_gain_both_halves_and_positive_daily_bootstrap() -> None:
    index = _hourly_days("2025-04-13 22:00:00", 60)
    actual = np.zeros(len(index))
    baseline = np.full(len(index), 2.0)
    candidate = np.full(len(index), 1.0)

    score = screen._score_gate(index, actual, baseline, candidate)

    assert score["all"]["gain"] == pytest.approx(1.0)
    assert score["first_half"]["days"] == 30
    assert score["last_half"]["days"] == 30
    assert score["bootstrap"]["low"] > 0.0
    assert score["passes"] is True


def test_gate_rejects_one_losing_half_even_if_overall_threshold_passes() -> None:
    index = _hourly_days("2025-04-13 22:00:00", 60)
    local_days = pd.Index(index.tz_convert("Europe/Brussels").date)
    last_half = local_days.isin(local_days.unique()[30:])
    actual = np.zeros(len(index))
    baseline = np.full(len(index), 2.0)
    candidate = np.full(len(index), 0.5)
    candidate[last_half] = 2.1

    score = screen._score_gate(index, actual, baseline, candidate)

    assert score["all"]["gain"] > screen.B1_MINIMUM_GAIN
    assert score["last_half"]["gain"] < 0.0
    assert score["gate_checks"]["last_30_day_gain_positive"] is False
    assert score["passes"] is False


def test_gate_does_not_round_a_gain_just_below_point_ten() -> None:
    index = _hourly_days("2025-04-13 22:00:00", 60)
    actual = np.zeros(len(index))
    baseline = np.full(len(index), 2.0)
    candidate = np.full(len(index), 2.0 - 0.0996)

    score = screen._score_gate(index, actual, baseline, candidate)

    assert score["all"]["gain"] == pytest.approx(0.0996)
    assert score["bootstrap"]["low"] > 0.0
    assert score["gate_checks"]["overall_gain_at_least_0_10"] is False
    assert score["passes"] is False


def test_origin_audit_enforces_exact_shared_d_minus_one_08_cutoff() -> None:
    index = pd.date_range("2025-03-29 23:00:00Z", periods=47, freq="h")
    origins = screen.zone_screen._expected_cutoff(
        index, cutoff_timezone=screen.CUTOFF_TIMEZONE
    )
    raw = pd.DataFrame({"forecast_origin_utc": origins}, index=index)

    audit = screen._origin_audit(raw, context="synthetic")
    assert audit["cutoff_violations"] == 0

    contaminated = raw.copy()
    contaminated.iloc[-1, 0] = contaminated.iloc[-1, 0] + pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="forecast origins differ"):
        screen._origin_audit(contaminated, context="synthetic")


def test_b2_seal_fails_closed_before_data_access(tmp_path: Path) -> None:
    frozen = {
        "name": "convex_l1_levels",
        "parameters": {
            "kind": "convex_l1_levels",
            "zone_order": list(screen.ZONE_ORDER),
            "weights": [1.0, 0.0, 0.0, 0.0, 0.0],
        },
    }
    payload = {
        "protocol": {
            "phase": "b1",
            "final_phase_exposed": False,
            "storm_used_as_feature": False,
            "B2_loaded": False,
            "family_predeclared": screen.FAMILY_DEFINITION,
        },
        "frozen_candidate": frozen,
        "frozen_candidate_sha256": screen._canonical_sha256(frozen),
        "B1": {"passes": False},
    }
    path = tmp_path / "failed_b1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="B2 is blocked"):
        screen._load_passing_b1_seal(path)


def test_cli_exposes_no_final_and_requires_phase_seals(tmp_path: Path) -> None:
    output = str(tmp_path / "result.json")
    with pytest.raises(SystemExit):
        screen.parse_args(["--phase", "final", "--output-json", output])
    with pytest.raises(SystemExit):
        screen.parse_args(["--phase", "b1", "--output-json", output])
    with pytest.raises(SystemExit):
        screen.parse_args(["--phase", "b2", "--output-json", output])

    parsed = screen.parse_args(
        [
            "--phase",
            "b2",
            "--b1-seal",
            str(tmp_path / "b1.json"),
            "--output-json",
            output,
        ]
    )
    assert parsed.phase == "b2"
    assert not hasattr(parsed, "final")


def test_storm_token_is_rejected_from_any_input_schema() -> None:
    with pytest.raises(ValueError, match="forbidden Storm token"):
        screen._reject_storm(
            ["chronos2__q50", "storm_dashboard_official__q50"],
            context="candidate schema",
        )
