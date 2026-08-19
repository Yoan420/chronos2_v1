from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import screen_multizone_exogenous_strict as screen


def _passing_score() -> dict[str, object]:
    return {
        "all": {
            "gain_vs_frozen_reference": 0.20,
            "gain_vs_calendar_control": 0.05,
        },
        "first30": {"gain_vs_frozen_reference": 0.10},
        "last30": {"gain_vs_frozen_reference": 0.12},
        "paired_day_interval_vs_frozen_reference": {
            "bootstrap_95_low": 0.01
        },
    }


def test_cli_deliberately_exposes_no_final_phase() -> None:
    with pytest.raises(SystemExit):
        screen.main(["--phase", "final", "--zone", "DE"])


@pytest.mark.parametrize("column", ["storm_signal", "target_price", "mkonline_fcst"])
def test_candidate_feature_names_reject_storm_target_and_prices(column: str) -> None:
    index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="forbidden"):
        screen._family_features(pd.DataFrame({column: [1.0, 2.0]}, index=index), "UTC")


def test_b1_gate_requires_both_halves_and_positive_bootstrap_low() -> None:
    score = _passing_score()
    assert screen._passes_b1(score, minimum_gain=0.10, minimum_bootstrap_low=0.0)

    score["last30"]["gain_vs_frozen_reference"] = -0.01
    assert not screen._passes_b1(
        score, minimum_gain=0.10, minimum_bootstrap_low=0.0
    )

    score = _passing_score()
    score["paired_day_interval_vs_frozen_reference"]["bootstrap_95_low"] = -0.01
    assert not screen._passes_b1(
        score, minimum_gain=0.10, minimum_bootstrap_low=0.0
    )


def test_b2_gate_cannot_use_storm_and_requires_both_halves() -> None:
    score = _passing_score()
    assert screen._passes_b2(score, minimum_bootstrap_low=0.0)
    score["first30"]["gain_vs_frozen_reference"] = 0.0
    assert not screen._passes_b2(score, minimum_bootstrap_low=0.0)


def test_parquet_filters_preserve_disjoint_a_and_b2_windows() -> None:
    a = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    b2 = pd.date_range("2025-03-01", periods=3, freq="h", tz="UTC")
    filters = screen._parquet_filters_for_index("value_time_utc", a.append(b2))
    assert len(filters) == 2
    assert filters[0][0][2] == a[0].to_pydatetime()
    assert filters[0][1][2] == a[-1].to_pydatetime()
    assert filters[1][0][2] == b2[0].to_pydatetime()
    assert filters[1][1][2] == b2[-1].to_pydatetime()


def test_storm_statistics_are_explicitly_evaluation_only() -> None:
    index = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
    actual = pd.Series(np.arange(48, dtype=float), index=index)
    frame = pd.DataFrame(
        {"actual": actual, "reference": actual + 2.0}, index=index
    )
    candidate = actual + 0.5
    storm = actual + 1.0
    result = screen._storm_statistics(
        frame, candidate, storm, timezone="Europe/Berlin"
    )
    assert result["role"] == "evaluation_only"
    assert result["used_by_model"] is False
    assert result["used_by_family_selection"] is False
    assert result["used_by_b1_or_b2_gate"] is False
    assert result["mae"]["daily_win_rate"] == 1.0
    assert result["rmse"]["daily_win_rate"] == 1.0
    assert result["abs_bias"]["daily_win_rate"] == 1.0


@pytest.mark.parametrize("zone", tuple(screen.ZONE_SPECS))
def test_real_frozen_autonomous_references_are_checksum_verified(zone: str) -> None:
    audit = screen._verify_frozen_reference(
        zone, screen._default_run_dir(zone), screen._default_recipe_path(zone)
    )
    assert audit["zone"] == zone
    assert audit["storm_used_as_feature"] is False
    assert audit["reference_prediction"].startswith("reconstructed frozen")


def test_inventory_is_research_only_and_reports_current_gaps() -> None:
    inventory = screen._inventory()
    assert inventory["production_changed"] is False
    assert inventory["shared"]["continental_revision_panel_ready"] is True
    assert inventory["shared"]["eco2mix"]["promotable"] is False
    assert inventory["shared"]["nonstorm_price_experts"][
        "compatible_with_autonomous_only"
    ] is False
    assert inventory["shared"]["storm"]["feature_use"] is False
    assert all(
        details["frozen_reference"] == "verified"
        for details in inventory["zones"].values()
    )


def test_b2_recipe_must_match_zone_and_have_negative_storm_proof(tmp_path: Path) -> None:
    recipe = {
        "protocol": {
            "phase": "b1_selection",
            "zone": "BE",
            "B2_rows_loaded": 0,
            "final_rows_loaded": 0,
            "storm_used_as_feature": False,
        },
        "selection_passed": True,
        "frozen_recipe": {"family": "local_revision"},
    }
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe), encoding="utf-8")
    with pytest.raises(ValueError, match="same-zone"):
        screen._load_b1_recipe(path, zone="DE")

    recipe["protocol"]["zone"] = "DE"
    recipe["protocol"]["storm_used_as_feature"] = True
    path.write_text(json.dumps(recipe), encoding="utf-8")
    with pytest.raises(ValueError, match="Storm"):
        screen._load_b1_recipe(path, zone="DE")


def _daily_fundamental_fixture(
    tmp_path: Path,
    *,
    bad_cutoff: bool = False,
) -> tuple[Path, Path, pd.DatetimeIndex]:
    timezone = "Europe/Brussels"
    expected = screen._expected_index(
        "2024-10-26", "2024-10-29", timezone
    )
    local_days = pd.date_range(
        "2024-10-26", "2024-10-28", freq="D", tz=timezone
    )
    cutoffs = pd.DatetimeIndex(
        [
            (day.tz_localize(None) - pd.Timedelta(days=1) + pd.Timedelta(hours=8))
            .tz_localize("Europe/Paris")
            .tz_convert("UTC")
            for day in local_days
        ]
    )
    if bad_cutoff:
        cutoffs = cutoffs + pd.Timedelta(hours=1)
    frame = pd.DataFrame(
        {
            "value_time_utc": local_days.tz_convert("UTC"),
            "snapshot_time_utc": cutoffs,
            "revision_time_utc": cutoffs,
            "value": [3.0, 2.5, 2.0],
        }
    )
    parquet = tmp_path / "daily.parquet"
    frame.to_parquet(parquet, index=False)
    manifest = {
        "zone": "BE",
        "timezone": timezone,
        "output_sha256": screen._sha256(parquet),
        "causality": {
            "classification": "strict_daily_asof_d_minus_1_08_europe_paris",
            "strict_pit_eligible": True,
            "storm_used": False,
            "target_or_price_used": False,
            "cutoff_violations": 0,
        },
        "causality_contract": {
            "delivery_timezone": timezone,
            "cutoff_timezone": "Europe/Paris",
            "cutoff_local_time": "08:00",
        },
        "feature_contract": {
            "kind": "fundamental_daily_local_midnight_broadcast",
            "feature_alias": "be_nuclear_availability_gw",
            "source_value_column": "value",
            "broadcast_rule": "repeat over physical delivery hours",
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return parquet, manifest_path, expected


def test_daily_fundamental_broadcast_preserves_25_hour_day(tmp_path: Path) -> None:
    parquet, manifest, expected = _daily_fundamental_fixture(tmp_path)
    frame, audit = screen._read_extra_exogenous(
        parquet,
        manifest,
        expected,
        zone="BE",
        timezone="Europe/Brussels",
    )
    assert frame.index.equals(expected)
    assert len(frame) == 73
    assert frame.loc[
        frame.index.tz_convert("Europe/Brussels").date
        == pd.Timestamp("2024-10-27").date()
    ].shape[0] == 25
    assert audit["feature_kind"] == "fundamental"
    assert audit["source_rows_loaded"] == 3
    assert audit["rows_loaded"] == 73


def test_daily_fundamental_rejects_snapshot_after_declared_cutoff(
    tmp_path: Path,
) -> None:
    parquet, manifest, expected = _daily_fundamental_fixture(
        tmp_path, bad_cutoff=True
    )
    with pytest.raises(ValueError, match="snapshot differs"):
        screen._read_extra_exogenous(
            parquet,
            manifest,
            expected,
            zone="BE",
            timezone="Europe/Brussels",
        )
