from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from run_mkonline_blend_hourly import (
    EXPECTED_DAY_HISTOGRAM,
    WEIGHT_MK,
    _blend_quantiles,
    _complete_local_range,
    _day_histogram,
    _expected_cutoff,
    _load_primary,
    _load_storm_evaluation_only,
    _validate_da_target_availability,
)


def test_frozen_blend_common_shift_preserves_width_and_order() -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    autonomous = pd.DataFrame(
        {"q10": [0.0, 1.0, 2.0], "q50": [10.0, 11.0, 12.0], "q90": [30.0, 31.0, 32.0]},
        index=index,
    )
    mk = pd.Series([20.0, 21.0, 22.0], index=index)
    result = _blend_quantiles(autonomous, mk)
    expected = (1.0 - WEIGHT_MK) * autonomous["q50"] + WEIGHT_MK * mk
    np.testing.assert_allclose(result["q50"], expected)
    np.testing.assert_allclose(result["q90"] - result["q10"], autonomous["q90"] - autonomous["q10"])
    assert bool((result["q10"] <= result["q50"]).all())
    assert bool((result["q50"] <= result["q90"]).all())
    with pytest.raises(ValueError, match="frozen"):
        _blend_quantiles(autonomous, mk, weight=0.5)


def test_final_timeline_and_civil_cutoff_are_dst_safe() -> None:
    index = _complete_local_range("2025-08-12", "2026-08-11")
    assert len(index) == 8760
    assert _day_histogram(index) == EXPECTED_DAY_HISTOGRAM
    for day, expected_hours in (("2025-10-26", 25), ("2026-03-29", 23)):
        selected = index[pd.Index(index.tz_convert("Europe/Paris").date) == pd.Timestamp(day).date()]
        assert len(selected) == expected_hours
        cutoff = _expected_cutoff(selected)
        expected_local = pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
        assert cutoff[0].tz_convert("Europe/Paris").strftime("%Y-%m-%d %H:%M") == expected_local.strftime("%Y-%m-%d %H:%M")


def test_primary_loader_fails_closed_on_wrong_cutoff(tmp_path: Path) -> None:
    index = _complete_local_range("2025-01-02", "2025-01-02")
    cutoff = _expected_cutoff(index)
    frame = pd.DataFrame({"value_time_utc": index, "snapshot_time_utc": cutoff, "revision_time_utc": cutoff, "value": np.arange(len(index), dtype=float)})
    good = tmp_path / "good.parquet"
    frame.to_parquet(good, index=False)
    values, _, audit = _load_primary(good, expected_index=index, expected_histogram={24: 1})
    assert len(values) == 24 and audit["coverage"] == 1.0
    frame.loc[0, "snapshot_time_utc"] = cutoff[0] + pd.Timedelta(hours=1)
    bad = tmp_path / "bad.parquet"
    frame.to_parquet(bad, index=False)
    with pytest.raises(ValueError, match="PIT marker"):
        _load_primary(bad, expected_index=index, expected_histogram={24: 1})


def test_primary_loader_rejects_missing_hour(tmp_path: Path) -> None:
    index = _complete_local_range("2025-01-02", "2025-01-02")
    cutoff = _expected_cutoff(index)
    frame = pd.DataFrame({"value_time_utc": index[:-1], "snapshot_time_utc": cutoff[:-1], "revision_time_utc": cutoff[:-1], "value": 1.0})
    path = tmp_path / "missing.parquet"
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="timeline"):
        _load_primary(path, expected_index=index, expected_histogram={24: 1})


def test_storm_evaluation_loader_selects_last_eligible_vintage(tmp_path: Path) -> None:
    index = _complete_local_range("2025-01-02", "2025-01-02")
    cutoff = _expected_cutoff(index)
    vintages = []
    for delta, value in ((-2, 1.0), (-1, 2.0), (1, 999.0)):
        marker = cutoff + pd.Timedelta(hours=delta)
        vintages.append(
            pd.DataFrame(
                {
                    "value_time_utc": index,
                    "snapshot_time_utc": marker,
                    "revision_time_utc": marker,
                    "value": value,
                }
            )
        )
    path = tmp_path / "storm.parquet"
    pd.concat(vintages, ignore_index=True).to_parquet(path, index=False)

    values, selected, audit = _load_storm_evaluation_only(
        path,
        expected_index=index,
    )

    np.testing.assert_allclose(values, 2.0)
    assert len(selected) == 24
    assert audit["used_for_prediction"] is False
    assert audit["used_for_live_forecast"] is False


def test_da_target_availability_contract_accepts_previous_delivery_day() -> None:
    live = _complete_local_range("2026-08-12", "2026-08-12")
    audit = _validate_da_target_availability(
        training_end_utc="2026-08-11 21:00:00+00:00",
        forecast_index=live,
    )
    assert audit["training_end_delivery_day_local"] == "2026-08-11"
    with pytest.raises(ValueError, match="not available"):
        _validate_da_target_availability(
            training_end_utc="2026-08-12 21:00:00+00:00",
            forecast_index=live,
        )
