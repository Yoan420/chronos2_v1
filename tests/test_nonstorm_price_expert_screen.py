from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from runs.tmp.screen_nonstorm_price_experts_extended import (
    _expected_cutoff,
    _load_dependency_audit,
    _read_expert,
    _score,
)


def test_expected_cutoff_is_civil_d_minus_1_08_across_dst() -> None:
    delivery = pd.DatetimeIndex(
        ["2024-03-31T00:00:00Z", "2024-10-27T00:00:00Z"]
    )
    expected = pd.DatetimeIndex(
        ["2024-03-30T07:00:00Z", "2024-10-26T06:00:00Z"]
    )
    assert _expected_cutoff(delivery).equals(expected)


def test_native_vintage_schema_is_loaded_and_cutoff_checked() -> None:
    delivery = pd.date_range("2024-01-02T00:00:00Z", periods=2, freq="h")
    cutoff = _expected_cutoff(delivery)
    frame = pd.DataFrame(
        {
            "value_time_utc": delivery,
            "snapshot_time_utc": cutoff,
            "revision_time_utc": cutoff,
            "value": [10.0, 20.0],
        }
    )
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "pointcarbon.parquet"
        frame.to_parquet(path, index=False)
        values, audit = _read_expert(path, "pointcarbon_flowspot", delivery)
    np.testing.assert_allclose(values.to_numpy(), [10.0, 20.0])
    assert audit["coverage"] == 1.0
    assert audit["cutoff_violations"] == 0


def test_storm_guard_rejects_any_storm_column() -> None:
    delivery = pd.date_range("2024-01-02T00:00:00Z", periods=1, freq="h")
    cutoff = _expected_cutoff(delivery)
    frame = pd.DataFrame(
        {
            "value_time_utc": delivery,
            "snapshot_time_utc": cutoff,
            "revision_time_utc": cutoff,
            "value": [10.0],
            "storm_shadow": [11.0],
        }
    )
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "safe.parquet"
        frame.to_parquet(path, index=False)
        with pytest.raises(ValueError, match="Storm guard"):
            _read_expert(path, "pointcarbon_flowspot", delivery)


def test_selection_requires_threshold_aggregate_and_positive_halves() -> None:
    index = pd.date_range("2025-04-13T22:00:00Z", periods=60 * 24, freq="h")
    actual = np.zeros(len(index))
    baseline = np.ones(len(index))
    candidate = np.concatenate(
        [np.zeros(30 * 24), np.full(30 * 24, 0.273)]
    )
    score = _score(index, actual, baseline, candidate)
    assert score["all"]["gain"] > 0.75
    assert score["last30"]["gain"] < 0.75
    assert score["last30"]["gain"] > 0.0
    assert score["passes"] is True


def test_selection_rejects_a_negative_half_despite_aggregate_threshold() -> None:
    index = pd.date_range("2025-04-13T22:00:00Z", periods=60 * 24, freq="h")
    actual = np.zeros(len(index))
    baseline = np.concatenate(
        [np.full(30 * 24, 10.0), np.ones(30 * 24)]
    )
    candidate = np.concatenate(
        [np.zeros(30 * 24), np.full(30 * 24, 1.05)]
    )
    score = _score(index, actual, baseline, candidate)
    assert score["all"]["gain"] >= 0.75
    assert score["last30"]["gain"] < 0.0
    assert score["passes"] is False


def test_dependency_audit_requires_exact_series_and_clear_tokens() -> None:
    payload = {
        "series": {
            "power.price.fr.euromwh.h.fcst.pointcarbon.flowspot": {
                "storm_token_found": False
            },
            "power.price.fr.euromwh.h.fcst.mkonline.ecop": {
                "storm_token_found": False
            },
        },
        "overall": {
            "mkonline_no_storm_dependency_proven": True,
            "pointcarbon_no_saturn_storm_dependency_proven": True,
            "pointcarbon_external_identity_confirmed": False,
        },
    }
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "audit.json"
        import json

        path.write_text(json.dumps(payload), encoding="utf-8")
        result = _load_dependency_audit(path)
    assert result["pointcarbon_no_saturn_storm_dependency_proven"] is True
    assert result["pointcarbon_external_identity_confirmed"] is False
