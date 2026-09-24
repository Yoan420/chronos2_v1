"""A round-trip guard, not a general market-price tolerance relaxation."""
import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.observation_precision import validate_observation_precision


def test_exact_values_have_an_explicit_audit():
    audit = validate_observation_precision([10.0, -5.0], [10.0 + 1e-12, -5.0])
    assert audit["mode"] == "exact"
    assert audit["compared_hours"] == 2
    assert audit["hours_above_exact_tolerance"] == 0


def test_real_float32_csv_roundtrip_is_accepted_without_mutation():
    canonical = np.array([338.8625, -412.55, -496.8625, 313.355, 260.5425])
    frozen = canonical.astype(np.float32)
    before = frozen.copy(), canonical.copy()
    audit = validate_observation_precision(frozen, canonical, name="FINAL365")
    assert audit["mode"] == "float32_roundtrip"
    assert audit["compared_hours"] == 5
    assert audit["float32_representations_equal"] is True
    assert audit["hours_above_exact_tolerance"] == 5
    assert audit["max_absolute_difference_eur_mwh"] == pytest.approx(1.220703126e-5)
    assert np.array_equal(frozen, before[0]) and np.array_equal(canonical, before[1])


@pytest.mark.parametrize("frozen,canonical", [
    ([100.0], [100.01]),                       # A real price change.
    ([10.0], [10.0 + 1e-6]),                 # Small, but not the same float32 value.
    ([10000.0], [10000.0 + 1e-4]),           # Same float32, but exceeds the replay ceiling.
    ([float("nan")], [float("nan")]),
    ([float("inf")], [float("inf")]),
    ([], []), ([1.0], [1.0, 2.0]), ([[1.0]], [[1.0]]),
])
def test_invalid_or_really_different_values_fail_closed(frozen, canonical):
    with pytest.raises(ValueError):
        validate_observation_precision(frozen, canonical)


def test_hourly_series_must_share_exact_physical_timestamps():
    index = pd.date_range("2026-10-25T00:00:00Z", periods=2, freq="h")
    with pytest.raises(ValueError, match="heures"):
        validate_observation_precision(pd.Series([1.0, 2.0], index=index),
                                       pd.Series([1.0, 2.0], index=index + pd.Timedelta(hours=1)))
