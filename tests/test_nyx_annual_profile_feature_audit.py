"""Recovered source revisions must not be described as historical vintages."""
import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.nyx_forecast_profile_features import build_forecast_profile_features


def test_profile_recovery_changes_provenance_without_changing_feature_math():
    index = pd.date_range("2026-09-28", periods=48, freq="h", tz="UTC")
    columns = [f"known__{z}_{s}_fcst" for z in ("fr", "de", "be", "nl")
               for s in ("residual_load", "solar_generation")]
    columns.append("known__fr_nuclear_generation_fcst_gw")
    source = pd.DataFrame({c: np.arange(len(index), dtype=float) for c in columns}, index=index)
    for c in columns:
        source[c + "__available"] = 1.
    original, _ = build_forecast_profile_features(source)
    contract = {"profile_history_policy": "own_origin_with_current_fit_recovery_v1",
                "profile_revision_ceiling_utc": "2026-09-29T06:00:00+00:00",
                "profile_origin_snapshot_verified": False}
    recovered, audit = build_forecast_profile_features(source, profile_history_contract=contract)
    pd.testing.assert_frame_equal(recovered, original, check_exact=True)
    assert {key: audit[key] for key in contract} == contract
    assert "logical reconstruction" in audit["source"]
    assert "own D-1" not in audit["source"]
    with pytest.raises(ValueError):
        build_forecast_profile_features(source, profile_history_contract={
            **contract, "profile_origin_snapshot_verified": True})
