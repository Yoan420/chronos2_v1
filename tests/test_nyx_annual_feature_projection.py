"""Exact annual CWE subset projection, including an optional archive parity."""
import pandas as pd
import pytest

from chronos2_hourly.nyx_annual_feature_projection import FULL, POOLED, COMPACT, project_country
from chronos2_hourly.nyx_annual_live_preflight import load_schema
from run_nyx_annual_feature_projection import HISTORICAL_ROOT, historical_parity


def test_all_selected_columns_are_in_the_503_source():
    schema = load_schema()
    for zone in ("FR", "DE", "BE", "NL"):
        full = set(schema["families"][FULL]["columns"][zone])
        pooled = set(schema["families"][POOLED]["columns"][zone])
        for family in (POOLED, COMPACT):
            assert set(schema["families"][family]["columns"][zone]) <= full
        extra = full - pooled
        assert len(extra) == 54
        assert sum(name.startswith("canon__") for name in extra) == 8
        assert sum(name.startswith("thermal__") for name in extra) == 36
        assert sum(name.startswith("exchange__") for name in extra) == 10


def test_projection_preserves_order_values_and_types():
    schema = load_schema()
    zone = "FR"
    index = pd.date_range("2026-09-28", periods=3, freq="h", tz="UTC")
    columns = schema["families"][FULL]["columns"][zone]
    frame = pd.DataFrame({name: 1.0 if name.endswith("__available") else 2.0
                          for name in columns}, index=index)
    # The shared JAO flag is a valid all-present state as well.
    pooled = frame.loc[:, schema["families"][POOLED]["columns"][zone]].copy()
    result = project_country(frame, pooled, zone=zone, expected_index=index, schema=schema)
    names = schema["families"][COMPACT]["columns"][zone]
    pd.testing.assert_frame_equal(result, frame.loc[:, names], check_exact=True)
    assert result.columns.tolist() == names
    with pytest.raises(ValueError, match="ordered annual feature schema"):
        project_country(frame.iloc[:, ::-1], pooled, zone=zone, expected_index=index, schema=schema)
    with pytest.raises(ValueError, match="ordered annual feature schema"):
        project_country(frame, pooled.iloc[:, ::-1], zone=zone, expected_index=index, schema=schema)
    pooled.iloc[0, pooled.columns.get_loc("price_be_d1_hour")] += 1.0
    with pytest.raises(AssertionError):
        project_country(frame, pooled, zone=zone, expected_index=index, schema=schema)
    pooled = frame.loc[:, schema["families"][POOLED]["columns"][zone]].copy()
    jao = [name for name in pooled if name.startswith("extra_jao_")]
    pooled.loc[index[0], jao] = float("nan")
    pooled.loc[index[0], "extra_jao__available"] = 0.0
    # Different JAO vintages are deliberate; the shared non-JAO values remain exact.
    project_country(frame, pooled, zone=zone, expected_index=index, schema=schema)


def test_historical_projection_is_bit_exact_when_archives_exist():
    source = HISTORICAL_ROOT / "pooled_jao_refresh_v1" / "features_FR.parquet"
    if not source.is_file():
        pytest.skip("Ignored local CWE feature archive is unavailable")
    report = historical_parity()
    assert report["passed"] is True
    assert len(report["checks"]) == 12
    assert all(report["checks"][f"{zone}/{COMPACT}"]["bit_exact_projection"] is True
               for zone in ("FR", "DE", "BE", "NL"))
    assert all(report["checks"][f"{zone}/{POOLED}"]["jao_difference_hours"] == 271
               for zone in ("FR", "DE", "BE", "NL"))
    assert all(report["checks"][f"{zone}/{POOLED}"]["jao_difference_cells"] == 7588
               for zone in ("FR", "DE", "BE", "NL"))
