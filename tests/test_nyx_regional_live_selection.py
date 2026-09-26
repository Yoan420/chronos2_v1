import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.nyx_regional_live_selection import select_country


def test_fixed_country_arithmetic_and_exact_reference_copy():
    index = pd.date_range("2026-09-24", periods=4, freq="h", tz="UTC")
    fr = pd.DataFrame({"reference": [10., 10., 10., 10.],
                       "residual": [29.99, 30., -10., 9.]}, index=index)
    assert select_country("FR", fr).tolist() == [10., 30., -10., 10.]
    pair = pd.DataFrame({"reference": [10., 10., 10., 10.],
                         "compact": [20., 20., -30., 12.],
                         "exchange_absolute": [39.98, 40., 10., 14.]}, index=index)
    assert select_country("BE", pair).tolist() == [10., 30., -10., 10.]
    assert select_country("NL", pair).tolist() == [29.99, 30., -10., 13.]


def test_rejects_wrong_country_schema_and_nonfinite_inputs():
    index = pd.date_range("2026-09-24", periods=2, freq="h", tz="UTC")
    points = pd.DataFrame({"reference": [1., 2.], "residual": [3., 4.]}, index=index)
    with pytest.raises(ValueError):
        select_country("DE", points)
    with pytest.raises(ValueError):
        select_country("FR", points.rename(columns={"residual": "storm"}))
    with pytest.raises(ValueError):
        select_country("FR", points.set_axis(index[::-1]))
    points.iloc[0, 1] = np.nan
    with pytest.raises(ValueError):
        select_country("FR", points)
