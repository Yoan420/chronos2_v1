from __future__ import annotations

from pathlib import Path

import pandas as pd

from chronos2_modular.common import ZoneData
from chronos2_modular.feature_selection import filter_zone_data


def test_filter_keeps_required_lag_alias(tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="h", tz="Europe/Paris")
    data = ZoneData(
        zone="FR",
        timezone="Europe/Paris",
        frequency="h",
        target=pd.Series([1.0, 2.0, 3.0, 4.0], index=index),
        covariates=pd.DataFrame(
            {
                "de_price_da": [10.0, 11.0, 12.0, 13.0],
                "fr_residual_load_fcst": [20.0, 21.0, 22.0, 23.0],
            },
            index=index,
        ),
        model_context_covariates=pd.DataFrame(
            {
                "de_price_da": [10.0, 11.0, 12.0, 13.0],
                "known_de_price_da_lag24": [1.0, 2.0, 3.0, 4.0],
                "fr_residual_load_fcst": [20.0, 21.0, 22.0, 23.0],
                "known_fr_residual_load_fcst_oracle": [20.0, 21.0, 22.0, 23.0],
                "known_hour_sin": [0.0, 0.1, 0.2, 0.3],
            },
            index=index,
        ),
        known_future_columns=[
            "known_de_price_da_lag24",
            "known_fr_residual_load_fcst_oracle",
            "known_hour_sin",
        ],
        coverage=pd.DataFrame(),
        input_manifest=pd.DataFrame(),
        diagnostics={},
    )
    config = {
        "feature_selection": {
            "enabled": True,
            "selected_groups": ["prices"],
            "group_definitions": {
                "prices": {"patterns": ["*de_price_da*"]},
                "calendar": {"patterns": ["known_hour_*"]},
            },
        }
    }

    result = filter_zone_data(data, config, tmp_path)

    assert list(result.model_context_covariates.columns) == [
        "de_price_da",
        "known_de_price_da_lag24",
    ]
    assert list(result.covariates.columns) == ["de_price_da"]
    assert result.known_future_columns == ["known_de_price_da_lag24"]


def test_empty_selection_removes_all_exogenous(tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01", periods=2, freq="h", tz="Europe/Paris")
    data = ZoneData(
        zone="FR",
        timezone="Europe/Paris",
        frequency="h",
        target=pd.Series([1.0, 2.0], index=index),
        covariates=pd.DataFrame({"x": [1.0, 2.0]}, index=index),
        model_context_covariates=pd.DataFrame(
            {"known_x_oracle": [1.0, 2.0]}, index=index
        ),
        known_future_columns=["known_x_oracle"],
        coverage=pd.DataFrame(),
        input_manifest=pd.DataFrame(),
        diagnostics={},
    )
    config = {
        "feature_selection": {
            "enabled": True,
            "selected_groups": [],
            "group_definitions": {"x": {"patterns": ["*x*"]}},
        }
    }

    result = filter_zone_data(data, config, tmp_path)
    assert result.model_context_covariates.empty
    assert result.covariates.empty
    assert result.known_future_columns == []
