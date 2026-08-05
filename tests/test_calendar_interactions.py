from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from chronos2_modular.calendar_interactions import (
    add_calendar_interaction_features,
)


def test_calendar_interactions_are_correct(tmp_path):
    historical_index = pd.date_range(
        "2026-01-01 00:00",
        periods=24,
        freq="h",
        tz="Europe/Paris",
    )
    future_index = pd.date_range(
        historical_index[-1] + pd.Timedelta(hours=1),
        periods=24,
        freq="h",
        tz="Europe/Paris",
    )
    index = historical_index.union(future_index)

    frame = pd.DataFrame(index=index)
    frame["known_fr_residual_load_fcst_oracle"] = 50.0
    frame["known_fr_nuclear_generation_fcst_oracle"] = 40.0
    frame["known_cal_morning_peak_oracle"] = (
        index.hour == 8
    ).astype(float)
    frame["known_cal_evening_peak_oracle"] = (
        index.hour == 18
    ).astype(float)
    frame["known_cal_holiday_fr_oracle"] = 1.0
    frame["known_is_weekend"] = 0.0
    frame["known_cal_bridge_day_fr_oracle"] = 0.0
    frame["known_cal_holiday_de_oracle"] = 1.0
    frame["known_cal_holiday_be_oracle"] = 0.0
    frame["known_cal_holiday_nl_oracle"] = 1.0
    frame["known_cal_holiday_es_oracle"] = 0.0

    data = SimpleNamespace(
        zone="FR",
        target=pd.Series(0.0, index=historical_index),
        model_context_covariates=frame,
        known_future_columns=[],
        input_manifest=pd.DataFrame(),
        diagnostics={},
    )
    config = {
        "exogenous_extensions": {
            "calendar_interactions": {
                "enabled": True,
                "neighbour_countries": [
                    "DE", "BE", "NL", "ES"
                ],
            }
        }
    }

    columns = add_calendar_interaction_features(
        data,
        config,
        tmp_path,
    )

    assert len(columns) == 10
    assert np.allclose(
        data.model_context_covariates[
            "known_residual_load_after_nuclear_oracle"
        ],
        10.0,
    )
    assert np.allclose(
        data.model_context_covariates[
            (
                "known_interaction_residual_load_"
                "neighbour_holiday_count_oracle"
            )
        ],
        100.0,
    )

    eight_am = index.hour == 8
    assert np.allclose(
        data.model_context_covariates.loc[
            eight_am,
            (
                "known_interaction_residual_after_nuclear_"
                "morning_peak_oracle"
            ),
        ],
        10.0,
    )
    assert set(columns).issubset(
        set(data.known_future_columns)
    )
