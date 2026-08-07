from __future__ import annotations

import numpy as np
import pandas as pd

from chronos2_structural_market.scarcity_features import (
    ScarcityFeatureParams,
    build_scarcity_feature_frame,
)


def _params():
    return ScarcityFeatureParams(
        alpha=100.0,
        tau_gw=2.0,
        beta_ramp=20.0,
        gamma_startup=15.0,
        ramp_scale=50.0,
        startup_scale=3.0,
    )


def test_features_have_expected_columns_and_no_future_lookahead():
    frame = pd.DataFrame(
        {
            "milp_reserve_margin_gw": [
                8.0,
                5.0,
                2.0,
                0.5,
            ],
            "milp_ramp_shadow_eur_mwh": [
                0.0,
                5.0,
                20.0,
                50.0,
            ],
            "milp_startups": [
                0.0,
                0.0,
                1.0,
                3.0,
            ],
        },
        index=pd.date_range(
            "2026-08-08",
            periods=4,
            freq="h",
            tz="Europe/Paris",
        ),
    )

    result = build_scarcity_feature_frame(
        frame,
        _params(),
        ewm_alpha=0.65,
    )

    expected = {
        "milp_reserve_pressure",
        "milp_ramp_pressure",
        "milp_startup_pressure",
        "milp_scarcity_probability_feature",
        "milp_scarcity_adder_feature",
        "milp_scarcity_probability_ewm",
        "milp_scarcity_adder_ewm",
        "milp_scarcity_delta",
        "milp_reserve_pressure_delta",
    }
    assert set(result.columns) == expected
    assert result.notna().all().all()

    # Le signal de rareté doit être plus fort dans le cas le plus tendu.
    assert (
        result[
            "milp_scarcity_probability_feature"
        ].iloc[-1]
        >
        result[
            "milp_scarcity_probability_feature"
        ].iloc[0]
    )

    # Les deltas ne regardent que t et t-1.
    expected_delta = (
        result[
            "milp_scarcity_probability_feature"
        ].iloc[2]
        -
        result[
            "milp_scarcity_probability_feature"
        ].iloc[1]
    )
    assert np.isclose(
        result[
            "milp_scarcity_delta"
        ].iloc[2],
        expected_delta,
    )


def test_ewm_reduces_one_hour_spike():
    frame = pd.DataFrame(
        {
            "milp_reserve_margin_gw": [
                8.0,
                8.0,
                0.1,
                8.0,
            ],
            "milp_ramp_shadow_eur_mwh": [
                0.0,
                0.0,
                100.0,
                0.0,
            ],
            "milp_startups": [
                0.0,
                0.0,
                3.0,
                0.0,
            ],
        }
    )

    result = build_scarcity_feature_frame(
        frame,
        _params(),
        ewm_alpha=0.65,
    )

    raw_jump = abs(
        result[
            "milp_scarcity_adder_feature"
        ].iloc[2]
        -
        result[
            "milp_scarcity_adder_feature"
        ].iloc[1]
    )
    smooth_jump = abs(
        result[
            "milp_scarcity_adder_ewm"
        ].iloc[2]
        -
        result[
            "milp_scarcity_adder_ewm"
        ].iloc[1]
    )

    assert smooth_jump < raw_jump
