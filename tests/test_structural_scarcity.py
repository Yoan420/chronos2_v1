from __future__ import annotations

import numpy as np
import pandas as pd

from chronos2_structural_market.scarcity import (
    ScarcityParams,
    apply_scarcity_layer,
)


def _frame():
    return pd.DataFrame(
        {
            "milp_structural_price": [50.0, 75.0, 100.0],
            "milp_reserve_margin_gw": [8.0, 3.0, 0.5],
            "milp_ramp_shadow_eur_mwh": [0.0, 10.0, 50.0],
            "milp_startups": [0.0, 1.0, 3.0],
        }
    )


def test_adder_is_non_negative_and_price_identity_holds():
    params = ScarcityParams(
        alpha=100.0,
        tau_gw=2.0,
        beta_ramp=20.0,
        gamma_startup=15.0,
        ramp_scale=50.0,
        startup_scale=3.0,
    )
    result = apply_scarcity_layer(
        _frame(),
        params,
    )

    assert (
        result["milp_scarcity_adder"] >= 0
    ).all()
    assert np.allclose(
        result["milp_structural_price_adjusted"],
        result["milp_structural_price_raw"]
        + result["milp_scarcity_adder"],
    )


def test_scarcity_probability_rises_in_tighter_case():
    params = ScarcityParams(
        alpha=100.0,
        tau_gw=2.0,
        beta_ramp=20.0,
        gamma_startup=15.0,
        ramp_scale=50.0,
        startup_scale=3.0,
    )
    result = apply_scarcity_layer(
        _frame(),
        params,
    )

    assert (
        result["milp_scarcity_probability"].iloc[-1]
        >
        result["milp_scarcity_probability"].iloc[0]
    )
