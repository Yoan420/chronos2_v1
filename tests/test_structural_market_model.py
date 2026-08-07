from __future__ import annotations

import numpy as np
import pandas as pd

from chronos2_structural_market.config import parse_structural_model_config
from chronos2_structural_market.model import solve_structural_day


def test_milp_then_fixed_lp_produces_structural_features():
    index = pd.date_range(
        "2026-01-15 00:00",
        periods=24,
        freq="h",
        tz="Europe/Paris",
    )
    load = np.concatenate(
        [
            np.full(8, 32.0),
            np.linspace(35.0, 58.0, 8),
            np.linspace(56.0, 38.0, 8),
        ]
    )
    frame = pd.DataFrame(
        {
            "residual_load_gw": load,
            "nuclear_capacity_gw": 40.0,
            "fixed_net_exports_gw": 0.0,
            "gas_price_eur_mwhth": np.nan,
            "coal_price_eur_mwhth": np.nan,
            "eua_price_eur_t": np.nan,
        },
        index=index,
    )
    config = parse_structural_model_config({})
    result = solve_structural_day(frame, config)

    required = {
        "milp_structural_price",
        "milp_reserve_margin_gw",
        "milp_committed_thermal_gw",
        "milp_online_units",
        "milp_startups",
        "milp_scarcity_gw",
        "milp_ramp_shadow_eur_mwh",
        "milp_reserve_shadow_eur_mwh",
    }
    assert required.issubset(result.features.columns)
    assert len(result.features) == 24
    assert np.isfinite(result.features["milp_structural_price"]).all()
    assert result.features["milp_scarcity_gw"].max() < 1e-6
    assert result.diagnostics["success"] is True
    assert result.diagnostics["mip_node_count"] >= 0

    low_price = result.features["milp_structural_price"].iloc[:8].median()
    peak_price = result.features["milp_structural_price"].iloc[10:17].median()
    assert peak_price >= low_price
