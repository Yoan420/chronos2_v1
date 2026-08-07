#!/usr/bin/env python
from __future__ import annotations

import run_chronos2_extended_exogenous as extended

from chronos2_structural_market.alignment import (
    make_prepare_zone_data_with_structural_alignment,
)
from chronos2_structural_market.residual import (
    make_prepare_zone_data_for_residual,
    make_run_backtest_variant_for_residual,
    make_run_live_forecast_variant_for_residual,
)


_aligned_prepare = make_prepare_zone_data_with_structural_alignment(
    extended.runner.prepare_zone_data
)
extended.runner.prepare_zone_data = make_prepare_zone_data_for_residual(
    _aligned_prepare
)

extended.runner.run_backtest_variant = make_run_backtest_variant_for_residual(
    extended.runner.run_backtest_variant
)
extended.runner.run_live_forecast_variant = (
    make_run_live_forecast_variant_for_residual(
        extended.runner.run_live_forecast_variant
    )
)


_base_run_zone = extended.runner.run_zone


def _run_zone_and_restore_price_target(*args, **kwargs):
    result = _base_run_zone(*args, **kwargs)
    data = result.zone_data
    if hasattr(data, "price_target_original"):
        data.target = data.price_target_original.copy()
        data.target.name = "target"
    return result


extended.runner.run_zone = _run_zone_and_restore_price_target


if __name__ == "__main__":
    raise SystemExit(extended.runner.main())
