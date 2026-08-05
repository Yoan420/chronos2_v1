#!/usr/bin/env python
from __future__ import annotations

import chronos2_modular.forecasting as forecasting
from chronos2_order_signals.chronos_compat import future_proxy_frame_fixed

forecasting.future_proxy_frame = future_proxy_frame_fixed

import run_chronos2_modular as runner  # noqa: E402

try:
    from chronos2_modular.regime import (
        future_proxy_frame_with_regime,
        prepare_zone_data_with_regime,
    )
except ImportError:
    pass
else:
    runner.prepare_zone_data = prepare_zone_data_with_regime
    forecasting.future_proxy_frame = future_proxy_frame_with_regime

from chronos2_modular.exogenous_extensions import (  # noqa: E402
    make_prepare_zone_data_with_extensions,
)
from chronos2_modular.calendar_interactions import (  # noqa: E402
    make_prepare_zone_data_with_calendar_interactions,
)

runner.prepare_zone_data = make_prepare_zone_data_with_extensions(
    runner.prepare_zone_data
)
runner.prepare_zone_data = (
    make_prepare_zone_data_with_calendar_interactions(
        runner.prepare_zone_data
    )
)

if __name__ == "__main__":
    raise SystemExit(runner.main())
