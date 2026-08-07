#!/usr/bin/env python
from __future__ import annotations

import chronos2_modular.forecasting as forecasting
import run_chronos2_extended_exogenous as extended

from chronos2_modular.cutoff_persistence import (
    make_future_proxy_frame_with_cutoff_persistence,
    make_prepare_zone_data_with_cutoff_persistence,
)


extended.runner.prepare_zone_data = (
    make_prepare_zone_data_with_cutoff_persistence(
        extended.runner.prepare_zone_data
    )
)

forecasting.future_proxy_frame = (
    make_future_proxy_frame_with_cutoff_persistence(
        forecasting.future_proxy_frame
    )
)


if __name__ == "__main__":
    raise SystemExit(extended.runner.main())
