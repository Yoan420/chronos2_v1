#!/usr/bin/env python
from __future__ import annotations

import chronos2_order_signals.pipeline as pipeline
from chronos2_modular.data import prepare_zone_data as base_prepare_zone_data
from chronos2_modular.exogenous_extensions import (
    make_prepare_zone_data_with_extensions,
)

pipeline.prepare_zone_data = make_prepare_zone_data_with_extensions(
    base_prepare_zone_data
)

from build_order_signal_vintages import main


if __name__ == "__main__":
    raise SystemExit(main())
