#!/usr/bin/env python
from __future__ import annotations

"""Lance Chronos-2 avec les covariables de régime point-in-time."""

import chronos2_modular.forecasting as forecasting
import run_chronos2_modular as runner

from chronos2_modular.regime import (
    future_proxy_frame_with_regime,
    prepare_zone_data_with_regime,
)


# run_chronos2_modular importe directement prepare_zone_data :
# on remplace donc sa référence locale.
runner.prepare_zone_data = prepare_zone_data_with_regime

# build_origin_frames et build_live_frames cherchent future_proxy_frame dans
# le namespace du module forecasting au moment de leur exécution.
forecasting.future_proxy_frame = future_proxy_frame_with_regime


if __name__ == "__main__":
    raise SystemExit(runner.main())
