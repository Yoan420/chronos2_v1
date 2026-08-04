#!/usr/bin/env python
from __future__ import annotations

"""Runner unifié : correction oracle live + régimes optionnels + Chronos-2."""

import chronos2_modular.forecasting as forecasting

from chronos2_order_signals.chronos_compat import future_proxy_frame_fixed

# Cette affectation doit précéder l'import du module regime : celui-ci capture
# future_proxy_frame comme fonction de base au moment de son import.
forecasting.future_proxy_frame = future_proxy_frame_fixed

import run_chronos2_modular as runner  # noqa: E402

try:
    from chronos2_modular.regime import (  # noqa: E402
        future_proxy_frame_with_regime,
        prepare_zone_data_with_regime,
    )
except ImportError:
    pass
else:
    runner.prepare_zone_data = prepare_zone_data_with_regime
    forecasting.future_proxy_frame = future_proxy_frame_with_regime


if __name__ == "__main__":
    raise SystemExit(runner.main())
