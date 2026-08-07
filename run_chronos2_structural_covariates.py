#!/usr/bin/env python
from __future__ import annotations

import run_chronos2_extended_exogenous as extended

from chronos2_structural_market.alignment import (
    make_prepare_zone_data_with_structural_alignment,
)


extended.runner.prepare_zone_data = (
    make_prepare_zone_data_with_structural_alignment(
        extended.runner.prepare_zone_data
    )
)


if __name__ == "__main__":
    raise SystemExit(extended.runner.main())
