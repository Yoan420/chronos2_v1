#!/usr/bin/env python
from __future__ import annotations

# Réutilise exactement le runner résiduel C3.
# Le YAML C4 change seulement residual_price_alias vers
# milp_structural_price_adjusted.
import run_chronos2_structural_residual as residual_runner


if __name__ == "__main__":
    raise SystemExit(
        residual_runner.extended.runner.main()
    )
