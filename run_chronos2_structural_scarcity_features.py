#!/usr/bin/env python
from __future__ import annotations

# C5 réutilise exactement le runner résiduel C3.
# Le YAML conserve milp_structural_price comme ancre et ajoute seulement
# les features scarcity comme covariables known-future.
import run_chronos2_structural_residual as residual_runner


if __name__ == "__main__":
    raise SystemExit(
        residual_runner.extended.runner.main()
    )
