
#!/usr/bin/env python
from __future__ import annotations

# C6A = C3 résiduel + probabilités plateau comme covariables.
# Aucun soft gate n'est appliqué à ce stade.
import run_chronos2_structural_residual as residual_runner


if __name__ == "__main__":
    raise SystemExit(residual_runner.extended.runner.main())
