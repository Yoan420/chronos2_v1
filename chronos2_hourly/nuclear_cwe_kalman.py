"""Connect the frozen CWE covariate schema to the operational Kalman API.

This boundary has its own audited delegate identity. Keeping it separate from
the frozen input adapter preserves valid Chronos and residual replay caches.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from . import kalman_residual


def build_nuclear_cwe_kalman_view(*, covariates: pd.DataFrame, **kwargs: Any):
    """Rename only the time column; retain every value and Kalman validation."""
    if not isinstance(covariates, pd.DataFrame) or covariates.empty:
        raise kalman_residual.KalmanResidualError("Covariables CWE non vides requises.")
    if covariates.columns.has_duplicates:
        raise kalman_residual.KalmanResidualError("Colonnes de covariables CWE dupliquees.")
    time_columns = [name for name in ("timestamp", "delivery_start_utc") if name in covariates]
    if len(time_columns) != 1:
        raise kalman_residual.KalmanResidualError(
            "Covariables CWE : une seule colonne timestamp ou delivery_start_utc est requise."
        )
    normalized = covariates.copy(deep=True).rename(columns={time_columns[0]: "timestamp"})
    cache_dir = kwargs.get("rolling_refit_cache_dir")
    if cache_dir is not None and os.name == "nt":
        # The delegate identity adds a directory. Its temporary cache files can
        # exceed MAX_PATH; the extended spelling addresses the same location.
        absolute = str(Path(cache_dir).expanduser().resolve())
        if not absolute.startswith("\\\\?\\"):
            absolute = ("\\\\?\\UNC\\" + absolute[2:] if absolute.startswith("\\\\")
                        else "\\\\?\\" + absolute)
        kwargs = dict(kwargs, rolling_refit_cache_dir=Path(absolute))
    return kalman_residual.build_operational_kalman_view(covariates=normalized, **kwargs)
