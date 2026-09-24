"""Recognise the existing float32/CSV observation representation contract.

This is not a general tolerance relaxation: changed observations are refused
unless the two finite curves have the same float32 representation AND satisfy
the historical replay's absolute 5e-5 EUR/MWh ceiling. Nothing is modified.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


EXACT_OBSERVATION_ATOL = 1e-9
FLOAT32_REPLAY_ATOL = 5e-5


def validate_observation_precision(frozen: Any, canonical: Any, *, name: str = "observations") -> dict[str, Any]:
    """Validate exact or explicitly bounded float32 round-trip equivalence."""
    if isinstance(frozen, pd.Series) and isinstance(canonical, pd.Series) and not frozen.index.equals(canonical.index):
        raise ValueError(f"{name}: les heures des observations sont differentes.")
    left = np.asarray(frozen, dtype=np.float64)
    right = np.asarray(canonical, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape or not len(left):
        raise ValueError(f"{name}: deux courbes horaires non vides de meme taille sont requises.")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError(f"{name}: les observations comparees doivent etre finies.")
    differences = np.abs(left - right)
    maximum = float(differences.max())
    exact = bool((differences <= EXACT_OBSERVATION_ATOL).all())
    with np.errstate(over="ignore", invalid="ignore"):
        left32, right32 = left.astype(np.float32), right.astype(np.float32)
    roundtrip = (np.isfinite(left32).all() and np.isfinite(right32).all()
                 and np.array_equal(left32, right32) and maximum <= FLOAT32_REPLAY_ATOL)
    if not exact and not roundtrip:
        changed = int((differences > EXACT_OBSERVATION_ATOL).sum())
        raise ValueError(
            f"{name}: observations divergentes sur {changed}/{len(left)} heures; "
            f"ecart max={maximum:.12g} EUR/MWh, hors equivalence float32 exacte "
            f"bornee a {FLOAT32_REPLAY_ATOL:g} EUR/MWh."
        )
    return {
        "mode": "exact" if exact else "float32_roundtrip",
        "compared_hours": len(left),
        "max_absolute_difference_eur_mwh": maximum,
        "hours_above_exact_tolerance": int((differences > EXACT_OBSERVATION_ATOL).sum()),
        "exact_tolerance_eur_mwh": EXACT_OBSERVATION_ATOL,
        "float32_roundtrip_ceiling_eur_mwh": FLOAT32_REPLAY_ATOL,
        "float32_representations_equal": bool(np.array_equal(left32, right32)),
        "inputs_modified": False,
    }


__all__ = ["validate_observation_precision"]
