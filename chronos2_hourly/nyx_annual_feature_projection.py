"""Project the selected annual CWE 503 matrix to its exact compact subset.

This is a pure column selection.  The caller owns the source's prospective
collection, as-of evidence, and complete training window.  No archived cache
is read here and no absent value is invented.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .nyx_annual_live_preflight import ZONES, load_schema, validate_feature_frame


FULL = "cwe_absolute_2000"
POOLED = "fr_residual_1000"
COMPACT = "cwe_residual_2000"


def project_country(
    full503: pd.DataFrame, pooled449: pd.DataFrame, *, zone: str,
    expected_index: pd.DatetimeIndex,
    schema: Mapping | None = None,
) -> pd.DataFrame:
    """Validate supplied 503 and 449 frames and return the ordered 123 subset.

    The 449 matrix is NOT projected from the 503 matrix: the selected
    historical recipes have different JAO vintages for 271 late hours.  Their
    other 421 shared columns must be exactly equal.  ``expected_index`` is
    the prospective UTC training plus delivery grid, or the sealed historical
    grid when auditing archival parity.
    """
    if zone not in ZONES:
        raise ValueError(f"Unsupported annual CWE country: {zone}")
    schema = load_schema() if schema is None else schema
    families = schema["families"]
    full_columns = families[FULL]["columns"][zone]
    validate_feature_frame(full503, full_columns, expected_index, f"{zone}/503")
    pooled_columns = families[POOLED]["columns"][zone]
    validate_feature_frame(pooled449, pooled_columns, expected_index, f"{zone}/449")
    other = [name for name in pooled_columns if not name.startswith("extra_jao_")]
    if len(other) != 421:
        raise ValueError(f"{zone}: expected 421 shared non-JAO columns")
    pd.testing.assert_frame_equal(full503.loc[:, other], pooled449.loc[:, other],
                                  check_exact=True, check_dtype=True)
    for name in other:
        left, right = full503[name].to_numpy(), pooled449[name].to_numpy()
        if pd.api.types.is_float_dtype(pooled449[name].dtype):
            known = ~np.isnan(right)
            if left[known].tobytes() != right[known].tobytes():
                raise ValueError(f"{zone}/{name}: shared feature bits differ")
    names = families[COMPACT]["columns"][zone]
    if any(name not in full503 for name in names):
        raise ValueError(f"{zone}/{COMPACT}: source 503 lacks a selected column")
    compact = full503.loc[:, names].copy()
    validate_feature_frame(compact, names, expected_index, f"{zone}/123")
    return compact


__all__ = ["FULL", "POOLED", "COMPACT", "project_country"]
