"""Supervised hourly experts and their leakage-safe ensemble."""

from .base import LeakageRiskError, OptionalDependencyError
from .blended_residual_corrector import BlendedResidualCorrector
from .calibration import RollingMedianCalibrator
from .catboost_hourly import HourlyCatBoost, catboost_available
from .ensemble import NonNegativeOOFEnsemble, purged_expanding_splits
from .lear import HourlyLEAR
from .residual_corrector import (
    ResidualCorrectionError,
    ResidualCorrector,
    ResidualMetaFeatureBuilder,
    apply_residual_correction,
    build_residual_meta_features,
)

__all__ = [
    "BlendedResidualCorrector",
    "HourlyLEAR",
    "HourlyCatBoost",
    "LeakageRiskError",
    "NonNegativeOOFEnsemble",
    "OptionalDependencyError",
    "RollingMedianCalibrator",
    "ResidualCorrectionError",
    "ResidualCorrector",
    "ResidualMetaFeatureBuilder",
    "apply_residual_correction",
    "build_residual_meta_features",
    "catboost_available",
    "purged_expanding_splits",
]
