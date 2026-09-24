"""Isolated experimentation tools for Chronos-2 auxiliary models.

Nothing in the live forecasting pipeline imports this package.  The lab reads
published run artefacts, trains challengers below ``runs/experiments`` and
reuses the production model implementations without mutating their sources.
"""

from .config import AuxiliaryLabConfig, load_lab_config
from .runner import compare_runs, evaluate_run, predict_artifact, train_experiment

__all__ = [
    "AuxiliaryLabConfig",
    "compare_runs",
    "evaluate_run",
    "load_lab_config",
    "predict_artifact",
    "train_experiment",
]
