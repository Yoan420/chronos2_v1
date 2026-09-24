#!/usr/bin/env python
"""Run the sealed, causal PriceFM-inspired topology experiment.

This runner is deliberately separate from every production and live runner.
It trains a small residual model on already-frozen autonomous forecasts, tests
the local-only and direct-neighbour topology arms with chronological gates,
and publishes only below ``runs/experiments``.  PriceFM code, weights and data
are not imported: only the sparse graph idea is reimplemented clean-room.

The scientific protocol is fail-closed:

* 365 local delivery days are split 120 / 65 / 60 / 30 / 30 / 60;
* radius 0 and radius 1 use identical HGB hyperparameters;
* radius, identity and correction scale are selected on A; the following
  60-day block is development-only and is never a formal gate;
* the frozen recipe is sealed before B1 is opened, then refitted causally;
* every formal gate requires a 0.05 EUR/MWh MAE gain, positive gains in
  both chronological halves, and a positive paired daily bootstrap lower CI;
* Storm is not opened until the candidate prediction file has been frozen and
  hashed;
* MKOnline is never a topology feature and is only opened for FR/NL after the
  autonomous protocol has passed all of its gates.

The optional operational path deliberately freezes the published calibration:
it serializes the promoted BE corrector once, applies identity elsewhere, and
passes the existing FR/NL production blend through unchanged.  It is not a
rolling-365 implementation and never writes below ``runs/live``.

The annual reporting path performs no fit or inference.  It freezes a
stage-to-action policy before opening previously spared annual outcomes, then
copies only already-sealed formal predictions into two full 365-day strategies
with an explicit identity fallback.

The rolling-365 backtest is a separate, explicitly trained comparison.  For
each of the 365 evaluation delivery days it refits the frozen topological HGB
recipe on the immediately preceding 365 local delivery days, then predicts the
next day.  Radius, scale and hyperparameters remain those frozen before the
formal calibration gates; no rolling result is used for model selection.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import html
import io
import json
import math
import os
import platform
from pathlib import Path
import shutil
import sys
from typing import Any, Final, Protocol
from uuid import uuid4

import numpy as np
import pandas as pd
import joblib
import sklearn

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.reporting import write_hourly_html_report
from chronos2_hourly.models.topology_residual_corrector import (
    QUANTILE_COLUMNS,
    TopologyResidualCorrector,
)
from chronos2_hourly.topology_context import (
    TOPOLOGY_CONTEXT_COLUMNS,
    ZONE_TIMEZONES,
    build_topology_context,
)
from chronos2_hourly.topology_contract import (
    TopologyExperimentContract,
    audit_blend_source,
    load_topology_contract,
)
from chronos2_hourly.topology_report import (
    BLEND_ZONES,
    TopologyAnnualReportArtifact,
    TopologyReportArtifact,
    load_topology_annual_evaluation,
    write_topology_annual_html_report,
    write_topology_annual_report_index,
    write_topology_html_report,
    write_topology_report_index,
)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DEFAULT_CONFIG: Final[Path] = (
    PROJECT_ROOT / "config" / "pricefm_topology_experiment.yaml"
)
DEFAULT_OPERATIONAL_DIRECTORY: Final[Path] = (
    PROJECT_ROOT / "runs" / "experiments" / "pricefm_topology_operational_v1"
)
DEFAULT_DAILY_OUTPUT_ROOT: Final[Path] = (
    PROJECT_ROOT / "runs" / "experiments" / "pricefm_topology_daily"
)
DEFAULT_ANNUAL_OUTPUT_DIRECTORY: Final[Path] = (
    PROJECT_ROOT
    / "runs"
    / "experiments"
    / "pricefm_topology_annual_365_v1"
)
DEFAULT_ROLLING365_OUTPUT_DIRECTORY: Final[Path] = (
    PROJECT_ROOT
    / "runs"
    / "experiments"
    / "pricefm_topology_rolling365_v1"
)
SUPPORTED_ZONES: Final[tuple[str, ...]] = ("FR", "DE", "BE", "NL", "ES")
DAILY_MODES: Final[tuple[str, ...]] = (
    "production",
    "autonomous",
    "blend",
    "both",
)
REPORTING_INPUT_FILENAMES: Final[tuple[str, ...]] = (
    "aligned_inputs.csv.gz",
    "input_coverage.csv",
    "input_manifest.csv",
    "model_covariates_with_future.csv.gz",
)
STAGE_NAMES: Final[tuple[str, ...]] = (
    "seed",
    "a",
    "development",
    "b1",
    "b2",
    "final",
)
GATED_STAGES: Final[tuple[str, ...]] = ("b1", "b2", "final")
DEFAULT_SPLIT_DAYS: Final[Mapping[str, int]] = {
    "seed": 120,
    "a": 65,
    "development": 60,
    "b1": 30,
    "b2": 30,
    "final": 60,
}
DEFAULT_SCALE_GRID: Final[tuple[float, ...]] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
FROZEN_MODEL_PARAMETERS: Final[Mapping[str, object]] = {
    "learning_rate": 0.03,
    "max_iter": 100,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 240,
    "l2_regularization": 50.0,
    "correction_clip": 10.0,
    "min_training_rows": 168,
}
PROTOCOL_REVISION_REASON: Final[str] = (
    "initial 60-day holdout was inspected during model regularization and "
    "reclassified as development before formal gates"
)
CALIBRATION_V1_ARTIFACT_MANIFEST_SHA256: Final[str] = (
    "b11548b6a1d520e9163aacd3204b0bc97e8ca99fbb9d48856ea92dda82cbf405"
)


class TopologyExperimentError(ValueError):
    """Raised when an experiment would violate its sealed protocol."""


class _Corrector(Protocol):
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        base_predictions: pd.DataFrame,
    ) -> "_Corrector": ...

    def predict(
        self,
        X: pd.DataFrame,
        base_predictions: pd.DataFrame,
    ) -> pd.DataFrame: ...

    def predict_correction(
        self,
        X: pd.DataFrame,
        base_predictions: pd.DataFrame,
    ) -> pd.Series: ...

    def hyperparameter_sha256(self) -> str: ...

    def audit_metadata(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class ProtocolSplits:
    """Exact physical timestamps belonging to each local-day block."""

    timezone: str
    days: Mapping[str, tuple[str, ...]]
    indices: Mapping[str, pd.DatetimeIndex]

    @property
    def all_index(self) -> pd.DatetimeIndex:
        pieces = [self.indices[name] for name in STAGE_NAMES]
        return pieces[0].append(pieces[1:])


@dataclass(frozen=True)
class SegmentMetrics:
    """Paired point metrics on one chronological segment."""

    n_hours: int
    n_days: int
    candidate_mae: float
    baseline_mae: float
    gain_eur_mwh: float
    daily_win_rate: float
    first_half_gain_eur_mwh: float
    second_half_gain_eur_mwh: float
    bootstrap_ci95_lower_eur_mwh: float
    bootstrap_ci95_upper_eur_mwh: float


@dataclass(frozen=True)
class GateResult:
    """All four predeclared gate decisions for one sealed segment."""

    passes: bool
    reasons: tuple[str, ...]
    checks: Mapping[str, bool]
    metrics: SegmentMetrics


@dataclass
class AutonomousProtocolResult:
    """Predictions and audit trail from the autonomous topology protocol."""

    zone: str
    selected_radius: int
    selected_scale: float
    selected_arm: str
    neighbours: tuple[str, ...]
    selected_predictions: pd.DataFrame
    a_arm_predictions: Mapping[str, pd.DataFrame]
    development_predictions: pd.DataFrame
    metrics: dict[str, SegmentMetrics]
    gates: dict[str, GateResult]
    model_audits: dict[str, Mapping[str, object]]
    a_arm_metrics: dict[str, Mapping[str, float]]
    hyperparameter_sha256: str
    opened_stages: tuple[str, ...]
    promoted: bool


@dataclass
class DevelopmentProtocolResult:
    """Frozen A selection plus explicitly non-formal development diagnostic."""

    zone: str
    selected_radius: int
    selected_scale: float
    selected_arm: str
    neighbours: tuple[str, ...]
    a_arm_predictions: Mapping[str, pd.DataFrame]
    a_arm_metrics: dict[str, Mapping[str, float]]
    development_predictions: pd.DataFrame
    development_metrics: SegmentMetrics
    model_audits: dict[str, Mapping[str, object]]
    selected_hyperparameter_sha256: str
    training_actual: pd.Series
    training_base: pd.DataFrame
    training_context: pd.DataFrame


@dataclass(frozen=True)
class FormalStageData:
    """One formal block opened only after the recipe freeze or prior gate."""

    actual: pd.Series
    base_predictions: pd.DataFrame
    context: pd.DataFrame


@dataclass(frozen=True)
class RecipeSeal:
    """Hash boundary proving formal periods were unopened during selection."""

    path: Path
    sha256: str
    selected_radius: int
    selected_scale: float
    config_sha256: str


@dataclass
class BlendProtocolResult:
    """FR/NL-only topology-then-MKOnline blend evaluation."""

    zone: str
    selected_weight_mkonline: float
    selected_weight_autonomous: float
    previous_weight_mkonline: float
    previous_weight_autonomous: float
    predictions: pd.DataFrame
    metrics: dict[str, SegmentMetrics]
    gates: dict[str, GateResult]
    opened_stages: tuple[str, ...]
    promoted: bool
    grid_step: float


@dataclass(frozen=True)
class CandidateSeal:
    """Immutable-by-protocol candidate prediction and its digest sidecar."""

    prediction_path: Path
    manifest_path: Path
    prediction_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class ExperimentConfig:
    """Small normalized view of the YAML contract used by orchestration."""

    source_path: Path
    experiment_id: str
    output_directory: Path
    zones: tuple[str, ...]
    split_days: Mapping[str, int]
    gate_minimum_gain: float
    bootstrap_samples: int
    bootstrap_seed: int
    model_parameters: Mapping[str, object]
    scale_grid: tuple[float, ...]
    blend_grid_step: float
    protocol_frozen_at_utc: str
    config_sha256: str
    raw: Mapping[str, object]
    contract: TopologyExperimentContract


@dataclass(frozen=True)
class PublishedCalibration:
    """Fully checksum-verified calibration used to build an operational bundle."""

    directory: Path
    artifact_manifest_sha256: str
    experiment_manifest: Mapping[str, object]
    evaluations: Mapping[str, Mapping[str, object]]
    recipes: Mapping[str, Mapping[str, object]]
    candidate_seals: Mapping[str, CandidateSeal]


@dataclass(frozen=True)
class OperationalBundle:
    """Validated fixed-calibration model bundle safe for daily application."""

    directory: Path
    artifact_manifest_sha256: str
    manifest: Mapping[str, object]
    model: TopologyResidualCorrector | None
    evaluations: Mapping[str, Mapping[str, object]]
    candidate_seals: Mapping[str, CandidateSeal]


@dataclass(frozen=True)
class DailyArchive:
    """One exact, immutable live archive opened read-only for a delivery day."""

    zone: str
    timezone: str
    delivery_day: str
    directory: Path
    forecast_path: Path
    artifact_manifest_sha256: str


@dataclass(frozen=True)
class AnnualPolicySeal:
    """A stage-to-action policy frozen before unopened annual outcomes."""

    zone: str
    path: Path
    sha256: str
    governed_actions: Mapping[str, str]
    formal_shadow_actions: Mapping[str, str]


@dataclass(frozen=True)
class AnnualSourceData:
    """Exact sealed current-model observations for the governed 365 days."""

    actual: pd.Series
    base_predictions: pd.DataFrame
    forecast_origins: pd.DatetimeIndex
    source_sha256: str


@dataclass(frozen=True)
class Rolling365ZoneData:
    """Two-year causal source window used by one rolling-365 zone."""

    actual: pd.Series
    ensemble_base: pd.DataFrame
    current_autonomous: pd.DataFrame
    forecast_origins: pd.DatetimeIndex
    contexts: Mapping[int, pd.DataFrame]
    evaluation_index: pd.DatetimeIndex
    source_sha256: str


@dataclass
class Rolling365ZoneResult:
    """Daily-refitted predictions and their immutable training audit."""

    zone: str
    timezone: str
    selected_radius: int
    selected_scale: float
    selected_arm: str
    recipe_sha256: str
    hyperparameter_sha256: str
    predictions: pd.DataFrame
    refits: pd.DataFrame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _normalise_zones(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        values = [str(values)]
    result: list[str] = []
    for raw in values:
        zone = str(raw).strip().upper()
        if zone not in SUPPORTED_ZONES:
            raise TopologyExperimentError(
                f"Zone non supportee: {raw!r}; choix={SUPPORTED_ZONES}."
            )
        if zone in result:
            raise TopologyExperimentError(f"Zone dupliquee: {zone}.")
        result.append(zone)
    if not result:
        raise TopologyExperimentError("Selectionnez au moins une zone.")
    return tuple(result)


def _validate_utc_index(index: pd.Index, *, name: str) -> pd.DatetimeIndex:
    values = pd.DatetimeIndex(index)
    if values.tz is None:
        raise TopologyExperimentError(f"{name} doit etre timezone-aware.")
    values = values.tz_convert("UTC")
    if values.has_duplicates or not values.is_monotonic_increasing:
        raise TopologyExperimentError(
            f"{name} doit etre strictement croissant et sans doublon."
        )
    if len(values) > 1:
        delta = np.diff(values.asi8)
        expected = pd.Timedelta(hours=1).value
        if not bool((delta == expected).all()):
            raise TopologyExperimentError(
                f"{name} doit etre une timeline physique UTC horaire continue."
            )
    values.name = index.name or "delivery_start_utc"
    return values


def build_protocol_splits(
    index: pd.Index,
    *,
    timezone_name: str,
    split_days: Mapping[str, int] = DEFAULT_SPLIT_DAYS,
) -> ProtocolSplits:
    """Split exactly 365 complete local days without assuming 24 h per day."""

    utc = _validate_utc_index(index, name="forecast_index")
    if tuple(split_days) != STAGE_NAMES:
        raise TopologyExperimentError(
            f"split_days doit suivre exactement l'ordre {STAGE_NAMES}."
        )
    counts: dict[str, int] = {}
    for name in STAGE_NAMES:
        value = split_days[name]
        if isinstance(value, bool) or int(value) <= 0:
            raise TopologyExperimentError(f"split_days.{name} doit etre > 0.")
        counts[name] = int(value)
    if sum(counts.values()) != 365:
        raise TopologyExperimentError("Le protocole doit couvrir exactement 365 jours.")

    local = utc.tz_convert(timezone_name)
    day_values = pd.Index(local.strftime("%Y-%m-%d"), name="local_day")
    unique_days = tuple(dict.fromkeys(day_values.tolist()))
    if len(unique_days) != 365:
        raise TopologyExperimentError(
            f"365 jours locaux requis, observes={len(unique_days)}."
        )
    expected_days = tuple(
        day.strftime("%Y-%m-%d")
        for day in pd.date_range(unique_days[0], unique_days[-1], freq="D")
    )
    if unique_days != expected_days:
        raise TopologyExperimentError("Les 365 jours locaux doivent etre contigus.")
    for day in unique_days:
        observed = utc[day_values == day]
        expected = local_delivery_day_index(day, timezone=timezone_name)
        if not observed.equals(expected):
            raise TopologyExperimentError(
                f"Jour local incomplet ou mal ordonne: {day}."
            )

    days_by_stage: dict[str, tuple[str, ...]] = {}
    indices: dict[str, pd.DatetimeIndex] = {}
    cursor = 0
    for name in STAGE_NAMES:
        selected_days = unique_days[cursor : cursor + counts[name]]
        cursor += counts[name]
        days_by_stage[name] = selected_days
        selected = utc[day_values.isin(selected_days)]
        selected.name = utc.name
        indices[name] = selected
    result = ProtocolSplits(
        timezone=timezone_name,
        days=days_by_stage,
        indices=indices,
    )
    if not result.all_index.equals(utc):
        raise RuntimeError("Le split n'a pas preserve la timeline exacte.")
    return result


def _slice_context(frame: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Slice a topology frame while explicitly retaining its sealed attrs."""

    result = frame.loc[index].copy()
    result.attrs = dict(frame.attrs)
    return result


def _validate_quantiles(
    frame: pd.DataFrame,
    *,
    expected_index: pd.DatetimeIndex | None = None,
    name: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} doit etre un DataFrame.")
    missing = [column for column in QUANTILE_COLUMNS if column not in frame]
    if missing:
        raise TopologyExperimentError(f"{name}: quantiles absents={missing}.")
    result = frame.loc[:, list(QUANTILE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if expected_index is not None and not result.index.equals(expected_index):
        raise TopologyExperimentError(f"{name}: index different de la timeline attendue.")
    values = result.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise TopologyExperimentError(f"{name}: quantiles non finis.")
    if bool(((result["q10"] > result["q50"]) | (result["q50"] > result["q90"])).any()):
        raise TopologyExperimentError(f"{name}: quantiles croises.")
    return result.astype(float)


def _daily_paired_gains(
    actual: pd.Series,
    candidate: pd.Series,
    baseline: pd.Series,
    *,
    timezone_name: str,
) -> pd.Series:
    index = _validate_utc_index(actual.index, name="metric_index")
    if not candidate.index.equals(index) or not baseline.index.equals(index):
        raise TopologyExperimentError("Les series de metriques ne sont pas alignees.")
    values = np.column_stack(
        [
            pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(candidate, errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(baseline, errors="coerce").to_numpy(dtype=float),
        ]
    )
    if not np.isfinite(values).all():
        raise TopologyExperimentError("Les metriques appariees exigent des valeurs finies.")
    local_days = pd.Index(index.tz_convert(timezone_name).strftime("%Y-%m-%d"))
    hourly_gain = np.abs(values[:, 0] - values[:, 2]) - np.abs(
        values[:, 0] - values[:, 1]
    )
    result = pd.Series(hourly_gain, index=local_days, name="daily_mae_gain")
    return result.groupby(level=0, sort=False).mean()


def paired_segment_metrics(
    actual: pd.Series,
    candidate: pd.Series,
    baseline: pd.Series,
    *,
    timezone_name: str,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 120,
) -> SegmentMetrics:
    """Compute paired hourly MAE and a paired local-day bootstrap interval."""

    if isinstance(bootstrap_samples, bool) or int(bootstrap_samples) < 1:
        raise TopologyExperimentError("bootstrap_samples doit etre >= 1.")
    daily = _daily_paired_gains(
        actual,
        candidate,
        baseline,
        timezone_name=timezone_name,
    )
    if len(daily) < 2:
        raise TopologyExperimentError("Au moins deux jours sont requis pour le bootstrap.")
    actual_values = actual.to_numpy(dtype=float)
    candidate_values = candidate.to_numpy(dtype=float)
    baseline_values = baseline.to_numpy(dtype=float)
    candidate_mae = float(np.mean(np.abs(actual_values - candidate_values)))
    baseline_mae = float(np.mean(np.abs(actual_values - baseline_values)))
    gains = daily.to_numpy(dtype=float)
    midpoint = len(gains) // 2
    if midpoint < 1 or midpoint == len(gains):
        raise TopologyExperimentError("Deux moities chronologiques non vides sont requises.")
    rng = np.random.default_rng(int(bootstrap_seed))
    # 20k x 60 is small, deterministic and avoids an accidental hourly bootstrap.
    draws = rng.integers(0, len(gains), size=(int(bootstrap_samples), len(gains)))
    bootstrap = gains[draws].mean(axis=1)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return SegmentMetrics(
        n_hours=int(len(actual)),
        n_days=int(len(daily)),
        candidate_mae=candidate_mae,
        baseline_mae=baseline_mae,
        gain_eur_mwh=baseline_mae - candidate_mae,
        daily_win_rate=float(np.mean(gains > 0.0)),
        first_half_gain_eur_mwh=float(np.mean(gains[:midpoint])),
        second_half_gain_eur_mwh=float(np.mean(gains[midpoint:])),
        bootstrap_ci95_lower_eur_mwh=float(lower),
        bootstrap_ci95_upper_eur_mwh=float(upper),
    )


def evaluate_gate(
    metrics: SegmentMetrics,
    *,
    minimum_gain: float = 0.05,
) -> GateResult:
    """Apply all predeclared requirements; there is no discretionary override."""

    if not math.isfinite(minimum_gain) or minimum_gain < 0.0:
        raise TopologyExperimentError("minimum_gain doit etre fini et positif ou nul.")
    checks = {
        "gain_at_least_minimum": metrics.gain_eur_mwh >= minimum_gain,
        "first_half_positive": metrics.first_half_gain_eur_mwh > 0.0,
        "second_half_positive": metrics.second_half_gain_eur_mwh > 0.0,
        "bootstrap_ci95_lower_positive": (
            metrics.bootstrap_ci95_lower_eur_mwh > 0.0
        ),
    }
    labels = {
        "gain_at_least_minimum": (
            f"gain MAE < {minimum_gain:.3f} EUR/MWh"
        ),
        "first_half_positive": "gain de la premiere moitie <= 0",
        "second_half_positive": "gain de la seconde moitie <= 0",
        "bootstrap_ci95_lower_positive": "borne basse bootstrap appariee <= 0",
    }
    reasons = tuple(labels[name] for name, passed in checks.items() if not passed)
    return GateResult(
        passes=not reasons,
        reasons=reasons,
        checks=checks,
        metrics=metrics,
    )


def _metric_summary(
    actual: pd.Series,
    prediction: pd.Series,
) -> Mapping[str, float]:
    values = np.column_stack(
        [actual.to_numpy(dtype=float), prediction.to_numpy(dtype=float)]
    )
    if not np.isfinite(values).all():
        raise TopologyExperimentError("Valeurs non finies dans la selection A.")
    return {
        "mae": float(np.mean(np.abs(values[:, 0] - values[:, 1]))),
        "bias": float(np.mean(values[:, 1] - values[:, 0])),
    }


def _new_corrector(
    parameters: Mapping[str, object],
    *,
    correction_scale: float,
) -> TopologyResidualCorrector:
    kwargs = dict(parameters)
    kwargs["correction_scale"] = float(correction_scale)
    return TopologyResidualCorrector(**kwargs)


def _common_shift_prediction(
    base: pd.DataFrame,
    raw_correction: pd.Series,
    *,
    scale: float,
    clip: float,
) -> pd.DataFrame:
    base_values = _validate_quantiles(base, name="common_shift_base")
    raw = pd.to_numeric(raw_correction, errors="coerce").reindex(base_values.index)
    if not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise TopologyExperimentError("La correction brute contient des valeurs non finies.")
    shift = np.clip(raw.to_numpy(dtype=float) * float(scale), -clip, clip)
    result = base_values.add(shift, axis=0)
    return _validate_quantiles(
        result,
        expected_index=base_values.index,
        name="common_shift_prediction",
    )


def _raw_model_correction(
    model: _Corrector,
    context: pd.DataFrame,
    base: pd.DataFrame,
) -> pd.Series:
    correction = model.predict_correction(context, base)
    raw = correction.attrs.get("raw_correction")
    if not isinstance(raw, pd.Series):
        raise TopologyExperimentError(
            "Le correcteur doit exposer raw_correction pour la grille de scale A."
        )
    return pd.to_numeric(raw, errors="coerce").astype(float)


def run_development_protocol(
    *,
    zone: str,
    actual: pd.Series,
    base_predictions: pd.DataFrame,
    contexts: Mapping[int, pd.DataFrame],
    splits: ProtocolSplits,
    model_parameters: Mapping[str, object],
    scale_grid: Sequence[float] = DEFAULT_SCALE_GRID,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 120,
    corrector_factory: Callable[[float], _Corrector] | None = None,
) -> DevelopmentProtocolResult:
    """Select exclusively on A, then run a non-formal development diagnostic.

    ``actual`` and both contexts must stop at the end of ``development``.  A
    caller cannot accidentally leak B1 through this API because a longer
    timeline is rejected before fitting.
    """

    zone_key = str(zone).strip().upper()
    if zone_key not in SUPPORTED_ZONES:
        raise TopologyExperimentError(f"Zone non supportee: {zone!r}.")
    preformal_index = splits.indices["seed"].append(splits.indices["a"])
    preformal_index = preformal_index.append(splits.indices["development"])
    if not actual.index.equals(preformal_index):
        raise TopologyExperimentError(
            "La phase development doit s'arreter avant la premiere heure de B1."
        )
    actual_values = pd.to_numeric(actual, errors="coerce").astype(float)
    if not np.isfinite(actual_values.to_numpy(dtype=float)).all():
        raise TopologyExperimentError("actual contient des valeurs non finies.")
    base = _validate_quantiles(
        base_predictions,
        expected_index=preformal_index,
        name="base_predictions",
    )
    if set(contexts) != {0, 1}:
        raise TopologyExperimentError("Les contextes radius 0 et radius 1 sont requis.")
    for radius, context in contexts.items():
        if not context.index.equals(preformal_index):
            raise TopologyExperimentError(
                f"Le contexte radius={radius} ne couvre pas la timeline exacte."
            )
        metadata = context.attrs.get("topology_context", {})
        if not isinstance(metadata, Mapping) or metadata.get("radius") != radius:
            raise TopologyExperimentError(
                f"Metadonnees de contexte invalides pour radius={radius}."
            )
        if metadata.get("target_zone") != zone_key:
            raise TopologyExperimentError("Le contexte appartient a une autre zone.")

    fixed_parameters = dict(model_parameters)
    if fixed_parameters != dict(FROZEN_MODEL_PARAMETERS):
        raise TopologyExperimentError(
            "Les hyperparametres different du bloc gouverne et fige."
        )
    scales = tuple(float(value) for value in scale_grid)
    if scales != DEFAULT_SCALE_GRID:
        raise TopologyExperimentError(
            f"scale_grid doit rester exactement {DEFAULT_SCALE_GRID}."
        )
    factory: Callable[[float], _Corrector]
    if corrector_factory is None:
        factory = lambda scale: _new_corrector(
            fixed_parameters,
            correction_scale=scale,
        )
    else:
        factory = corrector_factory

    seed_index = splits.indices["seed"]
    a_index = splits.indices["a"]
    development_index = splits.indices["development"]
    a_predictions: dict[str, pd.DataFrame] = {
        "identity": base.loc[a_index].copy()
    }
    a_audits: dict[str, Mapping[str, object]] = {}
    a_metrics: dict[str, Mapping[str, float]] = {
        "identity": _metric_summary(actual_values.loc[a_index], base.loc[a_index, "q50"])
    }
    clip = float(fixed_parameters["correction_clip"])
    radius_model_hashes: set[str] = set()
    for radius in (0, 1):
        # Fit the tree once.  Scale is a deterministic common-shift transform
        # evaluated on A; it never changes tree learning or sees development.
        model = factory(1.0)
        model.fit(
            _slice_context(contexts[radius], seed_index),
            actual_values.loc[seed_index],
            base.loc[seed_index],
        )
        raw = _raw_model_correction(
            model,
            _slice_context(contexts[radius], a_index),
            base.loc[a_index],
        )
        radius_model_hashes.add(model.hyperparameter_sha256())
        a_audits[f"radius{radius}_seed_to_a"] = model.audit_metadata()
        # Keep the declared zero-scale arms in the audit table even though
        # they are mathematically identical to ``identity``.  Identity was
        # inserted first, so the deterministic tie-break still selects the
        # explicit fallback rather than attaching a fictitious radius to it.
        for scale in scales:
            arm = f"radius{radius}_scale{scale:g}"
            predicted = _common_shift_prediction(
                base.loc[a_index],
                raw,
                scale=scale,
                clip=clip,
            )
            a_predictions[arm] = predicted
            a_metrics[arm] = _metric_summary(
                actual_values.loc[a_index], predicted["q50"]
            )
    if len(radius_model_hashes) != 1:
        raise TopologyExperimentError(
            "Radius 0 et radius 1 n'ont pas les memes hyperparametres A."
        )
    selected_arm = min(
        a_metrics,
        key=lambda arm: (a_metrics[arm]["mae"], tuple(a_metrics).index(arm)),
    )
    if selected_arm == "identity":
        selected_radius = 0
        selected_scale = 0.0
    else:
        selected_radius = int(selected_arm[len("radius")])
        selected_scale = float(selected_arm.rsplit("scale", 1)[1])

    train_to_development = seed_index.append(a_index)
    if selected_scale == 0.0:
        development_prediction = base.loc[development_index].copy()
        selected_hash = _canonical_sha256(
            {"identity": True, "parameters": fixed_parameters, "scale": 0.0}
        )
    else:
        selected_model = factory(selected_scale)
        selected_model.fit(
            _slice_context(contexts[selected_radius], train_to_development),
            actual_values.loc[train_to_development],
            base.loc[train_to_development],
        )
        development_prediction = _validate_quantiles(
            selected_model.predict(
                _slice_context(contexts[selected_radius], development_index),
                base.loc[development_index],
            ),
            expected_index=development_index,
            name="development_prediction",
        )
        selected_hash = selected_model.hyperparameter_sha256()
        a_audits["selected_refit_to_development"] = selected_model.audit_metadata()
    development_metrics = paired_segment_metrics(
        actual_values.loc[development_index],
        development_prediction["q50"],
        base.loc[development_index, "q50"],
        timezone_name=splits.timezone,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    metadata = contexts[selected_radius].attrs["topology_context"]
    return DevelopmentProtocolResult(
        zone=zone_key,
        selected_radius=selected_radius,
        selected_scale=selected_scale,
        selected_arm=selected_arm,
        neighbours=tuple(str(value) for value in metadata.get("neighbours", ())),
        a_arm_predictions=a_predictions,
        development_predictions=development_prediction,
        development_metrics=development_metrics,
        model_audits=a_audits,
        a_arm_metrics=a_metrics,
        selected_hyperparameter_sha256=selected_hash,
        training_actual=actual_values.copy(),
        training_base=base.copy(),
        training_context=_slice_context(contexts[selected_radius], preformal_index),
    )


def freeze_protocol_recipe(
    directory: str | Path,
    development: DevelopmentProtocolResult,
    *,
    config: ExperimentConfig,
) -> RecipeSeal:
    """Freeze every modelling decision before a B1 value is opened."""

    destination = Path(directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "topology_recipe_seal.json"
    if path.exists():
        raise FileExistsError(path)
    payload = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "zone": development.zone,
        "selected_arm": development.selected_arm,
        "selected_radius": development.selected_radius,
        "selected_scale": development.selected_scale,
        "selected_neighbours": list(development.neighbours),
        "selection_A_arm_metrics": development.a_arm_metrics,
        "development_diagnostic": _segment_payload(
            development.development_metrics
        ),
        "model_parameters": dict(config.model_parameters),
        "scale_grid": list(config.scale_grid),
        "split_days": dict(config.split_days),
        "formal_gate_policy": {
            "minimum_mae_gain_eur_mwh": config.gate_minimum_gain,
            "bootstrap_samples": config.bootstrap_samples,
            "bootstrap_seed": config.bootstrap_seed,
            "positive_chronological_halves_required": True,
            "bootstrap_lower_ci95_positive_required": True,
        },
        "model_hyperparameters_sha256": development.selected_hyperparameter_sha256,
        "selection_period": "A",
        "development_used_for_formal_gate": False,
        "formal_gate_periods_unopened_before_freeze": True,
        "protocol_revision_reason": PROTOCOL_REVISION_REASON,
        "protocol_frozen_at_utc": config.protocol_frozen_at_utc,
        "config_sha256": config.config_sha256,
        "recipe_frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "storm_loaded_before_recipe_freeze": False,
        "mkonline_loaded_before_autonomous_gates": False,
    }
    _write_json(path, payload)
    return RecipeSeal(
        path=path,
        sha256=_sha256(path),
        selected_radius=development.selected_radius,
        selected_scale=development.selected_scale,
        config_sha256=config.config_sha256,
    )


def run_formal_autonomous_protocol(
    *,
    development: DevelopmentProtocolResult,
    recipe_seal: RecipeSeal,
    stage_loader: Callable[[str], FormalStageData],
    splits: ProtocolSplits,
    model_parameters: Mapping[str, object],
    gate_minimum_gain: float = 0.05,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 120,
    corrector_factory: Callable[[float], _Corrector] | None = None,
) -> AutonomousProtocolResult:
    """Open B1/B2/final sequentially only after verifying the recipe seal."""

    if not recipe_seal.path.is_file() or _sha256(recipe_seal.path) != recipe_seal.sha256:
        raise TopologyExperimentError("La recette a change avant l'ouverture de B1.")
    if recipe_seal.selected_radius != development.selected_radius or not math.isclose(
        recipe_seal.selected_scale,
        development.selected_scale,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise TopologyExperimentError("La recette scellee contredit la selection A.")
    fixed_parameters = dict(model_parameters)
    if fixed_parameters != dict(FROZEN_MODEL_PARAMETERS):
        raise TopologyExperimentError("Les hyperparametres formels ne sont plus figes.")
    if corrector_factory is None:
        factory = lambda scale: _new_corrector(
            fixed_parameters,
            correction_scale=scale,
        )
    else:
        factory = corrector_factory

    all_index = splits.all_index
    selected = pd.DataFrame(
        np.nan,
        index=all_index,
        columns=list(QUANTILE_COLUMNS),
        dtype=float,
    )
    a_index = splits.indices["a"]
    development_index = splits.indices["development"]
    selected.loc[a_index, list(QUANTILE_COLUMNS)] = development.a_arm_predictions[
        development.selected_arm
    ].to_numpy(dtype=float)
    selected.loc[development_index, list(QUANTILE_COLUMNS)] = (
        development.development_predictions.to_numpy(dtype=float)
    )
    train_actual = development.training_actual.copy()
    train_base = development.training_base.copy()
    train_context = development.training_context.copy()
    train_context.attrs = dict(development.training_context.attrs)
    metrics: dict[str, SegmentMetrics] = {
        "development": development.development_metrics
    }
    gates: dict[str, GateResult] = {}
    audits = dict(development.model_audits)
    opened = ["a", "development"]
    if development.selected_arm == "identity":
        # Identity winning A is already the fail-closed decision.  Opening a
        # formal holdout cannot promote a non-topological arm and would only
        # spend untouched evidence.  In particular stage_loader is never
        # invoked in this branch.
        return AutonomousProtocolResult(
            zone=development.zone,
            selected_radius=development.selected_radius,
            selected_scale=0.0,
            selected_arm="identity",
            neighbours=development.neighbours,
            selected_predictions=selected,
            a_arm_predictions=development.a_arm_predictions,
            development_predictions=development.development_predictions,
            metrics=metrics,
            gates=gates,
            model_audits=audits,
            a_arm_metrics=development.a_arm_metrics,
            hyperparameter_sha256=development.selected_hyperparameter_sha256,
            opened_stages=tuple(opened),
            promoted=False,
        )
    for stage in GATED_STAGES:
        # This is the first operation capable of exposing the stage outcome.
        # It is invoked only after the recipe hash check and previous gate.
        stage_data = stage_loader(stage)
        expected = splits.indices[stage]
        stage_actual = pd.to_numeric(stage_data.actual, errors="coerce").astype(float)
        if not stage_actual.index.equals(expected):
            raise TopologyExperimentError(f"{stage}: actual ne couvre pas le bloc exact.")
        if not np.isfinite(stage_actual.to_numpy(dtype=float)).all():
            raise TopologyExperimentError(f"{stage}: actual non fini.")
        stage_base = _validate_quantiles(
            stage_data.base_predictions,
            expected_index=expected,
            name=f"{stage}_base",
        )
        stage_context = stage_data.context
        if not stage_context.index.equals(expected):
            raise TopologyExperimentError(f"{stage}: contexte mal aligne.")
        stage_context = _slice_context(stage_context, expected)
        if development.selected_scale == 0.0:
            predicted = stage_base.copy()
        else:
            model = factory(development.selected_scale)
            model.fit(train_context, train_actual, train_base)
            predicted = _validate_quantiles(
                model.predict(stage_context, stage_base),
                expected_index=expected,
                name=f"selected_{stage}",
            )
            if model.hyperparameter_sha256() != development.selected_hyperparameter_sha256:
                raise TopologyExperimentError(
                    "Le hash hyperparametre a change apres le gel de recette."
                )
            audits[f"selected_refit_to_{stage}"] = model.audit_metadata()
        selected.loc[expected, list(QUANTILE_COLUMNS)] = predicted.to_numpy(dtype=float)
        segment = paired_segment_metrics(
            stage_actual,
            predicted["q50"],
            stage_base["q50"],
            timezone_name=splits.timezone,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        gate = evaluate_gate(segment, minimum_gain=gate_minimum_gain)
        metrics[stage] = segment
        gates[stage] = gate
        opened.append(stage)
        if not gate.passes:
            break
        train_actual = pd.concat([train_actual, stage_actual])
        train_base = pd.concat([train_base, stage_base])
        previous_attrs = dict(train_context.attrs)
        train_context = pd.concat([train_context, stage_context])
        train_context.attrs = previous_attrs
    promoted = tuple(opened) == (
        "a",
        "development",
        "b1",
        "b2",
        "final",
    ) and gates["final"].passes
    return AutonomousProtocolResult(
        zone=development.zone,
        selected_radius=development.selected_radius,
        selected_scale=development.selected_scale,
        selected_arm=development.selected_arm,
        neighbours=development.neighbours,
        selected_predictions=selected,
        a_arm_predictions=development.a_arm_predictions,
        development_predictions=development.development_predictions,
        metrics=metrics,
        gates=gates,
        model_audits=audits,
        a_arm_metrics=development.a_arm_metrics,
        hyperparameter_sha256=development.selected_hyperparameter_sha256,
        opened_stages=tuple(opened),
        promoted=promoted,
    )


def _blend_quantiles(
    autonomous: pd.DataFrame,
    mkonline_q50: pd.Series,
    *,
    weight_mkonline: float,
) -> pd.DataFrame:
    auto = _validate_quantiles(autonomous, name="topology_autonomous")
    mk = pd.to_numeric(mkonline_q50, errors="coerce").reindex(auto.index)
    if not np.isfinite(mk.to_numpy(dtype=float)).all():
        raise TopologyExperimentError("MKOnline ne couvre pas la timeline exacte.")
    weight = float(weight_mkonline)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise TopologyExperimentError("Le poids MKOnline doit etre dans [0,1].")
    q50 = (1.0 - weight) * auto["q50"] + weight * mk
    shift = q50 - auto["q50"]
    result = auto.add(shift, axis=0)
    result.attrs["weight_mkonline"] = weight
    result.attrs["weight_topology_autonomous"] = 1.0 - weight
    return _validate_quantiles(result, expected_index=auto.index, name="topology_blend")


def select_l1_blend_weight(
    actual: pd.Series,
    topology_q50: pd.Series,
    mkonline_q50: pd.Series,
    *,
    grid_step: float = 0.025,
) -> tuple[float, Mapping[str, object]]:
    """Fit one convex MKOnline weight on A only under hourly L1 loss."""

    if not math.isfinite(grid_step) or not 0.0 < grid_step <= 1.0:
        raise TopologyExperimentError("grid_step doit etre dans ]0,1].")
    values = np.column_stack(
        [
            actual.to_numpy(dtype=float),
            topology_q50.reindex(actual.index).to_numpy(dtype=float),
            mkonline_q50.reindex(actual.index).to_numpy(dtype=float),
        ]
    )
    if not np.isfinite(values).all():
        raise TopologyExperimentError("Selection blend A: valeurs non finies.")
    steps = int(round(1.0 / grid_step))
    if not math.isclose(steps * grid_step, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise TopologyExperimentError("grid_step doit diviser exactement 1.0.")
    grid = np.linspace(0.0, 1.0, steps + 1)
    losses = []
    for weight in grid:
        prediction = (1.0 - weight) * values[:, 1] + weight * values[:, 2]
        losses.append(float(np.mean(np.abs(values[:, 0] - prediction))))
    minimum = min(losses)
    # A deterministic tie goes to the smallest external-model weight.
    selected_index = next(
        index
        for index, loss in enumerate(losses)
        if math.isclose(loss, minimum, rel_tol=0.0, abs_tol=1e-12)
    )
    weight = float(grid[selected_index])
    return weight, {
        "fit_method": "constrained_l1_grid",
        "fitted_on": "A",
        "final_used_for_tuning": False,
        "grid_step": float(grid_step),
        "selected_weight_mkonline": weight,
        "selected_mae": minimum,
        "grid_sha256": _canonical_sha256(
            {"weights": grid.tolist(), "mae": losses}
        ),
    }


def run_blend_protocol(
    *,
    zone: str,
    autonomous: AutonomousProtocolResult,
    actual: pd.Series,
    mkonline_q50_loader: Callable[[], pd.Series],
    production_autonomous_loader: Callable[[], pd.DataFrame],
    splits: ProtocolSplits,
    previous_weight_mkonline: float,
    grid_step: float = 0.025,
    gate_minimum_gain: float = 0.05,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 120,
) -> BlendProtocolResult | None:
    """Evaluate the optional blend, with guards preceding every MKOnline read."""

    zone_key = str(zone).strip().upper()
    if zone_key not in BLEND_ZONES:
        # This check is intentionally before invoking either loader.
        return None
    if autonomous.zone != zone_key:
        raise TopologyExperimentError("Le resultat autonome appartient a une autre zone.")
    if not autonomous.promoted:
        # MKOnline remains unopened when the autonomous candidate failed.
        return None
    mkonline = pd.to_numeric(mkonline_q50_loader(), errors="coerce").astype(float)
    production_autonomous = _validate_quantiles(
        production_autonomous_loader(),
        expected_index=splits.all_index,
        name="production_autonomous_baseline",
    )
    if not mkonline.index.equals(splits.all_index):
        raise TopologyExperimentError("MKOnline ne couvre pas les 365 jours exacts.")
    old_weight = float(previous_weight_mkonline)
    if not math.isfinite(old_weight) or not 0.0 <= old_weight <= 1.0:
        raise TopologyExperimentError("Poids MKOnline de production invalide.")
    # Rebuild the current production comparator from its sealed ingredients;
    # no already-scored blend file is accepted as an unexplained baseline.
    current_blend = _blend_quantiles(
        production_autonomous,
        mkonline,
        weight_mkonline=old_weight,
    )
    a_index = splits.indices["a"]
    weight, _audit = select_l1_blend_weight(
        actual.loc[a_index],
        autonomous.selected_predictions.loc[a_index, "q50"],
        mkonline.loc[a_index],
        grid_step=grid_step,
    )
    candidate = pd.DataFrame(
        np.nan,
        index=splits.all_index,
        columns=list(QUANTILE_COLUMNS),
        dtype=float,
    )
    metrics: dict[str, SegmentMetrics] = {}
    gates: dict[str, GateResult] = {}
    opened = ["a"]
    candidate.loc[a_index, list(QUANTILE_COLUMNS)] = _blend_quantiles(
        autonomous.selected_predictions.loc[a_index],
        mkonline.loc[a_index],
        weight_mkonline=weight,
    ).to_numpy(dtype=float)
    for stage in GATED_STAGES:
        stage_index = splits.indices[stage]
        auto_stage = autonomous.selected_predictions.loc[stage_index]
        if not np.isfinite(auto_stage.to_numpy(dtype=float)).all():
            raise TopologyExperimentError(
                f"Predictions autonomes absentes avant la gate blend {stage}."
            )
        predicted = _blend_quantiles(
            auto_stage,
            mkonline.loc[stage_index],
            weight_mkonline=weight,
        )
        candidate.loc[stage_index, list(QUANTILE_COLUMNS)] = predicted.to_numpy(
            dtype=float
        )
        segment = paired_segment_metrics(
            actual.loc[stage_index],
            predicted["q50"],
            current_blend.loc[stage_index, "q50"],
            timezone_name=splits.timezone,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        gate = evaluate_gate(segment, minimum_gain=gate_minimum_gain)
        metrics[stage] = segment
        gates[stage] = gate
        opened.append(stage)
        if not gate.passes:
            break
    promoted = tuple(opened) == ("a", "b1", "b2", "final") and gates[
        "final"
    ].passes
    return BlendProtocolResult(
        zone=zone_key,
        selected_weight_mkonline=weight,
        selected_weight_autonomous=1.0 - weight,
        previous_weight_mkonline=old_weight,
        previous_weight_autonomous=1.0 - old_weight,
        predictions=candidate,
        metrics=metrics,
        gates=gates,
        opened_stages=tuple(opened),
        promoted=promoted,
        grid_step=float(grid_step),
    )


def _write_deterministic_csv_gzip(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, lineterminator="\n")


def seal_candidate_predictions(
    directory: str | Path,
    frame: pd.DataFrame,
    *,
    zone: str,
    experiment_id: str,
) -> CandidateSeal:
    """Write and hash the candidate before any comparator loader may run."""

    destination = Path(directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    prediction_path = destination / "candidate_predictions.csv.gz"
    manifest_path = destination / "candidate_prediction_seal.json"
    if prediction_path.exists() or manifest_path.exists():
        raise FileExistsError("Le candidat est deja scelle; aucun ecrasement permis.")
    _write_deterministic_csv_gzip(prediction_path, frame)
    prediction_sha = _sha256(prediction_path)
    manifest = {
        "schema_version": 1,
        "experiment_id": str(experiment_id),
        "zone": str(zone).upper(),
        "candidate_path": prediction_path.name,
        "candidate_sha256": prediction_sha,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "storm_loaded_before_seal": False,
        "mkonline_used_as_topology_input": False,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest_path, manifest)
    return CandidateSeal(
        prediction_path=prediction_path,
        manifest_path=manifest_path,
        prediction_sha256=prediction_sha,
        manifest_sha256=_sha256(manifest_path),
    )


def load_comparator_after_seal(
    seal: CandidateSeal,
    loader: Callable[[], pd.DataFrame | pd.Series],
) -> pd.DataFrame | pd.Series:
    """Enforce the observable ordering boundary before opening Storm."""

    if not seal.prediction_path.is_file() or not seal.manifest_path.is_file():
        raise TopologyExperimentError("Le candidat doit exister avant Storm.")
    if _sha256(seal.prediction_path) != seal.prediction_sha256:
        raise TopologyExperimentError("Le candidat a change apres son gel.")
    if _sha256(seal.manifest_path) != seal.manifest_sha256:
        raise TopologyExperimentError("Le manifeste du candidat a change.")
    return loader()


def _assert_experiment_output(
    output: str | Path,
    *,
    project_root: str | Path,
) -> Path:
    root = Path(project_root).expanduser().resolve()
    experiments = (root / "runs" / "experiments").resolve()
    destination = Path(output).expanduser()
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    if destination == experiments or not destination.is_relative_to(experiments):
        raise TopologyExperimentError(
            f"La sortie doit etre un sous-dossier de {experiments}."
        )
    if any(part.casefold() == "live" for part in destination.parts):
        raise TopologyExperimentError("Une experience ne peut jamais publier dans live.")
    return destination


def atomic_experiment_publish(
    output: str | Path,
    *,
    project_root: str | Path,
    writer: Callable[[Path], Any],
) -> Path:
    """Build in a sibling staging directory and rename exactly once."""

    destination = _assert_experiment_output(output, project_root=project_root)
    if destination.exists():
        raise FileExistsError(
            f"Refus d'ecraser une experience existante: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{uuid4().hex}"
    if staging.exists():
        raise RuntimeError(f"Collision staging inattendue: {staging}")
    staging.mkdir(parents=False, exist_ok=False)
    try:
        writer(staging)
        if destination.exists():
            raise FileExistsError(
                f"Collision avant publication atomique: {destination}"
            )
        os.replace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination


@dataclass(frozen=True)
class ZoneWindowData:
    """Outcome, autonomous baseline and both causal topology contexts."""

    actual: pd.Series
    base_predictions: pd.DataFrame
    contexts: Mapping[int, pd.DataFrame]


def _contract_model_parameters(contract: TopologyExperimentContract) -> dict[str, object]:
    model = contract.model
    result: dict[str, object] = {
        "learning_rate": model.learning_rate,
        "max_iter": model.max_iter,
        "max_leaf_nodes": model.max_leaf_nodes,
        "min_samples_leaf": model.min_samples_leaf,
        "l2_regularization": model.l2_regularization,
        "correction_clip": model.correction_clip_eur_mwh,
        "min_training_rows": model.min_training_rows,
    }
    if result != dict(FROZEN_MODEL_PARAMETERS):
        raise TopologyExperimentError("Le contrat modele differe des valeurs figees.")
    return result


def load_experiment_config(
    path: str | Path,
    *,
    zones: Sequence[str] = SUPPORTED_ZONES,
    project_root: str | Path = PROJECT_ROOT,
    allow_existing_output: bool = False,
    verify_hashes: bool = True,
) -> ExperimentConfig:
    """Load the strict read-only contract without inspecting formal outcomes."""

    source = Path(path).expanduser().resolve()
    contract = load_topology_contract(
        source,
        project_root=project_root,
        verify_hashes=verify_hashes,
    )
    selected = _normalise_zones(zones)
    if any(zone not in contract.sources for zone in selected):
        raise TopologyExperimentError("Une zone selectionnee manque au contrat.")
    output = _assert_experiment_output(
        contract.output_directory,
        project_root=contract.project_root,
    )
    if output.exists() and not allow_existing_output:
        raise FileExistsError(f"Experience deja publiee: {output}")
    frozen_at = datetime.fromtimestamp(
        source.stat().st_mtime,
        tz=timezone.utc,
    ).isoformat()
    # The contract itself rejects every protocol value that differs from this
    # revision.  The normalized wrapper merely makes those values convenient
    # to the runner and records the immutable YAML digest/timestamp.
    return ExperimentConfig(
        source_path=source,
        experiment_id=contract.experiment_id,
        output_directory=output,
        zones=selected,
        split_days=dict(DEFAULT_SPLIT_DAYS),
        gate_minimum_gain=contract.gates.minimum_mae_gain_eur_mwh,
        bootstrap_samples=contract.gates.bootstrap_samples,
        bootstrap_seed=contract.gates.bootstrap_seed,
        model_parameters=_contract_model_parameters(contract),
        scale_grid=tuple(contract.model.correction_scale_grid),
        blend_grid_step=contract.blend_weight_grid_step,
        protocol_frozen_at_utc=frozen_at,
        config_sha256=contract.source_sha256,
        raw={
            "schema_version": contract.schema_version,
            "feature_schema_sha256": contract.feature_schema_sha256,
            "selection_min_gain_eur_mwh": contract.selection_min_gain_eur_mwh,
        },
        contract=contract,
    )


def _forecast_origins_for_delivery(
    index: pd.DatetimeIndex,
    *,
    timezone_name: str,
) -> pd.DatetimeIndex:
    """Return the governed previous-local-day 08:00 cutoff for each hour."""

    local_days = index.tz_convert(timezone_name).date
    return pd.DatetimeIndex(
        [
            (
                pd.Timestamp(day - timedelta(days=1), tz=timezone_name)
                .replace(hour=8)
                .tz_convert("UTC")
            )
            for day in local_days
        ],
        name="forecast_origin_utc",
    )


def audit_experiment_sources(
    config: ExperimentConfig,
) -> dict[str, dict[str, object]]:
    """Audit sealed inputs without decoding any outcome/prediction value.

    The strict contract loader has already authenticated every declared file.
    This second, protocol-aware audit opens only harmless OOF metadata
    (timestamps and forecast origins), PIT forecast inputs and target-cache
    timestamps.  In particular it never asks pandas to decode ``actual`` or
    any ``__q*`` column, including in B1/B2/final.
    """

    result: dict[str, dict[str, object]] = {}
    contract = config.contract
    for zone in config.zones:
        source = contract.sources[zone]
        splits = _splits_for_contract(contract, zone)
        expected = splits.all_index

        backtest_metadata = _read_csv_window_without_future_outcomes(
            source.backtest_file,
            timestamp_column="delivery_start_utc",
            columns=("forecast_origin_utc",),
            expected_index=expected,
        )
        origins = pd.DatetimeIndex(
            pd.to_datetime(
                backtest_metadata["forecast_origin_utc"],
                utc=True,
                errors="raise",
            )
        )
        expected_origins = _forecast_origins_for_delivery(
            expected,
            timezone_name=source.timezone,
        )
        origin_violations = int(
            np.count_nonzero(origins.asi8 != expected_origins.asi8)
        )
        if origin_violations:
            raise TopologyExperimentError(
                f"{zone}: violations du cutoff forecast_origin={origin_violations}."
            )

        pit = _read_csv_window_without_future_outcomes(
            source.aligned_inputs_file,
            timestamp_column="timestamp",
            columns=tuple(contract.residual_load_columns.values()),
            expected_index=expected,
        )
        pit_coverage = {
            alias: float(
                np.isfinite(
                    pd.to_numeric(pit[column], errors="coerce").to_numpy(dtype=float)
                ).mean()
            )
            for alias, column in contract.residual_load_columns.items()
        }
        below = {
            alias: value
            for alias, value in pit_coverage.items()
            if value < contract.minimum_pit_coverage
        }
        if below:
            raise TopologyExperimentError(
                f"{zone}: couverture PIT sous {contract.minimum_pit_coverage:.2f}: "
                f"{below}."
            )

        # Timestamp-only cache inspection proves the physical t-24 mask and
        # coverage without reading the future price/outcome values themselves.
        target_index = _timestamp_positions(
            source.target_cache,
            timestamp_column="timestamp",
        )
        lag_index = expected - pd.Timedelta(hours=24)
        target_locations = target_index.get_indexer(expected)
        lag_locations = target_index.get_indexer(lag_index)
        delivery_days = expected.tz_convert(source.timezone).date
        lag_days = lag_index.tz_convert(source.timezone).date
        strict_prior_day = np.asarray(
            [
                source_day < delivery_day
                for source_day, delivery_day in zip(lag_days, delivery_days)
            ],
            dtype=bool,
        )
        if any(
            source_day > delivery_day
            for source_day, delivery_day in zip(lag_days, delivery_days)
        ):
            raise TopologyExperimentError(
                f"{zone}: le lag prix pointe vers un jour local futur."
            )
        if bool((target_locations < 0).any()):
            raise TopologyExperimentError(
                f"{zone}: timeline de livraison absente du cache cible scelle."
            )
        if bool((lag_locations[strict_prior_day] < 0).any()):
            raise TopologyExperimentError(
                f"{zone}: timestamps t-24 causaux absents du cache cible scelle."
            )
        masked_same_day_hours = int(np.count_nonzero(~strict_prior_day))
        result[zone] = {
            "checksum_manifest_sha256": source.checksum_manifest_sha256,
            "backtest_sha256": source.backtest_sha256,
            "aligned_inputs_sha256": source.aligned_inputs_sha256,
            "run_manifest_sha256": source.run_manifest_sha256,
            "feature_manifest_sha256": source.feature_manifest_sha256,
            "recipe_sha256": source.recipe_sha256,
            "metrics_sha256": source.metrics_sha256,
            "target_cache_sha256": source.target_cache_sha256,
            "hashes_verified_by_contract_loader": True,
            "start_local_day": splits.days["seed"][0],
            "end_local_day": splits.days["final"][-1],
            "n_days": 365,
            "n_hours": int(len(expected)),
            "forecast_origin_violations": origin_violations,
            "pit_coverage": pit_coverage,
            "minimum_pit_coverage": contract.minimum_pit_coverage,
            "price_da_lag24h_coverage": float(np.mean(strict_prior_day)),
            "price_da_lag24h_strict_prior_local_day": True,
            "price_lag_strict_prior_day_hours": int(
                np.count_nonzero(strict_prior_day)
            ),
            "price_lag_masked_same_day_hours": masked_same_day_hours,
            "target_delivery_timestamp_coverage": float(
                np.mean(target_locations >= 0)
            ),
            "target_lag_timestamp_coverage_on_unmasked_hours": float(
                np.mean(lag_locations[strict_prior_day] >= 0)
            ),
            "outcome_value_columns_opened": False,
            "prediction_value_columns_opened": False,
            "target_cache_value_column_opened": False,
            "formal_outcomes_unopened_during_metadata_audit": True,
            "storm_used_as_prediction_input": False,
            "mkonline_used_as_topology_input": False,
        }
    return result


def _timestamp_positions(path: Path, *, timestamp_column: str) -> pd.DatetimeIndex:
    raw = pd.read_csv(path, usecols=[timestamp_column])
    values = pd.DatetimeIndex(
        pd.to_datetime(raw[timestamp_column], utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    if values.has_duplicates or not values.is_monotonic_increasing:
        raise TopologyExperimentError(f"Timeline source invalide: {path}")
    return values


def _read_csv_window_without_future_outcomes(
    path: Path,
    *,
    timestamp_column: str,
    columns: Sequence[str],
    expected_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Read outcomes only through the requested end row of a sorted CSV.

    Timestamp discovery is harmless protocol metadata.  The second read uses
    ``nrows`` ending exactly at the requested block, so a B1 call cannot decode
    B2/final target or prediction columns from the gzip stream.
    """

    source_index = _timestamp_positions(path, timestamp_column=timestamp_column)
    locations = source_index.get_indexer(expected_index)
    if bool((locations < 0).any()) or not np.array_equal(
        locations,
        np.arange(locations[0], locations[0] + len(expected_index)),
    ):
        raise TopologyExperimentError(f"{path}: fenetre physique absente ou discontinue.")
    requested = tuple(dict.fromkeys((timestamp_column, *columns)))
    raw = pd.read_csv(
        path,
        usecols=list(requested),
        nrows=int(locations[-1] + 1),
    )
    index = pd.DatetimeIndex(
        pd.to_datetime(raw.pop(timestamp_column), utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    raw.index = index
    selected = raw.reindex(expected_index)
    if not selected.index.equals(expected_index) or len(selected) != len(expected_index):
        raise TopologyExperimentError(f"{path}: reindexation de fenetre impossible.")
    return selected


def _load_zone_window(
    contract: TopologyExperimentContract,
    *,
    zone: str,
    expected_index: pd.DatetimeIndex,
) -> ZoneWindowData:
    source = contract.sources[zone]
    backtest = _read_csv_window_without_future_outcomes(
        source.backtest_file,
        timestamp_column="delivery_start_utc",
        columns=(
            "actual",
            "residual_corrected__q10",
            "residual_corrected__q50",
            "residual_corrected__q90",
        ),
        expected_index=expected_index,
    )
    actual = pd.to_numeric(backtest["actual"], errors="coerce").astype(float)
    base = backtest.rename(
        columns={
            "residual_corrected__q10": "q10",
            "residual_corrected__q50": "q50",
            "residual_corrected__q90": "q90",
        }
    ).loc[:, list(QUANTILE_COLUMNS)]
    base = _validate_quantiles(base, expected_index=expected_index, name=f"{zone}_base")
    if not np.isfinite(actual.to_numpy(dtype=float)).all():
        raise TopologyExperimentError(f"{zone}: actual non fini sur la fenetre ouverte.")

    extended_index = pd.date_range(
        expected_index[0] - pd.Timedelta(hours=24),
        expected_index[-1],
        freq="h",
        name="delivery_start_utc",
    )
    residual_columns = tuple(contract.residual_load_columns.values())
    residual_raw = _read_csv_window_without_future_outcomes(
        source.aligned_inputs_file,
        timestamp_column="timestamp",
        columns=residual_columns,
        expected_index=extended_index,
    )
    residual = pd.DataFrame(
        {
            code: pd.to_numeric(
                residual_raw[contract.residual_load_columns[code]],
                errors="coerce",
            )
            for code in SUPPORTED_ZONES
        },
        index=extended_index,
    )
    # Only the physical t-24 source rows are decoded from the target caches.
    # Contemporary stage outcomes are deliberately left unopened here; the
    # core shifts these history values onto the requested delivery hours.
    price_source_index = expected_index - pd.Timedelta(hours=24)
    price_frame = pd.DataFrame(
        np.nan,
        index=extended_index,
        columns=list(SUPPORTED_ZONES),
    )
    for price_zone in SUPPORTED_ZONES:
        price_raw = _read_csv_window_without_future_outcomes(
            contract.sources[price_zone].target_cache,
            timestamp_column="timestamp",
            columns=("value",),
            expected_index=price_source_index,
        )
        price_frame.loc[price_source_index, price_zone] = pd.to_numeric(
            price_raw["value"], errors="coerce"
        ).to_numpy(dtype=float)
    contexts: dict[int, pd.DataFrame] = {}
    for radius in (0, 1):
        full = build_topology_context(
            residual,
            price_frame,
            target_zone=zone,
            radius=radius,
            timezone=source.timezone,
        )
        contexts[radius] = _slice_context(full, expected_index)
    return ZoneWindowData(actual=actual, base_predictions=base, contexts=contexts)


def _splits_for_contract(
    contract: TopologyExperimentContract,
    zone: str,
) -> ProtocolSplits:
    timezone_name = contract.sources[zone].timezone
    start = pd.Timestamp(contract.start_local_day, tz=timezone_name).tz_convert("UTC")
    stop = pd.Timestamp(
        contract.start_local_day + pd.Timedelta(days=365),
        tz=timezone_name,
    ).tz_convert("UTC")
    index = pd.date_range(
        start,
        stop,
        freq="h",
        inclusive="left",
        name="delivery_start_utc",
    )
    return build_protocol_splits(
        index,
        timezone_name=timezone_name,
        split_days=DEFAULT_SPLIT_DAYS,
    )


def _segment_payload(metrics: SegmentMetrics) -> dict[str, object]:
    return asdict(metrics)


def _gate_payload(gate: GateResult) -> dict[str, object]:
    return {
        "passes": gate.passes,
        "reasons": list(gate.reasons),
        "checks": dict(gate.checks),
        **asdict(gate.metrics),
    }


def _context_pit_coverage(context: pd.DataFrame) -> dict[str, float]:
    column = "topology__residual_load__pool_coverage"
    values = pd.to_numeric(context[column], errors="coerce")
    finite = values[np.isfinite(values.to_numpy(dtype=float))]
    if finite.empty:
        return {"selected_pool_mean": 0.0, "selected_pool_min": 0.0}
    return {
        "selected_pool_mean": float(finite.mean()),
        "selected_pool_min": float(finite.min()),
    }


def _opened_candidate_frame(
    *,
    development: DevelopmentProtocolResult,
    autonomous: AutonomousProtocolResult,
    splits: ProtocolSplits,
    formal_data: Mapping[str, ZoneWindowData],
    blend: BlendProtocolResult | None = None,
    production_blend: pd.DataFrame | None = None,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for stage in autonomous.opened_stages:
        if stage == "a":
            index = splits.indices["a"]
            actual = development.training_actual.loc[index]
            base = development.training_base.loc[index]
        elif stage == "development":
            index = splits.indices["development"]
            actual = development.training_actual.loc[index]
            base = development.training_base.loc[index]
        else:
            block = formal_data[stage]
            index = splits.indices[stage]
            actual = block.actual
            base = block.base_predictions
        candidate = autonomous.selected_predictions.loc[index]
        frame = pd.DataFrame(
            {
                "delivery_start_utc": index,
                "stage": stage,
                "actual": actual.to_numpy(dtype=float),
                "residual_corrected__q10": base["q10"].to_numpy(dtype=float),
                "residual_corrected__q50": base["q50"].to_numpy(dtype=float),
                "residual_corrected__q90": base["q90"].to_numpy(dtype=float),
                "topology_autonomous__q10": candidate["q10"].to_numpy(dtype=float),
                "topology_autonomous__q50": candidate["q50"].to_numpy(dtype=float),
                "topology_autonomous__q90": candidate["q90"].to_numpy(dtype=float),
            }
        )
        if blend is not None and stage in blend.opened_stages:
            blend_values = blend.predictions.loc[index]
            frame["topology_mkonline_blend__q10"] = blend_values["q10"].to_numpy(
                dtype=float
            )
            frame["topology_mkonline_blend__q50"] = blend_values["q50"].to_numpy(
                dtype=float
            )
            frame["topology_mkonline_blend__q90"] = blend_values["q90"].to_numpy(
                dtype=float
            )
        if production_blend is not None:
            current = production_blend.loc[index]
            frame["mkonline_blend__q10"] = current["q10"].to_numpy(dtype=float)
            frame["mkonline_blend__q50"] = current["q50"].to_numpy(dtype=float)
            frame["mkonline_blend__q90"] = current["q90"].to_numpy(dtype=float)
        pieces.append(frame)
    result = pd.concat(pieces, ignore_index=True)
    delivery = pd.DatetimeIndex(pd.to_datetime(result["delivery_start_utc"], utc=True))
    result["forecast_origin_utc"] = _forecast_origins_for_delivery(
        delivery,
        timezone_name=splits.timezone,
    )
    return result


def _display_stage_from_opened(opened_stages: Sequence[str]) -> str:
    """Return the last stage actually opened by the sequential protocol."""

    if not opened_stages:
        raise TopologyExperimentError("Aucun stage ouvert pour le rapport.")
    return str(opened_stages[-1])


def _combine_full_training_data(
    development: DevelopmentProtocolResult,
    formal_data: Mapping[str, ZoneWindowData],
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    actual_parts = [development.training_actual]
    base_parts = [development.training_base]
    context_parts = [development.training_context]
    for stage in GATED_STAGES:
        if stage not in formal_data:
            break
        block = formal_data[stage]
        actual_parts.append(block.actual)
        base_parts.append(block.base_predictions)
        context_parts.append(block.contexts[development.selected_radius])
    actual = pd.concat(actual_parts)
    base = pd.concat(base_parts)
    attrs = dict(development.training_context.attrs)
    context = pd.concat(context_parts)
    context.attrs = attrs
    return actual, base, context


def _load_current_context(
    *,
    contract: TopologyExperimentContract,
    zone: str,
    archive: Path,
    forecast_index: pd.DatetimeIndex,
    radius: int,
) -> pd.DataFrame:
    extended = pd.date_range(
        forecast_index[0] - pd.Timedelta(hours=24),
        forecast_index[-1],
        freq="h",
        name="delivery_start_utc",
    )
    # ``aligned_inputs`` is intentionally history-only in a live archive and
    # stops at forecast_start - 1h.  The runner-produced future-covariate
    # artifact is the causal source that carries the five PIT residual-load
    # forecasts across D+1; price history remains loaded separately below from
    # the sealed target caches.
    future_covariates = (
        archive / "inputs" / "model_covariates_with_future.csv.gz"
    )
    residual_raw = _read_csv_window_without_future_outcomes(
        future_covariates,
        timestamp_column="timestamp",
        columns=tuple(contract.residual_load_columns.values()),
        expected_index=extended,
    )
    residual = pd.DataFrame(
        {
            code: pd.to_numeric(
                residual_raw[contract.residual_load_columns[code]], errors="coerce"
            )
            for code in SUPPORTED_ZONES
        },
        index=extended,
    )
    history = pd.date_range(
        extended[0],
        forecast_index[0] - pd.Timedelta(hours=1),
        freq="h",
        name="delivery_start_utc",
    )
    prices = pd.DataFrame(np.nan, index=extended, columns=list(SUPPORTED_ZONES))
    for price_zone in SUPPORTED_ZONES:
        raw = _read_csv_window_without_future_outcomes(
            contract.sources[price_zone].target_cache,
            timestamp_column="timestamp",
            columns=("value",),
            expected_index=history,
        )
        prices.loc[history, price_zone] = pd.to_numeric(
            raw["value"], errors="coerce"
        ).to_numpy(dtype=float)
    full = build_topology_context(
        residual,
        prices,
        target_zone=zone,
        radius=radius,
        timezone=contract.sources[zone].timezone,
    )
    return _slice_context(full, forecast_index)


def _predict_current_autonomous(
    *,
    config: ExperimentConfig,
    development: DevelopmentProtocolResult,
    autonomous: AutonomousProtocolResult,
    formal_data: Mapping[str, ZoneWindowData],
    current_base: pd.DataFrame,
    current_context: pd.DataFrame,
) -> pd.DataFrame:
    if not autonomous.promoted:
        return current_base.copy()
    train_actual, train_base, train_context = _combine_full_training_data(
        development,
        formal_data,
    )
    model = _new_corrector(
        config.model_parameters,
        correction_scale=autonomous.selected_scale,
    )
    model.fit(train_context, train_actual, train_base)
    if model.hyperparameter_sha256() != autonomous.hyperparameter_sha256:
        raise TopologyExperimentError("Le modele current differe de la recette scellee.")
    return _validate_quantiles(
        model.predict(current_context, current_base),
        expected_index=current_base.index,
        name="current_topology_autonomous",
    )


def _load_storm_after_candidate(
    *,
    seal: CandidateSeal,
    zone: str,
    archive: Path,
    expected_index: pd.DatetimeIndex,
) -> tuple[pd.Series | None, dict[str, object]]:
    zone_key = str(zone).strip().upper()
    path = archive / "inputs" / "storm_dashboard_official_statistics.parquet"
    audit_path = archive / "statistics_history_audit.json"
    source_column = "storm_dashboard_official__q50"

    def loader() -> tuple[pd.Series | None, dict[str, object]]:
        if zone_key == "ES":
            return None, {
                "available": False,
                "reason": "no native dashboard contract for ES",
                "source_column": None,
                "benchmark_contract": "not_available_for_zone",
                "native_dashboard": False,
                "used_for_prediction": False,
                "loaded_after_candidate_sha256": seal.prediction_sha256,
            }
        if not path.is_file():
            return None, {
                "available": False,
                "reason": "no native dashboard snapshot for this zone",
                "source_column": None,
                "benchmark_contract": "native_dashboard_snapshot",
                "native_dashboard": False,
                "used_for_prediction": False,
                "loaded_after_candidate_sha256": seal.prediction_sha256,
            }
        if not audit_path.is_file():
            raise TopologyExperimentError(
                "Snapshot Storm dashboard present sans audit Statistics scelle."
            )
        try:
            source_history_audit = json.loads(
                audit_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise TopologyExperimentError(
                "Audit Statistics du snapshot Storm illisible."
            ) from exc
        if not isinstance(source_history_audit, Mapping):
            raise TopologyExperimentError("Audit Statistics Storm invalide.")
        if source_history_audit.get("storm_primary_report_benchmark") != source_column:
            raise TopologyExperimentError(
                "Le snapshot Storm officiel n'est pas le benchmark declare."
            )
        source_dashboard = source_history_audit.get("storm_dashboard")
        if not isinstance(source_dashboard, Mapping):
            sealed_benchmark = source_history_audit.get("sealed_benchmark")
            if isinstance(sealed_benchmark, Mapping):
                source_dashboard = sealed_benchmark.get("storm_dashboard")
        if not isinstance(source_dashboard, Mapping):
            raise TopologyExperimentError("Audit Storm dashboard natif absent.")
        declared_column = source_dashboard.get("column")
        if declared_column not in (None, source_column):
            raise TopologyExperimentError("Colonne Storm contraire a son audit scelle.")
        source_dst = source_dashboard.get("dst")
        if not isinstance(source_dst, Mapping):
            raise TopologyExperimentError("Audit DST Storm dashboard absent.")
        if (
            source_dst.get("interpolation") is not False
            or source_dst.get("strict_08_fallback") is not False
            or source_dst.get("native_actual_missing_matches_allowed") is not True
        ):
            raise TopologyExperimentError("Politique DST Storm non native ou non exacte.")
        allowed_raw = source_dst.get("native_allowed_missing_utc", [])
        if not isinstance(allowed_raw, list):
            raise TopologyExperimentError("Allow-list DST Storm invalide.")
        allowed_missing = {
            pd.Timestamp(value).tz_convert("UTC")
            if pd.Timestamp(value).tzinfo is not None
            else pd.Timestamp(value, tz="UTC")
            for value in allowed_raw
        }
        raw = pd.read_parquet(path)
        if "delivery_start_utc" not in raw or source_column not in raw:
            return None, {
                "available": False,
                "reason": "native dashboard snapshot schema unsupported",
                "source_path": str(path),
                "source_sha256": _sha256(path),
                "source_column": source_column,
                "benchmark_contract": "native_dashboard_snapshot",
                "native_dashboard": False,
                "used_for_prediction": False,
                "loaded_after_candidate_sha256": seal.prediction_sha256,
            }
        index = pd.DatetimeIndex(
            pd.to_datetime(raw["delivery_start_utc"], utc=True, errors="raise"),
            name="delivery_start_utc",
        )
        if index.has_duplicates or not index.is_monotonic_increasing:
            raise TopologyExperimentError(
                "Timeline du snapshot Storm dupliquee ou non ordonnee."
            )
        values = pd.Series(
            pd.to_numeric(raw[source_column], errors="coerce").to_numpy(dtype=float),
            index=index,
            name=source_column,
        ).reindex(expected_index)
        finite = np.isfinite(values.to_numpy(dtype=float))
        missing = set(values.index[~finite])
        allowed_in_window = allowed_missing.intersection(set(expected_index))
        for timestamp in allowed_in_window:
            local = timestamp.tz_convert(ZONE_TIMEZONES[zone_key])
            transition_day = local_delivery_day_index(
                str(local.date()),
                timezone=ZONE_TIMEZONES[zone_key],
            )
            if len(transition_day) != 25 or local.hour != 2 or local.fold != 0:
                raise TopologyExperimentError(
                    "L'allow-list Storm contient un trou qui n'est pas le "
                    "premier fold de l'heure DST automnale."
                )
        if missing != allowed_in_window:
            raise TopologyExperimentError(
                "Les trous Storm ne correspondent pas exactement a l'allow-list DST "
                f"native: missing={sorted(map(str, missing))}, "
                f"allowed={sorted(map(str, allowed_in_window))}."
            )
        coverage = float(finite.mean())
        source_sha = _sha256(path)
        declared_sha = source_dashboard.get("normalized_artifact_sha256")
        if declared_sha not in (None, source_sha):
            raise TopologyExperimentError("Le hash Storm contredit son audit scelle.")
        return values, {
            "available": bool(coverage > 0.0),
            "coverage": coverage,
            "source_path": str(path),
            "source_sha256": source_sha,
            "source_audit_path": str(audit_path),
            "source_audit_sha256": _sha256(audit_path),
            "source_column": source_column,
            "benchmark_contract": "native_dashboard_snapshot",
            "native_dashboard": True,
            "native_allowed_missing_utc": [
                timestamp.isoformat()
                for timestamp in sorted(allowed_in_window)
            ],
            "native_allowed_missing_hours": len(allowed_in_window),
            "native_actual_missing_matches_allowed": True,
            "interpolation": False,
            "strict_08_fallback": False,
            "used_for_prediction": False,
            "loaded_after_candidate_sha256": seal.prediction_sha256,
        }

    loaded = load_comparator_after_seal(seal, loader)
    assert isinstance(loaded, tuple)
    return loaded


def _storm_evaluation_metrics(
    candidate_frame: pd.DataFrame,
    storm: pd.Series,
    *,
    stage: str,
    candidate_model: str,
    baseline_model: str,
    timezone_name: str,
) -> dict[str, object]:
    """Score a sealed candidate against native Storm without any gate use."""

    selected = candidate_frame.loc[candidate_frame["stage"].eq(stage)].copy()
    if selected.empty:
        raise TopologyExperimentError(f"Storm: stage ouvert absent: {stage}.")
    index = pd.DatetimeIndex(
        pd.to_datetime(selected["delivery_start_utc"], utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    candidate_column = f"{candidate_model}__q50"
    baseline_column = f"{baseline_model}__q50"
    missing_columns = [
        column
        for column in ("actual", candidate_column, baseline_column)
        if column not in selected
    ]
    if missing_columns:
        raise TopologyExperimentError(
            f"Storm: colonnes de comparaison absentes={missing_columns}."
        )
    paired = pd.DataFrame(
        {
            "actual": pd.to_numeric(selected["actual"], errors="coerce").to_numpy(
                dtype=float
            ),
            "candidate": pd.to_numeric(
                selected[candidate_column], errors="coerce"
            ).to_numpy(dtype=float),
            "baseline": pd.to_numeric(
                selected[baseline_column], errors="coerce"
            ).to_numpy(dtype=float),
            "storm": pd.to_numeric(storm.reindex(index), errors="coerce").to_numpy(
                dtype=float
            ),
        },
        index=index,
    )
    finite = np.isfinite(paired.to_numpy(dtype=float)).all(axis=1)
    evaluated = paired.loc[finite]
    if evaluated.empty:
        raise TopologyExperimentError(
            f"Storm: aucune heure appariee pour le stage {stage}."
        )
    errors = pd.DataFrame(
        {
            "candidate": np.abs(evaluated["actual"] - evaluated["candidate"]),
            "baseline": np.abs(evaluated["actual"] - evaluated["baseline"]),
            "storm": np.abs(evaluated["actual"] - evaluated["storm"]),
        },
        index=evaluated.index,
    )
    local_days = pd.Index(
        evaluated.index.tz_convert(timezone_name).strftime("%Y-%m-%d"),
        name="local_day",
    )
    daily = errors.set_axis(local_days).groupby(level=0, sort=False).mean()
    candidate_mae = float(errors["candidate"].mean())
    baseline_mae = float(errors["baseline"].mean())
    storm_mae = float(errors["storm"].mean())
    return {
        "stage": stage,
        "candidate_model": candidate_model,
        "baseline_model": baseline_model,
        "benchmark_model": "storm_dashboard_official",
        "n_expected_hours": int(len(selected)),
        "n_paired_hours": int(len(evaluated)),
        "n_paired_days": int(len(daily)),
        "pairing_coverage": float(len(evaluated) / len(selected)),
        "candidate_mae": candidate_mae,
        "baseline_mae": baseline_mae,
        "storm_mae": storm_mae,
        "candidate_gain_vs_storm_eur_mwh": storm_mae - candidate_mae,
        "baseline_gain_vs_storm_eur_mwh": storm_mae - baseline_mae,
        "candidate_vs_storm_daily_win_rate": float(
            np.mean(daily["candidate"] < daily["storm"])
        ),
        "used_for_gate": False,
        "used_for_selection_or_tuning": False,
        "used_for_promotion": False,
    }


def _write_artifact_checksums(
    directory: Path,
    *,
    declared_directory: Path | None = None,
) -> Path:
    artifacts = []
    for path in sorted(directory.rglob("*"), key=lambda item: str(item).casefold()):
        if not path.is_file() or path.name == "artifact_checksums.json":
            continue
        artifacts.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "role": "experiment_artifact",
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    output = directory / "artifact_checksums.json"
    _write_json(
        output,
        {
            "algorithm": "sha256",
            "output_directory": str(declared_directory or directory),
            "artifacts": artifacts,
        },
    )
    return output


def _read_json_object(path: Path, *, name: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TopologyExperimentError(f"{name} illisible: {path}") from exc
    if not isinstance(payload, dict):
        raise TopologyExperimentError(f"{name} doit contenir un objet JSON.")
    return payload


def _validate_artifact_checksums(directory: Path) -> str:
    """Rehash a complete immutable directory without trusting path entries."""

    root = directory.expanduser().resolve()
    manifest_path = root / "artifact_checksums.json"
    payload = _read_json_object(manifest_path, name="manifest de checksums")
    if payload.get("algorithm") != "sha256":
        raise TopologyExperimentError("Le manifest doit utiliser sha256.")
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise TopologyExperimentError("artifacts doit etre une liste.")
    declared: dict[str, tuple[str, int]] = {}
    for position, raw in enumerate(raw_artifacts):
        if not isinstance(raw, Mapping):
            raise TopologyExperimentError(
                f"Entree artifact_checksums invalide a l'index {position}."
            )
        relative_text = raw.get("path")
        digest = raw.get("sha256")
        size = raw.get("size_bytes")
        if not isinstance(relative_text, str) or not relative_text.strip():
            raise TopologyExperimentError("Chemin artifact_checksums invalide.")
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise TopologyExperimentError(
                f"Chemin artifact_checksums non sur: {relative_text}"
            )
        normalized = relative.as_posix()
        if normalized in declared:
            raise TopologyExperimentError(
                f"Artefact duplique dans le manifest: {normalized}"
            )
        if not isinstance(digest, str) or len(digest) != 64:
            raise TopologyExperimentError(f"SHA-256 invalide pour {normalized}.")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise TopologyExperimentError(f"Taille invalide pour {normalized}.")
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise TopologyExperimentError(f"Artefact absent/non sur: {normalized}")
        if target.is_symlink():
            raise TopologyExperimentError(f"Lien symbolique interdit: {normalized}")
        if target.stat().st_size != size or _sha256(target) != digest.lower():
            raise TopologyExperimentError(f"Artefact modifie: {normalized}")
        declared[normalized] = (digest.lower(), size)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_checksums.json"
    }
    if actual != set(declared):
        missing = sorted(actual.difference(declared))
        stale = sorted(set(declared).difference(actual))
        raise TopologyExperimentError(
            "Couverture artifact_checksums incomplete: "
            f"non_declares={missing}, declares_absents={stale}."
        )
    return _sha256(manifest_path)


def _assert_existing_experiment_directory(
    directory: str | Path,
    *,
    project_root: Path,
    name: str,
) -> Path:
    resolved = Path(directory).expanduser()
    if not resolved.is_absolute():
        resolved = project_root / resolved
    resolved = resolved.resolve()
    experiments = (project_root / "runs" / "experiments").resolve()
    if resolved == experiments or not resolved.is_relative_to(experiments):
        raise TopologyExperimentError(f"{name} doit rester sous {experiments}.")
    if any(part.casefold() == "live" for part in resolved.parts):
        raise TopologyExperimentError(f"{name} ne peut jamais etre sous runs/live.")
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return resolved


def audit_published_calibration(
    config: ExperimentConfig,
    calibration_dir: str | Path,
    *,
    expected_manifest_sha256: str = CALIBRATION_V1_ARTIFACT_MANIFEST_SHA256,
) -> PublishedCalibration:
    """Validate the published five-zone decision before operational use."""

    root = _assert_existing_experiment_directory(
        calibration_dir,
        project_root=config.contract.project_root,
        name="calibration_dir",
    )
    manifest_path = root / "artifact_checksums.json"
    if _sha256(manifest_path) != expected_manifest_sha256:
        raise TopologyExperimentError(
            "Trust anchor calibration invalide; artifact et manifest reecrits refuses."
        )
    artifact_manifest_sha = _validate_artifact_checksums(root)
    experiment = _read_json_object(
        root / "experiment_manifest.json",
        name="experiment_manifest",
    )
    if experiment.get("experiment_id") != config.experiment_id:
        raise TopologyExperimentError("Experiment id de calibration incompatible.")
    if experiment.get("config_sha256") != config.config_sha256:
        raise TopologyExperimentError("La calibration ne correspond pas au config SHA.")
    if experiment.get("production_changed") is not False:
        raise TopologyExperimentError("La calibration ne doit pas modifier production.")
    calibrated_zones = experiment.get("zones")
    if calibrated_zones != list(SUPPORTED_ZONES):
        raise TopologyExperimentError(
            "Le bundle operationnel exige la calibration exacte des cinq zones."
        )

    evaluations: dict[str, Mapping[str, object]] = {}
    recipes: dict[str, Mapping[str, object]] = {}
    candidate_seals: dict[str, CandidateSeal] = {}
    for zone in SUPPORTED_ZONES:
        zone_root = root / zone.lower()
        _validate_artifact_checksums(zone_root / "autonomous")
        evaluation = _read_json_object(
            zone_root / "autonomous" / "topology_evaluation.json",
            name=f"{zone} topology_evaluation",
        )
        variants = evaluation.get("variants")
        if not isinstance(variants, Mapping):
            raise TopologyExperimentError(f"{zone}: variants absent.")
        autonomous = variants.get("autonomous")
        if not isinstance(autonomous, Mapping):
            raise TopologyExperimentError(f"{zone}: decision autonome absente.")
        opened = autonomous.get("opened_stages")
        if not isinstance(opened, list) or not opened:
            raise TopologyExperimentError(f"{zone}: opened_stages invalide.")
        if autonomous.get("sequential_decision_complete") is not True:
            raise TopologyExperimentError(f"{zone}: decision sequentielle incomplete.")
        promoted = autonomous.get("promoted")
        expected_promotion = zone == "BE"
        if promoted is not expected_promotion:
            raise TopologyExperimentError(
                f"{zone}: politique operationnelle attend promoted={expected_promotion}."
            )
        expected_decision = "promoted" if expected_promotion else "fallback_identity"
        if autonomous.get("promotion_decision") != expected_decision:
            raise TopologyExperimentError(
                f"{zone}: promotion_decision incompatible avec {expected_decision}."
            )
        if expected_promotion:
            if opened != ["a", "development", "b1", "b2", "final"]:
                raise TopologyExperimentError("BE: la promotion exige le final complet.")
            gates = autonomous.get("gates")
            if not isinstance(gates, Mapping) or any(
                not isinstance(gates.get(stage), Mapping)
                or gates[stage].get("passes") is not True
                for stage in GATED_STAGES
            ):
                raise TopologyExperimentError(
                    "BE: les gates B1/B2/final doivent toutes etre promues."
                )
        elif autonomous.get("recommended_variant") not in (None, "autonomous"):
            raise TopologyExperimentError(f"{zone}: fallback autonome inattendu.")

        recipe_path = zone_root / "topology_recipe_seal.json"
        recipe = _read_json_object(recipe_path, name=f"{zone} recipe seal")
        recipe_sha = _sha256(recipe_path)
        if autonomous.get("recipe_seal_sha256") != recipe_sha:
            raise TopologyExperimentError(f"{zone}: SHA recette non relie au sidecar.")
        if recipe.get("config_sha256") != config.config_sha256:
            raise TopologyExperimentError(f"{zone}: recette/config SHA incoherent.")
        for key in ("selected_arm", "selected_radius", "selected_scale"):
            if recipe.get(key) != autonomous.get(key):
                raise TopologyExperimentError(f"{zone}: recette diverge sur {key}.")
        if zone == "BE" and (
            recipe.get("selected_radius") != 1
            or not math.isclose(
                float(recipe.get("selected_scale", math.nan)),
                0.25,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise TopologyExperimentError("BE: recette promue attendue radius1/scale0.25.")
        if recipe.get("model_hyperparameters_sha256") != evaluation.get(
            "model_hyperparameters_sha256"
        ):
            raise TopologyExperimentError(f"{zone}: hash hyperparametres divergent.")

        seal_path = zone_root / "candidate_prediction_seal.json"
        seal_payload = _read_json_object(
            seal_path,
            name=f"{zone} candidate seal",
        )
        candidate_path = zone_root / "candidate_predictions.csv.gz"
        candidate_sha = seal_payload.get("candidate_sha256")
        if not isinstance(candidate_sha, str) or candidate_sha != _sha256(candidate_path):
            raise TopologyExperimentError(f"{zone}: candidat scelle modifie.")
        if autonomous.get("candidate_prediction_sha256") != candidate_sha:
            raise TopologyExperimentError(f"{zone}: candidat non relie au sidecar.")
        candidate = pd.read_csv(candidate_path, usecols=["stage"])
        observed_stages = list(dict.fromkeys(candidate["stage"].astype(str).tolist()))
        if observed_stages != opened:
            raise TopologyExperimentError(
                f"{zone}: stages candidats {observed_stages} != {opened}."
            )
        evaluations[zone] = evaluation
        recipes[zone] = recipe
        candidate_seals[zone] = CandidateSeal(
            prediction_path=candidate_path,
            manifest_path=seal_path,
            prediction_sha256=candidate_sha,
            manifest_sha256=_sha256(seal_path),
        )
    return PublishedCalibration(
        directory=root,
        artifact_manifest_sha256=artifact_manifest_sha,
        experiment_manifest=experiment,
        evaluations=evaluations,
        recipes=recipes,
        candidate_seals=candidate_seals,
    )


def _fit_fixed_operational_be_model(
    config: ExperimentConfig,
    calibration: PublishedCalibration,
) -> tuple[TopologyResidualCorrector, dict[str, object]]:
    """Refit the promoted BE recipe once; no selection or gate is rerun."""

    recipe = calibration.recipes["BE"]
    selected_radius = int(recipe["selected_radius"])
    selected_scale = float(recipe["selected_scale"])
    raw_parameters = recipe.get("model_parameters")
    if not isinstance(raw_parameters, Mapping):
        raise TopologyExperimentError("BE: model_parameters absents de la recette.")
    if dict(raw_parameters) != dict(config.model_parameters):
        raise TopologyExperimentError("BE: parametres recette/config divergents.")

    splits = _splits_for_contract(config.contract, "BE")
    full_index = splits.indices["seed"]
    for stage in ("a", "development", "b1", "b2", "final"):
        full_index = full_index.append(splits.indices[stage])
    training = _load_zone_window(
        config.contract,
        zone="BE",
        expected_index=full_index,
    )
    model = _new_corrector(
        raw_parameters,
        correction_scale=selected_scale,
    )
    if not isinstance(model, TopologyResidualCorrector):
        raise TopologyExperimentError(
            "Le bundle operationnel exige TopologyResidualCorrector reel."
        )
    model.fit(
        training.contexts[selected_radius],
        training.actual,
        training.base_predictions,
    )
    expected_hyperparameter_sha = recipe.get("model_hyperparameters_sha256")
    if model.hyperparameter_sha256() != expected_hyperparameter_sha:
        raise TopologyExperimentError("BE: hash hyperparametres du refit divergent.")

    published_forecast_path = (
        calibration.directory / "be" / "autonomous" / "forecast_hourly_be.csv"
    )
    published = pd.read_csv(published_forecast_path)
    index = pd.DatetimeIndex(
        pd.to_datetime(published["delivery_start_utc"], utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    base = pd.DataFrame(
        {
            quantile: pd.to_numeric(
                published[f"residual_corrected__{quantile}"],
                errors="raise",
            ).to_numpy(dtype=float)
            for quantile in QUANTILE_COLUMNS
        },
        index=index,
    )
    expected = pd.DataFrame(
        {
            quantile: pd.to_numeric(
                published[f"topology_autonomous__{quantile}"],
                errors="raise",
            ).to_numpy(dtype=float)
            for quantile in QUANTILE_COLUMNS
        },
        index=index,
    )
    current_context = _load_current_context(
        contract=config.contract,
        zone="BE",
        archive=calibration.directory / "be" / "autonomous",
        forecast_index=index,
        radius=selected_radius,
    )
    reproduced = model.predict(current_context, base)
    maximum_error = float(
        np.max(
            np.abs(
                reproduced.loc[:, list(QUANTILE_COLUMNS)].to_numpy(dtype=float)
                - expected.loc[:, list(QUANTILE_COLUMNS)].to_numpy(dtype=float)
            )
        )
    )
    if not math.isfinite(maximum_error) or maximum_error > 1e-9:
        raise TopologyExperimentError(
            "BE: le modele refitte ne reproduit pas le forecast publie "
            f"(max_abs_error={maximum_error})."
        )
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    buffer.seek(0)
    roundtrip_model = joblib.load(buffer)
    if not isinstance(roundtrip_model, TopologyResidualCorrector):
        raise TopologyExperimentError("BE: classe invalide apres round-trip joblib.")
    roundtrip_prediction = roundtrip_model.predict(current_context, base)
    roundtrip_error = float(
        np.max(
            np.abs(
                roundtrip_prediction.loc[:, list(QUANTILE_COLUMNS)].to_numpy(
                    dtype=float
                )
                - reproduced.loc[:, list(QUANTILE_COLUMNS)].to_numpy(dtype=float)
            )
        )
    )
    if roundtrip_error != 0.0:
        raise TopologyExperimentError(
            "BE: predictions modifiees par le round-trip joblib "
            f"(max_abs_error={roundtrip_error})."
        )
    model = roundtrip_model
    return model, {
        "training_start_utc": full_index[0].isoformat(),
        "training_end_utc": full_index[-1].isoformat(),
        "training_hours": int(len(full_index)),
        "training_local_days": 365,
        "training_window_policy": "fixed_published_365_not_rolling",
        "rolling365_enabled": False,
        "published_forecast_path": str(published_forecast_path),
        "published_forecast_sha256": _sha256(published_forecast_path),
        "published_forecast_reproduction_max_abs_error": maximum_error,
        "joblib_roundtrip_prediction_max_abs_error": roundtrip_error,
        "joblib_roundtrip_predictions_identical": True,
        "model_audit": model.audit_metadata(),
    }


def prepare_operational_bundle(
    config: ExperimentConfig,
    *,
    calibration_dir: str | Path,
    output_dir: str | Path = DEFAULT_OPERATIONAL_DIRECTORY,
    expected_calibration_manifest_sha256: str = (
        CALIBRATION_V1_ARTIFACT_MANIFEST_SHA256
    ),
    expected_existing_bundle_sha256: str | None = None,
) -> Path:
    """Build one immutable fixed-calibration bundle for daily inference."""

    calibration_anchor = str(expected_calibration_manifest_sha256).lower()
    if calibration_anchor != CALIBRATION_V1_ARTIFACT_MANIFEST_SHA256:
        raise TopologyExperimentError(
            "Le bundle operationnel v1 exige l'ancre de calibration v1 figee."
        )
    destination = Path(output_dir).expanduser()
    if not destination.is_absolute():
        destination = config.contract.project_root / destination
    destination = destination.resolve()
    calibration = audit_published_calibration(
        config,
        calibration_dir,
        expected_manifest_sha256=calibration_anchor,
    )
    if destination.exists():
        if expected_existing_bundle_sha256 is None:
            raise TopologyExperimentError(
                "Bundle existant: fournissez son SHA artifact manifest attendu."
            )
        existing = load_operational_bundle(
            config,
            destination,
            expected_manifest_sha256=expected_existing_bundle_sha256,
        )
        if (
            existing.manifest.get("calibration_artifact_manifest_sha256")
            != calibration.artifact_manifest_sha256
        ):
            raise TopologyExperimentError(
                "Le bundle existant provient d'une autre calibration."
            )
        return destination
    audit_experiment_sources(config)

    def writer(staging: Path) -> None:
        model, fit_audit = _fit_fixed_operational_be_model(config, calibration)
        calibration_copy = staging / "calibration"
        for zone in SUPPORTED_ZONES:
            source = calibration.directory / zone.lower()
            target = calibration_copy / zone.lower()
            target.mkdir(parents=True, exist_ok=False)
            for name in (
                "candidate_predictions.csv.gz",
                "candidate_prediction_seal.json",
                "topology_recipe_seal.json",
            ):
                shutil.copy2(source / name, target / name)
            shutil.copy2(
                source / "autonomous" / "topology_evaluation.json",
                target / "topology_evaluation.json",
            )
        config_copy = staging / "pricefm_topology_experiment.yaml"
        shutil.copy2(config.source_path, config_copy)
        model_path = staging / "be_topology_model.joblib"
        joblib.dump(model, model_path, compress=3)
        model_sha = _sha256(model_path)
        policy = {
            zone: ("topology_promoted" if zone == "BE" else "fallback_identity")
            for zone in SUPPORTED_ZONES
        }
        manifest = {
            "schema_version": 1,
            "bundle_type": "fixed_pricefm_topology_operational",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_id": config.experiment_id,
            "calibration_directory": str(calibration.directory),
            "calibration_artifact_manifest_sha256": (
                calibration.artifact_manifest_sha256
            ),
            "config_sha256": config.config_sha256,
            "config_copy": config_copy.name,
            "config_copy_sha256": _sha256(config_copy),
            "promotion_policy": policy,
            "blend_policy": {
                "FR": "production_mkonline_blend_passthrough",
                "NL": "production_mkonline_blend_passthrough",
                "weights_retuned": False,
                "topology_blend_promoted": False,
            },
            "model": {
                "zone": "BE",
                "path": model_path.name,
                "sha256": model_sha,
                "class": "TopologyResidualCorrector",
                "selected_radius": calibration.recipes["BE"]["selected_radius"],
                "selected_scale": calibration.recipes["BE"]["selected_scale"],
                "hyperparameter_sha256": model.hyperparameter_sha256(),
            },
            "fit_audit": fit_audit,
            "runtime_versions": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
                "joblib": joblib.__version__,
            },
            "production_changed": False,
            "runs_live_written": False,
            "rolling365_enabled": False,
            "fixed_calibration_warning": (
                "Ce bundle reutilise la fenetre publiee de 365 jours; "
                "il ne constitue pas un refit rolling365."
            ),
        }
        _write_json(staging / "bundle_manifest.json", manifest)
        _write_artifact_checksums(
            staging,
            declared_directory=destination,
        )

    return atomic_experiment_publish(
        destination,
        project_root=config.contract.project_root,
        writer=writer,
    )


def load_operational_bundle(
    config: ExperimentConfig,
    directory: str | Path,
    *,
    expected_manifest_sha256: str,
    load_model: bool = True,
) -> OperationalBundle:
    """Validate every bundle byte before loading the local joblib model."""

    root = _assert_existing_experiment_directory(
        directory,
        project_root=config.contract.project_root,
        name="operational_dir",
    )
    manifest_path = root / "artifact_checksums.json"
    observed_manifest_sha = _sha256(manifest_path)
    expected_manifest_sha256 = str(expected_manifest_sha256).lower()
    if len(expected_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in expected_manifest_sha256
    ):
        raise TopologyExperimentError("Trust anchor bundle doit etre un SHA-256.")
    if observed_manifest_sha != expected_manifest_sha256:
        raise TopologyExperimentError("Trust anchor du bundle operationnel invalide.")
    artifact_manifest_sha = _validate_artifact_checksums(root)
    manifest = _read_json_object(root / "bundle_manifest.json", name="bundle manifest")
    if manifest.get("bundle_type") != "fixed_pricefm_topology_operational":
        raise TopologyExperimentError("Type de bundle operationnel invalide.")
    if manifest.get("config_sha256") != config.config_sha256:
        raise TopologyExperimentError("Bundle/config SHA incompatibles.")
    if manifest.get("calibration_artifact_manifest_sha256") != (
        CALIBRATION_V1_ARTIFACT_MANIFEST_SHA256
    ):
        raise TopologyExperimentError("Trust anchor calibration absent du bundle.")
    config_copy = (root / str(manifest.get("config_copy", ""))).resolve()
    if config_copy.parent != root or config_copy.name != (
        "pricefm_topology_experiment.yaml"
    ):
        raise TopologyExperimentError("Chemin config copie invalide dans le bundle.")
    if (
        _sha256(config_copy) != manifest.get("config_copy_sha256")
        or manifest.get("config_copy_sha256") != config.config_sha256
    ):
        raise TopologyExperimentError("Config copie du bundle alteree.")
    if manifest.get("rolling365_enabled") is not False:
        raise TopologyExperimentError("Ce chemin n'autorise pas rolling365.")
    expected_policy = {
        zone: ("topology_promoted" if zone == "BE" else "fallback_identity")
        for zone in SUPPORTED_ZONES
    }
    if manifest.get("promotion_policy") != expected_policy:
        raise TopologyExperimentError("Politique de promotion du bundle invalide.")
    expected_blend_policy = {
        "FR": "production_mkonline_blend_passthrough",
        "NL": "production_mkonline_blend_passthrough",
        "weights_retuned": False,
        "topology_blend_promoted": False,
    }
    if manifest.get("blend_policy") != expected_blend_policy:
        raise TopologyExperimentError("Politique blend du bundle invalide.")
    versions = manifest.get("runtime_versions")
    if not isinstance(versions, Mapping):
        raise TopologyExperimentError("Versions runtime absentes du bundle.")
    for name, current in (
        ("python", platform.python_version()),
        ("pandas", pd.__version__),
        ("numpy", np.__version__),
        ("sklearn", sklearn.__version__),
        ("joblib", joblib.__version__),
    ):
        if versions.get(name) != current:
            raise TopologyExperimentError(
                f"Runtime {name}={current} incompatible avec {versions.get(name)}."
            )

    model_block = manifest.get("model")
    if not isinstance(model_block, Mapping):
        raise TopologyExperimentError("Bloc model absent du bundle.")
    model_path = (root / str(model_block.get("path", ""))).resolve()
    if model_path.parent != root or model_path.name != "be_topology_model.joblib":
        raise TopologyExperimentError("Chemin modele operationnel invalide.")
    expected_model_sha = model_block.get("sha256")
    if not isinstance(expected_model_sha, str) or _sha256(model_path) != expected_model_sha:
        raise TopologyExperimentError("SHA modele operationnel invalide.")
    model: TopologyResidualCorrector | None = None
    if load_model:
        model = joblib.load(model_path)
        if _sha256(model_path) != expected_model_sha:
            raise TopologyExperimentError("Le modele a change pendant son chargement.")
        if not isinstance(model, TopologyResidualCorrector):
            raise TopologyExperimentError("Classe du modele operationnel invalide.")
        if model.hyperparameter_sha256() != model_block.get(
            "hyperparameter_sha256"
        ):
            raise TopologyExperimentError(
                "Hyperparametres du modele operationnel invalides."
            )
        if not math.isclose(
            model.correction_scale,
            float(model_block.get("selected_scale", math.nan)),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise TopologyExperimentError("Scale du modele operationnel invalide.")
        model_audit = model.audit_metadata()
        context_audit = model_audit.get("topology_context")
        if not isinstance(context_audit, Mapping) or context_audit.get(
            "target_zone"
        ) != "BE":
            raise TopologyExperimentError("Le modele operationnel n'est pas BE.")
        if context_audit.get("radius") != model_block.get("selected_radius"):
            raise TopologyExperimentError("Rayon du modele operationnel invalide.")

    evaluations: dict[str, Mapping[str, object]] = {}
    candidate_seals: dict[str, CandidateSeal] = {}
    for zone in SUPPORTED_ZONES:
        zone_root = root / "calibration" / zone.lower()
        evaluation = _read_json_object(
            zone_root / "topology_evaluation.json",
            name=f"bundle {zone} evaluation",
        )
        recipe = _read_json_object(
            zone_root / "topology_recipe_seal.json",
            name=f"bundle {zone} recipe",
        )
        variants = evaluation.get("variants")
        autonomous = variants.get("autonomous") if isinstance(variants, Mapping) else None
        if not isinstance(autonomous, Mapping):
            raise TopologyExperimentError(f"Bundle {zone}: decision autonome absente.")
        if autonomous.get("promoted") is not (zone == "BE"):
            raise TopologyExperimentError(f"Bundle {zone}: promotion alteree.")
        recipe_path = zone_root / "topology_recipe_seal.json"
        if autonomous.get("recipe_seal_sha256") != _sha256(recipe_path):
            raise TopologyExperimentError(f"Bundle {zone}: recette alteree.")
        if recipe.get("config_sha256") != config.config_sha256:
            raise TopologyExperimentError(f"Bundle {zone}: config recette alteree.")
        seal_path = zone_root / "candidate_prediction_seal.json"
        seal_payload = _read_json_object(seal_path, name=f"bundle {zone} candidate seal")
        candidate_path = zone_root / "candidate_predictions.csv.gz"
        candidate_sha = seal_payload.get("candidate_sha256")
        if not isinstance(candidate_sha, str) or _sha256(candidate_path) != candidate_sha:
            raise TopologyExperimentError(f"Bundle {zone}: candidat altere.")
        if autonomous.get("candidate_prediction_sha256") != candidate_sha:
            raise TopologyExperimentError(f"Bundle {zone}: candidat non relie.")
        evaluations[zone] = evaluation
        candidate_seals[zone] = CandidateSeal(
            prediction_path=candidate_path,
            manifest_path=seal_path,
            prediction_sha256=candidate_sha,
            manifest_sha256=_sha256(seal_path),
        )
    return OperationalBundle(
        directory=root,
        artifact_manifest_sha256=artifact_manifest_sha,
        manifest=manifest,
        model=model,
        evaluations=evaluations,
        candidate_seals=candidate_seals,
    )


def _annual_period_payload(splits: ProtocolSplits) -> dict[str, object]:
    index = splits.all_index
    days = tuple(day for stage in STAGE_NAMES for day in splits.days[stage])
    if len(days) != 365 or len(set(days)) != 365 or len(index) != 8760:
        raise TopologyExperimentError(
            "Le reporting annuel exige exactement 365 jours locaux et 8760 heures."
        )
    return {
        "start_local_day": days[0],
        "end_local_day": days[-1],
        "start_utc": index[0].isoformat(),
        "end_utc": index[-1].isoformat(),
        "n_local_days": 365,
        "n_hours": 8760,
    }


def _annual_autonomous_section(
    evaluation: Mapping[str, object],
    *,
    zone: str,
) -> Mapping[str, object]:
    variants = evaluation.get("variants")
    autonomous = variants.get("autonomous") if isinstance(variants, Mapping) else None
    if not isinstance(autonomous, Mapping):
        raise TopologyExperimentError(f"{zone}: decision autonome annuelle absente.")
    opened = autonomous.get("opened_stages")
    if not isinstance(opened, list) or opened[:2] != ["a", "development"]:
        raise TopologyExperimentError(f"{zone}: prefixe A/development invalide.")
    formal = [stage for stage in opened if stage in GATED_STAGES]
    if formal != list(GATED_STAGES[: len(formal)]):
        raise TopologyExperimentError(f"{zone}: stages formels non causaux.")
    gates = autonomous.get("gates")
    if not isinstance(gates, Mapping):
        gates = {}
    for stage in formal[:-1]:
        gate = gates.get(stage)
        if not isinstance(gate, Mapping) or gate.get("passes") is not True:
            raise TopologyExperimentError(
                f"{zone}: stage {stage} suivi sans gate precedente passee."
            )
    if formal:
        last_gate = gates.get(formal[-1])
        if not isinstance(last_gate, Mapping) or not isinstance(
            last_gate.get("passes"), bool
        ):
            raise TopologyExperimentError(f"{zone}: gate {formal[-1]} incomplete.")
    return autonomous


def _write_annual_policy_seal(
    directory: Path,
    *,
    config: ExperimentConfig,
    calibration: PublishedCalibration,
    zone: str,
    splits: ProtocolSplits,
) -> AnnualPolicySeal:
    """Freeze both annual strategies before decoding any spared outcome."""

    evaluation = calibration.evaluations[zone]
    autonomous = _annual_autonomous_section(evaluation, zone=zone)
    opened = tuple(str(stage) for stage in autonomous["opened_stages"])
    raw_gates = autonomous.get("gates")
    gates = raw_gates if isinstance(raw_gates, Mapping) else {}

    def passed(stage: str) -> bool:
        gate = gates.get(stage)
        return isinstance(gate, Mapping) and gate.get("passes") is True

    governed: dict[str, str] = {
        "seed": "identity",
        "a": "identity",
        "development": "identity",
        "b1": "identity",
        "b2": (
            "sealed_topology_candidate"
            if "b2" in opened and passed("b1")
            else "identity"
        ),
        "final": (
            "sealed_topology_candidate"
            if "final" in opened and passed("b1") and passed("b2")
            else "identity"
        ),
    }
    formal_shadow = {
        stage: (
            "sealed_topology_candidate"
            if stage in GATED_STAGES and stage in opened
            else "identity"
        )
        for stage in STAGE_NAMES
    }
    governed_reasons = {
        "seed": "seed_training",
        "a": "selection_A_excluded",
        "development": "development_non_formal_excluded",
        "b1": "awaiting_first_formal_gate",
        "b2": (
            "prior_gate_b1_passed"
            if governed["b2"] == "sealed_topology_candidate"
            else ("prior_gate_b1_failed" if "b1" in opened else "stage_unopened")
        ),
        "final": (
            "prior_gates_b1_b2_passed"
            if governed["final"] == "sealed_topology_candidate"
            else "prior_gate_failed_or_stage_unopened"
        ),
    }
    shadow_reasons = {
        stage: (
            "formal_holdout_opened_with_frozen_recipe"
            if formal_shadow[stage] == "sealed_topology_candidate"
            else (
                "selection_or_non_formal_stage_excluded"
                if stage in {"seed", "a", "development"}
                else "formal_stage_unopened_after_gate_failure"
            )
        )
        for stage in STAGE_NAMES
    }
    candidate = calibration.candidate_seals[zone]
    payload = {
        "schema_version": 1,
        "policy_type": "sealed_annual_365_no_new_prediction",
        "experiment_id": config.experiment_id,
        "zone": zone,
        "timezone": splits.timezone,
        "period": _annual_period_payload(splits),
        "out_of_sample_scope": "mixed_sequential_governed",
        "formal_candidate_scope": "formal_holdouts_only",
        "selection_A_excluded_from_strategies": True,
        "development_excluded_from_strategies": True,
        "identity_fallback_is_not_candidate_prediction": True,
        "opened_stages": list(opened),
        "formal_gate_passes": {
            stage: passed(stage) if stage in opened else None
            for stage in GATED_STAGES
        },
        "strategies": {
            "sequential_governed_strategy": {
                "stage_actions": governed,
                "stage_reasons": governed_reasons,
                "primary": True,
            },
            "causal_formal_shadow_strategy": {
                "stage_actions": formal_shadow,
                "stage_reasons": shadow_reasons,
                "primary": False,
            },
        },
        "calibration_artifact_manifest_sha256": (
            calibration.artifact_manifest_sha256
        ),
        "source_candidate_prediction_sha256": candidate.prediction_sha256,
        "source_candidate_seal_sha256": candidate.manifest_sha256,
        "config_sha256": config.config_sha256,
        "full_year_outcomes_opened_before_policy_seal": False,
        "no_fit": True,
        "no_predict": True,
        "no_refit": True,
        "no_new_prediction": True,
        "used_for_gate": False,
        "used_for_selection": False,
        "used_for_promotion": False,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = directory / "annual_policy_seal.json"
    if path.exists():
        raise FileExistsError(path)
    _write_json(path, payload)
    return AnnualPolicySeal(
        zone=zone,
        path=path,
        sha256=_sha256(path),
        governed_actions=governed,
        formal_shadow_actions=formal_shadow,
    )


def _load_annual_source_data(
    contract: TopologyExperimentContract,
    *,
    zone: str,
    expected_index: pd.DatetimeIndex,
) -> AnnualSourceData:
    source = contract.sources[zone]
    if _sha256(source.backtest_file) != source.backtest_sha256:
        raise TopologyExperimentError(f"{zone}: source annuelle modifiee avant lecture.")
    raw = _read_csv_window_without_future_outcomes(
        source.backtest_file,
        timestamp_column="delivery_start_utc",
        columns=(
            "forecast_origin_utc",
            "actual",
            "residual_corrected__q10",
            "residual_corrected__q50",
            "residual_corrected__q90",
        ),
        expected_index=expected_index,
    )
    if _sha256(source.backtest_file) != source.backtest_sha256:
        raise TopologyExperimentError(f"{zone}: source annuelle modifiee pendant lecture.")
    actual = pd.to_numeric(raw["actual"], errors="coerce").astype(float)
    if not np.isfinite(actual.to_numpy(dtype=float)).all():
        raise TopologyExperimentError(f"{zone}: outcomes annuels non finis.")
    base = raw.rename(
        columns={
            "residual_corrected__q10": "q10",
            "residual_corrected__q50": "q50",
            "residual_corrected__q90": "q90",
        }
    ).loc[:, list(QUANTILE_COLUMNS)]
    base = _validate_quantiles(
        base,
        expected_index=expected_index,
        name=f"{zone}_annual_baseline",
    )
    origins = pd.DatetimeIndex(
        pd.to_datetime(raw["forecast_origin_utc"], utc=True, errors="raise"),
        name="forecast_origin_utc",
    )
    expected_origins = _forecast_origins_for_delivery(
        expected_index,
        timezone_name=source.timezone,
    )
    if not np.array_equal(origins.asi8, expected_origins.asi8):
        raise TopologyExperimentError(f"{zone}: origins annuelles non causales.")
    if _sha256(source.metrics_file) != source.metrics_sha256:
        raise TopologyExperimentError(f"{zone}: metriques source annuelles alterees.")
    declared = _read_json_object(source.metrics_file, name=f"{zone} source metrics")
    diagnostics = declared.get("training_diagnostics")
    metric_rows = declared.get("metrics")
    if (
        not isinstance(diagnostics, Mapping)
        or diagnostics.get("metric_scope") != "sealed_final_365_delivery_days"
        or diagnostics.get("evaluation_start_local_date") != "2025-08-12"
        or diagnostics.get("evaluation_end_local_date") != "2026-08-11"
        or diagnostics.get("n_evaluation") != 8760
        or not isinstance(metric_rows, list)
    ):
        raise TopologyExperimentError(
            f"{zone}: scope des metriques source different des 365 jours courants."
        )
    residual_rows = [
        row
        for row in metric_rows
        if isinstance(row, Mapping) and row.get("model") == "residual_corrected"
    ]
    annual_mae = float(
        np.mean(np.abs(actual.to_numpy(dtype=float) - base["q50"].to_numpy(dtype=float)))
    )
    if (
        len(residual_rows) != 1
        or residual_rows[0].get("n_scored") != 8760
        or not math.isclose(
            float(residual_rows[0].get("mae", math.nan)),
            annual_mae,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or _sha256(source.metrics_file) != source.metrics_sha256
    ):
        raise TopologyExperimentError(
            f"{zone}: MAE recalculee differente du modele courant scelle."
        )
    return AnnualSourceData(
        actual=actual,
        base_predictions=base,
        forecast_origins=origins,
        source_sha256=source.backtest_sha256,
    )


def _load_annual_opened_candidate(
    calibration: PublishedCalibration,
    *,
    zone: str,
    splits: ProtocolSplits,
    source: AnnualSourceData,
) -> pd.DataFrame:
    seal = calibration.candidate_seals[zone]
    if (
        _sha256(seal.prediction_path) != seal.prediction_sha256
        or _sha256(seal.manifest_path) != seal.manifest_sha256
    ):
        raise TopologyExperimentError(f"{zone}: candidat source altere avant lecture.")
    raw = pd.read_csv(seal.prediction_path)
    if _sha256(seal.prediction_path) != seal.prediction_sha256:
        raise TopologyExperimentError(f"{zone}: candidat source altere pendant lecture.")
    required = {
        "delivery_start_utc",
        "forecast_origin_utc",
        "stage",
        "actual",
        *(f"residual_corrected__{q}" for q in QUANTILE_COLUMNS),
        *(f"topology_autonomous__{q}" for q in QUANTILE_COLUMNS),
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise TopologyExperimentError(f"{zone}: candidat annuel incomplet={missing}.")
    index = pd.DatetimeIndex(
        pd.to_datetime(raw["delivery_start_utc"], utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise TopologyExperimentError(f"{zone}: timeline candidat invalide.")
    raw.index = index
    stages = tuple(dict.fromkeys(raw["stage"].astype(str).tolist()))
    autonomous = _annual_autonomous_section(calibration.evaluations[zone], zone=zone)
    if stages != tuple(str(stage) for stage in autonomous["opened_stages"]):
        raise TopologyExperimentError(f"{zone}: stages candidats != decision scellee.")
    for stage in stages:
        stage_index = raw.index[raw["stage"].eq(stage)]
        if stage not in splits.indices or not stage_index.equals(splits.indices[stage]):
            raise TopologyExperimentError(f"{zone}: timeline stage {stage} non exacte.")
    expected_origins = _forecast_origins_for_delivery(index, timezone_name=splits.timezone)
    observed_origins = pd.DatetimeIndex(
        pd.to_datetime(raw["forecast_origin_utc"], utc=True, errors="raise")
    )
    if not np.array_equal(observed_origins.asi8, expected_origins.asi8):
        raise TopologyExperimentError(f"{zone}: origins candidat non exactes.")
    locations = source.actual.index.get_indexer(index)
    if bool((locations < 0).any()):
        raise TopologyExperimentError(f"{zone}: candidat hors periode annuelle.")
    comparisons = {
        "actual": source.actual.reindex(index).to_numpy(dtype=float),
        **{
            f"residual_corrected__{q}": source.base_predictions[q]
            .reindex(index)
            .to_numpy(dtype=float)
            for q in QUANTILE_COLUMNS
        },
    }
    for column, expected in comparisons.items():
        observed = pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(observed, expected, rtol=0.0, atol=1e-12):
            raise TopologyExperimentError(
                f"{zone}: candidat/source divergent pour {column}."
            )
    topology = raw.loc[
        :, [f"topology_autonomous__{q}" for q in QUANTILE_COLUMNS]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(topology.to_numpy(dtype=float)).all():
        raise TopologyExperimentError(f"{zone}: candidat topology non fini.")
    return raw


def _annual_stage_labels(splits: ProtocolSplits) -> pd.Series:
    labels = pd.Series("", index=splits.all_index, dtype="object")
    for stage in STAGE_NAMES:
        labels.loc[splits.indices[stage]] = stage
    if bool(labels.eq("").any()):
        raise RuntimeError("Stage annuel non attribue.")
    return labels


def _build_annual_strategy_frame(
    *,
    zone: str,
    splits: ProtocolSplits,
    source: AnnualSourceData,
    candidate: pd.DataFrame,
    policy: AnnualPolicySeal,
) -> pd.DataFrame:
    index = splits.all_index
    stages = _annual_stage_labels(splits)
    opened = candidate.reindex(index)
    available = opened["stage"].notna()
    formal_oos = available & stages.isin(GATED_STAGES)
    frame = pd.DataFrame(
        {
            "delivery_start_utc": index,
            "forecast_origin_utc": source.forecast_origins,
            "protocol_stage": stages.to_numpy(dtype=object),
            "actual": source.actual.to_numpy(dtype=float),
        }
    )
    for quantile in QUANTILE_COLUMNS:
        base_values = source.base_predictions[quantile].to_numpy(dtype=float)
        frame[f"residual_corrected__{quantile}"] = base_values
        opened_values = pd.to_numeric(
            opened[f"topology_autonomous__{quantile}"], errors="coerce"
        ).to_numpy(dtype=float)
        frame[f"topology_opened_candidate__{quantile}"] = opened_values
        governed = base_values.copy()
        shadow = base_values.copy()
        for stage in STAGE_NAMES:
            stage_mask = stages.eq(stage).to_numpy(dtype=bool)
            if policy.governed_actions[stage] == "sealed_topology_candidate":
                if not np.isfinite(opened_values[stage_mask]).all():
                    raise TopologyExperimentError(
                        f"{zone}: candidat gouverne absent pour {stage}."
                    )
                governed[stage_mask] = opened_values[stage_mask]
            if policy.formal_shadow_actions[stage] == "sealed_topology_candidate":
                if not np.isfinite(opened_values[stage_mask]).all():
                    raise TopologyExperimentError(
                        f"{zone}: candidat shadow absent pour {stage}."
                    )
                shadow[stage_mask] = opened_values[stage_mask]
        frame[f"sequential_governed_strategy__{quantile}"] = governed
        frame[f"causal_formal_shadow_strategy__{quantile}"] = shadow
    policy_payload = _read_json_object(policy.path, name=f"{zone} annual policy")
    strategies = policy_payload.get("strategies")
    if not isinstance(strategies, Mapping):
        raise TopologyExperimentError(f"{zone}: strategies absentes de la policy.")
    governed_section = strategies["sequential_governed_strategy"]
    shadow_section = strategies["causal_formal_shadow_strategy"]
    if not isinstance(governed_section, Mapping) or not isinstance(
        shadow_section, Mapping
    ):
        raise TopologyExperimentError(f"{zone}: policy strategy invalide.")
    governed_reasons = governed_section["stage_reasons"]
    shadow_reasons = shadow_section["stage_reasons"]
    if not isinstance(governed_reasons, Mapping) or not isinstance(
        shadow_reasons, Mapping
    ):
        raise TopologyExperimentError(f"{zone}: raisons policy invalides.")
    frame["candidate_available"] = available.to_numpy(dtype=bool)
    frame["candidate_formal_oos"] = formal_oos.to_numpy(dtype=bool)
    frame["governed_topology_active"] = stages.map(
        lambda stage: policy.governed_actions[str(stage)]
        == "sealed_topology_candidate"
    ).to_numpy(dtype=bool)
    frame["governed_reason"] = stages.map(
        lambda stage: str(governed_reasons[str(stage)])
    ).to_numpy(dtype=object)
    frame["formal_shadow_topology_active"] = stages.map(
        lambda stage: policy.formal_shadow_actions[str(stage)]
        == "sealed_topology_candidate"
    ).to_numpy(dtype=bool)
    frame["formal_shadow_reason"] = stages.map(
        lambda stage: str(shadow_reasons[str(stage)])
    ).to_numpy(dtype=object)
    for model in (
        "sequential_governed_strategy",
        "causal_formal_shadow_strategy",
    ):
        _validate_quantiles(
            frame.set_index("delivery_start_utc").loc[
                :, [f"{model}__{q}" for q in QUANTILE_COLUMNS]
            ].rename(columns={f"{model}__{q}": q for q in QUANTILE_COLUMNS}),
            expected_index=index,
            name=f"{zone}_{model}",
        )
    return frame


def _daily_outcome_summary(
    actual: pd.Series,
    candidate: pd.Series,
    baseline: pd.Series,
    *,
    timezone_name: str,
    active_mask: pd.Series,
) -> dict[str, object]:
    daily = _daily_paired_gains(
        actual,
        candidate,
        baseline,
        timezone_name=timezone_name,
    )
    local_days = pd.Index(actual.index.tz_convert(timezone_name).strftime("%Y-%m-%d"))
    active_days = pd.Series(
        active_mask.to_numpy(dtype=bool),
        index=local_days,
    ).groupby(level=0, sort=False).any()

    def summarize(values: pd.Series) -> dict[str, object]:
        numeric = values.to_numpy(dtype=float)
        tolerance = 1e-12
        wins = int(np.count_nonzero(numeric > tolerance))
        losses = int(np.count_nonzero(numeric < -tolerance))
        ties = int(len(numeric) - wins - losses)
        count = int(len(numeric))
        return {
            "n_days": count,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "win_rate": float(wins / count) if count else None,
            "tie_rate": float(ties / count) if count else None,
            "loss_rate": float(losses / count) if count else None,
        }

    active_labels = active_days.index[active_days.to_numpy(dtype=bool)]
    return {
        "all_days": summarize(daily),
        "active_days": summarize(daily.reindex(active_labels)),
    }


def _annual_strategy_metrics(
    frame: pd.DataFrame,
    *,
    model: str,
    active_column: str,
    timezone_name: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    index = pd.DatetimeIndex(
        pd.to_datetime(frame["delivery_start_utc"], utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    actual = pd.Series(frame["actual"].to_numpy(dtype=float), index=index)
    candidate = pd.Series(frame[f"{model}__q50"].to_numpy(dtype=float), index=index)
    baseline = pd.Series(
        frame["residual_corrected__q50"].to_numpy(dtype=float), index=index
    )
    active = pd.Series(frame[active_column].to_numpy(dtype=bool), index=index)
    full = paired_segment_metrics(
        actual,
        candidate,
        baseline,
        timezone_name=timezone_name,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    active_metrics: dict[str, object] | None = None
    if bool(active.any()):
        active_metrics = asdict(
            paired_segment_metrics(
                actual.loc[active],
                candidate.loc[active],
                baseline.loc[active],
                timezone_name=timezone_name,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
        )
    local_days = pd.Index(index.tz_convert(timezone_name).strftime("%Y-%m-%d"))
    active_days = int(
        pd.Series(active.to_numpy(dtype=bool), index=local_days)
        .groupby(level=0, sort=False)
        .any()
        .sum()
    )
    return {
        "full_period_metrics": asdict(full),
        "active_only_metrics": active_metrics,
        "active_hours": int(active.sum()),
        "active_days": active_days,
        "active_coverage": float(active.mean()),
        "daily_outcomes": _daily_outcome_summary(
            actual,
            candidate,
            baseline,
            timezone_name=timezone_name,
            active_mask=active,
        ),
    }


def _seal_annual_strategy(
    directory: Path,
    frame: pd.DataFrame,
    *,
    zone: str,
    experiment_id: str,
    policy: AnnualPolicySeal,
) -> CandidateSeal:
    prediction_path = directory / "annual_strategy_hourly.csv.gz"
    manifest_path = directory / "annual_strategy_seal.json"
    if prediction_path.exists() or manifest_path.exists():
        raise FileExistsError(f"{zone}: strategie annuelle deja scellee.")
    _write_deterministic_csv_gzip(prediction_path, frame)
    prediction_sha = _sha256(prediction_path)
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "seal_type": "annual_strategy_before_comparators",
            "experiment_id": experiment_id,
            "zone": zone,
            "annual_policy_sha256": policy.sha256,
            "prediction_path": prediction_path.name,
            "prediction_sha256": prediction_sha,
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "storm_loaded_before_seal": False,
            "mkonline_loaded_before_seal": False,
            "no_fit": True,
            "no_predict": True,
            "no_new_prediction": True,
            "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return CandidateSeal(
        prediction_path=prediction_path,
        manifest_path=manifest_path,
        prediction_sha256=prediction_sha,
        manifest_sha256=_sha256(manifest_path),
    )


def _load_annual_production_blend_reference(
    contract: TopologyExperimentContract,
    *,
    zone: str,
    base: pd.DataFrame,
    expected_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read only the already-governed FR/NL production comparator."""

    if zone not in BLEND_ZONES:
        raise TopologyExperimentError(f"{zone}: reference MKOnline indisponible.")
    dependency = contract.blend_dependency_for(zone)
    paths = (
        (dependency.recipe_manifest, dependency.recipe_manifest_sha256, "recipe"),
        (
            dependency.dependency_manifest,
            dependency.dependency_manifest_sha256,
            "dependency",
        ),
        (dependency.forecast_file, dependency.forecast_file_sha256, "forecast"),
    )
    for path, expected_sha, label in paths:
        if _sha256(path) != expected_sha:
            raise TopologyExperimentError(f"{zone}: SHA blend {label} invalide.")
    recipe = _read_json_object(dependency.recipe_manifest, name=f"{zone} blend recipe")
    weights = recipe.get("weights")
    if (
        recipe.get("zone") != zone
        or recipe.get("status") != "frozen_before_final_opening"
        or recipe.get("source_autonomous_model") != "residual_corrected"
        or not isinstance(weights, Mapping)
        or not math.isclose(
            float(weights.get("autonomous", math.nan)),
            dependency.current_weight_autonomous,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(weights.get("mkonline_primary", math.nan)),
            dependency.current_weight_mkonline,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise TopologyExperimentError(f"{zone}: recette blend production invalide.")
    external = recipe.get("external_expert")
    if (
        not isinstance(external, Mapping)
        or external.get("series") != dependency.primary_series
        or external.get("dependency_manifest_sha256")
        != dependency.dependency_manifest_sha256
        or external.get("storm_used_as_feature") is not False
        or external.get("interpolation_allowed") is not False
    ):
        raise TopologyExperimentError(f"{zone}: expert MKOnline non conforme.")
    selection = recipe.get("selection_protocol")
    if not isinstance(selection, Mapping):
        raise TopologyExperimentError(f"{zone}: selection blend invalide.")
    final_used = recipe.get(
        "final_target_used_for_weight_or_hyperparameters",
        selection.get("final_target_used_for_weight_or_hyperparameters"),
    )
    if final_used is not False:
        raise TopologyExperimentError(
            f"{zone}: la reference blend a utilise final pour son reglage."
        )
    dependency_manifest = _read_json_object(
        dependency.dependency_manifest,
        name=f"{zone} blend dependency",
    )
    if (
        dependency_manifest.get("dependency_gate_passed") is not True
        or dependency_manifest.get("storm_token_found") is not False
        or dependency_manifest.get("terminal_series") != dependency.primary_series
    ):
        raise TopologyExperimentError(f"{zone}: dependency MKOnline invalide.")
    raw = pd.read_parquet(dependency.forecast_file)
    required = {"value_time_utc", "snapshot_time_utc", "revision_time_utc", "value"}
    if not required.issubset(raw.columns):
        raise TopologyExperimentError(f"{zone}: schema forecast MKOnline invalide.")
    delivery = pd.DatetimeIndex(
        pd.to_datetime(raw["value_time_utc"], utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    if not delivery.equals(expected_index):
        raise TopologyExperimentError(f"{zone}: timeline MKOnline non annuelle exacte.")
    primary = pd.Series(
        pd.to_numeric(raw["value"], errors="coerce").to_numpy(dtype=float),
        index=delivery,
        name="mkonline_primary__q50",
    )
    if not np.isfinite(primary.to_numpy(dtype=float)).all():
        raise TopologyExperimentError(f"{zone}: MKOnline non fini.")
    local_days = delivery.tz_convert(ZONE_TIMEZONES[zone]).date
    expected_cutoff = pd.DatetimeIndex(
        [
            pd.Timestamp(day - timedelta(days=1), tz="Europe/Paris")
            .replace(hour=8)
            .tz_convert("UTC")
            for day in local_days
        ]
    )
    snapshot = pd.DatetimeIndex(
        pd.to_datetime(raw["snapshot_time_utc"], utc=True, errors="raise")
    )
    revision = pd.DatetimeIndex(
        pd.to_datetime(raw["revision_time_utc"], utc=True, errors="raise")
    )
    violations = int(
        np.count_nonzero(snapshot.asi8 != expected_cutoff.asi8)
        + np.count_nonzero(revision.asi8 != expected_cutoff.asi8)
    )
    if violations:
        raise TopologyExperimentError(f"{zone}: cutoff MKOnline non causal.")
    for path, expected_sha, label in paths:
        if _sha256(path) != expected_sha:
            raise TopologyExperimentError(
                f"{zone}: blend {label} modifie pendant lecture."
            )
    blended = _blend_quantiles(
        base,
        primary,
        weight_mkonline=dependency.current_weight_mkonline,
    )
    return blended, {
        "recipe_manifest_sha256": dependency.recipe_manifest_sha256,
        "dependency_manifest_sha256": dependency.dependency_manifest_sha256,
        "forecast_file_sha256": dependency.forecast_file_sha256,
        "primary_series": dependency.primary_series,
        "cutoff_violations": violations,
        "coverage": 1.0,
    }


def _load_annual_storm_reference(
    *,
    seal: CandidateSeal,
    zone: str,
    evaluation: Mapping[str, object],
    expected_index: pd.DatetimeIndex,
) -> tuple[pd.Series | None, dict[str, object]]:
    original = evaluation.get("storm_evaluation")
    if not isinstance(original, Mapping):
        raise TopologyExperimentError(f"{zone}: audit Storm calibration absent.")
    if zone == "ES":
        if original.get("available") is not False:
            raise TopologyExperimentError("ES: indisponibilite Storm non scellee.")
        return None, {
            "available": False,
            "reason": "no native dashboard contract for ES",
            "used_for_prediction": False,
        }
    source_path = Path(str(original.get("source_path", ""))).expanduser().resolve()
    audit_path = Path(str(original.get("source_audit_path", ""))).expanduser().resolve()
    if (
        source_path.name != "storm_dashboard_official_statistics.parquet"
        or source_path.parent.name != "inputs"
        or audit_path != source_path.parent.parent / "statistics_history_audit.json"
    ):
        raise TopologyExperimentError(f"{zone}: chemins Storm calibration invalides.")
    if (
        _sha256(source_path) != original.get("source_sha256")
        or _sha256(audit_path) != original.get("source_audit_sha256")
    ):
        raise TopologyExperimentError(f"{zone}: source Storm a change depuis calibration.")
    storm, audit = _load_storm_after_candidate(
        seal=seal,
        zone=zone,
        archive=source_path.parent.parent,
        expected_index=expected_index,
    )
    if (
        audit.get("source_sha256") != original.get("source_sha256")
        or audit.get("source_audit_sha256") != original.get("source_audit_sha256")
    ):
        raise TopologyExperimentError(f"{zone}: audit Storm annuel divergent.")
    return storm, audit


def report_topology_annual_365(
    config: ExperimentConfig,
    *,
    calibration_dir: str | Path,
    output_dir: str | Path = DEFAULT_ANNUAL_OUTPUT_DIRECTORY,
    expected_calibration_manifest_sha256: str = (
        CALIBRATION_V1_ARTIFACT_MANIFEST_SHA256
    ),
) -> Path:
    """Publish a no-fit annual comparison from already-sealed predictions."""

    destination = _assert_experiment_output(
        output_dir,
        project_root=config.contract.project_root,
    )
    calibration = audit_published_calibration(
        config,
        calibration_dir,
        expected_manifest_sha256=expected_calibration_manifest_sha256,
    )
    # This audit intentionally does not decode outcome/prediction values.
    source_audits = audit_experiment_sources(config)

    def writer(staging: Path) -> None:
        splits_by_zone = {
            zone: _splits_for_contract(config.contract, zone)
            for zone in config.zones
        }
        periods = {
            zone: _annual_period_payload(splits_by_zone[zone])
            for zone in config.zones
        }
        first_period = periods[config.zones[0]]
        if any(period != first_period for period in periods.values()):
            raise TopologyExperimentError("Les periodes annuelles pays divergent.")

        # Phase 1: every stage/action policy is durable before previously
        # spared B2/final outcomes can be decoded by Phase 2.
        policy_seals: dict[str, AnnualPolicySeal] = {}
        for zone in config.zones:
            zone_dir = staging / zone.lower()
            zone_dir.mkdir(parents=True, exist_ok=False)
            policy_seals[zone] = _write_annual_policy_seal(
                zone_dir,
                config=config,
                calibration=calibration,
                zone=zone,
                splits=splits_by_zone[zone],
            )
        for policy in policy_seals.values():
            if not policy.path.is_file() or _sha256(policy.path) != policy.sha256:
                raise TopologyExperimentError("Une policy annuelle a change avant outcomes.")

        zone_manifests: dict[str, object] = {}
        annual_report_artifacts: list[TopologyAnnualReportArtifact] = []
        for zone in config.zones:
            zone_dir = staging / zone.lower()
            splits = splits_by_zone[zone]
            policy = policy_seals[zone]
            source = _load_annual_source_data(
                config.contract,
                zone=zone,
                expected_index=splits.all_index,
            )
            candidate = _load_annual_opened_candidate(
                calibration,
                zone=zone,
                splits=splits,
                source=source,
            )
            frame = _build_annual_strategy_frame(
                zone=zone,
                splits=splits,
                source=source,
                candidate=candidate,
                policy=policy,
            )
            strategy_seal = _seal_annual_strategy(
                zone_dir,
                frame,
                zone=zone,
                experiment_id=config.experiment_id,
                policy=policy,
            )
            if _sha256(policy.path) != policy.sha256:
                raise TopologyExperimentError(f"{zone}: policy modifiee apres outcomes.")

            strategies = {
                "sequential_governed_strategy": _annual_strategy_metrics(
                    frame,
                    model="sequential_governed_strategy",
                    active_column="governed_topology_active",
                    timezone_name=splits.timezone,
                    bootstrap_samples=config.bootstrap_samples,
                    bootstrap_seed=config.bootstrap_seed,
                ),
                "causal_formal_shadow_strategy": _annual_strategy_metrics(
                    frame,
                    model="causal_formal_shadow_strategy",
                    active_column="formal_shadow_topology_active",
                    timezone_name=splits.timezone,
                    bootstrap_samples=config.bootstrap_samples,
                    bootstrap_seed=config.bootstrap_seed,
                ),
            }
            actual = pd.Series(
                frame["actual"].to_numpy(dtype=float), index=splits.all_index
            )
            baseline = pd.Series(
                frame["residual_corrected__q50"].to_numpy(dtype=float),
                index=splits.all_index,
            )
            governed = pd.Series(
                frame["sequential_governed_strategy__q50"].to_numpy(dtype=float),
                index=splits.all_index,
            )

            mkonline_reference: dict[str, object]
            if zone in BLEND_ZONES:
                production_blend, blend_audit = load_comparator_after_seal(
                    strategy_seal,
                    lambda _zone=zone: _load_annual_production_blend_reference(
                        config.contract,
                        zone=_zone,
                        base=source.base_predictions,
                        expected_index=splits.all_index,
                    ),
                )
                if not isinstance(production_blend, pd.DataFrame):
                    raise RuntimeError("Reference blend annuelle invalide.")
                blend_series = production_blend["q50"]
                paired_baseline = paired_segment_metrics(
                    actual,
                    blend_series,
                    baseline,
                    timezone_name=splits.timezone,
                    bootstrap_samples=config.bootstrap_samples,
                    bootstrap_seed=config.bootstrap_seed,
                )
                paired_governed = paired_segment_metrics(
                    actual,
                    blend_series,
                    governed,
                    timezone_name=splits.timezone,
                    bootstrap_samples=config.bootstrap_samples,
                    bootstrap_seed=config.bootstrap_seed,
                )
                dependency = config.contract.blend_dependency_for(zone)
                mkonline_reference = {
                    "available": True,
                    "model": "production_mkonline_blend",
                    "comparator_only": True,
                    "topology_blend_candidate_available": False,
                    "weights_recomputed": False,
                    "weights": {
                        "autonomous": dependency.current_weight_autonomous,
                        "mkonline_primary": dependency.current_weight_mkonline,
                    },
                    "metrics": {
                        "n_hours": int(len(blend_series)),
                        "n_days": 365,
                        "mae": float(np.mean(np.abs(actual - blend_series))),
                        "gain_vs_sequential_governed_eur_mwh": (
                            float(np.mean(np.abs(actual - governed)))
                            - float(np.mean(np.abs(actual - blend_series)))
                        ),
                        "gain_vs_residual_corrected_eur_mwh": (
                            float(np.mean(np.abs(actual - baseline)))
                            - float(np.mean(np.abs(actual - blend_series)))
                        ),
                    },
                    "paired_vs_residual_corrected": asdict(paired_baseline),
                    "paired_vs_sequential_governed": asdict(paired_governed),
                    "audit": blend_audit,
                    "used_for_prediction": False,
                    "used_for_gate": False,
                    "used_for_selection": False,
                    "used_for_promotion": False,
                }
            else:
                mkonline_reference = {
                    "available": False,
                    "reason": "production MKOnline blend is available only for FR/NL",
                    "comparator_only": True,
                    "topology_blend_candidate_available": False,
                    "weights_recomputed": False,
                }

            storm, storm_audit = _load_annual_storm_reference(
                seal=strategy_seal,
                zone=zone,
                evaluation=calibration.evaluations[zone],
                expected_index=splits.all_index,
            )
            if storm is None:
                storm_section: dict[str, object] = {
                    **storm_audit,
                    "available": False,
                    "comparator_only": True,
                    "used_for_gate": False,
                    "used_for_selection": False,
                    "used_for_promotion": False,
                }
            else:
                storm_frame = frame.copy()
                storm_frame["stage"] = "annual"
                storm_section = {
                    "available": True,
                    "comparator_only": True,
                    "audit": storm_audit,
                    "metrics": _storm_evaluation_metrics(
                        storm_frame,
                        storm,
                        stage="annual",
                        candidate_model="sequential_governed_strategy",
                        baseline_model="residual_corrected",
                        timezone_name=splits.timezone,
                    ),
                    "used_for_prediction": False,
                    "used_for_gate": False,
                    "used_for_selection": False,
                    "used_for_promotion": False,
                }

            metrics_payload = {
                "schema_version": 1,
                "report_type": "sealed_annual_365_strategy",
                "experiment_id": config.experiment_id,
                "zone": zone,
                "timezone": splits.timezone,
                "period": periods[zone],
                "n_days": 365,
                "n_hours": 8760,
                "out_of_sample_scope": "mixed_sequential_governed",
                "formal_candidate_scope": "formal_holdouts_only",
                "selection_A_excluded_from_strategies": True,
                "development_excluded_from_strategies": True,
                "identity_fallback_is_not_candidate_prediction": True,
                "baseline": {
                    "model": "residual_corrected",
                    "n_hours": 8760,
                    "n_days": 365,
                    "mae": float(np.mean(np.abs(actual - baseline))),
                },
                "primary_strategy": "sequential_governed_strategy",
                "strategies": strategies,
                "pure_topology_annual": {
                    "available": False,
                    "reason": (
                        "No sealed pure-topology prediction covers all 365 days; "
                        "identity fallback hours must not be relabelled as candidate."
                    ),
                },
                "storm": storm_section,
                "mkonline_production_reference": mkonline_reference,
                "calibration_artifact_manifest_sha256": (
                    calibration.artifact_manifest_sha256
                ),
                "source_backtest_sha256": source.source_sha256,
                "source_candidate_prediction_sha256": (
                    calibration.candidate_seals[zone].prediction_sha256
                ),
                "annual_policy_sha256": policy.sha256,
                "annual_strategy_sha256": strategy_seal.prediction_sha256,
                "annual_strategy_seal_sha256": strategy_seal.manifest_sha256,
                "config_sha256": config.config_sha256,
                "no_fit": True,
                "no_predict": True,
                "no_refit": True,
                "no_new_prediction": True,
                "rolling365_enabled": False,
                "used_for_gate": False,
                "used_for_selection": False,
                "used_for_promotion": False,
                "production_changed": False,
                "runs_live_written": False,
            }
            _write_json(zone_dir / "annual_metrics.json", metrics_payload)
            run_manifest = {
                "schema_version": 1,
                "run_type": "sealed_topology_annual_365_reporting",
                "zone": zone,
                "timezone": splits.timezone,
                "period": periods[zone],
                "n_days": 365,
                "n_hours": 8760,
                "primary_strategy": "sequential_governed_strategy",
                "annual_policy_sha256": policy.sha256,
                "annual_strategy_sha256": strategy_seal.prediction_sha256,
                "annual_metrics_sha256": _sha256(zone_dir / "annual_metrics.json"),
                "calibration_artifact_manifest_sha256": (
                    calibration.artifact_manifest_sha256
                ),
                "source_audit": source_audits[zone],
                "no_fit": True,
                "no_predict": True,
                "no_refit": True,
                "no_new_prediction": True,
                "rolling365_enabled": False,
                "production_changed": False,
                "runs_live_written": False,
            }
            _write_json(zone_dir / "run_manifest.json", run_manifest)

            # The renderer authenticates this exact data-only snapshot and
            # refuses to mutate any sealed input.  Once it returns, the zone
            # manifest is finalized a second time so that the immutable HTML
            # is itself covered by the published checksum inventory.
            _write_artifact_checksums(
                zone_dir,
                declared_directory=destination / zone.lower(),
            )
            report_artifact = write_topology_annual_html_report(
                zone_dir,
                project_root=config.contract.project_root,
            )
            expected_report = (
                zone_dir / "reports" / f"topology_annual_{zone.lower()}.html"
            ).resolve()
            if report_artifact.path.resolve() != expected_report:
                raise TopologyExperimentError(
                    f"{zone}: chemin du rapport annuel renderer inattendu."
                )
            if (
                not report_artifact.path.is_file()
                or _sha256(report_artifact.path) != report_artifact.sha256
            ):
                raise TopologyExperimentError(
                    f"{zone}: rapport annuel absent ou modifie apres rendu."
                )
            annual_report_artifacts.append(report_artifact)
            zone_manifest_path = _write_artifact_checksums(
                zone_dir,
                declared_directory=destination / zone.lower(),
            )
            finalized_record = load_topology_annual_evaluation(
                zone_dir,
                project_root=config.contract.project_root,
            )
            if finalized_record.zone != zone:
                raise TopologyExperimentError(
                    f"{zone}: le bundle annuel final recharge une autre zone."
                )
            zone_manifests[zone] = {
                "directory": zone.lower(),
                "artifact_manifest_sha256": _sha256(zone_manifest_path),
                "annual_policy_sha256": policy.sha256,
                "annual_strategy_sha256": strategy_seal.prediction_sha256,
                "annual_metrics_sha256": _sha256(zone_dir / "annual_metrics.json"),
                "annual_report_path": (
                    Path(zone.lower())
                    / "reports"
                    / f"topology_annual_{zone.lower()}.html"
                ).as_posix(),
                "annual_report_sha256": report_artifact.sha256,
            }

        annual_index = write_topology_annual_report_index(
            annual_report_artifacts,
            output_path=staging / "reports" / "topology_annual_365_index.html",
            selected_zones=config.zones,
            project_root=config.contract.project_root,
        )
        annual_index_sha256 = _sha256(annual_index)
        _write_json(
            staging / "annual_manifest.json",
            {
                "schema_version": 1,
                "run_type": "sealed_topology_annual_365_reporting",
                "experiment_id": config.experiment_id,
                "zones": list(config.zones),
                "period": first_period,
                "primary_strategy": "sequential_governed_strategy",
                "secondary_strategy": "causal_formal_shadow_strategy",
                "out_of_sample_scope": "mixed_sequential_governed",
                "calibration_artifact_manifest_sha256": (
                    calibration.artifact_manifest_sha256
                ),
                "config_sha256": config.config_sha256,
                "zone_artifacts": zone_manifests,
                "annual_index_path": "reports/topology_annual_365_index.html",
                "annual_index_sha256": annual_index_sha256,
                "policy_sealed_before_full_year_outcomes": True,
                "comparators_loaded_after_annual_strategy_seal": True,
                "no_fit": True,
                "no_predict": True,
                "no_refit": True,
                "no_new_prediction": True,
                "rolling365_enabled": False,
                "used_for_gate": False,
                "used_for_selection": False,
                "used_for_promotion": False,
                "production_changed": False,
                "runs_live_written": False,
            },
        )
        _write_artifact_checksums(staging, declared_directory=destination)

    return atomic_experiment_publish(
        destination,
        project_root=config.contract.project_root,
        writer=writer,
    )


def _rolling365_combined_index(splits: ProtocolSplits) -> pd.DatetimeIndex:
    evaluation = splits.all_index
    local = evaluation.tz_convert(splits.timezone)
    first_day = pd.Timestamp(local[0].date(), tz=splits.timezone)
    last_day = pd.Timestamp(local[-1].date(), tz=splits.timezone)
    start = first_day - pd.DateOffset(days=365)
    stop = last_day + pd.DateOffset(days=1)
    index = pd.date_range(
        start=start.tz_convert("UTC"),
        end=stop.tz_convert("UTC") - pd.Timedelta(hours=1),
        freq="h",
        name="delivery_start_utc",
    )
    days = pd.Index(index.tz_convert(splits.timezone).date).unique()
    if len(days) != 730 or not evaluation.equals(index[index.isin(evaluation)]):
        raise TopologyExperimentError(
            "Le backtest rolling365 exige 365 jours d'amorce puis 365 jours eval." 
        )
    return index


def _load_rolling365_zone_data(
    contract: TopologyExperimentContract,
    *,
    zone: str,
    splits: ProtocolSplits,
) -> Rolling365ZoneData:
    """Load the exact autonomous OOF and causal topology inputs for 730 days."""

    source = contract.sources[zone]
    combined_index = _rolling365_combined_index(splits)
    if _sha256(source.backtest_file) != source.backtest_sha256:
        raise TopologyExperimentError(f"{zone}: backtest source rolling365 modifie.")
    raw = _read_csv_window_without_future_outcomes(
        source.backtest_file,
        timestamp_column="delivery_start_utc",
        columns=(
            "forecast_origin_utc",
            "actual",
            "ensemble__q10",
            "ensemble__q50",
            "ensemble__q90",
            "residual_corrected__q10",
            "residual_corrected__q50",
            "residual_corrected__q90",
        ),
        expected_index=combined_index,
    )
    if _sha256(source.backtest_file) != source.backtest_sha256:
        raise TopologyExperimentError(
            f"{zone}: backtest source rolling365 modifie pendant lecture."
        )
    actual = pd.to_numeric(raw["actual"], errors="coerce").astype(float)
    if not np.isfinite(actual.to_numpy(dtype=float)).all():
        raise TopologyExperimentError(f"{zone}: cible rolling365 non finie.")
    ensemble = raw.rename(
        columns={f"ensemble__{q}": q for q in QUANTILE_COLUMNS}
    ).loc[:, list(QUANTILE_COLUMNS)]
    ensemble = _validate_quantiles(
        ensemble,
        expected_index=combined_index,
        name=f"{zone}_rolling365_ensemble",
    )
    current = raw.rename(
        columns={f"residual_corrected__{q}": q for q in QUANTILE_COLUMNS}
    ).loc[splits.all_index, list(QUANTILE_COLUMNS)]
    current = _validate_quantiles(
        current,
        expected_index=splits.all_index,
        name=f"{zone}_rolling365_current_autonomous",
    )
    origins = pd.DatetimeIndex(
        pd.to_datetime(raw["forecast_origin_utc"], utc=True, errors="raise"),
        name="forecast_origin_utc",
    )
    expected_origins = _forecast_origins_for_delivery(
        combined_index,
        timezone_name=source.timezone,
    )
    if not np.array_equal(origins.asi8, expected_origins.asi8):
        raise TopologyExperimentError(f"{zone}: origins OOF rolling365 non causales.")

    extended_index = pd.date_range(
        combined_index[0] - pd.Timedelta(hours=24),
        combined_index[-1],
        freq="h",
        name="delivery_start_utc",
    )
    residual_raw = _read_csv_window_without_future_outcomes(
        source.aligned_inputs_file,
        timestamp_column="timestamp",
        columns=tuple(contract.residual_load_columns.values()),
        expected_index=extended_index,
    )
    residual = pd.DataFrame(
        {
            code: pd.to_numeric(
                residual_raw[contract.residual_load_columns[code]], errors="coerce"
            )
            for code in SUPPORTED_ZONES
        },
        index=extended_index,
    )
    price_source_index = combined_index - pd.Timedelta(hours=24)
    prices = pd.DataFrame(
        np.nan,
        index=extended_index,
        columns=list(SUPPORTED_ZONES),
    )
    for price_zone in SUPPORTED_ZONES:
        price_source = contract.sources[price_zone]
        if _sha256(price_source.target_cache) != price_source.target_cache_sha256:
            raise TopologyExperimentError(
                f"{price_zone}: cache prix modifie avant rolling365."
            )
        price_raw = _read_csv_window_without_future_outcomes(
            price_source.target_cache,
            timestamp_column="timestamp",
            columns=("value",),
            expected_index=price_source_index,
        )
        prices.loc[price_source_index, price_zone] = pd.to_numeric(
            price_raw["value"], errors="coerce"
        ).to_numpy(dtype=float)
        if _sha256(price_source.target_cache) != price_source.target_cache_sha256:
            raise TopologyExperimentError(
                f"{price_zone}: cache prix modifie pendant rolling365."
            )
    contexts: dict[int, pd.DataFrame] = {}
    for radius in (0, 1):
        full = build_topology_context(
            residual,
            prices,
            target_zone=zone,
            radius=radius,
            timezone=source.timezone,
        )
        contexts[radius] = _slice_context(full, combined_index)
    return Rolling365ZoneData(
        actual=actual,
        ensemble_base=ensemble,
        current_autonomous=current,
        forecast_origins=origins,
        contexts=contexts,
        evaluation_index=splits.all_index,
        source_sha256=source.backtest_sha256,
    )


def _fit_rolling365_zone(
    config: ExperimentConfig,
    *,
    zone: str,
    recipe: Mapping[str, object],
    recipe_sha256: str,
) -> Rolling365ZoneResult:
    """Refit one frozen topology recipe for every evaluation delivery day."""

    splits = _splits_for_contract(config.contract, zone)
    data = _load_rolling365_zone_data(config.contract, zone=zone, splits=splits)
    radius = int(recipe.get("selected_radius", -1))
    scale = float(recipe.get("selected_scale", float("nan")))
    arm = str(recipe.get("selected_arm", ""))
    if radius not in (0, 1) or not math.isfinite(scale) or not arm:
        raise TopologyExperimentError(f"{zone}: recette rolling365 invalide.")
    if recipe.get("config_sha256") != config.config_sha256:
        raise TopologyExperimentError(f"{zone}: recette/config rolling365 divergentes.")
    if dict(recipe.get("model_parameters", {})) != dict(config.model_parameters):
        raise TopologyExperimentError(
            f"{zone}: hyperparametres recette rolling365 divergents."
        )
    probe = _new_corrector(config.model_parameters, correction_scale=scale)
    expected_hyper = str(recipe.get("model_hyperparameters_sha256", ""))
    if probe.hyperparameter_sha256() != expected_hyper:
        raise TopologyExperimentError(
            f"{zone}: hash hyperparametres rolling365 divergent."
        )

    local_days = tuple(
        dict.fromkeys(
            pd.Index(data.evaluation_index.tz_convert(splits.timezone).date).tolist()
        )
    )
    if len(local_days) != 365:
        raise TopologyExperimentError(f"{zone}: evaluation rolling365 != 365 jours.")
    prediction_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    context = data.contexts[radius]
    for position, delivery_day in enumerate(local_days, start=1):
        delivery_start = pd.Timestamp(delivery_day, tz=splits.timezone)
        training_start = delivery_start - pd.DateOffset(days=365)
        training_stop = delivery_start
        training_index = data.actual.index[
            (data.actual.index >= training_start.tz_convert("UTC"))
            & (data.actual.index < training_stop.tz_convert("UTC"))
        ]
        training_days = pd.Index(
            training_index.tz_convert(splits.timezone).date
        ).unique()
        if len(training_days) != 365:
            raise TopologyExperimentError(
                f"{zone} {delivery_day}: fenetre train != 365 jours."
            )
        forecast_index = local_delivery_day_index(
            delivery_day,
            timezone=splits.timezone,
        )
        if not forecast_index.isin(data.evaluation_index).all():
            raise TopologyExperimentError(
                f"{zone} {delivery_day}: jour forecast hors evaluation."
            )
        origin = _forecast_origins_for_delivery(
            forecast_index,
            timezone_name=splits.timezone,
        )[0]
        if not bool((data.forecast_origins[data.actual.index.get_indexer(forecast_index)] == origin).all()):
            raise TopologyExperimentError(
                f"{zone} {delivery_day}: origin rolling365 divergent."
            )
        model = _new_corrector(config.model_parameters, correction_scale=scale)
        model.fit(
            context.loc[training_index],
            data.actual.loc[training_index],
            data.ensemble_base.loc[training_index],
        )
        predicted = model.predict(
            context.loc[forecast_index],
            data.ensemble_base.loc[forecast_index],
        )
        part = pd.DataFrame(
            {
                "delivery_start_utc": forecast_index,
                "forecast_origin_utc": origin,
                "actual": data.actual.loc[forecast_index].to_numpy(dtype=float),
            }
        )
        for quantile in QUANTILE_COLUMNS:
            part[f"ensemble__{quantile}"] = data.ensemble_base.loc[
                forecast_index, quantile
            ].to_numpy(dtype=float)
            part[f"residual_corrected__{quantile}"] = data.current_autonomous.loc[
                forecast_index, quantile
            ].to_numpy(dtype=float)
            part[f"topology_rolling365__{quantile}"] = predicted[quantile].to_numpy(
                dtype=float
            )
        correction = predicted.attrs.get("topology_correction")
        if not isinstance(correction, pd.Series):
            raise RuntimeError("Correction rolling365 absente du modele.")
        part["topology_rolling365_correction"] = correction.to_numpy(dtype=float)
        part["rolling_training_start_utc"] = training_index[0].isoformat()
        part["rolling_training_end_utc"] = training_index[-1].isoformat()
        part["rolling_training_days"] = 365
        part["rolling_training_hours"] = len(training_index)
        prediction_parts.append(part)
        audit = model.audit_metadata()
        audit_rows.append(
            {
                "delivery_day": str(delivery_day),
                "forecast_origin_utc": origin.isoformat(),
                "training_start_utc": training_index[0].isoformat(),
                "training_end_utc": training_index[-1].isoformat(),
                "training_local_days": 365,
                "training_physical_hours": int(len(training_index)),
                "n_training_rows": int(audit["n_training_rows"]),
                "n_dropped_target_rows": int(audit["n_dropped_target_rows"]),
                "fit_all_missing_feature_rows": int(
                    audit["fit_all_missing_feature_rows"]
                ),
                "prediction_hours": int(len(forecast_index)),
                "selected_radius": radius,
                "selected_scale": scale,
                "hyperparameter_sha256": expected_hyper,
                "target_availability_contract": (
                    "day_ahead_prices_through_D_minus_1_available_before_origin"
                ),
            }
        )
        if position == 1 or position % 30 == 0 or position == len(local_days):
            print(
                f"[{zone}] rolling365 {position}/{len(local_days)} jours",
                flush=True,
            )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    index = pd.DatetimeIndex(
        pd.to_datetime(predictions["delivery_start_utc"], utc=True),
        name="delivery_start_utc",
    )
    if not index.equals(data.evaluation_index) or len(predictions) != 8760:
        raise TopologyExperimentError(f"{zone}: sortie rolling365 != 8760 heures.")
    for model_name in (
        "ensemble",
        "residual_corrected",
        "topology_rolling365",
    ):
        _validate_quantiles(
            predictions.set_index("delivery_start_utc").loc[
                :, [f"{model_name}__{q}" for q in QUANTILE_COLUMNS]
            ].rename(columns={f"{model_name}__{q}": q for q in QUANTILE_COLUMNS}),
            expected_index=data.evaluation_index,
            name=f"{zone}_{model_name}_rolling365",
        )
    return Rolling365ZoneResult(
        zone=zone,
        timezone=splits.timezone,
        selected_radius=radius,
        selected_scale=scale,
        selected_arm=arm,
        recipe_sha256=recipe_sha256,
        hyperparameter_sha256=expected_hyper,
        predictions=predictions,
        refits=pd.DataFrame(audit_rows),
    )


def _rolling365_zone_worker(
    config_path: str,
    project_root: str,
    zone: str,
    recipe: Mapping[str, object],
    recipe_sha256: str,
) -> Rolling365ZoneResult:
    config = load_experiment_config(
        config_path,
        zones=(zone,),
        project_root=project_root,
        allow_existing_output=True,
    )
    return _fit_rolling365_zone(
        config,
        zone=zone,
        recipe=recipe,
        recipe_sha256=recipe_sha256,
    )


def _rolling365_metrics(
    frame: pd.DataFrame,
    *,
    model: str,
    baseline_model: str,
    timezone_name: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    index = pd.DatetimeIndex(
        pd.to_datetime(frame["delivery_start_utc"], utc=True),
        name="delivery_start_utc",
    )
    actual = pd.Series(frame["actual"].to_numpy(dtype=float), index=index)
    candidate = pd.Series(frame[f"{model}__q50"].to_numpy(dtype=float), index=index)
    baseline = pd.Series(
        frame[f"{baseline_model}__q50"].to_numpy(dtype=float), index=index
    )
    paired = paired_segment_metrics(
        actual,
        candidate,
        baseline,
        timezone_name=timezone_name,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    error = candidate - actual
    denominator = actual.abs().clip(lower=1.0)
    q10 = frame[f"{model}__q10"].to_numpy(dtype=float)
    q90 = frame[f"{model}__q90"].to_numpy(dtype=float)
    return {
        **asdict(paired),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mape_percent": float(np.mean(np.abs(error) / denominator) * 100.0),
        "bias": float(np.mean(error)),
        "p10_p90_empirical_coverage": float(
            np.mean((actual.to_numpy(dtype=float) >= q10) & (actual.to_numpy(dtype=float) <= q90))
        ),
    }


def _write_rolling365_reporting_inputs(
    directory: Path,
    frame: pd.DataFrame,
) -> None:
    inputs = directory / "inputs"
    inputs.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(
        {
            "timestamp": frame["delivery_start_utc"],
            "target": frame["actual"],
        }
    ).to_csv(inputs / "aligned_inputs.csv.gz", index=False)
    pd.DataFrame(
        {
            "timestamp": frame["delivery_start_utc"],
            "rolling_training_days": frame["rolling_training_days"],
        }
    ).to_csv(inputs / "model_covariates_with_future.csv.gz", index=False)
    pd.DataFrame(
        [
            {
                "zone": "rolling365",
                "alias": "rolling_training_days",
                "series": "sealed_rolling365_audit",
                "coverage_exact": 1.0,
                "coverage_after_fill": 1.0,
                "missing_after_fill": 0,
            }
        ]
    ).to_csv(inputs / "input_coverage.csv", index=False)
    pd.DataFrame(
        [
            {
                "alias": "rolling_training_days",
                "role": "training_window_audit",
                "series": "sealed_rolling365_audit",
                "description": "Nombre de jours locaux dans chaque refit",
                "future_strategies": "",
                "known_future": True,
                "source": "rolling365_refits",
            }
        ]
    ).to_csv(inputs / "input_manifest.csv", index=False)


def _rolling365_banner(
    *,
    zone: str,
    variant: str,
    metrics: Mapping[str, object],
    radius: int,
    scale: float,
) -> str:
    return f"""
<section id="topology-rolling365-audit" style="border:2px solid #2563eb;background:#eff6ff;border-radius:14px;padding:18px;margin-bottom:18px">
<p style="margin:0;color:#1d4ed8;font-weight:800;letter-spacing:.06em">ROLLING WINDOW 365 JOURS</p>
<h2 style="margin:6px 0">{html.escape(zone)} · {html.escape(variant)} · 365 refits causaux</h2>
<p>Chaque journée est prédite par un modèle réentraîné sur les 365 journées locales précédentes. Rayon {radius}, échelle {scale:g}; recette et hyperparamètres gelés avant ce backtest.</p>
<p><b>MAE candidat :</b> {float(metrics['candidate_mae']):.6f} €/MWh · <b>MAE référence :</b> {float(metrics['baseline_mae']):.6f} €/MWh · <b>gain :</b> {float(metrics['gain_eur_mwh']):+.6f} €/MWh · <b>win rate journalier :</b> {100.0 * float(metrics['daily_win_rate']):.1f}%.</p>
<p style="font-size:12px;color:#475569">Storm est uniquement un comparateur Statistics post-scellement. MKOnline n'est jamais une feature et ses poids de production restent inchangés.</p>
</section>"""


def _write_rolling365_variant(
    directory: Path,
    *,
    zone: str,
    timezone_name: str,
    variant: str,
    frame: pd.DataFrame,
    candidate_model: str,
    baseline_model: str,
    metrics: Mapping[str, object],
    radius: int,
    scale: float,
    storm: pd.Series | None,
    storm_audit: Mapping[str, object],
    final_directory: Path,
) -> Path:
    directory.mkdir(parents=True, exist_ok=False)
    backtest = frame.copy()
    if storm is not None:
        index = pd.DatetimeIndex(
            pd.to_datetime(backtest["delivery_start_utc"], utc=True)
        )
        backtest["storm_dashboard_official__q50"] = storm.reindex(index).to_numpy(
            dtype=float
        )
    _write_deterministic_csv_gzip(directory / "backtest_hourly_oof.csv.gz", backtest)
    local = pd.DatetimeIndex(
        pd.to_datetime(backtest["delivery_start_utc"], utc=True)
    ).tz_convert(timezone_name)
    last_day = local[-1].date()
    forecast = backtest.loc[pd.Index(local.date) == last_day].copy()
    forecast.to_csv(directory / f"forecast_hourly_{zone.lower()}.csv", index=False)
    metric_rows = []
    actual = backtest["actual"].to_numpy(dtype=float)
    for model in dict.fromkeys((baseline_model, candidate_model)):
        predicted = backtest[f"{model}__q50"].to_numpy(dtype=float)
        metric_rows.append(
            {
                "model": model,
                "mae": float(np.mean(np.abs(actual - predicted))),
                "n_scored": int(len(backtest)),
                "n_expected": int(len(backtest)),
                "prediction_coverage": 1.0,
                "score_coverage": 1.0,
            }
        )
    pd.DataFrame(metric_rows).to_csv(directory / "metrics_hourly.csv", index=False)
    _write_json(
        directory / "metrics_hourly.json",
        {
            "metrics": metric_rows,
            "training_diagnostics": {
                "metric_scope": "rolling365_daily_refit_365_delivery_days",
                "evaluation_start_local_date": str(local[0].date()),
                "evaluation_end_local_date": str(last_day),
                "rolling_window_local_days": 365,
                "n_refits": 365,
                "candidate_model": candidate_model,
                "baseline_model": baseline_model,
            },
        },
    )
    _write_json(
        directory / "run_manifest.json",
        {
            "schema_version": 1,
            "run_type": "topology_rolling365_backtest",
            "zone": zone,
            "timezone": timezone_name,
            "variant": variant,
            "candidate_model": candidate_model,
            "baseline_model": baseline_model,
            "rolling365_enabled": True,
            "rolling_window_local_days": 365,
            "refit_frequency_delivery_days": 1,
            "n_refits": 365,
            "production_changed": False,
            "runs_live_written": False,
        },
    )
    statistics = backtest.copy()
    _write_deterministic_csv_gzip(
        directory / "statistics_history_hourly.csv.gz", statistics
    )
    statistics_audit: dict[str, object] = {
        "schema_version": 1,
        "status": "complete_rolling365_backtest",
        "statistics_scope": "rolling365_daily_refit_365_delivery_days",
        "variant": variant,
        "statistics_history_path": "statistics_history_hourly.csv.gz",
        "statistics_history_sha256": _sha256(
            directory / "statistics_history_hourly.csv.gz"
        ),
        "report_scope_note": (
            "365 jours de previsions walk-forward; chaque jour utilise un refit "
            "sur les 365 jours locaux precedents."
        ),
    }
    if storm is not None:
        storm_values = pd.to_numeric(
            statistics["storm_dashboard_official__q50"], errors="coerce"
        ).to_numpy(dtype=float)
        statistics_index = pd.DatetimeIndex(
            pd.to_datetime(statistics["delivery_start_utc"], utc=True),
            name="delivery_start_utc",
        )
        missing_mask = ~np.isfinite(storm_values)
        actual_missing = set(statistics_index[missing_mask])
        allowed_raw = storm_audit.get("native_allowed_missing_utc", [])
        if not isinstance(allowed_raw, list):
            raise TopologyExperimentError(
                f"{zone}: allow-list DST Storm rolling365 invalide."
            )
        allowed_missing = {
            (
                pd.Timestamp(value).tz_convert("UTC")
                if pd.Timestamp(value).tzinfo is not None
                else pd.Timestamp(value, tz="UTC")
            )
            for value in allowed_raw
        }.intersection(set(statistics_index))
        if actual_missing != allowed_missing:
            raise TopologyExperimentError(
                f"{zone}: trous Storm rolling365 differents de l'audit DST."
            )
        statistics_audit["storm_primary_report_benchmark"] = (
            "storm_dashboard_official__q50"
        )
        statistics_audit["storm_dashboard"] = {
            "column": "storm_dashboard_official__q50",
            "benchmark_contract": "native_dashboard_snapshot",
            "expected_hours": int(len(statistics)),
            "available_hours": int(np.count_nonzero(~missing_mask)),
            "missing_hours": int(np.count_nonzero(missing_mask)),
            "coverage": float(np.mean(~missing_mask)),
            "used_for_prediction": False,
            "used_for_gate": False,
            "candidate_frozen_before_comparator_attachment": True,
            "source_sha256": storm_audit.get("source_sha256"),
            "source_audit_sha256": storm_audit.get("source_audit_sha256"),
            "dst": {
                "interpolation": False,
                "strict_08_fallback": False,
                "native_allowed_missing_hours": len(allowed_missing),
                "native_allowed_missing_utc": [
                    timestamp.isoformat()
                    for timestamp in sorted(allowed_missing)
                ],
                "native_actual_missing_matches_allowed": True,
            },
        }
    else:
        statistics_audit["storm_dashboard"] = dict(storm_audit)
    _write_json(directory / "statistics_history_audit.json", statistics_audit)
    _write_json(
        directory / "rolling365_statistics.json",
        {
            "schema_version": 1,
            "zone": zone,
            "variant": variant,
            "candidate_model": candidate_model,
            "baseline_model": baseline_model,
            "statistics": dict(metrics),
            "n_hours": int(len(frame)),
            "n_days": 365,
        },
    )
    _write_rolling365_reporting_inputs(directory, frame)
    reports = directory / "reports"
    reports.mkdir(parents=True, exist_ok=False)
    report_path = reports / f"topology_rolling365_{zone.lower()}_{variant}.html"
    write_hourly_html_report(
        directory,
        output_path=report_path,
        title=f"Chronos-2 {zone} · rolling365 · {variant}",
        native_model=candidate_model,
        baseline_model=baseline_model,
        zone=zone,
        timezone=timezone_name,
        history_hours=8760,
    )
    source = report_path.read_text(encoding="utf-8")
    if "<main>" not in source:
        raise TopologyExperimentError("Point d'insertion HTML rolling365 absent.")
    source = source.replace(
        "<main>",
        "<main>" + _rolling365_banner(
            zone=zone,
            variant=variant,
            metrics=metrics,
            radius=radius,
            scale=scale,
        ),
        1,
    )
    report_path.write_text(source, encoding="utf-8")
    _write_artifact_checksums(directory, declared_directory=final_directory)
    return report_path


def _write_rolling365_index(
    output: Path,
    *,
    rows: Sequence[Mapping[str, object]],
) -> Path:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td><b>{html.escape(str(row['zone']))}</b></td>"
            f"<td>{html.escape(str(row['variant']))}</td>"
            f"<td>{float(row['candidate_mae']):.6f}</td>"
            f"<td>{float(row['baseline_mae']):.6f}</td>"
            f"<td>{float(row['gain_eur_mwh']):+.6f}</td>"
            f"<td>{100.0 * float(row['daily_win_rate']):.1f}%</td>"
            f"<td><a href=\"{html.escape(str(row['href']))}\">Ouvrir</a></td>"
            "</tr>"
        )
    document = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Chronos-2 · rolling365 topologique</title><style>body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f8;color:#172033}}header{{background:#0f172a;color:white;padding:28px 5vw}}main{{max-width:1400px;margin:24px auto;padding:0 24px}}section{{background:white;border:1px solid #dbe2ea;border-radius:14px;padding:20px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #dbe2ea;text-align:left}}th{{background:#eff6ff}}a{{color:#0369a1}}</style></head><body><header><h1>Backtest topologique · rolling window 365 jours</h1><p>365 refits par pays, un entraînement sur les 365 jours locaux précédents pour chaque journée évaluée.</p></header><main><section><table><thead><tr><th>Pays</th><th>Variante</th><th>MAE candidat</th><th>MAE référence</th><th>Gain</th><th>Win rate journalier</th><th>Rapport</th></tr></thead><tbody>{''.join(body)}</tbody></table></section></main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output


def run_topology_rolling365(
    config: ExperimentConfig,
    *,
    calibration_dir: str | Path,
    output_dir: str | Path = DEFAULT_ROLLING365_OUTPUT_DIRECTORY,
    expected_calibration_manifest_sha256: str = (
        CALIBRATION_V1_ARTIFACT_MANIFEST_SHA256
    ),
    workers: int = 1,
) -> Path:
    """Train 365 causal daily refits per zone and publish Statistics/HTML."""

    if isinstance(workers, bool) or int(workers) < 1:
        raise TopologyExperimentError("rolling365 workers doit etre >= 1.")
    destination = _assert_experiment_output(
        output_dir,
        project_root=config.contract.project_root,
    )
    if destination.exists():
        raise FileExistsError(f"Sortie rolling365 deja publiee: {destination}")
    calibration = audit_published_calibration(
        config,
        calibration_dir,
        expected_manifest_sha256=expected_calibration_manifest_sha256,
    )
    source_audits = audit_experiment_sources(config)
    recipes = {zone: calibration.recipes[zone] for zone in config.zones}
    recipe_shas = {
        zone: _sha256(calibration.directory / zone.lower() / "topology_recipe_seal.json")
        for zone in config.zones
    }
    results: dict[str, Rolling365ZoneResult] = {}
    if int(workers) == 1 or len(config.zones) == 1:
        for zone in config.zones:
            results[zone] = _fit_rolling365_zone(
                config,
                zone=zone,
                recipe=recipes[zone],
                recipe_sha256=recipe_shas[zone],
            )
    else:
        maximum = min(int(workers), len(config.zones))
        # Threads avoid Windows spawn/handle failures while the underlying
        # sklearn fit releases the GIL and is itself constrained to one core.
        with ThreadPoolExecutor(max_workers=maximum) as executor:
            futures = {
                executor.submit(
                    _rolling365_zone_worker,
                    str(config.source_path),
                    str(config.contract.project_root),
                    zone,
                    dict(recipes[zone]),
                    recipe_shas[zone],
                ): zone
                for zone in config.zones
            }
            for future in as_completed(futures):
                zone = futures[future]
                results[zone] = future.result()
                print(f"[{zone}] rolling365 termine", flush=True)

    def writer(staging: Path) -> None:
        index_rows: list[dict[str, object]] = []
        zone_entries: dict[str, object] = {}
        for zone in config.zones:
            result = results[zone]
            zone_dir = staging / zone.lower()
            zone_dir.mkdir(parents=True, exist_ok=False)
            prediction_path = zone_dir / "rolling365_predictions.csv.gz"
            refit_path = zone_dir / "rolling365_refits.csv.gz"
            _write_deterministic_csv_gzip(prediction_path, result.predictions)
            _write_deterministic_csv_gzip(refit_path, result.refits)
            seal_path = zone_dir / "rolling365_prediction_seal.json"
            _write_json(
                seal_path,
                {
                    "schema_version": 1,
                    "zone": zone,
                    "report_type": "topology_rolling365_daily_refit",
                    "prediction_path": prediction_path.name,
                    "prediction_sha256": _sha256(prediction_path),
                    "refit_path": refit_path.name,
                    "refit_sha256": _sha256(refit_path),
                    "n_evaluation_days": 365,
                    "n_evaluation_hours": 8760,
                    "n_refits": 365,
                    "rolling_window_local_days": 365,
                    "selected_radius": result.selected_radius,
                    "selected_scale": result.selected_scale,
                    "selected_arm": result.selected_arm,
                    "recipe_sha256": result.recipe_sha256,
                    "hyperparameter_sha256": result.hyperparameter_sha256,
                    "comparators_loaded_before_prediction_seal": False,
                    "storm_used_for_prediction": False,
                    "mkonline_used_as_feature": False,
                    "production_changed": False,
                    "runs_live_written": False,
                },
            )
            seal = CandidateSeal(
                prediction_path=prediction_path,
                manifest_path=seal_path,
                prediction_sha256=_sha256(prediction_path),
                manifest_sha256=_sha256(seal_path),
            )
            frame = result.predictions.copy()
            eval_index = pd.DatetimeIndex(
                pd.to_datetime(frame["delivery_start_utc"], utc=True),
                name="delivery_start_utc",
            )
            storm, storm_audit = _load_annual_storm_reference(
                seal=seal,
                zone=zone,
                evaluation=calibration.evaluations[zone],
                expected_index=eval_index,
            )
            autonomous_metrics = _rolling365_metrics(
                frame,
                model="topology_rolling365",
                baseline_model="residual_corrected",
                timezone_name=result.timezone,
                bootstrap_samples=config.bootstrap_samples,
                bootstrap_seed=config.bootstrap_seed,
            )
            auto_dir = zone_dir / "autonomous"
            auto_report = _write_rolling365_variant(
                auto_dir,
                zone=zone,
                timezone_name=result.timezone,
                variant="autonomous",
                frame=frame,
                candidate_model="topology_rolling365",
                baseline_model="residual_corrected",
                metrics=autonomous_metrics,
                radius=result.selected_radius,
                scale=result.selected_scale,
                storm=storm,
                storm_audit=storm_audit,
                final_directory=destination / zone.lower() / "autonomous",
            )
            index_rows.append(
                {
                    "zone": zone,
                    "variant": "autonomous",
                    **autonomous_metrics,
                    "href": (
                        Path("..") / auto_report.relative_to(staging)
                    ).as_posix(),
                }
            )
            variants: dict[str, object] = {
                "autonomous": {
                    "directory": "autonomous",
                    "artifact_manifest_sha256": _sha256(
                        auto_dir / "artifact_checksums.json"
                    ),
                    "report_sha256": _sha256(auto_report),
                    "statistics": autonomous_metrics,
                }
            }
            if zone in BLEND_ZONES:
                rolling_base = frame.set_index("delivery_start_utc").loc[
                    :, [f"topology_rolling365__{q}" for q in QUANTILE_COLUMNS]
                ].rename(
                    columns={f"topology_rolling365__{q}": q for q in QUANTILE_COLUMNS}
                )
                rolling_base.index = eval_index
                current_base = frame.set_index("delivery_start_utc").loc[
                    :, [f"residual_corrected__{q}" for q in QUANTILE_COLUMNS]
                ].rename(
                    columns={f"residual_corrected__{q}": q for q in QUANTILE_COLUMNS}
                )
                current_base.index = eval_index
                rolling_blend, blend_audit = load_comparator_after_seal(
                    seal,
                    lambda: _load_annual_production_blend_reference(
                        config.contract,
                        zone=zone,
                        base=rolling_base,
                        expected_index=eval_index,
                    ),
                )
                current_blend, _ = load_comparator_after_seal(
                    seal,
                    lambda: _load_annual_production_blend_reference(
                        config.contract,
                        zone=zone,
                        base=current_base,
                        expected_index=eval_index,
                    ),
                )
                for quantile in QUANTILE_COLUMNS:
                    frame[f"topology_rolling365_mkonline_blend__{quantile}"] = (
                        rolling_blend[quantile].to_numpy(dtype=float)
                    )
                    frame[f"mkonline_blend__{quantile}"] = current_blend[
                        quantile
                    ].to_numpy(dtype=float)
                blend_metrics = _rolling365_metrics(
                    frame,
                    model="topology_rolling365_mkonline_blend",
                    baseline_model="mkonline_blend",
                    timezone_name=result.timezone,
                    bootstrap_samples=config.bootstrap_samples,
                    bootstrap_seed=config.bootstrap_seed,
                )
                blend_dir = zone_dir / "mkonline_blend"
                blend_report = _write_rolling365_variant(
                    blend_dir,
                    zone=zone,
                    timezone_name=result.timezone,
                    variant="mkonline_blend",
                    frame=frame,
                    candidate_model="topology_rolling365_mkonline_blend",
                    baseline_model="mkonline_blend",
                    metrics=blend_metrics,
                    radius=result.selected_radius,
                    scale=result.selected_scale,
                    storm=storm,
                    storm_audit={**storm_audit, "blend_source": blend_audit},
                    final_directory=destination / zone.lower() / "mkonline_blend",
                )
                index_rows.append(
                    {
                        "zone": zone,
                        "variant": "mkonline_blend",
                        **blend_metrics,
                        "href": (
                            Path("..") / blend_report.relative_to(staging)
                        ).as_posix(),
                    }
                )
                variants["mkonline_blend"] = {
                    "directory": "mkonline_blend",
                    "artifact_manifest_sha256": _sha256(
                        blend_dir / "artifact_checksums.json"
                    ),
                    "report_sha256": _sha256(blend_report),
                    "statistics": blend_metrics,
                    "weights_recomputed": False,
                    "blend_source_audit": blend_audit,
                }
            _write_json(
                zone_dir / "rolling365_zone_manifest.json",
                {
                    "schema_version": 1,
                    "zone": zone,
                    "timezone": result.timezone,
                    "prediction_sha256": seal.prediction_sha256,
                    "prediction_seal_sha256": seal.manifest_sha256,
                    "refit_sha256": _sha256(refit_path),
                    "source_backtest_sha256": config.contract.sources[
                        zone
                    ].backtest_sha256,
                    "source_audit": source_audits[zone],
                    "recipe_sha256": result.recipe_sha256,
                    "variants": variants,
                    "n_refits": 365,
                    "rolling_window_local_days": 365,
                    "production_changed": False,
                    "runs_live_written": False,
                },
            )
            zone_manifest_path = _write_artifact_checksums(
                zone_dir,
                declared_directory=destination / zone.lower(),
            )
            zone_entries[zone] = {
                "directory": zone.lower(),
                "artifact_manifest_sha256": _sha256(zone_manifest_path),
                "variants": variants,
            }
        index_path = _write_rolling365_index(
            staging / "reports" / "topology_rolling365_index.html",
            rows=index_rows,
        )
        # Last-moment integrity gate: all training and comparator inputs are
        # rehashed after report generation and immediately before publication.
        if _sha256(config.source_path) != config.config_sha256:
            raise TopologyExperimentError(
                "La configuration rolling365 a change pendant l'execution."
            )
        if (
            _sha256(calibration.directory / "artifact_checksums.json")
            != calibration.artifact_manifest_sha256
        ):
            raise TopologyExperimentError(
                "La calibration rolling365 a change pendant l'execution."
            )
        for source_zone, source in config.contract.sources.items():
            source_members = (
                (source.checksum_manifest, source.checksum_manifest_sha256),
                (source.backtest_file, source.backtest_sha256),
                (source.aligned_inputs_file, source.aligned_inputs_sha256),
                (source.run_manifest_file, source.run_manifest_sha256),
                (source.feature_manifest_file, source.feature_manifest_sha256),
                (source.recipe_file, source.recipe_sha256),
                (source.metrics_file, source.metrics_sha256),
                (source.target_cache, source.target_cache_sha256),
            )
            for member, expected_sha in source_members:
                if _sha256(member) != expected_sha:
                    raise TopologyExperimentError(
                        f"{source_zone}: source rolling365 modifiee avant publication."
                    )
        for blend_zone in BLEND_ZONES:
            dependency = config.contract.blend_dependency_for(blend_zone)
            blend_members = (
                (dependency.recipe_manifest, dependency.recipe_manifest_sha256),
                (
                    dependency.dependency_manifest,
                    dependency.dependency_manifest_sha256,
                ),
                (dependency.forecast_file, dependency.forecast_file_sha256),
            )
            for member, expected_sha in blend_members:
                if _sha256(member) != expected_sha:
                    raise TopologyExperimentError(
                        f"{blend_zone}: comparateur MKOnline modifie avant publication."
                    )
        _write_json(
            staging / "rolling365_manifest.json",
            {
                "schema_version": 1,
                "run_type": "topology_rolling365_daily_refit",
                "experiment_id": config.experiment_id,
                "zones": list(config.zones),
                "evaluation_start_local_day": "2025-08-12",
                "evaluation_end_local_day": "2026-08-11",
                "evaluation_local_days": 365,
                "evaluation_physical_hours": 8760,
                "rolling_window_local_days": 365,
                "refit_frequency_delivery_days": 1,
                "n_refits_per_zone": 365,
                "base_model_for_topology": "ensemble",
                "comparison_current_autonomous": "residual_corrected",
                "selection_repeated_on_rolling_results": False,
                "storm_used_for_prediction": False,
                "mkonline_used_as_feature": False,
                "mkonline_weights_recomputed": False,
                "calibration_artifact_manifest_sha256": (
                    calibration.artifact_manifest_sha256
                ),
                "config_sha256": config.config_sha256,
                "zone_artifacts": zone_entries,
                "index_path": index_path.relative_to(staging).as_posix(),
                "index_sha256": _sha256(index_path),
                "production_changed": False,
                "runs_live_written": False,
            },
        )
        _write_artifact_checksums(staging, declared_directory=destination)

    return atomic_experiment_publish(
        destination,
        project_root=config.contract.project_root,
        writer=writer,
    )


def normalise_daily_mode(value: str) -> str:
    mode = str(value).strip().casefold()
    if mode not in DAILY_MODES:
        raise TopologyExperimentError(
            "mode doit etre production, autonomous, blend ou both."
        )
    return mode


def _normalise_daily_zones(zones: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(zone).strip().upper() for zone in zones)
    if not normalized:
        raise TopologyExperimentError("Selectionnez au moins une zone.")
    if len(set(normalized)) != len(normalized):
        raise TopologyExperimentError("La selection quotidienne contient un doublon.")
    invalid = [zone for zone in normalized if zone not in SUPPORTED_ZONES]
    if invalid:
        raise TopologyExperimentError(f"Zones quotidiennes invalides: {invalid}.")
    return tuple(zone for zone in SUPPORTED_ZONES if zone in normalized)


def _daily_variants_for_zone(mode: str, zone: str) -> tuple[str, ...]:
    selected_mode = normalise_daily_mode(mode)
    canonical = str(zone).strip().upper()
    if canonical not in SUPPORTED_ZONES:
        raise TopologyExperimentError(f"Zone quotidienne invalide: {zone}.")
    if selected_mode == "autonomous":
        return ("autonomous",)
    if selected_mode == "blend":
        if canonical not in BLEND_ZONES:
            raise TopologyExperimentError(
                f"Le mode blend est reserve a FR/NL; zone recue={canonical}."
            )
        return ("mkonline_blend",)
    if selected_mode == "both":
        return (
            ("autonomous", "mkonline_blend")
            if canonical in BLEND_ZONES
            else ("autonomous",)
        )
    # Production is one recommended output per country: the unchanged
    # production blend for FR/NL, promoted topology for BE, identity otherwise.
    return (
        ("mkonline_blend",)
        if canonical in BLEND_ZONES
        else ("autonomous",)
    )


def _normalise_delivery_day(value: str | date) -> str:
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.isoformat()
    try:
        return date.fromisoformat(str(value)).isoformat()
    except ValueError as exc:
        raise TopologyExperimentError(
            "delivery-day doit respecter YYYY-MM-DD."
        ) from exc


def _load_exact_daily_archives(
    *,
    project_root: Path,
    zones: Sequence[str],
    delivery_day: str,
) -> dict[str, DailyArchive]:
    """Open an explicit day only; a newer latest archive is irrelevant."""

    from chronos2_hourly.app_service import (
        inspect_zone_statuses,
        validate_existing_forecast_archive,
    )

    requested = _normalise_daily_zones(zones)
    registry = project_root / "chronos2_hourly_live_zones.yaml"
    statuses = inspect_zone_statuses(registry, zones=requested)
    status_by_zone = {str(status.code).upper(): status for status in statuses}
    result: dict[str, DailyArchive] = {}
    for zone in requested:
        status = status_by_zone.get(zone)
        if status is None or not status.launchable:
            raise TopologyExperimentError(f"{zone}: statut live non validable.")
        archive = validate_existing_forecast_archive(
            status,
            project_root=project_root,
            delivery_day=delivery_day,
        )
        if archive is None:
            raise TopologyExperimentError(
                f"{zone}: archive exacte absente pour {delivery_day}."
            )
        archive = Path(archive).resolve()
        forecast_path = archive / f"forecast_hourly_{zone.lower()}.csv"
        if not forecast_path.is_file():
            raise FileNotFoundError(forecast_path)
        checksum_manifest = archive / "artifact_checksums.json"
        if not checksum_manifest.is_file():
            raise FileNotFoundError(checksum_manifest)
        result[zone] = DailyArchive(
            zone=zone,
            timezone=str(status.timezone),
            delivery_day=delivery_day,
            directory=archive,
            forecast_path=forecast_path,
            artifact_manifest_sha256=_sha256(checksum_manifest),
        )
    return result


def _verified_archive_member(archive: DailyArchive, path: Path) -> str:
    """Require a live input/report to be covered by its audited run manifest."""

    target = path.resolve()
    if not target.is_relative_to(archive.directory) or not target.is_file():
        raise TopologyExperimentError(
            f"{archive.zone}: membre archive invalide: {target}."
        )
    relative = target.relative_to(archive.directory).as_posix()
    payload = _read_json_object(
        archive.directory / "artifact_checksums.json",
        name=f"{archive.zone} artifact_checksums",
    )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise TopologyExperimentError(f"{archive.zone}: artifacts live invalides.")
    matches = [
        item
        for item in artifacts
        if isinstance(item, Mapping) and item.get("path") == relative
    ]
    if len(matches) != 1:
        raise TopologyExperimentError(
            f"{archive.zone}: membre non scelle dans l'archive: {relative}."
        )
    expected_sha = matches[0].get("sha256")
    observed_sha = _sha256(target)
    if not isinstance(expected_sha, str) or expected_sha != observed_sha:
        raise TopologyExperimentError(
            f"{archive.zone}: SHA archive divergent pour {relative}."
        )
    return observed_sha


def _read_daily_archive_quantiles(
    archive: DailyArchive,
    *,
    variant: str,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    from chronos2_hourly.app_service import load_forecast_curve

    before = _verified_archive_member(archive, archive.forecast_path)
    dataset = load_forecast_curve(
        archive.forecast_path,
        timezone_name=archive.timezone,
        variant=variant,
    )
    if _sha256(archive.forecast_path) != before:
        raise TopologyExperimentError(
            f"{archive.zone}: forecast modifie pendant la lecture."
        )
    index = pd.DatetimeIndex(
        pd.to_datetime(dataset.frame["timestamp"], utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    expected = local_delivery_day_index(
        archive.delivery_day,
        timezone=archive.timezone,
    )
    if not index.equals(expected):
        raise TopologyExperimentError(
            f"{archive.zone}: timeline {variant} differente du jour exact."
        )
    quantiles = pd.DataFrame(
        {
            "q10": dataset.frame["P10"].to_numpy(dtype=float),
            "q50": dataset.frame["P50"].to_numpy(dtype=float),
            "q90": dataset.frame["P90"].to_numpy(dtype=float),
        },
        index=index,
    )
    quantiles = _validate_quantiles(
        quantiles,
        expected_index=index,
        name=f"{archive.zone}_{variant}_daily",
    )
    raw = pd.read_csv(archive.forecast_path, usecols=["forecast_origin_utc"])
    origins = pd.DatetimeIndex(
        pd.to_datetime(raw["forecast_origin_utc"], utc=True, errors="raise"),
        name="forecast_origin_utc",
    )
    if len(origins) != len(index) or bool((origins >= index).any()):
        raise TopologyExperimentError(
            f"{archive.zone}: origines quotidiennes non causales."
        )
    return quantiles, origins


def _load_operational_be_context(
    *,
    contract: TopologyExperimentContract,
    archives: Mapping[str, DailyArchive],
    forecast_index: pd.DatetimeIndex,
    radius: int,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Build BE D+1 context only from fully audited same-day archives."""

    missing_archives = sorted(set(SUPPORTED_ZONES).difference(archives))
    if missing_archives:
        raise TopologyExperimentError(
            f"BE: archives de dependance absentes: {missing_archives}."
        )
    extended = pd.date_range(
        forecast_index[0] - pd.Timedelta(hours=24),
        forecast_index[-1],
        freq="h",
        name="delivery_start_utc",
    )
    residual_path = (
        archives["BE"].directory
        / "inputs"
        / "model_covariates_with_future.csv.gz"
    )
    residual_sha = _verified_archive_member(archives["BE"], residual_path)
    residual_raw = _read_csv_window_without_future_outcomes(
        residual_path,
        timestamp_column="timestamp",
        columns=tuple(contract.residual_load_columns.values()),
        expected_index=extended,
    )
    if _sha256(residual_path) != residual_sha:
        raise TopologyExperimentError("BE: covariables modifiees pendant lecture.")
    residual = pd.DataFrame(
        {
            zone: pd.to_numeric(
                residual_raw[contract.residual_load_columns[zone]],
                errors="coerce",
            )
            for zone in SUPPORTED_ZONES
        },
        index=extended,
    )
    history = pd.date_range(
        extended[0],
        forecast_index[0] - pd.Timedelta(hours=1),
        freq="h",
        name="delivery_start_utc",
    )
    prices = pd.DataFrame(np.nan, index=extended, columns=list(SUPPORTED_ZONES))
    source_hashes = {"be_future_covariates_sha256": residual_sha}
    for zone in SUPPORTED_ZONES:
        aligned = archives[zone].directory / "inputs" / "aligned_inputs.csv.gz"
        aligned_sha = _verified_archive_member(archives[zone], aligned)
        raw = _read_csv_window_without_future_outcomes(
            aligned,
            timestamp_column="timestamp",
            columns=("target",),
            expected_index=history,
        )
        if _sha256(aligned) != aligned_sha:
            raise TopologyExperimentError(
                f"{zone}: aligned_inputs modifie pendant lecture."
            )
        prices.loc[history, zone] = pd.to_numeric(
            raw["target"], errors="coerce"
        ).to_numpy(dtype=float)
        source_hashes[f"{zone.lower()}_aligned_inputs_sha256"] = aligned_sha
    full = build_topology_context(
        residual,
        prices,
        target_zone="BE",
        radius=radius,
        timezone=contract.sources["BE"].timezone,
    )
    return _slice_context(full, forecast_index), source_hashes


def _daily_forecast_frame(
    *,
    index: pd.DatetimeIndex,
    origins: pd.DatetimeIndex,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    if variant == "autonomous":
        baseline_model = "residual_corrected"
        candidate_model = "topology_autonomous"
    elif variant == "mkonline_blend":
        baseline_model = "mkonline_blend"
        candidate_model = "topology_mkonline_blend"
    else:
        raise TopologyExperimentError(f"Variante quotidienne invalide: {variant}.")
    baseline = _validate_quantiles(
        baseline,
        expected_index=index,
        name=f"daily_{baseline_model}",
    )
    candidate = _validate_quantiles(
        candidate,
        expected_index=index,
        name=f"daily_{candidate_model}",
    )
    frame = pd.DataFrame(
        {
            "delivery_start_utc": index,
            "forecast_origin_utc": origins,
        }
    )
    for quantile in QUANTILE_COLUMNS:
        frame[f"{baseline_model}__{quantile}"] = baseline[quantile].to_numpy(
            dtype=float
        )
        frame[f"{candidate_model}__{quantile}"] = candidate[quantile].to_numpy(
            dtype=float
        )
    return frame


def _apply_daily_autonomous_policy(
    *,
    zone: str,
    base: pd.DataFrame,
    model: TopologyResidualCorrector | None = None,
    context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply BE topology only; every other autonomous zone is exact identity."""

    canonical = str(zone).strip().upper()
    validated = _validate_quantiles(base, name=f"{canonical}_daily_base")
    if canonical != "BE":
        return validated.copy()
    if model is None or context is None:
        raise TopologyExperimentError("BE: modele/contexte operationnel absent.")
    return _validate_quantiles(
        model.predict(context, validated),
        expected_index=validated.index,
        name="BE_daily_topology",
    )


def _write_daily_passthrough_blend(
    *,
    directory: Path,
    published_directory: Path,
    zone: str,
    forecast: pd.DataFrame,
    archive: DailyArchive,
    application: Mapping[str, object],
) -> Path:
    """Copy the exact audited production blend and its current live HTML report."""

    directory.mkdir(parents=True, exist_ok=False)
    forecast_path = directory / f"forecast_hourly_{zone.lower()}.csv"
    forecast.to_csv(forecast_path, index=False)
    reports = sorted(archive.directory.glob("*.html"))
    if len(reports) != 1:
        raise TopologyExperimentError(
            f"{zone}: rapport live unique introuvable ({len(reports)} trouves)."
        )
    source_report = reports[0]
    source_sha = _verified_archive_member(archive, source_report)
    report_directory = directory / "reports"
    report_directory.mkdir()
    report_path = report_directory / f"topology_{zone.lower()}_mkonline_blend.html"
    shutil.copy2(source_report, report_path)
    if _sha256(source_report) != source_sha or _sha256(report_path) != source_sha:
        raise TopologyExperimentError(f"{zone}: rapport blend modifie pendant copie.")
    proof = {
        **dict(application),
        "zone": zone,
        "variant": "mkonline_blend",
        "policy": "production_mkonline_blend_passthrough",
        "topology_blend_promoted": False,
        "weights_recomputed": False,
        "weights_changed": False,
        "source_report": str(source_report),
        "source_report_sha256": source_sha,
        "candidate_equals_production_blend": True,
    }
    _write_json(directory / "daily_application.json", proof)
    _write_json(
        directory / "run_manifest.json",
        {
            "schema_version": 1,
            "run_type": "daily_fixed_pricefm_topology_application",
            "zone": zone,
            "timezone": archive.timezone,
            "variant": "mkonline_blend",
            "delivery_day": archive.delivery_day,
            "policy": "production_mkonline_blend_passthrough",
            "production_changed": False,
            "runs_live_written": False,
            "weights_recomputed": False,
        },
    )
    _write_artifact_checksums(
        directory,
        declared_directory=published_directory,
    )
    return report_path


def _revalidate_consumed_archive_members(
    archives: Mapping[str, DailyArchive],
    consumed_members: Mapping[Path, str],
) -> None:
    """Rehash every live byte used by inference/reporting before publication."""

    for path, expected_sha in consumed_members.items():
        target = path.resolve()
        owners = [
            archive
            for archive in archives.values()
            if target.is_relative_to(archive.directory)
        ]
        if len(owners) != 1:
            raise TopologyExperimentError(
                f"Membre live consomme sans archive unique: {target}."
            )
        observed_sha = _verified_archive_member(owners[0], target)
        if observed_sha != expected_sha:
            raise TopologyExperimentError(
                f"Membre live modifie pendant l'application: {target}."
            )
    for zone, archive in archives.items():
        if _sha256(archive.directory / "artifact_checksums.json") != (
            archive.artifact_manifest_sha256
        ):
            raise TopologyExperimentError(
                f"{zone}: manifest archive live modifie pendant l'application."
            )


def _write_daily_index(
    output_path: Path,
    *,
    delivery_day: str,
    mode: str,
    reports: Sequence[tuple[str, str, Path, str]],
) -> Path:
    rows = []
    for zone, variant, report_path, policy in reports:
        href = report_path.relative_to(output_path.parent).as_posix()
        rows.append(
            "<tr>"
            f"<td>{html.escape(zone)}</td>"
            f"<td>{html.escape(variant)}</td>"
            f"<td>{html.escape(policy)}</td>"
            f'<td><a href="{html.escape(href)}">Ouvrir le rapport</a></td>'
            "</tr>"
        )
    source = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Forecast topologique quotidien {html.escape(delivery_day)}</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:2rem;background:#f5f7fa;color:#16202a}}
table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:.7rem;border:1px solid #d8dee8;text-align:left}}
th{{background:#edf2f7}}.note{{padding:1rem;background:#fff4d6;border-left:4px solid #d89b00;margin-bottom:1rem}}</style>
</head><body><h1>Forecast topologique quotidien — {html.escape(delivery_day)}</h1>
<div class="note">Mode {html.escape(mode)}. Calibration fixe publiée : ce run n'est pas un rolling-365 et ne modifie jamais runs/live.</div>
<table><thead><tr><th>Pays</th><th>Variante</th><th>Politique</th><th>Rapport</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></body></html>"""
    output_path.write_text(source, encoding="utf-8")
    return output_path


def _validate_existing_daily_output(
    directory: Path,
    *,
    delivery_day: str,
    mode: str,
    zones: Sequence[str],
    bundle_sha256: str,
) -> Path:
    _validate_artifact_checksums(directory)
    manifest = _read_json_object(directory / "daily_manifest.json", name="daily manifest")
    expected = {
        "delivery_day": delivery_day,
        "mode": mode,
        "zones": list(zones),
        "operational_bundle_artifact_manifest_sha256": bundle_sha256,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise TopologyExperimentError(
            f"Sortie quotidienne existante incompatible: {mismatches}."
        )
    if manifest.get("complete") is not True:
        raise TopologyExperimentError("Sortie quotidienne existante incomplete.")
    return directory


def _validate_existing_daily_source_fingerprints(
    directory: Path,
    *,
    archives: Mapping[str, DailyArchive],
) -> None:
    """Require an idempotent rerun to resolve to the identical live sources."""

    manifest = _read_json_object(directory / "daily_manifest.json", name="daily manifest")
    expected_sources = manifest.get("source_archives")
    if not isinstance(expected_sources, Mapping) or set(expected_sources) != set(
        archives
    ):
        raise TopologyExperimentError(
            "Empreintes des archives sources absentes ou de couverture differente."
        )
    for zone, archive in archives.items():
        expected = expected_sources.get(zone)
        if not isinstance(expected, Mapping):
            raise TopologyExperimentError(f"{zone}: empreinte source absente.")
        observed_forecast_sha = _verified_archive_member(
            archive,
            archive.forecast_path,
        )
        observed = {
            "directory": str(archive.directory),
            "artifact_manifest_sha256": archive.artifact_manifest_sha256,
            "forecast_sha256": observed_forecast_sha,
        }
        if dict(expected) != observed:
            raise TopologyExperimentError(
                f"{zone}: empreinte source live differente du run existant."
            )
        if _sha256(archive.directory / "artifact_checksums.json") != (
            archive.artifact_manifest_sha256
        ):
            raise TopologyExperimentError(
                f"{zone}: manifest source live modifie pendant la validation."
            )


def apply_daily_topology(
    config: ExperimentConfig,
    *,
    operational_dir: str | Path,
    operational_manifest_sha256: str,
    delivery_day: str | date,
    zones: Sequence[str],
    mode: str = "production",
    output_root: str | Path = DEFAULT_DAILY_OUTPUT_ROOT,
) -> Path:
    """Apply the fixed published policy to one exact, audited delivery day."""

    selected_zones = _normalise_daily_zones(zones)
    selected_mode = normalise_daily_mode(mode)
    # Validate the complete request before opening a bundle or live archive.
    variants_by_zone = {
        zone: _daily_variants_for_zone(selected_mode, zone)
        for zone in selected_zones
    }
    needs_be_topology = (
        "BE" in variants_by_zone
        and "autonomous" in variants_by_zone["BE"]
    )
    archive_zones = set(selected_zones)
    if needs_be_topology:
        archive_zones.update(SUPPORTED_ZONES)
    ordered_archive_zones = tuple(
        zone for zone in SUPPORTED_ZONES if zone in archive_zones
    )
    day = _normalise_delivery_day(delivery_day)
    expected_bundle_sha = str(operational_manifest_sha256).lower()
    if len(expected_bundle_sha) != 64 or any(
        character not in "0123456789abcdef" for character in expected_bundle_sha
    ):
        raise TopologyExperimentError(
            "operational_manifest_sha256 doit etre un SHA-256 explicite."
        )
    daily_root = Path(output_root).expanduser()
    if not daily_root.is_absolute():
        daily_root = config.contract.project_root / daily_root
    destination = (daily_root / day / selected_mode).resolve()
    _assert_experiment_output(destination, project_root=config.contract.project_root)
    if destination.exists():
        validated = _validate_existing_daily_output(
            destination,
            delivery_day=day,
            mode=selected_mode,
            zones=selected_zones,
            bundle_sha256=expected_bundle_sha,
        )
        # Metadata-only validation preserves idempotence without deserializing
        # or invoking the BE predictor.
        load_operational_bundle(
            config,
            operational_dir,
            expected_manifest_sha256=expected_bundle_sha,
            load_model=False,
        )
        existing_archives = _load_exact_daily_archives(
            project_root=config.contract.project_root,
            zones=ordered_archive_zones,
            delivery_day=day,
        )
        _validate_existing_daily_source_fingerprints(
            destination,
            archives=existing_archives,
        )
        return validated
    bundle = load_operational_bundle(
        config,
        operational_dir,
        expected_manifest_sha256=expected_bundle_sha,
        load_model=needs_be_topology,
    )
    archives = _load_exact_daily_archives(
        project_root=config.contract.project_root,
        zones=ordered_archive_zones,
        delivery_day=day,
    )

    autonomous_inputs: dict[
        str, tuple[pd.DataFrame, pd.DatetimeIndex, pd.DataFrame]
    ] = {}
    blend_inputs: dict[str, tuple[pd.DataFrame, pd.DatetimeIndex]] = {}
    for zone, variants in variants_by_zone.items():
        if "autonomous" in variants:
            base, origins = _read_daily_archive_quantiles(
                archives[zone], variant="autonomous"
            )
            candidate = (
                base.copy()
                if zone == "BE"
                else _apply_daily_autonomous_policy(zone=zone, base=base)
            )
            autonomous_inputs[zone] = (base, origins, candidate)
        if "mkonline_blend" in variants:
            blend_inputs[zone] = _read_daily_archive_quantiles(
                archives[zone], variant="mkonline_blend"
            )

    be_input_hashes: dict[str, str] = {}
    if needs_be_topology:
        if bundle.model is None:
            raise TopologyExperimentError("Modele BE non charge pour la topologie BE.")
        be_base, be_origins, _identity = autonomous_inputs["BE"]
        model_block = bundle.manifest.get("model")
        if not isinstance(model_block, Mapping):
            raise TopologyExperimentError("Bloc model du bundle absent.")
        be_context, be_input_hashes = _load_operational_be_context(
            contract=config.contract,
            archives=archives,
            forecast_index=be_base.index,
            radius=int(model_block["selected_radius"]),
        )
        be_candidate = _apply_daily_autonomous_policy(
            zone="BE",
            base=be_base,
            model=bundle.model,
            context=be_context,
        )
        if bundle.model.hyperparameter_sha256() != model_block.get(
            "hyperparameter_sha256"
        ):
            raise TopologyExperimentError("Modele BE modifie pendant prediction.")
        autonomous_inputs["BE"] = (be_base, be_origins, be_candidate)

    archive_fingerprints = {
        zone: {
            "directory": str(archive.directory),
            "artifact_manifest_sha256": archive.artifact_manifest_sha256,
            "forecast_sha256": _sha256(archive.forecast_path),
        }
        for zone, archive in archives.items()
    }
    consumed_archive_members: dict[Path, str] = {
        archive.forecast_path.resolve(): str(
            archive_fingerprints[zone]["forecast_sha256"]
        )
        for zone, archive in archives.items()
    }
    if needs_be_topology:
        consumed_archive_members[
            (
                archives["BE"].directory
                / "inputs"
                / "model_covariates_with_future.csv.gz"
            ).resolve()
        ] = be_input_hashes["be_future_covariates_sha256"]
        for dependency_zone in SUPPORTED_ZONES:
            consumed_archive_members[
                (
                    archives[dependency_zone].directory
                    / "inputs"
                    / "aligned_inputs.csv.gz"
                ).resolve()
            ] = be_input_hashes[
                f"{dependency_zone.lower()}_aligned_inputs_sha256"
            ]
    for zone, variants in variants_by_zone.items():
        if "autonomous" not in variants:
            continue
        for name in REPORTING_INPUT_FILENAMES:
            reporting_input = archives[zone].directory / "inputs" / name
            consumed_archive_members[reporting_input.resolve()] = (
                _verified_archive_member(archives[zone], reporting_input)
            )
    daily_model_block = bundle.manifest.get("model")
    if not isinstance(daily_model_block, Mapping):
        raise TopologyExperimentError("Bloc model du bundle absent du manifeste daily.")
    daily_model_sha = daily_model_block.get("sha256")
    daily_hyperparameter_sha = daily_model_block.get("hyperparameter_sha256")
    if not isinstance(daily_model_sha, str) or not isinstance(
        daily_hyperparameter_sha, str
    ):
        raise TopologyExperimentError("Empreintes du modele BE absentes du bundle.")
    common_application: dict[str, object] = {
        "schema_version": 1,
        "application_type": "daily_fixed_pricefm_topology",
        "delivery_day": day,
        "mode": selected_mode,
        "operational_bundle": str(bundle.directory),
        "operational_bundle_artifact_manifest_sha256": (
            bundle.artifact_manifest_sha256
        ),
        "calibration_artifact_manifest_sha256": bundle.manifest[
            "calibration_artifact_manifest_sha256"
        ],
        "be_model_sha256": daily_model_sha,
        "be_model_hyperparameter_sha256": daily_hyperparameter_sha,
        "be_model_loaded": needs_be_topology,
        "be_model_predicted": needs_be_topology,
        "fixed_calibration": True,
        "rolling365_enabled": False,
        "selection_recomputed": False,
        "gates_recomputed": False,
        "weights_recomputed": False,
        "model_refitted": False,
        "production_changed": False,
        "runs_live_written": False,
        "source_archives": archive_fingerprints,
        "be_context_inputs": be_input_hashes,
    }

    def writer(staging: Path) -> None:
        report_links: list[tuple[str, str, Path, str]] = []
        outputs: list[dict[str, object]] = []
        for zone in selected_zones:
            for variant in variants_by_zone[zone]:
                if variant == "autonomous":
                    base, origins, candidate = autonomous_inputs[zone]
                    forecast = _daily_forecast_frame(
                        index=base.index,
                        origins=origins,
                        baseline=base,
                        candidate=candidate,
                        variant=variant,
                    )
                    evaluation = json.loads(
                        json.dumps(bundle.evaluations[zone], ensure_ascii=False)
                    )
                    evaluation["daily_application"] = {
                        **common_application,
                        "zone": zone,
                        "variant": variant,
                        "policy": (
                            "topology_promoted"
                            if zone == "BE"
                            else "fallback_identity"
                        ),
                    }
                    variants = evaluation.get("variants")
                    autonomous_section = (
                        variants.get("autonomous")
                        if isinstance(variants, Mapping)
                        else None
                    )
                    if not isinstance(autonomous_section, dict):
                        raise TopologyExperimentError(
                            f"{zone}: sidecar autonome bundle invalide."
                        )
                    autonomous_section["daily_current_policy"] = (
                        "topology_promoted" if zone == "BE" else "fallback_identity"
                    )
                    opened = autonomous_section.get("opened_stages")
                    if not isinstance(opened, list):
                        raise TopologyExperimentError(f"{zone}: opened_stages absent.")
                    display_stage = _display_stage_from_opened(opened)
                    seal = bundle.candidate_seals[zone]
                    candidate_frame = pd.read_csv(seal.prediction_path)
                    opened_index = pd.DatetimeIndex(
                        pd.to_datetime(
                            candidate_frame["delivery_start_utc"],
                            utc=True,
                            errors="raise",
                        ),
                        name="delivery_start_utc",
                    )
                    storm, storm_audit = _load_storm_after_candidate(
                        seal=seal,
                        zone=zone,
                        archive=archives[zone].directory,
                        expected_index=opened_index,
                    )
                    if zone != "ES" and storm is None:
                        raise TopologyExperimentError(
                            f"{zone}: Storm officiel requis pour le rapport quotidien."
                        )
                    for path_key, sha_key in (
                        ("source_path", "source_sha256"),
                        ("source_audit_path", "source_audit_sha256"),
                    ):
                        source_path = storm_audit.get(path_key)
                        source_sha = storm_audit.get(sha_key)
                        if source_path is None and source_sha is None:
                            continue
                        if not isinstance(source_path, str) or not isinstance(
                            source_sha, str
                        ):
                            raise TopologyExperimentError(
                                f"{zone}: preuve Storm incomplete ({path_key})."
                            )
                        resolved_source = Path(source_path).resolve()
                        if _verified_archive_member(
                            archives[zone], resolved_source
                        ) != source_sha:
                            raise TopologyExperimentError(
                                f"{zone}: preuve Storm non scellee ({path_key})."
                            )
                        consumed_archive_members[resolved_source] = source_sha
                    if storm is not None:
                        storm_audit = {
                            **storm_audit,
                            "metrics": {
                                "autonomous": _storm_evaluation_metrics(
                                    candidate_frame,
                                    storm,
                                    stage=display_stage,
                                    candidate_model="topology_autonomous",
                                    baseline_model="residual_corrected",
                                    timezone_name=config.contract.sources[zone].timezone,
                                )
                            },
                            "metrics_computed_after_candidate_freeze": True,
                            "metrics_used_for_gate": False,
                            "metrics_used_for_promotion": False,
                        }
                    evaluation["storm_evaluation"] = storm_audit
                    run_dir = staging / zone.lower() / "autonomous"
                    artifact = _write_variant_run(
                        directory=run_dir,
                        zone=zone,
                        variant="autonomous",
                        candidate_frame=candidate_frame,
                        current_forecast=forecast,
                        evaluation=evaluation,
                        source_directory=config.contract.sources[zone].run_directory,
                        reporting_inputs_directory=archives[zone].directory,
                        published_directory=(
                            destination / zone.lower() / "autonomous"
                        ),
                        display_stage=display_stage,
                        storm=storm,
                        project_root=config.contract.project_root,
                        run_type="daily_fixed_pricefm_topology_application",
                        manifest_extra={
                            "delivery_day": day,
                            "daily_mode": selected_mode,
                            "fixed_calibration": True,
                            "rolling365_enabled": False,
                            "model_refitted": False,
                        },
                    )
                    policy = (
                        "topologie BE promue"
                        if zone == "BE"
                        else "fallback identite"
                    )
                    report_links.append((zone, variant, artifact.path, policy))
                    outputs.append(
                        {
                            "zone": zone,
                            "variant": variant,
                            "policy": policy,
                            "forecast_path": (
                                Path(zone.lower())
                                / "autonomous"
                                / f"forecast_hourly_{zone.lower()}.csv"
                            ).as_posix(),
                            "report_path": artifact.path.relative_to(staging).as_posix(),
                        }
                    )
                else:
                    blend, origins = blend_inputs[zone]
                    # Exact pass-through: no blend helper, weight grid or tuning.
                    forecast = _daily_forecast_frame(
                        index=blend.index,
                        origins=origins,
                        baseline=blend,
                        candidate=blend.copy(),
                        variant=variant,
                    )
                    run_dir = staging / zone.lower() / "mkonline_blend"
                    source_reports = sorted(archives[zone].directory.glob("*.html"))
                    if len(source_reports) != 1:
                        raise TopologyExperimentError(
                            f"{zone}: rapport live unique introuvable "
                            f"({len(source_reports)} trouves)."
                        )
                    source_report_sha = _verified_archive_member(
                        archives[zone], source_reports[0]
                    )
                    consumed_archive_members[
                        source_reports[0].resolve()
                    ] = source_report_sha
                    report_path = _write_daily_passthrough_blend(
                        directory=run_dir,
                        published_directory=(
                            destination / zone.lower() / "mkonline_blend"
                        ),
                        zone=zone,
                        forecast=forecast,
                        archive=archives[zone],
                        application=common_application,
                    )
                    policy = "blend MKOnline production inchange"
                    report_links.append((zone, variant, report_path, policy))
                    outputs.append(
                        {
                            "zone": zone,
                            "variant": variant,
                            "policy": policy,
                            "forecast_path": (
                                Path(zone.lower())
                                / "mkonline_blend"
                                / f"forecast_hourly_{zone.lower()}.csv"
                            ).as_posix(),
                            "report_path": report_path.relative_to(staging).as_posix(),
                        }
                    )
        _write_daily_index(
            staging / "index.html",
            delivery_day=day,
            mode=selected_mode,
            reports=report_links,
        )
        _write_json(
            staging / "daily_manifest.json",
            {
                **common_application,
                "zones": list(selected_zones),
                "variants_by_zone": {
                    zone: list(variants) for zone, variants in variants_by_zone.items()
                },
                "outputs": outputs,
                "complete": True,
            },
        )
        _revalidate_consumed_archive_members(
            archives,
            consumed_archive_members,
        )
        if _validate_artifact_checksums(bundle.directory) != (
            bundle.artifact_manifest_sha256
        ):
            raise TopologyExperimentError("Bundle modifie pendant l'application.")
        _write_artifact_checksums(staging, declared_directory=destination)

    return atomic_experiment_publish(
        destination,
        project_root=config.contract.project_root,
        writer=writer,
    )


def _evaluation_payload(
    *,
    config: ExperimentConfig,
    zone: str,
    development: DevelopmentProtocolResult,
    autonomous: AutonomousProtocolResult,
    recipe_seal: RecipeSeal,
    candidate_seal: CandidateSeal,
    storm_audit: Mapping[str, object],
    source_audit: Mapping[str, object],
    current_audit: Mapping[str, object],
    blend: BlendProtocolResult | None,
    blend_audit: Mapping[str, object] | None,
) -> dict[str, object]:
    source_pit = source_audit.get("pit_coverage")
    if not isinstance(source_pit, Mapping):
        raise TopologyExperimentError(f"{zone}: audit PIT par alias absent.")
    reported_pit_coverage = {
        "source_aliases": dict(source_pit),
        "selected_pool": _context_pit_coverage(development.training_context),
    }
    autonomous_decision_stage = (
        "A_identity"
        if autonomous.selected_arm == "identity"
        else next(
            (
                stage
                for stage in GATED_STAGES
                if stage in autonomous.gates and not autonomous.gates[stage].passes
            ),
            "final",
        )
    )
    autonomous_section: dict[str, object] = {
        "candidate_model": "topology_autonomous",
        "baseline_model": "residual_corrected",
        "selected_arm": autonomous.selected_arm,
        "selected_radius": autonomous.selected_radius,
        "selected_scale": autonomous.selected_scale,
        "topology_context": {
            "radius": autonomous.selected_radius,
            "neighbours": list(autonomous.neighbours),
        },
        "pit_coverage": reported_pit_coverage,
        "selection_A": {
            "arms": development.a_arm_metrics,
            "selected_arm": development.selected_arm,
            "identity_included": True,
            "zero_scale_equivalent_to_identity": True,
            "scale_grid": list(config.scale_grid),
            "used_development": False,
            "used_formal_targets": False,
        },
        "development": {
            "used_for_formal_gate": False,
            "metrics": _segment_payload(development.development_metrics),
        },
        "metrics": {
            name: _segment_payload(metrics)
            for name, metrics in autonomous.metrics.items()
            if name in GATED_STAGES
        },
        "gates": {
            name: _gate_payload(gate) for name, gate in autonomous.gates.items()
        },
        "opened_stages": list(autonomous.opened_stages),
        "decision_stage": autonomous_decision_stage,
        "promotion_decision": (
            "promoted" if autonomous.promoted else "fallback_identity"
        ),
        "sequential_decision_complete": True,
        "unopened_holdouts_spared": [
            stage for stage in GATED_STAGES if stage not in autonomous.opened_stages
        ],
        "promoted": autonomous.promoted,
        "recipe_seal_sha256": recipe_seal.sha256,
        "candidate_prediction_sha256": candidate_seal.prediction_sha256,
        "current_forecast": dict(current_audit),
    }
    variants: dict[str, object] = {"autonomous": autonomous_section}
    if blend is not None:
        assert blend_audit is not None
        blend_decision_stage = next(
            (
                stage
                for stage in GATED_STAGES
                if stage in blend.gates and not blend.gates[stage].passes
            ),
            "final",
        )
        blend_section: dict[str, object] = {
            "candidate_model": "topology_mkonline_blend",
            "baseline_model": "mkonline_blend",
            "selected_radius": autonomous.selected_radius,
            "selected_scale": autonomous.selected_scale,
            "topology_context": {
                "radius": autonomous.selected_radius,
                "neighbours": list(autonomous.neighbours),
            },
            "pit_coverage": reported_pit_coverage,
            "candidate_weights": {
                "topology_autonomous": blend.selected_weight_autonomous,
                "mkonline_primary": blend.selected_weight_mkonline,
            },
            "previous_production_weights": {
                "autonomous": blend.previous_weight_autonomous,
                "mkonline_primary": blend.previous_weight_mkonline,
            },
            "weight_grid_step": blend.grid_step,
            "weight_fit_method": "constrained_l1_grid",
            "fitted_on": "A",
            "final_used_for_tuning": False,
            "autonomous_gates_passed_before_mkonline_load": True,
            "recipe_frozen_before_final": True,
            "dependency_manifest_sha256": blend_audit[
                "dependency_manifest_sha256"
            ],
            "metrics": {
                name: _segment_payload(metrics)
                for name, metrics in blend.metrics.items()
            },
            "gates": {
                name: _gate_payload(gate) for name, gate in blend.gates.items()
            },
            "opened_stages": list(blend.opened_stages),
            "decision_stage": blend_decision_stage,
            "promotion_decision": (
                "promoted" if blend.promoted else "fallback_production_blend"
            ),
            "sequential_decision_complete": True,
            "unopened_holdouts_spared": [
                stage for stage in GATED_STAGES if stage not in blend.opened_stages
            ],
            "promoted": blend.promoted,
            "recommended_variant": (
                "mkonline_blend"
                if blend.promoted
                else "production_mkonline_blend"
            ),
            "production_weights_unchanged": not blend.promoted,
            "blend_source_audit": dict(blend_audit),
        }
        variants["mkonline_blend"] = blend_section
    return {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "zone": zone,
        "timezone": config.contract.sources[zone].timezone,
        "feature_schema_sha256": config.contract.feature_schema_sha256,
        "model_hyperparameters_sha256": autonomous.hyperparameter_sha256,
        "protocol_revision_reason": PROTOCOL_REVISION_REASON,
        "development_used_for_formal_gate": False,
        "formal_gate_periods_unopened_before_freeze": True,
        "protocol_frozen_at_utc": config.protocol_frozen_at_utc,
        "config_sha256": config.config_sha256,
        "recipe_seal_sha256": recipe_seal.sha256,
        "candidate_prediction_sha256": candidate_seal.prediction_sha256,
        "storm_loaded_after_candidate_freeze": True,
        "storm_used_as_prediction_input": False,
        "mkonline_used_by_topology": False,
        "storm_evaluation": dict(storm_audit),
        "sealed_source_audit": dict(source_audit),
        "production_changed": False,
        "variants": variants,
    }


def _copy_reporting_inputs(source_directory: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in REPORTING_INPUT_FILENAMES:
        source = source_directory / "inputs" / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination / name)


def _write_storm_statistics_view(
    directory: Path,
    selected: pd.DataFrame,
    *,
    variant: str,
    display_stage: str,
    evaluation: Mapping[str, object],
    storm_column: str,
) -> None:
    """Materialize the opened decision stage under reporting's Storm contract."""

    if storm_column not in selected:
        raise TopologyExperimentError("Colonne Storm absente de la vue Statistics.")
    values = pd.to_numeric(selected[storm_column], errors="coerce").to_numpy(
        dtype=float
    )
    missing_mask = ~np.isfinite(values)
    delivery = pd.DatetimeIndex(
        pd.to_datetime(selected["delivery_start_utc"], utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    actual_missing = set(delivery[missing_mask])
    storm_audit = evaluation.get("storm_evaluation")
    if not isinstance(storm_audit, Mapping):
        raise TopologyExperimentError("Audit Storm absent du sidecar.")
    allowed_raw = storm_audit.get("native_allowed_missing_utc", [])
    if not isinstance(allowed_raw, list):
        raise TopologyExperimentError("Allow-list Storm sidecar invalide.")
    allowed_missing = {
        pd.Timestamp(value).tz_convert("UTC")
        if pd.Timestamp(value).tzinfo is not None
        else pd.Timestamp(value, tz="UTC")
        for value in allowed_raw
    }.intersection(set(delivery))
    if actual_missing != allowed_missing:
        raise TopologyExperimentError(
            "Vue Statistics: trous Storm differents de l'allow-list DST native."
        )
    variants = evaluation.get("variants")
    variant_section = variants.get(variant) if isinstance(variants, Mapping) else None
    if not isinstance(variant_section, Mapping):
        raise TopologyExperimentError(f"Sidecar variant absent: {variant}.")
    spared = variant_section.get("unopened_holdouts_spared", [])
    if not isinstance(spared, list):
        spared = []
    spared_text = ", ".join(str(item) for item in spared) or "aucune"
    scope_note = (
        "Vue d'evaluation experimentale du holdout decisionnel ouvert: "
        f"{display_stage}; etapes ulterieures non ouvertes/epargnees: "
        f"{spared_text}. Ce fichier n'est pas l'historique live Statistics. "
        "Storm dashboard natif est joint uniquement apres le gel du candidat "
        "et n'est utilise ni pour selection, ni pour gate, ni pour promotion."
    )
    statistics_path = directory / "statistics_history_hourly.csv.gz"
    _write_deterministic_csv_gzip(statistics_path, selected)
    _write_json(
        directory / "statistics_history_audit.json",
        {
            "schema_version": 1,
            "status": "complete_experimental_decision_view",
            "statistics_scope": "opened_experimental_decision_holdout",
            "variant": variant,
            "decision_stage": display_stage,
            "unopened_holdouts_spared": [str(item) for item in spared],
            "statistics_history_path": statistics_path.name,
            "statistics_history_sha256": _sha256(statistics_path),
            "storm_primary_report_benchmark": storm_column,
            "report_scope_note": scope_note,
            "storm_dashboard": {
                "column": storm_column,
                "benchmark_contract": "native_dashboard_snapshot",
                "expected_hours": int(len(selected)),
                "available_hours": int(np.count_nonzero(~missing_mask)),
                "missing_hours": int(np.count_nonzero(missing_mask)),
                "coverage": float(np.mean(~missing_mask)),
                "used_for_prediction": False,
                "used_for_gate": False,
                "candidate_frozen_before_comparator_attachment": True,
                "dst": {
                    "interpolation": False,
                    "strict_08_fallback": False,
                    "native_allowed_missing_hours": len(allowed_missing),
                    "native_allowed_missing_utc": [
                        timestamp.isoformat()
                        for timestamp in sorted(allowed_missing)
                    ],
                    "native_actual_missing_matches_allowed": True,
                },
            },
        },
    )


def _write_variant_run(
    *,
    directory: Path,
    zone: str,
    variant: str,
    candidate_frame: pd.DataFrame,
    current_forecast: pd.DataFrame,
    evaluation: Mapping[str, object],
    source_directory: Path,
    reporting_inputs_directory: Path,
    published_directory: Path,
    display_stage: str,
    storm: pd.Series | None,
    project_root: Path,
    run_type: str = "offline_pricefm_topology_experiment",
    manifest_extra: Mapping[str, object] | None = None,
) -> TopologyReportArtifact:
    directory.mkdir(parents=True, exist_ok=False)
    selected = candidate_frame.loc[candidate_frame["stage"].eq(display_stage)].copy()
    if selected.empty:
        raise TopologyExperimentError(f"{zone}: aucune ligne pour {display_stage}.")
    selected_index = pd.DatetimeIndex(
        pd.to_datetime(selected["delivery_start_utc"], utc=True),
        name="delivery_start_utc",
    )
    if storm is not None:
        paired = storm.reindex(selected_index)
        storm_column = storm.name or "storm_dashboard_official__q50"
        selected[storm_column] = paired.to_numpy(dtype=float)
        _write_storm_statistics_view(
            directory,
            selected,
            variant=variant,
            display_stage=display_stage,
            evaluation=evaluation,
            storm_column=storm_column,
        )
    selected.to_csv(directory / "backtest_hourly_oof.csv.gz", index=False)
    current_forecast.to_csv(
        directory / f"forecast_hourly_{zone.lower()}.csv",
        index=False,
    )
    local_days = pd.DatetimeIndex(
        pd.to_datetime(selected["delivery_start_utc"], utc=True)
    ).tz_convert(config_timezone := str(evaluation["timezone"]))
    metrics = {
        "metrics": [],
        "training_diagnostics": {
            "metric_scope": f"topology_{display_stage}_paired_display",
            "evaluation_start_local_date": str(local_days.date.min()),
            "evaluation_end_local_date": str(local_days.date.max()),
            "development_used_for_formal_gate": False,
        },
    }
    _write_json(directory / "metrics_hourly.json", metrics)
    pd.DataFrame(metrics["metrics"]).to_csv(directory / "metrics_hourly.csv", index=False)
    _write_json(directory / "topology_evaluation.json", evaluation)
    run_manifest: dict[str, object] = {
            "schema_version": 1,
            "run_type": run_type,
            "zone": zone,
            "timezone": config_timezone,
            "variant": variant,
            "display_stage": display_stage,
            "protocol_revision_reason": PROTOCOL_REVISION_REASON,
            "development_used_for_formal_gate": False,
            "formal_gate_periods_unopened_before_freeze": True,
            "config_sha256": evaluation["config_sha256"],
            "protocol_frozen_at_utc": evaluation["protocol_frozen_at_utc"],
            "production_changed": False,
            "storm_used_as_prediction_input": False,
            "mkonline_used_by_topology": False,
            "source_autonomous_run": str(source_directory),
        }
    if manifest_extra is not None:
        forbidden = sorted(set(run_manifest).intersection(manifest_extra))
        if forbidden:
            raise TopologyExperimentError(
                f"run_manifest extra ne peut remplacer: {forbidden}."
            )
        run_manifest.update(dict(manifest_extra))
    _write_json(directory / "run_manifest.json", run_manifest)
    _copy_reporting_inputs(reporting_inputs_directory, directory / "inputs")
    artifact = write_topology_html_report(
        directory,
        variant=variant,
        project_root=project_root,
    )
    _write_artifact_checksums(
        directory,
        declared_directory=published_directory,
    )
    return artifact


def _current_archives_after_candidate_seals(
    config: ExperimentConfig,
    seals: Mapping[str, CandidateSeal],
):
    for zone in config.zones:
        seal = seals[zone]
        if _sha256(seal.prediction_path) != seal.prediction_sha256:
            raise TopologyExperimentError(f"{zone}: candidat modifie avant archive live.")
    from chronos2_hourly.app_service import (
        inspect_zone_statuses,
        load_latest_forecast_comparison,
    )

    registry = config.contract.project_root / "chronos2_hourly_live_zones.yaml"
    statuses = inspect_zone_statuses(registry, zones=config.zones)
    return load_latest_forecast_comparison(
        statuses,
        project_root=config.contract.project_root,
        zones=config.zones,
        allow_mixed_delivery_days=False,
        variant="autonomous",
    )


def run_experiment(config: ExperimentConfig) -> Path:
    """Execute all zones under one atomic experiment publication."""

    contract = config.contract
    source_audits = audit_experiment_sources(config)

    def writer(staging: Path) -> None:
        splits_by_zone: dict[str, ProtocolSplits] = {}
        developments: dict[str, DevelopmentProtocolResult] = {}
        recipe_seals: dict[str, RecipeSeal] = {}
        formal_results: dict[str, AutonomousProtocolResult] = {}
        formal_blocks: dict[str, dict[str, ZoneWindowData]] = {}
        blend_results: dict[str, BlendProtocolResult | None] = {}
        blend_audits: dict[str, Mapping[str, object] | None] = {}
        production_blends: dict[str, pd.DataFrame | None] = {}

        # Phase 1: every country is selected/developed and recipe-frozen before
        # the first formal B1 outcome of any country is opened.
        for zone in config.zones:
            splits = _splits_for_contract(contract, zone)
            splits_by_zone[zone] = splits
            preformal = splits.indices["seed"].append(splits.indices["a"])
            preformal = preformal.append(splits.indices["development"])
            data = _load_zone_window(contract, zone=zone, expected_index=preformal)
            development = run_development_protocol(
                zone=zone,
                actual=data.actual,
                base_predictions=data.base_predictions,
                contexts=data.contexts,
                splits=splits,
                model_parameters=config.model_parameters,
                scale_grid=config.scale_grid,
                bootstrap_samples=config.bootstrap_samples,
                bootstrap_seed=config.bootstrap_seed,
            )
            developments[zone] = development
            recipe_seals[zone] = freeze_protocol_recipe(
                staging / zone.lower(),
                development,
                config=config,
            )

        # Phase 2: formal blocks are opened strictly one at a time. A failed
        # gate prevents the loader for every later block from being called.
        for zone in config.zones:
            cache: dict[str, ZoneWindowData] = {}
            formal_blocks[zone] = cache
            splits = splits_by_zone[zone]
            development = developments[zone]

            def stage_loader(stage: str, *, _zone: str = zone) -> FormalStageData:
                block = _load_zone_window(
                    contract,
                    zone=_zone,
                    expected_index=splits_by_zone[_zone].indices[stage],
                )
                cache[stage] = block
                return FormalStageData(
                    actual=block.actual,
                    base_predictions=block.base_predictions,
                    context=block.contexts[developments[_zone].selected_radius],
                )

            formal_results[zone] = run_formal_autonomous_protocol(
                development=development,
                recipe_seal=recipe_seals[zone],
                stage_loader=stage_loader,
                splits=splits,
                model_parameters=config.model_parameters,
                gate_minimum_gain=config.gate_minimum_gain,
                bootstrap_samples=config.bootstrap_samples,
                bootstrap_seed=config.bootstrap_seed,
            )

        # Phase 3: MKOnline is touched only for eligible, promoted zones. The
        # production comparator is reconstructed from its sealed weights.
        for zone in config.zones:
            autonomous = formal_results[zone]
            if zone not in BLEND_ZONES or not autonomous.promoted:
                blend_results[zone] = None
                blend_audits[zone] = None
                production_blends[zone] = None
                continue
            source_audit = audit_blend_source(
                contract,
                zone,
                autonomous_promoted=True,
            )
            dependency = contract.blend_dependency_for(zone)
            if _sha256(dependency.forecast_file) != dependency.forecast_file_sha256:
                raise TopologyExperimentError(
                    f"{zone}: le forecast MKOnline a change apres son audit."
                )
            mk_raw = pd.read_parquet(dependency.forecast_file)
            if _sha256(dependency.forecast_file) != dependency.forecast_file_sha256:
                raise TopologyExperimentError(
                    f"{zone}: le forecast MKOnline a change pendant sa lecture."
                )
            mk_index = pd.DatetimeIndex(
                pd.to_datetime(mk_raw["value_time_utc"], utc=True),
                name="delivery_start_utc",
            )
            mk = pd.Series(
                pd.to_numeric(mk_raw["value"], errors="coerce").to_numpy(dtype=float),
                index=mk_index,
                name="mkonline_primary__q50",
            )
            actual, base, _context = _combine_full_training_data(
                developments[zone],
                formal_blocks[zone],
            )
            result = run_blend_protocol(
                zone=zone,
                autonomous=autonomous,
                actual=actual,
                mkonline_q50_loader=lambda _mk=mk: _mk,
                production_autonomous_loader=lambda _base=base: _base,
                splits=splits_by_zone[zone],
                previous_weight_mkonline=dependency.current_weight_mkonline,
                grid_step=config.blend_grid_step,
                gate_minimum_gain=config.gate_minimum_gain,
                bootstrap_samples=config.bootstrap_samples,
                bootstrap_seed=config.bootstrap_seed,
            )
            blend_results[zone] = result
            blend_audits[zone] = source_audit
            production_blends[zone] = _blend_quantiles(
                base,
                mk,
                weight_mkonline=dependency.current_weight_mkonline,
            )

        # Phase 4: freeze every autonomous/blend candidate before any validated
        # current archive (which contains Storm statistics) is opened.
        candidate_frames: dict[str, pd.DataFrame] = {}
        candidate_seals: dict[str, CandidateSeal] = {}
        for zone in config.zones:
            frame = _opened_candidate_frame(
                development=developments[zone],
                autonomous=formal_results[zone],
                splits=splits_by_zone[zone],
                formal_data=formal_blocks[zone],
                blend=blend_results[zone],
                production_blend=production_blends[zone],
            )
            candidate_frames[zone] = frame
            candidate_seals[zone] = seal_candidate_predictions(
                staging / zone.lower(),
                frame,
                zone=zone,
                experiment_id=config.experiment_id,
            )

        comparison = _current_archives_after_candidate_seals(config, candidate_seals)
        archive_by_zone = {item.zone: item for item in comparison.archives}
        report_artifacts: list[TopologyReportArtifact] = []
        for zone in config.zones:
            autonomous = formal_results[zone]
            development = developments[zone]
            archive_record = archive_by_zone[zone]
            current_rows = comparison.frame.loc[comparison.frame["zone"].eq(zone)]
            current_index = pd.DatetimeIndex(
                pd.to_datetime(current_rows["timestamp_utc"], utc=True),
                name="delivery_start_utc",
            )
            current_base = pd.DataFrame(
                {
                    "q10": current_rows["P10"].to_numpy(dtype=float),
                    "q50": current_rows["P50"].to_numpy(dtype=float),
                    "q90": current_rows["P90"].to_numpy(dtype=float),
                },
                index=current_index,
            )
            current_context = (
                _load_current_context(
                    contract=contract,
                    zone=zone,
                    archive=archive_record.archive_path,
                    forecast_index=current_index,
                    radius=autonomous.selected_radius,
                )
                if autonomous.promoted
                else pd.DataFrame(index=current_index)
            )
            current_auto = _predict_current_autonomous(
                config=config,
                development=development,
                autonomous=autonomous,
                formal_data=formal_blocks[zone],
                current_base=current_base,
                current_context=current_context,
            )
            forecast = pd.DataFrame(
                {
                    "delivery_start_utc": current_index,
                    "residual_corrected__q10": current_base["q10"],
                    "residual_corrected__q50": current_base["q50"],
                    "residual_corrected__q90": current_base["q90"],
                    "topology_autonomous__q10": current_auto["q10"],
                    "topology_autonomous__q50": current_auto["q50"],
                    "topology_autonomous__q90": current_auto["q90"],
                }
            )
            blend = blend_results[zone]
            if blend is not None:
                if _sha256(archive_record.forecast_path) != archive_record.forecast_sha256:
                    raise TopologyExperimentError(
                        f"{zone}: l'archive current a change avant lecture MKOnline."
                    )
                raw_current = pd.read_csv(archive_record.forecast_path)
                if _sha256(archive_record.forecast_path) != archive_record.forecast_sha256:
                    raise TopologyExperimentError(
                        f"{zone}: l'archive current a change pendant sa lecture MKOnline."
                    )
                raw_current_index = pd.DatetimeIndex(
                    pd.to_datetime(
                        raw_current["delivery_start_utc"],
                        utc=True,
                        errors="raise",
                    ),
                    name="delivery_start_utc",
                )
                _validate_utc_index(
                    raw_current_index,
                    name=f"{zone}_current_mkonline_index",
                )
                mk_current = pd.Series(
                    pd.to_numeric(
                        raw_current["mkonline_primary__q50"], errors="coerce"
                    ).to_numpy(dtype=float),
                    index=raw_current_index,
                    name="mkonline_primary__q50",
                ).reindex(current_index)
                if not np.isfinite(mk_current.to_numpy(dtype=float)).all():
                    raise TopologyExperimentError(
                        f"{zone}: MKOnline current ne couvre pas l'archive exacte."
                    )
                prod_current = _blend_quantiles(
                    current_base,
                    mk_current,
                    weight_mkonline=blend.previous_weight_mkonline,
                )
                topology_current = (
                    _blend_quantiles(
                        current_auto,
                        mk_current,
                        weight_mkonline=blend.selected_weight_mkonline,
                    )
                    if blend.promoted
                    else prod_current
                )
                for quantile in QUANTILE_COLUMNS:
                    forecast[f"mkonline_blend__{quantile}"] = prod_current[quantile]
                    forecast[f"topology_mkonline_blend__{quantile}"] = (
                        topology_current[quantile]
                    )

            display_stage = _display_stage_from_opened(
                autonomous.opened_stages
            )
            blend_display = (
                _display_stage_from_opened(blend.opened_stages)
                if blend is not None
                else None
            )
            opened_candidate_index = pd.DatetimeIndex(
                pd.to_datetime(
                    candidate_frames[zone]["delivery_start_utc"],
                    utc=True,
                    errors="raise",
                ),
                name="delivery_start_utc",
            )
            storm, storm_audit = _load_storm_after_candidate(
                seal=candidate_seals[zone],
                zone=zone,
                archive=archive_record.archive_path,
                expected_index=opened_candidate_index,
            )
            if zone != "ES" and storm is None:
                raise TopologyExperimentError(
                    f"{zone}: snapshot Storm dashboard natif audite obligatoire."
                )
            if storm is not None:
                storm_metrics: dict[str, object] = {
                    "autonomous": _storm_evaluation_metrics(
                        candidate_frames[zone],
                        storm,
                        stage=display_stage,
                        candidate_model="topology_autonomous",
                        baseline_model="residual_corrected",
                        timezone_name=splits_by_zone[zone].timezone,
                    )
                }
                if blend is not None and blend_display is not None:
                    storm_metrics["mkonline_blend"] = _storm_evaluation_metrics(
                        candidate_frames[zone],
                        storm,
                        stage=blend_display,
                        candidate_model="topology_mkonline_blend",
                        baseline_model="mkonline_blend",
                        timezone_name=splits_by_zone[zone].timezone,
                    )
                storm_audit = {
                    **storm_audit,
                    "metrics": storm_metrics,
                    "metrics_computed_after_candidate_freeze": True,
                    "metrics_used_for_gate": False,
                    "metrics_used_for_promotion": False,
                }
            current_audit = {
                "delivery_day": archive_record.delivery_day,
                "archive_path": str(archive_record.archive_path),
                "forecast_sha256": archive_record.forecast_sha256,
                "checksum_manifest_sha256": (
                    archive_record.checksum_manifest_sha256
                ),
                "fallback_identity": not autonomous.promoted,
            }
            evaluation = _evaluation_payload(
                config=config,
                zone=zone,
                development=development,
                autonomous=autonomous,
                recipe_seal=recipe_seals[zone],
                candidate_seal=candidate_seals[zone],
                storm_audit=storm_audit,
                source_audit=source_audits[zone],
                current_audit=current_audit,
                blend=blend,
                blend_audit=blend_audits[zone],
            )
            auto_dir = staging / zone.lower() / "autonomous"
            report_artifacts.append(
                _write_variant_run(
                    directory=auto_dir,
                    zone=zone,
                    variant="autonomous",
                    candidate_frame=candidate_frames[zone],
                    current_forecast=forecast,
                    evaluation=evaluation,
                    source_directory=contract.sources[zone].run_directory,
                    reporting_inputs_directory=archive_record.archive_path,
                    published_directory=(
                        config.output_directory / zone.lower() / "autonomous"
                    ),
                    display_stage=display_stage,
                    storm=storm,
                    project_root=contract.project_root,
                )
            )
            if blend is not None:
                assert blend_display is not None
                report_artifacts.append(
                    _write_variant_run(
                        directory=staging / zone.lower() / "mkonline_blend",
                        zone=zone,
                        variant="mkonline_blend",
                        candidate_frame=candidate_frames[zone],
                        current_forecast=forecast,
                        evaluation=evaluation,
                        source_directory=contract.sources[zone].run_directory,
                        reporting_inputs_directory=archive_record.archive_path,
                        published_directory=(
                            config.output_directory
                            / zone.lower()
                            / "mkonline_blend"
                        ),
                        display_stage=blend_display,
                        storm=storm,
                        project_root=contract.project_root,
                    )
                )

        write_topology_report_index(
            report_artifacts,
            output_path=staging / "index.html",
            selected_zones=config.zones,
            project_root=contract.project_root,
        )
        _write_json(
            staging / "experiment_manifest.json",
            {
                "schema_version": 1,
                "experiment_id": config.experiment_id,
                "zones": list(config.zones),
                "config_path": str(config.source_path),
                "config_sha256": config.config_sha256,
                "protocol_frozen_at_utc": config.protocol_frozen_at_utc,
                "protocol_revision_reason": PROTOCOL_REVISION_REASON,
                "development_used_for_formal_gate": False,
                "formal_gate_periods_unopened_before_freeze": True,
                "sealed_source_audits": source_audits,
                "production_changed": False,
            },
        )
        _write_artifact_checksums(
            staging,
            declared_directory=config.output_directory,
        )

    return atomic_experiment_publish(
        config.output_directory,
        project_root=contract.project_root,
        writer=writer,
    )


# The remaining orchestration stays below the pure protocol helpers.  This
# boundary lets tests prove the freeze/load sequence without real archives.


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lance l'experience causale PriceFM-inspired sous runs/experiments."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--zones",
        nargs="+",
        default=None,
        help=(
            "Pays a publier. Defaut: les cinq pays, sauf --mode blend qui "
            "selectionne automatiquement FR NL."
        ),
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--audit-only",
        action="store_true",
        help="Valide les sources et le plan sans entrainer ni publier.",
    )
    action.add_argument(
        "--prepare-operational",
        action="store_true",
        help="Scelle le modele BE fixe depuis la calibration publiee.",
    )
    action.add_argument(
        "--apply-daily",
        action="store_true",
        help="Applique le bundle scelle a un jour live exact.",
    )
    action.add_argument(
        "--report-365",
        action="store_true",
        help=(
            "Publie les strategies annuelles gouvernee/shadow depuis les "
            "predictions deja scellees, sans fit ni predict."
        ),
    )
    action.add_argument(
        "--rolling365-backtest",
        action="store_true",
        help=(
            "Reentraine chaque jour le correcteur topologique sur les 365 "
            "jours locaux precedents, puis publie Statistics et HTML."
        ),
    )
    parser.add_argument("--calibration-dir", default=None)
    parser.add_argument(
        "--calibration-manifest-sha256",
        default=CALIBRATION_V1_ARTIFACT_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--operational-dir",
        default=str(DEFAULT_OPERATIONAL_DIRECTORY),
    )
    parser.add_argument("--operational-manifest-sha256", default=None)
    parser.add_argument("--delivery-day", default=None)
    parser.add_argument("--mode", default="production")
    parser.add_argument(
        "--daily-output-root",
        default=str(DEFAULT_DAILY_OUTPUT_ROOT),
    )
    parser.add_argument(
        "--annual-output-dir",
        default=str(DEFAULT_ANNUAL_OUTPUT_DIRECTORY),
    )
    parser.add_argument(
        "--rolling365-output-dir",
        default=str(DEFAULT_ROLLING365_OUTPUT_DIRECTORY),
    )
    parser.add_argument(
        "--rolling365-workers",
        type=int,
        default=5,
        help="Nombre maximal de pays entraines en parallele (defaut: 5).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cli_mode = normalise_daily_mode(args.mode)
        cli_zones = (
            tuple(zone for zone in SUPPORTED_ZONES if zone in BLEND_ZONES)
            if args.apply_daily and args.zones is None and cli_mode == "blend"
            else (
                tuple(SUPPORTED_ZONES)
                if args.zones is None
                else tuple(args.zones)
            )
        )
        if args.apply_daily and args.delivery_day in (None, ""):
            raise TopologyExperimentError(
                "--apply-daily exige --delivery-day YYYY-MM-DD."
            )
        if args.apply_daily and not args.operational_manifest_sha256:
            raise TopologyExperimentError(
                "--apply-daily exige --operational-manifest-sha256."
            )
        if args.apply_daily:
            daily_day = _normalise_delivery_day(args.delivery_day)
            expected_bundle_sha = str(args.operational_manifest_sha256).lower()
            if len(expected_bundle_sha) != 64 or any(
                character not in "0123456789abcdef"
                for character in expected_bundle_sha
            ):
                raise TopologyExperimentError(
                    "--operational-manifest-sha256 doit etre un SHA-256."
                )
            selected_cli_zones = _normalise_daily_zones(cli_zones)
            for zone in selected_cli_zones:
                _daily_variants_for_zone(cli_mode, zone)
        else:
            daily_day = None
        if not args.apply_daily and args.delivery_day not in (None, ""):
            raise TopologyExperimentError(
                "--delivery-day est reserve a --apply-daily."
            )
        if not args.apply_daily and cli_mode != "production":
            raise TopologyExperimentError("--mode est reserve a --apply-daily.")
        operational_action = bool(args.prepare_operational or args.apply_daily)
        existing_output_action = bool(
            args.audit_only
            or operational_action
            or args.report_365
            or args.rolling365_backtest
        )
        config = load_experiment_config(
            args.config,
            zones=(SUPPORTED_ZONES if operational_action else cli_zones),
            allow_existing_output=existing_output_action,
            verify_hashes=not args.apply_daily,
        )
        if args.prepare_operational:
            calibration_dir = args.calibration_dir or config.output_directory
            output = prepare_operational_bundle(
                config,
                calibration_dir=calibration_dir,
                output_dir=args.operational_dir,
                expected_calibration_manifest_sha256=(
                    args.calibration_manifest_sha256
                ),
                expected_existing_bundle_sha256=(
                    args.operational_manifest_sha256
                ),
            )
            manifest_sha = _sha256(output / "artifact_checksums.json")
            print(f"Bundle operationnel publie: {output}")
            print(f"operational_manifest_sha256={manifest_sha}")
            return 0
        if args.report_365:
            calibration_dir = args.calibration_dir or config.output_directory
            output = report_topology_annual_365(
                config,
                calibration_dir=calibration_dir,
                output_dir=args.annual_output_dir,
                expected_calibration_manifest_sha256=(
                    args.calibration_manifest_sha256
                ),
            )
            manifest_sha = _sha256(output / "artifact_checksums.json")
            print(f"Reporting annuel 365 publie: {output}")
            print(f"annual_manifest_sha256={manifest_sha}")
            return 0
        if args.rolling365_backtest:
            calibration_dir = args.calibration_dir or config.output_directory
            output = run_topology_rolling365(
                config,
                calibration_dir=calibration_dir,
                output_dir=args.rolling365_output_dir,
                expected_calibration_manifest_sha256=(
                    args.calibration_manifest_sha256
                ),
                workers=args.rolling365_workers,
            )
            manifest_sha = _sha256(output / "artifact_checksums.json")
            print(f"Backtest rolling365 publie: {output}")
            print(f"rolling365_manifest_sha256={manifest_sha}")
            return 0
        if args.apply_daily:
            output = apply_daily_topology(
                config,
                operational_dir=args.operational_dir,
                operational_manifest_sha256=args.operational_manifest_sha256,
                delivery_day=daily_day,
                zones=selected_cli_zones,
                mode=cli_mode,
                output_root=args.daily_output_root,
            )
            print(f"Forecasts topologiques quotidiens publies: {output}")
            return 0

        # Imported lazily so contract-only audits never instantiate a model.
        plan = audit_experiment_sources(config)
        for zone, summary in plan.items():
            pit = summary["pit_coverage"]
            assert isinstance(pit, Mapping)
            pit_minimum = min(float(value) for value in pit.values())
            print(
                f"[{zone}] sources scellees OK | "
                f"cutoff_violations={summary['forecast_origin_violations']} | "
                f"pit_min={pit_minimum:.3%} | "
                f"lag_masked_dst={summary['price_lag_masked_same_day_hours']} | "
                "formal_outcomes=unopened"
            )
        if args.audit_only:
            print("Audit termine; aucun fichier d'experience n'a ete cree.")
            return 0
        run_experiment(config)
        print(f"Rapports publies dans {config.output_directory}")
        return 0
    except Exception as exc:
        print(f"Erreur experience topologique: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
