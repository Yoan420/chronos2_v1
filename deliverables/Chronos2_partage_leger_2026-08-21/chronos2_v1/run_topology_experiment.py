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
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Final, Protocol
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.models.topology_residual_corrector import (
    QUANTILE_COLUMNS,
    TopologyResidualCorrector,
)
from chronos2_hourly.topology_context import (
    TOPOLOGY_CONTEXT_COLUMNS,
    ZONE_TIMEZONES,
    build_topology_context,
)
from chronos2_hourly.topology_report import (
    BLEND_ZONES,
    TopologyReportArtifact,
    write_topology_html_report,
    write_topology_report_index,
)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DEFAULT_CONFIG: Final[Path] = (
    PROJECT_ROOT / "config" / "pricefm_topology_experiment.yaml"
)
SUPPORTED_ZONES: Final[tuple[str, ...]] = ("FR", "DE", "BE", "NL", "ES")
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
        a_audits[f"radius{radius}_seed_to_a"] = model.audit_metadata()
        for scale in scales[1:]:
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
        "model_parameters": dict(config.model_parameters),
        "scale_grid": list(config.scale_grid),
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
    for offset, stage in enumerate(GATED_STAGES, start=1):
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
            bootstrap_seed=bootstrap_seed + offset,
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
    current_blend_predictions_loader: Callable[[], pd.DataFrame],
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
    current_blend = _validate_quantiles(
        current_blend_predictions_loader(),
        expected_index=splits.all_index,
        name="current_production_blend",
    )
    if not mkonline.index.equals(splits.all_index):
        raise TopologyExperimentError("MKOnline ne couvre pas les 365 jours exacts.")
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
    for offset, stage in enumerate(GATED_STAGES, start=1):
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
            bootstrap_seed=bootstrap_seed + 10 + offset,
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
    old_weight = float(previous_weight_mkonline)
    if not math.isfinite(old_weight) or not 0.0 <= old_weight <= 1.0:
        raise TopologyExperimentError("Poids MKOnline de production invalide.")
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
        os.replace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination


# The source-contract adapter and full filesystem orchestration are defined
# below the pure protocol helpers.  Keeping this boundary explicit makes the
# tests able to prove causal sequencing without opening any real archive.


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lance l'experience causale PriceFM-inspired sous runs/experiments."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--zones", nargs="+", default=list(SUPPORTED_ZONES))
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Valide les sources et le plan sans entrainer ni publier.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        # Imported lazily so contract-only audits never instantiate a model.
        config = load_experiment_config(args.config, zones=args.zones)
        plan = audit_experiment_sources(config)
        for zone, summary in plan.items():
            print(f"[{zone}] sources scellees OK | {summary}")
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
