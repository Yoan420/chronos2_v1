"""Inference on the original rank-16 training/validation prefix only.

Original training covariates can have missing *context* values: build_fit_inputs
preserves them, and Chronos2Model._prepare_patched_context builds a native mask
with torch.isnan(context).logical_not(). This narrowly scoped reconstruction
does the same. It is not the production/shadow inference contract, a neural
OOF reconstruction, or evidence of historical publication availability.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .evaluation import _input_sha256, _predict_quantiles
from .lora_finetune import ExogenousFineTuneConfig
from .prospective_auxiliary import MARKET_COLUMNS, RAW_MODEL, QUANTILES


PREFIX_FIRST_DAY = date(2024, 9, 3)
PREFIX_LAST_DAY = date(2025, 9, 2)


class RollingResearchPrefixError(ValueError):
    """An input is outside the frozen research-prefix contract."""


def _timestamps(values: pd.Series, *, name: str) -> pd.DatetimeIndex:
    result = pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="raise"))
    if result.hasnans:
        raise RollingResearchPrefixError(f"Préfixe : timestamp manquant dans {name}.")
    return result


def predict_prefix_group(
    group: pd.DataFrame, candidate: ExogenousFineTuneConfig, pipeline: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Predict one original training-prefix origin, without reading future labels.

    Target arrays contain exactly the 2048 context hours. Only covariates are
    selected from the physical 23/24/25-hour delivery suffix; its target values
    are neither converted, tested, hashed, scored nor passed to the pipeline.
    Context NaNs remain native NaNs, while context infinities, missing targets
    and missing future covariates are rejected before any model call.
    """
    if (candidate.context_length != 2048 or candidate.prediction_length != 24
            or candidate.frequency != "h" or candidate.cutoff_local_time != "08:00"
            or len(candidate.target_columns) != 1
            or candidate.lora_config.get("r") != 16):
        raise RollingResearchPrefixError("Préfixe réservé au contrat original LoRA rang 16, contexte2048/J−1 08:00.")
    if not set(MARKET_COLUMNS).issubset(candidate.known_future_covariates):
        raise RollingResearchPrefixError("Les cinq residual_load futurs du checkpoint original sont requis.")
    covariates = tuple(candidate.covariate_columns)
    if len(set(covariates)) != len(covariates):
        raise RollingResearchPrefixError("Le contrat des covariables contient des doublons.")
    required = {candidate.timestamp_column, candidate.origin_column, candidate.item_column,
        candidate.feature_available_at_column, *candidate.target_columns, *covariates}
    missing = sorted(required.difference(group.columns))
    if missing or group.empty:
        raise RollingResearchPrefixError(f"Groupe du préfixe vide ou incomplet : {missing}.")
    timestamps = _timestamps(group[candidate.timestamp_column], name=candidate.timestamp_column)
    origins = _timestamps(group[candidate.origin_column], name=candidate.origin_column)
    available = _timestamps(group[candidate.feature_available_at_column], name=candidate.feature_available_at_column)
    if timestamps.has_duplicates or len(origins.unique()) != 1:
        raise RollingResearchPrefixError("Le préfixe exige une origine unique et des timestamps sans doublon.")
    if group[candidate.item_column].isna().any():
        raise RollingResearchPrefixError("Zone du préfixe manquante.")
    items = group[candidate.item_column].astype(str).unique()
    if len(items) != 1 or items[0] not in {"FR", "DE", "BE", "NL"}:
        raise RollingResearchPrefixError("Une seule zone originale FR/DE/BE/NL est autorisée par groupe.")
    origin = origins[0]
    local_origin = origin.tz_convert(candidate.timezone)
    if (local_origin.hour, local_origin.minute, local_origin.second, local_origin.microsecond,
            local_origin.nanosecond) != (8, 0, 0, 0, 0):
        raise RollingResearchPrefixError("L’origine historique doit être exactement à08:00 locale.")
    delivery_day = local_origin.date() + timedelta(days=1)
    if not PREFIX_FIRST_DAY <= delivery_day <= PREFIX_LAST_DAY:
        raise RollingResearchPrefixError(
            f"Jour {delivery_day} hors du préfixe original {PREFIX_FIRST_DAY} → {PREFIX_LAST_DAY}."
        )
    if bool((available > origin).any()):
        raise RollingResearchPrefixError("Une feature est déclarée disponible après l’origine historique.")
    local_start = pd.Timestamp(delivery_day, tz=candidate.timezone)
    local_end = pd.Timestamp(delivery_day + timedelta(days=1), tz=candidate.timezone)
    horizon = pd.date_range(local_start, local_end, freq="h", inclusive="left").tz_convert("UTC")
    context = pd.date_range(end=horizon[0] - pd.Timedelta(hours=1), periods=2048, freq="h")
    expected = context.append(horizon)
    order = np.argsort(timestamps.asi8, kind="stable")
    if not timestamps.take(order).equals(expected):
        raise RollingResearchPrefixError("Le groupe doit avoir2048 heures régulières puis le suffixe physique23/24/25h exact.")
    ordered = group.iloc[order].reset_index(drop=True)
    if "phase" in ordered:
        expected_phase = np.asarray(["context"] * 2048 + ["horizon"] * len(horizon))
        if not np.array_equal(ordered["phase"].astype(str).to_numpy(), expected_phase):
            raise RollingResearchPrefixError("Les phases contexte/horizon divergent de la timeline physique.")
    if "delivery_day" in ordered:
        if not ordered["delivery_day"].astype(str).eq(str(delivery_day)).all():
            raise RollingResearchPrefixError("delivery_day divergent de l’origine historique.")
    # Never select the horizon target: even non-numeric sentinel values there
    # are irrelevant to this computation and must not enter its identity.
    context_target = ordered.iloc[:2048].loc[:, list(candidate.target_columns)].to_numpy(dtype=np.float32).T
    if not np.isfinite(context_target).all():
        raise RollingResearchPrefixError("Cible du contexte non finie : aucune imputation autorisée.")
    past: dict[str, np.ndarray] = {}
    future: dict[str, np.ndarray] = {}
    missing_by_column: dict[str, int] = {}
    for column in covariates:
        values = ordered.iloc[:2048][column].to_numpy(dtype=np.float32)
        if np.isinf(values).any():
            raise RollingResearchPrefixError(f"Covariable de contexte {column!r} infinie.")
        past[column] = values.copy()
        missing_by_column[column] = int(np.isnan(values).sum())
    for column in candidate.known_future_covariates:
        values = ordered.iloc[2048:][column].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise RollingResearchPrefixError(f"Covariable future {column!r} non finie.")
        future[column] = values.copy()
    payload = {"target": context_target, "past_covariates": past, "future_covariates": future}
    input_digest = _input_sha256(payload)
    quantiles = _predict_quantiles(pipeline, payload, prediction_length=len(horizon),
        context_length=2048, batch_size=64)
    if quantiles.shape != (1, len(horizon), 3):
        raise RollingResearchPrefixError("Le checkpoint doit produire une seule cible de prix.")
    raw = pd.DataFrame({"delivery_start_utc": horizon, "forecast_origin_utc": origin,
        "actual": np.nan, "input_sha256": input_digest})
    for quantile_index, quantile in enumerate(QUANTILES):
        raw[f"{RAW_MODEL}__{quantile}"] = quantiles[0, :, quantile_index]
    for column in MARKET_COLUMNS:
        # Keep the original precision for the downstream auxiliary filters,
        # just like prospective_trial._predict_group; neural tensors are float32.
        raw[column] = ordered.iloc[2048:][column].to_numpy(dtype=float)
    audit = {
        "kind": "original_rank16_training_prefix_native_nan_mask_v1",
        "zone": str(items[0]), "delivery_day": str(delivery_day),
        "forecast_origin_utc": origin.isoformat(), "timezone": candidate.timezone,
        "scope_first_day": str(PREFIX_FIRST_DAY), "scope_last_day": str(PREFIX_LAST_DAY),
        "context_hours": 2048, "horizon_hours": len(horizon),
        "feature_available_at_max_utc": available.max().isoformat(),
        "context_missing_by_column": missing_by_column,
        "context_missing_values": int(sum(missing_by_column.values())),
        "native_nan_mask": True, "context_imputation": False,
        "future_covariates_finite": True, "horizon_observations_read": False,
        "horizon_observations_used": 0, "input_sha256": input_digest,
        "batch_size": 64, "cross_learning": False,
        "diagnostic_only": True, "neural_in_sample": True, "neural_oof": False,
        "production_pit_evidence": False, "production_pipeline_evidence": False,
        "promotion_eligible": False,
    }
    return raw, audit
