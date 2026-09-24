"""Three isolated, preregistered ablations of the frozen scarcity-regime v1.

No disk, network, source synchronization, production integration or persistence
is performed here. The caller pins the v1 artifacts/code and supplies strictly
earlier labels. All outputs are residual quantiles, each added to the SAME
frozen interaction-40 P50; none is an arithmetic mean relabelled as a median.

1. direct_quantile: one unweighted MultiQuantile residual model, all features.
2. regime_hour_local: exactly v1 after deleting its four daily-peak features.
3. regime_calibration_oof90: identical v1 conditional experts, recalibration of
   SAVED PREQUENTIAL v1 probabilities (already calibrated by v1, NOT raw logits).
   A shared logit slope and strongly regularized country offset use only a
   complete preceding 90-day DE/NL panel. Reproduction of the original mixture
   quantiles is mandatory before the new probabilities are substituted.

This remains retrospective, with uncertified source-publication PIT. The
22 September event informed the hypothesis and is not an independent holdout.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Sequence

import numpy as np
import pandas as pd

from . import solar_wind_scarcity_regime as v1


PROTOCOL_VERSION = "solar_wind_scarcity_ablation_v1"
VARIANTS = ("direct_quantile", "regime_hour_local", "regime_calibration_oof90")
DAILY_PEAK_COLUMNS = (
    "own_daily_peak_residual_stress", "own_daily_peak_deficit_stress",
    "other_daily_peak_residual_stress", "other_daily_peak_deficit_stress",
)
CALIBRATION_DAYS = 90
CALIBRATION_C = 0.1
MINIMUM_CALIBRATION_SPIKES = 20
MINIMUM_COUNTRY_SPIKES = 5
MINIMUM_CALIBRATION_NORMAL = 32
LOGIT_EPSILON = 1e-6
PARENT_QUANTILE_COLUMNS = ("q10", "q50", "q90")
REPRODUCTION_ATOL = 1e-7
REPRODUCTION_RTOL = 1e-8


def _parameters(threads: int, iterations: int, seed: int) -> dict:
    if threads != 1 or not 1 <= iterations <= 120:
        raise ValueError("Ablation resource contract requires one thread and 1-120 trees")
    return {"iterations": int(iterations), "depth": 4, "learning_rate": .05,
            "l2_leaf_reg": 3.0, "random_seed": int(seed), "thread_count": 1,
            "allow_writing_files": False, "verbose": False, "task_type": "CPU"}


def _training_arrays(train_X, train_residual, test_X, train_days, origin_day):
    origin = pd.Timestamp(origin_day).date()
    if (not isinstance(train_X, pd.DataFrame) or not isinstance(test_X, pd.DataFrame)
            or train_X.empty or test_X.empty or not train_X.columns.is_unique
            or not train_X.columns.equals(test_X.columns)):
        raise ValueError("Nonempty unique, identically ordered train/test feature columns required")
    train, test = train_X.to_numpy(dtype=float), test_X.to_numpy(dtype=float)
    residual = np.asarray(train_residual, dtype=float)
    parsed_days = pd.DatetimeIndex(pd.to_datetime(list(train_days)))
    if (residual.shape != (len(train),) or len(parsed_days) != len(train)
            or parsed_days.hasnans):
        raise ValueError("Training residuals and civil days must match feature rows")
    if not all(np.isfinite(a).all() for a in (train, test, residual)):
        raise ValueError("Finite training/test features and residuals required; no imputation")
    days = np.asarray(parsed_days.date)
    if np.any(days >= origin) or np.any(days < origin - timedelta(days=365)):
        raise ValueError("Training days must lie in [origin-365,origin); no origin/future labels")
    if len(set(days)) < 90:
        raise ValueError("At least 90 distinct earlier training days are required")
    return train, residual, test, days, origin


def select_hour_local_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Delete exactly four repeated daily peaks; leave all other values intact."""
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("A feature frame with unique columns is required")
    if not set(DAILY_PEAK_COLUMNS).issubset(frame.columns):
        raise ValueError("Hour-local ablation requires all four original v1 daily-peak columns")
    return frame.drop(columns=list(DAILY_PEAK_COLUMNS)).copy()


def _test_keys(test_X: pd.DataFrame, origin: date) -> pd.MultiIndex:
    if "country_is_nl" not in test_X:
        raise ValueError("country_is_nl is required for aligned country recalibration")
    flags = test_X["country_is_nl"].to_numpy(dtype=float)
    if not np.isfinite(flags).all() or not np.isin(flags, [0., 1.]).all():
        raise ValueError("country_is_nl must contain exactly 0 or 1")
    zones = np.where(flags == 1, "NL", "DE")
    if isinstance(test_X.index, pd.MultiIndex):
        if list(test_X.index.names) != ["timestamp_utc", "zone"]:
            raise ValueError("Test MultiIndex must be (timestamp_utc,zone)")
        timestamps = pd.DatetimeIndex(test_X.index.get_level_values("timestamp_utc"))
        if not np.array_equal(test_X.index.get_level_values("zone"), zones):
            raise ValueError("Test index zones disagree with country_is_nl")
    else:
        timestamps = test_X.index
    if (not isinstance(timestamps, pd.DatetimeIndex) or str(timestamps.tz) != "UTC"
            or timestamps.hasnans or not timestamps.equals(timestamps.floor("h"))):
        raise ValueError("Test timestamps must be finite hour-aligned UTC timestamps")
    if np.any(np.asarray(timestamps.tz_convert("Europe/Berlin").date) < origin):
        raise ValueError("Prediction targets cannot precede their model origin")
    keys = pd.MultiIndex.from_arrays([timestamps, zones], names=["timestamp_utc", "zone"])
    if not keys.is_unique:
        raise ValueError("Duplicate test (timestamp_utc,zone) identity")
    return keys


def _aligned_parent(parent_probability, parent_prediction_origin, test_X, origin):
    keys = _test_keys(test_X, origin)
    for value, name in ((parent_probability, "parent_probability"),
                        (parent_prediction_origin, "parent_prediction_origin")):
        if (not isinstance(value, pd.Series) or not isinstance(value.index, pd.MultiIndex)
                or list(value.index.names) != ["timestamp_utc", "zone"]
                or not value.index.is_unique or not value.index.equals(keys)):
            raise ValueError(f"{name} must be an exactly aligned (timestamp_utc,zone) Series")
    probability = parent_probability.to_numpy(dtype=float)
    if (not np.isfinite(probability).all()
            or not ((probability >= 0) & (probability <= 1)).all()):
        raise ValueError("Parent probabilities must be finite and in [0,1]")
    origins = pd.DatetimeIndex(pd.to_datetime(parent_prediction_origin.to_numpy()))
    if origins.hasnans or not np.all(np.asarray(origins.date) == origin):
        raise ValueError("Every sealed parent prediction must use this exact weekly origin")
    return keys, probability


def recalibrate_probability(
    parent_probability: pd.Series, test_X: pd.DataFrame, calibration_frame: pd.DataFrame,
    *, origin_day: str | date, parent_prediction_origin: pd.Series,
) -> tuple[np.ndarray, dict]:
    """Recalibrate prior v1 probabilities using a strict complete prequential panel.

    calibration_frame requires timestamp_utc, zone, fit_origin,
    spike_probability and residual. It may contain older rows, ignored after
    validation, but origin/future targets are rejected. fit_origin <= target
    civil day is checked for every row. The caller must establish that each
    saved probability was actually generated without its subsequent label.
    Missing coverage or insufficient classes returns EXACT original p, never
    an invented or imputed probability. A nonpositive fitted slope also falls
    back to identity rather than silently reversing the original ordering.
    """
    origin = pd.Timestamp(origin_day).date()
    _, original = _aligned_parent(parent_probability, parent_prediction_origin, test_X, origin)
    required = {"timestamp_utc", "zone", "fit_origin", "spike_probability", "residual"}
    if (not isinstance(calibration_frame, pd.DataFrame)
            or not calibration_frame.columns.is_unique
            or not required.issubset(calibration_frame.columns)):
        raise ValueError("Calibration needs explicit timestamp, country, fit origin, p and residual")
    frame = calibration_frame.loc[:, sorted(required)].copy()
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp_utc"]))
    if (str(timestamps.tz) != "UTC" or timestamps.hasnans
            or not timestamps.equals(timestamps.floor("h"))):
        raise ValueError("Calibration timestamps must be finite hour-aligned UTC")
    zones = frame["zone"].to_numpy()
    if not np.isin(zones, ["DE", "NL"]).all():
        raise ValueError("Calibration countries must be DE or NL")
    keys = pd.MultiIndex.from_arrays([timestamps, zones], names=["timestamp_utc", "zone"])
    if not keys.is_unique:
        raise ValueError("Duplicate calibration (timestamp_utc,zone) identity")
    days = np.asarray(timestamps.tz_convert("Europe/Berlin").date)
    fit_origins = pd.DatetimeIndex(pd.to_datetime(frame["fit_origin"]))
    if fit_origins.hasnans or np.any(np.asarray(fit_origins.date) > days):
        raise ValueError("Calibration model origins must not follow their target delivery day")
    if np.any(days >= origin):
        raise ValueError("Origin/future calibration targets and labels are forbidden")
    values = frame[["spike_probability", "residual"]].to_numpy(dtype=float)
    if (not np.isfinite(values).all()
            or not ((values[:, 0] >= 0) & (values[:, 0] <= 1)).all()):
        raise ValueError("Calibration probabilities/labels must be finite with p in [0,1]")
    first = origin - timedelta(days=CALIBRATION_DAYS)
    selected = days >= first
    selected_keys = keys[selected]
    left = pd.Timestamp(first).tz_localize("Europe/Berlin")
    right = pd.Timestamp(origin).tz_localize("Europe/Berlin")
    expected_times = pd.date_range(left, right, freq="h", inclusive="left").tz_convert("UTC")
    expected_keys = pd.MultiIndex.from_product(
        [expected_times, ["DE", "NL"]], names=["timestamp_utc", "zone"],
    )
    complete = (len(selected_keys) == len(expected_keys)
                and selected_keys.sort_values().equals(expected_keys.sort_values()))
    selected_values = values[selected]
    selected_zones = zones[selected]
    targets = (selected_values[:, 1] > v1.SPIKE_THRESHOLD).astype(int)
    positives = int(targets.sum())
    country_positives = {zone: int(targets[selected_zones == zone].sum()) for zone in ("DE", "NL")}
    audit = {
        "method": "identity", "window_days": CALIBRATION_DAYS,
        "first_day": first.isoformat(), "last_day": (origin - timedelta(days=1)).isoformat(),
        "origin_day": origin.isoformat(), "rows": int(selected.sum()),
        "expected_rows": len(expected_keys), "complete_two_country_panel": bool(complete),
        "older_rows_ignored": int((~selected).sum()), "positive_rows": positives,
        "negative_rows": len(targets) - positives, "positive_rows_by_country": country_positives,
        "minimum_positive_rows": MINIMUM_CALIBRATION_SPIKES,
        "minimum_country_positive_rows": MINIMUM_COUNTRY_SPIKES,
        "minimum_negative_rows": MINIMUM_CALIBRATION_NORMAL,
        "input_probability": "saved prequential v1 probability, including original v1 calibration",
        "class_weights_used": False, "C": CALIBRATION_C,
        "logit_probability_epsilon": LOGIT_EPSILON, "slope": None, "country_offset": None,
        "intercept": None, "probability_order": "aligned timestamp_utc + country identity",
        "label_definition": "realized residual against frozen interaction40 P50 > 50",
        "prequential_provenance_verified_by_module": False,
    }
    if not complete:
        audit["fallback_reason"] = "incomplete_exact_90_day_two_country_panel"
        return original.copy(), audit
    if (positives < MINIMUM_CALIBRATION_SPIKES
            or any(n < MINIMUM_COUNTRY_SPIKES for n in country_positives.values())
            or len(targets) - positives < MINIMUM_CALIBRATION_NORMAL):
        audit["fallback_reason"] = "insufficient_pooled_or_country_class_counts"
        return original.copy(), audit
    from sklearn.linear_model import LogisticRegression
    # Canonical order makes the tiny calibration fit independent of file layout.
    order = np.lexsort((selected_zones, timestamps[selected].asi8))
    clipped = np.clip(selected_values[:, 0], LOGIT_EPSILON, 1 - LOGIT_EPSILON)
    features = np.column_stack([np.log(clipped / (1 - clipped)), (selected_zones == "NL").astype(float)])
    model = LogisticRegression(C=CALIBRATION_C, solver="lbfgs", max_iter=1000,
                               class_weight=None, random_state=20260923)
    model.fit(features[order], targets[order])
    slope, offset = map(float, model.coef_[0])
    audit.update(slope=slope, country_offset=offset, intercept=float(model.intercept_[0]))
    if not np.isfinite([slope, offset, model.intercept_[0]]).all() or slope <= 0:
        audit["fallback_reason"] = "nonpositive_or_nonfinite_calibration_slope"
        return original.copy(), audit
    clipped_test = np.clip(original, LOGIT_EPSILON, 1 - LOGIT_EPSILON)
    test_features = np.column_stack([
        np.log(clipped_test / (1 - clipped_test)), test_X["country_is_nl"].to_numpy(dtype=float),
    ])
    probability = model.predict_proba(test_features)[:, 1]
    audit.update(method="logistic_oof90", fallback_reason=None,
                 logistic_iterations=int(model.n_iter_.max()))
    return probability, audit


def fit_variant(
    variant: str, train_X: pd.DataFrame, train_residual: Sequence[float], test_X: pd.DataFrame,
    train_days: Sequence, *, origin_day: str | date, parent_probability: pd.Series | None = None,
    parent_prediction_origin: pd.Series | None = None,
    parent_residual_quantiles: pd.DataFrame | None = None,
    calibration_frame: pd.DataFrame | None = None, threads: int = 1,
    iterations: int = 120, seed: int = 20260923,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Run one bounded isolated fit; no orchestration or persistent side effects.

    Variant 3 requires sealed parent probabilities, per-row fit origins, and
    residual quantiles, all exactly indexed by (timestamp_utc,zone). Parent
    residual-quantile columns must be q10,q50,q90 in that order. Reconstruction
    with original p must match the parent before recalibrated p is applied.
    Direct quantile has no gate: its returned probability is None, not a proxy.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown preregistered ablation: {variant}")
    parameters = _parameters(threads, iterations, seed)
    train, residual, test, days, origin = _training_arrays(
        train_X, train_residual, test_X, train_days, origin_day,
    )
    audit = {
        "protocol_version": PROTOCOL_VERSION, "variant": variant,
        "origin_day": origin.isoformat(), "train_first_day": min(days).isoformat(),
        "train_last_day": max(days).isoformat(), "train_unique_days": len(set(days)),
        "train_rows": len(train), "test_rows": len(test), "feature_names": list(train_X.columns),
        "model_parameters": parameters, "correction_clip": None, "class_weights_used": False,
        "output_definition": "residual q10/q50/q90, each added to SAME frozen interaction40 P50",
        "pit_publication_evidence_verified": False, "prospective_calibration_verified": False,
    }
    if variant == "regime_hour_local":
        quantiles, probability, original_audit = v1.fit_predict(
            select_hour_local_features(train_X), residual, select_hour_local_features(test_X),
            train_days, origin_day=origin, threads=threads, iterations=iterations, seed=seed,
        )
        audit.update(dropped_columns=list(DAILY_PEAK_COLUMNS),
                     feature_names=list(select_hour_local_features(train_X).columns),
                     parent_recipe_audit=original_audit)
        return quantiles, probability, audit
    if variant == "direct_quantile":
        if np.ptp(residual) == 0 or not np.any(np.ptp(train, axis=0) > 0):
            quantiles = np.repeat(np.quantile(residual, [.1, .5, .9])[None, :], len(test), axis=0)
            audit.update(method="empirical_quantiles", fallback_reason="constant_features_or_targets",
                         quantile_crossing_rows=0)
        else:
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(loss_function="MultiQuantile:alpha=0.1,0.5,0.9", **parameters)
            model.fit(train, residual)
            raw = np.asarray(model.predict(test), dtype=float)
            quantiles = np.sort(raw, axis=1)
            audit.update(method="catboost_multiquantile", fallback_reason=None,
                         quantile_crossing_rows=int(np.any(np.diff(raw, axis=1) < 0, axis=1).sum()),
                         monotonicity_repair="increasing rearrangement of predicted quantiles")
        audit["gate_probability"] = "not defined; no gate or spike Brier metric for this variant"
        return quantiles, None, audit
    keys, original_p = _aligned_parent(parent_probability, parent_prediction_origin, test_X, origin)
    if (not isinstance(parent_residual_quantiles, pd.DataFrame)
            or not parent_residual_quantiles.index.is_unique
            or not parent_residual_quantiles.index.equals(keys)
            or list(parent_residual_quantiles.columns) != list(PARENT_QUANTILE_COLUMNS)):
        raise ValueError("Parent residual quantiles must be aligned (timestamp_utc,zone) q10/q50/q90")
    parent_q = parent_residual_quantiles.to_numpy(dtype=float)
    if not np.isfinite(parent_q).all() or np.any(np.diff(parent_q, axis=1) < 0):
        raise ValueError("Parent residual quantiles must be finite and nondecreasing")
    target = residual > v1.SPIKE_THRESHOLD
    normal, normal_audit = v1._expert(train[~target], residual[~target], test, "normal", parameters)
    spike, spike_audit = v1._expert(train[target], residual[target], test, "spike", parameters)
    reconstructed = v1.mixture_quantiles(normal, spike, original_p)
    difference = float(np.max(np.abs(reconstructed - parent_q)))
    if not np.allclose(reconstructed, parent_q, atol=REPRODUCTION_ATOL, rtol=REPRODUCTION_RTOL):
        raise ValueError(f"Frozen v1 expert/mixture reproduction mismatch; max absolute difference={difference}")
    probability, calibration_audit = recalibrate_probability(
        parent_probability, test_X, calibration_frame, origin_day=origin,
        parent_prediction_origin=parent_prediction_origin,
    )
    quantiles = v1.mixture_quantiles(normal, spike, probability)
    audit.update(method="v1_experts_with_prequential_probability_recalibration",
                 experts={"normal": normal_audit, "spike": spike_audit},
                 calibration=calibration_audit,
                 parent_reproduction_max_absolute_difference=difference,
                 parent_reproduction_atol=REPRODUCTION_ATOL,
                 parent_reproduction_rtol=REPRODUCTION_RTOL,
                 current_gate_refit_performed=False,
                 combination="generalized inverse of (1-p)*F_normal+p*F_spike")
    return quantiles, probability, audit


__all__ = ["PROTOCOL_VERSION", "VARIANTS", "DAILY_PEAK_COLUMNS",
           "fit_variant", "select_hour_local_features", "recalibrate_probability"]
