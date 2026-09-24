"""Causal shadow challenger for sudden day-ahead price regimes.

The challenger is deliberately a post-processing layer.  It consumes a sealed
day-ahead control forecast and point-in-time residual-load vintages, estimates
the probability and magnitude of an *upward* solar-hours miss, and writes no
production state.  All labels used for a delivery day D stop at D-2.

The module is data-only: filesystem orchestration lives in
``run_price_regime_challenger.py``.  Keeping the transformations here pure
makes the leakage and DST contracts directly testable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import roc_auc_score


DELIVERY_TIMEZONE = "Europe/Paris"
RESIDUAL_ALIASES: tuple[str, ...] = tuple(
    f"{zone}_residual_load_fcst" for zone in ("fr", "de", "be", "nl", "es")
)
QUANTILES: tuple[str, ...] = ("q10", "q50", "q90")


class RegimeChallengerError(ValueError):
    """Raised when the causal challenger contract is violated."""


@dataclass(frozen=True)
class RegimeChallengerConfig:
    source_path: Path
    challenger_id: str
    timezone: str
    label_delay_days: int
    solar_hours: tuple[int, ...]
    shock_error_threshold_eur_mwh: float
    evaluation_days: int
    minimum_training_days: int
    refit_every_days: int
    minimum_training_rows: int
    minimum_positive_rows: int
    gate_probability_threshold: float
    n_estimators: int
    max_depth: int
    classifier_min_samples_leaf: int
    regressor_min_samples_leaf: int
    max_premium_eur_mwh: float
    q10_premium_multiplier: float
    q50_premium_multiplier: float
    q90_premium_multiplier: float
    random_state: int
    pit_root: Path
    pit_files: Mapping[str, Path]
    output_root: Path


@dataclass(frozen=True)
class FittedRegimeModel:
    feature_names: tuple[str, ...]
    medians: pd.Series
    classifier: ExtraTreesClassifier | None
    regressor: ExtraTreesRegressor | None
    constant_probability: float
    constant_magnitude: float
    feature_importance: pd.DataFrame
    diagnostics: Mapping[str, Any]


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} doit etre un mapping.")
    return value


def _resolve(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _positive_int(value: Any, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} doit etre un entier.")
    result = int(value)
    if result != value or result < minimum:
        raise ValueError(f"{name} doit etre >= {minimum}.")
    return result


def load_regime_challenger_config(
    path: str | Path,
    *,
    project_root: str | Path,
) -> RegimeChallengerConfig:
    """Load the frozen shadow recipe and reject unsafe output locations."""

    root = Path(project_root).expanduser().resolve()
    source = _resolve(path, base=root)
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload = _mapping(raw, name="configuration")
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("schema_version doit valoir 1.")
    challenger_id = str(payload.get("challenger_id", "")).strip()
    if not challenger_id or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
        for character in challenger_id
    ):
        raise ValueError("challenger_id doit etre un slug minuscule sur.")
    if bool(payload.get("production_eligible", True)):
        raise ValueError("Un challenger shadow exige production_eligible: false.")

    protocol = _mapping(payload.get("protocol"), name="protocol")
    timezone = str(protocol.get("timezone", ""))
    if timezone != DELIVERY_TIMEZONE:
        raise ValueError(f"timezone doit rester {DELIVERY_TIMEZONE}.")
    if str(protocol.get("forecast_origin_local_time", "")) != "08:00":
        raise ValueError("forecast_origin_local_time doit rester 08:00.")
    label_delay = _positive_int(
        protocol.get("label_delay_days", 0), name="label_delay_days"
    )
    if label_delay < 2:
        raise ValueError("label_delay_days doit etre >= 2.")
    solar_raw = protocol.get("solar_hours")
    if not isinstance(solar_raw, Sequence) or isinstance(solar_raw, (str, bytes)):
        raise TypeError("solar_hours doit etre une liste d'heures locales.")
    solar_hours = tuple(dict.fromkeys(int(item) for item in solar_raw))
    if not solar_hours or any(item < 0 or item > 23 for item in solar_hours):
        raise ValueError("solar_hours doit contenir des heures entre 0 et 23.")

    validation = _mapping(payload.get("validation"), name="validation")
    model = _mapping(payload.get("model"), name="model")
    quantiles = _mapping(model.get("quantile_adjustment"), name="quantile_adjustment")
    inputs = _mapping(payload.get("inputs"), name="inputs")
    pit_root = _resolve(inputs.get("pit_root", ""), base=root)
    files = _mapping(inputs.get("residual_load_files"), name="residual_load_files")
    if set(files) != set(RESIDUAL_ALIASES):
        raise ValueError(
            "residual_load_files doit contenir exactement les cinq aliases."
        )
    pit_files = {
        alias: _resolve(str(files[alias]), base=pit_root)
        for alias in RESIDUAL_ALIASES
    }

    outputs = _mapping(payload.get("outputs"), name="outputs")
    output_root = _resolve(outputs.get("root", ""), base=root)
    allowed = (root / "runs" / "challengers").resolve()
    try:
        relative = output_root.relative_to(allowed)
    except ValueError as exc:
        raise ValueError("La sortie challenger doit rester sous runs/challengers.") from exc
    if not relative.parts:
        raise ValueError("La racine runs/challengers elle-meme est interdite.")

    gate_threshold = float(model.get("gate_probability_threshold", 0.35))
    if not 0.0 < gate_threshold < 1.0:
        raise ValueError("gate_probability_threshold doit etre dans ]0,1[.")
    multipliers = tuple(float(quantiles.get(key, np.nan)) for key in QUANTILES)
    if not all(np.isfinite(multipliers)) or not (
        0.0 <= multipliers[0] <= multipliers[1] <= multipliers[2]
    ):
        raise ValueError("Les multiplicateurs q10/q50/q90 doivent etre croissants.")
    shock_threshold = float(
        protocol.get("shock_error_threshold_eur_mwh", 25.0)
    )
    maximum_premium = float(model.get("max_premium_eur_mwh", 120.0))
    if not np.isfinite(shock_threshold) or shock_threshold <= 0.0:
        raise ValueError("shock_error_threshold_eur_mwh doit etre fini et > 0.")
    if not np.isfinite(maximum_premium) or maximum_premium <= 0.0:
        raise ValueError("max_premium_eur_mwh doit etre fini et > 0.")

    return RegimeChallengerConfig(
        source_path=source,
        challenger_id=challenger_id,
        timezone=timezone,
        label_delay_days=label_delay,
        solar_hours=solar_hours,
        shock_error_threshold_eur_mwh=shock_threshold,
        evaluation_days=_positive_int(
            validation.get("evaluation_days", 180), name="evaluation_days"
        ),
        minimum_training_days=_positive_int(
            validation.get("minimum_training_days", 120),
            name="minimum_training_days",
        ),
        refit_every_days=_positive_int(
            validation.get("refit_every_days", 28), name="refit_every_days"
        ),
        minimum_training_rows=_positive_int(
            model.get("minimum_training_rows", 800), name="minimum_training_rows"
        ),
        minimum_positive_rows=_positive_int(
            model.get("minimum_positive_rows", 24), name="minimum_positive_rows"
        ),
        gate_probability_threshold=gate_threshold,
        n_estimators=_positive_int(model.get("n_estimators", 160), name="n_estimators"),
        max_depth=_positive_int(model.get("max_depth", 7), name="max_depth"),
        classifier_min_samples_leaf=_positive_int(
            model.get("classifier_min_samples_leaf", 16),
            name="classifier_min_samples_leaf",
        ),
        regressor_min_samples_leaf=_positive_int(
            model.get("regressor_min_samples_leaf", 8),
            name="regressor_min_samples_leaf",
        ),
        max_premium_eur_mwh=maximum_premium,
        q10_premium_multiplier=multipliers[0],
        q50_premium_multiplier=multipliers[1],
        q90_premium_multiplier=multipliers[2],
        random_state=int(model.get("random_state", 425)),
        pit_root=pit_root,
        pit_files=pit_files,
        output_root=output_root,
    )


def scheduled_origin_utc(
    delivery_day: date | str | pd.Timestamp,
    *,
    timezone: str = DELIVERY_TIMEZONE,
) -> pd.Timestamp:
    """Return the strict D-1 08:00 local forecast origin."""

    stamp = pd.Timestamp(delivery_day)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(timezone).tz_localize(None)
    local = stamp.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
    return local.tz_localize(timezone, ambiguous="raise", nonexistent="raise").tz_convert(
        "UTC"
    )


def _utc_index(index: pd.Index, *, name: str, contiguous: bool = False) -> pd.DatetimeIndex:
    result = pd.DatetimeIndex(pd.to_datetime(index, utc=True, errors="raise"))
    if result.hasnans or not result.is_unique or not result.is_monotonic_increasing:
        raise RegimeChallengerError(f"{name} doit etre UTC, unique et croissant.")
    if contiguous and len(result) > 1:
        gaps = result[1:] - result[:-1]
        if not bool((gaps == pd.Timedelta(hours=1)).all()):
            raise RegimeChallengerError(f"{name} doit etre horaire et contigu.")
    return result


def normalize_vintage_store(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Normalize a Saturn vintage store without weakening its PIT timestamps."""

    required = {"value_time_utc", "snapshot_time_utc", "revision_time_utc", "value"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RegimeChallengerError(f"{name}: colonnes PIT absentes: {missing}.")
    result = frame.copy().reset_index(drop=True)
    if "downloaded_at_utc" not in result:
        result["downloaded_at_utc"] = pd.NaT
    for column in (
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
        "downloaded_at_utc",
    ):
        result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
    if result[["value_time_utc", "snapshot_time_utc", "revision_time_utc"]].isna().any().any():
        raise RegimeChallengerError(f"{name}: timestamp PIT absent ou invalide.")
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    if bool(np.isinf(result["value"].to_numpy(float)).any()):
        raise RegimeChallengerError(f"{name}: valeur infinie.")
    result["_row_order"] = np.arange(len(result), dtype=np.int64)
    return result


def _origin_by_value_time(
    value_time: pd.DatetimeIndex,
    *,
    timezone: str,
) -> pd.Series:
    local_days = value_time.tz_convert(timezone).date
    origins = {
        day: scheduled_origin_utc(day, timezone=timezone)
        for day in dict.fromkeys(local_days)
    }
    return pd.Series(
        [origins[day] for day in local_days],
        index=value_time,
        dtype="datetime64[ns, UTC]",
    )


def select_forecasts_asof(
    store: pd.DataFrame,
    *,
    alias: str,
    grid: pd.DatetimeIndex,
    timezone: str = DELIVERY_TIMEZONE,
) -> tuple[pd.Series, dict[str, Any]]:
    """Select the latest vintage eligible at each delivery day's D-1 cutoff."""

    timeline = _utc_index(grid, name="grid", contiguous=True)
    normalized = normalize_vintage_store(store, name=alias)
    lower = timeline[0]
    upper = timeline[-1]
    frame = normalized.loc[
        normalized["value_time_utc"].between(lower, upper, inclusive="both")
    ].copy()
    value_index = pd.DatetimeIndex(frame["value_time_utc"])
    frame["cutoff_utc"] = _origin_by_value_time(
        value_index, timezone=timezone
    ).to_numpy()
    eligible = frame.loc[
        frame["snapshot_time_utc"].le(frame["cutoff_utc"])
        & frame["revision_time_utc"].le(frame["cutoff_utc"])
    ]
    selected = (
        eligible.sort_values(
            [
                "value_time_utc",
                "revision_time_utc",
                "snapshot_time_utc",
                "downloaded_at_utc",
                "_row_order",
            ],
            kind="stable",
            na_position="first",
        )
        .drop_duplicates("value_time_utc", keep="last")
        .set_index("value_time_utc")
        .sort_index()
    )
    aligned = selected.reindex(timeline)
    expected_origins = _origin_by_value_time(timeline, timezone=timezone)
    violation = aligned["snapshot_time_utc"].gt(expected_origins) | aligned[
        "revision_time_utc"
    ].gt(expected_origins)
    if bool(violation.fillna(False).any()):
        raise AssertionError(f"{alias}: une vintage future a traverse le filtre PIT.")
    values = pd.to_numeric(aligned["value"], errors="coerce").astype(float)
    values.index = timeline
    values.name = alias
    finite = np.isfinite(values.to_numpy(float))
    local_days = timeline.tz_convert(timezone).date
    last_local_day = local_days[-1]
    last_mask = np.asarray(local_days == last_local_day)
    last_finite = finite & last_mask
    revision_age_hours = (
        expected_origins.reset_index(drop=True)
        - aligned["revision_time_utc"].reset_index(drop=True)
    ).dt.total_seconds() / 3600.0
    snapshot_age_hours = (
        expected_origins.reset_index(drop=True)
        - aligned["snapshot_time_utc"].reset_index(drop=True)
    ).dt.total_seconds() / 3600.0
    audit = {
        "alias": alias,
        "raw_rows": int(len(normalized)),
        "rows_in_grid_envelope": int(len(frame)),
        "eligible_rows": int(len(eligible)),
        "selected_rows": int(len(selected)),
        "coverage": float(finite.mean()),
        "missing_rows": int((~finite).sum()),
        "cutoff_violations": 0,
        "maximum_selected_snapshot_time_utc": aligned["snapshot_time_utc"].max(),
        "maximum_selected_revision_time_utc": aligned["revision_time_utc"].max(),
        "last_local_day": last_local_day,
        "last_day_rows": int(last_mask.sum()),
        "last_day_coverage": float(last_finite.sum() / last_mask.sum()),
        "last_day_max_snapshot_age_hours": float(
            snapshot_age_hours.loc[last_finite].max()
        )
        if bool(last_finite.any())
        else np.nan,
        "last_day_max_revision_age_hours": float(
            revision_age_hours.loc[last_finite].max()
        )
        if bool(last_finite.any())
        else np.nan,
    }
    return values, audit


def previous_local_day_values(
    series: pd.Series,
    *,
    timezone: str = DELIVERY_TIMEZONE,
) -> pd.Series:
    """Align D-1 by civil hour and repeated-hour occurrence (DST safe)."""

    index = _utc_index(series.index, name="series.index", contiguous=True)
    local = index.tz_convert(timezone)
    keys = pd.DataFrame(
        {
            "local_date": local.date,
            "local_hour": local.hour,
            "value": pd.to_numeric(series, errors="coerce").to_numpy(float),
        },
        index=index,
    )
    keys["occurrence"] = keys.groupby(["local_date", "local_hour"]).cumcount()
    lookup = {
        (row.local_date, int(row.local_hour), int(row.occurrence)): float(row.value)
        for row in keys.itertuples()
    }
    previous = [
        lookup.get(
            (day - timedelta(days=1), int(hour), int(occurrence)), np.nan
        )
        for day, hour, occurrence in keys[
            ["local_date", "local_hour", "occurrence"]
        ].itertuples(index=False, name=None)
    ]
    return pd.Series(previous, index=index, name=series.name, dtype=float)


def _daily_stat(
    values: pd.Series,
    local_dates: Sequence[date],
    mask: np.ndarray,
    operation: str,
) -> pd.Series:
    frame = pd.DataFrame(
        {"value": pd.to_numeric(values, errors="coerce"), "local_date": local_dates},
        index=values.index,
    )
    selected = frame.loc[mask]
    grouped = selected.groupby("local_date")["value"]
    aggregate = getattr(grouped, operation)()
    return pd.Series(local_dates, index=values.index).map(aggregate).astype(float)


def build_regime_features(
    residual_load: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    peer_prices: Mapping[str, pd.Series] | None = None,
    timezone: str = DELIVERY_TIMEZONE,
    solar_hours: Sequence[int] = tuple(range(9, 17)),
) -> pd.DataFrame:
    """Create only forecast-time-known level, ramp and J/J-1 features."""

    index = _utc_index(residual_load.index, name="residual_load.index", contiguous=True)
    if not index.equals(_utc_index(baseline.index, name="baseline.index", contiguous=True)):
        raise RegimeChallengerError("Residual load et baseline doivent partager la grille.")
    missing_aliases = sorted(set(RESIDUAL_ALIASES).difference(residual_load.columns))
    missing_quantiles = sorted(set(QUANTILES).difference(baseline.columns))
    if missing_aliases or missing_quantiles:
        raise RegimeChallengerError(
            f"Colonnes absentes: residual={missing_aliases}, baseline={missing_quantiles}."
        )
    local = index.tz_convert(timezone)
    local_dates = np.asarray(local.date)
    local_hours = np.asarray(local.hour, dtype=int)
    solar_mask = np.isin(local_hours, np.asarray(tuple(solar_hours), dtype=int))
    features = pd.DataFrame(index=index)
    features["local_hour_sin"] = np.sin(2.0 * np.pi * local_hours / 24.0)
    features["local_hour_cos"] = np.cos(2.0 * np.pi * local_hours / 24.0)
    features["dow_sin"] = np.sin(2.0 * np.pi * np.asarray(local.dayofweek) / 7.0)
    features["dow_cos"] = np.cos(2.0 * np.pi * np.asarray(local.dayofweek) / 7.0)
    features["doy_sin"] = np.sin(2.0 * np.pi * np.asarray(local.dayofyear) / 365.25)
    features["doy_cos"] = np.cos(2.0 * np.pi * np.asarray(local.dayofyear) / 365.25)
    features["is_weekend"] = (np.asarray(local.dayofweek) >= 5).astype(float)
    features["is_solar_hour"] = solar_mask.astype(float)

    delta_columns: list[str] = []
    for alias in RESIDUAL_ALIASES:
        values = pd.to_numeric(residual_load[alias], errors="coerce").astype(float)
        previous = previous_local_day_values(values, timezone=timezone)
        prefix = alias.removesuffix("_fcst")
        features[prefix] = values
        features[f"{prefix}__available"] = values.notna().astype(float)
        delta_name = f"{prefix}__delta_d1"
        features[delta_name] = values - previous
        delta_columns.append(delta_name)
        local_frame = pd.DataFrame(
            {"value": values.to_numpy(float), "local_date": local_dates}, index=index
        )
        features[f"{prefix}__ramp_1h"] = local_frame.groupby("local_date")[
            "value"
        ].diff()
        day_mean = _daily_stat(values, local_dates, np.ones(len(index), dtype=bool), "mean")
        solar_mean = _daily_stat(values, local_dates, solar_mask, "mean")
        solar_min = _daily_stat(values, local_dates, solar_mask, "min")
        solar_max = _daily_stat(values, local_dates, solar_mask, "max")
        features[f"{prefix}__day_mean"] = day_mean
        features[f"{prefix}__solar_mean"] = solar_mean
        features[f"{prefix}__solar_min"] = solar_min
        features[f"{prefix}__solar_max"] = solar_max
        features[f"{prefix}__solar_mean_delta_d1"] = solar_mean - previous_local_day_values(
            solar_mean, timezone=timezone
        )

    features["residual_delta_mean"] = features[delta_columns].mean(axis=1)
    features["residual_delta_max"] = features[delta_columns].max(axis=1)
    features["residual_delta_min"] = features[delta_columns].min(axis=1)
    features["residual_delta_dispersion"] = features[delta_columns].std(axis=1)
    availability_columns = [
        f"{alias.removesuffix('_fcst')}__available" for alias in RESIDUAL_ALIASES
    ]
    features["residual_available_zone_count"] = features[
        availability_columns
    ].sum(axis=1)
    features["residual_all_zones_available"] = features[
        "residual_available_zone_count"
    ].eq(float(len(RESIDUAL_ALIASES))).astype(float)
    features["fr_vs_neighbours_delta"] = features[
        "fr_residual_load__delta_d1"
    ] - features[
        [
            "de_residual_load__delta_d1",
            "be_residual_load__delta_d1",
            "nl_residual_load__delta_d1",
        ]
    ].mean(axis=1)

    for quantile in QUANTILES:
        features[f"baseline_{quantile}"] = pd.to_numeric(
            baseline[quantile], errors="coerce"
        )
    features["baseline_interval_width"] = (
        features["baseline_q90"] - features["baseline_q10"]
    )
    baseline_previous = previous_local_day_values(
        features["baseline_q50"], timezone=timezone
    )
    features["baseline_q50_delta_d1"] = features["baseline_q50"] - baseline_previous

    for zone, series in sorted((peer_prices or {}).items()):
        peer = pd.to_numeric(series.reindex(index), errors="coerce").astype(float)
        name = f"peer_{str(zone).lower()}_q50"
        features[name] = peer
        features[f"baseline_spread_vs_{str(zone).lower()}"] = (
            features["baseline_q50"] - peer
        )
    features.index.name = "delivery_start_utc"
    return features.astype(float)


def _model_matrix(
    features: pd.DataFrame,
    *,
    feature_names: Sequence[str] | None = None,
    medians: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    numeric = features.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    if feature_names is not None:
        missing = sorted(set(feature_names).difference(numeric.columns))
        if missing:
            raise RegimeChallengerError(f"Features live absentes: {missing}.")
        numeric = numeric.loc[:, list(feature_names)]
    if medians is None:
        medians = numeric.median(axis=0, skipna=True).fillna(0.0)
    matrix = numeric.fillna(medians).fillna(0.0)
    if bool(np.isinf(matrix.to_numpy(float)).any()):
        raise RegimeChallengerError("Matrice de features infinie apres imputation.")
    return matrix.astype(float), medians.astype(float)


def fit_regime_model(
    features: pd.DataFrame,
    actual: pd.Series,
    baseline_q50: pd.Series,
    *,
    config: RegimeChallengerConfig,
    thread_count: int = 1,
) -> FittedRegimeModel:
    """Fit the gate and positive-premium expert on solar-hour rows only."""

    if int(thread_count) < 1:
        raise ValueError("thread_count doit etre >= 1.")
    # Keep the shadow fit single-process.  On Windows, sklearn's parallel
    # forest backend creates OS pipes; those are unavailable in hardened
    # forecast sessions and do not change the deterministic recipe.
    model_jobs = 1

    aligned_actual = pd.to_numeric(actual.reindex(features.index), errors="coerce")
    aligned_baseline = pd.to_numeric(
        baseline_q50.reindex(features.index), errors="coerce"
    )
    solar = features["is_solar_hour"].eq(1.0)
    usable = solar & aligned_actual.notna() & aligned_baseline.notna()
    if int(usable.sum()) < config.minimum_training_rows:
        raise RegimeChallengerError(
            f"Entrainement insuffisant: {int(usable.sum())} lignes solaires < "
            f"{config.minimum_training_rows}."
        )
    x, medians = _model_matrix(features.loc[usable])
    error = (aligned_actual - aligned_baseline).loc[usable].astype(float)
    label = error.ge(config.shock_error_threshold_eur_mwh).astype(int)
    positives = int(label.sum())
    probability = float(label.mean())
    classifier: ExtraTreesClassifier | None = None
    regressor: ExtraTreesRegressor | None = None
    classifier_importance = np.zeros(x.shape[1], dtype=float)
    regressor_importance = np.zeros(x.shape[1], dtype=float)
    if label.nunique() == 2 and positives >= config.minimum_positive_rows:
        classifier = ExtraTreesClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.classifier_min_samples_leaf,
            class_weight="balanced",
            max_features="sqrt",
            n_jobs=model_jobs,
            random_state=config.random_state,
        )
        classifier.fit(x, label)
        classifier_importance = classifier.feature_importances_.astype(float)
        positive_mask = label.eq(1)
        regressor = ExtraTreesRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.regressor_min_samples_leaf,
            max_features=0.8,
            n_jobs=model_jobs,
            random_state=config.random_state + 1,
        )
        regressor.fit(x.loc[positive_mask], error.loc[positive_mask])
        regressor_importance = regressor.feature_importances_.astype(float)
    positive_errors = error.loc[label.eq(1)]
    constant_magnitude = float(
        positive_errors.median()
        if len(positive_errors)
        else config.shock_error_threshold_eur_mwh
    )
    combined = 0.6 * classifier_importance + 0.4 * regressor_importance
    if float(combined.sum()) > 0:
        combined /= float(combined.sum())
    importance = pd.DataFrame(
        {"feature": x.columns.astype(str), "importance": combined}
    ).sort_values("importance", ascending=False, kind="stable")
    diagnostics = {
        "training_rows_all_hours": int(len(features)),
        "training_rows_solar": int(usable.sum()),
        "positive_rows": positives,
        "positive_rate": probability,
        "classifier_status": "fitted" if classifier is not None else "constant_fallback",
        "regressor_status": "fitted" if regressor is not None else "median_fallback",
        "training_start_utc": features.index[usable][0],
        "training_end_utc": features.index[usable][-1],
        "shock_threshold_eur_mwh": config.shock_error_threshold_eur_mwh,
        "model_jobs": model_jobs,
        "n_features": int(len(x.columns)),
        "feature_names": list(x.columns.astype(str)),
    }
    return FittedRegimeModel(
        feature_names=tuple(x.columns.astype(str)),
        medians=medians,
        classifier=classifier,
        regressor=regressor,
        constant_probability=probability,
        constant_magnitude=constant_magnitude,
        feature_importance=importance.reset_index(drop=True),
        diagnostics=diagnostics,
    )


def predict_regime_adjustment(
    model: FittedRegimeModel,
    features: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    config: RegimeChallengerConfig,
) -> pd.DataFrame:
    """Predict a gated upward premium and coherent adjusted quantiles."""

    x, _ = _model_matrix(
        features, feature_names=model.feature_names, medians=model.medians
    )
    if model.classifier is None:
        probability = np.full(len(x), model.constant_probability, dtype=float)
    else:
        positive_index = int(np.flatnonzero(model.classifier.classes_ == 1)[0])
        probability = model.classifier.predict_proba(x)[:, positive_index]
    if model.regressor is None:
        magnitude = np.full(len(x), model.constant_magnitude, dtype=float)
    else:
        magnitude = model.regressor.predict(x)
    solar = features["is_solar_hour"].eq(1.0).to_numpy()
    active = (probability >= config.gate_probability_threshold) & solar
    premium = np.where(active, probability * np.maximum(magnitude, 0.0), 0.0)
    premium = np.clip(premium, 0.0, config.max_premium_eur_mwh)
    result = pd.DataFrame(index=features.index)
    result["shock_probability"] = np.where(solar, probability, 0.0)
    result["shock_magnitude_if_active"] = np.where(
        solar, np.maximum(magnitude, 0.0), 0.0
    )
    result["shock_premium"] = premium
    result["regime_predicted"] = active.astype(int)
    for quantile, multiplier in (
        ("q10", config.q10_premium_multiplier),
        ("q50", config.q50_premium_multiplier),
        ("q90", config.q90_premium_multiplier),
    ):
        result[f"baseline_{quantile}"] = pd.to_numeric(
            baseline[quantile].reindex(features.index), errors="coerce"
        )
        result[f"challenger_{quantile}"] = (
            result[f"baseline_{quantile}"] + multiplier * premium
        )
    ordered = result[["challenger_q10", "challenger_q50", "challenger_q90"]]
    if bool((ordered.diff(axis=1).iloc[:, 1:] < -1e-9).any().any()):
        raise AssertionError("Le correctif challenger a croise les quantiles.")
    result.index.name = "delivery_start_utc"
    return result


def prequential_regime_predictions(
    features: pd.DataFrame,
    actual: pd.Series,
    baseline: pd.DataFrame,
    *,
    config: RegimeChallengerConfig,
    thread_count: int = 1,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Generate expanding-window evaluation predictions with a D-2 embargo."""

    local_dates = pd.Series(
        features.index.tz_convert(config.timezone).date,
        index=features.index,
    )
    valid = actual.reindex(features.index).notna() & baseline["q50"].notna()
    days = sorted(set(local_dates.loc[valid]))
    if len(days) <= config.minimum_training_days:
        raise RegimeChallengerError(
            "Historique insuffisant pour une evaluation prequentielle."
        )
    first_by_training = days[config.minimum_training_days]
    first_by_window = days[max(0, len(days) - config.evaluation_days)]
    first_evaluation = max(first_by_training, first_by_window)
    evaluation_days = [day for day in days if day >= first_evaluation]
    pieces: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    position = 0
    while position < len(evaluation_days):
        block_days = evaluation_days[position : position + config.refit_every_days]
        block_start = block_days[0]
        training_end = block_start - timedelta(days=config.label_delay_days)
        train_mask = valid & local_dates.le(training_end)
        predict_mask = local_dates.isin(block_days) & valid
        if not bool(predict_mask.any()):
            position += config.refit_every_days
            continue
        fitted = fit_regime_model(
            features.loc[train_mask],
            actual.loc[train_mask],
            baseline.loc[train_mask, "q50"],
            config=config,
            thread_count=thread_count,
        )
        predicted = predict_regime_adjustment(
            fitted,
            features.loc[predict_mask],
            baseline.loc[predict_mask],
            config=config,
        )
        predicted["actual"] = pd.to_numeric(actual.loc[predict_mask], errors="coerce")
        predicted["local_date"] = local_dates.loc[predict_mask].astype(str)
        predicted["forecast_origin_utc"] = [
            scheduled_origin_utc(day, timezone=config.timezone)
            for day in local_dates.loc[predict_mask]
        ]
        pieces.append(predicted)
        audits.append(
            {
                "block_start_day": block_days[0],
                "block_end_day": block_days[-1],
                "training_end_day": training_end,
                "training_rows": int(train_mask.sum()),
                "prediction_rows": int(predict_mask.sum()),
                **dict(fitted.diagnostics),
            }
        )
        position += config.refit_every_days
    if not pieces:
        raise RegimeChallengerError("Aucune prediction prequentielle produite.")
    result = pd.concat(pieces).sort_index()
    if result.index.has_duplicates:
        raise AssertionError("Evaluation prequentielle dupliquee.")
    return result, audits


def _pinball(actual: np.ndarray, forecast: np.ndarray, quantile: float) -> float:
    error = actual - forecast
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def summarize_metrics(
    hourly: pd.DataFrame,
    *,
    probability_threshold: float,
    solar_hours: Sequence[int] = tuple(range(9, 17)),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute paired point, quantile and regime statistics by scope/variant."""

    required = {
        "zone",
        "variant",
        "local_date",
        "local_hour",
        "actual",
        "baseline_q10",
        "baseline_q50",
        "baseline_q90",
        "challenger_q10",
        "challenger_q50",
        "challenger_q90",
        "shock_probability",
        "regime_label",
    }
    missing = sorted(required.difference(hourly.columns))
    if missing:
        raise RegimeChallengerError(f"Evaluation incomplete: {missing}.")

    metric_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    for (zone, variant), group in hourly.groupby(["zone", "variant"], sort=True):
        scopes = {
            "all": pd.Series(True, index=group.index),
            "solar_hours": group["local_hour"].isin(solar_hours),
            "actual_shock": group["regime_label"].eq(1),
        }
        for scope, scope_mask in scopes.items():
            selected = group.loc[scope_mask].copy()
            finite = selected[
                ["actual", "baseline_q50", "challenger_q50"]
            ].notna().all(axis=1)
            selected = selected.loc[finite]
            if selected.empty:
                continue
            actual_values = selected["actual"].to_numpy(float)
            baseline_values = selected["baseline_q50"].to_numpy(float)
            challenger_values = selected["challenger_q50"].to_numpy(float)
            baseline_error = baseline_values - actual_values
            challenger_error = challenger_values - actual_values
            labels = selected["regime_label"].astype(int).to_numpy()
            predicted = (
                selected["shock_probability"].to_numpy(float)
                >= probability_threshold
            ).astype(int)
            probabilities = selected["shock_probability"].to_numpy(float)
            true_positive = int(((labels == 1) & (predicted == 1)).sum())
            predicted_positive = int((predicted == 1).sum())
            actual_positive = int((labels == 1).sum())
            row: dict[str, Any] = {
                "zone": zone,
                "variant": variant,
                "scope": scope,
                "n_hours": int(len(selected)),
                "baseline_mae": float(np.mean(np.abs(baseline_error))),
                "challenger_mae": float(np.mean(np.abs(challenger_error))),
                "mae_gain": float(
                    np.mean(np.abs(baseline_error))
                    - np.mean(np.abs(challenger_error))
                ),
                "baseline_rmse": float(np.sqrt(np.mean(np.square(baseline_error)))),
                "challenger_rmse": float(
                    np.sqrt(np.mean(np.square(challenger_error)))
                ),
                "baseline_bias": float(np.mean(baseline_error)),
                "challenger_bias": float(np.mean(challenger_error)),
                "challenger_win_rate": float(
                    np.mean(np.abs(challenger_error) < np.abs(baseline_error))
                ),
                "baseline_interval_coverage": float(
                    np.mean(
                        (actual_values >= selected["baseline_q10"].to_numpy(float))
                        & (actual_values <= selected["baseline_q90"].to_numpy(float))
                    )
                ),
                "challenger_interval_coverage": float(
                    np.mean(
                        (actual_values >= selected["challenger_q10"].to_numpy(float))
                        & (actual_values <= selected["challenger_q90"].to_numpy(float))
                    )
                ),
                "brier_score": float(np.mean(np.square(probabilities - labels))),
                "regime_precision": (
                    float(true_positive / predicted_positive)
                    if predicted_positive
                    else np.nan
                ),
                "regime_recall": (
                    float(true_positive / actual_positive) if actual_positive else np.nan
                ),
                "regime_base_rate": float(labels.mean()),
            }
            for quantile, level in (("q10", 0.1), ("q50", 0.5), ("q90", 0.9)):
                row[f"baseline_pinball_{quantile}"] = _pinball(
                    actual_values, selected[f"baseline_{quantile}"].to_numpy(float), level
                )
                row[f"challenger_pinball_{quantile}"] = _pinball(
                    actual_values,
                    selected[f"challenger_{quantile}"].to_numpy(float),
                    level,
                )
            row["regime_roc_auc"] = (
                float(roc_auc_score(labels, probabilities))
                if len(np.unique(labels)) == 2
                else np.nan
            )
            metric_rows.append(row)

        for local_day, day_frame in group.groupby("local_date", sort=True):
            finite = day_frame[
                ["actual", "baseline_q50", "challenger_q50"]
            ].notna().all(axis=1)
            selected = day_frame.loc[finite]
            if selected.empty:
                continue
            baseline_abs = (selected["baseline_q50"] - selected["actual"]).abs()
            challenger_abs = (selected["challenger_q50"] - selected["actual"]).abs()
            daily_rows.append(
                {
                    "zone": zone,
                    "variant": variant,
                    "local_date": str(local_day),
                    "n_hours": int(len(selected)),
                    "baseline_mae": float(baseline_abs.mean()),
                    "challenger_mae": float(challenger_abs.mean()),
                    "mae_gain": float(baseline_abs.mean() - challenger_abs.mean()),
                    "actual_max": float(selected["actual"].max()),
                    "baseline_max": float(selected["baseline_q50"].max()),
                    "challenger_max": float(selected["challenger_q50"].max()),
                    "maximum_shock_probability": float(
                        selected["shock_probability"].max()
                    ),
                    "mean_shock_premium": float(selected["shock_premium"].mean()),
                    "actual_shock_hours": int(selected["regime_label"].sum()),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(daily_rows)


__all__ = [
    "DELIVERY_TIMEZONE",
    "FittedRegimeModel",
    "QUANTILES",
    "RESIDUAL_ALIASES",
    "RegimeChallengerConfig",
    "RegimeChallengerError",
    "build_regime_features",
    "fit_regime_model",
    "load_regime_challenger_config",
    "normalize_vintage_store",
    "predict_regime_adjustment",
    "prequential_regime_predictions",
    "previous_local_day_values",
    "scheduled_origin_utc",
    "select_forecasts_asof",
    "summarize_metrics",
]
