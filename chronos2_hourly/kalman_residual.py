"""Causal Kalman overlays for already-corrected day-ahead price forecasts.

The upstream residual corrector remains untouched.  This module estimates a
slow, reportable state from *previously realised* residuals and adds one common
shift to q10/q50/q90.  The common shift preserves interval width and quantile
ordering by construction.

The operational unit is one local delivery day.  A state is frozen before the
first hour of a day, used for all 23/24/25 physical hours, and updated only
after that day is complete.  No smoother and no EM fit are used: online
``filter_update`` is the only pykalman inference primitive in the linear and
UKF implementations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .kalman_covariates import (
    BASE_RESIDUAL_LOAD_COVARIATES,
    KalmanCovariateConfig,
    KalmanCovariateError,
    covariate_coverage,
    materialize_kalman_covariates,
)

try:
    from pykalman import AdditiveUnscentedKalmanFilter, KalmanFilter
except ImportError as exc:  # pragma: no cover - exercised by deployment checks.
    AdditiveUnscentedKalmanFilter = None  # type: ignore[assignment]
    KalmanFilter = None  # type: ignore[assignment]
    _PYKALMAN_IMPORT_ERROR: ImportError | None = exc
else:
    _PYKALMAN_IMPORT_ERROR = None


QUANTILES: tuple[str, str, str] = ("q10", "q50", "q90")
DEFAULT_MODEL_KEY = "residual_kalman"
REQUIRED_MARKET_COVARIATES: tuple[str, ...] = BASE_RESIDUAL_LOAD_COVARIATES
SUPPORTED_FILTER_KINDS = frozenset(
    {
        "linear_bias",
        "linear_harmonic",
        "linear_market",
        "linear_weather",
        "linear_renewables",
        "linear_fundamental",
        "linear_fuel",
        "linear_market_weather",
        "linear_market_weather_fuel",
        "linear_scale",
        "ekf_scale",
        "ukf_scale",
    }
)
EXOGENOUS_FILTER_GROUPS: Mapping[str, str] = {
    "linear_market": "market",
    "linear_weather": "weather",
    "linear_renewables": "renewables",
    "linear_fundamental": "fundamentals",
    "linear_fuel": "fuel",
    "linear_market_weather": "market_weather",
    "linear_market_weather_fuel": "market_weather_fuel",
}
_COMMON_EXOGENOUS_FEATURES: tuple[str, ...] = (
    "base_level",
    "interval_width",
    "residual_shift",
)
ROLLING_REFIT_CACHE_SCHEMA_VERSION = 1
_ROLLING_REFIT_CACHE_REQUIRED_FIT_KEYS = frozenset(
    {
        "predictions",
        "candidate_predictions",
        "daily_audit",
        "state_audit",
        "window_audit",
        "market_scalers",
        "observation_variance",
        "state_before_transition",
        "state_after_transition",
        "candidate_feature_counts",
        "target_clipped_shifts",
        "training_clipped_shifts",
        "governance_realised_days",
        "governance_realised_hours",
    }
)


class KalmanResidualError(ValueError):
    """Raised when the causal Kalman contract is violated."""


@dataclass(frozen=True)
class KalmanResidualConfig:
    """Frozen replay/live policy shared by every zone."""

    q_over_r: float = 0.001
    scale_q_over_r: float = 0.0001
    innovation_clip_sigma: float = 3.0
    shift_clip_eur_mwh: float = 20.0
    governance_lookback_days: int = 60
    governance_minimum_days: int = 14
    governance_confirmation_days: int = 0
    governance_weight_step: float = 0.05
    minimum_gain_eur_mwh: float = 0.05
    minimum_relative_gain: float = 0.005
    scale_minimum: float = 0.5
    scale_maximum: float = 1.5
    ukf_scale_persistence: float = 0.99
    market_feature_clip: float = 5.0
    candidate_kinds: tuple[str, ...] = (
        "linear_bias",
        "linear_harmonic",
        "linear_market",
        "linear_scale",
        "ukf_scale",
    )

    def validate(self) -> None:
        numeric_positive = {
            "q_over_r": self.q_over_r,
            "scale_q_over_r": self.scale_q_over_r,
            "innovation_clip_sigma": self.innovation_clip_sigma,
            "shift_clip_eur_mwh": self.shift_clip_eur_mwh,
            "governance_weight_step": self.governance_weight_step,
            "minimum_gain_eur_mwh": self.minimum_gain_eur_mwh,
            "market_feature_clip": self.market_feature_clip,
        }
        for name, value in numeric_positive.items():
            if not np.isfinite(value) or float(value) <= 0.0:
                raise KalmanResidualError(f"{name} doit etre fini et > 0.")
        if not 0.0 < self.governance_weight_step <= 1.0:
            raise KalmanResidualError(
                "governance_weight_step doit appartenir a ]0, 1]."
            )
        if self.governance_lookback_days < 1:
            raise KalmanResidualError(
                "governance_lookback_days doit etre strictement positif."
            )
        if not 1 <= self.governance_minimum_days <= self.governance_lookback_days:
            raise KalmanResidualError(
                "governance_minimum_days doit etre compris dans le lookback."
            )
        confirmation_raw = self.governance_confirmation_days
        if (
            isinstance(confirmation_raw, (bool, np.bool_))
            or not isinstance(
                confirmation_raw,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(float(confirmation_raw))
            or int(confirmation_raw) != float(confirmation_raw)
        ):
            raise KalmanResidualError(
                "governance_confirmation_days doit etre un entier."
            )
        confirmation_days = int(confirmation_raw)
        if not 0 <= confirmation_days < self.governance_lookback_days:
            raise KalmanResidualError(
                "governance_confirmation_days doit appartenir a [0, lookback[."
            )
        if confirmation_days >= self.governance_minimum_days:
            raise KalmanResidualError(
                "governance_confirmation_days doit etre strictement inferieur "
                "a governance_minimum_days afin de conserver une fenetre de "
                "selection non vide."
            )
        if not 0.0 <= self.minimum_relative_gain < 1.0:
            raise KalmanResidualError(
                "minimum_relative_gain doit appartenir a [0, 1[."
            )
        if not 0.0 < self.scale_minimum < 1.0 < self.scale_maximum:
            raise KalmanResidualError(
                "Les bornes d'echelle doivent encadrer strictement 1."
            )
        if not 0.0 < self.ukf_scale_persistence <= 1.0:
            raise KalmanResidualError(
                "ukf_scale_persistence doit appartenir a ]0, 1]."
            )
        kinds = tuple(map(str, self.candidate_kinds))
        if not kinds or len(kinds) != len(set(kinds)):
            raise KalmanResidualError(
                "candidate_kinds doit contenir des valeurs uniques non vides."
            )
        unknown = sorted(set(kinds).difference(SUPPORTED_FILTER_KINDS))
        if unknown:
            raise KalmanResidualError(f"Filtres Kalman non supportes: {unknown}.")


def _validate_training_lookback_days(value: int | None) -> int | None:
    """Validate the optional rolling-refit policy without changing YAML ABI."""

    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
    ):
        raise KalmanResidualError(
            "training_lookback_days doit etre un entier strictement positif "
            "ou null."
        )
    return int(value)


def _validate_rolling_refit_workers(value: int) -> int:
    """Validate bounded lab parallelism without changing model semantics."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not 1 <= int(value) <= 8
    ):
        raise KalmanResidualError(
            "rolling_refit_workers doit etre un entier compris entre 1 et 8."
        )
    return int(value)


@dataclass(frozen=True)
class KalmanReplayResult:
    predictions: pd.DataFrame
    candidate_predictions: pd.DataFrame
    daily_audit: pd.DataFrame
    state_audit: pd.DataFrame
    audit: Mapping[str, Any]
    future_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    future_candidate_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    future_audit: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KalmanOperationalView:
    """Causal FINAL365 view ready for an autonomous live export."""

    replay: KalmanReplayResult
    statistics: pd.DataFrame
    backtest: pd.DataFrame
    forecast: pd.DataFrame
    evaluation_start_day: date
    evaluation_end_day: date
    evaluation_index: pd.DatetimeIndex
    future_index: pd.DatetimeIndex


def _rolling_cache_frame_digest(frame: pd.DataFrame) -> str:
    """Return a deterministic content digest for one physical local-day block."""

    descriptor = {
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "index_dtype": str(frame.index.dtype),
        "rows": int(len(frame)),
    }
    row_hashes = pd.util.hash_pandas_object(
        frame,
        index=True,
        categorize=False,
    ).to_numpy(dtype=np.uint64, copy=False)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            descriptor,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(row_hashes.tobytes(order="C"))
    return digest.hexdigest()


def _rolling_cache_target_digest(frame: pd.DataFrame) -> str:
    """Hash only information legally available when forecasting the target day."""

    # Historical normalisation adds the realised target and its derived error;
    # future normalisation deliberately does not. Neither value is read by
    # _fit_rolling_target_day for the frozen target. Excluding them allows the
    # J+1 fit cached yesterday to become today's identical historical fit.
    target = frame.drop(columns=["actual", "_upstream_error"], errors="ignore")
    return _rolling_cache_frame_digest(target)


def _rolling_cache_contract_digest(
    *,
    timezone: str,
    upstream_model: str,
    output_model: str,
    lookback_days: int,
    config: KalmanResidualConfig,
    covariate_columns: Sequence[str],
    candidate_feature_columns: Mapping[str, Sequence[str]],
) -> str:
    """Fingerprint every non-data input capable of changing a rolling fit."""

    try:
        pykalman_version = metadata.version("pykalman")
    except metadata.PackageNotFoundError:  # pragma: no cover - guarded earlier.
        pykalman_version = "missing"
    module_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    contract = {
        "schema_version": ROLLING_REFIT_CACHE_SCHEMA_VERSION,
        "module_sha256": module_sha256,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "pykalman_version": pykalman_version,
        "timezone": timezone,
        "upstream_model": upstream_model,
        "output_model": output_model,
        "lookback_days": int(lookback_days),
        "config": asdict(config),
        "covariate_columns": list(map(str, covariate_columns)),
        "candidate_feature_columns": {
            str(kind): list(map(str, columns))
            for kind, columns in sorted(candidate_feature_columns.items())
        },
    }
    return hashlib.sha256(
        json.dumps(
            contract,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _rolling_cache_input_digest(
    *,
    contract_digest: str,
    target_day: date,
    training_days: Sequence[date],
    block_digests: Mapping[date, str],
    target_digest: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(contract_digest.encode("ascii"))
    digest.update(target_day.isoformat().encode("ascii"))
    for training_day in training_days:
        digest.update(training_day.isoformat().encode("ascii"))
        digest.update(block_digests[training_day].encode("ascii"))
    digest.update(target_digest.encode("ascii"))
    return digest.hexdigest()


def _rolling_cache_contract_dir(cache_dir: Path, contract_digest: str) -> Path:
    """Isolate configurations so concurrent lab trials never evict each other."""

    return (
        cache_dir
        / f"schema-{ROLLING_REFIT_CACHE_SCHEMA_VERSION}"
        / contract_digest
    )


def _rolling_cache_path(
    cache_dir: Path,
    contract_digest: str,
    target_day: date,
) -> Path:
    return _rolling_cache_contract_dir(
        cache_dir,
        contract_digest,
    ) / f"{target_day.isoformat()}.pickle"


def _valid_cached_rolling_fit(
    fit: object,
    *,
    target_day: date,
    target_index: pd.DatetimeIndex,
    output_model: str,
    candidate_kinds: Sequence[str],
) -> bool:
    if not isinstance(fit, Mapping):
        return False
    if not _ROLLING_REFIT_CACHE_REQUIRED_FIT_KEYS.issubset(fit):
        return False
    predictions = fit.get("predictions")
    candidate_predictions = fit.get("candidate_predictions")
    if not isinstance(predictions, pd.DataFrame) or not isinstance(
        candidate_predictions, pd.DataFrame
    ):
        return False
    if not predictions.index.equals(target_index) or not candidate_predictions.index.equals(
        target_index
    ):
        return False
    expected_predictions = {
        *(f"{output_model}__{quantile}" for quantile in QUANTILES),
        "kalman_raw_correction",
        "kalman_weight",
        "kalman_correction",
        "kalman_selected_filter",
    }
    expected_candidates = {
        f"{kind}__q50" for kind in candidate_kinds
    }
    if set(predictions.columns) != expected_predictions:
        return False
    if set(candidate_predictions.columns) != expected_candidates:
        return False
    daily_audit = fit.get("daily_audit")
    window_audit = fit.get("window_audit")
    if not isinstance(daily_audit, Mapping) or not isinstance(window_audit, Mapping):
        return False
    return (
        str(daily_audit.get("local_day")) == target_day.isoformat()
        and str(window_audit.get("target_day")) == target_day.isoformat()
    )


def _load_cached_rolling_fit(
    path: Path,
    *,
    input_digest: str,
    target_day: date,
    target_index: pd.DatetimeIndex,
    output_model: str,
    candidate_kinds: Sequence[str],
) -> tuple[Mapping[str, Any] | None, str]:
    if not path.is_file():
        return None, "missing"
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
    except Exception:
        # Cache files are disposable. This also covers objects made unreadable
        # by an interrupted dependency upgrade (AttributeError/ImportError).
        return None, "invalid"
    if not isinstance(payload, Mapping):
        return None, "invalid"
    if (
        payload.get("schema_version") != ROLLING_REFIT_CACHE_SCHEMA_VERSION
        or payload.get("input_digest") != input_digest
    ):
        return None, "stale"
    fit = payload.get("fit")
    if not _valid_cached_rolling_fit(
        fit,
        target_day=target_day,
        target_index=target_index,
        output_model=output_model,
        candidate_kinds=candidate_kinds,
    ):
        return None, "invalid"
    return fit, "hit"  # type: ignore[return-value]


def _write_cached_rolling_fit(
    path: Path,
    *,
    input_digest: str,
    target_day: date,
    fit: Mapping[str, Any],
) -> None:
    """Atomically publish a disposable cache entry; model output never depends on it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            pickle.dump(
                {
                    "schema_version": ROLLING_REFIT_CACHE_SCHEMA_VERSION,
                    "target_day": target_day.isoformat(),
                    "input_digest": input_digest,
                    "fit": fit,
                },
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_pykalman() -> None:
    if _PYKALMAN_IMPORT_ERROR is not None:
        raise RuntimeError(
            "pykalman 0.11.2 est requis pour la couche Kalman. Installez "
            "requirements_hourly.txt."
        ) from _PYKALMAN_IMPORT_ERROR


def _timestamped_frame(
    raw: pd.DataFrame,
    *,
    timestamp_column: str,
    name: str,
) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise KalmanResidualError(f"{name} doit etre un DataFrame non vide.")
    if timestamp_column not in raw:
        raise KalmanResidualError(f"{name}: {timestamp_column} est absent.")
    frame = raw.copy()
    index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop(timestamp_column), utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise KalmanResidualError(f"{name}: timeline UTC invalide.")
    if bool((index.minute != 0).any()) or bool((index.second != 0).any()):
        raise KalmanResidualError(f"{name}: timestamps non alignes sur l'heure.")
    frame.index = index
    return frame


def _normalise_input(
    history: pd.DataFrame,
    *,
    upstream_model: str,
    timezone: str,
    covariates: pd.DataFrame | None,
    covariate_config: KalmanCovariateConfig,
) -> tuple[pd.DataFrame, tuple[str, ...], Mapping[str, Any]]:
    frame = _timestamped_frame(
        history,
        timestamp_column="delivery_start_utc",
        name="history",
    )
    required = {
        "actual",
        *(f"{upstream_model}__{quantile}" for quantile in QUANTILES),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KalmanResidualError(f"Colonnes upstream absentes: {missing}.")
    numeric_columns = sorted(required)
    for optional in (
        "chronos2__q50",
        "residual_correction",
        "forecast_origin_utc",
    ):
        if optional in frame:
            numeric_columns.append(optional)
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    finite_required = np.isfinite(
        frame.loc[:, sorted(required)].to_numpy(dtype=float)
    ).all(axis=1)
    frame = frame.loc[finite_required].copy()
    if frame.empty:
        raise KalmanResidualError("Aucune ligne finie pour le replay Kalman.")

    q10 = frame[f"{upstream_model}__q10"].to_numpy(dtype=float)
    q50 = frame[f"{upstream_model}__q50"].to_numpy(dtype=float)
    q90 = frame[f"{upstream_model}__q90"].to_numpy(dtype=float)
    if bool(((q10 > q50) | (q50 > q90)).any()):
        raise KalmanResidualError("Les quantiles upstream se croisent.")
    local = frame.index.tz_convert(timezone)
    frame["_local_day"] = local.date
    frame["_hour_sin"] = np.sin(2.0 * np.pi * local.hour / 24.0)
    frame["_hour_cos"] = np.cos(2.0 * np.pi * local.hour / 24.0)
    frame["_hour_sin2"] = np.sin(4.0 * np.pi * local.hour / 24.0)
    frame["_hour_cos2"] = np.cos(4.0 * np.pi * local.hour / 24.0)
    frame["_base_q50"] = frame[f"{upstream_model}__q50"].astype(float)
    frame["_interval_width"] = (
        frame[f"{upstream_model}__q90"] - frame[f"{upstream_model}__q10"]
    ).astype(float)
    frame["_upstream_error"] = (
        frame["actual"] - frame[f"{upstream_model}__q50"]
    ).astype(float)

    if "residual_correction" in frame:
        frame["_residual_shift"] = pd.to_numeric(
            frame["residual_correction"], errors="coerce"
        )
    elif "chronos2__q50" in frame:
        frame["_residual_shift"] = (
            frame[f"{upstream_model}__q50"]
            - pd.to_numeric(frame["chronos2__q50"], errors="coerce")
        )
    else:
        frame["_residual_shift"] = 0.0
    frame["_pre_residual_q50"] = frame["_base_q50"] - frame[
        "_residual_shift"
    ].fillna(0.0)

    covariate_columns: tuple[str, ...] = ()
    covariate_audit: dict[str, Any] = {
        "coverage": [],
        "structural_ramp_hours": {},
    }
    if covariates is not None:
        cov = _timestamped_frame(
            covariates,
            timestamp_column="timestamp",
            name="covariates",
        )
        try:
            materialized = materialize_kalman_covariates(
                cov,
                covariate_config,
                timezone=timezone,
            )
        except KalmanCovariateError as exc:
            raise KalmanResidualError(str(exc)) from exc
        selected = list(covariate_config.feature_columns)
        numeric_cov = materialized.loc[:, selected]
        frame = frame.join(numeric_cov, how="left", validate="one_to_one")
        covariate_columns = tuple(selected)
        history_covariates = materialized.reindex(frame.index)
        coverage_frame = covariate_coverage(
            history_covariates,
            covariate_config,
        )
        covariate_audit = {
            "coverage": coverage_frame.to_dict(orient="records"),
            "structural_ramp_hours": dict(
                materialized.attrs.get("structural_ramp_hours", {})
            ),
        }
        feature_coverage = coverage_frame.loc[
            coverage_frame["column"].isin(selected),
            ["column", "coverage"],
        ]
        insufficient = feature_coverage.loc[
            feature_coverage["coverage"]
            < float(covariate_config.minimum_history_coverage)
        ]
        if not insufficient.empty:
            detail = ", ".join(
                f"{row.column}={float(row.coverage):.3f}"
                for row in insufficient.itertuples(index=False)
            )
            raise KalmanResidualError(
                "Couverture historique Kalman insuffisante: " + detail
            )
        if covariate_config.history_missing_policy == "complete_trailing":
            historical_values = frame.loc[:, selected].to_numpy(dtype=float)
            if not np.isfinite(historical_values).all():
                incomplete = [
                    str(column)
                    for column in selected
                    if not np.isfinite(
                        pd.to_numeric(
                            frame[column], errors="coerce"
                        ).to_numpy(dtype=float)
                    ).all()
                ]
                raise KalmanResidualError(
                    "La politique complete_trailing exige que le chargeur "
                    f"fournisse un suffixe historique complet: {incomplete}."
                )
    return frame, covariate_columns, covariate_audit


def _expected_local_day_index(
    local_day: object,
    *,
    timezone: str,
) -> pd.DatetimeIndex:
    start = pd.Timestamp(local_day).tz_localize(timezone)
    end = start + pd.DateOffset(days=1)
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def _normalise_future_input(
    future_upstream: pd.DataFrame,
    *,
    upstream_model: str,
    timezone: str,
    covariates: pd.DataFrame | None,
    covariate_columns: Sequence[str],
    covariate_config: KalmanCovariateConfig,
) -> pd.DataFrame:
    """Normalise one complete future day without ever reading an actual."""

    frame = _timestamped_frame(
        future_upstream,
        timestamp_column="delivery_start_utc",
        name="future_upstream",
    )
    required = [f"{upstream_model}__{quantile}" for quantile in QUANTILES]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise KalmanResidualError(
            f"Colonnes upstream futures absentes: {missing}."
        )
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame.loc[:, required].to_numpy(dtype=float)).all():
        raise KalmanResidualError(
            "La journee future contient des quantiles upstream non finis."
        )
    q10 = frame[f"{upstream_model}__q10"].to_numpy(dtype=float)
    q50 = frame[f"{upstream_model}__q50"].to_numpy(dtype=float)
    q90 = frame[f"{upstream_model}__q90"].to_numpy(dtype=float)
    if bool(((q10 > q50) | (q50 > q90)).any()):
        raise KalmanResidualError("Les quantiles upstream futurs se croisent.")

    local = frame.index.tz_convert(timezone)
    local_days = pd.Index(local.date).unique()
    if len(local_days) != 1:
        raise KalmanResidualError(
            "future_upstream doit contenir exactement une journee locale."
        )
    expected_index = _expected_local_day_index(local_days[0], timezone=timezone)
    if not frame.index.equals(expected_index):
        raise KalmanResidualError(
            "future_upstream doit contenir toutes les heures physiques de la "
            "journee locale (23, 24 ou 25)."
        )

    frame["_local_day"] = local.date
    frame["_hour_sin"] = np.sin(2.0 * np.pi * local.hour / 24.0)
    frame["_hour_cos"] = np.cos(2.0 * np.pi * local.hour / 24.0)
    frame["_hour_sin2"] = np.sin(4.0 * np.pi * local.hour / 24.0)
    frame["_hour_cos2"] = np.cos(4.0 * np.pi * local.hour / 24.0)
    frame["_base_q50"] = frame[f"{upstream_model}__q50"].astype(float)
    frame["_interval_width"] = (
        frame[f"{upstream_model}__q90"] - frame[f"{upstream_model}__q10"]
    ).astype(float)
    if "residual_correction" in frame:
        frame["_residual_shift"] = pd.to_numeric(
            frame["residual_correction"], errors="coerce"
        )
    elif "chronos2__q50" in frame:
        frame["_residual_shift"] = (
            frame[f"{upstream_model}__q50"]
            - pd.to_numeric(frame["chronos2__q50"], errors="coerce")
        )
    else:
        frame["_residual_shift"] = 0.0
    frame["_residual_shift"] = frame["_residual_shift"].fillna(0.0)
    frame["_pre_residual_q50"] = (
        frame["_base_q50"] - frame["_residual_shift"]
    )

    if covariate_columns:
        if covariates is not None:
            cov = _timestamped_frame(
                covariates,
                timestamp_column="timestamp",
                name="future_covariates",
            )
            try:
                numeric_cov = materialize_kalman_covariates(
                    cov,
                    covariate_config,
                    timezone=timezone,
                ).loc[:, list(covariate_columns)]
            except KalmanCovariateError as exc:
                raise KalmanResidualError(str(exc)) from exc
            frame = frame.join(numeric_cov, how="left", validate="one_to_one")
        for column in covariate_columns:
            if column not in frame:
                frame[str(column)] = np.nan
        if covariate_config.require_future_complete:
            future_values = frame.loc[:, list(covariate_columns)].to_numpy(
                dtype=float
            )
            if not np.isfinite(future_values).all():
                missing = [
                    str(column)
                    for column in covariate_columns
                    if not np.isfinite(
                        pd.to_numeric(frame[column], errors="coerce").to_numpy(
                            dtype=float
                        )
                    ).all()
                ]
                raise KalmanResidualError(
                    "Covariables Kalman futures incompletes: "
                    f"{missing}."
                )
    return frame


def _robust_location_scale(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size == 0:
        return 0.0, 1.0
    location = float(np.median(numeric))
    mad = float(np.median(np.abs(numeric - location)))
    scale = max(1.4826 * mad, float(np.std(numeric)), 1e-6)
    return location, scale


def _market_features(
    frame: pd.DataFrame,
    *,
    covariate_columns: Sequence[str],
    calibration_days: set[object],
    clip: float,
) -> tuple[pd.DataFrame, Mapping[str, Mapping[str, float]]]:
    raw = pd.DataFrame(index=frame.index)
    raw["base_level"] = frame["_base_q50"]
    raw["interval_width"] = frame["_interval_width"]
    raw["residual_shift"] = frame["_residual_shift"]
    for column in covariate_columns:
        raw[f"covariate::{column}"] = frame[column]

    calibration_mask = frame["_local_day"].isin(calibration_days)
    standardised = pd.DataFrame(index=frame.index)
    audit: dict[str, Mapping[str, float]] = {}
    for column in raw:
        location, scale = _robust_location_scale(raw.loc[calibration_mask, column])
        values = (pd.to_numeric(raw[column], errors="coerce") - location) / scale
        standardised[column] = values.clip(-float(clip), float(clip)).fillna(0.0)
        audit[column] = {"location": location, "scale": scale}
    return standardised, audit


def _candidate_market_feature_columns(
    kind: str,
    *,
    covariate_config: KalmanCovariateConfig,
    covariate_columns: Sequence[str],
) -> tuple[str, ...]:
    """Resolve the exact standardised features seen by one candidate."""

    if kind not in EXOGENOUS_FILTER_GROUPS:
        return ()
    group_name = EXOGENOUS_FILTER_GROUPS[kind]
    configured = tuple(covariate_config.groups.get(group_name, ()))
    available = set(map(str, covariate_columns))
    # Historical API compatibility: linear_market was usable without an
    # external covariate frame and still saw the three upstream descriptors.
    if not available and kind == "linear_market":
        return _COMMON_EXOGENOUS_FEATURES
    if not configured:
        raise KalmanResidualError(
            f"Le filtre {kind} exige le groupe de covariables {group_name}."
        )
    missing = sorted(set(configured).difference(available))
    if missing:
        raise KalmanResidualError(
            f"Le filtre {kind} exige des covariables absentes: {missing}."
        )
    return (
        *_COMMON_EXOGENOUS_FEATURES,
        *(f"covariate::{column}" for column in configured),
    )


def _observation_variance(
    frame: pd.DataFrame,
    *,
    calibration_days: set[object],
) -> float:
    values = frame.loc[frame["_local_day"].isin(calibration_days), "_upstream_error"]
    location, scale = _robust_location_scale(values)
    del location
    return max(float(scale * scale), 1.0)


def _sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    clipped = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class _Candidate:
    def __init__(
        self,
        *,
        kind: str,
        config: KalmanResidualConfig,
        observation_variance: float,
        market: pd.DataFrame,
        market_feature_columns: Sequence[str] = (),
    ) -> None:
        _require_pykalman()
        self.kind = kind
        self.config = config
        self.r = float(observation_variance)
        self.market = market
        self.minimum_eigenvalue = float("inf")
        self.covariance_repairs = 0
        self.innovation_clips = 0
        self.market_feature_columns = tuple(map(str, market_feature_columns))
        if kind == "linear_bias":
            self.feature_names = ("intercept",)
            initial = np.zeros(1, dtype=float)
        elif kind == "linear_harmonic":
            self.feature_names = ("intercept", "hour_sin", "hour_cos")
            initial = np.zeros(3, dtype=float)
        elif kind in EXOGENOUS_FILTER_GROUPS:
            self.feature_names = (
                "intercept",
                "hour_sin",
                "hour_cos",
                *self.market_feature_columns,
            )
            initial = np.zeros(len(self.feature_names), dtype=float)
        elif kind == "linear_scale":
            self.feature_names = ("bias", "residual_scale")
            initial = np.asarray([0.0, 1.0], dtype=float)
        elif kind in {"ekf_scale", "ukf_scale"}:
            self.feature_names = ("bias", "bounded_scale_logit")
            initial = np.asarray([0.0, 0.0], dtype=float)
        else:  # pragma: no cover - guarded by config validation.
            raise KalmanResidualError(f"Filtre inconnu: {kind}.")
        self.mean = initial
        dimension = len(initial)
        self.covariance = np.eye(dimension, dtype=float) * self.r
        if kind in {"linear_scale", "ekf_scale", "ukf_scale"}:
            self.covariance[1, 1] = 0.05**2
        self.identity = np.eye(dimension, dtype=float)
        self.zero_q = np.zeros((dimension, dimension), dtype=float)
        self.transition = self.identity.copy()
        if kind in {"ekf_scale", "ukf_scale"}:
            self.transition[1, 1] = config.ukf_scale_persistence
        self.q = np.eye(dimension, dtype=float) * (
            config.q_over_r * self.r
        )
        if kind in {"linear_scale", "ekf_scale", "ukf_scale"}:
            self.q[1, 1] = config.scale_q_over_r * 0.05**2
        self.linear_filter = KalmanFilter(
            transition_matrices=self.identity,
            observation_matrices=np.zeros((1, dimension)),
            transition_covariance=self.zero_q,
            observation_covariance=np.asarray([[self.r]]),
            initial_state_mean=self.mean,
            initial_state_covariance=self.covariance,
        )
        self.ukf = (
            AdditiveUnscentedKalmanFilter(
                transition_functions=lambda state: state,
                observation_functions=lambda state: np.asarray([state[0]]),
                transition_covariance=self.zero_q,
                observation_covariance=np.asarray([[self.r]]),
                initial_state_mean=self.mean,
                initial_state_covariance=self.covariance,
                random_state=0,
            )
            if kind == "ukf_scale"
            else None
        )

    def _scale(self, state: np.ndarray | None = None) -> float:
        values = self.mean if state is None else np.asarray(state, dtype=float)
        width = self.config.scale_maximum - self.config.scale_minimum
        return float(self.config.scale_minimum + width * _sigmoid(values[1]))

    def _linear_features(self, block: pd.DataFrame) -> np.ndarray:
        if self.kind == "linear_bias":
            return np.ones((len(block), 1), dtype=float)
        if self.kind == "linear_harmonic":
            return np.column_stack(
                [
                    np.ones(len(block), dtype=float),
                    block["_hour_sin"].to_numpy(dtype=float),
                    block["_hour_cos"].to_numpy(dtype=float),
                ]
            )
        if self.kind in EXOGENOUS_FILTER_GROUPS:
            return np.column_stack(
                [
                    np.ones(len(block), dtype=float),
                    block["_hour_sin"].to_numpy(dtype=float),
                    block["_hour_cos"].to_numpy(dtype=float),
                    self.market.loc[
                        block.index,
                        list(self.market_feature_columns),
                    ].to_numpy(dtype=float),
                ]
            )
        if self.kind == "linear_scale":
            return np.column_stack(
                [
                    np.ones(len(block), dtype=float),
                    block["_residual_shift"].to_numpy(dtype=float),
                ]
            )
        raise KalmanResidualError(f"Features lineaires indisponibles: {self.kind}.")

    def transition_day(self) -> None:
        self.mean = self.transition @ self.mean
        self.covariance = (
            self.transition @ self.covariance @ self.transition.T + self.q
        )
        self._repair_covariance()

    def raw_correction(self, block: pd.DataFrame) -> np.ndarray:
        if self.kind in {
            "linear_bias",
            "linear_harmonic",
            *EXOGENOUS_FILTER_GROUPS,
        }:
            return self._linear_features(block) @ self.mean
        residual_shift = block["_residual_shift"].to_numpy(dtype=float)
        if self.kind == "linear_scale":
            return self.mean[0] + (self.mean[1] - 1.0) * residual_shift
        scale = self._scale()
        return self.mean[0] + (scale - 1.0) * residual_shift

    def _predicted_observation(
        self,
        row: pd.Series,
        *,
        state: np.ndarray | None = None,
    ) -> float:
        values = self.mean if state is None else np.asarray(state, dtype=float)
        if self.kind in {
            "linear_bias",
            "linear_harmonic",
            *EXOGENOUS_FILTER_GROUPS,
        }:
            block = row.to_frame().T
            block.index = pd.DatetimeIndex([row.name])
            return float(self._linear_features(block)[0] @ values)
        residual_shift = float(row["_residual_shift"])
        if self.kind == "linear_scale":
            return float(values[0] + values[1] * residual_shift)
        return float(values[0] + self._scale(values) * residual_shift)

    def _robust_observation(self, row: pd.Series) -> float:
        target = (
            float(row["_upstream_error"])
            if self.kind in {
                "linear_bias",
                "linear_harmonic",
                *EXOGENOUS_FILTER_GROUPS,
            }
            else float(row["actual"] - row["_pre_residual_q50"])
        )
        predicted = self._predicted_observation(row)
        innovation = target - predicted
        threshold = self.config.innovation_clip_sigma * np.sqrt(self.r)
        clipped = float(np.clip(innovation, -threshold, threshold))
        if not np.isclose(clipped, innovation, rtol=0.0, atol=0.0):
            self.innovation_clips += 1
        return predicted + clipped

    def update_hour(self, row: pd.Series) -> tuple[float, float]:
        observation = self._robust_observation(row)
        predicted = self._predicted_observation(row)
        innovation = observation - predicted
        if self.kind in {
            "linear_bias",
            "linear_harmonic",
            *EXOGENOUS_FILTER_GROUPS,
            "linear_scale",
        }:
            if self.kind in {
                "linear_bias",
                "linear_harmonic",
                *EXOGENOUS_FILTER_GROUPS,
            }:
                block = row.to_frame().T
                block.index = pd.DatetimeIndex([row.name])
                observation_matrix = self._linear_features(block)
            else:
                observation_matrix = np.asarray(
                    [[1.0, float(row["_residual_shift"])]], dtype=float
                )
            self.mean, self.covariance = self.linear_filter.filter_update(
                self.mean,
                self.covariance,
                observation=np.asarray([observation]),
                transition_matrix=self.identity,
                transition_covariance=self.zero_q,
                observation_matrix=observation_matrix,
                observation_covariance=np.asarray([[self.r]]),
            )
            if self.kind == "linear_scale":
                self.mean[1] = float(
                    np.clip(
                        self.mean[1],
                        self.config.scale_minimum,
                        self.config.scale_maximum,
                    )
                )
        elif self.kind == "ekf_scale":
            residual_shift = float(row["_residual_shift"])
            probability = float(_sigmoid(self.mean[1]))
            width = self.config.scale_maximum - self.config.scale_minimum
            jacobian = np.asarray(
                [1.0, residual_shift * width * probability * (1.0 - probability)],
                dtype=float,
            )
            variance = float(jacobian @ self.covariance @ jacobian + self.r)
            gain = self.covariance @ jacobian / variance
            prior = self.covariance.copy()
            self.mean = self.mean + gain * innovation
            residual_matrix = self.identity - np.outer(gain, jacobian)
            self.covariance = (
                residual_matrix @ prior @ residual_matrix.T
                + np.outer(gain, gain) * self.r
            )
        else:
            assert self.ukf is not None
            residual_shift = float(row["_residual_shift"])

            def observe(state: np.ndarray) -> np.ndarray:
                return np.asarray(
                    [
                        state[0]
                        + self._scale(np.asarray(state, dtype=float))
                        * residual_shift
                    ],
                    dtype=float,
                )

            self.mean, self.covariance = self.ukf.filter_update(
                self.mean,
                self.covariance,
                observation=np.asarray([observation]),
                transition_function=lambda state: state,
                transition_covariance=self.zero_q,
                observation_function=observe,
                observation_covariance=np.asarray([[self.r]]),
            )
        self._repair_covariance()
        return predicted, innovation

    def update_hour_rolling(self, row: pd.Series) -> tuple[float, float]:
        """Numerically equivalent small-matrix update for repeated refits.

        Calling the generic pykalman dispatch millions of times dominates a
        365-by-365 replay.  This specialised path keeps the same equations,
        clipping and covariance repair while avoiding parameter discovery and
        temporary masked arrays on every scalar observation.  The legacy
        expanding replay continues to call :meth:`update_hour` unchanged.
        """

        linear_kinds = {
            "linear_bias",
            "linear_harmonic",
            *EXOGENOUS_FILTER_GROUPS,
            "linear_scale",
        }
        observation_vector: np.ndarray | None = None
        if self.kind in linear_kinds:
            if self.kind in {
                "linear_bias",
                "linear_harmonic",
                *EXOGENOUS_FILTER_GROUPS,
            }:
                if self.kind == "linear_bias":
                    observation_vector = np.ones(1, dtype=float)
                elif self.kind == "linear_harmonic":
                    observation_vector = np.asarray(
                        [1.0, float(row["_hour_sin"]), float(row["_hour_cos"])],
                        dtype=float,
                    )
                else:
                    observation_vector = np.concatenate(
                        [
                            np.asarray(
                                [
                                    1.0,
                                    float(row["_hour_sin"]),
                                    float(row["_hour_cos"]),
                                ],
                                dtype=float,
                            ),
                            self.market.loc[
                                row.name, list(self.market_feature_columns)
                            ].to_numpy(dtype=float),
                        ]
                    )
            else:
                observation_vector = np.asarray(
                    [1.0, float(row["_residual_shift"])], dtype=float
                )
            predicted = float(observation_vector @ self.mean)
        else:
            predicted = self._predicted_observation(row)
        target = (
            float(row["_upstream_error"])
            if self.kind in {
                "linear_bias",
                "linear_harmonic",
                *EXOGENOUS_FILTER_GROUPS,
            }
            else float(row["actual"] - row["_pre_residual_q50"])
        )
        raw_innovation = target - predicted
        threshold = self.config.innovation_clip_sigma * np.sqrt(self.r)
        innovation = float(np.clip(raw_innovation, -threshold, threshold))
        if not np.isclose(innovation, raw_innovation, rtol=0.0, atol=0.0):
            self.innovation_clips += 1
        observation = predicted + innovation
        if self.kind in linear_kinds:
            assert observation_vector is not None
            prior = self.covariance
            projected = prior @ observation_vector
            variance = float(observation_vector @ projected + self.r)
            gain = projected / variance
            self.mean = self.mean + gain * innovation
            self.covariance = prior - np.outer(gain, observation_vector @ prior)
            if self.kind == "linear_scale":
                self.mean[1] = float(
                    np.clip(
                        self.mean[1],
                        self.config.scale_minimum,
                        self.config.scale_maximum,
                    )
                )
        elif self.kind == "ekf_scale":
            residual_shift = float(row["_residual_shift"])
            probability = float(_sigmoid(self.mean[1]))
            width = self.config.scale_maximum - self.config.scale_minimum
            jacobian = np.asarray(
                [1.0, residual_shift * width * probability * (1.0 - probability)],
                dtype=float,
            )
            variance = float(jacobian @ self.covariance @ jacobian + self.r)
            gain = self.covariance @ jacobian / variance
            prior = self.covariance.copy()
            self.mean = self.mean + gain * innovation
            residual_matrix = self.identity - np.outer(gain, jacobian)
            self.covariance = (
                residual_matrix @ prior @ residual_matrix.T
                + np.outer(gain, gain) * self.r
            )
        else:
            # pykalman's additive UKF defaults are alpha=1, beta=0 and
            # kappa=3-n.  Reproduce its sigma-point correction directly for
            # this fixed two-dimensional state and scalar observation.
            dimension = len(self.mean)
            scaling = 3.0
            root = np.linalg.cholesky(self.covariance).T * np.sqrt(scaling)
            points = np.tile(self.mean, (2 * dimension + 1, 1))
            points[1 : dimension + 1] += root.T
            points[dimension + 1 :] -= root.T
            weights = np.full(2 * dimension + 1, 0.5 / scaling, dtype=float)
            weights[0] = (scaling - dimension) / scaling
            residual_shift = float(row["_residual_shift"])
            observed_points = np.asarray(
                [
                    point[0] + self._scale(point) * residual_shift
                    for point in points
                ],
                dtype=float,
            )
            observed_mean = float(weights @ observed_points)
            centered_points = points - self.mean
            centered_observations = observed_points - observed_mean
            cross = centered_points.T @ (weights * centered_observations)
            observed_variance = float(
                weights @ np.square(centered_observations) + self.r
            )
            gain = cross / observed_variance
            self.mean = self.mean + gain * (observation - observed_mean)
            self.covariance = self.covariance - np.outer(gain, cross)
        self._repair_covariance()
        return predicted, innovation

    def _repair_covariance(self) -> None:
        covariance = 0.5 * (self.covariance + self.covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        minimum = float(np.min(eigenvalues))
        self.minimum_eigenvalue = min(self.minimum_eigenvalue, minimum)
        if minimum < 1e-12:
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            self.covariance_repairs += 1
        self.covariance = covariance

    def state_values(self) -> Mapping[str, float]:
        values = {
            name: float(value)
            for name, value in zip(self.feature_names, self.mean, strict=True)
        }
        if self.kind in {"ekf_scale", "ukf_scale"}:
            values["bounded_scale"] = self._scale()
        return values


def _governance_choice(
    realised: pd.DataFrame,
    *,
    candidate_kinds: Sequence[str],
    config: KalmanResidualConfig,
) -> tuple[str, float, Mapping[str, float]]:
    if realised.empty:
        return "identity", 0.0, {"identity": float("nan")}
    days = list(pd.Index(realised["local_day"]).drop_duplicates())
    if len(days) < config.governance_minimum_days:
        return "identity", 0.0, {"identity": float("nan")}
    selected_days = set(days[-config.governance_lookback_days :])
    trailing = realised.loc[realised["local_day"].isin(selected_days)]
    confirmation_days = int(config.governance_confirmation_days)
    if confirmation_days > 0:
        ordered_trailing_days = list(
            pd.Index(trailing["local_day"]).drop_duplicates()
        )
        selection_day_set = set(ordered_trailing_days[:-confirmation_days])
        confirmation_day_set = set(ordered_trailing_days[-confirmation_days:])
        selection = trailing.loc[trailing["local_day"].isin(selection_day_set)]
        confirmation = trailing.loc[
            trailing["local_day"].isin(confirmation_day_set)
        ]
    else:
        selection = trailing
        confirmation = pd.DataFrame()
    actual = selection["actual"].to_numpy(dtype=float)
    base = selection["base"].to_numpy(dtype=float)
    baseline_loss = float(np.mean(np.abs(base - actual)))
    losses: dict[str, float] = {"identity": baseline_loss}
    best_kind = "identity"
    best_weight = 0.0
    best_loss = baseline_loss
    steps = int(round(1.0 / config.governance_weight_step))
    weights = np.linspace(0.0, 1.0, steps + 1)
    candidate_weights: dict[str, float] = {}
    for kind in candidate_kinds:
        correction_column = f"raw::{kind}"
        if correction_column not in selection:
            continue
        correction = selection[correction_column].to_numpy(dtype=float)
        candidate_losses = np.asarray(
            [
                np.mean(np.abs(base + weight * correction - actual))
                for weight in weights
            ],
            dtype=float,
        )
        winner = int(np.argmin(candidate_losses))
        loss = float(candidate_losses[winner])
        losses[kind] = loss
        candidate_weights[kind] = float(weights[winner])
        if loss < best_loss - 1e-12:
            best_kind = kind
            best_weight = float(weights[winner])
            best_loss = loss
    minimum_gain = max(
        config.minimum_gain_eur_mwh,
        config.minimum_relative_gain * baseline_loss,
    )
    selection_gain = baseline_loss - best_loss
    selection_pass = selection_gain >= minimum_gain
    if confirmation_days > 0:
        confirmation_actual = confirmation["actual"].to_numpy(dtype=float)
        confirmation_base = confirmation["base"].to_numpy(dtype=float)
        confirmation_baseline = float(
            np.mean(np.abs(confirmation_base - confirmation_actual))
        )
        losses["confirmation::identity"] = confirmation_baseline
        losses["diagnostic::selection_days"] = float(
            selection["local_day"].nunique()
        )
        losses["diagnostic::confirmation_days"] = float(
            confirmation["local_day"].nunique()
        )
        losses["diagnostic::selection_gain"] = selection_gain
        losses["diagnostic::selection_required_gain"] = minimum_gain
        losses["diagnostic::selection_pass"] = float(selection_pass)
        for kind, weight in candidate_weights.items():
            correction_column = f"raw::{kind}"
            correction = confirmation[correction_column].to_numpy(dtype=float)
            confirmation_loss = float(
                np.mean(
                    np.abs(
                        confirmation_base
                        + weight * correction
                        - confirmation_actual
                    )
                )
            )
            required_gain = max(
                config.minimum_gain_eur_mwh,
                config.minimum_relative_gain * confirmation_baseline,
            )
            losses[f"confirmation::{kind}"] = confirmation_loss
            losses[f"weight::{kind}"] = weight
            losses[f"confirmation_gain::{kind}"] = (
                confirmation_baseline - confirmation_loss
            )
            losses[f"confirmation_required_gain::{kind}"] = required_gain
            losses[f"confirmation_pass::{kind}"] = float(
                confirmation_baseline - confirmation_loss >= required_gain
            )
    if not selection_pass:
        return "identity", 0.0, losses
    if confirmation_days > 0:
        if losses[f"confirmation_pass::{best_kind}"] < 0.5:
            return "identity", 0.0, losses
    return best_kind, best_weight, losses


def _governance_audit_maps(
    losses: Mapping[str, float],
    *,
    candidate_kinds: Sequence[str],
) -> tuple[Mapping[str, float | None], Mapping[str, float]]:
    """Split stable candidate MAEs from optional confirmation diagnostics."""

    candidate_mae = {
        kind: losses.get(kind)
        for kind in ("identity", *map(str, candidate_kinds))
    }
    diagnostics = {
        str(key): float(value)
        for key, value in losses.items()
        if "::" in str(key)
    }
    return candidate_mae, diagnostics


def _require_complete_local_day(
    block: pd.DataFrame,
    *,
    local_day: date,
    timezone: str,
    context: str,
) -> None:
    """Enforce the civil-day/DST contract on one already normalised block."""

    expected = _expected_local_day_index(local_day, timezone=timezone)
    actual = pd.DatetimeIndex(block.sort_index().index)
    if not actual.equals(expected):
        missing = expected.difference(actual)
        unexpected = actual.difference(expected)
        raise KalmanResidualError(
            f"{context}: journee locale incomplete {local_day} "
            f"({len(actual)} heures contre {len(expected)} attendues; "
            f"manquantes={len(missing)}, inattendues={len(unexpected)})."
        )


def _rolling_training_frame(
    blocks: Mapping[date, pd.DataFrame],
    *,
    target_day: date,
    lookback_days: int,
    timezone: str,
) -> tuple[pd.DataFrame, tuple[date, ...]]:
    """Select exactly D-lookback .. D-1 as complete local civil days."""

    expected_days = tuple(
        target_day - timedelta(days=offset)
        for offset in range(lookback_days, 0, -1)
    )
    missing_days = [day for day in expected_days if day not in blocks]
    if missing_days:
        preview = ", ".join(map(str, missing_days[:5]))
        suffix = "..." if len(missing_days) > 5 else ""
        raise KalmanResidualError(
            "Fenetre d'entrainement glissante incomplete pour "
            f"{target_day}: {len(missing_days)} jour(s) absent(s) "
            f"dans D-{lookback_days}..D-1 ({preview}{suffix})."
        )
    selected: list[pd.DataFrame] = []
    for local_day in expected_days:
        block = blocks[local_day].sort_index()
        _require_complete_local_day(
            block,
            local_day=local_day,
            timezone=timezone,
            context="Fenetre d'entrainement glissante",
        )
        selected.append(block)
    return pd.concat(selected, axis=0), expected_days


def _fit_rolling_target_day(
    *,
    training_frame: pd.DataFrame,
    training_days: Sequence[date],
    target_block: pd.DataFrame,
    target_day: date,
    timezone: str,
    upstream_model: str,
    output_model: str,
    config: KalmanResidualConfig,
    covariate_columns: Sequence[str],
    candidate_feature_columns: Mapping[str, Sequence[str]],
) -> Mapping[str, Any]:
    """Refit candidates on one physical window, then forecast D without D actual.

    This deliberately does not call :func:`replay_kalman_overlay`: training-day
    governance outputs are irrelevant to the target and avoiding their repeated
    construction keeps a 365 x 365 rolling-origin replay tractable.  Candidate
    state updates themselves use exactly the same ``_Candidate`` primitives as
    the legacy expanding replay.
    """

    _require_complete_local_day(
        target_block,
        local_day=target_day,
        timezone=timezone,
        context="Cible de la fenetre glissante",
    )
    training_day_set = set(training_days)
    feature_frame = pd.concat([training_frame, target_block], axis=0)
    market, scaler_audit = _market_features(
        feature_frame,
        covariate_columns=covariate_columns,
        calibration_days=training_day_set,
        clip=config.market_feature_clip,
    )
    observation_variance = _observation_variance(
        training_frame,
        calibration_days=training_day_set,
    )
    candidates = {
        kind: _Candidate(
            kind=kind,
            config=config,
            observation_variance=observation_variance,
            market=market,
            market_feature_columns=candidate_feature_columns[kind],
        )
        for kind in config.candidate_kinds
    }

    # Only the trailing governor window must retain candidate forecasts.  All
    # earlier days still update every state but allocate no realised frame.
    governance_days = set(training_days[-config.governance_lookback_days :])
    realised_rows: list[pd.DataFrame] = []
    training_clipped_shifts = 0
    for local_day, block in training_frame.groupby("_local_day", sort=True):
        block = block.sort_index()
        for candidate in candidates.values():
            candidate.transition_day()
        if local_day in governance_days:
            raw_unclipped = {
                kind: candidate.raw_correction(block)
                for kind, candidate in candidates.items()
            }
            raw_by_kind = {
                kind: np.clip(
                    values,
                    -config.shift_clip_eur_mwh,
                    config.shift_clip_eur_mwh,
                )
                for kind, values in raw_unclipped.items()
            }
            training_clipped_shifts += sum(
                int(
                    np.count_nonzero(
                        np.abs(values) > config.shift_clip_eur_mwh
                    )
                )
                for values in raw_unclipped.values()
            )
            base_q50 = block[f"{upstream_model}__q50"].to_numpy(dtype=float)
            realised_rows.append(
                pd.DataFrame(
                    {
                        "local_day": local_day,
                        "actual": block["actual"].to_numpy(dtype=float),
                        "base": base_q50,
                        **{
                            f"raw::{kind}": raw_by_kind[kind]
                            for kind in candidates
                        },
                    }
                )
            )
        for _, row in block.iterrows():
            for candidate in candidates.values():
                candidate.update_hour_rolling(row)

    state_before_transition = {
        kind: candidate.state_values()
        for kind, candidate in candidates.items()
    }
    for candidate in candidates.values():
        candidate.transition_day()
    state_at_forecast = {
        kind: candidate.state_values()
        for kind, candidate in candidates.items()
    }
    raw_unclipped = {
        kind: candidate.raw_correction(target_block)
        for kind, candidate in candidates.items()
    }
    raw_by_kind = {
        kind: np.clip(
            values,
            -config.shift_clip_eur_mwh,
            config.shift_clip_eur_mwh,
        )
        for kind, values in raw_unclipped.items()
    }
    realised = (
        pd.concat(realised_rows, ignore_index=True)
        if realised_rows
        else pd.DataFrame()
    )
    selected_kind, selected_weight, losses = _governance_choice(
        realised,
        candidate_kinds=tuple(candidates),
        config=config,
    )
    candidate_trailing_mae, governance_diagnostics = _governance_audit_maps(
        losses,
        candidate_kinds=tuple(candidates),
    )
    selected_raw = (
        np.zeros(len(target_block), dtype=float)
        if selected_kind == "identity"
        else raw_by_kind[selected_kind]
    )
    applied = selected_weight * selected_raw
    output = pd.DataFrame(index=target_block.index)
    for quantile in QUANTILES:
        output[f"{output_model}__{quantile}"] = (
            target_block[f"{upstream_model}__{quantile}"].to_numpy(dtype=float)
            + applied
        )
    output["kalman_raw_correction"] = selected_raw
    output["kalman_weight"] = selected_weight
    output["kalman_correction"] = applied
    output["kalman_selected_filter"] = selected_kind
    output.index.name = "delivery_start_utc"

    base_q50 = target_block[f"{upstream_model}__q50"].to_numpy(dtype=float)
    candidate_output = pd.DataFrame(index=target_block.index)
    for kind, values in raw_by_kind.items():
        candidate_output[f"{kind}__q50"] = base_q50 + values
    candidate_output.index.name = "delivery_start_utc"

    training_start = training_days[0]
    training_end = training_days[-1]
    target_clipped_shifts = sum(
        int(np.count_nonzero(np.abs(values) > config.shift_clip_eur_mwh))
        for values in raw_unclipped.values()
    )
    daily_audit = {
        "local_day": str(target_day),
        "hours": int(len(target_block)),
        "selected_filter": selected_kind,
        "selected_weight": float(selected_weight),
        "baseline_trailing_mae": losses.get("identity"),
        "selected_trailing_mae": losses.get(selected_kind),
        "candidate_trailing_mae": candidate_trailing_mae,
        "governance_diagnostics": governance_diagnostics,
        "raw_correction_mean": float(np.mean(selected_raw)),
        "applied_correction_mean": float(np.mean(applied)),
        "applied_correction_abs_max": float(np.max(np.abs(applied))),
        "last_observation_used": str(training_end),
        "training_window_start": str(training_start),
        "training_window_end": str(training_end),
        "training_window_days": int(len(training_days)),
        "training_window_hours": int(len(training_frame)),
        "training_window_complete": True,
        "target_observations_assimilated": 0,
        "observation_variance": float(observation_variance),
        "clipped_raw_shifts": int(target_clipped_shifts),
    }
    state_rows = [
        {
            "local_day": str(target_day),
            "filter_kind": kind,
            "state_before": state_at_forecast[kind],
            # A rolling-origin target is never assimilated.  ``state_after``
            # therefore certifies the same frozen pre-delivery state.
            "state_after": state_at_forecast[kind],
            "mean_innovation": float("nan"),
            "innovation_clips_total": candidate.innovation_clips,
            "minimum_covariance_eigenvalue": candidate.minimum_eigenvalue,
            "covariance_repairs_total": candidate.covariance_repairs,
            "training_window_start": str(training_start),
            "training_window_end": str(training_end),
            "training_window_days": int(len(training_days)),
            "training_window_hours": int(len(training_frame)),
            "target_observations_assimilated": 0,
        }
        for kind, candidate in candidates.items()
    ]
    window_audit = {
        "target_day": str(target_day),
        "training_window_start": str(training_start),
        "training_window_end": str(training_end),
        "training_window_days": int(len(training_days)),
        "training_window_hours": int(len(training_frame)),
        "target_hours": int(len(target_block)),
        "last_training_timestamp_utc": training_frame.index.max().isoformat(),
        "target_first_timestamp_utc": target_block.index.min().isoformat(),
        "target_observations_assimilated": 0,
        "observation_variance": float(observation_variance),
    }
    return {
        "predictions": output,
        "candidate_predictions": candidate_output,
        "daily_audit": daily_audit,
        "state_audit": state_rows,
        "window_audit": window_audit,
        "market_scalers": scaler_audit,
        "observation_variance": float(observation_variance),
        "state_before_transition": state_before_transition,
        "state_after_transition": state_at_forecast,
        "candidate_feature_counts": {
            kind: int(len(candidate.feature_names))
            for kind, candidate in candidates.items()
        },
        "target_clipped_shifts": int(target_clipped_shifts),
        "training_clipped_shifts": int(training_clipped_shifts),
        "governance_realised_days": int(realised["local_day"].nunique())
        if not realised.empty
        else 0,
        "governance_realised_hours": int(len(realised)),
    }


def _fit_rolling_target_day_chunk(
    *,
    blocks: Mapping[date, pd.DataFrame],
    target_days: Sequence[date],
    lookback_days: int,
    timezone: str,
    upstream_model: str,
    output_model: str,
    config: KalmanResidualConfig,
    covariate_columns: Sequence[str],
    candidate_feature_columns: Mapping[str, Sequence[str]],
) -> list[tuple[date, Mapping[str, Any]]]:
    """Fit an ordered group of independent rolling origins in one worker."""

    results: list[tuple[date, Mapping[str, Any]]] = []
    for target_day in target_days:
        training_frame, training_days = _rolling_training_frame(
            blocks,
            target_day=target_day,
            lookback_days=lookback_days,
            timezone=timezone,
        )
        results.append(
            (
                target_day,
                _fit_rolling_target_day(
                    training_frame=training_frame,
                    training_days=training_days,
                    target_block=blocks[target_day],
                    target_day=target_day,
                    timezone=timezone,
                    upstream_model=upstream_model,
                    output_model=output_model,
                    config=config,
                    covariate_columns=covariate_columns,
                    candidate_feature_columns=candidate_feature_columns,
                ),
            )
        )
    return results


def _replay_rolling_kalman_overlay(
    history: pd.DataFrame,
    *,
    timezone: str,
    upstream_model: str,
    output_model: str,
    evaluation_start_day: str | pd.Timestamp | None,
    covariates: pd.DataFrame | None,
    future_upstream: pd.DataFrame | None,
    future_covariates: pd.DataFrame | None,
    training_lookback_days: int,
    rolling_refit_workers: int,
    rolling_refit_cache_dir: Path | None,
    config: KalmanResidualConfig,
    covariate_config: KalmanCovariateConfig,
) -> KalmanReplayResult:
    """Exact fixed-length rolling-origin replay in local civil days."""

    lookback_days = training_lookback_days
    frame, covariate_columns, covariate_audit = _normalise_input(
        history,
        upstream_model=upstream_model,
        timezone=timezone,
        covariates=covariates,
        covariate_config=covariate_config,
    )
    blocks: dict[date, pd.DataFrame] = {
        local_day: block.sort_index()
        for local_day, block in frame.groupby("_local_day", sort=True)
    }
    all_days = tuple(sorted(blocks))
    if evaluation_start_day is None:
        if len(all_days) <= lookback_days:
            raise KalmanResidualError(
                "Historique insuffisant pour demarrer une evaluation "
                f"rolling D-{lookback_days} sans evaluation_start_day."
            )
        evaluation_start = all_days[lookback_days]
    else:
        evaluation_start = pd.Timestamp(evaluation_start_day).date()
    evaluation_days = tuple(day for day in all_days if day >= evaluation_start)
    if not evaluation_days:
        raise KalmanResidualError(
            "evaluation_start_day est posterieur a tout l'historique disponible."
        )

    candidate_feature_columns = {
        kind: _candidate_market_feature_columns(
            kind,
            covariate_config=covariate_config,
            covariate_columns=covariate_columns,
        )
        for kind in config.candidate_kinds
    }
    output = pd.DataFrame(index=frame.index)
    for quantile in QUANTILES:
        output[f"{output_model}__{quantile}"] = frame[
            f"{upstream_model}__{quantile}"
        ].to_numpy(dtype=float)
    output["kalman_raw_correction"] = 0.0
    output["kalman_weight"] = 0.0
    output["kalman_correction"] = 0.0
    output["kalman_selected_filter"] = "identity"
    output.index.name = "delivery_start_utc"

    candidate_output = pd.DataFrame(index=frame.index)
    base_q50 = frame[f"{upstream_model}__q50"].to_numpy(dtype=float)
    for kind in config.candidate_kinds:
        candidate_output[f"{kind}__q50"] = base_q50
    candidate_output.index.name = "delivery_start_utc"

    daily_by_day: dict[date, dict[str, Any]] = {
        local_day: {
            "local_day": str(local_day),
            "hours": int(len(block)),
            "selected_filter": "identity",
            "selected_weight": 0.0,
            "baseline_trailing_mae": float("nan"),
            "selected_trailing_mae": float("nan"),
            "candidate_trailing_mae": {},
            "governance_diagnostics": {},
            "raw_correction_mean": 0.0,
            "applied_correction_mean": 0.0,
            "applied_correction_abs_max": 0.0,
            "last_observation_used": None,
            "training_window_start": None,
            "training_window_end": None,
            "training_window_days": 0,
            "training_window_hours": 0,
            "training_window_complete": False,
            "target_observations_assimilated": 0,
            "observation_variance": float("nan"),
            "clipped_raw_shifts": 0,
        }
        for local_day, block in blocks.items()
    }
    state_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    latest_fit: Mapping[str, Any] | None = None
    total_clipped_shifts = 0
    target_fits: dict[date, Mapping[str, Any]] = {}
    worker_count = min(rolling_refit_workers, len(evaluation_days))
    cache_root = (
        rolling_refit_cache_dir.expanduser().resolve()
        if rolling_refit_cache_dir is not None
        else None
    )
    cache_stats: dict[str, int] = {
        "history_hits": 0,
        "history_misses": 0,
        "future_hits": 0,
        "future_misses": 0,
        "stale_entries": 0,
        "invalid_entries": 0,
        "writes": 0,
        "write_errors": 0,
    }
    cache_input_digests: dict[date, str] = {}
    block_digests: dict[date, str] = {}
    cache_contract_digest: str | None = None
    fit_days: tuple[date, ...] = evaluation_days
    if cache_root is not None:
        # A cache hit must retain the same physical-day validation as a fresh
        # fit. Validate every *used* block once instead of repeating it for
        # each of the 365 overlapping rolling windows. Unused prefix days keep
        # the exact legacy semantics and cannot make caching reject a replay.
        required_cache_days = set(evaluation_days)
        for target_day in evaluation_days:
            required_cache_days.update(
                target_day - timedelta(days=offset)
                for offset in range(lookback_days, 0, -1)
            )
        for local_day in sorted(required_cache_days.intersection(blocks)):
            block = blocks[local_day]
            _require_complete_local_day(
                block,
                local_day=local_day,
                timezone=timezone,
                context="Cache du refit glissant",
            )
            block_digests[local_day] = _rolling_cache_frame_digest(block)
        cache_contract_digest = _rolling_cache_contract_digest(
            timezone=timezone,
            upstream_model=upstream_model,
            output_model=output_model,
            lookback_days=lookback_days,
            config=config,
            covariate_columns=covariate_columns,
            candidate_feature_columns=candidate_feature_columns,
        )
        missing_fit_days: list[date] = []
        for target_day in evaluation_days:
            training_days = tuple(
                target_day - timedelta(days=offset)
                for offset in range(lookback_days, 0, -1)
            )
            missing_days = [day for day in training_days if day not in block_digests]
            if missing_days:
                preview = ", ".join(map(str, missing_days[:5]))
                suffix = "..." if len(missing_days) > 5 else ""
                raise KalmanResidualError(
                    "Fenetre d'entrainement glissante incomplete pour "
                    f"{target_day}: {len(missing_days)} jour(s) absent(s) "
                    f"dans D-{lookback_days}..D-1 ({preview}{suffix})."
                )
            input_digest = _rolling_cache_input_digest(
                contract_digest=cache_contract_digest,
                target_day=target_day,
                training_days=training_days,
                block_digests=block_digests,
                target_digest=_rolling_cache_target_digest(blocks[target_day]),
            )
            cache_input_digests[target_day] = input_digest
            cached, cache_status = _load_cached_rolling_fit(
                _rolling_cache_path(
                    cache_root,
                    cache_contract_digest,
                    target_day,
                ),
                input_digest=input_digest,
                target_day=target_day,
                target_index=blocks[target_day].index,
                output_model=output_model,
                candidate_kinds=config.candidate_kinds,
            )
            if cached is not None:
                target_fits[target_day] = cached
                cache_stats["history_hits"] += 1
                continue
            cache_stats["history_misses"] += 1
            if cache_status == "stale":
                cache_stats["stale_entries"] += 1
            elif cache_status == "invalid":
                cache_stats["invalid_entries"] += 1
            missing_fit_days.append(target_day)
        fit_days = tuple(missing_fit_days)

    fit_worker_count = min(rolling_refit_workers, len(fit_days)) if fit_days else 0
    chunks = (
        [fit_days[offset::fit_worker_count] for offset in range(fit_worker_count)]
        if fit_worker_count
        else []
    )
    if fit_worker_count == 1:
        chunk_results = [
            _fit_rolling_target_day_chunk(
                blocks=blocks,
                target_days=chunks[0],
                lookback_days=lookback_days,
                timezone=timezone,
                upstream_model=upstream_model,
                output_model=output_model,
                config=config,
                covariate_columns=covariate_columns,
                candidate_feature_columns=candidate_feature_columns,
            )
        ]
    elif fit_worker_count > 1:
        try:
            from joblib import Parallel, delayed
        except ImportError as exc:  # pragma: no cover - sklearn installs it.
            raise KalmanResidualError(
                "joblib est requis lorsque rolling_refit_workers > 1."
            ) from exc
        chunk_results = Parallel(n_jobs=fit_worker_count, backend="loky")(
            delayed(_fit_rolling_target_day_chunk)(
                blocks=blocks,
                target_days=chunk,
                lookback_days=lookback_days,
                timezone=timezone,
                upstream_model=upstream_model,
                output_model=output_model,
                config=config,
                covariate_columns=covariate_columns,
                candidate_feature_columns=candidate_feature_columns,
            )
            for chunk in chunks
        )
    else:
        chunk_results = []
    for chunk in chunk_results:
        target_fits.update(chunk)
    if cache_root is not None:
        for target_day in fit_days:
            try:
                _write_cached_rolling_fit(
                    _rolling_cache_path(
                        cache_root,
                        cache_contract_digest,
                        target_day,
                    ),
                    input_digest=cache_input_digests[target_day],
                    target_day=target_day,
                    fit=target_fits[target_day],
                )
                cache_stats["writes"] += 1
            except Exception:
                # Caching is a performance layer. A transient filesystem or
                # serialisation failure must never change a valid forecast.
                cache_stats["write_errors"] += 1

    for target_day in evaluation_days:
        target_block = blocks[target_day]
        fit = target_fits[target_day]
        target_index = target_block.index
        output.loc[target_index, fit["predictions"].columns] = fit[
            "predictions"
        ].to_numpy()
        candidate_output.loc[
            target_index, fit["candidate_predictions"].columns
        ] = fit["candidate_predictions"].to_numpy()
        daily_by_day[target_day] = dict(fit["daily_audit"])
        state_rows.extend(fit["state_audit"])
        window_rows.append(dict(fit["window_audit"]))
        total_clipped_shifts += int(fit["target_clipped_shifts"])
        latest_fit = fit

    future_frame: pd.DataFrame | None = None
    future_output = pd.DataFrame()
    future_candidate_output = pd.DataFrame()
    future_audit: dict[str, Any] = {}
    if future_upstream is not None:
        future_frame = _normalise_future_input(
            future_upstream,
            upstream_model=upstream_model,
            timezone=timezone,
            covariates=future_covariates,
            covariate_columns=covariate_columns,
            covariate_config=covariate_config,
        )
        history_last_day = all_days[-1]
        _require_complete_local_day(
            blocks[history_last_day],
            local_day=history_last_day,
            timezone=timezone,
            context="Derniere journee historique",
        )
        future_day = future_frame["_local_day"].iloc[0]
        if future_day != history_last_day + timedelta(days=1):
            raise KalmanResidualError(
                "future_upstream doit etre la journee locale suivant "
                "immediatement l'historique."
            )
        training_days = tuple(
            future_day - timedelta(days=offset)
            for offset in range(lookback_days, 0, -1)
        )
        fit: Mapping[str, Any] | None = None
        future_input_digest: str | None = None
        if cache_root is not None:
            assert cache_contract_digest is not None
            future_input_digest = _rolling_cache_input_digest(
                contract_digest=cache_contract_digest,
                target_day=future_day,
                training_days=training_days,
                block_digests=block_digests,
                target_digest=_rolling_cache_target_digest(future_frame),
            )
            fit, cache_status = _load_cached_rolling_fit(
                _rolling_cache_path(
                    cache_root,
                    cache_contract_digest,
                    future_day,
                ),
                input_digest=future_input_digest,
                target_day=future_day,
                target_index=future_frame.index,
                output_model=output_model,
                candidate_kinds=config.candidate_kinds,
            )
            if fit is not None:
                cache_stats["future_hits"] += 1
            else:
                cache_stats["future_misses"] += 1
                if cache_status == "stale":
                    cache_stats["stale_entries"] += 1
                elif cache_status == "invalid":
                    cache_stats["invalid_entries"] += 1
        if fit is None:
            training_frame, training_days = _rolling_training_frame(
                blocks,
                target_day=future_day,
                lookback_days=lookback_days,
                timezone=timezone,
            )
            fit = _fit_rolling_target_day(
                training_frame=training_frame,
                training_days=training_days,
                target_block=future_frame,
                target_day=future_day,
                timezone=timezone,
                upstream_model=upstream_model,
                output_model=output_model,
                config=config,
                covariate_columns=covariate_columns,
                candidate_feature_columns=candidate_feature_columns,
            )
            if cache_root is not None and future_input_digest is not None:
                try:
                    _write_cached_rolling_fit(
                        _rolling_cache_path(
                            cache_root,
                            cache_contract_digest,
                            future_day,
                        ),
                        input_digest=future_input_digest,
                        target_day=future_day,
                        fit=fit,
                    )
                    cache_stats["writes"] += 1
                except Exception:
                    cache_stats["write_errors"] += 1
        future_output = fit["predictions"]
        future_candidate_output = fit["candidate_predictions"]
        future_audit = {
            "status": "complete",
            **dict(fit["daily_audit"]),
            "actual_column_ignored": bool("actual" in future_upstream.columns),
            "observations_assimilated": 0,
            "last_observation_timestamp_utc": blocks[
                training_days[-1]
            ].index.max().isoformat(),
            "governance_realised_days": int(fit["governance_realised_days"]),
            "governance_realised_hours": int(fit["governance_realised_hours"]),
            "state_before_transition": fit["state_before_transition"],
            "state_after_transition": fit["state_after_transition"],
        }
        total_clipped_shifts += int(fit["target_clipped_shifts"])
        latest_fit = fit

    prediction_values = output[
        [f"{output_model}__{quantile}" for quantile in QUANTILES]
    ].to_numpy(dtype=float)
    if not np.isfinite(prediction_values).all():
        raise KalmanResidualError("Le replay Kalman produit des valeurs non finies.")
    if bool(
        (output[f"{output_model}__q10"] > output[f"{output_model}__q50"]).any()
        or (output[f"{output_model}__q50"] > output[f"{output_model}__q90"]).any()
    ):
        raise KalmanResidualError("Le replay Kalman croise les quantiles.")
    if not future_output.empty:
        future_values = future_output[
            [f"{output_model}__{quantile}" for quantile in QUANTILES]
        ].to_numpy(dtype=float)
        if not np.isfinite(future_values).all():
            raise KalmanResidualError(
                "La prevision Kalman future produit des valeurs non finies."
            )
        if bool(
            (
                future_output[f"{output_model}__q10"]
                > future_output[f"{output_model}__q50"]
            ).any()
            or (
                future_output[f"{output_model}__q50"]
                > future_output[f"{output_model}__q90"]
            ).any()
        ):
            raise KalmanResidualError(
                "La prevision Kalman future croise les quantiles."
            )

    assert latest_fit is not None
    evaluation_index = frame.index[frame["_local_day"].isin(evaluation_days)]
    training_hours = [int(row["training_window_hours"]) for row in window_rows]
    audit: dict[str, Any] = {
        "schema_version": 2,
        "status": "complete",
        "model_key": output_model,
        "upstream_model": upstream_model,
        "algorithm": "governed_daily_kf_ekf_ukf_rolling_refit",
        "pykalman_version": metadata.version("pykalman"),
        "filter_only": True,
        "smoother_used": False,
        "em_used": False,
        "state_update_frequency": (
            "independent refit per target local day; one transition per "
            "training day; scalar hourly updates; target frozen"
        ),
        "quantile_policy": "same additive shift on q10/q50/q90",
        "candidate_kinds": list(config.candidate_kinds),
        "config": asdict(config),
        "covariate_config": covariate_config.to_dict(),
        "observation_variance": float(latest_fit["observation_variance"]),
        "market_scalers": latest_fit["market_scalers"],
        "market_scalers_scope": "latest rolling target only",
        "covariate_columns": list(covariate_columns),
        "covariate_groups": {
            name: list(columns)
            for name, columns in covariate_config.groups.items()
        },
        "covariate_coverage": list(covariate_audit["coverage"]),
        "structural_ramp_hours": dict(
            covariate_audit["structural_ramp_hours"]
        ),
        "candidate_feature_columns": {
            kind: list(columns)
            for kind, columns in candidate_feature_columns.items()
        },
        "candidate_feature_counts": latest_fit["candidate_feature_counts"],
        "training_policy": "fixed_length_rolling_local_days",
        "training_lookback_days": int(lookback_days),
        "rolling_refit_workers": int(worker_count),
        "rolling_refit_active_workers": int(fit_worker_count),
        "rolling_refit_cache": {
            "enabled": cache_root is not None,
            "schema_version": ROLLING_REFIT_CACHE_SCHEMA_VERSION,
            "directory": str(cache_root) if cache_root is not None else None,
            "contract_directory": (
                str(
                    _rolling_cache_contract_dir(
                        cache_root,
                        cache_contract_digest,
                    )
                )
                if cache_root is not None and cache_contract_digest is not None
                else None
            ),
            "contract_digest": cache_contract_digest,
            **cache_stats,
            "history_fitted_days": int(len(fit_days)),
            "future_fitted_days": int(
                future_frame is not None and cache_stats["future_hits"] == 0
            ),
        },
        "training_window_timezone": timezone,
        "training_windows_audited": int(len(window_rows)),
        "training_window_hours_min": int(min(training_hours)),
        "training_window_hours_max": int(max(training_hours)),
        "rolling_training_windows": window_rows,
        "warmup_start_day": window_rows[0]["training_window_start"],
        "warmup_end_day": window_rows[0]["training_window_end"],
        "warmup_days": int(lookback_days),
        "evaluation_start_day": str(evaluation_days[0]),
        "evaluation_end_day": str(evaluation_days[-1]),
        "evaluation_days": int(len(evaluation_days)),
        "evaluation_hours": int(len(evaluation_index)),
        "last_observation_assimilated_day": (
            future_audit.get("last_observation_used")
            if future_audit
            else window_rows[-1]["training_window_end"]
        ),
        "future_forecast_produced": future_frame is not None,
        "future_forecast_day": future_audit.get("local_day"),
        "future_forecast_hours": int(len(future_output)),
        "future_last_observation_used": future_audit.get("last_observation_used"),
        "future_observations_assimilated": int(
            future_audit.get("observations_assimilated", 0)
        ),
        "causality_violations": 0,
        "quantile_crossings": 0,
        "clipped_raw_shifts": int(total_clipped_shifts),
        "selected_filter_counts": (
            pd.Series(
                [daily_by_day[day]["selected_filter"] for day in evaluation_days]
            )
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "target_actuals_assimilated_before_forecast": 0,
        "used_for_storm": False,
        "storm_used_as_input": False,
    }
    return KalmanReplayResult(
        predictions=output,
        candidate_predictions=candidate_output,
        daily_audit=pd.DataFrame(
            [daily_by_day[day] for day in all_days]
        ),
        state_audit=pd.DataFrame(state_rows),
        audit=audit,
        future_predictions=future_output,
        future_candidate_predictions=future_candidate_output,
        future_audit=future_audit,
    )


def replay_kalman_overlay(
    history: pd.DataFrame,
    *,
    timezone: str,
    upstream_model: str = "residual_corrected",
    output_model: str = DEFAULT_MODEL_KEY,
    evaluation_start_day: str | pd.Timestamp | None = None,
    covariates: pd.DataFrame | None = None,
    future_upstream: pd.DataFrame | None = None,
    future_covariates: pd.DataFrame | None = None,
    training_lookback_days: int | None = None,
    rolling_refit_workers: int = 1,
    rolling_refit_cache_dir: str | Path | None = None,
    config: KalmanResidualConfig | None = None,
    covariate_config: KalmanCovariateConfig | None = None,
) -> KalmanReplayResult:
    """Replay a governed family of causal KF/EKF/UKF candidates.

    ``history`` may include a warm-up prefix.  When ``evaluation_start_day`` is
    supplied, all robust scales are frozen from the strict prefix before that
    day; the evaluation suffix is never used to fit scaling parameters.

    ``future_upstream`` may contain the immediately following complete local
    delivery day.  Its forecast is produced after one daily transition, using
    governance and states learned exclusively from realised history.  An
    ``actual`` column, if present in that frame, is deliberately ignored and
    no future observation is assimilated.

    Set ``training_lookback_days`` (typically 365) to replace the legacy
    expanding state with an independent fixed-length refit for every evaluated
    local day D.  State, robust scalers, observation variance and governance
    then see exactly D-lookback .. D-1.  ``None`` preserves legacy behaviour.
    ``rolling_refit_cache_dir`` stores content-addressed daily fits; a hit is
    accepted only when the exact training/target blocks, policy, dependencies
    and implementation hash still match.
    """

    selected_config = config or KalmanResidualConfig()
    selected_config.validate()
    selected_training_lookback = _validate_training_lookback_days(
        training_lookback_days
    )
    selected_rolling_workers = _validate_rolling_refit_workers(
        rolling_refit_workers
    )
    selected_covariate_config = covariate_config or KalmanCovariateConfig()
    try:
        selected_covariate_config.validate()
    except KalmanCovariateError as exc:
        raise KalmanResidualError(str(exc)) from exc
    _require_pykalman()
    if not output_model or output_model == upstream_model:
        raise KalmanResidualError("output_model doit etre distinct de l'upstream.")
    if future_upstream is None and future_covariates is not None:
        raise KalmanResidualError(
            "future_covariates requiert future_upstream."
        )
    if selected_training_lookback is not None:
        return _replay_rolling_kalman_overlay(
            history,
            timezone=timezone,
            upstream_model=upstream_model,
            output_model=output_model,
            evaluation_start_day=evaluation_start_day,
            covariates=covariates,
            future_upstream=future_upstream,
            future_covariates=future_covariates,
            training_lookback_days=selected_training_lookback,
            rolling_refit_workers=selected_rolling_workers,
            rolling_refit_cache_dir=(
                Path(rolling_refit_cache_dir)
                if rolling_refit_cache_dir not in (None, "")
                else None
            ),
            config=selected_config,
            covariate_config=selected_covariate_config,
        )
    frame, covariate_columns, covariate_audit = _normalise_input(
        history,
        upstream_model=upstream_model,
        timezone=timezone,
        covariates=covariates,
        covariate_config=selected_covariate_config,
    )
    all_days = list(pd.Index(frame["_local_day"]).drop_duplicates())
    future_frame: pd.DataFrame | None = None
    if future_upstream is not None:
        future_frame = _normalise_future_input(
            future_upstream,
            upstream_model=upstream_model,
            timezone=timezone,
            covariates=future_covariates,
            covariate_columns=covariate_columns,
            covariate_config=selected_covariate_config,
        )
        last_history_day = all_days[-1]
        history_last_index = frame.loc[
            frame["_local_day"] == last_history_day
        ].index
        expected_history_index = _expected_local_day_index(
            last_history_day,
            timezone=timezone,
        )
        if not history_last_index.equals(expected_history_index):
            raise KalmanResidualError(
                "La derniere journee historique doit etre complete avant une "
                "prevision future."
            )
        future_day = future_frame["_local_day"].iloc[0]
        expected_future_day = (
            pd.Timestamp(last_history_day) + pd.Timedelta(days=1)
        ).date()
        if future_day != expected_future_day:
            raise KalmanResidualError(
                "future_upstream doit etre la journee locale suivant "
                "immediatement l'historique."
            )
        if future_frame.index.intersection(frame.index).size:
            raise KalmanResidualError(
                "La timeline future chevauche la timeline historique."
            )
    if evaluation_start_day is None:
        evaluation_start = all_days[0]
        calibration_days = set(all_days[: min(30, len(all_days))])
    else:
        evaluation_start = pd.Timestamp(evaluation_start_day).date()
        calibration_days = {day for day in all_days if day < evaluation_start}
        if not calibration_days:
            raise KalmanResidualError(
                "Une periode de warm-up stricte est requise avant l'evaluation."
            )
    feature_frame = (
        pd.concat([frame, future_frame], axis=0)
        if future_frame is not None
        else frame
    )
    market, scaler_audit = _market_features(
        feature_frame,
        covariate_columns=covariate_columns,
        calibration_days=calibration_days,
        clip=selected_config.market_feature_clip,
    )
    observation_variance = _observation_variance(
        frame,
        calibration_days=calibration_days,
    )
    candidate_feature_columns = {
        kind: _candidate_market_feature_columns(
            kind,
            covariate_config=selected_covariate_config,
            covariate_columns=covariate_columns,
        )
        for kind in selected_config.candidate_kinds
    }
    candidates = {
        kind: _Candidate(
            kind=kind,
            config=selected_config,
            observation_variance=observation_variance,
            market=market,
            market_feature_columns=candidate_feature_columns[kind],
        )
        for kind in selected_config.candidate_kinds
    }

    candidate_columns = {
        kind: pd.Series(np.nan, index=frame.index, dtype=float)
        for kind in candidates
    }
    output = pd.DataFrame(index=frame.index)
    daily_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    realised_rows: list[pd.DataFrame] = []
    last_observed_day: object | None = None
    causality_violations = 0
    total_clipped_shifts = 0

    for local_day, block in frame.groupby("_local_day", sort=True):
        block = block.sort_index()
        last_used_before_forecast = last_observed_day
        if (
            last_used_before_forecast is not None
            and last_used_before_forecast >= local_day
        ):
            causality_violations += 1
        for candidate in candidates.values():
            candidate.transition_day()
        raw_unclipped_by_kind = {
            kind: candidate.raw_correction(block)
            for kind, candidate in candidates.items()
        }
        raw_by_kind = {
            kind: np.clip(
                raw_unclipped_by_kind[kind],
                -selected_config.shift_clip_eur_mwh,
                selected_config.shift_clip_eur_mwh,
            )
            for kind in candidates
        }
        total_clipped_shifts += sum(
            int(
                np.count_nonzero(
                    np.abs(raw_unclipped_by_kind[kind])
                    > selected_config.shift_clip_eur_mwh
                )
            )
            for kind in candidates
        )
        realised = (
            pd.concat(realised_rows, ignore_index=True)
            if realised_rows
            else pd.DataFrame()
        )
        selected_kind, selected_weight, losses = _governance_choice(
            realised,
            candidate_kinds=tuple(candidates),
            config=selected_config,
        )
        candidate_trailing_mae, governance_diagnostics = _governance_audit_maps(
            losses,
            candidate_kinds=tuple(candidates),
        )
        selected_raw = (
            np.zeros(len(block), dtype=float)
            if selected_kind == "identity"
            else raw_by_kind[selected_kind]
        )
        applied = selected_weight * selected_raw
        base_q50 = block[f"{upstream_model}__q50"].to_numpy(dtype=float)
        for quantile in QUANTILES:
            output.loc[block.index, f"{output_model}__{quantile}"] = (
                block[f"{upstream_model}__{quantile}"].to_numpy(dtype=float)
                + applied
            )
        output.loc[block.index, "kalman_raw_correction"] = selected_raw
        output.loc[block.index, "kalman_weight"] = selected_weight
        output.loc[block.index, "kalman_correction"] = applied
        output.loc[block.index, "kalman_selected_filter"] = selected_kind
        for kind, values in raw_by_kind.items():
            candidate_columns[kind].loc[block.index] = base_q50 + values

        state_before = {
            kind: candidate.state_values() for kind, candidate in candidates.items()
        }
        innovation_sums = {kind: 0.0 for kind in candidates}
        for _, row in block.iterrows():
            for kind, candidate in candidates.items():
                _, innovation = candidate.update_hour(row)
                innovation_sums[kind] += float(innovation)
        last_observed_day = local_day
        for kind, candidate in candidates.items():
            state_rows.append(
                {
                    "local_day": str(local_day),
                    "filter_kind": kind,
                    "state_before": state_before[kind],
                    "state_after": candidate.state_values(),
                    "mean_innovation": innovation_sums[kind] / len(block),
                    "innovation_clips_total": candidate.innovation_clips,
                    "minimum_covariance_eigenvalue": candidate.minimum_eigenvalue,
                    "covariance_repairs_total": candidate.covariance_repairs,
                }
            )
        daily_rows.append(
            {
                "local_day": str(local_day),
                "hours": int(len(block)),
                "selected_filter": selected_kind,
                "selected_weight": float(selected_weight),
                "baseline_trailing_mae": losses.get("identity"),
                "selected_trailing_mae": losses.get(selected_kind),
                "candidate_trailing_mae": candidate_trailing_mae,
                "governance_diagnostics": governance_diagnostics,
                "raw_correction_mean": float(np.mean(selected_raw)),
                "applied_correction_mean": float(np.mean(applied)),
                "applied_correction_abs_max": float(np.max(np.abs(applied))),
                "last_observation_used": (
                    str(last_used_before_forecast)
                    if last_used_before_forecast is not None
                    else None
                ),
            }
        )
        realised_block = pd.DataFrame(
            {
                "local_day": local_day,
                "actual": block["actual"].to_numpy(dtype=float),
                "base": base_q50,
                **{
                    f"raw::{kind}": raw_by_kind[kind]
                    for kind in candidates
                },
            }
        )
        realised_rows.append(realised_block)

    future_output = pd.DataFrame()
    future_candidate_prediction_frame = pd.DataFrame()
    future_audit: dict[str, Any] = {}
    if future_frame is not None:
        future_block = future_frame.sort_index()
        future_day = future_block["_local_day"].iloc[0]
        last_used_before_future = last_observed_day
        state_before_transition = {
            kind: candidate.state_values()
            for kind, candidate in candidates.items()
        }
        for candidate in candidates.values():
            candidate.transition_day()
        state_after_transition = {
            kind: candidate.state_values()
            for kind, candidate in candidates.items()
        }
        future_raw_unclipped = {
            kind: candidate.raw_correction(future_block)
            for kind, candidate in candidates.items()
        }
        future_raw_by_kind = {
            kind: np.clip(
                values,
                -selected_config.shift_clip_eur_mwh,
                selected_config.shift_clip_eur_mwh,
            )
            for kind, values in future_raw_unclipped.items()
        }
        realised = (
            pd.concat(realised_rows, ignore_index=True)
            if realised_rows
            else pd.DataFrame()
        )
        selected_kind, selected_weight, losses = _governance_choice(
            realised,
            candidate_kinds=tuple(candidates),
            config=selected_config,
        )
        candidate_trailing_mae, governance_diagnostics = _governance_audit_maps(
            losses,
            candidate_kinds=tuple(candidates),
        )
        selected_raw = (
            np.zeros(len(future_block), dtype=float)
            if selected_kind == "identity"
            else future_raw_by_kind[selected_kind]
        )
        applied = selected_weight * selected_raw
        future_output = pd.DataFrame(index=future_block.index)
        for quantile in QUANTILES:
            future_output[f"{output_model}__{quantile}"] = (
                future_block[f"{upstream_model}__{quantile}"].to_numpy(
                    dtype=float
                )
                + applied
            )
        future_output["kalman_raw_correction"] = selected_raw
        future_output["kalman_weight"] = selected_weight
        future_output["kalman_correction"] = applied
        future_output["kalman_selected_filter"] = selected_kind
        future_output.index.name = "delivery_start_utc"

        future_candidate_prediction_frame = pd.DataFrame(
            index=future_block.index
        )
        future_base_q50 = future_block[
            f"{upstream_model}__q50"
        ].to_numpy(dtype=float)
        for kind, values in future_raw_by_kind.items():
            future_candidate_prediction_frame[f"{kind}__q50"] = (
                future_base_q50 + values
            )
        future_candidate_prediction_frame.index.name = "delivery_start_utc"

        future_prediction_values = future_output[
            [f"{output_model}__{quantile}" for quantile in QUANTILES]
        ].to_numpy(dtype=float)
        if not np.isfinite(future_prediction_values).all():
            raise KalmanResidualError(
                "La prevision Kalman future produit des valeurs non finies."
            )
        if bool(
            (
                future_output[f"{output_model}__q10"]
                > future_output[f"{output_model}__q50"]
            ).any()
            or (
                future_output[f"{output_model}__q50"]
                > future_output[f"{output_model}__q90"]
            ).any()
        ):
            raise KalmanResidualError(
                "La prevision Kalman future croise les quantiles."
            )
        future_clipped_shifts = sum(
            int(
                np.count_nonzero(
                    np.abs(values) > selected_config.shift_clip_eur_mwh
                )
            )
            for values in future_raw_unclipped.values()
        )
        future_audit = {
            "status": "complete",
            "local_day": str(future_day),
            "hours": int(len(future_block)),
            "selected_filter": selected_kind,
            "selected_weight": float(selected_weight),
            "baseline_trailing_mae": losses.get("identity"),
            "selected_trailing_mae": losses.get(selected_kind),
            "candidate_trailing_mae": candidate_trailing_mae,
            "governance_diagnostics": governance_diagnostics,
            "raw_correction_mean": float(np.mean(selected_raw)),
            "applied_correction_mean": float(np.mean(applied)),
            "applied_correction_abs_max": float(np.max(np.abs(applied))),
            "last_observation_used": (
                str(last_used_before_future)
                if last_used_before_future is not None
                else None
            ),
            "last_observation_timestamp_utc": frame.index.max().isoformat(),
            "observations_assimilated": 0,
            "actual_column_ignored": bool("actual" in future_upstream.columns),
            "governance_realised_days": int(
                realised["local_day"].nunique()
            ),
            "governance_realised_hours": int(len(realised)),
            "clipped_raw_shifts": int(future_clipped_shifts),
            "state_before_transition": state_before_transition,
            "state_after_transition": state_after_transition,
        }

    candidate_prediction_frame = pd.DataFrame(index=frame.index)
    for kind, values in candidate_columns.items():
        candidate_prediction_frame[f"{kind}__q50"] = values
    prediction_values = output[
        [f"{output_model}__{quantile}" for quantile in QUANTILES]
    ].to_numpy(dtype=float)
    if not np.isfinite(prediction_values).all():
        raise KalmanResidualError("Le replay Kalman produit des valeurs non finies.")
    if bool(
        (
            output[f"{output_model}__q10"]
            > output[f"{output_model}__q50"]
        ).any()
        or (
            output[f"{output_model}__q50"]
            > output[f"{output_model}__q90"]
        ).any()
    ):
        raise KalmanResidualError("Le replay Kalman croise les quantiles.")

    evaluation_mask = frame["_local_day"] >= evaluation_start
    evaluation_days = pd.Index(frame.loc[evaluation_mask, "_local_day"]).unique()
    if evaluation_days.empty:
        raise KalmanResidualError(
            "evaluation_start_day est posterieur a tout l'historique disponible."
        )
    audit: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "model_key": output_model,
        "upstream_model": upstream_model,
        "algorithm": "governed_daily_kf_ekf_ukf_filter_update",
        "pykalman_version": metadata.version("pykalman"),
        "filter_only": True,
        "smoother_used": False,
        "em_used": False,
        "state_update_frequency": "once transition per local day; scalar hourly corrections after day freeze",
        "quantile_policy": "same additive shift on q10/q50/q90",
        "candidate_kinds": list(selected_config.candidate_kinds),
        "config": asdict(selected_config),
        "covariate_config": selected_covariate_config.to_dict(),
        "observation_variance": observation_variance,
        "market_scalers": scaler_audit,
        "covariate_columns": list(covariate_columns),
        "covariate_groups": {
            name: list(columns)
            for name, columns in selected_covariate_config.groups.items()
        },
        "covariate_coverage": list(covariate_audit["coverage"]),
        "structural_ramp_hours": dict(
            covariate_audit["structural_ramp_hours"]
        ),
        "candidate_feature_columns": {
            kind: list(columns)
            for kind, columns in candidate_feature_columns.items()
        },
        "candidate_feature_counts": {
            kind: int(len(candidates[kind].feature_names))
            for kind in candidates
        },
        "warmup_start_day": str(min(calibration_days)),
        "warmup_end_day": str(max(calibration_days)),
        "warmup_days": int(len(calibration_days)),
        "evaluation_start_day": str(evaluation_start),
        "evaluation_end_day": str(evaluation_days[-1]),
        "evaluation_days": int(len(evaluation_days)),
        "evaluation_hours": int(evaluation_mask.sum()),
        "last_observation_assimilated_day": str(last_observed_day),
        "future_forecast_produced": future_frame is not None,
        "future_forecast_day": future_audit.get("local_day"),
        "future_forecast_hours": int(len(future_output)),
        "future_last_observation_used": future_audit.get(
            "last_observation_used"
        ),
        "future_observations_assimilated": int(
            future_audit.get("observations_assimilated", 0)
        ),
        "causality_violations": causality_violations,
        "quantile_crossings": 0,
        "clipped_raw_shifts": int(total_clipped_shifts),
        "selected_filter_counts": (
            pd.Series([row["selected_filter"] for row in daily_rows])
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "used_for_storm": False,
        "storm_used_as_input": False,
    }
    output.index.name = "delivery_start_utc"
    candidate_prediction_frame.index.name = "delivery_start_utc"
    return KalmanReplayResult(
        predictions=output,
        candidate_predictions=candidate_prediction_frame,
        daily_audit=pd.DataFrame(daily_rows),
        state_audit=pd.DataFrame(state_rows),
        audit=audit,
        future_predictions=future_output,
        future_candidate_predictions=future_candidate_prediction_frame,
        future_audit=future_audit,
    )


def validate_operational_kalman_history(
    statistics: pd.DataFrame,
    *,
    timezone: str,
    delivery_day: str | date | pd.Timestamp,
    upstream_model: str = "residual_corrected",
    evaluation_days: int = 365,
    training_lookback_days: int = 365,
) -> dict[str, Any]:
    """Audit the exact civil rolling-history window without fitting or writes.

    Only D-(evaluation+training) through D-1 is inspected. Older observations
    and the current/future delivery-day actual placeholders are outside this
    contract. UTC timestamps retain both physical hours of an autumn DST fold.
    """

    if evaluation_days != 365:
        raise KalmanResidualError("La vue operationnelle exige 365 jours.")
    lookback = _validate_training_lookback_days(training_lookback_days)
    if lookback is None:
        raise KalmanResidualError("training_lookback_days est obligatoire.")
    try:
        requested_timestamp = pd.Timestamp(delivery_day)
        if pd.isna(requested_timestamp):
            raise ValueError("date absente")
        requested_day = requested_timestamp.date()
    except (TypeError, ValueError) as exc:
        raise KalmanResidualError("delivery_day doit respecter YYYY-MM-DD.") from exc
    required_days = evaluation_days + lookback
    required_start = requested_day - timedelta(days=required_days)
    required_end = requested_day - timedelta(days=1)
    expected = pd.date_range(
        pd.Timestamp(required_start, tz=timezone),
        pd.Timestamp(requested_day, tz=timezone),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    requirement = (
        "Le replay rolling exige "
        f"{lookback} jours de calibration avant les {evaluation_days} jours "
        f"evalues : historique requis du {required_start} au {required_end} "
        f"({required_days} jours civils, {len(expected)} heures physiques, "
        f"{timezone})."
    )
    if not isinstance(statistics, pd.DataFrame):
        raise KalmanResidualError(f"{requirement} Statistics doit etre un DataFrame.")
    if "delivery_start_utc" not in statistics:
        raise KalmanResidualError(f"{requirement} delivery_start_utc est absent.")
    try:
        all_index = pd.DatetimeIndex(
            pd.to_datetime(statistics["delivery_start_utc"], utc=True, errors="raise")
        )
    except (TypeError, ValueError) as exc:
        raise KalmanResidualError(f"{requirement} Timeline UTC invalide.") from exc
    if all_index.hasnans:
        raise KalmanResidualError(f"{requirement} Timeline UTC avec date absente.")
    in_window = (all_index >= expected[0]) & (
        all_index < pd.Timestamp(requested_day, tz=timezone).tz_convert("UTC")
    )
    observed = all_index[in_window]
    window = statistics.loc[in_window]
    local_days = pd.Index(observed.tz_convert(timezone).date)
    available_start = str(local_days.min()) if len(local_days) else None
    available_end = str(local_days.max()) if len(local_days) else None
    available_hours = len(expected.intersection(observed))
    available = (
        f" Disponible dans cette fenetre : {available_start or 'aucun jour'}"
        f" au {available_end or 'aucun jour'}, {len(local_days.unique())} jours, "
        f"{available_hours}/{len(expected)} heures physiques."
    )
    missing = expected.difference(observed)
    unexpected = observed.difference(expected)
    if not observed.equals(expected):
        examples = ", ".join(timestamp.isoformat() for timestamp in missing[:3])
        raise KalmanResidualError(
            requirement + available
            + f" Heures manquantes={len(missing)}, inattendues={len(unexpected)}, "
            f"doublons={int(observed.duplicated().sum())}, "
            f"ordre_chronologique={observed.is_monotonic_increasing}."
            + (f" Exemples manquants UTC : {examples}." if examples else "")
        )
    required_columns = [
        "actual", *(f"{upstream_model}__{quantile}" for quantile in QUANTILES)
    ]
    missing_columns = sorted(set(required_columns).difference(window.columns))
    if missing_columns:
        raise KalmanResidualError(
            requirement + available + f" Colonnes upstream absentes : {missing_columns}."
        )
    numeric = window.loc[:, required_columns].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float))
    if not finite.all():
        invalid_rows, invalid_columns = np.nonzero(~finite)
        examples = ", ".join(
            f"{required_columns[column]}@{observed[row].isoformat()}"
            for row, column in zip(invalid_rows[:3], invalid_columns[:3])
        )
        raise KalmanResidualError(
            requirement + available
            + f" Valeurs non finies sur {len(np.unique(invalid_rows))} heure(s) "
            f"({len(invalid_rows)} cellule(s)); exemples UTC : {examples}."
        )
    quantiles = numeric.iloc[:, 1:].to_numpy(dtype=float)
    crossed = (quantiles[:, 0] > quantiles[:, 1]) | (
        quantiles[:, 1] > quantiles[:, 2]
    )
    if crossed.any():
        examples = ", ".join(timestamp.isoformat() for timestamp in observed[crossed][:3])
        raise KalmanResidualError(
            requirement + available
            + f" Les quantiles upstream se croisent sur {int(crossed.sum())} "
            f"heure(s); exemples UTC : {examples}."
        )
    return {
        "status": "complete",
        "timezone": timezone,
        "delivery_day": str(requested_day),
        "upstream_model": upstream_model,
        "evaluation_days": evaluation_days,
        "training_lookback_days": lookback,
        "required_start_day": str(required_start),
        "required_end_day": str(required_end),
        "available_start_day": available_start,
        "available_end_day": available_end,
        "evaluation_start_day": str(requested_day - timedelta(days=evaluation_days)),
        "evaluation_end_day": str(required_end),
        "required_days": required_days,
        "available_days": len(local_days.unique()),
        "expected_hours": len(expected),
        "available_hours": available_hours,
        "missing_hours": 0,
    }


def build_operational_kalman_view(
    *,
    statistics: pd.DataFrame,
    source_forecast: pd.DataFrame,
    covariates: pd.DataFrame,
    timezone: str,
    delivery_day: str | date | pd.Timestamp,
    config: KalmanResidualConfig | None = None,
    covariate_config: KalmanCovariateConfig | None = None,
    upstream_model: str = "residual_corrected",
    output_model: str = DEFAULT_MODEL_KEY,
    evaluation_days: int = 365,
    minimum_warmup_days: int = 14,
    training_lookback_days: int | None = None,
    rolling_refit_workers: int = 1,
    rolling_refit_cache_dir: str | Path | None = None,
) -> KalmanOperationalView:
    """Materialize one causal live view without mutating its source frames.

    The Statistics snapshot may already contain observations for the requested
    delivery day when an old date is replayed.  They are deliberately removed
    before the state, scalers or governor are built, so the forecast for D can
    use at most delivery days strictly earlier than D.

    ``training_lookback_days=365`` additionally requires 365 complete civil
    days before the first FINAL365 day (roughly 730 days of PIT history total).
    """

    if evaluation_days != 365:
        raise KalmanResidualError("La vue operationnelle exige 365 jours.")
    if minimum_warmup_days < 1:
        raise KalmanResidualError("Le warm-up minimum doit etre positif.")
    try:
        requested_day = pd.Timestamp(delivery_day).date()
    except (TypeError, ValueError) as exc:
        raise KalmanResidualError("delivery_day doit respecter YYYY-MM-DD.") from exc

    selected_training_lookback = _validate_training_lookback_days(
        training_lookback_days
    )
    if selected_training_lookback is not None:
        validate_operational_kalman_history(
            statistics,
            timezone=timezone,
            delivery_day=delivery_day,
            upstream_model=upstream_model,
            evaluation_days=evaluation_days,
            training_lookback_days=selected_training_lookback,
        )
    source_statistics_frame = _timestamped_frame(
        statistics,
        timestamp_column="delivery_start_utc",
        name="statistics",
    )
    source_local_days = pd.Index(
        source_statistics_frame.index.tz_convert(timezone).date
    )
    historical_mask = np.asarray(source_local_days < requested_day, dtype=bool)
    statistics_frame = source_statistics_frame.loc[historical_mask].copy()
    local_days = source_local_days[historical_mask]
    realised_delivery_frame = source_statistics_frame.loc[
        np.asarray(source_local_days == requested_day, dtype=bool)
    ].copy()
    if statistics_frame.empty:
        raise KalmanResidualError(
            "Statistics ne contient aucun jour strictement anterieur au forecast."
        )
    selected_rolling_workers = _validate_rolling_refit_workers(
        rolling_refit_workers
    )
    unique_days = list(local_days.drop_duplicates())
    if len(unique_days) < evaluation_days + minimum_warmup_days:
        raise KalmanResidualError(
            "Le replay requiert 365 jours et un warm-up causal d'au moins "
            f"{minimum_warmup_days} jours."
        )
    if selected_training_lookback is not None:
        required_days = evaluation_days + selected_training_lookback
        # Earlier rows are mathematically unused by every fixed-length
        # rolling origin.  Dropping them also makes the strict covariate
        # completeness check describe exactly the 365+365 operational
        # contract instead of an arbitrary older Statistics prefix.
        retained_mask = np.asarray(
            local_days >= requested_day - timedelta(days=required_days), dtype=bool
        )
        statistics_frame = statistics_frame.loc[retained_mask].copy()
        local_days = local_days[retained_mask]
        unique_days = list(local_days.drop_duplicates())
    evaluation_values = unique_days[-evaluation_days:]
    warmup_values = unique_days[:-evaluation_days]
    evaluation_start = evaluation_values[0]
    evaluation_end = evaluation_values[-1]
    expected_end = requested_day - timedelta(days=1)
    if evaluation_end != expected_end:
        raise KalmanResidualError(
            f"Dernier jour Statistics={evaluation_end}, attendu={expected_end}."
        )
    expected_evaluation = pd.date_range(
        pd.Timestamp(evaluation_start, tz=timezone),
        pd.Timestamp(requested_day, tz=timezone),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    evaluation_mask = np.asarray(
        pd.Index(local_days).isin(set(evaluation_values)), dtype=bool
    )
    observed_evaluation = statistics_frame.index[evaluation_mask]
    if not observed_evaluation.equals(expected_evaluation):
        raise KalmanResidualError(
            "La fenetre FINAL365 n'est pas physiquement complete."
        )
    future_frame = _timestamped_frame(
        source_forecast,
        timestamp_column="delivery_start_utc",
        name="source_forecast",
    )
    expected_future = pd.date_range(
        pd.Timestamp(requested_day, tz=timezone),
        pd.Timestamp(requested_day + timedelta(days=1), tz=timezone),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    if not future_frame.index.equals(expected_future):
        raise KalmanResidualError(
            "Le forecast source ne couvre pas exactement le jour demande."
        )

    required_history = {
        "actual",
        *(f"{upstream_model}__{quantile}" for quantile in QUANTILES),
    }
    missing_history = sorted(required_history.difference(statistics_frame.columns))
    if missing_history:
        raise KalmanResidualError(
            f"Historique upstream incomplet: {missing_history}."
        )
    required_future = {
        *(f"{upstream_model}__{quantile}" for quantile in QUANTILES),
    }
    missing_future = sorted(required_future.difference(future_frame.columns))
    if missing_future:
        raise KalmanResidualError(
            f"Forecast upstream incomplet: {missing_future}."
        )
    if not realised_delivery_frame.empty:
        if not realised_delivery_frame.index.equals(expected_future):
            raise KalmanResidualError(
                "Statistics contient un jour de livraison realise incomplet."
            )
        realised_required = {"actual", *required_future}
        missing_realised = sorted(
            realised_required.difference(realised_delivery_frame.columns)
        )
        if missing_realised:
            raise KalmanResidualError(
                f"Jour realise incomplet dans Statistics: {missing_realised}."
            )
        realised_forecasts = realised_delivery_frame.loc[
            :, sorted(required_future)
        ].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(realised_forecasts.to_numpy(dtype=float)).all():
            raise KalmanResidualError(
                "Le jour de livraison Statistics doit avoir des forecasts finis."
            )
        realised_actual = pd.to_numeric(
            realised_delivery_frame["actual"], errors="coerce"
        ).to_numpy(dtype=float)
        actual_finite = np.isfinite(realised_actual)
        if actual_finite.any() and not actual_finite.all():
            raise KalmanResidualError(
                "Le prix observe du jour de livraison doit etre complet ou "
                "entierement vide dans Statistics."
            )
        for column in sorted(required_future):
            if not np.allclose(
                pd.to_numeric(
                    realised_delivery_frame[column], errors="raise"
                ).to_numpy(dtype=float),
                pd.to_numeric(
                    future_frame[column], errors="raise"
                ).to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-9,
            ):
                raise KalmanResidualError(
                    f"Le forecast realise diverge du forecast emis: {column}."
                )
    residual_signal_columns = ("residual_correction", "chronos2__q50")
    if not any(column in statistics_frame for column in residual_signal_columns):
        raise KalmanResidualError(
            "Statistics doit fournir residual_correction ou chronos2__q50."
        )
    available_future_signals = [
        column for column in residual_signal_columns if column in future_frame
    ]
    if not available_future_signals:
        raise KalmanResidualError(
            "Le forecast doit fournir residual_correction ou chronos2__q50."
        )
    future_signal = pd.to_numeric(
        future_frame[available_future_signals[0]], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(future_signal).all():
        raise KalmanResidualError(
            f"Signal residuel futur incomplet: {available_future_signals[0]}."
        )

    selected_covariate_config = covariate_config or KalmanCovariateConfig()
    try:
        selected_covariate_config.validate()
    except KalmanCovariateError as exc:
        raise KalmanResidualError(str(exc)) from exc
    covariate_frame = _timestamped_frame(
        covariates,
        timestamp_column="timestamp",
        name="covariates_operationnelles",
    )
    missing_covariates = sorted(
        set(selected_covariate_config.input_columns).difference(
            covariate_frame.columns
        )
    )
    if missing_covariates:
        raise KalmanResidualError(
            "Covariables marche requises absentes dans le contrat Kalman: "
            f"{missing_covariates}."
        )
    try:
        materialized_covariates = materialize_kalman_covariates(
            covariate_frame,
            selected_covariate_config,
            timezone=timezone,
        )
    except KalmanCovariateError as exc:
        raise KalmanResidualError(str(exc)) from exc
    future_features = materialized_covariates.reindex(expected_future).loc[
        :, list(selected_covariate_config.feature_columns)
    ]
    if (
        selected_covariate_config.require_future_complete
        and not np.isfinite(future_features.to_numpy(dtype=float)).all()
    ):
        incomplete = [
            str(column)
            for column in future_features
            if not np.isfinite(
                pd.to_numeric(
                    future_features[column], errors="coerce"
                ).to_numpy(dtype=float)
            ).all()
        ]
        raise KalmanResidualError(
            "Les covariables Kalman futures doivent etre completes: "
            f"{incomplete}."
        )
    if selected_covariate_config.history_missing_policy == "complete_trailing":
        historical_features = materialized_covariates.reindex(
            statistics_frame.index
        ).loc[:, list(selected_covariate_config.feature_columns)]
        if not np.isfinite(historical_features.to_numpy(dtype=float)).all():
            incomplete = [
                str(column)
                for column in historical_features
                if not np.isfinite(
                    pd.to_numeric(
                        historical_features[column], errors="coerce"
                    ).to_numpy(dtype=float)
                ).all()
            ]
            raise KalmanResidualError(
                "La politique complete_trailing exige un historique "
                f"operationnel complet: {incomplete}."
            )
    causal_covariates = covariate_frame.loc[
        :, list(selected_covariate_config.input_columns)
    ].reset_index().rename(columns={"delivery_start_utc": "timestamp"})
    optional = ("chronos2__q50", "residual_correction", "forecast_origin_utc")
    history_input = statistics_frame.loc[
        :,
        [
            *sorted(required_history),
            *(column for column in optional if column in statistics_frame),
        ],
    ].reset_index()
    future_input = future_frame.loc[
        :,
        [
            *sorted(required_future),
            *(column for column in optional if column in future_frame),
        ],
    ].reset_index()
    replay = replay_kalman_overlay(
        history_input,
        timezone=timezone,
        upstream_model=upstream_model,
        output_model=output_model,
        evaluation_start_day=str(evaluation_start),
        covariates=causal_covariates,
        future_upstream=future_input,
        future_covariates=causal_covariates,
        training_lookback_days=selected_training_lookback,
        rolling_refit_workers=selected_rolling_workers,
        rolling_refit_cache_dir=rolling_refit_cache_dir,
        config=config,
        covariate_config=selected_covariate_config,
    )
    if not replay.predictions.index.equals(statistics_frame.index):
        raise KalmanResidualError(
            "La timeline Kalman historique diverge de Statistics."
        )
    if not replay.future_predictions.index.equals(expected_future):
        raise KalmanResidualError(
            "La timeline Kalman future diverge du forecast upstream."
        )

    # The causal prefix is required to fit the first rolling origin but is not
    # a report sample.  Publish only the audited FINAL365 support (plus a
    # realised delivery day below when available), which also keeps Storm's
    # report-only pairing independent from the warm-up rows.
    statistics_output_frame = statistics_frame.loc[evaluation_mask].copy()
    for column in replay.predictions:
        statistics_output_frame[column] = replay.predictions.loc[
            statistics_output_frame.index, column
        ].to_numpy()
    if "forecast_origin_utc" in statistics_output_frame:
        statistics_output_frame[f"{output_model}_forecast_origin_utc"] = (
            statistics_output_frame["forecast_origin_utc"]
        )
    backtest_output = statistics_output_frame.reset_index()

    # A historical rerun may already know actual(D).  It must never enter the
    # state used to predict D, but it remains valuable as a post-issuance
    # Statistics observation.  Attach the already-frozen future prediction
    # only after the causal replay has completed.
    if not realised_delivery_frame.empty:
        realised_output = realised_delivery_frame.copy()
        for column in replay.future_predictions:
            realised_output[column] = replay.future_predictions[column].to_numpy()
        realised_origin = (
            "residual_corrected_forecast_origin_utc"
            if "residual_corrected_forecast_origin_utc" in realised_output
            else "forecast_origin_utc"
        )
        if realised_origin in realised_output:
            realised_output[f"{output_model}_forecast_origin_utc"] = (
                realised_output[realised_origin]
            )
        statistics_output_frame = pd.concat(
            [statistics_output_frame, realised_output],
            axis=0,
        ).sort_index()
    statistics_output = statistics_output_frame.reset_index()

    forecast_output = future_frame.reset_index()
    for column in replay.future_predictions:
        forecast_output[column] = replay.future_predictions[column].to_numpy()
    for quantile in QUANTILES:
        forecast_output[quantile] = forecast_output[f"{output_model}__{quantile}"]
    forecast_output["price_eur_mwh"] = forecast_output[f"{output_model}__q50"]
    origin_column = (
        "residual_corrected_forecast_origin_utc"
        if "residual_corrected_forecast_origin_utc" in forecast_output
        else "forecast_origin_utc"
    )
    if origin_column in forecast_output:
        forecast_output[f"{output_model}_forecast_origin_utc"] = forecast_output[
            origin_column
        ]

    return KalmanOperationalView(
        replay=replay,
        statistics=statistics_output,
        backtest=backtest_output,
        forecast=forecast_output,
        evaluation_start_day=evaluation_start,
        evaluation_end_day=evaluation_end,
        evaluation_index=expected_evaluation,
        future_index=expected_future,
    )


__all__ = [
    "DEFAULT_MODEL_KEY",
    "REQUIRED_MARKET_COVARIATES",
    "EXOGENOUS_FILTER_GROUPS",
    "SUPPORTED_FILTER_KINDS",
    "KalmanReplayResult",
    "KalmanOperationalView",
    "KalmanResidualConfig",
    "KalmanResidualError",
    "build_operational_kalman_view",
    "replay_kalman_overlay",
    "validate_operational_kalman_history",
]
