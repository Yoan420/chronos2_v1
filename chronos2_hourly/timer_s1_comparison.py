"""Auditable comparison helpers for Timer-S1 and the frozen Chronos-2 run.

Timer-S1 is a univariate model.  It cannot consume the native multivariate
and known-future covariates used by Chronos-2 in this repository.  This module
therefore keeps two claims separate:

* a *native* comparison is fair only when both models are target-only;
* a *system* comparison may reuse the exact frozen 40-feature matrix and the
  exact 175-meta-feature residual recipe around each base forecast, but it is
  not a native feature-parity comparison between the two backbones.

The system comparison below reconstructs the current FR residual recipe from
its immutable run artifacts, fits a fresh copy on Timer-S1 residuals, and
evaluates both systems on the same sealed final 365 delivery days.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


QUANTILES: tuple[str, ...] = ("q10", "q50", "q90")
QUANTILE_LEVELS: tuple[float, ...] = (0.1, 0.5, 0.9)
DEFAULT_CALIBRATION_DAYS = 365
DEFAULT_EVALUATION_DAYS = 365
DEFAULT_EXTENDED_DAYS = 223
TIMER_S1_POST_V3_SENSITIVITY_START_LOCAL = "2026-04-10"


class TimerS1ComparisonError(ValueError):
    """Raised when an artifact cannot support the advertised comparison."""


@dataclass(frozen=True)
class SourceProtocol:
    """Frozen current-system artifacts and reconstructed feature matrix."""

    source_run: Path
    timezone: str
    target: pd.Series
    history_features: pd.DataFrame
    future_features: pd.DataFrame
    chronos_extended: pd.DataFrame
    chronos_main: pd.DataFrame
    current_backtest: pd.DataFrame
    calibration_index: pd.DatetimeIndex
    evaluation_index: pd.DatetimeIndex
    feature_names: tuple[str, ...]
    feature_manifest_sha256: str
    expected_meta_features: int
    pit_covariate_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    residual_recipe: Mapping[str, Any] = field(default_factory=dict)
    residual_recipe_sha256: str = ""

    @property
    def full_oof_index(self) -> pd.DatetimeIndex:
        return self.chronos_extended.index.append(self.chronos_main.index)


@dataclass(frozen=True)
class SystemComparisonResult:
    """Results of the frozen-feature Timer-S1 system comparison."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame
    paired_tests: dict[str, Any]
    feature_parity: dict[str, Any]
    corrector_diagnostics: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_target_series_with_timestamp_unit(
    target: pd.Series,
    *,
    timestamp_unit: str,
) -> str:
    """Hash target bytes using an explicit datetime storage unit."""
    if not isinstance(target, pd.Series) or target.empty:
        raise TimerS1ComparisonError("Target hash requires a non-empty Series.")
    if not isinstance(target.index, pd.DatetimeIndex) or target.index.tz is None:
        raise TimerS1ComparisonError(
            "Target hash requires a timezone-aware DatetimeIndex."
        )
    index = target.index.tz_convert("UTC")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise TimerS1ComparisonError(
            "Target hash requires unique, sorted timestamps."
        )
    try:
        index = index.as_unit(timestamp_unit, round_ok=False)
    except (OverflowError, ValueError) as exc:
        raise TimerS1ComparisonError(
            f"Target timestamps cannot be represented exactly as {timestamp_unit}."
        ) from exc
    values = pd.to_numeric(target, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise TimerS1ComparisonError("Target hash rejects non-finite values.")
    index_bytes = np.asarray(index.asi8, dtype="<i8").tobytes(order="C")
    value_bytes = np.asarray(values, dtype="<f8").tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(b"chronos2-timer-s1-target-v1\0")
    digest.update(np.asarray([len(index)], dtype="<u8").tobytes())
    digest.update(index_bytes)
    digest.update(value_bytes)
    return digest.hexdigest()


def sha256_target_series(target: pd.Series) -> str:
    """Hash a target canonically as UTC nanoseconds plus float64 values.

    Pandas 3 may preserve parsed timestamps as ``datetime64[us]`` whereas
    pandas 2 commonly materialises the same timestamps as ``datetime64[ns]``.
    Normalising the storage unit keeps provenance hashes stable across both.
    """

    return _sha256_target_series_with_timestamp_unit(
        target,
        timestamp_unit="ns",
    )


def legacy_target_hashes_by_timestamp_unit(target: pd.Series) -> dict[str, str]:
    """Return hashes emitted by the pre-canonical implementation.

    These candidates support audited reuse of already generated artifacts.
    Only exact, lossless timestamp conversions are admitted.
    """

    canonical = sha256_target_series(target)
    candidates: dict[str, str] = {}
    for unit in ("us", "ms", "s"):
        try:
            candidate = _sha256_target_series_with_timestamp_unit(
                target,
                timestamp_unit=unit,
            )
        except TimerS1ComparisonError:
            continue
        if candidate != canonical:
            candidates[unit] = candidate
    return candidates


def _require_target_only_provenance(
    provenance: Mapping[str, Any],
    protocol: SourceProtocol,
    *,
    model_name: str,
) -> dict[str, Any]:
    """Reject an OOF whose generation sidecar cannot support its label."""

    if not isinstance(provenance, Mapping):
        raise TimerS1ComparisonError(
            f"{model_name}: verified generation provenance is required."
        )
    manifest = dict(provenance)
    expected = {
        "model_name": model_name,
        "forecast_mode": "strict_native_target_only",
        "context_length": 2048,
        "source_feature_manifest_sha256": protocol.feature_manifest_sha256,
        "source_target_sha256": sha256_target_series(protocol.target),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise TimerS1ComparisonError(
                f"{model_name}: provenance {key}={manifest.get(key)!r}, "
                f"expected {value!r}."
            )
    if manifest.get("native_covariates") != []:
        raise TimerS1ComparisonError(
            f"{model_name}: provenance does not certify native_covariates=[]."
        )
    contract = manifest.get("inference_contract")
    if not isinstance(contract, Mapping):
        raise TimerS1ComparisonError(
            f"{model_name}: provenance has no inference_contract."
        )
    required_contract: dict[str, Any]
    if model_name == "timer_s1":
        required_contract = {
            "revin": True,
            "use_cache": False,
            "quantile_indices": {"q10": 0, "q50": 4, "q90": 8},
        }
    elif model_name == "chronos2_target_only":
        required_contract = {
            "cross_learning": False,
            "quantile_levels": [0.1, 0.5, 0.9],
        }
    else:  # pragma: no cover - internal call contract
        raise TimerS1ComparisonError(f"Unsupported provenance model: {model_name}.")
    for key, value in required_contract.items():
        if contract.get(key) != value:
            raise TimerS1ComparisonError(
                f"{model_name}: provenance inference_contract.{key}="
                f"{contract.get(key)!r}, expected {value!r}."
            )
    return manifest


def _read_timestamped(path: Path, *, timestamp_column: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if timestamp_column not in frame:
        raise TimerS1ComparisonError(
            f"{path}: missing timestamp column {timestamp_column!r}."
        )
    index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop(timestamp_column), utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    if index.empty or index.has_duplicates or not index.is_monotonic_increasing:
        raise TimerS1ComparisonError(
            f"{path}: timestamps must be non-empty, unique and sorted."
        )
    frame.index = index
    return frame


def _validate_quantile_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    require_actual: bool,
) -> pd.DataFrame:
    required = [*QUANTILES, "forecast_origin_utc"]
    if require_actual:
        required.append("actual")
    missing = [column for column in required if column not in frame]
    if missing:
        raise TimerS1ComparisonError(f"{name}: missing columns {missing}.")

    result = frame.copy()
    numeric_columns = [
        *QUANTILES,
        *(("actual",) if require_actual else ()),
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if not np.isfinite(result[numeric_columns].to_numpy(dtype=float)).all():
        raise TimerS1ComparisonError(f"{name}: non-finite numeric values.")
    values = result.loc[:, QUANTILES].to_numpy(dtype=float)
    if np.any(values[:, 0] > values[:, 1]) or np.any(values[:, 1] > values[:, 2]):
        raise TimerS1ComparisonError(f"{name}: crossing quantiles.")

    origins = pd.DatetimeIndex(
        pd.to_datetime(result["forecast_origin_utc"], utc=True, errors="raise")
    )
    if not np.asarray(origins < result.index).all():
        raise TimerS1ComparisonError(f"{name}: non-causal forecast origin.")
    result["forecast_origin_utc"] = origins
    return result


def read_quantile_artifact(
    path: str | Path,
    *,
    require_actual: bool = True,
) -> pd.DataFrame:
    """Read and strictly validate one canonical OOF/live artifact."""

    resolved = Path(path).expanduser().resolve()
    return _validate_quantile_frame(
        _read_timestamped(resolved, timestamp_column="delivery_start_utc"),
        name=str(resolved),
        require_actual=require_actual,
    )


def _local_day_groups(
    index: pd.DatetimeIndex,
    *,
    timezone: str,
) -> tuple[pd.Index, list[Any]]:
    if index.tz is None:
        raise TimerS1ComparisonError("A timezone-aware index is required.")
    local_dates = pd.Index(index.tz_convert(timezone).date)
    days = local_dates.unique().tolist()
    if days != pd.date_range(days[0], days[-1], freq="D").date.tolist():
        raise TimerS1ComparisonError("Delivery days are not consecutive.")
    return local_dates, days


def _require_complete_local_days(
    index: pd.DatetimeIndex,
    *,
    timezone: str,
    expected_days: int,
) -> list[Any]:
    from chronos2_hourly.hourly_contract import local_delivery_day_index

    local_dates, days = _local_day_groups(index, timezone=timezone)
    if len(days) != expected_days:
        raise TimerS1ComparisonError(
            f"Expected {expected_days} complete days, received {len(days)}."
        )
    for day in days:
        observed = index[np.asarray(local_dates == day)]
        expected = local_delivery_day_index(day, timezone=timezone)
        if not observed.equals(expected):
            raise TimerS1ComparisonError(
                f"Incomplete or DST-invalid local delivery day: {day}."
            )
    return days


def load_source_protocol(
    source_run: str | Path,
    *,
    timezone: str = "Europe/Paris",
    extended_days: int = DEFAULT_EXTENDED_DAYS,
    calibration_days: int = DEFAULT_CALIBRATION_DAYS,
    evaluation_days: int = DEFAULT_EVALUATION_DAYS,
) -> SourceProtocol:
    """Load the current frozen run and verify its exact feature protocol."""

    root = Path(source_run).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    # These functions are the frozen production recipe's source of truth.
    # Importing lazily keeps ordinary adapter use independent of CatBoost.
    from run_extended_residual_hourly import _load_features

    target, history_features, future_features = _load_features(
        root,
        timezone=timezone,
    )
    chronos_extended = read_quantile_artifact(
        root / "inputs" / "chronos_oof_extended.csv.gz"
    )
    chronos_main = read_quantile_artifact(root / "chronos_oof_hourly.csv.gz")
    _require_complete_local_days(
        chronos_extended.index,
        timezone=timezone,
        expected_days=extended_days,
    )
    main_days = _require_complete_local_days(
        chronos_main.index,
        timezone=timezone,
        expected_days=calibration_days + evaluation_days,
    )
    if chronos_extended.index[-1] + pd.Timedelta(hours=1) != chronos_main.index[0]:
        raise TimerS1ComparisonError(
            "Extended and main OOF windows are not exact hourly continuations."
        )

    local_dates = pd.Index(chronos_main.index.tz_convert(timezone).date)
    calibration_index = chronos_main.index[
        np.asarray(local_dates.isin(main_days[:calibration_days]), dtype=bool)
    ]
    evaluation_index = chronos_main.index[
        np.asarray(local_dates.isin(main_days[calibration_days:]), dtype=bool)
    ]

    current_backtest = _read_timestamped(
        root / "backtest_hourly_oof.csv.gz",
        timestamp_column="delivery_start_utc",
    )
    required_current = {
        "actual",
        *(f"chronos2__{quantile}" for quantile in QUANTILES),
        *(f"residual_corrected__{quantile}" for quantile in QUANTILES),
    }
    missing_current = sorted(required_current.difference(current_backtest.columns))
    if missing_current:
        raise TimerS1ComparisonError(
            f"Current backtest is missing columns {missing_current}."
        )
    if not evaluation_index.isin(current_backtest.index).all():
        raise TimerS1ComparisonError(
            "Current backtest does not cover the sealed evaluation index."
        )

    feature_manifest_path = root / "feature_manifest.csv"
    feature_manifest = pd.read_csv(feature_manifest_path)
    if "feature" not in feature_manifest:
        raise TimerS1ComparisonError("feature_manifest.csv has no feature column.")
    feature_names = tuple(feature_manifest["feature"].astype(str))
    if tuple(history_features.columns.astype(str)) != feature_names:
        raise TimerS1ComparisonError(
            "Reconstructed feature order differs from the frozen manifest."
        )
    forbidden = [name for name in feature_names if "storm" in name.casefold()]
    if forbidden:
        raise TimerS1ComparisonError(f"Forbidden forecast features: {forbidden}.")

    run_manifest_path = root / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if tuple(map(str, run_manifest.get("active_features", ()))) != feature_names:
        raise TimerS1ComparisonError(
            "run_manifest active_features differs from feature_manifest.csv."
        )
    input_diagnostics = run_manifest.get("input_diagnostics")
    if not isinstance(input_diagnostics, Mapping):
        raise TimerS1ComparisonError(
            "run_manifest has no auditable input_diagnostics mapping."
        )
    covariate_diagnostics = input_diagnostics.get("covariates")
    if not isinstance(covariate_diagnostics, Mapping):
        raise TimerS1ComparisonError(
            "run_manifest has no PIT covariate diagnostics."
        )
    expected_pit_aliases = (
        "fr_residual_load_fcst",
        "de_residual_load_fcst",
        "be_residual_load_fcst",
        "nl_residual_load_fcst",
        "es_residual_load_fcst",
    )
    pit_summary: dict[str, Any] = {}
    for alias in expected_pit_aliases:
        details = covariate_diagnostics.get(alias)
        if not isinstance(details, Mapping):
            raise TimerS1ComparisonError(f"Missing PIT diagnostics for {alias}.")
        if details.get("source") != "pit_parquet":
            raise TimerS1ComparisonError(
                f"{alias}: expected source='pit_parquet'."
            )
        violations = {
            key: int(details.get(key, -1))
            for key in (
                "cutoff_violations",
                "snapshot_cutoff_violations",
                "revision_cutoff_violations",
            )
        }
        if any(value != 0 for value in violations.values()):
            raise TimerS1ComparisonError(
                f"{alias}: non-zero PIT cutoff/revision violations: {violations}."
            )
        if str(details.get("forecast_origin_local_time")) != "08:00":
            raise TimerS1ComparisonError(
                f"{alias}: PIT origin is not the frozen 08:00 local cutoff."
            )
        pit_summary[alias] = {
            "source": "pit_parquet",
            "forecast_origin_local_time": "08:00",
            **violations,
        }
    recipe_path = root / "extended_residual_recipe.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    expected_meta_features = int(recipe.get("expected_meta_features", -1))
    if expected_meta_features < 1:
        raise TimerS1ComparisonError(
            "The residual recipe has no valid expected_meta_features."
        )

    target = target.copy()
    target.index = target.index.tz_convert("UTC")
    target.index.name = "delivery_start_utc"
    expected_target = target.loc[chronos_extended.index.append(chronos_main.index)]
    artifact_actual = pd.concat(
        [chronos_extended["actual"], chronos_main["actual"]]
    )
    if not np.allclose(
        expected_target.to_numpy(dtype=float),
        artifact_actual.to_numpy(dtype=float),
        rtol=0.0,
        atol=5e-5,
    ):
        raise TimerS1ComparisonError(
            "Frozen Chronos artifacts differ from the canonical target."
        )
    current_eval = current_backtest.loc[evaluation_index]
    frozen_eval = chronos_main.loc[evaluation_index]
    for quantile in QUANTILES:
        current_values = pd.to_numeric(
            current_eval[f"chronos2__{quantile}"], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.allclose(
            current_values,
            frozen_eval[quantile].to_numpy(dtype=float),
            rtol=0.0,
            atol=5e-5,
        ):
            raise TimerS1ComparisonError(
                f"Current backtest and frozen Chronos OOF differ for {quantile}."
            )
    current_actual = pd.to_numeric(
        current_eval["actual"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.allclose(
        current_actual,
        target.loc[evaluation_index].to_numpy(dtype=float),
        rtol=0.0,
        atol=5e-5,
    ):
        raise TimerS1ComparisonError(
            "Current backtest actual differs from the canonical target."
        )

    return SourceProtocol(
        source_run=root,
        timezone=timezone,
        target=target,
        history_features=history_features,
        future_features=future_features,
        chronos_extended=chronos_extended,
        chronos_main=chronos_main,
        current_backtest=current_backtest,
        calibration_index=calibration_index,
        evaluation_index=evaluation_index,
        feature_names=feature_names,
        feature_manifest_sha256=sha256_file(feature_manifest_path),
        expected_meta_features=expected_meta_features,
        pit_covariate_diagnostics=pit_summary,
        residual_recipe=recipe,
        residual_recipe_sha256=sha256_file(recipe_path),
    )


def validate_timer_oof(
    timer_oof: pd.DataFrame,
    protocol: SourceProtocol,
) -> pd.DataFrame:
    """Require exact Timer coverage, origins and targets for EXT+CAL+FINAL."""

    timer = _validate_quantile_frame(
        timer_oof,
        name="timer_s1_oof",
        require_actual=True,
    )
    expected_index = protocol.full_oof_index
    if not timer.index.equals(expected_index):
        missing = expected_index.difference(timer.index)
        unexpected = timer.index.difference(expected_index)
        raise TimerS1ComparisonError(
            "Timer-S1 OOF does not exactly cover EXT+CAL+FINAL: "
            f"missing={len(missing)}, unexpected={len(unexpected)}."
        )
    expected_actual = protocol.target.loc[expected_index].to_numpy(dtype=float)
    if not np.allclose(
        timer["actual"].to_numpy(dtype=float),
        expected_actual,
        rtol=0.0,
        atol=5e-5,
    ):
        raise TimerS1ComparisonError(
            "Timer-S1 actual values differ from the canonical target."
        )

    expected_origins = pd.Series(index=expected_index, dtype="datetime64[ns, UTC]")
    # The frozen Chronos artifacts carry the trusted D-1 08:00 origins.  Timer
    # must use the same origins, not merely any timestamp before delivery.
    frozen = pd.concat(
        [
            protocol.chronos_extended["forecast_origin_utc"],
            protocol.chronos_main["forecast_origin_utc"],
        ]
    )
    expected_origins.loc[:] = pd.to_datetime(frozen, utc=True).to_numpy()
    observed = pd.DatetimeIndex(
        pd.to_datetime(timer["forecast_origin_utc"], utc=True)
    )
    expected = pd.DatetimeIndex(expected_origins)
    if not observed.equals(expected):
        raise TimerS1ComparisonError(
            "Timer-S1 forecast origins differ from the frozen Chronos origins."
        )
    return timer


def _prediction_metrics(
    actual: pd.Series,
    predictions: pd.DataFrame,
    *,
    model: str,
    stage: str,
    native_information_parity: bool,
) -> dict[str, Any]:
    values = predictions.loc[:, QUANTILES].to_numpy(dtype=float)
    truth = actual.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(truth).all():
        raise TimerS1ComparisonError(f"{model}: non-finite evaluation values.")
    errors = values[:, 1] - truth
    row: dict[str, Any] = {
        "model": model,
        "stage": stage,
        "native_information_parity": bool(native_information_parity),
        "n_scored": int(len(truth)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "bias": float(np.mean(errors)),
        "coverage_80": float(np.mean((truth >= values[:, 0]) & (truth <= values[:, 2]))),
        "mean_width_80": float(np.mean(values[:, 2] - values[:, 0])),
    }
    pinballs: list[float] = []
    for column, level, position in zip(QUANTILES, QUANTILE_LEVELS, range(3)):
        residual = truth - values[:, position]
        loss = np.maximum(level * residual, (level - 1.0) * residual)
        score = float(np.mean(loss))
        row[f"pinball_{column}"] = score
        pinballs.append(score)
    row["mean_pinball_q10_q50_q90"] = float(np.mean(pinballs))
    return row


def paired_daily_mae_bootstrap(
    actual: pd.Series,
    baseline: pd.Series,
    candidate: pd.Series,
    *,
    timezone: str,
    samples: int = 20_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired delivery-day bootstrap; delta is candidate minus baseline."""

    if samples < 1:
        raise ValueError("samples must be >= 1.")
    if not (
        actual.index.equals(baseline.index)
        and actual.index.equals(candidate.index)
    ):
        raise TimerS1ComparisonError("Paired bootstrap inputs are not aligned.")
    local_days = pd.Index(actual.index.tz_convert(timezone).date)
    frame = pd.DataFrame(
        {
            "baseline_abs_error": np.abs(
                baseline.to_numpy(dtype=float) - actual.to_numpy(dtype=float)
            ),
            "candidate_abs_error": np.abs(
                candidate.to_numpy(dtype=float) - actual.to_numpy(dtype=float)
            ),
            "local_day": local_days,
        },
        index=actual.index,
    )
    daily = frame.groupby("local_day", sort=True)[
        ["baseline_abs_error", "candidate_abs_error"]
    ].mean()
    differences = (
        daily["candidate_abs_error"] - daily["baseline_abs_error"]
    ).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=float)
    # Chunking avoids a large samples x days allocation on CPU-only hosts.
    chunk_size = 1_000
    for start in range(0, samples, chunk_size):
        size = min(chunk_size, samples - start)
        positions = rng.integers(0, len(differences), size=(size, len(differences)))
        bootstrap[start : start + size] = differences[positions].mean(axis=1)
    delta = float(differences.mean())
    baseline_mae = float(daily["baseline_abs_error"].mean())
    candidate_mae = float(daily["candidate_abs_error"].mean())
    return {
        "estimator": "equal_weight_daily_mae",
        "delta_definition": (
            "candidate_minus_baseline_equal_weight_daily_mae"
        ),
        "n_delivery_days": int(len(daily)),
        "bootstrap_samples": int(samples),
        "seed": int(seed),
        "baseline_mae": baseline_mae,
        "candidate_mae": candidate_mae,
        "delta_mae": delta,
        "relative_improvement": (
            float(-delta / baseline_mae) if baseline_mae else None
        ),
        "ci95_delta_mae": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "bootstrap_fraction_delta_below_zero": float(
            np.mean(bootstrap < 0.0)
        ),
    }


def _timer_internal_base(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Alias Timer as Chronos internally to preserve the exact meta schema.

    The alias is deliberately confined to the frozen corrector's internal
    feature builder.  User-facing artifacts always identify Timer-S1.
    """

    base = frame.loc[:, QUANTILES].copy()
    experts = base.rename(
        columns={quantile: f"chronos2__{quantile}" for quantile in QUANTILES}
    )
    return base, experts


def _validate_corrector_recipe(corrector: Any, protocol: SourceProtocol) -> None:
    """Prove the mutable factory still matches the frozen recipe contract."""

    recipe = protocol.residual_recipe
    if not isinstance(recipe, Mapping) or not recipe:
        raise TimerS1ComparisonError("Frozen residual recipe is unavailable.")
    expected_weights = {
        str(name): float(value)
        for name, value in dict(recipe.get("weights", {})).items()
    }
    observed_weights = {
        str(name): float(value)
        for name, value in dict(getattr(corrector, "weights", {})).items()
    }
    if observed_weights != expected_weights:
        raise TimerS1ComparisonError(
            "Mutable corrector factory differs from frozen blend weights."
        )
    expected_clip = float(recipe.get("final_clip_eur_mwh"))
    if not np.isclose(
        float(getattr(corrector, "max_abs_correction", np.nan)),
        expected_clip,
        rtol=0.0,
        atol=0.0,
    ):
        raise TimerS1ComparisonError(
            "Mutable corrector factory differs from frozen correction clip."
        )
    components = getattr(corrector, "components", {})
    if set(components) != {"cat_v1", "hgb31"}:
        raise TimerS1ComparisonError(
            "Mutable corrector factory has unexpected components."
        )

    builder_options = {
        "timezone": protocol.timezone,
        "include_calendar": True,
        "include_rich_calendar": True,
        "rich_calendar_countries": ("FR", "DE", "BE", "ES", "NL"),
        "rich_calendar_primary_country": "FR",
        "include_daily_profiles": True,
        "include_fundamental_interactions": False,
        "include_missing_indicators": False,
        "exclude_historical_prices": True,
        "exclude_day_of_year": True,
    }
    for component in components.values():
        if getattr(component, "feature_builder_options", None) != builder_options:
            raise TimerS1ComparisonError(
                "Mutable corrector factory differs from frozen feature-builder options."
            )
        if getattr(component, "max_abs_correction", "unexpected") is not None:
            raise TimerS1ComparisonError(
                "Component-level clipping differs from the frozen recipe."
            )

    cat = components["cat_v1"]
    hgb = components["hgb31"]
    cat_recipe = dict(recipe.get("components", {}).get("cat_v1", {}))
    hgb_recipe = dict(recipe.get("components", {}).get("hgb31", {}))
    cat_expected = {
        "backend": "catboost",
        "iterations": int(cat_recipe.get("iterations", -1)),
        "depth": int(cat_recipe.get("depth", -1)),
        "learning_rate": float(cat_recipe.get("learning_rate", np.nan)),
        "l2_leaf_reg": float(cat_recipe.get("l2_leaf_reg", np.nan)),
        "random_state": int(cat_recipe.get("random_seed", -1)),
    }
    hgb_expected = {
        "backend": "sklearn",
        "iterations": int(hgb_recipe.get("max_iter", -1)),
        "depth": 5,  # 2**5 - 1 == frozen max_leaf_nodes=31
        "learning_rate": float(hgb_recipe.get("learning_rate", np.nan)),
        "l2_leaf_reg": float(hgb_recipe.get("l2_regularization", np.nan)),
        "min_samples_leaf": int(hgb_recipe.get("min_samples_leaf", -1)),
        "sklearn_early_stopping": bool(hgb_recipe.get("early_stopping")),
        "random_state": int(hgb_recipe.get("random_state", -1)),
    }
    for label, component, expected in (
        ("cat_v1", cat, cat_expected),
        ("hgb31", hgb, hgb_expected),
    ):
        for attribute, expected_value in expected.items():
            observed = getattr(component, attribute, None)
            if isinstance(expected_value, float):
                matches = bool(
                    np.isclose(
                        float(observed),
                        expected_value,
                        rtol=0.0,
                        atol=0.0,
                    )
                )
            else:
                matches = observed == expected_value
            if not matches:
                raise TimerS1ComparisonError(
                    f"Mutable {label}.{attribute}={observed!r}, expected "
                    f"frozen value {expected_value!r}."
                )


def _fit_frozen_corrector(
    protocol: SourceProtocol,
    *,
    extended: pd.DataFrame,
    main: pd.DataFrame,
    threads: int,
) -> tuple[pd.DataFrame, Any]:
    """Fit one fresh copy of the frozen recipe and score the sealed window.

    Both Chronos-2 and Timer-S1 must pass through this exact function.  Using
    the already-serialized Chronos corrected forecast against a newly fitted
    Timer corrector would leave a small software/runtime asymmetry in what is
    intended to be the paired system comparison.
    """

    from run_extended_residual_hourly import _new_corrector

    fit_frame = pd.concat(
        [extended, main.loc[protocol.calibration_index]],
        axis=0,
    )
    fit_index = protocol.chronos_extended.index.append(protocol.calibration_index)
    if not fit_frame.index.equals(fit_index):
        raise TimerS1ComparisonError(
            "Residual-corrector training rows differ from EXT+CAL."
        )
    fit_base, fit_experts = _timer_internal_base(fit_frame)
    fit_target = protocol.target.loc[fit_index]
    fit_features = protocol.history_features.loc[fit_index]
    if not (
        fit_features.index.equals(fit_target.index)
        and fit_features.index.equals(fit_base.index)
        and fit_features.index.equals(fit_experts.index)
    ):
        raise TimerS1ComparisonError("Residual training inputs are misaligned.")

    corrector = _new_corrector(
        threads=int(threads),
        timezone=protocol.timezone,
        primary_country="FR",
    )
    _validate_corrector_recipe(corrector, protocol)
    corrector.fit(fit_features, fit_target, fit_base, fit_experts)

    evaluation_frame = main.loc[protocol.evaluation_index]
    evaluation_base, evaluation_experts = _timer_internal_base(evaluation_frame)
    corrected = corrector.predict(
        protocol.history_features.loc[protocol.evaluation_index],
        evaluation_base,
        evaluation_experts,
    )
    corrected.index = protocol.evaluation_index
    return corrected, corrector


def compare_same_downstream_features(
    protocol: SourceProtocol,
    timer_oof: pd.DataFrame,
    *,
    timer_provenance: Mapping[str, Any],
    threads: int = -1,
    bootstrap_samples: int = 20_000,
    seed: int = 42,
) -> SystemComparisonResult:
    """Retrain the exact current residual recipe around Timer-S1 forecasts."""

    _require_target_only_provenance(
        timer_provenance,
        protocol,
        model_name="timer_s1",
    )
    timer = validate_timer_oof(timer_oof, protocol)
    timer_extended = timer.loc[protocol.chronos_extended.index]
    timer_main = timer.loc[protocol.chronos_main.index]

    corrected_timer, timer_corrector = _fit_frozen_corrector(
        protocol,
        extended=timer_extended,
        main=timer_main,
        threads=threads,
    )
    corrected_chronos, chronos_corrector = _fit_frozen_corrector(
        protocol,
        extended=protocol.chronos_extended,
        main=protocol.chronos_main,
        threads=threads,
    )

    current = protocol.current_backtest.loc[protocol.evaluation_index]
    evaluation_timer = timer_main.loc[protocol.evaluation_index]
    actual = protocol.target.loc[protocol.evaluation_index]
    predictions = pd.DataFrame(index=protocol.evaluation_index)
    for quantile in QUANTILES:
        predictions[f"chronos2_native__{quantile}"] = pd.to_numeric(
            current[f"chronos2__{quantile}"], errors="coerce"
        )
        predictions[f"timer_s1_native__{quantile}"] = evaluation_timer[quantile]
        predictions[f"chronos2_refit_same_corrector__{quantile}"] = (
            corrected_chronos[quantile]
        )
        predictions[f"timer_s1_refit_same_corrector__{quantile}"] = (
            corrected_timer[quantile]
        )
        predictions[f"chronos2_frozen_current_corrected__{quantile}"] = pd.to_numeric(
            current[f"residual_corrected__{quantile}"], errors="coerce"
        )
    predictions["actual"] = actual
    predictions.index.name = "delivery_start_utc"

    model_specs = (
        ("chronos2_native", "native", False),
        ("timer_s1_native", "native", False),
        ("chronos2_refit_same_corrector", "same_downstream_features", False),
        ("timer_s1_refit_same_corrector", "same_downstream_features", False),
        (
            "chronos2_frozen_current_corrected",
            "frozen_current_reference_only",
            False,
        ),
    )
    metric_rows: list[dict[str, Any]] = []
    for model, stage, parity in model_specs:
        model_frame = predictions[
            [f"{model}__{quantile}" for quantile in QUANTILES]
        ].copy()
        model_frame.columns = QUANTILES
        metric_rows.append(
            _prediction_metrics(
                actual,
                model_frame,
                model=model,
                stage=stage,
                native_information_parity=parity,
            )
        )
    metrics = pd.DataFrame(metric_rows)

    paired_tests = {
        "native_system_capability_not_information_parity": paired_daily_mae_bootstrap(
            actual,
            predictions["chronos2_native__q50"],
            predictions["timer_s1_native__q50"],
            timezone=protocol.timezone,
            samples=bootstrap_samples,
            seed=seed,
        ),
        "same_downstream_features_not_native_information_parity": (
            paired_daily_mae_bootstrap(
                actual,
                predictions["chronos2_refit_same_corrector__q50"],
                predictions["timer_s1_refit_same_corrector__q50"],
                timezone=protocol.timezone,
                samples=bootstrap_samples,
                seed=seed,
            )
        ),
    }

    timer_component = timer_corrector.components_.get("cat_v1")
    chronos_component = chronos_corrector.components_.get("cat_v1")
    timer_meta_names = tuple(getattr(timer_component, "feature_columns_", ()))
    chronos_meta_names = tuple(getattr(chronos_component, "feature_columns_", ()))
    if len(timer_meta_names) != protocol.expected_meta_features:
        raise TimerS1ComparisonError(
            "Timer corrector meta-feature count differs from the frozen recipe: "
            f"{len(timer_meta_names)} != {protocol.expected_meta_features}."
        )
    current_importance = pd.read_csv(
        protocol.source_run / "residual_feature_importance.csv"
    )
    current_meta_names = set(current_importance["feature"].astype(str))
    if set(map(str, timer_meta_names)) != current_meta_names:
        raise TimerS1ComparisonError(
            "Timer and current residual correctors do not share the same meta schema."
        )
    if timer_meta_names != chronos_meta_names:
        raise TimerS1ComparisonError(
            "Fresh Timer and Chronos correctors do not share the same ordered "
            "meta-feature schema."
        )

    feature_parity = {
        "claim": "same_downstream_feature_pipeline",
        "native_backbone_information_parity": False,
        "native_backbone_reason": (
            "Timer-S1 is univariate and has no native exogenous-covariate API; "
            "Chronos-2 uses target plus 17 context covariates and 12 known-future "
            "covariates in the current run."
        ),
        "strict_native_comparison_required": "target_only_for_both_models",
        "downstream_feature_manifest_exact": True,
        "downstream_feature_count": len(protocol.feature_names),
        "downstream_feature_names": list(protocol.feature_names),
        "downstream_feature_manifest_sha256": protocol.feature_manifest_sha256,
        "residual_meta_schema_exact": True,
        "residual_meta_feature_count": len(timer_meta_names),
        "frozen_residual_recipe_validated": True,
        "frozen_residual_recipe_sha256": protocol.residual_recipe_sha256,
        "both_downstream_correctors_refitted_by_same_function": True,
        "frozen_current_forecast_is_reference_only": True,
        "same_context_length_required": True,
        "same_delivery_origins_verified": True,
        "same_evaluation_rows_verified": True,
        "timer_features_are_never_stacked_as_independent_batch_series": True,
        "pit_covariate_provenance_verified": bool(
            protocol.pit_covariate_diagnostics
        ),
        "pit_covariate_diagnostics": dict(protocol.pit_covariate_diagnostics),
        "pretraining_contamination_not_ruled_out_on_primary_holdout": True,
        "post_publication_sensitivity_start_local": (
            TIMER_S1_POST_V3_SENSITIVITY_START_LOCAL
        ),
    }
    diagnostics = {
        "timer_s1": timer_corrector.diagnostics(),
        "chronos2_refit": chronos_corrector.diagnostics(),
    }
    return SystemComparisonResult(
        predictions=predictions,
        metrics=metrics,
        paired_tests=paired_tests,
        feature_parity=feature_parity,
        corrector_diagnostics=diagnostics,
    )


def validate_chronos_target_only_oof(
    chronos_target_only_oof: pd.DataFrame,
    protocol: SourceProtocol,
) -> pd.DataFrame:
    """Validate the target-only Chronos denominator against frozen plans."""

    chronos = _validate_quantile_frame(
        chronos_target_only_oof,
        name="chronos2_target_only_oof",
        require_actual=True,
    )
    if not chronos.index.equals(protocol.full_oof_index):
        raise TimerS1ComparisonError(
            "Target-only Chronos OOF does not exactly cover EXT+CAL+FINAL."
        )
    expected_actual = protocol.target.loc[chronos.index].to_numpy(dtype=float)
    if not np.allclose(
        chronos["actual"].to_numpy(dtype=float),
        expected_actual,
        rtol=0.0,
        atol=5e-5,
    ):
        raise TimerS1ComparisonError(
            "Target-only Chronos actual values differ from the canonical target."
        )
    frozen_origins = pd.DatetimeIndex(
        pd.to_datetime(
            pd.concat(
                [
                    protocol.chronos_extended["forecast_origin_utc"],
                    protocol.chronos_main["forecast_origin_utc"],
                ]
            ),
            utc=True,
        )
    )
    observed_origins = pd.DatetimeIndex(
        pd.to_datetime(chronos["forecast_origin_utc"], utc=True)
    )
    if not observed_origins.equals(frozen_origins):
        raise TimerS1ComparisonError(
            "Target-only Chronos origins differ from the frozen protocol."
        )
    return chronos


def compare_target_only_native(
    protocol: SourceProtocol,
    *,
    chronos_target_only_oof: pd.DataFrame,
    timer_oof: pd.DataFrame,
    chronos_provenance: Mapping[str, Any],
    timer_provenance: Mapping[str, Any],
    bootstrap_samples: int = 20_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare both official backbones on the strict target-only denominator."""

    _require_target_only_provenance(
        timer_provenance,
        protocol,
        model_name="timer_s1",
    )
    _require_target_only_provenance(
        chronos_provenance,
        protocol,
        model_name="chronos2_target_only",
    )
    timer = validate_timer_oof(timer_oof, protocol)
    chronos = validate_chronos_target_only_oof(
        chronos_target_only_oof,
        protocol,
    )

    index = protocol.evaluation_index
    actual = protocol.target.loc[index]
    chronos_eval = chronos.loc[index, QUANTILES]
    timer_eval = timer.loc[index, QUANTILES]
    metrics = pd.DataFrame(
        [
            _prediction_metrics(
                actual,
                chronos_eval,
                model="chronos2_target_only",
                stage="strict_native_target_only",
                native_information_parity=True,
            ),
            _prediction_metrics(
                actual,
                timer_eval,
                model="timer_s1_target_only",
                stage="strict_native_target_only",
                native_information_parity=True,
            ),
        ]
    )
    paired = paired_daily_mae_bootstrap(
        actual,
        chronos_eval["q50"],
        timer_eval["q50"],
        timezone=protocol.timezone,
        samples=bootstrap_samples,
        seed=seed,
    )
    paired["comparison_claim"] = "strict_native_target_only_information_parity"
    paired["pretraining_contamination_not_ruled_out_on_primary_holdout"] = True
    paired["post_publication_sensitivity_start_local"] = (
        TIMER_S1_POST_V3_SENSITIVITY_START_LOCAL
    )
    return metrics, paired


def compare_target_only_same_downstream_features(
    protocol: SourceProtocol,
    *,
    chronos_target_only_oof: pd.DataFrame,
    timer_oof: pd.DataFrame,
    chronos_provenance: Mapping[str, Any],
    timer_provenance: Mapping[str, Any],
    threads: int = -1,
    bootstrap_samples: int = 20_000,
    seed: int = 42,
) -> SystemComparisonResult:
    """Compare equal target-only backbones plus the exact same 40/175 recipe.

    This is the strongest apples-to-apples comparison available for Timer-S1:
    both foundation models receive the same target history, while two fresh
    copies of the frozen residual recipe receive the exact same feature matrix,
    split, hyperparameters and seed.
    """

    _require_target_only_provenance(
        timer_provenance,
        protocol,
        model_name="timer_s1",
    )
    _require_target_only_provenance(
        chronos_provenance,
        protocol,
        model_name="chronos2_target_only",
    )
    timer = validate_timer_oof(timer_oof, protocol)
    chronos = validate_chronos_target_only_oof(
        chronos_target_only_oof,
        protocol,
    )
    extended_index = protocol.chronos_extended.index
    main_index = protocol.chronos_main.index
    corrected_timer, timer_corrector = _fit_frozen_corrector(
        protocol,
        extended=timer.loc[extended_index],
        main=timer.loc[main_index],
        threads=threads,
    )
    corrected_chronos, chronos_corrector = _fit_frozen_corrector(
        protocol,
        extended=chronos.loc[extended_index],
        main=chronos.loc[main_index],
        threads=threads,
    )

    index = protocol.evaluation_index
    actual = protocol.target.loc[index]
    predictions = pd.DataFrame(index=index)
    for quantile in QUANTILES:
        predictions[f"chronos2_target_only_native__{quantile}"] = chronos.loc[
            index, quantile
        ]
        predictions[f"timer_s1_target_only_native__{quantile}"] = timer.loc[
            index, quantile
        ]
        predictions[f"chronos2_target_only_same_corrector__{quantile}"] = (
            corrected_chronos[quantile]
        )
        predictions[f"timer_s1_target_only_same_corrector__{quantile}"] = (
            corrected_timer[quantile]
        )
    predictions["actual"] = actual
    predictions.index.name = "delivery_start_utc"

    specs = (
        ("chronos2_target_only_native", "strict_native_target_only"),
        ("timer_s1_target_only_native", "strict_native_target_only"),
        (
            "chronos2_target_only_same_corrector",
            "strict_target_only_same_downstream_features",
        ),
        (
            "timer_s1_target_only_same_corrector",
            "strict_target_only_same_downstream_features",
        ),
    )
    metric_rows: list[dict[str, Any]] = []
    for model, stage in specs:
        model_frame = predictions[
            [f"{model}__{quantile}" for quantile in QUANTILES]
        ].copy()
        model_frame.columns = QUANTILES
        metric_rows.append(
            _prediction_metrics(
                actual,
                model_frame,
                model=model,
                stage=stage,
                native_information_parity=True,
            )
        )
    metrics = pd.DataFrame(metric_rows)
    paired_tests = {
        "strict_native_target_only": paired_daily_mae_bootstrap(
            actual,
            predictions["chronos2_target_only_native__q50"],
            predictions["timer_s1_target_only_native__q50"],
            timezone=protocol.timezone,
            samples=bootstrap_samples,
            seed=seed,
        ),
        "strict_target_only_same_downstream_features": paired_daily_mae_bootstrap(
            actual,
            predictions["chronos2_target_only_same_corrector__q50"],
            predictions["timer_s1_target_only_same_corrector__q50"],
            timezone=protocol.timezone,
            samples=bootstrap_samples,
            seed=seed,
        ),
    }

    timer_component = timer_corrector.components_.get("cat_v1")
    chronos_component = chronos_corrector.components_.get("cat_v1")
    timer_meta_names = tuple(getattr(timer_component, "feature_columns_", ()))
    chronos_meta_names = tuple(getattr(chronos_component, "feature_columns_", ()))
    if len(timer_meta_names) != protocol.expected_meta_features:
        raise TimerS1ComparisonError(
            "Target-only corrector meta-feature count differs from the recipe."
        )
    if timer_meta_names != chronos_meta_names:
        raise TimerS1ComparisonError(
            "Target-only correctors do not share an ordered meta-feature schema."
        )
    current_importance = pd.read_csv(
        protocol.source_run / "residual_feature_importance.csv"
    )
    if set(map(str, timer_meta_names)) != set(
        current_importance["feature"].astype(str)
    ):
        raise TimerS1ComparisonError(
            "Target-only correctors differ from the frozen 175-feature schema."
        )

    feature_parity = {
        "claim": "strict_target_only_and_same_downstream_feature_pipeline",
        "native_backbone_information_parity": True,
        "native_inputs": "same_target_history_only",
        "context_length_required": 2048,
        "downstream_feature_manifest_exact": True,
        "downstream_feature_count": len(protocol.feature_names),
        "downstream_feature_names": list(protocol.feature_names),
        "downstream_feature_manifest_sha256": protocol.feature_manifest_sha256,
        "residual_meta_schema_exact": True,
        "residual_meta_feature_count": len(timer_meta_names),
        "frozen_residual_recipe_validated": True,
        "frozen_residual_recipe_sha256": protocol.residual_recipe_sha256,
        "same_delivery_origins_verified": True,
        "same_evaluation_rows_verified": True,
        "both_downstream_correctors_refitted_by_same_function": True,
        "pit_covariate_provenance_verified": bool(
            protocol.pit_covariate_diagnostics
        ),
        "pit_covariate_diagnostics": dict(protocol.pit_covariate_diagnostics),
        "pretraining_contamination_not_ruled_out_on_primary_holdout": True,
        "post_publication_sensitivity_start_local": (
            TIMER_S1_POST_V3_SENSITIVITY_START_LOCAL
        ),
    }
    return SystemComparisonResult(
        predictions=predictions,
        metrics=metrics,
        paired_tests=paired_tests,
        feature_parity=feature_parity,
        corrector_diagnostics={
            "timer_s1": timer_corrector.diagnostics(),
            "chronos2_target_only": chronos_corrector.diagnostics(),
        },
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, pd.Timedelta, Path)):
        return str(value)
    return value


__all__ = [
    "DEFAULT_CALIBRATION_DAYS",
    "DEFAULT_EVALUATION_DAYS",
    "DEFAULT_EXTENDED_DAYS",
    "TIMER_S1_POST_V3_SENSITIVITY_START_LOCAL",
    "QUANTILES",
    "SourceProtocol",
    "SystemComparisonResult",
    "TimerS1ComparisonError",
    "compare_same_downstream_features",
    "compare_target_only_native",
    "compare_target_only_same_downstream_features",
    "json_safe",
    "load_source_protocol",
    "paired_daily_mae_bootstrap",
    "read_quantile_artifact",
    "sha256_file",
    "sha256_target_series",
    "validate_timer_oof",
    "validate_chronos_target_only_oof",
]
