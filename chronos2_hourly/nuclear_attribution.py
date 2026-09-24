"""Report-only attribution of the frozen nuclear-conditioned autonomous model.

Only the last causal residual fit is reconstructed. Historical Chronos replay
and Kalman training are never called. Kalman reports may show this explanation
only with the existing explicit *upstream autonomous* attribution scope.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from . import report_attribution_cache, variable_attribution
from .nuclear_forecast import NUCLEAR_ALIAS, NUCLEAR_KNOWN_COLUMN, QUANTILES, ZONE_TIMEZONES, _digest_frame
from .observation_precision import validate_observation_precision
from chronos2_modular.common import deep_get, set_reproducibility


def _indexed(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    value = frame.copy(deep=True)
    if "delivery_start_utc" in value:
        index = pd.DatetimeIndex(pd.to_datetime(value.pop("delivery_start_utc"), utc=True, errors="raise"))
    else:
        index = value.index
        if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
            raise ValueError(f"{name}: timezone-aware delivery index required.")
        index = index.tz_convert("UTC")
    if index.empty or index.has_duplicates or not index.is_monotonic_increasing or index.isna().any():
        raise ValueError(f"{name}: unique increasing nonempty delivery index required.")
    value.index = index.rename("delivery_start_utc")
    return value


def _quantiles(frame: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    columns = [prefix + name for name in QUANTILES]
    selected = frame.loc[:, columns].apply(pd.to_numeric, errors="raise")
    selected.columns = list(QUANTILES)
    if not np.isfinite(selected.to_numpy(dtype=float)).all():
        raise ValueError("Nuclear attribution requires finite frozen quantiles.")
    if ((selected.q10 > selected.q50) | (selected.q50 > selected.q90)).any():
        raise ValueError("Nuclear attribution refuses crossing quantiles.")
    return selected


def _identity_file(directory: Path, identity: Mapping[str, Any]) -> Path:
    raw = json.dumps(identity, sort_keys=True, ensure_ascii=False, default=str, allow_nan=False).encode("utf-8")
    key = hashlib.sha256(raw).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.json"
    try:
        with path.open("xb") as stream:
            stream.write(raw)
    except FileExistsError:
        if path.read_bytes() != raw:
            raise ValueError("Nuclear attribution input identity is corrupt.")
    return path


def _validate_forecast_copy(directory: Path, *, filename: str, expected: str) -> None:
    forecast = directory / filename
    if not forecast.is_file() or report_attribution_cache.sha256(forecast) != expected:
        raise ValueError("Nuclear attribution frozen forecast copy is missing or divergent.")
    audit = json.loads((directory / variable_attribution.VARIABLE_ATTRIBUTION_AUDIT).read_text(encoding="utf-8"))
    if audit.get("forecast_sha256") != expected:
        raise ValueError("Nuclear attribution audit identifies another forecast.")


def prepare_nuclear_attribution(
    result: Any, data: Any, config: Mapping[str, Any], workdir: str | Path,
    device: str = "auto", threads: int = 4, *,
    runtime_factory: Callable[..., Any] | None = None,
    residual_factory: Callable[[], Any] | None = None,
    feature_factory: Callable[..., Any] | None = None,
    attribution_writer: Callable[..., Any] | None = None,
) -> Path:
    """Explain an already frozen future forecast, caching all report artifacts.

    Dependency injection is reserved for light tests. The normal path loads only
    local Chronos weights and fits one residual corrector on D-365 through D-1.
    A reproduction failure aborts publication; there is no uncorrected fallback.
    """
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer.")
    from run_chronos2_hourly import _feature_inputs, _residual_corrector_factory
    from chronos2_modular.forecasting import load_model

    resolved = copy.deepcopy(dict(config))
    resolved.setdefault("hourly", {}).setdefault("residual_correction", {})["thread_count"] = threads
    frozen_recipe = result.audit.get("residual_recipe")
    if frozen_recipe is not None and frozen_recipe != resolved["hourly"]["residual_correction"]:
        raise ValueError("Nuclear attribution residual recipe differs from the frozen run.")
    local_data = copy.deepcopy(data)
    target, _history_covariates, future_covariates, features = (feature_factory or _feature_inputs)(local_data, resolved)
    history = _indexed(result.raw_history, "raw_history")
    forecast = _indexed(result.source_forecast, "source_forecast")
    zone = str(result.audit["zone"]).upper()
    if zone not in ZONE_TIMEZONES:
        raise ValueError("Nuclear attribution requires a supported zone.")
    timezone = ZONE_TIMEZONES[zone]
    day = pd.Timestamp(result.audit["delivery_day"])
    if day.tzinfo is not None or day != day.normalize():
        raise ValueError("Nuclear attribution delivery day must be a civil date.")
    first = (day - pd.Timedelta(days=365)).tz_localize(timezone).tz_convert("UTC")
    stop = day.tz_localize(timezone).tz_convert("UTC")
    expected_training = pd.date_range(first, stop, freq="h", inclusive="left")
    expected_future = pd.date_range(stop, (day + pd.Timedelta(days=1)).tz_localize(timezone).tz_convert("UTC"), freq="h", inclusive="left")
    if not forecast.index.equals(expected_future) or not future_covariates.index.equals(expected_future):
        raise ValueError("Nuclear attribution future differs from the frozen delivery day.")
    if target.index[-1] != stop - pd.Timedelta(hours=1):
        raise ValueError("Nuclear attribution target must stop before the forecast day.")
    training = history.loc[(history.index >= first) & (history.index < stop)]
    if not training.index.equals(expected_training):
        raise ValueError("Nuclear attribution requires exactly 365 complete prior training days.")
    if not expected_training.append(expected_future).isin(features.index).all():
        raise ValueError("Nuclear attribution features do not cover the frozen training/future hours.")
    if NUCLEAR_KNOWN_COLUMN not in features or not np.isfinite(pd.to_numeric(features.loc[expected_training.append(expected_future), NUCLEAR_KNOWN_COLUMN], errors="coerce")).all():
        raise ValueError("Nuclear attribution requires the complete nuclear forecast feature.")
    frozen_statistics = _indexed(result.residual_statistics, "residual_statistics")
    frozen_training = frozen_statistics.loc[(frozen_statistics.index >= first) & (frozen_statistics.index < stop)]
    if not frozen_training.index.equals(expected_training):
        raise ValueError("Nuclear attribution frozen residual labels require exactly the same 365 training days.")
    # Parquet retains the precise float32 labels used by the original fit;
    # the Chronos checkpoint CSV may reload their short decimal representation
    # as float64. Keep the original training labels rather than subtly changing
    # a second CatBoost fit while reconstructing a report-only explanation.
    actual = pd.to_numeric(frozen_training["actual"], errors="raise")
    checkpoint_actual = pd.to_numeric(training["actual"], errors="raise")
    target_train = pd.to_numeric(target.reindex(expected_training), errors="raise")
    observation_precision = {
        "frozen_labels_vs_target": validate_observation_precision(actual, target_train, name="Nuclear attribution frozen labels / target"),
        "checkpoint_labels_vs_frozen_labels": validate_observation_precision(checkpoint_actual, actual, name="Nuclear attribution checkpoint / frozen labels"),
    }
    base = _quantiles(training)
    frozen_quantile_columns = ["chronos2__" + q for q in QUANTILES]
    quantile_precision: dict[str, Any] = {}
    base_source = "raw_chronos_checkpoint"
    if any(column in frozen_training for column in frozen_quantile_columns):
        if not all(column in frozen_training for column in frozen_quantile_columns):
            raise ValueError("Nuclear attribution frozen raw quantile cache is incomplete.")
        frozen_base = _quantiles(frozen_training, "chronos2__")
        quantile_precision = {
            q: validate_observation_precision(frozen_base[q], base[q], name=f"Nuclear attribution frozen / checkpoint {q}")
            for q in QUANTILES
        }
        base = frozen_base
        base_source = "frozen_residual_cache_chronos2_quantiles"
    future_base = _quantiles(forecast, "chronos2__")
    official = _quantiles(forecast, "residual_corrected__")
    filename = f"forecast_hourly_{zone.lower()}.csv"
    forecast_bytes = forecast.reset_index().to_csv(index=False).encode("utf-8")
    forecast_sha = hashlib.sha256(forecast_bytes).hexdigest()
    identity = {
        "schema_version": 1, "zone": zone, "delivery_day": day.date().isoformat(),
        "config": resolved, "device": str(device), "forecast_sha256": forecast_sha,
        "raw_history": _digest_frame(history), "features": _digest_frame(features),
        "frozen_training_actuals": _digest_frame(actual), "observation_precision": observation_precision,
        "training_raw_quantiles": _digest_frame(base), "training_raw_quantiles_source": base_source,
        "training_raw_quantile_precision": quantile_precision,
        "target": _digest_frame(target), "covariates": _digest_frame(local_data.covariates),
        "model_context_covariates": _digest_frame(local_data.model_context_covariates),
        "known_future_columns": list(local_data.known_future_columns),
        "include_past_prices": True, "residual_training_days": 365,
    }
    workspace = Path(workdir).resolve()
    project = Path(__file__).resolve().parents[1]
    if workspace == project or any(
        workspace == protected or protected in workspace.parents
        for protected in (project / "runs/live", project / "runs/exports", project / "runs/cache")
    ):
        raise ValueError("Nuclear attribution must remain outside live archives and shared caches.")
    identity_path = _identity_file(workspace / "report_only" / "attribution_identities", identity)
    sources = [
        identity_path, Path(__file__), Path(variable_attribution.__file__),
        Path(report_attribution_cache.__file__), project / "run_chronos2_hourly.py",
        project / "chronos2_hourly/observation_precision.py",
        project / "chronos2_hourly/models/residual_corrector.py",
        project / "chronos2_modular/forecasting.py",
    ]
    if not all(path.is_file() for path in sources):
        raise ValueError("Nuclear attribution implementation source is missing.")

    def materialize(destination: Path) -> None:
        set_reproducibility(int(deep_get(resolved, "model.seed", 42)))
        configured_factory, base_model = _residual_corrector_factory(resolved, timezone=timezone)
        selected_factory = residual_factory or configured_factory
        if selected_factory is None or base_model != "chronos2":
            raise ValueError("Nuclear attribution requires the active Chronos residual recipe.")
        corrector = selected_factory()
        if len(training) < int(corrector.min_training_rows):
            raise ValueError("Nuclear attribution residual training history is insufficient.")
        corrector.fit(
            features.loc[expected_training].copy(), actual.copy(), base.copy(),
            base.rename(columns={q: f"chronos2__{q}" for q in QUANTILES}),
        )
        if not any(NUCLEAR_ALIAS in str(name) for name in corrector.feature_columns_):
            raise ValueError("Nuclear attribution residual fit dropped the nuclear feature.")
        fresh_future = features.loc[expected_future].copy()
        fresh_future.index.name = "delivery_start_utc"
        reproduced = corrector.predict(
            fresh_future.copy(), future_base.copy(),
            future_base.rename(columns={q: f"chronos2__{q}" for q in QUANTILES}),
        )
        if not reproduced.index.equals(expected_future) or not np.allclose(
            _quantiles(reproduced), official, atol=1e-3, rtol=0,
        ):
            raise ValueError("Nuclear attribution residual refit does not reproduce the frozen forecast.")
        forecast_path = destination / filename
        forecast_path.write_bytes(forecast_bytes)
        runtime = (runtime_factory(resolved, device, True) if runtime_factory is not None
                   else load_model(report_attribution_cache.local_attribution_model_config(resolved), device, True))
        (attribution_writer or variable_attribution.write_variable_attribution)(
            output_dir=destination, forecast_path=forecast_path, data=local_data,
            runtime=runtime, fresh_future=fresh_future, corrector=corrector,
            official_autonomous=official.copy(), required_covariates=tuple(map(str, local_data.covariates.columns)),
            context_length=int(deep_get(resolved, "model.context_length", 2048)),
            model_batch_size=int(deep_get(resolved, "model.model_batch_size", 128)),
            zone=zone, timezone=timezone, delivery_day=day.date().isoformat(),
            seed=int(deep_get(resolved, "model.seed", 42)), include_past_prices=True,
        )
        audit_path = destination / variable_attribution.VARIABLE_ATTRIBUTION_AUDIT
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["nuclear_report_preparation"] = {
            "residual_refits": 1, "chronos_training_performed": False,
            "historical_replay_performed": False, "kalman_replay_performed": False,
            "training_start_local_day": str(expected_training[0].tz_convert(timezone).date()),
            "training_end_local_day": str(expected_training[-1].tz_convert(timezone).date()),
            "training_rows": len(training), "training_window_days": 365,
            "frozen_forecast_reproduction_checked": True,
            "training_actuals_source": "residual_statistics_parquet_original_fit_labels",
            "observation_precision": observation_precision,
            "training_raw_quantiles_source": base_source,
            "training_raw_quantile_precision": quantile_precision,
        }
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        _validate_forecast_copy(destination, filename=filename, expected=forecast_sha)

    destination = report_attribution_cache.cached_attribution(
        root=workspace / "report_only" / "variable_attribution", sources=sources,
        materialize=materialize,
    )
    _validate_forecast_copy(destination, filename=filename, expected=forecast_sha)
    return destination


__all__ = ["prepare_nuclear_attribution"]
