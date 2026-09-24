from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

MODEL_ID = "amazon/chronos-2"
MODEL_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
CONTEXT_LENGTH = 2048
MAX_INTERNAL_GAP_HOURS = 6
DELIVERY_TIMEZONE = "Europe/Paris"
QUANTILE_LEVELS = (0.1, 0.5, 0.9)
SCHEMA_VERSION = 1

COUNTRY_OBSERVED_SERIES: Mapping[str, str] = {
    country: f"power.{country}.residual.load.entsoe.hourly.gw.obs"
    for country in ("fr", "de", "be", "nl", "es")
}
COUNTRY_ALIASES: Mapping[str, str] = {
    country: f"{country}_residual_load_fcst"
    for country in COUNTRY_OBSERVED_SERIES
}
EXPECTED_ALIASES = tuple(COUNTRY_ALIASES.values())
ARCHIVED_BUNDLE_DIRECTORY = "residual_load_bundle"
ARCHIVED_BUNDLE_MANIFEST = f"{ARCHIVED_BUNDLE_DIRECTORY}/manifest.json"
SEALED_SATURN_CONTROL_DIRECTORY = "sealed_saturn_control"
SEALED_SATURN_ALIGNED_INPUTS = "inputs/aligned_inputs.csv.gz"
SEALED_SATURN_MODEL_COVARIATES = (
    "inputs/model_covariates_with_future.csv.gz"
)
SEALED_SATURN_PRIMARY = "inputs/mkonline_primary_live.parquet"
RESIDUAL_LOAD_TREATMENT_COLUMNS = tuple(
    [*EXPECTED_ALIASES]
    + [f"known_{alias}_oracle" for alias in EXPECTED_ALIASES]
)


class ResidualLoadBundleError(RuntimeError):
    """Base error for the Chronos-2 residual-load provider."""


class BundleValidationError(ResidualLoadBundleError):
    """Raised when an immutable bundle does not satisfy its contract."""


def _now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _utc_timestamp(value: Any, *, name: str) -> pd.Timestamp:
    if value is None:
        raise ValueError(f"{name} est absent.")
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"{name} n'est pas un timestamp valide: {value!r}.") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{name} est NaT.")
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} doit contenir un fuseau horaire explicite.")
    return timestamp.tz_convert("UTC")


def _iso_utc(value: Any) -> str:
    return _utc_timestamp(value, name="timestamp").isoformat().replace(
        "+00:00", "Z"
    )


def _delivery_date(value: str | date | pd.Timestamp) -> date:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(DELIVERY_TIMEZONE).tz_localize(None)
    return timestamp.date()


def _delivery_index_utc(
    delivery_day: str | date | pd.Timestamp,
) -> pd.DatetimeIndex:
    day = _delivery_date(delivery_day)
    start_local = pd.Timestamp(day).tz_localize(DELIVERY_TIMEZONE)
    next_local = pd.Timestamp(day + pd.Timedelta(days=1)).tz_localize(
        DELIVERY_TIMEZONE
    )
    result = pd.date_range(
        start=start_local,
        end=next_local,
        inclusive="left",
        freq="h",
    ).tz_convert("UTC")
    if len(result) not in (23, 24, 25):
        raise AssertionError(
            f"Jour civil {day.isoformat()} invalide: {len(result)} heures."
        )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleValidationError(f"Manifeste illisible: {path}") from exc
    if not isinstance(payload, dict):
        raise BundleValidationError(f"{path}: le JSON doit etre un objet.")
    return payload


def _semantic_identity(
    delivery_day: str | date | pd.Timestamp,
    runtime_cutoff: pd.Timestamp,
) -> dict[str, Any]:
    delivery_index = _delivery_index_utc(delivery_day)
    return {
        "schema_version": SCHEMA_VERSION,
        "bundle_type": "chronos2_residual_load_live_pit",
        "provider": "chronos2",
        "delivery_day_local": _delivery_date(delivery_day).isoformat(),
        "delivery_timezone": DELIVERY_TIMEZONE,
        "delivery_start_utc": _iso_utc(delivery_index[0]),
        "delivery_end_utc": _iso_utc(delivery_index[-1]),
        "delivery_hours": len(delivery_index),
        "runtime_cutoff_utc": _iso_utc(runtime_cutoff),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "context_length": CONTEXT_LENGTH,
        "frequency": "h",
        "quantile_levels": list(QUANTILE_LEVELS),
        "cross_learning": False,
        "maximum_internal_gap_hours": MAX_INTERNAL_GAP_HOURS,
    }


def _bundle_directory(
    output_root: Path,
    delivery_day: str | date | pd.Timestamp,
    runtime_cutoff: pd.Timestamp,
) -> Path:
    day = _delivery_date(delivery_day).isoformat()
    cutoff_token = runtime_cutoff.strftime("%Y%m%dT%H%M%SZ")
    return output_root / (
        f"chronos2_residual_load_{day}_{cutoff_token}_"
        f"{MODEL_REVISION[:12]}"
    )


def planned_live_residual_load_manifest_path(
    *,
    delivery_day: str | date | pd.Timestamp,
    runtime_cutoff: Any,
    output_root: str | Path,
) -> Path:
    """Return the deterministic live manifest path without performing I/O."""

    cutoff = _utc_timestamp(runtime_cutoff, name="runtime_cutoff")
    return (
        _bundle_directory(
            Path(output_root).resolve(),
            delivery_day,
            cutoff,
        )
        / "manifest.json"
    ).resolve()


def _create_saturn_client(saturn_url: str, author: str) -> Any:
    if not saturn_url or saturn_url.lower() == "none":
        raise ValueError("saturn_url est absent ou invalide.")
    if not author:
        raise ValueError(
            "saturn_author est absent; renseigne-le ou definis SATURN_AUTHOR."
        )
    try:
        import tshistory_lite
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("tshistory_lite est requis pour lire Saturn.") from exc
    return tshistory_lite.Client(uri=saturn_url, author=author)


def _has_values(raw: Any) -> bool:
    try:
        return raw is not None and len(raw) > 0
    except TypeError:
        return False


def _fetch_observed_asof(
    client: Any,
    series_name: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    runtime_cutoff: pd.Timestamp,
) -> Any:
    if ".fcst" in series_name.lower() or not series_name.lower().endswith(
        ".obs"
    ):
        raise ResidualLoadBundleError(
            f"Serie Saturn interdite pour ce provider: {series_name!r}."
        )

    attempts = (
        {"from_value_date": start, "to_value_date": end},
        {"from_value": start, "to_value": end},
        {"start": start, "end": end},
    )
    type_errors: list[str] = []
    for dates in attempts:
        try:
            raw = client.get(
                series_name,
                **dates,
                revision_date=runtime_cutoff,
            )
        except TypeError as exc:
            type_errors.append(str(exc))
            continue
        if _has_values(raw):
            return raw
        raise ResidualLoadBundleError(
            f"Saturn ne renvoie aucune observation pour {series_name}."
        )
    detail = " | ".join(type_errors[-3:])
    raise ResidualLoadBundleError(
        "Le client Saturn ne permet pas une lecture as-of causale "
        f"pour {series_name}: {detail}"
    )


def _coerce_observed_series(raw: Any, *, series_name: str) -> pd.Series:
    if isinstance(raw, pd.DataFrame):
        if raw.shape[1] != 1:
            raise ResidualLoadBundleError(
                f"{series_name}: Saturn doit renvoyer une seule colonne."
            )
        raw = raw.iloc[:, 0]
    if not isinstance(raw, pd.Series):
        raise ResidualLoadBundleError(
            f"{series_name}: Saturn doit renvoyer une Series pandas."
        )
    try:
        index = pd.DatetimeIndex(pd.to_datetime(raw.index, errors="raise"))
    except Exception as exc:
        raise ResidualLoadBundleError(
            f"{series_name}: index temporel Saturn invalide."
        ) from exc
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    if any(
        getattr(index, field).any()
        for field in ("minute", "second", "microsecond", "nanosecond")
    ):
        raise ResidualLoadBundleError(
            f"{series_name}: les observations doivent etre alignees a l'heure."
        )
    values = pd.to_numeric(raw.to_numpy(), errors="coerce")
    result = pd.Series(values, index=index, name=series_name, dtype=float)
    return result.sort_index(kind="stable").groupby(level=0).last()


def _missing_runs(mask: pd.Series) -> list[pd.DatetimeIndex]:
    if not bool(mask.any()):
        return []
    groups = mask.ne(mask.shift(fill_value=False)).cumsum()
    return [
        pd.DatetimeIndex(group.index)
        for _, group in mask.groupby(groups)
        if bool(group.iloc[0])
    ]


def _prepare_context(
    raw: Any,
    *,
    series_name: str,
    runtime_cutoff: pd.Timestamp,
) -> tuple[pd.Series, dict[str, Any]]:
    observed = _coerce_observed_series(raw, series_name=series_name)
    after_cutoff = observed.index > runtime_cutoff
    discarded_after_cutoff = int(after_cutoff.sum())
    observed = observed.loc[~after_cutoff]
    observed = observed.dropna()
    if observed.empty:
        raise ResidualLoadBundleError(
            f"{series_name}: aucune observation disponible au cutoff."
        )

    full_index = pd.date_range(
        observed.index[0], observed.index[-1], freq="h", tz="UTC"
    )
    regular = observed.reindex(full_index)
    gaps = _missing_runs(regular.isna())
    too_long = [gap for gap in gaps if len(gap) > MAX_INTERNAL_GAP_HOURS]
    if too_long:
        first = too_long[0]
        raise ResidualLoadBundleError(
            f"{series_name}: gap interne de {len(first)} heures "
            f"({first[0].isoformat()} -> {first[-1].isoformat()}), "
            f"maximum autorise={MAX_INTERNAL_GAP_HOURS}."
        )
    imputed_index = pd.DatetimeIndex(
        [timestamp for gap in gaps for timestamp in gap], tz="UTC"
    )
    if len(imputed_index):
        regular = regular.interpolate(method="time", limit_area="inside")
    if regular.isna().any():
        raise ResidualLoadBundleError(
            f"{series_name}: contexte incomplet apres interpolation."
        )
    if len(regular) < CONTEXT_LENGTH:
        raise ResidualLoadBundleError(
            f"{series_name}: {len(regular)} observations horaires, "
            f"{CONTEXT_LENGTH} requises."
        )

    context = regular.iloc[-CONTEXT_LENGTH:].astype(float)
    used_imputed = imputed_index.intersection(context.index)
    audit = {
        "context_start_utc": _iso_utc(context.index[0]),
        "context_end_utc": _iso_utc(context.index[-1]),
        "context_rows": len(context),
        "last_observation_utc": _iso_utc(observed.index[-1]),
        "discarded_observations_after_cutoff": discarded_after_cutoff,
        "imputed_hours": len(used_imputed),
        "imputed_timestamps_utc": [_iso_utc(value) for value in used_imputed],
        "interpolation": "linear_time_internal_only",
    }
    return context, audit


def _load_pipeline(
    *,
    model_id: str,
    revision: str,
    device: str,
    local_files_only: bool,
) -> Any:
    try:
        from chronos import Chronos2Pipeline
        from chronos2_modular.common import (
            configure_huggingface_ssl,
            resolve_device,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            'Chronos-2 est requis: pip install "chronos-forecasting>=2.2.2,<3"'
        ) from exc

    resolved_device, dtype = resolve_device(device)
    configure_huggingface_ssl()
    kwargs: dict[str, Any] = {
        "revision": revision,
        "device_map": resolved_device,
        "local_files_only": bool(local_files_only),
    }
    try:
        return Chronos2Pipeline.from_pretrained(
            model_id, dtype=dtype, **kwargs
        )
    except TypeError:
        return Chronos2Pipeline.from_pretrained(
            model_id, torch_dtype=dtype, **kwargs
        )


def _quantile_column(frame: pd.DataFrame, level: float) -> Any:
    percent = int(round(level * 100))
    candidates: tuple[Any, ...] = (
        level,
        str(level),
        f"{level:.1f}",
        f"q{percent:02d}",
        f"p{percent:02d}",
    )
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ResidualLoadBundleError(
        f"Sortie Chronos sans quantile {level}: {list(frame.columns)}"
    )


def _predict_group(
    pipeline: Any,
    contexts: Mapping[str, pd.Series],
    *,
    prediction_length: int,
    batch_size: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for alias, context in contexts.items():
        rows.append(
            pd.DataFrame(
                {
                    "item_id": alias,
                    "timestamp": context.index.tz_convert("UTC").tz_localize(
                        None
                    ),
                    "target": context.to_numpy(dtype=float),
                }
            )
        )
    context_df = pd.concat(rows, ignore_index=True)
    raw = pipeline.predict_df(
        context_df,
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
        prediction_length=prediction_length,
        quantile_levels=list(QUANTILE_LEVELS),
        batch_size=max(1, int(batch_size)),
        context_length=CONTEXT_LENGTH,
        cross_learning=False,
        validate_inputs=False,
        freq="h",
    )
    if not isinstance(raw, pd.DataFrame):
        raise ResidualLoadBundleError(
            "Chronos-2 predict_df doit renvoyer un DataFrame."
        )
    prediction = raw.copy()
    if "target_name" in prediction.columns:
        prediction = prediction.loc[
            prediction["target_name"].astype(str).eq("target")
        ].copy()
    required = {"item_id", "timestamp"}
    if not required.issubset(prediction.columns):
        raise ResidualLoadBundleError(
            f"Sortie Chronos incomplete: {sorted(required - set(prediction.columns))}"
        )
    timestamps = pd.DatetimeIndex(
        pd.to_datetime(prediction["timestamp"], errors="raise")
    )
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize("UTC")
    else:
        timestamps = timestamps.tz_convert("UTC")
    prediction["timestamp"] = timestamps
    prediction = prediction.rename(
        columns={
            _quantile_column(prediction, level): f"q{int(level * 100):02d}"
            for level in QUANTILE_LEVELS
        }
    )
    prediction["item_id"] = prediction["item_id"].astype(str)
    if prediction.duplicated(["item_id", "timestamp"]).any():
        raise ResidualLoadBundleError(
            "Sortie Chronos dupliquee pour un item/timestamp."
        )
    return prediction[["item_id", "timestamp", "q10", "q50", "q90"]]


def _forecast_delivery_day(
    pipeline: Any,
    contexts: Mapping[str, pd.Series],
    delivery_index: pd.DatetimeIndex,
    *,
    batch_size: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    delivery_end = delivery_index[-1]
    grouped: dict[pd.Timestamp, dict[str, pd.Series]] = {}
    for alias, context in contexts.items():
        last = context.index[-1]
        if last >= delivery_index[0]:
            raise ResidualLoadBundleError(
                f"{alias}: derniere observation {last} non anterieure "
                "au jour de livraison."
            )
        grouped.setdefault(last, {})[alias] = context

    forecasts: dict[str, pd.DataFrame] = {}
    audits: dict[str, dict[str, Any]] = {}
    for last_observation, group in sorted(grouped.items(), key=lambda item: item[0]):
        horizon_delta = delivery_end - last_observation
        prediction_length_float = horizon_delta / pd.Timedelta(hours=1)
        if prediction_length_float != int(prediction_length_float):
            raise ResidualLoadBundleError(
                "Le pont de forecast n'est pas aligne sur une heure entiere."
            )
        prediction_length = int(prediction_length_float)
        if prediction_length <= 0:
            raise ResidualLoadBundleError("Horizon Chronos non positif.")
        prediction = _predict_group(
            pipeline,
            group,
            prediction_length=prediction_length,
            batch_size=batch_size,
        )
        for alias in group:
            selected = prediction.loc[
                prediction["item_id"].eq(alias)
            ].set_index("timestamp")
            selected = selected.reindex(delivery_index)
            if selected[["q10", "q50", "q90"]].isna().any().any():
                raise ResidualLoadBundleError(
                    f"{alias}: forecast Chronos incomplet pour le jour livre."
                )
            numeric = selected[["q10", "q50", "q90"]].astype(float)
            if not np.isfinite(numeric.to_numpy()).all():
                raise ResidualLoadBundleError(
                    f"{alias}: forecast Chronos contient des valeurs non finies."
                )
            forecasts[alias] = numeric
            audits[alias] = {
                "forecast_origin_utc": _iso_utc(last_observation),
                "forecast_first_utc": _iso_utc(last_observation + pd.Timedelta(hours=1)),
                "forecast_last_utc": _iso_utc(delivery_end),
                "bridge_horizon_hours": prediction_length,
                "lead_to_delivery_start_hours": int(
                    (delivery_index[0] - last_observation)
                    / pd.Timedelta(hours=1)
                ),
            }
    if set(forecasts) != set(EXPECTED_ALIASES):
        raise ResidualLoadBundleError(
            "Chronos n'a pas produit exactement les cinq aliases requis."
        )
    return forecasts, audits


def _validate_identity(
    manifest: Mapping[str, Any],
    *,
    expected_delivery_day: str | date | pd.Timestamp | None,
    expected_runtime_cutoff: Any | None,
) -> None:
    if expected_delivery_day is None:
        delivery_day = manifest.get("delivery_day_local")
    else:
        delivery_day = expected_delivery_day
    if expected_runtime_cutoff is None:
        cutoff_value = manifest.get("runtime_cutoff_utc")
        if cutoff_value is None:
            raise BundleValidationError("runtime_cutoff_utc absent du manifeste.")
        cutoff = _utc_timestamp(cutoff_value, name="runtime_cutoff_utc")
    else:
        cutoff = _utc_timestamp(
            expected_runtime_cutoff, name="expected_runtime_cutoff"
        )
    expected = _semantic_identity(delivery_day, cutoff)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise BundleValidationError(
                f"Manifeste incompatible: {key}={manifest.get(key)!r}, "
                f"attendu {value!r}."
            )


def validate_live_residual_load_bundle(
    manifest_path: str | Path,
    *,
    expected_delivery_day: str | date | pd.Timestamp | None = None,
    expected_runtime_cutoff: Any | None = None,
) -> dict[str, Any]:
    """Validate provenance, checksums and PIT coverage of an existing bundle."""

    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise BundleValidationError(f"Manifeste absent: {path}")
    manifest = _json_object(path)
    _validate_identity(
        manifest,
        expected_delivery_day=expected_delivery_day,
        expected_runtime_cutoff=expected_runtime_cutoff,
    )
    if manifest.get("target_source_kind") != "observed_entsoe":
        raise BundleValidationError("La source target doit etre observed_entsoe.")
    if manifest.get("target_source_provider") != "saturn":
        raise BundleValidationError("target_source_provider doit etre saturn.")
    if manifest.get("forecast_values_provider") != "chronos2":
        raise BundleValidationError("forecast_values_provider doit etre chronos2.")
    endpoint_sha = str(manifest.get("saturn_endpoint_sha256", "")).lower()
    if len(endpoint_sha) != 64 or any(
        character not in "0123456789abcdef" for character in endpoint_sha
    ):
        raise BundleValidationError("saturn_endpoint_sha256 invalide.")
    if manifest.get("saturn_forecast_series_used") is not False:
        raise BundleValidationError(
            "saturn_forecast_series_used doit etre explicitement false."
        )
    artifacts = manifest.get("artifacts")
    inputs = manifest.get("inputs")
    if not isinstance(artifacts, list) or not isinstance(inputs, list):
        raise BundleValidationError("artifacts/inputs invalides dans le manifeste.")
    if len(artifacts) != len(EXPECTED_ALIASES) or any(
        not isinstance(item, dict) for item in artifacts
    ):
        raise BundleValidationError("Le manifeste doit declarer cinq artifacts.")
    artifact_aliases = [str(item.get("alias", "")) for item in artifacts]
    if len(set(artifact_aliases)) != len(artifact_aliases) or set(
        artifact_aliases
    ) != set(EXPECTED_ALIASES):
        raise BundleValidationError("Le manifeste ne declare pas les cinq artifacts.")
    artifact_paths = [str(item.get("path", "")) for item in artifacts]
    if len(set(artifact_paths)) != len(artifact_paths):
        raise BundleValidationError("Chaque alias doit avoir son propre artifact.")
    if len(inputs) != len(EXPECTED_ALIASES):
        raise BundleValidationError("Le manifeste doit declarer cinq inputs.")
    source_by_alias: dict[str, str] = {}
    for item in inputs:
        if not isinstance(item, dict):
            raise BundleValidationError("Declaration input invalide.")
        alias = str(item.get("alias", ""))
        source = str(item.get("source_series", ""))
        if ".fcst" in source.lower() or not source.lower().endswith(".obs"):
            raise BundleValidationError(
                f"{alias}: source Saturn non observee/interdite: {source!r}."
            )
        if alias in source_by_alias:
            raise BundleValidationError(f"Input duplique pour {alias}.")
        source_by_alias[alias] = source
    expected_sources = {
        COUNTRY_ALIASES[country]: source
        for country, source in COUNTRY_OBSERVED_SERIES.items()
    }
    if source_by_alias != expected_sources:
        raise BundleValidationError("Les sources observees ne correspondent pas au contrat.")

    cutoff = _utc_timestamp(
        manifest["runtime_cutoff_utc"], name="runtime_cutoff_utc"
    )
    delivery_index = _delivery_index_utc(manifest["delivery_day_local"])
    for item in inputs:
        alias = str(item["alias"])
        if item.get("source_kind") != "observed_entsoe_asof":
            raise BundleValidationError(f"{alias}: source_kind invalide.")
        if item.get("context_rows") != CONTEXT_LENGTH:
            raise BundleValidationError(f"{alias}: context_rows divergent.")
        for field in (
            "context_start_utc",
            "context_end_utc",
            "last_observation_utc",
            "forecast_origin_utc",
        ):
            timestamp = _utc_timestamp(item.get(field), name=f"{alias}.{field}")
            if timestamp > cutoff:
                raise BundleValidationError(
                    f"{alias}: {field} est posterieur au cutoff."
                )
        if item.get("forecast_last_utc") != _iso_utc(delivery_index[-1]):
            raise BundleValidationError(f"{alias}: forecast_last_utc divergent.")
        imputed = item.get("imputed_timestamps_utc")
        if not isinstance(imputed, list) or item.get("imputed_hours") != len(imputed):
            raise BundleValidationError(f"{alias}: audit d'interpolation invalide.")
        if int(item.get("imputed_hours", -1)) < 0:
            raise BundleValidationError(f"{alias}: imputed_hours invalide.")
        if int(item.get("bridge_horizon_hours", 0)) <= 0:
            raise BundleValidationError(f"{alias}: bridge_horizon_hours invalide.")
    bundle_dir = path.parent
    for declaration in artifacts:
        alias = str(declaration.get("alias", ""))
        relative = Path(str(declaration.get("path", "")))
        artifact = (bundle_dir / relative).resolve()
        try:
            artifact.relative_to(bundle_dir)
        except ValueError as exc:
            raise BundleValidationError(
                f"{alias}: chemin artifact hors bundle."
            ) from exc
        if not artifact.is_file():
            raise BundleValidationError(f"{alias}: artifact absent: {artifact}")
        expected_sha = str(declaration.get("sha256", "")).lower()
        if len(expected_sha) != 64 or _sha256(artifact) != expected_sha:
            raise BundleValidationError(f"{alias}: SHA-256 divergent.")
        try:
            frame = pd.read_parquet(artifact)
        except Exception as exc:
            raise BundleValidationError(
                f"{alias}: parquet illisible."
            ) from exc
        expected_columns = {
            "value_time_utc",
            "snapshot_time_utc",
            "revision_time_utc",
            "value",
            "q10",
            "q50",
            "q90",
        }
        if set(frame.columns) != expected_columns:
            raise BundleValidationError(
                f"{alias}: colonnes PIT/quantiles invalides: {list(frame.columns)}"
            )
        value_time = pd.DatetimeIndex(
            pd.to_datetime(frame["value_time_utc"], utc=True, errors="raise")
        )
        if not value_time.equals(delivery_index):
            raise BundleValidationError(
                f"{alias}: couverture du jour livre incomplete ou desordonnee."
            )
        snapshot = pd.DatetimeIndex(
            pd.to_datetime(frame["snapshot_time_utc"], utc=True, errors="raise")
        )
        revision = pd.DatetimeIndex(
            pd.to_datetime(frame["revision_time_utc"], utc=True, errors="raise")
        )
        if not snapshot.equals(pd.DatetimeIndex([cutoff] * len(frame))):
            raise BundleValidationError(f"{alias}: snapshot_time_utc != cutoff.")
        if not revision.equals(pd.DatetimeIndex([cutoff] * len(frame))):
            raise BundleValidationError(f"{alias}: revision_time_utc != cutoff.")
        numeric = frame[["value", "q10", "q50", "q90"]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise BundleValidationError(f"{alias}: valeurs PIT non finies.")
        if not np.array_equal(frame["value"].to_numpy(), frame["q50"].to_numpy()):
            raise BundleValidationError(f"{alias}: value doit etre exactement q50.")
        if declaration.get("rows") != len(frame):
            raise BundleValidationError(f"{alias}: nombre de lignes divergent.")
    return manifest


def bundle_alias_paths(manifest_path: str | Path) -> dict[str, Path]:
    """Return validated alias -> PIT parquet paths for runtime overlays."""

    path = Path(manifest_path).resolve()
    manifest = validate_live_residual_load_bundle(path)
    return {
        str(item["alias"]): (path.parent / str(item["path"])).resolve()
        for item in manifest["artifacts"]
    }


def archive_live_residual_load_bundle(
    manifest_path: str | Path,
    *,
    archive_inputs_dir: str | Path,
    expected_delivery_day: str | date | pd.Timestamp,
    expected_runtime_cutoff: Any,
) -> dict[str, Any]:
    """Copy the complete validated upstream bundle into a run archive.

    Only the manifest and its five checksum-declared parquet artifacts cross
    this boundary. The returned references are relative to ``inputs`` so an
    atomic staging-directory rename cannot leak a temporary path into run
    provenance. Original absolute paths remain explicit provenance only.
    """

    origin_manifest = Path(manifest_path).expanduser().resolve()
    cutoff = _utc_timestamp(
        expected_runtime_cutoff,
        name="expected_runtime_cutoff",
    )
    manifest = validate_live_residual_load_bundle(
        origin_manifest,
        expected_delivery_day=expected_delivery_day,
        expected_runtime_cutoff=cutoff,
    )
    origin_paths = bundle_alias_paths(origin_manifest)
    inputs_dir = Path(archive_inputs_dir).expanduser().resolve()
    inputs_dir.mkdir(parents=True, exist_ok=True)
    destination = (inputs_dir / ARCHIVED_BUNDLE_DIRECTORY).resolve()
    try:
        destination.relative_to(inputs_dir)
    except ValueError as exc:
        raise BundleValidationError(
            "La racine du bundle archive sort du dossier inputs."
        ) from exc

    def validate_destination() -> tuple[Path, dict[str, Path]]:
        archived_manifest = destination / "manifest.json"
        archived = validate_live_residual_load_bundle(
            archived_manifest,
            expected_delivery_day=expected_delivery_day,
            expected_runtime_cutoff=cutoff,
        )
        if archived != manifest or _sha256(archived_manifest) != _sha256(
            origin_manifest
        ):
            raise BundleValidationError(
                "Le manifeste du bundle archive diverge de l'amont d'origine."
            )
        archived_paths = bundle_alias_paths(archived_manifest)
        for alias in EXPECTED_ALIASES:
            if _sha256(archived_paths[alias]) != _sha256(origin_paths[alias]):
                raise BundleValidationError(
                    f"{alias}: parquet archive divergent de l'amont d'origine."
                )
        return archived_manifest.resolve(), archived_paths

    if destination.exists():
        archived_manifest, archived_paths = validate_destination()
    else:
        temporary = Path(
            tempfile.mkdtemp(
                dir=inputs_dir,
                prefix=f".{ARCHIVED_BUNDLE_DIRECTORY}.",
            )
        )
        try:
            shutil.copy2(origin_manifest, temporary / "manifest.json")
            for declaration in manifest["artifacts"]:
                alias = str(declaration["alias"])
                relative = Path(str(declaration["path"]))
                archived_artifact = (temporary / relative).resolve()
                try:
                    archived_artifact.relative_to(temporary.resolve())
                except ValueError as exc:
                    raise BundleValidationError(
                        f"{alias}: chemin archive hors bundle."
                    ) from exc
                archived_artifact.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(origin_paths[alias], archived_artifact)
            validate_live_residual_load_bundle(
                temporary / "manifest.json",
                expected_delivery_day=expected_delivery_day,
                expected_runtime_cutoff=cutoff,
            )
            try:
                temporary.replace(destination)
            except FileExistsError:
                pass
            archived_manifest, archived_paths = validate_destination()
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    return {
        "origin_manifest_path": str(origin_manifest),
        "origin_manifest_sha256": _sha256(origin_manifest),
        "archived_manifest_path": archived_manifest.relative_to(
            inputs_dir
        ).as_posix(),
        "archived_manifest_sha256": _sha256(archived_manifest),
        "files": {
            alias: {
                "origin_path": str(origin_paths[alias]),
                "archived_path": archived_paths[alias]
                .relative_to(inputs_dir)
                .as_posix(),
                "sha256": _sha256(archived_paths[alias]),
            }
            for alias in EXPECTED_ALIASES
        },
    }


def _sealed_checksum_entry(
    checksum_manifest: Mapping[str, Any],
    *,
    relative_path: str,
    role: str,
) -> Mapping[str, Any]:
    artifacts = checksum_manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise BundleValidationError(
            "Controle Saturn: artifact_checksums.artifacts est invalide."
        )
    matches = [
        item
        for item in artifacts
        if isinstance(item, Mapping)
        and str(item.get("path", "")).replace("\\", "/") == relative_path
        and item.get("role") == role
    ]
    if len(matches) != 1:
        raise BundleValidationError(
            "Controle Saturn: entree checksum absente ou ambigue pour "
            f"{role}/{relative_path}."
        )
    return matches[0]


def _verify_sealed_artifact(
    archive: Path,
    checksum_manifest: Mapping[str, Any],
    *,
    relative_path: str,
    role: str,
) -> tuple[Path, str]:
    declaration = _sealed_checksum_entry(
        checksum_manifest,
        relative_path=relative_path,
        role=role,
    )
    path = (archive / Path(relative_path)).resolve()
    try:
        path.relative_to(archive)
    except ValueError as exc:  # pragma: no cover - constant relative paths
        raise BundleValidationError(
            f"Controle Saturn: artifact hors archive: {relative_path}."
        ) from exc
    if not path.is_file():
        raise BundleValidationError(
            f"Controle Saturn: artifact absent: {path}."
        )
    expected_sha = str(declaration.get("sha256") or "").lower()
    observed_sha = _sha256(path)
    if len(expected_sha) != 64 or observed_sha != expected_sha:
        raise BundleValidationError(
            f"Controle Saturn: SHA-256 divergent pour {relative_path}."
        )
    try:
        declared_size = int(declaration.get("size_bytes"))
    except (TypeError, ValueError) as exc:
        raise BundleValidationError(
            f"Controle Saturn: taille invalide pour {relative_path}."
        ) from exc
    if declared_size != path.stat().st_size:
        raise BundleValidationError(
            f"Controle Saturn: taille divergente pour {relative_path}."
        )
    return path, observed_sha


def _read_sealed_csv(
    path: Path,
    *,
    required_columns: tuple[str, ...],
    label: str,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise BundleValidationError(
            f"Controle Saturn: {label} illisible: {path}."
        ) from exc
    if "timestamp" not in frame.columns:
        raise BundleValidationError(
            f"Controle Saturn: timestamp absent de {label}."
        )
    missing = sorted(set(required_columns).difference(frame.columns))
    if missing:
        raise BundleValidationError(
            f"Controle Saturn: colonnes absentes de {label}: {missing}."
        )
    try:
        index = pd.DatetimeIndex(
            pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        )
    except (TypeError, ValueError) as exc:
        raise BundleValidationError(
            f"Controle Saturn: timeline invalide dans {label}."
        ) from exc
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise BundleValidationError(
            f"Controle Saturn: timeline dupliquee ou non triee dans {label}."
        )
    result = frame.copy()
    result["timestamp"] = index
    return result, index


def validate_sealed_saturn_control_archive(
    archive_dir: str | Path,
    *,
    expected_delivery_day: str | date | pd.Timestamp,
    expected_zone: str,
    require_primary: bool = False,
) -> dict[str, Any]:
    """Validate the exact issued-Saturn archive used as challenger control.

    This is deliberately fail-closed: the canonical archive name, run
    identity, checksum manifest, materialized historical inputs and complete
    J+1 grid must all agree before any value is reused by a challenger.
    """

    archive = Path(archive_dir).expanduser().resolve()
    zone = str(expected_zone).strip().upper()
    day = _delivery_date(expected_delivery_day)
    expected_name = f"{zone.lower()}_day_ahead_{day.isoformat()}"
    if archive.name != expected_name:
        raise BundleValidationError(
            "Controle Saturn: nom d'archive non canonique: "
            f"{archive.name!r}, attendu {expected_name!r}."
        )
    if not archive.is_dir():
        raise BundleValidationError(
            f"Controle Saturn publie absent: {archive}."
        )

    run_manifest_path = archive / "run_manifest.json"
    checksum_path = archive / "artifact_checksums.json"
    run_manifest = _json_object(run_manifest_path)
    checksum_manifest = _json_object(checksum_path)
    if checksum_manifest.get("algorithm") != "sha256":
        raise BundleValidationError(
            "Controle Saturn: algorithm doit etre sha256."
        )
    declared_output = checksum_manifest.get("output_directory")
    if declared_output not in (None, "") and Path(
        str(declared_output)
    ).expanduser().resolve() != archive:
        raise BundleValidationError(
            "Controle Saturn: output_directory divergent."
        )
    _verify_sealed_artifact(
        archive,
        checksum_manifest,
        relative_path="run_manifest.json",
        role="run_artifact",
    )
    expected_identity = {
        "zone": zone,
        "delivery_day_local": day.isoformat(),
        "run_type": "live_day_ahead",
        "forecast_status": "issued_live",
    }
    for key, expected in expected_identity.items():
        if run_manifest.get(key) != expected:
            raise BundleValidationError(
                f"Controle Saturn: run_manifest.{key}="
                f"{run_manifest.get(key)!r}, attendu {expected!r}."
            )
    source = str(run_manifest.get("residual_load_source") or "saturn").lower()
    if source != "saturn":
        raise BundleValidationError(
            "Controle Saturn: residual_load_source doit etre saturn."
        )

    aligned_path, aligned_sha = _verify_sealed_artifact(
        archive,
        checksum_manifest,
        relative_path=SEALED_SATURN_ALIGNED_INPUTS,
        role="run_artifact",
    )
    model_path, model_sha = _verify_sealed_artifact(
        archive,
        checksum_manifest,
        relative_path=SEALED_SATURN_MODEL_COVARIATES,
        role="run_artifact",
    )
    aligned, aligned_index = _read_sealed_csv(
        aligned_path,
        required_columns=("target", *EXPECTED_ALIASES),
        label="aligned_inputs",
    )
    model, model_index = _read_sealed_csv(
        model_path,
        required_columns=RESIDUAL_LOAD_TREATMENT_COLUMNS,
        label="model_covariates_with_future",
    )
    delivery_index = _delivery_index_utc(day)
    delivery_mask = model_index.isin(delivery_index)
    if int(delivery_mask.sum()) != len(delivery_index):
        raise BundleValidationError(
            "Controle Saturn: model_covariates ne couvre pas exactement J+1."
        )
    if not model_index[delivery_mask].equals(delivery_index):
        raise BundleValidationError(
            "Controle Saturn: grille J+1 des model_covariates divergente."
        )
    if not bool((model_index < delivery_index[0]).any()):
        raise BundleValidationError(
            "Controle Saturn: historique model_covariates vide."
        )
    expected_history_end = delivery_index[0] - pd.Timedelta(hours=1)
    if aligned_index.empty or aligned_index[-1] != expected_history_end:
        raise BundleValidationError(
            "Controle Saturn: aligned_inputs ne finit pas exactement avant J+1."
        )
    if not aligned_index.equals(model_index[model_index < delivery_index[0]]):
        raise BundleValidationError(
            "Controle Saturn: timelines historiques aligned/model divergentes."
        )

    files: dict[str, dict[str, Any]] = {
        "aligned_inputs": {
            "path": aligned_path,
            "sha256": aligned_sha,
            "relative_path": SEALED_SATURN_ALIGNED_INPUTS,
        },
        "model_covariates_with_future": {
            "path": model_path,
            "sha256": model_sha,
            "relative_path": SEALED_SATURN_MODEL_COVARIATES,
        },
    }
    if require_primary:
        primary_path, primary_sha = _verify_sealed_artifact(
            archive,
            checksum_manifest,
            relative_path=SEALED_SATURN_PRIMARY,
            role="run_artifact",
        )
        files["mkonline_primary_live"] = {
            "path": primary_path,
            "sha256": primary_sha,
            "relative_path": SEALED_SATURN_PRIMARY,
        }

    return {
        "archive_path": archive,
        "zone": zone,
        "delivery_day_local": day.isoformat(),
        "run_manifest_path": run_manifest_path.resolve(),
        "run_manifest_sha256": _sha256(run_manifest_path),
        "checksum_manifest_path": checksum_path.resolve(),
        "checksum_manifest_sha256": _sha256(checksum_path),
        "aligned": aligned,
        "aligned_index": aligned_index,
        "model": model,
        "model_index": model_index,
        "files": files,
    }


def _materialize_sealed_saturn_pit_controls(
    control: Mapping[str, Any],
    *,
    destination: Path,
) -> dict[str, Path]:
    """Materialize immutable PIT-shaped rows from the sealed model context."""

    model = control["model"]
    if not isinstance(model, pd.DataFrame):  # pragma: no cover - internal API
        raise TypeError("Controle Saturn model invalide.")
    index = pd.DatetimeIndex(control["model_index"])
    timezone = DELIVERY_TIMEZONE
    local = index.tz_convert(timezone)
    availability = (
        local.normalize()
        - pd.Timedelta(days=1)
        + pd.Timedelta(hours=8)
    ).tz_convert("UTC")
    destination = destination.resolve()
    manifest_path = destination / "manifest.json"
    if destination.exists():
        manifest = _json_object(manifest_path)
        paths: dict[str, Path] = {}
        for alias in EXPECTED_ALIASES:
            declaration = manifest.get("files", {}).get(alias, {})
            path = destination / f"{alias}.parquet"
            if (
                not path.is_file()
                or declaration.get("sha256") != _sha256(path)
            ):
                raise BundleValidationError(
                    f"Controle Saturn PIT archive divergent pour {alias}."
                )
            paths[alias] = path.resolve()
        if manifest.get("source_model_covariates_sha256") != control["files"][
            "model_covariates_with_future"
        ]["sha256"]:
            raise BundleValidationError(
                "Controle Saturn PIT issu d'un autre model_covariates."
            )
        return paths

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.")
    )
    try:
        declarations: dict[str, dict[str, Any]] = {}
        for alias in EXPECTED_ALIASES:
            values = pd.to_numeric(model[alias], errors="coerce")
            finite = np.isfinite(values.to_numpy(dtype=float))
            frame = pd.DataFrame(
                {
                    "value_time_utc": index[finite],
                    "snapshot_time_utc": availability[finite],
                    "revision_time_utc": availability[finite],
                    "value": values.loc[finite].to_numpy(dtype=float),
                }
            )
            if frame.empty:
                raise BundleValidationError(
                    f"Controle Saturn: aucune valeur finie pour {alias}."
                )
            path = temporary / f"{alias}.parquet"
            frame.to_parquet(path, index=False)
            declarations[alias] = {
                "path": path.name,
                "sha256": _sha256(path),
                "rows": len(frame),
            }
        manifest = {
            "schema_version": 1,
            "control_type": "sealed_saturn_materialized_model_context",
            "source_archive_path": str(control["archive_path"]),
            "source_checksum_manifest_sha256": control[
                "checksum_manifest_sha256"
            ],
            "source_model_covariates_sha256": control["files"][
                "model_covariates_with_future"
            ]["sha256"],
            "files": declarations,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return {
            alias: (destination / f"{alias}.parquet").resolve()
            for alias in EXPECTED_ALIASES
        }
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def seal_challenger_zone_data_from_saturn_control(
    data: Any,
    *,
    saturn_archive_dir: str | Path,
    archive_inputs_dir: str | Path,
    expected_delivery_index: pd.DatetimeIndex,
    expected_zone: str,
) -> dict[str, Any]:
    """Freeze every non-treatment input to the issued Saturn control.

    ``aligned_inputs`` is reused byte-for-byte. ``model_covariates`` starts
    from the sealed Saturn frame and receives only the ten J+1 treatment
    columns (five residual loads plus their ``known_*_oracle`` mirrors) from
    the freshly prepared Chronos-2 challenger.
    """

    delivery_index = pd.DatetimeIndex(expected_delivery_index).tz_convert("UTC")
    control = validate_sealed_saturn_control_archive(
        saturn_archive_dir,
        expected_delivery_day=delivery_index[0].tz_convert(
            DELIVERY_TIMEZONE
        ).date(),
        expected_zone=expected_zone,
    )
    fresh_model = getattr(data, "model_context_covariates", None)
    if not isinstance(fresh_model, pd.DataFrame):
        raise BundleValidationError(
            "Challenger: model_context_covariates frais absent."
        )
    fresh_model = fresh_model.copy()
    if not isinstance(fresh_model.index, pd.DatetimeIndex) or fresh_model.index.tz is None:
        raise BundleValidationError(
            "Challenger: timeline model_context_covariates invalide."
        )
    fresh_model.index = fresh_model.index.tz_convert("UTC")
    control_model = control["model"].drop(columns=["timestamp"]).copy()
    control_model.index = pd.DatetimeIndex(control["model_index"])
    if list(fresh_model.columns) != list(control_model.columns):
        raise BundleValidationError(
            "Challenger: schema model_covariates different du controle Saturn."
        )
    if not fresh_model.index.equals(control_model.index):
        raise BundleValidationError(
            "Challenger: timeline model_covariates differente du controle Saturn."
        )
    delivery_mask = control_model.index.isin(delivery_index)
    if int(delivery_mask.sum()) != len(delivery_index):
        raise BundleValidationError(
            "Challenger: masque J+1 incomplet dans le controle Saturn."
        )
    treatment = list(RESIDUAL_LOAD_TREATMENT_COLUMNS)
    future_treatment = fresh_model.loc[delivery_mask, treatment].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(future_treatment.to_numpy(dtype=float)).all():
        raise BundleValidationError(
            "Challenger: traitement residual-load J+1 non fini."
        )
    hybrid = control_model.copy()
    hybrid.loc[delivery_mask, treatment] = future_treatment.to_numpy(dtype=float)

    aligned = control["aligned"].drop(columns=["timestamp"]).copy()
    aligned.index = pd.DatetimeIndex(control["aligned_index"])
    if "target" not in aligned:
        raise BundleValidationError("Controle Saturn: target alignee absente.")
    timezone = str(getattr(data, "timezone", DELIVERY_TIMEZONE))
    local_aligned_index = aligned.index.tz_convert(timezone)
    local_model_index = hybrid.index.tz_convert(timezone)
    data.target = pd.Series(
        pd.to_numeric(aligned.pop("target"), errors="raise").to_numpy(dtype=float),
        index=local_aligned_index,
        name="target",
    )
    data.covariates = aligned.set_axis(local_aligned_index, axis=0)
    data.model_context_covariates = hybrid.set_axis(local_model_index, axis=0)

    inputs_dir = Path(archive_inputs_dir).expanduser().resolve()
    inputs_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir = (inputs_dir / SEALED_SATURN_CONTROL_DIRECTORY).resolve()
    try:
        sealed_dir.relative_to(inputs_dir)
    except ValueError as exc:  # pragma: no cover - constant child path
        raise BundleValidationError(
            "Controle Saturn archive hors dossier inputs."
        ) from exc
    sealed_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "aligned_inputs.csv.gz": control["files"]["aligned_inputs"]["path"],
        "model_covariates_with_future.csv.gz": control["files"][
            "model_covariates_with_future"
        ]["path"],
        "run_manifest.json": control["run_manifest_path"],
        "artifact_checksums.json": control["checksum_manifest_path"],
    }
    copied: dict[str, str] = {}
    for name, source in sources.items():
        destination = sealed_dir / name
        if destination.exists() and _sha256(destination) != _sha256(Path(source)):
            raise BundleValidationError(
                f"Controle Saturn archive deja divergent: {destination}."
            )
        if not destination.exists():
            shutil.copy2(source, destination)
        copied[name] = _sha256(destination)

    shutil.copy2(
        control["files"]["aligned_inputs"]["path"],
        inputs_dir / "aligned_inputs.csv.gz",
    )
    data.model_context_covariates.reset_index(
        names="timestamp"
    ).to_csv(
        inputs_dir / "model_covariates_with_future.csv.gz",
        index=False,
        compression="gzip",
    )
    written_aligned, written_aligned_index = _read_sealed_csv(
        inputs_dir / "aligned_inputs.csv.gz",
        required_columns=("target", *EXPECTED_ALIASES),
        label="aligned_inputs challenger",
    )
    written_model, written_model_index = _read_sealed_csv(
        inputs_dir / "model_covariates_with_future.csv.gz",
        required_columns=RESIDUAL_LOAD_TREATMENT_COLUMNS,
        label="model_covariates challenger",
    )
    if not written_aligned.equals(control["aligned"]):
        raise BundleValidationError(
            "Challenger: aligned_inputs ecrit diverge du controle Saturn."
        )
    if not written_aligned_index.equals(control["aligned_index"]):
        raise BundleValidationError(
            "Challenger: timeline aligned_inputs ecrite divergente."
        )
    if list(written_model.columns) != ["timestamp", *hybrid.columns] or not (
        written_model_index.equals(hybrid.index)
    ):
        raise BundleValidationError(
            "Challenger: schema/timeline model_covariates ecrit divergent."
        )
    historical_mask = ~delivery_mask
    for column in hybrid.columns:
        observed = written_model[column]
        if column in RESIDUAL_LOAD_TREATMENT_COLUMNS:
            if not observed.loc[historical_mask].reset_index(drop=True).equals(
                control["model"].loc[
                    historical_mask, column
                ].reset_index(drop=True)
            ):
                raise BundleValidationError(
                    "Challenger: historique de traitement divergent apres ecriture "
                    f"pour {column}."
                )
            if not np.allclose(
                pd.to_numeric(observed.loc[delivery_mask], errors="coerce"),
                hybrid.loc[delivery_mask, column],
                rtol=0.0,
                atol=1e-12,
                equal_nan=False,
            ):
                raise BundleValidationError(
                    f"Challenger: traitement J+1 divergent apres ecriture pour {column}."
                )
        elif not observed.reset_index(drop=True).equals(
            control["model"][column].reset_index(drop=True)
        ):
            raise BundleValidationError(
                "Challenger: input hors traitement divergent apres ecriture "
                f"pour {column}."
            )

    return {
        "control_type": "issued_saturn_same_zone_same_delivery_day",
        "archive_origin_path": str(control["archive_path"]),
        "zone": control["zone"],
        "delivery_day_local": control["delivery_day_local"],
        "checksum_manifest_sha256": control["checksum_manifest_sha256"],
        "run_manifest_sha256": control["run_manifest_sha256"],
        "aligned_inputs_sha256": control["files"]["aligned_inputs"]["sha256"],
        "model_covariates_sha256": control["files"][
            "model_covariates_with_future"
        ]["sha256"],
        "archived_directory": SEALED_SATURN_CONTROL_DIRECTORY,
        "archived_files": copied,
        "aligned_inputs_reused_byte_for_byte": True,
        "model_context_base": "sealed_saturn_control",
        "only_overlaid_columns": treatment,
        "only_overlaid_scope": "delivery_day_j_plus_1",
        "overlaid_hours": len(delivery_index),
    }


def copy_sealed_saturn_primary(
    saturn_archive_dir: str | Path,
    *,
    destination: str | Path,
    expected_delivery_day: str | date | pd.Timestamp,
    expected_zone: str,
) -> dict[str, Any]:
    """Reuse MKOnline primary bytes from the paired Saturn archive."""

    control = validate_sealed_saturn_control_archive(
        saturn_archive_dir,
        expected_delivery_day=expected_delivery_day,
        expected_zone=expected_zone,
        require_primary=True,
    )
    source = Path(control["files"]["mkonline_primary_live"]["path"])
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise BundleValidationError(
            f"Destination MKOnline deja existante: {target}."
        )
    shutil.copy2(source, target)
    expected_sha = control["files"]["mkonline_primary_live"]["sha256"]
    if _sha256(target) != expected_sha:
        raise BundleValidationError(
            "Copie MKOnline divergente du controle Saturn scelle."
        )
    return {
        "source": "sealed_saturn_control",
        "source_archive_path": str(control["archive_path"]),
        "source_relative_path": SEALED_SATURN_PRIMARY,
        "source_sha256": expected_sha,
        "copied_byte_for_byte": True,
    }


def _configured_production_pit_paths(
    config: Mapping[str, Any],
    *,
    config_dir: Path,
) -> dict[str, Path]:
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise BundleValidationError(
            "config.data est requis pour resoudre les PIT de production."
        )
    configured = data.get("pit_files")
    if not isinstance(configured, Mapping):
        raise BundleValidationError(
            "config.data.pit_files est requis pour composer l'overlay."
        )
    pit_root = Path(str(data.get("pit_vintage_dir", "."))).expanduser()
    if not pit_root.is_absolute():
        pit_root = config_dir / pit_root
    pit_root = pit_root.resolve()

    paths: dict[str, Path] = {}
    for alias in EXPECTED_ALIASES:
        configured_path = configured.get(alias)
        if configured_path in (None, ""):
            raise BundleValidationError(
                f"config.data.pit_files.{alias} est absent."
            )
        path = Path(str(configured_path)).expanduser()
        if not path.is_absolute():
            path = pit_root / path
        path = path.resolve()
        if not path.is_file():
            raise BundleValidationError(
                f"PIT de production absent pour {alias}: {path}"
            )
        paths[alias] = path
    return paths


def _read_pit_frame(path: Path, *, label: str) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise BundleValidationError(f"{label}: parquet illisible: {path}") from exc
    if not isinstance(frame.index, pd.RangeIndex):
        frame = frame.reset_index(drop=True)
    required = {
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
        "value",
    }
    if not required.issubset(frame.columns):
        raise BundleValidationError(
            f"{label}: colonnes PIT absentes: {sorted(required - set(frame.columns))}"
        )
    result = frame.copy()
    for column in ("value_time_utc", "snapshot_time_utc", "revision_time_utc"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    timestamp_columns = [
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
    ]
    if result[timestamp_columns].isna().any().any():
        raise BundleValidationError(
            f"{label}: timestamps PIT obligatoires manquants."
        )
    return result


def _assert_frames_equal(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    label: str,
) -> None:
    try:
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_like=False,
        )
    except AssertionError as exc:
        raise BundleValidationError(f"{label}: contenu divergent.") from exc


def _resolve_relative_artifact_reference(
    root: Path,
    declared_path: Any,
    *,
    label: str,
) -> Path:
    """Resolve one portable, bundle-relative artifact declaration."""

    raw = str(declared_path or "")
    relative = Path(raw)
    if not raw or relative.is_absolute():
        raise BundleValidationError(f"{label}: le chemin doit etre relatif.")
    base = root.resolve()
    artifact = (base / relative).resolve()
    try:
        artifact.relative_to(base)
    except ValueError as exc:
        raise BundleValidationError(f"{label}: chemin hors dossier.") from exc
    return artifact


def _validate_composite_overlay(
    audit_path: Path,
    *,
    upstream_manifest_path: Path,
    upstream_paths: Mapping[str, Path],
    production_paths: Mapping[str, Path],
    delivery_index: pd.DatetimeIndex,
    cutoff: pd.Timestamp,
) -> tuple[dict[str, Any], dict[str, Path]]:
    audit = _json_object(audit_path)
    expected_top_level = {
        "schema_version": 1,
        "overlay_type": "chronos2_residual_load_control_history_overlay",
        "historical_context_source": "production_pit_unchanged",
        "delivery_day_values_source": "chronos2",
        "saturn_forecast_rows_on_delivery_day": 0,
        "upstream_manifest_path": str(upstream_manifest_path),
        "upstream_manifest_sha256": _sha256(upstream_manifest_path),
        "delivery_start_utc": _iso_utc(delivery_index[0]),
        "delivery_end_utc": _iso_utc(delivery_index[-1]),
        "delivery_hours": len(delivery_index),
        "runtime_cutoff_utc": _iso_utc(cutoff),
    }
    for key, value in expected_top_level.items():
        if audit.get(key) != value:
            raise BundleValidationError(
                f"Overlay incompatible: {key}={audit.get(key)!r}, attendu {value!r}."
            )
    declarations = audit.get("artifacts")
    if not isinstance(declarations, list) or len(declarations) != len(
        EXPECTED_ALIASES
    ):
        raise BundleValidationError("Overlay: cinq artifacts sont requis.")
    by_alias = {
        str(item.get("alias", "")): item
        for item in declarations
        if isinstance(item, dict)
    }
    if set(by_alias) != set(EXPECTED_ALIASES):
        raise BundleValidationError("Overlay: aliases declares invalides.")

    overlay_paths: dict[str, Path] = {}
    for alias in EXPECTED_ALIASES:
        declaration = by_alias[alias]
        production_path = production_paths[alias]
        upstream_path = upstream_paths[alias]
        expected_fields = {
            "production_pit_path": str(production_path),
            "production_pit_sha256": _sha256(production_path),
            "upstream_forecast_path": str(upstream_path),
            "upstream_forecast_sha256": _sha256(upstream_path),
        }
        for key, value in expected_fields.items():
            if declaration.get(key) != value:
                raise BundleValidationError(
                    f"Overlay {alias}: {key} divergent."
                )
        overlay_path = _resolve_relative_artifact_reference(
            audit_path.parent,
            declaration.get("path"),
            label=f"Overlay {alias}",
        )
        if not overlay_path.is_file():
            raise BundleValidationError(
                f"Overlay {alias}: parquet absent: {overlay_path}"
            )
        if _sha256(overlay_path) != declaration.get("sha256"):
            raise BundleValidationError(f"Overlay {alias}: SHA-256 divergent.")

        production = _read_pit_frame(
            production_path, label=f"production {alias}"
        )
        upstream = _read_pit_frame(upstream_path, label=f"upstream {alias}")
        composite = _read_pit_frame(overlay_path, label=f"overlay {alias}")
        production_times = pd.DatetimeIndex(production["value_time_utc"])
        historical = production.loc[
            production_times < delivery_index[0]
        ].reset_index(drop=True)
        if declaration.get("historical_rows") != len(historical):
            raise BundleValidationError(
                f"Overlay {alias}: historical_rows divergent."
            )
        if declaration.get("delivery_day_rows") != len(delivery_index):
            raise BundleValidationError(
                f"Overlay {alias}: delivery_day_rows divergent."
            )
        if declaration.get("total_rows") != len(historical) + len(delivery_index):
            raise BundleValidationError(
                f"Overlay {alias}: total_rows divergent."
            )
        removed_rows = int((production_times >= delivery_index[0]).sum())
        if declaration.get(
            "removed_production_rows_at_or_after_delivery_start"
        ) != removed_rows:
            raise BundleValidationError(
                f"Overlay {alias}: nombre de lignes production retirees divergent."
            )
        if len(composite) != len(historical) + len(delivery_index):
            raise BundleValidationError(
                f"Overlay {alias}: nombre total de lignes divergent."
            )
        _assert_frames_equal(
            composite.iloc[: len(historical)][list(production.columns)],
            historical,
            label=f"Overlay {alias} historique de controle",
        )
        delivery = composite.iloc[len(historical) :].reset_index(drop=True)
        delivery_times = pd.DatetimeIndex(delivery["value_time_utc"])
        if not delivery_times.equals(delivery_index):
            raise BundleValidationError(
                f"Overlay {alias}: timeline J+1 divergente."
            )
        common_columns = [
            column for column in upstream.columns if column in delivery.columns
        ]
        _assert_frames_equal(
            delivery[common_columns],
            upstream[common_columns],
            label=f"Overlay {alias} valeurs J+1 Chronos",
        )
        delivery_snapshot = pd.DatetimeIndex(delivery["snapshot_time_utc"])
        delivery_revision = pd.DatetimeIndex(delivery["revision_time_utc"])
        expected_cutoffs = pd.DatetimeIndex([cutoff] * len(delivery))
        if not delivery_snapshot.equals(expected_cutoffs) or not delivery_revision.equals(
            expected_cutoffs
        ):
            raise BundleValidationError(
                f"Overlay {alias}: cutoff PIT J+1 divergent."
            )
        overlay_paths[alias] = overlay_path
    return audit, overlay_paths


def validate_archived_residual_load_overlay(
    archive_inputs_dir: str | Path,
    *,
    upstream_manifest_path: str | Path,
    upstream_origin_manifest_path: str | Path | None = None,
    expected_delivery_day: str | date | pd.Timestamp,
    expected_runtime_cutoff: Any,
) -> dict[str, Any]:
    """Re-audit an archived composite without trusting mutable live PIT files.

    The archived overlay is checksum-sealed by the run itself. This validator
    additionally proves that its J+1 suffix is exactly the immutable upstream
    Chronos-2 bundle and that no production/Saturn forecast row survives on
    the delivery day. Historical production PIT files are deliberately not
    reopened because they may have advanced after this archive was published.
    """

    inputs_dir = Path(archive_inputs_dir).resolve()
    if not inputs_dir.is_dir():
        raise BundleValidationError(f"Dossier inputs archive absent: {inputs_dir}")
    upstream_manifest = Path(upstream_manifest_path).resolve()
    upstream_origin_manifest = Path(
        upstream_origin_manifest_path or upstream_manifest
    ).expanduser().resolve()
    cutoff = _utc_timestamp(
        expected_runtime_cutoff,
        name="expected_runtime_cutoff",
    )
    delivery_index = _delivery_index_utc(expected_delivery_day)
    validate_live_residual_load_bundle(
        upstream_manifest,
        expected_delivery_day=expected_delivery_day,
        expected_runtime_cutoff=cutoff,
    )
    upstream_paths = bundle_alias_paths(upstream_manifest)

    overlay_root = (inputs_dir / "residual_load_pit_overlay").resolve()
    try:
        overlay_root.relative_to(inputs_dir)
    except ValueError as exc:  # pragma: no cover - constant child path
        raise BundleValidationError("Racine overlay hors inputs archive.") from exc
    audit_path = overlay_root / "composite_manifest.json"
    audit = _json_object(audit_path)
    expected_top_level = {
        "schema_version": 1,
        "overlay_type": "chronos2_residual_load_control_history_overlay",
        "historical_context_source": "production_pit_unchanged",
        "delivery_day_values_source": "chronos2",
        "saturn_forecast_rows_on_delivery_day": 0,
        "upstream_manifest_path": str(upstream_origin_manifest),
        "upstream_manifest_sha256": _sha256(upstream_manifest),
        "delivery_start_utc": _iso_utc(delivery_index[0]),
        "delivery_end_utc": _iso_utc(delivery_index[-1]),
        "delivery_hours": len(delivery_index),
        "runtime_cutoff_utc": _iso_utc(cutoff),
    }
    for key, expected in expected_top_level.items():
        if audit.get(key) != expected:
            raise BundleValidationError(
                f"Overlay archive incompatible: {key}={audit.get(key)!r}, "
                f"attendu {expected!r}."
            )

    declarations = audit.get("artifacts")
    if not isinstance(declarations, list) or len(declarations) != len(
        EXPECTED_ALIASES
    ):
        raise BundleValidationError("Overlay archive: cinq artifacts sont requis.")
    by_alias = {
        str(item.get("alias", "")): item
        for item in declarations
        if isinstance(item, dict)
    }
    if set(by_alias) != set(EXPECTED_ALIASES):
        raise BundleValidationError("Overlay archive: aliases declares invalides.")

    upstream_payload = validate_live_residual_load_bundle(upstream_manifest)
    upstream_declarations = {
        str(item["alias"]): item for item in upstream_payload["artifacts"]
    }
    validated_files: dict[str, dict[str, Any]] = {}
    for alias in EXPECTED_ALIASES:
        declaration = by_alias[alias]
        upstream_path = upstream_paths[alias]
        origin_upstream_path = (
            upstream_origin_manifest.parent
            / Path(str(upstream_declarations[alias]["path"]))
        ).resolve()
        expected_upstream_fields = {
            "upstream_forecast_path": str(origin_upstream_path),
            "upstream_forecast_sha256": _sha256(upstream_path),
        }
        for key, expected in expected_upstream_fields.items():
            if declaration.get(key) != expected:
                raise BundleValidationError(
                    f"Overlay archive {alias}: {key} divergent."
                )
        production_sha = str(
            declaration.get("production_pit_sha256") or ""
        ).lower()
        if len(production_sha) != 64 or any(
            character not in "0123456789abcdef" for character in production_sha
        ):
            raise BundleValidationError(
                f"Overlay archive {alias}: production_pit_sha256 invalide."
            )
        if not str(declaration.get("production_pit_path") or "").strip():
            raise BundleValidationError(
                f"Overlay archive {alias}: production_pit_path absent."
            )

        overlay_path = _resolve_relative_artifact_reference(
            overlay_root,
            declaration.get("path"),
            label=f"Overlay archive {alias}",
        )
        if not overlay_path.is_file():
            raise BundleValidationError(
                f"Overlay archive {alias}: parquet absent: {overlay_path}"
            )
        declared_sha = str(declaration.get("sha256") or "").lower()
        if len(declared_sha) != 64 or _sha256(overlay_path) != declared_sha:
            raise BundleValidationError(
                f"Overlay archive {alias}: SHA-256 divergent."
            )

        upstream = _read_pit_frame(upstream_path, label=f"upstream {alias}")
        composite = _read_pit_frame(overlay_path, label=f"overlay archive {alias}")
        historical_rows = declaration.get("historical_rows")
        delivery_rows = declaration.get("delivery_day_rows")
        total_rows = declaration.get("total_rows")
        if not isinstance(historical_rows, int) or historical_rows <= 0:
            raise BundleValidationError(
                f"Overlay archive {alias}: historical_rows invalide."
            )
        if delivery_rows != len(delivery_index):
            raise BundleValidationError(
                f"Overlay archive {alias}: delivery_day_rows divergent."
            )
        if total_rows != historical_rows + len(delivery_index):
            raise BundleValidationError(
                f"Overlay archive {alias}: total_rows divergent."
            )
        if len(composite) != total_rows:
            raise BundleValidationError(
                f"Overlay archive {alias}: nombre de lignes divergent."
            )
        removed_rows = declaration.get(
            "removed_production_rows_at_or_after_delivery_start"
        )
        if not isinstance(removed_rows, int) or removed_rows < 0:
            raise BundleValidationError(
                f"Overlay archive {alias}: nombre de lignes retirees invalide."
            )

        historical = composite.iloc[:historical_rows]
        historical_times = pd.DatetimeIndex(historical["value_time_utc"])
        if historical_times.empty or not bool(
            (historical_times < delivery_index[0]).all()
        ):
            raise BundleValidationError(
                f"Overlay archive {alias}: contexte historique non causal."
            )
        delivery = composite.iloc[historical_rows:].reset_index(drop=True)
        delivery_times = pd.DatetimeIndex(delivery["value_time_utc"])
        if not delivery_times.equals(delivery_index):
            raise BundleValidationError(
                f"Overlay archive {alias}: timeline J+1 divergente."
            )
        missing_upstream_columns = sorted(
            set(upstream.columns).difference(delivery.columns)
        )
        if missing_upstream_columns:
            raise BundleValidationError(
                f"Overlay archive {alias}: colonnes Chronos absentes: "
                f"{missing_upstream_columns}."
            )
        common_columns = [
            column for column in upstream.columns if column in delivery.columns
        ]
        _assert_frames_equal(
            delivery[common_columns],
            upstream[common_columns],
            label=f"Overlay archive {alias} valeurs J+1 Chronos",
        )
        expected_cutoffs = pd.DatetimeIndex([cutoff] * len(delivery))
        if not pd.DatetimeIndex(delivery["snapshot_time_utc"]).equals(
            expected_cutoffs
        ) or not pd.DatetimeIndex(delivery["revision_time_utc"]).equals(
            expected_cutoffs
        ):
            raise BundleValidationError(
                f"Overlay archive {alias}: cutoff PIT J+1 divergent."
            )
        validated_files[alias] = {
            "path": overlay_path,
            "sha256": declared_sha,
            "historical_rows": historical_rows,
            "delivery_day_rows": delivery_rows,
        }

    return {
        "manifest_path": audit_path,
        "manifest_sha256": _sha256(audit_path),
        "upstream_manifest_path": upstream_manifest,
        "upstream_manifest_sha256": _sha256(upstream_manifest),
        "upstream_origin_manifest_path": str(upstream_origin_manifest),
        "audit": audit,
        "files": validated_files,
    }


def _build_composite_overlay(
    *,
    overlay_dir: Path,
    upstream_manifest_path: Path,
    upstream_paths: Mapping[str, Path],
    production_paths: Mapping[str, Path],
    delivery_index: pd.DatetimeIndex,
    cutoff: pd.Timestamp,
) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    audit_path = overlay_dir / "composite_manifest.json"
    if overlay_dir.exists():
        audit, paths = _validate_composite_overlay(
            audit_path,
            upstream_manifest_path=upstream_manifest_path,
            upstream_paths=upstream_paths,
            production_paths=production_paths,
            delivery_index=delivery_index,
            cutoff=cutoff,
        )
        return audit_path.resolve(), audit, paths

    overlay_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            dir=overlay_dir.parent, prefix=f".{overlay_dir.name}."
        )
    )
    try:
        declarations: list[dict[str, Any]] = []
        for alias in EXPECTED_ALIASES:
            production_path = production_paths[alias]
            upstream_path = upstream_paths[alias]
            production = _read_pit_frame(
                production_path, label=f"production {alias}"
            )
            upstream = _read_pit_frame(
                upstream_path, label=f"upstream {alias}"
            )
            production_times = pd.DatetimeIndex(production["value_time_utc"])
            historical = production.loc[
                production_times < delivery_index[0]
            ].reset_index(drop=True)
            if historical.empty:
                raise BundleValidationError(
                    f"{alias}: historique PIT de controle vide avant J+1."
                )
            removed_rows = int((production_times >= delivery_index[0]).sum())
            columns = list(production.columns)
            columns.extend(
                column for column in upstream.columns if column not in columns
            )
            combined_columns: dict[str, pd.Series] = {}
            for column in columns:
                if column in historical:
                    historical_part = historical[column].reset_index(drop=True)
                else:
                    historical_part = pd.Series(
                        np.nan,
                        index=pd.RangeIndex(len(historical)),
                        dtype=float,
                    )
                if column in upstream:
                    upstream_part = upstream[column].reset_index(drop=True)
                elif isinstance(historical_part.dtype, pd.DatetimeTZDtype):
                    upstream_part = pd.Series(
                        pd.NaT,
                        index=pd.RangeIndex(len(upstream)),
                        dtype=historical_part.dtype,
                    )
                elif pd.api.types.is_numeric_dtype(historical_part.dtype):
                    upstream_part = pd.Series(
                        np.nan,
                        index=pd.RangeIndex(len(upstream)),
                        dtype=float,
                    )
                else:
                    upstream_part = pd.Series(
                        pd.NA,
                        index=pd.RangeIndex(len(upstream)),
                        dtype="object",
                    )
                combined_columns[column] = pd.concat(
                    [historical_part, upstream_part], ignore_index=True
                )
            composite = pd.DataFrame(combined_columns, columns=columns)
            artifact = temporary / f"{alias}.parquet"
            composite.to_parquet(artifact, index=False)
            declarations.append(
                {
                    "alias": alias,
                    "path": artifact.name,
                    "sha256": _sha256(artifact),
                    "historical_rows": len(historical),
                    "delivery_day_rows": len(upstream),
                    "total_rows": len(composite),
                    "removed_production_rows_at_or_after_delivery_start": removed_rows,
                    "production_pit_path": str(production_path),
                    "production_pit_sha256": _sha256(production_path),
                    "upstream_forecast_path": str(upstream_path),
                    "upstream_forecast_sha256": _sha256(upstream_path),
                }
            )
        audit: dict[str, Any] = {
            "schema_version": 1,
            "overlay_type": "chronos2_residual_load_control_history_overlay",
            "historical_context_source": "production_pit_unchanged",
            "delivery_day_values_source": "chronos2",
            "saturn_forecast_rows_on_delivery_day": 0,
            "upstream_manifest_path": str(upstream_manifest_path),
            "upstream_manifest_sha256": _sha256(upstream_manifest_path),
            "delivery_start_utc": _iso_utc(delivery_index[0]),
            "delivery_end_utc": _iso_utc(delivery_index[-1]),
            "delivery_hours": len(delivery_index),
            "runtime_cutoff_utc": _iso_utc(cutoff),
            "artifacts": declarations,
        }
        temporary_audit = temporary / "composite_manifest.json"
        temporary_audit.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _validate_composite_overlay(
            temporary_audit,
            upstream_manifest_path=upstream_manifest_path,
            upstream_paths=upstream_paths,
            production_paths=production_paths,
            delivery_index=delivery_index,
            cutoff=cutoff,
        )
        try:
            temporary.replace(overlay_dir)
        except FileExistsError:
            pass
        final_audit, final_paths = _validate_composite_overlay(
            audit_path,
            upstream_manifest_path=upstream_manifest_path,
            upstream_paths=upstream_paths,
            production_paths=production_paths,
            delivery_index=delivery_index,
            cutoff=cutoff,
        )
        return audit_path.resolve(), final_audit, final_paths
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def apply_residual_load_bundle(
    config: dict[str, Any],
    *,
    manifest_path: str | Path,
    expected_delivery_index: pd.DatetimeIndex,
    expected_cutoff_utc: Any,
    config_dir: str | Path,
    overlay_dir: str | Path,
    saturn_control_archive: str | Path,
    expected_zone: str,
) -> dict[str, Any]:
    """Apply a validated bundle to an already-copied runtime config.

    The supplied ``config`` is intentionally mutated in place.  Callers must
    pass their ephemeral/deep-copied live configuration, never the object that
    represents the frozen YAML contract.  The return value is provenance only.
    """

    if not isinstance(config, dict):
        raise TypeError("config doit etre un dictionnaire mutable.")
    base = Path(config_dir).resolve()
    path = Path(manifest_path)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    overlay = Path(overlay_dir)
    if not overlay.is_absolute():
        overlay = base / overlay
    overlay = overlay.resolve()
    cutoff = _utc_timestamp(expected_cutoff_utc, name="expected_cutoff_utc")

    supplied_index = pd.DatetimeIndex(expected_delivery_index)
    if supplied_index.tz is None:
        raise BundleValidationError(
            "expected_delivery_index doit contenir un fuseau horaire."
        )
    supplied_index = supplied_index.tz_convert("UTC")
    if not supplied_index.is_unique or not supplied_index.is_monotonic_increasing:
        raise BundleValidationError(
            "expected_delivery_index doit etre unique et strictement ordonne."
        )
    if len(supplied_index) not in (23, 24, 25):
        raise BundleValidationError(
            "expected_delivery_index doit contenir 23, 24 ou 25 heures."
        )
    if len(supplied_index) > 1 and not (
        supplied_index[1:] - supplied_index[:-1] == pd.Timedelta(hours=1)
    ).all():
        raise BundleValidationError(
            "expected_delivery_index doit etre une grille UTC horaire reguliere."
        )

    manifest = validate_live_residual_load_bundle(
        path,
        expected_runtime_cutoff=cutoff,
    )
    manifest_index = _delivery_index_utc(manifest["delivery_day_local"])
    if not supplied_index.equals(manifest_index):
        raise BundleValidationError(
            "La timeline attendue ne correspond pas au jour civil du bundle."
        )
    upstream_paths = bundle_alias_paths(path)
    control = validate_sealed_saturn_control_archive(
        saturn_control_archive,
        expected_delivery_day=manifest["delivery_day_local"],
        expected_zone=expected_zone,
    )

    existing_data = config.get("data")
    if existing_data is not None and not isinstance(existing_data, dict):
        raise TypeError("config.data doit etre un dictionnaire mutable.")
    existing_pit_files = (
        existing_data.get("pit_files") if existing_data is not None else None
    )
    if existing_pit_files is not None and not isinstance(
        existing_pit_files, dict
    ):
        raise TypeError("config.data.pit_files doit etre un dictionnaire mutable.")
    zones = config.get("zones")
    if not isinstance(zones, dict) or not zones:
        raise BundleValidationError("config.zones doit contenir au moins une zone.")

    covariate_configs: list[dict[str, Any]] = []
    for zone, zone_config in zones.items():
        if not isinstance(zone_config, dict):
            raise TypeError(f"config.zones.{zone} doit etre un dictionnaire.")
        covariates = zone_config.get("covariates")
        if not isinstance(covariates, dict):
            raise BundleValidationError(
                f"config.zones.{zone}.covariates est absent ou invalide."
            )
        missing = sorted(set(EXPECTED_ALIASES) - set(covariates))
        if missing:
            raise BundleValidationError(
                f"config.zones.{zone}: covariates residual-load absentes: {missing}"
            )
        for alias in EXPECTED_ALIASES:
            raw = covariates[alias]
            if not isinstance(raw, dict):
                raise TypeError(
                    f"config.zones.{zone}.covariates.{alias} doit etre un dictionnaire."
                )
            covariate_configs.append(raw)

    production_paths = _materialize_sealed_saturn_pit_controls(
        control,
        destination=overlay.parent / "sealed_saturn_control_pit",
    )
    if any(
        overlay == production.parent or overlay in production.parents
        for production in production_paths.values()
    ):
        raise BundleValidationError(
            "overlay_dir ne doit pas contenir un PIT de production."
        )
    composite_manifest, composite_audit, composite_paths = (
        _build_composite_overlay(
            overlay_dir=overlay,
            upstream_manifest_path=path,
            upstream_paths=upstream_paths,
            production_paths=production_paths,
            delivery_index=supplied_index,
            cutoff=cutoff,
        )
    )

    data = config.setdefault("data", {})
    pit_files = data.setdefault("pit_files", {})
    for alias, artifact in composite_paths.items():
        pit_files[alias] = str(artifact)
    for raw in covariate_configs:
        raw["series"] = None
        raw["source"] = "pit_parquet"
        raw.pop("pit_file", None)

    declarations = {
        str(item["alias"]): item for item in manifest["artifacts"]
    }
    composite_declarations = {
        str(item["alias"]): item for item in composite_audit["artifacts"]
    }
    provenance_base = overlay.parent.resolve()

    def stable_composite_reference(artifact: Path) -> str:
        try:
            relative = artifact.resolve().relative_to(provenance_base)
        except ValueError as exc:  # pragma: no cover - guarded by the builder
            raise BundleValidationError(
                "Un artifact composite est hors du dossier runtime inputs."
            ) from exc
        return relative.as_posix()

    return {
        "provider": "chronos2",
        "target_source_kind": "observed_entsoe",
        "saturn_forecast_series_used": False,
        "historical_context_source": "production_pit_unchanged",
        "effective_historical_context_source": (
            "sealed_saturn_same_zone_same_delivery_day"
        ),
        "delivery_day_values_source": "chronos2",
        "manifest_path": str(path),
        "manifest_sha256": _sha256(path),
        "archived_manifest_path": ARCHIVED_BUNDLE_MANIFEST,
        "archived_manifest_sha256": _sha256(path),
        "composite_path_base": "runtime_inputs_directory",
        "composite_manifest_path": stable_composite_reference(
            composite_manifest
        ),
        "composite_manifest_sha256": _sha256(composite_manifest),
        "delivery_day_local": manifest["delivery_day_local"],
        "runtime_cutoff_utc": manifest["runtime_cutoff_utc"],
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "sealed_saturn_control": {
            "archive_origin_path": str(control["archive_path"]),
            "zone": control["zone"],
            "delivery_day_local": control["delivery_day_local"],
            "checksum_manifest_sha256": control[
                "checksum_manifest_sha256"
            ],
            "run_manifest_sha256": control["run_manifest_sha256"],
            "aligned_inputs_sha256": control["files"][
                "aligned_inputs"
            ]["sha256"],
            "model_covariates_sha256": control["files"][
                "model_covariates_with_future"
            ]["sha256"],
        },
        "files": {
            alias: {
                "path": stable_composite_reference(composite_paths[alias]),
                "sha256": str(composite_declarations[alias]["sha256"]),
                "historical_rows": int(
                    composite_declarations[alias]["historical_rows"]
                ),
                "delivery_day_rows": int(
                    composite_declarations[alias]["delivery_day_rows"]
                ),
                "production_pit_path": str(production_paths[alias]),
                "production_pit_sha256": str(
                    composite_declarations[alias]["production_pit_sha256"]
                ),
                "upstream_forecast_path": str(upstream_paths[alias]),
                "upstream_forecast_sha256": str(
                    declarations[alias]["sha256"]
                ),
                "archived_upstream_forecast_path": (
                    Path(ARCHIVED_BUNDLE_DIRECTORY)
                    / Path(str(declarations[alias]["path"]))
                ).as_posix(),
            }
            for alias in EXPECTED_ALIASES
        },
    }


def build_live_residual_load_bundle(
    *,
    delivery_day: str | date | pd.Timestamp,
    runtime_cutoff: Any,
    output_root: str | Path,
    saturn_url: str,
    saturn_author: str | None = None,
    device: str = "auto",
    local_files_only: bool = False,
    batch_size: int = 8,
) -> Path:
    """Build or reuse a causal, immutable five-country live PIT bundle.

    Saturn is queried only for realized ENTSO-E ``.obs`` residual-load
    targets, always with ``revision_date=runtime_cutoff``.  All future values
    in the returned PIT artifacts are generated by the pinned Chronos-2 model.
    """

    cutoff = _utc_timestamp(runtime_cutoff, name="runtime_cutoff")
    delivery_index = _delivery_index_utc(delivery_day)
    if cutoff > _now_utc():
        raise ValueError(
            "runtime_cutoff est encore dans le futur; le bundle live ne peut "
            "pas etre publie causalement avant le cutoff D-1 08:00."
        )
    if cutoff >= delivery_index[0]:
        raise ValueError(
            "runtime_cutoff doit etre strictement anterieur au jour de livraison."
        )
    if int(batch_size) <= 0:
        raise ValueError("batch_size doit etre strictement positif.")

    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    bundle_dir = _bundle_directory(root, delivery_day, cutoff)
    manifest_path = bundle_dir / "manifest.json"
    if bundle_dir.exists():
        return_path = manifest_path.resolve()
        existing = validate_live_residual_load_bundle(
            return_path,
            expected_delivery_day=delivery_day,
            expected_runtime_cutoff=cutoff,
        )
        expected_endpoint_sha = hashlib.sha256(
            saturn_url.encode("utf-8")
        ).hexdigest()
        if existing.get("saturn_endpoint_sha256") != expected_endpoint_sha:
            raise BundleValidationError(
                "Le bundle existant provient d'un autre endpoint Saturn."
            )
        LOGGER.info("Bundle residual-load Chronos-2 reutilise: %s", return_path)
        return return_path

    author = saturn_author or os.getenv("SATURN_AUTHOR", "")
    client = _create_saturn_client(saturn_url, author)
    history_start = cutoff.floor("h") - pd.Timedelta(
        hours=CONTEXT_LENGTH + 24 * 14
    )
    contexts: dict[str, pd.Series] = {}
    input_audits: dict[str, dict[str, Any]] = {}
    for country, source_series in COUNTRY_OBSERVED_SERIES.items():
        alias = COUNTRY_ALIASES[country]
        raw = _fetch_observed_asof(
            client,
            source_series,
            start=history_start,
            end=cutoff,
            runtime_cutoff=cutoff,
        )
        context, audit = _prepare_context(
            raw,
            series_name=source_series,
            runtime_cutoff=cutoff,
        )
        contexts[alias] = context
        input_audits[alias] = audit

    pipeline = _load_pipeline(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        device=device,
        local_files_only=local_files_only,
    )
    forecasts, forecast_audits = _forecast_delivery_day(
        pipeline,
        contexts,
        delivery_index,
        batch_size=int(batch_size),
    )

    temporary = Path(tempfile.mkdtemp(dir=root, prefix=f".{bundle_dir.name}."))
    try:
        artifact_declarations: list[dict[str, Any]] = []
        for country in COUNTRY_OBSERVED_SERIES:
            alias = COUNTRY_ALIASES[country]
            prediction = forecasts[alias]
            artifact = temporary / f"{alias}.parquet"
            frame = pd.DataFrame(
                {
                    "value_time_utc": delivery_index,
                    "snapshot_time_utc": pd.DatetimeIndex(
                        [cutoff] * len(delivery_index)
                    ),
                    "revision_time_utc": pd.DatetimeIndex(
                        [cutoff] * len(delivery_index)
                    ),
                    "value": prediction["q50"].to_numpy(dtype=float),
                    "q10": prediction["q10"].to_numpy(dtype=float),
                    "q50": prediction["q50"].to_numpy(dtype=float),
                    "q90": prediction["q90"].to_numpy(dtype=float),
                }
            )
            frame.to_parquet(artifact, index=False)
            artifact_declarations.append(
                {
                    "alias": alias,
                    "path": artifact.name,
                    "sha256": _sha256(artifact),
                    "rows": len(frame),
                    "value_start_utc": _iso_utc(delivery_index[0]),
                    "value_end_utc": _iso_utc(delivery_index[-1]),
                    "snapshot_time_utc": _iso_utc(cutoff),
                    "revision_time_utc": _iso_utc(cutoff),
                }
            )

        manifest: dict[str, Any] = {
            **_semantic_identity(delivery_day, cutoff),
            "target_source_provider": "saturn",
            "target_source_kind": "observed_entsoe",
            "saturn_endpoint_sha256": hashlib.sha256(
                saturn_url.encode("utf-8")
            ).hexdigest(),
            "saturn_forecast_series_used": False,
            "forecast_values_provider": "chronos2",
            "execution": {
                "requested_device": device,
                "local_files_only": bool(local_files_only),
                "batch_size": int(batch_size),
            },
            "inputs": [],
            "artifacts": artifact_declarations,
        }
        for country, source_series in COUNTRY_OBSERVED_SERIES.items():
            alias = COUNTRY_ALIASES[country]
            manifest["inputs"].append(
                {
                    "country": country.upper(),
                    "alias": alias,
                    "source_series": source_series,
                    "source_kind": "observed_entsoe_asof",
                    **input_audits[alias],
                    **forecast_audits[alias],
                }
            )
        temporary_manifest = temporary / "manifest.json"
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_live_residual_load_bundle(
            temporary_manifest,
            expected_delivery_day=delivery_day,
            expected_runtime_cutoff=cutoff,
        )
        try:
            temporary.replace(bundle_dir)
        except FileExistsError:
            validate_live_residual_load_bundle(
                manifest_path,
                expected_delivery_day=delivery_day,
                expected_runtime_cutoff=cutoff,
            )
        return manifest_path.resolve()
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


__all__ = [
    "BundleValidationError",
    "CONTEXT_LENGTH",
    "COUNTRY_ALIASES",
    "COUNTRY_OBSERVED_SERIES",
    "EXPECTED_ALIASES",
    "MODEL_ID",
    "MODEL_REVISION",
    "ResidualLoadBundleError",
    "apply_residual_load_bundle",
    "build_live_residual_load_bundle",
    "bundle_alias_paths",
    "copy_sealed_saturn_primary",
    "planned_live_residual_load_manifest_path",
    "seal_challenger_zone_data_from_saturn_control",
    "validate_archived_residual_load_overlay",
    "validate_live_residual_load_bundle",
    "validate_sealed_saturn_control_archive",
]
