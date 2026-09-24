"""Checkpointed Chronos price replay for a fixed rolling-origin protocol.

The expensive model execution is intentionally injected.  This module owns
only the immutable protocol around it: exact civil delivery days, resumable
prefix checkpoints, target parity and a checksum-sealed sidecar.  It is used
by the residual-load source comparison so both branches execute the same
price model on the same origins.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .chronos_adapter import (
    ChronosDeliveryPlan,
    load_chronos_oof,
    normalize_chronos_oof,
)


SCHEMA_VERSION = 1


class HistoricalPriceReplayError(RuntimeError):
    """Raised when a price replay or one of its checkpoints is incompatible."""


ChunkExecutor = Callable[[Sequence[ChronosDeliveryPlan]], pd.DataFrame]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def _expected_index(
    plans: Sequence[ChronosDeliveryPlan],
) -> pd.DatetimeIndex:
    if not plans:
        raise ValueError("Le replay prix exige au moins un jour de livraison.")
    values = np.concatenate([plan.delivery_index_utc.asi8 for plan in plans])
    result = pd.DatetimeIndex(
        values,
        tz="UTC",
        name="delivery_start_utc",
    )
    if result.has_duplicates or not result.is_monotonic_increasing:
        raise HistoricalPriceReplayError(
            "Les plans prix ne forment pas une timeline UTC unique et croissante."
        )
    return result


def _expected_origins(
    plans: Sequence[ChronosDeliveryPlan],
) -> pd.DatetimeIndex:
    _expected_index(plans)
    values = np.concatenate(
        [
            np.repeat(plan.forecast_origin_utc.value, plan.horizon)
            for plan in plans
        ]
    )
    return pd.DatetimeIndex(values, tz="UTC", name="forecast_origin_utc")


def _target_utc(target: pd.Series) -> pd.Series:
    if not isinstance(target, pd.Series) or target.empty:
        raise TypeError("target doit etre une Series pandas non vide.")
    if not isinstance(target.index, pd.DatetimeIndex) or target.index.tz is None:
        raise TypeError("target doit avoir un DatetimeIndex timezone-aware.")
    if target.index.has_duplicates or not target.index.is_monotonic_increasing:
        raise ValueError("target doit etre unique et croissante.")
    result = pd.to_numeric(target, errors="coerce").astype(float).copy()
    result.index = result.index.tz_convert("UTC")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("target contient une valeur manquante ou non finie.")
    return result


def replay_identity(
    plans: Sequence[ChronosDeliveryPlan],
    *,
    zone: str,
    timezone: str,
    model_id: str,
    model_revision: str,
    residual_load_source: str,
    source_hashes: Mapping[str, str],
    feature_schema_sha256: str,
    execution_signature: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the immutable semantic identity stored beside every checkpoint."""

    expected = _expected_index(plans)
    expected_origins = _expected_origins(plans)
    origin_sha256 = hashlib.sha256(
        expected_origins.asi8.astype("<i8", copy=False).tobytes()
    ).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "chronos2_price_rolling_origin_replay",
        "zone": str(zone).upper(),
        "timezone": str(timezone),
        "first_delivery_day_local": plans[0].delivery_date.isoformat(),
        "last_delivery_day_local": plans[-1].delivery_date.isoformat(),
        "n_delivery_days": len(plans),
        "n_delivery_hours": len(expected),
        "first_delivery_utc": str(expected[0]),
        "last_delivery_utc": str(expected[-1]),
        "first_forecast_origin_utc": str(expected_origins[0]),
        "last_forecast_origin_utc": str(expected_origins[-1]),
        "forecast_origins_sha256": origin_sha256,
        "model_id": str(model_id),
        "model_revision": str(model_revision),
        "residual_load_source": str(residual_load_source),
        "source_hashes": dict(sorted(source_hashes.items())),
        "feature_schema_sha256": str(feature_schema_sha256),
        "execution_signature": dict(execution_signature or {}),
        "with_covariates": True,
    }


def _manifest_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".manifest.json")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoricalPriceReplayError(f"Manifeste illisible: {path}") from exc
    if not isinstance(value, dict):
        raise HistoricalPriceReplayError(f"{path}: objet JSON attendu.")
    return value


def _require_identity(
    manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    path: Path,
) -> None:
    for key, value in identity.items():
        if manifest.get(key) != value:
            raise HistoricalPriceReplayError(
                f"{path}: identite incompatible pour {key}: "
                f"{manifest.get(key)!r} != {value!r}."
            )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(
                _json_safe(value),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _deterministic_csv_gzip_bytes(frame: pd.DataFrame) -> bytes:
    csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return gzip.compress(csv_bytes, compresslevel=9, mtime=0)


def _atomic_write_oof(frame: pd.DataFrame, path: Path) -> None:
    if not path.name.lower().endswith(".csv.gz"):
        raise ValueError("Le checkpoint prix doit etre un fichier .csv.gz.")
    path.parent.mkdir(parents=True, exist_ok=True)
    export = frame.reset_index()
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.csv.gz", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(_deterministic_csv_gzip_bytes(export))
        load_chronos_oof(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _completed_days(
    frame: pd.DataFrame,
    plans: Sequence[ChronosDeliveryPlan],
) -> int:
    cursor = 0
    completed = 0
    for plan in plans:
        next_cursor = cursor + plan.horizon
        if next_cursor <= len(frame):
            completed += 1
            cursor = next_cursor
            continue
        if cursor != len(frame):
            raise HistoricalPriceReplayError(
                "Le checkpoint prix s'arrete au milieu d'un jour civil."
            )
        break
    return completed


def _validate_prefix(
    frame: pd.DataFrame,
    plans: Sequence[ChronosDeliveryPlan],
    target: pd.Series,
) -> int:
    expected = _expected_index(plans)
    if len(frame) > len(expected) or not frame.index.equals(expected[: len(frame)]):
        raise HistoricalPriceReplayError(
            "Le checkpoint prix n'est pas un prefixe exact du protocole."
        )
    observed_origins = pd.DatetimeIndex(
        pd.to_datetime(frame["forecast_origin_utc"], utc=True, errors="raise")
    )
    expected_origins = _expected_origins(plans)[: len(frame)]
    if not observed_origins.equals(expected_origins):
        raise HistoricalPriceReplayError(
            "Les origines du checkpoint prix different du protocole."
        )
    target_values = _target_utc(target).reindex(frame.index)
    if target_values.isna().any() or not np.allclose(
        frame["actual"].to_numpy(dtype=float),
        target_values.to_numpy(dtype=float),
        rtol=0.0,
        atol=5e-5,
    ):
        raise HistoricalPriceReplayError(
            "La cible du checkpoint prix differe de la cible canonique."
        )
    return _completed_days(frame, plans)


def generate_checkpointed_price_replay(
    plans: Sequence[ChronosDeliveryPlan],
    *,
    target: pd.Series,
    execute_chunk: ChunkExecutor,
    output_path: str | Path,
    identity: Mapping[str, Any],
    chunk_days: int = 31,
    resume: bool = True,
) -> pd.DataFrame:
    """Generate an exact price replay, publishing every full-day prefix.

    ``execute_chunk`` must execute the supplied plans with the branch's
    already-built data object and pinned Chronos runtime.  A resumed run is
    accepted only when both its semantic identity and target are unchanged.
    """

    if chunk_days < 1:
        raise ValueError("chunk_days doit etre >= 1.")
    plans = tuple(plans)
    expected = _expected_index(plans)
    canonical_target = _target_utc(target)
    missing_target = expected.difference(canonical_target.index)
    if len(missing_target):
        raise HistoricalPriceReplayError(
            f"La cible prix ne couvre pas {len(missing_target)} heure(s)."
        )
    output = Path(output_path).expanduser().resolve()
    sidecar = _manifest_path(output)
    accumulated: pd.DataFrame | None = None
    completed = 0
    if output.exists() and resume:
        if not sidecar.is_file():
            raise HistoricalPriceReplayError(
                f"Checkpoint sans manifeste: {sidecar}"
            )
        manifest = _read_json(sidecar)
        _require_identity(manifest, identity, path=sidecar)
        if manifest.get("output_sha256") != _sha256(output):
            raise HistoricalPriceReplayError(
                f"Checksum du checkpoint prix invalide: {output}"
            )
        accumulated = load_chronos_oof(output)
        completed = _validate_prefix(accumulated, plans, canonical_target)

    pending = plans[completed:]
    for offset in range(0, len(pending), chunk_days):
        chunk = pending[offset : offset + chunk_days]
        generated = normalize_chronos_oof(execute_chunk(chunk))
        chunk_expected = _expected_index(chunk)
        if not generated.index.equals(chunk_expected):
            raise HistoricalPriceReplayError(
                "L'executeur prix n'a pas retourne exactement le chunk demande."
            )
        chunk_origins = _expected_origins(chunk)
        generated_origins = pd.DatetimeIndex(
            generated["forecast_origin_utc"]
        )
        if not generated_origins.equals(chunk_origins):
            raise HistoricalPriceReplayError(
                "L'executeur prix a retourne des origines differentes du protocole."
            )
        chunk_target = canonical_target.reindex(chunk_expected)
        if not np.allclose(
            generated["actual"].to_numpy(dtype=float),
            chunk_target.to_numpy(dtype=float),
            rtol=0.0,
            atol=5e-5,
        ):
            raise HistoricalPriceReplayError(
                "L'executeur prix a retourne une cible differente."
            )
        accumulated = normalize_chronos_oof(
            generated
            if accumulated is None
            else pd.concat(
                [accumulated.reset_index(), generated.reset_index()],
                ignore_index=True,
            )
        )
        completed += len(chunk)
        validated_completed = _validate_prefix(
            accumulated, plans, canonical_target
        )
        if validated_completed != completed:
            raise HistoricalPriceReplayError(
                "Le prefixe prix valide ne correspond pas aux jours executes."
            )
        _atomic_write_oof(accumulated, output)
        payload = {
            **dict(identity),
            "status": "complete" if completed == len(plans) else "running",
            "completed_days": completed,
            "completed_hours": len(accumulated),
            "output_path": str(output),
            "output_size_bytes": output.stat().st_size,
            "output_sha256": _sha256(output),
        }
        _atomic_write_json(sidecar, payload)

    if accumulated is None:
        raise HistoricalPriceReplayError("Le replay prix n'a produit aucune ligne.")
    if not accumulated.index.equals(expected):
        raise HistoricalPriceReplayError(
            "Le replay prix final ne couvre pas exactement le protocole."
        )
    _validate_prefix(accumulated, plans, canonical_target)
    return accumulated


def select_replay_days(
    replay: pd.DataFrame,
    plans: Sequence[ChronosDeliveryPlan],
) -> pd.DataFrame:
    """Return one exact contiguous subset from a complete replay."""

    expected = _expected_index(tuple(plans))
    selected = replay.reindex(expected)
    if selected.isna().any().any():
        raise HistoricalPriceReplayError(
            "Le replay prix ne couvre pas integralement la plage demandee."
        )
    return normalize_chronos_oof(selected.reset_index())


__all__ = [
    "HistoricalPriceReplayError",
    "generate_checkpointed_price_replay",
    "replay_identity",
    "select_replay_days",
]
