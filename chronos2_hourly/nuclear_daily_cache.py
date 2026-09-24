"""Reusable daily Chronos quantiles keyed by the actual model input frames.

The cache never stores realised prices.  The same issued-day prediction can
therefore become an OOF row later, with observations attached from that run.
Fingerprint construction delegates to the production frame builders, including
their point-in-time, lag, persistence and float32 conversion rules.
"""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .chronos_adapter import (
    ChronosDeliveryPlan,
    normalize_chronos_future,
    normalize_chronos_oof,
)


SCHEMA_VERSION = 1
QUANTILES = ("q10", "q50", "q90")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _frame_digest(frame: pd.DataFrame | None) -> str | None:
    if frame is None:
        return None
    frame = frame.drop(columns="item_id").reset_index(drop=True)
    digest = hashlib.sha256(_json_bytes({
        "columns": list(map(str, frame.columns)),
        "dtypes": list(map(str, frame.dtypes)),
    }))
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _module_digest(module: Any) -> str:
    path = getattr(module, "__file__", None)
    if not path:
        raise ValueError("A file-backed forecasting module is required for persistent caching")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dependencies() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "pandas", "torch", "chronos-forecasting",
                    "transformers", "safetensors", "accelerate"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


class NuclearDailyChronosCache:
    """Content-addressed cache shared across dated nuclear workspaces.

    ``execution_signature`` should contain caller-controlled numerical settings
    such as the CPU thread count. Paths and the outer run date intentionally do
    not enter the identity. A concrete local model revision is mandatory.
    """

    def __init__(
        self, cache_dir: str | Path, *, data: Any, config: Mapping[str, Any],
        context_length: int, device: str = "cpu", model_batch_size: int = 128,
        origin_batch_size: int = 31, forecasting_module: Any | None = None,
        execution_signature: Mapping[str, Any] | None = None,
    ) -> None:
        from chronos2_modular import common, data as data_module, forecasting
        from . import chronos_adapter

        if min(context_length, model_batch_size, origin_batch_size) < 1:
            raise ValueError("Chronos context and batch sizes must be positive")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Invalid Chronos cache device")
        model = dict(config.get("model", {}))
        revision = str(model.get("revision", ""))
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
            raise ValueError("Daily Chronos cache requires a pinned local model revision")
        selected_module = forecasting_module or forecasting
        for name in ("build_origin_frames", "build_live_frames", "prepare_chronos_frame"):
            if not callable(getattr(selected_module, name, None)):
                raise ValueError(f"Forecasting module is missing {name}")
        resolved_device, dtype = common.resolve_device(device)
        self.data = data
        self.context_length = int(context_length)
        self.forecasting = selected_module
        zone = str(data.zone).upper()
        if not re.fullmatch(r"[A-Z0-9_-]+", zone):
            raise ValueError("Invalid cache zone")
        self.directory = Path(cache_dir).expanduser().resolve() / zone.lower()
        self.contract = {
            "schema_version": SCHEMA_VERSION,
            "zone": zone, "timezone": str(data.timezone), "frequency": str(data.frequency),
            "model_id": str(model.get("model_id", "amazon/chronos-2")),
            "model_revision": revision,
            "attention": str(model.get("attn_implementation", "auto")),
            "seed": int(model.get("seed", 42)),
            "device": resolved_device, "dtype": str(dtype),
            "context_length": int(context_length),
            "model_batch_size": int(model_batch_size),
            "origin_batch_size": int(origin_batch_size),
            "quantiles": list(common.DEFAULT_QUANTILES),
            "cross_learning": False, "with_covariates": True,
            "execution_signature": dict(execution_signature or {}),
            "dependencies": _dependencies(),
            "implementation": {
                "forecasting": _module_digest(selected_module),
                "common": _module_digest(common), "data": _module_digest(data_module),
                "adapter": _module_digest(chronos_adapter),
                "daily_cache": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
        }
        self.audit = {"history_hits": 0, "history_misses": 0,
                      "future_hits": 0, "future_misses": 0,
                      "invalid_entries": 0, "writes": 0, "write_errors": 0}

    def identity(self, plan: ChronosDeliveryPlan) -> dict[str, Any]:
        """Hash precisely the context/future frames the real runner consumes."""
        target = self.data.target
        index = pd.DatetimeIndex(target.index)
        if index.tz is None or index.has_duplicates or not index.is_monotonic_increasing:
            raise ValueError("Chronos target must have a unique ordered timezone-aware index")
        index = index.tz_convert("UTC")
        origin = int(index.searchsorted(plan.delivery_start_utc))
        if origin < self.context_length:
            raise ValueError("Insufficient target context for daily cache")
        if origin == len(target):
            if index[-1] + pd.Timedelta(hours=1) != plan.delivery_start_utc:
                raise ValueError("Future cache plan must immediately follow the target")
            context, future = self.forecasting.build_live_frames(
                self.data, self.context_length, plan.horizon, "cached_day", True,
            )
        else:
            if not index[origin:origin + plan.horizon].equals(plan.delivery_index_utc):
                raise ValueError("Target does not exactly cover the historical cache day")
            context, future, _ = self.forecasting.build_origin_frames(
                self.data, origin, self.context_length, plan.horizon, "cached_day", True,
            )
        context = self.forecasting.prepare_chronos_frame(context, "daily_cache_context")
        future = self.forecasting.prepare_chronos_frame(future, "daily_cache_future")
        return {
            **self.contract, "delivery_day": plan.delivery_date.isoformat(),
            "forecast_origin_utc": plan.forecast_origin_utc.isoformat(),
            "delivery_index_ns": plan.delivery_index_utc.asi8.tolist(),
            "context_sha256": _frame_digest(context), "future_sha256": _frame_digest(future),
        }

    def _path(self, identity: Mapping[str, Any]) -> Path:
        return self.directory / str(identity["delivery_day"]) / (_digest(identity) + ".json")

    def _load(self, plan: ChronosDeliveryPlan, identity: dict[str, Any]) -> pd.DataFrame | None:
        path = self._path(identity)
        try:
            if not path.exists():
                return None
            if path.stat().st_size > 1_000_000:
                raise ValueError("Oversized daily cache")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["identity"] != identity:
                raise ValueError("Daily cache identity mismatch")
            prediction = payload["prediction"]
            if payload["prediction_sha256"] != _digest(prediction):
                raise ValueError("Daily cache checksum mismatch")
            if set(prediction) != {"delivery_index_ns", "quantile_dtypes", *QUANTILES}:
                raise ValueError("Unexpected daily cache payload fields")
            index = pd.to_datetime(prediction["delivery_index_ns"], utc=True)
            dtypes = prediction["quantile_dtypes"]
            if set(dtypes) != set(QUANTILES) or any(
                dtype not in {"float32", "float64", "int32", "int64"} for dtype in dtypes.values()
            ):
                raise ValueError("Invalid cached quantile dtype")
            frame = pd.DataFrame({q: np.asarray(prediction[q], dtype=dtypes[q]) for q in QUANTILES},
                                 index=index)
            return normalize_chronos_future(frame, plan)
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            self.audit["invalid_entries"] += 1
            return None

    def load(self, plan: ChronosDeliveryPlan, historical: bool = False) -> pd.DataFrame | None:
        frame = self._load(plan, self.identity(plan))
        phase = "history" if historical else "future"
        self.audit[phase + ("_hits" if frame is not None else "_misses")] += 1
        if frame is not None and historical:
            frame = self._with_actual(frame)
        return frame

    def _with_actual(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        # Match build_origin_frames' actual precision on both cold and warm runs.
        result["actual"] = self.data.target.reindex(result.index).to_numpy(dtype=np.float32)
        return normalize_chronos_oof(result)

    def store(self, plan: ChronosDeliveryPlan, frame: pd.DataFrame) -> None:
        """Validate before publishing an atomic JSON entry; never persist actuals."""
        normalized = normalize_chronos_future(frame, plan)
        identity = self.identity(plan)
        prediction = {"delivery_index_ns": normalized.index.asi8.tolist(),
                      "quantile_dtypes": {q: str(normalized[q].dtype) for q in QUANTILES},
                      **{q: normalized[q].to_numpy().tolist() for q in QUANTILES}}
        payload = {"identity": identity, "prediction": prediction,
                   "prediction_sha256": _digest(prediction)}
        path = self._path(identity)
        temporary: str | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                             prefix=".daily-", suffix=".tmp", delete=False) as handle:
                temporary = handle.name
                handle.write(_json_bytes(payload))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
            self.audit["writes"] += 1
        except OSError:
            # Disk cache failure must not invalidate an otherwise valid forecast.
            self.audit["write_errors"] += 1
        finally:
            if temporary is not None:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass

    def resolve_history(
        self, plans: Sequence[ChronosDeliveryPlan],
        execute_chunk: Callable[[Sequence[ChronosDeliveryPlan]], pd.DataFrame],
    ) -> pd.DataFrame:
        """Execute only missing days, suitable inside the existing checkpoint loop."""
        plans = tuple(plans)
        if not plans:
            raise ValueError("At least one cache plan is required")
        frames: dict[Any, pd.DataFrame] = {}
        missing = []
        for plan in plans:
            cached = self.load(plan, historical=True)
            if cached is None:
                missing.append(plan)
            else:
                frames[plan.delivery_date] = cached
        # The existing OOF normalizer deliberately requires contiguous hours.
        # Revisions can invalidate isolated days, so never give it sparse plans.
        missing_groups: list[list[ChronosDeliveryPlan]] = []
        for plan in missing:
            if (not missing_groups or
                    missing_groups[-1][-1].delivery_index_utc[-1] + pd.Timedelta(hours=1)
                    != plan.delivery_start_utc):
                missing_groups.append([])
            missing_groups[-1].append(plan)
        for group in missing_groups:
            generated = execute_chunk(tuple(group))
            if not isinstance(generated.index, pd.DatetimeIndex):
                generated = normalize_chronos_oof(generated)
            expected = group[0].delivery_index_utc
            for plan in group[1:]:
                expected = expected.append(plan.delivery_index_utc)
            if not generated.index.equals(expected):
                raise ValueError("Generated cache misses do not exactly match requested days")
            for plan in group:
                frame = normalize_chronos_future(generated.loc[plan.delivery_index_utc], plan)
                self.store(plan, frame)
                frames[plan.delivery_date] = self._with_actual(frame)
        return normalize_chronos_oof(pd.concat([frames[plan.delivery_date] for plan in plans]))

    def resolve_future(
        self, plan: ChronosDeliveryPlan, execute_future: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        frame = self.load(plan, historical=False)
        if frame is None:
            frame = normalize_chronos_future(execute_future(), plan)
            self.store(plan, frame)
        return frame


__all__ = ["NuclearDailyChronosCache"]
