"""Disposable, content-addressed daily residual predictions.

Only the inputs to a day's causal fit and prediction participate in its key.
In particular, observing that day's target tomorrow cannot invalidate the
prediction made today. Values are hashed once per local delivery-day block,
so a rolling year combines small digests instead of rehashing its data.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from hashlib import sha256
from importlib import metadata
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
QUANTILES = ("q10", "q50", "q90")


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported residual cache contract value: {type(value).__name__}")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False, default=_json_default,
    ).encode("utf-8")


def _runtime_contract() -> dict[str, Any]:
    dependencies: dict[str, str] = {}
    for name in ("pandas", "numpy", "catboost", "scikit-learn", "pyarrow"):
        try:
            dependencies[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            dependencies[name] = "unavailable"
    package = Path(__file__).resolve().parent
    code = {}
    for relative in (
        "nuclear_residual_cache.py", "models/residual_corrector.py",
        "models/blended_residual_corrector.py", "models/catboost_hourly.py", "features.py",
    ):
        path = package / relative
        code[relative] = sha256(path.read_bytes()).hexdigest() if path.is_file() else "absent"
    return {"dependencies": dependencies, "code": code}


def _require_index(index: pd.Index) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("Residual cache requires timezone-aware DatetimeIndex")
    if not index.is_monotonic_increasing or not index.is_unique or index.hasnans:
        raise ValueError("Residual cache indices must be sorted, unique and finite")
    return index


class _DailyFrameFingerprint:
    """Prehash each complete day; support exact contiguous or sparse selections."""

    def __init__(self, frame: pd.DataFrame, timezone: str):
        self.index = _require_index(frame.index).copy()
        self.schema = sha256(_json_bytes({
            "columns": [str(column) for column in frame.columns],
            "column_types": [type(column).__name__ for column in frame.columns],
            "dtypes": [str(dtype) for dtype in frame.dtypes],
            "index_dtype": str(frame.index.dtype),
        })).digest()
        # The row hashes contain the timestamp as well as every input value.
        self.rows = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="uint64")
        local_days = np.asarray(self.index.tz_convert(timezone).date)
        self.starts = [0] if len(frame) else []
        self.starts.extend((np.flatnonzero(local_days[1:] != local_days[:-1]) + 1).tolist())
        self.ends = self.starts[1:] + ([len(frame)] if len(frame) else [])
        self.blocks = [self._rows_digest(start, end) for start, end in zip(self.starts, self.ends)]

    def _rows_digest(self, start: int, end: int) -> bytes:
        return sha256(self.rows[start:end].astype("<u8", copy=False).tobytes()).digest()

    def digest(self, selected: pd.DatetimeIndex) -> str:
        selected = _require_index(selected)
        digest = sha256(self.schema)
        digest.update(len(selected).to_bytes(8, "big"))
        if not len(selected):
            return digest.hexdigest()
        begin = int(self.index.searchsorted(selected[0]))
        end = begin + len(selected)
        if not self.index[begin:end].equals(selected):
            positions = self.index.get_indexer(selected)
            if (positions < 0).any():
                raise ValueError("Residual cache input does not cover selected index")
            # Explicit marker separates sparse selections from daily-block encoding.
            digest.update(b"sparse")
            digest.update(self.rows[positions].astype("<u8", copy=False).tobytes())
            return digest.hexdigest()
        digest.update(b"daily")
        block = bisect_right(self.starts, begin) - 1
        while begin < end:
            stop = min(end, self.ends[block])
            if begin == self.starts[block] and stop == self.ends[block]:
                digest.update(self.blocks[block])
            else:
                digest.update(self._rows_digest(begin, stop))
            begin = stop
            block += 1
        return digest.hexdigest()


def _valid_predictions(frame: pd.DataFrame, expected: pd.DatetimeIndex) -> bool:
    if not isinstance(frame, pd.DataFrame) or not frame.index.equals(expected):
        return False
    if tuple(frame.columns) != QUANTILES:
        return False
    try:
        values = frame.to_numpy(dtype=float)
        return bool(np.isfinite(values).all() and (values[:, 0] <= values[:, 1]).all()
                    and (values[:, 1] <= values[:, 2]).all())
    except (TypeError, ValueError):
        return False


class ResidualDayCache:
    """Cache corrected daily quantiles, never fitted Python model objects.

    ``contract`` must describe every fit/predict option and any custom factory.
    Input frames are fingerprinted at construction and must represent the
    immutable input snapshot for one replay. A cache failure is always a miss;
    predictions can still be computed normally when its directory is unwritable.
    """

    def __init__(
        self, cache_dir: str | Path, contract: Mapping[str, Any], *,
        features: pd.DataFrame, raw: pd.DataFrame, timezone: str,
    ):
        self.cache_dir = Path(cache_dir)
        self.timezone = timezone
        self.contract_digest = sha256(_json_bytes({
            "schema": SCHEMA_VERSION, "timezone": timezone,
            "recipe": dict(contract), "runtime": _runtime_contract(),
        })).hexdigest()
        self.features = _DailyFrameFingerprint(features, timezone)
        self.training_raw = _DailyFrameFingerprint(raw.loc[:, [*QUANTILES, "actual"]], timezone)
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def key(
        self, day: str | date, training_index: pd.DatetimeIndex,
        predicted_index: pd.DatetimeIndex, prediction_base: pd.DataFrame,
    ) -> str:
        if not prediction_base.index.equals(predicted_index):
            raise ValueError("Residual cache prediction base index differs from requested day")
        base = _DailyFrameFingerprint(prediction_base.loc[:, list(QUANTILES)], self.timezone)
        return sha256(_json_bytes({
            "contract": self.contract_digest, "day": pd.Timestamp(day).date().isoformat(),
            "training_features": self.features.digest(training_index),
            "prediction_features": self.features.digest(predicted_index),
            "training_raw": self.training_raw.digest(training_index),
            "prediction_base": base.digest(predicted_index),
        })).hexdigest()

    def load(
        self, day: str | date, training_index: pd.DatetimeIndex,
        predicted_index: pd.DatetimeIndex, prediction_base: pd.DataFrame,
    ) -> tuple[pd.DataFrame, tuple[str, ...]] | None:
        try:
            key = self.key(day, training_index, predicted_index, prediction_base)
            record = json.loads((self.cache_dir / f"{key}.json").read_bytes())
            record_checksum = record.pop("record_sha256")
            if sha256(_json_bytes(record)).hexdigest() != record_checksum:
                raise ValueError("Residual cache metadata checksum mismatch")
            if record["schema"] != SCHEMA_VERSION or record["key"] != key:
                raise ValueError("Residual cache metadata mismatch")
            checksum = record["payload_sha256"]
            if not isinstance(checksum, str) or len(checksum) != 64 or any(
                character not in "0123456789abcdef" for character in checksum
            ):
                raise ValueError("Invalid residual cache payload checksum")
            payload = (self.cache_dir / f"{checksum}.parquet").read_bytes()
            if sha256(payload).hexdigest() != checksum:
                raise ValueError("Residual cache payload checksum mismatch")
            columns = record["feature_columns"]
            if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns):
                raise ValueError("Invalid residual cache feature columns")
            predicted = pd.read_parquet(BytesIO(payload))
            if not _valid_predictions(predicted, predicted_index):
                raise ValueError("Invalid residual cache quantiles or index")
            self.hits += 1
            return predicted, tuple(columns)
        except Exception:
            # All cache files are disposable. Rebuild after truncation, an old
            # schema, unavailable Parquet support, or a concurrent cleanup.
            self.misses += 1
            return None

    def store(
        self, day: str | date, training_index: pd.DatetimeIndex,
        predicted_index: pd.DatetimeIndex, prediction_base: pd.DataFrame,
        predicted: pd.DataFrame, feature_columns: Sequence[str],
    ) -> bool:
        try:
            if not _valid_predictions(predicted, predicted_index):
                return False
            if not all(isinstance(column, str) for column in feature_columns):
                return False
            key = self.key(day, training_index, predicted_index, prediction_base)
            buffer = BytesIO()
            persisted = predicted.copy(deep=False)
            # ResidualCorrector attaches Series-valued diagnostic attrs. They
            # are not model inputs or cached outputs and cannot be JSON encoded
            # by Parquet's pandas metadata serializer.
            persisted.attrs = {}
            persisted.to_parquet(buffer, index=True)
            payload = buffer.getvalue()
            checksum = sha256(payload).hexdigest()
            record = {
                "schema": SCHEMA_VERSION, "key": key, "payload_sha256": checksum,
                "feature_columns": list(feature_columns),
            }
            record["record_sha256"] = sha256(_json_bytes(record)).hexdigest()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Metadata is the commit point. The content-addressed payload name
            # allows concurrent writers of the same key without mixed pairs.
            self._atomic_write(self.cache_dir / f"{checksum}.parquet", payload)
            self._atomic_write(self.cache_dir / f"{key}.json", _json_bytes(record))
            self.writes += 1
            return True
        except Exception:
            return False

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        # Keep temporary names short enough for Windows' usual path limit.
        temporary = path.with_name(f".{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
