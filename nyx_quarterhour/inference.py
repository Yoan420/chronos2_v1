"""Strict, local-only CPU Chronos-2 inference for isolated resolution research.

No acquisition, training, interpolation, label scoring or operational cache access
occurs here. Naive model timestamps represent UTC; aware inputs are normalized to
naive UTC. The caller owns target provenance and historical availability checks.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_PIPELINE: Any = None
_MODEL_ID: str | None = None
_CHECKPOINT_IDENTITY: dict[str, Any] | None = None
_CHECKPOINT_SIGNATURES: dict[str, tuple[int, int]] = {}


def _file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(local_model_id: str | None = None) -> dict[str, Any]:
    """Seal local config/weight hashes without loading a model, or verify that seal.

    This can be called before the experiment protocol is frozen. Every public
    call hashes the actual file bytes again; runtime-only thread/seed settings
    are deliberately absent so the identity is stable before/after loading.
    """
    global _CHECKPOINT_IDENTITY, _CHECKPOINT_SIGNATURES, _MODEL_ID
    requested = local_model_id or _MODEL_ID or "amazon/chronos-2"
    checkpoint = _local_checkpoint(requested)
    if _CHECKPOINT_IDENTITY is not None and (requested != _MODEL_ID or checkpoint != _CHECKPOINT_IDENTITY["checkpoint_path"]):
        raise RuntimeError("The cached model path changed; use an isolated fresh process.")
    files = {name: Path(checkpoint)/name for name in ("config.json", "model.safetensors")}
    signatures = {str(path): _file_signature(path) for path in files.values()}
    identity = {"requested_model_id": requested, "checkpoint_path": checkpoint,
                "snapshot": Path(checkpoint).name,
                "config_sha256": _sha256(files["config.json"]),
                "weights_sha256": _sha256(files["model.safetensors"]),
                "weights_bytes": signatures[str(files["model.safetensors"])][0],
                "device": "cpu", "dtype": "float32", "local_files_only": True}
    if signatures != {filename: _file_signature(Path(filename)) for filename in signatures}:
        raise RuntimeError("Checkpoint files changed while hashing; refusing this identity.")
    if _CHECKPOINT_IDENTITY is not None and identity != _CHECKPOINT_IDENTITY:
        raise RuntimeError("Loaded checkpoint files changed; use an isolated fresh process.")
    _CHECKPOINT_IDENTITY, _CHECKPOINT_SIGNATURES, _MODEL_ID = identity, signatures, requested
    return dict(identity)


def _check_loaded_file_signatures() -> None:
    """Cheap per-batch guard; the runner separately rehashes at the end."""
    for filename, signature in _CHECKPOINT_SIGNATURES.items():
        if _file_signature(Path(filename)) != signature:
            raise RuntimeError("Loaded checkpoint files changed; use an isolated fresh process.")


def _local_checkpoint(model_id: str) -> str:
    """Resolve a cache entry without the library's eager remote adapter probe."""
    directory = Path(model_id).expanduser()
    if not directory.is_dir():
        from huggingface_hub import try_to_load_from_cache

        config = try_to_load_from_cache(model_id, "config.json", revision="main")
        if not isinstance(config, str):
            raise FileNotFoundError("Checkpoint is not cached locally; downloads are disabled.")
        # Preserve the snapshot directory, even if config.json itself is a link
        # into the shared blob store.
        directory = Path(config).parent
    directory = directory.resolve()
    if not (directory/"config.json").is_file() or not (directory/"model.safetensors").is_file():
        raise FileNotFoundError("A complete local Chronos-2 config and model.safetensors are required.")
    return str(directory)


def load_pipeline(
    local_model_id: str = "amazon/chronos-2", *, torch_threads: int = 8, seed: int = 42,
) -> Any:
    """Load cached weights once, on CPU float32, without downloading anything."""
    global _PIPELINE, _MODEL_ID, _CHECKPOINT_IDENTITY, _CHECKPOINT_SIGNATURES
    if not isinstance(local_model_id, str) or not local_model_id.strip():
        raise ValueError("A nonempty local model path or cached model id is required.")
    if type(torch_threads) is not int or torch_threads not in {2, 4, 8}:
        raise ValueError("The research CPU protocol permits two, four or eight Torch threads.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if _PIPELINE is not None and _MODEL_ID != local_model_id:
        raise RuntimeError("A different model is already loaded; use an isolated process.")
    identity = checkpoint_identity(local_model_id)
    checkpoint = identity["checkpoint_path"]
    import torch

    torch.set_num_threads(torch_threads)
    torch.manual_seed(seed)
    if _PIPELINE is None:
        from chronos import Chronos2Pipeline

        signatures = dict(_CHECKPOINT_SIGNATURES)
        kwargs = {"device_map": "cpu", "local_files_only": True}
        try:
            pipeline = Chronos2Pipeline.from_pretrained(checkpoint, dtype=torch.float32, **kwargs)
        except TypeError as exc:
            # Compatibility with older Transformers spelling only; other loader
            # failures must propagate and never trigger an online retry.
            if "dtype" not in str(exc):
                raise
            pipeline = Chronos2Pipeline.from_pretrained(checkpoint, torch_dtype=torch.float32, **kwargs)
        if signatures != {filename: _file_signature(Path(filename)) for filename in signatures}:
            raise RuntimeError("Checkpoint files changed while loading; refusing this model.")
        _PIPELINE, _MODEL_ID = pipeline, local_model_id
        _CHECKPOINT_IDENTITY, _CHECKPOINT_SIGNATURES = identity, signatures
    return _PIPELINE


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _timestamps(values: pd.Series, label: str) -> pd.DatetimeIndex:
    try:
        result = pd.DatetimeIndex(values)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label}: invalid timestamps.") from exc
    if result.hasnans:
        raise ValueError(f"{label}: missing timestamps.")
    if result.tz is not None:
        result = result.tz_convert("UTC").tz_localize(None)
    return result


def _frame(frame: pd.DataFrame, label: str, *, target: bool, step: pd.Timedelta) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{label}: a nonempty DataFrame is required.")
    if frame.columns.has_duplicates or not all(isinstance(c, str) for c in frame):
        raise ValueError(f"{label}: unique string column names required.")
    required = {"item_id", "timestamp", "target"} if target else {"item_id", "timestamp"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{label}: missing required columns {sorted(required-set(frame.columns))}.")
    if not target and "target" in frame:
        raise ValueError(f"{label}: future targets are forbidden.")
    if frame.item_id.nunique(dropna=False) != 1 or not frame.item_id.map(lambda x: isinstance(x, str) and bool(x)).all():
        raise ValueError(f"{label}: exactly one nonempty string item_id is required.")
    result = frame.copy(deep=True)
    times = _timestamps(result.timestamp, label)
    if not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError(f"{label}: timestamps must be unique and increasing.")
    if (times.asi8 % step.value != 0).any() or (len(times)>1 and not ((times[1:]-times[:-1])==step).all()):
        raise ValueError(f"{label}: timestamps must form an aligned, gap-free {step} grid.")
    result["timestamp"] = times
    numeric = [c for c in result if c not in {"item_id", "timestamp"}]
    for column in numeric:
        try:
            values = pd.to_numeric(result[column], errors="raise")
            if np.iscomplexobj(values):
                raise ValueError("Complex values are invalid.")
            result[column] = values.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}: {column} must be numeric.") from exc
    if not np.isfinite(result[numeric].to_numpy(float)).all():
        raise ValueError(f"{label}: incomplete/nonfinite inputs are not filled by this executor.")
    return result


def infer_batch(
    contexts: list[pd.DataFrame], futures: list[pd.DataFrame], *, freq: str,
    context_length: int, prediction_length: int, model_batch_size: int = 32,
    pipeline: Any = None,
) -> pd.DataFrame:
    """Predict contiguous physical horizons; return UTC item keys and q10/q50/q90.

    ``pipeline`` optionally injects a previously loaded instance or test double.
    Otherwise ``load_pipeline`` must have been called in this process. Contexts
    and futures contain one item each and the same known covariate schema. The
    future begins exactly one step after the context: a missing day cannot be
    skipped. Contexts may be shorter than the cap, but are never silently clipped.
    Cross-learning between origins/countries is disabled. ``point``, when present,
    is the library's point column (Chronos 2.3.1 returns its median there).
    Averaging quarter medians does not establish an hourly median or intervals.
    """
    context_length = _positive_integer(context_length, "context_length")
    prediction_length = _positive_integer(prediction_length, "prediction_length")
    model_batch_size = _positive_integer(model_batch_size, "model_batch_size")
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
        step = pd.Timedelta(offset.nanos)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("freq must be physical hourly or quarter-hourly.") from exc
    if step not in {pd.Timedelta(hours=1), pd.Timedelta(minutes=15)}:
        raise ValueError("Only hourly or native quarter-hourly execution is supported.")
    canonical_freq = "h" if step == pd.Timedelta(hours=1) else "15min"
    if not isinstance(contexts, list) or not isinstance(futures, list) or not contexts or len(contexts)!=len(futures):
        raise ValueError("Nonempty, equally sized context and future lists are required.")
    prepared_contexts, prepared_futures = [], []
    ids: set[str] = set()
    schema: set[str] | None = None
    for number, (context, future) in enumerate(zip(contexts, futures)):
        c = _frame(context, f"context[{number}]", target=True, step=step)
        f = _frame(future, f"future[{number}]", target=False, step=step)
        item = c.item_id.iloc[0]
        if item in ids or f.item_id.iloc[0] != item:
            raise ValueError("Context/future item IDs must match and be unique across tasks.")
        ids.add(item)
        if len(c)>context_length:
            raise ValueError("Context exceeds context_length; select the intended history explicitly.")
        if len(f)!=prediction_length:
            raise ValueError("Every future must contain exactly prediction_length steps.")
        if f.timestamp.iloc[0] != c.timestamp.iloc[-1]+step:
            raise ValueError("Future must start immediately after context; do not skip an unknown day.")
        covariates = set(c.columns)-{"item_id", "timestamp", "target"}
        if set(f.columns)-{"item_id", "timestamp"} != covariates:
            raise ValueError("Context and future must have the same declared known covariates.")
        if schema is not None and schema != covariates:
            raise ValueError("All batched items must share one covariate schema.")
        schema = covariates
        prepared_contexts.append(c)
        prepared_futures.append(f)
    active = pipeline if pipeline is not None else _PIPELINE
    if active is None:
        raise RuntimeError("Call load_pipeline before infer_batch; implicit model loading is disabled.")
    if active is _PIPELINE:
        _check_loaded_file_signatures()
    maximum_context = int(getattr(active, "model_context_length", 8192))
    maximum_horizon = int(getattr(active, "model_prediction_length", 1024))
    if context_length>maximum_context or prediction_length>maximum_horizon:
        raise ValueError("Requested context/horizon exceeds this checkpoint's native limits.")
    n_variates = 1+len(schema or ())
    context_df = pd.concat(prepared_contexts, ignore_index=True).sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    future_df = pd.concat(prepared_futures, ignore_index=True).sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    output = active.predict_df(
        context_df, future_df=future_df, id_column="item_id", timestamp_column="timestamp",
        target="target", prediction_length=prediction_length, quantile_levels=[.1, .5, .9],
        batch_size=max(model_batch_size, n_variates), context_length=context_length,
        cross_learning=False, validate_inputs=True, freq=canonical_freq,
    )
    if not isinstance(output, pd.DataFrame) or output.columns.has_duplicates:
        raise ValueError("Chronos returned an invalid forecast table.")
    required = {"item_id", "timestamp", "0.1", "0.5", "0.9"}
    if not required.issubset(output):
        raise ValueError("Chronos output is missing item keys or required quantiles.")
    if "target_name" in output and not output.target_name.eq("target").all():
        raise ValueError("Chronos output includes an unexpected target.")
    output = output.copy()
    output["timestamp"] = _timestamps(output.timestamp, "prediction")
    if output.duplicated(["item_id", "timestamp"]).any():
        raise ValueError("Chronos returned duplicate forecast identities.")
    expected = future_df[["item_id", "timestamp"]]
    observed_keys = pd.MultiIndex.from_frame(output[["item_id", "timestamp"]])
    expected_keys = pd.MultiIndex.from_frame(expected)
    if len(output)!=len(expected) or len(observed_keys.difference(expected_keys)) or len(expected_keys.difference(observed_keys)):
        raise ValueError("Chronos output does not exactly cover the requested physical horizon.")
    names = {"0.1": "q10", "0.5": "q50", "0.9": "q90", "predictions": "point"}
    columns = ["item_id", "timestamp", "0.1", "0.5", "0.9"]
    if "predictions" in output:
        columns.append("predictions")
    result = expected.merge(output[columns], on=["item_id", "timestamp"], validate="one_to_one").rename(columns=names)
    values = result[[c for c in result if c not in {"item_id", "timestamp"}]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Chronos returned nonfinite forecasts.")
    if not ((result.q10<=result.q50)&(result.q50<=result.q90)).all():
        raise ValueError("Chronos returned crossed quantiles; no repair is applied.")
    return result
