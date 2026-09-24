"""Chronos inference retaining NYX's distinct past-only and known-future inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nyx_quarterhour.inference import _frame, _positive_integer, _timestamps, _check_loaded_file_signatures


def infer_batch(contexts, futures, *, freq, context_length, prediction_length, model_batch_size=64, pipeline):
    context_length = _positive_integer(context_length, "context_length")
    prediction_length = _positive_integer(prediction_length, "prediction_length")
    model_batch_size = _positive_integer(model_batch_size, "model_batch_size")
    if freq not in {"h", "15min"}:
        raise ValueError("Explicit hourly or quarter-hour cadence required.")
    step = pd.Timedelta(hours=1) if freq == "h" else pd.Timedelta(minutes=15)
    if not isinstance(contexts, list) or not isinstance(futures, list) or not contexts or len(contexts) != len(futures):
        raise ValueError("Paired nonempty context/future lists required.")
    cs, fs, ids, schemas = [], [], set(), None
    for number, (context, future) in enumerate(zip(contexts, futures)):
        c = _frame(context, f"context[{number}]", target=True, step=step)
        f = _frame(future, f"future[{number}]", target=False, step=step)
        item = c.item_id.iloc[0]
        if item in ids or item != f.item_id.iloc[0]:
            raise ValueError("Unique matching item identities required.")
        ids.add(item)
        if len(c) != context_length or len(f) != prediction_length:
            raise ValueError("Exact physical context and horizon required; no truncation.")
        if f.timestamp.iloc[0] != c.timestamp.iloc[-1] + step:
            raise ValueError("Future must immediately follow history.")
        history_columns = set(c) - {"item_id", "timestamp", "target"}
        future_columns = set(f) - {"item_id", "timestamp"}
        if not future_columns.issubset(history_columns):
            raise ValueError("Known future variables must also have context history.")
        # The six raw forecast aliases are historical inputs; only explicitly
        # known_* variables can be supplied for the delivery day.
        if not all(name.startswith("known_") for name in future_columns):
            raise ValueError("Future variables require explicit known_ declarations.")
        current = history_columns, future_columns
        if schemas is not None and current != schemas:
            raise ValueError("Every task must share one past/future input contract.")
        schemas = current
        cs.append(c)
        fs.append(f)
    if context_length > int(getattr(pipeline, "model_context_length", 8192)) or prediction_length > int(getattr(pipeline, "model_prediction_length", 1024)):
        raise ValueError("Requested context/horizon exceeds checkpoint limits.")
    _check_loaded_file_signatures()
    context_df = pd.concat(cs, ignore_index=True).sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    future_df = pd.concat(fs, ignore_index=True).sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    output = pipeline.predict_df(context_df, future_df=future_df, id_column="item_id", timestamp_column="timestamp",
                                 target="target", prediction_length=prediction_length, quantile_levels=[.1, .5, .9],
                                 batch_size=max(model_batch_size, 1 + len(schemas[0])), context_length=context_length,
                                 cross_learning=False, validate_inputs=True, freq=freq)
    if not isinstance(output, pd.DataFrame) or output.columns.has_duplicates:
        raise ValueError("Invalid Chronos output.")
    if not {"item_id", "timestamp", "0.1", "0.5", "0.9"}.issubset(output):
        raise ValueError("Chronos quantile output incomplete.")
    output = output.copy()
    if "target_name" in output and not output.target_name.eq("target").all():
        raise ValueError("Unexpected target in model output.")
    output["timestamp"] = _timestamps(output.timestamp, "prediction")
    if output.duplicated(["item_id", "timestamp"]).any():
        raise ValueError("Duplicate model prediction.")
    expected = future_df[["item_id", "timestamp"]]
    observed_keys = pd.MultiIndex.from_frame(output[["item_id", "timestamp"]])
    expected_keys = pd.MultiIndex.from_frame(expected)
    if len(output) != len(expected) or len(observed_keys.difference(expected_keys)) or len(expected_keys.difference(observed_keys)):
        raise ValueError("Model output does not match the requested delivery horizon.")
    result = expected.merge(output[["item_id", "timestamp", "0.1", "0.5", "0.9"]], on=["item_id", "timestamp"], validate="one_to_one")
    result.rename(columns={"0.1": "q10", "0.5": "q50", "0.9": "q90"}, inplace=True)
    q = result[["q10", "q50", "q90"]].to_numpy(float)
    if not np.isfinite(q).all() or (q[:, 0] > q[:, 1]).any() or (q[:, 1] > q[:, 2]).any():
        raise ValueError("Invalid or crossed Chronos quantiles.")
    return result
