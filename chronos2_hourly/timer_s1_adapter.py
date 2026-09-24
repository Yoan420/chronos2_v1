"""Strict, DST-safe Timer-S1 adapter for the Chronos delivery contract.

Timer-S1's released ``generate`` interface is *univariate*: each batch row is
one history of the target being forecast.  It has no native API for Chronos-2
past/future covariates.  Consequently this adapter consumes only
``data.target`` (or a target :class:`pandas.Series`) and deliberately ignores
any covariate tables carried by ``data``.

Feature columns must never be smuggled into Timer-S1 as additional batch rows.
That would make the model treat features as independent time series and would
not be equivalent to Chronos-2 multivariate conditioning.  Here, batching is
used only for multiple forecast origins that share the same 23/24/25-hour
horizon and the exact same ``context_length``.

Heavy dependencies are imported lazily.  Tests and offline callers can inject
either a :class:`TimerS1Runtime` or a model plus a torch-compatible module,
without importing Transformers or downloading the checkpoint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .chronos_adapter import (
    ChronosAdapterError,
    ChronosDeliveryPlan,
    ChronosGroupExecutor,
    ChronosLiveExecutor,
    execute_chronos_live_forecast,
    execute_grouped_chronos_backtest,
)


DEFAULT_TIMER_S1_MODEL_ID = "bytedance-research/Timer-S1"
DEFAULT_TIMER_S1_REVISION = (
    "8911430cc7f32add5c8913afe12e3b05742f5bb2"
)
TIMER_S1_QUANTILE_LEVELS = tuple(value / 10 for value in range(1, 10))
TIMER_S1_QUANTILE_INDICES = {"q10": 0, "q50": 4, "q90": 8}
TIMER_S1_MAX_CONTEXT_LENGTH = 11_520
_VALID_HORIZONS = frozenset({23, 24, 25})


class TimerS1AdapterError(ChronosAdapterError):
    """Raised when Timer-S1 input or output violates the strict contract."""


@dataclass(frozen=True)
class TimerS1Runtime:
    """Loaded Timer-S1 model and its lazily imported torch module."""

    model: Any
    torch_module: Any
    model_id: str = DEFAULT_TIMER_S1_MODEL_ID
    revision: str = DEFAULT_TIMER_S1_REVISION


class TimerS1Model(Protocol):
    """Minimum interface used from the remote-code Timer-S1 model."""

    device: Any

    def eval(self) -> Any:
        """Switch to deterministic evaluation mode."""

    def generate(
        self,
        sequences: Any,
        *,
        max_new_tokens: int,
        revin: bool,
    ) -> Any:
        """Return the official ``[B, 9, H]`` quantile tensor."""


def _positive_integer(
    value: int,
    *,
    name: str,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool):
        raise TimerS1AdapterError(f"{name} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TimerS1AdapterError(
            f"{name} must be a positive integer."
        ) from exc
    if parsed != value or parsed <= 0:
        raise TimerS1AdapterError(f"{name} must be a positive integer.")
    if maximum is not None and parsed > maximum:
        raise TimerS1AdapterError(
            f"{name} must not exceed Timer-S1's {maximum}-point context."
        )
    return parsed


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Timer-S1 requires torch; install the checkpoint's documented "
            "runtime dependencies."
        ) from exc
    return torch


def load_timer_s1_runtime(
    *,
    model_id: str = DEFAULT_TIMER_S1_MODEL_ID,
    revision: str = DEFAULT_TIMER_S1_REVISION,
    device_map: Any = "auto",
    local_files_only: bool = False,
    torch_dtype: Any | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    torch_module: Any | None = None,
    auto_model_class: Any | None = None,
) -> TimerS1Runtime:
    """Load the pinned Timer-S1 checkpoint without eager heavy imports.

    ``auto_model_class`` and ``torch_module`` are injection seams for tests.
    Production loading follows the model card and always enables the remote
    model implementation with ``trust_remote_code=True``.
    """

    if not str(model_id).strip():
        raise TimerS1AdapterError("model_id must not be empty.")
    if not str(revision).strip():
        raise TimerS1AdapterError("revision must not be empty.")
    if torch_module is None:
        torch_module = _load_torch()
    if auto_model_class is None:
        try:
            import transformers
            from packaging.version import Version
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Timer-S1 requires transformers~=4.57.1."
            ) from exc
        installed = Version(transformers.__version__)
        if not (Version("4.57.1") <= installed < Version("4.58")):
            raise ImportError(
                "Timer-S1 requires transformers>=4.57.1,<4.58; "
                f"installed version is {installed}."
            )
        auto_model_class = transformers.AutoModelForCausalLM

    kwargs: dict[str, Any] = dict(model_kwargs or {})
    protected = {
        "trust_remote_code",
        "revision",
        "device_map",
        "local_files_only",
    }
    overlap = sorted(protected.intersection(kwargs))
    if overlap:
        raise TimerS1AdapterError(
            "model_kwargs must not override pinned loading arguments: "
            + ", ".join(overlap)
            + "."
        )
    kwargs.update(
        {
            "trust_remote_code": True,
            "revision": str(revision),
            "device_map": device_map,
            "local_files_only": bool(local_files_only),
        }
    )
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    model = auto_model_class.from_pretrained(str(model_id), **kwargs)
    if not callable(getattr(model, "eval", None)):
        raise TimerS1AdapterError("Loaded Timer-S1 model has no eval() method.")
    if not callable(getattr(model, "generate", None)):
        raise TimerS1AdapterError(
            "Loaded Timer-S1 model has no generate() method."
        )
    model.eval()
    return TimerS1Runtime(
        model=model,
        torch_module=torch_module,
        model_id=str(model_id),
        revision=str(revision),
    )


def _extract_target(data: Any) -> pd.Series:
    target = data if isinstance(data, pd.Series) else getattr(data, "target", None)
    if not isinstance(target, pd.Series) or target.empty:
        raise TimerS1AdapterError(
            "data.target must be a non-empty pandas Series."
        )
    if not isinstance(target.index, pd.DatetimeIndex):
        raise TimerS1AdapterError("data.target must use a DatetimeIndex.")
    if target.index.tz is None:
        raise TimerS1AdapterError(
            "data.target timestamps must be timezone-aware."
        )

    index = pd.DatetimeIndex(target.index).tz_convert("UTC")
    index.name = "delivery_start_utc"
    if index.has_duplicates:
        raise TimerS1AdapterError("data.target contains duplicate timestamps.")
    if not index.is_monotonic_increasing:
        raise TimerS1AdapterError("data.target timestamps must be sorted.")
    aligned = (
        (index.minute == 0)
        & (index.second == 0)
        & (index.microsecond == 0)
        & (index.nanosecond == 0)
    )
    if not bool(np.all(aligned)):
        raise TimerS1AdapterError(
            "data.target must be aligned to complete UTC hours."
        )
    if len(index) > 1 and not bool(
        np.all(index[1:] - index[:-1] == pd.Timedelta(hours=1))
    ):
        raise TimerS1AdapterError(
            "data.target must be continuous at an hourly UTC frequency."
        )

    values = pd.to_numeric(target, errors="coerce").to_numpy(dtype=np.float64)
    if not bool(np.isfinite(values).all()):
        raise TimerS1AdapterError(
            "data.target contains missing or non-finite values."
        )
    return pd.Series(values, index=index, name=target.name or "target")


def _validate_plan(plan: ChronosDeliveryPlan, *, horizon: int) -> None:
    if not isinstance(plan, ChronosDeliveryPlan):
        raise TimerS1AdapterError(
            "Timer-S1 executors require ChronosDeliveryPlan instances."
        )
    if horizon not in _VALID_HORIZONS or plan.horizon != horizon:
        raise TimerS1AdapterError(
            "Timer-S1 horizon must exactly match a 23/24/25-hour plan."
        )
    delivery = pd.DatetimeIndex(plan.delivery_index_utc)
    if delivery.tz is None or str(delivery.tz) != "UTC":
        raise TimerS1AdapterError("Plan deliveries must use UTC.")
    if len(delivery) > 1 and not bool(
        np.all(delivery[1:] - delivery[:-1] == pd.Timedelta(hours=1))
    ):
        raise TimerS1AdapterError("Plan deliveries must be continuous in UTC.")
    origin = pd.Timestamp(plan.forecast_origin_utc)
    if origin.tzinfo is None or str(origin.tz_convert("UTC").tz) != "UTC":
        raise TimerS1AdapterError("Plan forecast origin must use UTC.")
    if not origin.tz_convert("UTC") < plan.delivery_start_utc:
        raise TimerS1AdapterError(
            "Plan forecast origin must precede its first delivery."
        )


def _backtest_context_and_actual(
    target: pd.Series,
    plan: ChronosDeliveryPlan,
    *,
    context_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    start = plan.delivery_start_utc
    try:
        start_position = int(target.index.get_loc(start))
    except KeyError as exc:
        raise TimerS1AdapterError(
            f"Target does not cover delivery start {start}."
        ) from exc
    if start_position < context_length:
        raise TimerS1AdapterError(
            f"Target has fewer than {context_length} context hours before "
            f"delivery day {plan.delivery_date}."
        )

    context = target.iloc[start_position - context_length : start_position]
    expected_context = pd.date_range(
        end=start - pd.Timedelta(hours=1),
        periods=context_length,
        freq="h",
        tz="UTC",
        name="delivery_start_utc",
    )
    if not context.index.equals(expected_context):
        raise TimerS1AdapterError(
            "Every Timer-S1 target context must end exactly one UTC hour "
            "before its first delivery and contain context_length hours."
        )

    delivery_locations = target.index.get_indexer(plan.delivery_index_utc)
    if bool((delivery_locations < 0).any()):
        raise TimerS1AdapterError(
            f"Target does not cover every hour of {plan.delivery_date}."
        )
    expected_locations = np.arange(start_position, start_position + plan.horizon)
    if not np.array_equal(delivery_locations, expected_locations):
        raise TimerS1AdapterError(
            f"Target coverage is not continuous through {plan.delivery_date}."
        )
    actual = target.iloc[
        start_position : start_position + plan.horizon
    ].to_numpy(dtype=np.float64)
    if len(actual) != plan.horizon or not bool(np.isfinite(actual).all()):
        raise TimerS1AdapterError(
            f"Target actuals are incomplete or non-finite for {plan.delivery_date}."
        )
    return context.to_numpy(dtype=np.float32), actual


def _live_context(
    target: pd.Series,
    plan: ChronosDeliveryPlan,
    *,
    context_length: int,
) -> np.ndarray:
    expected_end = plan.delivery_start_utc - pd.Timedelta(hours=1)
    if target.index[-1] != expected_end:
        raise TimerS1AdapterError(
            "Live data.target must end exactly one UTC hour before the plan."
        )
    if len(target) < context_length:
        raise TimerS1AdapterError(
            f"Live target has fewer than {context_length} context hours."
        )
    context = target.iloc[-context_length:]
    expected = pd.date_range(
        end=expected_end,
        periods=context_length,
        freq="h",
        tz="UTC",
        name="delivery_start_utc",
    )
    if not context.index.equals(expected):
        raise TimerS1AdapterError(
            "Live Timer-S1 context is not the exact hourly continuation "
            "ending before the plan."
        )
    return context.to_numpy(dtype=np.float32)


def _output_to_numpy(output: Any) -> np.ndarray:
    value = output
    if callable(getattr(value, "detach", None)):
        value = value.detach()
    # Timer-S1 weights are BF16; converting before NumPy avoids unsupported
    # bfloat16 arrays on common NumPy versions.
    if callable(getattr(value, "float", None)):
        value = value.float()
    if callable(getattr(value, "cpu", None)):
        value = value.cpu()
    if callable(getattr(value, "numpy", None)):
        value = value.numpy()
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TimerS1AdapterError(
            "Timer-S1 output cannot be converted to a numeric array."
        ) from exc


def _predict_quantiles(
    contexts: np.ndarray,
    *,
    horizon: int,
    model: TimerS1Model,
    torch_module: Any,
) -> dict[str, np.ndarray]:
    if contexts.ndim != 2 or contexts.shape[0] == 0:
        raise TimerS1AdapterError(
            "Timer-S1 input must have shape [batch, context_length]."
        )
    if not bool(np.isfinite(contexts).all()):
        raise TimerS1AdapterError("Timer-S1 input contains non-finite values.")
    if not callable(getattr(model, "eval", None)):
        raise TimerS1AdapterError("Injected Timer-S1 model has no eval() method.")
    if not callable(getattr(model, "generate", None)):
        raise TimerS1AdapterError(
            "Injected Timer-S1 model has no generate() method."
        )
    if not callable(getattr(torch_module, "as_tensor", None)):
        raise TimerS1AdapterError(
            "Injected torch module has no as_tensor() function."
        )
    if not callable(getattr(torch_module, "inference_mode", None)):
        raise TimerS1AdapterError(
            "Injected torch module has no inference_mode() context."
        )

    # Match the loaded patch projection.  In particular, a BF16 checkpoint
    # loaded with ``torch_dtype=torch.bfloat16`` cannot safely receive an
    # unconditional float32 tensor (the first Linear layer may reject mixed
    # mat1/mat2 dtypes).  The released model's patch-embedding Linear weight
    # is the exact contract; ``PreTrainedModel.dtype`` and the first parameter
    # are conservative fallbacks, while float32 keeps test doubles supported.
    core = getattr(model, "model", None)
    embed_layer = getattr(core, "embed_layer", None)
    hidden_layer = getattr(embed_layer, "hidden_layer", None)
    embed_weight = getattr(hidden_layer, "weight", None)
    dtype = getattr(embed_weight, "dtype", None)
    if dtype is None:
        dtype = getattr(model, "dtype", None)
    if dtype is None and callable(getattr(model, "parameters", None)):
        try:
            dtype = getattr(next(iter(model.parameters())), "dtype", None)
        except (StopIteration, TypeError):
            dtype = None
    if dtype is None:
        dtype = getattr(torch_module, "float32", None)
    tensor = torch_module.as_tensor(contexts, dtype=dtype)
    device = getattr(model, "device", None)
    if device is not None:
        if not callable(getattr(tensor, "to", None)):
            raise TimerS1AdapterError(
                "Timer-S1 input tensor cannot be moved to model.device."
            )
        tensor = tensor.to(device)

    model.eval()
    with torch_module.inference_mode():
        output = model.generate(
            tensor,
            max_new_tokens=int(horizon),
            revin=True,
        )
    array = _output_to_numpy(output)
    expected_shape = (contexts.shape[0], 9, horizon)
    if array.shape != expected_shape:
        raise TimerS1AdapterError(
            "Timer-S1 generate() must return official shape [B, 9, H]; "
            f"expected {expected_shape}, received {array.shape}."
        )
    if not bool(np.isfinite(array).all()):
        raise TimerS1AdapterError(
            "Timer-S1 generate() returned missing or non-finite quantiles."
        )
    selected = {
        name: array[:, index, :]
        for name, index in TIMER_S1_QUANTILE_INDICES.items()
    }
    if bool(
        (selected["q10"] > selected["q50"]).any()
        or (selected["q50"] > selected["q90"]).any()
    ):
        raise TimerS1AdapterError("Timer-S1 returned crossing quantiles.")
    return selected


def _runtime_components(
    *,
    runtime: TimerS1Runtime | Any | None,
    model: TimerS1Model | Any | None,
    torch_module: Any | None,
    model_id: str,
    revision: str,
    device_map: Any,
    local_files_only: bool,
    torch_dtype: Any | None,
    model_kwargs: Mapping[str, Any] | None,
) -> tuple[Any, Any]:
    if runtime is not None and model is not None:
        raise TimerS1AdapterError("Inject runtime or model, not both.")
    if runtime is None and model is None:
        loaded = load_timer_s1_runtime(
            model_id=model_id,
            revision=revision,
            device_map=device_map,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
            model_kwargs=model_kwargs,
            torch_module=torch_module,
        )
        return loaded.model, loaded.torch_module
    if runtime is not None:
        runtime_model = getattr(runtime, "model", None)
        if runtime_model is None:
            # Accept the repository's pipeline-style runtime as an injection
            # convenience, provided it exposes Timer-S1's generate interface.
            runtime_model = getattr(runtime, "pipeline", None)
        if runtime_model is None:
            raise TimerS1AdapterError("Injected runtime exposes no model.")
        runtime_torch = getattr(runtime, "torch_module", None)
        return runtime_model, runtime_torch or torch_module or _load_torch()
    return model, torch_module or _load_torch()


def make_timer_s1_backtest_executor(
    *,
    data: Any,
    context_length: int,
    batch_size: int = 1,
    runtime: TimerS1Runtime | Any | None = None,
    model: TimerS1Model | Any | None = None,
    torch_module: Any | None = None,
    model_id: str = DEFAULT_TIMER_S1_MODEL_ID,
    revision: str = DEFAULT_TIMER_S1_REVISION,
    device_map: Any = "auto",
    local_files_only: bool = False,
    torch_dtype: Any | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> ChronosGroupExecutor:
    """Create an executor accepted by ``execute_grouped_chronos_backtest``.

    Each context contains only the target and ends immediately before the
    plan's first delivery timestamp.  ``batch_size`` batches forecast origins
    *within one horizon group*; it never batches feature columns.

    The raw executor returns exactly ``delivery_start_utc``, ``q10``, ``q50``,
    ``q90`` and ``actual``.  The shared Chronos execution wrapper attaches the
    trusted plan origins and performs a second exact-coverage validation.
    """

    context_length = _positive_integer(
        context_length,
        name="context_length",
        maximum=TIMER_S1_MAX_CONTEXT_LENGTH,
    )
    batch_size = _positive_integer(batch_size, name="batch_size")
    target = _extract_target(data)
    component_cache: list[tuple[Any, Any]] = []

    def components() -> tuple[Any, Any]:
        if not component_cache:
            component_cache.append(
                _runtime_components(
                    runtime=runtime,
                    model=model,
                    torch_module=torch_module,
                    model_id=model_id,
                    revision=revision,
                    device_map=device_map,
                    local_files_only=local_files_only,
                    torch_dtype=torch_dtype,
                    model_kwargs=model_kwargs,
                )
            )
        return component_cache[0]

    def execute(
        *,
        plans: Sequence[ChronosDeliveryPlan],
        horizon: int,
    ) -> pd.DataFrame:
        if not plans:
            raise TimerS1AdapterError("Timer-S1 horizon group is empty.")
        horizon = _positive_integer(horizon, name="horizon")
        for plan in plans:
            _validate_plan(plan, horizon=horizon)

        rows: list[pd.DataFrame] = []
        for start in range(0, len(plans), batch_size):
            plan_batch = tuple(plans[start : start + batch_size])
            contexts: list[np.ndarray] = []
            actuals: list[np.ndarray] = []
            for plan in plan_batch:
                context, actual = _backtest_context_and_actual(
                    target,
                    plan,
                    context_length=context_length,
                )
                contexts.append(context)
                actuals.append(actual)
            context_matrix = np.stack(contexts, axis=0)
            if context_matrix.shape != (len(plan_batch), context_length):
                raise TimerS1AdapterError(
                    "Timer-S1 contexts do not share the exact context_length."
                )
            # Validate target coverage before a lazy production runtime can
            # trigger a multi-gigabyte checkpoint load.
            inference_model, inference_torch = components()
            quantiles = _predict_quantiles(
                context_matrix,
                horizon=horizon,
                model=inference_model,
                torch_module=inference_torch,
            )
            for row, (plan, actual) in enumerate(zip(plan_batch, actuals)):
                rows.append(
                    pd.DataFrame(
                        {
                            "delivery_start_utc": plan.delivery_index_utc,
                            "q10": quantiles["q10"][row],
                            "q50": quantiles["q50"][row],
                            "q90": quantiles["q90"][row],
                            "actual": actual,
                        }
                    )
                )
        result = pd.concat(rows, ignore_index=True)
        expected = pd.DatetimeIndex(
            np.concatenate(
                [plan.delivery_index_utc.to_numpy() for plan in plans]
            ),
            tz="UTC",
            name="delivery_start_utc",
        )
        delivered = pd.DatetimeIndex(
            result["delivery_start_utc"],
            name="delivery_start_utc",
        )
        if delivered.has_duplicates or not delivered.equals(expected):
            raise TimerS1AdapterError(
                "Timer-S1 output does not exactly preserve plan coverage."
            )
        numeric = result[["q10", "q50", "q90", "actual"]].to_numpy(
            dtype=np.float64
        )
        if not bool(np.isfinite(numeric).all()):
            raise TimerS1AdapterError(
                "Timer-S1 backtest output contains non-finite values."
            )
        return result

    return execute


def make_timer_s1_live_executor(
    *,
    data: Any,
    context_length: int,
    runtime: TimerS1Runtime | Any | None = None,
    model: TimerS1Model | Any | None = None,
    torch_module: Any | None = None,
    model_id: str = DEFAULT_TIMER_S1_MODEL_ID,
    revision: str = DEFAULT_TIMER_S1_REVISION,
    device_map: Any = "auto",
    local_files_only: bool = False,
    torch_dtype: Any | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> ChronosLiveExecutor:
    """Create a strict univariate Timer-S1 live executor.

    The target must finish exactly one physical UTC hour before the complete
    23/24/25-hour delivery plan.  No native covariates are accepted or encoded.
    """

    context_length = _positive_integer(
        context_length,
        name="context_length",
        maximum=TIMER_S1_MAX_CONTEXT_LENGTH,
    )
    target = _extract_target(data)
    component_cache: list[tuple[Any, Any]] = []

    def components() -> tuple[Any, Any]:
        if not component_cache:
            component_cache.append(
                _runtime_components(
                    runtime=runtime,
                    model=model,
                    torch_module=torch_module,
                    model_id=model_id,
                    revision=revision,
                    device_map=device_map,
                    local_files_only=local_files_only,
                    torch_dtype=torch_dtype,
                    model_kwargs=model_kwargs,
                )
            )
        return component_cache[0]

    def execute(
        *,
        plan: ChronosDeliveryPlan,
        horizon: int,
    ) -> pd.DataFrame:
        horizon = _positive_integer(horizon, name="horizon")
        _validate_plan(plan, horizon=horizon)
        context = _live_context(
            target,
            plan,
            context_length=context_length,
        )
        inference_model, inference_torch = components()
        quantiles = _predict_quantiles(
            context[np.newaxis, :],
            horizon=horizon,
            model=inference_model,
            torch_module=inference_torch,
        )
        result = pd.DataFrame(
            {
                "delivery_start_utc": plan.delivery_index_utc,
                "q10": quantiles["q10"][0],
                "q50": quantiles["q50"][0],
                "q90": quantiles["q90"][0],
            }
        )
        numeric = result[["q10", "q50", "q90"]].to_numpy(dtype=np.float64)
        if not bool(np.isfinite(numeric).all()):
            raise TimerS1AdapterError(
                "Timer-S1 live output contains non-finite values."
            )
        return result

    return execute


def run_timer_s1_backtest(
    plans: Sequence[ChronosDeliveryPlan],
    **executor_kwargs: Any,
) -> pd.DataFrame:
    """Run Timer-S1 and return the shared strict Chronos OOF contract."""

    executor = make_timer_s1_backtest_executor(**executor_kwargs)
    return execute_grouped_chronos_backtest(plans, executor)


def run_timer_s1_live_forecast(
    plan: ChronosDeliveryPlan,
    **executor_kwargs: Any,
) -> pd.DataFrame:
    """Run Timer-S1 and return the shared strict Chronos future contract."""

    executor = make_timer_s1_live_executor(**executor_kwargs)
    return execute_chronos_live_forecast(plan, executor)


# Names parallel to the existing Chronos adapter, kept as explicit aliases so
# integration code can swap factories without guessing different terminology.
make_timer_s1_forecasting_executor = make_timer_s1_backtest_executor
make_timer_s1_live_forecast_executor = make_timer_s1_live_executor


__all__ = [
    "DEFAULT_TIMER_S1_MODEL_ID",
    "DEFAULT_TIMER_S1_REVISION",
    "TIMER_S1_MAX_CONTEXT_LENGTH",
    "TIMER_S1_QUANTILE_INDICES",
    "TIMER_S1_QUANTILE_LEVELS",
    "TimerS1AdapterError",
    "TimerS1Model",
    "TimerS1Runtime",
    "load_timer_s1_runtime",
    "make_timer_s1_backtest_executor",
    "make_timer_s1_forecasting_executor",
    "make_timer_s1_live_executor",
    "make_timer_s1_live_forecast_executor",
    "run_timer_s1_backtest",
    "run_timer_s1_live_forecast",
]
