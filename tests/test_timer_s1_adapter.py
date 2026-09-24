from __future__ import annotations

import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.chronos_adapter import (
    build_delivery_plan,
    execute_chronos_live_forecast,
    execute_grouped_chronos_backtest,
    generate_delivery_plans,
)
from chronos2_hourly.timer_s1_adapter import (
    DEFAULT_TIMER_S1_MODEL_ID,
    DEFAULT_TIMER_S1_REVISION,
    TimerS1AdapterError,
    TimerS1Runtime,
    load_timer_s1_runtime,
    make_timer_s1_backtest_executor,
    make_timer_s1_live_executor,
    run_timer_s1_backtest,
    run_timer_s1_live_forecast,
)


class FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values)
        self.device = None

    def to(self, device):
        self.device = device
        return self


class _InferenceContext:
    def __init__(self, owner: "FakeTorch") -> None:
        self.owner = owner

    def __enter__(self):
        assert not self.owner.in_inference_mode
        self.owner.in_inference_mode = True
        self.owner.inference_entries += 1
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.owner.in_inference_mode = False
        return False


class FakeTorch:
    float32 = np.float32

    def __init__(self) -> None:
        self.in_inference_mode = False
        self.inference_entries = 0
        self.tensors: list[FakeTensor] = []

    def as_tensor(self, values, *, dtype=None):
        tensor = FakeTensor(np.asarray(values, dtype=dtype))
        self.tensors.append(tensor)
        return tensor

    def inference_mode(self):
        return _InferenceContext(self)


class FakeTimerModel:
    device = "fake:0"

    def __init__(
        self,
        torch_module: FakeTorch,
        *,
        bad_shape: bool = False,
        nonfinite: bool = False,
        crossing: bool = False,
    ) -> None:
        self.torch_module = torch_module
        self.bad_shape = bad_shape
        self.nonfinite = nonfinite
        self.crossing = crossing
        self.eval_calls = 0
        self.calls: list[dict[str, object]] = []

    def eval(self):
        self.eval_calls += 1
        return self

    def generate(self, sequences, *, max_new_tokens, revin):
        assert self.torch_module.in_inference_mode
        assert isinstance(sequences, FakeTensor)
        context = sequences.values.copy()
        batch = context.shape[0]
        horizon = int(max_new_tokens)
        self.calls.append(
            {
                "context": context,
                "horizon": horizon,
                "revin": revin,
                "device": sequences.device,
            }
        )
        if self.bad_shape:
            return np.zeros((batch, 8, horizon), dtype=np.float32)

        steps = np.arange(horizon, dtype=np.float32)[None, None, :]
        quantile_offsets = (
            np.arange(9, dtype=np.float32)[None, :, None] * 10.0
        )
        last_value = context[:, -1][:, None, None]
        output = last_value + quantile_offsets + steps
        if self.nonfinite:
            output[0, 4, 0] = np.nan
        if self.crossing:
            output[0, 4, 0] = output[0, 0, 0] - 1.0
        return output


def _target_for_plans(
    plans,
    *,
    context_length: int,
    timezone: str = "UTC",
) -> pd.Series:
    index = pd.date_range(
        plans[0].delivery_start_utc - pd.Timedelta(hours=context_length),
        plans[-1].delivery_index_utc[-1],
        freq="h",
        tz="UTC",
    ).tz_convert(timezone)
    return pd.Series(
        np.arange(len(index), dtype=np.float64) + 100.0,
        index=index,
        name="price",
    )


def _live_target(plan, *, context_length: int) -> pd.Series:
    index = pd.date_range(
        end=plan.delivery_start_utc - pd.Timedelta(hours=1),
        periods=context_length,
        freq="h",
        tz="UTC",
    )
    return pd.Series(np.arange(context_length, dtype=float), index=index)


def _runtime(
    *,
    bad_shape: bool = False,
    nonfinite: bool = False,
    crossing: bool = False,
) -> tuple[TimerS1Runtime, FakeTimerModel, FakeTorch]:
    fake_torch = FakeTorch()
    model = FakeTimerModel(
        fake_torch,
        bad_shape=bad_shape,
        nonfinite=nonfinite,
        crossing=crossing,
    )
    runtime = TimerS1Runtime(model=model, torch_module=fake_torch)
    return runtime, model, fake_torch


def test_loader_pins_checkpoint_and_uses_eval_without_downloading() -> None:
    fake_torch = FakeTorch()
    model = FakeTimerModel(fake_torch)
    captured: dict[str, object] = {}

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            captured["model_id"] = model_id
            captured.update(kwargs)
            return model

    runtime = load_timer_s1_runtime(
        torch_module=fake_torch,
        auto_model_class=FakeAutoModel,
    )

    assert runtime.model is model
    assert runtime.model_id == DEFAULT_TIMER_S1_MODEL_ID
    assert runtime.revision == DEFAULT_TIMER_S1_REVISION
    assert captured == {
        "model_id": DEFAULT_TIMER_S1_MODEL_ID,
        "trust_remote_code": True,
        "revision": DEFAULT_TIMER_S1_REVISION,
        "device_map": "auto",
        "local_files_only": False,
    }
    assert model.eval_calls == 1


def test_loader_rejects_unsupported_transformers_before_model_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_transformers = ModuleType("transformers")
    fake_transformers.__version__ = "5.0.0"
    fake_transformers.AutoModelForCausalLM = object
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(ImportError, match=r">=4\.57\.1,<4\.58"):
        load_timer_s1_runtime(torch_module=FakeTorch())


def test_backtest_batches_only_equal_horizons_and_maps_official_quantiles() -> None:
    context_length = 48
    plans = generate_delivery_plans("2024-03-30", "2024-04-01")
    data = SimpleNamespace(
        target=_target_for_plans(
            plans,
            context_length=context_length,
            timezone="Europe/Paris",
        ),
        # These are deliberately not passed to Timer-S1: its released API is
        # univariate, and treating columns as batch rows would be incorrect.
        covariates=pd.DataFrame({"forbidden_feature": [1.0]}),
    )
    runtime, model, fake_torch = _runtime()
    executor = make_timer_s1_backtest_executor(
        data=data,
        context_length=context_length,
        batch_size=8,
        runtime=runtime,
    )

    result = execute_grouped_chronos_backtest(plans, executor)

    assert [(call["horizon"], call["context"].shape[0]) for call in model.calls] == [
        (23, 1),
        (24, 2),
    ]
    assert all(call["revin"] is True for call in model.calls)
    assert all(call["device"] == "fake:0" for call in model.calls)
    assert fake_torch.inference_entries == 2
    assert model.eval_calls == 2
    assert result.index.equals(
        plans[0].delivery_index_utc.append(
            [plan.delivery_index_utc for plan in plans[1:]]
        )
    )
    assert list(result.columns) == [
        "forecast_origin_utc",
        "q10",
        "q50",
        "q90",
        "actual",
    ]
    assert np.allclose(result["q50"] - result["q10"], 40.0)
    assert np.allclose(result["q90"] - result["q50"], 40.0)
    assert np.allclose(
        result["actual"].to_numpy(),
        data.target.tz_convert("UTC").reindex(result.index).to_numpy(),
    )
    assert bool((result["forecast_origin_utc"] < result.index).all())

    context_by_horizon = {
        int(call["horizon"]): np.asarray(call["context"])
        for call in model.calls
    }
    spring_context = context_by_horizon[23][0]
    spring_plan = plans[1]
    expected = data.target.copy()
    expected.index = expected.index.tz_convert("UTC")
    expected_context = expected.loc[
        spring_plan.delivery_start_utc - pd.Timedelta(hours=context_length) :
        spring_plan.delivery_start_utc - pd.Timedelta(hours=1)
    ]
    assert len(expected_context) == context_length
    assert np.array_equal(spring_context, expected_context.to_numpy(np.float32))


def test_input_uses_timer_patch_embedding_weight_dtype() -> None:
    plan = build_delivery_plan("2024-06-15")
    target = _target_for_plans((plan,), context_length=24)
    runtime, model, _ = _runtime()
    model.model = SimpleNamespace(
        embed_layer=SimpleNamespace(
            hidden_layer=SimpleNamespace(
                weight=SimpleNamespace(dtype=np.float16),
            )
        )
    )

    run_timer_s1_backtest(
        (plan,),
        data=target,
        context_length=24,
        runtime=runtime,
    )

    assert np.asarray(model.calls[0]["context"]).dtype == np.dtype(np.float16)


def test_raw_executor_has_exact_compatible_columns_and_batches_origins() -> None:
    context_length = 24
    plans = generate_delivery_plans("2024-06-14", "2024-06-15")
    target = _target_for_plans(plans, context_length=context_length)
    runtime, model, _ = _runtime()
    executor = make_timer_s1_backtest_executor(
        data=target,
        context_length=context_length,
        batch_size=16,
        runtime=runtime,
    )

    raw = executor(plans=plans, horizon=24)

    assert list(raw.columns) == [
        "delivery_start_utc",
        "q10",
        "q50",
        "q90",
        "actual",
    ]
    assert str(pd.DatetimeIndex(raw["delivery_start_utc"]).tz) == "UTC"
    assert len(model.calls) == 1
    assert np.asarray(model.calls[0]["context"]).shape == (2, context_length)


@pytest.mark.parametrize(
    ("day", "horizon"),
    [("2024-03-31", 23), ("2024-10-27", 25)],
)
def test_dst_backtest_context_and_complete_coverage(day: str, horizon: int) -> None:
    plan = build_delivery_plan(day)
    context_length = 72
    target = _target_for_plans((plan,), context_length=context_length)
    runtime, model, _ = _runtime()

    result = run_timer_s1_backtest(
        (plan,),
        data=target,
        context_length=context_length,
        batch_size=4,
        runtime=runtime,
    )

    assert len(result) == horizon
    assert result.index.equals(plan.delivery_index_utc)
    assert np.asarray(model.calls[0]["context"]).shape == (1, context_length)
    assert np.asarray(model.calls[0]["context"])[0, -1] == target.iloc[
        context_length - 1
    ]


def test_autumn_dst_live_executor_preserves_25_physical_utc_hours() -> None:
    plan = build_delivery_plan("2024-10-27")
    context_length = 96
    target = _live_target(plan, context_length=context_length)
    runtime, model, _ = _runtime()

    raw_executor = make_timer_s1_live_executor(
        data=SimpleNamespace(target=target),
        context_length=context_length,
        runtime=runtime,
    )
    result = execute_chronos_live_forecast(plan, raw_executor)

    assert len(result) == 25
    assert result.index.equals(plan.delivery_index_utc)
    assert bool(result["forecast_origin_utc"].eq(plan.forecast_origin_utc).all())
    assert model.calls[0]["horizon"] == 25
    assert model.calls[0]["revin"] is True
    assert np.asarray(model.calls[0]["context"]).shape == (1, context_length)


def test_live_wrapper_accepts_direct_model_and_fake_torch() -> None:
    plan = build_delivery_plan("2024-03-31")
    target = _live_target(plan, context_length=32)
    fake_torch = FakeTorch()
    model = FakeTimerModel(fake_torch)

    result = run_timer_s1_live_forecast(
        plan,
        data=target,
        context_length=32,
        model=model,
        torch_module=fake_torch,
    )

    assert len(result) == 23
    assert model.calls[0]["horizon"] == 23


@pytest.mark.parametrize(
    ("runtime_options", "message"),
    [
        ({"bad_shape": True}, r"shape \[B, 9, H\]"),
        ({"nonfinite": True}, "non-finite"),
        ({"crossing": True}, "crossing quantiles"),
    ],
)
def test_rejects_malformed_model_outputs(runtime_options, message) -> None:
    plan = build_delivery_plan("2024-06-15")
    target = _target_for_plans((plan,), context_length=24)
    runtime, _, _ = _runtime(**runtime_options)
    executor = make_timer_s1_backtest_executor(
        data=target,
        context_length=24,
        runtime=runtime,
    )

    with pytest.raises(TimerS1AdapterError, match=message):
        executor(plans=(plan,), horizon=24)


@pytest.mark.parametrize("defect", ["nan", "gap", "naive"])
def test_rejects_nonfinite_or_noncontinuous_utc_target(defect: str) -> None:
    plan = build_delivery_plan("2024-06-15")
    target = _target_for_plans((plan,), context_length=24)
    if defect == "nan":
        target.iloc[3] = np.nan
        message = "non-finite"
    elif defect == "gap":
        target = target.drop(target.index[3])
        message = "continuous"
    else:
        target.index = target.index.tz_localize(None)
        message = "timezone-aware"

    runtime, _, _ = _runtime()
    with pytest.raises(TimerS1AdapterError, match=message):
        make_timer_s1_backtest_executor(
            data=target,
            context_length=24,
            runtime=runtime,
        )


def test_rejects_insufficient_context_delivery_coverage_and_wrong_horizon() -> None:
    plan = build_delivery_plan("2024-06-15")
    full = _target_for_plans((plan,), context_length=24)
    runtime, _, _ = _runtime()

    insufficient = full.iloc[10:]
    executor = make_timer_s1_backtest_executor(
        data=insufficient,
        context_length=24,
        runtime=runtime,
    )
    with pytest.raises(TimerS1AdapterError, match="fewer than 24"):
        executor(plans=(plan,), horizon=24)

    missing_delivery = full.iloc[:-1]
    executor = make_timer_s1_backtest_executor(
        data=missing_delivery,
        context_length=24,
        runtime=runtime,
    )
    with pytest.raises(TimerS1AdapterError, match="does not cover every hour"):
        executor(plans=(plan,), horizon=24)

    executor = make_timer_s1_backtest_executor(
        data=full,
        context_length=24,
        runtime=runtime,
    )
    with pytest.raises(TimerS1AdapterError, match="23/24/25-hour plan"):
        executor(plans=(plan,), horizon=23)


def test_live_target_must_end_immediately_before_plan() -> None:
    plan = build_delivery_plan("2024-06-15")
    target = _live_target(plan, context_length=24).iloc[:-1]
    runtime, _, _ = _runtime()
    executor = make_timer_s1_live_executor(
        data=target,
        context_length=23,
        runtime=runtime,
    )

    with pytest.raises(TimerS1AdapterError, match="end exactly one UTC hour"):
        executor(plan=plan, horizon=24)
