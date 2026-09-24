from __future__ import annotations

import sys
import types
from unittest.mock import patch

import numpy as np
import pandas as pd

from chronos2_modular.common import ZoneData
from chronos2_modular.forecasting import future_proxy_frame, load_model


class _RecordingPipeline:
    calls: list[tuple[str, dict[str, object]]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: object) -> object:
        cls.calls.append((model_id, kwargs))
        return object()


def _load_with_model_config(model_config: dict[str, object]) -> dict[str, object]:
    chronos_module = types.ModuleType("chronos")
    chronos_module.Chronos2Pipeline = _RecordingPipeline
    _RecordingPipeline.calls.clear()
    with (
        patch.dict(sys.modules, {"chronos": chronos_module}),
        patch(
            "chronos2_modular.forecasting.resolve_device",
            return_value=("cpu", "test-dtype"),
        ),
        patch("chronos2_modular.forecasting.configure_huggingface_ssl"),
    ):
        load_model({"model": model_config}, None, True)
    return _RecordingPipeline.calls[0][1]


def test_load_model_pins_configured_revision() -> None:
    kwargs = _load_with_model_config(
        {
            "model_id": "amazon/chronos-2",
            "revision": "29ec3766d36d6f73f0696f85560a422f50e8498c",
        }
    )

    assert kwargs["revision"] == "29ec3766d36d6f73f0696f85560a422f50e8498c"


def test_load_model_keeps_previous_behavior_without_revision() -> None:
    kwargs = _load_with_model_config({"model_id": "amazon/chronos-2"})

    assert kwargs == {
        "device_map": "cpu",
        "local_files_only": True,
        "dtype": "test-dtype",
    }


def test_future_proxy_keeps_non_oracle_strategies_history_only() -> None:
    history_index = pd.date_range(
        "2024-04-01 00:00",
        periods=48,
        freq="h",
        tz="Europe/Paris",
    )
    origin_position = 24
    future_index = history_index[origin_position:]
    covariates = pd.DataFrame(
        {"load": np.arange(48, dtype=float)},
        index=history_index,
    )
    model_context = pd.DataFrame(
        {
            "known_load_oracle": np.concatenate(
                [np.full(24, np.nan), np.arange(100.0, 124.0)]
            ),
            "known_load_persistence": np.concatenate(
                [np.full(24, np.nan), np.arange(1000.0, 1024.0)]
            ),
        },
        index=history_index,
    )
    data = ZoneData(
        zone="FR",
        timezone="Europe/Paris",
        frequency="h",
        target=pd.Series(np.arange(48, dtype=float), index=history_index),
        covariates=covariates,
        model_context_covariates=model_context,
        known_future_columns=[
            "known_load_oracle",
            "known_load_persistence",
        ],
        coverage=pd.DataFrame(),
        input_manifest=pd.DataFrame(),
        diagnostics={},
    )

    proxy = future_proxy_frame(data, future_index, origin_position)

    np.testing.assert_array_equal(
        proxy["known_load_oracle"].to_numpy(),
        np.arange(100.0, 124.0, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        proxy["known_load_persistence"].to_numpy(),
        np.full(24, 23.0, dtype=np.float32),
    )
