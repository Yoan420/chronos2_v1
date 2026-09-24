from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.chronos_adapter import generate_delivery_plans
from chronos2_hourly.historical_price_replay import (
    HistoricalPriceReplayError,
    generate_checkpointed_price_replay,
    replay_identity,
    select_replay_days,
)


def _target() -> pd.Series:
    index = pd.date_range(
        "2024-03-28", "2024-04-03", inclusive="left", freq="h", tz="UTC"
    )
    return pd.Series(np.arange(len(index), dtype=float), index=index)


def _plans():
    return generate_delivery_plans(
        "2024-03-30",
        "2024-04-01",
        forecast_origin_local_time="08:00",
        timezone="Europe/Paris",
    )


def _identity(plans):
    return replay_identity(
        plans,
        zone="FR",
        timezone="Europe/Paris",
        model_id="amazon/chronos-2",
        model_revision="pinned",
        residual_load_source="chronos2_historical_replay",
        source_hashes={"pit": "abc"},
        feature_schema_sha256="schema",
    )


def _executor(target: pd.Series, calls: list[int]):
    target_utc = target.copy()
    target_utc.index = target_utc.index.tz_convert("UTC")

    def execute(plans):
        calls.append(len(plans))
        pieces = []
        for plan in plans:
            actual = target_utc.loc[plan.delivery_index_utc]
            pieces.append(
                pd.DataFrame(
                    {
                        "delivery_start_utc": plan.delivery_index_utc,
                        "forecast_origin_utc": plan.forecast_origin_utc,
                        "q10": actual.to_numpy() - 1.0,
                        "q50": actual.to_numpy(),
                        "q90": actual.to_numpy() + 1.0,
                        "actual": actual.to_numpy(),
                    }
                )
            )
        return pd.concat(pieces, ignore_index=True)

    return execute


def test_checkpoint_resume_and_dst_day_lengths(tmp_path: Path) -> None:
    plans = _plans()
    assert [plan.horizon for plan in plans] == [24, 23, 24]
    target = _target()
    output = tmp_path / "price.csv.gz"
    calls: list[int] = []
    first = generate_checkpointed_price_replay(
        plans,
        target=target,
        execute_chunk=_executor(target, calls),
        output_path=output,
        identity=_identity(plans),
        chunk_days=1,
    )
    assert len(first) == 71
    assert calls == [1, 1, 1]

    resumed_calls: list[int] = []
    second = generate_checkpointed_price_replay(
        plans,
        target=target,
        execute_chunk=_executor(target, resumed_calls),
        output_path=output,
        identity=_identity(plans),
        chunk_days=2,
    )
    assert resumed_calls == []
    pd.testing.assert_frame_equal(first, second)

    rebuilt_calls: list[int] = []
    rebuilt = generate_checkpointed_price_replay(
        plans,
        target=target,
        execute_chunk=_executor(target, rebuilt_calls),
        output_path=output,
        identity=_identity(plans),
        chunk_days=2,
        resume=False,
    )
    assert rebuilt_calls == [2, 1]
    pd.testing.assert_frame_equal(first, rebuilt)


def test_resume_rejects_identity_change(tmp_path: Path) -> None:
    plans = _plans()
    target = _target()
    output = tmp_path / "price.csv.gz"
    generate_checkpointed_price_replay(
        plans,
        target=target,
        execute_chunk=_executor(target, []),
        output_path=output,
        identity=_identity(plans),
    )
    incompatible = {**_identity(plans), "model_revision": "changed"}
    with pytest.raises(HistoricalPriceReplayError, match="model_revision"):
        generate_checkpointed_price_replay(
            plans,
            target=target,
            execute_chunk=_executor(target, []),
            output_path=output,
            identity=incompatible,
        )


def test_rejects_an_executor_with_the_wrong_forecast_origin(
    tmp_path: Path,
) -> None:
    plans = _plans()
    target = _target()
    valid = _executor(target, [])

    def wrong_origin(chunk):
        frame = valid(chunk)
        frame["forecast_origin_utc"] = pd.to_datetime(
            frame["forecast_origin_utc"], utc=True
        ) - pd.Timedelta(hours=1)
        return frame

    with pytest.raises(HistoricalPriceReplayError, match="origines"):
        generate_checkpointed_price_replay(
            plans,
            target=target,
            execute_chunk=wrong_origin,
            output_path=tmp_path / "wrong-origin.csv.gz",
            identity=_identity(plans),
        )


def test_select_replay_days_is_exact(tmp_path: Path) -> None:
    plans = _plans()
    target = _target()
    replay = generate_checkpointed_price_replay(
        plans,
        target=target,
        execute_chunk=_executor(target, []),
        output_path=tmp_path / "price.csv.gz",
        identity=_identity(plans),
    )
    selected = select_replay_days(replay, plans[1:])
    assert len(selected) == 47
    assert selected.index.equals(
        plans[1].delivery_index_utc.append(plans[2].delivery_index_utc)
    )
