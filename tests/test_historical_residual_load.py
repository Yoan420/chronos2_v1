from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import historical_residual_load as replay


class FakeChronosPipeline:
    def __init__(self, *, reversed_quantiles: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.reversed_quantiles = reversed_quantiles

    def predict_df(self, context: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        self.calls.append({"context": context.copy(), **kwargs})
        frames: list[pd.DataFrame] = []
        for position, (item_id, item) in enumerate(
            context.groupby("item_id", sort=True), start=1
        ):
            last = pd.Timestamp(item["timestamp"].max())
            length = int(kwargs["prediction_length"])
            index = pd.date_range(
                last + pd.Timedelta(hours=1), periods=length, freq="h"
            )
            median = position * 100.0 + np.arange(length, dtype=float)
            lower = median + 1.0 if self.reversed_quantiles else median - 1.0
            frames.append(
                pd.DataFrame(
                    {
                        "item_id": item_id,
                        "timestamp": index,
                        "target_name": "target",
                        "0.1": lower,
                        "0.5": median,
                        "0.9": median + 1.0,
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)


def _vintage_frame(
    end: pd.Timestamp,
    *,
    periods: int = replay.CONTEXT_LENGTH + 96,
    offset: float = 0.0,
) -> pd.DataFrame:
    index = pd.date_range(end=end, periods=periods, freq="h", tz="UTC")
    availability = index - pd.Timedelta(minutes=5)
    return pd.DataFrame(
        {
            "value_time_utc": index,
            "snapshot_time_utc": availability,
            "revision_time_utc": availability,
            "value": offset + np.arange(periods, dtype=float),
            "downloaded_at_utc": availability + pd.Timedelta(minutes=1),
        }
    )


def _five_vintages(end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    return {
        series_name: _vintage_frame(end, offset=position * 10_000.0)
        for position, series_name in enumerate(
            replay.COUNTRY_OBSERVED_SERIES.values(), start=1
        )
    }


def test_delivery_plans_use_local_cutoff_dst_and_full_day_mask() -> None:
    spring = replay.build_historical_delivery_plans("2025-03-30", "2025-03-30")[0]
    autumn = replay.build_historical_delivery_plans("2025-10-26", "2025-10-26")[0]

    assert len(spring.delivery_index_utc) == 23
    assert spring.request_cutoff_utc == pd.Timestamp("2025-03-29 07:00", tz="UTC")
    assert len(autumn.delivery_index_utc) == 25
    assert autumn.request_cutoff_utc == pd.Timestamp("2025-10-25 06:00", tz="UTC")
    post_spring = replay.build_historical_delivery_plans(
        "2025-03-31", "2025-03-31"
    )[0]
    post_autumn = replay.build_historical_delivery_plans(
        "2025-10-27", "2025-10-27"
    )[0]
    assert post_spring.request_cutoff_utc == pd.Timestamp(
        "2025-03-30 06:00", tz="UTC"
    )
    assert post_autumn.request_cutoff_utc == pd.Timestamp(
        "2025-10-26 07:00", tz="UTC"
    )

    expected = spring.delivery_index_utc
    mask = pd.Series(True, index=expected)
    selected = replay.build_historical_delivery_plans(
        "2025-03-29", "2025-03-31", delivery_mask=mask
    )
    assert [plan.delivery_day_local.isoformat() for plan in selected] == [
        "2025-03-30"
    ]
    with pytest.raises(replay.HistoricalResidualLoadError, match="partie|partiel"):
        replay.build_historical_delivery_plans(
            "2025-03-30",
            "2025-03-30",
            delivery_mask=mask.iloc[:-1],
        )


def test_context_selection_uses_both_pit_clocks_and_interpolates_six_hours() -> None:
    cutoff = pd.Timestamp("2025-03-29 07:00", tz="UTC")
    source_name = replay.COUNTRY_OBSERVED_SERIES["fr"]
    frame = _vintage_frame(cutoff)
    target_time = frame["value_time_utc"].iloc[-1]
    baseline_value = float(frame["value"].iloc[-1])
    future_revision = frame.iloc[[-1]].copy()
    future_revision["value"] = 999_999.0
    future_revision["revision_time_utc"] = cutoff + pd.Timedelta(minutes=1)
    future_snapshot = frame.iloc[[-1]].copy()
    future_snapshot["value"] = 888_888.0
    future_snapshot["snapshot_time_utc"] = cutoff + pd.Timedelta(minutes=1)
    gap_rows = np.arange(len(frame) - 100, len(frame) - 94)
    frame.loc[gap_rows, "value"] = np.nan
    frame = pd.concat([frame, future_revision, future_snapshot], ignore_index=True)

    context, audit = replay.select_observed_context_asof(
        frame,
        series_name=source_name,
        request_cutoff_utc=cutoff,
    )

    assert len(context) == replay.CONTEXT_LENGTH
    assert context.index[-1] == target_time
    assert context.iloc[-1] == baseline_value
    assert audit["imputed_hours"] == 6
    assert audit["maximum_selected_snapshot_time_utc"] <= cutoff
    assert audit["maximum_selected_revision_time_utc"] <= cutoff

    seven_hour_gap = _vintage_frame(cutoff)
    seven_hour_gap.loc[
        np.arange(len(seven_hour_gap) - 100, len(seven_hour_gap) - 93), "value"
    ] = np.nan
    with pytest.raises(replay.HistoricalResidualLoadError, match="gap de 7 heures"):
        replay.select_observed_context_asof(
            seven_hour_gap,
            series_name=source_name,
            request_cutoff_utc=cutoff,
        )


def test_replay_loads_one_pipeline_and_emits_five_complete_dst_days() -> None:
    end = pd.Timestamp("2025-03-30 07:00", tz="UTC")
    sources = _five_vintages(end)
    created: list[FakeChronosPipeline] = []

    def factory() -> FakeChronosPipeline:
        pipeline = FakeChronosPipeline()
        created.append(pipeline)
        return pipeline

    result = replay.replay_historical_residual_load(
        sources,
        start_day="2025-03-29",
        end_day="2025-03-30",
        pipeline_factory=factory,
        checkpoint_days=1,
    )

    assert len(created) == 1
    assert set(result.predictions["alias"]) == set(replay.EXPECTED_ALIASES)
    assert len(result.predictions) == 5 * (24 + 23)
    counts = result.predictions.groupby(["delivery_day_local", "alias"]).size()
    assert set(counts.loc["2025-03-29"]) == {24}
    assert set(counts.loc["2025-03-30"]) == {23}
    assert len(result.audits) == 10
    assert result.predictions["source_series"].str.endswith(".obs").all()
    assert (
        result.predictions["maximum_selected_snapshot_time_utc"]
        <= result.predictions["request_cutoff_utc"]
    ).all()
    assert (
        result.predictions["maximum_selected_revision_time_utc"]
        <= result.predictions["request_cutoff_utc"]
    ).all()
    assert (result.predictions["q10"] <= result.predictions["q50"]).all()
    assert (result.predictions["q50"] <= result.predictions["q90"]).all()


def test_replay_rejects_forecast_sources_and_unordered_quantiles() -> None:
    end = pd.Timestamp("2025-03-29 07:00", tz="UTC")
    sources = _five_vintages(end)
    observed_name = replay.COUNTRY_OBSERVED_SERIES["fr"]
    forbidden = dict(sources)
    forbidden[observed_name.replace(".obs", ".fcst")] = forbidden.pop(observed_name)
    with pytest.raises(replay.HistoricalResidualLoadError, match="seules les observations"):
        replay.replay_historical_residual_load(
            forbidden,
            start_day="2025-03-30",
            end_day="2025-03-30",
            pipeline=FakeChronosPipeline(),
        )

    with pytest.raises(replay.HistoricalResidualLoadError, match="non ordonnes"):
        replay.replay_historical_residual_load(
            sources,
            start_day="2025-03-30",
            end_day="2025-03-30",
            pipeline=FakeChronosPipeline(reversed_quantiles=True),
        )


def test_atomic_checkpoint_resumes_without_pipeline_and_converts_to_pit(tmp_path) -> None:
    end = pd.Timestamp("2025-03-29 07:00", tz="UTC")
    sources = _five_vintages(end)
    first = replay.replay_historical_residual_load(
        sources,
        start_day="2025-03-30",
        end_day="2025-03-30",
        pipeline=FakeChronosPipeline(),
        checkpoint_dir=tmp_path,
    )
    assert len(first.checkpoint_paths) == 1
    assert first.checkpoint_paths[0].is_file()
    assert first.checkpoint_paths[0].with_suffix(
        first.checkpoint_paths[0].suffix + ".manifest.json"
    ).is_file()

    def must_not_load() -> Any:
        raise AssertionError("Le checkpoint valide aurait du eviter le chargement.")

    resumed = replay.replay_historical_residual_load(
        sources,
        start_day="2025-03-30",
        end_day="2025-03-30",
        pipeline_factory=must_not_load,
        checkpoint_dir=tmp_path,
    )
    pd.testing.assert_frame_equal(first.predictions, resumed.predictions)

    changed_batch = replay.replay_historical_residual_load(
        sources,
        start_day="2025-03-30",
        end_day="2025-03-30",
        pipeline=FakeChronosPipeline(),
        batch_size=9,
        checkpoint_dir=tmp_path,
    )
    assert changed_batch.checkpoint_paths != first.checkpoint_paths

    pit = resumed.to_pit_frames()
    assert set(pit) == set(replay.EXPECTED_ALIASES)
    assert all(len(frame) == 23 for frame in pit.values())
    assert all(frame["value"].equals(frame["q50"]) for frame in pit.values())
    wide = resumed.to_wide()
    assert list(wide.columns) == list(replay.EXPECTED_ALIASES)
    assert len(wide) == 23
