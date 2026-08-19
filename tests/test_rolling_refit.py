from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.rolling_refit import (
    RollingRefitBlock,
    RollingRefitContractError,
    RollingRefitPolicy,
    select_rolling_refit_window,
)


ORIGIN_TIMEZONE = "Europe/Paris"


def _complete_index(
    start: str | date,
    days: int,
    *,
    timezone: str,
) -> pd.DatetimeIndex:
    values = pd.date_range(pd.Timestamp(start), periods=days, freq="D")
    result = local_delivery_day_index(values[0].date(), timezone=timezone)
    for value in values[1:]:
        result = result.append(
            local_delivery_day_index(value.date(), timezone=timezone)
        )
    return pd.DatetimeIndex(
        result.tz_convert("UTC"),
        name="delivery_start_utc",
    )


def _origins(
    index: pd.DatetimeIndex,
    *,
    delivery_timezone: str,
) -> pd.Series:
    local_days = pd.Index(index.tz_convert(delivery_timezone).date)
    cache: dict[date, pd.Timestamp] = {}
    values: list[pd.Timestamp] = []
    for day in local_days:
        if day not in cache:
            cache[day] = (
                pd.Timestamp(day)
                - pd.Timedelta(days=1)
                + pd.Timedelta(hours=8)
            ).tz_localize(ORIGIN_TIMEZONE).tz_convert("UTC")
        values.append(cache[day])
    return pd.Series(values, index=index, name="forecast_origin_utc")


def _block(
    index: pd.DatetimeIndex,
    *,
    delivery_timezone: str,
    kind: str = "sealed_oof",
    source_id: str = "sealed",
    digest_character: str = "a",
    feature_shift: float = 0.0,
) -> RollingRefitBlock:
    position = np.arange(len(index), dtype=float)
    origins = _origins(index, delivery_timezone=delivery_timezone)
    q50 = 50.0 + np.sin(position / 24.0)
    return RollingRefitBlock(
        source_kind=kind,
        source_id=source_id,
        source_sha256=digest_character * 64,
        features=pd.DataFrame(
            {
                "safe_feature": position + feature_shift,
                "known_load": 40.0 + np.cos(position / 48.0),
            },
            index=index,
        ),
        target=pd.Series(q50 + 1.5, index=index, name="actual"),
        chronos_quantiles=pd.DataFrame(
            {"q10": q50 - 5.0, "q50": q50, "q90": q50 + 5.0},
            index=index,
        ),
        forecast_origin_utc=origins,
        pit_inputs_present=pd.Series(
            True,
            index=index,
            name="pit_inputs_present",
            dtype=bool,
        ),
        maximum_snapshot_time_utc=origins - pd.Timedelta(hours=2),
        maximum_revision_time_utc=origins - pd.Timedelta(minutes=5),
    )


def _replace_block(
    block: RollingRefitBlock,
    **changes,
) -> RollingRefitBlock:
    values = {
        "source_kind": block.source_kind,
        "source_id": block.source_id,
        "source_sha256": block.source_sha256,
        "features": block.features,
        "target": block.target,
        "chronos_quantiles": block.chronos_quantiles,
        "forecast_origin_utc": block.forecast_origin_utc,
        "pit_inputs_present": block.pit_inputs_present,
        "maximum_snapshot_time_utc": block.maximum_snapshot_time_utc,
        "maximum_revision_time_utc": block.maximum_revision_time_utc,
    }
    values.update(changes)
    return RollingRefitBlock(**values)


def test_exact_365_day_window_merges_sealed_replay_and_live_sources() -> None:
    timezone = "Europe/Paris"
    forecast_day = date(2026, 8, 16)
    full = _complete_index("2025-08-01", 380, timezone=timezone)
    local_days = pd.Index(full.tz_convert(timezone).date)
    sealed_index = full[local_days <= date(2026, 8, 11)]
    replay_index = full[
        (local_days >= date(2026, 8, 12))
        & (local_days <= date(2026, 8, 14))
    ]
    live_index = full[local_days == date(2026, 8, 15)]
    blocks = [
        _block(sealed_index, delivery_timezone=timezone),
        _block(
            replay_index,
            delivery_timezone=timezone,
            kind="pit_replay",
            source_id="replays-12-14",
            digest_character="b",
            feature_shift=float(len(sealed_index)),
        ),
        _block(
            live_index,
            delivery_timezone=timezone,
            kind="issued_live",
            source_id="live-15",
            digest_character="c",
            feature_shift=float(len(sealed_index) + len(replay_index)),
        ),
    ]

    selected = select_rolling_refit_window(
        blocks,
        forecast_delivery_day=forecast_day,
        delivery_timezone=timezone,
    )

    expected = _complete_index("2025-08-16", 365, timezone=timezone)
    assert selected.X.index.equals(expected)
    assert selected.y.index.equals(expected)
    assert selected.base.index.equals(expected)
    assert selected.experts.index.equals(expected)
    assert list(selected.experts) == [
        "chronos2__q10",
        "chronos2__q50",
        "chronos2__q90",
    ]
    assert selected.audit["window_start_day_local"] == "2025-08-16"
    assert selected.audit["window_end_day_local"] == "2026-08-15"
    assert selected.audit["training_complete_days"] == 365
    assert selected.audit["training_rows"] == len(expected)
    assert selected.audit["source_kinds"]["pit_replay"] == len(replay_index)
    assert selected.audit["source_kinds"]["issued_live"] == len(live_index)
    assert selected.audit["storm_used_as_feature"] is False
    assert selected.audit["mkonline_used_as_feature"] is False
    assert selected.audit["chronos_pretrained_weights_refit"] is False
    assert selected.audit["residual_corrector_refit"] is True


@pytest.mark.parametrize(
    ("forecast_day", "expected_hours", "expected_histogram"),
    [
        (date(2026, 3, 31), 71, {"23": 1, "24": 2}),
        (date(2026, 10, 27), 73, {"24": 2, "25": 1}),
    ],
)
@pytest.mark.parametrize(
    "timezone",
    [
        "Europe/Paris",
        "Europe/Berlin",
        "Europe/Brussels",
        "Europe/Amsterdam",
        "Europe/Madrid",
    ],
)
def test_window_uses_complete_local_days_across_dst(
    forecast_day: date,
    expected_hours: int,
    expected_histogram: dict[str, int],
    timezone: str,
) -> None:
    start = forecast_day - pd.Timedelta(days=3)
    index = _complete_index(start, 3, timezone=timezone)
    selected = select_rolling_refit_window(
        [_block(index, delivery_timezone=timezone)],
        forecast_delivery_day=forecast_day,
        delivery_timezone=timezone,
        policy=RollingRefitPolicy(window_days=3),
    )
    assert len(selected.X) == expected_hours
    assert selected.audit["physical_day_hour_histogram"] == expected_histogram


@pytest.mark.parametrize(
    "forecast_day",
    [date(2026, 3, 29), date(2026, 10, 25)],
)
def test_full_year_window_never_assumes_365_times_24(
    forecast_day: date,
) -> None:
    timezone = "Europe/Paris"
    index = _complete_index(
        forecast_day - pd.Timedelta(days=365),
        365,
        timezone=timezone,
    )
    selected = select_rolling_refit_window(
        [_block(index, delivery_timezone=timezone)],
        forecast_delivery_day=forecast_day,
        delivery_timezone=timezone,
    )
    assert selected.X.index.equals(index)
    assert selected.audit["training_complete_days"] == 365
    assert selected.audit["training_rows"] == len(index)
    assert set(selected.audit["physical_day_hour_histogram"]).issubset(
        {"23", "24", "25"}
    )


def test_missing_hour_and_insufficient_history_fail_closed() -> None:
    timezone = "Europe/Paris"
    forecast_day = date(2026, 4, 1)
    complete = _complete_index("2026-03-29", 3, timezone=timezone)
    missing = complete.delete(10)
    with pytest.raises(RollingRefitContractError, match="missing=1"):
        select_rolling_refit_window(
            [_block(missing, delivery_timezone=timezone)],
            forecast_delivery_day=forecast_day,
            delivery_timezone=timezone,
            policy=RollingRefitPolicy(window_days=3),
        )

    short = _complete_index("2025-04-02", 364, timezone=timezone)
    with pytest.raises(RollingRefitContractError, match="missing"):
        select_rolling_refit_window(
            [_block(short, delivery_timezone=timezone)],
            forecast_delivery_day=forecast_day,
            delivery_timezone=timezone,
        )


def test_duplicate_timestamp_inside_one_source_is_rejected() -> None:
    timezone = "Europe/Paris"
    index = _complete_index("2026-01-01", 2, timezone=timezone)
    duplicate = index.insert(1, index[0])
    with pytest.raises(RollingRefitContractError, match="duplicate"):
        select_rolling_refit_window(
            [_block(duplicate, delivery_timezone=timezone)],
            forecast_delivery_day=date(2026, 1, 3),
            delivery_timezone=timezone,
            policy=RollingRefitPolicy(window_days=2),
        )


def test_equivalent_overlap_is_deterministic_and_sealed_oof_wins() -> None:
    timezone = "Europe/Paris"
    index = _complete_index("2026-01-01", 2, timezone=timezone)
    sealed = _block(index, delivery_timezone=timezone)
    live = _replace_block(
        sealed,
        source_kind="issued_live",
        source_id="live",
        source_sha256="b" * 64,
    )
    forward = select_rolling_refit_window(
        [sealed, live],
        forecast_delivery_day=date(2026, 1, 3),
        delivery_timezone=timezone,
        policy=RollingRefitPolicy(window_days=2),
    )
    reverse = select_rolling_refit_window(
        [live, sealed],
        forecast_delivery_day=date(2026, 1, 3),
        delivery_timezone=timezone,
        policy=RollingRefitPolicy(window_days=2),
    )
    assert set(forward.provenance["source_kind"]) == {"sealed_oof"}
    assert forward.audit["equivalent_duplicate_rows_resolved"] == len(index)
    assert (
        forward.audit["training_corpus_sha256"]
        == reverse.audit["training_corpus_sha256"]
    )
    pd.testing.assert_frame_equal(forward.X, reverse.X)
    pd.testing.assert_frame_equal(forward.provenance, reverse.provenance)


def test_conflicting_overlap_fails_closed() -> None:
    timezone = "Europe/Paris"
    index = _complete_index("2026-01-01", 1, timezone=timezone)
    sealed = _block(index, delivery_timezone=timezone)
    conflict = _block(
        index,
        delivery_timezone=timezone,
        kind="pit_replay",
        source_id="conflict",
        digest_character="b",
        feature_shift=1.0,
    )
    with pytest.raises(RollingRefitContractError, match="conflicting immutable"):
        select_rolling_refit_window(
            [sealed, conflict],
            forecast_delivery_day=date(2026, 1, 2),
            delivery_timezone=timezone,
            policy=RollingRefitPolicy(window_days=1),
        )


def test_future_labels_are_excluded_and_cannot_change_selected_corpus() -> None:
    timezone = "Europe/Paris"
    full = _complete_index("2026-01-01", 4, timezone=timezone)
    block = _block(full, delivery_timezone=timezone)
    forecast_day = date(2026, 1, 4)
    first = select_rolling_refit_window(
        [block],
        forecast_delivery_day=forecast_day,
        delivery_timezone=timezone,
        policy=RollingRefitPolicy(window_days=3),
    )
    changed_target = block.target.copy()
    future = pd.Index(full.tz_convert(timezone).date) >= forecast_day
    changed_target.loc[future] = 1_000_000.0
    second = select_rolling_refit_window(
        [_replace_block(block, target=changed_target)],
        forecast_delivery_day=forecast_day,
        delivery_timezone=timezone,
        policy=RollingRefitPolicy(window_days=3),
    )
    pd.testing.assert_series_equal(first.y, second.y)
    assert (
        first.audit["training_corpus_sha256"]
        == second.audit["training_corpus_sha256"]
    )
    assert max(first.y.index.tz_convert(timezone).date) == date(2026, 1, 3)


def test_training_corpus_hash_commits_causal_snapshot_and_revision_times() -> None:
    timezone = "Europe/Paris"
    index = _complete_index("2026-01-01", 2, timezone=timezone)
    block = _block(index, delivery_timezone=timezone)
    baseline = select_rolling_refit_window(
        [block],
        forecast_delivery_day=date(2026, 1, 3),
        delivery_timezone=timezone,
        policy=RollingRefitPolicy(window_days=2),
    )
    changed_revision = block.maximum_revision_time_utc.copy()
    changed_revision.iloc[0] -= pd.Timedelta(seconds=1)
    changed = select_rolling_refit_window(
        [_replace_block(block, maximum_revision_time_utc=changed_revision)],
        forecast_delivery_day=date(2026, 1, 3),
        delivery_timezone=timezone,
        policy=RollingRefitPolicy(window_days=2),
    )

    assert baseline.audit["causal_timestamps_in_training_corpus_hash"] is True
    assert (
        baseline.audit["training_corpus_sha256"]
        != changed.audit["training_corpus_sha256"]
    )


def test_missing_pit_evidence_is_allowed_only_when_mask_is_false() -> None:
    timezone = "Europe/Paris"
    index = _complete_index("2026-01-01", 1, timezone=timezone)
    block = _block(index, delivery_timezone=timezone)
    pit_inputs_present = block.pit_inputs_present.copy()
    pit_inputs_present.iloc[-2:] = False
    snapshot = block.maximum_snapshot_time_utc.copy()
    revision = block.maximum_revision_time_utc.copy()
    snapshot.iloc[-2:] = pd.NaT
    revision.iloc[-2:] = pd.NaT
    selection = select_rolling_refit_window(
        [
            _replace_block(
                block,
                pit_inputs_present=pit_inputs_present,
                maximum_snapshot_time_utc=snapshot,
                maximum_revision_time_utc=revision,
            )
        ],
        forecast_delivery_day=date(2026, 1, 2),
        delivery_timezone=timezone,
        policy=RollingRefitPolicy(window_days=1),
    )

    assert selection.audit["pit_input_present_hours"] == 22
    assert selection.audit["pit_input_missing_hours"] == 2
    assert selection.pit_inputs_present.iloc[-2:].eq(False).all()


@pytest.mark.parametrize("field", ["maximum_snapshot_time_utc", "maximum_revision_time_utc"])
def test_pit_evidence_presence_must_match_mask(field: str) -> None:
    timezone = "Europe/Paris"
    index = _complete_index("2026-01-01", 1, timezone=timezone)
    block = _block(index, delivery_timezone=timezone)
    timestamp = getattr(block, field).copy()
    timestamp.iloc[0] = pd.NaT
    with pytest.raises(RollingRefitContractError, match="present exactly"):
        select_rolling_refit_window(
            [_replace_block(block, **{field: timestamp})],
            forecast_delivery_day=date(2026, 1, 2),
            delivery_timezone=timezone,
            policy=RollingRefitPolicy(window_days=1),
        )


def test_origin_must_equal_own_civil_d_minus_one_eight_cutoff() -> None:
    timezone = "Europe/Paris"
    index = _complete_index("2026-03-29", 1, timezone=timezone)
    block = _block(index, delivery_timezone=timezone)
    bad_origin = block.forecast_origin_utc.copy()
    bad_origin.iloc[0] += pd.Timedelta(minutes=1)
    with pytest.raises(RollingRefitContractError, match="raw Chronos origins"):
        select_rolling_refit_window(
            [_replace_block(block, forecast_origin_utc=bad_origin)],
            forecast_delivery_day=date(2026, 3, 30),
            delivery_timezone=timezone,
            policy=RollingRefitPolicy(window_days=1),
        )


@pytest.mark.parametrize(
    "field",
    ["maximum_snapshot_time_utc", "maximum_revision_time_utc"],
)
def test_covariate_pit_markers_cannot_exceed_own_origin(field: str) -> None:
    timezone = "Europe/Paris"
    index = _complete_index("2026-01-01", 1, timezone=timezone)
    block = _block(index, delivery_timezone=timezone)
    bad = getattr(block, field).copy()
    bad.iloc[-1] = block.forecast_origin_utc.iloc[-1] + pd.Timedelta(seconds=1)
    with pytest.raises(RollingRefitContractError, match="snapshot/revision"):
        select_rolling_refit_window(
            [_replace_block(block, **{field: bad})],
            forecast_delivery_day=date(2026, 1, 2),
            delivery_timezone=timezone,
            policy=RollingRefitPolicy(window_days=1),
        )


@pytest.mark.parametrize("column", ["storm_dashboard__q50", "mkonline__q50"])
def test_external_forecast_tokens_are_rejected_from_features(column: str) -> None:
    timezone = "Europe/Paris"
    index = _complete_index("2026-01-01", 1, timezone=timezone)
    block = _block(index, delivery_timezone=timezone)
    unsafe = block.features.assign(**{column: 1.0})
    with pytest.raises(RollingRefitContractError, match="forbidden"):
        select_rolling_refit_window(
            [_replace_block(block, features=unsafe)],
            forecast_delivery_day=date(2026, 1, 2),
            delivery_timezone=timezone,
            policy=RollingRefitPolicy(window_days=1),
        )


@pytest.mark.parametrize("mutation", ["target", "quantile", "feature"])
def test_invalid_numeric_inputs_fail_closed(mutation: str) -> None:
    timezone = "Europe/Paris"
    index = _complete_index("2026-01-01", 1, timezone=timezone)
    block = _block(index, delivery_timezone=timezone)
    if mutation == "target":
        value = block.target.copy()
        value.iloc[0] = np.nan
        block = _replace_block(block, target=value)
    elif mutation == "quantile":
        value = block.chronos_quantiles.copy()
        value.iloc[0, 1] = np.inf
        block = _replace_block(block, chronos_quantiles=value)
    else:
        value = block.features.copy()
        value.iloc[0, 0] = np.inf
        block = _replace_block(block, features=value)
    with pytest.raises(RollingRefitContractError, match="non-finite|infinite"):
        select_rolling_refit_window(
            [block],
            forecast_delivery_day=date(2026, 1, 2),
            delivery_timezone=timezone,
            policy=RollingRefitPolicy(window_days=1),
        )


def test_declared_missing_features_are_preserved_for_model_imputation() -> None:
    timezone = "Europe/Paris"
    index = _complete_index("2026-01-01", 1, timezone=timezone)
    block = _block(index, delivery_timezone=timezone)
    features = block.features.copy()
    features.iloc[0, 0] = np.nan
    selected = select_rolling_refit_window(
        [_replace_block(block, features=features)],
        forecast_delivery_day=date(2026, 1, 2),
        delivery_timezone=timezone,
        policy=RollingRefitPolicy(window_days=1),
    )
    assert np.isnan(selected.X.iloc[0, 0])


def test_one_delivery_day_cannot_be_split_across_sources() -> None:
    timezone = "Europe/Paris"
    index = _complete_index("2026-01-01", 1, timezone=timezone)
    first = index[:12]
    second = index[12:]
    with pytest.raises(RollingRefitContractError, match="split across"):
        select_rolling_refit_window(
            [
                _block(first, delivery_timezone=timezone),
                _block(
                    second,
                    delivery_timezone=timezone,
                    kind="pit_replay",
                    source_id="replay",
                    digest_character="b",
                    feature_shift=float(len(first)),
                ),
            ],
            forecast_delivery_day=date(2026, 1, 2),
            delivery_timezone=timezone,
            policy=RollingRefitPolicy(window_days=1),
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"source_kind": "legacy"}, "source_kind"),
        ({"source_sha256": "not-a-sha"}, "SHA-256"),
    ],
)
def test_source_identity_is_strict(change: dict, message: str) -> None:
    timezone = "Europe/Paris"
    index = _complete_index("2026-01-01", 1, timezone=timezone)
    block = _replace_block(
        _block(index, delivery_timezone=timezone),
        **change,
    )
    with pytest.raises(RollingRefitContractError, match=message):
        select_rolling_refit_window(
            [block],
            forecast_delivery_day=date(2026, 1, 2),
            delivery_timezone=timezone,
            policy=RollingRefitPolicy(window_days=1),
        )


@pytest.mark.parametrize("window_days", [0, -1, True, 365.0])
def test_policy_rejects_invalid_window_days(window_days) -> None:
    with pytest.raises(RollingRefitContractError, match="window_days"):
        RollingRefitPolicy(window_days=window_days)
