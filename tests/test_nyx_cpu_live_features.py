from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.nyx_cpu_live_features import (
    FEATURE_COLUMNS, RESIDUAL_COLUMNS, ZONES, build_country_features,
)


def fixture(first="2026-03-20", stop="2026-04-05"):
    index = pd.date_range(first, stop, tz="Europe/Paris", freq="h",
                          inclusive="left").tz_convert("UTC")
    local_days = index.tz_convert("Europe/Paris").date
    cutoffs = pd.DatetimeIndex([
        pd.Timestamp(f"{day - timedelta(days=1)} 08:00", tz="Europe/Paris")
        .tz_convert("UTC") for day in local_days
    ])
    bank = pd.DataFrame({"value_time_utc": index, "snapshot_time_utc": cutoffs,
                         "revision_time_utc": cutoffs, "cutoff_time_utc": cutoffs})
    for number, column in enumerate(RESIDUAL_COLUMNS):
        bank[column] = 10. + number + np.arange(len(index)) / 1000.
    nuclear = pd.DataFrame({"value_time_utc": index, "snapshot_time_utc": cutoffs,
                            "revision_time_utc": cutoffs,
                            "value": 40. + np.arange(len(index)) / 1000.})
    history = pd.date_range(
        str(pd.Timestamp(first).date() - timedelta(days=7)), stop,
        tz="Europe/Paris", freq="h", inclusive="left",
    ).tz_convert("UTC")
    prices = {zone: pd.Series(25. * number + np.arange(len(history), dtype=float),
                              index=history, name=f"{zone}_actual")
              for number, zone in enumerate(ZONES)}
    return index, prices, bank, nuclear


def make(index, prices, bank, nuclear, zone="FR"):
    return build_country_features(zone=zone, delivery_index=index, prices=prices,
                                  residual_bank=bank, nuclear_store=nuclear)


def test_schema_causal_future_prices_and_spring_dst():
    index, prices, bank, nuclear = fixture()
    frame, audit = make(index, prices, bank, nuclear)
    assert len(FEATURE_COLUMNS) == 77
    assert tuple(frame.columns) == FEATURE_COLUMNS
    assert frame.index.equals(index)
    assert audit["all_saturn_hours_present_and_asof_cutoff"] is True
    # 29 March 2026 has no civil 02:00, so next day's D-1 same-hour lag
    # is genuinely absent and explicitly marked rather than filled.
    target = pd.Timestamp("2026-03-30 02:00", tz="Europe/Paris").tz_convert("UTC")
    assert np.isnan(frame.loc[target, "price_fr_d1_hour"])
    assert frame.loc[target, "price_fr_d1_hour__available"] == 0.
    assert np.isfinite(frame.loc[target, "price_fr_d1_mean"])
    changed = dict(prices)
    changed["FR"] = prices["FR"].copy()
    future = changed["FR"].index.tz_convert("Europe/Paris").date == pd.Timestamp("2026-04-04").date()
    changed["FR"].loc[future] += 1e6
    repeated, _ = make(index, changed, bank, nuclear)
    pd.testing.assert_frame_equal(frame, repeated, check_exact=True)


def test_fall_fold_is_averaged_for_same_civil_hour():
    index, prices, bank, nuclear = fixture("2026-10-20", "2026-11-03")
    frame, _ = make(index, prices, bank, nuclear)
    source_day = pd.Timestamp("2026-10-25").date()
    local = prices["FR"].index.tz_convert("Europe/Paris")
    actual = prices["FR"].loc[(local.date == source_day) & (local.hour == 2)].mean()
    target = pd.Timestamp("2026-10-26 02:00", tz="Europe/Paris").tz_convert("UTC")
    assert frame.loc[target, "price_fr_d1_hour"] == actual
    assert frame.loc[target, "price_fr_d1_hour__available"] == 1.


def test_rejects_missing_or_late_saturn_and_price_history():
    index, prices, bank, nuclear = fixture()
    late = bank.copy()
    late.loc[0, "revision_time_utc"] = late.loc[0, "cutoff_time_utc"] + pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="post-cutoff"):
        make(index, prices, late, nuclear)
    with pytest.raises(ValueError, match="required historical or future hour"):
        make(index, prices, bank.drop(index=0).reset_index(drop=True), nuclear)
    missing = dict(prices)
    missing["BE"] = prices["BE"].iloc[1:]
    with pytest.raises(ValueError, match="incomplete D-1 through D-7"):
        make(index, missing, bank, nuclear)
