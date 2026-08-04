from __future__ import annotations

import numpy as np
import pandas as pd

from build_extended_exogenous_inputs import build_uncertainty_metrics
from chronos2_modular.exogenous_extensions import rich_calendar_frame


def test_christmas_is_holiday() -> None:
    index = pd.date_range(
        "2026-12-24", periods=72, freq="h", tz="Europe/Paris"
    )
    frame = rich_calendar_frame(
        index,
        countries=["FR", "DE", "BE", "NL", "ES"],
        primary_country="FR",
    )
    day = pd.Timestamp("2026-12-25", tz="Europe/Paris")
    assert frame.loc[
        frame.index.normalize() == day,
        "known_cal_holiday_fr_oracle",
    ].eq(1.0).all()


def test_uncertainty_uses_only_revisions_before_cutoff() -> None:
    delivery = pd.Timestamp("2026-08-05 18:00", tz="UTC")
    frame = pd.DataFrame(
        {
            "value_time_utc": [delivery] * 3,
            "snapshot_time_utc": pd.to_datetime(
                [
                    "2026-08-04 04:00Z",
                    "2026-08-04 05:00Z",
                    "2026-08-04 08:00Z",
                ],
                utc=True,
            ),
            "revision_time_utc": pd.to_datetime(
                [
                    "2026-08-04 04:00Z",
                    "2026-08-04 05:00Z",
                    "2026-08-04 08:00Z",
                ],
                utc=True,
            ),
            "value": [40.0, 42.0, 1000.0],
        }
    )
    result = build_uncertainty_metrics(
        frame,
        timezone="Europe/Paris",
        config={"data": {"forecast_origin_local_time": "08:00"}},
        max_revisions=6,
    )
    # En août, 08:00 Europe/Paris = 06:00 UTC.
    assert len(result) == 1
    assert np.isclose(result.iloc[0]["revision_abs_delta"], 2.0)
    assert np.isclose(result.iloc[0]["revision_std"], 1.0)


def test_single_revision_is_not_missing() -> None:
    delivery = pd.Timestamp("2026-08-05 10:00", tz="UTC")
    frame = pd.DataFrame(
        {
            "value_time_utc": [delivery],
            "snapshot_time_utc": pd.to_datetime(
                ["2026-08-04 05:00Z"], utc=True
            ),
            "revision_time_utc": pd.to_datetime(
                ["2026-08-04 05:00Z"], utc=True
            ),
            "value": [50.0],
        }
    )
    result = build_uncertainty_metrics(
        frame,
        timezone="Europe/Paris",
        config={"data": {"forecast_origin_local_time": "08:00"}},
        max_revisions=6,
    )
    assert result.iloc[0]["revision_std"] == 0.0
    assert result.iloc[0]["revision_abs_delta"] == 0.0
    assert result.iloc[0]["revision_count"] == 1
