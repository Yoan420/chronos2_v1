from __future__ import annotations

import pandas as pd
import pytest

from build_da_cap_system_asymmetry import (
    COMPONENTS,
    derive_asymmetry_vintages,
)


def _events(alias: str, values: list[tuple[str, str, float]]):
    return pd.DataFrame(
        {
            "value_time_utc": pd.to_datetime(
                [x[0] for x in values], utc=True
            ),
            "event_time_utc": pd.to_datetime(
                [x[1] for x in values], utc=True
            ),
            "alias": alias,
            "component_value_mw": [x[2] for x in values],
        }
    )


def test_asymmetry_replays_only_known_revisions():
    delivery = "2026-01-02T12:00:00Z"
    initial = "2026-01-01T06:00:00Z"
    update = "2026-01-01T07:00:00Z"

    base = {
        "da_cap_es_fr": 2000.0,
        "da_cap_it_north_fr": 3000.0,
        "da_cap_uk_fr": 1000.0,
        "da_cap_fr_it_north": 2500.0,
        "da_cap_fr_uk": 1000.0,
    }

    frames = {
        alias: _events(alias, [(delivery, initial, value)])
        for alias, value in base.items()
    }
    frames["da_cap_it_north_fr"] = _events(
        "da_cap_it_north_fr",
        [
            (delivery, initial, 3000.0),
            (delivery, update, 3500.0),
        ],
    )

    result = derive_asymmetry_vintages(frames, output_unit="GW")

    assert result["value"].tolist() == pytest.approx([2.5, 3.0])
    assert result["snapshot_time_utc"].tolist() == list(
        pd.to_datetime([initial, update], utc=True)
    )


def test_missing_component_is_rejected():
    delivery = "2026-01-02T12:00:00Z"
    initial = "2026-01-01T06:00:00Z"
    frames = {
        alias: _events(alias, [(delivery, initial, 1000.0)])
        for alias in list(COMPONENTS)[:-1]
    }

    with pytest.raises(KeyError):
        derive_asymmetry_vintages(frames)
