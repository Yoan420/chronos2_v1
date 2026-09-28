"""The future auction source is bounded, complete and not a model bundle."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_nyx_annual_auction_prices_source as source
from chronos2_hourly.nyx_annual_live_preflight import (
    delivery_grid, inspect_bundle,
)


DAY = "2026-09-29"


def _capture(monkeypatch: pytest.MonkeyPatch, day: str = DAY,
             missing_zone: str | None = None):
    expected, cutoff = source.price_grid(day)
    calls = []

    def fake_fetch(client, series_name, start, end, timezone, **kwargs):
        zone = next(zone for zone, name in source.SERIES.items()
                    if name == series_name)
        calls.append((zone, start, end, timezone, kwargs))
        index = expected[1:] if zone == missing_zone else expected
        # The Saturn response may also include the future delivery hour;
        # collect() must discard it rather than treating it as a label.
        index = index.append(pd.DatetimeIndex([expected[-1] + pd.Timedelta(hours=1)]))
        return pd.Series(np.arange(len(index), dtype=float), index=index)

    monkeypatch.setattr(source, "fetch_saturn_series_from_client", fake_fetch)
    return expected, cutoff, calls


def test_collect_and_publish_four_complete_price_histories(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    expected, cutoff, calls = _capture(monkeypatch)
    values, evidence = source.collect(object(), DAY, now_utc=cutoff)
    assert len(calls) == 4
    assert [call[0] for call in calls] == list(source.SERIES)
    assert all(call[1] == expected[0] and call[2] == expected[-1]
               and call[3] == "UTC" and call[4]["revision_date"] == cutoff
               and call[4]["naive_timezone"] == "UTC" for call in calls)
    assert all(frame.index[-1] == expected[-1] for frame in values.values())
    assert all(expected[-1] + pd.Timedelta(hours=1) not in frame.index
               for frame in values.values())
    plan = {"saturn_url": "https://example.invalid/api",
            "config_sha256": {name: "a" * 64 for name in source.ZONE_CONFIGS.values()}}
    bundle = tmp_path / "bundle"
    path = source.publish(bundle, DAY, values, evidence, plan, now_utc=cutoff)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["source_group"] == "auction_prices"
    assert receipt["asof_state_utc"] == cutoff.isoformat()
    assert receipt["provider_publication_timestamp_verified"] is False
    assert receipt["model_inputs_complete"] is False
    assert receipt["hours_per_zone"] == len(expected)
    assert len(receipt["artifact_sha256"]) == 4
    assert source.publish(bundle, DAY, values, evidence, plan,
                          now_utc=cutoff + pd.Timedelta(hours=1)) == path
    checks = inspect_bundle(bundle, DAY)["checks"]
    assert next(check for check in checks if check["input"] ==
                "source/auction_prices")["passed"] is True
    assert any(not check["passed"] for check in checks)


def test_collect_rejects_early_or_missing_price_hour(monkeypatch: pytest.MonkeyPatch):
    _, cutoff, _ = _capture(monkeypatch, missing_zone="BE")
    with pytest.raises(ValueError, match="cutoff has not occurred"):
        source.collect(object(), DAY, now_utc=cutoff - pd.Timedelta(seconds=1))
    with pytest.raises(ValueError, match="BE: 1 auction price hours missing"):
        source.collect(object(), DAY, now_utc=cutoff)


def test_existing_source_artifact_cannot_be_silently_replaced(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _, cutoff, _ = _capture(monkeypatch)
    values, evidence = source.collect(object(), DAY, now_utc=cutoff)
    plan = {"saturn_url": "https://example.invalid/api",
            "config_sha256": {name: "a" * 64 for name in source.ZONE_CONFIGS.values()}}
    bundle = tmp_path / "bundle"
    source.publish(bundle, DAY, values, evidence, plan, now_utc=cutoff)
    changed = values.copy()
    changed["FR"] = changed["FR"].copy()
    changed["FR"].iloc[0, 0] += 1.
    with pytest.raises(ValueError, match="Existing auction artifact differs"):
        source.publish(bundle, DAY, changed, evidence, plan, now_utc=cutoff)


def test_source_grid_preserves_physical_dst_hours():
    for day in ("2026-03-30", "2026-10-26"):
        expected, cutoff = source.price_grid(day)
        _, current, expected_cutoff = delivery_grid(day)
        assert cutoff == expected_cutoff
        assert expected.tz is not None and str(expected.tz) == "UTC"
        assert expected.is_unique and expected.is_monotonic_increasing
        assert expected[-1] + pd.Timedelta(hours=1) == current[0]
        assert len(current) == 24


def test_checked_in_target_series_are_canonical():
    plan = source.load_plan()
    assert plan["series"] == source.SERIES
    assert len(plan["config_sha256"]) == 4
