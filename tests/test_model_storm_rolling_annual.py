"""Annual rolling scores preserve civil-day boundaries and honest coverage."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from chronos2_hourly import model_storm_rolling as rolling


def _annual_payload(day="2026-09-19", days=366):
    rows = []
    for offset in reversed(range(days)):
        current = date.fromisoformat(day) - timedelta(days=offset)
        # A distinct vector for each provider/day makes repeated scheduling visible.
        base = float(current.toordinal() * 100)
        for hour, stamp in enumerate(rolling._hours(current.isoformat(), "Europe/Paris")):
            observed = base + hour / 100
            rows.append({"timestamp_utc": stamp.isoformat(), "observed": observed,
                         "storm": observed + 1, "model": observed + 2,
                         "model_p10": observed - 8, "model_p90": observed + 12})
    return {"delivery_day": day, "zones": [{"zone": "FR", "timezone": "Europe/Paris",
                                            "rolling_history": {"rows": rows}}]}


@pytest.fixture
def schedules(monkeypatch):
    from chronos2_hourly import model_storm_bess as bess
    calls = []
    original = bess.simulate_bess_day

    def spy(forecast, observed, **kwargs):
        calls.append((kwargs["strategy"], tuple(forecast)))
        return original(forecast, observed, **kwargs)

    monkeypatch.setattr(bess, "simulate_bess_day", spy)
    monkeypatch.setattr(rolling, "optimize_storage_schedule",
                        lambda _: pytest.fail("Paper report must not call the legacy MILP"))
    return calls


def _zone(payload):
    return rolling.build_rolling_performance(payload)["zones"][0]


@pytest.mark.parametrize("published", [True, False])
def test_annual_bounds_and_full_coverage_with_or_without_current_observations(schedules, published):
    payload = _annual_payload()
    if not published:
        for row in payload["zones"][0]["rolling_history"]["rows"][-24:]:
            row["observed"] = None
    result = _zone(payload)
    annual = result["windows"]["365"]
    assert result["anchor_day"] == ("2026-09-19" if published else "2026-09-18")
    assert annual["start_day"] == ("2025-09-20" if published else "2025-09-19")
    assert annual["end_day"] == result["anchor_day"]
    assert annual["expected_hours"] == annual["paired_hours"] == 8760
    assert annual["observed_complete_days"] == annual["complete_paired_days"] == 365
    assert annual["frequencies"]["day"]["samples"] == 365
    assert annual["frequencies"]["60min"]["samples"] == 8760
    for provider in annual["frequencies"]["60min"]["providers"]:
        assert provider["mae"] == (1.0 if provider["key"] == "storm" else 2.0)
    # Accuracy retains the autumn repeated hour even when that day predates
    # calibration. The spring 23-hour day is eligible for both strategies.
    assert sum(len(plan) == 23 for _, plan in schedules) == 4
    assert annual["pnl_days"] == (306 if published else 305)


def test_annual_pnl_schedules_each_provider_day_once_for_every_window_and_frequency(schedules):
    result = _zone(_annual_payload())
    assert 365 in rolling.WINDOWS
    assert len(schedules) == len(set(schedules)) == 306 * 2 * 2
    for days in rolling.WINDOWS:
        section = result["windows"][str(days)]
        assert section["complete_paired_days"] == days
        assert section["pnl_days"] == min(days, 306)
        assert section["pnl_excluded_days"] == max(days - 306, 0)
        for frequency in ("60min", "day"):
            assert all(provider["pnl_days"] == min(days, 306) and provider["daily_pnl"] == 0
                       for provider in section["frequencies"][frequency]["providers"])


@pytest.mark.parametrize("missing", ["model", "storm"])
def test_annual_missing_hour_retains_paired_hour_coverage_but_excludes_incomplete_day(schedules, missing):
    payload = _annual_payload()
    rows = payload["zones"][0]["rolling_history"]["rows"]
    # Only one of the two autumn 02:00 hours is missing. The physical UTC hour
    # is not substituted by its identically labelled local neighbour.
    stamp = pd.Timestamp("2025-10-26T01:00:00Z")
    row = next(row for row in rows if pd.Timestamp(row["timestamp_utc"]) == stamp)
    row[missing] = None
    annual = _zone(payload)["windows"]["365"]
    assert annual["expected_hours"] == 8760
    assert annual["paired_hours"] == 8759
    assert annual["observed_complete_days"] == 365
    assert annual["complete_paired_days"] == 364
    # The missing hour falls inside the initial calibration period. A missing
    # Storm day delays calibration; a missing NYX day does not train Storm.
    eligible = 305 if missing == "storm" else 306
    assert annual["pnl_days"] == eligible
    assert annual["frequencies"]["day"]["samples"] == 364
    assert all(provider["samples"] == 8759 for provider in annual["frequencies"]["60min"]["providers"])
    assert len(schedules) == eligible * 2 * 2


def test_annual_short_source_is_not_stretched_or_imputed(schedules):
    annual = _zone(_annual_payload(days=90))["windows"]["365"]
    assert annual["start_day"] == "2025-09-20"
    assert annual["expected_hours"] == 8760
    assert annual["paired_hours"] == 90 * 24
    assert annual["observed_complete_days"] == annual["complete_paired_days"] == 90
    assert annual["frequencies"]["day"]["samples"] == 90
    assert annual["pnl_days"] == 30
    assert len(schedules) == 30 * 2 * 2
