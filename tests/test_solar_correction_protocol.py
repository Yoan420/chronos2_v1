"""Preregistration and strict civil-time evidence tests; entirely synthetic."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from chronos2_hourly import solar_correction_protocol as protocol


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "_utc_now", lambda: datetime(2026, 9, 20, 19, 6, tzinfo=timezone.utc))
    return protocol.create_or_load_protocol(tmp_path, recipe_contract={"code_sha256": "a" * 64, "config": {"alpha": 1}})


def records(frozen, day="2026-09-22", sealed_at="2026-09-21T05:59:00+00:00"):
    return [{"delivery_date": day, "zone": zone, "candidate": candidate, "complete": True,
             "sealed_at": sealed_at, "protocol_sha256": frozen["protocol_sha256"],
             "forecast_sha256": "f" * 64}
            for zone in protocol.ZONES for candidate in protocol.CANDIDATES]


def test_freeze_wallclock_is_immutable_and_first_delivery_is_22(tmp_path, frozen):
    path = tmp_path / "solar_correction_protocol.json"
    original = path.read_bytes()
    loaded = protocol.create_or_load_protocol(tmp_path, recipe_contract=frozen["recipe_contract"])
    assert loaded == frozen
    assert path.read_bytes() == original
    assert frozen["first_prospective_delivery"] == "2026-09-22"
    assert frozen["frozen_at_utc"] == "2026-09-20T19:06:00+00:00"
    assert frozen["country_specific_selection"] is frozen["promotion_allowed"] is False
    with pytest.raises(ValueError, match="frozen"):
        protocol.create_or_load_protocol(tmp_path, recipe_contract={"changed": True})
    assert path.read_bytes() == original


def test_tampering_is_rejected(tmp_path, frozen):
    path = tmp_path / "solar_correction_protocol.json"
    frozen["frozen_at_utc"] = "2025-01-01T00:00:00+00:00"
    path.write_text(json.dumps(frozen), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        protocol.create_or_load_protocol(tmp_path, recipe_contract=frozen["recipe_contract"])


def test_historical_and_after_freeze_replays_never_become_prospective(frozen, monkeypatch):
    monkeypatch.setattr(protocol, "_utc_now", lambda: datetime(2026, 9, 23, tzinfo=timezone.utc))
    assert protocol.classify_delivery(frozen, "2026-09-19") == "historical_diagnostic"
    assert protocol.classify_delivery(frozen, "2026-09-20") == "postfreeze_retrospective"
    assert protocol.classify_delivery(frozen, "2026-09-21", sealed_at="2026-09-20T06:00:00+00:00") == "postfreeze_retrospective"
    assert protocol.classify_delivery(frozen, "2026-09-22", sealed_at="2026-09-21T06:00:00+00:00") == "postfreeze_retrospective"
    assert protocol.classify_delivery(frozen, "2026-09-22", sealed_at="2026-09-21T05:59:59+00:00") == "prospective"
    assert protocol.classify_delivery(frozen, "2026-09-25", sealed_at="2026-09-24T05:00:00+00:00") == "postfreeze_retrospective"
    with pytest.raises(ValueError, match="timezone-aware"):
        protocol.classify_delivery(frozen, "2026-09-22", sealed_at="2026-09-21T05:59:59")


@pytest.mark.parametrize("day,hours,offset", [("2026-03-29", 23, 2), ("2026-10-25", 25, 1), ("2026-09-22", 24, 2)])
def test_complete_physical_hours_and_civil_cutoff(day, hours, offset):
    start = datetime.fromisoformat(day).replace(tzinfo=protocol._CIVIL).astimezone(timezone.utc)
    stamps = [start + timedelta(hours=index) for index in range(hours)]
    assert protocol.complete_delivery(day, stamps, [1.] * hours, [2.] * hours)
    assert not protocol.complete_delivery(day, stamps[:-1], [1.] * hours, [2.] * hours)
    assert not protocol.complete_delivery(day, stamps[:-1] + stamps[:1], [1.] * hours, [2.] * hours)
    assert not protocol.complete_delivery(day, stamps, [float("nan")] * hours, [2.] * hours)
    next_day = (datetime.fromisoformat(day).date() + timedelta(days=1)).isoformat()
    assert protocol.delivery_cutoff(next_day).utcoffset() == timedelta(hours=offset)


def test_gate_requires_both_candidates_all_countries_and_verified_seal(frozen, monkeypatch):
    monkeypatch.setattr(protocol, "_utc_now", lambda: datetime(2026, 9, 23, tzinfo=timezone.utc))
    rows = records(frozen)
    assert protocol.prospective_gate(frozen, rows)["prospective_complete_days"] == 1
    assert protocol.prospective_gate(frozen, rows[:-1])["prospective_complete_days"] == 0
    rows[-1]["complete"] = False
    assert protocol.prospective_gate(frozen, rows)["prospective_complete_days"] == 0
    rows[-1]["complete"] = True
    rows[-1].pop("forecast_sha256")
    result = protocol.prospective_gate(frozen, rows)
    assert result["prospective_complete_days"] == 0
    assert result["common_days"]["postfreeze_retrospective"] == ["2026-09-22"]
    with pytest.raises(ValueError, match="Duplicate"):
        protocol.prospective_gate(frozen, rows + rows[:1])


def test_thirty_common_days_only_unlock_description_not_promotion(frozen, monkeypatch):
    monkeypatch.setattr(protocol, "_utc_now", lambda: datetime(2026, 11, 1, tzinfo=timezone.utc))
    rows = []
    for index in range(30):
        day = (datetime(2026, 9, 22).date() + timedelta(days=index)).isoformat()
        seal = (protocol.delivery_cutoff(day) - timedelta(minutes=1)).isoformat()
        rows.extend(records(frozen, day, seal))
    assert protocol.prospective_gate(frozen, rows[:-8])["descriptive_ready"] is False
    result = protocol.prospective_gate(frozen, rows)
    assert result["descriptive_ready"] is True
    assert result["promotion_allowed"] is False
    assert result["prospective_complete_days"] == 30
    assert result["remaining_prospective_days"] == 0


def test_no_new_data_is_pending_and_today_is_not_complete(frozen):
    result = protocol.prospective_gate(frozen, [])
    assert result["status"] == "pending_new_prospective_days"
    assert result["remaining_prospective_days"] == 30
    assert not any(protocol.prospective_gate(frozen, records(frozen, "2026-09-20"))["common_day_counts"].values())
