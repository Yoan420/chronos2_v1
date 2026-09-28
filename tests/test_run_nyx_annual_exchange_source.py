"""Prospective exchange capture tests use synthetic API bodies only."""
from __future__ import annotations

import json
import zipfile

import pandas as pd
import pytest

import run_nyx_annual_exchange_source as m


class Response:
    status_code = 200

    def __init__(self, content: bytes):
        self.content = content


class Session:
    def __init__(self, bodies):
        self.bodies = bodies
        self.calls = []

    def get(self, url, *, params, timeout, allow_redirects):
        zone = "DE" if url.endswith("/cbpf") else "FR"
        self.calls.append((zone, params))
        assert timeout == (10, 45) and allow_redirects is False
        return Response(self.bodies[zone])


def body(day: str, zone: str, *, missing_quarter=False,
         generated_after_cutoff=False, null_value=False) -> bytes:
    current, first, stop, cutoff = m._source_window(day)
    index = pd.date_range(first, stop, freq="15min", inclusive="left")
    if missing_quarter:
        index = index.delete(1)
    timezone = "Europe/Berlin" if zone == "DE" else "Europe/Paris"
    fields = ("france", "netherlands", "belgium", "sum") if zone == "DE" else (
        "cross_border_electricity_trading",)
    row = ({"france": -1.0, "netherlands": 2.0, "belgium": 0.0,
            "sum": 3.0} if zone == "DE" else
           {"cross_border_electricity_trading": -1000.0})
    payload = {
        "schema_version": "2.0", "country": zone.lower(),
        "endpoint": "cbpf" if zone == "DE" else "public_power",
        "unit": "GW" if zone == "DE" else "MW",
        "timezone": timezone, "interval_minutes": 15, "resolution": "PT15M",
        "generated_at": (cutoff + pd.Timedelta(minutes=1) if generated_after_cutoff
                         else cutoff - pd.Timedelta(hours=1)).isoformat(),
        "available_from": first.isoformat(),
        "available_until": (stop - pd.Timedelta(minutes=15)).isoformat(),
        "attributes": ({"sign_convention": "positive = import, negative = export"}
                       if zone == "DE" else {}),
        "series": [{"id": name} for name in fields],
        "data": [{"timestamp": t.tz_convert(timezone).isoformat(),
                  "values": row} for t in index],
    }
    if null_value:
        name = fields[0]
        payload["data"][0]["values"] = dict(row, **{name: None})
    assert len(current) in (23, 24, 25)
    return json.dumps(payload, separators=(",", ":")).encode()


@pytest.mark.parametrize("day,expected_hours", [("2026-03-29", 23),
                                                ("2026-10-25", 25)])
def test_daily_capture_is_causal_immutable_and_preserves_signs(tmp_path, day, expected_hours):
    cutoff = m._source_window(day)[-1]
    session = Session({zone: body(day, zone) for zone in ("DE", "FR")})
    receipt_path = m.capture(day, tmp_path, session=session,
                             now_utc=cutoff - pd.Timedelta(minutes=5))
    assert len(session.calls) == 2
    assert all(pd.Timestamp(params["end"]) < cutoff for _, params in session.calls)
    frame, receipt = m._verify_capture(receipt_path.parent, day)
    assert len(frame) == expected_hours
    assert frame[m.exchange.LEVELS[0]].eq(-1.).all()
    assert frame[m.exchange.LEVELS[-1]].eq(-1.).all()
    assert frame.filter(like="__available").eq(1.).all().all()
    assert receipt["provider_first_publication_timestamp_verified"] is False
    before = receipt_path.read_bytes()
    m.capture(day, tmp_path, session=Session({}),
              now_utc=cutoff - pd.Timedelta(minutes=5))
    assert receipt_path.read_bytes() == before


def test_late_or_unavailable_api_never_publishes_capture(tmp_path):
    day = "2026-09-29"
    cutoff = m._source_window(day)[-1]
    valid = {zone: body(day, zone) for zone in ("DE", "FR")}
    late = Session(valid)
    with pytest.raises(ValueError, match="starts after"):
        m.capture(day, tmp_path, session=late,
                  now_utc=cutoff + pd.Timedelta(minutes=1))
    assert late.calls == []
    after = Session({"DE": body(day, "DE", generated_after_cutoff=True),
                     "FR": valid["FR"]})
    with pytest.raises(ValueError, match="before D-1 08"):
        m.capture(day, tmp_path, session=after,
                  now_utc=cutoff - pd.Timedelta(minutes=5))
    assert not (tmp_path / day / "capture.json").exists()
    incomplete = Session({"DE": body(day, "DE", missing_quarter=True),
                          "FR": valid["FR"]})
    with pytest.raises(ValueError, match="incomplete source interval grid"):
        m.capture(day, tmp_path, session=incomplete,
                  now_utc=cutoff - pd.Timedelta(minutes=5))
    assert not (tmp_path / day / "capture.json").exists()


def test_explicit_null_retains_historical_missing_flag_without_imputation(tmp_path):
    day = "2026-09-29"
    cutoff = m._source_window(day)[-1]
    session = Session({"DE": body(day, "DE"),
                       "FR": body(day, "FR", null_value=True)})
    path = m.capture(day, tmp_path, session=session,
                     now_utc=cutoff - pd.Timedelta(minutes=5))
    frame, receipt = m._verify_capture(path.parent, day)
    level = m.exchange.LEVELS[-1]
    assert pd.isna(frame[level].iloc[0])
    assert frame[level + "__available"].iloc[0] == 0
    assert receipt["missing_hours_by_feature"][level] == 1


def test_annual_assembly_refuses_missing_daily_vintages(tmp_path):
    day = "2026-09-29"
    cutoff = m._source_window(day)[-1]
    session = Session({zone: body(day, zone) for zone in ("DE", "FR")})
    m.capture(day, tmp_path / "archive", session=session,
              now_utc=cutoff - pd.Timedelta(minutes=5))
    bundle = tmp_path / "bundle"
    with pytest.raises(ValueError, match="No PIT exchange capture"):
        m.assemble(day, tmp_path / "archive", bundle)
    assert not (bundle / "source_receipts/lagged_exchange.json").exists()


def test_assembly_binds_verified_captures_to_preflight_receipt(tmp_path, monkeypatch):
    # Use the real daily captures and receipt validator with a two-day fixture.
    # Production's _capture_days remains the full 366 days.
    days = ["2026-09-28", "2026-09-29"]
    archive = tmp_path / "archive"
    for day in days:
        cutoff = m._source_window(day)[-1]
        session = Session({zone: body(day, zone) for zone in ("DE", "FR")})
        m.capture(day, archive, session=session,
                  now_utc=cutoff - pd.Timedelta(minutes=5))
    original_grid = m.delivery_grid

    def two_day_grid(day):
        full, current, cutoff = original_grid(day)
        if day == days[-1]:
            prior = original_grid(days[0])[1]
            return prior.append(current), current, cutoff
        return full, current, cutoff

    monkeypatch.setattr(m, "delivery_grid", two_day_grid)
    monkeypatch.setattr(m, "_capture_days", lambda day: days)
    bundle = tmp_path / "bundle"
    destination = m.assemble(days[-1], archive, bundle)
    receipt = json.loads(destination.read_text(encoding="utf-8"))
    assert receipt["source_group"] == "lagged_exchange"
    assert receipt["origin_snapshot_capture_verified"] is True
    assert receipt["provider_first_publication_timestamp_verified"] is False
    assert receipt["captures"] == 2
    m.validate_source_receipt(receipt, group="lagged_exchange", day=days[-1],
                              bundle=bundle, cutoff=original_grid(days[-1])[-1])
    frame = pd.read_parquet(bundle / "source_artifacts/lagged_exchange/features.parquet")
    assert len(frame) == 48 and list(frame.columns) == list(m.exchange.COLUMNS)
    with zipfile.ZipFile(bundle / "source_artifacts/lagged_exchange/pit_captures.zip") as saved:
        assert len(saved.namelist()) == 2 * len(m.CAPTURE_NAMES)


def test_tampered_raw_body_is_rejected_before_annual_assembly(tmp_path):
    day = "2026-09-29"
    cutoff = m._source_window(day)[-1]
    session = Session({zone: body(day, zone) for zone in ("DE", "FR")})
    receipt = m.capture(day, tmp_path, session=session,
                        now_utc=cutoff - pd.Timedelta(minutes=5))
    (receipt.parent / "FR.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="bytes changed"):
        m._verify_capture(receipt.parent, day)
