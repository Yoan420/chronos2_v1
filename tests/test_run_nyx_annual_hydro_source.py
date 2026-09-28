import json
import pandas as pd
import pytest
import run_nyx_annual_hydro_source as m


def payload(day):
    _, first, stop, cutoff = m.source_window(day)
    return {"schema_version": "2.0", "country": "fr", "endpoint": "public_power",
        "unit": "MW", "timezone": "Europe/Paris", "interval_minutes": 60, "resolution": "PT1H",
        "generated_at": (cutoff - pd.Timedelta(minutes=10)).isoformat(),
        "available_from": first.isoformat(), "available_until": (stop - pd.Timedelta(hours=1)).isoformat(),
        "series": [{"id": name} for name in m.hydro.SERIES],
        "data": [{"timestamp": t.isoformat(), "values": {name: 2000. for name in m.hydro.SERIES}}
                 for t in pd.date_range(first, stop, freq="h", inclusive="left")]}


class Session:
    def __init__(self, data):
        self.data = data
        self.calls = 0
    def get(self, *a, **kw):
        self.calls += 1
        return type("Response", (), {"status_code": 200, "content": json.dumps(self.data).encode()})()


@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2026-10-25", 25)])
def test_capture_preserves_dst_and_actual_pre_cutoff_evidence(tmp_path, day, hours):
    session = Session(payload(day))
    m.capture(day, tmp_path, session=session, now_utc=m.source_window(day)[-1] - pd.Timedelta(minutes=5))
    frame, receipt = m.verify_capture(tmp_path / day, day)
    assert len(frame) == hours and tuple(frame) == m.COLUMNS
    assert (frame.extra_hydro_fr_ror_gw_mean168h_lag48h == 2).all()
    assert (frame.extra_hydro_fr_ror_gw_trend24h_minus168h_lag48h == 0).all()
    assert receipt["origin_snapshot_capture_verified"] is True
    m.capture(day, tmp_path, session=session)
    assert session.calls == 1
    (tmp_path / day / "FR.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        m.verify_capture(tmp_path / day, day)


def test_late_capture_and_missing_archive_fail(tmp_path):
    day = "2026-09-30"
    session = Session(payload(day))
    with pytest.raises(ValueError, match="before D-1"):
        m.capture(day, tmp_path, session=session, now_utc=m.source_window(day)[-1])
    assert session.calls == 0
    with pytest.raises(FileNotFoundError):
        m.assemble(day, tmp_path, tmp_path / "bundle")
    assert not (tmp_path / "bundle/source_receipts/public_hydro.json").exists()
