from pathlib import Path
import pytest
from chronos2_hourly import nyx_live_sources as s


def test_paths_cannot_reach_production():
    with pytest.raises(ValueError):
        s.safe(s.ROOT / "data/pit/solar_wind_v1")
    with pytest.raises(ValueError):
        s.safe(s.OUTPUT)
    assert s.safe(s.OUTPUT / "2026-09-24/sources").is_relative_to(s.OUTPUT)


def test_exact_native_series_no_nl_ecmwf_replacement():
    six = s.plans(s.OUTPUT / "2026-09-24/sources")
    eight = s.plans(s.OUTPUT / "2026-09-24/sources", True)
    assert len(six) == 6 and len(eight) == 8
    assert {p.alias: p.series for p in six} == s.wind.GENERATION_SERIES
    extra = {p.zone for p in eight if p.alias not in {x.alias for x in six}}
    assert extra == {"BE", "FR"}
    assert all(p.value_scale == 1 and p.unit == "GW" for p in eight)


def test_future_cutoff_refused_without_write(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "OUTPUT", tmp_path)
    with pytest.raises(ValueError, match="has not occurred"):
        s.ensure_live_sources(tmp_path / "capture", start_day="2099-01-01", delivery_day="2099-01-02")
    assert not (tmp_path / "capture").exists()
