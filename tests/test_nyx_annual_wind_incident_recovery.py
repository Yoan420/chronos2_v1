"""Replay the mailed NL wind failure against the tracked historical recipe."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_annual_saturn_source as source


ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = ROOT / "data/pit/solar_wind_v1/_wind_days/nl_wind_generation_fcst/125d3f413c656d9a5c3d"
ALIAS = "nl_wind_generation_fcst"


@pytest.mark.parametrize("day", ["2025-03-30", "2026-03-29"])
def test_actual_historical_wind_recipe_recovered_at_own_cutoff(tmp_path, monkeypatch, day):
    path = ARCHIVES / f"{day}.parquet"
    audit = json.loads(path.with_suffix(".parquet.audit.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == audit["sha256"]
    archived = pd.read_parquet(path)
    expected = archived.set_index("value_time_utc")["value"].sort_index()
    evidence = audit["source_substitutions"][0]
    missing = pd.Timestamp(evidence["value_time_utc"])
    native = expected.drop(missing)
    native.attrs = {}
    requests = []
    native_revisions = []

    class Client:
        def get(self, series, **kwargs):
            requests.append((series, kwargs))
            assert series == evidence["fallback_series"]
            assert kwargs["revision_date"] == source.cutoff(day)
            assert kwargs["nocache"] is True
            return pd.Series([evidence["raw_value_mw"]], index=pd.DatetimeIndex([missing]))

    def fetch(client, series, start, end, timezone, **kwargs):
        if series == audit["series"]:
            native_revisions.append(kwargs["revision_date"])
            return native.copy()
        return pd.Series(1., index=pd.date_range(start, end, freq="h"))

    monkeypatch.setattr(source, "fetch_saturn_series_from_client", fetch)
    monkeypatch.setattr(source.time, "sleep", lambda _: None)
    outer = "2026-09-30"
    source._sync_profile_day(day, outer, tmp_path, Client)
    directory = source._profile_cache_directory(tmp_path, day, outer)
    frame, receipt = source.verify_profile_day(directory, day, outer_day=outer)
    assert len(frame) == 23
    np.testing.assert_array_equal(frame[ALIAS].to_numpy(), expected.to_numpy())
    assert len(requests) == 1
    assert native_revisions == [source.cutoff(day)]
    assert receipt["forecast_origin_utc"] == source.cutoff(day).isoformat()
    assert receipt["source_substitution_count"] == 1
    substitution = receipt["source_substitutions"][0]
    for key in ("policy", "native_series", "fallback_series", "value_time_utc",
                "query_cutoff_utc", "raw_value_mw", "value_scale", "scaled_value_gw"):
        assert substitution[key] == evidence[key]
    assert substitution["production_pit_evidence"] is False
    # Own-cutoff days remain reusable for another outer delivery, without calls.
    monkeypatch.setattr(source, "fetch_saturn_series_from_client",
                        lambda *a, **k: pytest.fail("Cached day contacted Saturn"))
    source._sync_profile_day(day, "2026-10-01", tmp_path, Client)
    assert len(requests) == 1
