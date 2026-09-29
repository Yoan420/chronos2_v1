"""Reproduce the mailed 2024-08-17 failure using the actual tracked NL archive."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_annual_saturn_source as source
from chronos2_hourly import nyx_annual_saturn_archive as archive


def test_missing_fr_origin_and_unavailable_nl_server_use_verified_repository_profile(tmp_path, monkeypatch):
    day, outer = "2024-08-17", "2026-09-30"
    calls = []

    def fetch(client, series, start, end, timezone, **kwargs):
        revision = kwargs["revision_date"]
        calls.append((series, revision))
        if series == "power.fr.residual.load.hourly.gw.fcst" and revision == source.cutoff(day):
            raise RuntimeError("Historical FR profile empty")
        if series == archive.SERIES:
            if revision == source.cutoff(day):
                # The legacy archive shows only 22 eligible hours at this origin.
                return pd.Series(1., index=source.grid(day)[:-2])
            raise RuntimeError("NL profile empty at the outer cutoff")
        return pd.Series(np.arange(len(pd.date_range(start, end, freq="h")), dtype=float),
                         index=pd.date_range(start, end, freq="h"))

    monkeypatch.setattr(source, "fetch_saturn_series_from_client", fetch)
    monkeypatch.setattr(source.time, "sleep", lambda _: None)
    client = lambda: SimpleNamespace(session=SimpleNamespace(close=lambda: None))
    source._sync_profile_day(day, outer, tmp_path, client)
    directory = source._profile_cache_directory(tmp_path, day, outer)
    frame, receipt = source.verify_profile_day(directory, day, outer_day=outer)
    expected, evidence = archive.recover_nl_profile(day, outer)
    assert len(frame) == 24 and np.isfinite(frame.to_numpy(float)).all()
    np.testing.assert_array_equal(frame[archive.ALIAS].to_numpy(), expected.to_numpy())
    assert evidence["snapshot_time_utc"] == "2026-08-05T12:17:00+00:00"
    assert receipt["alias_revisions_utc"][archive.ALIAS] == evidence["snapshot_time_utc"]
    assert receipt["alias_revisions_utc"]["fr_residual_load_fcst"] == source.cutoff(outer).isoformat()
    assert receipt["alias_revisions_utc"]["de_residual_load_fcst"] == source.cutoff(day).isoformat()
    assert all(revision == source.cutoff(day) for series, revision in calls
               if series == "power.de.residual.load.hourly.gw.fcst")
    monkeypatch.setattr(source, "fetch_saturn_series_from_client",
                        lambda *a, **k: pytest.fail("Resume contacted Saturn"))
    source._sync_profile_day(day, outer, tmp_path, client)
