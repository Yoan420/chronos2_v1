from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_regional_cpu_sources as sources
from chronos2_hourly.nyx_regional_cpu import grid


def _source_files(tmp_path, monkeypatch, *, missing_hour=False):
    from chronos2_hourly import nuclear_residual_inputs, nuclear_sources
    monkeypatch.setattr(nuclear_sources, "audit_nuclear_store",
                        lambda *args, **kwargs: {"complete": True, "blockers": []})
    monkeypatch.setattr(nuclear_residual_inputs, "audit_residual_bank",
                        lambda *args, **kwargs: {"complete": True, "blockers": []})
    day = "2026-09-10"
    paths = {"residual_bank": tmp_path / "residual.parquet",
             "nuclear_store": tmp_path / "nuclear.parquet"}
    pd.DataFrame({"value_time_utc": [pd.Timestamp("2026-09-10T00:00Z")],
                  "value": [1.]}).to_parquet(paths["residual_bank"])
    pd.DataFrame({"value_time_utc": [pd.Timestamp("2026-09-10T00:00Z")],
                  "value": [1.]}).to_parquet(paths["nuclear_store"])
    for key in ("residual_bank", "nuclear_store"):
        paths[key].with_name(paths[key].name + ".audit.json").write_text("{}")
    for zone in ("FR", "DE", "BE", "NL"):
        index = grid("2025-09-03", "2026-09-11", zone)
        if missing_hour and zone == "FR":
            index = index.delete(5)
        frame = pd.DataFrame({"timestamp": [stamp.isoformat() for stamp in index],
                              "value": np.ones(len(index))})
        path = tmp_path / f"target_{zone}.csv.gz"
        frame.to_csv(path, index=False, compression="gzip")
        paths[f"target_{zone}"] = path
    monkeypatch.setattr(sources, "source_paths", lambda root: paths)
    return paths, day


def test_load_sources_excludes_delivery_day_labels(tmp_path, monkeypatch):
    paths, day = _source_files(tmp_path, monkeypatch)
    preflight = sources.preflight_sources(tmp_path)
    assert preflight["ready"] is True
    loaded, audit = sources.load_sources(tmp_path, day)
    assert set(loaded["prices"]) == {"FR", "DE", "BE", "NL"}
    assert all(price.index.max() < grid(day, "2026-09-11", zone)[0]
               for zone, price in loaded["prices"].items())
    assert audit["future_actuals_loaded"] is False
    assert set(audit["source_sha256"]) == set(paths)


def test_load_sources_refuses_missing_historical_hour(tmp_path, monkeypatch):
    _source_files(tmp_path, monkeypatch, missing_hour=True)
    with pytest.raises(ValueError, match="canonical Saturn target is missing 1"):
        sources.load_sources(tmp_path, "2026-09-10")
