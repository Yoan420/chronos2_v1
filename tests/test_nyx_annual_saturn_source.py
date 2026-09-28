"""No network: origin-specific targets, physical DST hours and immutable caches."""
import json
import numpy as np
import pandas as pd
import pytest
from chronos2_hourly import nyx_annual_saturn_source as m


@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2026-10-25", 25)])
def test_snapshot_queries_own_cutoff_and_never_future_labels(monkeypatch, tmp_path, day, hours):
    calls = []
    def fetch(client, series, start, end, timezone, **kwargs):
        calls.append((series, start, end, kwargs))
        index = pd.date_range(start, end, freq="h", tz="UTC")
        return pd.Series(np.arange(len(index), dtype=float), index=index)
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    receipt = m.capture_day(object(), day, tmp_path, now_utc=m.cutoff(day))
    cov, prices, _ = m.verify_day(tmp_path / day, day)
    assert len(cov) == hours and list(cov) == list(m.ALIASES)
    assert prices.index[-1] < cov.index[0]
    assert len(calls) == 18
    assert all(c[3]["revision_date"] == m.cutoff(day) for c in calls)
    assert all(c[3]["incomplete_dst_policy"] == "raise" for c in calls if ".price.da." in c[0])
    assert receipt["publication_verified"] is False
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", lambda *a, **k: pytest.fail("cached snapshot contacted Saturn"))
    assert m.capture_day(object(), day, tmp_path) == receipt
    path = tmp_path / day / "prices.parquet"
    prices.iloc[0, 0] += 1
    prices.to_parquet(path)
    with pytest.raises(ValueError, match="modified prices"):
        m.capture_day(object(), day, tmp_path)


def test_missing_covariate_does_not_emit_receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", lambda *a, **k: pd.Series([], index=pd.DatetimeIndex([], tz="UTC"), dtype=float))
    with pytest.raises(ValueError, match="missing/nonfinite"):
        m.capture_day(object(), "2026-09-29", tmp_path, now_utc=m.cutoff("2026-09-29"))
    assert not list(tmp_path.rglob("receipt.json"))


def test_future_cutoff_rejected_before_queries(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", lambda *a, **k: pytest.fail("early query"))
    with pytest.raises(ValueError, match="not reached"):
        m.capture_day(object(), "2026-09-29", tmp_path, now_utc=m.cutoff("2026-09-29") - pd.Timedelta(seconds=1))


def test_bootstrap_start_remains_stable_as_days_advance(tmp_path):
    (tmp_path / "plan.json").write_text(json.dumps({"protocol": m.PROTOCOL, "first_delivery_day": "2024-06-01"}))
    assert str(m.source_start(tmp_path, "2026-09-29")) == "2024-06-01"
    assert str(m.source_start(tmp_path, "2026-10-29")) == "2024-06-01"
    with pytest.raises(ValueError, match="deeper history"):
        m.source_start(tmp_path, "2026-09-29", "2024-01-01")
