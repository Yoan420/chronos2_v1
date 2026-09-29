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
    with pytest.raises(m.SaturnSourceError, match="missing/nonfinite"):
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


def test_missing_profile_reports_origin_alias_and_first_physical_hours(monkeypatch, tmp_path):
    day = "2025-03-30"
    expected = m.grid(day)
    absent = expected[expected.tz_convert("Europe/Amsterdam").strftime("%H:%M").isin(["04:00", "06:00"])]
    def fetch(client, series, start, end, timezone, **kwargs):
        index = pd.date_range(start, end, freq="h", tz="UTC")
        if series == "power.nl.residual.load.hourly.gw.fcst":
            index = index.difference(absent)
        return pd.Series(1., index=index)
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    with pytest.raises(m.SaturnSourceError) as caught:
        m.capture_day(object(), day, tmp_path, now_utc=m.cutoff(day))
    error = caught.value
    assert error.day == day and error.alias == "nl_residual_load_fcst"
    assert error.phase == "forecast_profile" and isinstance(error.__cause__, ValueError)
    assert "2 missing/nonfinite" in str(error)
    assert all(stamp.isoformat() in str(error) for stamp in absent)
    assert not list(tmp_path.rglob("receipt.json"))  # Diagnostic never fills missing values.


def test_target_failure_reports_country_origin_and_preserves_cause(monkeypatch, tmp_path):
    cause = RuntimeError("password=example-secret")
    def fetch(client, series, start, end, timezone, **kwargs):
        if ".price.da." in series:
            raise cause
        return pd.Series(1., index=pd.date_range(start, end, freq="h", tz="UTC"))
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    with pytest.raises(m.SaturnSourceError) as caught:
        m.capture_day(object(), "2026-09-29", tmp_path, now_utc=m.cutoff("2026-09-29"))
    assert caught.value.alias == "FR_target" and caught.value.phase == "target_history"
    assert caught.value.__cause__ is cause
    assert "example-secret" not in str(caught.value)


def test_future_failure_adds_failed_day_for_client_creation(monkeypatch, tmp_path):
    import run_nyx_annual_auction_prices_source
    monkeypatch.setattr(run_nyx_annual_auction_prices_source, "load_plan", lambda: {})
    cause = OSError("password=example-secret")
    def client():
        raise cause
    with pytest.raises(m.SaturnSourceError) as caught:
        m.sync("2026-09-29", first_day="2026-09-29", cache=tmp_path,
               workers=1, client_factory=client)
    assert caught.value.day == "2026-09-29" and caught.value.phase == "daily_capture"
    assert caught.value.__cause__ is cause and "example-secret" not in str(caught.value)


def test_cli_error_is_structured_dated_and_redacts_exception_chain(monkeypatch, tmp_path, capsys):
    import run_nyx_annual_saturn_source as cli
    cause = RuntimeError("password=example-secret")
    def sync(*args, **kwargs):
        raise m.SaturnSourceError("2025-03-30", "forecast_profile", cause,
                                 alias="nl_residual_load_fcst", series="power.nl.residual.load.hourly.gw.fcst") from cause
    monkeypatch.setattr(cli, "sync", sync)
    monkeypatch.setattr(cli, "publish", lambda *a, **k: pytest.fail("failed sync published a receipt"))
    assert cli.main(["--delivery-day", "2026-09-29", "--cache", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    result = json.loads(output)
    assert result["state"] == "ERROR" and result["source_group"] == "saturn"
    assert result["delivery_day"] == "2026-09-29" and result["failed_day"] == "2025-03-30"
    assert result["alias"] == "nl_residual_load_fcst" and len(result["exception_chain"]) == 2
    assert "example-secret" not in output


def test_cli_publish_error_is_distinct_from_download_error(monkeypatch, tmp_path, capsys):
    import run_nyx_annual_saturn_source as cli
    monkeypatch.setattr(cli, "sync", lambda *a, **k: pytest.fail("assemble-only downloaded sources"))
    def publish(*args, **kwargs):
        raise FileNotFoundError("Missing daily receipt for 2025-03-30")
    monkeypatch.setattr(cli, "publish", publish)
    assert cli.main(["--delivery-day", "2026-09-29", "--cache", str(tmp_path), "--assemble-only"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["phase"] == "publish" and "2025-03-30" in result["error"]
