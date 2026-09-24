from copy import deepcopy
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import model_storm_vps_metrics as vps
from chronos2_hourly import model_storm_rolling as rolling


def zone(day="2026-09-19"):
    hours = vps._hours(day, "Europe/Paris")
    return {"zone": "FR", "name": "France", "timezone": "Europe/Paris",
            "rolling_history": {"rows": [{"timestamp_utc": t.isoformat(), "model": 100.123 + i,
                "observed": 90.456 + i, "storm_dashboard_cache": 101 + i, "storm": 101 + i}
                for i, t in enumerate(hours)], "sources": {"storm_dashboard_cache": {
                    "status": "complete", "exact_cache_hour_mask_verified": True}}},
            "vps_history": {"source": {"status": "partial", "zone": "FR", "timezone": "Europe/Paris",
                "series": "power.vps.fr.euromwh.h.da.pnl.storm", "extracted_at_utc": "2026-09-19T09:00:00Z"},
                "rows": [{"timestamp_utc": t.isoformat(), "pnl": 10.0} for t in hours]}}


@pytest.fixture
def mock_plan(monkeypatch):
    captured = []
    def plan(forecast):
        captured.append(list(forecast))
        return {"charge_mw": [0.] * len(forecast), "discharge_mw": [1.] * len(forecast),
                "soc_mwh": [0.] * (len(forecast)+1)}
    monkeypatch.setattr(vps, "optimize_vps_schedule", plan)
    return captured


def test_calendar_denominator_no_zero_imputation_and_raw_settlement(mock_plan):
    data = zone()
    before = deepcopy(data)
    windows = vps.build_vps_windows(data, "2026-09-19", (7, 30, 365))
    for n in (7, 30, 365):
        result = windows[str(n)]
        assert result["storm_daily_pnl"] == pytest.approx(240 / n)
        expected = sum(np.round(r["observed"], 2) for r in data["rolling_history"]["rows"]) / n
        assert result["model_daily_pnl"] == pytest.approx(expected)
        assert result["published_days"] == result["estimated_days"] == 1
        assert result["missing_publication_days"] == n - 1
        assert result["same_support"]
    assert mock_plan == [[r["model"] for r in data["rolling_history"]["rows"]]]
    assert data == before
    json.dumps(windows, allow_nan=False)


@pytest.mark.parametrize("change,reason", [("partial_vps", "partial_storm_cashflows"),
    ("missing_model", "missing_model_or_observed"), ("missing_observed", "missing_model_or_observed"),
    ("duplicate_price", "missing_model_or_observed"), ("negative_first", "unvalidated_first_hour_sign"),
    ("zero_first", "unvalidated_first_hour_sign")])
def test_incomplete_model_is_not_ranked_on_subset(change, reason, mock_plan):
    data = zone()
    rows = data["rolling_history"]["rows"]
    if change == "partial_vps": data["vps_history"]["rows"][0]["pnl"] = None
    if change == "missing_model": rows[4]["model"] = None
    if change == "missing_observed": rows[4]["observed"] = None
    if change == "duplicate_price": rows.append(deepcopy(rows[0]))
    if change in ("negative_first", "zero_first"): rows[0]["model"] = -1 if change == "negative_first" else 0
    result = vps.build_vps_windows(data, "2026-09-19", (7,))["7"]
    assert result["storm_daily_pnl"] == pytest.approx((230 if change == "partial_vps" else 240) / 7)
    assert result["model_daily_pnl"] is None
    assert not result["same_support"]
    assert result["exclusion_counts"] == {reason: 1}
    assert mock_plan == []


@pytest.mark.parametrize("fault", ["duplicate", "naive", "inf", "zone", "series", "timezone", "invalid"])
def test_invalid_published_source_fails_closed(fault, mock_plan):
    data = zone()
    history = data["vps_history"]
    if fault == "duplicate": history["rows"].append(deepcopy(history["rows"][0]))
    if fault == "naive": history["rows"][0]["timestamp_utc"] = "2026-09-19T00:00:00"
    if fault == "inf": history["rows"][0]["pnl"] = float("inf")
    if fault in ("zone", "series", "timezone"): history["source"][fault] = "wrong"
    if fault == "invalid": history["source"]["status"] = "invalid"
    result = vps.build_vps_windows(data, "2026-09-19", (7,))["7"]
    assert not result["source_verified"]
    assert result["storm_daily_pnl"] is result["model_daily_pnl"] is None
    assert mock_plan == []


@pytest.mark.parametrize("value,expected", [(None, None), (0., 0.), (-10., -240./7)])
def test_missing_zero_and_negative_cashflows_differ(value, expected, mock_plan):
    data = zone()
    for row in data["vps_history"]["rows"]: row["pnl"] = value
    result = vps.build_vps_windows(data, "2026-09-19", (7,))["7"]
    assert result["storm_daily_pnl"] == expected
    assert result["same_support"] is (value is not None)


def test_future_and_old_cashflows_excluded(mock_plan):
    data = zone()
    for stamp in ("2026-09-19T22:00:00Z", "2025-01-01T00:00:00Z"):
        data["vps_history"]["rows"].append({"timestamp_utc": stamp, "pnl": 1000000.})
    result = vps.build_vps_windows(data, "2026-09-19", (7,))["7"]
    assert result["storm_daily_pnl"] == pytest.approx(240/7)
    assert result["published_days"] == 1


def test_failed_solver_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(vps, "optimize_vps_schedule", lambda _: None)
    result = vps.build_vps_windows(zone(), "2026-09-19", (7,))["7"]
    assert result["storm_daily_pnl"] == pytest.approx(240/7)
    assert result["model_daily_pnl"] is None
    assert result["exclusion_counts"] == {"solver_unavailable": 1}


def test_short_extraction_must_not_look_like_365_day_source(mock_plan):
    data = zone()
    data["vps_history"]["source"].update(window_start_utc="2026-06-19T00:00:00Z", window_end_utc="2026-09-20T23:00:00Z")
    windows = vps.build_vps_windows(data, "2026-09-19", (90, 365))
    assert windows["90"]["storm_daily_pnl"] == pytest.approx(240/90)
    assert not windows["365"]["query_covers_window"]
    assert windows["365"]["storm_daily_pnl"] is windows["365"]["model_daily_pnl"] is None


@pytest.mark.parametrize("day,n", [("2026-03-29", 23), ("2026-10-25", 25)])
def test_dst_physical_hours_and_extension_label(day, n, mock_plan):
    result = vps.build_vps_windows(zone(day), day, (7,))["7"]
    assert result["published_hours"] == n
    assert result["dst_extension_days"] == 1
    assert result["storm_daily_pnl"] == pytest.approx(n*10/7)
    assert len(mock_plan[0]) == n


@pytest.mark.parametrize("n", [23, 24, 25])
def test_physical_plan_constraints_and_no_shared_mutation(n):
    prices = [100.] * 6 + [25.] * 6 + [300.] * (n-12)
    plan = vps.optimize_vps_schedule(prices)
    assert plan is not None
    c, d, s = (np.asarray(plan[k]) for k in ("charge_mw", "discharge_mw", "soc_mwh"))
    assert s[0] == pytest.approx(2)
    assert s[-1] == pytest.approx(1)
    assert c[0] == pytest.approx(0)
    assert d[0] == pytest.approx(1)
    assert np.max(np.abs(np.diff(s)-.85*c+d)) < 2e-6
    assert not np.any((c>2e-6)&(d>2e-6))
    assert min(s) >= -2e-6 and max(s) <= 4+2e-6
    plan["charge_mw"][0] = 200
    assert vps.optimize_vps_schedule(prices)["charge_mw"][0] == pytest.approx(0)


def test_sign_invalid_arrays_and_time_limited_solution(monkeypatch):
    assert vps.optimize_vps_schedule([0.]*24) is None
    assert vps.optimize_vps_schedule([-1.]*24) is None
    for prices in ([1.]*22, [True]*24, [float("nan")]*24):
        with pytest.raises(ValueError): vps.optimize_vps_schedule(prices)
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, "milp", lambda *a, **k: SimpleNamespace(success=True, status=1, x=np.zeros(97)))
    vps._schedule.cache_clear()
    assert vps.optimize_vps_schedule([110.1]*24) is None
    vps._schedule.cache_clear()


def test_paper_report_ignores_legacy_vps_cashflows_and_keeps_price_metrics(monkeypatch, mock_plan):
    monkeypatch.setattr(rolling, "WINDOWS", (7, 90))
    def forbidden(*args, **kwargs):
        pytest.fail("Legacy VPS cashflows and scheduling must not enter the paper report")
    monkeypatch.setattr(rolling, "optimize_storage_schedule", forbidden)
    monkeypatch.setattr(vps, "build_vps_windows", forbidden)
    data = {"delivery_day": "2026-09-19", "zones": [zone()]}
    before = deepcopy(data)
    result = rolling.build_rolling_performance(data)
    window = result["zones"][0]["windows"]["90"]
    assert window["frequencies"]["day"]["providers"][0]["daily_pnl"] is None
    assert "dashboard_vps_enabled" not in result
    assert result["source_scope"] == "internal_completed"
    assert data == before
    assert mock_plan == []
    changed = deepcopy(data)
    for row in changed["zones"][0]["vps_history"]["rows"]:
        row["pnl"] = -1000000.
    assert rolling.build_rolling_performance(changed) == result
    without = deepcopy(data)
    del without["zones"][0]["vps_history"]
    reference = rolling.build_rolling_performance(without)
    assert reference == result
    for n in ("7", "90"):
        for frequency in ("60min", "day"):
            for got, old in zip(result["zones"][0]["windows"][n]["frequencies"][frequency]["providers"],
                                reference["zones"][0]["windows"][n]["frequencies"][frequency]["providers"]):
                for metric in ("mae", "bias", "rmse", "hit_rate", "r2", "samples"):
                    assert got[metric] == old[metric]
