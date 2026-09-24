import json
from pathlib import Path
import pandas as pd
import pytest
import run_nyx_test2_live as r


def test_reference_verified_read_only():
    manifest = r.verify_reference()
    assert manifest["reference_identity"] == "3a08985a6007d5b8"
    assert manifest["active_production_changed"] is False


def test_plan_order_resources_and_frozen_day():
    plan = r.build_plan("2026-09-24")
    assert plan["pairs"] == [["DE", "NL"], ["BE", "FR"]]
    assert plan["zone_order"] == ["DE", "NL", "BE", "FR"]
    assert plan["threads"] == plan["workers"] == 1
    assert plan["device"] == "cpu" and not plan["automatic_promotion"]
    assert plan["delivery_day"] == "2026-09-24"


@pytest.mark.parametrize("day", ["2026-09-24T01:00:00", "2026-09-24T00:00:00Z", "not-a-day"])
def test_bad_day(day):
    with pytest.raises(ValueError):
        r.build_plan(day)


def test_status_final_and_error(monkeypatch):
    saved = []
    monkeypatch.setattr(r, "atomic_json", lambda path, value: saved.append(dict(value)))
    obj = object.__new__(r.Run)
    obj.root, obj.state = Path("unused"), {}
    obj.status("error", status="FAILED", error="test")
    assert saved[-1]["status"] == "FAILED"
    obj.status("complete", status="COMPLETE", eta_seconds=0)
    assert saved[-1]["status"] == "COMPLETE"


def test_indexed_rejects_naive_and_duplicates():
    with pytest.raises(ValueError):
        r.indexed(pd.DataFrame(index=pd.date_range("2026-01-01", periods=2, freq="h")))
    index = pd.DatetimeIndex(["2026-01-01T00:00Z"] * 2)
    with pytest.raises(ValueError):
        r.indexed(pd.DataFrame(index=index))


@pytest.mark.parametrize("damage", [None, "extra_file", "wrong_route", "future_actual"])
def test_final_scientific_verification(tmp_path, monkeypatch, damage):
    from chronos2_hourly import nyx_live_sources as sources
    monkeypatch.setattr(sources, "OUTPUT", tmp_path)
    work = tmp_path / "pair"
    work.mkdir()
    identity = {"delivery_day": "2026-09-24", "pair": ["DE", "NL"]}
    for name, start, stop in (("historical_hybrid", "2026-06-23", "2026-09-24"),
                             ("forecast_hybrid", "2026-09-24", "2026-09-25")):
        idx = pd.date_range(pd.Timestamp(start, tz="Europe/Berlin"), pd.Timestamp(stop, tz="Europe/Berlin"),
                            freq="h", inclusive="left").tz_convert("UTC")
        panels = []
        for zone in identity["pair"]:
            frame = pd.DataFrame({"zone": zone, "selected_test2": False}, index=idx)
            for model in ("nyx", "test2", "hybrid"):
                for quantile, value in (("q10", 10.), ("q50", 50.), ("q90", 90.)):
                    frame[f"{model}__{quantile}"] = value
            if name == "historical_hybrid": frame["actual"] = 60.
            if damage == "wrong_route": frame["hybrid__q50"] = 51.
            if damage == "future_actual" and name == "forecast_hybrid": frame["actual"] = 99.
            panels.append(frame)
        pd.concat(panels).to_parquet(work / f"{name}.parquet")
    inventory = {p.name: r.sha(p) for p in work.iterdir()}
    r.atomic_json(work / "completion.json", {"status": "COMPLETE", "identity": identity, "files": inventory})
    if damage == "extra_file":
        r.atomic_json(work / "unexpected.json", {})
    if damage is None:
        assert r.verify_completed_pair(work, identity)["status"] == "COMPLETE"
    else:
        with pytest.raises(ValueError): r.verify_completed_pair(work, identity)
