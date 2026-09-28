"""Thermal live capture contract with an in-memory Saturn state service."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_nyx_annual_thermal_source as m
from chronos2_hourly.nyx_annual_live_preflight import (
    delivery_grid, validate_source_receipt,
)


DAY = "2026-09-29"


class Saturn:
    def __init__(self, specs: dict, *, mismatch: bool = False):
        self.names = {spec["series"]: i for i, spec in enumerate(specs.values())}
        self.mismatch = mismatch
        self.calls: list[tuple] = []

    def value(self, series: str, day: pd.Timestamp) -> float:
        if self.names[series] == 0 and day == pd.Timestamp("2025-11-08"):
            return np.nan
        if self.names[series] == 1 and day == pd.Timestamp("2026-01-01"):
            return 0.0
        return float(self.names[series] + 1)

    def block_staircase(self, series: str, *, from_value_date, to_value_date, **request):
        assert request == m.REQUEST
        self.calls.append(("block", series, from_value_date, to_value_date))
        days = pd.date_range(from_value_date, to_value_date, freq="D")
        return pd.Series([self.value(series, day) for day in days], index=days)

    def get(self, series: str, *, from_value_date, to_value_date, revision_date):
        day = from_value_date + pd.Timedelta(days=1)
        assert revision_date == m.civil_cutoff(day)
        self.calls.append(("state", series, day))
        value = self.value(series, day)
        if self.mismatch and day == pd.Timestamp(DAY):
            value += 1.0
        if not np.isfinite(value):
            return None
        return pd.Series([value], index=pd.DatetimeIndex([day]))


def test_collect_and_publish_thermal_source(tmp_path: Path):
    source_plan = m.plan()
    saturn = Saturn(source_plan["specs"])
    _, current, cutoff = delivery_grid(DAY)
    daily, evidence = m.collect(saturn, DAY, now_utc=cutoff)
    assert len(daily) == 13
    assert len([call for call in saturn.calls if call[0] == "block"]) == 13
    assert len([call for call in saturn.calls if call[0] == "state"]) == 13 * 366
    assert all(series.index[0] == pd.Timestamp("2025-09-29")
               and series.index[-1] == pd.Timestamp(DAY) for series in daily.values())
    assert evidence["requested_days_per_series"] == 366
    assert evidence["states_verified_per_series"] == 366
    assert any(probe["day"] == "2025-11-08"
               for probe in evidence["independent_daily_state_checks"][m.SOURCES[0]])

    bundle = tmp_path / "live" / DAY
    path = m.publish(bundle, DAY, daily, evidence, source_plan, now_utc=cutoff)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["source_group"] == "thermal_capacity"
    assert receipt["model_inputs_complete"] is False
    assert receipt["provider_publication_timestamp_verified"] is False
    assert receipt["missing_days_by_series"][m.SOURCES[0]] == ["2025-11-08"]
    assert len(receipt["artifact_sha256"]) == 17
    assert receipt["asof_state_utc"] == cutoff.isoformat()
    validate_source_receipt(receipt, group="thermal_capacity", day=DAY,
                            bundle=bundle, cutoff=cutoff)
    assert m.publish(bundle, DAY, daily, evidence, source_plan, now_utc=cutoff) == path

    _, full, _ = m.grids(DAY)
    fr = pd.read_parquet(bundle / "source_artifacts/thermal_capacity/features_FR.parquet")
    assert fr.index.equals(full)
    assert list(fr.columns) == list(m.COLUMNS)
    missing_hours = full.tz_convert("Europe/Paris").date == pd.Timestamp("2025-11-08").date()
    column = f"thermal__{m.SOURCES[0]}_pmax_gw"
    assert fr.loc[missing_hours, column].isna().all()
    assert (fr.loc[missing_hours, column + "__available"] == 0).all()
    assert len(full[full.tz_convert("Europe/Paris").date == pd.Timestamp("2025-10-26").date()]) == 25
    assert len(full[full.tz_convert("Europe/Paris").date == pd.Timestamp("2026-03-29").date()]) == 23
    assert len(current) == 24


def test_blocks_future_or_conflicting_saturn_state():
    source_plan = m.plan()
    _, _, cutoff = delivery_grid(DAY)
    with pytest.raises(ValueError, match="has not occurred"):
        m.collect(Saturn(source_plan["specs"]), DAY,
                  now_utc=cutoff - pd.Timedelta(seconds=1))
    with pytest.raises(ValueError, match="disagrees"):
        m.collect(Saturn(source_plan["specs"], mismatch=True), DAY,
                  now_utc=cutoff)


def test_missing_sources_never_become_known_zero():
    days, full, _ = m.grids(DAY)
    daily = {source: pd.Series(1.0, index=days, name="pmax_gw")
             for source in m.SOURCES}
    daily["fr_ccgt"].loc[pd.Timestamp(DAY)] = np.nan
    daily["fr_gt"].loc[pd.Timestamp(DAY)] = 0.0
    hourly = {source: m.source_hourly(series, DAY)
              for source, series in daily.items()}
    fr = m.build_features(hourly, DAY)["FR"]
    on_day = full.tz_convert("Europe/Paris").date == pd.Timestamp(DAY).date()
    assert fr.loc[on_day, "thermal__own_ccgt_pmax_gw"].isna().all()
    assert (fr.loc[on_day, "thermal__own_ccgt_pmax_gw__available"] == 0).all()
    assert (fr.loc[on_day, "thermal__own_gt_pmax_gw"] == 0).all()
    assert (fr.loc[on_day, "thermal__own_gt_pmax_gw__available"] == 1).all()
    assert fr["thermal__own_nuclear_pmax_gw"].isna().all()
    assert (fr["thermal__own_nuclear_pmax_gw__available"] == 0).all()


def test_receipt_refuses_incomplete_daily_state_evidence(tmp_path: Path):
    source_plan = m.plan()
    _, _, cutoff = delivery_grid(DAY)
    daily, evidence = m.collect(Saturn(source_plan["specs"]), DAY,
                                now_utc=cutoff)
    evidence["independent_daily_state_checks"][m.SOURCES[0]].pop()
    with pytest.raises(ValueError, match="366 daily as-of states"):
        m.publish(tmp_path / "bundle", DAY, daily, evidence, source_plan,
                  now_utc=cutoff)


def test_incremental_state_cache_only_queries_thirteen_new_days(tmp_path):
    specs = m.plan()["specs"]
    cutoff = delivery_grid(DAY)[2]
    first_client = Saturn(specs)
    first, evidence = m.collect(first_client, DAY, now_utc=cutoff, cache=tmp_path)
    assert sum(call[0] == "state" for call in first_client.calls) == 366 * 13
    second_client = Saturn(specs)
    second, repeated = m.collect(second_client, DAY, now_utc=cutoff, cache=tmp_path)
    assert sum(call[0] == "block" for call in second_client.calls) == 13
    assert not any(call[0] == "state" for call in second_client.calls)
    assert evidence == repeated
    for name in m.SOURCES:
        pd.testing.assert_series_equal(first[name], second[name])
    next_day = (pd.Timestamp(DAY) + pd.Timedelta(days=1)).date().isoformat()
    third_client = Saturn(specs)
    m.collect(third_client, next_day, now_utc=delivery_grid(next_day)[2], cache=tmp_path)
    assert sum(call[0] == "state" for call in third_client.calls) == 13


def test_incremental_state_cache_rejects_tampering(tmp_path):
    specs = m.plan()["specs"]
    cutoff = delivery_grid(DAY)[2]
    _partial_cache(tmp_path, specs, cutoff)
    first_day = m.grids(DAY)[0][0].date().isoformat()
    path = tmp_path / m.SOURCES[0] / f"{first_day}.json"
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["state"]["value"] += 10.
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="cache checksum changed"):
        m.collect(Saturn(specs), DAY, now_utc=cutoff, cache=tmp_path)


def test_incremental_state_cache_rejects_supplier_revision(tmp_path):
    specs = m.plan()["specs"]
    cutoff = delivery_grid(DAY)[2]
    _partial_cache(tmp_path, specs, cutoff)

    class Revised(Saturn):
        def block_staircase(self, *args, **kwargs):
            result = super().block_staircase(*args, **kwargs)
            result.iloc[0] += 1.
            return result

    client = Revised(specs)
    with pytest.raises(ValueError, match="block staircase disagrees with D-1 state"):
        m.collect(client, DAY, now_utc=cutoff, cache=tmp_path)
    assert not any(call[0] == "state" for call in client.calls)


def _partial_cache(directory, specs, cutoff):
    """A real interrupted collection leaves its first verified state resumable."""
    class Interrupted(Saturn):
        def get(self, *args, **kwargs):
            if any(call[0] == "state" for call in self.calls):
                raise RuntimeError("fixture interruption after one state")
            return super().get(*args, **kwargs)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        m.collect(Interrupted(specs), DAY, now_utc=cutoff, cache=directory)
