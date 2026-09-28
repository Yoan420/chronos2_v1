import copy

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_lagged_exchange_features as m


def payload(country, starts, step=15):
    fields = ("france", "netherlands", "belgium", "sum") if country == "de" else ("cross_border_electricity_trading",)
    return {"schema_version": "2.0", "country": country,
            "endpoint": "cbpf" if country == "de" else "public_power",
            "unit": "GW" if country == "de" else "MW", "timezone": m.TIMEZONES[country.upper()],
            "interval_minutes": step, "resolution": "PT15M" if step == 15 else "PT1H",
            "attributes": {"sign_convention": "positive = import, negative = export"} if country == "de" else {},
            "series": [{"id": c} for c in fields],
            "data": [{"timestamp": t.tz_convert(m.TIMEZONES[country.upper()]).isoformat(),
                      "values": {c: (-1. if j == 0 else float(j))*(1 if country == "de" else 1000)
                                 for j, c in enumerate(fields)}} for t in starts]}


def pair(target):
    hours = target-pd.Timedelta(hours=48)
    quarters = pd.DatetimeIndex([t+pd.Timedelta(minutes=q) for t in hours for q in (0, 15, 30, 45)])
    return payload("de", quarters), payload("fr", quarters)


@pytest.mark.parametrize("day,hours", [("2025-10-26", 25), ("2026-03-29", 23),
                                      ("2025-10-27", 24), ("2026-03-30", 24)])
def test_exact_lag48_dst_civil_cutoff_and_signs(day, hours):
    d = pd.Timestamp(day)
    target = pd.date_range(d, d+pd.Timedelta(days=1), freq="h", inclusive="left", tz="Europe/Paris").tz_convert("UTC")
    de, fr = pair(target)
    before = copy.deepcopy((de, fr))
    hourly = m.exchange_hourly([de], [fr])
    outputs, audit = m.build_features(hourly, target)
    assert len(target) == hours and all(f.shape == (hours, 10) for f in outputs.values())
    for f in outputs.values():
        assert tuple(f.columns) == m.COLUMNS
        assert (f[m.LEVELS[0]] == -1).all() and (f[m.LEVELS[4]] == -1).all()
        assert (f.filter(like="__available") == 1).all().all()
    assert pd.DatetimeIndex(audit["rows"].source_interval_start_utc).equals(target-pd.Timedelta(hours=48))
    assert (audit["rows"].nominal_publication_utc <= audit["rows"].origin_utc).all()
    assert audit["minimum_nominal_margin_hours"] >= 6
    assert (de, fr) == before


def test_fr_divide_before_mean_exact_and_real_zero():
    q = pd.date_range("2025-02-01T00Z", periods=4, freq="15min")
    de, fr = payload("de", q), payload("fr", q)
    vals = [1e16, 1.1, -1e16, 1.3]
    for row, value in zip(fr["data"], vals): row["values"]["cross_border_electricity_trading"] = value
    for row in de["data"]: row["values"]["belgium"] = 0.
    h = m.exchange_hourly([de], [fr])
    expected = pd.Series(np.array(vals, dtype=float)/1000., index=q).resample("h").mean().iloc[0]
    assert h["fr_commercial_gw"].iloc[0] == expected
    out, _ = m.build_features(h, pd.DatetimeIndex([q[0]+pd.Timedelta(hours=48)]))
    assert out["DE"][m.LEVELS[2]].iloc[0] == 0 and out["DE"][m.LEVELS[2]+"__available"].iloc[0] == 1


def test_fr_declared60_with_four_quarters_and_hourly():
    q = pd.date_range("2025-01-01T00Z", periods=8, freq="15min")
    fr = payload("fr", pd.DatetimeIndex([q[0], *q[4:]]), step=60)
    h = m.exchange_hourly([payload("de", q)], [fr])
    assert h["fr_commercial_gw"].eq(-1).all()
    assert len(h.attrs["interval_metadata"]["FR"]["corrections"]) == 1
    assert h.attrs["interval_metadata"]["FR"]["effective_hour_counts"] == {"60": 1, "15": 1}


def test_incomplete_declared60_subhour_is_rejected():
    q = pd.date_range("2025-01-01T00Z", periods=3, freq="15min")
    with pytest.raises(ValueError, match="Incomplete subhour"):
        m.exchange_hourly([], [payload("fr", q, step=60)])


def test_missing_quarter_and_null_not_promoted_to_hour():
    q = pd.date_range("2025-01-01T00Z", periods=8, freq="15min")
    de, fr = payload("de", q), payload("fr", q)
    de["data"].pop(1)
    fr["data"][4]["values"]["cross_border_electricity_trading"] = None
    h = m.exchange_hourly([de], [fr])
    assert np.isnan(h.de_fr_gw.iloc[0]) and h.de_fr_gw__coverage.iloc[0] == .75
    assert np.isnan(h.fr_commercial_gw.iloc[1]) and h.fr_commercial_gw__coverage.iloc[1] == .75
    idx = pd.date_range(q[0]+pd.Timedelta(hours=48), periods=3, freq="h")
    out, audit = m.build_features(h, idx)
    assert out["DE"][m.LEVELS[0]+"__available"].tolist() == [0, 1, 0]
    assert out["DE"].iloc[2].filter(like="__available").eq(0).all()
    assert audit["rows"].iloc[2].filter(like="__coverage").eq(0).all()


@pytest.mark.parametrize("kind", ["wrong_unit", "wrong_sign", "naive", "offgrid", "conflict", "mixed", "inf", "duplicate_schema"])
def test_source_ambiguities_rejected(kind):
    q = pd.date_range("2025-01-01T00Z", periods=4, freq="15min")
    de, fr = payload("de", q), payload("fr", q)
    if kind == "wrong_unit": de["unit"] = "MW"
    elif kind == "wrong_sign": de["attributes"]["sign_convention"] = "positive = export"
    elif kind == "naive": de["data"][0]["timestamp"] = "2025-01-01T00:00"
    elif kind == "offgrid": de["data"][0]["timestamp"] = "2025-01-01T00:01Z"
    elif kind == "inf": de["data"][0]["values"]["france"] = float("inf")
    elif kind == "duplicate_schema": de["series"].append(de["series"][0])
    elif kind == "conflict":
        dupe = copy.deepcopy(de["data"][0]); dupe["values"]["france"] = 7; de["data"].append(dupe)
    else:
        with pytest.raises(ValueError, match="resolutions"):
            m.exchange_hourly([de], [fr, payload("fr", q[:1], step=60)])
        return
    with pytest.raises(ValueError): m.exchange_hourly([de], [fr])


def test_same_overlaps_allowed_and_conflicting_revision_refused():
    q = pd.date_range("2025-01-01T00Z", periods=4, freq="15min")
    de, fr = payload("de", q), payload("fr", q)
    assert len(m.exchange_hourly([de, de], [fr, fr])) == 1
    other = copy.deepcopy(de); other["data"][0]["values"]["france"] += 1
    with pytest.raises(ValueError, match="revisions"): m.exchange_hourly([de, other], [fr])


def test_future_mutation_no_effect_and_origins_rejected():
    target = pd.date_range("2025-01-03T00Z", periods=24, freq="h")
    de, fr = pair(target)
    h = m.exchange_hourly([de], [fr]); original, _ = m.build_features(h, target)
    future = payload("de", pd.date_range("2025-01-04T00Z", periods=4, freq="15min"))
    for row in future["data"]: row["values"]["france"] = 1e9
    changed, _ = m.build_features(m.exchange_hourly([de, future], [fr]), target)
    pd.testing.assert_frame_equal(original["DE"], changed["DE"], check_exact=True)
    with pytest.raises(ValueError, match="Origin later"):
        m.build_features(h, target, pd.DatetimeIndex([pd.Timestamp("2025-01-03T00Z")]*len(target)))
    with pytest.raises(ValueError, match="publication"):
        m.build_features(h, target, target-pd.Timedelta(hours=47))


def test_empty_and_append_preserve_source_exact():
    target = pd.date_range("2025-01-03T00Z", periods=3, freq="h")
    extras, _ = m.build_features(m.exchange_hourly([], []), target)
    base = {z: pd.DataFrame({"x": np.arange(3, dtype=np.int64)}, index=target) for z in m.ZONES}
    combined = m.append_features(base, extras)
    for z in m.ZONES:
        pd.testing.assert_frame_equal(combined[z][["x"]], base[z], check_exact=True)
        assert combined[z].filter(like="__available").eq(0).all().all()
