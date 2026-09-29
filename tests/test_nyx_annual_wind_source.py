"""Exact historical NL spring substitution, with no broader gap filling."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_annual_saturn_source as source
from chronos2_hourly import nyx_annual_wind_source as wind


def native(day):
    expected = source.grid(day)
    return pd.Series(np.arange(len(expected), dtype=float) + 1, index=expected, name=wind.ALIAS)


@pytest.mark.parametrize("day", ["2025-03-30", "2026-03-29"])
@pytest.mark.parametrize("kind", ["absent", "nan"])
def test_only_authorized_product_uses_same_origin_component(day, kind):
    values, hour = native(day), pd.Timestamp(day + "T02:00:00Z")
    supplied = values.drop(hour) if kind == "absent" else values.mask(values.index == hour)
    calls = []
    def get(name, **kwargs):
        calls.append((name, kwargs))
        return pd.Series([4321., 9999.], index=pd.DatetimeIndex([hour, hour + pd.Timedelta(hours=1)]))
    complete, audit = wind.normalize_nl_wind_profile(SimpleNamespace(get=get), supplied,
        day, source.grid(day), source.cutoff(day))
    assert len(complete) == 23 and complete.loc[hour] == 4.321
    pd.testing.assert_series_equal(complete.drop(hour), values.drop(hour), check_freq=False)
    assert calls == [(wind.COMPONENT, {"from_value_date": hour, "to_value_date": hour + pd.Timedelta(hours=1),
                                     "revision_date": source.cutoff(day), "nocache": True})]
    assert audit[0]["native_gap_kind"] == kind and audit[0]["query_cutoff_utc"].endswith("T07:00:00+00:00")
    assert audit[0]["production_pit_evidence"] is False
    wind.verify_nl_wind_substitutions(complete, audit, day, source.cutoff(day))


@pytest.mark.parametrize("case", ["complete_native", "outer_revision", "other_day"])
def test_native_priority_and_scope_never_query_component(case):
    day = "2025-03-31" if case == "other_day" else "2025-03-30"
    values = native(day)
    revision = source.cutoff("2026-09-30") if case == "outer_revision" else source.cutoff(day)
    if case != "complete_native":
        values = values.drop(pd.Timestamp(day + "T02:00:00Z"))
    result, audit = wind.normalize_nl_wind_profile(SimpleNamespace(get=lambda *a, **k: pytest.fail("out-of-scope request")),
        values, day, source.grid(day), revision)
    pd.testing.assert_series_equal(result, values)
    assert audit == []


@pytest.mark.parametrize("case", ["extra_gap", "wrong_gap", "other_nan", "infinite", "negative", "duplicate"])
def test_other_native_defects_are_never_repaired(case):
    day, hour = "2025-03-30", pd.Timestamp("2025-03-30T02:00Z")
    values = native(day)
    if case == "extra_gap":
        values = values.drop([hour, hour + pd.Timedelta(hours=1)])
    elif case == "wrong_gap":
        values = values.drop(hour + pd.Timedelta(hours=1))
    elif case == "other_nan":
        values.loc[hour + pd.Timedelta(hours=1)] = np.nan
        values = values.drop(hour)
    elif case in ("infinite", "negative"):
        values.loc[hour] = np.inf if case == "infinite" else -1.
    else:
        values = pd.concat([values, values.iloc[[0]]])
    with pytest.raises(ValueError):
        wind.normalize_nl_wind_profile(SimpleNamespace(get=lambda *a, **k: pytest.fail("invalid native request")),
            values, day, source.grid(day), source.cutoff(day))


@pytest.mark.parametrize("case", ["naive", "duplicate", "nan", "infinite", "negative", "wrong_hour"])
def test_component_requires_one_physical_finite_nonnegative_mw_value(case):
    day, hour = "2025-03-30", pd.Timestamp("2025-03-30T02:00Z")
    value = {"nan": np.nan, "infinite": np.inf, "negative": -1.}.get(case, 4321.)
    stamp = hour.tz_localize(None) if case == "naive" else hour + pd.Timedelta(hours=1) if case == "wrong_hour" else hour
    raw = pd.Series([value], index=pd.DatetimeIndex([stamp]))
    if case == "duplicate":
        raw = pd.concat([raw, raw])
    with pytest.raises(ValueError):
        wind.normalize_nl_wind_profile(SimpleNamespace(get=lambda *a, **k: raw), native(day).drop(hour),
            day, source.grid(day), source.cutoff(day))


@pytest.mark.parametrize("case", ["origin", "component", "scale", "value", "native_hash", "native_missing"])
def test_substitution_proof_tampering_is_rejected(case):
    day, hour = "2025-03-30", pd.Timestamp("2025-03-30T02:00Z")
    values, proof = wind.normalize_nl_wind_profile(SimpleNamespace(get=lambda *a, **k:
        pd.Series([4321.], index=pd.DatetimeIndex([hour]))), native(day).drop(hour), day, source.grid(day), source.cutoff(day))
    proof = copy.deepcopy(proof)
    key, value = {"origin": ("query_cutoff_utc", source.cutoff("2026-09-30").isoformat()),
        "component": ("fallback_series", "another.wind.series"), "scale": ("value_scale", 1.),
        "value": ("scaled_value_gw", 4321.), "native_hash": ("native_values_sha256", "0" * 64),
        "native_missing": ("native_missing", False)}[case]
    proof[0][key] = value
    with pytest.raises(ValueError):
        wind.verify_nl_wind_substitutions(values, proof, day, source.cutoff(day))
    with pytest.raises(ValueError, match="own historical origin"):
        wind.verify_nl_wind_substitutions(values, proof, day, source.cutoff("2026-09-30"))


def test_partial_day_preserves_substitution_and_rejects_changed_cutoff(monkeypatch, tmp_path):
    day, outer, hour = "2025-03-30", "2026-09-30", pd.Timestamp("2025-03-30T02:00Z")
    def fetch(client, name, *args, **kwargs):
        if name == source.specs()["fr_residual_load_fcst"]["series"]:
            raise RuntimeError("Unavailable FR")
        result = native(day)
        return result.drop(hour) if name == source.specs()[wind.ALIAS]["series"] else result
    monkeypatch.setattr(source, "fetch_saturn_series_from_client", fetch)
    monkeypatch.setattr(source.time, "sleep", lambda _: None)
    client = SimpleNamespace(get=lambda *a, **k: pd.Series([4321.], index=pd.DatetimeIndex([hour])))
    with pytest.raises(source.SaturnSourceError):
        source._sync_profile_day(day, outer, tmp_path, lambda: client)
    path = tmp_path / "profiles_per_series_v2" / outer / day / "series" / f"{wind.ALIAS}.json"
    saved = json.loads(path.read_text())
    assert len(saved["evidence"]["source_substitutions"]) == 1
    saved["evidence"]["source_substitutions"][0]["query_cutoff_utc"] = source.cutoff(outer).isoformat()
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="substitution"):
        source._sync_profile_day(day, outer, tmp_path, lambda: client)
