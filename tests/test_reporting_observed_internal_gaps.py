from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import run_multicountry_forecast as launcher


def _inputs():
    expected = pd.date_range("2026-08-31T22:00Z", "2026-09-10T21:00Z", freq="h")
    complete = pd.Series(np.arange(len(expected), dtype=float) / 8, index=expected)
    internal = expected[(expected >= "2026-09-07T22:00Z") & (expected < "2026-09-08T22:00Z")]
    suffix = expected[expected >= "2026-09-09T22:00Z"]
    canonical = complete.drop(internal.union(suffix))
    spec = SimpleNamespace(
        zone="DE", timezone="Europe/Berlin", delivery_day="2026-09-10",
        history_contract=SimpleNamespace(target_series="canonical_target"),
    )
    return expected, complete, internal, suffix, canonical, spec


def _fetch(monkeypatch, canonical, fallback):
    calls = []

    def fetch(_client, series, start, end, timezone, **kwargs):
        calls.append((series, kwargs))
        assert kwargs == {"naive_timezone": "UTC", "nocache": True, "live": True}
        values = canonical if series == "canonical_target" else fallback
        return values.loc[(values.index >= start) & (values.index <= end)]

    monkeypatch.setattr(launcher, "fetch_saturn_series_from_client", fetch)
    return calls


def test_latest_observed_fills_internal_day_and_suffix_with_precise_report_only_provenance(monkeypatch):
    expected, complete, internal, suffix, canonical, spec = _inputs()
    fallback = complete.copy()
    canonical_hour = pd.Timestamp("2026-09-09T12:00Z")
    fallback.loc[canonical_hour] += 5e-10
    calls = _fetch(monkeypatch, canonical, fallback)

    observed, source = launcher._fetch_latest_observed_snapshot(
        object(), spec=spec, expected_index=expected,
        extracted_at_utc=pd.Timestamp("2026-09-10T08:00Z"),
    )

    pd.testing.assert_series_equal(observed.reindex(expected), complete.rename("actual"))
    assert observed.loc[canonical_hour] == canonical.loc[canonical_hour]
    audit = source["post_auction_fallback"]
    assert audit["status"] == "complete_missing_hours"
    assert audit["validation_paired_hours"] == 192
    assert audit["missing_canonical_internal_hours"] == audit["applied_internal_hours"] == 24
    assert audit["missing_canonical_suffix_hours"] == audit["applied_suffix_hours"] == 24
    assert audit["applied_hours"] == 48
    assert pd.DatetimeIndex(audit["applied_value_times_utc"]).equals(internal.union(suffix))
    assert source["used_for_prediction"] is False and audit["used_for_prediction"] is False
    assert len(calls) == 2


@pytest.mark.parametrize("when", ["2026-09-07T12:00Z", "2026-09-09T12:00Z"])
def test_reporting_internal_repair_rejects_divergence_before_and_after_hole(monkeypatch, when):
    expected, complete, _internal, _suffix, canonical, spec = _inputs()
    complete.loc[pd.Timestamp(when)] += 0.01
    _fetch(monkeypatch, canonical, complete)
    with pytest.raises(ValueError, match="source post-enchere diverge"):
        launcher._fetch_latest_observed_snapshot(object(), spec=spec, expected_index=expected)


def test_reporting_internal_repair_keeps_unavailable_hours_missing(monkeypatch):
    expected, complete, internal, _suffix, canonical, spec = _inputs()
    _fetch(monkeypatch, canonical, complete.drop(internal[0]))
    observed, source = launcher._fetch_latest_observed_snapshot(object(), spec=spec, expected_index=expected)
    assert np.isnan(observed.reindex(expected).loc[internal[0]])
    assert source["post_auction_fallback"]["status"] == "incomplete_missing_hours"
    assert source["post_auction_fallback"]["applied_hours"] == 47


def test_reporting_internal_repair_fetches_authoritative_validation_prefix(monkeypatch):
    expected, complete, _internal, _suffix, canonical, spec = _inputs()
    calls = _fetch(monkeypatch, canonical, complete)
    short_expected = expected[expected >= "2026-09-07T00:00Z"]
    observed, source = launcher._fetch_latest_observed_snapshot(object(), spec=spec, expected_index=short_expected)
    pd.testing.assert_series_equal(observed.reindex(short_expected), complete.reindex(short_expected).rename("actual"))
    assert len(calls) == 3
    assert [series for series, _kwargs in calls] == [
        "canonical_target", "canonical_target", launcher.POST_AUCTION_OBSERVED_SERIES_BY_ZONE["DE"]
    ]
    assert source["post_auction_fallback"]["validation_paired_hours"] == 192


def test_reporting_rejects_nonfinite_validation_pairs(monkeypatch):
    expected, complete, _internal, _suffix, canonical, spec = _inputs()
    complete.loc[pd.Timestamp("2026-09-07T12:00Z")] = np.inf
    _fetch(monkeypatch, canonical, complete)
    with pytest.raises(ValueError, match="recouvrement insuffisant"):
        launcher._fetch_latest_observed_snapshot(object(), spec=spec, expected_index=expected)


def test_reporting_replaces_nonfinite_internal_observation(monkeypatch):
    expected, complete, _internal, _suffix, _canonical, spec = _inputs()
    canonical = complete.copy()
    when = pd.Timestamp("2026-09-08T12:00Z")
    canonical.loc[when] = np.inf
    _fetch(monkeypatch, canonical, complete)
    observed, source = launcher._fetch_latest_observed_snapshot(object(), spec=spec, expected_index=expected)
    assert observed.loc[when] == complete.loc[when]
    assert source["post_auction_fallback"]["applied_internal_hours"] == 1

