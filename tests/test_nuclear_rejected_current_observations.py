"""Legacy canonical/EPEX admission remains strict outside EPEX-only reporting."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import run_multicountry_forecast as launcher
import chronos2_hourly.nuclear_reporting_refresh as refresh


TZ = "Europe/Paris"
TARGET = "synthetic_canonical_observations"
CONFIG = {"zones": {"FR": {"target": {"series": TARGET}}}}
DAYS = [("2026-09-19", 24), ("2026-03-29", 23), ("2026-10-25", 25)]


def install_sources(monkeypatch, *, day="2026-09-19", available_current=0,
                    failures=3, divergence=20.8, historical_gap=None, corrupt=None):
    expected, current = refresh._support(day, TZ)
    spec = SimpleNamespace(zone="FR", timezone=TZ, delivery_day=day,
                           history_contract=SimpleNamespace(target_series=TARGET))
    when = current[0] - pd.Timedelta(days=3)
    state = {"attempt": 0, "calls": [], "sleeps": [], "inputs": [],
             "allow_pending": [], "when": when, "canonical": None}

    def fetch(_client, series, _start, _end, _timezone, **kwargs):
        assert kwargs == {"naive_timezone": "UTC", "nocache": True, "live": True}
        assert series in {TARGET, launcher.POST_AUCTION_OBSERVED_SERIES_BY_ZONE["FR"]}
        state["calls"].append(series)
        if series == TARGET:
            state["attempt"] += 1
        # Changing both canonical and EPEX values detects reuse of a stale pair.
        values = pd.Series(float(100 * state["attempt"]), index=expected)
        if series == TARGET:
            values.loc[current[available_current:]] = np.nan
            if historical_gap == "previous_day":
                previous = expected[expected.tz_convert(TZ).date == pd.Timestamp(day).date() - pd.Timedelta(days=1)]
                values.loc[previous] = np.nan
            elif historical_gap == "internal":
                values.loc[current[0] - pd.Timedelta(hours=30)] = np.nan
        elif state["attempt"] <= failures:
            values.loc[when] += divergence
        if corrupt is not None:
            values = corrupt(series, values, when)
        if series == TARGET:
            state["canonical"] = values.copy(deep=True)
        state["inputs"].append((values, values.copy(deep=True)))
        return values

    original_reader = launcher._fetch_latest_observed_snapshot

    def traced_reader(*args, **kwargs):
        state["allow_pending"].append(kwargs.get("allow_pending_current_day", False))
        return original_reader(*args, **kwargs)

    monkeypatch.setattr(launcher, "fetch_saturn_series_from_client", fetch)
    monkeypatch.setattr(launcher, "_fetch_latest_observed_snapshot", traced_reader)
    monkeypatch.setattr(refresh.time, "sleep", state["sleeps"].append)
    return state, expected, current, spec


def assert_unmodified(state):
    for values, before in state["inputs"]:
        pd.testing.assert_series_equal(values, before)


def assert_rejected(source, count, *, divergence=20.8):
    audit = source["post_auction_fallback"]
    assert source["current_delivery_actual_reason"] == "post_auction_source_rejected"
    assert source["used_for_prediction"] is False
    assert audit["status"] == "rejected_divergent_optional_current_day"
    assert audit["validation_status"] == "rejected_divergence"
    assert audit["candidate_available_hours"] == count
    assert audit["validation_paired_hours"] >= 168
    assert audit["validation_maximum_absolute_difference_eur_mwh"] == pytest.approx(divergence)
    assert audit["rejection_diagnostic"]["maximum_absolute_difference_eur_mwh"] == pytest.approx(divergence)
    assert audit["applied_value_times_utc"] == []
    assert audit["used_for_prediction"] is False
    for field in ("applied_hours", "applied_current_hours", "applied_suffix_hours", "applied_internal_hours"):
        assert audit[field] == 0


@pytest.mark.parametrize("divergence", [20.8, .005])
def test_default_export_remains_strict_even_when_only_delivery_labels_are_missing(monkeypatch, divergence):
    state, expected, _current, spec = install_sources(monkeypatch, divergence=divergence)
    with pytest.raises(launcher.PostAuctionObservationDivergenceError):
        launcher._fetch_latest_observed_snapshot(object(), spec=spec, expected_index=expected)
    assert state["allow_pending"] == [False]
    assert len(state["calls"]) == 2 and state["sleeps"] == []
    assert_unmodified(state)


@pytest.mark.parametrize("day,hours", DAYS)
def test_explicit_pending_policy_rejects_all_discrepant_delivery_prices_including_dst(monkeypatch, day, hours):
    state, expected, current, spec = install_sources(monkeypatch, day=day)
    observed, source = launcher._fetch_latest_observed_snapshot(
        object(), spec=spec, expected_index=expected, allow_pending_current_day=True)
    assert len(current) == hours
    pd.testing.assert_series_equal(observed, state["canonical"].rename("actual"))
    assert observed.loc[current].isna().all()
    assert observed.index.equals(expected) and observed.index.is_unique
    assert_rejected(source, hours)
    assert source["post_auction_fallback"]["rejection_diagnostic"]["timestamp_utc"] == str(state["when"])
    assert_unmodified(state)


@pytest.mark.parametrize("day,hours", DAYS)
def test_legacy_reader_retries_and_preserves_rejection_evidence_for_partial_delivery(
    monkeypatch, capsys, day, hours,
):
    # The current reporting writer uses EPEX directly. Keep exercising the
    # legacy pair reader without routing its output into an EPEX publication.
    # Complete-day masking, including DST, remains covered by refresh tests.
    state, expected, current, spec = install_sources(monkeypatch, day=day, available_current=6)
    observed, source = refresh._fetch_observed(object(), spec=spec, expected_index=expected)
    assert len(current) == hours
    pd.testing.assert_series_equal(observed, state["canonical"].rename("actual"))
    assert observed.loc[current[:6]].eq(300).all()
    assert observed.loc[current[6:]].isna().all()
    assert observed.loc[expected.difference(current)].eq(300.0).all()
    assert_rejected(source, hours - 6)
    retry = source["observation_pair_read"]
    assert retry["attempts"] == retry["divergence_failures_count"] == 3
    assert [record["attempt"] for record in retry["failures"]] == [1, 2, 3]
    assert retry["optional_current_day_fallback_rejected"] is True
    assert retry["retry_delays_seconds"] == state["sleeps"] == [2, 4]
    assert state["allow_pending"] == [False, False, True]
    assert state["calls"] == [TARGET, launcher.POST_AUCTION_OBSERVED_SERIES_BY_ZONE["FR"]] * 3
    final = retry["failures"][-1]
    assert {k: v for k, v in final.items() if k != "attempt"} == source["post_auction_fallback"]["rejection_diagnostic"]
    assert pd.Timestamp(final["extracted_at_utc"]) == pd.Timestamp(source["extracted_at_utc"])
    assert "prix realises du jour non valides" in capsys.readouterr().out
    assert_unmodified(state)


@pytest.mark.parametrize("day,hours", DAYS)
@pytest.mark.parametrize("allow_pending", [False, True])
def test_coherent_post_auction_prices_still_complete_delivery_without_rejection(monkeypatch, day, hours, allow_pending):
    state, expected, current, spec = install_sources(monkeypatch, day=day, failures=0)
    observed, source = launcher._fetch_latest_observed_snapshot(
        object(), spec=spec, expected_index=expected, allow_pending_current_day=allow_pending)
    assert observed.loc[current].eq(100).all() and len(current) == hours
    assert source["post_auction_fallback"]["applied_hours"] == hours
    assert source["post_auction_fallback"]["status"] == "complete_missing_suffix"
    assert "current_delivery_actual_reason" not in source
    assert "rejection_diagnostic" not in source["post_auction_fallback"]
    assert_unmodified(state)


@pytest.mark.parametrize("gap", ["previous_day", "internal"])
def test_historical_gaps_remain_fatal_after_three_pairs_even_with_optional_delivery_policy(monkeypatch, gap):
    state, expected, _current, spec = install_sources(monkeypatch, historical_gap=gap)
    with pytest.raises(launcher.PostAuctionObservationDivergenceError) as captured:
        refresh._fetch_observed(object(), spec=spec, expected_index=expected)
    assert state["allow_pending"] == [False, False, True]
    assert len(state["calls"]) == 6 and state["sleeps"] == [2, 4]
    assert captured.value.diagnostic["maximum_absolute_difference_eur_mwh"] == pytest.approx(20.8)
    assert_unmodified(state)


@pytest.mark.parametrize("which", ["canonical", "fallback"])
@pytest.mark.parametrize("fault", ["timezone", "duplicate"])
def test_opt_in_does_not_suppress_invalid_source_timeline(monkeypatch, which, fault):
    def corrupt(series, values, _when):
        is_canonical = series == TARGET
        if is_canonical != (which == "canonical"):
            return values
        if fault == "timezone":
            values.index = values.index.tz_localize(None)
            return values
        return pd.concat([values, values.iloc[[0]]])

    state, expected, _current, spec = install_sources(monkeypatch, corrupt=corrupt)
    with pytest.raises(ValueError, match="sans timezone|dupliquees"):
        launcher._fetch_latest_observed_snapshot(object(), spec=spec, expected_index=expected,
                                                allow_pending_current_day=True)
    assert state["sleeps"] == []


@pytest.mark.parametrize("fault", ["nonfinite_overlap", "transport"])
def test_opt_in_does_not_suppress_invalid_overlap_or_source_errors(monkeypatch, fault):
    def corrupt(series, values, when):
        if series != TARGET:
            if fault == "transport":
                raise RuntimeError("synthetic transport failure")
            values.loc[when] = np.inf
        return values

    state, expected, _current, spec = install_sources(monkeypatch, corrupt=corrupt)
    error, message = (RuntimeError, "synthetic transport failure") if fault == "transport" else (ValueError, "recouvrement insuffisant")
    with pytest.raises(error, match=message):
        launcher._fetch_latest_observed_snapshot(object(), spec=spec, expected_index=expected,
                                                allow_pending_current_day=True)
    assert state["sleeps"] == []


def test_transient_divergence_resolves_on_second_fresh_pair_without_rejecting_delivery(monkeypatch):
    state, expected, current, spec = install_sources(monkeypatch, failures=1)
    observed, source = refresh._fetch_observed(object(), spec=spec, expected_index=expected)
    assert observed.eq(200).all() and observed.loc[current].notna().all()
    assert state["allow_pending"] == [False, False]
    assert len(state["calls"]) == 4 and state["sleeps"] == [2]
    retry = source["observation_pair_read"]
    assert retry["attempts"] == 2 and retry["divergence_failures_count"] == 1
    assert retry["optional_current_day_fallback_rejected"] is False
    assert source["post_auction_fallback"]["applied_current_hours"] == len(current)
    assert "current_delivery_actual_reason" not in source
    assert_unmodified(state)
