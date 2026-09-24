"""Synthetic report-only observations: an unused EPEX fallback proves nothing.

No network, model execution, or production-data mutation is used by these tests.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import run_multicountry_forecast as launcher
import chronos2_hourly.nuclear_reporting_refresh as refresh


DAY = "2026-09-19"
TZ = "Europe/Paris"
TARGET = "synthetic_canonical_target"


def inputs():
    expected = pd.date_range("2026-09-10", "2026-09-20", freq="h", inclusive="left", tz=TZ).tz_convert("UTC")
    current = expected[expected.tz_convert(TZ).date == pd.Timestamp(DAY).date()]
    canonical = pd.Series(100.0 + np.arange(len(expected)) / 8, index=expected)
    spec = SimpleNamespace(zone="FR", timezone=TZ, delivery_day=DAY,
                           history_contract=SimpleNamespace(target_series=TARGET))
    return expected, current, canonical, spec


def unavailable(values, index, representation):
    values = values.copy()
    if representation == "omitted":
        return values.drop(index)
    values.loc[index] = {"nan": np.nan, "inf": np.inf, "negative_inf": -np.inf}[representation]
    return values


def install_sources(monkeypatch, canonical, fallback):
    calls, returned = [], []

    def fetch(_client, series, _start, _end, _timezone, **kwargs):
        assert kwargs == {"naive_timezone": "UTC", "nocache": True, "live": True}
        assert series in {TARGET, launcher.POST_AUCTION_OBSERVED_SERIES_BY_ZONE["FR"]}
        calls.append(series)
        value = canonical if series == TARGET else fallback
        returned.append((value, value.copy(deep=True)))
        return value

    monkeypatch.setattr(launcher, "fetch_saturn_series_from_client", fetch)
    return calls, returned


def assert_unused(source):
    fallback = source["post_auction_fallback"]
    assert fallback["status"] == "not_applied_no_available_observations"
    assert fallback["validation_status"] == "not_required_no_values_applied"
    assert fallback["validation_paired_hours"] == 0
    assert fallback["validation_maximum_absolute_difference_eur_mwh"] is None
    assert fallback["applied_value_times_utc"] == []
    for key in ("applied_hours", "applied_current_hours", "applied_suffix_hours", "applied_internal_hours"):
        assert fallback[key] == 0
    assert source["used_for_prediction"] is False
    assert fallback["used_for_prediction"] is False


@pytest.mark.parametrize("divergence", [20.8, 0.005])
@pytest.mark.parametrize("canonical_form,fallback_form", [
    ("omitted", "omitted"), ("nan", "nan"),
    ("omitted", "nan"), ("nan", "omitted"),
    ("inf", "negative_inf"), ("negative_inf", "inf"),
])
def test_unpublished_delivery_keeps_canonical_despite_unused_historical_divergence(
    monkeypatch, divergence, canonical_form, fallback_form,
):
    expected, current, complete, spec = inputs()
    canonical = unavailable(complete, current, canonical_form)
    fallback = unavailable(complete, current, fallback_form)
    fallback.loc[pd.Timestamp("2026-09-13T15:00Z")] += divergence
    calls, returned = install_sources(monkeypatch, canonical, fallback)

    observed, source = launcher._fetch_latest_observed_snapshot(object(), spec=spec, expected_index=expected)

    pd.testing.assert_series_equal(observed, canonical.rename("actual"))
    assert not np.isfinite(observed.reindex(current).to_numpy()).any()
    assert_unused(source)
    assert source["post_auction_fallback"]["missing_canonical_current_hours"] == 24
    assert len(calls) == 2
    for original, before in returned:
        pd.testing.assert_series_equal(original, before)


@pytest.mark.parametrize("divergence", [20.8, 0.005])
def test_same_partial_delivery_coverage_never_replaces_finite_canonical_prices(monkeypatch, divergence):
    expected, current, complete, spec = inputs()
    canonical = unavailable(complete, current[6:], "nan")
    fallback = unavailable(complete, current[6:], "omitted")
    fallback.loc[pd.Timestamp("2026-09-13T15:00Z")] += divergence
    # Finite fallback values for already-known hours are not usable additions.
    fallback.loc[current[:6]] += 25
    install_sources(monkeypatch, canonical, fallback)

    observed, source = launcher._fetch_latest_observed_snapshot(object(), spec=spec, expected_index=expected)

    pd.testing.assert_series_equal(observed, canonical.rename("actual"))
    assert observed.reindex(current).notna().sum() == 6
    assert_unused(source)
    assert source["post_auction_fallback"]["missing_canonical_current_hours"] == 18


@pytest.mark.parametrize("divergence", [20.8, 0.005])
@pytest.mark.parametrize("addition", ["delivery", "historical"])
def test_one_usable_fallback_price_still_requires_historical_equivalence(monkeypatch, divergence, addition):
    expected, current, complete, spec = inputs()
    canonical = unavailable(complete, current, "nan")
    fallback = unavailable(complete, current, "nan")
    when = current[0] if addition == "delivery" else pd.Timestamp("2026-09-17T12:00Z")
    canonical.loc[when] = np.nan
    fallback.loc[when] = complete.loc[when]
    fallback.loc[pd.Timestamp("2026-09-13T15:00Z")] += divergence
    _, returned = install_sources(monkeypatch, canonical, fallback)

    with pytest.raises(launcher.PostAuctionObservationDivergenceError, match="source post-enchere diverge"):
        launcher._fetch_latest_observed_snapshot(object(), spec=spec, expected_index=expected)

    for original, before in returned:
        pd.testing.assert_series_equal(original, before)


def test_unfillable_historical_gap_remains_missing_and_refresh_refuses_before_other_sources(tmp_path, monkeypatch):
    expected, current = refresh._support(DAY, TZ)
    complete = pd.Series(100.0, index=expected)
    hole = pd.Timestamp("2026-09-17T12:00Z")
    canonical = unavailable(complete, current.union(pd.DatetimeIndex([hole])), "nan")
    fallback = unavailable(complete, current.union(pd.DatetimeIndex([hole])), "omitted")
    fallback.loc[pd.Timestamp("2026-09-13T15:00Z")] += 20.8
    spec = SimpleNamespace(zone="FR", timezone=TZ, delivery_day=DAY,
                           history_contract=SimpleNamespace(target_series=TARGET))
    calls, _ = install_sources(monkeypatch, canonical, fallback)
    observed, source = launcher._fetch_latest_observed_snapshot(object(), spec=spec, expected_index=expected)
    pd.testing.assert_series_equal(observed, canonical.rename("actual"))
    assert pd.isna(observed.loc[hole])
    assert_unused(source)

    def unexpected(*_args, **_kwargs):
        pytest.fail("A missing historical actual must fail before Storm, client creation or retry sleep.")

    monkeypatch.setattr(refresh, "fetch_native_dashboard_snapshot", unexpected)
    monkeypatch.setattr(refresh, "create_saturn_client", unexpected)
    monkeypatch.setattr(refresh.time, "sleep", unexpected)
    from chronos2_hourly.reporting_observations import epex_reporting_identity
    epex_reads = []
    def epex_fetch(client, **kwargs):
        epex_reads.append(kwargs)
        return fallback.copy(), {
            **epex_reporting_identity("FR", TZ),
            "extracted_at_utc": str(kwargs["extracted_at_utc"]),
        }
    monkeypatch.setattr(refresh, "_fetch_epex_observed", epex_fetch)
    config = {"zones": {"FR": {"target": {"series": TARGET}}}}
    output = tmp_path / "report_only"
    with pytest.raises(refresh.NuclearReportingRefreshError, match="observations historiques recentes incompletes"):
        refresh.refresh_nuclear_reporting_sources(config, "FR", TZ, DAY, output, client=object())
    assert len(calls) == 2  # The legacy pair is not read by EPEX-only reporting.
    assert len(epex_reads) == 1
    assert not output.exists()


@pytest.mark.parametrize("which", ["canonical", "fallback"])
@pytest.mark.parametrize("invalid", ["naive", "duplicate"])
def test_unused_fallback_does_not_bypass_source_timeline_validation(monkeypatch, which, invalid):
    expected, current, complete, spec = inputs()
    sources = {"canonical": unavailable(complete, current, "nan"),
               "fallback": unavailable(complete, current, "omitted")}
    bad = sources[which]
    if invalid == "naive":
        bad = bad.copy()
        bad.index = bad.index.tz_localize(None)
    else:
        bad = pd.concat([bad, bad.iloc[[0]]])
    sources[which] = bad
    install_sources(monkeypatch, sources["canonical"], sources["fallback"])
    with pytest.raises(ValueError, match="sans timezone|dupliquees"):
        launcher._fetch_latest_observed_snapshot(object(), spec=spec, expected_index=expected)


@pytest.mark.parametrize("invalid", ["naive", "duplicate"])
def test_invalid_requested_timeline_fails_before_fetch(monkeypatch, invalid):
    expected, _current, _complete, spec = inputs()
    expected = expected.tz_localize(None) if invalid == "naive" else expected.append(expected[:1])

    def unexpected(*_args, **_kwargs):
        pytest.fail("An invalid requested timeline must fail before source access.")

    monkeypatch.setattr(launcher, "fetch_saturn_series_from_client", unexpected)
    with pytest.raises(ValueError, match="timezone-aware|dupliquee"):
        launcher._fetch_latest_observed_snapshot(object(), spec=spec, expected_index=expected)
