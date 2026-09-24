"""Synthetic report-only source revisions; no network or model execution."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import run_multicountry_forecast as launcher
import chronos2_hourly.nuclear_reporting_refresh as refresh


@pytest.fixture
def changing_sources(monkeypatch):
    expected = pd.date_range('2026-08-31T22:00Z', '2026-09-12T21:00Z', freq='h')
    missing = pd.date_range('2026-09-08T22:00Z', periods=24, freq='h')
    divergence_hour = pd.Timestamp('2026-09-07T11:00Z')
    spec = SimpleNamespace(zone='DE', timezone='Europe/Berlin', delivery_day='2026-09-12',
                           history_contract=SimpleNamespace(target_series='canonical_target'))
    state = {'attempt': 0, 'failures': 1, 'calls': [], 'sleeps': [], 'inputs': []}

    def fetch(client, series, start, end, timezone, **kwargs):
        assert kwargs == {'naive_timezone': 'UTC', 'nocache': True, 'live': True}
        if series == 'canonical_target':
            state['attempt'] += 1
        state['calls'].append(series)
        # Each read pair has genuinely new canonical finite values. Reusing
        # the first canonical curve and retrying only EPEX cannot pass.
        values = pd.Series(float(state['attempt'] * 100), index=expected)
        if series == 'canonical_target':
            values = values.drop(missing)
        elif state['attempt'] <= state['failures']:
            values.loc[divergence_hour] += 7.89
        values = values.loc[(values.index >= start) & (values.index <= end)]
        state['inputs'].append((values, values.copy()))
        return values

    monkeypatch.setattr(launcher, 'fetch_saturn_series_from_client', fetch)
    monkeypatch.setattr(refresh.time, 'sleep', state['sleeps'].append)
    return state, expected, spec, divergence_hour


@pytest.mark.parametrize('failures', [1, 2])
def test_transient_divergence_rereads_both_sources_and_audits_success(changing_sources, failures):
    state, expected, spec, when = changing_sources
    state['failures'] = failures
    initial = pd.Timestamp('2000-01-01T00:00Z')
    values, source = refresh._fetch_observed(object(), spec=spec, expected_index=expected, extracted_at_utc=initial)
    attempts = failures + 1
    assert state['calls'] == ['canonical_target', launcher.POST_AUCTION_OBSERVED_SERIES_BY_ZONE['DE']] * attempts
    assert values.reindex(expected).eq(attempts * 100).all()
    assert state['sleeps'] == [2, 4][:failures]
    audit = source['observation_pair_read']
    assert audit['attempts'] == attempts and audit['divergence_failures_count'] == failures
    assert audit['retry_delays_seconds'] == state['sleeps'] and audit['used_for_prediction'] is False
    assert audit['failures'][0]['timestamp_utc'] == str(when)
    assert audit['failures'][0]['extracted_at_utc'] == str(initial)
    assert pd.Timestamp(source['extracted_at_utc']) > initial
    assert source['nocache'] is True and source['live_recomputation'] is True
    assert source['post_auction_fallback']['validation_maximum_absolute_difference_eur_mwh'] == 0
    for original, before in state['inputs']:
        pd.testing.assert_series_equal(original, before)


def test_permanent_divergence_stops_after_three_pairs_with_exact_hour_and_values(changing_sources):
    state, expected, spec, when = changing_sources
    state['failures'] = 10
    with pytest.raises(launcher.PostAuctionObservationDivergenceError) as captured:
        refresh._fetch_observed(object(), spec=spec, expected_index=expected)
    error = captured.value
    assert isinstance(error, ValueError)
    assert len(state['calls']) == 6 and state['sleeps'] == [2, 4]
    assert str(when) in str(error) and 'canonique=300' in str(error) and 'post-enchere=307.89' in str(error)
    assert error.diagnostic['divergent_hours'] == 1
    assert error.diagnostic['maximum_absolute_difference_eur_mwh'] == pytest.approx(7.89)


@pytest.mark.parametrize('error', [ValueError('invalid coverage'), RuntimeError('transport failure'), KeyError('invalid source')])
def test_other_errors_are_not_retried_by_observation_coherence_wrapper(monkeypatch, error):
    calls, sleeps = [], []

    def fail(*args, **kwargs):
        calls.append(1)
        raise error

    monkeypatch.setattr(launcher, '_fetch_latest_observed_snapshot', fail)
    monkeypatch.setattr(refresh.time, 'sleep', sleeps.append)
    with pytest.raises(type(error), match=str(error.args[0])):
        refresh._fetch_observed(object(), spec=object(), expected_index=object())
    assert calls == [1] and sleeps == []


def test_direct_export_reader_keeps_single_attempt_contract(changing_sources):
    state, expected, spec, _ = changing_sources
    with pytest.raises(launcher.PostAuctionObservationDivergenceError):
        launcher._fetch_latest_observed_snapshot(object(), spec=spec, expected_index=expected)
    assert len(state['calls']) == 2 and state['sleeps'] == []


def test_success_without_divergence_reads_once_and_does_not_change_timestamp(changing_sources):
    state, expected, spec, _ = changing_sources
    state['failures'] = 0
    extracted = pd.Timestamp('2026-09-11T13:00Z')
    _, source = refresh._fetch_observed(object(), spec=spec, expected_index=expected, extracted_at_utc=extracted)
    assert len(state['calls']) == 2 and state['sleeps'] == []
    assert source['extracted_at_utc'] == str(extracted)
    assert source['observation_pair_read']['attempts'] == 1
    assert source['observation_pair_read']['failures'] == []


def test_snapshot_and_storm_audit_use_epex_reader_extraction_time(tmp_path, monkeypatch):
    from test_nuclear_reporting_refresh import sources, run_refresh

    sources(monkeypatch, actual_count=24, storm_count=24)
    successful_reader = refresh._fetch_epex_observed
    storm_reader = refresh.fetch_native_dashboard_snapshot
    later = pd.Timestamp('2026-09-11T13:49:09.123456Z')
    storm_times = []

    def recovered_reader(client, **kwargs):
        return successful_reader(client, **{**kwargs, 'extracted_at_utc': later})

    def storm(client, **kwargs):
        storm_times.append(kwargs['extracted_at_utc'])
        return storm_reader(client, **kwargs)

    monkeypatch.setattr(refresh, '_fetch_epex_observed', recovered_reader)
    monkeypatch.setattr(refresh, 'fetch_native_dashboard_snapshot', storm)
    _, directory, audit = run_refresh(tmp_path)
    assert audit['extracted_at_utc'] == str(later)
    assert audit['observed']['source']['extracted_at_utc'] == str(later)
    assert directory.name.startswith(later.strftime('%Y%m%dT%H%M%S%fZ'))
    assert storm_times == [later]
