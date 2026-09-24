"""One explicit EPEX reference for all report labels, never model inputs."""
from copy import deepcopy
import hashlib

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nuclear_reporting_refresh as refresh
from chronos2_hourly import reporting_observations as observations
from chronos2_hourly.storm_dashboard import storm_dashboard_series


ZONES = [("BE", "Europe/Brussels"), ("DE", "Europe/Berlin"),
         ("FR", "Europe/Paris"), ("NL", "Europe/Amsterdam")]
DAY = "2026-09-19"


def install(monkeypatch, *, zone="FR", timezone="Europe/Paris", day=DAY,
            available=None, fault=None):
    expected, current = refresh._support(day, timezone)
    # Deliberately varying prices (including negatives) detect source replacement
    # and accidental smoothing. They need not match the model training target.
    values = pd.Series(np.arange(len(expected), dtype=float) / 4 - 150, index=expected)
    if available is not None:
        values.loc[current[available:]] = np.nan
    if fault == "history_gap":
        values.iloc[10] = np.nan
    elif fault == "infinite":
        values.iloc[10] = np.inf
    elif fault == "not_numeric":
        values = values.astype(object)
        values.iloc[10] = "unavailable"
    elif fault == "duplicate":
        values = pd.concat([values.iloc[[0]], values])
    elif fault == "naive":
        values.index = values.index.tz_localize(None)
    elif fault == "unordered":
        values = values.iloc[::-1]
    elif fault == "quarter_hour":
        values.index = values.index + pd.Timedelta(minutes=15)
    elif fault == "empty":
        values = values.iloc[:0]
    before = values.copy(deep=True)
    calls = []

    def actual(client, series, start, end, tz, **kwargs):
        calls.append((series, start, end, tz, kwargs))
        assert series == observations.EPEX_REPORTING_SERIES_BY_ZONE[zone]
        assert kwargs == {"naive_timezone": "UTC", "nocache": True, "live": True}
        if fault == "transport":
            raise RuntimeError("source unavailable")
        return values

    def storm(client, *, zone, expected_index, extracted_at_utc):
        series = storm_dashboard_series(zone)
        return pd.Series(100., index=expected_index), {
            "kind": "saturn_storm_day_ahead_cache_with_native_gap_fallback",
            "series": series, "primary_series": series, "zone": zone,
            "extracted_at_utc": str(extracted_at_utc), "used_for_prediction": False,
        }

    def forbidden(*args, **kwargs):
        pytest.fail("EPEX reporting must not fetch or compare canonical observations")

    monkeypatch.setattr(observations, "fetch_saturn_series_from_client", actual)
    monkeypatch.setattr(refresh, "fetch_native_dashboard_snapshot", storm)
    monkeypatch.setattr(refresh, "_fetch_observed", forbidden)
    config = {"zones": {zone: {"target": {"series": f"canonical_training_{zone}"}}}}
    return config, expected, current, values, before, calls


def run(tmp_path, config, *, zone="FR", timezone="Europe/Paris", day=DAY):
    return refresh.refresh_nuclear_reporting_sources(
        config, zone, timezone, day, tmp_path / "report_only", client=object())


@pytest.mark.parametrize("zone,timezone", ZONES)
def test_all_reporting_prices_come_from_one_explicit_epex_series(tmp_path, monkeypatch, zone, timezone):
    config, expected, current, raw, before, calls = install(monkeypatch, zone=zone, timezone=timezone)
    original_config = deepcopy(config)
    frozen = tmp_path / "frozen_model.bin"
    frozen.write_bytes(b"existing model and predictions")
    values, directory, audit = run(tmp_path, config, zone=zone, timezone=timezone)
    pd.testing.assert_series_equal(values, before.rename("actual"))
    pd.testing.assert_series_equal(raw, before)
    assert config == original_config and frozen.read_bytes() == b"existing model and predictions"
    assert len(calls) == 1
    assert calls[0][1:4] == (expected[0] - pd.Timedelta(hours=2), expected[-1] + pd.Timedelta(hours=2), timezone)
    assert values.loc[current].notna().sum() == 24
    assert audit["schema_version"] == 2
    assert audit["observation_policy"] == observations.EPEX_REPORTING_POLICY
    assert audit["training_target_series"] == f"canonical_training_{zone}"
    source = audit["observed"]["source"]
    for key, value in observations.epex_reporting_identity(zone, timezone).items():
        assert source[key] == value
    assert "post_auction_fallback" not in source and "observation_pair_read" not in source
    assert audit["observed"]["artifact_sha256"] == hashlib.sha256(
        (directory / "inputs/observed_latest.parquet").read_bytes()).hexdigest()
    verified = refresh.verify_refreshed_observations(values, audit, zone=zone, timezone=timezone, delivery_day=DAY)
    assert verified["actual_reference"] == verified["actual_reference_label"] == "EPEX"
    assert verified["policy"] == observations.EPEX_REPORTING_POLICY
    assert verified["series"] == observations.EPEX_REPORTING_SERIES_BY_ZONE[zone]
    assert verified["used_for_prediction"] is False


@pytest.mark.parametrize("day,hours", [(DAY, 24), ("2026-03-29", 23), ("2026-10-25", 25)])
@pytest.mark.parametrize("complete", [False, True])
def test_dst_days_keep_each_physical_hour_and_partial_days_are_fully_masked(
    tmp_path, monkeypatch, day, hours, complete,
):
    count = hours if complete else hours - 1
    config, expected, current, raw, before, calls = install(monkeypatch, day=day, available=count)
    values, _, audit = run(tmp_path, config, day=day)
    assert len(current) == hours and values.index.equals(expected)
    assert values.index.is_unique
    pd.testing.assert_series_equal(values.loc[expected.difference(current)], before.rename("actual").loc[expected.difference(current)])
    assert values.loc[current].notna().sum() == (hours if complete else 0)
    assert audit["observed"]["source"]["current_delivery_raw_available_hours"] == count
    assert audit["observed"]["source"]["current_delivery_expected_hours"] == hours
    assert refresh.verify_refreshed_observations(values, audit, zone="FR", timezone="Europe/Paris", delivery_day=day)["status"] == "verified"
    pd.testing.assert_series_equal(raw, before)


@pytest.mark.parametrize("fault", ["history_gap", "infinite", "not_numeric", "duplicate", "naive",
                                   "unordered", "quarter_hour", "empty", "transport"])
def test_invalid_or_missing_epex_data_never_substitutes_another_source(tmp_path, monkeypatch, fault):
    config, _, _, raw, before, calls = install(monkeypatch, fault=fault)
    with pytest.raises((refresh.NuclearReportingRefreshError, RuntimeError)):
        run(tmp_path, config)
    assert len(calls) == 1 and not (tmp_path / "report_only").exists()
    pd.testing.assert_series_equal(raw, before)


@pytest.mark.parametrize("field,value", [
    ("series", "power.price.de.euromwh.h.obs.epex"), ("kind", "saturn_target_latest_extraction"),
    ("policy", "legacy"), ("provider", "ENTSO-E"), ("origin", "ENTSO-E"),
    ("actual_reference", "ENTSO-E"), ("source_mixing", True), ("fallback_policy", "canonical"),
    ("zone", "DE"), ("timezone", "UTC"), ("frequency", "15min"),
    ("price_unit", "EUR/kWh"), ("used_for_prediction", True),
])
def test_epex_identity_cannot_be_relabelled_or_redirected(tmp_path, monkeypatch, field, value):
    config, *_ = install(monkeypatch)
    values, _, audit = run(tmp_path, config)
    audit["observed"]["source"][field] = value
    with pytest.raises(refresh.NuclearReportingRefreshError):
        refresh.verify_refreshed_observations(values, audit, zone="FR", timezone="Europe/Paris", delivery_day=DAY)


@pytest.mark.parametrize("mutation", ["series", "schema", "policy", "fallback", "sha", "values"])
def test_epex_snapshot_verifier_rejects_invalid_audit_or_changed_prices(tmp_path, monkeypatch, mutation):
    config, *_ = install(monkeypatch)
    values, _, audit = run(tmp_path, config)
    if mutation == "series":
        audit["observed"]["series"] = "canonical_training_FR"
    elif mutation == "schema":
        audit["schema_version"] = 1
    elif mutation == "policy":
        del audit["observation_policy"]
    elif mutation == "fallback":
        audit["observed"]["source"]["post_auction_fallback"] = {"applied_hours": 1}
    elif mutation == "sha":
        audit["observed"]["artifact_sha256"] = "0" * 64
    else:
        values.iloc[10] += .001
    with pytest.raises(refresh.NuclearReportingRefreshError):
        refresh.verify_refreshed_observations(values, audit, zone="FR", timezone="Europe/Paris", delivery_day=DAY)


@pytest.mark.parametrize("mixed", [False, True])
def test_old_canonical_snapshots_stay_verifiable_with_their_original_source_identity(tmp_path, monkeypatch, mixed):
    config, *_ = install(monkeypatch)
    values, _, audit = run(tmp_path, config)
    audit["schema_version"] = 1
    del audit["observation_policy"]
    del audit["training_target_series"]
    old = audit["observed"]["source"]
    legacy = {key: old[key] for key in (
        "zone", "nocache", "live_recomputation", "used_for_prediction", "extracted_at_utc",
        "current_delivery_day_local", "current_delivery_expected_hours", "current_delivery_actual_status",
        "current_delivery_raw_available_hours", "partial_daily_average_forbidden",
    )}
    legacy.update(kind="saturn_target_latest_extraction", series="canonical_training_FR")
    if mixed:
        legacy["post_auction_fallback"] = {"applied_hours": 24}
    audit["observed"].update(series="canonical_training_FR", source=legacy)
    audit["canonical_actuals"]["source"] = legacy
    verified = refresh.verify_refreshed_observations(values, audit, zone="FR", timezone="Europe/Paris", delivery_day=DAY)
    assert verified["actual_reference"] == "ENTSO-E"
    assert verified["actual_reference_label"] == ("ENTSO-E + EPEX" if mixed else "ENTSO-E")
    assert verified["policy"] == "legacy_canonical_with_validated_fallback"
