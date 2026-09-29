from datetime import date, timedelta
import json
from pathlib import Path

import pandas as pd
import pytest

from chronos2_hourly import jao_flowbased as jao
from chronos2_hourly import nyx_annual_jao_source as live
from chronos2_hourly import nyx_annual_jao_history as history
from test_nyx_annual_jao_source import _fetch, FakeClient


DAY = date(2026, 9, 25)


def write_partition(root, origin, retrieved):
    fetched = _fetch(origin, retrieved=retrieved)
    normalised, audit = jao.normalise_initial_computation(fetched, delivery_day=origin)
    jao.write_daily_flowbased_bundle(root, delivery_day=origin, fetch=fetched,
        normalised=normalised,
        features=jao.build_hourly_flowbased_features(normalised, daily_audit=audit),
        audit=audit, tls_verification=True)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cutoff = jao.expected_cutoff_utc(DAY)
    start = jao.local_day_utc_bounds(DAY - timedelta(days=2))[0]
    current_start, end = jao.local_day_utc_bounds(DAY)
    full = pd.date_range(start, end, freq="h", inclusive="left")
    current = pd.date_range(current_start, end, freq="h", inclusive="left")
    monkeypatch.setattr(live, "delivery_grid", lambda day: (full, current, cutoff))
    cache = tmp_path / "live"
    live.capture_day(day=DAY, cache_root=cache, client=FakeClient(_fetch(DAY)),
                     now_utc=cutoff - pd.Timedelta(minutes=9))
    return dict(live_cache_root=cache, history_cache_root=tmp_path / "history",
                legacy_cache_root=tmp_path / "legacy"), tmp_path / "bundle", cutoff


class DynamicClient:
    def __init__(self, retrieved, fail_on=None):
        self.retrieved, self.fail_on, self.calls = retrieved, fail_on, []

    def fetch_initial_day(self, day):
        self.calls.append(day)
        if day == self.fail_on:
            raise RuntimeError("simulated transient network interruption")
        return _fetch(day, retrieved=self.retrieved)


def load_receipt(bundle):
    return json.loads((bundle / "source_receipts/jao_initial.json").read_text(encoding="utf-8"))


def test_real_historical_retrieval_is_preserved_and_reconstructed(setup):
    kwargs, bundle, cutoff = setup
    retrieved = cutoff - pd.Timedelta(hours=1)
    for origin in (DAY - timedelta(days=2), DAY - timedelta(days=1)):
        write_partition(kwargs["legacy_cache_root"], origin, retrieved)
    client = DynamicClient(cutoff + pd.Timedelta(hours=1))
    result = history.publish_history(str(DAY), bundle, client=client, **kwargs)
    assert result["state"] == "COMPLETE" and result["asof_cutoff_verified"] is True
    assert client.calls == []
    receipt = load_receipt(bundle)
    assert receipt["training_snapshot_max_retrieved_at_utc"] == retrieved.isoformat()
    assert receipt["actual_pre_cutoff_capture_verified"] is False
    verified = history.verify_bundle_history(bundle, str(DAY), receipt)
    assert verified["daily_raw_captures_recomputed"] == 3
    assert verified["asof_cutoff_verified"] is True
    # Reuse is portable: it does not depend on still-existing input cache paths.
    assert history.publish_history(str(DAY), bundle, client=client, **{
        **kwargs, "live_cache_root": bundle / "missing_live"})["reused"] is True


def test_after_current_cutoff_bootstrap_is_preparation_only_and_incomplete_replaceable(setup):
    kwargs, bundle, cutoff = setup
    receipt_path = bundle / "source_receipts/jao_initial.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps({"state": "INCOMPLETE"}), encoding="utf-8")
    retrieved = cutoff + pd.Timedelta(hours=1)
    client = DynamicClient(retrieved)
    history.publish_history(str(DAY), bundle, client=client, **kwargs)
    receipt = load_receipt(bundle)
    assert receipt["asof_cutoff_verified"] is False
    assert receipt["training_snapshot_max_retrieved_at_utc"] == retrieved.isoformat()
    assert receipt["delivery_snapshot_pre_cutoff_verified"] is True
    assert history.verify_bundle_history(bundle, str(DAY), receipt)["asof_cutoff_verified"] is False
    receipt["asof_cutoff_verified"] = True
    with pytest.raises(ValueError, match="causality differs"):
        history.verify_bundle_history(bundle, str(DAY), receipt)


def test_late_delivery_capture_is_rejected_before_history_network(setup):
    kwargs, bundle, cutoff = setup
    kwargs["live_cache_root"] = bundle / "late_live"
    write_partition(kwargs["live_cache_root"], DAY, cutoff + pd.Timedelta(minutes=1))
    client = DynamicClient(cutoff)
    with pytest.raises(ValueError, match="not a direct verified initial capture"):
        history.publish_history(str(DAY), bundle, client=client, **kwargs)
    assert client.calls == []


def test_partial_bootstrap_resumes_only_missing_days(setup):
    kwargs, bundle, cutoff = setup
    first, second = DAY - timedelta(days=2), DAY - timedelta(days=1)
    client = DynamicClient(cutoff, fail_on=second)
    with pytest.raises(RuntimeError, match="network interruption"):
        history.publish_history(str(DAY), bundle, client=client, **kwargs)
    assert client.calls == [first, second]
    assert (kwargs["history_cache_root"] / "partitions" / str(first)).is_dir()
    assert not (bundle / "source_receipts/jao_initial.json").exists()
    resumed = DynamicClient(cutoff)
    history.publish_history(str(DAY), bundle, client=resumed, **kwargs)
    assert resumed.calls == [second]
    assert history.verify_bundle_history(bundle, str(DAY), load_receipt(bundle))["asof_cutoff_verified"]


def test_resealed_corrupt_normalisation_fails_independent_raw_reconstruction(setup):
    kwargs, bundle, cutoff = setup
    history.publish_history(str(DAY), bundle, client=DynamicClient(cutoff), **kwargs)
    root = bundle / live.SOURCE_SUBDIR / "captures"
    origin = DAY - timedelta(days=2)
    _, audit_path, normalised_path, _ = live._paths(root, origin)
    normalised = pd.read_parquet(normalised_path)
    normalised.loc[0, "ram"] += 100
    normalised.to_parquet(normalised_path, index=False)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["normalised_sha256"] = live.sha256(normalised_path)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    receipt = load_receipt(bundle)
    for path in (normalised_path, audit_path):
        receipt["artifact_sha256"][path.relative_to(bundle).as_posix()] = live.sha256(path)
    with pytest.raises(AssertionError, match="ram"):
        history.verify_bundle_history(bundle, str(DAY), receipt)


def test_corrupt_cached_partition_is_not_replaced_with_new_download(setup):
    kwargs, bundle, cutoff = setup
    first, second = DAY - timedelta(days=2), DAY - timedelta(days=1)
    with pytest.raises(RuntimeError):
        history.publish_history(str(DAY), bundle,
            client=DynamicClient(cutoff, fail_on=second), **kwargs)
    raw = live._paths(kwargs["history_cache_root"] / "partitions" / str(first), first)[0]
    raw.write_bytes(b"corrupt")
    client = DynamicClient(cutoff)
    with pytest.raises(ValueError, match="checksum differs"):
        history.publish_history(str(DAY), bundle, client=client, **kwargs)
    assert client.calls == []


def test_empty_historical_api_partition_stays_unavailable_without_fallback(setup):
    kwargs, bundle, cutoff = setup
    first = DAY - timedelta(days=2)

    class EmptyDayClient(DynamicClient):
        def fetch_initial_day(self, day):
            result = super().fetch_initial_day(day)
            if day == first:
                return jao.JaoFetchResult(**{**result.__dict__, "rows": (), "total_rows": 0,
                    "last_modified_utc": None, "page_last_modified_utc": ()})
            return result

    history.publish_history(str(DAY), bundle, client=EmptyDayClient(cutoff), **kwargs)
    frame = pd.read_parquet(bundle / live.SOURCE_SUBDIR / live.FEATURE_NAME)
    missing = frame.index.tz_convert("Europe/Paris").date == first
    assert frame.loc[missing, live.AVAILABLE].eq(0).all()
    assert frame.loc[missing, list(live.VALUE_COLUMNS)].isna().all().all()
    assert frame.loc[~missing, live.AVAILABLE].eq(1).all()
    assert history.verify_bundle_history(bundle, str(DAY), load_receipt(bundle))["passed"]


def test_historical_watermark_after_own_cutoff_is_allowed_only_for_current_fit(setup):
    kwargs, bundle, cutoff = setup

    class RevisedHistoryClient(DynamicClient):
        def fetch_initial_day(self, day):
            result = super().fetch_initial_day(day)
            modified = cutoff - pd.Timedelta(minutes=30)
            return jao.JaoFetchResult(**{**result.__dict__, "last_modified_utc": modified,
                                        "page_last_modified_utc": (modified,)})

    history.publish_history(str(DAY), bundle, client=RevisedHistoryClient(cutoff), **kwargs)
    receipt = load_receipt(bundle)
    assert receipt["asof_cutoff_verified"] is True
    assert receipt["origin_snapshot_capture_verified"] is False
    assert history.verify_bundle_history(bundle, str(DAY), receipt)["passed"]
