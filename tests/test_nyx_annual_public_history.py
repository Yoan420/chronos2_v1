"""Current-fit bootstrap never fabricates old vintages or changes delivery inputs."""
import json

import pandas as pd
import pytest

from chronos2_hourly import nyx_annual_public_history as m
import run_nyx_annual_hydro_source as hydro_capture
import run_nyx_annual_exchange_source as exchange_capture


class Session:
    def __init__(self, now, *, fail_at=None, null_first=False):
        self.now = now
        self.calls = []
        self.fail_at = fail_at
        self.null_first = null_first

    def get(self, url, *, params, timeout, allow_redirects):
        self.calls.append((url, params))
        assert timeout == (10, 45) and allow_redirects is False
        if self.fail_at == len(self.calls):
            return type("Response", (), {"status_code": 403, "content": b""})()
        zone = params["country"]
        first, stop = pd.Timestamp(params["start"]), pd.Timestamp(params["end"]) + pd.Timedelta(minutes=1)
        fields = (("france", "netherlands", "belgium", "sum") if zone == "de"
                  else (*m.hydro.SERIES, "cross_border_electricity_trading"))
        value = -1.0 if zone == "de" else 2000.0
        payload = {"schema_version": "2.0", "country": zone,
            "endpoint": "cbpf" if zone == "de" else "public_power",
            "unit": "GW" if zone == "de" else "MW",
            "timezone": "Europe/Berlin" if zone == "de" else "Europe/Paris",
            "interval_minutes": 15, "resolution": "PT15M", "generated_at": self.now.isoformat(),
            "available_from": first.isoformat(), "available_until": (stop - pd.Timedelta(minutes=15)).isoformat(),
            "series": [{"id": name} for name in fields],
            "attributes": {"sign_convention": "positive = import, negative = export"} if zone == "de" else {},
            "data": [{"timestamp": t.isoformat(), "values": {name: value for name in fields}}
                for t in pd.date_range(first, stop, freq="15min", inclusive="left")]}
        if self.null_first:
            payload["data"][0]["values"][fields[0]] = None
        return type("Response", (), {"status_code": 200, "content": json.dumps(payload).encode()})()


def setup_case(tmp_path, monkeypatch, group, day="2026-10-25"):
    monkeypatch.setattr(m, "MIN_REQUEST_INTERVAL_SECONDS", 0)
    actual_grid = m.delivery_grid
    _, current, cutoff = actual_grid(day)
    # Enough history to cross a UTC month and exercise resumable partitioning.
    training = pd.date_range(current[0] - pd.Timedelta(days=30), current[0], freq="h", inclusive="left")
    monkeypatch.setattr(m, "delivery_grid", lambda _: (training.append(current), current, cutoff))
    archive = tmp_path / "captures"
    capture_clock = cutoff - pd.Timedelta(minutes=5)
    capture = hydro_capture.capture if group == "public_hydro" else exchange_capture.capture
    capture(day, archive, session=Session(capture_clock - pd.Timedelta(minutes=1)), now_utc=capture_clock)
    return day, cutoff, archive, current


@pytest.mark.parametrize("group", m.GROUPS)
def test_bootstrap_reconstructs_history_preserves_strict_delivery_and_dst(tmp_path, monkeypatch, group):
    day, cutoff, archive, current = setup_case(tmp_path, monkeypatch, group)
    later = cutoff + pd.Timedelta(hours=1)
    session = Session(later - pd.Timedelta(seconds=1), null_first=True)
    bundle, cache = tmp_path / "bundle", tmp_path / "cache"
    destination = m.publish_history(group, day, bundle, archive=archive,
        history_archive=cache, session=session, now_utc=later)
    receipt = json.loads(destination.read_text())
    assert receipt["training_snapshot_max_retrieved_at_utc"] == later.isoformat()
    assert receipt["asof_cutoff_verified"] is False
    assert receipt["origin_snapshot_capture_verified"] is False
    assert receipt["delivery_snapshot_pre_cutoff_verified"] is True
    assert receipt["target_hours"] == 25
    assert pd.Timestamp(receipt["asof_state_utc"]) < cutoff
    facts = m.verify_bundle_history(bundle, day, receipt)
    assert facts["asof_cutoff_verified"] is False
    _, _, verify = m._capture_module(group)
    captured, _ = verify(archive / day, day)
    published = pd.read_parquet(bundle / f"source_artifacts/{group}/features.parquet")
    pd.testing.assert_frame_equal(published.loc[current.rename("timestamp_utc")], captured, check_freq=False, check_exact=True)
    again = Session(later)
    m.publish_history(group, day, bundle, archive=archive, history_archive=cache, session=again)
    assert again.calls == []
    receipt["asof_cutoff_verified"] = True
    with pytest.raises(ValueError, match="asof_cutoff_verified"):
        m.verify_bundle_history(bundle, day, receipt)


def test_pre_cutoff_training_snapshot_is_valid_for_current_fit_only(tmp_path, monkeypatch):
    group = "public_hydro"
    day, cutoff, archive, _ = setup_case(tmp_path, monkeypatch, group, "2026-03-29")
    now = cutoff - pd.Timedelta(minutes=3)
    path = m.publish_history(group, day, tmp_path / "bundle", archive=archive,
        # A provider clock rounded one second ahead must not move our actual
        # retrieval time. Strict delivery capture verification remains intact.
        history_archive=tmp_path / "cache", session=Session(now + pd.Timedelta(seconds=1)), now_utc=now)
    receipt = json.loads(path.read_text())
    assert receipt["asof_cutoff_verified"] is True
    assert receipt["training_snapshot_max_retrieved_at_utc"] == now.isoformat()
    assert receipt["origin_snapshot_capture_verified"] is False
    assert receipt["provider_revision_vintage_verified"] is False
    assert receipt["target_hours"] == 23


def test_interrupted_download_resumes_immutable_partitions(tmp_path, monkeypatch):
    day, cutoff, archive, _ = setup_case(tmp_path, monkeypatch, "public_hydro")
    now = cutoff + pd.Timedelta(hours=1)
    cache, bundle = tmp_path / "cache", tmp_path / "bundle"
    broken = Session(now, fail_at=2)
    with pytest.raises(ValueError, match="HTTP 403"):
        m.publish_history("public_hydro", day, bundle, archive=archive,
                          history_archive=cache, session=broken, now_utc=now)
    files = list(cache.rglob("*.zip"))
    assert len(files) == 1
    before = files[0].read_bytes()
    good = Session(now + pd.Timedelta(minutes=1))
    path = m.publish_history("public_hydro", day, bundle, archive=archive,
        history_archive=cache, session=good, now_utc=now + pd.Timedelta(minutes=1))
    assert len(good.calls) == 1
    assert files[0].read_bytes() == before
    assert json.loads(path.read_text())["training_snapshot_max_retrieved_at_utc"] == (now + pd.Timedelta(minutes=1)).isoformat()


def test_missing_delivery_capture_never_downloads_a_replacement(tmp_path):
    session = Session(pd.Timestamp("2026-09-29T12:00:00Z"))
    with pytest.raises((ValueError, FileNotFoundError)):
        m.publish_history("public_hydro", "2026-09-30", tmp_path / "bundle",
            archive=tmp_path / "absent", history_archive=tmp_path / "cache", session=session)
    assert session.calls == []
    assert not (tmp_path / "bundle/source_receipts/public_hydro.json").exists()


def test_rehashed_feature_tampering_fails_raw_reconstruction(tmp_path, monkeypatch):
    day, cutoff, archive, _ = setup_case(tmp_path, monkeypatch, "lagged_exchange")
    now = cutoff + pd.Timedelta(hours=1)
    bundle = tmp_path / "bundle"
    destination = m.publish_history("lagged_exchange", day, bundle, archive=archive,
        history_archive=tmp_path / "cache", session=Session(now), now_utc=now)
    receipt = json.loads(destination.read_text())
    relative = "source_artifacts/lagged_exchange/features.parquet"
    frame = pd.read_parquet(bundle / relative)
    frame.iloc[0, 0] += 10
    frame.to_parquet(bundle / relative)
    receipt["artifact_sha256"][relative] = m.sha256(bundle / relative)
    with pytest.raises(AssertionError):
        m.verify_bundle_history(bundle, day, receipt)
