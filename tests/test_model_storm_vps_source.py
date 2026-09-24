from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.model_storm_vps import refresh_vps_payload


ZONE_TIMEZONES = {
    "BE": "Europe/Brussels",
    "DE": "Europe/Berlin",
    "FR": "Europe/Paris",
    "NL": "Europe/Amsterdam",
}


def _payload(day: str = "2026-09-19", zones=("FR",)) -> dict:
    return {
        "delivery_day": day,
        "zones": [
            {"zone": zone, "timezone": ZONE_TIMEZONES[zone], "rows": []}
            for zone in zones
        ],
    }


def _window(day: str, timezone: str) -> pd.DatetimeIndex:
    parsed = date.fromisoformat(day)
    return pd.date_range(
        pd.Timestamp(parsed - timedelta(days=364), tz=timezone),
        pd.Timestamp(parsed + timedelta(days=1), tz=timezone),
        freq="h", inclusive="left",
    ).tz_convert("UTC")


def _values(day="2026-09-19", zone="FR", *, value=1.0) -> pd.Series:
    index = _window(day, ZONE_TIMEZONES[zone])
    return pd.Series(np.full(len(index), value, dtype=float), index=index)


class FakeClient:
    def __init__(self, values: dict[str, object], *, delay=0.0):
        self.values = values
        self.delay = delay
        self.calls = []
        self._lock = threading.Lock()
        self._active = 0
        self.maximum_active = 0

    def get(self, name, **kwargs):
        with self._lock:
            self.calls.append((name, kwargs))
            self._active += 1
            self.maximum_active = max(self.maximum_active, self._active)
        try:
            if self.delay:
                time.sleep(self.delay)
            value = self.values[name]
            if isinstance(value, Exception):
                raise value
            return value.copy() if hasattr(value, "copy") else value
        finally:
            with self._lock:
                self._active -= 1


def _name(zone: str) -> str:
    return f"power.vps.{zone.lower()}.euromwh.h.da.pnl.storm"


def _source(result: dict, index=0) -> dict:
    return result["zones"][index]["vps_history"]["source"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_complete_fresh_extraction_is_immutable_and_checksummed(tmp_path: Path):
    payload = _payload()
    payload["zones"][0]["vps_history"] = {"stale": "must-not-survive"}
    before = deepcopy(payload)
    values = _values()
    values.iloc[0], values.iloc[1] = -12.5, 0.0
    client = FakeClient({_name("FR"): values})

    result = refresh_vps_payload(payload, tmp_path, client=client)

    assert payload == before
    assert result is not payload and result["zones"][0] is not payload["zones"][0]
    source = _source(result)
    assert source["status"] == "complete"
    assert source["series"] == _name("FR")
    assert source["available_hours"] == source["expected_physical_hours"] == len(values)
    assert source["last_available_at_utc"] == values.index[-1].isoformat().replace("+00:00", "Z")
    assert result["zones"][0]["vps_history"]["rows"][0]["pnl"] == -12.5
    assert result["zones"][0]["vps_history"]["rows"][1]["pnl"] == 0.0
    name, kwargs = client.calls[0]
    assert name == _name("FR")
    assert kwargs == {
        "from_value_date": values.index[0], "to_value_date": values.index[-1],
        "nocache": True, "live": False, "_keep_nans": True,
    }

    snapshot = result["vps_snapshot"]
    root = tmp_path / "runs/reports/model_storm/vps_snapshots"
    directory = tmp_path / snapshot["directory"]
    assert directory.parent == root.resolve()
    assert _sha(tmp_path / snapshot["raw_artifact_path"]) == snapshot["raw_artifact_sha256"]
    assert _sha(tmp_path / snapshot["audit_path"]) == snapshot["audit_sha256"]
    assert _sha(tmp_path / snapshot["checksum_manifest_path"]) == snapshot["checksum_manifest_sha256"]
    manifest = json.loads((tmp_path / snapshot["checksum_manifest_path"]).read_text(encoding="utf-8"))
    assert {item["path"] for item in manifest["artifacts"]} == {
        "vps_hourly_raw.parquet", "vps_audit.json",
    }
    raw = pd.read_parquet(tmp_path / snapshot["raw_artifact_path"])
    assert list(raw.columns) == ["zone", "series", "timestamp_utc", "pnl"]
    assert len(raw) == len(values)


def test_nan_is_preserved_as_none_and_is_not_confused_with_zero(tmp_path: Path):
    values = _values()
    values.iloc[3] = 0.0
    values.iloc[4] = np.nan
    result = refresh_vps_payload(
        _payload(), tmp_path, client=FakeClient({_name("FR"): values})
    )
    source = _source(result)
    rows = result["zones"][0]["vps_history"]["rows"]
    assert source["status"] == "partial"
    assert source["nan_hours"] == 1
    assert source["missing_physical_hours"] == 0
    assert rows[3]["pnl"] == 0.0
    assert rows[4]["pnl"] is None
    raw = pd.read_parquet(tmp_path / result["vps_snapshot"]["raw_artifact_path"])
    assert raw.iloc[3].pnl == 0.0
    assert pd.isna(raw.iloc[4].pnl)


@pytest.mark.parametrize(
    "mutator,code",
    [
        (lambda values: pd.Series(values.to_numpy(), index=values.index.tz_localize(None)),
         "timezone_required"),
        (lambda values: pd.concat([values.iloc[:1], values]),
         "duplicate_physical_hour"),
        (lambda values: values.mask(values.index == values.index[2], np.inf),
         "infinite_pnl"),
    ],
)
def test_invalid_physical_sources_fail_closed(tmp_path: Path, mutator, code):
    result = refresh_vps_payload(
        _payload(), tmp_path,
        client=FakeClient({_name("FR"): mutator(_values())}),
    )
    source = _source(result)
    assert source["status"] == "invalid"
    assert source["error_code"] == code
    assert result["zones"][0]["vps_history"]["rows"] == []


def test_out_of_window_values_are_excluded_before_value_validation(tmp_path: Path):
    values = _values()
    outside = pd.Series(
        [np.inf, 99.0],
        index=[values.index[0] - pd.Timedelta(hours=1),
               values.index[-1] + pd.Timedelta(hours=1)],
    )
    raw = pd.concat([outside.iloc[:1], values, outside.iloc[1:]])
    result = refresh_vps_payload(
        _payload(), tmp_path, client=FakeClient({_name("FR"): raw})
    )
    assert _source(result)["status"] == "complete"
    rows = result["zones"][0]["vps_history"]["rows"]
    assert len(rows) == len(values)
    assert rows[0]["timestamp_utc"] == values.index[0].isoformat().replace("+00:00", "Z")
    assert rows[-1]["timestamp_utc"] == values.index[-1].isoformat().replace("+00:00", "Z")


def test_dst_windows_keep_every_unique_physical_hour(tmp_path: Path):
    for number, (day, expected_hours) in enumerate([
        ("2026-03-29", 8759),
        ("2026-10-25", 8761),
    ]):
        root = tmp_path / str(number)
        root.mkdir()
        values = _values(day=day)
        assert len(values) == expected_hours
        result = refresh_vps_payload(
            _payload(day), root, client=FakeClient({_name("FR"): values})
        )
        source = _source(result)
        assert source["status"] == "complete"
        assert source["expected_physical_hours"] == expected_hours
        stamps = [row["timestamp_utc"] for row in result["zones"][0]["vps_history"]["rows"]]
        assert len(stamps) == len(set(stamps)) == expected_hours


def test_zone_failure_is_nonblocking_and_never_reuses_stale_rows(tmp_path: Path):
    payload = _payload(zones=("FR", "DE"))
    payload["zones"][1]["vps_history"] = {
        "source": {"status": "complete"},
        "rows": [{"timestamp_utc": "2000-01-01T00:00:00Z", "pnl": 999.0}],
    }
    fr = _values(zone="FR").iloc[:-5]
    secret = "token=SHOULD_NOT_APPEAR"
    client = FakeClient({
        _name("FR"): fr,
        _name("DE"): ConnectionError(secret),
    }, delay=0.01)

    result = refresh_vps_payload(payload, tmp_path, client=client)

    assert _source(result, 0)["status"] == "partial"
    failed = _source(result, 1)
    assert failed["status"] == "unavailable"
    assert failed["error_type"] == "network_error"
    assert secret not in json.dumps(result)
    assert result["zones"][1]["vps_history"]["rows"] == []
    assert client.maximum_active <= 2
    assert client.maximum_active == 1  # injected client is deliberately sequential
    audit = (tmp_path / result["vps_snapshot"]["audit_path"]).read_text(encoding="utf-8")
    assert secret not in audit


@pytest.mark.parametrize("empty", [
    None,
    pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC")),
])
def test_empty_source_is_unavailable_not_a_complete_zero_history(tmp_path: Path, empty):
    result = refresh_vps_payload(
        _payload(), tmp_path, client=FakeClient({_name("FR"): empty})
    )
    source = _source(result)
    assert source["status"] == "unavailable"
    assert source["available_hours"] == 0
    assert source["missing_physical_hours"] == source["expected_physical_hours"]
    assert result["zones"][0]["vps_history"]["rows"] == []


def test_owned_clients_are_bounded_thread_local_timed_and_closed(tmp_path, monkeypatch):
    from chronos2_hourly import model_storm_vps as module
    from types import SimpleNamespace

    monkeypatch.setattr(module, "load_yaml", lambda _: {"data": {"saturn_url": "https://example.invalid", "saturn_author": "test"}})
    monkeypatch.setattr(module, "_sha256", lambda _: "fixture_hash")
    clients = []
    def create(*settings):
        client = FakeClient({_name(z): _values(zone=z) for z in ZONE_TIMEZONES}, delay=.03)
        calls = []
        client.session = SimpleNamespace(request=lambda *a, **k: calls.append(k), close=lambda: calls.append("closed"))
        client.session_calls = calls
        clients.append(client)
        return client
    monkeypatch.setattr(module, "create_saturn_client", create)
    result = module.refresh_vps_payload(_payload(zones=tuple(ZONE_TIMEZONES)), tmp_path)
    assert all(z["vps_history"]["source"]["status"] == "complete" for z in result["zones"])
    assert 1 <= len(clients) <= 2
    assert sum(len(c.calls) for c in clients) == 4
    for client in clients:
        assert client.maximum_active == 1
        client.session.request("GET", "ignored")
        assert client.session_calls[-1]["timeout"] == (8, 60)
        assert "closed" in client.session_calls


def test_missing_owned_configuration_is_explicitly_unavailable(tmp_path):
    result = refresh_vps_payload(_payload(), tmp_path)
    assert _source(result)["status"] == "unavailable"
    assert _source(result)["error_type"] == "client_unavailable"


def test_invalid_zone_timezone_rejected_before_collection(tmp_path):
    payload = _payload()
    payload["zones"][0]["timezone"] = "UTC"
    client = FakeClient({})
    with pytest.raises(ValueError, match="canonical timezone"):
        refresh_vps_payload(payload, tmp_path, client=client)
    assert client.calls == []
    assert not (tmp_path / "runs").exists()


def test_checksum_verification_detects_corruption(tmp_path):
    from chronos2_hourly.model_storm_vps import _verify_snapshot
    result = refresh_vps_payload(_payload(), tmp_path, client=FakeClient({_name("FR"): _values()}))
    snapshot = result["vps_snapshot"]
    with pytest.raises(ValueError, match="checksum"):
        _verify_snapshot(tmp_path / snapshot["directory"], raw_sha="wrong", audit_sha=snapshot["audit_sha256"])
