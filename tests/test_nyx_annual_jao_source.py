from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import jao_flowbased as jao
from chronos2_hourly import nyx_annual_jao_source as source
from chronos2_hourly.nyx_annual_live_preflight import (
    delivery_grid,
    load_schema,
    validate_source_receipt,
)


def _raw_row(timestamp: pd.Timestamp, identifier: int) -> dict:
    return {
        "id": identifier,
        "dateTimeUtc": timestamp.isoformat().replace("+00:00", "Z"),
        "tso": "RTE", "cneName": f"CNE {identifier}",
        "cneEic": f"EIC-{identifier}", "cneStatus": "no CRA",
        "direction": "DIRECT", "hubFrom": "FR", "hubTo": "DE",
        "contName": "BASE", "contingencies": [],
        "presolved": True, "cnec": True,
        "ram": 400.0 + identifier, "fmax": 1000.0,
        "frm": 100.0, "frefInit": 0.0, "fcore": 0.0,
        "fall": 0.0, "fuaf": 0.0,
        "ptdf_ALBE": 0.16, "ptdf_ALDE": -0.14,
        "ptdf_AT": 0.2, "ptdf_BE": 0.1,
        "ptdf_CZ": -0.2, "ptdf_DE": -0.3,
        "ptdf_FR": 0.4, "ptdf_HR": 0.05,
        "ptdf_HU": -0.05, "ptdf_NL": -0.1,
        "ptdf_PL": -0.15, "ptdf_RO": 0.03,
        "ptdf_SI": 0.12, "ptdf_SK": -0.08,
    }


def _fetch(day: date, *, retrieved: pd.Timestamp | None = None,
           last_modified: pd.Timestamp | None = None,
           empty: bool = False) -> jao.JaoFetchResult:
    start, end = jao.local_day_utc_bounds(day)
    cutoff = jao.expected_cutoff_utc(day)
    rows = () if empty else tuple(
        _raw_row(hour, position + 1)
        for position, hour in enumerate(pd.date_range(start, end, freq="h", inclusive="left")))
    modified = last_modified or cutoff - pd.Timedelta(hours=1)
    return jao.JaoFetchResult(
        endpoint="initialComputation", start_utc=start, end_utc=end,
        rows=rows, total_rows=len(rows), last_modified_utc=modified,
        retrieved_at_utc=retrieved or cutoff - pd.Timedelta(minutes=10),
        filters={"Presolved": True}, requests=1,
        page_last_modified_utc=(modified,) if rows else (),
    )


class FakeClient:
    def __init__(self, result: jao.JaoFetchResult):
        self.result = result
        self.calls = []

    def fetch_initial_day(self, day: date) -> jao.JaoFetchResult:
        self.calls.append(day)
        return self.result


def test_descriptor_schema_matches_all_annual_country_projections() -> None:
    schema = load_schema()
    for zone in ("FR", "DE", "BE", "NL"):
        full = {name for name in schema["families"]["cwe_absolute_2000"]["columns"][zone]
                if name.startswith("extra_jao_")}
        compact = {name for name in schema["families"]["cwe_residual_2000"]["columns"][zone]
                   if name.startswith("extra_jao_")}
        assert full == set(source.FEATURE_COLUMNS)
        assert len(compact) == 5 and compact <= full


def test_rejects_late_capture_despite_old_last_modified() -> None:
    day = date(2026, 9, 30)
    cutoff = jao.expected_cutoff_utc(day)
    with pytest.raises(ValueError, match="after D-1 08:00"):
        source._check_fetch(_fetch(day, retrieved=cutoff + pd.Timedelta(minutes=1)), day)


def test_rejects_empty_initial_and_wrong_endpoint() -> None:
    day = date(2026, 9, 30)
    with pytest.raises(ValueError, match="absent or empty"):
        source._check_fetch(_fetch(day, empty=True), day)
    result = _fetch(day)
    result = jao.JaoFetchResult(**{**result.__dict__, "endpoint": "finalComputation"})
    with pytest.raises(ValueError, match="wrong endpoint"):
        source._check_fetch(result, day)


def test_no_retroactive_fetch_after_cutoff(tmp_path: Path) -> None:
    day = date(2026, 9, 30)
    cutoff = jao.expected_cutoff_utc(day)
    client = FakeClient(_fetch(day))
    with pytest.raises(ValueError, match="D-1 01:15–08:00"):
        source.capture_day(day=day, cache_root=tmp_path, client=client,
                           now_utc=cutoff + pd.Timedelta(seconds=1))
    assert client.calls == []
    assert not (tmp_path / "raw/initialComputation").exists()


def test_previously_captured_day_can_be_verified_after_cutoff(tmp_path: Path) -> None:
    day = date(2026, 9, 30)
    cutoff = jao.expected_cutoff_utc(day)
    first = FakeClient(_fetch(day))
    original = source.capture_day(day=day, cache_root=tmp_path, client=first,
                                  now_utc=cutoff - pd.Timedelta(minutes=9))
    second = FakeClient(_fetch(day, retrieved=cutoff + pd.Timedelta(hours=2)))
    repeated = source.capture_day(day=day, cache_root=tmp_path, client=second,
                                  now_utc=cutoff + pd.Timedelta(hours=2))
    assert repeated == original
    assert second.calls == []


def test_actual_precutoff_capture_is_archived_but_365_day_gap_blocks_receipt(
    tmp_path: Path,
) -> None:
    day = date(2026, 9, 30)
    cutoff = jao.expected_cutoff_utc(day)
    cache = tmp_path / "cache"
    bundle = tmp_path / "bundle"
    client = FakeClient(_fetch(day))
    captured = source.capture_day(
        day=day, cache_root=cache, client=client,
        now_utc=cutoff - pd.Timedelta(minutes=9))
    assert client.calls == [day]
    assert captured["retrieved_at_utc"] == (cutoff - pd.Timedelta(minutes=10)).isoformat()
    assert captured["rows"] == 24
    assert source.verify_daily_capture(cache, day) == captured
    result = source.publish_jao_receipt(day=day, cache_root=cache, bundle=bundle)
    assert result["state"] == "INCOMPLETE"
    assert result["captured_days"] == 1
    assert result["missing_days"] == 365
    receipt = json.loads((bundle / "source_receipts/jao_initial.json").read_text(encoding="utf-8"))
    assert receipt["training_window_complete"] is False
    assert receipt["actual_pre_cutoff_capture_verified"] is True
    assert receipt["provider_publication_timestamp_verified"] is False
    _, _, current_cutoff = delivery_grid(day.isoformat())
    with pytest.raises(ValueError, match="prospective source receipt incomplete"):
        validate_source_receipt(receipt, group="jao_initial", day=day.isoformat(),
                                bundle=bundle, cutoff=current_cutoff)
    assert not (bundle / source.SOURCE_SUBDIR / source.FEATURE_NAME).exists()


def test_raw_builder_uses_complete_hours_without_legacy_imputation(tmp_path: Path) -> None:
    day = date(2026, 9, 30)
    cache = tmp_path / "cache"
    cutoff = jao.expected_cutoff_utc(day)
    fetched = _fetch(day)
    # The legacy materializer's sidecar fills this gap with a daily median.
    # The annual source must keep the gap unavailable instead.
    fetched = jao.JaoFetchResult(**{**fetched.__dict__,
                                    "rows": fetched.rows[:-1],
                                    "total_rows": len(fetched.rows) - 1})
    source.capture_day(day=day, cache_root=cache, client=FakeClient(fetched),
                       now_utc=cutoff - pd.Timedelta(minutes=9))
    start, end = jao.local_day_utc_bounds(day)
    index = pd.date_range(start, end, freq="h", inclusive="left")
    frame, audit = source.build_strict_history_features(cache, index)
    assert list(frame.columns) == list(source.FEATURE_COLUMNS)
    assert audit["imputation"] is False
    assert audit["available_hours"] == 23
    assert frame[source.AVAILABLE].tolist() == [1] * 23 + [0]
    assert frame.iloc[-1][list(source.VALUE_COLUMNS)].isna().all()
    assert frame.iloc[0]["extra_jao_ram_min_mw"] == 401.0
    assert frame.iloc[0]["extra_jao_stress_hhi"] == 1.0


def test_live_capture_accepts_dst_day_25_physical_hours(tmp_path: Path) -> None:
    day = date(2026, 10, 25)
    cutoff = jao.expected_cutoff_utc(day)
    source.capture_day(day=day, cache_root=tmp_path, client=FakeClient(_fetch(day)),
                       now_utc=cutoff - pd.Timedelta(minutes=9))
    start, end = jao.local_day_utc_bounds(day)
    index = pd.date_range(start, end, freq="h", inclusive="left")
    frame, audit = source.build_strict_history_features(tmp_path, index)
    assert len(frame) == 25
    assert audit["available_hours"] == 25


def test_edited_capture_time_fails_closed(tmp_path: Path) -> None:
    day = date(2026, 9, 30)
    cache = tmp_path / "cache"
    cutoff = jao.expected_cutoff_utc(day)
    source.capture_day(day=day, cache_root=cache, client=FakeClient(_fetch(day)),
                       now_utc=cutoff - pd.Timedelta(minutes=9))
    audit = cache / "raw/initialComputation" / f"{day}.audit.json"
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["retrieved_at_utc"] = (cutoff + pd.Timedelta(minutes=1)).isoformat()
    audit.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="after cutoff"):
        source.verify_daily_capture(cache, day)


def test_complete_ledger_produces_preflight_compatible_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    day = date(2026, 9, 30)
    cache = tmp_path / "cache"
    bundle = tmp_path / "bundle"
    cutoff = jao.expected_cutoff_utc(day)
    source.capture_day(day=day, cache_root=cache, client=FakeClient(_fetch(day)),
                       now_utc=cutoff - pd.Timedelta(minutes=9))
    current_capture = source.verify_daily_capture(cache, day)
    history = [{**current_capture, "delivery_day": (day - timedelta(days=offset)).isoformat()}
               for offset in range(365, -1, -1)]
    # This wiring fixture mocks RAW reconstruction below. Still provide every
    # named partition so publication must copy a self-contained source packet.
    current_bytes = [path.read_bytes() for path in source._paths(cache, day)]
    for item in history[:-1]:
        for path, raw in zip(source._paths(cache, date.fromisoformat(item["delivery_day"])),
                             current_bytes, strict=True):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    monkeypatch.setattr(source, "inspect_history", lambda _cache, _day: (history, []))
    full, _, current_cutoff = delivery_grid(day.isoformat())
    frame = pd.DataFrame(0.0, index=full, columns=source.FEATURE_COLUMNS)
    frame[source.AVAILABLE] = np.int8(0)
    monkeypatch.setattr(source, "build_strict_history_features",
                        lambda *_: (frame, {"historical_fallback": False,
                                           "imputation": False}))
    result = source.publish_jao_receipt(day=day, cache_root=cache, bundle=bundle)
    assert result["state"] == "COMPLETE"
    receipt = json.loads((bundle / "source_receipts/jao_initial.json").read_text(encoding="utf-8"))
    validate_source_receipt(receipt, group="jao_initial", day=day.isoformat(),
                            bundle=bundle, cutoff=current_cutoff)
    assert receipt["captured_training_and_delivery_days"] == 366
    assert receipt["asof_state_utc"] == current_capture["retrieved_at_utc"]
    assert (bundle / source.SOURCE_SUBDIR / source.FEATURE_NAME).is_file()
    captured_members = [name for name in receipt["artifact_sha256"] if "/captures/" in name]
    assert len(captured_members) == 366 * 4
    first = day - timedelta(days=365)
    assert (bundle / source.SOURCE_SUBDIR / "captures/raw/initialComputation" /
            f"{first.isoformat()}.json.gz").is_file()
