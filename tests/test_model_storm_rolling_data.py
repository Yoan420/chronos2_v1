"""Rolling CWE inputs reuse verified archives without refreshing or fitting."""
from __future__ import annotations

import json
from copy import deepcopy

import pandas as pd
import pytest

from chronos2_hourly import model_storm_data as loader
from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.storm_dashboard import STORM_DASHBOARD_COLUMN
from test_model_storm_data import DAY, frozen, refreshed, workspace, zone
from test_model_storm_history import archive, audited_sources, fingerprint, sha


@pytest.fixture
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("msr")


def rows_for(history, day, timezone="Europe/Paris"):
    return [row for row in history["rows"]
            if str(pd.Timestamp(row["timestamp_utc"]).tz_convert(timezone).date()) == day]


def test_rolling_reads_once_and_uses_latest_observations_never_training_labels(tmp_path, monkeypatch):
    frozen(tmp_path)
    refreshed(tmp_path, monkeypatch)
    counts = {"archive": 0, "storm": 0, "observed": 0}
    for attr, name in (("load_nuclear_result_bundle", "archive"),
                       ("_load_verified_snapshot", "storm"),
                       ("verify_refreshed_observations", "observed")):
        original = getattr(loader, attr)
        def spy(*args, _original=original, _name=name, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(loader, attr, spy)
    before = fingerprint(tmp_path)
    fr = zone(loader.load_model_storm_payload(tmp_path, DAY))
    history = fr["rolling_history"]
    assert counts == {"archive": 1, "storm": 1, "observed": 1}
    assert history["window_days"] == 366
    assert history["start_day"] == "2025-09-10"
    assert history["end_day"] == DAY
    assert history["expected_hours"] == len(history["rows"]) == 366 * 24
    assert history["status"] == "partial"  # Official Storm's audited autumn DST gap is not filled.
    assert history["coverage"] == {"model": 8784, "storm": 8783, "observed": 8784,
                                   "storm_dashboard_cache": 0, "model_p10": 8784, "model_p90": 8784}
    # This legacy fixture has no fallback-hour inventory: cache-only extraction
    # is unavailable, but the established merged comparator is unaffected.
    assert history["sources"]["storm_dashboard_cache"]["status"] == "invalid"
    assert all(row["model"] == 51.5 for row in history["rows"])
    assert all((row["model_p10"], row["model_p90"]) == (41.5, 61.5) for row in history["rows"])
    assert history["sources"]["model"]["quantile_columns"] == {
        "p10": "residual_kalman__q10", "p50": "residual_kalman__q50", "p90": "residual_kalman__q90"}
    assert history["sources"]["storm"]["quantile_columns"] == {"p50": STORM_DASHBOARD_COLUMN}
    assert history["sources"]["storm"]["quantile_interval_available"] is False
    assert all("storm_p10" not in row and "storm_p90" not in row for row in history["rows"])
    assert all(row["observed"] == 100 for row in history["rows"][:-24])
    assert all(row["observed"] == 120 for row in history["rows"][-24:])
    assert not any(row["observed"] == 49 for row in history["rows"]), "Frozen training actual must never leak into scores"
    assert history["sources"]["model"]["observations_from_backtest"] is False
    assert len(history["sources"]["model"]["artifacts"]) == 2
    for source in ("storm", "observed"):
        assert history["sources"][source]["artifact_sha256"] == fr["sources"][source]["artifact_sha256"]
        assert history["sources"][source]["audit_sha256"] == fr["sources"][source]["audit_sha256"]
    assert fingerprint(tmp_path) == before
    json.dumps(history, allow_nan=False)


@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2026-10-25", 25)])
def test_rolling_preserves_dst_physical_hours_without_interpolation(tmp_path, monkeypatch, day, hours):
    frozen(tmp_path, day=day)
    refreshed(tmp_path, monkeypatch, day=day, actual_count=hours, storm_count=hours)
    history = zone(loader.load_model_storm_payload(tmp_path, day))["rolling_history"]
    current = rows_for(history, day)
    expected = local_delivery_day_index(day, timezone="Europe/Paris")
    assert len(current) == hours
    assert [row["timestamp_utc"] for row in current] == [stamp.isoformat() for stamp in expected]
    assert len({row["timestamp_utc"] for row in history["rows"]}) == history["expected_hours"]
    assert all(row["model"] == 51.5 and row["observed"] == 120 for row in current)
    assert all((row["model_p10"], row["model_p90"]) == (41.5, 61.5) for row in current)
    assert history["coverage"]["storm"] == history["expected_hours"] - 1  # Audited earlier DST hole retained.


@pytest.mark.parametrize("rejected", [False, True])
def test_pending_or_rejected_delivery_does_not_remove_365_observed_history_days(tmp_path, monkeypatch, rejected):
    frozen(tmp_path)
    _, directory, audit = refreshed(tmp_path, monkeypatch, actual_count=0)
    if rejected:
        for field in ("observed", "canonical_actuals"):
            audit[field]["source"]["current_delivery_actual_reason"] = "post_auction_source_rejected"
        (directory / "statistics_history_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    fr = zone(loader.load_model_storm_payload(tmp_path, DAY))
    history = fr["rolling_history"]
    assert history["status"] == "partial"
    assert history["coverage"]["observed"] == history["expected_hours"] - 24
    assert all(row["observed"] is None for row in rows_for(history, DAY))
    assert all(row["observed"] == 100 for row in rows_for(history, "2026-09-09"))
    if rejected:
        assert fr["sources"]["observed"]["current_delivery_actual_reason"] == "post_auction_source_rejected"


def test_historical_report_never_contains_later_days_from_source_snapshot_or_forecast(tmp_path, monkeypatch):
    target, source = "2026-09-14", "2026-09-15"
    archive(tmp_path, source_day=source, target_day=target)
    audited_sources(tmp_path, monkeypatch, source, target)
    history = zone(loader.load_model_storm_payload(tmp_path, target, history_from_delivery=source))["rolling_history"]
    assert history["start_day"] == "2025-09-14"
    assert history["end_day"] == target
    assert max(pd.Timestamp(row["timestamp_utc"]).tz_convert("Europe/Paris").date().isoformat()
               for row in history["rows"]) == target
    assert len(history["sources"]["model"]["artifacts"]) == 1
    assert history["sources"]["model"]["artifacts"][0]["artifact_path"].endswith("kalman_backtest.parquet")
    assert history["coverage"]["model"] == history["expected_hours"] - 24
    for name in ("model", "storm", "observed"):
        assert history["sources"][name]["source_delivery_day"] == source
        assert history["sources"][name]["delivery_day"] == target
    assert [row["observed"] for row in rows_for(history, target)] == list(range(1000, 1024))
    assert [(row["model_p10"], row["model"], row["model_p90"])
            for row in rows_for(history, target)] == [(n - 28, n - 20, n - 8) for n in range(24)]


def test_missing_zones_remain_explicitly_audited_with_empty_values(tmp_path):
    payload = loader.load_model_storm_payload(tmp_path, DAY)
    for item in payload["zones"]:
        history = item["rolling_history"]
        assert history["status"] == "unavailable"
        assert set(history["sources"]) == {"model", "storm", "observed", "storm_dashboard_cache"}
        assert all(source["status"] == "unavailable" for source in history["sources"].values())
        assert history["coverage"] == {"model": 0, "storm": 0, "observed": 0, "storm_dashboard_cache": 0,
                                       "model_p10": 0, "model_p90": 0}
        assert all(row["model"] is row["storm"] is row["observed"] is None for row in history["rows"])


@pytest.mark.parametrize("artifact,invalid,other", [
    ("inputs/observed_latest.parquet", "observed", "storm"),
    ("inputs/storm_dashboard_official_statistics.parquet", "storm", "observed"),
])
def test_latest_corrupted_comparator_invalidates_rolling_without_fallback(tmp_path, monkeypatch, artifact, invalid, other):
    frozen(tmp_path)
    refreshed(tmp_path, monkeypatch)
    _, newest, _ = refreshed(tmp_path, monkeypatch)
    (newest / artifact).write_bytes(b"intentionally corrupted fixture")
    fr = zone(loader.load_model_storm_payload(tmp_path, DAY))
    history = fr["rolling_history"]
    assert history["sources"][invalid]["status"] == "invalid"
    assert history["coverage"][invalid] == 0
    assert history["coverage"][other] > 8700
    assert all(row[invalid] is None for row in history["rows"])
    assert fr["sources"][invalid]["status"] == "invalid"


def test_corrupted_historical_bundle_does_not_use_unverified_forecast_or_actuals(tmp_path, monkeypatch):
    directory = frozen(tmp_path)
    refreshed(tmp_path, monkeypatch)
    (directory / "kalman_backtest.parquet").write_bytes(b"intentionally corrupted fixture")
    history = zone(loader.load_model_storm_payload(tmp_path, DAY))["rolling_history"]
    assert history["sources"]["model"]["status"] == "invalid"
    assert history["coverage"]["model"] == 0
    assert history["coverage"]["observed"] > 8700
    assert history["coverage"]["model_p10"] == history["coverage"]["model_p90"] == 0
    assert all(row["model"] is row["model_p10"] is row["model_p90"] is None for row in history["rows"])


@pytest.mark.parametrize("damage", ["missing", "nonfinite", "crossed"])
def test_rolling_rejects_invalid_historical_quantiles_without_synthetic_replacement(tmp_path, monkeypatch, damage):
    bundle = archive(tmp_path, source_day=DAY, target_day="2026-09-09")
    frame = bundle.result.kalman_view.backtest
    if damage == "missing":
        frame.drop(columns="residual_kalman__q10", inplace=True)
    else:
        column = frame.columns.get_loc("residual_kalman__q10")
        frame.iloc[0, column] = float("nan") if damage == "nonfinite" else 1000.0
    monkeypatch.setattr(loader, "load_nuclear_result_bundle", lambda **kwargs: bundle.result)
    fr = zone(loader.load_model_storm_payload(tmp_path, DAY))
    assert fr["coverage"]["model"] == 24, "A rejected historical interval must not change a valid live forecast"
    history = fr["rolling_history"]
    assert history["sources"]["model"]["status"] == "invalid"
    assert history["coverage"]["model"] == history["coverage"]["model_p10"] == history["coverage"]["model_p90"] == 0
    assert all(row["model"] is row["model_p10"] is row["model_p90"] is None for row in history["rows"])
    json.dumps(history, allow_nan=False)


def test_unreadable_latest_audit_rejects_all_comparator_history(tmp_path, monkeypatch):
    frozen(tmp_path)
    _, directory, _ = refreshed(tmp_path, monkeypatch)
    (directory / "statistics_history_audit.json").write_text("not json", encoding="utf-8")
    history = zone(loader.load_model_storm_payload(tmp_path, DAY))["rolling_history"]
    assert history["coverage"]["model"] == history["expected_hours"]
    assert history["coverage"]["observed"] == history["coverage"]["storm"] == 0
    assert history["sources"]["observed"]["status"] == history["sources"]["storm"]["status"] == "invalid"


def test_verified_historical_storm_without_delivery_placeholder_remains_usable_for_rolling(tmp_path, monkeypatch):
    frozen(tmp_path)
    _, directory, audit = refreshed(tmp_path, monkeypatch)
    path = directory / "inputs/storm_dashboard_official_statistics.parquet"
    frame = pd.read_parquet(path).iloc[:-24].copy()
    frame.to_parquet(path, index=False)
    storm = audit["storm_dashboard"]
    storm.pop("delivery_placeholder")
    storm["period"]["end_utc"] = str(frame.delivery_start_utc.iloc[-1])
    storm["period"]["end_local_day"] = "2026-09-09"
    storm["expected_hours"] = storm["source_rows"] = len(frame)
    storm["available_hours"] = len(frame) - storm["missing_hours"]
    storm["normalized_artifact_sha256"] = sha(path)
    (directory / "statistics_history_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    fr = zone(loader.load_model_storm_payload(tmp_path, DAY))
    assert fr["coverage"]["storm"] == 0
    history = fr["rolling_history"]
    assert history["coverage"]["storm"] == history["expected_hours"] - 25
    assert history["sources"]["storm"]["status"] == "partial"
    assert all(row["storm"] is None for row in rows_for(history, DAY))
    assert all(row["storm"] == 105 for row in rows_for(history, "2026-09-09"))


def with_cache_metadata(directory, audit, days):
    """Supply real collector fields absent from the legacy refresh fixture."""
    path = directory / "inputs/storm_dashboard_official_statistics.parquet"
    frame = pd.read_parquet(path)
    index = pd.DatetimeIndex(frame.delivery_start_utc)
    removed = sum(len(local_delivery_day_index(day, timezone="Europe/Paris")) for day in days)
    available = int(frame[STORM_DASHBOARD_COLUMN].notna().sum()) - removed
    source = audit["storm_dashboard"]["source"]
    source.update(fallback_used_local_days=days, fallback_used_hours=removed,
                  cache_available_hours=available, cache_missing_hours=len(index) - available)
    (directory / "statistics_history_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    return source


def test_cache_only_rolling_excludes_fallback_without_changing_existing_storm_or_day_chart(tmp_path, monkeypatch):
    frozen(tmp_path)
    _, directory, audit = refreshed(tmp_path, monkeypatch)
    with_cache_metadata(directory, audit, ["2026-09-08", DAY])
    before = fingerprint(tmp_path)
    fr = zone(loader.load_model_storm_payload(tmp_path, DAY))
    history = fr["rolling_history"]
    assert history["schema_version"] == 2
    assert history["coverage"]["storm_dashboard_cache"] == 8783 - 48
    assert history["coverage"]["storm"] == 8783
    for day in ("2026-09-08", DAY):
        selected = rows_for(history, day)
        assert len(selected) == 24
        assert all(row["storm_dashboard_cache"] is None and row["storm"] is not None for row in selected)
    assert all(row["storm_dashboard_cache"] == row["storm"] == 105
               for row in rows_for(history, "2026-09-09"))
    source = history["sources"]["storm_dashboard_cache"]
    assert source["method"] == "audited_complete_fallback_days"
    assert source["exact_cache_hour_mask_verified"] is True
    assert source["official_dashboard_formula_verified"] is False
    assert source["raw_official_hourly_export_verified"] is False
    assert source["removed_fallback_hours"] == 48
    assert source["artifact_sha256"] == history["sources"]["storm"]["artifact_sha256"]
    assert source["audit_sha256"] == history["sources"]["storm"]["audit_sha256"]
    assert all(row["storm"] == 125 for row in fr["rows"])
    assert all("storm_dashboard_cache" not in row for row in fr["rows"])
    assert set(fr["sources"]) == {"observed", "storm", "model"}
    assert fingerprint(tmp_path) == before


def test_cache_only_history_is_available_without_any_model(tmp_path, monkeypatch):
    _, directory, audit = refreshed(tmp_path, monkeypatch)
    with_cache_metadata(directory, audit, [])
    history = zone(loader.load_model_storm_payload(tmp_path, DAY))["rolling_history"]
    assert history["coverage"]["model"] == 0
    assert history["coverage"]["storm_dashboard_cache"] == history["coverage"]["storm"] == 8783
    assert all(row["storm_dashboard_cache"] == row["storm"] for row in history["rows"])


def test_cache_only_historical_slice_keeps_source_day_identity_and_no_future_values(tmp_path, monkeypatch):
    target, source = "2026-09-14", "2026-09-15"
    archive(tmp_path, source_day=source, target_day=target)
    directory, audit, _ = audited_sources(tmp_path, monkeypatch, source, target)
    with_cache_metadata(directory, audit, [target, source])
    history = zone(loader.load_model_storm_payload(tmp_path, target, history_from_delivery=source))["rolling_history"]
    metadata = history["sources"]["storm_dashboard_cache"]
    assert metadata["exact_cache_hour_mask_verified"] is True
    assert metadata["source_delivery_day"] == source and metadata["delivery_day"] == target
    assert all(row["storm_dashboard_cache"] is None and row["storm"] is not None
               for row in rows_for(history, target))
    assert rows_for(history, source) == []


@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2026-10-25", 25)])
def test_cache_only_uses_physical_dst_fallback_capacity(tmp_path, monkeypatch, day, hours):
    _, directory, audit = refreshed(tmp_path, monkeypatch, day=day, actual_count=hours, storm_count=hours)
    with_cache_metadata(directory, audit, [day])
    fr = zone(loader.load_model_storm_payload(tmp_path, day))
    history = fr["rolling_history"]
    assert history["sources"]["storm_dashboard_cache"]["removed_fallback_hours"] == hours
    selected = rows_for(history, day)
    assert len(selected) == hours
    assert all(row["storm_dashboard_cache"] is None and row["storm"] == 125 for row in selected)
    assert fr["coverage"]["storm"] == hours


@pytest.mark.parametrize("change", [
    {"fallback_used_hours": 1},
    {"fallback_used_local_days": [DAY, DAY]},
    {"fallback_used_local_days": ["2026-09-11"]},
    {"fallback_used_local_days": ["2026-09-10T00:00:00"]},
    {"fallback_used_hours": True},
    {"cache_available_hours": 100},
    {"cache_missing_hours": 0},
    {"fallback_used_local_days": None},
])
def test_unprovable_cache_mask_fails_closed_without_losing_valid_merged_storm(tmp_path, monkeypatch, change):
    _, directory, audit = refreshed(tmp_path, monkeypatch)
    source = with_cache_metadata(directory, audit, [DAY])
    source.update(change)
    (directory / "statistics_history_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    fr = zone(loader.load_model_storm_payload(tmp_path, DAY))
    history = fr["rolling_history"]
    assert history["coverage"]["storm_dashboard_cache"] == 0
    assert history["sources"]["storm_dashboard_cache"]["status"] == "invalid"
    assert history["sources"]["storm_dashboard_cache"]["exact_cache_hour_mask_verified"] is False
    assert history["coverage"]["storm"] == 8783
    assert fr["coverage"]["storm"] == 24
    assert all(row["storm_dashboard_cache"] is None for row in history["rows"])


def test_cache_mask_rejects_fallback_claim_for_unavailable_merged_hour():
    expected = local_delivery_day_index(DAY, timezone="Europe/Paris")
    values = pd.Series(100.0, index=expected)
    values.iloc[0] = float("nan")
    audit = {"source": {"cache_precedence": True,
                        "fallback_policy": "native_only_where_day_ahead_cache_is_missing",
                        "fallback_used_local_days": [DAY], "fallback_used_hours": 24,
                        "cache_available_hours": 0, "cache_missing_hours": 24}}
    with pytest.raises(ValueError, match="unavailable merged"):
        loader._cache_only_history(values, audit, timezone="Europe/Paris")


def test_cache_mask_requires_explicit_counts_instead_of_guessing_no_fallback():
    expected = local_delivery_day_index(DAY, timezone="Europe/Paris")
    values = pd.Series(100.0, index=expected)
    source = {"cache_precedence": True, "fallback_policy": "native_only_where_day_ahead_cache_is_missing",
              "fallback_used_local_days": [], "fallback_used_hours": 0,
              "cache_available_hours": 24, "cache_missing_hours": 0}
    for field in ("fallback_used_hours", "cache_available_hours", "cache_missing_hours"):
        damaged = deepcopy(source)
        damaged.pop(field)
        with pytest.raises(ValueError, match=field):
            loader._cache_only_history(values, {"source": damaged}, timezone="Europe/Paris")


def test_corrupted_storm_invalidates_cache_only_as_well(tmp_path, monkeypatch):
    _, directory, audit = refreshed(tmp_path, monkeypatch)
    with_cache_metadata(directory, audit, [])
    (directory / "inputs/storm_dashboard_official_statistics.parquet").write_bytes(b"bad fixture")
    history = zone(loader.load_model_storm_payload(tmp_path, DAY))["rolling_history"]
    assert history["sources"]["storm_dashboard_cache"]["status"] == "invalid"
    assert history["coverage"]["storm_dashboard_cache"] == history["coverage"]["storm"] == 0


def test_changing_source_audit_invalidates_cache_only_as_well(tmp_path, monkeypatch):
    _, directory, audit = refreshed(tmp_path, monkeypatch)
    with_cache_metadata(directory, audit, [])
    audit_path = directory / "statistics_history_audit.json"
    original = loader._sha256
    count = 0

    def changed(path):
        nonlocal count
        if path == audit_path:
            count += 1
            if count >= 2:
                return "0" * 64
        return original(path)

    monkeypatch.setattr(loader, "_sha256", changed)
    history = zone(loader.load_model_storm_payload(tmp_path, DAY))["rolling_history"]
    assert history["coverage"]["storm_dashboard_cache"] == 0
    assert history["sources"]["storm_dashboard_cache"]["status"] == "invalid"
    assert "changed" in history["sources"]["storm_dashboard_cache"]["error"]
