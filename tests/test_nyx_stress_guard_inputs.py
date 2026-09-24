import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nyx_stress_guard import inputs as module, ledger, runner
from test_nyx_stress_guard_features import panel


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_fixture(tmp_path, monkeypatch, *, observed=False):
    import run_chronos2_exogenous_panel as resolver
    contracts = {}
    calls = []
    for zone in module.ZONES:
        directory = tmp_path/"data/cache"/zone
        directory.mkdir(parents=True)
        cache = directory/"canonical.csv.gz"
        cache.write_bytes(b"unchanged canonical cache "+zone.encode())
        base, live = directory/"base.yaml", directory/"live.yaml"
        base.write_text("constant base"); live.write_text("constant live")
        contracts[zone] = {"series": "canonical.exact."+zone, "cache_path": str(cache),
                           "base_config": str(base), "live_config": str(live)}
    monkeypatch.setattr(resolver, "_canonical_target_path", lambda root, zone: (Path(contracts[zone]["cache_path"]), contracts[zone]))
    def capture(**kwargs):
        calls.append(kwargs)
        first = module._now()
        days = pd.date_range(kwargs["start"], kwargs["end"], freq="D").strftime("%Y-%m-%d")
        index = pd.DatetimeIndex([t for day in days for t in ledger._index(day)])
        values = pd.Series(100. if observed else np.nan, index=index)
        contract = contracts[kwargs["zone"]]
        return values, {**contract, "fresh_api_read": True, "nocache": True,
            "fallback_used": False, "source_cache_modified": False,
            "canonical_source_cache_sha256": digest(Path(contract["cache_path"])),
            "capture_started_at_utc": first.isoformat(), "capture_completed_at_utc": module._now().isoformat()}
    monkeypatch.setattr(module, "_capture_target", capture)
    return calls, contracts, capture


def test_fresh_check_queries_exact_canonical_all_countries_without_mutation(tmp_path, monkeypatch):
    calls, contracts, _ = target_fixture(tmp_path, monkeypatch)
    before = {z: Path(c["cache_path"]).read_bytes() for z, c in contracts.items()}
    started = module._now()
    receipt = module.fresh_label_check(tmp_path, "2026-09-16", list(module.ZONES))
    completed = module._now()
    ledger._fresh_receipt(receipt, "2026-09-16", list(module.ZONES), started, completed)
    assert len(calls) == 4 and all(c["refresh"] is True for c in calls)
    assert receipt["observed_hours_by_zone"] == dict.fromkeys(module.ZONES, 0)
    assert all(Path(contracts[z]["cache_path"]).read_bytes() == before[z] for z in contracts)
    assert receipt["canonical_cache_modified"] is False


def test_observed_values_are_counted_not_mistaken_for_missing_or_hidden(tmp_path, monkeypatch):
    target_fixture(tmp_path, monkeypatch, observed=True)
    receipt = module.fresh_label_check(tmp_path, "2025-10-26", ["DE", "BE"])
    assert receipt["observed_hours_by_zone"] == {"DE": 25, "BE": 25}


def test_network_failure_does_not_return_fake_zero_observation_receipt(tmp_path, monkeypatch):
    target_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "_capture_target", lambda **kwargs: (_ for _ in ()).throw(ConnectionError("offline")))
    with pytest.raises(ConnectionError, match="offline"):
        module.fresh_label_check(tmp_path, "2026-09-16", ["FR"])


@pytest.mark.parametrize("bad", ["cached", "fallback", "wrong_series", "wrong_hash", "naive", "duplicate", "foreign_hour", "infinite", "clock"])
def test_bad_canonical_receipts_or_time_grid_fail_closed(tmp_path, monkeypatch, bad):
    _, _, original = target_fixture(tmp_path, monkeypatch)
    def wrong(**kwargs):
        values, audit = original(**kwargs)
        if bad == "cached": audit["nocache"] = False
        elif bad == "fallback": audit["fallback_used"] = True
        elif bad == "wrong_series": audit["series"] = "some.other.series"
        elif bad == "wrong_hash": audit["canonical_source_cache_sha256"] = "0"*64
        elif bad == "naive": values.index = values.index.tz_localize(None)
        elif bad == "duplicate": values = pd.concat([values, values.iloc[:1]])
        elif bad == "foreign_hour": values.index += pd.Timedelta(hours=1)
        elif bad == "infinite": values.iloc[0] = np.inf
        else: audit["capture_completed_at_utc"] = "2099-01-01T00:00:00Z"
        return values, audit
    monkeypatch.setattr(module, "_capture_target", wrong)
    with pytest.raises(ValueError):
        module.fresh_label_check(tmp_path, "2026-09-16", ["FR"])


def test_cross_country_cache_change_detected_after_prior_country_read(tmp_path, monkeypatch):
    _, contracts, original = target_fixture(tmp_path, monkeypatch)
    def race(**kwargs):
        result = original(**kwargs)
        if kwargs["zone"] == "DE": Path(contracts["FR"]["cache_path"]).write_bytes(b"concurrent external writer")
        return result
    monkeypatch.setattr(module, "_capture_target", race)
    with pytest.raises(ValueError, match="changed during"):
        module.fresh_label_check(tmp_path, "2026-09-16", ["FR", "DE"])


def test_observation_collection_preserves_nan_dst_and_selects_requested_days_only(tmp_path, monkeypatch):
    calls, _, _ = target_fixture(tmp_path, monkeypatch)
    frame, receipt = module.collect_observations(tmp_path, ["2025-10-27", "2025-10-25"], ["FR", "DE"])
    assert len(frame) == 96 and frame.actual.isna().all()
    assert len(calls) == 2 and calls[0]["start"] == "2025-10-25" and calls[0]["end"] == "2025-10-27"
    assert receipt["kind"] == "canonical_target_observations" and receipt["missing_hours"] == 96
    assert receipt["forecast_recalculated"] is False
    spring, _ = module.collect_observations(tmp_path, "2026-03-29", ["FR"])
    assert len(spring) == 23


@pytest.mark.parametrize("days", [[], ["2026-01-01"]*2, ["2026-01-01", "2026-02-01"], ["bad"]])
def test_observation_span_and_dates_bounded_before_api(tmp_path, monkeypatch, days):
    calls, _, _ = target_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError): module.collect_observations(tmp_path, days, ["FR"])
    assert not calls


def capture_fixture(tmp_path, monkeypatch, day="2026-09-16"):
    snapshot = tmp_path/runner.NAMESPACE/"snapshots/test"
    write_json(snapshot/"source_audit.json", {"source_config": {"baseline_model": "nuclear_kalman", "data": {}}})
    monkeypatch.setattr(runner, "read_suite", lambda value, root: (snapshot, {}, {}))
    monkeypatch.setattr(runner, "verify_result", lambda *args: {"status": "completed"})
    now = pd.Timestamp(day, tz="Europe/Paris")-pd.Timedelta(hours=13)  # D-1 11:00
    monkeypatch.setattr(module, "_now", lambda: now)
    checks = []
    def fresh(root, delivery, zones):
        checks.append(delivery)
        return {"observed_hours_by_zone": dict.fromkeys(zones, 0), "schema_version": 1}
    monkeypatch.setattr(module, "fresh_label_check", fresh)
    physical = tmp_path/"data/pit/physical.parquet"
    physical.parent.mkdir(parents=True); physical.write_bytes(b"unchanged physical inputs")
    source = {"path": str(physical), "sha256": digest(physical), "finite_hours": len(ledger._index(day))}
    selection = {"selected": {"fr_residual_load": source}, "optional_missing_hours": {}, "completed_refresh_manifests_considered": []}
    def select(root, config, delivery):
        return {**config, "delivery_day": delivery, "end_day": None}, selection
    monkeypatch.setattr(module, "_select_sources", select)
    frame = panel(day)
    frame["actual"] = np.nan
    frame["benchmark_forecast"] = 123.
    columns = [c for c in frame if c.startswith("feature_") and c != "feature_eligible"]
    audit = {"feature_columns": columns, "sources": {"fr_residual_load": source}, "baseline": {"sources": []}}
    monkeypatch.setattr(module.scarcity_data, "load_inputs", lambda config, root: (frame.copy(deep=True), audit))
    output = tmp_path/runner.NAMESPACE/"captures/fresh"
    return snapshot, output, frame, audit, checks, physical


def test_capture_sanitizes_labels_storm_preserves_baseline_and_seals_panel(tmp_path, monkeypatch):
    snapshot, output, original, _, checks, physical = capture_fixture(tmp_path, monkeypatch)
    before = physical.read_bytes()
    result, receipt = module.capture_inputs(tmp_path, snapshot, "2026-09-16", output)
    assert checks == ["2026-09-16", "2026-09-16"]
    assert "actual" not in result and "benchmark_forecast" not in result
    saved = pd.read_parquet(output/"panel.parquet")
    pd.testing.assert_frame_equal(saved, result, check_exact=True)
    expected = original.sort_values(["zone", "timestamp_utc"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(result[["forecast", "q10", "q90"]], expected[["forecast", "q10", "q90"]], check_exact=True)
    assert receipt["publication_evidence"] == "historical_asof" and not receipt["source_publication_certified"]
    assert not receipt["model_refitted"] and not receipt["physical_sources_refreshed"]
    assert physical.read_bytes() == before
    ledger._input_evidence(tmp_path, receipt, "2026-09-16", module._now())
    assert receipt["artifacts"]["panel"]["sha256"] == digest(output/"panel.parquet")


@pytest.mark.parametrize("date,n", [("2026-03-29", 92), ("2025-10-26", 100)])
def test_capture_accepts_dst_complete_physical_day(tmp_path, monkeypatch, date, n):
    snapshot, output, _, _, _, _ = capture_fixture(tmp_path, monkeypatch, date)
    frame, _ = module.capture_inputs(tmp_path, snapshot, date, output)
    assert len(frame) == n


@pytest.mark.parametrize("bad", ["known_before", "known_after", "report_actual", "missing_hour", "missing_country", "duplicate", "quantile", "oracle", "physical_hole", "future_cutoff", "source_race", "existing"])
def test_invalid_capture_never_publishes_ready_evidence(tmp_path, monkeypatch, bad):
    snapshot, output, frame, audit, checks, physical = capture_fixture(tmp_path, monkeypatch)
    if bad in {"known_before", "known_after"}:
        def known(root, day, zones):
            checks.append(day)
            return {"observed_hours_by_zone": {z: int(bad == "known_before" or len(checks) > 1) for z in zones}}
        monkeypatch.setattr(module, "fresh_label_check", known)
    elif bad == "report_actual": frame.loc[0, "actual"] = 200.
    elif bad == "missing_hour": frame.drop(index=0, inplace=True)
    elif bad == "missing_country": frame.drop(index=frame.loc[frame.zone.eq("NL")].index, inplace=True)
    elif bad == "duplicate":
        duplicate = pd.concat([frame, frame.iloc[:1]])
        monkeypatch.setattr(module.scarcity_data, "load_inputs", lambda config, root: (duplicate, audit))
    elif bad == "quantile": frame["q90"] = 50.
    elif bad == "oracle":
        frame["feature_actual_oracle"] = 999.
        audit["feature_columns"].append("feature_actual_oracle")
    elif bad == "physical_hole": frame["feature_de_solar_generation_gw"] = np.nan
    elif bad == "future_cutoff": monkeypatch.setattr(module, "_now", lambda: pd.Timestamp("2026-09-15T05:00Z"))
    elif bad == "source_race": physical.write_bytes(b"concurrent update after selection")
    elif bad == "existing": output.mkdir(parents=True)
    with pytest.raises(ValueError): module.capture_inputs(tmp_path, snapshot, "2026-09-16", output)
    assert not (output/"input_evidence.json").exists()


def test_capture_output_cannot_escape_private_namespace(tmp_path, monkeypatch):
    snapshot, _, _, _, checks, _ = capture_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError): module.capture_inputs(tmp_path, snapshot, "2026-09-16", tmp_path/"runs/exports/overwrite")
    assert not checks


def completed_refresh(root, token, day, path="data/pit/fresh.parquet", *, status="complete", key="fr_residual_load"):
    directory = root/"runs/experiments/nyx_scarcity_v1/source_refresh"/token
    config = directory/"config.json"
    write_json(config, {"data": {"source_overrides": {key: path}}})
    write_json(directory/"refresh_audit.json", {"status": status, "delivery_day": day,
        "required_sources_complete": status == "complete", "original_sources_modified": False,
        "saved_config": str(config), "source_dir": str(directory)})
    return directory


def test_refresh_discovery_is_bounded_ignores_failed_and_old_day(tmp_path):
    for i in range(25):
        completed_refresh(tmp_path, f"202609{i+1:02d}T120000Z_00000000", "2026-09-16")
    candidates = module._refresh_candidates(tmp_path, "2026-09-16")
    assert len(candidates) == 20
    completed_refresh(tmp_path, "20260926T120000Z_00000000", "2026-09-16", status="failed")
    completed_refresh(tmp_path, "20260927T120000Z_00000000", "2026-09-15")
    assert len(module._refresh_candidates(tmp_path, "2026-09-16")) == 18


def test_foreign_override_and_manifest_path_rejected(tmp_path):
    directory = completed_refresh(tmp_path, "20260914T120000Z_00000000", "2026-09-16", key="storm_oracle")
    with pytest.raises(ValueError, match="exact registered"): module._refresh_candidates(tmp_path, "2026-09-16")
    write_json(directory/"config.json", {"data": {"source_overrides": {}}})
    path = directory/"refresh_audit.json"
    audit = json.loads(path.read_text()); audit["source_dir"] = str(tmp_path)
    write_json(path, audit)
    with pytest.raises(ValueError, match="own directory|inside the project"): module._refresh_candidates(tmp_path, "2026-09-16")


def test_source_selection_prefers_qualified_private_then_operational_without_changing_recipe(tmp_path, monkeypatch):
    completed_refresh(tmp_path, "20260914T120000Z_00000000", "2026-09-16", path="data/pit/private.parquet")
    registry = {"fr_residual_load": {"path": "data/pit/current.parquet", "feature": "feature_fr_residual_load_gw"}}
    monkeypatch.setattr(module.scarcity_data, "source_registry", lambda: registry)
    calls = []
    def read(root, key, spec, expected):
        calls.append(spec["path"])
        value = np.nan if "private" in spec["path"] else 10.
        return pd.DataFrame({spec["feature"]: value}, index=expected), {"finite_hours": 0 if np.isnan(value) else len(expected)}
    monkeypatch.setattr(module.scarcity_data, "_read_source", read)
    config = {"data": {"ccgt_efficiency": .58, "source_overrides": {"fr_residual_load": "data/pit/seed.parquet"}}, "policy": {"unchanged": 1}}
    resolved, audit = module._select_sources(tmp_path, config, "2026-09-16")
    assert calls == ["data/pit/private.parquet", "data/pit/current.parquet"]
    assert resolved["data"]["source_overrides"]["fr_residual_load"] == "data/pit/current.parquet"
    assert resolved["data"]["ccgt_efficiency"] == .58 and resolved["policy"] == config["policy"]
    assert config["data"]["source_overrides"]["fr_residual_load"] == "data/pit/seed.parquet"


def test_required_source_gap_fails_with_refresh_hint_optional_gap_preserved(tmp_path, monkeypatch):
    registry = {"fr_residual_load": {"path": "data/pit/current.parquet", "feature": "feature_fr_residual_load_gw"}}
    monkeypatch.setattr(module.scarcity_data, "source_registry", lambda: registry)
    monkeypatch.setattr(module.scarcity_data, "_read_source", lambda root, key, spec, expected: (pd.DataFrame({spec["feature"]: np.nan}, index=expected), {}))
    with pytest.raises(ValueError, match="Scarcity.ps1 -Action Refresh"):
        module._select_sources(tmp_path, {"data": {}}, "2026-09-16")
    registry = {"fr_temperature": {"path": "data/pit/temp.parquet", "feature": "feature_fr_temperature_c"}}
    resolved, audit = module._select_sources(tmp_path, {"data": {}}, "2026-09-16")
    assert audit["optional_missing_hours"] == {"fr_temperature": 24}
