from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import materialize_saturn_kalman_fuel as fuel
from nyx_scarcity import fuel_refresh as refresh
from nyx_scarcity.data import ScarcityDataError


def daily(days, **kwargs):
    index = pd.DatetimeIndex(days)
    elapsed = (index - pd.Timestamp("2020-01-01")).days.to_numpy(float)
    cutoff = pd.DatetimeIndex([fuel._civil_cutoff(day).tz_convert("UTC") for day in index])
    return pd.DataFrame({
        "ttf_m1_eur_mwh_th": 30 + elapsed * .01,
        "ttf_m1_eur_mwh_th__source_value_time_utc": cutoff - pd.Timedelta(hours=10),
        "ttf_m1_eur_mwh_th__cutoff_utc": cutoff,
        "eua_first_dec_eur_tco2": 60 + elapsed * .02,
        "eua_first_dec_eur_tco2__source_value_time_utc": cutoff - pd.Timedelta(hours=12),
        "eua_first_dec_eur_tco2__cutoff_utc": cutoff, "cutoff_time_utc": cutoff,
    }, index=index)


@pytest.fixture
def seed(tmp_path, monkeypatch):
    monkeypatch.setattr(fuel, "_fetch_daily_market", daily)
    directory = tmp_path / "data/pit/marginal_cost_expert/fuel"
    args = ["--output-dir", str(directory), "--start-day", "2026-03-27", "--end-day", "2026-03-28", "--skip-residual-load"]
    assert fuel.main(args) == 0
    path = directory / fuel.OUTPUT_NAME
    config = {"delivery_day": "2026-03-30", "zones": ["DE", "BE"], "data": {"source_overrides": {}},
              "policy": {"training_window_days": 365}, "diagnostic_only": True}
    return tmp_path, path, config


def execute_fake(monkeypatch, *, after=None):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        result = fuel.main(command[2:])
        if after:
            directory = Path(command[command.index("--output-dir") + 1])
            after(directory / fuel.OUTPUT_NAME)
        return SimpleNamespace(returncode=result)
    monkeypatch.setattr(refresh.subprocess, "run", run)
    return calls


def test_suffix_refresh_is_private_exact_and_dst_safe(seed, monkeypatch):
    root, path, config = seed
    before, original = path.read_bytes(), deepcopy(config)
    sidecar = Path(str(path) + ".audit.json")
    audit_before = sidecar.read_bytes()
    calls = execute_fake(monkeypatch)
    result = refresh.refresh_fuels(config, root=root)
    assert result["required_sources_complete"]
    assert result["status"] == "complete"
    assert result["seed_history_semantically_unchanged"]
    assert result["original_sources_unchanged"]
    assert path.read_bytes() == before and sidecar.read_bytes() == audit_before
    assert config == original
    private = root / result["config"]["data"]["source_overrides"]["ttf"]
    assert private != path
    assert private.parent.name == "fuel"
    assert private.is_relative_to(root / "runs/experiments/nyx_scarcity_v1/source_refresh")
    assert result["config"]["data"]["source_overrides"]["ttf"] == result["config"]["data"]["source_overrides"]["eua"]
    assert len(pd.read_parquet(private)) == 95  # four civil days including spring23h
    assert result["requested_suffix_days"] == 2
    assert result["overlap_query_days"] == 20
    assert result["maximum_series_day_queries"] == 44
    command, kwargs = calls[0]
    assert command[command.index("--start-day") + 1] == "2026-03-27"
    assert command[command.index("--end-day") + 1] == "2026-03-30"
    assert "--skip-residual-load" in command and "--force-rebuild" not in command
    assert command[command.index("--series-workers") + 1] == "1"
    assert command[command.index("--day-workers") + 1] == "1"
    assert kwargs["shell"] is False and kwargs["cwd"] == root
    assert kwargs["timeout"] > 0
    assert Path(result["saved_config"]).is_file() and Path(result["audit_path"]).is_file()


def test_already_complete_seed_never_queries(seed, monkeypatch):
    root, path, config = seed
    config["delivery_day"] = "2026-03-28"
    monkeypatch.setattr(refresh.subprocess, "run", lambda *a, **k: pytest.fail("No network/process for complete prefix"))
    result = refresh.refresh_fuels(config, root=root)
    assert result["required_sources_complete"]
    assert result["command"] == []
    assert result["maximum_series_day_queries"] == 0


def test_more_than_30_suffix_days_rejected_before_writes(seed, monkeypatch):
    root, path, config = seed
    config["delivery_day"] = "2026-05-01"
    monkeypatch.setattr(refresh.subprocess, "run", lambda *a, **k: pytest.fail("No full backfill"))
    with pytest.raises(ScarcityDataError, match="limited to 30"):
        refresh.refresh_fuels(config, root=root)
    assert not (root / "runs").exists()


def test_command_preserves_audited_seed_assumptions_instead_of_model_cost_parameters(seed, monkeypatch):
    root, path, config = seed
    # Build another independently audited seed contract, deliberately unlike
    # the challenger's thermal-emission factor used only in its own features.
    assert fuel.main(["--output-dir", str(path.parent), "--start-day", "2026-03-27", "--end-day", "2026-03-28",
                      "--skip-residual-load", "--force-rebuild", "--ccgt-efficiency", ".61",
                      "--ccgt-emission-tco2-mwh", ".31", "--ccgt-vom-eur-mwh", "4.5"]) == 0
    config["data"].update(ccgt_efficiency=.58, gas_emissions_tco2_mwh_th=.202)
    calls = execute_fake(monkeypatch)
    result = refresh.refresh_fuels(config, root=root)
    command = calls[0][0]
    assert result["required_sources_complete"]
    assert command[command.index("--ccgt-efficiency") + 1] == "0.61"
    assert command[command.index("--ccgt-emission-tco2-mwh") + 1] == "0.31"
    assert command[command.index("--ccgt-vom-eur-mwh") + 1] == "4.5"
    assert result["config"]["data"]["gas_emissions_tco2_mwh_th"] == .202
    assert not result["native_saturn_cgc_used"]


def rewrite(path, mutator):
    frame = pd.read_parquet(path)
    sidecar = Path(str(path) + ".audit.json")
    audit = json.loads(sidecar.read_text())
    mutator(frame)
    fuel._write_bundle(frame, path, sidecar, audit)


def test_valid_hash_does_not_hide_changed_historical_derived_values(seed, monkeypatch):
    root, source, config = seed
    def changed(path):
        def mutate(frame):
            prefix = pd.to_datetime(frame.value_time_utc, utc=True) < pd.Timestamp("2026-03-28", tz="Europe/Paris")
            frame.loc[prefix, "ttf_change_1d"] += 1.
        rewrite(path, mutate)
    execute_fake(monkeypatch, after=changed)
    result = refresh.refresh_fuels(config, root=root)
    assert not result["required_sources_complete"]
    assert result["status"] == "failed"
    assert "AssertionError" in result["error"]


def test_post_cutoff_source_timestamp_is_rejected_even_with_valid_sha(seed, monkeypatch):
    root, source, config = seed
    def changed(path):
        def mutate(frame):
            suffix = pd.to_datetime(frame.value_time_utc, utc=True) >= pd.Timestamp("2026-03-29", tz="Europe/Paris")
            frame.loc[suffix, "ttf_source_value_time_utc"] = frame.loc[suffix, "snapshot_time_utc"] + pd.Timedelta(hours=1)
        rewrite(path, mutate)
    execute_fake(monkeypatch, after=changed)
    result = refresh.refresh_fuels(config, root=root)
    assert not result["required_sources_complete"]
    assert "posterieur au cutoff" in result["error"]


def test_command_failure_preserves_seed_and_returns_nonready_private_audit(seed, monkeypatch):
    root, source, config = seed
    before = source.read_bytes()
    monkeypatch.setattr(refresh.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    result = refresh.refresh_fuels(config, root=root)
    assert source.read_bytes() == before
    assert result["status"] == "failed" and not result["required_sources_complete"]
    assert Path(result["audit_path"]).is_file()


def test_concurrent_original_audit_change_invalidates_refresh(seed, monkeypatch):
    root, source, config = seed
    sidecar = Path(str(source) + ".audit.json")
    def changed(_):
        audit = json.loads(sidecar.read_text())
        audit["concurrent_test_revision"] = True
        sidecar.write_text(json.dumps(audit), encoding="utf-8")
    execute_fake(monkeypatch, after=changed)
    result = refresh.refresh_fuels(config, root=root)
    assert not result["original_sources_unchanged"]
    assert not result["required_sources_complete"]
    assert "concurrently" in result["error"]


def test_tampered_seed_fails_before_any_write(seed):
    root, source, config = seed
    sidecar = Path(str(source) + ".audit.json")
    audit = json.loads(sidecar.read_text())
    audit["sha256"] = "0" * 64
    sidecar.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ScarcityDataError, match="Checksum"):
        refresh.refresh_fuels(config, root=root)
    assert not (root / "runs").exists()


def test_distinct_fuel_seed_overrides_are_not_silently_merged(seed):
    root, source, config = seed
    config["data"]["source_overrides"]["eua"] = "data/pit/another/market_fuel_features.parquet"
    with pytest.raises(ScarcityDataError, match="one jointly audited"):
        refresh.refresh_fuels(config, root=root)
    assert not (root / "runs").exists()


def test_seed_outside_pit_namespace_is_rejected(seed):
    root, source, config = seed
    config["data"]["source_overrides"] = {"ttf": "runs/live/secrets.parquet", "eua": "runs/live/secrets.parquet"}
    with pytest.raises(ScarcityDataError, match="inside data/pit"):
        refresh.refresh_fuels(config, root=root)


def test_config_can_impose_a_stricter_suffix_limit(seed, monkeypatch):
    root, source, config = seed
    config["refresh"] = {"max_suffix_days": 1}
    monkeypatch.setattr(refresh.subprocess, "run", lambda *a, **k: pytest.fail("Suffix exceeds configured limit"))
    with pytest.raises(ScarcityDataError, match="limited to 1"):
        refresh.refresh_fuels(config, root=root)
    assert not (root / "runs").exists()


def test_config_retry_timeout_are_honoured_but_fuel_workers_stay_one(seed, monkeypatch):
    root, source, config = seed
    config["refresh"] = {"workers": 2, "max_suffix_days": 3, "retries": 1, "request_timeout_seconds": 20}
    calls = execute_fake(monkeypatch)
    result = refresh.refresh_fuels(config, root=root)
    command = calls[0][0]
    assert result["required_sources_complete"]
    assert command[command.index("--retries") + 1] == "1"
    assert command[command.index("--request-timeout-seconds") + 1] == "20"
    assert command[command.index("--series-workers") + 1] == "1"
    assert command[command.index("--day-workers") + 1] == "1"
    assert result["maximum_attempts_per_query"] == 1
    assert result["request_timeout_seconds"] == 20
    assert result["maximum_suffix_days"] == 3


@pytest.mark.parametrize("settings", [
    {"typo_retries": 2}, {"max_suffix_days": 31}, {"max_suffix_days": 0},
    {"max_suffix_days": True}, {"max_suffix_days": "30"}, {"retries": 0},
    {"retries": 4}, {"request_timeout_seconds": 0}, {"request_timeout_seconds": 121},
    {"request_timeout_seconds": 45.0}, {"workers": 3},
])
def test_refresh_typos_and_invalid_parameter_types_fail_before_writes(seed, settings):
    root, source, config = seed
    config["refresh"] = settings
    with pytest.raises(ScarcityDataError):
        refresh.refresh_fuels(config, root=root)
    assert not (root / "runs").exists()
