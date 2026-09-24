from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nyx_scarcity import data, refresh


def _seeds(root: Path):
    index = pd.date_range("2026-09-12T22:00Z", periods=24, freq="h")
    original = {}
    for key in refresh.REFRESH_KEYS:
        spec = data.source_registry()[key]
        path = root / spec["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({"value_time_utc": index,
            "snapshot_time_utc": data._origin(index, "Europe/Paris"),
            "revision_time_utc": data._origin(index, "Europe/Paris"), "value": np.arange(24) / 10 + 10})
        frame.to_parquet(path, index=False)
        audit = {"series": spec["series"], "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "cutoff_time": "08:00", "cutoff_timezone": "Europe/Paris", "unit": "GW",
            "provider_revision_timestamp_available": False, "revision_time_semantics": "query_asof_cutoff"}
        if "required_materialized_scale" in spec:
            audit["value_scale"] = spec["required_materialized_scale"]
        sidecar = Path(str(path) + ".audit.json")
        sidecar.write_text(json.dumps(audit), encoding="utf-8")
        original[path] = path.read_bytes()
        original[sidecar] = sidecar.read_bytes()
    return original


def _fake_run(calls, *, fail=None, partial=None):
    def run(command, **kwargs):
        calls.append((command, kwargs))
        path = Path(command[command.index("--output") + 1])
        assert "/source_refresh/" in path.as_posix()
        assert kwargs["shell"] is False
        assert command[command.index("--workers") + 1] == "1"
        assert command[command.index("--cutoff-time") + 1] == "08:00"
        alias = command[command.index("--alias") + 1]
        if fail and fail in alias:
            return SimpleNamespace(returncode=3)
        start = pd.Timestamp(command[command.index("--start-day") + 1])
        stop = pd.Timestamp(command[command.index("--end-day") + 1]) + pd.Timedelta(days=1)
        index = pd.date_range(start.tz_localize("Europe/Paris"), stop.tz_localize("Europe/Paris"),
                              freq="h", inclusive="left").tz_convert("UTC")
        if partial and partial in alias:
            index = index[:-1]
        frame = pd.DataFrame({"value_time_utc": index, "snapshot_time_utc": data._origin(index, "Europe/Paris"),
                              "revision_time_utc": data._origin(index, "Europe/Paris"), "value": 12.})
        pd.concat([pd.read_parquet(path), frame], ignore_index=True).to_parquet(path, index=False)
        sidecar = Path(str(path) + ".audit.json")
        audit = json.loads(sidecar.read_text())
        audit["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        sidecar.write_text(json.dumps(audit), encoding="utf-8")
        return SimpleNamespace(returncode=0)
    return run


def test_refresh_isolated_copies_exact_commands_and_originals_untouched(tmp_path, monkeypatch):
    originals = _seeds(tmp_path)
    config = {"delivery_day": "2026-09-15", "zones": ["DE"], "data": {}}
    previous = deepcopy(config)
    calls = []
    monkeypatch.setattr(refresh.subprocess, "run", _fake_run(calls))
    result = refresh.refresh_sources(config, root=tmp_path)
    assert result["status"] == "complete"
    assert result["required_sources_complete"]
    assert len(calls) == 17
    assert config == previous
    assert set(result["config"]["data"]["source_overrides"]) == set(refresh.REFRESH_KEYS)
    assert all(path.read_bytes() == raw for path, raw in originals.items())
    for command, kwargs in calls:
        assert command[command.index("--start-day") + 1] == "2026-09-14"
        assert command[command.index("--end-day") + 1] == "2026-09-15"
        assert command[command.index("--retries") + 1] == "2"
        assert "--merge-existing" in command
        if any("available" in str(x) for x in command):
            assert "--daily-broadcast" in command
        if "nl_wind_generation_fcst" in command:
            assert command[command.index("--value-scale") + 1] == "0.001"
        if "nl_solar_generation_fcst" in command:
            assert command[command.index("--incomplete-dst-policy") + 1] == "duplicate_zero_only"
    assert all(r["seed_history_semantically_unchanged"] for r in result["sources"].values())
    audit = json.loads(Path(result["audit_path"]).read_text())
    assert audit["production_pit_evidence"] is False
    assert json.loads(Path(result["saved_config"]).read_text()) == result["config"]


def test_refresh_resume_is_a_new_snapshot_without_requery_or_old_mutation(tmp_path, monkeypatch):
    _seeds(tmp_path)
    calls = []
    monkeypatch.setattr(refresh.subprocess, "run", _fake_run(calls))
    first = refresh.refresh_sources({"delivery_day": "2026-09-14"}, root=tmp_path)
    frozen = {p: p.read_bytes() for p in Path(first["source_dir"]).glob("*") if p.is_file()}
    calls.clear()
    second = refresh.refresh_sources(first["config"], root=tmp_path)
    assert second["status"] == "complete"
    assert first["source_dir"] != second["source_dir"]
    assert not calls
    assert all(p.read_bytes() == raw for p, raw in frozen.items())


def test_failed_source_not_claimed_ready_and_preserves_seed(tmp_path, monkeypatch):
    originals = _seeds(tmp_path)
    calls = []
    monkeypatch.setattr(refresh.subprocess, "run", _fake_run(calls, fail="de_gas"))
    result = refresh.refresh_sources({"delivery_day": "2026-09-14"}, root=tmp_path)
    assert result["status"] == "partial"
    assert not result["required_sources_complete"]
    record = result["sources"]["de_gas_available"]
    assert record["status"] == "failed"
    assert record["returncode"] == 3
    original = tmp_path / data.source_registry()["de_gas_available"]["path"]
    assert Path(record["source_path"]).read_bytes() == originals[original]
    assert all(p.read_bytes() == raw for p, raw in originals.items())


def test_missing_suffix_hour_stays_partial(tmp_path, monkeypatch):
    _seeds(tmp_path)
    calls = []
    monkeypatch.setattr(refresh.subprocess, "run", _fake_run(calls, partial="be_solar"))
    result = refresh.refresh_sources({"delivery_day": "2026-09-14"}, root=tmp_path)
    assert result["status"] == "partial"
    assert not result["required_sources_complete"]
    assert result["sources"]["be_solar_generation"]["missing_suffix_hours"] == 1


def test_full_backfill_rejected_before_any_write(tmp_path):
    _seeds(tmp_path)
    with pytest.raises(data.ScarcityDataError, match="suffix"):
        refresh.refresh_sources({"delivery_day": "2027-01-01"}, root=tmp_path)
    assert not (tmp_path / "runs").exists()


def test_bad_seed_rejected_before_any_write(tmp_path):
    _seeds(tmp_path)
    path = tmp_path / data.source_registry()["de_gas_available"]["path"]
    audit_path = Path(str(path) + ".audit.json")
    audit = json.loads(audit_path.read_text())
    audit["sha256"] = "bad"
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(data.ScarcityDataError, match="checksum"):
        refresh.refresh_sources({"delivery_day": "2026-09-14"}, root=tmp_path)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("options", [{"workers": 3}, {"workers": True}, {"series": "anything"}, {"max_suffix_days": 61}, {"retries": 4}])
def test_bounded_refresh_options(tmp_path, options):
    with pytest.raises(data.ScarcityDataError):
        refresh.refresh_sources({"delivery_day": "2026-09-14", "refresh": options}, root=tmp_path)
    assert not (tmp_path / "runs").exists()


def test_unknown_refresh_source_rejected_before_io(tmp_path):
    with pytest.raises(data.ScarcityDataError, match="registry"):
        refresh.refresh_sources({"delivery_day": "2026-09-14", "data": {"source_overrides": {"target": "data/pit/x"}}}, root=tmp_path)


def test_data_reader_allows_own_refresh_but_not_another_experiment(tmp_path):
    allowed = "runs/experiments/nyx_scarcity_v1/source_refresh/id/de.parquet"
    assert data._path(tmp_path, allowed) == (tmp_path / allowed).resolve()
    with pytest.raises(data.ScarcityDataError):
        data._path(tmp_path, "runs/experiments/nuclear_cwe_v1/de.parquet")


def test_timeout_does_not_mark_snapshot_ready(tmp_path, monkeypatch):
    _seeds(tmp_path)
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 90)
    monkeypatch.setattr(refresh.subprocess, "run", timeout)
    result = refresh.refresh_sources({"delivery_day": "2026-09-14"}, root=tmp_path)
    assert result["status"] == "failed"
    assert not result["required_sources_complete"]


@pytest.mark.parametrize("message,category", [
    ("ProxyError: Unable to connect to proxy 127.0.0.1:9", "proxy"),
    ("SSLError: CERTIFICATE_VERIFY_FAILED", "tls"),
    ("HTTPError: 401 Client Error: Unauthorized", "authentication"),
    ("HTTPError: 403 Client Error: Forbidden", "authentication"),
    ("407 Proxy Authentication Required", "proxy"),
    ("NameResolutionError: failed to resolve host", "network"),
    ("ConnectTimeout: timed out", "network"),
    ("Saturn indisponible ou vide pour power.fr.generation.solar.hourly.gw.fcst", None),
    ("ValueError: daily-broadcast exige exactement une valeur finie; obtenu=0", None),
    ("HTTPError: 404 Client Error: series not found", None),
])
def test_transport_classification_does_not_confuse_missing_series(tmp_path, message, category):
    log = tmp_path / "source.log"
    log.write_text(message, encoding="utf-8")
    assert refresh._transport_failure(log) == category


def test_proxy_failure_skips_remaining_queued_sources(tmp_path, monkeypatch):
    originals = _seeds(tmp_path)
    calls = []
    def proxy(command, **kwargs):
        calls.append(command)
        kwargs["stdout"].write("ProxyError: Unable to connect to proxy 127.0.0.1:9\n")
        kwargs["stdout"].flush()
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(refresh.subprocess, "run", proxy)
    result = refresh.refresh_sources({"delivery_day": "2026-09-14"}, root=tmp_path)
    assert 1 <= len(calls) <= 2
    assert result["status"] == "failed"
    assert result["required_sources_complete"] is False
    assert result["global_transport_failure"]["category"] == "proxy"
    assert result["queries_skipped_after_transport_failure"] >= len(refresh.REFRESH_KEYS) - 2
    assert len(result["sources"]) == len(refresh.REFRESH_KEYS)
    assert all(p.read_bytes() == raw for p, raw in originals.items())


def test_empty_curve_failure_does_not_skip_other_sources(tmp_path, monkeypatch):
    _seeds(tmp_path)
    calls = []
    successful = _fake_run(calls)
    def empty(command, **kwargs):
        if command[command.index("--alias") + 1] == "fr_wind_generation_fcst":
            calls.append((command, kwargs))
            kwargs["stdout"].write("Saturn indisponible ou vide; aucune valeur pour cette serie\n")
            return SimpleNamespace(returncode=1)
        return successful(command, **kwargs)
    monkeypatch.setattr(refresh.subprocess, "run", empty)
    result = refresh.refresh_sources({"delivery_day": "2026-09-14"}, root=tmp_path)
    assert len(calls) == len(refresh.REFRESH_KEYS)
    assert result["status"] == "partial"
    assert "global_transport_failure" not in result
    assert not any(r["status"] == "skipped_transport_failure" for r in result["sources"].values())
