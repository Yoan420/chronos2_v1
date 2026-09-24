from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.nuclear_run_archive import (
    NuclearRunArchiveError, load_nuclear_result_bundle, save_nuclear_result_bundle,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, delivery="2026-09-09"):
    work = tmp_path / "runs" / "experiments" / delivery / "fr"
    (work / "snapshot").mkdir(parents=True)
    source = work / "snapshot" / "target.csv"
    source.write_text("immutable frozen target", encoding="utf-8")
    resolved = work / "resolved_config.yaml"
    resolved.write_text("schema_version: 1\n", encoding="utf-8")
    (work / "input_snapshot.json").write_text(json.dumps({
        "identity": {"zone": "FR", "delivery_day": delivery},
        "files": [{"snapshot": str(source), "source": str(tmp_path / "mutable_target.csv"), "sha256": _sha(source)}],
        "resolved_config_sha256": _sha(resolved),
    }), encoding="utf-8")
    day = pd.Timestamp(delivery)
    grid = lambda start, end: pd.date_range(start.tz_localize("Europe/Paris"),
        end.tz_localize("Europe/Paris"), freq="h", inclusive="left").tz_convert("UTC").rename("delivery_start_utc")
    history = grid(day - pd.Timedelta(days=730), day)
    future = grid(day, day + pd.Timedelta(days=1))
    evaluation = grid(day - pd.Timedelta(days=365), day)
    raw = pd.DataFrame({"q10": np.float32(40), "q50": np.float32(50), "q90": np.float32(60),
                        "actual": np.float32(49), "forecast_origin_utc": history[0] - pd.Timedelta(days=1)}, index=history)
    residual = pd.DataFrame({"actual": np.float32(49)}, index=history)
    forecast = pd.DataFrame(index=future)
    for q, value in (("q10", 40.), ("q50", 50.), ("q90", 60.)):
        for output in (residual, forecast):
            output[f"chronos2__{q}"] = np.float32(value)
            output[f"residual_corrected__{q}"] = np.float32(value + 1)
    khistory, kforecast = residual.loc[evaluation].copy(), forecast.copy()
    for q in ("q10", "q50", "q90"):
        for output in (khistory, kforecast):
            output[f"residual_kalman__{q}"] = output[f"residual_corrected__{q}"] + np.float32(.5)
    audits = pd.DataFrame({"delivery_day": pd.date_range(day - pd.Timedelta(days=730), day).strftime("%Y-%m-%d"),
                           "residual_feature_columns": [["nuclear", "load"] for _ in range(731)]})
    result = SimpleNamespace(raw_history=raw, residual_statistics=residual.reset_index(),
        source_forecast=forecast.reset_index(),
        covariates=pd.DataFrame({"nuclear": 40.}, index=history.append(future)),
        residual_daily_audit=audits,
        kalman_view=SimpleNamespace(backtest=khistory.reset_index(), forecast=kforecast.reset_index(),
            replay=SimpleNamespace(audit={"training_days": 365, "rolling_refit_cache": {"history_hits": 0},
                                          "missing_value": float("nan")})),
        audit={"engine": "nuclear_forecast_v1", "zone": "FR", "delivery_day": delivery,
               "source_hashes": {"engine_file_sha256": "abc"}, "raw_future_cache_hit": False})
    return work, result


@pytest.mark.parametrize("delivery,hours", [("2026-09-09", 24), ("2026-03-29", 23), ("2026-10-25", 25)])
def test_roundtrip_preserves_frozen_prices_and_physical_dst_shape(tmp_path, delivery, hours):
    work, result = _fixture(tmp_path, delivery)
    path = save_nuclear_result_bundle(result, workdir=work)
    assert path == work / "report_only" / "frozen_result"
    restored = load_nuclear_result_bundle(workdir=work)
    for name in ("raw_history", "residual_statistics", "source_forecast", "covariates"):
        pd.testing.assert_frame_equal(getattr(restored, name), getattr(result, name), check_freq=False)
    pd.testing.assert_frame_equal(restored.kalman_view.forecast, result.kalman_view.forecast)
    assert len(restored.source_forecast) == hours
    assert restored.kalman_view.replay.audit["training_days"] == 365
    assert restored.kalman_view.replay.audit["missing_value"] is None
    assert restored.residual_daily_audit.residual_feature_columns.iloc[0].tolist() == ["nuclear", "load"]


def test_identical_repeated_save_never_rewrites_bundle(tmp_path):
    work, result = _fixture(tmp_path)
    path = save_nuclear_result_bundle(result, workdir=work)
    before = {p.name: (p.stat().st_mtime_ns, _sha(p)) for p in path.iterdir()}
    result.audit["raw_future_cache_hit"] = True
    result.audit["daily_chronos_cache"] = {"history_hits": 730, "future_hits": 1}
    result.audit["daily_residual_cache"] = {"hits": 731, "misses": 0, "writes": 0}
    result.kalman_view.replay.audit["rolling_refit_cache"] = {"history_hits": 365, "future_hits": 1}
    assert save_nuclear_result_bundle(result, workdir=work) == path
    assert {p.name: (p.stat().st_mtime_ns, _sha(p)) for p in path.iterdir()} == before


@pytest.mark.parametrize("permanent", [False, True])
def test_frozen_bundle_publication_retries_same_parquets_without_recalculation(tmp_path, monkeypatch, permanent):
    from chronos2_hourly import atomic_directory as atomic
    from test_atomic_directory import windows_error

    work, result = _fixture(tmp_path)
    source_before = _sha(work / "input_snapshot.json")
    rename = atomic._rename_no_replace
    attempts, hashes = [], []

    def locked(source, target):
        attempts.append((source, target))
        hashes.append({path.name: _sha(path) for path in source.iterdir()})
        assert len(hashes[-1]) == 9
        if permanent or len(attempts) < 3:
            raise windows_error(32)
        rename(source, target)

    monkeypatch.setattr(atomic, "_rename_no_replace", locked)
    monkeypatch.setattr(atomic.time, "sleep", lambda _: None)
    if permanent:
        with pytest.raises(atomic.AtomicDirectoryPublishError):
            save_nuclear_result_bundle(result, workdir=work)
        assert len(attempts) == 7
        stage, final = attempts[0]
        assert stage.is_dir() and not final.exists()
        monkeypatch.setattr(atomic, "_rename_no_replace", rename)
        atomic.publish_directory_no_replace(stage, final)
    else:
        save_nuclear_result_bundle(result, workdir=work)
        assert len(attempts) == 3
    assert len({source for source, _ in attempts}) == 1
    assert all(item == hashes[0] for item in hashes)
    assert _sha(work / "input_snapshot.json") == source_before
    restored = load_nuclear_result_bundle(workdir=work)
    pd.testing.assert_frame_equal(restored.kalman_view.forecast, result.kalman_view.forecast)


def test_divergent_result_is_rejected_without_rewriting_archive(tmp_path):
    work, result = _fixture(tmp_path)
    path = save_nuclear_result_bundle(result, workdir=work)
    before = _sha(path / "manifest.json")
    result.raw_history["q50"] += .01
    with pytest.raises(NuclearRunArchiveError, match="overwrite.*divergent"):
        save_nuclear_result_bundle(result, workdir=work)
    assert _sha(path / "manifest.json") == before


@pytest.mark.parametrize("filename", ["raw_history.parquet", "audits.json", "manifest.json"])
def test_corrupted_payload_or_manifest_is_rejected(tmp_path, filename):
    work, result = _fixture(tmp_path)
    path = save_nuclear_result_bundle(result, workdir=work)
    (path / filename).write_bytes(b"corrupt")
    with pytest.raises(NuclearRunArchiveError):
        load_nuclear_result_bundle(workdir=work)


def test_missing_payload_cannot_be_recreated_silently(tmp_path):
    work, result = _fixture(tmp_path)
    path = save_nuclear_result_bundle(result, workdir=work)
    (path / "kalman_forecast.parquet").unlink()
    with pytest.raises(NuclearRunArchiveError, match="inventory"):
        save_nuclear_result_bundle(result, workdir=work)


@pytest.mark.parametrize("filename", ["input_snapshot.json", "resolved_config.yaml", "snapshot/target.csv"])
def test_frozen_source_changes_are_rejected(tmp_path, filename):
    work, result = _fixture(tmp_path)
    save_nuclear_result_bundle(result, workdir=work)
    target = work / filename
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(NuclearRunArchiveError, match="checksum|snapshot/configuration"):
        load_nuclear_result_bundle(workdir=work)


def test_mutable_operational_target_is_not_part_of_frozen_contract(tmp_path):
    work, result = _fixture(tmp_path)
    save_nuclear_result_bundle(result, workdir=work)
    (tmp_path / "mutable_target.csv").write_text("new observed labels", encoding="utf-8")
    assert load_nuclear_result_bundle(workdir=work).audit["zone"] == "FR"


def test_missing_bundle_requests_run_not_implicit_training(tmp_path):
    work, _ = _fixture(tmp_path)
    with pytest.raises(NuclearRunArchiveError, match="run.*once before Report"):
        load_nuclear_result_bundle(workdir=work)


@pytest.mark.parametrize("fault", ["future_hole", "future_observation", "quantile_cross", "zone", "kalman_upstream", "daily_audit"])
def test_invalid_result_cannot_be_sealed(tmp_path, fault):
    work, result = _fixture(tmp_path)
    if fault == "future_hole":
        result.source_forecast = result.source_forecast.iloc[:-1]
    elif fault == "future_observation":
        result.source_forecast["actual"] = 50.
    elif fault == "quantile_cross":
        result.source_forecast["chronos2__q10"] = 99.
    elif fault == "zone":
        result.audit["zone"] = "DE"
    elif fault == "kalman_upstream":
        result.kalman_view.forecast["residual_corrected__q50"] += .01
    else:
        result.residual_daily_audit = result.residual_daily_audit.iloc[1:]
    with pytest.raises(NuclearRunArchiveError):
        save_nuclear_result_bundle(result, workdir=work)
    assert not (work / "report_only" / "frozen_result").exists()


def test_missing_snapshot_file_and_path_traversal_are_rejected(tmp_path):
    work, result = _fixture(tmp_path)
    manifest = work / "input_snapshot.json"
    payload = json.loads(manifest.read_text())
    external = tmp_path / "other.csv"
    external.write_text("not a frozen source")
    payload["files"][0].update(snapshot=str(external), sha256=_sha(external))
    manifest.write_text(json.dumps(payload))
    with pytest.raises(NuclearRunArchiveError, match="unsafe frozen snapshot"):
        save_nuclear_result_bundle(result, workdir=work)


def test_sealed_live_archive_is_protected(tmp_path):
    work, result = _fixture(tmp_path)
    (work / "artifact_checksums.json").write_text("{}")
    with pytest.raises(NuclearRunArchiveError, match="sealed live archive"):
        save_nuclear_result_bundle(result, workdir=work)
