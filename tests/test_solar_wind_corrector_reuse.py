"""Reuse is an inverse of an interior clipped shift, never an inferred fit."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import solar_wind_corrector_interaction as original
from chronos2_hourly import solar_wind_corrector_parallel as parallel
from chronos2_hourly import solar_wind_corrector_reuse as subject


def _origins(index):
    return pd.Series([
        (pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=8))
        .tz_localize("Europe/Berlin").tz_convert("UTC")
        for day in index.tz_convert("Europe/Berlin").date
    ], index=index)


def _fixture(monkeypatch, tmp_path, *, day="2026-01-01", correction=None, training_days=30):
    monkeypatch.setattr(original, "OUTPUT", tmp_path / "serial")
    monkeypatch.setattr(parallel, "OUTPUT", tmp_path / "parallel")
    monkeypatch.setattr(subject, "OUTPUT", tmp_path / "reuse")
    start = pd.Timestamp(day, tz="Europe/Berlin")
    index = pd.date_range(start, start + pd.DateOffset(days=1), freq="h", inclusive="left").tz_convert("UTC")
    index = index.rename("delivery_start_utc")
    training = pd.date_range(start - pd.DateOffset(days=training_days), start, freq="h", inclusive="left").tz_convert("UTC")
    training = training.rename(index.name)
    def raw(grid, actual=True):
        frame = pd.DataFrame({"q10": 17., "q50": 105., "q90": 193.}, index=grid)
        frame["forecast_origin_utc"] = _origins(grid)
        if actual:
            frame["actual"] = -9999.  # Reconstruction must not consume labels.
        return frame
    base = raw(index, day != original.DAY)
    if correction is None:
        correction = np.linspace(-31.125, 27.875, len(index))
    delta = np.broadcast_to(correction, (len(index),)).copy()
    corrected = base.copy(deep=True)
    for quantile in original.QUANTILES:
        corrected["chronos2__" + quantile] = base[quantile]
        corrected["residual_corrected__" + quantile] = base[quantile] + delta
    corrected["residual_correction"] = delta
    history = pd.concat([raw(training), base]) if day != original.DAY else raw(training)
    future = raw(index, False)
    provenance = {
        "delivery_day": day, "phase": "future" if day == original.DAY else "evaluation",
        "generation_source": "daily_prequential_refit" if len(training) >= 720 else "identity_chronos_cold_start",
        "training_rows": len(training), "minimum_training_rows": 720,
        "training_lookback_days": 365, "causality_violations": 0,
        "fit_start_day": str(training[0].tz_convert("Europe/Berlin").date()) if len(training) else None,
        "fit_end_day": str(training[-1].tz_convert("Europe/Berlin").date()) if len(training) else None,
        "forecast_hours": len(index), "residual_feature_columns": ["baseline_feature"] if len(training) >= 720 else [],
    }
    work = tmp_path / "baseline"
    manifest = work / "report_only/frozen_result/manifest.json"
    manifest.parent.mkdir(parents=True)
    sealed_files = {}
    for name in ("residual_statistics.parquet", "source_forecast.parquet", "residual_daily_audit.parquet"):
        path = manifest.parent / name
        path.write_bytes(("synthetic:" + name).encode())
        sealed_files[name] = original.sha(path)
    manifest.write_bytes(original.json_bytes({"files": sealed_files}))
    recipe = {"backend": "catboost", "correction_scale": 1., "max_abs_correction": 40.,
              "min_training_rows": 720, "thread_count": 2}
    identity = {"run_id": "serial-synthetic", "zone": "DE", "baseline_workdir": str(work),
        "baseline_manifest_sha256": original.sha(manifest), "residual_recipe": recipe,
        "feature": "de_low_wind_solar_stress", "pit_publication_evidence_verified": False,
        "kalman_config": {}, "kalman_covariate_config": {}}
    bundle = SimpleNamespace(residual_statistics=corrected, source_forecast=corrected.drop(columns="actual", errors="ignore"),
        residual_daily_audit=pd.DataFrame([provenance]), audit={"residual_recipe": recipe})
    prepared = SimpleNamespace(identity=identity, raw_history=history, raw_future=future, bundle=bundle,
        factory=lambda: SimpleNamespace(min_training_rows=720))
    return prepared, index, delta, manifest


@pytest.mark.parametrize("day", ["2026-01-01", "2025-10-26", "2026-03-29", "2026-09-22"])
def test_interior_inverse_is_analytically_correct_for_all_quantiles_and_dst(monkeypatch, tmp_path, day):
    prepared, index, expected, _ = _fixture(monkeypatch, tmp_path, day=day)
    before = deepcopy(prepared.bundle.residual_statistics)
    raw, audit, provenance = subject.reconstruct_baseline(prepared, day)
    # Independent analytic oracle, not the forward helper used by the engine.
    np.testing.assert_allclose(raw, expected, rtol=0, atol=1e-12)
    for quantile in original.QUANTILES:
        source = prepared.bundle.residual_statistics.loc[index]
        np.testing.assert_allclose(source["chronos2__" + quantile] + raw,
                                   source["residual_corrected__" + quantile], rtol=0, atol=1e-9)
    assert audit["generation_source"] == "reconstructed_sealed_baseline_interior"
    assert audit["source_generation_source"] == "daily_prequential_refit"
    assert provenance
    pd.testing.assert_frame_equal(prepared.bundle.residual_statistics, before)
    prepared.raw_history["actual"] = 1e20
    np.testing.assert_array_equal(subject.reconstruct_baseline(prepared, day)[0], raw)


@pytest.mark.parametrize("value", [-40., 40., -40. + 5e-7, 40. - 5e-7, -40.0001, 40.0001])
def test_one_noninvertible_hour_refuses_whole_day(monkeypatch, tmp_path, value):
    prepared, _, _, _ = _fixture(monkeypatch, tmp_path, correction=0.)
    frame = prepared.bundle.residual_statistics
    frame.loc[frame.index[7], "residual_correction"] = value
    for q in original.QUANTILES:
        frame.loc[frame.index[7], "residual_corrected__" + q] = frame.loc[frame.index[7], "chronos2__" + q] + value
    # Out-of-contract corrections are integrity failures, not ordinary clipping.
    if abs(value) > 40.:
        with pytest.raises(ValueError):
            subject.reconstruct_baseline(prepared, "2026-01-01")
    else:
        assert subject.reconstruct_baseline(prepared, "2026-01-01") is None


@pytest.mark.parametrize("kind", ["q10_shift", "q90_shift", "nan", "infinity", "crossed", "missing_quantile",
                                  "duplicate_hour", "missing_hour", "wrong_origin", "missing_audit", "duplicate_audit",
                                  "unknown_source", "future_training", "wrong_lookback", "wrong_rows", "wrong_hours",
                                  "wrong_scale", "wrong_cap", "wrong_backend"])
def test_reconstruction_rejects_invalid_science_grid_and_provenance(monkeypatch, tmp_path, kind):
    prepared, index, _, _ = _fixture(monkeypatch, tmp_path)
    frame, daily = prepared.bundle.residual_statistics, prepared.bundle.residual_daily_audit
    if kind in ("q10_shift", "q90_shift"):
        frame.loc[index[0], "residual_corrected__" + kind[:3]] += .01
    elif kind in ("nan", "infinity"):
        frame.loc[index[0], "residual_corrected__q50"] = np.nan if kind == "nan" else np.inf
    elif kind == "crossed":
        frame.loc[index[0], "residual_corrected__q10"] = 1e4
    elif kind == "missing_quantile":
        frame.drop(columns="residual_corrected__q90", inplace=True)
    elif kind == "duplicate_hour":
        prepared.bundle.residual_statistics = pd.concat([frame.iloc[:1], frame])
    elif kind == "missing_hour":
        prepared.bundle.residual_statistics = frame.iloc[1:]
    elif kind == "wrong_origin":
        frame.loc[index[0], "forecast_origin_utc"] = index[0]
    elif kind == "missing_audit":
        prepared.bundle.residual_daily_audit = daily.iloc[0:0]
    elif kind == "duplicate_audit":
        prepared.bundle.residual_daily_audit = pd.concat([daily, daily])
    elif kind == "unknown_source":
        daily.loc[0, "generation_source"] = "unknown_cached_model"
    elif kind == "future_training":
        daily.loc[0, "causality_violations"] = 1
    elif kind == "wrong_lookback":
        daily.loc[0, "training_lookback_days"] = 366
    elif kind == "wrong_rows":
        daily.loc[0, "training_rows"] += 24
    elif kind == "wrong_hours":
        daily.loc[0, "forecast_hours"] = 23
    elif kind == "wrong_scale":
        prepared.identity["residual_recipe"]["correction_scale"] = .5
    elif kind == "wrong_cap":
        prepared.identity["residual_recipe"]["max_abs_correction"] = 80.
    else:
        prepared.identity["residual_recipe"]["backend"] = "sklearn"
    with pytest.raises((ValueError, KeyError)):
        subject.reconstruct_baseline(prepared, "2026-01-01")


def test_explicit_cold_start_can_only_reuse_identity_correction(monkeypatch, tmp_path):
    prepared, _, _, _ = _fixture(monkeypatch, tmp_path, correction=0., training_days=3)
    raw, audit, _ = subject.reconstruct_baseline(prepared, "2026-01-01")
    np.testing.assert_array_equal(raw, np.zeros(24))
    assert audit["source_generation_source"] == "identity_chronos_cold_start"
    frame = prepared.bundle.residual_statistics
    for q in original.QUANTILES:
        frame["residual_corrected__" + q] += 2.
    frame["residual_correction"] += 2.
    with pytest.raises(ValueError):
        subject.reconstruct_baseline(prepared, "2026-01-01")


@pytest.mark.parametrize("margin", [0., 1e-9, -1., np.nan, np.inf, 1.])
def test_reconstruction_margin_cannot_relax_safety(monkeypatch, tmp_path, margin):
    prepared, _, _, _ = _fixture(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        subject.reconstruct_baseline(prepared, "2026-01-01", margin=margin)


@pytest.mark.parametrize("saturated", [False, True])
def test_worker_only_skips_baseline_fit_for_invertible_day_and_always_fits_interaction(monkeypatch, tmp_path, saturated):
    prepared, index, expected, _ = _fixture(monkeypatch, tmp_path, correction=40. if saturated else 7.125)
    monkeypatch.setattr(subject, "_PREPARED", prepared)
    monkeypatch.setattr(subject, "_CONTROL_RAW", {})
    calls = []
    def fake_fit(item, day, *, interaction):
        assert item is prepared
        calls.append(interaction)
        train = original.train_window(item.raw_history.index, day, "Europe/Berlin")
        assert len(train) == 720
        assert all(train.tz_convert("Europe/Berlin").date < pd.Timestamp(day).date())
        return np.full(len(index), 63. if interaction else 67.), {"generation_source": "synthetic_fit", "causality_violations": 0}
    monkeypatch.setattr(original, "_fit_raw", fake_fit)
    result = subject._compute_corrector({"day": "2026-01-01"})["payload"]
    assert calls == ([False, True] if saturated else [True])
    np.testing.assert_array_equal(result["raw"]["interaction"], np.full(24, 63.))
    np.testing.assert_array_equal(result["raw"]["original"], np.full(24, 67.) if saturated else expected)
    assert (result["baseline_reconstruction"] is None) is saturated
    base = subject._base_for_day(prepared, "2026-01-01")
    actual_80 = original.apply_variant(base, result["raw"]["original"], upper=80)
    # Using +40 as if it were raw would incorrectly lose 27 EUR/MWh here.
    np.testing.assert_array_equal(actual_80.q50, np.full(24, 172. if saturated else 112.125))


def test_worker_invalid_provenance_is_not_hidden_by_refit(monkeypatch, tmp_path):
    prepared, _, _, _ = _fixture(monkeypatch, tmp_path)
    prepared.bundle.residual_daily_audit.loc[0, "generation_source"] = "unknown"
    monkeypatch.setattr(subject, "_PREPARED", prepared)
    monkeypatch.setattr(subject, "_CONTROL_RAW", {})
    monkeypatch.setattr(original, "_fit_raw", lambda *args, **kwargs: pytest.fail("Integrity failure is not a refit fallback"))
    with pytest.raises(ValueError):
        subject._compute_corrector({"day": "2026-01-01"})


def _identity(prepared, monkeypatch):
    prior = parallel.make_identity(prepared, 4, 3.)
    monkeypatch.setattr(subject, "SOURCE_RUNS", {"DE": prior["run_id"]})
    return subject.make_identity(prepared, 4, 3.)


def _seal(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(subject.json_bytes({"payload": payload, "sha256": subject._digest(payload)}))


def _import_fixture(monkeypatch, tmp_path):
    prepared, index, delta, _ = _fixture(monkeypatch, tmp_path, correction=7.125)
    identity = _identity(prepared, monkeypatch)
    serial_dir = original.OUTPUT / "de" / prepared.identity["run_id"]
    serial_path = serial_dir / "corrector_daily/2026-01-01.json"
    serial_payload = {"identity": prepared.identity["run_id"], "day": "2026-01-01", "index_ns": index.asi8.tolist(),
        "raw": {"original": delta.tolist(), "interaction": [61.] * len(index)},
        "audit": {name: {"generation_source": "daily_prequential_refit", "causality_violations": 0} for name in ("original", "interaction")}}
    _seal(serial_path, serial_payload)
    (serial_dir / "experiment.json").write_bytes(subject.json_bytes(prepared.identity))
    donor_identity = identity["parallel_identity"]
    donor_dir = parallel.OUTPUT / "de" / donor_identity["run_id"]
    donor_path = donor_dir / "corrector_daily/2026-01-01.json"
    donor_payload, origin = parallel.import_original_checkpoint(prepared, donor_identity, serial_path, index)
    _seal(donor_path, donor_payload)
    (donor_dir / "experiment.json").write_bytes(subject.json_bytes(donor_identity))
    return prepared, identity, donor_path, donor_payload, index, serial_path


def test_import_parallel_preserves_values_and_nested_origin_without_source_writes(monkeypatch, tmp_path):
    prepared, identity, donor, payload, index, serial = _import_fixture(monkeypatch, tmp_path)
    paths = [donor, donor.parent.parent / "experiment.json", serial, serial.parent.parent / "experiment.json"]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    imported, provenance = subject.import_parallel_checkpoint(prepared, identity, donor, index)
    assert imported["identity"] == identity["run_id"]
    assert imported["raw"] == payload["raw"] and imported["audit"] == payload["audit"]
    assert provenance["previous_origin"] == payload["origin"]
    assert provenance["source_file_sha256"] == subject.sha(donor)
    assert provenance["raw_predictions_unchanged"] is True
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    assert not subject.OUTPUT.exists()


@pytest.mark.parametrize("kind", ["checksum", "identity", "day", "grid", "duplicate", "nan", "wrong_baseline", "manifest", "outside", "nested_checksum", "nested_source_changed", "nested_path"])
def test_import_refuses_changed_or_untrusted_donor(monkeypatch, tmp_path, kind):
    prepared, identity, donor, payload, index, serial = _import_fixture(monkeypatch, tmp_path)
    if kind == "identity":
        payload["identity"] = "wrong-run"
    elif kind == "day":
        payload["day"] = "2026-01-02"
    elif kind == "grid":
        payload["index_ns"][0] += 1
    elif kind == "duplicate":
        payload["index_ns"][1] = payload["index_ns"][0]
    elif kind == "nan":
        payload["raw"]["interaction"][0] = "nan"
    elif kind == "wrong_baseline":
        payload["raw"]["original"][0] += 1.
    elif kind == "manifest":
        (donor.parent.parent / "experiment.json").write_text("{}", encoding="utf-8")
    elif kind == "nested_checksum":
        payload["origin"]["original_file_sha256"] = "0" * 64
    elif kind == "nested_source_changed":
        serial.write_bytes(b"changed")
    elif kind == "nested_path":
        payload["origin"]["original_path"] = str(tmp_path / "unapproved.json")
    _seal(donor, payload)
    if kind == "checksum":
        record = json.loads(donor.read_text(encoding="utf-8"))
        record["payload"]["raw"]["interaction"][0] += 1.
        donor.write_text(json.dumps(record), encoding="utf-8")
    elif kind == "outside":
        outside = tmp_path / "other/2026-01-01.json"
        outside.parent.mkdir()
        outside.write_bytes(donor.read_bytes())
        donor = outside
    with pytest.raises(ValueError):
        subject.import_parallel_checkpoint(prepared, identity, donor, index)
    assert not subject.OUTPUT.exists()


def test_identity_pins_reconstruction_and_original_without_claiming_prospective_validation(monkeypatch, tmp_path):
    prepared, _, _, manifest = _fixture(monkeypatch, tmp_path)
    identity = _identity(prepared, monkeypatch)
    assert identity["original_identity"] == prepared.identity
    assert identity["original_identity"]["pit_publication_evidence_verified"] is False
    assert identity["reconstruction"]["source_bounds"] == [-40., 40.]
    assert identity["reconstruction"]["margin_eur_mwh"] == 1e-6
    assert identity["reconstruction"]["not_claimed_bitwise_raw_reproduction"] is True
    assert identity["promotion_eligible"] is False
    assert subject.make_identity(prepared, 2, 3.)["run_id"] != identity["run_id"]
    manifest.write_bytes(b"{}")
    with pytest.raises(ValueError, match="manifest"):
        subject.make_identity(prepared)


def test_validate_is_readonly_and_never_fits_or_starts_pool(monkeypatch, tmp_path):
    prepared, _, _, _ = _fixture(monkeypatch, tmp_path)
    identity = _identity(prepared, monkeypatch)
    monkeypatch.setattr(original, "prepare", lambda *args, **kwargs: prepared)
    def forbidden(*args, **kwargs):
        pytest.fail("validate is read-only")
    for name in ("_pool", "write_json", "write_frame", "exclusive_process_lock"):
        monkeypatch.setattr(subject, name, forbidden)
    monkeypatch.setattr(original, "_fit_raw", forbidden)
    assert subject.run(action="validate", zones=("DE",)) == {"DE": identity}
    assert not subject.OUTPUT.exists()


@pytest.mark.parametrize("namespace", ["serial", "parallel", "reuse"])
def test_all_three_batch_locks_prevent_duplicate_writer(monkeypatch, tmp_path, namespace):
    prepared, _, _, _ = _fixture(monkeypatch, tmp_path)
    _identity(prepared, monkeypatch)
    monkeypatch.setattr(original, "prepare", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(original, "corrector_controls", lambda *args, **kwargs: pytest.fail("Another batch owns a lock"))
    lock = {"serial": original.OUTPUT, "parallel": parallel.OUTPUT, "reuse": subject.OUTPUT}[namespace] / "batch.lock"
    with original.exclusive_process_lock(lock):
        before = lock.read_bytes()
        with pytest.raises(ValueError, match="Verrou"):
            subject.run(action="run", zones=("DE",))
        assert lock.read_bytes() == before


def _completion_fixture(monkeypatch, tmp_path):
    prepared, _, _, _ = _fixture(monkeypatch, tmp_path)
    identity = _identity(prepared, monkeypatch)
    directory = subject.OUTPUT / "de" / identity["run_id"]
    for name in subject.RESULT_FILES:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subject.json_bytes(identity) if name == "experiment.json" else b"sealed")
    receipt = {"identity": identity["run_id"], "annual_complete": True,
        "files": {name: subject.sha(directory / name) for name in subject.RESULT_FILES}}
    status = {"identity": identity["run_id"], "zone": "DE", "status": "COMPLETE", "annual_complete": True,
        "evaluation_days": 365, "evaluation_hours": 8760, "future_forecast_hours": 24}
    (directory / "completion.json").write_bytes(subject.json_bytes(receipt))
    (directory / "status.json").write_bytes(subject.json_bytes(status))
    return prepared, identity, directory, receipt, status


def test_exact_24_artifact_completion_resumes_without_any_fit(monkeypatch, tmp_path):
    prepared, identity, directory, _, _ = _completion_fixture(monkeypatch, tmp_path)
    assert len(subject.RESULT_FILES) == len(set(subject.RESULT_FILES)) == 24
    assert "reconstruction_audit.json" in subject.RESULT_FILES
    monkeypatch.setattr(original, "prepare", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(original, "corrector_controls", lambda *args, **kwargs: pytest.fail("Complete run must not refit"))
    assert subject._verify_completion(directory, identity)
    assert subject.run(action="run", zones=("DE",)) == {"DE": str(directory / "report.html")}


@pytest.mark.parametrize("kind", ["missing_reconstruction", "empty", "changed_reconstruction", "partial_year", "no_forecast", "identity"])
def test_partial_or_tampered_completion_never_counts_as_done(monkeypatch, tmp_path, kind):
    _, identity, directory, receipt, status = _completion_fixture(monkeypatch, tmp_path)
    if kind == "missing_reconstruction":
        del receipt["files"]["reconstruction_audit.json"]
    elif kind == "empty":
        receipt["files"] = {}
    elif kind == "changed_reconstruction":
        (directory / "reconstruction_audit.json").write_bytes(b"changed")
    elif kind == "partial_year":
        status["evaluation_days"] = 364
    elif kind == "no_forecast":
        status["future_forecast_hours"] = 0
    else:
        status["identity"] = "other"
    (directory / "completion.json").write_bytes(subject.json_bytes(receipt))
    (directory / "status.json").write_bytes(subject.json_bytes(status))
    with pytest.raises(ValueError):
        subject._verify_completion(directory, identity)


def test_targeted_stop_pauses_without_auto_resume_or_fit(monkeypatch, tmp_path):
    prepared, _, _, _ = _fixture(monkeypatch, tmp_path)
    identity = _identity(prepared, monkeypatch)
    monkeypatch.setattr(original, "prepare", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(original, "corrector_controls", lambda *args, **kwargs: pytest.fail("Stop requested"))
    subject.OUTPUT.mkdir()
    (subject.OUTPUT / "request_stop.json").write_bytes(subject.json_bytes({"identity": identity["run_id"]}))
    result = subject.run(action="run", zones=("DE",))
    assert result["status"] == "PAUSED" and result["reason"] == "user_stop"
    status = json.loads((subject.OUTPUT / "de" / identity["run_id"] / "status.json").read_text(encoding="utf-8"))
    assert status["automatic_resume"] is False


def test_stop_drains_inflight_and_keeps_completed_callbacks():
    committed, started = [], []
    def worker(task):
        started.append(task)
        return task
    with ThreadPoolExecutor(max_workers=2) as executor:
        result = subject.bounded_tasks(list(range(20)), worker, executor,
            lambda task, value: committed.append(value), lambda: bool(committed), lambda: 20., max_workers=2)
    assert result["stopped"] and result["reason"] == "user_stop"
    assert sorted(committed) == sorted(started)
    assert 1 <= len(committed) <= 2


def _annual_prepared(monkeypatch, tmp_path):
    prepared, _, _, _ = _fixture(monkeypatch, tmp_path)
    timezone = "Europe/Berlin"
    start, stop = pd.Timestamp("2025-09-22", tz=timezone), pd.Timestamp("2026-09-22", tz=timezone)
    history_index = pd.date_range(start, stop, freq="h", inclusive="left").tz_convert("UTC").rename("delivery_start_utc")
    future_index = pd.date_range(stop, periods=24, freq="h").tz_convert("UTC").rename(history_index.name)
    def frame(index, actual):
        result = pd.DataFrame({"q10": 80., "q50": 100., "q90": 130.}, index=index)
        result["forecast_origin_utc"] = _origins(index)
        if actual:
            result["actual"] = 103.
        return result
    def upstream(raw):
        result = raw.copy(deep=True)
        for q in original.QUANTILES:
            result["chronos2__" + q] = result[q]
            result["residual_corrected__" + q] = result[q]
        result["residual_correction"] = 0.
        return result
    def overlay(source):
        result = source.copy(deep=True)
        for q in original.QUANTILES:
            result["residual_kalman__" + q] = result["residual_corrected__" + q] + 2.
        return result
    prepared.raw_history, prepared.raw_future = frame(history_index, True), frame(future_index, False)
    history, future = upstream(prepared.raw_history), upstream(prepared.raw_future)
    prepared.bundle.residual_statistics, prepared.bundle.source_forecast = history, future
    prepared.bundle.kalman_view = SimpleNamespace(backtest=overlay(history), forecast=overlay(future))
    all_index = history_index.append(future_index)
    prepared.interaction = pd.DataFrame({prepared.identity["feature"]: .5}, index=all_index)
    prepared.bundle.covariates = pd.DataFrame({"de_residual_load_fcst": 12.}, index=all_index)
    prepared.config, prepared.covconfig, prepared.feature_audit = {}, {}, {}
    days = list(pd.Index(all_index.tz_convert(timezone).date).unique())
    records = []
    for day in days:
        train = original.train_window(history_index, day, timezone)
        records.append({"delivery_day": str(day), "generation_source": "daily_prequential_refit" if len(train) >= 720 else "identity_chronos_cold_start",
            "training_rows": len(train), "minimum_training_rows": 720, "training_lookback_days": 365,
            "phase": "future" if str(day) == original.DAY else "evaluation",
            "fit_start_day": str(train[0].tz_convert("Europe/Berlin").date()) if len(train) else None,
            "fit_end_day": str(train[-1].tz_convert("Europe/Berlin").date()) if len(train) else None,
            "forecast_hours": int((all_index.tz_convert(timezone).date == day).sum()),
            "causality_violations": 0, "residual_feature_columns": []})
    prepared.bundle.residual_daily_audit = pd.DataFrame(records)
    return prepared, overlay


def test_reconstructed_cap80_keeps_sealed_quantiles_bitwise_despite_inverse_roundoff(monkeypatch, tmp_path):
    prepared, _ = _annual_prepared(monkeypatch, tmp_path)
    history = prepared.bundle.residual_statistics
    history = history.loc[history.index.tz_convert("Europe/Berlin").date == pd.Timestamp("2026-01-01").date()].copy()
    prepared.bundle.residual_statistics = history
    future = prepared.bundle.source_forecast
    for source, raw in ((history, prepared.raw_history), (future, prepared.raw_future)):
        for q, base, corrected in zip(original.QUANTILES, (40., 40.1, 40.2), (.1, .2, .3)):
            source["chronos2__" + q] = base
            source["residual_corrected__" + q] = corrected
            raw.loc[source.index, q] = base
            source[q] = corrected
        source["residual_correction"] = -39.9
    inverse, _, _ = subject.reconstruct_baseline(prepared, "2026-01-01")
    # Demonstrate why inverse-then-add is NOT an adequate equality test.
    roundtrip = original.apply_variant(subject._base_for_day(prepared, "2026-01-01"), inverse, upper=80)
    assert not np.array_equal(roundtrip.q10, history.residual_corrected__q10)
    identity = _identity(prepared, monkeypatch)
    directory = subject.OUTPUT / "de" / identity["run_id"]
    directory.mkdir(parents=True)
    monkeypatch.setattr(original, "_fit_raw", lambda p, day, *, interaction: (np.ones(len(subject._base_for_day(p, day))), {"generation_source": "synthetic"}))
    def pool(item, identity, tasks, worker, on_result, progress, **options):
        monkeypatch.setattr(subject, "_PREPARED", item)
        monkeypatch.setattr(subject, "_CONTROL_RAW", {})
        for task in tasks:
            on_result(task, worker(task))
    monkeypatch.setattr(subject, "_run_pool_tasks", pool)
    frames, _, _ = subject._corrector_stage(prepared, identity, directory, lambda *args, **kwargs: None, {})
    for actual, sealed in zip(frames["cap_80"], (history, future)):
        for column in [*("residual_corrected__" + q for q in original.QUANTILES), "residual_correction"]:
            np.testing.assert_array_equal(actual[column], sealed[column])


@pytest.mark.parametrize("field", ["raw", "audit", "evidence", "missing", "origin"])
def test_resume_revalidates_reconstruction_evidence(monkeypatch, tmp_path, field):
    prepared, index, _, _ = _fixture(monkeypatch, tmp_path)
    identity = _identity(prepared, monkeypatch)
    raw, audit, evidence = subject.reconstruct_baseline(prepared, "2026-01-01")
    payload = {"identity": identity["run_id"], "day": "2026-01-01", "index_ns": index.asi8.tolist(),
        "raw": {"original": raw.tolist(), "interaction": [55.] * len(index)},
        "audit": {"original": audit, "interaction": {}}, "baseline_reconstruction": evidence, "origin": None}
    subject.verify_origin(prepared, identity, "2026-01-01", index, payload)
    if field == "raw":
        payload["raw"]["original"][0] += .1
    elif field == "audit":
        payload["audit"]["original"]["actual_fit_performed"] = True
    elif field == "evidence":
        payload["baseline_reconstruction"]["source_day_sha256"] = "0" * 64
    elif field == "missing":
        payload["baseline_reconstruction"] = None
    else:
        payload["origin"] = {"unapproved": "fitted_source"}
    with pytest.raises(ValueError):
        subject.verify_origin(prepared, identity, "2026-01-01", index, payload)


def test_mock_annual_reuse_preserves_dates_forecast_and_seals_24_artifacts(monkeypatch, tmp_path):
    """Whole coordinator with fake science, parent-only writes and reverse completion."""
    prepared, overlay = _annual_prepared(monkeypatch, tmp_path)
    identity = _identity(prepared, monkeypatch)
    monkeypatch.setattr(original, "prepare", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(original, "corrector_controls", lambda *args, **kwargs: ([], {}))
    monkeypatch.setattr(original, "baseline_controls", lambda *args, **kwargs: [])
    def confined(path):
        path = Path(path).absolute()
        assert path.is_relative_to(subject.OUTPUT.absolute())
        return path
    monkeypatch.setattr(subject, "safe_path", confined)  # Path guards have separate adversarial tests.
    fits = []
    def fake_fit(item, day, *, interaction):
        assert interaction is True  # Every zero baseline is provably interior.
        fits.append(str(day))
        train = original.train_window(item.raw_history.index, day, "Europe/Berlin")
        assert all(train.tz_convert("Europe/Berlin").date < pd.Timestamp(day).date())
        return np.full(len(subject._base_for_day(item, day)), 55.), {"generation_source": "synthetic_fit", "causality_violations": 0}
    monkeypatch.setattr(original, "_fit_raw", fake_fit)
    saved_frames = {}
    def write_frame(path, frame):
        path = confined(path)
        saved_frames[path] = frame.copy(deep=True)
        path.write_bytes(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    monkeypatch.setattr(subject, "write_frame", write_frame)
    submitted = []
    def fake_kalman(task):
        history, future = subject._UPSTREAMS[task["variant"]]
        selected = history.loc[history.index.tz_convert("Europe/Berlin").date == pd.Timestamp(task["day"]).date()]
        last = task["day"] == "2026-09-21"
        audit = {"causality_violations": 0, "quantile_crossings": 0, "future_observations_assimilated": 0,
            "evaluation_days": 1, "evaluation_hours": len(selected), "future_forecast_hours": 24 if last else 0,
            "training_lookback_days": 365, "config": prepared.config, "covariate_config": prepared.covconfig}
        return {"backtest": overlay(selected), "forecast": overlay(future) if last else None, "audit": audit, "elapsed_seconds": .01}
    monkeypatch.setattr(subject, "_compute_kalman", fake_kalman)
    def reverse_pool(item, identity, tasks, worker, on_result, progress, *, upstreams=None, control_raw=None):
        monkeypatch.setattr(subject, "_PREPARED", item)
        monkeypatch.setattr(subject, "_UPSTREAMS", upstreams)
        monkeypatch.setattr(subject, "_CONTROL_RAW", control_raw or {})
        for task in reversed(tasks):
            submitted.append(task["day"])
            on_result(task, worker(task))
    monkeypatch.setattr(subject, "_run_pool_tasks", reverse_pool)
    result = subject.run(action="run", zones=("DE",))
    directory = subject.OUTPUT / "de" / identity["run_id"]
    assert result == {"DE": str(directory / "report.html")}
    assert len(fits) == 366 and len(submitted) == 366 + 365 * 3
    assert fits[0] == "2026-09-22"
    assert subject._verify_completion(directory, identity)
    audit = json.loads((directory / "reconstruction_audit.json").read_text(encoding="utf-8"))
    assert audit["complete_daily_accounting"] == 366 and len(audit["reconstructed_days"]) == 366
    assert audit['reconstructed_days_count'] == 366
    assert audit['actual_new_baseline_fits_avoided'] == sum(
        day['source_generation_source'] == 'daily_prequential_refit' for day in audit['reconstructed_days'])
    assert 0 < audit['actual_new_baseline_fits_avoided'] < 366  # Cold starts are not fitted models.
    assert audit["source_results_unchanged"] is True
    for variant in subject.VARIANTS:
        backtest = saved_frames[directory / variant / "backtest.parquet"]
        forecast = saved_frames[directory / variant / "forecast.parquet"]
        pd.testing.assert_index_equal(backtest.index, prepared.raw_history.index)
        assert len(backtest) == 8760 and len(forecast) == 24
        pd.testing.assert_series_equal(backtest.actual, prepared.raw_history.actual)
        assert "actual" not in forecast
        expected = 142. if variant == "interaction_40" else 157. if variant == "interaction_80" else 102.
        np.testing.assert_array_equal(backtest["residual_kalman__q50"], np.full(8760, expected))
    monkeypatch.setattr(subject, "_run_pool_tasks", lambda *args, **kwargs: pytest.fail("No workers on complete resume"))
    assert subject.run(action="run", zones=("DE",)) == result


def _permission_error(code):
    error = PermissionError("synthetic Windows sharing error")
    error.winerror = code
    return error


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_transient_atomic_replace_retries_without_deleting_old_or_temporary(monkeypatch, tmp_path, winerror):
    destination, temporary = tmp_path / "status.json", tmp_path / "status.json.tmp"
    destination.write_bytes(b"previous")
    temporary.write_bytes(b"next")
    real_replace, attempts, delays = Path.replace, [], []
    def replace(path, target):
        assert path == temporary and target == destination
        assert temporary.read_bytes() == b"next" and destination.read_bytes() == b"previous"
        attempts.append(path)
        if len(attempts) < 3:
            raise _permission_error(winerror)
        return real_replace(path, target)
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(subject.time, "sleep", delays.append)
    subject._replace_retry(temporary, destination)
    assert len(attempts) == 3 and delays == [.05, .1]
    assert destination.read_bytes() == b"next" and not temporary.exists()


def test_persistent_atomic_access_denied_is_bounded_and_preserves_old_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(subject, "OUTPUT", tmp_path)
    destination, checkpoint = tmp_path / "status.json", tmp_path / "committed_checkpoint.json"
    destination.write_bytes(b"previous")
    checkpoint.write_bytes(b"already-sealed")
    attempts, delays = [], []
    def replace(path, target):
        attempts.append(path)
        assert destination.read_bytes() == b"previous"
        assert checkpoint.read_bytes() == b"already-sealed"
        raise _permission_error(5)
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(subject.time, "sleep", delays.append)
    with pytest.raises(PermissionError):
        subject.write_json(destination, {"status": "RUNNING"})
    assert len(attempts) == 8 and len(delays) == 7
    assert sum(delays) < 3.
    assert destination.read_bytes() == b"previous"
    assert checkpoint.read_bytes() == b"already-sealed"
    assert json.loads((tmp_path / "status.json.tmp").read_text(encoding="utf-8")) == {"status": "RUNNING"}


@pytest.mark.parametrize("error", [_permission_error(87), OSError("disk failure"), ValueError("bad destination")])
def test_nontransient_write_errors_are_not_retried(monkeypatch, tmp_path, error):
    temporary, destination = tmp_path / "pending", tmp_path / "existing"
    temporary.write_bytes(b"next")
    destination.write_bytes(b"previous")
    attempts = []
    def replace(path, target):
        attempts.append(path)
        raise error
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(subject.time, "sleep", lambda *args: pytest.fail("No retry of nontransient errors"))
    with pytest.raises(type(error)):
        subject._replace_retry(temporary, destination)
    assert len(attempts) == 1
    assert temporary.read_bytes() == b"next" and destination.read_bytes() == b"previous"


@pytest.mark.parametrize("kind", ["json", "frame", "text"])
def test_all_publication_writers_use_atomic_replace_helper(monkeypatch, tmp_path, kind):
    monkeypatch.setattr(subject, "OUTPUT", tmp_path)
    destination = tmp_path / ("result." + kind)
    destination.write_bytes(b"previous")
    calls = []
    def replace(temporary, target):
        assert target == destination
        assert target.read_bytes() == b"previous"
        assert temporary.exists() and temporary != target
        calls.append((temporary, target))
        temporary.replace(target)
    monkeypatch.setattr(subject, "_replace_retry", replace)
    if kind == "json":
        subject.write_json(destination, {"next": True})
    elif kind == "frame":
        subject.write_frame(destination, SimpleNamespace(to_parquet=lambda path: path.write_bytes(b"fake parquet")))
    else:
        subject.write_text(destination, "nouveau rapport")
    assert len(calls) == 1 and destination.read_bytes() != b"previous"
