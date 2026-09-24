"""Runner integrity tests: frozen synthetic artifacts, never model training."""
from __future__ import annotations

from datetime import date
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import run_catboost_rmse as runner
import nyx_catboost_rmse.core as core


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "OUTPUT", root / "runs/experiments/catboost_rmse_v1")
    monkeypatch.setattr(runner, "code_identity", lambda: {"files": {"synthetic": "sealed"}, "runtime": {}})
    return {"delivery_day": "2026-09-19"}, runner.OUTPUT / "2026-09-19"


def _day(day):
    start = pd.Timestamp(day, tz="Europe/Paris")
    return pd.date_range(start, start + pd.DateOffset(days=1), freq="h", inclusive="left").tz_convert("UTC").rename("delivery_start_utc")


def _predictions(day="2025-09-20", raw=55.):
    index = _day(day)
    applied = float(np.clip(raw, -40, 40))
    origin = (pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
    return pd.DataFrame({"q10": 20. + applied, "q50": 30. + applied, "q90": 50. + applied,
                         "raw_correction": raw, "applied_correction": applied,
                         "forecast_origin_utc": origin}, index=index)


def _audit(day="2025-09-20"):
    return {"loss": "RMSE", "delivery_day": day, "training_days": 365,
            "current_day_labels_used": False}


def _prepared(root):
    work = root / "fr"
    index = runner.period({"delivery_day": "2026-09-19"}, "Europe/Paris")
    raw = pd.DataFrame({"q10": 20., "q50": 30., "q90": 50., "actual": 45.,
                        "forecast_origin_utc": pd.Timestamp("2025-09-19T06:00Z")}, index=index[:-24])
    future = pd.DataFrame({"q10": 20., "q50": 30., "q90": 50.,
                           "forecast_origin_utc": pd.Timestamp("2026-09-18T06:00Z")}, index=index[-24:])
    reference = pd.DataFrame({"actual": 45., "chronos_q50": 30., "mae_q50": 30., "storm_q50": np.nan}, index=index)
    daily = pd.DataFrame({"delivery_day": pd.date_range("2025-09-20", "2026-09-19", freq="D").strftime("%Y-%m-%d"),
                          "generation_source": "daily_prequential_refit",
                          "residual_feature_columns": [["nuclear"] for _ in range(365)]})
    frames = {"features": pd.DataFrame({"nuclear": 40.}, index=index), "raw_history": raw,
              "future": future, "reference": reference, "baseline_daily_audit": daily}
    source = runner.safe(work / "inputs")
    source.mkdir(parents=True)
    for name, frame in frames.items():
        frame.to_parquet(source / (name + ".parquet"))
    runner.write(source / "recipe.json", {"fixture": "no real fits"})
    runner.write(source / "provenance.json", {"synthetic": True})
    record = {"schema_version": 1, "zone": "FR", "timezone": "Europe/Paris", "delivery_day": "2026-09-19",
              "evaluation_start": "2025-09-20", "evaluation_end": "2026-09-19",
              "files": {name: runner.sha(source / name) for name in runner.INPUT_FILES}}
    runner.write(source / "manifest.json", record)
    return work, record, frames


def test_safe_refuses_root_escape_and_noncanonical_path(isolated):
    _, root = isolated
    assert runner.safe(root / "fr/inputs") == root / "fr/inputs"
    for path in (runner.ROOT, runner.OUTPUT, runner.ROOT / "runs/live/evil.json",
                 runner.OUTPUT / ".." / "outside.json"):
        with pytest.raises(ValueError, match="unredirected experiment"):
            runner.safe(path)


def test_safe_refuses_existing_redirect(isolated, tmp_path):
    _, root = isolated
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "redirect"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating directory symlinks is unavailable on this Windows host.")
    with pytest.raises(ValueError, match="unredirected experiment"):
        runner.safe(link / "predictions.parquet")


def test_input_checksum_and_inventory_fail_closed(isolated):
    _, root = isolated
    work, _, _ = _prepared(root)
    assert runner.verify_prepared(work)["zone"] == "FR"
    runner.write(work / "inputs/recipe.json", {"changed": True})
    with pytest.raises(ValueError, match="checksum mismatch"):
        runner.verify_prepared(work)
    manifest = runner.read(work / "inputs/manifest.json")
    manifest["files"].pop("recipe.json")
    runner.write(work / "inputs/manifest.json", manifest)
    with pytest.raises(ValueError, match="inventory"):
        runner.verify_prepared(work)


def test_checkpoint_preserves_preclip_and_refuses_overwrite(isolated):
    _, root = isolated
    folder = root / "fr/checkpoints/contract/2025-09-20"
    predicted = _predictions(raw=55.)
    audit = {**_audit(), "raw_max": 55.}
    runner.save_day(folder, predicted, audit, "contract", date(2025, 9, 20))
    loaded, loaded_audit = runner.cached_day(folder, contract_sha="contract", day=date(2025, 9, 20), expected_index=predicted.index)
    pd.testing.assert_frame_equal(loaded, predicted, check_freq=False)
    assert loaded.raw_correction.eq(55).all() and loaded.applied_correction.eq(40).all()
    assert loaded_audit == audit
    with pytest.raises(ValueError, match="overwrite"):
        runner.save_day(folder, predicted, audit, "contract", date(2025, 9, 20))


@pytest.mark.parametrize("fault", ["bytes", "contract", "day", "index", "preclip", "audit", "crossed_quantile"])
def test_corrupt_checkpoint_is_retained_and_not_reused(isolated, fault):
    _, root = isolated
    folder = root / "fr/checkpoints/contract/2025-09-20"
    predicted = _predictions()
    runner.save_day(folder, predicted, _audit(), "contract", "2025-09-20")
    contract, day, index = "contract", "2025-09-20", predicted.index
    if fault == "bytes":
        runner.write(folder / "audit.json", {"tampered": True})
    elif fault == "contract":
        contract = "different"
    elif fault == "day":
        day = "2025-09-21"
    elif fault == "index":
        index = index[:-1]
    elif fault in ("preclip", "crossed_quantile"):
        broken = predicted.copy()
        broken["raw_correction" if fault == "preclip" else "q10"] = 0. if fault == "preclip" else 500.
        broken.to_parquet(folder / "predictions.parquet")
        manifest = runner.read(folder / "manifest.json")
        manifest["files"]["predictions.parquet"] = runner.sha(folder / "predictions.parquet")
        runner.write(folder / "manifest.json", manifest)
    elif fault == "audit":
        runner.write(folder / "audit.json", {**_audit(), "current_day_labels_used": True})
        manifest = runner.read(folder / "manifest.json")
        manifest["files"]["audit.json"] = runner.sha(folder / "audit.json")
        runner.write(folder / "manifest.json", manifest)
    with pytest.raises(ValueError):
        runner.cached_day(folder, contract_sha=contract, day=day, expected_index=index)
    assert folder.is_dir() and (folder / "predictions.parquet").is_file()


def test_prediction_day_strips_evaluation_labels_and_uses_frozen_future(isolated):
    _, root = isolated
    _, _, inputs = _prepared(root)
    past = runner.prediction_day(inputs, "2025-09-20", "Europe/Paris")
    future = runner.prediction_day(inputs, "2026-09-19", "Europe/Paris")
    assert "actual" not in past and "actual" not in future
    assert future.index.equals(inputs["future"].index)
    inputs["reference"].loc[future.index, "actual"] = 123456.
    pd.testing.assert_frame_equal(future, runner.prediction_day(inputs, "2026-09-19", "Europe/Paris"))
    joined = runner.paired(inputs, _predictions("2026-09-19"))
    assert joined.actual.eq(123456).all()


def test_paired_refuses_changed_frozen_chronos(isolated):
    _, root = isolated
    _, _, inputs = _prepared(root)
    predicted = _predictions()
    predicted["q50"] += .5
    with pytest.raises(ValueError, match="altered frozen Chronos"):
        runner.paired(inputs, predicted)


def test_resume_uses_checkpoint_without_fitting(isolated, monkeypatch):
    cfg, root = isolated
    work, _, inputs = _prepared(root)
    signature = runner.digest(runner.fit_contract(work, 2))
    cache = work / "checkpoints" / signature[:16]
    runner.save_day(cache / "2025-09-20", _predictions(), _audit(), signature, "2025-09-20")
    runner.write(cache / "mae_parity.json", {"contract_sha256": signature, "passed": True})
    def no_fit(**_kwargs):
        pytest.fail("A valid completed daily checkpoint must not retrain.")
    monkeypatch.setattr(core, "fit_day", no_fit)
    reports = []
    monkeypatch.setattr(runner, "report", lambda *args: reports.append(args) or {})
    runner.run_zone(cfg, root, "FR", 2, 1)
    assert runner.read(work / "status.json")["status"] == "partial"
    assert len(reports) == 2
    saved, _ = runner.cached_day(cache / "2025-09-20", contract_sha=signature, day="2025-09-20", expected_index=_day("2025-09-20"))
    assert saved.raw_correction.eq(55).all()


def test_report_action_reads_only_and_never_fits(isolated, monkeypatch):
    cfg, root = isolated
    work, _, _ = _prepared(root)
    contract = runner.fit_contract(work, 2)
    signature = runner.digest(contract)
    runner.write(work / "latest_fit.json", {"contract_sha256": signature, "contract": contract})
    runner.save_day(work / "checkpoints" / signature[:16] / "2025-09-20", _predictions(), _audit(), signature, "2025-09-20")
    def forbidden(*_args, **_kwargs):
        pytest.fail("The report action must never prepare inputs or fit a model.")
    monkeypatch.setattr(core, "fit_day", forbidden)
    monkeypatch.setattr(runner, "prepare_zone", forbidden)
    monkeypatch.setattr(runner, "run_zone", forbidden)
    monkeypatch.setattr(runner, "code_identity", forbidden)
    monkeypatch.setattr(runner, "settings", lambda _path: cfg)
    calls = []
    module = ModuleType("nyx_catboost_rmse.report")
    def write_report(frames, metadata, path):
        calls.append((frames, metadata, path))
        return {"html": path / "report.html"}
    module.write_report = write_report
    monkeypatch.setitem(sys.modules, "nyx_catboost_rmse.report", module)
    # Reporting honors the committed identity even if the CLI thread default differs.
    assert runner.main(["--action", "report", "--threads", "3"]) == 0
    frames, metadata, path = calls[0]
    assert set(frames) == {"FR"}
    assert metadata["states"]["FR"]["completed_days"] == 1
    assert metadata["states"]["FR"]["planned_days"] == 365
    assert metadata["production_modified"] is False
    assert frames["FR"].raw_correction.eq(55).all()
    assert path == root / "reports"


def test_resume_rejects_changed_contract_before_fitting(isolated, monkeypatch):
    cfg, root = isolated
    work, _, _ = _prepared(root)
    old_contract = runner.fit_contract(work, 2)
    runner.write(work / "latest_fit.json", {"contract_sha256": runner.digest(old_contract), "contract": old_contract})
    def forbidden(**_kwargs):
        pytest.fail("Incompatible resume must not fit.")
    monkeypatch.setattr(core, "fit_day", forbidden)
    with pytest.raises(ValueError, match="refusing to mix fits"):
        runner.run_zone(cfg, root, "FR", 3, 1)
    assert runner.read(work / "latest_fit.json")["contract"] == old_contract


def test_mae_parity_mismatch_blocks_rmse_before_receipt(isolated, monkeypatch):
    _, root = isolated
    work, record, inputs = _prepared(root)
    calls = []
    def mismatch(**kwargs):
        calls.append(kwargs)
        return _predictions(raw=1.), {"loss": "MAE"}
    monkeypatch.setattr(core, "fit_day", mismatch)
    cache = work / "checkpoints/contract"
    with pytest.raises(ValueError, match="MAE parity failed"):
        runner.parity(work, record, inputs, {}, 2, cache, "contract")
    assert len(calls) == 1 and calls[0]["loss"] == "MAE"
    assert "actual" not in calls[0]["future"]
    assert not (cache / "mae_parity.json").exists()
