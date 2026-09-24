"""Regression coverage for partial auction publication and safe bootstrap reuse."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

import chronos2_exogenous.prospective_trial as trial
from chronos2_exogenous import evaluation, lora_finetune, prospective_inputs
from chronos2_exogenous.prospective_auxiliary import MARKET_COLUMNS
from test_prospective_trial import _raw


def _observed(start, end, *, value=73.0):
    raw = _raw(start, (pd.Timestamp(end) - pd.Timestamp(start)).days + 1)
    return pd.Series(value, index=pd.DatetimeIndex(raw.delivery_start_utc))


def _forbidden(*_args, **_kwargs):
    raise AssertionError("This operation must not be reached")


def _scenario(tmp_path, monkeypatch, histories, targets, *, end_day="2026-09-08"):
    output = tmp_path / "trial"
    output.mkdir()
    config = {"output_root": str(output), "project_root": str(tmp_path), "candidate_config": "candidate.yaml"}
    manifest = {"config": config, "created_at_utc": "2026-09-01T05:00:00+00:00",
                "schema": {"known_future_covariates": list(MARKET_COLUMNS)},
                "zones": {zone: {"bundle": str(tmp_path / f"{zone}_bundle")} for zone in histories}}
    trial._write_json(output / "trial_manifest.json", manifest)
    candidate = SimpleNamespace(timestamp_column="timestamp", origin_column="origin_timestamp",
                                item_column="item_id")
    days = pd.date_range("2026-09-03", end_day, freq="D").strftime("%Y-%m-%d").tolist()
    pieces = []
    for zone in histories:
        for day in days:
            raw = _raw(day, 1)
            pieces.append(pd.DataFrame({"timestamp": raw.delivery_start_utc,
                "origin_timestamp": raw.forecast_origin_utc, "item_id": zone,
                "phase": "horizon", "delivery_day": day}))
    panel = pd.concat(pieces, ignore_index=True)
    capture_path = output / "capture.json"
    capture = {"panel_path": str(output / "panel.parquet"), "manifest_path": str(capture_path),
               "horizon_labels_available_at_capture": {
                   day: {zone: int(np.isfinite(target.reindex(pd.DatetimeIndex(_raw(day, 1).delivery_start_utc))).sum())
                         for zone, target in targets.items()} for day in days}}
    trial._write_json(capture_path, capture)
    model_loads, predictions, preparations, reuses = [], [], [], []
    monkeypatch.setattr(trial, "verify_trial", lambda _config: manifest)
    monkeypatch.setattr(trial, "load_history", lambda _manifest, zone: histories[zone].copy())
    monkeypatch.setattr(trial, "_targets", lambda _capture, zone: targets[zone].copy())
    monkeypatch.setattr(trial, "now_utc", lambda: pd.Timestamp("2026-09-08T06:30:00Z"))
    monkeypatch.setattr(lora_finetune, "load_config", lambda *_args, **_kwargs: candidate)
    def load(*args, **kwargs):
        model_loads.append((args, kwargs))
        return object()
    monkeypatch.setattr(lora_finetune, "load_checkpoint", load)
    monkeypatch.setattr(pd, "read_parquet", lambda *_args, **_kwargs: panel.copy())
    monkeypatch.setattr(evaluation, "_prepare_shadow_panel", lambda frame, _candidate: frame.drop(columns=["phase", "delivery_day"]))
    def prepare(**kwargs):
        preparations.append(kwargs)
        return capture
    monkeypatch.setattr(prospective_inputs, "prepare_trial_inputs", prepare)
    def reuse(*_args, **kwargs):
        reuses.append(kwargs)
        return capture
    monkeypatch.setattr(trial, "_reuse_bootstrap_inputs", reuse)
    def predict(group, _candidate, _pipeline):
        zone, day = group.item_id.unique().item(), group.delivery_day.unique().item()
        predictions.append((zone, day))
        return _raw(day, 1).assign(actual=np.nan)
    monkeypatch.setattr(trial, "_predict_group", predict)
    return SimpleNamespace(config=config, manifest=manifest, output=output, model_loads=model_loads,
                           predictions=predictions, preparations=preparations, reuses=reuses)


def test_bootstrap_resume_independent_countries_and_sealed_fr_preserved(tmp_path, monkeypatch):
    histories = {zone: _raw("2026-09-02", length) for zone, length in {"FR": 7, "DE": 6, "BE": 5, "NL": 4}.items()}
    targets = {zone: _observed("2026-09-03", "2026-09-08" if zone in {"FR", "BE"} else "2026-09-07")
               for zone in histories}
    case = _scenario(tmp_path, monkeypatch, histories, targets)
    old_fr = case.output / "days/2026-09-08/FR/manifest.json"
    old_fr.parent.mkdir(parents=True)
    trial._write_json(old_fr, {"immutable": "already sealed France"})
    old_bytes = old_fr.read_bytes()
    result = trial.execute_trial(case.config, delivery_day="2026-09-08", zones=list(histories), prospective=False)
    assert result["status"] == "waiting_for_observations"
    assert result["already_complete"] == ["FR"]
    assert [(entry["zone"], entry["delivery_day"], entry["observed_hours"]) for entry in result["pending"]] == [
        ("DE", "2026-09-08", 0), ("NL", "2026-09-08", 0)]
    assert case.predictions == [("BE", "2026-09-07"), ("BE", "2026-09-08"),
                                ("NL", "2026-09-06"), ("NL", "2026-09-07")]
    assert len(case.model_loads) == 1 and case.preparations == []
    assert case.reuses[0]["zones"] == ["DE", "BE", "NL"]
    assert old_fr.read_bytes() == old_bytes
    assert len(result["published"]) == 4
    for published in result["published"]:
        daily = trial._json(Path(published))
        assert daily["prospective_eligible"] is False
        assert daily["role"] == "retrospective_calibration_only"
        assert daily["promotion_eligible"] is False
        assert trial._read_frame(Path(daily["raw_predictions"]["path"])).actual.eq(73.0).all()
    assert not (case.output / "days/2026-09-08/DE").exists()
    assert not (case.output / "days/2026-09-08/NL").exists()


def test_bootstrap_waiting_does_not_load_checkpoint_or_panel(tmp_path, monkeypatch):
    case = _scenario(tmp_path, monkeypatch, {"DE": _raw("2026-09-02", 6)},
                     {"DE": _observed("2026-09-03", "2026-09-07")})
    monkeypatch.setattr(lora_finetune, "load_checkpoint", _forbidden)
    monkeypatch.setattr(lora_finetune, "load_config", _forbidden)
    monkeypatch.setattr(pd, "read_parquet", _forbidden)
    result = trial.execute_trial(case.config, delivery_day="2026-09-08", zones=["DE"], prospective=False)
    assert result["status"] == "waiting_for_observations"
    assert result["published"] == [] and len(result["pending"]) == 1
    assert case.predictions == []
    assert not (case.output / "days").exists()


def test_bootstrap_already_complete_does_not_refresh_or_reload_anything(tmp_path, monkeypatch):
    case = _scenario(tmp_path, monkeypatch, {"FR": _raw("2026-09-02", 7)}, {"FR": pd.Series(dtype=float)})
    monkeypatch.setattr(lora_finetune, "load_checkpoint", _forbidden)
    monkeypatch.setattr(prospective_inputs, "prepare_trial_inputs", _forbidden)
    monkeypatch.setattr(trial, "_reuse_bootstrap_inputs", _forbidden)
    monkeypatch.setattr(trial, "_targets", _forbidden)
    result = trial.execute_trial(case.config, delivery_day="2026-09-08", zones=["FR"], prospective=False)
    assert result["status"] == "already_complete"
    assert result["already_complete"] == ["FR"] and result["published"] == []
    assert case.model_loads == [] and case.predictions == []


def test_run_missing_previous_day_refuses_before_model_load(tmp_path, monkeypatch):
    case = _scenario(tmp_path, monkeypatch, {"DE": _raw("2026-09-02", 6)},
                     {"DE": _observed("2026-09-03", "2026-09-07")}, end_day="2026-09-09")
    monkeypatch.setattr(lora_finetune, "load_checkpoint", _forbidden)
    monkeypatch.setattr(pd, "read_parquet", _forbidden)
    monkeypatch.setattr(trial, "_reuse_bootstrap_inputs", _forbidden)
    with pytest.raises(trial.ProspectiveTrialError, match="Calibration prealable incomplete.*DE/2026-09-08"):
        trial.execute_trial(case.config, delivery_day="2026-09-09", zones=["DE"], prospective=True)
    assert case.predictions == [] and len(case.preparations) == 1
    assert not (case.output / "days").exists()


def test_run_existing_unresolved_previous_day_refuses_before_model_load(tmp_path, monkeypatch):
    history = _raw("2026-09-02", 7)
    history.loc[history.index[-24:], "actual"] = np.nan
    target = _observed("2026-09-03", "2026-09-08")
    target.iloc[-24:] = np.nan
    before = history.copy(deep=True)
    case = _scenario(tmp_path, monkeypatch, {"DE": history}, {"DE": target}, end_day="2026-09-09")
    monkeypatch.setattr(lora_finetune, "load_checkpoint", _forbidden)
    monkeypatch.setattr(lora_finetune, "load_config", _forbidden)
    monkeypatch.setattr(pd, "read_parquet", _forbidden)
    monkeypatch.setattr(trial, "_reuse_bootstrap_inputs", _forbidden)
    with pytest.raises(trial.ProspectiveTrialError, match="DE: labels historiques encore absents.*24 h.*2026-09-08"):
        trial.execute_trial(case.config, delivery_day="2026-09-09", zones=["DE"], prospective=True)
    assert case.predictions == [] and len(case.preparations) == 1
    assert not (case.output / "days").exists()
    pd.testing.assert_frame_equal(history, before)


def _old_capture(tmp_path, monkeypatch):
    output = tmp_path / "trial"
    directory = output / "captures/20260907T120000Z_original"
    directory.mkdir(parents=True)
    config = {"output_root": str(output), "project_root": str(tmp_path)}
    schema = {"known_future_covariates": list(MARKET_COLUMNS)}
    original = {"zones": ["FR", "DE", "BE", "NL"], "delivery_days": ["2026-09-07", "2026-09-08"],
                "known_future_covariates": list(MARKET_COLUMNS), "fresh_source_refresh": True,
                "kind": "original_lora_trial_inputs", "target_snapshots": {"DE": {"old": "unchanged"}},
                "source_root": str(directory / "sources")}
    for name in ("panel", "panel_audit", "seed_manifest"):
        path = directory / f"{name}.fixture"
        path.write_bytes(f"original immutable {name}".encode())
        original[f"{name}_path"], original[f"{name}_sha256"] = str(path), trial.sha256(path)
    trial._write_json(directory / "input_manifest.json", original)
    calls = []
    def fetch(**kwargs):
        calls.append(kwargs)
        return _observed(kwargs["start"], kwargs["end"]), {"fresh_api_read": True, "fallback_used": False}
    monkeypatch.setattr(prospective_inputs, "_capture_target", fetch)
    monkeypatch.setattr(prospective_inputs, "prepare_trial_inputs", _forbidden)
    return SimpleNamespace(config=config, schema=schema, directory=directory, output=output, original=original, calls=calls)


def test_reuse_verifies_old_inputs_refreshes_only_targets_and_writes_new_ledger(tmp_path, monkeypatch):
    case = _old_capture(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for path in case.directory.rglob("*") if path.is_file()}
    capture = trial._reuse_bootstrap_inputs(case.config, days=["2026-09-08"], zones=["DE", "NL"], schema=case.schema)
    assert [call["zone"] for call in case.calls] == ["DE", "NL"]
    assert all(call["refresh"] is True and call["start"] == call["end"] == "2026-09-08" for call in case.calls)
    assert capture["panel_path"] == case.original["panel_path"]
    assert capture["fresh_source_refresh"] is False
    assert capture["reuse_scope"] == "retrospective_calibration_only"
    assert capture["kind"] == "retrospective_bootstrap_input_reuse"
    assert capture["production_pit_evidence"] is False
    assert capture["horizon_labels_available_at_capture"] == {"2026-09-08": {"DE": 24, "NL": 24}}
    new_manifest = Path(capture["manifest_path"])
    assert new_manifest.is_relative_to(case.output / "bootstrap_attempts")
    assert new_manifest.is_file()
    assert trial._targets(capture, "DE").eq(73.0).all()
    assert capture["reused_input_manifest"] == trial._file(case.directory / "input_manifest.json")
    assert {path: path.read_bytes() for path in before} == before
    assert set(path for path in case.directory.rglob("*") if path.is_file()) == set(before)
    second = trial._reuse_bootstrap_inputs(case.config, days=["2026-09-08"], zones=["DE"], schema=case.schema)
    assert second["manifest_path"] != capture["manifest_path"]
    assert trial._json(new_manifest)["target_snapshots"] == capture["target_snapshots"]


@pytest.mark.parametrize("name", ["panel", "panel_audit", "seed_manifest"])
def test_reuse_tampered_input_refused_before_api_or_new_artifacts(tmp_path, monkeypatch, name):
    case = _old_capture(tmp_path, monkeypatch)
    Path(case.original[f"{name}_path"]).write_bytes(b"changed after capture")
    with pytest.raises(trial.ProspectiveTrialError, match="Empreinte"):
        trial._reuse_bootstrap_inputs(case.config, days=["2026-09-08"], zones=["DE"], schema=case.schema)
    assert case.calls == []
    assert not (case.output / "bootstrap_attempts").exists()


def test_reuse_sealed_capture_manifest_refuses_rehashed_panel_replacement(tmp_path, monkeypatch):
    case = _old_capture(tmp_path, monkeypatch)
    original_manifest = case.directory / "input_manifest.json"
    sealed_day = case.output / "days/2026-09-07/FR/manifest.json"
    sealed_day.parent.mkdir(parents=True)
    trial._write_json(sealed_day, {"input_manifest": trial._file(original_manifest)})
    original_seal_bytes = sealed_day.read_bytes()
    panel = Path(case.original["panel_path"])
    panel.write_bytes(b"replacement panel with internally matching updated hash")
    altered = {**case.original, "panel_sha256": trial.sha256(panel)}
    original_manifest.write_text(json.dumps(altered), encoding="utf-8")
    # The replacement is internally consistent, but not the capture previously
    # committed by the sealed day. Recomputing all inner SHAs must not hide it.
    trial._verify_file({"path": altered["panel_path"], "sha256": altered["panel_sha256"]})
    with pytest.raises(trial.ProspectiveTrialError, match="Empreinte"):
        trial._reuse_bootstrap_inputs(case.config, days=["2026-09-08"], zones=["DE"], schema=case.schema)
    assert case.calls == [] and not (case.output / "bootstrap_attempts").exists()
    assert sealed_day.read_bytes() == original_seal_bytes


@pytest.mark.parametrize("changes", [{"days": ["2026-09-09"]}, {"zones": ["ES"]},
                                     {"schema": {"known_future_covariates": ["different_covariate"]}}])
def test_incompatible_capture_is_not_reused(tmp_path, monkeypatch, changes):
    case = _old_capture(tmp_path, monkeypatch)
    kwargs = {"days": ["2026-09-08"], "zones": ["DE"], "schema": case.schema, **changes}
    assert trial._reuse_bootstrap_inputs(case.config, **kwargs) is None
    assert case.calls == [] and not (case.output / "bootstrap_attempts").exists()
