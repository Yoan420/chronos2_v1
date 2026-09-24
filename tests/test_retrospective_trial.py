"""Causal, immutable one-day comparison of the two rank-16 chains.

The neural pipeline and auxiliary fit are mocked, while the historical seals,
research protocol, observation snapshots and report files are exercised for real.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pandas as pd
import pytest

from chronos2_exogenous import lora_finetune, prospective_auxiliary, prospective_inputs
from chronos2_exogenous import prospective_trial as trial
from chronos2_exogenous import retrospective_trial as retrospective
from chronos2_exogenous.prospective_auxiliary import QUANTILES, RAW_MODEL
from test_prospective_trial import prepared, _daily, _raw


DAY = "2026-09-03"


def _forbidden(*_args, **_kwargs):
    raise AssertionError("A sealed comparison must not trigger this computation")


@pytest.fixture
def comparison_case(prepared, monkeypatch):
    config, manifest, history, candidate = prepared
    output = Path(config["output_root"])
    daily, raw = _daily(prepared, day=DAY)
    raw["actual"] = 999.0
    raw_path = Path(daily["raw_predictions"]["path"])
    raw.to_csv(raw_path, index=False, compression="gzip")
    daily["raw_predictions"] = trial._file(raw_path)
    daily_path = raw_path.parent / "manifest.json"
    daily_path.write_text(json.dumps(daily), encoding="utf-8")
    monkeypatch.setattr(retrospective, "COMPARISON_CODE", ("scientific.py",))
    monkeypatch.setattr(trial, "now_utc", lambda: pd.Timestamp("2026-09-02T14:00Z"))
    monkeypatch.setattr(lora_finetune, "load_checkpoint", _forbidden)
    monkeypatch.setattr(trial, "_predict_group", _forbidden)
    monkeypatch.setattr(prospective_inputs, "prepare_trial_inputs", _forbidden)
    observations = {"value": 51.0}
    target_reads, fits = [], []
    def capture(**kwargs):
        target_reads.append(kwargs)
        values = _raw(kwargs["start"], (pd.Timestamp(kwargs["end"]) - pd.Timestamp(kwargs["start"])).days + 1)
        return pd.Series(observations["value"], index=pd.DatetimeIndex(values.delivery_start_utc)), {
            "fresh_api_read": True, "fallback_used": False, "capture_completed_at_utc": trial.now_utc().isoformat(),
        }
    monkeypatch.setattr(prospective_inputs, "_capture_target", capture)
    def chains(raw_history, raw_future, **kwargs):
        fits.append((raw_history.copy(deep=True), raw_future.copy(deep=True), kwargs))
        assert raw_history.delivery_start_utc.max() < raw_future.delivery_start_utc.min()
        assert raw_history.actual.notna().all()
        assert not raw_history.actual.eq(999).any()
        assert raw_future.actual.isna().all()
        result = raw_future.copy()
        for prefix, shift in (("lora16_residual", 1.0), ("lora16_residual_kalman", 2.0)):
            for quantile in QUANTILES:
                result[f"{prefix}__{quantile}"] = result[f"{RAW_MODEL}__{quantile}"] + shift
        return SimpleNamespace(kalman_future=result, audit={"target_observations_used": 0, **trial.FLAGS})
    monkeypatch.setattr(prospective_auxiliary, "forecast_trial_chains", chains)
    return SimpleNamespace(config=config, manifest=manifest, history=history, candidate=candidate,
        output=output, daily=daily, raw=raw, raw_path=raw_path, fits=fits,
        observations=observations, target_reads=target_reads)


def _execute(case, **kwargs):
    return retrospective.execute_retrospective_comparison(
        case.config, delivery_day=DAY, zones=["FR"], **kwargs)


def _snapshot(path):
    return {file: file.read_bytes() for file in path.rglob("*") if file.is_file()}


def test_existing_target_day_raw_is_reused_but_its_known_actual_cannot_enter_calibration(comparison_case):
    case = comparison_case
    before = _snapshot(case.output / "days")
    protocol = (case.output / "trial_manifest.json").read_bytes()
    result = _execute(case)
    assert result["prospective_eligible"] is False
    assert result["promotion_eligible"] is False
    assert result["activation_performed"] is False
    assert len(case.fits) == 1
    history, future, _ = case.fits[0]
    assert len(history) == len(case.history)
    assert history.delivery_start_utc.max().tz_convert("Europe/Paris").date().isoformat() == "2026-09-02"
    for quantile in QUANTILES:
        np.testing.assert_array_equal(future[f"{RAW_MODEL}__{quantile}"], case.raw[f"{RAW_MODEL}__{quantile}"])
    assert _snapshot(case.output / "days") == before
    assert (case.output / "trial_manifest.json").read_bytes() == protocol
    assert trial.trial_status(case.config)["zones"]["FR"]["prospective_forecasts"] == 0


def test_missing_target_day_observations_do_not_block_two_chain_comparison(comparison_case):
    case = comparison_case
    case.observations["value"] = np.nan
    result = _execute(case)
    assert len(case.fits) == 1
    assert result["prospective_eligible"] is False
    # Only D-1 labels are required by the model; D labels are optional scoring data.
    assert case.fits[0][0].actual.notna().all()
    assert case.fits[0][1].actual.isna().all()
    assert result["observed_hours"] == {"FR": 0}
    report_manifest = trial._json(Path(result["report"]).parent / "report_manifest.json")
    evidence = trial._read_frame(Path(report_manifest["evidence"]["path"]))
    assert evidence.actual.isna().all()


def test_rerun_only_refreshes_observations_and_report_without_neural_or_auxiliary_refit(comparison_case, monkeypatch):
    case = comparison_case
    first = _execute(case)
    forecast_path = Path(first["forecast_directory"]) / "FR"
    frozen = _snapshot(forecast_path)
    old_report = _snapshot(Path(first["report"]).parent)
    first_manifest = trial._json(Path(first["report"]).parent / "report_manifest.json")
    first_values = trial._read_frame(Path(first_manifest["evidence"]["path"]))
    assert first_values.actual.eq(51).all()
    monkeypatch.setattr(prospective_auxiliary, "forecast_trial_chains", _forbidden)
    case.observations["value"] = 57.0
    second = _execute(case)
    assert second["report"] != first["report"]
    assert _snapshot(forecast_path) == frozen
    assert _snapshot(Path(first["report"]).parent) == old_report
    assert len(case.fits) == 1 and len(case.target_reads) == 2
    second_manifest = trial._json(Path(second["report"]).parent / "report_manifest.json")
    second_values = trial._read_frame(Path(second_manifest["evidence"]["path"]))
    assert second_values.actual.eq(57).all()
    for prefix in ("lora16_residual", "lora16_residual_kalman", RAW_MODEL):
        for quantile in QUANTILES:
            pd.testing.assert_series_equal(first_values[f"{prefix}__{quantile}"], second_values[f"{prefix}__{quantile}"])


@pytest.mark.parametrize("filename", ["predictions.csv.gz", "raw.csv.gz", "history.csv.gz", "auxiliary_audit.json"])
def test_tampered_cached_artifact_refused_without_recomputing(comparison_case, monkeypatch, filename):
    case = comparison_case
    first = _execute(case)
    monkeypatch.setattr(prospective_auxiliary, "forecast_trial_chains", _forbidden)
    artifact = Path(first["forecast_directory"]) / "FR" / filename
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    with pytest.raises(trial.ProspectiveTrialError, match="Empreinte"):
        _execute(case)
    assert len(case.fits) == 1 and len(case.target_reads) == 1


@pytest.mark.parametrize("filename", ["raw.csv.gz", "manifest.json"])
def test_cached_comparison_still_checks_original_raw_provenance(comparison_case, monkeypatch, filename):
    case = comparison_case
    _execute(case)
    monkeypatch.setattr(prospective_auxiliary, "forecast_trial_chains", _forbidden)
    original = case.raw_path.parent / filename
    original.write_bytes(original.read_bytes() + b"changed after first comparison")
    with pytest.raises(trial.ProspectiveTrialError, match="Empreinte"):
        _execute(case)
    assert len(case.fits) == 1 and len(case.target_reads) == 1


def test_scientific_source_checksum_change_is_not_hidden_by_retrospective_mode(comparison_case):
    case = comparison_case
    source = Path(case.config["project_root"]) / "scientific.py"
    source.write_text("# silently changed model implementation\n", encoding="utf-8")
    with pytest.raises(trial.ProspectiveTrialError):
        _execute(case)
    assert case.fits == [] and case.target_reads == []


def test_comparison_cannot_be_used_before_the_forecast_origin(comparison_case, monkeypatch):
    case = comparison_case
    monkeypatch.setattr(trial, "now_utc", lambda: pd.Timestamp("2026-09-02T05:59:59Z"))
    with pytest.raises(retrospective.RetrospectiveComparisonError):
        _execute(case)
    assert case.fits == [] and case.target_reads == []


@pytest.mark.parametrize("zones", [[], ["FR", "FR"], ["DE"], ["ES"]])
def test_invalid_zone_selection_is_rejected_without_running_models(comparison_case, zones):
    case = comparison_case
    with pytest.raises(retrospective.RetrospectiveComparisonError):
        retrospective.execute_retrospective_comparison(case.config, delivery_day=DAY, zones=zones)
    assert case.fits == []


def test_missing_de_nl_target_labels_still_allow_raw_inference_and_both_chains(comparison_case, monkeypatch):
    from chronos2_exogenous import evaluation
    case = comparison_case
    manifest = {**case.manifest, "zones": {
        zone: case.manifest["zones"]["FR"] for zone in ("DE", "NL")}}
    monkeypatch.setattr(trial, "verify_trial", lambda _config: manifest)
    monkeypatch.setattr(trial, "load_history", lambda _manifest, _zone: case.history.copy())
    case.observations["value"] = np.nan
    horizon = pd.DatetimeIndex(case.raw.delivery_start_utc)
    context = pd.date_range(end=horizon[0] - pd.Timedelta(hours=1), periods=4, freq="h")
    index = context.append(horizon)
    blocks = []
    for zone in ("DE", "NL"):
        block = pd.DataFrame({"timestamp": index, "origin_timestamp": pd.Timestamp("2026-09-02T06:00Z"),
            "item_id": zone, "target": [52.0] * 4 + [999.0] * len(horizon),
            "feature_available_at_utc": pd.Timestamp("2026-09-02T05:00Z"),
            "phase": ["context"] * 4 + ["horizon"] * len(horizon), "delivery_day": DAY})
        for column in prospective_auxiliary.MARKET_COLUMNS:
            block[column] = 20.0
        blocks.append(block)
    panel = pd.concat(blocks, ignore_index=True)
    capture = case.output / "input_comparison_fixture.json"
    panel_path = case.output / "input_comparison_fixture.parquet"
    panel_path.write_bytes(b"checked fixture bytes")
    trial._write_json(capture, {"fixture": True})
    monkeypatch.setattr(retrospective, "_find_existing_panel", lambda *_args: {
        "panel_path": str(panel_path), "panel_sha256": trial.sha256(panel_path), "manifest_path": str(capture),
    })
    monkeypatch.setattr(pd, "read_parquet", lambda *_args, **_kwargs: panel.copy(deep=True))
    seen = []
    def validate(frame, _candidate):
        assert frame.loc[frame.phase.eq("horizon"), "target"].isna().all()
        assert frame.loc[frame.phase.eq("context"), "target"].eq(52).all()
        return frame.drop(columns=["phase", "delivery_day"])
    monkeypatch.setattr(evaluation, "_prepare_shadow_panel", validate)
    loads = []
    def load(*args, **kwargs):
        loads.append((args, kwargs))
        return object()
    monkeypatch.setattr(lora_finetune, "load_checkpoint", load)
    def predict(group, _candidate, _pipeline):
        seen.append(group.item_id.unique().item())
        assert group.loc[group.phase.eq("horizon"), "target"].isna().all()
        return case.raw.assign(actual=np.nan)
    monkeypatch.setattr(trial, "_predict_group", predict)
    result = retrospective.execute_retrospective_comparison(case.config, delivery_day=DAY, zones=["DE", "NL"])
    assert seen == ["DE", "NL"] and len(loads) == 1 and len(case.fits) == 2
    assert result["observed_hours"] == {"DE": 0, "NL": 0}
    assert not (case.output / "days" / DAY / "DE").exists()
    assert not (case.output / "days" / DAY / "NL").exists()
    assert all(item[0].actual.eq(52).all() for item in case.fits)


@pytest.mark.parametrize(("action", "clock", "expected"), [
    ("run", "2026-09-07T09:44:59Z", "prospective"),
    ("run", "2026-09-07T09:45:00Z", "retrospective"),
    ("run", "2026-09-07T15:15:00Z", "retrospective"),
    ("compare", "2026-09-07T06:30:00Z", "retrospective"),
    ("bootstrap", "2026-09-07T15:15:00Z", "bootstrap"),
])
def test_cli_run_after_deadline_is_explicitly_retrospective_not_a_prospective_gate_bypass(monkeypatch, capsys, action, clock, expected):
    import run_lora_rank16_trial as cli
    monkeypatch.setattr(sys, "argv", ["run_lora_rank16_trial.py", "--action", action,
        "--delivery-day", "2026-09-08", "--zones", "FR", "DE", "BE", "NL"])
    monkeypatch.setattr(cli, "load_trial_config", lambda _path: {"fixture": True})
    monkeypatch.setattr(cli, "now_utc", lambda: pd.Timestamp(clock))
    calls = []
    def normal(_config, **kwargs):
        calls.append(("prospective" if kwargs["prospective"] else "bootstrap", kwargs))
        return {"status": "test"}
    def compare(_config, **kwargs):
        calls.append(("retrospective", kwargs))
        return {"status": "test"}
    resolutions = []
    monkeypatch.setattr(cli, "execute_trial", normal)
    monkeypatch.setattr(retrospective, "execute_retrospective_comparison", compare)
    monkeypatch.setattr(cli, "resolve_and_report", lambda _config: resolutions.append(True))
    assert cli.main() == 0
    assert len(calls) == 1 and calls[0][0] == expected
    assert calls[0][1]["delivery_day"] == "2026-09-08"
    assert calls[0][1]["zones"] == ["FR", "DE", "BE", "NL"]
    assert bool(resolutions) == (expected == "prospective")
    assert ("RETROSPECTIF" in capsys.readouterr().out) == (expected == "retrospective")


def test_existing_prospective_gate_still_refuses_after_deadline():
    with pytest.raises(trial.ProspectiveTrialError, match="aucune prediction antidatee"):
        trial.emission_window("2026-09-08", now=pd.Timestamp("2026-09-07T15:15:00Z"))


@pytest.mark.parametrize("name", ["panel", "panel_audit", "seed_manifest"])
def test_input_reuse_refuses_tampered_source_before_model_or_api(tmp_path, monkeypatch, name):
    from test_prospective_trial_resume import _old_capture
    case = _old_capture(tmp_path, monkeypatch)
    Path(case.original[f"{name}_path"]).write_bytes(b"source differs from captured checksum")
    with pytest.raises(trial.ProspectiveTrialError, match="Empreinte"):
        retrospective._find_existing_panel(case.config, {"schema": case.schema}, "2026-09-08", ["DE", "NL"])
    assert case.calls == []
