"""FullReport orchestration with real publication and no expensive computations.

Numerical and source-provider boundaries are mocked. The public OS-locking
wrapper, preflight checks, observation overlay, manifest/CSV publication and
CLI JSON serialization remain real.
"""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_exogenous import lora_finetune, prospective_trial as trial
from chronos2_exogenous import rolling_research as research
from chronos2_exogenous import rolling_research_auxiliary as auxiliary
from chronos2_exogenous import rolling_research_comparators as comparators
from chronos2_exogenous import rolling_research_reporting as reporting
from chronos2_exogenous.prospective_auxiliary import RAW_MODEL, QUANTILES, MARKET_COLUMNS, ResidualRecipe
from chronos2_hourly.kalman_residual import KalmanResidualConfig
from test_prospective_trial import _raw


END, START, SUPPORT = "2026-09-08", "2025-09-09", "2024-09-09"
MODELS = ("lora16_residual", "lora16_residual_kalman")


def _forbidden(*_args, **_kwargs):
    pytest.fail("No real neural inference, prospective publishing or production write is authorized in this test")


def _snapshot(root):
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def full_case(tmp_path, monkeypatch):
    raw = _raw(SUPPORT, 730)
    dates = research._local_days(raw)
    recent = raw.loc[dates.ge("2025-09-03")].copy()
    prefix = raw.loc[dates.lt("2025-09-03")].copy()
    evaluated = raw.loc[dates.ge(START)].copy().reset_index(drop=True)
    config = {"project_root": str(tmp_path), "output_root": str(tmp_path / "runs/experiments/original_trial"),
              "checkpoint_sha256": "a" * 64}
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True)
    sources = [output_root / "trial_manifest.json", tmp_path / "config/activation.yaml",
               tmp_path / "runs/live/old_day/forecast.html", tmp_path / "runs/exports/old_day/forecast.csv",
               tmp_path / "runs/experiments/original_training/weights.bin",
               tmp_path / "chronos2_exogenous/rolling_research_reporting.py"]
    for number, path in enumerate(sources):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"untouched source {number}".encode())
    manifest = {"recipe": asdict(ResidualRecipe()), "kalman_config": asdict(KalmanResidualConfig()),
                "zones": {zone: {} for zone in ("FR", "DE")}}
    histories = {zone: recent.copy(deep=True) for zone in manifest["zones"]}
    state = {"observation_delta": 10.0, "prefix_complete": True}
    events, fits, prefix_calls, renders = [], [], [], []
    cached_predictions = {}
    monkeypatch.setattr(trial, "verify_trial", lambda _config: manifest)
    monkeypatch.setattr(trial, "now_utc", lambda: pd.Timestamp("2026-09-07T15:15Z"))
    monkeypatch.setattr(lora_finetune, "load_checkpoint", _forbidden)
    def recent_provider(_config, _manifest, zone, end_day):
        assert end_day == END
        events.append(("recent", zone))
        return histories[zone].copy(deep=True), [trial._file(sources[0])]
    monkeypatch.setattr(research, "_recent_sources", recent_provider)
    raw_inputs = evaluated[["delivery_start_utc", *MARKET_COLUMNS]].rename(columns={"delivery_start_utc": "timestamp"})
    def input_provider(_config, _manifest, *, zone, start_day, end_day):
        assert (start_day, end_day) == (START, END)
        events.append(("inputs", zone))
        return raw_inputs.copy(deep=True), [trial._file(sources[0])]
    monkeypatch.setattr(research, "_report_inputs", input_provider)
    def prefix_provider(_config, _manifest, *, zone, start_day, **_kwargs):
        assert start_day == SUPPORT
        events.append(("prefix", zone))
        prefix_calls.append(zone)
        return prefix.copy(deep=True), {"complete": state["prefix_complete"], "expected_days": 359,
            "completed_days": 359 if state["prefix_complete"] else 1,
            "neural_oof": False, "neural_in_sample": True}
    monkeypatch.setattr(research, "reconstruct_prefix", prefix_provider)
    def auxiliary_provider(supplied, *, evaluation_start, end_day, output_directory, workers, identity):
        zone = identity["zone"]
        events.append(("auxiliary", zone))
        assert (evaluation_start, end_day) == (START, END)
        assert pd.DatetimeIndex(supplied.delivery_start_utc).equals(pd.DatetimeIndex(raw.delivery_start_utc))
        assert research._local_days(supplied).nunique() == 730
        assert supplied.actual.eq(52).all()  # Updated reporting labels must never enter calibration.
        assert identity["recipe"] == manifest["recipe"] and identity["kalman_config"] == manifest["kalman_config"]
        destination = Path(output_directory)
        if zone not in cached_predictions:
            fits.append(supplied.copy(deep=True))
            predictions = supplied.loc[research._local_days(supplied).ge(START)].copy().reset_index(drop=True)
            for model, shift in zip(MODELS, (1.0, 2.0)):
                for quantile in QUANTILES:
                    predictions[f"{model}__{quantile}"] = predictions[f"{RAW_MODEL}__{quantile}"] + shift
            cached_predictions[zone] = predictions.copy(deep=True)
            destination.mkdir(parents=True)
            trial._write_frame(destination / "stub_predictions.csv.gz", predictions.drop(columns="actual"))
        return cached_predictions[zone].copy(deep=True), {"cached_days": 365 if len(fits) < len([e for e in events if e[0] == "auxiliary"]) else 0,
            "evaluation_days": 365, "used_for_prediction": False}
    monkeypatch.setattr(auxiliary, "run_rolling_research_auxiliary", auxiliary_provider)
    def comparator_provider(root, *, zone, delivery_day, start_day):
        assert Path(root) == tmp_path and (delivery_day, start_day) == (END, START)
        assert ("auxiliary", zone) in events
        events.append(("comparator", zone))
        frame = pd.DataFrame({"timestamp": evaluated.delivery_start_utc,
            "actual": evaluated.actual + state["observation_delta"], "q50": 61.0, "zone": zone})
        return frame, {"storm_contract": {"report_label": "Storm officiel dashboard", "used_for_prediction": False},
                       "network_refreshed": False, "snapshot_only": True}
    monkeypatch.setattr(comparators, "load_rolling_report_comparators", comparator_provider)
    def renderer(predictions, inputs, *, output_directory, zone, delivery_day, metadata,
                 storm, storm_contract, history_target):
        events.append(("renderer", zone))
        assert delivery_day == END
        assert len(predictions) == 8760 and research._local_days(predictions).nunique() == 365
        assert (research._local_days(predictions).iloc[0], research._local_days(predictions).iloc[-1]) == (START, END)
        assert predictions.actual.eq(52.0 + state["observation_delta"]).all()
        assert storm.actual.eq(52.0 + state["observation_delta"]).all()
        evaluation_hours = pd.DatetimeIndex(evaluated.delivery_start_utc)
        assert history_target.loc[evaluation_hours].eq(52.0 + state["observation_delta"]).all()
        prior_hours = history_target.index.difference(evaluation_hours)
        assert history_target.loc[prior_hours].eq(52.0).all()
        assert len(history_target) == len(raw) and len(inputs) == 8760
        assert metadata["evaluation_days"] == metadata["calibration_days"] == 365
        assert metadata["prospective_eligible"] is False and metadata["promotion_eligible"] is False
        assert storm_contract["used_for_prediction"] is False
        renders.append((predictions.copy(deep=True), history_target.copy(deep=True), metadata))
        paths = {model: Path(output_directory) / f"{model}.html" for model in MODELS}
        for model, path in paths.items():
            path.write_text(f"<html>{model} : 365 jours, observation={52 + state['observation_delta']}</html>", encoding="utf-8")
        return paths  # Real Path objects must be converted for the normal FullReport CLI.
    monkeypatch.setattr(reporting, "render_rolling_research_reports", renderer)
    return SimpleNamespace(config=config, manifest=manifest, root=tmp_path, raw=raw, prefix=prefix,
        recent=histories, evaluated=evaluated, events=events, fits=fits, prefix_calls=prefix_calls,
        renders=renders, cached_predictions=cached_predictions, state=state, source_snapshot=_snapshot(tmp_path))


def _run(case, **kwargs):
    return research.run_rolling_report(case.config, delivery_day=END, zones=["FR"], **kwargs)


def _assert_sources_unchanged(case):
    for path, value in case.source_snapshot.items():
        assert path.read_bytes() == value
    added = set(_snapshot(case.root)).difference(case.source_snapshot)
    allowed = case.root / "runs/experiments" / research.OUTPUT_NAME
    assert all(path.is_relative_to(allowed) for path in added)


def test_full_workflow_publishes_exact365_without_using_fresh_observations_for_calibration(full_case):
    case = full_case
    result = _run(case)
    assert result["status"] == "rolling_research_reports_completed"
    assert (result["support_start"], result["evaluation_start"], result["evaluation_end"]) == (SUPPORT, START, END)
    assert len(case.fits) == len(case.renders) == 1
    assert case.fits[0].actual.eq(52).all() and case.raw.actual.eq(52).all()
    assert case.recent["FR"].actual.eq(52).all() and case.prefix.actual.eq(52).all()
    assert case.cached_predictions["FR"].actual.eq(52).all()
    assert [event[0] for event in case.events] == ["recent", "inputs", "prefix", "auxiliary", "comparator", "renderer"]
    zone_result = result["zones"]["FR"]
    assert set(zone_result["reports"]) == set(MODELS)
    assert all(isinstance(path, str) and Path(path).is_file() for path in zone_result["reports"].values())
    json.dumps(result, allow_nan=False)
    report_dir = Path(zone_result["report_directory"])
    manifest = trial._json(report_dir / "report_manifest.json")
    assert all(trial._verify_file(entry).is_file() for entry in manifest["reports"].values())
    frame = trial._read_frame(trial._verify_file(manifest["evidence"]))
    assert frame.actual.eq(62).all() and len(frame) == 8760
    counts = frame.groupby(research._local_days(frame)).size()
    assert counts["2025-10-26"] == 25 and counts["2026-03-29"] == 23
    _assert_sources_unchanged(case)


def test_fullreport_cli_serializes_actual_workflow_paths_and_does_not_enter_prospective_runner(full_case, monkeypatch, capsys):
    import run_lora_rank16_trial as cli
    case = full_case
    monkeypatch.setattr(cli, "load_trial_config", lambda _path: case.config)
    monkeypatch.setattr(cli, "execute_trial", _forbidden)
    monkeypatch.setattr(cli, "resolve_and_report", _forbidden)
    monkeypatch.setattr(sys, "argv", ["runner", "--action", "fullreport", "--delivery-day", END, "--zones", "FR"])
    assert cli.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "rolling_research_reports_completed"
    assert all(Path(path).is_file() for path in result["zones"]["FR"]["reports"].values())
    assert len(case.fits) == 1
    _assert_sources_unchanged(case)


def test_rerun_reuses_auxiliary_outputs_while_publishing_fresh_report_only_labels(full_case):
    case = full_case
    first = _run(case)
    old_dir = Path(first["zones"]["FR"]["report_directory"])
    old_reports = _snapshot(old_dir)
    auxiliary_root = Path(first["output_directory"]) / "FR/auxiliary"
    frozen_aux = _snapshot(auxiliary_root)
    case.state["observation_delta"] = 20.0
    second = _run(case)
    assert len(case.fits) == 1  # Numerical work was supplied by the mock immutable cache.
    assert len(case.renders) == 2 and _snapshot(auxiliary_root) == frozen_aux
    assert _snapshot(old_dir) == old_reports
    assert first["zones"]["FR"]["report_directory"] != second["zones"]["FR"]["report_directory"]
    assert case.cached_predictions["FR"].actual.eq(52).all()
    assert case.renders[0][0].actual.eq(62).all() and case.renders[1][0].actual.eq(72).all()
    pd.testing.assert_frame_equal(case.renders[0][0].drop(columns="actual"), case.renders[1][0].drop(columns="actual"))
    _assert_sources_unchanged(case)


@pytest.mark.parametrize("failure", ["day_gap", "hour_gap", "historical_nan"])
def test_preflight_problem_in_second_country_prevents_any_fit_in_first_country(full_case, failure):
    case = full_case
    frame = case.recent["DE"]
    if failure == "day_gap":
        frame = frame.loc[research._local_days(frame).ne("2026-09-04")].copy()
    elif failure == "hour_gap":
        frame = frame.drop(frame.index[100])
    else:
        frame = frame.copy()
        frame.loc[frame.index[100], "actual"] = np.nan
    case.recent["DE"] = frame
    with pytest.raises(ValueError):
        research.run_rolling_report(case.config, delivery_day=END, zones=["FR", "DE"])
    assert case.prefix_calls == [] and case.fits == [] and case.renders == []
    assert not any(event[0] in {"comparator", "auxiliary"} for event in case.events)
    _assert_sources_unchanged(case)
    # Failure releases the actual OS lock; a stale file is not treated as a running job.
    with research._compute_lock(case.root / "runs/experiments" / research.OUTPUT_NAME):
        pass


def test_incomplete_prefix_returns_stage_status_without_auxiliary_or_report(full_case):
    case = full_case
    case.state["prefix_complete"] = False
    result = _run(case, max_new_prefix_days=1)
    assert result["status"] == "prefix_stage_completed"
    assert result["zones"]["FR"]["complete"] is False
    assert case.fits == [] and case.renders == []
    assert not any(event[0] == "comparator" for event in case.events)
    _assert_sources_unchanged(case)


def test_plan_checks_recent_data_without_locking_or_fitting(full_case):
    case = full_case
    result = _run(case, stage="plan")
    assert result["status"] == "ready_for_research_replay"
    assert case.prefix_calls == [] and case.fits == []
    assert _snapshot(case.root) == case.source_snapshot


@pytest.mark.parametrize(("field", "parameter", "value"), [
    ("recipe", "ridge_alpha", 99.0), ("kalman_config", "q_over_r", 0.3),
])
def test_declared_auxiliary_parameters_cannot_silently_differ_from_executed_defaults(field, parameter, value):
    identity = {"recipe": asdict(ResidualRecipe()), "kalman_config": asdict(KalmanResidualConfig())}
    identity[field][parameter] = value
    with pytest.raises(auxiliary.RollingResearchAuxiliaryError, match="parametres standard figes"):
        auxiliary._contract(timezone="Europe/Paris", identity=identity)
