"""Full-chain orchestration with synthetic archives and mocked model fitting.

The real input/schema and checkpoint boundaries run; no foundation model,
CatBoost fit, market download or operational archive is touched.
"""
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from chronos2_hourly.kalman_residual import KalmanResidualConfig
from chronos2_hourly.nuclear_forecast import nuclear_kalman_covariate_config
from nyx_fullquarterhour import kalman, raw, residual, runner, storage
from nyx_intrahour.data import HOURLY_ALIASES, ZONES
from nyx_quarterhour.data import day_index
from nyx_quarterhour.sources import SOURCES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "synthetic_project"
    root.mkdir()
    for name in (
        "nyx_fullquarterhour/raw.py", "nyx_fullquarterhour/runner.py", "nyx_fullquarterhour/data.py",
        "nyx_fullquarterhour/inference.py", "nyx_fullquarterhour/storage.py", "nyx_fullquarterhour/kalman.py",
        "nyx_quarterhour/data.py", "nyx_quarterhour/inference.py", "nyx_quarterhour/sources.py",
        "nyx_quarterhour/evaluation.py", "nyx_intrahour/data.py", "nyx_intrahour/runner.py",
        "chronos2_hourly/models/residual_corrector.py", "chronos2_hourly/features.py",
        "chronos2_modular/exogenous_extensions.py", "chronos2_modular/common.py", "chronos2_hourly/kalman_residual.py",
        "chronos2_hourly/kalman_covariates.py", "chronos2_hourly/nuclear_forecast.py",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic provenance fixture; never executed.\n", encoding="utf-8")
    index = pd.date_range("2026-01-30", "2026-03-13", tz="Europe/Paris", freq="15min", inclusive="left").tz_convert("UTC")
    prices = pd.MultiIndex.from_product([index, ZONES], names=["timestamp_utc", "zone"]).to_frame(index=False)
    prices["actual_15m"] = 20 + prices.zone.map({z: i*3 for i, z in enumerate(ZONES)}) + prices.timestamp_utc.dt.hour*.3 + prices.timestamp_utc.dt.minute/15
    baseline = prices.assign(timestamp_utc=prices.timestamp_utc.dt.floor("h")).groupby(["timestamp_utc", "zone"], as_index=False).actual_15m.mean().rename(columns={"actual_15m": "actual"})
    baseline["training_actual"] = baseline.actual
    baseline["nyx_q50"] = baseline.actual + 5
    for i, alias in enumerate(HOURLY_ALIASES):
        baseline[f"feature_hourly_{alias}"] = 10+i+baseline.timestamp_utc.dt.hour*.05
    source = root / "fixtures"
    source.mkdir()
    prices.to_parquet(source / "native_prices.parquet", index=False)
    baseline.to_parquet(source / "baseline.parquet", index=False)
    write(source / "manifest.json", {"schema_version": 1, "artifact_type": "nyx_quarterhour_price_observations",
        "timezone": "UTC", "unit": "EUR/MWh", "resolution_minutes": 15, "interpolation": "none",
        "price_vintage": "latest_observations", "status": "complete", "provider_revision_timestamp_available": False,
        "sources": SOURCES, "data_file": "native_prices.parquet", "data_sha256": sha(source / "native_prices.parquet"),
        "start_day": "2026-01-30", "end_day": "2026-03-12", "fixture_only": True})
    config = {"schema_version": 1, "baseline_delivery_day": "2026-03-13", "source_manifest": "fixtures/manifest.json",
        "output_root": storage.NAMESPACE.as_posix(), "model_id": "amazon/chronos-2", "context_hours": 4,
        "torch_threads": 4, "model_batch_size": 16, "seed": 7, "raw_start_day": "2026-02-01",
        "start_day": "2026-03-03", "end_day": "2026-03-12", "training_lookback_days": 365,
        "minimum_training_days": 30, "worker_count": 2, "catboost_threads": 1, "bootstrap_repetitions": 20,
        "bootstrap_seed": 7, "diagnostic_only": True, "activation_performed": False,
        "postprocessing": "daily_catboost_then_daily_refitted_governed_kalman"}
    recipes = {}
    for i, zone in enumerate(ZONES):
        covariates = nuclear_kalman_covariate_config().to_dict()
        covariates["groups"]["market"] = list(reversed(covariates["groups"]["market"]))
        recipes[zone] = {"residual_recipe": {"enabled": True, "backend": "catboost", "base_model": "chronos2",
            "iterations": 17+i, "depth": 3+i, "learning_rate": .07, "min_training_rows": 720,
            "correction_scale": .5, "max_abs_correction": 31+i, "thread_count": 7,
            "feature_builder": {"timezone": "Europe/Paris", "rich_calendar_primary_country": zone,
                                "exclude_historical_prices": True, "exclude_day_of_year": True,
                                "include_daily_profiles": True, "include_rich_calendar": True}},
            "kalman_filter_parameters": {**asdict(KalmanResidualConfig()), "shift_clip_eur_mwh": 7.25+i},
            "kalman_covariate_config": covariates}
        path = root / "runs/experiments/nuclear_forecast_v1/2026-03-13" / zone.lower() / "civil_pit_v2/report_only/frozen_result/audits.json"
        path.parent.mkdir(parents=True)
        write(path, {"result": recipes[zone]})
    audit = {"fixture_only": True, "identities": [{"zone": z, "files": {str(source / "baseline.parquet"): sha(source / "baseline.parquet")}, "observations": {}} for z in ZONES]}
    state = {"raw": [], "fit": [], "kalman": [], "fail_worker": False, "on_kalman": None,
             "checkpoint": {"weights_sha256": "1"*64, "config_sha256": "2"*64, "fixture_only": True}}
    monkeypatch.setattr(raw, "load_baseline", lambda *_: (baseline.copy(), deepcopy(audit)))
    monkeypatch.setattr(raw, "checkpoint_identity", lambda *_: dict(state["checkpoint"]))
    monkeypatch.setattr(raw, "load_pipeline", lambda *args, **kwargs: object())
    monkeypatch.setattr(raw, "versions", lambda: {"synthetic": "1"})
    monkeypatch.setattr(runner, "versions", lambda: {"synthetic": "1"})

    def infer(contexts, futures, *, freq, context_length, prediction_length, **kwargs):
        assert len(contexts) == len(futures) == 4
        assert all(len(c) == context_length for c in contexts)
        assert all("target" not in f for f in futures)
        assert all(c.timestamp.max() < f.timestamp.min() for c, f in zip(contexts, futures))
        state["raw"].append((str(futures[0].timestamp.min()), freq))
        return pd.concat([pd.DataFrame({"item_id": f.item_id, "timestamp": f.timestamp,
            "q10": 31+f.timestamp.dt.hour*.1, "q50": 41+f.timestamp.dt.hour*.1+f.timestamp.dt.minute*.01,
            "q90": 61+f.timestamp.dt.hour*.1}) for f in futures], ignore_index=True)

    def fit(meta_train, actual_train, raw_train, meta_day, raw_day, **kwargs):
        assert actual_train.index.equals(meta_train.index) and meta_train.index.equals(raw_train.index)
        assert actual_train.index.max() < meta_day.index.min()
        assert not {"actual", "target", "nyx_q50"}.intersection(meta_day)
        assert meta_day.index.equals(raw_day.index)
        state["fit"].append({"last_label": actual_train.index.max(), "target": meta_day.index.min(), **deepcopy(kwargs)})
        target = meta_day.index.min().tz_convert(kwargs["timezone"]).date()
        predicted = raw_day+.5
        predicted.attrs["residual_correction"] = pd.Series(.5, index=predicted.index)
        return predicted, {"delivery_day": str(target), "fit_end_day": str(actual_train.index.max().tz_convert(kwargs["timezone"]).date()),
                            "generation_source": "synthetic_mock_fit", "target_observations_used": False}

    def forecast(prior, future, **kwargs):
        assert "actual" not in future or future.actual.isna().all()
        assert prior.index.max() < future.index.min()
        assert np.isfinite(prior.actual).all()
        state["kalman"].append({"last_label": prior.index.max(), "target": future.index.min(), **deepcopy(kwargs)})
        if state["on_kalman"] is not None:
            action, state["on_kalman"] = state["on_kalman"], None
            action()
        if state["fail_worker"]:
            raise RuntimeError("SYNTHETIC_KALMAN_FAILURE")
        frame = pd.DataFrame({f"residual_kalman__{q}": future[f"residual_corrected__{q}"]-.25 for q in ("q10", "q50", "q90")})
        return SimpleNamespace(predictions=frame, audit={"target_observations_assimilated": 0, "fixture_only": True},
            candidate_predictions=pd.DataFrame({"linear_bias__q50": frame.residual_kalman__q50}, index=frame.index),
            state_audit=pd.DataFrame({"target_observations_assimilated": [0]}))

    monkeypatch.setattr(raw, "infer_batch", infer)
    monkeypatch.setattr(residual, "fit_predict_day", fit)
    monkeypatch.setattr(kalman, "forecast_day", forecast)
    return SimpleNamespace(root=root, config=config, prices=prices, baseline=baseline, recipes=recipes, state=state)


@pytest.fixture
def sealed(project):
    return raw.run_raw(project.config, root=project.root)


def worker_protocol(project, directory):
    protocol = {"config": project.config, "recipes": read(directory / "raw_protocol.json")["protocol"]["recipes"]}
    protocol_id = storage.identity(protocol)
    write(directory / "full_protocol.json", {"identity": protocol_id, "protocol": protocol})
    return protocol_id


class InlinePool:
    """Expose genuine Future failure semantics without child models/processes."""
    def __init__(self, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def submit(self, fn, *args):
        result = Future()
        try:
            result.set_result(fn(*args))
        except BaseException as exc:
            result.set_exception(exc)
        return result


def test_raw_quantiles_use_real_input_contract_and_completed_resume_is_immutable(project, sealed):
    assert len(project.state["raw"]) == 80  # 40 synthetic delivery days at two cadences
    manifest = sealed / "raw_outputs.manifest.json"
    before = sha(manifest)
    assert read(manifest)["status"] == "complete"
    for frequency, per_zone in (("h", 960), ("15min", 3840)):
        frame = pd.read_parquet(sealed / f"raw_{frequency}.parquet")
        assert frame.groupby("zone").size().eq(per_zone).all()
        assert set(frame.zone) == set(ZONES)
        assert np.isfinite(frame[["q10", "q50", "q90"]]).all().all()
    raw.run_raw(project.config, root=project.root, resume=sealed)
    assert len(project.state["raw"]) == 80
    assert sha(manifest) == before


@pytest.mark.parametrize("frequency", ["h", "15min"])
def test_worker_keeps_labels_before_delivery_and_passes_complete_archived_recipe(project, sealed, frequency):
    protocol_id = worker_protocol(project, sealed)
    result = runner._worker("FR", frequency, str(sealed), str(project.root), protocol_id)
    assert len(project.state["fit"]) == 39
    assert len(project.state["kalman"]) == 10
    for record in project.state["fit"]:
        expected = {**project.recipes["FR"]["residual_recipe"], "thread_count": 1}
        assert record["recipe"] == expected
        assert record["minimum_training_days"] == 30 and record["max_lookback_days"] == 365
        assert record["frequency"] == frequency
    for record in project.state["kalman"]:
        assert record["config"].shift_clip_eur_mwh == project.recipes["FR"]["kalman_filter_parameters"]["shift_clip_eur_mwh"]
        assert tuple(record["config"].candidate_kinds) == tuple(KalmanResidualConfig().candidate_kinds)
        assert record["covariate_config"].to_dict() == project.recipes["FR"]["kalman_covariate_config"]
    work = Path(result["directory"])
    raw.verify_files(work, read(work / "manifest.json")["files"])
    assert len(pd.read_parquet(work / "kalman.parquet")) == (960 if frequency == "15min" else 240)
    calls = (len(project.state["fit"]), len(project.state["kalman"]))
    runner._worker("FR", frequency, str(sealed), str(project.root), protocol_id)
    assert calls == (len(project.state["fit"]), len(project.state["kalman"]))


@pytest.mark.parametrize("leaf", ["residual_daily/2026-03-03.parquet", "kalman_daily/2026-03-03.parquet",
                                  "kalman_daily/2026-03-03.candidates.parquet", "residual_daily/2026-03-03.json"])
def test_worker_resume_rejects_corrupt_checkpoint_including_candidates(project, sealed, leaf):
    protocol_id = worker_protocol(project, sealed)
    result = runner._worker("FR", "h", str(sealed), str(project.root), protocol_id)
    path = Path(result["directory"]) / leaf
    if path.suffix == ".json":
        record = read(path)
        record["audit"]["fit_end_day"] = "2027-01-01"
        write(path, record)
    else:
        path.write_bytes(path.read_bytes()+b"SYNTHETIC_MUTATION")
    with pytest.raises(ValueError, match="checksum|changed|checkpoint|identity|candidate"):
        runner._worker("FR", "h", str(sealed), str(project.root), protocol_id)


@pytest.mark.parametrize("mutation", ["checkpoint", "input_copy", "code", "archived_recipe"])
def test_raw_resume_rejects_provenance_mutations_without_inference(project, sealed, mutation):
    if mutation == "checkpoint":
        project.state["checkpoint"]["weights_sha256"] = "3"*64
    else:
        path = {"input_copy": sealed / "baseline.parquet", "code": project.root / "nyx_fullquarterhour/data.py",
                "archived_recipe": project.root / "runs/experiments/nuclear_forecast_v1/2026-03-13/fr/civil_pit_v2/report_only/frozen_result/audits.json"}[mutation]
        path.write_bytes(path.read_bytes()+b" \n")
    with pytest.raises(ValueError):
        raw.run_raw(project.config, root=project.root, resume=sealed)
    assert len(project.state["raw"]) == 80


def test_all_stage_resume_preserves_complete_raw_and_full_protocols(project, sealed, monkeypatch):
    monkeypatch.setattr(runner, "ProcessPoolExecutor", InlinePool)
    runner.run_postprocessing(project.config, root=project.root, directory=sealed)
    summary = read(sealed / "summary.json")
    assert summary["status"] == "complete", summary.get("reason")
    frame = pd.read_parquet(sealed / "predictions.parquet")
    assert set(frame.family) == {"nyx", "raw_hourly", "raw_quarterhour", "residual_hourly", "residual_quarterhour", "matched_full_hourly", "full_quarterhour"}
    assert frame.groupby(["family", "zone"]).size().eq(240).all()
    before = (sha(sealed / "raw_outputs.manifest.json"), sha(sealed / "full_protocol.json"))
    calls = tuple(len(project.state[k]) for k in ("raw", "fit", "kalman"))
    runner.run(project.config, root=project.root, resume=sealed, stage="all")
    assert read(sealed / "summary.json")["status"] == "complete"
    assert before == (sha(sealed / "raw_outputs.manifest.json"), sha(sealed / "full_protocol.json"))
    assert calls == tuple(len(project.state[k]) for k in ("raw", "fit", "kalman"))


def test_worker_failure_propagates_to_failed_result_and_no_scored_report(project, sealed, monkeypatch):
    monkeypatch.setattr(runner, "ProcessPoolExecutor", InlinePool)
    project.state["fail_worker"] = True
    runner.run_postprocessing(project.config, root=project.root, directory=sealed)
    summary = read(sealed / "summary.json")
    assert summary["status"] == "failed" and summary["results_valid"] is False
    assert "SYNTHETIC_KALMAN_FAILURE" in summary["reason"]
    assert summary["decision"]["encouraging"] is False
    assert not (sealed / "metrics.parquet").exists()
    assert not (sealed / "predictions.parquet").exists()
    report = (sealed / "report.html").read_text(encoding="utf-8")
    assert "MAE" not in report or "<table" not in report


def test_input_copy_changed_during_workers_invalidates_completion(project, sealed, monkeypatch):
    monkeypatch.setattr(runner, "ProcessPoolExecutor", InlinePool)

    def mutate_scoring():
        path = sealed / "scoring_baseline.parquet"
        changed = pd.read_parquet(path)
        changed["actual"] += 1
        changed.to_parquet(path, index=False)

    project.state["on_kalman"] = mutate_scoring
    runner.run_postprocessing(project.config, root=project.root, directory=sealed)
    summary = read(sealed / "summary.json")
    assert summary["status"] == "failed"
    assert summary["results_valid"] is False
    assert summary["decision"]["encouraging"] is False
    assert not (sealed / "metrics.parquet").exists()


@pytest.mark.parametrize("key,value", [("activation_performed", True), ("training_lookback_days", 180),
                                      ("minimum_training_days", 10), ("output_root", "runs/live"),
                                      ("postprocessing", "none"), ("worker_count", 0)])
def test_config_rejects_relaxed_architecture_and_publication(project, key, value):
    config = {**project.config, key: value}
    path = project.root / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        raw.load_config(path)
    assert not project.state["raw"]
