"""Runner integration with sealed synthetic sources and mocked inference only.

No network, foundation model, production archive or real forecast is used here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nyx_intrahour.data import HOURLY_ALIASES, ZONES
from nyx_quarterhour import data, inference, runner, sources


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "synthetic_project"
    root.mkdir()
    # These inert files let the runner exercise its real code-identity guard.
    for relative in ("nyx_quarterhour/runner.py", "nyx_quarterhour/data.py",
                     "nyx_intrahour/data.py", "nyx_intrahour/runner.py", "run_nyx_quarterhour.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic provenance fixture; never executed.\n", encoding="utf-8")
    begin = pd.Timestamp("2026-03-10", tz="Europe/Paris")
    end = pd.Timestamp("2026-03-30", tz="Europe/Paris")
    quarters = pd.date_range(begin, end, freq="15min", inclusive="left").tz_convert("UTC")
    native = pd.MultiIndex.from_product([quarters, ZONES], names=["timestamp_utc", "zone"]).to_frame(index=False)
    zone_offset = native.zone.map({zone: n * 3 for n, zone in enumerate(ZONES)})
    native["actual_15m"] = 20 + zone_offset + native.timestamp_utc.dt.hour * .3 + native.timestamp_utc.dt.minute / 15
    source_dir = root / "data/pit/nyx_quarterhour/synthetic_archive"
    source_dir.mkdir(parents=True)
    source_file = source_dir / "native_prices.parquet"
    native.to_parquet(source_file, index=False)
    source_manifest = source_dir / "manifest.json"
    source_manifest.write_text(json.dumps({
        "schema_version": 1, "artifact_type": "nyx_quarterhour_price_observations",
        "timezone": "UTC", "unit": "EUR/MWh", "resolution_minutes": 15,
        "interpolation": "none", "price_vintage": "latest_observations",
        "status": "complete", "provider_revision_timestamp_available": False,
        "sources": sources.SOURCES, "data_file": source_file.name,
        "data_sha256": sha(source_file), "start_day": "2026-03-10", "end_day": "2026-03-29",
        "fixture_only": True,
    }), encoding="utf-8")
    means = native.assign(timestamp_utc=native.timestamp_utc.dt.floor("h")).groupby(
        ["timestamp_utc", "zone"], as_index=False).actual_15m.mean()
    baseline = means.rename(columns={"actual_15m": "native_hourly_actual"})
    baseline["actual"] = baseline.native_hourly_actual + 2  # intentionally different archived label
    baseline["training_actual"] = baseline.actual
    baseline["nyx_q50"] = baseline.native_hourly_actual + 5
    for i, alias in enumerate(HOURLY_ALIASES):
        baseline[f"feature_hourly_{alias}"] = i + baseline.timestamp_utc.dt.hour / 24
    fixture_dir = root / "fixtures"
    fixture_dir.mkdir()
    baseline_file = fixture_dir / "synthetic_baseline.parquet"
    baseline.to_parquet(baseline_file, index=False)
    baseline_provenance = fixture_dir / "baseline_provenance.txt"
    baseline_provenance.write_text("Synthetic labels, not a scientific result.\n", encoding="utf-8")
    baseline_audit = {"delivery_day": "2026-03-30", "fixture_only": True, "identities": [
        {"zone": zone, "delivery_day": "2026-03-30", "files": {str(baseline_file): sha(baseline_file)},
         "observations": {str(baseline_provenance): sha(baseline_provenance)}} for zone in ZONES
    ]}
    cfg = {
        "schema_version": 1, "baseline_delivery_day": "2026-03-30",
        "source_manifest": str(source_manifest.relative_to(root)),
        "output_root": str(runner.NAMESPACE), "model_id": "amazon/chronos-2",
        "context_hours": 4, "torch_threads": 4, "model_batch_size": 16, "seed": 3,
        "start_day": "2026-03-20", "end_day": "2026-03-29",
        "bootstrap_repetitions": 20, "bootstrap_seed": 3,
        "diagnostic_only": True, "activation_performed": False, "postprocessing": "none",
    }
    state = {"calls": [], "loads": 0, "on_infer": None, "checkpoint": {
        "requested_model_id": cfg["model_id"], "checkpoint_path": str(root / "synthetic_checkpoint"),
        "snapshot": "SYNTHETIC_ONLY", "config_sha256": "1" * 64, "weights_sha256": "2" * 64,
        "weights_bytes": 0, "device": "cpu", "dtype": "float32", "local_files_only": True,
        "torch_threads": cfg["torch_threads"], "seed": cfg["seed"], "fixture_only": True,
    }}

    def load_mock(*args, **kwargs):
        state["loads"] += 1
        return object()

    def infer_mock(contexts, futures, *, freq, context_length, prediction_length, **kwargs):
        assert len(contexts) == len(futures) == 4
        assert all("target" not in future for future in futures)
        assert all(len(context) == context_length for context in contexts)
        assert all(len(future) == prediction_length for future in futures)
        state["calls"].append({"freq": freq, "contexts": [f.copy() for f in contexts],
                               "futures": [f.copy() for f in futures], "context_length": context_length})
        if state["on_infer"] is not None:
            action = state["on_infer"]
            state["on_infer"] = None
            action()
        records = []
        for zone, future in zip(ZONES, futures):
            # Closed-form fake output. No access to any realized delivery target.
            times = pd.DatetimeIndex(future.timestamp)
            offsets = times.minute / 15 if freq == "15min" else np.full(len(times), 1.5)
            prediction = 20 + ZONES.index(zone) * 3 + times.hour * .3 + offsets + (0.5 if freq == "15min" else 2)
            records.append(pd.DataFrame({"item_id": future.item_id.to_numpy(), "timestamp": times,
                                         "q50": prediction, "q10": prediction - 2, "q90": prediction + 2}))
        return pd.concat(records, ignore_index=True)

    monkeypatch.setattr(runner, "load_baseline", lambda *_: (baseline.copy(deep=True), json.loads(json.dumps(baseline_audit))))
    monkeypatch.setattr(inference, "load_pipeline", load_mock)
    monkeypatch.setattr(inference, "infer_batch", infer_mock)
    monkeypatch.setattr(inference, "checkpoint_identity", lambda *args, **kwargs: dict(state["checkpoint"]))
    return SimpleNamespace(root=root, config=cfg, native=native, baseline=baseline,
                           source_file=source_file, source_manifest=source_manifest,
                           baseline_file=baseline_file, baseline_provenance=baseline_provenance, state=state)


def assert_failed_without_scores(directory):
    summary = read_json(directory / "summary.json")
    assert summary["status"] == "failed"
    assert summary["results_valid"] is False
    assert summary["decision"]["encouraging"] is False
    assert summary["decision"]["promotion_allowed"] is False
    report = (directory / "report.html").read_text(encoding="utf-8")
    assert "Comparaison non validée" in report
    assert "<h2>Résultats horaires appariés</h2>" not in report
    assert "<h2>Écarts et incertitude</h2>" not in report
    return summary


def test_runner_seals_matched_four_country_forecasts_and_reconciles_native_labels(project):
    before = {p: sha(p) for p in project.root.rglob("*") if p.is_file()}
    directory = runner.run(project.config, root=project.root)
    summary = read_json(directory / "summary.json")
    assert summary["status"] == "complete", summary.get("reason")
    assert summary["activation_performed"] is False and summary["production_modified"] is False
    assert summary["decision"]["matched_control_verified"] is True
    assert summary["label_comparison"]["max_abs_difference_from_archived_hourly"] == pytest.approx(2)
    assert summary["label_comparison"]["scoring_target"] == "arithmetic_mean_of_four_native_quarter_hour_prices"
    assert len(project.state["calls"]) == 20
    assert {c["context_length"] for c in project.state["calls"] if c["freq"] == "h"} == {4}
    assert {c["context_length"] for c in project.state["calls"] if c["freq"] == "15min"} == {16}
    assert {len(c["futures"][0]) for c in project.state["calls"] if c["freq"] == "h"} == {23, 24}
    assert {len(c["futures"][0]) for c in project.state["calls"] if c["freq"] == "15min"} == {92, 96}
    predictions = pd.read_parquet(directory / "predictions.parquet")
    assert set(predictions.family) == {"nyx", "hourly_control", "native_quarterhour"}
    assert predictions.groupby(["family", "zone"]).size().eq(239).all()
    assert not predictions.duplicated(["family", "zone", "timestamp_utc"]).any()
    expected = project.native.assign(timestamp_utc=project.native.timestamp_utc.dt.floor("h")).groupby(
        ["timestamp_utc", "zone"]).actual_15m.mean()
    actual = predictions.loc[predictions.family.eq("nyx")].set_index(["timestamp_utc", "zone"]).actual
    np.testing.assert_allclose(actual, expected.reindex(actual.index))
    lock = read_json(directory / "protocol.lock.json")
    assert project.state["checkpoint"]["weights_sha256"] in json.dumps(lock)
    for manifest in ("inputs.manifest.json", "outputs.manifest.json"):
        for name, digest in read_json(directory / manifest)["files"].items():
            assert sha(directory / name) == digest
    for path, digest in before.items():
        assert sha(path) == digest
    new_files = [p for p in project.root.rglob("*") if p.is_file() and p not in before]
    assert all(p.is_relative_to(project.root / runner.NAMESPACE) for p in new_files)


def test_complete_resume_reuses_only_provenance_checked_daily_predictions(project):
    directory = runner.run(project.config, root=project.root)
    assert read_json(directory / "summary.json")["status"] == "complete"
    calls = len(project.state["calls"])
    forecasts_before = sha(directory / "native_quarterhour_predictions.parquet")
    assert runner.run(project.config, root=project.root, resume=directory) == directory
    assert read_json(directory / "summary.json")["status"] == "complete"
    assert len(project.state["calls"]) == calls
    assert sha(directory / "native_quarterhour_predictions.parquet") == forecasts_before


def test_interrupted_run_resumes_remaining_batches_without_replaying_completed_origins(project, monkeypatch):
    original = inference.infer_batch
    attempts = 0

    def interrupted(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 3:
            raise RuntimeError("Synthetic process interruption before the third forecast")
        return original(*args, **kwargs)

    monkeypatch.setattr(inference, "infer_batch", interrupted)
    directory = runner.run(project.config, root=project.root)
    assert_failed_without_scores(directory)
    completed = {p: sha(p) for p in (directory / "daily").glob("*.parquet")}
    assert len(completed) == 2
    monkeypatch.setattr(inference, "infer_batch", original)
    runner.run(project.config, root=project.root, resume=directory)
    assert read_json(directory / "summary.json")["status"] == "complete"
    assert len(project.state["calls"]) == 20
    assert all(sha(p) == digest for p, digest in completed.items())
    identities = [(c["freq"], str(c["futures"][0].timestamp.iloc[0])) for c in project.state["calls"]]
    assert len(identities) == len(set(identities))


def test_resume_rejects_changed_inference_library_version(project, monkeypatch):
    directory = runner.run(project.config, root=project.root)
    assert read_json(directory / "summary.json")["status"] == "complete"
    calls = len(project.state["calls"])
    original = runner.importlib.metadata.version
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda name:
                        "SYNTHETIC_CHANGED_VERSION" if name == "chronos-forecasting" else original(name))
    runner.run(project.config, root=project.root, resume=directory)
    assert_failed_without_scores(directory)
    assert len(project.state["calls"]) == calls


@pytest.mark.parametrize("target", ["native_prices.parquet", "baseline.parquet", "scoring_baseline.parquet", "inputs.manifest.json"])
def test_resume_rejects_modified_sealed_input_copies_before_reuse(project, target):
    directory = runner.run(project.config, root=project.root)
    assert read_json(directory / "summary.json")["status"] == "complete"
    calls = len(project.state["calls"])
    path = directory / target
    path.write_bytes(path.read_bytes() + b" \n")
    runner.run(project.config, root=project.root, resume=directory)
    assert_failed_without_scores(directory)
    assert len(project.state["calls"]) == calls


@pytest.mark.parametrize("mutation", ["code", "source_manifest", "source_data", "baseline", "checkpoint"])
def test_mutation_during_inference_invalidates_results_and_hides_scores(project, mutation):
    paths = {"code": project.root / "nyx_quarterhour/data.py", "source_manifest": project.source_manifest,
             "source_data": project.source_file, "baseline": project.baseline_provenance}

    def mutate():
        if mutation == "checkpoint":
            project.state["checkpoint"]["weights_sha256"] = "3" * 64
        else:
            path = paths[mutation]
            path.write_bytes(path.read_bytes() + b" \n")

    project.state["on_infer"] = mutate
    directory = runner.run(project.config, root=project.root)
    assert_failed_without_scores(directory)


@pytest.mark.parametrize("mutation", ["checkpoint_weights", "code", "daily_identity", "daily_bytes"])
def test_resume_rejects_changed_checkpoint_or_protocol_provenance(project, mutation):
    directory = runner.run(project.config, root=project.root)
    assert read_json(directory / "summary.json")["status"] == "complete"
    calls = len(project.state["calls"])
    if mutation == "checkpoint_weights":
        project.state["checkpoint"]["weights_sha256"] = "4" * 64
    elif mutation == "code":
        path = project.root / "nyx_quarterhour/data.py"
        path.write_bytes(path.read_bytes() + b"# changed\n")
    elif mutation == "daily_identity":
        path = directory / "daily/2026-03-20_h.json"
        value = read_json(path)
        value["protocol_identity"] = "wrong protocol"
        path.write_text(json.dumps(value), encoding="utf-8")
    else:
        path = directory / "daily/2026-03-20_h.parquet"
        path.write_bytes(path.read_bytes() + b"changed")
    runner.run(project.config, root=project.root, resume=directory)
    assert_failed_without_scores(directory)
    assert len(project.state["calls"]) == calls


def test_output_namespace_is_enforced_before_any_inference(project):
    config = {**project.config, "output_root": "runs/exports/production"}
    with pytest.raises(ValueError, match="namespace"):
        runner.run(config, root=project.root)
    assert not project.state["calls"]


@pytest.mark.parametrize("leaf", ["report.html", "summary.json.tmp", "daily"])
def test_redirected_output_descendants_are_rejected_on_resume(project, monkeypatch, leaf):
    directory = runner.run(project.config, root=project.root)
    assert read_json(directory / "summary.json")["status"] == "complete"
    target = directory / leaf
    if not target.exists():
        target.write_text("Synthetic existing temporary file", encoding="utf-8")
    original = Path.lstat

    def redirected(path, *args, **kwargs):
        if path == target:
            return SimpleNamespace(st_mode=stat.S_IFDIR if path.name == "daily" else stat.S_IFREG,
                                   st_file_attributes=0x400)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", redirected)
    with pytest.raises(ValueError, match="reparse|junction|link"):
        runner.run(project.config, root=project.root, resume=directory)


def test_delivery_targets_never_enter_origin_inputs_and_context_duration_matches(project):
    baseline = project.baseline
    original = data.MatchedInputs(project.native, baseline)
    modified_native = project.native.copy()
    delivery = pd.Timestamp("2026-03-29", tz="Europe/Paris").tz_convert("UTC")
    modified_native.loc[modified_native.timestamp_utc.ge(delivery), "actual_15m"] += 100000
    changed = data.MatchedInputs(modified_native, baseline)
    all_contexts = {}
    for frequency in ("h", "15min"):
        contexts, futures, audits = original.build_origin("2026-03-29", frequency, context_hours=4)
        other_contexts, other_futures, _ = changed.build_origin("2026-03-29", frequency, context_hours=4)
        all_contexts[frequency] = contexts
        for c, f, changed_c, changed_f, audit in zip(contexts, futures, other_contexts, other_futures, audits):
            pd.testing.assert_frame_equal(c, changed_c)
            pd.testing.assert_frame_equal(f, changed_f)
            assert "target" not in f
            assert pd.Timestamp(audit["context_last_utc"]) < delivery
            assert audit["publication_vintage_verified"] is False
            assert audit["forecast_origin_utc"] == "2026-03-28T07:00:00+00:00"
        if frequency == "15min":
            for future in futures:
                known = [f"known_{alias}" for alias in HOURLY_ALIASES]
                assert future.groupby(future.timestamp.dt.floor("h"))[known].nunique().le(1).all().all()
    for h, q in zip(all_contexts["h"], all_contexts["15min"]):
        assert h.timestamp.iloc[0] == q.timestamp.iloc[0]
        assert len(q) == 4 * len(h)
