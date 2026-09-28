"""Synthetic wiring checks for the disabled prospective annual CPU consumer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_annual_cpu_live as live
from chronos2_hourly.nyx_annual_live_preflight import delivery_grid
import run_nyx_annual_cpu_live as cli


def chronos_pin(weight_hash="b"*64):
    from chronos2_hourly.nyx_annual_cpu_baseline import chronos_identity_digest
    from chronos2_hourly.nyx_annual_nyx_quantiles_gate import CHRONOS_REVISION
    model = {"model_id": "amazon/chronos-2", "revision": CHRONOS_REVISION,
        "device": "cpu", "dtype": "torch.float32",
        "files": {"config.json": "a"*64, "model.safetensors": weight_hash}}
    return {"chronos_model_identity": model, "chronos_model_sha256": chronos_identity_digest(model)}


def test_current_manifest_and_missing_bundle_block_without_writes(tmp_path):
    output = tmp_path / "new_run"
    report = live.preflight(tmp_path / "missing_bundle", "2026-09-29", output)
    assert report["ready"] is False
    assert any("forecast_enabled:false" in message for message in report["blockers"])
    assert any("bundle" in message.lower() for message in report["blockers"])
    with pytest.raises(ValueError, match="forecast_enabled:false"):
        cli.run(bundle=tmp_path / "missing_bundle", delivery_day="2026-09-29",
                output=output)
    assert not output.exists()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    nested = live.preflight(bundle, "2026-09-29", bundle / "outputs")
    assert any("outside the immutable input bundle" in message
               for message in nested["blockers"])
    attempts_bundle = tmp_path / ".output.attempts"
    collision = live.preflight(attempts_bundle, "2026-09-29", tmp_path / "output")
    assert any("Interrupted attempts" in message for message in collision["blockers"])


def test_generic_bundle_cannot_bypass_cpu_nyx_baseline_lineage(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "verify_activation", lambda: {"qualification_sha256": "a" * 64})
    monkeypatch.setattr(live, "inspect_bundle", lambda *_: {"input_bundle_valid": True})
    report = live.preflight(tmp_path / "bundle", "2026-09-29", tmp_path / "output")
    assert report["ready"] is False
    assert report["nyx_cpu_baseline_inspection"] is None
    assert any("NYX CPU baseline" in reason for reason in report["blockers"])


def test_legacy_expert_qualification_cannot_enable_full_chain(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    manifest = json.loads(live.MANIFEST.read_text(encoding="utf-8"))
    manifest["forecast_enabled"] = True
    receipt = config / live.QUALIFICATION_RECEIPT.name
    receipt.write_text(json.dumps({"protocol": live.QUALIFICATION_PROTOCOL,
        "qualified": True, "full_input_chain_qualified": True}), encoding="utf-8")
    manifest["cpu_annual_qualification"] = {"path": "config/" + receipt.name,
                                            "sha256": live.sha256(receipt)}
    (config / live.MANIFEST.name).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="legacy expert-only scores"):
        live.verify_activation(tmp_path)


def test_live_bundle_requires_the_same_chronos_weights_as_annual_qualification(tmp_path, monkeypatch):
    from chronos2_hourly import nyx_annual_cpu_full_chain as chain
    monkeypatch.setattr(live, "verify_activation", lambda: chronos_pin())
    monkeypatch.setattr(live, "inspect_bundle", lambda *_: {"input_bundle_valid": True})
    monkeypatch.setattr(live, "validate_nyx_quantiles_source", lambda *_: {"passed": True})
    monkeypatch.setattr(chain, "_source_packet", lambda *_: {"baseline": chronos_pin()})
    assert live.preflight(tmp_path/"bundle", "2026-09-29", tmp_path/"output")["ready"]
    monkeypatch.setattr(chain, "_source_packet", lambda *_: {"baseline": chronos_pin("c"*64)})
    report = live.preflight(tmp_path/"bundle", "2026-09-29", tmp_path/"output")
    assert report["ready"] is False
    assert any("Chronos weights differ" in message for message in report["blockers"])


def test_frozen_compositions_use_exact_reference_and_threshold():
    index = pd.date_range("2026-09-28", periods=3, freq="h", tz="UTC")
    points = {"fr_residual_1000": pd.Series([119., 120., 80.], index=index),
              "cwe_residual_2000": pd.Series([139., 140., 80.], index=index),
              "cwe_absolute_2000": pd.Series([159., 160., 100.], index=index)}
    reference = pd.Series([100., 100., 100.], index=index)
    np.testing.assert_array_equal(live.compose_price("FR", points, reference),
                                  [100., 120., 80.])
    np.testing.assert_array_equal(live.compose_price("BE", points, reference),
                                  [149., 150., 100.])
    np.testing.assert_array_equal(live.compose_price("DE", points, reference),
                                  [149., 150., 100.])
    np.testing.assert_array_equal(live.compose_price("NL", points, reference),
                                  [149., 150., 90.])


def test_bundle_fingerprints_include_all_receipts_and_bound_artifacts(tmp_path):
    for family in live.FAMILIES:
        for zone in live.ZONES:
            path = tmp_path / "features" / family / f"{zone}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"feature")
    for folder, zones in (("baseline", live.ZONES), ("reference", live.COUNTRIES)):
        for zone in zones:
            path = tmp_path / folder / f"{zone}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"input")
    artifact = tmp_path / "source.bin"
    artifact.write_bytes(b"source")
    receipts = tmp_path / "source_receipts"
    receipts.mkdir()
    for group in live.SOURCE_GROUPS:
        (receipts / f"{group}.json").write_text(json.dumps({
            "artifact_sha256": {"source.bin": hashlib.sha256(b"source").hexdigest()}}),
            encoding="utf-8")
    (receipts / "materialization.json").write_text('{"protocol":"test"}', encoding="utf-8")
    first = live.bundle_hashes(tmp_path)
    assert len(first) == 12 + 4 + 4 + len(live.SOURCE_GROUPS) + 2
    assert "source_receipts/materialization.json" in first
    artifact.write_bytes(b"revised")
    second = live.bundle_hashes(tmp_path)
    assert first["source.bin"] != second["source.bin"]
    (receipts / "materialization.json").write_text('{"protocol":"changed"}', encoding="utf-8")
    third = live.bundle_hashes(tmp_path)
    assert second["source_receipts/materialization.json"] != third["source_receipts/materialization.json"]
    feature = tmp_path / "features" / next(iter(live.FAMILIES)) / "FR.parquet"
    feature.write_bytes(b"feature changed")
    fourth = live.bundle_hashes(tmp_path)
    assert third[f"features/{next(iter(live.FAMILIES))}/FR.parquet"] != fourth[
        f"features/{next(iter(live.FAMILIES))}/FR.parquet"]


def test_run_rejects_manifest_changed_during_fit(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    manifest = bundle / "source_receipts/materialization.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"before")
    output = tmp_path / "output"
    monkeypatch.setattr(cli, "preflight", lambda *_: {
        "ready": True, "blockers": [], "activation": {"test": True}})
    monkeypatch.setattr(cli, "bundle_hashes", lambda *_: {
        "source_receipts/materialization.json": hashlib.sha256(manifest.read_bytes()).hexdigest()})
    monkeypatch.setattr(cli, "load_bundle", lambda *_: object())

    def changed_during_fit(*_args, **_kwargs):
        manifest.write_bytes(b"after")
        return {}, {}

    monkeypatch.setattr(cli, "execute_models", changed_during_fit)
    with pytest.raises(ValueError, match="changed during CPU retraining"):
        cli.run(bundle=bundle, delivery_day="2026-09-29", output=output)
    assert not (output / "receipt.json").exists()
    assert not output.exists()
    attempts = list((tmp_path / ".output.attempts").glob("*/status.json"))
    assert len(attempts) == 1
    assert json.loads(attempts[0].read_text(encoding="utf-8"))["status"] == "FAILED"


def test_live_interruption_retry_preserves_attempts_and_atomically_publishes(tmp_path, monkeypatch):
    day, output, bundle = "2026-10-25", tmp_path / "output", tmp_path / "bundle"
    bundle.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "MANIFEST", manifest)
    activation = {"qualification_sha256": "a"*64, "de_price_performance_exception_used": True,
                  **chronos_pin()}
    monkeypatch.setattr(cli, "preflight", lambda *_: {
        "ready": True, "blockers": [], "activation": activation})
    monkeypatch.setattr(cli, "verify_activation", lambda: activation)
    monkeypatch.setattr(cli, "bundle_hashes", lambda *_: {"fixture": "a"*64})
    monkeypatch.setattr(cli, "load_bundle", lambda *_: object())
    monkeypatch.setattr(cli, "inspect_bundle", lambda *_: {"input_bundle_valid": True})
    # A former direct-write interrupted run is preserved as well.
    output.mkdir()
    (output / "status.json").write_text(json.dumps({"protocol": cli.PROTOCOL,
        "delivery_day": day, "status": "RUNNING"}), encoding="utf-8")
    (output / "partial.cbm").write_bytes(b"old interrupted model")
    calls = []

    def fit(_data, _day, model_dir, **_kwargs):
        calls.append(model_dir)
        model_dir.mkdir()
        if len(calls) == 1:
            (model_dir / "partial.cbm").write_bytes(b"interrupted model")
            raise RuntimeError("deliberate interruption")
        assert not output.exists()
        index = delivery_grid(day)[1]
        frames = {zone: pd.DataFrame({"price_eur_mwh": np.linspace(-10,100,len(index)),
                   "p_negative": np.linspace(1,0,len(index))}, index=index) for zone in live.COUNTRIES}
        audit = {"price_experts": {}, "negative_countries": {}, "compositions": live.COMPOSITIONS}
        for family in live.PRICE_EXPERTS:
            path = model_dir / f"{family}.cbm"
            path.write_bytes(b"price model")
            audit["price_experts"][family] = {"model": {"path": str(path), "sha256": live.sha256(path)}}
        for zone in live.COUNTRIES:
            path = model_dir / f"negative_{zone}.cbm"
            path.write_bytes(b"negative model")
            audit["negative_countries"][zone] = {"model": {"path": str(path), "sha256": live.sha256(path)}}
        return frames, audit

    monkeypatch.setattr(cli, "execute_models", fit)
    with pytest.raises(RuntimeError, match="deliberate interruption"):
        cli.run(bundle=bundle, delivery_day=day, output=output)
    assert not output.exists()
    receipt = cli.run(bundle=bundle, delivery_day=day, output=output)
    assert receipt["status"] == "COMPLETE" and len(calls) == 2
    attempts = list((tmp_path / ".output.attempts").iterdir())
    assert len(attempts) == 2
    assert any((attempt / "partial.cbm").is_file() for attempt in attempts)
    assert any((attempt / "models" / "partial.cbm").is_file() for attempt in attempts)
    for zone, item in receipt["countries"].items():
        assert item["hours"] == 25
        for extension in ("html", "parquet", "csv"):
            path = Path(item[extension])
            assert path.is_relative_to(output) and live.sha256(path) == item[f"{extension}_sha256"]
        assert ("DE : exception de performance" in Path(item["html"]).read_text(encoding="utf-8")) == (zone == "DE")
    for kind in ("price_experts", "negative_countries"):
        for model in receipt["model_audit"][kind].values():
            assert live.sha256(Path(model["model"]["path"])) == model["model"]["sha256"]
    sealed = live.sha256(output / "receipt.json")
    with pytest.raises(ValueError, match="Output already exists"):
        cli.run(bundle=bundle, delivery_day=day, output=output)
    assert live.sha256(output / "receipt.json") == sealed and len(calls) == 2


@pytest.mark.parametrize("fallback", [False, True])
def test_exact_three_price_and_three_negative_fits_from_supplied_frames(tmp_path,
                                                                        fallback):
    day = "2026-09-29"
    full, current, _ = delivery_grid(day)
    features = {family: {zone: pd.DataFrame({"known": np.ones(len(full))}, index=full)
                         for zone in live.ZONES} for family in live.FAMILIES}
    actual = {zone: pd.Series(np.r_[np.full(len(full) - len(current), 10.),
                                         np.full(len(current), np.nan)], index=full,
                              name="price") for zone in live.ZONES}
    nyx = {zone: pd.Series(np.full(len(full), 50.), index=full, name="nyx")
           for zone in live.ZONES}
    reference = {zone: pd.Series(np.full(len(current), 100.), index=current)
                 for zone in live.COUNTRIES}
    data = live.BundleData(features, actual, nyx, reference, current)
    price_calls, negative_calls = [], []

    def price_fit(inputs, labels, base, *, origin_day, stop_day, config,
                  model_path, thread_count):
        assert set(inputs) == set(labels) == set(base) == set(live.ZONES)
        assert labels["FR"].loc[current].isna().all()
        price_calls.append((config.target_mode, config.iterations, thread_count))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"price CBM")
        number = {1000: 120., 2000: 140. if config.target_mode == "residual" else 160.}[config.iterations]
        points = {zone: pd.DataFrame({"point": np.full(len(current), number)},
                                     index=current) for zone in live.ZONES}
        return points, {"tree_count": config.iterations,
            "training_days": 365,
            "training_labels_before_origin": True, "Storm_used_as_input": False,
            "model": {"saved": True, "sha256": live.sha256(model_path)}}

    class FakeClassifier:
        def save_model(self, filename, format):
            assert format == "cbm"
            Path(filename).write_bytes(b"negative CBM")

    def negative_fit(history_features, observed, future_features, *, zone,
                     origin_day, stop_day, threads):
        assert threads == 2
        assert len(history_features) == len(observed) and len(future_features) == len(current)
        assert np.isfinite(observed.to_numpy(float)).all()
        negative_calls.append(zone)
        probability = pd.DataFrame({"p_negative_raw": np.full(len(current), .6),
            "p_negative": np.full(len(current), .7),
            "is_negative_predicted": np.full(len(current), True)}, index=current)
        blocked = fallback and zone == "FR"
        return SimpleNamespace(estimator=None if blocked else FakeClassifier(),
            probabilities=probability,
            audit={"models_fitted": 0 if blocked else 1,
                   "tree_count": 0 if blocked else 120,
                   "fallback_reason": "one_class" if blocked else None,
                   "forecast_labels_used": False, "storm_used": False})

    if fallback:
        with pytest.raises(ValueError, match="fallback=one_class"):
            live.execute_models(data, day, tmp_path / "models",
                price_fit=price_fit, negative_fit=negative_fit)
        assert negative_calls == ["FR"]
        return
    outputs, audit = live.execute_models(data, day, tmp_path / "models",
        price_fit=price_fit, negative_fit=negative_fit)
    assert price_calls == [("residual", 1000, 8), ("residual", 2000, 8),
                           ("absolute", 2000, 8)]
    assert negative_calls == ["FR", "DE", "BE", "NL"]
    assert len(list((tmp_path / "models").glob("*.cbm"))) == 7
    assert all(len(frame) == len(current) for frame in outputs.values())
    assert outputs["FR"].price_eur_mwh.eq(120.).all()
    assert outputs["BE"].price_eur_mwh.eq(150.).all()
    assert outputs["NL"].price_eur_mwh.eq(150.).all()
    assert audit["negative_countries"]["FR"]["model"]["bytes"] > 0
