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


def test_generic_bundle_cannot_bypass_cpu_nyx_baseline_lineage(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "verify_activation", lambda: {"qualification_sha256": "a" * 64})
    monkeypatch.setattr(live, "inspect_bundle", lambda *_: {"input_bundle_valid": True})
    report = live.preflight(tmp_path / "bundle", "2026-09-29", tmp_path / "output")
    assert report["ready"] is False
    assert report["nyx_cpu_baseline_inspection"] is None
    assert any("NYX CPU baseline" in reason for reason in report["blockers"])


def test_activation_needs_pinned_annual_cpu_score_and_code(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    manifest = json.loads(live.MANIFEST.read_text(encoding="utf-8"))
    manifest["forecast_enabled"] = True
    qualification = {"protocol": live.QUALIFICATION_PROTOCOL, "qualified": True,
        "price_expert_replay_qualified": True,
        "negative_replay_verified": True,
        "full_input_chain_qualified": True,
        "price_experts": list(live.PRICE_EXPERTS),
        "negative_model_protocol": live.NEGATIVE_PROTOCOL,
        "compositions": live.COMPOSITIONS,
        "first_delivery_day": "2025-09-24", "last_delivery_day": "2026-09-23",
        "origins_per_country": 53,
        "price_threads": 8, "negative_threads": 2,
        "runtime_versions": live._runtime_versions(),
        "replay_receipts_sha256": {"price": "a" * 64, "negative": "b" * 64},
        "price_country_metrics": {zone: {"hours": 8759,
            "storm_common_hours": 8759, "rmse": 10., "storm_rmse": 11.,
            "strict_win_rate": .51} for zone in live.COUNTRIES},
        "negative_country_metrics": {zone: {"hours": 8760, "brier": .02}
            for zone in live.COUNTRIES},
        "code_sha256": {}}
    for name in live.QUALIFICATION_CODE:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
        qualification["code_sha256"][name] = hashlib.sha256(name.encode()).hexdigest()
    receipt = config / live.QUALIFICATION_RECEIPT.name
    payload = json.dumps(qualification).encode()
    receipt.write_bytes(payload)
    manifest["cpu_annual_qualification"] = {
        "path": "config/nyx_annual_cpu_qualification_receipt.json",
        "sha256": hashlib.sha256(payload).hexdigest()}
    manifest_path = config / live.MANIFEST.name
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    verified = live.verify_activation(tmp_path)
    assert verified["qualification_sha256"] == manifest["cpu_annual_qualification"]["sha256"]
    qualification["full_input_chain_qualified"] = False
    changed = json.dumps(qualification).encode()
    receipt.write_bytes(changed)
    manifest["cpu_annual_qualification"]["sha256"] = hashlib.sha256(changed).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Annual CPU qualification scope or recipe differs"):
        live.verify_activation(tmp_path)
    qualification["full_input_chain_qualified"] = True
    qualification["price_country_metrics"]["BE"]["rmse"] = 12.
    changed = json.dumps(qualification).encode()
    receipt.write_bytes(changed)
    manifest["cpu_annual_qualification"]["sha256"] = hashlib.sha256(changed).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="BE: CPU price does not qualify"):
        live.verify_activation(tmp_path)
    qualification["price_country_metrics"]["BE"]["rmse"] = 10.
    qualification["negative_country_metrics"]["NL"]["hours"] = 8759
    changed = json.dumps(qualification).encode()
    receipt.write_bytes(changed)
    manifest["cpu_annual_qualification"]["sha256"] = hashlib.sha256(changed).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="NL: annual CPU negative-probability score invalid"):
        live.verify_activation(tmp_path)


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
    assert len(first) == 12 + 4 + 3 + len(live.SOURCE_GROUPS) + 2
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
    assert json.loads((output / "status.json").read_text(encoding="utf-8"))["status"] == "FAILED"


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
    assert negative_calls == ["FR", "BE", "NL"]
    assert len(list((tmp_path / "models").glob("*.cbm"))) == 6
    assert all(len(frame) == len(current) for frame in outputs.values())
    assert outputs["FR"].price_eur_mwh.eq(120.).all()
    assert outputs["BE"].price_eur_mwh.eq(150.).all()
    assert outputs["NL"].price_eur_mwh.eq(150.).all()
    assert audit["negative_countries"]["FR"]["model"]["bytes"] > 0
