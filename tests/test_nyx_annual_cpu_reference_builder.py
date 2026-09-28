"""Complete OOF assembly with strict fit/score boundaries, no provider mocks."""
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_annual_cpu_reference_builder as producer


@dataclass
class FittedHGB:
    stop_day_exclusive: str


class FittedTest2:
    audit = {"fixture": True}

    def predict_day(self, features, nyx, *, zone, forecast_issued_at_utc):
        assert len(set(features.index.tz_convert("Europe/Paris").date)) == 1
        return pd.DataFrame({"test2__q50": nyx + 1., "spike_probability": .1}, index=nyx.index)


def test_weekly_anchor_and_worst_case_training_support():
    day = pd.Timestamp("2026-09-29").date()
    origin = producer.week_origin(day)
    assert str(origin) == "2026-09-23"
    first = producer.week_origin(origin - timedelta(days=90)) - timedelta(days=365)
    assert (day - first).days == 462


def test_reference_runs_own_origin_fits_daily_oof_and_four_outputs(tmp_path, monkeypatch):
    day = "2026-09-29"
    last = pd.Timestamp(day).date()
    index = producer._grid(last - timedelta(days=462), last + timedelta(days=1))
    baseline = {z: pd.DataFrame({"nyx__q50": 50.}, index=index) for z in producer.ZONES}
    features = {z: pd.DataFrame({"signal": np.sin(np.arange(len(index)) / 24)}, index=index) for z in producer.ZONES}
    test2 = {z: pd.DataFrame({"own_joint_deficit": .2, "own_residual_stress": .1}, index=index) for z in producer.ZONES}
    fits, snapshot_calls = [], []
    def labels(origin):
        snapshot_calls.append(origin)
        # Keep an intentionally enormous target-day label: it must be removed
        # by the orchestrator before reaching any estimator.
        stop = pd.Timestamp(origin).date()
        return {z: pd.Series(np.where(index.tz_convert("Europe/Paris").date < stop, 48., 1e9), index=index)
                for z in producer.ZONES}
    def hgb(matrix, actual, point, **kwargs):
        origin = pd.Timestamp(kwargs["origin_day"]).date()
        days = matrix.index.tz_convert("Europe/Paris").date
        assert np.isfinite(actual.loc[days < origin]).all()
        assert actual.loc[days >= origin].isna().all()
        assert (days >= origin).sum() in (23, 24, 25)
        fits.append((kwargs["zone"], kwargs["variant"], origin))
        return FittedHGB(kwargs["stop_day"]), None, {"origin": str(origin)}
    def test2_fit(matrix, residual, train_days, **kwargs):
        assert len(set(train_days)) == 365
        assert max(train_days) < pd.Timestamp(kwargs["origin_day"]).date()
        np.testing.assert_array_equal(residual, -2.)
        return FittedTest2()
    monkeypatch.setattr(producer, "_upstream", lambda *a: ("a" * 64, True, False))
    monkeypatch.setattr(producer, "archived_columns", lambda *a: ("signal",))
    monkeypatch.setattr(producer, "fit_hgb_block", hgb)
    monkeypatch.setattr(producer, "fit_test2_origin", test2_fit)
    monkeypatch.setattr(producer, "predict_saved_block", lambda fit, matrix, nyx: pd.DataFrame({"point": nyx + 2.}, index=nyx.index))
    for zone, frame in baseline.items():
        producer.write_frame(tmp_path / f"baseline_history/{zone}.parquet", frame)
    receipt = producer.build_cpu_reference_bundle(tmp_path, day, baselines=baseline,
        base_features=features, augmented_features=features, test2_features=test2,
        target_snapshots_by_day=labels)
    assert receipt["state"] == "COMPLETE"
    assert receipt["provider_publication_timestamp_verified"] is False
    assert len(fits) == 14 * 4 * 3
    assert receipt["producer"]["future_labels_used"] is False
    model_path = tmp_path / "reference_models/2026-09-23/FR_hist_residual_400.joblib"
    identity = json.loads(model_path.with_suffix(".json").read_text())["identity"]
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cached_model = tmp_path.parent / "_cpu_reference_cache" / key / "fitted.joblib"
    assert os.path.samefile(model_path, cached_model)
    for zone in producer.ZONES:
        output = pd.read_parquet(tmp_path / f"reference/{zone}.parquet")
        assert len(output) == 24
        np.testing.assert_allclose(output.reference, 51.75)
        oof = pd.read_parquet(tmp_path / f"reference_oof/{zone}.parquet")
        assert len(set(oof.index.tz_convert("Europe/Paris").date)) == 98
    # No loader is consulted at later origins to build earlier weekly fits.
    assert snapshot_calls[:-1] == sorted(snapshot_calls[:-1])
    from chronos2_hourly import nyx_annual_saturn_source as source
    monkeypatch.setattr(source, "load_target_snapshots", lambda _: labels)
    verification = producer.validate_cpu_reference_source(tmp_path, day)
    assert verification["cpu_model_artifacts_verified"] == 196
    assert verification["prior90_recomputed"] is True
    assert receipt["producer"]["code_config_hash_policy"] == producer.TEXT_HASH_POLICY
    # A transferred bundle still verifies after Git checks out the same source
    # using different newline bytes. Data/model transport hashes remain strict.
    checkout = tmp_path / "another_checkout"
    for name in producer.CODE_FILES:
        code = checkout / name
        code.parent.mkdir(parents=True, exist_ok=True)
        code.write_bytes((producer.ROOT / name).read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    monkeypatch.setattr(producer, "ROOT", checkout)
    assert producer.validate_cpu_reference_source(tmp_path, day) == verification
    # Replacing an OOF policy and updating only its transport hash cannot
    # forge the numerical producer evidence.
    policy_path = tmp_path / "reference_policies/DE.json"
    policy = json.loads(policy_path.read_text())
    policy["baseline_mae"] = -100.
    producer.write_json(policy_path, policy)
    receipt["artifact_sha256"]["reference_policies/DE.json"] = producer.sha256(policy_path)
    producer.write_json(tmp_path / "source_receipts/scarcity_confirmed_pair.json", receipt)
    with pytest.raises(ValueError, match="Prior90 policy differs"):
        producer.validate_cpu_reference_source(tmp_path, day)
