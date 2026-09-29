"""Exercise transitive baseline cache inputs through the real orchestrator.

Small numerical forecasters and shorter lookbacks keep the fixture cheap. The
production cache lookup, payload verification, state chaining and reuse run
unchanged; no assertion relies on reimplementing the state hash function.
"""
from datetime import timedelta
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import nyx_annual_cpu_baseline as producer
from chronos2_hourly import nyx_annual_saturn_source as saturn


class _TwoDaysComplete(Exception):
    pass


def test_first_cold_start_has_no_current_day_training_row():
    empty = producer._grid("2026-01-01", "2026-01-01")
    assert len(empty) == 0 and str(empty.tz) == "UTC"
    with pytest.raises(ValueError, match="reversed"):
        producer._grid("2026-01-02", "2026-01-01")


@pytest.fixture
def replay(tmp_path, monkeypatch):
    first = pd.Timestamp("2026-01-01").date()
    second = first + timedelta(days=1)
    cache = tmp_path / "cache"
    calls, contracts = [], {}

    # The warmup must still precede a Kalman score, but two inner days suffice
    # to test a dependency crossing a persisted checkpoint boundary.
    def short_window(*, days):
        return timedelta(days={365: 1, 469: 2}.get(days, days))

    monkeypatch.setattr(producer, "timedelta", short_window)
    monkeypatch.setattr(producer, "_sources", lambda *_:
                        ({"saturn": "a" * 64, "auction_prices": "b" * 64}, False, False))
    monkeypatch.setattr(saturn, "target_history_contract", lambda bundle: contracts[bundle.name])
    monkeypatch.setattr(producer, "_configuration", lambda *_: ({}, {}))
    monkeypatch.setattr(producer, "build_pair_interaction", lambda *_: (None, {}))

    def chronos(*, target, covariates, zone, delivery_day, pipeline):
        calls.append(("chronos", zone, delivery_day))
        plan = producer.build_delivery_plan(delivery_day, timezone="Europe/Paris")
        assert target.index.max() < plan.delivery_start_utc
        return pd.DataFrame({"q10": 9., "q50": 10., "q90": 11.,
                             "forecast_origin_utc": plan.forecast_origin_utc},
                            index=plan.delivery_index_utc)

    def residual(*, raw_history, raw_future, covariates, zone, delivery_day,
                 configuration, interaction):
        calls.append(("residual", zone, delivery_day))
        frame = raw_future.copy()
        for quantile in producer.QUANTILES:
            frame[f"residual_corrected__{quantile}"] = frame[quantile]
        return frame, {"future_labels_used": False}

    def kalman(*, history, future, covariates, zone, delivery_day, configuration):
        calls.append(("kalman", zone, delivery_day))
        # Make the numerical forecast observably depend on the upstream
        # corrected history, exactly the dependency a shallow key misses.
        center = history.residual_corrected__q50.mean()
        return pd.DataFrame({"nyx__q10": center - 1., "nyx__q50": center,
                             "nyx__q90": center + 1.}, index=future.index), {
            "training_window_days": 365, "target_observations_assimilated": 0}

    monkeypatch.setattr(producer, "predict_chronos_day", chronos)
    monkeypatch.setattr(producer, "predict_residual_day", residual)
    monkeypatch.setattr(producer, "predict_kalman_day", kalman)
    pipeline = SimpleNamespace(_nyx_annual_cpu_identity={
        "model_id": "amazon/chronos-2", "revision": producer.CHRONOS_REVISION,
        "device": "cpu", "dtype": "torch.float32",
        "files": {"config.json": "a" * 64, "model.safetensors": "b" * 64}})

    def run(name, outer_day="2026-01-04"):
        bundle = tmp_path / name
        day = pd.Timestamp(outer_day).date()
        index = producer._grid(first, day + timedelta(days=1))
        covariates = pd.DataFrame(1., index=index, columns=producer.RAW_ALIASES)
        producer.write_frame(bundle / "source_artifacts/saturn/covariates.parquet", covariates)
        contracts[name] = {
            "target_history_policy": saturn.TARGET_HISTORY_POLICY,
            "target_revision_utc": saturn.cutoff(outer_day).isoformat(),
            "target_origin_snapshot_verified": False,
            "target_future_labels_used": False}

        def snapshots(inner_day):
            stop = pd.Timestamp(inner_day, tz="Europe/Paris").tz_convert("UTC")
            # Preserve the actual 479-day target window. It is deliberately
            # identical in the two attempts at each given inner day.
            past = pd.date_range(end=stop - pd.Timedelta(hours=1),
                                 periods=479 * 24 + 2, freq="h", tz="UTC")
            return {zone: pd.Series(30., index=past, name="target") for zone in producer.ZONES}

        def stop_after_two(progress):
            if progress["completed_days"] == 2:
                raise _TwoDaysComplete

        with pytest.raises(_TwoDaysComplete):
            producer.build_cpu_baseline_bundle(bundle, outer_day,
                targets=snapshots(outer_day), covariates=covariates,
                target_snapshots_by_day=snapshots, pipeline=pipeline, threads=1,
                require_verified_sources=False, checkpoint_cache=cache,
                progress=stop_after_two)
        return bundle

    def checkpoint(bundle, day, zone="FR"):
        folder = bundle / "baseline_checkpoints" / zone / str(day)
        return folder, json.loads((folder / "receipt.json").read_text())

    return SimpleNamespace(run=run, checkpoint=checkpoint, calls=calls,
                           cache=cache, first=first, second=second)


def test_changed_earlier_corrector_invalidates_later_cached_kalman(replay):
    original = replay.run("original")
    _, first_record = replay.checkpoint(original, replay.first)
    second_folder, second_record = replay.checkpoint(original, replay.second)
    before = pd.read_parquet(second_folder / "baseline.parquet")

    key = hashlib.sha256(producer._json(first_record["identity"])).hexdigest()
    earlier_cache = replay.cache / "FR" / str(replay.first) / key
    changed = pd.read_parquet(earlier_cache / "residual.parquet")
    for quantile in producer.QUANTILES:
        changed[f"residual_corrected__{quantile}"] += 7.
    # Simulate a newly supplied, internally checksum-consistent upstream
    # checkpoint. Atomic replacement avoids altering the previous hard link.
    producer.write_frame(earlier_cache / "residual.parquet", changed)
    first_record["files"]["residual.parquet"] = producer.sha256(earlier_cache / "residual.parquet")
    producer.write_json(earlier_cache / "receipt.json", first_record)

    replay.calls.clear()
    refreshed = replay.run("refreshed")
    folder, refreshed_record = replay.checkpoint(refreshed, replay.second)
    after = pd.read_parquet(folder / "baseline.parquet")
    np.testing.assert_allclose(after.nyx__q50, before.nyx__q50 + 7.)
    prior_identity, identity = second_record["identity"], refreshed_record["identity"]
    assert identity["prices_sha256"] == prior_identity["prices_sha256"]
    assert identity["covariates_sha256"] == prior_identity["covariates_sha256"]
    assert identity["prior_state_sha256"] != prior_identity["prior_state_sha256"]
    assert {key: value for key, value in identity.items() if key != "prior_state_sha256"} == {
        key: value for key, value in prior_identity.items() if key != "prior_state_sha256"}
    assert replay.calls == [(stage, "FR", str(replay.second))
                            for stage in ("chronos", "residual", "kalman")]


def test_identical_inner_computations_are_reused_at_later_outer_cutoff(replay):
    original = replay.run("original")
    replay.calls.clear()
    later = replay.run("later", outer_day="2026-01-05")
    assert replay.calls == []
    for day in (replay.first, replay.second):
        for zone in producer.ZONES:
            _, before = replay.checkpoint(original, day, zone)
            _, after = replay.checkpoint(later, day, zone)
            assert after == before
            assert "target_revision_utc" not in after["identity"]
            assert after["identity"]["target_history_policy"] == saturn.TARGET_HISTORY_POLICY
