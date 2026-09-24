from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.chronos_adapter import (
    generate_delivery_plans, normalize_chronos_future, normalize_chronos_oof,
)
from chronos2_hourly.nuclear_daily_cache import NuclearDailyChronosCache
from chronos2_modular import forecasting


CONFIG = {"model": {"model_id": "amazon/chronos-2", "revision": "a" * 40, "seed": 42}}
CONTEXT = 48


def _fixture(start="2024-03-29", end="2024-04-02"):
    plans = generate_delivery_plans(start, end, timezone="Europe/Paris",
                                    forecast_origin_local_time="08:00")
    first = plans[0].delivery_start_utc - pd.Timedelta(hours=CONTEXT * 2)
    index = pd.date_range(first, plans[-1].delivery_index_utc[-1], freq="h", tz="UTC")
    target = pd.Series(np.arange(len(index), dtype=float) / 7, index=index, name="price")
    covariates = pd.DataFrame({"nuclear": np.arange(len(index), dtype=float) / 20,
                               "unused": 0.0}, index=index)
    model_context = pd.DataFrame({"known_nuclear_oracle": covariates.nuclear.astype(np.float32)},
                                 index=index)
    data = SimpleNamespace(zone="FR", timezone="Europe/Paris", frequency="h", target=target,
                           covariates=covariates, model_context_covariates=model_context,
                           known_future_columns=["known_nuclear_oracle"])
    return data, plans


def _cache(tmp_path, data, **kwargs):
    return NuclearDailyChronosCache(tmp_path, data=data, config=kwargs.pop("config", CONFIG),
                                    context_length=CONTEXT, **kwargs)


def _prediction(data, plan):
    """A deterministic fake model consumes exactly the production input frames."""
    origin = data.target.index.searchsorted(plan.delivery_start_utc)
    if origin == len(data.target):
        context, future = forecasting.build_live_frames(data, CONTEXT, plan.horizon, "test", True)
    else:
        context, future, _ = forecasting.build_origin_frames(
            data, origin, CONTEXT, plan.horizon, "test", True)
    center = context.target.mean() + future.known_nuclear_oracle.to_numpy(dtype=float)
    return pd.DataFrame({"q10": center - 2, "q50": center, "q90": center + 2,
                         "forecast_origin_utc": plan.forecast_origin_utc},
                        index=plan.delivery_index_utc)


def _executor(data, calls):
    def execute(plans):
        calls.append(tuple(plan.delivery_date for plan in plans))
        frames = []
        for plan in plans:
            frame = _prediction(data, plan)
            frame["actual"] = data.target.reindex(frame.index).to_numpy(dtype=np.float32)
            frames.append(frame)
        return normalize_chronos_oof(pd.concat(frames))
    return execute


@pytest.mark.parametrize("day, hours", [("2024-03-31", 23), ("2024-10-27", 25)])
def test_issued_future_becomes_history_without_recomputing_and_preserves_dst(tmp_path, day, hours):
    historical_data, plans = _fixture(day, day)
    plan = plans[0]
    live_data = deepcopy(historical_data)
    live_data.target = live_data.target.loc[live_data.target.index < plan.delivery_start_utc]
    # Production covariates stop at target end; explicit oracle values continue
    # in model_context_covariates. Both paths must produce identical fingerprints.
    live_data.covariates = live_data.covariates.reindex(live_data.target.index)
    live_cache = _cache(tmp_path, live_data)
    live = live_cache.resolve_future(plan, lambda: _prediction(live_data, plan))
    historical_cache = _cache(tmp_path, historical_data)
    assert live_cache.identity(plan) == historical_cache.identity(plan)
    result = historical_cache.resolve_history(plans, lambda _: pytest.fail("Unexpected replay"))
    assert len(result) == hours
    pd.testing.assert_frame_equal(live, result.drop(columns="actual"))
    np.testing.assert_array_equal(result.actual, historical_data.target.reindex(result.index).astype(np.float32))
    payload = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert "actual" not in payload["prediction"]
    assert historical_cache.audit["history_hits"] == 1


def test_cross_run_extension_computes_only_new_day_and_matches_uncached(tmp_path):
    data, plans = _fixture()
    calls = []
    first = _cache(tmp_path, data).resolve_history(plans[:-1], _executor(data, calls))
    cache = _cache(tmp_path, data)
    second = cache.resolve_history(plans, _executor(data, calls))
    expected = _executor(data, [])(plans)
    pd.testing.assert_frame_equal(second, expected)
    pd.testing.assert_frame_equal(second.loc[first.index], first)
    assert calls[-1] == (plans[-1].delivery_date,)
    assert cache.audit["history_hits"] == len(plans) - 1
    assert cache.audit["history_misses"] == 1


def test_preserves_float32_quantiles_across_json_roundtrip(tmp_path):
    data, plans = _fixture("2024-10-27", "2024-10-27")
    plan = plans[0]
    prediction = _prediction(data, plan)
    prediction[["q10", "q50", "q90"]] = prediction[["q10", "q50", "q90"]].astype(np.float32)
    cache = _cache(tmp_path, data)
    cache.store(plan, prediction)
    restored = _cache(tmp_path, data).load(plan)
    pd.testing.assert_frame_equal(restored, normalize_chronos_future(prediction, plan))


def test_changed_observation_does_not_change_its_own_prediction_but_invalidates_later_context(tmp_path):
    data, plans = _fixture()
    baseline = _cache(tmp_path, data)
    baseline.resolve_history(plans, _executor(data, []))
    changed = deepcopy(data)
    changed.target.loc[plans[1].delivery_index_utc] += 999
    revised = _cache(tmp_path, changed)
    assert revised.identity(plans[1]) == baseline.identity(plans[1])
    assert revised.identity(plans[2]) != baseline.identity(plans[2])
    assert revised.load(plans[1], historical=True).actual.iloc[0] > 999
    assert revised.load(plans[2], historical=True) is None


def test_fingerprints_selected_model_inputs_and_semantic_contract_not_outer_paths(tmp_path):
    data, plans = _fixture()
    plan = plans[2]
    original = _cache(tmp_path, data)
    identity = original.identity(plan)
    changed = deepcopy(data)
    changed.target.iloc[0] += 999
    changed.target.loc[plan.delivery_index_utc] += 999
    changed.covariates["unused"] += 999
    changed.covariates["nuclear"] += 999  # Oracle uses the named model-context column.
    config = deepcopy(CONFIG)
    config.update(output={"directory": "another-day"}, data={"runtime_as_of": "new-date"})
    assert _cache(tmp_path, changed, config=config).identity(plan) == identity
    changed.model_context_covariates.loc[plan.delivery_index_utc, "known_nuclear_oracle"] += 1
    assert _cache(tmp_path, changed).identity(plan) != identity
    config["model"]["revision"] = "b" * 40
    assert _cache(tmp_path, data, config=config).identity(plan) != identity
    assert _cache(tmp_path, data, model_batch_size=64).identity(plan) != identity
    assert _cache(tmp_path, data, execution_signature={"threads": 3}).identity(plan) != identity


def test_lag_and_persistence_inputs_are_hashed_even_when_outside_context(tmp_path):
    data, plans = _fixture()
    plan = plans[0]
    data.known_future_columns = ["known_nuclear_lag168", "known_nuclear_persistence"]
    # Cover lag input before the target context with an explicit covariate series.
    lag_index = plan.delivery_index_utc - pd.Timedelta(hours=168)
    extended = data.covariates.index.union(lag_index)
    data.covariates = data.covariates.reindex(extended).fillna(3)
    original = _cache(tmp_path, data).identity(plan)
    changed = deepcopy(data)
    changed.covariates.loc[lag_index, "nuclear"] += 1
    assert _cache(tmp_path, changed).identity(plan) != original


@pytest.mark.parametrize("damage", ["json", "checksum", "identity", "coverage"])
def test_corrupt_entries_are_misses_and_repaired_atomically(tmp_path, damage):
    data, plans = _fixture("2024-10-27", "2024-10-27")
    cache = _cache(tmp_path, data)
    original = cache.resolve_history(plans, _executor(data, []))
    path = next(tmp_path.rglob("*.json"))
    if damage == "json":
        path.write_text("{broken", encoding="utf-8")
    else:
        payload = json.loads(path.read_text())
        if damage == "checksum":
            payload["prediction"]["q50"][0] += 1
        elif damage == "identity":
            payload["identity"]["context_sha256"] = "bad"
        else:
            payload["prediction"]["delivery_index_ns"].pop()
        path.write_text(json.dumps(payload), encoding="utf-8")
    calls = []
    result = cache.resolve_history(plans, _executor(data, calls))
    pd.testing.assert_frame_equal(result, original)
    assert len(calls) == 1 and cache.audit["invalid_entries"] == 1
    assert not list(tmp_path.rglob("*.tmp"))
    pd.testing.assert_frame_equal(cache.load(plans[0], historical=True), original)


def test_separated_misses_do_not_pass_sparse_plans_to_strict_oof_executor(tmp_path):
    data, plans = _fixture()
    cache = _cache(tmp_path, data)
    cache.store(plans[1], _prediction(data, plans[1]))
    cache.store(plans[3], _prediction(data, plans[3]))
    calls = []
    result = cache.resolve_history(plans, _executor(data, calls))
    assert calls == [(plans[i].delivery_date,) for i in (0, 2, 4)]
    pd.testing.assert_frame_equal(result, _executor(data, [])(plans))


def test_rejects_mutable_revision_and_wrong_day_without_publishing(tmp_path):
    data, plans = _fixture()
    with pytest.raises(ValueError, match="pinned"):
        _cache(tmp_path, data, config={"model": {"revision": "main"}})
    cache = _cache(tmp_path, data)
    with pytest.raises(ValueError):
        cache.store(plans[0], _prediction(data, plans[1]))
    assert not list(tmp_path.rglob("*.json"))
