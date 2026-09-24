from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import date, timedelta
import json
from pathlib import Path

import pytest

from chronos2_hourly.nuclear_incremental import (
    NuclearIncrementalError, prepare_incremental_settings,
    resolve_incremental_epoch, retained_source_start,
)


def test_epoch_keeps_anchor_and_isolates_older_requests(tmp_path: Path) -> None:
    first = date(2026, 9, 10)
    namespace, anchor = resolve_incremental_epoch(tmp_path, first, {"recipe": "a"})
    sealed = (namespace / "epoch.json").read_bytes()
    assert anchor == first - timedelta(days=730)
    assert resolve_incremental_epoch(tmp_path, first + timedelta(days=1), {"recipe": "a"}) == (namespace, anchor)
    backfill, old_anchor = resolve_incremental_epoch(tmp_path, first - timedelta(days=1), {"recipe": "a"})
    assert backfill == namespace / "backfills" / "2026-09-09"
    assert old_anchor == anchor - timedelta(days=1)
    assert (namespace / "epoch.json").read_bytes() == sealed
    changed, _ = resolve_incremental_epoch(tmp_path, first, {"recipe": "b"})
    assert changed != namespace


def test_concurrent_first_delivery_publishes_one_complete_anchor(tmp_path: Path) -> None:
    def resolve(_):
        return resolve_incremental_epoch(tmp_path, "2026-09-10", {"seed": 42})

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(resolve, range(12)))
    assert len(set(results)) == 1
    assert len(list(tmp_path.rglob("epoch.json"))) == 1
    assert not list(tmp_path.rglob(".epoch-*"))


@pytest.mark.parametrize("change", ["anchor", "contract", "malformed"])
def test_invalid_epoch_is_not_silently_rebuilt(tmp_path: Path, change: str) -> None:
    namespace, _ = resolve_incremental_epoch(tmp_path, "2026-09-10", {"seed": 42})
    path = namespace / "epoch.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if change == "anchor":
        value["anchor_day"] = "2000-01-01"
    elif change == "contract":
        value["contract_digest"] = "invalid"
    damaged = "{" if change == "malformed" else json.dumps(value)
    path.write_text(damaged, encoding="utf-8")
    with pytest.raises(NuclearIncrementalError):
        resolve_incremental_epoch(tmp_path, "2026-09-11", {"seed": 42})
    assert path.read_text(encoding="utf-8") == damaged


def _config(cache: Path) -> dict:
    return {"model": {"model_id": "amazon/chronos-2", "revision": "local-commit", "context_length": 2048},
            "hourly": {"residual_correction": {"iterations": 50, "thread_count": 4}},
            "zones": {"FR": {"timezone": "Europe/Paris", "target": {"file": "old/target.parquet"},
                              "covariates": {"nuclear": {"series": "nuclear.fr", "pit_file": "old/pit.parquet"}}}},
            "data": {"runtime_as_of": "2026-09-09T08:00:00+02:00", "project_root": "old/root"},
            "nuclear_experiment": {"mode": "incremental", "incremental_cache_dir": str(cache)}}


def test_daily_scope_reuses_semantic_recipe_across_snapshot_paths(tmp_path: Path) -> None:
    config = _config(tmp_path / "daily")
    namespace, anchor, start = prepare_incremental_settings(config, "2026-09-10")
    assert start == anchor
    changed = deepcopy(config)
    changed["zones"]["FR"]["target"]["file"] = "new/target.parquet"
    changed["zones"]["FR"]["covariates"]["nuclear"]["pit_file"] = "new/pit.parquet"
    changed["data"].update(runtime_as_of="2026-09-10T08:00:00+02:00", project_root="new/root")
    changed["hourly"]["residual_correction"]["thread_count"] = 8
    assert prepare_incremental_settings(changed, "2026-09-11") == (namespace, anchor, anchor)
    future = date(2028, 9, 10)
    assert prepare_incremental_settings(changed, future) == (namespace, anchor, future - timedelta(days=1095))
    changed["hourly"]["residual_correction"]["iterations"] = 51
    new_namespace, new_anchor, _ = prepare_incremental_settings(changed, "2026-09-11")
    assert new_namespace != namespace
    assert new_anchor == date(2026, 9, 11) - timedelta(days=730)


def test_source_scope_retains_bootstrap_and_is_bounded(tmp_path: Path) -> None:
    day = date(2026, 9, 10)
    assert retained_source_start(tmp_path, day) == day - timedelta(days=730)
    resolve_incremental_epoch(tmp_path / "fr/civil_pit_v2", day, {"seed": 42})
    assert retained_source_start(tmp_path, day + timedelta(days=5)) == day - timedelta(days=730)
    future = day + timedelta(days=500)
    assert retained_source_start(tmp_path, future) == future - timedelta(days=1095)
    assert retained_source_start(tmp_path, day - timedelta(days=1)) == day - timedelta(days=731)


def test_full_and_legacy_modes_do_not_create_incremental_state(tmp_path: Path) -> None:
    day = date(2026, 9, 10)
    for config in ({}, {"nuclear_experiment": {"mode": "full", "incremental_cache_dir": str(tmp_path / "unused")}}):
        assert prepare_incremental_settings(config, day) == (None, day - timedelta(days=730), day - timedelta(days=730))
    assert not (tmp_path / "unused").exists()
    with pytest.raises(NuclearIncrementalError, match="incremental_cache_dir"):
        prepare_incremental_settings({"nuclear_experiment": {"mode": "incremental"}}, day)
