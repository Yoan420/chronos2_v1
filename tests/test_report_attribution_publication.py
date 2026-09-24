"""Report attribution must survive Windows publication locks without recomputing."""
from __future__ import annotations

import errno
import json
import shutil
from pathlib import Path

import pytest

from chronos2_hourly import atomic_directory as atomic
from chronos2_hourly import report_attribution_cache as cache


MANIFEST = "report_cache_manifest.json"
COMPANION = "forecast_hourly_fr.csv"


def _windows_denial():
    error = PermissionError(errno.EACCES, "Access is denied")
    error.winerror = 5
    return error


def _write_payload(directory: Path, value: bytes = b"hourly attribution"):
    (directory / cache.FILES[0]).write_bytes(value)
    (directory / COMPANION).write_bytes(b"issued forecast")
    (directory / cache.FILES[1]).write_text(json.dumps({
        "groups": [{"key": "historical_target_price"}],
    }), encoding="utf-8")


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    source = tmp_path / "forecast.csv"
    source.write_text("immutable forecast", encoding="utf-8")
    calls = []

    def materialize(directory):
        calls.append(directory)
        _write_payload(directory)

    # Exercise the real bounded retry loop without waiting for artificial locks.
    monkeypatch.setattr(atomic.time, "sleep", lambda _: None)
    return dict(root=tmp_path / "cache", sources=[source], materialize=materialize), calls


def _leave_locked_stage(options, monkeypatch):
    attempts = []

    def locked(source, destination):
        attempts.append((source, destination))
        raise _windows_denial()

    with monkeypatch.context() as lock:
        lock.setattr(atomic, "_rename_no_replace", locked)
        with pytest.raises(atomic.AtomicDirectoryPublishError):
            cache.cached_attribution(**options)
    stage, destination = attempts[0]
    assert len(attempts) == 7
    assert attempts == [(stage, destination)] * 7
    assert stage.parent == destination.parent == options["root"]
    assert stage.name.startswith(f".attribution-{destination.name}-")
    assert (stage / MANIFEST).is_file()
    assert not destination.exists()
    return stage, destination


def test_transient_windows_denial_retries_publication_without_recomputing(inputs, monkeypatch):
    options, materializations = inputs
    native_rename = atomic._rename_no_replace
    attempts, sleeps = [], []

    def fail_twice(source, destination):
        attempts.append((source, destination))
        seal = json.loads((source / MANIFEST).read_text(encoding="utf-8"))
        assert all(cache.sha256(source / name) == seal["artifacts"][name] for name in cache.FILES)
        if len(attempts) < 3:
            raise _windows_denial()
        native_rename(source, destination)

    monkeypatch.setattr(atomic, "_rename_no_replace", fail_twice)
    monkeypatch.setattr(atomic.time, "sleep", sleeps.append)
    result = cache.cached_attribution(**options)

    assert len(materializations) == 1
    assert len(attempts) == 3 and attempts == [attempts[0]] * 3
    assert sleeps == [0.25, 0.5]
    assert result == attempts[0][1]
    assert (result / cache.FILES[0]).read_bytes() == b"hourly attribution"
    assert cache.cached_attribution(**options) == result
    assert len(materializations) == 1


def test_persistent_lock_retains_sealed_stage_and_next_call_publishes_it(inputs, monkeypatch):
    options, materializations = inputs
    stage, destination = _leave_locked_stage(options, monkeypatch)
    original = {name: (stage / name).read_bytes() for name in (*cache.FILES, COMPANION, MANIFEST)}
    assert len(materializations) == 1

    def must_not_recompute(_directory):
        pytest.fail("the complete verified attribution must be reused")

    assert cache.cached_attribution(**{**options, "materialize": must_not_recompute}) == destination
    assert not stage.exists()
    assert {name: (destination / name).read_bytes() for name in original} == original


@pytest.mark.parametrize("corrupt_competitor", [False, True])
def test_concurrent_destination_is_validated_and_never_overwritten(inputs, monkeypatch, corrupt_competitor):
    options, materializations = inputs
    destinations = []

    def competitor_publishes(source, destination):
        shutil.copytree(source, destination)
        destinations.append(destination)
        if corrupt_competitor:
            (destination / cache.FILES[0]).write_bytes(b"corrupt competitor")
        raise _windows_denial()

    monkeypatch.setattr(atomic, "_rename_no_replace", competitor_publishes)
    if corrupt_competitor:
        with pytest.raises(ValueError, match="divergent"):
            cache.cached_attribution(**options)
        assert (destinations[0] / cache.FILES[0]).read_bytes() == b"corrupt competitor"
    else:
        assert cache.cached_attribution(**options) == destinations[0]
        assert (destinations[0] / cache.FILES[0]).read_bytes() == b"hourly attribution"
    assert len(materializations) == 1
    assert len(destinations) == 1


@pytest.mark.parametrize("corruption", ["artifact", "sources", "missing_manifest"])
def test_existing_invalid_destination_is_rejected_without_materialization(inputs, corruption):
    options, materializations = inputs
    destination = cache.cached_attribution(**options)
    if corruption == "artifact":
        (destination / cache.FILES[0]).write_bytes(b"corrupt")
    elif corruption == "sources":
        seal_path = destination / MANIFEST
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        seal["sources"] = {}
        seal_path.write_text(json.dumps(seal), encoding="utf-8")
    else:
        (destination / MANIFEST).unlink()
    before = {path.name: path.read_bytes() for path in destination.iterdir()}

    with pytest.raises(ValueError, match="divergent|incomplet"):
        cache.cached_attribution(**options)

    assert len(materializations) == 1
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == before


@pytest.mark.parametrize("corruption", ["sources", "artifact", "companion", "unsealed", "invalid_json"])
def test_preserved_invalid_stage_is_not_reused(inputs, monkeypatch, corruption):
    options, materializations = inputs
    stage, destination = _leave_locked_stage(options, monkeypatch)
    if corruption == "sources":
        seal_path = stage / MANIFEST
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        seal["sources"] = {str(options["sources"][0].resolve()): "0" * 64}
        seal_path.write_text(json.dumps(seal), encoding="utf-8")
    elif corruption == "artifact":
        (stage / cache.FILES[0]).write_bytes(b"corrupt old attribution")
    elif corruption == "companion":
        (stage / COMPANION).write_bytes(b"corrupt old forecast")
    elif corruption == "unsealed":
        (stage / MANIFEST).unlink()
    else:
        (stage / MANIFEST).write_text("{invalid", encoding="utf-8")
    old_stage_bytes = {path.name: path.read_bytes() for path in stage.iterdir()}

    def fresh_materialize(directory):
        materializations.append(directory)
        _write_payload(directory, b"fresh valid attribution")

    assert cache.cached_attribution(**{**options, "materialize": fresh_materialize}) == destination
    assert len(materializations) == 2
    assert (destination / cache.FILES[0]).read_bytes() == b"fresh valid attribution"
    # Invalid recovery candidates are not repaired or resealed in place.
    assert {path.name: path.read_bytes() for path in stage.iterdir()} == old_stage_bytes


def test_interrupted_construction_is_not_reused_on_next_call(inputs):
    options, materializations = inputs
    incomplete = []

    def interrupted(directory):
        incomplete.append(directory)
        _write_payload(directory, b"uncommitted attribution")
        raise RuntimeError("interrupted attribution")

    with pytest.raises(RuntimeError, match="interrupted attribution"):
        cache.cached_attribution(**{**options, "materialize": interrupted})
    assert len(incomplete) == 1 and not incomplete[0].exists()
    assert not list(options["root"].iterdir())

    destination = cache.cached_attribution(**options)
    assert len(materializations) == 1
    assert (destination / cache.FILES[0]).read_bytes() == b"hourly attribution"


@pytest.mark.parametrize("race_boundary", ["validation", "publication"])
@pytest.mark.parametrize("corrupt_winner", [False, True])
def test_another_caller_publishes_fresh_sealed_stage_before_owner_finishes(
    inputs, monkeypatch, race_boundary, corrupt_winner,
):
    options, materializations = inputs
    native_validator = cache._validated_attribution
    native_publish = atomic.AtomicDirectoryStaging.publish
    winners = []

    def move_sealed_stage(source, destination):
        # The second caller discovers the owner's complete stage and publishes
        # that exact directory, rather than materializing another copy.
        assert (source / MANIFEST).is_file()
        atomic._rename_no_replace(source, destination)
        winners.append(destination)
        if corrupt_winner:
            (destination / cache.FILES[0]).write_bytes(b"corrupt published winner")

    def move_before_validation(directory, identity):
        if directory.name.startswith(".attribution-") and not winners:
            key = directory.name.removeprefix(".attribution-").split("-")[0]
            move_sealed_stage(directory, directory.parent / key)
        return native_validator(directory, identity)

    def move_before_publication(publication, destination):
        move_sealed_stage(publication.path, destination)
        return native_publish(publication, destination)

    if race_boundary == "validation":
        monkeypatch.setattr(cache, "_validated_attribution", move_before_validation)
    else:
        monkeypatch.setattr(atomic.AtomicDirectoryStaging, "publish", move_before_publication)

    if corrupt_winner:
        with pytest.raises(ValueError, match="divergent"):
            cache.cached_attribution(**options)
        assert (winners[0] / cache.FILES[0]).read_bytes() == b"corrupt published winner"
    else:
        result = cache.cached_attribution(**options)
        assert result == winners[0]
        assert (result / cache.FILES[0]).read_bytes() == b"hourly attribution"
        assert (result / COMPANION).read_bytes() == b"issued forecast"
        assert cache.cached_attribution(**options) == result
    assert len(winners) == len(materializations) == 1
    assert not materializations[0].exists()
