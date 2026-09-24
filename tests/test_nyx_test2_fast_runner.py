"""Bounded orchestration tests: no model fit, provider call or process launch."""
import json
from pathlib import Path

import pytest

import run_nyx_test2_fast as r


@pytest.fixture
def captured(tmp_path, monkeypatch):
    old = tmp_path / 'old'
    source = old / 'sources'
    source.mkdir(parents=True)
    plan = old / 'plan.json'
    plan.write_text('{"fixture": true}', encoding='utf-8')
    monkeypatch.setattr(r, 'ORIGINAL_ROOT', old)
    monkeypatch.setattr(r, 'ORIGINAL_PLAN_SHA256', r.sha(plan))
    aliases = ['de_solar_generation_fcst', 'nl_solar_generation_fcst',
               'be_solar_generation_fcst', 'fr_solar_generation_fcst',
               'de_wind_generation_fcst', 'nl_wind_generation_fcst']
    records = {}
    for alias in aliases:
        frame = source / (alias + '.parquet')
        frame.write_bytes(alias.encode())
        audit = source / (alias + '.parquet.audit.json')
        audit.write_text(json.dumps({'alias': alias}), encoding='utf-8')
        records[alias] = {'sha256': r.sha(frame), 'audit_sha256': r.sha(audit)}
    (source / 'receipt_denl.json').write_text(json.dumps({
        'delivery_day': '2026-09-24', 'sources': records}), encoding='utf-8')
    return old, r.frozen_source_manifest()


def test_captured_source_manifest_exact_and_copy_idempotent(captured, tmp_path, monkeypatch):
    old, manifest = captured
    assert len(manifest['files']) == 12
    from chronos2_hourly import nyx_fast_sources
    root = tmp_path / 'new'
    monkeypatch.setattr(nyx_fast_sources, 'OUTPUT', root)
    out = root / '2026-09-24/sources'
    r.clone_captured_sources(out, manifest)
    r.clone_captured_sources(out, manifest)
    r.verify_captured_copies(out, manifest)
    assert {p.name for p in out.iterdir()} == set(manifest['files'])
    for name, item in manifest['files'].items():
        assert r.sha(out / name) == item['sha256']
        assert (old / 'sources' / name).read_bytes() == (out / name).read_bytes()


def test_befr_extension_cannot_refresh_captured_curves(captured, tmp_path, monkeypatch):
    _, manifest = captured
    from chronos2_hourly import nyx_fast_sources
    root = tmp_path / 'new'
    monkeypatch.setattr(nyx_fast_sources, 'OUTPUT', root)
    out = root / 'sources'
    r.clone_captured_sources(out, manifest)
    (out / 'be_wind_generation_fcst.parquet').write_bytes(b'newly authorized extension')
    r.verify_captured_copies(out, manifest)
    (out / 'de_wind_generation_fcst.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError, match='frozen copied curve changed'):
        r.verify_captured_copies(out, manifest)


def test_captured_source_changed_refused(captured):
    old, _ = captured
    (old / 'sources/de_wind_generation_fcst.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Original immutable source changed'):
        r.frozen_source_manifest()


def test_existing_different_destination_never_overwritten(captured, tmp_path, monkeypatch):
    _, manifest = captured
    from chronos2_hourly import nyx_fast_sources
    root = tmp_path / 'new'
    monkeypatch.setattr(nyx_fast_sources, 'OUTPUT', root)
    out = root / 'sources'
    out.mkdir(parents=True)
    dest = out / 'be_solar_generation_fcst.parquet'
    dest.write_bytes(b'existing-different')
    with pytest.raises(ValueError, match='Frozen source copy mismatch'):
        r.clone_captured_sources(out, manifest)
    assert dest.read_bytes() == b'existing-different'


def test_plan_is_distinct_fixed_scientific_profile():
    plan = r.build_plan('2026-09-24')
    assert plan['engine'] == 'nyx_test2_fast_v1'
    assert plan['threads'] == r.CHRONOS_THREADS
    assert plan['workers'] == plan['residual_workers'] == 4
    assert plan['residual_threads'] == 2 and plan['test2_threads'] == 1
    assert plan['priority'] == 'Normal'
    assert plan['min_free_memory_gib'] == 3.5
    assert plan['pairs'] == [['DE', 'NL'], ['BE', 'FR']]
    assert plan['test2_iterations'] == 120 and plan['test2_seed'] == 20260923
    assert not plan['production_modified'] and not plan['automatic_promotion']
    assert 'chronos2_hourly/nyx_live_parallel.py' in plan['code']
    assert r.OUTPUT == r.ROOT / 'runs/experiments/n2'


@pytest.mark.parametrize('day', ['2026-09-23', '2026-09-25', '2026-09-24T01:00:00'])
def test_date_cannot_roll_or_migrate(day):
    with pytest.raises(ValueError):
        r.build_plan(day)


def test_pair_pipeline_stays_original_one_thread():
    import inspect
    source = inspect.getsource(r.Run.pair)
    assert 'from chronos2_hourly.nyx_live_hybrid import run_pair_pipeline' in source
    assert 'threads=1, iterations=120, seed=20260923' in source
    assert 'verify_completed_pair(work, identity)' in source
