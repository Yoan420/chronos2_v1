from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import materialize_saturn_kalman_fuel as fuel
import chronos2_hourly.nuclear_residual_inputs as inputs


@pytest.fixture
def bank_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fetch(days, **kwargs):
        calls.append(list(days))
        expected = fuel._expected_hourly_index(days[0], days[-1])
        cutoffs = fuel._cutoffs_for_value_times(expected)
        return {
            alias: pd.DataFrame({
                alias: np.full(len(expected), 20.0 + i),
                f"{alias}__snapshot_time_utc": cutoffs,
                f"{alias}__revision_time_utc": cutoffs,
                f"{alias}__spring_dst_repair": False,
            }, index=expected)
            for i, alias in enumerate(fuel.RESIDUAL_LOAD_ALIASES)
        }

    monkeypatch.setattr(fuel, "_fetch_residual_saturn_bank", fetch)
    monkeypatch.setattr(inputs, "ROOT", tmp_path)

    def make(start, end=None, directory="seed"):
        path, _ = fuel._materialize_residual_load_market_features(
            start_day=pd.Timestamp(start), end_day=pd.Timestamp(end or start),
            output_dir=tmp_path / directory,
        )
        return path

    return make, calls


def _destination(tmp_path):
    return tmp_path / "data/pit/nuclear_forecast" / fuel.RESIDUAL_LOAD_OUTPUT_NAME


@pytest.mark.parametrize("day,hours", [
    ("2025-01-15", 24), ("2025-03-30", 23), ("2025-10-26", 25),
    ("2025-03-31", 24), ("2025-10-27", 24),
])
def test_readonly_audit_preserves_strict_native_contract(bank_factory, day, hours):
    make, _ = bank_factory
    path = make(day)
    before = path.read_bytes()
    audit = inputs.audit_residual_bank(path, day, day)
    assert audit["complete"], audit["blockers"]
    assert audit["covered_hours"] == audit["expected_hours"] == hours
    assert audit["source_audit"]["naive_timezones"]["nl_residual_load_fcst"] == "UTC"
    assert audit["production_pit_evidence"] is False
    assert path.read_bytes() == before
    json.dumps(audit, allow_nan=False)


def test_audit_reports_missing_suffix(bank_factory):
    make, _ = bank_factory
    path = make("2025-01-15")
    audit = inputs.audit_residual_bank(path, "2025-01-15", "2025-01-17")
    assert not audit["complete"]
    assert audit["source_provenance_valid"]
    assert audit["missing_hour_count"] == len(audit["missing_hours"]) == 48
    assert audit["missing_days"] == ["2025-01-16", "2025-01-17"]


def test_verified_seed_then_only_missing_suffix_is_downloaded(tmp_path, bank_factory):
    make, calls = bank_factory
    seed = make("2025-01-15")
    original = seed.read_bytes()
    original_audit = seed.with_name(seed.name + ".audit.json").read_bytes()
    result = inputs.ensure_residual_bank(
        path=_destination(tmp_path), seed_path=seed,
        start_day="2025-01-15", end_day="2025-01-17", workers=4,
    )
    assert result["complete"]
    assert calls[-1] == list(pd.date_range("2025-01-16", "2025-01-17", freq="D"))
    assert seed.read_bytes() == original
    assert seed.with_name(seed.name + ".audit.json").read_bytes() == original_audit
    assert not _destination(tmp_path).with_name(seed.name + ".lock").exists()
    count = len(calls)
    again = inputs.ensure_residual_bank(
        path=_destination(tmp_path), seed_path=seed,
        start_day="2025-01-15", end_day="2025-01-17", workers=4,
    )
    assert again["complete"] and len(calls) == count


def test_sync_disabled_is_readonly_even_with_seed(tmp_path, bank_factory):
    make, calls = bank_factory
    seed = make("2025-01-15")
    result = inputs.ensure_residual_bank(
        path=_destination(tmp_path), seed_path=seed,
        start_day="2025-01-15", end_day="2025-01-17", workers=4, allow_sync=False,
    )
    assert not result["complete"]
    assert not _destination(tmp_path).parent.exists()
    assert len(calls) == 1


def test_corrupt_seed_is_not_resealed_or_downloaded(tmp_path, bank_factory):
    make, calls = bank_factory
    seed = make("2025-01-15")
    frame = pd.read_parquet(seed)
    frame.loc[0, "fr_residual_load_fcst"] += 5
    frame.to_parquet(seed, index=False)
    with pytest.raises(ValueError, match="seed refused"):
        inputs.ensure_residual_bank(
            path=_destination(tmp_path), seed_path=seed,
            start_day="2025-01-15", end_day="2025-01-17", workers=4,
        )
    assert len(calls) == 1
    assert not _destination(tmp_path).exists()


def test_late_cutoff_is_refused_even_with_matching_sha(bank_factory):
    make, _ = bank_factory
    seed = make("2025-03-31")
    frame = pd.read_parquet(seed)
    frame.loc[0, "snapshot_time_utc"] += pd.Timedelta(hours=1)
    frame.to_parquet(seed, index=False)
    audit_path = seed.with_name(seed.name + ".audit.json")
    payload = json.loads(audit_path.read_text())
    payload["sha256"] = payload["output_sha256"] = hashlib.sha256(seed.read_bytes()).hexdigest()
    audit_path.write_text(json.dumps(payload), encoding="utf-8")
    audit = inputs.audit_residual_bank(seed, "2025-03-31", "2025-03-31")
    assert not audit["source_provenance_valid"]
    assert not audit["complete"]


def test_missing_audit_and_existing_lock_fail_closed(tmp_path, bank_factory):
    make, calls = bank_factory
    seed = make("2025-01-15")
    destination = _destination(tmp_path)
    destination.parent.mkdir(parents=True)
    lock = destination.with_name(destination.name + ".lock")
    lock.write_text("another process", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already locked"):
        inputs.ensure_residual_bank(
            path=destination, seed_path=seed,
            start_day="2025-01-15", end_day="2025-01-17", workers=4,
        )
    assert lock.read_text() == "another process"
    assert len(calls) == 1
    seed.with_name(seed.name + ".audit.json").unlink()
    assert not inputs.audit_residual_bank(seed, "2025-01-15", "2025-01-15")["complete"]


def test_shared_destination_and_wrong_filename_are_forbidden(tmp_path, bank_factory):
    for path in (tmp_path / "data/pit/vintages" / fuel.RESIDUAL_LOAD_OUTPUT_NAME,
                 tmp_path / "data/pit/nuclear_forecast/wrong.parquet"):
        with pytest.raises(ValueError, match="writes require"):
            inputs.ensure_residual_bank(
                path=path, seed_path=None,
                start_day="2025-01-15", end_day="2025-01-17", workers=4,
            )


def test_missing_prefix_is_refused_instead_of_rebuilding(bank_factory):
    make, _ = bank_factory
    seed = make("2025-01-15")
    audit = inputs.audit_residual_bank(seed, "2025-01-14", "2025-01-15")
    assert not audit["source_provenance_valid"]
    assert any("prefixe" in text for text in audit["blockers"])
