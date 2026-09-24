from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from nyx_clean_fuel import sources as s


def source_evidence():
    return {alias: {"series": s.SERIES[alias], "formula": s.FORMULAS[alias],
                    "formula_sha256": s._sha(s.FORMULAS[alias].encode("utf8")),
                    "unit": s.UNITS[alias], "tzaware": False} for alias in s.SERIES}


def row(day):
    cutoff = s.civil_cutoff(day)
    stamp = (cutoff.tz_convert(s.TIMEZONE).normalize() - pd.DateOffset(days=1)).tz_convert("UTC")
    out = {"delivery_day": day, "cutoff_time_utc": cutoff.isoformat()}
    for number, alias in enumerate(s.SERIES):
        out.update({alias: 90.0 + number, alias + "__value_time_utc": stamp.isoformat(),
                    alias + "__age_hours": (cutoff - stamp).total_seconds() / 3600})
    return out


@pytest.fixture
def fake_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "ROOT", tmp_path)
    monkeypatch.setattr(s, "_source_evidence", lambda settings: source_evidence())
    monkeypatch.setattr(s, "_one_day", lambda day, settings: row(day))
    return tmp_path / "runs/experiments/nyx_clean_fuel_v1/inputs/test"


@pytest.mark.parametrize("day,expected", [("2026-03-30", "2026-03-29T06:00:00Z"),
                                          ("2026-10-26", "2026-10-25T07:00:00Z"),
                                          ("2026-09-14", "2026-09-13T06:00:00Z")])
def test_civil_cutoff_across_dst(day, expected):
    assert s.civil_cutoff(day) == pd.Timestamp(expected)


def test_prior_close_excludes_early_same_day_and_future():
    cutoff = s.civil_cutoff("2026-09-17")
    values = pd.Series([100.0, 200.0, 900.0], index=pd.to_datetime(["2026-09-15", "2026-09-16", "2026-09-17"]))
    value, stamp, age = s.select_previous_close(values, cutoff)
    assert value == 100.0
    assert stamp == pd.Timestamp("2026-09-14T22:00:00Z")
    assert age == 32.0


def test_weekend_selects_bounded_last_known_value():
    cutoff = s.civil_cutoff("2026-09-14")
    values = pd.Series([np.nan, 120.0], index=pd.to_datetime(["2026-09-10", "2026-09-11"]))
    assert s.select_previous_close(values, cutoff)[0] == 120.0


@pytest.mark.parametrize("raw", [None, pd.DataFrame({"x": [1]}),
                                    pd.Series([1, 2], index=pd.to_datetime(["2026-09-15", "2026-09-15"])),
                                    pd.Series([np.inf], index=pd.to_datetime(["2026-09-15"])),
                                    pd.Series(["bad"], index=pd.to_datetime(["2026-09-15"])),
                                    pd.Series([np.nan], index=pd.to_datetime(["2026-09-15"])),
                                    pd.Series([1], index=pd.to_datetime(["2026-09-01"])),
                                    pd.Series([1], index=pd.to_datetime(["2026-09-16"]))])
def test_bad_or_stale_source_fails_closed(raw):
    with pytest.raises(s.CleanFuelSourceError):
        s.select_previous_close(raw, s.civil_cutoff("2026-09-17"))


def test_native_registry_hubs_and_no_double_carbon():
    assert s.HUBS == {"fr": "peg", "de": "the", "be": "zee", "nl": "ttf"}
    assert s.SERIES["ccc"] == "ccc.price.mid.api2.everyday.month.1.ice.eurmwh"
    assert "6.9776" in s.FORMULAS["ccc"]
    assert "0.34" in s.FORMULAS["ccc"]
    for alias in ("cgc_fr", "cgc_de", "cgc_be", "cgc_nl"):
        assert "0.368" in s.FORMULAS[alias]


def test_materialize_roundtrip_and_reuse_without_network(fake_sources, monkeypatch):
    path = s.materialize({}, "2026-03-28", "2026-03-31", fake_sources)
    frame, audit = s.load_bank(path)
    assert frame.delivery_day.tolist() == s._days("2026-03-28", "2026-03-31")
    assert audit["same_day_closes_excluded"] is True
    assert audit["production_pit_evidence"] is False
    assert list(frame.columns) == s._columns()
    assert len(list((fake_sources / "daily").glob("*.json"))) == 4
    monkeypatch.setattr(s, "_source_evidence", lambda settings: pytest.fail("Unexpected network"))
    assert s.materialize({}, "2026-03-28", "2026-03-31", fake_sources) == path
    with pytest.raises(s.CleanFuelSourceError, match="immutable"):
        s.materialize({}, "2026-03-28", "2026-04-01", fake_sources)


@pytest.mark.parametrize("mutation", ["sha", "units", "series", "formula", "pit", "count", "age", "cutoff", "source_time", "naive_cutoff", "duplicate", "extra", "missing"])
def test_load_bank_rejects_tampering(fake_sources, mutation):
    path = s.materialize({}, "2026-09-14", "2026-09-15", fake_sources)
    audit_path = Path(str(path) + ".audit.json")
    audit = json.loads(audit_path.read_text())
    frame = pd.read_parquet(path)
    if mutation == "sha":
        audit["sha256"] = "bad"
    elif mutation == "units":
        audit["units"]["ccc"] = "EUR/MWh_th"
    elif mutation == "series":
        audit["series"]["cgc_fr"] = "Storm"
    elif mutation == "formula":
        audit["source_evidence"]["ccc"]["formula"] = "empty"
    elif mutation == "pit":
        audit["production_pit_evidence"] = True
    elif mutation == "count":
        audit["days"] = 1
    else:
        if mutation == "age":
            frame.loc[0, "ccc__age_hours"] = 0
        elif mutation == "cutoff":
            frame.loc[0, "cutoff_time_utc"] += pd.Timedelta(hours=1)
        elif mutation == "source_time":
            frame.loc[0, "ccc__value_time_utc"] = frame.loc[0, "cutoff_time_utc"]
            frame.loc[0, "ccc__age_hours"] = 0
        elif mutation == "naive_cutoff":
            frame["cutoff_time_utc"] = frame.cutoff_time_utc.dt.tz_localize(None)
        elif mutation == "duplicate":
            frame.loc[1, "delivery_day"] = frame.loc[0, "delivery_day"]
        elif mutation == "extra":
            frame["future"] = 42
        elif mutation == "missing":
            frame.loc[0, "cgc_de"] = np.nan
        frame.to_parquet(path, index=False)
        audit["sha256"] = s._sha(path.read_bytes())
    audit_path.write_text(json.dumps(audit), encoding="utf8")
    with pytest.raises(s.CleanFuelSourceError):
        s.load_bank(path)


def test_checkpoint_resume_validates_hash_and_contract(fake_sources, monkeypatch):
    original = s._one_day
    calls = []

    def fail_second(day, settings):
        calls.append(day)
        if day == "2026-09-15":
            raise s.CleanFuelSourceError("Missing source")
        return original(day, settings)

    monkeypatch.setattr(s, "_one_day", fail_second)
    with pytest.raises(s.CleanFuelSourceError, match="Missing source"):
        s.materialize({}, "2026-09-14", "2026-09-15", fake_sources, workers=1)
    assert not (fake_sources / "bank.parquet").exists()
    assert not (fake_sources / ".materialize.lock").exists()
    checkpoint = fake_sources / "daily/2026-09-14.json"
    assert checkpoint.is_file()
    checkpoint_record = json.loads(checkpoint.read_text())
    checkpoint_record["row"]["ccc"] += 5
    checkpoint.write_text(json.dumps(checkpoint_record))
    monkeypatch.setattr(s, "_one_day", original)
    with pytest.raises(s.CleanFuelSourceError, match="checksum"):
        s.materialize({}, "2026-09-14", "2026-09-15", fake_sources, workers=1)


def test_resume_reuses_valid_days_only(fake_sources, monkeypatch):
    directory = fake_sources / "daily"
    directory.mkdir(parents=True)
    contract = s._contract(176.0)
    known = row("2026-09-14")
    (directory / "2026-09-14.json").write_bytes(s._json_bytes({"contract_sha256": s._sha(s._json_bytes(contract)), "row": known, "row_sha256": s._sha(s._json_bytes(known))}))
    calls = []
    monkeypatch.setattr(s, "_one_day", lambda day, settings: calls.append(day) or row(day))
    path = s.materialize({}, "2026-09-14", "2026-09-15", fake_sources)
    assert calls == ["2026-09-15"]
    assert len(s.load_bank(path)[0]) == 2


def test_output_scope_and_concurrent_lock(fake_sources):
    with pytest.raises(s.CleanFuelSourceError, match="isolated"):
        s.materialize({}, "2026-09-14", "2026-09-15", s.ROOT / "runs/exports")
    fake_sources.mkdir(parents=True)
    lock = fake_sources / ".materialize.lock"
    lock.write_text("active")
    with pytest.raises(s.CleanFuelSourceError, match="owns"):
        s.materialize({}, "2026-09-14", "2026-09-15", fake_sources)
    assert lock.read_text() == "active"


@pytest.mark.parametrize("config", [{"sources": {"unknown": 1}}, {"sources": {"retries": 0}},
                                     {"sources": {"maximum_age_hours": 999}}, {"sources": {"request_timeout_seconds": True}}])
def test_invalid_settings_before_writes(fake_sources, config):
    with pytest.raises(s.CleanFuelSourceError):
        s.materialize(config, "2026-09-14", "2026-09-15", fake_sources)
    assert not fake_sources.exists()


def test_network_query_is_asof_and_bounded(monkeypatch):
    calls = []
    class Client:
        def get(self, name, **kwargs):
            calls.append((name, kwargs))
            return pd.Series([85.0, 99.0], index=pd.to_datetime(["2026-09-15", "2026-09-16"]))
    monkeypatch.setattr(s, "_client", lambda timeout: Client())
    actual = s._one_day("2026-09-17", deepcopy(s.DEFAULTS))
    assert all(actual[name] == 85 for name in s.SERIES)
    assert len(calls) == 5
    assert all(kwargs["revision_date"] == s.civil_cutoff("2026-09-17") for _, kwargs in calls)
    assert all(kwargs["to_value_date"] == kwargs["revision_date"] for _, kwargs in calls)


def test_no_tls_retry_or_validation_retry(monkeypatch):
    calls = []
    monkeypatch.setattr(s.time, "sleep", lambda seconds: None)
    def operation(error):
        calls.append(1)
        raise error
    with pytest.raises(requests.exceptions.SSLError):
        s._retry(lambda: operation(requests.exceptions.SSLError("cert")), 3)
    assert len(calls) == 1
    calls.clear()
    with pytest.raises(s.CleanFuelSourceError):
        s._retry(lambda: operation(s.CleanFuelSourceError("bad")), 3)
    assert len(calls) == 1
    calls.clear()
    with pytest.raises(requests.Timeout):
        s._retry(lambda: operation(requests.Timeout("slow")), 2)
    assert len(calls) == 2


def test_provider_formula_change_fails(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return "altered"
    class Session:
        def get(self, *args, **kwargs):
            return Response()
    class Client:
        session = Session()
    monkeypatch.setattr(s, "_client", lambda timeout: Client())
    with pytest.raises(s.CleanFuelSourceError, match="formula changed"):
        s._source_evidence(deepcopy(s.DEFAULTS))


def test_next_delivery_reuses_verified_bank_and_collects_only_new_day(fake_sources, monkeypatch):
    previous = fake_sources.parent / "2026-09-16"
    path = s.materialize({}, "2026-09-10", "2026-09-16", previous)
    original_data, original_audit = path.read_bytes(), Path(str(path) + ".audit.json").read_bytes()
    calls = []
    monkeypatch.setattr(s, "_one_day", lambda day, settings: calls.append(day) or row(day))
    new = s.materialize({}, "2026-09-11", "2026-09-17", fake_sources.parent / "2026-09-17")
    frame, audit = s.load_bank(new)
    assert calls == ["2026-09-17"]
    assert len(frame) == 7
    assert audit["seed_reuse"]["copied_days"] == 6
    used = audit["seed_reuse"]["sources"][0]
    assert used["sha256"] == s._sha(original_data)
    assert used["audit_sha256"] == s._sha(original_audit)
    assert used["copied_days"] == s._days("2026-09-11", "2026-09-16")
    assert path.read_bytes() == original_data
    assert Path(str(path) + ".audit.json").read_bytes() == original_audit


def test_multiple_disjoint_seed_banks_cover_target(fake_sources, monkeypatch):
    s.materialize({}, "2026-09-10", "2026-09-12", fake_sources.parent / "2026-09-12")
    s.materialize({}, "2026-09-14", "2026-09-16", fake_sources.parent / "2026-09-16")
    calls = []
    monkeypatch.setattr(s, "_one_day", lambda day, settings: calls.append(day) or row(day))
    new = s.materialize({}, "2026-09-10", "2026-09-17", fake_sources.parent / "2026-09-17", workers=1)
    _, audit = s.load_bank(new)
    assert calls == ["2026-09-13", "2026-09-17"]
    assert len(audit["seed_reuse"]["sources"]) == 2
    assert audit["seed_reuse"]["copied_days"] == 6


def test_selected_corrupt_seed_stops_instead_of_redownloading(fake_sources, monkeypatch):
    seed = s.materialize({}, "2026-09-10", "2026-09-16", fake_sources.parent / "2026-09-16")
    seed.write_bytes(seed.read_bytes() + b"corrupt")
    monkeypatch.setattr(s, "_one_day", lambda day, settings: pytest.fail("Must not silently redownload"))
    with pytest.raises(s.CleanFuelSourceError, match="Selected sibling seed is invalid"):
        s.materialize({}, "2026-09-11", "2026-09-17", fake_sources.parent / "2026-09-17")


def test_different_contract_seed_explicitly_skipped(fake_sources, monkeypatch):
    s.materialize({"sources": {"maximum_age_hours": 100}}, "2026-09-14", "2026-09-16", fake_sources.parent / "2026-09-16")
    calls = []
    monkeypatch.setattr(s, "_one_day", lambda day, settings: calls.append(day) or row(day))
    new = s.materialize({}, "2026-09-15", "2026-09-17", fake_sources.parent / "2026-09-17", workers=1)
    _, audit = s.load_bank(new)
    assert calls == ["2026-09-15", "2026-09-16", "2026-09-17"]
    assert audit["seed_reuse"]["copied_days"] == 0
    assert audit["seed_reuse"]["skipped"][0]["reason"] == "different_source_contract"


def test_seed_and_own_checkpoint_conflict_rejected(fake_sources):
    s.materialize({}, "2026-09-14", "2026-09-16", fake_sources.parent / "2026-09-16")
    next_dir = fake_sources.parent / "2026-09-17"
    daily = next_dir / "daily"
    daily.mkdir(parents=True)
    known = row("2026-09-15")
    known["ccc"] += 1
    (daily / "2026-09-15.json").write_bytes(s._json_bytes({"contract_sha256": s._sha(s._json_bytes(s._contract(176.0))),
                                                          "row": known, "row_sha256": s._sha(s._json_bytes(known))}))
    with pytest.raises(s.CleanFuelSourceError, match="checkpoint/seed values conflict"):
        s.materialize({}, "2026-09-15", "2026-09-17", next_dir)


def test_equal_seed_and_existing_checkpoint_can_resume(fake_sources):
    s.materialize({}, "2026-09-14", "2026-09-16", fake_sources.parent / "2026-09-16")
    next_dir = fake_sources.parent / "2026-09-17"
    daily = next_dir / "daily"
    daily.mkdir(parents=True)
    known = row("2026-09-15")
    (daily / "2026-09-15.json").write_bytes(s._json_bytes({"contract_sha256": s._sha(s._json_bytes(s._contract(176.0))),
                                                          "row": known, "row_sha256": s._sha(s._json_bytes(known))}))
    new = s.materialize({}, "2026-09-15", "2026-09-17", next_dir)
    assert s.load_bank(new)[1]["seed_reuse"]["copied_days"] == 1


def test_active_sibling_not_touched(fake_sources, monkeypatch):
    seed = s.materialize({}, "2026-09-14", "2026-09-16", fake_sources.parent / "2026-09-16")
    lock = seed.parent / ".materialize.lock"
    lock.write_text("active")
    calls = []
    monkeypatch.setattr(s, "_one_day", lambda day, settings: calls.append(day) or row(day))
    new = s.materialize({}, "2026-09-15", "2026-09-17", fake_sources.parent / "2026-09-17", workers=1)
    _, audit = s.load_bank(new)
    assert len(calls) == 3
    assert audit["seed_reuse"]["skipped"][0]["reason"] == "collection_active"
    assert lock.read_text() == "active"


def test_seed_mutation_during_collection_refuses_publication(fake_sources, monkeypatch):
    seed = s.materialize({}, "2026-09-14", "2026-09-16", fake_sources.parent / "2026-09-16")
    def mutate(day, settings):
        seed.write_bytes(seed.read_bytes() + b"concurrent change")
        return row(day)
    monkeypatch.setattr(s, "_one_day", mutate)
    new_dir = fake_sources.parent / "2026-09-17"
    with pytest.raises(s.CleanFuelSourceError, match="bank changed during collection"):
        s.materialize({}, "2026-09-15", "2026-09-17", new_dir)
    assert not (new_dir / "bank.parquet").exists()


def test_sibling_symlink_rejected(fake_sources, tmp_path):
    parent = fake_sources.parent
    parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (parent / "2026-09-16").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Windows symlink privileges unavailable")
    with pytest.raises(s.CleanFuelSourceError, match="symlinks"):
        s.materialize({}, "2026-09-15", "2026-09-17", parent / "2026-09-17")
