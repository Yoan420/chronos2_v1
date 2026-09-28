"""Daily orchestration tests do not make network requests."""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import pandas as pd
import pytest

import run_nyx_annual_daily_capture as daily


def _at(local: str) -> pd.Timestamp:
    return pd.Timestamp(local, tz="Europe/Paris").tz_convert("UTC")


@pytest.mark.parametrize("local,day", [
    ("2026-09-29 07:00", date(2026, 9, 30)),
    ("2026-10-24 07:00", date(2026, 10, 25)),
    ("2026-10-25 07:00", date(2026, 10, 26)),
])
def test_target_is_tomorrow_in_paris_across_dst(local: str, day: date) -> None:
    assert daily.target_day(_at(local)) == day


@pytest.mark.parametrize("local", ["2026-09-29 01:14:59",
                                   "2026-09-29 08:00:00"])
def test_outside_window_stops_before_network(monkeypatch, tmp_path, local):
    def forbidden(*args, **kwargs):
        raise AssertionError("A source was contacted outside the capture window")

    monkeypatch.setattr(daily, "_tls_configuration", forbidden)
    monkeypatch.setattr(daily.exchange_source, "capture", forbidden)
    with pytest.raises(ValueError, match="01:15–08:00"):
        daily.capture_daily(now_utc=_at(local), jao_cache=tmp_path / "jao",
                            exchange_archive=tmp_path / "exchange")


def test_requested_historical_day_is_rejected_before_sources(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("A historical day was contacted")

    monkeypatch.setattr(daily, "_tls_configuration", forbidden)
    monkeypatch.setattr(daily.exchange_source, "capture", forbidden)
    with pytest.raises(ValueError, match="Only tomorrow"):
        daily.capture_daily(now_utc=_at("2026-09-29 07:00"),
                            requested_day=date(2026, 9, 29),
                            jao_cache=tmp_path / "jao",
                            exchange_archive=tmp_path / "exchange")


class _Client:
    def __init__(self, *, verify):
        assert verify is True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_runs_both_existing_collectors_under_their_locks(monkeypatch, tmp_path):
    jao_cache = tmp_path / "jao"
    archive = tmp_path / "exchange"
    calls = []
    monkeypatch.setattr(daily, "_tls_configuration",
                        lambda args: (True, "verified_test_ca"))
    monkeypatch.setattr(daily, "JaoCoreClient", _Client)

    def jao_capture(*, day, cache_root, client, now_utc, tls_trust_source):
        assert day == date(2026, 9, 30)
        assert cache_root == jao_cache.resolve()
        assert isinstance(client, _Client)
        assert now_utc is None  # Live collectors must use actual retrieval time.
        assert tls_trust_source == "verified_test_ca"
        assert (jao_cache / "jao_live_capture.lock").is_file()
        calls.append("jao")
        return {"retrieved_at_utc": _at("2026-09-29 07:00").isoformat()}

    def exchange_capture(day, root, *, now_utc):
        assert day == "2026-09-30"
        assert root == archive.resolve()
        assert now_utc is None
        assert (archive / "exchange_source.lock").is_file()
        calls.append("exchange")
        return archive / day / "capture.json"

    monkeypatch.setattr(daily, "capture_day", jao_capture)
    monkeypatch.setattr(daily.exchange_source, "capture", exchange_capture)
    monkeypatch.setattr(daily, "_clock", lambda now: _at("2026-09-29 07:00"))
    outcome = daily.capture_daily(jao_cache=jao_cache,
                                  exchange_archive=archive)
    assert calls == ["jao", "exchange"]
    assert outcome["state"] == "COMPLETE"
    assert outcome["model_inputs_complete"] is False
    assert outcome["forecast_enabled"] is False
    assert not (jao_cache / "jao_live_capture.lock").exists()
    assert not (archive / "exchange_source.lock").exists()


def test_exchange_runs_even_if_jao_fails(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(daily, "_tls_configuration",
                        lambda args: (True, "verified_test_ca"))
    monkeypatch.setattr(daily, "JaoCoreClient", _Client)

    def jao_failed(**kwargs):
        calls.append("jao")
        raise ValueError("JAO initial not published")

    def exchange_capture(day, archive, *, now_utc):
        calls.append("exchange")
        return Path(archive) / day / "capture.json"

    monkeypatch.setattr(daily, "capture_day", jao_failed)
    monkeypatch.setattr(daily.exchange_source, "capture", exchange_capture)
    result = daily.capture_daily(now_utc=_at("2026-09-29 07:00"),
                                 jao_cache=tmp_path / "jao",
                                 exchange_archive=tmp_path / "exchange")
    assert calls == ["jao", "exchange"]
    assert result["state"] == "ERROR"
    assert result["sources"]["jao_initial"]["state"] == "ERROR"
    assert result["sources"]["lagged_exchange"]["state"] == "COMPLETE"


def test_cli_logs_each_scheduled_attempt_and_failure(monkeypatch, tmp_path, capsys):
    log = tmp_path / "capture.jsonl"

    def failed(**kwargs):
        raise ValueError("JAO window already closed")

    monkeypatch.setattr(daily, "capture_daily", failed)
    assert daily.main(["--delivery-day", "2026-09-30", "--log-file", str(log)]) == 1
    entry = json.loads(log.read_text(encoding="utf-8"))
    assert entry["delivery_day"] == "2026-09-30"
    assert entry["state"] == "ERROR"
    assert "window already closed" in entry["error"]
    assert entry["forecast_enabled"] is False
    assert json.loads(capsys.readouterr().out) == entry


def test_verify_only_checks_both_archives_without_network(monkeypatch, tmp_path):
    day = date(2026, 9, 30)
    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError("Verify-only contacted a provider")

    def jao_verify(root, selected):
        calls.append("jao")
        assert selected == day
        return {"retrieved_at_utc": _at("2026-09-29 07:00").isoformat()}

    def exchange_verify(root, selected):
        calls.append("exchange")
        assert selected == "2026-09-30"
        return None, {"retrieved_at_utc": _at("2026-09-29 07:01").isoformat()}

    monkeypatch.setattr(daily, "_tls_configuration", forbidden)
    monkeypatch.setattr(daily, "capture_day", forbidden)
    monkeypatch.setattr(daily.exchange_source, "capture", forbidden)
    monkeypatch.setattr(daily, "verify_daily_capture", jao_verify)
    monkeypatch.setattr(daily.exchange_source, "_verify_capture", exchange_verify)
    result = daily.verify_daily(day, jao_cache=tmp_path / "jao",
                                exchange_archive=tmp_path / "exchange")
    assert calls == ["jao", "exchange"]
    assert result["action"] == "verify"
    assert result["state"] == "COMPLETE"
    assert result["forecast_enabled"] is False
