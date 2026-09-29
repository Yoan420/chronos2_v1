import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import diagnose_nyx_annual_saturn_targets as m


@pytest.fixture
def client(monkeypatch):
    closed = []
    value = SimpleNamespace(session=SimpleNamespace(request=lambda *a, **k: None, close=lambda: closed.append(True)))
    monkeypatch.setattr(m, "load_plan", lambda: {"saturn_url": "https://private-saturn.example", "saturn_author": "private-user"})
    monkeypatch.setattr(m, "create_saturn_client", lambda *a: value)
    return value, closed


def test_four_strict_queries_and_posthoc_comparison_never_qualifies(monkeypatch, client):
    calls = []
    def fetch(client, series, start, end, timezone, **kwargs):
        calls.append((series, start, end, timezone, kwargs))
        return pd.Series(100., index=pd.date_range(start, end, freq="h"))
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    report = m.diagnose("2026-09-30", "2024-06-18", "FR")
    assert len(calls) == 4 and client[1] == [True]
    assert client[0].session.request.keywords == {"timeout": 60}
    assert all(call[4]["incomplete_dst_policy"] == "raise" and call[4]["nocache"] is True for call in calls)
    assert [call[4]["revision_date"] for call in calls] == [m.cutoff("2024-06-18"), m.cutoff("2024-06-18"), m.cutoff("2026-09-30"), m.cutoff("2024-06-18")]
    assert calls[-1][4]["naive_timezone"] == "Europe/Paris" and calls[-1][4]["request_padding_hours"] == 3
    assert report["probes"]["canonical_old_context2048"]["required_hours"] == 2048
    assert report["posthoc_comparison"]["paired_hours"] == m.TARGET_CONTEXT_HOURS
    assert report["posthoc_comparison"]["max_abs_difference_eur_mwh"] == 0
    assert report["posthoc_comparison"]["qualification_valid"] is False
    assert report["forecast_published"] is False and report["source_admitted"] is False


def test_missing_old_vintage_continues_other_probes_and_masks_connection(monkeypatch, client):
    def fetch(client, series, start, end, timezone, **kwargs):
        if series == m.SERIES["FR"] and kwargs["revision_date"] == m.cutoff("2024-06-18"):
            raise RuntimeError("https://private-saturn.example private-user password=example-secret")
        return pd.Series(1., index=pd.date_range(start, end, freq="h"))
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    report = m.diagnose("2026-09-30", "2024-06-18")
    assert len(report["probes"]) == 4 and client[1] == [True]
    assert report["probes"]["canonical_old_long"]["state"] == "ERROR"
    assert report["probes"]["canonical_current_long"]["state"] == "COMPLETE"
    serialized = json.dumps(report)
    assert all(secret not in serialized for secret in ("private-user", "private-saturn.example", "example-secret"))
    assert "coupure actuelle" in report["notes"][0]


def test_short_range_success_and_nonfinite_counts(monkeypatch, client):
    def fetch(client, series, start, end, timezone, **kwargs):
        values = pd.Series(5., index=pd.date_range(start, end, freq="h"))
        if len(values) > 2048:
            values.iloc[:2] = np.nan
        return values
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", fetch)
    report = m.diagnose("2026-09-30", "2024-06-18")
    assert report["probes"]["canonical_old_long"]["missing_nonfinite_hours"] == 2
    assert report["probes"]["canonical_old_context2048"]["state"] == "COMPLETE"
    assert "plage courte" in report["notes"][0]


def test_cli_writes_only_unique_diagnostic_reports(monkeypatch, tmp_path, client, capsys):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "fetch_saturn_series_from_client", lambda c, s, start, end, tz, **k: pd.Series(17.123456789, index=pd.date_range(start, end, freq="h")))
    for _ in range(2):
        assert m.main([]) == 0
    files = list(tmp_path.rglob("*"))
    reports = [path for path in files if path.is_file()]
    assert len(reports) == 2 and all(path.name.startswith("saturn_targets_") for path in reports)
    assert all("17.123456789" not in path.read_text(encoding="utf-8") for path in reports)
    assert "Diagnostic à transmettre" in capsys.readouterr().out


def test_client_creation_failure_reports_no_connection_identifiers(monkeypatch, client):
    def create(*args):
        raise RuntimeError("https://private-saturn.example private-user password=example-secret")
    monkeypatch.setattr(m, "create_saturn_client", create)
    report = m.diagnose("2026-09-30", "2024-06-18")
    assert report["state"] == "ERROR" and not report["probes"]
    assert "example-secret" not in json.dumps(report)


def test_reversed_cutoffs_rejected_before_connecting(monkeypatch):
    monkeypatch.setattr(m, "load_plan", lambda: pytest.fail("reversed dates contacted Saturn"))
    with pytest.raises(ValueError, match="ordonnées"):
        m.diagnose("2024-06-18", "2026-09-30")
