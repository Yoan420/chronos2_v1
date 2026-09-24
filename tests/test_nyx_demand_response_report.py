"""A synthetic mechanism must never masquerade as a qualified yearly forecast."""
import copy
import json
import re
import pytest

from nyx_demand_response.report import EXPERT_ID, render_report


@pytest.fixture
def payload():
    return {"schema_version": 1, "snapshot": "isolated/snapshot", "source_report": "frozen/source.html",
        "period": {"start_day": "2025-09-15", "end_day": "2026-09-14", "days": 365},
        "decision": {"integration_ready": False, "reason": "Insufficient evidence.",
            "qualified_country_hours": 0, "total_country_hours": 35040, "interventions": 0,
            "empirical_gain_demonstrated": False},
        "blockers": [{"id": "demand_curve", "detail": "No historical purchase bids."}],
        "sources": [{"name": "Source", "status": "unqualified", "detail": "Diagnostic only."}],
        "kpi_rows": [{"zone": z, "model_id": model, "mae_eur_mwh": 11., "rmse_eur_mwh": 22., "n_hours": 8760}
                     for z in ("ALL", "FR", "DE", "BE", "NL") for model in ("nuclear_kalman", "__storm__")],
        "cases": [{"zone": "DE", "timestamp_utc": "2026-09-14T17:00:00Z", "actual": 697.,
            "storm": 420., "nyx": 322., "expert": None, "status": "unqualified"}],
        "demos": [{"label": "Hypothesis", "demand_mw": 1000., "supply_mw": 900., "price_eur_mwh": 350.,
            "voluntary_reduction_mw": 100., "status": "synthetic"}],
        "sensitivity": [{"demand_mw": 1000., "price_eur_mwh": 350., "voluntary_reduction_mw": 100.}], "audit": {}}


def embedded(path):
    return json.loads(re.search(r'<script id="demand-data" type="application/json">(.*?)</script>',
                               path.read_text(encoding="utf8"), re.S).group(1))


def test_unqualified_expert_empty_not_fallback_and_original_unchanged(payload, tmp_path):
    before = copy.deepcopy(payload)
    report = render_report(payload, tmp_path/"report.html")
    data = embedded(report)
    expert = [r for r in data["kpi_rows"] if r["model_id"] == EXPERT_ID]
    assert len(expert) == 5
    assert all(r["n_hours"] == 0 and r["mae_eur_mwh"] is None and r["rmse_eur_mwh"] is None for r in expert)
    assert data["cases"][0]["expert"] is None
    assert data["decision"]["empirical_gain_demonstrated"] is False
    assert payload == before
    text = report.read_text(encoding="utf8")
    assert "DÉMONSTRATION SYNTHÉTIQUE" in text and "gain empirique" in text
    assert "pas zéro erreur" in text and "350 / 650" in text
    assert '<script src=' not in text and 'fetch(' not in text
    assert 'id="theme"' in text and 'id="zone"' in text
    assert 'Europe/Paris' in text and '08 h' in text


@pytest.mark.parametrize("damage", ["gain", "activation", "intervention", "real_price", "fallback_kpi", "negative", "too_many", "short_window", "wrong_dates", "duplicate", "naive", "infinity"])
def test_fail_closed_claims(payload, tmp_path, damage):
    if damage == "gain": payload["decision"]["empirical_gain_demonstrated"] = True
    elif damage == "activation": payload["decision"]["integration_ready"] = True
    elif damage == "intervention": payload["decision"]["interventions"] = 1
    elif damage == "real_price": payload["cases"][0]["expert"] = 322.
    elif damage == "fallback_kpi": payload["kpi_rows"].append({**payload["kpi_rows"][0], "model_id": EXPERT_ID})
    elif damage == "negative": payload["decision"]["qualified_country_hours"] = -1
    elif damage == "too_many": payload["decision"]["qualified_country_hours"] = 35041
    elif damage == "short_window": payload["period"]["days"] = 364
    elif damage == "wrong_dates": payload["period"]["start_day"] = "2025-09-14"
    elif damage == "duplicate": payload["kpi_rows"].append(payload["kpi_rows"][0].copy())
    elif damage == "naive": payload["cases"][0]["timestamp_utc"] = "2026-09-14T19:00:00"
    else: payload["demos"][0]["price_eur_mwh"] = float("inf")
    with pytest.raises(ValueError):
        render_report(payload, tmp_path/"rejected.html")
    assert not (tmp_path/"rejected.html").exists()


def test_html_injection_safe_missing_values_and_exclusive_write(payload, tmp_path):
    payload["decision"]["reason"] = '</script><script>alert("x")</script>&\u2028'
    payload["sources"][0]["url"] = "javascript:alert(1)"
    payload["kpi_rows"][0]["mae_eur_mwh"] = float("nan")
    target = render_report(payload, tmp_path/"report.html")
    text = target.read_text(encoding="utf8")
    assert '</script><script>alert' not in text and '\\u003c/script\\u003e' in text
    assert embedded(target)["kpi_rows"][0]["mae_eur_mwh"] is None
    assert 'textContent' in text and 'innerHTML' not in text
    assert '^https?:' in text
    with pytest.raises(FileExistsError):
        render_report(payload, target)


def test_qualified_bundle_comparison_is_separate_and_paired(payload, tmp_path):
    payload["decision"]["qualified_country_hours"] = 4
    rows = [{"zone": z, "model_id": m, "n_hours": 1, "mae_eur_mwh": 12., "rmse_eur_mwh": 12.}
            for z in ("FR", "DE", "BE", "NL") for m in (EXPERT_ID, "nuclear_kalman", "__storm__")]
    payload["paired_expert_evaluation"] = {"period": payload["period"].copy(), "rows": rows}
    target = render_report(payload, tmp_path/"paired.html")
    data = embedded(target)
    assert data["paired_expert_evaluation"]["rows"] == rows
    assert not any(r["model_id"] == EXPERT_ID for r in data["kpi_rows"])
    text = target.read_text(encoding="utf8")
    assert 'id="paired-kpi"' in text and 'id="paired-status"' in text
    assert "non validée indépendamment" in text and "Aucune évaluation expert possible" in text
    assert "Ne pas comparer directement" in text


def test_long_source_path_wraps_without_global_table_clipping(payload, tmp_path):
    payload["source_report"] = "C:/research/"+"a"*500+"/snapshot"
    target = render_report(payload, tmp_path/"mobile.html")
    text = target.read_text(encoding="utf8")
    assert '#kpi-support{overflow-wrap:anywhere;word-break:break-word}</style>' in text
    assert '.scroll{overflow-x:auto}' in text
    assert embedded(target)["source_report"] == payload["source_report"]


def test_synthetic_graphs_only_show_evaluated_points(payload, tmp_path):
    target = render_report(payload, tmp_path/"points.html")
    text = target.read_text(encoding="utf8")
    assert "Points testés, sans interpolation" in text
    assert "el('path'" not in text and "let path=''" not in text
    assert "el('circle'" in text and "r:4,fill:c" in text
    assert "pointermove" in text and "Scénario uniquement" in text


@pytest.mark.parametrize("damage", ["no_qualified", "different_count", "different_window", "duplicate"])
def test_paired_evaluation_rejects_inconsistent_support(payload, tmp_path, damage):
    payload["decision"]["qualified_country_hours"] = 2
    rows = [{"zone": "FR", "model_id": m, "n_hours": 2} for m in (EXPERT_ID, "nuclear_kalman")]
    paired = {"period": payload["period"].copy(), "rows": rows}
    payload["paired_expert_evaluation"] = paired
    if damage == "no_qualified": payload["decision"]["qualified_country_hours"] = 0
    elif damage == "different_count": rows[1]["n_hours"] = 3
    elif damage == "different_window": paired["period"]["days"] = 7
    else: rows.append(rows[0].copy())
    with pytest.raises(ValueError): render_report(payload, tmp_path/"rejected.html")
