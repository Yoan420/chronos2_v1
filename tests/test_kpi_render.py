from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import shutil
import subprocess

import pandas as pd
import pytest

from kpi_report.metrics import compute_kpis
from kpi_report.render import render_kpi
from kpi_report.runner import _coverage_by_zone


@pytest.fixture
def payload():
    hours = pd.date_range("2026-09-14", "2026-09-15", freq="h", inclusive="left", tz="Europe/Paris").tz_convert("UTC")
    frame = pd.DataFrame([
        {"model_id": model, "zone": zone, "timestamp_utc": hour,
         "actual": 0., "forecast": (0. if model == "nyx" else -2.), "storm": 1.}
        for zone in ("FR", "BE") for model in ("nyx", "trial") for hour in hours
    ])
    periods = {}
    for days in (365, 90, 30, 7):
        result = compute_kpis(frame, end_day="2026-09-14", days=days)
        result["coverage"] = _coverage_by_zone(result)
        result.pop("daily_rows")
        economic_rows = []
        for row in result["rows"]:
            gain = 0. if row["model_id"] == "__storm__" else 10. if row["model_id"] == "nyx" else -30.
            economic_rows.append({"zone": row["zone"], "model_id": row["model_id"],
                "pnl_net_eur": 90. + gain, "gain_vs_storm_eur": gain,
                "gain_vs_storm_per_potential_mwh": gain / 1200.,
                "n_country_hours": row["n_hours"], "potential_energy_mwh": 1200.})
        result["economic"] = {"rows": economic_rows, "coverage": [], "audit": {
            "portfolio_capacity_mw": 100., "zone_capacity_mw": {"FR": 25., "BE": 25., "DE": 25., "NL": 25.},
            "signal_hurdle_eur_mwh": 6., "net_cost_eur_mwh": 1., "executable_reference": False}}
        periods[str(days)] = result
    return {"catalog": [
        {"id": "nyx", "label": "NYX", "kind": "production", "family": "production", "source_path": "some/local.parquet"},
        {"id": "trial", "label": "<script>alert('not executable')</script>", "kind": "experiment", "family": "fundamental", "source_path": "</li><img onerror='bad' src=x>"},
    ], "zones": ["FR", "BE"], "periods": periods, "source_delivery_day": "2026-09-15", "generated_at_local": "15/09/2026"}


def test_embedded_json_preserves_data_and_escapes_script_injection(payload, tmp_path):
    path = render_kpi(payload, tmp_path / "KPI.html")
    html = path.read_text(encoding="utf-8")
    match = re.search(r'<script id="kpi-data" type="application/json">(.*?)</script>', html, re.DOTALL)
    assert json.loads(match[1]) == payload
    assert "<script>alert('not executable')</script>" not in html
    assert "<img onerror=" not in html
    assert "\\u003cscript>" in match[1]
    assert "&lt;img onerror=" in html
    assert '<script src=' not in html
    assert '<link ' not in html
    assert not re.search(r'\b(fetch|XMLHttpRequest|WebSocket)\s*\(', html)


def test_render_does_not_mutate_payload_or_overwrite_existing_file(payload, tmp_path):
    original = copy.deepcopy(payload)
    path = render_kpi(payload, tmp_path / "KPI.html")
    before = path.read_bytes()
    assert payload == original
    with pytest.raises(FileExistsError):
        render_kpi(payload, path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("key", ["catalog", "periods"])
def test_missing_data_rejected_before_writing(payload, tmp_path, key):
    payload[key] = [] if key == "catalog" else {}
    with pytest.raises(ValueError, match="catalog and computed periods"):
        render_kpi(payload, tmp_path / "KPI.html")
    assert not (tmp_path / "KPI.html").exists()


def test_nonfinite_json_rejected_instead_of_silent_null(payload, tmp_path):
    payload["periods"]["365"]["rows"][0]["mae_eur_mwh"] = float("nan")
    with pytest.raises(ValueError):
        render_kpi(payload, tmp_path / "KPI.html")
    assert not (tmp_path / "KPI.html").exists()


def test_unicode_separators_cannot_break_embedded_script(payload, tmp_path):
    payload["catalog"][0]["label"] = "A\u2028B\u2029C"
    html = render_kpi(payload, tmp_path / "KPI.html").read_text(encoding="utf-8")
    embedded = re.search(r'<script id="kpi-data" type="application/json">(.*?)</script>', html, re.DOTALL)[1]
    assert '\u2028' not in embedded
    assert '\u2029' not in embedded
    assert '\\u2028' in embedded
    assert json.loads(embedded)["catalog"][0]["label"] == "A\u2028B\u2029C"


def test_actual_javascript_ui_json_parity_filters_sort_theme_zero_ties(payload, tmp_path):
    bundled = Path("C:/Users/BQ6757/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe")
    node = shutil.which("node") or (str(bundled) if bundled.is_file() else None)
    if node is None:
        pytest.skip("Node required for actual JavaScript VM QA")
    path = render_kpi(payload, tmp_path / "KPI.html")
    qa = Path(__file__).resolve().parents[1] / "tmp/kpi_ui_qa.js"
    result = subprocess.run([node, str(qa), str(path)], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    audit = json.loads(result.stdout)
    assert audit["status"] == "passed"
    assert audit["checks"] >= 25
    assert audit["pixel_visual_qa"] is False


def test_optional_economic_absence_stays_valid(payload, tmp_path):
    for period in payload["periods"].values():
        period.pop("economic")
    bundled = Path("C:/Users/BQ6757/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe")
    node = shutil.which("node") or (str(bundled) if bundled.is_file() else None)
    if node is None:
        pytest.skip("Node required for actual JavaScript VM QA")
    path = render_kpi(payload, tmp_path / "KPI.html")
    qa = Path(__file__).resolve().parents[1] / "tmp/kpi_ui_qa.js"
    result = subprocess.run([node, str(qa), str(path)], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
