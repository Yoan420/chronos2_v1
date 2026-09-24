from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nyx_clean_fuel.report import _build_payload, render_report


def _panel(start="2025-09-18", end="2026-09-18", zones=("FR",)):
    index = pd.date_range(pd.Timestamp(start, tz="Europe/Paris"), pd.Timestamp(end, tz="Europe/Paris"),
                          freq="h", inclusive="left").tz_convert("UTC")
    parts = []
    for zone in zones:
        for model, offset in (("baseline", 4), ("clean_fuel", 2)):
            actual = np.full(len(index), 90.0)
            actual[5] = 400
            parts.append(pd.DataFrame({"delivery_start_utc": index, "zone": zone, "actual": actual,
                "model": model, "q10": actual + offset - 10, "q50": actual + offset,
                "q90": actual + offset + 10, "storm_q50": actual + 3, "phase": "history"}))
    return pd.concat(parts, ignore_index=True)


def test_annual_complete_paired_metrics():
    result = _build_payload(_panel(zones=("FR", "DE")), {})
    assert result["days"] == 365
    view = result["views"]["ALL"]
    clean = next(r for r in view["scores"] if r["model"] == "clean_fuel")
    assert clean["hours"] == 2 * 8760
    assert clean["mae"] == clean["rmse"] == 2
    assert clean["win_rate"] == 100
    assert clean["spike_hours"] == 2
    assert clean["spike_rmse"] == 2


def test_drop_one_hour_refused():
    panel = _panel().drop(index=3)
    with pytest.raises(ValueError, match="complete|different comparison"):
        _build_payload(panel, {})


def test_different_actual_or_storm_refused():
    for column in ("actual", "storm_q50"):
        panel = _panel()
        panel.loc[panel.model.eq("clean_fuel"), column] += 1
        with pytest.raises(ValueError, match="inconsistent"):
            _build_payload(panel, {})


def test_dst_and_partial_requires_explicit_diagnostic():
    panel = _panel(start="2025-10-25", end="2025-10-27")
    with pytest.raises(ValueError, match="complete"):
        _build_payload(panel, {})
    payload = _build_payload(panel, {"allow_partial_evaluation": True})
    assert payload["days"] == 2
    assert payload["views"]["FR"]["expected_hours"] == 49


def test_trim_warmup_and_storm_missing():
    panel = _panel(start="2024-09-18")
    panel["storm_q50"] = np.nan
    result = _build_payload(panel, {})
    assert result["start_day"] == "2025-09-18"
    assert len(result["views"]["FR"]["scores"]) == 2
    assert result["views"]["FR"]["scores"][0]["win_rate"] is None


def test_injection_escaped_and_self_contained(tmp_path: Path):
    panel = _panel()
    panel.loc[panel.model.eq("clean_fuel"), "model"] = "</script><script>alert(1)</script>"
    output = render_report(panel, tmp_path / "report.html", {"title": "<unsafe>"})
    text = output.read_text(encoding="utf-8")
    assert "<unsafe>" not in text
    assert "</script><script>alert(1)</script>" not in text
    assert "\\u003c/script\\u003e" in text
    assert '<script src=' not in text


def test_future_profile_and_correction():
    panel = _panel()
    future = _panel(start="2026-09-18", end="2026-09-19")
    future["actual"] = np.nan
    future["phase"] = "future"
    result = _build_payload(pd.concat([panel, future]), {})
    profiles = result["views"]["FR"]["future"]
    assert len(profiles) == 48
    clean = [r for r in profiles if r["model"] == "clean_fuel"]
    assert all(r["correction"] == -2 for r in clean)
    assert all(r["actual"] is None for r in clean)


def test_future_missing_one_model_refused():
    panel = _panel()
    future = _panel(start="2026-09-18", end="2026-09-19")
    future = future.loc[future.model.eq("baseline")]
    future["phase"] = "future"
    with pytest.raises(ValueError, match="different comparison"):
        _build_payload(pd.concat([panel, future]), {})


def test_storm_metrics_use_strict_common_hours():
    panel = _panel()
    first_hour = panel.delivery_start_utc.min()
    excluded = panel.delivery_start_utc.eq(first_hour)
    panel.loc[excluded, "storm_q50"] = np.nan
    for quantile in ("q10", "q50", "q90"):
        panel.loc[excluded, quantile] += 100
    view = _build_payload(panel, {})["views"]["FR"]
    assert view["expected_hours"] == 8760
    assert view["storm_hours"] == 8759
    assert {row["hours"] for row in view["paired_scores"]} == {8759}
    assert next(row for row in view["paired_scores"] if row["model"] == "clean_fuel")["mae"] == 2
    assert next(row for row in view["scores"] if row["model"] == "clean_fuel")["mae"] > 2


def test_model_family_references_for_metrics_and_future():
    history = _panel()
    future = _panel(start="2026-09-18", end="2026-09-19")
    future["phase"] = "future"
    source = pd.concat([history, future], ignore_index=True)
    kalman = source.copy()
    for model, offset in (("baseline", 1), ("clean_fuel", 1.5)):
        mask = kalman.model.eq(model)
        for quantile, delta in (("q10", -10), ("q50", 0), ("q90", 10)):
            kalman.loc[mask, quantile] = kalman.loc[mask, "actual"] + offset + delta
    kalman["model"] = kalman.model.map({"baseline": "nuclear_kalman", "clean_fuel": "clean_fuel_kalman"})
    result = _build_payload(pd.concat([source, kalman], ignore_index=True), {
        "baseline_by_model": {"clean_fuel": "baseline", "clean_fuel_kalman": "nuclear_kalman"},
    })
    row = next(row for row in result["views"]["FR"]["paired_scores"] if row["model"] == "clean_fuel_kalman")
    assert row["reference_model"] == "nuclear_kalman"
    assert row["delta_mae"] == 0.5
    assert result["baseline_by_model"]["nuclear_kalman"] == "nuclear_kalman"
    forecast = [row for row in result["views"]["FR"]["future"] if row["model"] == "clean_fuel_kalman"]
    assert all(row["correction"] == 0.5 for row in forecast)


def test_unknown_family_reference_refused():
    with pytest.raises(ValueError, match="baseline_by_model"):
        _build_payload(_panel(), {"baseline_by_model": {"clean_fuel": "absent"}})
