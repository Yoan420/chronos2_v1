from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chronos2_modular.hourly_comparison import (
    build_hourly_comparison_html, hourly_comparison_payload,
)


CONTRACT = {"id": "storm_official", "report_label": "Storm", "report_note": "Cache day-ahead apparié."}


def _result():
    return SimpleNamespace(
        zone="FR", statistics_candidate_label="Autonomous nucléaire",
        zone_data=SimpleNamespace(target=pd.Series(
            [1.0], index=pd.date_range("2026-01-01", periods=1, tz="Europe/Paris"),
        )),
    )


def _source(times, actual, candidate, benchmark, *, local=True):
    utc = pd.to_datetime(times, utc=True)
    frame = pd.DataFrame({
        "_timestamp_utc": utc, "actual": actual, "q50": candidate,
        "_benchmark_q50": benchmark,
    })
    if local:
        frame["_timestamp_local"] = utc.tz_convert("Europe/Paris")
    return frame


def test_hourly_mae_is_not_absolute_average_bias_and_prices_are_paired():
    source = _source(
        ["2026-01-01T11:00Z", "2026-01-02T11:00Z"],
        [100, 100], [90, 110], [104, 104],
    )
    before = source.copy(deep=True)
    output = hourly_comparison_payload(_result(), source=source, benchmark_contract=CONTRACT)
    row = output["records"][12]
    assert row["candidate_mae"] == 10
    assert row["benchmark_mae"] == 4
    assert row["mae_gain"] == -6
    assert row["candidate_bias"] == 0
    assert row["benchmark_bias"] == 4
    assert row["candidate_mean"] == row["actual_mean"] == 100
    assert row["benchmark_mean"] == 104
    assert row["paired_hours"] == row["paired_days"] == row["expected_hours"] == 2
    assert output["paired_hours"] == 2 and output["expected_hours"] == 48
    pd.testing.assert_frame_equal(source, before)
    json.dumps(output, allow_nan=False)


def test_nonfinite_observations_and_forecasts_are_excluded_from_every_average():
    source = _source(
        pd.date_range("2026-01-01T11:00Z", periods=5, freq="D"),
        [100, np.nan, 50, 70, -10], [110, 999, np.inf, 71, -12], [120, 999, 55, np.nan, -8],
    )
    output = hourly_comparison_payload(_result(), source=source, benchmark_contract=CONTRACT)
    row = output["records"][12]
    assert output["paired_hours"] == 2
    assert output["observed_hours"] == 4
    assert row["actual_mean"] == 45
    assert row["candidate_mean"] == 49
    assert row["benchmark_mean"] == 56
    assert row["candidate_mae"] == 6 and row["benchmark_mae"] == 11
    assert row["mae_gain"] == 5
    assert output["records"][0]["candidate_mae"] is None


@pytest.mark.parametrize("day,expected,occurrences", [
    ("2025-03-30", 23, 0), ("2025-10-26", 25, 2), ("2025-01-15", 24, 1),
])
def test_dst_counts_physical_folds_and_does_not_invent_missing_hour(day, expected, occurrences):
    start = pd.Timestamp(day)
    times = pd.date_range(
        start.tz_localize("Europe/Paris"),
        (start + pd.Timedelta(days=1)).tz_localize("Europe/Paris"),
        freq="h", inclusive="left",
    )
    source = _source(times, 10, 11, 13)
    output = hourly_comparison_payload(_result(), source=source, benchmark_contract=CONTRACT)
    assert output["expected_hours"] == output["paired_hours"] == expected
    assert output["coverage"] == 1
    assert output["records"][2]["expected_hours"] == occurrences
    assert output["records"][2]["paired_hours"] == occurrences
    assert output["records"][2]["paired_days"] == (1 if occurrences else 0)
    assert len(output["records"]) == 24


def test_no_independent_rolling_window_or_latest_date_shift():
    # Date filtering belongs exclusively to _statistics_source, not this view.
    times = pd.date_range("2025-01-01T11:00Z", periods=370, freq="D")
    source = _source(times, 10, 11, 13)
    source.loc[len(source) - 1, "actual"] = np.nan
    output = hourly_comparison_payload(_result(), source=source, benchmark_contract=CONTRACT)
    assert output["calendar_days"] == 370
    assert output["source_hours"] == 370
    assert output["paired_hours"] == 369
    assert output["end_day"] == "2026-01-05"


def test_timezone_falls_back_to_result_target_index_only_when_needed():
    source = _source(["2026-06-01T10:00Z"], [1], [2], [3], local=False)
    output = hourly_comparison_payload(_result(), source=source, benchmark_contract=CONTRACT)
    assert output["timezone"] == "Europe/Paris"
    assert output["records"][12]["paired_hours"] == 1


def test_local_timezone_in_statistics_takes_precedence():
    source = _source(["2026-06-01T10:00Z"], [1], [2], [3])
    source["_timestamp_local"] = source["_timestamp_utc"].dt.tz_convert("UTC")
    output = hourly_comparison_payload(_result(), source=source, benchmark_contract=CONTRACT)
    assert output["records"][10]["paired_hours"] == 1


@pytest.mark.parametrize("problem", ["duplicate", "nonhourly", "inconsistent_local", "naive_local", "nat"])
def test_invalid_timeline_is_refused(problem):
    source = _source(["2026-01-01T00:00Z", "2026-01-01T01:00Z"], 1, 2, 3)
    if problem == "duplicate":
        source.loc[1] = source.loc[0]
    elif problem == "nonhourly":
        source["_timestamp_utc"] += pd.Timedelta(minutes=15)
    elif problem == "inconsistent_local":
        source["_timestamp_local"] += pd.Timedelta(hours=1)
    elif problem == "naive_local":
        source["_timestamp_local"] = source["_timestamp_local"].dt.tz_localize(None)
    elif problem == "nat":
        source.loc[0, "_timestamp_utc"] = pd.NaT
    with pytest.raises(ValueError):
        hourly_comparison_payload(_result(), source=source, benchmark_contract=CONTRACT)


def test_no_benchmark_returns_honest_empty_state():
    source = _source(["2026-01-01T00:00Z"], [1], [2], [np.nan])
    rendered = build_hourly_comparison_html(_result(), source=source, benchmark_contract=CONTRACT)
    assert "Comparaison horaire indisponible" in rendered
    assert "Plotly.newPlot" not in rendered
    assert "remplacée par zéro" in rendered


def test_empty_frame_is_supported():
    source = _source([], [], [], [])
    output = hourly_comparison_payload(_result(), source=source, benchmark_contract=CONTRACT)
    assert output["paired_hours"] == 0 and output["coverage"] is None
    assert len(output["records"]) == 24
    assert "indisponible" in build_hourly_comparison_html(_result(), source=source, benchmark_contract=CONTRACT)


def test_html_has_plotly_controls_theme_and_counts_without_network():
    source = _source(["2026-01-01T00:00Z"], [1], [2], [3])
    rendered = build_hourly_comparison_html(_result(), source=source, benchmark_contract=CONTRACT)
    assert "Plotly.newPlot" in rendered
    assert "cdn.plot.ly" not in rendered
    for label in ("erreur absolue moyenne", "Prix moyens", "Biais "):
        assert label in rendered
    assert "chronos2-theme-change" in rendered
    assert "updatemenus[0].font.color" in rendered
    assert "1/24 heures" in rendered
    assert "appariées (4.2 %)" in rendered
    assert "les deux occurrences de 02 h" in rendered
    assert "Cache day-ahead apparié." in rendered


def test_untrusted_labels_are_escaped_and_ids_are_unique():
    result = _result()
    attack = '</script><img src=x onerror="alert(1)">'
    result.statistics_candidate_label = attack
    result.zone = attack
    contract = {"report_label": attack, "report_note": attack}
    source = _source(["2026-01-01T00:00Z"], [1], [2], [3])
    first = build_hourly_comparison_html(result, source=source, benchmark_contract=contract)
    second = build_hourly_comparison_html(result, source=source, benchmark_contract=contract)
    assert attack not in first and "<img" not in first
    assert first != second
