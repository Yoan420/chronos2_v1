from __future__ import annotations

from datetime import date, timedelta
from html.parser import HTMLParser
import json
from pathlib import Path
import re

import pandas as pd
import pytest

from chronos2_hourly import model_storm_report as report


ZONES = (
    ("BE", "Belgium", "Europe/Brussels"),
    ("DE", "Germany", "Europe/Berlin"),
    ("FR", "France", "Europe/Paris"),
    ("NL", "The Netherlands", "Europe/Amsterdam"),
)


def _payload(day: str = "2026-09-10") -> dict:
    zones = []
    for code, name, timezone in ZONES:
        start = pd.Timestamp(day, tz=timezone)
        end = pd.Timestamp(date.fromisoformat(day) + timedelta(days=1), tz=timezone)
        hours = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
        zones.append({
            "zone": code,
            "name": name,
            "timezone": timezone,
            "expected_hours": len(hours),
            "sources": {key: {"status": "complete"} for key in ("observed", "storm", "model")},
            "rows": [
                {"timestamp_utc": stamp.isoformat(), "observed": float(i),
                 "storm": float(i) - 2.0, "model": float(i) + 4.0}
                for i, stamp in enumerate(hours)
            ],
        })
    return {"delivery_day": day, "generated_at": "2026-09-09T16:00:00+02:00", "zones": zones}


@pytest.fixture(autouse=True)
def _stub_plotly(monkeypatch):
    monkeypatch.setattr(report, "get_plotlyjs", lambda: "/* TEST_INLINE_PLOTLY */")


def _render(tmp_path: Path, payload: dict) -> str:
    output = tmp_path / "daily.html"
    assert report.render_model_storm_report(payload, output) == output.resolve()
    return output.read_text(encoding="utf-8")


def _figure(document: str) -> dict:
    encoded = re.search(r'<script id="chart-data" type="application/json">(.*?)</script>', document, re.S)
    assert encoded is not None
    return json.loads(encoded.group(1))


@pytest.mark.parametrize("day, expected", [("2026-03-29", 23), ("2026-09-10", 24), ("2026-10-25", 25)])
def test_daily_means_include_every_physical_hour(day, expected):
    zone = _payload(day)["zones"][0]
    assert zone["expected_hours"] == expected
    metrics = report.daily_metrics(zone)
    assert metrics["observed"] == pytest.approx((expected - 1) / 2)
    assert metrics["storm"] == pytest.approx((expected - 1) / 2 - 2)
    assert metrics["model"] == pytest.approx((expected - 1) / 2 + 4)


@pytest.mark.parametrize("key", ["observed", "storm", "model"])
@pytest.mark.parametrize("missing", [None, float("nan"), float("inf"), True])
def test_daily_mean_rejects_incomplete_or_invalid_series(key, missing):
    zone = _payload()["zones"][0]
    zone["rows"][5][key] = missing
    metrics = report.daily_metrics(zone)
    assert metrics[key] is None
    for available in {"observed", "storm", "model"} - {key}:
        assert metrics[available] is not None
    if key in ("storm", "model"):
        assert metrics["models_mean"] is None
        assert metrics["models_std"] is None


def test_daily_mean_rejects_a_missing_hour_even_with_all_remaining_values_present():
    zone = _payload()["zones"][0]
    zone["rows"].pop(4)
    assert all(value is None for value in report.daily_metrics(zone).values())


def test_models_standard_deviation_is_population_std_of_the_two_daily_means():
    zone = _payload()["zones"][0]
    for index, row in enumerate(zone["rows"]):
        row.update(storm=10.0 + (-8 if index % 2 else 8), model=14.0)
    metrics = report.daily_metrics(zone)
    assert metrics["storm"] == 10.0
    assert metrics["model"] == 14.0
    assert metrics["models_mean"] == 12.0
    assert metrics["models_std"] == 2.0


def test_zero_and_negative_prices_are_real_values_with_signed_forecast_errors(tmp_path):
    payload = _payload()
    for zone in payload["zones"]:
        for row in zone["rows"]:
            row.update(observed=0.0, storm=-2.0, model=2.0)
    document = _render(tmp_path, payload)
    figure = _figure(document)
    assert ">0.00 €/MWh</strong>" in document
    assert "↓ 2.00" in document
    assert "↑ 2.00" in document
    assert "0.00 ± 2.00 €/MWh" in document
    assert figure["data"][0]["y"] == [0.0] * 24
    errors = [trace for trace in figure["data"] if " error" in trace["hovertemplate"]]
    assert all(trace["y"] == ([-2.0] * 24 if trace["name"] == "Storm" else [2.0] * 24) for trace in errors)


def test_only_requested_series_have_one_shared_legend_and_errors_do_not_bridge_gaps(tmp_path):
    payload = _payload()
    payload["zones"][0]["rows"][1]["observed"] = None
    payload["zones"][0]["rows"][2]["storm"] = None
    data = _figure(_render(tmp_path, payload))["data"]
    assert {trace["name"] for trace in data} == {"Observed", "Storm", "Model"}
    assert [trace["name"] for trace in data if trace["showlegend"]] == ["Observed", "Storm", "Model"]
    assert all(trace["legendgroup"] == trace["name"].lower() for trace in data)
    assert all(trace["connectgaps"] is False for trace in data)
    errors = [trace for trace in data if trace["xaxis"] == "x5"]
    assert [trace["name"] for trace in errors] == ["Storm", "Model"]
    assert errors[0]["y"][:4] == [-2.0, None, None, -2.0]
    assert errors[1]["y"][:4] == [4.0, None, 4.0, 4.0]


def test_germany_stays_present_when_its_model_is_unavailable(tmp_path):
    payload = _payload()
    germany = payload["zones"][1]
    germany["sources"]["model"]["status"] = "unavailable"
    for row in germany["rows"]:
        row["model"] = None
    document = _render(tmp_path, payload)
    figure = _figure(document)
    assert "Germany: 2026-09-10" in document
    assert "Model unavailable" in document
    model = next(trace for trace in figure["data"] if trace["name"] == "Model" and trace["xaxis"] == "x2")
    storm = next(trace for trace in figure["data"] if trace["name"] == "Storm" and trace["xaxis"] == "x2")
    assert model["y"] == [None] * 24
    assert storm["y"][0] == -2.0
    assert '<tr data-zone="DE">' in document


def test_entirely_unavailable_zone_has_empty_curves_and_no_daily_zero(tmp_path):
    payload = _payload()
    germany = payload["zones"][1]
    for row in germany["rows"]:
        row.update(observed=None, storm=None, model=None)
    document = _render(tmp_path, payload)
    card = re.search(r'<article[^>]*aria-label="Germany daily snapshot".*?</article>', document, re.S).group(0)
    assert "0.00" not in card
    assert "Data unavailable" in document
    assert "Observed prices unavailable" in document
    assert all(value is None for value in report.daily_metrics(germany).values())


def test_autumn_dst_retains_two_distinct_0200_hours_and_timezone_labels(tmp_path):
    payload = _payload("2026-10-25")
    document = _render(tmp_path, payload)
    observed = _figure(document)["data"][0]
    assert len(observed["x"]) == len(set(observed["x"])) == 25
    assert observed["customdata"][2:4] == ["02:00 CEST · 2026-10-25", "02:00 CET · 2026-10-25"]
    assert observed["y"][2:4] == [2.0, 3.0]
    assert ">02:00 CEST</td>" in document
    assert ">02:00 CET</td>" in document
    assert 'title="2026-10-25T00:00:00+00:00"' in document
    assert 'title="2026-10-25T01:00:00+00:00"' in document


class _Resources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.external_resources = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key == "src" or (tag == "link" and key == "href"):
                self.external_resources.append(value)


def test_report_is_one_standalone_html_with_inline_plotly_and_no_network_requests(tmp_path):
    document = _render(tmp_path, _payload())
    parser = _Resources()
    parser.feed(document)
    assert parser.external_resources == []
    assert "/* TEST_INLINE_PLOTLY */" in document
    assert not re.search(r"\b(?:fetch|XMLHttpRequest|WebSocket|EventSource)\s*\(", document)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["daily.html"]


def test_html_escapes_source_text_in_titles_attributes_and_footer(tmp_path):
    payload = _payload()
    attack = '\"><script>alert("x")</script>&'
    payload["zones"][0]["name"] = attack
    payload["generated_at"] = attack
    document = _render(tmp_path, payload)
    assert attack not in document
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;&amp;" in document
    assert '<script>alert("x")</script>' not in document
    _figure(document)


def test_template_markers_in_user_text_remain_literal(tmp_path):
    payload = _payload()
    payload["zones"][0]["name"] = "Belgium __ROWS__ __PLOTLY__"
    payload["generated_at"] = "timestamp __FIGURE__ __PLOTLY__"
    document = _render(tmp_path, payload)
    assert "Belgium __ROWS__ __PLOTLY__: 2026-09-10" in document
    assert "Generated timestamp __FIGURE__ __PLOTLY__ · standalone HTML" in document
    assert document.count("/* TEST_INLINE_PLOTLY */") == 1


def test_dashboard_rejects_missing_or_reordered_country_columns(tmp_path):
    payload = _payload()
    payload["zones"][0], payload["zones"][1] = payload["zones"][1], payload["zones"][0]
    with pytest.raises(ValueError, match="BE, DE, FR and NL"):
        report.render_model_storm_report(payload, tmp_path / "daily.html")
    assert not (tmp_path / "daily.html").exists()


def _payload_with_interval(day="2026-09-10"):
    payload = _payload(day)
    for zone in payload["zones"]:
        for row in zone["rows"]:
            row.update(model_p10=row["model"] - 10., model_p90=row["model"] + 18.)
    return payload


@pytest.mark.parametrize("day, hours", [("2026-03-29", 23), ("2026-09-10", 24), ("2026-10-25", 25)])
def test_model_band_uses_exact_hourly_bounds_and_leaves_daily_means_and_errors_unchanged(tmp_path, day, hours):
    payload = _payload_with_interval(day)
    document = _render(tmp_path, payload)
    figure = _figure(document)
    # Validate the real Plotly schema, including fills and legend options.
    from plotly.graph_objects import Figure
    Figure(figure)
    baseline = report._chart(_payload(day))
    intervals = [trace for trace in figure["data"] if trace.get("meta", {}).get("role") == "model_interval"]
    assert len(intervals) == 8
    assert sum(trace["showlegend"] for trace in intervals) == 1
    assert all(trace["legendgroup"] == "model_range" and not trace["connectgaps"] for trace in intervals)
    assert all(trace["legendrank"] > 1000 for trace in intervals)
    for index, zone in enumerate(payload["zones"]):
        xaxis = "x" + (str(index + 1) if index else "")
        lower, upper = [trace for trace in intervals if trace["xaxis"] == xaxis]
        assert lower["x"] == upper["x"] == list(range(hours))
        assert lower["y"] == [row["model_p10"] for row in zone["rows"]]
        assert upper["y"] == [row["model_p90"] for row in zone["rows"]]
        assert lower["fill"] == "none" and upper["fill"] == "tonexty"
        assert upper["fillcolor"] == "rgba(228,139,35,0.20)"
        assert report.daily_metrics(zone) == report.daily_metrics(_payload(day)["zones"][index])
        if hours == 25:
            assert lower["x"][2:4] == [2, 3]  # both autumn 02:00 hours
    bottom = lambda chart: [trace for trace in chart["data"] if trace["xaxis"] in {"x5", "x6", "x7", "x8"}]
    assert bottom(figure) == bottom(baseline)
    model = next(trace for trace in figure["data"] if trace["name"] == "Model" and trace["xaxis"] == "x")
    assert model["text"][0] == "P10–P90: -6.00 – 22.00 €/MWh"
    assert "%{text}" in model["hovertemplate"]
    assert "Model P10</th>" in document and "Model P90</th>" in document
    first_row = re.search(r'<tr data-zone="BE">(.*?)</tr>', document, re.S).group(1)
    values = re.findall(r'<td[^>]*>(.*?)</td>', first_row)
    assert values[1:] == ["0.00", "-2.00", "4.00", "-6.00", "22.00", "-2.00", "4.00"]


def test_price_axis_includes_quantile_extremes_without_changing_error_axis():
    payload = _payload_with_interval()
    baseline = report._chart(_payload())
    payload["zones"][2]["rows"][8].update(model_p10=-400., model_p90=1200.)
    chart = report._chart(payload)
    for suffix in ["", "2", "3", "4"]:
        low, high = chart["layout"]["yaxis" + suffix]["range"]
        assert low < -400. and high > 1200.
    for suffix in ["5", "6", "7", "8"]:
        assert chart["layout"]["yaxis" + suffix]["range"] == baseline["layout"]["yaxis" + suffix]["range"]


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), True, 1000.])
def test_model_band_never_bridges_missing_or_invalid_bounds(bad):
    payload = _payload_with_interval()
    payload["zones"][0]["rows"][5]["model_p10"] = bad
    chart = report._chart(payload)
    bands = [trace for trace in chart["data"] if trace.get("meta", {}).get("role") == "model_interval" and trace["xaxis"] == "x"]
    assert [trace["x"] for trace in bands] == [list(range(5)), list(range(5)), list(range(6, 24)), list(range(6, 24))]
    assert report._model_interval(payload["zones"][0]["rows"][5]) == (None, None)


def test_band_legend_remains_available_when_first_country_has_no_model():
    payload = _payload_with_interval()
    for row in payload["zones"][0]["rows"]:
        row["model"] = None
    chart = report._chart(payload)
    intervals = [trace for trace in chart["data"] if trace.get("meta", {}).get("role") == "model_interval"]
    assert all(trace["xaxis"] != "x" for trace in intervals)
    legend = [trace for trace in intervals if trace["showlegend"]]
    assert len(legend) == 1 and legend[0]["xaxis"] == "x2"
