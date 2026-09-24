"""Late reconstructions never become prospective or partial-day metrics."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_exogenous.retrospective_reporting import (
    _normalise,
    _summarise,
    render_retrospective_report,
)


MODELS = ("lora16_residual", "lora16_residual_kalman", "chronos2_exogenous")


def _day(day: str = "2026-09-08", *, zone: str = "FR", actual: float = 100.0) -> pd.DataFrame:
    start = pd.Timestamp(day).tz_localize("Europe/Paris")
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize("Europe/Paris")
    hours = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
    result = pd.DataFrame({"delivery_start_utc": hours, "zone": zone, "actual": actual})
    for model, prediction in zip(MODELS, [102.0, 101.0, 104.0], strict=True):
        for q, offset in [(10, -5), (50, 0), (90, 5)]:
            result[f"{model}__q{q}"] = prediction + offset
    return result


def _summary(frame: pd.DataFrame) -> dict:
    result, models, _ = _normalise(frame, "Europe/Paris")
    return _summarise(result, models)


def test_complete_single_day_metrics_and_retrospective_label(tmp_path: Path) -> None:
    summary = _summary(_day())
    assert summary["hours"] == 24
    assert summary["actual_mean"] == 100
    assert summary["models"][MODELS[0]] == {
        "mean": 102.0, "bias": 2.0, "daily_absolute_error": 2.0,
        "hourly_mae": 2.0, "evaluated_hours": 24,
    }
    html = render_retrospective_report(_day(), tmp_path / "day.html").read_text(encoding="utf-8")
    assert "RÉTROSPECTIF" in html
    assert "Hors statistiques prospectives" in html
    assert "365 jours strictement antérieurs" in html
    assert "ne sont pas un backtest indépendant de 365 jours" in html
    assert "aucun ajout au journal prospectif" in html
    assert "<svg" in html
    assert "<title>Prix observé" in html
    assert "Mode nuit" in html
    assert "--residual:#70baff" in html  # dark-mode curves have their own palette
    assert '<script src=' not in html
    assert '<link ' not in html


@pytest.mark.parametrize("missing", ["all", "one"])
def test_missing_observations_leave_all_metrics_blank(tmp_path: Path, missing: str) -> None:
    frame = _day()
    if missing == "all":
        frame["actual"] = np.nan
    else:
        frame.loc[3, "actual"] = np.nan
    summary = _summary(frame)
    assert summary["actual_mean"] is None
    for model in MODELS:
        values = summary["models"][model]
        assert values["mean"] is not None
        assert values["bias"] is None
        assert values["daily_absolute_error"] is None
        assert values["hourly_mae"] is None
        assert values["evaluated_hours"] == 0
    html = render_retrospective_report(frame, tmp_path / "pending.html").read_text(encoding="utf-8")
    assert 'class="number">102,00</td><td class="number"></td><td class="number"></td>' in html
    assert "Observations incomplètes" in html
    assert ">nan<" not in html.lower()


@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2026-09-08", 24), ("2026-10-25", 25)])
def test_metrics_use_exact_physical_day_including_dst(tmp_path: Path, day: str, hours: int) -> None:
    frame = _day(day)
    summary = _summary(frame)
    assert summary["hours"] == hours
    assert summary["models"][MODELS[0]]["evaluated_hours"] == hours
    assert summary["models"][MODELS[0]]["hourly_mae"] == 2
    text = render_retrospective_report(frame, tmp_path / "dst.html").read_text(encoding="utf-8")
    assert f"Observations : {hours}/{hours} heures" in text
    if hours == 25:
        assert "02:00 +0200" in text
        assert "02:00 +0100" in text


@pytest.mark.parametrize("day", ["2026-03-29", "2026-09-08", "2026-10-25"])
def test_missing_forecast_hour_is_rejected(tmp_path: Path, day: str) -> None:
    with pytest.raises(ValueError, match="couverture physique incomplète"):
        render_retrospective_report(_day(day).iloc[:-1], tmp_path / "bad.html")


def test_multiple_zones_share_delivery_day_but_not_observation_completion(tmp_path: Path) -> None:
    frame = pd.concat([_day(zone="FR"), _day(zone="DE", actual=np.nan)], ignore_index=True)
    html = render_retrospective_report(frame, tmp_path / "zones.html").read_text(encoding="utf-8")
    assert "FR · 2026-09-08" in html
    assert "DE · 2026-09-08" in html
    assert 'id="chart-0"' in html
    assert 'id="chart-1"' in html
    assert 'class="number">0/24</td>' in html
    assert 'class="number">24/24</td>' in html


def test_hourly_mae_differs_from_daily_mean_error() -> None:
    frame = _day()
    for model in MODELS:
        for q, offset in [(10, -5), (50, 0), (90, 5)]:
            frame[f"{model}__q{q}"] = np.tile([90., 110.], 12) + offset
    values = _summary(frame)["models"][MODELS[0]]
    assert values["hourly_mae"] == 10
    assert values["daily_absolute_error"] == 0


def test_optional_incumbent_full_and_missing_comparisons(tmp_path: Path) -> None:
    frame = _day()
    for q, offset in [(10, -5), (50, 0), (90, 5)]:
        frame[f"incumbent_autonomous__q{q}"] = 103 + offset
        frame[f"incumbent_kalman__q{q}"] = np.nan
    summary = _summary(frame)
    assert summary["models"]["incumbent_autonomous"]["hourly_mae"] == 3
    assert summary["models"]["incumbent_kalman"]["mean"] is None
    assert summary["models"]["incumbent_kalman"]["hourly_mae"] is None
    html = render_retrospective_report(frame, tmp_path / "incumbents.html").read_text(encoding="utf-8")
    assert "Run existant — autonome" in html
    assert "Run existant — Kalman" in html
    assert 'Run existant — Kalman</td><td class="number"></td>' in html


def test_partial_incumbent_is_not_scored_on_smaller_support() -> None:
    frame = _day()
    for q, offset in [(10, -5), (50, 0), (90, 5)]:
        frame[f"incumbent_kalman__q{q}"] = 99 + offset
    frame.loc[4, "incumbent_kalman__q50"] = np.nan
    values = _summary(frame)["models"]["incumbent_kalman"]
    assert values["mean"] is None
    assert values["hourly_mae"] is None
    assert values["evaluated_hours"] == 0


@pytest.mark.parametrize("corruption", [
    "duplicate", "naive", "infinite_actual", "infinite_forecast", "missing_forecast",
    "inversion", "two_days", "shifted_hour", "empty", "missing_quantile", "empty_zone",
])
def test_invalid_inputs_are_rejected_without_publishing(tmp_path: Path, corruption: str) -> None:
    frame = _day()
    if corruption == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif corruption == "naive":
        frame["delivery_start_utc"] = frame["delivery_start_utc"].dt.tz_localize(None)
    elif corruption == "infinite_actual":
        frame.loc[0, "actual"] = np.inf
    elif corruption == "infinite_forecast":
        frame.loc[0, f"{MODELS[0]}__q50"] = np.inf
    elif corruption == "missing_forecast":
        frame.loc[0, f"{MODELS[0]}__q50"] = np.nan
    elif corruption == "inversion":
        frame.loc[0, f"{MODELS[0]}__q10"] = 500
    elif corruption == "two_days":
        frame = pd.concat([frame, _day("2026-09-09", zone="BE")], ignore_index=True)
    elif corruption == "shifted_hour":
        frame.loc[0, "delivery_start_utc"] += pd.Timedelta(minutes=15)
    elif corruption == "empty":
        frame = frame.iloc[:0]
    elif corruption == "missing_quantile":
        frame["incumbent_kalman__q50"] = 99.0
    else:
        frame["zone"] = " "
    path = tmp_path / "bad.html"
    with pytest.raises(ValueError):
        render_retrospective_report(frame, path)
    assert not path.exists()


def test_metadata_and_zones_are_escaped(tmp_path: Path) -> None:
    zone = '\"><script>alert("zone")</script>'
    metadata = {"comment": '</pre><script>alert("metadata")</script>'}
    html = render_retrospective_report(_day(zone=zone), tmp_path / "safe.html", metadata=metadata).read_text(encoding="utf-8")
    assert zone not in html
    assert metadata["comment"] not in html
    assert "&lt;script&gt;" in html
    assert html.count("<script>") == 1  # only the fixed application script


def test_output_is_immutable_and_input_is_not_mutated(tmp_path: Path) -> None:
    path = tmp_path / "immutable.html"
    frame = _day().iloc[::-1]
    original = frame.copy(deep=True)
    render_retrospective_report(frame, path)
    before = path.read_bytes()
    pd.testing.assert_frame_equal(frame, original)
    with pytest.raises(FileExistsError, match="remplacement refusé"):
        render_retrospective_report(frame, path)
    assert path.read_bytes() == before
