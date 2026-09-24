"""The trial report must not turn calibration or pending prices into a test."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_exogenous.prospective_reporting import (
    _normalise,
    _summarise_days,
    render_trial_report,
)


MODELS = ("lora16_residual", "lora16_residual_kalman", "chronos2_exogenous")


def _day(day: str, *, zone: str = "FR", actual: float = 100.0, prospective: bool = True) -> pd.DataFrame:
    start = pd.Timestamp(day).tz_localize("Europe/Paris")
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize("Europe/Paris")
    index = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
    frame = pd.DataFrame({
        "delivery_start_utc": index,
        "zone": zone,
        "actual": actual,
        "prospective_eligible": prospective,
    })
    for model, prediction in zip(MODELS, [102.0, 101.0, 104.0], strict=True):
        for quantile, offset in [(10, -5), (50, 0), (90, 5)]:
            frame[f"{model}__q{quantile}"] = prediction + offset
    return frame


def _summary(frame: pd.DataFrame) -> list[dict]:
    return _summarise_days(_normalise(frame), "Europe/Paris")[0]


def test_metrics_exclude_retrospective_and_unpublished_days(tmp_path: Path) -> None:
    frame = pd.concat([
        _day("2026-09-01", actual=10000.0, prospective=False),
        _day("2026-09-02"),
        _day("2026-09-03", actual=np.nan),
    ], ignore_index=True)
    days = _summary(frame)
    assert [row["status_code"] for row in days] == ["retrospective", "evaluated", "pending"]
    assert days[0]["models"][MODELS[0]]["hourly_absolute_error_sum"] is None
    assert days[1]["models"][MODELS[0]]["hourly_absolute_error_sum"] == 48
    assert days[1]["models"][MODELS[1]]["daily_mean_absolute_error"] == 1
    assert days[2]["actual_mean"] is None
    assert days[2]["models"][MODELS[0]]["mean"] == 102
    output = render_trial_report(frame, tmp_path / "report.html")
    text = output.read_text(encoding="utf-8")
    assert "1 jours calendaires évalués" in text
    assert "1 journées-pays évaluées" in text
    assert "1 journées-pays en attente" in text
    assert '<td>2026-09-03</td><td>FR</td><td>24/24</td><td class="number"></td>' in text
    assert "Chronos-2 + LoRA rang 16 + correcteur résiduel, avec ou sans Kalman" in text
    assert "Recherche uniquement — aucune promotion en production" in text
    assert "aucun résultat historique de calibration" in text


@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2026-10-25", 25), ("2026-09-09", 24)])
def test_complete_physical_days_include_both_dst_transitions(day: str, hours: int) -> None:
    frame = _day(day)
    row = _summary(frame)[0]
    assert len(frame) == hours
    assert row["expected_hours"] == hours
    assert row["evaluated"]
    assert row["models"][MODELS[0]]["hourly_absolute_error_sum"] == 2 * hours


@pytest.mark.parametrize("day", ["2026-03-29", "2026-10-25", "2026-09-09"])
def test_a_missing_hour_excludes_the_whole_day(day: str) -> None:
    row = _summary(_day(day).iloc[:-1])[0]
    assert not row["evaluated"]
    assert row["status_code"] == "incomplete"
    assert row["actual_mean"] is None
    assert all(value["mean"] is None for value in row["models"].values())


def test_one_missing_actual_does_not_show_a_partial_daily_observation(tmp_path: Path) -> None:
    frame = _day("2026-09-09")
    frame.loc[3, "actual"] = np.nan
    row = _summary(frame)[0]
    assert row["actual_mean"] is None
    assert row["status_code"] == "pending"
    text = render_trial_report(frame, tmp_path / "pending.html").read_text(encoding="utf-8")
    assert '<td>24/24</td><td class="number"></td><td class="number">102,00</td>' in text
    assert "nan" not in text.lower()


def test_every_model_shares_common_complete_support() -> None:
    frame = _day("2026-09-09")
    frame.loc[2, "chronos2_exogenous__q50"] = np.nan
    row = _summary(frame)[0]
    assert not row["evaluated"]
    assert row["status_code"] == "incomplete"
    assert all(value["daily_mean_absolute_error"] is None for value in row["models"].values())


def test_one_nonprospective_hour_excludes_the_whole_day() -> None:
    frame = _day("2026-09-09")
    frame.loc[3, "prospective_eligible"] = False
    row = _summary(frame)[0]
    assert not row["evaluated"]
    assert row["status_code"] == "retrospective"


def test_window_is_exactly_365_calendar_days_not_365_rows() -> None:
    frames = [
        _day("2025-09-08"),  # One day before the inclusive window.
        _day("2025-09-09"),
        _day("2026-09-08"),
        _day("2026-09-08", zone="DE"),
    ]
    days = _summary(pd.concat(frames, ignore_index=True))
    old = next(row for row in days if row["day"] == "2025-09-08")
    assert old["status_code"] == "outside"
    assert sum(row["evaluated"] for row in days) == 3


def test_daily_mean_mae_is_not_hourly_mae(tmp_path: Path) -> None:
    frame = _day("2026-09-09")
    for model in MODELS:
        for quantile, offset in [(10, -5), (50, 0), (90, 5)]:
            frame[f"{model}__q{quantile}"] = np.tile([90.0, 110.0], 12) + offset
    row = _summary(frame)[0]
    assert row["models"][MODELS[0]]["hourly_absolute_error_sum"] == 240
    assert row["models"][MODELS[0]]["daily_mean_absolute_error"] == 0
    text = render_trial_report(frame, tmp_path / "metrics.html").read_text(encoding="utf-8")
    assert '<td class="number">10,00</td><td class="number">0,00</td>' in text


def test_metadata_and_zone_are_html_escaped(tmp_path: Path) -> None:
    zone = '\"><script>alert("zone")</script>'
    frame = _day("2026-09-09", zone=zone)
    metadata = {"notes": '</pre><script>alert("metadata")</script>'}
    text = render_trial_report(frame, tmp_path / "safe.html", metadata=metadata).read_text(encoding="utf-8")
    assert zone not in text
    assert '</pre><script>alert("metadata")</script>' not in text
    assert "&lt;script&gt;" in text
    assert 'data-zone="&quot;&gt;&lt;script&gt;' in text
    assert '<script src=' not in text
    assert '<link ' not in text


def test_report_never_overwrites_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "immutable.html"
    path.write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError, match="remplacement refusé"):
        render_trial_report(_day("2026-09-09"), path)
    assert path.read_text(encoding="utf-8") == "keep me"


def test_report_does_not_mutate_input_frame(tmp_path: Path) -> None:
    frame = _day("2026-09-09").iloc[::-1]
    before = frame.copy(deep=True)
    render_trial_report(frame, tmp_path / "report.html")
    pd.testing.assert_frame_equal(frame, before)


def test_empty_ledger_has_no_fabricated_statistics(tmp_path: Path) -> None:
    empty = _day("2026-09-09").iloc[:0]
    text = render_trial_report(empty, tmp_path / "empty.html").read_text(encoding="utf-8")
    assert "0 jours calendaires évalués" in text
    assert "Aucune prévision enregistrée" in text


@pytest.mark.parametrize("corruption", ["duplicate", "naive", "infinite", "inversion", "string_bool"])
def test_invalid_ledger_is_rejected(tmp_path: Path, corruption: str) -> None:
    frame = _day("2026-09-09")
    if corruption == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif corruption == "naive":
        frame["delivery_start_utc"] = frame["delivery_start_utc"].dt.tz_localize(None)
    elif corruption == "infinite":
        frame.loc[0, "actual"] = np.inf
    elif corruption == "inversion":
        frame.loc[0, "lora16_residual__q10"] = 500
    else:
        frame["prospective_eligible"] = "false"
    path = tmp_path / "invalid.html"
    with pytest.raises(ValueError):
        render_trial_report(frame, path)
    assert not path.exists()
