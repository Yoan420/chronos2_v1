from argparse import Namespace

import pandas as pd

from materialize_saturn_daily_asof import _civil_cutoff, _one_day


def test_civil_cutoff_stays_at_eight_after_autumn_dst_switch() -> None:
    cutoff = _civil_cutoff(
        pd.Timestamp("2024-10-28"),
        timezone="Europe/Paris",
        cutoff_time="08:00",
    )
    assert cutoff == pd.Timestamp("2024-10-27 08:00", tz="Europe/Paris")
    assert cutoff.tz_convert("UTC") == pd.Timestamp("2024-10-27 07:00Z")


def test_civil_cutoff_stays_at_eight_after_spring_dst_switch() -> None:
    cutoff = _civil_cutoff(
        pd.Timestamp("2025-03-31"),
        timezone="Europe/Paris",
        cutoff_time="08:00",
    )
    assert cutoff == pd.Timestamp("2025-03-30 08:00", tz="Europe/Paris")
    assert cutoff.tz_convert("UTC") == pd.Timestamp("2025-03-30 06:00Z")


def test_delivery_timezone_and_cutoff_timezone_are_independent(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_fetch(
        _client,
        _series,
        start,
        end,
        timezone,
        *,
        revision_date,
        naive_timezone,
    ) -> pd.Series:
        captured.update(
            start=start,
            end=end,
            timezone=timezone,
            revision_date=revision_date,
            naive_timezone=naive_timezone,
        )
        index = pd.date_range(
            "2026-08-14 00:00",
            periods=24,
            freq="h",
            tz="Europe/Berlin",
        )
        return pd.Series(range(24), index=index, dtype=float)

    monkeypatch.setattr("materialize_saturn_daily_asof._client", lambda _args: object())
    monkeypatch.setattr(
        "materialize_saturn_daily_asof.fetch_saturn_series_from_client",
        fake_fetch,
    )
    args = Namespace(
        timezone="Europe/Berlin",
        cutoff_timezone="Europe/Paris",
        cutoff_time="08:00",
        retries=1,
        series="41550_native",
        naive_timezone="UTC",
        hourly_on_the_hour=False,
        allow_incomplete_days=False,
    )

    frame = _one_day(pd.Timestamp("2026-08-14"), args)

    expected_cutoff = pd.Timestamp("2026-08-13 08:00", tz="Europe/Paris").tz_convert(
        "UTC"
    )
    assert captured["timezone"] == "Europe/Berlin"
    assert captured["revision_date"] == expected_cutoff
    assert len(frame) == 24
    assert frame["snapshot_time_utc"].eq(expected_cutoff).all()
