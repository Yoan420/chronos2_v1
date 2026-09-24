from __future__ import annotations

from argparse import Namespace
import json

import numpy as np
import pandas as pd
import pytest

from materialize_saturn_daily_asof import (
    _broadcast_daily_value,
    _civil_cutoff,
    _one_day,
    _physical_utc_index,
    _select_physical_delivery_day,
    _validate_dst_repair_scope,
    main,
)
from chronos2_modular.saturn import fetch_saturn_series_from_client
from materialize_saturn_kalman_weather import build_plan, reusable_output


@pytest.mark.parametrize(
    ("day", "expected_hours"),
    [
        ("2025-03-30", 23),
        ("2025-08-15", 24),
        ("2025-10-26", 25),
    ],
)
def test_daily_broadcast_preserves_physical_dst_hours(
    day: str,
    expected_hours: int,
) -> None:
    local_day = pd.Timestamp(day)
    source = pd.Series(
        [12.5],
        index=pd.DatetimeIndex([pd.Timestamp(day, tz="UTC")]),
    )

    result = _broadcast_daily_value(
        source,
        day=local_day,
        timezone="Europe/Paris",
    )

    assert len(result) == expected_hours
    assert result.index.equals(
        _physical_utc_index(local_day, timezone="Europe/Paris")
    )
    assert result.eq(12.5).all()
    assert not result.index.has_duplicates


def test_daily_broadcast_rejects_ambiguous_or_missing_values() -> None:
    day = pd.Timestamp("2026-08-29")
    with pytest.raises(RuntimeError, match="exactement une valeur finie"):
        _broadcast_daily_value(
            pd.Series([1.0, 2.0]),
            day=day,
            timezone="Europe/Paris",
        )
    with pytest.raises(RuntimeError, match="exactement une valeur finie"):
        _broadcast_daily_value(
            pd.Series([float("nan")]),
            day=day,
            timezone="Europe/Paris",
        )


def test_cutoff_is_built_in_civil_time_across_dst() -> None:
    spring = _civil_cutoff(
        pd.Timestamp("2025-03-31"),
        timezone="Europe/Paris",
        cutoff_time="08:00",
    )
    autumn = _civil_cutoff(
        pd.Timestamp("2025-10-27"),
        timezone="Europe/Paris",
        cutoff_time="08:00",
    )

    assert spring.hour == 8
    assert spring.tz_convert("UTC").hour == 6
    assert autumn.hour == 8
    assert autumn.tz_convert("UTC").hour == 7


def test_request_padding_never_changes_the_revision_cutoff() -> None:
    class Client:
        def __init__(self) -> None:
            self.kwargs = None

        def get(self, _series, **kwargs):
            self.kwargs = kwargs
            return pd.Series(
                [0.0],
                index=pd.DatetimeIndex(["2025-03-29 23:00"], tz="UTC"),
            )

    client = Client()
    start = pd.Timestamp("2025-03-30 00:00", tz="Europe/Amsterdam")
    end = pd.Timestamp("2025-03-31 00:00", tz="Europe/Amsterdam")
    cutoff = pd.Timestamp("2025-03-29 07:00", tz="UTC")

    fetch_saturn_series_from_client(
        client,
        "power.nl.generation.solar.hourly.gw.fcst",
        start,
        end,
        "Europe/Amsterdam",
        revision_date=cutoff,
        naive_timezone="Europe/Amsterdam",
        request_padding_hours=8,
    )

    assert client.kwargs["from_value_date"] == start - pd.Timedelta(hours=8)
    assert client.kwargs["to_value_date"] == end + pd.Timedelta(hours=8)
    assert client.kwargs["revision_date"] == cutoff


def test_zero_only_repair_is_scoped_to_the_audited_nl_solar_formula() -> None:
    base = dict(
        alias="nl_solar_generation_fcst",
        series="power.nl.generation.solar.hourly.gw.fcst",
        timezone="Europe/Amsterdam",
        cutoff_timezone="Europe/Amsterdam",
        naive_timezone="Europe/Amsterdam",
        incomplete_dst_policy="duplicate_zero_only",
        daily_broadcast=False,
    )
    _validate_dst_repair_scope(Namespace(**base))

    for override in (
        {"alias": "fr_solar_generation_fcst"},
        {"series": "power.fr.generation.solar.hourly.gw.fcst"},
        {"timezone": "UTC"},
        {"cutoff_timezone": "UTC"},
        {"naive_timezone": "UTC"},
        {"daily_broadcast": True},
    ):
        with pytest.raises(ValueError, match="reserve au solaire NL"):
            _validate_dst_repair_scope(Namespace(**{**base, **override}))

    # The historical generic duplicate policy remains a separate, explicit
    # covariate contract; this guard must not silently rewrite it.
    generic = {**base, "incomplete_dst_policy": "duplicate"}
    _validate_dst_repair_scope(Namespace(**generic))


@pytest.mark.parametrize("policy", ["duplicate", "duplicate_zero_only"])
def test_any_dst_duplication_is_rejected_for_day_ahead_targets(policy: str) -> None:
    args = Namespace(
        alias="target",
        series="power.price.da.nl.bzn.hourly.entsoe.eurmwh",
        timezone="Europe/Amsterdam",
        cutoff_timezone="Europe/Amsterdam",
        naive_timezone="Europe/Amsterdam",
        incomplete_dst_policy=policy,
        daily_broadcast=False,
    )
    with pytest.raises(ValueError, match="interdite pour une cible"):
        _validate_dst_repair_scope(args)


@pytest.mark.parametrize(
    ("timezone", "day", "expected_hours"),
    [
        ("Europe/Amsterdam", "2025-03-30", 23),
        ("Europe/Amsterdam", "2025-10-26", 25),
        ("Europe/London", "2025-03-30", 23),
        ("Europe/Madrid", "2025-10-26", 25),
        ("UTC", "2025-03-30", 24),
    ],
)
def test_physical_selector_keeps_exact_civil_products(
    timezone: str,
    day: str,
    expected_hours: int,
) -> None:
    local_day = pd.Timestamp(day)
    expected = _physical_utc_index(local_day, timezone=timezone)
    wide = expected.insert(0, expected[0] - pd.Timedelta(hours=8)).append(
        pd.DatetimeIndex([expected[-1] + pd.Timedelta(hours=8)])
    )
    series = pd.Series(np.arange(len(wide), dtype=float), index=wide)

    selected = _select_physical_delivery_day(
        series,
        day=local_day,
        timezone=timezone,
    )

    assert len(selected) == expected_hours
    assert selected.index.equals(expected)
    assert np.isfinite(selected.to_numpy()).all()


def test_physical_selector_rejects_missing_or_nonfinite_hours_even_with_same_count() -> None:
    day = pd.Timestamp("2025-03-30")
    expected = _physical_utc_index(day, timezone="Europe/Amsterdam")
    missing_inside = expected.delete(3).append(
        pd.DatetimeIndex([expected[-1] + pd.Timedelta(hours=1)])
    )
    same_count_wrong_grid = pd.Series(
        np.arange(len(missing_inside), dtype=float),
        index=missing_inside,
    )
    with pytest.raises(RuntimeError, match="absente.*non finie"):
        _select_physical_delivery_day(
            same_count_wrong_grid,
            day=day,
            timezone="Europe/Amsterdam",
        )

    nonfinite = pd.Series(1.0, index=expected)
    nonfinite.iloc[7] = np.nan
    with pytest.raises(RuntimeError, match="absente.*non finie"):
        _select_physical_delivery_day(
            nonfinite,
            day=day,
            timezone="Europe/Amsterdam",
        )


def test_one_day_overfetches_then_selects_exact_physical_grid(monkeypatch) -> None:
    captured: dict[str, object] = {}
    day = pd.Timestamp("2025-03-30")
    physical = _physical_utc_index(day, timezone="Europe/Amsterdam")

    def fake_fetch(
        _client,
        _series,
        start,
        end,
        timezone,
        **kwargs,
    ) -> pd.Series:
        captured.update(start=start, end=end, timezone=timezone, kwargs=kwargs)
        wide = physical.insert(0, physical[0] - pd.Timedelta(hours=8)).append(
            pd.DatetimeIndex([physical[-1] + pd.Timedelta(hours=8)])
        )
        return pd.Series(np.arange(len(wide), dtype=float), index=wide)

    monkeypatch.setattr("materialize_saturn_daily_asof._client", lambda _args: object())
    monkeypatch.setattr(
        "materialize_saturn_daily_asof.fetch_saturn_series_from_client",
        fake_fetch,
    )
    args = Namespace(
        timezone="Europe/Amsterdam",
        cutoff_timezone="Europe/Amsterdam",
        cutoff_time="08:00",
        retries=1,
        series="power.nl.generation.solar.hourly.gw.fcst",
        naive_timezone="Europe/Amsterdam",
        incomplete_dst_policy="raise",
        request_padding_hours=8,
        daily_broadcast=False,
        value_scale=1.0,
        hourly_on_the_hour=False,
        allow_incomplete_days=False,
    )

    frame = _one_day(day, args)

    assert captured["start"] == day.tz_localize("Europe/Amsterdam")
    assert captured["end"] == (day + pd.Timedelta(days=1)).tz_localize(
        "Europe/Amsterdam"
    )
    assert captured["kwargs"]["request_padding_hours"] == 8
    assert len(frame) == 23
    assert pd.DatetimeIndex(frame["value_time_utc"]).equals(physical)


def test_main_records_zero_only_fold_repair_in_sidecar(
    tmp_path,
    monkeypatch,
) -> None:
    day = pd.Timestamp("2025-10-26")
    physical = _physical_utc_index(day, timezone="Europe/Amsterdam")
    cutoff = pd.Timestamp("2025-10-25 06:00:00", tz="UTC")
    output = tmp_path / "nl_solar.parquet"
    args = Namespace(
        series="power.nl.generation.solar.hourly.gw.fcst",
        alias="nl_solar_generation_fcst",
        start_day="2025-10-26",
        end_day="2025-10-26",
        output=str(output),
        timezone="Europe/Amsterdam",
        cutoff_timezone="Europe/Amsterdam",
        cutoff_time="08:00",
        naive_timezone="Europe/Amsterdam",
        saturn_url="unused",
        author="test",
        workers=1,
        retries=1,
        value_scale=1.0,
        allow_incomplete_days=False,
        merge_existing=False,
        incomplete_dst_policy="duplicate_zero_only",
        daily_broadcast=False,
        hourly_on_the_hour=False,
        request_timeout_seconds=1.0,
        request_padding_hours=8,
    )

    def fake_one_day(_day, _args):
        frame = pd.DataFrame(
            {
                "value_time_utc": physical,
                "snapshot_time_utc": cutoff,
                "revision_time_utc": cutoff,
                "value": 0.0,
                "downloaded_at_utc": pd.Timestamp("2025-10-25 09:00", tz="UTC"),
            }
        )
        frame.attrs["dst_repairs"] = [
            {
                "policy": "duplicate_zero_only",
                "local_timestamp": "2025-10-26T02:00:00",
                "duplicated_value": 0.0,
                "physical_hours_utc": [
                    "2025-10-26T00:00:00+00:00",
                    "2025-10-26T01:00:00+00:00",
                ],
            }
        ]
        return frame

    monkeypatch.setattr("materialize_saturn_daily_asof.parse_args", lambda: args)
    monkeypatch.setattr("materialize_saturn_daily_asof._one_day", fake_one_day)

    assert main() == 0
    audit = json.loads(
        output.with_name(output.name + ".audit.json").read_text(encoding="utf-8")
    )
    assert audit["dst_zero_duplicate_repair_count"] == 1
    assert audit["dst_zero_duplicate_repairs"][0]["duplicated_value"] == 0.0
    assert audit["dst_zero_duplicate_repairs"][0]["physical_hours_utc"] == [
        "2025-10-26T00:00:00+00:00",
        "2025-10-26T01:00:00+00:00",
    ]
    base_plan = {
        item.alias: item for item in build_plan(["NL"], tmp_path)
    }["nl_solar_generation_fcst"]
    plan = type(base_plan)(**{**base_plan.__dict__, "output": output})
    reusable, reason = reusable_output(plan, start_day=day, end_day=day)
    assert reusable is True, reason

    audit["dst_zero_duplicate_repairs"][0]["physical_hours_utc"].reverse()
    plan.audit_path.write_text(json.dumps(audit), encoding="utf-8")
    reusable, reason = reusable_output(plan, start_day=day, end_day=day)
    assert reusable is False
    assert "folds physiques invalides" in reason
