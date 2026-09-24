from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import materialize_saturn_kalman_fuel as fuel


def _fake_daily(days: list[pd.Timestamp]) -> pd.DataFrame:
    index = pd.DatetimeIndex(days)
    base = pd.Timestamp("2020-01-01")
    elapsed = np.asarray([(day - base).days for day in index], dtype=float)
    cutoffs = pd.DatetimeIndex(
        [fuel._civil_cutoff(day).tz_convert("UTC") for day in index]
    )
    return pd.DataFrame(
        {
            "ttf_m1_eur_mwh_th": 30.0 + elapsed * 0.01,
            "ttf_m1_eur_mwh_th__source_value_time_utc": (
                cutoffs - pd.Timedelta(hours=10)
            ),
            "ttf_m1_eur_mwh_th__cutoff_utc": cutoffs,
            "eua_first_dec_eur_tco2": 60.0 + elapsed * 0.02,
            "eua_first_dec_eur_tco2__source_value_time_utc": (
                cutoffs - pd.Timedelta(hours=12)
            ),
            "eua_first_dec_eur_tco2__cutoff_utc": cutoffs,
            "cutoff_time_utc": cutoffs,
        },
        index=index,
    )


def _arguments(tmp_path: Path, start: str, end: str) -> list[str]:
    return [
        "--start-day",
        start,
        "--end-day",
        end,
        "--output-dir",
        str(tmp_path),
        "--series-workers",
        "1",
        "--day-workers",
        "1",
        "--skip-residual-load",
    ]


def _write_residual_vintages(
    vintage_root: Path,
    *,
    start: str,
    end: str,
) -> pd.DatetimeIndex:
    vintage_root.mkdir(parents=True, exist_ok=True)
    expected = fuel._expected_hourly_index(
        pd.Timestamp(start),
        pd.Timestamp(end),
    )
    cutoffs = fuel._cutoffs_for_value_times(expected)
    for alias_number, alias in enumerate(fuel.RESIDUAL_LOAD_ALIASES):
        records: list[dict[str, object]] = []
        for position, (value_time, cutoff) in enumerate(zip(expected, cutoffs)):
            chosen_value = float(alias_number * 1000 + position)
            revision = cutoff - pd.Timedelta(minutes=10 + alias_number)
            snapshot = cutoff - pd.Timedelta(minutes=20 + alias_number)
            base = {
                "value_time_utc": value_time,
                "snapshot_time_utc": cutoff - pd.Timedelta(hours=3),
                "revision_time_utc": cutoff - pd.Timedelta(hours=2),
                "value": -100.0,
                "downloaded_at_utc": pd.Timestamp("2026-01-01", tz="UTC"),
            }
            records.append(base)
            # A newer snapshot cannot beat a newer revision: revision is the
            # primary ordering key mandated by the local-vintage contract.
            records.append(
                {
                    **base,
                    "snapshot_time_utc": cutoff - pd.Timedelta(minutes=5),
                    "revision_time_utc": cutoff
                    - pd.Timedelta(minutes=15 + alias_number),
                    "value": 777_777.0,
                    "downloaded_at_utc": pd.Timestamp("2026-04-01", tz="UTC"),
                }
            )
            # Equal revision/snapshot: downloaded_at is the deterministic
            # final tie-breaker.
            records.append(
                {
                    **base,
                    "snapshot_time_utc": snapshot,
                    "revision_time_utc": revision,
                    "value": -123.0,
                    "downloaded_at_utc": pd.Timestamp("2026-02-01", tz="UTC"),
                }
            )
            records.append(
                {
                    **base,
                    "snapshot_time_utc": snapshot,
                    "revision_time_utc": revision,
                    "value": chosen_value,
                    "downloaded_at_utc": pd.Timestamp("2026-03-01", tz="UTC"),
                }
            )
            # Both forms of post-cutoff leakage must be rejected before the
            # ordering step, despite their deliberately huge values.
            records.append(
                {
                    **base,
                    "snapshot_time_utc": cutoff + pd.Timedelta(minutes=1),
                    "revision_time_utc": cutoff - pd.Timedelta(minutes=1),
                    "value": 888_888.0,
                    "downloaded_at_utc": pd.Timestamp("2026-05-01", tz="UTC"),
                }
            )
            records.append(
                {
                    **base,
                    "snapshot_time_utc": cutoff - pd.Timedelta(minutes=1),
                    "revision_time_utc": cutoff + pd.Timedelta(minutes=1),
                    "value": 999_999.0,
                    "downloaded_at_utc": pd.Timestamp("2026-06-01", tz="UTC"),
                }
            )
        pd.DataFrame.from_records(records).to_parquet(
            vintage_root / f"{alias}.parquet",
            index=False,
        )
    return expected


def test_civil_cutoff_is_exact_on_both_dst_transitions() -> None:
    autumn = fuel._civil_cutoff(pd.Timestamp("2024-10-28"))
    spring = fuel._civil_cutoff(pd.Timestamp("2025-03-31"))

    assert autumn.hour == 8
    assert autumn.tz_convert("UTC") == pd.Timestamp(
        "2024-10-27 07:00:00", tz="UTC"
    )
    assert spring.hour == 8
    assert spring.tz_convert("UTC") == pd.Timestamp(
        "2025-03-30 06:00:00", tz="UTC"
    )


def test_saturn_query_uses_revision_cutoff_and_rejects_future_values() -> None:
    calls: list[dict[str, object]] = []

    class Client:
        def get(self, series: str, **kwargs: object) -> pd.Series:
            calls.append({"series": series, **kwargs})
            cutoff = pd.Timestamp(kwargs["revision_date"])
            return pd.Series(
                [40.0, 999.0],
                index=pd.DatetimeIndex(
                    [cutoff - pd.Timedelta(hours=9), cutoff + pd.Timedelta(hours=1)]
                ),
            )

    spec = fuel.MARKET_SERIES[0]
    value, source_time, cutoff = fuel._query_market_value(
        Client(), spec, pd.Timestamp("2025-03-31")
    )

    assert value == 40.0
    assert source_time == cutoff - pd.Timedelta(hours=9)
    assert cutoff == pd.Timestamp("2025-03-30 06:00:00", tz="UTC")
    assert calls[0]["series"] == spec.series
    assert calls[0]["revision_date"] == cutoff


@pytest.mark.parametrize(
    "payload",
    [
        pd.Series([42.0], index=[pd.Timestamp("2025-01-02")]),
        pd.DataFrame(
            {"value": [42.0]},
            index=[pd.Timestamp("2025-01-02")],
        ),
        pd.DataFrame(
            {"price": [42.0], "provider_flag": ["settled"]},
            index=[pd.Timestamp("2025-01-02")],
        ),
    ],
)
def test_normalizer_accepts_saturn_series_and_named_value_frames(
    payload: pd.Series | pd.DataFrame,
) -> None:
    result = fuel._normalise_market_series(payload, series="fuel.test")

    assert result.iloc[0] == 42.0
    assert result.index.tz is not None


def test_initial_bundle_is_hourly_finite_audited_and_dst_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fake_fetch(
        days: list[pd.Timestamp],
        **_: object,
    ) -> pd.DataFrame:
        requests.append((days[0], days[-1]))
        return _fake_daily(days)

    monkeypatch.setattr(fuel, "_fetch_daily_market", fake_fetch)
    result = fuel.main(_arguments(tmp_path, "2025-03-29", "2025-03-31"))

    assert result == 0
    assert requests == [
        (pd.Timestamp("2025-03-09"), pd.Timestamp("2025-03-31"))
    ]
    output = tmp_path / fuel.OUTPUT_NAME
    audit_path = output.with_name(output.name + fuel.AUDIT_SUFFIX)
    frame = pd.read_parquet(output)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    assert tuple(frame.columns) == fuel.OUTPUT_COLUMNS
    assert len(frame) == 24 + 23 + 24
    assert np.isfinite(frame.loc[:, list(fuel.FEATURE_COLUMNS)]).all().all()
    assert frame["value_time_utc"].is_unique
    assert pd.Timestamp(
        frame.loc[
            pd.to_datetime(frame["value_time_utc"], utc=True)
            .dt.tz_convert(fuel.TIMEZONE)
            .dt.date
            == pd.Timestamp("2025-03-31").date(),
            "snapshot_time_utc",
        ].iloc[0]
    ) == pd.Timestamp("2025-03-30 06:00:00", tz="UTC")
    expected_ccgt = (
        frame["ttf_m1_eur_mwh_th"] / 0.58
        + frame["eua_first_dec_eur_tco2"] * 0.36
        + 3.0
    )
    assert np.allclose(frame["ccgt_marginal_cost_eur_mwh"], expected_ccgt)
    assert np.allclose(frame["ttf_change_1d"], 0.01)
    assert np.allclose(frame["ttf_change_5d"], 0.05)
    assert np.allclose(frame["eua_change_1d"], 0.02)
    assert np.allclose(frame["eua_change_5d"], 0.10)
    assert audit["series"]["ttf_m1_eur_mwh_th"] == (
        "gas.ttf.price.everyday.month.1.ice.eurmwh"
    )
    assert audit["series"]["eua_first_dec_eur_tco2"] == (
        "carbon.eu.price.everyday.eua.ice.1st.dec"
    )
    assert audit["information_type"] == "market_observation_known_before_cutoff"
    assert audit["causality_violations"] == 0
    assert audit["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_incremental_update_preserves_prefix_and_reuse_does_not_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fake_fetch(
        days: list[pd.Timestamp],
        **_: object,
    ) -> pd.DataFrame:
        requests.append((days[0], days[-1]))
        return _fake_daily(days)

    monkeypatch.setattr(fuel, "_fetch_daily_market", fake_fetch)
    fuel.main(_arguments(tmp_path, "2025-01-01", "2025-01-03"))
    output = tmp_path / fuel.OUTPUT_NAME
    prefix = pd.read_parquet(output)

    fuel.main(_arguments(tmp_path, "2025-01-01", "2025-01-05"))
    extended = pd.read_parquet(output)

    pd.testing.assert_frame_equal(
        extended.iloc[: len(prefix)].reset_index(drop=True),
        prefix.reset_index(drop=True),
    )
    assert requests[-1] == (
        pd.Timestamp("2024-12-15"),
        pd.Timestamp("2025-01-05"),
    )
    assert len(extended) == 5 * 24
    audit = json.loads(
        output.with_name(output.name + fuel.AUDIT_SUFFIX).read_text(encoding="utf-8")
    )
    assert audit["start_day"] == "2025-01-01"
    assert audit["end_day"] == "2025-01-05"

    def forbidden_fetch(*_: object, **__: object) -> pd.DataFrame:
        raise AssertionError("Une couverture deja valide ne doit pas etre requetee.")

    monkeypatch.setattr(fuel, "_fetch_daily_market", forbidden_fetch)
    assert fuel.main(_arguments(tmp_path, "2025-01-01", "2025-01-05")) == 0


def test_incremental_update_refuses_a_tampered_checksum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fuel,
        "_fetch_daily_market",
        lambda days, **_: _fake_daily(days),
    )
    fuel.main(_arguments(tmp_path, "2025-02-01", "2025-02-02"))
    output = tmp_path / fuel.OUTPUT_NAME
    audit_path = output.with_name(output.name + fuel.AUDIT_SUFFIX)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["sha256"] = "0" * 64
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(fuel.FuelMaterializationError, match="Checksum"):
        fuel.main(_arguments(tmp_path, "2025-02-01", "2025-02-03"))


@pytest.mark.parametrize(
    ("start", "end", "expected_rows"),
    [
        ("2025-03-29", "2025-03-31", 24 + 23 + 24),
        ("2024-10-26", "2024-10-28", 24 + 25 + 24),
    ],
)
def test_residual_load_sidecar_is_pit_exact_audited_and_dst_safe(
    tmp_path: Path,
    start: str,
    end: str,
    expected_rows: int,
) -> None:
    vintage_root = tmp_path / "vintages"
    output_dir = tmp_path / "output"
    expected = _write_residual_vintages(
        vintage_root,
        start=start,
        end=end,
    )

    output, audit_path = fuel._materialize_residual_load_market_features(
        start_day=pd.Timestamp(start),
        end_day=pd.Timestamp(end),
        output_dir=output_dir,
        source_mode="local-vintages",
        vintage_root=vintage_root,
    )

    frame = pd.read_parquet(output)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert tuple(frame.columns) == fuel.RESIDUAL_LOAD_OUTPUT_COLUMNS
    assert len(frame) == expected_rows
    assert pd.DatetimeIndex(frame["value_time_utc"]).equals(expected)
    cutoffs = fuel._cutoffs_for_value_times(expected)
    assert pd.DatetimeIndex(frame["cutoff_time_utc"]).equals(cutoffs)
    assert pd.DatetimeIndex(frame["snapshot_time_utc"]).equals(
        cutoffs - pd.Timedelta(minutes=20)
    )
    assert pd.DatetimeIndex(frame["revision_time_utc"]).equals(
        cutoffs - pd.Timedelta(minutes=10)
    )
    for alias_number, alias in enumerate(fuel.RESIDUAL_LOAD_ALIASES):
        assert np.array_equal(
            frame[alias].to_numpy(dtype=float),
            alias_number * 1000 + np.arange(expected_rows, dtype=float),
        )
        assert 777_777.0 not in set(frame[alias])
        assert 888_888.0 not in set(frame[alias])
        assert 999_999.0 not in set(frame[alias])
        raw = vintage_root / f"{alias}.parquet"
        assert audit["raw_input_sha256"][alias] == hashlib.sha256(
            raw.read_bytes()
        ).hexdigest()
    assert audit["causality_violations"] == 0
    assert audit["source_mode"] == "local-vintages"
    assert audit["fill_or_interpolation"] == (
        "none; exact physical-hour coverage required"
    )
    assert audit["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert audit["output_sha256"] == audit["sha256"]


def test_residual_load_sidecar_fails_closed_on_missing_hour(
    tmp_path: Path,
) -> None:
    vintage_root = tmp_path / "vintages"
    expected = _write_residual_vintages(
        vintage_root,
        start="2025-01-01",
        end="2025-01-02",
    )
    alias = fuel.RESIDUAL_LOAD_ALIASES[2]
    path = vintage_root / f"{alias}.parquet"
    raw = pd.read_parquet(path)
    raw = raw.loc[raw["value_time_utc"] != expected[7]]
    raw.to_parquet(path, index=False)

    with pytest.raises(fuel.FuelMaterializationError, match="sans vintage causale"):
        fuel._materialize_residual_load_market_features(
            start_day=pd.Timestamp("2025-01-01"),
            end_day=pd.Timestamp("2025-01-02"),
            output_dir=tmp_path / "output",
            source_mode="local-vintages",
            vintage_root=vintage_root,
        )


def test_residual_load_source_defaults_to_operational_saturn(
    tmp_path: Path,
) -> None:
    args = fuel.parse_args(
        [
            "--start-day",
            "2025-01-01",
            "--end-day",
            "2025-01-02",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert args.residual_load_source == "saturn"
    assert args.residual_start_day == "2025-01-01"

    distinct = fuel.parse_args(
        [
            "--start-day",
            "2024-06-30",
            "--residual-start-day",
            "2024-08-20",
            "--end-day",
            "2026-08-20",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert distinct.residual_start_day == "2024-08-20"


@pytest.mark.parametrize(
    ("start", "end", "expected_rows"),
    [
        ("2025-03-29", "2025-03-31", 24 + 23 + 24),
        ("2024-10-26", "2024-10-28", 24 + 25 + 24),
    ],
)
def test_saturn_residual_bank_queries_exact_cutoffs_and_dst_grid_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    start: str,
    end: str,
    expected_rows: int,
) -> None:
    calls: list[dict[str, object]] = []
    spec_by_series = {spec.series: spec for spec in fuel.RESIDUAL_LOAD_SERIES}

    def fake_fetch(
        _client: object,
        series_name: str,
        query_start: pd.Timestamp,
        query_end: pd.Timestamp,
        timezone: str,
        **kwargs: object,
    ) -> pd.Series:
        calls.append(
            {
                "series": series_name,
                "start": query_start,
                "end": query_end,
                "timezone": timezone,
                **kwargs,
            }
        )
        spec = spec_by_series[series_name]
        local_start = query_start + pd.Timedelta(hours=8)
        day = local_start.normalize().tz_localize(None)
        expected = fuel._physical_utc_index(day)
        alias_number = fuel.RESIDUAL_LOAD_ALIASES.index(spec.alias)
        values = alias_number * 1000 + np.arange(len(expected), dtype=float)
        if spec.alias == fuel.NL_SPRING_DST_REPAIR_ALIAS and len(expected) == 23:
            local = expected.tz_convert(spec.delivery_timezone)
            keep = ~local.strftime("%H:%M").isin(
                fuel.NL_SPRING_DST_REPAIR_LOCAL_TIMES
            )
            expected = expected[keep]
            values = values[keep]
        return pd.Series(
            values,
            index=expected.tz_convert(timezone),
            name=series_name,
        )

    monkeypatch.setattr(fuel, "_client", lambda _timeout: object())
    monkeypatch.setattr(fuel, "fetch_saturn_series_from_client", fake_fetch)
    output, audit_path = fuel._materialize_residual_load_market_features(
        start_day=pd.Timestamp(start),
        end_day=pd.Timestamp(end),
        output_dir=tmp_path / "output",
        source_mode="saturn",
        series_workers=2,
        day_workers=2,
        retries=1,
    )

    frame = pd.read_parquet(output)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = fuel._expected_hourly_index(pd.Timestamp(start), pd.Timestamp(end))
    cutoffs = fuel._cutoffs_for_value_times(expected)
    assert len(frame) == expected_rows
    assert pd.DatetimeIndex(frame["value_time_utc"]).equals(expected)
    assert pd.DatetimeIndex(frame["snapshot_time_utc"]).equals(cutoffs)
    assert pd.DatetimeIndex(frame["revision_time_utc"]).equals(cutoffs)
    assert len(calls) == len(fuel.RESIDUAL_LOAD_SERIES) * 3
    for call in calls:
        spec = spec_by_series[str(call["series"])]
        query_day = (pd.Timestamp(call["start"]) + pd.Timedelta(hours=8))
        query_day = query_day.normalize().tz_localize(None)
        assert call["revision_date"] == fuel._civil_cutoff(query_day).tz_convert(
            "UTC"
        )
        assert call["timezone"] == spec.delivery_timezone
        assert call["naive_timezone"] == spec.naive_timezone
        assert call["incomplete_dst_policy"] == "duplicate"
    assert audit["source_mode"] == "saturn"
    assert audit["schema_version"] == fuel.RESIDUAL_LOAD_SCHEMA_VERSION
    assert audit["series"] == {
        spec.alias: spec.series for spec in fuel.RESIDUAL_LOAD_SERIES
    }
    assert audit["naive_timezones"]["nl_residual_load_fcst"] == (
        "UTC"
    )
    assert audit["revision_query"] == (
        "revision_date=cutoff_time_utc for every delivery day"
    )
    assert audit["fill_or_interpolation"] == fuel.SATURN_RESIDUAL_FILL_POLICY
    expected_repairs = 2 if expected_rows == 24 + 23 + 24 else 0
    assert audit["spring_dst_repair_count"][
        fuel.NL_SPRING_DST_REPAIR_ALIAS
    ] == expected_repairs
    repair_details = audit["spring_dst_repair_details"][
        fuel.NL_SPRING_DST_REPAIR_ALIAS
    ]
    assert len(repair_details) == expected_repairs
    if repair_details:
        assert [
            pd.Timestamp(detail["value_time_local"]).strftime("%H:%M")
            for detail in repair_details
        ] == list(fuel.NL_SPRING_DST_REPAIR_LOCAL_TIMES)
        for detail in repair_details:
            assert detail["method"] == "linear_mean"
            assert detail["same_asof_forecast"] is True
            assert len(detail["donor_value_times_utc"]) == 2
    assert audit["causality_violations"] == 0


@pytest.mark.parametrize("day_text", ["2025-03-30", "2026-03-29"])
def test_nl_spring_dst_repairs_only_missing_04_and_06_from_same_forecast(
    monkeypatch: pytest.MonkeyPatch,
    day_text: str,
) -> None:
    spec = next(
        item
        for item in fuel.RESIDUAL_LOAD_SERIES
        if item.alias == fuel.NL_SPRING_DST_REPAIR_ALIAS
    )
    assert spec.naive_timezone == "UTC"
    day = pd.Timestamp(day_text)
    expected = fuel._physical_utc_index(day)
    local = expected.tz_convert(spec.delivery_timezone)
    missing_mask = local.strftime("%H:%M").isin(
        fuel.NL_SPRING_DST_REPAIR_LOCAL_TIMES
    )
    missing_utc = expected[missing_mask]
    source_index = expected[~missing_mask]
    source_values = pd.Series(
        np.arange(len(expected), dtype=float)[~missing_mask],
        index=source_index.tz_localize(None),
        name=spec.series,
    )
    calls: list[dict[str, object]] = []

    class Client:
        def get(self, _series: str, **kwargs: object) -> pd.Series:
            calls.append(kwargs)
            return source_values

    monkeypatch.setattr(fuel, "_client", lambda _timeout: Client())
    repaired = fuel._query_residual_saturn_day(
        spec,
        day,
        retries=1,
        timeout_seconds=1.0,
    )

    assert repaired.index.equals(expected)
    assert calls[0]["revision_date"] == fuel._civil_cutoff(day).tz_convert("UTC")
    marker = repaired[f"{spec.alias}__spring_dst_repair"]
    assert int(marker.sum()) == 2
    assert bool(marker.loc[missing_utc].all())
    for timestamp in missing_utc:
        previous = float(
            repaired.loc[timestamp - pd.Timedelta(hours=1), spec.alias]
        )
        following = float(
            repaired.loc[timestamp + pd.Timedelta(hours=1), spec.alias]
        )
        assert float(repaired.loc[timestamp, spec.alias]) == pytest.approx(
            (previous + following) / 2.0
        )


@pytest.mark.parametrize(
    (
        "day_text",
        "missing_local_times",
        "add_duplicate",
        "add_extra",
        "nan_neighbour",
    ),
    [
        ("2025-03-30", ("06:00",), False, False, False),
        ("2025-03-30", ("04:00", "06:00", "08:00"), False, False, False),
        ("2025-03-30", ("04:00", "06:00"), True, False, False),
        ("2025-03-30", ("04:00", "06:00"), False, True, False),
        ("2025-03-30", ("04:00", "06:00"), False, False, True),
        ("2025-03-29", ("04:00", "06:00"), False, False, False),
    ],
)
def test_nl_spring_dst_repair_rejects_every_other_signature(
    monkeypatch: pytest.MonkeyPatch,
    day_text: str,
    missing_local_times: tuple[str, ...],
    add_duplicate: bool,
    add_extra: bool,
    nan_neighbour: bool,
) -> None:
    spec = next(
        item
        for item in fuel.RESIDUAL_LOAD_SERIES
        if item.alias == fuel.NL_SPRING_DST_REPAIR_ALIAS
    )
    day = pd.Timestamp(day_text)
    expected = fuel._physical_utc_index(day)
    local = expected.tz_convert(spec.delivery_timezone)
    keep = ~local.strftime("%H:%M").isin(missing_local_times)
    source_index = expected[keep]
    values = np.arange(len(expected), dtype=float)[keep]
    if nan_neighbour:
        local_kept = source_index.tz_convert(spec.delivery_timezone)
        values[local_kept.strftime("%H:%M") == "03:00"] = np.nan
    if add_duplicate:
        source_index = source_index.append(pd.DatetimeIndex([source_index[0]]))
        values = np.append(values, values[0])
    if add_extra:
        source_index = source_index.append(
            pd.DatetimeIndex([source_index[0] + pd.Timedelta(minutes=30)])
        )
        values = np.append(values, values[0])
    source = pd.Series(
        values,
        index=source_index.tz_convert(spec.delivery_timezone),
        name=spec.series,
    )

    monkeypatch.setattr(fuel, "_client", lambda _timeout: object())
    monkeypatch.setattr(
        fuel,
        "fetch_saturn_series_from_client",
        lambda *_args, **_kwargs: source,
    )
    with pytest.raises(
        fuel.FuelMaterializationError,
        match="couverture Saturn incomplete",
    ):
        fuel._query_residual_saturn_day(
            spec,
            day,
            retries=1,
            timeout_seconds=1.0,
        )


def test_saturn_residual_cache_rejects_tampered_spring_repair_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    day = pd.Timestamp("2025-03-30")

    def fake_bank(
        days: list[pd.Timestamp],
        **_kwargs: object,
    ) -> dict[str, pd.DataFrame]:
        assert days == [day]
        expected = fuel._physical_utc_index(day)
        cutoffs = fuel._cutoffs_for_value_times(expected)
        local = expected.tz_convert("Europe/Amsterdam")
        repaired = local.strftime("%H:%M").isin(
            fuel.NL_SPRING_DST_REPAIR_LOCAL_TIMES
        )
        return {
            alias: pd.DataFrame(
                {
                    alias: np.arange(len(expected), dtype=float),
                    f"{alias}__snapshot_time_utc": cutoffs,
                    f"{alias}__revision_time_utc": cutoffs,
                    f"{alias}__spring_dst_repair": (
                        repaired
                        if alias == fuel.NL_SPRING_DST_REPAIR_ALIAS
                        else False
                    ),
                },
                index=expected,
            )
            for alias in fuel.RESIDUAL_LOAD_ALIASES
        }

    monkeypatch.setattr(fuel, "_fetch_residual_saturn_bank", fake_bank)
    output, audit_path = fuel._materialize_residual_load_market_features(
        start_day=day,
        end_day=day,
        output_dir=tmp_path,
        source_mode="saturn",
    )
    assert output.is_file()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    detail = audit["spring_dst_repair_details"][
        fuel.NL_SPRING_DST_REPAIR_ALIAS
    ][0]
    detail["donor_value_times_utc"][0] = pd.Timestamp(
        "2025-03-30 00:00", tz="UTC"
    ).isoformat()
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(
        fuel.FuelMaterializationError,
        match="Ledger des reparations DST contient un instant non autorise",
    ):
        fuel._materialize_residual_load_market_features(
            start_day=day,
            end_day=day,
            output_dir=tmp_path,
            source_mode="saturn",
        )


def test_saturn_residual_bank_fails_closed_on_incomplete_day_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken_series = fuel.RESIDUAL_LOAD_SERIES[0].series

    def fake_fetch(
        _client: object,
        series_name: str,
        query_start: pd.Timestamp,
        _query_end: pd.Timestamp,
        timezone: str,
        **_kwargs: object,
    ) -> pd.Series:
        local_start = query_start + pd.Timedelta(hours=8)
        day = local_start.normalize().tz_localize(None)
        expected = fuel._physical_utc_index(day)
        if series_name == broken_series:
            expected = expected[:-1]
        return pd.Series(
            np.arange(len(expected), dtype=float),
            index=expected.tz_convert(timezone),
            name=series_name,
        )

    monkeypatch.setattr(fuel, "_client", lambda _timeout: object())
    monkeypatch.setattr(fuel, "fetch_saturn_series_from_client", fake_fetch)
    with pytest.raises(fuel.FuelMaterializationError, match=r"jour\(s\) Saturn"):
        fuel._materialize_residual_load_market_features(
            start_day=pd.Timestamp("2025-01-01"),
            end_day=pd.Timestamp("2025-01-01"),
            output_dir=tmp_path / "output",
            source_mode="saturn",
            series_workers=2,
            day_workers=2,
            retries=1,
        )


def test_saturn_residual_series_recovers_only_failed_days_sequentially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = fuel.RESIDUAL_LOAD_SERIES[0]
    days = list(pd.date_range("2025-01-01", "2025-01-02", freq="D"))
    transient_day = days[0]
    attempts: dict[pd.Timestamp, int] = {day: 0 for day in days}

    def fake_query(
        current_spec: fuel.ResidualLoadSeries,
        day: pd.Timestamp,
        **_kwargs: object,
    ) -> pd.DataFrame:
        attempts[day] += 1
        if day == transient_day and attempts[day] == 1:
            raise fuel.FuelMaterializationError("proxy transitoire")
        expected = fuel._physical_utc_index(day)
        cutoff = fuel._civil_cutoff(day).tz_convert("UTC")
        return pd.DataFrame(
            {
                current_spec.alias: np.arange(len(expected), dtype=float),
                f"{current_spec.alias}__snapshot_time_utc": cutoff,
                f"{current_spec.alias}__revision_time_utc": cutoff,
            },
            index=expected,
        )

    monkeypatch.setattr(fuel, "_query_residual_saturn_day", fake_query)
    frame = fuel._fetch_residual_saturn_series(
        spec,
        days,
        day_workers=2,
        retries=1,
        timeout_seconds=1.0,
    )

    assert len(frame) == 48
    assert attempts[transient_day] == 2
    assert attempts[days[1]] == 1


def test_saturn_residual_sidecar_reuses_prefix_and_appends_only_missing_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fake_bank(
        days: list[pd.Timestamp],
        **_kwargs: object,
    ) -> dict[str, pd.DataFrame]:
        requests.append((days[0], days[-1]))
        expected = fuel._expected_hourly_index(days[0], days[-1])
        cutoff = fuel._cutoffs_for_value_times(expected)
        hour_number = expected.asi8.astype(float) / float(pd.Timedelta(hours=1).value)
        return {
            alias: pd.DataFrame(
                {
                        alias: hour_number + alias_number * 100_000.0,
                        f"{alias}__snapshot_time_utc": cutoff,
                        f"{alias}__revision_time_utc": cutoff,
                        f"{alias}__spring_dst_repair": False,
                },
                index=expected,
            )
            for alias_number, alias in enumerate(fuel.RESIDUAL_LOAD_ALIASES)
        }

    monkeypatch.setattr(fuel, "_fetch_residual_saturn_bank", fake_bank)
    output_dir = tmp_path / "output"
    output, _ = fuel._materialize_residual_load_market_features(
        start_day=pd.Timestamp("2025-01-01"),
        end_day=pd.Timestamp("2025-01-02"),
        output_dir=output_dir,
        source_mode="saturn",
    )
    prefix = pd.read_parquet(output)

    output, audit_path = fuel._materialize_residual_load_market_features(
        start_day=pd.Timestamp("2025-01-02"),
        end_day=pd.Timestamp("2025-01-04"),
        output_dir=output_dir,
        source_mode="saturn",
    )
    extended = pd.read_parquet(output)
    pd.testing.assert_frame_equal(
        extended.iloc[: len(prefix)].reset_index(drop=True),
        prefix.reset_index(drop=True),
    )
    assert requests == [
        (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")),
        (pd.Timestamp("2025-01-03"), pd.Timestamp("2025-01-04")),
    ]
    assert len(extended) == 4 * 24
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["start_day"] == "2025-01-01"
    assert audit["requested_start_day"] == "2025-01-02"
    assert audit["end_day"] == "2025-01-04"
    assert audit["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()

    # Exact same request is a pure validated reuse: no Saturn call and no write.
    before_bytes = output.read_bytes()
    fuel._materialize_residual_load_market_features(
        start_day=pd.Timestamp("2025-01-02"),
        end_day=pd.Timestamp("2025-01-04"),
        output_dir=output_dir,
        source_mode="saturn",
    )
    assert requests[-1] == (
        pd.Timestamp("2025-01-03"),
        pd.Timestamp("2025-01-04"),
    )
    assert output.read_bytes() == before_bytes

    # A validated superset is safe for a retrospective run: the operational
    # join reindexes the source onto that archive's timeline, so later rows
    # cannot enter the model.
    fuel._materialize_residual_load_market_features(
        start_day=pd.Timestamp("2025-01-03"),
        end_day=pd.Timestamp("2025-01-03"),
        output_dir=output_dir,
        source_mode="saturn",
    )
    assert requests[-1] == (
        pd.Timestamp("2025-01-03"),
        pd.Timestamp("2025-01-04"),
    )
    assert output.read_bytes() == before_bytes

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["output_sha256"] = "0" * 64
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(fuel.FuelMaterializationError, match="Checksum"):
        fuel._materialize_residual_load_market_features(
            start_day=pd.Timestamp("2025-01-02"),
            end_day=pd.Timestamp("2025-01-04"),
            output_dir=output_dir,
            source_mode="saturn",
        )


def test_residual_sidecar_reuse_fails_closed_on_partial_bundle(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / fuel.RESIDUAL_LOAD_OUTPUT_NAME).write_bytes(b"partial")

    with pytest.raises(fuel.FuelMaterializationError, match="incomplet"):
        fuel._materialize_residual_load_market_features(
            start_day=pd.Timestamp("2025-01-01"),
            end_day=pd.Timestamp("2025-01-01"),
            output_dir=output_dir,
            source_mode="saturn",
        )


def test_residual_load_sidecar_fails_closed_on_non_finite_selected_value(
    tmp_path: Path,
) -> None:
    vintage_root = tmp_path / "vintages"
    expected = _write_residual_vintages(
        vintage_root,
        start="2025-01-01",
        end="2025-01-01",
    )
    alias = fuel.RESIDUAL_LOAD_ALIASES[4]
    path = vintage_root / f"{alias}.parquet"
    raw = pd.read_parquet(path)
    raw.loc[raw["value_time_utc"] == expected[11], "value"] = np.nan
    raw.to_parquet(path, index=False)

    with pytest.raises(fuel.FuelMaterializationError, match="valeur non finie"):
        fuel._materialize_residual_load_market_features(
            start_day=pd.Timestamp("2025-01-01"),
            end_day=pd.Timestamp("2025-01-01"),
            output_dir=tmp_path / "output",
            source_mode="local-vintages",
            vintage_root=vintage_root,
        )
