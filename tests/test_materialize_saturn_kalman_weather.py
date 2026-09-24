from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from materialize_saturn_kalman_weather import (
    CATALOG_SCHEMA_VERSION,
    CATALOG_JSON,
    CATALOG_YAML,
    MATERIALIZER,
    build_catalog,
    build_command,
    build_plan,
    incremental_extension_start,
    reject_in_place_semantic_rewrite,
    reusable_output,
    write_catalog,
)
from chronos2_modular.saturn import normalize_saturn_series


def _argument(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_plan_uses_exact_saturn_weather_contracts(tmp_path: Path) -> None:
    plans = build_plan(["FR,DE", "BE", "NL,ES"], tmp_path)

    assert len(plans) == 15
    by_alias = {plan.alias: plan for plan in plans}
    assert by_alias["fr_wind_generation_fcst"].series == (
        "power.fr.generation.wind.hourly.gw.fcst"
    )
    assert by_alias["de_solar_generation_fcst"].series == (
        "power.de.generation.solar.hourly.gw.fcst"
    )
    assert by_alias["be_temperature_fcst"].series == (
        "meteo.nrjscan.be.t_2m.index.fcst.d"
    )
    assert by_alias["be_temperature_fcst"].daily_broadcast is True
    assert by_alias["es_temperature_fcst"].timezone == "Europe/Madrid"
    assert by_alias["nl_wind_generation_fcst"].series == (
        "power.nl.prod.total.wind.mw.ecmwf_avg.pointconnect.6h.cache"
    )
    assert by_alias["nl_wind_generation_fcst"].naive_timezone == "UTC"
    assert by_alias["nl_wind_generation_fcst"].value_scale == 0.001
    assert by_alias["nl_solar_generation_fcst"].series == (
        "power.nl.generation.solar.hourly.gw.fcst"
    )
    assert by_alias["nl_solar_generation_fcst"].naive_timezone == (
        "Europe/Amsterdam"
    )
    assert by_alias["fr_solar_generation_fcst"].naive_timezone == "Europe/Paris"


def test_commands_are_argv_safe_and_apply_dst_and_daily_contracts(
    tmp_path: Path,
) -> None:
    plans = build_plan(["FR", "NL"], tmp_path)
    by_alias = {plan.alias: plan for plan in plans}
    start = pd.Timestamp("2025-08-01")
    end = pd.Timestamp("2026-08-01")

    temperature = build_command(
        by_alias["fr_temperature_fcst"],
        start_day=start,
        end_day=end,
        day_workers=3,
    )
    assert temperature[0]
    assert temperature[1] == str(MATERIALIZER)
    assert _argument(temperature, "--timezone") == "Europe/Paris"
    assert _argument(temperature, "--cutoff-timezone") == "Europe/Paris"
    assert _argument(temperature, "--cutoff-time") == "08:00"
    assert _argument(temperature, "--workers") == "3"
    assert _argument(temperature, "--incomplete-dst-policy") == "duplicate"
    assert "--daily-broadcast" in temperature

    nl_wind = build_command(
        by_alias["nl_wind_generation_fcst"],
        start_day=start,
        end_day=end,
        day_workers=2,
    )
    assert "--daily-broadcast" not in nl_wind
    assert _argument(nl_wind, "--naive-timezone") == "UTC"
    assert _argument(nl_wind, "--value-scale") == "0.001"

    nl_solar = build_command(
        by_alias["nl_solar_generation_fcst"],
        start_day=start,
        end_day=end,
        day_workers=2,
    )
    assert _argument(nl_solar, "--timezone") == "Europe/Amsterdam"
    assert _argument(nl_solar, "--cutoff-timezone") == "Europe/Amsterdam"
    assert _argument(nl_solar, "--naive-timezone") == "Europe/Amsterdam"
    assert _argument(nl_solar, "--incomplete-dst-policy") == (
        "duplicate_zero_only"
    )
    assert _argument(nl_solar, "--request-padding-hours") == "8"


@pytest.mark.parametrize(
    "delivery_day",
    ["2025-01-15", "2025-03-30", "2025-07-15", "2025-10-26"],
)
def test_nl_solar_local_formula_matches_utc_aware_primary_on_all_seasons(
    tmp_path: Path,
    delivery_day: str,
) -> None:
    plan = {
        value.alias: value for value in build_plan(["NL"], tmp_path)
    }["nl_solar_generation_fcst"]
    local_start = pd.Timestamp(delivery_day).tz_localize(plan.timezone)
    local_end = (pd.Timestamp(delivery_day) + pd.Timedelta(days=1)).tz_localize(
        plan.timezone
    )
    physical = pd.date_range(
        local_start.tz_convert("UTC"),
        local_end.tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    # Mirror the audited Saturn definition:
    # (0.3 * naive(ECMWF_UTC, "CET") + 0.7 * Meteologica_local) / 1000.
    ecmwf_mw = pd.Series(
        100.0 + pd.RangeIndex(len(physical)).to_numpy(dtype=float),
        index=physical,
    )
    meteologica_mw = pd.Series(
        300.0 + 2.0 * pd.RangeIndex(len(physical)).to_numpy(dtype=float),
        index=physical,
    )
    local_labels = physical.tz_convert(plan.timezone).tz_localize(None)
    raw_formula = pd.Series(
        (0.3 * ecmwf_mw.to_numpy() + 0.7 * meteologica_mw.to_numpy()) / 1000.0,
        index=local_labels,
        dtype=float,
    )

    normalised = normalize_saturn_series(
        raw_formula,
        plan.series,
        plan.timezone,
        naive_timezone=plan.naive_timezone,
        incomplete_dst_policy="raise",
    )

    assert normalised.index.tz_convert("UTC").equals(physical)
    assert normalised.to_numpy().tolist() == raw_formula.to_numpy().tolist()


def test_nl_solar_utc_interpretation_is_rejected_by_spring_physical_grid(
    tmp_path: Path,
) -> None:
    plan = {
        value.alias: value for value in build_plan(["NL"], tmp_path)
    }["nl_solar_generation_fcst"]
    day = pd.Timestamp("2025-03-30")
    physical = pd.date_range(
        day.tz_localize(plan.timezone).tz_convert("UTC"),
        (day + pd.Timedelta(days=1)).tz_localize(plan.timezone).tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    local_labels = physical.tz_convert(plan.timezone).tz_localize(None)
    raw_formula = pd.Series(range(len(local_labels)), index=local_labels, dtype=float)

    wrongly_as_utc = normalize_saturn_series(
        raw_formula,
        plan.series,
        plan.timezone,
        naive_timezone="UTC",
        incomplete_dst_policy="raise",
    )

    assert not wrongly_as_utc.index.tz_convert("UTC").equals(physical)
    assert len(physical.difference(wrongly_as_utc.index.tz_convert("UTC"))) > 0


def test_nl_solar_collapsed_autumn_fold_requires_a_finite_zero(
    tmp_path: Path,
) -> None:
    plan = {
        value.alias: value for value in build_plan(["NL"], tmp_path)
    }["nl_solar_generation_fcst"]
    day = pd.Timestamp("2025-10-26")
    physical = pd.date_range(
        day.tz_localize(plan.timezone).tz_convert("UTC"),
        (day + pd.Timedelta(days=1)).tz_localize(plan.timezone).tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    labels = physical.tz_convert(plan.timezone).tz_localize(None)
    collapsed = pd.Series(range(len(labels)), index=labels, dtype=float).groupby(
        level=0
    ).last()

    with pytest.raises(ValueError, match="zero fini prouve"):
        normalize_saturn_series(
            collapsed,
            plan.series,
            plan.timezone,
            naive_timezone=plan.naive_timezone,
            incomplete_dst_policy="duplicate_zero_only",
        )

    collapsed.loc[pd.Timestamp("2025-10-26 02:00")] = 0.0
    repaired = normalize_saturn_series(
        collapsed,
        plan.series,
        plan.timezone,
        naive_timezone=plan.naive_timezone,
        incomplete_dst_policy="duplicate_zero_only",
    )
    assert repaired.index.tz_convert("UTC").equals(physical)
    assert len(repaired.attrs["dst_repairs"]) == 1
    evidence = repaired.attrs["dst_repairs"][0]
    assert evidence["duplicated_value"] == 0.0
    assert len(evidence["physical_hours_utc"]) == 2


def test_nl_solar_zero_only_rejects_nan_and_malformed_autumn_duplicates(
    tmp_path: Path,
) -> None:
    plan = {
        value.alias: value for value in build_plan(["NL"], tmp_path)
    }["nl_solar_generation_fcst"]
    day = pd.Timestamp("2025-10-26")
    physical = pd.date_range(
        day.tz_localize(plan.timezone).tz_convert("UTC"),
        (day + pd.Timedelta(days=1)).tz_localize(plan.timezone).tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    labels = physical.tz_convert(plan.timezone).tz_localize(None)
    singleton = pd.Series(0.0, index=labels).groupby(level=0).last()
    ambiguous = pd.Timestamp("2025-10-26 02:00")

    singleton.loc[ambiguous] = float("nan")
    with pytest.raises(ValueError, match="zero fini prouve"):
        normalize_saturn_series(
            singleton,
            plan.series,
            plan.timezone,
            naive_timezone=plan.naive_timezone,
            incomplete_dst_policy="duplicate_zero_only",
        )

    three_folds = pd.concat(
        [
            pd.Series(range(len(labels)), index=labels, dtype=float),
            pd.Series([0.0], index=pd.DatetimeIndex([ambiguous])),
        ]
    )
    with pytest.raises(ValueError, match="apparait 3 fois|apparaît 3 fois"):
        normalize_saturn_series(
            three_folds,
            plan.series,
            plan.timezone,
            naive_timezone=plan.naive_timezone,
            incomplete_dst_policy="duplicate_zero_only",
        )


def test_nl_solar_zero_only_preserves_two_real_folds_and_rejects_wrong_duplicate(
    tmp_path: Path,
) -> None:
    plan = {
        value.alias: value for value in build_plan(["NL"], tmp_path)
    }["nl_solar_generation_fcst"]
    day = pd.Timestamp("2025-10-26")
    physical = pd.date_range(
        day.tz_localize(plan.timezone).tz_convert("UTC"),
        (day + pd.Timedelta(days=1)).tz_localize(plan.timezone).tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    labels = physical.tz_convert(plan.timezone).tz_localize(None)
    genuine = pd.Series(range(len(labels)), index=labels, dtype=float)

    normalised = normalize_saturn_series(
        genuine,
        plan.series,
        plan.timezone,
        naive_timezone=plan.naive_timezone,
        incomplete_dst_policy="duplicate_zero_only",
    )
    assert normalised.index.tz_convert("UTC").equals(physical)
    assert normalised.attrs["dst_repairs"] == []

    wrong_duplicate = pd.concat(
        [
            genuine,
            pd.Series(
                [99.0],
                index=pd.DatetimeIndex([pd.Timestamp("2025-10-26 03:00")]),
            ),
        ]
    )
    with pytest.raises(ValueError, match="duplicate physique inattendu"):
        normalize_saturn_series(
            wrong_duplicate,
            plan.series,
            plan.timezone,
            naive_timezone=plan.naive_timezone,
            incomplete_dst_policy="duplicate_zero_only",
        )


def test_zero_only_neither_repairs_spring_nor_modifies_a_utc_source(
    tmp_path: Path,
) -> None:
    plan = {
        value.alias: value for value in build_plan(["NL"], tmp_path)
    }["nl_solar_generation_fcst"]
    spring_day = pd.Timestamp("2025-03-30")
    spring_start = spring_day.tz_localize(plan.timezone).tz_convert("UTC")
    spring_end = (spring_day + pd.Timedelta(days=1)).tz_localize(
        plan.timezone
    ).tz_convert("UTC")
    spring_physical = pd.date_range(
        spring_start, spring_end, freq="h", inclusive="left"
    )
    spring_labels = spring_physical.tz_convert(plan.timezone).tz_localize(None)
    nonexistent = pd.concat(
        [
            pd.Series(0.0, index=spring_labels),
            pd.Series(
                [0.0],
                index=pd.DatetimeIndex([pd.Timestamp("2025-03-30 02:00")]),
            ),
        ]
    )
    with pytest.raises(ValueError, match="non desambiguisables|désambiguïsables"):
        normalize_saturn_series(
            nonexistent,
            plan.series,
            plan.timezone,
            naive_timezone=plan.naive_timezone,
            incomplete_dst_policy="duplicate_zero_only",
        )

    autumn_day = pd.Timestamp("2025-10-26")
    autumn_physical = pd.date_range(
        autumn_day.tz_localize(plan.timezone).tz_convert("UTC"),
        (autumn_day + pd.Timedelta(days=1))
        .tz_localize(plan.timezone)
        .tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    utc_naive = pd.Series(
        range(len(autumn_physical)),
        index=autumn_physical.tz_localize(None),
        dtype=float,
    )
    normalised_utc = normalize_saturn_series(
        utc_naive,
        plan.series,
        plan.timezone,
        naive_timezone="UTC",
        incomplete_dst_policy="duplicate_zero_only",
    )
    assert normalised_utc.index.tz_convert("UTC").equals(autumn_physical)
    assert normalised_utc.attrs["dst_repairs"] == []


def test_legacy_nl_solar_zero_fold_cache_is_validated_without_rewrite(
    tmp_path: Path,
) -> None:
    plan = {
        value.alias: value for value in build_plan(["NL"], tmp_path)
    }["nl_solar_generation_fcst"]
    day = pd.Timestamp("2025-10-26")
    index = pd.date_range(
        day.tz_localize(plan.timezone).tz_convert("UTC"),
        (day + pd.Timedelta(days=1)).tz_localize(plan.timezone).tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    local_wall = index.tz_convert(plan.timezone).tz_localize(None)
    values = pd.Series(range(len(index)), dtype=float)
    values.loc[local_wall.duplicated(keep=False)] = 0.0
    cutoff = pd.Timestamp("2025-10-25 06:00", tz="UTC")
    plan.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "value_time_utc": index,
            "snapshot_time_utc": cutoff,
            "revision_time_utc": cutoff,
            "value": values,
            "downloaded_at_utc": pd.Timestamp("2025-10-25 09:00", tz="UTC"),
        }
    ).to_parquet(plan.output, index=False)
    digest = hashlib.sha256(plan.output.read_bytes()).hexdigest()
    plan.audit_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "series": plan.series,
                "alias": plan.alias,
                "timezone": plan.timezone,
                "cutoff_timezone": plan.timezone,
                "cutoff_time": "08:00",
                "naive_timezone": plan.naive_timezone,
                "incomplete_dst_policy": "duplicate",
                "daily_broadcast": False,
                "value_scale": 1.0,
                "start_day": day.date().isoformat(),
                "end_day": day.date().isoformat(),
                "days": 1,
                "rows": len(index),
                "first_delivery_utc": index[0].isoformat(),
                "last_delivery_utc": index[-1].isoformat(),
                "sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    parquet_before = plan.output.read_bytes()
    audit_before = plan.audit_path.read_bytes()
    parquet_mtime = plan.output.stat().st_mtime_ns
    audit_mtime = plan.audit_path.stat().st_mtime_ns

    reusable, reason = reusable_output(plan, start_day=day, end_day=day)

    assert reusable is True, reason
    assert plan.output.read_bytes() == parquet_before
    assert plan.audit_path.read_bytes() == audit_before
    assert plan.output.stat().st_mtime_ns == parquet_mtime
    assert plan.audit_path.stat().st_mtime_ns == audit_mtime


def test_exact_prefix_is_extended_without_redownloading_history(
    tmp_path: Path,
) -> None:
    plan = build_plan(["FR"], tmp_path)[2]
    start = pd.Timestamp("2026-08-28")
    existing_end = pd.Timestamp("2026-08-29")
    requested_end = pd.Timestamp("2026-08-31")
    index = pd.date_range(
        start.tz_localize(plan.timezone).tz_convert("UTC"),
        (existing_end + pd.Timedelta(days=1))
        .tz_localize(plan.timezone)
        .tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    cutoffs = [
        (day - pd.Timedelta(days=1) + pd.Timedelta(hours=8))
        .tz_localize(plan.timezone)
        .tz_convert("UTC")
        for day in index.tz_convert(plan.timezone).normalize().tz_localize(None)
    ]
    plan.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "value_time_utc": index,
            "snapshot_time_utc": cutoffs,
            "revision_time_utc": cutoffs,
            "value": 19.0,
            "downloaded_at_utc": pd.Timestamp("2026-08-29", tz="UTC"),
        }
    ).to_parquet(plan.output, index=False)
    digest = hashlib.sha256(plan.output.read_bytes()).hexdigest()
    plan.audit_path.write_text(
        json.dumps(
            {
                "series": plan.series,
                "alias": plan.alias,
                "timezone": plan.timezone,
                "cutoff_timezone": plan.timezone,
                "cutoff_time": "08:00",
                "naive_timezone": plan.naive_timezone,
                "incomplete_dst_policy": "duplicate",
                "daily_broadcast": True,
                "value_scale": 1.0,
                "start_day": start.date().isoformat(),
                "end_day": existing_end.date().isoformat(),
                "days": 2,
                "rows": len(index),
                "first_delivery_utc": index[0].isoformat(),
                "last_delivery_utc": index[-1].isoformat(),
                "sha256": digest,
            }
        ),
        encoding="utf-8",
    )

    reusable, reason = reusable_output(
        plan,
        start_day=start,
        end_day=start,
    )
    assert reusable is True, reason
    assert "prefixe demande couvert" in reason

    historical_catalog = build_catalog(
        [plan],
        start_day=start,
        end_day=start,
        series_workers=1,
        day_workers=2,
    )
    assert historical_catalog["start_day"] == "2026-08-28"
    assert historical_catalog["end_day"] == "2026-08-29"
    assert historical_catalog["requested_start_day"] == "2026-08-28"
    assert historical_catalog["requested_end_day"] == "2026-08-28"

    extension, reason = incremental_extension_start(
        plan,
        start_day=start,
        end_day=requested_end,
    )

    assert extension == pd.Timestamp("2026-08-30")
    assert "extension incrementale" in reason
    command = build_command(
        plan,
        start_day=extension,
        end_day=requested_end,
        day_workers=2,
        merge_existing=True,
    )
    assert _argument(command, "--start-day") == "2026-08-30"
    assert "--merge-existing" in command


def test_resume_validation_and_catalog_snippets_are_reproducible(
    tmp_path: Path,
) -> None:
    plan = build_plan(["FR"], tmp_path)[2]
    start = pd.Timestamp("2026-08-28")
    end = pd.Timestamp("2026-08-28")
    index = pd.date_range(
        start.tz_localize(plan.timezone).tz_convert("UTC"),
        (end + pd.Timedelta(days=1)).tz_localize(plan.timezone).tz_convert("UTC"),
        freq="h",
        inclusive="left",
    )
    cutoff = pd.Timestamp("2026-08-27 06:00:00", tz="UTC")
    plan.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "value_time_utc": index,
            "snapshot_time_utc": cutoff,
            "revision_time_utc": cutoff,
            "value": 19.0,
            "downloaded_at_utc": pd.Timestamp("2026-08-28", tz="UTC"),
        }
    ).to_parquet(plan.output, index=False)
    digest = hashlib.sha256(plan.output.read_bytes()).hexdigest()
    plan.audit_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "series": plan.series,
                "alias": plan.alias,
                "timezone": plan.timezone,
                "cutoff_timezone": plan.timezone,
                "cutoff_time": "08:00",
                "naive_timezone": plan.naive_timezone,
                "incomplete_dst_policy": "duplicate",
                "daily_broadcast": True,
                "value_scale": 1.0,
                "start_day": "2026-08-28",
                "end_day": "2026-08-28",
                "days": 1,
                "rows": len(index),
                "first_delivery_utc": index[0].isoformat(),
                "last_delivery_utc": index[-1].isoformat(),
                "sha256": digest,
            }
        ),
        encoding="utf-8",
    )

    reusable, reason = reusable_output(plan, start_day=start, end_day=end)
    assert reusable is True, reason

    catalog = build_catalog(
        [plan],
        start_day=start,
        end_day=end,
        series_workers=1,
        day_workers=2,
    )
    json_path, yaml_path = write_catalog(catalog, tmp_path)
    assert json_path.name == CATALOG_JSON
    assert yaml_path.name == CATALOG_YAML
    assert json.loads(json_path.read_text(encoding="utf-8")) == yaml.safe_load(
        yaml_path.read_text(encoding="utf-8")
    )
    lab = catalog["additional_sources_snippets"]["auxiliary_lab"]
    live = catalog["additional_sources_snippets"]["operational_live"]
    assert lab["additional_sources"][0]["columns"] == {
        "fr_temperature_fcst": "value"
    }
    assert live["additional_sources"]["fr_temperature_fcst"]["sha256"] == digest
    assert live["additional_sources"]["fr_temperature_fcst"][
        "information_type"
    ] == "day_ahead_forecast"
    assert catalog["start_day"] == "2026-08-28"
    assert catalog["end_day"] == "2026-08-28"
    assert catalog["requested_start_day"] == "2026-08-28"
    assert catalog["requested_end_day"] == "2026-08-28"
    assert catalog["schema_version"] == CATALOG_SCHEMA_VERSION
    assert "contract_version" not in catalog
    assert catalog["series"][0]["audit_sha256"] == hashlib.sha256(
        plan.audit_path.read_bytes()
    ).hexdigest()
    assert catalog["series"][0]["incomplete_dst_policy"] == "duplicate"

    # A semantically different request must rebuild even if files still exist.
    changed_plan = type(plan)(**{**plan.__dict__, "value_scale": 0.5})
    reusable, reason = reusable_output(
        changed_plan,
        start_day=start,
        end_day=end,
    )
    assert reusable is False
    assert "value_scale" in reason


def test_semantic_timezone_change_cannot_overwrite_legacy_cache(
    tmp_path: Path,
) -> None:
    corrected = {
        value.alias: value for value in build_plan(["NL"], tmp_path)
    }["nl_wind_generation_fcst"]
    corrected.output.parent.mkdir(parents=True, exist_ok=True)
    corrected.output.write_bytes(b"immutable legacy parquet")
    corrected.audit_path.write_text(
        json.dumps(
            {
                "series": corrected.series,
                "alias": corrected.alias,
                "naive_timezone": "Europe/Amsterdam",
            }
        ),
        encoding="utf-8",
    )

    try:
        reject_in_place_semantic_rewrite([corrected])
    except RuntimeError as exc:
        assert "Reecriture semantique in-place refusee" in str(exc)
        assert "Europe/Amsterdam" in str(exc)
        assert "UTC" in str(exc)
    else:  # pragma: no cover - explicit safety assertion.
        raise AssertionError("La reecriture semantique aurait du etre refusee.")
    assert corrected.output.read_bytes() == b"immutable legacy parquet"
