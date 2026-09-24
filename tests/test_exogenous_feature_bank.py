from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_exogenous.feature_bank import (
    CALENDAR_COLUMNS,
    ConsumerRoute,
    ExogenousBankError,
    FLOWBASED_COMPACT_COLUMNS,
    ParquetFeatureSource,
    build_exogenous_bank,
    compress_flowbased_features,
    cutoff_by_delivery_hour,
    default_project_sources,
    delivery_utc_index,
)


def test_default_sources_accept_explicit_weather_root(tmp_path: Path) -> None:
    versioned = tmp_path / "data" / "pit" / "alternate_weather_bank"

    sources = default_project_sources(
        tmp_path,
        zone="NL",
        pack="residual_weather",
        weather_root=Path("data/pit/alternate_weather_bank"),
    )

    weather = [source for source in sources if source.family == "weather"]
    assert len(weather) == 3
    assert all(source.path.parent == versioned.resolve() for source in weather)
    assert any(
        source.path.name == "nl_solar_generation_fcst.parquet"
        for source in weather
    )


def _simple_pit(path: Path, day: str = "2026-01-10") -> pd.DatetimeIndex:
    index = delivery_utc_index(day, day)
    cutoff = cutoff_by_delivery_hour(index)
    frame = pd.DataFrame(
        {
            "value_time_utc": index,
            "snapshot_time_utc": cutoff,
            "revision_time_utc": cutoff - pd.Timedelta(hours=2),
            "value": np.arange(len(index), dtype=float),
        }
    )
    frame.to_parquet(path, index=False)
    return index


def _simple_source(path: Path, *, route: tuple[str, ...] = ("chronos",)) -> ParquetFeatureSource:
    return ParquetFeatureSource(
        name="wind_fr",
        family="weather",
        path=path,
        value_columns={"fr_wind_fcst": "value"},
        route=ConsumerRoute(route),
        cutoff_column="snapshot_time_utc",
        information_time_columns=("snapshot_time_utc", "revision_time_utc"),
        age_column="revision_time_utc",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _source_audit(
    path: Path,
    parquet: Path,
    *,
    evidence_kind: str | None = None,
    production_ready: bool | None = None,
    operational_capture_violations: int | None = None,
    cutoff_timezone: str = "Europe/Paris",
) -> Path:
    payload: dict[str, object] = {
        "output_sha256": _sha256(parquet),
        "causality_violations": 0,
        "cutoff_time": "08:00",
        "cutoff_timezone": cutoff_timezone,
    }
    if evidence_kind is not None:
        payload["production_evidence_kind"] = evidence_kind
    if production_ready is not None:
        payload["production_pit_evidence"] = production_ready
    if operational_capture_violations is not None:
        payload["operational_capture_violations"] = operational_capture_violations
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_delivery_index_preserves_23_and_25_hour_civil_days() -> None:
    spring = delivery_utc_index("2026-03-29", "2026-03-29")
    autumn = delivery_utc_index("2026-10-25", "2026-10-25")

    assert len(spring) == 23
    assert len(autumn) == 25
    assert spring.is_unique and autumn.is_unique
    assert spring.tz is not None and autumn.tz is not None


def test_bank_routes_values_and_quality_without_cross_consumer_leakage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wind.parquet"
    index = _simple_pit(path)
    bank = build_exogenous_bank(
        [_simple_source(path, route=("chronos", "kalman"))],
        start_day="2026-01-10",
        end_day="2026-01-10",
    )

    assert bank.frame.index.equals(index)
    assert bank.columns_for("residual") == ()
    assert bank.columns_for("chronos", include_quality=False) == (
        "fr_wind_fcst",
        *CALENDAR_COLUMNS,
    )
    assert bank.columns_for("kalman", include_quality=False) == ("fr_wind_fcst",)
    assert set(bank.columns_for("chronos")) == {
        "fr_wind_fcst",
        "wind_fr__coverage",
        "wind_fr__available",
        "wind_fr__age_hours",
        *CALENDAR_COLUMNS,
    }
    assert bank.frame["wind_fr__available"].eq(1.0).all()
    assert bank.frame["wind_fr__age_hours"].eq(2.0).all()
    assert bank.audit["complete"] is True


def test_unproven_source_is_never_production_ready(tmp_path: Path) -> None:
    path = tmp_path / "wind.parquet"
    _simple_pit(path)
    source = _simple_source(path)

    bank = build_exogenous_bank(
        [source], start_day="2026-01-10", end_day="2026-01-10"
    )
    assert bank.audit["production_ready"] is False
    assert bank.audit["historical_backtest_ready"] is False
    assert bank.audit["sources"]["wind_fr"]["production_pit_evidence"] is None
    with pytest.raises(ExogenousBankError, match="Preuve PIT operationnelle"):
        build_exogenous_bank(
            [source],
            start_day="2026-01-10",
            end_day="2026-01-10",
            require_operational_evidence=True,
        )


def test_versioned_revision_history_is_research_only_even_with_bound_audit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wind.parquet"
    _simple_pit(path)
    audit_path = _source_audit(tmp_path / "wind.audit.json", path)
    source = replace(
        _simple_source(path),
        audit_path=audit_path,
        production_evidence_kind="versioned_revision_history",
    )

    bank = build_exogenous_bank(
        [source],
        start_day="2026-01-10",
        end_day="2026-01-10",
    )
    source_audit = bank.audit["sources"]["wind_fr"]["source_audit"]
    assert bank.audit["production_ready"] is False
    assert bank.audit["historical_backtest_ready"] is False
    assert bank.audit["sources"]["wind_fr"]["production_pit_evidence"] is False
    assert source_audit["classification"] == "research_versioned_history"
    assert bank.audit["source_audit_hashes"]["wind_fr"] == _sha256(audit_path)
    with pytest.raises(ExogenousBankError, match="Preuve PIT operationnelle"):
        build_exogenous_bank(
            [source],
            start_day="2026-01-10",
            end_day="2026-01-10",
            require_operational_evidence=True,
        )


def test_prospective_capture_requires_row_flags_and_bound_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "wind.parquet"
    _simple_pit(path)
    frame = pd.read_parquet(path)
    frame["operational_capture_ok"] = True
    frame.to_parquet(path, index=False)
    audit_path = _source_audit(
        tmp_path / "wind.audit.json",
        path,
        evidence_kind="prospective_capture",
        production_ready=True,
        operational_capture_violations=0,
    )
    source = replace(
        _simple_source(path),
        audit_path=audit_path,
        operational_eligibility_column="operational_capture_ok",
        production_evidence_kind="prospective_capture",
    )

    bank = build_exogenous_bank(
        [source],
        start_day="2026-01-10",
        end_day="2026-01-10",
        require_operational_evidence=True,
    )

    assert bank.audit["production_ready"] is True
    assert bank.audit["historical_backtest_ready"] is True
    assert bank.audit["sources"]["wind_fr"]["production_pit_evidence"] is True
    assert (
        bank.audit["sources"]["wind_fr"]["source_audit"][
            "prospective_capture_verified"
        ]
        is True
    )


def test_source_audit_hash_must_match_parquet(tmp_path: Path) -> None:
    path = tmp_path / "wind.parquet"
    _simple_pit(path)
    audit_path = _source_audit(tmp_path / "wind.audit.json", path)
    frame = pd.read_parquet(path)
    frame.loc[0, "value"] += 1.0
    frame.to_parquet(path, index=False)
    source = replace(_simple_source(path), audit_path=audit_path)

    with pytest.raises(ExogenousBankError, match="non lie au Parquet"):
        build_exogenous_bank(
            [source], start_day="2026-01-10", end_day="2026-01-10"
        )


def test_source_cutoff_timezone_is_distinct_from_forecast_origin_timezone(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wind.parquet"
    index = delivery_utc_index("2026-01-10", "2026-01-10")
    london_cutoff = cutoff_by_delivery_hour(index, timezone="Europe/London")
    pd.DataFrame(
        {
            "value_time_utc": index,
            "snapshot_time_utc": london_cutoff,
            "revision_time_utc": london_cutoff - pd.Timedelta(hours=1),
            "value": np.arange(len(index), dtype=float),
        }
    ).to_parquet(path, index=False)
    audit_path = _source_audit(
        tmp_path / "wind.audit.json",
        path,
        cutoff_timezone="Europe/London",
    )
    source = replace(
        _simple_source(path),
        cutoff_timezone="Europe/London",
        audit_path=audit_path,
        production_evidence_kind="versioned_revision_history",
    )
    bank = build_exogenous_bank(
        [source],
        start_day="2026-01-10",
        end_day="2026-01-10",
        timezone="Europe/Paris",
    )
    source_result = bank.audit["sources"]["wind_fr"]
    assert source_result["forecast_origin_timezone"] == "Europe/Paris"
    assert source_result["source_cutoff_timezone"] == "Europe/London"

    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["cutoff_timezone"] = "Europe/Paris"
    audit_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExogenousBankError, match="fuseau source Europe/London"):
        build_exogenous_bank(
            [source],
            start_day="2026-01-10",
            end_day="2026-01-10",
            timezone="Europe/Paris",
        )


def test_information_after_cutoff_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "wind.parquet"
    _simple_pit(path)
    frame = pd.read_parquet(path)
    frame.loc[0, "revision_time_utc"] = frame.loc[0, "snapshot_time_utc"] + pd.Timedelta(
        minutes=1
    )
    frame.to_parquet(path, index=False)

    with pytest.raises(ExogenousBankError, match="posterieure au cutoff"):
        build_exogenous_bank(
            [_simple_source(path)],
            start_day="2026-01-10",
            end_day="2026-01-10",
        )


def test_cutoff_must_equal_civil_d_minus_one_0800(tmp_path: Path) -> None:
    path = tmp_path / "wind.parquet"
    _simple_pit(path)
    frame = pd.read_parquet(path)
    frame.loc[0, "snapshot_time_utc"] -= pd.Timedelta(hours=1)
    frame.to_parquet(path, index=False)

    with pytest.raises(ExogenousBankError, match="D-1 08:00"):
        build_exogenous_bank(
            [_simple_source(path)],
            start_day="2026-01-10",
            end_day="2026-01-10",
        )


def test_missing_hours_are_masked_audited_and_optionally_fail_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wind.parquet"
    _simple_pit(path)
    pd.read_parquet(path).iloc[:-1].to_parquet(path, index=False)

    bank = build_exogenous_bank(
        [_simple_source(path)],
        start_day="2026-01-10",
        end_day="2026-01-10",
    )
    assert bank.audit["complete"] is False
    assert bank.audit["sources"]["wind_fr"]["missing_hours"] == 1
    assert bank.frame["wind_fr__available"].iloc[-1] == 0.0
    with pytest.raises(ExogenousBankError, match="incompletes"):
        bank.assert_complete("chronos")
    with pytest.raises(ExogenousBankError, match="Banque incomplete"):
        build_exogenous_bank(
            [_simple_source(path)],
            start_day="2026-01-10",
            end_day="2026-01-10",
            require_complete=True,
        )


def _flowbased_values(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flowbased_cnec_mtu_availability": np.ones(rows),
            "flowbased_hour_imputed": np.zeros(rows),
            "flowbased_cnec_count": np.full(rows, 100.0),
            "flowbased_external_ram_p10_mw": np.full(rows, 900.0),
            "flowbased_ram_p10_mw": np.full(rows, 500.0),
            "flowbased_ram_median_mw": np.full(rows, 1500.0),
            "flowbased_ram_below_1000_share": np.full(rows, 0.25),
            "flowbased_ram_to_fmax_p05": np.full(rows, 0.1),
            "flowbased_fr_de_ptdf_spread_p90": np.full(rows, 0.3),
            "flowbased_fr_be_ptdf_spread_p90": np.full(rows, 0.2),
            "flowbased_fr_nl_ptdf_spread_p90": np.full(rows, 0.4),
            "flowbased_fr_neighbor_ram_stress_p95_per_gw": np.full(rows, 1.2),
            "flowbased_core_ram_stress_p95_per_gw": np.full(rows, 2.0),
            "flowbased_stress_hhi": np.full(rows, 0.15),
        }
    )


def test_flowbased_compression_is_stable_and_semantic() -> None:
    result = compress_flowbased_features(_flowbased_values(2))

    assert tuple(result.columns) == FLOWBASED_COMPACT_COLUMNS
    assert result["flowbased_external_ram_p10_gw"].eq(0.9).all()
    assert result["flowbased_ram_p10_gw"].eq(0.5).all()
    assert result["flowbased_ram_headroom_p10_to_median_gw"].eq(1.0).all()
    assert result["flowbased_fr_neighbor_ptdf_spread_p90"].eq(0.4).all()


def test_flowbased_research_vintage_is_valid_but_not_production_ready(
    tmp_path: Path,
) -> None:
    day = "2026-01-10"
    index = delivery_utc_index(day, day)
    cutoff = cutoff_by_delivery_hour(index)
    frame = _flowbased_values(len(index))
    frame.insert(0, "value_time_utc", index)
    frame["flowbased_cutoff_time_utc"] = cutoff
    frame["flowbased_source_last_modified_utc"] = cutoff - pd.Timedelta(hours=1)
    frame["flowbased_pit_eligible"] = True
    frame["flowbased_operational_pit_eligible"] = False
    frame["flowbased_publication_stage"] = "initial_computation"
    path = tmp_path / "flowbased.parquet"
    frame.to_parquet(path, index=False)
    source = ParquetFeatureSource(
        name="flowbased_core",
        family="flowbased",
        path=path,
        value_columns={},
        cutoff_column="flowbased_cutoff_time_utc",
        information_time_columns=("flowbased_source_last_modified_utc",),
        age_column="flowbased_source_last_modified_utc",
        eligibility_column="flowbased_pit_eligible",
        operational_eligibility_column="flowbased_operational_pit_eligible",
        stage_column="flowbased_publication_stage",
        allowed_stages=("initial_computation",),
        transform="flowbased_compact",
    )

    bank = build_exogenous_bank([source], start_day=day, end_day=day)
    assert bank.audit["complete"] is True
    assert bank.audit["production_ready"] is False
    assert bank.audit["sources"]["flowbased_core"]["operational_eligible_share"] == 0.0
    with pytest.raises(ExogenousBankError, match="Preuve PIT operationnelle"):
        build_exogenous_bank(
            [source],
            start_day=day,
            end_day=day,
            require_operational_evidence=True,
        )
