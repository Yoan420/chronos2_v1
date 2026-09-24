from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from chronos2_hourly.kalman_configuration import (
    KalmanConfigurationError,
    attach_additional_kalman_sources,
    attach_kalman_upstream_history,
    load_kalman_operational_configuration,
    render_kalman_weather_operational_configuration,
)
from chronos2_hourly.kalman_covariates import BASE_RESIDUAL_LOAD_COVARIATES


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_config(
    tmp_path: Path,
    source: Path,
    *,
    source_timestamps: str = "delivery_start_utc",
    source_column: str = "temperature_forecast",
    checksum: str | None = None,
    training_lookback_days: int | None = None,
    information_type: str = "day_ahead_forecast",
) -> Path:
    raw = pd.read_csv(source)
    if source_timestamps in raw:
        delivery = pd.to_datetime(raw[source_timestamps], utc=True, errors="coerce")
        if delivery.notna().all():
            local_days = delivery.dt.tz_convert("Europe/Paris").dt.normalize().dt.tz_localize(None)
            cutoffs = pd.Series(
                [
                    (day - pd.Timedelta(days=1) + pd.Timedelta(hours=8))
                    .tz_localize("Europe/Paris")
                    .tz_convert("UTC")
                    .isoformat()
                    for day in local_days
                ]
            )
            raw["forecast_cutoff_utc"] = cutoffs
            raw["forecast_origin_utc"] = (
                pd.to_datetime(cutoffs, utc=True)
                .sub(pd.Timedelta(hours=6))
                .map(pd.Timestamp.isoformat)
            )
            raw.to_csv(source, index=False)
    payload = {
        "version": 1,
        "filter_parameters": {
            "candidate_kinds": ["linear_bias", "linear_weather"],
        },
        "covariates": {
            "input_columns": ["fr_residual_load_fcst", "fr_temperature_fcst"],
            "derived": {},
            "groups": {
                "market": ["fr_residual_load_fcst"],
                "weather": ["fr_temperature_fcst"],
            },
            "require_future_complete": True,
        },
        "additional_sources": {
            "weather_pit": {
                "path": str(source),
                "timestamp_column": source_timestamps,
                "columns": {"fr_temperature_fcst": source_column},
                "information_type": information_type,
                "cutoff_policy": "latest vintage available before 08:00 Europe/Paris",
                "origin_column": "forecast_origin_utc",
                "cutoff_column": "forecast_cutoff_utc",
                "cutoff_time": "08:00",
                "sha256": checksum or _digest(source),
            }
        },
        "provenance": {"owner": "test", "pit": True},
    }
    if training_lookback_days is not None:
        payload["training_lookback_days"] = training_lookback_days
    path = tmp_path / "kalman.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_residual_market_config(
    tmp_path: Path,
    source: Path,
    *,
    aliases: tuple[str, ...] = BASE_RESIDUAL_LOAD_COVARIATES,
) -> Path:
    payload = {
        "version": 1,
        "filter_parameters": {"candidate_kinds": ["linear_market"]},
        "covariates": {
            "input_columns": list(BASE_RESIDUAL_LOAD_COVARIATES),
            "derived": {},
            "groups": {"market": list(BASE_RESIDUAL_LOAD_COVARIATES)},
            "require_future_complete": True,
        },
        "additional_sources": {
            "residual_market": {
                "path": str(source),
                "timestamp_column": "value_time_utc",
                "columns": {alias: alias for alias in aliases},
                "information_type": "day_ahead_forecast",
                "cutoff_policy": "as-of D-1 08:00",
                "origin_column": "snapshot_time_utc",
                "revision_column": "revision_time_utc",
                "cutoff_column": "cutoff_time_utc",
                "cutoff_time": "08:00",
                "sha256": _digest(source),
            }
        },
    }
    config_path = tmp_path / "residual.yaml"
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    return config_path


@pytest.mark.parametrize("zone", ("FR", "DE", "BE", "NL", "ES"))
def test_default_operational_config_keeps_filter_and_adds_pinned_country_history(zone) -> None:
    config = load_kalman_operational_configuration(
        PROJECT_ROOT / "config" / "kalman_operational.yaml",
        project_root=PROJECT_ROOT,
        zone=zone,
    )

    assert config.filter_config.candidate_kinds == (
        "linear_bias",
        "linear_harmonic",
        "linear_market",
        "linear_scale",
        "ukf_scale",
    )
    assert config.covariate_config.input_columns == (
        "fr_residual_load_fcst",
        "de_residual_load_fcst",
        "be_residual_load_fcst",
        "nl_residual_load_fcst",
        "es_residual_load_fcst",
    )
    assert config.additional_sources == ()
    assert config.training_lookback_days == 365
    assert config.rolling_refit_workers == 4
    assert config.upstream_history is not None
    assert config.upstream_history.path == (
        PROJECT_ROOT / "data" / "pit" / "kalman_weather"
        / f"{zone.lower()}_residual_corrected_prequential.csv.gz"
    )
    assert config.upstream_history.sha256 == _digest(config.upstream_history.path)
    assert config.upstream_history.audit_sha256 == _digest(config.upstream_history.audit_path)
    assert config.contract_dict()["training_lookback_days"] == 365
    assert config.contract_dict()["rolling_refit_workers"] == 4
    assert config.provenance["upstream_history_selection"]["zone"] == zone
    assert len(config.sha256) == 64


def test_training_lookback_days_is_exposed_in_operational_contract(
    tmp_path: Path,
) -> None:
    timestamps = pd.date_range("2026-08-27T22:00:00Z", periods=2, freq="h")
    source = tmp_path / "weather.csv"
    pd.DataFrame(
        {
            "delivery_start_utc": timestamps.astype(str),
            "temperature_forecast": [17.0, 16.0],
        }
    ).to_csv(source, index=False)

    config = load_kalman_operational_configuration(
        _write_config(tmp_path, source, training_lookback_days=365),
        project_root=tmp_path,
    )

    assert config.training_lookback_days == 365
    assert config.contract_dict()["training_lookback_days"] == 365


def _write_upstream_country_fixture(tmp_path: Path, zone: str, level: float) -> dict:
    prefix = tmp_path / f"{zone.lower()}_upstream.csv.gz"
    pd.DataFrame({
        "delivery_start_utc": pd.date_range("2026-08-27T22:00:00Z", periods=2, freq="h"),
        "actual": [level, level + 1],
        "residual_corrected__q10": [level - 5, level - 4],
        "residual_corrected__q50": [level, level + 1],
        "residual_corrected__q90": [level + 5, level + 6],
        "chronos2__q50": [level - 1, level],
        "residual_correction": [1.0, 1.0],
    }).to_csv(prefix, index=False, compression="gzip")
    protocol = "blocked_prequential_residual_then_issued_evaluation_overlay"
    audit = tmp_path / f"{zone.lower()}_upstream.audit.json"
    audit.write_text(json.dumps({
        "protocol": protocol,
        "selected_recipe_is_frozen": True,
        "fit_label_rule": "local_delivery_day < block_start_day",
        "causality_violations": 0,
        "published_origin_causality_violations": 0,
        "published_origin_contract_mismatches": 0,
    }), encoding="utf-8")
    return {
        "path": str(prefix), "sha256": _digest(prefix),
        "audit_path": str(audit), "audit_sha256": _digest(audit),
        "upstream_model": "residual_corrected", "protocol": protocol,
    }


@pytest.fixture
def country_upstream_config(tmp_path: Path):
    payload = {
        "version": 1,
        "training_lookback_days": 365,
        "rolling_refit_workers": 4,
        "filter_parameters": {"candidate_kinds": ["linear_bias"]},
        "covariates": {
            "input_columns": list(BASE_RESIDUAL_LOAD_COVARIATES),
            "derived": {},
            "groups": {"market": list(BASE_RESIDUAL_LOAD_COVARIATES)},
        },
        "upstream_history_by_zone": {
            "FR": _write_upstream_country_fixture(tmp_path, "FR", 40.0),
            "NL": _write_upstream_country_fixture(tmp_path, "NL", 90.0),
        },
        "additional_sources": {},
        "provenance": {"purpose": "test_country_prefix"},
    }
    path = tmp_path / "country_kalman.yaml"

    def persist():
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    persist()
    return path, payload, persist


@pytest.mark.parametrize(("zone", "level"), (("FR", 40.0), ("NL", 90.0)))
def test_upstream_history_country_selection_and_attachment_are_isolated(
    tmp_path: Path, country_upstream_config, zone, level,
):
    path, payload, persist = country_upstream_config
    other_zone = "NL" if zone == "FR" else "FR"
    payload["upstream_history_by_zone"][other_zone]["path"] = str(tmp_path / "not_available.csv.gz")
    persist()
    config = load_kalman_operational_configuration(path, project_root=tmp_path, zone=zone.lower())
    assert config.upstream_history.path.name == f"{zone.lower()}_upstream.csv.gz"
    assert config.provenance["upstream_history_selection"] == {
        "source": "upstream_history_by_zone", "status": "selected", "zone": zone,
        "upstream_model": "residual_corrected", "incumbent_history_attached": True,
    }
    prefix = pd.read_csv(config.upstream_history.path)
    issued = prefix.iloc[-1:].copy()
    combined, audit = attach_kalman_upstream_history(issued, config, timezone="Europe/Paris")
    assert combined["residual_corrected__q50"].tolist() == [level, level + 1]
    assert audit["overlap_prediction_conflicts"] == 0
    assert config.training_lookback_days == 365


@pytest.mark.parametrize(("zone", "message"), (
    (None, "zone est obligatoire"), ("", "zone doit etre"),
    ("GB", "zone doit etre"), ("BE", "aucun prefixe pour BE"),
))
def test_country_prefix_requires_supported_configured_zone(
    tmp_path: Path, country_upstream_config, zone, message,
):
    path, _, _ = country_upstream_config
    with pytest.raises(KalmanConfigurationError, match=message):
        load_kalman_operational_configuration(path, project_root=tmp_path, zone=zone)


@pytest.mark.parametrize("hash_field", ("sha256", "audit_sha256"))
def test_country_prefix_verifies_both_preexisting_hashes(
    tmp_path: Path, country_upstream_config, hash_field,
):
    path, payload, persist = country_upstream_config
    payload["upstream_history_by_zone"]["FR"][hash_field] = "0" * 64
    persist()
    with pytest.raises(KalmanConfigurationError, match="Checksum invalide pour upstream_history"):
        load_kalman_operational_configuration(path, project_root=tmp_path, zone="FR")


@pytest.mark.parametrize("single_is_null", (False, True))
def test_single_and_country_prefix_contracts_are_mutually_exclusive(
    tmp_path: Path, country_upstream_config, single_is_null,
):
    path, payload, persist = country_upstream_config
    payload["upstream_history"] = None if single_is_null else payload["upstream_history_by_zone"]["FR"]
    persist()
    with pytest.raises(KalmanConfigurationError, match="mutuellement exclusifs"):
        load_kalman_operational_configuration(path, project_root=tmp_path, zone="FR")


@pytest.mark.parametrize("upstream_model", ("exogenous_residual_corrected", "chronos2_lora_raw"))
@pytest.mark.parametrize("zone", ("FR", None))
def test_lora_excludes_incumbent_prefix_without_reading_it(
    tmp_path: Path, country_upstream_config, upstream_model, zone,
):
    path, payload, persist = country_upstream_config
    for source in payload["upstream_history_by_zone"].values():
        source["path"] = str(tmp_path / "unavailable_incumbent.csv.gz")
        source["audit_path"] = str(tmp_path / "unavailable_incumbent.audit.json")
    # A user-supplied provenance claim cannot conceal the computed exclusion.
    payload["provenance"]["upstream_history_selection"] = {"incumbent_history_attached": True}
    persist()
    config = load_kalman_operational_configuration(
        path, project_root=tmp_path, zone=zone, upstream_model=upstream_model,
    )
    assert config.upstream_history is None
    assert config.provenance["upstream_history_selection"] == {
        "source": "upstream_history_by_zone", "status": "excluded_incompatible_upstream_model",
        "zone": zone, "upstream_model": upstream_model, "incumbent_history_attached": False,
    }
    issued = pd.DataFrame({"exogenous_residual_corrected__q50": [140.0]})
    unchanged, audit = attach_kalman_upstream_history(issued, config, timezone="Europe/Paris")
    pd.testing.assert_frame_equal(unchanged, issued)
    assert audit == {"status": "not_configured"}


def test_single_prefix_preserves_api_but_refuses_incompatible_model(
    tmp_path: Path, country_upstream_config,
):
    path, payload, persist = country_upstream_config
    payload["upstream_history"] = payload.pop("upstream_history_by_zone")["FR"]
    persist()
    config = load_kalman_operational_configuration(path, project_root=tmp_path)
    assert config.upstream_history.upstream_model == "residual_corrected"
    with pytest.raises(KalmanConfigurationError, match="incompatible.*upstream_model"):
        load_kalman_operational_configuration(
            path, project_root=tmp_path, upstream_model="exogenous_residual_corrected",
        )


@pytest.mark.parametrize("zone_map", (None, {}, {"GB": {}}, {"FR": None}, {"fr": {}, "FR": {}}))
def test_country_prefix_rejects_malformed_or_ambiguous_mapping(
    tmp_path: Path, country_upstream_config, zone_map,
):
    path, payload, persist = country_upstream_config
    payload["upstream_history_by_zone"] = zone_map
    persist()
    with pytest.raises(KalmanConfigurationError):
        load_kalman_operational_configuration(path, project_root=tmp_path, zone="FR")


@pytest.mark.parametrize("invalid", [0, -1, True, 365.0, "365"])
def test_training_lookback_days_rejects_invalid_values(
    tmp_path: Path,
    invalid: object,
) -> None:
    path = tmp_path / "invalid-lookback.yaml"
    payload = {
        "version": 1,
        "training_lookback_days": invalid,
        "filter_parameters": {},
        "covariates": {},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(KalmanConfigurationError, match="training_lookback_days"):
        load_kalman_operational_configuration(path, project_root=tmp_path)


@pytest.mark.parametrize("invalid", [0, 9, -1, True, 4.0, "4"])
def test_rolling_refit_workers_rejects_invalid_values(
    tmp_path: Path,
    invalid: object,
) -> None:
    path = tmp_path / "invalid-workers.yaml"
    payload = {
        "version": 1,
        "rolling_refit_workers": invalid,
        "filter_parameters": {},
        "covariates": {},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(KalmanConfigurationError, match="rolling_refit_workers"):
        load_kalman_operational_configuration(path, project_root=tmp_path)


def test_additional_forecast_source_is_joined_by_aware_utc_timestamp(
    tmp_path: Path,
) -> None:
    timestamps = pd.date_range("2026-08-27T22:00:00Z", periods=4, freq="h")
    source = tmp_path / "weather.csv"
    pd.DataFrame(
        {
            "delivery_start_utc": timestamps.astype(str),
            "temperature_forecast": [17.0, 16.0, 15.0, 14.0],
        }
    ).to_csv(source, index=False)
    config = load_kalman_operational_configuration(
        _write_config(tmp_path, source), project_root=tmp_path
    )
    base = pd.DataFrame(
        {
            "timestamp": timestamps.astype(str),
            "fr_residual_load_fcst": [41.0, 42.0, 43.0, 44.0],
        }
    )

    joined, audit = attach_additional_kalman_sources(
        base,
        config,
        required_future_index=timestamps[-2:],
    )

    assert joined["fr_temperature_fcst"].tolist() == [17.0, 16.0, 15.0, 14.0]
    assert audit["config_sha256"] == config.sha256
    assert audit["sources"][0]["sha256"] == _digest(source)
    assert audit["sources"][0]["cutoff_policy"].startswith("latest vintage")


def test_last_known_market_price_source_is_explicitly_supported(
    tmp_path: Path,
) -> None:
    timestamps = pd.date_range("2026-08-27T22:00:00Z", periods=2, freq="h")
    source = tmp_path / "fuel.csv"
    pd.DataFrame(
        {
            "delivery_start_utc": timestamps.astype(str),
            "temperature_forecast": [29.0, 29.0],
        }
    ).to_csv(source, index=False)

    config = load_kalman_operational_configuration(
        _write_config(
            tmp_path,
            source,
            information_type="last_known_market_prices",
        ),
        project_root=tmp_path,
    )

    assert config.additional_sources[0].information_type == (
        "last_known_market_prices"
    )


def test_complete_pit_residual_load_source_is_authoritative_and_audited(
    tmp_path: Path,
) -> None:
    timestamps = pd.date_range("2026-08-27T22:00:00Z", periods=2, freq="h")
    cutoff = pd.Timestamp("2026-08-27T06:00:00Z")
    source = tmp_path / "residual.parquet"
    authoritative = {
        "fr_residual_load_fcst": [41.0, 42.0],
        "de_residual_load_fcst": [51.0, 52.0],
        "be_residual_load_fcst": [20.0, 21.0],
        "nl_residual_load_fcst": [30.1234567, 31.0],
        "es_residual_load_fcst": [12.0, 13.0],
    }
    pd.DataFrame(
        {
            "value_time_utc": timestamps,
            "snapshot_time_utc": cutoff,
            "revision_time_utc": cutoff,
            "cutoff_time_utc": cutoff,
            **authoritative,
        }
    ).to_parquet(source, index=False)
    config = load_kalman_operational_configuration(
        _write_residual_market_config(tmp_path, source), project_root=tmp_path
    )
    base = pd.DataFrame(
        {
            "timestamp": timestamps,
            # Missing native cells, exact matches, tiny rounding drift and very
            # large divergences all exercise the same authoritative policy.
            "fr_residual_load_fcst": [float("nan"), 420.0],
            "de_residual_load_fcst": [510.0, float("nan")],
            "be_residual_load_fcst": [20.0, 21.0],
            "nl_residual_load_fcst": [30.1234568, 31.0000000005],
            "es_residual_load_fcst": [112.0, -87.0],
        }
    )

    joined, audit = attach_additional_kalman_sources(
        base,
        config,
        required_future_index=timestamps,
    )
    for alias, expected in authoritative.items():
        assert joined[alias].tolist() == expected

    source_audit = audit["sources"][0]
    assert isinstance(source_audit["replacement_policy"], str)
    assert "authorit" in source_audit["replacement_policy"].casefold()
    assert source_audit["comparison_absolute_tolerance"] == pytest.approx(1e-9)
    assert source_audit["overlap_count"] == {
        "fr_residual_load_fcst": 1,
        "de_residual_load_fcst": 1,
        "be_residual_load_fcst": 2,
        "nl_residual_load_fcst": 2,
        "es_residual_load_fcst": 2,
    }
    assert source_audit["mismatch_count"] == {
        "fr_residual_load_fcst": 1,
        "de_residual_load_fcst": 1,
        "be_residual_load_fcst": 0,
        "nl_residual_load_fcst": 1,
        "es_residual_load_fcst": 2,
    }
    assert source_audit["replacement_count"] == {
        alias: 2 for alias in BASE_RESIDUAL_LOAD_COVARIATES
    }
    assert source_audit["max_abs_difference"] == pytest.approx(
        {
            "fr_residual_load_fcst": 378.0,
            "de_residual_load_fcst": 459.0,
            "be_residual_load_fcst": 0.0,
            "nl_residual_load_fcst": 1e-7,
            "es_residual_load_fcst": 100.0,
        }
    )


def test_partial_residual_load_source_cannot_collide_with_native_archive(
    tmp_path: Path,
) -> None:
    timestamps = pd.date_range("2026-08-27T22:00:00Z", periods=2, freq="h")
    cutoff = pd.Timestamp("2026-08-27T06:00:00Z")
    source = tmp_path / "partial-residual.parquet"
    pd.DataFrame(
        {
            "value_time_utc": timestamps,
            "snapshot_time_utc": cutoff,
            "revision_time_utc": cutoff,
            "cutoff_time_utc": cutoff,
            "fr_residual_load_fcst": [41.0, 42.0],
        }
    ).to_parquet(source, index=False)
    config = load_kalman_operational_configuration(
        _write_residual_market_config(
            tmp_path,
            source,
            aliases=("fr_residual_load_fcst",),
        ),
        project_root=tmp_path,
    )
    base = pd.DataFrame(
        {
            "timestamp": timestamps,
            **{
                alias: [1.0, 2.0]
                for alias in BASE_RESIDUAL_LOAD_COVARIATES
            },
        }
    )

    with pytest.raises(KalmanConfigurationError, match="collision"):
        attach_additional_kalman_sources(base, config)


def test_arbitrary_additional_source_collision_remains_forbidden(
    tmp_path: Path,
) -> None:
    timestamps = pd.date_range("2026-08-27T22:00:00Z", periods=2, freq="h")
    source = tmp_path / "weather.csv"
    pd.DataFrame(
        {
            "delivery_start_utc": timestamps.astype(str),
            "temperature_forecast": [17.0, 16.0],
        }
    ).to_csv(source, index=False)
    config = load_kalman_operational_configuration(
        _write_config(tmp_path, source), project_root=tmp_path
    )
    base = pd.DataFrame(
        {
            "timestamp": timestamps,
            "fr_residual_load_fcst": [41.0, 42.0],
            "fr_temperature_fcst": [99.0, 99.0],
        }
    )

    with pytest.raises(KalmanConfigurationError, match="collision"):
        attach_additional_kalman_sources(base, config)


def test_authoritative_residual_source_future_hole_is_not_masked_by_native_data(
    tmp_path: Path,
) -> None:
    timestamps = pd.date_range("2026-08-27T22:00:00Z", periods=3, freq="h")
    cutoff = pd.Timestamp("2026-08-27T06:00:00Z")
    source = tmp_path / "incomplete-residual.parquet"
    source_values = {
        alias: [float(position), float(position + 1)]
        for position, alias in enumerate(BASE_RESIDUAL_LOAD_COVARIATES)
    }
    pd.DataFrame(
        {
            "value_time_utc": timestamps[:2],
            "snapshot_time_utc": cutoff,
            "revision_time_utc": cutoff,
            "cutoff_time_utc": cutoff,
            **source_values,
        }
    ).to_parquet(source, index=False)
    config = load_kalman_operational_configuration(
        _write_residual_market_config(tmp_path, source), project_root=tmp_path
    )
    base = pd.DataFrame(
        {
            "timestamp": timestamps,
            **{
                alias: [100.0, 101.0, 102.0]
                for alias in BASE_RESIDUAL_LOAD_COVARIATES
            },
        }
    )

    with pytest.raises(KalmanConfigurationError, match="futures incompletes"):
        attach_additional_kalman_sources(
            base,
            config,
            required_future_index=timestamps,
        )


def test_additional_source_rejects_naive_dst_timestamps(tmp_path: Path) -> None:
    source = tmp_path / "weather.csv"
    pd.DataFrame(
        {
            "delivery_start_utc": ["2026-10-25 02:00:00", "2026-10-25 03:00:00"],
            "temperature_forecast": [12.0, 13.0],
        }
    ).to_csv(source, index=False)
    config = load_kalman_operational_configuration(
        _write_config(tmp_path, source), project_root=tmp_path
    )
    base = pd.DataFrame(
        {
            "timestamp": ["2026-10-25T00:00:00+00:00", "2026-10-25T01:00:00+00:00"],
            "fr_residual_load_fcst": [40.0, 41.0],
        }
    )

    with pytest.raises(KalmanConfigurationError, match="sans offset"):
        attach_additional_kalman_sources(base, config)


def test_additional_source_fails_closed_on_future_hole(tmp_path: Path) -> None:
    timestamps = pd.date_range("2026-08-27T22:00:00Z", periods=3, freq="h")
    source = tmp_path / "weather.csv"
    pd.DataFrame(
        {
            "delivery_start_utc": timestamps[:2].astype(str),
            "temperature_forecast": [17.0, 16.0],
        }
    ).to_csv(source, index=False)
    config = load_kalman_operational_configuration(
        _write_config(tmp_path, source), project_root=tmp_path
    )
    base = pd.DataFrame(
        {
            "timestamp": timestamps.astype(str),
            "fr_residual_load_fcst": [41.0, 42.0, 43.0],
        }
    )

    with pytest.raises(KalmanConfigurationError, match="futures incompletes"):
        attach_additional_kalman_sources(
            base, config, required_future_index=timestamps
        )


@pytest.mark.parametrize(
    "unsafe_column",
    [
        "actual_temperature",
        "observed_wind",
        "storm_temperature_forecast",
        "wind_oracle",
    ],
)
def test_additional_source_rejects_noncausal_column_names(
    tmp_path: Path,
    unsafe_column: str,
) -> None:
    source = tmp_path / "weather.csv"
    pd.DataFrame(
        {
            "delivery_start_utc": ["2026-08-28T00:00:00+00:00"],
            unsafe_column: [17.0],
        }
    ).to_csv(source, index=False)

    with pytest.raises(KalmanConfigurationError, match="non causales"):
        load_kalman_operational_configuration(
            _write_config(
                tmp_path,
                source,
                source_column=unsafe_column,
            ),
            project_root=tmp_path,
        )


def test_configuration_rejects_unknown_root_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("version: 1\nmagic: true\n", encoding="utf-8")

    with pytest.raises(KalmanConfigurationError, match="inconnus"):
        load_kalman_operational_configuration(path, project_root=tmp_path)


def test_configuration_never_silently_falls_back_when_contract_is_missing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "incomplete.yaml"
    path.write_text("version: 1\nfilter_parameters: {}\n", encoding="utf-8")

    with pytest.raises(KalmanConfigurationError, match="obligatoires absents"):
        load_kalman_operational_configuration(path, project_root=tmp_path)


def test_configuration_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        "version: 1\nversion: 1\nfilter_parameters: {}\ncovariates: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(KalmanConfigurationError, match="dupliquee"):
        load_kalman_operational_configuration(path, project_root=tmp_path)


def test_weather_template_renders_runtime_hashes_and_attaches_causal_prefix(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    source_manifest = data / "source_bundle_manifest.json"
    source_manifest.write_text("{}", encoding="utf-8")
    timestamps = pd.date_range("2026-08-27T22:00:00Z", periods=2, freq="h")
    cutoff = pd.Timestamp("2026-08-27T06:00:00Z")
    weather = data / "fr_temperature_fcst.parquet"
    pd.DataFrame(
        {
            "value_time_utc": timestamps,
            "snapshot_time_utc": cutoff,
            "revision_time_utc": cutoff,
            "value": [17.0, 16.0],
        }
    ).to_parquet(weather, index=False)
    prefix = data / "fr_residual_corrected_prequential.csv.gz"
    pd.DataFrame(
        {
            "delivery_start_utc": timestamps,
            "actual": [40.0, 41.0],
            "residual_corrected__q10": [35.0, 36.0],
            "residual_corrected__q50": [40.0, 41.0],
            "residual_corrected__q90": [45.0, 46.0],
            "chronos2__q50": [39.0, 40.0],
            "residual_correction": [1.0, 1.0],
        }
    ).to_csv(prefix, index=False, compression="gzip")
    prefix_audit = data / "fr_residual_corrected_prequential.audit.json"
    prefix_audit.write_text(
        json.dumps(
            {
                "protocol": (
                    "blocked_prequential_residual_then_issued_evaluation_overlay"
                ),
                "selected_recipe_is_frozen": True,
                "fit_label_rule": "local_delivery_day < block_start_day",
                "causality_violations": 0,
                "published_origin_causality_violations": 0,
                "published_origin_contract_mismatches": 0,
            }
        ),
        encoding="utf-8",
    )
    template = tmp_path / "weather-template.yaml"
    template.write_text(
        """
version: 1
training_lookback_days: 365
rolling_refit_workers: 4
filter_parameters:
  candidate_kinds: [linear_weather]
covariates:
  input_columns:
    - fr_residual_load_fcst
    - de_residual_load_fcst
    - be_residual_load_fcst
    - nl_residual_load_fcst
    - es_residual_load_fcst
    - "${zone_lower}_temperature_fcst"
  derived: {}
  groups:
    market:
      - fr_residual_load_fcst
      - de_residual_load_fcst
      - be_residual_load_fcst
      - nl_residual_load_fcst
      - es_residual_load_fcst
    weather: ["${zone_lower}_temperature_fcst"]
upstream_history:
  path: "${runtime_source_root}/${zone_lower}_residual_corrected_prequential.csv.gz"
  audit_path: "${runtime_source_root}/${zone_lower}_residual_corrected_prequential.audit.json"
  upstream_model: residual_corrected
  protocol: blocked_prequential_residual_then_issued_evaluation_overlay
additional_sources:
  "${zone_lower}_temperature_fcst":
    path: "${runtime_source_root}/${zone_lower}_temperature_fcst.parquet"
    timestamp_column: value_time_utc
    columns: {"${zone_lower}_temperature_fcst": value}
    information_type: day_ahead_forecast
    cutoff_policy: Saturn as-of D-1 08:00 civil
    origin_column: snapshot_time_utc
    revision_column: revision_time_utc
    cutoff_column: snapshot_time_utc
    cutoff_time: "08:00"
provenance: {purpose: test}
""".lstrip(),
        encoding="utf-8",
    )

    config = render_kalman_weather_operational_configuration(
        template,
        zone="FR",
        delivery_day="2026-08-28",
        output_path=tmp_path / "runtime" / "fr.yaml",
        project_root=tmp_path,
        runtime_source_root=data,
    )

    assert config.training_lookback_days == 365
    assert config.rolling_refit_workers == 4
    assert config.upstream_history is not None
    assert config.upstream_history.path.parent == data.resolve()
    assert config.upstream_history.sha256 == _digest(prefix)
    assert config.additional_sources[0].sha256 == _digest(weather)
    runtime_provenance = config.provenance["runtime_sidecar"]
    assert runtime_provenance["runtime_source_root"] == str(data.resolve())
    assert runtime_provenance["runtime_source_manifest"] == str(
        source_manifest.resolve()
    )
    assert runtime_provenance["runtime_source_manifest_sha256"] == _digest(
        source_manifest
    )
    issued = pd.read_csv(prefix).iloc[-1:].copy()
    issued.loc[:, "actual"] = 43.0
    combined, audit = attach_kalman_upstream_history(
        issued,
        config,
        timezone="Europe/Paris",
    )
    assert len(combined) == 2
    assert audit["overlap_rows"] == 1
    assert audit["overlap_conflicts"] == 0
    assert audit["overlap_actual_updates"] == 1
    assert combined.iloc[-1]["actual"] == 43.0

    divergent = issued.copy()
    divergent.loc[:, "residual_corrected__q50"] = 99.0
    with pytest.raises(KalmanConfigurationError, match="predictions"):
        attach_kalman_upstream_history(
            divergent,
            config,
            timezone="Europe/Paris",
        )
