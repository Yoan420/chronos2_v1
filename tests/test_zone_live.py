from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from chronos2_hourly.zone_live import (
    MARKET_ZONES,
    ZoneBundleError,
    audit_zone_live_bundle,
    build_zone_runner_command,
    build_zone_training_config,
    canonical_zone,
    load_zone_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_canonical_zone_accepts_uk_alias_without_changing_timezone() -> None:
    assert canonical_zone("uk") == "GB"
    assert MARKET_ZONES["GB"].timezone == "Europe/London"
    with pytest.raises(ZoneBundleError, match="Zone inconnue"):
        canonical_zone("CH")


def _template() -> dict:
    return {
        "model": {"model_id": "amazon/chronos-2", "seed": 42},
        "data": {"pit_files": {}},
        "hourly": {
            "feature_engineering": {
                "timezone": "Europe/Paris",
                "target_lags": [24, 48, 168, 336],
            },
            "residual_correction": {
                "iterations": 700,
                "feature_builder": {
                    "timezone": "Europe/Paris",
                    "rich_calendar_primary_country": "FR",
                },
            },
        },
        "output": {"directory": "runs/fr"},
        "report": {"filename": "fr.html", "title": "FR"},
        "zones": {
            "FR": {
                "timezone": "Europe/Paris",
                "target": {"series": "fr-target"},
                "covariates": {
                    "fr_residual_load_fcst": {
                        "source": "pit_parquet",
                        "series": "fr-load",
                        "enabled": True,
                    },
                    "be_residual_load_fcst": {
                        "source": "pit_parquet",
                        "series": "be-load",
                        "enabled": True,
                    },
                },
            }
        },
    }


def test_build_zone_training_config_keeps_model_recipe_and_does_not_mutate_fr() -> None:
    template = _template()
    before = copy.deepcopy(template)
    result = build_zone_training_config(
        template,
        zone="BE",
        contract={
            "timezone": "Europe/Brussels",
            "bidding_zone": "BE",
            "target_series": "be-target",
            "target_status": "audited_dst_strict",
            "price_unit": "EUR/MWh",
            "required_covariates": [
                "fr_residual_load_fcst",
                "be_residual_load_fcst",
            ],
        },
    )

    assert template == before
    assert result["model"] == before["model"]
    assert result["hourly"]["feature_engineering"]["target_lags"] == [24, 48, 168, 336]
    assert set(result["zones"]) == {"BE"}
    assert result["zones"]["BE"]["timezone"] == "Europe/Brussels"
    assert result["zones"]["BE"]["target"]["series"] == "be-target"
    assert result["hourly"]["residual_correction"]["iterations"] == 700
    assert result["hourly"]["residual_correction"]["feature_builder"][
        "rich_calendar_primary_country"
    ] == "BE"


def test_zone_covariate_override_is_merged_with_fr_template() -> None:
    result = build_zone_training_config(
        _template(),
        zone="NL",
        contract={
            "delivery_timezone": "Europe/Amsterdam",
            "bidding_zone": "NL",
            "target_series": "nl-target",
            "target_status": "audited_dst_strict",
            "price_unit": "EUR/MWh",
            "required_covariates": ["fr_residual_load_fcst"],
            "covariates": {
                "fr_residual_load_fcst": {"naive_timezone": "UTC"}
            },
        },
    )
    covariate = result["zones"]["NL"]["covariates"]["fr_residual_load_fcst"]
    assert covariate["series"] == "fr-load"
    assert covariate["source"] == "pit_parquet"
    assert covariate["naive_timezone"] == "UTC"


def test_build_zone_training_config_refuses_incomplete_or_ambiguous_contract() -> None:
    with pytest.raises(ZoneBundleError, match="target_series"):
        build_zone_training_config(
            _template(),
            zone="DE",
            contract={
                "timezone": "Europe/Berlin",
                "bidding_zone": "DE_LU",
                "target_status": "audited_dst_strict",
                "price_unit": "EUR/MWh",
                "required_covariates": ["de_residual_load_fcst"],
            },
        )
    with pytest.raises(ZoneBundleError, match="zone de prix explicite"):
        build_zone_training_config(
            _template(),
            zone="IT",
            contract={
                "timezone": "Europe/Rome",
                "bidding_zone": "IT",
                "target_series": "it-target",
                "target_status": "audited_dst_strict",
                "price_unit": "EUR/MWh",
                "required_covariates": ["fr_residual_load_fcst"],
            },
        )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fake_ready_bundle(tmp_path: Path) -> dict:
    (tmp_path / "runner.py").write_text("pass\n", encoding="utf-8")
    pit = tmp_path / "pit"
    pit.mkdir()
    (pit / "load.parquet").write_bytes(b"pit")
    (pit / "storm.parquet").write_bytes(b"storm")
    cache = tmp_path / "cache" / "be"
    cache.mkdir(parents=True)
    (cache / "target__abc.csv.gz").write_bytes(b"cache")
    frozen = tmp_path / "frozen"
    benchmark = tmp_path / "benchmark"
    _write_json(
        frozen / "run_manifest.json",
        {"zone": "BE", "timezone": "Europe/Brussels"},
    )
    _write_json(
        benchmark / "run_manifest.json",
        {
            "zone": "BE",
            "timezone": "Europe/Brussels",
            "evaluation_only_comparators": [
                "power.price.be.euromwh.h.fcst.3mv.storm.da.cache"
            ],
        },
    )
    (frozen / "artifact_checksums.json").write_text("{}", encoding="utf-8")
    (benchmark / "artifact_checksums.json").write_text("{}", encoding="utf-8")
    _write_json(
        tmp_path / "recipe.json",
        {
            "zone": "BE",
            "timezone": "Europe/Brussels",
            "prediction_mode": "mkonline_blend",
            "mkonline_enabled": True,
            "external_expert": {"series": "be-primary"},
        },
    )
    _write_json(
        tmp_path / "dependency.json",
        {"terminal_series": "be-primary", "storm_token_found": False},
    )
    base = {
        "data": {
            "cache_dir": "cache",
            "pit_vintage_dir": "pit",
            "pit_files": {"be_load": "load.parquet"},
        },
        "zones": {
            "BE": {
                "timezone": "Europe/Brussels",
                "target": {"series": "be-target"},
                "covariates": {
                    "be_load": {
                        "enabled": True,
                        "source": "pit_parquet",
                    }
                },
            }
        },
    }
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    live = {
        "live": {
            "zone": "BE",
            "delivery_timezone": "Europe/Brussels",
            "forecast_origin_timezone": "Europe/Paris",
            "prediction_mode": "mkonline_blend",
            "mkonline_enabled": True,
            "base_config": "base.yaml",
            "frozen_autonomous_run": "frozen",
            "sealed_benchmark_run": "benchmark",
            "recipe_manifest": "recipe.json",
            "dependency_manifest": "dependency.json",
        }
    }
    (tmp_path / "live.yaml").write_text(yaml.safe_dump(live), encoding="utf-8")
    return {
        "schema_version": 1,
        "zones": {
            "BE": {
                "enabled": True,
                "production_ready": True,
                "delivery_timezone": "Europe/Brussels",
                "forecast_origin_timezone": "Europe/Paris",
                "forecast_origin_local_time": "08:00",
                "prediction_mode": "mkonline_blend",
                "mkonline_enabled": True,
                "bidding_zone": "BE",
                "runner": "runner.py",
                "live_config": "live.yaml",
                "target_series": "be-target",
                "target_status": "audited_dst_strict",
                "price_unit": "EUR/MWh",
                "primary_series": "be-primary",
                "primary_status": "audited_primary",
                "storm_series": (
                    "power.price.be.euromwh.h.fcst.3mv.storm.da.cache"
                ),
                "storm_status": "audited_day_ahead_cache",
                "storm_primary_series": (
                    "power.price.be.euromwh.h.fcst.3mv.storm.da.cache"
                ),
                "storm_naive_timezone": "Europe/Brussels",
                "storm_strict_08_series": (
                    "power.price.be.euromwh.h.fcst.3mv.storm.da.basecase"
                ),
                "storm_pit_path": "pit/storm.parquet",
                "required_covariates": ["be_load"],
            }
        },
    }


def test_audit_ready_bundle_and_build_command(tmp_path: Path) -> None:
    registry = _fake_ready_bundle(tmp_path)
    audit = audit_zone_live_bundle(registry, zone="BE", registry_dir=tmp_path)
    assert audit.ready, audit.blockers
    command = build_zone_runner_command(
        audit,
        delivery_day="2026-08-14",
        local_files_only=True,
    )
    assert command[1] == str((tmp_path / "runner.py").resolve())
    assert command[2:4] == ["--config", str((tmp_path / "live.yaml").resolve())]
    assert command[-3:] == ["--delivery-day", "2026-08-14", "--local-files-only"]


def test_build_command_keeps_default_saturn_argv_unchanged(tmp_path: Path) -> None:
    registry = _fake_ready_bundle(tmp_path)
    audit = audit_zone_live_bundle(registry, zone="BE", registry_dir=tmp_path)

    default_command = build_zone_runner_command(
        audit,
        delivery_day="2026-08-14",
        local_files_only=True,
    )
    explicit_saturn_command = build_zone_runner_command(
        audit,
        delivery_day="2026-08-14",
        local_files_only=True,
        residual_load_source="saturn",
        residual_load_bundle_manifest=tmp_path / "ignored.json",
    )

    assert explicit_saturn_command == default_command
    assert "--residual-load-source" not in default_command
    assert "--residual-load-bundle-manifest" not in default_command


def test_build_command_transmits_chronos2_bundle_manifest(tmp_path: Path) -> None:
    registry = _fake_ready_bundle(tmp_path)
    audit = audit_zone_live_bundle(registry, zone="BE", registry_dir=tmp_path)
    manifest = tmp_path / "chronos2_residual_load_bundle.json"

    command = build_zone_runner_command(
        audit,
        residual_load_source="Chronos2",
        residual_load_bundle_manifest=manifest,
    )

    assert command[-4:] == [
        "--residual-load-source",
        "chronos2",
        "--residual-load-bundle-manifest",
        str(manifest),
    ]


def test_build_command_requires_manifest_for_chronos2(tmp_path: Path) -> None:
    registry = _fake_ready_bundle(tmp_path)
    audit = audit_zone_live_bundle(registry, zone="BE", registry_dir=tmp_path)

    with pytest.raises(ZoneBundleError, match="manifest est obligatoire"):
        build_zone_runner_command(audit, residual_load_source="chronos2")


def test_audit_ready_autonomous_bundle_needs_no_primary_or_dependency(
    tmp_path: Path,
) -> None:
    registry = _fake_ready_bundle(tmp_path)
    zone = registry["zones"]["BE"]
    zone.update(
        {
            "prediction_mode": "autonomous_only",
            "mkonline_enabled": False,
            "primary_series": None,
            "primary_status": "not_used_autonomous_only",
        }
    )
    live_path = tmp_path / "live.yaml"
    live = yaml.safe_load(live_path.read_text(encoding="utf-8"))
    live["live"].update(
        {
            "prediction_mode": "autonomous_only",
            "mkonline_enabled": False,
            "dependency_manifest": None,
        }
    )
    live_path.write_text(yaml.safe_dump(live), encoding="utf-8")
    _write_json(
        tmp_path / "recipe.json",
        {
            "zone": "BE",
            "timezone": "Europe/Brussels",
            "prediction_mode": "autonomous_only",
            "mkonline_enabled": False,
            "external_expert": None,
            "autonomous_validation": {
                "approved": True,
                "final_loaded": False,
                "reason": "B1/B2 rejected MKOnline",
            },
        },
    )

    audit = audit_zone_live_bundle(registry, zone="BE", registry_dir=tmp_path)

    assert audit.ready, audit.blockers
    assert "dependency_manifest:not_used_autonomous_only" in audit.checks


def test_audit_never_falls_back_to_fr_runner(tmp_path: Path) -> None:
    registry = _fake_ready_bundle(tmp_path)
    (tmp_path / "run_mkonline_live_hourly.py").write_text("pass\n", encoding="utf-8")
    registry["zones"]["BE"]["runner"] = "run_mkonline_live_hourly.py"
    audit = audit_zone_live_bundle(registry, zone="BE", registry_dir=tmp_path)
    assert not audit.ready
    assert any("runner FR" in blocker for blocker in audit.blockers)
    with pytest.raises(ZoneBundleError, match="lancement refuse"):
        build_zone_runner_command(audit)


def test_audit_rejects_unavailable_status_when_native_storm_is_verified(
    tmp_path: Path,
) -> None:
    registry = _fake_ready_bundle(tmp_path)
    zone = registry["zones"]["BE"]
    zone["storm_series"] = None
    zone["storm_status"] = "native_dashboard_unavailable"
    zone["storm_primary_series"] = None
    zone["storm_naive_timezone"] = None
    zone["storm_pit_path"] = None

    audit = audit_zone_live_bundle(registry, zone="BE", registry_dir=tmp_path)

    assert not audit.ready
    assert any(
        "cache Storm day-ahead dashboard" in item
        for item in audit.blockers
    )
    assert "storm_strict_08:pit_unavailable_optional" in audit.checks


def test_checked_in_registry_has_all_requested_zones_and_is_fail_closed() -> None:
    registry, root = load_zone_registry(PROJECT_ROOT / "chronos2_hourly_live_zones.yaml")
    assert set(registry["zones"]) == {"FR", "BE", "DE", "ES", "NL", "GB", "IT"}
    assert registry["zones"]["FR"]["production_ready"] is True
    assert registry["zones"]["GB"]["delivery_timezone"] == "Europe/London"
    assert registry["zones"]["GB"]["forecast_origin_timezone"] == "Europe/Paris"
    for zone in ("FR", "BE", "DE", "ES", "NL", "GB", "IT"):
        audit = audit_zone_live_bundle(registry, zone=zone, registry_dir=root)
        declared_ready = bool(
            registry["zones"][zone].get("enabled")
            and registry["zones"][zone].get("production_ready")
        )
        assert audit.ready is declared_ready
        assert bool(audit.blockers) is (not declared_ready)


def test_audited_nl_target_can_generate_training_candidate_config() -> None:
    registry, _root = load_zone_registry(PROJECT_ROOT / "chronos2_hourly_live_zones.yaml")
    result = build_zone_training_config(
        _template(),
        zone="NL",
        contract={
            **registry["zones"]["NL"],
            "required_covariates": ["fr_residual_load_fcst"],
        },
    )
    assert result["zones"]["NL"]["target"]["series"] == (
        "power.price.da.nl.bzn.hourly.entsoe.utc.cdh.eurmwh"
    )


def test_italian_incomplete_target_cannot_generate_training_config() -> None:
    registry, _root = load_zone_registry(PROJECT_ROOT / "chronos2_hourly_live_zones.yaml")
    with pytest.raises(ZoneBundleError, match="cible DST-stricte"):
        build_zone_training_config(
            _template(),
            zone="IT",
            contract={
                **registry["zones"]["IT"],
                "bidding_zone": "IT_NORD",
                "required_covariates": ["fr_residual_load_fcst"],
            },
        )
