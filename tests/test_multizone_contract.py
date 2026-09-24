from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

import chronos2_hourly.multizone_contract as multizone_contract
from chronos2_hourly.multizone_contract import (
    FR_BLEND_WEIGHTS,
    LIVE_REFIT_SOURCE_CODE_PATHS,
    ZoneModelContractError,
    load_zone_model_contract,
)
from chronos2_hourly.zone_live import ZoneBundleAudit, strict_contract_preflight


ZONE_CASES = {
    "BE": {
        "timezone": "Europe/Brussels",
        "target": "power.price.da.be.bzn.hourly.entsoe.utc.cdh.eurmwh",
        "primary": "41555_native",
        "storm": "power.price.be.euromwh.h.fcst.3mv.storm.da.cache",
        "storm_primary": "power.price.be.euromwh.h.fcst.3mv.storm.da.cache",
    },
    "DE": {
        "timezone": "Europe/Berlin",
        "target": "power.price.da.de_lu.bzn.hourly.entsoe.utc.cdh.eurmwh",
        "primary": "41550_native",
        "storm": "power.price.de.euromwh.h.fcst.3mv.storm.da.cache",
        "storm_primary": "power.price.de.euromwh.h.fcst.3mv.storm.da.cache",
    },
    "NL": {
        "timezone": "Europe/Amsterdam",
        "target": "power.price.da.nl.bzn.hourly.entsoe.utc.cdh.eurmwh",
        "primary": "41554_native",
        "storm": "power.price.nl.euromwh.h.fcst.3mv.storm.da.cache",
        "storm_primary": "power.price.nl.euromwh.h.fcst.3mv.storm.da.cache",
    },
    "ES": {
        "timezone": "Europe/Madrid",
        "target": "power.price.da.es.bzn.hourly.entsoe.utc.cdh.eurmwh",
        "primary": "58307_native",
        "storm": None,
        "storm_primary": None,
    },
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_run(
    path: Path,
    *,
    zone: str,
    timezone: str,
    forecast_filename: str,
    comparators: list[str] | None = None,
    recipe_mode: str = "mkonline_blend",
) -> str:
    path.mkdir(parents=True)
    (path / "backtest_hourly_oof.csv.gz").write_bytes(b"sealed-backtest")
    (path / "feature_manifest.csv").write_text(
        "feature,dtype\nknown_hour_sin,float32\n",
        encoding="utf-8",
    )
    (path / "chronos_oof_hourly.csv.gz").write_bytes(b"sealed-chronos-oof")
    (path / "chronos_live_hourly.csv").write_bytes(b"sealed-chronos-live")
    inputs = path / "inputs"
    inputs.mkdir()
    for filename in (
        "chronos_oof_extended.csv.gz",
        "aligned_inputs.csv.gz",
        "model_covariates_with_future.csv.gz",
    ):
        (inputs / filename).write_bytes(f"sealed-{filename}".encode("utf-8"))
    (path / forecast_filename).write_text(
        "delivery_start_utc,q50\n2026-08-13T22:00:00Z,50\n",
        encoding="utf-8",
    )
    _write_json(
        path / "run_manifest.json",
        {
            "zone": zone,
            "timezone": timezone,
            "target_series": ZONE_CASES[zone]["target"],
            "storm_used_as_feature": False,
            "recipe_mode": recipe_mode,
            "prediction_inputs": (
                ["autonomous_extended_residual"]
                if recipe_mode == "autonomous_only"
                else ["autonomous", "mkonline_primary"]
            ),
            "evaluation_only_comparators": comparators or [],
        },
    )
    artifacts = []
    for filename in (
        "run_manifest.json",
        "backtest_hourly_oof.csv.gz",
        forecast_filename,
        "feature_manifest.csv",
        "chronos_oof_hourly.csv.gz",
        "chronos_live_hourly.csv",
        "inputs/chronos_oof_extended.csv.gz",
        "inputs/aligned_inputs.csv.gz",
        "inputs/model_covariates_with_future.csv.gz",
    ):
        artifact = path / filename
        artifacts.append(
            {
                "path": filename,
                "role": (
                    "materialized_input"
                    if filename.startswith("inputs/")
                    else "run_artifact"
                ),
                "size_bytes": artifact.stat().st_size,
                "sha256": _sha256(artifact),
            }
        )
    for filename in LIVE_REFIT_SOURCE_CODE_PATHS:
        source = PROJECT_ROOT / filename
        artifacts.append(
            {
                "path": filename,
                "role": "source_code",
                "size_bytes": source.stat().st_size,
                "sha256": _sha256(source),
            }
        )
    _write_json(
        path / "artifact_checksums.json",
        {"algorithm": "sha256", "artifacts": artifacts},
    )
    return _sha256(path / "artifact_checksums.json")


def _bundle(
    tmp_path: Path,
    *,
    zone: str = "BE",
    weights: tuple[float, float] = (0.6, 0.4),
) -> dict[str, Any]:
    spec = ZONE_CASES[zone]
    zone_key = zone.lower()
    root = tmp_path / zone_key
    root.mkdir()
    required_covariates = (f"{zone_key}_load_fcst", "fr_load_fcst")
    forecast_filename = f"forecast_hourly_{zone_key}.csv"

    base_config_path = root / f"model_{zone_key}.yaml"
    _write_yaml(
        base_config_path,
        {
            "zones": {
                zone: {
                    "timezone": spec["timezone"],
                    "target": {"series": spec["target"]},
                    "covariates": {
                        alias: {
                            "enabled": True,
                            "source": "pit_parquet",
                            "series": f"power.{alias}",
                        }
                        for alias in required_covariates
                    },
                }
            }
        },
    )
    dashboard = spec["storm"]
    comparators = [dashboard] if dashboard is not None else []
    frozen_path = root / f"autonomous_{zone_key}"
    benchmark_path = root / f"benchmark_{zone_key}"
    frozen_hash = _write_run(
        frozen_path,
        zone=zone,
        timezone=spec["timezone"],
        forecast_filename=forecast_filename,
    )
    benchmark_hash = _write_run(
        benchmark_path,
        zone=zone,
        timezone=spec["timezone"],
        forecast_filename=forecast_filename,
        comparators=comparators,
    )

    dependency_path = root / f"dependency_{zone_key}.json"
    _write_json(
        dependency_path,
        {
            "zone": zone,
            "timezone": spec["timezone"],
            "wrapper_series": f"power.price.{zone_key}.euromwh.h.fcst.mkonline.ecop",
            "terminal_series": spec["primary"],
            "terminal_type": "primary",
            "terminal_formula": None,
            "terminal_metadata": {"mercure:country": zone},
            "storm_token_found": False,
            "dependency_gate_passed": True,
        },
    )
    dependency_hash = _sha256(dependency_path)
    recipe_path = root / f"recipe_{zone_key}.json"
    _write_json(
        recipe_path,
        {
            "prediction_mode": "mkonline_blend",
            "mkonline_enabled": True,
            "zone": zone,
            "timezone": spec["timezone"],
            "source_autonomous_run": str(frozen_path),
            "source_autonomous_checksum_manifest_sha256": frozen_hash,
            "external_expert": {
                "series": spec["primary"],
                "wrapper_series": (
                    f"power.price.{zone_key}.euromwh.h.fcst.mkonline.ecop"
                ),
                "dependency_manifest": str(dependency_path),
                "dependency_manifest_sha256": dependency_hash,
                "storm_used_as_feature": False,
            },
            "weights": {
                "autonomous": weights[0],
                "mkonline_primary": weights[1],
            },
        },
    )

    live_path = root / f"live_{zone_key}.yaml"
    live = {
        "live": {
            "contract_schema_version": 1,
            "zone": zone,
            "delivery_timezone": spec["timezone"],
            "forecast_origin_timezone": "Europe/Paris",
            "forecast_origin_local_time": "08:00",
            "target_series": spec["target"],
            "primary_series": spec["primary"],
            "storm_dashboard_series": dashboard,
            "storm_dashboard_primary_series": spec["storm_primary"],
            "storm_dashboard_naive_timezone": (
                spec["timezone"] if dashboard is not None else None
            ),
            "storm_strict_08_series": (
                f"power.price.{zone_key}.euromwh.h.fcst.3mv.storm.da.basecase"
            ),
            "forecast_filename": forecast_filename,
            "required_covariates": list(required_covariates),
            "base_config": str(base_config_path),
            "frozen_autonomous_run": str(frozen_path),
            "sealed_benchmark_run": str(benchmark_path),
            "recipe_manifest": str(recipe_path),
            "dependency_manifest": str(dependency_path),
            "output_root": str(root / f"live_runs_{zone_key}"),
            "expected_hashes": {
                "base_config_sha256": _sha256(base_config_path),
                "frozen_autonomous_checksum_manifest_sha256": frozen_hash,
                "sealed_benchmark_checksum_manifest_sha256": benchmark_hash,
                "recipe_manifest_sha256": _sha256(recipe_path),
                "dependency_manifest_sha256": dependency_hash,
            },
            "weights": {
                "autonomous": weights[0],
                "mkonline_primary": weights[1],
            },
            "prediction_mode": "mkonline_blend",
            "mkonline_enabled": True,
        }
    }
    _write_yaml(live_path, live)

    registry_path = root / "registry.yaml"
    registry_zone = {
        "enabled": True,
        "production_ready": True,
        "delivery_timezone": spec["timezone"],
        "forecast_origin_timezone": "Europe/Paris",
        "forecast_origin_local_time": "08:00",
        "prediction_mode": "mkonline_blend",
        "mkonline_enabled": True,
        "live_config": str(live_path),
        "target_series": spec["target"],
        "target_status": "audited_dst_strict",
        "price_unit": "EUR/MWh",
        "primary_series": spec["primary"],
        "primary_status": "audited_primary",
        "storm_series": dashboard,
        "storm_status": (
            "audited_day_ahead_cache"
            if dashboard is not None
            else "native_dashboard_unavailable"
        ),
        "storm_primary_series": spec["storm_primary"],
        "storm_naive_timezone": (
            spec["timezone"] if dashboard is not None else None
        ),
        "storm_strict_08_series": (
            f"power.price.{zone_key}.euromwh.h.fcst.3mv.storm.da.basecase"
        ),
        "required_covariates": list(required_covariates),
    }
    _write_yaml(
        registry_path,
        {"schema_version": 1, "zones": {zone: registry_zone}},
    )
    return {
        "root": root,
        "spec": spec,
        "live_path": live_path,
        "live": live,
        "registry_path": registry_path,
        "registry_zone": registry_zone,
        "base_config_path": base_config_path,
        "frozen_path": frozen_path,
        "benchmark_path": benchmark_path,
        "recipe_path": recipe_path,
        "dependency_path": dependency_path,
    }


def _make_autonomous_only(bundle: dict[str, Any]) -> None:
    live = bundle["live"]["live"]
    registry = bundle["registry_zone"]
    recipe = json.loads(bundle["recipe_path"].read_text(encoding="utf-8"))
    recipe.update(
        {
            "prediction_mode": "autonomous_only",
            "recipe_mode": "autonomous_only",
            "mkonline_enabled": False,
            "autonomous_only_reason": "B1/B2 rejected the external expert",
            "external_expert": None,
            "weights": {"autonomous": 1.0, "mkonline_primary": 0.0},
        }
    )
    _write_json(bundle["recipe_path"], recipe)
    live.update(
        {
            "primary_series": None,
            "dependency_manifest": None,
            "prediction_mode": "autonomous_only",
            "mkonline_enabled": False,
            "weights": {"autonomous": 1.0, "mkonline_primary": 0.0},
        }
    )
    live["expected_hashes"]["recipe_manifest_sha256"] = _sha256(
        bundle["recipe_path"]
    )
    live["expected_hashes"]["dependency_manifest_sha256"] = None
    registry.update(
        {
            "primary_series": None,
            "primary_status": "not_used_autonomous_only",
            "prediction_mode": "autonomous_only",
            "mkonline_enabled": False,
        }
    )
    manifest_path = bundle["benchmark_path"] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recipe_mode"] = "autonomous_only"
    manifest["prediction_inputs"] = ["autonomous_extended_residual"]
    _write_json(manifest_path, manifest)
    checksum_path = bundle["benchmark_path"] / "artifact_checksums.json"
    checksum = json.loads(checksum_path.read_text(encoding="utf-8"))
    for artifact in checksum["artifacts"]:
        if artifact["path"] == "run_manifest.json":
            artifact["sha256"] = _sha256(manifest_path)
            artifact["size_bytes"] = manifest_path.stat().st_size
    _write_json(checksum_path, checksum)
    live["expected_hashes"]["sealed_benchmark_checksum_manifest_sha256"] = (
        _sha256(checksum_path)
    )
    _rewrite_live(bundle)
    _rewrite_registry(bundle)


@pytest.mark.parametrize("zone", ["BE", "DE", "ES"])
def test_autonomous_only_contract_needs_no_primary_or_dependency(
    tmp_path: Path,
    zone: str,
) -> None:
    bundle = _bundle(tmp_path, zone=zone)
    _make_autonomous_only(bundle)

    contract = load_zone_model_contract(
        bundle["live_path"], bundle["registry_path"]
    )

    assert contract.prediction_mode == "autonomous_only"
    assert contract.mkonline_enabled is False
    assert contract.primary_series is None
    assert contract.paths.dependency_manifest is None
    assert contract.checksum_hashes.dependency_manifest_sha256 is None
    assert contract.weights == type(contract.weights)(1.0, 0.0)
    assert contract.as_dict()["paths"]["dependency_manifest"] is None


def test_autonomous_recipe_must_pin_the_frozen_source(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, zone="BE")
    _make_autonomous_only(bundle)
    recipe = json.loads(bundle["recipe_path"].read_text(encoding="utf-8"))
    recipe["source_autonomous_run"] = str(bundle["benchmark_path"])
    _write_json(bundle["recipe_path"], recipe)
    bundle["live"]["live"]["expected_hashes"]["recipe_manifest_sha256"] = (
        _sha256(bundle["recipe_path"])
    )
    _rewrite_live(bundle)

    with pytest.raises(ZoneModelContractError, match="source_autonomous_run"):
        load_zone_model_contract(bundle["live_path"], bundle["registry_path"])


def test_autonomous_only_rejects_dependency_or_primary(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, zone="BE")
    _make_autonomous_only(bundle)
    bundle["live"]["live"]["primary_series"] = "41555_native"
    bundle["registry_zone"]["primary_series"] = "41555_native"
    bundle["registry_zone"]["primary_status"] = "audited_primary"
    _rewrite_live(bundle)
    _rewrite_registry(bundle)

    with pytest.raises(ZoneModelContractError, match="null primary_series"):
        load_zone_model_contract(bundle["live_path"], bundle["registry_path"])


def test_dispatch_preflight_rejects_a_stale_declared_hash(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, zone="BE")
    _make_autonomous_only(bundle)
    audit = ZoneBundleAudit(
        zone="BE",
        timezone="Europe/Brussels",
        forecast_origin_timezone="Europe/Paris",
        ready=True,
        enabled=True,
        production_ready=True,
        checks=("shallow:ok",),
        blockers=(),
        runner=bundle["root"] / "run_mkonline_live_model.py",
        live_config=bundle["live_path"],
    )
    bundle["base_config_path"].write_text("tampered: true\n", encoding="utf-8")

    checked = strict_contract_preflight(
        audit,
        registry_path=bundle["registry_path"],
    )

    assert checked.ready is False
    assert any("SHA-256" in blocker for blocker in checked.blockers)
    with pytest.raises(Exception, match="lancement refuse"):
        checked.require_ready()


@pytest.mark.parametrize("zone", ["DE", "BE", "ES"])
def test_checked_in_autonomous_live_bundle_passes_real_strict_contract(
    tmp_path: Path,
    zone: str,
) -> None:
    """Activate only a temporary registry copy; production remains untouched."""

    source_registry = PROJECT_ROOT / "chronos2_hourly_live_zones.yaml"
    registry = yaml.safe_load(source_registry.read_text(encoding="utf-8"))
    raw = registry["zones"][zone]
    raw.update(
        {
            "enabled": True,
            "production_ready": True,
            "runner": str(PROJECT_ROOT / "run_mkonline_live_model.py"),
            "live_config": str(
                PROJECT_ROOT
                / f"chronos2_hourly_{zone.lower()}_mkonline_live_v1.yaml"
            ),
            "prediction_mode": "autonomous_only",
            "mkonline_enabled": False,
            "primary_series": None,
            "primary_status": "not_used_autonomous_only",
            "blockers": [],
        }
    )
    registry_path = tmp_path / "registry.yaml"
    _write_yaml(registry_path, registry)

    contract = load_zone_model_contract(
        raw["live_config"],
        registry_path,
    )

    assert contract.zone == zone
    assert contract.prediction_mode == "autonomous_only"
    assert contract.mkonline_enabled is False
    assert contract.primary_series is None
    assert contract.paths.dependency_manifest is None


def _rewrite_live(bundle: dict[str, Any]) -> None:
    _write_yaml(bundle["live_path"], bundle["live"])


def _rewrite_registry(bundle: dict[str, Any]) -> None:
    _write_yaml(
        bundle["registry_path"],
        {
            "schema_version": 1,
            "zones": {
                bundle["live"]["live"]["zone"]: bundle["registry_zone"]
            },
        },
    )


@pytest.mark.parametrize("zone", ["BE", "DE", "NL", "ES"])
def test_strict_contract_loads_independent_zone_bundle(
    tmp_path: Path,
    zone: str,
) -> None:
    bundle = _bundle(tmp_path, zone=zone)
    contract = load_zone_model_contract(
        bundle["live_path"], bundle["registry_path"]
    )

    assert contract.zone == zone
    assert contract.delivery_timezone == bundle["spec"]["timezone"]
    assert contract.primary_series == bundle["spec"]["primary"]
    assert contract.storm_dashboard_series == bundle["spec"]["storm"]
    assert contract.forecast_filename == f"forecast_hourly_{zone.lower()}.csv"
    assert contract.paths.live_config == bundle["live_path"].resolve()
    assert contract.weights.autonomous == 0.6
    assert isinstance(contract.required_covariates, tuple)
    audit = contract.as_dict()
    assert audit["paths"]["recipe_manifest"] == str(
        bundle["recipe_path"].resolve()
    )


def test_es_can_be_model_ready_only_with_explicit_unavailable_native_storm(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path, zone="ES")
    probable_but_unverified = (
        "power.price.es.euromwh.h.fcst.3mv.storm.da.cache"
    )
    live = bundle["live"]["live"]
    live["storm_dashboard_series"] = probable_but_unverified
    live["storm_dashboard_primary_series"] = "unverified_native"
    live["storm_dashboard_naive_timezone"] = "Europe/Madrid"
    registry = bundle["registry_zone"]
    registry["storm_series"] = probable_but_unverified
    registry["storm_primary_series"] = "unverified_native"
    registry["storm_naive_timezone"] = "Europe/Madrid"
    registry["storm_status"] = "audited_day_ahead_cache"
    _rewrite_live(bundle)
    _rewrite_registry(bundle)

    with pytest.raises(ZoneModelContractError, match="has not been verified"):
        load_zone_model_contract(bundle["live_path"], bundle["registry_path"])


def test_verified_native_storm_cannot_be_marked_unavailable(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, zone="BE")
    live = bundle["live"]["live"]
    live["storm_dashboard_series"] = None
    live["storm_dashboard_primary_series"] = None
    live["storm_dashboard_naive_timezone"] = None
    registry = bundle["registry_zone"]
    registry["storm_series"] = None
    registry["storm_primary_series"] = None
    registry["storm_naive_timezone"] = None
    registry["storm_status"] = "native_dashboard_unavailable"
    _rewrite_live(bundle)
    _rewrite_registry(bundle)

    with pytest.raises(ZoneModelContractError, match="mandatory"):
        load_zone_model_contract(bundle["live_path"], bundle["registry_path"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda live: live.pop("expected_hashes"), "expected_hashes"),
        (lambda live: live.pop("weights"), "weights"),
        (lambda live: live.pop("forecast_filename"), "forecast_filename"),
        (lambda live: live.pop("storm_dashboard_series"), "storm_dashboard_series"),
    ],
)
def test_contract_requires_all_explicit_declarations(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    bundle = _bundle(tmp_path)
    mutation(bundle["live"]["live"])
    _rewrite_live(bundle)

    with pytest.raises(ZoneModelContractError, match=message):
        load_zone_model_contract(bundle["live_path"], bundle["registry_path"])


def test_contract_rejects_non_strict_mode(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    with pytest.raises(ZoneModelContractError, match="non-strict"):
        load_zone_model_contract(
            bundle["live_path"], bundle["registry_path"], strict=False
        )


def test_contract_rejects_tampered_run_artifact(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (bundle["frozen_path"] / "forecast_hourly_be.csv").write_text(
        "tampered\n", encoding="utf-8"
    )

    with pytest.raises(ZoneModelContractError, match="artifact checksum mismatch"):
        load_zone_model_contract(bundle["live_path"], bundle["registry_path"])


def test_contract_rejects_tampered_live_fit_input(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (bundle["frozen_path"] / "inputs" / "aligned_inputs.csv.gz").write_bytes(
        b"tampered-live-fit-input"
    )

    with pytest.raises(ZoneModelContractError, match="artifact checksum mismatch"):
        load_zone_model_contract(bundle["live_path"], bundle["registry_path"])


def test_contract_rejects_tampered_live_refit_model_source_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    isolated_source_root = tmp_path / "isolated_algorithm_source"
    for relative in LIVE_REFIT_SOURCE_CODE_PATHS:
        source = PROJECT_ROOT / relative
        isolated = isolated_source_root / relative
        isolated.parent.mkdir(parents=True, exist_ok=True)
        isolated.write_bytes(source.read_bytes())
    monkeypatch.setattr(
        multizone_contract,
        "ALGORITHM_SOURCE_ROOT",
        isolated_source_root,
    )

    model_source = (
        isolated_source_root
        / "chronos2_hourly/models/residual_corrector.py"
    )
    tampered = bytearray(model_source.read_bytes())
    tampered[-1] ^= 1
    model_source.write_bytes(tampered)

    with pytest.raises(
        ZoneModelContractError,
        match=r"source_code checksum mismatch: .*residual_corrector\.py",
    ):
        load_zone_model_contract(bundle["live_path"], bundle["registry_path"])


def test_contract_rejects_unsafe_run_artifact_path(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    checksum_path = bundle["frozen_path"] / "artifact_checksums.json"
    checksum = json.loads(checksum_path.read_text(encoding="utf-8"))
    checksum["artifacts"].append(
        {
            "path": "../foreign-zone-secret.csv",
            "role": "run_artifact",
            "sha256": "0" * 64,
        }
    )
    _write_json(checksum_path, checksum)
    bundle["live"]["live"]["expected_hashes"][
        "frozen_autonomous_checksum_manifest_sha256"
    ] = _sha256(checksum_path)
    recipe = json.loads(bundle["recipe_path"].read_text(encoding="utf-8"))
    recipe["source_autonomous_checksum_manifest_sha256"] = _sha256(checksum_path)
    _write_json(bundle["recipe_path"], recipe)
    bundle["live"]["live"]["expected_hashes"]["recipe_manifest_sha256"] = (
        _sha256(bundle["recipe_path"])
    )
    _rewrite_live(bundle)

    with pytest.raises(
        ZoneModelContractError,
        match="unsafe checksummed artifact path",
    ):
        load_zone_model_contract(bundle["live_path"], bundle["registry_path"])


def test_contract_rejects_registry_live_config_cross_wire(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle["registry_zone"]["live_config"] = str(
        bundle["root"] / "another_zone.yaml"
    )
    _rewrite_registry(bundle)

    with pytest.raises(ZoneModelContractError, match="live_config.*mismatch"):
        load_zone_model_contract(bundle["live_path"], bundle["registry_path"])


def test_contract_rejects_zone_manifest_cross_wire(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    manifest = bundle["benchmark_path"] / "run_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["zone"] = "FR"
    _write_json(manifest, payload)
    # Reseal the checksum declaration and its explicit outer fingerprint so
    # the zone check, not a stale hash, proves the cross-wire is rejected.
    checksum = bundle["benchmark_path"] / "artifact_checksums.json"
    checksum_payload = json.loads(checksum.read_text(encoding="utf-8"))
    for artifact in checksum_payload["artifacts"]:
        if artifact["path"] == "run_manifest.json":
            artifact["sha256"] = _sha256(manifest)
            artifact["size_bytes"] = manifest.stat().st_size
    _write_json(checksum, checksum_payload)
    bundle["live"]["live"]["expected_hashes"][
        "sealed_benchmark_checksum_manifest_sha256"
    ] = _sha256(checksum)
    _rewrite_live(bundle)

    with pytest.raises(ZoneModelContractError, match="manifest.zone mismatch"):
        load_zone_model_contract(bundle["live_path"], bundle["registry_path"])


def test_contract_rejects_fr_primary_outside_fr(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    fr_primary = "41551_native"
    bundle["live"]["live"]["primary_series"] = fr_primary
    bundle["registry_zone"]["primary_series"] = fr_primary
    dependency = json.loads(bundle["dependency_path"].read_text(encoding="utf-8"))
    dependency["terminal_series"] = fr_primary
    _write_json(bundle["dependency_path"], dependency)
    dependency_hash = _sha256(bundle["dependency_path"])
    recipe = json.loads(bundle["recipe_path"].read_text(encoding="utf-8"))
    recipe["external_expert"]["series"] = fr_primary
    recipe["external_expert"]["dependency_manifest_sha256"] = dependency_hash
    _write_json(bundle["recipe_path"], recipe)
    hashes = bundle["live"]["live"]["expected_hashes"]
    hashes["dependency_manifest_sha256"] = dependency_hash
    hashes["recipe_manifest_sha256"] = _sha256(bundle["recipe_path"])
    _rewrite_live(bundle)
    _rewrite_registry(bundle)

    with pytest.raises(ZoneModelContractError, match="France primary"):
        load_zone_model_contract(bundle["live_path"], bundle["registry_path"])


def test_contract_rejects_sealed_fr_weights_outside_fr(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, weights=FR_BLEND_WEIGHTS)

    with pytest.raises(ZoneModelContractError, match="France blend weights"):
        load_zone_model_contract(bundle["live_path"], bundle["registry_path"])


def test_contract_rejects_fr_labelled_artifact_outside_fr(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    fr_recipe = bundle["root"] / "recipe_fr.json"
    fr_recipe.write_bytes(bundle["recipe_path"].read_bytes())
    bundle["live"]["live"]["recipe_manifest"] = str(fr_recipe)
    bundle["live"]["live"]["expected_hashes"]["recipe_manifest_sha256"] = (
        _sha256(fr_recipe)
    )
    _rewrite_live(bundle)

    with pytest.raises(ZoneModelContractError, match="France-labelled"):
        load_zone_model_contract(bundle["live_path"], bundle["registry_path"])


def test_contract_rejects_storm_as_model_covariate(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle["live"]["live"]["required_covariates"].append("storm_price")
    bundle["registry_zone"]["required_covariates"].append("storm_price")
    _rewrite_live(bundle)
    _rewrite_registry(bundle)

    with pytest.raises(ZoneModelContractError, match="evaluation-only"):
        load_zone_model_contract(bundle["live_path"], bundle["registry_path"])


def test_contract_rejects_foreign_storm_comparator(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    manifest_path = bundle["benchmark_path"] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluation_only_comparators"].append(
        "power.price.fr.euromwh.h.fcst.3mv.storm.da.cache"
    )
    _write_json(manifest_path, manifest)
    checksum_path = bundle["benchmark_path"] / "artifact_checksums.json"
    checksum = json.loads(checksum_path.read_text(encoding="utf-8"))
    for artifact in checksum["artifacts"]:
        if artifact["path"] == "run_manifest.json":
            artifact["sha256"] = _sha256(manifest_path)
            artifact["size_bytes"] = manifest_path.stat().st_size
    _write_json(checksum_path, checksum)
    bundle["live"]["live"]["expected_hashes"][
        "sealed_benchmark_checksum_manifest_sha256"
    ] = _sha256(checksum_path)
    _rewrite_live(bundle)

    with pytest.raises(ZoneModelContractError, match="foreign-zone Storm"):
        load_zone_model_contract(bundle["live_path"], bundle["registry_path"])


def test_contract_rejects_registry_that_still_declares_blockers(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    bundle["registry_zone"]["blockers"] = ["benchmark not independently sealed"]
    _rewrite_registry(bundle)

    with pytest.raises(ZoneModelContractError, match="still declares blockers"):
        load_zone_model_contract(bundle["live_path"], bundle["registry_path"])


def test_contract_rejects_output_root_overlapping_frozen_input(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    bundle["live"]["live"]["output_root"] = str(
        bundle["frozen_path"] / "live_output_be"
    )
    _rewrite_live(bundle)

    with pytest.raises(ZoneModelContractError, match="must not overlap"):
        load_zone_model_contract(bundle["live_path"], bundle["registry_path"])
