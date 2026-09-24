"""Fail-closed contracts for extending the hourly pipeline to market zones.

The validated France live runner is intentionally not parameterised here.  A
zone becomes dispatchable only after it owns a complete, zone-specific bundle
(training run, sealed benchmark, blend recipe, comparator and PIT inputs).
This prevents a missing Belgian/German/etc. artefact from silently falling
back to the French model.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from chronos2_hourly.storm_dashboard import (
    STORM_DASHBOARD_CACHE_SERIES_BY_ZONE,
    STORM_DASHBOARD_NATIVE_NAIVE_TIMEZONE_BY_ZONE,
    STORM_DASHBOARD_PRIMARY_SERIES_BY_ZONE,
)
from chronos2_modular.common import load_yaml


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MarketZone:
    code: str
    timezone: str
    holiday_country: str
    aliases: tuple[str, ...] = ()
    requires_explicit_bidding_zone: bool = False


MARKET_ZONES: dict[str, MarketZone] = {
    "FR": MarketZone("FR", "Europe/Paris", "FR"),
    "BE": MarketZone("BE", "Europe/Brussels", "BE"),
    "DE": MarketZone("DE", "Europe/Berlin", "DE"),
    "ES": MarketZone("ES", "Europe/Madrid", "ES"),
    "NL": MarketZone("NL", "Europe/Amsterdam", "NL"),
    # The user-facing UK alias is normalised to the canonical GB country code.
    "GB": MarketZone("GB", "Europe/London", "GB", aliases=("UK",)),
    # Italy has several bidding zones; a generic IT price must never be guessed.
    "IT": MarketZone(
        "IT",
        "Europe/Rome",
        "IT",
        requires_explicit_bidding_zone=True,
    ),
}


class ZoneBundleError(ValueError):
    """Raised when a market-zone bundle is not safe to train or dispatch."""


@dataclass(frozen=True)
class ZoneBundleAudit:
    zone: str
    timezone: str
    forecast_origin_timezone: str
    ready: bool
    enabled: bool
    production_ready: bool
    checks: tuple[str, ...]
    blockers: tuple[str, ...]
    runner: Path | None = None
    live_config: Path | None = None

    def require_ready(self) -> "ZoneBundleAudit":
        if not self.ready:
            detail = "; ".join(self.blockers) or "bundle incomplet"
            raise ZoneBundleError(f"{self.zone}: lancement refuse: {detail}")
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "zone": self.zone,
            "timezone": self.timezone,
            "delivery_timezone": self.timezone,
            "forecast_origin_timezone": self.forecast_origin_timezone,
            "ready": self.ready,
            "enabled": self.enabled,
            "production_ready": self.production_ready,
            "checks": list(self.checks),
            "blockers": list(self.blockers),
            "runner": str(self.runner) if self.runner is not None else None,
            "live_config": (
                str(self.live_config) if self.live_config is not None else None
            ),
        }


def canonical_zone(value: str) -> str:
    requested = str(value).strip().upper()
    for code, spec in MARKET_ZONES.items():
        if requested == code or requested in spec.aliases:
            return code
    allowed = sorted(
        {code for code in MARKET_ZONES}
        | {alias for spec in MARKET_ZONES.values() for alias in spec.aliases}
    )
    raise ZoneBundleError(
        f"Zone inconnue {value!r}; valeurs autorisees: {', '.join(allowed)}"
    )


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ZoneBundleError(f"{name} doit etre un mapping")
    return value


def _text(value: Any) -> str:
    """Normalise optional YAML scalars without turning null into ``'None'``."""

    return "" if value is None else str(value).strip()


def _resolve(value: Any, *, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_zone_registry(path: str | Path) -> tuple[dict[str, Any], Path]:
    registry_path = Path(path).expanduser().resolve()
    payload = load_yaml(registry_path)
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ZoneBundleError(
            f"{registry_path}: schema_version={payload.get('schema_version')!r}, "
            f"attendu={SCHEMA_VERSION}"
        )
    zones = _mapping(payload.get("zones"), name="zones")
    unknown = sorted(set(str(key).upper() for key in zones) - set(MARKET_ZONES))
    if unknown:
        raise ZoneBundleError(f"Zones non supportees dans le registre: {unknown}")
    return payload, registry_path.parent


def _registry_zone(
    registry: Mapping[str, Any],
    zone: str,
) -> tuple[str, MarketZone, Mapping[str, Any]]:
    code = canonical_zone(zone)
    zones = _mapping(registry.get("zones"), name="zones")
    raw = zones.get(code)
    if not isinstance(raw, Mapping):
        raise ZoneBundleError(f"{code}: zone absente du registre")
    return code, MARKET_ZONES[code], raw


def _read_json(path: Path, *, label: str, blockers: list[str]) -> Mapping[str, Any]:
    if not path.is_file():
        blockers.append(f"{label} absent: {path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(f"{label} illisible: {path} ({exc})")
        return {}
    if not isinstance(payload, Mapping):
        blockers.append(f"{label} invalide: objet JSON attendu")
        return {}
    return payload


def _expect_equal(
    observed: Any,
    expected: Any,
    *,
    label: str,
    blockers: list[str],
) -> None:
    if observed != expected:
        blockers.append(f"{label}: {observed!r} != {expected!r}")


def audit_zone_live_bundle(
    registry: Mapping[str, Any],
    *,
    zone: str,
    registry_dir: str | Path,
) -> ZoneBundleAudit:
    """Audit a local bundle without refreshing Saturn or running a model."""

    code, market, raw = _registry_zone(registry, zone)
    root = Path(registry_dir).expanduser().resolve()
    checks: list[str] = []
    blockers: list[str] = []
    enabled = bool(raw.get("enabled", False))
    production_ready = bool(raw.get("production_ready", False))
    if not enabled:
        blockers.append("zone desactivee dans le registre")
    if not production_ready:
        blockers.append("bundle non marque production_ready")

    declared_blockers = raw.get("blockers", [])
    if isinstance(declared_blockers, Sequence) and not isinstance(
        declared_blockers, (str, bytes)
    ):
        blockers.extend(f"registre: {_text(item)}" for item in declared_blockers if _text(item))

    timezone = _text(raw.get("delivery_timezone", raw.get("timezone")))
    _expect_equal(
        timezone,
        market.timezone,
        label="timezone du registre",
        blockers=blockers,
    )
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        blockers.append(f"timezone IANA invalide: {timezone!r}")
    forecast_origin_timezone = _text(raw.get("forecast_origin_timezone"))
    try:
        ZoneInfo(forecast_origin_timezone)
    except (ZoneInfoNotFoundError, ValueError):
        blockers.append(
            "forecast_origin_timezone IANA invalide: "
            f"{forecast_origin_timezone!r}"
        )
    forecast_origin_local_time = _text(raw.get("forecast_origin_local_time"))
    try:
        hour_text, minute_text = forecast_origin_local_time.split(":", maxsplit=1)
        valid_origin_time = (
            len(hour_text) == 2
            and len(minute_text) == 2
            and 0 <= int(hour_text) <= 23
            and 0 <= int(minute_text) <= 59
        )
    except (AttributeError, TypeError, ValueError):
        valid_origin_time = False
    if not valid_origin_time:
        blockers.append(
            "forecast_origin_local_time invalide; format HH:MM requis"
        )

    bidding_zone = _text(raw.get("bidding_zone"))
    if not bidding_zone:
        blockers.append("bidding_zone explicite absent")
    if market.requires_explicit_bidding_zone and bidding_zone.upper() == "IT":
        blockers.append("IT exige une zone de prix explicite (p. ex. IT_NORD)")

    target_series = _text(raw.get("target_series"))
    primary_series = _text(raw.get("primary_series"))
    registry_prediction_mode = _text(raw.get("prediction_mode"))
    # The sealed France bundle predates the explicit multi-zone mode fields.
    # Its audited primary, recipe and dedicated runner already define the
    # MKOnline blend; keep this compatibility shim confined to FR.
    if code == "FR" and not registry_prediction_mode:
        registry_prediction_mode = "mkonline_blend"
    storm_series = _text(raw.get("storm_series"))
    if not target_series:
        blockers.append("target_series non auditee")
    if registry_prediction_mode == "mkonline_blend" and not primary_series:
        blockers.append("primary_series non auditee pour mkonline_blend")
    if registry_prediction_mode == "autonomous_only" and primary_series:
        blockers.append("primary_series doit etre null pour autonomous_only")
    expected_statuses = {
        "target_status": "audited_dst_strict",
        "primary_status": (
            "not_used_autonomous_only"
            if registry_prediction_mode == "autonomous_only"
            else "audited_primary"
        ),
    }
    for field, expected in expected_statuses.items():
        observed = _text(raw.get(field))
        if observed != expected:
            blockers.append(f"{field}: {observed or 'absent'} != {expected}")

    # Storm is an evaluation-only dashboard comparator, never a model input.
    # The frozen day-ahead cache identifiers have been verified for these zones.
    # A strict D-1 08:00 basecase materialisation remains useful diagnostics,
    # but its absence cannot make an otherwise independent model unavailable.
    expected_dashboard = STORM_DASHBOARD_CACHE_SERIES_BY_ZONE.get(code)
    storm_status = _text(raw.get("storm_status"))
    storm_primary = _text(raw.get("storm_primary_series"))
    storm_timezone = _text(raw.get("storm_naive_timezone"))
    if (
        expected_dashboard is None
        and storm_status == "native_dashboard_unavailable"
    ):
        if storm_series or storm_primary or storm_timezone:
            blockers.append(
                "Storm dashboard indisponible exige series/primary/timezone null"
            )
        checks.append("storm_dashboard:unavailable_optional")
    elif expected_dashboard is not None:
        _expect_equal(
            storm_series,
            expected_dashboard,
            label="cache Storm day-ahead dashboard",
            blockers=blockers,
        )
        _expect_equal(
            storm_status,
            "audited_day_ahead_cache",
            label="storm_status",
            blockers=blockers,
        )
        _expect_equal(
            storm_primary,
            STORM_DASHBOARD_PRIMARY_SERIES_BY_ZONE[code],
            label="storm_primary_series",
            blockers=blockers,
        )
        _expect_equal(
            storm_timezone,
            STORM_DASHBOARD_NATIVE_NAIVE_TIMEZONE_BY_ZONE[code],
            label="storm_naive_timezone",
            blockers=blockers,
        )
        checks.append("storm_dashboard:day_ahead_cache")
    elif storm_status == "native_dashboard_unavailable":
        # Kept for readability of the two-state contract above.  This branch
        # is reachable only when a future zone is added without a native map.
        checks.append("storm_dashboard:unavailable_optional")
    elif not storm_series and not storm_status:
        blockers.append(
            "storm_status absent; audited_day_ahead_cache ou "
            "native_dashboard_unavailable requis"
        )
    elif storm_series:
        blockers.append(
            f"cache Storm day-ahead dashboard non verifie pour {code}"
        )
    price_unit = _text(raw.get("price_unit"))
    if price_unit != "EUR/MWh":
        blockers.append(
            f"price_unit: {price_unit or 'absent'}; le pipeline courant exige EUR/MWh"
        )

    runner = _resolve(raw.get("runner"), base=root)
    live_config_path = _resolve(raw.get("live_config"), base=root)
    if runner is None or not runner.is_file():
        blockers.append(f"runner absent: {runner}")
    if live_config_path is None or not live_config_path.is_file():
        blockers.append(f"configuration live absente: {live_config_path}")
    if code != "FR" and runner is not None and runner.name == "run_mkonline_live_hourly.py":
        blockers.append("le runner FR scelle ne peut pas etre reutilise hors FR")

    if live_config_path is None or not live_config_path.is_file():
        return ZoneBundleAudit(
            zone=code,
            timezone=market.timezone,
            forecast_origin_timezone=forecast_origin_timezone,
            ready=False,
            enabled=enabled,
            production_ready=production_ready,
            checks=tuple(checks),
            blockers=tuple(dict.fromkeys(blockers)),
            runner=runner,
            live_config=live_config_path,
        )

    try:
        live_config = load_yaml(live_config_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        blockers.append(f"configuration live illisible: {exc}")
        live_config = {}
    live = live_config.get("live")
    if not isinstance(live, Mapping):
        blockers.append("section live absente de la configuration")
        live = {}
    if code != "FR":
        _expect_equal(live.get("zone"), code, label="live.zone", blockers=blockers)
        _expect_equal(
            live.get("delivery_timezone"),
            market.timezone,
            label="live.delivery_timezone",
            blockers=blockers,
        )
        _expect_equal(
            live.get("forecast_origin_timezone"),
            forecast_origin_timezone,
            label="live.forecast_origin_timezone",
            blockers=blockers,
        )
        prediction_mode = _text(live.get("prediction_mode"))
        mkonline_enabled = live.get("mkonline_enabled")
        if prediction_mode not in {"mkonline_blend", "autonomous_only"}:
            blockers.append(
                "live.prediction_mode explicite mkonline_blend/autonomous_only requis"
            )
        if not isinstance(mkonline_enabled, bool):
            blockers.append("live.mkonline_enabled booleen explicite requis")
        _expect_equal(
            prediction_mode,
            _text(raw.get("prediction_mode")),
            label="live.prediction_mode versus registre",
            blockers=blockers,
        )
        _expect_equal(
            mkonline_enabled,
            raw.get("mkonline_enabled"),
            label="live.mkonline_enabled versus registre",
            blockers=blockers,
        )

    config_dir = live_config_path.parent
    base_config_path = _resolve(live.get("base_config"), base=config_dir)
    frozen_run = _resolve(live.get("frozen_autonomous_run"), base=config_dir)
    benchmark_run = _resolve(live.get("sealed_benchmark_run"), base=config_dir)
    recipe_path = _resolve(live.get("recipe_manifest"), base=config_dir)
    dependency_path = _resolve(live.get("dependency_manifest"), base=config_dir)

    required_paths = (
        ("base_config", base_config_path, False),
        ("frozen_autonomous_run", frozen_run, True),
        ("sealed_benchmark_run", benchmark_run, True),
        ("recipe_manifest", recipe_path, False),
        ("dependency_manifest", dependency_path, False),
    )
    for label, path, directory in required_paths:
        if label == "dependency_manifest" and registry_prediction_mode == "autonomous_only":
            if path is not None:
                blockers.append(
                    "dependency_manifest doit etre null pour autonomous_only"
                )
            else:
                checks.append("dependency_manifest:not_used_autonomous_only")
            continue
        exists = path is not None and (path.is_dir() if directory else path.is_file())
        if not exists:
            blockers.append(f"{label} absent: {path}")
        else:
            checks.append(f"{label}:present")

    base_config: Mapping[str, Any] = {}
    if base_config_path is not None and base_config_path.is_file():
        try:
            base_config = load_yaml(base_config_path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            blockers.append(f"base_config illisible: {exc}")
    zones = base_config.get("zones") if isinstance(base_config, Mapping) else None
    zone_config = zones.get(code) if isinstance(zones, Mapping) else None
    if not isinstance(zone_config, Mapping):
        blockers.append(f"base_config ne definit pas zones.{code}")
        zone_config = {}
    _expect_equal(
        str(zone_config.get("timezone", "")),
        market.timezone,
        label="timezone base_config",
        blockers=blockers,
    )
    target = zone_config.get("target")
    configured_target = target.get("series") if isinstance(target, Mapping) else None
    _expect_equal(
        configured_target,
        target_series,
        label="serie cible base_config",
        blockers=blockers,
    )

    required_covariates = raw.get("required_covariates")
    if not isinstance(required_covariates, Sequence) or isinstance(
        required_covariates, (str, bytes)
    ) or not required_covariates:
        blockers.append("required_covariates doit etre une liste non vide")
        required_covariates = []
    covariates = zone_config.get("covariates")
    covariates = covariates if isinstance(covariates, Mapping) else {}
    data = base_config.get("data") if isinstance(base_config, Mapping) else {}
    data = data if isinstance(data, Mapping) else {}
    pit_files = data.get("pit_files")
    pit_files = pit_files if isinstance(pit_files, Mapping) else {}
    pit_root = _resolve(data.get("pit_vintage_dir", "data/pit/vintages"), base=config_dir)
    for alias_value in required_covariates:
        alias = str(alias_value).strip().lower()
        definition = covariates.get(alias)
        if not isinstance(definition, Mapping) or not bool(definition.get("enabled", True)):
            blockers.append(f"covariable requise inactive/absente: {alias}")
            continue
        if definition.get("source") == "pit_parquet":
            pit_file = definition.get("pit_file") or pit_files.get(alias)
            pit_path = _resolve(pit_file, base=pit_root or config_dir)
            if pit_path is None or not pit_path.is_file():
                blockers.append(f"PIT local absent pour {alias}: {pit_path}")
            else:
                checks.append(f"pit:{alias}:present")

    base_dir = base_config_path.parent if base_config_path is not None else config_dir
    cache_dir = _resolve(data.get("cache_dir", "data/cache"), base=base_dir)
    target_cache = cache_dir / code.lower() if cache_dir is not None else None
    if target_cache is None or not target_cache.is_dir() or not any(
        target_cache.glob("target__*.csv.gz")
    ):
        blockers.append(f"cache cible local absent pour {code}: {target_cache}")
    else:
        checks.append("target_cache:present")

    frozen_manifest = (
        _read_json(
            frozen_run / "run_manifest.json",
            label="manifest autonome",
            blockers=blockers,
        )
        if frozen_run is not None and frozen_run.is_dir()
        else {}
    )
    benchmark_manifest = (
        _read_json(
            benchmark_run / "run_manifest.json",
            label="manifest benchmark",
            blockers=blockers,
        )
        if benchmark_run is not None and benchmark_run.is_dir()
        else {}
    )
    for label, manifest in (
        ("manifest autonome", frozen_manifest),
        ("manifest benchmark", benchmark_manifest),
    ):
        if manifest:
            _expect_equal(manifest.get("zone"), code, label=f"{label}.zone", blockers=blockers)
            _expect_equal(
                manifest.get("timezone"),
                market.timezone,
                label=f"{label}.timezone",
                blockers=blockers,
            )
    if benchmark_manifest and storm_series:
        comparators = benchmark_manifest.get("evaluation_only_comparators") or []
        strict_08_series = _text(raw.get("storm_strict_08_series"))
        if storm_series in comparators:
            checks.append("storm_dashboard:benchmark_comparator")
        elif strict_08_series and strict_08_series in comparators:
            checks.append("storm_strict_08:benchmark_comparator")
        else:
            checks.append("storm_comparator:not_in_sealed_benchmark_optional")

    recipe = (
        _read_json(recipe_path, label="recette blend", blockers=blockers)
        if recipe_path is not None
        else {}
    )
    dependency = (
        _read_json(dependency_path, label="dependance primaire", blockers=blockers)
        if dependency_path is not None
        else {}
    )
    if recipe:
        _expect_equal(recipe.get("zone"), code, label="recette.zone", blockers=blockers)
        _expect_equal(
            recipe.get("timezone"),
            market.timezone,
            label="recette.timezone",
            blockers=blockers,
        )
        recipe_mode = _text(
            recipe.get("prediction_mode", recipe.get("recipe_mode"))
        )
        if code == "FR" and not recipe_mode and isinstance(
            recipe.get("external_expert"), Mapping
        ):
            recipe_mode = "mkonline_blend"
        _expect_equal(
            recipe_mode,
            registry_prediction_mode,
            label="recette.prediction_mode",
            blockers=blockers,
        )
        recipe_mkonline_enabled = recipe.get("mkonline_enabled")
        if code == "FR" and recipe_mkonline_enabled is None:
            recipe_mkonline_enabled = isinstance(
                recipe.get("external_expert"), Mapping
            )
        if recipe_mkonline_enabled is not raw.get("mkonline_enabled"):
            blockers.append("recette.mkonline_enabled differe du registre")
        expert = recipe.get("external_expert")
        observed_primary = expert.get("series") if isinstance(expert, Mapping) else None
        if registry_prediction_mode == "mkonline_blend":
            _expect_equal(
                observed_primary,
                primary_series,
                label="recette.external_expert.series",
                blockers=blockers,
            )
        elif expert not in (None, {}):
            blockers.append(
                "recette.external_expert doit etre null pour autonomous_only"
            )
    if dependency:
        _expect_equal(
            dependency.get("terminal_series"),
            primary_series,
            label="dependance.terminal_series",
            blockers=blockers,
        )
        if dependency.get("storm_token_found") is not False:
            blockers.append("dependance primaire non certifiee anti-Storm")

    storm_path = _resolve(raw.get("storm_pit_path"), base=root)
    if storm_path is not None and storm_path.is_file():
        checks.append("storm_pit:present")
    else:
        checks.append("storm_strict_08:pit_unavailable_optional")

    # Both manifests are mandatory because the underlying runners validate
    # their checksums before forecasting.  Preflight catches missing bundles
    # before a model or a network refresh is started.
    for label, directory in (
        ("autonome", frozen_run),
        ("benchmark", benchmark_run),
    ):
        checksum = directory / "artifact_checksums.json" if directory else None
        if checksum is None or not checksum.is_file():
            blockers.append(f"manifest de checksums {label} absent: {checksum}")
        else:
            checks.append(f"checksums:{label}:present")

    blockers = list(dict.fromkeys(blockers))
    return ZoneBundleAudit(
        zone=code,
        timezone=market.timezone,
        forecast_origin_timezone=forecast_origin_timezone,
        ready=not blockers,
        enabled=enabled,
        production_ready=production_ready,
        checks=tuple(checks),
        blockers=tuple(blockers),
        runner=runner,
        live_config=live_config_path,
    )


def strict_contract_preflight(
    audit: ZoneBundleAudit,
    *,
    registry_path: str | Path,
) -> ZoneBundleAudit:
    """Attach the immutable non-FR model-contract check to a shallow audit.

    The local import avoids a module cycle: ``multizone_contract`` consumes
    market identities from this module.  A contract failure is converted into
    a blocker so the dispatcher and UI expose it before starting a child
    process, loading a model, or touching the network.
    """

    if audit.zone == "FR" or not audit.ready:
        return audit
    if audit.live_config is None:
        return replace(
            audit,
            ready=False,
            blockers=(*audit.blockers, "contrat strict: live_config absent"),
        )
    try:
        from chronos2_hourly.multizone_contract import load_zone_model_contract

        load_zone_model_contract(
            audit.live_config,
            Path(registry_path).expanduser().resolve(),
            strict=True,
        )
    except (OSError, ValueError, ZoneBundleError) as exc:
        return replace(
            audit,
            ready=False,
            blockers=(*audit.blockers, f"contrat strict invalide: {exc}"),
        )
    return replace(
        audit,
        checks=(*audit.checks, "zone_model_contract:verified"),
    )


def build_zone_training_config(
    template: Mapping[str, Any],
    *,
    zone: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a zone-specific autonomous-training config without mutating FR.

    The statistical recipe (model, lags and estimators) is copied byte-for-byte
    at the value level.  Only market identity, target/covariate routing,
    calendar timezone and output labels are changed.
    """

    code = canonical_zone(zone)
    market = MARKET_ZONES[code]
    configured_timezone = _text(
        contract.get("delivery_timezone", contract.get("timezone", market.timezone))
    )
    if configured_timezone != market.timezone:
        raise ZoneBundleError(
            f"{code}: timezone {configured_timezone!r} != {market.timezone!r}"
        )
    bidding_zone = _text(contract.get("bidding_zone"))
    if not bidding_zone:
        raise ZoneBundleError(f"{code}: bidding_zone explicite requis")
    if market.requires_explicit_bidding_zone and bidding_zone.upper() == "IT":
        raise ZoneBundleError("IT: choisissez une zone de prix explicite")
    target_status = _text(contract.get("target_status"))
    if target_status != "audited_dst_strict":
        raise ZoneBundleError(
            f"{code}: target_status={target_status or 'absent'}; "
            "une cible DST-stricte auditee est requise"
        )
    price_unit = _text(contract.get("price_unit"))
    if price_unit != "EUR/MWh":
        raise ZoneBundleError(
            f"{code}: price_unit={price_unit or 'absent'}; EUR/MWh requis"
        )
    target_series = _text(contract.get("target_series"))
    if not target_series:
        raise ZoneBundleError(f"{code}: target_series auditee requise")
    required = contract.get("required_covariates")
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes)) or not required:
        raise ZoneBundleError(f"{code}: required_covariates non vide requis")

    source_zones = _mapping(template.get("zones"), name="template.zones")
    source_zone = source_zones.get("FR")
    if not isinstance(source_zone, Mapping):
        raise ZoneBundleError("Le template doit contenir zones.FR")
    source_covariates = _mapping(
        source_zone.get("covariates"),
        name="template.zones.FR.covariates",
    )
    overrides = contract.get("covariates", {})
    overrides = _mapping(overrides, name=f"{code}.covariates")
    selected_covariates: dict[str, Any] = {}
    for value in required:
        alias = str(value).strip().lower()
        source_definition = source_covariates.get(alias)
        override_definition = overrides.get(alias, {})
        if not isinstance(source_definition, Mapping):
            raise ZoneBundleError(
                f"{code}: definition de covariable absente pour {alias}"
            )
        if not isinstance(override_definition, Mapping):
            raise ZoneBundleError(
                f"{code}: override de covariable invalide pour {alias}"
            )
        selected_covariates[alias] = copy.deepcopy(dict(source_definition))
        selected_covariates[alias].update(copy.deepcopy(dict(override_definition)))
        selected_covariates[alias]["enabled"] = True

    result = copy.deepcopy(dict(template))
    target = {
        "series": target_series,
        "naive_timezone": str(contract.get("target_naive_timezone", "UTC")),
        "description": (
            f"Prix Day-Ahead {bidding_zone} horaire (timeline UTC)"
        ),
    }
    result["zones"] = {
        code: {
            "enabled": True,
            "timezone": market.timezone,
            "include_calendar": True,
            "target": target,
            "covariates": selected_covariates,
        }
    }
    hourly = result.setdefault("hourly", {})
    feature_engineering = hourly.setdefault("feature_engineering", {})
    feature_engineering["timezone"] = market.timezone
    residual = hourly.setdefault("residual_correction", {})
    builder = residual.setdefault("feature_builder", {})
    builder["timezone"] = market.timezone
    builder["rich_calendar_primary_country"] = market.holiday_country

    output = result.setdefault("output", {})
    output["directory"] = f"runs/chronos2_hourly_{code.lower()}_residual_v1"
    report = result.setdefault("report", {})
    report["filename"] = f"chronos2_hourly_{code.lower()}_residual_v1.html"
    report["title"] = f"Chronos-2 horaire {code} - correcteur residuel v1"
    return result


def build_zone_runner_command(
    audit: ZoneBundleAudit,
    *,
    data_as_of: str | None = None,
    delivery_day: str | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    threads: int | None = None,
    workers: int | None = None,
    local_files_only: bool = False,
    pit_replay: bool = False,
    residual_load_source: str = "saturn",
    residual_load_bundle_manifest: str | Path | None = None,
) -> list[str]:
    audit.require_ready()
    assert audit.runner is not None and audit.live_config is not None
    command = [sys.executable, str(audit.runner), "--config", str(audit.live_config)]
    for flag, value in (
        ("--data-as-of", data_as_of),
        ("--delivery-day", delivery_day),
        ("--output-dir", output_dir),
        ("--device", device),
        ("--threads", threads),
        ("--workers", workers),
    ):
        if value not in (None, ""):
            command.extend([flag, str(value)])
    if local_files_only:
        command.append("--local-files-only")
    if pit_replay:
        command.append("--pit-replay")
    normalized_residual_load_source = str(residual_load_source).strip().lower()
    if normalized_residual_load_source not in {"saturn", "chronos2"}:
        raise ZoneBundleError(
            "residual_load_source doit valoir 'saturn' ou 'chronos2'"
        )
    if normalized_residual_load_source == "chronos2":
        if residual_load_bundle_manifest in (None, ""):
            raise ZoneBundleError(
                "residual_load_bundle_manifest est obligatoire lorsque "
                "residual_load_source='chronos2'"
            )
        command.extend(
            [
                "--residual-load-source",
                "chronos2",
                "--residual-load-bundle-manifest",
                str(residual_load_bundle_manifest),
            ]
        )
    return command
