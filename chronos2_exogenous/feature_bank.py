"""Strict point-in-time exogenous feature bank for the Chronos-2 POC.

The module joins already materialised hourly Parquet artefacts.  It performs
no network access, no target lookup and no interpolation.  Every delivery hour
is checked against its civil ``D-1 08:00`` cutoff before a value can enter the
bank.  Missing hours remain missing and are exposed through explicit quality
features and the audit payload.

The bank also owns consumer routing.  This lets an ablation send a family to
Chronos-2 without silently giving the same signal to the residual corrector or
the Kalman overlay.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .pit_evidence import (
    HISTORICAL_EVIDENCE_KIND,
    HistoricalPITEvidenceError,
    verify_historical_asof_evidence,
)


SCHEMA_VERSION = 1
DEFAULT_TIMEZONE = "Europe/Paris"
DEFAULT_CUTOFF_TIME = "08:00"
SUPPORTED_CONSUMERS = ("chronos", "residual", "kalman")
SUPPORTED_TRANSFORMS = frozenset({"identity", "flowbased_compact"})
SUPPORTED_SOURCE_EVIDENCE_KINDS = frozenset(
    {
        "versioned_revision_history",
        "attested_historical_asof_archive",
        "prospective_capture",
    }
)
# Backward-compatible name retained for callers which imported the initial
# two-scope implementation.
SUPPORTED_PRODUCTION_EVIDENCE_KINDS = SUPPORTED_SOURCE_EVIDENCE_KINDS

ABLATION_PACKS: Mapping[str, tuple[str, ...]] = {
    "residual_only": ("residual_load",),
    "residual_weather": ("residual_load", "weather"),
    "residual_fuel": ("residual_load", "fuel"),
    "residual_flowbased": ("residual_load", "flowbased"),
    "full": ("residual_load", "weather", "fuel", "flowbased"),
}

RESIDUAL_LOAD_COLUMNS: tuple[str, ...] = (
    "fr_residual_load_fcst",
    "de_residual_load_fcst",
    "be_residual_load_fcst",
    "nl_residual_load_fcst",
    "es_residual_load_fcst",
)

FUEL_COLUMNS: tuple[str, ...] = (
    "ttf_m1_eur_mwh_th",
    "eua_first_dec_eur_tco2",
    "ttf_change_1d",
    "ttf_change_5d",
    "eua_change_1d",
    "eua_change_5d",
    "fuel_volatility_20d",
    "ccgt_marginal_cost_eur_mwh",
)

FLOWBASED_COMPACT_COLUMNS: tuple[str, ...] = (
    "flowbased_availability",
    "flowbased_hour_imputed",
    "flowbased_cnec_count",
    "flowbased_external_ram_p10_gw",
    "flowbased_ram_p10_gw",
    "flowbased_ram_headroom_p10_to_median_gw",
    "flowbased_low_ram_share",
    "flowbased_ram_to_fmax_p05",
    "flowbased_fr_neighbor_ptdf_spread_p90",
    "flowbased_fr_neighbor_ram_stress_p95_per_gw",
    "flowbased_core_ram_stress_p95_per_gw",
    "flowbased_stress_hhi",
)

CALENDAR_COLUMNS: tuple[str, ...] = (
    "known_hour_sin",
    "known_hour_cos",
    "known_dow_sin",
    "known_dow_cos",
    "known_doy_sin",
    "known_doy_cos",
    "known_is_weekend",
)

_FLOWBASED_REQUIRED_COLUMNS = frozenset(
    {
        "flowbased_cnec_mtu_availability",
        "flowbased_hour_imputed",
        "flowbased_cnec_count",
        "flowbased_external_ram_p10_mw",
        "flowbased_ram_p10_mw",
        "flowbased_ram_median_mw",
        "flowbased_ram_below_1000_share",
        "flowbased_ram_to_fmax_p05",
        "flowbased_fr_de_ptdf_spread_p90",
        "flowbased_fr_be_ptdf_spread_p90",
        "flowbased_fr_nl_ptdf_spread_p90",
        "flowbased_fr_neighbor_ram_stress_p95_per_gw",
        "flowbased_core_ram_stress_p95_per_gw",
        "flowbased_stress_hhi",
    }
)

_SAFE_NAME = re.compile(r"^[a-z][a-z0-9_]{1,127}$")
_FORBIDDEN_TOKENS = ("storm", "mkonline", "oracle", "observed", "actual")


class ExogenousBankError(ValueError):
    """Raised when an exogenous input is non-causal or structurally unsafe."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_SOURCE_AUDIT_PARQUET_HASH_FIELDS = (
    "output_sha256",
    "parquet_sha256",
    "sha256",
)


def _strict_nonnegative_integer(value: object, *, field: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ExogenousBankError(
            f"{source}: sidecar audit invalide, {field} doit etre un entier."
        )
    number = int(value)
    if number < 0:
        raise ExogenousBankError(
            f"{source}: sidecar audit invalide, {field} doit etre positif ou nul."
        )
    return number


def _historical_source_contract_sha256(
    source: "ParquetFeatureSource",
    *,
    forecast_origin_timezone: str,
    source_cutoff_timezone: str,
    cutoff_time: str,
) -> str:
    """Bind approved bytes to their exact model-facing interpretation."""

    contract = {
        "schema_version": 1,
        "source_name": source.name,
        "family": source.family,
        "value_columns": {
            str(key): str(value)
            for key, value in sorted(source.value_columns.items())
        },
        "consumer_route": list(source.route.consumers),
        "timestamp_column": source.timestamp_column,
        "cutoff_column": source.cutoff_column,
        "forecast_origin_timezone": forecast_origin_timezone,
        "source_cutoff_timezone": source_cutoff_timezone,
        "cutoff_time": cutoff_time,
        "information_time_columns": list(source.information_time_columns),
        "age_column": source.age_column,
        "eligibility_column": source.eligibility_column,
        "operational_eligibility_column": source.operational_eligibility_column,
        "stage_column": source.stage_column,
        "allowed_stages": list(source.allowed_stages),
        "known_future": source.known_future,
        "transform": source.transform,
        "production_evidence_kind": source.production_evidence_kind,
    }
    encoded = json.dumps(
        contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_source_audit(
    source: "ParquetFeatureSource",
    *,
    parquet_path: Path,
    parquet_sha256: str,
    forecast_origin_timezone: str,
    source_cutoff_timezone: str,
    cutoff_time: str,
    expected: pd.DatetimeIndex,
    expected_cutoffs: pd.DatetimeIndex,
) -> dict[str, Any] | None:
    """Verify an optional materializer sidecar and bind it to exact Parquet bytes.

    A declaration in Python is never sufficient evidence for production.  A
    production claim additionally needs an independently written sidecar whose
    Parquet hash, cutoff contract and prospective-capture fields all validate.
    Historical ``revision_date=...`` reconstructions remain useful research PIT
    evidence, but are deliberately not promoted to operational capture proof.
    """

    if source.audit_path is None:
        return None
    audit_path = Path(source.audit_path).expanduser().resolve()
    if not audit_path.is_file():
        raise ExogenousBankError(
            f"{source.name}: sidecar audit absent: {audit_path}."
        )
    try:
        audit_bytes = audit_path.read_bytes()
        payload = json.loads(audit_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExogenousBankError(
            f"{source.name}: sidecar audit illisible: {audit_path}."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ExogenousBankError(
            f"{source.name}: le sidecar audit doit etre un objet JSON."
        )

    declared_hashes = {
        field: payload[field]
        for field in _SOURCE_AUDIT_PARQUET_HASH_FIELDS
        if field in payload
    }
    if not declared_hashes:
        raise ExogenousBankError(
            f"{source.name}: sidecar audit sans hash du Parquet."
        )
    for field, value in declared_hashes.items():
        declared = str(value).strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", declared) is None:
            raise ExogenousBankError(
                f"{source.name}: sidecar audit, {field} n'est pas un SHA-256."
            )
        if declared != parquet_sha256:
            raise ExogenousBankError(
                f"{source.name}: sidecar audit non lie au Parquet ({field} divergent)."
            )

    causal_counts: dict[str, int] = {}
    for field in ("causality_violations", "pit_violations"):
        if field not in payload:
            continue
        count = _strict_nonnegative_integer(
            payload[field], field=field, source=source.name
        )
        causal_counts[field] = count
        if count:
            raise ExogenousBankError(
                f"{source.name}: sidecar audit declare {count} {field}."
            )
    if "cutoff_time" in payload and str(payload["cutoff_time"]).strip() != cutoff_time:
        raise ExogenousBankError(
            f"{source.name}: cutoff_time du sidecar incompatible avec {cutoff_time}."
        )
    if (
        "cutoff_timezone" in payload
        and str(payload["cutoff_timezone"]).strip() != source_cutoff_timezone
    ):
        raise ExogenousBankError(
            f"{source.name}: cutoff_timezone du sidecar incompatible avec le "
            f"fuseau source {source_cutoff_timezone}."
        )

    operational_violations: int | None = None
    if "operational_capture_violations" in payload:
        operational_violations = _strict_nonnegative_integer(
            payload["operational_capture_violations"],
            field="operational_capture_violations",
            source=source.name,
        )
    declared_kind = payload.get("production_evidence_kind")
    declared_ready = payload.get("production_pit_evidence")
    prospective_sidecar = bool(
        source.production_evidence_kind == "prospective_capture"
        and declared_kind == "prospective_capture"
        and declared_ready is True
        and operational_violations == 0
        and bool(causal_counts)
    )
    historical_evidence: dict[str, Any] | None = None
    historical_source_contract_sha256: str | None = None
    if source.production_evidence_kind == "attested_historical_asof_archive":
        if (
            declared_ready is True
            or declared_kind == "prospective_capture"
            or payload.get("prospective_capture_evidence") is True
        ):
            raise ExogenousBankError(
                f"{source.name}: une preuve historique hors ligne ne peut pas "
                "revendiquer une capture prospective ou production."
            )
        if payload.get("historical_backtest_pit_evidence") is not True:
            raise ExogenousBankError(
                f"{source.name}: le sidecar doit declarer "
                "historical_backtest_pit_evidence=true."
            )
        if payload.get("historical_evidence_kind") != HISTORICAL_EVIDENCE_KIND:
            raise ExogenousBankError(
                f"{source.name}: historical_evidence_kind absent ou invalide."
            )
        declared_manifest_hash = str(
            payload.get("historical_evidence_manifest_sha256", "")
        ).strip().lower()
        configured_manifest_hash = str(
            source.historical_evidence_manifest_sha256
        ).strip().lower()
        if declared_manifest_hash != configured_manifest_hash:
            raise ExogenousBankError(
                f"{source.name}: SHA du manifeste historique divergent du sidecar."
            )
        local_days = pd.DatetimeIndex(expected).tz_convert(
            source_cutoff_timezone
        ).date
        day_cutoffs: dict[str, pd.Timestamp] = {}
        for day, cutoff in zip(local_days, expected_cutoffs, strict=True):
            token = day.isoformat()
            timestamp = pd.Timestamp(cutoff)
            previous = day_cutoffs.setdefault(token, timestamp)
            if previous != timestamp:
                raise ExogenousBankError(
                    f"{source.name}: plusieurs cutoffs pour {token}."
                )
        try:
            historical_source_contract_sha256 = (
                _historical_source_contract_sha256(
                    source,
                    forecast_origin_timezone=forecast_origin_timezone,
                    source_cutoff_timezone=source_cutoff_timezone,
                    cutoff_time=cutoff_time,
                )
            )
            historical_evidence = verify_historical_asof_evidence(
                source.historical_evidence_manifest_path,
                expected_manifest_sha256=configured_manifest_hash,
                source_name=source.name,
                parquet_sha256=parquet_sha256,
                source_contract_sha256=(
                    historical_source_contract_sha256
                ),
                required_days_and_cutoffs=day_cutoffs,
            )
        except HistoricalPITEvidenceError as exc:
            raise ExogenousBankError(
                f"{source.name}: preuve PIT historique refusee: {exc}"
            ) from exc
    classification = (
        "prospective_capture"
        if prospective_sidecar
        else (
            "attested_historical_asof_archive"
            if historical_evidence is not None
            else (
                "research_versioned_history"
                if source.production_evidence_kind == "versioned_revision_history"
                else "research_only"
            )
        )
    )
    return {
        "path": str(audit_path),
        "sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "parquet_path": str(parquet_path),
        "parquet_sha256": parquet_sha256,
        "parquet_hash_fields": sorted(declared_hashes),
        "forecast_origin_timezone": forecast_origin_timezone,
        "source_cutoff_timezone": source_cutoff_timezone,
        "declared_cutoff_timezone": payload.get("cutoff_timezone"),
        "declared_delivery_timezone": payload.get("timezone"),
        "causal_violation_counts": causal_counts,
        "declared_production_evidence_kind": declared_kind,
        "declared_production_pit_evidence": declared_ready,
        "operational_capture_violations": operational_violations,
        "prospective_capture_verified": prospective_sidecar,
        "historical_backtest_verified": historical_evidence is not None,
        "historical_evidence": historical_evidence,
        "historical_source_contract_sha256": (
            historical_source_contract_sha256
        ),
        "classification": classification,
    }


@dataclass(frozen=True)
class ConsumerRoute:
    """Explicit consumers of one exogenous source family."""

    consumers: tuple[str, ...] = ("chronos",)

    def validate(self) -> None:
        if not self.consumers or len(self.consumers) != len(set(self.consumers)):
            raise ExogenousBankError("Une route doit contenir des consommateurs uniques.")
        unknown = sorted(set(self.consumers).difference(SUPPORTED_CONSUMERS))
        if unknown:
            raise ExogenousBankError(f"Consommateurs inconnus: {unknown}.")

    def includes(self, consumer: str) -> bool:
        code = str(consumer).strip().casefold()
        if code not in SUPPORTED_CONSUMERS:
            raise ExogenousBankError(f"Consommateur inconnu: {consumer!r}.")
        return code in self.consumers


@dataclass(frozen=True)
class ParquetFeatureSource:
    """Declarative schema of one already-materialised PIT Parquet source."""

    name: str
    family: str
    path: Path
    value_columns: Mapping[str, str]
    audit_path: Path | None = None
    route: ConsumerRoute = ConsumerRoute()
    timestamp_column: str = "value_time_utc"
    cutoff_column: str | None = None
    # The shared experiment can use a Europe/Paris forecast-origin contract
    # while a zone-local materializer records its civil cutoff in another
    # (equivalent or non-equivalent) IANA timezone.  Keep both identities
    # explicit and compare causal instants in UTC.
    cutoff_timezone: str | None = None
    information_time_columns: tuple[str, ...] = ()
    age_column: str | None = None
    eligibility_column: str | None = None
    operational_eligibility_column: str | None = None
    stage_column: str | None = None
    allowed_stages: tuple[str, ...] = ()
    known_future: bool = True
    transform: str = "identity"
    production_evidence_kind: str | None = None
    historical_evidence_manifest_path: Path | None = None
    historical_evidence_manifest_sha256: str | None = None

    def validate(self) -> None:
        for label, value in (("source", self.name), ("famille", self.family)):
            if not _SAFE_NAME.fullmatch(str(value)):
                raise ExogenousBankError(f"Nom de {label} invalide: {value!r}.")
            lowered = str(value).casefold()
            if any(token in lowered for token in _FORBIDDEN_TOKENS):
                raise ExogenousBankError(f"Nom de {label} interdit: {value!r}.")
        self.route.validate()
        if self.transform not in SUPPORTED_TRANSFORMS:
            raise ExogenousBankError(
                f"{self.name}: transformation inconnue {self.transform!r}."
            )
        if (
            self.production_evidence_kind is not None
            and self.production_evidence_kind not in SUPPORTED_PRODUCTION_EVIDENCE_KINDS
        ):
            raise ExogenousBankError(
                f"{self.name}: preuve de production inconnue "
                f"{self.production_evidence_kind!r}."
            )
        if self.production_evidence_kind is not None and self.audit_path is None:
            raise ExogenousBankError(
                f"{self.name}: production_evidence_kind exige un sidecar audit_path."
            )
        if self.production_evidence_kind is not None and (
            self.cutoff_column is None or not self.information_time_columns
        ):
            raise ExogenousBankError(
                f"{self.name}: une preuve de production exige cutoff et "
                "information_time_columns."
            )
        historical_fields = (
            self.historical_evidence_manifest_path,
            self.historical_evidence_manifest_sha256,
        )
        if any(value is not None for value in historical_fields) and not all(
            value is not None for value in historical_fields
        ):
            raise ExogenousBankError(
                f"{self.name}: chemin et SHA du manifeste historique doivent etre "
                "declares ensemble."
            )
        if self.production_evidence_kind == "attested_historical_asof_archive":
            if not all(value is not None for value in historical_fields):
                raise ExogenousBankError(
                    f"{self.name}: la preuve historique attestee exige un "
                    "manifeste SHA-epingle."
                )
            digest = str(self.historical_evidence_manifest_sha256).strip().lower()
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ExogenousBankError(
                    f"{self.name}: SHA du manifeste historique invalide."
                )
        elif any(value is not None for value in historical_fields):
            raise ExogenousBankError(
                f"{self.name}: un manifeste historique n'est admis qu'avec "
                "attested_historical_asof_archive."
            )
        if self.transform == "identity" and not self.value_columns:
            raise ExogenousBankError(f"{self.name}: value_columns est vide.")
        for output, source in self.value_columns.items():
            if not _SAFE_NAME.fullmatch(str(output)):
                raise ExogenousBankError(
                    f"{self.name}: alias de feature invalide {output!r}."
                )
            if not isinstance(source, str) or not source.strip():
                raise ExogenousBankError(
                    f"{self.name}: colonne source invalide pour {output!r}."
                )
            lowered = str(output).casefold()
            if any(token in lowered for token in _FORBIDDEN_TOKENS):
                raise ExogenousBankError(
                    f"{self.name}: feature interdite {output!r}."
                )
        if self.stage_column is None and self.allowed_stages:
            raise ExogenousBankError(
                f"{self.name}: allowed_stages exige stage_column."
            )
        if self.stage_column is not None and not self.allowed_stages:
            raise ExogenousBankError(
                f"{self.name}: stage_column exige allowed_stages."
            )


@dataclass(frozen=True)
class ExogenousBank:
    """Joined features, column catalogue and JSON-serialisable audit."""

    frame: pd.DataFrame
    catalogue: pd.DataFrame
    audit: Mapping[str, Any]

    def columns_for(self, consumer: str, *, include_quality: bool = True) -> tuple[str, ...]:
        code = str(consumer).strip().casefold()
        if code not in SUPPORTED_CONSUMERS:
            raise ExogenousBankError(f"Consommateur inconnu: {consumer!r}.")
        selected = self.catalogue.loc[self.catalogue[code].astype(bool)]
        if not include_quality:
            selected = selected.loc[selected["role"].eq("value")]
        return tuple(selected["column"].astype(str))

    def for_consumer(
        self,
        consumer: str,
        *,
        include_quality: bool = True,
        copy: bool = True,
    ) -> pd.DataFrame:
        columns = self.columns_for(consumer, include_quality=include_quality)
        result = self.frame.loc[:, list(columns)]
        return result.copy() if copy else result

    def assert_complete(
        self,
        consumer: str,
        *,
        delivery_day: str | pd.Timestamp | None = None,
    ) -> None:
        frame = self.for_consumer(consumer, include_quality=False, copy=False)
        if delivery_day is not None:
            expected = delivery_utc_index(
                delivery_day,
                delivery_day,
                timezone=str(self.audit["timezone"]),
            )
            frame = frame.reindex(expected)
        missing = frame.isna().sum()
        bad = {str(name): int(count) for name, count in missing.items() if count}
        if bad:
            raise ExogenousBankError(
                f"Features {consumer} incompletes: {bad}."
            )


def _naive_day(
    value: str | pd.Timestamp, *, timezone: str = DEFAULT_TIMEZONE
) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is not None:
        result = result.tz_convert(timezone).tz_localize(None)
    return result.normalize()


def delivery_utc_index(
    start_day: str | pd.Timestamp,
    end_day: str | pd.Timestamp,
    *,
    timezone: str = DEFAULT_TIMEZONE,
) -> pd.DatetimeIndex:
    """Return every physical hour in inclusive local civil-day bounds."""

    start = _naive_day(start_day, timezone=timezone)
    end = _naive_day(end_day, timezone=timezone)
    if end < start:
        raise ExogenousBankError("end_day doit etre >= start_day.")
    try:
        lower = start.tz_localize(timezone).tz_convert("UTC")
        upper = (end + pd.Timedelta(days=1)).tz_localize(timezone).tz_convert("UTC")
    except (TypeError, ValueError) as exc:
        raise ExogenousBankError(f"Timezone invalide: {timezone!r}.") from exc
    return pd.date_range(lower, upper, freq="h", inclusive="left", name="value_time_utc")


def cutoff_by_delivery_hour(
    index: pd.DatetimeIndex,
    *,
    timezone: str = DEFAULT_TIMEZONE,
    cutoff_time: str = DEFAULT_CUTOFF_TIME,
) -> pd.DatetimeIndex:
    """Map each delivery hour to its exact civil ``D-1 cutoff_time``."""

    timestamps = pd.DatetimeIndex(index)
    if timestamps.tz is None:
        raise ExogenousBankError("La timeline de livraison doit etre timezone-aware.")
    try:
        hour, minute = (int(token) for token in cutoff_time.split(":"))
    except (TypeError, ValueError) as exc:
        raise ExogenousBankError(f"Heure de cutoff invalide: {cutoff_time!r}.") from exc
    if hour not in range(24) or minute not in range(60):
        raise ExogenousBankError(f"Heure de cutoff invalide: {cutoff_time!r}.")
    local_days = timestamps.tz_convert(timezone).date
    unique_days = tuple(dict.fromkeys(local_days))
    by_day = {
        day: (
            pd.Timestamp(day)
            - pd.Timedelta(days=1)
            + pd.Timedelta(hours=hour, minutes=minute)
        ).tz_localize(timezone).tz_convert("UTC")
        for day in unique_days
    }
    return pd.DatetimeIndex([by_day[day] for day in local_days])


def deterministic_calendar_frame(
    index: pd.DatetimeIndex, *, timezone: str = DEFAULT_TIMEZONE
) -> pd.DataFrame:
    """Reproduce the calendar covariates used by the incumbent Chronos runner."""

    utc = pd.DatetimeIndex(index)
    if utc.tz is None:
        raise ExogenousBankError("La timeline calendrier doit etre timezone-aware.")
    local = utc.tz_convert(timezone)
    hour = local.hour.to_numpy(dtype=np.float64)
    dow = local.dayofweek.to_numpy(dtype=np.float64)
    doy = local.dayofyear.to_numpy(dtype=np.float64)
    return pd.DataFrame(
        {
            "known_hour_sin": np.sin(2.0 * np.pi * hour / 24.0),
            "known_hour_cos": np.cos(2.0 * np.pi * hour / 24.0),
            "known_dow_sin": np.sin(2.0 * np.pi * dow / 7.0),
            "known_dow_cos": np.cos(2.0 * np.pi * dow / 7.0),
            "known_doy_sin": np.sin(2.0 * np.pi * doy / 365.25),
            "known_doy_cos": np.cos(2.0 * np.pi * doy / 365.25),
            "known_is_weekend": (dow >= 5.0).astype(np.float32),
        },
        index=utc,
        dtype=np.float32,
    ).loc[:, list(CALENDAR_COLUMNS)]


def compress_flowbased_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministically compress stable CNEC/RAM aggregates.

    No PCA is used: every output keeps a stable economic interpretation across
    refits and changing CNEC identifiers.  MW variables are converted to GW to
    keep their scale close to the residual-load inputs.
    """

    missing = sorted(_FLOWBASED_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ExogenousBankError(f"Flow-based incomplet pour compression: {missing}.")
    numeric = frame.loc[:, sorted(_FLOWBASED_REQUIRED_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    output = pd.DataFrame(index=frame.index)
    output["flowbased_availability"] = numeric[
        "flowbased_cnec_mtu_availability"
    ].clip(0.0, 1.0)
    output["flowbased_hour_imputed"] = numeric["flowbased_hour_imputed"].clip(
        0.0, 1.0
    )
    output["flowbased_cnec_count"] = numeric["flowbased_cnec_count"]
    output["flowbased_external_ram_p10_gw"] = (
        numeric["flowbased_external_ram_p10_mw"] / 1000.0
    )
    output["flowbased_ram_p10_gw"] = numeric["flowbased_ram_p10_mw"] / 1000.0
    output["flowbased_ram_headroom_p10_to_median_gw"] = (
        numeric["flowbased_ram_median_mw"] - numeric["flowbased_ram_p10_mw"]
    ) / 1000.0
    output["flowbased_low_ram_share"] = numeric[
        "flowbased_ram_below_1000_share"
    ].clip(0.0, 1.0)
    output["flowbased_ram_to_fmax_p05"] = numeric["flowbased_ram_to_fmax_p05"]
    output["flowbased_fr_neighbor_ptdf_spread_p90"] = numeric[
        [
            "flowbased_fr_de_ptdf_spread_p90",
            "flowbased_fr_be_ptdf_spread_p90",
            "flowbased_fr_nl_ptdf_spread_p90",
        ]
    ].max(axis=1, skipna=False)
    output["flowbased_fr_neighbor_ram_stress_p95_per_gw"] = numeric[
        "flowbased_fr_neighbor_ram_stress_p95_per_gw"
    ]
    output["flowbased_core_ram_stress_p95_per_gw"] = numeric[
        "flowbased_core_ram_stress_p95_per_gw"
    ]
    output["flowbased_stress_hhi"] = numeric["flowbased_stress_hhi"].clip(
        0.0, 1.0
    )
    return output.loc[:, list(FLOWBASED_COMPACT_COLUMNS)]


def _parse_utc_column(frame: pd.DataFrame, column: str, *, source: str) -> pd.Series:
    try:
        # JAO serialises some timestamps with fractional seconds and others
        # without them.  ``mixed`` is strict per value while accepting both
        # valid ISO-8601 representations.
        parsed = pd.to_datetime(
            frame[column], utc=True, errors="raise", format="mixed"
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise ExogenousBankError(
            f"{source}: colonne temporelle invalide ou absente: {column}."
        ) from exc
    if parsed.isna().any():
        raise ExogenousBankError(f"{source}: {column} contient NaT.")
    return parsed


def _required_columns(source: ParquetFeatureSource) -> tuple[str, ...]:
    columns = {
        source.timestamp_column,
        *source.value_columns.values(),
        *source.information_time_columns,
    }
    for column in (
        source.cutoff_column,
        source.age_column,
        source.eligibility_column,
        source.operational_eligibility_column,
        source.stage_column,
    ):
        if column:
            columns.add(column)
    if source.transform == "flowbased_compact":
        columns.update(_FLOWBASED_REQUIRED_COLUMNS)
    return tuple(sorted(columns))


def _load_one_source(
    source: ParquetFeatureSource,
    *,
    expected: pd.DatetimeIndex,
    expected_cutoffs: pd.DatetimeIndex,
    timezone: str,
    cutoff_time: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    source.validate()
    path = Path(source.path).expanduser().resolve()
    if not path.is_file():
        raise ExogenousBankError(f"{source.name}: Parquet absent: {path}.")
    parquet_sha256 = _sha256_file(path)
    source_cutoff_timezone = source.cutoff_timezone or timezone
    source_expected_cutoffs = (
        expected_cutoffs
        if source_cutoff_timezone == timezone
        else cutoff_by_delivery_hour(
            expected,
            timezone=source_cutoff_timezone,
            cutoff_time=cutoff_time,
        )
    )
    source_sidecar = _load_source_audit(
        source,
        parquet_path=path,
        parquet_sha256=parquet_sha256,
        forecast_origin_timezone=timezone,
        source_cutoff_timezone=source_cutoff_timezone,
        cutoff_time=cutoff_time,
        expected=expected,
        expected_cutoffs=source_expected_cutoffs,
    )
    try:
        raw = pd.read_parquet(path, columns=list(_required_columns(source)))
    except Exception as exc:
        raise ExogenousBankError(
            f"{source.name}: Parquet illisible ou schema incomplet: {path}."
        ) from exc
    if (
        source.production_evidence_kind == "attested_historical_asof_archive"
        and _sha256_file(path) != parquet_sha256
    ):
        raise ExogenousBankError(
            f"{source.name}: le Parquet a change pendant sa verification/lecture."
        )
    if raw.empty:
        raise ExogenousBankError(f"{source.name}: Parquet vide.")
    timestamps = _parse_utc_column(raw, source.timestamp_column, source=source.name)
    if timestamps.duplicated().any():
        raise ExogenousBankError(f"{source.name}: heures de livraison dupliquees.")
    raw = raw.copy()
    raw.index = pd.DatetimeIndex(timestamps)
    raw = raw.sort_index().loc[lambda item: item.index.isin(expected)]
    expected_cutoff_series = pd.Series(source_expected_cutoffs, index=expected)
    row_cutoffs = expected_cutoff_series.reindex(raw.index)

    if source.cutoff_column:
        declared = _parse_utc_column(raw, source.cutoff_column, source=source.name)
        if not pd.DatetimeIndex(declared).equals(pd.DatetimeIndex(row_cutoffs)):
            bad = int((declared.reset_index(drop=True) != row_cutoffs.reset_index(drop=True)).sum())
            raise ExogenousBankError(
                f"{source.name}: {bad} cutoff(s) differents de D-1 08:00 civil."
            )
    for column in source.information_time_columns:
        information = _parse_utc_column(raw, column, source=source.name)
        if bool((information.reset_index(drop=True) > row_cutoffs.reset_index(drop=True)).any()):
            raise ExogenousBankError(
                f"{source.name}: {column} contient une information posterieure au cutoff."
            )
    if source.eligibility_column:
        eligible = raw[source.eligibility_column]
        if eligible.isna().any() or not bool(eligible.astype(bool).all()):
            raise ExogenousBankError(
                f"{source.name}: lignes non eligibles PIT dans {source.eligibility_column}."
            )
    if source.stage_column:
        stages = set(raw[source.stage_column].dropna().astype(str))
        invalid = sorted(stages.difference(source.allowed_stages))
        if invalid:
            raise ExogenousBankError(
                f"{source.name}: stages de publication interdits: {invalid}."
            )

    if source.transform == "flowbased_compact":
        values = compress_flowbased_features(raw)
    else:
        values = pd.DataFrame(index=raw.index)
        for output, column in source.value_columns.items():
            values[str(output)] = pd.to_numeric(raw[column], errors="coerce")
    infinite = np.isinf(values.to_numpy(dtype=float))
    if bool(infinite.any()):
        raise ExogenousBankError(f"{source.name}: valeurs infinies interdites.")
    values = values.astype(float).reindex(expected)

    coverage = values.notna().mean(axis=1).astype(float)
    quality = pd.DataFrame(index=expected)
    quality[f"{source.name}__coverage"] = coverage
    quality[f"{source.name}__available"] = coverage.eq(1.0).astype(float)
    if source.age_column:
        age_time = _parse_utc_column(raw, source.age_column, source=source.name)
        if bool((age_time.reset_index(drop=True) > row_cutoffs.reset_index(drop=True)).any()):
            raise ExogenousBankError(
                f"{source.name}: age_column posterieure au cutoff."
            )
        ages = pd.Series(
            (pd.DatetimeIndex(row_cutoffs) - pd.DatetimeIndex(age_time))
            / pd.Timedelta(hours=1),
            index=raw.index,
            dtype=float,
        ).reindex(expected)
        quality[f"{source.name}__age_hours"] = ages
    else:
        quality[f"{source.name}__age_hours"] = np.where(
            coverage.gt(0.0), 0.0, np.nan
        )

    routed = {consumer: source.route.includes(consumer) for consumer in SUPPORTED_CONSUMERS}
    catalogue: list[dict[str, Any]] = []
    for column in values:
        catalogue.append(
            {
                "column": str(column),
                "source": source.name,
                "family": source.family,
                "role": "value",
                "known_future": bool(source.known_future),
                **routed,
            }
        )
    for column in quality:
        catalogue.append(
            {
                "column": str(column),
                "source": source.name,
                "family": source.family,
                "role": "quality",
                "known_future": bool(source.known_future),
                **routed,
            }
        )

    operational_share: float | None = None
    if source.operational_eligibility_column:
        operational = raw[source.operational_eligibility_column]
        operational_share = float(operational.fillna(False).astype(bool).mean())
    stage_counts: dict[str, int] = {}
    if source.stage_column:
        stage_counts = {
            str(label): int(count)
            for label, count in raw[source.stage_column]
            .fillna("missing")
            .astype(str)
            .value_counts()
            .items()
        }
    day_lengths = _day_lengths(expected, timezone=timezone)
    finite = values.notna()
    age_values = quality[f"{source.name}__age_hours"]
    prospective_sidecar = bool(
        source_sidecar is not None
        and source_sidecar["prospective_capture_verified"] is True
    )
    historical_sidecar = bool(
        source_sidecar is not None
        and source_sidecar["historical_backtest_verified"] is True
    )
    production_pit_evidence: bool | None
    if source.production_evidence_kind is None and operational_share is None:
        production_pit_evidence = None
    else:
        # Both a row-level operational flag and a cryptographically bound,
        # explicitly prospective materializer audit are mandatory.  In
        # particular, a historical revision-date query is research evidence.
        production_pit_evidence = bool(
            operational_share == 1.0 and prospective_sidecar
        )
    backtest_pit_evidence: bool | None
    if source.production_evidence_kind is None:
        backtest_pit_evidence = None
    else:
        # Prospective capture is also valid historical evidence for those exact
        # rows.  An attested historical archive is deliberately *not* promoted
        # to prospective/live evidence.
        backtest_pit_evidence = bool(
            production_pit_evidence is True or historical_sidecar
        )
    audit = {
        "path": str(path),
        "sha256": parquet_sha256,
        "source_audit": source_sidecar,
        "source_audit_sha256": (
            source_sidecar["sha256"] if source_sidecar is not None else None
        ),
        "historical_source_contract_sha256": (
            source_sidecar.get("historical_source_contract_sha256")
            if source_sidecar is not None
            else None
        ),
        "family": source.family,
        "transform": source.transform,
        "value_columns": list(values.columns),
        "expected_hours": int(len(expected)),
        "source_hours": int(len(raw)),
        "complete_hours": int(finite.all(axis=1).sum()),
        "missing_hours": int((~finite.all(axis=1)).sum()),
        "minimum_column_coverage": float(finite.mean().min()),
        "maximum_age_hours": (
            None if age_values.dropna().empty else float(age_values.max())
        ),
        "operational_eligible_share": operational_share,
        "production_pit_evidence": production_pit_evidence,
        "historical_backtest_pit_evidence": backtest_pit_evidence,
        "production_evidence_kind": source.production_evidence_kind,
        "publication_stage_hour_counts": stage_counts,
        "day_length_counts": day_lengths,
        "causal_cutoff": "D-1 08:00 civil",
        "forecast_origin_timezone": timezone,
        "source_cutoff_timezone": source_cutoff_timezone,
        "causality_violations": 0,
    }
    if source.transform == "flowbased_compact":
        audit["flowbased_imputed_hours"] = int(
            values["flowbased_hour_imputed"].fillna(1.0).gt(0.0).sum()
        )
        audit["flowbased_mean_cnec_mtu_availability"] = float(
            values["flowbased_availability"].mean()
        )
    combined = pd.concat([values, quality], axis=1)
    return combined, catalogue, audit


def _day_lengths(index: pd.DatetimeIndex, *, timezone: str) -> dict[str, int]:
    local_days = pd.Index(pd.DatetimeIndex(index).tz_convert(timezone).date)
    counts = pd.Series(1, index=local_days).groupby(level=0).sum()
    invalid = counts.loc[~counts.isin([23, 24, 25])]
    if not invalid.empty:
        raise ExogenousBankError(
            f"Jours civils avec un nombre d'heures invalide: {invalid.to_dict()}."
        )
    histogram = counts.value_counts().sort_index()
    return {str(int(length)): int(number) for length, number in histogram.items()}


def build_exogenous_bank(
    sources: Sequence[ParquetFeatureSource],
    *,
    start_day: str | pd.Timestamp,
    end_day: str | pd.Timestamp,
    timezone: str = DEFAULT_TIMEZONE,
    cutoff_time: str = DEFAULT_CUTOFF_TIME,
    require_complete: bool = False,
    require_historical_backtest_evidence: bool = False,
    require_operational_evidence: bool = False,
) -> ExogenousBank:
    """Validate and join PIT sources on an exact physical-hour timeline."""

    if not sources:
        raise ExogenousBankError("Au moins une source exogene est requise.")
    names = [source.name for source in sources]
    if len(names) != len(set(names)):
        raise ExogenousBankError("Les noms de sources doivent etre uniques.")
    expected = delivery_utc_index(start_day, end_day, timezone=timezone)
    cutoffs = cutoff_by_delivery_hour(
        expected, timezone=timezone, cutoff_time=cutoff_time
    )
    pieces: list[pd.DataFrame] = []
    catalogue_rows: list[dict[str, Any]] = []
    source_audits: dict[str, Any] = {}
    columns: set[str] = set()
    blockers: list[str] = []
    production_blockers: list[str] = []
    backtest_blockers: list[str] = []
    for source in sources:
        piece, catalogue, audit = _load_one_source(
            source,
            expected=expected,
            expected_cutoffs=cutoffs,
            timezone=timezone,
            cutoff_time=cutoff_time,
        )
        overlap = sorted(columns.intersection(piece.columns))
        if overlap:
            raise ExogenousBankError(
                f"{source.name}: colonnes dupliquees dans la banque: {overlap}."
            )
        columns.update(map(str, piece.columns))
        pieces.append(piece)
        catalogue_rows.extend(catalogue)
        source_audits[source.name] = audit
        if audit["missing_hours"]:
            blockers.append(
                f"{source.name}: {audit['missing_hours']} heure(s) incomplete(s)"
            )
        evidence = audit["production_pit_evidence"]
        share = audit["operational_eligible_share"]
        if evidence is not True:
            sidecar = audit.get("source_audit")
            classification = (
                sidecar.get("classification")
                if isinstance(sidecar, Mapping)
                else "sidecar_absent"
            )
            detail = (
                f"couverture explicite {share:.1%}, classe={classification}"
                if share is not None
                else f"couverture operationnelle absente, classe={classification}"
            )
            production_blockers.append(
                f"{source.name}: preuve de capture operationnelle "
                f"insuffisante ({detail})"
            )
        historical_evidence = audit["historical_backtest_pit_evidence"]
        if historical_evidence is not True:
            sidecar = audit.get("source_audit")
            classification = (
                sidecar.get("classification")
                if isinstance(sidecar, Mapping)
                else "sidecar_absent"
            )
            backtest_blockers.append(
                f"{source.name}: preuve PIT historique attestee insuffisante "
                f"(classe={classification})"
            )

    calendar = deterministic_calendar_frame(expected, timezone=timezone)
    overlap = sorted(columns.intersection(calendar.columns))
    if overlap:
        raise ExogenousBankError(
            f"Calendrier: colonnes dupliquees dans la banque: {overlap}."
        )
    columns.update(map(str, calendar.columns))
    pieces.append(calendar)
    catalogue_rows.extend(
        {
            "column": column,
            "source": "deterministic_calendar",
            "family": "calendar",
            "role": "value",
            "known_future": True,
            "chronos": True,
            "residual": False,
            "kalman": False,
        }
        for column in CALENDAR_COLUMNS
    )
    calendar_sha = hashlib.sha256(
        f"calendar_v1|timezone={timezone}|formula=incumbent".encode("utf-8")
    ).hexdigest()
    source_audits["deterministic_calendar"] = {
        "path": None,
        "sha256": calendar_sha,
        "family": "calendar",
        "transform": "deterministic",
        "value_columns": list(CALENDAR_COLUMNS),
        "expected_hours": int(len(expected)),
        "source_hours": int(len(expected)),
        "complete_hours": int(len(expected)),
        "missing_hours": 0,
        "minimum_column_coverage": 1.0,
        "maximum_age_hours": 0.0,
        "operational_eligible_share": 1.0,
        "production_pit_evidence": True,
        "historical_backtest_pit_evidence": True,
        "production_evidence_kind": "deterministic_formula",
        "publication_stage_hour_counts": {},
        "day_length_counts": _day_lengths(expected, timezone=timezone),
        "causal_cutoff": "deterministic known future",
        "causality_violations": 0,
    }

    frame = pd.concat(pieces, axis=1)
    frame.index = expected
    catalogue = pd.DataFrame(catalogue_rows)
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "start_day": str(_naive_day(start_day, timezone=timezone).date()),
        "end_day": str(_naive_day(end_day, timezone=timezone).date()),
        "timezone": timezone,
        "cutoff_time": cutoff_time,
        "physical_hours": int(len(expected)),
        "calendar_days": int(len(set(expected.tz_convert(timezone).date))),
        "day_length_counts": _day_lengths(expected, timezone=timezone),
        "sources": source_audits,
        "source_hashes": {
            name: values["sha256"] for name, values in source_audits.items()
        },
        "source_audit_hashes": {
            name: values["source_audit_sha256"]
            for name, values in source_audits.items()
            if values.get("source_audit_sha256") is not None
        },
        "historical_evidence_manifest_hashes": {
            name: values["source_audit"]["historical_evidence"][
                "manifest_sha256"
            ]
            for name, values in source_audits.items()
            if isinstance(values.get("source_audit"), Mapping)
            and isinstance(
                values["source_audit"].get("historical_evidence"), Mapping
            )
        },
        "historical_evidence_manifest_paths": {
            name: values["source_audit"]["historical_evidence"][
                "manifest_path"
            ]
            for name, values in source_audits.items()
            if isinstance(values.get("source_audit"), Mapping)
            and isinstance(
                values["source_audit"].get("historical_evidence"), Mapping
            )
        },
        "historical_source_contract_hashes": {
            name: values["historical_source_contract_sha256"]
            for name, values in source_audits.items()
            if values.get("historical_source_contract_sha256") is not None
        },
        "production_pit_evidence": {
            name: values["production_pit_evidence"]
            for name, values in source_audits.items()
        },
        "historical_backtest_pit_evidence": {
            name: values["historical_backtest_pit_evidence"]
            for name, values in source_audits.items()
        },
        "routing": {
            consumer: list(
                catalogue.loc[catalogue[consumer].astype(bool), "column"].astype(str)
            )
            for consumer in SUPPORTED_CONSUMERS
        },
        "blockers": blockers,
        "production_blockers": production_blockers,
        "historical_backtest_blockers": backtest_blockers,
        "complete": not blockers,
        "historical_backtest_ready": not blockers and not backtest_blockers,
        "production_ready": not blockers and not production_blockers,
    }
    bank = ExogenousBank(frame=frame, catalogue=catalogue, audit=audit)
    if require_complete and blockers:
        raise ExogenousBankError("Banque incomplete: " + "; ".join(blockers))
    if require_historical_backtest_evidence and (blockers or backtest_blockers):
        raise ExogenousBankError(
            "Preuve PIT historique de backtest incomplete: "
            + "; ".join([*blockers, *backtest_blockers])
        )
    if require_operational_evidence and production_blockers:
        raise ExogenousBankError(
            "Preuve PIT operationnelle incomplete: "
            + "; ".join(production_blockers)
        )
    return bank


def _route_for_family(
    family: str,
    routing_by_family: Mapping[str, Sequence[str]] | None,
) -> ConsumerRoute:
    values = (routing_by_family or {}).get(family, ("chronos",))
    route = ConsumerRoute(tuple(str(value).casefold() for value in values))
    route.validate()
    return route


def default_project_sources(
    project_root: str | Path,
    *,
    zone: str,
    pack: str = "full",
    routing_by_family: Mapping[str, Sequence[str]] | None = None,
    weather_root: str | Path | None = None,
) -> tuple[ParquetFeatureSource, ...]:
    """Resolve the checked-in PIT artefacts without reading live configs."""

    root = Path(project_root).expanduser().resolve()
    code = str(zone).strip().upper()
    if code not in {"FR", "DE", "BE", "NL", "ES"}:
        raise ExogenousBankError(f"Zone non supportee: {zone!r}.")
    if pack not in ABLATION_PACKS:
        raise ExogenousBankError(
            f"Pack inconnu {pack!r}; disponibles={sorted(ABLATION_PACKS)}."
        )
    families = set(ABLATION_PACKS[pack])
    if weather_root is None:
        weather_directory = root / "data" / "pit" / "kalman_weather"
    else:
        weather_directory = Path(weather_root).expanduser()
        if not weather_directory.is_absolute():
            weather_directory = root / weather_directory
        weather_directory = weather_directory.resolve()
    result: list[ParquetFeatureSource] = []
    if "residual_load" in families:
        result.append(
            ParquetFeatureSource(
                name="residual_load",
                family="residual_load",
                path=root
                / "data"
                / "pit"
                / "kalman_hybrid"
                / "residual_load_market_features.parquet",
                audit_path=root
                / "data"
                / "pit"
                / "kalman_hybrid"
                / "residual_load_market_features.parquet.audit.json",
                value_columns={column: column for column in RESIDUAL_LOAD_COLUMNS},
                route=_route_for_family("residual_load", routing_by_family),
                cutoff_column="cutoff_time_utc",
                information_time_columns=(
                    "snapshot_time_utc",
                    "revision_time_utc",
                ),
                age_column="revision_time_utc",
                production_evidence_kind="versioned_revision_history",
            )
        )
    if "weather" in families:
        lower = code.casefold()
        for token, output in (
            ("temperature", f"{lower}_temperature_fcst"),
            ("wind_generation", f"{lower}_wind_generation_fcst"),
            ("solar_generation", f"{lower}_solar_generation_fcst"),
        ):
            result.append(
                ParquetFeatureSource(
                    name=f"weather_{lower}_{token}",
                    family="weather",
                    path=weather_directory / f"{lower}_{token}_fcst.parquet",
                    audit_path=(
                        weather_directory
                        / f"{lower}_{token}_fcst.parquet.audit.json"
                    ),
                    value_columns={output: "value"},
                    route=_route_for_family("weather", routing_by_family),
                    cutoff_column="snapshot_time_utc",
                    cutoff_timezone={
                        "FR": "Europe/Paris",
                        "DE": "Europe/Berlin",
                        "BE": "Europe/Brussels",
                        "NL": "Europe/Amsterdam",
                        "ES": "Europe/Madrid",
                    }[code],
                    information_time_columns=(
                        "snapshot_time_utc",
                        "revision_time_utc",
                    ),
                    age_column="revision_time_utc",
                    production_evidence_kind="versioned_revision_history",
                )
            )
    if "fuel" in families:
        result.append(
            ParquetFeatureSource(
                name="fuel_market",
                family="fuel",
                path=root
                / "data"
                / "pit"
                / "kalman_hybrid"
                / "market_fuel_features.parquet",
                audit_path=root
                / "data"
                / "pit"
                / "kalman_hybrid"
                / "market_fuel_features.parquet.audit.json",
                value_columns={column: column for column in FUEL_COLUMNS},
                route=_route_for_family("fuel", routing_by_family),
                cutoff_column="snapshot_time_utc",
                information_time_columns=(
                    "snapshot_time_utc",
                    "revision_time_utc",
                    "market_source_value_time_utc",
                ),
                age_column="market_source_value_time_utc",
                production_evidence_kind="versioned_revision_history",
            )
        )
    if "flowbased" in families:
        result.append(
            ParquetFeatureSource(
                name="flowbased_core",
                family="flowbased",
                path=root
                / "data"
                / "pit"
                / "jao_core_flowbased"
                / "flowbased_features.parquet",
                audit_path=root
                / "data"
                / "pit"
                / "jao_core_flowbased"
                / "flowbased_features.audit.json",
                value_columns={},
                route=_route_for_family("flowbased", routing_by_family),
                cutoff_column="flowbased_cutoff_time_utc",
                information_time_columns=(
                    "flowbased_source_last_modified_utc",
                ),
                age_column="flowbased_source_last_modified_utc",
                eligibility_column="flowbased_pit_eligible",
                operational_eligibility_column=(
                    "flowbased_operational_pit_eligible"
                ),
                stage_column="flowbased_publication_stage",
                allowed_stages=(
                    "initial_computation",
                    "initial_computation_fallback",
                ),
                transform="flowbased_compact",
                production_evidence_kind="versioned_revision_history",
            )
        )
    return tuple(result)


def build_default_project_bank(
    project_root: str | Path,
    *,
    zone: str,
    start_day: str | pd.Timestamp,
    end_day: str | pd.Timestamp,
    pack: str = "full",
    routing_by_family: Mapping[str, Sequence[str]] | None = None,
    weather_root: str | Path | None = None,
    timezone: str = DEFAULT_TIMEZONE,
    require_complete: bool = False,
    require_historical_backtest_evidence: bool = False,
    require_operational_evidence: bool = False,
) -> ExogenousBank:
    """Build one ablation bank from the project's current PIT artefacts."""

    sources = default_project_sources(
        project_root,
        zone=zone,
        pack=pack,
        routing_by_family=routing_by_family,
        weather_root=weather_root,
    )
    return build_exogenous_bank(
        sources,
        start_day=start_day,
        end_day=end_day,
        timezone=timezone,
        require_complete=require_complete,
        require_historical_backtest_evidence=(
            require_historical_backtest_evidence
        ),
        require_operational_evidence=require_operational_evidence,
    )


__all__ = [
    "ABLATION_PACKS",
    "ConsumerRoute",
    "ExogenousBank",
    "ExogenousBankError",
    "FLOWBASED_COMPACT_COLUMNS",
    "FUEL_COLUMNS",
    "CALENDAR_COLUMNS",
    "ParquetFeatureSource",
    "RESIDUAL_LOAD_COLUMNS",
    "build_default_project_bank",
    "build_exogenous_bank",
    "compress_flowbased_features",
    "cutoff_by_delivery_hour",
    "default_project_sources",
    "deterministic_calendar_frame",
    "delivery_utc_index",
]
