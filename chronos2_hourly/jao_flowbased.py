"""Point-in-time JAO Core CNEC/RAM acquisition and hourly features.

Only the ``initialComputation`` publication is eligible as a model input.  It
is normally available before the D-1 08:00 operational cutoff. Publications
unavailable at that cutoff (pre-final/final domains) or created after market
coupling (shadow prices, active constraints, net positions and scheduled
exchanges) are kept outside this module's feature contract.

The public JAO payload is a long CNEC table (thousands of rows per delivery
day).  :func:`build_hourly_flowbased_features` turns it into one UTC row per
hour, which is the only shape accepted by the Kalman experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
import gzip
import hashlib
import json
import os
from pathlib import Path
import ssl
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

import httpx
import numpy as np
import pandas as pd


JAO_CORE_DATA_URL = "https://publicationtool.jao.eu/core/api/data"
MODEL_INPUT_ENDPOINT = "initialComputation"
GATED_CHALLENGER_ENDPOINT = "preFinalComputation"
AUDIT_ONLY_ENDPOINTS = frozenset(
    {
        "shadowPrices",
        "netPos",
        "scheduledExchanges",
        "priceSpread",
        "finalComputation",
    }
)
SUPPORTED_ENDPOINTS = frozenset(
    {MODEL_INPUT_ENDPOINT, GATED_CHALLENGER_ENDPOINT, *AUDIT_ONLY_ENDPOINTS}
)
FLOWBASED_SCHEMA_VERSION = 3
DEFAULT_TIMEZONE = "Europe/Paris"
DEFAULT_CUTOFF_TIME = "08:00"
DEFAULT_INITIAL_PUBLICATION_TIME = "01:15"
CORE_PTDF_ZONES: tuple[str, ...] = (
    "ALBE",
    "ALDE",
    "AT",
    "BE",
    "CZ",
    "DE",
    "FR",
    "HR",
    "HU",
    "NL",
    "PL",
    "RO",
    "SI",
    "SK",
)

# Stable aggregate features only.  Individual CNEC identifiers are retained
# in the raw audit bundle, never one-hot encoded into the Kalman state.
FLOWBASED_FEATURE_COLUMNS: tuple[str, ...] = (
    "flowbased_cnec_mtu_availability",
    "flowbased_missing_mtu_share",
    "flowbased_hour_imputed",
    "flowbased_cnec_count",
    "flowbased_external_constraint_count",
    "flowbased_equality_constraint_count",
    "flowbased_external_ram_min_mw",
    "flowbased_external_ram_p10_mw",
    "flowbased_ram_min_mw",
    "flowbased_ram_p05_mw",
    "flowbased_ram_p10_mw",
    "flowbased_ram_median_mw",
    "flowbased_ram_iqr_mw",
    "flowbased_ram_below_500_share",
    "flowbased_ram_below_1000_share",
    "flowbased_ram_to_fmax_p05",
    "flowbased_ram_to_fmax_median",
    "flowbased_fr_de_ptdf_spread_p90",
    "flowbased_fr_be_ptdf_spread_p90",
    "flowbased_fr_nl_ptdf_spread_p90",
    "flowbased_fr_neighbor_ptdf_spread_p90",
    "flowbased_alegro_ptdf_spread_p90",
    "flowbased_core_ptdf_range_p90",
    "flowbased_fr_neighbor_ram_stress_p95_per_gw",
    "flowbased_alegro_ram_stress_p95_per_gw",
    "flowbased_core_ram_stress_p95_per_gw",
    "flowbased_stress_hhi",
)


class JaoFlowBasedError(RuntimeError):
    """Raised when the JAO/PIT/feature contract is incomplete or unsafe."""


class _JaoPaginationSnapshotChanged(JaoFlowBasedError):
    """Signal that a complete pagination must restart from its first page."""

    def __init__(self, message: str, *, requests: int) -> None:
        super().__init__(message)
        self.requests = int(requests)


def build_windows_trust_context() -> tuple[ssl.SSLContext, int]:
    """Return an SSL context enriched with trusted Windows root certificates."""

    enum_certificates = getattr(ssl, "enum_certificates", None)
    if os.name != "nt" or enum_certificates is None:
        raise JaoFlowBasedError(
            "Le magasin de certificats Windows n'est pas disponible."
        )
    context = ssl.create_default_context()
    loaded = 0
    for certificate, encoding, _trust in enum_certificates("ROOT"):
        if encoding != "x509_asn":
            continue
        try:
            pem = ssl.DER_cert_to_PEM_cert(certificate)
            context.load_verify_locations(cadata=pem)
        except (ValueError, ssl.SSLError) as exc:
            raise JaoFlowBasedError(
                "Impossible de charger une racine du magasin Windows."
            ) from exc
        loaded += 1
    if loaded == 0:
        raise JaoFlowBasedError(
            "Aucun certificat racine X.509 trouve dans le magasin Windows."
        )
    return context, loaded


def _aware_utc(value: Any, *, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise JaoFlowBasedError(f"{name}: timestamp invalide: {value!r}.") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise JaoFlowBasedError(f"{name}: un offset explicite est obligatoire.")
    return timestamp.tz_convert("UTC")


def _iso_utc(value: pd.Timestamp) -> str:
    return value.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def local_day_utc_bounds(
    delivery_day: date | str,
    *,
    timezone: str = DEFAULT_TIMEZONE,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return DST-safe UTC bounds for one civil delivery day."""

    day = pd.Timestamp(delivery_day).date()
    start = pd.Timestamp(day, tz=timezone).tz_convert("UTC")
    end = pd.Timestamp(day + timedelta(days=1), tz=timezone).tz_convert("UTC")
    if len(pd.date_range(start, end, freq="h", inclusive="left")) not in {23, 24, 25}:
        raise JaoFlowBasedError(f"Jour civil invalide pour {timezone}: {day}.")
    return start, end


def expected_cutoff_utc(
    delivery_day: date | str,
    *,
    timezone: str = DEFAULT_TIMEZONE,
    cutoff_time: str = DEFAULT_CUTOFF_TIME,
) -> pd.Timestamp:
    day = pd.Timestamp(delivery_day).date()
    try:
        clock = pd.Timedelta(
            cutoff_time + ":00" if cutoff_time.count(":") == 1 else cutoff_time
        )
    except (TypeError, ValueError) as exc:
        raise JaoFlowBasedError(f"Heure de cutoff invalide: {cutoff_time!r}.") from exc
    wall_time = pd.Timestamp(day - timedelta(days=1)) + clock
    return wall_time.tz_localize(
        timezone, ambiguous="raise", nonexistent="raise"
    ).tz_convert("UTC")


def expected_initial_publication_utc(
    delivery_day: date | str,
    *,
    timezone: str = DEFAULT_TIMEZONE,
    publication_time: str = DEFAULT_INITIAL_PUBLICATION_TIME,
) -> pd.Timestamp:
    day = pd.Timestamp(delivery_day).date()
    clock = pd.Timedelta(
        publication_time + ":00"
        if publication_time.count(":") == 1
        else publication_time
    )
    wall_time = pd.Timestamp(day - timedelta(days=1)) + clock
    return wall_time.tz_localize(
        timezone, ambiguous="raise", nonexistent="raise"
    ).tz_convert("UTC")


@dataclass(frozen=True)
class JaoFetchResult:
    endpoint: str
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp
    rows: tuple[Mapping[str, Any], ...]
    total_rows: int
    last_modified_utc: pd.Timestamp | None
    retrieved_at_utc: pd.Timestamp
    filters: Mapping[str, Any]
    requests: int
    pages: int = 1
    page_last_modified_utc: tuple[pd.Timestamp, ...] = ()
    snapshot_fingerprint_sha256: str | None = None
    snapshot_verification_scans: int = 1

    def audit_dict(self) -> dict[str, Any]:
        return {
            "api_base_url": JAO_CORE_DATA_URL,
            "http_method": "GET",
            "endpoint": self.endpoint,
            "start_utc": self.start_utc.isoformat(),
            "end_utc": self.end_utc.isoformat(),
            "total_rows": int(self.total_rows),
            "last_modified_utc": (
                self.last_modified_utc.isoformat()
                if self.last_modified_utc is not None
                else None
            ),
            "retrieved_at_utc": self.retrieved_at_utc.isoformat(),
            "filters": dict(self.filters),
            "requests": int(self.requests),
            "pages": int(self.pages),
            "page_last_modified_utc": [
                item.isoformat() for item in self.page_last_modified_utc
            ],
            "snapshot_fingerprint_sha256": self.snapshot_fingerprint_sha256,
            "snapshot_verification_scans": int(
                self.snapshot_verification_scans
            ),
            "to_utc_is_exclusive": True,
        }


class JaoCoreClient:
    """Small paginated client for the official JAO Core publication API."""

    def __init__(
        self,
        *,
        base_url: str = JAO_CORE_DATA_URL,
        page_size: int = 40000,
        timeout_seconds: float = 90.0,
        maximum_retries: int = 4,
        request_interval_seconds: float = 0.65,
        verify: bool | str | ssl.SSLContext = True,
        client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 1 <= int(page_size) <= 40000:
            raise ValueError("page_size doit appartenir a [1, 40000].")
        if maximum_retries < 0:
            raise ValueError("maximum_retries doit etre positif ou nul.")
        if request_interval_seconds < 0:
            raise ValueError("request_interval_seconds doit etre positif ou nul.")
        self.base_url = str(base_url).rstrip("/")
        self.page_size = int(page_size)
        self.maximum_retries = int(maximum_retries)
        self.request_interval_seconds = float(request_interval_seconds)
        self._sleeper = sleeper
        self._monotonic = monotonic
        self._last_request_started: float | None = None
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=float(timeout_seconds),
            verify=verify,
            follow_redirects=True,
            headers={"User-Agent": "Chronos2-flowbased-research/1"},
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "JaoCoreClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _throttle(self) -> None:
        if self._last_request_started is not None:
            remaining = self.request_interval_seconds - (
                self._monotonic() - self._last_request_started
            )
            if remaining > 0:
                self._sleeper(remaining)
        self._last_request_started = self._monotonic()

    def _request(self, endpoint: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        url = f"{self.base_url}/{endpoint}"
        last_error: Exception | None = None
        for attempt in range(self.maximum_retries + 1):
            self._throttle()
            try:
                response = self._client.get(url, params=params)
                if response.status_code == 429 or response.status_code >= 500:
                    snippet = response.text[:300]
                    raise JaoFlowBasedError(
                        f"JAO HTTP {response.status_code}: {snippet}"
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, Mapping):
                    raise JaoFlowBasedError("JAO: objet JSON attendu.")
                if payload.get("rejected") is True:
                    raise JaoFlowBasedError(
                        f"JAO a rejete la requete: {payload.get('messages')!r}."
                    )
                return payload
            except (httpx.HTTPError, ValueError, JaoFlowBasedError) as exc:
                last_error = exc
                if attempt >= self.maximum_retries:
                    break
                self._sleeper(min(2.0**attempt, 30.0))
        raise JaoFlowBasedError(
            f"Echec API JAO apres {self.maximum_retries + 1} tentative(s): "
            f"{last_error}"
        ) from last_error

    def _fetch_snapshot(
        self,
        endpoint: str,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        selected_filters: Mapping[str, Any],
    ) -> JaoFetchResult:
        rows: list[Mapping[str, Any]] = []
        page_last_modified: list[pd.Timestamp] = []
        skip = 0
        total: int | None = None
        request_count = 0
        while total is None or skip < total:
            params: dict[str, Any] = {
                "FromUtc": _iso_utc(start),
                "ToUtc": _iso_utc(end),
                "Skip": skip,
                "Take": self.page_size,
            }
            if selected_filters:
                params["Filter"] = json.dumps(
                    selected_filters, separators=(",", ":"), sort_keys=True
                )
            payload = self._request(endpoint, params)
            request_count += 1
            if payload.get("skip") != skip or payload.get("take") != self.page_size:
                raise JaoFlowBasedError(
                    "JAO: echo Skip/Take different de la requete."
                )
            if endpoint == MODEL_INPUT_ENDPOINT and selected_filters.get(
                "Presolved"
            ) is True:
                applied = payload.get("appliedFilter")
                if not isinstance(applied, Mapping) or applied.get("presolved") is not True:
                    raise JaoFlowBasedError(
                        "JAO: le filtre serveur Presolved=true n'a pas ete applique."
                    )
            raw_data = payload.get("data")
            if not isinstance(raw_data, list):
                raise JaoFlowBasedError("JAO: data doit etre une liste.")
            declared = payload.get("totalRowsWithFilter", payload.get("totalRows"))
            if isinstance(declared, bool) or not isinstance(declared, int):
                raise JaoFlowBasedError("JAO: totalRowsWithFilter invalide.")
            if total is None:
                total = int(declared)
            elif int(declared) != total:
                raise _JaoPaginationSnapshotChanged(
                    "totalRowsWithFilter a change entre deux pages",
                    requests=request_count,
                )
            modified = payload.get("lastModifiedOn")
            if modified:
                watermark = _aware_utc(modified, name="JAO.lastModifiedOn")
                page_last_modified.append(watermark)
            elif endpoint == MODEL_INPUT_ENDPOINT and int(declared) > 0:
                raise JaoFlowBasedError(
                    "JAO initialComputation: lastModifiedOn manque sur une page."
                )
            for item in raw_data:
                if not isinstance(item, Mapping):
                    raise JaoFlowBasedError("JAO: ligne non objet.")
                rows.append(dict(item))
            if not raw_data:
                if skip < total:
                    raise JaoFlowBasedError("JAO: page vide avant la fin declaree.")
                break
            skip += len(raw_data)
        if total is None or len(rows) != total:
            raise JaoFlowBasedError(
                f"JAO: {len(rows)} lignes lues, total declare={total}."
            )
        identifiers = [item.get("id") for item in rows if item.get("id") is not None]
        if len(identifiers) != len(set(identifiers)):
            raise JaoFlowBasedError("JAO: identifiants dupliques entre pages.")
        canonical_rows = sorted(
            json.dumps(
                dict(item),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
                default=str,
            )
            for item in rows
        )
        canonical_watermarks = sorted(_iso_utc(item) for item in page_last_modified)
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "total": total,
                    "rows": canonical_rows,
                    "page_last_modified_utc": canonical_watermarks,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        return JaoFetchResult(
            endpoint=endpoint,
            start_utc=start,
            end_utc=end,
            rows=tuple(rows),
            total_rows=total,
            last_modified_utc=(
                max(page_last_modified) if page_last_modified else None
            ),
            retrieved_at_utc=pd.Timestamp.now(tz="UTC"),
            filters=selected_filters,
            requests=request_count,
            pages=request_count,
            page_last_modified_utc=tuple(sorted(page_last_modified)),
            snapshot_fingerprint_sha256=fingerprint,
        )

    def fetch(
        self,
        endpoint: str,
        *,
        start_utc: Any,
        end_utc: Any,
        filters: Mapping[str, Any] | None = None,
    ) -> JaoFetchResult:
        if endpoint not in SUPPORTED_ENDPOINTS:
            raise JaoFlowBasedError(f"Endpoint JAO non autorise: {endpoint!r}.")
        start = _aware_utc(start_utc, name="start_utc")
        end = _aware_utc(end_utc, name="end_utc")
        if end <= start:
            raise JaoFlowBasedError("end_utc doit etre posterieur a start_utc.")
        selected_filters = dict(filters or {})
        total_requests = 0
        total_scans = 0
        instabilities = 0
        previous: JaoFetchResult | None = None
        last_snapshot_error: Exception | None = None
        while True:
            try:
                result = self._fetch_snapshot(
                    endpoint,
                    start=start,
                    end=end,
                    selected_filters=selected_filters,
                )
            except _JaoPaginationSnapshotChanged as exc:
                total_requests += exc.requests
                last_snapshot_error = exc
                instabilities += 1
                if instabilities > self.maximum_retries:
                    break
                previous = None
                self._sleeper(min(2.0 ** (instabilities - 1), 30.0))
                continue
            total_scans += 1
            total_requests += result.requests
            if result.requests <= 1:
                return replace(
                    result,
                    requests=total_requests,
                    snapshot_verification_scans=total_scans,
                )
            if previous is None:
                previous = result
                continue
            if (
                result.snapshot_fingerprint_sha256
                == previous.snapshot_fingerprint_sha256
            ):
                return replace(
                    result,
                    requests=total_requests,
                    snapshot_verification_scans=total_scans,
                )
            instabilities += 1
            last_snapshot_error = JaoFlowBasedError(
                "deux lectures completes consecutives ont des empreintes "
                "differentes"
            )
            if instabilities > self.maximum_retries:
                break
            previous = result
            self._sleeper(min(2.0 ** (instabilities - 1), 30.0))
        raise JaoFlowBasedError(
            f"JAO: pagination instable apres {total_scans} lecture(s) "
            f"complete(s) et {instabilities} changement(s); aucune page "
            "d'une lecture abandonnee n'a ete conservee. "
            f"Derniere cause: {last_snapshot_error}."
        ) from last_snapshot_error

    def fetch_initial_day(
        self,
        delivery_day: date | str,
        *,
        timezone: str = DEFAULT_TIMEZONE,
    ) -> JaoFetchResult:
        start, end = local_day_utc_bounds(delivery_day, timezone=timezone)
        # Server-side presolve reduces roughly 490k raw rows/day to a few
        # thousand useful CNECs while retaining RAM, Fmax and all Core PTDFs.
        return self.fetch(
            MODEL_INPUT_ENDPOINT,
            start_utc=start,
            end_utc=end,
            filters={"Presolved": True},
        )


def _contingency_key(row: Mapping[str, Any]) -> str:
    contingencies = row.get("contingencies")
    if isinstance(contingencies, list):
        parts = []
        for item in contingencies:
            if isinstance(item, Mapping):
                parts.append(
                    str(
                        item.get("branchEic")
                        or item.get("branchName")
                        or item.get("number")
                        or ""
                    ).strip()
                )
        if any(parts):
            return "|".join(sorted(part for part in parts if part))
    return str(row.get("contName") or "BASECASE").strip()


def normalise_initial_computation(
    result: JaoFetchResult,
    *,
    delivery_day: date | str,
    timezone: str = DEFAULT_TIMEZONE,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    """Validate and flatten one initial-domain daily publication."""

    if result.endpoint != MODEL_INPUT_ENDPOINT:
        raise JaoFlowBasedError(
            "Seul initialComputation peut etre normalise comme entree modele."
        )
    day = pd.Timestamp(delivery_day).date()
    cutoff = expected_cutoff_utc(day, timezone=timezone)
    scheduled = expected_initial_publication_utc(day, timezone=timezone)
    last_modified = result.last_modified_utc
    ptdf_columns = tuple(f"ptdf_{zone}" for zone in CORE_PTDF_ZONES)
    numeric_columns = (
        "ram",
        "fmax",
        "frm",
        "frefInit",
        "fcore",
        "fall",
        "fuaf",
        *ptdf_columns,
    )
    expected_hours = pd.date_range(
        *local_day_utc_bounds(day, timezone=timezone),
        freq="h",
        inclusive="left",
    )
    if not result.rows:
        empty_columns = (
            "source_row_id",
            "delivery_start_utc",
            "tso",
            "cne_name",
            "cne_eic",
            "cne_status",
            "direction",
            "hub_from",
            "hub_to",
            "contingency_key",
            "contingency_name",
            "presolved",
            "api_cnec",
            "cnec",
            "cnec_classification",
            "constraint_kind",
            *numeric_columns,
            "cnec_key",
        )
        empty = pd.DataFrame(columns=empty_columns)
        audit = {
            "schema_version": FLOWBASED_SCHEMA_VERSION,
            "delivery_day": day.isoformat(),
            "timezone": timezone,
            "publication_stage": "initial_computation",
            "model_input_eligible_stage": True,
            "cutoff_time_utc": cutoff.isoformat(),
            "scheduled_publication_time_utc": scheduled.isoformat(),
            "api_last_modified_utc": None,
            "retrieved_at_utc": result.retrieved_at_utc.isoformat(),
            "pit_eligible": False,
            "pit_status": "no_initial_publication",
            "pit_confidence": "not_eligible_until_causal_fallback",
            "operational_pit_eligible": False,
            "historical_retrieval": bool(result.retrieved_at_utc > cutoff),
            "pit_limit": (
                "Aucune publication initiale n'existe pour ce jour; un "
                "fallback depuis une publication initiale anterieure est requis."
            ),
            "api_rows": 0,
            "selected_presolved_rows": 0,
            "selected_cnec_rows": 0,
            "selected_external_or_equality_rows": 0,
            "selected_external_rows": 0,
            "selected_equality_rows": 0,
            "external_negative_ram_sentinel_rows": 0,
            "rejected_non_presolved_rows": 0,
            "descriptor_classified_non_cnec_rows": 0,
            "api_cnec_false_rows": 0,
            "physical_hours": int(len(expected_hours)),
            "physical_mtus": int(len(expected_hours)),
            "observed_source_mtus": 0,
            "missing_source_mtus": int(len(expected_hours)),
            "cnec_available_mtus": 0,
            "missing_cnec_mtus": int(len(expected_hours)),
            "missing_cnec_mtu_examples_utc": list(
                map(str, expected_hours[:8])
            ),
            "missing_initial_policy": "causal_previous_initial_day_required",
            "raw_initial_publication_empty": True,
            "mtu_minutes": 60,
            "source_requests": int(result.requests),
            "source_pages": int(result.pages),
            "page_last_modified_utc": [
                item.isoformat() for item in result.page_last_modified_utc
            ],
            "snapshot_fingerprint_sha256": result.snapshot_fingerprint_sha256,
            "snapshot_verification_scans": int(
                result.snapshot_verification_scans
            ),
            "causality_violations": 1,
            "operational_capture_violations": 1,
        }
        return empty, audit
    pit_eligible = bool(last_modified is not None and last_modified <= cutoff)
    operational_pit_eligible = bool(
        pit_eligible and result.retrieved_at_utc <= cutoff
    )
    pit_status = (
        "api_last_modified_pre_cutoff"
        if pit_eligible
        else (
            "api_last_modified_after_cutoff"
            if last_modified is not None
            else "api_last_modified_missing"
        )
    )
    rows: list[dict[str, Any]] = []
    rejected_not_presolved = 0
    descriptor_classified_non_cnec = 0
    api_cnec_false_rows = 0
    for position, raw in enumerate(result.rows):
        if raw.get("presolved") is not True:
            rejected_not_presolved += 1
            continue
        timestamp = _aware_utc(raw.get("dateTimeUtc"), name="dateTimeUtc")
        if not result.start_utc <= timestamp < result.end_utc:
            raise JaoFlowBasedError(
                f"JAO initial: heure hors jour a la ligne {position}."
            )
        raw_cnec_flag = raw.get("cnec")
        api_cnec = raw_cnec_flag if isinstance(raw_cnec_flag, bool) else None
        api_cnec_false_rows += int(api_cnec is False)
        descriptor = " ".join(
            str(raw.get(name) or "")
            for name in ("cnecType", "cneStatus", "cneName")
        ).casefold()
        is_equality = "equality constraint" in descriptor
        is_external = any(
            token in descriptor
            for token in (
                "external constraint",
                "allocation constraint",
                "import limit",
                "export limit",
            )
        )
        is_external_or_equality = is_external or is_equality
        if is_external_or_equality:
            # JAO's JSON ``cnec`` flag is also true for technical constraints
            # such as ALEGrO.  It is therefore retained for provenance, while
            # the descriptor defines the modelling category.
            descriptor_classified_non_cnec += 1
            is_cnec = False
            cnec_classification = "descriptor_external_or_equality"
        elif api_cnec is False:
            is_cnec = False
            cnec_classification = "api_non_cnec"
        else:
            is_cnec = True
            cnec_classification = "standard_cnec"
        flattened: dict[str, Any] = {
            "source_row_id": raw.get("id"),
            "delivery_start_utc": timestamp,
            "tso": str(raw.get("tso") or "").strip(),
            "cne_name": str(raw.get("cneName") or "").strip(),
            "cne_eic": str(raw.get("cneEic") or "").strip(),
            "cne_status": str(raw.get("cneStatus") or "").strip(),
            "direction": str(raw.get("direction") or "").strip(),
            "hub_from": str(raw.get("hubFrom") or "").strip(),
            "hub_to": str(raw.get("hubTo") or "").strip(),
            "contingency_key": _contingency_key(raw),
            "contingency_name": str(raw.get("contName") or "").strip(),
            "presolved": True,
            "api_cnec": api_cnec,
            "cnec": bool(is_cnec),
            "cnec_classification": cnec_classification,
            "constraint_kind": (
                "cnec"
                if is_cnec
                else "equality"
                if is_equality
                else "external"
                if is_external
                else "other_non_cnec"
            ),
        }
        for column in numeric_columns:
            flattened[column] = pd.to_numeric(raw.get(column), errors="coerce")
        stable = "|".join(
            (
                flattened["cne_eic"] or flattened["cne_name"],
                flattened["direction"],
                flattened["contingency_key"],
            )
        )
        flattened["cnec_key"] = hashlib.sha256(
            stable.encode("utf-8")
        ).hexdigest()[:24]
        rows.append(flattened)
    if not rows:
        raise JaoFlowBasedError(f"Aucun CNEC presolved exploitable pour {day}.")
    frame = pd.DataFrame(rows).sort_values(
        ["delivery_start_utc", "source_row_id"], kind="stable"
    )
    required_numeric = ["ram", *ptdf_columns]
    if not np.isfinite(frame[required_numeric].to_numpy(dtype=float)).all():
        raise JaoFlowBasedError(
            f"JAO initial {day}: RAM/PTDF requis non finis."
        )
    if frame["source_row_id"].duplicated().any():
        raise JaoFlowBasedError(f"JAO initial {day}: source_row_id duplique.")
    expected = pd.date_range(
        *local_day_utc_bounds(day, timezone=timezone),
        freq="h",
        inclusive="left",
    )
    observed_mtu = pd.DatetimeIndex(frame["delivery_start_utc"]).unique().sort_values()
    expected_quarter_hour = pd.date_range(
        *local_day_utc_bounds(day, timezone=timezone),
        freq="15min",
        inclusive="left",
    )
    hourly_subset = len(observed_mtu.difference(expected)) == 0
    quarter_hour_subset = len(observed_mtu.difference(expected_quarter_hour)) == 0
    contains_subhourly_mtu = any(timestamp.minute != 0 for timestamp in observed_mtu)
    if hourly_subset and not contains_subhourly_mtu:
        mtu_minutes = 60
        expected_mtu = expected
    elif quarter_hour_subset and (
        contains_subhourly_mtu or len(observed_mtu) > len(expected)
    ):
        mtu_minutes = 15
        expected_mtu = expected_quarter_hour
    else:
        extra = observed_mtu.difference(expected_quarter_hour)
        raise JaoFlowBasedError(
            f"JAO initial {day}: grille MTU invalide; "
            f"extra={list(map(str, extra[:3]))}."
        )
    missing_source_mtu = expected_mtu.difference(observed_mtu)
    cnec_by_mtu = frame.groupby("delivery_start_utc", sort=True)["cnec"].any()
    missing_cnec_mtu = expected_mtu.difference(
        pd.DatetimeIndex(cnec_by_mtu.index[cnec_by_mtu.to_numpy(dtype=bool)])
    )
    audit = {
        "schema_version": FLOWBASED_SCHEMA_VERSION,
        "delivery_day": day.isoformat(),
        "timezone": timezone,
        "publication_stage": "initial_computation",
        "model_input_eligible_stage": True,
        "cutoff_time_utc": cutoff.isoformat(),
        "scheduled_publication_time_utc": scheduled.isoformat(),
        "api_last_modified_utc": (
            last_modified.isoformat() if last_modified is not None else None
        ),
        "retrieved_at_utc": result.retrieved_at_utc.isoformat(),
        "pit_eligible": pit_eligible,
        "pit_status": pit_status,
        "pit_confidence": (
            "captured_before_cutoff"
            if operational_pit_eligible
            else "historical_last_modified_only"
            if pit_eligible
            else "not_eligible"
        ),
        "operational_pit_eligible": operational_pit_eligible,
        "historical_retrieval": bool(result.retrieved_at_utc > cutoff),
        "pit_limit": (
            "JAO lastModifiedOn est audite, mais le telechargement historique "
            "ne constitue pas a lui seul une archive de tous les vintages."
        ),
        "api_rows": int(result.total_rows),
        "selected_presolved_rows": int(len(frame)),
        "selected_cnec_rows": int(frame["cnec"].sum()),
        "selected_external_or_equality_rows": int((~frame["cnec"]).sum()),
        "selected_external_rows": int(
            frame["constraint_kind"].eq("external").sum()
        ),
        "selected_equality_rows": int(
            frame["constraint_kind"].eq("equality").sum()
        ),
        "external_negative_ram_sentinel_rows": int(
            (
                frame["constraint_kind"].eq("external")
                & (pd.to_numeric(frame["ram"], errors="coerce") < 0.0)
            ).sum()
        ),
        "rejected_non_presolved_rows": int(rejected_not_presolved),
        "descriptor_classified_non_cnec_rows": int(
            descriptor_classified_non_cnec
        ),
        "api_cnec_false_rows": int(api_cnec_false_rows),
        "physical_hours": int(len(expected)),
        "physical_mtus": int(len(expected_mtu)),
        "observed_source_mtus": int(len(observed_mtu)),
        "missing_source_mtus": int(len(missing_source_mtu)),
        "cnec_available_mtus": int(len(expected_mtu) - len(missing_cnec_mtu)),
        "missing_cnec_mtus": int(len(missing_cnec_mtu)),
        "missing_cnec_mtu_examples_utc": list(
            map(str, missing_cnec_mtu[:8])
        ),
        "missing_initial_policy": (
            "same_publication_daily_median_for_empty_hours; availability "
            "indicators retained; final/postcoupling data forbidden"
        ),
        "mtu_minutes": mtu_minutes,
        "source_requests": int(result.requests),
        "source_pages": int(result.pages),
        "page_last_modified_utc": [
            item.isoformat() for item in result.page_last_modified_utc
        ],
        "snapshot_fingerprint_sha256": result.snapshot_fingerprint_sha256,
        "snapshot_verification_scans": int(result.snapshot_verification_scans),
        "causality_violations": int(not pit_eligible),
        "operational_capture_violations": int(not operational_pit_eligible),
    }
    return frame.reset_index(drop=True), audit


def _quantile(values: pd.Series, probability: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    return float(np.quantile(numeric, probability)) if numeric.size else float("nan")


def build_hourly_flowbased_features(
    cnec: pd.DataFrame,
    *,
    daily_audit: Mapping[str, Any],
) -> pd.DataFrame:
    """Aggregate long CNEC/RAM rows into stable causal hourly features."""

    required = {
        "delivery_start_utc",
        "cnec_key",
        "tso",
        "cnec",
        "constraint_kind",
        "ram",
        "fmax",
        *(f"ptdf_{zone}" for zone in CORE_PTDF_ZONES),
    }
    missing = sorted(required.difference(cnec.columns))
    if missing:
        raise JaoFlowBasedError(f"CNEC normalise incomplet: {missing}.")
    frame = cnec.copy()
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["delivery_start_utc"], utc=True))
    if timestamps.hasnans:
        raise JaoFlowBasedError("CNEC: timestamps invalides.")
    frame["_hour_utc"] = timestamps.floor("h")
    frame["_mtu_utc"] = timestamps
    ram = pd.to_numeric(frame["ram"], errors="coerce")
    fmax = pd.to_numeric(frame["fmax"], errors="coerce")
    safe_ram = ram.abs().clip(lower=100.0)
    ratio = ram / fmax.where(fmax.abs() > 1e-9)
    frame["_ram_fmax"] = ratio
    ptdf = {
        zone: pd.to_numeric(frame[f"ptdf_{zone}"], errors="coerce")
        for zone in CORE_PTDF_ZONES
    }
    frame["_fr_de_spread"] = (ptdf["FR"] - ptdf["DE"]).abs()
    frame["_fr_be_spread"] = (ptdf["FR"] - ptdf["BE"]).abs()
    frame["_fr_nl_spread"] = (ptdf["FR"] - ptdf["NL"]).abs()
    frame["_fr_neighbor_spread"] = frame[
        ["_fr_de_spread", "_fr_be_spread", "_fr_nl_spread"]
    ].max(axis=1)
    frame["_fr_neighbor_stress"] = (
        frame["_fr_neighbor_spread"] / safe_ram * 1000.0
    )
    frame["_alegro_spread"] = (ptdf["ALBE"] - ptdf["ALDE"]).abs()
    frame["_alegro_stress"] = frame["_alegro_spread"] / safe_ram * 1000.0
    matrix = np.column_stack([ptdf[zone].to_numpy(dtype=float) for zone in ptdf])
    core_range = np.nanmax(matrix, axis=1) - np.nanmin(matrix, axis=1)
    frame["_core_stress"] = core_range / safe_ram.to_numpy(dtype=float) * 1000.0
    frame["_core_ptdf_range"] = core_range

    indicator_columns = {
        "flowbased_cnec_mtu_availability",
        "flowbased_missing_mtu_share",
        "flowbased_hour_imputed",
    }
    imputable_columns = tuple(
        column
        for column in FLOWBASED_FEATURE_COLUMNS
        if column not in indicator_columns
    )
    rows: list[dict[str, Any]] = []
    for hour, block in frame.groupby("_hour_utc", sort=True):
        cnec_block = block.loc[block["cnec"].astype(bool)].copy()
        mtu_count = int(block["_mtu_utc"].nunique())
        cnec_mtu_count = int(cnec_block["_mtu_utc"].nunique())
        external_block = block.loc[
            block["constraint_kind"].astype(str).eq("external")
        ].copy()
        external_ram = pd.to_numeric(
            external_block["ram"], errors="coerce"
        ).loc[lambda values: values >= 0.0]
        external_count = int(len(external_block))
        equality_count = int(
            block["constraint_kind"].astype(str).eq("equality").sum()
        )
        if cnec_block.empty:
            empty_row: dict[str, Any] = {
                "value_time_utc": pd.Timestamp(hour),
                **{column: float("nan") for column in imputable_columns},
                "flowbased_cnec_count": 0.0,
                "flowbased_external_constraint_count": float(
                    external_count / max(mtu_count, 1)
                ),
                "flowbased_equality_constraint_count": float(
                    equality_count / max(mtu_count, 1)
                ),
                "flowbased_external_ram_min_mw": (
                    float(external_ram.min()) if len(external_ram) else 0.0
                ),
                "flowbased_external_ram_p10_mw": (
                    _quantile(external_ram, 0.10) if len(external_ram) else 0.0
                ),
                "flowbased_source_mtu_count": mtu_count,
                "flowbased_source_cnec_mtu_count": 0,
                "flowbased_source_row_count": int(len(block)),
            }
            rows.append(empty_row)
            continue
        mtu_hhi: list[float] = []
        for _, mtu_block in cnec_block.groupby("_mtu_utc", sort=True):
            weights = pd.to_numeric(
                mtu_block["_fr_neighbor_stress"], errors="coerce"
            ).abs()
            finite_weights = weights[np.isfinite(weights)]
            weight_sum = float(finite_weights.sum())
            mtu_hhi.append(
                float(
                    np.square(
                        finite_weights.to_numpy(dtype=float) / weight_sum
                    ).sum()
                )
                if weight_sum > 0.0
                else 0.0
            )
        hhi = float(np.mean(mtu_hhi))
        block_ram = pd.to_numeric(cnec_block["ram"], errors="coerce")
        rows.append(
            {
                "value_time_utc": pd.Timestamp(hour),
                "flowbased_cnec_count": float(
                    len(cnec_block) / max(cnec_mtu_count, 1)
                ),
                "flowbased_external_constraint_count": float(
                    external_count / max(mtu_count, 1)
                ),
                "flowbased_equality_constraint_count": float(
                    equality_count / max(mtu_count, 1)
                ),
                "flowbased_external_ram_min_mw": (
                    float(external_ram.min()) if len(external_ram) else 0.0
                ),
                "flowbased_external_ram_p10_mw": (
                    _quantile(external_ram, 0.10) if len(external_ram) else 0.0
                ),
                "flowbased_ram_min_mw": float(block_ram.min()),
                "flowbased_ram_p05_mw": _quantile(block_ram, 0.05),
                "flowbased_ram_p10_mw": _quantile(block_ram, 0.10),
                "flowbased_ram_median_mw": _quantile(block_ram, 0.50),
                "flowbased_ram_iqr_mw": _quantile(block_ram, 0.75)
                - _quantile(block_ram, 0.25),
                "flowbased_ram_below_500_share": float((block_ram < 500.0).mean()),
                "flowbased_ram_below_1000_share": float((block_ram < 1000.0).mean()),
                "flowbased_ram_to_fmax_p05": _quantile(
                    cnec_block["_ram_fmax"], 0.05
                ),
                "flowbased_ram_to_fmax_median": _quantile(
                    cnec_block["_ram_fmax"], 0.50
                ),
                "flowbased_fr_de_ptdf_spread_p90": _quantile(
                    cnec_block["_fr_de_spread"], 0.90
                ),
                "flowbased_fr_be_ptdf_spread_p90": _quantile(
                    cnec_block["_fr_be_spread"], 0.90
                ),
                "flowbased_fr_nl_ptdf_spread_p90": _quantile(
                    cnec_block["_fr_nl_spread"], 0.90
                ),
                "flowbased_fr_neighbor_ptdf_spread_p90": _quantile(
                    cnec_block["_fr_neighbor_spread"], 0.90
                ),
                "flowbased_alegro_ptdf_spread_p90": _quantile(
                    cnec_block["_alegro_spread"], 0.90
                ),
                "flowbased_core_ptdf_range_p90": _quantile(
                    cnec_block["_core_ptdf_range"], 0.90
                ),
                "flowbased_fr_neighbor_ram_stress_p95_per_gw": _quantile(
                    cnec_block["_fr_neighbor_stress"], 0.95
                ),
                "flowbased_alegro_ram_stress_p95_per_gw": _quantile(
                    cnec_block["_alegro_stress"], 0.95
                ),
                "flowbased_core_ram_stress_p95_per_gw": _quantile(
                    cnec_block["_core_stress"], 0.95
                ),
                "flowbased_stress_hhi": hhi,
                "flowbased_source_mtu_count": mtu_count,
                "flowbased_source_cnec_mtu_count": cnec_mtu_count,
                "flowbased_source_row_count": int(len(block)),
                "flowbased_pit_eligible": bool(daily_audit.get("pit_eligible")),
                "flowbased_pit_status": str(daily_audit.get("pit_status")),
                "flowbased_pit_confidence": str(
                    daily_audit.get("pit_confidence")
                ),
                "flowbased_operational_pit_eligible": bool(
                    daily_audit.get("operational_pit_eligible")
                ),
                "flowbased_publication_stage": "initial_computation",
                "flowbased_cutoff_time_utc": daily_audit.get("cutoff_time_utc"),
                "flowbased_source_last_modified_utc": daily_audit.get(
                    "api_last_modified_utc"
                ),
                "flowbased_retrieved_at_utc": daily_audit.get("retrieved_at_utc"),
            }
        )
    timezone = str(daily_audit.get("timezone") or DEFAULT_TIMEZONE)
    day = pd.Timestamp(daily_audit.get("delivery_day")).date()
    expected_hours = pd.date_range(
        *local_day_utc_bounds(day, timezone=timezone),
        freq="h",
        inclusive="left",
    )
    mtu_minutes = int(daily_audit.get("mtu_minutes", 60))
    if mtu_minutes not in {15, 60}:
        raise JaoFlowBasedError(f"Frequence MTU non supportee: {mtu_minutes}.")
    expected_mtus_per_hour = 60 // mtu_minutes
    output = (
        pd.DataFrame(rows)
        .set_index("value_time_utc")
        .sort_index()
        .reindex(expected_hours)
    )
    if output.index.has_duplicates:
        raise JaoFlowBasedError("Features flow-based horaires dupliquees.")
    for column in (
        "flowbased_source_mtu_count",
        "flowbased_source_cnec_mtu_count",
        "flowbased_source_row_count",
    ):
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0)
    availability = (
        output["flowbased_source_cnec_mtu_count"] / expected_mtus_per_hour
    ).clip(lower=0.0, upper=1.0)
    output["flowbased_cnec_mtu_availability"] = availability
    output["flowbased_missing_mtu_share"] = 1.0 - availability
    output["flowbased_hour_imputed"] = availability.eq(0.0).astype(float)
    no_source = output["flowbased_source_mtu_count"].eq(0)
    for column in (
        "flowbased_cnec_count",
        "flowbased_external_constraint_count",
        "flowbased_equality_constraint_count",
        "flowbased_external_ram_min_mw",
        "flowbased_external_ram_p10_mw",
    ):
        output.loc[no_source, column] = 0.0
    for column in imputable_columns:
        numeric = pd.to_numeric(output[column], errors="coerce")
        if numeric.isna().any():
            daily_median = float(numeric.median(skipna=True))
            if not np.isfinite(daily_median):
                raise JaoFlowBasedError(
                    f"JAO initial {day}: aucune valeur causale disponible "
                    f"pour imputer {column}."
                )
            numeric = numeric.fillna(daily_median)
        output[column] = numeric
    # Every value used for the daily median belongs to the same publication
    # initialComputation, released as one D-1 snapshot. This is causal even for
    # a missing delivery hour and never consults final/post-coupling data.
    output["flowbased_pit_eligible"] = bool(daily_audit.get("pit_eligible"))
    output["flowbased_pit_status"] = str(daily_audit.get("pit_status"))
    output["flowbased_pit_confidence"] = str(
        daily_audit.get("pit_confidence")
    )
    output["flowbased_operational_pit_eligible"] = bool(
        daily_audit.get("operational_pit_eligible")
    )
    output["flowbased_publication_stage"] = "initial_computation"
    output["flowbased_cutoff_time_utc"] = daily_audit.get("cutoff_time_utc")
    output["flowbased_source_last_modified_utc"] = daily_audit.get(
        "api_last_modified_utc"
    )
    output["flowbased_retrieved_at_utc"] = daily_audit.get("retrieved_at_utc")
    output.index.name = "value_time_utc"
    output = output.reset_index()
    matrix_out = output.loc[:, list(FLOWBASED_FEATURE_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(matrix_out).all():
        bad = [
            column
            for column in FLOWBASED_FEATURE_COLUMNS
            if not np.isfinite(pd.to_numeric(output[column], errors="coerce")).all()
        ]
        raise JaoFlowBasedError(f"Features flow-based non finies: {bad}.")
    return output


def build_causal_empty_day_fallback(
    previous_features: pd.DataFrame,
    *,
    daily_audit: Mapping[str, Any],
    previous_audit: Mapping[str, Any],
    previous_day: date | str,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    """Build a causal fallback for an absent or post-cutoff publication.

    The copied profile is grouped by local wall-clock hour.  This remains
    causal because the source delivery day and its complete initial domain are
    strictly earlier than the target cutoff.  No final/post-coupling field is
    consulted.  Availability is forced to zero so governance can distinguish
    copied values from a genuine domain.
    """

    raw_empty = daily_audit.get("raw_initial_publication_empty") is True
    raw_pit_eligible = daily_audit.get("pit_eligible") is True
    if not raw_empty and raw_pit_eligible:
        raise JaoFlowBasedError(
            "Le fallback journalier exige une publication vide ou non PIT."
        )
    if previous_audit.get("pit_eligible") is not True:
        raise JaoFlowBasedError(
            "La partition precedente n'est pas admissible comme fallback causal."
        )
    required = {"value_time_utc", *FLOWBASED_FEATURE_COLUMNS}
    missing = sorted(required.difference(previous_features.columns))
    if missing:
        raise JaoFlowBasedError(f"Fallback precedent incomplet: {missing}.")
    previous = previous_features.copy()
    previous_index = pd.DatetimeIndex(
        pd.to_datetime(previous["value_time_utc"], utc=True)
    )
    if previous_index.hasnans or previous_index.has_duplicates:
        raise JaoFlowBasedError("Fallback precedent: timeline invalide.")
    matrix = previous.loc[:, list(FLOWBASED_FEATURE_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise JaoFlowBasedError("Fallback precedent: features non finies.")
    timezone = str(daily_audit.get("timezone") or DEFAULT_TIMEZONE)
    target_day = pd.Timestamp(daily_audit.get("delivery_day")).date()
    source_day = pd.Timestamp(previous_day).date()
    if source_day >= target_day:
        raise JaoFlowBasedError("Le fallback doit provenir d'un jour anterieur.")
    target_index = pd.date_range(
        *local_day_utc_bounds(target_day, timezone=timezone),
        freq="h",
        inclusive="left",
    )
    source_hour = previous_index.tz_convert(timezone).hour
    target_hour = target_index.tz_convert(timezone).hour
    indicator_columns = {
        "flowbased_cnec_mtu_availability",
        "flowbased_missing_mtu_share",
        "flowbased_hour_imputed",
    }
    copied_columns = tuple(
        column
        for column in FLOWBASED_FEATURE_COLUMNS
        if column not in indicator_columns
    )
    output = pd.DataFrame({"value_time_utc": target_index})
    for column in copied_columns:
        numeric = pd.to_numeric(previous[column], errors="coerce")
        wall_hour_profile = numeric.groupby(source_hour).median()
        copied = pd.Series(target_hour, dtype=int).map(wall_hour_profile)
        copied = pd.to_numeric(copied, errors="coerce")
        copied = copied.fillna(float(numeric.median()))
        if not np.isfinite(copied.to_numpy(dtype=float)).all():
            raise JaoFlowBasedError(
                f"Fallback {target_day}: impossible d'imputer {column}."
            )
        output[column] = copied.to_numpy(dtype=float)
    output["flowbased_cnec_mtu_availability"] = 0.0
    output["flowbased_missing_mtu_share"] = 1.0
    output["flowbased_hour_imputed"] = 1.0
    output["flowbased_source_mtu_count"] = 0
    output["flowbased_source_cnec_mtu_count"] = 0
    output["flowbased_source_row_count"] = 0
    source_origin_day = str(
        previous_audit.get("fallback_source_day") or source_day.isoformat()
    )
    source_last_modified = previous_audit.get("api_last_modified_utc")
    output["flowbased_pit_eligible"] = True
    output["flowbased_pit_status"] = "causal_previous_initial_fallback"
    output["flowbased_pit_confidence"] = "historical_previous_initial_only"
    output["flowbased_operational_pit_eligible"] = False
    output["flowbased_publication_stage"] = "initial_computation_fallback"
    output["flowbased_cutoff_time_utc"] = daily_audit.get("cutoff_time_utc")
    output["flowbased_source_last_modified_utc"] = source_last_modified
    output["flowbased_retrieved_at_utc"] = daily_audit.get("retrieved_at_utc")
    selected_audit = dict(daily_audit)
    raw_last_modified = daily_audit.get("api_last_modified_utc")
    fallback_reason = (
        "empty_initial_publication"
        if raw_empty
        else str(daily_audit.get("pit_status") or "initial_publication_non_pit")
    )
    source_mtu_minutes = int(previous_audit.get("mtu_minutes", 60))
    source_mtu_minutes = source_mtu_minutes if source_mtu_minutes in {15, 60} else 60
    physical_mtus = len(target_index) * (60 // source_mtu_minutes)
    selected_audit.update(
        {
            "raw_initial_pit_eligible": False,
            "raw_initial_pit_status": daily_audit.get("pit_status"),
            "raw_api_last_modified_utc": raw_last_modified,
            "raw_api_rows": int(daily_audit.get("api_rows", 0)),
            "raw_causality_violations": int(
                daily_audit.get("causality_violations", 1)
            ),
            "api_last_modified_utc": source_last_modified,
            "pit_eligible": True,
            "pit_status": "causal_previous_initial_fallback",
            "pit_confidence": "historical_previous_initial_only",
            "pit_basis": "previous_initial_computation_only",
            "operational_pit_eligible": False,
            "publication_stage": "initial_computation_fallback",
            "fallback_reason": fallback_reason,
            "fallback_source_day": source_origin_day,
            "fallback_immediate_previous_day": source_day.isoformat(),
            "fallback_consecutive_days": int(
                previous_audit.get("fallback_consecutive_days", 0)
            )
            + 1,
            "fallback_policy": (
                "previous admissible initial publication grouped by local "
                "wall-clock hour; availability forced to zero"
            ),
            "pit_limit": (
                "La publication brute du jour est absente ou posterieure au "
                "cutoff. Les features proviennent uniquement d'une publication "
                "initiale anterieure; la disponibilite est forcee a zero."
            ),
            "mtu_minutes": source_mtu_minutes,
            "physical_mtus": int(physical_mtus),
            "observed_source_mtus": 0,
            "missing_source_mtus": int(physical_mtus),
            "cnec_available_mtus": 0,
            "missing_cnec_mtus": int(physical_mtus),
            "missing_initial_policy": "causal_previous_initial_day",
            "causality_violations": 0,
            "operational_capture_violations": 1,
        }
    )
    return output, selected_audit


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8") + b"\n")


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.parquet")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_daily_flowbased_bundle(
    output_root: str | Path,
    *,
    delivery_day: date | str,
    fetch: JaoFetchResult,
    normalised: pd.DataFrame,
    features: pd.DataFrame,
    audit: Mapping[str, Any],
    overwrite: bool = False,
    tls_verification: bool = True,
    tls_trust_source: str = "python_default",
) -> Mapping[str, Any]:
    """Write one immutable raw+normalised+feature daily partition."""

    root = Path(output_root).expanduser().resolve()
    token = pd.Timestamp(delivery_day).date().isoformat()
    raw_path = root / "raw" / MODEL_INPUT_ENDPOINT / f"{token}.json.gz"
    metadata_path = root / "raw" / MODEL_INPUT_ENDPOINT / f"{token}.audit.json"
    normalised_path = root / "normalised" / f"{token}.parquet"
    feature_path = root / "daily_features" / f"{token}.parquet"
    targets = (raw_path, metadata_path, normalised_path, feature_path)
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        if len(existing) != len(targets):
            raise JaoFlowBasedError(
                f"Partition JAO partielle pour {token}: {existing}."
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if sha256_file(raw_path) != metadata.get("raw_gzip_sha256"):
            raise JaoFlowBasedError(f"Checksum raw JAO invalide pour {token}.")
        if metadata.get("schema_version") != FLOWBASED_SCHEMA_VERSION or tuple(
            metadata.get("feature_columns", ())
        ) != FLOWBASED_FEATURE_COLUMNS:
            raise JaoFlowBasedError(
                f"Partition JAO {token} issue d'un ancien schema; "
                "relancer avec --overwrite."
            )
        return metadata
    raw_payload = {
        "schema_version": FLOWBASED_SCHEMA_VERSION,
        "source": "JAO Core Publication Tool",
        "license_notice": "Verifier les conditions JAO avant redistribution.",
        "fetch": fetch.audit_dict(),
        "records": list(fetch.rows),
    }
    compressed = gzip.compress(_json_bytes(raw_payload), compresslevel=6, mtime=0)
    _atomic_bytes(raw_path, compressed)
    _atomic_parquet(normalised_path, normalised)
    _atomic_parquet(feature_path, features)
    metadata = {
        **dict(audit),
        "raw_path": str(raw_path),
        "raw_gzip_sha256": _sha256_bytes(compressed),
        "raw_gzip_size_bytes": int(len(compressed)),
        "normalised_path": str(normalised_path),
        "normalised_sha256": sha256_file(normalised_path),
        "features_path": str(feature_path),
        "features_sha256": sha256_file(feature_path),
        "feature_columns": list(FLOWBASED_FEATURE_COLUMNS),
        "feature_rows": int(len(features)),
        "tls_verification": bool(tls_verification),
        "tls_trust_source": str(tls_trust_source),
    }
    _atomic_json(metadata_path, metadata)
    return metadata


def assemble_flowbased_feature_store(
    output_root: str | Path,
    *,
    start_day: date | str,
    end_day: date | str,
    require_research_pit: bool = True,
    require_operational_pit: bool = False,
    overwrite: bool = False,
) -> tuple[Path, Mapping[str, Any]]:
    """Assemble verified daily partitions into one hourly Parquet sidecar."""

    root = Path(output_root).expanduser().resolve()
    start = pd.Timestamp(start_day).date()
    end = pd.Timestamp(end_day).date()
    if end < start:
        raise JaoFlowBasedError("end_day doit etre posterieur a start_day.")
    store = root / "flowbased_features.parquet"
    manifest_path = root / "flowbased_features.audit.json"
    if store.is_file() and manifest_path.is_file() and not overwrite:
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing_start = pd.Timestamp(existing["start_day"]).date()
            existing_end = pd.Timestamp(existing["end_day"]).date()
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise JaoFlowBasedError("Manifeste flow-based existant invalide.") from exc
        if sha256_file(store) != existing.get("parquet_sha256"):
            raise JaoFlowBasedError("Checksum du store flow-based existant invalide.")
        if existing.get("schema_version") != FLOWBASED_SCHEMA_VERSION or tuple(
            existing.get("feature_columns", ())
        ) != FLOWBASED_FEATURE_COLUMNS:
            raise JaoFlowBasedError(
                "Store flow-based issu d'un ancien schema; relancer avec --overwrite."
            )
        if existing_start <= start and existing_end >= end:
            if require_research_pit and existing.get(
                "all_partitions_research_pit_eligible",
                existing.get("all_partitions_pit_eligible"),
            ) is not True:
                raise JaoFlowBasedError(
                    "Le store flow-based superset contient une partition non PIT."
                )
            if require_operational_pit and existing.get(
                "all_partitions_operational_pit_eligible"
            ) is not True:
                raise JaoFlowBasedError(
                    "Le store flow-based superset n'est pas une capture "
                    "operationnelle pre-cutoff."
                )
            return store, existing
    partitions: list[pd.DataFrame] = []
    audits: list[Mapping[str, Any]] = []
    for timestamp in pd.date_range(start, end, freq="D"):
        token = timestamp.date().isoformat()
        feature_path = root / "daily_features" / f"{token}.parquet"
        audit_path = root / "raw" / MODEL_INPUT_ENDPOINT / f"{token}.audit.json"
        if not feature_path.is_file() or not audit_path.is_file():
            raise JaoFlowBasedError(f"Partition JAO absente pour {token}.")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if sha256_file(feature_path) != audit.get("features_sha256"):
            raise JaoFlowBasedError(f"Checksum features invalide pour {token}.")
        if audit.get("schema_version") != FLOWBASED_SCHEMA_VERSION or tuple(
            audit.get("feature_columns", ())
        ) != FLOWBASED_FEATURE_COLUMNS:
            raise JaoFlowBasedError(
                f"Partition {token} issue d'un ancien schema; "
                "relancer avec --overwrite."
            )
        if require_research_pit and audit.get("pit_eligible") is not True:
            raise JaoFlowBasedError(
                f"Partition {token} non admissible PIT: {audit.get('pit_status')}."
            )
        if require_operational_pit and audit.get(
            "operational_pit_eligible"
        ) is not True:
            raise JaoFlowBasedError(
                f"Partition {token} non capturee avant le cutoff operationnel."
            )
        partition = pd.read_parquet(feature_path)
        missing_features = sorted(
            set(FLOWBASED_FEATURE_COLUMNS).difference(partition.columns)
        )
        if missing_features:
            raise JaoFlowBasedError(
                f"Partition {token}: features absentes {missing_features}."
            )
        partitions.append(partition)
        audits.append(audit)
    combined = pd.concat(partitions, ignore_index=True).sort_values("value_time_utc")
    index = pd.DatetimeIndex(pd.to_datetime(combined["value_time_utc"], utc=True))
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise JaoFlowBasedError("Store flow-based: timeline dupliquee/non croissante.")
    expected = pd.date_range(
        local_day_utc_bounds(start)[0],
        local_day_utc_bounds(end)[1],
        freq="h",
        inclusive="left",
    )
    if not index.equals(expected):
        raise JaoFlowBasedError("Store flow-based: couverture horaire non contigue.")
    _atomic_parquet(store, combined)
    manifest = {
        "schema_version": FLOWBASED_SCHEMA_VERSION,
        "source": "JAO Core initialComputation / Presolved=true",
        "model_input_stage": MODEL_INPUT_ENDPOINT,
        "audit_only_endpoints": sorted(AUDIT_ONLY_ENDPOINTS),
        "start_day": start.isoformat(),
        "end_day": end.isoformat(),
        "calendar_days": int((end - start).days + 1),
        "physical_hours": int(len(combined)),
        "partial_initial_days": int(
            sum(
                0 < int(item.get("missing_cnec_mtus", 0))
                < int(item.get("physical_mtus", 0))
                for item in audits
            )
        ),
        "empty_initial_fallback_days": int(
            sum(item.get("raw_initial_publication_empty") is True for item in audits)
        ),
        "late_initial_fallback_days": int(
            sum(
                item.get("fallback_reason")
                == "api_last_modified_after_cutoff"
                for item in audits
            )
        ),
        "causal_fallback_days": int(
            sum(item.get("fallback_reason") is not None for item in audits)
        ),
        "imputed_hours": int(
            pd.to_numeric(
                combined["flowbased_hour_imputed"], errors="coerce"
            ).sum()
        ),
        "missing_cnec_mtus": int(
            sum(int(item.get("missing_cnec_mtus", 0)) for item in audits)
        ),
        "feature_columns": list(FLOWBASED_FEATURE_COLUMNS),
        "all_partitions_pit_eligible": bool(
            all(item.get("pit_eligible") is True for item in audits)
        ),
        "all_partitions_research_pit_eligible": bool(
            all(item.get("pit_eligible") is True for item in audits)
        ),
        "all_partitions_operational_pit_eligible": bool(
            all(item.get("operational_pit_eligible") is True for item in audits)
        ),
        "all_partitions_tls_verified": bool(
            all(item.get("tls_verification") is True for item in audits)
        ),
        "tls_trust_sources": sorted(
            {
                str(item.get("tls_trust_source", "unknown"))
                for item in audits
            }
        ),
        "pit_violations": int(
            sum(item.get("pit_eligible") is not True for item in audits)
        ),
        "operational_capture_violations": int(
            sum(
                item.get("operational_pit_eligible") is not True
                for item in audits
            )
        ),
        "historical_vintage_limitation": (
            "PIT qualifie via JAO lastModifiedOn; une archive locale D-1 reste "
            "necessaire avant promotion operationnelle. Les jours entierement "
            "vides utilisent uniquement une publication initiale anterieure "
            "et sont neutralises par le runner."
        ),
        "empty_initial_policy": "causal_previous_initial_then_identity",
        "assembly_requirements": {
            "research_pit": bool(require_research_pit),
            "operational_pit": bool(require_operational_pit),
        },
        "parquet_path": str(store),
        "parquet_sha256": sha256_file(store),
        "parquet_size_bytes": int(store.stat().st_size),
    }
    _atomic_json(manifest_path, manifest)
    return store, manifest


__all__ = [
    "AUDIT_ONLY_ENDPOINTS",
    "DEFAULT_TIMEZONE",
    "FLOWBASED_FEATURE_COLUMNS",
    "GATED_CHALLENGER_ENDPOINT",
    "JaoCoreClient",
    "JaoFetchResult",
    "JaoFlowBasedError",
    "MODEL_INPUT_ENDPOINT",
    "assemble_flowbased_feature_store",
    "build_windows_trust_context",
    "build_causal_empty_day_fallback",
    "build_hourly_flowbased_features",
    "expected_cutoff_utc",
    "local_day_utc_bounds",
    "normalise_initial_computation",
    "sha256_file",
    "write_daily_flowbased_bundle",
]
