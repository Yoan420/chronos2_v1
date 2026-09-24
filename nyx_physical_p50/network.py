"""Strict initial-domain directional descriptors, never import capacities.

Only original JAO ``initialComputation`` records enter these features. The
initial RAM is RefProg-referenced: neither its reference nor the complete
external boundaries are qualified for dispatch. Signed PTDF differences can
describe exposure to a hypothetical transfer, but do not identify binding
constraints, deliverable imports, or the cause of an observed price spike.

Historical last-modified watermarks and actual pre-cutoff captures are reported
separately. Missing, late, quarter-hour or incomplete source hours abstain;
the old interpolated Kalman feature store is never read.
"""
from __future__ import annotations

from datetime import date
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from chronos2_hourly.jao_flowbased import (
    JAO_CORE_DATA_URL, JaoCoreClient, build_windows_trust_context,
    expected_cutoff_utc, local_day_utc_bounds,
)
from marginal_cost_expert.network import (
    NetworkContractError, load_network_day, normalise_network_payload, _empty_day,
)
from chronos2_hourly.process_lock import exclusive_process_lock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = PROJECT_ROOT / "runs" / "experiments" / "nyx_physical_p50_v1"
DEFAULT_RAW_ROOTS = (
    PROJECT_ROOT / "data" / "pit" / "jao_core_flowbased",
    PROJECT_ROOT / "data" / "pit" / "marginal_cost_expert_v2" / "network",
    NAMESPACE,
)
ZONES = ("FR", "DE", "BE", "NL")
PREFIX = "feature_network_"
SENSITIVITY_EPSILON = 1e-9
LOW_RAM_MW = 500.0
MAX_CAPTURE_DAYS = 30
PAIR_FIELDS = (
    "tightening_fraction", "relieving_fraction", "signed_ptdf_p10", "signed_ptdf_p90",
    "tightening_sensitivity_p90", "tightening_ram_p10_mw",
    "tightening_negative_ram_share", "tightening_low_ram_share", "relieving_ram_p10_mw",
)
GLOBAL_FIELDS = (
    "constraint_count", "physical_cnec_count", "external_constraint_count",
    "raw_ram_p10_mw", "raw_ram_median_mw", "raw_negative_ram_share",
    "peer_tightening_fraction_mean", "peer_tightening_sensitivity_max",
    "peer_tightening_ram_p10_min_mw", "peer_negative_ram_share_max",
    "peer_low_ram_share_max", "peer_low_ram_share_range",
)
VALUE_COLUMNS = tuple(PREFIX + name for name in GLOBAL_FIELDS) + tuple(
    PREFIX + "from_" + peer.lower() + "_" + name for peer in ZONES for name in PAIR_FIELDS
)
FEATURE_COLUMNS = VALUE_COLUMNS + tuple(name + "__missing" for name in VALUE_COLUMNS)
METADATA_COLUMNS = (
    "network_eligible", "network_operational_capture_eligible", "network_source_hour_present",
)


def _utc(values: pd.Series, name: str) -> pd.Series:
    if values.isna().any() or any(pd.Timestamp(value).tzinfo is None for value in values):
        raise NetworkContractError(f"{name}: complete timezone-aware timestamps required")
    return pd.to_datetime(values, utc=True, errors="raise").reset_index(drop=True)


def _panel_identity(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"zone", "timestamp_utc", "forecast_origin_utc"}
    if (not isinstance(panel, pd.DataFrame) or panel.empty or panel.columns.has_duplicates
            or not required.issubset(panel)):
        raise NetworkContractError("Nonempty unique-column panel with country/hour/origin required")
    if not panel.zone.map(lambda value: isinstance(value, str) and value in ZONES).all():
        raise NetworkContractError("Only exact FR, DE, BE, NL country identities are allowed")
    stamps = _utc(panel.timestamp_utc, "timestamp_utc")
    origins = _utc(panel.forecast_origin_utc, "forecast_origin_utc")
    result = pd.DataFrame({"zone": panel.zone.to_numpy(), "timestamp_utc": stamps,
                           "forecast_origin_utc": origins})
    if (not stamps.eq(stamps.dt.floor("h")).all()
            or result.duplicated(["zone", "timestamp_utc"]).any()):
        raise NetworkContractError("Unique physical hourly country/delivery identities required")
    local_day = stamps.dt.tz_convert("Europe/Paris").dt.tz_localize(None).dt.normalize()
    expected = (local_day - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).dt.tz_localize(
        "Europe/Paris", ambiguous="raise", nonexistent="raise").dt.tz_convert("UTC")
    if not origins.eq(expected).all():
        raise NetworkContractError("Exact D-1 08:00 Europe/Paris civil origin required")
    result["day"] = local_day.dt.strftime("%Y-%m-%d")
    return result


def _dataset_root(value: str | Path) -> Path:
    """Accept a dataset root or its explicit ``raw`` directory, not a glob."""
    root = Path(value).resolve()
    return root.parent if root.name == "raw" else root


def _selected_root(roots: Sequence[Path], day: str) -> Path:
    # Last root wins if *either* member exists. A broken newer pair never
    # silently falls back to an older, numerically convenient source.
    for root in reversed(roots):
        directory = root / "raw" / "initialComputation"
        if any((directory / (day + suffix)).exists() for suffix in (".json.gz", ".audit.json")):
            return root
    return roots[-1]


def _normalise_payload(payload, day, **kwargs):
    """An explicitly empty API result has no mandatory modification watermark.

    The legacy physical-domain parser expects a nonempty revision timestamp.
    Validate an empty envelope here and abstain, without inventing such a date
    or changing that sealed parser's behaviour for any nonempty domain.
    """
    if payload.get("records") != []:
        return normalise_network_payload(payload, day, **kwargs)
    fetch = payload.get("fetch", {})
    start, end = local_day_utc_bounds(day, timezone="Europe/Paris")
    def stamp(name):
        value = pd.Timestamp(fetch.get(name))
        if pd.isna(value) or value.tzinfo is None:
            raise NetworkContractError(f"Empty API {name}: explicit timezone required")
        return value.tz_convert("UTC")
    if (fetch.get("endpoint") != "initialComputation" or fetch.get("api_base_url") != JAO_CORE_DATA_URL
            or fetch.get("filters", {}).get("Presolved") is not True
            or type(fetch.get("total_rows")) is not int or fetch["total_rows"] != 0
            or stamp("start_utc") != start or stamp("end_utc") != end):
        raise NetworkContractError("Invalid empty initial-domain source identity, bounds or count")
    retrieved = stamp("retrieved_at_utc")
    modified = fetch.get("last_modified_utc")
    if modified is not None and stamp("last_modified_utc") > retrieved:
        raise NetworkContractError("Empty API modification is after retrieval")
    result = _empty_day(pd.Timestamp(day).date(), "Europe/Paris", "api_empty_original_domain")
    result.audit.update(api_empty_result=True, retrieved_at_utc=retrieved.isoformat(),
                        last_modified_utc=modified, original_api_total_rows=0)
    return result


def _load_day(root, day, **kwargs):
    directory = Path(root)/"raw"/"initialComputation"
    raw, seal = directory/(day+".json.gz"), directory/(day+".audit.json")
    if not raw.is_file() or not seal.is_file():
        return load_network_day(root, day, **kwargs)
    blob, audit_blob = raw.read_bytes(), seal.read_bytes()
    checksum = hashlib.sha256(blob).hexdigest()
    audit = json.loads(audit_blob)
    if audit.get("raw_gzip_sha256") != checksum:
        raise NetworkContractError("Raw checksum mismatch: " + str(raw))
    payload = json.loads(gzip.decompress(blob))
    if payload.get("records") != []:
        return load_network_day(root, day, **kwargs)
    result = _normalise_payload(payload, day, **kwargs)
    result.audit.update(raw_path=str(raw.resolve()), source_audit_path=str(seal.resolve()),
        raw_gzip_sha256=checksum, source_audit_sha256=hashlib.sha256(audit_blob).hexdigest())
    return result


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability)) if len(values) else math.nan


def _finite_summary(values: Sequence[float], kind: str) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan
    return float({"max": np.max, "min": np.min, "mean": np.mean, "range": np.ptp}[kind](values))


def directional_descriptors(constraints: pd.DataFrame, zone: str) -> dict[str, float]:
    """Describe one qualified original hour; no optimization or MW conversion.

    For an injection in ``peer`` and withdrawal in ``zone``, positive
    ``PTDF_peer - PTDF_zone`` tightens the recorded directed constraint.
    Negative RAM is retained as negative; it is never absolutized, clipped,
    or divided by a sensitivity to manufacture a transfer capacity.
    """
    required = {"ram_mw", "constraint_kind", *["ptdf_" + z for z in ZONES]}
    if zone not in ZONES or constraints.empty or not required.issubset(constraints):
        raise NetworkContractError("Qualified nonempty RAM/PTDF hour and supported zone required")
    numeric = constraints[["ram_mw", *["ptdf_" + z for z in ZONES]]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise NetworkContractError("Missing/nonfinite RAM/PTDF must abstain before feature extraction")
    ram = numeric[:, 0]
    own = constraints["ptdf_" + zone].to_numpy(dtype=float)
    output = {name: math.nan for name in VALUE_COLUMNS}
    plain = {
        "constraint_count": float(len(constraints)),
        "physical_cnec_count": float(constraints.constraint_kind.eq("cnec").sum()),
        "external_constraint_count": float(constraints.constraint_kind.eq("external_constraint").sum()),
        "raw_ram_p10_mw": _quantile(ram, .1), "raw_ram_median_mw": _quantile(ram, .5),
        "raw_negative_ram_share": float(np.mean(ram < 0)),
    }
    pairs = []
    for peer in ZONES:
        if peer == zone:
            continue
        response = constraints["ptdf_" + peer].to_numpy(dtype=float) - own
        tightening = response > SENSITIVITY_EPSILON
        relieving = response < -SENSITIVITY_EPSILON
        exposed = ram[tightening]
        pair = {
            "tightening_fraction": float(np.mean(tightening)),
            "relieving_fraction": float(np.mean(relieving)),
            "signed_ptdf_p10": _quantile(response, .1), "signed_ptdf_p90": _quantile(response, .9),
            "tightening_sensitivity_p90": _quantile(response[tightening], .9),
            "tightening_ram_p10_mw": _quantile(exposed, .1),
            "tightening_negative_ram_share": float(np.mean(exposed < 0)) if len(exposed) else math.nan,
            "tightening_low_ram_share": float(np.mean(exposed < LOW_RAM_MW)) if len(exposed) else math.nan,
            "relieving_ram_p10_mw": _quantile(ram[relieving], .1),
        }
        output.update({PREFIX + "from_" + peer.lower() + "_" + name: value for name, value in pair.items()})
        pairs.append(pair)
    for output_name, source_name, operation in (
        ("peer_tightening_fraction_mean", "tightening_fraction", "mean"),
        ("peer_tightening_sensitivity_max", "tightening_sensitivity_p90", "max"),
        ("peer_tightening_ram_p10_min_mw", "tightening_ram_p10_mw", "min"),
        ("peer_negative_ram_share_max", "tightening_negative_ram_share", "max"),
        ("peer_low_ram_share_max", "tightening_low_ram_share", "max"),
        ("peer_low_ram_share_range", "tightening_low_ram_share", "range"),
    ):
        plain[output_name] = _finite_summary([pair[source_name] for pair in pairs], operation)
    output.update({PREFIX + name: value for name, value in plain.items()})
    return output


def prepare_network_features(
    panel: pd.DataFrame, raw_roots: Sequence[str | Path] | None = None,
    *, require_operational_capture: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read each original day once and return feature-only rows on panel.index.

    Roots are ordered lowest to highest precedence. Only the corresponding
    physical source hour can supply a panel row. A partially published day may
    therefore supply its valid hours, never its missing hours. Malformed or
    tampered archives raise; legitimate absence/late publication abstains.
    """
    identity = _panel_identity(panel)
    roots = [_dataset_root(root) for root in (DEFAULT_RAW_ROOTS if raw_roots is None else raw_roots)]
    if not roots or len(set(roots)) != len(roots):
        raise NetworkContractError("A nonempty ordered list of distinct original raw roots is required")
    if not isinstance(require_operational_capture, bool):
        raise NetworkContractError("require_operational_capture must be boolean")
    values = pd.DataFrame(np.nan, index=range(len(panel)), columns=VALUE_COLUMNS, dtype=float)
    metadata = pd.DataFrame(False, index=range(len(panel)), columns=METADATA_COLUMNS, dtype=bool)
    daily = []
    source_files: dict[str, str] = {}
    for day, positions in identity.groupby("day", sort=True).groups.items():
        root = _selected_root(roots, day)
        original = _load_day(root, day, cutoff_time="08:00",
                                    require_operational_capture=require_operational_capture)
        quality = original.qualification.set_index("delivery_start_utc")
        grouped = {stamp: block for stamp, block in original.constraints.groupby("delivery_start_utc", sort=False)}
        for (zone, stamp), subset in identity.loc[positions].groupby(["zone", "timestamp_utc"], sort=False):
            q = quality.loc[stamp]
            idx = subset.index
            metadata.loc[idx, "network_source_hour_present"] = bool(q.constraint_count > 0)
            metadata.loc[idx, "network_eligible"] = bool(q.inputs_qualified)
            metadata.loc[idx, "network_operational_capture_eligible"] = bool(q.operational_pit_eligible)
            if q.inputs_qualified:
                descriptors = directional_descriptors(grouped[stamp], zone)
                values.loc[idx, list(descriptors)] = np.asarray(list(descriptors.values()), dtype=float)
        for path_key, sha_key in (("raw_path", "raw_gzip_sha256"), ("source_audit_path", "source_audit_sha256")):
            if original.audit.get(path_key):
                source_files[original.audit[path_key]] = original.audit[sha_key]
        daily.append({"delivery_day": day, "selected_root": str(root),
                      "requested_country_hours": len(positions),
                      "eligible_country_hours": int(metadata.loc[positions, "network_eligible"].sum()),
                      "source_audit": original.audit})
    # Seal what was actually read and fail closed on concurrent replacement.
    for path, expected in source_files.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise NetworkContractError("Original network source changed during feature extraction: " + path)
    flags = values.isna().astype(float).rename(columns=lambda name: name + "__missing")
    flags.loc[~metadata.network_eligible, :] = np.nan
    output = pd.concat([values, flags, metadata], axis=1)
    output.index = panel.index.copy()
    audit = {
        "schema_version": 1, "kind": "initial_domain_directional_vulnerability_v1",
        "endpoint": "initialComputation", "api_base_url": JAO_CORE_DATA_URL, "presolved_filter": True,
        "cutoff": "D-1 08:00 Europe/Paris civil", "raw_roots_low_to_high_precedence": list(map(str, roots)),
        "source_files": source_files, "days": daily, "rows": len(panel),
        "eligible_rows": int(metadata.network_eligible.sum()),
        "operational_capture_rows": int(metadata.network_operational_capture_eligible.sum()),
        "require_operational_capture": require_operational_capture,
        "feature_columns": list(FEATURE_COLUMNS), "value_columns": list(VALUE_COLUMNS),
        "metadata_columns": list(METADATA_COLUMNS), "original_panel_columns_read": ["zone", "timestamp_utc", "forecast_origin_utc"],
        "ptdf_direction": "peer injection minus local withdrawal: PTDF_peer - PTDF_zone",
        "tightening_epsilon": SENSITIVITY_EPSILON, "low_ram_threshold_mw": LOW_RAM_MW,
        "conditional_empty_exposure": "NaN, not zero headroom; same-country peer fields are not applicable",
        "initial_ram_reference": "RefProg-balanced raw; not translated or qualified for zero-net-position dispatch",
        "domain_reference_qualified": False, "boundary_qualified": False,
        "feasible_imports_computed": False, "constraint_binding_claimed": False,
        "physical_capacity_or_label_normalization_used": False, "imputation_performed": False,
        "previous_day_fallback_used": False, "post_coupling_features_used": False,
        "production_pit_evidence": False, "publication_timestamp_certified": False,
        "source_semantics": "Historical API lastModifiedOn watermark is not a certified contemporaneous publication archive",
        "source_override_policy": "Latest configured root with either archive member; malformed/partial pairs never fall back",
    }
    return output, audit


def _day(value: str | date) -> str:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.normalize():
        raise NetworkContractError("Capture dates must be timezone-naive civil date labels")
    return stamp.strftime("%Y-%m-%d")


def _capture_root(value: str | Path) -> Path:
    root = Path(value).resolve()
    allowed = (NAMESPACE / "raw").resolve()
    if root != allowed:
        raise NetworkContractError("New network captures must use exactly nyx_physical_p50_v1/raw")
    return root


def _client_factory() -> JaoCoreClient:
    context, _ = build_windows_trust_context()
    return JaoCoreClient(verify=context, timeout_seconds=45., maximum_retries=2,
                         request_interval_seconds=.65, page_size=40000)


def capture_network_days(
    days: Iterable[str | date], *, output_root: str | Path = NAMESPACE / "raw",
    max_days: int = MAX_CAPTURE_DAYS, client_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Capture a bounded set of days in the new namespace, safely resumable.

    One TLS-verified pooled client, fixed initial endpoint/filters, two bounded
    retries per request. Existing pairs are validated and reused without any
    request; incomplete/corrupt pairs fail closed. This never runs implicitly
    during feature preparation and never changes an operational source bank.
    """
    if isinstance(days, (str, bytes)):
        raise NetworkContractError("Pass an explicit bounded list of civil dates")
    chosen = [_day(day) for day in days]
    if (isinstance(max_days, bool) or not isinstance(max_days, int)
            or not 1 <= max_days <= MAX_CAPTURE_DAYS or not chosen or len(chosen) > max_days
            or len(set(chosen)) != len(chosen)):
        raise NetworkContractError(f"Choose 1..{MAX_CAPTURE_DAYS} distinct capture dates, within max_days")
    chosen.sort()
    root = _capture_root(output_root)
    endpoint_dir = root / "initialComputation"
    endpoint_dir.mkdir(parents=True, exist_ok=True)
    # Reuse the OS-held project lock: a proven-dead owner can be recovered,
    # while corrupt/foreign/inaccessible owner records remain fail-closed.
    lock_path = root / ".capture.lock"
    lock = exclusive_process_lock(lock_path)
    lock.__enter__()
    records, client = [], None
    try:
        for day in chosen:
            raw_path, audit_path = endpoint_dir / (day + ".json.gz"), endpoint_dir / (day + ".audit.json")
            if raw_path.exists() or audit_path.exists():
                # Reuse the strict source validator, then require the private
                # collector contract to rule out silently changing requests.
                loaded = _load_day(root.parent, day)
                saved = json.loads(audit_path.read_text(encoding="utf-8"))
                if saved.get("capture_contract") != _capture_contract(day):
                    raise NetworkContractError("Existing capture request contract differs: " + day)
                records.append({"day": day, "reused": True, "audit": loaded.audit})
                continue
            if client is None:
                client = (client_factory or _client_factory)()
            fetched = client.fetch_initial_day(day, timezone="Europe/Paris")
            payload = {"schema_version": 1, "source": "JAO Core production initialComputation",
                       "fetch": fetched.audit_dict(), "records": list(fetched.rows)}
            validated = _normalise_payload(payload, day, cutoff_time="08:00")
            blob = gzip.compress(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                            allow_nan=False).encode("utf-8"), mtime=0)
            audit = {**validated.audit, "raw_gzip_sha256": hashlib.sha256(blob).hexdigest(),
                     "capture_contract": _capture_contract(day), "tls_verified": True,
                     "request_timeout_seconds": 45., "maximum_retries": 2,
                     "collection_context": "isolated research; historical retrieval is not capture at forecast origin"}
            # Exclusive writes: a process crash may leave a partial pair; a
            # subsequent attempt refuses it instead of replacing audit evidence.
            with raw_path.open("xb") as stream:
                stream.write(blob)
            with audit_path.open("x", encoding="utf-8") as stream:
                json.dump(audit, stream, sort_keys=True, indent=2, allow_nan=False)
            records.append({"day": day, "reused": False, "audit": _load_day(root.parent, day).audit})
    finally:
        try:
            if client is not None:
                client.close()
        finally:
            lock.__exit__(None, None, None)
    return {"schema_version": 1, "status": "complete", "output_root": str(root),
            "days": records, "new_days": sum(not record["reused"] for record in records),
            "reused_days": sum(record["reused"] for record in records), "production_modified": False,
            "production_pit_evidence": False, "source_endpoint": "initialComputation"}


def _capture_contract(day: str) -> dict[str, Any]:
    return {"schema_version": 1, "api_base_url": JAO_CORE_DATA_URL, "endpoint": "initialComputation",
            "filters": {"Presolved": True}, "delivery_day": day, "timezone": "Europe/Paris",
            "cutoff_utc": expected_cutoff_utc(day).isoformat(), "imputation": False}


__all__ = ["prepare_network_features", "capture_network_days", "directional_descriptors",
           "FEATURE_COLUMNS", "VALUE_COLUMNS", "METADATA_COLUMNS", "DEFAULT_RAW_ROOTS", "PREFIX"]
