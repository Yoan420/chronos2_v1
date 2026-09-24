"""Immutable solar ablation preregistration; no retrospective promotion or IO sync."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo


ZONES = ("FR", "DE", "BE", "NL")
CANDIDATES = ("solar_residual_standard_kalman", "solar_residual_solar_kalman")
PHASES = ("historical_diagnostic", "postfreeze_retrospective", "prospective")
_CIVIL = ZoneInfo("Europe/Paris")  # All four zones share civil DST transitions.


def _utc_now():
    return datetime.now(timezone.utc)


def _timestamp(value):
    value = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("An auditable timezone-aware timestamp is required")
    return value.astimezone(timezone.utc)


def _day(value):
    return date.fromisoformat(str(value))


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate(protocol):
    content = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if protocol.get("protocol_sha256") != _digest(content):
        raise ValueError("Immutable solar protocol checksum mismatch")


def delivery_cutoff(delivery_date):
    """D-1 08:00 civil, including summer/winter offsets."""
    return datetime.combine(_day(delivery_date) - timedelta(days=1), time(8), _CIVIL)


def create_or_load_protocol(output_root, *, recipe_contract, historical_end="2026-09-19"):
    """Create once from wall clock, or reject a changed recipe on subsequent loads.

    recipe_contract must contain frozen code/config/baseline identities. The local
    digest detects accidental changes, not adversarial rewriting or notarization.
    No caller-supplied freeze date is accepted, and existing files are never rewritten.
    """
    if not isinstance(recipe_contract, dict) or not recipe_contract:
        raise ValueError("A nonempty frozen code/config/baseline contract is required")
    path = Path(output_root) / "solar_correction_protocol.json"
    contract = json.loads(json.dumps(recipe_contract, allow_nan=False))
    if not path.exists():
        now = _utc_now()
        first = now.astimezone(_CIVIL).date() + timedelta(days=1)
        if delivery_cutoff(first) <= now:
            first += timedelta(days=1)
        protocol = {
            "schema_version": 1, "frozen_at_utc": now.isoformat(),
            "historical_diagnostic_start": "2025-09-20",
            "historical_diagnostic_end": _day(historical_end).isoformat(),
            "first_prospective_delivery": first.isoformat(),
            "zones": list(ZONES), "candidates": list(CANDIDATES),
            "recipe_contract": contract, "recipe_contract_sha256": _digest(contract),
            "forecast_cutoff": "D-1 08:00 Europe/Paris; seal strictly before cutoff",
            "metrics": ["mae", "rmse", "daily_mae_win_rate"],
            "win_rate_definition": "fraction of common complete days with candidate MAE < baseline MAE; ties are not wins",
            "spike_thresholds_eur_mwh": [200, 300], "spike_slices_ex_post_only": True,
            "minimum_complete_common_days": 30, "country_specific_selection": False,
            "promotion_allowed": False, "baseline_policy": "identical_frozen_nuclear_chronos",
        }
        protocol["protocol_sha256"] = _digest(protocol)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as stream:
                json.dump(protocol, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
        except FileExistsError:
            pass  # A concurrent creator wins; verify its exact contract below.
    protocol = json.loads(path.read_text(encoding="utf-8"))
    _validate(protocol)
    if (protocol["recipe_contract"] != contract
            or protocol["historical_diagnostic_end"] != _day(historical_end).isoformat()):
        raise ValueError("Solar protocol is frozen; changed recipe or historical boundary rejected")
    return protocol


def classify_delivery(protocol, delivery_date, *, sealed_at=None):
    """A PIT replay without an actual post-freeze, timely seal is retrospective.

    This classifies supplied evidence, not a forecast's nominal origin timestamp.
    The gate additionally requires that evidence to identify this protocol and
    the forecast content. A timestamp on its own is not a verified archive.
    """
    _validate(protocol)
    day = _day(delivery_date)
    if day <= _day(protocol["historical_diagnostic_end"]):
        return PHASES[0]
    if sealed_at is not None:
        seal = _timestamp(sealed_at)
        if (_timestamp(protocol["frozen_at_utc"]) <= seal < delivery_cutoff(day)
                and seal <= _utc_now()):
            return PHASES[2]
    return PHASES[1]


def complete_delivery(delivery_date, timestamps, actual, forecast):
    """Require each physical UTC hour exactly once (23/24/25), all values finite."""
    day = _day(delivery_date)
    start = datetime.combine(day, time(), _CIVIL).astimezone(timezone.utc)
    end = datetime.combine(day + timedelta(days=1), time(), _CIVIL).astimezone(timezone.utc)
    expected = {start + timedelta(hours=i) for i in range(int((end - start).total_seconds() / 3600))}
    stamps, observed, predicted = list(timestamps), list(actual), list(forecast)
    try:
        return (len(stamps) == len(observed) == len(predicted) == len(expected)
                and {_timestamp(stamp) for stamp in stamps} == expected
                and all(math.isfinite(float(value)) for value in observed + predicted))
    except (TypeError, ValueError, OverflowError):
        return False


def prospective_gate(protocol, records):
    """Intersect complete days over both candidates and all zones, never select.

    Each daily record supplies delivery_date, zone, candidate, complete (derived
    with complete_delivery), sealed_at (actual archive creation, NOT model origin),
    protocol_sha256 and forecast_sha256 (content digest from its immutable seal).
    Caller must verify archive bytes against the seal; this pure gate cannot do IO.
    Missing seal evidence or a late replay can only count as retrospective.
    """
    _validate(protocol)
    required = {(zone, candidate) for zone in ZONES for candidate in CANDIDATES}
    days, seen = {}, set()
    today = _utc_now().astimezone(_CIVIL).date()
    for row in records:
        pair = (row["zone"], row["candidate"])
        if pair not in required:
            continue  # Autonomous solar output is diagnostic, not a primary candidate.
        day = _day(row["delivery_date"])
        key = (day, pair)
        if key in seen:
            raise ValueError("Duplicate day/zone/candidate; no ex-post row selection allowed")
        seen.add(key)
        if row.get("complete") is not True or day >= today:
            continue
        sha = row.get("forecast_sha256", "")
        verified = (row.get("protocol_sha256") == protocol["protocol_sha256"]
                    and isinstance(sha, str) and len(sha) == 64
                    and all(char in "0123456789abcdef" for char in sha))
        phase = classify_delivery(protocol, day, sealed_at=row.get("sealed_at") if verified else None)
        days.setdefault(day, {})[pair] = phase
    grouped = {phase: [] for phase in PHASES}
    for day, pairs in sorted(days.items()):
        if set(pairs) == required:
            phases = set(pairs.values())
            phase = next(iter(phases)) if len(phases) == 1 else PHASES[1]
            grouped[phase].append(day.isoformat())
    count = len(grouped[PHASES[2]])
    minimum = protocol["minimum_complete_common_days"]
    return {"common_days": grouped, "common_day_counts": {p: len(d) for p, d in grouped.items()},
            "prospective_complete_days": count, "minimum_complete_common_days": minimum,
            "descriptive_ready": count >= minimum, "remaining_prospective_days": max(0, minimum - count),
            "status": "descriptive_ready" if count >= minimum else "pending_new_prospective_days",
            "promotion_allowed": False, "country_specific_selection": False}
