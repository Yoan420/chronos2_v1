"""Read-only desktop summary of the qualified annual CPU historical replay.

This summary deliberately never offers a live forecast capability. The pinned
receipt covers a retrospective replay; its future input chain is not qualified.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


RESULTS_PATH = Path("config/nyx_annual_cpu_desktop_results.json")
RECEIPT_PATH = Path("config/nyx_annual_cpu_expert_replay_20260928.json")
DE_RESULTS_PATH = Path("config/nyx_annual_cpu_de_results_20260929.json")
LEGACY_COUNTRIES = frozenset(("FR", "BE", "NL"))
COUNTRIES = LEGACY_COUNTRIES | {"DE"}


def _invalid(message: str) -> dict[str, Any]:
    return {
        "available": False,
        "forecast_ready": False,
        "blockers": [{"code": "cpu_replay_results_unavailable", "message": message}],
    }


def _fraction(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def _positive(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _inspect_de(root: Path) -> dict[str, Any]:
    try:
        summary = json.loads((root / RESULTS_PATH).read_text(encoding="utf-8"))
        pin = summary["de_cpu_results"]
        raw = (root / DE_RESULTS_PATH).read_bytes()
        receipt = json.loads(raw)
        if not (pin["path"] == DE_RESULTS_PATH.as_posix()
            and pin["sha256"] == hashlib.sha256(raw).hexdigest()
            and receipt["identity"] == "nyx_de_cpu_annual_results_20260929_v1"
            and receipt["country"] == "DE"
            and receipt["composition"] == "boosting_mean_disagreement20"
            and receipt["first_delivery_day"] == "2025-09-24"
            and receipt["last_delivery_day"] == "2026-09-23"
            and receipt["price_fits"] == 106 and receipt["negative_fits"] == 53
            and receipt["frozen_legacy_baseline_and_reference"] is True
            and receipt["negative_replay_verified"] is True
            and receipt["full_input_chain_qualified"] is False
            and receipt["qualified"] is False
            and receipt["performance_exception_authorized"] is True
            and set(receipt["replay_receipts_sha256"]) == {"price", "negative"}
            and all(_digest(v) for v in receipt["replay_receipts_sha256"].values())
            and set(receipt["output_sha256"]) == {"price", "negative"}
            and all(_digest(v) for v in receipt["output_sha256"].values())):
            return _invalid("Pinned DE CPU replay scope or source evidence differs.")
        price, negative = receipt["price"], receipt["negative"]
        if not (price["hours"] == price["storm_common_hours"] == 8759
            and type(price["strict_wins"]) is int and 0 <= price["strict_wins"] <= 8759
            and _positive(price["rmse"]) and _positive(price["storm_rmse"])
            and _fraction(price["strict_win_rate"])
            and price["strict_win_rate"] == price["strict_wins"] / 8759
            and negative["hours"] == 8760
            and all(_fraction(negative[n]) for n in ("brier", "precision", "recall"))):
            return _invalid("DE annual CPU metrics are incomplete or invalid.")
        meets = price["rmse"] < price["storm_rmse"] and price["strict_win_rate"] > .5
        return {"available": True, "forecast_ready": False,
            "replay_date": receipt["replay_date"], "first_delivery_day": receipt["first_delivery_day"],
            "last_delivery_day": receipt["last_delivery_day"], "retrospective_evaluation": True,
            "price_expert_replay_qualified": meets, "negative_replay_verified": True,
            "full_input_chain_qualified": False, "qualified": False,
            "performance_exception_authorized": True,
            "price": {**price, "composition": receipt["composition"], "both_criteria_met": meets},
            "negative": negative,
            "blockers": [{"code": "future_input_chain_unqualified",
                "message": "La chaîne complète des entrées futures n'est pas qualifiée : prévision annuelle indisponible."}]}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return _invalid("Le complément annuel CPU DE n'est pas disponible ou son empreinte a changé.")


def inspect_annual_cpu_results(root: Path, country: str) -> dict[str, Any]:
    """Return pinned replay scores for one country without writing or fitting."""

    if country not in COUNTRIES:
        return _invalid(f"Unsupported annual CPU country: {country}")
    if country == "DE":
        return _inspect_de(root)
    try:
        summary = json.loads((root / RESULTS_PATH).read_text(encoding="utf-8"))
        receipt_bytes = (root / RECEIPT_PATH).read_bytes()
        receipt = json.loads(receipt_bytes)
    except (OSError, ValueError) as exc:
        return _invalid(f"Cannot load the annual CPU results: {exc}")
    if not isinstance(summary, dict) or not isinstance(receipt, dict):
        return _invalid("Annual CPU results must be JSON objects.")
    try:
        contract_valid = (
            summary.get("schema_version") == 1
            and summary.get("identity") == "nyx_annual_cpu_desktop_results_2026-09-28_v1"
            and summary.get("replay_date") == "2026-09-28"
            and summary.get("qualification_receipt") == RECEIPT_PATH.as_posix()
            and summary.get("qualification_receipt_sha256") == hashlib.sha256(receipt_bytes).hexdigest()
            and summary.get("negative_replay_receipt_sha256")
            == receipt.get("replay_receipts_sha256", {}).get("negative")
            and set(summary.get("negative_diagnostics", {})) == LEGACY_COUNTRIES
            and receipt.get("protocol") == "nyx_annual_cpu_price_qualification_v1"
            and receipt.get("first_delivery_day") == "2025-09-24"
            and receipt.get("last_delivery_day") == "2026-09-23"
            and receipt.get("retrospective_evaluation") is True
            and receipt.get("price_expert_replay_qualified") is True
            and receipt.get("negative_replay_verified") is True
            and receipt.get("full_input_chain_qualified") is False
            and receipt.get("qualified") is False
            and set(receipt.get("compositions", {})) == LEGACY_COUNTRIES
            and set(receipt.get("price_country_metrics", {})) == LEGACY_COUNTRIES
            and set(receipt.get("negative_country_metrics", {})) == LEGACY_COUNTRIES
        )
    except (AttributeError, TypeError, ValueError):
        contract_valid = False
    if not contract_valid:
        return _invalid("Pinned annual CPU receipt, schema, or qualification state differs.")
    try:
        price = receipt["price_country_metrics"][country]
        negative = receipt["negative_country_metrics"][country]
        diagnostics = summary["negative_diagnostics"][country]
        if not (
            price["hours"] == price["storm_common_hours"] == 8759
            and type(price["strict_wins"]) is int
            and 0 <= price["strict_wins"] <= 8759
            and _positive(price["rmse"])
            and _positive(price["storm_rmse"])
            and _fraction(price["strict_win_rate"])
            and negative["hours"] == 8760
            and _fraction(negative["brier"])
            and _fraction(diagnostics["precision"])
            and _fraction(diagnostics["recall"])
        ):
            raise ValueError("invalid CPU metric")
    except (KeyError, TypeError, ValueError):
        return _invalid("Annual CPU replay metrics are incomplete or invalid.")
    return {
        "available": True,
        "forecast_ready": False,
        "replay_date": summary["replay_date"],
        "first_delivery_day": receipt["first_delivery_day"],
        "last_delivery_day": receipt["last_delivery_day"],
        "retrospective_evaluation": True,
        "price_expert_replay_qualified": True,
        "negative_replay_verified": True,
        "full_input_chain_qualified": False,
        "qualified": False,
        "price": {
            "composition": receipt["compositions"][country],
            "hours": price["hours"],
            "rmse": price["rmse"],
            "storm_rmse": price["storm_rmse"],
            "strict_wins": price["strict_wins"],
            "strict_win_rate": price["strict_win_rate"],
        },
        "negative": {
            "hours": negative["hours"],
            "brier": negative["brier"],
            "precision": diagnostics["precision"],
            "recall": diagnostics["recall"],
        },
        "blockers": [{
            "code": "future_input_chain_unqualified",
            "message": "La chaîne complète des entrées futures n'est pas qualifiée : prévision annuelle indisponible.",
        }],
    }
