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
RECEIPT_PATH = Path("config/nyx_annual_cpu_qualification_receipt.json")
COUNTRIES = frozenset(("FR", "BE", "NL"))


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


def inspect_annual_cpu_results(root: Path, country: str) -> dict[str, Any]:
    """Return pinned replay scores for one country without writing or fitting."""

    if country not in COUNTRIES:
        return _invalid(f"Unsupported annual CPU country: {country}")
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
            and set(summary.get("negative_diagnostics", {})) == COUNTRIES
            and receipt.get("protocol") == "nyx_annual_cpu_price_qualification_v1"
            and receipt.get("first_delivery_day") == "2025-09-24"
            and receipt.get("last_delivery_day") == "2026-09-23"
            and receipt.get("retrospective_evaluation") is True
            and receipt.get("price_expert_replay_qualified") is True
            and receipt.get("negative_replay_verified") is True
            and receipt.get("full_input_chain_qualified") is False
            and receipt.get("qualified") is False
            and set(receipt.get("compositions", {})) == COUNTRIES
            and set(receipt.get("price_country_metrics", {})) == COUNTRIES
            and set(receipt.get("negative_country_metrics", {})) == COUNTRIES
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
