"""Read-only audit of the history needed by the strict LoRA OOF pipeline.

The residual calibration contract needs three consecutive 365-day blocks:
an initial fit window, an out-of-fold calibration window and a final holdout.
This module inventories the exact local artefacts used by the exogenous model.
It never downloads data and never upgrades a historical as-of reconstruction
to operational/prospective PIT evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .feature_bank import (
    ExogenousBankError,
    build_exogenous_bank,
    default_project_sources,
    delivery_utc_index,
)
from .panel import load_target_cache


class HistoryRecoveryAuditError(ValueError):
    """Raised when an audit request or local artefact is invalid."""


@dataclass(frozen=True)
class RecoveryWindow:
    """Exact chronological support required by the OOF contract."""

    end_day: pd.Timestamp
    first_delivery_day: pd.Timestamp
    bank_materialization_start_day: pd.Timestamp
    first_origin_utc: pd.Timestamp
    first_context_utc: pd.Timestamp
    last_delivery_utc: pd.Timestamp
    training_days: int
    oof_days: int
    holdout_days: int
    context_length: int
    timezone: str

    @property
    def required_origins(self) -> int:
        return self.training_days + self.oof_days + self.holdout_days

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_days": self.training_days,
            "oof_days": self.oof_days,
            "holdout_days": self.holdout_days,
            "required_origins": self.required_origins,
            "delivery_start": self.first_delivery_day.date().isoformat(),
            "delivery_end": self.end_day.date().isoformat(),
            "bank_materialization_start_day": (
                self.bank_materialization_start_day.date().isoformat()
            ),
            "first_origin_utc": self.first_origin_utc.isoformat(),
            "first_context_utc": self.first_context_utc.isoformat(),
            "last_delivery_utc": self.last_delivery_utc.isoformat(),
            "context_length_hours": self.context_length,
            "timezone": self.timezone,
            "exogenous_context_policy": (
                "NaN autorises dans le contexte historique; chaque horizon D+1 "
                "doit rester complet"
            ),
        }


def derive_recovery_window(
    end_day: str | pd.Timestamp,
    *,
    training_days: int = 365,
    oof_days: int = 365,
    holdout_days: int = 365,
    context_length: int = 2048,
    timezone: str = "Europe/Paris",
) -> RecoveryWindow:
    """Derive inclusive delivery bounds and the exact first context hour."""

    integers = {
        "training_days": training_days,
        "oof_days": oof_days,
        "holdout_days": holdout_days,
        "context_length": context_length,
    }
    for name, value in integers.items():
        if isinstance(value, bool) or int(value) <= 0:
            raise HistoryRecoveryAuditError(f"{name} doit etre un entier positif.")
    end = pd.Timestamp(end_day)
    if end.tzinfo is not None:
        end = end.tz_convert(timezone).tz_localize(None)
    end = end.normalize()
    required = int(training_days) + int(oof_days) + int(holdout_days)
    first = end - pd.Timedelta(days=required - 1)
    first_horizon = delivery_utc_index(first, first, timezone=timezone)
    last_horizon = delivery_utc_index(end, end, timezone=timezone)
    first_context = first_horizon[0] - pd.Timedelta(hours=int(context_length))
    # Keep this identical to run_chronos2_exogenous_panel.py: the small civil
    # margin makes timezone/DST boundary materialisation unambiguous.
    context_margin_days = (int(context_length) + 23) // 24 + 2
    bank_start = first - pd.Timedelta(days=context_margin_days)
    local_origin = first - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
    first_origin = local_origin.tz_localize(timezone).tz_convert("UTC")
    return RecoveryWindow(
        end_day=end,
        first_delivery_day=first,
        bank_materialization_start_day=bank_start,
        first_origin_utc=first_origin,
        first_context_utc=first_context,
        last_delivery_utc=last_horizon[-1],
        training_days=int(training_days),
        oof_days=int(oof_days),
        holdout_days=int(holdout_days),
        context_length=int(context_length),
        timezone=str(timezone),
    )


def _daily_coverage(
    available: pd.Series,
    *,
    timezone: str,
) -> dict[str, Any]:
    flags = available.fillna(False).astype(bool)
    days = pd.Index(pd.DatetimeIndex(flags.index).tz_convert(timezone).date)
    complete_by_day = flags.groupby(days).all().astype(bool)
    complete_days = [day for day, complete in complete_by_day.items() if complete]
    suffix_start = None
    for day, complete in reversed(list(complete_by_day.items())):
        if not bool(complete):
            break
        suffix_start = day
    missing_days = [
        day.isoformat() for day, complete in complete_by_day.items() if not complete
    ]
    return {
        "expected_days": int(len(complete_by_day)),
        "complete_days": int(complete_by_day.sum()),
        "first_complete_day": (
            complete_days[0].isoformat() if complete_days else None
        ),
        "last_complete_day": (
            complete_days[-1].isoformat() if complete_days else None
        ),
        "contiguous_complete_suffix_start": (
            suffix_start.isoformat() if suffix_start is not None else None
        ),
        "missing_day_count": int((~complete_by_day).sum()),
        "first_missing_days": missing_days[:10],
    }


def _source_sidecar_bounds(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"audit_path": str(path) if path is not None else None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"audit_path": str(path), "audit_error": str(exc)}
    return {
        "audit_path": str(path),
        "artifact_start_day": payload.get("start_day"),
        "artifact_end_day": payload.get("end_day"),
        "first_delivery_utc": payload.get("first_delivery_utc"),
        "last_delivery_utc": payload.get("last_delivery_utc"),
        "provider_revision_timestamp_available": payload.get(
            "provider_revision_timestamp_available"
        ),
        "historical_vintage_limitation": payload.get(
            "historical_vintage_limitation"
        ),
        "operational_capture_violations": payload.get(
            "operational_capture_violations"
        ),
    }


def _target_audit(
    path: Path,
    *,
    zone: str,
    window: RecoveryWindow,
) -> dict[str, Any]:
    target = load_target_cache(path, zone=zone)
    horizon = delivery_utc_index(
        window.first_delivery_day,
        window.end_day,
        timezone=window.timezone,
    )
    context = pd.date_range(
        start=window.first_context_utc,
        periods=window.context_length,
        freq="h",
    )
    expected = context.append(horizon)
    aligned = pd.to_numeric(target.reindex(expected), errors="coerce")
    finite = pd.Series(
        np.isfinite(aligned.to_numpy(dtype=float)), index=expected, dtype=bool
    )
    return {
        "zone": zone,
        "path": str(path),
        "source_first_utc": target.index.min().isoformat(),
        "source_last_utc": target.index.max().isoformat(),
        "required_hours": int(len(expected)),
        "complete_hours": int(finite.sum()),
        "missing_hours": int((~finite).sum()),
        "covers_context_and_horizons": bool(finite.all()),
        "first_required_utc": expected[0].isoformat(),
        "last_required_utc": expected[-1].isoformat(),
    }


def audit_local_history(
    project_root: str | Path,
    *,
    target_paths: Mapping[str, str | Path],
    end_day: str | pd.Timestamp,
    zones: Sequence[str] = ("FR", "DE", "BE", "NL"),
    pack: str = "full",
    training_days: int = 365,
    oof_days: int = 365,
    holdout_days: int = 365,
    context_length: int = 2048,
    timezone: str = "Europe/Paris",
) -> dict[str, Any]:
    """Audit local source coverage without fetching or mutating any artefact."""

    root = Path(project_root).expanduser().resolve()
    selected_zones = tuple(dict.fromkeys(str(zone).strip().upper() for zone in zones))
    if not selected_zones:
        raise HistoryRecoveryAuditError("Au moins une zone est requise.")
    missing_targets = sorted(set(selected_zones).difference(target_paths))
    if missing_targets:
        raise HistoryRecoveryAuditError(f"Cibles absentes: {missing_targets}.")
    window = derive_recovery_window(
        end_day,
        training_days=training_days,
        oof_days=oof_days,
        holdout_days=holdout_days,
        context_length=context_length,
        timezone=timezone,
    )
    targets = [
        _target_audit(
            Path(target_paths[zone]).expanduser().resolve(),
            zone=zone,
            window=window,
        )
        for zone in selected_zones
    ]

    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for zone in selected_zones:
        for source in default_project_sources(root, zone=zone, pack=pack):
            key = (
                str(Path(source.path).expanduser().resolve()),
                str(source.cutoff_timezone or timezone),
                source.name,
            )
            if key in seen:
                continue
            seen.add(key)
            record: dict[str, Any] = {
                "zone": zone if source.family == "weather" else "COMMON",
                "name": source.name,
                "family": source.family,
                "path": str(Path(source.path).expanduser().resolve()),
                "production_evidence_kind": source.production_evidence_kind,
                **_source_sidecar_bounds(source.audit_path),
            }
            try:
                bank = build_exogenous_bank(
                    (source,),
                    start_day=window.first_delivery_day,
                    end_day=window.end_day,
                    timezone=timezone,
                    require_complete=False,
                    require_operational_evidence=False,
                )
                details = dict(bank.audit["sources"][source.name])
                available_column = f"{source.name}__available"
                daily = _daily_coverage(
                    bank.frame[available_column].eq(1.0), timezone=timezone
                )
                record.update(
                    {
                        "expected_hours": int(details["expected_hours"]),
                        "complete_hours": int(details["complete_hours"]),
                        "missing_hours": int(details["missing_hours"]),
                        "minimum_column_coverage": float(
                            details["minimum_column_coverage"]
                        ),
                        "causality_violations": int(
                            details["causality_violations"]
                        ),
                        "production_pit_evidence": details[
                            "production_pit_evidence"
                        ],
                        "source_audit_classification": (
                            details.get("source_audit") or {}
                        ).get("classification"),
                        "covers_required_horizons": bool(
                            details["missing_hours"] == 0
                        ),
                        **daily,
                    }
                )
            except (ExogenousBankError, OSError, ValueError) as exc:
                record.update(
                    {
                        "audit_error": str(exc),
                        "covers_required_horizons": False,
                        "production_pit_evidence": False,
                    }
                )
            sources.append(record)

    target_ready = all(item["covers_context_and_horizons"] for item in targets)
    source_ready = all(item["covers_required_horizons"] for item in sources)
    production_ready = bool(
        target_ready
        and source_ready
        and all(item.get("production_pit_evidence") is True for item in sources)
    )
    blockers = [
        f"target:{item['zone']} missing_hours={item['missing_hours']}"
        for item in targets
        if not item["covers_context_and_horizons"]
    ]
    blockers.extend(
        f"{item['zone']}:{item['name']} missing_hours={item.get('missing_hours', 'unknown')}"
        for item in sources
        if not item["covers_required_horizons"]
    )
    production_blockers = [
        f"{item['zone']}:{item['name']} sans capture prospective verifiee"
        for item in sources
        if item.get("production_pit_evidence") is not True
    ]
    return {
        "schema_version": 1,
        "purpose": "strict_lora_residual_oof_history_recovery_audit",
        "network_access": False,
        "mutates_source_artifacts": False,
        "pack": pack,
        "zones": list(selected_zones),
        "window": window.to_dict(),
        "targets": targets,
        "sources": sources,
        "research_horizon_ready": bool(target_ready and source_ready),
        "production_ready": production_ready,
        "blockers": blockers,
        "production_blockers": production_blockers,
        "evidence_scope": {
            "historical_asof": (
                "Recherche seulement: revision_date=cutoff peut prouver la causalite "
                "historique mais pas une capture locale realisee avant ce cutoff."
            ),
            "prospective_capture": (
                "Obligatoire separement pour production_pit_evidence=true."
            ),
        },
    }


__all__ = [
    "HistoryRecoveryAuditError",
    "RecoveryWindow",
    "audit_local_history",
    "derive_recovery_window",
]
