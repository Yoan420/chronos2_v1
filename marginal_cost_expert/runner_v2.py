"""Isolated supply/08:00-network qualification with a rolling365 diagnostic.

The zonal simulation is explicitly NOT a coupled market. Raw simulated prices
can be calibrated for research, but never enter conditional intervention while
the fleet/reference/boundary contracts are unqualified. No production imports.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import yaml

from .dispatch import build_offers, dispatch_periods
from .evaluation import digest_file, physical_index, rolling_select, score_results
from .runner import _json, _output, _path, _snapshot
from .supply_sources import load_supply_inputs


SNAPSHOT_FILES = {"features.parquet", "references.parquet", "targets.parquet", "config.json",
                  "sources_config.json", "network_audit.json"}
RESULT_FILES = {"predictions.parquet", "metrics.parquet", "daily_metrics.parquet",
                "diagnostic_metrics.parquet", "diagnostic_daily.parquet", "paired_diagnostic_metrics.parquet",
                "rolling_selection.parquet", "candidate_predictions.parquet"}


def validate_config(config: dict) -> None:
    if not isinstance(config, dict) or config.get("schema_version") != 2:
        raise ValueError("Expected marginal-cost experiment schema_version: 2.")
    dates = config["evaluation"]
    if pd.Timestamp(dates["end_day"]) - pd.Timestamp(dates["start_day"]) != pd.Timedelta(days=364):
        raise ValueError("Exactly 365 evaluation calendar days are required.")
    if dates.get("training_days") != 365:
        raise ValueError("Exactly 365 preceding calibration calendar days are required.")
    if dates.get("cutoff_time") != "08:00" or dates.get("timezone") != "Europe/Paris":
        raise ValueError("Strict D-1 08:00 Europe/Paris is mandatory.")
    network = config["network"]
    if network.get("cutoff_time") != "08:00":
        raise ValueError("Network must use the same strict 08:00 cutoff.")
    if network.get("domain_reference_qualified") is not False or network.get("boundary_qualified") is not False:
        raise ValueError("This qualification POC cannot enable a coupled domain by a configuration flag.")
    if config.get("activate") is not False:
        raise ValueError("This experimental workflow cannot activate a production model.")
    if not config.get("scenarios") or "central" not in config["scenarios"]:
        raise ValueError("Declare the finite scenario bank and its central diagnostic.")


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    validate_config(config)
    return config


def _verify_files(path: Path, expected: dict, required: set[str]) -> None:
    if set(expected) != required:
        raise ValueError("Incomplete V2 checksum manifest.")
    for name, digest in expected.items():
        if Path(name).name != name or digest_file(path / name) != digest:
            raise ValueError(f"Frozen V2 checksum mismatch: {name}")


def _network_audit(path: Path, *, first: str, last: str) -> dict:
    """Verify archived qualification and every referenced source pair, not late data."""
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("cutoff_time") != "08:00" or audit.get("timezone") != "Europe/Paris":
        raise ValueError("Network audit must explicitly prove the strict civil 08:00 recipe.")
    if audit.get("first_day") != first or audit.get("last_day") != last or audit.get("calendar_days") != 730:
        raise ValueError("Network audit must retain all 730 support days including absences.")
    days = [row.get("delivery_day") for row in audit["daily"]]
    if days != [str(d.date()) for d in pd.date_range(first, last)]:
        raise ValueError("Network audit calendar is duplicated, shortened or out of order.")
    if audit.get("domain_reference_qualified") is not False or audit.get("boundary_qualified") is not False:
        raise ValueError("Unreviewed network reference/boundary qualification is forbidden.")
    code_digest = digest_file(Path(__file__).with_name("network.py"))
    if audit.get("network_code_sha256") != code_digest:
        raise ValueError("Re-run network qualification after changing its validation code.")
    for row in audit["daily"]:
        if row.get("raw_gzip_sha256"):
            for field, sha in (("raw_path", "raw_gzip_sha256"), ("source_audit_path", "source_audit_sha256")):
                if not row.get(field) or not row.get(sha) or digest_file(Path(row[field])) != row[sha]:
                    raise ValueError(f"Network audit source changed: {row['delivery_day']}/{field}")
        elif row.get("inputs_qualified"):
            raise ValueError("A qualified network day requires original archive seals.")
    return audit


def prepare(config: dict, *, project_root: Path, zones: list[str] | None = None) -> Path:
    validate_config(config)
    root = project_root.resolve()
    old = _path(root, config["reference_snapshot"])
    old_config, old_audit = _snapshot(old, root)
    dates = config["evaluation"]
    start, end = str(dates["start_day"]), str(dates["end_day"])
    if (old_audit["evaluation_start"], old_audit["evaluation_end"]) != (start, end):
        raise ValueError("V1 and V2 must use exactly the same frozen 365-day comparison.")
    selected = list(old_audit["zones"]) if zones is None else list(zones)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(old_audit["zones"]):
        raise ValueError("Choose unique countries present in the fixed reference snapshot.")
    first = str((pd.Timestamp(start) - pd.Timedelta(days=365)).date())
    source_path = _path(root, config["sources_config"])
    source_hash = digest_file(source_path)
    sources = yaml.safe_load(source_path.read_text(encoding="utf-8-sig"))
    if sources.get("timezone") != dates["timezone"]:
        raise ValueError("Supply and evaluation civil timezones must agree.")
    for zone in selected:
        source_zone = sources["zones"][zone]
        if source_zone.get("demand_basis") != config["expert"]["demand_basis"]:
            raise ValueError(f"{zone}: supply and dispatch demand_basis disagree.")
        if source_zone.get("residual_netting") != config["expert"]["residual_netting"][zone]:
            raise ValueError(f"{zone}: supply and dispatch residual netting disagree.")
    network_path = _path(root, config["network"]["audit_path"])
    network = _network_audit(network_path, first=first, last=end)
    print("[Marginal V2] gel des sources, observations et comparateurs ; cutoff strict 08:00", flush=True)
    features, supply = load_supply_inputs(sources, project_root=root, start_day=first, end_day=end, zones=selected)
    if digest_file(source_path) != source_hash:
        raise ValueError("Supply configuration changed during preparation.")
    references = pd.read_parquet(old / "references.parquet")
    targets = pd.read_parquet(old / "targets.parquet")
    references, targets = [frame.loc[frame.zone.isin(selected)].copy() for frame in (references, targets)]
    _snapshot(old, root)  # detect a concurrent change while reading
    for frame, first_day in ((references, start), (targets, first)):
        expected = physical_index(first_day, end, dates["timezone"])
        for zone in selected:
            block = frame.loc[frame.zone.eq(zone)].sort_values("timestamp")
            if not pd.DatetimeIndex(block.timestamp).equals(expected) or not np.isfinite(block.actual).all():
                raise ValueError(f"{zone}: incomplete or ambiguous frozen hourly observations.")
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    destination = _output(root, config["output_root"]) / "snapshots" / token
    destination.mkdir(parents=True, exist_ok=False)
    for name, frame in (("features", features), ("references", references), ("targets", targets)):
        frame.to_parquet(destination / f"{name}.parquet", index=False)
    for name, value in (("config", config), ("sources_config", sources), ("network_audit", network)):
        _json(destination / f"{name}.json", value)
    audit = {"schema_version": 2, "status": "prepared", "zones": selected,
             "evaluation_start": start, "evaluation_end": end, "evaluation_days": 365,
             "training_start": first, "training_days": 365, "cutoff_time": "08:00",
             "reference_snapshot": str(old), "reference_snapshot_seals": old_audit["snapshot_files"],
             "source_config_sha256": source_hash, "supply": supply,
             "network_audit_source": str(network_path), "network_audit_sha256": digest_file(network_path),
             "source_code_sha256": {name: digest_file(Path(__file__).with_name(name)) for name in
                                     ("runner_v2.py", "dispatch.py", "supply_sources.py", "network.py", "evaluation.py")},
             "production_modified": False, "activation_performed": False, "promotion_eligible": False,
             "baseline_neural_oof_certified": False, "production_pit_evidence": False,
             "network_used_for_price": False, "coupled_expert_qualified": False,
             "snapshot_files": {name: digest_file(destination / name) for name in sorted(SNAPSHOT_FILES)}}
    _json(destination / "audit.json", audit)
    return destination


def _read_snapshot(snapshot: Path, root: Path) -> tuple[Path, dict, dict]:
    snapshot = _output(root, str(snapshot))
    audit = json.loads((snapshot / "audit.json").read_text(encoding="utf-8"))
    _verify_files(snapshot, audit["snapshot_files"], SNAPSHOT_FILES)
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    validate_config(config)
    dates = config["evaluation"]
    first = str((pd.Timestamp(dates["start_day"]) - pd.Timedelta(days=365)).date())
    if (audit.get("evaluation_start"), audit.get("evaluation_end"), audit.get("evaluation_days"),
        audit.get("training_start"), audit.get("training_days"), audit.get("cutoff_time")) != (
            dates["start_day"], dates["end_day"], 365, first, 365, "08:00"):
        raise ValueError("Snapshot audit disagrees with the sealed 365/365-day configuration.")
    if not audit.get("zones") or len(set(audit["zones"])) != len(audit["zones"]):
        raise ValueError("Unique snapshot zones required.")
    for name, key, first_day in (("features", "delivery_start_utc", first),
                                 ("targets", "timestamp", first), ("references", "timestamp", dates["start_day"])):
        frame = pd.read_parquet(snapshot / f"{name}.parquet")
        expected = physical_index(first_day, dates["end_day"], dates["timezone"])
        if set(frame.zone) != set(audit["zones"]):
            raise ValueError(f"{name}: snapshot countries disagree with audit.")
        for zone, rows in frame.groupby("zone"):
            if not pd.DatetimeIndex(rows.sort_values(key)[key]).equals(expected):
                raise ValueError(f"{name}/{zone}: incomplete physical-hour support.")
    return snapshot, config, audit


def _physical_candidates(features: pd.DataFrame, config: dict, snapshot: Path) -> pd.DataFrame:
    """Label-free zonal blocks; disjoint countries avoid cross-zone gap propagation."""
    blocks = []
    if set(features.demand_basis) != {config["expert"]["demand_basis"]}:
        raise ValueError("Feature demand_basis disagrees with dispatch configuration.")
    for zone, panel in features.groupby("zone", sort=True):
        for name, efficiencies in config["scenarios"].items():
            physical = deepcopy(config["expert"])
            ids = {segment["id"] for segment in physical["segments"]}
            if set(efficiencies) - ids:
                raise ValueError("Scenario references an unknown offer segment.")
            for segment in physical["segments"]:
                if segment["id"] in efficiencies:
                    segment["bid"]["efficiency"] = efficiencies[segment["id"]]
            print(f"[Marginal V2/{zone}] scenario {name} : simulation zonale, sans reseau fictif", flush=True)
            book = build_offers(panel, physical)
            result = dispatch_periods(book.demand, book.offers, qualification=book.qualification,
                                      scarcity_price_eur_mwh=physical["scarcity_price_eur_mwh"], candidate_id=name)
            # Belt-and-braces: missing consumed source values are not a small fleet.
            valid = panel.set_index(["delivery_start_utc", "zone"]).inputs_complete
            keys = pd.MultiIndex.from_frame(result.prices[["delivery_start_utc", "zone"]])
            missing = ~valid.reindex(keys).fillna(False).to_numpy(bool)
            result.prices.loc[missing, ["price_eur_mwh", "raw_price_eur_mwh"]] = np.nan
            result.prices.loc[missing, "eligible"] = False
            result.prices.loc[missing, "status"] = "missing_source_inputs"
            blocks.append(result.prices)
    return pd.concat(blocks, ignore_index=True)


def assemble_predictions(reference: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """Raw physical diagnostics NEVER become an approved coupled intervention."""
    cols = [name for name in ("timestamp", "zone", "expert", "candidate_id", "shortage_mw",
                              "capacity_margin_mw", "curtailment_mw", "expert_unavailable_reason") if name in selected]
    renamed = selected[cols].rename(columns={"expert": "diagnostic_expert"})
    frame = reference.merge(renamed, on=["timestamp", "zone"], how="left", validate="one_to_one")
    frame["expert"] = np.nan
    frame["expert_available"] = False
    frame["expert_oof"] = False
    frame["weight"] = 0.0
    frame["guarded"] = frame.base
    frame["abstention_reason"] = "fleet_and_08h_network_domain_not_qualified"
    return frame


def backtest(snapshot: Path, *, project_root: Path) -> Path:
    snapshot, config, audit = _read_snapshot(snapshot, project_root)
    if (snapshot / "result_audit.json").exists():
        raise ValueError("V2 results already sealed. Use Report or create a new snapshot.")
    for name, sha in audit["source_code_sha256"].items():
        if digest_file(Path(__file__).with_name(name)) != sha:
            raise ValueError(f"Code changed since Prepare: {name}; create a new snapshot.")
    features = pd.read_parquet(snapshot / "features.parquet")
    reference = pd.read_parquet(snapshot / "references.parquet")
    targets = pd.read_parquet(snapshot / "targets.parquet")
    candidates = _physical_candidates(features, config, snapshot)
    candidates.to_parquet(snapshot / "candidate_predictions.parquet", index=False)
    # Deliberately separate raw research calibration from the qualified output.
    raw = candidates.copy()
    raw["price_eur_mwh"] = raw.raw_price_eur_mwh
    print("[Marginal V2] calibration diagnostique : fenetres precedentes de 365 jours, sans trous compresses", flush=True)
    selected, selections = rolling_select(raw, targets, evaluation_start=audit["evaluation_start"],
                                          evaluation_end=audit["evaluation_end"], unavailable_policy="abstain")
    predicted = assemble_predictions(reference, selected)
    metrics, daily = score_results(predicted)
    diagnostic = predicted.copy()
    diagnostic["expert"] = diagnostic.diagnostic_expert
    diagnostic_metrics, diagnostic_daily = score_results(diagnostic)
    diagnostic_metrics = diagnostic_metrics.loc[diagnostic_metrics.model.eq("expert")].assign(model="raw_zonal_diagnostic")
    diagnostic_daily = diagnostic_daily.loc[diagnostic_daily.model.eq("expert")].assign(model="raw_zonal_diagnostic")
    paired = diagnostic.loc[np.isfinite(diagnostic[["actual", "base", "expert", "storm"]]).all(axis=1)]
    paired_metrics, _ = score_results(paired) if not paired.empty else (metrics.iloc[:0], None)
    paired_metrics = paired_metrics.replace({"model": {"expert": "raw_zonal_diagnostic"}})
    frames = {"predictions": predicted, "metrics": metrics, "daily_metrics": daily,
              "diagnostic_metrics": diagnostic_metrics, "diagnostic_daily": diagnostic_daily,
              "paired_diagnostic_metrics": paired_metrics, "rolling_selection": selections}
    for name, frame in frames.items():
        frame.to_parquet(snapshot / f"{name}.parquet", index=False)
    audit.update(status="completed_qualification_diagnostic", intervention_hours=0,
                 governance_status="abstention_before_weight_fitting; qualified coupled candidate unavailable",
                 annual_improvement_demonstrated=False,
                 result_summary="Offre enrichie testée, mais expert couplé non qualifié à 08 h. Référence intégralement conservée.",
                 calibration="Scenario selected on exactly preceding365 calendar days; latest frozen canonical label revisions are research-only.",
                 future_validation="Qualify fleet, initial RAM reference and all boundary coordinates before testing conditional intervention.",
                 result_files={name: digest_file(snapshot / name) for name in sorted(RESULT_FILES)})
    _json(snapshot / "result_audit.json", audit)
    path = report(snapshot, project_root=project_root)
    _json(_output(project_root, config["output_root"]) / "latest.json", {"snapshot": str(snapshot), "report": str(path)})
    return snapshot


def report(snapshot: Path, *, project_root: Path) -> Path:
    from .reporting_v2 import render_report
    snapshot, config, audit = _read_snapshot(snapshot, project_root)
    result = json.loads((snapshot / "result_audit.json").read_text(encoding="utf-8"))
    _verify_files(snapshot, result["result_files"], RESULT_FILES)
    path = render_report(snapshot, config, result)
    print(f"[Marginal V2] rapport : {path}", flush=True)
    return path
