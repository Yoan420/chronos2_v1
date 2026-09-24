"""Isolated experiment orchestration; no production launcher or activation imports."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import yaml

from .data import load_zonal_inputs, MarginalCostDataError
from .evaluation import digest_file, physical_index, read_report_comparator, read_target, rolling_select, score_results
from .governance import GuardConfig, walkforward_guard
from .model import MarginalCostExpert


def _json(path: Path, value) -> None:
    temporary = path.with_name(".tmp_" + uuid.uuid4().hex[:12] + ".json")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _output(root: Path, value: str) -> Path:
    path = _path(root, value)
    allowed = (root / "runs" / "experiments").resolve()
    if not path.is_relative_to(allowed) or path == allowed:
        raise ValueError("Experiment outputs must be a dedicated child of runs/experiments.")
    return path


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Expected experiment schema_version: 1.")
    dates = config["evaluation"]
    start, end = pd.Timestamp(dates["start_day"]), pd.Timestamp(dates["end_day"])
    if end - start != pd.Timedelta(days=364) or int(dates.get("training_days", 0)) != 365:
        raise ValueError("This experiment requires exactly 365 evaluation days and 365 prior training days.")
    if config.get("activate", False):
        raise ValueError("This research pipeline cannot activate a production model.")
    if config.get("evaluation_labels", "frozen_report_observed") not in {"frozen_report_observed", "canonical_preferred"}:
        raise ValueError("Unknown evaluation label policy.")
    if config.get("expert_training_labels", "canonical_snapshot") != "canonical_snapshot":
        raise ValueError("Expert training labels must be an explicitly frozen canonical snapshot.")


def prepare(config: dict, *, project_root: Path, zones: list[str] | None = None) -> Path:
    """Freeze a new reproducible input snapshot. Existing files are never overwritten."""
    validate_config(config)
    root = project_root.resolve()
    output = _output(root, config["output_root"])
    dates = config["evaluation"]
    tz = dates.get("timezone", "Europe/Paris")
    start, end = str(dates["start_day"]), str(dates["end_day"])
    support_start = str((pd.Timestamp(start) - pd.Timedelta(days=365)).date())
    selected = list(config["references"]) if zones is None else [z.upper() for z in zones]
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(config["references"]):
        raise ValueError("Choose unique configured reference countries.")
    source_path = _path(root, config["sources_config"])
    source_config = yaml.safe_load(source_path.read_text(encoding="utf-8-sig"))
    source_config["allow_missing_hours"] = config.get("allow_missing_hours", False)
    features, observations, references, audits, blockers = [], [], [], {}, {}
    target_audits, reference_audits = {}, {}
    evaluation_hours = physical_index(start, end, tz)
    support_hours = physical_index(support_start, end, tz)
    for zone in selected:
        print(f"[Marginal/{zone}] audit des sources et gel du comparateur", flush=True)
        try:
            panel, audit = load_zonal_inputs(source_config, project_root=root, start_day=support_start,
                                             end_day=end, zones=[zone])
            features.append(panel)
            audits[zone] = audit
        except MarginalCostDataError as error:
            blockers[zone] = str(error)
            print(f"[Marginal/{zone}] expert indisponible : {error}. Reference conservee.", flush=True)
        target_path = _path(root, config["targets"][zone])
        before_hash = digest_file(target_path)
        target = read_target(target_path, zone).set_index("timestamp").reindex(support_hours)
        if digest_file(target_path) != before_hash:
            raise ValueError(f"{zone}: target changed while creating snapshot; retry after synchronization.")
        target["zone"] = zone
        target["label_source"] = "canonical_snapshot"
        target_audits[zone] = {"path": str(target_path), "sha256": before_hash,
                               "revision_proof": False, "semantics": "canonical_latest_revision_research_labels"}
        spec = config["references"][zone]
        ref, ref_audit = read_report_comparator(_path(root, spec["path"]), zone=zone, label=spec["label"],
                                               timezone=tz)
        ref = ref.set_index("timestamp").reindex(evaluation_hours)
        if not np.isfinite(ref.base.to_numpy(float)).all():
            raise ValueError(f"{zone}: fixed reference does not cover every hour of the specified 365 days.")
        # A published report may already contain newer observations than the local
        # target cache. Only its OBSERVED trace can supplement missing labels;
        # never fill from a model forecast or overwrite a known canonical label.
        from_report = target.actual.isna() & target.index.isin(ref.index[ref.actual.notna()])
        target.loc[from_report, "actual"] = ref.actual.reindex(target.index[from_report])
        target.loc[from_report, "label_source"] = "frozen_report_observed_fallback"
        target_audits[zone]["missing_labels_from_observed_report_trace"] = int(from_report.sum())
        if not np.isfinite(target.actual.to_numpy(float)).all():
            raise ValueError(f"{zone}: canonical/report observations do not cover the complete 730-day support.")
        observations.append(target.rename_axis("timestamp").reset_index())
        refreshed = target.actual.reindex(evaluation_hours)
        differences = (ref.actual - refreshed).abs()
        ref_audit["cache_report_disagreement_hours"] = int(differences.fillna(np.inf).gt(1e-9).sum())
        ref_audit["max_label_revision_eur_mwh"] = float(differences.max()) if differences.notna().any() else None
        label_policy = config.get("evaluation_labels", "frozen_report_observed")
        ref_audit["evaluation_label_policy"] = label_policy
        ref_audit["refreshed_label_hours"] = ref_audit["cache_report_disagreement_hours"] if label_policy == "canonical_preferred" else 0
        if label_policy == "canonical_preferred":
            ref["actual"] = refreshed
        if not np.isfinite(ref.actual.to_numpy(float)).all():
            raise ValueError(f"{zone}: selected scoring observations are incomplete.")
        ref["zone"] = zone
        references.append(ref.rename_axis("timestamp").reset_index())
        reference_audits[zone] = ref_audit
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    destination = output / "snapshots" / token
    destination.mkdir(parents=True, exist_ok=False)
    (pd.concat(features, ignore_index=True) if features else pd.DataFrame()).to_parquet(destination / "features.parquet", index=False)
    pd.concat(observations, ignore_index=True).to_parquet(destination / "targets.parquet", index=False)
    pd.concat(references, ignore_index=True).to_parquet(destination / "references.parquet", index=False)
    _json(destination / "config.json", config)
    _json(destination / "sources_config.json", source_config)
    audit = {"schema_version": 1, "status": "prepared", "zones": selected,
             "evaluation_start": start, "evaluation_end": end, "evaluation_days": 365,
             "training_start": support_start, "training_window_days": 365,
             "evaluation_hours_per_zone": len(evaluation_hours), "sources": audits,
             "unavailable_experts": blockers, "targets": target_audits, "references": reference_audits,
             "sources_config_sha256": digest_file(source_path),
             "production_modified": False, "promotion_eligible": False, "activation_performed": False,
             "production_pit_evidence": False, "network_used": False,
             "network_limit": "Full zonal supply/boundary and original valid CNEC/PTDF/RAM vintages not assembled; no fictional exchange capacities.",
             "reference_choice": "Fixed nuclear Kalman for FR/BE/NL, ordinary Kalman for DE; no per-day hindsight selection.",
             "baseline_neural_oof_certified": False,
             "expert_training_labels": "canonical_snapshot; missing labels explicitly traced to observed report fallback",
             "evaluation_and_guard_labels": config.get("evaluation_labels", "frozen_report_observed"),
             "label_availability_assumption": "DA label treated as known at D-1 18:00 civil; actual revision timestamps unproven; retrospective diagnostic only.",
             "hourly_approximation": "Hourly market proxy, not a reproduction of 15-minute SDAC or non-convex EUPHEMIA.",
             "June_24_26_2026": "Preidentified post-hoc diagnostic episode, not an untouched final test."}
    audit["snapshot_files"] = {name: digest_file(destination / name) for name in
                                ("features.parquet", "targets.parquet", "references.parquet", "config.json", "sources_config.json")}
    _json(destination / "audit.json", audit)
    print(f"[Marginal] snapshot : {destination}", flush=True)
    return destination


def _snapshot(path: Path, root: Path) -> tuple[dict, dict]:
    path = _output(root, str(path))
    audit = json.loads((path / "audit.json").read_text(encoding="utf-8"))
    required = {"features.parquet", "targets.parquet", "references.parquet", "config.json", "sources_config.json"}
    if set(audit.get("snapshot_files", {})) != required:
        raise ValueError("Snapshot checksum manifest is incomplete.")
    for name, expected in audit["snapshot_files"].items():
        if Path(name).name != name or digest_file(path / name) != expected:
            raise ValueError(f"Frozen snapshot checksum mismatch: {name}")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    validate_config(config)
    return config, audit


def physical_candidates(features: pd.DataFrame, config: dict, *, cache: Path) -> pd.DataFrame:
    """Cache label-free monthly blocks by their consumed content and engine code."""
    engine = MarginalCostExpert(config)
    code_hash = digest_file(Path(__file__).with_name("model.py"))
    if "demand_basis" not in features or set(features.demand_basis) != {config.get("demand_basis", "gross")}:
        raise ValueError("Input demand_basis must exactly match the physical engine configuration.")
    from importlib.metadata import version
    libraries = {name: version(name) for name in ("numpy", "pandas", "scipy")}
    panel = features.drop(columns=["demand_basis"], errors="ignore").copy()
    physical_fields = [name for name in panel if name.endswith("_mw") or name in {"ttf_eur_mwh_th", "eua_eur_tco2", "coal_eur_mwh_th"}]
    # Missing rows never reach the physical engine. Calendar gaps are retained
    # by rolling_select's exact reindex, so 710 days cannot masquerade as 730.
    panel = panel.loc[np.isfinite(panel[physical_fields]).all(axis=1)]
    if panel.empty:
        raise ValueError("No complete physical input row is available.")
    months = pd.to_datetime(panel.delivery_start_utc, utc=True).dt.strftime("%Y-%m")
    output = []
    cache.mkdir(parents=True, exist_ok=True)
    for month, block in panel.groupby(months, sort=True):
        block = block.sort_values(["delivery_start_utc", "zone"]).reset_index(drop=True)
        payload_hash = hashlib.sha256(pd.util.hash_pandas_object(block, index=False).to_numpy().tobytes()).hexdigest()
        key = hashlib.sha256(json.dumps({"code": code_hash, "libraries": libraries, "config": config, "values": payload_hash,
                                        "columns": list(block), "dtypes": [str(x) for x in block.dtypes]}, sort_keys=True).encode()).hexdigest()
        path, meta = cache / (key + ".parquet"), cache / (key + ".json")
        if path.exists() and meta.exists() and json.loads(meta.read_text())["sha256"] == digest_file(path):
            predicted = pd.read_parquet(path)
            print(f"[Marginal] physique {month} : cache valide", flush=True)
        else:
            predicted = engine.predict_candidates(block)
            temporary = path.with_name(f".tmp_{uuid.uuid4().hex[:12]}.parquet")
            predicted.to_parquet(temporary, index=False)
            temporary.replace(path)
            _json(meta, {"sha256": digest_file(path), "labels_used": False, "model_code_sha256": code_hash})
            print(f"[Marginal] physique {month} : {len(predicted)} scenarios horaires", flush=True)
        output.append(predicted)
    return pd.concat(output, ignore_index=True)


def diagnose_interventions(frame: pd.DataFrame, daily: pd.DataFrame) -> dict:
    result = {}
    for zone, block in frame.groupby("zone"):
        improvement = (block.base - block.actual).abs() - (block.guarded - block.actual).abs()
        active = block.weight.gt(0)
        dg = daily.loc[daily.zone.eq(zone)].pivot(index="day", columns="model", values="mae")
        delta = dg.base - dg.guarded
        result[zone] = {"annual_mae_gain": float(improvement.mean()), "active_hours": int(active.sum()),
                        "beneficial_active_hours": int((active & improvement.gt(1e-9)).sum()),
                        "harmful_active_hours": int((active & improvement.lt(-1e-9)).sum()),
                        "worst_daily_mae_increase": float((-delta).clip(lower=0).max()),
                        "days_improved": int(delta.gt(1e-9).sum()), "days_worsened": int(delta.lt(-1e-9).sum()),
                        "annual_non_regression_observed": bool(improvement.mean() >= -1e-9),
                        "zero_intervention_is_not_predictive_gain": not bool(active.any())}
    return result


def backtest(snapshot: Path, *, project_root: Path) -> Path:
    config, audit = _snapshot(snapshot, project_root)
    if (snapshot / "result_audit.json").exists():
        raise ValueError("Backtest snapshot already completed. Use Report or Prepare a new snapshot; no overwrite.")
    panel = pd.read_parquet(snapshot / "features.parquet")
    targets = pd.read_parquet(snapshot / "targets.parquet")
    reference = pd.read_parquet(snapshot / "references.parquet")
    tz = config["evaluation"].get("timezone", "Europe/Paris")
    selected = None
    if not panel.empty:
        candidates = physical_candidates(panel, config["expert"], cache=_output(project_root, config["output_root"]) / "physical_cache")
        candidates.to_parquet(snapshot / "candidate_predictions.parquet", index=False)
        print("[Marginal] selection chronologique : 365 jours precedents pour chaque jour evalue", flush=True)
        selected, selections = rolling_select(candidates, targets, evaluation_start=audit["evaluation_start"],
                                                evaluation_end=audit["evaluation_end"], timezone=tz,
                                                unavailable_policy="abstain")
        selections.to_parquet(snapshot / "rolling_selection.parquet", index=False)
        selected = selected.drop(columns=["delivery_start_utc"], errors="ignore").rename(columns={"network_mode": "mode"})
    if selected is not None:
        data = reference.merge(selected, on=["timestamp", "zone"], how="left", validate="one_to_one")
    else:
        data = reference.copy()
        data["expert"] = np.nan
        data["expert_oof"] = False
    data["expert_available"] = np.isfinite(data.expert)
    data["expert_oof"] = data.expert_oof.eq(True)
    data["risk"] = "neutral"
    # Predeclared physical thresholds. No realised volatility or target enters these regimes.
    if "capacity_margin_mw" in data:
        demand = data.demand_mw.clip(lower=1)
        tight = data.capacity_margin_mw.div(demand).le(config["risk"]["tight_capacity_margin_fraction"])
        surplus = data.curtailment_mw.gt(config["risk"]["surplus_curtailment_mw"])
        data.loc[tight & data.expert_available, "risk"] = "tight"
        data.loc[surplus & data.expert_available, "risk"] = "surplus"
    days = pd.to_datetime(data.timestamp, utc=True).dt.tz_convert(tz).dt.tz_localize(None).dt.normalize()
    data["label_available_at_utc"] = (days - pd.Timedelta(days=1) + pd.Timedelta(hours=18)).dt.tz_localize(tz).dt.tz_convert("UTC")
    data["mode"] = data.get("mode", pd.Series(index=data.index, dtype=object)).fillna("unavailable")
    print("[Marginal] gouvernance prequentielle : apprentissage passe, decisions hebdomadaires", flush=True)
    guard_config = GuardConfig(**config.get("guard", {}))
    guarded = walkforward_guard(data, config=guard_config)
    predicted = guarded.predictions
    metrics, daily = score_results(predicted, timezone=tz)
    # The unavailable expert does not remove any reference/guard hour from the annual table.
    available_zone = predicted.groupby("zone").expert.transform(lambda x: np.isfinite(x).any())
    paired = predicted.loc[np.isfinite(predicted[["actual", "base", "guarded", "storm"]]).all(axis=1)
                           & (np.isfinite(predicted.expert) | ~available_zone)]
    paired_metrics, _ = score_results(paired, timezone=tz) if not paired.empty else (metrics.iloc[:0].copy(), None)
    decisions = diagnose_interventions(predicted, daily)
    _json(snapshot / "policies.json", {"config": asdict(guard_config), "policies": guarded.policies, "audit": guarded.audit})
    predicted.to_parquet(snapshot / "predictions.parquet", index=False)
    metrics.to_parquet(snapshot / "metrics.parquet", index=False)
    paired_metrics.to_parquet(snapshot / "paired_metrics.parquet", index=False)
    daily.to_parquet(snapshot / "daily_metrics.parquet", index=False)
    metrics.to_csv(snapshot / "metrics.csv", index=False)
    daily.to_csv(snapshot / "daily_metrics.csv", index=False)
    audit.update(status="completed_diagnostic", intervention_diagnostics=decisions,
                 guard=asdict(guard_config), annual_non_regression_guaranteed=False,
                 scenario_selection="past_365_calendar_days_only; no evaluation-day label",
                 governance_warmup="First 60 complete expert OOF days are a zero-weight prequential warmup, not 365 extra unseen calibration days.")
    audit["physical_diagnostics"] = {}
    for zone, block in predicted.groupby("zone"):
        available = block.loc[np.isfinite(block.expert)]
        audit["physical_diagnostics"][zone] = {
            "expert_available_hours": len(available), "reference_hours": len(block),
            "expert_unavailable_hours": int(block.expert.isna().sum()),
            "shortage_proxy_share": float(available.shortage_mw.gt(1e-6).mean()) if len(available) else None,
            "interpretation": "Deficit of the represented partial stack, NOT evidence of real system scarcity.",
        }
    audit["physical_qualification"] = "partial_gas_nuclear_residual_stack_only; full supply and network not qualified"
    audit["experiment_source_code_sha256"] = {name: digest_file(Path(__file__).with_name(name)) for name in
                                               ("model.py", "evaluation.py", "governance.py", "runner.py", "data.py")}
    no_gain = all(x["annual_mae_gain"] < guard_config.minimum_gain_eur_mwh for x in decisions.values())
    audit["result_summary"] = ("Aucun gain annuel démontré : conserver les modèles actuels." if no_gain else
                                "Résultats expérimentaux hétérogènes : lire les gains et dégradations par pays. Aucun changement opérationnel.")
    audit["result_files"] = {name: digest_file(snapshot / name) for name in
                             ("predictions.parquet", "metrics.parquet", "paired_metrics.parquet", "daily_metrics.parquet", "policies.json")}
    _json(snapshot / "result_audit.json", audit)
    report(snapshot, project_root=project_root)
    _json(_output(project_root, config["output_root"]) / "latest.json", {"snapshot": str(snapshot), "report": str(snapshot / "marginal_cost_report.html")})
    return snapshot


def report(snapshot: Path, *, project_root: Path) -> Path:
    """Offline report refresh: archived scores only, no training/source requests."""
    from .reporting import render_report
    snapshot = _output(project_root, str(snapshot))
    audit = json.loads((snapshot / "result_audit.json").read_text(encoding="utf-8"))
    required = {"predictions.parquet", "metrics.parquet", "paired_metrics.parquet", "daily_metrics.parquet", "policies.json"}
    if set(audit.get("result_files", {})) != required:
        raise ValueError("Result checksum manifest is incomplete.")
    for name, expected in audit["result_files"].items():
        if Path(name).name != name or digest_file(snapshot / name) != expected:
            raise ValueError(f"Result checksum mismatch: {name}")
    path = render_report(pd.read_parquet(snapshot / "predictions.parquet"), pd.read_parquet(snapshot / "metrics.parquet"),
                         pd.read_parquet(snapshot / "paired_metrics.parquet"), pd.read_parquet(snapshot / "daily_metrics.parquet"),
                         audit, snapshot / "marginal_cost_report.html")
    print(f"[Marginal] rapport : {path}", flush=True)
    return path
