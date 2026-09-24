"""Append-only research trial of the two rank-16 auxiliary chains.

Previously inspected neural backtests are calibration, not independent tests.
Only forecasts actually completed before auction can enter the new scorecard.
No existing bundle, production setting, or shared data cache is written here.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from .prospective_auxiliary import MARKET_COLUMNS, QUANTILES, RAW_MODEL, ResidualRecipe


class ProspectiveTrialError(ValueError):
    pass


FLAGS = dict(diagnostic_only=True, production_pit_evidence=False,
             production_pipeline_evidence=False, promotion_eligible=False,
             activation_performed=False, neural_oof=False)
CODE_FILES = (
    "chronos2_exogenous/prospective_trial.py", "chronos2_exogenous/prospective_auxiliary.py",
    "chronos2_exogenous/prospective_inputs.py", "chronos2_exogenous/evaluation.py",
    "chronos2_exogenous/prospective_reporting.py",
    "chronos2_exogenous/prospective_bootstrap.py", "chronos2_exogenous/prospective_maintenance.py",
    "chronos2_exogenous/lora_finetune.py", "chronos2_hourly/kalman_residual.py",
    "chronos2_hourly/kalman_covariates.py", "chronos2_exogenous/panel.py",
    "chronos2_exogenous/feature_bank.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False,
                  default=lambda obj: obj.item() if isinstance(obj, np.generic) else str(obj))
        stream.write("\n")


def _file(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _verify_file(entry: Mapping[str, str]) -> Path:
    path = Path(entry["path"])
    if not path.is_file() or sha256(path) != entry["sha256"]:
        raise ProspectiveTrialError(f"Empreinte divergente ou fichier absent: {path}.")
    return path


def _day(value: str) -> str:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.normalize():
        raise ProspectiveTrialError("Une date civile YYYY-MM-DD est requise.")
    return stamp.date().isoformat()


def load_trial_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or type(config.get("version")) is not int or config["version"] != 1:
        raise ProspectiveTrialError("Version du protocole inconnue.")
    root = (path.parent / config["project_root"]).resolve()
    for key in ("output_root", "candidate_config", "zone_artifacts_root", "calibration_panel"):
        config[key] = str((root / config[key]).resolve())
    output = Path(config["output_root"])
    if output == root / "runs/experiments" or not output.is_relative_to(root / "runs/experiments"):
        raise ProspectiveTrialError("Le laboratoire doit rester dans runs/experiments.")
    zones = config["zones"]
    if (not isinstance(zones, list) or not zones or any(not isinstance(z, str) for z in zones)
            or len(zones) != len(set(zones)) or set(zones) - {"FR", "DE", "BE", "NL"}):
        raise ProspectiveTrialError("Zones invalides pour le candidat rang 16.")
    ResidualRecipe(**config["residual_recipe"]).validate()
    return {**config, "project_root": str(root), "config_path": str(path)}


def emission_window(delivery_day: str, *, now: pd.Timestamp | None = None) -> dict[str, str]:
    """Real-time gate: no retrodated forecasts, conservative pre-auction deadline."""
    day = pd.Timestamp(_day(delivery_day))
    origin = (day - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris")
    deadline = (day - pd.Timedelta(days=1) + pd.Timedelta(hours=11, minutes=45)).tz_localize("Europe/Paris")
    current = now_utc() if now is None else pd.Timestamp(now)
    if current.tzinfo is None or not origin <= current < deadline:
        raise ProspectiveTrialError(
            f"Emission autorisee uniquement le {origin.date()} entre 08:00 et 11:45 Paris "
            f"pour la livraison {delivery_day}; aucune prediction antidatee.")
    return {"forecast_origin_utc": origin.tz_convert("UTC").isoformat(),
            "deadline_utc": deadline.tz_convert("UTC").isoformat()}


def _read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("delivery_start_utc", "forecast_origin_utc"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def _write_frame(path: Path, frame: pd.DataFrame) -> dict[str, str]:
    if path.exists():
        raise ProspectiveTrialError(f"Ecrasement refuse: {path}.")
    frame.to_csv(path, index=False, compression="gzip")
    return _file(path)


def prepare_trial(config: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze source identity and reuse only raw, already completed evaluations."""
    from .lora_finetune import _schema_payload, load_config, verify_bundle
    from .prospective_auxiliary import fit_residual_corrector
    from chronos2_hourly.kalman_residual import KalmanResidualConfig
    output, root = Path(config["output_root"]), Path(config["project_root"])
    if (output / "trial_manifest.json").exists():
        return verify_trial(config)
    if output.exists():
        raise ProspectiveTrialError("Preparation incomplete deja presente; conserver et inspecter ce dossier.")
    candidate = load_config(config["candidate_config"])
    panel = Path(config["calibration_panel"])
    if sha256(panel) != config["calibration_panel_sha256"]:
        raise ProspectiveTrialError("Le panel historique ne correspond pas au candidat selectionne.")
    entries, frames = {}, {}
    for zone in config["zones"]:
        bundle = Path(config["zone_artifacts_root"]) / zone / "artifact"
        model = verify_bundle(bundle)
        schema = _json(bundle / "schema.json")
        evaluation = _json(bundle / "evaluation_manifest.json")
        if model["checkpoint_sha256"] != config["checkpoint_sha256"] or model["training"]["lora_config"]["r"] != 16:
            raise ProspectiveTrialError(f"{zone}: checkpoint non conforme au rang 16 selectionne.")
        if schema != _schema_payload(candidate):
            raise ProspectiveTrialError(f"{zone}: schema du YAML different du bundle.")
        if (evaluation["bundle_manifest_sha256"] != sha256(bundle / "experiment_manifest.json")
                or evaluation["item_id"] != zone or evaluation["comparison"]["residual_corrector_applied"]):
            raise ProspectiveTrialError(f"{zone}: evaluation brute non liee au bundle.")
        evidence = evaluation["artifacts"]["evidence"]
        prediction_path = (bundle / evidence["relative_path"]).resolve()
        if not prediction_path.is_relative_to(bundle.resolve()):
            raise ProspectiveTrialError("Chemin d'evaluation hors bundle.")
        _verify_file({"path": str(prediction_path), "sha256": evidence["sha256"]})
        raw = _read_frame(prediction_path).rename(columns={f"candidate_{q}": f"{RAW_MODEL}__{q}" for q in QUANTILES})
        covs = pd.read_parquet(panel, columns=["timestamp", "origin_timestamp", *MARKET_COLUMNS],
                              filters=[("phase", "==", "horizon"), ("item_id", "==", zone)])
        covs = covs.rename(columns={"timestamp": "delivery_start_utc", "origin_timestamp": "forecast_origin_utc"})
        for column in ("delivery_start_utc", "forecast_origin_utc"):
            covs[column] = pd.to_datetime(covs[column], utc=True)
        raw = raw[["delivery_start_utc", "forecast_origin_utc", "actual", *(f"{RAW_MODEL}__{q}" for q in QUANTILES)]].merge(
            covs, on=["delivery_start_utc", "forecast_origin_utc"], how="left", validate="one_to_one")
        days = raw.delivery_start_utc.dt.tz_convert("Europe/Paris").dt.date
        if days.min().isoformat() != "2025-09-03" or days.max().isoformat() != "2026-09-02" or days.nunique() != 365:
            raise ProspectiveTrialError(f"{zone}: les 365 jours de calibration originaux sont incomplets.")
        fit_residual_corrector(raw, target_day="2026-09-03", require_full_window=True)
        frames[zone] = raw
        entries[zone] = {"bundle": str(bundle), "bundle_manifest": _file(bundle / "experiment_manifest.json"),
                         "evaluation_manifest": _file(bundle / "evaluation_manifest.json"), "raw_evidence": _file(prediction_path)}
    output.mkdir(parents=True, exist_ok=False)
    (output / "calibration").mkdir()
    for zone, raw in frames.items():
        entries[zone]["calibration"] = _write_frame(output / "calibration" / f"{zone}.csv.gz", raw)
    manifest = {**FLAGS, "kind": "rank16_two_chain_research_trial", "created_at_utc": now_utc().isoformat(),
                "historical_role": "previously_inspected_calibration_not_independent_test",
                "models": ["lora16_residual", "lora16_residual_kalman"],
                "raw_model_role": "diagnostic_reference_only", "config": dict(config),
                "config_file": _file(Path(config["config_path"])),
                "candidate_config": _file(Path(config["candidate_config"])),
                "panel": _file(panel), "schema": _schema_payload(candidate),
                "recipe": config["residual_recipe"], "kalman_config": asdict(KalmanResidualConfig()),
                "code_files": [_file(root / p) for p in CODE_FILES], "zones": entries}
    _write_json(output / "trial_manifest.json", manifest)
    return manifest


def verify_trial(config: Mapping[str, Any]) -> dict[str, Any]:
    from .lora_finetune import verify_bundle
    from .prospective_maintenance import verify_trial_code
    manifest = _json(Path(config["output_root"]) / "trial_manifest.json")
    if manifest["config"] != dict(config) or any(manifest.get(k) != v for k, v in FLAGS.items()):
        raise ProspectiveTrialError("Protocole fige different de la configuration actuelle.")
    for entry in [manifest["config_file"], manifest["candidate_config"]]:
        _verify_file(entry)
    try:
        maintenance_time = verify_trial_code(manifest, Path(config["output_root"]))
    except ValueError as exc:
        raise ProspectiveTrialError(str(exc)) from exc
    for entry in manifest["zones"].values():
        for key in ("bundle_manifest", "evaluation_manifest", "raw_evidence", "calibration"):
            _verify_file(entry[key])
        if verify_bundle(entry["bundle"])["checkpoint_sha256"] != config["checkpoint_sha256"]:
            raise ProspectiveTrialError("Checkpoint du candidat modifie.")
    return {**manifest, "_effective_freeze_utc": str(maintenance_time or manifest["created_at_utc"])}


def load_history(manifest: Mapping[str, Any], zone: str) -> pd.DataFrame:
    output = Path(manifest["config"]["output_root"])
    blocks = [_read_frame(_verify_file(manifest["zones"][zone]["calibration"]))]
    # One sealed entry per zone/day, retrospective calibration OR actual issuance.
    for path in sorted((output / "days").glob(f"*/{zone}/manifest.json")):
        daily = _verify_daily(path, output=output, expected_zone=zone)
        blocks.append(_read_frame(_verify_file(daily["raw_predictions"])))
    history = pd.concat(blocks, ignore_index=True).sort_values("delivery_start_utc").reset_index(drop=True)
    if history.delivery_start_utc.duplicated().any():
        raise ProspectiveTrialError("Plusieurs predictions pour une meme heure de calibration.")
    return history


def _verify_daily(path: Path, *, output: Path, expected_zone: str | None = None) -> dict[str, Any]:
    daily = _json(path)
    zone, day = path.parent.name, path.parent.parent.name
    protocol = _json(output / "trial_manifest.json")
    prospective = daily.get("prospective_eligible")
    if (zone not in {"FR", "DE", "BE", "NL"} or (expected_zone and zone != expected_zone)
            or zone not in protocol["zones"]
            or daily.get("zone") != zone or daily.get("delivery_day") != _day(day)
            or type(prospective) is not bool or any(daily.get(k) != v for k, v in FLAGS.items())
            or daily.get("trial_manifest_sha256") != sha256(output / "trial_manifest.json")
            or daily.get("role") != ("prospective_two_chain_forecast" if prospective else "retrospective_calibration_only")):
        raise ProspectiveTrialError(f"Identite/contrat de journee incoherent: {path}.")
    raw_path = _verify_file(daily["raw_predictions"])
    if raw_path.resolve() != (path.parent / "raw.csv.gz").resolve():
        raise ProspectiveTrialError("Predictions brutes hors du dossier de la journee.")
    raw = _read_frame(raw_path)
    expected = pd.date_range(pd.Timestamp(day, tz="Europe/Paris"),
                            (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize("Europe/Paris"),
                            inclusive="left", freq="h").tz_convert("UTC")
    if not pd.DatetimeIndex(raw.delivery_start_utc).equals(expected):
        raise ProspectiveTrialError("Heures brutes incompatibles avec la journee declaree.")
    capture = _json(_verify_file(daily["input_manifest"]))
    if prospective:
        from .prospective_maintenance import verify_trial_code
        try:
            maintenance_time = verify_trial_code(protocol, output)
        except ValueError as exc:
            raise ProspectiveTrialError(str(exc)) from exc
        window = emission_window(day, now=pd.Timestamp(daily["completed_at_utc"]))
        check = daily.get("prepublication_target_check", {})
        if (daily.get("emission_window") != window
                or pd.Timestamp(maintenance_time or protocol["created_at_utc"]) > pd.Timestamp(window["forecast_origin_utc"])
                or capture.get("fresh_source_refresh") is not True
                or capture["horizon_labels_available_at_capture"][day][zone] != 0
                or check.get("fresh_api_read") is not True or check.get("fallback_used") is not False
                or daily.get("prepublication_observed_hours") != 0
                or raw.actual.notna().any()
                or not (pd.Timestamp(capture["capture_completed_at_utc"])
                        <= pd.Timestamp(check["capture_completed_at_utc"])
                        <= pd.Timestamp(daily["completed_at_utc"]))):
            raise ProspectiveTrialError("Emission sans preuve coherente de controle avant publication.")
        for key, filename in (("predictions", "predictions.csv.gz"), ("auxiliary_audit", "auxiliary_audit.json")):
            if _verify_file(daily[key]).resolve() != (path.parent / filename).resolve():
                raise ProspectiveTrialError("Artefact prospectif hors du dossier de la journee.")
    return daily


def _targets(capture: Mapping[str, Any], zone: str) -> pd.Series:
    entry = capture["target_snapshots"][zone]
    path = _verify_file({"path": entry["snapshot_path"], "sha256": entry["snapshot_sha256"]})
    frame = pd.read_csv(path)
    return pd.Series(frame.target.to_numpy(float), index=pd.to_datetime(frame.timestamp, utc=True))


def _attach_actuals(raw: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    result = raw.copy()
    latest = target.reindex(pd.DatetimeIndex(result.delivery_start_utc)).to_numpy(float)
    if "actual" not in result:
        result["actual"] = np.nan
    mask = np.isfinite(latest)
    result.loc[mask, "actual"] = latest[mask]
    return result


def _predict_group(group: pd.DataFrame, candidate: Any, pipeline: Any) -> pd.DataFrame:
    from .evaluation import build_inference_input, _predict_quantiles, _input_sha256
    payload, index, _ = build_inference_input(group, candidate, allow_missing_actual=True)
    values = _predict_quantiles(pipeline, payload, prediction_length=len(index),
                               context_length=candidate.context_length, batch_size=64)[0]
    horizon = group.loc[group.phase.eq("horizon")].sort_values("timestamp")
    raw = pd.DataFrame({"delivery_start_utc": index.tz_convert("UTC"),
                        "forecast_origin_utc": pd.Timestamp(group.origin_timestamp.iloc[0]),
                        "actual": np.nan, "input_sha256": _input_sha256(payload)})
    for i, q in enumerate(QUANTILES):
        raw[f"{RAW_MODEL}__{q}"] = values[:, i]
    for column in MARKET_COLUMNS:
        raw[column] = horizon[column].to_numpy(float)
    return raw


def _reuse_bootstrap_inputs(config: Mapping[str, Any], *, days: list[str], zones: list[str],
                            schema: Mapping[str, Any]) -> dict[str, Any] | None:
    """Reuse only verified retrospective inputs; always reread canonical labels.

    The old panel and its manifests remain immutable. The new observation
    snapshot has its own capture ledger and can never serve as fresh live input.
    """
    from .prospective_inputs import _capture_target
    output = Path(config["output_root"])
    for path in sorted((output / "captures").glob("*/input_manifest.json"), reverse=True):
        original = _json(path)
        if (not set(days).issubset(original.get("delivery_days", []))
                or not set(zones).issubset(original.get("zones", []))
                or original.get("known_future_covariates") != schema["known_future_covariates"]):
            continue
        # If this capture already backed a sealed day, its manifest identity
        # must still match that day, not just its internally declared hashes.
        for daily_path in (output / "days").glob("*/*/manifest.json"):
            reference = _json(daily_path).get("input_manifest", {})
            if reference.get("path") and Path(reference["path"]).resolve() == path.resolve():
                _verify_file(reference)
        for name in ("panel", "panel_audit", "seed_manifest"):
            _verify_file({"path": original[f"{name}_path"], "sha256": original[f"{name}_sha256"]})
        attempt = output / "bootstrap_attempts" / (now_utc().strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
        attempt.mkdir(parents=True, exist_ok=False)
        started, snapshots, targets = now_utc().isoformat(), {}, {}
        for zone in zones:
            target, audit = _capture_target(root=Path(config["project_root"]), zone=zone,
                                             start=min(days), end=max(days), refresh=True)
            target_path = attempt / f"{zone.lower()}_canonical.csv.gz"
            _write_frame(target_path, target.rename("target").rename_axis("timestamp").reset_index())
            snapshots[zone] = {**audit, "snapshot_path": str(target_path), "snapshot_sha256": sha256(target_path)}
            targets[zone] = target
        counts = {day: {zone: int(np.isfinite(target.loc[
            pd.DatetimeIndex(target.index).tz_convert("Europe/Paris").date == pd.Timestamp(day).date()
        ].to_numpy(float)).sum()) for zone, target in targets.items()} for day in days}
        capture = {**original, **FLAGS, "kind": "retrospective_bootstrap_input_reuse",
                   "zones": zones, "delivery_days": days, "fresh_source_refresh": False,
                   "reused_input_manifest": _file(path), "reuse_scope": "retrospective_calibration_only",
                   "capture_started_at_utc": started, "capture_completed_at_utc": now_utc().isoformat(),
                   "target_snapshots": snapshots, "horizon_labels_available_at_capture": counts}
        new_path = attempt / "input_manifest.json"
        _write_json(new_path, capture)
        return {**capture, "manifest_path": str(new_path)}
    return None


def execute_trial(config: Mapping[str, Any], *, delivery_day: str, zones: list[str],
                  prospective: bool, device: str = "auto", threads: int = 4) -> dict[str, Any]:
    """Bootstrap past raw gaps or issue two future chains, never rerun sealed days."""
    from .lora_finetune import load_config, load_checkpoint
    from .evaluation import _prepare_shadow_panel
    from .prospective_inputs import prepare_trial_inputs, _capture_target
    from .prospective_auxiliary import forecast_trial_chains
    from .prospective_bootstrap import plan_bootstrap
    from chronos2_hourly.kalman_residual import KalmanResidualConfig
    day = _day(delivery_day)
    manifest = verify_trial(config)
    if type(threads) is not int or not 1 <= threads <= 64:
        raise ProspectiveTrialError("threads doit etre un entier entre 1 et 64.")
    if not zones or set(zones) - set(manifest["zones"]) or len(zones) != len(set(zones)):
        raise ProspectiveTrialError("Selection des pays invalide.")
    if prospective:
        window = emission_window(day)
        if pd.Timestamp(manifest.get("_effective_freeze_utc", manifest["created_at_utc"])) > pd.Timestamp(window["forecast_origin_utc"]):
            raise ProspectiveTrialError("La recette doit etre figee avant l'origine de la premiere emission.")
    output, root = Path(config["output_root"]), Path(config["project_root"])
    histories, required_days, already_complete = {}, set(), []
    for zone in zones:
        if (output / "days" / day / zone).exists():
            if prospective or not (output / "days" / day / zone / "manifest.json").exists():
                raise ProspectiveTrialError(f"{zone}/{day}: deja capture ou tentative incomplete; aucune reecriture.")
        history = load_history(manifest, zone)
        last = history.delivery_start_utc.max().tz_convert("Europe/Paris").date()
        if not prospective and last >= pd.Timestamp(day).date():
            already_complete.append(zone)
            continue
        histories[zone] = history
        required_days.update(d.date().isoformat() for d in pd.date_range(pd.Timestamp(last) + pd.Timedelta(days=1), day))
    if not required_days:
        return {"status": "already_complete", "published": [], "pending": [],
                "already_complete": already_complete, **FLAGS}
    active_zones = list(histories)
    capture_root = output / "captures" / (now_utc().strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    capture = None if prospective else _reuse_bootstrap_inputs(config, days=sorted(required_days),
                                                               zones=active_zones, schema=manifest["schema"])
    if capture is None:
        capture = prepare_trial_inputs(project_root=root, output_directory=capture_root,
                                       delivery_days=sorted(required_days), zones=active_zones,
                                       expected_schema=manifest["schema"])
    if prospective:
        emission_window(day)
        if any(capture["horizon_labels_available_at_capture"][day][zone] for zone in active_zones):
            raise ProspectiveTrialError("Prix du jour cible deja publie: impossible de compter ce forecast comme prospectif.")
    targets = {zone: _targets(capture, zone) for zone in active_zones}
    planning_end = (pd.Timestamp(day) - pd.Timedelta(days=1)).date().isoformat() if prospective else day
    plan = plan_bootstrap(histories, targets, planning_end)
    pending = plan["pending"]
    for item in pending:
        print(f"[LoRA16 trial] {item['zone']} {item['delivery_day']}: EN ATTENTE des prix canoniques "
              f"({item['observed_hours']}/{item['expected_hours']} h); aucun calcul pour cette journee.", flush=True)
    if prospective and pending:
        raise ProspectiveTrialError("Calibration prealable incomplete; aucune emission: " + "; ".join(
            f"{p['zone']}/{p['delivery_day']} {p['observed_hours']}/{p['expected_hours']} h" for p in pending))
    if prospective:
        for zone, history in histories.items():
            labelled = _attach_actuals(history, targets[zone])
            missing = ~np.isfinite(pd.to_numeric(labelled.actual, errors="coerce").to_numpy(float))
            if missing.any():
                first = pd.Timestamp(labelled.loc[missing, "delivery_start_utc"].iloc[0]).tz_convert("Europe/Paris")
                raise ProspectiveTrialError(f"{zone}: labels historiques encore absents ({int(missing.sum())} h, "
                                            f"premiere {first}); aucun chargement du modele.")
    ready_by_zone = plan["ready_by_zone"]
    if prospective:
        for zone in active_zones:
            ready_by_zone[zone].append(day)
    if not any(ready_by_zone.values()):
        return {"status": "waiting_for_observations", "published": [], "pending": pending,
                "already_complete": already_complete, **FLAGS}
    candidate = load_config(config["candidate_config"])
    original_panel = pd.read_parquet(capture["panel_path"])
    panel = _prepare_shadow_panel(original_panel, candidate)
    keys = [candidate.timestamp_column, candidate.origin_column, candidate.item_column]
    tags = original_panel[[*keys, "phase", "delivery_day"]].copy()
    for column in (candidate.timestamp_column, candidate.origin_column):
        tags[column] = pd.to_datetime(tags[column], utc=True)
    panel = panel.merge(tags, on=keys, how="left", validate="one_to_one")
    # Identical pinned checkpoint, loaded once; individual country inputs, no cross-learning.
    import torch
    torch.set_num_threads(threads)
    pipeline = load_checkpoint(manifest["zones"][active_zones[0]]["bundle"], device_map=device)
    published = []
    for zone in active_zones:
        history = _attach_actuals(histories[zone], targets[zone])
        for target_day in ready_by_zone[zone]:
            if target_day <= history.delivery_start_utc.max().tz_convert("Europe/Paris").date().isoformat():
                continue
            is_future = prospective and target_day == day
            if is_future:
                emission_window(day)
            group = panel.loc[panel.item_id.eq(zone) & panel.delivery_day.eq(target_day)]
            print(f"[LoRA16 trial] {zone} {target_day}: {'deux chaines prospectives' if is_future else 'calibration retrospective uniquement'}", flush=True)
            raw = _predict_group(group, candidate, pipeline)
            raw = _attach_actuals(raw, targets[zone])
            if not is_future and not np.isfinite(raw.actual.to_numpy()).all():
                raise ProspectiveTrialError(f"{zone}/{target_day}: labels manquants pour la calibration retrospective.")
            result = None
            if is_future:
                result = forecast_trial_chains(history, raw, recipe=ResidualRecipe(**manifest["recipe"]),
                                               kalman_config=KalmanResidualConfig(**manifest["kalman_config"]))
                fresh, check = _capture_target(root=root, zone=zone, start=day, end=day, refresh=True)
                if fresh.notna().any():
                    raise ProspectiveTrialError("Prix publie pendant le calcul: emission prospective refusee.")
                emission_window(day)
            daily_dir = output / "days" / target_day / zone
            daily_dir.mkdir(parents=True, exist_ok=False)
            daily = {**FLAGS, "zone": zone, "delivery_day": target_day,
                     "prospective_eligible": is_future, "completed_at_utc": now_utc().isoformat(),
                     "role": "prospective_two_chain_forecast" if is_future else "retrospective_calibration_only",
                     "trial_manifest_sha256": sha256(output / "trial_manifest.json"),
                     "input_manifest": _file(Path(capture["manifest_path"])),
                     "raw_predictions": _write_frame(daily_dir / "raw.csv.gz", raw)}
            if result is not None:
                combined = result.kalman_future.copy()
                combined["zone"], combined["actual"], combined["prospective_eligible"] = zone, np.nan, True
                daily["predictions"] = _write_frame(daily_dir / "predictions.csv.gz", combined)
                _write_json(daily_dir / "auxiliary_audit.json", result.audit)
                daily["auxiliary_audit"] = _file(daily_dir / "auxiliary_audit.json")
                daily["prepublication_target_check"] = check
                daily["prepublication_observed_hours"] = 0
                daily["emission_window"] = emission_window(day)
            _write_json(daily_dir / "manifest.json", daily)
            published.append(str(daily_dir / "manifest.json"))
            history = pd.concat([history, raw], ignore_index=True)
    return {"status": "waiting_for_observations" if pending else "completed", "published": published,
            "pending": pending, "already_complete": already_complete, **FLAGS}


def resolve_and_report(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read fresh observations into new report snapshots, without neural inference."""
    from .prospective_inputs import _capture_target
    from .prospective_reporting import render_trial_report
    verify_trial(config)
    output = Path(config["output_root"])
    frames, sources = [], []
    for path in sorted((output / "days").glob("*/*/manifest.json")):
        daily = _verify_daily(path, output=output)
        if not daily["prospective_eligible"]:
            continue
        frame = _read_frame(_verify_file(daily["predictions"]))
        observed, audit = _capture_target(root=Path(config["project_root"]), zone=daily["zone"],
                                          start=daily["delivery_day"], end=daily["delivery_day"], refresh=True)
        frames.append(_attach_actuals(frame, observed))
        sources.append({"emission_manifest": _file(path), "observation_read": audit})
    if not frames:
        return {"status": "no_prospective_forecasts", **FLAGS}
    destination = output / "reports" / (now_utc().strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    frame = pd.concat(frames, ignore_index=True)
    evidence = _write_frame(destination / "resolved_predictions.csv.gz", frame)
    report = render_trial_report(frame, destination / "rank16_trial.html", metadata=FLAGS)
    _write_json(destination / "report_manifest.json", {**FLAGS, "generated_at_utc": now_utc().isoformat(),
                "sources": sources, "evidence": evidence, "report": _file(report)})
    return {"status": "reported", "report": str(report), **FLAGS}


def trial_status(config: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(config["output_root"])
    if not (output / "trial_manifest.json").exists():
        return {"status": "not_prepared", "output_root": str(output), **FLAGS}
    manifest = verify_trial(config)
    rows = {}
    for zone in manifest["zones"]:
        history = load_history(manifest, zone)
        rows[zone] = {"raw_calibration_last_day": history.delivery_start_utc.max().tz_convert("Europe/Paris").date().isoformat(),
                      "prospective_forecasts": sum(_json(p).get("prospective_eligible", False)
                          for p in (output / "days").glob(f"*/{zone}/manifest.json"))}
    return {"status": "prepared", "models": manifest["models"], "zones": rows, **FLAGS}
