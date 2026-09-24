"""Isolated 365-day research reports for the selected rank-16 checkpoint.

No production publisher, activation, original bundle or prospective ledger is
written. The historical neural calibration prefix is explicitly in-sample.
"""
from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager
import hashlib
import json
import os
from time import monotonic
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from . import prospective_trial as trial
from .prospective_auxiliary import MARKET_COLUMNS, RAW_MODEL, QUANTILES


class RollingResearchError(ValueError):
    pass


FLAGS = {**trial.FLAGS, "prospective_eligible": False,
         "historical_metrics_are_independent_test": False,
         "neural_calibration_prefix_in_sample": True,
         "kalman_calibration_residual_warmup": "identity_30_days_then_expanding_to_365",
         "kind": "rank16_fixed_checkpoint_rolling365_research_v1"}
OUTPUT_NAME = "chronos2_exogenous_rank16_rolling365_research_v1"


def _bounds(delivery_day: str) -> tuple[str, str, str]:
    end = pd.Timestamp(trial._day(delivery_day))
    return ((end - pd.Timedelta(days=729)).date().isoformat(),
            (end - pd.Timedelta(days=364)).date().isoformat(), end.date().isoformat())


def _local_days(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame.delivery_start_utc, utc=True).dt.tz_convert("Europe/Paris").dt.strftime("%Y-%m-%d")


def _raw_identity(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    # Reporting/code layout changes cannot invalidate expensive neural inference.
    # All numerical inference functions live in the frozen, verified trial code.
    from . import rolling_research_prefix
    return {"protocol": "selected_rank16_training_prefix_inference_v1",
            "checkpoint_sha256": config["checkpoint_sha256"],
            "panel": manifest["panel"], "schema": manifest["schema"],
            "inference_code": [entry for entry in manifest["code_files"]
                               if Path(entry["path"]).name in {"evaluation.py", "lora_finetune.py"}],
            "prefix_inference_code": trial._file(Path(rolling_research_prefix.__file__)),
            "batch_size": 64, "horizon_observations_used": 0, "neural_oof": False}


def _prefix_record(directory: Path, identity: Mapping[str, Any], *, zone: str, day: str) -> dict | None:
    records = sorted(directory.glob("*/manifest.json")) if directory.exists() else []
    if len(records) > 1:
        raise RollingResearchError(f"{zone}/{day}: plusieurs calculs bruts scelles, inspection requise.")
    if not records:
        return None
    record = trial._json(records[0])
    if record.get("identity") != dict(identity) or record.get("zone") != zone or record.get("delivery_day") != day:
        raise RollingResearchError(f"{zone}/{day}: identite du prefixe divergente; aucun ecrasement.")
    path = trial._verify_file(record["raw"])
    if path.resolve() != (records[0].parent / "raw.csv.gz").resolve():
        raise RollingResearchError("Artefact brut hors de son dossier scelle.")
    return record


def reconstruct_prefix(config: Mapping[str, Any], manifest: Mapping[str, Any], *, zone: str,
                       start_day: str, device: str = "auto", threads: int = 4,
                       max_new_days: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Reuse every sealed day; infer only missing training-prefix days, sequentially."""
    from .lora_finetune import load_config, load_checkpoint
    from .rolling_research_prefix import predict_prefix_group
    panel_path = trial._verify_file(manifest["panel"])
    candidate = load_config(config["candidate_config"])
    identity = _raw_identity(config, manifest)
    identity_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(config["project_root"]) / "runs/experiments" / OUTPUT_NAME / "raw_prefix" / identity_key / zone
    expected = pd.date_range(start_day, "2025-09-02", freq="D").strftime("%Y-%m-%d").tolist()
    if not expected or start_day < "2024-09-03":
        raise RollingResearchError("Le panel original ne couvre pas les 365 jours de calibration requis.")
    records, pending = {}, []
    for day in expected:
        record = _prefix_record(directory / day, identity, zone=zone, day=day)
        if record is None:
            pending.append(day)
        else:
            records[day] = record
    requested = pending if max_new_days is None else pending[:max_new_days]
    print(f"[LoRA365] {zone}: prefixe neural IN-SAMPLE; {len(records)}/{len(expected)} jours reutilisables, "
          f"{len(requested)} a calculer.", flush=True)
    if requested:
        import torch
        torch.set_num_threads(threads)
        pipeline = load_checkpoint(manifest["zones"][zone]["bundle"], device_map=device)
        started = monotonic()
        # Bounded parquet reads; the 730-origin panel expands substantially in RAM.
        for offset in range(0, len(requested), 8):
            chunk = pd.read_parquet(panel_path, filters=[("item_id", "==", zone),
                                                        ("delivery_day", "in", requested[offset:offset + 8])])
            for number, day in enumerate(requested[offset:offset + 8], offset + 1):
                group = chunk.loc[chunk.delivery_day.eq(day)].copy()
                if group.empty:
                    raise RollingResearchError(f"{zone}/{day}: origine absente du panel fige.")
                horizon = group.loc[group.phase.eq("horizon")].sort_values("timestamp")
                labels = pd.Series(horizon.target.to_numpy(float), index=pd.to_datetime(horizon.timestamp, utc=True))
                if not np.isfinite(labels.to_numpy()).all():
                    raise RollingResearchError(f"{zone}/{day}: labels de calibration incomplets.")
                raw, input_audit = predict_prefix_group(group, candidate, pipeline)
                raw["actual"] = labels.reindex(pd.DatetimeIndex(raw.delivery_start_utc)).to_numpy(float)
                attempt = directory / day / uuid4().hex[:12]
                attempt.mkdir(parents=True, exist_ok=False)
                record = {**FLAGS, "identity": identity, "zone": zone, "delivery_day": day,
                          "created_at_utc": trial.now_utc().isoformat(),
                          "role": "in_sample_neural_calibration_prefix_not_evaluated",
                          "input_audit": input_audit,
                          "raw": trial._write_frame(attempt / "raw.csv.gz", raw)}
                trial._write_json(attempt / "manifest.json", record)
                records[day] = record
                elapsed = monotonic() - started
                remaining = len(pending) - number
                print(f"[LoRA365] {zone}: prefixe {len(records)}/{len(expected)} {day}; "
                      f"{elapsed / number:.1f} s/jour; reste estime {remaining * elapsed / number / 60:.1f} min.", flush=True)
        del pipeline
    frame = pd.concat([trial._read_frame(trial._verify_file(records[d]["raw"])) for d in expected if d in records],
                      ignore_index=True) if records else pd.DataFrame()
    return frame, {"expected_days": len(expected), "completed_days": len(records),
                   "complete": len(records) == len(expected), "identity": identity,
                   "neural_oof": False, "neural_in_sample": True,
                   "sources": [records[d]["raw"] for d in expected if d in records]}


def _recent_sources(config: Mapping[str, Any], manifest: Mapping[str, Any], zone: str,
                    end_day: str) -> tuple[pd.DataFrame, list[dict]]:
    from .retrospective_trial import _comparison_identity, _verify_zone_cache
    history = trial.load_history(manifest, zone)
    history = history.loc[_local_days(history).le(end_day)].copy()
    sources = [manifest["zones"][zone]["calibration"]]
    for path in sorted((Path(config["output_root"]) / "days").glob(f"*/{zone}/manifest.json")):
        if path.parent.parent.name <= end_day:
            sources.append(trial._file(path))
    if not _local_days(history).eq(end_day).any():
        path = Path(config["output_root"]) / "retrospective" / end_day / zone / "forecast_manifest.json"
        if not path.exists():
            raise RollingResearchError(f"{zone}/{end_day}: LoRA brut absent. Lancer LoRATrial -Action Compare pour cette date.")
        record = _verify_zone_cache(path, _comparison_identity(config, manifest), zone=zone, day=end_day)
        history = pd.concat([history, trial._read_frame(trial._verify_file(record["raw_future"]))], ignore_index=True)
        sources.append(trial._file(path))
    history = history.sort_values("delivery_start_utc").reset_index(drop=True)
    if history.delivery_start_utc.duplicated().any():
        raise RollingResearchError("Plusieurs forecasts bruts pour une meme heure.")
    return history, sources


def _report_inputs(config: Mapping[str, Any], manifest: Mapping[str, Any], *, zone: str,
                   start_day: str, end_day: str) -> tuple[pd.DataFrame, list[dict]]:
    from .retrospective_trial import _find_existing_panel
    columns = ["timestamp", "delivery_day", *manifest["schema"]["known_future_covariates"]]
    original = trial._verify_file(manifest["panel"])
    frames = [pd.read_parquet(original, columns=columns, filters=[("phase", "==", "horizon"),
                ("item_id", "==", zone), ("delivery_day", ">=", start_day), ("delivery_day", "<=", end_day)])]
    sources = [manifest["panel"]]
    recent_days = pd.date_range(max(start_day, "2026-09-03"), end_day).strftime("%Y-%m-%d").tolist()
    panels = {}
    for day in recent_days:
        capture = _find_existing_panel(config, manifest, day, [zone])
        if capture is None:
            raise RollingResearchError(f"{zone}/{day}: inputs de rapport absents; aucun input incumbent substitue.")
        panels.setdefault(capture["panel_path"], {"capture": capture, "days": []})["days"].append(day)
    for path, entry in panels.items():
        frames.append(pd.read_parquet(path, columns=columns, filters=[("phase", "==", "horizon"),
                      ("item_id", "==", zone), ("delivery_day", "in", entry["days"])]))
        sources.append(trial._file(Path(entry["capture"]["manifest_path"])))
    inputs = pd.concat(frames, ignore_index=True).drop(columns="delivery_day")
    inputs["timestamp"] = pd.to_datetime(inputs.timestamp, utc=True)
    expected = pd.date_range(pd.Timestamp(start_day, tz="Europe/Paris"),
        pd.Timestamp(end_day, tz="Europe/Paris") + pd.DateOffset(days=1), freq="h", inclusive="left").tz_convert("UTC")
    inputs = inputs.sort_values("timestamp").reset_index(drop=True)
    if not pd.DatetimeIndex(inputs.timestamp).equals(expected):
        raise RollingResearchError("Inputs: les 365 jours physiques doivent etre complets, sans doublon.")
    return inputs, sources


def _run_rolling_report(config: Mapping[str, Any], *, delivery_day: str, zones: Sequence[str],
                       device: str = "auto", threads: int = 4, workers: int = 4,
                       stage: str = "run", max_new_prefix_days: int | None = None) -> dict:
    if stage not in {"plan", "prefix", "run"}:
        raise RollingResearchError("Etape inconnue.")
    if type(threads) is not int or not 1 <= threads <= 64 or type(workers) is not int or not 1 <= workers <= 8:
        raise RollingResearchError("threads 1..64 et workers 1..8 requis.")
    if max_new_prefix_days is not None and (type(max_new_prefix_days) is not int or max_new_prefix_days < 1):
        raise RollingResearchError("max_new_prefix_days doit etre un entier positif.")
    support_start, start, end = _bounds(delivery_day)
    if support_start < "2024-09-03" or start <= "2025-09-02":
        raise RollingResearchError("La fenetre evaluee doit suivre la validation LoRA et disposer de 365 jours de support.")
    if (pd.Timestamp(end) - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris") > trial.now_utc():
        raise RollingResearchError("Reconstitution impossible avant l'origine de prevision.")
    manifest = trial.verify_trial(config)
    if not zones or len(zones) != len(set(zones)) or set(zones) - set(manifest["zones"]):
        raise RollingResearchError("Selection de pays invalide.")
    root = Path(config["project_root"])
    output = root / "runs/experiments" / OUTPUT_NAME / end
    result = {**FLAGS, "evaluation_start": start, "evaluation_end": end, "evaluation_days": 365,
              "support_start": support_start, "calibration_days": 365, "output_directory": str(output), "zones": {}}
    # Preflight every selected country before spending compute on the first.
    recent, inputs, source_refs = {}, {}, {}
    for zone in zones:
        recent[zone], raw_refs = _recent_sources(config, manifest, zone, end)
        inputs[zone], input_refs = _report_inputs(config, manifest, zone=zone, start_day=start, end_day=end)
        source_refs[zone] = {"recent": raw_refs, "inputs": input_refs}
        dates = _local_days(recent[zone])
        prior = recent[zone].loc[dates.lt(end)]
        if not np.isfinite(prior.actual.to_numpy(float)).all():
            raise RollingResearchError(f"{zone}: observations de calibration anterieures manquantes; completer Bootstrap.")
        from .prospective_auxiliary import _normalise_raw
        _normalise_raw(prior, timezone="Europe/Paris", history=True)
        expected_recent = pd.date_range("2025-09-03", end).strftime("%Y-%m-%d").tolist()
        if list(dict.fromkeys(dates)) != expected_recent:
            raise RollingResearchError(f"{zone}: historique recent non contigu jusqu'a la livraison demandee.")
        result["zones"][zone] = {"prefix_days_needed": (pd.Timestamp("2025-09-03") - pd.Timestamp(support_start)).days,
                                  "recent_raw_days": dates.nunique()}
    if stage == "plan":
        return {**result, "status": "ready_for_research_replay"}
    for zone in zones:
        prefix, prefix_audit = reconstruct_prefix(config, manifest, zone=zone, start_day=support_start,
            device=device, threads=threads, max_new_days=max_new_prefix_days)
        if stage == "prefix" or not prefix_audit["complete"]:
            result["zones"][zone].update(prefix_audit)
            continue
        from .rolling_research_auxiliary import run_rolling_research_auxiliary
        from .rolling_research_comparators import load_rolling_report_comparators
        from .rolling_research_reporting import render_rolling_research_reports
        raw = pd.concat([prefix, recent[zone]], ignore_index=True).sort_values("delivery_start_utc").reset_index(drop=True)
        raw = raw.loc[_local_days(raw).between(support_start, end)].copy()
        identity = {"protocol": FLAGS["kind"], "history_start": support_start,
                    "checkpoint_sha256": config["checkpoint_sha256"], "trial_manifest": trial._file(
                        Path(config["output_root"]) / "trial_manifest.json"), "recipe": manifest["recipe"],
                    "kalman_config": manifest["kalman_config"], "zone": zone}
        predictions, auxiliary_audit = run_rolling_research_auxiliary(raw, evaluation_start=start, end_day=end,
            output_directory=output / zone / "auxiliary", workers=workers, identity=identity)
        # Reporting observations are fetched from verified incumbent report snapshots
        # only AFTER all predictions exist; never feed them to auxiliary calibration.
        storm, comparison_audit = load_rolling_report_comparators(root, zone=zone, delivery_day=end, start_day=start)
        report_stamp = trial.now_utc().strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        report_dir = output / zone / "reports" / report_stamp
        report_dir.mkdir(parents=True, exist_ok=False)
        display_target = pd.Series(raw.actual.to_numpy(float).copy(), index=pd.DatetimeIndex(raw.delivery_start_utc))
        if "actual" in storm:
            observed = pd.Series(storm.actual.to_numpy(float), index=pd.to_datetime(storm.timestamp, utc=True))
            predictions["actual"] = observed.reindex(pd.DatetimeIndex(predictions.delivery_start_utc)).to_numpy(float)
            common = display_target.index.intersection(observed.index)
            display_target.loc[common] = observed.reindex(common).to_numpy(float)
        evidence = trial._write_frame(report_dir / "evaluation_predictions.csv.gz", predictions)
        trial._write_json(report_dir / "auxiliary_audit.json", auxiliary_audit)
        metadata = {**result, "zone": zone, "identity": identity, "comparator_audit": comparison_audit,
                    "kalman_history_start_differs_from_single_day_trial": True,
                    "attribution_recomputed": False, "neural_retrained_daily": False}
        rendered_paths = render_rolling_research_reports(predictions, inputs[zone], output_directory=report_dir,
            zone=zone, delivery_day=end, metadata=metadata, storm=storm,
            storm_contract=comparison_audit.get("storm_contract"),
            history_target=display_target)
        reports = {model: str(path) for model, path in rendered_paths.items()}
        report_manifest = {**metadata, "sources": source_refs[zone], "prefix_audit": prefix_audit,
            "evidence": evidence, "created_at_utc": trial.now_utc().isoformat(),
            "reports": {model: trial._file(path) for model, path in rendered_paths.items()},
            "auxiliary_audit": trial._file(report_dir / "auxiliary_audit.json"),
            "renderer_code": trial._file(root / "chronos2_exogenous/rolling_research_reporting.py")}
        trial._write_json(report_dir / "report_manifest.json", report_manifest)
        result["zones"][zone].update({"reports": reports, "report_directory": str(report_dir)})
    complete = all("reports" in entry for entry in result["zones"].values())
    return {**result, "status": "rolling_research_reports_completed" if complete else "prefix_stage_completed"}


@contextmanager
def _compute_lock(directory: Path):
    """OS lock released even on process exit; never infer liveness from a stale PID."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".compute.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RollingResearchError("Un FullReport LoRA365 est deja en cours; ne pas lancer un doublon.") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run_rolling_report(config: Mapping[str, Any], **kwargs: Any) -> dict:
    if kwargs.get("stage") == "plan" or "project_root" not in config:
        return _run_rolling_report(config, **kwargs)
    directory = Path(config["project_root"]) / "runs/experiments" / OUTPUT_NAME
    with _compute_lock(directory):
        return _run_rolling_report(config, **kwargs)


__all__ = ["run_rolling_report", "reconstruct_prefix", "RollingResearchError"]
