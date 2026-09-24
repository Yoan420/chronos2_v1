"""Isolated rolling-365 auxiliary replay of a fixed, selected LoRA checkpoint.

This is retrospective research, not neural OOF or a production qualification.
The residual history starts with an explicitly audited expanding-window warmup.
Only evaluated days have two complete, strictly prior 365-day fit windows.
No existing checkpoint, prospective forecast or operational cache is modified.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import date, timedelta
import hashlib
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import stat
from typing import Any, Mapping
import uuid

import numpy as np
import pandas as pd

from . import prospective_auxiliary as auxiliary


WINDOW_DAYS = 365
SCHEMA_VERSION = 1
CACHE_KEY_HEX_LENGTH = 24
PARTIAL_NONCE_HEX_LENGTH = 12


class RollingResearchAuxiliaryError(ValueError):
    """A research replay or immutable cache cannot establish its contract."""


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RollingResearchAuxiliaryError("Identite/audit JSON non fini ou non serialisable.") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _day_cache_path(root: Path, day: date, identity_sha256: str) -> Path:
    # The full SHA remains in the seal and is always compared on cache reads.
    # Short directory names avoid Windows MAX_PATH with the real experiment root.
    return root / "days" / str(day) / identity_sha256[:CACHE_KEY_HEX_LENGTH]


def _partial_cache_path(path: Path) -> Path:
    return path.parent / f".{path.name}.partial-{uuid.uuid4().hex[:PARTIAL_NONCE_HEX_LENGTH]}"


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_sha(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256(_json_bytes({"columns": list(frame.columns),
                                       "dtypes": [str(x) for x in frame.dtypes]}))
    values = pd.util.hash_pandas_object(frame, index=False, categorize=False).to_numpy(dtype="<u8")
    digest.update(values.tobytes())
    return digest.hexdigest()


def _regular_path(path: Path) -> None:
    for candidate in (path, *path.parents):
        if not candidate.exists() and not candidate.is_symlink():
            continue
        info = candidate.lstat()
        if candidate.is_symlink() or (getattr(info, "st_file_attributes", 0)
                                     & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
            raise RollingResearchAuxiliaryError(f"Lien/reparse interdit dans le cache: {candidate}.")


def _read_json(path: Path) -> dict[str, Any]:
    _regular_path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RollingResearchAuxiliaryError(f"Cache JSON absent ou invalide: {path}.") from exc
    if not isinstance(payload, dict):
        raise RollingResearchAuxiliaryError(f"Objet JSON requis: {path}.")
    return payload


def _write_json_new(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(_json_bytes(value) + b"\n")


def _day(value: str | date, *, label: str) -> date:
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.normalize():
            raise ValueError("jour civil sans fuseau attendu")
        return stamp.date()
    except (TypeError, ValueError) as exc:
        raise RollingResearchAuxiliaryError(f"{label}: jour civil YYYY-MM-DD requis.") from exc


def _normalise(raw: pd.DataFrame, *, evaluation_start: date, end_day: date,
               timezone: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if (end_day - evaluation_start).days != WINDOW_DAYS - 1:
        raise RollingResearchAuxiliaryError("Le rapport exige exactement 365 jours calendaires evalues.")
    if raw.columns.has_duplicates:
        raise RollingResearchAuxiliaryError("Colonnes dupliquees dans le support brut.")
    required = {"delivery_start_utc", "actual", *auxiliary.MARKET_COLUMNS,
                *(f"{auxiliary.RAW_MODEL}__{q}" for q in auxiliary.QUANTILES)}
    if not required.issubset(raw.columns):
        raise RollingResearchAuxiliaryError(f"Colonnes absentes: {sorted(required.difference(raw.columns))}.")
    frame = raw.copy(deep=True)
    frame["delivery_start_utc"] = auxiliary._index(frame["delivery_start_utc"], label="support brut")
    frame = frame.sort_values("delivery_start_utc", kind="stable").reset_index(drop=True)
    index = pd.DatetimeIndex(frame["delivery_start_utc"])
    expected = auxiliary._expected_index(evaluation_start - timedelta(days=WINDOW_DAYS),
                                         end_day + timedelta(days=1), timezone)
    if not index.equals(expected):
        raise RollingResearchAuxiliaryError(
            "Support exige: 365 jours anterieurs + 365 jours evalues, exactement 730 jours "
            "consecutifs avec toutes les heures physiques 23/24/25, sans doublon ni jour supplementaire.")
    try:
        actual = pd.to_numeric(frame["actual"], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise RollingResearchAuxiliaryError("Observations historiques non numeriques.") from exc
    if np.isinf(actual.to_numpy()).any():
        raise RollingResearchAuxiliaryError("Observation infinie interdite.")
    local_days = index.tz_convert(timezone).date
    prior = local_days < end_day
    if not np.isfinite(actual.loc[prior]).all():
        raise RollingResearchAuxiliaryError("Labels manquants avant D: calibration 365 incomplete.")
    frame["actual"] = actual
    observations = pd.Series(actual.to_numpy(), index=index, name="actual")
    history = auxiliary._normalise_raw(frame.loc[prior], timezone=timezone, history=True)
    # The last observation, whether known or absent, never enters a fit or a cache key.
    future = auxiliary._normalise_raw(frame.loc[~prior], timezone=timezone, history=False)
    return history, future, observations


def _contract(*, timezone: str, identity: Mapping[str, Any] | None) -> dict[str, Any]:
    from chronos2_hourly import kalman_covariates, kalman_residual

    config = kalman_residual.KalmanResidualConfig()
    config.validate()
    recipe = auxiliary.ResidualRecipe()
    recipe.validate()
    if tuple(config.candidate_kinds) != auxiliary.STANDARD_CANDIDATES:
        raise RollingResearchAuxiliaryError("Les cinq candidats Kalman standard sont requis.")
    declared_identity = dict(identity or {})
    for name, expected in (("recipe", asdict(recipe)), ("residual_recipe", asdict(recipe)),
                           ("kalman_config", asdict(config)), ("kalman_configuration", asdict(config))):
        if name not in declared_identity:
            continue
        declared = json.loads(_json_bytes(declared_identity[name]))
        normalised_expected = json.loads(_json_bytes(expected))
        if (not isinstance(declared, dict) or declared != normalised_expected
                or any(isinstance(declared.get(key), bool) for key, value in expected.items()
                       if isinstance(value, (int, float)) and not isinstance(value, bool))):
            raise RollingResearchAuxiliaryError(
                f"Identite {name} divergente: le calcul utilise exclusivement les parametres standard figes.")
    source_paths = (Path(__file__), Path(auxiliary.__file__), Path(kalman_residual.__file__),
                    Path(kalman_covariates.__file__))
    payload = {
        "kind": "chronos2_lora16_rolling365_research_auxiliary_cache",
        "schema_version": SCHEMA_VERSION, "timezone": timezone,
        "evaluation_days": WINDOW_DAYS, "training_days": WINDOW_DAYS,
        "identity": declared_identity, "residual_recipe": asdict(recipe),
        "kalman_config": asdict(config),
        "source_sha256": {path.name: _file_sha(path) for path in source_paths},
        "numerical_versions": {name: version(name) for name in ("numpy", "pandas", "pykalman")},
        "historical_corrector_protocol": "expanding_then_rolling365_prior_labels_fixed_checkpoint",
        "historical_neural_prefix_in_sample": True, "neural_oof": False,
        "historical_metrics_are_independent_test": False,
        "diagnostic_only": True, "production_pit_evidence": False,
        "production_pipeline_evidence": False, "promotion_eligible": False,
        "activation_performed": False,
    }
    # Freeze tuple/list representations and reject non-JSON or non-finite identities.
    return json.loads(_json_bytes(payload))


def _ensure_contract(root: Path, contract: dict[str, Any]) -> str:
    _regular_path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "cache_contract.json"
    expected = {"contract": contract, "contract_sha256": _digest(contract)}
    if path.exists():
        if _read_json(path) != expected:
            raise RollingResearchAuxiliaryError("Identite/recette/source divergente du cache; utiliser un nouveau dossier.")
    else:
        if any(root.iterdir()):
            raise RollingResearchAuxiliaryError("Dossier de cache non vide sans contrat; aucun fichier existant ne sera ecrase.")
        try:
            _write_json_new(path, expected)
        except FileExistsError:
            if _read_json(path) != expected:
                raise RollingResearchAuxiliaryError("Creation concurrente du contrat avec une identite divergente.")
    return expected["contract_sha256"]


def _validate_prediction(frame: pd.DataFrame, *, day: date, timezone: str) -> None:
    if "actual" in frame or "delivery_start_utc" not in frame:
        raise RollingResearchAuxiliaryError("Cache predictif invalide: timestamp absent ou actual conserve.")
    index = auxiliary._index(frame["delivery_start_utc"], label="cache predictif")
    if not index.equals(auxiliary._expected_index(day, day + timedelta(days=1), timezone)):
        raise RollingResearchAuxiliaryError("Heures physiques divergentes dans le cache predictif.")
    for model in (auxiliary.RAW_MODEL, auxiliary.RESIDUAL_MODEL, auxiliary.KALMAN_MODEL):
        columns = [f"{model}__{q}" for q in auxiliary.QUANTILES]
        if not set(columns).issubset(frame):
            raise RollingResearchAuxiliaryError(f"Quantiles absents du cache: {model}.")
        values = frame[columns].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values[:, :-1] > values[:, 1:]).any():
            raise RollingResearchAuxiliaryError("Quantiles non finis ou croises dans le cache.")


def _load_day(path: Path, expected_identity: dict[str, Any], *, timezone: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    _regular_path(path)
    seal = _read_json(path / "seal.json")
    if seal.get("identity") != expected_identity or seal.get("identity_sha256") != _digest(expected_identity):
        raise RollingResearchAuxiliaryError(f"Identite du cache journalier divergente: {path}.")
    expected_files = {"predictions.parquet", "audit.json"}
    if set(seal.get("files", {})) != expected_files or {p.name for p in path.iterdir()} != expected_files | {"seal.json"}:
        raise RollingResearchAuxiliaryError(f"Fichiers inattendus ou absents dans le cache: {path}.")
    for name in expected_files:
        candidate = path / name
        _regular_path(candidate)
        if not candidate.is_file() or _file_sha(candidate) != seal["files"][name]:
            raise RollingResearchAuxiliaryError(f"SHA divergent dans le cache journalier: {candidate}.")
    frame = pd.read_parquet(path / "predictions.parquet")
    audit = _read_json(path / "audit.json")
    if audit.get("daily_identity_sha256") != seal["identity_sha256"]:
        raise RollingResearchAuxiliaryError("Audit journalier non lie a son identite.")
    _validate_prediction(frame, day=_day(expected_identity["target_day"], label="cache"), timezone=timezone)
    return frame, audit


def _run_day(training: pd.DataFrame, future: pd.DataFrame, *, daily_identity: dict[str, Any],
             residual_fit: dict[str, Any], path: Path, timezone: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Process-pool boundary: numeric-only worker; no neural model is imported."""
    from chronos2_hourly.kalman_residual import KalmanResidualConfig
    from threadpoolctl import threadpool_limits

    _regular_path(path)
    if path.exists():
        return _load_day(path, daily_identity, timezone=timezone)
    with threadpool_limits(limits=1):
        fitted = auxiliary._fit_kalman_future(training, future, timezone=timezone, config=KalmanResidualConfig())
    if fitted["window_audit"].get("training_window_days") != WINDOW_DAYS:
        raise RollingResearchAuxiliaryError("Le Kalman n'a pas utilise les 365 jours declares.")
    if fitted["window_audit"].get("target_observations_assimilated") != 0:
        raise RollingResearchAuxiliaryError("Assimilation des observations du jour cible interdite.")
    predictions = future.merge(fitted["predictions"].reset_index(), on="delivery_start_utc",
                               how="left", validate="one_to_one")
    day = _day(daily_identity["target_day"], label="cible")
    _validate_prediction(predictions, day=day, timezone=timezone)
    audit = auxiliary._audit_values({
        "target_day": day.isoformat(), "daily_identity_sha256": _digest(daily_identity),
        "residual_fit": residual_fit, "daily_audit": fitted["daily_audit"],
        "window_audit": fitted["window_audit"], "state_audit": fitted["state_audit"],
        "market_scalers": fitted["market_scalers"],
        "covariate_audit": fitted["covariate_audit"],
        "covariate_columns": fitted["covariate_columns"],
        "historical_corrector_warmup": daily_identity["historical_corrector_warmup"],
        "diagnostic_only": True, "neural_oof": False, "promotion_eligible": False,
    })
    parent = path.parent
    _regular_path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    temporary = _partial_cache_path(path)
    temporary.mkdir()
    predictions.to_parquet(temporary / "predictions.parquet", index=False)
    _write_json_new(temporary / "audit.json", audit)
    _write_json_new(temporary / "seal.json", {
        "identity": daily_identity, "identity_sha256": _digest(daily_identity),
        "files": {name: _file_sha(temporary / name) for name in ("predictions.parquet", "audit.json")},
    })
    _load_day(temporary, daily_identity, timezone=timezone)
    # Rename a complete directory; existing published caches are never overwritten.
    if path.exists():
        return _load_day(path, daily_identity, timezone=timezone)
    try:
        temporary.rename(path)
    except OSError:
        if not path.exists():
            raise
        return _load_day(path, daily_identity, timezone=timezone)
    return _load_day(path, daily_identity, timezone=timezone)


def run_rolling_research_auxiliary(
    raw: pd.DataFrame, *, evaluation_start: str | date, end_day: str | date,
    output_directory: str | Path, timezone: str = "Europe/Paris", workers: int = 4,
    identity: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay exactly 365 physical daily origins with immutable per-origin caches.

    ``raw`` contains exactly the preceding 365 days and the 365 evaluation days.
    Only the last day's observations may be missing/partial. Current-day actuals
    are attached to the returned frame after inference, never to prediction files.
    Changing observations changes only cache identities for affected later fits.
    """
    if type(workers) is not int or not 1 <= workers <= 8:
        raise RollingResearchAuxiliaryError("workers doit etre un entier entre 1 et 8.")
    first, last = _day(evaluation_start, label="evaluation_start"), _day(end_day, label="end_day")
    history, future, observations = _normalise(raw, evaluation_start=first, end_day=last, timezone=timezone)
    contract = _contract(timezone=timezone, identity=identity)
    root = Path(output_directory).absolute()
    contract_sha = _ensure_contract(root, contract)
    # This is deliberately one pass, not 365 reconstructions of the same history.
    corrected, residual_audits = auxiliary.build_prequential_residual_history(history, timezone=timezone)
    final_fit = auxiliary.fit_residual_corrector(history, target_day=last, timezone=timezone, require_full_window=True)
    final_corrected = auxiliary.apply_residual_corrector(future, final_fit)
    residual_audits.append(final_fit.to_audit())
    all_corrected = pd.concat([corrected, final_corrected], ignore_index=True)
    local_days = pd.DatetimeIndex(all_corrected["delivery_start_utc"]).tz_convert(timezone).date
    fits = {record["target_day"]: record for record in residual_audits}
    blocks = {day: (int(offsets.iloc[0]), int(offsets.iloc[-1]) + 1)
              for day, offsets in pd.Series(np.arange(len(local_days))).groupby(local_days, sort=True)}
    tasks, results = [], {}
    for offset in range(WINDOW_DAYS):
        target = first + timedelta(days=offset)
        left = target - timedelta(days=WINDOW_DAYS)
        start_at, stop_at = blocks[left][0], blocks[target][0]
        fit = fits[target.isoformat()]
        if fit["training_days"] != WINDOW_DAYS or fit["training_end_day"] != str(target - timedelta(days=1)):
            raise RollingResearchAuxiliaryError("Fenetre residuelle evaluee non complete ou non causale.")
        training = all_corrected.iloc[start_at:stop_at].copy()
        target_frame = all_corrected.iloc[slice(*blocks[target])].drop(columns="actual").copy()
        prior_fits = [fits[str(left + timedelta(days=i))] for i in range(WINDOW_DAYS)]
        warmup = {
            "history_start_day": str(local_days[0]),
            "identity_cold_start_days": sum(bool(x["identity_cold_start"]) for x in prior_fits),
            "expanding_fit_days": sum(not x["identity_cold_start"] and x["training_days"] < WINDOW_DAYS for x in prior_fits),
            "full_365_fit_days": sum(x["training_days"] == WINDOW_DAYS for x in prior_fits),
            "minimum_residual_training_days": min(x["training_days"] for x in prior_fits),
            "maximum_residual_training_days": max(x["training_days"] for x in prior_fits),
            "historical_residual_fits_sha256": _digest(prior_fits),
            "all_historical_correctors_trained_on_365_days": all(x["training_days"] == WINDOW_DAYS for x in prior_fits),
        }
        day_identity = {"contract_sha256": contract_sha, "target_day": str(target),
                        "training_start_day": str(left), "training_end_day": str(target - timedelta(days=1)),
                        "training_sha256": _frame_sha(training), "future_without_actual_sha256": _frame_sha(target_frame),
                        "residual_fit_sha256": _digest(fit), "historical_corrector_warmup": warmup}
        path = _day_cache_path(root, target, _digest(day_identity))
        _regular_path(path)
        if path.exists():
            results[target] = _load_day(path, day_identity, timezone=timezone)
        else:
            tasks.append((target, start_at, stop_at, path, day_identity, fit))
    cached_days = len(results)
    print(f"[LoRA365] Kalman: {cached_days}/365 caches verifies; {len(tasks)} refits; workers={workers}.", flush=True)

    def arguments(task: tuple[Any, ...]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        target, start_at, stop_at, path, day_identity, fit = task
        return (all_corrected.iloc[start_at:stop_at].copy(),
                all_corrected.iloc[slice(*blocks[target])].drop(columns="actual").copy()), {
                    "daily_identity": day_identity, "residual_fit": fit, "path": path, "timezone": timezone}

    def record(task: tuple[Any, ...], result: tuple[pd.DataFrame, dict[str, Any]]) -> None:
        results[task[0]] = result
        print(f"[LoRA365] Kalman {len(results)}/365 {task[0]}: calcule et scelle.", flush=True)

    if workers == 1:
        for task in tasks:
            args, kwargs = arguments(task)
            record(task, _run_day(*args, **kwargs))
    elif tasks:
        # Bound queued frames instead of pickling 365 whole windows at once.
        iterator = iter(tasks)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            pending = {}
            def enqueue() -> bool:
                task = next(iterator, None)
                if task is None:
                    return False
                args, kwargs = arguments(task)
                pending[pool.submit(_run_day, *args, **kwargs)] = task
                return True
            for _ in range(workers * 2):
                if not enqueue():
                    break
            try:
                while pending:
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for completed in finished:
                        task = pending.pop(completed)
                        record(task, completed.result())
                        enqueue()
            except BaseException:
                for pending_future in pending:
                    pending_future.cancel()
                raise
    ordered = [results[first + timedelta(days=i)] for i in range(WINDOW_DAYS)]
    predictions = pd.concat([item[0] for item in ordered], ignore_index=True)
    index = pd.DatetimeIndex(predictions["delivery_start_utc"])
    expected = auxiliary._expected_index(first, last + timedelta(days=1), timezone)
    if not index.equals(expected):
        raise RollingResearchAuxiliaryError("Agregat final incomplet ou heures dupliquees.")
    predictions["actual"] = observations.reindex(index).to_numpy()
    audits = [item[1] for item in ordered]
    audit = {
        **contract, "cache_contract_sha256": contract_sha,
        "evaluation_start_day": str(first), "evaluation_end_day": str(last),
        "evaluation_hours": len(predictions), "raw_support_start_day": str(local_days[0]),
        "raw_support_days": len(blocks), "cached_days": cached_days, "computed_days": len(tasks),
        "target_observations_used_for_same_day_fit": 0,
        "residual_history_built_once": True,
        "evaluated_residual_training_window_days": WINDOW_DAYS,
        "evaluated_kalman_training_window_days": WINDOW_DAYS,
        "historical_correctors_all_have_full365": False,
        "residual_fits": [fits[str(first + timedelta(days=i))] for i in range(WINDOW_DAYS)],
        "historical_residual_fits": residual_audits,
        "daily_audit": [item["daily_audit"] for item in audits],
        "window_audit": [item["window_audit"] for item in audits],
        "state_audit": [row for item in audits for row in item["state_audit"]],
        "daily_fit_audits": audits,
    }
    return predictions, auxiliary._audit_values(audit)


__all__ = ["RollingResearchAuxiliaryError", "run_rolling_research_auxiliary"]
