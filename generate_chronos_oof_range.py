#!/usr/bin/env python
"""Generate a canonical Chronos-2 OOF artifact for one explicit zone/date range.

The generator reuses the hourly runner's strict delivery-day adapter:

* delivery dates are civil dates in the selected zone's configured timezone;
* every 23/24/25-hour day is represented on the canonical UTC timeline;
* the trusted forecast origin is D-1 at the configured local cutoff;
* the target context contains only already-published day-ahead prices;
* output is checkpointed atomically and can be resumed.

``native_covariates`` reproduces the Chronos expert used by the hourly runner.
``price_only`` is a separate, homogeneous ablation and must not be silently
mixed with native-covariate predictions when training a residual corrector.
No external or legacy price forecast is read by this utility.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from chronos2_hourly.chronos_adapter import (
    ChronosDeliveryPlan,
    execute_grouped_chronos_backtest,
    generate_delivery_plans,
    load_chronos_oof,
    make_existing_forecasting_executor,
    normalize_chronos_oof,
)
from chronos2_modular.common import (
    CALENDAR_COLUMNS,
    ZoneConfig,
    build_zone_configs,
    deep_get,
    load_yaml,
    set_reproducibility,
)
from chronos2_modular.data import prepare_zone_data
from chronos2_modular.forecasting import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Génère un artefact Chronos OOF horaire causal et DST-safe "
            "pour une zone du YAML."
        )
    )
    parser.add_argument(
        "--config",
        default="chronos2_hourly_fr_baseline_v235.yaml",
    )
    parser.add_argument(
        "--zone",
        default="FR",
        help="Zone du YAML à traiter (FR par défaut, insensible à la casse).",
    )
    parser.add_argument("--start-day", required=True, help="Date locale YYYY-MM-DD.")
    parser.add_argument("--end-day", required=True, help="Date locale YYYY-MM-DD incluse.")
    parser.add_argument("--output", required=True, help="CSV.GZ ou Parquet de sortie.")
    parser.add_argument(
        "--mode",
        choices=("native_covariates", "price_only"),
        default="native_covariates",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--origin-batch-size", type=int, default=None)
    parser.add_argument("--model-batch-size", type=int, default=None)
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=31,
        help="Nombre de jours entre deux checkpoints atomiques.",
    )
    parser.add_argument(
        "--minimum-native-coverage",
        type=float,
        default=0.90,
        help=(
            "Couverture minimale de chaque covariable future sur la plage; "
            "ignoré en mode price_only."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reprend un préfixe canonique déjà présent dans --output.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Audite la plage et la couverture sans charger Chronos.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_local_day(raw: str, *, name: str) -> pd.Timestamp:
    value = pd.Timestamp(raw)
    if value.tzinfo is not None or value != value.normalize():
        raise ValueError(f"{name} doit être une date locale naïve YYYY-MM-DD.")
    return value


def _select_zone(config: Mapping[str, Any], requested_zone: str) -> ZoneConfig:
    """Return exactly one configured zone, after canonicalising its code."""

    zone_code = str(requested_zone).strip().upper()
    if not zone_code:
        raise ValueError("--zone ne peut pas être vide.")
    zones = build_zone_configs(config, [zone_code], None, None)
    if len(zones) != 1 or str(zones[0].zone).upper() != zone_code:
        raise ValueError(
            "La configuration doit sélectionner exactement la zone "
            f"{zone_code}."
        )
    return zones[0]


def _local_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve a cached Hub id to a real directory, without any Hub request.

    Some Transformers releases still probe ``adapter_config.json`` on the Hub
    even when ``local_files_only=True`` is forwarded to ``from_pretrained``.
    Passing the already-cached snapshot directory avoids that non-causal
    network dependency entirely.
    """

    resolved = deepcopy(dict(config))
    model_id = str(deep_get(config, "model.model_id", "amazon/chronos-2"))
    candidate = Path(model_id).expanduser()
    if candidate.is_dir():
        snapshot = candidate.resolve()
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - Chronos installs it.
            raise RuntimeError(
                "huggingface_hub est requis pour résoudre le snapshot local."
            ) from exc
        snapshot = Path(
            snapshot_download(repo_id=model_id, local_files_only=True)
        ).resolve()
    required = (snapshot / "config.json", snapshot / "model.safetensors")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Snapshot Chronos local incomplet; fichier(s) absent(s): "
            + ", ".join(missing)
        )
    model = dict(resolved.get("model", {}) or {})
    model["model_id"] = str(snapshot)
    model["local_files_only"] = True
    resolved["model"] = model
    return resolved


def _expected_index(plans: Sequence[ChronosDeliveryPlan]) -> pd.DatetimeIndex:
    if not plans:
        raise ValueError("Aucun plan de livraison.")
    return pd.DatetimeIndex(
        np.concatenate([plan.delivery_index_utc.asi8 for plan in plans]),
        tz="UTC",
        name="delivery_start_utc",
    )


def _coverage_audit(data: Any, plans: Sequence[ChronosDeliveryPlan]) -> dict[str, Any]:
    index = _expected_index(plans)
    missing_index = index.difference(data.model_context_covariates.index.tz_convert("UTC"))
    if len(missing_index):
        raise ValueError(
            "Les covariables modèle ne couvrent pas la plage demandée: "
            f"{len(missing_index)} heure(s) absente(s)."
        )
    frame = data.model_context_covariates.copy()
    frame.index = frame.index.tz_convert("UTC")
    non_calendar = [
        column
        for column in data.known_future_columns
        if column not in CALENDAR_COLUMNS
    ]
    selected = frame.loc[index, non_calendar]
    by_column = {
        str(column): float(pd.to_numeric(selected[column], errors="coerce").notna().mean())
        for column in selected.columns
    }
    complete_rows = (
        float(selected.notna().all(axis=1).mean())
        if len(selected.columns)
        else 1.0
    )
    local = index.tz_convert(data.timezone)
    return {
        "n_delivery_days": int(len(plans)),
        "n_delivery_hours": int(len(index)),
        "first_delivery_utc": str(index[0]),
        "last_delivery_utc": str(index[-1]),
        "dst_day_lengths": {
            str(length): int(sum(plan.horizon == length for plan in plans))
            for length in (23, 24, 25)
        },
        "first_local_delivery": str(local[0]),
        "last_local_delivery": str(local[-1]),
        "known_future_non_calendar_columns": non_calendar,
        "coverage_by_column": by_column,
        "complete_known_future_row_coverage": complete_rows,
    }


def _validate_existing_prefix(
    path: Path,
    plans: Sequence[ChronosDeliveryPlan],
    target: pd.Series,
) -> tuple[pd.DataFrame | None, int]:
    if not path.exists():
        return None, 0
    existing = load_chronos_oof(path)
    expected = _expected_index(plans)
    if len(existing) > len(expected) or not existing.index.equals(expected[: len(existing)]):
        raise ValueError(
            "L'artefact à reprendre n'est pas un préfixe exact de la plage demandée."
        )
    target_utc = target.copy()
    target_utc.index = target_utc.index.tz_convert("UTC")
    expected_actual = target_utc.loc[existing.index].to_numpy(dtype=float)
    if not np.allclose(
        existing["actual"].to_numpy(dtype=float),
        expected_actual,
        rtol=0.0,
        atol=5e-5,
    ):
        raise ValueError("La cible de l'artefact à reprendre diffère de la cible canonique.")

    completed = 0
    cursor = 0
    for plan in plans:
        next_cursor = cursor + plan.horizon
        if next_cursor <= len(existing):
            completed += 1
            cursor = next_cursor
            continue
        if cursor != len(existing):
            raise ValueError("Le checkpoint s'arrête au milieu d'un jour de livraison.")
        break
    return existing, completed


def _atomic_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix not in {".gz", ".parquet", ".pq"}:
        raise ValueError("--output doit se terminer par .csv.gz, .parquet ou .pq.")
    temporary_suffix = ".parquet" if suffix in {".parquet", ".pq"} else ".csv.gz"
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.",
        suffix=temporary_suffix,
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        export = frame.reset_index()
        if suffix in {".parquet", ".pq"}:
            export.to_parquet(temporary, index=False)
        else:
            export.to_csv(temporary, index=False, compression="gzip")
        load_chronos_oof(temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_manifest(
    path: Path,
    *,
    config_path: Path,
    args: argparse.Namespace,
    audit: Mapping[str, Any],
    status: str,
    elapsed_seconds: float,
    completed_days: int,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "generator": str(Path(__file__).resolve()),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "zone": args.zone,
        "mode": args.mode,
        "start_day": args.start_day,
        "end_day": args.end_day,
        "context_length": args.context_length,
        "origin_batch_size": args.origin_batch_size,
        "model_batch_size": args.model_batch_size,
        "chunk_days": args.chunk_days,
        "completed_days": int(completed_days),
        "elapsed_seconds": float(elapsed_seconds),
        "audit": dict(audit),
    }
    if path.exists():
        payload["output"] = str(path.resolve())
        payload["output_size_bytes"] = int(path.stat().st_size)
        payload["output_sha256"] = _sha256(path)
    manifest = path.with_name(path.name + ".manifest.json")
    with manifest.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.chunk_days < 1:
        raise ValueError("--chunk-days doit être >= 1.")
    if not 0.0 <= args.minimum_native_coverage <= 1.0:
        raise ValueError("--minimum-native-coverage doit être dans [0, 1].")

    start = _parse_local_day(args.start_day, name="--start-day")
    end = _parse_local_day(args.end_day, name="--end-day")
    if end < start:
        raise ValueError("--end-day doit être postérieur ou égal à --start-day.")

    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    config_dir = config_path.parent
    set_reproducibility(int(deep_get(config, "model.seed", 42)))
    zone = _select_zone(config, args.zone)
    args.zone = zone.zone
    plans = generate_delivery_plans(
        start.date(),
        end.date(),
        forecast_origin_local_time=str(
            deep_get(config, "data.forecast_origin_local_time", "08:00")
        ),
        timezone=zone.timezone,
    )

    output = Path(args.output).expanduser().resolve()
    inputs_dir = output.parent / f".{output.name}.inputs"
    data = prepare_zone_data(zone, config, config_dir, False, inputs_dir)
    target_utc = data.target.index.tz_convert("UTC")
    expected = _expected_index(plans)
    missing_target = expected.difference(target_utc)
    if len(missing_target):
        raise ValueError(
            f"La cible canonique ne couvre pas {len(missing_target)} heure(s) demandée(s)."
        )

    context_length = int(
        args.context_length or deep_get(config, "model.context_length", 2048)
    )
    origin_batch_size = int(
        args.origin_batch_size or deep_get(config, "model.origin_batch_size", 12)
    )
    model_batch_size = int(
        args.model_batch_size or deep_get(config, "model.model_batch_size", 128)
    )
    if min(context_length, origin_batch_size, model_batch_size) < 1:
        raise ValueError("Les tailles de contexte et de batch doivent être positives.")
    first_position = int(target_utc.get_indexer(plans[0].delivery_index_utc)[0])
    if first_position < context_length:
        raise ValueError(
            "Historique cible insuffisant avant le premier jour: "
            f"origin={first_position}, context_length={context_length}."
        )

    audit = _coverage_audit(data, plans)
    audit.update(
        {
            "zone": zone.zone,
            "timezone": zone.timezone,
            "forecast_origin_local_time": str(
                deep_get(config, "data.forecast_origin_local_time", "08:00")
            ),
            "first_origin_utc": str(plans[0].forecast_origin_utc),
            "last_origin_utc": str(plans[-1].forecast_origin_utc),
            "context_length": context_length,
            "mode": args.mode,
        }
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)

    coverage = audit["coverage_by_column"]
    weak = {
        column: value
        for column, value in coverage.items()
        if value < args.minimum_native_coverage
    }
    if args.mode == "native_covariates" and weak:
        details = ", ".join(f"{key}={value:.1%}" for key, value in weak.items())
        raise ValueError(
            "Couverture PIT incompatible avec un Chronos natif comparable au run "
            f"canonique (minimum={args.minimum_native_coverage:.1%}): {details}. "
            "Utilisez une plage couverte, matérialisez les PIT manquants, ou "
            "choisissez explicitement --mode price_only."
        )

    # Resolve defaults in the manifest rather than preserving CLI None values.
    args.context_length = context_length
    args.origin_batch_size = origin_batch_size
    args.model_batch_size = model_batch_size
    if args.plan_only:
        _write_manifest(
            output,
            config_path=config_path,
            args=args,
            audit=audit,
            status="planned",
            elapsed_seconds=0.0,
            completed_days=0,
        )
        print("Plan validé; Chronos n'a pas été chargé.", flush=True)
        return 0

    existing: pd.DataFrame | None = None
    completed_days = 0
    if output.exists():
        if not args.resume:
            raise FileExistsError(
                f"La sortie existe déjà: {output}. Utilisez --resume ou changez --output."
            )
        existing, completed_days = _validate_existing_prefix(output, plans, data.target)
    pending = list(plans[completed_days:])
    if not pending:
        print(f"Artefact déjà complet: {output}", flush=True)
        return 0

    runtime_config = _local_model_config(config) if args.local_files_only else config
    runtime = load_model(runtime_config, args.device, args.local_files_only)
    executor = make_existing_forecasting_executor(
        data=data,
        runtime=runtime,
        context_length=context_length,
        origin_batch_size=origin_batch_size,
        model_batch_size=model_batch_size,
        with_covariates=args.mode == "native_covariates",
        variant=f"hourly_oof_{args.mode}",
    )
    started = time.perf_counter()
    accumulated = existing
    for offset in range(0, len(pending), args.chunk_days):
        chunk = pending[offset : offset + args.chunk_days]
        generated = execute_grouped_chronos_backtest(chunk, executor)
        accumulated = normalize_chronos_oof(
            generated
            if accumulated is None
            else pd.concat([accumulated.reset_index(), generated.reset_index()], ignore_index=True)
        )
        _atomic_write(accumulated, output)
        completed_days += len(chunk)
        elapsed = time.perf_counter() - started
        _write_manifest(
            output,
            config_path=config_path,
            args=args,
            audit=audit,
            status="running" if completed_days < len(plans) else "complete",
            elapsed_seconds=elapsed,
            completed_days=completed_days,
        )
        print(
            f"Checkpoint {completed_days}/{len(plans)} jours | "
            f"{len(accumulated)} heures | {elapsed / 60.0:.1f} min | {output}",
            flush=True,
        )

    if not accumulated.index.equals(expected):
        raise RuntimeError("La sortie finale ne couvre pas exactement la plage demandée.")
    print(f"Artefact Chronos OOF complet: {output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
