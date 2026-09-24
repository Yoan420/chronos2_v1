#!/usr/bin/env python
"""Train and evaluate an isolated Foundation-MoE autonomous challenger.

The runner consumes only checksum-verified OOF/live artifacts from the frozen
autonomous runs.  It never writes to ``runs/live`` and never changes the
production configuration.  Model/mode selection is performed on validation;
the final test partition remains untouched until that choice is frozen.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd
import yaml

from chronos2_hourly.models.foundation_moe import (
    FoundationMoEConfig,
    FoundationMoEForecast,
    FoundationMoEForecaster,
    PREDICTION_MODES,
    QUANTILES,
)


LOGGER = logging.getLogger("foundation_moe_challenger")
SCRIPT_VERSION = "1.0.2-foundation-moe-challenger"
EXPERT_NAMES: tuple[str, ...] = (
    "autonomous",
    "chronos2",
    "ensemble",
    "catboost",
    "lear",
)
BACKTEST_EXPERT_COLUMNS: dict[str, str] = {
    "autonomous": "residual_corrected__q50",
    "chronos2": "chronos2__q50",
    "ensemble": "ensemble__q50",
    "catboost": "catboost__q50",
    "lear": "lear__q50",
}
LIVE_EXPERT_COLUMNS: dict[str, tuple[str, ...]] = {
    "autonomous": ("residual_corrected__q50",),
    "chronos2": ("chronos2__q50",),
    "ensemble": ("ensemble_uncorrected__q50", "ensemble__q50"),
    "catboost": ("catboost__q50",),
    "lear": ("lear__q50",),
}


@dataclass(frozen=True)
class SourceSpec:
    zone: str
    run_dir: Path
    timezone: str


@dataclass
class Partition:
    experts: pd.DataFrame
    anchor: pd.DataFrame
    target: pd.Series | None
    markets: pd.Series
    horizons: pd.Series
    curves: pd.Series

    def take(self, mask: np.ndarray) -> "Partition":
        positions = np.flatnonzero(mask)
        return Partition(
            experts=self.experts.iloc[positions].copy(),
            anchor=self.anchor.iloc[positions].copy(),
            target=(
                None
                if self.target is None
                else self.target.iloc[positions].copy()
            ),
            markets=self.markets.iloc[positions].copy(),
            horizons=self.horizons.iloc[positions].copy(),
            curves=self.curves.iloc[positions].copy(),
        )


@dataclass(frozen=True)
class LoadedSource:
    spec: SourceSpec
    train: Partition
    validation: Partition
    test: Partition
    live: Partition
    source_contract: Mapping[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Foundation-MoE Top-2 sur les forecasts autonomes OOF scelles; "
            "sorties exclusivement experimentales."
        )
    )
    parser.add_argument(
        "--config",
        default="config/foundation_moe_challenger.yaml",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} doit etre un mapping YAML.")
    return value


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    return _mapping(payload, name=str(path))


def _resolve(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, pd.Timedelta, Path)):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_entry(
    manifest: Mapping[str, Any],
    relative_path: str,
    *,
    role: str | None = None,
) -> Mapping[str, Any]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("artifact_checksums.json: artifacts invalide.")
    normalized = relative_path.replace("\\", "/")
    matches = []
    for item in artifacts:
        if not isinstance(item, Mapping):
            continue
        if role is not None and str(item.get("role", "")) != role:
            continue
        candidate = str(item.get("path", "")).replace("\\", "/")
        if candidate == normalized or candidate.endswith("/" + normalized):
            matches.append(item)
    if len(matches) != 1:
        raise ValueError(
            f"Manifest: attendu exactement une entree pour {relative_path!r}, "
            f"recu={len(matches)}."
        )
    return matches[0]


def _verify_source_artifacts(spec: SourceSpec) -> dict[str, Any]:
    manifest_path = spec.run_dir / "artifact_checksums.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = (
        "backtest_hourly_oof.csv.gz",
        f"forecast_hourly_{spec.zone.lower()}.csv",
        "run_manifest.json",
        "extended_residual_recipe.json",
    )
    artifacts: dict[str, Any] = {}
    for relative in required:
        path = spec.run_dir / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        entry = _manifest_entry(manifest, relative, role="run_artifact")
        observed_size = path.stat().st_size
        observed_sha = _sha256(path)
        if int(entry.get("size_bytes", -1)) != observed_size:
            raise ValueError(f"Taille divergente pour {path}.")
        if str(entry.get("sha256", "")) != observed_sha:
            raise ValueError(f"SHA256 divergent pour {path}.")
        artifacts[relative] = {
            "path": str(path),
            "size_bytes": observed_size,
            "sha256": observed_sha,
        }
    run_manifest = json.loads(
        (spec.run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    if str(run_manifest.get("zone", "")).upper() != spec.zone:
        raise ValueError(f"{spec.run_dir}: zone du run_manifest divergente.")
    return {
        "zone": spec.zone,
        "timezone": spec.timezone,
        "run_dir": str(spec.run_dir),
        "artifact_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "artifacts": artifacts,
    }


def _read_timestamped(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "delivery_start_utc" not in frame:
        raise ValueError(f"{path}: delivery_start_utc absent.")
    index = pd.DatetimeIndex(
        pd.to_datetime(
            frame.pop("delivery_start_utc"),
            utc=True,
            errors="raise",
        ),
        name="delivery_start_utc",
    )
    frame.index = index
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{path}: timestamps dupliques ou non tries.")
    return frame


def _numeric_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    name: str,
) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{name}: colonnes absentes: {missing}.")
    result = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(f"{name}: valeur manquante ou infinie.")
    return result.astype(float)


def _partition(
    frame: pd.DataFrame,
    *,
    zone: str,
    timezone: str,
    include_target: bool,
    live: bool,
) -> Partition:
    if live:
        expert_columns: dict[str, str] = {}
        for name, candidates in LIVE_EXPERT_COLUMNS.items():
            present = [column for column in candidates if column in frame]
            if len(present) != 1:
                raise ValueError(
                    f"{zone} live: colonne de l'expert {name!r} ambigue/absente: "
                    f"{present}."
                )
            expert_columns[name] = present[0]
    else:
        expert_columns = dict(BACKTEST_EXPERT_COLUMNS)
    experts = _numeric_columns(
        frame,
        list(expert_columns.values()),
        name=f"{zone}.experts",
    ).rename(columns={value: key for key, value in expert_columns.items()})
    anchor_columns = [f"residual_corrected__{quantile}" for quantile in QUANTILES]
    anchor = _numeric_columns(
        frame,
        anchor_columns,
        name=f"{zone}.anchor",
    )
    anchor.columns = list(QUANTILES)
    if not np.allclose(
        experts["autonomous"].to_numpy(),
        anchor["q50"].to_numpy(),
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError(f"{zone}: q50 autonome et ancre divergent.")
    if bool(((anchor["q10"] > anchor["q50"]) | (anchor["q50"] > anchor["q90"])).any()):
        raise ValueError(f"{zone}: quantiles autonomes croises.")
    target: pd.Series | None = None
    if include_target:
        target_values = _numeric_columns(frame, ["actual"], name=f"{zone}.target")
        target = target_values["actual"].rename("actual")
    local = frame.index.tz_convert(timezone)
    local_dates = pd.Series(
        [str(value) for value in local.date],
        index=frame.index,
        dtype="string",
    )
    horizon = local_dates.groupby(local_dates, sort=False).cumcount().astype(int)
    day_sizes = horizon.groupby(local_dates, sort=False).transform("size")
    if not day_sizes.isin([23, 24, 25]).all():
        bad = sorted(set(day_sizes.loc[~day_sizes.isin([23, 24, 25])].tolist()))
        raise ValueError(f"{zone}: jours locaux incomplets, tailles={bad}.")
    markets = pd.Series(zone, index=frame.index, dtype="string")
    curves = (markets + ":" + local_dates).astype("string")
    return Partition(
        experts=experts,
        anchor=anchor,
        target=target,
        markets=markets,
        horizons=horizon,
        curves=curves,
    )


def _load_source(
    spec: SourceSpec,
    *,
    train_days: int,
    validation_days: int,
    test_days: int,
) -> LoadedSource:
    contract = _verify_source_artifacts(spec)
    backtest_path = spec.run_dir / "backtest_hourly_oof.csv.gz"
    raw = _read_timestamped(backtest_path)
    required = [
        *BACKTEST_EXPERT_COLUMNS.values(),
        *(f"residual_corrected__{quantile}" for quantile in QUANTILES),
        "actual",
        "forecast_origin_utc",
    ]
    missing = [column for column in required if column not in raw]
    if missing:
        raise ValueError(f"{backtest_path}: colonnes absentes: {missing}.")
    selected_mask = raw[
        [
            *(f"residual_corrected__{quantile}" for quantile in QUANTILES),
            *BACKTEST_EXPERT_COLUMNS.values(),
            "actual",
        ]
    ].notna().all(axis=1)
    selected = raw.loc[selected_mask].copy()
    if selected.empty:
        raise ValueError(f"{backtest_path}: aucun forecast autonome scelle.")
    origins = pd.to_datetime(
        selected["forecast_origin_utc"],
        utc=True,
        errors="raise",
    )
    if not np.asarray(origins < selected.index).all():
        raise ValueError(f"{backtest_path}: origine de forecast non causale.")
    local_dates = pd.Index(selected.index.tz_convert(spec.timezone).date)
    days = local_dates.unique().tolist()
    expected_days = train_days + validation_days + test_days
    if len(days) != expected_days:
        raise ValueError(
            f"{spec.zone}: {len(days)} jours autonomes, attendu {expected_days}."
        )
    continuous = pd.date_range(days[0], days[-1], freq="D").date.tolist()
    if days != continuous:
        raise ValueError(f"{spec.zone}: jours autonomes non continus.")
    train_set = set(days[:train_days])
    validation_set = set(days[train_days : train_days + validation_days])
    test_set = set(days[-test_days:])
    full = _partition(
        selected,
        zone=spec.zone,
        timezone=spec.timezone,
        include_target=True,
        live=False,
    )
    train_mask = np.asarray([value in train_set for value in local_dates])
    validation_mask = np.asarray([value in validation_set for value in local_dates])
    test_mask = np.asarray([value in test_set for value in local_dates])
    if not np.all(train_mask | validation_mask | test_mask):
        raise RuntimeError(f"{spec.zone}: lignes hors split.")
    live_path = spec.run_dir / f"forecast_hourly_{spec.zone.lower()}.csv"
    live_frame = _read_timestamped(live_path)
    live_partition = _partition(
        live_frame,
        zone=spec.zone,
        timezone=spec.timezone,
        include_target=False,
        live=True,
    )
    contract = {
        **contract,
        "sealed_window": {
            "start_local_date": str(days[0]),
            "end_local_date": str(days[-1]),
            "days": len(days),
            "hours": len(selected),
        },
        "splits": {
            "train": [str(days[0]), str(days[train_days - 1])],
            "validation": [
                str(days[train_days]),
                str(days[train_days + validation_days - 1]),
            ],
            "test": [str(days[-test_days]), str(days[-1])],
        },
    }
    return LoadedSource(
        spec=spec,
        train=full.take(train_mask),
        validation=full.take(validation_mask),
        test=full.take(test_mask),
        live=live_partition,
        source_contract=contract,
    )


def _combine(partitions: Sequence[Partition]) -> Partition:
    if not partitions:
        raise ValueError("Aucune partition a combiner.")
    experts = pd.concat([partition.experts for partition in partitions])
    anchor = pd.concat([partition.anchor for partition in partitions])
    markets = pd.concat([partition.markets for partition in partitions])
    horizons = pd.concat([partition.horizons for partition in partitions])
    curves = pd.concat([partition.curves for partition in partitions])
    if all(partition.target is not None for partition in partitions):
        target = pd.concat(
            [partition.target for partition in partitions if partition.target is not None]
        )
    elif all(partition.target is None for partition in partitions):
        target = None
    else:
        raise ValueError("Impossible de melanger des partitions avec/sans cible.")
    order = np.lexsort(
        (
            markets.astype(str).to_numpy(),
            experts.index.asi8,
        )
    )
    return Partition(
        experts=experts.iloc[order].copy(),
        anchor=anchor.iloc[order].copy(),
        target=None if target is None else target.iloc[order].copy(),
        markets=markets.iloc[order].copy(),
        horizons=horizons.iloc[order].copy(),
        curves=curves.iloc[order].copy(),
    )


def _predict_partition(
    model: FoundationMoEForecaster,
    partition: Partition,
    *,
    mode: str,
    require_after_validation: bool,
) -> FoundationMoEForecast:
    return model.predict(
        partition.experts,
        partition.anchor,
        partition.markets,
        partition.horizons,
        partition.curves,
        mode=mode,
        require_after_validation=require_after_validation,
    )


def _metrics(
    partition: Partition,
    forecasts: Mapping[str, pd.DataFrame],
    *,
    tail_quantile: float = 0.95,
) -> pd.DataFrame:
    if partition.target is None:
        raise ValueError("Une cible est requise pour les metriques.")
    if not 0.0 < tail_quantile < 1.0:
        raise ValueError("tail_quantile doit appartenir a ]0, 1[.")
    target = partition.target.to_numpy(dtype=float)
    markets = partition.markets.astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    scopes = ["ALL", *sorted(set(markets.tolist()))]
    for scope in scopes:
        mask = np.ones(len(target), dtype=bool) if scope == "ALL" else markets == scope
        actual = target[mask]
        # The paper's high-price metric is defined within the evaluated
        # segment.  Recompute q95 for each market instead of reusing the pooled
        # threshold, which otherwise changes the nominal tail share by zone.
        tail_threshold = float(np.quantile(actual, tail_quantile))
        tail = actual >= tail_threshold
        for model_name, forecast in forecasts.items():
            prediction = forecast["q50"].to_numpy(dtype=float)[mask]
            error = prediction - actual
            rows.append(
                {
                    "scope": scope,
                    "model": model_name,
                    "n_tokens": int(mask.sum()),
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": float(np.sqrt(np.mean(np.square(error)))),
                    "bias": float(np.mean(error)),
                    "tail_threshold": tail_threshold,
                    "tail_share": float(np.mean(tail)),
                    "top5_mae": (
                        float(np.mean(np.abs(error[tail])))
                        if bool(tail.any())
                        else float("nan")
                    ),
                    "quantile_crossings": int(
                        (
                            (forecast["q10"].to_numpy()[mask] > prediction)
                            | (prediction > forecast["q90"].to_numpy()[mask])
                        ).sum()
                    ),
                }
            )
    result = pd.DataFrame(rows)
    baseline = result.loc[result["model"].eq("autonomous"), ["scope", "mae"]].rename(
        columns={"mae": "autonomous_mae"}
    )
    result = result.merge(baseline, on="scope", how="left", validate="many_to_one")
    result["mae_gain_vs_autonomous"] = result["autonomous_mae"] - result["mae"]
    result["mae_gain_percent"] = (
        100.0 * result["mae_gain_vs_autonomous"] / result["autonomous_mae"]
    )
    return result


def _daily_metrics(
    partition: Partition,
    candidate: pd.DataFrame,
) -> pd.DataFrame:
    if partition.target is None:
        raise ValueError("Une cible est requise.")
    curve_labels = partition.curves.astype(str).to_numpy()
    delivery_dates = pd.Series(curve_labels).str.extract(
        r"(\d{4}-\d{2}-\d{2})$",
        expand=False,
    )
    if delivery_dates.isna().any():
        raise ValueError("curve_id doit se terminer par une date ISO YYYY-MM-DD.")
    frame = pd.DataFrame(
        {
            "market": partition.markets.astype(str).to_numpy(),
            "curve_id": curve_labels,
            "delivery_date": delivery_dates.to_numpy(),
            "actual": partition.target.to_numpy(dtype=float),
            "autonomous": partition.anchor["q50"].to_numpy(dtype=float),
            "candidate": candidate["q50"].to_numpy(dtype=float),
        }
    )
    frame["autonomous_abs_error"] = np.abs(frame["autonomous"] - frame["actual"])
    frame["candidate_abs_error"] = np.abs(frame["candidate"] - frame["actual"])
    daily = frame.groupby(["delivery_date", "market", "curve_id"], sort=True).agg(
        n_tokens=("actual", "size"),
        autonomous_mae=("autonomous_abs_error", "mean"),
        candidate_mae=("candidate_abs_error", "mean"),
    )
    daily["mae_gain"] = daily["autonomous_mae"] - daily["candidate_mae"]
    return daily.reset_index()


def _paired_bootstrap(
    daily: pd.DataFrame,
    *,
    samples: int,
    seed: int,
    block_length_days: int = 7,
) -> dict[str, Any]:
    required = {
        "delivery_date",
        "market",
        "n_tokens",
        "autonomous_mae",
        "candidate_mae",
    }
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"Bootstrap: colonnes absentes: {missing}.")
    if samples < 100:
        raise ValueError("Bootstrap: au moins 100 tirages requis.")
    if block_length_days < 1:
        raise ValueError("block_length_days doit etre positif.")
    work = daily.loc[:, sorted(required)].copy()
    work["autonomous_error_sum"] = work["autonomous_mae"] * work["n_tokens"]
    work["candidate_error_sum"] = work["candidate_mae"] * work["n_tokens"]
    clustered = work.groupby("delivery_date", sort=True).agg(
        n_tokens=("n_tokens", "sum"),
        autonomous_error_sum=("autonomous_error_sum", "sum"),
        candidate_error_sum=("candidate_error_sum", "sum"),
        markets=("market", "nunique"),
    )
    if len(clustered) < 2:
        raise ValueError("Bootstrap: au moins deux dates de livraison requises.")
    errors = (
        clustered["autonomous_error_sum"] - clustered["candidate_error_sum"]
    ).to_numpy(dtype=float)
    tokens = clustered["n_tokens"].to_numpy(dtype=float)
    date_gains = errors / tokens
    n_dates = len(clustered)
    block_length = min(int(block_length_days), n_dates)
    blocks_per_draw = int(np.ceil(n_dates / block_length))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_dates, size=(samples, blocks_per_draw))
    offsets = np.arange(block_length, dtype=int)
    draws = (starts[:, :, None] + offsets[None, None, :]) % n_dates
    draws = draws.reshape(samples, -1)[:, :n_dates]
    means = errors[draws].sum(axis=1) / tokens[draws].sum(axis=1)
    return {
        "unit": "delivery_date_circular_moving_block",
        "n_units": n_dates,
        "block_length_days": block_length,
        "markets_per_date_min": int(clustered["markets"].min()),
        "markets_per_date_max": int(clustered["markets"].max()),
        "samples": samples,
        "seed": seed,
        "observed_mean_gain": float(errors.sum() / tokens.sum()),
        "ci95_lower": float(np.quantile(means, 0.025)),
        "ci95_upper": float(np.quantile(means, 0.975)),
        "probability_gain_positive": float(np.mean(means > 0.0)),
        "positive_day_share": float(np.mean(date_gains > 0.0)),
    }


def _half_gains(partition: Partition, candidate: pd.DataFrame) -> dict[str, float]:
    if partition.target is None:
        raise ValueError("Une cible est requise.")
    curves = partition.curves.astype(str).to_numpy()
    delivery_dates = pd.Series(curves).str.extract(
        r"(\d{4}-\d{2}-\d{2})$",
        expand=False,
    )
    if delivery_dates.isna().any():
        raise ValueError("curve_id doit se terminer par une date ISO YYYY-MM-DD.")
    dates = delivery_dates.to_numpy()
    unique = sorted(pd.Index(dates).unique().tolist())
    midpoint = len(unique) // 2
    halves = {
        "first_half": set(unique[:midpoint]),
        "second_half": set(unique[midpoint:]),
    }
    actual = partition.target.to_numpy(dtype=float)
    baseline = partition.anchor["q50"].to_numpy(dtype=float)
    predicted = candidate["q50"].to_numpy(dtype=float)
    result: dict[str, float] = {}
    for name, selected in halves.items():
        mask = np.asarray([date in selected for date in dates])
        result[name] = float(
            np.mean(np.abs(baseline[mask] - actual[mask]))
            - np.mean(np.abs(predicted[mask] - actual[mask]))
        )
    return result


def _write_checksums(directory: Path) -> Path:
    artifacts = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == "artifact_checksums.json":
            continue
        artifacts.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    output = directory / "artifact_checksums.json"
    _write_json(output, {"algorithm": "sha256", "artifacts": artifacts})
    return output


def _output_directory(
    configured: str | Path,
    *,
    project_root: Path,
) -> Path:
    output = _resolve(configured, base=project_root)
    experiments = (project_root / "runs" / "experiments").resolve()
    if not output.is_relative_to(experiments) or output == experiments:
        raise ValueError("output_dir doit etre un sous-dossier de runs/experiments.")
    if output.exists():
        raise FileExistsError(
            f"Le run experimental existe deja et reste immuable: {output}"
        )
    return output


def run(args: argparse.Namespace) -> Path:
    project_root = Path(__file__).resolve().parent
    config_path = _resolve(args.config, base=project_root)
    config = _load_yaml(config_path)
    experiment = _mapping(config.get("experiment"), name="experiment")
    source_config = _mapping(config.get("sources"), name="sources")
    expert_config = _mapping(config.get("experts"), name="experts")
    model_values = dict(_mapping(config.get("model"), name="model"))
    if args.epochs is not None:
        model_values["epochs"] = int(args.epochs)
        model_values["min_epochs"] = min(
            int(model_values.get("min_epochs", args.epochs)),
            int(args.epochs),
        )
    if args.device is not None:
        model_values["device"] = str(args.device)
    model_config = FoundationMoEConfig(**model_values)
    train_days = int(experiment.get("train_days", 219))
    validation_days = int(experiment.get("validation_days", 73))
    test_days = int(experiment.get("test_days", 73))
    if min(train_days, validation_days, test_days) < 1:
        raise ValueError("Les trois splits doivent contenir au moins un jour.")
    selectable_modes = tuple(experiment.get("selectable_modes", ("paper_balanced",)))
    diagnostic_modes = tuple(experiment.get("diagnostic_modes", ()))
    unknown_modes = sorted(
        (set(selectable_modes) | set(diagnostic_modes)) - set(PREDICTION_MODES)
    )
    if unknown_modes or not selectable_modes:
        raise ValueError(f"Modes invalides ou selection vide: {unknown_modes}.")
    anchor_expert = str(expert_config.get("anchor", "autonomous"))
    foundation_experts = tuple(
        str(value)
        for value in expert_config.get("foundation_experts", EXPERT_NAMES)
    )
    output = _output_directory(
        args.output_dir or str(experiment.get("output_dir")),
        project_root=project_root,
    )
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    sources: list[LoadedSource] = []
    for zone, raw in source_config.items():
        values = _mapping(raw, name=f"sources.{zone}")
        spec = SourceSpec(
            zone=str(zone).upper(),
            run_dir=_resolve(values["run_dir"], base=project_root),
            timezone=str(values["timezone"]),
        )
        LOGGER.info("Verification et chargement %s: %s", spec.zone, spec.run_dir)
        sources.append(
            _load_source(
                spec,
                train_days=train_days,
                validation_days=validation_days,
                test_days=test_days,
            )
        )
    train = _combine([source.train for source in sources])
    validation = _combine([source.validation for source in sources])
    test = _combine([source.test for source in sources])
    if train.target is None or validation.target is None or test.target is None:
        raise RuntimeError("Les partitions historiques doivent contenir actual.")
    LOGGER.info(
        "Fit Foundation-MoE: train=%d, validation=%d, test scelle=%d tokens.",
        len(train.experts),
        len(validation.experts),
        len(test.experts),
    )
    model = FoundationMoEForecaster(model_config).fit(
        train.experts,
        train.target,
        train.anchor,
        train.markets,
        train.horizons,
        train.curves,
        validation_experts=validation.experts,
        validation_target=validation.target,
        validation_anchor_quantiles=validation.anchor,
        validation_markets=validation.markets,
        validation_horizon_tokens=validation.horizons,
        validation_curve_ids=validation.curves,
        anchor_expert=anchor_expert,
        foundation_experts=foundation_experts,
    )
    modes = tuple(dict.fromkeys((*selectable_modes, *diagnostic_modes)))
    validation_forecasts: dict[str, pd.DataFrame] = {
        "autonomous": validation.anchor,
    }
    for mode in modes:
        validation_forecasts[mode] = _predict_partition(
            model,
            validation,
            mode=mode,
            require_after_validation=False,
        ).predictions
    validation_metrics = _metrics(
        validation,
        validation_forecasts,
    )
    pooled_validation = validation_metrics.loc[
        validation_metrics["scope"].eq("ALL")
        & validation_metrics["model"].isin(selectable_modes)
    ].sort_values(["mae", "top5_mae", "model"])
    selected_mode = str(pooled_validation.iloc[0]["model"])
    LOGGER.info("Mode fige sur validation: %s", selected_mode)

    test_outputs: dict[str, FoundationMoEForecast] = {}
    test_forecasts: dict[str, pd.DataFrame] = {"autonomous": test.anchor}
    for mode in modes:
        result = _predict_partition(
            model,
            test,
            mode=mode,
            require_after_validation=True,
        )
        test_outputs[mode] = result
        test_forecasts[mode] = result.predictions
    selected = test_outputs[selected_mode]
    test_metrics = _metrics(test, test_forecasts)
    bootstrap_samples = int(experiment.get("bootstrap_samples", 5000))
    bootstrap_seed = int(experiment.get("seed", model_config.seed))
    bootstrap_block_days = int(experiment.get("bootstrap_block_days", 7))
    daily_by_mode: dict[str, pd.DataFrame] = {}
    bootstrap_by_mode: dict[str, dict[str, Any]] = {}
    half_gains_by_mode: dict[str, dict[str, float]] = {}
    for mode, result in test_outputs.items():
        mode_daily = _daily_metrics(test, result.predictions)
        daily_by_mode[mode] = mode_daily
        bootstrap_by_mode[mode] = _paired_bootstrap(
            mode_daily,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            block_length_days=bootstrap_block_days,
        )
        half_gains_by_mode[mode] = _half_gains(test, result.predictions)
    daily = daily_by_mode[selected_mode]
    bootstrap = bootstrap_by_mode[selected_mode]
    half_gains = half_gains_by_mode[selected_mode]
    selected_pooled = test_metrics.loc[
        test_metrics["scope"].eq("ALL")
        & test_metrics["model"].eq(selected_mode)
    ].iloc[0]
    minimum_gain = float(experiment.get("minimum_mae_gain_eur_mwh", 0.05))
    gates = {
        "minimum_gain_eur_mwh": minimum_gain,
        "passes_minimum_gain": bool(
            selected_pooled["mae_gain_vs_autonomous"] >= minimum_gain
        ),
        "passes_positive_first_half": bool(half_gains["first_half"] > 0.0),
        "passes_positive_second_half": bool(half_gains["second_half"] > 0.0),
        "passes_ci95_positive": bool(bootstrap["ci95_lower"] > 0.0),
    }
    gates["passes_all"] = bool(
        gates["passes_minimum_gain"]
        and gates["passes_positive_first_half"]
        and gates["passes_positive_second_half"]
        and gates["passes_ci95_positive"]
    )

    model.save(staging / "foundation_moe.pt")
    model.history_.to_csv(staging / "training_history.csv", index=False)
    validation_metrics.to_csv(staging / "validation_metrics.csv", index=False)
    test_metrics.to_csv(staging / "test_metrics.csv", index=False)
    daily.to_csv(staging / "test_metrics_by_day.csv", index=False)
    _write_json(staging / "paired_bootstrap.json", bootstrap)
    _write_json(staging / "paired_bootstrap_by_mode.json", bootstrap_by_mode)
    selected.diagnostics.reset_index().to_csv(
        staging / "router_diagnostics.csv.gz",
        index=False,
        compression="gzip",
    )
    test_frame = pd.DataFrame(
        {
            "delivery_start_utc": test.experts.index,
            "market": test.markets.to_numpy(),
            "curve_id": test.curves.to_numpy(),
            "horizon_token": test.horizons.to_numpy(),
            "actual": test.target.to_numpy(),
            "autonomous__q10": test.anchor["q10"].to_numpy(),
            "autonomous__q50": test.anchor["q50"].to_numpy(),
            "autonomous__q90": test.anchor["q90"].to_numpy(),
        }
    )
    for mode, result in test_outputs.items():
        for quantile in QUANTILES:
            test_frame[f"{mode}__{quantile}"] = result.predictions[
                quantile
            ].to_numpy()
    test_frame.to_csv(
        staging / "test_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    live_directory = staging / "live"
    live_directory.mkdir(parents=True)
    live_manifest: dict[str, Any] = {}
    for source in sources:
        live_result = _predict_partition(
            model,
            source.live,
            mode=selected_mode,
            require_after_validation=True,
        )
        frame = pd.DataFrame(
            {
                "delivery_start_utc": source.live.experts.index,
                "market": source.live.markets.to_numpy(),
                "curve_id": source.live.curves.to_numpy(),
                "horizon_token": source.live.horizons.to_numpy(),
                "autonomous__q10": source.live.anchor["q10"].to_numpy(),
                "autonomous__q50": source.live.anchor["q50"].to_numpy(),
                "autonomous__q90": source.live.anchor["q90"].to_numpy(),
                "foundation_moe__q10": live_result.predictions["q10"].to_numpy(),
                "foundation_moe__q50": live_result.predictions["q50"].to_numpy(),
                "foundation_moe__q90": live_result.predictions["q90"].to_numpy(),
                "top1_expert": live_result.diagnostics["top1_expert"].to_numpy(),
                "top1_weight": live_result.diagnostics["top1_weight"].to_numpy(),
                "top2_expert": live_result.diagnostics["top2_expert"].to_numpy(),
                "top2_weight": live_result.diagnostics["top2_weight"].to_numpy(),
            }
        )
        filename = f"forecast_hourly_{source.spec.zone.lower()}.csv"
        frame.to_csv(live_directory / filename, index=False)
        live_manifest[source.spec.zone] = {
            "file": f"live/{filename}",
            "tokens": len(frame),
            "start_utc": str(frame["delivery_start_utc"].iloc[0]),
            "end_utc": str(frame["delivery_start_utc"].iloc[-1]),
        }

    _write_json(
        staging / "source_contract.json",
        {
            "config": {
                "path": str(config_path),
                "sha256": _sha256(config_path),
            },
            "sources": [source.source_contract for source in sources],
            "expert_columns": {
                "backtest": BACKTEST_EXPERT_COLUMNS,
                "live": LIVE_EXPERT_COLUMNS,
            },
        },
    )
    _write_json(staging / "model_diagnostics.json", model.diagnostics())
    _write_json(
        staging / "run_manifest.json",
        {
            "script_version": SCRIPT_VERSION,
            "experiment_name": str(experiment.get("name", output.name)),
            "created_from_config": str(config_path),
            "selected_mode_frozen_on_validation": selected_mode,
            "selectable_modes": list(selectable_modes),
            "diagnostic_modes": list(diagnostic_modes),
            "splits": {
                "train_days_per_zone": train_days,
                "validation_days_per_zone": validation_days,
                "test_days_per_zone": test_days,
                "train_tokens": len(train.experts),
                "validation_tokens": len(validation.experts),
                "test_tokens": len(test.experts),
            },
            "test_result": {
                "autonomous_mae": float(selected_pooled["autonomous_mae"]),
                "candidate_mae": float(selected_pooled["mae"]),
                "mae_gain_eur_mwh": float(
                    selected_pooled["mae_gain_vs_autonomous"]
                ),
                "mae_gain_percent": float(selected_pooled["mae_gain_percent"]),
                "paired_bootstrap": bootstrap,
                "half_gains": half_gains,
                "gates": gates,
            },
            "diagnostic_test_results": {
                mode: {
                    "mae": float(
                        test_metrics.loc[
                            test_metrics["scope"].eq("ALL")
                            & test_metrics["model"].eq(mode),
                            "mae",
                        ].iloc[0]
                    ),
                    "mae_gain_eur_mwh": float(
                        test_metrics.loc[
                            test_metrics["scope"].eq("ALL")
                            & test_metrics["model"].eq(mode),
                            "mae_gain_vs_autonomous",
                        ].iloc[0]
                    ),
                    "paired_bootstrap": bootstrap_by_mode[mode],
                    "half_gains": half_gains_by_mode[mode],
                }
                for mode in modes
            },
            "live_outputs": live_manifest,
            "paper_reproduction": {
                "status": "adapted_not_exact",
                "reason": (
                    "Le preprint ne publie ni code ni plusieurs hyperparametres; "
                    "les experts disponibles remplacent TimesFM/Moirai/Chronos-T5."
                ),
                "anchor": "frozen_current_autonomous_residual_corrected",
                "routing": f"Top-{model_config.top_k}",
                "calibration": "paper_horizon_bias_q60_down_q40_up",
            },
            "production_changed": False,
            "runs_live_written": False,
            "comparator_only": True,
            "automatic_promotion": False,
            "model_config": asdict(model_config),
        },
    )
    checksum = _write_checksums(staging)
    if not checksum.is_file():
        raise RuntimeError("Le manifest de checksums n'a pas ete ecrit.")
    staging.rename(output)
    return output


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    output = run(args)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    result = manifest["test_result"]
    print(f"Run Foundation-MoE : {output}")
    print(f"Mode selectionne : {manifest['selected_mode_frozen_on_validation']}")
    print(
        "MAE test autonome -> challenger : "
        f"{result['autonomous_mae']:.4f} -> {result['candidate_mae']:.4f} EUR/MWh"
    )
    print(
        f"Gain MAE : {result['mae_gain_eur_mwh']:.4f} EUR/MWh "
        f"({result['mae_gain_percent']:.2f}%)"
    )
    print(f"Gates de promotion passees : {result['gates']['passes_all']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.error("Execution interrompue.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.exception("Echec du challenger Foundation-MoE: %s", exc)
        raise SystemExit(1)
