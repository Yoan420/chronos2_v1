#!/usr/bin/env python
"""Publish the frozen FR autonomous + MKOnline primary day-ahead blend.

The blend weight and every selection artefact are sealed before this runner is
allowed to read the annual final period.  MKOnline is fetched directly from
the terminal Saturn primary series ``41551_native`` at the civil D-1 08:00
Europe/Paris cutoff.  Missing hours, interpolation, formula wrappers and
fallback forecasts are forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping
import uuid

import numpy as np
import pandas as pd

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.reporting import write_hourly_html_report
from chronos2_modular.common import load_yaml
from evaluate_hourly_backtest import evaluate


LOGGER = logging.getLogger("mkonline_blend_hourly")
TIMEZONE = "Europe/Paris"
PRIMARY_SERIES = "41551_native"
STORM_COMPARATOR_SERIES = "power.price.fr.euromwh.h.fcst.3mv.storm.da.basecase"
STORM_COMPARATOR_PATH = Path("data/pit/vintages/fr_price_storm_da_fcst.parquet")
WEIGHT_MK = 0.5022390717075804
WEIGHT_AUTONOMOUS = 0.4977609282924196
EXPECTED_RECIPE_SHA256 = "9fa7f86a4550b64dd490e48297abd8a7eabd360925f39c75cbdb4654c84f8543"
EXPECTED_DEPENDENCY_SHA256 = "2dae7a579aee1c1f8bb2e96b70e6e0224a7e8b133e5a9e5f836dee5a646f53cc"
EXPECTED_SOURCE_CHECKSUM_MANIFEST_SHA256 = "e08eacdabf7d3c9d03e061706b20839ee75689138c682a22880e388de7b466bc"
FINAL_START = "2025-08-12"
FINAL_END = "2026-08-11"
LIVE_DAY = "2026-08-12"
EXPECTED_FINAL_HOURS = 8760
EXPECTED_FINAL_DAYS = 365
EXPECTED_DAY_HISTOGRAM = {23: 1, 24: 363, 25: 1}
QUANTILES = ("q10", "q50", "q90")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _resolve(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _expected_cutoff(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    values = pd.DatetimeIndex(index)
    if values.tz is None:
        raise ValueError("delivery index must be timezone-aware")
    local_days = pd.DatetimeIndex(values.tz_convert(TIMEZONE).date)
    civil = local_days - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
    return civil.tz_localize(
        TIMEZONE, ambiguous="raise", nonexistent="raise"
    ).tz_convert("UTC")


def _complete_local_range(start_day: str, end_day: str) -> pd.DatetimeIndex:
    pieces = [
        local_delivery_day_index(day, timezone=TIMEZONE)
        for day in pd.date_range(start_day, end_day, freq="D")
    ]
    result = pieces[0].append(pieces[1:])
    result.name = "delivery_start_utc"
    return result


def _day_histogram(index: pd.DatetimeIndex) -> dict[int, int]:
    local_days = pd.Index(index.tz_convert(TIMEZONE).date)
    counts = pd.Series(1, index=local_days).groupby(level=0).sum()
    return {int(hours): int(number) for hours, number in counts.value_counts().items()}


def _blend_quantiles(
    autonomous: pd.DataFrame,
    mkonline_q50: pd.Series,
    *,
    weight: float = WEIGHT_MK,
) -> pd.DataFrame:
    """Pure frozen blend; no external comparator is accepted by this API."""

    if abs(float(weight) - WEIGHT_MK) > 1e-15:
        raise ValueError("MKOnline weight differs from the frozen recipe")
    missing = [name for name in QUANTILES if name not in autonomous]
    if missing:
        raise ValueError(f"autonomous quantiles missing: {missing}")
    auto = autonomous.loc[:, list(QUANTILES)].apply(pd.to_numeric, errors="coerce")
    mk = pd.to_numeric(mkonline_q50, errors="coerce").reindex(auto.index)
    if not np.isfinite(auto.to_numpy(float)).all() or not np.isfinite(mk.to_numpy(float)).all():
        raise ValueError("non-finite autonomous or MKOnline prediction")
    auto_values = auto.to_numpy(float)
    if not (
        (auto_values[:, 0] <= auto_values[:, 1])
        & (auto_values[:, 1] <= auto_values[:, 2])
    ).all():
        raise ValueError("crossed autonomous quantiles")
    blended_q50 = WEIGHT_AUTONOMOUS * auto["q50"] + WEIGHT_MK * mk
    shift = blended_q50 - auto["q50"]
    result = pd.DataFrame(index=auto.index)
    for quantile in QUANTILES:
        result[quantile] = auto[quantile] + shift
    result["shift"] = shift
    result["mkonline_q50"] = mk
    values = result.loc[:, list(QUANTILES)].to_numpy(float)
    if not ((values[:, 0] <= values[:, 1]) & (values[:, 1] <= values[:, 2])).all():
        raise RuntimeError("common shift did not preserve quantile order")
    before_width = auto["q90"].to_numpy(float) - auto["q10"].to_numpy(float)
    after_width = result["q90"].to_numpy(float) - result["q10"].to_numpy(float)
    if not np.allclose(before_width, after_width, rtol=0.0, atol=1e-10):
        raise RuntimeError("common shift changed interval width")
    return result


def _load_primary(
    path: Path,
    *,
    expected_index: pd.DatetimeIndex,
    expected_histogram: Mapping[int, int],
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    frame = pd.read_parquet(path)
    required = {
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
        "value",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing PIT columns {missing}")
    delivery = pd.DatetimeIndex(
        pd.to_datetime(frame["value_time_utc"], utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    if delivery.has_duplicates or not delivery.is_monotonic_increasing:
        raise ValueError(f"{path}: duplicate or unordered delivery timestamps")
    if not delivery.equals(expected_index):
        raise ValueError(f"{path}: exact physical delivery timeline mismatch")
    if _day_histogram(delivery) != dict(expected_histogram):
        raise ValueError(f"{path}: invalid 23/24/25 day histogram")
    cutoff = pd.DatetimeIndex(
        pd.to_datetime(frame["snapshot_time_utc"], utc=True, errors="raise")
    )
    revision = pd.DatetimeIndex(
        pd.to_datetime(frame["revision_time_utc"], utc=True, errors="raise")
    )
    expected_cutoff = _expected_cutoff(delivery)
    if not cutoff.equals(expected_cutoff) or not revision.equals(expected_cutoff):
        raise ValueError(f"{path}: PIT marker differs from civil D-1 08:00")
    values = pd.Series(
        pd.to_numeric(frame["value"], errors="coerce").to_numpy(float),
        index=delivery,
        name="mkonline_primary__q50",
    )
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError(f"{path}: non-finite MKOnline values")
    return values, pd.Series(cutoff, index=delivery, name="mkonline_cutoff_utc"), {
        "series": PRIMARY_SERIES,
        "path": str(path),
        "sha256": _sha256(path),
        "hours": int(len(frame)),
        "local_days": int(len(pd.Index(delivery.tz_convert(TIMEZONE).date).unique())),
        "local_day_hour_histogram": {str(k): v for k, v in _day_histogram(delivery).items()},
        "coverage": 1.0,
        "cutoff": "civil D-1 08:00 Europe/Paris",
        "cutoff_violations": 0,
        "interpolation": False,
    }


def _load_storm_evaluation_only(
    path: Path,
    *,
    expected_index: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    """Select the last strictly eligible Storm vintage for evaluation only.

    This function is intentionally separate from ``_blend_quantiles`` and is
    called only after the candidate prediction has been frozen.
    """

    required = {
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
        "value",
    }
    start = expected_index[0]
    end = expected_index[-1]
    try:
        frame = pd.read_parquet(
            path,
            columns=sorted(required),
            filters=[
                ("value_time_utc", ">=", start.to_pydatetime()),
                ("value_time_utc", "<=", end.to_pydatetime()),
            ],
        )
    except (TypeError, ValueError):
        frame = pd.read_parquet(path, columns=sorted(required))
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing Storm PIT columns {missing}")
    for column in ("value_time_utc", "snapshot_time_utc", "revision_time_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    frame = frame.loc[
        frame["value_time_utc"].between(start, end, inclusive="both")
    ].copy()
    if frame.empty:
        raise ValueError(f"{path}: no Storm vintages on the final period")
    delivery = pd.DatetimeIndex(frame["value_time_utc"])
    cutoff_by_delivery = pd.Series(
        _expected_cutoff(delivery),
        index=frame.index,
    )
    eligible = (
        (frame["snapshot_time_utc"] <= cutoff_by_delivery)
        & (frame["revision_time_utc"] <= cutoff_by_delivery)
    )
    frame = frame.loc[eligible].sort_values(
        ["value_time_utc", "snapshot_time_utc", "revision_time_utc"],
        kind="stable",
    )
    selected = frame.drop_duplicates("value_time_utc", keep="last").copy()
    selected_index = pd.DatetimeIndex(
        selected["value_time_utc"],
        name="delivery_start_utc",
    )
    if not selected_index.equals(expected_index):
        missing_hours = expected_index.difference(selected_index)
        extra_hours = selected_index.difference(expected_index)
        raise ValueError(
            "Storm PIT comparator does not cover the exact final timeline: "
            f"missing={len(missing_hours)}, extra={len(extra_hours)}"
        )
    values = pd.Series(
        pd.to_numeric(selected["value"], errors="coerce").to_numpy(float),
        index=selected_index,
        name="storm_evaluation_only__q50",
    )
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError("Storm PIT comparator contains non-finite values")
    selected_cutoff = _expected_cutoff(selected_index)
    selected_snapshot = pd.DatetimeIndex(selected["snapshot_time_utc"])
    selected_revision = pd.DatetimeIndex(selected["revision_time_utc"])
    if not bool((selected_snapshot <= selected_cutoff).all()):
        raise ValueError("Storm snapshot cutoff violation after selection")
    if not bool((selected_revision <= selected_cutoff).all()):
        raise ValueError("Storm revision cutoff violation after selection")
    selected = selected.reset_index(drop=True)
    return values, selected, {
        "role": "evaluation_only_comparator",
        "series": STORM_COMPARATOR_SERIES,
        "source_path": str(path),
        "source_sha256": _sha256(path),
        "hours": int(len(selected)),
        "local_days": int(
            len(pd.Index(selected_index.tz_convert(TIMEZONE).date).unique())
        ),
        "local_day_hour_histogram": {
            str(k): v for k, v in _day_histogram(selected_index).items()
        },
        "coverage": 1.0,
        "cutoff": "snapshot and revision <= civil D-1 08:00 Europe/Paris",
        "cutoff_violations": 0,
        "interpolation": False,
        "used_for_prediction": False,
        "used_for_live_forecast": False,
    }


def _validate_da_target_availability(
    *,
    training_end_utc: Any,
    forecast_index: pd.DatetimeIndex,
) -> dict[str, str]:
    """Validate the publication contract for day-ahead training labels."""

    training_end = pd.Timestamp(training_end_utc)
    if training_end.tzinfo is None:
        raise ValueError("live-fit training_end_utc must be timezone-aware")
    forecast_days = pd.Index(forecast_index.tz_convert(TIMEZONE).date).unique()
    if len(forecast_days) != 1:
        raise ValueError("live forecast must contain one local delivery day")
    forecast_day = pd.Timestamp(forecast_days[0])
    latest_allowed_delivery_day = forecast_day - pd.Timedelta(days=1)
    training_end_day = pd.Timestamp(training_end.tz_convert(TIMEZONE).date())
    if training_end_day > latest_allowed_delivery_day:
        raise ValueError(
            "live residual corrector uses a day-ahead target not available "
            "before the forecast origin"
        )
    return {
        "contract": (
            "For forecast delivery day F at origin F-1 08:00 Europe/Paris, "
            "the complete day-ahead curve for F-1 was cleared on F-2 and is "
            "available; live-fit delivery days must be <= F-1."
        ),
        "training_end_delivery_day_local": str(training_end_day.date()),
        "forecast_delivery_day_local": str(forecast_day.date()),
        "latest_allowed_training_delivery_day_local": str(
            latest_allowed_delivery_day.date()
        ),
    }


def _load_recipe(
    recipe_path: Path,
    dependency_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    if _sha256(recipe_path) != EXPECTED_RECIPE_SHA256:
        raise ValueError("frozen recipe SHA-256 mismatch")
    if _sha256(dependency_path) != EXPECTED_DEPENDENCY_SHA256:
        raise ValueError("MKOnline dependency manifest SHA-256 mismatch")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    dependency = json.loads(dependency_path.read_text(encoding="utf-8"))
    weights = _mapping(recipe.get("weights"), name="recipe.weights")
    if abs(float(weights.get("mkonline_primary")) - WEIGHT_MK) > 1e-15:
        raise ValueError("recipe MKOnline weight mismatch")
    if abs(float(weights.get("autonomous")) - WEIGHT_AUTONOMOUS) > 1e-15:
        raise ValueError("recipe autonomous weight mismatch")
    external = _mapping(recipe.get("external_expert"), name="recipe.external_expert")
    if external.get("series") != PRIMARY_SERIES or external.get("storm_used_as_feature") is not False:
        raise ValueError("recipe external expert contract mismatch")
    if dependency.get("terminal_series") != PRIMARY_SERIES:
        raise ValueError("dependency does not resolve to 41551_native")
    if dependency.get("terminal_type") != "primary" or dependency.get("terminal_formula") is not None:
        raise ValueError("MKOnline dependency is not a terminal primary series")
    metadata = _mapping(dependency.get("terminal_metadata"), name="dependency.metadata")
    if metadata.get("mercure:provider") != "MKONLINE" or metadata.get("mercure:source") != "WATTSIGHT":
        raise ValueError("unexpected terminal provider/source")
    if dependency.get("storm_token_found") is not False or dependency.get("dependency_gate_passed") is not True:
        raise ValueError("dependency anti-Storm gate failed")
    selection = _mapping(recipe.get("selection_protocol"), name="selection_protocol")
    for artifact_key, hash_key in (
        ("b1_gate_artifact", "b1_gate_artifact_sha256"),
        ("b2_veto_artifact", "b2_veto_artifact_sha256"),
    ):
        artifact = _resolve(str(selection[artifact_key]), base=project_root)
        if _sha256(artifact) != str(selection[hash_key]).lower():
            raise ValueError(f"selection artifact checksum mismatch: {artifact_key}")
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        block = "B1" if artifact_key.startswith("b1") else "B2"
        if _mapping(payload.get(block), name=block).get("passes") is not True:
            raise ValueError(f"selection gate {block} did not pass")
        if _mapping(payload.get("protocol"), name="protocol").get("final_loaded") is not False:
            raise ValueError(f"selection gate {block} loaded final data")
        observed_weight = (
            payload.get("learned_mkonline_weight")
            if block == "B1"
            else payload["protocol"].get("frozen_mkonline_weight")
        )
        if abs(float(observed_weight) - WEIGHT_MK) > 1e-15:
            raise ValueError(f"selection gate {block} weight mismatch")
    return recipe


def _verify_source_run(source_run: Path, recipe: Mapping[str, Any]) -> None:
    checksum_manifest = source_run / "artifact_checksums.json"
    expected = str(recipe["source_autonomous_checksum_manifest_sha256"]).lower()
    if _sha256(checksum_manifest) != expected or expected != EXPECTED_SOURCE_CHECKSUM_MANIFEST_SHA256:
        raise ValueError("autonomous source checksum manifest mismatch")
    payload = json.loads(checksum_manifest.read_text(encoding="utf-8"))
    entries = payload.get("artifacts", [])
    trusted_artifacts = {
        "backtest_hourly_oof.csv.gz": "run_artifact",
        "forecast_hourly_fr.csv": "run_artifact",
        "chronos_live_hourly.csv": "run_artifact",
        "run_manifest.json": "run_artifact",
        "feature_manifest.csv": "run_artifact",
        "inputs/aligned_inputs.csv.gz": "materialized_input",
    }
    for filename, role in trusted_artifacts.items():
        matches = [
            item for item in entries
            if item.get("role") == role and item.get("path") == filename
        ]
        if len(matches) != 1:
            raise ValueError(f"source checksum entry is ambiguous for {filename}")
        if _sha256(source_run / filename) != str(matches[0]["sha256"]).lower():
            raise ValueError(f"source run artifact checksum mismatch: {filename}")
    manifest = json.loads((source_run / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("target_contract") != "hourly_utc_no_interpolation":
        raise ValueError("source target contract is not canonical")
    if manifest.get("delivery_horizon") != "dynamic_23_24_25":
        raise ValueError("source delivery horizon is not DST-safe")
    if manifest.get("uses_legacy_price_forecast") is not False:
        raise ValueError("autonomous source uses a forbidden legacy forecast")
    external = manifest.get("external_price_forecasts_loaded", [])
    if external not in (None, []):
        raise ValueError("autonomous source unexpectedly loads an external price forecast")
    feature_manifest = pd.read_csv(source_run / "feature_manifest.csv")
    feature_text = " ".join(str(value) for value in feature_manifest.to_numpy().ravel())
    if "storm" in feature_text.casefold():
        raise ValueError("forbidden Storm feature in autonomous source manifest")


def _materialize(
    *,
    project_root: Path,
    start_day: str,
    end_day: str,
    output: Path,
    workers: int,
) -> list[str]:
    command = [
        sys.executable,
        str(project_root / "materialize_saturn_daily_asof.py"),
        "--series", PRIMARY_SERIES,
        "--alias", "mkonline_fr_primary",
        "--start-day", start_day,
        "--end-day", end_day,
        "--output", str(output),
        "--workers", str(workers),
        "--hourly-on-the-hour",
    ]
    subprocess.run(command, cwd=project_root, check=True)
    return command


def _metric_rows(backtest: pd.DataFrame, index: pd.DatetimeIndex) -> list[dict[str, Any]]:
    frame = backtest.loc[index]
    actual = pd.to_numeric(frame["actual"], errors="coerce").to_numpy(float)
    rows: list[dict[str, Any]] = []
    for model in (
        "lear", "catboost", "chronos2", "ensemble",
        "residual_corrected", "mkonline_primary", "mkonline_blend",
        "storm_evaluation_only",
    ):
        column = f"{model}__q50"
        if column not in frame:
            continue
        prediction = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        valid = np.isfinite(actual) & np.isfinite(prediction)
        rows.append({
            "model": model,
            "mae": float(np.mean(np.abs(actual[valid] - prediction[valid]))),
            "n_scored": int(valid.sum()),
            "n_expected": int(len(frame)),
            "prediction_coverage": float(np.isfinite(prediction).mean()),
            "score_coverage": float(valid.mean()),
        })
    return rows


def _publish(staging: Path, output: Path, *, overwrite: bool) -> None:
    previous: Path | None = None
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output exists: {output}; pass --overwrite")
        previous = output.with_name(f".{output.name}.previous-{uuid.uuid4().hex}")
        output.replace(previous)
    try:
        staging.replace(output)
    except Exception:
        if previous is not None and previous.exists() and not output.exists():
            previous.replace(output)
        raise
    if previous is not None:
        shutil.rmtree(previous)


def _checksums(
    staging: Path,
    *,
    output: Path,
    config_path: Path,
    recipe_path: Path,
    dependency_path: Path,
    project_root: Path,
) -> None:
    checksum_path = staging / "artifact_checksums.json"
    entries: list[dict[str, Any]] = []
    for role, path in (
        ("source_config", config_path),
        ("frozen_recipe", recipe_path),
        ("dependency_manifest", dependency_path),
    ):
        entries.append({"path": str(path), "role": role, "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path != checksum_path:
            entries.append({
                "path": path.relative_to(staging).as_posix(),
                "role": "run_artifact",
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    for path in (
        project_root / "run_mkonline_blend_hourly.py",
        project_root / "materialize_saturn_daily_asof.py",
        project_root / "chronos2_hourly/reporting.py",
        project_root / "chronos2_modular/report.py",
        project_root / "evaluate_hourly_backtest.py",
    ):
        entries.append({"path": path.relative_to(project_root).as_posix(), "role": "source_code", "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    _write_json(checksum_path, {"algorithm": "sha256", "output_directory": str(output), "artifacts": entries})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen FR autonomous + MKOnline primary blend")
    parser.add_argument("--config", default="chronos2_hourly_fr_mkonline_blend_v1.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    settings = _mapping(config.get("mkonline_blend"), name="mkonline_blend")
    config_dir = config_path.parent
    source_run = _resolve(settings["source_run"], base=config_dir)
    recipe_path = _resolve(settings["recipe_manifest"], base=config_dir)
    dependency_path = _resolve(settings["dependency_manifest"], base=config_dir)
    output_value = args.output_dir or _mapping(config.get("output"), name="output")["directory"]
    output = _resolve(output_value, base=config_dir)
    if output == source_run:
        raise ValueError("output must differ from autonomous source run")
    recipe = _load_recipe(recipe_path, dependency_path, project_root)
    _verify_source_run(source_run, recipe)
    LOGGER.warning("%s", recipe["external_expert"]["commercial_entitlement_status"])

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        shutil.copytree(source_run, staging, dirs_exist_ok=True)
        inputs = staging / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        final_primary = inputs / "mkonline_primary_final.parquet"
        live_primary = inputs / "mkonline_primary_live.parquet"
        commands: list[list[str]] = []
        commands.append(_materialize(project_root=project_root, start_day=FINAL_START, end_day=FINAL_END, output=final_primary, workers=int(settings.get("workers", 8))))
        commands.append(_materialize(project_root=project_root, start_day=LIVE_DAY, end_day=LIVE_DAY, output=live_primary, workers=int(settings.get("workers", 8))))

        final_index = _complete_local_range(FINAL_START, FINAL_END)
        live_index = _complete_local_range(LIVE_DAY, LIVE_DAY)
        if len(final_index) != EXPECTED_FINAL_HOURS or _day_histogram(final_index) != EXPECTED_DAY_HISTOGRAM:
            raise RuntimeError("internal final timeline contract mismatch")
        mk_final, cutoff_final, final_audit = _load_primary(final_primary, expected_index=final_index, expected_histogram=EXPECTED_DAY_HISTOGRAM)
        mk_live, cutoff_live, live_audit = _load_primary(live_primary, expected_index=live_index, expected_histogram={24: 1})

        backtest = pd.read_csv(source_run / "backtest_hourly_oof.csv.gz")
        forbidden_columns = [name for name in backtest if "storm" in str(name).casefold()]
        if forbidden_columns:
            raise ValueError(f"forbidden Storm columns in autonomous backtest: {forbidden_columns}")
        backtest.index = pd.DatetimeIndex(pd.to_datetime(backtest.pop("delivery_start_utc"), utc=True, errors="raise"), name="delivery_start_utc")
        if backtest.index.has_duplicates or not backtest.index.is_monotonic_increasing:
            raise ValueError("autonomous backtest timeline invalid")
        if not final_index.isin(backtest.index).all():
            raise ValueError("autonomous backtest does not cover final index")
        final = backtest.loc[final_index]
        actual = pd.to_numeric(final["actual"], errors="coerce")
        if not np.isfinite(actual.to_numpy(float)).all():
            raise ValueError("canonical final actual is incomplete")
        aligned = pd.read_csv(source_run / "inputs/aligned_inputs.csv.gz")
        aligned.index = pd.DatetimeIndex(
            pd.to_datetime(aligned.pop("timestamp"), utc=True, errors="raise"),
            name="delivery_start_utc",
        )
        canonical_target = pd.to_numeric(aligned["target"], errors="coerce").reindex(final_index)
        if not np.isfinite(canonical_target.to_numpy(float)).all():
            raise ValueError("canonical aligned target is incomplete on final")
        if not np.allclose(
            actual.to_numpy(float), canonical_target.to_numpy(float), rtol=0.0, atol=2e-5
        ):
            raise ValueError("backtest actual differs from canonical aligned target")
        auto_final = final.loc[:, [f"residual_corrected__{q}" for q in QUANTILES]].rename(columns={f"residual_corrected__{q}": q for q in QUANTILES})
        blended_final = _blend_quantiles(auto_final, mk_final)
        # Storm is deliberately loaded only after the candidate forecast is
        # frozen. It is an evaluation-only comparator and cannot affect the
        # blend, its quantiles, its origin or the live forecast.
        storm_source = (project_root / STORM_COMPARATOR_PATH).resolve()
        storm_final, storm_selected, storm_audit = _load_storm_evaluation_only(
            storm_source,
            expected_index=final_index,
        )
        storm_selected.to_parquet(
            inputs / "storm_evaluation_only_final.parquet",
            index=False,
        )
        auto_origin = pd.DatetimeIndex(pd.to_datetime(final["forecast_origin_utc"], utc=True, errors="raise"))
        candidate_origin = pd.DatetimeIndex(np.maximum(auto_origin.asi8, pd.DatetimeIndex(cutoff_final).asi8), tz="UTC")
        if not bool((candidate_origin < final_index).all()):
            raise ValueError("candidate final origin is not causal")
        for q in QUANTILES:
            backtest[f"mkonline_blend__{q}"] = np.nan
            backtest.loc[final_index, f"mkonline_blend__{q}"] = blended_final[q].to_numpy(float)
        backtest["mkonline_primary__q50"] = np.nan
        backtest.loc[final_index, "mkonline_primary__q50"] = mk_final.to_numpy(float)
        backtest["storm_evaluation_only__q50"] = np.nan
        backtest.loc[final_index, "storm_evaluation_only__q50"] = storm_final.to_numpy(float)
        backtest["mkonline_blend_shift"] = np.nan
        backtest.loc[final_index, "mkonline_blend_shift"] = blended_final["shift"].to_numpy(float)
        backtest["mkonline_blend_forecast_origin_utc"] = pd.NaT
        backtest.loc[final_index, "mkonline_blend_forecast_origin_utc"] = candidate_origin.astype(str)
        backtest.reset_index().to_csv(staging / "backtest_hourly_oof.csv.gz", index=False, compression="gzip")

        forecast = pd.read_csv(source_run / "forecast_hourly_fr.csv")
        forbidden_forecast_columns = [name for name in forecast if "storm" in str(name).casefold()]
        if forbidden_forecast_columns:
            raise ValueError(f"forbidden Storm columns in autonomous live forecast: {forbidden_forecast_columns}")
        forecast_index = pd.DatetimeIndex(pd.to_datetime(forecast["delivery_start_utc"], utc=True, errors="raise"), name="delivery_start_utc")
        if not forecast_index.equals(live_index):
            raise ValueError("autonomous live forecast does not match required delivery day")
        forecast.index = forecast_index
        auto_live = forecast.loc[:, [f"residual_corrected__{q}" for q in QUANTILES]].rename(columns={f"residual_corrected__{q}": q for q in QUANTILES})
        blended_live = _blend_quantiles(auto_live, mk_live)
        chronos_live = pd.read_csv(source_run / "chronos_live_hourly.csv")
        chronos_live.index = pd.DatetimeIndex(pd.to_datetime(chronos_live.pop("delivery_start_utc"), utc=True, errors="raise"))
        if not chronos_live.index.equals(live_index):
            raise ValueError("source live origin timeline mismatch")
        auto_live_origin = pd.DatetimeIndex(pd.to_datetime(chronos_live["forecast_origin_utc"], utc=True, errors="raise"))
        candidate_live_origin = pd.DatetimeIndex(np.maximum(auto_live_origin.asi8, pd.DatetimeIndex(cutoff_live).asi8), tz="UTC")
        if not bool((candidate_live_origin < live_index).all()):
            raise ValueError("candidate live origin is not causal")
        for q in QUANTILES:
            forecast[f"mkonline_blend__{q}"] = blended_live[q].to_numpy(float)
            forecast[q] = blended_live[q].to_numpy(float)
        forecast["mkonline_primary__q50"] = mk_live.to_numpy(float)
        forecast["mkonline_blend_shift"] = blended_live["shift"].to_numpy(float)
        forecast["mkonline_blend_forecast_origin_utc"] = candidate_live_origin.astype(str)
        forecast["price_eur_mwh"] = forecast["q50"]
        forecast.reset_index(drop=True).to_csv(staging / "forecast_hourly_fr.csv", index=False)

        metric_rows = _metric_rows(backtest, final_index)
        pd.DataFrame(metric_rows).to_csv(staging / "metrics_hourly.csv", index=False)
        source_metrics = json.loads((source_run / "metrics_hourly.json").read_text(encoding="utf-8"))
        diagnostics = dict(source_metrics.get("training_diagnostics", {}))
        target_availability_audit = _validate_da_target_availability(
            training_end_utc=diagnostics.get("training_end_utc"),
            forecast_index=live_index,
        )
        diagnostics.update({
            "evaluation_start_local_date": FINAL_START,
            "evaluation_end_local_date": FINAL_END,
            "mkonline_blend": {
                "enabled": True,
                "primary_series": PRIMARY_SERIES,
                "mkonline_weight": WEIGHT_MK,
                "autonomous_weight": WEIGHT_AUTONOMOUS,
                "interpolation": False,
                "storm_used_as_feature": False,
                "commercial_entitlement_status": recipe["external_expert"]["commercial_entitlement_status"],
                "target_availability": target_availability_audit,
            },
            "storm_evaluation_only": storm_audit,
        })
        _write_json(staging / "metrics_hourly.json", {
            "metrics": metric_rows,
            "ensemble_weights": source_metrics.get("ensemble_weights", {}),
            "training_diagnostics": diagnostics,
            "forecast_diagnostics": {
                "n_forecast_hours": len(forecast),
                "forecast_start_utc": str(live_index[0]),
                "forecast_end_utc": str(live_index[-1]),
                "model": "mkonline_blend",
            },
        })
        summary, daily, monthly, hourly = evaluate(
            backtest.reset_index(),
            baseline="residual_corrected__q50",
            candidate="mkonline_blend__q50",
            actual="actual",
            timezone=TIMEZONE,
            bootstrap_samples=20_000,
            seed=42,
        )
        _write_json(staging / "evaluation_summary.json", summary)
        daily.to_csv(staging / "evaluation_by_day.csv", index=False)
        monthly.to_csv(staging / "evaluation_by_month.csv", index=False)
        hourly.to_csv(staging / "evaluation_by_hour.csv", index=False)
        storm_summary, storm_daily, storm_monthly, storm_hourly = evaluate(
            backtest.reset_index(),
            baseline="storm_evaluation_only__q50",
            candidate="mkonline_blend__q50",
            actual="actual",
            timezone=TIMEZONE,
            bootstrap_samples=20_000,
            seed=42,
        )
        _write_json(staging / "evaluation_vs_storm_summary.json", storm_summary)
        storm_daily.to_csv(staging / "evaluation_vs_storm_by_day.csv", index=False)
        storm_monthly.to_csv(staging / "evaluation_vs_storm_by_month.csv", index=False)
        storm_hourly.to_csv(staging / "evaluation_vs_storm_by_hour.csv", index=False)

        shutil.copy2(recipe_path, staging / "mkonline_blend_recipe.json")
        shutil.copy2(dependency_path, staging / "mkonline_primary_dependency.json")
        selection = recipe["selection_protocol"]
        shutil.copy2(_resolve(selection["b1_gate_artifact"], base=project_root), staging / "mkonline_primary_b1_gate.json")
        shutil.copy2(_resolve(selection["b2_veto_artifact"], base=project_root), staging / "mkonline_primary_b2_veto.json")
        _write_json(staging / "mkonline_pit_audit.json", {
            "final": final_audit,
            "live": live_audit,
            "materialization_commands": commands,
            "terminal_dependency_gate": True,
            "storm_used_as_feature": False,
        })
        _write_json(staging / "storm_evaluation_only_pit_audit.json", storm_audit)
        source_manifest = json.loads((source_run / "run_manifest.json").read_text(encoding="utf-8"))
        source_manifest.update({
            "script_version": "1.0.0-mkonline-primary-frozen-blend",
            "config": str(config_path),
            "source_run": str(source_run),
            "model_id": "chronos2_extended_residual_plus_mkonline_primary",
            "native_model": "mkonline_blend",
            "baseline_model": "residual_corrected",
            "prediction_inputs": ["autonomous_extended_residual", PRIMARY_SERIES],
            "external_price_forecasts_loaded": [PRIMARY_SERIES],
            "evaluation_only_comparators": [STORM_COMPARATOR_SERIES],
            "storm_evaluation_only_loaded_after_candidate_frozen": True,
            "uses_legacy_price_forecast": False,
            "storm_used_as_feature": False,
            "mkonline_weight": WEIGHT_MK,
            "n_evaluation_hours": EXPECTED_FINAL_HOURS,
            "n_evaluation_days": EXPECTED_FINAL_DAYS,
            "evaluation_start_local_date": FINAL_START,
            "evaluation_end_local_date": FINAL_END,
            "commercial_entitlement_status": recipe["external_expert"]["commercial_entitlement_status"],
            "target_availability": target_availability_audit,
            "sha256_manifest": "artifact_checksums.json",
        })
        _write_json(staging / "run_manifest.json", source_manifest)

        report = _mapping(config.get("report"), name="report")
        report_path = staging / str(report["filename"])
        write_hourly_html_report(
            staging,
            output_path=report_path,
            title=str(report["title"]),
            native_model="mkonline_blend",
            baseline_model="residual_corrected",
            zone="FR",
            timezone=TIMEZONE,
            extreme_threshold=float(report.get("extreme_threshold", 150.0)),
            history_hours=int(report.get("forecast_history_hours", 168)),
        )
        _checksums(staging, output=output, config_path=config_path, recipe_path=recipe_path, dependency_path=dependency_path, project_root=project_root)
        report_name = report_path.name
        candidate_mae = float(summary["candidate_mae"])
        baseline_mae = float(summary["baseline_mae"])
        storm_mae = float(storm_summary["baseline_mae"])
        _publish(staging, output, overwrite=args.overwrite)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"Run horaire : {output}")
    print(f"Forecast horaire : {output / 'forecast_hourly_fr.csv'}")
    print(f"Rapport HTML : {output / report_name}")
    print(f"MAE autonome : {baseline_mae:.6f} EUR/MWh")
    print(f"MAE MKOnline blend : {candidate_mae:.6f} EUR/MWh")
    print(f"MAE Storm (evaluation only) : {storm_mae:.6f} EUR/MWh")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.exception("MKOnline blend runner failed: %s", exc)
        raise SystemExit(1)
