#!/usr/bin/env python
"""Strict multi-zone B1/B2 screen for causal exogenous enrichment.

This is a research-only utility.  It never opens the sealed final block and
never changes a live registry, configuration, recipe, or forecast archive.
The reference for DE/BE/NL/ES is the checksum-pinned autonomous recipe.  B1
fits on A and selects one predeclared feature family; B2 only vetoes the
already-frozen B1 recipe.

Storm is accepted only through ``--storm-evaluation-file``.  That optional
file is loaded after prediction and is absent from feature construction,
model fitting, family selection, and all promotion gates.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parent
PIT_ROOT = ROOT / "data/pit/vintages"

ZONE_SPECS: dict[str, dict[str, str]] = {
    "DE": {
        "timezone": "Europe/Berlin",
        "target_series": "power.price.da.de_lu.bzn.hourly.entsoe.utc.cdh.eurmwh",
    },
    "BE": {
        "timezone": "Europe/Brussels",
        "target_series": "power.price.da.be.bzn.hourly.entsoe.utc.cdh.eurmwh",
    },
    "NL": {
        "timezone": "Europe/Amsterdam",
        "target_series": "power.price.da.nl.bzn.hourly.entsoe.utc.cdh.eurmwh",
    },
    "ES": {
        "timezone": "Europe/Madrid",
        "target_series": "power.price.da.es.bzn.hourly.entsoe.utc.cdh.eurmwh",
    },
}

ALL_PHYSICAL_ZONES = ("FR", "DE", "BE", "NL", "ES")
A_START_DAY = "2024-08-12"
B1_START_DAY = "2025-04-14"
B2_START_DAY = "2025-06-13"
CAL_END_EXCLUSIVE_DAY = "2025-08-12"
A_DAYS = 245
B1_DAYS = 60
B2_DAYS = 60

CROSSFIT_WARMUP_DAYS = 49
CROSSFIT_FOLD_DAYS = 49
CROSSFIT_FOLDS = 4
RAW_CORRECTION_CLIP = 20.0
MODEL_SPEC: dict[str, Any] = {
    "loss": "absolute_error",
    "learning_rate": 0.035,
    "max_iter": 350,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 48,
    "l2_regularization": 100.0,
    "max_bins": 127,
    "early_stopping": False,
    "random_state": 42,
}

FORBIDDEN_FEATURE_RE = re.compile(
    r"(?:^|_)(?:storm|target|label|actual|price|spot|epex|mkonline|chronos|"
    r"forecast_error|residual_error)(?:_|$)",
    re.I,
)
WEATHER_RE = re.compile(
    r"(?:^|_)(?:temp|temperature|wind_speed|windspeed|shortwave|radiation|"
    r"cloud|cloudcover|precip|precipitation)(?:_|$)",
    re.I,
)
FUNDAMENTAL_RE = re.compile(
    r"(?:^|_)(?:load|wind|solar|hydro|nuclear|availability|outage|"
    r"capacity|flow|schedule|gas|lignite|coal)(?:_|$)",
    re.I,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: Mapping[str, Any], destination: Path | None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(rendered)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(destination)
    print(rendered)


def _day_boundary(day: str, timezone: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz=timezone).tz_convert("UTC")


def _expected_index(
    start_day: str, end_exclusive_day: str, timezone: str
) -> pd.DatetimeIndex:
    return pd.date_range(
        _day_boundary(start_day, timezone),
        _day_boundary(end_exclusive_day, timezone),
        freq="h",
        inclusive="left",
        name="delivery_start_utc",
    )


def _default_run_dir(zone: str) -> Path:
    return ROOT / f"runs/chronos2_hourly_{zone.lower()}_residual_extended_v1"


def _default_recipe_path(zone: str) -> Path:
    return ROOT / f"chronos2_hourly_{zone.lower()}_autonomous_recipe_v1.json"


def _artifact_entry(
    checksum_manifest: Mapping[str, Any], relative_path: str, *, role: str
) -> Mapping[str, Any]:
    matches = [
        item
        for item in checksum_manifest.get("artifacts", [])
        if item.get("role") == role
        and Path(str(item.get("path", ""))).as_posix() == relative_path
    ]
    if len(matches) != 1:
        raise ValueError(
            f"checksum manifest must pin exactly one {role} {relative_path!r}"
        )
    return matches[0]


def _verify_frozen_reference(
    zone: str, run_dir: Path, recipe_path: Path
) -> dict[str, Any]:
    zone = zone.upper()
    spec = ZONE_SPECS[zone]
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if recipe.get("status") != "sealed_before_final_opening":
        raise ValueError("reference recipe is not sealed_before_final_opening")
    if recipe.get("zone") != zone or recipe.get("timezone") != spec["timezone"]:
        raise ValueError("reference recipe zone/timezone identity mismatch")
    if recipe.get("recipe_mode") != "autonomous_only":
        raise ValueError("multi-zone exogenous screen requires autonomous_only reference")
    if recipe.get("mkonline_enabled") is not False:
        raise ValueError("reference recipe enables a price expert")
    if recipe.get("final_target_used_for_weight_or_hyperparameters") is not False:
        raise ValueError("reference recipe used final targets")
    source_run = (ROOT / str(recipe.get("source_autonomous_run", ""))).resolve()
    if source_run != run_dir.resolve():
        raise ValueError("reference recipe points to another autonomous run")

    checksum_path = run_dir / "artifact_checksums.json"
    if _sha256(checksum_path) != recipe.get(
        "source_autonomous_checksum_manifest_sha256"
    ):
        raise ValueError("autonomous checksum manifest differs from sealed recipe")
    checksum_manifest = json.loads(checksum_path.read_text(encoding="utf-8"))
    required = {
        "backtest_hourly_oof.csv.gz": ("run_artifact", run_dir),
        "extended_residual_recipe.json": ("run_artifact", run_dir),
        "feature_manifest.csv": ("run_artifact", run_dir),
        "run_manifest.json": ("run_artifact", run_dir),
        "inputs/input_manifest.csv": ("materialized_input", run_dir),
        "inputs/aligned_inputs.csv.gz": ("materialized_input", run_dir),
        "inputs/model_covariates_with_future.csv.gz": (
            "materialized_input",
            run_dir,
        ),
        "chronos2_hourly/features.py": ("source_code", ROOT),
        "chronos2_hourly/models/residual_corrector.py": ("source_code", ROOT),
    }
    verified: dict[str, str] = {}
    for relative, (role, base_directory) in required.items():
        entry = _artifact_entry(checksum_manifest, relative, role=role)
        path = base_directory / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _sha256(path)
        if digest != entry.get("sha256"):
            raise ValueError(f"frozen run artifact checksum mismatch: {relative}")
        verified[relative] = digest

    run_manifest = json.loads((run_dir / "run_manifest.json").read_text("utf-8"))
    if (
        run_manifest.get("zone") != zone
        or run_manifest.get("timezone") != spec["timezone"]
    ):
        raise ValueError("frozen run manifest zone/timezone identity mismatch")
    features = pd.read_csv(run_dir / "feature_manifest.csv")
    feature_names = features.get("feature", pd.Series(dtype=str)).astype(str).tolist()
    if any("storm" in value.casefold() for value in feature_names):
        raise ValueError("Storm appears in the autonomous feature manifest")
    input_manifest = pd.read_csv(run_dir / "inputs/input_manifest.csv")
    target_rows = input_manifest.loc[input_manifest["role"].eq("target")]
    if len(target_rows) != 1 or target_rows.iloc[0]["series"] != spec["target_series"]:
        raise ValueError("frozen target series identity mismatch")
    source_recipe = json.loads(
        (run_dir / "extended_residual_recipe.json").read_text(encoding="utf-8")
    )
    if source_recipe.get("external_price_forecasts_loaded") != []:
        raise ValueError("frozen autonomous recipe contains an external price forecast")
    if source_recipe.get("name") != "blend_cat_hgb_w0.50":
        raise ValueError("unsupported frozen autonomous recipe name")
    if source_recipe.get("schema") != "chronos_only":
        raise ValueError("unsupported frozen autonomous recipe schema")
    if source_recipe.get("weights") != {"cat_v1": 0.5, "hgb31": 0.5}:
        raise ValueError("unsupported frozen autonomous recipe weights")
    if float(source_recipe.get("final_clip_eur_mwh", np.nan)) != 40.0:
        raise ValueError("unsupported frozen autonomous correction clip")
    if _sha256(run_dir / "extended_residual_recipe.json") != recipe.get(
        "source_autonomous_recipe_sha256"
    ):
        raise ValueError("source recipe hash differs from sealed recipe")
    return {
        "zone": zone,
        "timezone": spec["timezone"],
        "target_series": spec["target_series"],
        "recipe_path": str(recipe_path.resolve()),
        "recipe_sha256": _sha256(recipe_path),
        "run_dir": str(run_dir.resolve()),
        "checksum_manifest_sha256": _sha256(checksum_path),
        "verified_run_artifacts": verified,
        "reference_prediction": "reconstructed frozen residual_corrected__q50",
        "storm_used_as_feature": False,
    }


def _read_reference_blocks(
    run_dir: Path,
    *,
    timezone: str,
    blocks: Iterable[str],
) -> dict[str, pd.DataFrame]:
    """Stream calibration rows only; never parse a final target/prediction."""

    requested = set(blocks)
    if not requested.issubset({"A", "B1", "B2"}) or not requested:
        raise ValueError("reference blocks must be a non-empty A/B1/B2 subset")
    ranges = {
        "A": (_day_boundary(A_START_DAY, timezone), _day_boundary(B1_START_DAY, timezone)),
        "B1": (_day_boundary(B1_START_DAY, timezone), _day_boundary(B2_START_DAY, timezone)),
        "B2": (
            _day_boundary(B2_START_DAY, timezone),
            _day_boundary(CAL_END_EXCLUSIVE_DAY, timezone),
        ),
    }
    calibration_end = ranges["B2"][1]
    path = run_dir / "backtest_hourly_oof.csv.gz"
    rows: dict[str, list[tuple[pd.Timestamp, float, float]]] = {
        name: [] for name in requested
    }
    with gzip.open(path, mode="rt", encoding="utf-8", newline="") as handle:
        header = next(csv.reader([handle.readline()]))
        required = ("delivery_start_utc", "actual", "residual_corrected__q50")
        missing = [column for column in required if column not in header]
        if missing:
            raise ValueError(f"reference OOF is missing columns: {missing}")
        positions = {column: header.index(column) for column in required}
        if positions["delivery_start_utc"] != 0:
            raise ValueError("delivery_start_utc must be the first OOF column")
        for raw_line in handle:
            # Inspect only the first CSV field before deciding whether the row
            # is admissible.  Values from the sealed final are never parsed.
            timestamp_text = raw_line.split(",", 1)[0]
            timestamp = pd.Timestamp(timestamp_text)
            if timestamp.tzinfo is None:
                raise ValueError("reference OOF timestamp is timezone-naive")
            timestamp = timestamp.tz_convert("UTC")
            if timestamp >= calibration_end:
                break
            block = next(
                (
                    name
                    for name, (start, end) in ranges.items()
                    if start <= timestamp < end
                ),
                None,
            )
            if block not in requested:
                continue
            values = next(csv.reader([raw_line]))
            actual = float(values[positions["actual"]])
            reference = float(values[positions["residual_corrected__q50"]])
            if not np.isfinite(actual) or not np.isfinite(reference):
                raise ValueError(f"non-finite frozen reference row in {block}")
            rows[block].append((timestamp, actual, reference))

    result: dict[str, pd.DataFrame] = {}
    expected_bounds = {
        "A": (A_START_DAY, B1_START_DAY),
        "B1": (B1_START_DAY, B2_START_DAY),
        "B2": (B2_START_DAY, CAL_END_EXCLUSIVE_DAY),
    }
    for block in requested:
        expected = _expected_index(*expected_bounds[block], timezone)
        frame = pd.DataFrame(
            rows[block], columns=("delivery_start_utc", "actual", "reference")
        ).set_index("delivery_start_utc")
        frame.index = pd.DatetimeIndex(frame.index, name="delivery_start_utc")
        if not frame.index.equals(expected):
            raise ValueError(
                f"frozen reference {block} physical timeline mismatch: "
                f"expected={len(expected)}, actual={len(frame)}"
            )
        result[block] = frame
    return result


def _rebuild_frozen_reference_blocks(
    zone: str,
    run_dir: Path,
    *,
    timezone: str,
    phase: str,
    threads: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Rebuild the sealed autonomous recipe without opening the final block.

    ``residual_corrected__q50`` is intentionally null on calibration rows in
    the published run: the stored correction is the one evaluated on final.
    A valid calibration comparator must therefore be reconstructed with the
    frozen recipe.  A is predicted by five expanding 49-day cross-fit folds;
    B1/B2 is predicted by the recipe fitted on EXT+A only.
    """

    if phase not in {"b1", "b2"}:
        raise ValueError("reference reconstruction exposes B1/B2 only")
    import screen_mkonline_zone_crossfit as crossfit

    a_start_day = pd.Timestamp(A_START_DAY)
    contract = crossfit.CalibrationContract(
        zone=zone,
        timezone=timezone,
        a_start_day=a_start_day,
        a_start_utc=_day_boundary(A_START_DAY, timezone),
        a_end_utc=_day_boundary(B1_START_DAY, timezone),
        b1_end_utc=_day_boundary(B2_START_DAY, timezone),
        b2_end_utc=_day_boundary(CAL_END_EXCLUSIVE_DAY, timezone),
    )
    end = contract.b1_end_utc if phase == "b1" else contract.b2_end_utc
    features = crossfit._load_features_to(  # noqa: SLF001 - shared research contract
        run_dir, end_exclusive_utc=end, timezone=timezone
    )
    extended_path = (
        ROOT
        / "runs/chronos_oof_extended"
        / zone.lower()
        / "chronos_oof_native_20240102_20240811.csv.gz"
    )
    extended = crossfit._load_external(  # noqa: SLF001
        extended_path, features, contract=contract
    )
    current = crossfit._load_existing(  # noqa: SLF001
        run_dir, features, contract=contract, phase=phase
    )
    local_days = current["local_days"]
    days = current["days"]
    mask_a = np.asarray(local_days.isin(days[:A_DAYS]), dtype=bool)
    a_index = current["meta"].index[mask_a]
    actual_a = pd.to_numeric(
        current["raw"].loc[a_index, "actual"], errors="raise"
    ).astype(float)
    reference_a = np.full(len(a_index), np.nan, dtype=float)
    fold_audits: list[dict[str, Any]] = []
    for fold in range(crossfit.CROSSFIT_FOLDS):
        start_day = fold * crossfit.CROSSFIT_DAYS
        end_day = start_day + crossfit.CROSSFIT_DAYS
        validation_days = days[start_day:end_day]
        validation_mask = np.asarray(local_days.isin(validation_days), dtype=bool)
        prior_mask = np.asarray(local_days.isin(days[:start_day]), dtype=bool)
        fit_x, fit_y = crossfit._fit_frame(extended, current, prior_mask)  # noqa: SLF001
        models = crossfit._fit_models(fit_x, fit_y, threads=threads)  # noqa: SLF001
        validation_index = current["meta"].index[validation_mask]
        prediction = crossfit._autonomous_prediction(  # noqa: SLF001
            models, current["meta"].loc[validation_index]
        )
        positions = a_index.get_indexer(validation_index)
        if bool((positions < 0).any()):
            raise RuntimeError("reference cross-fit validation lies outside A")
        reference_a[positions] = prediction
        fold_audits.append(
            {
                "fold": fold + 1,
                "fit_EXT_days": crossfit.EXT_DAYS,
                "fit_prior_A_days": start_day,
                "validation_A_days": crossfit.CROSSFIT_DAYS,
                "validation_start": str(validation_days[0]),
                "validation_end": str(validation_days[-1]),
            }
        )
    if not bool(np.isfinite(reference_a).all()):
        raise RuntimeError("reference cross-fit did not cover every A hour")

    score_block = "B1" if phase == "b1" else "B2"
    score_days = (
        days[A_DAYS : A_DAYS + B1_DAYS]
        if phase == "b1"
        else days[A_DAYS + B1_DAYS : A_DAYS + B1_DAYS + B2_DAYS]
    )
    score_mask = np.asarray(local_days.isin(score_days), dtype=bool)
    score_index = current["meta"].index[score_mask]
    fit_x, fit_y = crossfit._fit_frame(extended, current, mask_a)  # noqa: SLF001
    models = crossfit._fit_models(fit_x, fit_y, threads=threads)  # noqa: SLF001
    score_reference = crossfit._autonomous_prediction(  # noqa: SLF001
        models, current["meta"].loc[score_index]
    )
    score_actual = pd.to_numeric(
        current["raw"].loc[score_index, "actual"], errors="raise"
    ).astype(float)
    blocks = {
        "A": pd.DataFrame(
            {"actual": actual_a.to_numpy(float), "reference": reference_a},
            index=a_index,
        ),
        score_block: pd.DataFrame(
            {
                "actual": score_actual.to_numpy(float),
                "reference": score_reference,
            },
            index=score_index,
        ),
    }
    expected = {
        "A": _expected_index(A_START_DAY, B1_START_DAY, timezone),
        score_block: _expected_index(
            B1_START_DAY if phase == "b1" else B2_START_DAY,
            B2_START_DAY if phase == "b1" else CAL_END_EXCLUSIVE_DAY,
            timezone,
        ),
    }
    for name, frame in blocks.items():
        frame.index = pd.DatetimeIndex(frame.index, name="delivery_start_utc")
        if not frame.index.equals(expected[name]):
            raise ValueError(f"reconstructed reference {name} timeline mismatch")
    return blocks, {
        "method": "frozen_EXT_plus_prior_A_crossfit_then_EXT_plus_A",
        "reference_recipe": "blend_cat_hgb_w0.50",
        "reference_clip_eur_mwh": 40.0,
        "extended_oof_path": str(extended_path.resolve()),
        "extended_oof_sha256": _sha256(extended_path),
        "implementation_path": str((ROOT / "screen_mkonline_zone_crossfit.py").resolve()),
        "implementation_sha256": _sha256(ROOT / "screen_mkonline_zone_crossfit.py"),
        "crossfit_folds": fold_audits,
        "B1_rows_loaded": B1_DAYS * 24 if phase == "b1" else B1_DAYS * 24,
        "B1_rows_used_for_fit": 0,
        "B2_rows_loaded": B2_DAYS * 24 if phase == "b2" else 0,
        "final_rows_loaded": 0,
    }


def _decision_cutoffs(index: pd.DatetimeIndex, timezone: str) -> pd.DatetimeIndex:
    local_dates = index.tz_convert(timezone).date
    cache: dict[object, pd.Timestamp] = {}
    cutoffs: list[pd.Timestamp] = []
    for delivery_date in local_dates:
        if delivery_date not in cache:
            day = pd.Timestamp(delivery_date)
            cutoff_local = (day - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize(
                timezone, ambiguous="raise", nonexistent="raise"
            )
            cache[delivery_date] = cutoff_local.tz_convert("UTC")
        cutoffs.append(cache[delivery_date])
    return pd.DatetimeIndex(cutoffs)


def _parquet_filters_for_index(
    timestamp_column: str, expected: pd.DatetimeIndex
) -> list[list[tuple[str, str, object]]]:
    """Return OR-ed UTC ranges for a possibly non-contiguous hourly index."""

    if expected.empty or expected.has_duplicates or not expected.is_monotonic_increasing:
        raise ValueError("expected parquet index must be non-empty, unique and sorted")
    breaks = np.flatnonzero(
        np.diff(expected.asi8) != pd.Timedelta(hours=1).value
    )
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(expected) - 1]
    return [
        [
            (timestamp_column, ">=", expected[int(start)].to_pydatetime()),
            (timestamp_column, "<=", expected[int(end)].to_pydatetime()),
        ]
        for start, end in zip(starts, ends)
    ]


def _read_pit_series(
    path: Path,
    expected: pd.DatetimeIndex,
    *,
    timezone: str,
    alias: str,
) -> tuple[pd.Series, dict[str, Any]]:
    import pyarrow.parquet as pq

    required = {
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
        "value",
    }
    schema = set(pq.read_schema(path).names)
    if not required.issubset(schema):
        raise ValueError(f"{path.name} lacks strict PIT columns")
    filters = _parquet_filters_for_index("value_time_utc", expected)
    frame = pd.read_parquet(path, columns=sorted(required), filters=filters)
    index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("value_time_utc"), utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(f"{path.name} timeline is duplicated or unsorted")
    extra = index.difference(expected)
    if len(extra):
        raise ValueError(f"{path.name} returned out-of-window rows")
    snapshot = pd.DatetimeIndex(
        pd.to_datetime(frame["snapshot_time_utc"], utc=True, errors="raise")
    )
    revision = pd.DatetimeIndex(
        pd.to_datetime(frame["revision_time_utc"], utc=True, errors="raise")
    )
    cutoffs = _decision_cutoffs(index, timezone)
    if bool((snapshot > cutoffs).any()) or bool((revision > cutoffs).any()):
        raise ValueError(f"{path.name} contains a post-cutoff observation")
    if bool((revision > snapshot).any()):
        raise ValueError(f"{path.name} revision occurs after its snapshot")
    values = pd.to_numeric(frame["value"], errors="raise").astype(float)
    if not bool(np.isfinite(values.to_numpy()).all()):
        raise ValueError(f"{path.name} contains non-finite values")
    values.index = index
    coverage = float(index.isin(expected).sum() / len(expected))
    if coverage < 0.90:
        raise ValueError(f"{path.name} coverage is below the frozen 90% floor")
    values = values.reindex(expected)
    values.name = alias
    return values, {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "rows_loaded": int(values.notna().sum()),
        "expected_rows": len(expected),
        "missing_rows": int(values.isna().sum()),
        "coverage": coverage,
        "cutoff_violations": 0,
        "first_delivery_utc": str(index[0]),
        "last_delivery_utc": str(index[-1]),
    }


def _revision_families(
    expected: pd.DatetimeIndex, *, zone: str, timezone: str
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    aliases: dict[str, Path] = {}
    for physical_zone in ALL_PHYSICAL_ZONES:
        slug = physical_zone.lower()
        for signal in ("abs_delta", "std", "age_hours"):
            alias = f"{slug}_residual_revision_{signal}"
            aliases[alias] = PIT_ROOT / f"{slug}_residual_load_revision_{signal}.parquet"
    missing = [str(path) for path in aliases.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing revision PIT inputs: {missing}")
    series: dict[str, pd.Series] = {}
    audits: dict[str, Any] = {}
    for alias, path in aliases.items():
        values, audit = _read_pit_series(
            path, expected, timezone=timezone, alias=alias
        )
        series[alias] = values
        audits[alias] = audit
    panel = pd.DataFrame(series, index=expected)
    local_columns = [
        column for column in panel if column.startswith(f"{zone.lower()}_")
    ]
    families = {
        "local_revision": panel.loc[:, local_columns].copy(),
        "continental_revision": panel.copy(),
    }
    return families, {
        "classification": "strict_daily_asof_d_minus_1_08_local",
        "storm_used": False,
        "target_or_price_used": False,
        "files": audits,
    }


def _read_extra_exogenous(
    path: Path,
    manifest_path: Path,
    expected: pd.DatetimeIndex,
    *,
    zone: str,
    timezone: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import pyarrow.parquet as pq

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("output_sha256") or manifest.get("output", {}).get(
        "sha256"
    )
    actual_hash = _sha256(path)
    if expected_hash != actual_hash:
        raise ValueError("extra exogenous parquet hash differs from its manifest")
    causality = manifest.get("causality", {})
    if causality.get("strict_pit_eligible") is not True:
        raise ValueError("extra exogenous source is not strict PIT eligible")
    if causality.get("storm_used") is not False:
        raise ValueError("extra exogenous source lacks a negative Storm proof")
    if causality.get("target_or_price_used") is not False:
        raise ValueError("extra exogenous source uses a target or price")
    if int(causality.get("cutoff_violations", -1)) != 0:
        raise ValueError("extra exogenous source has cutoff violations")
    declared_zone = manifest.get("zone")
    declared_timezone = manifest.get("causality_contract", {}).get(
        "delivery_timezone", manifest.get("timezone")
    )
    if declared_zone is not None and str(declared_zone).upper() != zone:
        raise ValueError("extra exogenous manifest has another zone")
    if declared_timezone != timezone:
        raise ValueError("extra exogenous manifest timezone does not match the zone")

    schema = set(pq.read_schema(path).names)
    timestamp_column = (
        "delivery_start_utc"
        if "delivery_start_utc" in schema
        else "value_time_utc"
        if "value_time_utc" in schema
        else None
    )
    if timestamp_column is None:
        raise ValueError("extra exogenous parquet has no UTC delivery column")
    feature_contract = manifest.get("feature_contract", {})
    feature_kind = feature_contract.get("kind")
    if feature_kind == "fundamental_daily_local_midnight_broadcast":
        required = {
            timestamp_column,
            "snapshot_time_utc",
            "revision_time_utc",
            str(feature_contract.get("source_value_column", "value")),
        }
        missing = sorted(required.difference(schema))
        if missing:
            raise ValueError(
                f"daily fundamental parquet misses required columns: {missing}"
            )
        source_value_column = str(
            feature_contract.get("source_value_column", "value")
        )
        feature_alias = str(feature_contract.get("feature_alias", ""))
        if not feature_alias or not FUNDAMENTAL_RE.search(feature_alias):
            raise ValueError("daily fundamental alias is not predeclared/recognised")
        if FORBIDDEN_FEATURE_RE.search(feature_alias):
            raise ValueError("daily fundamental alias is forbidden")
        frame = pd.read_parquet(
            path,
            columns=[
                timestamp_column,
                "snapshot_time_utc",
                "revision_time_utc",
                source_value_column,
            ],
            filters=_parquet_filters_for_index(timestamp_column, expected),
        )
        index = pd.DatetimeIndex(
            pd.to_datetime(frame[timestamp_column], utc=True, errors="raise"),
            name="delivery_start_utc",
        )
        local_index = index.tz_convert(timezone)
        source_days = local_index.normalize()
        if index.has_duplicates or not local_index.equals(source_days):
            raise ValueError(
                "daily fundamental must contain one local-midnight row per day"
            )
        expected_days = pd.DatetimeIndex(
            expected.tz_convert(timezone).normalize().unique()
        )
        if not source_days.equals(expected_days):
            raise ValueError(
                "daily fundamental does not cover every requested local day exactly"
            )

        cutoff_timezone = manifest.get("causality_contract", {}).get(
            "cutoff_timezone"
        )
        cutoff_local_time = manifest.get("causality_contract", {}).get(
            "cutoff_local_time"
        )
        if cutoff_timezone != "Europe/Paris" or cutoff_local_time != "08:00":
            raise ValueError(
                "daily fundamental must use D-1 08:00 Europe/Paris"
            )
        clock = pd.Timedelta("08:00:00")
        expected_cutoffs = pd.DatetimeIndex(
            [
                (day.tz_localize(None) - pd.Timedelta(days=1) + clock)
                .tz_localize(cutoff_timezone)
                .tz_convert("UTC")
                for day in source_days
            ]
        )
        snapshots = pd.DatetimeIndex(
            pd.to_datetime(frame["snapshot_time_utc"], utc=True, errors="raise")
        )
        revisions = pd.DatetimeIndex(
            pd.to_datetime(frame["revision_time_utc"], utc=True, errors="raise")
        )
        if not snapshots.equals(expected_cutoffs):
            raise ValueError("daily fundamental snapshot differs from its civil cutoff")
        if not revisions.equals(expected_cutoffs):
            raise ValueError("daily fundamental revision marker differs from cutoff")

        values = pd.to_numeric(
            frame[source_value_column], errors="coerce"
        ).astype(float)
        if not bool(np.isfinite(values.to_numpy()).all()):
            raise ValueError("daily fundamental contains missing/non-finite values")
        by_day = pd.Series(values.to_numpy(), index=source_days)
        hourly_days = expected.tz_convert(timezone).normalize()
        broadcast = pd.DataFrame(
            {feature_alias: by_day.reindex(hourly_days).to_numpy(dtype=float)},
            index=expected,
        )
        if len(broadcast) != len(expected) or broadcast.isna().any().any():
            raise ValueError("daily fundamental broadcast is incomplete")
        return broadcast, {
            "path": str(path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "sha256": actual_hash,
            "classification": causality.get("classification"),
            "strict_pit_eligible": True,
            "storm_used": False,
            "target_or_price_used": False,
            "feature_kind": "fundamental",
            "feature_alias": feature_alias,
            "source_rows_loaded": len(frame),
            "rows_loaded": len(broadcast),
            "broadcast_rule": feature_contract.get("broadcast_rule"),
        }
    frame = pd.read_parquet(
        path,
        filters=_parquet_filters_for_index(timestamp_column, expected),
    )
    index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop(timestamp_column), utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    if index.has_duplicates or not index.equals(expected):
        raise ValueError("extra exogenous source has an inexact physical timeline")
    frame.index = index
    audit_time_columns = [
        column
        for column in (
            "weather_fixed_lead_reference_time_utc",
            "weather_cutoff_time_utc",
            "snapshot_time_utc",
            "revision_time_utc",
        )
        if column in frame
    ]
    if {
        "weather_fixed_lead_reference_time_utc",
        "weather_cutoff_time_utc",
    }.issubset(frame.columns):
        reference = pd.to_datetime(
            frame["weather_fixed_lead_reference_time_utc"], utc=True, errors="raise"
        )
        cutoff = pd.to_datetime(
            frame["weather_cutoff_time_utc"], utc=True, errors="raise"
        )
        if not bool((reference < cutoff).all()):
            raise ValueError("weather reference is not strictly before cutoff")
    frame = frame.drop(columns=audit_time_columns)
    forbidden = [column for column in frame if FORBIDDEN_FEATURE_RE.search(str(column))]
    if forbidden:
        raise ValueError(f"forbidden extra exogenous columns: {sorted(forbidden)}")
    non_numeric = [
        column
        for column in frame
        if not (is_numeric_dtype(frame[column]) or is_bool_dtype(frame[column]))
    ]
    if non_numeric:
        raise ValueError(f"extra exogenous columns must be numeric: {non_numeric}")
    frame = frame.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    weather_columns = [column for column in frame if WEATHER_RE.search(str(column))]
    if not weather_columns:
        raise ValueError("extra exogenous source has no predeclared weather column")
    weather = frame.loc[:, weather_columns]
    if float(weather.notna().mean().min()) < 0.995:
        raise ValueError("extra exogenous weather coverage is below 99.5%")
    return weather, {
        "path": str(path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "sha256": actual_hash,
        "classification": causality.get("classification"),
        "strict_pit_eligible": True,
        "storm_used": False,
        "target_or_price_used": False,
        "feature_kind": "weather",
        "rows_loaded": len(weather),
    }


def _calendar_features(index: pd.DatetimeIndex, timezone: str) -> pd.DataFrame:
    local = index.tz_convert(timezone)
    hour = local.hour.to_numpy(float)
    weekday = local.dayofweek.to_numpy(float)
    month = local.month.to_numpy(float)
    result = pd.DataFrame(index=index)
    result["cal_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    result["cal_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    result["cal_weekday_sin"] = np.sin(2.0 * np.pi * weekday / 7.0)
    result["cal_weekday_cos"] = np.cos(2.0 * np.pi * weekday / 7.0)
    result["cal_month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
    result["cal_month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
    result["cal_weekend"] = (weekday >= 5.0).astype(float)
    result["cal_dst_fold"] = np.asarray(
        [stamp.fold for stamp in local.to_pydatetime()], dtype=float
    )
    return result


def _family_features(frame: pd.DataFrame, timezone: str) -> pd.DataFrame:
    if any(FORBIDDEN_FEATURE_RE.search(str(column)) for column in frame):
        raise ValueError("candidate family contains a forbidden feature name")
    result = _calendar_features(frame.index, timezone)
    for column in frame:
        values = pd.to_numeric(frame[column], errors="coerce").astype(float)
        result[f"exo__{column}"] = values
        result[f"missing__{column}"] = values.isna().astype(float)
    return result


def _new_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**MODEL_SPEC)


def _raw_correction(
    model: HistGradientBoostingRegressor, features: pd.DataFrame
) -> np.ndarray:
    return np.clip(
        np.asarray(model.predict(features), dtype=float),
        -RAW_CORRECTION_CLIP,
        RAW_CORRECTION_CLIP,
    )


def _exact_l1_scale(residual: np.ndarray, correction: np.ndarray) -> float:
    residual = np.asarray(residual, dtype=float)
    correction = np.asarray(correction, dtype=float)
    active = np.abs(correction) > 1e-12
    if not bool(active.any()):
        return 0.0
    ratios = residual[active] / correction[active]
    weights = np.abs(correction[active])
    order = np.argsort(ratios, kind="mergesort")
    ratios = ratios[order]
    weights = weights[order]
    position = int(
        np.searchsorted(np.cumsum(weights), 0.5 * weights.sum(), side="left")
    )
    return float(np.clip(ratios[position], 0.0, 1.0))


def _crossfit_scale(
    features: pd.DataFrame,
    actual: pd.Series,
    reference: pd.Series,
    *,
    timezone: str,
) -> tuple[float, dict[str, Any]]:
    local_days = pd.Index(features.index.tz_convert(timezone).date)
    days = local_days.unique()
    if len(days) != A_DAYS:
        raise ValueError(f"cross-fit requires exactly {A_DAYS} A days")
    residual = actual - reference
    oof_prediction = np.full(len(features), np.nan, dtype=float)
    folds: list[dict[str, Any]] = []
    for fold in range(CROSSFIT_FOLDS):
        validation_start = CROSSFIT_WARMUP_DAYS + fold * CROSSFIT_FOLD_DAYS
        validation_end = validation_start + CROSSFIT_FOLD_DAYS
        train_days = days[:validation_start]
        validation_days = days[validation_start:validation_end]
        train_mask = np.asarray(local_days.isin(train_days), dtype=bool)
        validation_mask = np.asarray(local_days.isin(validation_days), dtype=bool)
        model = _new_model().fit(features.loc[train_mask], residual.loc[train_mask])
        oof_prediction[validation_mask] = _raw_correction(
            model, features.loc[validation_mask]
        )
        folds.append(
            {
                "fold": fold + 1,
                "fit_days": len(train_days),
                "validation_days": len(validation_days),
                "validation_start": str(validation_days[0]),
                "validation_end": str(validation_days[-1]),
            }
        )
    covered = np.isfinite(oof_prediction)
    expected_covered = np.asarray(
        local_days.isin(days[CROSSFIT_WARMUP_DAYS:]), dtype=bool
    )
    if not np.array_equal(covered, expected_covered):
        raise RuntimeError("A cross-fit coverage differs from frozen protocol")
    scale = _exact_l1_scale(
        residual.to_numpy(float)[covered], oof_prediction[covered]
    )
    baseline_mae = float(np.mean(np.abs(residual.to_numpy(float)[covered])))
    candidate_mae = float(
        np.mean(
            np.abs(
                residual.to_numpy(float)[covered]
                - scale * oof_prediction[covered]
            )
        )
    )
    return scale, {
        "warmup_days": CROSSFIT_WARMUP_DAYS,
        "covered_days": A_DAYS - CROSSFIT_WARMUP_DAYS,
        "folds": folds,
        "learned_scale": scale,
        "reference_mae": baseline_mae,
        "candidate_mae": candidate_mae,
        "gain": baseline_mae - candidate_mae,
    }


def _fit_full_a_predict(
    features_a: pd.DataFrame,
    actual_a: pd.Series,
    reference_a: pd.Series,
    features_out: pd.DataFrame,
    scale: float,
) -> pd.Series:
    model = _new_model().fit(features_a, actual_a - reference_a)
    correction = float(scale) * _raw_correction(model, features_out)
    return pd.Series(correction, index=features_out.index, name="correction")


def _paired_daily_interval(
    index: pd.DatetimeIndex,
    actual: pd.Series,
    reference: pd.Series,
    candidate: pd.Series,
    *,
    timezone: str,
    draws: int = 10_000,
) -> dict[str, float]:
    local_days = pd.Index(index.tz_convert(timezone).date)
    gain = np.abs(actual.to_numpy(float) - reference.to_numpy(float)) - np.abs(
        actual.to_numpy(float) - candidate.to_numpy(float)
    )
    daily = pd.Series(gain, index=local_days).groupby(level=0, sort=False).mean()
    generator = np.random.default_rng(42)
    samples = generator.choice(
        daily.to_numpy(float), size=(draws, len(daily)), replace=True
    ).mean(axis=1)
    return {
        "mean_daily_gain": float(daily.mean()),
        "bootstrap_95_low": float(np.quantile(samples, 0.025)),
        "bootstrap_95_high": float(np.quantile(samples, 0.975)),
        "daily_win_rate": float((daily > 0.0).mean()),
    }


def _score_block(
    frame: pd.DataFrame,
    candidate: pd.Series,
    control: pd.Series,
    *,
    timezone: str,
) -> dict[str, Any]:
    local_days = pd.Index(frame.index.tz_convert(timezone).date)
    days = local_days.unique()
    if len(days) != 60:
        raise ValueError("B1/B2 scoring block must contain exactly 60 local days")
    masks = {
        "all": np.ones(len(frame), dtype=bool),
        "first30": np.asarray(local_days.isin(days[:30]), dtype=bool),
        "last30": np.asarray(local_days.isin(days[30:]), dtype=bool),
    }
    actual = frame["actual"]
    reference = frame["reference"]
    output: dict[str, Any] = {}
    for name, mask in masks.items():
        actual_values = actual.to_numpy(float)[mask]
        reference_values = reference.to_numpy(float)[mask]
        candidate_values = candidate.to_numpy(float)[mask]
        control_values = control.to_numpy(float)[mask]
        reference_mae = float(np.mean(np.abs(actual_values - reference_values)))
        candidate_mae = float(np.mean(np.abs(actual_values - candidate_values)))
        control_mae = float(np.mean(np.abs(actual_values - control_values)))
        output[name] = {
            "reference_mae": reference_mae,
            "candidate_mae": candidate_mae,
            "calendar_control_mae": control_mae,
            "gain_vs_frozen_reference": reference_mae - candidate_mae,
            "gain_vs_calendar_control": control_mae - candidate_mae,
        }
    output["paired_day_interval_vs_frozen_reference"] = _paired_daily_interval(
        frame.index,
        actual,
        reference,
        candidate,
        timezone=timezone,
    )
    return output


def _passes_b1(
    score: Mapping[str, Any], *, minimum_gain: float, minimum_bootstrap_low: float
) -> bool:
    return bool(
        score["all"]["gain_vs_frozen_reference"] >= minimum_gain
        and score["first30"]["gain_vs_frozen_reference"] > 0.0
        and score["last30"]["gain_vs_frozen_reference"] > 0.0
        and score["all"]["gain_vs_calendar_control"] > 0.0
        and score["paired_day_interval_vs_frozen_reference"]["bootstrap_95_low"]
        >= minimum_bootstrap_low
    )


def _passes_b2(score: Mapping[str, Any], *, minimum_bootstrap_low: float) -> bool:
    return bool(
        score["all"]["gain_vs_frozen_reference"] > 0.0
        and score["first30"]["gain_vs_frozen_reference"] > 0.0
        and score["last30"]["gain_vs_frozen_reference"] > 0.0
        and score["paired_day_interval_vs_frozen_reference"]["bootstrap_95_low"]
        >= minimum_bootstrap_low
    )


def _load_storm_evaluation(
    path: Path, expected: pd.DatetimeIndex
) -> pd.Series:
    """Load Storm only after candidate predictions exist (evaluation only)."""

    import pyarrow.parquet as pq

    schema = set(pq.read_schema(path).names)
    timestamp_column = next(
        (
            name
            for name in ("delivery_start_utc", "value_time_utc")
            if name in schema
        ),
        None,
    )
    prediction_column = next(
        (
            name
            for name in ("storm_prediction", "prediction", "value", "storm__q50")
            if name in schema
        ),
        None,
    )
    if timestamp_column is None or prediction_column is None:
        raise ValueError("Storm evaluation parquet lacks time/prediction columns")
    frame = pd.read_parquet(
        path,
        columns=(timestamp_column, prediction_column),
        filters=_parquet_filters_for_index(timestamp_column, expected),
    )
    index = pd.DatetimeIndex(
        pd.to_datetime(frame[timestamp_column], utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    if index.has_duplicates or not index.equals(expected):
        raise ValueError("Storm evaluation timeline is not exact")
    values = pd.to_numeric(frame[prediction_column], errors="raise").astype(float)
    if not bool(np.isfinite(values.to_numpy()).all()):
        raise ValueError("Storm evaluation contains non-finite values")
    values.index = index
    values.name = "storm_evaluation_only"
    return values


def _storm_statistics(
    frame: pd.DataFrame,
    candidate: pd.Series,
    storm: pd.Series,
    *,
    timezone: str,
) -> dict[str, Any]:
    daily = pd.DataFrame(index=frame.index)
    actual = frame["actual"].to_numpy(float)
    candidate_values = candidate.to_numpy(float)
    storm_values = storm.to_numpy(float)
    daily["candidate_abs"] = np.abs(actual - candidate_values)
    daily["storm_abs"] = np.abs(actual - storm_values)
    daily["candidate_sq"] = np.square(actual - candidate_values)
    daily["storm_sq"] = np.square(actual - storm_values)
    daily["candidate_signed"] = actual - candidate_values
    daily["storm_signed"] = actual - storm_values
    daily["day"] = frame.index.tz_convert(timezone).date
    grouped = daily.groupby("day", sort=False)
    metrics = pd.DataFrame(
        {
            "candidate_mae": grouped["candidate_abs"].mean(),
            "storm_mae": grouped["storm_abs"].mean(),
            "candidate_rmse": np.sqrt(grouped["candidate_sq"].mean()),
            "storm_rmse": np.sqrt(grouped["storm_sq"].mean()),
            "candidate_abs_bias": grouped["candidate_signed"].mean().abs(),
            "storm_abs_bias": grouped["storm_signed"].mean().abs(),
        }
    )
    output: dict[str, Any] = {
        "role": "evaluation_only",
        "used_by_model": False,
        "used_by_family_selection": False,
        "used_by_b1_or_b2_gate": False,
        "hours": len(frame),
        "days": len(metrics),
    }
    for metric in ("mae", "rmse", "abs_bias"):
        candidate_column = f"candidate_{metric}"
        storm_column = f"storm_{metric}"
        output[metric] = {
            "candidate": float(metrics[candidate_column].mean()),
            "storm": float(metrics[storm_column].mean()),
            "daily_win_rate": float(
                (metrics[candidate_column] < metrics[storm_column]).mean()
            ),
        }
    return output


def _build_families(
    expected: pd.DatetimeIndex,
    *,
    zone: str,
    timezone: str,
    extra_path: Path | None,
    extra_manifest: Path | None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    families, audit = _revision_families(
        expected, zone=zone, timezone=timezone
    )
    if extra_path is not None:
        if extra_manifest is None:
            raise ValueError("extra exogenous parquet requires its manifest")
        extra, extra_audit = _read_extra_exogenous(
            extra_path,
            extra_manifest,
            expected,
            zone=zone,
            timezone=timezone,
        )
        if extra_audit.get("feature_kind") == "fundamental":
            families["fundamental"] = extra
            families["local_revision_plus_fundamental"] = pd.concat(
                [families["local_revision"], extra], axis=1
            )
            families["continental_revision_plus_fundamental"] = pd.concat(
                [families["continental_revision"], extra], axis=1
            )
        else:
            families["weather"] = extra
            families["continental_revision_plus_weather"] = pd.concat(
                [families["continental_revision"], extra], axis=1
            )
        audit["extra_exogenous"] = extra_audit
    return families, audit


def _fit_control(
    reference_a: pd.DataFrame,
    score_frame: pd.DataFrame,
    *,
    timezone: str,
    frozen_scale: float | None = None,
) -> tuple[pd.Series, float, dict[str, Any]]:
    features_a = _calendar_features(reference_a.index, timezone)
    features_score = _calendar_features(score_frame.index, timezone)
    if frozen_scale is None:
        scale, crossfit = _crossfit_scale(
            features_a,
            reference_a["actual"],
            reference_a["reference"],
            timezone=timezone,
        )
    else:
        scale = float(frozen_scale)
        crossfit = {"learned_scale": scale, "relearned": False}
    correction = _fit_full_a_predict(
        features_a,
        reference_a["actual"],
        reference_a["reference"],
        features_score,
        scale,
    )
    return score_frame["reference"] + correction, scale, crossfit


def _run_b1(args: argparse.Namespace) -> int:
    zone = args.zone.upper()
    timezone = ZONE_SPECS[zone]["timezone"]
    run_dir = Path(args.run_dir or _default_run_dir(zone)).resolve()
    recipe_path = Path(args.reference_recipe or _default_recipe_path(zone)).resolve()
    reference_audit = _verify_frozen_reference(zone, run_dir, recipe_path)
    blocks, reconstruction_audit = _rebuild_frozen_reference_blocks(
        zone,
        run_dir,
        timezone=timezone,
        phase="b1",
        threads=int(args.threads),
    )
    expected = blocks["A"].index.append(blocks["B1"].index)
    families, source_audit = _build_families(
        expected,
        zone=zone,
        timezone=timezone,
        extra_path=Path(args.extra_exogenous).resolve()
        if args.extra_exogenous
        else None,
        extra_manifest=Path(args.extra_manifest).resolve()
        if args.extra_manifest
        else None,
    )
    control, control_scale, control_crossfit = _fit_control(
        blocks["A"], blocks["B1"], timezone=timezone
    )
    candidates: dict[str, Any] = {}
    predictions: dict[str, pd.Series] = {}
    for family_name, family_frame in families.items():
        features = _family_features(family_frame, timezone)
        features_a = features.loc[blocks["A"].index]
        features_b1 = features.loc[blocks["B1"].index]
        scale, crossfit = _crossfit_scale(
            features_a,
            blocks["A"]["actual"],
            blocks["A"]["reference"],
            timezone=timezone,
        )
        correction = _fit_full_a_predict(
            features_a,
            blocks["A"]["actual"],
            blocks["A"]["reference"],
            features_b1,
            scale,
        )
        prediction = blocks["B1"]["reference"] + correction
        score = _score_block(
            blocks["B1"], prediction, control, timezone=timezone
        )
        passed = _passes_b1(
            score,
            minimum_gain=float(args.minimum_b1_gain),
            minimum_bootstrap_low=float(args.minimum_bootstrap_low),
        )
        candidates[family_name] = {
            "columns": family_frame.columns.astype(str).tolist(),
            "crossfit_A": crossfit,
            "B1": score,
            "passes": passed,
        }
        predictions[family_name] = prediction
    passing = [name for name, result in candidates.items() if result["passes"]]
    selected = min(
        passing,
        key=lambda name: candidates[name]["B1"]["all"]["candidate_mae"],
        default=None,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": {
            "phase": "b1_selection",
            "zone": zone,
            "timezone": timezone,
            "fit_A": [A_START_DAY, "2025-04-13"],
            "selection_B1": [B1_START_DAY, "2025-06-12"],
            "B2_rows_loaded": 0,
            "final_rows_loaded": 0,
            "storm_used_as_feature": False,
            "external_price_forecast_used": False,
            "model_spec": MODEL_SPEC,
            "raw_correction_clip": RAW_CORRECTION_CLIP,
            "minimum_b1_gain": float(args.minimum_b1_gain),
            "minimum_bootstrap_low": float(args.minimum_bootstrap_low),
            "production_changed": False,
            "automatic_promotion": False,
        },
        "reference_audit": reference_audit,
        "reference_reconstruction_audit": reconstruction_audit,
        "source_audit": source_audit,
        "calendar_control": {"crossfit_A": control_crossfit},
        "candidates": candidates,
        "selected_family": selected,
        "selection_passed": selected is not None,
        "research_candidate_only": True,
        "production_changed": False,
        "status": "frozen_for_b2_veto" if selected else "no_family_passed_b1",
    }
    if selected is not None:
        payload["frozen_recipe"] = {
            "family": selected,
            "columns": candidates[selected]["columns"],
            "correction_scale": candidates[selected]["crossfit_A"][
                "learned_scale"
            ],
            "calendar_control_scale": control_scale,
            "source_hashes": {
                name: details["sha256"]
                for name, details in source_audit["files"].items()
            },
            "extra_source_sha256": source_audit.get("extra_exogenous", {}).get(
                "sha256"
            ),
            "reference_recipe_sha256": reference_audit["recipe_sha256"],
            "reference_checksum_manifest_sha256": reference_audit[
                "checksum_manifest_sha256"
            ],
            "reference_reconstruction_implementation_sha256": reconstruction_audit[
                "implementation_sha256"
            ],
            "extended_oof_sha256": reconstruction_audit["extended_oof_sha256"],
            "screen_implementation_sha256": _sha256(Path(__file__).resolve()),
            "minimum_bootstrap_low": float(args.minimum_bootstrap_low),
        }
        if args.storm_evaluation_file:
            storm = _load_storm_evaluation(
                Path(args.storm_evaluation_file).resolve(), blocks["B1"].index
            )
            payload["storm_evaluation"] = _storm_statistics(
                blocks["B1"], predictions[selected], storm, timezone=timezone
            )
            payload["storm_evaluation"].update(
                {
                    "path": str(Path(args.storm_evaluation_file).resolve()),
                    "sha256": _sha256(Path(args.storm_evaluation_file).resolve()),
                }
            )
        else:
            payload["storm_evaluation"] = {
                "status": "not_available",
                "role": "evaluation_only",
                "used_by_b1_or_b2_gate": False,
            }
    _atomic_json(
        payload,
        Path(args.output_json).resolve() if args.output_json else None,
    )
    return 0 if selected is not None else 2


def _load_b1_recipe(path: Path, *, zone: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol", {})
    if protocol.get("phase") != "b1_selection" or protocol.get("zone") != zone:
        raise ValueError("B2 recipe is not a same-zone B1 output from this screen")
    if payload.get("selection_passed") is not True or "frozen_recipe" not in payload:
        raise ValueError("B1 output has no passing frozen recipe")
    if protocol.get("final_rows_loaded") != 0 or protocol.get("B2_rows_loaded") != 0:
        raise ValueError("B1 recipe did not preserve the holdout contract")
    if protocol.get("storm_used_as_feature") is not False:
        raise ValueError("B1 recipe lacks a negative Storm feature proof")
    return payload, payload["frozen_recipe"]


def _verify_source_hashes(source_audit: Mapping[str, Any], recipe: Mapping[str, Any]) -> None:
    actual = {
        name: details["sha256"] for name, details in source_audit["files"].items()
    }
    if actual != recipe.get("source_hashes"):
        raise ValueError("revision PIT sources changed after B1 recipe freeze")
    if source_audit.get("extra_exogenous", {}).get("sha256") != recipe.get(
        "extra_source_sha256"
    ):
        raise ValueError("extra exogenous source changed after B1 recipe freeze")


def _run_b2(args: argparse.Namespace) -> int:
    if not args.recipe:
        raise ValueError("phase b2 requires --recipe")
    zone = args.zone.upper()
    timezone = ZONE_SPECS[zone]["timezone"]
    run_dir = Path(args.run_dir or _default_run_dir(zone)).resolve()
    reference_recipe = Path(
        args.reference_recipe or _default_recipe_path(zone)
    ).resolve()
    reference_audit = _verify_frozen_reference(zone, run_dir, reference_recipe)
    b1_payload, recipe = _load_b1_recipe(Path(args.recipe).resolve(), zone=zone)
    if recipe.get("reference_recipe_sha256") != reference_audit["recipe_sha256"]:
        raise ValueError("sealed reference recipe changed after B1")
    if recipe.get("reference_checksum_manifest_sha256") != reference_audit[
        "checksum_manifest_sha256"
    ]:
        raise ValueError("sealed reference run changed after B1")
    if recipe.get("screen_implementation_sha256") != _sha256(Path(__file__).resolve()):
        raise ValueError("exogenous screen implementation changed after B1")
    if float(recipe.get("minimum_bootstrap_low", np.nan)) != float(
        args.minimum_bootstrap_low
    ):
        raise ValueError("B2 bootstrap gate differs from the frozen B1 gate")
    blocks, reconstruction_audit = _rebuild_frozen_reference_blocks(
        zone,
        run_dir,
        timezone=timezone,
        phase="b2",
        threads=int(args.threads),
    )
    if recipe.get("reference_reconstruction_implementation_sha256") != (
        reconstruction_audit["implementation_sha256"]
    ):
        raise ValueError("reference reconstruction code changed after B1")
    if recipe.get("extended_oof_sha256") != reconstruction_audit[
        "extended_oof_sha256"
    ]:
        raise ValueError("extended OOF source changed after B1")
    expected = blocks["A"].index.append(blocks["B2"].index)
    families, source_audit = _build_families(
        expected,
        zone=zone,
        timezone=timezone,
        extra_path=Path(args.extra_exogenous).resolve()
        if args.extra_exogenous
        else None,
        extra_manifest=Path(args.extra_manifest).resolve()
        if args.extra_manifest
        else None,
    )
    _verify_source_hashes(source_audit, recipe)
    family_name = str(recipe["family"])
    if family_name not in families:
        raise ValueError("frozen B1 family is unavailable in B2")
    if families[family_name].columns.astype(str).tolist() != recipe.get("columns"):
        raise ValueError("B2 family schema differs from frozen B1 schema")
    features = _family_features(families[family_name], timezone)
    features_a = features.loc[blocks["A"].index]
    features_b2 = features.loc[blocks["B2"].index]
    correction = _fit_full_a_predict(
        features_a,
        blocks["A"]["actual"],
        blocks["A"]["reference"],
        features_b2,
        float(recipe["correction_scale"]),
    )
    prediction = blocks["B2"]["reference"] + correction
    control, _, _ = _fit_control(
        blocks["A"],
        blocks["B2"],
        timezone=timezone,
        frozen_scale=float(recipe["calendar_control_scale"]),
    )
    score = _score_block(blocks["B2"], prediction, control, timezone=timezone)
    passed = _passes_b2(
        score, minimum_bootstrap_low=float(args.minimum_bootstrap_low)
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": {
            "phase": "b2_veto",
            "zone": zone,
            "timezone": timezone,
            "fit_A": [A_START_DAY, "2025-04-13"],
            "B1_rows_used_for_fit_or_selection": 0,
            "veto_B2": [B2_START_DAY, "2025-08-11"],
            "final_rows_loaded": 0,
            "family_or_scale_reselected_on_B2": False,
            "storm_used_as_feature": False,
            "external_price_forecast_used": False,
            "minimum_bootstrap_low": float(args.minimum_bootstrap_low),
            "production_changed": False,
            "automatic_promotion": False,
        },
        "b1_recipe_path": str(Path(args.recipe).resolve()),
        "b1_recipe_sha256": _sha256(Path(args.recipe).resolve()),
        "b1_selection_summary": {
            "selected_family": b1_payload.get("selected_family"),
            "status": b1_payload.get("status"),
        },
        "frozen_recipe": recipe,
        "reference_audit": reference_audit,
        "reference_reconstruction_audit": reconstruction_audit,
        "source_audit": source_audit,
        "B2": score,
        "veto_passed": passed,
        "research_candidate_only": True,
        "production_changed": False,
        "status": "research_candidate_passed_b2" if passed else "b2_veto_failed",
    }
    if args.storm_evaluation_file:
        storm = _load_storm_evaluation(
            Path(args.storm_evaluation_file).resolve(), blocks["B2"].index
        )
        payload["storm_evaluation"] = _storm_statistics(
            blocks["B2"], prediction, storm, timezone=timezone
        )
        payload["storm_evaluation"].update(
            {
                "path": str(Path(args.storm_evaluation_file).resolve()),
                "sha256": _sha256(Path(args.storm_evaluation_file).resolve()),
            }
        )
    else:
        payload["storm_evaluation"] = {
            "status": "not_available",
            "role": "evaluation_only",
            "used_by_b1_or_b2_gate": False,
        }
    _atomic_json(
        payload,
        Path(args.output_json).resolve() if args.output_json else None,
    )
    return 0 if passed else 2


def _inventory() -> dict[str, Any]:
    zones: dict[str, Any] = {}
    for zone in ZONE_SPECS:
        run_dir = _default_run_dir(zone)
        recipe = _default_recipe_path(zone)
        try:
            reference = _verify_frozen_reference(zone, run_dir, recipe)
            status = "verified"
            error = None
        except Exception as exc:  # inventory reports, screens fail closed
            reference = None
            status = "blocked"
            error = str(exc)
        local_files = [
            PIT_ROOT / f"{zone.lower()}_residual_load_revision_{signal}.parquet"
            for signal in ("abs_delta", "std", "age_hours")
        ]
        zones[zone] = {
            "frozen_reference": status,
            "reference": reference,
            "reference_error": error,
            "already_in_production": {
                "calendar": True,
                "price_lags": [24, 48, 168, 336],
                "pit_residual_load_panel": list(ALL_PHYSICAL_ZONES),
            },
            "strict_local_enrichment_ready": all(path.is_file() for path in local_files),
            "strict_local_enrichment": [str(path.resolve()) for path in local_files],
            "weather_gap": (
                "FR-only fixed-lead Open-Meteo artifact; no same-zone local artifact"
            ),
            "generation_and_flow_gap": (
                "no full-window, zone-specific, checksum-manifested PIT panel"
            ),
            "storm_evaluation": (
                "sealed final only for DE/BE/NL; unavailable for ES; never a feature"
            ),
        }
    openmeteo = ROOT / "runs/tmp/openmeteo_d2_ab1b2/weather.parquet"
    openmeteo_manifest = ROOT / "runs/tmp/openmeteo_d2_ab1b2/weather.manifest.json"
    eco2mix = ROOT / "runs/tmp/eco2mix_ab1b2_exploratory_non_pit.parquet"
    return {
        "schema_version": 1,
        "scope": "research_only_no_production_change",
        "zones": zones,
        "shared": {
            "continental_revision_panel_ready": all(
                (
                    PIT_ROOT
                    / f"{zone.lower()}_residual_load_revision_{signal}.parquet"
                ).is_file()
                for zone in ALL_PHYSICAL_ZONES
                for signal in ("abs_delta", "std", "age_hours")
            ),
            "openmeteo_fr_fixed_lead": {
                "data_exists": openmeteo.is_file(),
                "manifest_exists": openmeteo_manifest.is_file(),
                "production_license_must_be_confirmed": True,
            },
            "eco2mix": {
                "data_exists": eco2mix.is_file(),
                "classification": "revision_latest_non_pit_exploratory_only",
                "promotable": False,
            },
            "neighbour_actual_prices": {
                "available": True,
                "permitted_use": "causal lags only",
                "contemporaneous_delivery_price_forbidden": True,
            },
            "nonstorm_price_experts": {
                "materialized": True,
                "compatible_with_autonomous_only": False,
                "included_in_this_screen": False,
            },
            "storm": {
                "feature_use": False,
                "selection_or_gate_use": False,
                "role": "optional evaluation_only after prediction",
            },
        },
        "production_changed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict multi-zone causal exogenous B1/B2 research screen"
    )
    parser.add_argument("--phase", choices=("inventory", "b1", "b2"), required=True)
    parser.add_argument("--zone", choices=tuple(ZONE_SPECS))
    parser.add_argument("--run-dir")
    parser.add_argument("--reference-recipe")
    parser.add_argument("--extra-exogenous")
    parser.add_argument("--extra-manifest")
    parser.add_argument("--recipe", help="Frozen B1 output; required by B2")
    parser.add_argument("--storm-evaluation-file")
    parser.add_argument("--output-json")
    parser.add_argument("--minimum-b1-gain", type=float, default=0.10)
    parser.add_argument("--minimum-bootstrap-low", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=-1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.phase == "inventory":
        _atomic_json(
            _inventory(),
            Path(args.output_json).resolve() if args.output_json else None,
        )
        return 0
    if not args.zone:
        parser.error("B1/B2 requires --zone")
    if args.extra_manifest and not args.extra_exogenous:
        parser.error("--extra-manifest requires --extra-exogenous")
    if args.minimum_b1_gain < 0.0:
        parser.error("--minimum-b1-gain must be non-negative")
    return _run_b1(args) if args.phase == "b1" else _run_b2(args)


if __name__ == "__main__":
    raise SystemExit(main())
