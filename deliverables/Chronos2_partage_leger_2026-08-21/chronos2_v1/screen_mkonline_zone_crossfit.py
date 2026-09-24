#!/usr/bin/env python
"""Strict, zone-aware MKOnline blend calibration (B1/B2 only).

The script deliberately exposes no ``final`` phase.  Autonomous CSV inputs are
read only up to the end of the requested calibration block and primary Parquet
inputs are predicate-filtered to that same block.  The protocol is fixed:

* EXT: 223 days immediately preceding A;
* A: 245 days, used for a 5 x 49-day chronological cross-fit;
* B1: 60 days, one-shot selection gate;
* B2: 60 days, one-shot veto with an already-frozen weight.

The constrained L1 blend weight is learned exactly (weighted median), not by a
grid search.  A block passes only when its overall MAE gain is at least
0.75 EUR/MWh and each chronological 30-day half has a strictly positive gain.

``--timezone`` is the delivery/model timezone.  ``--cutoff-timezone`` is kept
separate because the historical primary materialisations currently available
for the coupled zones were queried at civil D-1 08:00 Europe/Paris.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chronos2_hourly.features import build_hourly_feature_matrix  # noqa: E402
from chronos2_hourly.models.residual_corrector import (  # noqa: E402
    ResidualMetaFeatureBuilder,
)


SCRIPT_VERSION = "1.0.0"
CHRONOS_EXPERT_COLUMNS = tuple(
    f"chronos2__{quantile}" for quantile in ("q10", "q50", "q90")
)
RICH_CALENDAR_COUNTRIES = ("FR", "DE", "BE", "ES", "NL")
SUPPORTED_ZONES = frozenset(("FR", "DE", "BE", "NL", "ES"))
COUNTRY_NAMES_BY_ZONE = {
    "FR": frozenset(("FR", "FRANCE")),
    "DE": frozenset(("DE", "GERMANY", "DEUTSCHLAND")),
    "BE": frozenset(("BE", "BELGIUM", "BELGIQUE")),
    "NL": frozenset(("NL", "NETHERLANDS", "THE NETHERLANDS")),
    "ES": frozenset(("ES", "SPAIN", "ESPANA", "ESPAÑA")),
}

EXT_DAYS = 223
A_DAYS = 245
B1_DAYS = 60
B2_DAYS = 60
CROSSFIT_FOLDS = 5
CROSSFIT_DAYS = 49
MINIMUM_BLOCK_GAIN = 0.75
ORIGIN_CLOCK = "08:00"


@dataclass(frozen=True)
class CalibrationContract:
    """Civil delivery-day boundaries for the immutable calibration protocol."""

    zone: str
    timezone: str
    a_start_day: pd.Timestamp
    a_start_utc: pd.Timestamp
    a_end_utc: pd.Timestamp
    b1_end_utc: pd.Timestamp
    b2_end_utc: pd.Timestamp

    @property
    def ext_start_day(self) -> pd.Timestamp:
        return self.a_start_day - pd.Timedelta(days=EXT_DAYS)

    @property
    def a_end_day(self) -> pd.Timestamp:
        return self.a_start_day + pd.Timedelta(days=A_DAYS)

    @property
    def b1_end_day(self) -> pd.Timestamp:
        return self.a_start_day + pd.Timedelta(days=A_DAYS + B1_DAYS)

    @property
    def b2_end_day(self) -> pd.Timestamp:
        return self.a_start_day + pd.Timedelta(
            days=A_DAYS + B1_DAYS + B2_DAYS
        )


def _civil_midnight_utc(day: pd.Timestamp, timezone: str) -> pd.Timestamp:
    naive = pd.Timestamp(day).normalize()
    if naive.tzinfo is not None:
        naive = naive.tz_localize(None)
    return naive.tz_localize(timezone).tz_convert("UTC")


def _calibration_contract(
    run_dir: Path, *, zone: str, timezone: str
) -> CalibrationContract:
    path = run_dir / "backtest_hourly_oof.csv.gz"
    # The published backtest includes a longer price-history prefix whose
    # predictions are intentionally NaN. A begins at the first genuine OOF
    # fold, not at the first historical target row. Read metadata only here:
    # no actual, expert prediction, B2 score, or final label is exposed.
    metadata = pd.read_csv(
        path,
        usecols=["delivery_start_utc", "fold_id"],
    )
    observed = metadata.loc[metadata["fold_id"].notna(), "delivery_start_utc"]
    if observed.empty:
        raise ValueError(f"{path}: autonomous artifact contains no OOF fold")
    first = pd.Timestamp(observed.iloc[0])
    if first.tzinfo is None:
        raise ValueError(f"{path}: naive first OOF timestamp")
    first = first.tz_convert("UTC")
    first_local = first.tz_convert(timezone)
    a_start_day = pd.Timestamp(first_local.date())
    a_start_utc = _civil_midnight_utc(a_start_day, timezone)
    if first != a_start_utc:
        raise ValueError(
            f"{path}: OOF must start at local midnight in {timezone}, got {first}"
        )
    return CalibrationContract(
        zone=zone,
        timezone=timezone,
        a_start_day=a_start_day,
        a_start_utc=a_start_utc,
        a_end_utc=_civil_midnight_utc(a_start_day + pd.Timedelta(days=A_DAYS), timezone),
        b1_end_utc=_civil_midnight_utc(
            a_start_day + pd.Timedelta(days=A_DAYS + B1_DAYS), timezone
        ),
        b2_end_utc=_civil_midnight_utc(
            a_start_day + pd.Timedelta(days=A_DAYS + B1_DAYS + B2_DAYS),
            timezone,
        ),
    )


def _read_csv_exactly_to(
    path: Path, timestamp_column: str, end_exclusive_utc: pd.Timestamp
) -> pd.DataFrame:
    """Read no CSV row at or beyond ``end_exclusive_utc``."""

    first_row = pd.read_csv(path, nrows=1, usecols=[timestamp_column])
    if first_row.empty:
        raise ValueError(f"{path}: empty input")
    first = pd.Timestamp(first_row[timestamp_column].iloc[0])
    if first.tzinfo is None:
        raise ValueError(f"{path}: naive first timestamp")
    first = first.tz_convert("UTC")
    end = pd.Timestamp(end_exclusive_utc).tz_convert("UTC")
    hours = (end - first) / pd.Timedelta(hours=1)
    if not np.isfinite(hours) or float(hours) != int(hours) or hours <= 0:
        raise ValueError(f"{path}: invalid hourly boundary {first}..{end}")
    frame = pd.read_csv(path, nrows=int(hours))
    index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop(timestamp_column), utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    expected = pd.date_range(first, end, freq="h", inclusive="left")
    if not index.equals(expected):
        raise ValueError(f"{path}: timeline before calibration cutoff is not exact")
    frame.index = index
    return frame


def _contains_storm(value: object) -> bool:
    return "storm" in str(value).casefold()


def _reject_storm_values(values: Sequence[object], *, context: str) -> None:
    offenders = sorted({str(value) for value in values if _contains_storm(value)})
    if offenders:
        raise ValueError(f"{context}: forbidden Storm token in {offenders}")


def _scalar_dependency_values(value: object) -> list[object]:
    """Collect dependency evidence values, excluding the negative-proof key."""

    if isinstance(value, Mapping):
        result: list[object] = []
        for key, nested in value.items():
            if str(key).casefold() == "storm_token_found":
                continue
            result.append(key)
            result.extend(_scalar_dependency_values(nested))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for nested in value:
            result.extend(_scalar_dependency_values(nested))
        return result
    return [value]


def _audit_autonomous_run(
    run_dir: Path, *, zone: str, timezone: str
) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_zone = str(manifest.get("zone", "")).upper()
    manifest_timezone = str(manifest.get("timezone", ""))
    if manifest_zone != zone:
        raise ValueError(
            f"{manifest_path}: zone {manifest_zone!r}, expected {zone!r}"
        )
    if manifest_timezone != timezone:
        raise ValueError(
            f"{manifest_path}: timezone {manifest_timezone!r}, expected {timezone!r}"
        )
    if manifest.get("target_contract") != "hourly_utc_no_interpolation":
        raise ValueError(f"{manifest_path}: unsafe target contract")
    if manifest.get("delivery_horizon") != "dynamic_23_24_25":
        raise ValueError(f"{manifest_path}: unsafe delivery horizon contract")

    feature_manifest = pd.read_csv(run_dir / "feature_manifest.csv")
    if "feature" not in feature_manifest:
        raise ValueError(f"{run_dir / 'feature_manifest.csv'}: missing feature column")
    features = feature_manifest["feature"].astype(str).tolist()
    _reject_storm_values(features, context="autonomous feature schema")
    active_features = [str(value) for value in manifest.get("active_features", [])]
    _reject_storm_values(active_features, context="autonomous active features")

    origins: set[str] = set()
    covariates = (
        manifest.get("input_diagnostics", {}).get("covariates", {})
        if isinstance(manifest.get("input_diagnostics", {}), Mapping)
        else {}
    )
    if isinstance(covariates, Mapping):
        _reject_storm_values(
            _scalar_dependency_values(covariates),
            context="autonomous covariate inputs",
        )
        for details in covariates.values():
            if isinstance(details, Mapping) and details.get(
                "forecast_origin_local_time"
            ) is not None:
                origins.add(str(details["forecast_origin_local_time"]))
    prediction_inputs = manifest.get("prediction_inputs", [])
    if isinstance(prediction_inputs, (list, tuple)):
        _reject_storm_values(
            list(prediction_inputs), context="autonomous prediction inputs"
        )
    if origins != {ORIGIN_CLOCK}:
        raise ValueError(
            f"{manifest_path}: forecast origins {sorted(origins)}, expected {ORIGIN_CLOCK}"
        )
    return {
        "run_dir": str(run_dir),
        "zone": manifest_zone,
        "delivery_timezone": manifest_timezone,
        "target_contract": manifest["target_contract"],
        "delivery_horizon": manifest["delivery_horizon"],
        "forecast_origin_clock_values": sorted(origins),
        "storm_feature_count": 0,
    }


def _load_features_to(
    run_dir: Path, *, end_exclusive_utc: pd.Timestamp, timezone: str
) -> pd.DataFrame:
    aligned = _read_csv_exactly_to(
        run_dir / "inputs" / "aligned_inputs.csv.gz",
        "timestamp",
        end_exclusive_utc,
    )
    if "target" not in aligned:
        raise ValueError("aligned_inputs.csv.gz: missing target")
    target = pd.to_numeric(aligned.pop("target"), errors="raise").astype(float)
    context = _read_csv_exactly_to(
        run_dir / "inputs" / "model_covariates_with_future.csv.gz",
        "timestamp",
        end_exclusive_utc,
    )
    manifest = pd.read_csv(run_dir / "feature_manifest.csv")
    expected_columns = manifest["feature"].astype(str).tolist()
    _reject_storm_values(expected_columns, context="feature manifest")
    input_columns = [column for column in expected_columns if column in context]
    covariates = context.loc[:, input_columns].apply(pd.to_numeric, errors="coerce")
    features = build_hourly_feature_matrix(
        target,
        covariates.loc[target.index],
        price_lags=(24, 48, 168, 336),
        rolling_windows=(24, 72, 168),
        timezone=timezone,
    )
    if list(features.columns) != expected_columns:
        raise RuntimeError(
            "Reconstructed feature schema differs from the autonomous zone manifest"
        )
    features.index = features.index.tz_convert("UTC")
    features.index.name = "delivery_start_utc"
    return features


def _build_meta(
    X: pd.DataFrame,
    experts: pd.DataFrame,
    *,
    timezone: str,
    primary_country: str,
) -> pd.DataFrame:
    missing = sorted(set(CHRONOS_EXPERT_COLUMNS).difference(experts.columns))
    if missing:
        raise ValueError(f"Missing Chronos expert columns: {missing}")
    _reject_storm_values(list(experts.columns), context="expert schema")
    base = experts.loc[:, list(CHRONOS_EXPERT_COLUMNS)].copy()
    base.columns = ["q10", "q50", "q90"]
    combined = pd.concat(
        [base.add_prefix("base__"), experts.loc[:, list(CHRONOS_EXPERT_COLUMNS)]],
        axis=1,
    )
    countries = tuple(dict.fromkeys((*RICH_CALENDAR_COUNTRIES, primary_country)))
    builder = ResidualMetaFeatureBuilder(
        timezone=timezone,
        include_calendar=True,
        include_rich_calendar=True,
        rich_calendar_countries=countries,
        rich_calendar_primary_country=primary_country,
        include_daily_profiles=True,
        include_fundamental_interactions=False,
        include_missing_indicators=False,
        exclude_historical_prices=True,
        exclude_day_of_year=True,
    )
    return builder.fit_transform(X, combined)


def _load_external(
    path: Path,
    X_all: pd.DataFrame,
    *,
    contract: CalibrationContract,
) -> dict[str, Any]:
    _reject_storm_values([path.name], context="extended OOF path")
    raw = pd.read_csv(path)
    _reject_storm_values(list(raw.columns), context="extended OOF schema")
    if not set(CHRONOS_EXPERT_COLUMNS).issubset(raw.columns):
        raw = raw.rename(
            columns={q: f"chronos2__{q}" for q in ("q10", "q50", "q90")}
        )
    required = {"delivery_start_utc", "actual", *CHRONOS_EXPERT_COLUMNS}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"{path}: incomplete extended OOF artifact: {missing}")
    raw.index = pd.DatetimeIndex(
        pd.to_datetime(raw.pop("delivery_start_utc"), utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    expected = pd.date_range(
        _civil_midnight_utc(contract.ext_start_day, contract.timezone),
        contract.a_start_utc,
        freq="h",
        inclusive="left",
    )
    if not raw.index.equals(expected):
        raise ValueError(
            f"{path}: EXT must be the exact {EXT_DAYS}-day physical timeline "
            "immediately preceding A"
        )
    if not expected.isin(X_all.index).all():
        raise ValueError(f"{path}: EXT timestamps are absent from autonomous features")
    experts = raw.loc[:, list(CHRONOS_EXPERT_COLUMNS)].apply(
        pd.to_numeric, errors="raise"
    )
    actual = pd.to_numeric(raw["actual"], errors="raise").astype(float)
    if not np.isfinite(experts.to_numpy(float)).all() or not np.isfinite(
        actual.to_numpy(float)
    ).all():
        raise ValueError(f"{path}: non-finite EXT values")
    meta = _build_meta(
        X_all.loc[raw.index],
        experts,
        timezone=contract.timezone,
        primary_country=contract.zone,
    )
    base_q50 = experts["chronos2__q50"].astype(float)
    return {
        "raw": raw,
        "meta": meta,
        "base_q50": base_q50,
        "residual": actual - base_q50,
        "days": pd.Index(raw.index.tz_convert(contract.timezone).date).unique().tolist(),
    }


def _load_existing(
    run_dir: Path,
    X_all: pd.DataFrame,
    *,
    contract: CalibrationContract,
    phase: str,
) -> dict[str, Any]:
    end = contract.b1_end_utc if phase == "b1" else contract.b2_end_utc
    raw = _read_csv_exactly_to(
        run_dir / "backtest_hourly_oof.csv.gz", "delivery_start_utc", end
    )
    if "fold_id" not in raw:
        raise ValueError("autonomous OOF artifact has no fold_id")
    raw = raw.loc[raw["fold_id"].notna()].copy()
    expected = pd.date_range(contract.a_start_utc, end, freq="h", inclusive="left")
    if not raw.index.equals(expected):
        raise ValueError(f"autonomous OOF does not exactly cover A through {phase.upper()}")
    missing = sorted({"actual", *CHRONOS_EXPERT_COLUMNS}.difference(raw.columns))
    if missing:
        raise ValueError(f"autonomous OOF missing columns: {missing}")
    experts = raw.loc[:, list(CHRONOS_EXPERT_COLUMNS)].apply(
        pd.to_numeric, errors="raise"
    )
    actual = pd.to_numeric(raw["actual"], errors="raise").astype(float)
    if not np.isfinite(experts.to_numpy(float)).all() or not np.isfinite(
        actual.to_numpy(float)
    ).all():
        raise ValueError("autonomous OOF contains non-finite actual/expert values")
    meta = _build_meta(
        X_all.loc[raw.index],
        experts,
        timezone=contract.timezone,
        primary_country=contract.zone,
    )
    base_q50 = experts["chronos2__q50"].astype(float)
    local_days = pd.Index(raw.index.tz_convert(contract.timezone).date)
    days = local_days.unique().tolist()
    expected_days = A_DAYS + B1_DAYS + (B2_DAYS if phase == "b2" else 0)
    if len(days) != expected_days:
        raise ValueError(
            f"autonomous OOF has {len(days)} local days, expected {expected_days}"
        )
    return {
        "raw": raw,
        "meta": meta,
        "base_q50": base_q50,
        "residual": actual - base_q50,
        "local_days": local_days,
        "days": days,
    }


def _dependency_entries(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    series = payload.get("series")
    if isinstance(series, list):
        return [entry for entry in series if isinstance(entry, Mapping)]
    if isinstance(series, Mapping):
        entries: list[Mapping[str, Any]] = []
        for name, raw in series.items():
            if not isinstance(raw, Mapping):
                continue
            item = dict(raw)
            item.setdefault("root", name)
            entries.append(item)
        return entries
    return [payload]


def _entry_terminal(entry: Mapping[str, Any]) -> object:
    for key in ("terminal_series", "terminal", "dependency", "direct_series"):
        if entry.get(key) is not None:
            return entry[key]
    return None


def _audit_primary_dependency(
    path: Path, *, primary_series: str, zone: str
) -> dict[str, Any]:
    if _contains_storm(primary_series):
        raise ValueError("primary series contains forbidden Storm token")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: dependency manifest must be a JSON object")
    matches = [
        entry
        for entry in _dependency_entries(payload)
        if str(_entry_terminal(entry)) == primary_series
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{path}: expected one dependency entry for {primary_series}, got {len(matches)}"
        )
    entry = matches[0]
    metadata = entry.get("terminal_metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = entry.get("dependency_metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}

    terminal_type = next(
        (
            entry[key]
            for key in ("terminal_type", "dependency_type", "direct_type")
            if entry.get(key) is not None
        ),
        None,
    )
    terminal_formula = next(
        (
            entry[key]
            for key in ("terminal_formula", "dependency_formula", "direct_formula")
            if key in entry
        ),
        None,
    )
    provider = entry.get("provider", metadata.get("mercure:provider"))
    source = entry.get("source", metadata.get("mercure:source"))
    country = entry.get("country", metadata.get("mercure:country"))
    if str(terminal_type).casefold() != "primary":
        raise ValueError(f"{path}: {primary_series} is not proven terminal primary")
    if terminal_formula is not None:
        raise ValueError(f"{path}: terminal primary unexpectedly has a formula")
    if str(provider).upper() != "MKONLINE" or str(source).upper() != "WATTSIGHT":
        raise ValueError(
            f"{path}: unexpected provider/source {provider!r}/{source!r}"
        )
    if str(country).upper() not in COUNTRY_NAMES_BY_ZONE[zone]:
        raise ValueError(
            f"{path}: terminal country {country!r} does not match zone {zone}"
        )
    storm_flag = entry.get("storm_token_found", payload.get("storm_token_found"))
    if storm_flag is not False:
        raise ValueError(f"{path}: dependency manifest has no negative Storm proof")
    gate = entry.get("dependency_gate_passed", payload.get("dependency_gate_passed"))
    if gate is not True:
        raise ValueError(f"{path}: dependency gate is not explicitly passed")
    identity_fields: list[object] = [
        primary_series,
        entry.get("root"),
        entry.get("wrapper_series"),
        entry.get("formula"),
        entry.get("wrapper_formula"),
        provider,
        source,
        country,
        entry.get("label"),
        entry.get("description"),
        metadata.get("mercure:label"),
        metadata.get("mercure:description"),
    ]
    _reject_storm_values(identity_fields, context="primary dependency identity")
    _reject_storm_values(
        _scalar_dependency_values(entry), context="full primary dependency evidence"
    )
    return {
        "manifest": str(path),
        "terminal_series": primary_series,
        "terminal_type": "primary",
        "terminal_formula": None,
        "provider": "MKONLINE",
        "source": "WATTSIGHT",
        "country": str(country),
        "storm_token_found": False,
        "dependency_gate_passed": True,
    }


def _expected_cutoff(
    index: pd.DatetimeIndex, *, cutoff_timezone: str
) -> pd.DatetimeIndex:
    """Civil D-1 08:00 cutoffs in the materialisation's own timezone."""

    delivery_days = pd.DatetimeIndex(index.tz_convert(cutoff_timezone).date)
    civil = delivery_days - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
    return civil.tz_localize(
        cutoff_timezone, ambiguous="raise", nonexistent="raise"
    ).tz_convert("UTC")


def _read_primary_window(path: Path, expected: pd.DatetimeIndex) -> pd.DataFrame:
    """Predicate-filter a Parquet so later (including final) values are not read."""

    if expected.empty:
        raise ValueError("empty expected primary window")
    columns = [
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
        "value",
    ]
    try:
        frame = pd.read_parquet(
            path,
            columns=columns,
            filters=[
                ("value_time_utc", ">=", expected[0].to_pydatetime()),
                ("value_time_utc", "<=", expected[-1].to_pydatetime()),
            ],
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{path}: invalid primary PIT schema: {exc}") from exc
    return frame.sort_values("value_time_utc", kind="mergesort").reset_index(drop=True)


def _load_primary(
    path: Path,
    expected: pd.DatetimeIndex,
    *,
    block: str,
    delivery_timezone: str,
    cutoff_timezone: str,
    expected_local_days: int,
) -> tuple[pd.Series, dict[str, Any]]:
    _reject_storm_values([path.name], context=f"{block} primary path")
    frame = _read_primary_window(path, expected)
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
    if not delivery.equals(expected):
        raise ValueError(f"{path}: {block} delivery timeline is not exact")
    expected_markers = _expected_cutoff(
        delivery, cutoff_timezone=cutoff_timezone
    )
    snapshot = pd.DatetimeIndex(
        pd.to_datetime(frame["snapshot_time_utc"], utc=True, errors="raise")
    )
    revision = pd.DatetimeIndex(
        pd.to_datetime(frame["revision_time_utc"], utc=True, errors="raise")
    )
    snapshot_violations = int(
        np.count_nonzero(snapshot.asi8 != expected_markers.asi8)
    )
    revision_violations = int(
        np.count_nonzero(revision.asi8 != expected_markers.asi8)
    )
    if snapshot_violations or revision_violations:
        raise ValueError(
            f"{path}: {block} markers are not civil D-1 {ORIGIN_CLOCK} "
            f"in {cutoff_timezone}"
        )
    values = pd.Series(
        pd.to_numeric(frame["value"], errors="coerce").to_numpy(float),
        index=delivery,
        name="mkonline_primary",
    )
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError(f"{path}: non-finite primary values")
    local_days = pd.Index(delivery.tz_convert(delivery_timezone).date)
    unique_days = local_days.unique()
    if len(unique_days) != expected_local_days:
        raise ValueError(
            f"{path}: {block} has {len(unique_days)} delivery days, "
            f"expected {expected_local_days}"
        )
    counts = pd.Series(1, index=local_days).groupby(level=0).sum()
    return values, {
        "path": str(path),
        "block": block,
        "rows_loaded": int(len(frame)),
        "delivery_local_days": int(len(unique_days)),
        "delivery_timezone": delivery_timezone,
        "cutoff_timezone": cutoff_timezone,
        "cutoff_contract": f"civil D-1 {ORIGIN_CLOCK} {cutoff_timezone}",
        "snapshot_cutoff_violations": snapshot_violations,
        "revision_cutoff_violations": revision_violations,
        "physical_day_hour_counts": {
            str(int(hours)): int(count)
            for hours, count in counts.value_counts().sort_index().items()
        },
        "coverage": 1.0,
        "final_rows_loaded": 0,
    }


def _cat_v1(threads: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=700,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=15.0,
        random_seed=42,
        thread_count=threads,
        allow_writing_files=False,
        verbose=False,
        has_time=True,
        nan_mode="Min",
    )


def _hgb31() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.035,
        max_iter=550,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=60,
        early_stopping=False,
        random_state=42,
    )


def _fit_models(
    X: pd.DataFrame, residual: pd.Series, *, threads: int
) -> dict[str, Any]:
    cat = _cat_v1(threads).fit(X, residual)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    Xi = imputer.fit_transform(X)
    hgb = _hgb31().fit(Xi, residual)
    return {"cat_v1": cat, "hgb31": hgb, "imputer": imputer}


def _predict(models: Mapping[str, Any], X: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "cat_v1": np.asarray(models["cat_v1"].predict(X), dtype=float),
        "hgb31": np.asarray(
            models["hgb31"].predict(models["imputer"].transform(X)), dtype=float
        ),
    }


def _autonomous_prediction(
    models: Mapping[str, Any], meta: pd.DataFrame
) -> np.ndarray:
    prediction = _predict(models, meta)
    correction = np.clip(
        0.5 * prediction["cat_v1"] + 0.5 * prediction["hgb31"], -40.0, 40.0
    )
    return meta["base__q50"].to_numpy(float) + correction


def _fit_frame(
    external: Mapping[str, Any], existing: Mapping[str, Any], mask: np.ndarray
) -> tuple[pd.DataFrame, pd.Series]:
    if tuple(external["meta"].columns) != tuple(existing["meta"].columns):
        raise RuntimeError("EXT and autonomous A use different meta-feature schemas")
    X = pd.concat([external["meta"], existing["meta"].loc[mask]], axis=0)
    y = pd.concat([external["residual"], existing["residual"].loc[mask]], axis=0)
    if X.index.has_duplicates or not X.index.is_monotonic_increasing:
        raise RuntimeError("EXT+A fit frame is not unique and chronological")
    if not y.index.equals(X.index):
        raise RuntimeError("EXT+A residuals are not aligned to meta-features")
    return X, y


def _exact_l1_weight(
    actual: np.ndarray, autonomous: np.ndarray, expert: np.ndarray
) -> float:
    """Return the exact constrained L1 blend weight by weighted median."""

    actual = np.asarray(actual, dtype=float)
    autonomous = np.asarray(autonomous, dtype=float)
    expert = np.asarray(expert, dtype=float)
    if not (actual.shape == autonomous.shape == expert.shape):
        raise ValueError("actual/autonomous/expert arrays must have the same shape")
    if not (
        np.isfinite(actual).all()
        and np.isfinite(autonomous).all()
        and np.isfinite(expert).all()
    ):
        raise ValueError("exact L1 inputs must be finite")
    delta = expert - autonomous
    active = np.abs(delta) > 1e-12
    if not bool(active.any()):
        return 0.0
    ratios = (actual[active] - autonomous[active]) / delta[active]
    weights = np.abs(delta[active])
    order = np.argsort(ratios, kind="mergesort")
    ratios = ratios[order]
    weights = weights[order]
    position = int(
        np.searchsorted(np.cumsum(weights), 0.5 * weights.sum(), side="left")
    )
    return float(np.clip(ratios[position], 0.0, 1.0))


def _mae(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - prediction)))


def _score_block(
    index: pd.DatetimeIndex,
    actual: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    timezone: str,
) -> dict[str, Any]:
    local_days = pd.Index(index.tz_convert(timezone).date)
    days = local_days.unique()
    if len(days) != 60:
        raise ValueError(f"score block must contain 60 local days, got {len(days)}")
    masks = {
        "all": np.ones(len(index), dtype=bool),
        "first30": np.asarray(local_days.isin(days[:30]), dtype=bool),
        "last30": np.asarray(local_days.isin(days[30:]), dtype=bool),
    }
    result: dict[str, Any] = {}
    for name, mask in masks.items():
        baseline_mae = _mae(actual[mask], baseline[mask])
        candidate_mae = _mae(actual[mask], candidate[mask])
        result[name] = {
            "baseline_mae": baseline_mae,
            "candidate_mae": candidate_mae,
            "gain": baseline_mae - candidate_mae,
        }
    result["passes"] = bool(
        result["all"]["gain"] >= MINIMUM_BLOCK_GAIN
        and result["first30"]["gain"] > 0.0
        and result["last30"]["gain"] > 0.0
    )
    return result


def _atomic_emit(payload: Mapping[str, Any], output: Path) -> None:
    rendered = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(rendered)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(output)
    print(rendered)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict zone-aware MKOnline A/B1/B2 cross-fit calibration"
    )
    parser.add_argument("--zone", required=True)
    parser.add_argument("--timezone", required=True)
    parser.add_argument("--cutoff-timezone", default="Europe/Paris")
    parser.add_argument("--autonomous-run", required=True)
    parser.add_argument("--extended-oof-file", required=True)
    parser.add_argument("--primary-series", required=True)
    parser.add_argument("--primary-file", required=True)
    parser.add_argument("--dependency-manifest", required=True)
    parser.add_argument("--phase", choices=("b1", "b2"), required=True)
    parser.add_argument("--b2-primary-file")
    parser.add_argument("--frozen-weight", type=float)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--threads", type=int, default=-1)
    args = parser.parse_args(argv)
    args.zone = str(args.zone).upper()
    if args.zone not in SUPPORTED_ZONES:
        parser.error(f"--zone must be one of {sorted(SUPPORTED_ZONES)}")
    if args.phase == "b2":
        if not args.b2_primary_file:
            parser.error("B2 requires --b2-primary-file")
        if args.frozen_weight is None:
            parser.error("B2 requires --frozen-weight learned before opening B2")
        if not (0.0 <= float(args.frozen_weight) <= 1.0):
            parser.error("--frozen-weight must be in [0, 1]")
    elif args.frozen_weight is not None:
        parser.error("B1 learns the weight; do not pass --frozen-weight")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    zone = args.zone
    run_dir = Path(args.autonomous_run).expanduser().resolve()
    extended_path = Path(args.extended_oof_file).expanduser().resolve()
    primary_path = Path(args.primary_file).expanduser().resolve()
    dependency_path = Path(args.dependency_manifest).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()

    run_audit = _audit_autonomous_run(
        run_dir, zone=zone, timezone=args.timezone
    )
    dependency = _audit_primary_dependency(
        dependency_path, primary_series=args.primary_series, zone=zone
    )
    contract = _calibration_contract(
        run_dir, zone=zone, timezone=args.timezone
    )
    end = contract.b1_end_utc if args.phase == "b1" else contract.b2_end_utc
    X_all = _load_features_to(
        run_dir, end_exclusive_utc=end, timezone=args.timezone
    )
    external = _load_external(extended_path, X_all, contract=contract)
    current = _load_existing(
        run_dir, X_all, contract=contract, phase=args.phase
    )
    days = current["days"]
    local_days = current["local_days"]
    mask_a = np.asarray(local_days.isin(days[:A_DAYS]), dtype=bool)

    common_protocol = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "zone": zone,
        "delivery_timezone": args.timezone,
        "primary_cutoff_timezone": args.cutoff_timezone,
        "primary_cutoff_contract": (
            f"civil D-1 {ORIGIN_CLOCK} {args.cutoff_timezone}"
        ),
        "fit_EXT_days": EXT_DAYS,
        "fit_A_days": A_DAYS,
        "gate": (
            "overall MAE gain >= 0.75 EUR/MWh and both chronological "
            "30-day half gains > 0"
        ),
        "autonomous_recipe": (
            "EXT+prior-A CatBoost-v1/HistGBR31 residual blend 50/50, clip 40"
        ),
        "storm_used_as_feature": False,
        "final_phase_exposed": False,
        "final_loaded": False,
    }

    if args.phase == "b2":
        frozen = float(args.frozen_weight)
        mask_b2 = np.asarray(
            local_days.isin(days[A_DAYS + B1_DAYS : A_DAYS + B1_DAYS + B2_DAYS]),
            dtype=bool,
        )
        fit_X, fit_y = _fit_frame(external, current, mask_a)
        models = _fit_models(fit_X, fit_y, threads=args.threads)
        index = current["meta"].index[mask_b2]
        autonomous = _autonomous_prediction(models, current["meta"].loc[index])
        primary, materialization = _load_primary(
            Path(args.b2_primary_file).expanduser().resolve(),
            index,
            block="B2",
            delivery_timezone=args.timezone,
            cutoff_timezone=args.cutoff_timezone,
            expected_local_days=B2_DAYS,
        )
        actual = current["raw"].loc[index, "actual"].to_numpy(float)
        candidate = (1.0 - frozen) * autonomous + frozen * primary.to_numpy(float)
        result = {
            "protocol": {
                **common_protocol,
                "phase": "b2",
                "veto_B2": [str(days[A_DAYS + B1_DAYS]), str(days[-1])],
                "frozen_mkonline_weight": frozen,
                "weight_relearned_or_tuned_on_B2": False,
                "B1_used_for_fit_weight_or_scoring": False,
                "B1_primary_loaded": False,
            },
            "autonomous_run_audit": run_audit,
            "dependency_audit": dependency,
            "materialization_audit": materialization,
            "B2": _score_block(
                index,
                actual,
                autonomous,
                candidate,
                timezone=args.timezone,
            ),
        }
        _atomic_emit(result, output_path)
        return 0 if result["B2"]["passes"] else 2

    mask_b1 = np.asarray(
        local_days.isin(days[A_DAYS : A_DAYS + B1_DAYS]), dtype=bool
    )
    full_index = external["meta"].index.append(current["meta"].index)
    if full_index.has_duplicates or not full_index.is_monotonic_increasing:
        raise RuntimeError("EXT+A+B1 index is not unique and chronological")
    primary, materialization = _load_primary(
        primary_path,
        full_index,
        block="EXT+A+B1",
        delivery_timezone=args.timezone,
        cutoff_timezone=args.cutoff_timezone,
        expected_local_days=EXT_DAYS + A_DAYS + B1_DAYS,
    )

    a_index = current["meta"].index[mask_a]
    a_actual = current["raw"].loc[a_index, "actual"].to_numpy(float)
    a_auto_oof = np.full(len(a_index), np.nan, dtype=float)
    fold_audits: list[dict[str, Any]] = []
    for fold in range(CROSSFIT_FOLDS):
        start_day = fold * CROSSFIT_DAYS
        end_day = start_day + CROSSFIT_DAYS
        validation_days = days[start_day:end_day]
        validation_mask = np.asarray(local_days.isin(validation_days), dtype=bool)
        prior_mask = np.asarray(local_days.isin(days[:start_day]), dtype=bool)
        fit_X, fit_y = _fit_frame(external, current, prior_mask)
        models = _fit_models(fit_X, fit_y, threads=args.threads)
        validation_index = current["meta"].index[validation_mask]
        prediction = _autonomous_prediction(
            models, current["meta"].loc[validation_index]
        )
        positions = a_index.get_indexer(validation_index)
        if bool((positions < 0).any()):
            raise RuntimeError("cross-fit validation fold lies outside A")
        a_auto_oof[positions] = prediction
        fold_audits.append(
            {
                "fold": fold + 1,
                "fit_EXT_days": EXT_DAYS,
                "fit_prior_A_days": start_day,
                "validation_A_days": CROSSFIT_DAYS,
                "validation_range": [
                    str(validation_days[0]),
                    str(validation_days[-1]),
                ],
                "validation_hours": int(len(validation_index)),
            }
        )
    if not np.isfinite(a_auto_oof).all():
        raise RuntimeError("cross-fit did not cover every A hour")
    a_primary = primary.loc[a_index].to_numpy(float)
    weight = _exact_l1_weight(a_actual, a_auto_oof, a_primary)
    a_candidate = (1.0 - weight) * a_auto_oof + weight * a_primary

    fit_X, fit_y = _fit_frame(external, current, mask_a)
    models = _fit_models(fit_X, fit_y, threads=args.threads)
    b1_index = current["meta"].index[mask_b1]
    b1_autonomous = _autonomous_prediction(models, current["meta"].loc[b1_index])
    b1_primary = primary.loc[b1_index].to_numpy(float)
    b1_candidate = (1.0 - weight) * b1_autonomous + weight * b1_primary
    b1_actual = current["raw"].loc[b1_index, "actual"].to_numpy(float)
    result = {
        "protocol": {
            **common_protocol,
            "phase": "b1",
            "read_range": [str(contract.ext_start_day.date()), str(days[-1])],
            "crossfit_A_days": A_DAYS,
            "crossfit_folds": (
                f"{CROSSFIT_FOLDS} expanding folds x {CROSSFIT_DAYS} days"
            ),
            "selection_B1": [str(days[A_DAYS]), str(days[-1])],
            "weight_learning": (
                "exact constrained L1 weighted median on A cross-fit predictions"
            ),
            "B1_used_for_weight_or_hyperparameters": False,
            "B2_loaded": False,
        },
        "autonomous_run_audit": run_audit,
        "dependency_audit": dependency,
        "materialization_audit": materialization,
        "crossfit_folds": fold_audits,
        "learned_mkonline_weight": weight,
        "learned_autonomous_weight": 1.0 - weight,
        "A_crossfit": {
            "autonomous_mae": _mae(a_actual, a_auto_oof),
            "mkonline_mae": _mae(a_actual, a_primary),
            "frozen_blend_mae": _mae(a_actual, a_candidate),
        },
        "B1": _score_block(
            b1_index,
            b1_actual,
            b1_autonomous,
            b1_candidate,
            timezone=args.timezone,
        ),
    }
    _atomic_emit(result, output_path)
    return 0 if result["B1"]["passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
