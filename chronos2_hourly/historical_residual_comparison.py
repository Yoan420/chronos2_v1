"""Publish a sealed, like-for-like historical residual-load comparison.

The inputs are two independently recalculated hourly runs for the same zone:
one using the historical Saturn residual-load forecasts and one using a
Chronos-2 point-in-time replay.  This module never borrows the control run's
Statistics history.  It verifies the paired experiment, recomputes the final
365-day metrics, and renders the ordinary hourly HTML report from a combined
artifact set.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .chronos_residual_load import RESIDUAL_LOAD_TREATMENT_COLUMNS
from .reporting import write_hourly_html_report


FINAL_DAYS = 365
FINAL_HOURS = 8760
QUANTILES: tuple[str, ...] = ("q10", "q50", "q90")
CONTROL_MODEL = "saturn_residual_load"
CHALLENGER_MODEL = "chronos_residual_load"
CONTROL_SOURCE = "saturn"
CHALLENGER_SOURCE = "chronos2_historical_replay"
SCHEMA_VERSION = 1


class HistoricalResidualComparisonError(RuntimeError):
    """Raised when the two recalculated runs are not strictly comparable."""


@dataclass(frozen=True)
class HistoricalResidualComparisonResult:
    """Paths and headline scores from one immutable comparison publication."""

    output_dir: Path
    report_path: Path
    metrics_path: Path
    control_mae: float
    challenger_mae: float
    final_start_local_date: str
    final_end_local_date: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HistoricalResidualComparisonError(f"JSON absent: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoricalResidualComparisonError(f"JSON invalide: {path}") from exc
    if not isinstance(payload, dict):
        raise HistoricalResidualComparisonError(
            f"{path}: le JSON doit contenir un objet."
        )
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise HistoricalResidualComparisonError(f"Artefact absent: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _checksum_entries(run_dir: Path) -> list[Mapping[str, Any]]:
    payload = _read_json(run_dir / "artifact_checksums.json")
    if str(payload.get("algorithm", "")).lower() != "sha256":
        raise HistoricalResidualComparisonError(
            f"{run_dir}: artifact_checksums.json doit utiliser sha256."
        )
    entries = payload.get("artifacts")
    if not isinstance(entries, list):
        raise HistoricalResidualComparisonError(
            f"{run_dir}: liste artifacts absente du manifeste de checksums."
        )
    return [entry for entry in entries if isinstance(entry, Mapping)]


def _entry_targets(entry: Mapping[str, Any], run_dir: Path) -> tuple[Path, ...]:
    raw = entry.get("path")
    if raw in (None, ""):
        return ()
    declared = Path(str(raw)).expanduser()
    if declared.is_absolute():
        return (declared.resolve(),)
    return ((run_dir / declared).resolve(),)


def _verify_sealed_file(
    path: Path,
    *,
    run_dir: Path,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    resolved = path.resolve()
    matches = [
        entry
        for entry in entries
        if resolved in _entry_targets(entry, run_dir)
    ]
    if len(matches) != 1:
        raise HistoricalResidualComparisonError(
            f"{run_dir}: declaration checksum absente ou ambigue pour {path}."
        )
    entry = matches[0]
    declared_hash = str(entry.get("sha256", "")).lower()
    actual_hash = _sha256(resolved)
    if declared_hash != actual_hash:
        raise HistoricalResidualComparisonError(
            f"{path}: checksum divergent ({actual_hash} != {declared_hash})."
        )
    declared_size = entry.get("size_bytes")
    if declared_size is not None and int(declared_size) != resolved.stat().st_size:
        raise HistoricalResidualComparisonError(f"{path}: taille scellee divergente.")
    return {
        "path": str(resolved),
        "role": str(entry.get("role", "")),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": actual_hash,
    }


def _verify_run_seal(run_dir: Path, *, zone: str) -> dict[str, Any]:
    entries = _checksum_entries(run_dir)
    required = (
        run_dir / "backtest_hourly_oof.csv.gz",
        run_dir / f"forecast_hourly_{zone.lower()}.csv",
        run_dir / "run_manifest.json",
        run_dir / "inputs" / "aligned_inputs.csv.gz",
        run_dir / "inputs" / "model_covariates_with_future.csv.gz",
        run_dir / "inputs" / "input_manifest.csv",
        run_dir / "inputs" / "input_coverage.csv",
    )
    verified: dict[str, Any] = {}
    for path in required:
        verified[path.relative_to(run_dir).as_posix()] = _verify_sealed_file(
            path,
            run_dir=run_dir,
            entries=entries,
        )

    inputs_dir = run_dir / "inputs"
    for path in sorted(inputs_dir.rglob("*")):
        if path.is_symlink():
            raise HistoricalResidualComparisonError(
                f"Lien symbolique interdit dans les inputs scelles: {path}"
            )
        if path.is_file():
            relative = path.relative_to(run_dir).as_posix()
            if relative not in verified:
                verified[relative] = _verify_sealed_file(
                    path,
                    run_dir=run_dir,
                    entries=entries,
                )
    return {
        "checksum_manifest": str((run_dir / "artifact_checksums.json").resolve()),
        "checksum_manifest_sha256": _sha256(run_dir / "artifact_checksums.json"),
        "verified_artifacts": verified,
    }


def _utc_index(frame: pd.DataFrame, *, path: Path) -> pd.DatetimeIndex:
    if "delivery_start_utc" not in frame:
        raise HistoricalResidualComparisonError(
            f"{path}: delivery_start_utc est absent."
        )
    try:
        index = pd.DatetimeIndex(
            pd.to_datetime(frame["delivery_start_utc"], utc=True, errors="raise"),
            name="delivery_start_utc",
        )
    except Exception as exc:
        raise HistoricalResidualComparisonError(
            f"{path}: timeline de livraison invalide."
        ) from exc
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise HistoricalResidualComparisonError(
            f"{path}: timeline dupliquee ou non triee."
        )
    return index


def _timestamp_index(
    frame: pd.DataFrame,
    *,
    path: Path,
    column: str = "timestamp",
) -> pd.DatetimeIndex:
    if column not in frame:
        raise HistoricalResidualComparisonError(f"{path}: {column} est absent.")
    try:
        index = pd.DatetimeIndex(
            pd.to_datetime(frame[column], utc=True, errors="raise"),
            name=column,
        )
    except Exception as exc:
        raise HistoricalResidualComparisonError(
            f"{path}: colonne {column} invalide."
        ) from exc
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise HistoricalResidualComparisonError(
            f"{path}: timeline {column} dupliquee ou non triee."
        )
    return index


def _numeric_equal(
    left: pd.Series,
    right: pd.Series,
    *,
    name: str,
    atol: float = 0.0,
) -> float:
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    if left_values.shape != right_values.shape:
        raise HistoricalResidualComparisonError(f"{name}: tailles differentes.")
    finite_left = np.isfinite(left_values)
    finite_right = np.isfinite(right_values)
    if not np.array_equal(finite_left, finite_right):
        raise HistoricalResidualComparisonError(
            f"{name}: masques de disponibilite differents."
        )
    maximum = 0.0
    if finite_left.any():
        differences = np.abs(left_values[finite_left] - right_values[finite_left])
        maximum = float(differences.max(initial=0.0))
        if maximum > float(atol):
            raise HistoricalResidualComparisonError(
                f"{name}: valeurs differentes, ecart maximal={maximum:.9g}."
            )
    return maximum


def _series_equal(left: pd.Series, right: pd.Series, *, name: str) -> None:
    if len(left) != len(right):
        raise HistoricalResidualComparisonError(f"{name}: tailles differentes.")
    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    left_non_missing = left.notna().to_numpy()
    right_non_missing = right.notna().to_numpy()
    numeric_usable = bool(
        ((~left_non_missing) | left_numeric.notna().to_numpy()).all()
        and ((~right_non_missing) | right_numeric.notna().to_numpy()).all()
    )
    if numeric_usable:
        _numeric_equal(left_numeric, right_numeric, name=name)
        return
    left_text = left.astype("string").fillna("<NA>").to_numpy(dtype=str)
    right_text = right.astype("string").fillna("<NA>").to_numpy(dtype=str)
    if not np.array_equal(left_text, right_text):
        raise HistoricalResidualComparisonError(f"{name}: valeurs differentes.")


def _compare_non_treatment_inputs(
    control_run: Path,
    challenger_run: Path,
) -> dict[str, Any]:
    audits: dict[str, Any] = {}
    model_treatment_differences = 0
    for filename in (
        "aligned_inputs.csv.gz",
        "model_covariates_with_future.csv.gz",
    ):
        control_path = control_run / "inputs" / filename
        challenger_path = challenger_run / "inputs" / filename
        control = _read_frame(control_path)
        challenger = _read_frame(challenger_path)
        control_index = _timestamp_index(control, path=control_path)
        challenger_index = _timestamp_index(challenger, path=challenger_path)
        if not control_index.equals(challenger_index):
            raise HistoricalResidualComparisonError(
                f"{filename}: timelines control/challenger differentes."
            )

        control_columns = set(control.columns) - {"timestamp"}
        challenger_columns = set(challenger.columns) - {"timestamp"}
        if control_columns != challenger_columns:
            raise HistoricalResidualComparisonError(
                f"{filename}: schemas control/challenger differents."
            )
        present_treatment = sorted(
            control_columns.intersection(RESIDUAL_LOAD_TREATMENT_COLUMNS)
        )
        treatment_differences = 0
        for column in present_treatment:
            control_values = pd.to_numeric(
                control[column], errors="coerce"
            ).to_numpy(dtype=float)
            challenger_values = pd.to_numeric(
                challenger[column], errors="coerce"
            ).to_numpy(dtype=float)
            control_finite = np.isfinite(control_values)
            challenger_finite = np.isfinite(challenger_values)
            if not np.array_equal(control_finite, challenger_finite):
                raise HistoricalResidualComparisonError(
                    f"{filename}/{column}: masque de disponibilite du "
                    "traitement different."
                )
            treatment_differences += int(
                np.sum(
                    control_finite
                    & (np.abs(control_values - challenger_values) > 1e-12)
                )
            )
        non_treatment = sorted(control_columns - set(present_treatment))
        for column in non_treatment:
            _series_equal(
                control[column],
                challenger[column],
                name=f"{filename}/{column}",
            )

        if filename == "model_covariates_with_future.csv.gz":
            missing_treatment = sorted(
                set(RESIDUAL_LOAD_TREATMENT_COLUMNS) - control_columns
            )
            if missing_treatment:
                raise HistoricalResidualComparisonError(
                    "Features de traitement absentes du contexte modele: "
                    + ", ".join(missing_treatment)
                )
            for alias in RESIDUAL_LOAD_TREATMENT_COLUMNS[:5]:
                mirror = f"known_{alias}_oracle"
                _numeric_equal(
                    control[alias],
                    control[mirror],
                    name=f"{filename}/control/{alias}_mirror",
                )
                _numeric_equal(
                    challenger[alias],
                    challenger[mirror],
                    name=f"{filename}/challenger/{alias}_mirror",
                )
            model_treatment_differences = treatment_differences
        audits[filename] = {
            "rows": int(len(control)),
            "first_timestamp_utc": str(control_index[0]),
            "last_timestamp_utc": str(control_index[-1]),
            "treatment_columns": present_treatment,
            "non_treatment_columns": non_treatment,
            "non_treatment_values_identical": True,
            "treatment_availability_identical": True,
            "treatment_value_differences": treatment_differences,
        }
    if model_treatment_differences == 0:
        raise HistoricalResidualComparisonError(
            "Le challenger ne modifie aucune valeur de residual_load; "
            "la comparaison serait vide."
        )
    return audits


def _manifest_source(
    run_dir: Path,
    *,
    expected_zone: str,
    expected_source: str,
) -> tuple[dict[str, Any], str, str]:
    manifest = _read_json(run_dir / "run_manifest.json")
    zone = str(manifest.get("zone", "")).strip().upper()
    if zone != expected_zone:
        raise HistoricalResidualComparisonError(
            f"{run_dir}: zone={zone!r}, attendu {expected_zone!r}."
        )
    source = str(manifest.get("residual_load_source", "")).strip().lower()
    if source != expected_source:
        raise HistoricalResidualComparisonError(
            f"{run_dir}: residual_load_source={source!r}, "
            f"attendu {expected_source!r}."
        )
    timezone = str(manifest.get("timezone", "")).strip()
    if not timezone:
        raise HistoricalResidualComparisonError(f"{run_dir}: timezone absente.")
    native_model = str(manifest.get("native_model", "residual_corrected")).strip()
    if not native_model:
        raise HistoricalResidualComparisonError(f"{run_dir}: native_model absent.")
    return manifest, timezone, native_model


def _fold_series(frame: pd.DataFrame, *, path: Path) -> pd.Series:
    if "fold_id" not in frame:
        raise HistoricalResidualComparisonError(f"{path}: fold_id est absent.")
    numeric = pd.to_numeric(frame["fold_id"], errors="coerce")
    invalid = frame["fold_id"].notna() & numeric.isna()
    if bool(invalid.any()):
        raise HistoricalResidualComparisonError(f"{path}: fold_id invalide.")
    return numeric.astype("Int64")


def _final_index(
    shared_index: pd.DatetimeIndex,
    *,
    timezone: str,
) -> tuple[np.ndarray, pd.DatetimeIndex, date, date]:
    local_dates = pd.Index(shared_index.tz_convert(timezone).date)
    unique_dates = local_dates.unique().tolist()
    if len(unique_dates) < FINAL_DAYS:
        raise HistoricalResidualComparisonError(
            f"Historique insuffisant: {len(unique_dates)} jours, {FINAL_DAYS} requis."
        )
    final_dates = unique_dates[-FINAL_DAYS:]
    expected_dates = pd.date_range(
        pd.Timestamp(final_dates[0]), periods=FINAL_DAYS, freq="D"
    ).date.tolist()
    if final_dates != expected_dates:
        raise HistoricalResidualComparisonError(
            "FINAL365 ne contient pas 365 jours civils consecutifs."
        )
    start_local = pd.Timestamp(final_dates[0]).tz_localize(timezone)
    end_local = pd.Timestamp(final_dates[-1] + pd.Timedelta(days=1)).tz_localize(
        timezone
    )
    expected = pd.date_range(
        start_local,
        end_local,
        inclusive="left",
        freq="h",
    ).tz_convert("UTC")
    mask = np.asarray(local_dates.isin(final_dates), dtype=bool)
    observed = shared_index[mask]
    if len(expected) != FINAL_HOURS or len(observed) != FINAL_HOURS:
        raise HistoricalResidualComparisonError(
            f"FINAL365 doit contenir {FINAL_HOURS} heures physiques; "
            f"attendu={len(expected)}, observe={len(observed)}."
        )
    if not observed.equals(pd.DatetimeIndex(expected, name=shared_index.name)):
        raise HistoricalResidualComparisonError(
            "FINAL365 ne correspond pas a la timeline physique locale exacte."
        )
    return mask, observed, final_dates[0], final_dates[-1]


def _validate_forecast_day(
    index: pd.DatetimeIndex,
    *,
    timezone: str,
) -> None:
    local_dates = pd.Index(index.tz_convert(timezone).date).unique().tolist()
    if len(local_dates) != 1:
        raise HistoricalResidualComparisonError(
            "Le forecast combine doit couvrir un unique jour civil local."
        )
    start_local = pd.Timestamp(local_dates[0]).tz_localize(timezone)
    end_local = pd.Timestamp(local_dates[0] + pd.Timedelta(days=1)).tz_localize(
        timezone
    )
    expected = pd.date_range(
        start_local,
        end_local,
        inclusive="left",
        freq="h",
    ).tz_convert("UTC")
    if len(expected) not in (23, 24, 25) or not index.equals(
        pd.DatetimeIndex(expected, name=index.name)
    ):
        raise HistoricalResidualComparisonError(
            "Le forecast ne respecte pas la timeline physique 23/24/25 heures."
        )


def _source_quantiles(
    frame: pd.DataFrame,
    *,
    model: str,
    allow_final_columns: bool,
    label: str,
) -> pd.DataFrame:
    qualified = {quantile: f"{model}__{quantile}" for quantile in QUANTILES}
    if all(column in frame for column in qualified.values()):
        columns = qualified
    elif allow_final_columns and all(quantile in frame for quantile in QUANTILES):
        columns = {quantile: quantile for quantile in QUANTILES}
    else:
        raise HistoricalResidualComparisonError(
            f"{label}: quantiles absents pour le modele {model!r}."
        )
    result = pd.DataFrame(
        {
            quantile: pd.to_numeric(frame[column], errors="coerce")
            for quantile, column in columns.items()
        },
        index=frame.index,
    )
    finite = np.isfinite(result.to_numpy(dtype=float)).all(axis=1)
    if finite.any():
        values = result.loc[finite, list(QUANTILES)].to_numpy(dtype=float)
        if bool((values[:, 0] > values[:, 1]).any()) or bool(
            (values[:, 1] > values[:, 2]).any()
        ):
            raise HistoricalResidualComparisonError(
                f"{label}: quantiles croises."
            )
    return result


def _origin_series(frame: pd.DataFrame, *, model: str) -> pd.Series:
    qualified = f"{model}_forecast_origin_utc"
    column = qualified if qualified in frame else "forecast_origin_utc"
    if column not in frame:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    return pd.Series(
        pd.to_datetime(frame[column], utc=True, errors="coerce").to_numpy(),
        index=frame.index,
        dtype="datetime64[ns, UTC]",
    )


def _validate_paired_origins(
    control: pd.Series,
    challenger: pd.Series,
    delivery: pd.DatetimeIndex,
    *,
    required_mask: np.ndarray,
    label: str,
) -> None:
    mask = np.asarray(required_mask, dtype=bool)
    if mask.shape != (len(delivery),):
        raise HistoricalResidualComparisonError(
            f"{label}: masque d'origine incoherent."
        )
    control_values = pd.DatetimeIndex(control.iloc[mask])
    challenger_values = pd.DatetimeIndex(challenger.iloc[mask])
    if control_values.isna().any() or challenger_values.isna().any():
        raise HistoricalResidualComparisonError(
            f"{label}: origine manquante sur la fenetre requise."
        )
    if not control_values.equals(challenger_values):
        raise HistoricalResidualComparisonError(
            f"{label}: origines control/challenger differentes."
        )
    delivery_values = delivery[mask]
    if bool((control_values >= delivery_values).any()):
        raise HistoricalResidualComparisonError(
            f"{label}: origine non causale detectee."
        )


def _metrics_row(
    *,
    model: str,
    actual: np.ndarray,
    quantiles: pd.DataFrame,
) -> dict[str, Any]:
    values = quantiles.loc[:, list(QUANTILES)].to_numpy(dtype=float)
    actual_finite = np.isfinite(actual)
    prediction_finite = np.isfinite(values).all(axis=1)
    scored = actual_finite & prediction_finite
    if int(scored.sum()) != FINAL_HOURS:
        raise HistoricalResidualComparisonError(
            f"{model}: FINAL365 incomplet, {int(scored.sum())}/{FINAL_HOURS} heures."
        )
    selected_actual = actual[scored]
    selected = values[scored]
    error = selected[:, 1] - selected_actual

    def pinball(level: float, prediction: np.ndarray) -> float:
        residual = selected_actual - prediction
        return float(
            np.mean(np.maximum(level * residual, (level - 1.0) * residual))
        )

    denominator = np.abs(selected[:, 1]) + np.abs(selected_actual)
    smape = np.divide(
        2.0 * np.abs(error),
        denominator,
        out=np.zeros_like(error),
        where=denominator > 0,
    )
    return {
        "model": model,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "median_absolute_error": float(np.median(np.abs(error))),
        "smape_pct": float(100.0 * np.mean(smape)),
        "coverage_q10_q90": float(
            np.mean((selected_actual >= selected[:, 0]) & (selected_actual <= selected[:, 2]))
        ),
        "interval_width_q10_q90": float(np.mean(selected[:, 2] - selected[:, 0])),
        "pinball_q10": pinball(0.1, selected[:, 0]),
        "pinball_q50": pinball(0.5, selected[:, 1]),
        "pinball_q90": pinball(0.9, selected[:, 2]),
        "n_scored": int(scored.sum()),
        "n_expected": FINAL_HOURS,
        "prediction_coverage": float(prediction_finite.mean()),
        "score_coverage": float(scored.sum() / actual_finite.sum()),
    }


def _write_csv_gzip(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )


def _copy_challenger_inputs(challenger_run: Path, staging: Path) -> None:
    source = challenger_run / "inputs"
    destination = staging / "inputs"
    forbidden = [
        path
        for path in source.rglob("*")
        if path.is_file() and path.name.startswith("statistics_history")
    ]
    if forbidden:
        raise HistoricalResidualComparisonError(
            "Un artefact statistics_history est interdit dans les inputs "
            f"challenger: {forbidden[0]}"
        )
    if destination.exists():  # defensive: staging is expected to be empty
        raise HistoricalResidualComparisonError(
            f"Destination inputs inattendue: {destination}"
        )
    shutil.copytree(source, destination)


def _output_checksums(staging: Path, *, final_output: Path) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(staging.rglob("*")):
        if not path.is_file() or path.name == "artifact_checksums.json":
            continue
        relative = path.relative_to(staging).as_posix()
        artifacts.append(
            {
                "path": relative,
                "role": (
                    "materialized_input" if relative.startswith("inputs/") else "run_artifact"
                ),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    return {
        "algorithm": "sha256",
        "schema_version": SCHEMA_VERSION,
        "output_directory": str(final_output),
        "artifacts": artifacts,
    }


def publish_historical_residual_comparison(
    control_run: str | Path,
    challenger_run: str | Path,
    output_dir: str | Path,
    *,
    zone: str,
    title: str | None = None,
    extreme_threshold: float = 150.0,
    history_hours: int = 168,
) -> HistoricalResidualComparisonResult:
    """Validate and atomically publish one zone's historical comparison.

    ``output_dir`` must not exist.  Both source runs remain read-only and every
    source input copied into the publication comes from the challenger run.
    """

    control = Path(control_run).expanduser().resolve()
    challenger = Path(challenger_run).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    zone_code = str(zone).strip().upper()
    if not zone_code:
        raise ValueError("zone ne peut pas etre vide.")
    if control == challenger:
        raise HistoricalResidualComparisonError(
            "Les runs control et challenger doivent etre distincts."
        )
    if destination.exists():
        raise FileExistsError(
            f"La destination immutable existe deja: {destination}"
        )
    if not control.is_dir() or not challenger.is_dir():
        raise HistoricalResidualComparisonError(
            "Les runs control et challenger doivent etre des dossiers existants."
        )

    _control_manifest, control_timezone, control_native = _manifest_source(
        control,
        expected_zone=zone_code,
        expected_source=CONTROL_SOURCE,
    )
    _challenger_manifest, challenger_timezone, challenger_native = _manifest_source(
        challenger,
        expected_zone=zone_code,
        expected_source=CHALLENGER_SOURCE,
    )
    if control_timezone != challenger_timezone:
        raise HistoricalResidualComparisonError(
            "Les timezones control et challenger different."
        )
    if control_native != challenger_native:
        raise HistoricalResidualComparisonError(
            "Le modele aval natif differe entre control et challenger: "
            f"{control_native!r} != {challenger_native!r}."
        )

    control_seal = _verify_run_seal(control, zone=zone_code)
    challenger_seal = _verify_run_seal(challenger, zone=zone_code)
    input_audit = _compare_non_treatment_inputs(control, challenger)

    control_backtest_path = control / "backtest_hourly_oof.csv.gz"
    challenger_backtest_path = challenger / "backtest_hourly_oof.csv.gz"
    control_backtest = _read_frame(control_backtest_path)
    challenger_backtest = _read_frame(challenger_backtest_path)
    control_index = _utc_index(control_backtest, path=control_backtest_path)
    challenger_index = _utc_index(
        challenger_backtest, path=challenger_backtest_path
    )
    if not control_index.equals(challenger_index):
        raise HistoricalResidualComparisonError(
            "Les timelines backtest control et challenger different."
        )
    if "actual" not in control_backtest or "actual" not in challenger_backtest:
        raise HistoricalResidualComparisonError("actual est absent d'un backtest.")
    actual_max_difference = _numeric_equal(
        control_backtest["actual"],
        challenger_backtest["actual"],
        name="backtest/actual",
        atol=5e-5,
    )
    control_fold = _fold_series(control_backtest, path=control_backtest_path)
    challenger_fold = _fold_series(
        challenger_backtest, path=challenger_backtest_path
    )
    if not control_fold.equals(challenger_fold):
        raise HistoricalResidualComparisonError(
            "Les fold_id control et challenger different."
        )

    final_mask, final_timeline, final_start, final_end = _final_index(
        control_index,
        timezone=control_timezone,
    )
    if bool(control_fold.iloc[final_mask].isna().any()):
        raise HistoricalResidualComparisonError(
            "FINAL365 contient au moins un fold_id manquant."
        )
    control_quantiles = _source_quantiles(
        control_backtest,
        model=control_native,
        allow_final_columns=False,
        label="backtest control",
    )
    challenger_quantiles = _source_quantiles(
        challenger_backtest,
        model=challenger_native,
        allow_final_columns=False,
        label="backtest challenger",
    )
    final_actual = pd.to_numeric(
        challenger_backtest.loc[final_mask, "actual"], errors="coerce"
    ).to_numpy(dtype=float)
    metrics = [
        _metrics_row(
            model=CONTROL_MODEL,
            actual=final_actual,
            quantiles=control_quantiles.loc[final_mask],
        ),
        _metrics_row(
            model=CHALLENGER_MODEL,
            actual=final_actual,
            quantiles=challenger_quantiles.loc[final_mask],
        ),
    ]

    combined_backtest = pd.DataFrame(
        {
            "delivery_start_utc": control_index.astype(str),
            "actual": pd.to_numeric(
                challenger_backtest["actual"], errors="coerce"
            ).to_numpy(dtype=float),
            "fold_id": challenger_fold,
        }
    )
    control_origin = _origin_series(control_backtest, model=control_native)
    challenger_origin = _origin_series(
        challenger_backtest, model=challenger_native
    )
    _validate_paired_origins(
        control_origin,
        challenger_origin,
        control_index,
        required_mask=final_mask,
        label="backtest",
    )
    combined_backtest["forecast_origin_utc"] = challenger_origin.astype(str)
    combined_backtest[f"{CONTROL_MODEL}_forecast_origin_utc"] = (
        control_origin.astype(str)
    )
    combined_backtest[f"{CHALLENGER_MODEL}_forecast_origin_utc"] = (
        challenger_origin.astype(str)
    )
    for quantile in QUANTILES:
        combined_backtest[f"{CONTROL_MODEL}__{quantile}"] = (
            control_quantiles[quantile].to_numpy(dtype=float)
        )
        combined_backtest[f"{CHALLENGER_MODEL}__{quantile}"] = (
            challenger_quantiles[quantile].to_numpy(dtype=float)
        )
        # ``reporting._default_models`` predates experiment-specific model
        # names and performs inference even when explicit names are supplied.
        # These exact aliases keep the standard renderer reusable; all scored
        # and audited columns remain the two semantic prefixes above.
        combined_backtest[f"ensemble__{quantile}"] = control_quantiles[
            quantile
        ].to_numpy(dtype=float)
        combined_backtest[f"residual_corrected__{quantile}"] = (
            challenger_quantiles[quantile].to_numpy(dtype=float)
        )

    control_forecast_path = control / f"forecast_hourly_{zone_code.lower()}.csv"
    challenger_forecast_path = (
        challenger / f"forecast_hourly_{zone_code.lower()}.csv"
    )
    control_forecast = _read_frame(control_forecast_path)
    challenger_forecast = _read_frame(challenger_forecast_path)
    control_forecast_index = _utc_index(control_forecast, path=control_forecast_path)
    challenger_forecast_index = _utc_index(
        challenger_forecast, path=challenger_forecast_path
    )
    if not control_forecast_index.equals(challenger_forecast_index):
        raise HistoricalResidualComparisonError(
            "Les timelines forecast control et challenger different."
        )
    _validate_forecast_day(control_forecast_index, timezone=control_timezone)
    control_future = _source_quantiles(
        control_forecast,
        model=control_native,
        allow_final_columns=True,
        label="forecast control",
    )
    challenger_future = _source_quantiles(
        challenger_forecast,
        model=challenger_native,
        allow_final_columns=True,
        label="forecast challenger",
    )
    combined_forecast = challenger_forecast.copy()
    for column in list(combined_forecast.columns):
        if column.startswith(f"{control_native}__"):
            combined_forecast = combined_forecast.drop(columns=column)
    control_future_origin = _origin_series(control_forecast, model=control_native)
    challenger_future_origin = _origin_series(
        challenger_forecast, model=challenger_native
    )
    _validate_paired_origins(
        control_future_origin,
        challenger_future_origin,
        control_forecast_index,
        required_mask=np.ones(len(control_forecast_index), dtype=bool),
        label="forecast",
    )
    combined_forecast[f"{CONTROL_MODEL}_forecast_origin_utc"] = (
        control_future_origin.astype(str).to_numpy()
    )
    combined_forecast[f"{CHALLENGER_MODEL}_forecast_origin_utc"] = (
        challenger_future_origin.astype(str).to_numpy()
    )
    for quantile in QUANTILES:
        combined_forecast[f"{CONTROL_MODEL}__{quantile}"] = (
            control_future[quantile].to_numpy(dtype=float)
        )
        combined_forecast[f"{CHALLENGER_MODEL}__{quantile}"] = (
            challenger_future[quantile].to_numpy(dtype=float)
        )
        combined_forecast[quantile] = challenger_future[quantile].to_numpy(
            dtype=float
        )
    combined_forecast["price_eur_mwh"] = combined_forecast["q50"]

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    report_name = (
        f"chronos2_hourly_{zone_code.lower()}_"
        "residual_load_source_comparison.html"
    )
    try:
        _copy_challenger_inputs(challenger, staging)
        challenger_feature_manifest = challenger / "feature_manifest.csv"
        if challenger_feature_manifest.is_file():
            _verify_sealed_file(
                challenger_feature_manifest,
                run_dir=challenger,
                entries=_checksum_entries(challenger),
            )
            shutil.copy2(challenger_feature_manifest, staging / "feature_manifest.csv")

        _write_csv_gzip(combined_backtest, staging / "backtest_hourly_oof.csv.gz")
        combined_forecast.to_csv(
            staging / f"forecast_hourly_{zone_code.lower()}.csv", index=False
        )
        metrics_frame = pd.DataFrame(metrics)
        metrics_frame.to_csv(staging / "metrics_hourly.csv", index=False)
        metrics_by_model = {str(row["model"]): row for row in metrics}
        control_mae = float(metrics_by_model[CONTROL_MODEL]["mae"])
        challenger_mae = float(metrics_by_model[CHALLENGER_MODEL]["mae"])
        metrics_payload = {
            "metrics": metrics,
            "training_diagnostics": {
                "metric_scope": "sealed_final_365_delivery_days",
                "n_evaluation": FINAL_HOURS,
                "evaluation_start_local_date": final_start.isoformat(),
                "evaluation_end_local_date": final_end.isoformat(),
                "comparison": {
                    "control_model": CONTROL_MODEL,
                    "challenger_model": CHALLENGER_MODEL,
                    "mae_improvement_chronos_vs_saturn": (
                        control_mae - challenger_mae
                    ),
                    "mae_delta_chronos_minus_saturn": (
                        challenger_mae - control_mae
                    ),
                    "inputs_recomputed": True,
                    "statistics_history_reused": False,
                },
            },
        }
        _write_json(staging / "metrics_hourly.json", metrics_payload)

        run_manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_type": "historical_residual_load_source_comparison",
            "zone": zone_code,
            "timezone": control_timezone,
            "target_contract": "hourly_utc_no_interpolation",
            "delivery_horizon": "dynamic_23_24_25",
            "residual_load_source": CHALLENGER_SOURCE,
            "native_model": CHALLENGER_MODEL,
            "baseline_model": CONTROL_MODEL,
            "source_native_model": control_native,
            "renderer_compatibility_aliases": {
                "ensemble": CONTROL_MODEL,
                "residual_corrected": CHALLENGER_MODEL,
            },
            "statistics_comparison": {
                "source": "recalculated_paired_backtest",
                "candidate_model": CHALLENGER_MODEL,
                "benchmark_model": CONTROL_MODEL,
                "candidate_label": "Prix aval — residual_load Chronos-2",
                "benchmark_label": "Prix aval — residual_load Saturn",
                "scope_note": (
                    "FINAL365 scelle, deux branches entierement recalculees "
                    "sur les memes heures, cibles, folds et features hors "
                    "residual_load."
                ),
                "benchmark_contract": {
                    "id": "saturn_residual_load_control",
                    "label": "Residual_load Saturn",
                    "report_label": "Prix aval — residual_load Saturn",
                    "official_dashboard_metric": False,
                    "report_note": (
                        "Branche controle recalculee avec les forecasts "
                        "residual_load Saturn; comparateur d'evaluation "
                        "uniquement."
                    ),
                },
            },
            "n_evaluation_hours": FINAL_HOURS,
            "n_evaluation_days": FINAL_DAYS,
            "evaluation_start_local_date": final_start.isoformat(),
            "evaluation_end_local_date": final_end.isoformat(),
            "statistics_history_reused": False,
            "inputs_source": "challenger_recalculated_run",
            "control": {
                "run": str(control),
                "residual_load_source": CONTROL_SOURCE,
                "run_manifest_sha256": _sha256(control / "run_manifest.json"),
                "checksum_manifest_sha256": control_seal[
                    "checksum_manifest_sha256"
                ],
            },
            "challenger": {
                "run": str(challenger),
                "residual_load_source": CHALLENGER_SOURCE,
                "run_manifest_sha256": _sha256(challenger / "run_manifest.json"),
                "checksum_manifest_sha256": challenger_seal[
                    "checksum_manifest_sha256"
                ],
            },
        }
        _write_json(staging / "run_manifest.json", run_manifest)
        comparison_audit = {
            "schema_version": SCHEMA_VERSION,
            "status": "paired_historical_recalculation_validated",
            "zone": zone_code,
            "timezone": control_timezone,
            "control_source": CONTROL_SOURCE,
            "challenger_source": CHALLENGER_SOURCE,
            "source_native_model": control_native,
            "backtest_timestamps_identical": True,
            "actual_values_identical": True,
            "actual_max_absolute_difference": actual_max_difference,
            "fold_id_identical": True,
            "final_days": FINAL_DAYS,
            "final_hours": FINAL_HOURS,
            "final_start_local_date": final_start.isoformat(),
            "final_end_local_date": final_end.isoformat(),
            "final_start_utc": str(final_timeline[0]),
            "final_end_utc": str(final_timeline[-1]),
            "non_treatment_inputs": input_audit,
            "treatment_columns": list(RESIDUAL_LOAD_TREATMENT_COLUMNS),
            "inputs_copied_from": str(challenger),
            "statistics_history_reused": False,
            "statistics_history_written": False,
            "control_seal": control_seal,
            "challenger_seal": challenger_seal,
            "metrics_recomputed_from_combined_backtest": True,
        }
        _write_json(staging / "comparison_audit.json", comparison_audit)

        report_path = write_hourly_html_report(
            staging,
            output_path=staging / report_name,
            title=(
                title
                or f"Comparaison charge residuelle Saturn vs Chronos-2 — {zone_code}"
            ),
            native_model=CHALLENGER_MODEL,
            baseline_model=CONTROL_MODEL,
            zone=zone_code,
            timezone=control_timezone,
            extreme_threshold=float(extreme_threshold),
            history_hours=int(history_hours),
        )
        if not report_path.is_file() or report_path.stat().st_size == 0:
            raise HistoricalResidualComparisonError(
                "Le renderer standard n'a pas produit de rapport HTML."
            )
        _write_json(
            staging / "artifact_checksums.json",
            _output_checksums(staging, final_output=destination),
        )

        if destination.exists():
            raise FileExistsError(
                f"La destination immutable existe deja: {destination}"
            )
        os.rename(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return HistoricalResidualComparisonResult(
        output_dir=destination,
        report_path=destination / report_name,
        metrics_path=destination / "metrics_hourly.csv",
        control_mae=control_mae,
        challenger_mae=challenger_mae,
        final_start_local_date=final_start.isoformat(),
        final_end_local_date=final_end.isoformat(),
    )


__all__: Sequence[str] = (
    "CHALLENGER_MODEL",
    "CHALLENGER_SOURCE",
    "CONTROL_MODEL",
    "CONTROL_SOURCE",
    "FINAL_DAYS",
    "FINAL_HOURS",
    "HistoricalResidualComparisonError",
    "HistoricalResidualComparisonResult",
    "publish_historical_residual_comparison",
)
