from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import StructuralModelConfig, parse_structural_model_config
from .model import StructuralDayResult, solve_structural_day

LOGGER = logging.getLogger("chronos2_structural_market")


def _source_series(data: Any, alias: str) -> pd.Series | None:
    if alias in data.model_context_covariates.columns:
        return pd.to_numeric(
            data.model_context_covariates[alias], errors="coerce"
        )
    if alias in data.covariates.columns:
        return pd.to_numeric(data.covariates[alias], errors="coerce")
    return None


def build_standardized_inputs(
    data: Any,
    structural: StructuralModelConfig,
) -> pd.DataFrame:
    index = pd.DatetimeIndex(data.model_context_covariates.index)
    result = pd.DataFrame(index=index)

    for standard_name, spec in structural.inputs.items():
        if spec.alias:
            source = _source_series(data, spec.alias)
            if source is None:
                if spec.default is None:
                    result[standard_name] = np.nan
                else:
                    result[standard_name] = float(spec.default)
            else:
                result[standard_name] = (
                    source.reindex(index).astype(float) * float(spec.scale)
                )
                if spec.default is not None:
                    result[standard_name] = result[standard_name].fillna(
                        float(spec.default)
                    )
        else:
            result[standard_name] = (
                np.nan if spec.default is None else float(spec.default)
            )

    if "residual_load_gw" not in result:
        raise KeyError("structural_model.inputs.residual_load_gw est requis.")

    return result.astype(float)


def _day_key(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index.normalize())


def _repair_residual_load_for_day(
    frame: pd.DataFrame,
    structural: StructuralModelConfig,
) -> tuple[pd.DataFrame, int, list[str]]:
    """Répare au maximum quelques trous de residual_load_gw dans une journée.

    La réparation reste strictement intra-journalière. Elle n'utilise que les
    points de la même courbe de forecast déjà matérialisée par le pipeline
    point-in-time. Aucun jour voisin n'est utilisé.

    Les trous internes sont interpolés dans le temps. Les trous de bord sont
    remplis par la valeur valide la plus proche de la même journée.
    """
    if "residual_load_gw" not in frame:
        raise KeyError("residual_load_gw absente des entrées structurelles.")

    repaired = frame.copy()
    series = pd.to_numeric(
        repaired["residual_load_gw"],
        errors="coerce",
    )
    missing_mask = series.isna()
    missing_count = int(missing_mask.sum())
    missing_timestamps = [
        str(ts) for ts in repaired.index[missing_mask]
    ]

    if missing_count == 0:
        return repaired, 0, []

    if not structural.impute_residual_load:
        return repaired, missing_count, missing_timestamps

    max_missing = int(
        structural.max_residual_load_missing_hours_per_day
    )
    if max_missing < 0:
        raise ValueError(
            "max_residual_load_missing_hours_per_day doit être >= 0."
        )

    if missing_count > max_missing:
        return repaired, missing_count, missing_timestamps

    valid_count = int(series.notna().sum())
    if valid_count < max(2, len(series) - max_missing):
        return repaired, missing_count, missing_timestamps

    filled = series.interpolate(
        method="time",
        limit=max_missing,
        limit_direction="both",
    )

    # Garde-fou : aucune valeur ne doit rester manquante après une réparation
    # autorisée. Sinon la journée reste rejetée par solve_all_days.
    if filled.isna().any():
        return repaired, missing_count, missing_timestamps

    repaired["residual_load_gw"] = filled.astype(float)
    return repaired, 0, missing_timestamps


def solve_all_days(
    standardized_inputs: pd.DataFrame,
    structural: StructuralModelConfig,
    *,
    max_days: int | None = None,
    continue_on_error: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if standardized_inputs.empty:
        raise ValueError("Aucune entrée structurelle.")

    day_groups = list(standardized_inputs.groupby(_day_key(standardized_inputs.index)))
    if max_days is not None:
        day_groups = day_groups[-int(max_days) :]

    feature_parts: list[pd.DataFrame] = []
    dispatch_parts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []

    for number, (day, frame) in enumerate(day_groups, start=1):
        frame = frame.sort_index()

        raw_missing_residual = int(
            frame["residual_load_gw"].isna().sum()
        )
        frame, remaining_missing_residual, repaired_timestamps = (
            _repair_residual_load_for_day(
                frame,
                structural,
            )
        )
        imputed_residual = (
            raw_missing_residual - remaining_missing_residual
        )

        if remaining_missing_residual:
            diagnostics.append(
                {
                    "day": str(day),
                    "success": False,
                    "error_type": "MissingInput",
                    "error": (
                        f"{remaining_missing_residual} valeurs "
                        "residual_load_gw manquantes après réparation "
                        f"(brut={raw_missing_residual}, "
                        "max_autorisé="
                        f"{structural.max_residual_load_missing_hours_per_day})"
                    ),
                    "hours": int(len(frame)),
                    "residual_load_missing_raw": raw_missing_residual,
                    "residual_load_imputed_hours": imputed_residual,
                    "residual_load_missing_timestamps": " | ".join(
                        repaired_timestamps
                    ),
                }
            )
            continue

        if imputed_residual:
            LOGGER.info(
                "Réparation residual_load_gw %s : %d heure(s) imputée(s) : %s",
                day,
                imputed_residual,
                ", ".join(repaired_timestamps),
            )

        try:
            result: StructuralDayResult = solve_structural_day(frame, structural)
        except Exception as exc:
            LOGGER.exception("Échec du MILP pour %s : %s", day, exc)
            diagnostics.append(
                {
                    "day": str(day),
                    "success": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "hours": int(len(frame)),
                }
            )
            if not continue_on_error:
                raise
            continue

        feature_parts.append(result.features)
        dispatch_parts.append(result.dispatch)
        diagnostics.append(
            {
                "day": str(day),
                "residual_load_missing_raw": raw_missing_residual,
                "residual_load_imputed_hours": imputed_residual,
                "residual_load_missing_timestamps": " | ".join(
                    repaired_timestamps
                ),
                **result.diagnostics,
            }
        )
        if number == 1 or number % 25 == 0 or number == len(day_groups):
            LOGGER.info(
                "MILP structurel : %d/%d journées résolues.",
                number,
                len(day_groups),
            )

    if not feature_parts:
        raise RuntimeError("Aucune journée structurelle n'a été résolue.")

    features = pd.concat(feature_parts).sort_index()
    features.index.name = "timestamp"
    dispatch = (
        pd.concat(dispatch_parts, ignore_index=True)
        if dispatch_parts
        else pd.DataFrame()
    )
    diagnostics_frame = pd.DataFrame(diagnostics)
    return features, dispatch, diagnostics_frame


def save_structural_outputs(
    features: pd.DataFrame,
    dispatch: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    output_path: Path,
    diagnostics_path: Path,
    metadata_path: Path | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

    features.reset_index().to_csv(
        output_path,
        index=False,
        compression="gzip" if output_path.suffix.lower() == ".gz" else None,
    )
    diagnostics.to_csv(diagnostics_path, index=False)

    if not dispatch.empty:
        dispatch_path = output_path.with_name(
            output_path.name.replace(".csv.gz", "_dispatch.csv.gz")
        )
        dispatch.to_csv(dispatch_path, index=False, compression="gzip")

    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "rows": int(len(features)),
            "columns": list(features.columns),
            "first_timestamp": str(features.index.min()),
            "last_timestamp": str(features.index.max()),
            "days_successful": int(diagnostics.get("success", pd.Series(dtype=bool)).sum()),
            "days_total": int(len(diagnostics)),
        }
        if extra_metadata:
            payload.update(dict(extra_metadata))
        metadata_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def structural_config_from_mapping(config: Mapping[str, Any]) -> StructuralModelConfig:
    return parse_structural_model_config(config)
