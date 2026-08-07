from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chronos2_modular.common import LOGGER, ZoneConfig, ZoneData, deep_get


def structural_alignment_enabled(config: Mapping[str, Any]) -> bool:
    return bool(
        deep_get(
            config,
            "structural_model.alignment.enabled",
            True,
        )
    )


def structural_feature_aliases(config: Mapping[str, Any]) -> list[str]:
    aliases = deep_get(
        config,
        "structural_model.feature_aliases",
        [],
    ) or []
    return [str(alias) for alias in aliases]


def structural_price_alias(config: Mapping[str, Any]) -> str:
    return str(
        deep_get(
            config,
            "structural_model.residual_price_alias",
            "milp_structural_price",
        )
    )


def _required_aliases(
    data: ZoneData,
    config: Mapping[str, Any],
) -> list[str]:
    configured = deep_get(
        config,
        "structural_model.alignment.required_aliases",
        None,
    )
    if configured:
        if isinstance(configured, str):
            aliases = [configured]
        else:
            aliases = [str(value) for value in configured]
    else:
        aliases = structural_feature_aliases(config)

    # Le prix structurel est toujours requis.
    price_alias = structural_price_alias(config)
    if price_alias not in aliases:
        aliases.insert(0, price_alias)

    return aliases


def _minimum_history_hours(config: Mapping[str, Any]) -> int:
    explicit = deep_get(
        config,
        "structural_model.alignment.minimum_history_hours",
        None,
    )
    if explicit not in (None, ""):
        return int(explicit)

    context = int(
        deep_get(config, "model.context_length", 512)
    )
    windows = int(
        deep_get(config, "backtest.windows", 60)
    )
    horizon = int(
        deep_get(config, "model.horizon", 24)
    )

    # Suffisant pour le contexte + toutes les origines quotidiennes demandées.
    return context + (windows + 1) * horizon


def _contiguous_suffix_start(
    complete: pd.Series,
) -> pd.Timestamp | None:
    complete = complete.astype(bool)
    if complete.empty or not bool(complete.iloc[-1]):
        return None

    missing_positions = np.flatnonzero(
        ~complete.to_numpy(dtype=bool)
    )
    start_position = (
        int(missing_positions[-1]) + 1
        if len(missing_positions)
        else 0
    )
    return pd.Timestamp(complete.index[start_position])


def _rewrite_outputs(
    data: ZoneData,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.concat(
        [
            data.target.rename("target"),
            data.covariates,
        ],
        axis=1,
    ).reset_index(
        names="timestamp"
    ).to_csv(
        output_dir / "aligned_inputs.csv.gz",
        index=False,
        compression="gzip",
    )

    data.model_context_covariates.reset_index(
        names="timestamp"
    ).to_csv(
        output_dir / "model_covariates_with_future.csv.gz",
        index=False,
        compression="gzip",
    )


def align_zone_data_to_structural_window(
    data: ZoneData,
    config: Mapping[str, Any],
    output_dir: Path,
) -> ZoneData:
    if not structural_alignment_enabled(config):
        return data

    required = _required_aliases(data, config)
    missing_columns = [
        alias
        for alias in required
        if alias not in data.model_context_covariates.columns
    ]
    if missing_columns:
        raise KeyError(
            "Alignement structurel : colonnes absentes après chargement : "
            + ", ".join(missing_columns)
        )

    history_index = pd.DatetimeIndex(data.target.index)
    structural_history = data.model_context_covariates.reindex(
        history_index
    )[required]

    complete = structural_history.notna().all(axis=1)
    start = _contiguous_suffix_start(complete)

    if start is None:
        last_timestamp = (
            str(history_index[-1]) if len(history_index) else None
        )
        raise ValueError(
            "Alignement structurel : aucune période continue de features "
            f"jusqu'à la fin de l'historique cible ({last_timestamp}). "
            "Consulte structural_market_daily_diagnostics.csv."
        )

    suffix_hours = int((history_index >= start).sum())
    minimum_hours = _minimum_history_hours(config)

    if suffix_hours < minimum_hours:
        raise ValueError(
            "Alignement structurel : historique continu trop court. "
            f"Début={start}, heures={suffix_hours}, minimum={minimum_hours}. "
            "Le fichier MILP ressemble encore à un smoke test ou contient "
            "des échecs récents. Reconstruis le backfill complet."
        )

    future_index = data.model_context_covariates.index[
        data.model_context_covariates.index > history_index[-1]
    ]
    horizon = int(
        deep_get(config, "model.horizon", 24)
    )
    future_required = data.model_context_covariates.reindex(
        future_index[:horizon]
    )[required]

    if len(future_required) < horizon:
        raise ValueError(
            "Alignement structurel : horizon live incomplet dans les "
            f"features MILP ({len(future_required)}/{horizon})."
        )

    missing_future = int(
        (~future_required.notna().all(axis=1)).sum()
    )
    if missing_future:
        raise ValueError(
            "Alignement structurel : "
            f"{missing_future}/{horizon} heures futures MILP sont manquantes."
        )

    original_start = history_index[0]
    data.target = data.target.loc[data.target.index >= start].copy()
    data.covariates = data.covariates.reindex(
        data.target.index
    ).copy()
    data.model_context_covariates = (
        data.model_context_covariates.loc[
            data.model_context_covariates.index >= start
        ].copy()
    )

    # Recalcule la couverture sur la fenêtre effectivement utilisée.
    for alias in required:
        if alias in data.covariates.columns:
            mask = data.coverage["alias"].eq(alias)
            coverage = float(
                data.covariates[alias].notna().mean()
            )
            if mask.any():
                data.coverage.loc[
                    mask,
                    "coverage_exact",
                ] = coverage
                data.coverage.loc[
                    mask,
                    "coverage_after_fill",
                ] = coverage
                data.coverage.loc[
                    mask,
                    "missing_after_fill",
                ] = int(data.covariates[alias].isna().sum())

    data.diagnostics["structural_alignment"] = {
        "enabled": True,
        "original_target_start": str(original_start),
        "aligned_target_start": str(start),
        "aligned_target_end": str(data.target.index[-1]),
        "aligned_history_hours": int(len(data.target)),
        "minimum_required_hours": minimum_hours,
        "required_aliases": required,
        "future_hours_checked": horizon,
    }

    _rewrite_outputs(data, output_dir)

    LOGGER.info(
        "[%s] Fenêtre structurelle alignée : %s -> %s "
        "(%d heures, %d features requises).",
        data.zone,
        data.target.index[0],
        data.target.index[-1],
        len(data.target),
        len(required),
    )
    return data


def make_prepare_zone_data_with_structural_alignment(
    base_prepare: Callable[..., ZoneData],
) -> Callable[..., ZoneData]:
    def wrapped(
        zone: ZoneConfig,
        config: Mapping[str, Any],
        config_dir: Path,
        refresh: bool,
        output_dir: Path,
    ) -> ZoneData:
        data = base_prepare(
            zone,
            config,
            config_dir,
            refresh,
            output_dir,
        )
        return align_zone_data_to_structural_window(
            data,
            config,
            output_dir,
        )

    return wrapped
