from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any
import re

import pandas as pd

from .common import LOGGER, ZoneConfig, ZoneData, deep_get


_FUTURE_PATTERN = re.compile(
    r"^known_(.+)_(lag24|lag168|persistence|oracle)$"
)


def _as_patterns(
    definitions: Mapping[str, Any],
    selected_groups: Sequence[str],
) -> list[str]:
    patterns: list[str] = []
    for group in selected_groups:
        if group not in definitions:
            raise KeyError(
                f"Groupe de sélection inconnu : {group}. "
                f"Disponibles : {sorted(definitions)}"
            )
        raw = definitions[group]
        if isinstance(raw, Mapping):
            raw_patterns = raw.get("patterns", [])
        else:
            raw_patterns = raw
        if isinstance(raw_patterns, str):
            raw_patterns = [raw_patterns]
        for pattern in raw_patterns or []:
            value = str(pattern)
            if value not in patterns:
                patterns.append(value)
    return patterns


def _matches(value: str, patterns: Sequence[str]) -> bool:
    return any(fnmatchcase(value, pattern) for pattern in patterns)


def filter_zone_data(
    data: ZoneData,
    config: Mapping[str, Any],
    output_dir: Path | None = None,
) -> ZoneData:
    settings = deep_get(config, "feature_selection", {}) or {}
    if not bool(settings.get("enabled", False)):
        return data

    definitions = settings.get("group_definitions", {}) or {}
    selected_groups = [str(x) for x in settings.get("selected_groups", [])]
    if not selected_groups:
        LOGGER.warning(
            "[%s] feature_selection actif sans groupe : aucun exogène retenu.",
            data.zone,
        )

    patterns = _as_patterns(definitions, selected_groups)
    all_model_columns = list(data.model_context_covariates.columns)
    retained_model_columns = [
        column
        for column in all_model_columns
        if _matches(str(column), patterns)
    ]
    dropped_model_columns = [
        column
        for column in all_model_columns
        if column not in retained_model_columns
    ]

    retained_future_columns = [
        column
        for column in data.known_future_columns
        if column in retained_model_columns
    ]

    required_aliases: set[str] = set()
    for column in retained_future_columns:
        match = _FUTURE_PATTERN.match(str(column))
        if match:
            required_aliases.add(match.group(1))

    retained_covariates = [
        alias
        for alias in data.covariates.columns
        if _matches(str(alias), patterns) or alias in required_aliases
    ]
    dropped_covariates = [
        alias
        for alias in data.covariates.columns
        if alias not in retained_covariates
    ]

    data.model_context_covariates = data.model_context_covariates.loc[
        :, retained_model_columns
    ].copy()
    data.known_future_columns = retained_future_columns
    data.covariates = data.covariates.loc[:, retained_covariates].copy()

    diagnostics = data.diagnostics.setdefault("feature_selection", {})
    diagnostics.update(
        {
            "selected_groups": selected_groups,
            "patterns": patterns,
            "retained_model_columns": retained_model_columns,
            "dropped_model_columns": dropped_model_columns,
            "retained_covariates": retained_covariates,
            "dropped_covariates": dropped_covariates,
            "retained_known_future_columns": retained_future_columns,
        }
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "column": all_model_columns,
                "retained": [
                    column in retained_model_columns
                    for column in all_model_columns
                ],
            }
        ).to_csv(
            output_dir / "feature_selection_columns.csv",
            index=False,
        )
        data.model_context_covariates.reset_index(
            names="timestamp"
        ).to_csv(
            output_dir / "model_covariates_selected.csv.gz",
            index=False,
            compression="gzip",
        )

    LOGGER.info(
        "[%s] Sélection : %d/%d colonnes modèle et %d/%d covariables brutes.",
        data.zone,
        len(retained_model_columns),
        len(all_model_columns),
        len(retained_covariates),
        len(retained_covariates) + len(dropped_covariates),
    )
    return data


def make_prepare_zone_data_with_feature_selection(
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
        return filter_zone_data(data, config, output_dir)

    return wrapped
