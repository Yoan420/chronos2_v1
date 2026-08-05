from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import LOGGER, ZoneConfig, ZoneData, deep_get


def _settings(
    config: Mapping[str, Any],
    zone_name: str,
) -> dict[str, Any]:
    global_settings = (
        deep_get(
            config,
            "exogenous_extensions.interconnection_capacities",
            {},
        )
        or {}
    )
    zone_settings = (
        deep_get(
            config,
            (
                f"zones.{zone_name}.exogenous_extensions."
                "interconnection_capacities"
            ),
            {},
        )
        or {}
    )
    return {**global_settings, **zone_settings}


def _append_unique(values: list[str], item: str) -> None:
    if item not in values:
        values.append(item)


def _add_derived_with_lags(
    data: ZoneData,
    alias: str,
    values: pd.Series,
    *,
    description: str,
    lags: tuple[int, ...],
) -> None:
    history = pd.to_numeric(
        values.reindex(data.target.index),
        errors="coerce",
    ).astype(np.float32)

    data.covariates[alias] = history
    data.model_context_covariates.loc[
        data.target.index, alias
    ] = history.to_numpy()

    future_index = data.model_context_covariates.index.difference(
        data.target.index
    )
    data.model_context_covariates.loc[future_index, alias] = np.nan

    expanded = history.reindex(data.model_context_covariates.index)
    for lag in lags:
        known_column = f"known_{alias}_lag{lag}"
        data.model_context_covariates[known_column] = expanded.shift(lag)
        _append_unique(data.known_future_columns, known_column)

    coverage = float(history.notna().mean())
    data.coverage = pd.concat(
        [
            data.coverage,
            pd.DataFrame(
                [
                    {
                        "zone": data.zone,
                        "alias": alias,
                        "series": "derived_interconnection_capacity",
                        "coverage_exact": coverage,
                        "coverage_after_fill": coverage,
                        "missing_after_fill": int(history.isna().sum()),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    data.input_manifest = pd.concat(
        [
            data.input_manifest,
            pd.DataFrame(
                [
                    {
                        "alias": alias,
                        "role": "derived_past_covariate",
                        "series": "derived_interconnection_capacity",
                        "description": description,
                        "future_strategies": ", ".join(
                            f"lag{lag}" for lag in lags
                        ),
                        "known_future": False,
                        "source": "derived",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )


def add_interconnection_capacity_features(
    data: ZoneData,
    config: Mapping[str, Any],
) -> list[str]:
    settings = _settings(config, data.zone)
    if not bool(settings.get("enabled", False)):
        return []

    aliases = settings.get("aliases", {}) or {}
    import_aliases = [
        str(alias)
        for alias in aliases.get("import", [])
        if str(alias) in data.covariates
    ]
    export_aliases = [
        str(alias)
        for alias in aliases.get("export", [])
        if str(alias) in data.covariates
    ]

    if not import_aliases and not export_aliases:
        raise ValueError(
            "Extension interconnexion activée, mais aucune série de "
            "capacité configurée n'a été chargée."
        )

    lags = tuple(
        int(value)
        for value in settings.get("derived_lags", [24, 168])
    )
    added: list[str] = []

    import_frame = pd.DataFrame(
        {
            alias: pd.to_numeric(
                data.covariates[alias], errors="coerce"
            )
            for alias in import_aliases
        },
        index=data.target.index,
    )
    export_frame = pd.DataFrame(
        {
            alias: pd.to_numeric(
                data.covariates[alias], errors="coerce"
            )
            for alias in export_aliases
        },
        index=data.target.index,
    )

    derived: dict[str, tuple[pd.Series, str]] = {}

    if not import_frame.empty:
        derived.update(
            {
                "interconnection_import_capacity_total": (
                    import_frame.sum(axis=1, min_count=1),
                    "Somme des capacités d'import vers la France.",
                ),
                "interconnection_import_capacity_min": (
                    import_frame.min(axis=1, skipna=True),
                    "Capacité d'import minimale parmi les frontières.",
                ),
                "interconnection_import_capacity_dispersion": (
                    import_frame.std(axis=1, skipna=True),
                    "Dispersion des capacités d'import par frontière.",
                ),
            }
        )

    if not export_frame.empty:
        derived.update(
            {
                "interconnection_export_capacity_total": (
                    export_frame.sum(axis=1, min_count=1),
                    "Somme des capacités d'export depuis la France.",
                ),
                "interconnection_export_capacity_min": (
                    export_frame.min(axis=1, skipna=True),
                    "Capacité d'export minimale parmi les frontières.",
                ),
                "interconnection_export_capacity_dispersion": (
                    export_frame.std(axis=1, skipna=True),
                    "Dispersion des capacités d'export par frontière.",
                ),
            }
        )

    if not import_frame.empty and not export_frame.empty:
        import_total = import_frame.sum(axis=1, min_count=1)
        export_total = export_frame.sum(axis=1, min_count=1)
        derived["interconnection_capacity_asymmetry"] = (
            import_total - export_total,
            "Capacité totale d'import moins capacité totale d'export.",
        )

    for alias, (values, description) in derived.items():
        _add_derived_with_lags(
            data,
            alias,
            values,
            description=description,
            lags=lags,
        )
        added.append(alias)

    data.diagnostics["interconnection_capacities"] = {
        "enabled": True,
        "raw_import_aliases": import_aliases,
        "raw_export_aliases": export_aliases,
        "derived_columns": added,
        "derived_lags": list(lags),
        "causality": (
            "Les capacités brutes et agrégées sont injectées avec des "
            "lags stricts de 24 h et 168 h."
        ),
    }

    LOGGER.info(
        "[%s] Capacités d'interconnexion : %d imports, %d exports, "
        "%d facteurs dérivés.",
        data.zone,
        len(import_aliases),
        len(export_aliases),
        len(added),
    )
    return added


def make_prepare_zone_data_with_interconnections(
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
        add_interconnection_capacity_features(data, config)
        data.model_context_covariates = (
            data.model_context_covariates.astype(np.float32)
        )
        return data

    return wrapped
