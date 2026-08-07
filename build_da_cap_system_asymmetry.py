#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


DERIVED_ALIAS = "da_cap_system_asymmetry"

# Version "screening-compatible clean":
# seules les séries à couverture horaire élevée sont conservées.
COMPONENTS: dict[str, dict[str, str]] = {
    "da_cap_es_fr": {
        "series": (
            "power.storm.flow.dayahead.capacity."
            "es.fr.mw.h.obs.entsoe"
        ),
        "side": "import",
    },
    "da_cap_it_north_fr": {
        "series": (
            "power.storm.flow.dayahead.capacity."
            "it_north.fr.mw.h.obs.entsoe"
        ),
        "side": "import",
    },
    "da_cap_uk_fr": {
        "series": (
            "power.storm.flow.dayahead.capacity."
            "uk.fr.mw.h.obs.entsoe"
        ),
        "side": "import",
    },
    "da_cap_fr_it_north": {
        "series": (
            "power.storm.flow.dayahead.capacity."
            "fr.it_north.mw.h.obs.entsoe"
        ),
        "side": "export",
    },
    "da_cap_fr_uk": {
        "series": (
            "power.storm.flow.dayahead.capacity."
            "fr.uk.mw.h.obs.entsoe"
        ),
        "side": "export",
    },
}


def _as_utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True)


def _normalize_output_vintage(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalisation minimale au format PIT attendu par Chronos-2."""
    required = [
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
        "value",
        "downloaded_at_utc",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"Colonnes PIT absentes : {missing}")

    result = frame.loc[:, required].copy()
    for column in (
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
        "downloaded_at_utc",
    ):
        result[column] = pd.to_datetime(
            result[column], errors="coerce", utc=True
        )
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    result = result.dropna(
        subset=[
            "value_time_utc",
            "snapshot_time_utc",
            "revision_time_utc",
            "value",
        ]
    )
    return (
        result.sort_values(
            ["revision_time_utc", "value_time_utc"]
        )
        .drop_duplicates(
            ["revision_time_utc", "value_time_utc"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def component_store_paths(raw_dir: Path) -> dict[str, Path]:
    return {
        alias: raw_dir / f"{alias}.parquet"
        for alias in COMPONENTS
    }


def build_component_sync_config(
    base_config: Mapping[str, Any],
    *,
    zone: str,
    raw_dir: Path,
) -> dict[str, Any]:
    """
    Configuration en mémoire dédiée aux cinq stores PIT techniques.
    Les cinq capacités ne sont jamais envoyées directement à Chronos.
    """
    from chronos2_modular.common import deep_get

    config = copy.deepcopy(dict(base_config))
    zone = zone.upper()
    base_zone = deep_get(base_config, f"zones.{zone}")
    if not isinstance(base_zone, Mapping):
        raise KeyError(f"Zone absente de la configuration : {zone}")

    config.setdefault("data", {})
    config["data"]["pit_vintage_dir"] = str(raw_dir.resolve())
    config["data"]["pit_files"] = {
        alias: f"{alias}.parquet"
        for alias in COMPONENTS
    }
    config["zones"] = {
        zone: {
            "enabled": True,
            "timezone": base_zone.get("timezone", "Europe/Paris"),
            "include_calendar": False,
            "target": copy.deepcopy(base_zone["target"]),
            "covariates": {
                alias: {
                    "enabled": True,
                    "source": "pit_parquet",
                    "series": settings["series"],
                    "description": (
                        "Capacité d'interconnexion Day-Ahead "
                        f"{alias.removeprefix('da_cap_')}"
                    ),
                    "fill_method": "none",
                    "minimum_coverage": 0.0,
                    "future": {
                        "known_future": True,
                        "strategies": ["oracle"],
                    },
                }
                for alias, settings in COMPONENTS.items()
            },
        }
    }
    return config


def synchronize_component_stores(
    base_config: Mapping[str, Any],
    *,
    config_dir: Path,
    zone: str,
    raw_dir: Path,
    full: bool,
    as_of: str | None,
) -> pd.DataFrame:
    from chronos2_modular.common import build_zone_configs
    from chronos2_modular.saturn import sync_saturn_data

    raw_dir.mkdir(parents=True, exist_ok=True)
    sync_config = build_component_sync_config(
        base_config,
        zone=zone,
        raw_dir=raw_dir,
    )
    zones = build_zone_configs(
        sync_config,
        [zone],
        list(COMPONENTS),
        None,
    )
    return sync_saturn_data(
        zones,
        sync_config,
        config_dir,
        full=full,
        as_of=as_of,
    )


def read_component_events(
    path: Path,
    alias: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier PIT source absent pour {alias}: {path}"
        )

    frame = pd.read_parquet(path)
    required = {
        "value_time_utc",
        "snapshot_time_utc",
        "revision_time_utc",
        "value",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{path.name}: colonnes absentes {missing}")
    if frame.empty:
        raise ValueError(f"Fichier PIT vide pour {alias}: {path}")

    event_time = pd.concat(
        [
            _as_utc(frame["snapshot_time_utc"]),
            _as_utc(frame["revision_time_utc"]),
        ],
        axis=1,
    ).max(axis=1)

    events = pd.DataFrame(
        {
            "value_time_utc": _as_utc(frame["value_time_utc"]),
            "event_time_utc": event_time,
            "alias": alias,
            "component_value_mw": pd.to_numeric(
                frame["value"], errors="coerce"
            ),
        }
    ).dropna(
        subset=[
            "value_time_utc",
            "event_time_utc",
            "component_value_mw",
        ]
    )

    return (
        events.sort_values(
            ["value_time_utc", "event_time_utc"]
        )
        .drop_duplicates(
            ["value_time_utc", "event_time_utc", "alias"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def derive_asymmetry_vintages(
    component_frames: Mapping[str, pd.DataFrame],
    *,
    output_unit: str = "GW",
    downloaded_at_utc: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Rejoue chronologiquement les révisions pour chaque heure de livraison.

    Formule:
      ES->FR + IT_NORTH->FR + UK->FR
      - FR->IT_NORTH - FR->UK
    """
    missing = sorted(set(COMPONENTS) - set(component_frames))
    if missing:
        raise KeyError(f"Composantes absentes : {missing}")

    long = pd.concat(
        [component_frames[alias] for alias in COMPONENTS],
        ignore_index=True,
    )
    if long.empty:
        raise ValueError("Aucun événement de capacité disponible.")

    # ?vite pivot_table, qui peut construire un produit cart?sien
    # gigantesque entre les timestamps et consommer plusieurs Go de RAM.
    long = (
        long.sort_values(
            [
                "value_time_utc",
                "event_time_utc",
                "alias",
            ]
        )
        .drop_duplicates(
            [
                "value_time_utc",
                "event_time_utc",
                "alias",
            ],
            keep="last",
        )
    )

    wide = (
        long.set_index(
            [
                "value_time_utc",
                "event_time_utc",
                "alias",
            ]
        )["component_value_mw"]
        .unstack("alias")
        .sort_index()
        .reindex(columns=list(COMPONENTS))
    )

    wide = wide.groupby(
        level="value_time_utc",
        group_keys=False,
    ).ffill()
    complete = wide.dropna(subset=list(COMPONENTS)).copy()
    if complete.empty:
        raise ValueError(
            "Aucun snapshot ne contient les cinq composantes."
        )

    import_aliases = [
        alias
        for alias, settings in COMPONENTS.items()
        if settings["side"] == "import"
    ]
    export_aliases = [
        alias
        for alias, settings in COMPONENTS.items()
        if settings["side"] == "export"
    ]

    value_mw = (
        complete[import_aliases].sum(axis=1)
        - complete[export_aliases].sum(axis=1)
    )

    unit = output_unit.strip().upper()
    if unit == "GW":
        value = value_mw / 1000.0
    elif unit == "MW":
        value = value_mw
    else:
        raise ValueError("output_unit doit valoir MW ou GW.")

    previous = value.groupby(level="value_time_utc").shift()
    changed = previous.isna() | ~np.isclose(
        value.to_numpy(dtype=float),
        previous.to_numpy(dtype=float),
        equal_nan=True,
        atol=1e-9,
        rtol=0.0,
    )
    value = value.loc[changed]

    value_time = value.index.get_level_values("value_time_utc")
    event_time = value.index.get_level_values("event_time_utc")
    downloaded_at = pd.Timestamp(
        downloaded_at_utc or pd.Timestamp.now(tz="UTC")
    )
    if downloaded_at.tzinfo is None:
        downloaded_at = downloaded_at.tz_localize("UTC")
    else:
        downloaded_at = downloaded_at.tz_convert("UTC")

    result = pd.DataFrame(
        {
            "value_time_utc": value_time,
            "snapshot_time_utc": event_time,
            "revision_time_utc": event_time,
            "value": value.to_numpy(dtype=np.float32),
            "downloaded_at_utc": downloaded_at,
        }
    )
    return _normalize_output_vintage(result)


def build_derived_store(
    *,
    raw_dir: Path,
    output_path: Path,
    output_unit: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stores = component_store_paths(raw_dir)
    frames = {
        alias: read_component_events(path, alias)
        for alias, path in stores.items()
    }
    derived = derive_asymmetry_vintages(
        frames,
        output_unit=output_unit,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    derived.to_parquet(output_path, index=False)

    manifest_rows = []
    for alias, frame in frames.items():
        manifest_rows.append(
            {
                "alias": alias,
                "series": COMPONENTS[alias]["series"],
                "side": COMPONENTS[alias]["side"],
                "events": int(len(frame)),
                "delivery_hours": int(
                    frame["value_time_utc"].nunique()
                ),
                "first_delivery_utc": frame[
                    "value_time_utc"
                ].min(),
                "last_delivery_utc": frame[
                    "value_time_utc"
                ].max(),
                "first_event_utc": frame["event_time_utc"].min(),
                "last_event_utc": frame["event_time_utc"].max(),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(
        output_path.with_name(
            f"{output_path.stem}_components_manifest.csv"
        ),
        index=False,
    )

    metadata = {
        "alias": DERIVED_ALIAS,
        "unit": output_unit.upper(),
        "formula": (
            "ES->FR + IT_NORTH->FR + UK->FR "
            "- FR->IT_NORTH - FR->UK"
        ),
        "rows": int(len(derived)),
        "delivery_hours": int(
            derived["value_time_utc"].nunique()
        ),
        "first_delivery_utc": str(
            derived["value_time_utc"].min()
        ),
        "last_delivery_utc": str(
            derived["value_time_utc"].max()
        ),
        "first_revision_utc": str(
            derived["revision_time_utc"].min()
        ),
        "last_revision_utc": str(
            derived["revision_time_utc"].max()
        ),
    }
    output_path.with_name(
        f"{output_path.stem}_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return derived, manifest


def resolve_default_paths(
    config: Mapping[str, Any],
    config_dir: Path,
) -> tuple[Path, Path]:
    from chronos2_modular.common import deep_get, resolve_path

    project_root = resolve_path(
        deep_get(config, "data.project_root", "."),
        config_dir,
    )
    main_pit_dir = resolve_path(
        deep_get(
            config,
            "data.pit_vintage_dir",
            "data/pit/vintages",
        ),
        project_root,
    )
    raw_dir = main_pit_dir / "da_capacity_components"

    configured = deep_get(
        config,
        f"data.pit_files.{DERIVED_ALIAS}",
        f"{DERIVED_ALIAS}.parquet",
    )
    output_path = Path(str(configured)).expanduser()
    if not output_path.is_absolute():
        output_path = main_pit_dir / output_path
    return raw_dir.resolve(), output_path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronise les capacités Day-Ahead point-in-time et "
            "construit da_cap_system_asymmetry."
        )
    )
    parser.add_argument(
        "--config",
        default="chronos2_selected_core.yaml",
    )
    parser.add_argument("--zone", default="FR")
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--unit",
        choices=("GW", "MW"),
        default="GW",
    )
    return parser.parse_args()


def main() -> int:
    from chronos2_modular.common import load_yaml

    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    config_dir = config_path.parent

    default_raw_dir, default_output = resolve_default_paths(
        config, config_dir
    )
    raw_dir = (
        Path(args.raw_dir).expanduser().resolve()
        if args.raw_dir
        else default_raw_dir
    )
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_output
    )

    if not args.skip_sync:
        manifest = synchronize_component_stores(
            config,
            config_dir=config_dir,
            zone=args.zone,
            raw_dir=raw_dir,
            full=args.full,
            as_of=args.as_of,
        )
        sync_manifest_path = raw_dir / "sync_manifest.csv"
        manifest.to_csv(sync_manifest_path, index=False)
        print(
            f"Manifest de synchronisation : {sync_manifest_path}"
        )

    derived, _ = build_derived_store(
        raw_dir=raw_dir,
        output_path=output_path,
        output_unit=args.unit,
    )

    print("\nSérie dérivée construite :")
    print(f"  fichier    : {output_path}")
    print(f"  lignes     : {len(derived):,}")
    print(
        "  livraisons : "
        f"{derived['value_time_utc'].nunique():,}"
    )
    print(
        "  période    : "
        f"{derived['value_time_utc'].min()} -> "
        f"{derived['value_time_utc'].max()}"
    )
    print(f"  unité      : {args.unit}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
