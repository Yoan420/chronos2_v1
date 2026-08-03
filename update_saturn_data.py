#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from chronos2_modular.common import (
    LOGGER,
    build_zone_configs,
    deep_get,
    load_yaml,
    resolve_path,
)
from chronos2_modular.saturn import sync_saturn_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extrait et actualise les séries Saturn utilisées par "
            "Chronos-2, avec historique de révisions pour les forecasts."
        )
    )
    parser.add_argument(
        "--config",
        default="chronos2_inputs_asof_jplus1.yaml",
    )
    parser.add_argument("--zones", nargs="+", default=None)
    parser.add_argument(
        "--include-covariates",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--exclude-covariates",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Reconstruit les fichiers depuis la date initiale au lieu "
            "d'effectuer une mise à jour incrémentale."
        ),
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help=(
            "Plafond de révision ISO-8601. Une date naïve est interprétée "
            "en UTC. Par défaut : maintenant."
        ),
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Chemin facultatif du manifeste CSV de synchronisation.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    config_dir = config_path.parent
    zone_configs = build_zone_configs(
        config,
        args.zones,
        args.include_covariates,
        args.exclude_covariates,
    )
    if not zone_configs:
        raise ValueError("Aucune zone sélectionnée.")

    manifest = sync_saturn_data(
        zone_configs,
        config,
        config_dir,
        full=args.full,
        as_of=args.as_of,
    )

    project_root = resolve_path(
        deep_get(config, "data.project_root", "."),
        config_dir,
    )
    manifest_path = (
        resolve_path(args.manifest, Path.cwd())
        if args.manifest
        else resolve_path(
            deep_get(
                config,
                "data.saturn_sync.manifest",
                "data/saturn_sync_manifest.csv",
            ),
            project_root,
        )
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    if manifest.empty:
        LOGGER.warning("Aucune série Saturn à actualiser.")
    else:
        columns = [
            "zone",
            "alias",
            "kind",
            "status",
            "rows_downloaded",
            "rows_after",
        ]
        print("\n" + manifest.loc[:, columns].to_string(index=False))

    print(f"\nManifeste Saturn : {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.error("Actualisation Saturn interrompue.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.exception("Échec de l'actualisation Saturn : %s", exc)
        raise SystemExit(1)
