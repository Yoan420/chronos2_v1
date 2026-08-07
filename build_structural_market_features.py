#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

import run_chronos2_extended_exogenous as extended
from chronos2_modular.common import (
    build_zone_configs,
    deep_get,
    load_yaml,
    resolve_path,
)
from chronos2_modular.saturn import sync_saturn_data
from chronos2_structural_market.config import (
    parse_structural_model_config,
)
from chronos2_structural_market.features import (
    build_standardized_inputs,
    save_structural_outputs,
    solve_all_days,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construit les covariables structurelles en résolvant un MILP "
            "agrégé par journée, puis un LP de pricing à commitment fixé."
        )
    )
    parser.add_argument(
        "--config",
        default="chronos2_selected_core_structural_covariates.yaml",
    )
    parser.add_argument("--zone", default="FR")
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--full-data-refresh", action="store_true")
    parser.add_argument("--data-as-of", default=None)
    parser.add_argument("--fail-fast", action="store_true")
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
    if args.data_as_of is not None:
        config.setdefault("data", {})["runtime_as_of"] = args.data_as_of
    structural = parse_structural_model_config(config)
    if not structural.enabled:
        raise ValueError("structural_model.enabled=false")

    zone_configs = build_zone_configs(
        config,
        [args.zone],
        None,
        list(structural.feature_aliases),
    )
    zone = zone_configs[0]

    project_root = resolve_path(
        deep_get(config, "data.project_root", "."),
        config_dir,
    )
    temporary_output = project_root / "runs" / "structural_feature_build" / args.zone.lower()
    temporary_output.mkdir(parents=True, exist_ok=True)

    if args.refresh_data or args.full_data_refresh or args.data_as_of is not None:
        manifest = sync_saturn_data(
            zone_configs,
            config,
            config_dir,
            full=args.full_data_refresh,
            as_of=args.data_as_of,
        )
        manifest.to_csv(
            temporary_output / "saturn_sync_manifest.csv",
            index=False,
        )

    data = extended.runner.prepare_zone_data(
        zone,
        config,
        config_dir,
        False,
        temporary_output,
    )
    inputs = build_standardized_inputs(data, structural)
    features, dispatch, diagnostics = solve_all_days(
        inputs,
        structural,
        max_days=args.max_days,
        continue_on_error=not args.fail_fast,
    )

    output_path = resolve_path(structural.output_file, project_root)
    diagnostics_path = resolve_path(structural.diagnostics_file, project_root)
    metadata_path = output_path.with_name("structural_market_metadata.json")
    save_structural_outputs(
        features,
        dispatch,
        diagnostics,
        output_path=output_path,
        diagnostics_path=diagnostics_path,
        metadata_path=metadata_path,
        extra_metadata={
            "technology_code_map": {
                technology.name: index + 1
                for index, technology in enumerate(structural.technologies)
            },
            "residual_price_alias": structural.residual_price_alias,
            "solver": {
                "mip_rel_gap": structural.solver.mip_rel_gap,
                "time_limit_seconds": structural.solver.time_limit_seconds,
                "node_limit": structural.solver.node_limit,
            },
        },
    )

    successful = int(diagnostics.get("success", pd.Series(dtype=bool)).sum())
    print("\nModèle structurel terminé")
    print(f"  fichier features : {output_path}")
    print(f"  lignes            : {len(features):,}")
    print(f"  journées résolues : {successful}/{len(diagnostics)}")
    print(f"  période           : {features.index.min()} -> {features.index.max()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
