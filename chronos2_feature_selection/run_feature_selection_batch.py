#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pandas as pd
import yaml

import chronos2_modular.forecasting as forecasting
from chronos2_order_signals.chronos_compat import future_proxy_frame_fixed

forecasting.future_proxy_frame = future_proxy_frame_fixed

import run_chronos2_modular as runner  # noqa: E402
from chronos2_modular.common import (  # noqa: E402
    build_zone_configs,
    deep_get,
    load_yaml,
    resolve_path,
    set_reproducibility,
)

try:
    from chronos2_modular.regime import (
        future_proxy_frame_with_regime,
        prepare_zone_data_with_regime,
    )
except ImportError:
    pass
else:
    runner.prepare_zone_data = prepare_zone_data_with_regime
    forecasting.future_proxy_frame = future_proxy_frame_with_regime

try:
    from chronos2_modular.exogenous_extensions import (
        make_prepare_zone_data_with_extensions,
    )
except ImportError:
    pass
else:
    runner.prepare_zone_data = make_prepare_zone_data_with_extensions(
        runner.prepare_zone_data
    )

from chronos2_modular.feature_selection import (  # noqa: E402
    make_prepare_zone_data_with_feature_selection,
)

runner.prepare_zone_data = make_prepare_zone_data_with_feature_selection(
    runner.prepare_zone_data
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablations temporelles de familles ou composants Chronos-2."
    )
    parser.add_argument(
        "--config",
        default="chronos2_inputs_extended_exogenous.yaml",
    )
    parser.add_argument(
        "--groups",
        default="feature_groups.yaml",
    )
    parser.add_argument(
        "--level",
        choices=("family", "component"),
        default="family",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "only", "loo", "both"),
        default="only",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Familles ou composants à tester. Par défaut : tous.",
    )
    parser.add_argument(
        "--fold-asof",
        nargs="+",
        default=None,
        help=(
            "Cutoffs opérationnels ISO-8601. Sans valeur, utilise le cutoff "
            "de la configuration courante."
        ),
    )
    parser.add_argument("--backtest-windows", type=int, default=60)
    parser.add_argument("--zones", nargs="+", default=["FR"])
    parser.add_argument(
        "--output-dir",
        default="runs/feature_selection",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_group_spec(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise TypeError("feature_groups.yaml doit contenir un mapping.")
    return value


def component_definitions(spec: Mapping[str, Any]) -> dict[str, Any]:
    definitions = spec.get("components", {}) or {}
    if not isinstance(definitions, Mapping):
        raise TypeError("components doit être un mapping.")
    return {str(k): v for k, v in definitions.items()}


def family_components(spec: Mapping[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for family, raw in (spec.get("families", {}) or {}).items():
        if not isinstance(raw, Mapping):
            raise TypeError(f"Famille invalide : {family}")
        result[str(family)] = [str(x) for x in raw.get("components", [])]
    return result


def subject_components(
    spec: Mapping[str, Any],
    level: str,
) -> dict[str, list[str]]:
    components = component_definitions(spec)
    if level == "component":
        return {name: [name] for name in components}
    return family_components(spec)


def all_components(spec: Mapping[str, Any]) -> list[str]:
    return list(component_definitions(spec))


def build_scenarios(
    spec: Mapping[str, Any],
    level: str,
    mode: str,
    subjects: list[str] | None,
) -> list[dict[str, Any]]:
    mapping = subject_components(spec, level)
    chosen = subjects or list(mapping)
    unknown = sorted(set(chosen) - set(mapping))
    if unknown:
        raise KeyError(
            f"Sujets inconnus au niveau {level} : {unknown}. "
            f"Disponibles : {sorted(mapping)}"
        )

    universe = all_components(spec)
    scenarios: list[dict[str, Any]] = [
        {
            "scenario": "full",
            "selection_kind": "full",
            "subject": "all",
            "selected_components": universe,
        }
    ]
    if mode in {"only", "both"}:
        for subject in chosen:
            scenarios.append(
                {
                    "scenario": f"only__{subject}",
                    "selection_kind": "only",
                    "subject": subject,
                    "selected_components": mapping[subject],
                }
            )
    if mode in {"loo", "both"}:
        for subject in chosen:
            removed = set(mapping[subject])
            scenarios.append(
                {
                    "scenario": f"without__{subject}",
                    "selection_kind": "loo",
                    "subject": subject,
                    "selected_components": [
                        component
                        for component in universe
                        if component not in removed
                    ],
                }
            )
    if mode == "full":
        return scenarios[:1]
    return scenarios


def safe_slug(value: str) -> str:
    return (
        value.replace(":", "")
        .replace("+", "p")
        .replace("-", "")
        .replace("T", "_")
        .replace("/", "_")
    )


def flatten_metrics(prefix: str, metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    if metrics is None:
        return {}
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config_path = Path(args.config).expanduser().resolve()
    groups_path = Path(args.groups).expanduser().resolve()
    base_config = load_yaml(config_path)
    spec = load_group_spec(groups_path)
    definitions = component_definitions(spec)
    scenarios = build_scenarios(
        spec,
        args.level,
        args.mode,
        args.subjects,
    )

    folds: list[str | None] = args.fold_asof or [
        deep_get(base_config, "data.runtime_as_of")
    ]
    if folds == [None]:
        folds = [None]

    set_reproducibility(int(deep_get(base_config, "model.seed", 42)))
    runtime = runner.load_model(
        base_config,
        args.device,
        args.local_files_only,
    )

    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    runner_args = SimpleNamespace(
        context_length=None,
        backtest_windows=args.backtest_windows,
    )

    for fold_number, fold_asof in enumerate(folds, start=1):
        fold_name = (
            f"fold_{fold_number:02d}_{safe_slug(str(fold_asof))}"
            if fold_asof
            else f"fold_{fold_number:02d}_current"
        )
        baseline_metrics: dict[str, Any] | None = None
        full_metrics: dict[str, Any] | None = None

        for scenario_number, scenario in enumerate(scenarios, start=1):
            config = copy.deepcopy(base_config)
            if fold_asof:
                config.setdefault("data", {})["runtime_as_of"] = fold_asof
            config.setdefault("backtest", {})[
                "price_only_baseline"
            ] = scenario_number == 1
            config["feature_selection"] = {
                "enabled": True,
                "selected_groups": scenario["selected_components"],
                "group_definitions": definitions,
            }

            zone_configs = build_zone_configs(
                config,
                args.zones,
                None,
                None,
            )
            if len(zone_configs) != 1:
                raise ValueError(
                    "Le batch de sélection attend exactement une zone."
                )

            scenario_root = root / fold_name / scenario["scenario"]
            scenario_root.mkdir(parents=True, exist_ok=True)

            result = runner.run_zone(
                zone_configs[0],
                config,
                config_path.parent,
                runner_args,
                runtime,
                scenario_root,
                data_refresh=False,
            )

            if result.metrics_baseline is not None:
                baseline_metrics = dict(result.metrics_baseline)
            if scenario["selection_kind"] == "full":
                full_metrics = dict(result.metrics_native)

            row: dict[str, Any] = {
                "fold": fold_name,
                "fold_asof": fold_asof,
                "level": args.level,
                "mode": args.mode,
                "scenario": scenario["scenario"],
                "selection_kind": scenario["selection_kind"],
                "subject": scenario["subject"],
                "selected_components": ",".join(
                    scenario["selected_components"]
                ),
                "selected_component_count": len(
                    scenario["selected_components"]
                ),
                "retained_model_column_count": len(
                    result.zone_data.model_context_covariates.columns
                ),
                "retained_known_future_count": len(
                    result.zone_data.known_future_columns
                ),
            }
            row.update(flatten_metrics("native_", result.metrics_native))
            row.update(flatten_metrics("baseline_", baseline_metrics))
            row.update(flatten_metrics("full_", full_metrics))
            rows.append(row)

            output_csv = root / "selection_results.csv"
            pd.DataFrame(rows).to_csv(output_csv, index=False)
            with (scenario_root / "selection_scenario.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    {**scenario, "fold_asof": fold_asof},
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )

    print((root / "selection_results.csv").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
