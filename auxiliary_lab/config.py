"""Strict, small YAML contract for the auxiliary-model laboratory."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
import inspect
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import yaml

from chronos2_hourly.kalman_covariates import (
    KalmanCovariateConfig,
    KalmanCovariateError,
)
from chronos2_hourly.kalman_residual import (
    EXOGENOUS_FILTER_GROUPS,
    SUPPORTED_FILTER_KINDS,
    KalmanResidualConfig,
)
from chronos2_hourly.models.residual_corrector import ResidualCorrector


SCHEMA_VERSION = 1
MODEL_NAMES = ("residual_corrector", "mkonline_blend", "kalman")
SUPPORTED_METRICS = (
    "mae",
    "rmse",
    "bias",
    "pinball_q10",
    "pinball_q50",
    "pinball_q90",
    "coverage80",
    "interval_width80",
    "mean_observed_price",
    "mean_forecast_price",
    "mean_price_error",
)
_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{2,79}$")
_RESIDUAL_PARAMETERS = frozenset(
    set(inspect.signature(ResidualCorrector.__init__).parameters)
    .difference({"self", "feature_builder"})
)
_KALMAN_PARAMETERS = frozenset(field.name for field in dataclass_fields(KalmanResidualConfig))


class AuxiliaryLabConfigError(ValueError):
    """Raised when the lab contract is ambiguous or unsafe."""


@dataclass(frozen=True)
class SplitConfig:
    validation_days: int
    test_days: int
    minimum_training_days: int
    horizon_hours: int | None


@dataclass(frozen=True)
class ModelConfig:
    enabled: bool
    fixed_parameters: Mapping[str, Any]
    parameter_grid: Mapping[str, tuple[Any, ...]]
    options: Mapping[str, Any]


@dataclass(frozen=True)
class KalmanCovariateSource:
    """One immutable local table mapped into the Kalman raw-input contract."""

    path: Path
    timestamp_column: str
    columns: Mapping[str, str]
    origin_column: str
    revision_column: str | None
    cutoff_column: str | None
    cutoff_time: str


@dataclass(frozen=True)
class AuxiliaryLabConfig:
    schema_version: int
    experiment_id: str
    source_path: Path
    project_root: Path
    source_run: Path
    output_directory: Path
    timezone: str
    split: SplitConfig
    objective: str
    metrics: tuple[str, ...]
    random_seed: int
    report_enabled: bool
    report_embed_plotly: bool
    models: Mapping[str, ModelConfig]

    @property
    def enabled_models(self) -> tuple[str, ...]:
        return tuple(name for name in MODEL_NAMES if self.models[name].enabled)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AuxiliaryLabConfigError(f"{name} doit etre un objet YAML.")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
    name: str,
) -> None:
    missing = sorted(required.difference(value))
    unknown = sorted(set(value).difference(required | optional))
    if missing or unknown:
        raise AuxiliaryLabConfigError(
            f"Schema invalide pour {name}: missing={missing}, unknown={unknown}."
        )


def _positive_int(value: Any, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AuxiliaryLabConfigError(f"{name} doit etre un entier >= {minimum}.")
    return int(value)


def _path_inside(
    value: Any,
    *,
    base: Path,
    root: Path,
    name: str,
    must_exist: bool,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AuxiliaryLabConfigError(f"{name} doit etre un chemin non vide.")
    raw = Path(value.strip()).expanduser()
    path = (raw if raw.is_absolute() else base / raw).resolve()
    if not path.is_relative_to(root.resolve()):
        raise AuxiliaryLabConfigError(f"{name} doit rester sous {root.resolve()}.")
    if must_exist and not path.is_dir():
        raise AuxiliaryLabConfigError(f"{name} est introuvable: {path}.")
    return path


def _file_inside(
    value: Any,
    *,
    base: Path,
    root: Path,
    name: str,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AuxiliaryLabConfigError(f"{name} doit etre un chemin non vide.")
    raw = Path(value.strip()).expanduser()
    path = (raw if raw.is_absolute() else base / raw).resolve()
    if not path.is_relative_to(root.resolve()):
        raise AuxiliaryLabConfigError(f"{name} doit rester sous {root.resolve()}.")
    if not path.is_file():
        raise AuxiliaryLabConfigError(f"{name} est introuvable: {path}.")
    lowered = path.name.casefold()
    if not (
        lowered.endswith(".csv")
        or lowered.endswith(".csv.gz")
        or lowered.endswith(".parquet")
        or lowered.endswith(".pq")
    ):
        raise AuxiliaryLabConfigError(
            f"{name} doit etre un CSV/CSV.GZ/Parquet: {path}."
        )
    return path


def _kalman_additional_sources(
    value: Any,
    *,
    project_root: Path,
) -> tuple[KalmanCovariateSource, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AuxiliaryLabConfigError(
            "models.kalman.additional_sources doit etre une liste."
        )
    sources: list[KalmanCovariateSource] = []
    all_aliases: set[str] = set()
    forbidden_tokens = ("storm", "mkonline", "oracle", "observed", "actual", "target")
    for index, item in enumerate(value):
        name = f"models.kalman.additional_sources[{index}]"
        raw = _mapping(item, name=name)
        _exact_keys(
            raw,
            required={"path", "timestamp_column", "columns", "origin_column"},
            optional={"revision_column", "cutoff_column", "cutoff_time"},
            name=name,
        )
        timestamp_column = str(raw["timestamp_column"]).strip()
        if not timestamp_column:
            raise AuxiliaryLabConfigError(f"{name}.timestamp_column doit etre non vide.")
        origin_column = str(raw["origin_column"]).strip()
        revision_column = (
            str(raw["revision_column"]).strip()
            if raw.get("revision_column") is not None
            else None
        )
        cutoff_column = (
            str(raw["cutoff_column"]).strip()
            if raw.get("cutoff_column") is not None
            else None
        )
        cutoff_time = str(raw.get("cutoff_time", "08:00")).strip()
        if not origin_column or not cutoff_time:
            raise AuxiliaryLabConfigError(
                f"{name}.origin_column/cutoff_time doit etre non vide."
            )
        if revision_column == "" or cutoff_column == "":
            raise AuxiliaryLabConfigError(
                f"{name}: colonne de provenance vide."
            )
        columns_raw = _mapping(raw["columns"], name=f"{name}.columns")
        if not columns_raw:
            raise AuxiliaryLabConfigError(f"{name}.columns ne doit pas etre vide.")
        columns: dict[str, str] = {}
        # Direction unique partagee avec le sidecar live:
        # alias_modele: colonne_du_fichier.
        for alias, source_column in columns_raw.items():
            alias_name = str(alias).strip()
            source_name = str(source_column).strip()
            if not source_name or not alias_name:
                raise AuxiliaryLabConfigError(
                    f"{name}.columns contient un nom vide."
                )
            for candidate in (source_name, alias_name):
                lowered = candidate.casefold()
                if any(token in lowered for token in forbidden_tokens):
                    raise AuxiliaryLabConfigError(
                        f"{name}: colonne interdite pour prevenir une fuite: {candidate!r}."
                    )
            if alias_name in all_aliases:
                raise AuxiliaryLabConfigError(
                    f"Alias Kalman duplique entre sources: {alias_name!r}."
                )
            columns[alias_name] = source_name
            all_aliases.add(alias_name)
        sources.append(
            KalmanCovariateSource(
                path=_file_inside(
                    raw["path"],
                    base=project_root,
                    root=project_root,
                    name=f"{name}.path",
                ),
                timestamp_column=timestamp_column,
                columns=columns,
                origin_column=origin_column,
                revision_column=revision_column,
                cutoff_column=cutoff_column,
                cutoff_time=cutoff_time,
            )
        )
    return tuple(sources)


def _normalise_grid(value: Any, *, name: str) -> dict[str, tuple[Any, ...]]:
    raw = _mapping(value, name=name)
    result: dict[str, tuple[Any, ...]] = {}
    for key, candidates in raw.items():
        if not isinstance(key, str) or not key.strip():
            raise AuxiliaryLabConfigError(f"{name}: nom de parametre invalide.")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise AuxiliaryLabConfigError(f"{name}.{key} doit etre une liste.")
        values = tuple(candidates)
        if not values:
            raise AuxiliaryLabConfigError(f"{name}.{key} ne doit pas etre vide.")
        result[key.strip()] = values
    return result


def _model_config(
    name: str,
    value: Any,
    *,
    project_root: Path,
) -> ModelConfig:
    raw = _mapping(value, name=f"models.{name}")
    common = {"enabled", "fixed_parameters", "parameter_grid"}
    option_keys = {
        "residual_corrector": {
            "base_model",
            "expert_models",
            "recipe",
            "components",
            "weight_candidates",
            "blend_max_abs_correction",
            "prequential_bridge",
        },
        "mkonline_blend": {"autonomous_model", "primary_column", "weight_step"},
        "kalman": {
            "upstream_model",
            "covariates",
            "additional_sources",
            "training_lookback_days",
            "rolling_refit_workers",
        },
    }[name]
    _exact_keys(
        raw,
        required=common,
        optional=option_keys,
        name=f"models.{name}",
    )
    enabled = raw["enabled"]
    if not isinstance(enabled, bool):
        raise AuxiliaryLabConfigError(f"models.{name}.enabled doit etre booleen.")
    fixed = dict(_mapping(raw["fixed_parameters"], name=f"models.{name}.fixed_parameters"))
    grid = _normalise_grid(raw["parameter_grid"], name=f"models.{name}.parameter_grid")
    overlap = sorted(set(fixed).intersection(grid))
    if overlap:
        raise AuxiliaryLabConfigError(
            f"models.{name}: parametres a la fois fixes et en grille: {overlap}."
        )
    options = {key: raw[key] for key in option_keys if key in raw}
    if name == "residual_corrector":
        base = options.get("base_model", "chronos2")
        experts = options.get("expert_models", ["chronos2"])
        if not isinstance(base, str) or not base.strip():
            raise AuxiliaryLabConfigError("base_model doit etre non vide.")
        if not isinstance(experts, Sequence) or isinstance(experts, (str, bytes)):
            raise AuxiliaryLabConfigError("expert_models doit etre une liste.")
        expert_names = tuple(str(item).strip() for item in experts)
        if not expert_names or any(not item for item in expert_names):
            raise AuxiliaryLabConfigError("expert_models contient une valeur vide.")
        forbidden = [
            value
            for value in (base, *expert_names)
            if any(token in str(value).casefold() for token in ("storm", "mkonline"))
        ]
        if forbidden:
            raise AuxiliaryLabConfigError(
                f"Storm/MKOnline sont interdits comme inputs du correcteur autonome: {forbidden}."
            )
        recipe = str(options.get("recipe", "single")).strip().lower()
        if recipe not in {"single", "blend"}:
            raise AuxiliaryLabConfigError("recipe doit valoir single ou blend.")
        components_raw = options.get("components", {})
        components = {
            str(component_name): dict(
                _mapping(component_value, name=f"models.{name}.components.{component_name}")
            )
            for component_name, component_value in _mapping(
                components_raw, name=f"models.{name}.components"
            ).items()
        }
        weight_candidates_raw = options.get("weight_candidates", [])
        if not isinstance(weight_candidates_raw, Sequence) or isinstance(
            weight_candidates_raw, (str, bytes)
        ):
            raise AuxiliaryLabConfigError("weight_candidates doit etre une liste.")
        weight_candidates: list[dict[str, float]] = []
        for index, candidate in enumerate(weight_candidates_raw):
            weights = {
                str(key): float(value)
                for key, value in _mapping(
                    candidate, name=f"weight_candidates[{index}]"
                ).items()
            }
            if set(weights) != set(components):
                raise AuxiliaryLabConfigError(
                    "Chaque weight_candidate doit couvrir exactement components."
                )
            if any(not np.isfinite(value) or value < 0.0 for value in weights.values()):
                raise AuxiliaryLabConfigError("Les poids residuels doivent etre finis et >= 0.")
            if not np.isclose(sum(weights.values()), 1.0, rtol=0.0, atol=1e-12):
                raise AuxiliaryLabConfigError("La somme des poids residuels doit valoir 1.")
            weight_candidates.append(weights)
        if recipe == "blend" and (not components or not weight_candidates):
            raise AuxiliaryLabConfigError(
                "La recette blend exige components et weight_candidates non vides."
            )
        blend_clip = options.get("blend_max_abs_correction", 40.0)
        if blend_clip is not None:
            blend_clip = float(blend_clip)
            if not np.isfinite(blend_clip) or blend_clip <= 0.0:
                raise AuxiliaryLabConfigError(
                    "blend_max_abs_correction doit etre positif ou null."
                )
        bridge_raw = _mapping(
            options.get("prequential_bridge", {}),
            name="models.residual_corrector.prequential_bridge",
        )
        _exact_keys(
            bridge_raw,
            required=set(),
            optional={
                "refit_cadence_days",
                "training_lookback_days",
                "cold_start_policy",
                "history_prefix_path",
            },
            name="models.residual_corrector.prequential_bridge",
        )
        refit_cadence_days = _positive_int(
            bridge_raw.get("refit_cadence_days", 28),
            name=(
                "models.residual_corrector.prequential_bridge."
                "refit_cadence_days"
            ),
        )
        bridge_lookback_days = _positive_int(
            bridge_raw.get("training_lookback_days", 365),
            name=(
                "models.residual_corrector.prequential_bridge."
                "training_lookback_days"
            ),
            minimum=2,
        )
        cold_start_policy = str(
            bridge_raw.get("cold_start_policy", "identity")
        ).strip().lower()
        if cold_start_policy != "identity":
            raise AuxiliaryLabConfigError(
                "models.residual_corrector.prequential_bridge."
                "cold_start_policy doit valoir identity."
            )
        history_prefix_path = (
            _file_inside(
                bridge_raw["history_prefix_path"],
                base=project_root,
                root=project_root,
                name=(
                    "models.residual_corrector.prequential_bridge."
                    "history_prefix_path"
                ),
            )
            if bridge_raw.get("history_prefix_path") is not None
            else None
        )
        options = {
            "base_model": base.strip(),
            "expert_models": expert_names,
            "recipe": recipe,
            "components": components,
            "weight_candidates": tuple(weight_candidates),
            "blend_max_abs_correction": blend_clip,
            "prequential_bridge": {
                "refit_cadence_days": refit_cadence_days,
                "training_lookback_days": bridge_lookback_days,
                "cold_start_policy": cold_start_policy,
                "history_prefix_path": history_prefix_path,
            },
        }
    elif name == "mkonline_blend":
        step = float(options.get("weight_step", 0.01))
        if not 0.0 < step <= 1.0:
            raise AuxiliaryLabConfigError("weight_step doit appartenir a ]0,1].")
        options = {
            "autonomous_model": str(options.get("autonomous_model", "residual_corrected")),
            "primary_column": str(options.get("primary_column", "mkonline_primary__q50")),
            "weight_step": step,
        }
    else:
        upstream = str(options.get("upstream_model", "residual_corrected"))
        if any(token in upstream.casefold() for token in ("storm", "mkonline")):
            raise AuxiliaryLabConfigError(
                "Storm/MKOnline sont interdits comme upstream Kalman."
            )
        try:
            covariates = KalmanCovariateConfig.from_mapping(
                options.get("covariates")
            )
        except KalmanCovariateError as exc:
            raise AuxiliaryLabConfigError(str(exc)) from exc
        additional_sources = _kalman_additional_sources(
            options.get("additional_sources"),
            project_root=project_root,
        )
        source_aliases = {
            alias
            for source in additional_sources
            for alias in source.columns
        }
        unused_aliases = sorted(source_aliases.difference(covariates.input_columns))
        if unused_aliases:
            raise AuxiliaryLabConfigError(
                "additional_sources fournit des alias absents de "
                f"covariates.input_columns: {unused_aliases}."
            )
        training_lookback_raw = options.get("training_lookback_days")
        training_lookback_days = (
            None
            if training_lookback_raw is None
            else _positive_int(
                training_lookback_raw,
                name="models.kalman.training_lookback_days",
                minimum=2,
            )
        )
        rolling_refit_workers = _positive_int(
            options.get("rolling_refit_workers", 1),
            name="models.kalman.rolling_refit_workers",
        )
        if rolling_refit_workers > 8:
            raise AuxiliaryLabConfigError(
                "models.kalman.rolling_refit_workers doit etre <= 8."
            )
        options = {
            "upstream_model": upstream,
            "covariate_config": covariates,
            "additional_sources": additional_sources,
            "training_lookback_days": training_lookback_days,
            "rolling_refit_workers": rolling_refit_workers,
        }
    if name == "residual_corrector":
        invalid_fixed = sorted(set(fixed).difference(_RESIDUAL_PARAMETERS))
        invalid_components = {
            component: sorted(set(parameters).difference(_RESIDUAL_PARAMETERS))
            for component, parameters in options["components"].items()
            if set(parameters).difference(_RESIDUAL_PARAMETERS)
        }
        invalid_grid: list[str] = []
        for parameter in grid:
            if parameter.startswith("components."):
                parts = parameter.split(".", 2)
                if (
                    options["recipe"] != "blend"
                    or len(parts) != 3
                    or parts[1] not in options["components"]
                    or parts[2] not in _RESIDUAL_PARAMETERS
                ):
                    invalid_grid.append(parameter)
            elif parameter not in _RESIDUAL_PARAMETERS:
                invalid_grid.append(parameter)
        if invalid_fixed or invalid_components or invalid_grid:
            raise AuxiliaryLabConfigError(
                "Parametres residuels inconnus: "
                f"fixed={invalid_fixed}, components={invalid_components}, "
                f"grid={sorted(invalid_grid)}."
            )
    elif name == "mkonline_blend":
        invalid = sorted(
            (set(fixed) | set(grid)).difference({"max_abs_shift_eur_mwh"})
        )
        if invalid:
            raise AuxiliaryLabConfigError(
                f"Parametres MKOnline inconnus: {invalid}."
            )
    else:
        invalid = sorted((set(fixed) | set(grid)).difference(_KALMAN_PARAMETERS))
        if invalid:
            raise AuxiliaryLabConfigError(f"Parametres Kalman inconnus: {invalid}.")
        candidate_options: list[Sequence[Any]] = []
        if "candidate_kinds" in fixed:
            candidate_options.append(fixed["candidate_kinds"])
        if "candidate_kinds" in grid:
            candidate_options.extend(grid["candidate_kinds"])
        candidate_names: list[str] = []
        for candidate_list in candidate_options:
            if not isinstance(candidate_list, Sequence) or isinstance(
                candidate_list, (str, bytes)
            ):
                raise AuxiliaryLabConfigError(
                    "candidate_kinds doit etre une liste de familles Kalman."
                )
            candidate_names.extend(str(candidate) for candidate in candidate_list)
        unknown_candidates = sorted(
            set(candidate_names).difference(SUPPORTED_FILTER_KINDS)
        )
        if unknown_candidates:
            raise AuxiliaryLabConfigError(
                f"Famille Kalman inconnue: {unknown_candidates}."
            )
        for candidate_name in candidate_names:
            group = EXOGENOUS_FILTER_GROUPS.get(candidate_name)
            if group is not None and group not in options["covariate_config"].groups:
                raise AuxiliaryLabConfigError(
                    f"{candidate_name} exige le groupe covariates.groups.{group}."
                )
    return ModelConfig(
        enabled=enabled,
        fixed_parameters=fixed,
        parameter_grid=grid,
        options=options,
    )


def _discover_project_root(source_path: Path) -> Path:
    for candidate in source_path.parents:
        if (candidate / "chronos2_hourly").is_dir() and (
            candidate / "Forecast.ps1"
        ).is_file():
            return candidate.resolve()
    raise AuxiliaryLabConfigError(
        "Racine projet introuvable; fournissez project_root explicitement."
    )


def load_lab_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> AuxiliaryLabConfig:
    """Load and validate a lab YAML without creating any output."""

    source_path = Path(path).expanduser().resolve()
    try:
        payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AuxiliaryLabConfigError(f"Configuration illisible: {source_path}.") from exc
    raw = _mapping(payload, name="configuration")
    _exact_keys(
        raw,
        required={
            "schema_version",
            "experiment_id",
            "source_run",
            "output_directory",
            "timezone",
            "split",
            "objective",
            "metrics",
            "random_seed",
            "report",
            "models",
        },
        optional=set(),
        name="configuration",
    )
    if raw["schema_version"] != SCHEMA_VERSION:
        raise AuxiliaryLabConfigError(
            f"schema_version={raw['schema_version']!r}; attendu={SCHEMA_VERSION}."
        )
    experiment_id = str(raw["experiment_id"]).strip()
    if not _ID_RE.fullmatch(experiment_id):
        raise AuxiliaryLabConfigError(
            "experiment_id doit utiliser lettres minuscules, chiffres, '-' ou '_'."
        )
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else _discover_project_root(source_path)
    )
    source_run = _path_inside(
        raw["source_run"],
        base=root,
        root=root,
        name="source_run",
        must_exist=True,
    )
    experiments_root = (root / "runs" / "experiments" / "auxiliary_lab").resolve()
    output = _path_inside(
        raw["output_directory"],
        base=root,
        root=experiments_root,
        name="output_directory",
        must_exist=False,
    )
    if output == experiments_root:
        raise AuxiliaryLabConfigError("output_directory doit etre sous auxiliary_lab/.")
    timezone = str(raw["timezone"]).strip()
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise AuxiliaryLabConfigError(f"Timezone IANA invalide: {timezone!r}.") from exc

    split_raw = _mapping(raw["split"], name="split")
    _exact_keys(
        split_raw,
        required={"validation_days", "test_days", "minimum_training_days", "horizon_hours"},
        optional=set(),
        name="split",
    )
    horizon = split_raw["horizon_hours"]
    if horizon is not None:
        horizon = _positive_int(horizon, name="split.horizon_hours")
    split = SplitConfig(
        validation_days=_positive_int(split_raw["validation_days"], name="split.validation_days"),
        test_days=_positive_int(split_raw["test_days"], name="split.test_days"),
        minimum_training_days=_positive_int(
            split_raw["minimum_training_days"], name="split.minimum_training_days", minimum=2
        ),
        horizon_hours=horizon,
    )
    objective = str(raw["objective"]).strip()
    if objective not in {"mae", "rmse"}:
        raise AuxiliaryLabConfigError("objective doit valoir mae ou rmse.")
    metrics_raw = raw["metrics"]
    if not isinstance(metrics_raw, Sequence) or isinstance(metrics_raw, (str, bytes)):
        raise AuxiliaryLabConfigError("metrics doit etre une liste.")
    metrics = tuple(str(value).strip() for value in metrics_raw)
    if not metrics or len(metrics) != len(set(metrics)):
        raise AuxiliaryLabConfigError("metrics doit contenir des valeurs uniques.")
    unknown_metrics = sorted(set(metrics).difference(SUPPORTED_METRICS))
    if unknown_metrics:
        raise AuxiliaryLabConfigError(f"Metriques non supportees: {unknown_metrics}.")
    random_seed = _positive_int(raw["random_seed"], name="random_seed", minimum=0)

    report = _mapping(raw["report"], name="report")
    _exact_keys(report, required={"enabled", "embed_plotly"}, optional=set(), name="report")
    if not isinstance(report["enabled"], bool) or not isinstance(report["embed_plotly"], bool):
        raise AuxiliaryLabConfigError("report.enabled/embed_plotly doivent etre booleens.")

    raw_models = _mapping(raw["models"], name="models")
    _exact_keys(raw_models, required=set(MODEL_NAMES), optional=set(), name="models")
    models = {
        name: _model_config(name, raw_models[name], project_root=root)
        for name in MODEL_NAMES
    }
    if not any(model.enabled for model in models.values()):
        raise AuxiliaryLabConfigError("Au moins un modele auxiliaire doit etre active.")
    return AuxiliaryLabConfig(
        schema_version=SCHEMA_VERSION,
        experiment_id=experiment_id,
        source_path=source_path,
        project_root=root,
        source_run=source_run,
        output_directory=output,
        timezone=timezone,
        split=split,
        objective=objective,
        metrics=metrics,
        random_seed=random_seed,
        report_enabled=bool(report["enabled"]),
        report_embed_plotly=bool(report["embed_plotly"]),
        models=models,
    )


__all__ = [
    "AuxiliaryLabConfig",
    "AuxiliaryLabConfigError",
    "KalmanCovariateSource",
    "MODEL_NAMES",
    "ModelConfig",
    "SUPPORTED_METRICS",
    "SplitConfig",
    "load_lab_config",
]
