"""Strict causal covariate contract for the governed Kalman overlay.

The contract deliberately separates raw inputs, deterministic transformations
and candidate feature groups.  This prevents accidental aggregation of
incompatible units (for example GW, degrees Celsius and W/m2) and makes the
same recipe reusable by the auxiliary laboratory and the live exporter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


BASE_RESIDUAL_LOAD_COVARIATES: tuple[str, ...] = (
    "fr_residual_load_fcst",
    "de_residual_load_fcst",
    "be_residual_load_fcst",
    "nl_residual_load_fcst",
    "es_residual_load_fcst",
)
SUPPORTED_COVARIATE_GROUPS = frozenset(
    {
        "market",
        "weather",
        "renewables",
        "fundamentals",
        "fuel",
        "market_weather",
        "market_weather_fuel",
    }
)
SUPPORTED_DERIVED_KINDS = frozenset(
    {
        "ramp",
        "heating_degree",
        "cooling_degree",
        "mean",
        "spread",
        "difference",
    }
)
SUPPORTED_HISTORY_MISSING_POLICIES = frozenset({"neutral", "complete_trailing"})
MAX_CANDIDATE_FEATURES = 64
_SAFE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{1,127}$")
_FORBIDDEN_TOKENS = (
    "storm",
    "mkonline",
    "oracle",
    "observed",
    "actual",
    "target",
)


class KalmanCovariateError(ValueError):
    """Raised when the exogenous feature contract is unsafe or ambiguous."""


@dataclass(frozen=True)
class DerivedCovariateSpec:
    """One deterministic transformation known at the day-ahead cutoff."""

    name: str
    kind: str
    sources: tuple[str, ...]
    periods: int | None = None
    threshold: float | None = None

    def validate(self, *, input_columns: Sequence[str]) -> None:
        if not _SAFE_NAME.fullmatch(self.name):
            raise KalmanCovariateError(
                f"Nom de covariable derivee invalide: {self.name!r}."
            )
        kind = str(self.kind)
        if kind not in SUPPORTED_DERIVED_KINDS:
            raise KalmanCovariateError(
                f"Transformation Kalman inconnue pour {self.name}: {kind!r}."
            )
        missing = sorted(set(self.sources).difference(input_columns))
        if missing:
            raise KalmanCovariateError(
                f"{self.name}: sources brutes absentes: {missing}."
            )
        expected_sources = {
            "ramp": 1,
            "heating_degree": 1,
            "cooling_degree": 1,
            "difference": 2,
        }
        if kind in expected_sources and len(self.sources) != expected_sources[kind]:
            raise KalmanCovariateError(
                f"{self.name}: {kind} exige {expected_sources[kind]} source(s)."
            )
        if kind in {"mean", "spread"} and len(self.sources) < 2:
            raise KalmanCovariateError(
                f"{self.name}: {kind} exige au moins deux sources de meme unite."
            )
        if kind == "ramp":
            if isinstance(self.periods, bool) or not isinstance(self.periods, int):
                raise KalmanCovariateError(
                    f"{self.name}: periods doit etre un entier positif."
                )
            if self.periods < 1 or self.periods > 168:
                raise KalmanCovariateError(
                    f"{self.name}: periods doit appartenir a [1, 168]."
                )
        elif self.periods is not None:
            raise KalmanCovariateError(
                f"{self.name}: periods est reserve a la transformation ramp."
            )
        if kind in {"heating_degree", "cooling_degree"}:
            if self.threshold is None or not np.isfinite(float(self.threshold)):
                raise KalmanCovariateError(
                    f"{self.name}: threshold doit etre fini."
                )
        elif self.threshold is not None:
            raise KalmanCovariateError(
                f"{self.name}: threshold est reserve aux degree-days."
            )


def _default_derived() -> tuple[DerivedCovariateSpec, ...]:
    return (
        DerivedCovariateSpec(
            name="residual_load_mean",
            kind="mean",
            sources=BASE_RESIDUAL_LOAD_COVARIATES,
        ),
        DerivedCovariateSpec(
            name="residual_load_spread",
            kind="spread",
            sources=BASE_RESIDUAL_LOAD_COVARIATES,
        ),
    )


def _default_groups() -> Mapping[str, tuple[str, ...]]:
    columns = (
        *BASE_RESIDUAL_LOAD_COVARIATES,
        "residual_load_mean",
        "residual_load_spread",
    )
    return {"market": columns, "fundamentals": columns}


@dataclass(frozen=True)
class KalmanCovariateConfig:
    """Raw inputs, transformations and explicit candidate feature groups."""

    input_columns: tuple[str, ...] = BASE_RESIDUAL_LOAD_COVARIATES
    groups: Mapping[str, tuple[str, ...]] = field(default_factory=_default_groups)
    derived: tuple[DerivedCovariateSpec, ...] = field(default_factory=_default_derived)
    history_missing_policy: str = "neutral"
    minimum_history_coverage: float = 0.0
    require_future_complete: bool = True

    @property
    def available_columns(self) -> tuple[str, ...]:
        return (*self.input_columns, *(item.name for item in self.derived))

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                column
                for group in self.groups.values()
                for column in group
            )
        )

    def validate(self) -> None:
        inputs = tuple(map(str, self.input_columns))
        if not inputs or len(inputs) != len(set(inputs)):
            raise KalmanCovariateError(
                "input_columns doit contenir des noms uniques non vides."
            )
        for name in inputs:
            _validate_safe_covariate_name(name)
        derived_names = tuple(item.name for item in self.derived)
        if len(derived_names) != len(set(derived_names)):
            raise KalmanCovariateError("Les covariables derivees doivent etre uniques.")
        overlap = sorted(set(inputs).intersection(derived_names))
        if overlap:
            raise KalmanCovariateError(
                f"Covariables brutes et derivees en conflit: {overlap}."
            )
        for item in self.derived:
            item.validate(input_columns=inputs)
            _validate_safe_covariate_name(item.name)
        groups = {
            str(name): tuple(map(str, values))
            for name, values in self.groups.items()
        }
        unknown_groups = sorted(set(groups).difference(SUPPORTED_COVARIATE_GROUPS))
        if unknown_groups:
            raise KalmanCovariateError(
                f"Groupes de covariables inconnus: {unknown_groups}."
            )
        if "market" not in groups or not groups["market"]:
            raise KalmanCovariateError("Le groupe market doit etre non vide.")
        available = set(self.available_columns)
        for group_name, columns in groups.items():
            if not columns or len(columns) != len(set(columns)):
                raise KalmanCovariateError(
                    f"Le groupe {group_name} doit contenir des colonnes uniques."
                )
            missing = sorted(set(columns).difference(available))
            if missing:
                raise KalmanCovariateError(
                    f"Groupe {group_name}: colonnes absentes: {missing}."
                )
            # Six dimensions are always added by an exogenous linear candidate:
            # intercept, two harmonics and three upstream-market descriptors.
            if len(columns) + 6 > MAX_CANDIDATE_FEATURES:
                raise KalmanCovariateError(
                    f"Groupe {group_name}: {len(columns) + 6} features Kalman, "
                    f"maximum autorise={MAX_CANDIDATE_FEATURES}."
                )
        if self.history_missing_policy not in SUPPORTED_HISTORY_MISSING_POLICIES:
            raise KalmanCovariateError(
                "history_missing_policy doit valoir neutral ou complete_trailing."
            )
        coverage = float(self.minimum_history_coverage)
        if not np.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            raise KalmanCovariateError(
                "minimum_history_coverage doit appartenir a [0, 1]."
            )
        if not isinstance(self.require_future_complete, bool):
            raise KalmanCovariateError("require_future_complete doit etre booleen.")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["groups"] = {
            str(name): list(columns) for name, columns in self.groups.items()
        }
        value["input_columns"] = list(self.input_columns)
        return value

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> "KalmanCovariateConfig":
        if raw is None:
            result = cls()
            result.validate()
            return result
        if not isinstance(raw, Mapping):
            raise KalmanCovariateError("covariates doit etre un objet YAML.")
        allowed = {
            "input_columns",
            "groups",
            "derived",
            "history_missing_policy",
            "minimum_history_coverage",
            "require_future_complete",
        }
        unknown = sorted(set(raw).difference(allowed))
        if unknown:
            raise KalmanCovariateError(
                f"Champs covariates inconnus: {unknown}."
            )
        inputs_raw = raw.get("input_columns", BASE_RESIDUAL_LOAD_COVARIATES)
        inputs = _string_sequence(inputs_raw, name="covariates.input_columns")
        derived_raw = raw.get("derived")
        if derived_raw is None:
            derived = _default_derived() if inputs == BASE_RESIDUAL_LOAD_COVARIATES else ()
        else:
            if not isinstance(derived_raw, Mapping):
                raise KalmanCovariateError("covariates.derived doit etre un objet YAML.")
            derived = tuple(
                _derived_from_mapping(str(name), value)
                for name, value in derived_raw.items()
            )
        groups_raw = raw.get("groups")
        if groups_raw is None:
            if (
                inputs == BASE_RESIDUAL_LOAD_COVARIATES
                and derived == _default_derived()
            ):
                groups = _default_groups()
            else:
                all_columns = (*inputs, *(item.name for item in derived))
                groups = {
                    "market": all_columns,
                    "fundamentals": all_columns,
                }
        else:
            if not isinstance(groups_raw, Mapping):
                raise KalmanCovariateError("covariates.groups doit etre un objet YAML.")
            groups = {
                str(name): _string_sequence(
                    values,
                    name=f"covariates.groups.{name}",
                )
                for name, values in groups_raw.items()
            }
        result = cls(
            input_columns=inputs,
            groups=groups,
            derived=derived,
            history_missing_policy=str(
                raw.get("history_missing_policy", "neutral")
            ),
            minimum_history_coverage=float(
                raw.get("minimum_history_coverage", 0.0)
            ),
            require_future_complete=raw.get("require_future_complete", True),
        )
        result.validate()
        return result


def _validate_safe_covariate_name(name: str) -> None:
    if not _SAFE_NAME.fullmatch(str(name)):
        raise KalmanCovariateError(f"Nom de covariable invalide: {name!r}.")
    lowered = str(name).casefold()
    if any(token in lowered for token in _FORBIDDEN_TOKENS):
        raise KalmanCovariateError(
            f"Covariable interdite pour prevenir une fuite/concurrence: {name!r}."
        )


def _string_sequence(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise KalmanCovariateError(f"{name} doit etre une liste.")
    values = tuple(str(item).strip() for item in value)
    if not values or any(not item for item in values) or len(values) != len(set(values)):
        raise KalmanCovariateError(f"{name} doit contenir des noms uniques non vides.")
    return values


def _derived_from_mapping(name: str, raw: Any) -> DerivedCovariateSpec:
    if not isinstance(raw, Mapping):
        raise KalmanCovariateError(f"covariates.derived.{name} doit etre un objet YAML.")
    kind = str(raw.get("kind", "")).strip()
    allowed = {"kind", "source", "sources", "periods", "threshold"}
    unknown = sorted(set(raw).difference(allowed))
    if unknown:
        raise KalmanCovariateError(
            f"covariates.derived.{name}: champs inconnus: {unknown}."
        )
    if "source" in raw and "sources" in raw:
        raise KalmanCovariateError(
            f"covariates.derived.{name}: utilisez source ou sources, pas les deux."
        )
    if "source" in raw:
        sources = (str(raw["source"]).strip(),)
    else:
        sources = _string_sequence(
            raw.get("sources", ()),
            name=f"covariates.derived.{name}.sources",
        )
    result = DerivedCovariateSpec(
        name=name,
        kind=kind,
        sources=sources,
        periods=raw.get("periods"),
        threshold=(
            None if raw.get("threshold") is None else float(raw["threshold"])
        ),
    )
    return result


def materialize_kalman_covariates(
    raw: pd.DataFrame,
    config: KalmanCovariateConfig | None = None,
    *,
    timezone: str = "UTC",
) -> pd.DataFrame:
    """Return numeric raw+derived covariates without mutating the input."""

    selected = config or KalmanCovariateConfig()
    selected.validate()
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise KalmanCovariateError("Le frame de covariables doit etre non vide.")
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise KalmanCovariateError("Les covariables doivent avoir un DatetimeIndex.")
    if raw.index.tz is None or raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise KalmanCovariateError(
            "La timeline des covariables doit etre aware, unique et croissante."
        )
    missing = sorted(set(selected.input_columns).difference(raw.columns))
    if missing:
        raise KalmanCovariateError(f"Covariables brutes absentes: {missing}.")
    output = raw.loc[:, list(selected.input_columns)].apply(
        pd.to_numeric,
        errors="coerce",
    ).astype(float)
    if bool(np.isinf(output.to_numpy(dtype=float)).any()):
        raise KalmanCovariateError("Les covariables contiennent des valeurs infinies.")
    structural_missing: dict[str, int] = {}
    local_days = pd.Index(output.index.tz_convert(timezone).date)
    for item in selected.derived:
        if item.kind == "ramp":
            periods = int(item.periods)
            source = output[item.sources[0]]
            ramp = source.groupby(local_days).diff(periods)
            position = source.groupby(local_days).cumcount()
            structural = position < periods
            # Zero is a structural boundary value only when the source itself
            # is known.  A genuinely missing forecast must remain missing and
            # be caught by coverage/future-completeness checks.
            ramp.loc[structural & source.notna()] = 0.0
            output[item.name] = ramp
            structural_missing[item.name] = int(
                (structural & source.notna()).sum()
            )
        elif item.kind == "heating_degree":
            output[item.name] = (
                float(item.threshold) - output[item.sources[0]]
            ).clip(lower=0.0)
        elif item.kind == "cooling_degree":
            output[item.name] = (
                output[item.sources[0]] - float(item.threshold)
            ).clip(lower=0.0)
        elif item.kind == "mean":
            output[item.name] = output.loc[:, list(item.sources)].mean(
                axis=1,
                skipna=False,
            )
        elif item.kind == "spread":
            block = output.loc[:, list(item.sources)]
            output[item.name] = block.max(axis=1, skipna=False) - block.min(
                axis=1,
                skipna=False,
            )
        elif item.kind == "difference":
            output[item.name] = output[item.sources[0]] - output[item.sources[1]]
        else:  # pragma: no cover - validated above.
            raise AssertionError(item.kind)
    output.attrs["structural_ramp_hours"] = structural_missing
    return output


def covariate_coverage(
    frame: pd.DataFrame,
    config: KalmanCovariateConfig,
) -> pd.DataFrame:
    """Auditable finite coverage for every raw and derived feature."""

    rows: list[dict[str, Any]] = []
    feature_columns = set(config.feature_columns)
    input_columns = set(config.input_columns)
    for column in config.available_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        rows.append(
            {
                "column": column,
                "role": (
                    "input+feature"
                    if column in input_columns and column in feature_columns
                    else "input"
                    if column in input_columns
                    else "derived+feature"
                    if column in feature_columns
                    else "derived"
                ),
                "finite_hours": int(finite.sum()),
                "total_hours": int(len(values)),
                "coverage": float(finite.mean()),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "BASE_RESIDUAL_LOAD_COVARIATES",
    "DerivedCovariateSpec",
    "MAX_CANDIDATE_FEATURES",
    "KalmanCovariateConfig",
    "KalmanCovariateError",
    "SUPPORTED_COVARIATE_GROUPS",
    "SUPPORTED_DERIVED_KINDS",
    "covariate_coverage",
    "materialize_kalman_covariates",
]
