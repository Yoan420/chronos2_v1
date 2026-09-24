"""Read-only adapters from published Chronos-2 runs to lab datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.kalman_covariates import (
    KalmanCovariateConfig,
    KalmanCovariateError,
    covariate_coverage,
    materialize_kalman_covariates,
)

from .config import AuxiliaryLabConfig, KalmanCovariateSource, ModelConfig
from .metrics import DaySplit, validate_and_split_days


QUANTILES = ("q10", "q50", "q90")


@dataclass(frozen=True)
class ResidualDataset:
    X: pd.DataFrame
    actual: pd.Series
    base: pd.DataFrame
    experts: pd.DataFrame
    split: DaySplit
    base_model: str
    expert_models: tuple[str, ...]


@dataclass(frozen=True)
class BlendDataset:
    frame: pd.DataFrame
    split: DaySplit
    autonomous_model: str
    primary_column: str


@dataclass(frozen=True)
class KalmanDataset:
    history: pd.DataFrame
    covariates: pd.DataFrame
    split: DaySplit
    upstream_model: str
    covariate_config: KalmanCovariateConfig
    coverage: pd.DataFrame
    source_audit: tuple[MappingAudit, ...]


@dataclass(frozen=True)
class MappingAudit:
    path: str
    timestamp_column: str
    columns: Mapping[str, str]
    origin_column: str
    revision_column: str | None
    cutoff_column: str | None
    cutoff_time: str
    latest_origin_utc: str
    causality_violations: int
    source_hours: int
    matched_hours: int
    unmatched_source_hours: int


def _read_indexed(path: Path, *, timestamp_column: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.casefold() in {".parquet", ".pq"}:
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path)
    if timestamp_column not in raw:
        raise ValueError(f"{path}: colonne {timestamp_column!r} absente.")
    index = pd.DatetimeIndex(
        pd.to_datetime(raw.pop(timestamp_column), utc=True, errors="raise"),
        name="delivery_start_utc",
    )
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(f"{path}: timeline dupliquee ou non croissante.")
    if bool((index.minute != 0).any()) or bool((index.second != 0).any()):
        raise ValueError(f"{path}: timestamps non alignes sur l'heure.")
    raw.index = index
    return raw


def _read_external_indexed(path: Path, *, timestamp_column: str) -> pd.DataFrame:
    """Read an additional PIT source and reject timezone-naive timestamps."""

    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.casefold() in {".parquet", ".pq"}:
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path)
    if timestamp_column not in raw:
        raise ValueError(f"{path}: colonne {timestamp_column!r} absente.")
    timestamp_values = raw.pop(timestamp_column)
    parsed: list[pd.Timestamp] = []
    for position, value in enumerate(timestamp_values):
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: timestamp externe invalide a la ligne {position}."
            ) from exc
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError(
                f"{path}: chaque timestamp externe doit declarer un offset/tz "
                f"explicite (ligne {position})."
            )
        parsed.append(stamp.tz_convert("UTC"))
    index = pd.DatetimeIndex(parsed, name="delivery_start_utc")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(f"{path}: timeline dupliquee ou non croissante.")
    if bool((index.minute != 0).any()) or bool((index.second != 0).any()):
        raise ValueError(f"{path}: timestamps non alignes sur l'heure.")
    raw.index = index
    return raw


def _aware_utc_values(
    values: pd.Series,
    *,
    path: Path,
    column: str,
) -> pd.DatetimeIndex:
    parsed: list[pd.Timestamp] = []
    for position, value in enumerate(values):
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: {column} invalide a la ligne {position}."
            ) from exc
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError(
                f"{path}: {column} doit declarer un offset/tz explicite "
                f"(ligne {position})."
            )
        parsed.append(stamp.tz_convert("UTC"))
    return pd.DatetimeIndex(parsed)


def _expected_civil_cutoffs(
    index: pd.DatetimeIndex,
    *,
    timezone: str,
    cutoff_time: str,
) -> pd.DatetimeIndex:
    try:
        clock = pd.Timedelta(
            cutoff_time + ":00" if cutoff_time.count(":") == 1 else cutoff_time
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cutoff_time invalide: {cutoff_time!r}.") from exc
    local_days = index.tz_convert(timezone).normalize().tz_localize(None)
    return pd.DatetimeIndex(
        [
            (day - pd.Timedelta(days=1) + clock)
            .tz_localize(timezone)
            .tz_convert("UTC")
            for day in local_days
        ]
    )


def _validate_external_pit_provenance(
    external: pd.DataFrame,
    source: KalmanCovariateSource,
    *,
    timezone: str,
) -> tuple[pd.DatetimeIndex, int]:
    metadata_columns = {
        source.origin_column,
        *(item for item in (source.revision_column, source.cutoff_column) if item),
    }
    missing = sorted(metadata_columns.difference(external.columns))
    if missing:
        raise ValueError(
            f"{source.path}: colonnes de provenance PIT absentes: {missing}."
        )
    expected = _expected_civil_cutoffs(
        external.index,
        timezone=timezone,
        cutoff_time=source.cutoff_time,
    )
    origin = _aware_utc_values(
        external[source.origin_column],
        path=source.path,
        column=source.origin_column,
    )
    violations = origin > expected
    if source.revision_column:
        revision = _aware_utc_values(
            external[source.revision_column],
            path=source.path,
            column=source.revision_column,
        )
        violations = np.asarray(violations) | np.asarray(revision > expected)
    if source.cutoff_column:
        declared = _aware_utc_values(
            external[source.cutoff_column],
            path=source.path,
            column=source.cutoff_column,
        )
        mismatched = declared != expected
        if bool(np.asarray(mismatched).any()):
            first = int(np.flatnonzero(np.asarray(mismatched))[0])
            raise ValueError(
                f"{source.path}: cutoff declare different du cutoff civil "
                f"a la ligne {first}: declare={declared[first]}, "
                f"attendu={expected[first]}."
            )
    count = int(np.asarray(violations).sum())
    if count:
        first = int(np.flatnonzero(np.asarray(violations))[0])
        raise ValueError(
            f"{source.path}: {count} violation(s) PIT; origine/revision "
            f"posterieure au cutoff a la ligne {first}."
        )
    return origin, count


def _issued_history(source_run: Path) -> pd.DataFrame:
    """Combine the sealed OOF base with the latest issued Statistics overlay.

    The live Statistics file is deliberately limited to the reporting window,
    whereas the sealed OOF file may contain the older training prefix required
    by a FINAL365 experiment.  Values published in Statistics take precedence;
    the OOF base is retained before that window and for columns not materialised
    by a particular Statistics row.
    """

    statistics_path = source_run / "statistics_history_hourly.csv.gz"
    sealed_path = source_run / "backtest_hourly_oof.csv.gz"
    available = [path for path in (sealed_path, statistics_path) if path.is_file()]
    if not available:
        raise FileNotFoundError(
            f"{source_run}: backtest_hourly_oof.csv.gz ou "
            "statistics_history_hourly.csv.gz est requis."
        )
    if len(available) == 1:
        return _read_indexed(available[0], timestamp_column="delivery_start_utc")

    sealed = _read_indexed(sealed_path, timestamp_column="delivery_start_utc")
    statistics = _read_indexed(
        statistics_path,
        timestamp_column="delivery_start_utc",
    )
    columns = list(dict.fromkeys([*sealed.columns, *statistics.columns]))
    index = sealed.index.union(statistics.index).sort_values()
    combined = statistics.reindex(index=index, columns=columns).combine_first(
        sealed.reindex(index=index, columns=columns)
    )
    combined.index.name = "delivery_start_utc"
    return combined


def _numeric(frame: pd.DataFrame, columns: Iterable[str], *, name: str) -> pd.DataFrame:
    selected = list(columns)
    missing = sorted(set(selected).difference(frame))
    if missing:
        raise ValueError(f"{name}: colonnes absentes: {missing}.")
    result = frame.loc[:, selected].apply(pd.to_numeric, errors="coerce")
    if bool(np.isinf(result.to_numpy(dtype=float)).any()):
        raise ValueError(f"{name}: valeurs infinies.")
    return result.astype(float)


def join_kalman_additional_sources(
    base: pd.DataFrame,
    sources: Sequence[KalmanCovariateSource],
    *,
    timezone: str,
) -> tuple[pd.DataFrame, tuple[MappingAudit, ...]]:
    """Left-align local PIT sources without shrinking or imputing the base timeline."""

    result = base.copy()
    audits: list[MappingAudit] = []
    for source in sources:
        external = _read_external_indexed(
            source.path,
            timestamp_column=source.timestamp_column,
        )
        origins, causality_violations = _validate_external_pit_provenance(
            external,
            source,
            timezone=timezone,
        )
        missing = sorted(set(source.columns.values()).difference(external.columns))
        if missing:
            raise ValueError(
                f"{source.path}: colonnes externes absentes: {missing}."
            )
        aliases = tuple(source.columns)
        collisions = sorted(set(aliases).intersection(result.columns))
        if collisions:
            raise ValueError(
                "Collision de covariables Kalman; aucun ecrasement implicite: "
                f"{collisions}."
            )
        source_to_alias = {
            source_column: alias
            for alias, source_column in source.columns.items()
        }
        selected = external.loc[:, list(source.columns.values())].rename(
            columns=source_to_alias
        )
        selected = _numeric(
            selected,
            aliases,
            name=f"source Kalman {source.path}",
        )
        matched = int(selected.index.isin(result.index).sum())
        result = result.join(selected, how="left", validate="one_to_one", sort=False)
        audits.append(
            MappingAudit(
                path=str(source.path),
                timestamp_column=source.timestamp_column,
                columns=dict(source.columns),
                origin_column=source.origin_column,
                revision_column=source.revision_column,
                cutoff_column=source.cutoff_column,
                cutoff_time=source.cutoff_time,
                latest_origin_utc=origins.max().isoformat(),
                causality_violations=causality_violations,
                source_hours=int(len(selected)),
                matched_hours=matched,
                unmatched_source_hours=int(len(selected) - matched),
            )
        )
    if not result.index.equals(base.index):
        raise RuntimeError("La jointure externe a modifie la timeline Kalman de base.")
    return result, tuple(audits)


def load_kalman_covariate_frame(
    *,
    source_run: Path,
    model: ModelConfig,
    timezone: str,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[MappingAudit, ...]]:
    """Load raw+derived Kalman covariates under the resolved lab contract."""

    base = _read_indexed(
        source_run / "inputs" / "model_covariates_with_future.csv.gz",
        timestamp_column="timestamp",
    )
    raw, source_audit = join_kalman_additional_sources(
        base,
        tuple(model.options["additional_sources"]),
        timezone=timezone,
    )
    selected_config = model.options["covariate_config"]
    try:
        materialized = materialize_kalman_covariates(
            raw,
            selected_config,
            timezone=timezone,
        )
    except KalmanCovariateError as exc:
        raise ValueError(str(exc)) from exc
    return raw, materialized, source_audit


def _phase_coverage(
    materialized: pd.DataFrame,
    *,
    index: pd.DatetimeIndex,
    split: DaySplit,
    timezone: str,
    config: KalmanCovariateConfig,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for phase in ("train", "validation", "test"):
        mask = split.phase_mask(index, timezone=timezone, phase=phase)
        block = materialized.reindex(index[mask])
        coverage = covariate_coverage(block, config)
        coverage.insert(0, "phase", phase)
        rows.append(coverage)
    result = pd.concat(rows, ignore_index=True)
    feature_rows = result.loc[result["column"].isin(config.feature_columns)]
    insufficient = feature_rows.loc[
        feature_rows["coverage"] + 1e-12 < config.minimum_history_coverage
    ]
    if not insufficient.empty:
        details = ", ".join(
            f"{row.phase}/{row.column}={row.coverage:.3f}"
            for row in insufficient.itertuples(index=False)
        )
        raise ValueError(
            "Couverture historique Kalman inferieure a "
            f"minimum_history_coverage={config.minimum_history_coverage:.3f}: "
            f"{details}."
        )
    return result


def _trailing_complete(
    frame: pd.DataFrame,
    *,
    timezone: str,
    finite_columns: Iterable[str],
) -> pd.DataFrame:
    columns = list(finite_columns)
    numeric = _numeric(frame, columns, name="dataset")
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    valid = frame.loc[finite].copy()
    if valid.empty:
        raise ValueError("Aucune observation finie dans le dataset.")
    local_days = pd.Index(valid.index.tz_convert(timezone).date)
    complete: list[object] = []
    for day in local_days.unique():
        observed = valid.index[local_days == day]
        if observed.equals(local_delivery_day_index(day, timezone=timezone)):
            complete.append(day)
    if not complete:
        raise ValueError("Aucune journee locale physique complete.")
    trailing = [complete[-1]]
    for day in reversed(complete[:-1]):
        if pd.Timestamp(trailing[0]) - pd.Timestamp(day) != pd.Timedelta(days=1):
            break
        trailing.insert(0, day)
    mask = pd.Index(valid.index.tz_convert(timezone).date).isin(trailing)
    result = valid.loc[np.asarray(mask, dtype=bool)].copy()
    result_numeric = _numeric(result, columns, name="dataset complet")
    if not np.isfinite(result_numeric.to_numpy(dtype=float)).all():
        raise RuntimeError("La selection de jours complets contient des trous.")
    return result


def _residual_history_with_prefix(
    *,
    source_run: Path,
    model: ModelConfig,
    timezone: str,
) -> pd.DataFrame:
    issued = _issued_history(source_run)
    prefix_path = model.options["prequential_bridge"].get(
        "history_prefix_path"
    )
    if prefix_path is None:
        return issued
    if str(model.options["base_model"]) != "chronos2" or tuple(
        model.options["expert_models"]
    ) != ("chronos2",):
        raise ValueError(
            "history_prefix_path est un contrat Chronos q10/q50/q90 et exige "
            "base_model=chronos2, expert_models=[chronos2]."
        )

    prefix = _read_external_indexed(
        Path(prefix_path), timestamp_column="delivery_start_utc"
    )
    expected_columns = {
        "forecast_origin_utc",
        "q10",
        "q50",
        "q90",
        "actual",
    }
    if set(prefix.columns) != expected_columns:
        raise ValueError(
            f"{prefix_path}: schema prefixe Chronos invalide; "
            f"attendu={sorted(expected_columns)}, recu={sorted(prefix.columns)}."
        )
    numeric = prefix.loc[:, ["q10", "q50", "q90", "actual"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{prefix_path}: quantiles/actual non finis.")
    if bool(
        ((numeric["q10"] > numeric["q50"]) | (numeric["q50"] > numeric["q90"])).any()
    ):
        raise ValueError(f"{prefix_path}: quantiles Chronos croises.")
    origins = _aware_utc_values(
        prefix["forecast_origin_utc"],
        path=Path(prefix_path),
        column="forecast_origin_utc",
    )
    cutoffs = _expected_civil_cutoffs(
        prefix.index,
        timezone=timezone,
        cutoff_time="08:00",
    )
    mismatched_origins = np.asarray(origins != cutoffs)
    if bool(mismatched_origins.any()):
        first = int(np.flatnonzero(mismatched_origins)[0])
        raise ValueError(
            f"{prefix_path}: forecast_origin_utc different du cutoff civil "
            f"contractuel D-1 08:00 a la ligne {first}."
        )

    overlap = prefix.index.intersection(issued.index)
    mapped = pd.DataFrame(
        {
            "chronos2__q10": numeric["q10"],
            "chronos2__q50": numeric["q50"],
            "chronos2__q90": numeric["q90"],
            "actual": numeric["actual"],
            "forecast_origin_utc": origins,
        },
        index=prefix.index,
    )
    if len(overlap):
        issued_overlap = issued.reindex(overlap)
        prefix_overlap = mapped.reindex(overlap)
        issued_quantiles = issued_overlap.reindex(
            columns=["chronos2__q10", "chronos2__q50", "chronos2__q90"]
        ).apply(pd.to_numeric, errors="coerce")
        prefix_quantiles = prefix_overlap.loc[
            :, ["chronos2__q10", "chronos2__q50", "chronos2__q90"]
        ]
        finite_counts = np.isfinite(
            issued_quantiles.to_numpy(dtype=float)
        ).sum(axis=1)
        if bool(((finite_counts != 0) & (finite_counts != 3)).any()):
            raise ValueError(
                f"{prefix_path}: quantiles Chronos partiellement remplis sur "
                "le chevauchement."
            )
        populated = finite_counts == 3
        if bool(populated.any()) and not np.allclose(
            issued_quantiles.to_numpy(dtype=float)[populated],
            prefix_quantiles.to_numpy(dtype=float)[populated],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{prefix_path}: conflit de quantiles Chronos sur le chevauchement."
            )
        issued_actual = pd.to_numeric(
            issued_overlap.get("actual"), errors="coerce"
        ).to_numpy(dtype=float)
        prefix_actual = prefix_overlap["actual"].to_numpy(dtype=float)
        actual_populated = np.isfinite(issued_actual)
        if bool(actual_populated.any()) and not np.allclose(
            issued_actual[actual_populated],
            prefix_actual[actual_populated],
            rtol=0.0,
            # The issued target prefix was persisted as float32 in the older
            # aligned-input artifact; tolerate only that sub-cent rounding.
            atol=1e-5,
        ):
            raise ValueError(
                f"{prefix_path}: conflit d'actual sur le chevauchement."
            )
        if "forecast_origin_utc" in issued_overlap:
            raw_origin = issued_overlap["forecast_origin_utc"]
            origin_populated = raw_origin.notna().to_numpy()
            if bool(origin_populated.any()):
                existing_origins = _aware_utc_values(
                    raw_origin.loc[origin_populated],
                    path=source_run / "backtest_hourly_oof.csv.gz",
                    column="forecast_origin_utc",
                )
                expected_existing = pd.DatetimeIndex(
                    prefix_overlap.loc[origin_populated, "forecast_origin_utc"]
                )
                if bool(np.asarray(existing_origins != expected_existing).any()):
                    raise ValueError(
                        f"{prefix_path}: conflit de forecast_origin_utc sur "
                        "le chevauchement."
                    )
    columns = list(dict.fromkeys([*mapped.columns, *issued.columns]))
    combined_index = prefix.index.union(issued.index).sort_values()
    combined = issued.reindex(index=combined_index, columns=columns)
    combined.loc[mapped.index, mapped.columns] = mapped.to_numpy()
    if combined.index.has_duplicates:
        raise RuntimeError("Le prefixe Chronos a cree des timestamps dupliques.")
    required = ["chronos2__q10", "chronos2__q50", "chronos2__q90", "actual"]
    valid = np.isfinite(
        combined.loc[:, required].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=float
        )
    ).all(axis=1)
    valid_index = combined.index[valid]
    if len(valid_index) > 1 and not bool(
        valid_index.to_series().diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
    ):
        raise ValueError(
            f"{prefix_path}: le prefixe et l'historique emis ne forment pas "
            "un suffixe Chronos/actual horaire contigu."
        )
    combined.index.name = "delivery_start_utc"
    return combined


def load_residual_dataset(
    config: AuxiliaryLabConfig,
    model: ModelConfig,
) -> ResidualDataset:
    backtest = _residual_history_with_prefix(
        source_run=config.source_run,
        model=model,
        timezone=config.timezone,
    )
    aligned = _read_indexed(
        config.source_run / "inputs" / "aligned_inputs.csv.gz",
        timestamp_column="timestamp",
    )
    base_model = str(model.options["base_model"])
    expert_models = tuple(model.options["expert_models"])
    base_columns = [f"{base_model}__{quantile}" for quantile in QUANTILES]
    expert_columns = [
        f"{expert}__{quantile}"
        for expert in expert_models
        for quantile in QUANTILES
    ]
    required = list(dict.fromkeys(["actual", *base_columns, *expert_columns]))
    selected_backtest = _numeric(backtest, required, name="backtest residuel")
    common = selected_backtest.index.intersection(aligned.index, sort=False)
    if common.empty:
        raise ValueError("Aucun timestamp commun entre backtest et aligned_inputs.")
    joined = selected_backtest.loc[common].join(aligned.loc[common], how="inner")
    joined = _trailing_complete(
        joined,
        timezone=config.timezone,
        finite_columns=required,
    )
    feature_columns = [column for column in aligned if column != "target"]
    if not feature_columns:
        raise ValueError("aligned_inputs ne contient aucune covariable.")
    X = _numeric(joined, feature_columns, name="features residuelles")
    empty = [column for column in X if X[column].notna().sum() == 0]
    if empty:
        raise ValueError(f"Features entierement manquantes: {empty}.")
    base = joined.loc[:, base_columns].copy()
    base.columns = list(QUANTILES)
    experts = joined.loc[:, expert_columns].copy()
    split = validate_and_split_days(X.index, timezone=config.timezone, split=config.split)
    return ResidualDataset(
        X=X,
        actual=joined["actual"].astype(float),
        base=base.astype(float),
        experts=experts.astype(float),
        split=split,
        base_model=base_model,
        expert_models=expert_models,
    )


def load_blend_dataset(
    config: AuxiliaryLabConfig,
    model: ModelConfig,
) -> BlendDataset:
    backtest = _issued_history(config.source_run)
    autonomous = str(model.options["autonomous_model"])
    primary = str(model.options["primary_column"])
    autonomous_columns = [f"{autonomous}__{quantile}" for quantile in QUANTILES]
    required = ["actual", *autonomous_columns, primary]
    frame = _numeric(backtest, required, name="backtest MKOnline")
    frame = _trailing_complete(frame, timezone=config.timezone, finite_columns=required)
    split = validate_and_split_days(frame.index, timezone=config.timezone, split=config.split)
    return BlendDataset(
        frame=frame,
        split=split,
        autonomous_model=autonomous,
        primary_column=primary,
    )


def load_kalman_dataset(
    config: AuxiliaryLabConfig,
    model: ModelConfig,
    *,
    history_override: pd.DataFrame | None = None,
) -> KalmanDataset:
    statistics = (
        _issued_history(config.source_run)
        if history_override is None
        else history_override.copy()
    )
    if not isinstance(statistics.index, pd.DatetimeIndex):
        raise ValueError("history_override Kalman doit avoir un DatetimeIndex.")
    if (
        statistics.index.tz is None
        or statistics.index.has_duplicates
        or not statistics.index.is_monotonic_increasing
    ):
        raise ValueError(
            "history_override Kalman doit etre UTC-aware, unique et croissant."
        )
    raw_covariates, materialized_covariates, source_audit = load_kalman_covariate_frame(
        source_run=config.source_run,
        model=model,
        timezone=config.timezone,
    )
    covariate_config = model.options["covariate_config"]
    upstream = str(model.options["upstream_model"])
    upstream_columns = [f"{upstream}__{quantile}" for quantile in QUANTILES]
    required = ["actual", *upstream_columns]
    history = _numeric(statistics, required, name="historique Kalman")
    if history.index.intersection(raw_covariates.index, sort=False).empty:
        raise ValueError("Aucun timestamp commun entre Statistics et covariates.")
    # Preserve the complete issued-history support. Missing exogenous values
    # remain explicit NaNs and are governed by the selected contract.
    joined = history.join(materialized_covariates, how="left", sort=False)
    complete_columns = list(required)
    if covariate_config.history_missing_policy == "complete_trailing":
        complete_columns.extend(covariate_config.feature_columns)
    joined = _trailing_complete(
        joined,
        timezone=config.timezone,
        finite_columns=complete_columns,
    )
    training_lookback_days = model.options.get("training_lookback_days")
    if training_lookback_days is not None:
        lookback = int(training_lookback_days)
        initial_split = validate_and_split_days(
            joined.index,
            timezone=config.timezone,
            split=config.split,
        )
        earliest_evaluation = pd.Timestamp(initial_split.validation_days[0])
        required_start = (
            earliest_evaluation - pd.Timedelta(days=lookback)
        ).date()
        local_days = pd.Index(joined.index.tz_convert(config.timezone).date)
        joined = joined.loc[np.asarray(local_days >= required_start, dtype=bool)]
    history = joined.loc[:, required].copy()
    split = validate_and_split_days(history.index, timezone=config.timezone, split=config.split)
    coverage = _phase_coverage(
        materialized_covariates,
        index=history.index,
        split=split,
        timezone=config.timezone,
        config=covariate_config,
    )

    forecasts = sorted(config.source_run.glob("forecast_hourly_*.csv"))
    if len(forecasts) == 1:
        future_index = _read_indexed(
            forecasts[0],
            timestamp_column="delivery_start_utc",
        ).index
        future_materialized = materialized_covariates.reindex(future_index)
        future_coverage = covariate_coverage(future_materialized, covariate_config)
        future_coverage.insert(0, "phase", "future")
        coverage = pd.concat([coverage, future_coverage], ignore_index=True)
        if covariate_config.require_future_complete:
            future_features = future_materialized.loc[
                :, list(covariate_config.feature_columns)
            ]
            if not np.isfinite(future_features.to_numpy(dtype=float)).all():
                incomplete = [
                    column
                    for column in future_features
                    if not np.isfinite(
                        future_features[column].to_numpy(dtype=float)
                    ).all()
                ]
                raise ValueError(
                    "Les covariables Kalman futures ne couvrent pas exactement "
                    f"le forecast: {incomplete}."
                )
    raw_selected = raw_covariates.reindex(history.index).loc[
        :, list(covariate_config.input_columns)
    ]
    return KalmanDataset(
        history=history,
        covariates=raw_selected,
        split=split,
        upstream_model=upstream,
        covariate_config=covariate_config,
        coverage=coverage,
        source_audit=source_audit,
    )


__all__ = [
    "BlendDataset",
    "KalmanDataset",
    "MappingAudit",
    "QUANTILES",
    "ResidualDataset",
    "load_blend_dataset",
    "load_kalman_dataset",
    "load_kalman_covariate_frame",
    "load_residual_dataset",
    "join_kalman_additional_sources",
]
