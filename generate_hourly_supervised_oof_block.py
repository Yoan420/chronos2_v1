#!/usr/bin/env python
"""Generate one fixed-origin LEAR/CatBoost OOF block for one hourly zone.

The supervised experts are fitted once on complete local delivery days ending
before an explicit purge, then frozen over the requested test block.  This is
the causal companion to ``generate_chronos_oof_range.py`` for extending the
residual-corrector calibration set without moving the sealed final year.

When ``--chronos-oof-file`` is supplied, the output contains all nine expert
quantiles and is checked against the exact 188-column residual-v1 meta-feature
schema.  Without it, a supervised-only checkpoint is produced and can later
be regenerated/merged with the same command after Chronos is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from chronos2_hourly.chronos_adapter import (
    ChronosDeliveryPlan,
    generate_delivery_plans,
    load_chronos_oof,
)
from chronos2_hourly.models import HourlyCatBoost, HourlyLEAR
from chronos2_hourly.models.residual_corrector import ResidualMetaFeatureBuilder
from chronos2_modular.common import (
    ZoneConfig,
    build_zone_configs,
    deep_get,
    load_yaml,
    set_reproducibility,
)
from chronos2_modular.data import prepare_zone_data
from run_chronos2_hourly import (
    _feature_inputs,
    _trim_to_complete_delivery_days,
)


EXPERT_COLUMNS = tuple(
    f"{model}__{quantile}"
    for model in ("lear", "catboost", "chronos2")
    for quantile in ("q10", "q50", "q90")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Génère un bloc OOF LEAR/CatBoost causal pour une zone du YAML."
    )
    parser.add_argument(
        "--config",
        default="chronos2_hourly_fr_residual_v1.yaml",
    )
    parser.add_argument(
        "--zone",
        default="FR",
        help="Zone du YAML à traiter (FR par défaut, insensible à la casse).",
    )
    parser.add_argument(
        "--timezone",
        default=None,
        help=(
            "Assertion optionnelle sur le fuseau de livraison; elle doit "
            "correspondre à zones.<ZONE>.timezone dans le YAML."
        ),
    )
    parser.add_argument(
        "--calendar-primary-country",
        "--rich-calendar-primary-country",
        dest="calendar_primary_country",
        default=None,
        help=(
            "Assertion optionnelle sur le pays primaire du calendrier riche; "
            "elle doit correspondre au feature_builder YAML."
        ),
    )
    parser.add_argument("--train-end-day", required=True)
    parser.add_argument("--test-start-day", required=True)
    parser.add_argument("--test-end-day", required=True)
    parser.add_argument("--gap-days", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chronos-oof-file", default=None)
    parser.add_argument(
        "--canonical-run-dir",
        default=None,
        help=(
            "Run v1 utilisé uniquement pour vérifier le schéma des "
            "meta-features; par défaut, output.directory du YAML."
        ),
    )
    parser.add_argument(
        "--smoke-fast",
        action="store_true",
        help="Réduit CatBoost à 5 itérations; artefact marqué non canonique.",
    )
    parser.add_argument(
        "--catboost-threads",
        type=int,
        default=None,
        help=(
            "Override the CatBoost thread count. By default, use "
            "hourly.catboost.thread_count from the YAML config."
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Valide les dates/features sans ajuster LEAR ou CatBoost.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _parse_day(raw: str, *, name: str) -> pd.Timestamp:
    day = pd.Timestamp(raw)
    if day.tzinfo is not None or day != day.normalize():
        raise ValueError(f"{name} doit être une date locale naïve YYYY-MM-DD.")
    return day


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_index(plans: Sequence[ChronosDeliveryPlan]) -> pd.DatetimeIndex:
    if not plans:
        raise ValueError("Aucun plan de livraison demandé.")
    return pd.DatetimeIndex(
        np.concatenate([plan.delivery_index_utc.asi8 for plan in plans]),
        tz="UTC",
        name="delivery_start_utc",
    )


def _day_mask(index: pd.DatetimeIndex, days: set[object], timezone: str) -> np.ndarray:
    return np.asarray(pd.Index(index.tz_convert(timezone).date).isin(days), dtype=bool)


def _complete_days(index: pd.DatetimeIndex, timezone: str) -> list[object]:
    return pd.Index(index.tz_convert(timezone).date).unique().tolist()


def _validate_split(
    *,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    gap_days: int,
) -> list[pd.Timestamp]:
    if gap_days < 0:
        raise ValueError("--gap-days doit être positif ou nul.")
    if test_end < test_start:
        raise ValueError("--test-end-day doit être >= --test-start-day.")
    expected_train_end = test_start - pd.Timedelta(days=gap_days + 1)
    if train_end != expected_train_end:
        raise ValueError(
            "Découpage non conforme: avec un test commençant le "
            f"{test_start.date()} et gap_days={gap_days}, train_end_day doit "
            f"être {expected_train_end.date()}, reçu {train_end.date()}."
        )
    return list(pd.date_range(train_end + pd.Timedelta(days=1), test_start - pd.Timedelta(days=1)))


def _active_training_columns(X_train: pd.DataFrame) -> tuple[list[str], list[str]]:
    active = [column for column in X_train if X_train[column].notna().any()]
    dropped = [column for column in X_train if column not in active]
    if not active:
        raise ValueError("Toutes les features sont manquantes au train.")
    return active, dropped


def _model_configs(
    config: Mapping[str, Any],
    *,
    timezone: str,
    feature_columns: Sequence[str],
    smoke_fast: bool,
    catboost_threads: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lear_raw = dict(deep_get(config, "hourly.lear", {}) or {})
    cat_raw = dict(deep_get(config, "hourly.catboost", {}) or {})
    lear_raw.update(
        {
            "feature_columns": tuple(feature_columns),
            "hour_column": "calendar_local_hour",
            "timezone": timezone,
        }
    )
    cat_raw.update(
        {
            "feature_columns": tuple(feature_columns),
            "hour_column": "calendar_local_hour",
            "timezone": timezone,
        }
    )
    if catboost_threads is not None:
        if catboost_threads == 0 or catboost_threads < -1:
            raise ValueError("--catboost-threads must be -1 or a positive integer.")
        cat_raw["thread_count"] = int(catboost_threads)
    if smoke_fast:
        cat_raw["iterations"] = 5
        cat_raw["depth"] = min(int(cat_raw.get("depth", 7)), 4)
        cat_raw["thread_count"] = 1
    return lear_raw, cat_raw


def _resolve_zone_contract(
    config: Mapping[str, Any],
    requested_zone: str,
    *,
    timezone_override: str | None = None,
    primary_country_override: str | None = None,
) -> tuple[ZoneConfig, str, tuple[str, ...], str]:
    """Resolve the delivery timezone and residual-v1 calendar contract.

    The timezone is shared by data preparation, local-day splits, model
    features and delivery plans.  ``--timezone`` therefore acts as a
    fail-closed assertion instead of silently mutating only part of the run.
    """

    zone_code = str(requested_zone).strip().upper()
    if not zone_code:
        raise ValueError("--zone ne peut pas être vide.")
    zones = build_zone_configs(config, [zone_code], None, None)
    if len(zones) != 1 or str(zones[0].zone).upper() != zone_code:
        raise ValueError(
            "La configuration doit sélectionner exactement la zone "
            f"{zone_code}."
        )
    zone = zones[0]

    configured_timezone = str(zone.timezone).strip()
    builder_raw = deep_get(
        config,
        "hourly.residual_correction.feature_builder",
        {},
    )
    if not isinstance(builder_raw, Mapping):
        raise ValueError(
            "hourly.residual_correction.feature_builder doit être un mapping."
        )
    builder_timezone = str(
        builder_raw.get("timezone", configured_timezone)
    ).strip()
    if builder_timezone != configured_timezone:
        raise ValueError(
            "Le fuseau du feature_builder doit correspondre au fuseau de la "
            f"zone {zone_code}: {builder_timezone!r} != {configured_timezone!r}."
        )
    if timezone_override is not None:
        asserted_timezone = str(timezone_override).strip()
        if asserted_timezone != configured_timezone:
            raise ValueError(
                "--timezone doit correspondre au fuseau de la zone dans le YAML: "
                f"{asserted_timezone!r} != {configured_timezone!r}."
            )
    feature_timezone = str(
        deep_get(
            config,
            "hourly.feature_engineering.timezone",
            configured_timezone,
        )
    ).strip()
    if feature_timezone != configured_timezone:
        raise ValueError(
            "Le fuseau du feature engineering doit correspondre au fuseau de "
            f"la zone {zone_code}: {feature_timezone!r} != "
            f"{configured_timezone!r}."
        )

    countries_raw = builder_raw.get(
        "rich_calendar_countries",
        ("FR", "DE", "BE", "ES", "NL"),
    )
    if isinstance(countries_raw, str):
        countries_raw = [countries_raw]
    if not isinstance(countries_raw, Sequence):
        raise ValueError("rich_calendar_countries doit être une séquence.")
    countries = tuple(
        dict.fromkeys(str(country).strip().upper() for country in countries_raw)
    )
    if not countries or any(not country for country in countries):
        raise ValueError("rich_calendar_countries ne peut pas contenir de pays vide.")

    primary = str(
        builder_raw.get("rich_calendar_primary_country", zone_code)
    ).strip().upper()
    if not primary:
        raise ValueError("Le pays primaire du calendrier ne peut pas être vide.")
    if primary not in countries:
        raise ValueError(
            "Le pays primaire du calendrier doit appartenir à "
            f"rich_calendar_countries: {primary} absent de {countries}."
        )
    if primary_country_override is not None:
        asserted_primary = str(primary_country_override).strip().upper()
        if asserted_primary != primary:
            raise ValueError(
                "--calendar-primary-country doit correspondre au "
                "feature_builder du YAML: "
                f"{asserted_primary!r} != {primary!r}."
            )
    return zone, configured_timezone, countries, primary


def build_v1_meta_features(
    X: pd.DataFrame,
    experts: pd.DataFrame,
    *,
    timezone: str = "Europe/Paris",
    rich_calendar_countries: Sequence[str] = ("FR", "DE", "BE", "ES", "NL"),
    rich_calendar_primary_country: str = "FR",
) -> pd.DataFrame:
    """Build the frozen residual-v1 schema used by strict model screens."""

    missing = [column for column in EXPERT_COLUMNS if column not in experts]
    if missing:
        raise ValueError(f"Quantiles experts absents: {missing}.")
    base = experts.loc[:, [
        "chronos2__q10",
        "chronos2__q50",
        "chronos2__q90",
    ]].copy()
    base.columns = ["q10", "q50", "q90"]
    combined = pd.concat(
        [base.add_prefix("base__"), experts.loc[:, list(EXPERT_COLUMNS)]],
        axis=1,
    )
    builder = ResidualMetaFeatureBuilder(
        timezone=timezone,
        include_calendar=True,
        include_rich_calendar=True,
        rich_calendar_countries=rich_calendar_countries,
        rich_calendar_primary_country=rich_calendar_primary_country,
        include_daily_profiles=True,
        include_fundamental_interactions=False,
        include_missing_indicators=False,
        exclude_historical_prices=True,
        exclude_day_of_year=True,
    )
    meta = builder.fit_transform(X, combined)
    if meta.shape[1] != 188:
        raise RuntimeError(
            f"Schéma résiduel v1 inattendu: {meta.shape[1]} colonnes au lieu de 188."
        )
    return meta


def _canonical_meta_schema(
    run_dir: Path,
    *,
    timezone: str = "Europe/Paris",
    rich_calendar_countries: Sequence[str] = ("FR", "DE", "BE", "ES", "NL"),
    rich_calendar_primary_country: str = "FR",
) -> tuple[str, ...]:
    manifest = pd.read_csv(run_dir / "feature_manifest.csv")
    x_columns = manifest["feature"].astype(str).tolist()
    raw = pd.read_csv(run_dir / "backtest_hourly_oof.csv.gz")
    raw.index = pd.DatetimeIndex(
        pd.to_datetime(raw.pop("delivery_start_utc"), utc=True),
        name="delivery_start_utc",
    )
    raw = raw.loc[raw["fold_id"].notna()].head(25).copy()
    context = pd.read_csv(run_dir / "inputs" / "model_covariates_with_future.csv.gz")
    context.index = pd.DatetimeIndex(pd.to_datetime(context.pop("timestamp"), utc=True))
    # Reuse the already exported canonical feature matrix columns when they
    # are raw context features; derived price/calendar columns are rebuilt by
    # the caller and therefore cannot be recovered from this context alone.
    # The schema itself depends on names, not values, so a finite placeholder
    # with the exact feature names is sufficient for this independent check.
    X = pd.DataFrame(1.0, index=raw.index, columns=x_columns)
    experts = raw.loc[:, list(EXPERT_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    return tuple(
        build_v1_meta_features(
            X,
            experts,
            timezone=timezone,
            rich_calendar_countries=rich_calendar_countries,
            rich_calendar_primary_country=rich_calendar_primary_country,
        ).columns
    )


def _merge_chronos(
    supervised: pd.DataFrame,
    chronos_path: Path,
    plans: Sequence[ChronosDeliveryPlan],
) -> pd.DataFrame:
    chronos = load_chronos_oof(chronos_path)
    expected = _expected_index(plans)
    if not chronos.index.equals(expected):
        raise ValueError(
            "L'artefact Chronos ne couvre pas exactement le bloc test demandé."
        )
    expected_origins = pd.DatetimeIndex(
        np.concatenate(
            [np.repeat(plan.forecast_origin_utc.value, plan.horizon) for plan in plans]
        ),
        tz="UTC",
    )
    actual_origins = pd.DatetimeIndex(chronos["forecast_origin_utc"])
    if not actual_origins.equals(expected_origins):
        raise ValueError("Les origines Chronos diffèrent des plans D-1 attendus.")
    if not np.allclose(
        supervised["actual"].to_numpy(dtype=float),
        chronos["actual"].to_numpy(dtype=float),
        rtol=0.0,
        atol=5e-5,
    ):
        raise ValueError("La cible Chronos diffère de la cible canonique.")
    result = supervised.copy()
    for quantile in ("q10", "q50", "q90"):
        result[f"chronos2__{quantile}"] = chronos[quantile].to_numpy(dtype=float)
    return result


def _atomic_write(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.name.lower().endswith(".csv.gz"):
        raise ValueError("--output doit se terminer par .csv.gz.")
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.stem}.", suffix=".csv.gz", dir=output.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.reset_index().to_csv(temporary, index=False, compression="gzip")
        pd.read_csv(temporary, nrows=2)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    train_end = _parse_day(args.train_end_day, name="--train-end-day")
    test_start = _parse_day(args.test_start_day, name="--test-start-day")
    test_end = _parse_day(args.test_end_day, name="--test-end-day")
    purged_days = _validate_split(
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        gap_days=int(args.gap_days),
    )
    config_path = Path(args.config).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite and not args.plan_only:
        raise FileExistsError(f"La sortie existe déjà: {output}.")
    config = load_yaml(config_path)
    set_reproducibility(int(deep_get(config, "model.seed", 42)))
    zone, timezone, calendar_countries, calendar_primary_country = (
        _resolve_zone_contract(
            config,
            args.zone,
            timezone_override=args.timezone,
            primary_country_override=args.calendar_primary_country,
        )
    )
    args.zone = zone.zone
    args.timezone = timezone
    args.calendar_primary_country = calendar_primary_country
    inputs_dir = output.parent / f".{output.name}.inputs"
    data = prepare_zone_data(zone, config, config_path.parent, False, inputs_dir)
    target, _, _, all_features = _feature_inputs(data, config)
    X, y, _ = _trim_to_complete_delivery_days(
        all_features.loc[target.index].copy(),
        target,
        timezone=timezone,
    )
    local_days = pd.Index(X.index.tz_convert(timezone).date)
    train_days = {day for day in local_days.unique() if day <= train_end.date()}
    requested_test_days = set(pd.date_range(test_start, test_end, freq="D").date)
    train_mask = _day_mask(X.index, train_days, timezone)
    test_mask = _day_mask(X.index, requested_test_days, timezone)
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    if not len(X_train) or not len(X_test):
        raise ValueError("Bloc train ou test vide.")
    observed_test_days = _complete_days(X_test.index, timezone)
    if observed_test_days != sorted(requested_test_days):
        raise ValueError("Le test ne contient pas exactement les jours locaux demandés.")
    active_columns, dropped_columns = _active_training_columns(X_train)
    plans = generate_delivery_plans(
        test_start.date(),
        test_end.date(),
        forecast_origin_local_time=str(
            deep_get(config, "data.forecast_origin_local_time", "08:00")
        ),
        timezone=timezone,
    )
    expected = _expected_index(plans)
    if not X_test.index.equals(expected):
        raise ValueError("L'index test ne correspond pas aux plans DST canoniques.")
    audit: dict[str, Any] = {
        "zone": zone.zone,
        "timezone": timezone,
        "rich_calendar_countries": list(calendar_countries),
        "rich_calendar_primary_country": calendar_primary_country,
        "train_start_local_day": str(min(train_days)),
        "train_end_local_day": str(max(train_days)),
        "n_train_days": int(len(train_days)),
        "n_train_hours": int(len(X_train)),
        "purged_local_days": [str(day.date()) for day in purged_days],
        "test_start_local_day": str(test_start.date()),
        "test_end_local_day": str(test_end.date()),
        "n_test_days": int(len(observed_test_days)),
        "n_test_hours": int(len(X_test)),
        "active_feature_columns": active_columns,
        "dropped_train_all_missing_columns": dropped_columns,
        "chronos_attached": bool(args.chronos_oof_file),
        "smoke_fast_noncanonical": bool(args.smoke_fast),
    }
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    manifest_path = output.with_name(output.name + ".manifest.json")
    if args.plan_only:
        output.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "status": "planned",
                    "config": str(config_path),
                    "config_sha256": _sha256(config_path),
                    "zone": zone.zone,
                    "timezone": timezone,
                    "rich_calendar_primary_country": calendar_primary_country,
                    "audit": audit,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return 0

    lear_config, cat_config = _model_configs(
        config,
        timezone=timezone,
        feature_columns=active_columns,
        smoke_fast=bool(args.smoke_fast),
        catboost_threads=args.catboost_threads,
    )
    started = time.perf_counter()
    lear = HourlyLEAR(**lear_config).fit(X_train, y_train)
    catboost = HourlyCatBoost(**cat_config).fit(X_train, y_train)
    lear_prediction = lear.predict(X_test)
    catboost_prediction = catboost.predict(X_test)
    result = pd.DataFrame(index=expected)
    for quantile in ("q10", "q50", "q90"):
        result[f"lear__{quantile}"] = lear_prediction[quantile].to_numpy(dtype=float)
        result[f"catboost__{quantile}"] = catboost_prediction[quantile].to_numpy(dtype=float)
    result["actual"] = y_test.to_numpy(dtype=float)
    result["forecast_origin_utc"] = pd.DatetimeIndex(
        np.concatenate(
            [np.repeat(plan.forecast_origin_utc.value, plan.horizon) for plan in plans]
        ),
        tz="UTC",
    )
    meta_columns: list[str] | None = None
    if args.chronos_oof_file:
        result = _merge_chronos(result, Path(args.chronos_oof_file).expanduser().resolve(), plans)
        experts = result.loc[:, list(EXPERT_COLUMNS)]
        meta = build_v1_meta_features(
            X_test,
            experts,
            timezone=timezone,
            rich_calendar_countries=calendar_countries,
            rich_calendar_primary_country=calendar_primary_country,
        )
        canonical_run_dir_raw = args.canonical_run_dir or deep_get(
            config,
            "output.directory",
            f"runs/chronos2_hourly_{zone.zone.lower()}_residual_v1",
        )
        canonical_run_dir = Path(canonical_run_dir_raw).expanduser()
        if not canonical_run_dir.is_absolute():
            canonical_run_dir = (config_path.parent / canonical_run_dir).resolve()
        expected_schema = _canonical_meta_schema(
            canonical_run_dir,
            timezone=timezone,
            rich_calendar_countries=calendar_countries,
            rich_calendar_primary_country=calendar_primary_country,
        )
        if tuple(meta.columns) != expected_schema:
            raise RuntimeError("Le schéma des 188 meta-features diffère du run v1.")
        meta_columns = list(meta.columns)
    _atomic_write(result, output)
    elapsed = time.perf_counter() - started
    manifest_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "config": str(config_path),
                "config_sha256": _sha256(config_path),
                "zone": zone.zone,
                "timezone": timezone,
                "rich_calendar_countries": list(calendar_countries),
                "rich_calendar_primary_country": calendar_primary_country,
                "output": str(output),
                "output_sha256": _sha256(output),
                "elapsed_seconds": elapsed,
                "lear_config": {**lear_config, "feature_columns": active_columns},
                "catboost_config": {**cat_config, "feature_columns": active_columns},
                "meta_feature_count": len(meta_columns) if meta_columns else None,
                "meta_feature_columns": meta_columns,
                "audit": audit,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(f"Bloc OOF supervisé: {output} | {elapsed:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
