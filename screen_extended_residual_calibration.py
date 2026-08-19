#!/usr/bin/env python
"""Strict nested screen for a residual corrector with an earlier Chronos block.

Protocol (local delivery days):

* EXT: external native-expert OOF block, 2024-01-02..2024-08-11;
* A: existing calibration days 1..245;
* B1: existing calibration days 246..305, used for recipe selection;
* B2: existing calibration days 306..365, untouched veto;
* FINAL: existing sealed final year, loaded only after explicit B1/B2 passes.

The historical comparator is fitted on A only.  Extended candidates are fitted
on EXT+A, so the screen measures the value of the earlier block as well as the
frozen residual recipe.  By default the common schema contains only Chronos
quantiles and causal X features; it does not require earlier LEAR/CatBoost OOF.
No external price forecast is loaded by this utility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chronos2_hourly.features import (
    build_history_future_feature_matrix,
    build_hourly_feature_matrix,
)
from chronos2_hourly.models.residual_corrector import ResidualMetaFeatureBuilder


TIMEZONE = "Europe/Paris"
FINAL_VALIDATION_PATH = ROOT / "runs" / "chronos2_hourly_fr_extended_residual" / "validation_final.json"
CALIBRATION_END_EXCLUSIVE_UTC = pd.Timestamp(
    "2025-08-12", tz=TIMEZONE
).tz_convert("UTC")
V1_EXPERT_COLUMNS = tuple(
    f"{model}__{quantile}"
    for model in ("lear", "catboost", "chronos2")
    for quantile in ("q10", "q50", "q90")
)
CHRONOS_EXPERT_COLUMNS = tuple(f"chronos2__{quantile}" for quantile in ("q10", "q50", "q90"))


def _expert_columns(schema: str) -> tuple[str, ...]:
    return CHRONOS_EXPERT_COLUMNS if schema == "chronos_only" else V1_EXPERT_COLUMNS


def _read_timestamped_input(path: Path, *, include_final: bool) -> pd.DataFrame:
    if include_final:
        frame = pd.read_csv(path)
    else:
        chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(path, chunksize=720):
            timestamps = pd.to_datetime(chunk["timestamp"], utc=True)
            before_cutoff = timestamps < CALIBRATION_END_EXCLUSIVE_UTC
            if bool(before_cutoff.any()):
                chunks.append(chunk.loc[before_cutoff].copy())
            if bool((~before_cutoff).any()):
                break
        if not chunks:
            raise RuntimeError(f"Aucune entrée avant le cutoff dans {path}.")
        frame = pd.concat(chunks, ignore_index=True)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("timestamp"), utc=True))
    return frame


def _load_features(run_dir: Path, *, include_final: bool) -> pd.DataFrame:
    aligned = _read_timestamped_input(
        run_dir / "inputs" / "aligned_inputs.csv.gz", include_final=include_final
    )
    target = pd.to_numeric(aligned.pop("target"), errors="coerce").astype(float)
    context = _read_timestamped_input(
        run_dir / "inputs" / "model_covariates_with_future.csv.gz",
        include_final=include_final,
    )
    manifest = pd.read_csv(run_dir / "feature_manifest.csv")
    input_columns = [str(value) for value in manifest["feature"] if str(value) in context]
    covariates = context.loc[:, input_columns].apply(pd.to_numeric, errors="coerce")
    historical = covariates.loc[target.index]
    if include_final:
        future = covariates.loc[covariates.index.difference(target.index, sort=False)]
        features = build_history_future_feature_matrix(
            target,
            historical,
            future,
            price_lags=(24, 48, 168, 336),
            rolling_windows=(24, 72, 168),
            timezone=TIMEZONE,
            scope="all",
        ).loc[target.index]
    else:
        features = build_hourly_feature_matrix(
            target,
            historical,
            price_lags=(24, 48, 168, 336),
            rolling_windows=(24, 72, 168),
            timezone=TIMEZONE,
        )
    expected = manifest["feature"].astype(str).tolist()
    if list(features.columns) != expected:
        raise RuntimeError("Reconstruction des features différente du manifeste v1.")
    features.index = features.index.tz_convert("UTC")
    features.index.name = "delivery_start_utc"
    return features


def _build_meta(
    X: pd.DataFrame,
    experts: pd.DataFrame,
    expert_columns: tuple[str, ...],
) -> pd.DataFrame:
    missing = [column for column in expert_columns if column not in experts]
    if missing:
        raise ValueError(f"Colonnes expertes absentes: {missing}.")
    base = experts.loc[:, [
        "chronos2__q10",
        "chronos2__q50",
        "chronos2__q90",
    ]].copy()
    base.columns = ["q10", "q50", "q90"]
    combined = pd.concat(
        [base.add_prefix("base__"), experts.loc[:, list(expert_columns)]],
        axis=1,
    )
    builder = ResidualMetaFeatureBuilder(
        timezone=TIMEZONE,
        include_calendar=True,
        include_rich_calendar=True,
        rich_calendar_countries=("FR", "DE", "BE", "ES", "NL"),
        rich_calendar_primary_country="FR",
        include_daily_profiles=True,
        include_fundamental_interactions=False,
        include_missing_indicators=False,
        exclude_historical_prices=True,
        exclude_day_of_year=True,
    )
    meta = builder.fit_transform(X, combined)
    return meta


def _load_external(path: Path, X_all: pd.DataFrame, *, schema: str) -> dict[str, Any]:
    raw = pd.read_csv(path)
    expert_columns = _expert_columns(schema)
    if schema == "chronos_only" and not set(expert_columns).issubset(raw.columns):
        raw = raw.rename(
            columns={quantile: f"chronos2__{quantile}" for quantile in ("q10", "q50", "q90")}
        )
    required = {"delivery_start_utc", "actual", *expert_columns}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Artefact extended incomplet: {missing}.")
    raw.index = pd.DatetimeIndex(
        pd.to_datetime(raw.pop("delivery_start_utc"), utc=True),
        name="delivery_start_utc",
    )
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError("Index extended dupliqué ou non trié.")
    local_days = pd.Index(raw.index.tz_convert(TIMEZONE).date)
    days = local_days.unique().tolist()
    if len(days) != 223:
        raise ValueError(f"Le bloc extended doit contenir 223 jours, reçu {len(days)}.")
    if str(days[0]) != "2024-01-02" or str(days[-1]) != "2024-08-11":
        raise ValueError(f"Plage extended inattendue: {days[0]}..{days[-1]}.")
    if len(raw) != 5351:
        raise ValueError(f"Le bloc extended doit contenir 5351 heures, reçu {len(raw)}.")
    X = X_all.loc[raw.index].copy()
    experts = raw.loc[:, list(expert_columns)].apply(pd.to_numeric, errors="coerce")
    meta = _build_meta(X, experts, expert_columns)
    base_q50 = experts["chronos2__q50"].astype(float)
    residual = pd.to_numeric(raw["actual"], errors="coerce") - base_q50
    return {
        "raw": raw,
        "meta": meta,
        "base_q50": base_q50,
        "residual": residual,
        "days": days,
    }


def _load_existing(
    run_dir: Path,
    X_all: pd.DataFrame,
    *,
    include_final: bool,
    schema: str,
) -> dict[str, Any]:
    path = run_dir / "backtest_hourly_oof.csv.gz"
    if include_final:
        raw = pd.read_csv(path)
    else:
        chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(path, chunksize=720):
            timestamps = pd.to_datetime(chunk["delivery_start_utc"], utc=True)
            before_cutoff = timestamps < CALIBRATION_END_EXCLUSIVE_UTC
            if bool(before_cutoff.any()):
                chunks.append(chunk.loc[before_cutoff].copy())
            if bool((~before_cutoff).any()):
                break
        raw = pd.concat(chunks, ignore_index=True)
    raw.index = pd.DatetimeIndex(
        pd.to_datetime(raw.pop("delivery_start_utc"), utc=True),
        name="delivery_start_utc",
    )
    raw = raw.loc[raw["fold_id"].notna()].copy()
    local_days = pd.Index(raw.index.tz_convert(TIMEZONE).date)
    days = local_days.unique().tolist()
    expected_days = 730 if include_final else 365
    if len(days) != expected_days:
        raise RuntimeError(
            f"Attendu {expected_days} jours OOF existants, reçu {len(days)}."
        )
    expert_columns = _expert_columns(schema)
    experts = raw.loc[:, list(expert_columns)].apply(pd.to_numeric, errors="coerce")
    meta = _build_meta(X_all.loc[raw.index], experts, expert_columns)
    base_q50 = experts["chronos2__q50"].astype(float)
    residual = pd.to_numeric(raw["actual"], errors="coerce") - base_q50
    return {
        "raw": raw,
        "meta": meta,
        "base_q50": base_q50,
        "residual": residual,
        "days": days,
        "local_days": local_days,
    }


def _cat_v1(threads: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=700,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=15.0,
        random_seed=42,
        thread_count=threads,
        allow_writing_files=False,
        verbose=False,
        has_time=True,
        nan_mode="Min",
    )


def _hgb31() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.035,
        max_iter=550,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=60,
        early_stopping=False,
        random_state=42,
    )


def _fit_models(X: pd.DataFrame, residual: pd.Series, threads: int) -> dict[str, Any]:
    cat = _cat_v1(threads).fit(X, residual)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    Xi = imputer.fit_transform(X)
    hgb = _hgb31().fit(Xi, residual)
    return {"cat_v1": cat, "hgb31": hgb, "imputer": imputer}


def _predict(models: Mapping[str, Any], X: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "cat_v1": np.asarray(models["cat_v1"].predict(X), dtype=float),
        "hgb31": np.asarray(
            models["hgb31"].predict(models["imputer"].transform(X)),
            dtype=float,
        ),
    }


def _clip(values: np.ndarray, bound: float | None = 40.0) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array if bound is None else np.clip(array, -float(bound), float(bound))


def _parse_recipe(recipe: str, prediction: Mapping[str, np.ndarray]) -> np.ndarray:
    if recipe in prediction:
        return _clip(prediction[recipe], 40.0)
    if recipe.startswith("cal_"):
        body = recipe[len("cal_"):]
        source, calibration = body.rsplit("_s", 1)
        scale, bound = calibration.rsplit("_c", 1)
        return _clip(
            float(scale) * prediction[source],
            None if bound == "none" else float(bound),
        )
    if recipe.startswith("blend_cat_hgb_w"):
        weight = float(recipe.rsplit("w", 1)[1])
        return _clip(
            weight * prediction["hgb31"]
            + (1.0 - weight) * prediction["cat_v1"],
            40.0,
        )
    raise ValueError(f"Recette inconnue: {recipe}.")


def _mae(actual: np.ndarray, base: np.ndarray, correction: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - (base + correction))))


def _score_block(
    *,
    raw: pd.DataFrame,
    base_q50: pd.Series,
    prediction: Mapping[str, np.ndarray],
    recipe: str,
) -> dict[str, float]:
    actual = pd.to_numeric(raw["actual"], errors="coerce").to_numpy(dtype=float)
    base = base_q50.to_numpy(dtype=float)
    baseline = _mae(actual, base, _clip(prediction["baseline_cat_v1"], 40.0))
    candidate = _mae(actual, base, _parse_recipe(recipe, prediction))
    return {"baseline_cat_v1_mae": baseline, "candidate_mae": candidate, "gain": baseline - candidate}


def _paired_daily_gain_diagnostics(
    index: pd.DatetimeIndex,
    actual: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    *,
    seed: int = 42,
    draws: int = 20_000,
) -> dict[str, Any]:
    local_days = pd.Index(index.tz_convert(TIMEZONE).date, name="delivery_day")
    gains = pd.Series(
        np.abs(actual - baseline_prediction) - np.abs(actual - candidate_prediction),
        index=local_days,
    ).groupby(level=0, sort=False).mean().to_numpy(dtype=float)
    generator = np.random.default_rng(seed)
    samples = generator.choice(
        gains, size=(draws, len(gains)), replace=True
    ).mean(axis=1)
    return {
        "mean_daily_gain": float(gains.mean()),
        "daily_bootstrap_gain_ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "n_days_candidate_better": int((gains > 0.0).sum()),
        "n_days_candidate_worse": int((gains < 0.0).sum()),
    }


def _emit_final_result(payload: Mapping[str, Any]) -> None:
    """Persist the one-shot sealed-final result atomically, then print it."""

    serialized = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    FINAL_VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".validation_final.",
        suffix=".json",
        dir=FINAL_VALIDATION_PATH.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(serialized)
        handle.write("\n")
    temporary.replace(FINAL_VALIDATION_PATH)
    print(serialized)


def _fit_frame(external: dict[str, Any], existing: dict[str, Any], mask: np.ndarray) -> tuple[pd.DataFrame, pd.Series]:
    if tuple(external["meta"].columns) != tuple(existing["meta"].columns):
        raise RuntimeError("EXT+A n'est pas aligné sur les mêmes meta-features.")
    X = pd.concat([external["meta"], existing["meta"].loc[mask]], axis=0)
    y = pd.concat([external["residual"], existing["residual"].loc[mask]], axis=0)
    if not X.index.is_monotonic_increasing or X.index.has_duplicates:
        raise RuntimeError("EXT+A n'est pas strictement chronologique et unique.")
    if not y.index.equals(X.index):
        raise RuntimeError("EXT+A n'est pas aligné sur les meta-features.")
    return X, y


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("b1", "b2", "final"), required=True)
    parser.add_argument("--extended-oof-file", required=True)
    parser.add_argument("--run-dir", default="runs/chronos2_hourly_fr_residual_v1")
    parser.add_argument(
        "--schema",
        choices=("chronos_only", "v1_all_experts"),
        default="chronos_only",
    )
    parser.add_argument("--recipe")
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--b1-passed", action="store_true")
    parser.add_argument("--b2-passed", action="store_true")
    args = parser.parse_args()
    if args.phase in {"b2", "final"} and not args.recipe:
        parser.error("--recipe est requis pour b2/final")
    if args.phase == "final" and not (args.b1_passed and args.b2_passed):
        parser.error("final exige --b1-passed --b2-passed")

    run_dir = Path(args.run_dir).expanduser().resolve()
    X_all = _load_features(run_dir, include_final=args.phase == "final")
    external = _load_external(
        Path(args.extended_oof_file).expanduser().resolve(), X_all, schema=args.schema
    )
    existing = _load_existing(
        run_dir, X_all, include_final=args.phase == "final", schema=args.schema
    )
    if tuple(external["meta"].columns) != tuple(existing["meta"].columns):
        raise RuntimeError("Le schéma EXT diffère du schéma du run existant.")
    days = existing["days"]
    local_days = existing["local_days"]
    masks = {
        "A": np.asarray(local_days.isin(days[:245]), dtype=bool),
        "B1": np.asarray(local_days.isin(days[245:305]), dtype=bool),
        "B2": np.asarray(local_days.isin(days[305:365]), dtype=bool),
        "CAL": np.asarray(local_days.isin(days[:365]), dtype=bool),
        "FINAL": np.asarray(local_days.isin(days[365:]), dtype=bool),
    }
    fit_mask = masks["CAL"] if args.phase == "final" else masks["A"]
    fit_X, fit_y = _fit_frame(external, existing, fit_mask)
    extended_models = _fit_models(fit_X, fit_y, args.threads)
    baseline_models = _fit_models(
        existing["meta"].loc[fit_mask],
        existing["residual"].loc[fit_mask],
        args.threads,
    )

    if args.phase == "b1":
        index = existing["meta"].index[masks["B1"]]
        prediction = _predict(extended_models, existing["meta"].loc[index])
        prediction["baseline_cat_v1"] = _predict(
            baseline_models, existing["meta"].loc[index]
        )["cat_v1"]
        raw = existing["raw"].loc[index]
        base = existing["base_q50"].loc[index]
        recipes: dict[str, np.ndarray] = {
            "cat_v1": _clip(prediction["cat_v1"], 40.0),
            "hgb31": _clip(prediction["hgb31"], 40.0),
        }
        for weight in (0.25, 0.50, 0.75):
            recipes[f"blend_cat_hgb_w{weight:.2f}"] = _clip(
                weight * prediction["hgb31"] + (1.0 - weight) * prediction["cat_v1"],
                40.0,
            )
        for source in ("cat_v1", "hgb31"):
            for scale in (0.70, 0.80, 0.90, 1.00, 1.10):
                for bound in (30.0, 40.0, 60.0):
                    recipes[f"cal_{source}_s{scale:.2f}_c{int(bound)}"] = _clip(
                        scale * prediction[source], bound
                    )
        actual = raw["actual"].to_numpy(dtype=float)
        q50 = base.to_numpy(dtype=float)
        scores = {name: _mae(actual, q50, correction) for name, correction in recipes.items()}
        baseline = _mae(actual, q50, _clip(prediction["baseline_cat_v1"], 40.0))
        half = len(actual) // 2
        baseline_halves = (
            _mae(actual[:half], q50[:half], _clip(prediction["baseline_cat_v1"][:half], 40.0)),
            _mae(actual[half:], q50[half:], _clip(prediction["baseline_cat_v1"][half:], 40.0)),
        )
        detail: dict[str, dict[str, float | bool]] = {}
        for name, correction in recipes.items():
            candidate_halves = (
                _mae(actual[:half], q50[:half], correction[:half]),
                _mae(actual[half:], q50[half:], correction[half:]),
            )
            detail[name] = {
                "mae_b1": scores[name],
                "gain_vs_A_only_cat_v1": baseline - scores[name],
                "gain_first30": baseline_halves[0] - candidate_halves[0],
                "gain_last30": baseline_halves[1] - candidate_halves[1],
                "robust_b1": bool(
                    baseline - scores[name] > 0.0
                    and baseline_halves[0] - candidate_halves[0] > 0.0
                    and baseline_halves[1] - candidate_halves[1] > 0.0
                ),
            }
        eligible = [name for name, values in detail.items() if values["robust_b1"]]
        recommended = min(eligible, key=lambda name: scores[name]) if eligible else None
        result = {
            "protocol": {
                "fit_EXT": [str(external["days"][0]), str(external["days"][-1])],
                "fit_A": [str(days[0]), str(days[244])],
                "selection_B1": [str(days[245]), str(days[304])],
                "B2_not_read": True,
                "final_not_loaded": True,
                "baseline_fit_days": 245,
                "candidate_fit_days": 223 + 245,
                "schema": args.schema,
                "n_meta_features": int(fit_X.shape[1]),
            },
            "baseline_A_only_cat_v1_mae_b1": baseline,
            "recommended_recipe": recommended,
            "b1_passed": recommended is not None,
            "recipes": dict(sorted(detail.items(), key=lambda item: item[1]["mae_b1"])),
        }
        print(json.dumps(result, indent=2))
        return 0

    block = "B2" if args.phase == "b2" else "FINAL"
    index = existing["meta"].index[masks[block]]
    prediction = _predict(extended_models, existing["meta"].loc[index])
    prediction["baseline_cat_v1"] = _predict(
        baseline_models, existing["meta"].loc[index]
    )["cat_v1"]
    scores = _score_block(
        raw=existing["raw"].loc[index],
        base_q50=existing["base_q50"].loc[index],
        prediction=prediction,
        recipe=args.recipe,
    )
    if args.phase == "b2":
        actual = existing["raw"].loc[index, "actual"].to_numpy(dtype=float)
        q50 = existing["base_q50"].loc[index].to_numpy(dtype=float)
        baseline_correction = _clip(prediction["baseline_cat_v1"], 40.0)
        candidate_correction = _parse_recipe(args.recipe, prediction)
        half = len(actual) // 2
        gain_first30 = _mae(
            actual[:half], q50[:half], baseline_correction[:half]
        ) - _mae(actual[:half], q50[:half], candidate_correction[:half])
        gain_last30 = _mae(
            actual[half:], q50[half:], baseline_correction[half:]
        ) - _mae(actual[half:], q50[half:], candidate_correction[half:])
        passed = bool(scores["gain"] > 0.0 and gain_first30 > 0.0 and gain_last30 > 0.0)
        baseline_prediction = q50 + baseline_correction
        candidate_prediction = q50 + candidate_correction
        robustness = _paired_daily_gain_diagnostics(
            index, actual, baseline_prediction, candidate_prediction
        )
        print(json.dumps({
            "protocol": {
                "fit_EXT_plus_A_days": 223 + 245,
                "veto_B2": [str(days[305]), str(days[364])],
                "frozen_recipe": args.recipe,
                "final_not_loaded": True,
            },
            **scores,
            "gain_first30": gain_first30,
            "gain_last30": gain_last30,
            **robustness,
            "passed": passed,
        }, indent=2))
        return 0 if passed else 2

    actual = existing["raw"].loc[index, "actual"].to_numpy(dtype=float)
    v1 = existing["raw"].loc[index, "residual_corrected__q50"].to_numpy(dtype=float)
    q50 = existing["base_q50"].loc[index].to_numpy(dtype=float)
    baseline_correction = _clip(prediction["baseline_cat_v1"], 40.0)
    candidate_correction = _parse_recipe(args.recipe, prediction)
    baseline_prediction = q50 + baseline_correction
    candidate_prediction = q50 + candidate_correction
    robustness = _paired_daily_gain_diagnostics(
        index, actual, baseline_prediction, candidate_prediction
    )
    local_days_final = pd.Index(index.tz_convert(TIMEZONE).date)
    unique_final_days = local_days_final.unique()
    half_details: dict[str, dict[str, float]] = {}
    for name, block_days in {
        "H1": unique_final_days[:182],
        "H2": unique_final_days[182:],
    }.items():
        mask = np.asarray(local_days_final.isin(block_days), dtype=bool)
        base_value = float(np.mean(np.abs(actual[mask] - baseline_prediction[mask])))
        candidate_value = float(
            np.mean(np.abs(actual[mask] - candidate_prediction[mask]))
        )
        half_details[name] = {
            "baseline_mae": base_value,
            "candidate_mae": candidate_value,
            "gain": base_value - candidate_value,
        }
    result = {
        "protocol": {
            "fit_EXT_plus_calibration_days": 223 + 365,
            "sealed_final": [str(days[365]), str(days[-1])],
            "frozen_recipe": args.recipe,
        },
        **scores,
        **robustness,
        "half_years": half_details,
        "v1_artifact_mae_final": float(np.mean(np.abs(actual - v1))),
    }
    _emit_final_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
