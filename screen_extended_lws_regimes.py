#!/usr/bin/env python
"""Strict EXT+A screen of causal FR load/wind/solar regime features.

All model candidates use the same frozen residual recipe:

    clip(0.50 * CatBoost_MAE + 0.50 * HistGBR_absolute_error, -40, 40)

The comparator and every candidate are fitted on the same EXT+A rows.  The
only difference is the non-price, D-1 08:00 PIT feature family.  B1 selection
must clear a deliberately high gain threshold on the aggregate and both fixed
30-day halves before B2 can be opened.  This utility has no final phase.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chronos2_modular.common import SeriesSpec, load_yaml
from chronos2_modular.data import read_pit_vintage_series
from screen_extended_residual_calibration import (
    TIMEZONE,
    _fit_models,
    _load_existing,
    _load_external,
    _load_features,
    _predict,
)


BASE_RECIPE = "0.50_cat_v1_plus_0.50_hgb31_clip40"
LOAD = "known_fr_load_fcst_oracle"
WIND = "known_fr_wind_generation_fcst_oracle"
SOLAR = "known_fr_solar_generation_fcst_oracle"


def _load_pit(
    path: Path,
    *,
    alias: str,
    index: pd.DatetimeIndex,
    scale: float,
    config: dict[str, Any],
) -> tuple[pd.Series, dict[str, Any]]:
    series, metadata = read_pit_vintage_series(
        path,
        SeriesSpec(
            alias=alias,
            source="pit_parquet",
            value_col="value",
            fill_method="none",
            known_future=True,
            future_strategies=("oracle",),
        ),
        TIMEZONE,
        config,
    )
    if int(metadata["cutoff_violations"]):
        raise RuntimeError(f"PIT cutoff violation in {path}")
    series.index = series.index.tz_convert("UTC")
    aligned = (series.reindex(index).astype(float) * float(scale)).rename(alias)
    return aligned, metadata


def _daily_profiles(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    local_days = pd.Series(
        frame.index.tz_convert(TIMEZONE).date,
        index=frame.index,
        name="delivery_day",
    )
    derived: dict[str, pd.Series] = {}
    for column in columns:
        values = frame[column].astype(float)
        grouped = values.groupby(local_days, sort=False)
        mean = grouped.transform("mean")
        minimum = grouped.transform("min")
        maximum = grouped.transform("max")
        ramp = grouped.diff().mask(grouped.cumcount().eq(0), 0.0)
        derived[f"{column}__day_mean"] = mean
        derived[f"{column}__day_min"] = minimum
        derived[f"{column}__day_max"] = maximum
        derived[f"{column}__day_range"] = maximum - minimum
        derived[f"{column}__day_centered"] = values - mean
        derived[f"{column}__ramp_1h"] = ramp
        derived[f"{column}__abs_ramp_1h"] = ramp.abs()
        derived[f"{column}__day_ramp_abs_max"] = ramp.abs().groupby(
            local_days, sort=False
        ).transform("max")
    return pd.DataFrame(derived, index=frame.index)


def _families(
    *,
    index: pd.DatetimeIndex,
    load: pd.Series,
    wind: pd.Series,
    solar: pd.Series,
    native_residual: pd.Series,
    chronos: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    raw = pd.concat([load, wind, solar], axis=1).reindex(index)
    raw["known_fr_load_fcst_oracle__missing"] = load.isna().astype(float)
    vre = wind + solar
    net = load - vre
    denominator = load.abs().clip(lower=1.0)
    physical = pd.DataFrame(
        {
            "fr_variable_renewables_fcst": vre,
            "fr_net_load_wind_solar_fcst": net,
            "fr_variable_renewables_share": vre / denominator,
            "fr_wind_share": wind / denominator,
            "fr_solar_share": solar / denominator,
            "fr_wind_solar_interaction": wind * solar,
            "fr_residual_load_identity_gap": native_residual - net,
        },
        index=index,
    )
    for threshold in (30.0, 20.0, 10.0, 0.0):
        label = int(threshold)
        physical[f"fr_net_load_below_{label}_gw"] = (
            threshold - net
        ).clip(lower=0.0)
        physical[f"fr_net_load_regime_below_{label}_gw"] = (
            net < threshold
        ).astype(float)

    compact_profiles = _daily_profiles(
        pd.concat([raw, physical], axis=1),
        [
            "fr_net_load_wind_solar_fcst",
            "fr_variable_renewables_fcst",
            "fr_variable_renewables_share",
        ],
    )
    full_profiles = _daily_profiles(
        pd.concat([raw, physical], axis=1),
        [
            LOAD,
            WIND,
            SOLAR,
            "fr_net_load_wind_solar_fcst",
            "fr_variable_renewables_fcst",
            "fr_variable_renewables_share",
        ],
    )

    q10 = chronos["chronos2__q10"].astype(float)
    q50 = chronos["chronos2__q50"].astype(float)
    q90 = chronos["chronos2__q90"].astype(float)
    width = q90 - q10
    negative_q10 = (-q10).clip(lower=0.0)
    negative_q50 = (-q50).clip(lower=0.0)
    regime = pd.DataFrame(
        {
            "chronos_q10_negative_magnitude": negative_q10,
            "chronos_q50_negative_magnitude": negative_q50,
            "chronos_q10_negative_flag": (q10 < 0.0).astype(float),
            "chronos_q50_negative_flag": (q50 < 0.0).astype(float),
        },
        index=index,
    )
    for threshold in (30, 20, 10, 0):
        hinge = physical[f"fr_net_load_below_{threshold}_gw"]
        flag = physical[f"fr_net_load_regime_below_{threshold}_gw"]
        regime[f"fr_rupture_{threshold}_x_q10_negative"] = hinge * (
            q10 < 0.0
        ).astype(float)
        regime[f"fr_rupture_{threshold}_x_q50_negative"] = hinge * (
            q50 < 0.0
        ).astype(float)
        regime[f"fr_rupture_{threshold}_x_chronos_width"] = flag * width

    return {
        "base_composite": pd.DataFrame(index=index),
        "raw_lws": raw,
        "net_load_compact": pd.concat([raw, physical], axis=1),
        "net_load_profiles": pd.concat(
            [raw, physical, compact_profiles], axis=1
        ),
        "rupture_regimes": pd.concat(
            [raw, physical, compact_profiles, regime], axis=1
        ),
        "full_physical_profiles": pd.concat(
            [raw, physical, full_profiles, regime], axis=1
        ),
    }


def _fit_frame(
    ext: pd.DataFrame,
    existing: pd.DataFrame,
    A_mask: np.ndarray,
) -> pd.DataFrame:
    if tuple(ext.columns) != tuple(existing.columns):
        raise RuntimeError("EXT and A candidate schemas differ")
    result = pd.concat([ext, existing.loc[A_mask]], axis=0)
    if result.index.has_duplicates or not result.index.is_monotonic_increasing:
        raise RuntimeError("EXT+A feature frame is not strictly chronological")
    return result


def _composite(models: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    prediction = _predict(models, X)
    return np.clip(
        0.5 * prediction["cat_v1"] + 0.5 * prediction["hgb31"],
        -40.0,
        40.0,
    )


def _mae(actual: np.ndarray, base: np.ndarray, correction: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - (base + correction))))


def _score_blocks(
    *,
    index: pd.DatetimeIndex,
    actual: np.ndarray,
    base: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    phase: str,
) -> dict[str, dict[str, float | int]]:
    local_days = pd.Index(index.tz_convert(TIMEZONE).date)
    days = local_days.unique()
    blocks = {
        phase.upper(): days,
        f"{phase.upper()}a": days[:30],
        f"{phase.upper()}b": days[30:],
    }
    result: dict[str, dict[str, float | int]] = {}
    for name, block_days in blocks.items():
        mask = np.asarray(local_days.isin(block_days), dtype=bool)
        old = _mae(actual[mask], base[mask], baseline[mask])
        new = _mae(actual[mask], base[mask], candidate[mask])
        result[name] = {
            "n": int(mask.sum()),
            "baseline_mae": old,
            "candidate_mae": new,
            "gain": old - new,
        }
    return result


def _daily_bootstrap(
    index: pd.DatetimeIndex,
    actual: np.ndarray,
    base: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    draws: int = 20_000,
) -> dict[str, float | int]:
    local_days = pd.Index(index.tz_convert(TIMEZONE).date)
    hourly = pd.Series(
        np.abs(actual - (base + baseline))
        - np.abs(actual - (base + candidate)),
        index=local_days,
    )
    daily = hourly.groupby(level=0, sort=False).mean().to_numpy(dtype=float)
    rng = np.random.default_rng(42)
    samples = rng.choice(daily, size=(draws, len(daily)), replace=True).mean(axis=1)
    return {
        "mean_daily_gain": float(daily.mean()),
        "bootstrap_95_low": float(np.quantile(samples, 0.025)),
        "bootstrap_95_high": float(np.quantile(samples, 0.975)),
        "days_better": int((daily > 0.0).sum()),
        "days_worse": int((daily < 0.0).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("b1", "b2"), required=True)
    parser.add_argument("--variant")
    parser.add_argument("--b1-passed", action="store_true")
    parser.add_argument("--minimum-block-gain", type=float, default=0.75)
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument(
        "--extended-oof-file",
        default="runs/chronos_oof_extended/supervised_oof_20240102_20240811.csv.gz",
    )
    parser.add_argument(
        "--run-dir", default="runs/chronos2_hourly_fr_residual_v1"
    )
    parser.add_argument(
        "--config", default="chronos2_hourly_fr_residual_saturn_v2.yaml"
    )
    parser.add_argument("--pit-ext", default="runs/tmp/pit_extended")
    parser.add_argument("--pit-cal", default="runs/tmp/pit_calibration")
    args = parser.parse_args()
    if args.phase == "b2" and (not args.variant or not args.b1_passed):
        parser.error("B2 requires --variant and explicit --b1-passed")

    run_dir = Path(args.run_dir).expanduser().resolve()
    X_all = _load_features(run_dir, include_final=False)
    external = _load_external(
        Path(args.extended_oof_file).expanduser().resolve(),
        X_all,
        schema="chronos_only",
    )
    existing = _load_existing(
        run_dir, X_all, include_final=False, schema="chronos_only"
    )
    days = existing["days"]
    local_days = existing["local_days"]
    masks = {
        "A": np.asarray(local_days.isin(days[:245]), dtype=bool),
        "B1": np.asarray(local_days.isin(days[245:305]), dtype=bool),
        "B2": np.asarray(local_days.isin(days[305:365]), dtype=bool),
    }
    config = load_yaml(Path(args.config).expanduser().resolve())
    pit_ext = Path(args.pit_ext).expanduser().resolve()
    pit_cal = Path(args.pit_cal).expanduser().resolve()

    def inputs(
        index: pd.DatetimeIndex,
        *,
        ext: bool,
    ) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, Any]]:
        root = pit_ext if ext else pit_cal
        names = (
            ("fr_load_ext.parquet", "ecmwf_wind_avg_ext.parquet", "ecmwf_solar_avg_ext.parquet")
            if ext
            else ("fr_load_fcst.parquet", "ecmwf_wind_avg_corrected.parquet", "ecmwf_solar_avg.parquet")
        )
        load, load_meta = _load_pit(
            root / names[0], alias=LOAD, index=index, scale=1.0, config=config
        )
        wind, wind_meta = _load_pit(
            root / names[1], alias=WIND, index=index, scale=0.001, config=config
        )
        solar, solar_meta = _load_pit(
            root / names[2], alias=SOLAR, index=index, scale=0.001, config=config
        )
        audit = {
            "load_coverage": float(load.notna().mean()),
            "wind_coverage": float(wind.notna().mean()),
            "solar_coverage": float(solar.notna().mean()),
            "load_missing": int(load.isna().sum()),
            "wind_missing": int(wind.isna().sum()),
            "solar_missing": int(solar.isna().sum()),
            "cutoff_violations": int(
                load_meta["cutoff_violations"]
                + wind_meta["cutoff_violations"]
                + solar_meta["cutoff_violations"]
            ),
            "load_missing_policy": "preserve NaN plus indicator; Cat native NaN, HGB median fit on EXT+A",
        }
        return load, wind, solar, audit

    ext_index = external["meta"].index
    cal_index = existing["meta"].index
    ext_load, ext_wind, ext_solar, ext_audit = inputs(ext_index, ext=True)
    cal_load, cal_wind, cal_solar, cal_audit = inputs(cal_index, ext=False)
    ext_native = X_all.loc[ext_index, "known_fr_residual_load_fcst_oracle"]
    cal_native = X_all.loc[cal_index, "known_fr_residual_load_fcst_oracle"]
    ext_chronos = external["raw"].loc[
        :, ["chronos2__q10", "chronos2__q50", "chronos2__q90"]
    ]
    cal_chronos = existing["raw"].loc[
        :, ["chronos2__q10", "chronos2__q50", "chronos2__q90"]
    ]
    ext_families = _families(
        index=ext_index,
        load=ext_load,
        wind=ext_wind,
        solar=ext_solar,
        native_residual=ext_native,
        chronos=ext_chronos,
    )
    cal_families = _families(
        index=cal_index,
        load=cal_load,
        wind=cal_wind,
        solar=cal_solar,
        native_residual=cal_native,
        chronos=cal_chronos,
    )

    variant_names = list(ext_families)
    if args.phase == "b2":
        if args.variant not in variant_names or args.variant == "base_composite":
            parser.error(f"Unknown/non-candidate variant: {args.variant}")
        variant_names = ["base_composite", args.variant]

    phase_mask = masks[args.phase.upper()]
    eval_index = cal_index[phase_mask]
    actual = existing["raw"].loc[eval_index, "actual"].to_numpy(dtype=float)
    base_q50 = existing["base_q50"].loc[eval_index].to_numpy(dtype=float)
    fit_y = pd.concat(
        [external["residual"], existing["residual"].loc[masks["A"]]], axis=0
    )

    predictions: dict[str, np.ndarray] = {}
    feature_counts: dict[str, int] = {}
    for variant in variant_names:
        ext_meta = pd.concat(
            [external["meta"], ext_families[variant]], axis=1
        )
        cal_meta = pd.concat(
            [existing["meta"], cal_families[variant]], axis=1
        )
        if not ext_meta.columns.is_unique or not cal_meta.columns.is_unique:
            raise RuntimeError(f"Duplicate feature in {variant}")
        fit_X = _fit_frame(ext_meta, cal_meta, masks["A"])
        if not fit_y.index.equals(fit_X.index):
            raise RuntimeError("Residual labels differ from EXT+A feature index")
        print(
            f"FIT {variant} | rows={len(fit_X)} features={fit_X.shape[1]}",
            flush=True,
        )
        models = _fit_models(fit_X, fit_y, args.threads)
        predictions[variant] = _composite(models, cal_meta.loc[eval_index])
        feature_counts[variant] = int(fit_X.shape[1])

    baseline = predictions["base_composite"]
    scores: dict[str, Any] = {}
    for variant in variant_names:
        block_scores = _score_blocks(
            index=eval_index,
            actual=actual,
            base=base_q50,
            baseline=baseline,
            candidate=predictions[variant],
            phase=args.phase,
        )
        robust = _daily_bootstrap(
            eval_index,
            actual,
            base_q50,
            baseline,
            predictions[variant],
        )
        scores[variant] = {
            "blocks": block_scores,
            **robust,
            "n_meta_features": feature_counts[variant],
            "passes_minimum_block_gain": bool(
                variant != "base_composite"
                and all(
                    float(details["gain"]) >= float(args.minimum_block_gain)
                    for details in block_scores.values()
                )
            ),
        }

    eligible = [
        name
        for name in variant_names
        if scores[name]["passes_minimum_block_gain"]
    ]
    phase_key = args.phase.upper()
    selected = (
        min(
            eligible,
            key=lambda name: float(
                scores[name]["blocks"][phase_key]["candidate_mae"]
            ),
        )
        if eligible
        else None
    )
    payload = {
        "protocol": {
            "fit_EXT": [str(external["days"][0]), str(external["days"][-1])],
            "fit_A": [str(days[0]), str(days[244])],
            "evaluation": [
                str(pd.Index(eval_index.tz_convert(TIMEZONE).date).unique()[0]),
                str(pd.Index(eval_index.tz_convert(TIMEZONE).date).unique()[-1]),
            ],
            "phase": args.phase,
            "frozen_recipe": BASE_RECIPE,
            "minimum_gain_each_aggregate_and_30d_block": float(
                args.minimum_block_gain
            ),
            "B2_not_read": args.phase == "b1",
            "final_not_loaded": True,
            "final_phase_implemented": False,
            "uses_storm": False,
            "uses_external_price_forecast": False,
        },
        "pit_audit": {"EXT": ext_audit, "calibration": cal_audit},
        "scores": scores,
        "selected": selected,
        "passed": selected is not None,
    }
    output = ROOT / "runs" / "tmp" / "pit_extended" / (
        f"lws_regimes_{args.phase}_results.json"
    )
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
