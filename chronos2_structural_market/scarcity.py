from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


EPS = 1e-6


@dataclass(frozen=True)
class ScarcityParams:
    alpha: float
    tau_gw: float
    beta_ramp: float
    gamma_startup: float
    ramp_scale: float
    startup_scale: float
    probability_slope: float = 3.0
    probability_center: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return {
            key: float(value)
            for key, value in asdict(self).items()
        }


def sigmoid(values: np.ndarray) -> np.ndarray:
    x = np.clip(
        np.asarray(values, dtype=float),
        -40.0,
        40.0,
    )
    return 1.0 / (1.0 + np.exp(-x))


def robust_positive_scale(
    values: pd.Series | np.ndarray,
    *,
    quantile: float = 0.95,
    minimum: float = 1.0,
) -> float:
    array = np.asarray(values, dtype=float)
    array = np.abs(array[np.isfinite(array)])
    if len(array) == 0:
        return float(minimum)
    scale = float(np.quantile(array, quantile))
    return max(scale, float(minimum))


def scarcity_components(
    frame: pd.DataFrame,
    params: ScarcityParams,
) -> pd.DataFrame:
    reserve = pd.to_numeric(
        frame["milp_reserve_margin_gw"],
        errors="coerce",
    ).to_numpy(dtype=float)

    ramp = np.abs(
        pd.to_numeric(
            frame["milp_ramp_shadow_eur_mwh"],
            errors="coerce",
        ).to_numpy(dtype=float)
    )
    startups = np.maximum(
        pd.to_numeric(
            frame["milp_startups"],
            errors="coerce",
        ).to_numpy(dtype=float),
        0.0,
    )

    reserve_positive = np.maximum(reserve, 0.0)
    reserve_pressure = np.exp(
        -reserve_positive / max(params.tau_gw, EPS)
    )

    ramp_pressure = np.clip(
        ramp / max(params.ramp_scale, EPS),
        0.0,
        3.0,
    )
    startup_pressure = np.clip(
        startups / max(params.startup_scale, EPS),
        0.0,
        3.0,
    )

    score = params.probability_slope * (
        reserve_pressure
        + 0.50 * ramp_pressure
        + 0.50 * startup_pressure
        - params.probability_center
    )
    probability = sigmoid(score)

    raw_adder = (
        params.alpha * reserve_pressure
        + params.beta_ramp * ramp_pressure
        + params.gamma_startup * startup_pressure
    )
    adder = probability * np.maximum(raw_adder, 0.0)

    return pd.DataFrame(
        {
            "reserve_pressure": reserve_pressure,
            "ramp_pressure": ramp_pressure,
            "startup_pressure": startup_pressure,
            "milp_scarcity_probability": probability,
            "milp_scarcity_adder": adder,
        },
        index=frame.index,
    )


def apply_scarcity_layer(
    frame: pd.DataFrame,
    params: ScarcityParams,
) -> pd.DataFrame:
    result = frame.copy()
    raw_price = pd.to_numeric(
        result["milp_structural_price"],
        errors="coerce",
    )
    components = scarcity_components(
        result,
        params,
    )

    result["milp_structural_price_raw"] = raw_price
    result["milp_scarcity_probability"] = (
        components["milp_scarcity_probability"].to_numpy()
    )
    result["milp_scarcity_adder"] = (
        components["milp_scarcity_adder"].to_numpy()
    )
    result["milp_structural_price_adjusted"] = (
        raw_price.to_numpy(dtype=float)
        + result["milp_scarcity_adder"].to_numpy(dtype=float)
    )
    return result


def weighted_mae(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    extreme_threshold: float,
    extreme_weight: float,
) -> float:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    if not valid.any():
        return float("inf")
    y = y[valid]
    p = p[valid]

    weights = np.ones(len(y), dtype=float)
    weights += (
        float(extreme_weight)
        * (np.abs(y) >= float(extreme_threshold))
    )
    return float(
        np.average(
            np.abs(y - p),
            weights=weights,
        )
    )


def fit_scarcity_layer(
    train: pd.DataFrame,
    *,
    extreme_threshold: float = 150.0,
    extreme_weight: float = 2.0,
    bias_penalty: float = 0.20,
    seed: int = 42,
    maxiter: int = 100,
) -> ScarcityParams:
    required = [
        "actual_price",
        "milp_structural_price",
        "milp_reserve_margin_gw",
        "milp_ramp_shadow_eur_mwh",
        "milp_startups",
    ]
    clean = train.dropna(subset=required).copy()
    if len(clean) < 24 * 30:
        raise ValueError(
            "Pas assez d'observations pour calibrer la couche scarcity : "
            f"{len(clean)} lignes."
        )

    ramp_scale = robust_positive_scale(
        clean["milp_ramp_shadow_eur_mwh"],
        quantile=0.95,
        minimum=5.0,
    )
    startup_scale = robust_positive_scale(
        clean["milp_startups"],
        quantile=0.95,
        minimum=1.0,
    )

    actual = clean["actual_price"].to_numpy(dtype=float)

    def objective(vector: np.ndarray) -> float:
        alpha, tau_gw, beta_ramp, gamma_startup = vector
        params = ScarcityParams(
            alpha=float(alpha),
            tau_gw=float(tau_gw),
            beta_ramp=float(beta_ramp),
            gamma_startup=float(gamma_startup),
            ramp_scale=ramp_scale,
            startup_scale=startup_scale,
        )
        candidate = apply_scarcity_layer(
            clean,
            params,
        )["milp_structural_price_adjusted"].to_numpy(
            dtype=float
        )
        loss = weighted_mae(
            actual,
            candidate,
            extreme_threshold=extreme_threshold,
            extreme_weight=extreme_weight,
        )
        bias = abs(
            float(
                np.nanmean(
                    actual - candidate
                )
            )
        )
        return float(loss + bias_penalty * bias)

    result = differential_evolution(
        objective,
        bounds=[
            (0.0, 300.0),   # alpha
            (0.10, 15.0),   # tau_gw
            (0.0, 150.0),   # beta_ramp
            (0.0, 150.0),   # gamma_startup
        ],
        seed=int(seed),
        maxiter=int(maxiter),
        polish=True,
        updating="immediate",
        workers=1,
        tol=1e-7,
    )

    alpha, tau_gw, beta_ramp, gamma_startup = result.x
    return ScarcityParams(
        alpha=float(alpha),
        tau_gw=float(tau_gw),
        beta_ramp=float(beta_ramp),
        gamma_startup=float(gamma_startup),
        ramp_scale=ramp_scale,
        startup_scale=startup_scale,
    )


def metric_summary(
    frame: pd.DataFrame,
    price_column: str,
    *,
    extreme_threshold: float,
) -> dict[str, Any]:
    valid = frame[
        ["actual_price", price_column]
    ].dropna()
    if valid.empty:
        return {}

    actual = valid["actual_price"].to_numpy(dtype=float)
    predicted = valid[price_column].to_numpy(dtype=float)
    error = predicted - actual
    extreme = np.abs(actual) >= float(extreme_threshold)

    summary: dict[str, Any] = {
        "n": int(len(valid)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
    }
    if extreme.any():
        summary["extreme_n"] = int(extreme.sum())
        summary["extreme_mae"] = float(
            np.mean(
                np.abs(error[extreme])
            )
        )
        summary["extreme_bias"] = float(
            np.mean(error[extreme])
        )
    else:
        summary["extreme_n"] = 0
        summary["extreme_mae"] = float("nan")
        summary["extreme_bias"] = float("nan")
    return summary
