from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


EPS = 1e-6


@dataclass(frozen=True)
class ScarcityFeatureParams:
    alpha: float
    tau_gw: float
    beta_ramp: float
    gamma_startup: float
    ramp_scale: float
    startup_scale: float
    probability_slope: float = 3.0
    probability_center: float = 1.0

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
    ) -> "ScarcityFeatureParams":
        return cls(
            alpha=float(raw["alpha"]),
            tau_gw=float(raw["tau_gw"]),
            beta_ramp=float(raw["beta_ramp"]),
            gamma_startup=float(raw["gamma_startup"]),
            ramp_scale=float(raw["ramp_scale"]),
            startup_scale=float(raw["startup_scale"]),
            probability_slope=float(
                raw.get("probability_slope", 3.0)
            ),
            probability_center=float(
                raw.get("probability_center", 1.0)
            ),
        )


C5_FEATURE_COLUMNS = (
    "milp_reserve_pressure",
    "milp_ramp_pressure",
    "milp_startup_pressure",
    "milp_scarcity_probability_feature",
    "milp_scarcity_adder_feature",
    "milp_scarcity_probability_ewm",
    "milp_scarcity_adder_ewm",
    "milp_scarcity_delta",
    "milp_reserve_pressure_delta",
)


def sigmoid(values: np.ndarray) -> np.ndarray:
    x = np.clip(
        np.asarray(values, dtype=float),
        -40.0,
        40.0,
    )
    return 1.0 / (1.0 + np.exp(-x))


def _as_numeric(
    frame: pd.DataFrame,
    column: str,
) -> np.ndarray:
    return pd.to_numeric(
        frame[column],
        errors="coerce",
    ).to_numpy(dtype=float)


def build_scarcity_feature_frame(
    structural: pd.DataFrame,
    params: ScarcityFeatureParams,
    *,
    ewm_alpha: float = 0.65,
) -> pd.DataFrame:
    """Construit les features C5 sans modifier le prix structurel.

    Toutes les entrées sont issues du modèle structurel day-ahead. Le prix
    de référence reste milp_structural_price et n'est jamais corrigé ici.
    """
    required = (
        "milp_reserve_margin_gw",
        "milp_ramp_shadow_eur_mwh",
        "milp_startups",
    )
    missing = [
        column
        for column in required
        if column not in structural.columns
    ]
    if missing:
        raise KeyError(
            "Features structurelles absentes : "
            + ", ".join(missing)
        )

    if not 0.0 < float(ewm_alpha) <= 1.0:
        raise ValueError("ewm_alpha doit être dans ]0, 1].")

    reserve = _as_numeric(
        structural,
        "milp_reserve_margin_gw",
    )
    ramp = np.abs(
        _as_numeric(
            structural,
            "milp_ramp_shadow_eur_mwh",
        )
    )
    startups = np.maximum(
        _as_numeric(
            structural,
            "milp_startups",
        ),
        0.0,
    )

    reserve_pressure = np.exp(
        -np.maximum(reserve, 0.0)
        / max(params.tau_gw, EPS)
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
    adder = (
        probability
        * np.maximum(raw_adder, 0.0)
    )

    result = pd.DataFrame(
        {
            "milp_reserve_pressure": reserve_pressure,
            "milp_ramp_pressure": ramp_pressure,
            "milp_startup_pressure": startup_pressure,
            "milp_scarcity_probability_feature": probability,
            "milp_scarcity_adder_feature": adder,
        },
        index=structural.index,
    )

    # Lissage strictement causal : ewm(adjust=False) ne regarde jamais t+1.
    result["milp_scarcity_probability_ewm"] = (
        result[
            "milp_scarcity_probability_feature"
        ]
        .ewm(
            alpha=float(ewm_alpha),
            adjust=False,
        )
        .mean()
    )
    result["milp_scarcity_adder_ewm"] = (
        result[
            "milp_scarcity_adder_feature"
        ]
        .ewm(
            alpha=float(ewm_alpha),
            adjust=False,
        )
        .mean()
    )

    # Les deltas utilisent t et t-1 uniquement. Comme la courbe structurelle
    # D complète est connue au cutoff, ils sont exploitables comme known future.
    result["milp_scarcity_delta"] = (
        result[
            "milp_scarcity_probability_feature"
        ].diff()
    )
    result["milp_reserve_pressure_delta"] = (
        result["milp_reserve_pressure"].diff()
    )

    # Le premier delta n'a pas de t-1 dans le fichier. Une variation nulle est
    # plus neutre qu'un ffill depuis le futur.
    result[
        [
            "milp_scarcity_delta",
            "milp_reserve_pressure_delta",
        ]
    ] = result[
        [
            "milp_scarcity_delta",
            "milp_reserve_pressure_delta",
        ]
    ].fillna(0.0)

    return result.astype(np.float32)
