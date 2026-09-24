"""Paper-inspired sparse residual routing for sealed hourly forecasts.

The implementation follows the method described in *Foundation-MoE: Sparse
Residual Routing over Pretrained Time-Series Models for Day-Ahead Electricity
Price Forecasting* while making the unspecified engineering choices explicit
and configurable.

The class deliberately operates on already materialised, timestamp-aligned
expert forecasts.  It does not train or call foundation models itself.  This
keeps the router independent from Chronos-2 and, more importantly, makes it
possible to require genuinely out-of-fold expert evidence at fit time.

The paper's three-stage calibration is implemented literally:

1. a shared horizon bias estimated from training residuals;
2. a downward disagreement gate calibrated at the validation 60th percentile;
3. an upward tail-lift gate calibrated at the validation 40th percentile.

Only a common shift is applied to q10/q50/q90, so quantile order and interval
width are preserved by construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F


QUANTILES: tuple[str, str, str] = ("q10", "q50", "q90")
PREDICTION_MODES: tuple[str, ...] = (
    "paper_balanced",
    "mae_downward_only",
    "anchor_bias",
    "routed",
)
RESIDUAL_EXPERT_NAMES: tuple[str, str, str] = (
    "spike_up",
    "spike_down",
    "robust",
)
SERIALIZATION_VERSION = 1


class FoundationMoEError(ValueError):
    """Raised when the router's causal or tabular contract is violated."""


@dataclass(frozen=True)
class FoundationMoEConfig:
    """Explicit defaults for choices left unspecified by the preprint."""

    hidden_size: int = 64
    market_embedding_dim: int = 8
    horizon_embedding_dim: int = 8
    top_k: int = 2
    epochs: int = 250
    min_epochs: int = 40
    early_stopping_patience: int = 35
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    gradient_clip: float = 5.0
    huber_delta: float = 5.0
    balance_weight: float = 0.01
    tail_weight: float = 0.04
    underprediction_weight: float = 0.02
    oracle_weight: float = 0.15
    regret_weight: float = 0.35
    validation_tail_weight: float = 0.0
    initial_gain: float = 0.08
    smoothing_radius: int = 1
    initial_temperature: float = 1.0
    max_temperature: float = 2.5
    exploration_noise: float = 0.08
    max_exploration_noise: float = 0.5
    clrs_entropy_floor: float = 0.72
    clrs_max_share: float = 0.55
    clrs_idle_share: float = 0.005
    clrs_growth: float = 1.08
    clrs_decay: float = 0.96
    clrs_revival_step: float = 0.08
    clrs_revival_cap: float = 1.25
    down_quantile: float = 0.60
    up_quantile: float = 0.40
    down_weight: float = 1.0
    up_weight: float = 0.35
    max_abs_correction: float | None = 40.0
    minimum_training_tokens: int = 256
    minimum_validation_tokens: int = 64
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        positive_ints = {
            "hidden_size": self.hidden_size,
            "market_embedding_dim": self.market_embedding_dim,
            "horizon_embedding_dim": self.horizon_embedding_dim,
            "epochs": self.epochs,
            "min_epochs": self.min_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "minimum_training_tokens": self.minimum_training_tokens,
            "minimum_validation_tokens": self.minimum_validation_tokens,
        }
        invalid = [name for name, value in positive_ints.items() if int(value) < 1]
        if invalid:
            raise ValueError(f"Parametres entiers strictement positifs requis: {invalid}.")
        if self.top_k not in {1, 2}:
            raise ValueError("top_k doit valoir 1 ou 2.")
        if self.min_epochs > self.epochs:
            raise ValueError("min_epochs ne peut pas depasser epochs.")
        positive_floats = {
            "learning_rate": self.learning_rate,
            "gradient_clip": self.gradient_clip,
            "huber_delta": self.huber_delta,
            "initial_temperature": self.initial_temperature,
            "max_temperature": self.max_temperature,
            "clrs_growth": self.clrs_growth,
        }
        if any(not np.isfinite(value) or value <= 0.0 for value in positive_floats.values()):
            raise ValueError("Les taux, clips, temperatures et delta doivent etre positifs.")
        nonnegative = (
            self.weight_decay,
            self.balance_weight,
            self.tail_weight,
            self.underprediction_weight,
            self.oracle_weight,
            self.regret_weight,
            self.validation_tail_weight,
            self.exploration_noise,
            self.max_exploration_noise,
            self.clrs_idle_share,
            self.clrs_revival_step,
            self.clrs_revival_cap,
            self.down_weight,
            self.up_weight,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in nonnegative):
            raise ValueError("Les poids et amplitudes doivent etre finis et non negatifs.")
        unit_interval = {
            "initial_gain": self.initial_gain,
            "clrs_entropy_floor": self.clrs_entropy_floor,
            "clrs_max_share": self.clrs_max_share,
            "clrs_decay": self.clrs_decay,
            "down_quantile": self.down_quantile,
            "up_quantile": self.up_quantile,
        }
        if any(not 0.0 < value < 1.0 for value in unit_interval.values()):
            raise ValueError(f"Parametres attendus dans ]0, 1[: {list(unit_interval)}.")
        if self.max_temperature < self.initial_temperature:
            raise ValueError("max_temperature doit etre >= initial_temperature.")
        if self.max_exploration_noise < self.exploration_noise:
            raise ValueError(
                "max_exploration_noise doit etre >= exploration_noise."
            )
        if self.smoothing_radius not in {0, 1}:
            raise ValueError("Cette version accepte smoothing_radius=0 ou 1.")
        if self.max_abs_correction is not None and (
            not np.isfinite(self.max_abs_correction)
            or self.max_abs_correction <= 0.0
        ):
            raise ValueError("max_abs_correction doit etre positif ou null.")


@dataclass(frozen=True)
class FoundationMoEForecast:
    """Forecast quantiles and auditable token-level routing diagnostics."""

    predictions: pd.DataFrame
    diagnostics: pd.DataFrame


@dataclass
class _PreparedSplit:
    index: pd.DatetimeIndex
    numeric: Tensor
    direct: Tensor
    anchor_quantiles: Tensor
    market: Tensor
    horizon: Tensor
    previous: Tensor
    following: Tensor
    market_labels: np.ndarray
    curve_labels: np.ndarray
    target: Tensor | None = None


def _datetime_index(index: pd.Index, *, name: str) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{name} doit etre un DatetimeIndex.")
    if index.tz is None:
        raise FoundationMoEError(f"{name} doit etre timezone-aware.")
    if not index.is_monotonic_increasing:
        raise FoundationMoEError(f"{name} doit etre trie chronologiquement.")
    return index.tz_convert("UTC")


def _numeric_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} doit etre un DataFrame.")
    if frame.empty:
        raise FoundationMoEError(f"{name} ne peut pas etre vide.")
    if not frame.columns.is_unique or any(
        not isinstance(column, str) or not column for column in frame.columns
    ):
        raise FoundationMoEError(f"{name} doit avoir des colonnes texte uniques.")
    selected = list(frame.columns if columns is None else columns)
    missing = [column for column in selected if column not in frame]
    if missing:
        raise FoundationMoEError(f"Colonnes absentes de {name}: {missing}.")
    result = frame.loc[:, selected].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise FoundationMoEError(f"{name} contient une valeur manquante ou infinie.")
    return result.astype(float)


def _aligned_labels(
    values: pd.Series | Sequence[Any] | np.ndarray,
    index: pd.DatetimeIndex,
    *,
    name: str,
    stringify: bool,
) -> np.ndarray:
    if isinstance(values, pd.Series):
        if not values.index.equals(index):
            raise FoundationMoEError(f"{name}.index doit etre identique aux forecasts.")
        array = values.to_numpy(copy=False)
    else:
        array = np.asarray(values)
    if array.ndim != 1 or len(array) != len(index):
        raise FoundationMoEError(
            f"{name} doit etre unidimensionnel et de longueur {len(index)}."
        )
    if stringify:
        result = np.asarray([str(value) for value in array], dtype=object)
        if any(not value or value.casefold() in {"nan", "none", "nat"} for value in result):
            raise FoundationMoEError(f"{name} contient une etiquette vide.")
        return result
    numeric = pd.to_numeric(pd.Series(array), errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise FoundationMoEError(f"{name} doit contenir des entiers finis.")
    integer = numeric.astype(np.int64)
    if (integer < 0).any():
        raise FoundationMoEError(f"{name} ne peut pas contenir d'entier negatif.")
    return integer


def _target_series(
    target: pd.Series | Sequence[float] | np.ndarray,
    index: pd.DatetimeIndex,
    *,
    name: str,
) -> np.ndarray:
    if isinstance(target, pd.Series):
        if not target.index.equals(index):
            raise FoundationMoEError(f"{name}.index doit etre identique aux forecasts.")
        raw = target.to_numpy(copy=False)
    else:
        raw = np.asarray(target)
    if raw.ndim != 1 or len(raw) != len(index):
        raise FoundationMoEError(f"{name} doit avoir exactement {len(index)} valeurs.")
    result = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(result).all():
        raise FoundationMoEError(f"{name} contient une cible manquante ou infinie.")
    return result


def _curve_neighbours(
    market: np.ndarray,
    curve: np.ndarray,
    horizon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    previous = np.full(len(horizon), -1, dtype=np.int64)
    following = np.full(len(horizon), -1, dtype=np.int64)
    groups: dict[tuple[str, str], list[int]] = {}
    for position, key in enumerate(zip(market.tolist(), curve.tolist(), strict=True)):
        groups.setdefault((str(key[0]), str(key[1])), []).append(position)
    for key, positions in groups.items():
        ordered = sorted(positions, key=lambda position: int(horizon[position]))
        tokens = [int(horizon[position]) for position in ordered]
        if len(tokens) != len(set(tokens)):
            raise FoundationMoEError(
                f"Horizon duplique dans la courbe market={key[0]!r}, curve={key[1]!r}."
            )
        for offset, position in enumerate(ordered):
            if offset:
                previous[position] = ordered[offset - 1]
            if offset + 1 < len(ordered):
                following[position] = ordered[offset + 1]
    return previous, following


def _raw_forecast_features(
    experts: pd.DataFrame,
    anchor: pd.DataFrame,
    market: np.ndarray,
    horizon: np.ndarray,
    curve: np.ndarray,
    *,
    horizon_period: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    direct = experts.to_numpy(dtype=float)
    anchor_values = anchor.to_numpy(dtype=float)
    keys = pd.Series(
        [f"{m}\x1f{c}" for m, c in zip(market, curve, strict=True)],
        index=experts.index,
        dtype="string",
    )
    grouped = experts.groupby(keys.to_numpy(), sort=False)
    curve_mean = grouped.transform("mean").to_numpy(dtype=float)
    centered = experts - pd.DataFrame(
        curve_mean,
        index=experts.index,
        columns=experts.columns,
    )
    curve_std = np.sqrt(
        centered.pow(2).groupby(keys.to_numpy(), sort=False).transform("mean")
    ).to_numpy(dtype=float)
    curve_min = grouped.transform("min").to_numpy(dtype=float)
    curve_max = grouped.transform("max").to_numpy(dtype=float)
    point_mean = direct.mean(axis=1, keepdims=True)
    point_std = direct.std(axis=1, keepdims=True)
    point_min = direct.min(axis=1, keepdims=True)
    point_max = direct.max(axis=1, keepdims=True)
    point_range = point_max - point_min
    anchor_difference = direct - anchor_values[:, [1]]
    interval_width = (anchor_values[:, [2]] - anchor_values[:, [0]])
    angle = 2.0 * math.pi * horizon.astype(float) / float(horizon_period)
    periodic = np.column_stack([np.sin(angle), np.cos(angle)])
    raw = np.column_stack(
        [
            direct,
            curve_mean,
            curve_std,
            curve_min,
            curve_max,
            point_mean,
            point_std,
            point_min,
            point_max,
            point_range,
            anchor_difference,
            interval_width,
            periodic,
        ]
    )
    names: list[str] = []
    names.extend(f"token__{column}" for column in experts.columns)
    for statistic in ("curve_mean", "curve_std", "curve_min", "curve_max"):
        names.extend(f"{statistic}__{column}" for column in experts.columns)
    names.extend(("point_mean", "point_std", "point_min", "point_max", "point_range"))
    names.extend(f"anchor_difference__{column}" for column in experts.columns)
    names.extend(("anchor_interval_width", "horizon_sin", "horizon_cos"))
    if not np.isfinite(raw).all():
        raise RuntimeError("La construction des features Foundation-MoE a produit un non-fini.")
    return raw.astype(np.float32), tuple(names)


class _FoundationMoENetwork(nn.Module):
    def __init__(
        self,
        *,
        numeric_features: int,
        markets: int,
        horizons: int,
        direct_experts: int,
        foundation_indices: Sequence[int],
        anchor_index: int,
        config: FoundationMoEConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.anchor_index = int(anchor_index)
        self.top_k = int(config.top_k)
        self.market_embedding = nn.Embedding(markets, config.market_embedding_dim)
        self.horizon_embedding = nn.Embedding(horizons, config.horizon_embedding_dim)
        total_input = (
            numeric_features
            + config.market_embedding_dim
            + config.horizon_embedding_dim
        )
        self.shared = nn.Sequential(
            nn.Linear(total_input, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
        )
        self.router = nn.Linear(config.hidden_size, direct_experts + 3)
        self.upward = nn.Linear(config.hidden_size, 1)
        self.downward = nn.Linear(config.hidden_size, 1)
        self.robust = nn.Linear(config.hidden_size, 1)
        initial_logit = math.log(config.initial_gain / (1.0 - config.initial_gain))
        self.gain_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
        self.register_buffer(
            "foundation_indices",
            torch.as_tensor(tuple(foundation_indices), dtype=torch.long),
        )

    @staticmethod
    def _smooth(raw: Tensor, previous: Tensor, following: Tensor) -> Tensor:
        total = raw.clone()
        count = torch.ones_like(raw)
        previous_mask = previous.ge(0)
        if bool(previous_mask.any()):
            total[previous_mask] = total[previous_mask] + raw[previous[previous_mask]]
            count[previous_mask] = count[previous_mask] + 1.0
        following_mask = following.ge(0)
        if bool(following_mask.any()):
            total[following_mask] = total[following_mask] + raw[following[following_mask]]
            count[following_mask] = count[following_mask] + 1.0
        return total / count

    def forward(
        self,
        *,
        numeric: Tensor,
        market: Tensor,
        horizon: Tensor,
        direct: Tensor,
        previous: Tensor,
        following: Tensor,
        temperature: float,
        exploration_noise: float,
        revival_bias: Tensor,
    ) -> dict[str, Tensor]:
        embedded = torch.cat(
            [
                numeric,
                self.market_embedding(market),
                self.horizon_embedding(horizon),
            ],
            dim=1,
        )
        hidden = self.shared(embedded)
        foundation_mean = direct.index_select(1, self.foundation_indices).mean(
            dim=1
        )
        upward = foundation_mean + F.softplus(self.upward(hidden).squeeze(1))
        downward = foundation_mean - F.softplus(self.downward(hidden).squeeze(1))
        robust_residual = self.robust(hidden).squeeze(1)
        if self.config.smoothing_radius:
            robust_residual = self._smooth(robust_residual, previous, following)
        robust = foundation_mean + robust_residual
        candidates = torch.cat(
            [direct, upward[:, None], downward[:, None], robust[:, None]],
            dim=1,
        )
        logits = self.router(hidden)
        if exploration_noise > 0.0:
            logits = logits + torch.randn_like(logits) * exploration_noise
        probabilities = torch.softmax(
            (logits + revival_bias[None, :]) / float(temperature),
            dim=1,
        )
        top_values, top_indices = torch.topk(
            probabilities,
            k=self.top_k,
            dim=1,
        )
        top_weights = top_values / top_values.sum(dim=1, keepdim=True)
        selected = torch.gather(candidates, 1, top_indices)
        mixture = (selected * top_weights).sum(dim=1)
        anchor = direct[:, self.anchor_index]
        gain = torch.sigmoid(self.gain_logit)
        routed = anchor + gain * (mixture - anchor)
        return {
            "candidates": candidates,
            "logits": logits,
            "probabilities": probabilities,
            "top_indices": top_indices,
            "top_weights": top_weights,
            "mixture": mixture,
            "anchor": anchor,
            "gain": gain,
            "routed": routed,
        }


class FoundationMoEForecaster:
    """Top-k sparse residual router around a frozen anchor forecast."""

    def __init__(self, config: FoundationMoEConfig | None = None) -> None:
        self.config = config or FoundationMoEConfig()

    def _device(self) -> torch.device:
        requested = str(self.config.device).lower()
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return torch.device("xpu")
            return torch.device("cpu")
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise FoundationMoEError("device='cuda' demande mais CUDA est indisponible.")
        if device.type == "xpu" and not (
            hasattr(torch, "xpu") and torch.xpu.is_available()
        ):
            raise FoundationMoEError("device='xpu' demande mais XPU est indisponible.")
        return device

    def _validate_inputs(
        self,
        experts: pd.DataFrame,
        anchor_quantiles: pd.DataFrame,
        markets: pd.Series | Sequence[str] | np.ndarray,
        horizon_tokens: pd.Series | Sequence[int] | np.ndarray,
        curve_ids: pd.Series | Sequence[str] | np.ndarray,
        *,
        target: pd.Series | Sequence[float] | np.ndarray | None,
        name: str,
        expected_experts: Sequence[str] | None,
    ) -> tuple[
        pd.DatetimeIndex,
        pd.DataFrame,
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
    ]:
        index = _datetime_index(experts.index, name=f"{name}.index")
        if not anchor_quantiles.index.equals(experts.index):
            raise FoundationMoEError(
                f"{name}.anchor_quantiles.index doit etre identique aux experts."
            )
        direct = _numeric_frame(
            experts,
            name=f"{name}.experts",
            columns=expected_experts,
        )
        anchor = _numeric_frame(
            anchor_quantiles,
            name=f"{name}.anchor_quantiles",
            columns=QUANTILES,
        )
        crossed = (anchor["q10"] > anchor["q50"]) | (
            anchor["q50"] > anchor["q90"]
        )
        if bool(crossed.any()):
            raise FoundationMoEError(f"{name}: quantiles ancres croises.")
        market = _aligned_labels(
            markets,
            experts.index,
            name=f"{name}.markets",
            stringify=True,
        )
        horizon = _aligned_labels(
            horizon_tokens,
            experts.index,
            name=f"{name}.horizon_tokens",
            stringify=False,
        )
        curve = _aligned_labels(
            curve_ids,
            experts.index,
            name=f"{name}.curve_ids",
            stringify=True,
        )
        duplicate_keys = pd.DataFrame(
            {"timestamp": index.asi8, "market": market}
        ).duplicated()
        if bool(duplicate_keys.any()):
            raise FoundationMoEError(
                f"{name}: timestamp duplique au sein d'un meme marche."
            )
        target_values = (
            None
            if target is None
            else _target_series(target, experts.index, name=f"{name}.target")
        )
        return index, direct, anchor, market, horizon, curve, target_values

    def _prepare(
        self,
        experts: pd.DataFrame,
        anchor_quantiles: pd.DataFrame,
        markets: pd.Series | Sequence[str] | np.ndarray,
        horizon_tokens: pd.Series | Sequence[int] | np.ndarray,
        curve_ids: pd.Series | Sequence[str] | np.ndarray,
        *,
        target: pd.Series | Sequence[float] | np.ndarray | None,
        name: str,
        fit_schema: bool,
    ) -> _PreparedSplit:
        expected = None if fit_schema else self.expert_names_
        (
            index,
            direct,
            anchor,
            market_labels,
            horizon,
            curve_labels,
            target_values,
        ) = self._validate_inputs(
            experts,
            anchor_quantiles,
            markets,
            horizon_tokens,
            curve_ids,
            target=target,
            name=name,
            expected_experts=expected,
        )
        if fit_schema:
            self.expert_names_ = tuple(direct.columns)
            self.market_to_index_ = {
                market: position
                for position, market in enumerate(sorted(set(market_labels.tolist())))
            }
            self.horizon_count_ = int(horizon.max()) + 1
            self.horizons_seen_ = tuple(sorted(set(horizon.tolist())))
        unknown_markets = sorted(set(market_labels) - set(self.market_to_index_))
        if unknown_markets:
            raise FoundationMoEError(f"{name}: marches inconnus: {unknown_markets}.")
        unknown_horizons = sorted(set(horizon.tolist()) - set(self.horizons_seen_))
        if unknown_horizons:
            raise FoundationMoEError(f"{name}: horizons jamais vus: {unknown_horizons}.")
        raw, feature_names = _raw_forecast_features(
            direct,
            anchor,
            market_labels,
            horizon,
            curve_labels,
            horizon_period=self.horizon_count_,
        )
        if fit_schema:
            self.feature_names_ = feature_names
            mean = raw.mean(axis=0, dtype=np.float64)
            scale = raw.std(axis=0, dtype=np.float64)
            scale[scale < 1.0e-6] = 1.0
            self.feature_mean_ = mean.astype(np.float32)
            self.feature_scale_ = scale.astype(np.float32)
        elif feature_names != self.feature_names_:
            raise RuntimeError("Le schema des features Foundation-MoE a change.")
        normalized = (raw - self.feature_mean_) / self.feature_scale_
        previous, following = _curve_neighbours(
            market_labels,
            curve_labels,
            horizon,
        )
        market_index = np.asarray(
            [self.market_to_index_[value] for value in market_labels],
            dtype=np.int64,
        )
        device = self.device_
        return _PreparedSplit(
            index=index,
            numeric=torch.as_tensor(normalized, dtype=torch.float32, device=device),
            direct=torch.as_tensor(
                direct.to_numpy(dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
            anchor_quantiles=torch.as_tensor(
                anchor.to_numpy(dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
            market=torch.as_tensor(market_index, dtype=torch.long, device=device),
            horizon=torch.as_tensor(horizon, dtype=torch.long, device=device),
            previous=torch.as_tensor(previous, dtype=torch.long, device=device),
            following=torch.as_tensor(following, dtype=torch.long, device=device),
            market_labels=market_labels,
            curve_labels=curve_labels,
            target=(
                None
                if target_values is None
                else torch.as_tensor(
                    target_values,
                    dtype=torch.float32,
                    device=device,
                )
            ),
        )

    def _forward(
        self,
        split: _PreparedSplit,
        *,
        temperature: float,
        exploration_noise: float = 0.0,
        revival_bias: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if revival_bias is None:
            revival_bias = torch.zeros(
                len(self.candidate_names_),
                dtype=torch.float32,
                device=self.device_,
            )
        return self.network_(
            numeric=split.numeric,
            market=split.market,
            horizon=split.horizon,
            direct=split.direct,
            previous=split.previous,
            following=split.following,
            temperature=temperature,
            exploration_noise=exploration_noise,
            revival_bias=revival_bias,
        )

    @staticmethod
    def _utilization(output: Mapping[str, Tensor], experts: int) -> Tensor:
        selected = output["top_indices"].reshape(-1)
        counts = torch.bincount(selected, minlength=experts).to(dtype=torch.float32)
        return counts / float(selected.numel())

    def _horizon_bias(
        self,
        split: _PreparedSplit,
    ) -> Tensor:
        assert split.target is not None
        residual = split.target - split.anchor_quantiles[:, 1]
        bias = torch.zeros(
            self.horizon_count_,
            dtype=torch.float32,
            device=self.device_,
        )
        for token in self.horizons_seen_:
            mask = split.horizon.eq(int(token))
            if not bool(mask.any()):
                raise RuntimeError(f"Aucune observation train pour l'horizon {token}.")
            bias[int(token)] = residual[mask].mean()
        return bias

    def _calibrated_median(
        self,
        output: Mapping[str, Tensor],
        horizon: Tensor,
        *,
        horizon_bias: Tensor,
        down_threshold: Tensor,
        up_threshold: Tensor,
        mode: str,
    ) -> Tensor:
        if mode not in PREDICTION_MODES:
            raise ValueError(f"mode doit appartenir a {PREDICTION_MODES}.")
        anchor = output["anchor"]
        if mode == "routed":
            final = output["routed"]
        else:
            final = anchor + horizon_bias[horizon]
            routed_residual = output["routed"] - anchor
            downward = torch.relu(-routed_residual)
            upward = torch.relu(routed_residual)
            if mode in {"paper_balanced", "mae_downward_only"}:
                final = final - self.config.down_weight * downward * (
                    downward >= down_threshold
                )
            if mode == "paper_balanced":
                final = final + self.config.up_weight * upward * (
                    upward >= up_threshold
                )
        if self.config.max_abs_correction is not None:
            correction = torch.clamp(
                final - anchor,
                min=-float(self.config.max_abs_correction),
                max=float(self.config.max_abs_correction),
            )
            final = anchor + correction
        return final

    def _loss(
        self,
        output: Mapping[str, Tensor],
        target: Tensor,
        *,
        high_price_threshold: Tensor,
    ) -> tuple[Tensor, dict[str, float], Tensor]:
        error = output["routed"] - target
        huber = F.huber_loss(
            output["routed"],
            target,
            delta=self.config.huber_delta,
        )
        tail_mask = target >= high_price_threshold
        if bool(tail_mask.any()):
            tail = error[tail_mask].abs().mean()
            under = torch.relu(target[tail_mask] - output["routed"][tail_mask]).mean()
        else:
            tail = torch.zeros((), dtype=target.dtype, device=target.device)
            under = torch.zeros((), dtype=target.dtype, device=target.device)
        utilization = self._utilization(
            output,
            experts=len(self.candidate_names_),
        )
        mean_probability = output["probabilities"].mean(dim=0)
        balance = len(self.candidate_names_) * torch.sum(
            mean_probability * utilization.detach()
        )
        oracle = torch.argmin(
            torch.abs(output["candidates"].detach() - target[:, None]),
            dim=1,
        )
        # The paper defines the oracle term on the router distribution.  Use
        # the exact probabilities that also drive sparse selection so the
        # temperature, CLRS revival bias and exploration noise all participate
        # in this loss instead of silently optimizing the unadjusted logits.
        oracle_loss = F.nll_loss(
            torch.log(output["probabilities"].clamp_min(torch.finfo(target.dtype).tiny)),
            oracle,
        )
        regret = torch.relu(
            torch.abs(output["routed"] - target)
            - torch.abs(output["anchor"] - target)
        ).mean()
        total = (
            huber
            + self.config.balance_weight * balance
            + self.config.tail_weight * tail
            + self.config.underprediction_weight * under
            + self.config.oracle_weight * oracle_loss
            + self.config.regret_weight * regret
        )
        components = {
            "loss": float(total.detach().cpu()),
            "huber": float(huber.detach().cpu()),
            "balance": float(balance.detach().cpu()),
            "tail": float(tail.detach().cpu()),
            "under": float(under.detach().cpu()),
            "oracle": float(oracle_loss.detach().cpu()),
            "regret": float(regret.detach().cpu()),
        }
        return total, components, utilization

    def fit(
        self,
        train_experts: pd.DataFrame,
        train_target: pd.Series | Sequence[float] | np.ndarray,
        train_anchor_quantiles: pd.DataFrame,
        train_markets: pd.Series | Sequence[str] | np.ndarray,
        train_horizon_tokens: pd.Series | Sequence[int] | np.ndarray,
        train_curve_ids: pd.Series | Sequence[str] | np.ndarray,
        *,
        validation_experts: pd.DataFrame,
        validation_target: pd.Series | Sequence[float] | np.ndarray,
        validation_anchor_quantiles: pd.DataFrame,
        validation_markets: pd.Series | Sequence[str] | np.ndarray,
        validation_horizon_tokens: pd.Series | Sequence[int] | np.ndarray,
        validation_curve_ids: pd.Series | Sequence[str] | np.ndarray,
        anchor_expert: str,
        foundation_experts: Sequence[str] | None = None,
    ) -> "FoundationMoEForecaster":
        """Fit on sealed train experts and calibrate on a later validation set."""

        self.device_ = self._device()
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        train = self._prepare(
            train_experts,
            train_anchor_quantiles,
            train_markets,
            train_horizon_tokens,
            train_curve_ids,
            target=train_target,
            name="train",
            fit_schema=True,
        )
        validation = self._prepare(
            validation_experts,
            validation_anchor_quantiles,
            validation_markets,
            validation_horizon_tokens,
            validation_curve_ids,
            target=validation_target,
            name="validation",
            fit_schema=False,
        )
        if len(train.index) < self.config.minimum_training_tokens:
            raise FoundationMoEError(
                f"Train trop court: {len(train.index)} < "
                f"{self.config.minimum_training_tokens}."
            )
        if len(validation.index) < self.config.minimum_validation_tokens:
            raise FoundationMoEError(
                f"Validation trop courte: {len(validation.index)} < "
                f"{self.config.minimum_validation_tokens}."
            )
        if train.index.max() >= validation.index.min():
            raise FoundationMoEError(
                "Le train doit finir strictement avant le debut de la validation."
            )
        if anchor_expert not in self.expert_names_:
            raise FoundationMoEError(
                f"anchor_expert={anchor_expert!r} absent de {self.expert_names_}."
            )
        self.anchor_expert_ = str(anchor_expert)
        self.anchor_index_ = self.expert_names_.index(self.anchor_expert_)
        if not np.allclose(
            train_experts[self.anchor_expert_].to_numpy(dtype=float),
            train_anchor_quantiles["q50"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1.0e-7,
        ):
            raise FoundationMoEError("L'expert ancre train differe de anchor_quantiles.q50.")
        if not np.allclose(
            validation_experts[self.anchor_expert_].to_numpy(dtype=float),
            validation_anchor_quantiles["q50"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1.0e-7,
        ):
            raise FoundationMoEError(
                "L'expert ancre validation differe de anchor_quantiles.q50."
            )
        selected_foundations = tuple(
            self.expert_names_
            if foundation_experts is None
            else tuple(foundation_experts)
        )
        if not selected_foundations or len(set(selected_foundations)) != len(
            selected_foundations
        ):
            raise FoundationMoEError("foundation_experts doit etre non vide et unique.")
        missing_foundations = sorted(set(selected_foundations) - set(self.expert_names_))
        if missing_foundations:
            raise FoundationMoEError(
                f"foundation_experts absents: {missing_foundations}."
            )
        self.foundation_experts_ = selected_foundations
        self.foundation_indices_ = tuple(
            self.expert_names_.index(name) for name in selected_foundations
        )
        self.candidate_names_ = (*self.expert_names_, *RESIDUAL_EXPERT_NAMES)
        if self.config.top_k > len(self.candidate_names_):
            raise FoundationMoEError("top_k depasse le nombre de candidats.")
        self.network_ = _FoundationMoENetwork(
            numeric_features=len(self.feature_names_),
            markets=len(self.market_to_index_),
            horizons=self.horizon_count_,
            direct_experts=len(self.expert_names_),
            foundation_indices=self.foundation_indices_,
            anchor_index=self.anchor_index_,
            config=self.config,
        ).to(self.device_)
        assert train.target is not None and validation.target is not None
        horizon_bias = self._horizon_bias(train)
        high_price_threshold = torch.quantile(train.target, 0.95)
        optimizer = torch.optim.AdamW(
            self.network_.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        temperature = float(self.config.initial_temperature)
        exploration_noise = float(self.config.exploration_noise)
        idle_epochs = np.zeros(len(self.candidate_names_), dtype=np.int64)
        revival_bias = torch.zeros(
            len(self.candidate_names_),
            dtype=torch.float32,
            device=self.device_,
        )
        best: dict[str, Any] | None = None
        history: list[dict[str, Any]] = []
        stale_epochs = 0
        for epoch in range(1, self.config.epochs + 1):
            self.network_.train()
            optimizer.zero_grad(set_to_none=True)
            output = self._forward(
                train,
                temperature=temperature,
                exploration_noise=exploration_noise,
                revival_bias=revival_bias,
            )
            loss, components, utilization = self._loss(
                output,
                train.target,
                high_price_threshold=high_price_threshold,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.network_.parameters(),
                max_norm=self.config.gradient_clip,
            )
            optimizer.step()

            utilization_np = utilization.detach().cpu().numpy()
            entropy = float(
                -np.sum(utilization_np * np.log(utilization_np + 1.0e-12))
                / math.log(len(utilization_np))
            )
            max_share = float(utilization_np.max())
            collapsed = (
                entropy < self.config.clrs_entropy_floor
                or max_share > self.config.clrs_max_share
            )
            if collapsed:
                temperature = min(
                    self.config.max_temperature,
                    temperature * self.config.clrs_growth,
                )
                exploration_noise = min(
                    self.config.max_exploration_noise,
                    max(
                        self.config.exploration_noise,
                        exploration_noise + 0.02,
                    ),
                )
            else:
                temperature = max(
                    self.config.initial_temperature,
                    temperature * self.config.clrs_decay,
                )
                exploration_noise = max(
                    self.config.exploration_noise,
                    exploration_noise * self.config.clrs_decay,
                )
            idle_epochs = np.where(
                utilization_np < self.config.clrs_idle_share,
                idle_epochs + 1,
                0,
            )
            revival_np = np.minimum(
                self.config.clrs_revival_cap,
                idle_epochs * self.config.clrs_revival_step,
            ).astype(np.float32)
            revival_bias = torch.as_tensor(
                revival_np,
                dtype=torch.float32,
                device=self.device_,
            )

            self.network_.eval()
            with torch.no_grad():
                validation_output = self._forward(
                    validation,
                    temperature=temperature,
                )
                validation_residual = (
                    validation_output["routed"] - validation_output["anchor"]
                )
                down_threshold = torch.quantile(
                    torch.relu(-validation_residual),
                    self.config.down_quantile,
                )
                up_threshold = torch.quantile(
                    torch.relu(validation_residual),
                    self.config.up_quantile,
                )
                validation_prediction = self._calibrated_median(
                    validation_output,
                    validation.horizon,
                    horizon_bias=horizon_bias,
                    down_threshold=down_threshold,
                    up_threshold=up_threshold,
                    mode="paper_balanced",
                )
                validation_mae = float(
                    torch.mean(
                        torch.abs(validation_prediction - validation.target)
                    ).cpu()
                )
                validation_tail_mask = validation.target >= high_price_threshold
                validation_tail_mae = (
                    float(
                        torch.mean(
                            torch.abs(
                                validation_prediction[validation_tail_mask]
                                - validation.target[validation_tail_mask]
                            )
                        ).cpu()
                    )
                    if bool(validation_tail_mask.any())
                    else 0.0
                )
                validation_score = (
                    validation_mae
                    + self.config.validation_tail_weight * validation_tail_mae
                )
            history.append(
                {
                    "epoch": epoch,
                    **components,
                    "validation_mae": validation_mae,
                    "validation_tail_mae": validation_tail_mae,
                    "validation_score": validation_score,
                    "temperature": temperature,
                    "exploration_noise": exploration_noise,
                    "utilization_entropy": entropy,
                    "maximum_expert_share": max_share,
                    "residual_gain": float(output["gain"].detach().cpu()),
                }
            )
            if best is None or validation_score < best["score"] - 1.0e-7:
                best = {
                    "score": validation_score,
                    "validation_mae": validation_mae,
                    "validation_tail_mae": validation_tail_mae,
                    "epoch": epoch,
                    "temperature": temperature,
                    "state": {
                        name: value.detach().cpu().clone()
                        for name, value in self.network_.state_dict().items()
                    },
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            if (
                epoch >= self.config.min_epochs
                and stale_epochs >= self.config.early_stopping_patience
            ):
                break
        if best is None:
            raise RuntimeError("Foundation-MoE n'a produit aucun checkpoint valide.")
        self.network_.load_state_dict(best["state"])
        self.network_.eval()
        self.inference_temperature_ = float(best["temperature"])
        self.horizon_bias_ = horizon_bias.detach().cpu().numpy().astype(np.float32)
        self.high_price_threshold_ = float(high_price_threshold.detach().cpu())
        with torch.no_grad():
            validation_output = self._forward(
                validation,
                temperature=self.inference_temperature_,
            )
            validation_residual = (
                validation_output["routed"] - validation_output["anchor"]
            )
            self.down_threshold_ = float(
                torch.quantile(
                    torch.relu(-validation_residual),
                    self.config.down_quantile,
                ).cpu()
            )
            self.up_threshold_ = float(
                torch.quantile(
                    torch.relu(validation_residual),
                    self.config.up_quantile,
                ).cpu()
            )
            final_utilization = self._utilization(
                validation_output,
                len(self.candidate_names_),
            ).cpu().numpy()
        self.best_epoch_ = int(best["epoch"])
        self.validation_mae_ = float(best["validation_mae"])
        self.validation_tail_mae_ = float(best["validation_tail_mae"])
        self.history_ = pd.DataFrame(history)
        self.validation_utilization_ = {
            name: float(final_utilization[position])
            for position, name in enumerate(self.candidate_names_)
        }
        self.fit_end_utc_ = validation.index.max()
        self.n_training_tokens_ = len(train.index)
        self.n_validation_tokens_ = len(validation.index)
        self.is_fitted_ = True
        return self

    def predict(
        self,
        expert_predictions: pd.DataFrame,
        anchor_quantiles: pd.DataFrame,
        markets: pd.Series | Sequence[str] | np.ndarray,
        horizon_tokens: pd.Series | Sequence[int] | np.ndarray,
        curve_ids: pd.Series | Sequence[str] | np.ndarray,
        *,
        mode: Literal[
            "paper_balanced",
            "mae_downward_only",
            "anchor_bias",
            "routed",
        ] = "paper_balanced",
        require_after_validation: bool = True,
    ) -> FoundationMoEForecast:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("FoundationMoEForecaster doit etre entraine avant predict().")
        split = self._prepare(
            expert_predictions,
            anchor_quantiles,
            markets,
            horizon_tokens,
            curve_ids,
            target=None,
            name="predict",
            fit_schema=False,
        )
        if require_after_validation and split.index.min() <= self.fit_end_utc_:
            raise FoundationMoEError(
                "Les forecasts doivent commencer strictement apres la validation."
            )
        horizon_bias = torch.as_tensor(
            self.horizon_bias_,
            dtype=torch.float32,
            device=self.device_,
        )
        down_threshold = torch.tensor(
            self.down_threshold_,
            dtype=torch.float32,
            device=self.device_,
        )
        up_threshold = torch.tensor(
            self.up_threshold_,
            dtype=torch.float32,
            device=self.device_,
        )
        self.network_.eval()
        with torch.no_grad():
            output = self._forward(
                split,
                temperature=self.inference_temperature_,
            )
            median = self._calibrated_median(
                output,
                split.horizon,
                horizon_bias=horizon_bias,
                down_threshold=down_threshold,
                up_threshold=up_threshold,
                mode=mode,
            )
            shift = median - output["anchor"]
        # Apply the learned float32 shift to the caller's float64 quantiles.
        # This retains the source interval widths to machine precision instead
        # of first rounding the three anchor quantiles to float32.
        shift_values = shift.detach().cpu().numpy().astype(float)
        predictions = _numeric_frame(
            anchor_quantiles,
            name="predict.anchor_quantiles",
            columns=QUANTILES,
        ).add(
            shift_values,
            axis=0,
        )
        if bool(
            ((predictions["q10"] > predictions["q50"]) | (
                predictions["q50"] > predictions["q90"]
            )).any()
        ):
            raise RuntimeError("La correction Foundation-MoE a croise les quantiles.")
        top_indices = output["top_indices"].detach().cpu().numpy()
        top_weights = output["top_weights"].detach().cpu().numpy()
        routed = output["routed"].detach().cpu().numpy()
        anchor = output["anchor"].detach().cpu().numpy()
        routed_residual = routed - anchor
        horizon_values = split.horizon.detach().cpu().numpy()
        diagnostics = pd.DataFrame(
            {
                "market": split.market_labels,
                "curve_id": split.curve_labels,
                "horizon_token": horizon_values,
                "anchor_q50": anchor,
                "routed_q50": routed,
                "routed_residual": routed_residual,
                "horizon_bias": self.horizon_bias_[horizon_values],
                "downward_evidence": np.maximum(-routed_residual, 0.0),
                "upward_evidence": np.maximum(routed_residual, 0.0),
                "final_correction": predictions["q50"].to_numpy() - anchor,
                "top1_expert": [
                    self.candidate_names_[position] for position in top_indices[:, 0]
                ],
                "top1_weight": top_weights[:, 0],
            },
            index=expert_predictions.index.copy(),
        )
        if self.config.top_k == 2:
            diagnostics["top2_expert"] = [
                self.candidate_names_[position] for position in top_indices[:, 1]
            ]
            diagnostics["top2_weight"] = top_weights[:, 1]
        else:
            diagnostics["top2_expert"] = pd.NA
            diagnostics["top2_weight"] = np.nan
        predictions.attrs.update(
            {
                "mode": mode,
                "anchor_expert": self.anchor_expert_,
                "residual_gain": float(
                    torch.sigmoid(self.network_.gain_logit).detach().cpu()
                ),
                "down_threshold": self.down_threshold_,
                "up_threshold": self.up_threshold_,
                "inference_temperature": self.inference_temperature_,
            }
        )
        return FoundationMoEForecast(
            predictions=predictions,
            diagnostics=diagnostics,
        )

    def diagnostics(self) -> dict[str, Any]:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("FoundationMoEForecaster n'est pas entraine.")
        return {
            "serialization_version": SERIALIZATION_VERSION,
            "best_epoch": self.best_epoch_,
            "n_training_tokens": self.n_training_tokens_,
            "n_validation_tokens": self.n_validation_tokens_,
            "validation_mae": self.validation_mae_,
            "validation_tail_mae": self.validation_tail_mae_,
            "fit_end_utc": str(self.fit_end_utc_),
            "anchor_expert": self.anchor_expert_,
            "expert_names": list(self.expert_names_),
            "foundation_experts": list(self.foundation_experts_),
            "candidate_names": list(self.candidate_names_),
            "market_to_index": dict(self.market_to_index_),
            "horizons_seen": list(self.horizons_seen_),
            "high_price_threshold": self.high_price_threshold_,
            "down_threshold": self.down_threshold_,
            "up_threshold": self.up_threshold_,
            "inference_temperature": self.inference_temperature_,
            "residual_gain": float(
                torch.sigmoid(self.network_.gain_logit).detach().cpu()
            ),
            "validation_utilization": dict(self.validation_utilization_),
            "config": asdict(self.config),
        }

    def save(self, path: str | Path) -> Path:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("FoundationMoEForecaster n'est pas entraine.")
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "serialization_version": SERIALIZATION_VERSION,
            "config": asdict(self.config),
            "state_dict": {
                name: value.detach().cpu()
                for name, value in self.network_.state_dict().items()
            },
            "expert_names": list(self.expert_names_),
            "anchor_expert": self.anchor_expert_,
            "foundation_experts": list(self.foundation_experts_),
            "market_to_index": dict(self.market_to_index_),
            "horizons_seen": list(self.horizons_seen_),
            "horizon_count": self.horizon_count_,
            "feature_names": list(self.feature_names_),
            "feature_mean": torch.as_tensor(self.feature_mean_),
            "feature_scale": torch.as_tensor(self.feature_scale_),
            "horizon_bias": torch.as_tensor(self.horizon_bias_),
            "high_price_threshold": self.high_price_threshold_,
            "down_threshold": self.down_threshold_,
            "up_threshold": self.up_threshold_,
            "inference_temperature": self.inference_temperature_,
            "best_epoch": self.best_epoch_,
            "validation_mae": self.validation_mae_,
            "validation_tail_mae": self.validation_tail_mae_,
            "validation_utilization": dict(self.validation_utilization_),
            "fit_end_utc": str(self.fit_end_utc_),
            "n_training_tokens": self.n_training_tokens_,
            "n_validation_tokens": self.n_validation_tokens_,
            "history": self.history_.to_dict("records"),
        }
        torch.save(payload, output)
        return output

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
    ) -> "FoundationMoEForecaster":
        source = Path(path).expanduser().resolve()
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if payload.get("serialization_version") != SERIALIZATION_VERSION:
            raise FoundationMoEError("Version de checkpoint Foundation-MoE incompatible.")
        config_values = dict(payload["config"])
        if device is not None:
            config_values["device"] = device
        result = cls(FoundationMoEConfig(**config_values))
        result.device_ = result._device()
        result.expert_names_ = tuple(payload["expert_names"])
        result.anchor_expert_ = str(payload["anchor_expert"])
        result.anchor_index_ = result.expert_names_.index(result.anchor_expert_)
        result.foundation_experts_ = tuple(payload["foundation_experts"])
        result.foundation_indices_ = tuple(
            result.expert_names_.index(name) for name in result.foundation_experts_
        )
        result.candidate_names_ = (*result.expert_names_, *RESIDUAL_EXPERT_NAMES)
        result.market_to_index_ = {
            str(name): int(position)
            for name, position in payload["market_to_index"].items()
        }
        result.horizons_seen_ = tuple(int(value) for value in payload["horizons_seen"])
        result.horizon_count_ = int(payload["horizon_count"])
        result.feature_names_ = tuple(payload["feature_names"])
        result.feature_mean_ = payload["feature_mean"].numpy().astype(np.float32)
        result.feature_scale_ = payload["feature_scale"].numpy().astype(np.float32)
        result.horizon_bias_ = payload["horizon_bias"].numpy().astype(np.float32)
        result.network_ = _FoundationMoENetwork(
            numeric_features=len(result.feature_names_),
            markets=len(result.market_to_index_),
            horizons=result.horizon_count_,
            direct_experts=len(result.expert_names_),
            foundation_indices=result.foundation_indices_,
            anchor_index=result.anchor_index_,
            config=result.config,
        ).to(result.device_)
        result.network_.load_state_dict(payload["state_dict"])
        result.network_.eval()
        result.high_price_threshold_ = float(payload["high_price_threshold"])
        result.down_threshold_ = float(payload["down_threshold"])
        result.up_threshold_ = float(payload["up_threshold"])
        result.inference_temperature_ = float(payload["inference_temperature"])
        result.best_epoch_ = int(payload["best_epoch"])
        result.validation_mae_ = float(payload["validation_mae"])
        result.validation_tail_mae_ = float(payload["validation_tail_mae"])
        result.validation_utilization_ = {
            str(name): float(value)
            for name, value in payload["validation_utilization"].items()
        }
        result.fit_end_utc_ = pd.Timestamp(payload["fit_end_utc"])
        result.n_training_tokens_ = int(payload["n_training_tokens"])
        result.n_validation_tokens_ = int(payload["n_validation_tokens"])
        result.history_ = pd.DataFrame(payload["history"])
        result.is_fitted_ = True
        return result


__all__ = [
    "FoundationMoEConfig",
    "FoundationMoEError",
    "FoundationMoEForecast",
    "FoundationMoEForecaster",
    "PREDICTION_MODES",
    "QUANTILES",
]
