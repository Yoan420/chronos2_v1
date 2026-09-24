"""Adapter from hourly run artifacts to the existing standalone HTML report.

The project already ships a rich Plotly report in ``chronos2_modular.report``.
This module deliberately reuses that renderer so hourly and legacy runs keep
the same layout, cards, figures, statistics filters and embedded-JavaScript
behaviour.  Only the artifact-to-``ZoneRunResult`` conversion lives here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from chronos2_modular.common import ZoneData, ZoneRunResult
from chronos2_modular.metrics import compute_metrics, metric_breakdowns
from chronos2_modular.report import write_html_report
from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.storm_dashboard import (
    STORM_DASHBOARD_COLUMN,
    STORM_DASHBOARD_LIVE_FORECAST_ARTIFACT,
    STORM_DASHBOARD_LIVE_FORECAST_AUDIT,
)


REPORT_QUANTILES: tuple[float, ...] = tuple(
    round(value / 10.0, 1) for value in range(1, 10)
)

VARIABLE_ATTRIBUTION_HOURLY_ARTIFACT = "variable_attribution_hourly.csv.gz"
VARIABLE_ATTRIBUTION_AUDIT_ARTIFACT = "variable_attribution_audit.json"
VARIABLE_ATTRIBUTION_COLUMNS: tuple[str, ...] = (
    "delivery_start_utc",
    "delivery_start_local",
    "variant",
    "variable_key",
    "variable_label",
    "baseline_reference",
    "forecast_q50",
    "counterfactual_q50",
    "contribution_eur_mwh",
    "absolute_contribution_eur_mwh",
    "weight_pct",
)
VARIABLE_ATTRIBUTION_METHODS = frozenset(
    {
        "exact_grouped_shapley_end_to_end",
        "permutation_grouped_shapley_end_to_end",
    }
)
VARIABLE_ATTRIBUTION_MODEL_VARIANTS: dict[str, str] = {
    "residual_corrected": "autonomous",
    "mkonline_blend": "mkonline_blend",
}
VARIABLE_ATTRIBUTION_TOLERANCE = 1e-6

KALMAN_FILTER_AUDIT_ARTIFACT = "kalman_filter_audit.json"
KALMAN_DAILY_AUDIT_ARTIFACT = "kalman_daily_audit.csv"
KALMAN_STATE_AUDIT_ARTIFACT = "kalman_state_audit.csv.gz"
KALMAN_REPORT_MODELS = frozenset(
    {"residual_kalman", "residual_kalman_weather", "residual_kalman_hybrid"}
)

MODEL_LABELS: dict[str, str] = {
    "chronos_residual_load": "Prix aval — residual_load Chronos-2",
    "saturn_residual_load": "Prix aval — residual_load Saturn",
    "mkonline_blend": "Blend autonome + MKOnline primaire",
    "mkonline_primary": "MKOnline primaire",
    "storm_evaluation_only": (
        "Storm disponible à 08:00 (évaluation uniquement)"
    ),
    "storm_dashboard_official": (
        "Storm officiel dashboard (évaluation uniquement)"
    ),
    "storm_strict_08": "Storm snapshot 08:00 (diagnostic uniquement)",
    "residual_corrected": "Correcteur résiduel",
    "residual_kalman": "Correcteur résiduel + Kalman gouverné",
    "residual_kalman_weather": (
        "Correcteur résiduel + Kalman météo gouverné"
    ),
    "residual_kalman_hybrid": (
        "Correcteur résiduel + banque Kalman marché-météo-combustibles gouvernée"
    ),
    "ensemble": "Ensemble horaire",
    "chronos2": "Chronos-2",
    "catboost": "CatBoost",
    "lear": "LEAR",
}


STORM_AVAILABLE_0800_CONTRACT_ID = "storm_available_at_0800"
STORM_DASHBOARD_CONTRACT_ID = "storm_official_dashboard"

STORM_DASHBOARD_NATIVE_SERIES_BY_ZONE: dict[str, str] = {
    zone: f"power.price.{zone.lower()}.euromwh.h.fcst.3mv.storm.da.cache"
    for zone in ("FR", "DE", "BE", "NL")
}

STORM_TIMEZONE_BY_ZONE: dict[str, str] = {
    "FR": "Europe/Paris",
    "DE": "Europe/Berlin",
    "BE": "Europe/Brussels",
    "NL": "Europe/Amsterdam",
}

# These contracts are deliberately distinct.  The current sealed artifacts
# contain the strict 08:00 comparator only; they must never be presented as
# the metric shown by the official Storm dashboard.  The second entry defines
# the explicit PIT artifact seam used by report-only dashboard snapshots.


def storm_benchmark_contracts(
    zone: str = "FR",
    *,
    timezone: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Build zone-aware metadata without loading or scoring any comparator."""

    zone_key = str(zone).strip().upper()
    zone_token = zone_key.lower()
    local_timezone = timezone or STORM_TIMEZONE_BY_ZONE.get(
        zone_key, "Europe/Paris"
    )
    dashboard_series = STORM_DASHBOARD_NATIVE_SERIES_BY_ZONE.get(zone_key)
    dashboard_series_verified = dashboard_series is not None
    if dashboard_series is None:
        dashboard_series = (
            f"power.price.{zone_token}.euromwh.h.fcst.3mv.storm.da.cache"
        )
    return {
        STORM_AVAILABLE_0800_CONTRACT_ID: {
            "id": STORM_AVAILABLE_0800_CONTRACT_ID,
            "label": "Storm disponible à 08:00",
            "report_label": (
                "Storm disponible à 08:00 (évaluation uniquement)"
            ),
            "series": (
                f"power.price.{zone_token}.euromwh.h.fcst.3mv."
                "storm.da.basecase"
            ),
            "zone": zone_key,
            "timezone": local_timezone,
            "availability_rule": (
                "dernier snapshot et dernière révision disponibles au "
                f"cutoff civil J-1 08:00 {local_timezone}"
            ),
            "official_dashboard_metric": False,
            "report_note": (
                "comparateur strict de disponibilité : dernier snapshot et "
                "dernière révision disponibles au cutoff civil J-1 08:00 "
                f"{local_timezone}. Cette statistique ne reproduit pas la "
                "métrique « Storm officiel dashboard »."
            ),
        },
        STORM_DASHBOARD_CONTRACT_ID: {
            "id": STORM_DASHBOARD_CONTRACT_ID,
            "label": "Storm officiel dashboard",
            "report_label": "Storm officiel dashboard",
            "series": dashboard_series,
            "series_kind": "frozen_day_ahead_cache",
            "series_identifier_verified": dashboard_series_verified,
            "zone": zone_key,
            "timezone": local_timezone,
            "artifact_filename": (
                "inputs/storm_dashboard_official_statistics.parquet"
            ),
            "point_column": "storm_dashboard_official__q50",
            "official_dashboard_metric": True,
            "availability_rule": (
                "cache day-ahead GEMS figé; fallback natif audité uniquement "
                "pour les jours absents du cache"
            ),
            "status": "not_loaded_without_explicit_dashboard_artifact",
            "report_note": (
                "contrat officiel dashboard : cache day-ahead GEMS, avec "
                "fallback natif seulement sur les trous historiques, figé "
                "dans un artefact séparé et audité."
            ),
        },
    }


# Backward-compatible FR catalogue for callers that import the constant.
STORM_BENCHMARK_CONTRACTS = storm_benchmark_contracts("FR")


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: le JSON doit contenir un objet.")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _variable_attribution_groups(
    raw_groups: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("L'audit d'attribution doit declarer des groupes non vides.")
    groups: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    required = {
        "key",
        "label",
        "context_columns",
        "future_columns",
        "baseline_reference",
    }
    for position, raw in enumerate(raw_groups):
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"Groupe d'attribution {position}: un mapping est attendu."
            )
        missing = sorted(required.difference(raw))
        if missing:
            raise ValueError(
                f"Groupe d'attribution {position}: champs absents {missing}."
            )
        key = str(raw["key"]).strip()
        label = str(raw["label"]).strip()
        baseline = str(raw["baseline_reference"]).strip()
        context_columns = raw["context_columns"]
        future_columns = raw["future_columns"]
        if (
            not key
            or not label
            or not baseline
            or not isinstance(context_columns, list)
            or not isinstance(future_columns, list)
            or any(not str(value).strip() for value in context_columns)
            or any(not str(value).strip() for value in future_columns)
        ):
            raise ValueError(f"Groupe d'attribution invalide: {key or position!r}.")
        if key in by_key:
            raise ValueError(f"Groupe d'attribution duplique: {key!r}.")
        group = dict(raw)
        group.update(
            {
                "key": key,
                "label": label,
                "baseline_reference": baseline,
                "context_columns": [str(value) for value in context_columns],
                "future_columns": [str(value) for value in future_columns],
            }
        )
        groups.append(group)
        by_key[key] = group
    return groups, by_key


def _variable_attribution_architecture_weights(
    raw_weights: Any,
    *,
    variants: Sequence[str],
) -> dict[str, dict[str, float]]:
    if not isinstance(raw_weights, Mapping):
        raise TypeError("architecture_weights doit etre un mapping par variante.")
    if set(map(str, raw_weights)) != set(variants):
        raise ValueError(
            "architecture_weights doit couvrir exactement les variantes declarees."
        )
    normalized: dict[str, dict[str, float]] = {}
    expected_components = {"autonomous", "mkonline_primary"}
    for variant in variants:
        raw_variant = raw_weights.get(variant)
        if not isinstance(raw_variant, Mapping):
            raise TypeError(f"Poids d'architecture invalides pour {variant!r}.")
        if set(map(str, raw_variant)) != expected_components:
            raise ValueError(
                f"{variant}: poids autonomous/mkonline_primary exacts requis."
            )
        try:
            weights = {
                component: float(raw_variant[component])
                for component in sorted(expected_components)
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{variant}: poids d'architecture non numeriques.") from exc
        values = np.asarray(list(weights.values()), dtype=float)
        if (
            not np.isfinite(values).all()
            or bool((values < 0.0).any())
            or bool((values > 1.0).any())
            or not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-12)
        ):
            raise ValueError(f"{variant}: poids d'architecture invalides.")
        if variant == "autonomous" and weights != {
            "autonomous": 1.0,
            "mkonline_primary": 0.0,
        }:
            raise ValueError("La variante autonome doit avoir les poids 1/0.")
        normalized[variant] = weights
    return normalized


def _attach_variable_attribution(
    result: ZoneRunResult,
    *,
    directory: Path,
    native_model: str,
    timezone: str,
) -> None:
    """Attach one audited, report-only local Shapley decomposition.

    Old archives may omit both files.  Once either artifact exists, however,
    the pair and its complete contract are validated fail-closed before any
    value reaches the HTML renderer.

    A whole-directory Kalman replay can retain the autonomous attribution
    produced before the filter is applied.  In that case the original sealed
    forecast checksum and the current ``residual_corrected`` values are both
    validated, and the decomposition is attached with an explicit upstream
    scope.  It must never be labelled as an explanation of the final Kalman
    forecast.
    """

    hourly_path = directory / VARIABLE_ATTRIBUTION_HOURLY_ARTIFACT
    audit_path = directory / VARIABLE_ATTRIBUTION_AUDIT_ARTIFACT
    present = (hourly_path.is_file(), audit_path.is_file())
    if not any(present):
        return
    if not all(present):
        raise FileNotFoundError(
            "La paire variable_attribution_hourly/audit est incomplete."
        )
    upstream_kalman_attribution = native_model in KALMAN_REPORT_MODELS
    selected_variant = (
        "autonomous"
        if upstream_kalman_attribution
        else VARIABLE_ATTRIBUTION_MODEL_VARIANTS.get(native_model)
    )
    if selected_variant is None:
        raise ValueError(
            f"Attribution disponible mais modele non supporte: {native_model!r}."
        )

    audit = _read_json(audit_path)
    expected_scalars = {
        "schema_version": 1,
        "status": "complete",
        "zone": str(result.zone).upper(),
        "timezone": timezone,
        "quantile": "q50",
        "scope": "local_delivery_day",
        "used_for_prediction": False,
        "storm_used": False,
        "forecast_modified": False,
    }
    for key, expected in expected_scalars.items():
        if audit.get(key) != expected:
            raise ValueError(
                f"Audit attribution invalide: {key}={audit.get(key)!r}, "
                f"attendu {expected!r}."
            )
    method = str(audit.get("method", ""))
    if method not in VARIABLE_ATTRIBUTION_METHODS:
        raise ValueError(f"Methode d'attribution non reconnue: {method!r}.")
    raw_variants = audit.get("variants")
    if (
        not isinstance(raw_variants, list)
        or not raw_variants
        or any(str(value) not in {"autonomous", "mkonline_blend"} for value in raw_variants)
    ):
        raise ValueError("Liste de variantes d'attribution invalide.")
    variants = [str(value) for value in raw_variants]
    if len(variants) != len(set(variants)) or selected_variant not in variants:
        raise ValueError("Variante du rapport absente ou dupliquee dans l'audit.")
    architecture_weights = _variable_attribution_architecture_weights(
        audit.get("architecture_weights"),
        variants=variants,
    )
    groups, groups_by_key = _variable_attribution_groups(audit.get("groups"))

    forecast_path = directory / f"forecast_hourly_{str(result.zone).lower()}.csv"
    if not forecast_path.is_file() and str(result.zone).upper() == "FR":
        forecast_path = directory / "forecast_hourly_fr.csv"
    if not forecast_path.is_file():
        raise FileNotFoundError(forecast_path)
    declared_forecast_sha = str(audit.get("forecast_sha256", "")).lower()
    if len(declared_forecast_sha) != 64 or any(
        character not in "0123456789abcdef"
        for character in declared_forecast_sha
    ):
        raise ValueError("Le checksum forecast de l'attribution est invalide.")
    if upstream_kalman_attribution:
        checksum_manifest_path = directory / "artifact_checksums.json"
        if not checksum_manifest_path.is_file():
            raise FileNotFoundError(
                "Attribution amont Kalman sans artifact_checksums.json original."
            )
        checksum_manifest = _read_json(checksum_manifest_path)
        if checksum_manifest.get("algorithm") != "sha256":
            raise ValueError(
                "artifact_checksums.json doit utiliser sha256 pour "
                "l'attribution amont Kalman."
            )
        artifacts = checksum_manifest.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError(
                "artifact_checksums.json.artifacts est invalide pour "
                "l'attribution amont Kalman."
            )
        forecast_entries = [
            item
            for item in artifacts
            if isinstance(item, Mapping)
            and Path(str(item.get("path", ""))).name == forecast_path.name
        ]
        if len(forecast_entries) != 1:
            raise ValueError(
                "artifact_checksums.json doit identifier exactement un "
                "forecast original pour l'attribution amont Kalman."
            )
        original_forecast_sha = str(
            forecast_entries[0].get("sha256", "")
        ).lower()
        if (
            len(original_forecast_sha) != 64
            or any(
                character not in "0123456789abcdef"
                for character in original_forecast_sha
            )
            or original_forecast_sha != declared_forecast_sha
        ):
            raise ValueError(
                "Le checksum du forecast original diverge de l'attribution "
                "amont Kalman."
            )
    elif _sha256_file(forecast_path) != declared_forecast_sha:
        raise ValueError("Le checksum forecast de l'attribution est divergent.")

    forecast_timestamps = pd.DatetimeIndex(
        pd.to_datetime(result.forecast_native["timestamp"], utc=True, errors="raise")
    )
    if (
        forecast_timestamps.empty
        or forecast_timestamps.has_duplicates
        or not forecast_timestamps.is_monotonic_increasing
    ):
        raise ValueError("Timeline forecast invalide pour l'attribution.")
    local_days = pd.Index(forecast_timestamps.tz_convert(timezone).date).unique()
    if len(local_days) != 1:
        raise ValueError("L'attribution doit couvrir un seul jour civil local.")
    delivery_day = pd.Timestamp(local_days[0]).date().isoformat()
    if audit.get("delivery_day") != delivery_day:
        raise ValueError("Jour de livraison divergent dans l'audit attribution.")
    if int(audit.get("expected_hours", -1)) != len(forecast_timestamps):
        raise ValueError("expected_hours attribution divergent du forecast.")

    expected_selected_q50 = pd.Series(
        pd.to_numeric(result.forecast_native["q50"], errors="raise").to_numpy(
            dtype=float
        ),
        index=forecast_timestamps,
    )
    if upstream_kalman_attribution:
        current_forecast = pd.read_csv(forecast_path)
        upstream_columns = {
            "delivery_start_utc",
            "residual_corrected__q50",
        }
        missing_upstream_columns = sorted(
            upstream_columns.difference(current_forecast.columns)
        )
        if missing_upstream_columns:
            raise ValueError(
                "Forecast Kalman sans référence amont residual_corrected: "
                f"{missing_upstream_columns}."
            )
        upstream_timestamps = pd.DatetimeIndex(
            pd.to_datetime(
                current_forecast["delivery_start_utc"],
                utc=True,
                errors="raise",
            )
        )
        if (
            upstream_timestamps.has_duplicates
            or not upstream_timestamps.is_monotonic_increasing
            or not upstream_timestamps.equals(forecast_timestamps)
        ):
            raise ValueError(
                "Timeline residual_corrected divergente pour l'attribution "
                "amont Kalman."
            )
        upstream_q50 = pd.to_numeric(
            current_forecast["residual_corrected__q50"], errors="raise"
        ).to_numpy(dtype=float)
        if not np.isfinite(upstream_q50).all():
            raise ValueError(
                "P50 residual_corrected non finie pour l'attribution amont Kalman."
            )
        expected_selected_q50 = pd.Series(
            upstream_q50,
            index=forecast_timestamps,
        )

    try:
        reproduction_tolerance = float(
            audit.get("reproduction_tolerance_eur_mwh")
        )
        declared_base_error = float(
            audit.get("max_base_prediction_error_eur_mwh")
        )
        declared_reconstruction_error = float(
            audit.get("max_reconstruction_error_eur_mwh")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Audit des erreurs d'attribution invalide.") from exc
    audited_errors = np.asarray(
        [declared_base_error, declared_reconstruction_error], dtype=float
    )
    if (
        not np.isfinite(reproduction_tolerance)
        or reproduction_tolerance <= 0.0
        or not np.isfinite(audited_errors).all()
        or bool((audited_errors < 0.0).any())
        or bool((audited_errors > reproduction_tolerance).any())
    ):
        raise ValueError(
            "Erreurs d'attribution superieures a la tolerance de reproduction."
        )
    price_atol = max(1e-12, reproduction_tolerance)
    audit_comparison_atol = max(1e-10, reproduction_tolerance * 1e-3)

    frame = pd.read_csv(hourly_path, compression="infer")
    if tuple(map(str, frame.columns)) != VARIABLE_ATTRIBUTION_COLUMNS:
        raise ValueError(
            "Schema variable_attribution_hourly invalide: "
            f"{list(frame.columns)!r}."
        )
    if frame.empty:
        raise ValueError("variable_attribution_hourly est vide.")
    delivery_utc = pd.DatetimeIndex(
        pd.to_datetime(frame["delivery_start_utc"], utc=True, errors="raise")
    )
    delivery_local_as_utc = pd.DatetimeIndex(
        pd.to_datetime(frame["delivery_start_local"], utc=True, errors="raise")
    )
    if not delivery_utc.equals(delivery_local_as_utc):
        raise ValueError("Timestamps UTC/local divergents dans l'attribution.")
    frame["delivery_start_utc"] = delivery_utc
    frame["delivery_start_local"] = delivery_utc.tz_convert(timezone)

    text_columns = (
        "variant",
        "variable_key",
        "variable_label",
        "baseline_reference",
    )
    for column in text_columns:
        frame[column] = frame[column].astype(str).str.strip()
        if bool(frame[column].eq("").any()):
            raise ValueError(f"Valeur texte vide dans {column}.")
    numeric_columns = (
        "forecast_q50",
        "counterfactual_q50",
        "contribution_eur_mwh",
        "absolute_contribution_eur_mwh",
        "weight_pct",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    numeric = frame.loc[:, numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Valeur non finie dans l'attribution.")
    if bool(frame.duplicated(["delivery_start_utc", "variant", "variable_key"]).any()):
        raise ValueError("L'attribution contient des lignes dupliquees.")
    if set(frame["variant"]) != set(variants):
        raise ValueError("Les variantes CSV divergent de l'audit attribution.")
    if not np.allclose(
        frame["absolute_contribution_eur_mwh"].to_numpy(dtype=float),
        np.abs(frame["contribution_eur_mwh"].to_numpy(dtype=float)),
        rtol=0.0,
        atol=price_atol,
    ):
        raise ValueError("Contributions absolues incoherentes.")

    reconstruction_errors: list[float] = []
    expected_group_keys = set(groups_by_key)
    for variant in variants:
        block = frame.loc[frame["variant"] == variant].copy()
        if set(block["variable_key"]) != expected_group_keys:
            raise ValueError(f"{variant}: groupes CSV incomplets ou inconnus.")
        for key, group in groups_by_key.items():
            variable = block.loc[block["variable_key"] == key].sort_values(
                "delivery_start_utc", kind="stable"
            )
            variable_timestamps = pd.DatetimeIndex(variable["delivery_start_utc"])
            if not variable_timestamps.equals(forecast_timestamps):
                raise ValueError(f"{variant}/{key}: timeline attribution divergente.")
            if set(variable["variable_label"]) != {group["label"]}:
                raise ValueError(f"{variant}/{key}: label divergent de l'audit.")
            if set(variable["baseline_reference"]) != {
                group["baseline_reference"]
            }:
                raise ValueError(
                    f"{variant}/{key}: reference contrefactuelle divergente."
                )
            weight_values = variable["weight_pct"].to_numpy(dtype=float)
            if not np.allclose(
                weight_values,
                weight_values[0],
                rtol=0.0,
                atol=VARIABLE_ATTRIBUTION_TOLERANCE,
            ):
                raise ValueError(f"{variant}/{key}: poids non constant par heure.")

        forecast_by_hour = block.groupby("delivery_start_utc", sort=True)[
            "forecast_q50"
        ]
        baseline_by_hour = block.groupby("delivery_start_utc", sort=True)[
            "counterfactual_q50"
        ]
        if (
            bool(forecast_by_hour.nunique(dropna=False).gt(1).any())
            or bool(baseline_by_hour.nunique(dropna=False).gt(1).any())
        ):
            raise ValueError(
                f"{variant}: forecast ou reference Shapley varie entre groupes."
            )
        hourly_forecast = forecast_by_hour.first().reindex(forecast_timestamps)
        hourly_baseline = baseline_by_hour.first().reindex(forecast_timestamps)
        hourly_contributions = block.groupby("delivery_start_utc", sort=True)[
            "contribution_eur_mwh"
        ].sum().reindex(forecast_timestamps)
        error = np.abs(
            hourly_forecast.to_numpy(dtype=float)
            - hourly_baseline.to_numpy(dtype=float)
            - hourly_contributions.to_numpy(dtype=float)
        )
        reconstruction_errors.extend(error.tolist())
        if float(np.max(error)) > reproduction_tolerance:
            raise ValueError(f"{variant}: decomposition Shapley non additive.")

        absolute_by_group = block.groupby("variable_key", sort=False)[
            "absolute_contribution_eur_mwh"
        ].sum()
        total_absolute = float(absolute_by_group.sum())
        if not np.isfinite(total_absolute) or total_absolute < 0.0:
            raise ValueError(f"{variant}: masse absolue d'attribution invalide.")
        expected_weights = (
            pd.Series(0.0, index=absolute_by_group.index)
            if total_absolute <= VARIABLE_ATTRIBUTION_TOLERANCE
            else 100.0 * absolute_by_group / total_absolute
        )
        declared_weights = block.groupby("variable_key", sort=False)[
            "weight_pct"
        ].first()
        if not np.allclose(
            declared_weights.reindex(expected_weights.index).to_numpy(dtype=float),
            expected_weights.to_numpy(dtype=float),
            rtol=0.0,
            atol=VARIABLE_ATTRIBUTION_TOLERANCE,
        ) or not np.isclose(
            float(declared_weights.sum()),
            0.0 if total_absolute <= VARIABLE_ATTRIBUTION_TOLERANCE else 100.0,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(f"{variant}: normalisation weight_pct incoherente.")

        if variant == selected_variant:
            if not np.allclose(
                hourly_forecast.to_numpy(dtype=float),
                expected_selected_q50.to_numpy(dtype=float),
                rtol=0.0,
                atol=price_atol,
            ):
                scope = "amont residual_corrected" if upstream_kalman_attribution else "affichee"
                raise ValueError(f"Le forecast attribue diverge de la variante {scope}.")

    observed_reconstruction_error = float(
        max(reconstruction_errors, default=0.0)
    )
    if not np.isclose(
        observed_reconstruction_error,
        declared_reconstruction_error,
        rtol=0.0,
        atol=audit_comparison_atol,
    ):
        raise ValueError("Erreur de reconstruction declaree incoherente.")

    selected = frame.loc[frame["variant"] == selected_variant].copy()
    result.variable_attribution = {
        "variant": selected_variant,
        "scope": (
            "upstream_model"
            if upstream_kalman_attribution
            else "final_forecast"
        ),
        "is_upstream_attribution": upstream_kalman_attribution,
        "explained_model": (
            "residual_corrected"
            if upstream_kalman_attribution
            else native_model
        ),
        "explained_model_label": MODEL_LABELS.get(
            "residual_corrected" if upstream_kalman_attribution else native_model,
            "residual_corrected" if upstream_kalman_attribution else native_model,
        ),
        "reported_model": native_model,
        "reported_model_label": MODEL_LABELS.get(native_model, native_model),
        "hourly": selected,
        "audit": audit,
        "method": method,
        "groups": groups,
        "architecture_weights": architecture_weights[selected_variant],
        "max_reconstruction_error_eur_mwh": observed_reconstruction_error,
        "max_base_prediction_error_eur_mwh": declared_base_error,
        "reproduction_tolerance_eur_mwh": reproduction_tolerance,
    }


def _first_existing_artifact(
    directory: Path,
    filenames: Sequence[str],
) -> Path | None:
    for filename in filenames:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def _attach_kalman_diagnostics(
    result: ZoneRunResult,
    *,
    directory: Path,
    native_model: str,
    baseline_model: str | None,
    timezone: str,
) -> None:
    """Attach audited Kalman replay diagnostics when the artifacts exist.

    Historical reports predate this optional layer and therefore remain fully
    renderable when none of the files is present.  As soon as one Kalman
    artifact exists, the JSON audit is mandatory and the safety-critical
    causal contract is validated before the diagnostics can be displayed.
    """

    audit_path = directory / KALMAN_FILTER_AUDIT_ARTIFACT
    daily_path = _first_existing_artifact(
        directory,
        (KALMAN_DAILY_AUDIT_ARTIFACT, "kalman_daily_audit.csv.gz"),
    )
    state_path = _first_existing_artifact(
        directory,
        (KALMAN_STATE_AUDIT_ARTIFACT, "kalman_state_audit.csv"),
    )
    if not audit_path.is_file() and daily_path is None and state_path is None:
        return
    if not audit_path.is_file():
        raise FileNotFoundError(
            "Un artefact Kalman est présent sans kalman_filter_audit.json."
        )

    audit = _read_json(audit_path)
    if audit.get("schema_version") not in {1, 2}:
        raise ValueError(
            "Audit Kalman invalide: schema_version doit valoir 1 ou 2."
        )
    expected = {
        "status": "complete",
        "filter_only": True,
        "smoother_used": False,
        "em_used": False,
        "storm_used_as_input": False,
    }
    for key, expected_value in expected.items():
        if audit.get(key) != expected_value:
            raise ValueError(
                f"Audit Kalman invalide: {key}={audit.get(key)!r}, "
                f"attendu {expected_value!r}."
            )
    if int(audit.get("causality_violations", -1)) != 0:
        raise ValueError("L'audit Kalman signale une violation de causalité.")
    if int(audit.get("quantile_crossings", -1)) != 0:
        raise ValueError("L'audit Kalman signale un croisement de quantiles.")
    start_value = audit.get("evaluation_start_day")
    end_value = audit.get("evaluation_end_day")
    if start_value in (None, "") or end_value in (None, ""):
        raise ValueError("Les bornes d'évaluation Kalman sont absentes.")
    evaluation_start = pd.Timestamp(start_value).date()
    evaluation_end = pd.Timestamp(end_value).date()
    if (
        evaluation_end < evaluation_start
        or int(audit.get("evaluation_days", -1)) <= 0
        or int(audit.get("evaluation_hours", -1)) <= 0
        or int(audit.get("warmup_days", -1)) <= 0
    ):
        raise ValueError("Périodes de warm-up/évaluation Kalman invalides.")
    audited_model = str(audit.get("model_key", "")).strip()
    if audited_model != native_model:
        raise ValueError(
            "Le modèle de l'audit Kalman diverge du modèle natif du rapport."
        )
    audited_upstream = str(audit.get("upstream_model", "")).strip()
    if baseline_model and audited_upstream != baseline_model:
        raise ValueError(
            "L'upstream de l'audit Kalman diverge de la baseline du rapport."
        )
    if audit.get("used_for_storm") not in (False, None):
        raise ValueError("Storm ne peut pas intervenir dans le filtre Kalman.")

    raw_kinds = audit.get("candidate_kinds")
    if (
        not isinstance(raw_kinds, list)
        or not raw_kinds
        or any(not str(value).strip() for value in raw_kinds)
    ):
        raise ValueError("candidate_kinds est absent ou invalide dans l'audit Kalman.")
    candidate_kinds = tuple(str(value).strip() for value in raw_kinds)
    if len(candidate_kinds) != len(set(candidate_kinds)):
        raise ValueError("candidate_kinds contient des doublons.")
    allowed_filters = {*candidate_kinds, "identity"}

    daily = pd.DataFrame()
    if daily_path is not None:
        daily = _read_frame(daily_path)
        required_daily = {
            "local_day",
            "selected_filter",
            "selected_weight",
            "applied_correction_mean",
        }
        missing = sorted(required_daily.difference(daily.columns))
        if missing:
            raise ValueError(
                f"{daily_path}: colonnes Kalman quotidiennes absentes: {missing}."
            )
        daily = daily.copy()
        daily["local_day"] = pd.to_datetime(
            daily["local_day"], errors="raise"
        ).dt.normalize()
        if (
            daily.empty
            or bool(daily["local_day"].duplicated().any())
            or not daily["local_day"].is_monotonic_increasing
        ):
            raise ValueError("Timeline de l'audit Kalman quotidien invalide.")
        daily["selected_filter"] = daily["selected_filter"].astype(str).str.strip()
        unknown = sorted(set(daily["selected_filter"]).difference(allowed_filters))
        if unknown:
            raise ValueError(f"Filtres sélectionnés inconnus: {unknown}.")
        for column in (
            "selected_weight",
            "applied_correction_mean",
            "raw_correction_mean",
            "applied_correction_abs_max",
            "baseline_trailing_mae",
            "selected_trailing_mae",
        ):
            if column in daily:
                daily[column] = pd.to_numeric(daily[column], errors="coerce")
        weights = daily["selected_weight"].to_numpy(dtype=float)
        corrections = daily["applied_correction_mean"].to_numpy(dtype=float)
        if (
            not np.isfinite(weights).all()
            or bool(((weights < 0.0) | (weights > 1.0)).any())
            or not np.isfinite(corrections).all()
        ):
            raise ValueError("Poids ou corrections Kalman quotidiens invalides.")
        if "last_observation_used" in daily:
            last_used = pd.to_datetime(
                daily["last_observation_used"], errors="coerce"
            ).dt.normalize()
            causal = last_used.isna() | (last_used < daily["local_day"])
            if not bool(causal.all()):
                raise ValueError(
                    "L'audit quotidien Kalman utilise une observation du jour courant."
                )

    state = pd.DataFrame()
    if state_path is not None:
        state = _read_frame(state_path)
        required_state = {"local_day", "filter_kind"}
        missing = sorted(required_state.difference(state.columns))
        if missing:
            raise ValueError(
                f"{state_path}: colonnes d'état Kalman absentes: {missing}."
            )
        state = state.copy()
        state["local_day"] = pd.to_datetime(
            state["local_day"], errors="raise"
        ).dt.normalize()
        state["filter_kind"] = state["filter_kind"].astype(str).str.strip()
        unknown = sorted(set(state["filter_kind"]).difference(candidate_kinds))
        if unknown:
            raise ValueError(f"États de filtres inconnus: {unknown}.")
        if bool(state.duplicated(["local_day", "filter_kind"]).any()):
            raise ValueError("États Kalman quotidiens dupliqués.")
        for column in (
            "mean_innovation",
            "innovation_clips_total",
            "minimum_covariance_eigenvalue",
            "covariance_repairs_total",
        ):
            if column in state:
                state[column] = pd.to_numeric(state[column], errors="coerce")

    result.kalman_diagnostics = {
        "audit": audit,
        "daily": daily,
        "state": state,
        "timezone": timezone,
        "native_model": native_model,
        "baseline_model": baseline_model,
        "candidate_kinds": candidate_kinds,
        "artifact_paths": {
            "audit": audit_path.name,
            "daily": daily_path.name if daily_path is not None else None,
            "state": state_path.name if state_path is not None else None,
        },
    }


def _model_prefixes(model: str) -> tuple[str, ...]:
    if model == "ensemble":
        return ("ensemble", "ensemble_uncorrected")
    return (model,)


def _quantile_source_columns(
    frame: pd.DataFrame,
    model: str,
    *,
    allow_final_columns: bool,
) -> dict[str, str]:
    for prefix in _model_prefixes(model):
        columns = {
            quantile: f"{prefix}__{quantile}"
            for quantile in ("q10", "q50", "q90")
        }
        if all(column in frame for column in columns.values()):
            return columns
    if allow_final_columns:
        columns = {quantile: quantile for quantile in ("q10", "q50", "q90")}
        if all(column in frame for column in columns.values()):
            return columns
    raise ValueError(
        f"Quantiles q10/q50/q90 introuvables pour le modèle {model!r}."
    )


def _expand_report_quantiles(
    frame: pd.DataFrame,
    columns: Mapping[str, str],
) -> pd.DataFrame:
    """Expand q10/q50/q90 monotonically for the legacy CRPS display.

    The exact three quantiles are preserved.  Intermediate deciles are only a
    piecewise-linear display approximation required by the legacy report's
    nine-quantile CRPS function; point metrics and interval coverage remain
    based on the original hourly artifacts.
    """

    q10 = pd.to_numeric(frame[columns["q10"]], errors="coerce").to_numpy(
        dtype=float
    )
    q50 = pd.to_numeric(frame[columns["q50"]], errors="coerce").to_numpy(
        dtype=float
    )
    q90 = pd.to_numeric(frame[columns["q90"]], errors="coerce").to_numpy(
        dtype=float
    )
    result = pd.DataFrame(index=frame.index)
    for quantile in REPORT_QUANTILES:
        name = f"q{int(100 * quantile):02d}"
        if quantile <= 0.5:
            weight = (quantile - 0.1) / 0.4
            values = q10 + weight * (q50 - q10)
        else:
            weight = (quantile - 0.5) / 0.4
            values = q50 + weight * (q90 - q50)
        result[name] = values
    result["point"] = q50
    return result


def _evaluation_mask(
    frame: pd.DataFrame,
    *,
    run_dir: Path,
    timezone: str,
) -> np.ndarray:
    metrics_payload = _read_json(run_dir / "metrics_hourly.json")
    diagnostics = metrics_payload.get("training_diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    start = diagnostics.get("evaluation_start_local_date")
    end = diagnostics.get("evaluation_end_local_date")
    if not start or not end:
        return np.ones(len(frame), dtype=bool)
    delivery = pd.to_datetime(
        frame["delivery_start_utc"],
        utc=True,
        errors="raise",
    ).dt.tz_convert(timezone)
    dates = pd.Index(delivery.dt.date)
    return np.asarray(
        (dates >= pd.Timestamp(start).date())
        & (dates <= pd.Timestamp(end).date()),
        dtype=bool,
    )


def _kalman_evaluation_mask(
    frame: pd.DataFrame,
    *,
    run_dir: Path,
    native_model: str,
    timezone: str,
) -> np.ndarray:
    """Constrain Kalman model cards to the audited rolling-365 suffix."""

    if native_model not in KALMAN_REPORT_MODELS:
        return np.ones(len(frame), dtype=bool)
    audit_path = run_dir / KALMAN_FILTER_AUDIT_ARTIFACT
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"Un rapport {native_model} exige kalman_filter_audit.json."
        )
    audit = _read_json(audit_path)
    start_value = audit.get("evaluation_start_day")
    end_value = audit.get("evaluation_end_day")
    if start_value in (None, "") or end_value in (None, ""):
        raise ValueError(
            "L'audit Kalman doit déclarer evaluation_start_day/end_day."
        )
    start = pd.Timestamp(start_value).date()
    end = pd.Timestamp(end_value).date()
    calendar_days = (end - start).days + 1
    if calendar_days != 365 or int(audit.get("evaluation_days", -1)) != 365:
        raise ValueError(
            "Le rapport Kalman exige une fenêtre d'évaluation de 365 jours."
        )
    delivery = pd.to_datetime(
        frame["delivery_start_utc"], utc=True, errors="raise"
    ).dt.tz_convert(timezone)
    local_days = pd.Index(delivery.dt.date)
    return np.asarray((local_days >= start) & (local_days <= end), dtype=bool)


def _backtest_prediction_frame(
    raw: pd.DataFrame,
    *,
    model: str,
    row_mask: np.ndarray,
    timezone: str,
    allow_missing_actual: bool = False,
) -> pd.DataFrame:
    columns = _quantile_source_columns(
        raw,
        model,
        allow_final_columns=False,
    )
    selected = raw.loc[row_mask].copy()
    quantiles = _expand_report_quantiles(selected, columns)
    delivery = pd.to_datetime(
        selected["delivery_start_utc"],
        utc=True,
        errors="raise",
    ).dt.tz_convert(timezone)
    model_origin_column = f"{model}_forecast_origin_utc"
    origin_column = (
        model_origin_column
        if model_origin_column in selected
        else "forecast_origin_utc"
    )
    if origin_column in selected:
        origin = pd.to_datetime(
            selected[origin_column],
            utc=True,
            errors="coerce",
        ).dt.tz_convert(timezone)
    else:
        origin = delivery.dt.normalize() - pd.Timedelta(days=1) + pd.Timedelta(
            hours=8
        )
    result = quantiles.copy()
    result.insert(0, "timestamp", delivery.to_numpy())
    result.insert(1, "origin_timestamp", origin.to_numpy())
    result["actual"] = pd.to_numeric(
        selected["actual"], errors="coerce"
    ).to_numpy(dtype=float)
    result = result.sort_values("timestamp", kind="stable").reset_index(drop=True)
    local_dates = pd.Series(result["timestamp"]).dt.date
    result["horizon_step"] = (
        result.groupby(local_dates, sort=False).cumcount() + 1
    )
    required = [
        "point",
        *[f"q{int(100 * value):02d}" for value in REPORT_QUANTILES],
    ]
    if not allow_missing_actual:
        required.insert(0, "actual")
    finite = np.isfinite(result[required].to_numpy(dtype=float)).all(axis=1)
    finite &= pd.notna(result["origin_timestamp"]).to_numpy()
    result = result.loc[finite].reset_index(drop=True)
    if result.empty:
        raise ValueError(f"Aucune prédiction évaluable pour {model!r}.")
    return result


def _backtest_point_prediction_frame(
    raw: pd.DataFrame,
    *,
    model: str,
    row_mask: np.ndarray,
    timezone: str,
    allow_missing_actual: bool = False,
) -> pd.DataFrame:
    """Build an evaluation-only point forecast frame.

    Unlike a model shown in the probabilistic charts, an evaluation benchmark
    only needs a P50.  In particular this lets the report compare the frozen
    candidate with Storm without inventing P10/P90 values and without loading
    Storm into the live-forecast path.
    """

    point_column = f"{model}__q50"
    if point_column not in raw:
        raise ValueError(
            f"Prévision centrale introuvable pour le benchmark {model!r}."
        )
    selected = raw.loc[row_mask].copy()
    delivery = pd.to_datetime(
        selected["delivery_start_utc"],
        utc=True,
        errors="raise",
    ).dt.tz_convert(timezone)
    result = pd.DataFrame(
        {
            "timestamp": delivery.to_numpy(),
            "actual": pd.to_numeric(
                selected["actual"], errors="coerce"
            ).to_numpy(dtype=float),
            "q50": pd.to_numeric(
                selected[point_column], errors="coerce"
            ).to_numpy(dtype=float),
        }
    )
    result["point"] = result["q50"]
    result = result.sort_values("timestamp", kind="stable").reset_index(
        drop=True
    )
    actual_finite = np.isfinite(
        pd.to_numeric(result["actual"], errors="coerce").to_numpy(dtype=float)
    )
    point_finite = np.isfinite(
        pd.to_numeric(result["q50"], errors="coerce").to_numpy(dtype=float)
    )
    finite = (
        point_finite
        if not allow_missing_actual
        else (point_finite | ~actual_finite)
    )
    if not allow_missing_actual:
        finite &= actual_finite
    result = result.loc[finite].reset_index(drop=True)
    if result.empty:
        raise ValueError(f"Aucune prédiction évaluable pour {model!r}.")
    return result


def _forecast_prediction_frame(
    raw: pd.DataFrame,
    *,
    model: str,
    timezone: str,
) -> pd.DataFrame:
    columns = _quantile_source_columns(
        raw,
        model,
        allow_final_columns=True,
    )
    quantiles = _expand_report_quantiles(raw, columns)
    delivery = pd.to_datetime(
        raw["delivery_start_utc"],
        utc=True,
        errors="raise",
    ).dt.tz_convert(timezone)
    result = quantiles.copy()
    result.insert(0, "timestamp", delivery.to_numpy())
    return result.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _zone_data(
    run_dir: Path,
    *,
    zone: str,
    timezone: str,
) -> ZoneData:
    inputs = run_dir / "inputs"
    aligned = _read_frame(inputs / "aligned_inputs.csv.gz")
    aligned_index = pd.to_datetime(
        aligned.pop("timestamp"),
        utc=True,
        errors="raise",
    ).dt.tz_convert(timezone)
    aligned.index = pd.DatetimeIndex(aligned_index, name="timestamp")
    target = pd.to_numeric(aligned.pop("target"), errors="coerce").rename(
        "target"
    )
    covariates = aligned.apply(pd.to_numeric, errors="coerce")

    context = _read_frame(inputs / "model_covariates_with_future.csv.gz")
    context_index = pd.to_datetime(
        context.pop("timestamp"),
        utc=True,
        errors="raise",
    ).dt.tz_convert(timezone)
    context.index = pd.DatetimeIndex(context_index, name="timestamp")
    context = context.apply(pd.to_numeric, errors="coerce")

    coverage = _read_frame(inputs / "input_coverage.csv")
    manifest = _read_frame(inputs / "input_manifest.csv")
    known_future = []
    if {"alias", "known_future"}.issubset(manifest):
        known_future = manifest.loc[
            manifest["known_future"].astype(str).str.lower().eq("true"),
            "alias",
        ].astype(str).tolist()
    diagnostics = _read_json(run_dir / "run_manifest.json")
    return ZoneData(
        zone=zone,
        timezone=timezone,
        frequency="h",
        target=target,
        covariates=covariates,
        model_context_covariates=context,
        known_future_columns=known_future,
        coverage=coverage,
        input_manifest=manifest,
        diagnostics=diagnostics,
    )


def _default_models(backtest: pd.DataFrame) -> tuple[str, str | None]:
    if "residual_kalman_hybrid__q50" in backtest:
        baseline = (
            "residual_corrected"
            if "residual_corrected__q50" in backtest
            else None
        )
        return "residual_kalman_hybrid", baseline
    if "residual_kalman_weather__q50" in backtest:
        baseline = (
            "residual_corrected"
            if "residual_corrected__q50" in backtest
            else None
        )
        return "residual_kalman_weather", baseline
    if "residual_kalman__q50" in backtest:
        baseline = (
            "residual_corrected"
            if "residual_corrected__q50" in backtest
            else None
        )
        return "residual_kalman", baseline
    if "residual_corrected__q50" in backtest:
        return "residual_corrected", "ensemble"
    if "ensemble__q50" in backtest:
        return "ensemble", "chronos2"
    if "chronos2__q50" in backtest:
        return "chronos2", None
    raise ValueError("Aucun modèle horaire reconnu dans le backtest.")


def _attach_live_storm_forecast(
    result: ZoneRunResult,
    *,
    directory: Path,
) -> None:
    artifact = directory / STORM_DASHBOARD_LIVE_FORECAST_ARTIFACT
    if not artifact.is_file():
        return
    audit_path = directory / STORM_DASHBOARD_LIVE_FORECAST_AUDIT
    if not audit_path.is_file():
        raise FileNotFoundError(f"Audit Storm du jour absent: {audit_path}")
    audit = _read_json(audit_path)
    if (
        audit.get("status") != "complete"
        or audit.get("used_for_prediction") is not False
        or str(audit.get("zone", "")).upper() != result.zone.upper()
    ):
        raise ValueError("Audit Storm du jour invalide ou ambigu.")
    frame = pd.read_parquet(artifact)
    required = {"delivery_start_utc", STORM_DASHBOARD_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"{artifact}: colonnes Storm du jour absentes: {missing}"
        )
    delivery = pd.DatetimeIndex(
        pd.to_datetime(frame["delivery_start_utc"], utc=True, errors="raise")
    )
    expected = pd.DatetimeIndex(
        pd.to_datetime(result.forecast_native["timestamp"], utc=True, errors="raise")
    )
    if (
        delivery.has_duplicates
        or not delivery.is_monotonic_increasing
        or not delivery.equals(expected)
    ):
        raise ValueError(
            "La timeline Storm du jour diffère de celle du forecast publié."
        )
    values = pd.to_numeric(frame[STORM_DASHBOARD_COLUMN], errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float))
    if not bool(finite.any()):
        raise ValueError("Storm du jour ne contient aucune valeur finie.")
    if int(audit.get("expected_hours", -1)) != len(expected):
        raise ValueError("L’audit Storm du jour déclare une longueur incorrecte.")
    if int(audit.get("available_hours", -1)) != int(finite.sum()):
        raise ValueError("L’audit Storm du jour déclare une couverture incorrecte.")
    result.forecast_benchmark = pd.DataFrame(
        {
            "timestamp": result.forecast_native["timestamp"].to_numpy(),
            "q50": values.to_numpy(dtype=float),
        }
    )
    result.forecast_benchmark_label = "Storm officiel dashboard"
    result.forecast_benchmark_audit = audit


def build_hourly_zone_result(
    run_dir: str | Path,
    *,
    native_model: str | None = None,
    baseline_model: str | None = None,
    zone: str = "FR",
    timezone: str = "Europe/Paris",
    extreme_threshold: float = 150.0,
) -> ZoneRunResult:
    directory = Path(run_dir).expanduser().resolve()
    raw_backtest = _read_frame(directory / "backtest_hourly_oof.csv.gz")
    default_native, default_baseline = _default_models(raw_backtest)
    native_model = native_model or default_native
    if baseline_model is None:
        baseline_model = default_baseline
    mask = _evaluation_mask(
        raw_backtest,
        run_dir=directory,
        timezone=timezone,
    )
    mask &= _kalman_evaluation_mask(
        raw_backtest,
        run_dir=directory,
        native_model=native_model,
        timezone=timezone,
    )
    native = _backtest_prediction_frame(
        raw_backtest,
        model=native_model,
        row_mask=mask,
        timezone=timezone,
    )
    baseline = (
        _backtest_prediction_frame(
            raw_backtest,
            model=baseline_model,
            row_mask=mask,
            timezone=timezone,
        )
        if baseline_model
        else None
    )
    if baseline is not None and not native["timestamp"].equals(
        baseline["timestamp"]
    ):
        raise ValueError("Les timelines native et baseline diffèrent.")

    # A live archive keeps the original sealed OOF benchmark immutable, while
    # Statistics appends each subsequently realized day-ahead forecast.  Use
    # that audited extension for all backtest/probability panels whenever it
    # contains the requested models; otherwise reports would remain frozen at
    # the benchmark end date even though newer realized predictions exist.
    statistics_path = directory / "statistics_history_hourly.csv.gz"
    if statistics_path.is_file():
        extended = _read_frame(statistics_path)
        required_models = tuple(
            model for model in (native_model, baseline_model) if model
        )
        required_columns = {
            "delivery_start_utc",
            "actual",
            *(
                f"{model}__{quantile}"
                for model in required_models
                for quantile in ("q10", "q50", "q90")
            ),
        }
        if required_columns.issubset(extended.columns):
            delivery = pd.DatetimeIndex(
                pd.to_datetime(
                    extended["delivery_start_utc"], utc=True, errors="raise"
                )
            )
            if delivery.has_duplicates or not delivery.is_monotonic_increasing:
                raise ValueError(
                    f"{statistics_path}: timeline étendue invalide."
                )
            # Statistics may legitimately contain newer issued-live rows for
            # the native candidate while an older frozen baseline has no
            # prediction for those hours.  The report compares models, so use
            # the exact common finite support instead of letting each model
            # silently drop a different suffix and then failing on timelines.
            numeric_columns = sorted(
                column
                for column in required_columns
                if column != "delivery_start_utc"
            )
            numeric_required = extended.loc[:, numeric_columns].apply(
                pd.to_numeric,
                errors="coerce",
            )
            extended_mask = np.isfinite(
                numeric_required.to_numpy(dtype=float)
            ).all(axis=1)
            extended_mask &= _kalman_evaluation_mask(
                extended,
                run_dir=directory,
                native_model=native_model,
                timezone=timezone,
            )
            common_origins = np.ones(len(extended), dtype=bool)
            for model in required_models:
                model_origin = f"{model}_forecast_origin_utc"
                origin_column = (
                    model_origin
                    if model_origin in extended
                    else "forecast_origin_utc"
                )
                if origin_column in extended:
                    common_origins &= pd.to_datetime(
                        extended[origin_column],
                        utc=True,
                        errors="coerce",
                    ).notna().to_numpy()
            extended_mask &= common_origins
            if not bool(extended_mask.any()):
                extended_mask = np.zeros(len(extended), dtype=bool)
            else:
                extended_native = _backtest_prediction_frame(
                    extended,
                    model=native_model,
                    row_mask=extended_mask,
                    timezone=timezone,
                )
                extended_baseline = (
                    _backtest_prediction_frame(
                        extended,
                        model=baseline_model,
                        row_mask=extended_mask,
                        timezone=timezone,
                    )
                    if baseline_model
                    else None
                )
                if extended_baseline is not None and not extended_native[
                    "timestamp"
                ].equals(extended_baseline["timestamp"]):
                    raise ValueError(
                        "Les timelines Statistics native et baseline diffèrent."
                    )
                if len(extended_native) >= len(native):
                    native = extended_native
                    baseline = extended_baseline

    if native_model in KALMAN_REPORT_MODELS:
        kalman_audit = _read_json(directory / KALMAN_FILTER_AUDIT_ARTIFACT)
        native_delivery = pd.DatetimeIndex(
            pd.to_datetime(native["timestamp"], utc=True, errors="raise")
        ).tz_convert(timezone)
        native_days = pd.Index(native_delivery.date).unique()
        expected_hours = int(kalman_audit.get("evaluation_hours", -1))
        expected_start = pd.Timestamp(
            kalman_audit["evaluation_start_day"]
        ).date()
        expected_end = pd.Timestamp(
            kalman_audit["evaluation_end_day"]
        ).date()
        exact_support = (
            len(native_days) == 365
            and native_days[0] == expected_start
            and native_days[-1] == expected_end
            and expected_hours > 0
            and len(native) == expected_hours
        )
        if not exact_support:
            raise ValueError(
                f"Le support principal {native_model} ne correspond pas "
                "exactement aux 365 jours/heures déclarés dans l'audit."
            )

    forecast_path = directory / f"forecast_hourly_{zone.lower()}.csv"
    if not forecast_path.is_file() and zone.upper() == "FR":
        # Compatibility with already-sealed French runs.
        forecast_path = directory / "forecast_hourly_fr.csv"
    raw_forecast = _read_frame(forecast_path)
    forecast_native = _forecast_prediction_frame(
        raw_forecast,
        model=native_model,
        timezone=timezone,
    )
    forecast_baseline = (
        _forecast_prediction_frame(
            raw_forecast,
            model=baseline_model,
            timezone=timezone,
        )
        if baseline_model
        else None
    )
    metrics_native = compute_metrics(native, extreme_threshold)
    metrics_baseline = (
        compute_metrics(baseline, extreme_threshold)
        if baseline is not None
        else None
    )
    by_horizon, by_hour = metric_breakdowns(
        native,
        baseline,
        extreme_threshold,
    )
    result = ZoneRunResult(
        zone=zone,
        metrics_native=metrics_native,
        metrics_baseline=metrics_baseline,
        backtest_native=native,
        backtest_baseline=baseline,
        metrics_by_horizon=by_horizon,
        metrics_by_hour=by_hour,
        forecast_native=forecast_native,
        forecast_baseline=forecast_baseline,
        zone_data=_zone_data(directory, zone=zone, timezone=timezone),
        output_dir=directory / zone.lower(),
    )
    result.statistics_candidate_label = MODEL_LABELS.get(
        native_model, native_model
    )
    from chronos2_modular.forecast_explanation import attach_forecast_components
    attach_forecast_components(result, raw_forecast, model=native_model)
    statistics_path = directory / "statistics_history_hourly.csv.gz"
    statistics_raw = raw_backtest
    statistics_mask = mask
    history_audit: dict[str, Any] = {}
    if statistics_path.is_file():
        statistics_raw = _read_frame(statistics_path)
        if "delivery_start_utc" not in statistics_raw:
            raise ValueError(f"{statistics_path}: delivery_start_utc est absent.")
        statistics_delivery = pd.to_datetime(
            statistics_raw["delivery_start_utc"], utc=True, errors="raise"
        )
        if bool(statistics_delivery.duplicated().any()):
            raise ValueError(f"{statistics_path}: timestamps dupliques.")
        statistics_mask = np.ones(len(statistics_raw), dtype=bool)
        history_audit = _read_json(directory / "statistics_history_audit.json")
        zone_contracts = storm_benchmark_contracts(
            zone,
            timezone=timezone,
        )
        dashboard_column = str(
            zone_contracts[STORM_DASHBOARD_CONTRACT_ID][
                "point_column"
            ]
        )
        dashboard_declared = (
            history_audit.get("storm_primary_report_benchmark")
            == dashboard_column
        )
        if dashboard_column in statistics_raw and not dashboard_declared:
            raise ValueError(
                "La colonne Storm officiel dashboard existe sans contrat "
                "actif dans statistics_history_audit.json."
            )
        if dashboard_declared:
            if dashboard_column not in statistics_raw:
                raise ValueError(
                    "Le contrat Storm officiel dashboard est actif mais sa "
                    "colonne Statistics est absente."
                )
            dashboard_audit = history_audit.get("storm_dashboard")
            if not isinstance(dashboard_audit, Mapping):
                raise ValueError(
                    "Le contrat Storm officiel dashboard n'a pas d'audit."
                )
            dashboard_values = pd.to_numeric(
                statistics_raw[dashboard_column], errors="coerce"
            ).to_numpy(dtype=float)
            expected_hours = int(dashboard_audit.get("expected_hours", -1))
            available_hours = int(dashboard_audit.get("available_hours", -1))
            missing_hours = int(dashboard_audit.get("missing_hours", -1))
            dst_audit = dashboard_audit.get("dst")
            if isinstance(dst_audit, Mapping):
                no_interpolation = dst_audit.get("interpolation") is False
                no_strict_fallback = dst_audit.get("strict_08_fallback") is False
                allowed_missing = int(
                    dst_audit.get("native_allowed_missing_hours", -1)
                )
                exact_native_dst = (
                    dst_audit.get("native_actual_missing_matches_allowed") is True
                    and missing_hours == allowed_missing
                )
            else:
                # Backward-compatible exact artifacts predate the explicit
                # DST audit. Partial native coverage always requires it.
                no_interpolation = missing_hours == 0
                no_strict_fallback = missing_hours == 0
                exact_native_dst = missing_hours == 0
            audited_pairing = (
                expected_hours == len(statistics_raw)
                and available_hours == int(np.isfinite(dashboard_values).sum())
                and missing_hours == expected_hours - available_hours
                and exact_native_dst
                and no_interpolation
                and no_strict_fallback
            )
            if not audited_pairing:
                raise ValueError(
                    "Storm officiel dashboard doit couvrir exactement les "
                    "heures declarees par sa couverture appariee et son "
                    "audit DST."
                )
            # Pair candidate and benchmark only on finite native timestamps.
            # The strict-08 curve remains separate and is never a fill value.
            statistics_actual = pd.to_numeric(
                statistics_raw["actual"], errors="coerce"
            ).to_numpy(dtype=float)
            # Historical rows remain strictly paired. The sole exception is
            # the current delivery placeholder, whose actual is intentionally
            # empty and whose Storm value may also still be unavailable.
            statistics_mask &= (
                np.isfinite(dashboard_values)
                | ~np.isfinite(statistics_actual)
            )
        result.statistics_candidate = _backtest_prediction_frame(
            statistics_raw,
            model=native_model,
            row_mask=statistics_mask,
            timezone=timezone,
            allow_missing_actual=True,
        )
        if history_audit.get("report_scope_note"):
            result.statistics_scope_note = str(history_audit["report_scope_note"])

    # Historical treatment comparisons can declare the already-recalculated,
    # paired backtest as their Statistics source.  This deliberately does not
    # create or borrow a live ``statistics_history`` artifact: both candidate
    # and benchmark are read from the same sealed evaluation mask above.
    declared_comparison = result.zone_data.diagnostics.get(
        "statistics_comparison"
    )
    if isinstance(declared_comparison, Mapping):
        declared_source = str(declared_comparison.get("source", ""))
        declared_candidate = str(
            declared_comparison.get("candidate_model", "")
        )
        declared_benchmark = str(
            declared_comparison.get("benchmark_model", "")
        )
        if declared_source != "recalculated_paired_backtest":
            raise ValueError(
                "statistics_comparison.source doit valoir "
                "'recalculated_paired_backtest'."
            )
        if declared_candidate != native_model or declared_benchmark != baseline_model:
            raise ValueError(
                "statistics_comparison doit nommer les modeles natif et "
                "baseline exacts du rapport."
            )
        if baseline is None:
            raise ValueError(
                "statistics_comparison exige une baseline recalculee."
            )
        result.statistics_candidate = native.copy()
        result.statistics_benchmark = baseline.copy()
        result.statistics_candidate_label = str(
            declared_comparison.get("candidate_label")
            or MODEL_LABELS.get(native_model, native_model)
        )
        result.statistics_benchmark_label = str(
            declared_comparison.get("benchmark_label")
            or MODEL_LABELS.get(baseline_model, baseline_model)
        )
        raw_contract = declared_comparison.get("benchmark_contract")
        if not isinstance(raw_contract, Mapping):
            raise ValueError(
                "statistics_comparison.benchmark_contract doit etre un mapping."
            )
        result.statistics_benchmark_contract = dict(raw_contract)
        scope_note = str(declared_comparison.get("scope_note", "")).strip()
        if scope_note:
            result.statistics_scope_note = scope_note
    dashboard_model = "storm_dashboard_official"
    dashboard_active = (
        history_audit.get("storm_primary_report_benchmark")
        == f"{dashboard_model}__q50"
    )
    storm_model = (
        dashboard_model if dashboard_active else "storm_evaluation_only"
    )
    if f"{storm_model}__q50" in statistics_raw:
        storm = _backtest_point_prediction_frame(
            statistics_raw,
            model=storm_model,
            row_mask=statistics_mask,
            timezone=timezone,
            allow_missing_actual=True,
        )
        statistics_candidate = getattr(result, "statistics_candidate", native)
        if not statistics_candidate["timestamp"].equals(storm["timestamp"]):
            raise ValueError(
                "Les timelines du candidat et de Storm diffèrent."
            )
        if not np.allclose(
            statistics_candidate["actual"].to_numpy(dtype=float),
            storm["actual"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-9,
            equal_nan=True,
        ):
            raise ValueError(
                "Les observations du candidat et de Storm diffèrent."
            )
        # ZoneRunResult predates evaluation-only comparators and is not
        # slotted.  Attaching these reporting-only attributes preserves the
        # public dataclass/API used by the modelling pipeline.
        result.statistics_benchmark = storm
        zone_contracts = storm_benchmark_contracts(
            zone,
            timezone=timezone,
        )
        active_contract = dict(
            zone_contracts[
                STORM_DASHBOARD_CONTRACT_ID
                if dashboard_active
                else STORM_AVAILABLE_0800_CONTRACT_ID
            ]
        )
        if dashboard_active:
            active_contract["status"] = "loaded_from_explicit_audited_artifact"
            dashboard_audit = history_audit.get("storm_dashboard")
            if isinstance(dashboard_audit, Mapping):
                active_contract["materialization_audit"] = dict(
                    dashboard_audit
                )
        result.statistics_benchmark_label = str(
            active_contract["report_label"]
        )
        result.statistics_benchmark_contract = active_contract
        result.statistics_benchmark_contract_catalog = {
            contract_id: dict(contract)
            for contract_id, contract in zone_contracts.items()
        }
    if statistics_path.is_file():
        candidate_statistics = getattr(result, "statistics_candidate", None)
        if candidate_statistics is not None and not candidate_statistics.empty:
            paired_delivery = pd.DatetimeIndex(
                pd.to_datetime(
                    candidate_statistics["timestamp"], utc=True, errors="raise"
                )
            )
            prefix_value = (
                history_audit.get("statistics_through_day_local")
                or history_audit.get("statistics_prefix_end_local")
            )
            prefix_day = (
                pd.Timestamp(prefix_value).date()
                if prefix_value not in (None, "")
                else paired_delivery[-1].tz_convert(timezone).date()
            )
            expected_day = local_delivery_day_index(prefix_day, timezone=timezone)
            dashboard_audit = history_audit.get("storm_dashboard")
            allowed_missing: set[pd.Timestamp] = set()
            if isinstance(dashboard_audit, Mapping):
                dst_audit = dashboard_audit.get("dst")
                if isinstance(dst_audit, Mapping):
                    allowed_missing = {
                        pd.Timestamp(value).tz_convert("UTC")
                        for value in dst_audit.get(
                            "native_allowed_missing_utc", []
                        )
                    }
            expected_common_hours = int(
                sum(value not in allowed_missing for value in expected_day)
            )
            common_hours = int(
                sum(
                    value.tz_convert(timezone).date() == prefix_day
                    for value in paired_delivery
                )
            )
            actual_audit = history_audit.get("canonical_actuals")
            actual_source = (
                actual_audit.get("source")
                if isinstance(actual_audit, Mapping)
                and isinstance(actual_audit.get("source"), Mapping)
                else {}
            )
            storm_source = (
                dashboard_audit.get("source")
                if isinstance(dashboard_audit, Mapping)
                and isinstance(dashboard_audit.get("source"), Mapping)
                else {}
            )
            if actual_source.get("extracted_at_utc") is not None:
                latest_complete_value = (
                    actual_audit.get("latest_complete_day_local")
                    if isinstance(actual_audit, Mapping)
                    else None
                )
                complete_day = (
                    pd.Timestamp(latest_complete_value).date()
                    if latest_complete_value not in (None, "")
                    else prefix_day
                )
                expected_complete_day = local_delivery_day_index(
                    complete_day, timezone=timezone
                )
                expected_common_hours = int(
                    sum(
                        value not in allowed_missing
                        for value in expected_complete_day
                    )
                )
                candidate_actual = pd.to_numeric(
                    candidate_statistics["actual"], errors="coerce"
                ).to_numpy(dtype=float)
                candidate_point = pd.to_numeric(
                    candidate_statistics["q50"], errors="coerce"
                ).to_numpy(dtype=float)
                common_mask = np.isfinite(candidate_actual) & np.isfinite(
                    candidate_point
                )
                benchmark_statistics = getattr(
                    result, "statistics_benchmark", None
                )
                if benchmark_statistics is not None:
                    common_mask &= np.isfinite(
                        pd.to_numeric(
                            benchmark_statistics["q50"], errors="coerce"
                        ).to_numpy(dtype=float)
                    )
                common_delivery = paired_delivery[common_mask]
                common_hours = int(
                    sum(
                        value.tz_convert(timezone).date() == complete_day
                        for value in common_delivery
                    )
                )
                result.statistics_freshness = {
                    "timezone": timezone,
                    "target_series": (
                        actual_audit.get("series")
                        if isinstance(actual_audit, Mapping)
                        else None
                    ),
                    "actual_extracted_at_utc": actual_source.get(
                        "extracted_at_utc"
                    ),
                    "actual_available_end_utc": actual_source.get(
                        "last_available_observation_utc"
                    ),
                    "actual_applied_end_utc": (
                        actual_audit.get("last_applied_observation_utc")
                        if isinstance(actual_audit, Mapping)
                        else None
                    ),
                    "storm_available": bool(dashboard_active),
                    "storm_extracted_at_utc": storm_source.get(
                        "extracted_at_utc"
                    ),
                    "storm_applied_end_utc": (
                        str(paired_delivery[-1]) if dashboard_active else None
                    ),
                    "common_delivery_end_utc": (
                        str(common_delivery[-1])
                        if len(common_delivery)
                        else None
                    ),
                    "last_complete_common_day_local": complete_day.isoformat(),
                    "common_hours_last_day": common_hours,
                    "expected_common_hours_last_day": expected_common_hours,
                }
    _attach_live_storm_forecast(result, directory=directory)
    _attach_kalman_diagnostics(
        result,
        directory=directory,
        native_model=native_model,
        baseline_model=baseline_model,
        timezone=timezone,
    )
    _attach_variable_attribution(
        result,
        directory=directory,
        native_model=native_model,
        timezone=timezone,
    )
    result.output_dir.mkdir(parents=True, exist_ok=True)
    return result


def _replace_report_labels(
    output_path: Path,
    *,
    native_label: str,
    baseline_label: str | None,
) -> None:
    source = output_path.read_text(encoding="utf-8")
    native_lower = native_label.lower()
    replacements: list[tuple[str, str]] = [
        ("Chronos-2 + covariables P50", f"{native_label} P50"),
        ("Covariables natives", native_label),
        ("Covariables P50", f"{native_label} P50"),
        ("MAE covariables", f"MAE {native_lower}"),
        ("Couverture covariables", f"Couverture {native_lower}"),
        ("Biais covariables", f"Biais {native_lower}"),
        ("Erreur Chronos-2", f"Erreur {native_lower}"),
    ]
    if baseline_label:
        baseline_lower = baseline_label.lower()
        replacements.extend(
            [
                ("Chronos-2 prix seul P50", f"{baseline_label} P50"),
                ("Comparaison au modèle prix seul", f"Comparaison à {baseline_label}"),
                ("Prix seul P50", f"{baseline_label} P50"),
                ("MAE prix seul", f"MAE {baseline_lower}"),
                ("Couverture prix seul", f"Couverture {baseline_lower}"),
                ("Biais prix seul", f"Biais {baseline_lower}"),
                ("Erreur prix seul", f"Erreur {baseline_lower}"),
                ("Prix seul", baseline_label),
            ]
        )
    for old, new in replacements:
        source = source.replace(old, new)
    output_path.write_text(source, encoding="utf-8")


def write_hourly_html_report(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    title: str | None = None,
    native_model: str | None = None,
    baseline_model: str | None = None,
    zone: str = "FR",
    timezone: str = "Europe/Paris",
    extreme_threshold: float = 150.0,
    history_hours: int = 168,
) -> Path:
    directory = Path(run_dir).expanduser().resolve()
    result = build_hourly_zone_result(
        directory,
        native_model=native_model,
        baseline_model=baseline_model,
        zone=zone,
        timezone=timezone,
        extreme_threshold=extreme_threshold,
    )
    inferred_native, inferred_baseline = _default_models(
        _read_frame(directory / "backtest_hourly_oof.csv.gz")
    )
    native_model = native_model or inferred_native
    if baseline_model is None:
        baseline_model = inferred_baseline
    native_label = MODEL_LABELS.get(native_model, native_model)
    baseline_label = (
        MODEL_LABELS.get(baseline_model, baseline_model)
        if baseline_model
        else None
    )
    path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else directory / f"chronos2_hourly_{zone.lower()}_{native_model}.html"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    report_config = {
        "report": {
            "title": title or f"Chronos-2 horaire {zone} — {native_label}",
            "forecast_history_hours": int(history_hours),
        }
    }
    write_html_report([result], report_config, path)
    _replace_report_labels(
        path,
        native_label=native_label,
        baseline_label=baseline_label,
    )
    return path


__all__: Sequence[str] = (
    "KALMAN_DAILY_AUDIT_ARTIFACT",
    "KALMAN_FILTER_AUDIT_ARTIFACT",
    "KALMAN_STATE_AUDIT_ARTIFACT",
    "MODEL_LABELS",
    "REPORT_QUANTILES",
    "VARIABLE_ATTRIBUTION_AUDIT_ARTIFACT",
    "VARIABLE_ATTRIBUTION_COLUMNS",
    "VARIABLE_ATTRIBUTION_HOURLY_ARTIFACT",
    "VARIABLE_ATTRIBUTION_METHODS",
    "VARIABLE_ATTRIBUTION_MODEL_VARIANTS",
    "VARIABLE_ATTRIBUTION_TOLERANCE",
    "STORM_AVAILABLE_0800_CONTRACT_ID",
    "STORM_BENCHMARK_CONTRACTS",
    "STORM_DASHBOARD_NATIVE_SERIES_BY_ZONE",
    "STORM_DASHBOARD_CONTRACT_ID",
    "build_hourly_zone_result",
    "_attach_kalman_diagnostics",
    "_attach_variable_attribution",
    "storm_benchmark_contracts",
    "write_hourly_html_report",
)
