"""Adapter from hourly run artifacts to the existing standalone HTML report.

The project already ships a rich Plotly report in ``chronos2_modular.report``.
This module deliberately reuses that renderer so hourly and legacy runs keep
the same layout, cards, figures, statistics filters and embedded-JavaScript
behaviour.  Only the artifact-to-``ZoneRunResult`` conversion lives here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from chronos2_modular.common import ZoneData, ZoneRunResult
from chronos2_modular.metrics import compute_metrics, metric_breakdowns
from chronos2_modular.report import write_html_report


REPORT_QUANTILES: tuple[float, ...] = tuple(
    round(value / 10.0, 1) for value in range(1, 10)
)

MODEL_LABELS: dict[str, str] = {
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
    "ensemble": "Ensemble horaire",
    "chronos2": "Chronos-2",
    "catboost": "CatBoost",
    "lear": "LEAR",
}


STORM_AVAILABLE_0800_CONTRACT_ID = "storm_available_at_0800"
STORM_DASHBOARD_CONTRACT_ID = "storm_official_dashboard"

STORM_DASHBOARD_NATIVE_SERIES_BY_ZONE: dict[str, str] = {
    zone: f"power.price.{zone.lower()}.euromwh.h.fcst.3mv.storm"
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
            f"power.price.{zone_token}.euromwh.h.fcst.3mv.storm"
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
            "series_kind": "native_exact",
            "series_identifier_verified": dashboard_series_verified,
            "zone": zone_key,
            "timezone": local_timezone,
            "artifact_filename": (
                "inputs/storm_dashboard_official_statistics.parquet"
            ),
            "point_column": "storm_dashboard_official__q50",
            "official_dashboard_metric": True,
            "availability_rule": (
                "extraction courante de la série native exacte, figée avec "
                "horodatage d'extraction et checksum"
            ),
            "status": "not_loaded_without_explicit_native_artifact",
            "report_note": (
                "contrat officiel dashboard : extraction courante de la "
                "série native exacte, normalisée en heure civile locale, "
                "figée dans un artefact séparé et audité."
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


def _backtest_prediction_frame(
    raw: pd.DataFrame,
    *,
    model: str,
    row_mask: np.ndarray,
    timezone: str,
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
        "actual",
        "point",
        *[f"q{int(100 * value):02d}" for value in REPORT_QUANTILES],
    ]
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
    finite = np.isfinite(
        result[["actual", "q50"]].to_numpy(dtype=float)
    ).all(axis=1)
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
    if "residual_corrected__q50" in backtest:
        return "residual_corrected", "ensemble"
    if "ensemble__q50" in backtest:
        return "ensemble", "chronos2"
    if "chronos2__q50" in backtest:
        return "chronos2", None
    raise ValueError("Aucun modèle horaire reconnu dans le backtest.")


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
            statistics_mask &= np.isfinite(dashboard_values)
        result.statistics_candidate = _backtest_prediction_frame(
            statistics_raw,
            model=native_model,
            row_mask=statistics_mask,
            timezone=timezone,
        )
        if history_audit.get("report_scope_note"):
            result.statistics_scope_note = str(history_audit["report_scope_note"])
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
    "MODEL_LABELS",
    "REPORT_QUANTILES",
    "STORM_AVAILABLE_0800_CONTRACT_ID",
    "STORM_BENCHMARK_CONTRACTS",
    "STORM_DASHBOARD_NATIVE_SERIES_BY_ZONE",
    "STORM_DASHBOARD_CONTRACT_ID",
    "build_hourly_zone_result",
    "storm_benchmark_contracts",
    "write_hourly_html_report",
)
