#!/usr/bin/env python
"""Run the isolated multi-zone price-regime shadow challenger.

By default this command only consumes already-issued official control archives,
then trains/evaluates the challenger and publishes under ``runs/challengers``.
``--run-forecasts-first`` explicitly opts into invoking the unchanged official
multi-country launcher before the isolated challenger.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from html import escape
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd
import sklearn

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.app_service import inspect_zone_statuses
from chronos2_hourly.multizone_live import _publish_staging_atomically
from chronos2_hourly.regime_challenger import (
    QUANTILES,
    RESIDUAL_ALIASES,
    RegimeChallengerConfig,
    RegimeChallengerError,
    build_regime_features,
    fit_regime_model,
    load_regime_challenger_config,
    predict_regime_adjustment,
    prequential_regime_predictions,
    scheduled_origin_utc,
    select_forecasts_asof,
    summarize_metrics,
)
from chronos2_hourly.regime_challenger_report import (
    write_regime_challenger_report,
)
from run_multicountry_forecast import (
    DEFAULT_LOG_DIR,
    DEFAULT_REGISTRY,
    normalize_zones,
    normalise_delivery_day,
    normalise_forecast_mode,
    print_batch_summary,
    run_forecast_batch,
)
from chronos2_modular.common import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "price_regime_challenger.yaml"
MODEL_BY_VARIANT = {
    "autonomous": "residual_corrected",
    "blend": "mkonline_blend",
}


class RegimeChallengerRunnerError(RuntimeError):
    """Raised when a shadow run cannot be produced honestly."""


def _normalise_challenger_mode(mode: str) -> str:
    selected = normalise_forecast_mode(mode)
    if selected not in {"autonomous", "blend", "both"}:
        raise ValueError(
            "Le challenger exige Mode Autonomous, Blend ou Both; "
            f"{selected.title()} ne fait pas partie de son contrat de comparaison."
        )
    return selected


def _requested_control_variants(mode: str, zone: str) -> tuple[str, ...]:
    """Keep this experiment's original autonomous/MKOnline control contract."""
    selected = _normalise_challenger_mode(mode)
    canonical = normalize_zones([zone])[0]
    if selected == "autonomous":
        return ("autonomous",)
    if selected == "blend":
        if canonical not in {"FR", "NL"}:
            raise ValueError(f"Le controle Blend du challenger n'est disponible que pour FR et NL; zone={canonical}.")
        return ("blend",)
    return ("autonomous", "blend") if canonical in {"FR", "NL"} else ("autonomous",)


@dataclass(frozen=True)
class ControlArchive:
    zone: str
    timezone: str
    delivery_day: str
    path: Path
    manifest: Mapping[str, Any]
    forecast: pd.DataFrame
    history: pd.DataFrame
    current_actual: pd.Series
    hashes: Mapping[str, str]


@dataclass(frozen=True)
class ChallengerRunResult:
    delivery_day: str
    mode: str
    output_dir: Path
    report_paths: tuple[Path, ...]
    reused_controls: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp, date)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _canonical_utc_frame(frame: pd.DataFrame, *, path: Path) -> pd.DataFrame:
    if "delivery_start_utc" not in frame:
        raise RegimeChallengerRunnerError(
            f"{path}: colonne delivery_start_utc absente."
        )
    result = frame.copy()
    result["delivery_start_utc"] = pd.to_datetime(
        result["delivery_start_utc"], utc=True, errors="coerce"
    )
    if result["delivery_start_utc"].isna().any():
        raise RegimeChallengerRunnerError(f"{path}: timestamps UTC invalides.")
    result = result.sort_values("delivery_start_utc", kind="stable")
    result = result.drop_duplicates("delivery_start_utc", keep="last")
    return result.set_index("delivery_start_utc")


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegimeChallengerRunnerError(f"JSON illisible: {path}") from exc
    if not isinstance(payload, Mapping):
        raise RegimeChallengerRunnerError(f"{path}: objet JSON attendu.")
    return payload


def _source_columns(variant: str) -> dict[str, str]:
    try:
        model = MODEL_BY_VARIANT[variant]
    except KeyError as exc:
        raise RegimeChallengerRunnerError(
            f"Variante challenger inconnue: {variant}."
        ) from exc
    return {quantile: f"{model}__{quantile}" for quantile in QUANTILES}


def _load_target_actual(
    manifest: Mapping[str, Any],
    *,
    project_root: Path,
) -> pd.Series:
    raw_path = manifest.get("target_source_path")
    if not raw_path:
        return pd.Series(dtype=float, name="actual")
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if not path.is_file():
        return pd.Series(dtype=float, name="actual")
    frame = pd.read_csv(path)
    timestamp_column = next(
        (column for column in ("timestamp", "delivery_start_utc") if column in frame),
        None,
    )
    if timestamp_column is None or "value" not in frame:
        return pd.Series(dtype=float, name="actual")
    index = pd.DatetimeIndex(
        pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
    )
    values = pd.to_numeric(frame["value"], errors="coerce")
    valid = index.notna() & values.notna().to_numpy()
    result = pd.Series(values.to_numpy(float)[valid], index=index[valid], name="actual")
    return result.loc[~result.index.duplicated(keep="last")].sort_index()


def _history_path(archive: Path) -> Path:
    statistics = archive / "statistics_history_hourly.csv.gz"
    return statistics if statistics.is_file() else archive / "backtest_hourly_oof.csv.gz"


def _verify_control_checksums(
    archive: Path,
    *,
    required_paths: Sequence[Path],
    project_root: Path,
) -> Mapping[str, str]:
    """Verify required control artifacts against the archive checksum manifest."""

    checksum_path = archive / "artifact_checksums.json"
    if not checksum_path.is_file():
        raise FileNotFoundError(checksum_path)
    payload = _load_json(checksum_path)
    if str(payload.get("algorithm", "")).lower() != "sha256":
        raise RegimeChallengerRunnerError(
            f"{checksum_path}: algorithme sha256 exige."
        )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise RegimeChallengerRunnerError(
            f"{checksum_path}: liste artifacts absente."
        )

    entries: list[tuple[set[Path], Mapping[str, Any]]] = []
    for item in artifacts:
        if not isinstance(item, Mapping) or not item.get("path"):
            raise RegimeChallengerRunnerError(
                f"{checksum_path}: entree artifact invalide."
            )
        raw = Path(str(item["path"])).expanduser()
        candidates = (
            {raw.resolve()}
            if raw.is_absolute()
            else {(archive / raw).resolve(), (project_root / raw).resolve()}
        )
        entries.append((candidates, item))

    for required in required_paths:
        resolved = required.resolve()
        matches = [item for candidates, item in entries if resolved in candidates]
        if len(matches) != 1:
            raise RegimeChallengerRunnerError(
                f"{checksum_path}: checksum unique absent pour {required.name}."
            )
        item = matches[0]
        declared_hash = str(item.get("sha256", "")).strip().lower()
        try:
            declared_size = int(item.get("size_bytes"))
        except (TypeError, ValueError) as exc:
            raise RegimeChallengerRunnerError(
                f"{checksum_path}: taille declaree invalide pour {required.name}."
            ) from exc
        actual_size = required.stat().st_size
        actual_hash = _sha256(required)
        if declared_size != actual_size or declared_hash != actual_hash:
            raise RegimeChallengerRunnerError(
                f"{required}: checksum de l'archive controle invalide."
            )
    return {
        "checksums_manifest_sha256": _sha256(checksum_path),
        "checksums_algorithm": "sha256",
    }


def _validate_exact_origin(
    values: pd.Series,
    *,
    expected: pd.Timestamp,
    context: str,
) -> None:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any() or not bool(parsed.eq(expected).all()):
        observed = sorted({str(value) for value in parsed.dropna().unique()})
        raise RegimeChallengerRunnerError(
            f"{context}: origine live attendue {expected}, observee {observed}."
        )


def load_control_archive(
    archive: str | Path,
    *,
    zone: str,
    delivery_day: str,
    mode: str,
    project_root: str | Path = PROJECT_ROOT,
) -> ControlArchive:
    """Validate one issued live archive without modifying it."""

    path = Path(archive).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()
    manifest_path = path / "run_manifest.json"
    forecast_path = path / f"forecast_hourly_{zone.lower()}.csv"
    history_path = _history_path(path)
    for required in (manifest_path, forecast_path, history_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    checksum_audit = _verify_control_checksums(
        path,
        required_paths=(manifest_path, forecast_path, history_path),
        project_root=root,
    )
    manifest = _load_json(manifest_path)
    manifest_zone = str(manifest.get("zone", "")).upper()
    manifest_day = str(manifest.get("delivery_day_local", ""))
    if manifest_zone != zone or manifest_day != delivery_day:
        raise RegimeChallengerRunnerError(
            f"Archive controle incoherente: {manifest_zone}/{manifest_day}, "
            f"attendu {zone}/{delivery_day}."
        )
    if str(manifest.get("run_type", "")) != "live_day_ahead":
        raise RegimeChallengerRunnerError(f"{zone}: run_type live_day_ahead exige.")
    if str(manifest.get("forecast_status", "")) != "issued_live":
        raise RegimeChallengerRunnerError(f"{zone}: forecast_status issued_live exige.")
    timezone = str(manifest.get("timezone", "")).strip()
    if not timezone:
        raise RegimeChallengerRunnerError(f"{zone}: timezone absente du manifeste.")
    expected_origin = scheduled_origin_utc(delivery_day, timezone=timezone)
    cutoff = pd.to_datetime(
        manifest.get("forecast_cutoff_utc"), utc=True, errors="coerce"
    )
    if pd.isna(cutoff) or cutoff != expected_origin:
        raise RegimeChallengerRunnerError(
            f"{zone}: forecast_cutoff_utc doit etre exactement {expected_origin}."
        )

    forecast = _canonical_utc_frame(pd.read_csv(forecast_path), path=forecast_path)
    expected = local_delivery_day_index(delivery_day, timezone=timezone)
    if not forecast.index.equals(expected):
        raise RegimeChallengerRunnerError(
            f"{zone}: la grille live n'est pas exactement le jour local ({len(expected)}h)."
        )
    required_variants = set(_requested_control_variants(mode, zone)) | {"autonomous"}
    origin_columns = {
        "autonomous": "forecast_origin_utc",
        "blend": "mkonline_blend_forecast_origin_utc",
    }
    for variant in sorted(required_variants):
        missing = sorted(set(_source_columns(variant).values()).difference(forecast.columns))
        if missing:
            raise RegimeChallengerRunnerError(
                f"{zone}/{variant}: quantiles controles absents: {missing}."
            )
        origin_column = origin_columns[variant]
        if origin_column not in forecast:
            raise RegimeChallengerRunnerError(
                f"{zone}/{variant}: origine live absente ({origin_column})."
            )
        _validate_exact_origin(
            forecast[origin_column],
            expected=expected_origin,
            context=f"{zone}/{variant}",
        )
    history = _canonical_utc_frame(pd.read_csv(history_path), path=history_path)
    for variant in sorted(required_variants):
        missing = sorted(set(_source_columns(variant).values()).difference(history.columns))
        if missing:
            raise RegimeChallengerRunnerError(
                f"{zone}/{variant}: historique controle incomplet: {missing}."
            )
    current_actual = _load_target_actual(manifest, project_root=root).reindex(expected)
    hashes: dict[str, str] = {
        **checksum_audit,
        "manifest_sha256": _sha256(manifest_path),
        "forecast_sha256": _sha256(forecast_path),
        "history_sha256": _sha256(history_path),
    }
    raw_target_path = manifest.get("target_source_path")
    if raw_target_path:
        target_path = Path(str(raw_target_path)).expanduser()
        if not target_path.is_absolute():
            target_path = root / target_path
        target_path = target_path.resolve()
        if target_path.is_file():
            hashes["current_actual_source_sha256"] = _sha256(target_path)
    return ControlArchive(
        zone=zone,
        timezone=timezone,
        delivery_day=delivery_day,
        path=path,
        manifest=manifest,
        forecast=forecast,
        history=history,
        current_actual=current_actual,
        hashes=hashes,
    )


def _variant_history(
    control: ControlArchive,
    *,
    variant: str,
    label_end_day: date,
) -> pd.DataFrame:
    columns = _source_columns(variant)
    missing = sorted(set(columns.values()).difference(control.history.columns))
    if missing:
        raise RegimeChallengerRunnerError(
            f"{control.zone}/{variant}: historique incomplet: {missing}."
        )
    if "actual" not in control.history:
        raise RegimeChallengerRunnerError(
            f"{control.zone}: actual absent de l'historique controle."
        )
    frame = pd.DataFrame(index=control.history.index)
    for quantile, source in columns.items():
        frame[quantile] = pd.to_numeric(control.history[source], errors="coerce")
    frame["actual"] = pd.to_numeric(control.history["actual"], errors="coerce")
    local_dates = pd.Series(
        frame.index.tz_convert(control.timezone).date, index=frame.index
    )
    origin_column = (
        "mkonline_blend_forecast_origin_utc"
        if variant == "blend"
        else "forecast_origin_utc"
    )
    if origin_column not in control.history:
        raise RegimeChallengerRunnerError(
            f"{control.zone}/{variant}: origine OOF absente ({origin_column})."
        )
    origins = pd.to_datetime(
        control.history[origin_column], utc=True, errors="coerce"
    )
    scheduled = pd.Series(
        [scheduled_origin_utc(day, timezone=control.timezone) for day in local_dates],
        index=frame.index,
        dtype="datetime64[ns, UTC]",
    )
    finite = frame[[*QUANTILES, "actual"]].notna().all(axis=1)
    eligible = origins.notna() & origins.le(scheduled) & local_dates.le(label_end_day)
    frame = frame.loc[finite & eligible]
    if frame.empty:
        raise RegimeChallengerRunnerError(
            f"{control.zone}/{variant}: aucun historique OOF eligible avant {label_end_day}."
        )
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise AssertionError("Historique filtre non canonique.")
    return frame


def _variant_live(control: ControlArchive, *, variant: str) -> pd.DataFrame:
    columns = _source_columns(variant)
    frame = pd.DataFrame(index=control.forecast.index)
    for quantile, source in columns.items():
        frame[quantile] = pd.to_numeric(control.forecast[source], errors="coerce")
    if frame.isna().any().any() or bool(np.isinf(frame.to_numpy(float)).any()):
        raise RegimeChallengerRunnerError(
            f"{control.zone}/{variant}: quantiles live non finis."
        )
    if bool((frame.diff(axis=1).iloc[:, 1:] < -1e-9).any().any()):
        raise RegimeChallengerRunnerError(
            f"{control.zone}/{variant}: quantiles live croises."
        )
    return frame


def _autonomous_price_context(
    controls: Mapping[str, ControlArchive],
    *,
    label_end_day: date,
    grid: pd.DatetimeIndex,
) -> dict[str, pd.Series]:
    result: dict[str, pd.Series] = {}
    for zone, control in controls.items():
        history = _variant_history(
            control, variant="autonomous", label_end_day=label_end_day
        )["q50"]
        live = _variant_live(control, variant="autonomous")["q50"]
        combined = pd.concat([history, live])
        combined = combined.loc[~combined.index.duplicated(keep="last")]
        result[zone] = combined.reindex(grid)
    return result


def _load_residual_features(
    config: RegimeChallengerConfig,
    *,
    grid: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    values: dict[str, pd.Series] = {}
    audits: dict[str, Any] = {}
    for alias in RESIDUAL_ALIASES:
        path = config.pit_files[alias]
        if not path.is_file():
            raise FileNotFoundError(path)
        selected, audit = select_forecasts_asof(
            pd.read_parquet(path),
            alias=alias,
            grid=grid,
            timezone=config.timezone,
        )
        values[alias] = selected
        audits[alias] = {
            **audit,
            "path": path,
            "sha256": _sha256(path),
        }
        if float(audit["last_day_coverage"]) != 1.0:
            raise RegimeChallengerRunnerError(
                f"{alias}: couverture J+1 {float(audit['last_day_coverage']):.2%}; "
                "aucune correction challenger n'est autorisee."
            )
        maximum_age = float(audit["last_day_max_revision_age_hours"])
        if not np.isfinite(maximum_age) or maximum_age > 24.0:
            raise RegimeChallengerRunnerError(
                f"{alias}: vintage J+1 trop agee ({maximum_age:.2f} h > 24 h)."
            )
    frame = pd.DataFrame(values, index=grid)
    complete_coverage = float(frame.notna().all(axis=1).mean())
    if complete_coverage < 0.80:
        raise RegimeChallengerRunnerError(
            f"Couverture commune residual load {complete_coverage:.2%} < 80%."
        )
    audits["common"] = {
        "coverage": complete_coverage,
        "complete_rows": int(frame.notna().all(axis=1).sum()),
        "total_rows": int(len(frame)),
        "cutoff_violations": 0,
    }
    return frame, audits


def _existing_control_paths(
    *,
    registry_path: str | Path,
    zones: Sequence[str],
    delivery_day: str,
) -> dict[str, Path]:
    """Resolve each zone's configured live root for archive-only replay."""

    registry = Path(registry_path).expanduser().resolve()
    statuses = inspect_zone_statuses(registry, zones=zones)
    result: dict[str, Path] = {}
    for status in statuses:
        if status.live_config is None:
            raise RegimeChallengerRunnerError(
                f"{status.code}: configuration live absente."
            )
        config_path = Path(status.live_config).expanduser().resolve()
        payload = load_yaml(config_path)
        live = payload.get("live")
        if not isinstance(live, Mapping) or not live.get("output_root"):
            raise RegimeChallengerRunnerError(
                f"{status.code}: live.output_root absent de {config_path}."
            )
        output_root = Path(str(live["output_root"])).expanduser()
        if not output_root.is_absolute():
            output_root = config_path.parent / output_root
        result[status.code] = (
            output_root.resolve()
            / f"{status.code.lower()}_day_ahead_{delivery_day}"
        )
    if set(result) != set(zones):
        raise RegimeChallengerRunnerError(
            "Le registre n'a pas resolu toutes les archives controles."
        )
    return result


def _evaluation_hourly(
    prequential: pd.DataFrame,
    *,
    zone: str,
    variant: str,
    timezone: str,
    threshold: float,
    solar_hours: Sequence[int],
) -> pd.DataFrame:
    frame = prequential.copy()
    frame["zone"] = zone
    frame["variant"] = variant
    local = frame.index.tz_convert(timezone)
    frame["local_date"] = local.date.astype(str)
    frame["local_hour"] = local.hour.astype(int)
    frame["regime_label"] = (
        frame["actual"].sub(frame["baseline_q50"]).ge(threshold)
        & frame["local_hour"].isin(solar_hours)
    ).astype(int)
    frame["delivery_start_utc"] = frame.index
    return frame.reset_index(drop=True)


def _live_hourly(
    predicted: pd.DataFrame,
    *,
    control: ControlArchive,
    variant: str,
    threshold: float,
    solar_hours: Sequence[int],
) -> pd.DataFrame:
    frame = predicted.copy()
    frame["actual"] = control.current_actual.reindex(frame.index)
    local = frame.index.tz_convert(control.timezone)
    frame["local_date"] = local.date.astype(str)
    frame["local_hour"] = local.hour.astype(int)
    observed_label = (
        frame["actual"].sub(frame["baseline_q50"]).ge(threshold)
        & frame["local_hour"].isin(solar_hours)
    )
    frame["regime_label"] = np.where(
        frame["actual"].notna(),
        np.where(observed_label, "shock", "normal"),
        np.where(frame["regime_predicted"].eq(1), "shock_predit", "normal_predit"),
    )
    frame["variant"] = variant
    frame["delivery_start_utc"] = frame.index
    return frame.reset_index(drop=True)


def _index_html(
    *,
    delivery_day: str,
    mode: str,
    reports: Sequence[tuple[str, Path]],
    metrics: pd.DataFrame,
    root: Path,
) -> str:
    links = "".join(
        '<li><a href="'
        + escape(path.relative_to(root).as_posix(), quote=True)
        + '">Rapport détaillé '
        + escape(zone)
        + "</a></li>"
        for zone, path in reports
    )
    preferred = [
        column
        for column in (
            "zone",
            "variant",
            "scope",
            "n_hours",
            "baseline_mae",
            "challenger_mae",
            "mae_gain",
            "regime_recall",
            "regime_roc_auc",
        )
        if column in metrics
    ]
    head = "".join(f"<th>{escape(column)}</th>" for column in preferred)
    rows = []
    for values in metrics.loc[:, preferred].itertuples(index=False, name=None):
        cells = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                rendered = "N/A" if not np.isfinite(value) else f"{float(value):.3f}"
            else:
                rendered = str(value)
            cells.append(f"<td>{escape(rendered)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Challenger régime {escape(delivery_day)}</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f7fb;color:#142033}}main{{max-width:1300px;margin:auto;padding:32px}}.banner,.card{{background:white;border:1px solid #d8e1ec;border-radius:14px;padding:20px;margin:16px 0;box-shadow:0 8px 24px #0f172a0d}}.banner{{border-left:6px solid #d97706}}a{{color:#1d4ed8;font-weight:700}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:8px;border-bottom:1px solid #e2e8f0;text-align:right}}th:first-child,td:first-child{{text-align:left}}.scroll{{overflow:auto}}code{{background:#eef2ff;padding:2px 5px;border-radius:4px}}</style></head>
<body><main><h1>Challenger de changement de régime</h1>
<p>Livraison <strong>{escape(delivery_day)}</strong> · mode <strong>{escape(mode)}</strong></p>
<div class="banner"><strong>SHADOW — NON PRODUCTION.</strong> Les prévisions officielles et leurs exports n'ont pas été remplacés.</div>
<section class="card"><h2>Rapports par zone</h2><ul>{links}</ul></section>
<section class="card"><h2>Synthèse préquentielle</h2><div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
</main></body></html>"""


def _atomic_publish(staging: Path, final: Path, *, overwrite: bool) -> None:
    """Publish the shadow tree with the live runner's Windows lock retries."""

    if final.exists() and not overwrite:
        raise FileExistsError(
            f"Sortie challenger existante: {final}; utilisez --overwrite."
        )
    final.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    try:
        if final.exists():
            backup = final.with_name(f".{final.name}.{uuid.uuid4().hex}.backup")
            _publish_staging_atomically(final, backup)
        _publish_staging_atomically(staging, final)
    except Exception:
        if backup is not None and backup.exists() and not final.exists():
            _publish_staging_atomically(backup, final)
        raise
    else:
        if backup is not None:
            shutil.rmtree(backup)


def run_challenger(
    *,
    zones: Sequence[str],
    mode: str,
    delivery_day: str | date | None,
    config_path: str | Path = DEFAULT_CONFIG,
    project_root: str | Path = PROJECT_ROOT,
    registry_path: str | Path = DEFAULT_REGISTRY,
    python_executable: str | Path = sys.executable,
    log_dir: str | Path = DEFAULT_LOG_DIR,
    device: str = "auto",
    threads: int = 4,
    workers: int = 4,
    local_files_only: bool = True,
    stop_on_error: bool = False,
    reuse_forecasts: bool = True,
    overwrite: bool = False,
) -> ChallengerRunResult:
    """Consume official controls and publish a completely separate shadow tree."""

    selected = normalize_zones(zones)
    selected_mode = _normalise_challenger_mode(mode)
    delivery = normalise_delivery_day(delivery_day)
    variants = {zone: _requested_control_variants(selected_mode, zone) for zone in selected}
    config = load_regime_challenger_config(
        config_path, project_root=project_root
    )
    root = Path(project_root).expanduser().resolve()
    if int(threads) < 1 or int(workers) < 1:
        raise ValueError("threads et workers doivent etre >= 1.")

    archive_paths: dict[str, Path] = {}
    if reuse_forecasts:
        archive_paths = _existing_control_paths(
            registry_path=registry_path,
            zones=selected,
            delivery_day=delivery,
        )
    else:
        batch = run_forecast_batch(
            zones=selected,
            delivery_day=delivery,
            project_root=root,
            registry_path=registry_path,
            python_executable=python_executable,
            log_dir=log_dir,
            device=device,
            threads=int(threads),
            workers=int(workers),
            local_files_only=local_files_only,
            stop_on_error=stop_on_error,
            # Both now requests Kalman exports in the live launcher. This
            # challenger only consumes the sealed native archive, whose
            # autonomous and FR/NL MKOnline controls are built by Production.
            mode="production" if selected_mode == "both" else selected_mode,
        )
        print_batch_summary(batch)
        failures = [result for result in batch.results if not result.ok]
        if failures:
            messages = "; ".join(
                f"{item.zone}: {item.message.splitlines()[0]}" for item in failures
            )
            raise RegimeChallengerRunnerError(
                "Le controle officiel a echoue; challenger non lance: " + messages
            )
        archive_paths = {
            result.zone: result.archive_path
            for result in batch.results
            if result.archive_path is not None
        }
        if set(archive_paths) != set(selected):
            raise RegimeChallengerRunnerError(
                "Le batch officiel n'a pas retourne toutes les archives controles."
            )

    controls = {
        zone: load_control_archive(
            archive_paths[zone],
            zone=zone,
            delivery_day=delivery,
            mode=selected_mode,
            project_root=root,
        )
        for zone in selected
    }
    label_end_day = date.fromisoformat(delivery) - timedelta(
        days=config.label_delay_days
    )
    histories: dict[tuple[str, str], pd.DataFrame] = {}
    live_baselines: dict[tuple[str, str], pd.DataFrame] = {}
    all_indices: list[pd.DatetimeIndex] = []
    for zone, control in controls.items():
        for variant in variants[zone]:
            history = _variant_history(
                control, variant=variant, label_end_day=label_end_day
            )
            live = _variant_live(control, variant=variant)
            histories[(zone, variant)] = history
            live_baselines[(zone, variant)] = live
            all_indices.extend([history.index, live.index])
    earliest = min(index[0] for index in all_indices) - pd.Timedelta(days=2)
    latest = max(index[-1] for index in all_indices)
    grid = pd.date_range(earliest, latest, freq="h", tz="UTC")
    residual, pit_audit = _load_residual_features(config, grid=grid)
    peer_prices = _autonomous_price_context(
        controls, label_end_day=label_end_day, grid=grid
    )

    evaluation_parts: list[pd.DataFrame] = []
    live_parts: list[pd.DataFrame] = []
    importance_parts: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for zone in selected:
        control = controls[zone]
        for variant in variants[zone]:
            key = (zone, variant)
            label = f"{zone}/{variant}"
            try:
                history = histories[key]
                live = live_baselines[key]
                combined = pd.concat([history.loc[:, list(QUANTILES)], live])
                combined = combined.loc[~combined.index.duplicated(keep="last")]
                baseline_grid = combined.reindex(grid)
                peers = {
                    peer_zone: series
                    for peer_zone, series in peer_prices.items()
                    if peer_zone != zone
                }
                if variant == "blend":
                    # The spread between the MKOnline blend and the autonomous
                    # view is known at issuance and is particularly informative
                    # when experts disagree during a regime transition.
                    peers[f"{zone.lower()}_autonomous"] = peer_prices[zone]
                features_grid = build_regime_features(
                    residual,
                    baseline_grid,
                    peer_prices=peers,
                    timezone=config.timezone,
                    solar_hours=config.solar_hours,
                )
                history_features = features_grid.reindex(history.index)
                prequential, fold_audit = prequential_regime_predictions(
                    history_features,
                    history["actual"],
                    history.loc[:, list(QUANTILES)],
                    config=config,
                    thread_count=int(threads),
                )
                evaluation_parts.append(
                    _evaluation_hourly(
                        prequential,
                        zone=zone,
                        variant=variant,
                        timezone=control.timezone,
                        threshold=config.shock_error_threshold_eur_mwh,
                        solar_hours=config.solar_hours,
                    )
                )
                fitted = fit_regime_model(
                    history_features,
                    history["actual"],
                    history["q50"],
                    config=config,
                    thread_count=int(threads),
                )
                live_prediction = predict_regime_adjustment(
                    fitted,
                    features_grid.reindex(live.index),
                    live,
                    config=config,
                )
                live_frame = _live_hourly(
                    live_prediction,
                    control=control,
                    variant=variant,
                    threshold=config.shock_error_threshold_eur_mwh,
                    solar_hours=config.solar_hours,
                )
                live_parts.append(live_frame.assign(zone=zone))
                importance_parts.append(
                    fitted.feature_importance.assign(zone=zone, variant=variant)
                )
                solar_live = live_frame.loc[
                    live_frame["local_hour"].isin(config.solar_hours)
                ]
                fr_delta_feature = "fr_residual_load__solar_mean_delta_d1"
                live_feature_slice = features_grid.reindex(live.index)
                diagnostics[label] = {
                    "fit": dict(fitted.diagnostics),
                    "prequential_folds": fold_audit,
                    "day_signal": {
                        "solar_probability_mean": float(
                            solar_live["shock_probability"].mean()
                        ),
                        "solar_probability_max": float(
                            solar_live["shock_probability"].max()
                        ),
                        "solar_premium_mean_eur_mwh": float(
                            solar_live["shock_premium"].mean()
                        ),
                        "solar_premium_max_eur_mwh": float(
                            solar_live["shock_premium"].max()
                        ),
                        "fr_residual_solar_mean_delta_d1_gw": float(
                            live_feature_slice[fr_delta_feature].dropna().iloc[0]
                        )
                        if live_feature_slice[fr_delta_feature].notna().any()
                        else np.nan,
                    },
                }
            except Exception as exc:
                errors[label] = str(exc)
                if stop_on_error:
                    raise

    if errors:
        raise RegimeChallengerRunnerError(
            "Echec d'une ou plusieurs branches challenger: "
            + "; ".join(f"{key}: {value}" for key, value in errors.items())
        )
    evaluation = pd.concat(evaluation_parts, ignore_index=True)
    live_hourly = pd.concat(live_parts, ignore_index=True)
    feature_importance = pd.concat(importance_parts, ignore_index=True)
    metrics, daily = summarize_metrics(
        evaluation,
        probability_threshold=config.gate_probability_threshold,
        solar_hours=config.solar_hours,
    )

    final = (config.output_root / delivery).resolve()
    config.output_root.mkdir(parents=True, exist_ok=True)
    staging = config.output_root / f".{delivery}.{uuid.uuid4().hex}.tmp"
    staging.mkdir(parents=False, exist_ok=False)
    report_paths: list[tuple[str, Path]] = []
    try:
        evaluation.to_csv(staging / "prequential_hourly.csv.gz", index=False)
        metrics.to_csv(staging / "metrics.csv", index=False)
        daily.to_csv(staging / "daily_metrics.csv", index=False)
        feature_importance.to_csv(staging / "feature_importance.csv", index=False)
        live_hourly.to_csv(staging / "live_forecasts.csv", index=False)
        _write_json(staging / "diagnostics.json", diagnostics)

        controls_manifest = {
            zone: {
                "path": control.path,
                **dict(control.hashes),
                "forecast_cutoff_utc": control.manifest.get("forecast_cutoff_utc"),
                "current_actual_hours": int(control.current_actual.notna().sum()),
            }
            for zone, control in controls.items()
        }
        common_manifest = {
            "schema_version": 1,
            "challenger_id": config.challenger_id,
            "run_type": "shadow_day_ahead",
            "forecast_status": "shadow_challenger",
            "production_eligible": False,
            "automatic_promotion": False,
            "delivery_day_local": delivery,
            "timezone": config.timezone,
            "mode": selected_mode,
            "zones": list(selected),
            "forecast_origin_utc": scheduled_origin_utc(
                delivery, timezone=config.timezone
            ),
            "latest_training_label_day": label_end_day,
            "label_delay_days": config.label_delay_days,
            "config": {
                "path": config.source_path,
                "sha256": _sha256(config.source_path),
            },
            "implementation": {
                "runner_sha256": _sha256(Path(__file__).resolve()),
                "model_sha256": _sha256(
                    root / "chronos2_hourly" / "regime_challenger.py"
                ),
                "report_sha256": _sha256(
                    root / "chronos2_hourly" / "regime_challenger_report.py"
                ),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "controls": controls_manifest,
            "pit_audit": pit_audit,
            "reuse_forecasts": reuse_forecasts,
            "official_control_run_invoked": not reuse_forecasts,
            "official_control_mode": ("production" if selected_mode == "both" else selected_mode) if not reuse_forecasts else None,
            "official_control_run_may_publish_runs_live": not reuse_forecasts,
            "production_changed_by_challenger": False,
            "writes_runs_live_by_challenger": False,
            "current_actual_loaded_with_control": True,
            "current_actual_used_for_fit": False,
            "current_actual_used_for_prediction": False,
            "current_actual_reporting_only": True,
            "limitations": [
                "Challenger specialise sur les chocs haussiers des heures solaires.",
                "Les cinq charges residuelles agregent charge, eolien et solaire; "
                "les composantes physiques restent a ajouter lorsque leur PIT est complet.",
                "Aucune promotion automatique: une fenetre shadow plus longue est requise.",
                "Le seuil du gate reste une recette exploratoire; il n'a pas encore "
                "ete valide sur un holdout final scelle independant.",
            ],
        }
        _write_json(staging / "run_manifest.json", common_manifest)

        for zone in selected:
            zone_directory = staging / zone.lower()
            zone_directory.mkdir(parents=True, exist_ok=True)
            zone_live = live_hourly.loc[live_hourly["zone"].eq(zone)].copy()
            zone_metrics = metrics.loc[metrics["zone"].eq(zone)].copy()
            zone_daily = daily.loc[daily["zone"].eq(zone)].copy()
            zone_importance = feature_importance.loc[
                feature_importance["zone"].eq(zone)
            ].copy()
            zone_live.to_csv(zone_directory / "forecast.csv", index=False)
            zone_metrics.to_csv(zone_directory / "metrics.csv", index=False)
            zone_daily.to_csv(zone_directory / "daily_metrics.csv", index=False)
            zone_importance.to_csv(
                zone_directory / "feature_importance.csv", index=False
            )
            zone_diagnostics = {
                "day_diagnostic": {
                    key.split("/", 1)[1]: value["day_signal"]
                    for key, value in diagnostics.items()
                    if key.startswith(zone + "/")
                },
                "calibration_gate": {
                    "passes": False,
                    "decision": "SHADOW_ONLY",
                    "reason": "automatic_promotion=false",
                },
                "pit": pit_audit,
                "provenance": controls_manifest[zone],
                "limitations": common_manifest["limitations"],
                "models": {
                    key.split("/", 1)[1]: value
                    for key, value in diagnostics.items()
                    if key.startswith(zone + "/")
                },
            }
            zone_manifest = {
                **common_manifest,
                "zone": zone,
                "control": controls_manifest[zone],
            }
            report = zone_directory / f"price_regime_challenger_{zone.lower()}_{delivery}.html"
            write_regime_challenger_report(
                report,
                manifest=zone_manifest,
                hourly=zone_live,
                metrics=zone_metrics,
                daily=zone_daily,
                feature_importance=zone_importance,
                diagnostics=zone_diagnostics,
            )
            report_paths.append((zone, report))

        index_path = staging / "price_regime_challenger.html"
        index_path.write_text(
            _index_html(
                delivery_day=delivery,
                mode=selected_mode,
                reports=report_paths,
                metrics=metrics,
                root=staging,
            ),
            encoding="utf-8",
        )
        checksums = []
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            checksums.append(
                {
                    "path": path.relative_to(staging).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        _write_json(
            staging / "artifact_checksums.json",
            {"algorithm": "sha256", "artifacts": checksums},
        )
        _atomic_publish(staging, final, overwrite=overwrite)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    # ``report_paths`` were expressed under staging.  Rebuild explicitly after
    # the atomic rename to avoid leaking a vanished temporary path.
    final_reports = tuple(
        final
        / zone.lower()
        / f"price_regime_challenger_{zone.lower()}_{delivery}.html"
        for zone in selected
    )
    return ChallengerRunResult(
        delivery_day=delivery,
        mode=selected_mode,
        output_dir=final,
        report_paths=final_reports,
        reused_controls=reuse_forecasts,
    )


def build_plan(
    *,
    zones: Sequence[str],
    mode: str,
    delivery_day: str | date | None,
    config_path: str | Path,
    project_root: str | Path = PROJECT_ROOT,
    reuse_forecasts: bool = True,
) -> Mapping[str, Any]:
    selected = normalize_zones(zones)
    selected_mode = _normalise_challenger_mode(mode)
    delivery = normalise_delivery_day(delivery_day)
    config = load_regime_challenger_config(config_path, project_root=project_root)
    return {
        "action": "price_regime_challenger",
        "challenger_id": config.challenger_id,
        "delivery_day": delivery,
        "mode": selected_mode,
        "zones": list(selected),
        "variants": {
            zone: list(_requested_control_variants(selected_mode, zone))
            for zone in selected
        },
        "reuse_forecasts": bool(reuse_forecasts),
        "official_run_first": not reuse_forecasts,
        "official_control_mode": ("production" if selected_mode == "both" else selected_mode) if not reuse_forecasts else None,
        "production_eligible": False,
        "output": config.output_root / delivery,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--zones", nargs="+", default=["FR", "DE", "BE", "NL", "ES"])
    parser.add_argument("--mode", default="Both")
    parser.add_argument("--delivery-day")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    control_source = parser.add_mutually_exclusive_group()
    control_source.add_argument(
        "--reuse-forecasts",
        dest="reuse_forecasts",
        action="store_true",
        help="Utiliser les archives officielles existantes (defaut).",
    )
    control_source.add_argument(
        "--run-forecasts-first",
        dest="reuse_forecasts",
        action="store_false",
        help="Lancer explicitement le batch officiel avant le challenger.",
    )
    parser.set_defaults(reuse_forecasts=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                _json_safe(
                    build_plan(
                        zones=args.zones,
                        mode=args.mode,
                        delivery_day=args.delivery_day,
                        config_path=args.config,
                        reuse_forecasts=args.reuse_forecasts,
                    )
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    try:
        result = run_challenger(
            zones=args.zones,
            mode=args.mode,
            delivery_day=args.delivery_day,
            config_path=args.config,
            registry_path=args.registry,
            python_executable=args.python_executable,
            log_dir=args.log_dir,
            device=args.device,
            threads=args.threads,
            workers=args.workers,
            local_files_only=not args.allow_model_download,
            stop_on_error=args.stop_on_error,
            reuse_forecasts=args.reuse_forecasts,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"ECHEC challenger regime: {exc}", file=sys.stderr)
        return 3
    print("\nChallenger de regime publie (shadow, non production)")
    print(f"Dossier:  {result.output_dir}")
    print(f"Synthese: {result.output_dir / 'price_regime_challenger.html'}")
    for report in result.report_paths:
        print(f"Rapport:  {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
