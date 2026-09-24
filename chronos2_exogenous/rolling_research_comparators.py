"""Read the exact observed/official-Storm snapshot of a published HTML report.

Only JSON data embedded by Plotly is decoded: no JavaScript is evaluated. The
HTML bytes are bound to the existing export manifest. These observations are
reporting-only, not an input to calibration or evidence of latest API state.
"""
from __future__ import annotations

import base64
import hashlib
from html import unescape
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from .retrospective_incumbents import _check_plain_path, _hours, _unique_json


class RollingReportComparatorError(ValueError):
    pass


_TZ = "Europe/Paris"
_STORM = "Storm officiel dashboard P50"
_PLOT = re.compile(r'Plotly\.newPlot\(\s*("(?:[^"\\]|\\.)*")\s*,\s*')


def _reject_constant(value: str) -> None:
    raise RollingReportComparatorError(f"Constante JSON non finie interdite : {value}.")


def _decoder() -> json.JSONDecoder:
    return json.JSONDecoder(object_pairs_hook=_unique_json, parse_constant=_reject_constant)


def _numeric(value: Any, *, count: int) -> np.ndarray:
    if isinstance(value, dict):
        if set(value) - {"dtype", "bdata", "shape"} or value.get("dtype") not in {"f8", "f4", "i4", "i2", "i1", "u4", "u2", "u1"}:
            raise RollingReportComparatorError("Encodage numérique Plotly non pris en charge.")
        if "shape" in value and str(value["shape"]).strip() != str(count):
            raise RollingReportComparatorError("Une série horaire Plotly doit être unidimensionnelle.")
        try:
            raw = base64.b64decode(value["bdata"], validate=True)
            result = np.frombuffer(raw, dtype=np.dtype(value["dtype"]).newbyteorder("<")).astype(float)
        except (ValueError, TypeError, KeyError) as exc:
            raise RollingReportComparatorError("Tableau binaire Plotly invalide.") from exc
    elif isinstance(value, list):
        if any(isinstance(item, (bool, str, list, dict)) for item in value):
            raise RollingReportComparatorError("Les valeurs horaires doivent être numériques ou nulles.")
        result = np.asarray(value, dtype=float)
    else:
        raise RollingReportComparatorError("Valeurs horaires Plotly absentes.")
    if result.shape != (count,) or np.isinf(result).any():
        raise RollingReportComparatorError("Longueur ou valeurs horaires Plotly invalides.")
    return result


def _series(trace: dict[str, Any], *, official: bool) -> pd.Series:
    timestamps = trace.get("x")
    if not isinstance(timestamps, list) or not timestamps or len(timestamps) > 20000 or any(not isinstance(x, str) for x in timestamps):
        raise RollingReportComparatorError("Timeline horaire Plotly invalide.")
    try:
        stamps = [pd.Timestamp(value) for value in timestamps]
        if any(pd.isna(value) for value in stamps):
            raise ValueError("NaT")
        aware = [value.tzinfo is not None for value in stamps]
        if all(aware):
            index = pd.DatetimeIndex(pd.to_datetime(stamps, utc=True))
        elif not any(aware):
            # The native official-dashboard contract retains standard time
            # (fold=1), never inventing the missing first autumn fold.
            # Full backtest traces contain both physical folds in order.
            index = pd.DatetimeIndex(stamps).tz_localize(
                _TZ, ambiguous=False if official else "infer", nonexistent="raise").tz_convert("UTC")
        else:
            raise ValueError("mixed timezone")
    except (ValueError, TypeError, OverflowError) as exc:
        raise RollingReportComparatorError("Dates Plotly invalides ou DST ambigu non résolu.") from exc
    if index.has_duplicates or not index.is_monotonic_increasing or any(value != value.floor("h") for value in index):
        raise RollingReportComparatorError("La timeline UTC doit être horaire, unique et ordonnée.")
    return pd.Series(_numeric(trace.get("y"), count=len(index)), index=index)


def _graphs(rendered: str) -> list[tuple[str, list[dict[str, Any]], dict[str, Any]]]:
    result = []
    decoder = _decoder()
    for match in _PLOT.finditer(rendered):
        try:
            traces, end = decoder.raw_decode(rendered, match.end())
            position = end
            while rendered[position].isspace():
                position += 1
            if rendered[position] != ",":
                raise ValueError("layout separator")
            position += 1
            while rendered[position].isspace():
                position += 1
            layout, _ = decoder.raw_decode(rendered, position)
        except (ValueError, IndexError) as exc:
            raise RollingReportComparatorError("Le graphe Plotly n'est pas un couple JSON données/layout valide.") from exc
        if not isinstance(traces, list) or any(not isinstance(trace, dict) for trace in traces) or not isinstance(layout, dict):
            raise RollingReportComparatorError("Structure de graphe Plotly inattendue.")
        result.append((json.loads(match.group(1)), traces, layout))
    return result


def _daily_payload(rendered: str, zone: str) -> dict[str, dict[str, Any]]:
    payloads = []
    for marker in re.finditer(r"const payload =\s*", rendered):
        try:
            value, _ = _decoder().raw_decode(rendered, marker.end())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("records"), list) and isinstance(value.get("metrics"), list):
            payloads.append(value)
    if len(payloads) != 1:
        raise RollingReportComparatorError("Payload Statistics unique introuvable.")
    records: dict[str, dict[str, Any]] = {}
    for record in payloads[0]["records"]:
        if not isinstance(record, dict) or record.get("zone") != zone or record.get("sample") != "daily":
            continue
        key = record.get("period_start")
        if not isinstance(key, str) or key in records:
            raise RollingReportComparatorError("Journée Statistics absente ou dupliquée.")
        records[key] = record
    return records


def _compare_mean(record: dict[str, Any], field: str, expected: float | None) -> None:
    found = record.get(field)
    if expected is None:
        if found is not None:
            raise RollingReportComparatorError(f"Statistics/{field} devrait rester vide.")
    elif isinstance(found, bool) or not isinstance(found, (int, float)) or not np.isfinite(found) or not np.isclose(found, expected, rtol=0.0, atol=1e-8):
        raise RollingReportComparatorError(f"Statistics/{field} divergent des vraies heures du graphe.")


def load_rolling_report_comparators(project_root: str | Path, *, zone: str,
                                    delivery_day: str, start_day: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return ``timestamp, delivery_start_utc, zone, actual, q50`` in UTC.

    The interval is inclusive of both civil dates. Prefer the Kalman HTML of
    the exact requested batch (autonomous only if that variant is absent).
    Missing Storm DST hours remain NaN; observed-only metrics retain physical
    hours recovered from the same HTML's complete backtest observed trace.
    """
    if zone not in {"FR", "DE", "BE", "NL"}:
        raise RollingReportComparatorError("Pays hors du protocole FR/DE/BE/NL.")
    beginning, ending = _hours(start_day, _TZ)[0], _hours(delivery_day, _TZ)[-1]
    if beginning > ending or (pd.Timestamp(delivery_day) - pd.Timestamp(start_day)).days > 365:
        raise RollingReportComparatorError("Fenêtre Statistics invalide (366 journées au maximum).")
    expected = pd.date_range(beginning, ending, freq="h")
    root = Path(project_root).absolute()
    batch = root / "runs/exports" / delivery_day
    manifest_path = batch / "current_batch_manifest.json"
    _check_plain_path(manifest_path, root)
    if not manifest_path.is_file():
        raise RollingReportComparatorError("Manifeste du batch demandé absent.")
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = _decoder().decode(manifest_bytes.decode("utf-8-sig"))
    except (ValueError, UnicodeError) as exc:
        raise RollingReportComparatorError("Manifeste du batch illisible.") from exc
    if not isinstance(manifest, dict) or manifest.get("delivery_day") != delivery_day or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1 or not isinstance(manifest.get("exports"), list):
        raise RollingReportComparatorError("Identité du batch incompatible.")
    selected = {}
    for entry in manifest["exports"]:
        if not isinstance(entry, dict):
            raise RollingReportComparatorError("Entrée du batch invalide.")
        if entry.get("zone") == zone and entry.get("variant") in {"kalman", "autonomous"}:
            if entry["variant"] in selected:
                raise RollingReportComparatorError("Rapport pays/variante dupliqué.")
            selected[entry["variant"]] = entry
    variant = "kalman" if "kalman" in selected else "autonomous"
    if variant not in selected:
        raise RollingReportComparatorError("Aucun rapport opérationnel compatible pour ce pays.")
    if selected[variant].get("source_model") != {"kalman": "residual_kalman", "autonomous": "residual_corrected"}[variant]:
        raise RollingReportComparatorError("Étape du modèle incohérente dans le manifeste du rapport.")
    html_entry = selected[variant].get("html", {})
    relative = f"{zone.lower()}/{variant}/forecast_{zone.lower()}_{delivery_day}_{variant}.html"
    if not isinstance(html_entry, dict) or not isinstance(html_entry.get("path"), str) or html_entry["path"].replace("\\", "/") != relative or not re.fullmatch(r"[a-f0-9]{64}", str(html_entry.get("sha256", ""))):
        raise RollingReportComparatorError("Chemin ou SHA du rapport HTML non conforme.")
    report_path = batch / relative
    _check_plain_path(report_path, root)
    raw = report_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != html_entry["sha256"]:
        raise RollingReportComparatorError("SHA256 du rapport HTML divergent.")
    rendered = raw.decode("utf-8-sig")
    graphs = _graphs(rendered)
    official_graphs = [(identifier, traces) for identifier, traces, _layout in graphs if any(trace.get("name") == _STORM for trace in traces)]
    if len(official_graphs) != 1:
        raise RollingReportComparatorError("Graphe horaire Storm officiel unique introuvable.")
    identifier, traces = official_graphs[0]
    observed_traces = [trace for trace in traces if trace.get("name") == "Observé"]
    storm_traces = [trace for trace in traces if trace.get("name") == _STORM]
    if len(observed_traces) != 1 or len(storm_traces) != 1:
        raise RollingReportComparatorError("Courbes observée/Storm uniques requises dans le même graphe.")
    actual = _series(observed_traces[0], official=True)
    storm = _series(storm_traces[0], official=True)
    if not actual.index.equals(storm.index):
        raise RollingReportComparatorError("Les heures observées et Storm du graphe diffèrent.")
    sources = [identifier]
    for full_id, full_traces, layout in graphs:
        if layout.get("title", {}).get("text") != f"{zone} — backtest glissant":
            continue
        candidates = [trace for trace in full_traces if trace.get("name") == "Observé"]
        if len(candidates) != 1:
            raise RollingReportComparatorError("Courbe physique observée non unique.")
        full = _series(candidates[0], official=False)
        if not full.index.equals(pd.date_range(full.index[0], full.index[-1], freq="h")):
            raise RollingReportComparatorError("La courbe observée supplémentaire n'est pas physiquement continue.")
        common = actual.index.intersection(full.index)
        if not np.allclose(actual.reindex(common), full.reindex(common), atol=1e-9, rtol=0.0, equal_nan=True):
            raise RollingReportComparatorError("Observations contradictoires entre les graphes du même rapport.")
        actual = actual.combine_first(full)
        sources.append(full_id)
    naive = expected.tz_convert(_TZ).tz_localize(None)
    canonical = naive.tz_localize(_TZ, ambiguous=False, nonexistent="raise").tz_convert("UTC")
    allowed_missing = expected[expected != canonical]
    in_window = storm.reindex(expected)
    missing = expected[~np.isfinite(in_window.to_numpy(float))]
    if len(missing.difference(allowed_missing)):
        raise RollingReportComparatorError("Storm officiel ne couvre pas la fenêtre demandée hors omission DST autorisée.")
    if len(expected.difference(actual.index)):
        raise RollingReportComparatorError("Heures observées physiques absentes du snapshot HTML.")
    frame = pd.DataFrame({"timestamp": expected, "delivery_start_utc": expected,
                          "zone": zone, "actual": actual.reindex(expected).to_numpy(float),
                          "q50": in_window.to_numpy(float)})
    daily = _daily_payload(rendered, zone)
    for day, block in frame.groupby(frame.timestamp.dt.tz_convert(_TZ).dt.strftime("%Y-%m-%d")):
        record = daily.get(day)
        if record is None:
            raise RollingReportComparatorError(f"Journée {day} absente de Statistics.")
        observed = np.isfinite(block.actual.to_numpy(float))
        if observed.any() and not observed.all():
            raise RollingReportComparatorError(f"Observations partielles pour {day} : une journée doit être complète ou vide.")
        paired = observed & np.isfinite(block.q50.to_numpy(float))
        _compare_mean(record, "observed_mean_price", float(block.loc[paired, "actual"].mean()) if paired.any() else None)
        _compare_mean(record, "benchmark_mean_price", float(block.loc[paired, "q50"].mean()) if paired.any() else float(block.q50.mean()))
    plain = unescape(re.sub(r"<[^>]*>", " ", rendered))
    generated = re.search(r"Généré le ([^<\r\n]{1,100}?)\s*[—–·]\s*Script", plain)
    extracted = sorted(set(re.findall(r"extraction (\d{4}-\d{2}-\d{2} [\d:.]+\+\d{2}:\d{2})", plain)))
    warnings = ["Snapshot du rapport publié, pas une nouvelle lecture API des dernières observations.",
                "Données exclusivement destinées au reporting : interdites comme calibration ou inputs du modèle.",
                "Storm officiel peut inclure le fallback historique natif audité du dashboard; aucun Storm strict08 substitué.",
                "Les moyennes Statistics validées utilisent les heures communes avec Storm; une moyenne observée sur toutes les heures physiques peut différer le jour DST."]
    from chronos2_hourly.reporting import STORM_DASHBOARD_CONTRACT_ID, storm_benchmark_contracts
    source = {"kind": "checksum_verified_operational_html_hourly_snapshot",
              "path": str(report_path), "sha256": digest,
              "observation_extraction_timestamps": extracted, "network_refreshed": False}
    storm_contract = {**storm_benchmark_contracts(zone, timezone=_TZ)[STORM_DASHBOARD_CONTRACT_ID],
        "status": "loaded_from_checksum_verified_report_snapshot",
        "artifact_filename": str(report_path), "source_representation": "plotly_json_in_verified_html",
        "used_for_prediction": False, "used_for_calibration": False, "latest_known_verified": False,
        "materialization_audit": {"source": source, "available_hours": int(frame.q50.notna().sum()),
            "dst": {"interpolation": False, "native_allowed_missing_utc": [stamp.isoformat() for stamp in missing]}},
        "report_note": "Snapshot horaire du rapport opérationnel publié, vérifié par SHA256; "
                       "aucune nouvelle lecture API. Cache day-ahead GEMS avec fallback historique natif "
                       "audité du dashboard; heures DST absentes non interpolées."}
    return frame, {"kind": "checksum_verified_operational_html_hourly_snapshot", "zone": zone,
        "report_label": "Storm officiel dashboard", "status": "loaded_from_checksum_verified_report_snapshot",
        "storm_contract": storm_contract,
        "delivery_day": delivery_day, "start_day": start_day, "timezone": _TZ, "variant": variant,
        "batch_manifest": {"path": str(manifest_path), "sha256": hashlib.sha256(manifest_bytes).hexdigest()},
        "html": {"path": str(report_path), "sha256": digest}, "hourly_graph_ids": sources,
        "snapshot_generated_text": generated.group(1).strip() if generated else None,
        "observation_extraction_timestamps": extracted, "network_refreshed": False,
        "latest_known_verified": False, "used_for_prediction": False, "used_for_calibration": False,
        "statistics_daily_means_verified": True, "physical_hours": len(frame),
        "statistics_observed_mean_scope": "same_finite_hours_as_official_storm",
        "all_observed_metric_scope": "all_physical_hours",
        "observed_hours": int(frame.actual.notna().sum()), "storm_hours": int(frame.q50.notna().sum()),
        "paired_hours": int((frame.actual.notna() & frame.q50.notna()).sum()),
        "dst_policy": "official_naive_standard_time_fold_only; full_observed_physical_folds_preserved",
        "missing_storm_utc": [stamp.isoformat() for stamp in missing], "interpolation": False,
        "warnings": warnings}
