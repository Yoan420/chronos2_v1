"""Frozen comparator loading and strictly chronological candidate selection."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


def digest_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def physical_index(start_day: str, end_day: str, timezone: str = "Europe/Paris") -> pd.DatetimeIndex:
    start = pd.Timestamp(start_day).tz_localize(timezone)
    end = (pd.Timestamp(end_day) + pd.Timedelta(days=1)).tz_localize(timezone)
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def _array(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        if set(value) - {"dtype", "bdata", "shape"}:
            raise ValueError("Unexpected encoded Plotly array.")
        dtype = np.dtype(value["dtype"])
        if dtype.kind not in "iuf" or dtype.itemsize > 8:
            raise ValueError("Unsupported numeric Plotly array.")
        result = np.frombuffer(base64.b64decode(value["bdata"], validate=True), dtype=dtype)
        if "shape" in value:
            shape = tuple(int(v.strip()) for v in str(value["shape"]).split(","))
            if shape != (len(result),):
                raise ValueError("Comparator Plotly arrays must be one-dimensional with matching shape.")
    else:
        result = np.asarray(value, dtype=float)
    if result.ndim != 1:
        raise ValueError("Comparator Plotly arrays must be one-dimensional.")
    return result


def _missing_pair_hours(path: Path, zone: str, explicit: Any) -> tuple[pd.DatetimeIndex, list[dict]]:
    """Read the declared Storm gap; never infer which autumn fold is absent."""
    if explicit is not None:
        values = [pd.Timestamp(v) for v in explicit]
        if any(pd.isna(v) or v.tzinfo is None for v in values):
            raise ValueError("allowed_missing_utc requires explicit aware UTC identities.")
        return pd.DatetimeIndex(pd.to_datetime(values, utc=True)), [{"source": "explicit_caller_contract"}]
    audit_paths = []
    nuclear = path.parent / "nuclear_report_audit.json"
    if nuclear.is_file():
        audit_paths.append(nuclear)
    sidecar = path.parent / "kalman_operational_sidecar_audit.json"
    sidecar_evidence = []
    if sidecar.is_file():
        raw = sidecar.read_bytes()
        payload = json.loads(raw)
        if payload.get("zone") not in (None, zone):
            raise ValueError("Comparator sidecar zone mismatch.")
        if payload.get("source_archive"):
            audit_paths.append(Path(payload["source_archive"]) / "statistics_history_audit.json")
            sidecar_evidence.append({"path": str(sidecar), "sha256": hashlib.sha256(raw).hexdigest()})
    declarations, evidence = [], []
    expected_series = f"power.price.{zone.lower()}.euromwh.h.fcst.3mv.storm.da.cache"

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("series") == expected_series and "missing_timestamps" in value:
                if value.get("source", {}).get("zone") not in (None, zone):
                    raise ValueError("Storm audit zone mismatch.")
                if value.get("missing_timestamps_truncated"):
                    raise ValueError("Truncated Storm missing-hour declaration.")
                stamps = [pd.Timestamp(row["utc"]) for row in value["missing_timestamps"]]
                if any(pd.isna(v) or v.tzinfo is None for v in stamps):
                    raise ValueError("Storm audit has ambiguous missing-hour identities.")
                declarations.append(tuple(sorted(v.tz_convert("UTC").isoformat() for v in stamps)))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for audit_path in audit_paths:
        if not audit_path.is_file():
            continue
        raw = audit_path.read_bytes()
        payload = json.loads(raw)
        if payload.get("zone") not in (None, zone):
            raise ValueError("Comparator report audit zone mismatch.")
        visit(payload)
        evidence.append({"path": str(audit_path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()})
    if declarations and len(set(declarations)) != 1:
        raise ValueError("Conflicting Storm missing-hour declarations.")
    values = declarations[0] if declarations else ()
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True)), sidecar_evidence + evidence


def _trace_index(values: Any, *, timezone: str, missing: pd.DatetimeIndex | None = None) -> pd.DatetimeIndex:
    """Recover only an exact physical grid from a report's explicit time axis.

    Plotly's published reports contain local-naive datetime arrays. Their
    ordering retains both autumn folds; a full grid must match every local
    timestamp in order. A paired grid may omit only explicitly audited UTC
    identities. This is not generic ambiguous='infer' or silent UTC guessing.
    """
    parsed = [pd.Timestamp(v) for v in values]
    if not parsed or any(pd.isna(v) for v in parsed):
        raise ValueError("Missing comparator timestamps.")
    aware = [v.tzinfo is not None for v in parsed]
    if any(aware) and not all(aware):
        raise ValueError("Mixed naive/aware comparator timestamps.")
    if all(aware):
        index = pd.DatetimeIndex(pd.to_datetime(parsed, utc=True))
    else:
        local = pd.DatetimeIndex(parsed)
        expected = physical_index(str(local.min().date()), str(local.max().date()), timezone)
        if missing is not None:
            expected = expected.difference(missing, sort=False)
        expected_local = expected.tz_convert(timezone).tz_localize(None)
        if not local.equals(expected_local):
            raise ValueError("Local comparator axis does not match an exact physical grid and audited DST gaps.")
        index = expected
    if index.has_duplicates or not index.is_monotonic_increasing or not index.equals(index.floor("h")):
        raise ValueError("Duplicate, unordered, or non-hourly comparator timestamps.")
    return index


def read_report_comparator(path: str | Path, *, zone: str, label: str,
                           timezone: str = "Europe/Paris",
                           allowed_missing_utc: Any = None) -> tuple[pd.DataFrame, dict]:
    """Read only JSON arguments, NEVER execute report JavaScript.

    The expected forecast trace is explicitly named by the caller. We read
    the full history and overlay the report's paired, refreshed observations.
    A comparator chosen here stays fixed: no hindsight best-model-per-day.
    """
    path = Path(path).resolve()
    raw = path.read_bytes()
    document = raw.decode("utf-8-sig")
    missing, time_evidence = _missing_pair_hours(path, zone, allowed_missing_utc)
    decoder = json.JSONDecoder()
    histories, paired = [], []
    for match in re.finditer(r"Plotly\.newPlot\(", document):
        tail = document[match.end():].lstrip()
        try:
            _, end = decoder.raw_decode(tail)
            tail = tail[end:].lstrip()
            if not tail.startswith(","):
                continue
            traces, _ = decoder.raw_decode(tail[1:].lstrip())
        except (ValueError, TypeError):
            continue
        if not isinstance(traces, list):
            continue
        matching = [t for t in traces if t.get("name") == label and len(t.get("x", [])) >= 1000]
        if not matching:
            continue
        if len(matching) != 1:
            raise ValueError(f"Ambiguous comparator trace in {path}.")
        observation = [t for t in traces if t.get("name") in {"Observé", "Prix observé"}]
        if len(observation) != 1:
            continue
        storms = [t for t in traces if str(t.get("name", "")).startswith("Storm officiel") and "P50" in t.get("name", "")]
        if len(storms) > 1:
            raise ValueError("Ambiguous Storm comparator trace.")
        gap = missing if storms else None
        trace = matching[0]
        index = _trace_index(trace["x"], timezone=timezone, missing=gap)
        frame = pd.DataFrame({"base": _array(trace["y"])}, index=index)
        observed = observation[0]
        observed_index = _trace_index(observed["x"], timezone=timezone, missing=gap)
        if not observed_index.equals(index):
            raise ValueError("Observed and candidate comparator hours disagree.")
        frame["actual"] = _array(observed["y"])
        frame["storm"] = np.nan
        if len(storms) == 1:
            t = storms[0]
            storm_index = _trace_index(t["x"], timezone=timezone, missing=gap)
            if not storm_index.equals(index):
                raise ValueError("Storm and candidate comparator hours disagree.")
            frame["storm"] = _array(t["y"])
            paired.append(frame)
        else:
            histories.append(frame)
    if not histories and not paired:
        raise ValueError(f"No explicitly named hourly comparator '{label}' in {path}.")
    candidates = sorted(histories, key=len, reverse=True)
    result = candidates[0].copy() if candidates else paired[0].copy()
    for frame in candidates[1:] + paired:
        common = result.index.intersection(frame.index)
        if not np.allclose(result.loc[common, "base"], frame.loc[common, "base"], atol=1e-6, rtol=0):
            raise ValueError("Frozen comparator forecast traces disagree on overlapping hours.")
        result = frame.combine_first(result)
    result = result.sort_index().rename_axis("timestamp").reset_index()
    result["zone"] = zone
    if not np.isfinite(result[["base", "actual"]].to_numpy(float)).all():
        raise ValueError("Frozen comparator contains unavailable observations/forecasts.")
    return result, {
        "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "label": label,
        "zone": zone, "selection": "explicit_fixed_reference_not_daily_oracle",
        "timezone": timezone, "time_axis": "exact_physical_grid; local_naive_not_assumed_UTC",
        "time_axis_audits": time_evidence,
        "allowed_missing_pair_utc": [stamp.isoformat() for stamp in missing],
        "reference_training_pit_certified": False,
        "reference_provenance": "published_report_replay_not_new_prospective_validation",
    }


def read_target(path: str | Path, zone: str) -> pd.DataFrame:
    """Canonical historical targets; latest revisions are research labels only."""
    frame = pd.read_csv(path)
    if set(frame) != {"timestamp", "value"}:
        raise ValueError(f"Unexpected canonical target schema: {path}")
    result = pd.DataFrame({"timestamp": pd.to_datetime(frame.timestamp, utc=True),
                           "actual": pd.to_numeric(frame.value, errors="raise"), "zone": zone})
    if result.timestamp.isna().any() or result.timestamp.duplicated().any():
        raise ValueError("Canonical target timestamps invalid.")
    return result.sort_values("timestamp")


def rolling_select(candidates: pd.DataFrame, targets: pd.DataFrame, *, evaluation_start: str,
                   evaluation_end: str, training_days: int = 365,
                   timezone: str = "Europe/Paris", unavailable_policy: str = "raise") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a physical scenario using only each origin's prior window.

    Scenario prices are calculated WITHOUT labels. Only the choice of the
    finite, predeclared scenario bank is calibrated to historical prices.
    No June-specific objective or selection on the final annual results.
    """
    if training_days < 1:
        raise ValueError("training_days must be positive")
    if unavailable_policy not in {"raise", "abstain"}:
        raise ValueError("unavailable_policy must be raise or abstain")
    frame = candidates.copy()
    frame["timestamp"] = pd.to_datetime(frame.delivery_start_utc, utc=True)
    if frame.duplicated(["timestamp", "zone", "candidate_id"]).any():
        raise ValueError("Duplicate scenario predictions.")
    targets = targets.copy()
    targets["timestamp"] = pd.to_datetime(targets.timestamp, utc=True)
    if targets.duplicated(["timestamp", "zone"]).any():
        raise ValueError("Duplicate target keys.")
    outputs, audits = [], []
    for zone, zframe in frame.groupby("zone", sort=True):
        prices = zframe.pivot(index="timestamp", columns="candidate_id", values="price_eur_mwh").sort_index()
        y = targets.loc[targets.zone.eq(zone)].set_index("timestamp").actual
        ids = sorted(prices.columns)
        for day in pd.date_range(evaluation_start, evaluation_end, freq="D"):
            prior_start = (day - pd.Timedelta(days=training_days)).date().isoformat()
            prior_end = (day - pd.Timedelta(days=1)).date().isoformat()
            train_idx = physical_index(prior_start, prior_end, timezone)
            test_idx = physical_index(day.date().isoformat(), day.date().isoformat(), timezone)
            train = prices.reindex(train_idx)
            actual = y.reindex(train_idx)
            origin = (day - pd.Timedelta(days=1) + pd.Timedelta(hours=8)).tz_localize(timezone).tz_convert("UTC")
            def abstain(reason: str, *, training_complete: bool = False) -> None:
                outputs.append(pd.DataFrame({"timestamp": test_idx, "zone": zone, "expert": np.nan,
                                             "expert_oof": False, "forecast_origin_utc": origin,
                                             "expert_unavailable_reason": reason}))
                audits.append({"zone": zone, "day": str(day.date()), "candidate_id": None,
                               "training_start": prior_start, "training_end": prior_end,
                               "training_days": training_days, "training_hours": len(train_idx),
                               "training_complete": training_complete, "reason": reason,
                               "evaluation_label_used": False})
            if not np.isfinite(train.to_numpy()).all() or not np.isfinite(actual.to_numpy()).all():
                if unavailable_policy == "abstain":
                    abstain(f"incomplete_{training_days}_day_training_window")
                    continue
                raise ValueError(f"{zone}/{day.date()}: incomplete {training_days}-day training window.")
            errors = train.sub(actual, axis=0).abs().mean()
            chosen = min(ids, key=lambda key: (float(errors[key]), key))
            selected = zframe.loc[zframe.candidate_id.eq(chosen)].set_index("timestamp").reindex(test_idx)
            if not np.isfinite(selected.price_eur_mwh.to_numpy(float)).all():
                if unavailable_policy == "abstain":
                    abstain("incomplete_forecast_inputs", training_complete=True)
                    continue
                raise ValueError(f"{zone}/{day.date()}: incomplete expert forecast.")
            selected = selected.copy()
            selected["timestamp"] = test_idx
            selected["zone"] = zone
            selected["expert"] = selected.price_eur_mwh
            selected["expert_oof"] = True
            # Publication timing of canonical historical revisions is NOT proven.
            selected["forecast_origin_utc"] = origin
            outputs.append(selected.reset_index(drop=True))
            audits.append({"zone": zone, "day": str(day.date()), "candidate_id": chosen,
                           "training_start": prior_start, "training_end": prior_end,
                           "training_days": training_days, "training_hours": len(train_idx),
                           "training_complete": True,
                           "training_mae": float(errors[chosen]), "evaluation_label_used": False,
                           "labels_evidence": "canonical_latest_revision_research_only"})
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(audits)


def score_results(frame: pd.DataFrame, *, timezone: str = "Europe/Paris") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Yearly scores preserve EVERY baseline hour; missing experts are exposed."""
    frame = frame.copy()
    frame["day"] = pd.to_datetime(frame.timestamp, utc=True).dt.tz_convert(timezone).dt.date.astype(str)
    metrics, daily = [], []
    for zone, group in frame.groupby("zone"):
        for name in ("base", "expert", "guarded", "storm"):
            ok = np.isfinite(group.actual) & np.isfinite(group[name])
            sample = group.loc[ok]
            if sample.empty:
                continue
            error = sample[name] - sample.actual
            dayrows = []
            for day, block in sample.groupby("day"):
                err = block[name] - block.actual
                row = {"zone": zone, "model": name, "day": day, "hours": len(block),
                       "mae": float(err.abs().mean()), "bias": float(err.mean()),
                       "observed_mean": float(block.actual.mean()), "forecast_mean": float(block[name].mean()),
                       "mean_price_error": float(abs(err.mean())),
                       "activation_share": float(block.weight.gt(0).mean()) if "weight" in block else 0.0}
                daily.append(row)
                dayrows.append(row)
            ds = pd.DataFrame(dayrows)
            metrics.append({"zone": zone, "model": name, "hours": len(sample), "days": len(ds),
                            "coverage": len(sample) / len(group), "mae": float(error.abs().mean()),
                            "rmse": float(np.sqrt(np.mean(error**2))), "bias": float(error.mean()),
                            "daily_mean_mae": float(ds.mean_price_error.mean()),
                            "worst10pct_days_mae": float(ds.mae.nlargest(max(1, int(np.ceil(len(ds)*.1)))).mean()),
                            "activation_share": float(group.weight.gt(0).mean()) if name == "guarded" else None})
    return pd.DataFrame(metrics), pd.DataFrame(daily)
