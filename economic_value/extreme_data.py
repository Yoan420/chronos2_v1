"""Frozen, strictly as-of input support for the isolated direct extreme expert.

No neural prediction is manufactured for the pre-evaluation training year.
The expert learns prices/movements from fundamentals and historical labels;
the incumbent and every evaluation settlement/reference remain those of EVA.
Historical query-as-of and assumed auction times are not certified vintages.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import EconomicDataError, TARGET_PATHS, _aware, _origin, _stable_bytes, load_reference_proxy
from marginal_cost_expert.evaluation import physical_index


CORE_ZONES = ("FR", "DE", "BE", "NL")
CORE_PATHS = {
    "residual_load": "data/pit/nuclear_forecast/residual_load_market_features.parquet",
    "nuclear": "data/pit/nuclear_forecast/fr_nuclear_generation_fcst_gw.parquet",
}


def _assert_source_identity(name: str, audit: dict) -> None:
    series = audit.get("series")
    if name == "residual_load":
        expected = {f"{z.lower()}_residual_load_fcst": f"power.{z.lower()}.residual.load.hourly.gw.fcst" for z in CORE_ZONES}
        valid = isinstance(series, dict) and all(series.get(key) == value for key, value in expected.items())
    elif name == "nuclear":
        valid = series == "power.fr.generation.nuclear.gw.fcst"
    elif name == "fuels":
        valid = isinstance(series, dict) and series.get("ttf_m1_eur_mwh_th") == "gas.ttf.price.everyday.month.1.ice.eurmwh" and series.get("eua_first_dec_eur_tco2") == "carbon.eu.price.everyday.eua.ice.1st.dec"
    else:
        country = name[:2]
        field = name[3:].removesuffix("_fcst")
        if field == "temperature":
            expected = f"meteo.nrjscan.{country}.t_2m.index.fcst.d"
        elif field in {"wind_generation", "solar_generation"}:
            technology = field.removesuffix("_generation")
            expected = ("power.nl.prod.total.wind.mw.ecmwf_avg.pointconnect.6h.cache" if country == "nl" and technology == "wind"
                        else f"power.{country}.generation.{technology}.hourly.gw.fcst")
        else:
            raise EconomicDataError(f"{name}: physical source is outside the explicit feature allowlist.")
        valid = series == expected
    if not valid:
        raise EconomicDataError(f"{name}: audited source series identity differs from the physical feature allowlist.")


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return (value if value.is_absolute() else root / value).resolve()


def _pit(root: Path, name: str, path: str, columns: dict[str, str], expected: pd.DatetimeIndex,
         *, optional: bool = False) -> tuple[pd.DataFrame, dict]:
    source = _resolve(root, path)
    audit_path = Path(str(source) + ".audit.json")
    if optional and not source.is_file() and not audit_path.is_file():
        return pd.DataFrame(np.nan, index=expected, columns=list(columns.values())), {
            "name": name, "path": str(source), "status": "missing_optional_source",
            "missing_hours_by_feature": {field: len(expected) for field in columns.values()}}
    raw, evidence = _stable_bytes(source)
    audit_raw, audit_evidence = _stable_bytes(audit_path)
    upstream = json.loads(audit_raw)
    _assert_source_identity(name, upstream)
    expected_hash = upstream.get("sha256") or upstream.get("output_sha256")
    if expected_hash != evidence["sha256"]:
        raise EconomicDataError(f"{name}: PIT artifact checksum differs from its audit.")
    if upstream.get("cutoff_time") != "08:00" or upstream.get("cutoff_timezone") not in {
            "Europe/Paris", "Europe/Berlin", "Europe/Brussels", "Europe/Amsterdam"}:
        raise EconomicDataError(f"{name}: explicit 08:00 civil cutoff contract required.")
    if upstream.get("causality_violations", 0):
        raise EconomicDataError(f"{name}: upstream audit declares causality violations.")
    frame = pd.read_parquet(BytesIO(raw))
    required = {"value_time_utc", "snapshot_time_utc", "revision_time_utc", *columns}
    if name == "fuels":
        required.update({"ttf_source_value_time_utc", "eua_source_value_time_utc"})
    if required.difference(frame):
        raise EconomicDataError(f"{name}: missing PIT columns {sorted(required.difference(frame))}.")
    delivery = _aware(frame.value_time_utc, name + "/delivery")
    snapshot = _aware(frame.snapshot_time_utc, name + "/snapshot")
    revision = _aware(frame.revision_time_utc, name + "/revision")
    if not delivery.equals(delivery.floor("h")):
        raise EconomicDataError(f"{name}: only physical hourly features are supported.")
    cutoff = _origin(delivery, "Europe/Paris")
    if "cutoff_time_utc" in frame and not _aware(frame.cutoff_time_utc, name + "/cutoff").equals(cutoff):
        raise EconomicDataError(f"{name}: declared per-row cutoff differs from D-1 08:00 civil.")
    data = pd.DataFrame({"timestamp_utc": delivery, "snapshot": snapshot, "revision": revision})
    for old, new in columns.items():
        values = pd.to_numeric(frame[old], errors="raise").to_numpy(float)
        if np.isinf(values).any():
            raise EconomicDataError(f"{name}/{old}: infinite feature values.")
        data[new] = values
    keys = ["timestamp_utc", "snapshot", "revision"]
    if data.groupby(keys, dropna=False)[list(columns.values())].nunique(dropna=False).gt(1).any().any():
        raise EconomicDataError(f"{name}: conflicting PIT values for one vintage.")
    in_range = delivery.isin(expected)
    late = (snapshot > cutoff) | (revision > cutoff)
    late_observation = np.zeros(len(delivery), dtype=bool)
    if name == "fuels":
        for column in ("ttf_source_value_time_utc", "eua_source_value_time_utc"):
            late_observation |= _aware(frame[column], name + "/" + column) > cutoff
        late |= late_observation
    selected = data.loc[in_range & ~late].sort_values(keys).drop_duplicates("timestamp_utc", keep="last")
    selected = selected.set_index("timestamp_utc")[list(columns.values())].reindex(expected)
    evidence.update({"name": name, "status": "loaded", "audit": audit_evidence,
                     "series": upstream.get("series"),
                     "provider_revision_timestamp_available": upstream.get("provider_revision_timestamp_available", False),
                     "snapshot_time_semantics": upstream.get("snapshot_time_semantics"),
                     "revision_time_semantics": upstream.get("revision_time_semantics"),
                     "fill_or_interpolation": upstream.get("fill_or_interpolation"),
                     "late_vintage_rows_excluded": int((in_range & late).sum()),
                     "late_fuel_observation_rows_excluded": int((in_range & late_observation).sum()),
                     "selected_cutoff_violations": 0,
                     "missing_hours_by_feature": selected.isna().sum().to_dict(),
                     "first_source_delivery_utc": delivery.min().isoformat(),
                     "last_source_delivery_utc": delivery.max().isoformat()})
    return selected, evidence


def _target(root: Path, zone: str) -> tuple[pd.Series, dict]:
    path = _resolve(root, TARGET_PATHS[zone])
    raw, evidence = _stable_bytes(path)
    frame = pd.read_csv(BytesIO(raw), compression="gzip")
    if set(frame) != {"timestamp", "value"}:
        raise EconomicDataError(f"{zone}: unexpected canonical target schema.")
    index = _aware(frame.timestamp, zone + "/target")
    values = pd.to_numeric(frame.value, errors="raise").to_numpy(float)
    if index.has_duplicates or not index.equals(index.floor("h")) or np.isinf(values).any():
        raise EconomicDataError(f"{zone}: invalid canonical target physical identities/values.")
    return pd.Series(values, index=index), evidence


def _snapshot(root: Path, config: dict) -> tuple[pd.DataFrame, dict, list[dict]]:
    source = _resolve(root, config["source_snapshot"])
    manifest_raw, manifest_evidence = _stable_bytes(source / "manifest.json")
    manifest = json.loads(manifest_raw)
    expected = manifest.get("snapshot_files", {})
    if set(expected) != {"panel.parquet", "config.json", "data_audit.json", "reference_audit.json"}:
        raise EconomicDataError("Source EVA snapshot has an incomplete input manifest.")
    evidence = [manifest_evidence]
    panel_raw = None
    for name, sha in expected.items():
        raw, item = _stable_bytes(source / name)
        if item["sha256"] != sha:
            raise EconomicDataError(f"Source EVA snapshot checksum mismatch: {name}.")
        evidence.append(item)
        if name == "panel.parquet":
            panel_raw = raw
    assert panel_raw is not None
    panel = pd.read_parquet(BytesIO(panel_raw))
    if manifest.get("evaluation_days") != 365 or manifest.get("cutoff_time") != "08:00":
        raise EconomicDataError("Source EVA requires 365 calendar days and the strict 08:00 cutoff.")
    if config.get("baseline_model", "nuclear_kalman") not in set(panel.model):
        raise EconomicDataError("Requested incumbent is absent from the EVA snapshot.")
    return panel, manifest, evidence


def load_extreme_inputs(root: str | Path, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return 730-day research support, frozen 365-day incumbent, and lineage.

    Core feature eligibility requires all four regional residual-load forecasts,
    French nuclear generation forecast, and the previous-civil-hour proxy.
    Optional fuel/weather columns retain NaN plus missingness flags, without
    invalidating an otherwise available core. Imputation, if any, belongs only
    inside the training fold, never in this reader.
    """
    root = Path(root).resolve()
    if config.get("training_days", 365) != 365:
        raise EconomicDataError("The direct expert requires a rolling 365-day training window.")
    for option in ("include_optional_fuels", "include_optional_weather"):
        if not isinstance(config.get(option, False), bool):
            raise EconomicDataError(f"{option} must be an explicit boolean.")
    source_panel, manifest, frozen_evidence = _snapshot(root, config)
    zones = config.get("zones", manifest["zones"])
    if not isinstance(zones, list) or not zones or len(zones) != len(set(zones)) or set(zones) - set(CORE_ZONES):
        raise EconomicDataError("Choose unique supported countries for the direct expert.")
    baseline = config.get("baseline_model", "nuclear_kalman")
    evaluation = source_panel.loc[source_panel.model.eq(baseline) & source_panel.zone.isin(zones)
                                  & source_panel["sample"].eq("evaluation")].copy()
    first, last = manifest["evaluation_start"], manifest["evaluation_end"]
    if (pd.Timestamp(last) - pd.Timestamp(first)).days != 364:
        raise EconomicDataError("The source EVA calendar does not contain exactly 365 days.")
    evaluation_index = physical_index(first, last)
    evaluation["timestamp_utc"] = _aware(evaluation.timestamp_utc, "Frozen EVA")
    for zone in zones:
        block = evaluation.loc[evaluation.zone.eq(zone)].sort_values("timestamp_utc")
        if not pd.DatetimeIndex(block.timestamp_utc).equals(evaluation_index):
            raise EconomicDataError(f"{zone}: frozen incumbent has incomplete or duplicate physical support.")
    history_first = str((pd.Timestamp(first) - pd.Timedelta(days=365)).date())
    index = physical_index(history_first, last)
    local = index.tz_convert("Europe/Paris")
    source_audits = {}
    rcols = {f"{z.lower()}_residual_load_fcst": f"feature_{z.lower()}_residual_load_gw" for z in CORE_ZONES}
    residual, source_audits["residual_load"] = _pit(root, "residual_load", CORE_PATHS["residual_load"], rcols, index)
    nuclear, source_audits["nuclear"] = _pit(root, "nuclear", CORE_PATHS["nuclear"],
                                            {"value": "feature_fr_nuclear_generation_gw"}, index)
    common = pd.concat([residual, nuclear], axis=1)
    core_columns = list(common)
    common["feature_residual_region_mean_gw"] = residual.mean(axis=1, skipna=False)
    common["feature_residual_region_range_gw"] = residual.max(axis=1, skipna=False) - residual.min(axis=1, skipna=False)
    common["feature_hour_sin"] = np.sin(2 * np.pi * local.hour / 24)
    common["feature_hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
    common["feature_weekday_sin"] = np.sin(2 * np.pi * local.dayofweek / 7)
    common["feature_weekday_cos"] = np.cos(2 * np.pi * local.dayofweek / 7)
    common["feature_month_sin"] = np.sin(2 * np.pi * (local.month - 1) / 12)
    common["feature_month_cos"] = np.cos(2 * np.pi * (local.month - 1) / 12)
    if config.get("include_optional_fuels", False):
        fuels, source_audits["fuels"] = _pit(root, "fuels", "data/pit/marginal_cost_expert/fuel/market_fuel_features.parquet",
            {"ttf_m1_eur_mwh_th": "feature_ttf_m1_eur_mwh_th", "eua_first_dec_eur_tco2": "feature_eua_eur_tco2",
             "fuel_volatility_20d": "feature_fuel_volatility_20d"}, index, optional=True)
        common = pd.concat([common, fuels], axis=1)
    target_audits, output = {}, []
    for zone in zones:
        prices, target_audits[zone] = _target(root, zone)
        block = common.copy()
        block["timestamp_utc"], block["zone"] = index, zone
        block["forecast_origin_utc"] = _origin(index, "Europe/Paris")
        # Auction labels for delivery D are assumed available on D-1 at 18:00.
        # This is not a recovered exchange publication timestamp.
        label_local = local.tz_localize(None).normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=18)
        block["label_available_at_utc"] = label_local.tz_localize("Europe/Paris").tz_convert("UTC")
        block["actual"] = prices.reindex(index).to_numpy()
        block["feature_local_residual_load_gw"] = block[f"feature_{zone.lower()}_residual_load_gw"]
        block["feature_local_residual_vs_region_gw"] = block.feature_local_residual_load_gw - block.feature_residual_region_mean_gw
        if config.get("include_optional_weather", False):
            for name in ("temperature", "wind_generation", "solar_generation"):
                key = f"{zone.lower()}_{name}_fcst"
                weather, source_audits[key] = _pit(root, key, f"data/pit/kalman_weather/{key}.parquet",
                                                  {"value": f"feature_local_{name}_fcst"}, index, optional=True)
                block = pd.concat([block, weather], axis=1)
        output.append(block.reset_index(drop=True))
    history = pd.concat(output, ignore_index=True)
    references, reference_audit = load_reference_proxy(root, history)
    history = history.merge(references, on=["timestamp_utc", "zone"], how="left", validate="one_to_one")
    frozen = evaluation.set_index(["timestamp_utc", "zone"])
    history = history.set_index(["timestamp_utc", "zone"])
    exact_columns = [c for c in ("actual", "reference_price", "reference_source_timestamp_utc",
                    "reference_available_at_utc", "reference_eligible", "reference_pit_certified",
                    "availability_assumed", "reference_kind", "reference_missing_reason", "forecast_origin_utc")
                    if c in frozen]
    # Assignment, not combine_first: an EVA NaN must stay a NaN even when the
    # current canonical cache now contains a value for that physical hour.
    for column in exact_columns:
        history.loc[frozen.index, column] = frozen[column]
    history = history.reset_index()
    history["feature_previous_da_price_eur_mwh"] = history.reference_price
    core_columns += ["feature_previous_da_price_eur_mwh"]
    history["feature_eligible"] = np.isfinite(history[core_columns].to_numpy(float)).all(axis=1) & history.reference_eligible
    history["label_eligible"] = np.isfinite(history.actual)
    history["label_publication_time_assumed"] = True
    history["feature_pit_certified"] = False
    features = [c for c in history if c.startswith("feature_") and c not in {"feature_eligible", "feature_pit_certified"}]
    for column in features:
        history[column + "_missing"] = history[column].isna().astype(float)
    features += [c + "_missing" for c in features]
    added = ["timestamp_utc", "zone", "label_available_at_utc", "label_publication_time_assumed",
             "feature_eligible", "feature_pit_certified", *features]
    evaluation = evaluation.merge(history[added], on=["timestamp_utc", "zone"], how="left", validate="one_to_one")
    if len(evaluation) != len(evaluation_index) * len(zones):
        raise EconomicDataError("Frozen evaluation support changed while attaching extreme features.")
    # Check the exact comparison fields against the original sealed incumbent.
    check = evaluation.set_index(["timestamp_utc", "zone"]).sort_index()
    original = frozen.sort_index()
    for column in ("forecast", "actual", "reference_price", "benchmark_forecast", "q10", "q90"):
        if not check[column].equals(original[column]):
            raise EconomicDataError(f"Extreme input assembly changed frozen EVA {column}.")
    evidence = frozen_evidence + list(target_audits.values())
    for item in source_audits.values():
        if item.get("sha256"):
            evidence.extend([item, item["audit"]])
    evidence += list(reference_audit["sources"].values())
    for item in evidence:
        if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise EconomicDataError(f"Source changed during extreme snapshot assembly: {item['path']}")
    source_hashes = list({e["path"]: {"path": e["path"], "sha256": e["sha256"]} for e in evidence}.values())
    audit = {"schema_version": 1, "source_snapshot": str(_resolve(root, config["source_snapshot"])),
             "baseline_model": baseline, "zones": zones, "training_days": 365,
             "history_start_day": history_first, "history_end_day": last,
             "history_calendar_days": 730, "evaluation_start_day": first, "evaluation_end_day": last,
             "evaluation_days": 365, "feature_columns": features, "required_core_feature_columns": core_columns,
             "features": source_audits, "source_hashes": source_hashes,
             "missing_hours_by_feature": history[features].isna().sum().to_dict(),
             "ineligible_core_hours_by_zone": history.groupby("zone").feature_eligible.apply(lambda x: int((~x).sum())).to_dict(),
             "missing_label_hours_by_zone": history.groupby("zone").label_eligible.apply(lambda x: int((~x).sum())).to_dict(),
             "target_sources": target_audits, "reference": reference_audit,
             "evaluation_prices_forecasts_references_unchanged": True,
             "evaluation_label_policy": "exact_frozen_EVA_labels_for_evaluation_and_later_training_not_current_cache_revisions",
             "training_label_policy": "canonical_latest_snapshot_before_evaluation; frozen_EVA_within_evaluation",
             "label_available_at_utc_assumption": "delivery D auction price available D-1 18:00 Europe/Paris, not original publication evidence",
             "cutoff_time": "08:00", "selected_cutoff_violations": 0,
             "optional_missing_policy": "NaN plus indicator; any estimator imputation must be fitted inside the training fold",
             "baseline_predictions_used_as_features": False, "storm_used_as_feature": False,
             "network_features_used": False, "forecast_and_label_vintages_certified": False,
             "baseline_neural_oof_certified": False, "diagnostic_only": True, "production_modified": False}
    return (history.sort_values(["timestamp_utc", "zone"]).reset_index(drop=True),
            evaluation.sort_values(["timestamp_utc", "zone"]).reset_index(drop=True), audit)
