"""Build prospective annual CWE features from explicit, receipted source frames.

No research directory, pinned delivery date, model checkpoint or realized future
price is consulted. The two JAO matrices are independent inputs: the original
449 recipe must never be reconstructed by projecting the refreshed 503 recipe.
The materialization seal is written only after baseline and reference producers
have finished and all nine source receipts and their files have been verified.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from io import BytesIO
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from . import nyx_annual_live_preflight as gate
from . import nyx_live_hybrid as test2
from . import nyx_thermal_capacity_features as thermal
from . import nyx_lagged_exchange_features as exchange
from .nyx_annual_feature_projection import FULL, POOLED, COMPACT, project_country
from .nyx_local_price_features import build_price_features
from .nyx_local_extra_features import build_additional_features, cutoff_times
from .nyx_pooled_calendar import build_pooled_calendar_features
from .nyx_fr_hydro_lagged_features import build_fr_hydro_features
from .nyx_forecast_profile_features import build_forecast_profile_features

PROTOCOL = "nyx_annual_cpu_feature_builder_v1"
OWN_TEST2 = ("own_low_wind", "own_low_solar", "own_joint_deficit",
             "own_residual_stress", "own_deficit_stress", "own_rl_ramp_previous",
             "own_rl_ramp_next1", "own_rl_ramp_next2")
JAO_COLUMNS = tuple(name for name in gate.load_schema()["families"][POOLED]["columns"]["FR"]
                    if name.startswith("extra_jao_"))


@dataclass
class BuiltFeatures:
    features: dict[str, dict[str, pd.DataFrame]]
    base292: dict[str, pd.DataFrame]
    augmented334: dict[str, pd.DataFrame]
    test2_features: dict[str, pd.DataFrame]
    audits: dict


def require(condition, message):
    if not condition:
        raise ValueError(message)


def known_columns(schema=None):
    schema = gate.load_schema() if schema is None else schema
    return [name.removeprefix("known__") for name in
            schema["families"][POOLED]["columns"]["FR"]
            if name.startswith("known__") and not name.endswith("__available")]


def _frame(frame, index, columns, label):
    require(isinstance(frame, pd.DataFrame) and frame.columns.is_unique,
            f"{label}: frame with unique columns required")
    require(frame.index.is_unique and frame.index.is_monotonic_increasing
            and isinstance(frame.index, pd.DatetimeIndex) and str(frame.index.tz) == "UTC"
            and index.isin(frame.index).all(), f"{label}: incomplete physical UTC grid")
    require(list(frame.columns) == list(columns), f"{label}: ordered columns differ")
    result = frame.loc[index].copy()
    gate.validate_feature_frame(result, list(columns), index, label)
    return result


def build_feature_matrices(delivery_day: str, *, prices, nyx_quantiles,
                           forecast_origins, covariates, fuel, jao_original,
                           jao_refreshed, thermal_sources, hydro_features=None,
                           hydro_hourly=None, exchange_features=None,
                           exchange_hourly=None, feature_index=None,
                           reference_index=None, price_snapshots=None,
                           price_history_contract=None) -> BuiltFeatures:
    """Pure, date-independent assembly of 292 -> 334 -> 449 -> 503 -> 123.

    ``covariates`` includes the preceding 365 days needed for Test2's causal
    normalizations. ``reference_index`` normally covers 455 days plus delivery
    (90 prior OOF origins plus a 365-day HGB window). The final expert matrices
    use the trailing 365-day plus delivery grid. Source collectors may supply
    per-origin hydro/exchange features, thereby preserving captured vintages,
    or audited hourly frames to execute the exact historical pure arithmetic.
    """
    full, current, _ = gate.delivery_grid(delivery_day)
    price_history_contract = dict(price_history_contract or {})
    if price_history_contract:
        expected_contract = {"target_history_policy": gate.TARGET_HISTORY_POLICY,
            "target_revision_utc": gate.delivery_grid(delivery_day)[2].isoformat(),
            "target_origin_snapshot_verified": False, "target_future_labels_used": False}
        require(price_history_contract == expected_contract and price_snapshots is not None,
                "Current-fit price history must bind the outer cutoff and bounded internal windows")
    index = full if feature_index is None else feature_index
    schema = gate.load_schema()
    zones = set(gate.ZONES)
    require(set(prices) == set(nyx_quantiles) == set(forecast_origins) == zones,
            "Four price/NYX/origin country inputs required")
    day = pd.Timestamp(delivery_day).date()
    # Keep seven extra NYX days for past_error_d7 at the earliest HGB row.
    # Those warmup quantiles are not themselves HGB training rows.
    ref_index = (pd.date_range(pd.Timestamp(day - timedelta(days=462), tz="Europe/Paris"),
        pd.Timestamp(day + timedelta(days=1), tz="Europe/Paris"), freq="h", inclusive="left").tz_convert("UTC")
        if reference_index is None else reference_index)
    require(index.isin(ref_index).all(), "Reference features must cover expert features")
    cutoff_day = pd.Timestamp(day, tz="Europe/Paris").tz_convert("UTC")
    for zone in gate.ZONES:
        require(not (prices[zone].index >= cutoff_day).any(),
                f"{zone}: realized delivery/future labels forbidden")
        require(nyx_quantiles[zone].index.equals(nyx_quantiles["FR"].index),
                "NYX country quantile grids differ")
        require(ref_index.isin(nyx_quantiles[zone].index).all(),
                f"{zone}: NYX reference grid incomplete")
    known = known_columns(schema)
    require(covariates.columns.is_unique and set(known) <= set(covariates),
            "Four-country and ES residual Saturn covariates missing")
    source = covariates.loc[:, known]
    require(isinstance(source.index, pd.DatetimeIndex) and str(source.index.tz) == "UTC"
            and source.index.is_unique and source.index.is_monotonic_increasing
            and ref_index.isin(source.index).all()
            and np.isfinite(source.to_numpy(dtype=float)).all(),
            "Complete finite UTC covariates required")
    # A forecast origin applies to every known source row. Earlier covariates
    # retain their own daily as-of origin; no current vintage is backdated here.
    origins = {zone: forecast_origins[zone].copy() for zone in gate.ZONES}
    for zone in gate.ZONES:
        require(source.index.isin(origins[zone].index).all(),
                f"{zone}: covariate origins missing (including normalization history)")
        stamps = pd.to_datetime(origins[zone].loc[source.index], utc=True)
        limits = cutoff_times(source.index, "Europe/Paris")
        require(stamps.notna().all() and (stamps <= limits).all(),
                f"{zone}: covariate origin exceeds that day's D-1 08:00")
    pairs, pair_audits = {}, {}
    for pair in test2.PAIRS:
        values, audit = test2.build_pair_features({zone: source for zone in pair}, pair)
        pairs.update(values)
        pair_audits["_".join(pair)] = audit
    test2_out = {zone: pairs[zone].loc[ref_index].copy() for zone in gate.ZONES}
    revised_days = {}
    if price_snapshots is not None:
        for origin_day in dict.fromkeys(ref_index.tz_convert("Europe/Paris").date):
            snapshot = price_snapshots(str(origin_day))
            require(set(snapshot) == zones, f"{origin_day}: four as-of auction snapshots required")
            start = pd.Timestamp(origin_day - timedelta(days=7), tz="Europe/Paris").tz_convert("UTC")
            stop = pd.Timestamp(origin_day, tz="Europe/Paris").tz_convert("UTC")
            wanted = pd.date_range(start, stop, freq="h", inclusive="left")
            changed = False
            bounded = {}
            for zone in gate.ZONES:
                require(not (snapshot[zone].index >= stop).any(),
                        f"{origin_day}/{zone}: snapshot contains current/future labels")
                bounded[zone] = snapshot[zone].loc[(snapshot[zone].index >= start)
                                                  & (snapshot[zone].index < stop)]
                before = prices[zone].reindex(wanted).to_numpy(dtype=float)
                after = bounded[zone].reindex(wanted).to_numpy(dtype=float)
                changed |= not np.array_equal(before, after, equal_nan=True)
            if changed:
                revised_days[origin_day] = bounded
    base, augmented, country_audits = {}, {}, {}
    previous = {zone: nyx_quantiles[zone]["q50"].rename("nyx_q50") for zone in gate.ZONES}
    for zone in gate.ZONES:
        frame, price_audit = build_price_features(prices, zone=zone,
            delivery_index=ref_index, nyx_predictions=nyx_quantiles[zone],
            past_predictions=previous, known_covariates=source,
            known_columns=known, known_delta_columns=known,
            forecast_origins=origins[zone])
        for origin_day, snapshot in revised_days.items():
            day_index = ref_index[ref_index.tz_convert("Europe/Paris").date == origin_day]
            left = pd.Timestamp(origin_day - timedelta(days=7), tz="Europe/Paris").tz_convert("UTC")
            right = pd.Timestamp(origin_day + timedelta(days=1), tz="Europe/Paris").tz_convert("UTC")
            short_cov = source.loc[(source.index >= left) & (source.index < right)]
            short_previous = {key: values.loc[(values.index >= left) & (values.index < right)]
                              for key, values in previous.items()}
            changed, changed_audit = build_price_features(snapshot, zone=zone,
                delivery_index=day_index, nyx_predictions=nyx_quantiles[zone].loc[day_index],
                past_predictions=short_previous, known_covariates=short_cov,
                known_columns=known, known_delta_columns=known, forecast_origins=origins[zone])
            frame.loc[day_index] = changed
            price_audit["daily_sources"] = [row if row["delivery_day"] != str(origin_day)
                else changed_audit["daily_sources"][0] for row in price_audit["daily_sources"]]
        price_audit["per_origin_price_snapshots_verified"] = price_snapshots is not None and not price_history_contract
        price_audit["internal_price_windows_verified"] = price_snapshots is not None
        price_audit.update(price_history_contract)
        price_audit["revised_price_days_rebuilt"] = [str(value) for value in revised_days]
        require(list(frame.columns) == schema["families"][POOLED]["columns"][zone][:292],
                f"{zone}: base292 recipe changed")
        additional, extra_audit = build_additional_features(zone=zone,
            delivery_index=ref_index, nyx_forecasts=nyx_quantiles,
            forecast_origins=origins, fuel=fuel)
        base[zone] = frame
        augmented[zone] = pd.concat([frame, additional], axis=1)
        require(list(augmented[zone]) == schema["families"][POOLED]["columns"][zone][:334],
                f"{zone}: augmented334 recipe changed")
        country_audits[zone] = {"price": price_audit, "additional": extra_audit}
    require((hydro_features is None) != (hydro_hourly is None),
            "Supply exactly one per-origin hydro feature frame or hourly hydro source")
    if hydro_features is None:
        hydro_features, hydro_audit = build_fr_hydro_features(hydro_hourly, index)
        hydro_features = hydro_features.rename(columns=lambda name: "extra_hydro_" + name)
        for name in tuple(hydro_features.columns):
            hydro_features[name + "__available"] = hydro_features[name].notna().astype(float)
        hydro_audit = {key: value for key, value in hydro_audit.items() if key != "rows"}
    else:
        hydro_audit = {"per_origin_features": True, "source_receipt_owned_by_caller": True}
    hydro_columns = [name for name in schema["families"][POOLED]["columns"]["FR"]
                     if name.startswith("extra_hydro_")]
    hydro = _frame(hydro_features, index, hydro_columns, "hydro")
    original = _frame(jao_original, index, JAO_COLUMNS, "original JAO")
    refreshed = _frame(jao_refreshed, index, JAO_COLUMNS, "refreshed JAO")
    if all(list(frame) == ["pmax_gw", "asof_query_utc"] for frame in thermal_sources.values()):
        from run_nyx_annual_thermal_source import build_features as prospective_thermal
        for name, frame in thermal_sources.items():
            require(pd.to_datetime(frame.asof_query_utc, utc=True).equals(
                cutoff_times(frame.index, "Europe/Paris").rename("asof_query_utc")),
                f"{name}: thermal origin differs from its own daily cutoff")
            groups = frame.pmax_gw.groupby(frame.index.tz_convert("Europe/Paris").date)
            require((groups.nunique(dropna=False) == 1).all(),
                    f"{name}: daily thermal capacity cannot change within its day")
        capacities = prospective_thermal(thermal_sources, delivery_day)
        capacities = {zone: frame.loc[index] for zone, frame in capacities.items()}
    else:
        capacities = thermal.build_features(thermal_sources, index)
    require((exchange_features is None) != (exchange_hourly is None),
            "Supply exactly one per-origin exchange feature frame or hourly source")
    if exchange_features is None:
        exchanges, exchange_audit = exchange.build_features(exchange_hourly, index)
        exchange_audit = {key: value for key, value in exchange_audit.items() if key != "rows"}
    else:
        selected = _frame(exchange_features, index, exchange.COLUMNS, "exchange")
        exchanges = {zone: selected for zone in gate.ZONES}
        exchange_audit = {"per_origin_features": True, "source_receipt_owned_by_caller": True}
    features = {family: {} for family in gate.FAMILIES}
    for zone in gate.ZONES:
        common = augmented[zone].loc[index]
        calendar, cal_audit = build_pooled_calendar_features(zone=zone, delivery_index=index)
        profiles, profiles_audit = build_forecast_profile_features(common)
        # Assemble both branches explicitly. Only the JAO block differs.
        before, after = [common, calendar, hydro], [profiles]
        pooled = pd.concat([*before, original, *after], axis=1)
        own = pairs[zone].loc[index, list(OWN_TEST2)].rename(columns=lambda name: "canon__" + name)
        full503 = pd.concat([*before, refreshed, *after, own, capacities[zone], exchanges[zone]], axis=1)
        compact = project_country(full503, pooled, zone=zone, expected_index=index, schema=schema)
        features[POOLED][zone], features[FULL][zone], features[COMPACT][zone] = pooled, full503, compact
        country_audits[zone].update(calendar=cal_audit, profiles=profiles_audit)
    return BuiltFeatures(features, base, augmented, test2_out,
        {"protocol": PROTOCOL, "delivery_day": delivery_day,
         "future_labels_used": False, "storm_used_as_model_input": False,
         "jao_branches_built_independently": True, "hydro": hydro_audit,
         "exchange": exchange_audit, "test2": pair_audits, "countries": country_audits})


def _write_immutable(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == content, f"Existing materialization differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        # Windows rename refuses an existing destination, avoiding overwrites.
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_frame(path, frame):
    buffer = BytesIO()
    frame.to_parquet(buffer)
    _write_immutable(path, buffer.getvalue())


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def source_graph(bundle: Path, delivery_day: str, groups=gate.SOURCE_GROUPS, *,
                 allow_training_bootstrap=False):
    _, _, cutoff = gate.delivery_grid(delivery_day)
    receipts, artifacts = {}, {}
    for group in groups:
        path = bundle / f"source_receipts/{group}.json"
        require(path.is_file(), f"{group}: source receipt missing")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        gate.validate_source_receipt(receipt, group=group, day=delivery_day,
                                     bundle=bundle, cutoff=cutoff,
                                     allow_training_bootstrap=allow_training_bootstrap)
        receipts[group] = gate.sha256(path)
        for name, digest in receipt["artifact_sha256"].items():
            require(name not in artifacts or artifacts[name] == digest,
                    f"Conflicting source checksum: {name}")
            artifacts[name] = digest
    return receipts, artifacts


def materialize_features(bundle: Path, delivery_day: str) -> BuiltFeatures:
    """Load bound normalized source artifacts and write 12 expert matrices.

    Baseline production precedes this stage; reference production follows it.
    ``seal_bundle`` is deliberately separate to avoid circular source receipts.
    """
    bundle = Path(bundle).resolve()
    from .nyx_annual_saturn_source import load_target_snapshots, target_history_contract
    groups = tuple(group for group in gate.SOURCE_GROUPS if group != "scarcity_confirmed_pair")
    receipts, artifacts = source_graph(bundle, delivery_day, groups, allow_training_bootstrap=True)

    def read(relative):
        require(relative in artifacts, f"Unreceipted feature source: {relative}")
        return pd.read_parquet(bundle / relative)

    covariates = read("source_artifacts/saturn/covariates.parquet")
    # The Saturn collector records origins separately, not inferred from a
    # feature delivery timestamp or backdated from the current run.
    source_origins = read("source_artifacts/saturn/origins.parquet")
    require("forecast_origin_utc" in source_origins, "Saturn origin column missing")
    quantiles, origins = {}, {}
    for zone in gate.ZONES:
        frame = read(f"sources/nyx_quantiles/{zone}.parquet")
        require({"q10", "q50", "q90", "forecast_origin_utc"} <= set(frame),
                f"{zone}: baseline quantiles and origins missing")
        quantiles[zone] = frame[["q10", "q50", "q90"]]
        combined = source_origins.forecast_origin_utc.copy()
        require(frame.index.isin(combined.index).all(), f"{zone}: missing Saturn origin rows")
        nyx_origins = pd.to_datetime(frame.forecast_origin_utc, utc=True)
        cov_origins = pd.to_datetime(combined.loc[frame.index], utc=True)
        # Shared origins must bound both independently supplied forecasters.
        combined.loc[frame.index] = nyx_origins.where(nyx_origins >= cov_origins, cov_origins)
        origins[zone] = combined
    prices = {zone: read(f"source_artifacts/auction_prices/{zone}.parquet").price_eur_mwh
              for zone in gate.ZONES}
    jao_path = "source_artifacts/jao_initial/features_365d_plus_delivery.parquet"
    original_path = "source_artifacts/jao_initial/original_features.parquet"
    refreshed_path = "source_artifacts/jao_initial/refreshed_features.parquet"
    if original_path in artifacts or refreshed_path in artifacts:
        original, refreshed = read(original_path), read(refreshed_path)
    else:
        # A prospective raw capture has one verified initial-computation
        # vintage for the day. Use it directly in both explicit assemblies;
        # historical archive import must instead supply its distinct views.
        jao_receipt = json.loads((bundle / "source_receipts/jao_initial.json").read_text(encoding="utf-8"))
        require(jao_receipt.get("actual_pre_cutoff_capture_verified") is True,
                "JAO source must attest prospective captures or supply original/refreshed views")
        original = read(jao_path)
        refreshed = original.copy(deep=True)
    built = build_feature_matrices(delivery_day, prices=prices, nyx_quantiles=quantiles,
        forecast_origins=origins, covariates=covariates,
        fuel=read("source_artifacts/fuel/market_fuel_features.parquet"),
        hydro_features=read("source_artifacts/public_hydro/features.parquet"),
        jao_original=original, jao_refreshed=refreshed,
        thermal_sources={name: read(f"source_artifacts/thermal_capacity/{name}.parquet")
                         for name in thermal.SOURCES},
        exchange_features=read("source_artifacts/lagged_exchange/features.parquet"),
        price_snapshots=load_target_snapshots(bundle),
        price_history_contract=target_history_contract(bundle))
    require(source_graph(bundle, delivery_day, groups, allow_training_bootstrap=True) == (receipts, artifacts),
            "Source snapshots changed while building features")
    for family, countries in built.features.items():
        for zone, frame in countries.items():
            _write_frame(bundle / f"features/{family}/{zone}.parquet", frame)
    for label, countries in (("base292", built.base292), ("augmented334", built.augmented334),
                              ("test2", built.test2_features)):
        for zone, frame in countries.items():
            _write_frame(bundle / f"reference_inputs/{label}/{zone}.parquet", frame)
    _write_immutable(bundle / "feature_build_audit.json", _json(built.audits))
    return built


def seal_bundle(bundle: Path, delivery_day: str) -> dict:
    """Bind verified sources, complete derived matrices and forecast references."""
    bundle = Path(bundle).resolve()
    receipts, artifacts = source_graph(bundle, delivery_day, allow_training_bootstrap=True)
    full, current, cutoff = gate.delivery_grid(delivery_day)
    schema = gate.load_schema()
    for family in gate.FAMILIES:
        for zone in gate.ZONES:
            relative = f"features/{family}/{zone}.parquet"
            gate.validate_feature_frame(pd.read_parquet(bundle / relative),
                schema["families"][family]["columns"][zone], full, relative)
    for zone in gate.ZONES:
        gate.validate_baseline(pd.read_parquet(bundle / f"baseline/{zone}.parquet"),
                               full, current, cutoff, f"baseline/{zone}")
    for relative in gate.materialized_outputs():
        if relative.startswith("reference/"):
            gate.validate_reference(pd.read_parquet(bundle / relative), current, cutoff, relative)
    manifest = {"protocol": gate.MATERIALIZATION_PROTOCOL, "state": "COMPLETE",
        "delivery_day": delivery_day, "asof_cutoff_utc": cutoff.isoformat(),
        "deterministic_transform": True, "future_labels_used": False,
        "storm_used_as_model_input": False, "schema_sha256": gate.sha256(gate.SCHEMA),
        "parameters": {"builder_protocol": PROTOCOL, "training_days": 365,
                       "jao_branches_built_independently": True},
        "source_receipts_sha256": receipts, "source_artifacts_sha256": artifacts,
        "output_sha256": {name: gate.sha256(bundle / name) for name in gate.materialized_outputs()},
        "transform_code_sha256": {name: gate.sha256(gate.ROOT / name)
                                  for name in gate.MATERIALIZER_CODE}}
    require(source_graph(bundle, delivery_day, allow_training_bootstrap=True) == (receipts, artifacts),
            "Sources changed while sealing materialization")
    _write_immutable(bundle / gate.MATERIALIZATION_PATH, _json(manifest))
    gate.validate_materialization_manifest(bundle, delivery_day, allow_training_bootstrap=True)
    return manifest
