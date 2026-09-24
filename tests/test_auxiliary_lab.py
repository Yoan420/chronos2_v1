from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from auxiliary_lab.config import AuxiliaryLabConfigError, load_lab_config
from auxiliary_lab.data import _issued_history, load_kalman_dataset, load_residual_dataset
from auxiliary_lab.metrics import validate_and_split_days
from auxiliary_lab.runner import (
    AuxiliaryLabError,
    _prequential_residual_history,
    compare_runs,
    evaluate_run,
    predict_artifact,
    train_experiment,
)


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _source_run(root: Path, *, days: int = 45) -> Path:
    source = root / "runs" / "source"
    inputs = source / "inputs"
    inputs.mkdir(parents=True)
    start = pd.Timestamp("2026-01-01", tz="Europe/Paris")
    end = start + pd.DateOffset(days=days)
    index = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
    position = np.arange(len(index), dtype=float)
    base_q50 = 55.0 + 8.0 * np.sin(2.0 * np.pi * position / 24.0)
    actual = base_q50 + 2.0 * np.cos(2.0 * np.pi * position / (24.0 * 7.0))
    primary = actual + 0.5 * np.sin(2.0 * np.pi * position / 48.0)
    forecast_origins = pd.DatetimeIndex(
        [
            (
                pd.Timestamp(day)
                - pd.Timedelta(days=1)
                + pd.Timedelta(hours=8)
            )
            .tz_localize("Europe/Paris")
            .tz_convert("UTC")
            for day in index.tz_convert("Europe/Paris").date
        ]
    )
    history = pd.DataFrame(
        {
            "delivery_start_utc": index,
            "forecast_origin_utc": forecast_origins,
            "chronos2__q10": base_q50 - 10.0,
            "chronos2__q50": base_q50,
            "chronos2__q90": base_q50 + 10.0,
            "residual_corrected__q10": base_q50 - 9.0,
            "residual_corrected__q50": base_q50 + 0.5,
            "residual_corrected__q90": base_q50 + 10.5,
            "residual_correction": 0.5,
            "mkonline_primary__q50": primary,
            "actual": actual,
        }
    )
    history.to_csv(source / "backtest_hourly_oof.csv.gz", index=False, compression="gzip")
    history.to_csv(source / "statistics_history_hourly.csv.gz", index=False, compression="gzip")
    covariates = pd.DataFrame({"timestamp": index})
    aliases = (
        "fr_residual_load_fcst",
        "de_residual_load_fcst",
        "be_residual_load_fcst",
        "nl_residual_load_fcst",
        "es_residual_load_fcst",
    )
    for offset, alias in enumerate(aliases):
        covariates[alias] = 30.0 + offset + np.sin(2.0 * np.pi * position / 24.0)
    aligned = covariates.copy()
    aligned["target"] = actual
    aligned.to_csv(inputs / "aligned_inputs.csv.gz", index=False, compression="gzip")

    future_start = end
    future_end = future_start + pd.DateOffset(days=1)
    future = pd.date_range(future_start, future_end, freq="h", inclusive="left").tz_convert("UTC")
    future_position = np.arange(len(future), dtype=float) + len(index)
    future_covariates = pd.DataFrame({"timestamp": future})
    for offset, alias in enumerate(aliases):
        future_covariates[alias] = 30.0 + offset + np.sin(
            2.0 * np.pi * future_position / 24.0
        )
    pd.concat([covariates, future_covariates], ignore_index=True).to_csv(
        inputs / "model_covariates_with_future.csv.gz",
        index=False,
        compression="gzip",
    )
    future_base = 55.0 + 8.0 * np.sin(2.0 * np.pi * future_position / 24.0)
    pd.DataFrame(
        {
            "delivery_start_utc": future,
            "chronos2__q10": future_base - 10.0,
            "chronos2__q50": future_base,
            "chronos2__q90": future_base + 10.0,
            "residual_corrected__q10": future_base - 9.0,
            "residual_corrected__q50": future_base + 0.5,
            "residual_corrected__q90": future_base + 10.5,
            "residual_correction": 0.5,
            "mkonline_primary__q50": future_base + 1.0,
        }
    ).to_csv(source / "forecast_hourly_fr.csv", index=False)
    (source / "run_manifest.json").write_text(
        json.dumps({"zone": "FR", "timezone": "Europe/Paris"}), encoding="utf-8"
    )
    return source


def _config(root: Path, source: Path, *, report: bool = True) -> Path:
    path = root / "lab.yaml"
    payload = {
        "schema_version": 1,
        "experiment_id": "synthetic_aux_lab",
        "source_run": str(source.relative_to(root)),
        "output_directory": "runs/experiments/auxiliary_lab/synthetic_aux_lab",
        "timezone": "Europe/Paris",
        "split": {
            "validation_days": 7,
            "test_days": 7,
            "minimum_training_days": 14,
            "horizon_hours": None,
        },
        "objective": "mae",
        "metrics": [
            "mae",
            "rmse",
            "bias",
            "pinball_q10",
            "pinball_q50",
            "pinball_q90",
            "coverage80",
            "interval_width80",
        ],
        "random_seed": 7,
        "report": {"enabled": report, "embed_plotly": False},
        "models": {
            "residual_corrector": {
                "enabled": True,
                "base_model": "chronos2",
                "expert_models": ["chronos2"],
                "recipe": "blend",
                "components": {
                    "hgb_a": {
                        "backend": "sklearn",
                        "iterations": 10,
                        "depth": 2,
                        "learning_rate": 0.08,
                        "l2_leaf_reg": 1.0,
                        "min_samples_leaf": 8,
                        "sklearn_early_stopping": False,
                    },
                    "hgb_b": {
                        "backend": "sklearn",
                        "iterations": 12,
                        "depth": 3,
                        "learning_rate": 0.05,
                        "l2_leaf_reg": 2.0,
                        "min_samples_leaf": 10,
                        "sklearn_early_stopping": False,
                    },
                },
                "weight_candidates": [
                    {"hgb_a": 0.5, "hgb_b": 0.5},
                    {"hgb_a": 0.7, "hgb_b": 0.3},
                ],
                "blend_max_abs_correction": 20.0,
                "fixed_parameters": {
                    "min_training_rows": 48,
                    "max_abs_correction": None,
                    "feature_builder_options": {
                        "timezone": "Europe/Paris",
                        "include_rich_calendar": False,
                        "include_daily_profiles": False,
                    },
                },
                "parameter_grid": {
                    "components.hgb_a.learning_rate": [0.06, 0.08]
                },
            },
            "mkonline_blend": {
                "enabled": True,
                "autonomous_model": "residual_corrected",
                "primary_column": "mkonline_primary__q50",
                "weight_step": 0.1,
                "fixed_parameters": {},
                "parameter_grid": {"max_abs_shift_eur_mwh": [None, 5.0]},
            },
            "kalman": {
                "enabled": True,
                "upstream_model": "residual_corrected",
                "fixed_parameters": {
                    "candidate_kinds": ["linear_bias"],
                    "governance_lookback_days": 5,
                    "governance_minimum_days": 2,
                    "governance_weight_step": 0.25,
                    "minimum_gain_eur_mwh": 0.01,
                    "minimum_relative_gain": 0.0,
                },
                "parameter_grid": {"q_over_r": [0.001, 0.002]},
            },
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _enable_weather_covariates(
    root: Path,
    source: Path,
    config_path: Path,
    *,
    minimum_coverage: float = 0.0,
) -> Path:
    base = pd.read_csv(
        source / "inputs" / "model_covariates_with_future.csv.gz"
    )
    position = np.arange(len(base), dtype=float)
    delivery = pd.to_datetime(base["timestamp"], utc=True)
    local_days = delivery.dt.tz_convert("Europe/Paris").dt.normalize().dt.tz_localize(None)
    cutoffs = pd.Series(
        [
            (day - pd.Timedelta(days=1) + pd.Timedelta(hours=8))
            .tz_localize("Europe/Paris")
            .tz_convert("UTC")
            .isoformat()
            for day in local_days
        ]
    )
    weather = pd.DataFrame(
        {
            "valid_time": base["timestamp"],
            "run_init_utc": pd.to_datetime(cutoffs, utc=True)
            .sub(pd.Timedelta(hours=6))
            .map(pd.Timestamp.isoformat),
            "cutoff_utc": cutoffs,
            "temperature_c": 12.0 + 5.0 * np.sin(2.0 * np.pi * position / 24.0),
            "wind_generation_gw": 8.0 + np.cos(2.0 * np.pi * position / 24.0),
        }
    )
    source_path = root / "data" / "pit" / "synthetic_weather.csv.gz"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(source_path, index=False, compression="gzip")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_aliases = [
        "fr_residual_load_fcst",
        "de_residual_load_fcst",
        "be_residual_load_fcst",
        "nl_residual_load_fcst",
        "es_residual_load_fcst",
    ]
    kalman = payload["models"]["kalman"]
    kalman["covariates"] = {
        "input_columns": [
            *base_aliases,
            "fr_temperature_fcst",
            "fr_wind_generation_fcst",
        ],
        "groups": {
            "market": base_aliases,
            "weather": [
                "fr_temperature_fcst",
                "fr_temperature_ramp",
                "fr_heating_degree",
            ],
            "renewables": [
                "fr_wind_generation_fcst",
                "fr_wind_ramp",
            ],
            "fundamentals": [
                "fr_temperature_fcst",
                "fr_wind_generation_fcst",
            ],
        },
        "derived": {
            "fr_temperature_ramp": {
                "kind": "ramp",
                "source": "fr_temperature_fcst",
                "periods": 1,
            },
            "fr_heating_degree": {
                "kind": "heating_degree",
                "source": "fr_temperature_fcst",
                "threshold": 15.0,
            },
            "fr_wind_ramp": {
                "kind": "ramp",
                "source": "fr_wind_generation_fcst",
                "periods": 1,
            },
        },
        "history_missing_policy": "neutral",
        "minimum_history_coverage": minimum_coverage,
        "require_future_complete": True,
    }
    kalman["additional_sources"] = [
        {
            "path": str(source_path.relative_to(root)),
            "timestamp_column": "valid_time",
            "origin_column": "run_init_utc",
            "cutoff_column": "cutoff_utc",
            "cutoff_time": "08:00",
            # Direction partagee lab/live: alias modele -> colonne fichier.
            "columns": {
                "fr_temperature_fcst": "temperature_c",
                "fr_wind_generation_fcst": "wind_generation_gw",
            },
        }
    ]
    kalman["fixed_parameters"]["candidate_kinds"] = [
        "linear_bias",
        "linear_weather",
        "linear_renewables",
        "linear_fundamental",
    ]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return source_path


def _add_residual_history_prefix(
    root: Path,
    source: Path,
    config_path: Path,
    *,
    days: int = 3,
) -> Path:
    existing = pd.read_csv(source / "inputs" / "aligned_inputs.csv.gz")
    first = pd.to_datetime(existing["timestamp"], utc=True).min().tz_convert(
        "Europe/Paris"
    )
    prefix_index = pd.date_range(
        first - pd.Timedelta(days=days),
        first,
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    position = np.arange(len(prefix_index), dtype=float)
    q50 = 48.0 + np.sin(2.0 * np.pi * position / 24.0)
    actual = q50 + 0.25
    origins = pd.DatetimeIndex(
        [
            (
                pd.Timestamp(day)
                - pd.Timedelta(days=1)
                + pd.Timedelta(hours=8)
            )
            .tz_localize("Europe/Paris")
            .tz_convert("UTC")
            for day in prefix_index.tz_convert("Europe/Paris").date
        ]
    )
    prefix = pd.DataFrame(
        {
            "delivery_start_utc": prefix_index,
            "forecast_origin_utc": origins,
            "q10": q50 - 8.0,
            "q50": q50,
            "q90": q50 + 8.0,
            "actual": actual,
        }
    )
    prefix_path = root / "runs" / "prefix" / "chronos_oof_prefix.csv.gz"
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    prefix.to_csv(prefix_path, index=False, compression="gzip")

    prefix_aligned = pd.DataFrame(
        {"timestamp": prefix_index, "target": actual}
    )
    aliases = (
        "fr_residual_load_fcst",
        "de_residual_load_fcst",
        "be_residual_load_fcst",
        "nl_residual_load_fcst",
        "es_residual_load_fcst",
    )
    for offset, alias in enumerate(aliases):
        prefix_aligned[alias] = 25.0 + offset + np.sin(
            2.0 * np.pi * position / 24.0
        )
    pd.concat([prefix_aligned, existing], ignore_index=True).to_csv(
        source / "inputs" / "aligned_inputs.csv.gz",
        index=False,
        compression="gzip",
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["models"]["residual_corrector"]["prequential_bridge"] = {
        "refit_cadence_days": 28,
        "training_lookback_days": 365,
        "cold_start_policy": "identity",
        "history_prefix_path": str(prefix_path.relative_to(root)),
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return prefix_path


def test_config_rejects_competing_forecast_as_autonomous_input(tmp_path: Path) -> None:
    source = _source_run(tmp_path)
    config_path = _config(tmp_path, source)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["models"]["residual_corrector"]["expert_models"] = ["storm_dashboard"]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(AuxiliaryLabConfigError, match="interdits"):
        load_lab_config(config_path, project_root=tmp_path)


def test_config_rejects_unknown_hyperparameter_before_training(tmp_path: Path) -> None:
    source = _source_run(tmp_path)
    config_path = _config(tmp_path, source)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["models"]["kalman"]["parameter_grid"] = {"q_over_rr": [0.001]}
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(AuxiliaryLabConfigError, match="Kalman inconnus"):
        load_lab_config(config_path, project_root=tmp_path)


def test_kalman_rolling_worker_count_is_bounded(tmp_path: Path) -> None:
    source = _source_run(tmp_path)
    config_path = _config(tmp_path, source)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["models"]["kalman"]["rolling_refit_workers"] = 4
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    config = load_lab_config(config_path, project_root=tmp_path)
    assert config.models["kalman"].options["rolling_refit_workers"] == 4

    payload["models"]["kalman"]["rolling_refit_workers"] = 9
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(AuxiliaryLabConfigError, match="rolling_refit_workers"):
        load_lab_config(config_path, project_root=tmp_path)


def test_prequential_bridge_config_is_strict_and_defaults_to_rolling365(
    tmp_path: Path,
) -> None:
    source = _source_run(tmp_path)
    config_path = _config(tmp_path, source)
    config = load_lab_config(config_path, project_root=tmp_path)

    assert config.models["residual_corrector"].options["prequential_bridge"] == {
        "refit_cadence_days": 28,
        "training_lookback_days": 365,
        "cold_start_policy": "identity",
        "history_prefix_path": None,
    }

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["models"]["residual_corrector"]["prequential_bridge"] = {
        "refit_cadence_days": 0,
        "training_lookback_days": 365,
        "cold_start_policy": "identity",
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(AuxiliaryLabConfigError, match="refit_cadence_days"):
        load_lab_config(config_path, project_root=tmp_path)


def test_residual_history_prefix_extends_the_causal_training_timeline(
    tmp_path: Path,
) -> None:
    source = _source_run(tmp_path)
    config_path = _config(tmp_path, source)
    prefix_path = _add_residual_history_prefix(
        tmp_path, source, config_path, days=3
    )
    config = load_lab_config(config_path, project_root=tmp_path)

    dataset = load_residual_dataset(
        config, config.models["residual_corrector"]
    )

    local_days = pd.Index(dataset.X.index.tz_convert(config.timezone).date)
    assert local_days.nunique() == 48
    assert local_days.min() == pd.Timestamp("2025-12-29").date()
    assert config.models["residual_corrector"].options[
        "prequential_bridge"
    ]["history_prefix_path"] == prefix_path.resolve()


def test_residual_history_prefix_rejects_a_noncontractual_origin(
    tmp_path: Path,
) -> None:
    source = _source_run(tmp_path)
    config_path = _config(tmp_path, source)
    prefix_path = _add_residual_history_prefix(
        tmp_path, source, config_path, days=3
    )
    prefix = pd.read_csv(prefix_path)
    prefix.loc[0, "forecast_origin_utc"] = (
        pd.Timestamp(prefix.loc[0, "forecast_origin_utc"])
        - pd.Timedelta(hours=1)
    ).isoformat()
    prefix.to_csv(prefix_path, index=False, compression="gzip")
    config = load_lab_config(config_path, project_root=tmp_path)

    with pytest.raises(ValueError, match="different du cutoff civil"):
        load_residual_dataset(config, config.models["residual_corrector"])


def test_published_residual_overlay_rejects_a_late_forecast_origin(
    tmp_path: Path,
) -> None:
    source = _source_run(tmp_path)
    config = load_lab_config(_config(tmp_path, source), project_root=tmp_path)
    model_config = config.models["residual_corrector"]
    dataset = load_residual_dataset(config, model_config)
    issued = _issued_history(source)
    first_validation = dataset.split.validation_days[0]
    local_days = pd.Index(issued.index.tz_convert(config.timezone).date)
    invalid = np.asarray(local_days == first_validation, dtype=bool)
    issued.loc[invalid, "forecast_origin_utc"] = (
        pd.Timestamp(first_validation)
        .tz_localize(config.timezone)
        .tz_convert("UTC")
        .isoformat()
    )

    with pytest.raises(AuxiliaryLabError, match="posterieur au cutoff"):
        _prequential_residual_history(
            dataset=dataset,
            model_config=model_config,
            selected_params={
                **model_config.fixed_parameters,
                "components.hgb_a.learning_rate": 0.06,
            },
            selected_weights=model_config.options["weight_candidates"][0],
            issued_history=issued,
            timezone_name=config.timezone,
            seed=config.random_seed,
        )


def test_prequential_residual_bridge_never_reads_day_or_future_labels(
    tmp_path: Path,
) -> None:
    source = _source_run(tmp_path)
    config = load_lab_config(_config(tmp_path, source), project_root=tmp_path)
    model_config = config.models["residual_corrector"]
    dataset = load_residual_dataset(config, model_config)
    params = {
        **model_config.fixed_parameters,
        "components.hgb_a.learning_rate": 0.06,
    }
    weights = model_config.options["weight_candidates"][0]
    issued_without_residual = _issued_history(source).drop(
        columns=[f"residual_corrected__{q}" for q in ("q10", "q50", "q90")]
    )

    original, audit, summary = _prequential_residual_history(
        dataset=dataset,
        model_config=model_config,
        selected_params=params,
        selected_weights=weights,
        issued_history=issued_without_residual,
        timezone_name=config.timezone,
        seed=config.random_seed,
    )
    fitted_audit = audit.loc[audit["generation_source"].eq("prequential_refit")]
    assert not fitted_audit.empty
    assert (
        pd.to_datetime(fitted_audit["fit_end_day"])
        < pd.to_datetime(fitted_audit["block_start_day"])
    ).all()
    assert summary["causality_violations"] == 0

    chosen = fitted_audit.iloc[len(fitted_audit) // 2]
    changed_from = pd.Timestamp(chosen["delivery_day"]).date()
    changed_actual = dataset.actual.copy()
    local_days = pd.Index(changed_actual.index.tz_convert(config.timezone).date)
    changed_actual.loc[np.asarray(local_days >= changed_from, dtype=bool)] += 10_000.0
    changed, changed_audit, _ = _prequential_residual_history(
        dataset=replace(dataset, actual=changed_actual),
        model_config=model_config,
        selected_params=params,
        selected_weights=weights,
        issued_history=issued_without_residual,
        timezone_name=config.timezone,
        seed=config.random_seed,
    )

    block_start = str(chosen["block_start_day"])
    protected_days = set(
        audit.loc[audit["block_start_day"].eq(block_start), "delivery_day"]
    )
    protected = np.asarray(
        pd.Index(original.index.tz_convert(config.timezone).date).astype(str).isin(
            protected_days
        ),
        dtype=bool,
    )
    forecast_columns = [
        "residual_corrected__q10",
        "residual_corrected__q50",
        "residual_corrected__q90",
        "residual_correction",
    ]
    pd.testing.assert_frame_equal(
        original.loc[protected, forecast_columns],
        changed.loc[protected, forecast_columns],
    )
    pd.testing.assert_series_equal(
        audit["fit_end_day"], changed_audit["fit_end_day"], check_names=False
    )


def test_prequential_bridge_overlays_only_complete_issued_evaluation_days(
    tmp_path: Path,
) -> None:
    source = _source_run(tmp_path)
    config = load_lab_config(_config(tmp_path, source), project_root=tmp_path)
    model_config = config.models["residual_corrector"]
    dataset = load_residual_dataset(config, model_config)
    params = {
        **model_config.fixed_parameters,
        "components.hgb_a.learning_rate": 0.06,
    }
    issued = _issued_history(source)

    history, audit, summary = _prequential_residual_history(
        dataset=dataset,
        model_config=model_config,
        selected_params=params,
        selected_weights=model_config.options["weight_candidates"][0],
        issued_history=issued,
        timezone_name=config.timezone,
        seed=config.random_seed,
    )

    evaluation_days = set(
        [*dataset.split.validation_days, *dataset.split.test_days]
    )
    local_days = pd.Index(history.index.tz_convert(config.timezone).date)
    evaluation_mask = np.asarray(local_days.isin(evaluation_days), dtype=bool)
    expected = issued.reindex(history.index[evaluation_mask]).loc[
        :, [f"residual_corrected__{q}" for q in ("q10", "q50", "q90")]
    ]
    expected.columns = [
        "residual_corrected__q10",
        "residual_corrected__q50",
        "residual_corrected__q90",
    ]
    pd.testing.assert_frame_equal(
        history.loc[
            evaluation_mask,
            [
                "residual_corrected__q10",
                "residual_corrected__q50",
                "residual_corrected__q90",
            ],
        ],
        expected,
        check_names=False,
    )
    evaluation_audit = audit.loc[audit["phase"].isin(["validation", "test"])]
    assert evaluation_audit["output_source"].eq("published_issued_history").all()
    assert summary["published_overlay_days"] == len(evaluation_days)


def test_config_validates_nested_candidate_kind_grid(tmp_path: Path) -> None:
    source = _source_run(tmp_path)
    config_path = _config(tmp_path, source)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["models"]["kalman"]["fixed_parameters"].pop("candidate_kinds")
    payload["models"]["kalman"]["parameter_grid"]["candidate_kinds"] = [
        ["linear_bias", "linear_weather"],
        ["linear_bias", "linear_typo"],
    ]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(AuxiliaryLabConfigError, match="Famille Kalman inconnue"):
        load_lab_config(config_path, project_root=tmp_path)


def test_kalman_weather_sources_are_aligned_audited_and_split(tmp_path: Path) -> None:
    source = _source_run(tmp_path)
    config_path = _config(tmp_path, source)
    source_path = _enable_weather_covariates(
        tmp_path,
        source,
        config_path,
        minimum_coverage=0.99,
    )
    config = load_lab_config(config_path, project_root=tmp_path)

    dataset = load_kalman_dataset(config, config.models["kalman"])

    assert len(dataset.history) == 45 * 24
    assert dataset.covariates.index.equals(dataset.history.index)
    assert set(dataset.coverage["phase"]) == {
        "train",
        "validation",
        "test",
        "future",
    }
    temperature = dataset.coverage.loc[
        dataset.coverage["column"].eq("fr_temperature_fcst")
    ]
    assert temperature["coverage"].eq(1.0).all()
    assert dataset.source_audit[0].path == str(source_path.resolve())
    assert dataset.source_audit[0].matched_hours == 46 * 24


def test_kalman_external_join_never_shrinks_or_fills_history(tmp_path: Path) -> None:
    source = _source_run(tmp_path)
    config_path = _config(tmp_path, source)
    source_path = _enable_weather_covariates(tmp_path, source, config_path)
    weather = pd.read_csv(source_path)
    missing_timestamp = pd.to_datetime(weather.loc[100, "valid_time"], utc=True)
    weather.loc[100, "temperature_c"] = np.nan
    weather.to_csv(source_path, index=False, compression="gzip")
    config = load_lab_config(config_path, project_root=tmp_path)

    dataset = load_kalman_dataset(config, config.models["kalman"])

    assert len(dataset.history) == 45 * 24
    assert np.isnan(dataset.covariates.loc[missing_timestamp, "fr_temperature_fcst"])
    observed = dataset.coverage.loc[
        (dataset.coverage["phase"] == "train")
        & (dataset.coverage["column"] == "fr_temperature_fcst"),
        "finite_hours",
    ].iloc[0]
    assert observed == len(dataset.split.train_days) * 24 - 1


def test_kalman_future_covariates_are_strict(tmp_path: Path) -> None:
    source = _source_run(tmp_path)
    config_path = _config(tmp_path, source)
    source_path = _enable_weather_covariates(tmp_path, source, config_path)
    weather = pd.read_csv(source_path)
    weather.loc[weather.index[-1], "wind_generation_gw"] = np.nan
    weather.to_csv(source_path, index=False, compression="gzip")
    config = load_lab_config(config_path, project_root=tmp_path)

    with pytest.raises(ValueError, match="futures ne couvrent pas exactement"):
        load_kalman_dataset(config, config.models["kalman"])


def test_kalman_external_source_rejects_naive_timestamps(tmp_path: Path) -> None:
    source = _source_run(tmp_path)
    config_path = _config(tmp_path, source)
    source_path = _enable_weather_covariates(tmp_path, source, config_path)
    weather = pd.read_csv(source_path)
    weather["valid_time"] = pd.to_datetime(weather["valid_time"], utc=True).dt.tz_localize(None)
    weather.to_csv(source_path, index=False, compression="gzip")
    config = load_lab_config(config_path, project_root=tmp_path)

    with pytest.raises(ValueError, match="offset/tz explicite"):
        load_kalman_dataset(config, config.models["kalman"])


def test_kalman_weather_training_persists_contract_and_coefficients(
    tmp_path: Path,
) -> None:
    source = _source_run(tmp_path)
    config_path = _config(tmp_path, source, report=True)
    source_path = _enable_weather_covariates(
        tmp_path,
        source,
        config_path,
        minimum_coverage=0.99,
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["models"]["residual_corrector"]["enabled"] = False
    payload["models"]["mkonline_blend"]["enabled"] = False
    payload["models"]["kalman"]["fixed_parameters"]["candidate_kinds"] = [
        "linear_bias",
        "linear_weather",
    ]
    payload["models"]["kalman"]["training_lookback_days"] = 15
    payload["models"]["kalman"]["rolling_refit_workers"] = 1
    payload["models"]["kalman"]["parameter_grid"] = {"q_over_r": [0.001]}
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    config = load_lab_config(config_path, project_root=tmp_path)

    output = train_experiment(config)

    model = json.loads(
        (output / "models" / "kalman" / "model.json").read_text(
            encoding="utf-8"
        )
    )
    assert model["covariates"]["groups"]["weather"] == [
        "fr_temperature_fcst",
        "fr_temperature_ramp",
        "fr_heating_degree",
    ]
    assert model["additional_sources"][0]["columns"] == {
        "fr_temperature_fcst": "temperature_c",
        "fr_wind_generation_fcst": "wind_generation_gw",
    }
    assert model["additional_sources"][0]["sha256_at_training"] == hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    assert model["training_lookback_days"] == 15
    assert model["rolling_refit_cache_policy"] == (
        "persistent_content_addressed_daily_fits"
    )
    assert model["selection_protocol"] == "daily_refit_trailing_15_days"
    assert model["final_evaluation_protocol"] == (
        "validation_selection_and_sealed_test_daily_refit_trailing_15_days"
    )
    coverage = pd.read_csv(
        output / "models" / "kalman" / "kalman_covariate_coverage.csv"
    )
    assert set(coverage["phase"]) == {"train", "validation", "test", "future"}
    coefficients = pd.read_csv(
        output / "models" / "kalman" / "kalman_final_coefficients.csv"
    )
    assert set(coefficients["filter_kind"]) == {"linear_weather"}
    assert set(coefficients["feature"]) == {
        "fr_temperature_fcst",
        "fr_temperature_ramp",
        "fr_heating_degree",
    }
    replay_audit = json.loads(
        (output / "models" / "kalman" / "kalman_replay_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert replay_audit["rolling_refit_cache"]["enabled"] is True
    assert list(
        (
            tmp_path
            / "runs"
            / "cache"
            / "kalman_rolling"
            / "auxiliary_lab"
            / config.experiment_id
        ).rglob("*.pickle")
    )
    report = (output / "report.html").read_text(encoding="utf-8")
    assert "ne démontrent aucune causalité" in report


def test_day_split_accepts_physical_dst_days() -> None:
    index = pd.date_range(
        pd.Timestamp("2026-03-20", tz="Europe/Paris"),
        pd.Timestamp("2026-04-10", tz="Europe/Paris"),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    from auxiliary_lab.config import SplitConfig

    split = validate_and_split_days(
        index,
        timezone="Europe/Paris",
        split=SplitConfig(
            validation_days=3,
            test_days=3,
            minimum_training_days=10,
            horizon_hours=None,
        ),
    )

    assert len(split.train_days) == 15
    spring_day = pd.Timestamp("2026-03-29").date()
    local = pd.Index(index.tz_convert("Europe/Paris").date)
    assert int((local == spring_day).sum()) == 23


def test_residual_loader_keeps_sealed_prefix_and_overlays_latest_statistics(
    tmp_path: Path,
) -> None:
    source = _source_run(tmp_path)
    statistics_path = source / "statistics_history_hourly.csv.gz"
    statistics = pd.read_csv(statistics_path).tail(10 * 24).copy()
    latest_timestamp = pd.to_datetime(
        statistics.iloc[-1]["delivery_start_utc"], utc=True
    )
    statistics.loc[
        statistics.index[-1],
        ["chronos2__q10", "chronos2__q50", "chronos2__q90"],
    ] += 3.0
    expected_latest_q50 = float(statistics.iloc[-1]["chronos2__q50"])
    statistics.to_csv(statistics_path, index=False, compression="gzip")

    config = load_lab_config(_config(tmp_path, source), project_root=tmp_path)
    dataset = load_residual_dataset(
        config,
        config.models["residual_corrector"],
    )

    local_days = pd.Index(dataset.X.index.tz_convert(config.timezone).date)
    assert local_days.nunique() == 45
    assert dataset.base.loc[latest_timestamp, "q50"] == pytest.approx(
        expected_latest_q50
    )


def test_all_auxiliary_models_train_persist_report_and_predict(
    tmp_path: Path,
) -> None:
    source = _source_run(tmp_path)
    source_before = _hash_tree(source)
    config = load_lab_config(_config(tmp_path, source), project_root=tmp_path)

    output = train_experiment(config)

    assert output == config.output_directory
    assert _hash_tree(source) == source_before
    assert (output / "run_manifest.json").is_file()
    assert (output / "checksums.json").is_file()
    assert (output / "report.html").is_file()
    assert "Mode nuit" in (output / "report.html").read_text(encoding="utf-8")
    leaderboard = pd.read_csv(output / "leaderboard.csv")
    assert set(leaderboard["model"]) == {
        "residual_corrector",
        "mkonline_blend",
        "kalman",
    }
    assert leaderboard.groupby("model")["selected"].sum().eq(1).all()
    predictions = pd.read_csv(output / "predictions.csv.gz")
    assert set(predictions["phase"]) == {"validation", "test"}
    assert predictions[["q10", "q50", "q90"]].notna().all().all()
    assert (predictions["q10"] <= predictions["q50"]).all()
    assert (predictions["q50"] <= predictions["q90"]).all()

    evaluation = evaluate_run(output, output_directory=tmp_path / "evaluation")
    assert (evaluation / "metrics.csv").is_file()
    assert (evaluation / "report.html").is_file()
    comparison = compare_runs(
        (output, output),
        output_directory=tmp_path / "comparison",
    )
    assert (comparison / "comparison_metrics.csv").is_file()
    assert (comparison / "comparison_report.html").is_file()

    residual_prediction = predict_artifact(
        output / "models" / "residual_corrector",
        source_run=source,
        output_path=tmp_path / "residual_future.csv",
    )
    blend_prediction = predict_artifact(
        output / "models" / "mkonline_blend" / "model.json",
        source_run=source,
        output_path=tmp_path / "blend_future.csv",
    )
    for path in (residual_prediction, blend_prediction):
        frame = pd.read_csv(path)
        assert len(frame) == 24
        assert frame[["q10", "q50", "q90"]].notna().all().all()
