from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly import chronos_residual_load as provider


class FakeSaturnClient:
    def __init__(self, histories: dict[str, pd.Series]) -> None:
        self.histories = histories
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, series_name: str, **kwargs: Any) -> pd.Series:
        self.calls.append((series_name, kwargs))
        return self.histories[series_name]


class FakeChronosPipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.maximum_context_timestamp: pd.Timestamp | None = None

    def predict_df(self, context: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        self.calls.append(dict(kwargs))
        maximum = pd.Timestamp(context["timestamp"].max()).tz_localize("UTC")
        if self.maximum_context_timestamp is None:
            self.maximum_context_timestamp = maximum
        else:
            self.maximum_context_timestamp = max(
                self.maximum_context_timestamp, maximum
            )
        length = int(kwargs["prediction_length"])
        frames: list[pd.DataFrame] = []
        for position, (item_id, item) in enumerate(
            context.groupby("item_id", sort=True), start=1
        ):
            last = pd.Timestamp(item["timestamp"].max())
            timestamps = pd.date_range(
                last + pd.Timedelta(hours=1), periods=length, freq="h"
            )
            median = position * 10.0 + np.arange(length, dtype=float)
            frames.append(
                pd.DataFrame(
                    {
                        "item_id": item_id,
                        "timestamp": timestamps,
                        "target_name": "target",
                        "predictions": median,
                        "0.1": median - 1.0,
                        "0.5": median,
                        "0.9": median + 1.0,
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)


def _write_saturn_control_archive(
    root: Path,
    *,
    delivery_day: str,
    zone: str = "FR",
    include_primary: bool = True,
    prefix_nan: bool = False,
) -> Path:
    delivery = provider._delivery_index_utc(delivery_day)
    history = pd.DatetimeIndex(
        [
            delivery[0] - pd.Timedelta(hours=2),
            delivery[0] - pd.Timedelta(hours=1),
        ]
    )
    archive = root / f"{zone.lower()}_day_ahead_{delivery_day}"
    inputs = archive / "inputs"
    inputs.mkdir(parents=True)
    aligned = pd.DataFrame(
        {
            "timestamp": history,
            "target": [50.0, 51.0],
            **{alias: [1.0, 2.0] for alias in provider.EXPECTED_ALIASES},
        }
    )
    aligned.to_csv(
        inputs / "aligned_inputs.csv.gz",
        index=False,
        compression="gzip",
    )
    model_index = history.append(delivery)
    model_values = [np.nan if prefix_nan else 1.0, 2.0] + [
        999999.0
    ] * len(delivery)
    model = pd.DataFrame(
        {
            "timestamp": model_index,
            **{
                alias: model_values
                for alias in provider.EXPECTED_ALIASES
            },
            **{
                f"known_{alias}_oracle": model_values
                for alias in provider.EXPECTED_ALIASES
            },
        }
    )
    model.to_csv(
        inputs / "model_covariates_with_future.csv.gz",
        index=False,
        compression="gzip",
    )
    if include_primary:
        pd.DataFrame(
            {
                "delivery_start_utc": delivery,
                "forecast_cutoff_utc": delivery[0] - pd.Timedelta(hours=18),
                "value": np.arange(len(delivery), dtype=float),
            }
        ).to_parquet(inputs / "mkonline_primary_live.parquet", index=False)
    run_manifest = {
        "zone": zone,
        "delivery_day_local": delivery_day,
        "run_type": "live_day_ahead",
        "forecast_status": "issued_live",
    }
    (archive / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    declarations = []
    for relative, role in (
        ("run_manifest.json", "run_artifact"),
        ("inputs/aligned_inputs.csv.gz", "run_artifact"),
        (
            "inputs/model_covariates_with_future.csv.gz",
            "run_artifact",
        ),
        *(
            (("inputs/mkonline_primary_live.parquet", "run_artifact"),)
            if include_primary
            else ()
        ),
    ):
        path = archive / relative
        declarations.append(
            {
                "path": relative,
                "role": role,
                "size_bytes": path.stat().st_size,
                "sha256": provider._sha256(path),
            }
        )
    (archive / "artifact_checksums.json").write_text(
        json.dumps(
            {
                "algorithm": "sha256",
                "output_directory": str(archive.resolve()),
                "artifacts": declarations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return archive


def _histories(
    cutoff: pd.Timestamp,
    *,
    gap_country: str | None = "be",
    gap_hours: int = 6,
    include_after_cutoff: bool = True,
) -> dict[str, pd.Series]:
    index = pd.date_range(
        end=cutoff - pd.Timedelta(hours=2),
        periods=2200,
        freq="h",
        tz="UTC",
    )
    result: dict[str, pd.Series] = {}
    for position, (country, series_name) in enumerate(
        provider.COUNTRY_OBSERVED_SERIES.items(), start=1
    ):
        country_index = index
        values = np.arange(len(index), dtype=float) + position * 1000.0
        series = pd.Series(values, index=country_index, name=series_name)
        if country == gap_country and gap_hours:
            gap = index[-100 : -100 + gap_hours]
            series = series.drop(gap)
        if include_after_cutoff:
            series.loc[cutoff + pd.Timedelta(hours=1)] = -999999.0
        result[series_name] = series.sort_index()
    return result


@pytest.mark.parametrize(
    ("delivery_day", "expected_hours"),
    (("2026-03-29", 23), ("2026-10-25", 25), ("2026-08-25", 24)),
)
def test_delivery_grid_respects_europe_paris_dst(
    delivery_day: str, expected_hours: int
) -> None:
    delivery = provider._delivery_index_utc(delivery_day)

    assert len(delivery) == expected_hours
    assert delivery.tz is not None
    assert str(delivery.tz) == "UTC"
    assert delivery.is_unique
    assert (delivery[1:] - delivery[:-1] == pd.Timedelta(hours=1)).all()


def test_builds_causal_five_country_pit_bundle_and_reuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cutoff = pd.Timestamp("2026-08-24 06:00:00Z")
    client = FakeSaturnClient(_histories(cutoff))
    pipeline = FakeChronosPipeline()
    load_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        provider,
        "_create_saturn_client",
        lambda saturn_url, author: client,
    )

    def fake_load(**kwargs: Any) -> FakeChronosPipeline:
        load_calls.append(kwargs)
        return pipeline

    monkeypatch.setattr(provider, "_load_pipeline", fake_load)

    manifest_path = provider.build_live_residual_load_bundle(
        delivery_day="2026-08-25",
        runtime_cutoff=cutoff,
        output_root=tmp_path,
        saturn_url="https://saturn.invalid",
        saturn_author="pytest",
        device="cpu",
        local_files_only=True,
        batch_size=3,
    )
    first_manifest_bytes = manifest_path.read_bytes()
    manifest = provider.validate_live_residual_load_bundle(
        manifest_path,
        expected_delivery_day="2026-08-25",
        expected_runtime_cutoff=cutoff,
    )
    alias_paths = provider.bundle_alias_paths(manifest_path)

    assert set(alias_paths) == set(provider.EXPECTED_ALIASES)
    assert manifest["provider"] == "chronos2"
    assert manifest["model_id"] == "amazon/chronos-2"
    assert manifest["model_revision"] == provider.MODEL_REVISION
    assert manifest["context_length"] == 2048
    assert manifest["saturn_forecast_series_used"] is False
    assert len(load_calls) == 1
    assert load_calls[0]["revision"] == provider.MODEL_REVISION
    assert pipeline.maximum_context_timestamp is not None
    assert pipeline.maximum_context_timestamp <= cutoff
    assert client.calls
    assert all(".fcst" not in series_name for series_name, _ in client.calls)
    assert all(series_name.endswith(".obs") for series_name, _ in client.calls)
    assert all(call["revision_date"] == cutoff for _, call in client.calls)
    assert all(call["to_value_date"] == cutoff for _, call in client.calls)

    input_by_alias = {item["alias"]: item for item in manifest["inputs"]}
    be_audit = input_by_alias["be_residual_load_fcst"]
    assert be_audit["imputed_hours"] == 6
    assert len(be_audit["imputed_timestamps_utc"]) == 6
    assert all(
        item["discarded_observations_after_cutoff"] == 1
        for item in manifest["inputs"]
    )

    assert provider.planned_live_residual_load_manifest_path(
        delivery_day="2026-08-25",
        runtime_cutoff=cutoff,
        output_root=tmp_path,
    ) == manifest_path
    production_root = tmp_path / "production_pit"
    production_root.mkdir()
    delivery_index = provider._delivery_index_utc("2026-08-25")
    production_index = pd.DatetimeIndex(
        [
            delivery_index[0] - pd.Timedelta(hours=2),
            delivery_index[0] - pd.Timedelta(hours=1),
            delivery_index[0],
            delivery_index[1],
            delivery_index[-1] + pd.Timedelta(hours=1),
        ]
    )
    for alias in provider.EXPECTED_ALIASES:
        pd.DataFrame(
            {
                "value_time_utc": production_index,
                "snapshot_time_utc": cutoff - pd.Timedelta(days=1),
                "revision_time_utc": cutoff - pd.Timedelta(days=1),
                "value": [1.0, 2.0, 999999.0, 999999.0, 999999.0],
                "downloaded_at_utc": cutoff,
            }
        ).to_parquet(production_root / f"{alias}.parquet", index=False)
    runtime_config: dict[str, Any] = {
        "data": {
            "pit_vintage_dir": "production_pit",
            "pit_files": {
                "unrelated": "keep.parquet",
                **{
                    alias: f"{alias}.parquet"
                    for alias in provider.EXPECTED_ALIASES
                },
            },
        },
        "zones": {
            "FR": {
                "covariates": {
                    alias: {
                        "series": f"power.{alias}.saturn.fcst",
                        "source": "pit_parquet",
                        "pit_file": "old.parquet",
                    }
                    for alias in provider.EXPECTED_ALIASES
                }
            }
        },
    }
    staging = tmp_path / ".live.tmp"
    staging_inputs = staging / "inputs"
    overlay_dir = staging_inputs / "residual_load_pit_overlay"
    saturn_control = _write_saturn_control_archive(
        tmp_path / "issued",
        delivery_day="2026-08-25",
    )
    provenance = provider.apply_residual_load_bundle(
        runtime_config,
        manifest_path=manifest_path,
        expected_delivery_index=provider._delivery_index_utc("2026-08-25"),
        expected_cutoff_utc=cutoff,
        config_dir=tmp_path,
        overlay_dir=overlay_dir,
        saturn_control_archive=saturn_control,
        expected_zone="FR",
    )
    archived_bundle = provider.archive_live_residual_load_bundle(
        manifest_path,
        archive_inputs_dir=staging_inputs,
        expected_delivery_day="2026-08-25",
        expected_runtime_cutoff=cutoff,
    )
    assert provenance["manifest_sha256"]
    assert provenance["saturn_forecast_series_used"] is False
    assert provenance["historical_context_source"] == "production_pit_unchanged"
    assert provenance["effective_historical_context_source"] == (
        "sealed_saturn_same_zone_same_delivery_day"
    )
    assert provenance["sealed_saturn_control"]["archive_origin_path"] == str(
        saturn_control.resolve()
    )
    assert provenance["delivery_day_values_source"] == "chronos2"
    assert provenance["composite_path_base"] == "runtime_inputs_directory"
    assert provenance["manifest_path"] == str(manifest_path.resolve())
    assert provenance["archived_manifest_path"] == (
        "residual_load_bundle/manifest.json"
    )
    assert archived_bundle["origin_manifest_path"] == str(
        manifest_path.resolve()
    )
    assert archived_bundle["archived_manifest_path"] == (
        "residual_load_bundle/manifest.json"
    )
    composite_manifest_reference = Path(
        provenance["composite_manifest_path"]
    )
    assert not composite_manifest_reference.is_absolute()
    assert (staging_inputs / composite_manifest_reference).is_file()
    assert str(staging.resolve()) not in json.dumps(provenance)
    assert set(provenance["files"]) == set(provider.EXPECTED_ALIASES)
    assert runtime_config["data"]["pit_files"]["unrelated"] == "keep.parquet"
    for alias in provider.EXPECTED_ALIASES:
        covariate = runtime_config["zones"]["FR"]["covariates"][alias]
        assert covariate["series"] is None
        assert covariate["source"] == "pit_parquet"
        assert "pit_file" not in covariate
        composite_path = Path(runtime_config["data"]["pit_files"][alias])
        assert composite_path.is_absolute()
        composite = pd.read_parquet(composite_path)
        assert len(composite) == 2 + len(delivery_index)
        assert composite.iloc[:2]["value"].tolist() == [1.0, 2.0]
        assert pd.DatetimeIndex(composite.iloc[2:]["value_time_utc"]).equals(
            delivery_index
        )
        upstream = pd.read_parquet(alias_paths[alias])
        assert composite.iloc[2:]["value"].reset_index(drop=True).equals(
            upstream["value"]
        )
        assert 999999.0 not in composite.iloc[2:]["value"].tolist()
        assert provenance["files"][alias]["historical_rows"] == 2
        assert provenance["files"][alias]["delivery_day_rows"] == 24
        assert provenance["files"][alias][
            "archived_upstream_forecast_path"
        ].startswith("residual_load_bundle/")
        assert (
            staging_inputs
            / Path(
                provenance["files"][alias][
                    "archived_upstream_forecast_path"
                ]
            )
        ).is_file()
        composite_reference = Path(provenance["files"][alias]["path"])
        assert not composite_reference.is_absolute()
        assert (staging_inputs / composite_reference).is_file()

    for artifact in alias_paths.values():
        frame = pd.read_parquet(artifact)
        assert len(frame) == 24
        assert frame["snapshot_time_utc"].eq(cutoff).all()
        assert frame["revision_time_utc"].eq(cutoff).all()
        assert frame["value"].equals(frame["q50"])

    reused = provider.build_live_residual_load_bundle(
        delivery_day="2026-08-25",
        runtime_cutoff=cutoff,
        output_root=tmp_path,
        saturn_url="https://saturn.invalid",
        saturn_author="pytest",
        device="cpu",
        local_files_only=True,
        batch_size=3,
    )
    assert reused == manifest_path
    assert reused.read_bytes() == first_manifest_bytes
    assert len(load_calls) == 1
    assert len(client.calls) == 5

    final_run = tmp_path / "live_final"
    staging.replace(final_run)
    final_inputs = final_run / "inputs"
    final_archived_manifest = final_inputs / Path(
        archived_bundle["archived_manifest_path"]
    )
    assert final_archived_manifest.is_file()
    assert (final_inputs / composite_manifest_reference).is_file()
    for alias in provider.EXPECTED_ALIASES:
        assert (
            final_inputs / Path(provenance["files"][alias]["path"])
        ).is_file()

    archived_overlay = provider.validate_archived_residual_load_overlay(
        final_inputs,
        upstream_manifest_path=final_archived_manifest,
        upstream_origin_manifest_path=manifest_path,
        expected_delivery_day="2026-08-25",
        expected_runtime_cutoff=cutoff,
    )
    assert archived_overlay["manifest_sha256"] == provenance[
        "composite_manifest_sha256"
    ]
    assert set(archived_overlay["files"]) == set(provider.EXPECTED_ALIASES)

    final_composite_manifest = final_inputs / composite_manifest_reference
    valid_composite_manifest_bytes = final_composite_manifest.read_bytes()
    tampered_overlay = json.loads(
        final_composite_manifest.read_text(encoding="utf-8")
    )
    tampered_overlay["saturn_forecast_rows_on_delivery_day"] = 1
    final_composite_manifest.write_text(
        json.dumps(tampered_overlay),
        encoding="utf-8",
    )
    with pytest.raises(
        provider.BundleValidationError,
        match="saturn_forecast_rows_on_delivery_day",
    ):
        provider.validate_archived_residual_load_overlay(
            final_inputs,
            upstream_manifest_path=final_archived_manifest,
            upstream_origin_manifest_path=manifest_path,
            expected_delivery_day="2026-08-25",
            expected_runtime_cutoff=cutoff,
        )
    tampered_overlay["saturn_forecast_rows_on_delivery_day"] = 0
    final_composite_manifest.write_text(
        json.dumps(tampered_overlay, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    final_composite_manifest = final_inputs / composite_manifest_reference
    composite_audit = json.loads(
        final_composite_manifest.read_text(encoding="utf-8")
    )
    first_alias = provider.EXPECTED_ALIASES[0]
    declarations = {
        str(item["alias"]): item for item in composite_audit["artifacts"]
    }
    declarations[first_alias]["path"] = str(
        final_inputs / Path(provenance["files"][first_alias]["path"])
    )
    final_composite_manifest.write_text(
        json.dumps(composite_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    production_paths = {
        alias: final_inputs / "sealed_saturn_control_pit" / f"{alias}.parquet"
        for alias in provider.EXPECTED_ALIASES
    }
    with pytest.raises(
        provider.BundleValidationError,
        match="production_pit_path divergent",
    ):
        provider._build_composite_overlay(
            overlay_dir=final_inputs / "residual_load_pit_overlay",
            upstream_manifest_path=manifest_path,
            upstream_paths=alias_paths,
            production_paths=production_paths,
            delivery_index=delivery_index,
            cutoff=cutoff,
        )

    final_composite_manifest.write_bytes(valid_composite_manifest_bytes)
    origin_bundle_directory = manifest_path.parent
    shutil.rmtree(origin_bundle_directory)
    assert not manifest_path.exists()
    autonomous_validation = provider.validate_archived_residual_load_overlay(
        final_inputs,
        upstream_manifest_path=final_archived_manifest,
        upstream_origin_manifest_path=manifest_path,
        expected_delivery_day="2026-08-25",
        expected_runtime_cutoff=cutoff,
    )
    assert autonomous_validation["upstream_manifest_path"] == (
        final_archived_manifest.resolve()
    )
    assert set(autonomous_validation["files"]) == set(
        provider.EXPECTED_ALIASES
    )


def test_rejects_live_bundle_before_auction_cutoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cutoff = pd.Timestamp("2026-08-24T06:00:00Z")
    monkeypatch.setattr(
        provider,
        "_now_utc",
        lambda: cutoff - pd.Timedelta(minutes=1),
    )

    with pytest.raises(ValueError, match="encore dans le futur"):
        provider.build_live_residual_load_bundle(
            delivery_day="2026-08-25",
            runtime_cutoff=cutoff,
            output_root=tmp_path / "upstream",
            saturn_url="https://saturn.invalid",
            saturn_author="pytest",
        )

    assert not (tmp_path / "upstream").exists()


def test_challenger_reuses_sealed_saturn_context_and_primary_byte_for_byte(
    tmp_path: Path,
) -> None:
    delivery_day = "2026-08-25"
    delivery = provider._delivery_index_utc(delivery_day)
    control_archive = _write_saturn_control_archive(
        tmp_path / "issued",
        delivery_day=delivery_day,
    )
    control = provider.validate_sealed_saturn_control_archive(
        control_archive,
        expected_delivery_day=delivery_day,
        expected_zone="FR",
        require_primary=True,
    )
    fresh = control["model"].drop(columns=["timestamp"]).copy()
    fresh.index = control["model_index"]
    delivery_mask = fresh.index.isin(delivery)
    fresh.loc[delivery_mask, list(provider.RESIDUAL_LOAD_TREATMENT_COLUMNS)] = 7.0
    data = SimpleNamespace(
        timezone="Europe/Paris",
        target=pd.Series([0.0], index=[delivery[0] - pd.Timedelta(hours=1)]),
        covariates=pd.DataFrame(),
        model_context_covariates=fresh.set_axis(
            fresh.index.tz_convert("Europe/Paris"), axis=0
        ),
        diagnostics={},
    )
    inputs = tmp_path / "challenger" / "inputs"

    audit = provider.seal_challenger_zone_data_from_saturn_control(
        data,
        saturn_archive_dir=control_archive,
        archive_inputs_dir=inputs,
        expected_delivery_index=delivery,
        expected_zone="FR",
    )

    hybrid = data.model_context_covariates.copy()
    hybrid.index = hybrid.index.tz_convert("UTC")
    historical = hybrid.index < delivery[0]
    expected_history = pd.DataFrame(
        [[1.0] * 10, [2.0] * 10],
        index=hybrid.index[historical],
        columns=provider.RESIDUAL_LOAD_TREATMENT_COLUMNS,
    )
    assert hybrid.loc[
        historical, list(provider.RESIDUAL_LOAD_TREATMENT_COLUMNS)
    ].eq(expected_history).all().all()
    assert hybrid.loc[
        delivery_mask, list(provider.RESIDUAL_LOAD_TREATMENT_COLUMNS)
    ].eq(7.0).all().all()
    assert provider._sha256(inputs / "aligned_inputs.csv.gz") == provider._sha256(
        control["files"]["aligned_inputs"]["path"]
    )
    assert audit["aligned_inputs_reused_byte_for_byte"] is True
    assert audit["only_overlaid_scope"] == "delivery_day_j_plus_1"

    primary_destination = inputs / "mkonline_primary_live.parquet"
    primary_audit = provider.copy_sealed_saturn_primary(
        control_archive,
        destination=primary_destination,
        expected_delivery_day=delivery_day,
        expected_zone="FR",
    )
    assert primary_audit["copied_byte_for_byte"] is True
    assert provider._sha256(primary_destination) == provider._sha256(
        control["files"]["mkonline_primary_live"]["path"]
    )

    control["files"]["model_covariates_with_future"]["path"].write_bytes(
        b"tampered"
    )
    with pytest.raises(provider.BundleValidationError, match="SHA-256 divergent"):
        provider.validate_sealed_saturn_control_archive(
            control_archive,
            expected_delivery_day=delivery_day,
            expected_zone="FR",
        )


def test_sealed_saturn_pit_control_filters_unavailable_nan_prefix(
    tmp_path: Path,
) -> None:
    archive = _write_saturn_control_archive(
        tmp_path / "issued",
        delivery_day="2026-08-25",
        prefix_nan=True,
    )
    control = provider.validate_sealed_saturn_control_archive(
        archive,
        expected_delivery_day="2026-08-25",
        expected_zone="FR",
    )

    paths = provider._materialize_sealed_saturn_pit_controls(
        control,
        destination=tmp_path / "runtime" / "sealed_saturn_control_pit",
    )

    for path in paths.values():
        frame = pd.read_parquet(path)
        assert len(frame) == 25
        assert np.isfinite(frame["value"].to_numpy(dtype=float)).all()


def test_rejects_internal_gap_longer_than_six_hours_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cutoff = pd.Timestamp("2026-08-24 06:00:00Z")
    client = FakeSaturnClient(
        _histories(
            cutoff,
            gap_country="be",
            gap_hours=provider.MAX_INTERNAL_GAP_HOURS + 1,
            include_after_cutoff=False,
        )
    )
    model_loaded = False

    monkeypatch.setattr(
        provider,
        "_create_saturn_client",
        lambda saturn_url, author: client,
    )

    def forbidden_load(**kwargs: Any) -> FakeChronosPipeline:
        nonlocal model_loaded
        model_loaded = True
        return FakeChronosPipeline()

    monkeypatch.setattr(provider, "_load_pipeline", forbidden_load)

    with pytest.raises(provider.ResidualLoadBundleError, match="gap interne de 7"):
        provider.build_live_residual_load_bundle(
            delivery_day="2026-08-25",
            runtime_cutoff=cutoff,
            output_root=tmp_path,
            saturn_url="https://saturn.invalid",
            saturn_author="pytest",
        )

    assert model_loaded is False
    assert not list(tmp_path.glob("*/manifest.json"))


def test_corrupted_hash_is_never_reused_or_silently_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cutoff = pd.Timestamp("2026-08-24 06:00:00Z")
    client = FakeSaturnClient(_histories(cutoff, gap_country=None))
    pipeline = FakeChronosPipeline()
    load_count = 0

    monkeypatch.setattr(
        provider,
        "_create_saturn_client",
        lambda saturn_url, author: client,
    )

    def fake_load(**kwargs: Any) -> FakeChronosPipeline:
        nonlocal load_count
        load_count += 1
        return pipeline

    monkeypatch.setattr(provider, "_load_pipeline", fake_load)
    manifest_path = provider.build_live_residual_load_bundle(
        delivery_day="2026-08-25",
        runtime_cutoff=cutoff,
        output_root=tmp_path,
        saturn_url="https://saturn.invalid",
        saturn_author="pytest",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = manifest_path.parent / manifest["artifacts"][0]["path"]
    artifact.write_bytes(b"corrupted")

    with pytest.raises(provider.BundleValidationError, match="SHA-256 divergent"):
        provider.build_live_residual_load_bundle(
            delivery_day="2026-08-25",
            runtime_cutoff=cutoff,
            output_root=tmp_path,
            saturn_url="https://saturn.invalid",
            saturn_author="pytest",
        )

    assert load_count == 1
    assert len(client.calls) == 5
    assert artifact.read_bytes() == b"corrupted"
