from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.rolling_refit import RollingRefitPolicy
from chronos2_hourly.rolling_refit_loader import (
    FilesystemBlockSpec,
    load_rolling_refit_filesystem,
)
from chronos2_hourly.rolling_capture import (
    BLOCK_FILENAME,
    CANDIDATE_FILENAME,
    CHECKSUM_FILENAME,
    MANIFEST_FILENAME,
    RollingCaptureError,
    capture_issued_live_block_isolated,
    finalize_target_pending_candidate,
    prepare_supported_capture_inputs,
    prove_frozen_builder_subset_equivalence,
    write_target_pending_candidate,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _published_archive(
    tmp_path: Path,
    *,
    day: str,
    target_source: Path,
) -> Path:
    archive = tmp_path / "published" / f"fr_day_ahead_{day}"
    archive.mkdir(parents=True)
    forecast = archive / "forecast_hourly_fr.csv"
    forecast.write_text("delivery_start_utc,q50\n", encoding="utf-8")
    run_manifest = archive / "run_manifest.json"
    run_manifest.write_text(
        json.dumps(
            {
                "zone": "FR",
                "target_series": "power.price.fr.euromwh.h.spot",
                "delivery_day_local": day,
                "run_type": "live_day_ahead",
                "forecast_status": "issued_live",
                "issued_at_utc": f"{day}T07:00:00Z",
                "target_source_path": str(target_source.resolve()),
                "target_source_sha256": _sha256(target_source),
                "input_diagnostics": {
                    "target": {"cache": str(target_source.resolve())}
                },
            }
        ),
        encoding="utf-8",
    )
    (archive / CHECKSUM_FILENAME).write_text(
        json.dumps(
            {
                "algorithm": "sha256",
                "artifacts": [
                    {
                        "path": forecast.name,
                        "role": "run_artifact",
                        "sha256": _sha256(forecast),
                    },
                    {
                        "path": run_manifest.name,
                        "role": "run_artifact",
                        "sha256": _sha256(run_manifest),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return archive


def _inputs(tmp_path: Path, day: str = "2026-08-16") -> dict[str, object]:
    index = local_delivery_day_index(day, timezone="Europe/Paris")
    origin = pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
    origin = origin.tz_localize("Europe/Paris").tz_convert("UTC")
    features = pd.DataFrame(
        {
            "known_fr_residual_load_fcst_oracle": np.arange(
                len(index), dtype=float
            ),
            "known_de_residual_load_fcst_oracle": np.arange(
                len(index), dtype=float
            )
            + 1.0,
            "hour_sin": np.sin(np.arange(len(index))),
        },
        index=index,
    )
    chronos = pd.DataFrame(
        {
            "forecast_origin_utc": origin,
            "q10": np.arange(len(index), dtype=float) + 40.0,
            "q50": np.arange(len(index), dtype=float) + 50.0,
            "q90": np.arange(len(index), dtype=float) + 60.0,
        },
        index=index,
    )
    pit_sources: dict[str, dict[str, str]] = {}
    for position, alias in enumerate(("fr_residual_load_fcst", "de_residual_load_fcst")):
        source = tmp_path / f"{alias}.parquet"
        frame = pd.DataFrame(
            {
                "value_time_utc": index,
                "snapshot_time_utc": origin - pd.Timedelta(hours=2 - position),
                "revision_time_utc": origin - pd.Timedelta(hours=3 - position),
                "value": np.arange(len(index), dtype=float) + position,
            }
        )
        frame.to_parquet(source, index=False)
        pit_sources[alias] = {
            "path": str(source),
            "sha256": _sha256(source),
            "feature_column": f"known_{alias}_oracle",
            "serialization_tolerance": 1e-12,
        }
    target_source = tmp_path / "target_cache.csv.gz"
    pd.DataFrame(
        {"timestamp": index, "value": np.arange(len(index), dtype=float) + 70.0}
    ).to_csv(target_source, index=False, compression="gzip")
    archive = _published_archive(tmp_path, day=day, target_source=target_source)
    return {
        "capture_root": tmp_path / "capture",
        "zone": "FR",
        "delivery_day": day,
        "delivery_timezone": "Europe/Paris",
        "full_features_for_equivalence": features.copy(),
        "fresh_features": features,
        "chronos_live": chronos,
        "pit_sources": pit_sources,
        "feature_provenance": {
            "known_fr_residual_load_fcst_oracle": "pit_asof",
            "known_de_residual_load_fcst_oracle": "pit_asof",
            "hour_sin": "deterministic_calendar",
        },
        "expected_config_sha256": "a" * 64,
        "expected_base_bundle_sha256": "b" * 64,
        "target_series": "power.price.fr.euromwh.h.spot",
        "target_source_path": target_source,
        "issued_live_archive": archive,
        "issued_live_forecast_filename": "forecast_hourly_fr.csv",
    }


def test_pending_then_final_block_is_immutable_and_checksummed(tmp_path: Path) -> None:
    kwargs = _inputs(tmp_path)
    pending, pending_audit = write_target_pending_candidate(**kwargs)
    assert pending.name == "2026-08-16"
    assert (pending / CANDIDATE_FILENAME).is_file()
    assert (pending / MANIFEST_FILENAME).is_file()
    assert pending_audit["status"] == "target_pending"
    candidate = pd.read_parquet(pending / CANDIDATE_FILENAME)
    assert "actual" not in candidate
    assert candidate["maximum_snapshot_time_utc"].nunique() == 1
    assert pd.Timestamp(candidate["maximum_snapshot_time_utc"].iloc[0]) == pd.Timestamp(
        "2026-08-15T05:00:00Z"
    )
    pending_checksum = _sha256(pending / CHECKSUM_FILENAME)

    index = local_delivery_day_index("2026-08-16", timezone="Europe/Paris")
    target = pd.Series(np.arange(len(index), dtype=float) + 70.0, index=index)
    observation = _published_archive(
        tmp_path,
        day="2026-08-17",
        target_source=Path(kwargs["target_source_path"]),
    )
    final, final_audit = finalize_target_pending_candidate(
        capture_root=kwargs["capture_root"],
        zone="FR",
        delivery_day="2026-08-16",
        delivery_timezone="Europe/Paris",
        canonical_target=target,
        target_series=kwargs["target_series"],
        target_source_path=kwargs["target_source_path"],
        target_observation_archive=observation,
        target_observation_forecast_filename="forecast_hourly_fr.csv",
    )
    assert final_audit["status"] == "complete"
    block = pd.read_csv(final / BLOCK_FILENAME)
    np.testing.assert_allclose(block["actual"], target.to_numpy())
    manifest = json.loads((final / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["status"] == "sealed"
    assert manifest["pending_artifact_checksums_sha256"] == pending_checksum
    assert manifest["storm_used_as_feature"] is False
    assert manifest["mkonline_used_as_feature"] is False
    assert _sha256(pending / CHECKSUM_FILENAME) == pending_checksum
    loaded = load_rolling_refit_filesystem(
        [
            FilesystemBlockSpec(
                directory=final,
                source_kind="issued_live",
                artifact_checksums_sha256=_sha256(final / CHECKSUM_FILENAME),
            )
        ],
        zone="FR",
        forecast_delivery_day="2026-08-17",
        delivery_timezone="Europe/Paris",
        expected_config_sha256="a" * 64,
        expected_base_bundle_sha256="b" * 64,
        rolling_policy=RollingRefitPolicy(window_days=1),
    )
    assert len(loaded.blocks) == 1
    assert loaded.selection.audit["training_rows"] == len(index)


@pytest.mark.parametrize("forbidden", ["storm_price", "mkonline_signal"])
def test_capture_rejects_forbidden_features_before_writing(
    tmp_path: Path,
    forbidden: str,
) -> None:
    kwargs = _inputs(tmp_path)
    features = kwargs["fresh_features"].copy()
    features[forbidden] = 1.0
    kwargs["fresh_features"] = features
    provenance = dict(kwargs["feature_provenance"])
    provenance[forbidden] = "pit_asof"
    kwargs["feature_provenance"] = provenance
    with pytest.raises(RollingCaptureError, match="forbidden"):
        write_target_pending_candidate(**kwargs)
    assert not Path(kwargs["capture_root"]).exists()


def test_capture_refuses_unproved_pit_checksum(tmp_path: Path) -> None:
    kwargs = _inputs(tmp_path)
    first = next(iter(kwargs["pit_sources"].values()))
    first["sha256"] = "0" * 64
    with pytest.raises(RollingCaptureError, match="checksum mismatch"):
        write_target_pending_candidate(**kwargs)
    assert not Path(kwargs["capture_root"]).exists()


def test_unpublished_official_archive_cannot_create_issued_live_block(
    tmp_path: Path,
) -> None:
    kwargs = _inputs(tmp_path)
    archive = Path(kwargs["issued_live_archive"])
    for path in archive.iterdir():
        path.unlink()
    archive.rmdir()
    with pytest.raises(RollingCaptureError, match="not published"):
        write_target_pending_candidate(**kwargs)
    assert not Path(kwargs["capture_root"]).exists()


def test_capture_rejects_feature_not_bound_to_its_pit_values(tmp_path: Path) -> None:
    kwargs = _inputs(tmp_path)
    features = kwargs["fresh_features"].copy()
    features.loc[features.index[3], "known_fr_residual_load_fcst_oracle"] += 0.1
    kwargs["fresh_features"] = features
    with pytest.raises(RollingCaptureError, match="do not match bound feature"):
        write_target_pending_candidate(**kwargs)
    assert not Path(kwargs["capture_root"]).exists()


def test_capture_rejects_naive_chronos_origin(tmp_path: Path) -> None:
    kwargs = _inputs(tmp_path)
    chronos = kwargs["chronos_live"].copy()
    chronos["forecast_origin_utc"] = pd.to_datetime(
        chronos["forecast_origin_utc"]
    ).dt.tz_localize(None)
    kwargs["chronos_live"] = chronos
    with pytest.raises(RollingCaptureError, match="explicit timezone"):
        write_target_pending_candidate(**kwargs)


def test_capture_requires_exact_d_minus_1_0800_paris_origin(
    tmp_path: Path,
) -> None:
    kwargs = _inputs(tmp_path)
    chronos = kwargs["chronos_live"].copy()
    chronos["forecast_origin_utc"] = pd.to_datetime(
        chronos["forecast_origin_utc"], utc=True
    ) + pd.Timedelta(minutes=1)
    kwargs["chronos_live"] = chronos
    with pytest.raises(RollingCaptureError, match="D-1 08:00 Europe/Paris"):
        write_target_pending_candidate(**kwargs)


def test_capture_rejects_naive_pit_timestamps(tmp_path: Path) -> None:
    kwargs = _inputs(tmp_path)
    alias, spec = next(iter(kwargs["pit_sources"].items()))
    path = Path(spec["path"])
    frame = pd.read_parquet(path)
    frame["revision_time_utc"] = frame["revision_time_utc"].dt.tz_localize(None)
    frame.to_parquet(path, index=False)
    spec["sha256"] = _sha256(path)
    with pytest.raises(RollingCaptureError, match="explicit timezone"):
        write_target_pending_candidate(**kwargs)


def test_isolated_capture_failure_is_non_fatal_and_has_no_official_input(
    tmp_path: Path,
) -> None:
    kwargs = _inputs(tmp_path)
    official = Path(kwargs["issued_live_archive"]) / kwargs[
        "issued_live_forecast_filename"
    ]
    before = official.read_bytes()
    kwargs["pit_sources"] = {}
    index = local_delivery_day_index("2026-08-15", timezone="Europe/Paris")
    result = capture_issued_live_block_isolated(
        **kwargs,
        canonical_target=pd.Series(np.arange(len(index), dtype=float), index=index),
    )
    assert result.status == "failed"
    assert result.audit["official_forecast_status"] == (
        "unmodified_already_published"
    )
    assert result.audit["storm_used_as_feature"] is False
    assert result.audit["mkonline_used_as_feature"] is False
    assert official.read_bytes() == before


def test_finalizer_refuses_tampered_pending_candidate(tmp_path: Path) -> None:
    kwargs = _inputs(tmp_path)
    pending, _audit = write_target_pending_candidate(**kwargs)
    with (pending / CANDIDATE_FILENAME).open("ab") as stream:
        stream.write(b"tampered")
    index = local_delivery_day_index("2026-08-16", timezone="Europe/Paris")
    with pytest.raises(RollingCaptureError, match="checksum mismatch"):
        finalize_target_pending_candidate(
            capture_root=kwargs["capture_root"],
            zone="FR",
            delivery_day="2026-08-16",
            delivery_timezone="Europe/Paris",
            canonical_target=pd.Series(np.arange(len(index), dtype=float), index=index),
            target_series=kwargs["target_series"],
            target_source_path=kwargs["target_source_path"],
            target_observation_archive=kwargs["issued_live_archive"],
            target_observation_forecast_filename="forecast_hourly_fr.csv",
        )


def test_finalizer_refuses_target_source_changed_after_d_plus_1_archive(
    tmp_path: Path,
) -> None:
    kwargs = _inputs(tmp_path)
    write_target_pending_candidate(**kwargs)
    observation = _published_archive(
        tmp_path,
        day="2026-08-17",
        target_source=Path(kwargs["target_source_path"]),
    )
    target_source = Path(kwargs["target_source_path"])
    with target_source.open("ab") as stream:
        stream.write(b"tampered")
    index = local_delivery_day_index("2026-08-16", timezone="Europe/Paris")
    with pytest.raises(RollingCaptureError, match="checksum differs"):
        finalize_target_pending_candidate(
            capture_root=kwargs["capture_root"],
            zone="FR",
            delivery_day="2026-08-16",
            delivery_timezone="Europe/Paris",
            canonical_target=pd.Series(
                np.arange(len(index), dtype=float) + 70.0,
                index=index,
            ),
            target_series=kwargs["target_series"],
            target_source_path=target_source,
            target_observation_archive=observation,
            target_observation_forecast_filename="forecast_hourly_fr.csv",
        )


def test_prepare_supported_inputs_excludes_unproved_price_lags(tmp_path: Path) -> None:
    kwargs = _inputs(tmp_path)
    features = kwargs["fresh_features"].copy()
    features["price_lag_24h"] = 42.0
    selected, sources, provenance, audit = prepare_supported_capture_inputs(
        fresh_features=features,
        required_pit_aliases=(
            "fr_residual_load_fcst",
            "de_residual_load_fcst",
        ),
        pit_freshness=kwargs["pit_sources"],
    )
    assert "price_lag_24h" not in selected
    assert audit["historical_price_features_excluded"] == ["price_lag_24h"]
    assert set(provenance.values()) == {"pit_asof"}
    assert set(sources) == {
        "fr_residual_load_fcst",
        "de_residual_load_fcst",
    }


def test_capture_subset_is_identical_under_frozen_residual_builder(
    tmp_path: Path,
) -> None:
    kwargs = _inputs(tmp_path)
    full = kwargs["fresh_features"].drop(columns="hour_sin").copy()
    full["known_hour_sin"] = np.sin(np.arange(len(full)))
    full["price_lag_24h"] = np.arange(len(full), dtype=float) + 100.0
    selected, _sources, _provenance, audit = prepare_supported_capture_inputs(
        fresh_features=full,
        required_pit_aliases=(
            "fr_residual_load_fcst",
            "de_residual_load_fcst",
        ),
        pit_freshness=kwargs["pit_sources"],
    )
    proof = prove_frozen_builder_subset_equivalence(
        full_features=full,
        captured_features=selected,
        chronos_live=kwargs["chronos_live"],
        delivery_timezone="Europe/Paris",
        primary_country="FR",
    )
    assert proof["excluded_by_frozen_builder"] is True
    assert proof["excluded_features"] == ["price_lag_24h"]
    assert proof["maximum_meta_feature_difference"] == 0.0
    assert audit["historical_price_features_excluded"] == ["price_lag_24h"]
