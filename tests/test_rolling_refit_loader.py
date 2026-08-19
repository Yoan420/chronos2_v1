from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.rolling_refit import RollingRefitPolicy
from chronos2_hourly.rolling_refit_loader import (
    BLOCK_MANIFEST_NAME,
    BLOCK_MANIFEST_ROLE,
    CHECKSUM_MANIFEST_NAME,
    FilesystemBlockSpec,
    ROWS_FILE_NAME,
    ROWS_ROLE,
    RollingRefitFilesystemError,
    load_blocks,
    load_rolling_refit_filesystem,
)


CONFIG_SHA = "a" * 64
BASE_SHA = "b" * 64
TIMEZONE = "Europe/Paris"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _origin(day: date) -> pd.Timestamp:
    return (
        pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
    ).tz_localize("Europe/Paris").tz_convert("UTC")


def _bundle(
    root: Path,
    *,
    source_kind: str,
    start_day: date,
    days: int = 1,
    source_id: str | None = None,
    bootstrap_replay: bool = False,
    missing_pit_last_hours: int = 0,
    pit_timestamp_missing_with_value: bool = False,
    causal_timestamp_scope: str = "delivery_day_all_pit_inputs",
) -> FilesystemBlockSpec:
    directory = root / (source_id or f"{source_kind}-{start_day}")
    directory.mkdir()
    indexes = [
        local_delivery_day_index(
            start_day + pd.Timedelta(days=offset), timezone=TIMEZONE
        ).tz_convert("UTC")
        for offset in range(days)
    ]
    index = indexes[0]
    for value in indexes[1:]:
        index = index.append(value)
    local_days = pd.Index(index.tz_convert(TIMEZONE).date)
    origins = pd.Series(
        [_origin(value) for value in local_days], index=index
    )
    position = np.arange(len(index), dtype=float)
    pit_feature = 40.0 + np.cos(position / 12.0)
    snapshot: list[str | None] = [
        str(value - pd.Timedelta(hours=2)) for value in origins
    ]
    revision: list[str | None] = [
        str(value - pd.Timedelta(minutes=5)) for value in origins
    ]
    pit_present = np.ones(len(index), dtype=bool)
    if missing_pit_last_hours:
        pit_feature[-missing_pit_last_hours:] = np.nan
    if pit_timestamp_missing_with_value:
        snapshot[0] = None
    q50 = 50.0 + np.sin(position / 24.0)
    rows = pd.DataFrame(
        {
            "delivery_start_utc": [str(value) for value in index],
            "actual": q50 + 1.0,
            "q10": q50 - 5.0,
            "q50": q50,
            "q90": q50 + 5.0,
            "forecast_origin_utc": [str(value) for value in origins],
            "maximum_snapshot_time_utc": snapshot,
            "maximum_revision_time_utc": revision,
            "pit_inputs_present": pit_present,
            "known_hour_sin": np.sin(2 * np.pi * position / 24.0),
            "known_residual_load": pit_feature,
        }
    )
    rows_path = directory / ROWS_FILE_NAME
    rows.to_csv(rows_path, index=False)
    run_contract = {
        "sealed_oof": ("sealed_oof", "sealed_oof"),
        "pit_replay": ("pit_replay", "pit_reconstruction"),
        "issued_live": ("live_day_ahead", "issued_live"),
    }[source_kind]
    manifest = {
        "schema_version": "rolling-refit-block/v1",
        "status": "sealed",
        "source_kind": source_kind,
        "source_id": source_id or f"{source_kind}-{start_day}",
        "run_type": run_contract[0],
        "forecast_status": run_contract[1],
        "zone": "FR",
        "timezone": TIMEZONE,
        "delivery_start_day_local": str(local_days.min()),
        "delivery_end_day_local": str(local_days.max()),
        "n_rows": len(rows),
        "config_sha256": CONFIG_SHA,
        "base_bundle_sha256": BASE_SHA,
        "storm_used_as_feature": False,
        "mkonline_used_as_feature": False,
        "raw_chronos_only": True,
        "chronos_artifact_kind": "raw_pre_residual_quantiles",
        "prediction_mode": "autonomous_only",
        "bootstrap_replay": bootstrap_replay,
        "feature_columns": ["known_hour_sin", "known_residual_load"],
        "feature_provenance": {
            "known_hour_sin": "deterministic_calendar",
            "known_residual_load": "pit_asof",
        },
        "timestamp_evidence": {
            "granularity": "per_delivery_hour",
            "forecast_origin_column": "forecast_origin_utc",
            "maximum_snapshot_time_column": "maximum_snapshot_time_utc",
            "maximum_revision_time_column": "maximum_revision_time_utc",
            "pit_inputs_present_column": "pit_inputs_present",
            "source": "materialized_pit_selection",
        },
        "rows_role": ROWS_ROLE,
        "causal_timestamp_scope": causal_timestamp_scope,
    }
    manifest_path = directory / BLOCK_MANIFEST_NAME
    _write_json(manifest_path, manifest)
    checksums = {
        "algorithm": "sha256",
        "artifacts": [
            {
                "path": BLOCK_MANIFEST_NAME,
                "role": BLOCK_MANIFEST_ROLE,
                "size_bytes": manifest_path.stat().st_size,
                "sha256": _sha(manifest_path),
            },
            {
                "path": ROWS_FILE_NAME,
                "role": ROWS_ROLE,
                "size_bytes": rows_path.stat().st_size,
                "sha256": _sha(rows_path),
            },
        ],
    }
    checksum_path = directory / CHECKSUM_MANIFEST_NAME
    _write_json(checksum_path, checksums)
    return FilesystemBlockSpec(
        directory=directory,
        source_kind=source_kind,
        artifact_checksums_sha256=_sha(checksum_path),
    )


def _rewrite_manifest(
    spec: FilesystemBlockSpec,
    mutate: Callable[[dict], None],
) -> FilesystemBlockSpec:
    directory = Path(spec.directory)
    manifest_path = directory / BLOCK_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    _write_json(manifest_path, manifest)
    checksum_path = directory / CHECKSUM_MANIFEST_NAME
    checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
    entry = next(
        value
        for value in checksums["artifacts"]
        if value["role"] == BLOCK_MANIFEST_ROLE
    )
    entry["size_bytes"] = manifest_path.stat().st_size
    entry["sha256"] = _sha(manifest_path)
    _write_json(checksum_path, checksums)
    return FilesystemBlockSpec(
        directory=directory,
        source_kind=spec.source_kind,
        artifact_checksums_sha256=_sha(checksum_path),
    )


def _rewrite_rows(
    spec: FilesystemBlockSpec,
    mutate: Callable[[pd.DataFrame], None],
) -> FilesystemBlockSpec:
    directory = Path(spec.directory)
    rows_path = directory / ROWS_FILE_NAME
    rows = pd.read_csv(rows_path)
    mutate(rows)
    rows.to_csv(rows_path, index=False)
    checksum_path = directory / CHECKSUM_MANIFEST_NAME
    checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
    entry = next(
        value for value in checksums["artifacts"] if value["role"] == ROWS_ROLE
    )
    entry["size_bytes"] = rows_path.stat().st_size
    entry["sha256"] = _sha(rows_path)
    _write_json(checksum_path, checksums)
    return FilesystemBlockSpec(
        directory=directory,
        source_kind=spec.source_kind,
        artifact_checksums_sha256=_sha(checksum_path),
    )


def _load(
    specs: list[FilesystemBlockSpec],
    *,
    forecast_day: date,
    window_days: int,
    replay_bound: int = 0,
    bootstrap_end: date | None = None,
):
    return load_rolling_refit_filesystem(
        specs,
        zone="FR",
        forecast_delivery_day=forecast_day,
        delivery_timezone=TIMEZONE,
        expected_config_sha256=CONFIG_SHA,
        expected_base_bundle_sha256=BASE_SHA,
        rolling_policy=RollingRefitPolicy(window_days=window_days),
        max_bootstrap_replay_days=replay_bound,
        bootstrap_replay_end_day=bootstrap_end,
    )


def test_loads_sealed_replay_and_live_with_anchored_checksums(tmp_path: Path) -> None:
    specs = [
        _bundle(
            tmp_path,
            source_kind="sealed_oof",
            start_day=date(2026, 1, 1),
        ),
        _bundle(
            tmp_path,
            source_kind="pit_replay",
            start_day=date(2026, 1, 2),
            bootstrap_replay=True,
        ),
        _bundle(
            tmp_path,
            source_kind="issued_live",
            start_day=date(2026, 1, 3),
        ),
    ]
    result = _load(
        specs,
        forecast_day=date(2026, 1, 4),
        window_days=3,
        replay_bound=1,
        bootstrap_end=date(2026, 1, 2),
    )

    assert len(result.blocks) == 3
    assert len(result.selection.X) == 72
    assert result.audit["external_checksum_anchors_verified"] is True
    assert result.audit["consumed_files_rehashed"] is True
    assert result.audit["bootstrap_replay_days"] == ["2026-01-02"]
    assert result.audit["storm_used_as_feature"] is False
    assert result.audit["mkonline_used_as_feature"] is False
    loaded_blocks = load_blocks(
        specs,
        zone="FR",
        forecast_delivery_day=date(2026, 1, 4),
        delivery_timezone=TIMEZONE,
        expected_config_sha256=CONFIG_SHA,
        expected_base_bundle_sha256=BASE_SHA,
        rolling_policy=RollingRefitPolicy(window_days=3),
        max_bootstrap_replay_days=1,
        bootstrap_replay_end_day=date(2026, 1, 2),
    )
    assert [block.source_id for block in loaded_blocks] == [
        block.source_id for block in result.blocks
    ]


def test_daily_scope_covers_raw_missing_tail_with_real_daily_maxima(
    tmp_path: Path,
) -> None:
    spec = _bundle(
        tmp_path,
        source_kind="sealed_oof",
        start_day=date(2026, 1, 1),
        missing_pit_last_hours=2,
    )
    result = _load([spec], forecast_day=date(2026, 1, 2), window_days=1)

    assert result.selection.audit["pit_input_missing_hours"] == 0
    assert result.selection.pit_inputs_present.iloc[-2:].eq(True).all()
    assert result.selection.X["known_residual_load"].iloc[-2:].isna().all()
    block_audit = result.audit["blocks"][0]
    assert block_audit["pit_input_missing_hours"] == 0
    assert (
        block_audit["causal_timestamp_scope"]
        == "delivery_day_all_pit_inputs"
    )


def test_missing_timestamp_with_pit_value_fails_closed(tmp_path: Path) -> None:
    spec = _bundle(
        tmp_path,
        source_kind="sealed_oof",
        start_day=date(2026, 1, 1),
        pit_timestamp_missing_with_value=True,
    )
    with pytest.raises(RollingRefitFilesystemError, match="required iff"):
        _load([spec], forecast_day=date(2026, 1, 2), window_days=1)


def test_modified_consumed_rows_are_detected(tmp_path: Path) -> None:
    spec = _bundle(
        tmp_path,
        source_kind="sealed_oof",
        start_day=date(2026, 1, 1),
    )
    rows_path = Path(spec.directory) / ROWS_FILE_NAME
    rows_path.write_bytes(rows_path.read_bytes() + b"tamper")
    with pytest.raises(RollingRefitFilesystemError, match="size mismatch"):
        _load([spec], forecast_day=date(2026, 1, 2), window_days=1)


def test_wrong_external_checksum_anchor_is_rejected(tmp_path: Path) -> None:
    spec = _bundle(
        tmp_path,
        source_kind="sealed_oof",
        start_day=date(2026, 1, 1),
    )
    untrusted = FilesystemBlockSpec(
        directory=spec.directory,
        source_kind=spec.source_kind,
        artifact_checksums_sha256="f" * 64,
    )
    with pytest.raises(RollingRefitFilesystemError, match="trust anchor"):
        _load([untrusted], forecast_day=date(2026, 1, 2), window_days=1)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("zone", "BE", "zone"),
        ("timezone", "Europe/Brussels", "timezone"),
        ("status", "complete", "status"),
        ("config_sha256", "c" * 64, "config_sha256"),
        ("base_bundle_sha256", "d" * 64, "base_bundle_sha256"),
        ("storm_used_as_feature", True, "storm_used_as_feature"),
        ("mkonline_used_as_feature", True, "mkonline_used_as_feature"),
        ("raw_chronos_only", False, "raw_chronos_only"),
        ("prediction_mode", "blended", "prediction_mode"),
        ("causal_timestamp_scope", "per_delivery_hour", "causal_timestamp_scope"),
    ],
)
def test_metadata_contract_fails_closed(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    spec = _bundle(
        tmp_path,
        source_kind="sealed_oof",
        start_day=date(2026, 1, 1),
    )
    spec = _rewrite_manifest(spec, lambda manifest: manifest.__setitem__(field, value))
    with pytest.raises(RollingRefitFilesystemError, match=message):
        _load([spec], forecast_day=date(2026, 1, 2), window_days=1)


def test_bootstrap_replay_bound_is_explicit(tmp_path: Path) -> None:
    spec = _bundle(
        tmp_path,
        source_kind="pit_replay",
        start_day=date(2026, 1, 1),
        bootstrap_replay=True,
    )
    with pytest.raises(RollingRefitFilesystemError, match="bound is zero"):
        _load(
            [spec],
            forecast_day=date(2026, 1, 2),
            window_days=1,
            bootstrap_end=date(2026, 1, 1),
        )


def test_daily_scope_requires_constant_real_daily_maximum(tmp_path: Path) -> None:
    spec = _bundle(
        tmp_path,
        source_kind="sealed_oof",
        start_day=date(2026, 1, 1),
    )

    def change(rows: pd.DataFrame) -> None:
        rows.loc[0, "maximum_revision_time_utc"] = str(
            pd.Timestamp(rows.loc[0, "maximum_revision_time_utc"])
            - pd.Timedelta(seconds=1)
        )

    spec = _rewrite_rows(spec, change)
    with pytest.raises(RollingRefitFilesystemError, match="constant real daily"):
        _load([spec], forecast_day=date(2026, 1, 2), window_days=1)


def test_daily_scope_never_allows_false_pit_mask(tmp_path: Path) -> None:
    spec = _bundle(
        tmp_path,
        source_kind="sealed_oof",
        start_day=date(2026, 1, 1),
    )

    def change(rows: pd.DataFrame) -> None:
        rows.loc[0, "pit_inputs_present"] = False
        rows.loc[0, "maximum_snapshot_time_utc"] = np.nan
        rows.loc[0, "maximum_revision_time_utc"] = np.nan

    spec = _rewrite_rows(spec, change)
    with pytest.raises(
        RollingRefitFilesystemError,
        match="requires pit_inputs_present=true",
    ):
        _load([spec], forecast_day=date(2026, 1, 2), window_days=1)


def test_non_bootstrap_replay_does_not_consume_bootstrap_allowance(
    tmp_path: Path,
) -> None:
    spec = _bundle(
        tmp_path,
        source_kind="pit_replay",
        start_day=date(2026, 1, 1),
        bootstrap_replay=False,
    )
    result = _load([spec], forecast_day=date(2026, 1, 2), window_days=1)
    assert result.audit["bootstrap_replay_days_count"] == 0


def test_bootstrap_replay_requires_explicit_suffix_end(tmp_path: Path) -> None:
    spec = _bundle(
        tmp_path,
        source_kind="pit_replay",
        start_day=date(2026, 1, 1),
        bootstrap_replay=True,
    )
    with pytest.raises(RollingRefitFilesystemError, match="end_day is None"):
        _load(
            [spec],
            forecast_day=date(2026, 1, 2),
            window_days=1,
            replay_bound=1,
        )


def test_bootstrap_days_must_be_contiguous_suffix(tmp_path: Path) -> None:
    specs = [
        _bundle(
            tmp_path,
            source_kind="pit_replay",
            start_day=date(2026, 1, 1),
            bootstrap_replay=True,
        ),
        _bundle(
            tmp_path,
            source_kind="sealed_oof",
            start_day=date(2026, 1, 2),
        ),
        _bundle(
            tmp_path,
            source_kind="pit_replay",
            start_day=date(2026, 1, 3),
            bootstrap_replay=True,
        ),
    ]
    with pytest.raises(RollingRefitFilesystemError, match="contiguous suffix"):
        _load(
            specs,
            forecast_day=date(2026, 1, 4),
            window_days=3,
            replay_bound=3,
            bootstrap_end=date(2026, 1, 3),
        )


def test_old_bootstrap_day_cannot_hide_below_count_bound(tmp_path: Path) -> None:
    specs = [
        _bundle(
            tmp_path,
            source_kind="pit_replay",
            start_day=date(2026, 1, 1),
            bootstrap_replay=True,
        ),
        _bundle(
            tmp_path,
            source_kind="sealed_oof",
            start_day=date(2026, 1, 2),
            days=2,
        ),
    ]
    with pytest.raises(RollingRefitFilesystemError, match="bounded suffix"):
        _load(
            specs,
            forecast_day=date(2026, 1, 4),
            window_days=3,
            replay_bound=1,
            bootstrap_end=date(2026, 1, 3),
        )


def test_future_bootstrap_block_is_rejected_even_outside_window(
    tmp_path: Path,
) -> None:
    specs = [
        _bundle(
            tmp_path,
            source_kind="sealed_oof",
            start_day=date(2026, 1, 1),
        ),
        _bundle(
            tmp_path,
            source_kind="pit_replay",
            start_day=date(2026, 1, 2),
            bootstrap_replay=True,
        ),
    ]
    with pytest.raises(RollingRefitFilesystemError, match="strictly before forecast"):
        _load(
            specs,
            forecast_day=date(2026, 1, 2),
            window_days=1,
            replay_bound=1,
            bootstrap_end=date(2026, 1, 1),
        )


def test_legacy_aggregate_archive_reports_precise_blocker(tmp_path: Path) -> None:
    directory = tmp_path / "legacy"
    directory.mkdir()
    legacy_manifest = directory / "run_manifest.json"
    _write_json(legacy_manifest, {"input_diagnostics": {"first_revision": "x"}})
    checksums = {
        "algorithm": "sha256",
        "artifacts": [
            {
                "path": legacy_manifest.name,
                "role": "run_manifest",
                "size_bytes": legacy_manifest.stat().st_size,
                "sha256": _sha(legacy_manifest),
            }
        ],
    }
    checksum_path = directory / CHECKSUM_MANIFEST_NAME
    _write_json(checksum_path, checksums)
    spec = FilesystemBlockSpec(
        directory=directory,
        source_kind="sealed_oof",
        artifact_checksums_sha256=_sha(checksum_path),
    )
    with pytest.raises(
        RollingRefitFilesystemError,
        match="Aggregate run_manifest/input_diagnostics ranges cannot be expanded",
    ):
        _load([spec], forecast_day=date(2026, 1, 2), window_days=1)


def test_forbidden_source_id_is_rejected_even_when_rows_are_raw(tmp_path: Path) -> None:
    spec = _bundle(
        tmp_path,
        source_kind="sealed_oof",
        start_day=date(2026, 1, 1),
    )
    spec = _rewrite_manifest(
        spec,
        lambda manifest: manifest.__setitem__("source_id", "mkonline-history"),
    )
    with pytest.raises(RollingRefitFilesystemError, match="forbidden"):
        _load([spec], forecast_day=date(2026, 1, 2), window_days=1)
