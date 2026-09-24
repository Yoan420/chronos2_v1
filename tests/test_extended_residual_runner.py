from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from run_extended_residual_hourly import (
    RECIPE,
    _assert_no_forbidden_forecast,
    _fit_inputs,
    _native_frame_from_source,
    _new_corrector,
    _publish_staging,
    _validate_source_artifacts,
    _validate_complete_days,
)


def test_frozen_recipe_matches_screen_hyperparameters() -> None:
    model = _new_corrector(threads=3)

    assert RECIPE == "blend_cat_hgb_w0.50"
    assert model.weights == {"cat_v1": 0.5, "hgb31": 0.5}
    assert model.max_abs_correction == 40.0
    cat = model.components["cat_v1"]
    assert cat.backend == "catboost"
    assert cat.iterations == 700
    assert cat.depth == 6
    assert cat.learning_rate == 0.03
    assert cat.l2_leaf_reg == 15.0
    assert cat.thread_count == 3
    assert cat.max_abs_correction is None
    hgb = model.components["hgb31"]
    assert hgb.backend == "sklearn"
    assert hgb.iterations == 550
    assert hgb.depth == 5  # 2**5 - 1 = 31 leaves
    assert hgb.learning_rate == 0.035
    assert hgb.l2_leaf_reg == 60.0
    assert hgb.min_samples_leaf == 30
    assert hgb.sklearn_early_stopping is False
    assert hgb.max_abs_correction is None
    assert cat.feature_builder_options == hgb.feature_builder_options


@pytest.mark.parametrize(
    ("zone", "timezone"),
    [
        ("DE", "Europe/Berlin"),
        ("BE", "Europe/Brussels"),
        ("NL", "Europe/Amsterdam"),
        ("ES", "Europe/Madrid"),
    ],
)
def test_extended_corrector_uses_the_zone_calendar_contract(
    zone: str,
    timezone: str,
) -> None:
    model = _new_corrector(
        threads=1,
        timezone=timezone,
        primary_country=zone,
    )
    for component in model.components.values():
        options = component.feature_builder_options
        assert options["timezone"] == timezone
        assert options["rich_calendar_primary_country"] == zone


@pytest.mark.parametrize("timezone", ["Europe/Berlin", "Europe/Madrid"])
def test_complete_days_use_the_delivery_timezone_on_dst(timezone: str) -> None:
    days = list(pd.date_range("2025-03-29", "2025-03-31", freq="D"))
    index = pd.DatetimeIndex([], tz="UTC")
    for day in days:
        start = day.tz_localize(timezone)
        stop = (day + pd.Timedelta(days=1)).tz_localize(timezone)
        index = index.append(pd.date_range(start, stop, freq="h", inclusive="left").tz_convert("UTC"))
    assert _validate_complete_days(index, expected_days=3, timezone=timezone) == [
        day.date() for day in days
    ]


def test_forbidden_legacy_forecast_is_rejected_case_insensitively() -> None:
    _assert_no_forbidden_forecast(
        pd.DataFrame({"chronos2__q50": [1.0]}),
        name="safe",
    )
    with pytest.raises(ValueError, match="forecast interdit"):
        _assert_no_forbidden_forecast(
            pd.DataFrame({"StOrM__q50": [1.0]}),
            name="unsafe",
        )


def test_source_native_frame_and_extended_fit_are_chronological() -> None:
    external_index = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    existing_index = pd.date_range("2024-01-02", periods=2, freq="h", tz="UTC")
    source = pd.DataFrame(
        {
            "chronos2__q10": [1.0, 2.0],
            "chronos2__q50": [2.0, 3.0],
            "chronos2__q90": [3.0, 4.0],
            "actual": [2.5, 3.5],
        },
        index=existing_index,
    )
    existing = _native_frame_from_source(source)
    external = pd.DataFrame(
        {
            "q10": [0.0, 1.0, 2.0],
            "q50": [1.0, 2.0, 3.0],
            "q90": [2.0, 3.0, 4.0],
            "actual": [1.5, 2.5, 3.5],
        },
        index=external_index,
    )
    full_index = external_index.append(existing_index)
    features = pd.DataFrame(
        {"safe_feature": np.arange(len(full_index), dtype=float)},
        index=full_index,
    )

    X, y, base, experts = _fit_inputs(external, existing, features)

    assert X.index.equals(full_index)
    assert y.tolist() == [1.5, 2.5, 3.5, 2.5, 3.5]
    assert list(base) == ["q10", "q50", "q90"]
    assert list(experts) == [
        "chronos2__q10",
        "chronos2__q50",
        "chronos2__q90",
    ]
    with pytest.raises(ValueError, match="chevauchent"):
        _fit_inputs(external, existing.set_axis(external_index[:2]), features)


def test_source_validation_respects_the_float32_chronos_storage_contract(
    monkeypatch,
) -> None:
    index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    source = pd.DataFrame(
        {
            "fold_id": [1, 1],
            "chronos2__q10": np.asarray([369.3578, 1.1], dtype=np.float32),
            "chronos2__q50": np.asarray([528.4442, 2.2], dtype=np.float32),
            "chronos2__q90": np.asarray([969.0972, 3.3], dtype=np.float32),
            "actual": np.asarray([936.28, 2.5], dtype=np.float32),
        },
        index=index,
    )
    chronos = pd.DataFrame(
        {
            "q10": [369.3578, 1.1],
            "q50": [528.4442, 2.2],
            "q90": [969.0972, 3.3],
            "actual": [936.28, 2.5],
        },
        index=index,
    )
    live_index = pd.date_range("2025-01-02", periods=2, freq="h", tz="UTC")
    live = pd.DataFrame({"q10": [1.0, 2.0], "q50": [2.0, 3.0], "q90": [3.0, 4.0]}, index=live_index)
    forecast = pd.DataFrame(
        {
            "delivery_start_utc": live_index.astype(str),
            "chronos2__q10": live["q10"].to_numpy(),
            "chronos2__q50": live["q50"].to_numpy(),
            "chronos2__q90": live["q90"].to_numpy(),
        }
    )
    monkeypatch.setattr(
        "run_extended_residual_hourly._validate_complete_days",
        lambda *_args, **_kwargs: [pd.Timestamp("2025-01-01").date()],
    )

    validated = _validate_source_artifacts(
        source_backtest=source,
        source_forecast=forecast,
        chronos_oof=chronos,
        chronos_live=live,
    )

    assert validated.index.equals(index)


def test_publish_staging_requires_explicit_overwrite_and_swaps_atomically(
    tmp_path,
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    refused = tmp_path / ".run.tmp-refused"
    refused.mkdir()
    (refused / "new.txt").write_text("new", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite"):
        _publish_staging(refused, output, overwrite=False)
    assert (output / "old.txt").is_file()
    assert (refused / "new.txt").is_file()

    _publish_staging(refused, output, overwrite=True)
    assert not (output / "old.txt").exists()
    assert (output / "new.txt").read_text(encoding="utf-8") == "new"
    assert not list(tmp_path.glob(".run.previous-*"))
