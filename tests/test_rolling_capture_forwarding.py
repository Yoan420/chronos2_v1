from __future__ import annotations

from pathlib import Path

import pandas as pd

import chronos2_hourly.rolling_capture as rolling_capture


def test_subset_equivalence_normalizes_only_meta_feature_column_order(
    monkeypatch,
) -> None:
    index = pd.date_range("2026-08-31", periods=2, freq="h", tz="UTC")
    full = pd.DataFrame({"full_only": [1.0, 2.0]}, index=index)
    captured = pd.DataFrame({"captured": [1.0, 2.0]}, index=index)
    chronos = pd.DataFrame(
        {"q10": [10.0, 11.0], "q50": [20.0, 21.0], "q90": [30.0, 31.0]},
        index=index,
    )

    def fake_builder(features, _experts, **_options):
        if "full_only" in features.columns:
            return pd.DataFrame(
                {"alpha": [1.0, 2.0], "beta": [3.0, 4.0]}, index=index
            )
        return pd.DataFrame(
            {"beta": [3.0, 4.0], "alpha": [1.0, 2.0]}, index=index
        )

    monkeypatch.setattr(
        "chronos2_hourly.models.residual_corrector.build_residual_meta_features",
        fake_builder,
    )

    audit = rolling_capture.prove_frozen_builder_subset_equivalence(
        full_features=full,
        captured_features=captured,
        chronos_live=chronos,
        delivery_timezone="Europe/Paris",
        primary_country="FR",
    )

    assert audit["meta_feature_column_order_normalized"] is True
    assert audit["maximum_meta_feature_difference"] == 0.0


def test_isolated_capture_forwards_target_source_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed: dict[str, Path] = {}

    def fake_finalize(**kwargs):
        observed["finalize"] = Path(kwargs["target_source_path"])
        return tmp_path / "finalized", {"status": "complete"}

    def fake_write(**kwargs):
        observed["write"] = Path(kwargs["target_source_path"])
        return tmp_path / "pending", {"status": "target_pending"}

    monkeypatch.setattr(
        rolling_capture,
        "finalize_target_pending_candidate",
        fake_finalize,
    )
    monkeypatch.setattr(
        rolling_capture,
        "write_target_pending_candidate",
        fake_write,
    )
    target_source = tmp_path / "target.csv"

    result = rolling_capture.capture_issued_live_block_isolated(
        capture_root=tmp_path / "capture",
        zone="FR",
        delivery_day="2026-08-26",
        delivery_timezone="Europe/Paris",
        full_features_for_equivalence=pd.DataFrame(),
        fresh_features=pd.DataFrame(),
        chronos_live=pd.DataFrame(),
        pit_sources={},
        feature_provenance={},
        expected_config_sha256="a" * 64,
        expected_base_bundle_sha256="b" * 64,
        target_series="power.price.da.fr",
        target_source_path=target_source,
        issued_live_archive=tmp_path / "issued",
        issued_live_forecast_filename="forecast_hourly_fr.csv",
        canonical_target=pd.Series(dtype=float),
    )

    assert result.status == "target_pending"
    assert observed == {"finalize": target_source, "write": target_source}
