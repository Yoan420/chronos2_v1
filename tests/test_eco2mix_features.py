from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.hourly_contract import build_delivery_metadata
from materialize_eco2mix_features import (
    SOURCE_FIELD_ALIASES,
    TIMEZONE,
    _build_session,
    _fetch,
    _source_window_utc,
    build_features,
)


BASE_DAY = date(2024, 1, 1)


def _profile_value(local_value: object, offset: float = 0.0) -> float:
    local = local_value
    return float(
        (local.date() - BASE_DAY).days * 100
        + local.hour * 2
        + int(local.fold)
        + offset
    )


def _raw_for_local_days(start_day: str, end_day: str) -> pd.DataFrame:
    start_local = pd.Timestamp(start_day).tz_localize(TIMEZONE)
    end_local = (
        pd.Timestamp(end_day) + pd.Timedelta(days=1)
    ).tz_localize(TIMEZONE)
    index = pd.date_range(
        start_local.tz_convert("UTC"),
        end_local.tz_convert("UTC"),
        freq="15min",
        inclusive="left",
    )
    local_values = index.tz_convert(TIMEZONE).to_pydatetime()
    raw = pd.DataFrame({"date_heure": index})
    raw["prevision_j1"] = [_profile_value(value) for value in local_values]
    half_hour = index.minute.isin([0, 30])
    for position, source in enumerate(SOURCE_FIELD_ALIASES):
        if source == "prevision_j1":
            continue
        values = np.array(
            [
                _profile_value(value, offset=(position + 1) * 10_000.0)
                for value in local_values
            ],
            dtype=float,
        )
        values[~half_hour] = np.nan
        raw[source] = values
    return raw


def _row_for_local_hour(
    frame: pd.DataFrame,
    *,
    hour: int,
    fold: int = 0,
) -> pd.Timestamp:
    metadata = build_delivery_metadata(frame.index, timezone=TIMEZONE)
    matches = metadata.index[
        metadata["local_hour"].eq(hour) & metadata["fold"].eq(fold)
    ]
    assert len(matches) == 1
    return matches[0]


def test_builder_exports_only_d2_d7_and_marks_revision_latest() -> None:
    raw = _raw_for_local_days("2025-01-02", "2025-01-08")

    result = build_features(raw, start_day="2025-01-09", end_day="2025-01-10")

    assert len(result) == 48
    load_profiles = sorted(
        column
        for column in result
        if column.startswith("eco2mix_load_fcst_j1_profile_")
        and not column.endswith("_flag")
    )
    assert load_profiles == [
        "eco2mix_load_fcst_j1_profile_d2",
        "eco2mix_load_fcst_j1_profile_d7",
    ]
    assert not any("price" in column or "target" in column for column in result)
    assert result["eco2mix_exploratory_non_pit_flag"].eq(1).all()
    assert result["eco2mix_source_revision_latest_flag"].eq(1).all()
    assert result.attrs["data_classification"] == "exploratory_non_pit"
    assert result.attrs["same_day_prevision_j1_exported"] is False

    first = result.index[0]
    target_local = first.tz_convert(TIMEZONE).to_pydatetime()
    expected_d2 = _profile_value(target_local - timedelta(days=2))
    expected_d7 = _profile_value(target_local - timedelta(days=7))
    assert result.at[first, "eco2mix_load_fcst_j1_profile_d2"] == expected_d2
    assert result.at[first, "eco2mix_load_fcst_j1_profile_d7"] == expected_d7
    assert result["eco2mix_feature_missing_count"].eq(0).all()


def test_spring_dst_missing_hour_is_nan_and_flagged_without_interpolation() -> None:
    # Target 2024-04-02 is a 24-hour day; its D-2 source is the 23-hour
    # transition day 2024-03-31, which has no local 02:00 hour.
    raw = _raw_for_local_days("2024-03-26", "2024-03-31")
    result = build_features(raw, start_day="2024-04-02", end_day="2024-04-02")
    hour_2 = _row_for_local_hour(result, hour=2)
    hour_3 = _row_for_local_hour(result, hour=3)
    feature = "eco2mix_load_fcst_j1_profile_d2"

    assert np.isnan(result.at[hour_2, feature])
    assert result.at[hour_2, f"{feature}_missing_flag"] == 1
    assert np.isfinite(result.at[hour_3, feature])
    assert result.at[hour_3, f"{feature}_missing_flag"] == 0
    assert result["eco2mix_day_shape_mismatch_d2_flag"].eq(1).all()
    assert result["eco2mix_day_shape_mismatch_d7_flag"].eq(0).all()


def test_autumn_repeated_fold_is_not_fabricated_from_normal_source_day() -> None:
    # Target 2024-10-27 contains both fold=0 and fold=1 at local 02:00;
    # D-2/D-7 source days are normal and cannot provide fold=1.
    raw = _raw_for_local_days("2024-10-20", "2024-10-25")
    result = build_features(raw, start_day="2024-10-27", end_day="2024-10-27")
    fold_0 = _row_for_local_hour(result, hour=2, fold=0)
    fold_1 = _row_for_local_hour(result, hour=2, fold=1)

    for lag in (2, 7):
        feature = f"eco2mix_consommation_profile_d{lag}"
        assert np.isfinite(result.at[fold_0, feature])
        assert np.isnan(result.at[fold_1, feature])
        assert result.at[fold_1, f"{feature}_missing_flag"] == 1
        assert result[f"eco2mix_day_shape_mismatch_d{lag}_flag"].eq(1).all()


def test_partial_quarter_hour_is_nan_instead_of_partial_average() -> None:
    raw = _raw_for_local_days("2025-01-02", "2025-01-07")
    local = raw["date_heure"].dt.tz_convert(TIMEZONE)
    missing_quarter = (
        local.dt.date.eq(date(2025, 1, 7))
        & local.dt.hour.eq(5)
        & local.dt.minute.eq(15)
    )
    raw = raw.loc[~missing_quarter].copy()
    result = build_features(raw, start_day="2025-01-09", end_day="2025-01-09")
    hour_5 = _row_for_local_hour(result, hour=5)

    load_feature = "eco2mix_load_fcst_j1_profile_d2"
    realised_feature = "eco2mix_consommation_profile_d2"
    assert np.isnan(result.at[hour_5, load_feature])
    assert result.at[hour_5, f"{load_feature}_missing_flag"] == 1
    # Realised values are expected at :00/:30, so removing the empty :15 row
    # does not make that separate half-hour series incomplete.
    assert np.isfinite(result.at[hour_5, realised_feature])
    assert result.at[hour_5, f"{realised_feature}_missing_flag"] == 0


def test_duplicate_or_misaligned_timestamps_are_rejected() -> None:
    raw = _raw_for_local_days("2025-01-02", "2025-01-07")
    duplicated = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate Eco2Mix timestamps"):
        build_features(duplicated, start_day="2025-01-09", end_day="2025-01-09")

    misaligned = raw.copy()
    misaligned.loc[0, "date_heure"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="quarter-hour aligned"):
        build_features(misaligned, start_day="2025-01-09", end_day="2025-01-09")


def test_source_query_is_minimal_and_respects_civil_dst_boundaries() -> None:
    start, end = _source_window_utc("2024-04-02", "2024-04-02")

    # D-7 starts at 2024-03-26 00:00 CET; D-2 ends at the midnight after
    # 2024-03-31, by then expressed in CEST.
    assert start == pd.Timestamp("2024-03-25T23:00:00Z")
    assert end == pd.Timestamp("2024-03-31T22:00:00Z")


def test_requests_session_uses_ca_bundle_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    certificate = tmp_path / "corporate-ca.pem"
    certificate.write_text("synthetic test certificate", encoding="utf-8")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(certificate))
    monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)

    session, variable = _build_session()
    try:
        assert variable == "REQUESTS_CA_BUNDLE"
        assert session.trust_env is True
        assert Path(session.verify) == certificate.resolve()
    finally:
        session.close()


def test_fetch_paginates_one_bounded_chunk_without_network() -> None:
    class Response:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self.payload

    class Session:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.responses = [
                Response(
                    {
                        "total_count": 3,
                        "results": [
                            {"date_heure": "2025-01-02T00:00:00Z"},
                            {"date_heure": "2025-01-02T00:15:00Z"},
                        ],
                    }
                ),
                Response(
                    {
                        "total_count": 3,
                        "results": [
                            {"date_heure": "2025-01-02T00:30:00Z"},
                        ],
                    }
                ),
            ]

        def get(self, url: str, **kwargs: object) -> Response:
            self.calls.append({"url": url, **kwargs})
            return self.responses.pop(0)

    session = Session()
    fetched = _fetch(
        "2025-01-09",
        "2025-01-09",
        page_size=2,
        chunk_days=31,
        transport="records",
        session=session,  # type: ignore[arg-type]
    )

    assert len(fetched) == 3
    assert len(session.calls) == 2
    assert session.calls[0]["params"]["offset"] == 0  # type: ignore[index]
    assert session.calls[1]["params"]["offset"] == 2  # type: ignore[index]
    where = session.calls[0]["params"]["where"]  # type: ignore[index]
    assert "2025-01-01T23:00:00Z" in where
    assert "2025-01-07T23:00:00Z" in where


def test_fetch_csv_transport_uses_one_request_per_chunk() -> None:
    class Response:
        text = (
            "date_heure,prevision_j1,consommation,nucleaire,eolien,solaire,"
            "hydraulique,pompage,ech_physiques\n"
            "2025-01-02T00:00:00Z,1,2,3,4,5,6,7,8\n"
        )

        def raise_for_status(self) -> None:
            return None

    class Session:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def get(self, url: str, **kwargs: object) -> Response:
            self.calls.append({"url": url, **kwargs})
            return Response()

    session = Session()
    fetched = _fetch(
        "2025-01-09",
        "2025-01-09",
        page_size=100,
        chunk_days=31,
        transport="csv",
        session=session,  # type: ignore[arg-type]
    )

    assert len(fetched) == 1
    assert len(session.calls) == 1
    assert str(session.calls[0]["url"]).endswith("/exports/csv")
    params = session.calls[0]["params"]  # type: ignore[assignment]
    assert params["delimiter"] == ","  # type: ignore[index]
    assert params["use_labels"] == "false"  # type: ignore[index]


def test_fetch_deduplicates_identical_and_quarantines_conflicts() -> None:
    header = (
        "date_heure,prevision_j1,consommation,nucleaire,eolien,solaire,"
        "hydraulique,pompage,ech_physiques\n"
    )

    class Response:
        def __init__(self, value: int) -> None:
            self.text = header + (
                f"2025-01-02T00:00:00Z,{value},2,3,4,5,6,7,8\n"
            )

        def raise_for_status(self) -> None:
            return None

    class Session:
        def __init__(self, values: list[int]) -> None:
            self.responses = [Response(value) for value in values]

        def get(self, url: str, **kwargs: object) -> Response:
            return self.responses.pop(0)

    identical = _fetch(
        "2025-01-09", "2025-01-10", page_size=100, chunk_days=1,
        transport="csv", session=Session([1, 1, 1, 1, 1, 1, 1]),  # type: ignore[arg-type]
    )
    assert len(identical) == 1
    assert identical.attrs["identical_duplicate_rows_removed"] == 6

    conflicts = _fetch(
        "2025-01-09", "2025-01-10", page_size=100, chunk_days=1,
        transport="csv", session=Session([1, 1, 1, 1, 1, 1, 2]),  # type: ignore[arg-type]
    )
    assert conflicts.empty
    assert conflicts.attrs["conflicting_duplicate_timestamps_quarantined"] == 1
    assert conflicts.attrs["conflicting_duplicate_rows_quarantined"] == 7
