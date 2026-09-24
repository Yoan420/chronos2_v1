from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import ssl

import certifi
import httpx
import numpy as np
import pandas as pd
import pytest

import materialize_jao_core_flowbased as flowbased_materializer
from chronos2_hourly.jao_flowbased import (
    FLOWBASED_FEATURE_COLUMNS,
    JaoCoreClient,
    JaoFetchResult,
    JaoFlowBasedError,
    assemble_flowbased_feature_store,
    build_causal_empty_day_fallback,
    build_hourly_flowbased_features,
    expected_cutoff_utc,
    local_day_utc_bounds,
    normalise_initial_computation,
    write_daily_flowbased_bundle,
)
from run_kalman_flowbased_experiment import (
    MODEL_KEY,
    KalmanFlowBasedError,
    _config_objects,
    _load_flow_store,
    _metrics,
    _neutralize_unavailable_flowbased_hours,
    _validate_preflight_support,
    _validate_safe_output,
)


def _raw_row(timestamp: pd.Timestamp, identifier: int) -> dict[str, object]:
    return {
        "id": identifier,
        "dateTimeUtc": timestamp.isoformat().replace("+00:00", "Z"),
        "tso": "RTE",
        "cneName": f"CNE {identifier}",
        "cneEic": f"EIC-{identifier}",
        "cneStatus": "no CRA",
        "direction": "DIRECT",
        "hubFrom": "FR",
        "hubTo": "DE",
        "contName": "BASE",
        "contingencies": [],
        "presolved": True,
        "cnec": True,
        "ram": 400.0 + identifier,
        "fmax": 1000.0,
        "frm": 100.0,
        "frefInit": 0.0,
        "fcore": 0.0,
        "fall": 0.0,
        "fuaf": 0.0,
        "ptdf_ALBE": 0.16,
        "ptdf_ALDE": -0.14,
        "ptdf_AT": 0.2,
        "ptdf_BE": 0.1,
        "ptdf_CZ": -0.2,
        "ptdf_DE": -0.3,
        "ptdf_FR": 0.4,
        "ptdf_HR": 0.05,
        "ptdf_HU": -0.05,
        "ptdf_NL": -0.1,
        "ptdf_PL": -0.15,
        "ptdf_RO": 0.03,
        "ptdf_SI": 0.12,
        "ptdf_SK": -0.08,
    }


def test_materializer_rejects_a_missing_explicit_ca_before_dry_run(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-proxy-ca.pem"
    with pytest.raises(FileNotFoundError, match="Bundle CA introuvable"):
        flowbased_materializer.main(
            [
                "--end-day",
                "2026-09-03",
                "--ca-bundle",
                str(missing),
                "--dry-run",
            ]
        )


def test_materializer_rejects_an_invalid_explicit_ca(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "invalid-proxy-ca.pem"
    invalid.write_text("ceci n'est pas un certificat", encoding="utf-8")
    with pytest.raises(JaoFlowBasedError, match="Bundle CA invalide"):
        flowbased_materializer.main(
            [
                "--end-day",
                "2026-09-03",
                "--ca-bundle",
                str(invalid),
                "--dry-run",
            ]
        )


def test_materializer_uses_requests_ca_bundle_when_no_option_is_given() -> None:
    args = flowbased_materializer.build_parser().parse_args(
        ["--end-day", "2026-09-03"]
    )
    verify, source = flowbased_materializer._tls_configuration(
        args,
        environ={"REQUESTS_CA_BUNDLE": certifi.where()},
    )
    assert isinstance(verify, ssl.SSLContext)
    assert source.startswith("environment_REQUESTS_CA_BUNDLE:")


def test_explicit_ca_does_not_fall_back_to_environment(tmp_path: Path) -> None:
    args = flowbased_materializer.build_parser().parse_args(
        [
            "--end-day",
            "2026-09-03",
            "--ca-bundle",
            str(tmp_path / "missing-explicit.pem"),
        ]
    )
    with pytest.raises(FileNotFoundError, match="explicit_ca_bundle"):
        flowbased_materializer._tls_configuration(
            args,
            environ={"REQUESTS_CA_BUNDLE": certifi.where()},
        )


def _daily_fetch(day: date, *, last_modified: str) -> JaoFetchResult:
    start, end = local_day_utc_bounds(day)
    rows = tuple(
        _raw_row(timestamp, position + 1)
        for position, timestamp in enumerate(
            pd.date_range(start, end, freq="h", inclusive="left")
        )
    )
    return JaoFetchResult(
        endpoint="initialComputation",
        start_utc=start,
        end_utc=end,
        rows=rows,
        total_rows=len(rows),
        last_modified_utc=pd.Timestamp(last_modified),
        retrieved_at_utc=pd.Timestamp("2026-09-05T10:00:00Z"),
        filters={"Presolved": True},
        requests=1,
    )


def test_local_day_bounds_are_dst_safe() -> None:
    spring = pd.date_range(
        *local_day_utc_bounds(date(2026, 3, 29)), freq="h", inclusive="left"
    )
    autumn = pd.date_range(
        *local_day_utc_bounds(date(2026, 10, 25)), freq="h", inclusive="left"
    )
    assert len(spring) == 23
    assert len(autumn) == 25


def test_cutoff_is_eight_oclock_wall_time_across_dst_switches() -> None:
    assert expected_cutoff_utc(date(2026, 3, 30)) == pd.Timestamp(
        "2026-03-29T06:00:00Z"
    )
    assert expected_cutoff_utc(date(2026, 10, 26)) == pd.Timestamp(
        "2026-10-25T07:00:00Z"
    )


def test_client_paginates_and_rejects_duplicate_ids() -> None:
    rows = [
        {"id": number, "dateTimeUtc": "2026-08-01T00:00:00Z"}
        for number in (1, 2, 3)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        skip = int(request.url.params["Skip"])
        take = int(request.url.params["Take"])
        return httpx.Response(
            200,
            json={
                "lastModifiedOn": "2026-07-31T04:00:00Z",
                "data": rows[skip : skip + take],
                "totalRowsWithFilter": 3,
                "skip": skip,
                "take": take,
                "appliedFilter": {"presolved": True},
                "rejected": False,
            },
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = JaoCoreClient(
        client=http,
        page_size=2,
        request_interval_seconds=0.0,
        sleeper=lambda _: None,
    )
    result = client.fetch(
        "initialComputation",
        start_utc="2026-08-01T00:00:00Z",
        end_utc="2026-08-01T01:00:00Z",
        filters={"Presolved": True},
    )
    assert result.total_rows == 3
    assert result.requests == 4
    assert result.snapshot_verification_scans == 2
    assert [row["id"] for row in result.rows] == [1, 2, 3]


def test_client_restarts_the_whole_pagination_when_watermark_changes() -> None:
    request_skips: list[int] = []
    first_snapshot = [
        {"id": number, "dateTimeUtc": "2026-08-01T00:00:00Z"}
        for number in (1, 2, 3)
    ]
    stable_snapshot = [
        {"id": number, "dateTimeUtc": "2026-08-01T00:00:00Z"}
        for number in (101, 102, 103)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        skip = int(request.url.params["Skip"])
        take = int(request.url.params["Take"])
        request_skips.append(skip)
        attempt = (len(request_skips) - 1) // 2
        if attempt == 0:
            rows = first_snapshot
            modified = (
                "2026-07-31T04:00:00Z"
                if skip == 0
                else "2026-07-31T04:01:00Z"
            )
        else:
            rows = stable_snapshot
            modified = "2026-07-31T04:01:00Z"
        return httpx.Response(
            200,
            json={
                "lastModifiedOn": modified,
                "data": rows[skip : skip + take],
                "totalRowsWithFilter": 3,
                "skip": skip,
                "take": take,
                "appliedFilter": {"presolved": True},
                "rejected": False,
            },
        )

    client = JaoCoreClient(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        page_size=2,
        maximum_retries=1,
        request_interval_seconds=0.0,
        sleeper=lambda _: None,
    )
    result = client.fetch(
        "initialComputation",
        start_utc="2026-08-01T00:00:00Z",
        end_utc="2026-08-01T01:00:00Z",
        filters={"Presolved": True},
    )

    assert request_skips == [0, 2, 0, 2, 0, 2]
    assert result.requests == 6
    assert result.last_modified_utc == pd.Timestamp("2026-07-31T04:01:00Z")
    assert [row["id"] for row in result.rows] == [101, 102, 103]


def test_client_bounds_retries_when_pagination_watermark_never_stabilises() -> None:
    request_skips: list[int] = []
    rows = [
        {"id": number, "dateTimeUtc": "2026-08-01T00:00:00Z"}
        for number in (1, 2, 3)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        skip = int(request.url.params["Skip"])
        take = int(request.url.params["Take"])
        request_skips.append(skip)
        attempt = (len(request_skips) - 1) // 2
        modified = pd.Timestamp("2026-07-31T04:00:00Z") + pd.Timedelta(
            minutes=attempt * 2 + (1 if skip else 0)
        )
        return httpx.Response(
            200,
            json={
                "lastModifiedOn": modified.isoformat().replace("+00:00", "Z"),
                "data": rows[skip : skip + take],
                "totalRowsWithFilter": 3,
                "skip": skip,
                "take": take,
                "appliedFilter": {"presolved": True},
                "rejected": False,
            },
        )

    client = JaoCoreClient(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        page_size=2,
        maximum_retries=1,
        request_interval_seconds=0.0,
        sleeper=lambda _: None,
    )
    with pytest.raises(
        JaoFlowBasedError,
        match=r"pagination instable.*3 lecture.*2 changement",
    ):
        client.fetch(
            "initialComputation",
            start_utc="2026-08-01T00:00:00Z",
            end_utc="2026-08-01T01:00:00Z",
            filters={"Presolved": True},
        )

    assert request_skips == [0, 2, 0, 2, 0, 2]


def test_client_accepts_stable_page_specific_watermarks_conservatively() -> None:
    request_skips: list[int] = []
    rows = [
        {"id": number, "dateTimeUtc": "2026-08-01T00:00:00Z"}
        for number in (1, 2, 3)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        skip = int(request.url.params["Skip"])
        take = int(request.url.params["Take"])
        request_skips.append(skip)
        return httpx.Response(
            200,
            json={
                "lastModifiedOn": (
                    "2026-07-31T04:00:00Z"
                    if skip == 0
                    else "2026-07-31T04:01:00Z"
                ),
                "data": rows[skip : skip + take],
                "totalRowsWithFilter": 3,
                "skip": skip,
                "take": take,
                "appliedFilter": {"presolved": True},
                "rejected": False,
            },
        )

    client = JaoCoreClient(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        page_size=2,
        maximum_retries=0,
        request_interval_seconds=0.0,
        sleeper=lambda _: None,
    )
    result = client.fetch(
        "initialComputation",
        start_utc="2026-08-01T00:00:00Z",
        end_utc="2026-08-01T01:00:00Z",
        filters={"Presolved": True},
    )

    assert request_skips == [0, 2, 0, 2]
    assert result.requests == 4
    assert result.snapshot_verification_scans == 2
    assert result.last_modified_utc == pd.Timestamp("2026-07-31T04:01:00Z")


def test_client_requires_last_modified_on_every_initial_page() -> None:
    rows = [
        {"id": number, "dateTimeUtc": "2026-08-01T00:00:00Z"}
        for number in (1, 2, 3)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        skip = int(request.url.params["Skip"])
        take = int(request.url.params["Take"])
        payload = {
            "data": rows[skip : skip + take],
            "totalRowsWithFilter": 3,
            "skip": skip,
            "take": take,
            "appliedFilter": {"presolved": True},
            "rejected": False,
        }
        if skip == 0:
            payload["lastModifiedOn"] = "2026-07-31T04:00:00Z"
        return httpx.Response(200, json=payload)

    client = JaoCoreClient(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        page_size=2,
        request_interval_seconds=0.0,
        sleeper=lambda _: None,
    )
    with pytest.raises(JaoFlowBasedError, match="lastModifiedOn manque"):
        client.fetch(
            "initialComputation",
            start_utc="2026-08-01T00:00:00Z",
            end_utc="2026-08-01T01:00:00Z",
            filters={"Presolved": True},
        )


def test_client_allows_an_empty_initial_page_without_watermark() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [],
                "totalRowsWithFilter": 0,
                "skip": 0,
                "take": int(request.url.params["Take"]),
                "appliedFilter": {"presolved": True},
                "rejected": False,
            },
        )

    client = JaoCoreClient(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        page_size=5000,
        request_interval_seconds=0.0,
        sleeper=lambda _: None,
    )
    result = client.fetch(
        "initialComputation",
        start_utc="2026-08-01T00:00:00Z",
        end_utc="2026-08-02T00:00:00Z",
        filters={"Presolved": True},
    )
    assert result.total_rows == 0
    assert result.rows == ()
    assert result.last_modified_utc is None


def test_normalisation_and_hourly_features_are_strictly_pre_cutoff() -> None:
    day = date(2026, 8, 1)
    fetch = _daily_fetch(day, last_modified="2026-07-31T04:00:00Z")
    normalised, audit = normalise_initial_computation(fetch, delivery_day=day)
    features = build_hourly_flowbased_features(normalised, daily_audit=audit)
    assert audit["pit_eligible"] is True
    assert audit["pit_confidence"] == "historical_last_modified_only"
    assert audit["operational_pit_eligible"] is False
    assert len(normalised) == 24
    assert len(features) == 24
    assert set(FLOWBASED_FEATURE_COLUMNS).issubset(features)
    assert np.isfinite(
        features.loc[:, list(FLOWBASED_FEATURE_COLUMNS)].to_numpy(dtype=float)
    ).all()
    assert not any("shadow" in column.casefold() for column in features)


def test_late_last_modified_is_kept_for_audit_but_not_pit_eligible() -> None:
    fetch = _daily_fetch(
        date(2026, 8, 1), last_modified="2026-07-31T08:30:00Z"
    )
    _, audit = normalise_initial_computation(fetch, delivery_day=date(2026, 8, 1))
    assert audit["pit_eligible"] is False
    assert audit["causality_violations"] == 1


def test_quarter_hour_grid_and_external_constraints_are_preserved() -> None:
    day = date(2026, 8, 1)
    start, end = local_day_utc_bounds(day)
    rows = [
        _raw_row(timestamp, position + 1)
        for position, timestamp in enumerate(
            pd.date_range(start, end, freq="15min", inclusive="left")
        )
    ]
    external = dict(rows[0])
    external.update(
        {
            "id": len(rows) + 1,
            # The real JAO payload also sets this API flag to true for
            # technical external constraints; the descriptor carries the
            # modelling category.
            "cnec": True,
            "cneName": "ALEGrO external constraint",
            "ram": 275.0,
        }
    )
    rows.append(external)
    external_sentinel = dict(external)
    external_sentinel.update(
        {
            "id": len(rows) + 1,
            "cneName": "External Constraint NL_import",
            "ram": -1.0,
        }
    )
    rows.append(external_sentinel)
    equality = dict(external)
    equality.update(
        {
            "id": len(rows) + 1,
            "cneName": "Equality Constraint BE balance",
            "ram": -1.0,
        }
    )
    rows.append(equality)
    fetch = JaoFetchResult(
        endpoint="initialComputation",
        start_utc=start,
        end_utc=end,
        rows=tuple(rows),
        total_rows=len(rows),
        last_modified_utc=pd.Timestamp("2026-07-31T04:00:00Z"),
        retrieved_at_utc=pd.Timestamp("2026-09-05T10:00:00Z"),
        filters={"Presolved": True},
        requests=1,
    )
    normalised, audit = normalise_initial_computation(fetch, delivery_day=day)
    features = build_hourly_flowbased_features(normalised, daily_audit=audit)
    assert audit["mtu_minutes"] == 15
    assert audit["physical_mtus"] == 96
    assert audit["selected_external_or_equality_rows"] == 3
    assert audit["selected_external_rows"] == 2
    assert audit["selected_equality_rows"] == 1
    assert audit["external_negative_ram_sentinel_rows"] == 1
    assert features["flowbased_external_constraint_count"].sum() == pytest.approx(0.5)
    assert features.iloc[0]["flowbased_external_ram_min_mw"] == pytest.approx(275.0)
    assert features["flowbased_equality_constraint_count"].sum() == pytest.approx(
        0.25
    )
    assert np.isfinite(features["flowbased_alegro_ptdf_spread_p90"]).all()


@pytest.mark.parametrize(
    ("day", "expected_mtus", "expected_hours"),
    (
        (date(2026, 3, 29), 92, 23),
        (date(2026, 10, 25), 100, 25),
    ),
)
def test_quarter_hour_dst_days_are_complete(
    day: date, expected_mtus: int, expected_hours: int
) -> None:
    start, end = local_day_utc_bounds(day)
    rows = tuple(
        _raw_row(timestamp, position + 1)
        for position, timestamp in enumerate(
            pd.date_range(start, end, freq="15min", inclusive="left")
        )
    )
    fetch = JaoFetchResult(
        endpoint="initialComputation",
        start_utc=start,
        end_utc=end,
        rows=rows,
        total_rows=len(rows),
        last_modified_utc=expected_cutoff_utc(day) - pd.Timedelta(hours=1),
        retrieved_at_utc=pd.Timestamp("2026-11-01T10:00:00Z"),
        filters={"Presolved": True},
        requests=1,
    )
    normalised, audit = normalise_initial_computation(fetch, delivery_day=day)
    features = build_hourly_flowbased_features(normalised, daily_audit=audit)
    assert audit["physical_mtus"] == expected_mtus
    assert len(features) == expected_hours
    assert set(features["flowbased_source_mtu_count"]) == {4}


def test_quarter_hour_grid_marks_an_external_only_mtu_as_partial() -> None:
    day = date(2026, 8, 1)
    start, end = local_day_utc_bounds(day)
    rows = [
        _raw_row(timestamp, position + 1)
        for position, timestamp in enumerate(
            pd.date_range(start, end, freq="15min", inclusive="left")
        )
    ]
    rows[0]["cnec"] = False
    fetch = JaoFetchResult(
        endpoint="initialComputation",
        start_utc=start,
        end_utc=end,
        rows=tuple(rows),
        total_rows=len(rows),
        last_modified_utc=pd.Timestamp("2026-07-31T04:00:00Z"),
        retrieved_at_utc=pd.Timestamp("2026-09-05T10:00:00Z"),
        filters={"Presolved": True},
        requests=1,
    )
    normalised, audit = normalise_initial_computation(fetch, delivery_day=day)
    features = build_hourly_flowbased_features(normalised, daily_audit=audit)
    assert audit["missing_cnec_mtus"] == 1
    assert features.iloc[0]["flowbased_cnec_mtu_availability"] == pytest.approx(
        0.75
    )
    assert features.iloc[0]["flowbased_missing_mtu_share"] == pytest.approx(0.25)
    assert features.iloc[0]["flowbased_hour_imputed"] == 0.0


def test_missing_hourly_mtus_are_causally_imputed_and_flagged() -> None:
    day = date(2026, 9, 1)
    start, end = local_day_utc_bounds(day)
    missing_hours = {
        pd.Timestamp("2026-09-01T00:00:00Z"),
        pd.Timestamp("2026-09-01T06:00:00Z"),
        pd.Timestamp("2026-09-01T07:00:00Z"),
    }
    rows = tuple(
        _raw_row(timestamp, position + 1)
        for position, timestamp in enumerate(
            pd.date_range(start, end, freq="h", inclusive="left")
        )
        if timestamp not in missing_hours
    )
    fetch = JaoFetchResult(
        endpoint="initialComputation",
        start_utc=start,
        end_utc=end,
        rows=rows,
        total_rows=len(rows),
        last_modified_utc=pd.Timestamp("2026-08-31T04:00:00Z"),
        retrieved_at_utc=pd.Timestamp("2026-09-05T10:00:00Z"),
        filters={"Presolved": True},
        requests=1,
    )
    normalised, audit = normalise_initial_computation(fetch, delivery_day=day)
    features = build_hourly_flowbased_features(normalised, daily_audit=audit)
    indexed = features.set_index("value_time_utc")
    assert audit["missing_source_mtus"] == 3
    assert audit["missing_cnec_mtus"] == 3
    assert len(features) == 24
    assert indexed.loc[list(missing_hours), "flowbased_hour_imputed"].eq(1.0).all()
    assert indexed.loc[list(missing_hours), "flowbased_missing_mtu_share"].eq(1.0).all()
    assert np.isfinite(
        features.loc[:, list(FLOWBASED_FEATURE_COLUMNS)].to_numpy(dtype=float)
    ).all()


def test_daily_bundle_is_checksum_verified_and_assembled(tmp_path: Path) -> None:
    day = date(2026, 8, 1)
    fetch = _daily_fetch(day, last_modified="2026-07-31T04:00:00Z")
    normalised, audit = normalise_initial_computation(fetch, delivery_day=day)
    features = build_hourly_flowbased_features(normalised, daily_audit=audit)
    metadata = write_daily_flowbased_bundle(
        tmp_path,
        delivery_day=day,
        fetch=fetch,
        normalised=normalised,
        features=features,
        audit=audit,
    )
    assert metadata["pit_eligible"] is True
    store, manifest = assemble_flowbased_feature_store(
        tmp_path, start_day=day, end_day=day
    )
    assert store.is_file()
    assert manifest["physical_hours"] == 24
    assert manifest["all_partitions_research_pit_eligible"] is True
    assert manifest["all_partitions_operational_pit_eligible"] is False
    assert manifest["all_partitions_tls_verified"] is True
    with pytest.raises(JaoFlowBasedError, match="capture operationnelle"):
        assemble_flowbased_feature_store(
            tmp_path,
            start_day=day,
            end_day=day,
            require_operational_pit=True,
        )
    with pytest.raises(JaoFlowBasedError, match="Checksum raw"):
        raw = tmp_path / "raw" / "initialComputation" / "2026-08-01.json.gz"
        raw.write_bytes(raw.read_bytes() + b"corrupt")
        write_daily_flowbased_bundle(
            tmp_path,
            delivery_day=day,
            fetch=fetch,
            normalised=normalised,
            features=features,
            audit=audit,
        )


def test_empty_initial_day_uses_audited_previous_initial_fallback(
    tmp_path: Path,
) -> None:
    previous_day = date(2026, 8, 1)
    target_day = date(2026, 8, 2)
    previous_fetch = _daily_fetch(
        previous_day, last_modified="2026-07-31T04:00:00Z"
    )
    previous_normalised, previous_audit = normalise_initial_computation(
        previous_fetch, delivery_day=previous_day
    )
    previous_features = build_hourly_flowbased_features(
        previous_normalised, daily_audit=previous_audit
    )
    previous_metadata = write_daily_flowbased_bundle(
        tmp_path,
        delivery_day=previous_day,
        fetch=previous_fetch,
        normalised=previous_normalised,
        features=previous_features,
        audit=previous_audit,
    )
    start, end = local_day_utc_bounds(target_day)
    empty_fetch = JaoFetchResult(
        endpoint="initialComputation",
        start_utc=start,
        end_utc=end,
        rows=(),
        total_rows=0,
        last_modified_utc=None,
        retrieved_at_utc=pd.Timestamp("2026-09-05T10:00:00Z"),
        filters={"Presolved": True},
        requests=1,
    )
    empty_normalised, empty_audit = normalise_initial_computation(
        empty_fetch, delivery_day=target_day
    )
    assert empty_normalised.empty
    assert empty_audit["pit_eligible"] is False
    fallback, fallback_audit = build_causal_empty_day_fallback(
        previous_features,
        daily_audit=empty_audit,
        previous_audit=previous_metadata,
        previous_day=previous_day,
    )
    assert len(fallback) == 24
    assert fallback["flowbased_hour_imputed"].eq(1.0).all()
    assert fallback["flowbased_cnec_mtu_availability"].eq(0.0).all()
    assert fallback_audit["pit_eligible"] is True
    assert fallback_audit["fallback_source_day"] == previous_day.isoformat()
    assert np.isfinite(
        fallback.loc[:, list(FLOWBASED_FEATURE_COLUMNS)].to_numpy(dtype=float)
    ).all()
    write_daily_flowbased_bundle(
        tmp_path,
        delivery_day=target_day,
        fetch=empty_fetch,
        normalised=empty_normalised,
        features=fallback,
        audit=fallback_audit,
    )
    store, manifest = assemble_flowbased_feature_store(
        tmp_path, start_day=previous_day, end_day=target_day
    )
    assert manifest["empty_initial_fallback_days"] == 1
    assert manifest["imputed_hours"] == 24
    assert manifest["all_partitions_research_pit_eligible"] is True
    loaded, _ = _load_flow_store(
        store, configured_columns=FLOWBASED_FEATURE_COLUMNS
    )
    assert len(loaded) == 48
    third_day = date(2026, 8, 3)
    third_start, third_end = local_day_utc_bounds(third_day)
    third_fetch = JaoFetchResult(
        endpoint="initialComputation",
        start_utc=third_start,
        end_utc=third_end,
        rows=(),
        total_rows=0,
        last_modified_utc=None,
        retrieved_at_utc=pd.Timestamp("2026-09-05T10:00:00Z"),
        filters={"Presolved": True},
        requests=1,
    )
    _, third_raw_audit = normalise_initial_computation(
        third_fetch, delivery_day=third_day
    )
    consecutive, consecutive_audit = build_causal_empty_day_fallback(
        fallback,
        daily_audit=third_raw_audit,
        previous_audit=fallback_audit,
        previous_day=target_day,
    )
    assert len(consecutive) == 24
    assert consecutive_audit["fallback_source_day"] == previous_day.isoformat()
    assert consecutive_audit["fallback_consecutive_days"] == 2


def test_post_cutoff_initial_day_uses_audited_causal_fallback(
    tmp_path: Path,
) -> None:
    previous_day = date(2026, 8, 1)
    target_day = date(2026, 8, 2)
    previous_fetch = _daily_fetch(
        previous_day,
        last_modified=(
            expected_cutoff_utc(previous_day) - pd.Timedelta(hours=1)
        ).isoformat(),
    )
    previous_normalised, previous_audit = normalise_initial_computation(
        previous_fetch, delivery_day=previous_day
    )
    previous_features = build_hourly_flowbased_features(
        previous_normalised, daily_audit=previous_audit
    )
    previous_metadata = write_daily_flowbased_bundle(
        tmp_path,
        delivery_day=previous_day,
        fetch=previous_fetch,
        normalised=previous_normalised,
        features=previous_features,
        audit=previous_audit,
    )
    raw_last_modified = expected_cutoff_utc(target_day) + pd.Timedelta(hours=1)
    late_fetch = _daily_fetch(
        target_day,
        last_modified=raw_last_modified.isoformat(),
    )
    late_normalised, late_audit = normalise_initial_computation(
        late_fetch, delivery_day=target_day
    )
    assert late_audit["pit_eligible"] is False

    fallback, fallback_audit = build_causal_empty_day_fallback(
        previous_features,
        daily_audit=late_audit,
        previous_audit=previous_metadata,
        previous_day=previous_day,
    )
    assert fallback["flowbased_cnec_mtu_availability"].eq(0.0).all()
    assert fallback_audit["pit_eligible"] is True
    assert fallback_audit["fallback_reason"] == "api_last_modified_after_cutoff"
    assert fallback_audit["raw_api_last_modified_utc"] == raw_last_modified.isoformat()
    assert fallback_audit["raw_api_rows"] == len(late_fetch.rows)
    write_daily_flowbased_bundle(
        tmp_path,
        delivery_day=target_day,
        fetch=late_fetch,
        normalised=late_normalised,
        features=fallback,
        audit=fallback_audit,
    )
    _, manifest = assemble_flowbased_feature_store(
        tmp_path, start_day=previous_day, end_day=target_day
    )
    assert manifest["late_initial_fallback_days"] == 1
    assert manifest["causal_fallback_days"] == 1
    assert manifest["all_partitions_research_pit_eligible"] is True


def test_materializer_repairs_only_an_existing_non_pit_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous_day = date(2026, 8, 1)
    target_day = date(2026, 8, 2)
    previous_fetch = _daily_fetch(
        previous_day,
        last_modified=(
            expected_cutoff_utc(previous_day) - pd.Timedelta(hours=1)
        ).isoformat(),
    )
    previous_normalised, previous_audit = normalise_initial_computation(
        previous_fetch, delivery_day=previous_day
    )
    previous_features = build_hourly_flowbased_features(
        previous_normalised, daily_audit=previous_audit
    )
    write_daily_flowbased_bundle(
        tmp_path,
        delivery_day=previous_day,
        fetch=previous_fetch,
        normalised=previous_normalised,
        features=previous_features,
        audit=previous_audit,
    )
    late_fetch = _daily_fetch(
        target_day,
        last_modified=(
            expected_cutoff_utc(target_day) + pd.Timedelta(hours=1)
        ).isoformat(),
    )
    late_normalised, late_audit = normalise_initial_computation(
        late_fetch, delivery_day=target_day
    )
    late_features = build_hourly_flowbased_features(
        late_normalised, daily_audit=late_audit
    )
    write_daily_flowbased_bundle(
        tmp_path,
        delivery_day=target_day,
        fetch=late_fetch,
        normalised=late_normalised,
        features=late_features,
        audit=late_audit,
    )

    class FakeJaoClient:
        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self) -> "FakeJaoClient":
            return self

        def __exit__(self, *_: object) -> None:
            self.close()

        def close(self) -> None:
            pass

        def fetch_initial_day(self, day: date) -> JaoFetchResult:
            assert day == target_day
            return late_fetch

    monkeypatch.setattr(flowbased_materializer, "JaoCoreClient", FakeJaoClient)
    exit_code = flowbased_materializer.main(
        [
            "--start-day",
            target_day.isoformat(),
            "--end-day",
            target_day.isoformat(),
            "--evaluation-days",
            "1",
            "--training-days",
            "0",
            "--future-days",
            "0",
            "--workers",
            "1",
            "--output-root",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    repaired_audit = json.loads(
        (
            tmp_path
            / "raw"
            / "initialComputation"
            / f"{target_day.isoformat()}.audit.json"
        ).read_text(encoding="utf-8")
    )
    repaired_features = pd.read_parquet(
        tmp_path / "daily_features" / f"{target_day.isoformat()}.parquet"
    )
    assert repaired_audit["pit_eligible"] is True
    assert repaired_audit["fallback_reason"] == "api_last_modified_after_cutoff"
    assert repaired_features["flowbased_cnec_mtu_availability"].eq(0.0).all()


def test_materializer_seeds_and_assembles_an_empty_first_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_day = date(2026, 8, 2)
    insecure_day = target_day - pd.Timedelta(days=1)
    insecure_fetch = _daily_fetch(
        insecure_day,
        last_modified=(
            expected_cutoff_utc(insecure_day) - pd.Timedelta(hours=1)
        ).isoformat(),
    )
    insecure_normalised, insecure_audit = normalise_initial_computation(
        insecure_fetch, delivery_day=insecure_day
    )
    insecure_features = build_hourly_flowbased_features(
        insecure_normalised, daily_audit=insecure_audit
    )
    write_daily_flowbased_bundle(
        tmp_path,
        delivery_day=insecure_day,
        fetch=insecure_fetch,
        normalised=insecure_normalised,
        features=insecure_features,
        audit=insecure_audit,
        tls_verification=False,
    )

    class FakeJaoClient:
        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self) -> "FakeJaoClient":
            return self

        def __exit__(self, *_: object) -> None:
            self.close()

        def close(self) -> None:
            pass

        def fetch_initial_day(self, day: date) -> JaoFetchResult:
            if day != target_day:
                return _daily_fetch(
                    day,
                    last_modified=(
                        expected_cutoff_utc(day) - pd.Timedelta(hours=1)
                    ).isoformat(),
                )
            start, end = local_day_utc_bounds(day)
            return JaoFetchResult(
                endpoint="initialComputation",
                start_utc=start,
                end_utc=end,
                rows=(),
                total_rows=0,
                last_modified_utc=None,
                retrieved_at_utc=pd.Timestamp("2026-09-05T10:00:00Z"),
                filters={"Presolved": True},
                requests=1,
            )

    monkeypatch.setattr(flowbased_materializer, "JaoCoreClient", FakeJaoClient)
    exit_code = flowbased_materializer.main(
        [
            "--start-day",
            target_day.isoformat(),
            "--end-day",
            target_day.isoformat(),
            "--evaluation-days",
            "1",
            "--training-days",
            "0",
            "--future-days",
            "0",
            "--workers",
            "1",
            "--output-root",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    manifest = json.loads(
        (tmp_path / "flowbased_features.audit.json").read_text(encoding="utf-8")
    )
    assert manifest["calendar_days"] == 1
    assert manifest["physical_hours"] == 24
    assert manifest["empty_initial_fallback_days"] == 1
    assert manifest["all_partitions_research_pit_eligible"] is True
    assert manifest["all_partitions_tls_verified"] is True
    target_audit = json.loads(
        (
            tmp_path
            / "raw"
            / "initialComputation"
            / f"{target_day.isoformat()}.audit.json"
        ).read_text(encoding="utf-8")
    )
    assert target_audit["fallback_source_day"] == (
        target_day - pd.Timedelta(days=2)
    ).isoformat()
    assert target_audit["fallback_source_tls_verification"] is True


def test_flowbased_poc_configuration_uses_aliases_without_core_cache_change() -> None:
    root = Path(__file__).resolve().parents[1]
    filter_config, covariates, aliases, raw = _config_objects(
        root / "config" / "kalman_flowbased_experimental.yaml"
    )
    assert filter_config.candidate_kinds == (
        "linear_bias",
        "linear_harmonic",
        "linear_market",
        "linear_fundamental",
        "linear_market_weather",
        "linear_scale",
    )
    assert aliases["linear_fundamental"] == "linear_flowbased"
    assert aliases["linear_market_weather"] == "linear_market_flowbased"
    assert set(raw["flowbased_feature_columns"]) == set(
        covariates.groups["fundamentals"]
    )


def test_poc_output_is_confined_to_experiments() -> None:
    root = Path(__file__).resolve().parents[1]
    protected = root / "runs" / "live" / "source"
    _validate_safe_output(
        root / "runs" / "experiments" / "safe-poc",
        protected_paths=(protected,),
    )
    with pytest.raises(KalmanFlowBasedError, match="runs/experiments"):
        _validate_safe_output(root, protected_paths=(protected,))
    with pytest.raises(KalmanFlowBasedError, match="non disjointe"):
        _validate_safe_output(
            root / "runs" / "experiments" / "nested",
            protected_paths=(root / "runs" / "experiments",),
        )


def test_metrics_pair_storm_on_one_common_hourly_mask() -> None:
    timezone = "Europe/Paris"
    end = pd.Timestamp("2026-09-03", tz=timezone)
    start = end - pd.DateOffset(years=1)
    index = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
    assert len(index) == 8760
    actual = pd.Series(50.0, index=np.arange(len(index)))
    storm = pd.Series(53.0, index=np.arange(len(index)))
    storm.iloc[0] = np.nan
    frame = pd.DataFrame(
        {
            "delivery_start_utc": index,
            "actual": actual,
            "residual_corrected__q50": 52.0,
            f"{MODEL_KEY}__q50": 51.0,
            "storm_dashboard_official__q50": storm,
        }
    )
    result = _metrics(frame, timezone=timezone)
    core = [
        row
        for row in result["models"]
        if row["comparison_scope"] == "FINAL365_exact_8760h"
    ]
    storm_paired = [
        row
        for row in result["models"]
        if row["comparison_scope"] == "paired_with_storm_dashboard_official"
    ]
    assert {row["hours"] for row in core} == {8760}
    assert len(storm_paired) == 3
    assert {row["hours"] for row in storm_paired} == {8759}


def test_unavailable_flowbased_hours_are_forced_to_identity() -> None:
    index = pd.date_range("2026-08-01T00:00:00Z", periods=3, freq="h")
    frame = pd.DataFrame({"delivery_start_utc": index})
    for quantile, baseline, challenger in (
        ("q10", 40.0, 35.0),
        ("q50", 50.0, 45.0),
        ("q90", 60.0, 55.0),
    ):
        frame[f"residual_corrected__{quantile}"] = baseline
        frame[f"{MODEL_KEY}__{quantile}"] = challenger
    flow = pd.DataFrame(
        {"flowbased_cnec_mtu_availability": [1.0, 0.0, 0.5]}, index=index
    )
    output, count = _neutralize_unavailable_flowbased_hours(frame, flow=flow)
    assert count == 1
    for quantile in ("q10", "q50", "q90"):
        assert output.loc[1, f"{MODEL_KEY}__{quantile}"] == output.loc[
            1, f"residual_corrected__{quantile}"
        ]
        assert output.loc[0, f"{MODEL_KEY}__{quantile}"] != output.loc[
            0, f"residual_corrected__{quantile}"
        ]


def test_preflight_requires_exact_730_history_days_plus_future() -> None:
    timezone = "Europe/Paris"
    delivery = date(2026, 9, 3)
    history = pd.date_range(
        pd.Timestamp(date(2024, 9, 3), tz=timezone),
        pd.Timestamp(delivery, tz=timezone),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    future = pd.date_range(
        pd.Timestamp(delivery, tz=timezone),
        pd.Timestamp(date(2026, 9, 4), tz=timezone),
        freq="h",
        inclusive="left",
    ).tz_convert("UTC")
    statistics = pd.DataFrame(
        {
            "delivery_start_utc": history,
            "actual": 50.0,
            "residual_corrected__q10": 40.0,
            "residual_corrected__q50": 50.0,
            "residual_corrected__q90": 60.0,
        }
    )
    source_forecast = pd.DataFrame({"delivery_start_utc": future})
    support = history.append(future)
    covariates = pd.DataFrame({"timestamp": support})
    for column in (
        "fr_residual_load_fcst",
        "de_residual_load_fcst",
        "be_residual_load_fcst",
        "nl_residual_load_fcst",
        "es_residual_load_fcst",
        "flowbased_ram_min_mw",
    ):
        covariates[column] = 1.0
    summary = _validate_preflight_support(
        statistics=statistics,
        source_forecast=source_forecast,
        covariates=covariates,
        configured_columns=("flowbased_ram_min_mw",),
        timezone=timezone,
        delivery_day=delivery,
    )
    assert summary["history_days"] == 730
    assert summary["history_hours"] == 17520
    assert summary["future_hours"] == 24
