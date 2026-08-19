from __future__ import annotations

from copy import deepcopy
from datetime import date
import json
from pathlib import Path

import pandas as pd
import pytest

from chronos2_hourly.event_features import (
    EventFeatureContractError,
    day_ahead_cutoff_utc,
    load_meteofrance_vigilance_snapshot,
    materialize_meteofrance_vigilance_features,
    parse_meteofrance_vigilance_snapshot,
    select_latest_causal_snapshot,
)


PERIOD_START = pd.Timestamp("2026-01-15T23:00:00Z")
PERIOD_END = pd.Timestamp("2026-01-16T23:00:00Z")
DELIVERY_INDEX = pd.date_range(PERIOD_START, periods=4, freq="h")


def _source_uri(day: str = "2026/01/15", hms: str = "050030") -> str:
    return (
        "https://files.data.gouv.fr/meteofrance/data/vigilance/metropole/"
        f"{day}/{hms}/CDP_CARTE_EXTERNE.json"
    )


def _domain(
    domain_id: str,
    *,
    wind_max_color: int = 1,
    wind_intervals: list[tuple[str, str, int]] | None = None,
    flood_max_color: int | None = None,
) -> dict[str, object]:
    phenomena: list[dict[str, object]] = [
        {
            "phenomenon_id": "1",
            "phenomenon_max_color_id": wind_max_color,
            "timelaps_items": [
                {"begin_time": start, "end_time": end, "color_id": color}
                for start, end, color in (wind_intervals or [])
            ],
        }
    ]
    if flood_max_color is not None:
        # Météo-France documents that flood chronology can be empty.  The
        # period maximum must remain distinct from hourly activity.
        phenomena.append(
            {
                "phenomenon_id": "4",
                "phenomenon_max_color_id": flood_max_color,
                "timelaps_items": [],
            }
        )
    return {
        "domain_id": domain_id,
        "max_color_id": max(wind_max_color, flood_max_color or 1),
        "phenomenon_items": phenomena,
    }


def _payload(
    *,
    snapshot_id: str = "snapshot-old",
    update_time: str = "2026-01-15T05:00:20Z",
    generation_time: str = "2026-01-15T05:00:25Z",
    wind_max_color: int = 3,
    wind_intervals: list[tuple[str, str, int]] | None = None,
    period_start: str = PERIOD_START.isoformat(),
    period_end: str = PERIOD_END.isoformat(),
) -> dict[str, object]:
    intervals = wind_intervals or [
        ("2026-01-15T23:00:00Z", "2026-01-16T01:00:00Z", 3),
        ("2026-01-16T01:00:00Z", "2026-01-16T02:00:00Z", 2),
    ]
    return {
        "product": {
            "warning_type": "vigilance",
            "type_cdp": "cdp_carte_externe",
            "version_vigilance": "V6",
            "version_cdp": "1.0.0",
            "update_time": update_time,
            "domain_id": "FRA",
            "global_max_color_id": str(max(wind_max_color, 4)),
            "periods": [
                {
                    "echeance": "J1",
                    "begin_validity_time": period_start,
                    "end_validity_time": period_end,
                    "timelaps": {
                        "domain_ids": [
                            _domain(
                                "01",
                                wind_max_color=wind_max_color,
                                wind_intervals=intervals,
                            ),
                            _domain(
                                "02",
                                wind_max_color=1,
                                wind_intervals=[(period_start, period_end, 1)],
                                flood_max_color=4,
                            ),
                            # National/coastal rows must not inflate department counts.
                            {
                                "domain_id": "FRA",
                                "max_color_id": 4,
                                "phenomenon_items": [],
                            },
                            {
                                "domain_id": "0210",
                                "max_color_id": 4,
                                "phenomenon_items": [],
                            },
                        ]
                    },
                }
            ],
        },
        "meta": {
            "snapshot_id": snapshot_id,
            "product_datetime": period_start,
            "generation_timestamp": generation_time,
        },
    }


def _snapshot(**kwargs: object):
    payload = _payload(**kwargs)
    update = pd.Timestamp(payload["product"]["update_time"])  # type: ignore[index]
    return parse_meteofrance_vigilance_snapshot(
        payload,
        source_uri=_source_uri(
            day=update.strftime("%Y/%m/%d"),
            hms=update.strftime("%H%M") + "30",
        ),
    )


def test_day_ahead_cutoff_is_dst_aware() -> None:
    assert day_ahead_cutoff_utc(date(2026, 3, 29)) == pd.Timestamp(
        "2026-03-28T07:00:00Z"
    )
    assert day_ahead_cutoff_utc(date(2026, 3, 30)) == pd.Timestamp(
        "2026-03-29T06:00:00Z"
    )
    assert day_ahead_cutoff_utc(date(2026, 10, 25)) == pd.Timestamp(
        "2026-10-24T06:00:00Z"
    )


def test_parser_builds_revision_aware_records_and_excludes_non_departments() -> None:
    snapshot = _snapshot()

    assert snapshot.snapshot_id == "snapshot-old"
    assert snapshot.published_at_utc == pd.Timestamp("2026-01-15T05:00:20Z")
    assert snapshot.revision_at_utc == pd.Timestamp("2026-01-15T05:00:25Z")
    assert snapshot.transmission_at_utc == pd.Timestamp("2026-01-15T05:00:30Z")
    assert snapshot.available_at_utc == snapshot.transmission_at_utc
    assert snapshot.department_ids == ("01", "02")
    assert len(snapshot.content_sha256) == 64
    assert {event.spatial_id for event in snapshot.events} == {"01", "02"}
    assert {event.temporal_precision for event in snapshot.events} == {
        "interval",
        "period_max",
    }
    assert all(event.available_at_utc == snapshot.available_at_utc for event in snapshot.events)


def test_materializer_uses_last_snapshot_available_by_cutoff_not_latest_revision() -> None:
    old = _snapshot()
    future = _snapshot(
        snapshot_id="snapshot-after-cutoff",
        update_time="2026-01-15T08:00:20Z",
        generation_time="2026-01-15T08:00:25Z",
        wind_max_color=4,
        wind_intervals=[
            ("2026-01-15T23:00:00Z", "2026-01-16T03:00:00Z", 4)
        ],
    )

    result = materialize_meteofrance_vigilance_features(
        [future, old],
        delivery_index_utc=DELIVERY_INDEX,
        cutoff_utc="2026-01-15T07:00:00Z",
        minimum_departments=2,
    )

    assert result.provenance["snapshot_id"] == "snapshot-old"
    assert result.provenance["available_at_utc"] == "2026-01-15T05:00:30+00:00"
    assert result.provenance["causal_rule"].endswith("<= cutoff")
    assert result.features.index.equals(DELIVERY_INDEX.rename("delivery_start_utc"))
    assert result.features["mf_vigilance_wind_max_level"].tolist() == [2.0, 2.0, 1.0, 0.0]
    assert result.features["mf_vigilance_wind_departments_orange_plus"].tolist() == [
        1.0,
        1.0,
        0.0,
        0.0,
    ]
    assert result.features["mf_vigilance_max_level"].max() == 2.0
    assert result.features["mf_vigilance_period_max_level"].eq(3.0).all()
    assert result.features["mf_vigilance_source_available"].eq(1.0).all()
    assert result.provenance["free_text_used"] is False
    assert result.provenance["image_used"] is False


def test_period_max_without_chronology_is_not_invented_as_hourly_activity() -> None:
    result = materialize_meteofrance_vigilance_features(
        [_snapshot()],
        delivery_index_utc=DELIVERY_INDEX,
        cutoff_utc="2026-01-15T07:00:00Z",
        minimum_departments=2,
    )

    assert result.features["mf_vigilance_flood_max_level"].eq(0.0).all()
    assert result.features["mf_vigilance_flood_departments_yellow_plus"].eq(0.0).all()
    assert result.features["mf_vigilance_flood_period_max_level"].eq(3.0).all()
    assert result.provenance["period_max_is_not_hourly_activity"] is True
    flood = result.events.loc[result.events["category"] == "flood"]
    assert set(flood["temporal_precision"]) == {"period_max"}


def test_archive_transmission_time_is_a_causal_guard_even_if_payload_claims_earlier() -> None:
    payload = _payload(
        snapshot_id="late-transmission",
        update_time="2026-01-15T06:59:40Z",
        generation_time="2026-01-15T06:59:50Z",
    )
    late = parse_meteofrance_vigilance_snapshot(
        payload, source_uri=_source_uri(hms="070010")
    )
    old = _snapshot()

    assert late.available_at_utc == pd.Timestamp("2026-01-15T07:00:10Z")
    selected = select_latest_causal_snapshot(
        [old, late], cutoff_utc="2026-01-15T07:00:00Z"
    )
    assert selected.snapshot_id == "snapshot-old"


def test_latest_only_or_mismatched_sources_are_rejected() -> None:
    payload = _payload()
    with pytest.raises(EventFeatureContractError, match="official files.data.gouv.fr"):
        parse_meteofrance_vigilance_snapshot(
            payload,
            source_uri="https://public-api.meteofrance.fr/public/DPVigilance/v1/cartevigilance/encours",
        )
    with pytest.raises(EventFeatureContractError, match="differ by more than 30 minutes"):
        parse_meteofrance_vigilance_snapshot(
            payload,
            source_uri=_source_uri(day="2026/01/16", hms="050030"),
        )


def test_no_snapshot_available_at_cutoff_has_an_actionable_error() -> None:
    with pytest.raises(EventFeatureContractError, match="No Météo-France Vigilance snapshot"):
        select_latest_causal_snapshot(
            [_snapshot()], cutoff_utc="2026-01-15T04:59:59Z"
        )


def test_overlapping_timelaps_are_rejected_instead_of_aggregated() -> None:
    payload = _payload(
        wind_intervals=[
            ("2026-01-15T23:00:00Z", "2026-01-16T02:00:00Z", 3),
            ("2026-01-16T01:00:00Z", "2026-01-16T03:00:00Z", 2),
        ]
    )
    with pytest.raises(EventFeatureContractError, match="Overlapping Vigilance intervals"):
        parse_meteofrance_vigilance_snapshot(
            payload, source_uri=_source_uri()
        )


def test_delivery_coverage_and_department_coverage_fail_closed() -> None:
    snapshot = _snapshot()
    with pytest.raises(EventFeatureContractError, match="department coverage is too low"):
        materialize_meteofrance_vigilance_features(
            [snapshot],
            delivery_index_utc=DELIVERY_INDEX,
            cutoff_utc="2026-01-15T07:00:00Z",
            minimum_departments=3,
        )

    outside = pd.date_range("2026-01-16T22:00:00Z", periods=3, freq="h")
    with pytest.raises(EventFeatureContractError, match="does not cover every delivery hour"):
        materialize_meteofrance_vigilance_features(
            [snapshot],
            delivery_index_utc=outside,
            cutoff_utc="2026-01-15T07:00:00Z",
            minimum_departments=2,
        )


def test_local_loader_hashes_exact_archived_bytes(tmp_path: Path) -> None:
    payload = _payload()
    raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    path = tmp_path / "CDP_CARTE_EXTERNE.json"
    path.write_bytes(raw)

    snapshot = load_meteofrance_vigilance_snapshot(path, source_uri=_source_uri())

    import hashlib

    assert snapshot.content_sha256 == hashlib.sha256(raw).hexdigest()


def test_unknown_hazard_and_non_utc_delivery_contract_are_rejected() -> None:
    payload = deepcopy(_payload())
    phenomenon = payload["product"]["periods"][0]["timelaps"]["domain_ids"][0][  # type: ignore[index]
        "phenomenon_items"
    ][0]
    phenomenon["phenomenon_id"] = "999"
    with pytest.raises(EventFeatureContractError, match="Unknown Vigilance phenomenon_id"):
        parse_meteofrance_vigilance_snapshot(payload, source_uri=_source_uri())

    naive = pd.date_range("2026-01-16", periods=4, freq="h")
    with pytest.raises(EventFeatureContractError, match="timezone-aware UTC"):
        materialize_meteofrance_vigilance_features(
            [_snapshot()],
            delivery_index_utc=naive,
            cutoff_utc="2026-01-15T07:00:00Z",
            minimum_departments=2,
        )
