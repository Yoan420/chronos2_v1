from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos2_hourly.hourly_contract import local_delivery_day_index
from chronos2_hourly.market_coupling import (
    MARKET_BORDERS,
    MARKET_ZONES,
    MarketCouplingDataError,
    PriceForecastBundle,
    build_frontier_detail_table,
    build_market_coupling_deck,
    build_market_coupling_view,
    discover_latest_common_price_day,
    load_live_price_forecasts,
)


TIMEZONES = {
    "FR": "Europe/Paris",
    "BE": "Europe/Brussels",
    "DE": "Europe/Berlin",
    "NL": "Europe/Amsterdam",
    "ES": "Europe/Madrid",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_dir(root: Path, zone: str, day: date) -> Path:
    token = day.isoformat()
    if zone == "FR":
        return root / "runs" / "live" / f"fr_day_ahead_{token}"
    lower = zone.lower()
    return root / "runs" / "live" / lower / f"{lower}_day_ahead_{token}"


def _publish_synthetic_forecast(
    root: Path,
    *,
    zone: str,
    day: date,
    cutoff: str = "2026-08-14T06:00:00Z",
    storm_used_as_feature: bool = False,
) -> Path:
    run_dir = _run_dir(root, zone, day)
    run_dir.mkdir(parents=True, exist_ok=True)
    index = local_delivery_day_index(day, timezone=TIMEZONES[zone])
    model = "mkonline_blend" if zone == "FR" else "residual_corrected"
    forecast_path = run_dir / f"forecast_hourly_{zone.lower()}.csv"
    pd.DataFrame(
        {
            "delivery_start_utc": index,
            f"{model}__q50": np.arange(len(index), dtype=float) + MARKET_ZONES.index(zone) * 10,
            # A tempting Storm column proves that the loader never selects it.
            "storm__q50": np.full(len(index), -9999.0),
        }
    ).to_csv(forecast_path, index=False)
    manifest = {
        "zone": zone,
        "timezone": TIMEZONES[zone],
        "delivery_day_local": day.isoformat(),
        "forecast_cutoff_utc": cutoff,
        "storm_used_as_feature": storm_used_as_feature,
    }
    manifest["native_model" if zone == "FR" else "candidate_model"] = model
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    checksums = {
        "algorithm": "sha256",
        "artifacts": [
            {
                "path": forecast_path.name,
                "role": "run_artifact",
                "sha256": _sha256(forecast_path),
            }
        ],
    }
    (run_dir / "artifact_checksums.json").write_text(
        json.dumps(checksums), encoding="utf-8"
    )
    return forecast_path


def _bundle(*, periods: int = 3) -> PriceForecastBundle:
    index = pd.date_range("2026-08-14T22:00:00Z", periods=periods, freq="h")
    values = {
        "FR": np.full(periods, 40.0),
        "BE": np.full(periods, 45.0),
        "DE": np.full(periods, 55.0),
        "NL": np.full(periods, 47.0),
        "ES": np.full(periods, 35.0),
    }
    provenance = pd.DataFrame(
        {
            "zone": MARKET_ZONES,
            "status": ["forecast_price"] * len(MARKET_ZONES),
            "model": ["candidate"] * len(MARKET_ZONES),
        }
    )
    return PriceForecastBundle(
        delivery_day=date(2026, 8, 15),
        timeline_utc=index,
        prices=pd.DataFrame(values, index=index),
        provenance=provenance,
        data_as_of_utc=pd.Timestamp("2026-08-14T06:00:00Z"),
    )


def _signal(
    bundle: PriceForecastBundle,
    *,
    zone_from: str = "FR",
    zone_to: str = "DE",
    value: float = 500.0,
    kind: str = "flow",
    timestamp: pd.Timestamp | None = None,
    causal_verified: bool = True,
    series: str = "power.storm.flow.net.fr.de.mw.h.obs.entsoe.scheduled",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_utc": [timestamp or bundle.timeline_utc[0]],
            "zone_from": [zone_from],
            "zone_to": [zone_to],
            "value": [value],
            "unit": ["MW"],
            "kind": [kind],
            "series": [series],
            "as_of_utc": [bundle.data_as_of_utc],
            "causal_verified": [causal_verified],
        }
    )


def test_loader_selects_latest_atomically_complete_common_day_and_never_storm(
    tmp_path: Path,
) -> None:
    complete_day = date(2026, 8, 15)
    for zone in MARKET_ZONES:
        _publish_synthetic_forecast(tmp_path, zone=zone, day=complete_day)

    # A staging directory for one later country is not an atomic publication.
    staging = _run_dir(tmp_path, "FR", date(2026, 8, 16))
    staging.mkdir(parents=True)
    (staging / "run_manifest.json").write_text("{}", encoding="utf-8")

    assert discover_latest_common_price_day(tmp_path) == complete_day
    loaded = load_live_price_forecasts(tmp_path)

    assert loaded.delivery_day == complete_day
    assert tuple(loaded.prices.columns) == MARKET_ZONES
    assert loaded.timeline_utc.equals(
        local_delivery_day_index(complete_day, timezone="Europe/Paris")
    )
    assert loaded.prices["FR"].iloc[0] == pytest.approx(0.0)
    assert not (loaded.prices == -9999.0).any().any()
    assert loaded.provenance["storm_used_as_feature"].eq(False).all()
    assert loaded.provenance["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_loader_fails_closed_on_checksum_or_storm_feature(tmp_path: Path) -> None:
    day = date(2026, 8, 15)
    paths = {
        zone: _publish_synthetic_forecast(tmp_path, zone=zone, day=day)
        for zone in MARKET_ZONES
    }
    paths["DE"].write_text("tampered", encoding="utf-8")
    with pytest.raises(MarketCouplingDataError, match="Checksum invalide"):
        load_live_price_forecasts(tmp_path, delivery_day=day)

    _publish_synthetic_forecast(
        tmp_path, zone="DE", day=day, storm_used_as_feature=True
    )
    with pytest.raises(MarketCouplingDataError, match="Storm est exclu"):
        load_live_price_forecasts(tmp_path, delivery_day=day)


def test_hourly_proxy_is_price_direction_and_explicitly_not_a_flow() -> None:
    bundle = _bundle()
    view = build_market_coupling_view(
        bundle, aggregation="hour", selected_time=bundle.timeline_utc[1]
    )

    fr_de = view.edges.set_index("border").loc["FR-DE"]
    assert (fr_de["source_zone"], fr_de["target_zone"]) == ("FR", "DE")
    assert fr_de["value"] == pytest.approx(15.0)
    assert fr_de["unit"] == "EUR/MWh"
    assert fr_de["status"] == "proxy"
    assert fr_de["signal_kind"] == "forecast_price_spread_proxy"
    assert fr_de["is_reported_flow"] is False or not fr_de["is_reported_flow"]
    assert "aucun flux physique" in fr_de["tooltip"]
    assert set(view.edges["border"]) == {"-".join(border) for border in MARKET_BORDERS}
    assert len(view.edges) == len(MARKET_BORDERS)
    assert view.uses_only_proxies


def test_daily_proxy_uses_signed_mean_for_direction_and_reports_mean_absolute() -> None:
    bundle = _bundle(periods=2)
    prices = bundle.prices.copy()
    prices["FR"] = [0.0, 100.0]
    prices["DE"] = [90.0, 20.0]
    bundle = PriceForecastBundle(
        delivery_day=bundle.delivery_day,
        timeline_utc=bundle.timeline_utc,
        prices=prices,
        provenance=bundle.provenance,
        data_as_of_utc=bundle.data_as_of_utc,
    )

    view = build_market_coupling_view(bundle, aggregation="day")
    fr_de = view.edges.set_index("border").loc["FR-DE"]

    assert view.selected_time_utc is None
    assert (fr_de["source_zone"], fr_de["target_zone"]) == ("FR", "DE")
    assert fr_de["signed_value_canonical"] == pytest.approx(5.0)
    # Daily magnitude/style/ranking use the mean absolute hourly gap; the
    # signed mean remains available without cancelling hourly reversals.
    assert fr_de["value"] == pytest.approx(85.0)
    assert fr_de["mean_abs_value"] == pytest.approx(85.0)
    assert fr_de["signed_value_canonical"] == pytest.approx(5.0)
    assert fr_de["mean_abs_price_spread_eur_mwh"] == pytest.approx(85.0)
    assert not bool(fr_de["direction_is_stable"])
    assert "Direction instable" in fr_de["direction_label"]


def test_verified_negative_flow_reverses_direction_and_beats_capacity() -> None:
    bundle = _bundle()
    flow = _signal(bundle, value=-600.0, kind="flow", series="verified.flow")
    capacity = _signal(bundle, value=1800.0, kind="capacity", series="verified.capacity")

    view = build_market_coupling_view(
        bundle,
        aggregation="hour",
        selected_time=bundle.timeline_utc[0],
        border_signals=pd.concat([capacity, flow], ignore_index=True),
    )
    fr_de = view.edges.set_index("border").loc["FR-DE"]

    assert fr_de["status"] == "flow"
    assert fr_de["status_label"] == "Flux vérifié"
    assert (fr_de["source_zone"], fr_de["target_zone"]) == ("DE", "FR")
    assert fr_de["value"] == pytest.approx(600.0)
    assert fr_de["unit"] == "MW"
    assert fr_de["series"] == "verified.flow"
    assert fr_de["is_reported_flow"] is True or fr_de["is_reported_flow"]


def test_incomplete_daily_verified_series_falls_back_to_proxy_without_invention() -> None:
    bundle = _bundle(periods=3)
    partial = _signal(bundle, kind="capacity", value=1400.0, series="partial.capacity")

    view = build_market_coupling_view(
        bundle, aggregation="day", border_signals=partial
    )
    fr_de = view.edges.set_index("border").loc["FR-DE"]

    assert fr_de["status"] == "proxy"
    assert fr_de["unit"] == "EUR/MWh"
    assert "partial.capacity" not in set(view.provenance.get("series", pd.Series(dtype=str)).dropna())


@pytest.mark.parametrize(
    ("signals", "message"),
    [
        (lambda bundle: _signal(bundle, zone_to="NL"), "Frontière non autorisée"),
        (
            lambda bundle: _signal(bundle, causal_verified=False),
            "causal_verified=True",
        ),
        (
            lambda bundle: _signal(bundle).assign(as_of_utc=pd.Timestamp("2026-08-14T07:00:00Z")),
            "après le cutoff causal",
        ),
        (
            lambda bundle: _signal(bundle, kind="capacity", value=-1.0),
            "capacité directionnelle ne peut pas être négative",
        ),
    ],
)
def test_unverified_or_non_adjacent_signals_are_rejected(
    signals: object, message: str
) -> None:
    bundle = _bundle()
    with pytest.raises(MarketCouplingDataError, match=message):
        build_market_coupling_view(
            bundle,
            aggregation="hour",
            selected_time=bundle.timeline_utc[0],
            border_signals=signals(bundle),  # type: ignore[operator]
        )


def test_pydeck_uses_modest_lines_price_badges_and_no_proxy_arrowheads() -> None:
    view = build_market_coupling_view(_bundle(), aggregation="day")
    deck = build_market_coupling_deck(view)

    assert [layer.type for layer in deck.layers] == [
        "LineLayer",
        "TextLayer",
        "ScatterplotLayer",
        "TextLayer",
    ]
    assert view.edges["width"].between(2.0, 6.0).all()
    assert view.edges["color"].map(lambda color: len(color) == 4).all()
    assert view.edges["arrow_position"].map(lambda point: len(point) == 2).all()
    assert deck.layers[1].data == []
    serialized = deck.to_json()
    assert '"widthUnits": "pixels"' in serialized
    assert "@@=pixels" not in serialized
    assert deck.initial_view_state.pitch == 0
    assert set(view.nodes["zone"]) == set(MARKET_ZONES)
    assert view.nodes["badge_label"].str.contains("EUR/MWh", regex=False).all()


def test_frontier_detail_ranks_price_gaps_and_exposes_both_prices() -> None:
    view = build_market_coupling_view(
        _bundle(), aggregation="hour", selected_time=_bundle().timeline_utc[0]
    )

    details = build_frontier_detail_table(view)

    assert details.iloc[0]["border"] == "FR-DE"
    assert list(details["rank"]) == list(range(1, len(details) + 1))
    assert {
        "source_zone",
        "origin_price_eur_mwh",
        "target_zone",
        "destination_price_eur_mwh",
        "price_spread_eur_mwh",
        "mean_abs_price_spread_eur_mwh",
        "direction_stability",
        "status_label",
    }.issubset(details.columns)
    assert details["mean_abs_price_spread_eur_mwh"].is_monotonic_decreasing
