from __future__ import annotations

from contextlib import nullcontext
from datetime import date
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import chronos2_hourly.market_coupling as market_coupling
from chronos2_hourly.market_coupling import (
    MARKET_BORDERS,
    MARKET_ZONES,
    MarketCouplingDataError,
    PriceForecastBundle,
    build_frontier_detail_table,
    build_market_coupling_deck,
    build_market_coupling_view,
    render_market_coupling_panel,
)


def _bundle(
    prices: dict[str, float | list[float]] | None = None,
    *,
    periods: int = 3,
) -> PriceForecastBundle:
    index = pd.date_range("2026-08-14T22:00:00Z", periods=periods, freq="h")
    defaults: dict[str, float | list[float]] = {
        "FR": 40.0,
        "BE": 45.0,
        "DE": 55.0,
        "NL": 47.0,
        "ES": 35.0,
    }
    defaults.update(prices or {})
    columns = {
        zone: np.full(periods, float(value)) if np.isscalar(value) else np.asarray(value)
        for zone, value in defaults.items()
    }
    return PriceForecastBundle(
        delivery_day=date(2026, 8, 15),
        timeline_utc=index,
        prices=pd.DataFrame(columns, index=index, columns=MARKET_ZONES, dtype=float),
        provenance=pd.DataFrame(
            {
                "zone": MARKET_ZONES,
                "status": ["forecast_price"] * len(MARKET_ZONES),
                "model": ["candidate"] * len(MARKET_ZONES),
            }
        ),
        data_as_of_utc=pd.Timestamp("2026-08-14T06:00:00Z"),
    )


def _edge_for_spread(spread: float) -> pd.Series:
    view = build_market_coupling_view(
        _bundle({"FR": 100.0, "DE": 100.0 + spread}),
        aggregation="hour",
        selected_time="2026-08-14T22:00:00Z",
    )
    return view.edges.set_index("border").loc["FR-DE"]


def test_view_exposes_a_detailed_auditable_frontier_schema() -> None:
    view = build_market_coupling_view(_bundle(), aggregation="day")

    required_edge_fields = {
        "border",
        "source_zone",
        "target_zone",
        "status",
        "status_label",
        "signal_kind",
        "value",
        "signed_value_canonical",
        "mean_abs_value",
        "unit",
        "origin_price_eur_mwh",
        "destination_price_eur_mwh",
        "price_spread_eur_mwh",
        "direction_label",
        "is_price_equilibrium",
        "series",
        "is_reported_flow",
        "coupling_label",
        "color",
        "width",
        "tooltip",
        "source_position",
        "target_position",
        "arrow_position",
        "arrow_angle",
    }
    required_node_fields = {
        "zone",
        "position",
        "price_eur_mwh",
        "label",
        "tooltip",
        "color",
    }

    assert required_edge_fields.issubset(view.edges.columns)
    assert required_node_fields.issubset(view.nodes.columns)
    assert set(view.edges["border"]) == {"-".join(pair) for pair in MARKET_BORDERS}
    assert set(view.nodes["zone"]) == set(MARKET_ZONES)
    assert view.edges["tooltip"].str.contains("aucun flux physique", regex=False).all()
    assert view.edges["status_label"].eq("Proxy de spread prévu").all()
    assert view.edges["unit"].eq("EUR/MWh").all()
    assert "ni un flux physique" in view.caveat
    assert "ni une capacité transfrontalière" in view.caveat


@pytest.mark.parametrize(
    ("spread", "expected_rgb", "expected_label"),
    [
        (0.0, [34, 197, 94], "fort"),
        (1e-9, [34, 197, 94], "fort"),
        (2.0, [34, 197, 94], "fort"),
        (2.0001, [245, 158, 11], "intermédiaire"),
        (10.0, [245, 158, 11], "intermédiaire"),
        (10.0001, [239, 68, 68], "faible"),
    ],
)
def test_spread_color_thresholds_have_stable_coupling_meaning(
    spread: float,
    expected_rgb: list[int],
    expected_label: str,
) -> None:
    edge = _edge_for_spread(spread)

    assert edge["color"][:3] == expected_rgb
    assert edge["coupling_label"] == expected_label
    assert 2.0 <= float(edge["width"]) <= 12.0


def test_line_width_is_finite_bounded_and_monotone_with_spread() -> None:
    spreads = [0.0, 1e-10, 0.5, 2.0, 5.0, 10.0, 100.0, 1_000_000.0]
    widths = [float(_edge_for_spread(spread)["width"]) for spread in spreads]

    assert all(np.isfinite(width) for width in widths)
    assert all(2.0 <= width <= 12.0 for width in widths)
    assert widths == sorted(widths)
    assert widths[0] == pytest.approx(2.0)
    assert widths[-1] > widths[0]


def test_zero_and_near_zero_spreads_remain_explicit_non_flow_proxies() -> None:
    zero = _edge_for_spread(0.0)
    negative_epsilon = _edge_for_spread(-1e-6)

    assert zero["value"] == pytest.approx(0.0)
    assert zero["status"] == "proxy"
    assert not bool(zero["is_reported_flow"])
    assert zero["source_zone"] == "FR"  # deterministic tie-break only
    assert zero["target_zone"] == "DE"
    assert "aucun flux physique" in zero["tooltip"]
    assert bool(zero["is_price_equilibrium"])
    assert "quilibre de prix" in zero["direction_label"]

    assert negative_epsilon["value"] == pytest.approx(1e-6)
    assert negative_epsilon["source_zone"] == "DE"
    assert negative_epsilon["target_zone"] == "FR"
    assert negative_epsilon["coupling_label"] == "fort"
    assert not bool(negative_epsilon["is_price_equilibrium"])
    assert not bool(negative_epsilon["is_reported_flow"])


def test_every_nonzero_proxy_direction_is_lower_to_higher_forecast_price() -> None:
    bundle = _bundle({"FR": 50.0, "BE": 40.0, "DE": 60.0, "NL": 30.0, "ES": 70.0})
    view = build_market_coupling_view(
        bundle,
        aggregation="hour",
        selected_time=bundle.timeline_utc[0],
    )
    price_by_zone = view.nodes.set_index("zone")["price_eur_mwh"]

    for edge in view.edges.itertuples(index=False):
        source_price = float(price_by_zone[edge.source_zone])
        target_price = float(price_by_zone[edge.target_zone])
        assert source_price < target_price
        assert edge.value == pytest.approx(target_price - source_price)


def test_frontier_detail_table_ranks_absolute_spreads_and_shows_both_prices() -> None:
    bundle = _bundle({"FR": 50.0, "BE": 40.0, "DE": 62.0, "NL": 41.0, "ES": 15.0})
    view = build_market_coupling_view(
        bundle,
        aggregation="hour",
        selected_time=bundle.timeline_utc[0],
    )

    detail = build_frontier_detail_table(view)

    assert list(detail.columns) == [
        "rank",
        "border",
        "direction_label",
        "source_zone",
        "origin_price_eur_mwh",
        "target_zone",
        "destination_price_eur_mwh",
        "price_spread_eur_mwh",
        "mean_abs_price_spread_eur_mwh",
        "direction_stability",
        "direction_is_stable",
        "status_label",
        "value",
        "signed_value_canonical",
        "mean_abs_value",
        "unit",
        "series",
    ]
    assert detail["rank"].tolist() == list(range(1, len(MARKET_BORDERS) + 1))
    assert detail["price_spread_eur_mwh"].abs().is_monotonic_decreasing
    assert np.allclose(
        detail["destination_price_eur_mwh"] - detail["origin_price_eur_mwh"],
        detail["price_spread_eur_mwh"],
    )
    assert detail.iloc[0]["border"] == "FR-ES"
    assert detail.iloc[0]["source_zone"] == "ES"
    assert detail.iloc[0]["target_zone"] == "FR"
    assert detail.iloc[0]["origin_price_eur_mwh"] == pytest.approx(15.0)
    assert detail.iloc[0]["destination_price_eur_mwh"] == pytest.approx(50.0)


def test_frontier_detail_table_rejects_a_displayed_spread_inconsistent_with_prices() -> None:
    view = build_market_coupling_view(_bundle(), aggregation="day")
    view.edges.loc[0, "price_spread_eur_mwh"] += 1.0

    with pytest.raises(MarketCouplingDataError, match="spread affich"):
        build_frontier_detail_table(view)


def test_proxy_map_uses_plain_links_without_directional_arrowheads() -> None:
    """Price direction belongs in the table; a proxy must not look physical."""

    view = build_market_coupling_view(_bundle(), aggregation="day")
    deck = build_market_coupling_deck(view)

    layer_types = [layer.type for layer in deck.layers]
    assert layer_types[0] == "LineLayer"
    assert "ArcLayer" not in layer_types
    # The optional physical-signal arrow layer must be empty for an all-proxy
    # view; the non-empty TextLayer only labels country price badges.
    text_layers = [layer for layer in deck.layers if layer.type == "TextLayer"]
    assert sorted(len(layer.data) for layer in text_layers) == [0, len(MARKET_ZONES)]

    serialized = json.loads(deck.to_json())
    line_layer = serialized["layers"][0]
    assert line_layer["@@type"] == "LineLayer"
    assert line_layer["widthUnits"] == "pixels"
    assert line_layer["widthUnits"] != "@@=pixels"
    assert 1 <= line_layer["widthMinPixels"] <= line_layer["widthMaxPixels"] <= 6
    assert serialized["initialViewState"]["pitch"] == 0


def test_directional_glyphs_are_restricted_to_verified_border_signals() -> None:
    bundle = _bundle()
    view = build_market_coupling_view(
        bundle,
        aggregation="hour",
        selected_time=bundle.timeline_utc[0],
        border_signals=_capacity_signal(bundle),
    )
    deck = build_market_coupling_deck(view)

    edge_text_layers = [
        layer
        for layer in deck.layers
        if layer.type == "TextLayer" and len(layer.data) != len(MARKET_ZONES)
    ]
    assert len(edge_text_layers) == 1
    arrow_rows = edge_text_layers[0].data
    assert len(arrow_rows) == 1
    assert arrow_rows[0]["border"] == "BE-NL"
    assert arrow_rows[0]["status"] == "capacity"
    assert arrow_rows[0]["arrow_glyph"]
    assert all(row["status"] != "proxy" for row in arrow_rows)


def test_daily_crossing_prices_use_mean_absolute_gap_without_fake_direction() -> None:
    """Hourly inversions must not cancel the daily convergence severity."""

    bundle = _bundle(
        {
            "FR": [100.0, 100.0],
            # Canonical DE-FR gaps are +20 then -18: signed mean 1, |gap| mean 19.
            "DE": [120.0, 82.0],
        },
        periods=2,
    )
    view = build_market_coupling_view(bundle, aggregation="day")
    edge = view.edges.set_index("border").loc["FR-DE"]

    assert edge["signed_value_canonical"] == pytest.approx(1.0)
    assert edge["mean_abs_value"] == pytest.approx(19.0)
    assert edge["color"][:3] == [239, 68, 68]
    assert edge["coupling_label"] == "faible"
    assert "→" not in edge["direction_label"]
    assert "prix bas vers prix haut" not in edge["tooltip"]

    detail = build_frontier_detail_table(view)
    assert "mean_abs_value" in detail.columns
    displayed = detail.set_index("border").loc["FR-DE"]
    assert displayed["mean_abs_value"] == pytest.approx(19.0)


def _capacity_signal(bundle: PriceForecastBundle) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_utc": [bundle.timeline_utc[0]],
            "zone_from": ["BE"],
            "zone_to": ["NL"],
            "value": [1400.0],
            "unit": ["MW"],
            "kind": ["capacity"],
            "series": ["verified.capacity.be.nl"],
            "as_of_utc": [bundle.data_as_of_utc],
            "causal_verified": [True],
        }
    )


def test_verified_capacity_is_visually_and_textually_distinct_from_a_flow() -> None:
    bundle = _bundle()
    view = build_market_coupling_view(
        bundle,
        aggregation="hour",
        selected_time=bundle.timeline_utc[0],
        border_signals=_capacity_signal(bundle),
    )
    edge = view.edges.set_index("border").loc["BE-NL"]

    assert edge["status"] == "capacity"
    assert edge["status_label"] == "Capacité vérifiée"
    assert edge["unit"] == "MW"
    assert edge["series"] == "verified.capacity.be.nl"
    assert edge["color"][:3] == [124, 58, 237]
    assert not bool(edge["is_reported_flow"])
    assert "Capacité vérifiée" in edge["tooltip"]
    assert "Cutoff causal vérifié" in edge["tooltip"]


class _StreamlitRecorder:
    def __init__(self) -> None:
        self.chart_widths: list[str] = []
        self.warnings: list[str] = []
        self.successes: list[str] = []
        self.tables: list[pd.DataFrame] = []
        self.captions: list[str] = []
        self.markdowns: list[str] = []
        self.metrics: list[tuple[str, str]] = []
        self.column_config = self

    def columns(self, specification: Any, **_kwargs: Any) -> list[Any]:
        count = specification if isinstance(specification, int) else len(specification)
        return [nullcontext() for _ in range(count)]

    def container(self, **_kwargs: Any) -> Any:
        return nullcontext()

    def markdown(self, message: str) -> None:
        self.markdowns.append(message)

    def metric(self, label: str, value: str, **_kwargs: Any) -> None:
        self.metrics.append((label, value))

    def NumberColumn(self, label: str, **kwargs: Any) -> dict[str, Any]:
        return {"type": "number", "label": label, **kwargs}

    def TextColumn(self, label: str, **kwargs: Any) -> dict[str, Any]:
        return {"type": "text", "label": label, **kwargs}

    def segmented_control(self, _label: str, **_kwargs: Any) -> str:
        return "Heure"

    def selectbox(self, _label: str, *, options: list[pd.Timestamp], **_kwargs: Any) -> pd.Timestamp:
        return options[0]

    def pydeck_chart(self, _deck: Any, *, width: str) -> None:
        self.chart_widths.append(width)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def success(self, message: str) -> None:
        self.successes.append(message)

    def dataframe(self, frame: pd.DataFrame, **_kwargs: Any) -> None:
        self.tables.append(frame.copy())

    def expander(self, _label: str) -> Any:
        return nullcontext()

    def caption(self, message: str) -> None:
        self.captions.append(message)


def test_streamlit_panel_exposes_proxy_warning_frontier_table_and_source_priority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    recorder = _StreamlitRecorder()
    sentinel_deck = object()
    monkeypatch.setattr(market_coupling, "load_live_price_forecasts", lambda *_a, **_k: bundle)
    monkeypatch.setattr(market_coupling, "build_market_coupling_deck", lambda _view: sentinel_deck)

    view = render_market_coupling_panel(recorder, tmp_path)

    assert view.uses_only_proxies
    assert recorder.chart_widths == ["stretch"]
    assert len(recorder.metrics) == 3
    assert not recorder.successes
    assert len(recorder.warnings) == 1
    assert "6/6 frontières" in recorder.warnings[0]
    assert "proxy" in recorder.warnings[0].lower()
    assert "ni un flux physique" in recorder.warnings[0]
    assert len(recorder.tables) == 2
    required_table_columns = {
        "border",
        "source_zone",
        "target_zone",
        "status_label",
        "value",
        "unit",
        "series",
    }
    assert required_table_columns.issubset(recorder.tables[0].columns)
    assert recorder.tables[0]["status_label"].eq("Proxy de spread prévu").all()
    assert recorder.tables[0]["unit"].eq("EUR/MWh").all()
    rendered_help = " ".join(recorder.markdowns + recorder.captions)
    assert "≤ 2 EUR/MWh" in rendered_help
    assert "> 2 à 10 EUR/MWh" in rendered_help
    assert "> 10 EUR/MWh" in rendered_help
    assert "sans flèche" in rendered_help
    assert "jamais à un proxy" in rendered_help
    assert any("flux causal vérifié" in caption for caption in recorder.captions)
    assert any("capacité causale vérifiée" in caption for caption in recorder.captions)
    assert any("proxy de spread p50" in caption.lower() for caption in recorder.captions)
