from __future__ import annotations

from copy import deepcopy
import json
import re

import pytest

from chronos2_hourly.model_storm_rolling_view import _best, _format, render_rolling_section


def tables():
    rows = [
        dict(key="storm", label="Storm", mae=13., bias=-2., rmse=20., hit_rate=.35, r2=.80, daily_pnl=40., pnl_comparison_eligible=True),
        dict(key="model", label="Model", mae=11., bias=1., rmse=25., hit_rate=.45, r2=.70, daily_pnl=45., pnl_comparison_eligible=True),
    ]
    zones = []
    for zone, name in (("BE", "Belgium"), ("DE", "Germany"), ("FR", "France"), ("NL", "The Netherlands")):
        windows = {}
        for days in (7, 30, 60, 90, 365):
            windows[str(days)] = dict(start_day="2026-06-22", end_day="2026-09-19",
                paired_hours=days * 24, expected_hours=days * 24, complete_paired_days=days, pnl_days=days,
                pnl_provider_days={"storm": days, "model": days},
                frequencies={"60min": {"samples": days * 24, "providers": deepcopy(rows)},
                             "day": {"samples": days, "providers": deepcopy(rows)}})
        zones.append(dict(zone=zone, name=name, windows=windows))
    quantile_zones = deepcopy(zones)
    for zone in quantile_zones:
        for window in zone["windows"].values():
            for view in window["frequencies"].values():
                view["providers"][0].update(label="Storm calibré", daily_pnl=35.)
                view["providers"][1]["daily_pnl"] = 50.
    return dict(schema_version=3, windows=[7, 30, 60, 90, 365], frequencies=["60min", "day"],
                default_window=90, default_frequency="60min", zones=zones,
                strategies=["quantile_based", "unlimited_bid"], default_strategy="unlimited_bid", quantile_alpha=.8,
                strategy_zones={"quantile_based": quantile_zones, "unlimited_bid": zones})


def test_default_summary_is_server_rendered_and_exactly_two_supported_frequencies():
    source = render_rolling_section(tables())
    assert len(re.findall(r'<article class="rolling-zone"', source)) == 4
    assert re.findall(r'data-rolling-frequency="([^"]+)"', source) == ["60min", "day"]
    assert re.findall(r'data-rolling-period="([^"]+)"', source) == ["7", "30", "60", "90", "365"]
    assert 'aria-pressed="false">LAST 365 DAYS</button>' in source
    assert 'data-rolling-period="90" aria-pressed="true"' in source
    assert 'data-rolling-frequency="60min" aria-pressed="true"' in source
    assert '<th scope="row">Storm</th>' in source
    assert '<th scope="row">Model</th>' in source
    assert "2160 hourly pairs" in source and "P&amp;L: 90 shared eligible days" in source
    for name in ("MAE", "BIAS", "RMSE", "Hit Rate (±€5)", "R²", "Daily P&amp;L (sim.)"):
        assert name in source
    assert "1 MWh battery" in source and "€25 per executed cycle" in source
    assert "DAY retains the same hourly strategy decisions and P&amp;L" in source
    assert "En cours de construction" in source
    assert "Model = NYX nuclear Kalman." not in source
    assert "60-MIN: errors on matched physical hourly prices." not in source


def test_quality_highlight_bias_is_nearer_zero_not_most_negative():
    rows = tables()["zones"][0]["windows"]["90"]["frequencies"]["60min"]["providers"]
    assert _best(rows, "bias", 1.) and not _best(rows, "bias", -2.)
    assert _best(rows, "mae", 11.) and _best(rows, "rmse", 20.)
    assert _best(rows, "hit_rate", .45) and _best(rows, "r2", .80)
    assert _best(rows, "daily_pnl", 45.)
    assert not _best([dict(mae=1.)], "mae", 1.)
    assert _best([dict(bias=-2.), dict(bias=2.)], "bias", -2.)
    assert _best([dict(bias=-2.), dict(bias=2.)], "bias", 2.)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True])
def test_missing_or_invalid_metrics_do_not_look_like_zeros(value):
    assert _format("mae", value) == "—"
    assert not _best([dict(mae=1.), dict(mae=value)], "mae", value)


def test_numeric_display_retains_sign_and_does_not_hide_negative_pnl_or_r2():
    assert _format("bias", -2.) == "-2.00"
    assert _format("r2", -4.) == "-4.00"
    assert _format("daily_pnl", -20.) == "-20.00 €"
    assert _format("mae", 0.) == "0.00"
    assert _format("hit_rate", 1.) == "100.0 %"


def test_empty_history_has_no_invented_scores_but_keeps_country_and_provider_rows():
    data = tables()
    data["zones"][0]["windows"] = {}
    source = render_rolling_section(data)
    card = re.search(r'<article class="rolling-zone".*?</article>', source, re.S).group()
    assert "No complete observed day" in card
    assert "Verified rolling history unavailable" in card
    assert "Storm" in card and "Model" in card
    assert len(re.findall(r'<td data-column="[^"]+">—</td>', card)) == 12
    assert "rolling-best" not in card


def test_no_complete_observation_never_looks_like_a_valid_anchored_window():
    data = tables()
    data["zones"][0]["anchor_day"] = None
    source = render_rolling_section(data)
    card = re.search(r'<article class="rolling-zone".*?</article>', source, re.S).group()
    assert "No complete observed day" in card
    assert "2026-06-22" not in card
    assert len(re.findall(r'<td data-column="[^"]+">—</td>', card)) == 12


def test_source_text_is_escaped_in_markup_and_cannot_close_json_script():
    data = tables()
    data["zones"][0]["name"] = 'Belgium </script><script>alert("x")</script>'
    source = render_rolling_section(data)
    assert '</script><script>alert("x")' not in source
    encoded = re.search(r'<script id="rolling-performance-data" type="application/json">(.*?)</script>', source, re.S).group(1)
    assert json.loads(encoded)["strategy_zones"]["unlimited_bid"][0]["name"] == data["zones"][0]["name"]
    assert "innerHTML" not in source


def test_new_section_is_always_integrated_and_hourly_filter_does_not_hide_summary_rows():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "chronos2_hourly/model_storm_report.py").read_text(encoding="utf-8")
    assert "render_rolling_section(build_rolling_performance(payload))" in source
    assert "querySelectorAll('#view-table tbody tr')" in source
    assert "querySelectorAll('tbody tr')" not in source


def test_two_strategy_tabs_replace_scopes_and_expose_completed_history_only():
    data = tables()
    source = render_rolling_section(data)
    assert 'data-active-strategy="unlimited_bid"' in source
    assert re.findall(r'data-rolling-strategy="([^"]+)"', source) == ["quantile_based", "unlimited_bid"]
    assert 'data-rolling-strategy="quantile_based" aria-pressed="false">Quantile Based' in source
    assert 'data-rolling-strategy="unlimited_bid" aria-pressed="true">Unlimited Bid' in source
    for obsolete in ("data-rolling-scope", "CACHE ONLY", "INTERNAL · COMPLETED HISTORY", "VPS", "4 MWh", "85%"):
        assert obsolete not in source
    assert "Completed history:" in source
    assert "both strategies and providers" in source
    assert 'td.dataset.column=key' in source


def test_different_provider_coverage_disables_green_rankings():
    rows = [dict(mae=1., comparison_eligible=False), dict(mae=2., comparison_eligible=False)]
    assert not _best(rows, "mae", 1.)


def test_hit_percent_midpoint_matches_dashboard_display():
    assert _format("hit_rate", 783 / 2160) == "36.3 %"


def test_strategy_tables_share_server_display_strings_without_mutating_metrics():
    data = tables()
    for zones in data["strategy_zones"].values():
        row = zones[0]["windows"]["90"]["frequencies"]["60min"]["providers"][0]
        row.update(rmse=2.625, hit_rate=.1235)
    before = deepcopy(data)
    source = render_rolling_section(data)
    encoded = re.search(r'<script id="rolling-performance-data" type="application/json">(.*?)</script>', source, re.S).group(1)
    result = json.loads(encoded)
    for zones in result["strategy_zones"].values():
        row = zones[0]["windows"]["90"]["frequencies"]["60min"]["providers"][0]
        assert row["display_values"]["rmse"] == "2.62"
        assert row["display_values"]["hit_rate"] == "12.4 %"
        assert row["rmse"] == 2.625
    assert "row.display_values?.[key]??format(key,row[key])" in source
    assert data == before


def test_unavailable_calibration_is_disclosed_and_has_no_pnl_winner():
    data = tables()
    data["default_strategy"] = "quantile_based"
    for view in data["strategy_zones"]["quantile_based"][0]["windows"]["90"]["frequencies"].values():
        view["providers"][0].update(daily_pnl=None, pnl_comparison_eligible=False,
                                    pnl_unavailable_reason='Insufficient calibration history <60 days>')
    source = render_rolling_section(data)
    card = re.search(r'<article class="rolling-zone".*?</article>', source, re.S).group()
    assert 'Storm calibré' in card
    assert 'Insufficient calibration history &lt;60 days&gt;' in card
    assert 'P&amp;L ranking disabled' in card
    assert not re.search(r'data-column="daily_pnl"[^>]*class="rolling-best"', card)
    assert re.search(r'data-column="mae"[^>]*class="rolling-best"', card)
    assert "past errors" in source and "at least 60 prior complete days" in source


def test_legacy_profit_cannot_silently_be_relabelled_as_paper_strategy():
    data = tables()
    del data["strategy_zones"]
    with pytest.raises(ValueError, match="legacy P&L cannot be relabelled"):
        render_rolling_section(data)


def test_total_profit_and_executed_cycles_are_exposed_without_changing_daily_mean():
    data = tables()
    row = data["strategy_zones"]["unlimited_bid"][0]["windows"]["90"]["frequencies"]["60min"]["providers"][0]
    row.update(total_pnl=3600., trade_days=45, pnl_days=90)
    source = render_rolling_section(data)
    card = re.search(r'<article class="rolling-zone".*?</article>', source, re.S).group()
    assert "3600.00 € total" in card and "45 executed cycles" in card
    assert "Total simulated profit: 3600.00 €" in card
    assert '>40.00 €</td>' in card


def test_offline_controller_switches_strategy_without_changing_scores_or_support(tmp_path):
    """Exercise the real JS with a small in-memory DOM; no browser or URL access."""
    import shutil
    import subprocess
    from chronos2_hourly.model_storm_rolling_view import JS
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the offline controller test")
    data = tables()
    row = data["strategy_zones"]["quantile_based"][0]["windows"]["90"]["frequencies"]["60min"]["providers"][0]
    row.update(daily_pnl=None, pnl_comparison_eligible=False, pnl_unavailable_reason="Calibration incomplete")
    html = render_rolling_section(data)
    encoded = re.search(r'<script id="rolling-performance-data" type="application/json">(.*?)</script>', html, re.S).group(1)
    fixture = tmp_path / "controller.json"
    fixture.write_text(json.dumps({"data": json.loads(encoded), "js": JS}), encoding="utf-8")
    harness = r'''
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const fixture=JSON.parse(fs.readFileSync(process.argv[1],'utf8')),data=fixture.data;
class Element {
  constructor(dataset={}){this.dataset=dataset;this.children=[];this.attrs={};this.handlers={};this.lookup={};this.textContent='';this.title='';}
  setAttribute(k,v){this.attrs[k]=v;}
  addEventListener(k,f){this.handlers[k]=f;}
  appendChild(el){this.children.push(el);}
  replaceChildren(){this.children=[];}
  querySelector(k){return this.lookup[k];}
  querySelectorAll(k){return this.lookup[k]||[];}
  closest(){return this.card;}
  click(){this.handlers.click();}
}
const root=new Element(),cards=new Map(),keys=['provider','mae','bias','rmse','hit_rate','r2','daily_pnl'];
const strategies=['quantile_based','unlimited_bid'].map(k=>new Element({rollingStrategy:k}));
const periods=data.windows.map(k=>new Element({rollingPeriod:String(k)}));
const frequencies=['60min','day'].map(k=>new Element({rollingFrequency:k}));
root.lookup['[data-rolling-strategy]']=strategies;root.lookup['[data-rolling-period]']=periods;
root.lookup['[data-rolling-frequency]']=frequencies;root.lookup['.rolling-strategy-note']=new Element();
root.lookup['[data-rolling-sort]']=[];
for(const zone of data.zones){
  const card=new Element({rollingZone:zone.zone});cards.set(zone.zone,card);
  for(const k of ['.rolling-dates','.rolling-coverage','.rolling-pnl-note','tbody'])card.lookup[k]=new Element();
  card.lookup['thead th']=keys.map(key=>{
    const th=new Element(),button=new Element({rollingSort:key});button.card=card;
    th.lookup.button=button;th.lookup['.rolling-arrow']=new Element();root.lookup['[data-rolling-sort]'].push(button);return th;
  });
  root.lookup['[data-rolling-zone="'+zone.zone+'"]']=card;
}
const document={getElementById:id=>id==='rolling-performance'?root:{textContent:JSON.stringify(data)},createElement:()=>new Element()};
vm.runInNewContext(fixture.js,{document,Map,Number,Math,String});
const table=zone=>cards.get(zone).lookup.tbody.children;
const scores=()=>[...cards.keys()].map(zone=>table(zone).map(row=>row.children.slice(1,6).map(cell=>cell.textContent)));
const coverage=()=>[...cards.values()].map(card=>card.lookup['.rolling-coverage'].textContent);
strategies[1].click();
const originalScores=scores(),originalCoverage=coverage();
assert.equal(table('BE')[0].children[6].textContent,'40.00 €');
strategies[0].click();
assert.equal(root.dataset.activeStrategy,'quantile_based');assert.equal(strategies[0].attrs['aria-pressed'],'true');
assert.deepEqual(scores(),originalScores);assert.deepEqual(coverage(),originalCoverage);
assert.equal(table('BE')[0].children[0].textContent,'Storm calibré');
assert.equal(table('BE')[0].children[6].textContent,'—');
assert.equal(table('BE')[0].children[6].title,'Calibration incomplete');
assert.ok(table('BE').every(row=>row.children[6].className!=='rolling-best'));
assert.match(cards.get('BE').lookup['.rolling-pnl-note'].textContent,/P&L ranking disabled/);
assert.equal(table('DE')[1].children[6].textContent,'50.00 €');
frequencies[1].click();assert.match(coverage()[1],/90 complete daily pairs/);
assert.equal(table('DE')[1].children[6].textContent,'50.00 €');
periods[0].click();assert.match(coverage()[1],/7 complete daily pairs/);
assert.equal(table('DE')[1].children[6].textContent,'50.00 €');
strategies[1].click();assert.equal(table('DE')[1].children[6].textContent,'45.00 €');
cards.get('DE').lookup['thead th'][2].lookup.button.click();
assert.equal(table('DE')[0].dataset.provider,'model');
assert.equal(cards.get('DE').lookup['thead th'][2].attrs['aria-sort'],'ascending');
'''
    result = subprocess.run([node, "-e", harness, str(fixture)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
