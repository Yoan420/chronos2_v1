"""Read-only Saturn discovery; writes only fresh diagnostic catalogue copies."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://saturn-energyscan.gem.myengie.com//api"


def flatten(doc):
    result = []
    for source, values in doc.items():
        if isinstance(values, list):
            for value in values:
                if isinstance(value, (list, tuple)):
                    result.append(dict(name=str(value[0]), kind=str(value[1]) if len(value)>1 else None, source=source))
                else:
                    result.append(dict(name=str(value), kind=None, source=source))
        else:
            result.append(dict(name=str(source), kind=str(values), source=None))
    return result


def main():
    session = requests.Session()
    results = {}
    for entity in ("series", "group"):
        destination = ROOT / "tmp" / ("saturn_demand_deep_catalog.json" if entity == "series" else "saturn_demand_deep_group_catalog.json")
        if destination.exists():
            doc = json.loads(destination.read_text(encoding="utf8"))
            print("CATALOG_REUSE", entity, flush=True)
        else:
            response = session.get(BASE+f"/{entity}/catalog", params={"allsources": True}, timeout=45)
            print("CATALOG_HTTP", entity, response.status_code, len(response.content), flush=True)
            response.raise_for_status()
            doc = response.json()
            if not isinstance(doc, dict):
                raise ValueError("Unexpected catalogue schema")
            # Fresh diagnostic cache only; exclusive creation and no production paths.
            with destination.open("x", encoding="utf8") as out:
                json.dump(doc, out)
        entries = flatten(doc)
        results[entity] = entries
        print("CATALOG_COUNT", entity, len(entries), "sources", {k: len(v) if isinstance(v,list) else None for k,v in doc.items()}, flush=True)
        categories = {
            "curve_non_forward": lambda n: "curv" in n and "forward" not in n,
            "aggregated_merit_order": lambda n: any(t in n for t in ["aggregat", "merit", "orderbook", "order.book", "order_book", "bidding", "bidcurve", "bid_curve"]),
            "demand_response": lambda n: any(t in n for t in ["effac", "nebef", "nebco", "demand_response", "demand.response", "demandresponse", "elastic", "destruct", "load.shed", "load_shed", "dsr"]),
            "auction_non_price_volume": lambda n: any(t in n for t in ["auction", "epex", "nemo"]) and not any(t in n for t in ["price.cleared", "volume.cleared", "price.spot", "da.epex", "epex.spot.volume", "yearly", "year.", "eua", "carbon."]),
            "power_bid_offer_order_buy_sell": lambda n: n.startswith("power.") and bool(re.search(r"(?:^|[._])(?:bid|bids|ask|offer|offers|order|orders|buy|sell|purchase|purchases|sales)(?:[._]|$)",n)) and not any(t in n for t in ["intraday", "afrr", "frr", "xbid", "id100", "activated", ".ev.", ".otc.", "system.buy", "system_buy", "system_sell", "system.buy.sell", "adjustment_volume"]),
        }
        for category, match in categories.items():
            found = [e for e in entries if match(e["name"].lower())]
            print("CATEGORY", entity, category, len(found), json.dumps(found, ensure_ascii=True), flush=True)
    session.close()


if __name__ == "__main__":
    main()
