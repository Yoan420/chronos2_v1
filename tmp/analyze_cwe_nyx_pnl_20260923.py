"""Read-only diagnostic of displayed PnL and daily trade decisions."""
from pathlib import Path
import json
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
EXPR = re.compile(r'<script id="rolling-performance-data" type="application/json">(.*?)</script>', re.S)
reports = []
daily = []
for day in range(19, 25):
    path = ROOT / f"runs/reports/model_storm/CWE_Model_Storm_2026-09-{day}.html"
    data = json.loads(EXPR.search(path.read_text(encoding="utf-8"))[1])
    summary = {"day": day, "nyx_better": 0, "storm_better": 0, "ties": 0, "comparisons": []}
    for strategy, zones in data["strategy_zones"].items():
        for zone in zones:
            for period, window in zone["windows"].items():
                providers = {r["key"]: r for r in window["frequencies"]["60min"]["providers"]}
                storm, model = providers["storm"], providers["model"]
                diff = model["daily_pnl"] - storm["daily_pnl"]
                summary["nyx_better" if diff > 1e-8 else "storm_better" if diff < -1e-8 else "ties"] += 1
                summary["comparisons"].append({"strategy": strategy, "zone": zone["zone"], "window": period,
                    "days": window["pnl_days"], "storm": storm["daily_pnl"], "nyx": model["daily_pnl"], "gain": diff,
                    "gain_percent": 100 * diff / abs(storm["daily_pnl"]) if storm["daily_pnl"] else None,
                    "storm_mae": storm["mae"], "nyx_mae": model["mae"]})
            if day != 24:
                continue
            window = zone["windows"]["90"]
            support = set(window["pnl_support_days"])
            records = [r for r in zone["pnl_audit"] if r["delivery_day"] in support]
            diffs, equal_pairs, wins, losses, ties, examples = [], 0, 0, 0, 0, []
            for record in records:
                s, n = record["providers"]["storm"], record["providers"]["model"]
                diff = n["pnl_eur"] - s["pnl_eur"]
                diffs.append(diff)
                equal_pairs += (s["buy_index"], s["sell_index"]) == (n["buy_index"], n["sell_index"])
                wins += diff > 1e-8
                losses += diff < -1e-8
                ties += abs(diff) <= 1e-8
                if diff < -1e-8:
                    examples.append({"day": record["delivery_day"], "storm": s["pnl_eur"], "nyx": n["pnl_eur"]})
            positive = sorted([d for d in diffs if d > 0], reverse=True)
            daily.append({"strategy": strategy, "zone": zone["zone"], "days": len(records),
                "nyx_win_days": wins, "storm_win_days": losses, "equal_pnl_days": ties,
                "same_pair_days": equal_pairs, "net_advantage": sum(diffs),
                "positive_advantages_total": sum(positive), "top5_positive_advantages": sum(positive[:5]),
                "mean_gap": statistics.mean(diffs), "median_gap": statistics.median(diffs),
                "example_storm_better": examples[-2:]})
    reports.append(summary)
output = {"reports": reports, "daily_90": daily}
target = ROOT / "tmp/cwe_nyx_pnl_diagnostic_20260923.json"
target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"report_counts": [{k: v for k, v in r.items() if k != "comparisons"} for r in reports],
                  "latest": reports[-1]["comparisons"], "daily_90": daily,
                  "saved": str(target)}, ensure_ascii=False, indent=2))
