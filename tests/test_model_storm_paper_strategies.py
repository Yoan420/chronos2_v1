from copy import deepcopy
import json

import pandas as pd
import pytest

from chronos2_hourly.model_storm_rolling import build_rolling_performance


def sample(days=75):
    index = pd.date_range('2026-01-01', periods=24 * days, freq='h', tz='Europe/Paris').tz_convert('UTC')
    rows = []
    for i, stamp in enumerate(index):
        hour = stamp.tz_convert('Europe/Paris').hour
        point = 10. if hour < 12 else 100.
        rows.append({'timestamp_utc': stamp.isoformat(), 'storm': point, 'model': point + 2,
                     'model_p10': point - 10, 'model_p90': point + 14,
                     'observed': point + (i // 24 % 7 - 3) * 2,
                     'storm_dashboard_cache': point + 999})
    return {'delivery_day': str(index[-1].tz_convert('Europe/Paris').date()),
            'zones': [{'zone': 'FR', 'timezone': 'Europe/Paris', 'rolling_history': {'rows': rows,
                       'sources': {'storm_dashboard_cache': {'status': 'complete'}}}}]}


def test_both_strategies_share_complete_days_after_calibration_and_keep_accuracy_identical():
    result = build_rolling_performance(sample())
    assert result['schema_version'] == 3
    assert result['source_scope'] == 'internal_completed'
    assert 'scopes' not in result and 'dashboard_methodology' not in result
    assert result['quantile_alpha'] == .8
    qb = result['strategy_zones']['quantile_based'][0]
    ub = result['strategy_zones']['unlimited_bid'][0]
    for window in ('7', '30', '365'):
        q, u = qb['windows'][window], ub['windows'][window]
        assert q['pnl_support_days'] == u['pnl_support_days']
        assert q['pnl_days'] == u['pnl_days'] == min(int(window), 15)
        for qrow, urow in zip(q['frequencies']['60min']['providers'], u['frequencies']['60min']['providers']):
            for metric in ('mae', 'rmse', 'bias', 'r2', 'samples'):
                assert qrow[metric] == urow[metric]
            assert qrow['pnl_days'] == urow['pnl_days']
            assert qrow['daily_pnl'] == pytest.approx(qrow['total_pnl'] / qrow['pnl_days'])
    assert qb['windows']['7']['frequencies']['60min']['providers'][0]['label'] == 'Storm calibré'
    assert qb['windows']['7']['frequencies']['day']['providers'][0]['daily_pnl'] == qb['windows']['7']['frequencies']['60min']['providers'][0]['daily_pnl']
    json.dumps(result, allow_nan=False)


def test_missing_nyx_quantile_removes_same_day_from_both_strategies_both_providers():
    data = sample()
    data['zones'][0]['rolling_history']['rows'][-5]['model_p10'] = None
    result = build_rolling_performance(data)
    for strategy in result['strategies']:
        window = result['strategy_zones'][strategy][0]['windows']['7']
        assert window['pnl_days'] == 6
        assert window['pnl_missing_quantile_days']['model'] == 1
        assert all(row['pnl_days'] == 6 for row in window['frequencies']['60min']['providers'])
        assert window['paired_hours'] == 7 * 24


def test_warmup_is_unavailable_not_zero_and_cache_changes_cannot_affect_any_result():
    data = sample(days=60)
    result = build_rolling_performance(data)
    for strategy in result['strategies']:
        for row in result['strategy_zones'][strategy][0]['windows']['365']['frequencies']['60min']['providers']:
            assert row['daily_pnl'] is None and row['total_pnl'] is None
            assert row['pnl_days'] == 0 and row['pnl_comparison_eligible'] is False
            assert row['pnl_unavailable_reason']
    changed = deepcopy(data)
    for row in changed['zones'][0]['rolling_history']['rows']:
        row['storm_dashboard_cache'] *= -100
    assert build_rolling_performance(changed) == result


def test_actuals_cannot_change_same_day_pair_selection_or_calibrated_limits():
    data = sample()
    original = build_rolling_performance(data)
    for row in data['zones'][0]['rolling_history']['rows'][-24:]:
        row['observed'] += 10000
    altered = build_rolling_performance(data)
    for strategy in original['strategies']:
        a = original['strategy_zones'][strategy][0]['pnl_audit'][-1]['providers']
        b = altered['strategy_zones'][strategy][0]['pnl_audit'][-1]['providers']
        for provider in ('storm', 'model'):
            for field in ('buy_index', 'sell_index', 'buy_limit_eur_mwh', 'sell_limit_eur_mwh', 'predicted_profit_eur'):
                assert a[provider][field] == b[provider][field]
