import numpy as np
import pytest

from chronos2_hourly.model_storm_bess import select_bess_trade, simulate_bess_day


def profile(buy=10., sell=100., n=24):
    return [buy] * 12 + [sell] * (n - 12)


def test_one_cycle_energy_efficiency_cost_and_stable_earliest_pair():
    forecast = profile()
    trade = simulate_bess_day(forecast, forecast)
    assert (trade['buy_index'], trade['sell_index']) == (0, 12)
    assert trade['grid_buy_mwh'] == pytest.approx(1 / .95)
    assert trade['grid_sell_mwh'] == .95
    assert trade['pnl_eur'] == pytest.approx(.95 * 100 - 10 / .95 - 25)
    assert trade['operating_cost_eur'] == 25
    assert trade['executed']


def test_selection_respects_buy_before_sell_and_never_uses_realized_prices():
    forecast = [100.] * 12 + [10.] * 12
    assert not select_bess_trade(forecast)['proposed']
    forecast = profile()
    good = simulate_bess_day(forecast, profile())
    loss = simulate_bess_day(forecast, profile(200, -20))
    assert (good['buy_index'], good['sell_index']) == (loss['buy_index'], loss['sell_index'])
    assert loss['executed'] and loss['pnl_eur'] < 0
    assert loss['predicted_profit_eur'] == good['predicted_profit_eur']


def test_nonpositive_forecast_profit_has_no_transaction_and_no_cost():
    trade = simulate_bess_day([0.] * 24, profile())
    assert not trade['proposed'] and not trade['executed']
    assert trade['pnl_eur'] == trade['operating_cost_eur'] == 0
    exact = profile(0, 25 / .95)
    assert not select_bess_trade(exact)['proposed']


def test_qb_uses_buy_upper_sell_lower_and_rejects_whole_loop_without_cost():
    forecast = profile()
    lo, hi = np.asarray(forecast) - 5, np.asarray(forecast) + 5
    accepted = simulate_bess_day(forecast, forecast, strategy='quantile_based', lower=lo, upper=hi)
    assert accepted['buy_limit_eur_mwh'] == 15
    assert accepted['sell_limit_eur_mwh'] == 95
    rejected = simulate_bess_day(forecast, profile(100, 10), strategy='quantile_based', lower=lo, upper=hi)
    assert rejected['proposed'] and not rejected['executed']
    assert rejected['pnl_eur'] == rejected['operating_cost_eur'] == 0
    assert rejected['loop_surplus_eur'] < 0


def test_loop_acceptance_is_combined_surplus_not_two_independent_orders():
    forecast = profile()
    # Buy clears above its individual limit, compensated by better sell price.
    trade = simulate_bess_day(forecast, profile(25, 150), strategy='quantile_based',
                              lower=np.asarray(forecast) - 5, upper=np.asarray(forecast) + 5)
    assert trade['executed']
    assert trade['loop_surplus_eur'] == pytest.approx((15 - 25) / .95 + .95 * (150 - 95))


def test_quantile_strategy_can_execute_a_loss_and_never_clips_it_to_zero():
    forecast = profile()
    trade = simulate_bess_day(forecast, profile(60, 70), strategy='quantile_based',
                              lower=np.asarray(forecast) - 100, upper=np.asarray(forecast) + 100)
    assert trade['executed'] and trade['pnl_eur'] < 0


@pytest.mark.parametrize('n', [23, 24, 25])
def test_physical_dst_hours_are_all_available_and_last_hour_can_be_selected(n):
    prices = [10.] * n
    prices[-1] = 100
    trade = simulate_bess_day(prices, prices)
    assert trade['sell_index'] == n - 1


def test_negative_prices_still_apply_physical_energy_losses_and_cost():
    prices = profile(-100, -20)
    trade = simulate_bess_day(prices, prices)
    assert trade['pnl_eur'] == pytest.approx(-19 + 100 / .95 - 25)


@pytest.mark.parametrize('prices', [[0.] * 22, [0.] * 26, [True] * 24, [None] * 24, [np.nan] * 24])
def test_invalid_price_vectors_are_not_silently_scored(prices):
    with pytest.raises(ValueError):
        select_bess_trade(prices)


@pytest.mark.parametrize('lo,hi', [(None, None), ([20.] * 24, [200.] * 24), ([0.] * 24, [1.] * 24)])
def test_missing_or_crossed_quantiles_are_not_replaced_with_the_point_forecast(lo, hi):
    with pytest.raises(ValueError):
        simulate_bess_day(profile(), profile(), strategy='quantile_based', lower=lo, upper=hi)
