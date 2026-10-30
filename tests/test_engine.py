import numpy as np
import pytest

from execbt import BacktestConfig, Market, max_drawdown, moving_average_cross, run_backtest, synthetic_market


def flat_market(n=6, price=100.0, half_spread=1.0, depth=1e9):
    mid = np.full(n, price)
    one = np.ones((n, 1))
    return Market(mid, (mid - half_spread)[:, None], one * depth, (mid + half_spread)[:, None], one * depth)


def test_zero_delay_is_rejected():
    with pytest.raises(ValueError):
        BacktestConfig(delay_bars=0)


def test_signal_fills_only_after_the_delay():
    signal = np.array([1, 1, 1, 1, 1, 1])
    r = run_backtest(flat_market(), signal, BacktestConfig(delay_bars=2, fee_rate=0.0))
    assert r.position[1] == 0
    assert r.position[2] > 0


def test_future_prices_cannot_change_past_results():
    a = synthetic_market(n=2000, seed=3)
    mid = a.mid.copy()
    mid[1500:] *= 2.0
    b = Market(mid, a.bid_px, a.bid_sz, a.ask_px, a.ask_sz)
    ra = run_backtest(a, moving_average_cross(a.mid))
    rb = run_backtest(b, moving_average_cross(b.mid))
    assert np.array_equal(ra.equity[:1500], rb.equity[:1500])


def test_round_trip_cost_matches_hand_calculation():
    signal = np.array([1, 0, 0, 0, 0, 0])
    r = run_backtest(flat_market(), signal, BacktestConfig(initial_quote=1010.0, fee_rate=0.01))
    # Buy 10 at 101, keep 9.9 after fee. Sell at 99 for 980.1, keep 970.299 after fee.
    assert r.equity[-1] == pytest.approx(970.299)
    assert r.n_fills == 2


def test_frictionless_round_trip_on_flat_prices_is_free():
    signal = np.array([1, 0, 1, 0, 0, 0])
    r = run_backtest(flat_market(), signal, BacktestConfig(execution="mid", fee_rate=0.0))
    assert r.equity[-1] == pytest.approx(100_000.0)


def test_thin_book_leaves_the_rest_for_later_bars():
    signal = np.ones(6, dtype=int)
    r = run_backtest(flat_market(depth=5.0), signal, BacktestConfig(initial_quote=1010.0, fee_rate=0.0))
    assert r.position[1] == pytest.approx(5.0)
    assert r.position[2] == pytest.approx(10.0)


def test_costs_never_improve_a_result():
    m = synthetic_market(n=5000, seed=1)
    s = moving_average_cross(m.mid)
    results = [
        run_backtest(m, s, BacktestConfig(execution="mid", fee_rate=0.0)).equity[-1],
        run_backtest(m, s, BacktestConfig(execution="book", fee_rate=0.0)).equity[-1],
        run_backtest(m, s, BacktestConfig(execution="book", fee_rate=0.001)).equity[-1],
    ]
    assert results[0] >= results[1] >= results[2]


def test_max_drawdown():
    assert max_drawdown(np.array([100.0, 120.0, 90.0, 130.0])) == pytest.approx(-0.25)


def test_invalid_signal_is_rejected():
    with pytest.raises(ValueError):
        run_backtest(flat_market(), np.array([0.5] * 6))
