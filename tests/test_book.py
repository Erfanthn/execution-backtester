import numpy as np
import pytest

from execbt import buy_with_quote, sell_base

PX_ASK, PX_BID = np.array([100.0, 101.0]), np.array([99.0, 98.0])
SZ = np.array([10.0, 10.0])


def test_buy_stays_on_best_level_when_it_is_deep_enough():
    assert buy_with_quote(PX_ASK, SZ, 500.0) == pytest.approx((5.0, 500.0))


def test_buy_walks_to_a_worse_level():
    base, spent = buy_with_quote(PX_ASK, SZ, 1505.0)
    assert base == pytest.approx(15.0)
    assert spent == pytest.approx(1505.0)


def test_buy_stops_when_depth_runs_out():
    base, spent = buy_with_quote(PX_ASK, SZ, 1e9)
    assert base == pytest.approx(20.0)
    assert spent == pytest.approx(2010.0)


def test_sell_walks_down_the_bids():
    assert sell_base(PX_BID, SZ, 15.0) == pytest.approx((15.0, 1480.0))
