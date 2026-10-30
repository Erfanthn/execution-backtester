"""Event-loop backtester for a long/flat strategy.

Timing rule: a signal computed from bar t can trade no earlier than bar t + delay_bars.
Orders walk the visible book. Depth that runs out leaves the rest to retry on the next bar.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .book import buy_with_quote, sell_base
from .market import Market


@dataclass(frozen=True)
class BacktestConfig:
    initial_quote: float = 100_000.0
    fee_rate: float = 0.001          # charged on what is received, per fill
    delay_bars: int = 1
    execution: str = "book"          # "book", or "mid" for the frictionless comparison
    min_notional: float = 1.0        # smaller balances are treated as dust

    def __post_init__(self) -> None:
        if self.delay_bars < 1:
            raise ValueError("delay_bars must be >= 1: a same-bar fill is look-ahead")
        if not 0 <= self.fee_rate < 1:
            raise ValueError("fee_rate must be in [0, 1)")
        if self.execution not in ("book", "mid"):
            raise ValueError("execution must be 'book' or 'mid'")


@dataclass(frozen=True)
class BacktestResult:
    equity: np.ndarray        # marked at mid, in quote currency
    position: np.ndarray      # base units held at each bar's close
    n_fills: int
    fees_paid: float          # in quote currency
    quote_traded: float
    benchmark: np.ndarray     # buy at the first bar's mid and hold


def run_backtest(market: Market, signal: np.ndarray, config: BacktestConfig = BacktestConfig()) -> BacktestResult:
    signal = np.asarray(signal)
    n = len(market)
    if len(signal) != n:
        raise ValueError("signal length must match market length")
    if not np.isin(signal, (0, 1)).all():
        raise ValueError("signal must contain only 0 (flat) and 1 (long)")

    quote, base = config.initial_quote, 0.0
    fees = traded = 0.0
    fills = 0
    equity, position = np.empty(n), np.empty(n)

    for t in range(n):
        want_long = t >= config.delay_bars and signal[t - config.delay_bars] == 1
        mid = market.mid[t]

        if want_long and quote > config.min_notional:
            if config.execution == "mid":
                got, spent = quote / mid, quote
            else:
                got, spent = buy_with_quote(market.ask_px[t], market.ask_sz[t], quote)
            fee = got * config.fee_rate
            quote -= spent
            base += got - fee
            fees += fee * mid
            traded += spent
            fills += 1
        elif not want_long and base * mid > config.min_notional:
            if config.execution == "mid":
                sold, received = base, base * mid
            else:
                sold, received = sell_base(market.bid_px[t], market.bid_sz[t], base)
            fee = received * config.fee_rate
            base -= sold
            quote += received - fee
            fees += fee
            traded += received
            fills += 1

        equity[t] = quote + base * mid
        position[t] = base

    benchmark = config.initial_quote * market.mid / market.mid[0]
    return BacktestResult(equity, position, fills, fees, traded, benchmark)
