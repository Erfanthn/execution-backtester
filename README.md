# execution-backtester

A small backtesting engine that charges a strategy for how orders actually get filled:
execution delay, spread, limited book depth and fees.

Many simple backtests fill at the mid price on the same bar as the signal. This engine
exists to measure how much that assumption can flatter a result.

## Result

`examples/cost_decomposition.py` runs one moving-average strategy on 20 synthetic markets
of 50,000 bars each. The markets are driftless random walks, so the strategy has no real
edge, and the differences below come only from execution assumptions.

| Execution assumption | Median return | Median fills |
|---|---:|---:|
| Mid price, no fee | -5.44% | 699 |
| Order book, no fee | -51.57% | 770 |
| Order book, 0.1% fee | -74.50% | 740 |

On this synthetic market, spread and depth cost more than the explicit fee. Fill counts
differ because an order that exhausts the visible depth finishes on later bars.

## Design

- **No same-bar fills.** `delay_bars` must be at least 1, and a test checks that changing
  future prices cannot change past equity.
- **Orders walk the book.** Each fill consumes levels in price order, and any remainder
  retries on the next bar.
- **Fees are charged on what is received:** in base currency on a buy, in quote on a sell.
- **Costs are checked by hand.** One test reproduces a full round trip with arithmetic
  written out in a comment.

## Limitations

- Long/flat only, with the whole balance traded at each switch.
- Market orders only. There is no queue model or limit-order simulation.
- A fill does not affect later book states, so there is no market impact beyond the bar.
- The synthetic market is a test fixture, not a model of any real market.

## Usage

```bash
pip install -e .[dev]
pytest
python examples/cost_decomposition.py
```

```python
from execbt import BacktestConfig, moving_average_cross, run_backtest, summary, synthetic_market

market = synthetic_market(seed=1)
signal = moving_average_cross(market.mid)
print(summary(run_backtest(market, signal, BacktestConfig(fee_rate=0.001, delay_bars=2))))
```

The included strategy is a demonstration only. It is not a trading recommendation, and
none of these numbers describe live trading performance.
