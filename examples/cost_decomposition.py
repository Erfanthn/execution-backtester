"""Show how much of a backtest's result is decided by execution assumptions.

The market is a driftless random walk, so the strategy has no real edge. Every number
below measures cost accounting, not predictive skill.
"""
import numpy as np

from execbt import BacktestConfig, moving_average_cross, run_backtest, summary, synthetic_market

SCENARIOS = {
    "mid price, no fee": BacktestConfig(execution="mid", fee_rate=0.0),
    "order book, no fee": BacktestConfig(execution="book", fee_rate=0.0),
    "order book, 0.1% fee": BacktestConfig(execution="book", fee_rate=0.001),
}
SEEDS = 20

print(f"{SEEDS} synthetic markets of 50,000 bars each\n")
print(f"{'scenario':<24}{'median return':>15}{'median fills':>15}")
for name, config in SCENARIOS.items():
    rows = []
    for seed in range(SEEDS):
        market = synthetic_market(seed=seed)
        signal = moving_average_cross(market.mid)
        rows.append(summary(run_backtest(market, signal, config)))
    print(f"{name:<24}{np.median([r['return'] for r in rows]):>15.2%}{np.median([r['fills'] for r in rows]):>15.0f}")
