"""Performance metrics computed from an equity curve."""
from __future__ import annotations

import numpy as np

from .engine import BacktestResult


def max_drawdown(equity: np.ndarray) -> float:
    """Largest peak-to-trough fall, as a negative fraction."""
    return float((equity / np.maximum.accumulate(equity) - 1.0).min())


def sharpe(equity: np.ndarray, bars_per_year: float) -> float:
    """Annualised Sharpe ratio of log returns, with a zero risk-free rate."""
    r = np.diff(np.log(equity))
    sd = r.std(ddof=1)
    return 0.0 if sd == 0 else float(r.mean() / sd * np.sqrt(bars_per_year))


def summary(result: BacktestResult, bars_per_year: float = 525_600) -> dict[str, float]:
    start = result.equity[0]
    return {
        "return": float(result.equity[-1] / start - 1),
        "benchmark_return": float(result.benchmark[-1] / result.benchmark[0] - 1),
        "max_drawdown": max_drawdown(result.equity),
        "sharpe": sharpe(result.equity, bars_per_year),
        "fills": float(result.n_fills),
        "fees_paid": result.fees_paid,
        "turnover": result.quote_traded / start,
    }
