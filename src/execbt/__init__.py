"""Order-book-aware backtesting."""
from .book import buy_with_quote, sell_base
from .engine import BacktestConfig, BacktestResult, run_backtest
from .market import Market, synthetic_market
from .metrics import max_drawdown, sharpe, summary
from .strategies import moving_average_cross

__all__ = [
    "BacktestConfig", "BacktestResult", "Market", "buy_with_quote", "max_drawdown",
    "moving_average_cross", "run_backtest", "sell_base", "sharpe", "summary", "synthetic_market",
]
