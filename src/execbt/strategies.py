"""Demonstration strategies only. They exist to exercise the engine, not to make money."""
from __future__ import annotations

import numpy as np


def _trailing_mean(x: np.ndarray, window: int) -> np.ndarray:
    total = np.concatenate(([0.0], np.cumsum(x)))
    out = np.full(len(x), np.nan)
    out[window - 1:] = (total[window:] - total[:-window]) / window
    return out


def moving_average_cross(price: np.ndarray, fast: int = 20, slow: int = 100) -> np.ndarray:
    """1 while the fast trailing mean is above the slow one. Uses data up to each bar only."""
    if not 0 < fast < slow:
        raise ValueError("need 0 < fast < slow")
    f, s = _trailing_mean(price, fast), _trailing_mean(price, slow)
    return ((f > s) & ~np.isnan(s)).astype(int)
