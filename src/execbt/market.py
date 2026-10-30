"""Market container and a synthetic generator, so the project runs with no external data."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Market:
    mid: np.ndarray      # shape (n,)
    bid_px: np.ndarray   # shape (n, levels), best level first
    bid_sz: np.ndarray
    ask_px: np.ndarray
    ask_sz: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.mid)
        for name in ("bid_px", "bid_sz", "ask_px", "ask_sz"):
            if getattr(self, name).shape[0] != n:
                raise ValueError(f"{name} must have {n} rows")
        if np.any(self.bid_px[:, 0] >= self.ask_px[:, 0]):
            raise ValueError("crossed book: best bid must be below best ask")

    def __len__(self) -> int:
        return len(self.mid)


def synthetic_market(
    n: int = 50_000,
    seed: int = 0,
    start: float = 100.0,
    levels: int = 5,
    calm_vol: float = 0.0004,
    stressed_vol: float = 0.0015,
    switch_prob: float = 0.002,
    mean_depth: float = 300.0,
) -> Market:
    """Driftless two-regime random walk with a spread that widens with volatility.

    It contains no predictable pattern by construction, so any strategy has an expected
    gross return of about zero. That makes it a good test of cost accounting.
    """
    rng = np.random.default_rng(seed)
    flips = rng.random(n) < switch_prob
    stressed = np.cumsum(flips) % 2 == 1
    vol = np.where(stressed, stressed_vol, calm_vol)
    mid = start * np.exp(np.cumsum(rng.normal(-0.5 * vol**2, vol)))

    half_spread = mid * (0.5 * vol + 0.00005)
    tick_steps = np.arange(levels)[None, :]
    step = (mid * 0.5 * vol)[:, None]
    bid_px = (mid - half_spread)[:, None] - step * tick_steps
    ask_px = (mid + half_spread)[:, None] + step * tick_steps
    bid_sz = rng.exponential(mean_depth, (n, levels)) + 1.0
    ask_sz = rng.exponential(mean_depth, (n, levels)) + 1.0
    return Market(mid, bid_px, bid_sz, ask_px, ask_sz)
