"""Fill orders by walking the levels of one side of an order book."""
from __future__ import annotations

import numpy as np


def buy_with_quote(ask_px: np.ndarray, ask_sz: np.ndarray, quote: float) -> tuple[float, float]:
    """Spend up to `quote` against asks (best first). Returns (base bought, quote spent)."""
    base = spent = 0.0
    for px, sz in zip(ask_px, ask_sz):
        remaining = quote - spent
        if remaining <= 0:
            break
        take = min(sz, remaining / px)
        base += take
        spent += take * px
    return base, spent


def sell_base(bid_px: np.ndarray, bid_sz: np.ndarray, base: float) -> tuple[float, float]:
    """Sell up to `base` into bids (best first). Returns (base sold, quote received)."""
    sold = received = 0.0
    for px, sz in zip(bid_px, bid_sz):
        remaining = base - sold
        if remaining <= 0:
            break
        take = min(sz, remaining)
        sold += take
        received += take * px
    return sold, received
