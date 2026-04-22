"""
strategy/signal.py
Signal dataclass and enumerations.
A Signal represents a trading recommendation for a single ticker at a point in time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SignalType(str, Enum):
    """Trading signal direction."""
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """
    A trading signal for a single ticker.

    Attributes
    ----------
    ticker : str
        Ticker symbol.
    signal_type : SignalType
        BUY, SELL, or HOLD.
    confidence : float
        Model confidence in the signal (0.0 to 1.0).
        Only signals above the configured threshold are acted upon.
    timestamp : datetime
        When the signal was generated.
    price : float
        Current price at signal generation time.
    atr : float
        Current ATR value (used for position sizing and stop loss).
    metadata : dict
        Optional additional context (model name, feature values, etc.).
    """
    ticker: str
    signal_type: SignalType
    confidence: float
    timestamp: datetime
    price: float
    atr: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_actionable(self, threshold: float = 0.60) -> bool:
        """Return True if confidence exceeds the threshold and signal is BUY or SELL."""
        return (
            self.signal_type in (SignalType.BUY, SignalType.SELL)
            and self.confidence >= threshold
        )

    def __repr__(self) -> str:
        return (
            f"Signal({self.ticker} {self.signal_type.value} "
            f"@ {self.price:.2f} conf={self.confidence:.3f} "
            f"ts={self.timestamp.strftime('%Y-%m-%d %H:%M')})"
        )
