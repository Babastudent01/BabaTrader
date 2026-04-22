"""
risk/sizing.py
Position sizing methods.
Determines how many shares to buy given portfolio state and signal.
"""
from __future__ import annotations

import logging

from config import Settings
from strategy.signal import Signal

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    Computes position size (number of shares) for a given signal.

    Supported methods (configured via risk.sizing_method):
        fixed      — Fixed dollar amount per trade.
        percent    — Fixed percentage of total portfolio equity.
        atr_based  — Risk a fixed % of equity per ATR unit (volatility-adjusted).
        kelly      — Simplified Kelly criterion (requires win_rate and avg_win/loss).

    All methods respect max_risk_per_trade_pct as an absolute cap.
    """

    def __init__(self, settings: Settings) -> None:
        risk_cfg = self._cfg = settings.risk
        exec_cfg = settings.execution
        self._method: str          = str(risk_cfg.get("sizing_method", "percent")).lower()
        self._fixed_amount: float  = float(risk_cfg.get("fixed_amount", 1000.0))
        self._position_pct: float  = float(risk_cfg.get("position_pct", 0.10))
        self._atr_mult: float      = float(risk_cfg.get("atr_risk_multiplier", 2.0))
        self._max_risk_pct: float  = float(risk_cfg.get("max_risk_per_trade_pct", 0.02))
        self._stop_loss_pct: float = float(risk_cfg.get("stop_loss_pct", 0.05))
        # Confidence-weighted sizing
        self._pct_min: float       = float(risk_cfg.get("position_pct_min", 0.05))
        self._pct_max: float       = float(risk_cfg.get("position_pct_max", 0.15))
        # Use the strategy confidence_threshold as the lower anchor
        self._conf_threshold: float = float(
            settings.strategy.get("confidence_threshold", 0.70)
        )
        # Fractional shares — when True, quantities are rounded to 3 d.p.
        # When False, quantities are floored to whole shares.
        self._fractional: bool = bool(exec_cfg.get("allow_fractional_shares", False))

    def compute_quantity(
        self,
        signal: Signal,
        portfolio_equity: float,
        current_price: float | None = None,
    ) -> float:
        """
        Compute the number of shares to buy for a given signal.

        Parameters
        ----------
        signal : Signal
            The trading signal (used for price and ATR).
        portfolio_equity : float
            Total portfolio value (cash + positions).
        current_price : float | None
            Override price (uses signal.price if None).

        Returns
        -------
        float
            Number of shares to buy (floored to whole shares, minimum 1).
            Returns 0 if sizing is not possible.
        """
        price = current_price or signal.price
        if price <= 0 or portfolio_equity <= 0:
            return 0.0

        # Compute raw dollar amount to invest
        if self._method == "fixed":
            dollar_amount = self._fixed_amount

        elif self._method == "percent":
            dollar_amount = portfolio_equity * self._position_pct

        elif self._method == "atr_based":
            dollar_amount = self._atr_based_sizing(signal, portfolio_equity, price)

        elif self._method == "kelly":
            dollar_amount = self._kelly_sizing(signal, portfolio_equity)

        elif self._method == "confidence_weighted":
            dollar_amount = self._confidence_weighted_sizing(signal, portfolio_equity)

        else:
            logger.warning("Unknown sizing method '%s', using percent.", self._method)
            dollar_amount = portfolio_equity * self._position_pct

        # Apply max risk cap
        max_dollar = portfolio_equity * self._max_risk_pct / max(self._stop_loss_pct, 0.001)
        dollar_amount = min(dollar_amount, max_dollar)

        # Convert dollar amount to shares
        quantity = dollar_amount / price
        if self._fractional:
            # Round to 3 decimal places (IBKR minimum fractional unit)
            quantity = round(quantity, 3)
            quantity = max(0.0, quantity)
        else:
            # Floor to whole shares
            quantity = max(0.0, float(int(quantity)))

        logger.debug(
            "PositionSizer [%s%s]: %s @ %.2f → %.3f shares (%.2f EUR).",
            self._method,
            "/fractional" if self._fractional else "/whole",
            signal.ticker, price, quantity, quantity * price,
        )
        return quantity

    def compute_stop_loss(self, entry_price: float, atr: float = 0.0) -> float:
        """
        Compute stop loss price.
        Uses ATR-based stop if ATR is available, else percentage-based.
        """
        if atr > 0 and self._atr_mult > 0:
            return entry_price - self._atr_mult * atr
        return entry_price * (1 - self._stop_loss_pct)

    def compute_take_profit(self, entry_price: float) -> float:
        """Compute take profit price based on configured percentage."""
        take_profit_pct = float(self._cfg.get("take_profit_pct", 0.15))
        return entry_price * (1 + take_profit_pct)

    # ── Sizing methods ────────────────────────────────────────────────────────

    def _atr_based_sizing(
        self,
        signal: Signal,
        portfolio_equity: float,
        price: float,
    ) -> float:
        """
        ATR-based position sizing.
        Risk = max_risk_pct * equity per trade.
        Stop distance = atr_mult * ATR.
        Shares = Risk / Stop distance.
        """
        atr = signal.atr
        if atr <= 0:
            # Fall back to percent sizing if ATR is unavailable
            logger.debug("ATR not available for %s, falling back to percent sizing.", signal.ticker)
            return portfolio_equity * self._position_pct

        risk_per_trade = portfolio_equity * self._max_risk_pct
        stop_distance  = self._atr_mult * atr
        shares = risk_per_trade / stop_distance
        return shares * price

    def _confidence_weighted_sizing(
        self,
        signal: Signal,
        portfolio_equity: float,
    ) -> float:
        """
        Confidence-weighted position sizing.

        Maps model confidence linearly from [confidence_threshold, 1.0]
        to [position_pct_min, position_pct_max].

        Examples (threshold=0.70, min=5%, max=15%):
            conf=0.70  →  5.0%  of portfolio
            conf=0.80  →  8.3%  of portfolio
            conf=0.90  → 11.7%  of portfolio
            conf=1.00  → 15.0%  of portfolio
        """
        conf = max(self._conf_threshold, min(1.0, signal.confidence))
        span = max(1.0 - self._conf_threshold, 1e-6)   # avoid div-by-zero
        t    = (conf - self._conf_threshold) / span     # normalise to [0, 1]
        pct  = self._pct_min + t * (self._pct_max - self._pct_min)

        logger.debug(
            "ConfidenceWeighted: conf=%.2f → %.1f%% of equity.",
            conf, pct * 100,
        )
        return portfolio_equity * pct

    def _kelly_sizing(self, signal: Signal, portfolio_equity: float) -> float:
        """
        Simplified Kelly criterion.
        Uses model confidence as a proxy for win probability.
        Kelly fraction = (p * b - q) / b
        where p = win prob, q = 1-p, b = avg_win/avg_loss ratio (assumed 1.5).
        """
        p = signal.confidence
        q = 1 - p
        b = 1.5  # Assumed reward/risk ratio — adjust based on historical data
        kelly_fraction = max(0.0, (p * b - q) / b)
        # Use half-Kelly for conservatism
        kelly_fraction = kelly_fraction * 0.5
        # Cap at position_pct
        kelly_fraction = min(kelly_fraction, self._position_pct)
        return portfolio_equity * kelly_fraction
