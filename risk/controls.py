"""
risk/controls.py
Portfolio-level risk controls and circuit breakers.
Prevents trading when portfolio risk limits are breached.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from brokers.base import BrokerInterface
from config import Settings

logger = logging.getLogger(__name__)


@dataclass
class RiskCheckResult:
    """Result of a portfolio risk check."""
    passed: bool
    reason: str = ""


class RiskControls:
    """
    Portfolio-level risk controls.

    Checks performed before any order is placed:
    1. Max open positions — prevents over-diversification / over-concentration.
    2. Max drawdown circuit breaker — halts trading if portfolio drops too much.
    3. Max daily loss — halts trading if daily loss exceeds threshold.
    4. Max position concentration — no single position > X% of portfolio.
    5. Duplicate position guard — prevents buying a ticker already held.

    All thresholds are configured in settings.yaml under risk.controls.
    """

    def __init__(self, settings: Settings) -> None:
        ctrl_cfg = settings.risk.get("controls", {}) or {}
        self._max_open_positions: int   = int(ctrl_cfg.get("max_open_positions", 10))
        self._max_drawdown_pct: float   = float(ctrl_cfg.get("max_drawdown_pct", 0.15))
        self._max_daily_loss_pct: float = float(ctrl_cfg.get("max_daily_loss_pct", 0.05))
        self._max_concentration: float  = float(ctrl_cfg.get("max_concentration_pct", 0.20))

        self._peak_equity: float = 0.0
        self._day_start_equity: float = 0.0
        self._trading_halted: bool = False

    def initialise(self, equity: float) -> None:
        """Call once at startup with the current portfolio equity."""
        self._peak_equity = equity
        self._day_start_equity = equity
        logger.info("RiskControls initialised: equity=%.2f.", equity)

    def new_day(self, equity: float) -> None:
        """Call at the start of each trading day to reset daily loss tracking."""
        self._day_start_equity = equity
        self._trading_halted = False
        logger.info("RiskControls: new day, equity=%.2f.", equity)

    def check_portfolio(self, broker: BrokerInterface) -> RiskCheckResult:
        """
        Run all portfolio-level risk checks.

        Parameters
        ----------
        broker : BrokerInterface
            Live broker connection to query positions and account info.

        Returns
        -------
        RiskCheckResult
            passed=True if all checks pass, False with reason if any fail.
        """
        if self._trading_halted:
            return RiskCheckResult(False, "Trading halted by circuit breaker.")

        try:
            account = broker.get_account_info()
            positions = broker.get_positions()
        except Exception as exc:  # noqa: BLE001
            return RiskCheckResult(False, f"Failed to query broker: {exc}")

        equity = account.total_equity

        # Update peak equity
        if equity > self._peak_equity:
            self._peak_equity = equity

        # ── Max drawdown check ────────────────────────────────────────────────
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            if drawdown >= self._max_drawdown_pct:
                self._trading_halted = True
                reason = (
                    f"Max drawdown breached: {drawdown:.1%} >= {self._max_drawdown_pct:.1%}. "
                    "Trading halted."
                )
                logger.critical("RISK CONTROL: %s", reason)
                return RiskCheckResult(False, reason)

        # ── Daily loss check ──────────────────────────────────────────────────
        if self._day_start_equity > 0:
            daily_loss = (self._day_start_equity - equity) / self._day_start_equity
            if daily_loss >= self._max_daily_loss_pct:
                self._trading_halted = True
                reason = (
                    f"Max daily loss breached: {daily_loss:.1%} >= "
                    f"{self._max_daily_loss_pct:.1%}. Trading halted for today."
                )
                logger.critical("RISK CONTROL: %s", reason)
                return RiskCheckResult(False, reason)

        # ── Max open positions check ──────────────────────────────────────────
        n_positions = len(positions)
        if n_positions >= self._max_open_positions:
            reason = (
                f"Max open positions reached: {n_positions} >= {self._max_open_positions}."
            )
            logger.warning("RISK CONTROL: %s", reason)
            return RiskCheckResult(False, reason)

        return RiskCheckResult(True)

    def check_new_position(
        self,
        ticker: str,
        quantity: float,
        price: float,
        broker: BrokerInterface,
    ) -> RiskCheckResult:
        """
        Check if a new position for a specific ticker is allowed.

        Parameters
        ----------
        ticker : str
            Ticker to buy.
        quantity : float
            Number of shares to buy.
        price : float
            Current price.
        broker : BrokerInterface
            Live broker connection.

        Returns
        -------
        RiskCheckResult
        """
        try:
            account   = broker.get_account_info()
            positions = broker.get_positions()
        except Exception as exc:  # noqa: BLE001
            return RiskCheckResult(False, f"Failed to query broker: {exc}")

        equity = account.total_equity

        # ── Duplicate position guard ──────────────────────────────────────────
        if ticker in positions:
            return RiskCheckResult(
                False,
                f"Already holding {ticker} — no duplicate positions allowed.",
            )

        # ── Concentration check ───────────────────────────────────────────────
        trade_value = quantity * price
        if equity > 0:
            concentration = trade_value / equity
            if concentration > self._max_concentration:
                reason = (
                    f"Position concentration {concentration:.1%} exceeds "
                    f"max {self._max_concentration:.1%} for {ticker}."
                )
                logger.warning("RISK CONTROL: %s", reason)
                return RiskCheckResult(False, reason)

        # ── Sufficient cash check ─────────────────────────────────────────────
        if trade_value > account.cash:
            reason = (
                f"Insufficient cash for {ticker}: need {trade_value:.2f}, "
                f"have {account.cash:.2f}."
            )
            logger.warning("RISK CONTROL: %s", reason)
            return RiskCheckResult(False, reason)

        return RiskCheckResult(True)

    def resume_trading(self) -> None:
        """Manually resume trading after a halt (use with caution)."""
        self._trading_halted = False
        logger.warning("RiskControls: Trading manually resumed.")

    @property
    def is_halted(self) -> bool:
        """Return True if trading has been halted by a circuit breaker."""
        return self._trading_halted
