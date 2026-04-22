"""
backtesting/portfolio.py
Portfolio tracker for backtesting.
Tracks cash, positions, equity curve, and trade history.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """A completed round-trip trade (entry + exit)."""
    ticker:       str
    entry_date:   datetime
    exit_date:    datetime
    entry_price:  float
    exit_price:   float
    quantity:     float
    side:         str  # "LONG"
    pnl:          float
    pnl_pct:      float
    commission:   float
    exit_reason:  str  = "SIGNAL"   # SIGNAL | STOP_LOSS | TAKE_PROFIT | END_OF_BACKTEST
    holding_days: int  = 0


@dataclass
class OpenPosition:
    """An open position during backtesting."""
    ticker:      str
    entry_date:  datetime
    entry_price: float
    quantity:    float
    stop_loss:   float = 0.0
    take_profit: float = 0.0
    commission:  float = 0.0


class BacktestPortfolio:
    """
    Tracks portfolio state during a backtest.

    Maintains:
    - Cash balance
    - Open positions
    - Equity curve (one value per bar)
    - Completed trade log
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        commission_pct: float = 0.001,
        commission_min: float = 1.0,
        slippage_pct: float = 0.0005,
    ) -> None:
        self._initial_cash   = initial_cash
        self._cash           = initial_cash
        self._commission_pct = commission_pct
        self._commission_min = commission_min
        self._slippage_pct   = slippage_pct

        self._positions: dict[str, OpenPosition] = {}
        self._trades: list[Trade] = []
        self._equity_curve: list[dict[str, Any]] = []

    # ── Portfolio state ───────────────────────────────────────────────────────

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def initial_cash(self) -> float:
        return self._initial_cash

    def portfolio_value(self, prices: dict[str, float]) -> float:
        """Total portfolio value = cash + market value of all positions."""
        pos_value = sum(
            pos.quantity * prices.get(pos.ticker, pos.entry_price)
            for pos in self._positions.values()
        )
        return self._cash + pos_value

    def get_positions(self) -> dict[str, OpenPosition]:
        return dict(self._positions)

    def get_trades(self) -> list[Trade]:
        return list(self._trades)

    def get_equity_curve(self) -> pd.Series:
        """Return equity curve as a DatetimeIndex Series."""
        if not self._equity_curve:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self._equity_curve)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["equity"]

    # ── Order execution ───────────────────────────────────────────────────────

    def buy(
        self,
        ticker: str,
        quantity: float,
        price: float,
        date: datetime,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
    ) -> bool:
        """
        Open a long position.

        Returns True if the order was filled, False if insufficient cash.
        """
        if quantity <= 0:
            return False

        fill_price = price * (1 + self._slippage_pct)
        commission = max(self._commission_min, quantity * fill_price * self._commission_pct)
        total_cost = quantity * fill_price + commission

        if total_cost > self._cash:
            logger.debug(
                "BUY %s: insufficient cash (need %.2f, have %.2f).",
                ticker, total_cost, self._cash,
            )
            return False

        self._cash -= total_cost
        self._positions[ticker] = OpenPosition(
            ticker=ticker,
            entry_date=date,
            entry_price=fill_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            commission=commission,
        )
        logger.debug(
            "BUY %s: %.0f shares @ %.4f (commission=%.2f, cash=%.2f).",
            ticker, quantity, fill_price, commission, self._cash,
        )
        return True

    def sell(
        self,
        ticker: str,
        price: float,
        date: datetime,
        reason: str = "SIGNAL",
    ) -> Trade | None:
        """
        Close a long position.

        Returns the completed Trade, or None if no position exists.
        """
        if ticker not in self._positions:
            return None

        pos = self._positions.pop(ticker)
        fill_price = price * (1 - self._slippage_pct)
        commission = max(self._commission_min, pos.quantity * fill_price * self._commission_pct)
        proceeds   = pos.quantity * fill_price - commission
        self._cash += proceeds

        pnl     = proceeds - (pos.quantity * pos.entry_price + pos.commission)
        pnl_pct = pnl / (pos.quantity * pos.entry_price + pos.commission)

        holding = max(0, (date - pos.entry_date).days) if hasattr(date, "days") else 0
        # Handle both datetime and Timestamp objects
        try:
            holding = max(0, (date.date() - pos.entry_date.date()).days)
        except Exception:
            holding = 0

        trade = Trade(
            ticker=ticker,
            entry_date=pos.entry_date,
            exit_date=date,
            entry_price=pos.entry_price,
            exit_price=fill_price,
            quantity=pos.quantity,
            side="LONG",
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=pos.commission + commission,
            exit_reason=reason,
            holding_days=holding,
        )
        self._trades.append(trade)

        logger.debug(
            "SELL %s: %.0f shares @ %.4f → PnL=%.2f (%.2f%%) [%s].",
            ticker, pos.quantity, fill_price, pnl, pnl_pct * 100, reason,
        )
        return trade

    def check_stops(self, prices: dict[str, float], date: datetime) -> list[Trade]:
        """
        Check stop-loss and take-profit levels for all open positions.
        Closes positions that have hit their levels.

        Returns list of trades that were closed.
        """
        closed: list[Trade] = []
        for ticker in list(self._positions.keys()):
            pos   = self._positions[ticker]
            price = prices.get(ticker, 0.0)
            if price <= 0:
                continue

            if pos.stop_loss > 0 and price <= pos.stop_loss:
                trade = self.sell(ticker, price, date, reason="STOP_LOSS")
                if trade:
                    closed.append(trade)
                    logger.info("Stop-loss triggered: %s @ %.4f.", ticker, price)

            elif pos.take_profit > 0 and price >= pos.take_profit:
                trade = self.sell(ticker, price, date, reason="TAKE_PROFIT")
                if trade:
                    closed.append(trade)
                    logger.info("Take-profit triggered: %s @ %.4f.", ticker, price)

        return closed

    def record_equity(self, date: datetime, prices: dict[str, float]) -> float:
        """Record the current portfolio equity for the equity curve."""
        equity = self.portfolio_value(prices)
        self._equity_curve.append({"date": date, "equity": equity})
        return equity

    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self._cash = self._initial_cash
        self._positions.clear()
        self._trades.clear()
        self._equity_curve.clear()

    # ── Summary statistics ────────────────────────────────────────────────────

    @staticmethod
    def _max_consecutive(flags: list[bool]) -> int:
        """Return the length of the longest consecutive True run."""
        best = cur = 0
        for f in flags:
            cur = cur + 1 if f else 0
            best = max(best, cur)
        return best

    def summary(self) -> dict[str, Any]:
        """Return an enriched summary of backtest performance."""
        trades = self._trades
        if not trades:
            return {"n_trades": 0, "total_pnl": 0.0}

        pnls    = [t.pnl for t in trades]
        winners = [t for t in trades if t.pnl > 0]
        losers  = [t for t in trades if t.pnl <= 0]

        equity_curve = self.get_equity_curve()
        total_return = (equity_curve.iloc[-1] / self._initial_cash - 1) if not equity_curve.empty else 0.0

        # ── Streak stats ──────────────────────────────────────────────────────
        win_flags  = [t.pnl > 0 for t in trades]
        loss_flags = [t.pnl <= 0 for t in trades]
        max_consec_wins   = self._max_consecutive(win_flags)
        max_consec_losses = self._max_consecutive(loss_flags)

        # ── Best / worst trades ───────────────────────────────────────────────
        best_trade  = max(trades, key=lambda t: t.pnl)
        worst_trade = min(trades, key=lambda t: t.pnl)

        # ── Profit factor & expectancy ────────────────────────────────────────
        gross_wins   = sum(t.pnl for t in winners)
        gross_losses = abs(sum(t.pnl for t in losers))
        profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else float("inf")

        win_rate  = len(winners) / len(trades)
        avg_win   = gross_wins / len(winners) if winners else 0.0
        avg_loss  = (sum(t.pnl for t in losers) / len(losers)) if losers else 0.0
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss   # avg $ per trade

        # ── Holding period ────────────────────────────────────────────────────
        avg_holding = sum(t.holding_days for t in trades) / len(trades)
        max_holding = max(t.holding_days for t in trades)
        min_holding = min(t.holding_days for t in trades)

        # ── Exit reason breakdown ─────────────────────────────────────────────
        n_signal     = sum(1 for t in trades if t.exit_reason == "SIGNAL")
        n_stop_loss  = sum(1 for t in trades if t.exit_reason == "STOP_LOSS")
        n_take_profit= sum(1 for t in trades if t.exit_reason == "TAKE_PROFIT")
        n_eob        = sum(1 for t in trades if t.exit_reason == "END_OF_BACKTEST")

        # ── Win/loss ratio ────────────────────────────────────────────────────
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        return {
            # Basic counts
            "n_trades":            len(trades),
            "n_winners":           len(winners),
            "n_losers":            len(losers),
            "win_rate":            win_rate,
            # P&L
            "total_pnl":           sum(pnls),
            "avg_pnl":             sum(pnls) / len(pnls),
            "avg_win":             avg_win,
            "avg_loss":            avg_loss,
            "best_trade_pnl":      best_trade.pnl,
            "best_trade_pct":      best_trade.pnl_pct,
            "best_trade_ticker":   best_trade.ticker,
            "worst_trade_pnl":     worst_trade.pnl,
            "worst_trade_pct":     worst_trade.pnl_pct,
            "worst_trade_ticker":  worst_trade.ticker,
            # Risk / quality
            "profit_factor":       round(profit_factor, 4),
            "expectancy":          expectancy,
            "win_loss_ratio":      round(win_loss_ratio, 4),
            # Streaks
            "max_consec_wins":     max_consec_wins,
            "max_consec_losses":   max_consec_losses,
            # Holding
            "avg_holding_days":    round(avg_holding, 1),
            "max_holding_days":    max_holding,
            "min_holding_days":    min_holding,
            # Exit reasons
            "n_signal_exit":       n_signal,
            "n_stop_loss":         n_stop_loss,
            "n_take_profit":       n_take_profit,
            "n_end_of_backtest":   n_eob,
            # Overall
            "total_return":        total_return,
            "final_equity":        equity_curve.iloc[-1] if not equity_curve.empty else self._cash,
            "total_commission":    sum(t.commission for t in trades),
        }
