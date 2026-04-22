"""
reporting/reporter.py
Performance reporter.
Generates text summaries and CSV exports from backtest and live trading results.
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backtesting.engine import BacktestResult
from backtesting.portfolio import Trade

logger = logging.getLogger(__name__)


class PerformanceReporter:
    """
    Generates performance reports from backtest results.

    Outputs:
    - Console summary (via logger)
    - CSV trade log
    - CSV equity curve
    - JSON metrics summary
    - Monthly returns table (console)
    """

    def __init__(self, output_dir: str | Path = "reports") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def report(self, result: BacktestResult, save: bool = True) -> dict[str, Any]:
        """
        Generate a full performance report for a single backtest result.

        Parameters
        ----------
        result : BacktestResult
            Backtest result to report on.
        save : bool
            If True, save CSV and JSON files to output_dir.

        Returns
        -------
        dict[str, Any]
            Summary metrics dictionary.
        """
        summary = self._build_summary(result)
        self._print_summary(summary, result.ticker)

        if save:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = self._output_dir / f"{result.ticker}_{ts}"
            self._save_metrics_json(summary, Path(f"{prefix}_metrics.json"))
            self._save_equity_csv(result.equity_curve, Path(f"{prefix}_equity.csv"))
            self._save_trades_csv(result.trades, Path(f"{prefix}_trades.csv"))

        return summary

    def report_universe(
        self,
        results: dict[str, BacktestResult],
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Generate a summary table for a universe of backtest results.

        Returns a DataFrame with one row per ticker.
        """
        rows = []
        for ticker, result in results.items():
            summary = self._build_summary(result)
            summary["ticker"] = ticker
            rows.append(summary)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("ticker")

        # Print universe summary
        logger.info("\n%s\nUniverse Backtest Summary\n%s", "=" * 60, "=" * 60)
        logger.info("\n%s", df[[
            "total_return_pct", "sharpe_ratio", "max_drawdown_pct",
            "win_rate_pct", "n_trades",
        ]].to_string())

        if save:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self._output_dir / f"universe_summary_{ts}.csv"
            df.to_csv(path)
            logger.info("Universe summary saved to %s.", path)

        return df

    def print_monthly_returns(self, equity_curve: pd.Series) -> None:
        """Print a monthly returns table to the logger."""
        if equity_curve.empty:
            logger.warning("Empty equity curve — cannot compute monthly returns.")
            return

        monthly = equity_curve.resample("ME").last().pct_change().dropna()
        df = monthly.to_frame("return")
        df["year"]  = df.index.year
        df["month"] = df.index.month_name().str[:3]

        pivot = df.pivot_table(index="year", columns="month", values="return", aggfunc="first")
        # Reorder months
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])
        pivot_pct = (pivot * 100).round(2)

        logger.info("\nMonthly Returns (%%):\n%s", pivot_pct.to_string())

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_summary(result: BacktestResult) -> dict[str, Any]:
        """Build a flat summary dict from a BacktestResult."""
        m = result.metrics
        return {
            # Identity
            "ticker":               result.ticker,
            "start_date":           result.start_date,
            "end_date":             result.end_date,
            "n_bars":               result.n_bars,
            # Trade counts
            "n_trades":             result.n_trades,
            "n_winners":            m.get("n_winners", 0),
            "n_losers":             m.get("n_losers", 0),
            # P&L
            "total_return_pct":     round(result.total_return * 100, 2),
            "total_pnl":            round(result.total_pnl, 2),
            "avg_pnl":              round(m.get("avg_pnl", 0), 2),
            "avg_win":              round(m.get("avg_win", 0), 2),
            "avg_loss":             round(m.get("avg_loss", 0), 2),
            "best_trade_pnl":       round(m.get("best_trade_pnl", 0), 2),
            "best_trade_pct":       round(m.get("best_trade_pct", 0) * 100, 2),
            "best_trade_ticker":    m.get("best_trade_ticker", ""),
            "worst_trade_pnl":      round(m.get("worst_trade_pnl", 0), 2),
            "worst_trade_pct":      round(m.get("worst_trade_pct", 0) * 100, 2),
            "worst_trade_ticker":   m.get("worst_trade_ticker", ""),
            # Risk metrics
            "sharpe_ratio":         round(result.sharpe_ratio, 4),
            "max_drawdown_pct":     round(result.max_drawdown * 100, 2),
            "profit_factor":        round(m.get("profit_factor", 0), 4),
            "expectancy":           round(m.get("expectancy", 0), 2),
            "win_rate_pct":         round(result.win_rate * 100, 2),
            "win_loss_ratio":       round(m.get("win_loss_ratio", 0), 4),
            # Streaks
            "max_consec_wins":      m.get("max_consec_wins", 0),
            "max_consec_losses":    m.get("max_consec_losses", 0),
            # Holding
            "avg_holding_days":     m.get("avg_holding_days", 0),
            "max_holding_days":     m.get("max_holding_days", 0),
            "min_holding_days":     m.get("min_holding_days", 0),
            # Exit reasons
            "n_signal_exit":        m.get("n_signal_exit", 0),
            "n_stop_loss":          m.get("n_stop_loss", 0),
            "n_take_profit":        m.get("n_take_profit", 0),
            "n_end_of_backtest":    m.get("n_end_of_backtest", 0),
            # Costs
            "total_commission":     round(result.total_commission, 2),
            # Model quality
            "accuracy":             round(m.get("accuracy", 0) * 100, 2),
            "f1":                   round(m.get("f1", 0), 4),
            "roc_auc":              round(m.get("roc_auc", 0), 4),
        }

    @staticmethod
    def _print_summary(summary: dict[str, Any], ticker: str) -> None:
        """Print a formatted enriched summary to the logger."""
        sep = "─" * 62
        logger.info(sep)
        logger.info("  📊  Backtest Report — %s", ticker)
        logger.info(sep)

        # ── Period & bars ──────────────────────────────────────────────────
        logger.info("  Period           : %s  →  %s", summary["start_date"], summary["end_date"])
        logger.info("  Bars             : %d", summary["n_bars"])

        # ── Overall performance ────────────────────────────────────────────
        logger.info("")
        logger.info("  ── Performance ──────────────────────────────────────")
        logger.info("  Total Return     : %+.2f%%", summary["total_return_pct"])
        logger.info("  Total PnL        : %+.2f €", summary["total_pnl"])
        logger.info("  Sharpe Ratio     : %.4f", summary["sharpe_ratio"])
        logger.info("  Max Drawdown     : %.2f%%", summary["max_drawdown_pct"])
        logger.info("  Commission paid  : %.2f €", summary["total_commission"])

        # ── Trade statistics ───────────────────────────────────────────────
        logger.info("")
        logger.info("  ── Trades ───────────────────────────────────────────")
        logger.info("  Total trades     : %d   (W: %d  L: %d)",
                    summary["n_trades"], summary["n_winners"], summary["n_losers"])
        logger.info("  Win rate         : %.1f%%", summary["win_rate_pct"])
        logger.info("  Avg trade PnL    : %+.2f €", summary["avg_pnl"])
        logger.info("  Avg win          : %+.2f €", summary["avg_win"])
        logger.info("  Avg loss         : %+.2f €", summary["avg_loss"])
        logger.info("  Win/Loss ratio   : %.2fx", summary["win_loss_ratio"])
        logger.info("  Profit factor    : %.2f  (gross wins / gross losses)",
                    summary["profit_factor"])
        logger.info("  Expectancy       : %+.2f € per trade", summary["expectancy"])

        # ── Best / worst trades ────────────────────────────────────────────
        logger.info("")
        logger.info("  ── Best / Worst trades ──────────────────────────────")
        logger.info("  Best  trade      : %+.2f € (%+.1f%%)  [%s]",
                    summary["best_trade_pnl"], summary["best_trade_pct"],
                    summary["best_trade_ticker"])
        logger.info("  Worst trade      : %+.2f € (%+.1f%%)  [%s]",
                    summary["worst_trade_pnl"], summary["worst_trade_pct"],
                    summary["worst_trade_ticker"])

        # ── Streaks ────────────────────────────────────────────────────────
        logger.info("")
        logger.info("  ── Streaks ──────────────────────────────────────────")
        logger.info("  Max consec. wins : %d", summary["max_consec_wins"])
        logger.info("  Max consec. loss : %d", summary["max_consec_losses"])

        # ── Holding ────────────────────────────────────────────────────────
        logger.info("")
        logger.info("  ── Holding period ───────────────────────────────────")
        logger.info("  Avg holding      : %.1f days", summary["avg_holding_days"])
        logger.info("  Min holding      : %d days", summary["min_holding_days"])
        logger.info("  Max holding      : %d days", summary["max_holding_days"])

        # ── Exit reasons ───────────────────────────────────────────────────
        logger.info("")
        logger.info("  ── Exit reasons ─────────────────────────────────────")
        logger.info("  Signal exit      : %d", summary["n_signal_exit"])
        logger.info("  Stop-loss        : %d", summary["n_stop_loss"])
        logger.info("  Take-profit      : %d", summary["n_take_profit"])
        logger.info("  End of backtest  : %d", summary["n_end_of_backtest"])

        # ── Model quality ──────────────────────────────────────────────────
        logger.info("")
        logger.info("  ── Model quality ────────────────────────────────────")
        logger.info("  Accuracy         : %.2f%%", summary["accuracy"])
        logger.info("  F1 Score         : %.4f", summary["f1"])
        logger.info("  ROC-AUC          : %.4f", summary["roc_auc"])
        logger.info(sep)

    @staticmethod
    def _save_metrics_json(summary: dict[str, Any], path: Path) -> None:
        """Save metrics to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Metrics saved to %s.", path)

    @staticmethod
    def _save_equity_csv(equity_curve: pd.Series, path: Path) -> None:
        """Save equity curve to CSV."""
        if equity_curve.empty:
            return
        equity_curve.to_csv(path, header=["equity"])
        logger.info("Equity curve saved to %s.", path)

    @staticmethod
    def _save_trades_csv(trades: list[Trade], path: Path) -> None:
        """Save trade log to CSV (includes exit_reason and holding_days)."""
        if not trades:
            return
        fieldnames = [
            "ticker", "entry_date", "exit_date", "holding_days",
            "entry_price", "exit_price", "quantity", "side",
            "pnl", "pnl_pct", "commission", "exit_reason",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for trade in trades:
                writer.writerow({
                    "ticker":        trade.ticker,
                    "entry_date":    trade.entry_date,
                    "exit_date":     trade.exit_date,
                    "holding_days":  trade.holding_days,
                    "entry_price":   round(trade.entry_price, 4),
                    "exit_price":    round(trade.exit_price, 4),
                    "quantity":      trade.quantity,
                    "side":          trade.side,
                    "pnl":           round(trade.pnl, 2),
                    "pnl_pct":       round(trade.pnl_pct * 100, 2),
                    "commission":    round(trade.commission, 2),
                    "exit_reason":   trade.exit_reason,
                })
        logger.info("Trade log saved to %s.", path)
