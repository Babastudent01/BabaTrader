"""
backtesting/engine.py
Event-driven backtesting engine.
Replays historical OHLCV data bar-by-bar, generating signals and executing trades.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from backtesting.portfolio import BacktestPortfolio, Trade
from config import Settings
from features.pipeline import FeaturePipeline
from models.base import BaseModel
from models.evaluator import ModelEvaluator
from risk.sizing import PositionSizer
from strategy.filters import SafetyFilter
from strategy.signal import Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Full results from a backtest run."""
    ticker:          str
    start_date:      str
    end_date:        str
    n_bars:          int
    n_trades:        int
    total_return:    float
    sharpe_ratio:    float
    max_drawdown:    float
    win_rate:        float
    total_pnl:       float
    total_commission: float
    equity_curve:    pd.Series
    trades:          list[Trade]
    metrics:         dict[str, float]


class BacktestEngine:
    """
    Bar-by-bar backtesting engine.

    For each bar:
    1. Update position prices and check stop-loss / take-profit.
    2. Apply feature pipeline to data up to current bar (no lookahead).
    3. Generate model signal.
    4. Execute BUY/SELL if signal is actionable.
    5. Record equity.

    Supports single-ticker and multi-ticker backtests.
    """

    def __init__(
        self,
        model: BaseModel,
        settings: Settings,
        feature_pipeline: FeaturePipeline | None = None,
    ) -> None:
        self._model     = model
        self._settings  = settings
        self._pipeline  = feature_pipeline or FeaturePipeline(settings)
        self._sizer     = PositionSizer(settings)
        self._filter    = SafetyFilter(settings)
        self._evaluator = ModelEvaluator()

        bt_cfg = settings.backtesting
        exec_cfg = settings.execution
        self._initial_cash    = float(bt_cfg.get("initial_cash", 100_000.0))
        self._commission_pct  = float(exec_cfg.get("commission_pct", 0.001))
        self._commission_min  = float(exec_cfg.get("commission_min", 1.0))
        self._slippage_pct    = float(exec_cfg.get("slippage_pct", 0.0005))
        self._threshold       = float(settings.strategy.get("confidence_threshold", 0.60))
        self._warmup_bars     = int(bt_cfg.get("warmup_bars", 50))

    def run(
        self,
        ticker: str,
        df: pd.DataFrame,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> BacktestResult:
        """
        Run a single-ticker backtest.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        df : pd.DataFrame
            OHLCV DataFrame with DatetimeIndex.
        start_date : str | None
            Start date filter (YYYY-MM-DD). Uses all data if None.
        end_date : str | None
            End date filter (YYYY-MM-DD). Uses all data if None.

        Returns
        -------
        BacktestResult
        """
        if not self._model.is_fitted:
            raise RuntimeError("Model must be fitted before running a backtest.")

        # Apply date filters
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        if len(df) < self._warmup_bars + 10:
            raise ValueError(
                f"Insufficient data for backtest: {len(df)} bars "
                f"(need at least {self._warmup_bars + 10})."
            )

        logger.info(
            "Starting backtest: %s, %d bars (%s → %s).",
            ticker, len(df),
            df.index[0].strftime("%Y-%m-%d"),
            df.index[-1].strftime("%Y-%m-%d"),
        )

        portfolio = BacktestPortfolio(
            initial_cash=self._initial_cash,
            commission_pct=self._commission_pct,
            commission_min=self._commission_min,
            slippage_pct=self._slippage_pct,
        )

        # Pre-compute features for the entire dataset (no lookahead — we slice per bar)
        df_feat = self._pipeline.transform(df, include_target=False)
        feature_cols = self._pipeline.feature_names

        # DeepModel needs seq_len rows of context to build a sequence.
        # For ML models this is 1; for DeepModel it is seq_len (default 20).
        n_ctx = max(getattr(self._model, "_seq_len", 1), 1)

        predictions: list[int] = []
        probabilities: list[float] = []
        actuals: list[int] = []

        for i in range(self._warmup_bars, len(df)):
            bar_date  = df.index[i]
            bar_close = float(df["close"].iloc[i])
            prices    = {ticker: bar_close}

            # Check stop-loss / take-profit
            portfolio.check_stops(prices, bar_date)

            # Skip if features not available for current bar
            if i >= len(df_feat) or df_feat.iloc[i].isna().any():
                portfolio.record_equity(bar_date, prices)
                continue

            # Get features for current bar with enough context for sequence models.
            # start_i is clamped to warmup_bars so we never include pre-warmup rows.
            start_i = max(self._warmup_bars, i - n_ctx + 1)
            rows = df_feat[feature_cols].iloc[start_i : i + 1]

            # Model prediction — take the LAST row's prediction
            try:
                proba = self._model.predict_proba(rows)
                p_up  = float(proba[-1, 1])   # prediction for bar i
            except Exception as exc:  # noqa: BLE001
                logger.debug("Prediction error at bar %d: %s", i, exc)
                portfolio.record_equity(bar_date, prices)
                continue

            # Determine signal
            if p_up >= self._threshold:
                signal_type = SignalType.BUY
            elif (1 - p_up) >= self._threshold:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD

            # Compute ATR for sizing
            atr_col = f"atr_{self._settings.features.get('atr_period', 14)}"
            atr = float(df_feat[atr_col].iloc[i]) if atr_col in df_feat.columns else 0.0

            signal = Signal(
                ticker=ticker,
                signal_type=signal_type,
                confidence=p_up if signal_type == SignalType.BUY else (1 - p_up),
                timestamp=bar_date,
                price=bar_close,
                atr=atr,
            )

            # Execute signal
            positions = portfolio.get_positions()

            if signal_type == SignalType.BUY and ticker not in positions:
                equity   = portfolio.portfolio_value(prices)
                quantity = self._sizer.compute_quantity(signal, equity, bar_close)
                stop_loss   = self._sizer.compute_stop_loss(bar_close, atr)
                take_profit = self._sizer.compute_take_profit(bar_close)
                portfolio.buy(ticker, quantity, bar_close, bar_date, stop_loss, take_profit)

            elif signal_type == SignalType.SELL and ticker in positions:
                portfolio.sell(ticker, bar_close, bar_date, reason="SIGNAL")

            # Record for metrics
            # Actual: did price go up next bar?
            if i + 1 < len(df):
                next_close = float(df["close"].iloc[i + 1])
                actual = 1 if next_close > bar_close else 0
                actuals.append(actual)
                predictions.append(1 if signal_type == SignalType.BUY else 0)
                probabilities.append(p_up)

            portfolio.record_equity(bar_date, prices)

        # Close any remaining open positions at last price
        last_price = float(df["close"].iloc[-1])
        last_date  = df.index[-1]
        for t in list(portfolio.get_positions().keys()):
            portfolio.sell(t, last_price, last_date, reason="END_OF_BACKTEST")

        # Compute metrics
        equity_curve = portfolio.get_equity_curve()
        summary      = portfolio.summary()
        trades       = portfolio.get_trades()

        # Classification metrics (if we have predictions)
        clf_metrics: dict[str, float] = {}
        if actuals and predictions:
            y_true  = pd.Series(actuals)
            y_pred  = pd.Series(predictions)
            y_proba = pd.Series(probabilities)
            clf_metrics = self._evaluator.compute_metrics(y_true, y_pred, y_proba)

        # Trading metrics from equity curve
        if not equity_curve.empty:
            returns = equity_curve.pct_change().dropna()
            sharpe  = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
            rolling_max = equity_curve.cummax()
            drawdown    = (equity_curve - rolling_max) / rolling_max
            max_dd      = float(drawdown.min())
        else:
            sharpe = 0.0
            max_dd = 0.0

        total_return = summary.get("total_return", 0.0)

        logger.info(
            "Backtest complete: %s | Return=%.2f%% | Sharpe=%.2f | "
            "MaxDD=%.2f%% | Trades=%d | WinRate=%.1f%%",
            ticker,
            total_return * 100,
            sharpe,
            max_dd * 100,
            summary.get("n_trades", 0),
            summary.get("win_rate", 0) * 100,
        )

        return BacktestResult(
            ticker=ticker,
            start_date=df.index[0].strftime("%Y-%m-%d"),
            end_date=df.index[-1].strftime("%Y-%m-%d"),
            n_bars=len(df),
            n_trades=summary.get("n_trades", 0),
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=summary.get("win_rate", 0.0),
            total_pnl=summary.get("total_pnl", 0.0),
            total_commission=summary.get("total_commission", 0.0),
            equity_curve=equity_curve,
            trades=trades,
            metrics={**summary, **clf_metrics},
        )

    def run_universe(
        self,
        data: dict[str, pd.DataFrame],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, BacktestResult]:
        """
        Run backtests for a universe of tickers.

        Returns a dict of ticker → BacktestResult.
        """
        results: dict[str, BacktestResult] = {}
        for ticker, df in data.items():
            try:
                result = self.run(ticker, df, start_date, end_date)
                results[ticker] = result
            except Exception as exc:  # noqa: BLE001
                logger.error("Backtest failed for %s: %s", ticker, exc)
        return results
