"""
models/evaluator.py
Model evaluation metrics for classification and trading performance.
Computes accuracy, precision, recall, F1, ROC-AUC, Sharpe ratio,
max drawdown, win rate, and strategy returns.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Full evaluation result for a model on a test set."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    strategy_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int


class ModelEvaluator:
    """
    Computes classification and trading performance metrics.

    Classification metrics:
        accuracy, precision, recall, F1, ROC-AUC

    Trading metrics (based on signal-driven strategy):
        strategy_return — total return if we follow model signals
        sharpe_ratio    — annualised Sharpe ratio of strategy returns
        max_drawdown    — maximum peak-to-trough drawdown
        win_rate        — fraction of trades that were profitable
        n_trades        — number of trades taken
    """

    def compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_proba: pd.Series | None = None,
        price_returns: pd.Series | None = None,
    ) -> dict[str, float]:
        """
        Compute all evaluation metrics.

        Parameters
        ----------
        y_true : pd.Series
            True binary labels (1=UP, 0=DOWN).
        y_pred : pd.Series
            Predicted binary labels.
        y_proba : pd.Series | None
            Predicted probability of UP class (for ROC-AUC).
        price_returns : pd.Series | None
            Actual price returns for the test period (for trading metrics).
            If None, trading metrics will be 0.

        Returns
        -------
        dict[str, float]
            Dictionary of metric name → value.
        """
        metrics: dict[str, float] = {}

        # ── Classification metrics ────────────────────────────────────────────
        metrics["accuracy"]  = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"]    = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"]        = float(f1_score(y_true, y_pred, zero_division=0))

        if y_proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            except ValueError:
                metrics["roc_auc"] = 0.5  # Only one class present
        else:
            metrics["roc_auc"] = 0.0

        # ── Trading metrics ───────────────────────────────────────────────────
        if price_returns is not None and not price_returns.empty:
            trading = self._compute_trading_metrics(y_pred, price_returns)
            metrics.update(trading)
        else:
            metrics["strategy_return"] = 0.0
            metrics["sharpe_ratio"]    = 0.0
            metrics["max_drawdown"]    = 0.0
            metrics["win_rate"]        = 0.0
            metrics["n_trades"]        = 0

        return metrics

    def compute_full(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_proba: pd.Series | None = None,
        price_returns: pd.Series | None = None,
    ) -> EvaluationResult:
        """Return a typed EvaluationResult dataclass."""
        m = self.compute_metrics(y_true, y_pred, y_proba, price_returns)
        return EvaluationResult(
            accuracy=m["accuracy"],
            precision=m["precision"],
            recall=m["recall"],
            f1=m["f1"],
            roc_auc=m["roc_auc"],
            strategy_return=m["strategy_return"],
            sharpe_ratio=m["sharpe_ratio"],
            max_drawdown=m["max_drawdown"],
            win_rate=m["win_rate"],
            n_trades=int(m["n_trades"]),
        )

    # ── Trading metric helpers ────────────────────────────────────────────────

    @staticmethod
    def _compute_trading_metrics(
        signals: pd.Series,
        price_returns: pd.Series,
    ) -> dict[str, float]:
        """
        Compute trading performance metrics from signals and actual returns.

        Assumes:
        - signal=1 → long position (earn the return)
        - signal=0 → flat/short (earn negative return or 0)
        - Long-only strategy: signal=0 means flat (0 return)
        """
        # Align signals and returns
        aligned = pd.DataFrame({
            "signal": signals,
            "ret": price_returns,
        }).dropna()

        if aligned.empty:
            return {
                "strategy_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "n_trades": 0,
            }

        # Long-only: earn return when signal=1, else 0
        strategy_returns = aligned["signal"] * aligned["ret"]

        # Total return (compounded)
        total_return = float((1 + strategy_returns).prod() - 1)

        # Sharpe ratio (annualised, assuming daily bars)
        if strategy_returns.std() > 0:
            sharpe = float(
                strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
            )
        else:
            sharpe = 0.0

        # Max drawdown
        equity = (1 + strategy_returns).cumprod()
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_dd = float(drawdown.min())

        # Win rate: fraction of trades where return > 0
        trades = aligned[aligned["signal"] == 1]
        n_trades = len(trades)
        win_rate = float((trades["ret"] > 0).mean()) if n_trades > 0 else 0.0

        return {
            "strategy_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "n_trades": float(n_trades),
        }

    @staticmethod
    def compute_monthly_returns(equity_curve: pd.Series) -> pd.DataFrame:
        """
        Compute monthly returns from an equity curve.

        Parameters
        ----------
        equity_curve : pd.Series
            DatetimeIndex, values = portfolio value.

        Returns
        -------
        pd.DataFrame
            Pivot table: rows=year, columns=month, values=monthly return.
        """
        monthly = equity_curve.resample("ME").last().pct_change().dropna()
        df = monthly.to_frame("return")
        df["year"]  = df.index.year
        df["month"] = df.index.month
        pivot = df.pivot(index="year", columns="month", values="return")
        pivot.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ][:len(pivot.columns)]
        return pivot

    @staticmethod
    def print_summary(metrics: dict[str, float], model_name: str = "") -> None:
        """Print a formatted metrics summary to the logger."""
        header = f"{'─' * 50}"
        logger.info(header)
        logger.info("Model Evaluation Summary%s", f" — {model_name}" if model_name else "")
        logger.info(header)
        logger.info("  Accuracy:        %.4f", metrics.get("accuracy", 0))
        logger.info("  Precision:       %.4f", metrics.get("precision", 0))
        logger.info("  Recall:          %.4f", metrics.get("recall", 0))
        logger.info("  F1 Score:        %.4f", metrics.get("f1", 0))
        logger.info("  ROC-AUC:         %.4f", metrics.get("roc_auc", 0))
        logger.info("  Strategy Return: %.2f%%", metrics.get("strategy_return", 0) * 100)
        logger.info("  Sharpe Ratio:    %.4f", metrics.get("sharpe_ratio", 0))
        logger.info("  Max Drawdown:    %.2f%%", metrics.get("max_drawdown", 0) * 100)
        logger.info("  Win Rate:        %.2f%%", metrics.get("win_rate", 0) * 100)
        logger.info("  N Trades:        %d", int(metrics.get("n_trades", 0)))
        logger.info(header)
