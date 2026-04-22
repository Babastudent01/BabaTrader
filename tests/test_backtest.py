"""
tests/test_backtest.py
Integration tests for the backtesting engine and portfolio tracker.
Uses synthetic data and a pre-fitted LogisticModel — no real data fetching.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from backtesting.portfolio import BacktestPortfolio
from backtesting.engine import BacktestEngine
from models.logistic import LogisticModel
from features.pipeline import FeaturePipeline


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """Generate 500 bars of synthetic OHLCV data."""
    np.random.seed(7)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.012, n))
    high  = close * (1 + np.abs(np.random.normal(0, 0.004, n)))
    low   = close * (1 - np.abs(np.random.normal(0, 0.004, n)))
    open_ = close * (1 + np.random.normal(0, 0.002, n))
    vol   = np.random.randint(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )


@pytest.fixture
def mock_settings():
    """Build a minimal Settings mock compatible with FeaturePipeline and BacktestEngine."""
    settings = MagicMock()

    # features config
    feat_cfg = {
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "atr_period": 14,
        "bb_period": 20,
        "bb_std": 2.0,
        "sma_windows": [10, 20],
        "ema_windows": [10, 20],
        "volume_windows": [5, 10],
        "return_periods": [1, 5],
        "volatility_windows": [5, 10],
        "target_horizon": 1,
        "target_threshold": 0.0,
    }
    settings.features.get.side_effect = lambda k, d=None: feat_cfg.get(k, d)

    # strategy config
    strat_cfg = {"confidence_threshold": 0.55}
    settings.strategy.get.side_effect = lambda k, d=None: strat_cfg.get(k, d)

    # backtesting config
    bt_cfg = {"initial_cash": 50_000.0, "warmup_bars": 50}
    settings.backtesting.get.side_effect = lambda k, d=None: bt_cfg.get(k, d)

    # execution config
    exec_cfg = {"commission_pct": 0.001, "commission_min": 1.0, "slippage_pct": 0.0005}
    settings.execution.get.side_effect = lambda k, d=None: exec_cfg.get(k, d)

    # risk config
    risk_cfg = {
        "sizing_method": "percent",
        "position_pct": 0.10,
        "max_risk_per_trade_pct": 0.02,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.15,
        "atr_risk_multiplier": 2.0,
    }
    settings.risk.get.side_effect = lambda k, d=None: risk_cfg.get(k, d)

    # strategy.filters config
    filter_cfg = {
        "min_avg_volume": 0,
        "max_atr_pct": 1.0,
        "min_price": 0.0,
        "allow_high_volatility": True,
    }
    settings.strategy.filters.get.side_effect = lambda k, d=None: filter_cfg.get(k, d)

    return settings


# ── BacktestPortfolio tests ───────────────────────────────────────────────────

class TestBacktestPortfolio:

    def test_initial_state(self):
        portfolio = BacktestPortfolio(initial_cash=50_000.0)
        assert portfolio.cash == 50_000.0
        assert portfolio.get_positions() == {}
        assert portfolio.get_trades() == []

    def test_buy_reduces_cash(self):
        portfolio = BacktestPortfolio(
            initial_cash=10_000.0, commission_pct=0.0, commission_min=0.0, slippage_pct=0.0
        )
        date = datetime(2023, 1, 1)
        filled = portfolio.buy("AAPL", quantity=10, price=100.0, date=date)
        assert filled is True
        assert portfolio.cash == pytest.approx(9_000.0, abs=1.0)
        assert "AAPL" in portfolio.get_positions()

    def test_sell_closes_position(self):
        portfolio = BacktestPortfolio(
            initial_cash=10_000.0, commission_pct=0.0, commission_min=0.0, slippage_pct=0.0
        )
        date = datetime(2023, 1, 1)
        portfolio.buy("AAPL", quantity=10, price=100.0, date=date)
        trade = portfolio.sell("AAPL", price=110.0, date=datetime(2023, 2, 1))
        assert trade is not None
        assert trade.pnl == pytest.approx(100.0, abs=1.0)  # 10 shares * $10 gain
        assert "AAPL" not in portfolio.get_positions()

    def test_insufficient_cash_rejected(self):
        portfolio = BacktestPortfolio(
            initial_cash=100.0, commission_pct=0.0, commission_min=0.0, slippage_pct=0.0
        )
        date = datetime(2023, 1, 1)
        filled = portfolio.buy("AAPL", quantity=10, price=100.0, date=date)
        assert filled is False

    def test_sell_nonexistent_position_returns_none(self):
        portfolio = BacktestPortfolio(initial_cash=10_000.0)
        trade = portfolio.sell("AAPL", price=100.0, date=datetime(2023, 1, 1))
        assert trade is None

    def test_stop_loss_triggered(self):
        portfolio = BacktestPortfolio(
            initial_cash=10_000.0, commission_pct=0.0, commission_min=0.0, slippage_pct=0.0
        )
        date = datetime(2023, 1, 1)
        portfolio.buy("AAPL", quantity=10, price=100.0, date=date, stop_loss=90.0)
        # Price drops below stop loss
        closed = portfolio.check_stops({"AAPL": 85.0}, datetime(2023, 1, 10))
        assert len(closed) == 1
        assert closed[0].ticker == "AAPL"
        assert "AAPL" not in portfolio.get_positions()

    def test_take_profit_triggered(self):
        portfolio = BacktestPortfolio(
            initial_cash=10_000.0, commission_pct=0.0, commission_min=0.0, slippage_pct=0.0
        )
        date = datetime(2023, 1, 1)
        portfolio.buy("AAPL", quantity=10, price=100.0, date=date, take_profit=120.0)
        closed = portfolio.check_stops({"AAPL": 125.0}, datetime(2023, 1, 10))
        assert len(closed) == 1
        assert closed[0].pnl > 0

    def test_equity_curve_recorded(self):
        portfolio = BacktestPortfolio(initial_cash=10_000.0)
        date = datetime(2023, 1, 1)
        portfolio.record_equity(date, {})
        curve = portfolio.get_equity_curve()
        assert len(curve) == 1
        assert curve.iloc[0] == pytest.approx(10_000.0)

    def test_reset(self):
        portfolio = BacktestPortfolio(
            initial_cash=10_000.0, commission_pct=0.0, commission_min=0.0, slippage_pct=0.0
        )
        portfolio.buy("AAPL", quantity=5, price=100.0, date=datetime(2023, 1, 1))
        portfolio.reset()
        assert portfolio.cash == 10_000.0
        assert portfolio.get_positions() == {}
        assert portfolio.get_trades() == []

    def test_portfolio_value_includes_positions(self):
        portfolio = BacktestPortfolio(
            initial_cash=10_000.0, commission_pct=0.0, commission_min=0.0, slippage_pct=0.0
        )
        portfolio.buy("AAPL", quantity=10, price=100.0, date=datetime(2023, 1, 1))
        # Value = cash (9000) + position (10 * 110) = 10100
        value = portfolio.portfolio_value({"AAPL": 110.0})
        assert value == pytest.approx(10_100.0, abs=1.0)


# ── BacktestEngine integration test ──────────────────────────────────────────

class TestBacktestEngineIntegration:
    """
    Integration test: fit a LogisticModel on synthetic features,
    then run the backtest engine. Validates that the engine runs
    without errors and returns a valid BacktestResult.
    """

    def test_engine_runs_without_error(self, synthetic_ohlcv, mock_settings):
        """Smoke test: engine should complete without raising exceptions."""
        # Build feature pipeline and fit model
        pipeline = FeaturePipeline(mock_settings)
        df_feat = pipeline.transform(synthetic_ohlcv, include_target=True)

        feature_cols = pipeline.feature_names
        X = df_feat[feature_cols]
        y = df_feat["target"].astype(int)

        model = LogisticModel()
        model.fit(X, y)
        assert model.is_fitted

        # Run backtest
        engine = BacktestEngine(model, mock_settings, feature_pipeline=pipeline)
        result = engine.run("TEST", synthetic_ohlcv)

        # Validate result structure
        assert result.ticker == "TEST"
        assert result.n_bars == len(synthetic_ohlcv)
        assert isinstance(result.total_return, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert result.max_drawdown <= 0  # Drawdown is always <= 0
        assert not result.equity_curve.empty

    def test_engine_raises_if_model_not_fitted(self, synthetic_ohlcv, mock_settings):
        """Engine should raise RuntimeError if model is not fitted."""
        pipeline = FeaturePipeline(mock_settings)
        model = LogisticModel()  # not fitted
        engine = BacktestEngine(model, mock_settings, feature_pipeline=pipeline)
        with pytest.raises(RuntimeError):
            engine.run("TEST", synthetic_ohlcv)

    def test_engine_result_has_trades_list(self, synthetic_ohlcv, mock_settings):
        """BacktestResult.trades should be a list."""
        pipeline = FeaturePipeline(mock_settings)
        df_feat = pipeline.transform(synthetic_ohlcv, include_target=True)
        X = df_feat[pipeline.feature_names]
        y = df_feat["target"].astype(int)

        model = LogisticModel()
        model.fit(X, y)

        engine = BacktestEngine(model, mock_settings, feature_pipeline=pipeline)
        result = engine.run("TEST", synthetic_ohlcv)

        assert isinstance(result.trades, list)
        assert isinstance(result.n_trades, int)
        assert result.n_trades == len(result.trades)
