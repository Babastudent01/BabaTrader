"""
tests/test_features.py
Unit tests for the feature engineering pipeline.
Tests the function-based technical indicators and return features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.technical import (
    add_rsi,
    add_macd,
    add_atr,
    add_bollinger_bands,
    add_sma,
    add_ema,
    add_volume_features,
    add_momentum,
    add_stochastic,
    add_candle_features,
)
from features.returns import add_returns, add_target


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with 200 bars."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, n))
    high  = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low   = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = close * (1 + np.random.normal(0, 0.003, n))
    vol   = np.random.randint(500_000, 5_000_000, n).astype(float)

    return pd.DataFrame({
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": vol,
    }, index=dates)


# ── Technical indicator tests ─────────────────────────────────────────────────

class TestTechnicalIndicators:

    def test_add_rsi_column_present(self, sample_ohlcv):
        result = add_rsi(sample_ohlcv, period=14)
        assert "rsi_14" in result.columns
        assert len(result) == len(sample_ohlcv)

    def test_rsi_bounded(self, sample_ohlcv):
        result = add_rsi(sample_ohlcv, period=14)
        rsi = result["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_add_macd_columns_present(self, sample_ohlcv):
        result = add_macd(sample_ohlcv, fast=12, slow=26, signal=9)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns

    def test_add_atr_positive(self, sample_ohlcv):
        result = add_atr(sample_ohlcv, period=14)
        assert "atr_14" in result.columns
        atr = result["atr_14"].dropna()
        assert (atr >= 0).all()

    def test_add_bollinger_bands_ordering(self, sample_ohlcv):
        result = add_bollinger_bands(sample_ohlcv, period=20)
        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns
        valid = result[["bb_upper", "bb_lower"]].dropna()
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_add_sma_columns_present(self, sample_ohlcv):
        result = add_sma(sample_ohlcv, windows=[10, 20, 50])
        assert "sma_10" in result.columns
        assert "sma_20" in result.columns
        assert "sma_50" in result.columns

    def test_sma_no_future_leakage(self, sample_ohlcv):
        """SMA at bar 4 (index 4) should equal mean of bars 0-4."""
        result = add_sma(sample_ohlcv, windows=[5])
        expected = sample_ohlcv["close"].iloc[:5].mean()
        actual   = result["sma_5"].iloc[4]
        assert abs(actual - expected) < 1e-6

    def test_add_ema_columns_present(self, sample_ohlcv):
        result = add_ema(sample_ohlcv, windows=[12, 26])
        assert "ema_12" in result.columns
        assert "ema_26" in result.columns

    def test_add_volume_features(self, sample_ohlcv):
        result = add_volume_features(sample_ohlcv, windows=[10, 20])
        assert "vol_sma_10" in result.columns
        assert "rel_vol_10" in result.columns
        assert "obv" in result.columns

    def test_add_momentum(self, sample_ohlcv):
        result = add_momentum(sample_ohlcv, periods=[1, 5, 10])
        assert "mom_1" in result.columns
        assert "mom_5" in result.columns
        assert "mom_10" in result.columns

    def test_add_stochastic(self, sample_ohlcv):
        result = add_stochastic(sample_ohlcv, k_period=14, d_period=3)
        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns
        stoch_k = result["stoch_k"].dropna()
        assert (stoch_k >= 0).all() and (stoch_k <= 100).all()

    def test_add_candle_features(self, sample_ohlcv):
        result = add_candle_features(sample_ohlcv)
        assert "candle_body" in result.columns
        assert "upper_shadow" in result.columns
        assert "lower_shadow" in result.columns

    def test_functions_do_not_modify_original(self, sample_ohlcv):
        """All add_* functions should return a copy, not modify in place."""
        original_cols = set(sample_ohlcv.columns)
        _ = add_rsi(sample_ohlcv, period=14)
        assert set(sample_ohlcv.columns) == original_cols


# ── Return feature tests ──────────────────────────────────────────────────────

class TestReturnFeatures:

    def test_add_returns_columns_present(self, sample_ohlcv):
        # add_returns produces ret_<p> and log_ret_<p> columns
        result = add_returns(sample_ohlcv, periods=[1, 5, 10])
        assert "ret_1" in result.columns
        assert "ret_5" in result.columns
        assert "ret_10" in result.columns
        assert "log_ret_1" in result.columns

    def test_add_target_binary(self, sample_ohlcv):
        result = add_target(sample_ohlcv, horizon=1)
        assert "target" in result.columns
        target = result["target"].dropna()
        assert set(target.unique()).issubset({0, 1})

    def test_target_is_forward_looking(self, sample_ohlcv):
        """Target at bar i should reflect whether close[i+1] > close[i]."""
        result = add_target(sample_ohlcv, horizon=1)
        # Check a few bars manually
        for i in range(5, 10):
            expected = 1 if sample_ohlcv["close"].iloc[i + 1] > sample_ohlcv["close"].iloc[i] else 0
            actual   = int(result["target"].iloc[i])
            assert actual == expected, f"Bar {i}: expected {expected}, got {actual}"
