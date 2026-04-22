"""
features/technical.py
Technical indicator feature engineering.
Uses pandas-ta for indicator calculations.
All functions take a DataFrame with open/high/low/close/volume columns
and return a DataFrame with new feature columns appended.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import pandas_ta; fall back to manual implementations if unavailable
try:
    import pandas_ta as ta
    _HAS_PANDAS_TA = True
except ImportError:
    _HAS_PANDAS_TA = False
    logger.warning(
        "pandas-ta not installed. Falling back to manual indicator implementations. "
        "Install with: pip install pandas-ta"
    )


# ── RSI ───────────────────────────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Relative Strength Index."""
    df = df.copy()
    if _HAS_PANDAS_TA:
        df[f"rsi_{period}"] = ta.rsi(df["close"], length=period)
    else:
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    return df


# ── MACD ──────────────────────────────────────────────────────────────────────

def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Add MACD line, signal line, and histogram."""
    df = df.copy()
    if _HAS_PANDAS_TA:
        macd_df = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
        if macd_df is not None and not macd_df.empty:
            df["macd"] = macd_df.iloc[:, 0]
            df["macd_signal"] = macd_df.iloc[:, 1]
            df["macd_hist"] = macd_df.iloc[:, 2]
    else:
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


# ── ATR ───────────────────────────────────────────────────────────────────────

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average True Range."""
    df = df.copy()
    if _HAS_PANDAS_TA:
        df[f"atr_{period}"] = ta.atr(df["high"], df["low"], df["close"], length=period)
    else:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f"atr_{period}"] = tr.ewm(com=period - 1, min_periods=period).mean()
    return df


# ── Bollinger Bands ───────────────────────────────────────────────────────────

def add_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std: float = 2.0,
) -> pd.DataFrame:
    """Add Bollinger Bands: upper, middle (SMA), lower, and %B."""
    df = df.copy()
    if _HAS_PANDAS_TA:
        bb = ta.bbands(df["close"], length=period, std=std)
        if bb is not None and not bb.empty:
            df["bb_lower"] = bb.iloc[:, 0]
            df["bb_mid"]   = bb.iloc[:, 1]
            df["bb_upper"] = bb.iloc[:, 2]
            df["bb_pct"]   = bb.iloc[:, 4]  # %B
    else:
        sma = df["close"].rolling(period).mean()
        std_dev = df["close"].rolling(period).std()
        df["bb_mid"]   = sma
        df["bb_upper"] = sma + std * std_dev
        df["bb_lower"] = sma - std * std_dev
        df["bb_pct"]   = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        ).replace(0, np.nan)
    return df


# ── Moving Averages ───────────────────────────────────────────────────────────

def add_sma(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Add Simple Moving Averages for multiple windows."""
    df = df.copy()
    for w in windows:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
        # Price relative to SMA (normalised)
        df[f"close_sma_{w}_ratio"] = df["close"] / df[f"sma_{w}"].replace(0, np.nan)
    return df


def add_ema(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Add Exponential Moving Averages for multiple windows."""
    df = df.copy()
    for w in windows:
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()
        df[f"close_ema_{w}_ratio"] = df["close"] / df[f"ema_{w}"].replace(0, np.nan)
    return df


# ── Volume Features ───────────────────────────────────────────────────────────

def add_volume_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Add volume-based features: rolling averages and relative volume."""
    df = df.copy()
    for w in windows:
        avg_vol = df["volume"].rolling(w).mean()
        df[f"vol_sma_{w}"] = avg_vol
        # Relative volume: today's volume vs. rolling average
        df[f"rel_vol_{w}"] = df["volume"] / avg_vol.replace(0, np.nan)

    # On-Balance Volume (OBV)
    direction = np.sign(df["close"].diff()).fillna(0)
    df["obv"] = (direction * df["volume"]).cumsum()

    # Volume-price trend
    df["vpt"] = (df["close"].pct_change() * df["volume"]).cumsum()

    return df


# ── Momentum ──────────────────────────────────────────────────────────────────

def add_momentum(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    """Add price momentum (rate of change) for multiple periods."""
    df = df.copy()
    for p in periods:
        df[f"mom_{p}"] = df["close"].pct_change(p)
    return df


# ── Stochastic Oscillator ─────────────────────────────────────────────────────

def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Add Stochastic %K and %D."""
    df = df.copy()
    if _HAS_PANDAS_TA:
        stoch = ta.stoch(df["high"], df["low"], df["close"], k=k_period, d=d_period)
        if stoch is not None and not stoch.empty:
            df["stoch_k"] = stoch.iloc[:, 0]
            df["stoch_d"] = stoch.iloc[:, 1]
    else:
        low_min  = df["low"].rolling(k_period).min()
        high_max = df["high"].rolling(k_period).max()
        df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
        df["stoch_d"] = df["stoch_k"].rolling(d_period).mean()
    return df


# ── Price patterns ────────────────────────────────────────────────────────────

def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add candlestick-derived features.

    Single-bar open/close features
    --------------------------------
    candle_body         : close − open  (signed)
    candle_body_pct     : (close − open) / open
    upper_shadow        : high − max(open, close)
    lower_shadow        : min(open, close) − low
    body_to_range       : |body| / (high − low)
    gap_open            : (open − prev_close) / prev_close  (overnight gap)
    intraday_position   : (close − low) / (high − low)  ∈ [0, 1]
    open_position       : (open  − low) / (high − low)  ∈ [0, 1]
    close_to_high       : (high  − close) / (high − low)  (how much sold off from high)

    Rolling open/close context (5-day window)
    ------------------------------------------
    gap_open_5d_mean    : average overnight gap over 5 days
    gap_open_5d_sum     : cumulative gap pressure over 5 days
    body_pct_5d_mean    : average intraday return over 5 days
    bull_days_5d        : fraction of last 5 days where close > open
    gap_up_days_5d      : fraction of last 5 days with positive overnight gap
    intraday_pos_5d     : rolling mean of intraday_position (where price closed in its range)
    """
    df = df.copy()
    body       = df["close"] - df["open"]
    full_range = (df["high"] - df["low"]).replace(0, np.nan)
    prev_close = df["close"].shift(1).replace(0, np.nan)

    # ── Single-bar features ───────────────────────────────────────────────
    df["candle_body"]       = body
    df["candle_body_pct"]   = body / df["open"].replace(0, np.nan)
    df["upper_shadow"]      = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_shadow"]      = df[["open", "close"]].min(axis=1) - df["low"]
    df["body_to_range"]     = body.abs() / full_range
    df["gap_open"]          = (df["open"] - prev_close) / prev_close
    df["intraday_position"] = (df["close"] - df["low"]) / full_range   # 1=closed at high
    df["open_position"]     = (df["open"]  - df["low"]) / full_range   # 1=opened at high
    df["close_to_high"]     = (df["high"]  - df["close"]) / full_range  # 0=closed at high

    # ── Rolling 5-day open/close context ─────────────────────────────────
    df["gap_open_5d_mean"]  = df["gap_open"].rolling(5, min_periods=1).mean()
    df["gap_open_5d_sum"]   = df["gap_open"].rolling(5, min_periods=1).sum()
    df["body_pct_5d_mean"]  = df["candle_body_pct"].rolling(5, min_periods=1).mean()
    df["bull_days_5d"]      = (df["close"] > df["open"]).rolling(5, min_periods=1).mean()
    df["gap_up_days_5d"]    = (df["gap_open"] > 0).rolling(5, min_periods=1).mean()
    df["intraday_pos_5d"]   = df["intraday_position"].rolling(5, min_periods=1).mean()

    return df
