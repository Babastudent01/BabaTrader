"""
features/returns.py
Return and volatility feature engineering.
Computes log returns, rolling volatility, drawdown features, and the ML target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_returns(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    """
    Add simple and log returns for multiple look-back periods.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'close' column.
    periods : list[int]
        Look-back periods in bars (e.g. [1, 3, 5, 10, 20]).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new return columns appended.
    """
    df = df.copy()
    for p in periods:
        df[f"ret_{p}"]     = df["close"].pct_change(p)
        df[f"log_ret_{p}"] = np.log(df["close"] / df["close"].shift(p))
    return df


def add_rolling_volatility(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Add rolling annualised volatility (std of log returns).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'close' column.
    windows : list[int]
        Rolling window sizes in bars.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with volatility columns appended.
    """
    df = df.copy()
    log_ret = np.log(df["close"] / df["close"].shift(1))
    for w in windows:
        # Annualise assuming 252 trading days
        df[f"vol_{w}"] = log_ret.rolling(w).std() * np.sqrt(252)
    return df


def add_atr_pct(df: pd.DataFrame, atr_col: str = "atr_14") -> pd.DataFrame:
    """
    Add ATR as a percentage of close price (normalised volatility measure).
    Requires ATR column to already exist (from add_atr).
    """
    df = df.copy()
    if atr_col in df.columns:
        df["atr_pct"] = df[atr_col] / df["close"].replace(0, np.nan)
    return df


def add_rolling_drawdown(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Add rolling maximum drawdown over a look-back window.
    Useful as a regime/risk feature.
    """
    df = df.copy()
    rolling_max = df["close"].rolling(window).max()
    df[f"drawdown_{window}"] = (df["close"] - rolling_max) / rolling_max.replace(0, np.nan)
    return df


def add_high_low_range(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Add rolling high-low range as a fraction of close (normalised range).
    """
    df = df.copy()
    for w in windows:
        rolling_high = df["high"].rolling(w).max()
        rolling_low  = df["low"].rolling(w).min()
        df[f"hl_range_{w}"] = (rolling_high - rolling_low) / df["close"].replace(0, np.nan)
        # Position of current close within the rolling range
        df[f"close_in_range_{w}"] = (df["close"] - rolling_low) / (
            (rolling_high - rolling_low).replace(0, np.nan)
        )
    return df


def add_target(
    df: pd.DataFrame,
    horizon: int = 1,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Add the ML target variable: binary direction of future return.

    Target = 1 if forward return > threshold, else 0.

    IMPORTANT: The target is computed using FUTURE data.
    It must be shifted so that at time t, the target reflects
    what happens at t+horizon. This is safe for training but
    the target column must NEVER be used as a feature.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'close' column.
    horizon : int
        Number of bars ahead to predict.
    threshold : float
        Minimum return to classify as UP (default 0.0 = any positive return).

    Returns
    -------
    pd.DataFrame
        DataFrame with 'target' column (1=UP, 0=DOWN/FLAT) and
        'future_return' column (raw forward return).
    """
    df = df.copy()
    # Forward return: shift(-horizon) gives the price horizon bars in the future
    future_close = df["close"].shift(-horizon)
    df["future_return"] = (future_close - df["close"]) / df["close"].replace(0, np.nan)
    df["target"] = (df["future_return"] > threshold).astype(int)
    # The last `horizon` rows will have NaN target — drop them during training
    return df
