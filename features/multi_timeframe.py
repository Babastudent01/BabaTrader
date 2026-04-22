"""
features/multi_timeframe.py
Multi-timeframe (MTF) feature engineering.

Fetches higher-timeframe OHLCV data (weekly, monthly) for a ticker,
computes key indicators on each timeframe, aligns them to the primary
(daily) index via forward-fill, and returns a combined DataFrame.

Higher timeframes are given stronger implicit weight by the model because:
  1. MTF features are explicitly labelled so the model associates them
     with their timeframe source.
  2. A composite ``mtf_trend_score`` encodes explicit weighting:
       Monthly × 4 + Weekly × 2 + Daily × 1  (sum / 7, normalised to [-1, +1])
  3. The ``mtf_alignment`` feature (0–1) measures how many TFs agree,
     acting as a confidence multiplier.

Supported timeframes
--------------------
  "1mo"  → monthly bars   (weight 4)
  "1wk"  → weekly bars    (weight 2)
  "1d"   → daily bars     (weight 1, the primary TF — already in the main df)

Available from Yahoo Finance (yfinance):
  - "1d"  up to 5–10 years
  - "1wk" up to 10+ years
  - "1mo" up to 10+ years
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Timeframe → (label, weight) — higher weight = more influence on mtf_trend_score
_TF_CONFIG: dict[str, tuple[str, int]] = {
    "1mo": ("1mo", 4),
    "1wk": ("1wk", 2),
}

_INDICATOR_WINDOWS = {
    "rsi":     14,
    "sma_fast": 10,
    "sma_slow": 20,
    "atr":      14,
}


def build_mtf_features(
    ticker: str,
    primary_index: pd.DatetimeIndex,
    timeframes: list[str] | None = None,
    fetch_period: str = "10y",
) -> pd.DataFrame:
    """
    Fetch higher-TF data, compute indicators, and align to *primary_index*.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "AAPL".
    primary_index : pd.DatetimeIndex
        The daily trading-day index that all features are aligned to.
    timeframes : list[str] | None
        List of Yahoo Finance interval strings to fetch.
        Default: ["1wk", "1mo"].
    fetch_period : str
        How far back to fetch data for each higher TF.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by *primary_index* with columns prefixed by
        timeframe label (e.g. ``1wk_rsi_14``, ``1mo_sma_pos``).
        Never contains NaN — gaps are forward-filled then backfilled.
    """
    if timeframes is None:
        timeframes = ["1wk", "1mo"]

    all_dfs: list[pd.DataFrame] = []

    for tf in timeframes:
        if tf not in _TF_CONFIG:
            logger.warning("MTF: unknown timeframe '%s' — skipping.", tf)
            continue
        label, _ = _TF_CONFIG[tf]
        try:
            tf_df = _fetch_tf(ticker, tf, fetch_period)
            if tf_df is None or len(tf_df) < 5:
                logger.warning("MTF: insufficient data for %s @ %s — skipping.", ticker, tf)
                continue
            feat_df = _compute_tf_features(tf_df, label, primary_index)
            all_dfs.append(feat_df)
            logger.debug("MTF %s %s: %d bars → %d features", ticker, tf, len(tf_df), len(feat_df.columns))
        except Exception as exc:
            logger.warning("MTF: failed to build features for %s @ %s: %s", ticker, tf, exc)

    if not all_dfs:
        return pd.DataFrame(index=primary_index)

    combined = pd.concat(all_dfs, axis=1)

    # Add composite MTF trend score (explicit higher-TF weighting)
    combined = _add_composite_score(combined, primary_index)

    return combined


# ── Internals ─────────────────────────────────────────────────────────────────

def _fetch_tf(ticker: str, interval: str, period: str) -> pd.DataFrame | None:
    """Download OHLCV data at the given interval using yfinance."""
    try:
        import yfinance as yf
        raw = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if raw is None or raw.empty:
            return None

        # Flatten MultiIndex columns that yfinance ≥ 0.2.x returns
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [c.lower() for c in raw.columns]

        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        return raw[["open", "high", "low", "close", "volume"]].dropna()
    except Exception as exc:
        logger.warning("MTF yfinance fetch failed for %s @ %s: %s", ticker, interval, exc)
        return None


def _compute_tf_features(
    df: pd.DataFrame,
    label: str,
    primary_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Compute indicators on *df*, then align to *primary_index* via forward-fill.
    All column names are prefixed with the timeframe label.
    """
    feat = pd.DataFrame(index=df.index)

    # ── RSI ───────────────────────────────────────────────────────────────────
    period = _INDICATOR_WINDOWS["rsi"]
    delta  = df["close"].diff()
    gain   = delta.clip(lower=0).rolling(period).mean()
    loss   = (-delta.clip(upper=0)).rolling(period).mean()
    rs     = gain / loss.replace(0, np.nan)
    feat[f"{label}_rsi_{period}"] = 100 - (100 / (1 + rs))

    # ── SMAs and position ─────────────────────────────────────────────────────
    fast = _INDICATOR_WINDOWS["sma_fast"]
    slow = _INDICATOR_WINDOWS["sma_slow"]
    feat[f"{label}_sma_{fast}"]   = df["close"].rolling(fast).mean()
    feat[f"{label}_sma_{slow}"]   = df["close"].rolling(slow).mean()
    # Price position relative to slow SMA: >1 = above, <1 = below
    feat[f"{label}_sma_pos"] = (
        df["close"] / feat[f"{label}_sma_{slow}"].replace(0, np.nan) - 1.0
    )

    # ── MACD direction ────────────────────────────────────────────────────────
    ema12   = df["close"].ewm(span=12, adjust=False).mean()
    ema26   = df["close"].ewm(span=26, adjust=False).mean()
    macd    = ema12 - ema26
    signal  = macd.ewm(span=9, adjust=False).mean()
    feat[f"{label}_macd_hist"]     = macd - signal      # histogram: + = bullish
    feat[f"{label}_macd_trend"]    = np.sign(macd - signal)  # +1/0/-1

    # ── ATR and volatility ────────────────────────────────────────────────────
    atr_p = _INDICATOR_WINDOWS["atr"]
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close  = (df["low"]  - df["close"].shift(1)).abs()
    tr  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    feat[f"{label}_atr_{atr_p}"]     = tr.rolling(atr_p).mean()
    feat[f"{label}_atr_pct"]         = feat[f"{label}_atr_{atr_p}"] / df["close"].replace(0, np.nan)

    # ── Trend direction encoding (used in composite score) ───────────────────
    # +1 = bullish (close > slow SMA, MACD bullish, RSI > 50)
    # -1 = bearish
    #  0 = neutral
    rsi_bull  = feat[f"{label}_rsi_{period}"] > 55
    rsi_bear  = feat[f"{label}_rsi_{period}"] < 45
    sma_bull  = df["close"] > feat[f"{label}_sma_{slow}"]
    sma_bear  = df["close"] < feat[f"{label}_sma_{slow}"]
    macd_bull = feat[f"{label}_macd_trend"] > 0
    macd_bear = feat[f"{label}_macd_trend"] < 0

    bull_score = (rsi_bull.astype(int) + sma_bull.astype(int) + macd_bull.astype(int))
    bear_score = (rsi_bear.astype(int) + sma_bear.astype(int) + macd_bear.astype(int))

    feat[f"{label}_direction"] = np.where(
        bull_score >= 2, 1, np.where(bear_score >= 2, -1, 0)
    )

    # ── Momentum ─────────────────────────────────────────────────────────────
    feat[f"{label}_mom_1"]  = df["close"].pct_change(1)
    feat[f"{label}_mom_3"]  = df["close"].pct_change(3)

    # ── Align to daily index via forward-fill ─────────────────────────────────
    aligned = feat.reindex(primary_index, method="ffill")
    # Backfill for the very start of the series (before first bar)
    aligned = aligned.bfill()
    # Any remaining NaN → 0
    aligned = aligned.fillna(0)

    return aligned


def _add_composite_score(
    combined: pd.DataFrame,
    primary_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Add two composite columns:

    ``mtf_trend_score``  : Weighted directional signal across all TFs.
                           Monthly × 4 + Weekly × 2 (normalised to [-1, +1]).

    ``mtf_alignment``    : Fraction of TFs with the same direction sign (0–1).
                           1.0 = all TFs aligned, 0.0 = all neutral/mixed.
    """
    directions: list[tuple[str, int]] = []
    for tf, (label, weight) in _TF_CONFIG.items():
        col = f"{label}_direction"
        if col in combined.columns:
            directions.append((col, weight))

    if not directions:
        combined["mtf_trend_score"] = 0.0
        combined["mtf_alignment"]   = 0.0
        return combined

    total_weight = sum(w for _, w in directions)
    score = sum(combined[col] * w for col, w in directions) / total_weight

    # Alignment: fraction of TFs that agree with the dominant direction
    dir_matrix   = pd.concat([combined[col] for col, _ in directions], axis=1)
    dominant_dir = score.apply(np.sign)
    agreement    = (dir_matrix.values == dominant_dir.values[:, None])
    combined["mtf_alignment"]   = agreement.mean(axis=1)
    combined["mtf_trend_score"] = score.clip(-1, 1)

    return combined
