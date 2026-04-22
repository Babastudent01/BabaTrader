"""
features/candles.py
Candle pattern detection using pure pandas arithmetic.

All pattern functions return a pd.Series with:
  +1 = bullish pattern detected
  -1 = bearish pattern detected
   0 = no pattern / neutral

Patterns implemented
--------------------
Single-candle:
  Doji, Hammer, Shooting Star, Bullish/Bearish Marubozu,
  Bullish/Bearish Pin Bar, Spinning Top

Two-candle:
  Bullish Engulfing, Bearish Engulfing, Tweezer Top/Bottom,
  Bullish/Bearish Harami

Three-candle:
  Morning Star, Evening Star, Three White Soldiers, Three Black Crows

Multi-bar:
  Inside Bar, Outside Bar, Fair Value Gap (FVG)

Composite:
  candle_pattern_score — weighted sum of all above
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Helpers ───────────────────────────────────────────────────────────────────

def _body(df: pd.DataFrame) -> pd.Series:
    """Absolute candle body size."""
    return (df["close"] - df["open"]).abs()

def _range(df: pd.DataFrame) -> pd.Series:
    """Full candle range (high - low)."""
    return df["high"] - df["low"]

def _upper_wick(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df[["open", "close"]].max(axis=1)

def _lower_wick(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].min(axis=1) - df["low"]

def _is_bullish(df: pd.DataFrame) -> pd.Series:
    return df["close"] > df["open"]

def _is_bearish(df: pd.DataFrame) -> pd.Series:
    return df["close"] < df["open"]

# ── Single-candle patterns ────────────────────────────────────────────────────

def detect_doji(df: pd.DataFrame, threshold: float = 0.10) -> pd.Series:
    """Doji: body ≤ threshold × range. Indecision."""
    r = _range(df)
    body = _body(df)
    return pd.Series(
        np.where((r > 0) & (body / r.replace(0, np.nan) <= threshold), 1, 0),
        index=df.index, name="cdl_doji",
    )

def detect_hammer(df: pd.DataFrame, wick_ratio: float = 2.0) -> pd.Series:
    """
    Hammer / Hanging Man: small body, lower wick ≥ wick_ratio × body,
    upper wick ≤ body. Bullish when at bottom of downtrend (+1).
    """
    body  = _body(df)
    upper = _upper_wick(df)
    lower = _lower_wick(df)
    cond = (body > 0) & (lower >= wick_ratio * body) & (upper <= body)
    return pd.Series(
        np.where(cond, 1, 0),
        index=df.index, name="cdl_hammer",
    )

def detect_shooting_star(df: pd.DataFrame, wick_ratio: float = 2.0) -> pd.Series:
    """
    Shooting Star / Inverted Hammer: small body, upper wick ≥ wick_ratio × body,
    lower wick ≤ body. Bearish when at top of uptrend (-1).
    """
    body  = _body(df)
    upper = _upper_wick(df)
    lower = _lower_wick(df)
    cond = (body > 0) & (upper >= wick_ratio * body) & (lower <= body)
    return pd.Series(
        np.where(cond, -1, 0),
        index=df.index, name="cdl_shooting_star",
    )

def detect_pin_bar(df: pd.DataFrame, wick_ratio: float = 2.5) -> pd.Series:
    """
    Pin Bar (Rejection Candle):
      Bullish: long lower wick (rejection of lower prices)
      Bearish: long upper wick (rejection of higher prices)
    """
    r     = _range(df)
    upper = _upper_wick(df)
    lower = _lower_wick(df)
    # Bullish: lower wick > wick_ratio × upper wick and lower wick > 60% of range
    bull = (r > 0) & (lower >= wick_ratio * upper.replace(0, np.nan).fillna(1e-9)) & (lower > 0.6 * r)
    # Bearish: upper wick > wick_ratio × lower wick
    bear = (r > 0) & (upper >= wick_ratio * lower.replace(0, np.nan).fillna(1e-9)) & (upper > 0.6 * r)
    result = np.where(bull, 1, np.where(bear, -1, 0))
    return pd.Series(result, index=df.index, name="cdl_pin_bar")

def detect_marubozu(df: pd.DataFrame, wick_threshold: float = 0.05) -> pd.Series:
    """
    Marubozu: full-body candle with tiny or no wicks.
    Bullish Marubozu: +1 (strong buying pressure)
    Bearish Marubozu: -1 (strong selling pressure)
    """
    r     = _range(df).replace(0, np.nan)
    upper = _upper_wick(df)
    lower = _lower_wick(df)
    body  = _body(df)
    small_wicks = (upper / r.fillna(1) <= wick_threshold) & (lower / r.fillna(1) <= wick_threshold)
    bull = small_wicks & _is_bullish(df)
    bear = small_wicks & _is_bearish(df)
    result = np.where(bull, 1, np.where(bear, -1, 0))
    return pd.Series(result, index=df.index, name="cdl_marubozu")

def detect_spinning_top(df: pd.DataFrame) -> pd.Series:
    """Spinning Top: small body, both wicks larger than body. Indecision."""
    r     = _range(df).replace(0, np.nan)
    body  = _body(df)
    upper = _upper_wick(df)
    lower = _lower_wick(df)
    cond = (r.notna()) & (body / r.fillna(1) < 0.3) & (upper > body) & (lower > body)
    return pd.Series(
        np.where(cond, 1, 0),
        index=df.index, name="cdl_spinning_top",
    )

# ── Two-candle patterns ───────────────────────────────────────────────────────

def detect_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Bullish / Bearish Engulfing:
    Bullish: bearish bar followed by larger bullish bar that engulfs previous body.
    Bearish: bullish bar followed by larger bearish bar that engulfs previous body.
    """
    prev_open  = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    curr_open  = df["open"]
    curr_close = df["close"]

    prev_bullish = prev_close > prev_open
    prev_bearish = prev_close < prev_open

    # Bullish engulfing: previous bearish, current opens below prev close,
    # closes above prev open
    bull = (
        prev_bearish &
        (curr_open  <= prev_close) &
        (curr_close >= prev_open) &
        (curr_close > curr_open)
    )
    # Bearish engulfing: previous bullish, current opens above prev close,
    # closes below prev open
    bear = (
        prev_bullish &
        (curr_open  >= prev_close) &
        (curr_close <= prev_open) &
        (curr_close < curr_open)
    )
    result = np.where(bull, 1, np.where(bear, -1, 0))
    return pd.Series(result, index=df.index, name="cdl_engulfing")

def detect_harami(df: pd.DataFrame) -> pd.Series:
    """
    Bullish / Bearish Harami:
    The second candle's body is contained within the first candle's body.
    """
    prev_high_body = df[["open", "close"]].shift(1).max(axis=1)
    prev_low_body  = df[["open", "close"]].shift(1).min(axis=1)
    curr_high_body = df[["open", "close"]].max(axis=1)
    curr_low_body  = df[["open", "close"]].min(axis=1)

    inside = (curr_high_body < prev_high_body) & (curr_low_body > prev_low_body)

    prev_bearish = df["close"].shift(1) < df["open"].shift(1)
    prev_bullish = df["close"].shift(1) > df["open"].shift(1)

    bull = inside & prev_bearish & (df["close"] > df["open"])
    bear = inside & prev_bullish & (df["close"] < df["open"])
    result = np.where(bull, 1, np.where(bear, -1, 0))
    return pd.Series(result, index=df.index, name="cdl_harami")

def detect_tweezer(df: pd.DataFrame, tolerance: float = 0.002) -> pd.Series:
    """
    Tweezer Bottom / Top:
    Two candles with matching lows (bottom) or highs (top).
    """
    price = df["close"].mean()
    tol   = tolerance * price

    tweezer_bottom = (
        ((df["low"] - df["low"].shift(1)).abs() < tol) &
        _is_bearish(df).shift(1).fillna(False) &
        _is_bullish(df)
    )
    tweezer_top = (
        ((df["high"] - df["high"].shift(1)).abs() < tol) &
        _is_bullish(df).shift(1).fillna(False) &
        _is_bearish(df)
    )
    result = np.where(tweezer_bottom, 1, np.where(tweezer_top, -1, 0))
    return pd.Series(result, index=df.index, name="cdl_tweezer")

# ── Three-candle patterns ─────────────────────────────────────────────────────

def detect_morning_evening_star(df: pd.DataFrame) -> pd.Series:
    """
    Morning Star (+1): Large bearish, small body, large bullish.
    Evening Star (-1): Large bullish, small body, large bearish.
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    body1 = (c.shift(2) - o.shift(2)).abs()
    body2 = (c.shift(1) - o.shift(1)).abs()
    body3 = (c - o).abs()
    range1 = (h.shift(2) - l.shift(2)).replace(0, np.nan)
    range3 = (h - l).replace(0, np.nan)

    morning = (
        _is_bearish(df).shift(2).fillna(False) &          # bar1 bearish
        (body2 / range1.fillna(body2 + 1) < 0.3) &        # bar2 small
        _is_bullish(df) &                                  # bar3 bullish
        (body3 / range3.fillna(body3 + 1) > 0.5) &        # bar3 large
        (c > (o.shift(2) + c.shift(2)) / 2)               # bar3 closes > midpoint bar1
    )
    evening = (
        _is_bullish(df).shift(2).fillna(False) &
        (body2 / range1.fillna(body2 + 1) < 0.3) &
        _is_bearish(df) &
        (body3 / range3.fillna(body3 + 1) > 0.5) &
        (c < (o.shift(2) + c.shift(2)) / 2)
    )
    result = np.where(morning, 1, np.where(evening, -1, 0))
    return pd.Series(result, index=df.index, name="cdl_star")

def detect_three_soldiers_crows(df: pd.DataFrame) -> pd.Series:
    """
    Three White Soldiers (+1): 3 consecutive bullish bars, each closing higher.
    Three Black Crows (-1): 3 consecutive bearish bars, each closing lower.
    """
    c = df["close"]
    o = df["open"]

    bull = (
        _is_bullish(df) & _is_bullish(df).shift(1) & _is_bullish(df).shift(2) &
        (c > c.shift(1)) & (c.shift(1) > c.shift(2)) &
        (o > o.shift(1)) & (o.shift(1) > o.shift(2))
    )
    bear = (
        _is_bearish(df) & _is_bearish(df).shift(1) & _is_bearish(df).shift(2) &
        (c < c.shift(1)) & (c.shift(1) < c.shift(2)) &
        (o < o.shift(1)) & (o.shift(1) < o.shift(2))
    )
    result = np.where(bull, 1, np.where(bear, -1, 0))
    return pd.Series(result, index=df.index, name="cdl_3_soldiers_crows")

# ── Multi-bar patterns ────────────────────────────────────────────────────────

def detect_inside_bar(df: pd.DataFrame) -> pd.Series:
    """
    Inside Bar: current candle's high/low range is within previous candle.
    Bullish context (+1) if following a bullish candle, bearish (-1) otherwise.
    """
    inside = (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))
    prev_bull = _is_bullish(df).shift(1).fillna(False)
    result = np.where(inside & prev_bull, 1, np.where(inside & ~prev_bull, -1, 0))
    return pd.Series(result, index=df.index, name="cdl_inside_bar")

def detect_outside_bar(df: pd.DataFrame) -> pd.Series:
    """
    Outside Bar: current candle engulfs previous candle's full range.
    Bullish if closes in upper half (+1), bearish if closes in lower half (-1).
    """
    outside = (df["high"] > df["high"].shift(1)) & (df["low"] < df["low"].shift(1))
    mid = (df["high"] + df["low"]) / 2
    bull = outside & (df["close"] > mid)
    bear = outside & (df["close"] <= mid)
    result = np.where(bull, 1, np.where(bear, -1, 0))
    return pd.Series(result, index=df.index, name="cdl_outside_bar")

def detect_fair_value_gap(df: pd.DataFrame) -> pd.Series:
    """
    Fair Value Gap (FVG / Imbalance):
    Bullish FVG (+1): gap between bar-3 high and bar-1 low (price skipped up).
    Bearish FVG (-1): gap between bar-3 low and bar-1 high (price skipped down).
    """
    bull_fvg = df["low"] > df["high"].shift(2)   # current low > 2-bars-ago high
    bear_fvg = df["high"] < df["low"].shift(2)   # current high < 2-bars-ago low
    result = np.where(bull_fvg, 1, np.where(bear_fvg, -1, 0))
    return pd.Series(result, index=df.index, name="cdl_fvg")

# ── Master function ───────────────────────────────────────────────────────────

def add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all candle pattern columns to *df*.
    Also adds a composite ``candle_score`` column (weighted sum, normalised to [-1, +1]).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with columns: open, high, low, close.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new candle pattern columns appended.
    """
    df = df.copy()

    patterns: list[tuple[pd.Series, float]] = [
        (detect_doji(df),                    0.5),   # indecision — neutral weight
        (detect_hammer(df),                  1.5),   # strong reversal signal
        (detect_shooting_star(df),           1.5),
        (detect_pin_bar(df),                 2.0),   # strongest single-bar signal
        (detect_marubozu(df),                1.5),
        (detect_spinning_top(df),            0.5),
        (detect_engulfing(df),               2.0),   # strongest two-bar signal
        (detect_harami(df),                  1.0),
        (detect_tweezer(df),                 1.0),
        (detect_morning_evening_star(df),    2.0),
        (detect_three_soldiers_crows(df),    2.0),
        (detect_inside_bar(df),              1.0),
        (detect_outside_bar(df),             1.0),
        (detect_fair_value_gap(df),          1.5),
    ]

    total_weight = sum(w for _, w in patterns)
    composite = pd.Series(0.0, index=df.index)

    for series, weight in patterns:
        df[series.name] = series
        composite += series.astype(float) * weight

    # Normalise to [-1, +1]
    df["candle_score"] = (composite / total_weight).clip(-1, 1)

    return df
