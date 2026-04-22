"""
features/session.py
Trading session, calendar, and economic event features.

Covers:
  - Trading session labels (London, New York, Asian, Overlap)
  - Day-of-week and calendar position features
  - Market holiday flags (NYSE + LSE)
  - Approximate high-impact economic event proximity:
      NFP   — first Friday of each month
      FOMC  — scheduled ~8 times/year
      CPI   — second or third week of each month
      Earnings season flag — Jan/Apr/Jul/Oct

All functions operate on the DataFrame's DatetimeIndex.
The helper ``add_session_features(df)`` applies every feature in one call.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Session constants (UTC hours) ─────────────────────────────────────────────
_TOKYO_OPEN_UTC   = 0    # 00:00 UTC
_TOKYO_CLOSE_UTC  = 9    # 09:00 UTC
_LONDON_OPEN_UTC  = 7    # 07:00 UTC (08:00 London winter / BST shifts ±1h)
_LONDON_CLOSE_UTC = 16   # 16:00 UTC
_NY_OPEN_UTC      = 13   # 13:30 UTC (market open, approx 13)
_NY_CLOSE_UTC     = 21   # 21:00 UTC (extended)


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all session, calendar, and economic event features to *df*.

    Works on any DatetimeIndex frequency (daily, hourly, intraday).
    For daily bars the session flags indicate which session dominates
    that trading day by convention; the economic event flags are exact.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex (timezone-aware or naive).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with new feature columns appended.
    """
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)

    # Normalise to UTC-naive for uniform arithmetic
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)

    df = _add_session_flags(df, idx)
    df = _add_calendar_features(df, idx)
    df = _add_holiday_flag(df, idx)
    df = _add_economic_event_features(df, idx)

    return df


# ── Session flags ─────────────────────────────────────────────────────────────

def _add_session_flags(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Binary flags for each major trading session.

    For daily bars (no intraday time component), all flags are 1 because a
    full trading day spans all sessions. In that case the ``session_*``
    columns are uninformative — the ``session_preferred`` and
    ``session_avoid`` columns below carry the real signal.

    For sub-daily bars, the flags correctly reflect the session.
    """
    hour = idx.hour

    df["session_asian"]   = ((hour >= _TOKYO_OPEN_UTC) & (hour < _TOKYO_CLOSE_UTC)).astype(int)
    df["session_london"]  = ((hour >= _LONDON_OPEN_UTC) & (hour < _LONDON_CLOSE_UTC)).astype(int)
    df["session_ny"]      = ((hour >= _NY_OPEN_UTC) & (hour < _NY_CLOSE_UTC)).astype(int)
    df["session_overlap"] = (
        (hour >= _NY_OPEN_UTC) & (hour < _LONDON_CLOSE_UTC)
    ).astype(int)   # London–NY overlap: most liquid hours (13:00–16:00 UTC)

    # Preferred: London or London–NY overlap
    df["session_preferred"] = (
        (df["session_london"] == 1) | (df["session_overlap"] == 1)
    ).astype(int)

    # Avoid: pure Asian session (no London/NY)
    df["session_avoid"] = (
        (df["session_asian"] == 1) &
        (df["session_london"] == 0) &
        (df["session_ny"] == 0)
    ).astype(int)

    return df


# ── Calendar features ─────────────────────────────────────────────────────────

def _add_calendar_features(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Day-of-week, month, and calendar position features.

    day_of_week : 0=Monday … 4=Friday
    is_monday   : Start-of-week — gaps from weekend, often choppy
    is_friday   : End-of-week  — position squaring, lower volume
    month       : 1–12
    is_earnings_season: Jan/Apr/Jul/Oct (earnings heavy months, higher vol)
    week_of_month : 1–5 (which week within the month)
    is_month_start: first 3 trading days of month (institutional rebalancing)
    is_month_end  : last 3 trading days (institutional rebalancing)
    is_quarter_end: last month of quarter (Mar/Jun/Sep/Dec)
    """
    df["day_of_week"]        = idx.dayofweek.astype(int)       # 0=Mon, 4=Fri
    df["is_monday"]          = (idx.dayofweek == 0).astype(int)
    df["is_friday"]          = (idx.dayofweek == 4).astype(int)
    df["month"]              = idx.month.astype(int)
    df["is_earnings_season"] = idx.month.isin([1, 4, 7, 10]).astype(int)
    df["week_of_month"]      = (idx.day // 7 + 1).astype(int)
    df["is_month_start"]     = (idx.day <= 3).astype(int)
    df["is_month_end"]       = (idx.is_month_end).astype(int)
    df["is_quarter_end"]     = (idx.is_quarter_end).astype(int)

    return df


# ── Holiday flag ──────────────────────────────────────────────────────────────

def _add_holiday_flag(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    ``is_holiday`` = 1 on NYSE/LSE bank holidays (no real trading, avoid trading).
    Uses the ``holidays`` package if installed; falls back to a minimal
    hardcoded list of US federal holidays otherwise.
    """
    dates = idx.normalize().date  # array of date objects

    try:
        import holidays as hol
        us_holidays = hol.US(years=list(range(idx.year.min(), idx.year.max() + 2)))
        uk_holidays = hol.GB(years=list(range(idx.year.min(), idx.year.max() + 2)),
                              subdiv="England")
        holiday_set = set(us_holidays.keys()) | set(uk_holidays.keys())
    except ImportError:
        # Minimal fallback: only US federal public holidays (approximate)
        logger.debug("'holidays' package not installed — using minimal holiday list.")
        holiday_set = _approximate_us_holidays(
            int(idx.year.min()), int(idx.year.max()) + 1
        )

    df["is_holiday"] = pd.array([1 if d in holiday_set else 0 for d in dates], dtype=int)
    return df


def _approximate_us_holidays(start_year: int, end_year: int) -> set[date]:
    """Very simplified US public holiday dates (no NYSE-specific adjustments)."""
    holidays: set[date] = set()
    for y in range(start_year, end_year + 1):
        holidays.add(date(y, 1, 1))    # New Year's Day
        holidays.add(date(y, 7, 4))    # Independence Day
        holidays.add(date(y, 11, 11))  # Veterans Day (approx)
        holidays.add(date(y, 12, 25))  # Christmas
    return holidays


# ── Economic event proximity ──────────────────────────────────────────────────

def _add_economic_event_features(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Approximate high-impact economic event dates.

    Features added:
    - ``is_nfp_week``          : 1 if the row falls in the NFP week (first Fri of month ± 2 days)
    - ``days_to_nfp``          : calendar days to the nearest NFP Friday
    - ``is_fomc_week``         : 1 if in an approximate FOMC announcement week
    - ``is_cpi_week``          : 1 if in approximate CPI release week (second week of month)
    - ``high_impact_event_day``: 1 if NFP, FOMC, or CPI announcement is within ±1 day
    """
    dates = idx.normalize()

    # ── NFP: first Friday of each month ──────────────────────────────────────
    nfp_dates = _get_first_fridays(int(idx.year.min()), int(idx.year.max()) + 1)
    nfp_ts = pd.DatetimeIndex(nfp_dates)

    days_to_nfp = _days_to_nearest(dates, nfp_ts)
    df["days_to_nfp"]  = days_to_nfp
    df["is_nfp_week"]  = (days_to_nfp.abs() <= 2).astype(int)

    # ── FOMC: 8 scheduled meetings/year — approximate via months ─────────────
    # Typical FOMC meeting months (2024/2025 schedule):
    # Jan, Mar, May, Jun, Jul, Sep, Nov, Dec
    fomc_months = {1, 3, 5, 6, 7, 9, 11, 12}
    # FOMC announcements are usually on Wednesdays in the 3rd week (day 15-22)
    fomc_dates = _get_fomc_dates(int(idx.year.min()), int(idx.year.max()) + 1, fomc_months)
    fomc_ts = pd.DatetimeIndex(fomc_dates)

    days_to_fomc = _days_to_nearest(dates, fomc_ts)
    df["days_to_fomc"]  = days_to_fomc
    df["is_fomc_week"]  = (days_to_fomc.abs() <= 2).astype(int)

    # ── CPI: typically released 2nd or 3rd Wednesday of the month ─────────────
    cpi_dates = _get_cpi_dates(int(idx.year.min()), int(idx.year.max()) + 1)
    cpi_ts = pd.DatetimeIndex(cpi_dates)

    days_to_cpi = _days_to_nearest(dates, cpi_ts)
    df["days_to_cpi"]  = days_to_cpi
    df["is_cpi_week"]  = (days_to_cpi.abs() <= 2).astype(int)

    # ── High-impact composite flag ────────────────────────────────────────────
    df["high_impact_event_day"] = (
        (days_to_nfp.abs() <= 1) |
        (days_to_fomc.abs() <= 1) |
        (days_to_cpi.abs() <= 1)
    ).astype(int)

    # ── No-news days (no high-impact event within ±5 days) ────────────────────
    # Strategy rule: don't trade when there's absolutely no catalyst
    min_days_to_any = pd.concat(
        [days_to_nfp.abs(), days_to_fomc.abs(), days_to_cpi.abs()], axis=1
    ).min(axis=1)
    df["no_major_news_week"] = (min_days_to_any > 5).astype(int)

    return df


def _days_to_nearest(dates: pd.DatetimeIndex, events: pd.DatetimeIndex) -> pd.Series:
    """For each date in *dates*, compute days to the nearest date in *events*."""
    if len(events) == 0:
        return pd.Series(999, index=dates, dtype=int)

    events_np = events.values.astype("datetime64[D]")
    dates_np  = dates.values.astype("datetime64[D]")

    # For each date, find minimum absolute distance to any event
    result = np.array([
        int(np.min(np.abs((events_np - d).astype(int))))
        for d in dates_np
    ])
    # Preserve sign: positive = event in future, negative = event in past
    sign = np.array([
        int(np.sign((events_np[np.argmin(np.abs((events_np - d).astype(int)))] - d).astype(int)))
        for d in dates_np
    ])
    return pd.Series(result * sign, index=dates, dtype=int)


def _get_first_fridays(start_year: int, end_year: int) -> list[date]:
    """Return the first Friday of every month from start_year to end_year."""
    result = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            d = date(y, m, 1)
            # Advance to first Friday
            days_until_friday = (4 - d.weekday()) % 7
            first_friday = d + timedelta(days=days_until_friday)
            result.append(first_friday)
    return result


def _get_fomc_dates(start_year: int, end_year: int, fomc_months: set[int]) -> list[date]:
    """Approximate FOMC announcement dates: 3rd Wednesday of FOMC months."""
    result = []
    for y in range(start_year, end_year + 1):
        for m in fomc_months:
            d = date(y, m, 1)
            # Find first Wednesday
            days_until_wed = (2 - d.weekday()) % 7
            first_wed = d + timedelta(days=days_until_wed)
            # Third Wednesday = first + 14 days
            third_wed = first_wed + timedelta(days=14)
            result.append(third_wed)
    return result


def _get_cpi_dates(start_year: int, end_year: int) -> list[date]:
    """Approximate CPI release dates: 2nd Wednesday of every month."""
    result = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            d = date(y, m, 1)
            days_until_wed = (2 - d.weekday()) % 7
            first_wed = d + timedelta(days=days_until_wed)
            second_wed = first_wed + timedelta(days=7)
            result.append(second_wed)
    return result
