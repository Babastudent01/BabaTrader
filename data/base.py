"""
data/base.py
Abstract base class for all data providers.
All concrete providers must implement this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import Optional
import re

import pandas as pd


class DataProvider(ABC):
    """
    Abstract base for OHLCV data providers.

    All providers must return DataFrames with a DatetimeIndex and columns:
        open, high, low, close, volume  (lowercase)
    """

    @abstractmethod
    def fetch_ohlcv(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for a single ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol (e.g. 'AAPL').
        start : date
            Start date (inclusive).
        end : date
            End date (inclusive).
        interval : str
            Bar interval: '1d', '1h', '15m', etc.

        Returns
        -------
        pd.DataFrame
            DatetimeIndex, columns: open, high, low, close, volume.
            Returns empty DataFrame if data unavailable.
        """

    def fetch(
        self,
        ticker: str,
        period: str = "5y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Convenience wrapper: fetch OHLCV using a period string instead of start/end dates.

        Parameters
        ----------
        ticker : str
            Ticker symbol (e.g. 'AAPL').
        period : str
            How far back to fetch, e.g. '5y', '1y', '6mo', '3mo', '1mo', '5d'.
        interval : str
            Bar interval: '1d', '1h', '15m', etc.

        Returns
        -------
        pd.DataFrame
            DatetimeIndex, columns: open, high, low, close, volume.
        """
        end_dt   = date.today()
        start_dt = self._period_to_start(period, end_dt)
        return self.fetch_ohlcv(ticker, start_dt, end_dt, interval)

    @staticmethod
    def _period_to_start(period: str, end: date) -> date:
        """Convert a period string to a start date."""
        period = period.strip().lower()
        match = re.fullmatch(r"(\d+)(y|mo|m|d|wk|w)", period)
        if not match:
            raise ValueError(
                f"Unrecognised period '{period}'. "
                "Use e.g. '5y', '6mo', '3m', '90d', '1wk'."
            )
        n, unit = int(match.group(1)), match.group(2)
        if unit == "y":
            return date(end.year - n, end.month, end.day)
        if unit in ("mo", "m"):
            # Approximate: 30 days per month
            return end - timedelta(days=30 * n)
        if unit in ("wk", "w"):
            return end - timedelta(weeks=n)
        # days
        return end - timedelta(days=n)

    def fetch_multiple(
        self,
        tickers: list[str],
        start: date,
        end: date,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV for multiple tickers.
        Default implementation calls fetch_ohlcv in a loop.
        Override for batch efficiency.
        """
        result: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                df = self.fetch_ohlcv(ticker, start, end, interval)
                if not df.empty:
                    result[ticker] = df
            except Exception as exc:  # noqa: BLE001
                # Log but continue — one bad ticker should not abort the run
                import logging
                logging.getLogger(__name__).warning(
                    "Failed to fetch %s: %s", ticker, exc
                )
        return result

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase column names and drop any 'adj close' duplicates."""
        df = df.copy()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        # Some providers return 'adj_close' — prefer it over 'close' if present
        if "adj_close" in df.columns:
            df["close"] = df["adj_close"]
            df.drop(columns=["adj_close"], inplace=True)
        # Keep only standard OHLCV columns
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[keep]
