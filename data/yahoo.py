"""
data/yahoo.py
Yahoo Finance data provider using yfinance.
Free, no API key required. Suitable for daily and intraday data.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd
import yfinance as yf

from data.base import DataProvider
from data.cache import DataCache

logger = logging.getLogger(__name__)


class YahooFinanceProvider(DataProvider):
    """
    Fetches OHLCV data from Yahoo Finance via yfinance.

    Notes
    -----
    - Adjusted close prices are used by default (split/dividend adjusted).
    - Yahoo Finance data quality can vary; always validate before use.
    - Rate limits apply for high-frequency requests.
    """

    def __init__(self, cache: DataCache | None = None) -> None:
        self._cache = cache

    def fetch_ohlcv(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.

        Parameters
        ----------
        ticker : str
            Yahoo Finance ticker symbol (e.g. 'AAPL', 'MC.PA' for Euronext).
        start : date
            Start date (inclusive).
        end : date
            End date (inclusive).
        interval : str
            '1d', '1wk', '1mo', '1h', '30m', '15m', '5m', '1m'.
            Note: intraday data is limited to 60 days history.

        Returns
        -------
        pd.DataFrame
            DatetimeIndex, columns: open, high, low, close, volume.
        """
        # Check cache first
        if self._cache is not None:
            cached = self._cache.get(ticker, start, end, interval)
            if cached is not None:
                return cached

        logger.info("Fetching %s from Yahoo Finance (%s -> %s, %s).", ticker, start, end, interval)

        try:
            raw = yf.download(
                tickers=ticker,
                start=str(start),
                end=str(end),
                interval=interval,
                auto_adjust=True,       # Use adjusted prices
                progress=False,
            )
        except Exception as exc:
            logger.error("yfinance download failed for %s: %s", ticker, exc)
            return pd.DataFrame()

        if raw is None or raw.empty:
            logger.warning("No data returned for %s (%s -> %s).", ticker, start, end)
            return pd.DataFrame()

        # yfinance returns MultiIndex columns when downloading single ticker
        # with auto_adjust=True — flatten if needed
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = self._normalise_columns(raw)

        # Ensure DatetimeIndex is timezone-naive for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df.sort_index(inplace=True)
        df.dropna(how="all", inplace=True)

        # Store in cache
        if self._cache is not None:
            self._cache.set(ticker, start, end, interval, df)

        logger.info("Fetched %d bars for %s.", len(df), ticker)
        return df

    def fetch_multiple(
        self,
        tickers: list[str],
        start: date,
        end: date,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """
        Batch download multiple tickers in a single yfinance call.
        More efficient than calling fetch_ohlcv in a loop.
        """
        # Separate cached vs. to-fetch
        result: dict[str, pd.DataFrame] = {}
        to_fetch: list[str] = []

        for ticker in tickers:
            if self._cache is not None:
                cached = self._cache.get(ticker, start, end, interval)
                if cached is not None:
                    result[ticker] = cached
                    continue
            to_fetch.append(ticker)

        if not to_fetch:
            return result

        logger.info(
            "Batch fetching %d tickers from Yahoo Finance (%s -> %s).",
            len(to_fetch), start, end,
        )

        try:
            raw = yf.download(
                tickers=to_fetch,
                start=str(start),
                end=str(end),
                interval=interval,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
            )
        except Exception as exc:
            logger.error("Batch yfinance download failed: %s", exc)
            # Fall back to individual fetches
            return super().fetch_multiple(tickers, start, end, interval)

        if raw is None or raw.empty:
            return result

        for ticker in to_fetch:
            try:
                if len(to_fetch) == 1:
                    # Single ticker: columns are not grouped
                    df_raw = raw
                else:
                    df_raw = raw[ticker]

                df = self._normalise_columns(df_raw)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df.sort_index(inplace=True)
                df.dropna(how="all", inplace=True)

                if not df.empty:
                    result[ticker] = df
                    if self._cache is not None:
                        self._cache.set(ticker, start, end, interval, df)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to process %s from batch: %s", ticker, exc)

        return result
