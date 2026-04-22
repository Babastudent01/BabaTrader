"""
data/alpha_vantage.py
Alpha Vantage data provider.
Requires ALPHA_VANTAGE_API_KEY in config/.env.
Free tier: 25 requests/day, 5 requests/minute.
"""
from __future__ import annotations

import logging
import time
from datetime import date
from typing import Any

import pandas as pd
import requests

from config import get_env
from data.base import DataProvider
from data.cache import DataCache

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.alphavantage.co/query"

# Map our interval strings to Alpha Vantage function names
_INTERVAL_MAP: dict[str, tuple[str, str | None]] = {
    "1d":  ("TIME_SERIES_DAILY_ADJUSTED", None),
    "1wk": ("TIME_SERIES_WEEKLY_ADJUSTED", None),
    "1mo": ("TIME_SERIES_MONTHLY_ADJUSTED", None),
    "60m": ("TIME_SERIES_INTRADAY", "60min"),
    "1h":  ("TIME_SERIES_INTRADAY", "60min"),
    "30m": ("TIME_SERIES_INTRADAY", "30min"),
    "15m": ("TIME_SERIES_INTRADAY", "15min"),
    "5m":  ("TIME_SERIES_INTRADAY", "5min"),
    "1m":  ("TIME_SERIES_INTRADAY", "1min"),
}


class AlphaVantageProvider(DataProvider):
    """
    Fetches OHLCV data from Alpha Vantage REST API.

    Notes
    -----
    - Requires ALPHA_VANTAGE_API_KEY environment variable.
    - Free tier is heavily rate-limited (25 req/day).
    - Intraday data is limited to the last 30 days on free tier.
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache: DataCache | None = None,
        request_delay: float = 12.0,  # seconds between requests (free tier: 5/min)
    ) -> None:
        self._api_key = api_key or get_env("ALPHA_VANTAGE_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Alpha Vantage API key not set. "
                "Add ALPHA_VANTAGE_API_KEY to config/.env."
            )
        self._cache = cache
        self._request_delay = request_delay
        self._last_request_time: float = 0.0

    def fetch_ohlcv(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV from Alpha Vantage."""
        if self._cache is not None:
            cached = self._cache.get(ticker, start, end, interval)
            if cached is not None:
                return cached

        if interval not in _INTERVAL_MAP:
            raise ValueError(
                f"Unsupported interval '{interval}' for Alpha Vantage. "
                f"Supported: {list(_INTERVAL_MAP.keys())}"
            )

        function, av_interval = _INTERVAL_MAP[interval]
        params: dict[str, Any] = {
            "function": function,
            "symbol": ticker,
            "apikey": self._api_key,
            "outputsize": "full",
            "datatype": "json",
        }
        if av_interval:
            params["interval"] = av_interval

        self._throttle()
        logger.info("Fetching %s from Alpha Vantage (%s).", ticker, interval)

        try:
            response = requests.get(_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            logger.error("Alpha Vantage request failed for %s: %s", ticker, exc)
            return pd.DataFrame()

        # Check for API error messages
        if "Error Message" in data:
            logger.error("Alpha Vantage error for %s: %s", ticker, data["Error Message"])
            return pd.DataFrame()
        if "Note" in data:
            logger.warning("Alpha Vantage rate limit note: %s", data["Note"])

        df = self._parse_response(data, function)
        if df.empty:
            return df

        # Filter to requested date range
        df = df.loc[str(start):str(end)]

        if self._cache is not None:
            self._cache.set(ticker, start, end, interval, df)

        logger.info("Fetched %d bars for %s from Alpha Vantage.", len(df), ticker)
        return df

    # ── Internals ─────────────────────────────────────────────────────────────

    def _parse_response(self, data: dict, function: str) -> pd.DataFrame:
        """Extract the time series from the API response."""
        # Find the time series key (varies by function)
        ts_key = next(
            (k for k in data if "Time Series" in k or "Weekly" in k or "Monthly" in k),
            None,
        )
        if ts_key is None:
            logger.warning("No time series key found in Alpha Vantage response.")
            return pd.DataFrame()

        ts = data[ts_key]
        records = []
        for dt_str, values in ts.items():
            try:
                records.append({
                    "date": pd.to_datetime(dt_str),
                    "open":   float(values.get("1. open",   values.get("1. open",   0))),
                    "high":   float(values.get("2. high",   values.get("2. high",   0))),
                    "low":    float(values.get("3. low",    values.get("3. low",    0))),
                    "close":  float(values.get("5. adjusted close", values.get("4. close", 0))),
                    "volume": float(values.get("6. volume", values.get("5. volume", 0))),
                })
            except (ValueError, KeyError) as exc:
                logger.debug("Skipping malformed row %s: %s", dt_str, exc)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).set_index("date").sort_index()
        return df

    def _throttle(self) -> None:
        """Enforce minimum delay between API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._request_delay:
            time.sleep(self._request_delay - elapsed)
        self._last_request_time = time.time()
