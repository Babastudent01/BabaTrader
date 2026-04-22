"""
data/macro.py
Macroeconomic data provider using FRED (Federal Reserve Economic Data).
Requires FRED_API_KEY in config/.env.
Free API — register at https://fred.stlouisfed.org/docs/api/api_key.html
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from config import get_env
from data.cache import DataCache

logger = logging.getLogger(__name__)

_FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


class FREDProvider:
    """
    Fetches macroeconomic time series from FRED.

    Common series IDs:
        DGS10    — 10-Year Treasury Constant Maturity Rate
        DGS2     — 2-Year Treasury Rate
        UNRATE   — Unemployment Rate
        CPIAUCSL — Consumer Price Index (All Urban Consumers)
        FEDFUNDS — Federal Funds Effective Rate
        VIXCLS   — CBOE Volatility Index (VIX)
        T10Y2Y   — 10-Year minus 2-Year Treasury Spread (yield curve)
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache: DataCache | None = None,
    ) -> None:
        self._api_key = api_key or get_env("FRED_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "FRED API key not set. Add FRED_API_KEY to config/.env. "
                "Register free at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        self._cache = cache

    def fetch_series(
        self,
        series_id: str,
        start: date,
        end: date,
    ) -> pd.Series:
        """
        Fetch a single FRED series as a pandas Series.

        Parameters
        ----------
        series_id : str
            FRED series identifier (e.g. 'DGS10').
        start : date
            Start date.
        end : date
            End date.

        Returns
        -------
        pd.Series
            DatetimeIndex, values as floats. Missing values are forward-filled.
        """
        # Use cache with a synthetic ticker name
        cache_ticker = f"FRED_{series_id}"
        if self._cache is not None:
            cached = self._cache.get(cache_ticker, start, end, "1d")
            if cached is not None and not cached.empty:
                return cached["value"]

        logger.info("Fetching FRED series %s (%s → %s).", series_id, start, end)

        params = {
            "series_id": series_id,
            "observation_start": str(start),
            "observation_end": str(end),
            "api_key": self._api_key,
            "file_type": "json",
            "frequency": "d",       # Daily frequency
            "aggregation_method": "avg",
        }

        try:
            response = requests.get(_FRED_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            logger.error("FRED request failed for %s: %s", series_id, exc)
            return pd.Series(dtype=float)

        observations = data.get("observations", [])
        if not observations:
            logger.warning("No observations returned for FRED series %s.", series_id)
            return pd.Series(dtype=float)

        records = []
        for obs in observations:
            val_str = obs.get("value", ".")
            if val_str == ".":
                continue  # Missing value in FRED
            try:
                records.append({
                    "date": pd.to_datetime(obs["date"]),
                    "value": float(val_str),
                })
            except (ValueError, KeyError):
                continue

        if not records:
            return pd.Series(dtype=float)

        series = (
            pd.DataFrame(records)
            .set_index("date")["value"]
            .sort_index()
        )

        # Cache as a single-column DataFrame
        if self._cache is not None:
            df_cache = series.to_frame("value")
            self._cache.set(cache_ticker, start, end, "1d", df_cache)

        logger.info("Fetched %d observations for FRED %s.", len(series), series_id)
        return series

    def fetch_multiple_series(
        self,
        series_ids: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """
        Fetch multiple FRED series and return as a DataFrame.
        Columns are series IDs. Index is DatetimeIndex (daily, forward-filled).
        """
        frames: dict[str, pd.Series] = {}
        for sid in series_ids:
            try:
                s = self.fetch_series(sid, start, end)
                if not s.empty:
                    frames[sid] = s
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to fetch FRED %s: %s", sid, exc)

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        # Forward-fill to handle weekends/holidays where macro data is not published
        df = df.ffill()
        return df
