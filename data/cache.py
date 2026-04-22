"""
data/cache.py
Simple disk-based cache for OHLCV DataFrames (Parquet format).
Avoids re-downloading data that is still fresh.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataCache:
    """
    Stores DataFrames as Parquet files on disk.
    Cache key = hash of (ticker, start, end, interval).
    Cache is considered stale after `ttl_hours` hours.
    """

    def __init__(self, cache_dir: str | Path, ttl_hours: int = 24) -> None:
        self.cache_dir = Path(cache_dir)
        self.ttl_hours = ttl_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def get(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str,
    ) -> pd.DataFrame | None:
        """Return cached DataFrame or None if missing/stale."""
        path = self._path(ticker, start, end, interval)
        if not path.exists():
            return None
        if self._is_stale(path):
            logger.debug("Cache stale for %s — will re-fetch.", ticker)
            return None
        try:
            df = pd.read_parquet(path)
            logger.debug("Cache hit for %s (%s → %s).", ticker, start, end)
            return df
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cache read error for %s: %s", ticker, exc)
            return None

    def set(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str,
        df: pd.DataFrame,
    ) -> None:
        """Persist a DataFrame to the cache."""
        if df.empty:
            return
        path = self._path(ticker, start, end, interval)
        try:
            df.to_parquet(path)
            logger.debug("Cached %s (%s → %s) → %s", ticker, start, end, path.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cache write error for %s: %s", ticker, exc)

    def invalidate(self, ticker: str, start: date, end: date, interval: str) -> None:
        """Remove a specific cache entry."""
        path = self._path(ticker, start, end, interval)
        if path.exists():
            path.unlink()

    def clear_all(self) -> None:
        """Delete all cached files."""
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
        logger.info("Cache cleared.")

    # ── Internals ─────────────────────────────────────────────────────────────

    def _path(self, ticker: str, start: date, end: date, interval: str) -> Path:
        key = f"{ticker}_{start}_{end}_{interval}"
        digest = hashlib.md5(key.encode()).hexdigest()[:12]
        safe_ticker = ticker.replace("/", "_").replace(".", "_")
        return self.cache_dir / f"{safe_ticker}_{interval}_{digest}.parquet"

    def _is_stale(self, path: Path) -> bool:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return datetime.now() - mtime > timedelta(hours=self.ttl_hours)
