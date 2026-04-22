"""
strategy/filters.py
Safety filters applied before generating signals.
Prevents trading in poor liquidity or extreme volatility conditions.
"""
from __future__ import annotations

import logging

import pandas as pd

from config import Settings

logger = logging.getLogger(__name__)


class SafetyFilter:
    """
    Rule-based safety filter that screens tickers before signal generation.

    Checks:
    1. Minimum average volume (liquidity filter)
    2. Maximum ATR as % of price (volatility filter)
    3. Minimum price (penny stock filter)
    4. Bank holiday / market closure (is_holiday feature)
    5. No-news quiet week (no_major_news_week feature, optional)
    6. Session filter — avoid Asian-only bars (optional, intraday)

    All checks can be configured in settings.yaml under strategy.filters.
    """

    def __init__(self, settings: Settings) -> None:
        filter_cfg = settings.strategy.filters
        self._min_avg_volume: float  = float(filter_cfg.get("min_avg_volume", 500_000))
        self._max_atr_pct: float     = float(filter_cfg.get("max_atr_pct", 0.05))
        self._min_price: float       = float(filter_cfg.get("min_price", 5.0))
        self._allow_high_vol: bool   = bool(filter_cfg.get("allow_high_volatility", False))
        self._block_holidays: bool   = bool(filter_cfg.get("block_holidays", True))
        self._prefer_news_days: bool = bool(filter_cfg.get("prefer_news_days", False))
        self._session_filter: bool   = bool(filter_cfg.get("session_filter", False))

    def is_tradeable(
        self,
        ticker: str,
        df: pd.DataFrame,
        lookback: int = 20,
    ) -> tuple[bool, str]:
        """
        Check if a ticker passes all safety filters.

        Parameters
        ----------
        ticker : str
            Ticker symbol (for logging).
        df : pd.DataFrame
            OHLCV DataFrame with feature columns. Must have at least `lookback` rows.
        lookback : int
            Number of bars to use for rolling averages.

        Returns
        -------
        (bool, str)
            (True, "") if tradeable, (False, reason) if filtered out.
        """
        # ── Feature-based filters (require session features to be computed first) ──
        last_row = df.iloc[-1] if not df.empty else None

        # Holiday filter: block trading on bank/public holidays
        if self._block_holidays and last_row is not None:
            if int(last_row.get("is_holiday", 0)) == 1:
                reason = "Bank holiday — market closed or thin liquidity"
                logger.debug("Filter REJECT %s: %s", ticker, reason)
                return False, reason

        # Session filter: avoid pure Asian-session bars (intraday only)
        if self._session_filter and last_row is not None:
            if int(last_row.get("session_avoid", 0)) == 1:
                reason = "Asian session only — not a preferred trading window"
                logger.debug("Filter REJECT %s: %s", ticker, reason)
                return False, reason

        # No-news filter: skip signals on calendar quiet weeks
        if self._prefer_news_days and last_row is not None:
            if int(last_row.get("no_major_news_week", 0)) == 1:
                reason = "No major economic events this week — skipping (prefer_news_days=true)"
                logger.debug("Filter REJECT %s: %s", ticker, reason)
                return False, reason

        if df.empty or len(df) < lookback:
            return False, f"Insufficient data ({len(df)} bars, need {lookback})"

        last = df.iloc[-1]

        # ── Price filter ──────────────────────────────────────────────────────
        price = float(last.get("close", 0))
        if price < self._min_price:
            reason = f"Price {price:.2f} below minimum {self._min_price:.2f}"
            logger.debug("Filter REJECT %s: %s", ticker, reason)
            return False, reason

        # ── Liquidity filter ──────────────────────────────────────────────────
        avg_volume = df["volume"].tail(lookback).mean()
        if avg_volume < self._min_avg_volume:
            reason = (
                f"Avg volume {avg_volume:,.0f} below minimum "
                f"{self._min_avg_volume:,.0f}"
            )
            logger.debug("Filter REJECT %s: %s", ticker, reason)
            return False, reason

        # ── Volatility filter ─────────────────────────────────────────────────
        if not self._allow_high_vol:
            # Use ATR % if available, else compute from OHLCV
            if "atr_pct" in df.columns:
                atr_pct = float(last.get("atr_pct", 0))
            elif "atr_14" in df.columns and price > 0:
                atr_pct = float(last.get("atr_14", 0)) / price
            else:
                # Fallback: use high-low range
                atr_pct = (float(last.get("high", price)) - float(last.get("low", price))) / price

            if atr_pct > self._max_atr_pct:
                reason = (
                    f"ATR% {atr_pct:.3f} exceeds maximum {self._max_atr_pct:.3f} "
                    f"(high volatility)"
                )
                logger.debug("Filter REJECT %s: %s", ticker, reason)
                return False, reason

        return True, ""

    def filter_universe(
        self,
        data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """
        Apply filters to a universe of tickers.

        Parameters
        ----------
        data : dict[str, pd.DataFrame]
            Ticker → OHLCV DataFrame mapping.

        Returns
        -------
        dict[str, pd.DataFrame]
            Filtered subset of tickers that pass all checks.
        """
        passed: dict[str, pd.DataFrame] = {}
        for ticker, df in data.items():
            ok, reason = self.is_tradeable(ticker, df)
            if ok:
                passed[ticker] = df
            else:
                logger.info("Filtered out %s: %s", ticker, reason)

        logger.info(
            "Safety filter: %d/%d tickers passed.",
            len(passed), len(data),
        )
        return passed
