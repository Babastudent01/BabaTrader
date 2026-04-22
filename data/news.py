"""
data/news.py
News sentiment provider using NewsAPI + VADER.

Fetches recent headlines for each ticker and scores them with VADER
(Valence Aware Dictionary and sEntiment Reasoner), a lexicon-based
sentiment analyser tuned for short financial news text.

Free-tier limitations (NewsAPI):
  - Max 30 days of historical data
  - 100 requests / day  ← this module caches each ticker's result to disk for
                           the current calendar day so repeated calls (live
                           polling, multiple training runs) never hit the API
                           more than once per ticker per day.

Cache location: data/cache/news/{TICKER}_{YYYY-MM-DD}.parquet
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Maps ticker symbols to richer search queries for better article recall.
# French tickers use the full company name because Yahoo Finance suffixes like
# ".PA" are meaningless to NewsAPI's full-text search engine.
_TICKER_QUERIES: dict[str, str] = {
    # ── US large-cap ──────────────────────────────────────────────────────
    "AAPL":  "Apple Inc stock",
    "MSFT":  "Microsoft stock",
    "GOOGL": "Google Alphabet stock",
    "AMZN":  "Amazon stock",
    "NVDA":  "NVIDIA stock",
    "META":  "Meta Facebook stock",
    "TSLA":  "Tesla stock",
    "NFLX":  "Netflix stock",
    "JPM":   "JPMorgan Chase stock",
    "GS":    "Goldman Sachs stock",
    "BABA":  "Alibaba stock",
    "TSM":   "TSMC semiconductor stock",
    # ── French CAC-40 ─────────────────────────────────────────────────────
    "HO.PA":  "Thales Group defense aerospace",
    "DSY.PA": "Dassault Systemes software",
    "TTE.PA": "TotalEnergies oil gas energy",
    "AIR.PA": "Airbus aircraft aerospace",
    "AI.PA":  "Air Liquide industrial gases",
    "BNP.PA": "BNP Paribas bank",
    "CS.PA":  "AXA insurance",
    "MC.PA":  "LVMH luxury",
    "ORA.PA": "Orange telecom France",
    "VIE.PA": "Veolia environment water",
    "GLE.PA": "Societe Generale bank France",
}

# Default values used for dates with no news coverage
_NEUTRAL_SENTIMENT     = 0.0
_NEUTRAL_POS_RATIO     = 0.5
_NEUTRAL_ARTICLE_COUNT = 0.0


class NewsSentimentProvider:
    """
    Fetches news headlines from NewsAPI and scores them with VADER sentiment.

    Produces three daily features per ticker:
    - ``news_sentiment``       — mean VADER compound score [-1 = very negative, +1 = very positive]
    - ``news_positive_ratio``  — fraction of articles with compound > 0.05
    - ``news_article_count``   — raw article count (indicator of news volume / market attention)

    Dates outside the 30-day lookback window (NewsAPI free-tier limit) are
    filled with neutral values so that pipeline ``dropna`` does not discard
    historical training rows.
    """

    _CACHE_DIR = Path("data/cache/news")

    def __init__(self, api_key: str, lookback_days: int = 30) -> None:
        self._api_key       = api_key
        self._lookback_days = min(lookback_days, 28)   # hard cap: 28 days (free-tier boundary is exclusive)
        self._client        = None
        self._analyser      = None
        self._CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_sentiment(
        self,
        ticker: str,
        index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Fetch and score news for *ticker*, then align to *index*.

        Parameters
        ----------
        ticker : str
            Ticker symbol, e.g. 'AAPL'.
        index : pd.DatetimeIndex
            The trading-day DatetimeIndex of the OHLCV DataFrame.
            The returned DataFrame shares this exact index.

        Returns
        -------
        pd.DataFrame
            Columns: news_sentiment, news_positive_ratio, news_article_count.
            Indexed by *index*. Never contains NaN.
        """
        # ── Daily cache check ─────────────────────────────────────────────
        today       = date.today()
        cache_path  = self._CACHE_DIR / f"{ticker.replace('.', '_')}_{today}.parquet"

        sentiment_df = self._load_cache(cache_path)
        if sentiment_df is not None:
            logger.info(
                "Fetching news sentiment for %s (%s -> %s)...", ticker, today, today
            )
            logger.info(
                "%s news: served from daily cache (%s).",
                ticker, cache_path.name,
            )
        else:
            # Cache miss — fetch from API
            self._ensure_client()
            self._ensure_analyser()

            query    = _TICKER_QUERIES.get(ticker, f"{ticker} stock")
            end_dt   = today
            start_dt = end_dt - timedelta(days=max(self._lookback_days - 2, 1))

            logger.info(
                "Fetching news sentiment for %s (%s -> %s)...", ticker, start_dt, end_dt
            )

            articles     = self._fetch_articles(query, start_dt, end_dt)
            daily_scores = self._score_articles(articles)

            # Build raw sentiment DataFrame (only days with real news)
            if daily_scores:
                rows = [
                    {
                        "date":                 pd.Timestamp(d),
                        "news_sentiment":       sum(sc) / len(sc),
                        "news_positive_ratio":  sum(1 for s in sc if s > 0.05) / len(sc),
                        "news_article_count":   float(len(sc)),
                    }
                    for d, sc in sorted(daily_scores.items())
                ]
                sentiment_df = pd.DataFrame(rows).set_index("date")
                sentiment_df.index = pd.DatetimeIndex(sentiment_df.index)
            else:
                sentiment_df = pd.DataFrame(
                    columns=["news_sentiment", "news_positive_ratio", "news_article_count"],
                    dtype=float,
                )
                sentiment_df.index = pd.DatetimeIndex([])

            # Save to today's cache file
            self._save_cache(sentiment_df, cache_path)

            n_real = len(sentiment_df)
            logger.info(
                "%s news: %d days with real sentiment, %d days with neutral fill.",
                ticker, n_real, len(index) - n_real,
            )

        n_real = len(sentiment_df)

        # Align to the trading-day index with forward-fill, then neutral fill
        aligned = sentiment_df.reindex(index, method="ffill")
        aligned["news_sentiment"]      = aligned["news_sentiment"].fillna(_NEUTRAL_SENTIMENT)
        aligned["news_positive_ratio"] = aligned["news_positive_ratio"].fillna(_NEUTRAL_POS_RATIO)
        aligned["news_article_count"]  = aligned["news_article_count"].fillna(_NEUTRAL_ARTICLE_COUNT)

        return aligned

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _load_cache(self, path: Path) -> pd.DataFrame | None:
        """Return cached sentiment DataFrame if today's file exists, else None."""
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            df.index = pd.DatetimeIndex(df.index)
            return df
        except Exception as exc:
            logger.warning("Failed to read news cache %s: %s", path, exc)
            return None

    def _save_cache(self, df: pd.DataFrame, path: Path) -> None:
        """Save sentiment DataFrame to today's cache file."""
        try:
            # Also prune cache files older than 2 days to avoid stale accumulation
            for old in self._CACHE_DIR.glob("*.parquet"):
                try:
                    file_date = date.fromisoformat(old.stem.split("_")[-1])
                    if (date.today() - file_date).days > 1:
                        old.unlink()
                except Exception:
                    pass
            df.to_parquet(path)
        except Exception as exc:
            logger.warning("Failed to write news cache %s: %s", path, exc)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _fetch_articles(
        self,
        query: str,
        start_dt: date,
        end_dt: date,
    ) -> list[dict]:
        """Call NewsAPI and return raw article list."""
        try:
            response = self._client.get_everything(
                q=query,
                from_param=str(start_dt),
                to=str(end_dt),
                language="en",
                sort_by="publishedAt",
                page_size=100,
            )
            return response.get("articles", [])
        except Exception as exc:
            logger.warning(
                "NewsAPI request failed for '%s': %s. Using neutral sentiment.", query, exc
            )
            return []

    def _score_articles(
        self,
        articles: list[dict],
    ) -> dict[date, list[float]]:
        """Score each article with VADER and group scores by publication date."""
        daily: dict[date, list[float]] = {}

        for article in articles:
            pub_str = article.get("publishedAt", "")[:10]   # "YYYY-MM-DD"
            try:
                pub_date = date.fromisoformat(pub_str)
            except ValueError:
                continue

            text = " ".join(filter(None, [
                article.get("title", ""),
                article.get("description", ""),
            ])).strip()
            if not text:
                continue

            compound = self._analyser.polarity_scores(text)["compound"]
            daily.setdefault(pub_date, []).append(compound)

        return daily

    def _ensure_client(self) -> None:
        if self._client is None:
            try:
                from newsapi import NewsApiClient
                self._client = NewsApiClient(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "newsapi-python is not installed. "
                    "Run: pip install newsapi-python"
                )

    def _ensure_analyser(self) -> None:
        if self._analyser is None:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self._analyser = SentimentIntensityAnalyzer()
            except ImportError:
                raise ImportError(
                    "vaderSentiment is not installed. "
                    "Run: pip install vaderSentiment"
                )
