"""
features/pipeline.py
Full feature engineering pipeline.
Orchestrates all feature functions and produces a clean, ML-ready DataFrame.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import Settings
from features.candles import add_candle_patterns
from features.session import add_session_features
from features.technical import (
    add_atr,
    add_bollinger_bands,
    add_candle_features,
    add_ema,
    add_macd,
    add_momentum,
    add_rsi,
    add_sma,
    add_stochastic,
    add_volume_features,
)
from features.returns import (
    add_atr_pct,
    add_high_low_range,
    add_returns,
    add_rolling_drawdown,
    add_rolling_volatility,
    add_target,
)

logger = logging.getLogger(__name__)

# Columns that must never be used as model features
_FORBIDDEN_FEATURE_COLS = {"target", "future_return", "open", "high", "low", "close", "volume"}


class FeaturePipeline:
    """
    Builds a complete feature matrix from raw OHLCV data.

    Usage
    -----
    pipeline = FeaturePipeline(settings)
    df_features = pipeline.transform(df_ohlcv, add_target=True)
    X, y = pipeline.get_Xy(df_features)
    feature_names = pipeline.feature_names
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cfg = settings.features
        # Cache whether news features are enabled at construction time.
        # This is used to decide whether to inject neutral news columns when
        # no sentiment data is provided, keeping feature count consistent.
        self._news_enabled: bool = bool(
            settings.data.get("news", {}).get("enabled", False)
        )

    @property
    def feature_names(self) -> list[str]:
        """Return the list of feature column names (set after first transform call)."""
        return getattr(self, "_feature_names", [])

    def transform(
        self,
        df: pd.DataFrame,
        include_target: bool = True,
        macro_df: pd.DataFrame | None = None,
        sentiment_df: pd.DataFrame | None = None,
        mtf_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Apply the full feature engineering pipeline to raw OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV DataFrame with DatetimeIndex and columns:
            open, high, low, close, volume.
        include_target : bool
            If True, compute and append the target column.
            Set to False for live inference (no future data available).
        macro_df : pd.DataFrame | None
            Optional macro features DataFrame (DatetimeIndex, any columns).
        sentiment_df : pd.DataFrame | None
            Optional news sentiment DataFrame (news_sentiment, news_positive_ratio,
            news_article_count). Dates outside lookback window are neutral-filled.
        mtf_df : pd.DataFrame | None
            Optional multi-timeframe features DataFrame produced by
            ``build_mtf_features()``. Columns prefixed with timeframe labels
            (e.g. ``1wk_rsi_14``, ``1mo_direction``, ``mtf_trend_score``).

        Returns
        -------
        pd.DataFrame
            Feature-enriched DataFrame. Raw OHLCV columns are retained
            but excluded from the feature set.
        """
        if df.empty:
            logger.warning("FeaturePipeline received empty DataFrame.")
            return df

        cfg = self._cfg

        # ── Technical indicators ──────────────────────────────────────────────
        df = add_rsi(df, period=int(cfg.get("rsi_period", 14)))
        df = add_macd(
            df,
            fast=int(cfg.get("macd_fast", 12)),
            slow=int(cfg.get("macd_slow", 26)),
            signal=int(cfg.get("macd_signal", 9)),
        )
        df = add_atr(df, period=int(cfg.get("atr_period", 14)))
        df = add_bollinger_bands(
            df,
            period=int(cfg.get("bb_period", 20)),
            std=float(cfg.get("bb_std", 2.0)),
        )
        df = add_sma(df, windows=list(cfg.get("sma_windows", [10, 20, 50])))
        df = add_ema(df, windows=list(cfg.get("ema_windows", [10, 20])))
        df = add_stochastic(df)
        df = add_candle_features(df)

        # ── Volume features ───────────────────────────────────────────────────
        df = add_volume_features(df, windows=list(cfg.get("volume_windows", [5, 10, 20])))

        # ── Return and volatility features ────────────────────────────────────
        return_periods = list(cfg.get("return_periods", [1, 3, 5, 10, 20]))
        df = add_returns(df, periods=return_periods)
        df = add_rolling_volatility(df, windows=list(cfg.get("volatility_windows", [5, 10, 20])))
        df = add_atr_pct(df, atr_col=f"atr_{cfg.get('atr_period', 14)}")
        df = add_rolling_drawdown(df, window=20)
        df = add_high_low_range(df, windows=[5, 10, 20])
        df = add_momentum(df, periods=return_periods)

        # ── Candle patterns ───────────────────────────────────────────────────
        df = add_candle_patterns(df)

        # ── Session / calendar / economic event features ──────────────────────
        df = add_session_features(df)

        # ── Multi-timeframe features (optional) ───────────────────────────────
        if mtf_df is not None and not mtf_df.empty:
            df = self._merge_mtf(df, mtf_df)

        # ── Macro features (optional) ─────────────────────────────────────────
        if macro_df is not None and not macro_df.empty:
            df = self._merge_macro(df, macro_df)

        # ── News sentiment features ────────────────────────────────────────────
        # Always add the 3 news columns when news is enabled so that the feature
        # count is IDENTICAL whether live news data is available or not.
        # This prevents the pipeline=133 / model=136 mismatch that occurs when
        # sentiment data is temporarily missing (API offline, market-closed fetch,
        # first poll cycle before the daily news refresh, etc.).
        if sentiment_df is not None and not sentiment_df.empty:
            df = self._merge_sentiment(df, sentiment_df)
        elif self._news_enabled:
            # News is configured but data unavailable for this ticker right now
            # (API offline, first poll cycle, market-closed skip, etc.)
            # → inject neutral values so feature count stays at 136 consistently.
            df["news_sentiment"]      = 0.0
            df["news_positive_ratio"] = 0.5
            df["news_article_count"]  = 0.0

        # ── Target variable ───────────────────────────────────────────────────
        if include_target:
            df = add_target(
                df,
                horizon=int(cfg.get("target_horizon", 1)),
                threshold=float(cfg.get("target_threshold", 0.0)),
            )

        # ── Drop rows with NaN (from rolling windows) ─────────────────────────
        n_before = len(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=self._get_feature_cols(df))
        n_after = len(df)
        logger.debug(
            "Feature pipeline: dropped %d rows with NaN/Inf (kept %d/%d).",
            n_before - n_after, n_after, n_before,
        )

        # Cache feature names (excluding raw OHLCV and target)
        self._feature_names = self._get_feature_cols(df)

        return df

    def get_Xy(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Split a feature DataFrame into X (features) and y (target).

        Parameters
        ----------
        df : pd.DataFrame
            Output of transform() with include_target=True.

        Returns
        -------
        X : pd.DataFrame
            Feature matrix (no target, no raw OHLCV).
        y : pd.Series
            Binary target series (1=UP, 0=DOWN).
        """
        if "target" not in df.columns:
            raise ValueError("DataFrame does not contain 'target' column. "
                             "Call transform(include_target=True) first.")

        feature_cols = self._get_feature_cols(df)
        X = df[feature_cols].copy()
        y = df["target"].copy()

        # Drop rows where target is NaN (last horizon rows)
        valid = y.notna()
        return X[valid], y[valid]

    # ── Internals ─────────────────────────────────────────────────────────────

    def _get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Return all columns that are valid features (not raw OHLCV or target)."""
        return [
            c for c in df.columns
            if c not in _FORBIDDEN_FEATURE_COLS
        ]

    @staticmethod
    def _merge_mtf(df: pd.DataFrame, mtf_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge multi-timeframe features into the OHLCV DataFrame.
        Aligns on DatetimeIndex; any remaining NaN is forward-filled then zero-filled.
        """
        aligned = mtf_df.reindex(df.index, method="ffill").fillna(0)
        df = pd.concat([df, aligned], axis=1)
        return df

    @staticmethod
    def _merge_sentiment(df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge news sentiment features into the OHLCV DataFrame.
        Aligns on DatetimeIndex; any remaining NaN is filled with neutral values
        so that pipeline dropna never removes rows due to missing news data.
        """
        # Forward-fill to nearest trading day, then neutral-fill gaps
        aligned = sentiment_df.reindex(df.index, method="ffill")
        if "news_sentiment" in aligned.columns:
            aligned["news_sentiment"]      = aligned["news_sentiment"].fillna(0.0)
        if "news_positive_ratio" in aligned.columns:
            aligned["news_positive_ratio"] = aligned["news_positive_ratio"].fillna(0.5)
        if "news_article_count" in aligned.columns:
            aligned["news_article_count"]  = aligned["news_article_count"].fillna(0.0)

        df = pd.concat([df, aligned], axis=1)
        return df

    @staticmethod
    def _merge_macro(df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge macro features into the OHLCV DataFrame.
        Aligns on DatetimeIndex, forward-fills macro data to match trading days.
        """
        # Prefix macro columns to avoid name collisions
        macro_df = macro_df.copy()
        macro_df.columns = [f"macro_{c}" for c in macro_df.columns]

        # Reindex macro to match OHLCV dates, forward-fill
        macro_aligned = macro_df.reindex(df.index, method="ffill")
        df = pd.concat([df, macro_aligned], axis=1)
        return df
