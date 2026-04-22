"""
strategy/generator.py
Signal generator: combines model predictions with safety filters
to produce actionable BUY/SELL/HOLD signals.
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import pytz

from config import Settings
from features.pipeline import FeaturePipeline
from models.base import BaseModel
from strategy.filters import SafetyFilter
from strategy.signal import Signal, SignalType

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Generates trading signals by:
    1. Applying safety filters to screen out untradeable tickers.
    2. Running the feature pipeline on OHLCV data.
    3. Getting model predictions and confidence scores.
    4. Emitting BUY/SELL/HOLD signals based on confidence threshold.

    Signal logic (long-only, conservative):
    - P(UP) >= threshold → BUY
    - P(UP) <= (1 - threshold) → SELL (exit existing position)
    - Otherwise → HOLD
    """

    def __init__(
        self,
        model: BaseModel,
        settings: Settings,
        feature_pipeline: FeaturePipeline | None = None,
    ) -> None:
        self._model = model
        self._settings = settings
        self._threshold = float(settings.strategy.get("confidence_threshold", 0.60))
        self._pipeline = feature_pipeline or FeaturePipeline(settings)
        self._filter = SafetyFilter(settings)
        tz_name = settings.get("timezone", "Europe/Paris")
        self._tz = pytz.timezone(tz_name)

    def generate(
        self,
        data: dict[str, pd.DataFrame],
        timestamp: datetime | None = None,
        sentiment_data: dict[str, pd.DataFrame] | None = None,
        mtf_data: dict[str, pd.DataFrame] | None = None,
    ) -> list[Signal]:
        """
        Generate signals for all tickers in the universe.

        Parameters
        ----------
        data : dict[str, pd.DataFrame]
            Ticker -> OHLCV DataFrame.
        timestamp : datetime | None
            Signal timestamp. Defaults to now (Europe/Paris).
        sentiment_data : dict[str, pd.DataFrame] | None
            Optional ticker -> news sentiment DataFrame.
        mtf_data : dict[str, pd.DataFrame] | None
            Optional ticker -> multi-timeframe feature DataFrame produced by
            ``build_mtf_features()``. Higher-TF indicators are merged into the
            pipeline before prediction.

        Returns
        -------
        list[Signal]
            List of signals (BUY/SELL/HOLD) for each ticker that passed filters.
        """
        if not self._model.is_fitted:
            raise RuntimeError("Model is not fitted. Train the model before generating signals.")

        ts = timestamp or datetime.now(self._tz)
        signals: list[Signal] = []

        for ticker, df in data.items():
            try:
                sdf   = sentiment_data.get(ticker) if sentiment_data else None
                mdf   = mtf_data.get(ticker)       if mtf_data       else None
                signal = self._generate_for_ticker(ticker, df, ts, sentiment_df=sdf, mtf_df=mdf)
                if signal is not None:
                    signals.append(signal)
            except Exception as exc:  # noqa: BLE001
                logger.error("Error generating signal for %s: %s", ticker, exc)

        logger.info(
            "Generated %d signals: %d BUY, %d SELL, %d HOLD.",
            len(signals),
            sum(1 for s in signals if s.signal_type == SignalType.BUY),
            sum(1 for s in signals if s.signal_type == SignalType.SELL),
            sum(1 for s in signals if s.signal_type == SignalType.HOLD),
        )
        return signals

    def generate_for_bar(
        self,
        ticker: str,
        df: pd.DataFrame,
        timestamp: datetime | None = None,
    ) -> Signal | None:
        """
        Generate a signal for a single ticker at the current bar.
        Convenience wrapper around generate().
        """
        ts = timestamp or datetime.now(self._tz)
        return self._generate_for_ticker(ticker, df, ts)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _generate_for_ticker(
        self,
        ticker: str,
        df: pd.DataFrame,
        ts: datetime,
        sentiment_df: pd.DataFrame | None = None,
        mtf_df: pd.DataFrame | None = None,
    ) -> Signal | None:
        """Core signal generation logic for a single ticker."""
        if df.empty:
            logger.warning("Empty DataFrame for %s — skipping.", ticker)
            return None

        # Apply safety filter on raw OHLCV (before feature engineering)
        ok, reason = self._filter.is_tradeable(ticker, df)
        if not ok:
            logger.debug("Skipping %s (filtered): %s", ticker, reason)
            return Signal(
                ticker=ticker,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                timestamp=ts,
                price=float(df["close"].iloc[-1]),
                metadata={"filtered": True, "reason": reason},
            )

        # Apply feature pipeline (no target — live inference)
        df_feat = self._pipeline.transform(
            df, include_target=False, sentiment_df=sentiment_df, mtf_df=mtf_df
        )
        if df_feat.empty:
            logger.warning("Feature pipeline returned empty DataFrame for %s.", ticker)
            return None

        feature_cols = self._pipeline.feature_names
        if not feature_cols:
            logger.warning("No feature columns available for %s.", ticker)
            return None

        # DeepModel needs seq_len rows to build a sequence.
        # Pass enough history so the model can compute a proper prediction.
        n_ctx = max(getattr(self._model, "_seq_len", 1), 1)
        last_rows = df_feat[feature_cols].iloc[-n_ctx:]  # (≤n_ctx rows, n_features)

        # ── Feature alignment ─────────────────────────────────────────────
        model_features = getattr(self._model, "feature_names", None)
        if model_features is not None and list(model_features) != list(last_rows.columns):
            missing = set(model_features) - set(last_rows.columns)
            extra   = set(last_rows.columns) - set(model_features)
            if missing or extra:
                logger.warning(
                    "%s: feature mismatch — pipeline=%d, model=%d cols "
                    "(%d missing filled with 0, %d extra dropped). "
                    "Retrain the model to fix permanently.",
                    ticker, len(last_rows.columns), len(model_features),
                    len(missing), len(extra),
                )
            last_rows = last_rows.reindex(columns=list(model_features), fill_value=0.0)

        # Get model prediction — take the LAST row's prediction
        # (for ML: independent per-row; for DeepModel: uses the full sequence)
        proba  = self._model.predict_proba(last_rows)
        p_up   = float(proba[-1, 1])   # Probability of UP  (last bar)
        p_down = float(proba[-1, 0])   # Probability of DOWN (last bar)

        # Determine signal type
        if p_up >= self._threshold:
            signal_type = SignalType.BUY
            confidence  = p_up
        elif p_down >= self._threshold:
            signal_type = SignalType.SELL
            confidence  = p_down
        else:
            signal_type = SignalType.HOLD
            confidence  = max(p_up, p_down)

        # Get current price and ATR
        price = float(df["close"].iloc[-1])
        atr_col = f"atr_{self._settings.features.get('atr_period', 14)}"
        atr = float(df_feat[atr_col].iloc[-1]) if atr_col in df_feat.columns else 0.0

        signal = Signal(
            ticker=ticker,
            signal_type=signal_type,
            confidence=confidence,
            timestamp=ts,
            price=price,
            atr=atr,
            metadata={
                "p_up": p_up,
                "p_down": p_down,
                "model": self._model.name,
            },
        )

        logger.info("Signal: %s", signal)
        return signal

    def get_actionable_signals(
        self,
        data: dict[str, pd.DataFrame],
        timestamp: datetime | None = None,
    ) -> list[Signal]:
        """
        Generate signals and return only those that are actionable
        (BUY or SELL with confidence >= threshold).
        """
        all_signals = self.generate(data, timestamp)
        return [s for s in all_signals if s.is_actionable(self._threshold)]
