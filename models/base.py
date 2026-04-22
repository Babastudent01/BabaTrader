"""
models/base.py
Abstract base class for all ML models in the trading bot.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Abstract base for all prediction models.

    All models must:
    - Accept a feature DataFrame and return probability predictions.
    - Support save/load for persistence.
    - Expose a name property for logging and reporting.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model on the given feature matrix and target.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix. Must NOT contain target or raw OHLCV columns.
        y : pd.Series
            Binary target (1=UP, 0=DOWN).
        """

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return class probabilities for each sample.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Column 0: P(DOWN), Column 1: P(UP).
        """

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return binary class predictions (0 or 1).
        Default: argmax of predict_proba.
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return the confidence (max probability) for each prediction.
        """
        proba = self.predict_proba(X)
        return proba.max(axis=1)

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the trained model to disk."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load a trained model from disk."""

    @property
    def feature_names(self) -> list[str] | None:
        """
        Feature names the model was trained on. Returns None if not stored.
        Subclasses that store ``_feature_names`` get this for free.
        The signal generator uses this for automatic alignment when the
        feature set changes between training runs.
        """
        names = getattr(self, "_feature_names", None)
        return list(names) if names else None

    @property
    def is_fitted(self) -> bool:
        """Return True if the model has been trained."""
        return getattr(self, "_fitted", False)
