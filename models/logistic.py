"""
models/logistic.py
Logistic Regression baseline model.
Fast, interpretable, good for establishing a performance baseline.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.base import BaseModel

logger = logging.getLogger(__name__)


class LogisticModel(BaseModel):
    """
    Logistic Regression with StandardScaler preprocessing.
    Uses sklearn Pipeline so the scaler is always fitted on training data only.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = "balanced",
    ) -> None:
        self._C = C
        self._max_iter = max_iter
        self._class_weight = class_weight
        self._pipeline: Pipeline | None = None
        self._fitted = False

    @property
    def name(self) -> str:
        return "LogisticRegression"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the logistic regression pipeline."""
        logger.info("Training %s on %d samples, %d features.", self.name, len(X), X.shape[1])
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=self._C,
                max_iter=self._max_iter,
                class_weight=self._class_weight,
                solver="lbfgs",
                random_state=42,
            )),
        ])
        self._pipeline.fit(X, y)
        self._fitted = True
        logger.info("%s training complete.", self.name)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities."""
        if not self._fitted or self._pipeline is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self._pipeline.predict_proba(X)

    def save(self, path: Path) -> None:
        """Save the fitted pipeline to disk using joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)
        logger.info("Saved %s to %s.", self.name, path)

    def load(self, path: Path) -> None:
        """Load a fitted pipeline from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self._pipeline = joblib.load(path)
        self._fitted = True
        logger.info("Loaded %s from %s.", self.name, path)
