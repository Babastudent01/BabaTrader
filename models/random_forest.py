"""
models/random_forest.py
Random Forest classifier model.
Good balance of performance and interpretability (feature importances).
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.base import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """
    Random Forest classifier.
    Does not strictly require scaling, but we include it for consistency.
    Provides feature_importances_ after training.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        min_samples_leaf: int = 20,
        class_weight: str = "balanced",
        n_jobs: int = -1,
    ) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._min_samples_leaf = min_samples_leaf
        self._class_weight = class_weight
        self._n_jobs = n_jobs
        self._pipeline: Pipeline | None = None
        self._feature_names: list[str] = []
        self._fitted = False

    @property
    def name(self) -> str:
        return "RandomForest"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the Random Forest pipeline."""
        logger.info("Training %s on %d samples, %d features.", self.name, len(X), X.shape[1])
        self._feature_names = list(X.columns)
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=self._n_estimators,
                max_depth=self._max_depth,
                min_samples_leaf=self._min_samples_leaf,
                class_weight=self._class_weight,
                n_jobs=self._n_jobs,
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

    @property
    def feature_importances(self) -> pd.Series | None:
        """Return feature importances as a sorted Series (if fitted)."""
        if not self._fitted or self._pipeline is None:
            return None
        clf = self._pipeline.named_steps["clf"]
        return pd.Series(
            clf.feature_importances_,
            index=self._feature_names,
        ).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        """Save the fitted pipeline to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self._pipeline, "feature_names": self._feature_names}, path)
        logger.info("Saved %s to %s.", self.name, path)

    def load(self, path: Path) -> None:
        """Load a fitted pipeline from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        data = joblib.load(path)
        self._pipeline = data["pipeline"]
        self._feature_names = data.get("feature_names", [])
        self._fitted = True
        logger.info("Loaded %s from %s.", self.name, path)
