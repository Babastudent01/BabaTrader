"""
models/gradient_boosting.py
Gradient Boosting model using XGBoost or LightGBM (with sklearn fallback).
Typically the strongest performer for tabular financial data.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from models.base import BaseModel

logger = logging.getLogger(__name__)

# Try XGBoost first, then LightGBM, then sklearn GradientBoosting
try:
    from xgboost import XGBClassifier
    _BACKEND = "xgboost"
except ImportError:
    try:
        from lightgbm import LGBMClassifier
        _BACKEND = "lightgbm"
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        _BACKEND = "sklearn"

logger.info("GradientBoostingModel using backend: %s", _BACKEND)


class GradientBoostingModel(BaseModel):
    """
    Gradient Boosting classifier.
    Automatically uses XGBoost > LightGBM > sklearn GradientBoosting
    depending on what is installed.

    Notes
    -----
    - XGBoost and LightGBM handle NaN natively; sklearn does not.
    - Feature importances are available after training.
    - Does not require feature scaling.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        use_gpu: bool = False,
    ) -> None:
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate
        self._max_depth = max_depth
        self._subsample = subsample
        self._colsample_bytree = colsample_bytree
        # Auto-detect NVIDIA GPU (overrides the config use_gpu setting)
        try:
            from models.gpu_utils import get_xgb_device as _xgb_dev
            self._use_gpu = _xgb_dev() == "cuda"
        except Exception:
            self._use_gpu = use_gpu
        self._model = None
        self._feature_names: list[str] = []
        self._fitted = False
        self._backend = _BACKEND

    @property
    def name(self) -> str:
        return f"GradientBoosting({self._backend})"

    def fit(self, X: pd.DataFrame, y: pd.Series, incremental: bool = False) -> None:
        """
        Train the gradient boosting model.

        Parameters
        ----------
        X : pd.DataFrame
        y : pd.Series
        incremental : bool
            If True AND the model has already been fitted, add ``n_estimators``
            more trees to the existing ensemble via warm-start instead of
            rebuilding from scratch.  If no prior model exists the flag is
            silently ignored and a fresh model is trained.
        """
        self._feature_names = list(X.columns)

        # ── Incremental (warm-start): add more trees to existing ensemble ──
        if incremental and self._fitted and self._model is not None:
            try:
                current = int(self._model.n_estimators)
                new_total = current + self._n_estimators
                self._model.set_params(warm_start=True, n_estimators=new_total)
                logger.info(
                    "Incremental %s: adding %d trees to existing %d (total: %d).",
                    self.name, self._n_estimators, current, new_total,
                )
                self._model.fit(X, y)
                self._fitted = True
                logger.info("%s incremental training complete.", self.name)
                return
            except Exception as exc:
                logger.warning(
                    "Incremental warm-start failed (%s) — falling back to fresh training.", exc
                )

        # ── Fresh model ────────────────────────────────────────────────────
        logger.info("Training %s on %d samples, %d features.", self.name, len(X), X.shape[1])

        if self._backend == "xgboost":
            device = "cuda" if self._use_gpu else "cpu"
            self._model = XGBClassifier(
                n_estimators=self._n_estimators,
                learning_rate=self._learning_rate,
                max_depth=self._max_depth,
                subsample=self._subsample,
                colsample_bytree=self._colsample_bytree,
                device=device,
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42,
                verbosity=0,
            )

        elif self._backend == "lightgbm":
            device = "gpu" if self._use_gpu else "cpu"
            self._model = LGBMClassifier(
                n_estimators=self._n_estimators,
                learning_rate=self._learning_rate,
                max_depth=self._max_depth,
                subsample=self._subsample,
                colsample_bytree=self._colsample_bytree,
                device=device,
                random_state=42,
                verbose=-1,
            )

        else:  # sklearn fallback
            from sklearn.ensemble import GradientBoostingClassifier
            self._model = GradientBoostingClassifier(
                n_estimators=self._n_estimators,
                learning_rate=self._learning_rate,
                max_depth=self._max_depth,
                subsample=self._subsample,
                random_state=42,
            )

        self._model.fit(X, y)
        self._fitted = True
        logger.info("%s training complete.", self.name)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities."""
        if not self._fitted or self._model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self._model.predict_proba(X)

    @property
    def feature_importances(self) -> pd.Series | None:
        """Return feature importances as a sorted Series (if fitted)."""
        if not self._fitted or self._model is None:
            return None
        importances = getattr(self._model, "feature_importances_", None)
        if importances is None:
            return None
        return pd.Series(
            importances,
            index=self._feature_names,
        ).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        """Save the fitted model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self._model,
            "feature_names": self._feature_names,
            "backend": self._backend,
        }, path)
        logger.info("Saved %s to %s.", self.name, path)

    def load(self, path: Path) -> None:
        """Load a fitted model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        data = joblib.load(path)
        self._model = data["model"]
        self._feature_names = data.get("feature_names", [])
        self._backend = data.get("backend", _BACKEND)
        self._fitted = True
        logger.info("Loaded %s from %s.", self.name, path)
