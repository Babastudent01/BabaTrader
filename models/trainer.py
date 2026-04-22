"""
models/trainer.py
Walk-forward cross-validation trainer.
Trains and evaluates models using time-series-safe splits.
Prevents data leakage by never using future data in training.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from config import Settings
from models.base import BaseModel
from models.evaluator import ModelEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""
    fold: int
    train_size: int
    test_size: int
    metrics: dict[str, float]
    predictions: pd.Series
    probabilities: pd.Series


@dataclass
class TrainingResult:
    """Aggregated results from walk-forward training."""
    model_name: str
    n_folds: int
    fold_results: list[FoldResult]
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]
    feature_names: list[str]


class ModelTrainer:
    """
    Trains ML models using walk-forward (time-series) cross-validation.

    Walk-forward validation:
    - Splits data into N folds chronologically.
    - Each fold: train on past data, test on future data.
    - A gap of `gap` bars is left between train and test to avoid leakage.
    - The final model is retrained on ALL available data.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings          # keep full settings for DeepModel clone
        self._cfg = settings.model
        self._wf_cfg = settings.model.walk_forward
        self._evaluator = ModelEvaluator()

    def walk_forward_validate(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        budget_seconds: float | None = None,
        groups: "np.ndarray | None" = None,
    ) -> TrainingResult:
        """
        Run walk-forward cross-validation.

        Parameters
        ----------
        model : BaseModel
            Unfitted model instance.
        X : pd.DataFrame
            Feature matrix (time-ordered, no future leakage).
        y : pd.Series
            Binary target series.
        budget_seconds : float | None
            Total time budget in seconds for **all WF folds combined**
            (30 % of the total ``--train-time`` budget).
            Divided evenly across folds and forwarded to the model's ``fit()``.

        Returns
        -------
        TrainingResult
            Aggregated metrics across all folds.
        """
        n_splits = int(self._wf_cfg.get("n_splits", 5))
        gap = int(self._wf_cfg.get("gap", 1))

        # Compute per-fold time budget (WF folds share 30 % of total budget)
        wf_budget = budget_seconds * 0.30 if budget_seconds is not None else None
        per_fold_budget = (wf_budget / n_splits) if wf_budget is not None else None
        if per_fold_budget is not None:
            logger.info(
                "WF budget: %.0f s total, %.0f s per fold.",
                wf_budget, per_fold_budget,
            )

        logger.info(
            "Starting walk-forward validation: %d folds, gap=%d bars, model=%s.",
            n_splits, gap, model.name,
        )

        import time as _t
        print(f"\n  📐  Walk-forward validation ({n_splits} folds)…")
        print(f"  {'Fold':<6} {'Train':>7} {'Test':>6}  {'ROC-AUC':>8}  {'Acc':>7}  {'F1':>7}  {'Time':>6}")
        print(f"  {'─'*6} {'─'*7} {'─'*6}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*6}")

        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        fold_results: list[FoldResult] = []

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test  = X.iloc[test_idx]
            y_test  = y.iloc[test_idx]

            logger.info(
                "Fold %d/%d — train: %d bars, test: %d bars.",
                fold_idx, n_splits, len(X_train), len(X_test),
            )

            print(f"  {fold_idx}/{n_splits}     ", end="", flush=True)
            fold_t0 = _t.monotonic()

            # Create a fresh model instance for each fold
            fold_model = self._clone_model(model)
            # Pass time budget + ticker groups if the model supports them
            import inspect
            fit_sig = inspect.signature(fold_model.fit).parameters
            fold_kwargs: dict = {}
            if per_fold_budget is not None and "max_seconds" in fit_sig:
                fold_kwargs["max_seconds"] = per_fold_budget
            if groups is not None and "groups" in fit_sig:
                fold_kwargs["groups"] = groups[train_idx]
            fold_model.fit(X_train, y_train, **fold_kwargs)

            fold_elapsed = _t.monotonic() - fold_t0
            proba = fold_model.predict_proba(X_test)[:, 1]
            preds = (proba >= 0.5).astype(int)

            metrics = self._evaluator.compute_metrics(
                y_true=y_test,
                y_pred=pd.Series(preds, index=y_test.index),
                y_proba=pd.Series(proba, index=y_test.index),
            )

            fold_results.append(FoldResult(
                fold=fold_idx,
                train_size=len(X_train),
                test_size=len(X_test),
                metrics=metrics,
                predictions=pd.Series(preds, index=y_test.index),
                probabilities=pd.Series(proba, index=y_test.index),
            ))

            roc = metrics.get("roc_auc", 0)
            acc = metrics.get("accuracy", 0)
            f1  = metrics.get("f1", 0)
            print(
                f"\r  {fold_idx}/{n_splits}  ✓  "
                f"{len(X_train):>7,}  {len(X_test):>6,}  "
                f"{roc:>8.4f}  {acc:>7.3f}  {f1:>7.3f}  "
                f"{fold_elapsed:>4.0f}s"
            )
            logger.info(
                "Fold %d — Accuracy: %.3f, F1: %.3f, ROC-AUC: %.3f",
                fold_idx, acc, f1, roc,
            )

        # Aggregate metrics across folds
        all_metric_keys = fold_results[0].metrics.keys()
        mean_metrics = {
            k: float(np.mean([f.metrics[k] for f in fold_results]))
            for k in all_metric_keys
        }
        std_metrics = {
            k: float(np.std([f.metrics[k] for f in fold_results]))
            for k in all_metric_keys
        }

        print(
            f"\n  Mean  —  ROC-AUC: {mean_metrics.get('roc_auc', 0):.4f} "
            f"± {std_metrics.get('roc_auc', 0):.4f}  |  "
            f"Acc: {mean_metrics.get('accuracy', 0):.3f}  |  "
            f"F1: {mean_metrics.get('f1', 0):.3f}\n"
        )
        logger.info(
            "Walk-forward complete — Mean Accuracy: %.3f ± %.3f, "
            "Mean F1: %.3f ± %.3f, Mean ROC-AUC: %.3f ± %.3f",
            mean_metrics.get("accuracy", 0), std_metrics.get("accuracy", 0),
            mean_metrics.get("f1", 0), std_metrics.get("f1", 0),
            mean_metrics.get("roc_auc", 0), std_metrics.get("roc_auc", 0),
        )

        return TrainingResult(
            model_name=model.name,
            n_folds=n_splits,
            fold_results=fold_results,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            feature_names=list(X.columns),
        )

    def train_final(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        save_path: Path | None = None,
        budget_seconds: float | None = None,
        incremental: bool = False,
        groups: "np.ndarray | None" = None,
    ) -> BaseModel:
        """
        Train the final model on ALL available data and optionally save it.

        When ``budget_seconds`` is given:
        * **DeepModel**   — passes ``max_seconds = budget * 0.70`` to ``fit()``.
        * **ML models**   — runs a timed random hyperparameter search for the
                            full 70 % of the budget, then retrains the best
                            configuration on the full dataset.

        Parameters
        ----------
        model : BaseModel
            Unfitted model instance.
        X : pd.DataFrame
            Full feature matrix.
        y : pd.Series
            Full target series.
        save_path : Path | None
            If provided, save the trained model to this path.
        budget_seconds : float | None
            Total time budget (seconds) shared with walk_forward_validate.
            Final training uses the remaining 70 %.

        Returns
        -------
        BaseModel
            Fitted model.
        """
        final_budget = budget_seconds * 0.70 if budget_seconds is not None else None

        import inspect
        model_name = type(model).__name__

        if final_budget is not None and "max_seconds" in inspect.signature(model.fit).parameters:
            # Deep learning model — time-budget epoch training
            logger.info(
                "Training final %s on full dataset (%d samples, budget=%.0f s).",
                model.name, len(X), final_budget,
            )
            budget_str = f", budget={final_budget/60:.1f} min" if final_budget else ""
            print(f"  🏋  Training final {model.name} on {len(X):,} samples{budget_str}…")
            fit_kw: dict = {"max_seconds": final_budget, "incremental": incremental}
            if groups is not None and "groups" in inspect.signature(model.fit).parameters:
                fit_kw["groups"] = groups
            model.fit(X, y, **fit_kw)

        elif final_budget is not None and model_name in ("GradientBoostingModel", "RandomForestModel", "LogisticModel"):
            print(f"  🏋  Hyperparameter search on {len(X):,} samples (budget={final_budget/60:.1f} min)…")
            # ML model — timed random hyperparameter search
            # When incremental=True, the loaded model's current score is used as
            # the baseline: any new config must beat it to replace it.
            model = self._timed_hyperparam_search(model, X, y, final_budget,
                                                   incremental=incremental)

        else:
            logger.info("Training final %s on full dataset (%d samples).", model.name, len(X))
            print(f"  🏋  Training final {model.name} on {len(X):,} samples…")
            fit_kwargs: dict = {}
            if incremental and "incremental" in inspect.signature(model.fit).parameters:
                fit_kwargs["incremental"] = True
            if groups is not None and "groups" in inspect.signature(model.fit).parameters:
                fit_kwargs["groups"] = groups
            model.fit(X, y, **fit_kwargs)

        if save_path is not None:
            model.save(save_path)
            logger.info("Final model saved to %s.", save_path)

        return model

    # ── Timed hyperparameter search for ML models ─────────────────────────────

    def _timed_hyperparam_search(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        budget_seconds: float,
        incremental: bool = False,
    ) -> BaseModel:
        """
        Run a random hyperparameter search within the given time budget.

        Each candidate is quickly evaluated on a single held-out 20% validation
        split (time-ordered), and the best configuration is retrained on the
        full dataset at the end.

        Supports: GradientBoostingModel, RandomForestModel, LogisticModel.
        Returns the best fitted model.
        """
        from sklearn.model_selection import TimeSeriesSplit
        import inspect

        model_name = type(model).__name__
        deadline   = time.monotonic() + budget_seconds

        logger.info(
            "Timed hyperparameter search for %s — budget=%.0f s (%.1f min).",
            model.name, budget_seconds, budget_seconds / 60,
        )

        # ── Parameter grids ───────────────────────────────────────────────
        def _random_gb_params() -> dict:
            return {
                "n_estimators":    random.choice([100, 200, 300, 500, 700, 1000]),
                "learning_rate":   random.choice([0.005, 0.01, 0.02, 0.05, 0.1]),
                "max_depth":       random.choice([3, 4, 5, 6]),
                "subsample":       random.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
                "colsample_bytree": random.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
            }

        def _random_rf_params() -> dict:
            return {
                "n_estimators":      random.choice([100, 200, 300, 500]),
                "max_depth":         random.choice([4, 6, 8, 10, None]),
                "min_samples_leaf":  random.choice([5, 10, 20, 30]),
            }

        def _random_lr_params() -> dict:
            return {
                "C":       random.choice([0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]),
                "max_iter": 2000,
            }

        param_fn = {
            "GradientBoostingModel": _random_gb_params,
            "RandomForestModel":     _random_rf_params,
            "LogisticModel":         _random_lr_params,
        }.get(model_name)

        if param_fn is None:
            logger.warning("No hyperparameter grid for %s — training with defaults.", model_name)
            model.fit(X, y)
            return model

        # Quick train/val split (last 20% for scoring)
        n_val   = max(1, int(len(X) * 0.20))
        n_train = len(X) - n_val
        X_tr, y_tr = X.iloc[:n_train], y.iloc[:n_train]
        X_val, y_val = X.iloc[n_train:], y.iloc[n_train:]

        # If incremental, score the already-loaded model as the initial best
        # so the search never regresses from the previous run.
        best_roc   = -1.0
        best_params: dict = {}
        n_tried    = 0

        if incremental and model._fitted:
            try:
                proba_existing = model.predict_proba(X_val)[:, 1]
                from sklearn.metrics import roc_auc_score
                best_roc = float(roc_auc_score(y_val, proba_existing))
                logger.info(
                    "  Incremental baseline: existing model ROC-AUC=%.4f "
                    "(any new config must beat this).",
                    best_roc,
                )
            except Exception as exc:
                logger.warning("Could not score existing model as baseline: %s", exc)

        while time.monotonic() < deadline:
            params = param_fn()
            try:
                candidate = self._build_model_with_params(model_name, params)
                candidate.fit(X_tr, y_tr)
                proba  = candidate.predict_proba(X_val)[:, 1]
                from sklearn.metrics import roc_auc_score
                roc = float(roc_auc_score(y_val, proba))
            except Exception as exc:
                logger.debug("Candidate failed (%s) — skipping.", exc)
                continue

            n_tried += 1
            elapsed = time.monotonic() - (deadline - budget_seconds)
            if roc > best_roc:
                best_roc    = roc
                best_params = params
                logger.info(
                    "  [%.0f s] New best: ROC-AUC=%.4f — %s",
                    elapsed, best_roc, params,
                )

        logger.info(
            "Hyperparameter search done: %d configs tried, best ROC-AUC=%.4f, params=%s",
            n_tried, best_roc, best_params,
        )

        # Retrain best config on full data (or keep existing model if nothing beat it)
        if best_params:
            final_model = self._build_model_with_params(model_name, best_params)
            logger.info("Retraining best config on full dataset (%d samples).", len(X))
            final_model.fit(X, y)
        elif incremental and model._fitted:
            # Nothing beat the existing model — just warm-start it with more trees
            logger.info(
                "No config beat the existing model — keeping it and adding more trees "
                "(incremental warm-start on full dataset)."
            )
            final_model = model
            if "incremental" in inspect.signature(final_model.fit).parameters:
                final_model.fit(X, y, incremental=True)
            else:
                final_model.fit(X, y)
        else:
            final_model = self._clone_model(model)
            logger.info("Retraining default config on full dataset (%d samples).", len(X))
            final_model.fit(X, y)
        return final_model

    def _build_model_with_params(self, model_name: str, params: dict) -> BaseModel:
        """Instantiate a model with given params (used in hyperparameter search)."""
        if model_name == "GradientBoostingModel":
            from models.gradient_boosting import GradientBoostingModel
            return GradientBoostingModel(
                n_estimators=params.get("n_estimators", 300),
                learning_rate=params.get("learning_rate", 0.05),
                max_depth=params.get("max_depth", 4),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
            )
        elif model_name == "RandomForestModel":
            from models.random_forest import RandomForestModel
            return RandomForestModel(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 6),
                min_samples_leaf=params.get("min_samples_leaf", 20),
            )
        elif model_name == "LogisticModel":
            from models.logistic import LogisticModel
            return LogisticModel(
                C=params.get("C", 1.0),
                max_iter=params.get("max_iter", 1000),
            )
        else:
            return self._clone_model(self._settings.model)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _clone_model(self, model: BaseModel) -> BaseModel:
        """Create a fresh (unfitted) copy of the model with the same hyperparameters."""
        # Re-instantiate using the same class and config
        model_type = type(model)
        cfg = self._cfg

        if model_type.__name__ == "LogisticModel":
            from models.logistic import LogisticModel
            lr_cfg = cfg.get("logistic", {}) or {}
            return LogisticModel(
                C=float(lr_cfg.get("C", 1.0) if isinstance(lr_cfg, dict) else 1.0),
                max_iter=int(lr_cfg.get("max_iter", 1000) if isinstance(lr_cfg, dict) else 1000),
                class_weight=str(lr_cfg.get("class_weight", "balanced") if isinstance(lr_cfg, dict) else "balanced"),
            )

        elif model_type.__name__ == "RandomForestModel":
            from models.random_forest import RandomForestModel
            rf_cfg = cfg.get("random_forest", {}) or {}
            return RandomForestModel(
                n_estimators=int(rf_cfg.get("n_estimators", 200) if isinstance(rf_cfg, dict) else 200),
                max_depth=int(rf_cfg.get("max_depth", 6) if isinstance(rf_cfg, dict) else 6),
                min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 20) if isinstance(rf_cfg, dict) else 20),
            )

        elif model_type.__name__ == "GradientBoostingModel":
            from models.gradient_boosting import GradientBoostingModel
            gb_cfg = cfg.get("gradient_boosting", {}) or {}
            return GradientBoostingModel(
                n_estimators=int(gb_cfg.get("n_estimators", 300) if isinstance(gb_cfg, dict) else 300),
                learning_rate=float(gb_cfg.get("learning_rate", 0.05) if isinstance(gb_cfg, dict) else 0.05),
                max_depth=int(gb_cfg.get("max_depth", 4) if isinstance(gb_cfg, dict) else 4),
                subsample=float(gb_cfg.get("subsample", 0.8) if isinstance(gb_cfg, dict) else 0.8),
                colsample_bytree=float(gb_cfg.get("colsample_bytree", 0.8) if isinstance(gb_cfg, dict) else 0.8),
            )

        elif model_type.__name__ == "DeepModel":
            from models.deep_learning import DeepModel
            return DeepModel(self._settings)

        else:
            raise ValueError(f"Cannot clone unknown model type: {model_type.__name__}")


def create_model(settings: Settings) -> BaseModel:
    """
    Factory function: create the model specified in settings.

    Parameters
    ----------
    settings : Settings
        Application settings.

    Returns
    -------
    BaseModel
        Unfitted model instance.
    """
    model_type = settings.model.get("type", "gradient_boosting").lower()
    cfg = settings.model

    if model_type == "logistic":
        from models.logistic import LogisticModel
        lr_cfg = cfg.get("logistic", {}) or {}
        return LogisticModel(
            C=float(lr_cfg.get("C", 1.0) if isinstance(lr_cfg, dict) else 1.0),
            max_iter=int(lr_cfg.get("max_iter", 1000) if isinstance(lr_cfg, dict) else 1000),
            class_weight=str(lr_cfg.get("class_weight", "balanced") if isinstance(lr_cfg, dict) else "balanced"),
        )

    elif model_type == "random_forest":
        from models.random_forest import RandomForestModel
        rf_cfg = cfg.get("random_forest", {}) or {}
        return RandomForestModel(
            n_estimators=int(rf_cfg.get("n_estimators", 200) if isinstance(rf_cfg, dict) else 200),
            max_depth=int(rf_cfg.get("max_depth", 6) if isinstance(rf_cfg, dict) else 6),
            min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 20) if isinstance(rf_cfg, dict) else 20),
        )

    elif model_type == "gradient_boosting":
        from models.gradient_boosting import GradientBoostingModel
        gb_cfg = cfg.get("gradient_boosting", {}) or {}
        return GradientBoostingModel(
            n_estimators=int(gb_cfg.get("n_estimators", 300) if isinstance(gb_cfg, dict) else 300),
            learning_rate=float(gb_cfg.get("learning_rate", 0.05) if isinstance(gb_cfg, dict) else 0.05),
            max_depth=int(gb_cfg.get("max_depth", 4) if isinstance(gb_cfg, dict) else 4),
            subsample=float(gb_cfg.get("subsample", 0.8) if isinstance(gb_cfg, dict) else 0.8),
            colsample_bytree=float(gb_cfg.get("colsample_bytree", 0.8) if isinstance(gb_cfg, dict) else 0.8),
        )

    else:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            "Supported: 'logistic', 'random_forest', 'gradient_boosting'."
        )
