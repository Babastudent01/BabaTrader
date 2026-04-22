"""
tests/test_models.py
Unit tests for ML models (logistic, random forest) and ModelEvaluator.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.logistic import LogisticModel
from models.random_forest import RandomForestModel
from models.evaluator import ModelEvaluator


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_dataset():
    """Generate a simple linearly separable dataset."""
    np.random.seed(0)
    n = 300
    X = pd.DataFrame({
        "feat_a": np.random.randn(n),
        "feat_b": np.random.randn(n),
        "feat_c": np.random.randn(n),
    })
    # Target: 1 if feat_a + feat_b > 0, else 0
    y = pd.Series((X["feat_a"] + X["feat_b"] > 0).astype(int))
    return X, y


@pytest.fixture
def train_test_split(synthetic_dataset):
    X, y = synthetic_dataset
    split = int(len(X) * 0.8)
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


# ── LogisticModel tests ───────────────────────────────────────────────────────

class TestLogisticModel:

    def test_fit_sets_is_fitted(self, train_test_split):
        X_train, y_train, _, _ = train_test_split
        model = LogisticModel()
        assert not model.is_fitted
        model.fit(X_train, y_train)
        assert model.is_fitted

    def test_predict_returns_binary(self, train_test_split):
        X_train, y_train, X_test, _ = train_test_split
        model = LogisticModel()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, train_test_split):
        X_train, y_train, X_test, _ = train_test_split
        model = LogisticModel()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        # Probabilities must sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_bounded(self, train_test_split):
        X_train, y_train, X_test, _ = train_test_split
        model = LogisticModel()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_not_fitted_raises(self, synthetic_dataset):
        X, _ = synthetic_dataset
        model = LogisticModel()
        with pytest.raises(RuntimeError):
            model.predict_proba(X)

    def test_save_and_load(self, train_test_split):
        X_train, y_train, X_test, _ = train_test_split
        model = LogisticModel()
        model.fit(X_train, y_train)
        proba_before = model.predict_proba(X_test)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "logistic.pkl"
            model.save(path)

            model2 = LogisticModel()
            model2.load(path)
            proba_after = model2.predict_proba(X_test)

        np.testing.assert_allclose(proba_before, proba_after, atol=1e-6)

    def test_accuracy_above_chance(self, train_test_split):
        """Logistic regression should beat random chance on linearly separable data."""
        X_train, y_train, X_test, y_test = train_test_split
        model = LogisticModel()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = (preds == y_test.values).mean()
        assert accuracy > 0.7, f"Expected accuracy > 0.7, got {accuracy:.3f}"


# ── RandomForestModel tests ───────────────────────────────────────────────────

class TestRandomForestModel:

    def test_fit_and_predict(self, train_test_split):
        X_train, y_train, X_test, _ = train_test_split
        model = RandomForestModel(n_estimators=20)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_feature_importances_is_series(self, train_test_split):
        X_train, y_train, _, _ = train_test_split
        model = RandomForestModel(n_estimators=20)
        model.fit(X_train, y_train)
        importances = model.feature_importances
        assert importances is not None
        assert isinstance(importances, pd.Series)
        assert len(importances) == X_train.shape[1]
        assert abs(importances.sum() - 1.0) < 1e-6

    def test_feature_importances_none_before_fit(self):
        model = RandomForestModel()
        assert model.feature_importances is None

    def test_save_and_load(self, train_test_split):
        X_train, y_train, X_test, _ = train_test_split
        model = RandomForestModel(n_estimators=10)
        model.fit(X_train, y_train)
        proba_before = model.predict_proba(X_test)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rf.pkl"
            model.save(path)
            model2 = RandomForestModel()
            model2.load(path)
            proba_after = model2.predict_proba(X_test)

        np.testing.assert_allclose(proba_before, proba_after, atol=1e-6)


# ── ModelEvaluator tests ──────────────────────────────────────────────────────

class TestModelEvaluator:

    def test_perfect_predictions(self):
        evaluator = ModelEvaluator()
        y_true = pd.Series([1, 0, 1, 0, 1])
        y_pred = pd.Series([1, 0, 1, 0, 1])
        y_proba = pd.Series([0.9, 0.1, 0.8, 0.2, 0.95])
        metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["roc_auc"] == 1.0

    def test_all_wrong_predictions(self):
        evaluator = ModelEvaluator()
        y_true = pd.Series([1, 1, 1, 0, 0])
        y_pred = pd.Series([0, 0, 0, 1, 1])
        metrics = evaluator.compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0

    def test_trading_metrics_with_returns(self):
        evaluator = ModelEvaluator()
        y_true  = pd.Series([1, 0, 1, 1, 0])
        y_pred  = pd.Series([1, 0, 1, 1, 0])
        returns = pd.Series([0.02, -0.01, 0.03, -0.005, 0.01])
        metrics = evaluator.compute_metrics(y_true, y_pred, price_returns=returns)
        assert "strategy_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "win_rate" in metrics

    def test_metrics_keys_present(self):
        evaluator = ModelEvaluator()
        y_true = pd.Series([1, 0, 1])
        y_pred = pd.Series([1, 1, 0])
        metrics = evaluator.compute_metrics(y_true, y_pred)
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert key in metrics

    def test_roc_auc_without_proba_is_zero(self):
        evaluator = ModelEvaluator()
        y_true = pd.Series([1, 0, 1])
        y_pred = pd.Series([1, 0, 0])
        metrics = evaluator.compute_metrics(y_true, y_pred)
        # roc_auc defaults to 0.0 when no proba is provided
        assert metrics["roc_auc"] == 0.0
