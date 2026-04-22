"""
forecasting/evaluator.py
Evaluates past forecasts against actual prices and produces a
correction dataset the model can learn from.

Workflow
--------
1. Load all forecast JSONs where the horizon has elapsed.
2. Fetch actual prices for the forecast period.
3. Compare:
   - Predicted direction vs. actual direction (correct / wrong)
   - Mean simulated path vs. actual price path (RMSE, MAE)
   - Per-step accuracy (how many days the model got right)
4. Flag "mistake" periods — days where the model was confidently
   wrong (confidence ≥ 0.60, actual direction opposite).
5. Optionally retrain the model on the actual data from the mistake
   periods using incremental training.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from models.base import BaseModel

from forecasting.simulator import load_forecast, list_past_forecasts, _FORECAST_DIR

logger = logging.getLogger(__name__)


@dataclass
class ForecastEvaluation:
    """Evaluation of one past forecast against actual prices."""
    ticker:               str
    forecast_date:        date
    horizon_label:        str
    horizon_days:         int

    # Overall
    predicted_direction:  str           # "UP" / "DOWN" / "UNCERTAIN"
    actual_direction:     str           # derived from start→end actual price
    direction_correct:    bool

    # Price comparison
    start_price:          float
    actual_end_price:     float
    simulated_end_price:  float         # mean path end
    actual_return_pct:    float
    simulated_return_pct: float
    price_mae:            float         # mean abs error between mean path and actual
    price_rmse:           float

    # Per-step
    step_accuracies:      list[bool]    = field(default_factory=list)
    step_accuracy_pct:    float         = 0.0

    # Mistake info
    mistake_steps:        list[int]     = field(default_factory=list)  # step indices (1-based)
    confident_mistakes:   int           = 0  # steps where conf≥0.60 AND direction wrong

    def summary_line(self) -> str:
        icon = "✅" if self.direction_correct else "❌"
        return (
            f"{icon} {self.ticker:<10}  [{self.horizon_label}]  "
            f"predicted={self.predicted_direction:<9}  "
            f"actual={self.actual_direction:<9}  "
            f"pred_ret={self.simulated_return_pct:>+6.2f}%  "
            f"actual_ret={self.actual_return_pct:>+6.2f}%  "
            f"step_acc={self.step_accuracy_pct:.0f}%  "
            f"conf_mistakes={self.confident_mistakes}"
        )


class ForecastEvaluator:
    """
    Loads past forecasts, compares them to actual data, and generates
    a correction dataset for incremental retraining.
    """

    def __init__(self, data_source, feature_pipeline) -> None:
        """
        Parameters
        ----------
        data_source : DataProvider
            Used to fetch actual historical prices.
        feature_pipeline : FeaturePipeline
            Used to build features for correction retraining.
        """
        self._data    = data_source
        self._pipeline = feature_pipeline

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate_all(
        self,
        horizon_elapsed_only: bool = True,
    ) -> list[ForecastEvaluation]:
        """
        Find all saved forecasts (optionally only those whose horizon has
        passed), fetch actual prices, and return evaluation results.
        """
        paths = list_past_forecasts(horizon_elapsed=horizon_elapsed_only)
        if not paths:
            logger.info("No past forecasts ready for evaluation.")
            return []

        results: list[ForecastEvaluation] = []
        for path in paths:
            try:
                ev = self._evaluate_one(path)
                if ev is not None:
                    results.append(ev)
            except Exception as exc:
                logger.warning("Could not evaluate %s: %s", path.name, exc)

        return results

    def print_report(self, evaluations: list[ForecastEvaluation]) -> None:
        """Print a formatted evaluation report to stdout."""
        if not evaluations:
            print("\n  No completed forecasts to evaluate yet.\n")
            return

        correct  = sum(1 for e in evaluations if e.direction_correct)
        total    = len(evaluations)
        avg_mae  = np.mean([e.price_mae  for e in evaluations])
        avg_rmse = np.mean([e.price_rmse for e in evaluations])

        line = "─" * 80
        print(f"\n{line}")
        print(f"  📊  FORECAST EVALUATION REPORT  —  {total} forecast(s) reviewed")
        print(line)
        print(f"  Direction accuracy : {correct}/{total}  ({correct/total*100:.0f}%)")
        print(f"  Avg price MAE      : {avg_mae:.4f}")
        print(f"  Avg price RMSE     : {avg_rmse:.4f}")
        print(line)
        for ev in sorted(evaluations, key=lambda e: e.forecast_date):
            print(f"  {ev.summary_line()}")
        print(line + "\n")

    def build_correction_dataset(
        self,
        evaluations: list[ForecastEvaluation],
    ) -> tuple[pd.DataFrame, pd.Series] | None:
        """
        For each evaluation with confident mistakes, fetch the actual OHLCV
        data for that period, run the feature pipeline, and return (X, y)
        that can be used for incremental retraining.

        Returns None if no correction data is available.
        """
        all_X: list[pd.DataFrame] = []
        all_y: list[pd.Series]    = []

        for ev in evaluations:
            if ev.confident_mistakes == 0:
                continue
            try:
                x_chunk, y_chunk = self._fetch_correction_chunk(ev)
                if x_chunk is not None and not x_chunk.empty:
                    all_X.append(x_chunk)
                    all_y.append(y_chunk)
                    logger.info(
                        "Correction data for %s [%s]: %d rows.",
                        ev.ticker, ev.horizon_label, len(x_chunk),
                    )
            except Exception as exc:
                logger.warning("Could not build correction chunk for %s: %s", ev.ticker, exc)

        if not all_X:
            logger.info("No correction data — model was right on all confident predictions.")
            return None

        X = pd.concat(all_X, ignore_index=True)
        y = pd.concat(all_y, ignore_index=True)
        logger.info(
            "Total correction dataset: %d rows across %d ticker(s).",
            len(X), len(all_X),
        )
        return X, y

    # ── Internals ─────────────────────────────────────────────────────────────

    def _evaluate_one(self, path: Path) -> ForecastEvaluation | None:
        """Evaluate a single forecast JSON against actual prices."""
        data         = load_forecast(path)
        ticker       = data["ticker"]
        fc_date      = date.fromisoformat(data["forecast_date"])
        horizon_days = int(data["horizon_days"])
        horizon_lbl  = data["horizon_label"]
        start_price  = float(data["start_price"])
        mean_path    = data["simulation"]["mean_path"]
        net_direction = data["summary"]["net_direction"]
        sim_return   = float(data["summary"]["predicted_return_pct"])
        step_preds   = data["step_predictions"]

        # Fetch actual prices (use a period wide enough to cover the horizon)
        fetch_days = horizon_days + 30
        df = self._data.fetch(ticker, period=f"{max(fetch_days // 21, 1)}mo")
        if df is None or df.empty:
            logger.warning("No actual data for %s — skipping.", ticker)
            return None

        # Slice actual prices from forecast date forward
        df.index = pd.DatetimeIndex(df.index).normalize()
        fc_ts    = pd.Timestamp(fc_date)
        actual   = df.loc[df.index >= fc_ts, "close"]
        if len(actual) < horizon_days:
            logger.info(
                "%s [%s]: only %d/%d actual days available — skipping.",
                ticker, horizon_lbl, len(actual), horizon_days,
            )
            return None

        actual = actual.iloc[:horizon_days + 1]  # day-0 (start) + horizon_days

        actual_start = float(actual.iloc[0])
        actual_end   = float(actual.iloc[-1])
        actual_ret   = (actual_end - actual_start) / actual_start * 100.0
        actual_dir   = "UP" if actual_ret > 0.5 else ("DOWN" if actual_ret < -0.5 else "UNCERTAIN")

        # Price path comparison
        sim_prices = mean_path[:len(actual)]
        actual_arr = actual.values[:len(sim_prices)]
        mae  = float(np.mean(np.abs(np.array(sim_prices) - actual_arr)))
        rmse = float(np.sqrt(np.mean((np.array(sim_prices) - actual_arr) ** 2)))

        # Per-step accuracy (day-over-day direction)
        step_accs: list[bool] = []
        mistake_steps: list[int] = []
        confident_mistakes = 0

        for sp in step_preds:
            idx = sp["step"]
            if idx >= len(actual):
                break
            actual_day_ret = (actual.iloc[idx] - actual.iloc[idx - 1]) / actual.iloc[idx - 1]
            actual_day_dir = "UP" if actual_day_ret > 0 else "DOWN"
            predicted_dir  = sp["direction"]
            correct = (predicted_dir == actual_day_dir) or predicted_dir == "UNCERTAIN"
            step_accs.append(correct)
            if not correct:
                mistake_steps.append(idx)
                if sp["confidence"] >= 0.60:
                    confident_mistakes += 1

        step_acc_pct = (sum(step_accs) / len(step_accs) * 100.0) if step_accs else 0.0

        return ForecastEvaluation(
            ticker               = ticker,
            forecast_date        = fc_date,
            horizon_label        = horizon_lbl,
            horizon_days         = horizon_days,
            predicted_direction  = net_direction,
            actual_direction     = actual_dir,
            direction_correct    = (net_direction == actual_dir) or net_direction == "UNCERTAIN",
            start_price          = actual_start,
            actual_end_price     = actual_end,
            simulated_end_price  = float(mean_path[min(horizon_days, len(mean_path) - 1)]),
            actual_return_pct    = actual_ret,
            simulated_return_pct = sim_return,
            price_mae            = mae,
            price_rmse           = rmse,
            step_accuracies      = step_accs,
            step_accuracy_pct    = step_acc_pct,
            mistake_steps        = mistake_steps,
            confident_mistakes   = confident_mistakes,
        )

    def _fetch_correction_chunk(
        self,
        ev: ForecastEvaluation,
    ) -> tuple[pd.DataFrame, pd.Series] | None:
        """
        Fetch actual OHLCV for the evaluation period, run the feature
        pipeline, and return (X, y) where y is the actual next-day direction.
        """
        fetch_period = f"{max(ev.horizon_days // 21 + 2, 3)}mo"
        df = self._data.fetch(ev.ticker, period=fetch_period)
        if df is None or df.empty:
            return None

        df_feat = self._pipeline.transform(df, include_target=True).dropna()
        if df_feat.empty:
            return None

        # Keep only rows within the forecast horizon window
        fc_ts    = pd.Timestamp(ev.forecast_date)
        end_ts   = fc_ts + timedelta(days=int(ev.horizon_days * 1.5))
        df_feat  = df_feat[(df_feat.index >= fc_ts) & (df_feat.index <= end_ts)]
        if df_feat.empty:
            return None

        feature_cols = self._pipeline.feature_names
        X = df_feat[feature_cols]
        y = df_feat["target"].astype(int)
        return X, y
