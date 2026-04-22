"""
forecasting/simulator.py
Price simulation engine for the forecast mode.

Strategy
--------
1. Run the model on the latest available feature bar to get today's
   P(up) / P(down) signal.
2. Use the signal confidence as a drift coefficient in a Geometric
   Brownian Motion (GBM) simulation:
       drift  = direction × confidence × σ_daily
       return ~ N(drift, σ_daily)   per step
3. Roll 200 Monte Carlo paths forward for the requested horizon.
4. Return the mean path, 80% and 95% confidence bands, and the
   per-step model signals so the chart can annotate each day.

Why GBM + model drift?
- The model outputs P(up/down) for the next bar, not an exact price.
- GBM is the standard financial random-walk model.
- Using the model's confidence as a drift multiplier keeps the
  simulation honest: high-confidence signals produce a more
  directional mean path; low-confidence ones look like a flat walk.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from models.base import BaseModel

logger = logging.getLogger(__name__)

# How many business days in each human-readable horizon label
HORIZON_DAYS: dict[str, int] = {
    "1w":  5,
    "2w": 10,
    "1m": 21,
    "2m": 42,
    "3m": 63,
}

_FORECAST_DIR = Path("reports/forecasts")


@dataclass
class StepPrediction:
    """Model output for one forecast step."""
    step:       int
    direction:  Literal["UP", "DOWN", "UNCERTAIN"]
    p_up:       float
    p_down:     float
    confidence: float  # max(p_up, p_down)


@dataclass
class SimulationResult:
    """Complete simulation result for one ticker."""
    ticker:         str
    forecast_date:  date
    horizon_label:  str
    horizon_days:   int
    model_type:     str
    start_price:    float

    # Per-step signals (length == horizon_days)
    step_predictions: list[StepPrediction] = field(default_factory=list)

    # Price paths  (shape: n_paths × horizon_days+1; column 0 = start_price)
    paths:        np.ndarray = field(default_factory=lambda: np.empty((0,)))

    # Summary stats (computed at build time)
    mean_path:    list[float] = field(default_factory=list)
    lower_80:     list[float] = field(default_factory=list)
    upper_80:     list[float] = field(default_factory=list)
    lower_95:     list[float] = field(default_factory=list)
    upper_95:     list[float] = field(default_factory=list)

    # Overall summary
    net_direction:         str   = "UNCERTAIN"
    avg_confidence:        float = 0.0
    predicted_return_pct:  float = 0.0
    steps_bullish:         int   = 0
    steps_bearish:         int   = 0

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for JSON persistence)."""
        return {
            "ticker":        self.ticker,
            "forecast_date": str(self.forecast_date),
            "horizon_label": self.horizon_label,
            "horizon_days":  self.horizon_days,
            "model_type":    self.model_type,
            "start_price":   round(self.start_price, 4),
            "step_predictions": [
                {
                    "step":       s.step,
                    "direction":  s.direction,
                    "p_up":       round(s.p_up, 4),
                    "p_down":     round(s.p_down, 4),
                    "confidence": round(s.confidence, 4),
                }
                for s in self.step_predictions
            ],
            "simulation": {
                "mean_path": [round(v, 4) for v in self.mean_path],
                "lower_80":  [round(v, 4) for v in self.lower_80],
                "upper_80":  [round(v, 4) for v in self.upper_80],
                "lower_95":  [round(v, 4) for v in self.lower_95],
                "upper_95":  [round(v, 4) for v in self.upper_95],
            },
            "summary": {
                "net_direction":        self.net_direction,
                "avg_confidence":       round(self.avg_confidence, 4),
                "predicted_return_pct": round(self.predicted_return_pct, 2),
                "steps_bullish":        self.steps_bullish,
                "steps_bearish":        self.steps_bearish,
            },
        }


class PriceSimulator:
    """
    Generates forward price simulations for a ticker using a trained model.

    Parameters
    ----------
    n_paths : int
        Number of Monte Carlo paths (default 400).
    drift_scale : float
        How strongly model confidence translates into drift.
        1.0 means confidence=1.0 → drift = 1σ per day.
        0.5 (default) is more conservative.
    random_seed : int | None
        Fix for reproducible charts.
    """

    def __init__(
        self,
        n_paths:     int   = 400,
        drift_scale: float = 0.5,
        random_seed: int | None = 42,
    ) -> None:
        self._n_paths     = n_paths
        self._drift_scale = drift_scale
        self._rng         = np.random.default_rng(random_seed)

    # ── Public API ────────────────────────────────────────────────────────────

    def simulate(
        self,
        model:        BaseModel,
        X_latest:     pd.DataFrame,      # last available feature row(s)
        df_price:     pd.DataFrame,      # full OHLCV history (for vol estimate)
        ticker:       str,
        horizon:      str = "1m",
        model_type:   str = "ml",
    ) -> SimulationResult:
        """
        Run a Monte Carlo simulation for *ticker* over *horizon*.

        Parameters
        ----------
        model : BaseModel
            Fitted model.
        X_latest : pd.DataFrame
            Feature matrix for the latest available bars (only the last row
            is used for the signal; must match the model's feature schema).
        df_price : pd.DataFrame
            Historical OHLCV data used to estimate daily volatility.
        ticker : str
        horizon : str
            One of '1w', '2w', '1m', '2m', '3m'.
        model_type : str

        Returns
        -------
        SimulationResult
        """
        horizon_days = HORIZON_DAYS.get(horizon, 21)
        start_price  = float(df_price["close"].iloc[-1])

        # ── Daily volatility from recent history ──────────────────────────
        daily_vol = self._estimate_daily_vol(df_price)

        # ── Model signal for the current bar ─────────────────────────────
        # We use the last row of X_latest as the feature vector.
        last_features = X_latest.iloc[[-1]]
        proba = model.predict_proba(last_features)  # shape (1, 2): [P(down), P(up)]
        p_down = float(proba[0, 0])
        p_up   = float(proba[0, 1])

        # Build per-step predictions (signal is constant across all steps
        # because we only have one model output; future versions could
        # do recursive prediction here).
        step_preds: list[StepPrediction] = []
        for step in range(1, horizon_days + 1):
            # Decay confidence slightly over time to reflect forecast uncertainty
            decay     = 0.97 ** (step - 1)
            conf      = max(p_up, p_down) * decay
            # Re-normalise probabilities with decay
            eff_p_up   = 0.5 + (p_up   - 0.5) * decay
            eff_p_down = 1.0 - eff_p_up
            if eff_p_up >= 0.55:
                direction = "UP"
            elif eff_p_down >= 0.55:
                direction = "DOWN"
            else:
                direction = "UNCERTAIN"
            step_preds.append(StepPrediction(
                step=step,
                direction=direction,
                p_up=round(eff_p_up, 4),
                p_down=round(eff_p_down, 4),
                confidence=round(conf, 4),
            ))

        # ── Monte Carlo GBM simulation ────────────────────────────────────
        paths = self._run_gbm(
            start_price  = start_price,
            horizon_days = horizon_days,
            daily_vol    = daily_vol,
            step_preds   = step_preds,
        )

        # ── Confidence bands ──────────────────────────────────────────────
        mean_path = paths.mean(axis=0).tolist()
        lower_80  = np.percentile(paths, 10, axis=0).tolist()
        upper_80  = np.percentile(paths, 90, axis=0).tolist()
        lower_95  = np.percentile(paths,  2.5, axis=0).tolist()
        upper_95  = np.percentile(paths, 97.5, axis=0).tolist()

        # ── Summary ───────────────────────────────────────────────────────
        steps_bullish = sum(1 for s in step_preds if s.direction == "UP")
        steps_bearish = sum(1 for s in step_preds if s.direction == "DOWN")
        avg_conf      = float(np.mean([s.confidence for s in step_preds]))

        net_direction: str
        if steps_bullish > steps_bearish * 1.5:
            net_direction = "UP"
        elif steps_bearish > steps_bullish * 1.5:
            net_direction = "DOWN"
        else:
            net_direction = "UNCERTAIN"

        predicted_return_pct = (mean_path[-1] - start_price) / start_price * 100.0

        result = SimulationResult(
            ticker             = ticker,
            forecast_date      = date.today(),
            horizon_label      = horizon,
            horizon_days       = horizon_days,
            model_type         = model_type,
            start_price        = start_price,
            step_predictions   = step_preds,
            paths              = paths,
            mean_path          = mean_path,
            lower_80           = lower_80,
            upper_80           = upper_80,
            lower_95           = lower_95,
            upper_95           = upper_95,
            net_direction      = net_direction,
            avg_confidence     = avg_conf,
            predicted_return_pct = predicted_return_pct,
            steps_bullish      = steps_bullish,
            steps_bearish      = steps_bearish,
        )

        logger.info(
            "Forecast %s [%s]: %s  avg_conf=%.1f%%  pred_return=%.2f%%  vol=%.3f",
            ticker, horizon, net_direction, avg_conf * 100,
            predicted_return_pct, daily_vol,
        )
        return result

    # ── Internals ─────────────────────────────────────────────────────────────

    def _estimate_daily_vol(self, df: pd.DataFrame, window: int = 60) -> float:
        """Annualised-to-daily volatility from recent log returns."""
        closes  = df["close"].iloc[-window:] if len(df) >= window else df["close"]
        log_ret = np.log(closes / closes.shift(1)).dropna()
        return float(log_ret.std()) if len(log_ret) > 1 else 0.015

    def _run_gbm(
        self,
        start_price:  float,
        horizon_days: int,
        daily_vol:    float,
        step_preds:   list[StepPrediction],
    ) -> np.ndarray:
        """
        Return (n_paths × horizon_days+1) array of simulated prices.
        Column 0 is always start_price.
        """
        paths = np.empty((self._n_paths, horizon_days + 1))
        paths[:, 0] = start_price

        for step_idx, sp in enumerate(step_preds, start=1):
            direction = 1.0 if sp.direction == "UP" else (-1.0 if sp.direction == "DOWN" else 0.0)
            drift     = direction * sp.confidence * self._drift_scale * daily_vol
            # GBM: S_{t+1} = S_t * exp(drift - 0.5*σ² + σ*ε)
            ito_adj  = -0.5 * daily_vol ** 2
            noise    = self._rng.standard_normal(self._n_paths) * daily_vol
            log_ret  = drift + ito_adj + noise
            paths[:, step_idx] = paths[:, step_idx - 1] * np.exp(log_ret)

        return paths


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_forecast(result: SimulationResult) -> Path:
    """Save a SimulationResult to a dated JSON file and return the path."""
    _FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{result.ticker.replace('.', '_')}_"
        f"{result.forecast_date}_"
        f"{result.horizon_label}.json"
    )
    path = _FORECAST_DIR / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info("Forecast saved: %s", path)
    return path


def load_forecast(path: Path) -> dict:
    """Load a previously saved forecast JSON."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def list_past_forecasts(horizon_elapsed: bool = True) -> list[Path]:
    """
    Return all saved forecast JSON paths.
    If horizon_elapsed=True, only return forecasts where the horizon has passed.
    """
    if not _FORECAST_DIR.exists():
        return []
    paths = sorted(_FORECAST_DIR.glob("*.json"))
    if not horizon_elapsed:
        return paths

    today = date.today()
    due: list[Path] = []
    for p in paths:
        try:
            data      = load_forecast(p)
            fc_date   = date.fromisoformat(data["forecast_date"])
            horizon_d = int(data["horizon_days"])
            # Add extra buffer: horizon + 5 trading days to allow data to settle
            due_date  = fc_date + timedelta(days=int(horizon_d * 1.4))
            if today >= due_date:
                due.append(p)
        except Exception:
            pass
    return due
