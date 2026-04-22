# Trading Bot — Python Quantitative Trading System

> **Educational and engineering project. Not a promise of profitability.**
> This system is designed for learning, research, and strategy development.
> Always validate thoroughly before using real money.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Quick-Start Command Reference](#quick-start-command-reference)
6. [Modes](#modes)
   - [train — Train the Model](#train--train-the-model)
   - [backtest — Historical Backtest](#backtest--historical-backtest)
   - [live — Paper / Live Trading](#live--paper--live-trading)
   - [chart — Price + Signal Charts](#chart--price--signal-charts)
   - [forecast — Price Simulation](#forecast--price-simulation)
   - [evaluate-forecast — Compare & Learn](#evaluate-forecast--compare--learn)
   - [report — Regenerate Reports](#report--regenerate-reports)
7. [GPU Acceleration](#gpu-acceleration)
8. [Features](#features)
9. [Risk Controls](#risk-controls)
10. [Live Broker Setup](#live-broker-setup)
11. [Key Risks and Limitations](#key-risks-and-limitations)

---

## Overview

This trading bot:

- Fetches OHLCV data from **Yahoo Finance** (17 tickers: US large-caps + French CAC-40)
- Engineers **127 features**: technical indicators, candle patterns, multi-timeframe analysis, session/calendar features, and news sentiment
- Supports **two model backends** selectable via `--model-type`:
  - **`ml`** — XGBoost / LightGBM / Logistic Regression trained on the 127-feature matrix
  - **`deep`** — 1D CNN with residual blocks or Bidirectional LSTM via PyTorch, operating on 20-bar rolling sequences
- Auto-detects the best available GPU (**NVIDIA CUDA → AMD DirectML → Apple MPS → CPU**)
- Supports **incremental / transfer learning** by default — each training run fine-tunes the previous model
- **Forecast mode** — simulates price paths 1 week to 3 months ahead using Monte Carlo GBM with model-informed drift; saves JSON + 3-panel fan charts
- **Self-correction** — after the horizon passes, compares forecasts to actual prices and optionally retrains on confident mistakes
- Fetches live news from **NewsAPI** scored with **VADER sentiment** (company-name queries for all tickers, including French)
- Generates BUY/SELL/HOLD signals — only acts when model confidence ≥ **70%** (configurable)
- Runs in **TEST mode** (virtual money) by default; **LIVE mode** only when explicitly enabled
- Produces rich 3-panel signal charts saved to `reports/`
- Is fully broker-agnostic (Mock, IBKR, Saxo) via adapter pattern

---

## Architecture

```
trading-bot/
├── config/
│   ├── settings.yaml          # All strategy, model, risk, and system config
│   ├── .env                   # Secrets: API keys, broker credentials (never commit)
│   └── .env.example           # Template for .env
│
├── data/
│   ├── base.py                # DataProvider abstract base class
│   ├── factory.py             # DataSourceFactory (selects provider from settings)
│   ├── yahoo.py               # Yahoo Finance OHLCV provider
│   ├── alpha_vantage.py       # Alpha Vantage provider (optional)
│   ├── news.py                # ★ NewsAPI + VADER — company-name queries for all tickers
│   ├── macro.py               # FRED macro data provider (optional)
│   └── cache.py               # Local disk cache (Parquet, TTL-based)
│
├── features/
│   ├── technical.py           # RSI, MACD, ATR, Bollinger Bands, Stochastic, etc.
│   ├── returns.py             # Returns, volatility, drawdown, high-low range
│   ├── candles.py             # ★ 14 candle patterns + composite score
│   ├── multi_timeframe.py     # ★ Weekly + monthly higher-TF features
│   ├── session.py             # ★ London/NY/Asian session + holidays + economic calendar
│   └── pipeline.py            # Full pipeline: OHLCV + candles + session + MTF + sentiment
│
├── forecasting/
│   ├── simulator.py           # ★ Monte Carlo GBM price simulation (400 paths)
│   └── evaluator.py           # ★ Compare past forecasts to actuals, extract mistakes
│
├── models/
│   ├── base.py                # BaseModel abstract base
│   ├── logistic.py            # Logistic Regression baseline
│   ├── random_forest.py       # Random Forest
│   ├── gradient_boosting.py   # XGBoost > LightGBM > sklearn GradientBoosting
│   ├── deep_learning.py       # ★ 1D CNN + LSTM via PyTorch (sklearn MLP fallback)
│   ├── gpu_utils.py           # ★ Auto GPU detection: CUDA → DirectML → MPS → CPU
│   ├── trainer.py             # Walk-forward validation + final training + incremental
│   └── evaluator.py           # Accuracy, F1, ROC-AUC metrics
│
├── strategy/
│   ├── signal.py              # Signal dataclass (BUY / SELL / HOLD)
│   ├── generator.py           # Signal generation
│   └── filters.py             # Liquidity, volatility, holiday, session, news-day filters
│
├── execution/
│   └── manager.py             # Order manager, pre-trade checks, dry-run support
│
├── brokers/
│   ├── base.py                # BrokerInterface abstract base
│   ├── factory.py             # BrokerFactory
│   ├── mock.py                # MockBroker for TEST/paper trading
│   ├── ibkr.py                # Interactive Brokers adapter (placeholder)
│   └── saxo.py                # Saxo Bank adapter (placeholder)
│
├── risk/
│   ├── sizing.py              # Position sizing (fixed, percent, ATR-based, Kelly)
│   └── controls.py            # Max drawdown, daily loss limit, kill switch
│
├── backtesting/
│   ├── engine.py              # Walk-forward backtest engine
│   └── portfolio.py           # Virtual portfolio: positions, equity, PnL
│
├── reporting/
│   ├── reporter.py            # Performance reports (Sharpe, CAGR, drawdown, etc.)
│   └── plots.py               # Equity curve, drawdown, trade dist, price+signal charts
│
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   └── test_backtest.py
│
├── models/saved/              # Trained model files (auto-created)
├── reports/                   # Report PNGs and CSVs (auto-created)
│   └── forecasts/             # Forecast JSONs + charts (auto-created)
├── data/cache/                # OHLCV cache (auto-created)
├── main.py                    # Entry point — dispatches all modes
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.12+
- pip

### Steps

```bash
# 1. Clone or download the project
cd trading-bot

# 2. Create a virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux / Mac:
source .venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. GPU support (optional — CPU works out of the box)
#    NVIDIA (CUDA 12.x):
pip install torch --index-url https://download.pytorch.org/whl/cu124
#    AMD on Windows (DirectML — e.g. RX 7800 XT):
pip install torch-directml
#    AMD on Linux (ROCm 6.2):
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

# 5. Copy the secrets template and fill it in
copy config\.env.example config\.env
# Edit config\.env — add your API keys (see Configuration below)

# 6. Review config\settings.yaml — tickers, risk limits, model settings
```

---

## Configuration

### `config/settings.yaml` — main config

#### Tickers (17 stocks across US and French markets)

```yaml
tickers:
  # ── US tech / large-cap ───────────────────────────────────────────────────
  - AAPL       # Apple
  - MSFT       # Microsoft
  - GOOGL      # Alphabet
  - AMZN       # Amazon
  - NVDA       # NVIDIA
  - TSLA       # Tesla
  # ── French CAC-40 / blue-chips ────────────────────────────────────────────
  - HO.PA      # Thales
  - DSY.PA     # Dassault Systèmes
  - TTE.PA     # TotalEnergies
  - AIR.PA     # Airbus
  - AI.PA      # Air Liquide
  - BNP.PA     # BNP Paribas
  - CS.PA      # AXA
  - MC.PA      # LVMH
  - ORA.PA     # Orange
  - VIE.PA     # Veolia
  - GLE.PA     # Société Générale
```

#### Key settings

```yaml
data:
  provider: yahoo
  history_years: 5
  news:
    enabled: true           # Fetch news via NewsAPI + VADER sentiment
    lookback_days: 30

strategy:
  confidence_threshold: 0.70   # Only act on ≥ 70% model confidence
  max_positions: 5              # Max simultaneous open positions (not max tickers watched)

model:
  type: gradient_boosting       # For --model-type ml
  deep_learning:
    architecture: cnn_1d        # cnn_1d | lstm
    sequence_length: 20
    epochs: 50
    patience: 10

risk:
  position_pct: 0.10            # 10% of portfolio per position
  stop_loss_pct: 0.05
  max_drawdown_pct: 0.15

# Data lookback for live trading and forecast mode (default: 3mo ≈ 63 bars).
# 3mo is the recommended minimum — enough for all indicators (SMA-50, RSI, etc.)
# while keeping the model focused on recent market conditions.
# Options: 1mo | 3mo | 6mo | 1y | 2y | 5y
live_period:     3mo   # --mode live — OHLCV lookback per poll cycle
forecast_period: 3mo   # --mode forecast — OHLCV lookback when generating simulations
```

### `config/.env` — secrets

```dotenv
NEWS_API_KEY=your_newsapi_key_here      # https://newsapi.org (free tier: 100 req/day)
ALPHA_VANTAGE_API_KEY=your_key_here     # Only if provider: alpha_vantage
FRED_API_KEY=your_key_here             # Only if macro.enabled: true

# Broker credentials (LIVE mode only)
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
```

> **Never commit `config/.env` to version control.**

---

## Quick-Start Command Reference

```bash
# ── Training ──────────────────────────────────────────────────────────────────

# Train gradient boosting (fast, ~10 s):
python main.py --mode train --model-type ml

# Train deep learning CNN — incremental by default (continues from last run):
python main.py --mode train --model-type deep --train-time 60

# Train with a 30-minute budget:
python main.py --mode train --model-type deep --train-time 30

# Discard previous model and train from scratch:
python main.py --mode train --model-type deep --train-time 60 --scratch

# Single ticker only:
python main.py --mode train --model-type ml --ticker AAPL

# ── Forecasting ───────────────────────────────────────────────────────────────

# 1-month forecast for all tickers (deep model):
python main.py --mode forecast --model-type deep --horizon 1m

# 1-week forecast, single ticker:
python main.py --mode forecast --model-type ml --horizon 1w --ticker AAPL

# 3-month forecast:
python main.py --mode forecast --model-type deep --horizon 3m

# After the horizon passes — compare predictions to actual prices:
python main.py --mode evaluate-forecast --model-type deep

# After the horizon passes — compare AND retrain on mistakes:
python main.py --mode evaluate-forecast --model-type deep --retrain

# ── Backtesting ───────────────────────────────────────────────────────────────

# Backtest all tickers (uses model_ml.pkl):
python main.py --mode backtest --model-type ml

# Custom date range for a single ticker:
python main.py --mode backtest --ticker NVDA --start 2022-01-01 --end 2024-12-31

# ── Live / Paper Trading ──────────────────────────────────────────────────────

# Paper trading with deep learning model (safe, no real orders):
python main.py --mode live --model-type deep --dry-run

# Paper trading with ML model:
python main.py --mode live --model-type ml --dry-run

# ── Charts ────────────────────────────────────────────────────────────────────

# Signal chart for all tickers (1 year):
python main.py --mode chart --model-type deep

# Single ticker, 3-month view:
python main.py --mode chart --ticker AAPL --period 3mo --model-type ml

# ── Reports ───────────────────────────────────────────────────────────────────
python main.py --mode report

# ── Tests ─────────────────────────────────────────────────────────────────────
python -m pytest tests/ -v
python -m pytest tests/ --cov=. --cov-report=term-missing
```

---

## Modes

### `train` — Train the Model

Downloads historical OHLCV data for all 17 tickers, fetches multi-timeframe and news sentiment features, engineers 127 features, runs **5-fold walk-forward cross-validation**, and trains the final model.

#### All `--mode train` flags

| Flag | Default | Description |
|---|---|---|
| `--model-type ml` | `ml` | XGBoost / LightGBM gradient boosting |
| `--model-type deep` | — | 1D CNN or LSTM via PyTorch |
| `--train-time N` | None | Budget in **minutes** (e.g. `60` = 1 hour) |
| `--scratch` | off | Discard previous model, train from random weights |
| `--ticker X` | all | Train on a single ticker only |

#### Incremental training (default behaviour)

By default every training run **continues from where the last one left off**. Each run improves on the previous one without resetting:

```bash
# Run 1: no saved model → trains from scratch automatically
python main.py --mode train --model-type ml
python main.py --mode train --model-type deep --train-time 60

# Run 2+: loads previous model and continues
python main.py --mode train --model-type ml
python main.py --mode train --model-type deep --train-time 60

# Reset: discard all previous learning, start fresh
python main.py --mode train --model-type ml --scratch
python main.py --mode train --model-type deep --train-time 60 --scratch
```

**What "incremental" means per model type:**

| Model | Incremental behaviour |
|---|---|
| **XGBoost** (`--model-type ml`) | Adds 300 more trees to the existing ensemble via `warm_start=True`. Each run the forest grows: 300 → 600 → 900 → … |
| **CNN / LSTM** (`--model-type deep`) | Loads previous weights and continues gradient descent from that checkpoint |
| **No saved model exists** | Silently starts from scratch (identical to first run) |
| **`--scratch` flag** | Discards everything and reinitialises randomly |

When using `--train-time` with `--model-type ml`, the timed hyperparameter search uses the **existing model's ROC-AUC as the floor** — any new configuration must beat it to replace it. If nothing beats it, the existing model gets more trees added instead of being discarded.

Log output tells you which path was taken:
```
Incremental mode: loaded existing model from models/saved/model_ml.pkl.
Incremental GradientBoosting(xgboost): adding 300 trees to existing 500 (total: 800).
# or
Incremental mode: no existing model at ... — starting fresh.
# or (--scratch)
From-scratch mode: ignoring any existing saved model.
```

#### Training budget and restarts

With `--train-time N`:
- **30%** of the budget is given to walk-forward validation (5 folds)
- **70%** is given to final model training on all data
- If early stopping fires while time remains, training **automatically restarts** from the best checkpoint with the learning rate halved — so the full budget is always used

#### Model type comparison

| | `--model-type ml` | `--model-type deep` |
|---|---|---|
| **Algorithm** | XGBoost (gradient boosting) | 1D CNN with residual blocks (PyTorch) |
| **Input** | Single row per bar (127 features) | 20-bar rolling sequence (127 × 20) |
| **Temporal context** | None — stateless bar-by-bar | Yes — 20-bar lookback window |
| **Walk-forward ROC-AUC** | ~0.72 | ~0.59 (improves with longer `--train-time`) |
| **Training time** | ~10 seconds | scales with `--train-time` |
| **Save path** | `models/saved/model_ml.pkl` | `models/saved/model_deep.pkl` |
| **GPU support** | No | Yes — CUDA / DirectML / MPS |
| **Incremental** | ✅ adds trees via warm-start | ✅ continues from saved weights |

#### Change deep learning architecture

```yaml
# config/settings.yaml
model:
  deep_learning:
    architecture: lstm     # cnn_1d (default) | lstm
    sequence_length: 20
    epochs: 50
    patience: 10
```

---

### `backtest` — Historical Backtest

```bash
# All tickers, default date range:
python main.py --mode backtest

# Custom range, single ticker:
python main.py --mode backtest --ticker NVDA --start 2022-01-01 --end 2024-12-31
```

Outputs equity curve, drawdown chart, trade distribution, and performance summary (Sharpe ratio, CAGR, max drawdown, win rate) to `reports/`.

---

### `live` — Paper / Live Trading

Polls market data every 60 seconds, fetches multi-timeframe + news sentiment features, generates signals and optionally places orders via a connected broker.

```bash
# Paper trading — all 17 tickers, no real orders:
python main.py --mode live --model-type deep --dry-run

# Stop with Ctrl+C
```

Each poll cycle:
1. **Checks market hours** — prints a status banner per exchange; skips closed markets; if all markets are closed, sleeps 5 min and retries automatically
2. Fetches the last **3 months** of OHLCV for active (open-market) tickers only — configurable via `live_period` in `settings.yaml`
3. Refreshes news sentiment **once per calendar day** (cached in memory between polls — never hits the API more than once per day per ticker)
4. Runs the full feature pipeline including the 15 open/close price features
5. Generates signals — only BUY/SELL at ≥ 70% confidence
6. Risk controls check daily loss, drawdown, kill switch before any order
7. Prints a portfolio snapshot and saves a row to `reports/portfolio_snapshots.csv`

**Market status banner** (shown every poll):
```
──────────────────────────────────────────────────────────────
  🕐  MARKET STATUS
──────────────────────────────────────────────────────────────
  🟢 OPEN    NASDAQ (New York)       14:33 EDT   (09:30 – 16:00 EDT)
  🔴 CLOSED  Euronext Paris          19:33 CET   (09:00 – 17:30 CET)
  ──────────────────────────────────────────────────────────
  Active tickers  (6): AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA
  Closed tickers  (11): HO.PA, DSY.PA … GLE.PA
──────────────────────────────────────────────────────────────
```

When all markets are closed (e.g. 10 PM Paris, weekends):
```
  💤  No markets open right now. Euronext Paris opens at 09:00 in 13h 24m.
  Sleeping 5 minutes…
```

> **`max_positions: 5`** limits how many positions can be **open simultaneously** — it does **not** limit which tickers are watched. All 17 tickers are polled every cycle.

---

### `chart` — Price + Signal Charts

```bash
# All tickers, 1-year view:
python main.py --mode chart --model-type deep

# Single ticker, 3-month zoom:
python main.py --mode chart --ticker AAPL --period 3mo --model-type ml
```

3-panel PNG chart:

| Panel | Content |
|---|---|
| **Top** | Close price + SMA-20 + ▲ BUY / ▼ SELL signal markers |
| **Middle** | Volume bars (green = up-day, red = down-day) |
| **Bottom** | Model confidence per bar with 70% threshold line |

Saved to `reports/{TICKER}_{PERIOD}_signals.png`.

---

### `forecast` — Price Simulation

Simulates how each stock's price could evolve over the next week, month, or quarter. Uses the model's current signal as a drift coefficient in a Geometric Brownian Motion, running 400 Monte Carlo paths to produce a probability fan chart.

```bash
# 1-month forecast for all 17 tickers:
python main.py --mode forecast --model-type deep --horizon 1m

# Available horizons:
#   1w = 5 trading days
#   2w = 10 trading days
#   1m = 21 trading days  (default)
#   2m = 42 trading days
#   3m = 63 trading days
```

**Output per ticker:**
- `reports/forecasts/{TICKER}_{DATE}_{HORIZON}.json` — full simulation data
- `reports/forecasts/{TICKER}_{DATE}_{HORIZON}.png` — 3-panel chart

**Chart panels:**

| Panel | Content |
|---|---|
| **Top** | Last 60 days of historical price + simulated fan (95% and 80% confidence bands + mean path) |
| **Middle** | Per-step model confidence (decays over time to reflect forecast uncertainty) |
| **Bottom** | Predicted direction per day — 🟢 UP / 🔴 DOWN / ⬛ UNCERTAIN |

**Console summary table:**
```
  Ticker     Direction    Pred %  AvgConf  Bullish  Bearish  Chart
  AAPL       ▲ UP          +2.34   74.1%       18        3   AAPL_2026-03-16_1m.png
  TSLA       ▼ DOWN        -1.87   68.3%        5       16   TSLA_2026-03-16_1m.png
  MC.PA      ◆ UNCERT      +0.41   52.1%       11       10   MC_PA_2026-03-16_1m.png
```

**How confidence decays:**
Model confidence is at its highest today and decays by 3% per step (`0.97^step`). By day 21 (~1 month), confidence has decayed to ~53% of today's reading — reflecting that the further out you forecast, the less reliable a single model prediction is.

---

### `evaluate-forecast` — Compare & Learn

After the forecast horizon has elapsed, compare each saved forecast to the actual price history and optionally retrain the model on its mistakes.

```bash
# View evaluation report (no retraining):
python main.py --mode evaluate-forecast --model-type deep

# View report AND retrain on confident mistakes:
python main.py --mode evaluate-forecast --model-type deep --retrain
```

**What it reports:**
```
────────────────────────────────────────────────────────────────────────────────
  📊  FORECAST EVALUATION REPORT  —  17 forecast(s) reviewed
────────────────────────────────────────────────────────────────────────────────
  Direction accuracy : 11/17  (65%)
  Avg price MAE      : 4.2300
  Avg price RMSE     : 6.1800
────────────────────────────────────────────────────────────────────────────────
  ✅ AAPL       [1m]  predicted=UP        actual=UP        pred_ret=+2.34%  actual_ret=+3.12%  step_acc=71%  conf_mistakes=2
  ❌ TSLA       [1m]  predicted=DOWN      actual=UP        pred_ret=-1.87%  actual_ret=+4.50%  step_acc=43%  conf_mistakes=7
```

**Self-correction (--retrain):**
For each forecast where the model was confidently wrong (confidence ≥ 60% but direction was wrong), the actual data from that period is fed back through the feature pipeline and used to fine-tune the model incrementally — so it learns from its mistakes without discarding what it already knows.

---

### `report` — Regenerate Reports

```bash
python main.py --mode report
```

Re-generates equity curve charts from previously saved backtest CSV files.

---

## GPU Acceleration

The deep learning model auto-detects the best available compute device at runtime — no code changes needed:

| GPU | Detection | Notes |
|---|---|---|
| **NVIDIA** | `torch.cuda.is_available()` | Install: `pip install torch --index-url .../cu124` |
| **AMD on Windows** | `torch_directml` present | Install: `pip install torch-directml` |
| **Apple M-series** | `torch.backends.mps.is_available()` | Built-in on macOS 12.3+ |
| **CPU** | Always available | Fallback if no GPU found |

The selected device is logged at the start of each training run:
```
GPU [AMD/DirectML]: AMD Radeon RX 7800 XT — using DirectML.
DeepModel using device: PRIVATEUSEONE:0
```

> **AMD DirectML note:** One Adam optimizer kernel (`lerp`) falls back to CPU silently — this is a known DirectML limitation affecting only ~0.01% of compute. The heavy conv/backprop operations all run on the GPU. The warning is suppressed automatically.

---

## Features

The model is trained on **127 features** across 6 categories:

| Category | Count | Key Features |
|---|---|---|
| **Technical** | ~35 | RSI, MACD, ATR, Bollinger Bands, Stochastic, SMA/EMA |
| **Returns & Vol** | ~20 | Log returns 1–20d, rolling volatility, drawdown, high-low range |
| **Volume** | ~5 | Volume ratio vs. MA at 5/10/20 days |
| **Open/Close Price** | 15 | Intraday position, open position, overnight gap, 5-day rolling context |
| **Candle Patterns** | ~15 | 14 patterns + composite score |
| **Session & Calendar** | ~20 | Day-of-week, sessions, holidays, NFP/FOMC/CPI proximity |
| **Multi-Timeframe** | ~26 | Weekly + monthly RSI, MACD, SMA, ATR, direction, composite |
| **News Sentiment** | 3 | VADER compound score, positive ratio, article count |

### Open/Close Price Features

Every bar contributes **15 open/close price features** that capture intraday structure, overnight gaps, and multi-day consistency:

**Single-bar (computed each day):**

| Feature | Formula | What it tells the model |
|---|---|---|
| `intraday_position` | (close − low) / (high − low) | 1 = closed at day high, 0 = closed at day low |
| `open_position` | (open − low) / (high − low) | Where the bar opened in the day's range |
| `close_to_high` | (high − close) / (high − low) | How much was "left on the table" vs. the day high |
| `gap_open` | (open − prev_close) / prev_close | Overnight gap — post-close sentiment |
| `candle_body` | close − open | Raw intraday move (signed) |
| `candle_body_pct` | (close − open) / open | Intraday return (signed %) |
| `upper_shadow` | high − max(open, close) | Rejection at the top of the bar |
| `lower_shadow` | min(open, close) − low | Support bounce at the bottom |
| `body_to_range` | \|body\| / (high − low) | How decisive the day was |

**Rolling 5-day (captures recent behavioural patterns):**

| Feature | What it tells the model |
|---|---|
| `gap_open_5d_mean` | Average overnight gap over 5 days — sustained gap pressure |
| `gap_open_5d_sum` | Cumulative overnight gap — is the market consistently gapping up or down? |
| `body_pct_5d_mean` | Average intraday return over 5 days — trend of close-vs-open momentum |
| `bull_days_5d` | Fraction of last 5 days where close > open — buyer/seller consistency |
| `gap_up_days_5d` | Fraction of last 5 days with a positive overnight gap |
| `intraday_pos_5d` | Rolling mean of intraday_position — did price consistently close near its daily highs? |

These features are especially relevant for `--mode live` and `--mode forecast`, which now use a **3-month lookback window** (`live_period` / `forecast_period` in `settings.yaml`) focused on recent price action.

---

### Candle Pattern Detection

14 patterns in pure pandas (no TA-Lib required):

| Type | Patterns |
|---|---|
| **Single-candle** | Doji, Hammer, Shooting Star, Pin Bar, Marubozu, Spinning Top |
| **Two-candle** | Bullish/Bearish Engulfing, Harami, Tweezer Top/Bottom |
| **Three-candle** | Morning Star, Evening Star, Three White Soldiers, Three Black Crows |
| **Multi-bar** | Inside Bar, Outside Bar, Fair Value Gap |

### Multi-Timeframe Analysis

Weekly (`1wk`) and monthly (`1mo`) indicators aligned to the daily index. Higher timeframes receive stronger weighting:
- `mtf_trend_score = (monthly×4 + weekly×2) / 6`
- Per-TF features: RSI, SMA position, MACD direction, trend vote

**Impact:** MTF features improved ROC-AUC from **0.530 → 0.719**.

### News Sentiment

French tickers use company-name queries (not the `.PA` ticker code) so NewsAPI returns real results:

| Ticker | Query |
|---|---|
| `HO.PA` | `"Thales Group defense aerospace"` |
| `AIR.PA` | `"Airbus aircraft aerospace"` |
| `MC.PA` | `"LVMH luxury"` |
| `AI.PA` | `"Air Liquide industrial gases"` |
| ... | ... |

Three daily features: `news_sentiment`, `news_positive_ratio`, `news_article_count`.  
Dates outside the 30-day free-tier window get neutral fill so no training rows are dropped.

#### News cache (rate-limit protection)

Each ticker's sentiment result is cached to disk for the current calendar day. Subsequent calls within the same day (e.g. multiple training runs, live polling every 60 s) load from the cache instead of hitting the API:

```
data/cache/news/AAPL_2026-03-16.parquet
data/cache/news/TSLA_2026-03-16.parquet
data/cache/news/AIR_PA_2026-03-16.parquet   # dots replaced with underscores
...
```

| Situation | Action |
|---|---|
| Cache file for today exists | Loaded from disk — **no API call** |
| Cache file missing (first call of the day) | Fetched from NewsAPI → saved to disk |
| Files older than 1 day | Automatically deleted on next write |

This keeps daily API usage at **1 request per ticker per day** (17 requests/day for 17 tickers), well within the 100 request/day free tier.

---

## Risk Controls

| Control | Setting | Default |
|---|---|---|
| Stop loss | `stop_loss_pct` | 5% |
| Take profit | `take_profit_pct` | 15% |
| Max daily loss | `max_daily_loss_pct` | 3% — pauses trading for the day |
| Max drawdown | `max_drawdown_pct` | 15% — activates kill switch |
| Max open positions | `max_positions` | 5 |
| Position size | `position_pct` | 10% of portfolio per trade |
| Kill switch | `KILL_SWITCH=true` in `.env` | Immediately stops all activity |

---

## Live Broker Setup

| Name | Status | Notes |
|---|---|---|
| `mock` | ✅ Ready | Virtual portfolio, paper trading, no real orders |
| `ibkr` | 🔧 Placeholder | Requires `ibapi` SDK + TWS/IB Gateway |
| `saxo` | 🔧 Placeholder | Requires Saxo OpenAPI credentials |

To use a real broker:
1. Set `broker.name: ibkr` in `settings.yaml`
2. Fill in credentials in `config/.env`
3. Set `live_trading_enabled: true` in `settings.yaml`
4. Always test in `--dry-run` mode first

---

## Key Risks and Limitations

1. **No profitability guarantee.** ML models trained on historical data may not generalise to future markets.
2. **NewsAPI free tier.** Only ~28 days of historical news; older training dates use neutral sentiment fill. News is cached daily to stay within the 100 req/day limit.
3. **Data quality.** Yahoo Finance data may have gaps, splits, or dividend-adjustment errors.
4. **Overfitting.** Walk-forward validation reduces but does not eliminate overfitting risk.
5. **Execution risk.** Slippage, latency, and partial fills can significantly impact live results.
6. **Regulatory risk.** Ensure compliance with applicable financial regulations (MiFID II, AMF, etc.).
7. **Broker adapters.** IBKR and Saxo adapters are stubs requiring full SDK implementation before live use.
8. **Market regime changes.** A model trained in a bull market may fail in a bear market.
9. **This is not financial advice.**
