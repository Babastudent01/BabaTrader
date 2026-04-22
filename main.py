"""
main.py
Trading Bot — entry point.

Modes (set via --mode CLI argument or MODE env var):
  backtest   — Run historical backtest on configured tickers.
  train      — Train / retrain the ML model and save it.
  live       — Connect to broker and run live trading loop.
  report     — Generate reports and plots from a saved backtest result.

Usage:
  python main.py --mode backtest
  python main.py --mode train
  python main.py --mode live
  python main.py --mode backtest --ticker AAPL --start 2020-01-01 --end 2023-12-31
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Force UTF-8 on Windows console so unicode chars don't crash ───────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Bootstrap logging before any other imports ────────────────────────────────
# Console: WARNING+ only (clean terminal output).
# File:    INFO+  (full detail always saved to trading_bot.log).
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.WARNING)
_console_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)

_file_handler = logging.FileHandler("trading_bot.log", encoding="utf-8")
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)

logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler])
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ML Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "train", "live", "chart", "report", "forecast", "evaluate-forecast"],
        default="backtest",
        help="Operating mode (default: backtest).",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Override ticker symbol (uses settings.yaml tickers if not set).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Backtest start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Backtest end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to settings YAML file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log orders without sending them to the broker.",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="1y",
        help="Data period for chart mode (e.g. '1y', '6mo', '3mo'). Default: 1y.",
    )
    parser.add_argument(
        "--horizon",
        type=str,
        default="1m",
        choices=["1w", "2w", "1m", "2m", "3m"],
        help="Forecast horizon (default: 1m). 1w=5d, 2w=10d, 1m=21d, 2m=42d, 3m=63d.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        default=False,
        help="(evaluate-forecast) Incrementally retrain the model on confident mistakes.",
    )
    parser.add_argument(
        "--model-type",
        choices=["ml", "deep"],
        default="ml",
        dest="model_type",
        help=(
            "Model backend to train / load for signal generation.\n"
            "  ml   — Gradient Boosting / Random Forest / Logistic (default)\n"
            "  deep — 1D CNN or LSTM via PyTorch (sklearn MLP fallback if torch not installed)"
        ),
    )
    parser.add_argument(
        "--train-time",
        type=float,
        default=None,
        dest="train_time",
        metavar="MINUTES",
        help=(
            "Training time budget in minutes.\n"
            "  ml   — uses the budget to run a timed random hyperparameter search.\n"
            "  deep — keeps training epochs until the budget is exhausted (early-stopping still applies).\n"
            "If not set, uses the defaults from config/settings.yaml."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        dest="verbose",
        help="Show INFO logs in the terminal (default: WARNING and above only).",
    )
    parser.add_argument(
        "--scratch",
        action="store_true",
        default=False,
        dest="scratch",
        help=(
            "Train from scratch, discarding any previously saved model weights.\n"
            "  deep — ignores existing CNN/LSTM weights and re-initialises randomly.\n"
            "  ml   — runs a full hyperparameter search from scratch.\n"
            "By default (without this flag) training is incremental: existing weights\n"
            "are loaded and fine-tuned if a saved model exists."
        ),
    )
    parser.add_argument(
        "--reset-portfolio",
        action="store_true",
        default=False,
        dest="reset_portfolio",
        help=(
            "Start live trading with a clean portfolio (100 000 € cash, no positions),\n"
            "discarding the state saved from the previous session.\n"
            "Without this flag, live trading resumes from where it left off."
        ),
    )
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_mtf_helper(settings):
    """
    Return a callable ``fetch(ticker, index) -> pd.DataFrame`` for MTF features,
    or None if multi_timeframe is disabled in settings.
    """
    mtf_cfg = settings.features.get("multi_timeframe", {})
    if not mtf_cfg.get("enabled", False):
        return None
    try:
        from features.multi_timeframe import build_mtf_features
        timeframes   = list(mtf_cfg.get("timeframes", ["1wk", "1mo"]))
        fetch_period = str(mtf_cfg.get("fetch_period", "10y"))

        def _fetch(ticker: str, index) -> "pd.DataFrame":
            return build_mtf_features(
                ticker, index, timeframes=timeframes, fetch_period=fetch_period
            )

        logger.info("Multi-timeframe features enabled: %s.", timeframes)
        return _fetch
    except Exception as exc:
        logger.warning("Failed to initialise MTF helper: %s", exc)
        return None


def _build_news_provider(settings):
    """
    Return a NewsSentimentProvider if news is enabled in settings, else None.
    Logs a clear message so the user always knows whether news is active.
    """
    news_cfg = settings.data.get("news", {})
    if not news_cfg.get("enabled", False):
        return None

    import os
    api_key = os.environ.get("NEWS_API_KEY", "")
    if not api_key:
        logger.warning("news.enabled=true but NEWS_API_KEY is not set — skipping news features.")
        return None

    try:
        from data.news import NewsSentimentProvider
        lookback = int(news_cfg.get("lookback_days", 30))
        provider = NewsSentimentProvider(api_key=api_key, lookback_days=lookback)
        logger.info("News sentiment enabled (NewsAPI, lookback=%d days).", lookback)
        return provider
    except Exception as exc:
        logger.warning("Failed to initialise NewsSentimentProvider: %s", exc)
        return None


# ── Training session summary ─────────────────────────────────────────────────

def _fetch_realtime_quotes(tickers: list[str]) -> dict[str, float]:
    """
    Fetch the latest intraday price for each ticker via yfinance fast_info.

    Uses the lightweight ``fast_info`` endpoint (no full history download).
    Falls back gracefully to an empty dict entry if a ticker fails.
    This is only called during live trading to keep portfolio PnL and order
    fill prices accurate intraday (daily bars only have yesterday's close).

    Returns
    -------
    dict[str, float]
        Ticker → latest real-time price.  Missing tickers are absent.
    """
    try:
        import yfinance as yf
    except ImportError:
        return {}

    quotes: dict[str, float] = {}
    for ticker in tickers:
        try:
            info  = yf.Ticker(ticker).fast_info
            price = getattr(info, "last_price", None)
            if price is not None and float(price) > 0:
                quotes[ticker] = float(price)
        except Exception as exc:
            logger.debug("Real-time quote failed for %s: %s", ticker, exc)

    if quotes:
        logger.info(
            "Real-time quotes fetched for %d/%d tickers.",
            len(quotes), len(tickers),
        )
    return quotes


def _print_session_summary(
    model,
    model_path: "Path",
    wf_mean_roc: float | None,
    wf_std_roc: float | None,
    baseline_wf_roc: float | None,
    interrupted: bool = False,
) -> None:
    """
    Print a walk-forward ROC-AUC comparison table at the end of a training run.

    Uses the walk-forward (out-of-sample) ROC-AUC as the primary metric — NOT
    in-sample evaluation, which would always appear inflated.
    """
    line = "─" * 62
    tag  = " [INTERRUPTED — best checkpoint]" if interrupted else ""
    print(f"\n{line}")
    print(f"  📊  TRAINING SESSION SUMMARY{tag}")
    print(line)
    print(f"  Model   : {model.name}")
    print(f"  Saved   : {model_path}")

    if wf_mean_roc is not None:
        std_str = f" ± {wf_std_roc:.4f}" if wf_std_roc is not None else ""
        print(f"  WF ROC-AUC (out-of-sample) : {wf_mean_roc:.4f}{std_str}")

        if baseline_wf_roc is not None:
            delta     = wf_mean_roc - baseline_wf_roc
            delta_pct = (delta / (baseline_wf_roc + 1e-8)) * 100
            arrow     = "▲" if delta > 0.001 else ("▼" if delta < -0.001 else "═")
            print(f"  Previous session           : {baseline_wf_roc:.4f}")
            print(f"  Change                     : {arrow}  {delta:+.4f}  ({delta_pct:+.1f}%)")
            if delta > 0.001:
                print("  Verdict : ✅  Model improved — weights saved.")
            elif delta < -0.001:
                print("  Verdict : ⚠️  Walk-forward score regressed — consider --scratch.")
            else:
                print("  Verdict : ✅  Stable — no meaningful change.")
        else:
            print("  Verdict : ✅  First training run — model saved.")

        # Interpretation guide
        print()
        print("  Interpretation (daily stock direction prediction):")
        if wf_mean_roc >= 0.72:
            print("    🟢  Excellent (≥ 0.72) — strong edge over random")
        elif wf_mean_roc >= 0.65:
            print("    🟢  Good     (0.65–0.72) — real predictive signal")
        elif wf_mean_roc >= 0.58:
            print("    🟡  Fair     (0.58–0.65) — modest but useful edge")
        else:
            print("    🔴  Weak     (< 0.58) — near random, keep training")
    else:
        print("  WF ROC-AUC : not available (training was interrupted before folds completed)")

    print(f"{line}\n")


# ── Mode handlers ─────────────────────────────────────────────────────────────

def run_backtest(args, settings) -> None:
    """Run historical backtest."""
    from data.factory import DataSourceFactory
    from features.pipeline import FeaturePipeline
    from models.trainer import ModelTrainer
    from backtesting.engine import BacktestEngine
    from reporting.reporter import PerformanceReporter
    from reporting.plots import plot_equity_curve, plot_trade_distribution

    model_type = getattr(args, "model_type", "ml")
    tickers    = [args.ticker] if args.ticker else settings.tickers
    logger.info("Backtest mode: tickers=%s, model_type=%s", tickers, model_type)

    data_source = DataSourceFactory.create(settings)
    pipeline    = FeaturePipeline(settings)
    trainer     = ModelTrainer(settings)
    reporter    = PerformanceReporter(output_dir="reports")

    # ── Load the saved model (avoids data leakage from training on test data) ──
    model_path = _model_save_path(settings, model_type)
    model      = _create_model_for_type(settings, model_type)
    _model_loaded_from_disk = False

    if model_path.exists():
        model.load(model_path)
        print(f"\n  📂  Loaded saved {model_type.upper()} model from {model_path}")
        print(
            "  ⚠️   This model was trained on ALL tickers up to its training date.\n"
            "       The backtest tests its out-of-sample performance on the date range below.\n"
        )
        _model_loaded_from_disk = True
    else:
        print(
            f"\n  ⚠️   No saved model found at {model_path}.\n"
            f"       Will train on data BEFORE --start (or first 70% if no --start given).\n"
            f"       Run --mode train --model-type {model_type} first for a proper backtest.\n"
        )

    for ticker in tickers:
        logger.info("── Processing %s ──", ticker)

        # Fetch data
        df = data_source.fetch(ticker, period=settings.data.get("period", "5y"))
        if df is None or df.empty:
            logger.warning("No data for %s — skipping.", ticker)
            continue

        # If no saved model: train on data BEFORE the backtest start date only
        # (avoids data leakage — the model never sees the test period)
        if not _model_loaded_from_disk:
            if args.start:
                train_df = df[df.index < args.start]
            else:
                cut = int(len(df) * 0.70)
                train_df = df.iloc[:cut]

            if len(train_df) < 100:
                logger.warning("%s: not enough pre-start data to train — skipping.", ticker)
                continue

            df_feat_train = pipeline.transform(train_df, include_target=True).dropna()
            X_train = df_feat_train[pipeline.feature_names]
            y_train = df_feat_train["target"].astype(int)
            model   = _create_model_for_type(settings, model_type)
            model   = trainer.train_final(model, X_train, y_train)
            print(f"  🏋  Trained {model_type.upper()} model on {len(X_train):,} bars before {args.start or 'split point'}")

        # Run backtest
        engine = BacktestEngine(model, settings, feature_pipeline=pipeline)
        result = engine.run(ticker, df, start_date=args.start, end_date=args.end)

        # Report
        reporter.report(result, save=True)
        reporter.print_monthly_returns(result.equity_curve)

        # Plots
        ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        Path("reports").mkdir(exist_ok=True)
        plot_equity_curve(
            result.equity_curve,
            ticker=ticker,
            save_path=Path("reports") / f"{ticker}_{ts}_equity.png",
        )
        plot_trade_distribution(
            result.trades,
            ticker=ticker,
            save_path=Path("reports") / f"{ticker}_{ts}_trades.png",
        )


def _model_save_path(settings, model_type: str) -> "Path":
    """Return the save path for the given model type."""
    from pathlib import Path
    save_dir = Path(settings.model.get("save_dir", "models/saved"))
    suffix   = "deep" if model_type == "deep" else "ml"
    return save_dir / f"model_{suffix}.pkl"


def _create_model_for_type(settings, model_type: str):
    """Instantiate the right model class based on --model-type."""
    if model_type == "deep":
        from models.deep_learning import DeepModel
        return DeepModel(settings)
    else:
        from models.trainer import create_model as _create_ml
        return _create_ml(settings)


def run_train(args, settings) -> None:
    """Train and save the ML model."""
    from data.factory import DataSourceFactory
    from features.pipeline import FeaturePipeline
    from models.trainer import ModelTrainer
    from reporting.plots import plot_feature_importances
    from pathlib import Path
    import pandas as pd

    model_type = getattr(args, "model_type", "ml")
    tickers    = [args.ticker] if args.ticker else settings.tickers
    logger.info("Train mode: tickers=%s, model_type=%s", tickers, model_type)

    data_source   = DataSourceFactory.create(settings)
    pipeline      = FeaturePipeline(settings)
    trainer       = ModelTrainer(settings)
    news_provider = _build_news_provider(settings)
    mtf_fetch     = _build_mtf_helper(settings)

    import time as _t
    _t0 = _t.monotonic()
    print(f"\n  🔄  Fetching data for {len(tickers)} ticker(s)...")

    all_features = []
    ticker_group_id = 0
    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i:2d}/{len(tickers)}] {ticker:<10}", end="", flush=True)
        df = data_source.fetch(ticker, period=settings.data.get("period", "5y"))
        if df is None or df.empty:
            print("  ✗  no data — skipped")
            logger.warning("No data for %s — skipping.", ticker)
            continue
        print(f"  {len(df):>5} bars", end="", flush=True)
        sentiment_df = (
            news_provider.fetch_sentiment(ticker, df.index)
            if news_provider is not None else None
        )
        mtf_df = mtf_fetch(ticker, df.index) if mtf_fetch is not None else None
        df_feat = pipeline.transform(
            df, include_target=True, sentiment_df=sentiment_df, mtf_df=mtf_df
        )
        # Tag each row with its ticker group so the deep model can build
        # sequences per-ticker (prevents cross-ticker sequence contamination).
        df_feat = df_feat.copy()
        df_feat["_ticker_group"] = ticker_group_id
        ticker_group_id += 1
        all_features.append(df_feat)
        print("  ✓")

    if not all_features:
        logger.error("No data available for training. Exiting.")
        sys.exit(1)

    # Combine all tickers, drop NaNs, split into X / y
    import numpy as np
    combined       = pd.concat(all_features, ignore_index=True).dropna()
    ticker_groups  = combined["_ticker_group"].values.astype(np.int32)
    X = combined[pipeline.feature_names]
    y = combined["target"].astype(int)
    print(
        f"\n  ✓  Dataset ready — {len(combined):,} rows × {len(pipeline.feature_names)} features"
        f"  ({len(np.unique(ticker_groups))} tickers, {_t.monotonic() - _t0:.1f}s)"
    )

    # Compute time budget
    train_time_min = getattr(args, "train_time", None)
    budget_seconds = float(train_time_min) * 60.0 if train_time_min is not None else None
    if budget_seconds is not None:
        logger.info(
            "Training time budget: %.1f minutes (%.0f seconds).",
            train_time_min, budget_seconds,
        )

    # ── Incremental (default) vs from-scratch ─────────────────────────────
    scratch     = getattr(args, "scratch", False)
    incremental = not scratch
    model_path  = _model_save_path(settings, model_type)
    model       = _create_model_for_type(settings, model_type)

    if scratch:
        logger.info("From-scratch mode: ignoring any existing saved model.")
    elif model_path.exists():
        try:
            model.load(model_path)
            logger.info("Incremental mode: loaded existing model from %s.", model_path)
        except Exception as exc:
            logger.warning("Could not load existing model (%s) — training from scratch.", exc)
            model = _create_model_for_type(settings, model_type)
            incremental = False
    else:
        logger.info("Incremental mode: no existing model at %s — starting fresh.", model_path)
        incremental = False

    # ── Load previous WF ROC-AUC for cross-session comparison ─────────────
    import json
    meta_path          = model_path.with_suffix(".meta.json")
    baseline_wf_roc: float | None = None
    if not scratch and meta_path.exists():
        try:
            _meta = json.loads(meta_path.read_text())
            baseline_wf_roc = float(_meta.get("wf_mean_roc", 0)) or None
            if baseline_wf_roc:
                print(f"\n  📏  Previous session WF ROC-AUC: {baseline_wf_roc:.4f}")
        except Exception:
            pass

    if not baseline_wf_roc:
        print("\n  🆕  No previous model — training from scratch." if scratch or not model_path.exists()
              else "\n  🆕  No previous WF baseline — starting fresh comparison.")
    print("  Press Ctrl+C at any time to stop and save the best checkpoint.\n")

    # ── Training with safe Ctrl+C interrupt ───────────────────────────────
    interrupted  = False
    wf_mean_roc: float | None = None
    wf_std_roc:  float | None = None

    try:
        logger.info("Running walk-forward validation for %s...", model.name)
        wf_result = trainer.walk_forward_validate(model, X, y, budget_seconds=budget_seconds,
                                                  groups=ticker_groups)
        wf_mean_roc = wf_result.mean_metrics.get("roc_auc")
        wf_std_roc  = wf_result.std_metrics.get("roc_auc")

        logger.info("Training final model on full dataset...")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model = trainer.train_final(
            model, X, y,
            save_path=model_path,
            budget_seconds=budget_seconds,
            incremental=incremental,
            groups=ticker_groups,
        )
        logger.info("Model saved to %s.", model_path)

        # Persist WF metrics for next session
        if wf_mean_roc is not None:
            meta_path.write_text(json.dumps({
                "wf_mean_roc": round(wf_mean_roc, 6),
                "wf_std_roc":  round(wf_std_roc or 0, 6),
                "n_folds":     wf_result.n_folds,
                "model_name":  model.name,
            }, indent=2))

    except KeyboardInterrupt:
        interrupted = True
        print("\n  ⚠️  Training interrupted by Ctrl+C.")
        if model.is_fitted:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(model_path)
            print(f"  ✅  Best model checkpoint saved to {model_path}")
            print("  💡  Resume training anytime — it will continue from this checkpoint.\n")
            # Persist whatever WF metrics we managed to collect before interrupt
            if wf_mean_roc is not None:
                meta_path.write_text(json.dumps({
                    "wf_mean_roc": round(wf_mean_roc, 6),
                    "wf_std_roc":  round(wf_std_roc or 0, 6),
                    "interrupted": True,
                }, indent=2))
        elif model_path.exists():
            print(f"  ℹ️  Interrupted during validation folds — final training had not started.")
            print(f"  ℹ️  Existing model at {model_path} is unchanged and still active.\n")
        else:
            print("  ❌  Model had not yet been fitted — nothing saved.\n")

    # ── Session summary (always printed, even after interrupt) ────────────
    if model.is_fitted:
        _print_session_summary(
            model, model_path,
            wf_mean_roc=wf_mean_roc,
            wf_std_roc=wf_std_roc,
            baseline_wf_roc=baseline_wf_roc,
            interrupted=interrupted,
        )

    # Plot feature importances if available (only on clean completion)
    if not interrupted:
        Path("reports").mkdir(exist_ok=True)
        importances = getattr(model, "feature_importances", None)
        if importances is not None:
            plot_feature_importances(
                importances,
                save_path=Path("reports") / "feature_importances.png",
            )


def run_live(args, settings) -> None:
    """Run live trading loop."""
    from brokers.factory import BrokerFactory  # type: ignore[import]
    from data.factory import DataSourceFactory
    from features.pipeline import FeaturePipeline
    from models.trainer import create_model
    from execution.manager import ExecutionManager
    from strategy.generator import SignalGenerator
    from pathlib import Path

    logger.info("Live trading mode. Press Ctrl+C to stop.")

    # Load pre-trained model using the model's own load() method
    model_type = getattr(args, "model_type", "ml")
    model_path = _model_save_path(settings, model_type)
    if not model_path.exists():
        # Try legacy path for backward compat
        legacy = Path(settings.model.get("save_dir", "models/saved")) / "model.pkl"
        if legacy.exists():
            model_path = legacy
        else:
            logger.error(
                "No trained model found at %s. Run --mode train --model-type %s first.",
                model_path, model_type,
            )
            sys.exit(1)

    model = _create_model_for_type(settings, model_type)
    model.load(model_path)
    logger.info("Loaded model: %s", model.name)

    broker        = BrokerFactory.create(settings)
    data_source   = DataSourceFactory.create(settings)
    pipeline      = FeaturePipeline(settings)
    generator     = SignalGenerator(model, settings, pipeline)
    executor      = ExecutionManager(broker, settings)
    news_provider = _build_news_provider(settings)
    mtf_fetch     = _build_mtf_helper(settings)

    if args.dry_run:
        settings._data["dry_run"] = True

    reset_portfolio = getattr(args, "reset_portfolio", False)
    executor.initialise(reset=reset_portfolio)
    tickers = settings.tickers

    from datetime import date as _date, timezone as _tz
    from data.market_hours import get_market_status, print_market_status

    sentiment_data: dict = {}
    _last_news_date: "_date | None" = None

    try:
        poll_interval = int(settings.get("live_poll_interval_seconds", 60))
        while True:
            # ── Market hours check ────────────────────────────────────────
            now_utc       = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
            mkt_status    = get_market_status(tickers, now_utc)
            print_market_status(mkt_status)

            active_tickers = mkt_status["open_tickers"]

            if not mkt_status["any_open"]:
                print(
                    f"  💤  No markets open right now. "
                    f"{mkt_status['next_open']}.\n"
                    f"  Sleeping 5 minutes…\n"
                )
                time.sleep(300)
                continue

            logger.info(
                "── Polling %d active tickers (%d closed) ──",
                len(active_tickers), len(mkt_status["closed_tickers"]),
            )
            executor.new_day()

            # Fetch OHLCV + MTF for active (open-market) tickers only.
            # Use a shorter period (default 3mo) so features reflect current
            # market conditions rather than year-old price action.
            live_period = str(settings.get("live_period", "3mo"))
            market_data: dict = {}
            mtf_data: dict    = {}
            for ticker in active_tickers:
                try:
                    df = data_source.fetch(ticker, period=live_period)
                    if df is not None and not df.empty:
                        market_data[ticker] = df
                        if mtf_fetch is not None:
                            mtf_data[ticker] = mtf_fetch(ticker, df.index)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Error fetching %s: %s", ticker, exc)

            # Refresh news once per calendar day (for all tickers, not just active)
            today = _date.today()
            if news_provider is not None and today != _last_news_date and market_data:
                logger.info("Fetching daily news sentiment for %d tickers…", len(active_tickers))
                sentiment_data = {}
                for ticker, df in market_data.items():
                    try:
                        sentiment_data[ticker] = news_provider.fetch_sentiment(
                            ticker, df.index
                        )
                    except Exception as exc:
                        logger.warning("News fetch failed for %s: %s", ticker, exc)
                _last_news_date = today
                logger.info(
                    "News sentiment updated for %s (period=%s).", today, live_period
                )

            if market_data:
                # ── Real-time quotes for accurate PnL + fill prices ───────
                # Daily bars only carry yesterday's close.  Fetching fast_info
                # gives the current intraday price so the portfolio snapshot
                # and order execution both reflect what the market is doing NOW.
                from strategy.signal import SignalType as _ST
                live_quotes = _fetch_realtime_quotes(list(market_data.keys()))

                # Build latest_prices: prefer real-time, fall back to last close
                latest_prices = {}
                for t, df in market_data.items():
                    if not df.empty:
                        last_close = float(df["close"].iloc[-1])
                        latest_prices[t] = live_quotes.get(t, last_close)

                executor.update_market_prices(latest_prices)

                signals = generator.generate(
                    market_data,
                    sentiment_data=sentiment_data or None,
                    mtf_data=mtf_data or None,
                )

                # Update signal prices with the real-time quote so position
                # sizing and fill prices reflect the CURRENT intraday price,
                # not yesterday's close.
                for sig in signals:
                    if sig.signal_type != _ST.HOLD and sig.ticker in live_quotes:
                        sig.price = live_quotes[sig.ticker]

                executor.execute_signals(signals)

            executor.print_portfolio_summary()

            logger.info("Sleeping %ds until next poll…", poll_interval)
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
    finally:
        executor.shutdown()


def run_chart(args, settings) -> None:
    """
    Generate a price + signal chart for each ticker using the trained model.
    Fetches recent historical data, runs the model on every bar, and produces
    a PNG with BUY/SELL markers, volume, SMA-20 and model confidence.
    Charts are saved to reports/ and auto-opened.
    """
    from data.factory import DataSourceFactory
    from features.pipeline import FeaturePipeline
    from models.trainer import create_model
    from reporting.plots import plot_price_with_signals
    import numpy as np
    import pandas as pd

    tickers   = [args.ticker] if args.ticker else settings.tickers
    period    = getattr(args, "period", "1y")
    threshold = float(settings.strategy.get("confidence_threshold", 0.60))

    logger.info("Chart mode: tickers=%s, period=%s", tickers, period)

    model_type = getattr(args, "model_type", "ml")
    model_path = _model_save_path(settings, model_type)
    if not model_path.exists():
        legacy = Path(settings.model.get("save_dir", "models/saved")) / "model.pkl"
        if legacy.exists():
            model_path = legacy
        else:
            logger.error("No trained model found at %s. Run --mode train --model-type %s first.", model_path, model_type)
            sys.exit(1)

    model = _create_model_for_type(settings, model_type)
    model.load(model_path)
    logger.info("Loaded model: %s", model.name)

    data_source   = DataSourceFactory.create(settings)
    pipeline      = FeaturePipeline(settings)
    news_provider = _build_news_provider(settings)
    mtf_fetch     = _build_mtf_helper(settings)
    Path("reports").mkdir(exist_ok=True)

    for ticker in tickers:
        logger.info("Generating chart for %s (%s)...", ticker, period)

        df = data_source.fetch(ticker, period=period)
        if df is None or df.empty:
            logger.warning("No data for %s — skipping.", ticker)
            continue

        sentiment_df = (
            news_provider.fetch_sentiment(ticker, df.index)
            if news_provider is not None else None
        )
        mtf_df = mtf_fetch(ticker, df.index) if mtf_fetch is not None else None
        df_feat = pipeline.transform(
            df, include_target=False, sentiment_df=sentiment_df, mtf_df=mtf_df
        ).dropna()
        if df_feat.empty:
            logger.warning("No feature data for %s — skipping.", ticker)
            continue

        X      = df_feat[pipeline.feature_names]
        proba  = model.predict_proba(X)          # (n, 2): [P(down), P(up)]
        p_up   = proba[:, 1]
        p_down = proba[:, 0]

        confidence = pd.Series(np.maximum(p_up, p_down), index=df_feat.index)

        buy_mask   = (p_up   >= threshold) & (p_up   > p_down)
        sell_mask  = (p_down >= threshold) & (p_down > p_up)
        buy_dates  = df_feat.index[buy_mask]
        sell_dates = df_feat.index[sell_mask]

        close_aligned = df["close"].reindex(df_feat.index, method="nearest")
        buy_prices    = close_aligned[buy_mask]  if len(buy_dates)  > 0 else None
        sell_prices   = close_aligned[sell_mask] if len(sell_dates) > 0 else None
        df_chart      = df.reindex(df_feat.index, method="nearest")

        save_path = Path("reports") / f"{ticker}_{period}_signals.png"
        plot_price_with_signals(
            df=df_chart,
            ticker=ticker,
            buy_dates=buy_dates,
            sell_dates=sell_dates,
            buy_prices=buy_prices,
            sell_prices=sell_prices,
            confidence=confidence,
            confidence_threshold=threshold,
            save_path=save_path,
        )
        logger.info(
            "%s: %d BUY, %d SELL signals -> %s",
            ticker, len(buy_dates), len(sell_dates), save_path,
        )


def run_forecast(args, settings) -> None:
    """
    Forecast mode — simulate price paths for each ticker over a given horizon.

    For each ticker:
    1. Fetch latest OHLCV + features.
    2. Run the model on the last bar to get P(up)/P(down).
    3. Simulate 400 Monte Carlo GBM paths using model confidence as drift.
    4. Save forecast JSON to reports/forecasts/.
    5. Generate 3-panel chart (fan chart + confidence + direction bars).
    6. Print a written summary table.
    """
    from data.factory import DataSourceFactory
    from features.pipeline import FeaturePipeline
    from forecasting.simulator import PriceSimulator, save_forecast
    from reporting.plots import plot_forecast_chart

    model_type = getattr(args, "model_type", "ml")
    horizon    = getattr(args, "horizon", "1m")
    tickers    = [args.ticker] if args.ticker else settings.tickers

    logger.info("Forecast mode: tickers=%d, horizon=%s, model=%s", len(tickers), horizon, model_type)

    # Load model
    model_path = _model_save_path(settings, model_type)
    if not model_path.exists():
        logger.error("No model at %s — run --mode train first.", model_path)
        sys.exit(1)
    model = _create_model_for_type(settings, model_type)
    model.load(model_path)

    data_source   = DataSourceFactory.create(settings)
    pipeline      = FeaturePipeline(settings)
    news_provider = _build_news_provider(settings)
    mtf_fetch     = _build_mtf_helper(settings)
    simulator     = PriceSimulator(n_paths=400, drift_scale=0.5, random_seed=42)

    Path("reports/forecasts").mkdir(parents=True, exist_ok=True)

    # Header
    line = "─" * 72
    print(f"\n{line}")
    print(f"  🔮  FORECAST  |  horizon={horizon}  |  model={model_type}  |  date={__import__('datetime').date.today()}")
    print(line)
    print(f"  {'Ticker':<10} {'Direction':<11} {'Pred %':>8} {'AvgConf':>8} {'Bullish':>8} {'Bearish':>8}  Chart")
    print(f"  {'─'*10} {'─'*11} {'─'*8} {'─'*8} {'─'*8} {'─'*8}  {'─'*30}")

    forecast_period = str(settings.get("forecast_period", "3mo"))
    logger.info("Forecast data period: %s", forecast_period)

    for ticker in tickers:
        try:
            df = data_source.fetch(ticker, period=forecast_period)
            if df is None or df.empty:
                logger.warning("No data for %s — skipping.", ticker)
                continue

            sentiment_df = (
                news_provider.fetch_sentiment(ticker, df.index)
                if news_provider is not None else None
            )
            mtf_df = mtf_fetch(ticker, df.index) if mtf_fetch is not None else None
            df_feat = pipeline.transform(
                df, include_target=False, sentiment_df=sentiment_df, mtf_df=mtf_df
            ).dropna()
            if df_feat.empty:
                logger.warning("No features for %s — skipping.", ticker)
                continue

            X_latest = df_feat[pipeline.feature_names]
            result   = simulator.simulate(
                model=model, X_latest=X_latest, df_price=df,
                ticker=ticker, horizon=horizon, model_type=model_type,
            )

            # Save JSON
            save_forecast(result)

            # Chart
            chart_path = Path("reports/forecasts") / f"{ticker.replace('.', '_')}_{result.forecast_date}_{horizon}.png"
            plot_forecast_chart(ticker=ticker, df_history=df, result=result, save_path=chart_path)

            # Print summary row
            dir_icon = {"UP": "▲ UP", "DOWN": "▼ DOWN", "UNCERTAIN": "◆ UNCERT"}.get(result.net_direction, result.net_direction)
            print(
                f"  {ticker:<10} {dir_icon:<11} "
                f"{result.predicted_return_pct:>+7.2f}%  "
                f"{result.avg_confidence:>8.1%}  "
                f"{result.steps_bullish:>8}  "
                f"{result.steps_bearish:>8}  "
                f"{chart_path.name}"
            )

        except Exception as exc:
            logger.error("Forecast failed for %s: %s", ticker, exc)

    print(f"{line}\n")
    print(f"  Forecast JSONs and charts saved to reports/forecasts/")
    print(f"  Run  python main.py --mode evaluate-forecast  after the horizon passes\n")


def run_evaluate_forecast(args, settings) -> None:
    """
    Evaluate past forecasts against actual prices.
    Prints a detailed report, then optionally retrains the model
    on the data from confident-mistake periods.
    """
    from data.factory import DataSourceFactory
    from features.pipeline import FeaturePipeline
    from forecasting.evaluator import ForecastEvaluator
    from models.trainer import ModelTrainer

    model_type = getattr(args, "model_type", "ml")
    retrain    = getattr(args, "retrain", False)

    logger.info("Evaluate-forecast mode: model=%s, retrain=%s", model_type, retrain)

    data_source = DataSourceFactory.create(settings)
    pipeline    = FeaturePipeline(settings)
    evaluator   = ForecastEvaluator(data_source, pipeline)

    evaluations = evaluator.evaluate_all(horizon_elapsed_only=True)
    evaluator.print_report(evaluations)

    if not evaluations:
        return

    if retrain:
        correction = evaluator.build_correction_dataset(evaluations)
        if correction is not None:
            X_corr, y_corr = correction
            model_path = _model_save_path(settings, model_type)
            if not model_path.exists():
                logger.warning("No model to retrain at %s.", model_path)
                return

            model = _create_model_for_type(settings, model_type)
            model.load(model_path)

            trainer = ModelTrainer(settings)
            logger.info(
                "Retraining %s on %d correction rows (incremental)…",
                model.name, len(X_corr),
            )
            model = trainer.train_final(
                model, X_corr, y_corr,
                save_path=model_path,
                incremental=True,
            )
            print(f"\n  ✅  Model retrained on correction data and saved to {model_path}\n")
        else:
            print("\n  ✅  No confident mistakes found — model does not need correction.\n")


def run_report(args, settings) -> None:
    """Generate reports from saved CSV files."""
    import glob
    from reporting.plots import plot_equity_curve
    import pandas as pd

    report_dir = Path("reports")
    equity_files = sorted(glob.glob(str(report_dir / "*_equity.csv")))

    if not equity_files:
        logger.warning("No equity CSV files found in reports/. Run --mode backtest first.")
        return

    for path in equity_files:
        ticker = Path(path).stem.split("_")[0]
        equity = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
        plot_equity_curve(equity, ticker=ticker)
        logger.info("Report generated for %s.", ticker)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Load settings
    from config import load_settings
    from pathlib import Path
    settings = load_settings(Path(args.config))

    # Apply verbose flag — elevate console to INFO if requested
    if getattr(args, "verbose", False):
        _console_handler.setLevel(logging.INFO)

    logger.info("Trading Bot starting — mode=%s, config=%s", args.mode, args.config)

    dispatch = {
        "backtest":          run_backtest,
        "train":             run_train,
        "live":              run_live,
        "chart":             run_chart,
        "report":            run_report,
        "forecast":          run_forecast,
        "evaluate-forecast": run_evaluate_forecast,
    }

    handler = dispatch.get(args.mode)
    if handler is None:
        logger.error("Unknown mode: %s", args.mode)
        sys.exit(1)

    try:
        handler(args, settings)
    except Exception as exc:
        logger.exception("Fatal error in mode '%s': %s", args.mode, exc)
        sys.exit(1)

    logger.info("Trading Bot finished.")


if __name__ == "__main__":
    main()
