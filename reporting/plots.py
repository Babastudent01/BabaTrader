"""
reporting/plots.py
Visualisation utilities for backtest results.
Generates equity curve, drawdown, monthly returns heatmap, and trade distribution plots.
Requires matplotlib and seaborn (optional — gracefully skips if not installed).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend (safe for servers)
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not installed — plots will be skipped.")

try:
    import seaborn as sns
    _SEABORN_AVAILABLE = True
except ImportError:
    _SEABORN_AVAILABLE = False


def _check_matplotlib() -> bool:
    if not _MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib is required for plotting. Install with: pip install matplotlib")
        return False
    return True


def plot_equity_curve(
    equity_curve: pd.Series,
    ticker: str = "",
    benchmark: pd.Series | None = None,
    save_path: Path | None = None,
) -> None:
    """
    Plot the equity curve with optional benchmark comparison.

    Parameters
    ----------
    equity_curve : pd.Series
        DatetimeIndex series of portfolio equity values.
    ticker : str
        Ticker label for the plot title.
    benchmark : pd.Series | None
        Optional benchmark equity curve (e.g. buy-and-hold).
    save_path : Path | None
        If provided, save the figure to this path. Otherwise display.
    """
    if not _check_matplotlib() or equity_curve.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    # ── Equity curve ──────────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(equity_curve.index, equity_curve.values, label="Strategy", color="#2196F3", linewidth=1.5)

    if benchmark is not None and not benchmark.empty:
        # Normalise benchmark to same starting value
        bench_norm = benchmark / benchmark.iloc[0] * equity_curve.iloc[0]
        ax1.plot(bench_norm.index, bench_norm.values, label="Buy & Hold",
                 color="#FF9800", linewidth=1.2, linestyle="--", alpha=0.8)

    ax1.set_ylabel("Portfolio Value (€)")
    ax1.set_title(f"Equity Curve — {ticker}" if ticker else "Equity Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"€{x:,.0f}"))

    # ── Drawdown ──────────────────────────────────────────────────────────────
    ax2 = axes[1]
    rolling_max = equity_curve.cummax()
    drawdown    = (equity_curve - rolling_max) / rolling_max * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="#F44336", alpha=0.4, label="Drawdown")
    ax2.plot(drawdown.index, drawdown.values, color="#F44336", linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_monthly_returns_heatmap(
    equity_curve: pd.Series,
    ticker: str = "",
    save_path: Path | None = None,
) -> None:
    """
    Plot a monthly returns heatmap (year × month).

    Parameters
    ----------
    equity_curve : pd.Series
        DatetimeIndex series of portfolio equity values.
    ticker : str
        Ticker label for the plot title.
    save_path : Path | None
        Save path or None to display.
    """
    if not _check_matplotlib() or equity_curve.empty:
        return

    monthly = equity_curve.resample("ME").last().pct_change().dropna() * 100
    df = monthly.to_frame("return")
    df["year"]  = df.index.year
    df["month"] = df.index.month

    pivot = df.pivot_table(index="year", columns="month", values="return", aggfunc="first")
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    pivot.columns = [month_names.get(c, str(c)) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.6)))

    if _SEABORN_AVAILABLE:
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            center=0,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "Return (%)"},
        )
    else:
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        plt.colorbar(im, ax=ax, label="Return (%)")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not pd.isna(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)

    ax.set_title(f"Monthly Returns (%) — {ticker}" if ticker else "Monthly Returns (%)")
    ax.set_xlabel("")
    ax.set_ylabel("Year")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_trade_distribution(
    trades: list[Any],
    ticker: str = "",
    save_path: Path | None = None,
) -> None:
    """
    Plot trade PnL distribution histogram and win/loss breakdown.

    Parameters
    ----------
    trades : list[Trade]
        List of completed trades.
    ticker : str
        Ticker label.
    save_path : Path | None
        Save path or None to display.
    """
    if not _check_matplotlib() or not trades:
        return

    pnls = [t.pnl for t in trades]
    pnl_pcts = [t.pnl_pct * 100 for t in trades]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── PnL histogram ─────────────────────────────────────────────────────────
    ax1 = axes[0]
    colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnls]
    ax1.bar(range(len(pnls)), pnls, color=colors, alpha=0.7, edgecolor="white")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("Trade #")
    ax1.set_ylabel("PnL (€)")
    ax1.set_title(f"Trade PnL — {ticker}" if ticker else "Trade PnL")
    ax1.grid(True, alpha=0.3, axis="y")

    # ── PnL % distribution ────────────────────────────────────────────────────
    ax2 = axes[1]
    n_bins = min(30, max(10, len(pnl_pcts) // 3))
    ax2.hist(pnl_pcts, bins=n_bins, color="#2196F3", alpha=0.7, edgecolor="white")
    ax2.axvline(0, color="black", linewidth=1.0)
    ax2.axvline(sum(pnl_pcts) / len(pnl_pcts), color="#FF9800",
                linewidth=1.5, linestyle="--", label=f"Mean: {sum(pnl_pcts)/len(pnl_pcts):.2f}%")
    ax2.set_xlabel("PnL (%)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("PnL Distribution (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_price_with_signals(
    df: pd.DataFrame,
    ticker: str = "",
    buy_dates: pd.DatetimeIndex | None = None,
    sell_dates: pd.DatetimeIndex | None = None,
    buy_prices: pd.Series | None = None,
    sell_prices: pd.Series | None = None,
    confidence: pd.Series | None = None,
    confidence_threshold: float = 0.60,
    save_path: Path | None = None,
) -> None:
    """
    Plot closing price with BUY / SELL signal markers and optional confidence.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex.
    ticker : str
        Ticker symbol for the title.
    buy_dates / sell_dates : DatetimeIndex
        Timestamps where BUY or SELL signals were generated.
    buy_prices / sell_prices : pd.Series
        Price at each BUY / SELL signal.
    confidence : pd.Series
        Model confidence per bar (aligned with df.index).
    confidence_threshold : float
        Threshold line drawn on the confidence subplot.
    save_path : Path | None
        Where to save the PNG. Opened automatically if provided.
    """
    if not _check_matplotlib() or df.empty:
        return

    n_panels = 3 if confidence is not None else 2
    height_ratios = [4, 1, 1] if n_panels == 3 else [4, 1]

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(16, 4 + n_panels * 2),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if n_panels == 2:
        axes = list(axes) + [None]

    ax_price, ax_vol, ax_conf = axes[0], axes[1], axes[2]

    close = df["close"] if "close" in df.columns else df.iloc[:, 3]

    # ── Price line ────────────────────────────────────────────────────────────
    ax_price.plot(close.index, close.values, color="#1565C0", linewidth=1.4, label="Close", zorder=2)

    # Shaded background: green above 20-day SMA, red below
    sma20 = close.rolling(20).mean()
    ax_price.plot(sma20.index, sma20.values, color="#90A4AE", linewidth=0.8,
                  linestyle="--", label="SMA-20", alpha=0.7, zorder=1)
    ax_price.fill_between(
        close.index, close.values, sma20.values,
        where=(close.values >= sma20.values),
        interpolate=True, alpha=0.08, color="#4CAF50",
    )
    ax_price.fill_between(
        close.index, close.values, sma20.values,
        where=(close.values < sma20.values),
        interpolate=True, alpha=0.08, color="#F44336",
    )

    # BUY markers (green triangles up)
    if buy_dates is not None and len(buy_dates) > 0:
        bp = buy_prices if buy_prices is not None else close.reindex(buy_dates, method="nearest")
        ax_price.scatter(
            buy_dates, bp.values,
            marker="^", color="#00C853", s=80, zorder=5, label="BUY", linewidths=0.5,
        )
        for ts, price in zip(buy_dates, bp.values):
            ax_price.annotate(
                f" B\n${price:.0f}", (ts, price),
                textcoords="offset points", xytext=(0, 8),
                fontsize=6, color="#00C853", ha="center",
            )

    # SELL markers (red triangles down)
    if sell_dates is not None and len(sell_dates) > 0:
        sp = sell_prices if sell_prices is not None else close.reindex(sell_dates, method="nearest")
        ax_price.scatter(
            sell_dates, sp.values,
            marker="v", color="#D50000", s=80, zorder=5, label="SELL", linewidths=0.5,
        )
        for ts, price in zip(sell_dates, sp.values):
            ax_price.annotate(
                f" S\n${price:.0f}", (ts, price),
                textcoords="offset points", xytext=(0, -14),
                fontsize=6, color="#D50000", ha="center",
            )

    ax_price.set_ylabel("Price (USD)")
    ax_price.set_title(f"Price + Model Signals — {ticker}" if ticker else "Price + Model Signals")
    ax_price.legend(loc="upper left", fontsize=8)
    ax_price.grid(True, alpha=0.25)
    ax_price.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.2f}"))

    # ── Volume bars ───────────────────────────────────────────────────────────
    if "volume" in df.columns:
        vol = df["volume"]
        bar_colors = []
        for i in range(len(close)):
            if i == 0:
                bar_colors.append("#90A4AE")
            else:
                bar_colors.append("#4CAF50" if close.iloc[i] >= close.iloc[i - 1] else "#F44336")
        ax_vol.bar(vol.index, vol.values, color=bar_colors, alpha=0.6, width=0.8)
        ax_vol.set_ylabel("Volume", fontsize=8)
        ax_vol.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
        ax_vol.grid(True, alpha=0.2)

    # ── Model confidence ──────────────────────────────────────────────────────
    if confidence is not None and ax_conf is not None:
        ax_conf.plot(confidence.index, confidence.values, color="#7B1FA2",
                     linewidth=1.0, label="Confidence")
        ax_conf.axhline(confidence_threshold, color="#FF6F00", linewidth=0.8,
                        linestyle="--", label=f"Threshold ({confidence_threshold:.0%})")
        ax_conf.fill_between(
            confidence.index, confidence.values, confidence_threshold,
            where=(confidence.values >= confidence_threshold),
            alpha=0.15, color="#7B1FA2",
        )
        ax_conf.set_ylim(0.4, 1.0)
        ax_conf.set_ylabel("Confidence", fontsize=8)
        ax_conf.legend(loc="upper left", fontsize=7)
        ax_conf.grid(True, alpha=0.2)

    # ── X-axis formatting ─────────────────────────────────────────────────────
    bottom_ax = ax_conf if ax_conf is not None else ax_vol
    bottom_ax.set_xlabel("Date")
    bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    bottom_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(bottom_ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    _save_or_show(fig, save_path)

    # Auto-open file on Windows
    if save_path is not None:
        try:
            import os
            os.startfile(str(save_path))
        except Exception:
            pass


def plot_feature_importances(
    importances: pd.Series,
    top_n: int = 20,
    ticker: str = "",
    save_path: Path | None = None,
) -> None:
    """
    Plot top-N feature importances as a horizontal bar chart.

    Parameters
    ----------
    importances : pd.Series
        Feature name → importance value (sorted descending).
    top_n : int
        Number of top features to display.
    ticker : str
        Label for the plot title.
    save_path : Path | None
        Save path or None to display.
    """
    if not _check_matplotlib() or importances.empty:
        return

    top = importances.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    ax.barh(top.index[::-1], top.values[::-1], color="#2196F3", alpha=0.8)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances — {ticker}" if ticker else f"Top {top_n} Feature Importances")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_forecast_chart(
    ticker:      str,
    df_history:  "pd.DataFrame",
    result:      Any,              # forecasting.simulator.SimulationResult
    save_path:   "Path | None" = None,
    history_bars: int = 60,
) -> None:
    """
    3-panel forecast chart:
      Top    — Historical close price (last *history_bars* days) + simulated
               fan chart (95% / 80% bands + mean path).
      Middle — Per-step model confidence decay curve.
      Bottom — Predicted direction per step (green=UP, red=DOWN, grey=UNCERTAIN).

    Parameters
    ----------
    ticker : str
    df_history : pd.DataFrame
        Full historical OHLCV DataFrame (columns: open, high, low, close, volume).
    result : SimulationResult
        Output from PriceSimulator.simulate().
    save_path : Path | None
    history_bars : int
        How many historical bars to show before the forecast starts.
    """
    if not _check_matplotlib():
        return

    import numpy as np
    import pandas as pd

    # ── Prepare data ──────────────────────────────────────────────────────
    hist = df_history["close"].iloc[-history_bars:]
    hist_index = list(range(-len(hist), 0))          # negative x = past
    fc_index   = list(range(0, result.horizon_days + 1))   # 0 = today, 1..N = future

    mean_path = result.mean_path
    lower_80  = result.lower_80
    upper_80  = result.upper_80
    lower_95  = result.lower_95
    upper_95  = result.upper_95

    confidences = [sp.confidence for sp in result.step_predictions]
    directions  = [sp.direction  for sp in result.step_predictions]
    step_x      = list(range(1, result.horizon_days + 1))

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10),
        gridspec_kw={"height_ratios": [4, 1.5, 1]},
        sharex=False,
    )
    ax_price, ax_conf, ax_dir = axes

    # ─ Panel 1: price ─────────────────────────────────────────────────────
    ax_price.plot(hist_index, hist.values, color="#1565C0", linewidth=1.8,
                  label="Historical close")
    ax_price.axvline(0, color="#555555", linewidth=1, linestyle="--", alpha=0.7)

    # Fan chart (95% band lightest, 80% darker)
    ax_price.fill_between(fc_index, lower_95, upper_95,
                          alpha=0.12, color="#7B1FA2", label="95% band")
    ax_price.fill_between(fc_index, lower_80, upper_80,
                          alpha=0.22, color="#7B1FA2", label="80% band")
    ax_price.plot(fc_index, mean_path, color="#7B1FA2", linewidth=2,
                  linestyle="--", label="Mean forecast")

    # Label net direction
    dir_color = {"UP": "#2E7D32", "DOWN": "#C62828", "UNCERTAIN": "#555555"}
    dir_emoji = {"UP": "▲", "DOWN": "▼", "UNCERTAIN": "◆"}
    dc = dir_color.get(result.net_direction, "#555555")
    de = dir_emoji.get(result.net_direction, "◆")
    ax_price.set_title(
        f"{ticker}  —  {result.horizon_label} Forecast  "
        f"{de} {result.net_direction}  "
        f"(avg conf {result.avg_confidence:.0%},  "
        f"pred return {result.predicted_return_pct:+.2f}%)\n"
        f"Forecast date: {result.forecast_date}",
        fontsize=12, color=dc,
    )
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left", fontsize=8)
    ax_price.grid(True, alpha=0.3)
    ax_price.axvspan(0, result.horizon_days, alpha=0.04, color="#7B1FA2")

    # ─ Panel 2: confidence ────────────────────────────────────────────────
    ax_conf.plot(step_x, confidences, color="#E65100", linewidth=1.6)
    ax_conf.fill_between(step_x, confidences, alpha=0.25, color="#E65100")
    ax_conf.axhline(0.6, color="#999", linestyle=":", linewidth=1)
    ax_conf.set_ylim(0, 1)
    ax_conf.set_ylabel("Confidence")
    ax_conf.grid(True, alpha=0.3)
    ax_conf.set_xticks(step_x[::max(1, len(step_x) // 10)])

    # ─ Panel 3: direction bars ────────────────────────────────────────────
    bar_colors = [
        dir_color.get(d, "#555555") for d in directions
    ]
    bar_heights = [1.0] * len(directions)
    ax_dir.bar(step_x, bar_heights, color=bar_colors, alpha=0.7, width=0.8)
    ax_dir.set_yticks([])
    ax_dir.set_xlabel("Days ahead")
    ax_dir.set_ylabel("Direction")
    ax_dir.set_xticks(step_x[::max(1, len(step_x) // 10)])
    ax_dir.grid(True, alpha=0.2, axis="x")

    # Legend patches for direction panel
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#2E7D32", label="UP"),
        Patch(facecolor="#C62828", label="DOWN"),
        Patch(facecolor="#555555", label="UNCERTAIN"),
    ]
    ax_dir.legend(handles=legend_els, loc="upper right", fontsize=7, ncol=3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_or_show(fig: Any, save_path: Path | None) -> None:
    """Save figure to file or display it."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Plot saved to %s.", save_path)
    else:
        plt.show()
    plt.close(fig)
