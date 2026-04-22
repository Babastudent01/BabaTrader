"""
execution/manager.py
Order execution manager.
Orchestrates signal → risk check → position sizing → order placement.
"""
from __future__ import annotations

import csv
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from brokers.base import BrokerInterface, OrderRequest, OrderResult, OrderSide, OrderType
from config import Settings
from risk.controls import RiskControls
from risk.sizing import PositionSizer
from strategy.signal import Signal, SignalType

logger = logging.getLogger(__name__)


class ExecutionManager:
    """
    Converts trading signals into broker orders.

    Workflow for each signal:
    1. Portfolio-level risk check (drawdown, daily loss, max positions).
    2. Per-position risk check (concentration, duplicate, cash).
    3. Position sizing (shares to buy).
    4. Order placement via broker.
    5. Stop-loss / take-profit order placement (if configured).

    All actions are logged. In DRY_RUN mode, orders are logged but not sent.
    """

    # Default path for the portfolio state JSON file
    STATE_PATH = Path("reports/portfolio_state.json")

    def __init__(
        self,
        broker: BrokerInterface,
        settings: Settings,
    ) -> None:
        self._broker   = broker
        self._settings = settings
        self._sizer    = PositionSizer(settings)
        self._controls = RiskControls(settings)
        self._dry_run: bool = bool(settings.get("dry_run", False))
        self._order_log: list[dict[str, Any]] = []
        self._start_equity: float = 0.0
        self._snapshot_path = Path("reports/portfolio_snapshots.csv")
        self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        # Write CSV header if file is new
        if not self._snapshot_path.exists():
            with self._snapshot_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "total_equity", "cash", "portfolio_value",
                    "unrealised_pnl", "realised_pnl", "net_pnl",
                    "open_positions", "positions_detail",
                ])

        if self._dry_run:
            logger.warning("ExecutionManager: DRY RUN mode — no real orders will be placed.")

    def initialise(self, reset: bool = False) -> None:
        """
        Connect to broker and initialise risk controls.

        Parameters
        ----------
        reset : bool
            If True, start with a clean portfolio (100 000 € cash, no positions)
            even if a saved state file exists.
            If False (default), restore the previous session's portfolio state.
        """
        self._broker.connect()

        # ── Portfolio state persistence ────────────────────────────────────
        if reset:
            # Wipe any saved state and start fresh
            if self.STATE_PATH.exists():
                self.STATE_PATH.unlink()
                print(
                    f"\n  🔄  --reset-portfolio: deleted {self.STATE_PATH} "
                    "— starting with clean portfolio.\n"
                )
            if hasattr(self._broker, "reset"):
                self._broker.reset()
        else:
            # Try to restore previous session
            if hasattr(self._broker, "load_state"):
                self._broker.load_state(self.STATE_PATH)

        account = self._broker.get_account_info()
        self._controls.initialise(account.total_equity)
        self._start_equity = account.total_equity
        logger.info(
            "ExecutionManager initialised: broker=%s, equity=%.2f %s, reset=%s.",
            self._broker.name, account.total_equity, account.currency, reset,
        )

    def shutdown(self) -> None:
        """Save portfolio state and disconnect from broker."""
        # Persist positions + cash before disconnecting
        if hasattr(self._broker, "save_state"):
            try:
                self._broker.save_state(self.STATE_PATH)
                print(f"\n  💾  Portfolio state saved → {self.STATE_PATH}")
                print("  💡  Next time you run --mode live it will resume from here.")
                print("      Use  --reset-portfolio  to start fresh.\n")
            except Exception as exc:
                logger.error("Failed to save portfolio state: %s", exc)
        self._broker.disconnect()
        logger.info("ExecutionManager shut down.")

    def update_market_prices(self, prices: dict[str, float]) -> None:
        """
        Push latest close prices to the broker so unrealised PnL stays current.
        Must be called each poll cycle before printing the portfolio summary.
        """
        if hasattr(self._broker, "update_prices"):
            self._broker.update_prices(prices)

    def print_portfolio_summary(self) -> None:
        """
        Always-visible portfolio snapshot printed to stdout (bypasses log level).
        Shows equity, cash, open positions, and overall P&L.
        """
        account   = self._broker.get_account_info()
        positions = self._broker.get_positions()

        total_unrealised = sum(p.unrealised_pnl for p in positions.values())
        total_realised   = sum(
            getattr(p, "realised_pnl", 0.0) for p in positions.values()
        )
        start_equity = self._start_equity if self._start_equity > 0 else account.total_equity
        net_pnl      = account.total_equity - start_equity

        # Compute long/short exposure separately for a clearer display.
        # When you open a short you receive cash proceeds BUT you also owe those
        # shares back — so raw cash is inflated by short proceeds.
        # "Available cash" = total_equity − long_exposure  (truly uncommitted capital).
        long_exposure  = sum(
            p.quantity * p.current_price
            for p in positions.values() if p.quantity > 0
        )
        short_exposure = sum(
            abs(p.quantity) * p.current_price
            for p in positions.values() if p.quantity < 0
        )
        available_cash = account.total_equity - long_exposure   # free for new trades

        line = "─" * 62
        print(f"\n{line}")
        print(f"  💼  PORTFOLIO SNAPSHOT  —  {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(line)
        print(f"  Total equity   : {account.total_equity:>12,.2f} {account.currency}")
        print(f"  Available cash : {available_cash:>12,.2f} {account.currency}  ← free for new trades")
        print(f"  Long exposure  : {long_exposure:>+12,.2f} {account.currency}  ← locked in long positions")
        if short_exposure > 0:
            print(f"  Short obligat. : {-short_exposure:>+12,.2f} {account.currency}  ← must cover to close shorts")
        print(f"  Unrealised P&L : {total_unrealised:>+12,.2f} {account.currency}")
        print(f"  Realised P&L   : {total_realised:>+12,.2f} {account.currency}")
        print(f"  Net P&L (total): {net_pnl:>+12,.2f} {account.currency}")

        if positions:
            print(f"\n  {'Ticker':<10} {'Type':>5} {'Qty':>9} {'Avg Cost':>10} {'Price':>10} {'Unreal PnL':>12}")
            print(f"  {'-'*10} {'-'*5} {'-'*9} {'-'*10} {'-'*10} {'-'*12}")
            for ticker, pos in sorted(positions.items()):
                side_lbl = "SHORT" if pos.quantity < 0 else "LONG"
                pnl_str  = f"{pos.unrealised_pnl:>+12,.2f}"
                print(
                    f"  {ticker:<10} {side_lbl:>5} {pos.quantity:>9,.2f} {pos.avg_cost:>10,.4f} "
                    f"{pos.current_price:>10,.4f} {pnl_str}"
                )
        else:
            print("\n  No open positions.")
        print(line + "\n")

        # ── Persist snapshot to CSV ───────────────────────────────────────
        positions_detail = "; ".join(
            f"{t}:{p.quantity:.2f}@{p.current_price:.4f}(pnl={p.unrealised_pnl:+.2f})"
            for t, p in sorted(positions.items())
        )
        try:
            with self._snapshot_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    f"{account.total_equity:.2f}",
                    f"{account.cash:.2f}",
                    f"{account.portfolio_value:.2f}",
                    f"{total_unrealised:+.2f}",
                    f"{total_realised:+.2f}",
                    f"{net_pnl:+.2f}",
                    len(positions),
                    positions_detail,
                ])
        except Exception as exc:
            logger.warning("Could not write portfolio snapshot: %s", exc)

    def execute_signals(self, signals: list[Signal]) -> list[OrderResult]:
        """
        Execute a list of trading signals.

        Parameters
        ----------
        signals : list[Signal]
            Signals to execute (BUY/SELL only; HOLD signals are ignored).

        Returns
        -------
        list[OrderResult]
            Results for all orders that were attempted.
        """
        results: list[OrderResult] = []

        for signal in signals:
            if signal.signal_type == SignalType.HOLD:
                continue

            try:
                result = self._execute_signal(signal)
                if result is not None:
                    results.append(result)
            except Exception as exc:  # noqa: BLE001
                logger.error("Error executing signal %s: %s", signal, exc)

        return results

    def execute_signal(self, signal: Signal) -> OrderResult | None:
        """Execute a single signal. Returns None if blocked by risk controls."""
        if signal.signal_type == SignalType.HOLD:
            return None
        return self._execute_signal(signal)

    def new_day(self) -> None:
        """Call at the start of each trading day to reset daily risk tracking."""
        account = self._broker.get_account_info()
        self._controls.new_day(account.total_equity)

    @property
    def order_log(self) -> list[dict[str, Any]]:
        """Return the full order log."""
        return list(self._order_log)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _execute_signal(self, signal: Signal) -> OrderResult | None:
        """Core execution logic for a single signal."""
        logger.info("Processing signal: %s", signal)

        # ── Portfolio-level risk check ────────────────────────────────────────
        portfolio_check = self._controls.check_portfolio(self._broker)
        if not portfolio_check.passed:
            logger.warning(
                "Signal BLOCKED (portfolio risk): %s — %s",
                signal.ticker, portfolio_check.reason,
            )
            return None

        if signal.signal_type == SignalType.BUY:
            return self._execute_buy(signal)
        elif signal.signal_type == SignalType.SELL:
            return self._execute_sell(signal)
        return None

    def _execute_buy(self, signal: Signal) -> OrderResult | None:
        """
        Execute a BUY signal.

        Decision tree:
        - Existing SHORT -> COVER (buy-to-close)
        - Existing LONG  -> skip (already long)
        - No position    -> open new LONG
        """
        positions = self._broker.get_positions()
        existing  = positions.get(signal.ticker)

        if existing is not None and existing.quantity < 0:
            return self._cover_short(signal)

        if existing is not None and existing.quantity > 0:
            logger.debug("BUY %s: already long — skipping.", signal.ticker)
            return None

        account  = self._broker.get_account_info()
        quantity = self._sizer.compute_quantity(signal, account.total_equity)
        if quantity <= 0:
            logger.warning("BUY %s: position sizer returned 0 shares — skipping.", signal.ticker)
            return None

        pos_check = self._controls.check_new_position(
            signal.ticker, quantity, signal.price, self._broker
        )
        if not pos_check.passed:
            logger.warning("BUY %s BLOCKED (position risk): %s", signal.ticker, pos_check.reason)
            return None

        stop_loss   = self._sizer.compute_stop_loss(signal.price, signal.atr)
        take_profit = self._sizer.compute_take_profit(signal.price)

        order = OrderRequest(
            ticker=signal.ticker,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
            stop_loss=stop_loss,
            take_profit=take_profit,
            client_order_id=str(uuid.uuid4())[:8],
        )
        return self._place_order(order, signal, label="LONG")

    def _execute_sell(self, signal: Signal) -> OrderResult | None:
        """
        Execute a SELL signal.

        Decision tree:
        - Existing LONG  -> close it (sell-to-close)
        - No position    -> open SHORT (if enabled)
        - Existing SHORT -> skip (already short)
        """
        positions = self._broker.get_positions()
        existing  = positions.get(signal.ticker)

        if existing is not None and existing.quantity > 0:
            order = OrderRequest(
                ticker=signal.ticker,
                side=OrderSide.SELL,
                quantity=existing.quantity,
                order_type=OrderType.MARKET,
                client_order_id=str(uuid.uuid4())[:8],
            )
            return self._place_order(order, signal, label="SELL")

        if existing is not None and existing.quantity < 0:
            logger.debug("SELL %s: already short — skipping.", signal.ticker)
            return None

        allow_short = bool(self._settings.execution.get("allow_short_selling", True))
        if not allow_short:
            logger.debug("SELL %s: no position and short selling disabled — skipping.", signal.ticker)
            return None
        return self._open_short(signal)

    def _open_short(self, signal: Signal) -> OrderResult | None:
        """Open a new SHORT position on a confident DOWN signal."""
        account  = self._broker.get_account_info()
        quantity = self._sizer.compute_quantity(signal, account.total_equity)
        if quantity <= 0:
            logger.warning("SHORT %s: position sizer returned 0 shares — skipping.", signal.ticker)
            return None

        pos_check = self._controls.check_new_position(
            signal.ticker, quantity, signal.price, self._broker
        )
        if not pos_check.passed:
            logger.warning("SHORT %s BLOCKED (position risk): %s", signal.ticker, pos_check.reason)
            return None

        order = OrderRequest(
            ticker=signal.ticker,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType.MARKET,
            client_order_id=str(uuid.uuid4())[:8],
        )
        return self._place_order(order, signal, label="SHORT")

    def _cover_short(self, signal: Signal) -> OrderResult | None:
        """Cover (close) an existing SHORT position on a BUY signal."""
        positions = self._broker.get_positions()
        pos = positions.get(signal.ticker)
        if pos is None or pos.quantity >= 0:
            logger.debug("COVER %s: no short position found — skipping.", signal.ticker)
            return None

        order = OrderRequest(
            ticker=signal.ticker,
            side=OrderSide.BUY,
            quantity=abs(pos.quantity),
            order_type=OrderType.MARKET,
            client_order_id=str(uuid.uuid4())[:8],
        )
        return self._place_order(order, signal, label="COVER")

    def _place_order(
        self,
        order: OrderRequest,
        signal: Signal,
        label: str | None = None,
    ) -> OrderResult | None:
        """
        Place an order via the broker (or log it in dry-run mode).

        label : "LONG" | "SELL" | "SHORT" | "COVER"
            Human-readable action shown in the fill line.  Defaults to the
            raw OrderSide value when not supplied.
        """
        _lbl = label or order.side.value
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "ticker": order.ticker,
            "side": order.side.value,
            "action": _lbl,
            "quantity": order.quantity,
            "signal_confidence": signal.confidence,
            "dry_run": self._dry_run,
        }

        account = self._broker.get_account_info()

        if self._dry_run:
            cost = order.quantity * signal.price
            print(
                f"\n  🔔 DRY RUN  {_lbl:5s}  {order.ticker:<10}  "
                f"{order.quantity:>8,.2f} shares  @  {signal.price:>10,.4f}  "
                f"~  {cost:>10,.2f}  |  conf={signal.confidence:.1%}  "
                f"|  equity={account.total_equity:,.2f}"
            )
            log_entry["status"] = "DRY_RUN"
            self._order_log.append(log_entry)
            return None

        result = self._broker.place_order(order)
        log_entry.update({
            "order_id": result.order_id,
            "fill_price": result.avg_fill_price,
            "filled_qty": result.filled_quantity,
            "commission": result.commission,
            "status": result.status.value,
        })
        self._order_log.append(log_entry)

        # Always-visible fill confirmation (bypasses log level)
        account_after = self._broker.get_account_info()
        _icons = {"LONG": "🟢", "SELL": "🔴", "SHORT": "🔻", "COVER": "🟩"}
        icon = _icons.get(_lbl, "🟢" if result.side == OrderSide.BUY else "🔴")

        # Show TTF for French large-cap BUY orders (ETFs in ttf_exempt_tickers are excluded)
        _ttf_pct    = float(self._settings.execution.get("ttf_pct", 0.003))
        _ttf_suffix = str(self._settings.execution.get("ttf_suffix", ".PA"))
        _ttf_exempt = set(self._settings.execution.get("ttf_exempt_tickers", []))
        _ttf = (
            result.filled_quantity * result.avg_fill_price * _ttf_pct
            if (result.side == OrderSide.BUY
                and result.ticker.endswith(_ttf_suffix)
                and result.ticker not in _ttf_exempt)
            else 0.0
        )
        _ttf_str = f"  ttf={_ttf:.2f} €" if _ttf > 0 else ""

        print(
            f"\n  {icon} {_lbl:5s}  {result.ticker:<10}  "
            f"{result.filled_quantity:>8,.2f} shares  @  {result.avg_fill_price:>10,.4f}  "
            f"=  {result.filled_quantity * result.avg_fill_price:>10,.2f}  "
            f"|  commission={result.commission:.2f} €{_ttf_str}  "
            f"|  equity={account_after.total_equity:,.2f}  "
            f"|  order={result.order_id}"
        )

        logger.info(
            "Order %s: %s %s x%.0f filled @ %.4f (commission=%.2f).",
            result.order_id, _lbl, result.ticker,
            result.filled_quantity, result.avg_fill_price, result.commission,
        )
        return result
