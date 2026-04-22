"""
brokers/mock.py
Mock broker for TEST mode / paper trading.
Simulates order execution with configurable commission and slippage.
No real orders are ever sent.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

import pytz

from brokers.base import (
    AccountInfo,
    BrokerInterface,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    Position,
)
from config import Settings

logger = logging.getLogger(__name__)


class MockBroker(BrokerInterface):
    """
    Simulated broker for paper trading and backtesting.

    Features:
    - Virtual cash account with configurable initial balance.
    - Market orders filled immediately at current price + slippage.
    - Commission deducted from cash on each trade.
    - Tracks open positions and realised PnL.
    - Never sends any real orders.
    """

    def __init__(self, settings: Settings) -> None:
        exec_cfg = settings.execution
        bt_cfg   = settings.backtesting

        # Use a smaller starting cash for live/paper trading to keep it realistic.
        # --mode live  →  live_initial_cash  (default €1,000)
        # --mode backtest → initial_cash     (default €100,000)
        _mode = str(settings.get("mode", "TEST")).upper()
        if _mode == "LIVE":
            self._initial_cash: float = float(bt_cfg.get("live_initial_cash", 1_000.0))
        else:
            self._initial_cash: float = float(bt_cfg.get("initial_cash", 100_000.0))
        self._cash: float            = self._initial_cash
        self._commission_pct: float  = float(exec_cfg.get("commission_pct", 0.001))
        self._commission_min: float  = float(exec_cfg.get("commission_min", 1.0))
        self._slippage_pct: float    = float(exec_cfg.get("slippage_pct", 0.0005))
        self._short_margin_pct: float = float(exec_cfg.get("short_margin_pct", 1.0))
        # TTF — Taxe sur les Transactions Financières (France)
        # Applied to BUY orders on French large-cap equities (.PA suffix by default).
        # ETFs listed on Euronext Paris are legally exempt — list them in ttf_exempt_tickers.
        self._ttf_pct: float    = float(exec_cfg.get("ttf_pct", 0.003))
        self._ttf_suffix: str   = str(exec_cfg.get("ttf_suffix", ".PA"))
        self._ttf_exempt: set[str] = set(exec_cfg.get("ttf_exempt_tickers", []))

        self._positions: dict[str, Position] = {}
        self._orders: dict[str, OrderResult] = {}
        self._trade_log: list[dict[str, Any]] = []
        # Latest known market prices (updated every poll cycle via update_prices).
        # Separate from positions so NEW buys also get a valid fill price.
        self._market_prices: dict[str, float] = {}

        tz_name = settings.get("timezone", "Europe/Paris")
        self._tz = pytz.timezone(tz_name)
        self._connected = False

    # ── BrokerInterface implementation ────────────────────────────────────────

    @property
    def name(self) -> str:
        return "MockBroker"

    @property
    def is_market_open(self) -> bool:
        """Mock broker is always 'open'."""
        return True

    def connect(self) -> None:
        self._connected = True
        logger.info("MockBroker connected (virtual account, cash=%.2f).", self._cash)

    def disconnect(self) -> None:
        self._connected = False
        logger.info("MockBroker disconnected.")

    def get_account_info(self) -> AccountInfo:
        portfolio_value = sum(
            p.quantity * p.current_price for p in self._positions.values()
        )
        total_equity = self._cash + portfolio_value
        return AccountInfo(
            account_id="MOCK-001",
            cash=self._cash,
            portfolio_value=portfolio_value,
            total_equity=total_equity,
            currency="EUR",
        )

    def get_positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def place_order(self, order: OrderRequest) -> OrderResult:
        """Simulate order execution with slippage and commission."""
        ts = datetime.now(self._tz)
        order_id = str(uuid.uuid4())[:8]

        # Apply slippage to fill price
        if order.side == OrderSide.BUY:
            fill_price = order.limit_price or self._get_last_price(order.ticker)
            fill_price *= (1 + self._slippage_pct)
        else:
            fill_price = order.limit_price or self._get_last_price(order.ticker)
            fill_price *= (1 - self._slippage_pct)

        # Calculate commission and TTF
        trade_value = order.quantity * fill_price
        commission  = max(self._commission_min, trade_value * self._commission_pct)
        # TTF applies to BUY orders on French large-cap equities only.
        # ETFs listed in ttf_exempt_tickers are excluded even if they end with .PA.
        ttf = (
            trade_value * self._ttf_pct
            if (order.side == OrderSide.BUY
                and order.ticker.endswith(self._ttf_suffix)
                and order.ticker not in self._ttf_exempt)
            else 0.0
        )

        existing = self._positions.get(order.ticker)

        if order.side == OrderSide.BUY:
            if existing is not None and existing.quantity < 0:
                # ── Cover an existing SHORT position ──────────────────────────
                cover_qty  = min(order.quantity, abs(existing.quantity))
                cover_cost = cover_qty * fill_price + commission + ttf
                if cover_cost > self._cash:
                    logger.warning(
                        "MockBroker: Insufficient cash to COVER %s short "
                        "(need %.2f, have %.2f).",
                        order.ticker, cover_cost, self._cash,
                    )
                    return OrderResult(
                        order_id=order_id, client_order_id=order.client_order_id,
                        ticker=order.ticker, side=order.side,
                        quantity=order.quantity, filled_quantity=0.0,
                        avg_fill_price=0.0, status=OrderStatus.REJECTED,
                        timestamp=ts, commission=0.0,
                    )
                self._cash -= cover_cost
                self._cover_short_position(order.ticker, cover_qty, fill_price)
            else:
                # ── Open / add to a LONG position ──────────────────────────────
                total_cost = trade_value + commission + ttf
                if total_cost > self._cash:
                    logger.warning(
                        "MockBroker: Insufficient cash for %s BUY %.2f @ %.2f "
                        "(need %.2f, have %.2f).",
                        order.ticker, order.quantity, fill_price, total_cost, self._cash,
                    )
                    return OrderResult(
                        order_id=order_id, client_order_id=order.client_order_id,
                        ticker=order.ticker, side=order.side,
                        quantity=order.quantity, filled_quantity=0.0,
                        avg_fill_price=0.0, status=OrderStatus.REJECTED,
                        timestamp=ts, commission=0.0,
                    )
                self._cash -= total_cost
                self._update_position_buy(order.ticker, order.quantity, fill_price)

        else:  # SELL
            if existing is not None and existing.quantity > 0:
                # ── Close an existing LONG position ───────────────────────────
                sell_qty = min(order.quantity, existing.quantity)
                proceeds = sell_qty * fill_price - commission
                self._cash += proceeds
                self._update_position_sell(order.ticker, sell_qty, fill_price)

            elif existing is None:
                # ── Open a new SHORT position ─────────────────────────────────
                # Margin check: require 100% of position value as cash collateral.
                # Even though you receive the proceeds, the collateral ensures
                # the bot can cover at current price without going negative.
                margin_required = trade_value * float(
                    getattr(self, "_short_margin_pct", 1.0)
                )
                if margin_required > self._cash:
                    logger.warning(
                        "MockBroker: Insufficient cash margin for SHORT %s "
                        "(need %.2f collateral, have %.2f).",
                        order.ticker, margin_required, self._cash,
                    )
                    return OrderResult(
                        order_id=order_id, client_order_id=order.client_order_id,
                        ticker=order.ticker, side=order.side,
                        quantity=order.quantity, filled_quantity=0.0,
                        avg_fill_price=0.0, status=OrderStatus.REJECTED,
                        timestamp=ts, commission=0.0,
                    )
                # Receive short-sale proceeds (price goes into cash)
                self._cash += trade_value - commission
                self._open_short_position(order.ticker, order.quantity, fill_price)

            else:
                # Already short — no stacking
                logger.warning(
                    "MockBroker: Already short %s — no duplicate short positions.",
                    order.ticker,
                )
                return OrderResult(
                    order_id=order_id, client_order_id=order.client_order_id,
                    ticker=order.ticker, side=order.side,
                    quantity=order.quantity, filled_quantity=0.0,
                    avg_fill_price=0.0, status=OrderStatus.REJECTED,
                    timestamp=ts, commission=0.0,
                )

        result = OrderResult(
            order_id=order_id,
            client_order_id=order.client_order_id,
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            filled_quantity=order.quantity,
            avg_fill_price=fill_price,
            status=OrderStatus.FILLED,
            timestamp=ts,
            commission=commission,
        )

        self._orders[order_id] = result
        self._trade_log.append({
            "timestamp":  ts.isoformat(),
            "ticker":     order.ticker,
            "side":       order.side.value,
            "quantity":   order.quantity,
            "fill_price": fill_price,
            "commission": commission,
            "ttf":        round(ttf, 4),
            "cash_after": self._cash,
        })

        ttf_str = f", ttf={ttf:.2f}" if ttf > 0 else ""
        logger.info(
            "MockBroker: %s %s %.4f @ %.4f (commission=%.2f%s, cash=%.2f).",
            order.side.value, order.ticker, order.quantity,
            fill_price, commission, ttf_str, self._cash,
        )
        return result

    def cancel_order(self, order_id: str) -> bool:
        """Mock: orders are filled immediately, so cancellation always fails."""
        logger.warning("MockBroker: Cannot cancel order %s (already filled).", order_id)
        return False

    def get_order_status(self, order_id: str) -> OrderResult | None:
        return self._orders.get(order_id)

    def get_quote(self, ticker: str) -> float:
        """Return last known price for a ticker (from position data)."""
        if ticker in self._positions:
            return self._positions[ticker].current_price
        return 0.0

    def update_prices(self, prices: dict[str, float]) -> None:
        """
        Update current prices for all known tickers AND open positions.
        Call this at each bar to keep unrealised PnL current.
        """
        # Always update the market-price cache (covers new tickers with no position yet)
        self._market_prices.update(prices)
        # Also refresh unrealised PnL for open positions
        for ticker, price in prices.items():
            if ticker in self._positions:
                pos = self._positions[ticker]
                pos.current_price = price
                pos.unrealised_pnl = (price - pos.avg_cost) * pos.quantity

    @property
    def trade_log(self) -> list[dict[str, Any]]:
        """Return the full trade log."""
        return list(self._trade_log)

    def save_state(self, path: "Path") -> None:
        """
        Persist the current portfolio state to a JSON file.

        Saves cash, all open positions, and the latest market-price cache.
        Call this before shutdown so the next session can resume seamlessly.
        """
        import json
        from datetime import datetime as _dt
        from pathlib import Path as _Path

        state = {
            "timestamp":    _dt.now().isoformat(),
            "cash":         round(self._cash, 6),
            "initial_cash": round(self._initial_cash, 6),
            "positions": {
                t: {
                    "ticker":         p.ticker,
                    "quantity":       p.quantity,
                    "avg_cost":       p.avg_cost,
                    "current_price":  p.current_price,
                    "unrealised_pnl": p.unrealised_pnl,
                    "realised_pnl":   p.realised_pnl,
                }
                for t, p in self._positions.items()
            },
            "market_prices": dict(self._market_prices),
        }
        _Path(path).parent.mkdir(parents=True, exist_ok=True)
        _Path(path).write_text(json.dumps(state, indent=2), encoding="utf-8")
        logger.info(
            "Portfolio state saved → %s  (cash=%.2f, %d open positions).",
            path, self._cash, len(self._positions),
        )

    def load_state(self, path: "Path") -> bool:
        """
        Restore portfolio state from a previously saved JSON file.

        Returns True if the state was loaded successfully, False otherwise
        (e.g. file does not exist or is corrupted).
        """
        import json
        from pathlib import Path as _Path

        _p = _Path(path)
        if not _p.exists():
            return False
        try:
            state = json.loads(_p.read_text(encoding="utf-8"))
            self._cash         = float(state["cash"])
            self._initial_cash = float(state.get("initial_cash", self._initial_cash))
            self._positions    = {}
            for t, d in state.get("positions", {}).items():
                pos = Position(
                    ticker=d["ticker"],
                    quantity=float(d["quantity"]),
                    avg_cost=float(d["avg_cost"]),
                    current_price=float(d["current_price"]),
                    unrealised_pnl=float(d.get("unrealised_pnl", 0.0)),
                    realised_pnl=float(d.get("realised_pnl", 0.0)),
                )
                self._positions[t] = pos
            self._market_prices = {
                k: float(v) for k, v in state.get("market_prices", {}).items()
            }
            print(
                f"\n  📂  Portfolio state restored from {path}\n"
                f"       Cash: {self._cash:,.2f}   "
                f"Positions: {len(self._positions)}  "
                f"(saved {state.get('timestamp', '?')})\n"
            )
            logger.info(
                "Portfolio state loaded — cash=%.2f, %d positions (as of %s).",
                self._cash, len(self._positions), state.get("timestamp", "?"),
            )
            return True
        except Exception as exc:
            logger.error("Failed to load portfolio state from %s: %s", path, exc)
            return False

    def reset(self) -> None:
        """Reset the mock broker to its initial state (discards all positions and cash)."""
        self._cash = self._initial_cash
        self._positions.clear()
        self._orders.clear()
        self._trade_log.clear()
        self._market_prices.clear()
        logger.info("MockBroker reset to initial state (cash=%.2f).", self._cash)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _get_last_price(self, ticker: str) -> float:
        """
        Get the latest known market price for a ticker.

        Priority:
        1. Market-price cache (updated every poll cycle via update_prices).
        2. Existing position's current_price (already up-to-date).
        3. 0.0 — should never reach this if update_prices is called first.
        """
        if ticker in self._market_prices:
            return self._market_prices[ticker]
        if ticker in self._positions:
            return self._positions[ticker].current_price
        logger.warning(
            "MockBroker: no market price for %s — order will fill at 0.  "
            "Call update_prices() before execute_signals().", ticker
        )
        return 0.0

    def _update_position_buy(self, ticker: str, qty: float, price: float) -> None:
        """Update or create a long position after a BUY."""
        if ticker in self._positions:
            pos = self._positions[ticker]
            total_qty  = pos.quantity + qty
            avg_cost   = (pos.avg_cost * pos.quantity + price * qty) / total_qty
            pos.quantity  = total_qty
            pos.avg_cost  = avg_cost
            pos.current_price = price
        else:
            self._positions[ticker] = Position(
                ticker=ticker,
                quantity=qty,
                avg_cost=price,
                current_price=price,
            )

    def _update_position_sell(self, ticker: str, qty: float, price: float) -> None:
        """Reduce or close a long position after a SELL."""
        if ticker not in self._positions:
            return
        pos = self._positions[ticker]
        realised = (price - pos.avg_cost) * qty
        pos.realised_pnl += realised
        pos.quantity -= qty
        pos.current_price = price
        if pos.quantity <= 1e-8:
            del self._positions[ticker]

    def _open_short_position(self, ticker: str, qty: float, price: float) -> None:
        """
        Create a new SHORT position (stored as negative quantity).

        Cash was already credited with the short-sale proceeds by the caller.
        Unrealised PnL = (avg_cost - current_price) * abs(qty).
        The generic formula (price - avg_cost) * quantity also gives the right
        sign because quantity is negative.
        """
        self._positions[ticker] = Position(
            ticker=ticker,
            quantity=-qty,      # negative = SHORT
            avg_cost=price,
            current_price=price,
            unrealised_pnl=0.0,
        )

    def _cover_short_position(self, ticker: str, qty: float, price: float) -> None:
        """
        Reduce or close a SHORT position after a COVER (buy-to-close).

        Cash was already debited with the cover cost by the caller.
        Profit = (avg_cost - cover_price) * qty  (positive when price fell).
        """
        if ticker not in self._positions:
            return
        pos = self._positions[ticker]
        # Realised profit on a short: we shorted at avg_cost, cover at price
        realised = (pos.avg_cost - price) * qty
        pos.realised_pnl += realised
        pos.quantity += qty   # quantity is negative; adding positive moves toward 0
        pos.current_price = price
        if abs(pos.quantity) <= 1e-8:
            del self._positions[ticker]
