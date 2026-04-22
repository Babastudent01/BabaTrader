"""
brokers/base.py
Abstract broker interface.
All broker adapters must implement this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"


class OrderStatus(str, Enum):
    PENDING   = "PENDING"
    FILLED    = "FILLED"
    PARTIAL   = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED  = "REJECTED"


@dataclass
class OrderRequest:
    """Request to place an order."""
    ticker:     str
    side:       OrderSide
    quantity:   float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_loss:   float | None = None
    take_profit: float | None = None
    client_order_id: str = ""


@dataclass
class OrderResult:
    """Result of a placed order."""
    order_id:        str
    client_order_id: str
    ticker:          str
    side:            OrderSide
    quantity:        float
    filled_quantity: float
    avg_fill_price:  float
    status:          OrderStatus
    timestamp:       datetime
    commission:      float = 0.0
    metadata:        dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """An open position."""
    ticker:        str
    quantity:      float
    avg_cost:      float
    current_price: float
    unrealised_pnl: float = 0.0
    realised_pnl:   float = 0.0


@dataclass
class AccountInfo:
    """Broker account summary."""
    account_id:    str
    cash:          float
    portfolio_value: float
    total_equity:  float
    currency:      str = "EUR"


class BrokerInterface(ABC):
    """
    Abstract broker interface.

    All concrete broker adapters (Mock, IBKR, Saxo, etc.) must implement
    every method defined here. This ensures the rest of the system is
    completely broker-agnostic.
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the broker."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the broker connection gracefully."""

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """Return current account balance and equity."""

    @abstractmethod
    def get_positions(self) -> dict[str, Position]:
        """Return all open positions keyed by ticker."""

    @abstractmethod
    def place_order(self, order: OrderRequest) -> OrderResult:
        """
        Submit an order to the broker.

        Parameters
        ----------
        order : OrderRequest
            Order details.

        Returns
        -------
        OrderResult
            Execution result including fill price and status.
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True if successful."""

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderResult | None:
        """Return the current status of an order."""

    @abstractmethod
    def get_quote(self, ticker: str) -> float:
        """Return the current market price for a ticker."""

    @property
    @abstractmethod
    def is_market_open(self) -> bool:
        """Return True if the market is currently open for trading."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Broker name (for logging)."""
