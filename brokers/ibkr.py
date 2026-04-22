"""
brokers/ibkr.py
Interactive Brokers (IBKR) broker adapter — PLACEHOLDER IMPLEMENTATION.

This adapter is NOT functional. It provides the correct interface structure
and documents exactly what needs to be implemented.

To implement:
1. Install the IBKR API: pip install ibapi
2. Run TWS or IB Gateway locally
3. Fill in the TODO sections below
4. Set IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, IBKR_ACCOUNT_ID in config/.env

Official documentation:
    https://interactivebrokers.github.io/tws-api/
    https://github.com/InteractiveBrokers/tws-api
"""
from __future__ import annotations

import logging
from datetime import datetime

from brokers.base import (
    AccountInfo,
    BrokerInterface,
    OrderRequest,
    OrderResult,
    OrderStatus,
    Position,
)
from config import get_env

logger = logging.getLogger(__name__)


class IBKRBroker(BrokerInterface):
    """
    Interactive Brokers adapter.

    ⚠️  PLACEHOLDER — NOT FUNCTIONAL ⚠️
    All methods raise NotImplementedError until implemented.

    Required environment variables (config/.env):
        IBKR_HOST        — TWS/Gateway host (default: 127.0.0.1)
        IBKR_PORT        — TWS paper: 7497, TWS live: 7496,
                           Gateway paper: 4002, Gateway live: 4001
        IBKR_CLIENT_ID   — Unique client ID (integer)
        IBKR_ACCOUNT_ID  — Your IBKR account ID (e.g. DU1234567)
    """

    def __init__(self) -> None:
        self._host      = get_env("IBKR_HOST", "127.0.0.1")
        self._port      = int(get_env("IBKR_PORT", "7497") or "7497")
        self._client_id = int(get_env("IBKR_CLIENT_ID", "1") or "1")
        self._account   = get_env("IBKR_ACCOUNT_ID", "")
        self._connected = False

        # TODO: Import and initialise the IBKR EClient/EWrapper here
        # from ibapi.client import EClient
        # from ibapi.wrapper import EWrapper
        # self._app = IBKRApp()  # Your EClient+EWrapper subclass
        logger.warning(
            "IBKRBroker is a placeholder. "
            "Implement using the official ibapi SDK before use."
        )

    @property
    def name(self) -> str:
        return "InteractiveBrokers"

    @property
    def is_market_open(self) -> bool:
        # TODO: Implement using IBKR market hours API or a schedule library
        # Example: use `exchange_calendars` package for NYSE/NASDAQ hours
        raise NotImplementedError(
            "IBKRBroker.is_market_open not implemented. "
            "Use exchange_calendars or IBKR's reqMarketDataType."
        )

    def connect(self) -> None:
        # TODO: Connect to TWS/Gateway
        # self._app.connect(self._host, self._port, self._client_id)
        # Start the EClient message loop in a background thread
        raise NotImplementedError(
            "IBKRBroker.connect() not implemented. "
            "See: https://interactivebrokers.github.io/tws-api/connection.html"
        )

    def disconnect(self) -> None:
        # TODO: self._app.disconnect()
        raise NotImplementedError("IBKRBroker.disconnect() not implemented.")

    def get_account_info(self) -> AccountInfo:
        # TODO: Use reqAccountSummary() or reqAccountUpdates()
        # Parse NetLiquidation, TotalCashValue, etc.
        raise NotImplementedError(
            "IBKRBroker.get_account_info() not implemented. "
            "See: reqAccountSummary() in IBKR API docs."
        )

    def get_positions(self) -> dict[str, Position]:
        # TODO: Use reqPositions() and parse position updates
        raise NotImplementedError(
            "IBKRBroker.get_positions() not implemented. "
            "See: reqPositions() in IBKR API docs."
        )

    def place_order(self, order: OrderRequest) -> OrderResult:
        # TODO: Build an ibapi.order.Order object and call placeOrder()
        # Map OrderRequest fields to IBKR Contract + Order objects
        # Handle order acknowledgement via orderStatus() callback
        raise NotImplementedError(
            "IBKRBroker.place_order() not implemented. "
            "See: placeOrder() in IBKR API docs."
        )

    def cancel_order(self, order_id: str) -> bool:
        # TODO: self._app.cancelOrder(int(order_id))
        raise NotImplementedError("IBKRBroker.cancel_order() not implemented.")

    def get_order_status(self, order_id: str) -> OrderResult | None:
        # TODO: Track order status via orderStatus() callback
        raise NotImplementedError("IBKRBroker.get_order_status() not implemented.")

    def get_quote(self, ticker: str) -> float:
        # TODO: Use reqMktData() or reqTickByTickData() for real-time quotes
        raise NotImplementedError(
            "IBKRBroker.get_quote() not implemented. "
            "See: reqMktData() in IBKR API docs."
        )
