"""
brokers/saxo.py
Saxo Bank OpenAPI broker adapter — PLACEHOLDER IMPLEMENTATION.

This adapter is NOT functional. It provides the correct interface structure
and documents exactly what needs to be implemented.

To implement:
1. Register a developer account at https://www.developer.saxo/
2. Create an app and get SAXO_APP_KEY and SAXO_APP_SECRET
3. Implement OAuth2 authentication flow
4. Fill in the TODO sections below
5. Set all SAXO_* variables in config/.env

Official documentation:
    https://www.developer.saxo/openapi/learn
    https://github.com/SaxoBank/openapi-samples-python
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


class SaxoBroker(BrokerInterface):
    """
    Saxo Bank OpenAPI adapter.

    ⚠️  PLACEHOLDER — NOT FUNCTIONAL ⚠️
    All methods raise NotImplementedError until implemented.

    Required environment variables (config/.env):
        SAXO_APP_KEY      — Your Saxo app key
        SAXO_APP_SECRET   — Your Saxo app secret
        SAXO_REDIRECT_URI — OAuth2 redirect URI
        SAXO_BASE_URL     — API base URL
                            Simulation: https://gateway.saxobank.com/sim/openapi
                            Live:       https://gateway.saxobank.com/openapi
        SAXO_ACCOUNT_KEY  — Your Saxo account key

    Authentication:
        Saxo uses OAuth2. You must implement the token refresh flow.
        See: https://www.developer.saxo/openapi/learn/oauth-authorization-code-grant
    """

    def __init__(self) -> None:
        self._app_key      = get_env("SAXO_APP_KEY", "")
        self._app_secret   = get_env("SAXO_APP_SECRET", "")
        self._redirect_uri = get_env("SAXO_REDIRECT_URI", "http://localhost:8080/callback")
        self._base_url     = get_env("SAXO_BASE_URL", "https://gateway.saxobank.com/sim/openapi")
        self._account_key  = get_env("SAXO_ACCOUNT_KEY", "")
        self._access_token: str | None = None
        self._connected = False

        # TODO: Implement OAuth2 token management
        # self._token_manager = SaxoTokenManager(...)
        logger.warning(
            "SaxoBroker is a placeholder. "
            "Implement OAuth2 and REST calls using the Saxo OpenAPI before use."
        )

    @property
    def name(self) -> str:
        return "SaxoBank"

    @property
    def is_market_open(self) -> bool:
        # TODO: Call GET /ref/v1/exchanges/{ExchangeId} to check trading hours
        # Or use exchange_calendars package for schedule-based check
        raise NotImplementedError(
            "SaxoBroker.is_market_open not implemented. "
            "See: GET /ref/v1/exchanges in Saxo OpenAPI docs."
        )

    def connect(self) -> None:
        # TODO: Implement OAuth2 authorization code flow
        # 1. Redirect user to authorization URL
        # 2. Exchange code for access + refresh tokens
        # 3. Store tokens and set up refresh scheduler
        raise NotImplementedError(
            "SaxoBroker.connect() not implemented. "
            "See: https://www.developer.saxo/openapi/learn/oauth-authorization-code-grant"
        )

    def disconnect(self) -> None:
        # TODO: Revoke tokens if needed
        self._access_token = None
        self._connected = False
        logger.info("SaxoBroker disconnected.")

    def get_account_info(self) -> AccountInfo:
        # TODO: GET /port/v1/accounts/{AccountKey}
        # Parse TotalValue, CashBalance, etc.
        raise NotImplementedError(
            "SaxoBroker.get_account_info() not implemented. "
            "See: GET /port/v1/accounts in Saxo OpenAPI docs."
        )

    def get_positions(self) -> dict[str, Position]:
        # TODO: GET /port/v1/positions?AccountKey={AccountKey}
        raise NotImplementedError(
            "SaxoBroker.get_positions() not implemented. "
            "See: GET /port/v1/positions in Saxo OpenAPI docs."
        )

    def place_order(self, order: OrderRequest) -> OrderResult:
        # TODO: POST /trade/v2/orders
        # Map OrderRequest to Saxo order body:
        # {
        #   "AccountKey": self._account_key,
        #   "AssetType": "Stock",
        #   "BuySell": "Buy" or "Sell",
        #   "Amount": order.quantity,
        #   "OrderType": "Market" or "Limit",
        #   "Uic": <Saxo UIC for the instrument>,  # Must look up UIC from ticker
        #   ...
        # }
        raise NotImplementedError(
            "SaxoBroker.place_order() not implemented. "
            "See: POST /trade/v2/orders in Saxo OpenAPI docs. "
            "Note: You must map ticker symbols to Saxo UIC codes first."
        )

    def cancel_order(self, order_id: str) -> bool:
        # TODO: DELETE /trade/v2/orders/{OrderId}?AccountKey={AccountKey}
        raise NotImplementedError("SaxoBroker.cancel_order() not implemented.")

    def get_order_status(self, order_id: str) -> OrderResult | None:
        # TODO: GET /trade/v2/orders/{OrderId}
        raise NotImplementedError("SaxoBroker.get_order_status() not implemented.")

    def get_quote(self, ticker: str) -> float:
        # TODO: GET /trade/v1/infoprices?Uic={Uic}&AssetType=Stock
        # Must first resolve ticker → Saxo UIC via /ref/v1/instruments
        raise NotImplementedError(
            "SaxoBroker.get_quote() not implemented. "
            "See: GET /trade/v1/infoprices in Saxo OpenAPI docs."
        )
