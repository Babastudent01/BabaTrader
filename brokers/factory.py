"""
brokers/factory.py
Factory for creating the correct broker adapter based on config.
"""
from __future__ import annotations

import logging

from config import Settings
from brokers.base import BrokerInterface

logger = logging.getLogger(__name__)


class BrokerFactory:
    """Creates the configured broker adapter."""

    @staticmethod
    def create(settings: Settings) -> BrokerInterface:
        """
        Instantiate and return the configured broker.

        Broker is set via settings.yaml:
            broker:
              name: mock   # mock | ibkr | saxo

        Parameters
        ----------
        settings : Settings
            Loaded application settings.

        Returns
        -------
        BrokerInterface
            Concrete broker instance.
        """
        broker_cfg  = settings.broker
        broker_name: str = broker_cfg.get("name", "mock").lower()

        if broker_name == "mock":
            from brokers.mock import MockBroker
            logger.info("Using MockBroker (paper trading / no real orders).")
            return MockBroker(settings)

        elif broker_name == "ibkr":
            from brokers.ibkr import IBKRBroker  # type: ignore[import]
            logger.info("Using Interactive Brokers (IBKR) adapter.")
            return IBKRBroker(settings)

        elif broker_name == "saxo":
            from brokers.saxo import SaxoBroker  # type: ignore[import]
            logger.info("Using Saxo Bank adapter.")
            return SaxoBroker(settings)

        else:
            raise ValueError(
                f"Unknown broker '{broker_name}'. "
                "Supported: 'mock', 'ibkr', 'saxo'."
            )
