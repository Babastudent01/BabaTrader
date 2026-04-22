"""
data/factory.py
Factory for creating the correct DataProvider based on config.
Exposes both a standalone function and a class-based factory for convenience.
"""
from __future__ import annotations

import logging

from config import Settings
from data.base import DataProvider
from data.cache import DataCache

logger = logging.getLogger(__name__)


class DataSourceFactory:
    """Class-based factory for creating DataProvider instances."""

    @staticmethod
    def create(settings: Settings) -> DataProvider:
        """Instantiate and return the configured DataProvider."""
        return create_data_provider(settings)


def create_data_provider(settings: Settings) -> DataProvider:
    """
    Instantiate and return the configured DataProvider.

    Parameters
    ----------
    settings : Settings
        Loaded application settings.

    Returns
    -------
    DataProvider
        Concrete provider instance (Yahoo, AlphaVantage, etc.).
    """
    data_cfg = settings.data
    provider_name: str = data_cfg.get("provider", "yahoo").lower()

    # Build cache if enabled
    cache: DataCache | None = None
    if data_cfg.get("cache_enabled", True):
        cache = DataCache(
            cache_dir=data_cfg.get("cache_dir", "data/cache"),
            ttl_hours=int(data_cfg.get("cache_ttl_hours", 24)),
        )
        logger.info("Data cache enabled at '%s'.", data_cfg.get("cache_dir", "data/cache"))

    if provider_name == "yahoo":
        from data.yahoo import YahooFinanceProvider
        logger.info("Using Yahoo Finance data provider.")
        return YahooFinanceProvider(cache=cache)

    elif provider_name == "alpha_vantage":
        from data.alpha_vantage import AlphaVantageProvider
        logger.info("Using Alpha Vantage data provider.")
        return AlphaVantageProvider(cache=cache)

    else:
        raise ValueError(
            f"Unknown data provider '{provider_name}'. "
            "Supported: 'yahoo', 'alpha_vantage'."
        )
