"""
data/market_hours.py
Exchange market-hours awareness for the live trading loop.

Knows the trading hours and timezones for each exchange and maps
every configured ticker to its exchange so the live loop can:
  - Warn when all markets are closed.
  - Filter signals to only tickers whose exchange is currently open.
  - Report which market opens next and when.
"""
from __future__ import annotations

from datetime import datetime, time
from typing import NamedTuple

import pytz

# ── Exchange definitions ──────────────────────────────────────────────────────

class ExchangeInfo(NamedTuple):
    name:     str
    timezone: str
    open:     time
    close:    time

EXCHANGES: dict[str, ExchangeInfo] = {
    "NYSE": ExchangeInfo(
        name="NYSE (New York)",
        timezone="America/New_York",
        open=time(9, 30),
        close=time(16, 0),
    ),
    "NASDAQ": ExchangeInfo(
        name="NASDAQ (New York)",
        timezone="America/New_York",
        open=time(9, 30),
        close=time(16, 0),
    ),
    "EURONEXT_PARIS": ExchangeInfo(
        name="Euronext Paris",
        timezone="Europe/Paris",
        open=time(9, 0),
        close=time(17, 30),
    ),
}

# ── Ticker → exchange mapping ─────────────────────────────────────────────────

def _ticker_exchange(ticker: str) -> str:
    """
    Return the exchange key for a given ticker symbol.
    Tickers ending in `.PA` → Euronext Paris.
    All others → NASDAQ (covers NYSE stocks too — same hours).
    """
    if ticker.upper().endswith(".PA"):
        return "EURONEXT_PARIS"
    return "NASDAQ"


# ── Open/closed checks ────────────────────────────────────────────────────────

def is_exchange_open(exchange_key: str, now_utc: datetime | None = None) -> bool:
    """
    Return True if the given exchange is currently in its regular trading session.
    Weekends are always closed. No holiday calendar (use with awareness).
    """
    info = EXCHANGES.get(exchange_key)
    if info is None:
        return False

    tz    = pytz.timezone(info.timezone)
    now   = (now_utc or datetime.utcnow().replace(tzinfo=pytz.utc)).astimezone(tz)

    # Weekends
    if now.weekday() >= 5:
        return False

    local_time = now.time().replace(second=0, microsecond=0)
    return info.open <= local_time < info.close


def is_ticker_market_open(ticker: str, now_utc: datetime | None = None) -> bool:
    """Return True if the market for *ticker* is currently open."""
    return is_exchange_open(_ticker_exchange(ticker), now_utc)


def get_market_status(tickers: list[str], now_utc: datetime | None = None) -> dict:
    """
    Return a status dict for all exchanges touched by *tickers*.

    Returns
    -------
    dict with keys:
      - ``exchange_status`` : {exchange_key → {open: bool, local_time: str, ...}}
      - ``open_tickers``    : list of tickers whose market is open
      - ``closed_tickers``  : list of tickers whose market is closed
      - ``any_open``        : bool — at least one ticker is tradeable
      - ``next_open``       : str — human description of the next market to open
    """
    now = now_utc or datetime.utcnow().replace(tzinfo=pytz.utc)

    exchange_status: dict[str, dict] = {}
    for key, info in EXCHANGES.items():
        tz         = pytz.timezone(info.timezone)
        local_now  = now.astimezone(tz)
        is_open    = is_exchange_open(key, now)
        exchange_status[key] = {
            "name":       info.name,
            "open":       is_open,
            "local_time": local_now.strftime("%H:%M %Z"),
            "session":    f"{info.open.strftime('%H:%M')} – {info.close.strftime('%H:%M')} {local_now.strftime('%Z')}",
        }

    open_tickers   = [t for t in tickers if is_ticker_market_open(t, now)]
    closed_tickers = [t for t in tickers if not is_ticker_market_open(t, now)]

    # Find next market to open (only among exchanges used by configured tickers)
    used_exchanges = {_ticker_exchange(t) for t in tickers}
    next_open_desc = _next_open_description(used_exchanges, now)

    return {
        "exchange_status": exchange_status,
        "open_tickers":    open_tickers,
        "closed_tickers":  closed_tickers,
        "any_open":        len(open_tickers) > 0,
        "next_open":       next_open_desc,
    }


def print_market_status(status: dict) -> None:
    """Print a concise, always-visible market-hours summary."""
    line = "─" * 62
    print(f"\n{line}")
    print("  🕐  MARKET STATUS")
    print(line)

    for key, info in status["exchange_status"].items():
        used = key in {_ticker_exchange(t) for t in status["open_tickers"] + status["closed_tickers"]}
        if not used:
            continue
        icon   = "🟢 OPEN  " if info["open"] else "🔴 CLOSED"
        print(f"  {icon}  {info['name']:<22}  {info['local_time']:<10}  ({info['session']})")

    print(f"  {'─'*58}")
    if status["open_tickers"]:
        print(f"  Active tickers  ({len(status['open_tickers'])}): {', '.join(status['open_tickers'])}")
    if status["closed_tickers"]:
        print(f"  Closed tickers  ({len(status['closed_tickers'])}): {', '.join(status['closed_tickers'])}")

    if not status["any_open"]:
        print(f"\n  ⚠️  ALL MARKETS ARE CURRENTLY CLOSED")
        print(f"  Next open: {status['next_open']}")
    print(f"{line}\n")


# ── Internals ─────────────────────────────────────────────────────────────────

def _next_open_description(exchange_keys: set[str], now_utc: datetime) -> str:
    """
    Return a human-readable string saying when the next market opens.
    E.g. 'Euronext Paris opens at 09:00 CET in 13h 24m'
    """
    from datetime import timedelta

    candidates: list[tuple[float, str]] = []

    for key in exchange_keys:
        info = EXCHANGES.get(key)
        if info is None:
            continue
        tz        = pytz.timezone(info.timezone)
        local_now = now_utc.astimezone(tz)

        # Try today first, then tomorrow, then keep adding days (skip weekends)
        for day_offset in range(1, 8):
            candidate_day = local_now + timedelta(days=day_offset)
            # Skip weekends
            if candidate_day.weekday() >= 5:
                continue
            # Build the open datetime
            open_dt_local = tz.localize(
                candidate_day.replace(
                    hour=info.open.hour, minute=info.open.minute,
                    second=0, microsecond=0,
                ).replace(tzinfo=None)
            )
            diff_secs = (open_dt_local.astimezone(pytz.utc) - now_utc).total_seconds()
            if diff_secs > 0:
                h = int(diff_secs // 3600)
                m = int((diff_secs % 3600) // 60)
                label = f"{info.name} opens at {info.open.strftime('%H:%M')} in {h}h {m}m"
                candidates.append((diff_secs, label))
                break

    if not candidates:
        return "unknown"
    return min(candidates, key=lambda x: x[0])[1]
