"""
Polymarket API Client

Simple client for fetching market data from the Polymarket Gamma API.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://gamma-api.polymarket.com"


@dataclass
class Market:
    """Polymarket prediction market."""

    id: str
    question: str
    description: str
    slug: str
    outcomes: list
    prices: list
    volume: float
    liquidity: float
    end_date: str

    @property
    def yes_price(self):
        return self.prices[0] if self.prices else 0.0

    @property
    def no_price(self):
        return self.prices[1] if len(self.prices) > 1 else 1.0 - self.yes_price

    def to_dict(self):
        return asdict(self)


def _parse_market(data):
    """Parse market data from API response."""
    try:
        prices_str = data.get("outcomePrices", "[]")
        prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
        prices = [float(p) for p in prices] if prices else [0.0, 1.0]

        return Market(
            id=str(data.get("id", "")),
            question=data.get("question", ""),
            description=data.get("description", ""),
            slug=data.get("slug", ""),
            outcomes=data.get("outcomes", ["Yes", "No"]),
            prices=prices,
            volume=float(data.get("volume", 0)),
            liquidity=float(data.get("liquidity", 0)),
            end_date=data.get("endDate", ""),
        )
    except (ValueError, KeyError, TypeError) as e:
        logger.debug(f"Failed to parse market: {e}")
        return None


def fetch_markets(limit=100, min_liquidity=1000, min_volume=1000):
    """Fetch active markets from Polymarket."""
    markets = []
    offset = 0

    while True:
        params = {"limit": limit, "offset": offset, "closed": "false", "active": "true"}

        try:
            resp = requests.get(f"{BASE_URL}/markets", params=params, timeout=30)
            resp.raise_for_status()
            batch = resp.json()
        except Exception as e:
            logger.error(f"API error: {e}")
            break

        if not batch:
            break

        for item in batch:
            market = _parse_market(item)
            if market and market.liquidity >= min_liquidity and market.volume >= min_volume:
                markets.append(market)

        if len(batch) < limit:
            break

        offset += limit
        time.sleep(0.1)

    logger.info(f"Fetched {len(markets)} active markets")
    return markets


def fetch_closed_markets(limit=100, max_pages=10):
    """Fetch closed/resolved markets from Polymarket."""
    markets = []
    offset = 0

    for _ in range(max_pages):
        params = {"limit": limit, "offset": offset, "closed": "true"}

        try:
            resp = requests.get(f"{BASE_URL}/markets", params=params, timeout=30)
            resp.raise_for_status()
            batch = resp.json()
        except Exception as e:
            logger.error(f"API error: {e}")
            break

        if not batch:
            break

        for item in batch:
            market = _parse_market(item)
            if market:
                markets.append(market)

        if len(batch) < limit:
            break

        offset += limit
        time.sleep(0.1)

    logger.info(f"Fetched {len(markets)} closed markets")
    return markets
