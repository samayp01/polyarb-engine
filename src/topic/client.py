"""
Polymarket API Client

Fetches market data from the Polymarket Gamma API with retry logic.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass

from .utils.http import make_session, safe_get_json

logger = logging.getLogger(__name__)


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


class PolymarketClient:
  """Client for Polymarket Gamma API with retry logic."""

  BASE_URL = "https://gamma-api.polymarket.com"

  def __init__(self, timeout=30):
    self.timeout = timeout
    self.session = make_session()
    self.session.headers.update({"User-Agent": "polymarket-trading/1.0"})

  def fetch_markets(self, limit=100, min_liquidity=1000, min_volume=1000):
    """Fetch active markets."""
    markets = []
    offset = 0

    while True:
      batch = self._fetch_batch(limit, offset, closed=False)
      if not batch:
        break

      for item in batch:
        market = self._parse_market(item)
        if market and market.liquidity >= min_liquidity and market.volume >= min_volume:
          markets.append(market)

      if len(batch) < limit:
        break

      offset += limit
      time.sleep(0.1)

    logger.info(f"Fetched {len(markets)} active markets")
    return markets

  def fetch_closed_markets(self, limit=100, max_pages=10):
    """Fetch closed/resolved markets."""
    markets = []
    offset = 0

    for _ in range(max_pages):
      batch = self._fetch_batch(limit, offset, closed=True)
      if not batch:
        break

      for item in batch:
        market = self._parse_market(item)
        if market:
          markets.append(market)

      if len(batch) < limit:
        break

      offset += limit
      time.sleep(0.1)

    logger.info(f"Fetched {len(markets)} closed markets")
    return markets

  def _fetch_batch(self, limit, offset, closed=False):
    """Fetch a batch of markets."""
    params = {"limit": limit, "offset": offset}
    if closed:
      params["closed"] = "true"
    else:
      params["closed"] = "false"
      params["active"] = "true"

    data = safe_get_json(
      self.session,
      f"{self.BASE_URL}/markets",
      params=params,
      timeout=self.timeout,
    )

    return data if data else []

  def _parse_market(self, data):
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
