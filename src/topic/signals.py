"""
Signal Engine

Generates trading signals when leader markets resolve.

When a market resolves:
1. Look up outgoing edges from that market
2. For each edge, compute expected price move in follower
3. Compare expected vs current price
4. Emit signal if mispricing exceeds threshold
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .client import PolymarketClient
from .models import Signal, Outcome
from .graph import EventGraph
from .ingestion import ResolutionTracker

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
SIGNALS_FILE = DATA_DIR / "signals.json"


class SignalEngine:
  """
  Generates trading signals from market resolutions.

  Monitors for resolutions and generates signals based on
  learned conditional relationships in the event graph.
  """

  MIN_MISPRICING = 0.03   # Minimum expected move to generate signal
  MIN_CONFIDENCE = 0.5    # Minimum edge confidence
  MIN_LIQUIDITY = 5000    # Minimum follower liquidity

  def __init__(self, graph=None, client=None):
    self.graph = graph or EventGraph()
    self.client = client or PolymarketClient()
    self._markets = {}
    self._signals = []
    self._load_signals()

  def _load_signals(self):
    """Load historical signals."""
    if SIGNALS_FILE.exists():
      try:
        with open(SIGNALS_FILE) as f:
          data = json.load(f)
        self._signals = [Signal.from_dict(s) for s in data]
        logger.info(f"Loaded {len(self._signals)} historical signals")
      except Exception as e:
        logger.error(f"Failed to load signals: {e}")

  def _save_signals(self):
    """Save signals to storage."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SIGNALS_FILE, "w") as f:
      json.dump([s.to_dict() for s in self._signals], f, indent=2)

  def refresh_markets(self):
    """Refresh cache of active markets."""
    markets = self.client.fetch_markets(min_liquidity=self.MIN_LIQUIDITY)
    self._markets = {m.id: m for m in markets}
    logger.info(f"Refreshed {len(self._markets)} active markets")

  def get_price(self, market_id):
    """Get current price for a market."""
    if market_id not in self._markets:
      self.refresh_markets()
    market = self._markets.get(market_id)
    return market.yes_price if market else None

  def get_liquidity(self, market_id):
    """Get current liquidity for a market."""
    market = self._markets.get(market_id)
    return market.liquidity if market else 0.0

  def on_resolution(self, resolution):
    """
    Handle a market resolution event.

    Returns list of generated signals.
    """
    logger.info(f"Processing resolution: {resolution.market_id} -> {resolution.outcome.name}")

    # Get edges where resolved market is the leader
    edges = self.graph.get_outgoing(resolution.market_id)

    if not edges:
      return []

    signals = []
    for edge in edges:
      signal = self._generate_signal(resolution, edge)
      if signal:
        signals.append(signal)
        self._signals.append(signal)

    if signals:
      self._save_signals()
      logger.info(f"Generated {len(signals)} signals")

    return signals

  def _generate_signal(self, resolution, edge):
    """Generate signal for a single edge."""
    # Check edge confidence
    if edge.confidence < self.MIN_CONFIDENCE:
      return None

    # Get conditional delta
    delta = edge.get_delta(resolution.outcome)
    if not delta:
      return None

    # Get current follower price
    current_price = self.get_price(edge.to_market_id)
    if current_price is None:
      return None

    # Check liquidity
    if self.get_liquidity(edge.to_market_id) < self.MIN_LIQUIDITY:
      return None

    # Compute expected price
    expected_price = max(0.0, min(1.0, current_price + delta.avg_delta))
    mispricing = expected_price - current_price

    # Check minimum threshold
    if abs(mispricing) < self.MIN_MISPRICING:
      return None

    direction = "BUY" if mispricing > 0 else "SELL"

    signal = Signal(
      market_id=edge.to_market_id,
      direction=direction,
      expected_move=mispricing,
      current_price=current_price,
      expected_price=expected_price,
      confidence=edge.confidence * (delta.sample_count / 10),
      source_edge=edge,
      generated_at=datetime.now(timezone.utc),
      leader_market_id=resolution.market_id,
      leader_outcome=resolution.outcome,
    )

    logger.info(f"Signal: {direction} {edge.to_market_id[:8]}... ({current_price:.2%} -> {expected_price:.2%})")
    return signal

  def get_recent(self, max_age_hours=24):
    """Get signals from the last N hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    return [s for s in self._signals if s.generated_at > cutoff]

  def get_all(self):
    """Get all historical signals."""
    return list(self._signals)


class SignalMonitor:
  """Continuously monitors for new resolutions and generates signals."""

  def __init__(self, engine=None, tracker=None):
    self.engine = engine or SignalEngine()
    self.tracker = tracker or ResolutionTracker()

  def check(self):
    """Check for new resolutions and generate signals."""
    new_resolutions = self.tracker.check_new_resolutions()

    if new_resolutions:
      self.engine.refresh_markets()

    signals = []
    for resolution in new_resolutions:
      signals.extend(self.engine.on_resolution(resolution))

    return signals

  def run_forever(self, interval=60):
    """Run continuous monitoring loop."""
    logger.info(f"Starting signal monitor (interval: {interval}s)")

    while True:
      try:
        signals = self.check()

        if signals:
          print(f"\n{'='*50}")
          print(f"NEW SIGNALS: {len(signals)}")
          print("=" * 50)
          for s in signals:
            print(f"\n{s.direction} {s.market_id[:16]}...")
            print(f"  {s.current_price:.2%} -> {s.expected_price:.2%} ({s.expected_move:+.2%})")

        time.sleep(interval)
      except KeyboardInterrupt:
        logger.info("Monitor stopped")
        break
      except Exception as e:
        logger.error(f"Monitor error: {e}")
        time.sleep(60)


def print_signals(signals):
  """Print signal summary."""
  if not signals:
    print("\nNo signals to display.")
    return

  print(f"\n{'='*60}")
  print(f"SIGNALS ({len(signals)} total)")
  print("=" * 60)

  buy = [s for s in signals if s.direction == "BUY"]
  sell = [s for s in signals if s.direction == "SELL"]

  print(f"BUY: {len(buy)}, SELL: {len(sell)}")

  if signals:
    avg_move = sum(abs(s.expected_move) for s in signals) / len(signals)
    print(f"Avg expected move: {avg_move:.2%}")

  print("\n" + "-" * 60)
  for s in sorted(signals, key=lambda x: x.generated_at, reverse=True)[:10]:
    age = (datetime.now(timezone.utc) - s.generated_at).total_seconds() / 60
    print(f"\n[{s.direction}] {s.market_id[:20]}...")
    print(f"  {s.current_price:.2%} -> {s.expected_price:.2%} ({s.expected_move:+.2%})")
    print(f"  Age: {age:.0f}m, Confidence: {s.confidence:.2%}")
