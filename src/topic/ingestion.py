"""
Data Ingestion

Handles continuous collection of market data:
- SnapshotIngester: Captures point-in-time price snapshots
- ResolutionTracker: Tracks market resolutions
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from client import PolymarketClient
from models import MarketSnapshot, MarketResolution, Outcome

logger = logging.getLogger(__name__)

# Storage paths
DATA_DIR = Path("data")
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
RESOLUTIONS_FILE = DATA_DIR / "resolutions.json"


class SnapshotIngester:
  """
  Captures point-in-time price snapshots for active markets.

  Snapshots are stored in daily JSON files for historical analysis.
  """

  def __init__(self, client=None):
    self.client = client or PolymarketClient()
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

  def capture(self, min_liquidity=1000, min_volume=1000):
    """
    Capture current prices for all active markets.

    Returns list of snapshots captured.
    """
    timestamp = datetime.utcnow()
    markets = self.client.fetch_markets(
      min_liquidity=min_liquidity,
      min_volume=min_volume,
    )

    snapshots = []
    for market in markets:
      snapshot = MarketSnapshot(
        market_id=market.id,
        timestamp=timestamp,
        yes_price=market.yes_price,
        volume=market.volume,
        liquidity=market.liquidity,
      )
      snapshots.append(snapshot)

    self._save_snapshots(snapshots)
    logger.info(f"Captured {len(snapshots)} snapshots at {timestamp}")
    return snapshots

  def _save_snapshots(self, snapshots):
    """Append snapshots to daily JSON file."""
    if not snapshots:
      return

    date_str = snapshots[0].timestamp.strftime("%Y-%m-%d")
    filepath = SNAPSHOTS_DIR / f"snapshots_{date_str}.json"

    existing = []
    if filepath.exists():
      with open(filepath) as f:
        existing = json.load(f)

    existing.extend([s.to_dict() for s in snapshots])

    with open(filepath, "w") as f:
      json.dump(existing, f)

  def load_snapshots(self, start_date=None, end_date=None, market_id=None):
    """
    Load historical snapshots from storage.

    Args:
      start_date: Filter snapshots after this time.
      end_date: Filter snapshots before this time.
      market_id: Filter to specific market.

    Returns:
      List of matching snapshots sorted by timestamp.
    """
    snapshots = []

    for filepath in sorted(SNAPSHOTS_DIR.glob("snapshots_*.json")):
      with open(filepath) as f:
        data = json.load(f)

      for item in data:
        snapshot = MarketSnapshot.from_dict(item)

        if start_date and snapshot.timestamp < start_date:
          continue
        if end_date and snapshot.timestamp > end_date:
          continue
        if market_id and snapshot.market_id != market_id:
          continue

        snapshots.append(snapshot)

    snapshots.sort(key=lambda s: s.timestamp)
    return snapshots

  def get_price_at_time(self, market_id, target_time, tolerance_seconds=300):
    """Get market price closest to target time."""
    snapshots = self.load_snapshots(market_id=market_id)

    # Normalize target_time to naive UTC for comparison
    if target_time.tzinfo is not None:
      target_time = target_time.replace(tzinfo=None)

    best = None
    best_diff = float("inf")

    for snapshot in snapshots:
      snap_time = snapshot.timestamp
      if snap_time.tzinfo is not None:
        snap_time = snap_time.replace(tzinfo=None)

      diff = abs((snap_time - target_time).total_seconds())
      if diff < best_diff:
        best_diff = diff
        best = snapshot

    if best and best_diff <= tolerance_seconds:
      return best.yes_price
    return None


class ResolutionTracker:
  """
  Tracks market resolutions by polling for closed markets.

  Each resolution is stored once, keyed by market ID.
  """

  def __init__(self, client=None):
    self.client = client or PolymarketClient()
    self._resolutions = {}
    self._load()

  def _load(self):
    """Load existing resolutions from storage."""
    if RESOLUTIONS_FILE.exists():
      with open(RESOLUTIONS_FILE) as f:
        data = json.load(f)
      for item in data:
        resolution = MarketResolution.from_dict(item)
        self._resolutions[resolution.market_id] = resolution
      logger.info(f"Loaded {len(self._resolutions)} known resolutions")

  def _save(self):
    """Persist resolutions to storage."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESOLUTIONS_FILE, "w") as f:
      json.dump([r.to_dict() for r in self._resolutions.values()], f, indent=2)

  def check_new_resolutions(self):
    """
    Poll API for newly resolved markets.

    Returns list of new resolutions not previously tracked.
    """
    new_resolutions = []
    closed_markets = self.client.fetch_closed_markets()

    for market in closed_markets:
      if market.id in self._resolutions:
        continue

      resolution = self._market_to_resolution(market)
      if resolution:
        self._resolutions[market.id] = resolution
        new_resolutions.append(resolution)
        logger.info(f"New resolution: {resolution.question[:50]}... -> {resolution.outcome.name}")

    if new_resolutions:
      self._save()

    return new_resolutions

  def _market_to_resolution(self, market):
    """Convert Market object to MarketResolution."""
    try:
      outcome = Outcome.YES if market.yes_price > 0.5 else Outcome.NO

      # Parse resolution timestamp
      if market.end_date:
        resolved_at = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
      else:
        resolved_at = datetime.now(timezone.utc)

      return MarketResolution(
        market_id=market.id,
        resolved_at=resolved_at,
        outcome=outcome,
        question=market.question,
      )
    except Exception as e:
      logger.debug(f"Failed to parse resolution: {e}")
      return None

  def get(self, market_id):
    """Get resolution for a specific market."""
    return self._resolutions.get(market_id)

  def get_all(self):
    """Get all tracked resolutions."""
    return list(self._resolutions.values())


class IngestionRunner:
  """
  Runs continuous ingestion loop.

  Polls for snapshots and resolutions at configurable intervals.
  """

  def __init__(
    self,
    snapshot_interval=300,
    resolution_interval=60,
    min_liquidity=1000,
    min_volume=1000,
  ):
    self.snapshot_interval = snapshot_interval
    self.resolution_interval = resolution_interval
    self.min_liquidity = min_liquidity
    self.min_volume = min_volume

    client = PolymarketClient()
    self.snapshots = SnapshotIngester(client)
    self.resolutions = ResolutionTracker(client)

    self._last_snapshot = 0.0
    self._last_resolution = 0.0

  def run_once(self):
    """Run a single ingestion cycle."""
    now = time.time()
    new_snapshots = []
    new_resolutions = []

    if now - self._last_snapshot >= self.snapshot_interval:
      new_snapshots = self.snapshots.capture(
        min_liquidity=self.min_liquidity,
        min_volume=self.min_volume,
      )
      self._last_snapshot = now

    if now - self._last_resolution >= self.resolution_interval:
      new_resolutions = self.resolutions.check_new_resolutions()
      self._last_resolution = now

    return new_snapshots, new_resolutions

  def run_forever(self, on_resolution=None):
    """Run continuous ingestion loop."""
    logger.info("Starting ingestion loop...")

    while True:
      try:
        snapshots, resolutions = self.run_once()

        if on_resolution:
          for resolution in resolutions:
            on_resolution(resolution)

        time.sleep(min(self.snapshot_interval, self.resolution_interval) / 2)
      except KeyboardInterrupt:
        logger.info("Ingestion stopped")
        break
      except Exception as e:
        logger.error(f"Ingestion error: {e}")
        time.sleep(60)
