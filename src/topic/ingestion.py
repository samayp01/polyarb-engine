"""
Data Ingestion

Tracks market resolutions by polling for closed markets.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .utils.client import fetch_closed_markets
from .utils.models import MarketResolution, Outcome

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
RESOLUTIONS_FILE = DATA_DIR / "resolutions.json"


class ResolutionTracker:
    """Tracks market resolutions."""

    def __init__(self):
        self._resolutions = {}
        self._load()

    def _load(self):
        if RESOLUTIONS_FILE.exists():
            try:
                with RESOLUTIONS_FILE.open() as f:
                    data = json.load(f)
                for item in data:
                    res = MarketResolution.from_dict(item)
                    self._resolutions[res.market_id] = res
                logger.info(f"Loaded {len(self._resolutions)} resolutions")
            except Exception as e:
                logger.error(f"Failed to load resolutions: {e}")

    def _save(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with RESOLUTIONS_FILE.open("w") as f:
            json.dump([r.to_dict() for r in self._resolutions.values()], f, indent=2)

    def check_new(self):
        """
        Poll for newly resolved markets.

        Returns list of new MarketResolution objects.
        """
        new_resolutions = []
        closed_markets = fetch_closed_markets()

        for market in closed_markets:
            if market.id in self._resolutions:
                continue

            resolution = self._to_resolution(market)
            if resolution:
                self._resolutions[market.id] = resolution
                new_resolutions.append(resolution)
                logger.info(f"New: {resolution.question[:40]}... -> {resolution.outcome.name}")

        if new_resolutions:
            self._save()

        return new_resolutions

    def _to_resolution(self, market):
        try:
            outcome = Outcome.YES if market.yes_price > 0.5 else Outcome.NO
            resolved_at = (
                datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
                if market.end_date
                else datetime.now(timezone.utc)
            )
            return MarketResolution(
                market_id=market.id,
                resolved_at=resolved_at,
                outcome=outcome,
                question=market.question,
            )
        except Exception:
            return None

    def get_all(self):
        return list(self._resolutions.values())
