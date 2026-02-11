"""
Signal Engine

Generates trading signals when source markets resolve.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .graph import EventGraph
from .utils.client import fetch_markets
from .utils.models import Outcome, Signal

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
SIGNALS_FILE = DATA_DIR / "signals.json"


class SignalEngine:
    """Generates trading signals from market resolutions."""

    MIN_MISPRICING = 0.05
    MIN_CONFIDENCE = 0.5
    MIN_LIQUIDITY = 5000

    def __init__(self, graph=None):
        self.graph = graph or EventGraph()
        self._markets = {}
        self._signals = []
        self._load_signals()

    def _load_signals(self):
        if SIGNALS_FILE.exists():
            try:
                with SIGNALS_FILE.open() as f:
                    self._signals = json.load(f)
            except Exception:
                pass

    def _save_signals(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with SIGNALS_FILE.open("w") as f:
            json.dump(self._signals, f, indent=2)

    def refresh_markets(self):
        """Refresh cache of active markets."""
        markets = fetch_markets(min_liquidity=self.MIN_LIQUIDITY)
        self._markets = {m.id: m for m in markets}
        logger.info(f"Refreshed {len(self._markets)} active markets")

    def on_resolution(self, market_id, outcome):
        """
        Handle a market resolution and generate signals.

        Args:
            market_id: ID of the resolved market.
            outcome: Outcome.YES or Outcome.NO

        Returns:
            List of Signal objects.
        """
        logger.info(f"Processing resolution: {market_id[:12]}... -> {outcome.name}")

        edges = self.graph.get_related(market_id)
        if not edges:
            return []

        if not self._markets:
            self.refresh_markets()

        signals = []
        for edge in edges:
            signal = self._generate_signal(market_id, outcome, edge)
            if signal:
                signals.append(signal)
                self._signals.append(signal.to_dict())

        if signals:
            self._save_signals()
            logger.info(f"Generated {len(signals)} signals")

        return signals

    def _generate_signal(self, source_id, outcome, edge):
        """Generate signal for a single edge."""
        if edge.confidence < self.MIN_CONFIDENCE:
            return None

        target = self._markets.get(edge.to_market_id)
        if not target or target.liquidity < self.MIN_LIQUIDITY:
            return None

        current_price = target.yes_price

        # Correlated markets move in same direction
        if outcome == Outcome.YES:
            expected_price = min(1.0, current_price + edge.similarity * 0.2)
        else:
            expected_price = max(0.0, current_price - edge.similarity * 0.2)

        mispricing = expected_price - current_price
        if abs(mispricing) < self.MIN_MISPRICING:
            return None

        direction = "BUY" if mispricing > 0 else "SELL"

        return Signal(
            market_id=edge.to_market_id,
            direction=direction,
            current_price=current_price,
            expected_price=expected_price,
            confidence=edge.confidence,
            source_market_id=source_id,
            source_outcome=outcome,
            generated_at=datetime.now(timezone.utc),
        )

    def get_signals(self):
        """Get all signals."""
        return self._signals
