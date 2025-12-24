"""
Data Models

Core data structures for the Polymarket trading system:
- MarketSnapshot: Point-in-time price data
- MarketResolution: Market resolution outcome
- EventEdge: Relationship between markets
- Signal: Trading signal
- BacktestResult: Backtest performance metrics
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Outcome(Enum):
  """Market resolution outcome."""
  YES = True
  NO = False


@dataclass
class MarketSnapshot:
  """
  Point-in-time price snapshot for a market.

  Used to track historical prices for analyzing
  how markets react to related market resolutions.
  """
  market_id: str
  timestamp: datetime
  yes_price: float
  volume: float
  liquidity: float

  def to_dict(self):
    return {
      "market_id": self.market_id,
      "timestamp": self.timestamp.isoformat(),
      "yes_price": self.yes_price,
      "volume": self.volume,
      "liquidity": self.liquidity,
    }

  @classmethod
  def from_dict(cls, data):
    return cls(
      market_id=data["market_id"],
      timestamp=datetime.fromisoformat(data["timestamp"]),
      yes_price=data["yes_price"],
      volume=data["volume"],
      liquidity=data["liquidity"],
    )


@dataclass
class MarketResolution:
  """
  Record of a market's final resolution.

  When a market resolves, we check if any related markets
  should reprice based on the outcome.
  """
  market_id: str
  resolved_at: datetime
  outcome: Outcome
  question: str = ""

  def to_dict(self):
    return {
      "market_id": self.market_id,
      "resolved_at": self.resolved_at.isoformat(),
      "outcome": self.outcome.value,
      "question": self.question,
    }

  @classmethod
  def from_dict(cls, data):
    return cls(
      market_id=data["market_id"],
      resolved_at=datetime.fromisoformat(data["resolved_at"]),
      outcome=Outcome(data["outcome"]),
      question=data.get("question", ""),
    )


@dataclass
class ConditionalDelta:
  """
  Price movement statistics conditioned on a leader market's outcome.

  Tracks how follower market prices historically moved when
  the leader market resolved to a specific outcome.
  """
  condition: Outcome
  avg_delta: float      # Average price change
  median_delta: float   # Median price change
  std_delta: float      # Standard deviation
  avg_lag_seconds: float    # Average time to reach new price
  median_lag_seconds: float
  sample_count: int     # Number of observations

  def to_dict(self):
    return {
      "condition": self.condition.value,
      "avg_delta": self.avg_delta,
      "median_delta": self.median_delta,
      "std_delta": self.std_delta,
      "avg_lag_seconds": self.avg_lag_seconds,
      "median_lag_seconds": self.median_lag_seconds,
      "sample_count": self.sample_count,
    }

  @classmethod
  def from_dict(cls, data):
    return cls(
      condition=Outcome(data["condition"]),
      avg_delta=data["avg_delta"],
      median_delta=data["median_delta"],
      std_delta=data["std_delta"],
      avg_lag_seconds=data["avg_lag_seconds"],
      median_lag_seconds=data["median_lag_seconds"],
      sample_count=data["sample_count"],
    )


@dataclass
class EventEdge:
  """
  Directed edge representing a relationship between two markets.

  When the 'from' market (leader) resolves, we expect the 'to' market
  (follower) to reprice according to the learned conditional deltas.
  """
  from_market_id: str   # Leader market
  to_market_id: str     # Follower market
  similarity: float     # Semantic similarity (0-1)
  yes_delta: ConditionalDelta = None  # Price change when leader resolves YES
  no_delta: ConditionalDelta = None   # Price change when leader resolves NO
  confidence: float = 0.0
  last_updated: datetime = None

  def to_dict(self):
    return {
      "from_market_id": self.from_market_id,
      "to_market_id": self.to_market_id,
      "similarity": self.similarity,
      "yes_delta": self.yes_delta.to_dict() if self.yes_delta else None,
      "no_delta": self.no_delta.to_dict() if self.no_delta else None,
      "confidence": self.confidence,
      "last_updated": self.last_updated.isoformat() if self.last_updated else None,
    }

  @classmethod
  def from_dict(cls, data):
    return cls(
      from_market_id=data["from_market_id"],
      to_market_id=data["to_market_id"],
      similarity=data["similarity"],
      yes_delta=ConditionalDelta.from_dict(data["yes_delta"]) if data.get("yes_delta") else None,
      no_delta=ConditionalDelta.from_dict(data["no_delta"]) if data.get("no_delta") else None,
      confidence=data.get("confidence", 0.0),
      last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else None,
    )

  def get_delta(self, outcome):
    """Get the conditional delta for a specific outcome."""
    return self.yes_delta if outcome == Outcome.YES else self.no_delta

  def is_valid(self, min_samples=1, min_delta=0.03, min_lag=60):
    """Check if edge has enough historical data to be actionable."""
    for delta in [self.yes_delta, self.no_delta]:
      if delta is None:
        continue
      if delta.sample_count < min_samples:
        return False
      if abs(delta.avg_delta) < min_delta:
        return False
      if delta.avg_lag_seconds < min_lag:
        return False
    return self.yes_delta is not None or self.no_delta is not None


@dataclass
class Signal:
  """
  Trading signal generated when a leader market resolves.

  Indicates an expected price movement in a follower market
  based on the learned conditional relationship.
  """
  market_id: str          # Market to trade
  direction: str          # "BUY" or "SELL"
  expected_move: float    # Expected price change
  current_price: float    # Price at signal generation
  expected_price: float   # Expected future price
  confidence: float       # Signal confidence
  source_edge: EventEdge  # Edge that generated this signal
  generated_at: datetime = field(default_factory=datetime.utcnow)
  leader_market_id: str = ""
  leader_outcome: Outcome = None

  def to_dict(self):
    return {
      "market_id": self.market_id,
      "direction": self.direction,
      "expected_move": self.expected_move,
      "current_price": self.current_price,
      "expected_price": self.expected_price,
      "confidence": self.confidence,
      "source_edge": self.source_edge.to_dict(),
      "generated_at": self.generated_at.isoformat(),
      "leader_market_id": self.leader_market_id,
      "leader_outcome": self.leader_outcome.value if self.leader_outcome else None,
    }

  @classmethod
  def from_dict(cls, data):
    return cls(
      market_id=data["market_id"],
      direction=data["direction"],
      expected_move=data["expected_move"],
      current_price=data["current_price"],
      expected_price=data["expected_price"],
      confidence=data["confidence"],
      source_edge=EventEdge.from_dict(data["source_edge"]),
      generated_at=datetime.fromisoformat(data["generated_at"]),
      leader_market_id=data.get("leader_market_id", ""),
      leader_outcome=Outcome(data["leader_outcome"]) if data.get("leader_outcome") is not None else None,
    )


@dataclass
class BacktestResult:
  """Results from backtesting signals on historical data."""
  total_signals: int
  profitable_signals: int
  total_pnl: float
  avg_pnl_per_signal: float
  hit_rate: float
  avg_decay_vs_lag: float
  signals: list = field(default_factory=list)

  def to_dict(self):
    return {
      "total_signals": self.total_signals,
      "profitable_signals": self.profitable_signals,
      "total_pnl": self.total_pnl,
      "avg_pnl_per_signal": self.avg_pnl_per_signal,
      "hit_rate": self.hit_rate,
      "avg_decay_vs_lag": self.avg_decay_vs_lag,
      "signals": self.signals,
    }
