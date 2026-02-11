"""
Data Models

Core data structures for the trading system.
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
    """Point-in-time price snapshot."""

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
    """Record of a market's resolution."""

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
class EventEdge:
    """Directed edge representing a relationship between two markets."""

    from_market_id: str
    to_market_id: str
    similarity: float
    confidence: float = 0.0
    relation_type: str = "embedding"  # "embedding", "same_event", "causal"
    reasoning: str = ""
    last_updated: datetime = None

    def to_dict(self):
        return {
            "from_market_id": self.from_market_id,
            "to_market_id": self.to_market_id,
            "similarity": self.similarity,
            "confidence": self.confidence,
            "relation_type": self.relation_type,
            "reasoning": self.reasoning,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            from_market_id=data["from_market_id"],
            to_market_id=data["to_market_id"],
            similarity=data["similarity"],
            confidence=data.get("confidence", 0.0),
            relation_type=data.get("relation_type", "embedding"),
            reasoning=data.get("reasoning", ""),
            last_updated=datetime.fromisoformat(data["last_updated"])
            if data.get("last_updated")
            else None,
        )


@dataclass
class Signal:
    """Trading signal."""

    market_id: str
    direction: str  # "BUY" or "SELL"
    current_price: float
    expected_price: float
    confidence: float
    source_market_id: str
    source_outcome: Outcome
    generated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def expected_move(self):
        return self.expected_price - self.current_price

    def to_dict(self):
        return {
            "market_id": self.market_id,
            "direction": self.direction,
            "current_price": self.current_price,
            "expected_price": self.expected_price,
            "confidence": self.confidence,
            "source_market_id": self.source_market_id,
            "source_outcome": self.source_outcome.value if self.source_outcome else None,
            "generated_at": self.generated_at.isoformat(),
        }
