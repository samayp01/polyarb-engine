"""Tests for data models."""

from datetime import datetime, timezone

from src.topic.utils.models import (
    EventEdge,
    MarketResolution,
    Outcome,
    Signal,
)


def test_outcome_enum():
    """Test Outcome enum values."""
    assert Outcome.YES.value is True
    assert Outcome.NO.value is False
    assert Outcome(True) == Outcome.YES


def test_event_edge_creation():
    """Test EventEdge dataclass."""
    edge = EventEdge(
        from_market_id="123",
        to_market_id="456",
        similarity=0.9,
        confidence=0.85,
        last_updated=datetime.now(timezone.utc),
    )

    assert edge.from_market_id == "123"
    assert edge.to_market_id == "456"
    assert edge.similarity == 0.9


def test_event_edge_serialization():
    """Test EventEdge to_dict and from_dict."""
    now = datetime.now(timezone.utc)

    edge = EventEdge(
        from_market_id="123",
        to_market_id="456",
        similarity=0.9,
        confidence=0.85,
        last_updated=now,
    )

    edge_dict = edge.to_dict()
    assert edge_dict["from_market_id"] == "123"
    assert edge_dict["similarity"] == 0.9

    restored = EventEdge.from_dict(edge_dict)
    assert restored.from_market_id == "123"
    assert restored.to_market_id == "456"


def test_market_resolution():
    """Test MarketResolution creation."""
    now = datetime.now(timezone.utc)
    resolution = MarketResolution(
        market_id="789",
        resolved_at=now,
        outcome=Outcome.YES,
        question="Will it rain?",
    )

    assert resolution.market_id == "789"
    assert resolution.outcome == Outcome.YES
    assert resolution.question == "Will it rain?"

    # Test serialization
    res_dict = resolution.to_dict()
    restored = MarketResolution.from_dict(res_dict)
    assert restored.market_id == "789"
    assert restored.outcome == Outcome.YES


def test_signal_creation():
    """Test Signal dataclass."""
    now = datetime.now(timezone.utc)

    signal = Signal(
        market_id="123",
        direction="BUY",
        current_price=0.5,
        expected_price=0.65,
        confidence=0.8,
        source_market_id="456",
        source_outcome=Outcome.YES,
        generated_at=now,
    )

    assert signal.market_id == "123"
    assert signal.direction == "BUY"
    assert abs(signal.expected_move - 0.15) < 1e-10
    assert signal.confidence == 0.8
