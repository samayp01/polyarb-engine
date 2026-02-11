"""Tests for signals module."""

from datetime import datetime, timezone

import pytest

from src.topic.utils.models import EventEdge, Outcome


@pytest.fixture
def mock_edge():
    """Create a mock edge for testing."""
    return EventEdge(
        from_market_id="source123",
        to_market_id="target456",
        similarity=0.9,
        confidence=0.85,
        last_updated=datetime.now(timezone.utc),
    )


def test_signal_engine_initialization():
    """Test SignalEngine creates empty state."""
    from src.topic.signals import SignalEngine

    engine = SignalEngine()
    assert len(engine.get_signals()) == 0


def test_signal_engine_generate_signal(mock_edge):
    """Test signal generation."""
    from src.topic.signals import SignalEngine
    from src.topic.utils.client import Market

    engine = SignalEngine()
    engine._markets = {
        "target456": Market(
            id="target456",
            question="Test?",
            description="Test",
            slug="test",
            outcomes=["Yes", "No"],
            prices=[0.5, 0.5],
            volume=10000.0,
            liquidity=10000.0,
            end_date="2024-12-31",
        )
    }

    signal = engine._generate_signal("source123", Outcome.YES, mock_edge)

    assert signal is not None
    assert signal.direction == "BUY"
    assert signal.expected_price > signal.current_price


def test_signal_engine_below_threshold():
    """Test no signal when mispricing below threshold."""
    from src.topic.signals import SignalEngine
    from src.topic.utils.client import Market

    edge = EventEdge(
        from_market_id="source_a",
        to_market_id="target_b",
        similarity=0.1,  # Low similarity = small expected move
        confidence=0.85,
        last_updated=datetime.now(timezone.utc),
    )

    engine = SignalEngine()
    engine._markets = {
        "target_b": Market(
            id="target_b",
            question="Test?",
            description="Test",
            slug="test",
            outcomes=["Yes", "No"],
            prices=[0.5, 0.5],
            volume=10000.0,
            liquidity=10000.0,
            end_date="2024-12-31",
        )
    }

    signal = engine._generate_signal("source_a", Outcome.YES, edge)
    # 0.1 * 0.2 = 0.02 < MIN_MISPRICING (0.05)
    assert signal is None
