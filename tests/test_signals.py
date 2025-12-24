"""Tests for signals module."""

from datetime import datetime, timezone

import pytest

from src.topic.models import ConditionalDelta, EventEdge, MarketResolution, Outcome


@pytest.fixture
def mock_edge():
  """Create a mock edge for testing."""
  yes_delta = ConditionalDelta(
    condition=Outcome.YES,
    avg_delta=0.15,
    median_delta=0.15,
    std_delta=0.05,
    avg_lag_seconds=3600,
    median_lag_seconds=3600,
    sample_count=5,
  )

  return EventEdge(
    from_market_id="leader123",
    to_market_id="follower456",
    similarity=0.9,
    yes_delta=yes_delta,
    no_delta=None,
    confidence=0.85,
    last_updated=datetime.now(timezone.utc),
  )


@pytest.fixture
def mock_resolution():
  """Create a mock resolution."""
  return MarketResolution(
    market_id="leader123",
    resolved_at=datetime.now(timezone.utc),
    outcome=Outcome.YES,
    question="Leader market question",
  )


def test_signal_engine_initialization():
  """Test SignalEngine creates empty state."""
  from src.topic.signals import SignalEngine

  engine = SignalEngine()
  assert len(engine.get_all()) == 0


def test_signal_engine_get_price():
  """Test getting market price."""
  from src.topic.signals import SignalEngine

  engine = SignalEngine()

  # Mock markets cache
  from src.topic.client import Market

  engine._markets = {
    "test123": Market(
      id="test123",
      question="Test?",
      description="Test",
      slug="test",
      outcomes=["Yes", "No"],
      prices=[0.6, 0.4],
      volume=1000.0,
      liquidity=500.0,
      end_date="2024-12-31",
    )
  }

  price = engine.get_price("test123")
  assert price == 0.6


def test_signal_engine_generate_signal_threshold(mock_edge, mock_resolution):
  """Test signal generation respects minimum thresholds."""
  from src.topic.signals import SignalEngine

  engine = SignalEngine()
  engine._markets = {
    "follower456": type(
      "Market",
      (),
      {
        "yes_price": 0.5,
        "liquidity": 10000,
      },
    )()
  }

  # Edge delta is 0.15, current price 0.5, expected 0.65
  # Mispricing = 0.15, which exceeds MIN_MISPRICING (0.03)
  signal = engine._generate_signal(mock_resolution, mock_edge)

  assert signal is not None
  assert signal.direction == "BUY"
  assert abs(signal.expected_move - 0.15) < 0.001


def test_signal_engine_below_threshold():
  """Test no signal when mispricing below threshold."""
  from src.topic.signals import SignalEngine

  # Create edge with small delta
  small_delta = ConditionalDelta(
    condition=Outcome.YES,
    avg_delta=0.01,  # Below MIN_MISPRICING
    median_delta=0.01,
    std_delta=0.01,
    avg_lag_seconds=3600,
    median_lag_seconds=3600,
    sample_count=5,
  )

  edge = EventEdge(
    from_market_id="leader",
    to_market_id="follower",
    similarity=0.9,
    yes_delta=small_delta,
    no_delta=None,
    confidence=0.85,
    last_updated=datetime.now(timezone.utc),
  )

  resolution = MarketResolution(
    market_id="leader",
    resolved_at=datetime.now(timezone.utc),
    outcome=Outcome.YES,
    question="Test",
  )

  engine = SignalEngine()
  engine._markets = {
    "follower": type("Market", (), {"yes_price": 0.5, "liquidity": 10000})()
  }

  signal = engine._generate_signal(resolution, edge)
  assert signal is None


def test_signal_monitor_initialization():
  """Test SignalMonitor creates components."""
  from src.topic.signals import SignalMonitor

  monitor = SignalMonitor()
  assert monitor.engine is not None
  assert monitor.tracker is not None


def test_signal_engine_get_recent():
  """Test getting recent signals."""
  from src.topic.signals import SignalEngine

  engine = SignalEngine()

  # No signals initially
  recent = engine.get_recent(max_age_hours=24)
  assert len(recent) == 0
