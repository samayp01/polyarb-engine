"""Tests for data models."""

from datetime import datetime, timezone

import pytest

from src.topic.models import (
  ConditionalDelta,
  EventEdge,
  MarketResolution,
  Outcome,
  Signal,
)


def test_outcome_enum():
  """Test Outcome enum values."""
  assert Outcome.YES.value == "YES"
  assert Outcome.NO.value == "NO"
  assert Outcome("YES") == Outcome.YES


def test_conditional_delta_creation():
  """Test ConditionalDelta dataclass."""
  delta = ConditionalDelta(
    condition=Outcome.YES,
    avg_delta=0.25,
    median_delta=0.24,
    std_delta=0.05,
    avg_lag_seconds=1800,
    median_lag_seconds=1700,
    sample_count=10,
  )

  assert delta.avg_delta == 0.25
  assert delta.sample_count == 10
  assert delta.condition == Outcome.YES


def test_event_edge_get_delta():
  """Test EventEdge get_delta method."""
  yes_delta = ConditionalDelta(
    condition=Outcome.YES,
    avg_delta=0.3,
    median_delta=0.3,
    std_delta=0.1,
    avg_lag_seconds=3600,
    median_lag_seconds=3600,
    sample_count=5,
  )
  no_delta = ConditionalDelta(
    condition=Outcome.NO,
    avg_delta=-0.2,
    median_delta=-0.2,
    std_delta=0.1,
    avg_lag_seconds=3600,
    median_lag_seconds=3600,
    sample_count=5,
  )

  edge = EventEdge(
    from_market_id="123",
    to_market_id="456",
    similarity=0.9,
    yes_delta=yes_delta,
    no_delta=no_delta,
    confidence=0.85,
    last_updated=datetime.now(timezone.utc),
  )

  assert edge.get_delta(Outcome.YES) == yes_delta
  assert edge.get_delta(Outcome.NO) == no_delta


def test_event_edge_is_valid():
  """Test EventEdge validation logic."""
  yes_delta = ConditionalDelta(
    condition=Outcome.YES,
    avg_delta=0.05,
    median_delta=0.05,
    std_delta=0.01,
    avg_lag_seconds=120,
    median_lag_seconds=120,
    sample_count=10,
  )

  edge = EventEdge(
    from_market_id="123",
    to_market_id="456",
    similarity=0.9,
    yes_delta=yes_delta,
    no_delta=None,
    confidence=0.85,
    last_updated=datetime.now(timezone.utc),
  )

  # Should be valid with these defaults
  assert edge.is_valid(min_samples=1, min_delta=0.03, min_lag=60)

  # Should fail with higher thresholds
  assert not edge.is_valid(min_samples=20, min_delta=0.03, min_lag=60)
  assert not edge.is_valid(min_samples=1, min_delta=0.10, min_lag=60)
  assert not edge.is_valid(min_samples=1, min_delta=0.03, min_lag=200)


def test_event_edge_serialization():
  """Test EventEdge to_dict and from_dict."""
  now = datetime.now(timezone.utc)
  yes_delta = ConditionalDelta(
    condition=Outcome.YES,
    avg_delta=0.3,
    median_delta=0.3,
    std_delta=0.1,
    avg_lag_seconds=3600,
    median_lag_seconds=3600,
    sample_count=5,
  )

  edge = EventEdge(
    from_market_id="123",
    to_market_id="456",
    similarity=0.9,
    yes_delta=yes_delta,
    no_delta=None,
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
    expected_move=0.15,
    current_price=0.5,
    expected_price=0.65,
    confidence=0.8,
    source_edge=None,
    generated_at=now,
    leader_market_id="456",
    leader_outcome=Outcome.YES,
  )

  assert signal.market_id == "123"
  assert signal.direction == "BUY"
  assert signal.expected_move == 0.15
  assert signal.confidence == 0.8
