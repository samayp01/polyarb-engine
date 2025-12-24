"""Tests for backtest module."""

import json
from pathlib import Path

import pytest

from src.topic.backtest import BacktestEngine, Trade
from src.topic.models import Outcome


@pytest.fixture
def outcomes_file(tmp_path):
  """Create temporary outcomes file."""
  outcomes_path = tmp_path / "market_outcomes.json"
  outcomes_data = {
    "market1": "YES",
    "market2": "NO",
    "market3": "YES",
    "market4": "NO",
    "market5": "YES",
  }
  with open(outcomes_path, "w") as f:
    json.dump(outcomes_data, f)
  return outcomes_path


@pytest.fixture
def mock_graph(monkeypatch):
  """Mock EventGraph with test edges."""
  from src.topic.graph import EventGraph
  from src.topic.models import EventEdge, ConditionalDelta

  class MockGraph:
    def get_valid(self):
      delta_yes = ConditionalDelta(
        condition=Outcome.YES,
        avg_delta=0.3,
        median_delta=0.3,
        std_delta=0.1,
        avg_lag_seconds=3600,
        median_lag_seconds=3600,
        sample_count=1,
      )
      delta_no = ConditionalDelta(
        condition=Outcome.NO,
        avg_delta=-0.3,
        median_delta=-0.3,
        std_delta=0.1,
        avg_lag_seconds=3600,
        median_lag_seconds=3600,
        sample_count=1,
      )

      return [
        EventEdge(
          from_market_id="market1",
          to_market_id="market2",
          similarity=0.9,
          yes_delta=delta_yes,
          no_delta=delta_no,
          confidence=0.9,
          last_updated=None,
        ),
        EventEdge(
          from_market_id="market3",
          to_market_id="market4",
          similarity=0.85,
          yes_delta=delta_yes,
          no_delta=delta_no,
          confidence=0.85,
          last_updated=None,
        ),
      ]

  return MockGraph()


def test_backtest_loads_outcomes(outcomes_file, monkeypatch):
  """Test that backtest loads outcomes correctly."""
  monkeypatch.setattr("src.topic.backtest.OUTCOMES_FILE", outcomes_file)

  from src.topic.backtest import BacktestEngine

  engine = BacktestEngine()

  assert len(engine.outcomes) == 5
  assert engine.outcomes["market1"] == Outcome.YES
  assert engine.outcomes["market2"] == Outcome.NO


def test_backtest_deterministic_split(outcomes_file, monkeypatch, mock_graph):
  """Test that backtest produces same train/test split."""
  monkeypatch.setattr("src.topic.backtest.OUTCOMES_FILE", outcomes_file)

  from src.topic.backtest import BacktestEngine

  engine = BacktestEngine(graph=mock_graph)

  # Run twice
  result1 = engine.run(test_fraction=0.3)
  result2 = engine.run(test_fraction=0.3)

  # Should have identical results
  assert result1.total_signals == result2.total_signals
  assert result1.hit_rate == result2.hit_rate


def test_backtest_empty_result():
  """Test empty result when no edges."""
  from src.topic.backtest import BacktestEngine

  engine = BacktestEngine()
  result = engine._empty_result()

  assert result.total_signals == 0
  assert result.profitable_signals == 0
  assert result.hit_rate == 0.0


def test_trade_creation():
  """Test Trade dataclass creation."""
  trade = Trade(
    test_market_id="test123",
    train_market_id="train456",
    similarity=0.95,
    train_outcome=Outcome.YES,
    predicted_outcome=Outcome.YES,
    actual_outcome=Outcome.NO,
    profitable=False,
  )

  assert trade.test_market_id == "test123"
  assert trade.train_market_id == "train456"
  assert trade.similarity == 0.95
  assert not trade.profitable
