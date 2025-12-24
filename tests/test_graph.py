"""Tests for graph module."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.topic.graph import EventGraph
from src.topic.models import ConditionalDelta, EventEdge, Outcome


@pytest.fixture
def sample_edges():
  """Create sample edges for testing."""
  yes_delta = ConditionalDelta(
    condition=Outcome.YES,
    avg_delta=0.3,
    median_delta=0.3,
    std_delta=0.1,
    avg_lag_seconds=3600,
    median_lag_seconds=3600,
    sample_count=5,
  )

  edges = [
    EventEdge(
      from_market_id="m1",
      to_market_id="m2",
      similarity=0.95,
      yes_delta=yes_delta,
      no_delta=None,
      confidence=0.9,
      last_updated=datetime.now(timezone.utc),
    ),
    EventEdge(
      from_market_id="m1",
      to_market_id="m3",
      similarity=0.85,
      yes_delta=yes_delta,
      no_delta=None,
      confidence=0.8,
      last_updated=datetime.now(timezone.utc),
    ),
  ]
  return edges


def test_graph_add_edges(sample_edges):
  """Test adding edges to graph."""
  graph = EventGraph()
  graph.add_edges(sample_edges)

  assert len(graph.get_all()) == 2
  assert graph.stats()["total_edges"] == 2


def test_graph_get_outgoing(sample_edges):
  """Test retrieving outgoing edges."""
  graph = EventGraph()
  graph.add_edges(sample_edges)

  outgoing = graph.get_outgoing("m1")
  assert len(outgoing) == 2
  assert all(e.from_market_id == "m1" for e in outgoing)


def test_graph_get_incoming(sample_edges):
  """Test retrieving incoming edges."""
  graph = EventGraph()
  graph.add_edges(sample_edges)

  incoming = graph.get_incoming("m2")
  assert len(incoming) == 1
  assert incoming[0].to_market_id == "m2"


def test_graph_get_valid(sample_edges):
  """Test filtering valid edges."""
  graph = EventGraph()
  graph.add_edges(sample_edges)

  valid = graph.get_valid(min_samples=1, min_delta=0.03, min_lag=60)
  assert len(valid) == 2

  # Higher threshold should filter some out
  valid_strict = graph.get_valid(min_samples=10, min_delta=0.03, min_lag=60)
  assert len(valid_strict) == 0


def test_graph_clear(sample_edges):
  """Test clearing graph."""
  graph = EventGraph()
  graph.add_edges(sample_edges)
  assert len(graph.get_all()) == 2

  graph.clear()
  assert len(graph.get_all()) == 0


def test_graph_stats(sample_edges):
  """Test graph statistics."""
  graph = EventGraph()
  graph.add_edges(sample_edges)

  stats = graph.stats()
  assert stats["total_edges"] == 2
  assert stats["valid_edges"] == 2
  assert stats["unique_leaders"] == 1
  assert stats["unique_followers"] == 2


def test_graph_persistence(sample_edges, tmp_path):
  """Test saving and loading graph."""
  graph = EventGraph()
  graph.add_edges(sample_edges)

  # Override data file path
  test_file = tmp_path / "test_graph.json"
  import src.topic.graph as graph_module

  original_file = graph_module.GRAPH_FILE
  graph_module.GRAPH_FILE = test_file

  try:
    # Save
    graph_module.EventGraph.save = lambda self: self._save()
    new_graph = EventGraph()
    new_graph._edges = graph._edges
    new_graph._outgoing = graph._outgoing
    new_graph._incoming = graph._incoming

    # Manual save
    data = [e.to_dict() for e in new_graph.get_all()]
    with open(test_file, "w") as f:
      json.dump(data, f)

    # Load
    loaded_graph = EventGraph()
    assert len(loaded_graph.get_all()) == 2

  finally:
    graph_module.GRAPH_FILE = original_file


def test_graph_no_duplicate_edges():
  """Test that duplicate edges are handled."""
  graph = EventGraph()

  yes_delta = ConditionalDelta(
    condition=Outcome.YES,
    avg_delta=0.3,
    median_delta=0.3,
    std_delta=0.1,
    avg_lag_seconds=3600,
    median_lag_seconds=3600,
    sample_count=5,
  )

  edge1 = EventEdge(
    from_market_id="m1",
    to_market_id="m2",
    similarity=0.95,
    yes_delta=yes_delta,
    no_delta=None,
    confidence=0.9,
    last_updated=datetime.now(timezone.utc),
  )

  edge2 = EventEdge(
    from_market_id="m1",
    to_market_id="m2",
    similarity=0.96,
    yes_delta=yes_delta,
    no_delta=None,
    confidence=0.91,
    last_updated=datetime.now(timezone.utc),
  )

  graph.add_edges([edge1, edge2])

  # Should only keep latest version
  assert len(graph.get_all()) == 1
  assert graph.get_all()[0].similarity == 0.96
