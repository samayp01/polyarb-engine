"""Tests for graph module."""

from datetime import datetime, timezone

import pytest

from src.topic.graph import EventGraph
from src.topic.utils.models import EventEdge


@pytest.fixture
def sample_edges():
    """Create sample edges for testing."""
    return [
        EventEdge(
            from_market_id="m1",
            to_market_id="m2",
            similarity=0.95,
            confidence=0.9,
            last_updated=datetime.now(timezone.utc),
        ),
        EventEdge(
            from_market_id="m1",
            to_market_id="m3",
            similarity=0.85,
            confidence=0.8,
            last_updated=datetime.now(timezone.utc),
        ),
    ]


def test_graph_add_edges(sample_edges, tmp_path):
    """Test adding edges to graph."""
    graph = EventGraph(filepath=tmp_path / "test.json")
    graph.add_edges(sample_edges)

    stats = graph.stats()
    assert stats["total_edges"] == 2


def test_graph_get_related(sample_edges, tmp_path):
    """Test retrieving related edges."""
    graph = EventGraph(filepath=tmp_path / "test.json")
    graph.add_edges(sample_edges)

    related = graph.get_related("m1")
    assert len(related) == 2
    assert all(e.from_market_id == "m1" for e in related)


def test_graph_clear(sample_edges, tmp_path):
    """Test clearing graph."""
    graph = EventGraph(filepath=tmp_path / "test.json")
    graph.add_edges(sample_edges)
    assert graph.stats()["total_edges"] == 2

    graph.clear()
    assert graph.stats()["total_edges"] == 0


def test_graph_stats(sample_edges, tmp_path):
    """Test graph statistics."""
    graph = EventGraph(filepath=tmp_path / "test.json")
    graph.add_edges(sample_edges)

    stats = graph.stats()
    assert stats["total_edges"] == 2
    assert stats["unique_sources"] == 1


def test_graph_persistence(sample_edges, tmp_path):
    """Test saving and loading graph."""
    test_file = tmp_path / "test_graph.json"

    graph = EventGraph(filepath=test_file)
    graph.add_edges(sample_edges)
    graph._save()

    loaded_graph = EventGraph(filepath=test_file)
    assert loaded_graph.stats()["total_edges"] == 2


def test_graph_no_duplicate_edges(tmp_path):
    """Test that duplicate edges are replaced."""
    graph = EventGraph(filepath=tmp_path / "test.json")

    edge1 = EventEdge(
        from_market_id="m1",
        to_market_id="m2",
        similarity=0.95,
        confidence=0.9,
        last_updated=datetime.now(timezone.utc),
    )

    edge2 = EventEdge(
        from_market_id="m1",
        to_market_id="m2",
        similarity=0.96,
        confidence=0.91,
        last_updated=datetime.now(timezone.utc),
    )

    graph.add_edges([edge1, edge2])

    # Latest version should replace previous
    assert graph.stats()["total_edges"] == 1
    related = graph.get_related("m1")
    assert related[0].similarity == 0.96
