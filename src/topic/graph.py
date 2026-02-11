"""
Event Graph

Stores relationships between markets based on semantic similarity.
When a source market resolves, related target markets may be affected.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import MAX_DAYS_APART, MIN_EMBEDDING_SIMILARITY
from .utils.embeddings import cluster_markets
from .utils.models import EventEdge

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
GRAPH_FILE = DATA_DIR / "event_graph.json"


class EventGraph:
    """Graph of market relationships."""

    def __init__(self, filepath=GRAPH_FILE):
        self.filepath = filepath
        self._edges = {}  # key -> EventEdge
        self._outgoing = {}  # source_id -> [keys]
        self._load()

    def _key(self, from_id, to_id):
        return f"{from_id}::{to_id}"

    def _load(self):
        if not self.filepath.exists():
            return
        try:
            with self.filepath.open() as f:
                data = json.load(f)
            for edge_data in data.get("edges", []):
                edge = EventEdge.from_dict(edge_data)
                self._index_edge(edge)
            logger.info(f"Loaded {len(self._edges)} edges")
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")

    def _save(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "edges": [e.to_dict() for e in self._edges.values()],
        }
        with self.filepath.open("w") as f:
            json.dump(data, f, indent=2)

    def _index_edge(self, edge):
        key = self._key(edge.from_market_id, edge.to_market_id)
        self._edges[key] = edge
        self._outgoing.setdefault(edge.from_market_id, [])
        if key not in self._outgoing[edge.from_market_id]:
            self._outgoing[edge.from_market_id].append(key)

    def add_edges(self, edges):
        """Add multiple edges."""
        for edge in edges:
            self._index_edge(edge)
        self._save()

    def get_related(self, market_id):
        """Get edges where given market is the source."""
        keys = self._outgoing.get(market_id, [])
        return [self._edges[k] for k in keys if k in self._edges]

    def clear(self):
        """Remove all edges."""
        self._edges.clear()
        self._outgoing.clear()
        self._save()

    def stats(self):
        """Get graph statistics."""
        return {
            "total_edges": len(self._edges),
            "unique_sources": len(self._outgoing),
        }


def _parse_end_date(end_date_str):
    """Parse end_date string to datetime, returning None on failure."""
    if not end_date_str:
        return None
    try:
        if "T" in end_date_str:
            return datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        return datetime.fromisoformat(end_date_str)
    except (ValueError, TypeError):
        return None


def _order_by_end_date(market_a, market_b, max_days_apart=None):
    """
    Order markets so the earlier-resolving one is first.

    Returns (source, target) where source.end_date <= target.end_date,
    or (None, None) if dates can't be parsed or are too far apart.
    """
    if max_days_apart is None:
        max_days_apart = MAX_DAYS_APART
    date_a = _parse_end_date(market_a.end_date)
    date_b = _parse_end_date(market_b.end_date)

    if date_a is None or date_b is None:
        return None, None

    diff = abs((date_a - date_b).days)
    if diff > max_days_apart:
        return None, None

    if date_a <= date_b:
        return market_a, market_b
    return market_b, market_a


def build_embedding_graph(markets, min_similarity=None):
    """
    Build embedding-based graph from markets.

    Creates edges between semantically similar markets, ordered chronologically
    (source resolves before target) and filtered by time proximity.

    Args:
        markets: List of Market objects.
        min_similarity: Minimum embedding similarity threshold.

    Returns:
        List of EventEdge objects.
    """
    if min_similarity is None:
        min_similarity = MIN_EMBEDDING_SIMILARITY

    if len(markets) < 2:
        return []

    logger.info(f"Building embedding graph from {len(markets)} markets...")
    candidate_pairs = cluster_markets(markets, min_similarity=min_similarity)
    logger.info(f"Found {len(candidate_pairs)} pairs above similarity {min_similarity}")

    if not candidate_pairs:
        return []

    edges = []
    for m1, m2, sim in candidate_pairs:
        source, target = _order_by_end_date(m1, m2)
        if source is not None and target is not None:
            edge = EventEdge(
                from_market_id=source.id,
                to_market_id=target.id,
                similarity=sim,
                confidence=sim,
                relation_type="embedding",
                last_updated=datetime.now(timezone.utc),
            )
            edges.append(edge)

    logger.info(f"Created {len(edges)} edges after time filtering")
    return edges


def save_graph(edges, markets, filepath=GRAPH_FILE):
    """
    Save graph with market details.

    Args:
        edges: List of EventEdge objects.
        markets: List of Market objects.
        filepath: Path to save the JSON file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    market_by_id = {m.id: m for m in markets}

    edge_data = []
    for edge in edges:
        source = market_by_id.get(edge.from_market_id)
        target = market_by_id.get(edge.to_market_id)

        edge_data.append(
            {
                "from_market_id": edge.from_market_id,
                "to_market_id": edge.to_market_id,
                "similarity": edge.similarity,
                "source_question": source.question if source else "Unknown",
                "source_end_date": source.end_date if source else None,
                "source_yes_price": source.yes_price if source else None,
                "target_question": target.question if target else "Unknown",
                "target_end_date": target.end_date if target else None,
                "target_yes_price": target.yes_price if target else None,
            }
        )

    data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_edges": len(edges),
        "min_similarity": MIN_EMBEDDING_SIMILARITY,
        "edges": edge_data,
    }

    with filepath.open("w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved graph with {len(edges)} edges to {filepath}")


def load_graph(filepath=GRAPH_FILE):
    """
    Load graph from file.

    Returns:
        Tuple of (edges_data, metadata) where edges_data is a list of dicts.
    """
    if not filepath.exists():
        logger.warning(f"Graph not found at {filepath}")
        return [], {}

    with filepath.open() as f:
        data = json.load(f)

    edges = data.get("edges", [])
    metadata = {
        "created_at": data.get("created_at"),
        "total_edges": data.get("total_edges"),
        "min_similarity": data.get("min_similarity"),
    }

    logger.info(f"Loaded {len(edges)} edges from graph")
    return edges, metadata
