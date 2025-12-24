"""
Event Graph

Stores and manages the graph of relationships between markets.
Provides edge building from semantic similarity and graph persistence.

The graph represents learned relationships: when a leader market resolves,
related follower markets tend to reprice in predictable ways.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .models import EventEdge
from .utils.embeddings import cluster_markets

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
GRAPH_FILE = DATA_DIR / "event_graph.json"


class EventGraph:
  """
  Directed graph of market relationships.

  Each edge represents: when the 'from' market (leader) resolves,
  the 'to' market (follower) is expected to reprice.
  """

  def __init__(self, filepath=GRAPH_FILE):
    self.filepath = filepath
    self._edges = {}       # key -> EventEdge
    self._outgoing = {}    # leader_id -> [keys]
    self._incoming = {}    # follower_id -> [keys]
    self._load()

  def _key(self, from_id, to_id):
    """Generate unique key for an edge."""
    return f"{from_id}::{to_id}"

  def _load(self):
    """Load graph from storage."""
    if not self.filepath.exists():
      return

    try:
      with open(self.filepath) as f:
        data = json.load(f)

      for edge_data in data.get("edges", []):
        edge = EventEdge.from_dict(edge_data)
        self._index_edge(edge)

      logger.info(f"Loaded {len(self._edges)} edges from {self.filepath}")
    except Exception as e:
      logger.error(f"Failed to load graph: {e}")

  def _save(self):
    """Persist graph to storage."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = {
      "version": 1,
      "updated_at": datetime.now(timezone.utc).isoformat(),
      "edge_count": len(self._edges),
      "edges": [edge.to_dict() for edge in self._edges.values()],
    }

    with open(self.filepath, "w") as f:
      json.dump(data, f, indent=2)

  def _index_edge(self, edge):
    """Add edge to internal indices."""
    key = self._key(edge.from_market_id, edge.to_market_id)
    self._edges[key] = edge

    # Index by leader (outgoing)
    if edge.from_market_id not in self._outgoing:
      self._outgoing[edge.from_market_id] = []
    if key not in self._outgoing[edge.from_market_id]:
      self._outgoing[edge.from_market_id].append(key)

    # Index by follower (incoming)
    if edge.to_market_id not in self._incoming:
      self._incoming[edge.to_market_id] = []
    if key not in self._incoming[edge.to_market_id]:
      self._incoming[edge.to_market_id].append(key)

  def add_edge(self, edge, save=True):
    """Add or update an edge."""
    self._index_edge(edge)
    if save:
      self._save()

  def add_edges(self, edges):
    """Add multiple edges, saving once at the end."""
    for edge in edges:
      self._index_edge(edge)
    self._save()

  def get_edge(self, from_id, to_id):
    """Get edge between two specific markets."""
    return self._edges.get(self._key(from_id, to_id))

  def get_outgoing(self, leader_id):
    """Get all edges where given market is the leader."""
    keys = self._outgoing.get(leader_id, [])
    return [self._edges[k] for k in keys if k in self._edges]

  def get_incoming(self, follower_id):
    """Get all edges where given market is the follower."""
    keys = self._incoming.get(follower_id, [])
    return [self._edges[k] for k in keys if k in self._edges]

  def get_all(self):
    """Get all edges."""
    return list(self._edges.values())

  def get_valid(self, min_samples=1, min_delta=0.03, min_lag=60):
    """Get edges with enough historical data to be actionable."""
    return [e for e in self._edges.values() if e.is_valid(min_samples, min_delta, min_lag)]

  def clear(self):
    """Remove all edges."""
    self._edges.clear()
    self._outgoing.clear()
    self._incoming.clear()
    self._save()

  def stats(self):
    """Get graph statistics."""
    edges = list(self._edges.values())

    if not edges:
      return {
        "total_edges": 0,
        "valid_edges": 0,
        "unique_leaders": 0,
        "unique_followers": 0,
      }

    valid = [e for e in edges if e.is_valid()]

    return {
      "total_edges": len(edges),
      "valid_edges": len(valid),
      "unique_leaders": len(self._outgoing),
      "unique_followers": len(self._incoming),
      "avg_similarity": sum(e.similarity for e in edges) / len(edges),
    }


def build_edges(markets, resolutions=None, n_clusters=None):
  """
  Build edges from a set of markets based on semantic similarity.

  Args:
    markets: List of Market objects.
    resolutions: Dict of market_id -> MarketResolution (optional).
    n_clusters: Number of clusters for grouping (default: auto).

  Returns:
    List of EventEdge objects.
  """
  if len(markets) < 2:
    return []

  logger.info(f"Building edges from {len(markets)} markets...")

  # Find related market pairs using clustering
  related_pairs = cluster_markets(markets, n_clusters=n_clusters)
  logger.info(f"Found {len(related_pairs)} related market pairs")

  if not related_pairs:
    return []

  # Convert to edges
  resolutions = resolutions or {}
  edges = []

  for market_i, market_j, similarity in related_pairs:
    # Determine leader/follower based on resolution times (if available)
    res_i = resolutions.get(market_i.id)
    res_j = resolutions.get(market_j.id)

    if res_i and res_j:
      if res_i.resolved_at < res_j.resolved_at:
        leader_id, follower_id = market_i.id, market_j.id
      else:
        leader_id, follower_id = market_j.id, market_i.id
    elif res_i:
      leader_id, follower_id = market_i.id, market_j.id
    elif res_j:
      leader_id, follower_id = market_j.id, market_i.id
    else:
      # Neither resolved - use arbitrary order
      leader_id, follower_id = market_i.id, market_j.id

    edge = EventEdge(
      from_market_id=leader_id,
      to_market_id=follower_id,
      similarity=similarity,
      confidence=similarity,
      last_updated=datetime.now(timezone.utc),
    )
    edges.append(edge)

  logger.info(f"Built {len(edges)} edges")
  return edges


def build_historical_edges(closed_markets, n_clusters=None, min_similarity=0.80):
  """
  Build edges from closed/resolved markets with actual outcome data.

  For each pair of similar resolved markets, we record whether they
  resolved the same way (both YES or both NO) and use that to
  compute expected deltas.

  Args:
    closed_markets: List of resolved Market objects.
    n_clusters: Number of clusters for grouping.
    min_similarity: Minimum semantic similarity.

  Returns:
    Tuple of (edges, market_outcomes) where:
    - edges: List of EventEdge objects with outcome correlation data
    - market_outcomes: Dict of market_id -> Outcome for backtesting
  """
  from models import ConditionalDelta, Outcome

  if len(closed_markets) < 2:
    return [], {}

  logger.info(f"Building edges from {len(closed_markets)} resolved markets...")

  # Determine resolution outcome for each market
  # Final yes_price > 0.5 means resolved YES, else NO
  def get_outcome(market):
    return Outcome.YES if market.yes_price > 0.5 else Outcome.NO

  market_outcomes = {m.id: get_outcome(m) for m in closed_markets}

  # Find related pairs
  related_pairs = cluster_markets(
    closed_markets,
    n_clusters=n_clusters,
    min_similarity=min_similarity,
  )
  logger.info(f"Found {len(related_pairs)} related pairs")

  # Build edges with outcome correlation
  edges = []
  same_outcome = 0
  diff_outcome = 0

  for market_i, market_j, similarity in related_pairs:
    outcome_i = market_outcomes[market_i.id]
    outcome_j = market_outcomes[market_j.id]

    # Track correlation
    correlated = (outcome_i == outcome_j)
    if correlated:
      same_outcome += 1
    else:
      diff_outcome += 1

    # Create delta: positive if outcomes match, negative if opposite
    # This means: "if leader resolves YES and markets are correlated,
    # follower price should increase toward YES"
    delta_value = 0.3 if correlated else -0.3

    yes_delta = ConditionalDelta(
      condition=Outcome.YES,
      avg_delta=delta_value,
      median_delta=delta_value,
      std_delta=0.1,
      avg_lag_seconds=3600,
      median_lag_seconds=3600,
      sample_count=1,
    )

    no_delta = ConditionalDelta(
      condition=Outcome.NO,
      avg_delta=-delta_value,
      median_delta=-delta_value,
      std_delta=0.1,
      avg_lag_seconds=3600,
      median_lag_seconds=3600,
      sample_count=1,
    )

    edge = EventEdge(
      from_market_id=market_i.id,
      to_market_id=market_j.id,
      similarity=similarity,
      yes_delta=yes_delta,
      no_delta=no_delta,
      confidence=similarity,
      last_updated=datetime.now(timezone.utc),
    )
    edges.append(edge)

  logger.info(f"Built {len(edges)} edges (correlated: {same_outcome}, anti-correlated: {diff_outcome})")
  return edges, market_outcomes


def print_summary(graph):
  """Print human-readable graph summary."""
  stats = graph.stats()

  print("\n" + "=" * 50)
  print("EVENT GRAPH SUMMARY")
  print("=" * 50)
  print(f"Total edges: {stats['total_edges']}")
  print(f"Valid edges: {stats['valid_edges']}")
  print(f"Candidate edges: {stats['total_edges'] - stats['valid_edges']}")
  print(f"Unique leaders: {stats['unique_leaders']}")
  print(f"Unique followers: {stats['unique_followers']}")

  if stats["total_edges"] > 0:
    print(f"Avg similarity: {stats['avg_similarity']:.2%}")

  # Show top edges
  all_edges = graph.get_all()
  valid_edges = graph.get_valid()

  if valid_edges:
    print("\n" + "-" * 50)
    print("TOP VALID EDGES")
    print("-" * 50)
    for edge in sorted(valid_edges, key=lambda e: e.confidence, reverse=True)[:10]:
      print(f"\n{edge.from_market_id[:8]}... -> {edge.to_market_id[:8]}...")
      print(f"  Similarity: {edge.similarity:.2%}")
      if edge.yes_delta:
        print(f"  YES: delta={edge.yes_delta.avg_delta:+.2%}, n={edge.yes_delta.sample_count}")
      if edge.no_delta:
        print(f"  NO: delta={edge.no_delta.avg_delta:+.2%}, n={edge.no_delta.sample_count}")

  elif all_edges:
    print("\n" + "-" * 50)
    print("TOP CANDIDATE EDGES BY SIMILARITY")
    print("-" * 50)
    print("(No historical data yet)")
    for edge in sorted(all_edges, key=lambda e: e.similarity, reverse=True)[:10]:
      print(f"\n{edge.from_market_id[:8]}... -> {edge.to_market_id[:8]}...")
      print(f"  Similarity: {edge.similarity:.2%}")
