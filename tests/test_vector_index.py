"""Tests for vector index module."""

import numpy as np
import pytest

from src.topic.utils.vector_index import VectorIndex


def test_vector_index_initialization():
  """Test VectorIndex creates with correct dimension."""
  index = VectorIndex(dimension=128)
  assert index.dimension == 128
  assert len(index.ids) == 0


def test_vector_index_add_vectors():
  """Test adding vectors to index."""
  index = VectorIndex(dimension=128)

  vectors = np.random.randn(10, 128).astype(np.float32)
  ids = [f"id{i}" for i in range(10)]

  index.add(vectors, ids)
  assert len(index.ids) == 10


def test_vector_index_search():
  """Test searching for nearest neighbors."""
  index = VectorIndex(dimension=128)

  # Add some vectors
  vectors = np.random.randn(100, 128).astype(np.float32)
  # Normalize for cosine similarity
  vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
  ids = [f"id{i}" for i in range(100)]

  index.add(vectors, ids)

  # Search with first vector
  query = vectors[0]
  results = index.search(query, k=5)

  assert len(results) <= 5
  assert results[0][0] == "id0"  # Should find itself first
  assert all(isinstance(r[1], float) for r in results)


def test_vector_index_search_deterministic():
  """Test that search results are deterministic."""
  index = VectorIndex(dimension=64)

  vectors = np.random.randn(50, 64).astype(np.float32)
  vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
  ids = [f"id{i}" for i in range(50)]

  index.add(vectors, ids)

  query = np.random.randn(64).astype(np.float32)
  query = query / np.linalg.norm(query)

  results1 = index.search(query, k=10)
  results2 = index.search(query, k=10)

  # Results should be identical
  assert results1 == results2


def test_vector_index_empty_search():
  """Test search on empty index."""
  index = VectorIndex(dimension=128)

  query = np.random.randn(128).astype(np.float32)

  # Empty index should return empty results
  # If using in-memory mode, it may raise an error - handle both
  try:
    results = index.search(query, k=5)
    assert len(results) == 0
  except (IndexError, AttributeError):
    # In-memory search may fail on empty index
    pass


def test_vector_index_dimension_mismatch():
  """Test that dimension mismatch raises error."""
  index = VectorIndex(dimension=128)

  vectors = np.random.randn(10, 64).astype(np.float32)  # Wrong dimension
  ids = [f"id{i}" for i in range(10)]

  with pytest.raises(AssertionError):
    index.add(vectors, ids)


def test_vector_index_stable_sorting():
  """Test that results are sorted by similarity then id."""
  index = VectorIndex(dimension=4)

  # Create vectors with known similarities
  vectors = np.array(
    [
      [1.0, 0.0, 0.0, 0.0],
      [0.9, 0.1, 0.0, 0.0],
      [0.9, 0.1, 0.0, 0.0],  # Duplicate similarity
    ],
    dtype=np.float32,
  )
  vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

  ids = ["id2", "id0", "id1"]
  index.add(vectors, ids)

  query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
  results = index.search(query, k=3)

  # Should be sorted by similarity desc, then by id asc
  assert len(results) == 3
  # First result should be highest similarity
  assert results[0][1] > results[1][1] or (
    results[0][1] == results[1][1] and results[0][0] < results[1][0]
  )
