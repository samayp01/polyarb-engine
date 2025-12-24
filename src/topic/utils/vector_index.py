"""Simple FAISS vector index for fast similarity search."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VectorIndex:
  """Simple FAISS-based vector index with in-memory fallback."""

  def __init__(self, dimension: int):
    self.dimension = dimension
    self.index = None
    self.ids = []
    self.use_faiss = False

    try:
      import faiss

      self.index = faiss.IndexFlatIP(dimension)
      self.use_faiss = True
      logger.info(f"Using FAISS index (dimension={dimension})")
    except ImportError:
      logger.warning("FAISS not available, using in-memory search")
      self.vectors = None

  def add(self, vectors: np.ndarray, ids: List[str]):
    """Add vectors to index."""
    assert vectors.shape[1] == self.dimension, f"Expected dim {self.dimension}"
    assert len(vectors) == len(ids), "Vectors and IDs must have same length"

    if self.use_faiss:
      self.index.add(vectors.astype(np.float32))
    else:
      if self.vectors is None:
        self.vectors = vectors.astype(np.float32)
      else:
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])

    self.ids.extend(ids)

  def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
    """
    Search for k nearest neighbors.

    Returns:
      List of (id, similarity) tuples sorted by similarity
    """
    query = query.astype(np.float32).reshape(1, -1)

    if self.use_faiss:
      scores, indices = self.index.search(query, min(k, len(self.ids)))
      results = [
        (self.ids[idx], float(score))
        for idx, score in zip(indices[0], scores[0])
        if idx < len(self.ids)
      ]
    else:
      # In-memory search using dot product
      if self.vectors is None or len(self.vectors) == 0:
        return []

      scores = np.dot(self.vectors, query.T).flatten()
      top_k_indices = np.argsort(-scores)[:k]

      results = [
        (self.ids[idx], float(scores[idx]))
        for idx in top_k_indices
        if idx < len(self.ids)
      ]

    # Sort by similarity desc, then by ID for determinism
    results.sort(key=lambda x: (-x[1], x[0]))
    return results

  def save(self, path: Path):
    """Save index to disk."""
    if self.use_faiss:
      import faiss

      faiss.write_index(self.index, str(path))
      # Save IDs separately
      ids_path = path.with_suffix(".ids.npy")
      np.save(ids_path, self.ids)
    else:
      if self.vectors is not None:
        np.savez(path, vectors=self.vectors, ids=self.ids)

  def load(self, path: Path):
    """Load index from disk."""
    if self.use_faiss:
      import faiss

      self.index = faiss.read_index(str(path))
      ids_path = path.with_suffix(".ids.npy")
      self.ids = list(np.load(ids_path, allow_pickle=True))
    else:
      data = np.load(path, allow_pickle=True)
      self.vectors = data["vectors"]
      self.ids = list(data["ids"])
