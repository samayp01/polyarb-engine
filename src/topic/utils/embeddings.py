"""
Embedding and Clustering Utilities

Provides semantic embeddings for market text using sentence transformers,
and clustering functionality to group related markets.
"""

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")


def sanitize_embeddings(arr: np.ndarray) -> np.ndarray:
  """
  Sanitize embeddings by removing NaN/Inf and normalizing.

  Args:
    arr: Input embedding array

  Returns:
    Sanitized float32 array with unit norm
  """
  # Convert to float32 and replace NaN/Inf with 0
  arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

  # Normalize to unit vectors
  norms = np.linalg.norm(arr, axis=-1, keepdims=True)
  arr = arr / np.maximum(norms, 1e-10)

  return arr


class EmbeddingModel:
  """
  Generates semantic embeddings for text using sentence transformers.

  Uses the all-mpnet-base-v2 model which provides good quality embeddings
  for semantic similarity tasks. All embeddings are sanitized and normalized.
  """

  def __init__(
    self,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    seed: int = 42,
  ):
    self.model_name = model_name
    self.seed = seed
    self.model = SentenceTransformer(model_name)

    # Get transformer version if available
    try:
      import transformers
      self.transformers_version = transformers.__version__
    except (ImportError, AttributeError):
      self.transformers_version = "unknown"

  def embed(self, text: str) -> np.ndarray:
    """Generate embedding for a single text string."""
    embedding = self.model.encode([text], normalize_embeddings=False)
    return sanitize_embeddings(embedding)[0]

  def embed_batch(self, texts: List[str]) -> np.ndarray:
    """Generate embeddings for a batch of texts."""
    embeddings = self.model.encode(texts, normalize_embeddings=False)
    return sanitize_embeddings(embeddings)

  def get_metadata(self) -> Dict:
    """Get model metadata for reproducibility."""
    return {
      "model_name": self.model_name,
      "transformers_version": self.transformers_version,
      "seed": self.seed,
      "timestamp": datetime.now(timezone.utc).isoformat(),
    }

  def save_embeddings(
    self, embeddings: np.ndarray, ids: List[str], output_path: Path
  ):
    """Save embeddings with metadata."""
    data = {
      "metadata": self.get_metadata(),
      "embeddings": embeddings.tolist(),
      "ids": ids,
    }
    with open(output_path, "w") as f:
      json.dump(data, f)

  @staticmethod
  def load_embeddings(input_path: Path) -> Tuple[np.ndarray, List[str], Dict]:
    """Load embeddings with metadata."""
    with open(input_path) as f:
      data = json.load(f)

    embeddings = np.array(data["embeddings"], dtype=np.float32)
    ids = data["ids"]
    metadata = data.get("metadata", {})

    return embeddings, ids, metadata


def get_embeddings(texts: List[str], model: Optional[EmbeddingModel] = None) -> np.ndarray:
  """Get embeddings for texts, creating model if needed."""
  if model is None:
    model = EmbeddingModel()
  return model.embed_batch(texts)


def cluster_markets(markets, embedder=None, n_clusters=None, min_similarity=0.70):
  """
  Cluster markets by semantic similarity and find related pairs.

  Args:
    markets: List of Market objects with question and description fields.
    embedder: EmbeddingModel instance (created if not provided).
    n_clusters: Number of clusters (default: len(markets) // 10).
    min_similarity: Minimum cosine similarity to consider markets related.

  Returns:
    List of (market_i, market_j, similarity) tuples for related market pairs.
  """
  if len(markets) < 2:
    return []

  if embedder is None:
    embedder = EmbeddingModel()

  # Generate embeddings for all markets
  texts = [f"{m.question} {m.description[:200]}" for m in markets]
  embeddings = embedder.embed_batch(texts)

  # All embeddings are already sanitized and normalized
  assert np.all(np.isfinite(embeddings)), "Embeddings contain NaN/Inf"

  # Determine cluster count
  if n_clusters is None:
    n_clusters = max(len(markets) // 10, 5)
  n_clusters = min(n_clusters, len(markets))

  # Cluster markets using k-means
  kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
  labels = kmeans.fit_predict(embeddings)

  # Group markets by cluster
  clusters = {}
  for i, label in enumerate(labels):
    if label not in clusters:
      clusters[label] = []
    clusters[label].append(i)

  # Find related pairs within each cluster
  related_pairs = []

  for cluster_indices in clusters.values():
    if len(cluster_indices) < 2:
      continue

    # Compute pairwise similarities within cluster using dot product
    cluster_embeddings = embeddings[cluster_indices]

    # Use stable sorting for deterministic results
    for i in range(len(cluster_indices)):
      for j in range(i + 1, len(cluster_indices)):
        # Direct dot product on normalized vectors
        similarity = float(np.dot(cluster_embeddings[i], cluster_embeddings[j]))

        if similarity < min_similarity:
          continue

        market_i = markets[cluster_indices[i]]
        market_j = markets[cluster_indices[j]]

        # Skip if same market ID
        if market_i.id == market_j.id:
          continue

        related_pairs.append((market_i, market_j, similarity))

  # Sort by similarity (descending) then by IDs for determinism
  related_pairs.sort(key=lambda x: (-x[2], x[0].id, x[1].id))

  return related_pairs
