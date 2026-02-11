"""
Embedding Utilities

Provides semantic embeddings for market text and clustering to find related markets.
Uses a cached model to avoid reloading on every call.
"""

import logging
import os
import warnings
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

logger = logging.getLogger(__name__)

# Module-level cached model
_model = None


def get_model():
    """Get cached embedding model, loading if needed."""
    global _model  # noqa: PLW0603
    if _model is None:
        logger.info("Loading embedding model (one-time)...")
        _model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return _model


def embed_texts(texts):
    """Generate normalized embeddings for texts."""
    model = get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embeddings, dtype=np.float32)


def _parse_end_date(end_date_str):
    """Parse end_date string to datetime, returning None on failure."""
    if not end_date_str:
        return None
    try:
        # Handle various ISO formats
        clean = end_date_str.replace("Z", "+00:00")
        return datetime.fromisoformat(clean)
    except (ValueError, TypeError):
        return None


def _dates_within_days(date1, date2, days=7):
    """Check if two dates are within N days of each other."""
    if date1 is None or date2 is None:
        return False
    diff = abs((date1 - date2).days)
    return diff <= days


def cluster_markets(markets, min_similarity=0.70, max_days_apart=7):
    """
    Find related market pairs by semantic similarity and time proximity.

    Args:
        markets: List of Market objects with question, description, and end_date.
        min_similarity: Minimum cosine similarity threshold.
        max_days_apart: Maximum days between end_dates to consider related.

    Returns:
        List of (market_i, market_j, similarity) tuples.
    """
    if len(markets) < 2:
        return []

    # Generate embeddings once for all markets
    texts = [f"{m.question} {m.description[:200]}" for m in markets]
    embeddings = embed_texts(texts)

    # Parse end dates for time filtering
    end_dates = [_parse_end_date(m.end_date) for m in markets]

    # Cluster markets
    n_clusters = max(len(markets) // 10, 5)
    n_clusters = min(n_clusters, len(markets))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Group by cluster
    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(i)

    # Find related pairs within clusters
    pairs = []
    for indices in clusters.values():
        if len(indices) < 2:
            continue

        cluster_emb = embeddings[indices]
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_i, idx_j = indices[i], indices[j]

                # Check time proximity
                if not _dates_within_days(end_dates[idx_i], end_dates[idx_j], max_days_apart):
                    continue

                sim = float(np.dot(cluster_emb[i], cluster_emb[j]))
                if sim >= min_similarity:
                    pairs.append((markets[idx_i], markets[idx_j], sim))

    pairs.sort(key=lambda x: (-x[2], x[0].id, x[1].id))
    return pairs
