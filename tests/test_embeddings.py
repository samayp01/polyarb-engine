"""Tests for embedding module."""

import numpy as np
import pytest

from src.topic.utils.embeddings import EmbeddingModel, sanitize_embeddings


def test_sanitize_embeddings_removes_nan():
  """Test that sanitize_embeddings removes NaN values."""
  arr = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
  result = sanitize_embeddings(arr)

  assert np.all(np.isfinite(result))
  assert result.dtype == np.float32


def test_sanitize_embeddings_removes_inf():
  """Test that sanitize_embeddings removes Inf values."""
  arr = np.array([[1.0, np.inf, 3.0], [4.0, -np.inf, 6.0]])
  result = sanitize_embeddings(arr)

  assert np.all(np.isfinite(result))
  assert result.dtype == np.float32


def test_sanitize_embeddings_normalizes():
  """Test that sanitize_embeddings produces unit vectors."""
  arr = np.array([[3.0, 4.0], [1.0, 0.0]])
  result = sanitize_embeddings(arr)

  norms = np.linalg.norm(result, axis=1)
  np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_embedding_model_returns_finite():
  """Test that EmbeddingModel returns finite embeddings."""
  model = EmbeddingModel()
  text = "This is a test sentence"

  embedding = model.embed(text)

  assert np.all(np.isfinite(embedding))
  assert embedding.dtype == np.float32


def test_embedding_model_batch_deterministic():
  """Test that repeated calls produce same results."""
  model = EmbeddingModel(seed=42)
  texts = ["First text", "Second text"]

  result1 = model.embed_batch(texts)
  result2 = model.embed_batch(texts)

  np.testing.assert_array_equal(result1, result2)


def test_embedding_metadata():
  """Test that embedding metadata is saved correctly."""
  model = EmbeddingModel(seed=42)
  metadata = model.get_metadata()

  assert "model_name" in metadata
  assert "seed" in metadata
  assert "timestamp" in metadata
  assert metadata["seed"] == 42
