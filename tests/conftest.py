"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest

from src.topic.deterministic_bootstrap import init_deterministic


@pytest.fixture(scope="session", autouse=True)
def deterministic_session():
  """Ensure deterministic behavior for all tests."""
  init_deterministic(seed=42)


@pytest.fixture
def sample_embeddings():
  """Sample embeddings for testing."""
  return np.random.randn(10, 768).astype(np.float32)


@pytest.fixture
def sample_market_data():
  """Sample market data for testing."""
  return {
    "id": "123",
    "question": "Will it rain tomorrow?",
    "description": "Weather forecast market",
    "slug": "rain-tomorrow",
    "outcomes": ["Yes", "No"],
    "outcomePrices": "[0.6, 0.4]",
    "volume": "10000",
    "liquidity": "5000",
    "endDate": "2024-12-31T23:59:59Z",
  }
