"""Tests for ingestion module."""

from datetime import datetime, timezone

import pytest

from src.topic.client import Market
from src.topic.models import Outcome


def test_resolution_tracker_initialization():
  """Test ResolutionTracker creates empty state."""
  from src.topic.ingestion import ResolutionTracker

  tracker = ResolutionTracker()
  assert len(tracker.get_all()) == 0


def test_market_to_resolution():
  """Test converting Market to MarketResolution."""
  from src.topic.ingestion import ResolutionTracker

  tracker = ResolutionTracker()

  market = Market(
    id="test123",
    question="Will it rain?",
    description="Weather market",
    slug="rain",
    outcomes=["Yes", "No"],
    prices=[0.7, 0.3],
    volume=1000.0,
    liquidity=500.0,
    end_date="2024-12-31T23:59:59Z",
  )

  resolution = tracker._market_to_resolution(market)

  assert resolution is not None
  assert resolution.market_id == "test123"
  assert resolution.question == "Will it rain?"
  assert resolution.outcome == Outcome.YES  # yes_price > 0.5


def test_market_to_resolution_no_outcome():
  """Test resolution with NO outcome."""
  from src.topic.ingestion import ResolutionTracker

  tracker = ResolutionTracker()

  market = Market(
    id="test456",
    question="Will it snow?",
    description="Weather market",
    slug="snow",
    outcomes=["Yes", "No"],
    prices=[0.3, 0.7],
    volume=1000.0,
    liquidity=500.0,
    end_date="2024-12-31T23:59:59Z",
  )

  resolution = tracker._market_to_resolution(market)

  assert resolution.outcome == Outcome.NO  # yes_price < 0.5


def test_resolution_persistence(tmp_path, monkeypatch):
  """Test saving and loading resolutions."""
  from src.topic.ingestion import ResolutionTracker

  # Override data directory
  test_file = tmp_path / "resolutions.json"
  monkeypatch.setattr("src.topic.ingestion.RESOLUTIONS_FILE", test_file)

  tracker = ResolutionTracker()

  market = Market(
    id="test789",
    question="Will it rain?",
    description="Test",
    slug="test",
    outcomes=["Yes", "No"],
    prices=[0.6, 0.4],
    volume=1000.0,
    liquidity=500.0,
    end_date="2024-12-31T23:59:59Z",
  )

  resolution = tracker._market_to_resolution(market)
  tracker._resolutions["test789"] = resolution
  tracker._save()

  # Load in new tracker
  new_tracker = ResolutionTracker()
  assert len(new_tracker.get_all()) == 1
  assert new_tracker.get("test789").market_id == "test789"


def test_snapshot_capture():
  """Test SnapshotCapture initialization."""
  from src.topic.ingestion import SnapshotCapture

  capture = SnapshotCapture()
  assert capture.client is not None


def test_ingestion_runner_initialization():
  """Test IngestionRunner creates components."""
  from src.topic.ingestion import IngestionRunner

  runner = IngestionRunner(
    snapshot_interval=3600,
    resolution_interval=1800,
  )

  assert runner.snapshot_interval == 3600
  assert runner.resolution_interval == 1800
  assert runner.snapshot_capture is not None
  assert runner.resolution_tracker is not None
