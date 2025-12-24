"""Tests for Polymarket client."""

import pytest
import requests
import responses

from src.topic.client import Market, PolymarketClient


@responses.activate
def test_parse_market(sample_market_data):
  """Test market data parsing."""
  client = PolymarketClient()
  market = client._parse_market(sample_market_data)

  assert market is not None
  assert market.id == "123"
  assert market.question == "Will it rain tomorrow?"
  assert market.yes_price == 0.6
  assert market.no_price == 0.4


@responses.activate
def test_fetch_batch():
  """Test fetching a batch of markets."""
  responses.add(
    responses.GET,
    "https://gamma-api.polymarket.com/markets",
    json=[{"id": "1", "question": "Test?", "outcomePrices": "[0.5, 0.5]"}],
    status=200,
  )

  client = PolymarketClient()
  batch = client._fetch_batch(limit=10, offset=0, closed=False)

  assert len(batch) == 1
  assert batch[0]["id"] == "1"


@responses.activate
def test_fetch_batch_handles_errors():
  """Test that fetch_batch handles HTTP errors gracefully."""
  responses.add(
    responses.GET,
    "https://gamma-api.polymarket.com/markets",
    status=500,
  )

  client = PolymarketClient()
  batch = client._fetch_batch(limit=10, offset=0, closed=False)

  assert batch == []


def test_market_properties():
  """Test Market property calculations."""
  market = Market(
    id="1",
    question="Test?",
    description="Test market",
    slug="test",
    outcomes=["Yes", "No"],
    prices=[0.7, 0.3],
    volume=1000.0,
    liquidity=500.0,
    end_date="2024-12-31",
  )

  assert market.yes_price == 0.7
  assert market.no_price == 0.3
  assert market.to_dict()["id"] == "1"
