"""Tests for Polymarket client."""

import responses

from src.topic.utils.client import Market, _parse_market


def test_parse_market():
    """Test market data parsing."""
    data = {
        "id": "123",
        "question": "Will it rain tomorrow?",
        "description": "Test market",
        "slug": "test",
        "outcomes": ["Yes", "No"],
        "outcomePrices": "[0.6, 0.4]",
        "volume": 1000,
        "liquidity": 500,
        "endDate": "2024-12-31",
    }
    market = _parse_market(data)

    assert market is not None
    assert market.id == "123"
    assert market.question == "Will it rain tomorrow?"
    assert market.yes_price == 0.6
    assert market.no_price == 0.4


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


@responses.activate
def test_fetch_markets():
    """Test fetching markets from API."""
    from src.topic.utils.client import fetch_markets

    responses.add(
        responses.GET,
        "https://gamma-api.polymarket.com/markets",
        json=[
            {
                "id": "1",
                "question": "Test?",
                "description": "Test",
                "slug": "test",
                "outcomes": ["Yes", "No"],
                "outcomePrices": "[0.5, 0.5]",
                "volume": 10000,
                "liquidity": 10000,
                "endDate": "",
            }
        ],
        status=200,
    )

    markets = fetch_markets(min_liquidity=1000, min_volume=1000)
    assert len(markets) == 1
    assert markets[0].id == "1"
