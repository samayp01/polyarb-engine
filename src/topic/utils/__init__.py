"""Utilities for embeddings and API client."""

from .client import Market, fetch_closed_markets, fetch_markets
from .embeddings import cluster_markets, embed_texts
from .models import EventEdge, MarketResolution, MarketSnapshot, Outcome, Signal

__all__ = [
    "EventEdge",
    "Market",
    "MarketResolution",
    "MarketSnapshot",
    "Outcome",
    "Signal",
    "cluster_markets",
    "embed_texts",
    "fetch_closed_markets",
    "fetch_markets",
]
