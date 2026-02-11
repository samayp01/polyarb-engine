"""
Configuration for Polymarket Trading System

Centralized config for parameters shared across build, backtest, and monitoring.
Edit these values to tune the system behavior.
"""

import os

# === Embedding Graph Settings ===
# Minimum cosine similarity for two markets to be considered related
MIN_EMBEDDING_SIMILARITY = float(os.getenv("MIN_EMBEDDING_SIMILARITY", "0.3"))

# Maximum days apart for market pairs (filters out unrelated timeframes)
MAX_DAYS_APART = int(os.getenv("MAX_DAYS_APART", "90"))


# === Data Fetching ===
# Number of markets to fetch per API page
MARKETS_PER_PAGE = int(os.getenv("MARKETS_PER_PAGE", "100"))

# Maximum pages to fetch for closed markets (100 per page = 5000 markets at 50 pages)
MAX_PAGES = int(os.getenv("MAX_PAGES", "50"))

# Total max markets = MARKETS_PER_PAGE * MAX_PAGES


# === Backtest Settings ===
# Maximum markets to analyze in backtest (higher = more pairs but slower)
BACKTEST_MAX_MARKETS = int(os.getenv("BACKTEST_MAX_MARKETS", "2000"))

# Show verbose output during backtest
BACKTEST_VERBOSE = os.getenv("BACKTEST_VERBOSE", "true").lower() == "true"

# Save backtest results to file for inspection
BACKTEST_SAVE_RESULTS = os.getenv("BACKTEST_SAVE_RESULTS", "true").lower() == "true"
