# Topic - Polymarket Signal System

A simple pipeline for ingesting Polymarket data, building similarity graphs, and monitoring for trading signals.

## What It Does

1. **Ingest** - Fetches market data from Polymarket API
2. **Build** - Creates a similarity graph using semantic embeddings
3. **Monitor** - Watches for market resolutions and emits signals when related markets may be affected

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <your-repo>
cd topic
uv sync
```

## Quick Start

```bash
# Build the similarity graph from resolved markets
uv run python -m topic.run build

# Run backtest to evaluate the strategy
uv run python -m topic.run backtest

# Start monitoring for signals
uv run python -m topic.run monitor
```

## Commands

| Command | Description |
|---------|-------------|
| `build` | Build similarity graph from resolved markets |
| `backtest` | Test strategy on historical data |
| `ingest` | Poll for new market resolutions |
| `monitor` | Watch for resolutions and emit signals |
| `status` | Show system status |

## Configuration

Environment variables (with defaults):

```bash
# Embedding settings
MIN_EMBEDDING_SIMILARITY=0.3  # Min cosine similarity for related markets
MAX_DAYS_APART=90             # Max days between market end dates

# Data fetching
MARKETS_PER_PAGE=100
MAX_PAGES=50

# Backtest
BACKTEST_MAX_MARKETS=2000
BACKTEST_VERBOSE=true
BACKTEST_SAVE_RESULTS=true
```

## Project Structure

```
src/topic/
├── run.py           # CLI entry point
├── config.py        # Configuration
├── graph.py         # Similarity graph
├── signals.py       # Signal generation
├── ingestion.py     # Market data ingestion
├── backtest.py      # Historical backtesting
└── utils/
    ├── client.py    # Polymarket API client
    ├── models.py    # Data models
    └── embeddings.py # Semantic similarity
```

## Data Files

```
data/
├── event_graph.json      # Similarity graph
├── resolutions.json      # Tracked resolutions
├── signals.json          # Generated signals
└── backtest_results.json # Backtest output
```

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/
```

## License

MIT
