# Polymarket Trading System

Detects mispricing opportunities in prediction markets by analyzing
relationships between semantically similar markets.

## How It Works

1. **Build Graph**: Fetches resolved markets from Polymarket, clusters them
   by semantic similarity, and builds a graph of related market pairs.

2. **Backtest**: Tests whether similar markets tend to resolve the same way.

3. **Monitor**: When a "leader" market resolves, generates trading signals
   for related "follower" markets that haven't repriced yet.

## Setup

```bash
uv sync
```

## Usage

```bash
# Build event graph from historical data
uv run python run.py build

# Run backtest
uv run python run.py backtest

# Monitor for signals
uv run python run.py monitor

# Check status
uv run python run.py status
```

## Development

```bash
# Install development dependencies
uv sync --all-extras

# Run tests
uv run pytest tests/ -v

# Lint code
uv run ruff check src/ tests/
uv run black src/ tests/

# Type check
uv run mypy src/
```

## Project Structure

```
├── src/topic/             # Main package
│   ├── client.py          # Polymarket API client
│   ├── models.py          # Data models
│   ├── graph.py           # Event graph and edge building
│   ├── signals.py         # Signal generation
│   ├── backtest.py        # Backtesting engine
│   ├── ingestion.py       # Data collection
│   ├── deterministic_bootstrap.py  # Reproducibility setup
│   └── utils/
│       ├── embeddings.py  # Semantic embeddings with sanitization
│       ├── http.py        # HTTP client with retry logic
│       └── vector_index.py # FAISS vector index
├── tests/                 # Test suite
└── run.py                 # CLI entry point
```

## Commands

| Command   | Description |
|-----------|-------------|
| `build`   | Build event graph from resolved markets |
| `backtest`| Test prediction accuracy on historical data |
| `monitor` | Watch for trading signals in real-time |
| `status`  | Show system status |

## Key Concepts

- **Edge**: Relationship between two similar markets
- **Leader**: Market that resolves first
- **Follower**: Related market expected to move after leader resolves
- **Signal**: Trading opportunity when follower hasn't repriced yet

## Reproducibility

For deterministic results across runs:

- Set `PYTHONHASHSEED=0` environment variable
- Backtest module auto-initializes with `seed=42`
- BLAS threading limited to 1 thread (OMP/MKL/OPENBLAS)
- All embeddings are sanitized (NaN/Inf → 0, normalized to unit vectors)

Run backtest twice to verify identical results:

```bash
uv run python run.py backtest > run1.txt
uv run python run.py backtest > run2.txt
diff run1.txt run2.txt  # Should be empty
```

## Architecture Notes

### Embeddings
- All embeddings saved with metadata (model name, version, seed, timestamp)
- Sanitization ensures no NaN/Inf values and unit norm
- Uses dot product on normalized vectors instead of sklearn cosine_similarity to avoid matmul warnings

### HTTP Retry Logic
- Polymarket API calls use requests.Session with exponential backoff
- Retries on 429, 500, 502, 503, 504 status codes
- All API calls use proper params dict (not URL string interpolation)

### Vector Index
- Simple FAISS adapter for fast similarity search
- Falls back to in-memory numpy if FAISS unavailable
- Stable sorting: (-similarity, id) for deterministic top-k results

## Docker

```bash
# Build image
docker build -t topic .

# Run backtest
docker run --rm -v $(pwd)/data:/app/data topic uv run python run.py backtest
```

## Testing

Tests enforce:
- Deterministic behavior (same seed → same results)
- Finite embeddings (no NaN/Inf)
- Stable sorting in search results
- HTTP error handling with mocked responses

Run tests:
```bash
uv run pytest tests/ -v --cov=src
```
